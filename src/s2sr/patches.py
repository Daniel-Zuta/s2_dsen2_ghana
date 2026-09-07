"""Patch extraction: spatial sampling of aligned 10m/20m band patches with per-patch AOI + SCL
validity filtering, stratified by ESA WorldCover land-cover class.

Deliberately NOT doing Wald's-protocol downsampling here — patches are saved at native
resolution (`hr_10m`, `hr_20m`), and the (input, target) training pairs get constructed later
(training notebook / dataset code), so the downsampling method can be changed without
re-extracting patches. See agents.md "Patch sampling strategy".
"""

import contextlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
import rasterio.windows

from . import config, landcover
from .quality import SCL_NODATA_CLASSES, SCL_UNUSABLE_CLASSES
from .stac import refresh_item


def patch_windows(width, height, transform, aoi_geom, patch_size):
    """Yield non-overlapping patch_size x patch_size Windows covering aoi_geom's bounding box,
    snapped to the patch grid and clipped to the raster's extent.
    """
    win = rasterio.windows.from_bounds(*aoi_geom.bounds, transform=transform)
    row0 = max(0, (int(win.row_off) // patch_size) * patch_size)
    col0 = max(0, (int(win.col_off) // patch_size) * patch_size)
    row1 = min(height, -(-int(win.row_off + win.height) // patch_size) * patch_size)
    col1 = min(width, -(-int(win.col_off + win.width) // patch_size) * patch_size)
    for row in range(row0, row1, patch_size):
        for col in range(col0, col1, patch_size):
            if row + patch_size <= height and col + patch_size <= width:
                yield rasterio.windows.Window(col, row, patch_size, patch_size)


def _load_patch_record(item, out_path: Path) -> dict:
    """Reconstruct a manifest record from an already-saved patch — no network involved."""
    cached = np.load(out_path)
    return {
        "patch_id": out_path.stem,
        "scene_id": item.id,
        "mgrs_tile": item.properties.get("s2:mgrs_tile"),
        "datetime": item.properties.get("datetime"),
        "row_off": int(cached["row_off"]),
        "col_off": int(cached["col_off"]),
        "aoi_coverage": float(cached["aoi_coverage"]),
        "nodata_fraction": float(cached["nodata_fraction"]),
        "unusable_fraction": float(cached["unusable_fraction"]),
        "path": str(out_path),
    }


def _fetch_landcover_for_aoi(aoi_geom, ref, scl_src):
    """Fetches a WorldCover grid covering just `aoi_geom`'s bounding box, at the same
    20m-equivalent resolution/grid as the SCL validity checks below — so a candidate
    window's already-computed `scl_window` can be used directly to slice it.

    Returns `(grid, row0_20m, col0_20m)`, or `(None, None, None)` if the fetch fails (the
    caller falls back to unstratified sampling rather than failing the whole scene over a
    land-cover lookup — see agents.md, this fetch is unverified against the live API and may
    need fixing).
    """
    try:
        aoi_win = rasterio.windows.from_bounds(*aoi_geom.bounds, transform=ref.transform)
        row0 = max(0, int(aoi_win.row_off))
        col0 = max(0, int(aoi_win.col_off))
        row1 = min(ref.height, int(aoi_win.row_off + aoi_win.height))
        col1 = min(ref.width, int(aoi_win.col_off + aoi_win.width))
        row0_20m, col0_20m = row0 // 2, col0 // 2
        shape_20m = (max(1, (row1 - row0) // 2), max(1, (col1 - col0) // 2))
        lc_transform = rasterio.windows.transform(
            rasterio.windows.Window(col0_20m, row0_20m, shape_20m[1], shape_20m[0]),
            scl_src.transform,
        )
        bounds_4326 = rasterio.warp.transform_bounds(ref.crs, "EPSG:4326", *aoi_geom.bounds)
        grid = landcover.fetch_landcover_grid(bounds_4326, scl_src.crs, lc_transform, shape_20m)
        return grid, row0_20m, col0_20m
    except Exception:
        return None, None, None


def extract_patches_for_scene(
    item,
    aoi: gpd.GeoDataFrame,
    out_dir,
    patch_size_10m: int = config.PATCH_SIZE_10M,
    min_aoi_coverage: float = config.MIN_PATCH_AOI_COVERAGE,
    max_unusable_fraction: float = config.MAX_PATCH_UNUSABLE_FRACTION,
    max_nodata_fraction: float = config.MAX_PATCH_NODATA_FRACTION,
    max_patches: int = config.MAX_PATCHES_PER_SCENE,
    seed: int = 0,
    use_landcover_stratification: bool = True,
):
    """Extract up to `max_patches` valid patches from one scene, stratified across land-cover
    classes (round-robin) rather than purely at random, so the kept subset isn't dominated by
    whichever land cover happens to be spatially/numerically dominant in this scene.

    Each patch is saved as a compressed `.npz` with `hr_10m` (4 x P x P, native 10m bands),
    `hr_20m` (6 x P/2 x P/2, native 20m bands) — no downsampling applied — plus its validity
    stats.

    Three passes: (1) validity-check up to `MAX_PATCHES_PER_SCENE *
    LANDCOVER_CANDIDATE_MULTIPLIER` shuffled candidate windows and bucket the survivors by
    majority land-cover class (one WorldCover fetch per scene, via `s2sr.landcover` — falls
    back to unstratified sampling if that fetch fails); (2) round-robin across class buckets to
    pick the actual kept subset; (3) only now do the expensive full-resolution band reads, for
    the winners. This means a from-scratch scene does more (cheap, small) SCL-window reads than
    a purely-random selection would (which could stop as soon as it found enough winners) — the
    `LANDCOVER_CANDIDATE_MULTIPLIER` cap bounds how much more.

    Resumable and bandwidth-conscious (a real constraint — home connection is a mobile hotspot):
    a patch already saved from a previous run is reused from disk, never re-read over the
    network. If this scene already has `max_patches` saved patches, this returns immediately from
    a local directory listing — no network call at all, not even to open the bands. Note: a scene
    with fewer than `max_patches` *possible* valid windows will still do a full re-scan on every
    rerun (there's no persisted "already exhaustively searched" marker) — harmless, just not fully
    free for that scene.

    Returns a list of manifest dicts, one per patch (reused or newly extracted).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob(f"{item.id}__*.npz"))
    if len(existing) >= max_patches:
        return [_load_patch_record(item, p) for p in existing[:max_patches]]

    rng = np.random.default_rng(seed)
    item = refresh_item(item)
    band_hrefs = {b: item.assets[b].href for b in [*config.BANDS_10M, *config.BANDS_20M, "SCL"]}

    records = [_load_patch_record(item, p) for p in existing]
    existing_ids = {p.stem for p in existing}
    n_needed = max_patches - len(records)

    with contextlib.ExitStack() as stack:
        srcs = {b: stack.enter_context(rasterio.open(href)) for b, href in band_hrefs.items()}
        ref = srcs["B04"]
        aoi_geom = aoi.to_crs(ref.crs).union_all()
        windows = list(patch_windows(ref.width, ref.height, ref.transform, aoi_geom, patch_size_10m))
        rng.shuffle(windows)

        landcover_grid = landcover_row0 = landcover_col0 = None
        if use_landcover_stratification:
            landcover_grid, landcover_row0, landcover_col0 = _fetch_landcover_for_aoi(
                aoi_geom, ref, srcs["SCL"]
            )

        # Floored at MAX_PATCHES_PER_SCENE (not the runtime `max_patches`) so a deliberately
        # small `max_patches` (e.g. a quick single-scene test) doesn't shrink the candidate
        # pool below what a scene's actual validity pass rate needs — a scene that only
        # yields enough valid windows after scanning a few hundred candidates for a real
        # (30-patch) run would otherwise see its candidate pool cut to a handful and easily
        # find nothing. Caught via exactly this: `max_patches=5` returning zero patches for a
        # scene that reliably filled all 30 in the real run.
        n_candidates = max(
            max_patches * config.LANDCOVER_CANDIDATE_MULTIPLIER,
            config.MAX_PATCHES_PER_SCENE * config.LANDCOVER_CANDIDATE_MULTIPLIER,
        )
        candidates = windows[:n_candidates]

        # Pass 1: validity-check candidates, bucket survivors by land-cover class.
        class_buckets: dict[int, list] = {}
        for window in candidates:
            patch_id = f"{item.id}__r{window.row_off}_c{window.col_off}"
            if patch_id in existing_ids:
                continue

            scl_window = rasterio.windows.Window(
                window.col_off // 2, window.row_off // 2, patch_size_10m // 2, patch_size_10m // 2
            )
            scl = srcs["SCL"].read(1, window=scl_window)
            scl_transform = rasterio.windows.transform(scl_window, srcs["SCL"].transform)
            aoi_mask = rasterio.features.geometry_mask(
                [aoi_geom], out_shape=scl.shape, transform=scl_transform, invert=True
            )
            aoi_coverage = float(aoi_mask.mean())
            if aoi_coverage < min_aoi_coverage:
                continue

            is_nodata = np.isin(scl, list(SCL_NODATA_CLASSES))
            nodata_fraction = float(is_nodata.sum() / scl.size)
            if nodata_fraction > max_nodata_fraction:
                continue

            valid = ~is_nodata
            if not valid.any():
                continue
            unusable_fraction = float(np.isin(scl, list(SCL_UNUSABLE_CLASSES)).sum() / valid.sum())
            if unusable_fraction > max_unusable_fraction:
                continue

            land_class = 0
            if landcover_grid is not None:
                lc_row = scl_window.row_off - landcover_row0
                lc_col = scl_window.col_off - landcover_col0
                lc_patch = landcover_grid[lc_row : lc_row + scl_window.height, lc_col : lc_col + scl_window.width]
                if lc_patch.size:
                    land_class = int(np.bincount(lc_patch.ravel()).argmax())

            class_buckets.setdefault(land_class, []).append(
                (window, scl_window, aoi_coverage, nodata_fraction, unusable_fraction)
            )

        # Pass 2: round-robin across classes so the kept subset is diversified.
        class_keys = list(class_buckets.keys())
        rng.shuffle(class_keys)
        selected = []
        while len(selected) < n_needed and any(class_buckets.values()):
            for cls in class_keys:
                if len(selected) >= n_needed:
                    break
                bucket = class_buckets.get(cls)
                if bucket:
                    selected.append(bucket.pop())

        # Pass 3: expensive full-resolution reads, only for the selected windows.
        for window, scl_window, aoi_coverage, nodata_fraction, unusable_fraction in selected:
            patch_id = f"{item.id}__r{window.row_off}_c{window.col_off}"
            hr_10m = np.stack([srcs[b].read(1, window=window) for b in config.BANDS_10M]).astype("float32")
            hr_20m = np.stack([srcs[b].read(1, window=scl_window) for b in config.BANDS_20M]).astype("float32")

            out_path = out_dir / f"{patch_id}.npz"
            np.savez_compressed(
                out_path,
                hr_10m=hr_10m,
                hr_20m=hr_20m,
                row_off=window.row_off,
                col_off=window.col_off,
                aoi_coverage=aoi_coverage,
                nodata_fraction=nodata_fraction,
                unusable_fraction=unusable_fraction,
            )

            records.append(
                {
                    "patch_id": patch_id,
                    "scene_id": item.id,
                    "mgrs_tile": item.properties.get("s2:mgrs_tile"),
                    "datetime": item.properties.get("datetime"),
                    "row_off": window.row_off,
                    "col_off": window.col_off,
                    "aoi_coverage": aoi_coverage,
                    "nodata_fraction": nodata_fraction,
                    "unusable_fraction": unusable_fraction,
                    "path": str(out_path),
                }
            )

    return records
