"""Patch extraction: spatial sampling of aligned 10m/20m band patches with per-patch AOI + SCL
validity filtering.

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
import rasterio.windows

from . import config
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
):
    """Extract up to `max_patches` valid patches from one scene.

    Each patch is saved as a compressed `.npz` with `hr_10m` (4 x P x P, native 10m bands),
    `hr_20m` (6 x P/2 x P/2, native 20m bands) — no downsampling applied — plus its validity
    stats. Candidate patch windows are shuffled before filtering, so the kept subset is a random
    sample of the AOI, not just whatever comes first in raster order.

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

    with contextlib.ExitStack() as stack:
        srcs = {b: stack.enter_context(rasterio.open(href)) for b, href in band_hrefs.items()}
        ref = srcs["B04"]
        aoi_geom = aoi.to_crs(ref.crs).union_all()
        windows = list(patch_windows(ref.width, ref.height, ref.transform, aoi_geom, patch_size_10m))
        rng.shuffle(windows)

        for window in windows:
            if len(records) >= max_patches:
                break

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
