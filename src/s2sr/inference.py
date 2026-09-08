"""Real (non-simulated) inference: apply the trained 20m->10m model to actual Sentinel-2
scenes, producing a proper georeferenced GeoTIFF at 10m resolution for the 20m bands.

Unlike training (see `dataset.WaldPairDataset`), no degradation is applied here — `guide`
is the scene's real native 10m bands, and the model's low-res input is the scene's real
native 20m bands bicubic-upsampled directly to the 10m grid. This preserves the same
*relative* scale relationship (guide : low-res input = 2:1) the model was trained on under
Wald's protocol, which is what lets a model trained on synthetically-downsampled pairs
generalize to real full-resolution data — see agents.md "Methodology".
"""

import contextlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
import torch
from rasterio.merge import merge

from . import config
from .dataset import upsample_bicubic
from .patches import patch_windows
from .stac import refresh_item


def superresolve_scene(
    item,
    aoi: gpd.GeoDataFrame,
    model: torch.nn.Module,
    device: torch.device,
    out_path,
    patch_size: int = config.PATCH_SIZE_10M,
) -> Path:
    """Runs `model` over every `patch_size` x `patch_size` (10m) tile intersecting `aoi`
    within `item`, and writes a complete 10-band product to `out_path` as one
    correctly-georeferenced GeoTIFF, all at native 10m resolution: the 4 native 10m bands
    (`config.BANDS_10M`, passed through unchanged — they're already full resolution, the model
    never touches them) followed by the 6 super-resolved 20m bands (`config.BANDS_20M`). Band
    order in the file is `BANDS_10M + BANDS_20M`; `dst.descriptions` names each one.

    Tiling matches `patches.patch_windows` (non-overlapping, snapped to the patch grid), so
    tiles partially outside the raster's extent are simply not covered — same known edge
    limitation as patch extraction (see agents.md notebook 3 description).
    """
    model.eval()
    item = refresh_item(item)
    band_hrefs = {b: item.assets[b].href for b in [*config.BANDS_10M, *config.BANDS_20M]}
    downsample_factor = config.DOWNSAMPLE_FACTOR_20M

    with contextlib.ExitStack() as stack:
        srcs = {b: stack.enter_context(rasterio.open(href)) for b, href in band_hrefs.items()}
        ref = srcs[config.BANDS_10M[0]]
        aoi_geom = aoi.to_crs(ref.crs).union_all()
        windows = list(patch_windows(ref.width, ref.height, ref.transform, aoi_geom, patch_size))

        profile = ref.profile.copy()
        profile.update(
            count=len(config.BANDS_10M) + len(config.BANDS_20M),
            dtype="float32",
            compress="deflate",
            nodata=None,
            BIGTIFF="YES",  # a 10-band float32 output can exceed plain TIFF's 4GB offset limit
        )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.descriptions = tuple(config.BANDS_10M) + tuple(config.BANDS_20M)
            for window in windows:
                native_20m_window = rasterio.windows.Window(
                    window.col_off // downsample_factor,
                    window.row_off // downsample_factor,
                    patch_size // downsample_factor,
                    patch_size // downsample_factor,
                )
                guide = np.stack([srcs[b].read(1, window=window) for b in config.BANDS_10M]).astype("float32")
                native_20m = np.stack(
                    [srcs[b].read(1, window=native_20m_window) for b in config.BANDS_20M]
                ).astype("float32")
                upsampled = upsample_bicubic(native_20m, out_size=patch_size)

                guide_t = torch.from_numpy(guide).unsqueeze(0).to(device)
                upsampled_t = torch.from_numpy(upsampled).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(guide_t, upsampled_t)[0].cpu().numpy()

                dst.write(np.concatenate([guide, pred], axis=0), window=window)

    return out_path


def mosaic_geotiffs(paths, out_path) -> Path:
    """Merge multiple single-scene outputs from `superresolve_scene` into one GeoTIFF — for an
    AOI that spans more than one Sentinel-2 tile/scene, which `superresolve_scene` itself
    doesn't handle (it processes exactly one scene). Where two source tiles overlap,
    `rasterio.merge`'s default (first non-nodata pixel wins, in `paths` order) applies.
    """
    srcs = [rasterio.open(p) for p in paths]
    try:
        mosaic, mosaic_transform = merge(srcs)
        profile = srcs[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=mosaic_transform,
            BIGTIFF="YES",  # a multi-tile, 10-band float32 mosaic can easily exceed plain TIFF's 4GB limit
        )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.descriptions = srcs[0].descriptions
            dst.write(mosaic)
    finally:
        for src in srcs:
            src.close()

    return out_path
