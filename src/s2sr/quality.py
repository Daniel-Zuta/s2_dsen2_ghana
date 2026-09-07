"""Per-scene quality checks: SCL cloud/shadow/snow fraction and a simple Harmattan-haze indicator.

Both checks read bands decimated and cached (via `raster.read_band_decimated_cached`), so they're
cheap enough to run over hundreds of scenes without downloading full-resolution imagery, and
reruns hit the local cache instead of re-downloading — see agents.md "Patch sampling strategy".
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
from rasterio.enums import Resampling

from . import raster

# Sentinel-2 L2A Scene Classification (SCL) band legend.
# "Unusable" = cloud shadow, cloud (any confidence), thin cirrus, snow/ice.
SCL_UNUSABLE_CLASSES = {3, 8, 9, 10, 11}
# "No real data" = missing / saturated-defective — excluded from both numerator and denominator.
SCL_NODATA_CLASSES = {0, 1}


def aoi_mask(shape, transform, crs, aoi: gpd.GeoDataFrame) -> np.ndarray:
    """Rasterize the AOI polygon (reprojected to `crs`) onto a `shape` grid defined by `transform`.
    Returns a boolean array, True where a pixel is inside the AOI.
    """
    aoi_geom = aoi.to_crs(crs).union_all()
    return rasterio.features.geometry_mask(
        [aoi_geom], out_shape=shape, transform=transform, invert=True
    )


def scl_stats(item, aoi: gpd.GeoDataFrame, out_size: int = 256) -> dict:
    """AOI-intersecting SCL stats for one scene, from a single read:

    - `cloud_shadow_fraction`: fraction of *valid* (non-nodata) AOI pixels that are
      cloud/shadow/cirrus/snow. 1.0 if there's no valid data in the AOI at all.
    - `nodata_fraction`: fraction of *all* AOI pixels that are nodata (missing/saturated).

    Kept as two separate numbers deliberately — `cloud_shadow_fraction` alone can't tell a clean
    scene from one that's mostly a partial-swath gap with a small cloud-free sliver of real data
    (nodata pixels are excluded from its denominator). Both need checking.
    """
    scl, transform, crs = raster.read_band_decimated_cached(
        item, "SCL", out_size=out_size, resampling=Resampling.mode
    )
    mask = aoi_mask(scl.shape, transform, crs, aoi)
    if not mask.any():
        return {"cloud_shadow_fraction": 1.0, "nodata_fraction": 1.0}

    is_nodata = np.isin(scl, list(SCL_NODATA_CLASSES))
    nodata_fraction = float((mask & is_nodata).sum() / mask.sum())

    valid = mask & ~is_nodata
    if not valid.any():
        cloud_shadow_fraction = 1.0
    else:
        unusable = valid & np.isin(scl, list(SCL_UNUSABLE_CLASSES))
        cloud_shadow_fraction = float(unusable.sum() / valid.sum())

    return {"cloud_shadow_fraction": cloud_shadow_fraction, "nodata_fraction": nodata_fraction}


def blue_reflectance_in_aoi(item, aoi: gpd.GeoDataFrame, out_size: int = 256) -> float:
    """Mean B02 (blue) reflectance within the AOI — a cheap haze indicator (Harmattan dust and
    other haze raise blue-band reflectance relative to a clear scene over the same tile).
    """
    blue, transform, crs = raster.read_band_decimated_cached(item, "B02", out_size=out_size)
    mask = aoi_mask(blue.shape, transform, crs, aoi)
    valid = mask & (blue > 0)
    if not valid.any():
        return float("nan")
    return float(blue[valid].mean())


def haze_modified_zscore(series: pd.Series) -> pd.Series:
    """Modified z-score (median/MAD based, robust to outliers) of a per-tile blue-reflectance
    series. Meant to be used via `.groupby("mgrs_tile")["blue_mean"].transform(haze_modified_zscore)`
    so each scene is compared against its own tile's typical brightness, not a global one.
    """
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=series.index)
    return 0.6745 * (series - median) / mad
