"""Helpers for reading Sentinel-2 raster bands from signed Planetary Computer asset URLs."""

import os

import joblib
import numpy as np
import rasterio
from rasterio.enums import Resampling

from . import config
from .stac import fetch_item_by_id

# Recommended settings for efficient repeated reads of cloud-optimized GeoTIFFs over HTTP —
# avoids extra directory-listing requests and restricts curl to the extensions we actually need.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

# Disk-backed memoization for decimated scene-level reads. Bandwidth is a real constraint on the
# user's connection (mobile hotspot at home), and notebook 2 reruns the same ~657 scenes
# repeatedly while thresholds get tuned — this makes reruns hit disk instead of re-downloading.
# joblib hashes the cached function's own source as part of the cache key, so editing the read
# logic below auto-invalidates old entries — no manual cache-clearing needed when the code changes
# (unlike a hand-rolled cache keyed only on scene id/band/out_size).
_memory = joblib.Memory(location=config.RASTER_CACHE_DIR, verbose=0)


@_memory.cache
def _read_band_decimated_by_id(item_id: str, band: str, out_size: int, resampling: Resampling):
    """Fetch one item fresh by id and read one band decimated to ~out_size x out_size.

    Deliberately keyed on `item_id` (stable) rather than a signed href (changes every fetch, so
    would never hit the cache) — the id-to-href lookup happens inside here, only on a cache miss.
    Uses the COG's built-in overviews via rasterio's `out_shape` decimated read, so a cache miss
    still only transfers a small amount of data — this is for scene-level statistics (cloud
    fraction, mean reflectance), not full-resolution patch extraction.

    `resampling` matters: `Resampling.average` is right for continuous reflectance bands, but
    wrong for a categorical band like SCL — averaging class codes numerically (e.g. blending
    vegetation=4 and cloud=8 near a boundary) can produce a value that doesn't correspond to any
    real class, or worse, coincidentally matches a *different* class's code. Callers reading SCL
    should pass `Resampling.mode` (majority vote) instead.
    """
    item = fetch_item_by_id(item_id)
    if item is None:
        raise ValueError(f"Could not find item {item_id!r} in STAC collection {config.S2_COLLECTION!r}")
    href = item.assets[band].href

    with rasterio.open(href) as src:
        scale = min(out_size / src.width, out_size / src.height, 1.0)
        out_shape = (src.count, max(1, round(src.height * scale)), max(1, round(src.width * scale)))
        data = src.read(out_shape=out_shape, resampling=resampling)
        transform = src.transform * src.transform.scale(
            src.width / out_shape[-1], src.height / out_shape[-2]
        )
        return data[0], transform, src.crs


def read_band_decimated_cached(item, band: str, out_size: int = 256, resampling: Resampling = Resampling.average):
    """Decimated read of one band for `item`, cached to disk by (item id, band, out_size, resampling).

    Defaults to `Resampling.average`, correct for continuous reflectance bands (B02/B03/B04/...).
    Pass `resampling=Resampling.mode` when reading a categorical band (SCL) — see the docstring on
    `_read_band_decimated_by_id` for why this isn't just a performance knob.
    """
    return _read_band_decimated_by_id(item.id, band, out_size, resampling)


def percentile_stretch(array: np.ndarray, percentile: tuple = (2, 98)) -> np.ndarray:
    """Simple percentile stretch of an array (any shape) to float32 [0, 1], for display.
    Zero/nodata pixels are excluded from the percentile computation itself.
    """
    valid = array[array > 0]
    if valid.size == 0:
        return np.zeros_like(array, dtype="float32")
    lo, hi = np.percentile(valid, percentile)
    return np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1).astype("float32")


def read_rgb_preview(item, out_size: int = 256, percentile: tuple = (2, 98)) -> np.ndarray:
    """Read a decimated true-color (B04, B03, B02) composite for quick visual inspection.
    Returns an (H, W, 3) float32 array in [0, 1], ready for `plt.imshow`.
    """
    bands = [
        read_band_decimated_cached(item, band, out_size=out_size)[0].astype("float32")
        for band in ("B04", "B03", "B02")
    ]
    rgb = np.stack(bands, axis=-1)
    return percentile_stretch(rgb, percentile)
