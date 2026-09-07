"""ESA WorldCover land-cover fetching, for stratifying patch sampling by land-cover class.

**Unverified against the live API** — built without network access to confirm the exact
collection id / asset key Planetary Computer uses (best understanding: collection
`esa-worldcover`, asset `map`). Test with the isolated cell in `03_patch_extraction.ipynb`
("WorldCover fetch test") before this gets wired into `patches.extract_patches_for_scene`'s
actual sampling — see agents.md open caveat on land-cover stratification.
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject

from .stac import get_catalog

WORLDCOVER_COLLECTION = "esa-worldcover"

# ESA WorldCover v200 (2021) class legend.
WORLDCOVER_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse_vegetation",
    70: "snow_ice",
    80: "water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}


def fetch_landcover_grid(bounds_4326, dst_crs, dst_transform, dst_shape) -> np.ndarray:
    """Returns a `dst_shape` array of WorldCover class codes, reprojected onto the given
    target grid (`dst_crs`/`dst_transform`) with majority-vote resampling — WorldCover is
    categorical (see `WORLDCOVER_CLASSES`), so plain averaging would be wrong here, same
    reasoning as the `Resampling.mode` fix for SCL (see agents.md gotcha on that).

    `bounds_4326`: (minx, miny, maxx, maxy) in EPSG:4326, used to search for intersecting
    WorldCover tiles. Returns an all-zero (class 0 = "no data") array if none are found.
    """
    catalog = get_catalog()
    items = list(catalog.search(collections=[WORLDCOVER_COLLECTION], bbox=bounds_4326).items())
    if not items:
        return np.zeros(dst_shape, dtype="uint8")

    srcs = [rasterio.open(item.assets["map"].href) for item in items]
    try:
        mosaic, mosaic_transform = merge(srcs)
        dst = np.zeros(dst_shape, dtype="uint8")
        reproject(
            source=mosaic[0],
            destination=dst,
            src_transform=mosaic_transform,
            src_crs=srcs[0].crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.mode,
        )
        return dst
    finally:
        for src in srcs:
            src.close()
