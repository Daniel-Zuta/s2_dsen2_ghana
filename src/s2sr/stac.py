"""Helpers for searching Sentinel-2 L2A scenes on the Microsoft Planetary Computer STAC API."""

import geopandas as gpd
import pandas as pd
import planetary_computer
import pystac_client
from shapely.geometry import shape

from . import config

_catalog = None


def get_catalog() -> pystac_client.Client:
    """Return a cached STAC client, opening it only on the first call.

    `Client.open()` does a real HTTP round-trip (landing page + conformance). `refresh_item`
    calls this on every re-fetch, so without caching, a loop over several items opens a brand new
    connection each time — needless overhead, and each one is a fresh chance for a network hiccup
    to stall (no timeout is configured anywhere in this stack, so a stall hangs rather than errors).
    """
    global _catalog
    if _catalog is None:
        _catalog = pystac_client.Client.open(
            config.PC_STAC_URL,
            modifier=planetary_computer.sign_inplace,
        )
    return _catalog


def fetch_item_by_id(item_id: str, collection: str = config.S2_COLLECTION):
    """Fetch a single freshly-signed item from STAC by its id, or None if not found.

    The lower-level primitive behind `refresh_item` — takes just the id string (not an existing
    `item` object), so callers that only have an id on hand (e.g. a disk-cache miss, where no
    `item` object exists yet) can fetch one without needing to construct a fake item first.
    """
    catalog = get_catalog()
    results = list(catalog.search(collections=[collection], ids=[item_id]).items())
    return results[0] if results else None


def refresh_item(item, collection: str = config.S2_COLLECTION):
    """Return a freshly-signed copy of `item`, re-fetched from STAC by id.

    Planetary Computer's signed URLs expire; items fetched at the start of a long-running
    notebook (a search over hundreds of scenes, an hours-long patch extraction loop) can have
    stale tokens by the time they're actually read. Deliberately re-fetches from STAC rather than
    re-signing the existing item's href in place (`planetary_computer.sign()` on an href that may
    already carry a SAS token query string isn't guaranteed to cleanly replace it — that produced
    a real 403 in practice). Re-fetching goes through the exact same `search()` + `sign_inplace`
    path already proven to work, at the cost of one small extra metadata request. Cheap, and safe
    to call even when the existing token is still fresh.
    """
    return fetch_item_by_id(item.id, collection) or item


def search_scenes(
    bbox=config.GHANA_BBOX,
    datetime_range: str = config.DATETIME_RANGE,
    max_cloud_cover: float = config.MAX_CLOUD_COVER,
    collection: str = config.S2_COLLECTION,
):
    """Search Sentinel-2 L2A scenes over an AOI/date range, coarse-filtered on whole-granule cloud cover.

    This is only the first, cheap filtering pass (metadata only, no raster reads). Per-AOI cloud/shadow
    fraction and haze filtering happen in a later step — see agents.md "Patch sampling strategy".
    """
    catalog = get_catalog()
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    return list(search.items())


def items_to_dataframe(items) -> gpd.GeoDataFrame:
    """Flatten a list of pystac Items into a GeoDataFrame (one row per scene, with its footprint
    geometry) for quick inspection/filtering — including local AOI overlap filtering via
    `filter_by_aoi`, which needs the geometry and doesn't require any new network calls.
    """
    rows = []
    geometries = []
    for item in items:
        props = item.properties
        rows.append(
            {
                "id": item.id,
                "datetime": props.get("datetime"),
                "mgrs_tile": props.get("s2:mgrs_tile"),
                "cloud_cover": props.get("eo:cloud_cover"),
            }
        )
        geometries.append(shape(item.geometry))
    return (
        gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")
        .sort_values(["mgrs_tile", "datetime"])
        .reset_index(drop=True)
    )


def filter_by_aoi(
    gdf: gpd.GeoDataFrame,
    aoi_path=config.GHANA_BOUNDARY_PATH,
    min_overlap_fraction: float = config.MIN_AOI_OVERLAP_FRACTION,
) -> gpd.GeoDataFrame:
    """Keep only scenes whose footprint overlaps the real AOI boundary by at least
    `min_overlap_fraction` of the footprint's own area. Drops tiles that only clip the coarse
    search bbox (e.g. mostly ocean or a neighboring country) without narrowing the STAC search
    itself. Purely local (geometry already returned by the search) — no network calls.
    """
    aoi = gpd.read_file(aoi_path).to_crs(gdf.crs)
    aoi_union = aoi.union_all()

    overlap_fraction = gdf.geometry.apply(
        lambda geom: geom.intersection(aoi_union).area / geom.area if geom.area else 0.0
    )
    return (
        gdf.assign(aoi_overlap_fraction=overlap_fraction)
        .loc[lambda d: d["aoi_overlap_fraction"] >= min_overlap_fraction]
        .reset_index(drop=True)
    )
