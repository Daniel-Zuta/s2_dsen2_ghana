"""Fetch and cache country boundary polygons from OpenStreetMap's Nominatim geocoder."""

from pathlib import Path

import geopandas as gpd
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim's usage policy requires a real identifying User-Agent on requests.
USER_AGENT = "s2_dsen2-research-script (contact: zutadaniel@gmail.com)"


def fetch_country_boundary(name: str, out_path: Path) -> gpd.GeoDataFrame:
    """Return a country's admin-0 boundary as a GeoDataFrame.

    Cached to `out_path` on first call; later calls just read the cached file, so this only
    hits the network once per boundary.
    """
    out_path = Path(out_path)
    if out_path.exists():
        return gpd.read_file(out_path)

    response = requests.get(
        NOMINATIM_URL,
        params={
            "q": name,
            "format": "geojson",
            "polygon_geojson": 1,
            "featureType": "country",
            "limit": 1,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    features = response.json()["features"]
    if not features:
        raise ValueError(f"Nominatim returned no boundary for {name!r}")

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    return gdf
