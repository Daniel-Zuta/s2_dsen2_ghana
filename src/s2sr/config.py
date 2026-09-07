"""Project-wide configuration constants. See agents.md for the reasoning behind these choices."""

from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Study area: Ghana, national extent (EPSG:4326 bbox: minx, miny, maxx, maxy).
# Deliberately generous — used only as the STAC search filter (cheap superset). Precise AOI
# tightening happens locally against GHANA_BOUNDARY_PATH, not by narrowing this bbox, so the
# search query itself never needs to change.
GHANA_BBOX = (-3.262441, 4.736723, 1.191781, 11.173301)

# Ghana's real admin-0 boundary (fetched from OSM Nominatim, cached here — see s2sr.boundaries).
# Used to drop scenes whose footprint barely overlaps Ghana (e.g. tiles that mostly cover ocean
# or a neighboring country but happen to clip GHANA_BBOX).
GHANA_BOUNDARY_PATH = REPO_ROOT / "data" / "boundaries" / "ghana_adm0.geojson"

# Minimum fraction of a scene's footprint area that must overlap the real boundary to keep it.
MIN_AOI_OVERLAP_FRACTION = 0.01

# Dry season composite window
DATE_START = date(2025, 1, 1)
DATE_END = date(2025, 5, 31)
DATETIME_RANGE = f"{DATE_START.isoformat()}/{DATE_END.isoformat()}"

# STAC (Microsoft Planetary Computer — primary source, see agents.md)
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"

# Coarse scene filter on whole-granule metadata; refine later with a per-AOI SCL cloud/shadow
# fraction check plus a haze check (Harmattan dust isn't flagged by cloud metadata — see agents.md).
MAX_CLOUD_COVER = 20  # percent

# Per-AOI SCL cloud/shadow/cirrus/snow fraction filter (refines the coarse eo:cloud_cover pass
# above, which is whole-granule and often misleading — see agents.md). Starting point; tune after
# inspecting real results.
MAX_SCL_CLOUD_SHADOW_FRACTION = 0.10

# Modified z-score threshold (per MGRS tile) for flagging Harmattan-haze-affected scenes via
# blue-band reflectance. Starting point; tune after inspecting real results.
HAZE_ZSCORE_THRESHOLD = 3.5

# Time bin used when picking one best scene per tile per window (pandas period frequency string).
SELECTION_TIME_WINDOW = "M"  # monthly

# Sentinel-2 band groups by native resolution
BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]
BANDS_60M = ["B01", "B09"]

# Patch sampling (see agents.md "Patch sampling strategy")
PATCH_SIZE_10M = 128
DOWNSAMPLE_FACTOR_20M = 2
DOWNSAMPLE_FACTOR_60M = 6

# Per-patch validity thresholds (see s2sr.patches). Starting points; tune after inspecting results.
# Stricter than the scene-level MIN_AOI_OVERLAP_FRACTION on purpose — a training patch should be
# almost entirely inside the AOI, not just clip it.
MIN_PATCH_AOI_COVERAGE = 0.95
MAX_PATCH_UNUSABLE_FRACTION = 0.05
MAX_PATCHES_PER_SCENE = 30

# Max fraction of AOI-intersecting SCL pixels allowed to be nodata (missing/saturated), checked
# SEPARATELY from cloud/shadow fraction — nodata pixels are excluded from that fraction's own
# denominator, so a scene/patch that's mostly a partial-swath gap and cloud-free in the small
# valid remainder would otherwise score as "clean". Found via eyeball-checking notebook 2 (a
# batch of high-blue-mean survivors turned out to be half-nodata scenes, not haze).
MAX_SCL_NODATA_FRACTION = 0.05
MAX_PATCH_NODATA_FRACTION = 0.05

# Local cache for decimated scene-level reads (notebook 2) — keyed by scene id/band/out_size, not
# by href (signed hrefs change every re-fetch). Bandwidth on a metered connection is a real
# constraint here; without this, every notebook rerun re-downloads identical data from scratch.
RASTER_CACHE_DIR = REPO_ROOT / "data" / "interim" / "raster_cache"

# Land-cover-stratified patch sampling (see s2sr.patches, s2sr.landcover). Stratification needs
# to see a pool of candidates before picking winners, so it can't short-circuit early the way
# pure-random sampling could — this bounds that pool to `MAX_PATCHES_PER_SCENE *
# LANDCOVER_CANDIDATE_MULTIPLIER` candidate windows per scene, trading some stratification
# quality for a predictable worst-case cost per scene.
LANDCOVER_CANDIDATE_MULTIPLIER = 8
