"""
config.py — single source of truth for the drought model pipeline.

Mirrors the flood model's config pattern (per Section 9, Person A owns the
shared grid/pipeline convention — reuse the same GRID cell size and region
bounding boxes here so flood + drought outputs line up on the same map).
"""

# ---------------------------------------------------------------------------
# Demo regions (drought-prone, per Section 1 "Core decisions locked in")
# Pick ONE of these as your primary demo region at hour 0 — whichever has the
# cleanest India Drought Monitor district coverage for your judging window.
# bbox format: [min_lon, min_lat, max_lon, max_lat]
# ---------------------------------------------------------------------------
REGIONS = {
    "marathwada": {
        "bbox": [74.5, 17.5, 78.0, 20.5],
        "districts": [
            "Aurangabad", "Jalna", "Beed", "Latur", "Osmanabad",
            "Nanded", "Parbhani", "Hingoli",
        ],
        "state": "Maharashtra",
    },
    "bundelkhand": {
        "bbox": [78.0, 24.0, 81.5, 26.5],
        "districts": [
            "Jhansi", "Lalitpur", "Hamirpur", "Mahoba", "Banda",
            "Chitrakoot", "Tikamgarh", "Chhatarpur", "Panna", "Damoh",
            "Sagar", "Datia",
        ],
        "state": "Uttar Pradesh / Madhya Pradesh",
    },
    "rayalaseema": {
        "bbox": [77.0, 13.0, 79.5, 16.0],
        "districts": [
            "Anantapur", "Kadapa", "Kurnool", "Chittoor",
        ],
        "state": "Andhra Pradesh",
    },
}

DEFAULT_REGION = "marathwada"

# ---------------------------------------------------------------------------
# Grid definition — keep in sync with the flood pipeline's grid
# ---------------------------------------------------------------------------
GRID_SCALE_M = 5000          # ~5km cells, matches Section 5 flood grid
FEATURE_REDUCE_SCALE_M = 1000 # per-source native-res reduction before averaging into a cell (1000m fits GEE getInfo memory limit)

# UTM zone per region (needed for ee.Projection — pick the zone covering your AOI)
UTM_EPSG = {
    "marathwada": "EPSG:32643",
    "bundelkhand": "EPSG:32644",
    "rayalaseema": "EPSG:32644",
}

# ---------------------------------------------------------------------------
# Climatology baseline window — used for all "anomaly vs normal" features
# (rainfall anomaly, NDVI anomaly, LST anomaly). 30-year baseline is standard
# practice (WMO convention); India Drought Atlas covers 1901-2021 if you want
# a deeper baseline, but CHIRPS itself only starts 1981, so 1991-2020 is the
# practical ceiling for an in-pipeline climatology computed from CHIRPS/MODIS.
# ---------------------------------------------------------------------------
CLIMATOLOGY_START_YEAR = 1991
CLIMATOLOGY_END_YEAR = 2020

# Rainfall deficit accumulation windows (days) — Section 5.2 feature list
DEFICIT_WINDOWS = [7, 30, 60, 90]

# Dry-spell threshold — a day counts as "dry" below this rainfall (mm)
DRY_DAY_THRESHOLD_MM = 1.0

# ---------------------------------------------------------------------------
# GEE dataset IDs (Section 3 master reference)
# ---------------------------------------------------------------------------
DATASETS = {
    "chirps_daily": "UCSB-CHG/CHIRPS/DAILY",
    "smap_l4": "NASA/SMAP/SPL4SMGP/008",
    "modis_ndvi": "MODIS/061/MOD13Q1",
    "modis_lst": "MODIS/061/MOD11A1",
    "modis_et": "MODIS/061/MOD16A2GF",       # gap-filled — NOT MOD16A2
    "dynamic_world": "GOOGLE/DYNAMICWORLD/V1",
    "grace": "NASA/GRACE/MASS_GRIDS_V04/LAND",
}

# ---------------------------------------------------------------------------
# India Drought Monitor (IIT Gandhinagar) — historical CDI labels
# ---------------------------------------------------------------------------
DROUGHT_MONITOR_URL = "https://indiadroughtmonitor.in/"
DROUGHT_ATLAS_REPO = "https://github.com/wcl-iitgn/india-drought-atlas-data"

# 5-class Combined Drought Index -> ordinal severity used as regression target
CDI_CLASS_TO_SEVERITY = {
    "None": 0,
    "D0": 1,   # abnormally dry
    "D1": 2,
    "D2": 3,
    "D3": 4,
    "D4": 5,   # exceptional drought
}
MAX_SEVERITY = 5  # used to normalize severity -> 0-1 risk score

# ---------------------------------------------------------------------------
# Forecast horizon for the timeline slider (Section 6)
# Drought "forecast" = trend + climatology projection, NOT a hard forecast.
# ---------------------------------------------------------------------------
FORECAST_HORIZONS_DAYS = [7, 15, 30, 60, 90]

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUT_DIR = "data"
FEATURES_CSV = f"{OUT_DIR}/drought_features_{{region}}.csv"
LABELS_CSV = f"{OUT_DIR}/drought_labels_{{region}}.csv"
TRAINING_TABLE_CSV = f"{OUT_DIR}/drought_training_table_{{region}}.csv"
MODEL_PATH = f"{OUT_DIR}/drought_model_{{region}}.joblib"
RISK_GEOJSON_DIR = f"{OUT_DIR}/risk_geojson_{{region}}"
