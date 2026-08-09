#!/usr/bin/env python3
"""
api_server.py — Flask API that serves flood risk predictions from the trained XGBoost model.

Endpoints:
  GET /api/risk-grid/assam          → GeoJSON FeatureCollection with risk per district
  GET /api/risk-grid/assam?month=7  → Risk for a specific month (default: current)
  GET /api/districts                → List of all districts with latest risk
  GET /api/health                   → Health check

The frontend connects to this API to replace mock data with real model predictions.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys, os
sys.path.insert(0, str(Path(__file__).parent))
try:
    from live_weather import get_live_weather_features, clear_cache, get_weather_system_status
    LIVE_WEATHER_ENABLED = True
    print("  ✓ Live weather module loaded (Open-Meteo + Climatology Fallback)")
except ImportError:
    LIVE_WEATHER_ENABLED = False
    get_weather_system_status = lambda: {"status": "disabled", "warning": "live_weather module not loaded", "rate_limited": False}
    print("  ⚠ live_weather.py not found — falling back to historical data")

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-cors"])
    from flask import Flask, jsonify, request
    from flask_cors import CORS

try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost as xgb

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FEATURES_DIR = BASE_DIR / "data" / "features"
RAW_DIR = BASE_DIR / "data" / "raw"

# ── Feature columns (must match training order) ──
FEATURE_COLS = [
    "elevation", "slope", "flow_acc", "dist_to_river", "water_occurrence",
    "rain_monthly_mm", "rain_7d_mm", "rain_3d_mm", "rain_1d_mm",
    "rain_daily_mean_mm", "rain_anomaly",
    "sm_surface", "sm_rootzone",
    "lst_day_k", "et_mm", "built_frac", "water_frac",
    "dfsi_score", "pct_flooded_area", "mean_flood_duration",
    "population", "historical_fatalities", "hist_flood_frequency",
    "month",
]

# Risk thresholds matching the frontend's design system
def get_risk_level(score):
    if score >= 0.60: return "severe"
    if score >= 0.35: return "high"
    if score >= 0.15: return "moderate"
    return "low"

# ── Calibrated scoring using raw margins ──
# predict_proba gives binary 0/1 because the model is very confident.
# Raw margins (log-odds) have continuous variation; applying a softer
# sigmoid spreads them into a realistic risk gradient.
CALIBRATION_TEMP = 4.0

def calibrated_score_single(model_obj, X_df):
    """Get a calibrated risk score for a single sample using raw margins."""
    import xgboost as xgb
    dmat = xgb.DMatrix(X_df, feature_names=list(X_df.columns))
    margin = float(model_obj.get_booster().predict(dmat, output_margin=True)[0])
    return float(1.0 / (1.0 + np.exp(-margin / CALIBRATION_TEMP)))

def calibrated_scores_batch(model_obj, X_df):
    """Get calibrated risk scores for a batch of samples using raw margins."""
    import xgboost as xgb
    dmat = xgb.DMatrix(X_df, feature_names=list(X_df.columns))
    margins = model_obj.get_booster().predict(dmat, output_margin=True)
    return 1.0 / (1.0 + np.exp(-margins / CALIBRATION_TEMP))

def get_alert_message(district, risk_level, factors):
    messages = {
        "severe": f"SEVERE flood risk in {district}. Rainfall anomaly {factors.get('rain_anomaly', 'N/A')}σ above normal. Evacuate low-lying areas immediately.",
        "high": f"HIGH flood advisory for {district}. Elevated soil moisture ({factors.get('sm_surface', 'N/A')}%) and above-normal rainfall. Prepare flood defenses.",
        "moderate": f"MODERATE flood watch for {district}. Monitor water levels and prepare contingency plans.",
        "low": f"Low flood risk in {district}. Normal conditions.",
    }
    return messages.get(risk_level, "")

def get_days_to_event(risk_level):
    """Estimate days to potential event based on risk level."""
    if risk_level == "severe": return np.random.randint(1, 4)
    if risk_level == "high": return np.random.randint(3, 8)
    if risk_level == "moderate": return np.random.randint(7, 15)
    return 15


# ══════════════════════════════════════════════
# LOAD MODEL + DATA AT STARTUP
# ══════════════════════════════════════════════

print("Loading flood model and data...")

# Load trained model (prefer multistate if available)
model = xgb.XGBClassifier()
if (MODELS_DIR / "flood_model_multistate.json").exists():
    model.load_model(str(MODELS_DIR / "flood_model_multistate.json"))
    print(f"  ✓ Model loaded (4-state multistate)")
else:
    model.load_model(str(MODELS_DIR / "flood_model.json"))
    print(f"  ✓ Model loaded (Assam only)")

# Load training table (has all features + district assignments)
multistate_table = FEATURES_DIR / "training_table_multistate.csv"
if multistate_table.exists():
    training_table = pd.read_csv(multistate_table)
    print(f"  ✓ Training table (4-state): {training_table.shape}")
else:
    training_table = pd.read_csv(FEATURES_DIR / "training_table.csv")
    print(f"  ✓ Training table (Assam-only): {training_table.shape}")

# Load district boundaries for all states
import geopandas as gpd

STATE_BOUNDARY_FILES = {
    "assam":       ("assam_districts.geojson",       "NAME_2"),
    "bihar":       ("bihar_districts.geojson",       "shapeName"),
    "west_bengal": ("west_bengal_districts.geojson", "shapeName"),
    "odisha":      ("odisha_districts.geojson",      "shapeName"),
}
ASSAM_NAME_FIXES = {
    "SIBSAGAR": "SIVASAGAR", "NORTH CACHAR HILLS": "DIMA HASAO",
    "DIMAHASAO": "DIMA HASAO", "KARBIANGLONG": "KARBI ANGLONG",
    "KARBI ANGLONG WEST": "WEST KARBI ANGLONG",
    "KAMRUPMETROPOLITAN": "KAMRUP METROPOLITAN",
}

all_state_gdfs = {}  # state_key → GeoDataFrame
for state_key, (fname, name_col) in STATE_BOUNDARY_FILES.items():
    fpath = RAW_DIR / fname
    if fpath.exists():
        gdf = gpd.read_file(fpath)
        if name_col in gdf.columns:
            gdf["district_name"] = gdf[name_col].str.strip().str.upper()
        else:
            for c in gdf.columns:
                if "name" in c.lower() and gdf[c].dtype == object:
                    gdf["district_name"] = gdf[c].str.strip().str.upper()
                    break
        if state_key == "assam":
            gdf["district_name"] = gdf["district_name"].replace(ASSAM_NAME_FIXES)
        all_state_gdfs[state_key] = gdf

# Default to assam for backward compatibility
district_gdf = all_state_gdfs.get("assam", gpd.GeoDataFrame())
print(f"  ✓ District boundaries: {', '.join(f'{k}={len(v)}' for k,v in all_state_gdfs.items())}")

# Pre-compute district-level feature medians per month (for fast inference)
district_features = training_table.groupby(["district_name", "month"])[FEATURE_COLS[:-1]].median().reset_index()
district_features["month"] = district_features["month"].astype(int)
print(f"  ✓ District feature profiles: {district_features.shape}")

# Pre-compute district centroids for live weather lookup
district_centroids = {}
for _, row in district_gdf.iterrows():
    centroid = row["geometry"].centroid
    district_centroids[row["district_name"]] = {
        "lat": round(centroid.y, 4),
        "lon": round(centroid.x, 4),
    }
print(f"  ✓ District centroids: {len(district_centroids)}")

# Compute centroids for ALL states
for state_key, gdf in all_state_gdfs.items():
    for _, row in gdf.iterrows():
        dname = row["district_name"]
        if dname not in district_centroids:
            centroid = row["geometry"].centroid
            district_centroids[dname] = {
                "lat": round(centroid.y, 4),
                "lon": round(centroid.x, 4),
                "state": state_key,
            }
print(f"  ✓ All-state centroids: {len(district_centroids)}")

DISTRICT_TO_STATE = {}
for state_key, gdf in all_state_gdfs.items():
    for _, row in gdf.iterrows():
        DISTRICT_TO_STATE[row["district_name"]] = state_key

# Pre-warm live weather cache at startup (background, non-blocking)
_live_weather_cache = {}
if LIVE_WEATHER_ENABLED:
    print("  Fetching live weather for all districts...")
    import threading
    def _warm_cache():
        global _live_weather_cache
        month = max(5, min(10, datetime.now().month))
        for dist, coords in district_centroids.items():
            skey = DISTRICT_TO_STATE.get(dist, "assam")
            w = get_live_weather_features(coords["lat"], coords["lon"], month=month, state_key=skey)
            if w:
                _live_weather_cache[dist] = w
        print(f"  ✓ Live weather cached: {len(_live_weather_cache)}/{len(district_centroids)} districts")
    threading.Thread(target=_warm_cache, daemon=True).start()

# Available districts
available_districts = sorted(training_table["district_name"].unique())
print(f"  ✓ Districts: {available_districts}")

# ── Pre-compute per-CELL risk scores (one unique cell per lat/lon) ──
print("  Pre-computing per-cell risk scores...")
from pyproj import Transformer

# UTM 46N → WGS84 transformer
_transformer = Transformer.from_crs("EPSG:32646", "EPSG:4326", always_xy=True)

def cell_to_geojson_polygon(lon_val, lat_val, half_deg=0.0225, half_m=2500):
    """Convert cell coordinates (WGS84 degrees or UTM meters) to GeoJSON Polygon."""
    if lon_val > 180:  # UTM meters
        corners_utm = [
            (lon_val - half_m, lat_val - half_m),
            (lon_val + half_m, lat_val - half_m),
            (lon_val + half_m, lat_val + half_m),
            (lon_val - half_m, lat_val + half_m),
            (lon_val - half_m, lat_val - half_m),
        ]
        lons, lats = _transformer.transform(
            [c[0] for c in corners_utm],
            [c[1] for c in corners_utm],
        )
        ring = [[round(lon, 5), round(lat, 5)] for lon, lat in zip(lons, lats)]
    else:  # WGS84 degrees
        ring = [
            [round(lon_val - half_deg, 5), round(lat_val - half_deg, 5)],
            [round(lon_val + half_deg, 5), round(lat_val - half_deg, 5)],
            [round(lon_val + half_deg, 5), round(lat_val + half_deg, 5)],
            [round(lon_val - half_deg, 5), round(lat_val + half_deg, 5)],
            [round(lon_val - half_deg, 5), round(lat_val - half_deg, 5)],
        ]
    return {"type": "Polygon", "coordinates": [ring]}

def to_wgs_coords(lon_val, lat_val):
    """Helper to ensure WGS84 (lon, lat) tuple."""
    if lon_val > 180:
        w_lon, w_lat = _transformer.transform(lon_val, lat_val)
        return round(float(w_lon), 5), round(float(w_lat), 5)
    return round(float(lon_val), 5), round(float(lat_val), 5)

# Get unique cells from the training table across all states
cell_coords = training_table[["cell_lon", "cell_lat"]].drop_duplicates().reset_index(drop=True)

# Pre-compute cell polygons (done once at startup)
print(f"    Building {len(cell_coords):,} cell polygons...")
cell_polygons = {}  # key: (cell_lon, cell_lat) → geojson geometry
for _, row in cell_coords.iterrows():
    key = (row["cell_lon"], row["cell_lat"])
    cell_polygons[key] = cell_to_geojson_polygon(row["cell_lon"], row["cell_lat"])

# Pre-compute per-cell features per month (sample 1 row per cell per month)
print("    Sampling per-cell features per month...")
_non_month_feats = [c for c in FEATURE_COLS if c != "month"]

cell_month_features = (
    training_table
    .groupby(["cell_lon", "cell_lat", "month"])[_non_month_feats]
    .median()
    .reset_index()
)

# Add WGS84 centroid coords for bbox filtering
_wgs_list = [to_wgs_coords(x, y) for x, y in zip(cell_month_features["cell_lon"].values, cell_month_features["cell_lat"].values)]
cell_month_features["wgs_lon"] = [c[0] for c in _wgs_list]
cell_month_features["wgs_lat"] = [c[1] for c in _wgs_list]
# Add district_name lookup to cell_month_features
cell_district_map = (
    training_table[["cell_lon", "cell_lat", "district_name"]]
    .drop_duplicates(subset=["cell_lon", "cell_lat"])
    .set_index(["cell_lon", "cell_lat"])["district_name"]
    .to_dict()
)
cell_month_features["district_name"] = [
    cell_district_map.get((lon, lat), "Flood Zone")
    for lon, lat in zip(cell_month_features["cell_lon"].values, cell_month_features["cell_lat"].values)
]

print(f"  ✓ Cell-month feature table: {cell_month_features.shape}")





# ══════════════════════════════════════════════
app = Flask(__name__)
CORS(app)  # Allow frontend on different port


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Project Hydra — Flood & Drought EWS API",
        "frontend_app": "http://localhost:5173",
        "health": "http://localhost:5001/api/health",
        "endpoints": {
            "flood_districts": "http://localhost:5001/api/risk-grid/flood",
            "flood_5km_cells": "http://localhost:5001/api/risk-grid/flood/cells",
            "districts_list": "http://localhost:5001/api/districts",
            "feature_importance": "http://localhost:5001/api/feature-importance"
        }
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "XGBoost Flood Risk v1",
        "districts": len(available_districts),
        "features": len(FEATURE_COLS),
        "training_rows": len(training_table),
    })


@app.route("/api/risk-grid/flood", methods=["GET"])
@app.route("/api/risk-grid/assam", methods=["GET"])
@app.route("/api/risk-grid/<region>", methods=["GET"])
def risk_grid_flood(region=None):
    """
    Return GeoJSON FeatureCollection with risk score per district.
    Query params:
      - month (int, 5-10): monsoon month (default: current or 7)
      - day (int, -7 to 15): day forecast or historical offset (default: 0)
      - year (int): year context (default: 2023)
    """
    now = datetime.now()
    month = request.args.get("month", type=int, default=max(5, min(10, now.month)))
    day = request.args.get("day", type=int, default=0)
    year = request.args.get("year", type=int, default=2023)

    month = max(5, min(10, month))
    day = max(-7, min(15, day))

    features_list = []

    # Iterate over ALL states' district boundaries
    for state_key, gdf in all_state_gdfs.items():
        state_label = state_key.replace("_", " ").title()
        for _, row in gdf.iterrows():
            district = row["district_name"]
            geometry = row["geometry"]

            # Get median features for this district + month
            feat_row = district_features[
                (district_features["district_name"] == district) &
                (district_features["month"] == month)
            ]

            if feat_row.empty:
                feat_row = district_features[district_features["district_name"] == district]
                if feat_row.empty:
                    feat_row = district_features.iloc[[0]].copy()

            feat_row = feat_row.iloc[[0]]

            # Build feature vector
            X = feat_row[FEATURE_COLS[:-1]].copy()
            X["month"] = month

            # ── LIVE WEATHER (for display only, NOT injected into model) ──
            live_w = None
            if LIVE_WEATHER_ENABLED and district in district_centroids:
                coords = district_centroids[district]
                live_w = get_live_weather_features(
                    coords["lat"], coords["lon"], month=month, day_offset=day, state_key=state_key
                )
            # ─────────────────────────────────────────────────────────────

            X = X[FEATURE_COLS].fillna(X.median())

            # Predict risk using calibrated raw margins (stable gradient)
            base_score = calibrated_score_single(model, X)

            # Apply small weather-responsive adjustment based on live rain
            training_rain_7d = float(X["rain_7d_mm"].values[0]) if "rain_7d_mm" in X.columns else 70.0
            if live_w and training_rain_7d > 0:
                live_rain_7d = live_w.get("rain_7d_mm", training_rain_7d)
                rain_ratio = live_rain_7d / max(training_rain_7d, 1.0)
                adjustment = max(-0.20, min(0.30, (rain_ratio - 1.0) * 0.15))
            else:
                adjustment = 0.0

            risk_score = max(0.0, min(1.0, base_score + adjustment))
            risk_level = get_risk_level(risk_score)

            # Build factors dict — expose actual daily factors
            rain_1d   = live_w["rain_1d_mm"]   if live_w else feat_row["rain_1d_mm"].values[0]
            rain_3d   = live_w["rain_3d_mm"]   if live_w else feat_row["rain_3d_mm"].values[0]
            rain_7d   = live_w["rain_7d_mm"]   if live_w else feat_row["rain_7d_mm"].values[0]
            sm_surf   = live_w["sm_surface"]    if live_w else feat_row["sm_surface"].values[0]
            rain_daily= live_w["rain_daily_mean_mm"] if live_w else feat_row["rain_daily_mean_mm"].values[0]
            rain_anom = live_w["rain_anomaly"]  if live_w else feat_row["rain_anomaly"].values[0]
            elev      = feat_row['elevation'].values[0] if 'elevation' in feat_row else 50
            flow_acc  = feat_row['flow_acc'].values[0] if 'flow_acc' in feat_row else 100

            factors = {
                "rain_1d_mm": f"{rain_1d:.0f} mm",
                "rain_3d_mm": f"{rain_3d:.0f} mm",
                "rain_7d_mm": f"{rain_7d:.0f} mm",
                "rainfall_7d": f"{rain_7d:.0f} mm",
                "soil_moisture": f"{sm_surf*100:.0f}%",
                "sm_surface": f"{sm_surf*100:.0f}%",
                "rain_daily_mean_mm": f"{rain_daily:.1f} mm/day",
                "rain_anomaly": f"{rain_anom:+.1f}σ",
                "elevation": f"{elev:.0f} m",
                "flow_accumulation": f"{flow_acc:.0f}",
            }

            alert_message = get_alert_message(district, risk_level, {
                "rain_anomaly": f"{rain_anom:+.1f}",
                "sm_surface": f"{sm_surf*100:.0f}",
            })

            geojson_geom = json.loads(gpd.GeoSeries([geometry]).to_json())["features"][0]["geometry"]

            feature = {
                "type": "Feature",
                "geometry": geojson_geom,
                "properties": {
                    "id": f"district-{district.lower().replace(' ', '-')}",
                    "region": f"{state_label} - {district.title()}",
                    "state": state_key,
                    "district_name": district,
                    "model_type": "flood",
                    "risk_score": round(risk_score, 3),
                    "risk_level": risk_level,
                    "days_to_event": get_days_to_event(risk_level),
                    "data_source": "live" if live_w else "historical",
                    "alert_message": alert_message,
                    "factors": factors,
                    "day": day,
                    "day_offset": day,
                    "month": month,
                    "year": year,
                }
            }
            features_list.append(feature)

    # Sort by risk score descending
    features_list.sort(key=lambda f: f["properties"]["risk_score"], reverse=True)

    return jsonify({
        "type": "FeatureCollection",
        "features": features_list,
        "metadata": {
            "region": "all",
            "states": list(all_state_gdfs.keys()),
            "month": month,
            "day": day,
            "year": year,
            "model": "XGBoost Multistate v1",
            "districts": len(features_list),
            "weather_status": get_weather_system_status().get("status", "ok"),
            "weather_warning": get_weather_system_status().get("warning"),
        }
    })


@app.route("/api/districts", methods=["GET"])
def list_districts():
    """Return list of all districts with summary risk info."""
    month = request.args.get("month", type=int, default=7)
    month = max(5, min(10, month))

    districts = []
    for district in available_districts:
        feat_row = district_features[
            (district_features["district_name"] == district) &
            (district_features["month"] == month)
        ]
        if feat_row.empty:
            continue

        X = feat_row[FEATURE_COLS[:-1]].copy()
        X["month"] = month
        X = X[FEATURE_COLS].fillna(X.median())

        risk_score = calibrated_score_single(model, X)

        districts.append({
            "district_name": district,
            "risk_score": round(risk_score, 3),
            "risk_level": get_risk_level(risk_score),
        })

    districts.sort(key=lambda d: d["risk_score"], reverse=True)
    return jsonify({"districts": districts, "month": month})


@app.route("/api/risk-grid/flood/cells", methods=["GET"])
@app.route("/api/risk-grid/assam/cells", methods=["GET"])
@app.route("/api/risk-grid/<region>/cells", methods=["GET"])
def risk_grid_cells(region=None):
    """
    Return GeoJSON FeatureCollection with individual 5km grid cells.
    Query params:
      - month (int, 5-10): monsoon month (default: 7)
      - day (int, -7 to 15): day forecast or historical offset (default: 0)
      - min_risk (float): only return cells above this threshold (default: 0.0)
      - max_cells (int): cap response size (default: 3000)
    """
    month = request.args.get("month", type=int, default=7)
    day = request.args.get("day", type=int, default=0)
    month = max(5, min(10, month))
    day = max(-7, min(15, day))
    min_risk = request.args.get("min_risk", type=float, default=0.0)
    max_cells = request.args.get("max_cells", type=int, default=3000)

    # Optional bounding box filter (for searched region)
    minlon = request.args.get("minlon", type=float)
    maxlon = request.args.get("maxlon", type=float)
    minlat = request.args.get("minlat", type=float)
    maxlat = request.args.get("maxlat", type=float)

    # Get features for this month
    month_df = cell_month_features[cell_month_features["month"] == month].copy()
    if month_df.empty:
        return jsonify({"type": "FeatureCollection", "features": []})

    # Apply bbox filter if provided
    if all(v is not None for v in [minlon, maxlon, minlat, maxlat]):
        month_df = month_df[
            (month_df["wgs_lon"] >= minlon) & (month_df["wgs_lon"] <= maxlon) &
            (month_df["wgs_lat"] >= minlat) & (month_df["wgs_lat"] <= maxlat)
        ]

    if month_df.empty:
        return jsonify({"type": "FeatureCollection", "features": []})

    # ── LIVE WEATHER (for display & adjustment, NOT injected into model) ──
    _live_weather_per_district = {}
    if LIVE_WEATHER_ENABLED:
        for dist_name in month_df["district_name"].unique():
            if dist_name in district_centroids:
                coords = district_centroids[dist_name]
                state_key = DISTRICT_TO_STATE.get(dist_name, "assam")
                lw = get_live_weather_features(coords["lat"], coords["lon"], month=month, day_offset=day, state_key=state_key)
                if lw:
                    _live_weather_per_district[dist_name] = lw

    # Build feature matrix and predict in batch using TRAINING features
    X = month_df[_non_month_feats].copy()
    X["month"] = month
    X = X[FEATURE_COLS].fillna(X.median())
    base_scores = calibrated_scores_batch(model, X)

    # Apply per-district weather adjustment
    adjustments = np.zeros(len(month_df))
    for dist_name, lw in _live_weather_per_district.items():
        mask = (month_df["district_name"] == dist_name).values
        if mask.any():
            training_rain_7d = X.loc[mask, "rain_7d_mm"].median()
            live_rain_7d = lw.get("rain_7d_mm", training_rain_7d)
            rain_ratio = live_rain_7d / max(float(training_rain_7d), 1.0)
            adj = max(-0.20, min(0.30, (rain_ratio - 1.0) * 0.15))
            adjustments[mask] = adj

    month_df = month_df.copy()
    month_df["risk_score"] = np.clip(base_scores + adjustments, 0.0, 1.0)

    # Filter by min_risk
    if min_risk > 0:
        month_df = month_df[month_df["risk_score"] >= min_risk]

    # Sort by risk desc and cap
    month_df = month_df.sort_values("risk_score", ascending=False).head(max_cells)

    features_list = []
    for _, row in month_df.iterrows():
        key = (row["cell_lon"], row["cell_lat"])
        geom = cell_polygons.get(key)
        if geom is None:
            continue

        risk_score = float(row["risk_score"])
        risk_level = get_risk_level(risk_score)
        district = row.get("district_name", "Flood Zone")

        cell_id = f"cell-{str(row['wgs_lon']).replace('.','_')}-{str(row['wgs_lat']).replace('.','_')}"

        rain_1d   = row.get("rain_1d_mm", 0)
        rain_3d   = row.get("rain_3d_mm", 0)
        rain_7d   = row.get("rain_7d_mm", 0)
        sm_surf   = row.get("sm_surface", 0)
        rain_daily= row.get("rain_daily_mean_mm", 0)
        rain_anom = row.get("rain_anomaly", 0)

        state_key = DISTRICT_TO_STATE.get(district, "assam")
        state_label = state_key.replace("_", " ").title()

        features_list.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "id": cell_id,
                "region": f"{state_label} - {district.title()} Zone",
                "state": state_key,
                "district_name": district,
                "model_type": "flood",
                "risk_score": round(risk_score, 3),
                "risk_level": risk_level,
                "days_to_event": get_days_to_event(risk_level),
                "alert_message": get_alert_message(district, risk_level, {
                    "rain_anomaly": f"{rain_anom:+.1f}",
                    "sm_surface": f"{sm_surf*100:.0f}",
                }),
                "factors": {
                    "rain_1d_mm": f"{rain_1d:.0f} mm",
                    "rain_3d_mm": f"{rain_3d:.0f} mm",
                    "rain_7d_mm": f"{rain_7d:.0f} mm",
                    "rainfall_7d": f"{rain_7d:.0f} mm",
                    "soil_moisture": f"{sm_surf*100:.0f}%",
                    "sm_surface": f"{sm_surf*100:.0f}%",
                    "rain_daily_mean_mm": f"{rain_daily:.1f} mm/day",
                    "rain_anomaly": f"{rain_anom:+.1f}σ",
                    "elevation": f"{row.get('elevation', 0):.0f} m",
                    "flow_accumulation": f"{row.get('flow_acc', 0):.0f}",
                },
                "day": day,
                "day_offset": day,
                "month": month,
            }
        })

    return jsonify({
        "type": "FeatureCollection",
        "features": features_list,
        "metadata": {
            "region": "all",
            "mode": "cells",
            "month": month,
            "day": day,
            "cell_count": len(features_list),
            "cell_size_km": 5,
        }
    })


@app.route("/api/feature-importance", methods=["GET"])
def feature_importance():
    """Return feature importance directly from the deployed XGBoost model."""
    importance = model.feature_importances_
    feat_imp = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1])
    ]
    return jsonify({
        "feature_importance": feat_imp,
        "total_features": len(FEATURE_COLS),
        "model": "XGBoost Multistate v1"
    })


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  🌊 Project Hydra — Flood Risk API")
    print("  http://localhost:5001/api/risk-grid/assam")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5001, debug=True)
