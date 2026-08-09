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
    if score >= 0.75: return "severe"
    if score >= 0.50: return "high"
    if score >= 0.25: return "moderate"
    return "low"

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

# Load trained model
model = xgb.XGBClassifier()
model.load_model(str(MODELS_DIR / "flood_model.json"))
print(f"  ✓ Model loaded")

# Load training table (has all features + district assignments)
training_table = pd.read_csv(FEATURES_DIR / "training_table.csv")
print(f"  ✓ Training table: {training_table.shape}")

# Load district boundaries for GeoJSON polygons
import geopandas as gpd
district_gdf = gpd.read_file(RAW_DIR / "assam_districts.geojson")
district_gdf["district_name"] = district_gdf["NAME_2"].str.strip().str.upper()
# Apply same name fixes as build_training_table.py
name_map = {
    "SIBSAGAR":           "SIVASAGAR",
    "NORTH CACHAR HILLS": "DIMA HASAO",
    "DIMAHASAO":          "DIMA HASAO",
    "KARBIANGLONG":       "KARBI ANGLONG",
    "KARBI ANGLONG WEST": "WEST KARBI ANGLONG",
    "KAMRUPMETROPOLITAN": "KAMRUP METROPOLITAN",
}
district_gdf["district_name"] = district_gdf["district_name"].replace(name_map)
print(f"  ✓ District boundaries: {len(district_gdf)} polygons")

# Pre-compute district-level feature medians per month (for fast inference)
district_features = training_table.groupby(["district_name", "month"])[FEATURE_COLS[:-1]].median().reset_index()
district_features["month"] = district_features["month"].astype(int)
print(f"  ✓ District feature profiles: {district_features.shape}")

# Available districts
available_districts = sorted(training_table["district_name"].unique())
print(f"  ✓ Districts: {available_districts}")

# ── Pre-compute per-CELL risk scores (one unique cell per lat/lon) ──
print("  Pre-computing per-cell risk scores...")
from pyproj import Transformer

# UTM 46N → WGS84 transformer
_transformer = Transformer.from_crs("EPSG:32646", "EPSG:4326", always_xy=True)

def utm_cells_to_geojson_polygons(cell_lon_m, cell_lat_m, half=2500):
    """Convert UTM cell centres to WGS84 bounding box polygons."""
    polys = []
    corners_utm = [
        (cell_lon_m - half, cell_lat_m - half),
        (cell_lon_m + half, cell_lat_m - half),
        (cell_lon_m + half, cell_lat_m + half),
        (cell_lon_m - half, cell_lat_m + half),
        (cell_lon_m - half, cell_lat_m - half),  # close ring
    ]
    lons, lats = _transformer.transform(
        [c[0] for c in corners_utm],
        [c[1] for c in corners_utm],
    )
    ring = [[round(lon, 5), round(lat, 5)] for lon, lat in zip(lons, lats)]
    return {"type": "Polygon", "coordinates": [ring]}

# Get unique cells (unique lat/lon in the static features layer)
gee_static = pd.read_csv(FEATURES_DIR / "gee_static_features.csv")
cell_coords = gee_static[["cell_lon", "cell_lat"]].drop_duplicates().reset_index(drop=True)

# Pre-compute cell polygons (done once at startup)
print(f"    Building {len(cell_coords):,} cell polygons...")
cell_polygons = {}  # key: (cell_lon, cell_lat) → geojson geometry
for _, row in cell_coords.iterrows():
    key = (row["cell_lon"], row["cell_lat"])
    cell_polygons[key] = utm_cells_to_geojson_polygons(row["cell_lon"], row["cell_lat"])

# Pre-compute per-cell features per month (sample 1 row per cell per month)
print("    Sampling per-cell features per month...")
_non_month_feats = [c for c in FEATURE_COLS if c != "month"]

# District-level static features that are UNIFORM per district → they dominate unfairly
# Override these with Assam-wide medians so only cell-level terrain/climate drives variation
DISTRICT_STATIC_FEATS = ["dfsi_score", "pct_flooded_area", "mean_flood_duration",
                          "population", "historical_fatalities", "hist_flood_frequency"]
ASSAM_STATIC_MEANS = training_table[DISTRICT_STATIC_FEATS].median()

cell_month_features = (
    training_table
    .groupby(["cell_lon", "cell_lat", "month"])[_non_month_feats]
    .median()
    .reset_index()
)

# Replace district-level features with Assam-wide medians so every cell gets the same
# baseline district risk — variation then comes purely from terrain/climate per cell
for feat in DISTRICT_STATIC_FEATS:
    if feat in cell_month_features.columns:
        cell_month_features[feat] = ASSAM_STATIC_MEANS[feat]

# Add WGS84 centroid coords for bbox filtering
_lons, _lats = _transformer.transform(
    cell_month_features["cell_lon"].values,
    cell_month_features["cell_lat"].values,
)
cell_month_features["wgs_lon"] = _lons.round(5)
cell_month_features["wgs_lat"] = _lats.round(5)

print(f"  ✓ Cell-month feature table: {cell_month_features.shape}")





# ══════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════

app = Flask(__name__)
CORS(app)  # Allow frontend on different port


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "XGBoost Flood Risk v1",
        "districts": len(available_districts),
        "features": len(FEATURE_COLS),
        "training_rows": len(training_table),
    })


@app.route("/api/risk-grid/assam", methods=["GET"])
def risk_grid_assam():
    """
    Return GeoJSON FeatureCollection with risk score per Assam district.
    Query params:
      - month (int, 5-10): monsoon month to predict for (default: current or 7)
      - year (int): year context (default: 2023)
    """
    # Parse params
    now = datetime.now()
    month = request.args.get("month", type=int, default=max(5, min(10, now.month)))
    year = request.args.get("year", type=int, default=2023)

    # Clamp month to monsoon range
    month = max(5, min(10, month))

    features_list = []

    for _, row in district_gdf.iterrows():
        district = row["district_name"]
        geometry = row["geometry"]

        # Get median features for this district + month
        feat_row = district_features[
            (district_features["district_name"] == district) &
            (district_features["month"] == month)
        ]

        if feat_row.empty:
            # Try nearest month
            feat_row = district_features[district_features["district_name"] == district]
            if feat_row.empty:
                continue
            feat_row = feat_row.iloc[[0]]

        # Build feature vector
        X = feat_row[FEATURE_COLS[:-1]].copy()
        X["month"] = month
        X = X[FEATURE_COLS].fillna(X.median())

        # Predict risk
        risk_score = float(model.predict_proba(X)[:, 1][0])
        risk_level = get_risk_level(risk_score)

        # Build factors dict for the detail panel
        factors = {
            "rainfall_7d": f"{feat_row['rain_7d_mm'].values[0]:.0f} mm",
            "soil_moisture": f"{feat_row['sm_surface'].values[0]*100:.0f}%",
            "rain_anomaly": f"{feat_row['rain_anomaly'].values[0]:+.1f}σ",
            "elevation": f"{feat_row['elevation'].values[0]:.0f} m",
            "flow_accumulation": f"{feat_row['flow_acc'].values[0]:.0f}",
        }

        alert_message = get_alert_message(district, risk_level, {
            "rain_anomaly": f"{feat_row['rain_anomaly'].values[0]:+.1f}",
            "sm_surface": f"{feat_row['sm_surface'].values[0]*100:.0f}",
        })

        # Convert geometry to GeoJSON
        geojson_geom = json.loads(gpd.GeoSeries([geometry]).to_json())["features"][0]["geometry"]

        feature = {
            "type": "Feature",
            "geometry": geojson_geom,
            "properties": {
                "id": f"district-{district.lower().replace(' ', '-')}",
                "region": f"Assam - {district.title()}",
                "district_name": district,
                "model_type": "flood",
                "risk_score": round(risk_score, 3),
                "risk_level": risk_level,
                "days_to_event": get_days_to_event(risk_level),
                "alert_message": alert_message,
                "factors": factors,
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
            "region": "assam",
            "month": month,
            "year": year,
            "model": "XGBoost v1",
            "districts": len(features_list),
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

        risk_score = float(model.predict_proba(X)[:, 1][0])

        districts.append({
            "district_name": district,
            "risk_score": round(risk_score, 3),
            "risk_level": get_risk_level(risk_score),
        })

    districts.sort(key=lambda d: d["risk_score"], reverse=True)
    return jsonify({"districts": districts, "month": month})


@app.route("/api/risk-grid/assam/cells", methods=["GET"])
def risk_grid_cells():
    """
    Return GeoJSON FeatureCollection with individual 5km grid cells.
    Each cell has its own XGBoost risk score — much more precise than district view.
    Query params:
      - month (int, 5-10): monsoon month (default: 7)
      - min_risk (float): only return cells above this threshold (default: 0.25)
      - max_cells (int): cap response size (default: 3000)
      - minlon, maxlon, minlat, maxlat: WGS84 bounding box (optional — limits to searched region)
    """
    month = request.args.get("month", type=int, default=7)
    month = max(5, min(10, month))
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

    # Build feature matrix and predict in batch (fast vectorised XGBoost call)
    X = month_df[_non_month_feats].copy()
    X["month"] = month
    X = X[FEATURE_COLS].fillna(X.median())
    risk_scores = model.predict_proba(X)[:, 1]
    month_df = month_df.copy()
    month_df["risk_score"] = risk_scores

    # Filter by min_risk
    if min_risk > 0:
        month_df = month_df[month_df["risk_score"] >= min_risk]

    # Sort by risk desc and cap
    month_df = month_df.sort_values("risk_score", ascending=False).head(max_cells)

    # Build district lookup from a fast dict
    cell_district_map = (
        training_table[["cell_lon", "cell_lat", "district_name"]]
        .drop_duplicates(subset=["cell_lon", "cell_lat"])
        .set_index(["cell_lon", "cell_lat"])["district_name"]
        .to_dict()
    )

    features_list = []
    for _, row in month_df.iterrows():
        key = (row["cell_lon"], row["cell_lat"])
        geom = cell_polygons.get(key)
        if geom is None:
            continue

        risk_score = float(row["risk_score"])
        risk_level = get_risk_level(risk_score)
        district = cell_district_map.get(key, "Assam Zone")

        features_list.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "id": f"cell-{int(row['cell_lon'])}-{int(row['cell_lat'])}",
                "region": f"{district.title()} Zone",
                "district_name": district,
                "model_type": "flood",
                "risk_score": round(risk_score, 3),
                "risk_level": risk_level,
                "days_to_event": get_days_to_event(risk_level),
                "alert_message": get_alert_message(district, risk_level, {
                    "rain_anomaly": f"{row.get('rain_anomaly', 0):+.1f}",
                    "sm_surface": f"{row.get('sm_surface', 0)*100:.0f}",
                }),
                "factors": {
                    "rainfall_7d": f"{row.get('rain_7d_mm', 0):.0f} mm",
                    "soil_moisture": f"{row.get('sm_surface', 0)*100:.0f}%",
                    "rain_anomaly": f"{row.get('rain_anomaly', 0):+.1f}σ",
                    "elevation": f"{row.get('elevation', 0):.0f} m",
                    "flow_accum": f"{row.get('flow_acc', 0):.0f}",
                },
                "month": month,
            }
        })

    return jsonify({
        "type": "FeatureCollection",
        "features": features_list,
        "metadata": {
            "region": "assam",
            "mode": "cells",
            "month": month,
            "cell_count": len(features_list),
            "cell_size_km": 5,
        }
    })


@app.route("/api/feature-importance", methods=["GET"])
def feature_importance():
    """Return feature importance from the trained model."""
    importance = model.feature_importances_
    feat_imp = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1])
    ]
    return jsonify({"feature_importance": feat_imp})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  🌊 Project Hydra — Flood Risk API")
    print("  http://localhost:5001/api/risk-grid/assam")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5001, debug=True)
