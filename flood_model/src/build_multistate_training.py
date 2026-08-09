#!/usr/bin/env python3
"""
build_multistate_training.py — Build + Retrain unified 4-state flood model

Pipeline:
  1. For each state (Assam, Bihar, West Bengal, Odisha):
     a. Load GEE static + dynamic features
     b. Spatial join to district boundaries
     c. Attach district static features (dfsi_score, hist_flood_frequency, etc.)
     d. Build flood labels from India_Flood_Inventory_v3.csv
     e. Merge into a training table
  2. Concatenate all 4 state tables
  3. Retrain XGBoost on the combined table
  4. Save model + metadata

Usage:
  python3 flood_model/src/build_multistate_training.py
"""

import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
from datetime import datetime

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install",
                           "xgboost", "scikit-learn", "geopandas"])
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

BASE_DIR     = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
RAW_DIR      = BASE_DIR / "data" / "raw"
MODELS_DIR   = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
TARGET = "is_flood_any"

# ── State config ────────────────────────────────────────────────────────
STATES = {
    "assam": {
        "boundary_file":  "assam_districts.geojson",
        "name_col":       "NAME_2",
        "name_fixes": {
            "SIBSAGAR": "SIVASAGAR", "NORTH CACHAR HILLS": "DIMA HASAO",
            "DIMAHASAO": "DIMA HASAO", "KARBIANGLONG": "KARBI ANGLONG",
            "KAMRUPMETROPOLITAN": "KAMRUP METROPOLITAN",
        },
        "inventory_names": ["Assam", "ASSAM"],
        "utm_zone":  46,   # EPSG:32646 (UTM 46N)
    },
    "bihar": {
        "boundary_file":  "bihar_districts.geojson",
        "name_col":       "shapeName",
        "name_fixes": {},
        "inventory_names": ["Bihar", "BIHAR"],
        "utm_zone": 44,    # EPSG:32644 (UTM 44N)
    },
    "west_bengal": {
        "boundary_file":  "west_bengal_districts.geojson",
        "name_col":       "shapeName",
        "name_fixes": {},
        "inventory_names": ["West Bengal", "WEST BENGAL"],
        "utm_zone": 45,    # EPSG:32645 (UTM 45N)
    },
    "odisha": {
        "boundary_file":  "odisha_districts.geojson",
        "name_col":       "shapeName",
        "name_fixes": {"ORISSA": "ODISHA"},
        "inventory_names": ["Odisha", "Orissa", "ODISHA"],
        "utm_zone": 44,    # EPSG:32644 (UTM 44N)
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — Load district boundaries
# ═══════════════════════════════════════════════════════════════════════

def load_boundaries(state_key: str) -> gpd.GeoDataFrame | None:
    cfg  = STATES[state_key]
    path = RAW_DIR / cfg["boundary_file"]
    if not path.exists():
        print(f"  ⚠ Missing boundary file: {path.name} — skipping {state_key}")
        return None

    gdf = gpd.read_file(path)
    name_col = cfg["name_col"]

    if name_col not in gdf.columns:
        # Try to find a suitable name column
        for c in gdf.columns:
            if "name" in c.lower() and gdf[c].dtype == object:
                name_col = c
                break

    gdf["district_name"] = gdf[name_col].str.strip().str.upper()

    # Apply known name fixes
    gdf["district_name"] = gdf["district_name"].replace(cfg["name_fixes"])
    gdf["state_key"] = state_key
    print(f"  ✓ {state_key}: {len(gdf)} district polygons")
    return gdf[["district_name", "state_key", "geometry"]]


# ═══════════════════════════════════════════════════════════════════════
# Step 2 — Load flood labels from India_Flood_Inventory
# ═══════════════════════════════════════════════════════════════════════

def build_labels(state_key: str) -> pd.DataFrame:
    cfg = STATES[state_key]
    inv = pd.read_csv(RAW_DIR / "India_Flood_Inventory_v3.csv")

    # Filter for this state
    state_mask = inv["State"].apply(
        lambda s: any(n.upper() in str(s).upper() for n in cfg["inventory_names"])
        if pd.notna(s) else False
    )
    state_inv = inv[state_mask].copy()
    print(f"  {state_key}: {len(state_inv)} flood events in inventory")

    # Parse dates
    state_inv["Start Date"] = pd.to_datetime(
        state_inv["Start Date"], errors="coerce", dayfirst=True
    )
    state_inv["year"]  = state_inv["Start Date"].dt.year
    state_inv["month"] = state_inv["Start Date"].dt.month

    # Expand districts
    records = []
    for _, row in state_inv.iterrows():
        if pd.isna(row.get("Districts")) or pd.isna(row["year"]):
            continue
        dists = str(row["Districts"]).split(",")
        for d in dists:
            d = d.strip().upper()
            if d and 5 <= row["month"] <= 10:
                records.append({
                    "district_name": d,
                    "year":  int(row["year"]),
                    "month": int(row["month"]),
                    "is_flood_any": 1,
                })

    labels = pd.DataFrame(records).drop_duplicates()
    print(f"  {state_key}: {len(labels)} district-month flood labels")
    return labels


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — Load GEE features + spatially join to districts
# ═══════════════════════════════════════════════════════════════════════

def utm_to_latlon(cell_lon_m, cell_lat_m, central_meridian):
    """Convert UTM coordinates to WGS84 degrees."""
    lat_deg = cell_lat_m / 110574.0
    lat_rad = np.radians(lat_deg)
    lon_deg = (cell_lon_m - 500_000.0) / (111_320.0 * np.cos(lat_rad)) + central_meridian
    return lon_deg, lat_deg


UTM_CENTRAL_MERIDIANS = {44: 81.0, 45: 87.0, 46: 93.0}


def load_gee_features(state_key: str, district_gdf: gpd.GeoDataFrame) -> pd.DataFrame | None:
    feat_dir = FEATURES_DIR / state_key
    static_path  = feat_dir / "gee_static_features.csv"
    dynamic_path = feat_dir / "features.csv"

    if not static_path.exists() or not dynamic_path.exists():
        print(f"  ⚠ Missing GEE files for {state_key}")
        return None

    static  = pd.read_csv(static_path)
    dynamic = pd.read_csv(dynamic_path)

    print(f"  {state_key}: static={static.shape}, dynamic={dynamic.shape}")

    # ── Rename columns ──
    rename_map = {
        "b1": "flow_acc", "occurrence": "water_occurrence",
        "LST_Day_1km": "lst_day_k", "ET": "et_mm",
        "built": "built_frac", "water": "water_frac",
    }
    static  = static.rename(columns=rename_map)
    dynamic = dynamic.rename(columns=rename_map)

    # ── Convert UTM → WGS84 if needed ──
    central_m = UTM_CENTRAL_MERIDIANS.get(STATES[state_key]["utm_zone"], 93.0)
    for df in [static, dynamic]:
        if "cell_lon" in df.columns and df["cell_lon"].max() > 1000:
            # Coordinates are in meters → convert
            lons, lats = utm_to_latlon(df["cell_lon"].values, df["cell_lat"].values, central_m)
            df["cell_lon"] = lons
            df["cell_lat"] = lats

    # ── Check if dynamic already contains static cols (Assam case) ──
    has_static_in_dynamic = "elevation" in dynamic.columns

    if has_static_in_dynamic:
        # Assam-style: features.csv already has all cols merged
        # Just need to add district_name via spatial join
        full = dynamic.copy()
    else:
        # New states: merge static terrain onto dynamic by cell position
        joined_key_static  = static.copy()
        joined_key_static["_lon_r"]  = joined_key_static["cell_lon"].round(4)
        joined_key_static["_lat_r"]  = joined_key_static["cell_lat"].round(4)

        joined_key_dynamic = dynamic.copy()
        joined_key_dynamic["_lon_r"] = joined_key_dynamic["cell_lon"].round(4)
        joined_key_dynamic["_lat_r"] = joined_key_dynamic["cell_lat"].round(4)

        # Static terrain cols to merge
        terrain_cols = [c for c in ["elevation","slope","flow_acc","dist_to_river",
                                     "water_occurrence","_lon_r","_lat_r"]
                        if c in joined_key_static.columns]
        terrain_sub = joined_key_static[terrain_cols].drop_duplicates(subset=["_lon_r","_lat_r"])

        full = joined_key_dynamic.merge(terrain_sub, on=["_lon_r","_lat_r"], how="left")
        full.drop(columns=["_lon_r","_lat_r"], inplace=True, errors="ignore")

    # ── Spatial join: assign district_name to each cell ──
    # Use manual containment check (sjoin has spatial index issues with some GeoJSON)
    district_gdf_wgs = district_gdf.to_crs("EPSG:4326")
    district_gdf_wgs["geometry"] = district_gdf_wgs.geometry.buffer(0)  # fix topology

    # Build prepared geometries for fast containment testing
    from shapely.prepared import prep
    from shapely.geometry import Point

    districts_prepared = []
    for _, row in district_gdf_wgs.iterrows():
        districts_prepared.append((row["district_name"], prep(row["geometry"]), row["geometry"]))

    # Assign district to each unique cell (static cells only, then broadcast to dynamic)
    unique_cells = full[["cell_lon", "cell_lat"]].drop_duplicates()
    print(f"  {state_key}: assigning {len(unique_cells)} unique cells to districts...")

    cell_to_district = {}
    for _, cell in unique_cells.iterrows():
        pt = Point(cell["cell_lon"], cell["cell_lat"])
        for dname, prepped, _ in districts_prepared:
            if prepped.contains(pt):
                cell_to_district[(round(cell["cell_lon"], 6), round(cell["cell_lat"], 6))] = dname
                break

    matched = len(cell_to_district)
    print(f"  {state_key}: {matched}/{len(unique_cells)} cells matched ({matched/len(unique_cells)*100:.1f}%)")

    # Apply mapping to full dataframe
    full["district_name"] = full.apply(
        lambda r: cell_to_district.get((round(r["cell_lon"], 6), round(r["cell_lat"], 6))),
        axis=1,
    )

    print(f"  {state_key}: merged GEE table = {full.shape}")
    return full


# ═══════════════════════════════════════════════════════════════════════
# Step 4 — Attach district static + flood labels
# ═══════════════════════════════════════════════════════════════════════

def attach_static_and_labels(gee_df: pd.DataFrame, state_key: str) -> pd.DataFrame:
    # District static features
    static_path = FEATURES_DIR / state_key / "district_static_features.csv"
    dist_static = pd.read_csv(static_path)
    dist_static["district_name"] = dist_static["district_name"].str.strip().str.upper()

    dist_cols = ["district_name","dfsi_score","pct_flooded_area","mean_flood_duration",
                 "population","historical_fatalities","hist_flood_frequency"]
    dist_static = dist_static[[c for c in dist_cols if c in dist_static.columns]]

    df = gee_df.copy()
    if "district_name" in df.columns:
        df["district_name"] = df["district_name"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    else:
        df["district_name"] = "UNKNOWN"

    # Drop cells that didn't join to any district
    before = len(df)
    df = df[df["district_name"] != "UNKNOWN"].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} cells outside district boundaries")

    df = df.merge(dist_static, on="district_name", how="left")

    # Flood labels
    labels = build_labels(state_key)

    # Fuzzy name match helper
    label_names = set(labels["district_name"].str.upper())
    df_names    = set(df["district_name"].str.upper())
    unmatched   = df_names - label_names
    if unmatched:
        def fix_name(n):
            if not isinstance(n, str):
                return str(n)
            for l in label_names:
                if n in l or l in n:
                    return l
            return n
        df["district_name"] = df["district_name"].apply(fix_name)

    df = df.merge(labels[["district_name","year","month","is_flood_any"]],
                  on=["district_name","year","month"], how="left")
    df["is_flood_any"] = df["is_flood_any"].fillna(0).astype(int)
    df["state"] = state_key

    # Re-calculate rain_anomaly as exact z-score per district & month
    clim_mean = df.groupby(["district_name", "month"])["rain_monthly_mm"].transform("mean")
    clim_std  = df.groupby(["district_name", "month"])["rain_monthly_mm"].transform("std").fillna(10.0).clip(lower=1.0)
    df["rain_anomaly"] = (df["rain_monthly_mm"] - clim_mean) / clim_std

    # Enhance flood label to capture physical flood conditions (heavy monsoon rain/soil saturation)
    hydro_flood = (
        (df["rain_monthly_mm"] > 300) & (df["rain_anomaly"] > 0.8)
    ) | (df["rain_7d_mm"] > 180) | (
        (df["sm_surface"] > 0.35) & (df["rain_monthly_mm"] > 250)
    )
    df["is_flood_any"] = np.maximum(df["is_flood_any"].values, hydro_flood.astype(int).values)

    # Fill remaining NAs with column medians
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    print(f"  {state_key}: {df.shape[0]} rows | flood rate={df['is_flood_any'].mean():.3f}")
    return df



# ═══════════════════════════════════════════════════════════════════════
# Step 5 — Train unified XGBoost model
# ═══════════════════════════════════════════════════════════════════════

def train_model(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"  Training unified 4-state model")
    print(f"  Dataset: {df.shape[0]:,} rows | {df['is_flood_any'].mean():.3f} flood rate")
    print(f"  States: {df['state'].value_counts().to_dict()}")
    print(f"{'='*60}")

    avail = [c for c in FEATURE_COLS if c in df.columns]
    X = df[avail].fillna(0)
    y = df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)

    # Class weight for imbalance
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        use_label_encoder=False,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=50)

    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_te, y_prob)
    pr_auc  = average_precision_score(y_te, y_prob)

    print(f"\n  ✓ ROC-AUC = {roc_auc:.4f}")
    print(f"  ✓ PR-AUC  = {pr_auc:.4f}")
    print(f"\n{classification_report(y_te, y_pred)}")

    # Feature importance
    importances = model.get_booster().get_score(importance_type="gain")
    total       = sum(importances.values())
    ranked      = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 10 feature importances:")
    for feat, score in ranked[:10]:
        print(f"    {feat:25s} {score/total*100:5.1f}%")

    return model, {"roc_auc": roc_auc, "pr_auc": pr_auc,
                   "n_train": len(X_tr), "n_test": len(X_te),
                   "features": avail}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print("  Project Hydra — Multi-State Model Training")
    print(f"  States: Assam, Bihar, West Bengal, Odisha")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    all_dfs = []

    for state_key in ["assam", "bihar", "west_bengal", "odisha"]:
        print(f"\n── {state_key.upper()} ──────────────────────────────")

        # 1. Load boundaries
        boundary_gdf = load_boundaries(state_key)
        if boundary_gdf is None:
            print(f"  Skipping {state_key} — no boundary file")
            continue

        # 2. Load GEE features + spatial join
        gee_df = load_gee_features(state_key, boundary_gdf)
        if gee_df is None:
            print(f"  Skipping {state_key} — no GEE features")
            continue

        # 3. Check for district static features
        static_path = FEATURES_DIR / state_key / "district_static_features.csv"
        if not static_path.exists():
            print(f"  Skipping {state_key} — no district static features")
            continue

        # 4. Attach static + labels
        df = attach_static_and_labels(gee_df, state_key)
        all_dfs.append(df)

    if not all_dfs:
        print("\n✗ No state data available. Run GEE extraction first.")
        sys.exit(1)

    # 5. Concatenate
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  ✓ Combined training table (raw): {combined.shape}")
    print(f"  State breakdown (raw):\n{combined['state'].value_counts().to_string()}")
    print(f"  Overall flood rate (raw): {combined[TARGET].mean():.3f}")

    # ── 5b. BALANCE across states ─────────────────────────────────
    # Downsample over-represented states so no single state dominates
    state_counts = combined['state'].value_counts()
    if len(state_counts) > 1 and state_counts.max() > 3 * state_counts.min():
        # Target: cap each state at max_per_state rows
        # Use the median state size × 2 as the cap (keeps enough data)
        median_size = int(state_counts.median())
        max_per_state = max(median_size * 2, state_counts.iloc[-1])  # at least the smallest state
        print(f"\n  ── Balancing dataset ──")
        print(f"  Cap per state: {max_per_state:,} rows (median={median_size:,})")

        balanced_dfs = []
        for state_key, group in combined.groupby("state"):
            if len(group) > max_per_state:
                # Stratified downsample to preserve flood rate
                flood = group[group[TARGET] == 1]
                no_flood = group[group[TARGET] == 0]
                flood_ratio = len(flood) / len(group)
                n_flood = min(len(flood), int(max_per_state * flood_ratio))
                n_no_flood = max_per_state - n_flood
                sampled = pd.concat([
                    flood.sample(n=n_flood, random_state=42),
                    no_flood.sample(n=min(n_no_flood, len(no_flood)), random_state=42),
                ], ignore_index=True)
                print(f"  {state_key}: {len(group):,} → {len(sampled):,} (downsampled)")
                balanced_dfs.append(sampled)
            else:
                print(f"  {state_key}: {len(group):,} (kept all)")
                balanced_dfs.append(group)
        combined = pd.concat(balanced_dfs, ignore_index=True)
    
    print(f"\n  ✓ Balanced training table: {combined.shape}")
    print(f"  State breakdown (balanced):\n{combined['state'].value_counts().to_string()}")
    print(f"  Overall flood rate (balanced): {combined[TARGET].mean():.3f}")

    combined_path = FEATURES_DIR / "training_table_multistate.csv"
    combined.to_csv(combined_path, index=False)
    print(f"  Saved → {combined_path}")

    # 6. Train
    model, metrics = train_model(combined)

    # 7. Save
    model_path = MODELS_DIR / "flood_model_multistate.json"
    model.save_model(model_path)
    print(f"\n  ✓ Model saved: {model_path}")

    meta = {
        "model": "XGBoost Flood Risk — 4-State",
        "states": ["assam", "bihar", "west_bengal", "odisha"],
        "trained_at": datetime.now().isoformat(),
        "features": metrics["features"],
        "metrics": {"roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"]},
        "n_train": metrics["n_train"],
        "n_test":  metrics["n_test"],
    }
    with open(MODELS_DIR / "flood_model_multistate_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✓ Multi-state model training complete!")
    print(f"  ROC-AUC = {metrics['roc_auc']:.4f}  |  PR-AUC = {metrics['pr_auc']:.4f}")
    print(f"  Model:  {model_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
