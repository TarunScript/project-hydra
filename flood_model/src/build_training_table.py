#!/usr/bin/env python3
"""
build_training_table.py — Task 3: Join Features + Labels (Fixed spatial join)

Uses actual GADM district polygon boundaries to assign each GEE grid cell
to its correct Assam district — replacing the broken bounding-box approach.

Fix: geopandas sjoin (point-in-polygon) on GADM district boundaries.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"

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


def load_district_boundaries():
    """Load GADM Assam district polygons."""
    geojson_path = RAW_DIR / "assam_districts.geojson"
    if not geojson_path.exists():
        print("  ✗ assam_districts.geojson not found in data/raw/")
        sys.exit(1)

    gdf = gpd.read_file(geojson_path)

    # Normalize district names to uppercase to match labels
    gdf["district_name"] = gdf["NAME_2"].str.strip().str.upper()

    # Fix known name mismatches between GADM and IFI-Impacts
    name_map = {
        "SIBSAGAR":           "SIVASAGAR",
        "NORTH CACHAR HILLS": "DIMA HASAO",
        "DIMAHASAO":          "DIMA HASAO",
        "KARBIANGLONG":       "KARBI ANGLONG",
        "KARBI ANGLONG WEST": "WEST KARBI ANGLONG",
        "KAMRUPMETROPOLITAN": "KAMRUP METROPOLITAN",
        "BISWANATH CHARIALI": "BISWANATH",
        "HOJAI":              "HOJAI",
        "MAJULI":             "MAJULI",
        "CHARAIDEO":          "CHARAIDEO",
        "SOUTH SALMARA":      "SOUTH SALMARA MANCACHAR",
    }
    gdf["district_name"] = gdf["district_name"].replace(name_map)

    print(f"  ✓ Loaded {len(gdf)} Assam district polygons")
    print(f"  Districts: {sorted(gdf['district_name'].tolist())}")
    return gdf[["district_name", "geometry"]]


def utm_to_latlon(cell_lon_m, cell_lat_m):
    """
    Convert UTM Zone 46N coordinates (meters) to WGS84 degrees.
    GEE coveringGrid returns centroids in the projection's native units.
    For EPSG:32646 (UTM 46N):
      - Central meridian = 93°E
      - False easting    = 500,000 m  ← MUST subtract before converting
      - False northing   = 0 m (northern hemisphere)
    """
    lat_deg = cell_lat_m / 110574.0
    lat_rad = np.radians(lat_deg)
    # Subtract false easting (500000 m) before dividing
    lon_deg = (cell_lon_m - 500000.0) / (111320.0 * np.cos(lat_rad)) + 93.0
    return lon_deg, lat_deg


def assign_districts_spatially(features_df, district_gdf):
    """
    Assign each GEE grid cell to its Assam district using point-in-polygon.
    Returns features_df with a new 'district_name' column.
    """
    print("  Converting UTM centroids to lat/lon...")
    lon_deg, lat_deg = utm_to_latlon(
        features_df["cell_lon"].values,
        features_df["cell_lat"].values,
    )

    print(f"  Lon range: [{lon_deg.min():.2f}, {lon_deg.max():.2f}]")
    print(f"  Lat range: [{lat_deg.min():.2f}, {lat_deg.max():.2f}]")

    # Build GeoDataFrame of grid cell centroids
    print("  Building cell GeoDataFrame...")
    cell_gdf = gpd.GeoDataFrame(
        {"cell_idx": np.arange(len(features_df))},
        geometry=gpd.points_from_xy(lon_deg, lat_deg),
        crs="EPSG:4326",
    )

    # Ensure district boundaries are in same CRS
    district_gdf = district_gdf.set_crs("EPSG:4326", allow_override=True)

    # Spatial join: point-in-polygon
    print("  Running spatial join (point-in-polygon)...")
    joined = gpd.sjoin(cell_gdf, district_gdf, how="left", predicate="within")

    # Check coverage
    matched = joined["district_name"].notna().sum()
    total = len(joined)
    print(f"  Matched: {matched:,}/{total:,} cells ({matched/total*100:.1f}%)")

    if matched < total * 0.5:
        print("  ⚠ Less than 50% matched — checking coordinate conversion...")
        # Sample some points for debugging
        sample = joined[joined["district_name"].isna()].head(5)
        sample_lon = lon_deg[sample["cell_idx"].values]
        sample_lat = lat_deg[sample["cell_idx"].values]
        print(f"  Unmatched cell coords: lon={sample_lon[:3]}, lat={sample_lat[:3]}")

    # Fill unmatched (cells outside all district polygons) with nearest district
    unmatched_mask = joined["district_name"].isna()
    if unmatched_mask.sum() > 0:
        print(f"  Finding nearest district for {unmatched_mask.sum():,} unmatched cells...")
        unmatched_gdf = cell_gdf[unmatched_mask].copy()
        # Use nearest join for cells just outside polygon boundaries
        nearest = gpd.sjoin_nearest(
            unmatched_gdf, district_gdf, how="left"
        )
        joined.loc[unmatched_mask, "district_name"] = nearest["district_name"].values

    features_df = features_df.copy()
    features_df["district_name"] = joined["district_name"].values

    # Report district distribution
    dist_counts = features_df["district_name"].value_counts()
    print(f"\n  Districts assigned: {features_df['district_name'].nunique()}")
    print(f"  Cells per district (top 10):")
    for dist, count in dist_counts.head(10).items():
        print(f"    {dist:40s}: {count:,}")

    return features_df


def fill_nulls(df):
    """Fill remaining nulls with column medians."""
    null_before = df.isnull().sum().sum()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    null_after = df.isnull().sum().sum()
    if null_before > null_after:
        print(f"  Filled {null_before - null_after:,} null values with column medians")
    return df


def build_training_table():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Task 3 (Fixed): Building Training Table                   ║")
    print("║  Using geopandas spatial join for district assignment      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load all data ──
    print("\n  Loading datasets...")

    features_path = FEATURES_DIR / "features.csv"
    if not features_path.exists():
        print("  ✗ features.csv not found — run pull_features.py first")
        sys.exit(1)
    features = pd.read_csv(features_path)
    print(f"  ✓ GEE features: {features.shape}")

    static_district = pd.read_csv(FEATURES_DIR / "district_static_features.csv")
    static_district["district_name"] = static_district["district_name"].str.upper()
    print(f"  ✓ District static: {static_district.shape}")

    labels = pd.read_csv(FEATURES_DIR / "labels.csv")
    labels["district_name"] = labels["district_name"].str.upper()
    labels["date"] = pd.to_datetime(labels["date"])
    print(f"  ✓ Labels: {labels.shape}")

    # ── Load district boundaries ──
    print("\n  Loading GADM district boundaries...")
    district_gdf = load_district_boundaries()

    # ── Spatial join: assign district to each grid cell ──
    print("\n  Assigning districts to grid cells (spatial join)...")
    features = assign_districts_spatially(features, district_gdf)

    # ── Aggregate labels to monthly level ──
    print("\n  Aggregating labels to monthly level...")
    labels["month"] = labels["date"].dt.month
    monthly_labels = labels.groupby(["district_name", "year", "month"]).agg(
        is_flood_any=("is_flood", "max"),
        flood_day_count=("is_flood", "sum"),
        flood_fraction=("is_flood", "mean"),
    ).reset_index()
    print(f"  Monthly labels: {monthly_labels.shape}")
    print(f"  Label districts: {sorted(monthly_labels['district_name'].unique())[:10]}...")

    # ── Merge with district static features ──
    print("\n  Merging with district static features...")
    features = features.merge(static_district, on="district_name", how="left")
    unmatched_static = features["dfsi_score"].isna().sum()
    if unmatched_static > 0:
        print(f"  ⚠ {unmatched_static:,} cells have no static features match")
    else:
        print(f"  ✓ All cells matched to district static features")

    # ── Merge with labels ──
    print("\n  Merging with flood labels...")
    training = features.merge(
        monthly_labels,
        on=["district_name", "year", "month"],
        how="left"
    )
    matched = training["is_flood_any"].notna().sum()
    match_rate = matched / len(training) * 100
    print(f"  Label match: {matched:,}/{len(training):,} ({match_rate:.1f}%)")

    if match_rate < 50:
        print("  ⚠ Low match rate! Checking district name alignment...")
        gee_districts = set(features["district_name"].dropna().str.upper().unique())
        label_districts = set(monthly_labels["district_name"].str.upper().unique())
        in_gee_not_labels = gee_districts - label_districts
        in_labels_not_gee = label_districts - gee_districts
        if in_gee_not_labels:
            print(f"  In GEE not in labels: {sorted(in_gee_not_labels)}")
        if in_labels_not_gee:
            print(f"  In labels not in GEE: {sorted(in_labels_not_gee)}")

    # Fill unmatched label rows with 0
    training["is_flood_any"] = training["is_flood_any"].fillna(0).astype(int)
    training["flood_day_count"] = training["flood_day_count"].fillna(0).astype(int)
    training["flood_fraction"] = training["flood_fraction"].fillna(0.0)

    # ── Fill feature nulls ──
    print("\n  Filling null values...")
    training = fill_nulls(training)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("TASK 3 VERIFICATION SUMMARY")
    print("=" * 60)

    flood_count = training["is_flood_any"].sum()
    no_flood_count = len(training) - flood_count

    print(f"\n  Shape: {training.shape}")
    print(f"  Districts in training: {training['district_name'].nunique()}")
    print(f"\n  Target variable:")
    print(f"    Flood (1):    {int(flood_count):,} ({flood_count/len(training)*100:.1f}%)")
    print(f"    No-flood (0): {int(no_flood_count):,} ({no_flood_count/len(training)*100:.1f}%)")
    print(f"    Ratio:        1:{no_flood_count/max(flood_count,1):.1f}")

    null_remaining = training.isnull().sum().sum()
    print(f"\n  Remaining nulls: {null_remaining} {'✓' if null_remaining == 0 else '⚠'}")

    # Save
    out_path = FEATURES_DIR / "training_table.csv"
    training.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved: {out_path} ({out_path.stat().st_size/1024/1024:.1f} MB)")
    print("\n  ✓ Task 3 complete.")
    return training


if __name__ == "__main__":
    build_training_table()
