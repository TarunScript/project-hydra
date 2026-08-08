"""
fetch_drought_labels.py
=======================
Label integration for the drought risk-index model.

Priority order:
  1. Load REAL IDM (India Drought Monitor) labels from data/labels/*.csv
     - District-level CDI (D0-D4) area percentages
     - Convert to weighted risk score (0-1) per district
  2. FALLBACK to synthetic labels if no IDM data exists (clearly warned)

District spatial join:
  Uses approximate bounding boxes for Marathwada districts derived from
  actual administrative boundaries. For production, use GADM shapefiles.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

DATA_DIR = 'data'
LABELS_DIR = os.path.join(DATA_DIR, 'labels')
FEATURES_FILE = os.path.join(DATA_DIR, 'drought_features_all_regions.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'drought_features_labeled.csv')

# CDI -> risk score mapping
CDI_MAPPING = {
    'No Drought': 0.0,
    'D0': 0.2,
    'D1': 0.4,
    'D2': 0.6,
    'D3': 0.8,
    'D4': 1.0
}

# Approximate bounding boxes for Marathwada districts
# Format: (lat_min, lat_max, lon_min, lon_max)
# Derived from GADM administrative boundaries
MARATHWADA_DISTRICT_BOUNDS = {
    'Aurangabad': (19.5, 20.4, 74.6, 75.9),
    'Beed':       (18.5, 19.5, 75.3, 76.5),
    'Osmanabad':  (17.6, 18.5, 75.5, 76.6),
    'Hingoli':    (19.3, 20.0, 76.7, 77.6),
    'Jalna':      (19.5, 20.2, 75.7, 76.6),
    'Latur':      (17.9, 18.8, 76.2, 77.0),
    'Nanded':     (18.5, 19.7, 77.0, 78.3),
    'Parbhani':   (18.8, 19.8, 76.2, 77.1),
}

# Bundelkhand & Rayalaseema district bounds (approximate)
BUNDELKHAND_DISTRICT_BOUNDS = {
    'Banda':      (25.0, 25.8, 80.0, 81.0),
    'Chitrakoot': (24.8, 25.5, 80.5, 81.5),
    'Hamirpur':   (25.5, 26.1, 79.5, 80.5),
    'Jalaun':     (25.8, 26.5, 79.0, 80.0),
    'Jhansi':     (25.0, 25.8, 78.5, 79.5),
    'Lalitpur':   (24.2, 25.2, 78.0, 79.0),
    'Mahoba':     (25.0, 25.5, 79.5, 80.5),
    'Chhatarpur': (24.2, 25.2, 79.0, 80.2),
    'Damoh':      (23.5, 24.5, 79.2, 80.2),
    'Datia':      (25.5, 26.2, 78.2, 79.0),
    'Panna':      (24.0, 25.0, 80.0, 81.0),
    'Sagar':      (23.5, 24.5, 78.5, 79.5),
    'Tikamgarh':  (24.5, 25.5, 78.5, 79.5),
}

RAYALASEEMA_DISTRICT_BOUNDS = {
    'Anantapur':  (14.4, 15.9, 76.7, 78.1),
    'Chittoor':   (12.6, 14.0, 78.5, 79.9),
    'Kadapa':     (14.2, 15.5, 78.0, 79.5),
    'Kurnool':    (15.0, 16.2, 77.0, 78.5),
}

REGION_DISTRICT_BOUNDS = {
    'marathwada':  MARATHWADA_DISTRICT_BOUNDS,
    'bundelkhand': BUNDELKHAND_DISTRICT_BOUNDS,
    'rayalaseema': RAYALASEEMA_DISTRICT_BOUNDS,
}


def assign_district(lat, lon, region):
    """
    Assign a district based on which bounding box the point falls into.
    If point is in overlap or no box, assign nearest district by centroid distance.
    """
    bounds = REGION_DISTRICT_BOUNDS.get(region.lower(), {})
    if not bounds:
        return 'Unknown'

    # Check bounding boxes
    matches = []
    for name, (lat_min, lat_max, lon_min, lon_max) in bounds.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            matches.append(name)

    if len(matches) == 1:
        return matches[0]

    # If multiple matches (overlap) or no match, use nearest centroid
    best_dist = float('inf')
    best_name = list(bounds.keys())[0]
    for name, (lat_min, lat_max, lon_min, lon_max) in bounds.items():
        clat = (lat_min + lat_max) / 2
        clon = (lon_min + lon_max) / 2
        d = (lat - clat)**2 + (lon - clon)**2
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def load_idm_labels():
    """
    Load all IDM CSV files from data/labels/.
    Expected columns: district, none_pct, d0_pct, d1_pct, d2_pct, d3_pct, d4_pct
    Returns: dict of {district_name: risk_score} or None if no data.
    """
    if not os.path.exists(LABELS_DIR):
        return None

    csv_files = glob.glob(os.path.join(LABELS_DIR, 'idm_*.csv'))
    if not csv_files:
        return None

    print(f"Found {len(csv_files)} IDM label file(s) in {LABELS_DIR}")
    all_idm = []
    for f in csv_files:
        df = pd.read_csv(f)
        all_idm.append(df)
        print(f"  Loaded {f}: {len(df)} districts")

    idm = pd.concat(all_idm, ignore_index=True)

    # Compute weighted risk score from area percentages
    # D0=0.2, D1=0.4, D2=0.6, D3=0.8, D4=1.0, each weighted by area %
    label_dict = {}
    for _, row in idm.iterrows():
        name = row['district'].strip()
        score = (
            row.get('d0_pct', 0) * 0.2 +
            row.get('d1_pct', 0) * 0.4 +
            row.get('d2_pct', 0) * 0.6 +
            row.get('d3_pct', 0) * 0.8 +
            row.get('d4_pct', 0) * 1.0
        ) / 100.0
        label_dict[name] = round(score, 4)
        # Also add common alternate names
        if name == 'Aurangabad':
            label_dict['Chhatrapati Sambhajinagar'] = label_dict[name]
        elif name == 'Osmanabad':
            label_dict['Dharashiv'] = label_dict[name]

    print("\nIDM district risk scores:")
    for k, v in sorted(label_dict.items()):
        print(f"  {k:<30} {v:.3f}")

    return label_dict


def generate_synthetic_labels(df):
    """
    FALLBACK: Generate synthetic CDI labels from feature proxies.
    Only used when NO real IDM data is available.
    """
    print("\n" + "="*60)
    print("WARNING: USING SYNTHETIC/PROXY LABELS FOR DROUGHT SEVERITY.")
    print("No IDM data found in data/labels/. Generating labels based on")
    print("rain_90d_deficit_mm and ndvi_anomaly.")
    print("="*60 + "\n")

    df['rain_90d_deficit_mm'] = df.get('rain_90d_deficit_mm', pd.Series(np.zeros(len(df)))).fillna(0)
    df['ndvi_anomaly'] = df.get('ndvi_anomaly', pd.Series(np.zeros(len(df)))).fillna(0)

    proxy_score = df['rain_90d_deficit_mm'] + (df['ndvi_anomaly'] * 100)
    conditions = [
        (proxy_score < -150),
        (proxy_score < -100),
        (proxy_score < -50),
        (proxy_score < -20),
        (proxy_score < 0)
    ]
    choices = [1.0, 0.8, 0.6, 0.4, 0.2]
    df['drought_risk_score'] = np.select(conditions, choices, default=0.0)
    df['is_synthetic_label'] = True
    df['label_source'] = 'synthetic_proxy'
    return df


def main():
    # -----------------------------------------------------------------------
    # Load feature data
    # -----------------------------------------------------------------------
    if os.path.exists(FEATURES_FILE):
        print(f"Loading features from {FEATURES_FILE}...")
        df = pd.read_csv(FEATURES_FILE)
    else:
        print(f"Combined file not found at {FEATURES_FILE}")
        print("Looking for individual feature CSVs...")
        csv_files = glob.glob(os.path.join(DATA_DIR, 'drought_features_*_20*.csv'))
        if not csv_files:
            test_file = os.path.join(DATA_DIR, 'drought_features_marathwada_test.csv')
            if os.path.exists(test_file):
                csv_files = [test_file]
            else:
                print("Error: No feature CSV files found. Run build_drought_features.py first.")
                sys.exit(1)

        print(f"Found {len(csv_files)} feature files, concatenating...")
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Combined: {len(df)} rows from {len(csv_files)} files")

    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")

    # -----------------------------------------------------------------------
    # Assign districts using bounding boxes (replaces pseudo-random hash)
    # -----------------------------------------------------------------------
    print("\nAssigning districts using bounding box spatial join...")
    if 'region' in df.columns and 'lat' in df.columns and 'lon' in df.columns:
        df['district'] = df.apply(
            lambda row: assign_district(row['lat'], row['lon'], row['region']),
            axis=1
        )
        print("District distribution:")
        print(df['district'].value_counts().to_string())
    else:
        print("Warning: Missing lat/lon/region columns. Cannot assign districts.")
        df['district'] = 'Unknown'

    # -----------------------------------------------------------------------
    # Load real IDM labels OR fall back to synthetic
    # -----------------------------------------------------------------------
    idm_labels = load_idm_labels()

    if idm_labels:
        print(f"\nUsing REAL IDM labels for {len(idm_labels)} districts")

        # Assign risk score based on district
        df['drought_risk_score'] = df['district'].map(idm_labels).fillna(0.0)
        df['is_synthetic_label'] = False
        df['label_source'] = 'india_drought_monitor'

        # Report unmapped districts
        unmapped = df[df['drought_risk_score'] == 0.0]['district'].unique()
        mapped = df[df['drought_risk_score'] > 0.0]['district'].unique()
        print(f"\n  Districts with IDM labels: {list(mapped)}")
        if len(unmapped) > 0:
            print(f"  Districts WITHOUT IDM labels (default=0.0): {list(unmapped)}")

        print(f"\nLabel distribution (drought_risk_score):")
        print(df['drought_risk_score'].value_counts().sort_index().to_string())
    else:
        df = generate_synthetic_labels(df)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    print(f"\nSaving labeled features to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*60)
    print("LABELING COMPLETE")
    print("="*60)
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Total rows: {len(df)}")
    print(f"  Label source: {'REAL IDM' if idm_labels else 'SYNTHETIC (proxy)'}")
    print(f"  Regions: {df['region'].unique().tolist() if 'region' in df.columns else 'N/A'}")
    print(f"  Dates: {df['date'].nunique() if 'date' in df.columns else 'N/A'} unique")
    print(f"  Districts: {df['district'].nunique()}")


if __name__ == "__main__":
    main()
