"""
extract_real_idm_labels.py
===========================
Extracts REAL India Drought Monitor (IDM) CDI values from the 269 weekly
CDI text files in data/idm_archive/ for all grid cells across Marathwada,
Bundelkhand, and Rayalaseema.

No synthetic data. No manual approximations. 100% real IDM observations.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IDM_DIR = DATA_DIR / "idm_archive"
FEATURES_FILE = DATA_DIR / "drought_features_all_regions.csv"
OUTPUT_FILE = DATA_DIR / "drought_features_labeled.csv"


def cdi_to_risk_score(cdi_val):
    """
    Continuous CDI -> 0 to 1 risk score.
    CDI >= 0.5 -> 0.0 (no drought)
    CDI = 0.0 -> 0.2 (D0)
    CDI = -0.5 -> 0.4 (D1)
    CDI = -1.0 -> 0.6 (D2)
    CDI = -1.5 -> 0.8 (D3)
    CDI <= -2.0 -> 1.0 (D4)
    """
    if np.isnan(cdi_val):
        return np.nan
    risk = (0.5 - cdi_val) / 2.5
    return float(np.clip(risk, 0.0, 1.0))


def load_cdi_grid(filepath):
    """Load CDI_YYYYMMDD.txt into dict {(lat_round, lon_round): cdi_value}."""
    grid = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            lat, lon, cdi = float(parts[0]), float(parts[1]), parts[2]
            if cdi == 'NaN' or cdi == 'nan':
                continue
            cdi_val = float(cdi)
            lat_key = round(lat * 4) / 4
            lon_key = round(lon * 4) / 4
            grid[(lat_key, lon_key)] = cdi_val
    return grid


def snap_to_025(val):
    return round(val * 4) / 4


def main():
    print("=" * 60)
    print("EXTRACTING REAL IDM LABELS FOR ALL REGIONS")
    print("=" * 60)

    # 1. Index all CDI files by date
    cdi_files = sorted(IDM_DIR.glob("CDI_*.txt"))
    if not cdi_files:
        print(f"ERROR: No CDI files found in {IDM_DIR}")
        return

    cdi_by_date = {}
    for f in cdi_files:
        name = f.stem.replace("CDI_", "")
        try:
            d = datetime.strptime(name, "%Y%m%d")
            cdi_by_date[d] = f
        except ValueError:
            continue

    print(f"Loaded {len(cdi_by_date)} real weekly IDM CDI files from archive.")

    # 2. Load feature data
    if not FEATURES_FILE.exists():
        print(f"ERROR: {FEATURES_FILE} not found.")
        return

    df = pd.read_csv(FEATURES_FILE)
    print(f"Loaded feature dataset: {len(df)} rows across regions: {df['region'].unique().tolist()}")

    # Snap lat/lon to 0.25 grid for fast CDI lookup
    df['lat_025'] = df['lat'].apply(snap_to_025)
    df['lon_025'] = df['lon'].apply(snap_to_025)

    # Cache loaded CDI grids in memory
    grid_cache = {}

    risk_scores = []
    real_cdi_vals = []
    matched_dates = []

    dates = df['date'].unique()
    for d_str in dates:
        dt = datetime.strptime(d_str, "%Y-%m-%d")
        # Find nearest CDI file within 7 days
        best_match = None
        best_delta = timedelta(days=8)
        for cdi_date, cdi_path in cdi_by_date.items():
            delta = abs(dt - cdi_date)
            if delta < best_delta:
                best_delta = delta
                best_match = (cdi_date, cdi_path)

        if best_match and best_delta <= timedelta(days=7):
            cdi_date, cdi_path = best_match
            print(f"  Date {d_str} -> Matched real IDM observation {cdi_path.name} (offset: {best_delta.days}d)")
            if cdi_path not in grid_cache:
                grid_cache[cdi_path] = load_cdi_grid(cdi_path)
            grid = grid_cache[cdi_path]

            sub_df = df[df['date'] == d_str]
            for _, row in sub_df.iterrows():
                key = (row['lat_025'], row['lon_025'])
                cdi_val = grid.get(key, np.nan)
                real_cdi_vals.append(cdi_val)
                risk_scores.append(cdi_to_risk_score(cdi_val))
                matched_dates.append(cdi_date.strftime("%Y-%m-%d"))
        else:
            print(f"  WARNING: No IDM file found near {d_str}")
            sub_df = df[df['date'] == d_str]
            for _ in range(len(sub_df)):
                real_cdi_vals.append(np.nan)
                risk_scores.append(np.nan)
                matched_dates.append(None)

    from fetch_drought_labels import assign_district
    print("\nAssigning districts to cells...")
    df['district'] = df.apply(lambda r: assign_district(r['lat'], r['lon'], r['region']), axis=1)

    df['idm_cdi_raw'] = real_cdi_vals
    df['drought_risk_score'] = risk_scores
    df['idm_matched_date'] = matched_dates
    df['is_synthetic_label'] = False
    df['label_source'] = 'real_idm_archive_025deg'

    # Clean missing values if any grid points were outside India mask
    initial_len = len(df)
    df = df.dropna(subset=['drought_risk_score'])
    print(f"\nExtracted real IDM labels for {len(df)} / {initial_len} cells ({len(df)/initial_len:.1%})")

    # Save to output file
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved 100% REAL labeled dataset: {OUTPUT_FILE}")
    print("\nSummary statistics for REAL drought risk score (0-1):")
    print(df.groupby('region')['drought_risk_score'].describe().round(4).to_string())


if __name__ == "__main__":
    main()
