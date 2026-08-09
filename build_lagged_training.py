"""
build_lagged_training.py
========================
Constructs lagged training pairs for lead-time drought prediction:
  features(week W) -> CDI(W+7 days) and CDI(W+15 days)

Uses:
  - GEE feature CSVs from build_drought_features.py (per region/date)
  - IDM CDI archive from download_idm_archive.py (265 weekly grid files)

OUTPUT:
  data/training_lead7d.csv   — features(W) paired with CDI 7 days later
  data/training_lead15d.csv  — features(W) paired with CDI 15 days later

CRITICAL: The CDI labels are on a 0.25-degree grid (~28km).
The GEE features are on a ~5km grid. We join by nearest 0.25-degree grid point.
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
OUTPUT_LEAD7 = DATA_DIR / "training_lead7d.csv"
OUTPUT_LEAD15 = DATA_DIR / "training_lead15d.csv"

# CDI value -> drought class mapping (for reporting)
def cdi_to_class(cdi_val):
    """Map continuous CDI value to drought class."""
    if np.isnan(cdi_val):
        return "Unknown"
    if cdi_val >= 0.5:
        return "No Drought"
    if cdi_val >= -0.5:
        return "D0"  # Abnormally dry
    if cdi_val >= -1.0:
        return "D1"  # Moderate
    if cdi_val >= -1.5:
        return "D2"  # Severe
    if cdi_val >= -2.0:
        return "D3"  # Extreme
    return "D4"  # Exceptional


def cdi_to_risk_score(cdi_val):
    """
    Map continuous CDI value to 0-1 risk score.
    More negative CDI = higher risk.
    CDI >= 0.5 -> risk 0.0 (no drought)
    CDI <= -2.0 -> risk 1.0 (exceptional drought)
    """
    if np.isnan(cdi_val):
        return np.nan
    # Linear mapping: 0.5 -> 0.0, -2.0 -> 1.0
    risk = (0.5 - cdi_val) / 2.5
    return max(0.0, min(1.0, risk))


def load_cdi_grid(filepath):
    """
    Load one CDI .txt file (lat lon cdi_value, space-separated).
    Returns dict: {(lat_round, lon_round): cdi_value}
    """
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
            # Round to 0.25-degree grid for lookup
            lat_key = round(lat * 4) / 4  # snap to 0.25 grid
            lon_key = round(lon * 4) / 4
            grid[(lat_key, lon_key)] = cdi_val
    return grid


def parse_cdi_date(filename):
    """Extract date from CDI_YYYYMMDD.txt filename."""
    name = Path(filename).stem  # e.g. CDI_20210714
    date_str = name.replace("CDI_", "")
    return datetime.strptime(date_str, "%Y%m%d")


def find_nearest_cdi_file(target_date, cdi_files_by_date, max_days_off=4):
    """
    Find the CDI file closest to target_date.
    IDM files are weekly (every Wednesday), so max_days_off=4 covers the gap.
    Returns (date, filepath) or None.
    """
    best = None
    best_delta = timedelta(days=max_days_off + 1)
    for cdi_date, path in cdi_files_by_date.items():
        delta = abs(target_date - cdi_date)
        if delta < best_delta:
            best_delta = delta
            best = (cdi_date, path)
    if best and best_delta <= timedelta(days=max_days_off):
        return best
    return None


def snap_to_025(val):
    """Snap lat/lon to nearest 0.25-degree grid point."""
    return round(val * 4) / 4


def main():
    print("=" * 60)
    print("BUILD LAGGED TRAINING PAIRS")
    print("  features(W) -> CDI(W+7d) and CDI(W+15d)")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Index all CDI archive files by date
    # ---------------------------------------------------------------
    cdi_files = sorted(IDM_DIR.glob("CDI_*.txt"))
    if not cdi_files:
        print("ERROR: No CDI files found. Run download_idm_archive.py first.")
        return

    cdi_by_date = {}
    for f in cdi_files:
        try:
            d = parse_cdi_date(f.name)
            cdi_by_date[d] = f
        except ValueError:
            continue

    print(f"\nIDM archive: {len(cdi_by_date)} weekly CDI files")
    dates_sorted = sorted(cdi_by_date.keys())
    print(f"  Range: {dates_sorted[0].strftime('%Y-%m-%d')} to {dates_sorted[-1].strftime('%Y-%m-%d')}")

    # ---------------------------------------------------------------
    # 2. Load all GEE feature CSVs
    # ---------------------------------------------------------------
    feature_files = sorted(DATA_DIR.glob("drought_features_*_20*.csv"))
    # Exclude the test file and the combined files
    feature_files = [f for f in feature_files
                     if '_test' not in f.name
                     and '_all' not in f.name
                     and '_labeled' not in f.name]

    if not feature_files:
        print("ERROR: No feature CSVs found. Run build_drought_features.py first.")
        return

    print(f"\nGEE feature files: {len(feature_files)}")
    for f in feature_files:
        print(f"  {f.name}")

    # ---------------------------------------------------------------
    # 3. For each feature file, find CDI at W, W+7, W+15
    # ---------------------------------------------------------------
    lead7_rows = []
    lead15_rows = []
    stats = {'matched_7d': 0, 'matched_15d': 0, 'no_cdi_7d': 0, 'no_cdi_15d': 0}

    for feat_file in feature_files:
        # Parse region and date from filename
        # Format: drought_features_<region>_YYYYMMDD.csv
        parts = feat_file.stem.replace("drought_features_", "").rsplit("_", 1)
        if len(parts) != 2:
            print(f"  Skipping unrecognized filename: {feat_file.name}")
            continue
        region = parts[0]
        date_str = parts[1]
        try:
            feat_date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(f"  Skipping: can't parse date from {feat_file.name}")
            continue

        print(f"\n  Processing: {region} @ {feat_date.strftime('%Y-%m-%d')}")

        # Load features
        df = pd.read_csv(feat_file)
        print(f"    Rows: {len(df)}")

        # Snap each cell's lat/lon to 0.25-degree grid
        df['lat_025'] = df['lat'].apply(snap_to_025)
        df['lon_025'] = df['lon'].apply(snap_to_025)

        # Find CDI files for W+7 and W+15
        for lead_days, lead_rows, label in [(7, lead7_rows, '7d'), (15, lead15_rows, '15d')]:
            target_date = feat_date + timedelta(days=lead_days)
            match = find_nearest_cdi_file(target_date, cdi_by_date, max_days_off=4)

            if match is None:
                print(f"    No CDI file within 4 days of {target_date.strftime('%Y-%m-%d')} (W+{lead_days})")
                stats[f'no_cdi_{label}'] += 1
                continue

            cdi_date, cdi_path = match
            print(f"    CDI for W+{lead_days}: {cdi_path.name} (actual date: {cdi_date.strftime('%Y-%m-%d')}, offset: {(cdi_date - target_date).days}d)")

            cdi_grid = load_cdi_grid(cdi_path)
            print(f"    CDI grid points: {len(cdi_grid)}")

            # Join: for each GEE cell, look up CDI at nearest 0.25-degree point
            cdi_vals = []
            for _, row in df.iterrows():
                key = (row['lat_025'], row['lon_025'])
                cdi_vals.append(cdi_grid.get(key, np.nan))

            df_copy = df.copy()
            df_copy['target_cdi'] = cdi_vals
            df_copy['target_risk_score'] = df_copy['target_cdi'].apply(cdi_to_risk_score)
            df_copy['target_drought_class'] = df_copy['target_cdi'].apply(cdi_to_class)
            df_copy['lead_days'] = lead_days
            df_copy['feature_date'] = feat_date.strftime('%Y-%m-%d')
            df_copy['target_date'] = cdi_date.strftime('%Y-%m-%d')

            # Drop rows where CDI is NaN (no label available)
            valid = df_copy.dropna(subset=['target_cdi'])
            na_count = len(df_copy) - len(valid)
            print(f"    Valid pairs: {len(valid)}, NaN CDI: {na_count}")

            lead_rows.extend(valid.to_dict('records'))
            stats[f'matched_{label}'] += 1

    # ---------------------------------------------------------------
    # 4. Save training tables
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for lead_days, rows, outpath in [(7, lead7_rows, OUTPUT_LEAD7), (15, lead15_rows, OUTPUT_LEAD15)]:
        if not rows:
            print(f"\n  No valid W+{lead_days} pairs found!")
            continue

        df_out = pd.DataFrame(rows)
        # Drop intermediate columns
        df_out = df_out.drop(columns=['lat_025', 'lon_025'], errors='ignore')
        df_out.to_csv(outpath, index=False)

        print(f"\n  W+{lead_days}d training table:")
        print(f"    File: {outpath}")
        print(f"    Rows: {len(df_out)}")
        regions_present = df_out['region'].unique().tolist() if 'region' in df_out.columns else ['?']
        print(f"    Regions: {regions_present}")
        dates_present = df_out['feature_date'].nunique()
        print(f"    Feature dates: {dates_present}")
        print(f"    Target drought class distribution:")
        print(df_out['target_drought_class'].value_counts().to_string())
        print(f"    Target risk score stats:")
        print(df_out['target_risk_score'].describe().round(4).to_string())

    print(f"\n  Stats: {stats}")


if __name__ == "__main__":
    main()
