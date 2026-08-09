#!/usr/bin/env python3
"""
build_labels.py — Task 1: Build ALL "made by us" datasets

Produces 3 output files from the already-downloaded IFI-Impacts data:

1. data/features/labels.csv
   - Flood labels: one row per (district × day) for May–Oct 2015–2023
   - is_flood = 1 if that district was in an active flood event on that day
   - is_flood = 0 otherwise (full monsoon calendar)

2. data/features/district_static_features.csv
   - Per-district static features: DFSI score, % flooded area,
     mean flood duration, population, historical flood frequency

3. data/features/rainfall_climatology.csv
   - Monthly rainfall normals from India Drought Atlas
   - (downloaded from GitHub if not cached)

User-chosen parameters:
  - Date range: 2015–2023
  - Months: May–October (monsoon)
  - Balancing: Full calendar, natural ratio (handle in model)
  - Granularity: District × Day
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
FEATURES_DIR = BASE_DIR / "data" / "features"

# ── User-chosen parameters ──
YEAR_START = 2015
YEAR_END = 2023
MONSOON_MONTHS = [5, 6, 7, 8, 9, 10]  # May–October
TARGET_STATE = "ASSAM"


# ═══════════════════════════════════════════════════════════
# DATASET 1: Flood labels (district × day)
# ═══════════════════════════════════════════════════════════

def build_flood_labels() -> pd.DataFrame:
    """
    Build binary flood labels: one row per (district, date).
    is_flood = 1 if that district had an active flood event on that date.
    """
    print("\n" + "=" * 60)
    print("DATASET 1: Flood Labels (district × day)")
    print("=" * 60)

    # ── Load inventory ──
    inv_path = RAW_DIR / "India_Flood_Inventory_v3.csv"
    if not inv_path.exists():
        print("  ✗ India_Flood_Inventory_v3.csv not found in data/raw/")
        sys.exit(1)

    inv = pd.read_csv(inv_path)

    # ── Filter to Assam events ──
    inv["state_clean"] = inv["State"].astype(str).str.strip().str.upper()
    inv_assam = inv[inv["state_clean"].str.contains("ASSAM", case=False, na=False)].copy()
    print(f"  Assam events in inventory: {len(inv_assam)}")

    # ── Parse dates ──
    inv_assam["start_dt"] = pd.to_datetime(
        inv_assam["Start Date"], format="mixed", dayfirst=True, errors="coerce"
    )
    inv_assam["end_dt"] = pd.to_datetime(
        inv_assam["End Date"], format="mixed", dayfirst=True, errors="coerce"
    )

    # Filter to our date range
    inv_assam = inv_assam[
        (inv_assam["start_dt"].dt.year >= YEAR_START)
        & (inv_assam["start_dt"].dt.year <= YEAR_END)
    ].copy()
    print(f"  Assam events in {YEAR_START}–{YEAR_END}: {len(inv_assam)}")

    # ── Extract district lists per event ──
    # The 'Districts' column has comma-separated district names
    # We expand each event into (district, date) rows

    flood_records = set()  # (district_name, date_str) tuples

    for _, row in inv_assam.iterrows():
        districts_raw = row.get("Districts", "")
        if pd.isna(districts_raw) or str(districts_raw).strip() == "":
            continue

        # Parse comma-separated district list
        districts = [d.strip().upper() for d in str(districts_raw).split(",") if d.strip()]

        start = row["start_dt"]
        end = row["end_dt"]
        if pd.isna(start):
            continue
        if pd.isna(end):
            end = start  # single-day event

        # Expand to each day in the event range
        current = start.date()
        end_date = end.date()
        while current <= end_date:
            # Only include monsoon months
            if current.month in MONSOON_MONTHS:
                for dist in districts:
                    flood_records.add((dist, current.isoformat()))
            current += timedelta(days=1)

    print(f"  Flood (district, day) pairs: {len(flood_records)}")

    # ── Get all Assam district names from DFSI ──
    dfsi = pd.read_csv(RAW_DIR / "DFSI.csv")
    dfsi_assam = dfsi[dfsi["State_Name"].str.contains("ASSAM", case=False, na=False)]
    all_districts = sorted(dfsi_assam.iloc[:, 0].str.strip().str.upper().unique())
    print(f"  All Assam districts (from DFSI): {len(all_districts)}")

    # ── Build full calendar: all districts × all monsoon days ──
    print(f"  Building full calendar for {YEAR_START}–{YEAR_END}, months {MONSOON_MONTHS}...")

    rows = []
    for year in range(YEAR_START, YEAR_END + 1):
        for month in MONSOON_MONTHS:
            # Get days in this month
            if month == 12:
                next_month_start = date(year + 1, 1, 1)
            else:
                next_month_start = date(year, month + 1, 1)
            month_start = date(year, month, 1)
            current = month_start
            while current < next_month_start:
                for dist in all_districts:
                    is_flood = 1 if (dist, current.isoformat()) in flood_records else 0
                    rows.append({
                        "district_name": dist,
                        "date": current.isoformat(),
                        "year": year,
                        "month": month,
                        "is_flood": is_flood,
                    })
                current += timedelta(days=1)

    labels = pd.DataFrame(rows)

    # ── Print summary ──
    total = len(labels)
    flood_count = labels["is_flood"].sum()
    no_flood_count = total - flood_count
    ratio = flood_count / total * 100

    print(f"\n  ── Label Summary ──")
    print(f"  Total rows:      {total:,}")
    print(f"  Flood days:      {int(flood_count):,} ({ratio:.1f}%)")
    print(f"  Non-flood days:  {int(no_flood_count):,} ({100 - ratio:.1f}%)")
    print(f"  Districts:       {labels['district_name'].nunique()}")
    print(f"  Date range:      {labels['date'].min()} to {labels['date'].max()}")
    print(f"  Imbalance ratio: 1:{no_flood_count / max(flood_count, 1):.1f} (flood : no-flood)")

    # Flood days per year
    print(f"\n  Flood days by year:")
    yearly = labels[labels["is_flood"] == 1].groupby("year").size()
    for yr, count in yearly.items():
        print(f"    {yr}: {count:,} flood-district-days")

    # Flood days per district (top 10)
    print(f"\n  Top 10 most flooded districts:")
    by_dist = labels[labels["is_flood"] == 1].groupby("district_name").size().sort_values(ascending=False)
    for dist, count in by_dist.head(10).items():
        print(f"    {dist}: {count} flood-days")

    return labels


# ═══════════════════════════════════════════════════════════
# DATASET 2: District static features
# ═══════════════════════════════════════════════════════════

def build_district_static_features() -> pd.DataFrame:
    """
    Compile per-district static features from the 3 already-downloaded files:
    - DFSI.csv → dfsi_score (normalized 0-1)
    - District_FloodedArea.csv → pct_flooded_area
    - District_FloodImpact.csv → mean_flood_duration, population, fatalities
    Plus derived: historical_flood_frequency from inventory
    """
    print("\n" + "=" * 60)
    print("DATASET 2: District Static Features")
    print("=" * 60)

    # ── DFSI ──
    dfsi = pd.read_csv(RAW_DIR / "DFSI.csv")
    dfsi_assam = dfsi[dfsi["State_Name"].str.contains("ASSAM", case=False, na=False)].copy()
    dfsi_assam = dfsi_assam.rename(columns={dfsi_assam.columns[0]: "district_name", "DFSI": "dfsi_raw"})
    dfsi_assam["district_name"] = dfsi_assam["district_name"].str.strip().str.upper()

    # Normalize DFSI globally (0-1 across all India)
    global_min = dfsi["DFSI"].min()
    global_max = dfsi["DFSI"].max()
    dfsi_assam["dfsi_score"] = (dfsi_assam["dfsi_raw"] - global_min) / (global_max - global_min)

    result = dfsi_assam[["district_name", "dfsi_raw", "dfsi_score"]].copy()
    print(f"  DFSI: {len(result)} districts, range [{result['dfsi_score'].min():.3f}, {result['dfsi_score'].max():.3f}]")

    # ── Flooded Area ──
    fa_path = RAW_DIR / "District_FloodedArea.csv"
    if fa_path.exists():
        fa = pd.read_csv(fa_path)
        fa = fa.rename(columns={"Dist_Name": "district_name"})
        fa["district_name"] = fa["district_name"].str.strip().str.upper()
        fa = fa[fa["district_name"].isin(result["district_name"])]
        result = result.merge(
            fa[["district_name", "Corrected_Percent_Flooded_Area"]].rename(
                columns={"Corrected_Percent_Flooded_Area": "pct_flooded_area"}
            ),
            on="district_name", how="left"
        )
        print(f"  FloodedArea: matched {result['pct_flooded_area'].notna().sum()} districts")

    # ── Flood Impact ──
    fi_path = RAW_DIR / "District_FloodImpact.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        fi = fi.rename(columns={"Dist_Name": "district_name"})
        fi["district_name"] = fi["district_name"].str.strip().str.upper()
        fi = fi[fi["district_name"].isin(result["district_name"])]
        result = result.merge(
            fi[["district_name", "Mean_Flood_Duration", "Population", "Human_fatality"]].rename(
                columns={
                    "Mean_Flood_Duration": "mean_flood_duration",
                    "Population": "population",
                    "Human_fatality": "historical_fatalities",
                }
            ),
            on="district_name", how="left"
        )
        print(f"  FloodImpact: matched {result['mean_flood_duration'].notna().sum()} districts")

    # ── Historical Flood Frequency (derived from inventory) ──
    inv_path = RAW_DIR / "India_Flood_Inventory_v3.csv"
    if inv_path.exists():
        inv = pd.read_csv(inv_path)
        inv_assam = inv[inv["State"].astype(str).str.upper().str.contains("ASSAM", na=False)]

        # Count how many events mention each district
        freq_counts = {}
        for _, row in inv_assam.iterrows():
            districts_raw = row.get("Districts", "")
            if pd.isna(districts_raw):
                continue
            for d in str(districts_raw).split(","):
                d = d.strip().upper()
                if d:
                    freq_counts[d] = freq_counts.get(d, 0) + 1

        freq_df = pd.DataFrame([
            {"district_name": k, "hist_flood_frequency": v}
            for k, v in freq_counts.items()
        ])
        result = result.merge(freq_df, on="district_name", how="left")
        result["hist_flood_frequency"] = result["hist_flood_frequency"].fillna(0).astype(int)
        print(f"  Flood frequency: range [{result['hist_flood_frequency'].min()}, {result['hist_flood_frequency'].max()}]")

    # ── Summary ──
    print(f"\n  ── District Static Features Summary ──")
    print(f"  Shape: {result.shape}")
    print(f"  Columns: {list(result.columns)}")
    print(f"\n  Null summary:")
    for col in result.columns:
        nulls = result[col].isnull().sum()
        print(f"    {'✓' if nulls == 0 else '⚠'} {col}: {nulls} nulls")
    print(f"\n  Full table:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(result.to_string(index=False))

    return result


# ═══════════════════════════════════════════════════════════
# DATASET 3: Rainfall climatology (India Drought Atlas)
# ═══════════════════════════════════════════════════════════

def build_rainfall_climatology() -> pd.DataFrame | None:
    """
    Download monthly precipitation climatology from India Drought Atlas
    GitHub repo (wcl-iitgn/india-drought-atlas-data).
    
    This gives us the long-term normal rainfall per grid cell per month,
    needed to compute rainfall anomaly features.
    """
    print("\n" + "=" * 60)
    print("DATASET 3: Rainfall Climatology (India Drought Atlas)")
    print("=" * 60)

    # The India Drought Atlas data repo has gridded monthly data
    # Try to find the precipitation climatology file
    repo_base = "https://raw.githubusercontent.com/wcl-iitgn/india-drought-atlas-data/main"

    # Try common file paths in the repo
    possible_paths = [
        "precipitation/monthly_climatology.csv",
        "data/precipitation_climatology.csv",
        "climatology/precipitation.csv",
        "IMD_RF_Clim.csv",
    ]

    # First, let's check what's actually in the repo
    print("  Checking India Drought Atlas repo structure...")
    api_url = "https://api.github.com/repos/wcl-iitgn/india-drought-atlas-data/contents/"

    try:
        resp = requests.get(api_url, timeout=15)
        if resp.status_code == 200:
            contents = resp.json()
            print(f"  Repo root contents:")
            for item in contents:
                print(f"    {'📁' if item['type'] == 'dir' else '📄'} {item['name']} ({item.get('size', 'dir')})")

            # Recursively check directories for data files
            for item in contents:
                if item["type"] == "dir":
                    sub_resp = requests.get(item["url"], timeout=15)
                    if sub_resp.status_code == 200:
                        sub_contents = sub_resp.json()
                        csv_files = [f for f in sub_contents if f["name"].endswith((".csv", ".nc", ".txt"))]
                        if csv_files:
                            print(f"\n    📁 {item['name']}/")
                            for f in csv_files[:10]:
                                print(f"      📄 {f['name']} ({f.get('size', '?')} bytes)")
        else:
            print(f"  ⚠ GitHub API returned {resp.status_code}")
    except Exception as e:
        print(f"  ⚠ Could not check repo: {e}")

    # For now, we'll create a simple monthly climatology from CHIRPS
    # (will be computed in the GEE pipeline as a long-term mean)
    print("\n  ℹ Rainfall climatology will be computed from CHIRPS in the GEE pipeline")
    print("    (long-term monthly mean for each grid cell)")
    print("    Saving a placeholder config for the GEE script to use")

    # Save config for GEE pipeline
    config = pd.DataFrame({
        "month": MONSOON_MONTHS,
        "description": [
            "May - pre-monsoon",
            "June - monsoon onset",
            "July - peak monsoon",
            "August - active monsoon",
            "September - retreating monsoon",
            "October - post-monsoon",
        ],
        "climatology_years": ["1981-2014"] * 6,  # baseline period
        "source": ["CHIRPS via GEE"] * 6,
    })

    return config


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Building ALL 'Made By Us' Datasets                        ║")
    print("║  Region: Assam | Range: 2015–2023 | Months: May–Oct        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Dataset 1: Flood Labels ──
    labels = build_flood_labels()
    labels_path = FEATURES_DIR / "labels.csv"
    labels.to_csv(labels_path, index=False)
    print(f"\n  ✓ Saved: {labels_path} ({labels_path.stat().st_size / 1024:.1f} kB)")

    # ── Dataset 2: District Static Features ──
    static_features = build_district_static_features()
    static_path = FEATURES_DIR / "district_static_features.csv"
    static_features.to_csv(static_path, index=False)
    print(f"\n  ✓ Saved: {static_path} ({static_path.stat().st_size / 1024:.1f} kB)")

    # ── Dataset 3: Rainfall Climatology ──
    climatology = build_rainfall_climatology()
    if climatology is not None:
        clim_path = FEATURES_DIR / "rainfall_climatology_config.csv"
        climatology.to_csv(clim_path, index=False)
        print(f"\n  ✓ Saved: {clim_path}")

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("ALL 'MADE BY US' DATASETS COMPLETE")
    print("=" * 60)
    print(f"\n  📄 labels.csv              → {len(labels):,} rows (district × day, flood/no-flood)")
    print(f"  📄 district_static_features.csv → {len(static_features)} rows (per-district features)")
    print(f"  📄 rainfall_climatology_config.csv → config for GEE to compute anomalies")
    print(f"\n  Files saved to: {FEATURES_DIR}")


if __name__ == "__main__":
    main()
