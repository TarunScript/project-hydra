#!/usr/bin/env python3
"""
build_district_static_multistate.py

Builds district_static_features.csv for Bihar, West Bengal, and Odisha
using the same raw datasets used for Assam:
  - DFSI.csv             → dfsi_score (normalised 0-1)
  - District_FloodedArea → pct_flooded_area
  - District_FloodImpact → population, historical_fatalities, mean_flood_duration
  - India_Flood_Inventory → hist_flood_frequency (events/year per district)

Output: flood_model/data/features/<state>/district_static_features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
RAW_DIR   = BASE_DIR / "data" / "raw"
FEAT_DIR  = BASE_DIR / "data" / "features"

STATES = {
    "bihar":       "BIHAR",
    "west_bengal": "WEST BENGAL",
    "odisha":      "ODISHA",
}

# ── Load raw files ──────────────────────────────────────────
print("Loading raw data files...")

dfsi_raw    = pd.read_csv(RAW_DIR / "DFSI.csv")
dfsi_raw.columns = ["district_name", "state_name", "dfsi_raw"]
dfsi_raw["district_name"] = dfsi_raw["district_name"].str.strip().str.upper()
dfsi_raw["state_name"]    = dfsi_raw["state_name"].str.strip().str.upper()

area_raw    = pd.read_csv(RAW_DIR / "District_FloodedArea.csv")
area_raw["district_name"] = area_raw["Dist_Name"].str.strip().str.upper()

impact_raw  = pd.read_csv(RAW_DIR / "District_FloodImpact.csv")
impact_raw["district_name"] = impact_raw["Dist_Name"].str.strip().str.upper()

inventory   = pd.read_csv(RAW_DIR / "India_Flood_Inventory_v3.csv")
inventory["State"] = inventory["State"].str.strip().str.upper()

# DFSI global normalisation (min-max across all districts, same as Assam)
dfsi_min = dfsi_raw["dfsi_raw"].min()
dfsi_max = dfsi_raw["dfsi_raw"].max()

print(f"  DFSI: {len(dfsi_raw)} rows | Area: {len(area_raw)} rows | Impact: {len(impact_raw)} rows | Inventory: {len(inventory)} rows")


# ── Compute hist_flood_frequency from inventory ──────────────
# Count distinct flood events per district per state, divide by year span
YEAR_SPAN = 2023 - 1967 + 1   # same span as inventory

def compute_flood_frequency(state_inventory_name):
    """Count flood events per district from the flood inventory."""
    state_inv = inventory[
        inventory["State"].str.contains(state_inventory_name, case=False, na=False)
    ].copy()

    freq_map = {}
    if "Districts" in state_inv.columns:
        for _, row in state_inv.iterrows():
            if pd.isna(row.get("Districts")):
                continue
            dists = str(row["Districts"]).split(",")
            for d in dists:
                d = d.strip().upper()
                if d:
                    freq_map[d] = freq_map.get(d, 0) + 1

    # Convert to events/year
    return {d: round(count / YEAR_SPAN, 3) for d, count in freq_map.items()}


# ── Process each state ────────────────────────────────────────
for state_key, state_display in STATES.items():
    print(f"\n{'='*55}")
    print(f"  Building: {state_display}")
    print(f"{'='*55}")

    out_dir = FEAT_DIR / state_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "district_static_features.csv"

    # ── 1. DFSI for this state ──
    state_dfsi = dfsi_raw[dfsi_raw["state_name"] == state_display].copy()
    state_dfsi["dfsi_score"] = (state_dfsi["dfsi_raw"] - dfsi_min) / (dfsi_max - dfsi_min)
    print(f"  DFSI districts: {len(state_dfsi)}")

    # ── 2. Flooded area ── (district names only, no state column in raw file)
    # We match by district name fuzzy matching against DFSI district names
    # Build merged table starting from DFSI as the master list
    merged = state_dfsi[["district_name", "dfsi_raw", "dfsi_score"]].copy()

    # ── 3. Merge flooded area ──
    area_sub = area_raw[["district_name", "Corrected_Percent_Flooded_Area"]].rename(
        columns={"Corrected_Percent_Flooded_Area": "pct_flooded_area"}
    )
    merged = merged.merge(area_sub, on="district_name", how="left")

    # Fill missing pct_flooded_area with state median
    state_area_median = merged["pct_flooded_area"].median()
    merged["pct_flooded_area"] = merged["pct_flooded_area"].fillna(state_area_median)

    # ── 4. Merge flood impact (population, fatalities, duration) ──
    impact_sub = impact_raw[["district_name", "Human_fatality", "Population", "Mean_Flood_Duration"]].rename(
        columns={
            "Human_fatality":     "historical_fatalities",
            "Population":         "population",
            "Mean_Flood_Duration": "mean_flood_duration",
        }
    )
    merged = merged.merge(impact_sub, on="district_name", how="left")

    # ── 5. hist_flood_frequency from inventory ──
    freq_map = compute_flood_frequency(state_display)
    merged["hist_flood_frequency"] = merged["district_name"].map(freq_map)

    # Fill missing values sensibly
    for col, fill_method in [
        ("mean_flood_duration",   "median"),
        ("historical_fatalities", "median"),
        ("population",            "median"),
        ("hist_flood_frequency",  "median"),
    ]:
        if fill_method == "median":
            merged[col] = merged[col].fillna(merged[col].median())

    # Scale hist_flood_frequency to raw count (events total) to match Assam format
    merged["hist_flood_frequency"] = (merged["hist_flood_frequency"] * YEAR_SPAN).round(0).astype(int)

    # ── Final column order (matches Assam district_static_features.csv) ──
    out_cols = [
        "district_name", "dfsi_raw", "dfsi_score",
        "pct_flooded_area", "mean_flood_duration",
        "population", "historical_fatalities", "hist_flood_frequency"
    ]
    merged = merged[out_cols].sort_values("dfsi_score", ascending=False).reset_index(drop=True)

    merged.to_csv(out_path, index=False)
    print(f"  ✓ {len(merged)} districts → {out_path}")

    # Summary
    print(f"\n  Top 5 highest-risk districts:")
    print(f"  {'District':25s} {'DFSI':6s} {'Flood Freq':10s} {'Pct Area':8s}")
    print(f"  {'-'*55}")
    for _, row in merged.head(5).iterrows():
        print(f"  {row['district_name']:25s} {row['dfsi_score']:.3f}  {int(row['hist_flood_frequency']):>8}    {row['pct_flooded_area']:.2f}%")

print(f"\n{'='*55}")
print("✓ All state district static features built.")
print(f"  Bihar:       {FEAT_DIR}/bihar/district_static_features.csv")
print(f"  West Bengal: {FEAT_DIR}/west_bengal/district_static_features.csv")
print(f"  Odisha:      {FEAT_DIR}/odisha/district_static_features.csv")
print(f"{'='*55}")
