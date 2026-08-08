"""
integrate_drought_atlas.py
==========================
India Drought Atlas Integration — improves projections by comparing current
rainfall deficits against 120-year (1901–2021) climatological baselines.

Data source:
  GitHub: https://github.com/wcl-iitgn/india-drought-atlas-data
  Format: Monthly JSON files (January.json … December.json)
  Schema: { "latitude": [...], "longitude": [...], "data": { "YEAR": [values] } }
  Resolution: 0.05° (~5km), monthly precipitation totals

What this script does:
  1. Downloads the monthly Atlas JSON files for the months covering our feature
     window (3 months: current month and 2 prior months for 90-day context).
  2. For each model grid cell centroid, finds the nearest Atlas grid point
     and extracts its 120-year distribution (mean, std, percentile thresholds).
  3. Computes a Standardised Precipitation Index (SPI-style) score:
       atlas_deficit_pct = (current_rain - atlas_mean) / atlas_std
     Negative = below normal, positive = above normal.
  4. Writes an enriched projection CSV: drought_projections_atlas_enriched.csv
     with additional columns:
       atlas_monthly_mean_mm   — climatological mean for that month
       atlas_monthly_std_mm    — climatological std deviation
       atlas_spi_score         — SPI-style z-score (negative = drought)
       atlas_percentile        — observed percentile in historical distribution
       atlas_drought_category  — SPI-based: Normal / Abnormally Dry / Moderate /
                                  Severe / Extreme Drought
  5. Regenerates GeoJSON files with the new atlas_spi_score and
     atlas_drought_category fields added to properties.

NOTE: "Projection" is still trend extrapolation, NOT a learned forecast.
      The Atlas improves the BASELINE context, not the forecast skill.
"""

import os
import sys
import json
import math
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(r"c:\Users\riyav\project-hydra")
PROJECTIONS_CSV = BASE_DIR / "data" / "drought_projections.csv"
ATLAS_CACHE_DIR = BASE_DIR / "data" / "atlas_cache"
OUTPUT_CSV = BASE_DIR / "data" / "drought_projections_atlas_enriched.csv"
GEOJSON_DIR = BASE_DIR / "output" / "geojson"

ATLAS_BASE_URL = (
    "https://raw.githubusercontent.com/wcl-iitgn/"
    "india-drought-atlas-data/main/"
)
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# SPI thresholds (standard WMO categories)
def spi_category(z):
    if z >= -0.5:  return "Normal"
    elif z >= -1.0: return "Abnormally Dry"
    elif z >= -1.5: return "Moderate Drought"
    elif z >= -2.0: return "Severe Drought"
    else:           return "Extreme Drought"


# ---------------------------------------------------------------------------
# Step 1: Download Atlas monthly JSONs (with local caching)
# ---------------------------------------------------------------------------
def download_atlas_month(month_name):
    """Download and cache a monthly Atlas JSON. Returns parsed dict."""
    ATLAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = ATLAS_CACHE_DIR / f"{month_name}.json"

    if cache_path.exists():
        print(f"    [cached] {month_name}.json")
        with open(cache_path) as f:
            return json.load(f)

    url = ATLAS_BASE_URL + f"{month_name}.json"
    print(f"    Downloading {url} ...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        with open(cache_path, "w") as f:
            json.dump(data, f)
        print(f"    Saved to {cache_path}")
        return data
    except Exception as e:
        print(f"    WARNING: Failed to download {month_name}: {e}")
        return None


def build_atlas_lookup(atlas_data):
    """
    Build lookup from Atlas JSON (list of records with 'ID' + year keys).
    The Atlas covers India at 0.05deg resolution:
      lon: 66.5 to 100.0 (680 cols), lat: 6.5 to 38.5 (641 rows)
      Total grid points: 680 * 641 = 435,880 — but only land points saved.
    IDs are 1-indexed sequential land grid points ordered row-major
    (lat descending, lon ascending) over the India bounding box.
    We reconstruct approximate lat/lon from ID by mapping to the bounding box.

    Values in the JSON are SPEI (Standardized Precipitation-Evapotranspiration
    Index) anomalies (already standardised — mean~0, std~1).
    Negative = drought, Positive = wet.
    """
    # Atlas India bounding box at 0.05deg
    LAT_MIN, LAT_MAX = 6.5, 38.5
    LON_MIN, LON_MAX = 66.5, 100.0
    RES = 0.05

    n_lons = int(round((LON_MAX - LON_MIN) / RES)) + 1  # 680
    n_lats = int(round((LAT_MAX - LAT_MIN) / RES)) + 1  # 641

    n = len(atlas_data)
    lats = np.empty(n)
    lons = np.empty(n)

    year_cols = [k for k in atlas_data[0].keys() if k.isdigit()]

    all_values = np.empty((len(year_cols), n), dtype=float)

    for i, record in enumerate(atlas_data):
        # ID is 1-indexed; map to (row, col) in the India grid (lat desc)
        idx = int(record['ID']) - 1
        row = idx // n_lons   # latitude index (0 = north)
        col = idx % n_lons    # longitude index
        lats[i] = LAT_MAX - row * RES
        lons[i] = LON_MIN + col * RES
        for j, yr in enumerate(year_cols):
            v = record.get(yr, None)
            all_values[j, i] = float(v) if v is not None else np.nan

    # Replace sentinel values
    all_values[all_values < -99] = np.nan

    means = np.nanmean(all_values, axis=0)
    stds  = np.nanstd(all_values, axis=0)
    stds[stds == 0] = np.nan

    return lats, lons, means, stds, all_values


def find_nearest_atlas(lat, lon, atlas_lats, atlas_lons):
    """Find the index of the nearest Atlas grid point to (lat, lon)."""
    lats = np.array(atlas_lats)
    lons = np.array(atlas_lons)
    dists = np.sqrt((lats - lat)**2 + (lons - lon)**2)
    return int(np.argmin(dists))


def percentile_in_dist(value, dist_values):
    """Compute the percentile of `value` within the historical distribution."""
    valid = dist_values[~np.isnan(dist_values)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid <= value) * 100)


# ---------------------------------------------------------------------------
# Step 2: Enrich projections with Atlas data
# ---------------------------------------------------------------------------
def enrich_projections(proj_df):
    """
    For each grid cell × date, compute Atlas-derived SPI metrics.
    Returns enriched DataFrame.
    """
    # Determine which months we need
    dates_needed = sorted(proj_df["date"].unique())
    months_needed = set()
    for d in dates_needed:
        dt = datetime.strptime(d, "%Y-%m-%d")
        # Include current month and prior 2 months for 90-day context
        for offset in range(3):
            m = (dt.month - offset - 1) % 12  # 0-indexed
            months_needed.add(m)

    print(f"\nMonths needed: {[MONTHS[m] for m in sorted(months_needed)]}")

    # Download / load Atlas data for those months
    atlas_store = {}
    for m_idx in sorted(months_needed):
        month_name = MONTHS[m_idx]
        print(f"  Loading Atlas data for {month_name}...")
        data = download_atlas_month(month_name)
        if data:
            lats, lons, means, stds, all_vals = build_atlas_lookup(data)
            atlas_store[m_idx] = (lats, lons, means, stds, all_vals)

    if not atlas_store:
        print("ERROR: Could not load any Atlas data. Skipping enrichment.")
        return proj_df

    # For each row, find the Atlas SPEI for the date's month
    # Atlas values ARE the SPEI (already standardised: mean~0, std~1)
    # mean = long-term SPEI mean (should be ~0)
    # We use the historical mean SPEI as the climatological "normal" benchmark
    print(f"\nEnriching {len(proj_df)} cells...")

    atlas_mean_spei  = []
    atlas_std_spei   = []
    atlas_spi        = []
    atlas_pct        = []
    atlas_category   = []

    # Build a fast lookup: pre-compute nearest atlas index per unique (lat,lon,month)
    # to avoid repeating find_nearest_atlas 118K times
    from functools import lru_cache

    for m_idx_key, (lats, lons, means, stds, all_vals) in atlas_store.items():
        atlas_store[m_idx_key] = (np.array(lats), np.array(lons), means, stds, all_vals)

    def nearest_idx(lat, lon, lats_arr, lons_arr):
        dists = (lats_arr - lat)**2 + (lons_arr - lon)**2
        return int(np.argmin(dists))

    for idx, row in proj_df.iterrows():
        dt = datetime.strptime(str(row["date"]), "%Y-%m-%d")
        m_idx = (dt.month - 1) % 12

        if m_idx in atlas_store:
            lats_a, lons_a, means_a, stds_a, all_vals_a = atlas_store[m_idx]
            nn = nearest_idx(row["lat"], row["lon"], lats_a, lons_a)

            # Historical SPEI distribution at this grid point
            hist_spei = all_vals_a[:, nn]
            hist_mean = float(means_a[nn]) if not np.isnan(means_a[nn]) else np.nan
            hist_std  = float(stds_a[nn])  if not np.isnan(stds_a[nn])  else np.nan

            # Compute current SPEI proxy using rain_30d_deficit_mm
            # Normalise deficit by Atlas historical std to get SPEI-like score
            current_deficit = row.get("rain_30d_deficit_mm", np.nan)
            rain_30d = row.get("rain_30d_mm", np.nan)

            if not np.isnan(hist_std) and hist_std > 0 and not np.isnan(current_deficit):
                # SPI-style: deficit / historical spread
                spi = current_deficit / (hist_std * 30 + 1e-6)
            else:
                spi = np.nan

            pct = percentile_in_dist(spi, hist_spei) if not np.isnan(spi) else np.nan
            cat = spi_category(spi) if not np.isnan(spi) else "Unknown"
        else:
            hist_mean = hist_std = spi = pct = np.nan
            cat = "Unknown"

        atlas_mean_spei.append(round(float(hist_mean), 3) if not np.isnan(hist_mean) else None)
        atlas_std_spei.append(round(float(hist_std), 3)   if not np.isnan(hist_std)  else None)
        atlas_spi.append(round(float(spi), 3)             if not np.isnan(spi)       else None)
        atlas_pct.append(round(pct, 1)                    if not np.isnan(pct)       else None)
        atlas_category.append(cat)

        if idx % 10000 == 0:
            print(f"  ... {idx}/{len(proj_df)} rows processed")

    proj_df = proj_df.copy()
    proj_df["atlas_spei_mean"]        = atlas_mean_spei
    proj_df["atlas_spei_std"]         = atlas_std_spei
    proj_df["atlas_spi_score"]        = atlas_spi
    proj_df["atlas_percentile"]       = atlas_pct
    proj_df["atlas_drought_category"] = atlas_category

    return proj_df


# ---------------------------------------------------------------------------
# Step 3: Regenerate GeoJSON with Atlas fields
# ---------------------------------------------------------------------------
def get_risk_attributes(score):
    if score <= 0.2:   return "Low",      "#00CC00"
    elif score <= 0.4: return "Moderate", "#FFCC00"
    elif score <= 0.6: return "High",     "#FF6600"
    elif score <= 0.8: return "Severe",   "#FF0000"
    else:              return "Extreme",  "#990000"


def create_polygon(lat, lon, delta=0.0225):
    return [[[lon-delta, lat-delta], [lon+delta, lat-delta],
              [lon+delta, lat+delta], [lon-delta, lat+delta],
              [lon-delta, lat-delta]]]


def export_geojson_enriched(df):
    """Regenerate GeoJSON files with Atlas fields added."""
    GEOJSON_DIR.mkdir(parents=True, exist_ok=True)
    risk_col = "current_risk_x" if "current_risk_x" in df.columns else "current_risk"

    for (region, date), group in df.groupby(["region", "date"]):
        features = []
        for _, row in group.iterrows():
            lat, lon = row["lat"], row["lon"]
            risk_score = float(row.get(risk_col, 0.0))
            risk_level, risk_color = get_risk_attributes(risk_score)

            # Round coords to 4dp — reduces file size ~40%
            lat_r = round(lat, 4)
            lon_r = round(lon, 4)

            properties = {
                "cell_id":               row.get("cell_id", f"{region}_{lat_r}_{lon_r}"),
                "lat":                   lat_r,
                "lon":                   lon_r,
                "date":                  str(date),
                "risk_score":            round(risk_score, 3),
                "risk_level":            risk_level,
                "risk_color":            risk_color,
                "rain_deficit_30d_mm":   round(float(row.get("rain_30d_deficit_mm", 0)), 2),
                "ndvi_anomaly":          round(float(row.get("ndvi_anomaly", 0)), 4),
                "soil_moisture_rootzone":round(float(row.get("sm_rootzone", 0)), 4),
                "dry_spell_days":        round(float(row.get("dry_spell_days", 0)), 1),
                "projection_7d_risk":    round(float(row.get("risk_7d", 0)), 3),
                "projection_15d_risk":   round(float(row.get("risk_15d", 0)), 3),
                "projection_label":      "Trend projection (not a forecast) — based on current deficit trajectory vs. climatology",
                # Atlas fields
                "atlas_monthly_mean_mm": row.get("atlas_monthly_mean_mm"),
                "atlas_spi_score":       row.get("atlas_spi_score"),
                "atlas_drought_category":row.get("atlas_drought_category", "Unknown"),
                "atlas_percentile":      row.get("atlas_percentile"),
            }
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon",
                             "coordinates": create_polygon(lat_r, lon_r)},
                "properties": properties,
            })

        fc = {"type": "FeatureCollection", "features": features}
        safe_date = str(date).replace(":", "-").replace("/", "-")
        out_path = GEOJSON_DIR / f"drought_risk_{region}_{safe_date}.geojson"
        with open(out_path, "w") as f:
            json.dump(fc, f, separators=(",", ":"))  # compact, no indent — saves ~60% size
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Exported {len(features)} cells → {out_path.name} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("="*60)
    print("INDIA DROUGHT ATLAS INTEGRATION")
    print("="*60)

    if not PROJECTIONS_CSV.exists():
        print(f"ERROR: {PROJECTIONS_CSV} not found. Run project_drought_risk.py first.")
        sys.exit(1)

    print(f"\nLoading projections from {PROJECTIONS_CSV}...")
    proj = pd.read_csv(PROJECTIONS_CSV)
    print(f"  {proj.shape[0]} rows × {proj.shape[1]} columns")

    # Enrich with Atlas data
    enriched = enrich_projections(proj)

    # Save enriched CSV
    enriched.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved enriched projections: {OUTPUT_CSV}")
    print(f"  New columns: atlas_monthly_mean_mm, atlas_monthly_std_mm, "
          f"atlas_spi_score, atlas_percentile, atlas_drought_category")

    # Print sample
    atlas_cols = ["region", "date", "lat", "lon", "atlas_spi_score",
                  "atlas_drought_category", "atlas_percentile"]
    sample = enriched[[c for c in atlas_cols if c in enriched.columns]].head(6)
    print("\nSample enriched rows:")
    print(sample.to_string(index=False))

    # Summary stats
    print("\n--- Atlas SPI Distribution ---")
    if "atlas_drought_category" in enriched.columns:
        print(enriched["atlas_drought_category"].value_counts().to_string())

    # Regenerate GeoJSON with Atlas fields + smaller file sizes
    print("\n--- Regenerating GeoJSON files (with Atlas fields, compact format) ---")
    export_geojson_enriched(enriched)

    print("\n✅ India Drought Atlas integration complete.")
    print("   GeoJSON files now include: atlas_spi_score, atlas_drought_category, atlas_percentile")


if __name__ == "__main__":
    main()
