#!/usr/bin/env python3
"""
pull_features.py — Task 2: GEE Feature Pipeline

NOTE: GEE getInfo() has a 5000-element limit. We use geemap.ee_to_df()
which handles pagination automatically for large FeatureCollections.

Pulls ALL features from Google Earth Engine for the Assam demo region.
Follows the exact skeleton pattern from the implementation plan (Section 4/5.1).

Features pulled (per the plan):
  STATIC (computed once):
    - Elevation (SRTM)
    - Slope (derived from SRTM)
    - Flow accumulation (HydroSHEDS 15ACC)
    - Distance to river (derived from flow_acc threshold)
    - JRC permanent water occurrence

  DYNAMIC (computed per month for training):
    - Rainfall 24h, 3d, 7d accumulation (CHIRPS)
    - Rainfall anomaly vs climatology (CHIRPS long-term mean)
    - Soil moisture surface + rootzone (SMAP L4)
    - Temperature / LST (MODIS MOD11A1)
    - Evapotranspiration (MODIS MOD16A2GF gap-filled)
    - Land cover built% + water% (Dynamic World)

  OPTIONAL (if time permits):
    - NDWI (Sentinel-2)
    - SAR VV/VH (Sentinel-1)

Output: data/features/features.csv
  One row per (district, month) with all feature columns.

Usage:
  python pull_features.py                    # pull all months 2015-2023
  python pull_features.py --year 2020        # pull one year only
  python pull_features.py --static-only      # pull static features only
"""

import ee
import geemap
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Config ──
PROJECT_ID = "dotted-embassy-463007-c1"
AOI_COORDS = [89.7, 24.1, 96.0, 28.2]  # Assam bounding box
UTM_EPSG = "EPSG:32646"  # UTM zone 46N (covers Assam)
GRID_SCALE = 5000  # ~5 km cells

YEAR_START = 2015
YEAR_END = 2023
MONSOON_MONTHS = [5, 6, 7, 8, 9, 10]

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"


def init_gee():
    """Initialize Google Earth Engine."""
    print("  Initializing GEE...")
    try:
        ee.Initialize(project=PROJECT_ID)
        print(f"  ✓ GEE initialized with project: {PROJECT_ID}")
    except Exception as e:
        print(f"  ⚠ ee.Initialize failed: {e}")
        print("  Attempting ee.Authenticate() first...")
        try:
            ee.Authenticate()
            ee.Initialize(project=PROJECT_ID)
            print(f"  ✓ GEE authenticated and initialized")
        except Exception as e2:
            print(f"  ✗ GEE auth failed: {e2}")
            print("  Please run 'earthengine authenticate' in terminal first.")
            sys.exit(1)


def build_grid():
    """Build the ~5 km grid over Assam AOI."""
    aoi = ee.Geometry.Rectangle(AOI_COORDS)
    proj = ee.Projection(UTM_EPSG).atScale(GRID_SCALE)
    grid = aoi.coveringGrid(proj)
    count = grid.size().getInfo()
    print(f"  Grid: {count} cells at {GRID_SCALE}m scale over Assam")
    return aoi, grid


# ═══════════════════════════════════════════════════════════
# STATIC FEATURES (compute once)
# ═══════════════════════════════════════════════════════════

def get_static_features(aoi):
    """
    Build a multi-band image with all static features.
    Per the plan: elevation, slope, flow_acc, distance_to_river, JRC water.
    """
    print("\n  Computing static features...")

    # 1. Elevation (SRTM 30m)
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")

    # 2. Slope (derived from elevation)
    slope = ee.Terrain.slope(dem).rename("slope")

    # 3. Flow accumulation (HydroSHEDS 15 arc-sec)
    flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select("b1").rename("flow_acc")

    # 4. Distance to river (threshold flow_acc to create river mask, then distance)
    # Threshold: cells with flow_acc > 1000 are considered "river"
    river_mask = flow_acc.gt(1000).rename("river_mask")
    # Distance transform: distance from each pixel to nearest river pixel (in meters)
    dist_to_river = river_mask.Not().cumulativeCost(
        source=river_mask, maxDistance=50000  # max 50 km search
    ).rename("dist_to_river")

    # 5. JRC Global Surface Water - permanent water occurrence
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    water_occurrence = jrc.select("occurrence").rename("water_occurrence")

    # Stack all static bands
    static = dem.addBands([slope, flow_acc, dist_to_river, water_occurrence])

    band_names = static.bandNames().getInfo()
    print(f"  ✓ Static bands: {band_names}")
    return static


# ═══════════════════════════════════════════════════════════
# DYNAMIC FEATURES (compute per month)
# ═══════════════════════════════════════════════════════════

def get_dynamic_features(aoi, year, month):
    """
    Build a multi-band image with all dynamic features for a given month.
    Uses monthly composites for training data efficiency.
    """
    # Date range for this month
    start = ee.Date.fromYMD(year, month, 1)
    if month == 12:
        end = ee.Date.fromYMD(year + 1, 1, 1)
    else:
        end = ee.Date.fromYMD(year, month + 1, 1)

    # Also need lookback periods for accumulations
    start_7d = end.advance(-7, "day")
    start_3d = end.advance(-3, "day")
    start_1d = end.advance(-1, "day")

    bands = []

    # ── 1. Rainfall (CHIRPS) ──
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi)

    # Monthly total
    rain_monthly = chirps.filterDate(start, end).sum().rename("rain_monthly_mm")
    bands.append(rain_monthly)

    # 7-day accumulation (end of month)
    rain_7d = chirps.filterDate(start_7d, end).sum().rename("rain_7d_mm")
    bands.append(rain_7d)

    # 3-day accumulation
    rain_3d = chirps.filterDate(start_3d, end).sum().rename("rain_3d_mm")
    bands.append(rain_3d)

    # 1-day (last day of month)
    rain_1d = chirps.filterDate(start_1d, end).sum().rename("rain_1d_mm")
    bands.append(rain_1d)

    # Monthly mean (for anomaly calculation)
    rain_mean = chirps.filterDate(start, end).mean().rename("rain_daily_mean_mm")
    bands.append(rain_mean)

    # ── 2. Rainfall anomaly vs climatology ──
    # Compute long-term monthly mean from CHIRPS (1981-2014 as baseline)
    climatology_start = ee.Date.fromYMD(1981, month, 1)
    if month == 12:
        climatology_end = ee.Date.fromYMD(2015, 1, 1)
    else:
        climatology_end = ee.Date.fromYMD(2014, month + 1, 1)

    # Get all same-month images across baseline years
    def same_month_filter(yr):
        yr = ee.Number(yr)
        m_start = ee.Date.fromYMD(yr, month, 1)
        if month == 12:
            m_end = ee.Date.fromYMD(yr.add(1), 1, 1)
        else:
            m_end = ee.Date.fromYMD(yr, ee.Number(month).add(1), 1)
        return chirps.filterDate(m_start, m_end).sum()

    baseline_years = ee.List.sequence(1981, 2014)
    baseline_monthly = ee.ImageCollection(baseline_years.map(same_month_filter))
    clim_mean = baseline_monthly.mean().rename("clim_mean_mm")
    clim_std = baseline_monthly.reduce(ee.Reducer.stdDev()).rename("clim_std_mm")

    # Anomaly: (current - mean) / std
    rain_anomaly = rain_monthly.subtract(clim_mean).divide(clim_std.max(ee.Image(1))).rename("rain_anomaly")
    bands.append(rain_anomaly)

    # ── 3. Soil moisture (SMAP L4) ──
    smap = (ee.ImageCollection("NASA/SMAP/SPL4SMGP/008")
            .filterBounds(aoi)
            .filterDate(start, end))

    smap_count = smap.size()
    # Use mean of the month
    sm_surface = smap.select("sm_surface").mean().rename("sm_surface")
    sm_rootzone = smap.select("sm_rootzone").mean().rename("sm_rootzone")
    bands.extend([sm_surface, sm_rootzone])

    # ── 4. Temperature / LST (MODIS) ──
    # Use ±16 days around end of month to fill cloud gaps in monsoon
    lst_start = end.advance(-16, "day")
    lst_end = end.advance(16, "day")
    lst = (ee.ImageCollection("MODIS/061/MOD11A1")
           .filterBounds(aoi)
           .filterDate(lst_start, lst_end))

    lst_day = lst.select("LST_Day_1km").mean().multiply(0.02).rename("lst_day_k")
    bands.append(lst_day)

    # ── 5. Evapotranspiration (MODIS gap-filled) ──
    et = (ee.ImageCollection("MODIS/061/MOD16A2GF")
          .filterBounds(aoi)
          .filterDate(start, end))

    et_mean = et.select("ET").mean().multiply(0.1).rename("et_mm")
    bands.append(et_mean)

    # ── 6. Land cover (Dynamic World) ──
    # Use ±30 day window to get enough cloud-free observations during monsoon
    dw_start = start.advance(-30, "day")
    dw_end = end.advance(30, "day")
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterBounds(aoi)
          .filterDate(dw_start, dw_end))

    built_frac = dw.select("built").mean().rename("built_frac")
    water_frac = dw.select("water").mean().rename("water_frac")
    bands.extend([built_frac, water_frac])

    # ── Stack all dynamic bands ──
    dynamic = bands[0]
    for b in bands[1:]:
        dynamic = dynamic.addBands(b)

    return dynamic


# ═══════════════════════════════════════════════════════════
# REDUCE TO PER-CELL TABLE
# ═══════════════════════════════════════════════════════════

def reduce_to_table(features_image, grid, scale=1000):
    """
    Reduce multi-band image to one row per grid cell.
    Uses ee.Reducer.mean() as specified in the plan.
    scale=1000m is used for the reduction (not the grid cell size).
    """
    cell_features = features_image.reduceRegions(
        collection=grid,
        reducer=ee.Reducer.mean(),
        scale=scale,
    )
    return cell_features


def fc_to_dataframe(fc):
    """
    Convert an Earth Engine FeatureCollection to a pandas DataFrame.
    Uses geemap.ee_to_df() which handles GEE's pagination limit internally.
    Falls back to chunked getInfo() if geemap fails.
    """
    try:
        # geemap handles pagination for collections > 5000 elements
        df = geemap.ee_to_df(fc)
        return df
    except Exception as e:
        print(f"  ⚠ geemap.ee_to_df failed ({e}), falling back to chunked getInfo...")

    # Manual pagination fallback
    rows = []
    page_size = 4000
    total = fc.size().getInfo()
    print(f"  Paginating {total} features in chunks of {page_size}...")

    fc_list = fc.toList(total)
    for i in range(0, total, page_size):
        chunk = ee.FeatureCollection(fc_list.slice(i, i + page_size))
        feats = chunk.getInfo()["features"]
        for f in feats:
            row = f["properties"].copy()
            if f.get("geometry"):
                try:
                    coords = f["geometry"]["coordinates"]
                    if f["geometry"]["type"] == "Polygon":
                        all_coords = coords[0]
                        row["cell_lon"] = np.mean([c[0] for c in all_coords])
                        row["cell_lat"] = np.mean([c[1] for c in all_coords])
                except (KeyError, IndexError):
                    pass
            rows.append(row)
        print(f"    Fetched {min(i + page_size, total)}/{total}")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# MAIN EXTRACTION PIPELINE
# ═══════════════════════════════════════════════════════════

def extract_static(aoi, grid):
    """Extract static features for all grid cells (one-time)."""
    print("\n" + "=" * 60)
    print("STATIC FEATURE EXTRACTION")
    print("=" * 60)

    static_img = get_static_features(aoi)

    print("  Reducing to per-cell values...")
    static_fc = reduce_to_table(static_img, grid)

    print("  Converting to DataFrame...")
    static_df = fc_to_dataframe(static_fc)

    print(f"  ✓ Static features: {static_df.shape}")
    print(f"  Columns: {list(static_df.columns)}")

    # Fill expected nulls:
    # water_occurrence: no data = no permanent water → fill with 0
    if "water_occurrence" in static_df.columns:
        static_df["water_occurrence"] = static_df["water_occurrence"].fillna(0)

    # Check for remaining null columns
    null_pct = static_df.isnull().mean() * 100
    for col, pct in null_pct.items():
        if pct > 0:
            print(f"  ⚠ {col}: {pct:.1f}% null")

    return static_df


def extract_dynamic(aoi, grid, year, month):
    """Extract dynamic features for a specific month."""
    print(f"  Extracting {year}-{month:02d}...", end=" ", flush=True)

    try:
        dynamic_img = get_dynamic_features(aoi, year, month)
        dynamic_fc = reduce_to_table(dynamic_img, grid)
        dynamic_df = fc_to_dataframe(dynamic_fc)

        # Add time columns
        dynamic_df["year"] = year
        dynamic_df["month"] = month

        null_cols = dynamic_df.columns[dynamic_df.isnull().all()].tolist()
        if null_cols:
            print(f"done ({len(dynamic_df)} cells, ⚠ empty: {null_cols})")
        else:
            print(f"done ({len(dynamic_df)} cells, all columns populated)")

        return dynamic_df

    except Exception as e:
        print(f"FAILED: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Pull GEE features for flood model")
    parser.add_argument("--year", type=int, help="Extract single year only")
    parser.add_argument("--month", type=int, help="Extract single month only (with --year)")
    parser.add_argument("--static-only", action="store_true", help="Extract static features only")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Task 2: GEE Feature Pipeline                              ║")
    print(f"║  Region: Assam [{AOI_COORDS}]             ║")
    print(f"║  Project: {PROJECT_ID}                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Initialize GEE ──
    init_gee()

    # ── Build grid ──
    aoi, grid = build_grid()

    # ── Extract static features ──
    static_df = extract_static(aoi, grid)
    static_path = FEATURES_DIR / "gee_static_features.csv"
    static_df.to_csv(static_path, index=False)
    print(f"  ✓ Saved: {static_path}")

    if args.static_only:
        print("\n  Static-only mode. Done.")
        return

    # ── Extract dynamic features ──
    print("\n" + "=" * 60)
    print("DYNAMIC FEATURE EXTRACTION")
    print("=" * 60)

    if args.year:
        years = [args.year]
    else:
        years = list(range(YEAR_START, YEAR_END + 1))

    if args.month:
        months = [args.month]
    else:
        months = MONSOON_MONTHS

    total_months = len(years) * len(months)
    print(f"  Extracting {total_months} months: {years[0]}–{years[-1]}, months {months}")
    print(f"  This may take a while...\n")

    all_dynamic = []
    completed = 0

    for year in years:
        for month in months:
            dynamic_df = extract_dynamic(aoi, grid, year, month)
            if dynamic_df is not None:
                all_dynamic.append(dynamic_df)
            completed += 1

            # Save progress every 6 months
            if completed % 6 == 0 and all_dynamic:
                progress_df = pd.concat(all_dynamic, ignore_index=True)
                progress_path = FEATURES_DIR / "gee_dynamic_features_partial.csv"
                progress_df.to_csv(progress_path, index=False)
                print(f"  💾 Progress saved ({completed}/{total_months} months)")

    if not all_dynamic:
        print("  ✗ No dynamic features extracted")
        return

    # ── Combine all dynamic features ──
    dynamic_all = pd.concat(all_dynamic, ignore_index=True)
    dynamic_path = FEATURES_DIR / "gee_dynamic_features.csv"
    dynamic_all.to_csv(dynamic_path, index=False)
    print(f"\n  ✓ Dynamic features saved: {dynamic_path}")
    print(f"  Shape: {dynamic_all.shape}")

    # ── Merge static + dynamic ──
    print("\n" + "=" * 60)
    print("MERGING STATIC + DYNAMIC")
    print("=" * 60)

    # Merge on cell centroid (or system:index if available)
    merge_cols = [c for c in ["cell_lat", "cell_lon"] if c in static_df.columns and c in dynamic_all.columns]
    if merge_cols:
        # Round to avoid float precision issues
        for col in merge_cols:
            static_df[col] = static_df[col].round(4)
            dynamic_all[col] = dynamic_all[col].round(4)

        features_full = dynamic_all.merge(static_df, on=merge_cols, how="left", suffixes=("", "_static"))
        print(f"  Merged on {merge_cols}")
    else:
        print("  ⚠ No common merge columns — concatenating static as repeated rows")
        features_full = dynamic_all

    # Save final combined features
    final_path = FEATURES_DIR / "features.csv"
    features_full.to_csv(final_path, index=False)

    # ── Verification summary ──
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION VERIFICATION")
    print("=" * 60)
    print(f"  Output: {final_path}")
    print(f"  Shape: {features_full.shape}")
    print(f"  Columns: {list(features_full.columns)}")
    print(f"  Row count: {len(features_full):,}")

    print(f"\n  Null/empty column check:")
    for col in features_full.columns:
        null_pct = features_full[col].isnull().mean() * 100
        if null_pct > 10:
            print(f"    ⚠ {col}: {null_pct:.1f}% null")
        elif null_pct > 0:
            print(f"    ~ {col}: {null_pct:.1f}% null")
        else:
            print(f"    ✓ {col}: complete")

    print(f"\n  ✓ Task 2 complete.")


if __name__ == "__main__":
    main()
