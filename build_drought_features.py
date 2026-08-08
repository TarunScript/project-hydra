"""
build_drought_features.py
=========================
Feature pipeline for the drought risk-index model.
Builds a ~5 km grid over demo regions and pulls per-cell mean values from
Google Earth Engine datasets.

SCOPE: Per-grid-cell tabular features for XGBoost/RF drought risk model.
       NOT pixel-level image segmentation.

Demo regions: Marathwada (MH), Bundelkhand (UP/MP), Rayalaseema (AP)

GEE Dataset IDs used (from drought_datasets_reference.md):
  - CHIRPS daily rainfall: UCSB-CHG/CHIRPS/DAILY
  - SMAP L4 soil moisture: NASA/SMAP/SPL4SMGP/008
  - MODIS NDVI: MODIS/061/MOD13Q1
  - MODIS LST: MODIS/061/MOD11A1
  - MODIS ET (gap-filled): MODIS/061/MOD16A2GF  (NOT MOD16A2)
  - Dynamic World: GOOGLE/DYNAMICWORLD/V1
"""

import ee
import pandas as pd
import numpy as np
import json
import os
import sys
import time
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEE_PROJECT = 'dotted-embassy-463007-c1'
GRID_SCALE_M = 5000  # ~5 km grid cells
REDUCE_SCALE = 500   # scale for reduceRegions (m) — balance speed vs accuracy
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Demo regions: name -> [west, south, east, north] bounding box
REGIONS = {
    'marathwada': {
        'bbox': [75.0, 17.5, 78.5, 20.5],
        'utm_epsg': 'EPSG:32643',   # UTM 43N
        'description': 'Marathwada, Maharashtra'
    },
    'bundelkhand': {
        'bbox': [78.0, 24.0, 81.5, 26.5],
        'utm_epsg': 'EPSG:32644',   # UTM 44N
        'description': 'Bundelkhand, UP/MP'
    },
    'rayalaseema': {
        'bbox': [77.0, 14.5, 80.0, 16.5],
        'utm_epsg': 'EPSG:32644',   # UTM 44N
        'description': 'Rayalaseema, AP'
    },
}

# Date configuration — use a recent date with good data coverage
# We pull features for multiple dates to build a training dataset
# CHIRPS has ~5 day latency, SMAP ~3 days, MODIS NDVI ~16 days
TARGET_DATE = '2025-07-01'  # primary target date
# For training we pull features at multiple time snapshots
# Reduced to 8 strategic dates (all seasons × 2 years) for hackathon speed
# Full 24-date run would take ~2.5 hours on GEE
TRAINING_DATES = [
    '2023-08-01',   # monsoon peak
    '2023-11-01',   # post-monsoon / rabi start
    '2024-02-01',   # winter / dry season
    '2024-05-01',   # pre-monsoon / peak heat
    '2024-08-01',   # monsoon peak
    '2024-11-01',   # post-monsoon
    '2025-02-01',   # winter dry
    '2025-06-01',   # early monsoon
]


def initialize_gee():
    """Initialize Google Earth Engine."""
    ee.Initialize(project=GEE_PROJECT)
    # Quick sanity check
    img = ee.Image('USGS/SRTMGL1_003')
    _ = img.getInfo()
    print("[OK] GEE initialized successfully")


def build_grid(region_config):
    """
    Build a grid of ~5 km cells over the region's bounding box.
    Returns an ee.FeatureCollection of grid cells.
    """
    bbox = region_config['bbox']
    aoi = ee.Geometry.Rectangle(bbox)
    proj = ee.Projection(region_config['utm_epsg']).atScale(GRID_SCALE_M)
    grid = aoi.coveringGrid(proj)

    # Add centroid lat/lon to each cell
    def add_centroid(feature):
        centroid = feature.geometry().centroid(1)
        return feature.set({
            'centroid_lon': centroid.coordinates().get(0),
            'centroid_lat': centroid.coordinates().get(1),
        })

    grid = grid.map(add_centroid)
    return grid, aoi


def get_chirps_rainfall_features(target_date_str, aoi):
    """
    CHIRPS daily rainfall features:
    - 7, 30, 60, 90-day cumulative rainfall
    - Rainfall deficit vs. climatology (long-term mean for the same period)

    GEE ID: UCSB-CHG/CHIRPS/DAILY
    """
    target_date = ee.Date(target_date_str)

    features = {}
    windows = {'7d': 7, '30d': 30, '60d': 60, '90d': 90}

    for suffix, days in windows.items():
        start = target_date.advance(-days, 'day')
        rain = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                .filterDate(start, target_date)
                .filterBounds(aoi)
                .select('precipitation'))

        # Current cumulative rainfall
        rain_sum = rain.sum().rename(f'rain_{suffix}_mm')
        features[f'rain_{suffix}_mm'] = rain_sum

        # Climatology: same day-of-year window, averaged over 2000-2022
        # This gives us a long-term baseline to compute deficit
        clim_images = []
        for year in range(2000, 2023):
            yr_date = ee.Date(f'{year}-01-01').advance(
                target_date.getRelative('day', 'year'), 'day'
            )
            yr_start = yr_date.advance(-days, 'day')
            yr_rain = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                       .filterDate(yr_start, yr_date)
                       .filterBounds(aoi)
                       .select('precipitation')
                       .sum())
            clim_images.append(yr_rain)

        clim_mean = ee.ImageCollection(clim_images).mean().rename(f'rain_{suffix}_clim_mm')
        deficit = rain_sum.subtract(clim_mean).rename(f'rain_{suffix}_deficit_mm')
        features[f'rain_{suffix}_deficit_mm'] = deficit

    return features


def get_chirps_dry_spell(target_date_str, aoi):
    """
    Dry-spell duration: consecutive dry days (precip < 1mm) ending at target date.
    Derived from CHIRPS daily. Look back up to 120 days.

    GEE ID: UCSB-CHG/CHIRPS/DAILY
    """
    target_date = ee.Date(target_date_str)
    lookback = 120
    start = target_date.advance(-lookback, 'day')

    daily = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
             .filterDate(start, target_date)
             .filterBounds(aoi)
             .select('precipitation')
             .sort('system:time_start', False))  # newest first

    # Binary: 1 if dry (< 1mm), 0 if wet
    def mark_dry(img):
        return img.lt(1).rename('is_dry').copyProperties(img, ['system:time_start'])

    dry_days = daily.map(mark_dry)

    # Compute consecutive dry days from most recent day backwards
    # Using an iterative approach: accumulate until first wet day
    dry_list = dry_days.toList(lookback)
    n_images = dry_list.size()

    def accumulate(current, prev):
        prev = ee.Dictionary(prev)
        current = ee.Image(current)
        still_dry = ee.Image(prev.get('still_dry'))
        count = ee.Image(prev.get('count'))
        # If still in dry streak AND current day is dry, increment
        is_dry = current.And(still_dry)
        new_count = count.add(is_dry)
        return ee.Dictionary({
            'still_dry': is_dry,
            'count': new_count,
        })

    first_img = ee.Image(dry_list.get(0))
    initial = ee.Dictionary({
        'still_dry': first_img,
        'count': first_img,
    })

    result = ee.Dictionary(
        dry_list.slice(1).iterate(accumulate, initial)
    )
    dry_spell = ee.Image(result.get('count')).rename('dry_spell_days')
    return {'dry_spell_days': dry_spell}


def get_smap_soil_moisture(target_date_str, aoi):
    """
    NASA SMAP L4 surface + root-zone soil moisture.
    Most recent available composite near target date.

    GEE ID: NASA/SMAP/SPL4SMGP/008
    Bands: sm_surface, sm_rootzone
    """
    target_date = ee.Date(target_date_str)
    # SMAP has ~3 day latency; look back up to 10 days for latest
    start = target_date.advance(-10, 'day')

    smap = (ee.ImageCollection('NASA/SMAP/SPL4SMGP/008')
            .filterDate(start, target_date)
            .filterBounds(aoi)
            .select(['sm_surface', 'sm_rootzone']))

    # Take mean of available images in the window
    sm = smap.mean()
    return {
        'sm_surface': sm.select('sm_surface'),
        'sm_rootzone': sm.select('sm_rootzone'),
    }


def get_ndvi_features(target_date_str, aoi):
    """
    MODIS NDVI: current value + anomaly vs. climatology.

    GEE ID: MODIS/061/MOD13Q1
    Band: NDVI (scale factor: 0.0001)
    Resolution: 250m, 16-day composite
    """
    target_date = ee.Date(target_date_str)
    # MOD13Q1 is 16-day composite; look back 32 days for latest
    start = target_date.advance(-32, 'day')

    ndvi_col = (ee.ImageCollection('MODIS/061/MOD13Q1')
                .filterDate(start, target_date)
                .filterBounds(aoi)
                .select('NDVI'))

    # Current NDVI (most recent composite, scaled)
    current_ndvi = ndvi_col.mean().multiply(0.0001).rename('ndvi_current')

    # Climatology: same DOY window, 2000-2022
    doy = target_date.getRelative('day', 'year')
    clim_images = []
    for year in range(2000, 2023):
        yr_date = ee.Date(f'{year}-01-01').advance(doy, 'day')
        yr_start = yr_date.advance(-32, 'day')
        yr_ndvi = (ee.ImageCollection('MODIS/061/MOD13Q1')
                   .filterDate(yr_start, yr_date)
                   .filterBounds(aoi)
                   .select('NDVI')
                   .mean()
                   .multiply(0.0001))
        clim_images.append(yr_ndvi)

    clim_col = ee.ImageCollection(clim_images)
    ndvi_clim_mean = clim_col.mean().rename('ndvi_clim_mean')
    ndvi_clim_std = clim_col.reduce(ee.Reducer.stdDev()).rename('ndvi_clim_std')

    # Anomaly = (current - mean) / std
    ndvi_anomaly = (current_ndvi.subtract(ndvi_clim_mean)
                    .divide(ndvi_clim_std.max(0.001))  # avoid div-by-zero
                    .rename('ndvi_anomaly'))

    return {
        'ndvi_current': current_ndvi,
        'ndvi_anomaly': ndvi_anomaly,
    }


def get_lst_features(target_date_str, aoi):
    """
    MODIS Land Surface Temperature: current value + anomaly.

    GEE ID: MODIS/061/MOD11A1
    Band: LST_Day_1km (scale factor: 0.02, units: Kelvin)
    """
    target_date = ee.Date(target_date_str)
    start = target_date.advance(-8, 'day')

    lst_col = (ee.ImageCollection('MODIS/061/MOD11A1')
               .filterDate(start, target_date)
               .filterBounds(aoi)
               .select('LST_Day_1km'))

    # Current LST (mean of last 8 days, convert to Celsius)
    current_lst = lst_col.mean().multiply(0.02).subtract(273.15).rename('lst_current_c')

    # Climatology: same DOY window, 2003-2022
    doy = target_date.getRelative('day', 'year')
    clim_images = []
    for year in range(2003, 2023):
        yr_date = ee.Date(f'{year}-01-01').advance(doy, 'day')
        yr_start = yr_date.advance(-8, 'day')
        yr_lst = (ee.ImageCollection('MODIS/061/MOD11A1')
                  .filterDate(yr_start, yr_date)
                  .filterBounds(aoi)
                  .select('LST_Day_1km')
                  .mean()
                  .multiply(0.02)
                  .subtract(273.15))
        clim_images.append(yr_lst)

    clim_col = ee.ImageCollection(clim_images)
    lst_clim_mean = clim_col.mean().rename('lst_clim_mean_c')
    lst_anomaly = current_lst.subtract(lst_clim_mean).rename('lst_anomaly_c')

    return {
        'lst_current_c': current_lst,
        'lst_anomaly_c': lst_anomaly,
    }


def get_et_features(target_date_str, aoi):
    """
    MODIS Evapotranspiration (gap-filled).

    GEE ID: MODIS/061/MOD16A2GF  (NOT MOD16A2 which only covers 2021+)
    Band: ET (scale factor: 0.1, units: kg/m²/8day)
    Resolution: 500m, 8-day composite
    """
    target_date = ee.Date(target_date_str)
    start = target_date.advance(-16, 'day')

    et_col = (ee.ImageCollection('MODIS/061/MOD16A2GF')
              .filterDate(start, target_date)
              .filterBounds(aoi)
              .select('ET'))

    # Current ET (mean of recent composites, scaled)
    current_et = et_col.mean().multiply(0.1).rename('et_current_kg_m2')
    return {'et_current_kg_m2': current_et}


def get_landcover_features(target_date_str, aoi):
    """
    Dynamic World land cover and urban/built-up percentage.

    GEE ID: GOOGLE/DYNAMICWORLD/V1
    Resolution: 10m, updates every 2-5 days
    """
    target_date = ee.Date(target_date_str)
    start = target_date.advance(-30, 'day')

    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterDate(start, target_date)
          .filterBounds(aoi))

    # Built-up fraction (mean of 'built' probability band)
    built_frac = dw.select('built').mean().rename('urban_built_frac')

    # Dominant land cover class (mode of 'label' band)
    lc_mode = dw.select('label').mode().rename('landcover_class')

    # Cropland fraction
    crop_frac = dw.select('crops').mean().rename('cropland_frac')

    # Bare ground fraction
    bare_frac = dw.select('bare').mean().rename('bare_ground_frac')

    return {
        'urban_built_frac': built_frac,
        'landcover_class': lc_mode,
        'cropland_frac': crop_frac,
        'bare_ground_frac': bare_frac,
    }


def build_features_for_date(target_date_str, region_name, region_config):
    """
    Build all features for a single date and region.
    Returns a pandas DataFrame with one row per grid cell.
    """
    print(f"\n{'='*60}")
    print(f"Building features: {region_name} | {target_date_str}")
    print(f"{'='*60}")

    grid, aoi = build_grid(region_config)
    grid_size = grid.size().getInfo()
    print(f"  Grid cells: {grid_size}")

    # Collect all feature bands
    all_features = {}

    print("  [1/7] CHIRPS rainfall (7/30/60/90-day + deficit)...")
    try:
        chirps = get_chirps_rainfall_features(target_date_str, aoi)
        all_features.update(chirps)
        print("         OK")
    except Exception as e:
        print(f"         WARNING: {e}")

    print("  [2/7] CHIRPS dry-spell duration...")
    try:
        dry = get_chirps_dry_spell(target_date_str, aoi)
        all_features.update(dry)
        print("         OK")
    except Exception as e:
        print(f"         WARNING: {e}")

    print("  [3/7] SMAP soil moisture...")
    try:
        smap = get_smap_soil_moisture(target_date_str, aoi)
        all_features.update(smap)
        print("         OK")
    except Exception as e:
        print(f"         WARNING: {e}")

    print("  [4/7] MODIS NDVI + anomaly...")
    try:
        ndvi = get_ndvi_features(target_date_str, aoi)
        all_features.update(ndvi)
        print("         OK")
    except Exception as e:
        print(f"         WARNING: {e}")

    print("  [5/7] MODIS LST + anomaly...")
    try:
        lst = get_lst_features(target_date_str, aoi)
        all_features.update(lst)
        print("         OK")
    except Exception as e:
        print(f"         WARNING: {e}")

    print("  [6/7] MODIS ET (MOD16A2GF)...")
    try:
        et = get_et_features(target_date_str, aoi)
        all_features.update(et)
        print("         OK")
    except Exception as e:
        print(f"         WARNING: {e}")

    print("  [7/7] Dynamic World land cover...")
    try:
        lc = get_landcover_features(target_date_str, aoi)
        all_features.update(lc)
        print("         OK")
    except Exception as e:
        print(f"         WARNING: {e}")

    # Stack all features into a single multi-band image
    print(f"  Stacking {len(all_features)} feature bands...")
    band_list = list(all_features.values())
    stacked = band_list[0]
    for band in band_list[1:]:
        stacked = stacked.addBands(band)

    # Reduce to one row per grid cell, in chunks to avoid GEE's 5000-element limit
    CHUNK_SIZE = 4000
    print(f"  Reducing to per-cell values (scale={REDUCE_SCALE}m, chunks of {CHUNK_SIZE})...")

    grid_list = grid.toList(grid_size + 1)
    all_rows = []
    n_chunks = (grid_size + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, grid_size)
        chunk_size = end_idx - start_idx
        print(f"    Chunk {chunk_idx + 1}/{n_chunks} (cells {start_idx}-{end_idx-1})...")

        chunk_fc = ee.FeatureCollection(grid_list.slice(start_idx, end_idx))

        cell_features = stacked.reduceRegions(
            collection=chunk_fc,
            reducer=ee.Reducer.mean(),
            scale=REDUCE_SCALE,
        )

        try:
            features_info = cell_features.getInfo()
            chunk_features = features_info['features']
        except Exception as e:
            print(f"    WARNING: chunk failed at scale={REDUCE_SCALE} ({e})")
            print(f"    Retrying chunk at scale=1000m...")
            cell_features = stacked.reduceRegions(
                collection=chunk_fc,
                reducer=ee.Reducer.mean(),
                scale=1000,
            )
            chunk_features = cell_features.getInfo()['features']

        for feat in chunk_features:
            all_rows.append(feat['properties'])
        print(f"    Got {len(chunk_features)} rows")
        time.sleep(1)  # brief pause between chunks

    print(f"  Total downloaded: {len(all_rows)} rows")

    df = pd.DataFrame(all_rows)

    # Add metadata columns
    df['region'] = region_name
    df['date'] = target_date_str

    # Rename columns for consistency
    col_renames = {
        'centroid_lon': 'lon',
        'centroid_lat': 'lat',
    }
    df = df.rename(columns=col_renames)

    # Drop GEE internal columns if present
    drop_cols = [c for c in df.columns if c.startswith('system:')]
    df = df.drop(columns=drop_cols, errors='ignore')

    print(f"  Result: {len(df)} rows x {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    return df


def build_all_features():
    """
    Build features for all regions and all training dates.
    Saves per-region CSVs and a combined CSV.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_dfs = []

    for region_name, region_config in REGIONS.items():
        region_dfs = []

        for date_str in TRAINING_DATES:
            try:
                df = build_features_for_date(date_str, region_name, region_config)
                region_dfs.append(df)

                # Save per-date file too
                date_safe = date_str.replace('-', '')
                per_date_path = os.path.join(
                    OUTPUT_DIR, f'drought_features_{region_name}_{date_safe}.csv'
                )
                df.to_csv(per_date_path, index=False)
                print(f"  Saved: {per_date_path}")

                # Rate limiting — be nice to GEE
                time.sleep(2)

            except Exception as e:
                print(f"  ERROR for {region_name}/{date_str}: {e}")
                print(f"  Skipping this date and continuing...")
                continue

        if region_dfs:
            region_combined = pd.concat(region_dfs, ignore_index=True)
            region_path = os.path.join(OUTPUT_DIR, f'drought_features_{region_name}_all.csv')
            region_combined.to_csv(region_path, index=False)
            print(f"\n  Saved region combined: {region_path}")
            print(f"  Total rows: {len(region_combined)}")
            all_dfs.append(region_combined)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, 'drought_features_all_regions.csv')
        combined.to_csv(combined_path, index=False)
        print(f"\n{'='*60}")
        print(f"COMBINED FEATURE TABLE")
        print(f"{'='*60}")
        print(f"  File: {combined_path}")
        print(f"  Rows: {len(combined)}")
        print(f"  Columns: {len(combined.columns)}")
        print(f"  Regions: {combined['region'].unique().tolist()}")
        print(f"  Dates: {combined['date'].nunique()} unique dates")
        print(f"\n  Sample rows:")
        print(combined.head(3).to_string())
        print(f"\n  Feature statistics:")
        feature_cols = [c for c in combined.columns
                        if c not in ['lon', 'lat', 'region', 'date', 'landcover_class']]
        print(combined[feature_cols].describe().to_string())
        return combined

    return None


# ---------------------------------------------------------------------------
# Quick single-date test mode
# ---------------------------------------------------------------------------
def quick_test(region='marathwada', date='2025-07-01'):
    """
    Quick test: build features for a single region and date.
    Use this to verify everything works before the full run.
    """
    print(f"QUICK TEST: {region} on {date}")
    df = build_features_for_date(date, region, REGIONS[region])
    out_path = os.path.join(OUTPUT_DIR, f'drought_features_{region}_test.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved test output: {out_path}")
    print(f"\nSample data (first 5 rows):")
    print(df.head().to_string())
    print(f"\nColumn types:")
    print(df.dtypes.to_string())
    return df


if __name__ == '__main__':
    initialize_gee()

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Quick test mode: single region, single date
        region = sys.argv[2] if len(sys.argv) > 2 else 'marathwada'
        date = sys.argv[3] if len(sys.argv) > 3 else '2025-07-01'
        quick_test(region, date)
    else:
        # Full run: all regions, all training dates
        build_all_features()
