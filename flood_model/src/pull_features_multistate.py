#!/usr/bin/env python3
"""
pull_features_multistate.py — GEE Feature Extraction for Bihar, West Bengal, Odisha

Runs the same CHIRPS + SMAP + MODIS + DynamicWorld extraction pipeline
as pull_features.py (Assam), but for the three new states.

States covered:
  - BIHAR        (approx 83°E–88°E, 24°N–27.5°N)
  - WEST_BENGAL  (approx 85.8°E–89.9°E, 21.5°N–27.2°N)
  - ODISHA       (approx 81.4°E–87.5°E, 17.8°N–22.6°N)

Outputs (per state):
  flood_model/data/features/<state_lower>/features.csv
  flood_model/data/features/<state_lower>/gee_static_features.csv

Usage:
  python3 flood_model/src/pull_features_multistate.py
  python3 flood_model/src/pull_features_multistate.py --state bihar
"""

import ee
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# ── Auth ──
PROJECT_ID = "dotted-embassy-463007-c1"
try:
    ee.Initialize(project=PROJECT_ID)
    print(f"  ✓ GEE initialized: {PROJECT_ID}")
except Exception:
    try:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
    except Exception as e:
        print(f"  ✗ GEE auth failed: {e}")
        sys.exit(1)

# ── Config ──
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"

MONTHS = list(range(5, 11))   # May–Oct (monsoon)
YEARS  = list(range(2015, 2024))

STATES = {
    "bihar": {
        "name": "BIHAR",
        "bbox": [83.0, 24.0, 88.5, 27.5],   # [west, south, east, north]
        "grid_scale": 5000,                   # 5 km
    },
    "west_bengal": {
        "name": "WEST_BENGAL",
        "bbox": [85.8, 21.5, 89.9, 27.2],
        "grid_scale": 5000,
    },
    "odisha": {
        "name": "ODISHA",
        "bbox": [81.4, 17.8, 87.5, 22.6],
        "grid_scale": 5000,
    },
}


# ────────────────────────────────────────────────────────
# GEE Helper functions (identical to pull_features.py)
# ────────────────────────────────────────────────────────

def get_monthly_rainfall_stats(roi, year, month):
    """CHIRPS daily rainfall → monthly stats + anomaly."""
    start = ee.Date.fromYMD(year, month, 1)
    end   = start.advance(1, "month")
    daily = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
             .filterDate(start, end)
             .filterBounds(roi)
             .select("precipitation"))

    monthly_mm  = daily.sum().rename("rain_monthly_mm")
    daily_mean  = daily.mean().rename("rain_daily_mean_mm")

    # 7/3/1-day rolling max
    end_date = datetime(year, month, 1) + timedelta(days=31)
    end_date = end_date.replace(day=1) - timedelta(days=1)
    window_start_7 = ee.Date.fromYMD(year, month, max(1, end_date.day - 6))
    window_start_3 = ee.Date.fromYMD(year, month, max(1, end_date.day - 2))
    window_start_1 = ee.Date.fromYMD(year, month, end_date.day)

    rain_7d = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
               .filterDate(window_start_7, end).sum().rename("rain_7d_mm"))
    rain_3d = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
               .filterDate(window_start_3, end).sum().rename("rain_3d_mm"))
    rain_1d = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
               .filterDate(window_start_1, end).sum().rename("rain_1d_mm"))

    # Climatological mean (same month, 2000–2014)
    clim = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filter(ee.Filter.calendarRange(2000, 2014, "year"))
            .filter(ee.Filter.calendarRange(month, month, "month"))
            .mean().rename("clim_mean"))

    anomaly = monthly_mm.subtract(clim).divide(clim.add(1)).rename("rain_anomaly")

    return ee.Image.cat([monthly_mm, daily_mean, rain_7d, rain_3d, rain_1d, anomaly])


def get_monthly_soil_moisture(roi, year, month):
    """SMAP L4 soil moisture (surface + root-zone)."""
    start = ee.Date.fromYMD(year, month, 1)
    end   = start.advance(1, "month")
    smap  = (ee.ImageCollection("NASA/SMAP/SPL4SMGP/007")
             .filterDate(start, end)
             .filterBounds(roi)
             .select(["sm_surface", "sm_rootzone"])
             .mean())
    return smap


def get_monthly_lst(roi, year, month):
    """MODIS Terra LST — with ±16 day buffer for cloud gaps."""
    d1 = datetime(year, month, 1)
    buf_start = (d1 - timedelta(days=16)).strftime("%Y-%m-%d")
    buf_end   = (d1 + timedelta(days=31 + 16)).strftime("%Y-%m-%d")
    lst = (ee.ImageCollection("MODIS/061/MOD11A2")
           .filterDate(buf_start, buf_end)
           .filterBounds(roi)
           .select("LST_Day_1km")
           .mean()
           .multiply(0.02)
           .rename("lst_day_k"))
    return lst


def get_monthly_et(roi, year, month):
    """MODIS ET (MOD16A2)."""
    start = ee.Date.fromYMD(year, month, 1)
    end   = start.advance(1, "month")
    et    = (ee.ImageCollection("MODIS/061/MOD16A2")
             .filterDate(start, end)
             .filterBounds(roi)
             .select("ET")
             .mean()
             .multiply(0.1)
             .rename("et_mm"))
    return et


def get_dynamic_world_fractions(roi, year, month):
    """DynamicWorld built/water fractions with ±30 day buffer."""
    d1 = datetime(year, month, 1)
    buf_start = (d1 - timedelta(days=30)).strftime("%Y-%m-%d")
    buf_end   = (d1 + timedelta(days=31 + 30)).strftime("%Y-%m-%d")
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterDate(buf_start, buf_end)
          .filterBounds(roi)
          .select(["built", "water"])
          .mean()
          .rename(["built_frac", "water_frac"]))
    return dw


def get_static_features(roi):
    """Elevation, slope, flow accumulation, dist_to_river, water occurrence."""
    dem   = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    flow  = ee.Image("WWF/HydroSHEDS/15ACC").select("b1").rename("flow_acc")

    # Distance to rivers (JRC occurrence > 50%)
    jrc   = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    river = jrc.gte(50).selfMask()
    dist  = river.fastDistanceTransform(512).sqrt().multiply(30).rename("dist_to_river")
    water_occ = jrc.rename("water_occurrence")

    return ee.Image.cat([dem, slope, flow, dist, water_occ])


# ────────────────────────────────────────────────────────
# Core extraction per state
# ────────────────────────────────────────────────────────

def extract_state_features(state_key: str):
    cfg   = STATES[state_key]
    name  = cfg["name"]
    bbox  = cfg["bbox"]
    scale = cfg["grid_scale"]

    out_dir = FEATURES_DIR / state_key
    out_dir.mkdir(parents=True, exist_ok=True)

    roi = ee.Geometry.Rectangle(bbox)

    print(f"\n{'='*60}")
    print(f"  Extracting: {name}")
    print(f"  BBox: {bbox}  |  Scale: {scale}m")
    print(f"{'='*60}")

    # ── Build grid ──
    proj    = ee.Projection("EPSG:32646").atScale(scale)
    grid_fc = roi.coveringGrid(proj)
    n_cells = grid_fc.size().getInfo()
    print(f"  Grid cells: {n_cells:,}")

    # ── Static features (once) ──
    static_out = out_dir / "gee_static_features.csv"
    if static_out.exists():
        print(f"  Static features already cached → {static_out}")
    else:
        print("  Extracting static features...")
        static_img = get_static_features(roi)
        static_fc  = static_img.reduceRegions(
            collection=grid_fc,
            reducer=ee.Reducer.mean(),
            scale=scale,
            crs=proj,
        )

        def add_centroid(f):
            c = f.geometry().centroid(1)
            return f.set({
                "cell_lon": c.coordinates().get(0),
                "cell_lat": c.coordinates().get(1),
            })
        static_fc = static_fc.map(add_centroid)

        rows = _paginate_fc(static_fc, n_cells, label="static")
        df   = pd.DataFrame(rows)
        _clean_and_save(df, static_out)
        print(f"  ✓ Static features saved: {static_out}")

    # ── Dynamic features (per month/year) ──
    dyn_out = out_dir / "features.csv"
    done_log = out_dir / ".done_months.txt"
    done_set = set()
    if done_log.exists():
        done_set = set(done_log.read_text().strip().splitlines())

    all_rows = []
    total    = len(YEARS) * len(MONTHS)
    done_cnt = 0

    for year in YEARS:
        for month in MONTHS:
            tag = f"{year}-{month:02d}"
            if tag in done_set:
                done_cnt += 1
                continue

            done_cnt += 1
            print(f"  [{done_cnt}/{total}] {tag}...", end=" ", flush=True)

            try:
                img = ee.Image.cat([
                    get_monthly_rainfall_stats(roi, year, month),
                    get_monthly_soil_moisture(roi, year, month),
                    get_monthly_lst(roi, year, month),
                    get_monthly_et(roi, year, month),
                    get_dynamic_world_fractions(roi, year, month),
                ])

                fc = img.reduceRegions(
                    collection=grid_fc,
                    reducer=ee.Reducer.mean(),
                    scale=scale,
                    crs=proj,
                )

                def add_meta(f):
                    c = f.geometry().centroid(1)
                    return f.set({
                        "year": year, "month": month,
                        "cell_lon": c.coordinates().get(0),
                        "cell_lat": c.coordinates().get(1),
                    })
                fc = fc.map(add_meta)

                rows = _paginate_fc(fc, n_cells, label=tag)
                all_rows.extend(rows)

                # Checkpoint: append to CSV every month (skip empty chunks)
                if len(rows) > 0:
                    df_chunk = pd.DataFrame(rows)
                    df_chunk = _rename_cols(df_chunk)
                    # Drop GEE system columns
                    drop_cols = [c for c in df_chunk.columns if c.startswith(".geo") or c == "system:index"]
                    df_chunk = df_chunk.drop(columns=drop_cols, errors="ignore")
                    # Enforce canonical column order (alphabetical) so every chunk matches header
                    df_chunk = df_chunk.reindex(sorted(df_chunk.columns), axis=1)
                    write_header = not dyn_out.exists() or dyn_out.stat().st_size < 10
                    df_chunk.to_csv(dyn_out, mode="a", header=write_header, index=False)

                # Mark as done
                with open(done_log, "a") as f:
                    f.write(tag + "\n")

                print("done")

            except Exception as e:
                print(f"⚠ FAILED: {e}")

    print(f"\n  ✓ {name} extraction complete → {dyn_out}")
    return dyn_out


def _paginate_fc(fc, n_cells, label, chunk=4000):
    """Chunk getInfo to avoid GEE 5000-feature limit."""
    rows = []
    fetched = 0
    while fetched < n_cells:
        try:
            chunk_fc  = fc.toList(chunk, fetched)
            chunk_info = ee.FeatureCollection(chunk_fc).getInfo()
            for feat in chunk_info["features"]:
                rows.append(feat["properties"])
            fetched += len(chunk_info["features"])
            if fetched % 4000 == 0:
                print(f"  [{label}] fetched {fetched}/{n_cells}", flush=True)
            if len(chunk_info["features"]) < chunk:
                break
        except Exception as e:
            print(f"  ⚠ Chunk error at {fetched}: {e}")
            break
    return rows


def _rename_cols(df):
    renames = {
        "mean": "value",
        "LST_Day_1km": "lst_day_k",
        "ET": "et_mm",
        "sm_surface": "sm_surface",
        "sm_rootzone": "sm_rootzone",
        "built": "built_frac",
        "water": "water_frac",
        "b1": "flow_acc",
        "occurrence": "water_occurrence",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
    return df


def _clean_and_save(df, path):
    df = _rename_cols(df)
    drop_cols = [c for c in df.columns if c.startswith(".geo") or c == "system:index"]
    df = df.drop(columns=drop_cols, errors="ignore")
    df.to_csv(path, index=False)


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=list(STATES.keys()) + ["all"],
                        default="all",
                        help="Which state to extract (default: all)")
    args = parser.parse_args()

    states_to_run = list(STATES.keys()) if args.state == "all" else [args.state]

    print(f"\n{'='*60}")
    print("  Multi-State GEE Extraction")
    print(f"  States: {', '.join(s.upper() for s in states_to_run)}")
    print(f"  Period: {YEARS[0]}–{YEARS[-1]}, May–Oct")
    print(f"  Scale: 5 km grid")
    print(f"{'='*60}\n")

    for state in states_to_run:
        extract_state_features(state)

    print("\n✓ All extractions complete.")
