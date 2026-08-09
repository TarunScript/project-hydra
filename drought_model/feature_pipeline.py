"""
feature_pipeline.py — builds the per-grid-cell drought feature table.

Follows the same pattern as the flood model's starter script (Section 4):
  define area -> build grid -> pull + stack features -> reduce to per-cell
  values -> export.

Covers every row of the Section 5.2 feature table except the label column
(that's labels.py) and land-cover/GRACE, which are here too.

Run as a script:
    python feature_pipeline.py --region marathwada --date 2024-08-15 \
        --project YOUR-CLOUD-PROJECT-ID
"""
import argparse
import datetime as dt

import ee
import pandas as pd

from config import (
    REGIONS, GRID_SCALE_M, FEATURE_REDUCE_SCALE_M, UTM_EPSG,
    DEFICIT_WINDOWS, DRY_DAY_THRESHOLD_MM, DATASETS, FEATURES_CSV,
)
from climatology import (
    chirps_doy_climatology, modis_ndvi_doy_climatology, modis_lst_doy_climatology,
)
from gee_setup import init_ee


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def build_grid(region: str) -> ee.FeatureCollection:
    bbox = REGIONS[region]["bbox"]
    aoi = ee.Geometry.Rectangle(bbox)
    proj = ee.Projection(UTM_EPSG[region]).atScale(GRID_SCALE_M)
    grid = aoi.coveringGrid(proj)
    # tag each cell with a stable id so features/labels can be joined later
    grid = grid.map(lambda f: f.set("cell_id", f.geometry().centroid(1).coordinates().join("_")))
    return grid


# ---------------------------------------------------------------------------
# Individual feature blocks
# ---------------------------------------------------------------------------
def rainfall_deficit_features(target_date: ee.Date, climatology_doy: int) -> ee.Image:
    """
    7/30/60/90-day rainfall accumulation + anomaly vs climatology, per
    Section 5.2 row 1. Anomaly = (observed - climatology) / climatology,
    i.e. a signed fractional deficit (negative = drier than normal).
    """
    chirps = ee.ImageCollection(DATASETS["chirps_daily"])
    bands = []
    for window in DEFICIT_WINDOWS:
        start = target_date.advance(-window, "day")
        accum = chirps.filterDate(start, target_date).sum().rename(f"rain_{window}d_mm")
        bands.append(accum)

        clim = chirps_doy_climatology(climatology_doy, window_days=min(window // 2, 15))
        # scale the single-day climatology mean up to the window length for a fair comparison
        clim_window = clim.multiply(window).rename(f"rain_{window}d_climatology_mm")
        anomaly = (
            accum.subtract(clim_window)
            .divide(clim_window.max(1))  # avoid div-by-zero in near-zero-rainfall windows
            .rename(f"rain_{window}d_anomaly_frac")
        )
        bands.append(anomaly)

    return ee.Image.cat(bands)


def dry_spell_duration(target_date: ee.Date, lookback_days: int = 90) -> ee.Image:
    """
    Number of dry days (< DRY_DAY_THRESHOLD_MM rain) in the lookback_days window
    preceding target_date (Section 5.2). Vectorized in GEE to avoid server element limits.
    """
    chirps = ee.ImageCollection(DATASETS["chirps_daily"])
    start = target_date.advance(-lookback_days, "day")
    daily = chirps.filterDate(start, target_date)
    dry_days = daily.map(lambda img: img.lt(DRY_DAY_THRESHOLD_MM)).sum().rename("dry_spell_days")
    return dry_days


def soil_moisture_features() -> ee.Image:
    """Most recent SMAP L4 surface + root-zone soil moisture (Section 5.2 row 2)."""
    smap = (
        ee.ImageCollection(DATASETS["smap_l4"])
        .sort("system:time_start", False)
        .first()
        .select(["sm_surface", "sm_rootzone"])
    )
    return smap


def ndvi_anomaly_features(target_date: ee.Date, climatology_doy: int) -> ee.Image:
    """NDVI anomaly vs. climatology (Section 5.2 row 3)."""
    ndvi_coll = ee.ImageCollection(DATASETS["modis_ndvi"]).select("NDVI")
    current = (
        ndvi_coll.filterDate(target_date.advance(-16, "day"), target_date)
        .sort("system:time_start", False)
        .first()
        .multiply(0.0001)
        .rename("ndvi_current")
    )
    clim = modis_ndvi_doy_climatology(climatology_doy)
    anomaly = current.subtract(clim).rename("ndvi_anomaly")
    return current.addBands([clim, anomaly])


def lst_anomaly_features(target_date: ee.Date, climatology_doy: int) -> ee.Image:
    """Land surface temperature anomaly (Section 5.2 row 4)."""
    lst_coll = ee.ImageCollection(DATASETS["modis_lst"]).select("LST_Day_1km")
    current = (
        lst_coll.filterDate(target_date.advance(-3, "day"), target_date.advance(1, "day"))
        .sort("system:time_start", False)
        .first()
        .multiply(0.02)
        .subtract(273.15)
        .rename("lst_current_c")
    )
    clim = modis_lst_doy_climatology(climatology_doy)
    anomaly = current.subtract(clim).rename("lst_anomaly_c")
    return current.addBands([clim, anomaly])


def evapotranspiration_feature(target_date: ee.Date) -> ee.Image:
    """8-day gap-filled ET (Section 5.2 row 7). Use MOD16A2GF, not MOD16A2."""
    et = (
        ee.ImageCollection(DATASETS["modis_et"])
        .select("ET")
        .filterDate(target_date.advance(-8, "day"), target_date)
        .sort("system:time_start", False)
        .first()
        .multiply(0.1)  # MOD16A2GF ET scale factor
        .rename("et_mm_8day")
    )
    return et


def land_cover_features(target_date: ee.Date) -> ee.Image:
    """Dynamic World land-cover probabilities in the 30 days before target_date (Section 5.2 row 8)."""
    dw = (
        ee.ImageCollection(DATASETS["dynamic_world"])
        .filterDate(target_date.advance(-30, "day"), target_date.advance(1, "day"))
        .select(["crops", "built", "bare", "water"])
        .mean()
        .rename(["crops_frac", "built_frac", "bare_frac", "water_frac"])
    )
    return dw


def groundwater_trend_feature(target_date: ee.Date) -> ee.Image:
    """
    GRACE trend indicator (Section 5.2 row 9). Coarse (~300km) — broadcast
    the same value to every cell in the region; label it clearly as
    regional, not per-cell, downstream.
    """
    grace = ee.ImageCollection(DATASETS["grace"]).select("lwe_thickness_csr")
    # GRACE dataset ended in 2017. If target_date has no data, fallback to the latest available images.
    recent_coll = grace.filterDate(target_date.advance(-90, "day"), target_date)
    recent = ee.Image(ee.Algorithms.If(
        recent_coll.size().gt(0),
        recent_coll.mean(),
        grace.sort("system:time_start", False).limit(3).mean()
    ))
    older_coll = grace.filterDate(target_date.advance(-365, "day"), target_date.advance(-275, "day"))
    older = ee.Image(ee.Algorithms.If(
        older_coll.size().gt(0),
        older_coll.mean(),
        grace.sort("system:time_start", True).limit(12).mean()
    ))
    trend = recent.subtract(older).rename("grace_trend_cm")
    return trend


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_feature_image(region: str, target_date_str: str) -> ee.Image:
    target_date = ee.Date(target_date_str)
    py_date = dt.datetime.strptime(target_date_str, "%Y-%m-%d")
    doy = py_date.timetuple().tm_yday

    bands = [
        rainfall_deficit_features(target_date, doy),
        dry_spell_duration(target_date),
        soil_moisture_features(),
        ndvi_anomaly_features(target_date, doy),
        lst_anomaly_features(target_date, doy),
        evapotranspiration_feature(target_date),
        land_cover_features(target_date),
        groundwater_trend_feature(target_date),
    ]
    image = ee.Image.cat(bands)
    return image.set("date", target_date_str, "region", region)


def extract_cell_features(region: str, target_date_str: str) -> ee.FeatureCollection:
    grid = build_grid(region)
    image = build_feature_image(region, target_date_str)
    cell_features = image.reduceRegions(
        collection=grid, reducer=ee.Reducer.mean(), scale=FEATURE_REDUCE_SCALE_M
    )
    return cell_features.map(lambda f: f.set("date", target_date_str, "region", region))


def to_dataframe(fc: ee.FeatureCollection, chunk_size: int = 500) -> pd.DataFrame:
    """
    Pull a FeatureCollection to a local DataFrame in chunks to avoid GEE's
    5,000 element server limit ('Collection query aborted after accumulating over 5000 elements').
    """
    size = fc.size().getInfo()
    fc_list = fc.toList(size)
    rows = []
    for offset in range(0, size, chunk_size):
        sub_fc = ee.FeatureCollection(fc_list.slice(offset, offset + chunk_size))
        features = sub_fc.getInfo()["features"]
        for f in features:
            rows.append(f["properties"])
    return pd.DataFrame(rows)


def export_features_csv(region: str, dates: list[str], out_path: str = None, use_drive: bool = False):
    """
    Pull features for each date and write one combined CSV. For hackathon
    scale (one region, tens of dates) local pull via getInfo/to_dataframe is
    fine; flip use_drive=True and swap in ee.batch.Export.table.toDrive if a
    region/date range gets too big for synchronous pulls.
    """
    all_rows = []
    for d in dates:
        print(f"[feature_pipeline] extracting {region} @ {d} ...")
        fc = extract_cell_features(region, d)
        if use_drive:
            task = ee.batch.Export.table.toDrive(
                collection=fc,
                description=f"drought_features_{region}_{d}",
                fileFormat="CSV",
            )
            task.start()
            print(f"  -> export task started (check GEE Tasks tab): {task.id}")
            continue
        df = to_dataframe(fc)
        all_rows.append(df)

    if use_drive:
        return None

    full_df = pd.concat(all_rows, ignore_index=True)
    out_path = out_path or FEATURES_CSV.format(region=region)
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"[feature_pipeline] wrote {len(full_df)} rows -> {out_path}")
    return full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="marathwada", choices=list(REGIONS.keys()))
    parser.add_argument("--project", required=True, help="Your GEE Cloud Project ID")
    parser.add_argument(
        "--dates", nargs="+", default=None,
        help="One or more YYYY-MM-DD dates. Defaults to weekly dates for the last 6 months.",
    )
    parser.add_argument("--drive", action="store_true", help="Export via Drive instead of local pull")
    args = parser.parse_args()

    init_ee(args.project)

    if args.dates is None:
        today = dt.date.today()
        dates = [(today - dt.timedelta(weeks=w)).isoformat() for w in range(0, 26, 1)]
    else:
        dates = args.dates

    export_features_csv(args.region, dates, use_drive=args.drive)
