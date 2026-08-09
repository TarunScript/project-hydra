"""
climatology.py — long-term "normal" baselines so every drought feature can be
expressed as an anomaly (Section 5.2: "Rainfall anomaly ... vs. India Drought
Atlas climatology", "NDVI anomaly ... vs. climatology").

Two ways to get climatology, pick based on how much hour-0 time you have:

1. IN-PIPELINE (default, no extra download): compute the mean of the same
   calendar-day window across CLIMATOLOGY_START_YEAR-CLIMATOLOGY_END_YEAR
   directly from CHIRPS / MODIS in GEE. Slower per-call but zero setup.

2. INDIA DROUGHT ATLAS (static, 1901-2021, more "official"): download the
   gridded monthly precip/temp CSVs from the Atlas repo once, and use
   `load_drought_atlas_climatology()` to build a lookup table. Better if you
   want a citation-backed climatology in the pitch, but needs the repo
   cloned locally first (see DROUGHT_ATLAS_REPO in config.py).
"""
import ee
import pandas as pd

from config import CLIMATOLOGY_START_YEAR, CLIMATOLOGY_END_YEAR, DATASETS


def chirps_doy_climatology(day_of_year: int, window_days: int = 15) -> ee.Image:
    """
    Mean CHIRPS daily rainfall for calendar days within +/- window_days of
    `day_of_year`, averaged over CLIMATOLOGY_START_YEAR..CLIMATOLOGY_END_YEAR.
    Returns a single-band image: 'rain_climatology_mm'.
    """
    chirps = ee.ImageCollection(DATASETS["chirps_daily"])

    def year_window(year):
        base = ee.Date.fromYMD(year, 1, 1).advance(day_of_year - 1, "day")
        return chirps.filterDate(base.advance(-window_days, "day"), base.advance(window_days, "day"))

    years = ee.List.sequence(CLIMATOLOGY_START_YEAR, CLIMATOLOGY_END_YEAR)
    yearly_means = ee.ImageCollection(
        years.map(lambda y: year_window(ee.Number(y)).mean().set("yr", y))
    )
    return yearly_means.mean().rename("rain_climatology_mm")


def modis_ndvi_doy_climatology(day_of_year: int) -> ee.Image:
    """
    Mean NDVI for the matching 16-day MOD13Q1 composite period, averaged
    across the climatology years (MODIS data starts 2000). Returns 'ndvi_climatology'.
    """
    ndvi = ee.ImageCollection(DATASETS["modis_ndvi"]).select("NDVI")

    def year_window(year):
        base = ee.Date.fromYMD(year, 1, 1).advance(day_of_year - 1, "day")
        return ndvi.filterDate(base.advance(-8, "day"), base.advance(8, "day"))

    start_year = max(CLIMATOLOGY_START_YEAR, 2000)
    years = ee.List.sequence(start_year, CLIMATOLOGY_END_YEAR)
    yearly_means = ee.ImageCollection(
        years.map(lambda y: year_window(ee.Number(y)).mean().set("yr", y))
    )
    # MOD13Q1 NDVI is scaled by 10000
    return yearly_means.mean().multiply(0.0001).rename("ndvi_climatology")


def modis_lst_doy_climatology(day_of_year: int, window_days: int = 8) -> ee.Image:
    """Mean daytime LST (Celsius) around the given day-of-year (MODIS data starts 2000). Returns 'lst_climatology_c'."""
    lst = ee.ImageCollection(DATASETS["modis_lst"]).select("LST_Day_1km")

    def year_window(year):
        base = ee.Date.fromYMD(year, 1, 1).advance(day_of_year - 1, "day")
        return lst.filterDate(base.advance(-window_days, "day"), base.advance(window_days, "day"))

    start_year = max(CLIMATOLOGY_START_YEAR, 2000)
    years = ee.List.sequence(start_year, CLIMATOLOGY_END_YEAR)
    yearly_means = ee.ImageCollection(
        years.map(lambda y: year_window(ee.Number(y)).mean().set("yr", y))
    )
    # MODIS LST is scaled by 0.02, stored in Kelvin
    return yearly_means.mean().multiply(0.02).subtract(273.15).rename("lst_climatology_c")


def load_drought_atlas_climatology(repo_local_path: str) -> pd.DataFrame:
    """
    Load the India Drought Atlas gridded monthly precip/temp CSVs (after
    cloning DROUGHT_ATLAS_REPO locally) into a lookup DataFrame keyed by
    (lat, lon, month) -> long-term mean precip/temp.

    This is a loader stub: the Atlas repo's exact file layout can change, so
    inspect `repo_local_path` after cloning and adjust the glob/column names
    below before relying on this in the pipeline.
    """
    import glob
    import os

    csv_files = glob.glob(os.path.join(repo_local_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSVs found under {repo_local_path}. Clone {DATASETS} first: "
            "git clone https://github.com/wcl-iitgn/india-drought-atlas-data"
        )
    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    return df
