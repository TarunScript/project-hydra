"""
labels.py (v2) — pulls REAL historical drought labels directly from the
India Drought Monitor's public data files. No manual download, no district
shapefile join needed.

Confirmed endpoints (verified by fetching them directly):
    https://indiadroughtmonitor.in/data/Current_CDI.txt
        -> latest week, plain text, one row per grid point:
           "<lat> <lon> <CDI value>" on a 0.25-degree grid.
           NaN rows (ocean/no-data cells) appear as "NaN NaN NaN".
    https://indiadroughtmonitor.in/data/Drough_TS/CDI_YYYYMMDD.txt
        -> same format, for a specific past week. Data Download page states
           this goes back to July 2021. Confirm the exact weekly cadence
           yourself (curl -I the URL for your target date) before relying
           on it for a specific week — archive weeks likely land on a
           fixed weekday, not every date will 200.

Important: the site gives a CONTINUOUS CDI value (drier = more negative),
not a pre-baked D0-D4 class per point. The classification thresholds below
follow the standard SPI-style drought classification convention (the same
scale CDI is built on) — this is a reasonable, commonly-used mapping, but
it is OUR bucketing choice, not the site's own official class boundaries.
Say that plainly if you cite "IDM's D0-D4 classes" in the pitch — what you
actually have is CDI value -> your own severity bucketing.
"""
import argparse
import io
import os

import pandas as pd
import requests

from config import REGIONS, LABELS_CSV

DROUGHT_MONITOR_BASE = "https://indiadroughtmonitor.in"
CURRENT_CDI_URL = f"{DROUGHT_MONITOR_BASE}/data/Current_CDI.txt"
WEEKLY_CDI_URL_TEMPLATE = f"{DROUGHT_MONITOR_BASE}/data/Drough_TS/CDI_{{date}}.txt"
NATIONAL_TIMESERIES_URL = f"{DROUGHT_MONITOR_BASE}/data/India_Drought_Area_Timeseries.txt"


def nearest_archive_week(target_date: str) -> str:
    """
    Weekly CDI grids don't exist for every calendar date — confirmed from
    India_Drought_Area_Timeseries.txt that archive weeks land on
    Wednesdays. Given a 'YYYYMMDD' or 'YYYY-MM-DD' target, snap to the
    nearest Wednesday and return 'YYYYMMDD' for use in
    WEEKLY_CDI_URL_TEMPLATE. This avoids the trial-and-error of guessing
    which exact date 200s.
    """
    d = pd.to_datetime(target_date)
    # Wednesday = weekday() == 2
    offset = (2 - d.weekday()) % 7
    if offset > 3:
        offset -= 7  # snap to nearest Wednesday, not just the next one
    snapped = d + pd.Timedelta(days=offset)
    return snapped.strftime("%Y%m%d")

# Standard SPI/CDI-style severity bucketing (our own choice — see docstring).
# value thresholds are upper bounds (exclusive) for each class, driest first.
CDI_VALUE_THRESHOLDS = [
    (-2.0, "D4"),
    (-1.5, "D3"),
    (-1.2, "D2"),
    (-0.7, "D1"),
    (-0.5, "D0"),
    (float("inf"), "None"),
]
CDI_CLASS_TO_SEVERITY = {"None": 0, "D0": 1, "D1": 2, "D2": 3, "D3": 4, "D4": 5}


def classify_cdi_value(value: float) -> str:
    if pd.isna(value):
        return None
    for threshold, cls in CDI_VALUE_THRESHOLDS:
        if value < threshold:
            return cls
    return "None"


def fetch_cdi_grid(date_str: str = None, timeout: int = 30, auto_snap: bool = True) -> pd.DataFrame:
    """
    date_str: 'YYYYMMDD' (or 'YYYY-MM-DD') for a historical week, or None
    for the current (latest) week. Returns columns: [lat, lon, cdi_value].

    auto_snap: if True (default), snap date_str to the nearest Wednesday
    before fetching — archive weeks only exist on Wednesdays (confirmed via
    India_Drought_Area_Timeseries.txt), so an arbitrary date will 404.
    """
    if date_str is not None and auto_snap:
        snapped = nearest_archive_week(date_str)
        if snapped != date_str.replace("-", ""):
            print(f"[labels] snapped {date_str} -> nearest archive week {snapped}")
        date_str = snapped

    url = CURRENT_CDI_URL if date_str is None else WEEKLY_CDI_URL_TEMPLATE.format(date=date_str)
    resp = requests.get(url, timeout=timeout)
    if resp.status_code == 404:
        raise FileNotFoundError(
            f"No CDI grid at {url} — even after snapping to the nearest "
            "Wednesday, this week isn't in the archive. Check "
            f"{NATIONAL_TIMESERIES_URL} for the exact available dates."
        )
    resp.raise_for_status()

    df = pd.read_csv(
        io.StringIO(resp.text),
        sep=r"\s+",
        names=["lat", "lon", "cdi_value"],
        na_values=["NaN"],
    )
    df = df.dropna(subset=["lat", "lon"])  # keep rows even if cdi_value is NaN (data gaps)
    return df


def clip_to_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = REGIONS[region]["bbox"]
    mask = (
        (df["lon"] >= min_lon) & (df["lon"] <= max_lon)
        & (df["lat"] >= min_lat) & (df["lat"] <= max_lat)
    )
    return df.loc[mask].copy()


def fetch_region_labels(region: str, date_str: str = None) -> pd.DataFrame:
    """
    Full pipeline: fetch the national CDI grid for one week, clip to the
    region bbox, classify each point. Returns columns:
        [lat, lon, cdi_value, cdi_class, severity, date]

    The 'date' column reflects the ACTUAL archive week fetched (after
    snapping to the nearest Wednesday), not necessarily date_str as passed
    in — this matters for merging with feature_pipeline.py's dates later,
    so pull your feature dates from this returned 'date' column rather than
    assuming they match what you requested.
    """
    actual_date_str = nearest_archive_week(date_str) if date_str is not None else None
    grid = fetch_cdi_grid(date_str)
    region_grid = clip_to_region(grid, region)
    region_grid["cdi_class"] = region_grid["cdi_value"].apply(classify_cdi_value)
    region_grid["severity"] = region_grid["cdi_class"].map(CDI_CLASS_TO_SEVERITY)
    # normalize to YYYY-MM-DD to match feature_pipeline.py's date format for merging
    if actual_date_str is None:
        region_grid["date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    else:
        region_grid["date"] = pd.to_datetime(actual_date_str, format="%Y%m%d").strftime("%Y-%m-%d")
    return region_grid.dropna(subset=["cdi_class"])


def fetch_region_labels_for_dates(region: str, dates: list) -> pd.DataFrame:
    frames = []
    for d in dates:
        try:
            frames.append(fetch_region_labels(region, d))
            print(f"[labels] fetched {region} @ {d}: {len(frames[-1])} grid points")
        except FileNotFoundError as e:
            print(f"[labels] SKIPPED {d}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def snap_labels_to_grid_cells(labels_df: pd.DataFrame, cell_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Nearest-point join: for each of your feature_pipeline.py grid cells
    (which have cell_id + implied centroid from build_grid()), find the
    closest CDI grid point for that date and inherit its label.

    This assumes cell_features_df has 'cell_id', 'date', and centroid
    lat/lon columns — feature_pipeline.py's grid cells encode centroid
    coords in cell_id itself (see build_grid()'s "lon_lat" join), so parse
    those back out, or export lat/lon columns explicitly from GEE if you'd
    rather not parse the id string.
    """
    cells = cell_features_df.copy()
    if "lat" not in cells.columns or "lon" not in cells.columns:
        coords = cells["cell_id"].str.split("_", expand=True).astype(float)
        cells["lon"], cells["lat"] = coords[0], coords[1]

    results = []
    for date, date_cells in cells.groupby("date"):
        date_labels = labels_df[labels_df["date"] == date]
        if date_labels.empty:
            continue
        for _, cell in date_cells.iterrows():
            dists = ((date_labels["lat"] - cell["lat"]) ** 2 + (date_labels["lon"] - cell["lon"]) ** 2)
            nearest = date_labels.loc[dists.idxmin()]
            results.append({
                "cell_id": cell["cell_id"],
                "date": date,
                "cdi_class": nearest["cdi_class"],
                "severity": nearest["severity"],
            })
    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="marathwada", choices=list(REGIONS.keys()))
    parser.add_argument("--dates", nargs="+", default=[None], help="YYYYMMDD dates, or omit for current week only")
    args = parser.parse_args()

    dates = args.dates if args.dates != [None] else [None]
    labels_df = fetch_region_labels_for_dates(args.region, dates)

    out_path = LABELS_CSV.format(region=args.region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    labels_df.to_csv(out_path, index=False)
    print(f"[labels] wrote {len(labels_df)} rows -> {out_path}")
