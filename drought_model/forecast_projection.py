"""
forecast_projection.py — the drought "forecast" per Section 6:

    "Drought 'future' is framed honestly as trend + climatology projection
    ... skillful 60-90 day drought forecasts don't really exist even in
    production systems. Say this plainly in the pitch."

Approach: for each grid cell, take the recent trajectory of the model's
risk_score (e.g. last 4-8 weekly observations) and linearly extrapolate it
forward, then pull the projection back toward the seasonal climatological
norm the further out you go (a naive but honest way to encode "no real
forecast skill beyond a couple weeks" without just flatlining the value).

    projected(t) = current + trend_slope * t
    blended(t)   = (1 - w(t)) * projected(t) + w(t) * climatology_norm
    w(t)         = min(1, t / max_trend_horizon_days)

This is NOT presented as a hard forecast in the UI — label it "trend +
climatology projection" per the plan's honesty framing, distinct from the
flood model's actual weather-forecast-driven 1/3/7/15-day projection.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd

from config import FORECAST_HORIZONS_DAYS, MODEL_PATH, TRAINING_TABLE_CSV

MAX_TREND_HORIZON_DAYS = 21  # beyond this, projection reverts almost fully to climatology


def load_model(region: str):
    bundle = joblib.load(MODEL_PATH.format(region=region))
    return bundle["model"], bundle["feature_cols"], bundle["medians"]


def compute_recent_risk_series(region: str, cell_id: str, lookback_weeks: int = 8) -> pd.DataFrame:
    df = pd.read_csv(TRAINING_TABLE_CSV.format(region=region))
    df["date"] = pd.to_datetime(df["date"])
    cell_df = df[df["cell_id"] == cell_id].sort_values("date").tail(lookback_weeks)
    return cell_df[["date", "risk_score"]]


def fit_trend(series: pd.DataFrame) -> tuple:
    """Simple linear fit of risk_score vs. days elapsed. Returns (intercept, slope_per_day)."""
    if len(series) < 2:
        # not enough history — flat projection, fully deferring to climatology weighting
        return (series["risk_score"].iloc[-1] if len(series) else 0.0, 0.0)
    t0 = series["date"].min()
    x = (series["date"] - t0).dt.days.values.astype(float)
    y = series["risk_score"].values
    slope, intercept_at_x0 = np.polyfit(x, y, 1)
    # re-express intercept relative to the *last* observed point for extrapolation
    last_x = x[-1]
    intercept = intercept_at_x0 + slope * last_x
    return intercept, slope


def climatology_norm_for_cell(region: str, cell_id: str, target_month: int) -> float:
    """
    Seasonal norm to pull the projection toward at long horizons. Approximated
    here as the cell's historical mean risk_score for the same calendar
    month, computed from the training table. Swap in India Drought Atlas
    climatology (see climatology.load_drought_atlas_climatology) for a more
    rigorous baseline if time allows.
    """
    df = pd.read_csv(TRAINING_TABLE_CSV.format(region=region))
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    cell_df = df[(df["cell_id"] == cell_id) & (df["month"] == target_month)]
    if cell_df.empty:
        return float(df[df["cell_id"] == cell_id]["risk_score"].mean())
    return float(cell_df["risk_score"].mean())


def project_cell_risk(region: str, cell_id: str, horizon_days: int, as_of_date: pd.Timestamp = None) -> float:
    as_of_date = as_of_date or pd.Timestamp.today()
    series = compute_recent_risk_series(region, cell_id)
    intercept, slope = fit_trend(series)

    raw_projection = intercept + slope * horizon_days
    raw_projection = float(np.clip(raw_projection, 0, 1))

    target_date = as_of_date + pd.Timedelta(days=horizon_days)
    norm = climatology_norm_for_cell(region, cell_id, target_date.month)

    w = min(1.0, horizon_days / MAX_TREND_HORIZON_DAYS)
    blended = (1 - w) * raw_projection + w * norm
    return float(np.clip(blended, 0, 1))


def project_all_cells(region: str, horizons: list = None) -> pd.DataFrame:
    horizons = horizons or FORECAST_HORIZONS_DAYS
    df = pd.read_csv(TRAINING_TABLE_CSV.format(region=region))
    cell_ids = df["cell_id"].unique()

    rows = []
    for cell_id in cell_ids:
        for h in horizons:
            risk = project_cell_risk(region, cell_id, h)
            rows.append({"cell_id": cell_id, "horizon_days": h, "risk_score": risk})
    result = pd.DataFrame(rows)
    print(f"[forecast_projection] projected {len(cell_ids)} cells x {len(horizons)} horizons")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="marathwada")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = project_all_cells(args.region)
    out_path = args.out or f"data/drought_projection_{args.region}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"[forecast_projection] wrote -> {out_path}")
