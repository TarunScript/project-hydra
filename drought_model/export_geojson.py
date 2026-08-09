"""
export_geojson.py — precomputes static per-date risk GeoJSON files for the
frontend, per Section 8:

    "Timeline slider: swaps which precomputed day's risk layer is
    displayed... don't do live inference on every slider drag."

Produces one GeoJSON per historical date (from the training table, using the
model's own predictions for consistency with the future projections) and one
per forecast horizon (from forecast_projection.py), all in the same schema
so the frontend doesn't need to special-case past vs. future.

Schema per feature:
    { "cell_id", "risk_score" (0-1), "risk_level" (Low/Moderate/High/Severe),
      geometry: cell polygon }
"""
import argparse
import json
import os

import geopandas as gpd
import joblib
import pandas as pd

from config import MODEL_PATH, RISK_GEOJSON_DIR, TRAINING_TABLE_CSV
from alert_logic import risk_score_to_level


def load_grid_geometries(grid_geojson_path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(grid_geojson_path)[["cell_id", "geometry"]]


def export_historical_days(region: str, grid_geojson_path: str, out_dir: str = None):
    out_dir = out_dir or RISK_GEOJSON_DIR.format(region=region)
    os.makedirs(out_dir, exist_ok=True)

    grid = load_grid_geometries(grid_geojson_path)
    df = pd.read_csv(TRAINING_TABLE_CSV.format(region=region))

    for date, day_df in df.groupby("date"):
        merged = grid.merge(day_df[["cell_id", "risk_score"]], on="cell_id", how="inner")
        merged["risk_level"] = merged["risk_score"].apply(risk_score_to_level)
        out_path = os.path.join(out_dir, f"{date}.geojson")
        merged.to_file(out_path, driver="GeoJSON")

    print(f"[export_geojson] wrote {df['date'].nunique()} historical day files -> {out_dir}")


def export_forecast_horizons(region: str, grid_geojson_path: str, projection_csv: str, out_dir: str = None):
    out_dir = out_dir or RISK_GEOJSON_DIR.format(region=region)
    os.makedirs(out_dir, exist_ok=True)

    grid = load_grid_geometries(grid_geojson_path)
    proj = pd.read_csv(projection_csv)

    for horizon, horizon_df in proj.groupby("horizon_days"):
        merged = grid.merge(horizon_df[["cell_id", "risk_score"]], on="cell_id", how="inner")
        merged["risk_level"] = merged["risk_score"].apply(risk_score_to_level)
        out_path = os.path.join(out_dir, f"forecast_plus{int(horizon)}d.geojson")
        merged.to_file(out_path, driver="GeoJSON")

    print(f"[export_geojson] wrote {proj['horizon_days'].nunique()} forecast horizon files -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="marathwada")
    parser.add_argument("--grid-geojson", required=True)
    parser.add_argument("--projection-csv", default=None)
    args = parser.parse_args()

    export_historical_days(args.region, args.grid_geojson)
    if args.projection_csv:
        export_forecast_horizons(args.region, args.grid_geojson, args.projection_csv)
