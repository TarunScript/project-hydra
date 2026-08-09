"""
train_model.py — trains the drought risk model per Section 5.2:
    "XGBoost or Random Forest ... trained on the grid-cell feature tables
    ... with the historical labels as targets."

Target: CDI severity (0-5 ordinal) normalized to a 0-1 risk score, so the
output is directly comparable to the flood model's 0-1 risk score for the
shared map/legend (Section 7's Low/Moderate/High/Severe palette).

Usage:
    python train_model.py --region marathwada
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from config import FEATURES_CSV, LABELS_CSV, TRAINING_TABLE_CSV, MODEL_PATH, MAX_SEVERITY
from labels import snap_labels_to_grid_cells

NON_FEATURE_COLS = {"cell_id", "date", "region", "district", "cdi_class", "severity", "risk_score"}


def build_training_table(region: str) -> pd.DataFrame:
    features = pd.read_csv(FEATURES_CSV.format(region=region))
    raw_labels = pd.read_csv(LABELS_CSV.format(region=region))

    if "cell_id" not in features.columns:
        raise ValueError(
            "features CSV is missing 'cell_id' — make sure build_grid() tagged "
            "cells before running feature_pipeline.py."
        )

    # raw_labels from labels.py v2 are individual CDI grid points (lat, lon,
    # cdi_value, cdi_class, severity, date) — snap each of our feature grid
    # cells to its nearest CDI point before merging on cell_id.
    labels = snap_labels_to_grid_cells(raw_labels, features)

    merged = features.merge(labels, on=["cell_id", "date"], how="inner")
    if merged.empty:
        raise ValueError(
            "No rows after merging features and labels — check that dates in "
            "both CSVs actually overlap (features are typically weekly/whatever "
            "cadence you ran feature_pipeline.py at; CDI labels are weekly)."
        )
    merged["risk_score"] = merged["severity"] / MAX_SEVERITY

    out_path = TRAINING_TABLE_CSV.format(region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[train_model] training table: {len(merged)} rows -> {out_path}")
    return merged


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """
    Split by date rather than randomly, to avoid leaking future weeks into
    training and inflating validation scores — the honest way to evaluate a
    model that will actually be used to project forward in time.
    """
    dates = sorted(df["date"].unique())
    cutoff_idx = int(len(dates) * (1 - test_frac))
    cutoff_date = dates[cutoff_idx]
    train_df = df[df["date"] < cutoff_date]
    test_df = df[df["date"] >= cutoff_date]
    print(f"[train_model] time split at {cutoff_date}: train={len(train_df)} rows, test={len(test_df)} rows")
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def train(region: str, model_type: str = "xgboost"):
    df = build_training_table(region)
    feature_cols = get_feature_columns(df)
    print(f"[train_model] using {len(feature_cols)} features: {feature_cols}")

    train_df, test_df = time_based_split(df)
    X_train, y_train = train_df[feature_cols], train_df["risk_score"]
    X_test, y_test = test_df[feature_cols], test_df["risk_score"]

    # simple median imputation for any NaNs from missing satellite passes
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    if model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        )
    else:
        model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    preds = np.clip(model.predict(X_test), 0, 1)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print(f"[train_model] holdout MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("[train_model] top features:")
        print(importances.head(10).to_string())

    out_path = MODEL_PATH.format(region=region)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols, "medians": medians.to_dict()}, out_path)
    print(f"[train_model] saved model -> {out_path}")

    return model, {"mae": mae, "rmse": rmse, "r2": r2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="marathwada")
    parser.add_argument("--model-type", choices=["xgboost", "random_forest"], default="xgboost")
    args = parser.parse_args()
    train(args.region, args.model_type)
