"""
project_drought_risk.py
========================
Generates current + 7d/15d drought risk projections using the trained
Blended Ensemble (70% ExtraTrees + 30% XGBoost).

IMPORTANT LIMITATION:
  The +7d/+15d projections are simple trend extrapolations based on current
  deficit trajectory vs. climatology.  They are NOT learned forecast models.
"""

import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

from train_drought_model import engineer_advanced_features, get_feature_cols


def blend_predict(model_xgb, model_et, X):
    """70% ExtraTrees + 30% XGBoost, clipped to [0, 1]."""
    p = 0.70 * model_et.predict(X) + 0.30 * model_xgb.predict(X)
    return np.clip(p, 0.0, 1.0)


def main():
    base_dir = Path(r"c:\Users\riyav\project-hydra")
    xgb_path = base_dir / "models" / "drought_xgb_model.json"
    et_path  = base_dir / "models" / "drought_et_model.joblib"
    data_dir = base_dir / "data"

    if not xgb_path.exists():
        print(f"XGBoost model not found at {xgb_path}. Run train_drought_model.py first.")
        return
    if not et_path.exists():
        print(f"ExtraTrees model not found at {et_path}. Run train_drought_model.py first.")
        return

    print(f"Loading XGBoost model from {xgb_path}...")
    model_xgb = xgb.XGBRegressor()
    model_xgb.load_model(xgb_path)

    print(f"Loading ExtraTrees model from {et_path}...")
    model_et = joblib.load(et_path)

    # Priority: labeled features -> all regions features
    data_file = data_dir / "drought_features_labeled.csv"
    if not data_file.exists():
        data_file = data_dir / "drought_features_all_regions.csv"

    if not data_file.exists():
        print(f"No feature file found in {data_dir}. Exiting.")
        return

    print(f"Processing data from {data_file}...")
    df = pd.read_csv(data_file)

    # Clean infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Ensure cell_id is unique per cell per date
    if 'cell_id' not in df.columns:
        if 'region' in df.columns and 'date' in df.columns:
            df['cell_id'] = (df['region'] + "_" + df['date'].astype(str) + "_"
                             + df['lat'].round(4).astype(str) + "_"
                             + df['lon'].round(4).astype(str))
        else:
            df['cell_id'] = ("cell_" + df['lat'].round(4).astype(str) + "_"
                             + df['lon'].round(4).astype(str))

    print("Applying advanced feature engineering...")
    df_eng = engineer_advanced_features(df)
    feature_cols = get_feature_cols(df_eng)
    print(f"Using {len(feature_cols)} features for prediction")

    X = df_eng[feature_cols].fillna(0)

    # Predict current risk (blended ensemble)
    df['current_risk'] = blend_predict(model_xgb, model_et, X)

    # --- Project 7 days forward ---
    df_7d = df.copy()
    deficit_rate = df['rain_30d_deficit_mm'].fillna(0) / 30.0

    if 'rain_7d_deficit_mm' in df_7d.columns:
        df_7d['rain_7d_deficit_mm'] = df_7d['rain_7d_deficit_mm'] + (deficit_rate * 7)
    if 'sm_rootzone' in df_7d.columns:
        df_7d['sm_rootzone'] = df_7d['sm_rootzone'] * (1 + (deficit_rate / 100.0) * 7)
    if 'ndvi_anomaly' in df_7d.columns:
        df_7d['ndvi_anomaly'] = df_7d['ndvi_anomaly'] + (deficit_rate / 100.0) * 7

    df_7d_eng = engineer_advanced_features(df_7d)
    X_7d = df_7d_eng[feature_cols].fillna(0)
    df['risk_7d'] = blend_predict(model_xgb, model_et, X_7d)

    # --- Project 15 days forward ---
    df_15d = df.copy()
    if 'rain_7d_deficit_mm' in df_15d.columns:
        df_15d['rain_7d_deficit_mm'] = df_15d['rain_7d_deficit_mm'] + (deficit_rate * 15)
    if 'sm_rootzone' in df_15d.columns:
        df_15d['sm_rootzone'] = df_15d['sm_rootzone'] * (1 + (deficit_rate / 100.0) * 15)
    if 'ndvi_anomaly' in df_15d.columns:
        df_15d['ndvi_anomaly'] = df_15d['ndvi_anomaly'] + (deficit_rate / 100.0) * 15

    df_15d_eng = engineer_advanced_features(df_15d)
    X_15d = df_15d_eng[feature_cols].fillna(0)
    df['risk_15d'] = blend_predict(model_xgb, model_et, X_15d)

    df['projection_label'] = ('Trend projection (not a forecast) — '
                              'based on current deficit trajectory vs. climatology')

    out_path = data_dir / "drought_projections.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved projections to {out_path} ({len(df)} rows across "
          f"regions: {df['region'].unique().tolist()})")
    print("NOTE: These projections are NOT a learned forecast model. "
          "They are a simple trend extrapolation.")


if __name__ == "__main__":
    main()
