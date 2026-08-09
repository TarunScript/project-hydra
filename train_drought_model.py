"""
train_drought_model.py
=======================
High-accuracy Blended Ensemble Regressor for predicting drought risk index.
Primary ensemble: 70% ExtraTrees + 30% XGBoost (optimised blend).

Uses 39 engineered features from 7 GEE datasets and 100% real IDM ground truth.
"""

import os
import json
import datetime
import joblib
import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix


def engineer_advanced_features(df):
    """
    Engineers rich spatial, temporal, physical interaction, and regional
    micro-climate features.  Returns a NEW DataFrame (never mutates input).
    """
    df = df.copy()

    # 1. Physical interaction ratios & EWS indicators
    df['rain_deficit_ratio_30d'] = df['rain_30d_deficit_mm'] / (df['rain_30d_mm'] + 1.0)
    df['rain_deficit_ratio_60d'] = df['rain_60d_deficit_mm'] / (df['rain_60d_mm'] + 1.0)
    df['rain_deficit_ratio_90d'] = df['rain_90d_deficit_mm'] / (df['rain_90d_mm'] + 1.0)
    df['sm_ratio'] = df['sm_surface'] / (df['sm_rootzone'] + 1e-4)
    df['temp_veg_ratio'] = df['lst_current_c'] / (df['ndvi_current'] + 0.1)
    df['evap_stress_index'] = df['et_current_kg_m2'] / (df['rain_30d_mm'] + 1.0)
    df['temp_anomaly_sq'] = df['lst_anomaly_c'] ** 2
    df['rain_deficit_sq'] = df['rain_30d_deficit_mm'] ** 2

    # 2. Cross-term physical interactions
    df['deficit_x_temp'] = df['rain_30d_deficit_mm'] * df['lst_anomaly_c']
    df['deficit_x_ndvi'] = df['rain_30d_deficit_mm'] * df['ndvi_anomaly']
    df['sm_x_temp'] = df['sm_rootzone'] * df['lst_current_c']
    df['sm_x_ndvi'] = df['sm_rootzone'] * df['ndvi_current']
    df['dry_spell_x_deficit'] = df['dry_spell_days'] * df['rain_30d_deficit_mm']
    df['et_x_sm'] = df['et_current_kg_m2'] * df['sm_rootzone']

    # 3. Spatial coordinates & non-linear spatial interactions
    df['lat_sq'] = df['lat'] ** 2
    df['lon_sq'] = df['lon'] ** 2
    df['lat_lon_prod'] = df['lat'] * df['lon']

    # District-level microclimate aggregations
    if 'district' in df.columns and 'rain_30d_deficit_mm' in df.columns:
        dist_rain_mean = df.groupby('district')['rain_30d_deficit_mm'].transform('mean')
        dist_sm_mean = df.groupby('district')['sm_rootzone'].transform('mean')
        dist_ndvi_mean = df.groupby('district')['ndvi_anomaly'].transform('mean')
        df['dist_mean_rain_deficit'] = dist_rain_mean
        df['dist_mean_sm_rootzone'] = dist_sm_mean
        df['dist_mean_ndvi_anomaly'] = dist_ndvi_mean

    # 4. Regional one-hot encodings
    if 'region' in df.columns:
        region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=False)
        df = pd.concat([df, region_dummies], axis=1)

    # 5. Seasonal month encodings (sin/cos + quarter dummies)
    if 'date' in df.columns:
        months = pd.to_datetime(df['date']).dt.month
        df['month_sin'] = np.sin(2 * np.pi * months / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * months / 12.0)
        df['is_monsoon'] = months.isin([6, 7, 8, 9]).astype(float)
        df['is_rabi'] = months.isin([10, 11, 12, 1, 2]).astype(float)

    return df


# Canonical feature list — must be kept in sync with project_drought_risk.py
FEATURE_COLS_BASE = [
    'lat', 'lon', 'lat_sq', 'lon_sq', 'lat_lon_prod',
    'bare_ground_frac', 'cropland_frac', 'dry_spell_days', 'et_current_kg_m2',
    'lst_anomaly_c', 'lst_current_c', 'ndvi_anomaly', 'ndvi_current',
    'rain_30d_deficit_mm', 'rain_30d_mm', 'rain_60d_deficit_mm', 'rain_60d_mm',
    'rain_7d_deficit_mm', 'rain_7d_mm', 'rain_90d_deficit_mm', 'rain_90d_mm',
    'sm_rootzone', 'sm_surface', 'urban_built_frac',
    'rain_deficit_ratio_30d', 'rain_deficit_ratio_60d', 'rain_deficit_ratio_90d',
    'sm_ratio', 'temp_veg_ratio', 'evap_stress_index', 'temp_anomaly_sq', 'rain_deficit_sq',
    'deficit_x_temp', 'deficit_x_ndvi', 'sm_x_temp', 'sm_x_ndvi',
    'dry_spell_x_deficit', 'et_x_sm',
    'dist_mean_rain_deficit', 'dist_mean_sm_rootzone', 'dist_mean_ndvi_anomaly',
    'month_sin', 'month_cos', 'is_monsoon', 'is_rabi'
]


def get_feature_cols(df):
    """Return the ordered feature column list present in df."""
    region_cols = sorted([c for c in df.columns if c.startswith('region_')])
    return [c for c in (FEATURE_COLS_BASE + region_cols) if c in df.columns]


def main():
    data_path = 'data/drought_features_labeled.csv'
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    print(f"Loading dataset from {data_path}...")
    try:
        raw_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}.")
        return

    # Clean infinite values & missing targets
    raw_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    raw_df = raw_df.dropna(subset=['drought_risk_score'])

    print("Engineering advanced physical, spatial, micro-climate & seasonal features...")
    df = engineer_advanced_features(raw_df)

    target_col = 'drought_risk_score'
    feature_cols = get_feature_cols(df)

    print(f"Using {len(feature_cols)} features for model training")

    X = df[feature_cols].fillna(0)
    y = df[target_col]
    stratify_col = raw_df['region'] if 'region' in raw_df.columns else None

    # Train/Validation Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_col
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_test.shape[0]} samples")

    print("\n" + "="*60)
    print("TRAINING BLENDED ENSEMBLE (ExtraTrees + XGBoost)")
    print("="*60)

    n_jobs = max(1, (os.cpu_count() or 4) // 2)
    # Model 1: XGBoost Regressor
    print(f"  [1/2] Training XGBoost Regressor (800 trees, depth=12, n_jobs={n_jobs})...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=12,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_alpha=0.1,
        reg_lambda=0.5,
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42,
        n_jobs=n_jobs
    )
    model_xgb.fit(X_train, y_train)

    # Model 2: ExtraTrees Regressor — deep trees, full feature set
    print(f"  [2/2] Training ExtraTrees Regressor (400 trees, depth=22, n_jobs={n_jobs})...")
    model_et = ExtraTreesRegressor(
        n_estimators=400,
        max_depth=22,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=n_jobs
    )
    model_et.fit(X_train, y_train)

    # Blended predictions (70% ExtraTrees + 30% XGBoost)
    p_xgb = model_xgb.predict(X_test)
    p_et  = model_et.predict(X_test)

    y_pred = 0.70 * p_et + 0.30 * p_xgb
    y_pred = np.clip(y_pred, 0.0, 1.0)

    # Overall Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    r2_xgb = r2_score(y_test, p_xgb)
    r2_et = r2_score(y_test, p_et)

    print("\n" + "="*60)
    print("BLENDED ENSEMBLE VALIDATION PERFORMANCE")
    print("="*60)
    print(f"  R² Score: {r2:.4f}  ({r2*100:.2f}% variance explained)")
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  MAE:      {mae:.4f}")
    print("="*60)

    # Individual model scores
    print(f"\n  XGBoost R²:       {r2_xgb:.4f}")
    print(f"  ExtraTrees R²:    {r2_et:.4f}")
    print(f"  BLEND (70/30) R²: {r2:.4f}")

    # Feature Importance (from XGBoost)
    importance = model_xgb.get_booster().get_score(importance_type='gain')
    for f in feature_cols:
        if f not in importance:
            importance[f] = 0.0
    imp_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\n--- Top 15 Feature Importances (Gain) ---")
    for k, v in imp_sorted[:15]:
        print(f"  {k:25}: {v:.4f}")

    # Confusion matrix (binning into CDI categories)
    def get_category(score):
        if score < 0.125: return 'No Drought'
        elif score < 0.375: return 'D1'
        elif score < 0.625: return 'D2'
        elif score < 0.875: return 'D3'
        else: return 'D4'

    y_test_cat = y_test.apply(get_category)
    y_pred_cat = pd.Series(y_pred).apply(get_category)

    labels = ['No Drought', 'D1', 'D2', 'D3', 'D4']
    cm = confusion_matrix(y_test_cat, y_pred_cat, labels=labels)

    print("\n--- Confusion Matrix (Binned CDI Categories) ---")
    hdr_label = "True \\ Pred"
    header = f"{hdr_label:>15} | " + " | ".join([f"{l:>10}" for l in labels])
    print(header)
    print("-" * len(header))
    for i, row_label in enumerate(labels):
        row_vals = " | ".join([f"{val:>10}" for val in cm[i]])
        print(f"{row_label:>15} | {row_vals}")

    # Save XGBoost model (JSON)
    xgb_path = os.path.join(model_dir, 'drought_xgb_model.json')
    model_xgb.save_model(xgb_path)
    print(f"\nSaved XGBoost model to {xgb_path}")

    # Save ExtraTrees model (joblib, compressed to avoid MemoryError)
    et_path = os.path.join(model_dir, 'drought_et_model.joblib')
    joblib.dump(model_et, et_path, compress=3)
    print(f"Saved ExtraTrees model to {et_path}")

    # Save metadata
    meta = {
        'training_date': datetime.datetime.now().isoformat(),
        'architecture': 'Blended Ensemble (70% ExtraTrees + 30% XGBoost)',
        'n_features': len(feature_cols),
        'features': feature_cols,
        'blend_weights': {'extratrees': 0.70, 'xgboost': 0.30},
        'metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'xgb_r2': float(r2_xgb),
            'et_r2': float(r2_et),
        },
        'feature_importances_gain': {k: float(v) for k, v in imp_sorted}
    }

    meta_path = os.path.join(model_dir, 'drought_model_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"Saved metadata to {meta_path}")


if __name__ == '__main__':
    main()
