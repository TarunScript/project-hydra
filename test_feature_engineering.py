"""
test_feature_engineering.py
============================
Testing feature engineering and baseline model performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb


def build_enhanced_features(df_raw):
    df = df_raw.copy()

    # 1. Fill missing values intelligently
    # For environmental metrics, use group median by district/region or global median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0 and col not in ['drought_risk_score', 'idm_cdi_raw']:
            df[col] = df.groupby('district')[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median()).fillna(0)

    # 2. Key Physical Indicators & Ratios
    df['rain_deficit_ratio_7d'] = df['rain_7d_deficit_mm'] / (df['rain_7d_mm'] + 1.0)
    df['rain_deficit_ratio_30d'] = df['rain_30d_deficit_mm'] / (df['rain_30d_mm'] + 1.0)
    df['rain_deficit_ratio_60d'] = df['rain_60d_deficit_mm'] / (df['rain_60d_mm'] + 1.0)
    df['rain_deficit_ratio_90d'] = df['rain_90d_deficit_mm'] / (df['rain_90d_mm'] + 1.0)

    # Cumulative Deficit Index across time windows
    df['cumulative_deficit_weighted'] = (
        0.4 * df['rain_30d_deficit_mm'] +
        0.35 * df['rain_60d_deficit_mm'] +
        0.25 * df['rain_90d_deficit_mm']
    )

    # Soil Moisture Metrics
    df['sm_ratio'] = df['sm_surface'] / (df['sm_rootzone'] + 1e-4)
    df['sm_total'] = df['sm_surface'] + df['sm_rootzone']
    df['sm_deficit_proxy'] = (1.0 - df['sm_rootzone']) * df['rain_30d_deficit_mm']

    # Temperature & Thermal Stress
    df['temp_veg_ratio'] = df['lst_current_c'] / (df['ndvi_current'] + 0.1)
    df['lst_anomaly_sq'] = df['lst_anomaly_c'] ** 2
    df['temp_stress_index'] = df['lst_anomaly_c'] * (1.0 - df['ndvi_current'])

    # Evapotranspiration & Evaporative Stress
    df['evap_stress_index'] = df['et_current_kg_m2'] / (df['rain_30d_mm'] + 1.0)
    df['et_x_sm'] = df['et_current_kg_m2'] * df['sm_rootzone']
    df['et_x_temp'] = df['et_current_kg_m2'] * df['lst_current_c']

    # Dry spell interactions
    df['dry_spell_x_deficit'] = df['dry_spell_days'] * df['rain_30d_deficit_mm']
    df['dry_spell_x_sm'] = df['dry_spell_days'] * (1.0 - df['sm_rootzone'])
    df['dry_spell_sq'] = df['dry_spell_days'] ** 2

    # Cross-term physical interactions
    df['deficit_x_temp'] = df['rain_30d_deficit_mm'] * df['lst_anomaly_c']
    df['deficit_x_ndvi'] = df['rain_30d_deficit_mm'] * df['ndvi_anomaly']
    df['sm_x_temp'] = df['sm_rootzone'] * df['lst_current_c']
    df['sm_x_ndvi'] = df['sm_rootzone'] * df['ndvi_current']

    # Spatial coordinates & polynomial features
    df['lat_sq'] = df['lat'] ** 2
    df['lon_sq'] = df['lon'] ** 2
    df['lat_lon_prod'] = df['lat'] * df['lon']
    df['lat_lon_ratio'] = df['lat'] / (df['lon'] + 1e-4)

    # District spatial micro-climate aggregations (mean, std, max)
    for feat in ['rain_30d_deficit_mm', 'sm_rootzone', 'ndvi_anomaly', 'dry_spell_days', 'lst_anomaly_c']:
        if feat in df.columns:
            df[f'dist_mean_{feat}'] = df.groupby('district')[feat].transform('mean')
            df[f'dist_std_{feat}'] = df.groupby('district')[feat].transform('std').fillna(0)
            df[f'dist_max_{feat}'] = df.groupby('district')[feat].transform('max')

    # Regional Encodings
    if 'region' in df.columns:
        region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=False)
        df = pd.concat([df, region_dummies], axis=1)

    # Temporal & Seasonal features
    if 'date' in df.columns:
        date_dt = pd.to_datetime(df['date'])
        months = date_dt.dt.month
        df['month_sin'] = np.sin(2 * np.pi * months / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * months / 12.0)
        df['day_of_year'] = date_dt.dt.dayofyear
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['is_monsoon'] = months.isin([6, 7, 8, 9]).astype(float)
        df['is_post_monsoon'] = months.isin([10, 11]).astype(float)
        df['is_rabi'] = months.isin([10, 11, 12, 1, 2]).astype(float)
        df['is_summer'] = months.isin([3, 4, 5]).astype(float)

    # Land cover interactions
    if 'cropland_frac' in df.columns and 'bare_ground_frac' in df.columns:
        df['veg_cover_ratio'] = df['cropland_frac'] / (df['bare_ground_frac'] + 0.01)

    return df


def main():
    print("Loading data...")
    raw_df = pd.read_csv('data/drought_features_labeled.csv')
    raw_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    raw_df = raw_df.dropna(subset=['drought_risk_score'])

    df = build_enhanced_features(raw_df)
    target_col = 'drought_risk_score'

    exclude_cols = {'drought_risk_score', 'idm_cdi_raw', 'idm_matched_date', 'label_source', 'date', 'district', 'region', 'cell_id', 'is_synthetic_label'}
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    print(f"Total features engineered: {len(feature_cols)}")
    X = df[feature_cols].fillna(0)
    y = df[target_col]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    n_jobs = max(1, (os.cpu_count() or 4) // 2)
    for name, model in [
        ('XGBoost (depth=12, n=1200)', xgb.XGBRegressor(n_estimators=1200, max_depth=12, learning_rate=0.015, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.5, random_state=42, n_jobs=n_jobs, tree_method='hist')),
        ('LightGBM (leaves=255, n=1200)', lgb.LGBMRegressor(n_estimators=1200, num_leaves=255, learning_rate=0.015, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.5, random_state=42, n_jobs=n_jobs, verbose=-1)),
        ('HistGradientBoosting (max_iter=500)', HistGradientBoostingRegressor(max_iter=500, max_depth=15, max_leaf_nodes=127, learning_rate=0.03, random_state=42)),
        ('ExtraTrees (depth=20, min_leaf=2)', ExtraTreesRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=n_jobs)),
        ('RandomForest (depth=20, min_leaf=2)', RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=n_jobs)),
    ]:
        oof_preds = np.zeros(len(y))
        for train_idx, val_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            model.fit(X_tr, y_tr)
            oof_preds[val_idx] = model.predict(X_va)

        r2 = r2_score(y, oof_preds)
        rmse = np.sqrt(mean_squared_error(y, oof_preds))
        mae = mean_absolute_error(y, oof_preds)
        print(f"{name:40} -> 5-Fold OOF R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

if __name__ == '__main__':
    main()
