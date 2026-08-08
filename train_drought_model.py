"""
train_drought_model.py
Trains an XGBoost regressor for predicting drought risk index.
Note: "Forecast" in this project context implies trend + climatology projection, NOT a learned predictor. This script trains a diagnostic model.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
import json
import datetime
import os

def main():
    data_path = 'data/drought_features_labeled.csv'
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please ensure the data exists before running.")
        return

    # Clean infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    target_col = 'drought_risk_score'
    exclude_cols = ['lat', 'lon', 'region', 'date', 'landcover_class', target_col]

    feature_cols = [
        'bare_ground_frac', 'cropland_frac', 'dry_spell_days', 'et_current_kg_m2',
        'lst_anomaly_c', 'lst_current_c', 'ndvi_anomaly', 'ndvi_current',
        'rain_30d_deficit_mm', 'rain_30d_mm', 'rain_60d_deficit_mm', 'rain_60d_mm',
        'rain_7d_deficit_mm', 'rain_7d_mm', 'rain_90d_deficit_mm', 'rain_90d_mm',
        'sm_rootzone', 'sm_surface', 'urban_built_frac'
    ]
    
    # Check if all required features exist
    missing_feats = [f for f in feature_cols if f not in df.columns]
    if missing_feats:
        print(f"Warning: Missing features in dataset: {missing_feats}")
        feature_cols = [f for f in feature_cols if f in df.columns]

    print(f"Using {len(feature_cols)} features: {feature_cols}")

    X = df[feature_cols]
    y = df[target_col]
    stratify_col = df['region'] if 'region' in df.columns else None

    # Stratified split by region
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_col
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_test.shape[0]} samples")

    # Initialize XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42
    )

    print("Training model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Validation Metrics ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    # Feature Importance
    importance = model.get_booster().get_score(importance_type='gain')
    # If a feature wasn't used, it might not be in the dictionary
    for f in feature_cols:
        if f not in importance:
            importance[f] = 0.0
            
    imp_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\n--- Feature Importances (Gain) ---")
    for k, v in imp_sorted:
        print(f"{k:25}: {v:.4f}")

    # Confusion matrix (binning into CDI categories D0=0, D1=0.25, D2=0.5, D3=0.75, D4=1.0)
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
    
    print("\n--- Confusion Matrix (Binned) ---")
    hdr_label = "True \ Pred"
    header = f"{hdr_label:>15} | " + " | ".join([f"{l:>10}" for l in labels])
    print(header)
    print("-" * len(header))
    for i, row_label in enumerate(labels):
        row_vals = " | ".join([f"{val:>10}" for val in cm[i]])
        print(f"{row_label:>15} | {row_vals}")

    # Save model
    model_path = os.path.join(model_dir, 'drought_xgb_model.json')
    model.save_model(model_path)
    print(f"\nSaved model to {model_path}")

    # Save metadata
    meta = {
        'training_date': datetime.datetime.now().isoformat(),
        'features': feature_cols,
        'metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        },
        'feature_importances_gain': {k: float(v) for k, v in imp_sorted}
    }
    
    meta_path = os.path.join(model_dir, 'drought_model_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"Saved metadata to {meta_path}")

if __name__ == '__main__':
    main()
