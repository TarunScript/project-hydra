import os
import pandas as pd
import numpy as np
import xgboost as xgb
import glob
from pathlib import Path

# IMPORTANT LIMITATION:
# This script produces projected risk scores by comparing current deficit trajectory
# against climatology.
# This is NOT a learned forecast model. It is a simple trend extrapolation.
# There is no skillful 60-90 day drought forecast even in operational systems.

def main():
    base_dir = Path(r"c:\Users\riyav\project-hydra")
    model_path = base_dir / "models" / "drought_xgb_model.json"
    data_dir = base_dir / "data"
    
    if not model_path.exists():
        print(f"Model not found at {model_path}. Exiting.")
        return
        
    print(f"Loading model from {model_path}...")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # Find the most recent feature CSV
    csv_files = glob.glob(str(data_dir / "drought_features_*.csv"))
    if not csv_files:
        print(f"No feature CSV files found in {data_dir}. Exiting.")
        return
        
    # Just take the first one or sort by modified time
    latest_csv = sorted(csv_files, key=os.path.getmtime, reverse=True)[0]
    print(f"Processing data from {latest_csv}...")
    
    df = pd.read_csv(latest_csv)
    
    # Check if 'cell_id' is present, if not create one
    if 'cell_id' not in df.columns:
        if 'region' in df.columns:
            df['cell_id'] = df['region'] + "_" + df['lat'].astype(str) + "_" + df['lon'].astype(str)
        else:
            df['cell_id'] = "cell_" + df['lat'].astype(str) + "_" + df['lon'].astype(str)
            
    # Exact feature columns used in training — must match train_drought_model.py
    feature_cols = [
        'bare_ground_frac', 'cropland_frac', 'dry_spell_days', 'et_current_kg_m2',
        'lst_anomaly_c', 'lst_current_c', 'ndvi_anomaly', 'ndvi_current',
        'rain_30d_deficit_mm', 'rain_30d_mm', 'rain_60d_deficit_mm', 'rain_60d_mm',
        'rain_7d_deficit_mm', 'rain_7d_mm', 'rain_90d_deficit_mm', 'rain_90d_mm',
        'sm_rootzone', 'sm_surface', 'urban_built_frac'
    ]
    # Only use features that actually exist in the data
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Using {len(feature_cols)} features for prediction")
    
    # Predict current risk
    df['current_risk'] = model.predict(df[feature_cols])
    df['current_risk'] = df['current_risk'].clip(0, 1)
    
    # --- Project 7 days forward ---
    df_7d = df.copy()
    # Compute current rainfall deficit rate = rain_30d_deficit_mm / 30 (mm/day shortfall)
    deficit_rate = df['rain_30d_deficit_mm'] / 30.0
    
    if 'rain_7d_deficit_mm' in df_7d.columns:
        df_7d['rain_7d_deficit_mm'] = df_7d['rain_7d_deficit_mm'] + (deficit_rate * 7)
    
    # Adjust soil moisture and NDVI anomaly proportionally (simple linear trend)
    # Assume 1% degradation per day of deficit rate (heuristic)
    if 'sm_rootzone' in df_7d.columns:
        df_7d['sm_rootzone'] = df_7d['sm_rootzone'] * (1 + (deficit_rate / 100.0) * 7)
    if 'ndvi_anomaly' in df_7d.columns:
        df_7d['ndvi_anomaly'] = df_7d['ndvi_anomaly'] + (deficit_rate / 100.0) * 7
        
    risk_7d = model.predict(df_7d[feature_cols])
    
    # --- Project 15 days forward ---
    df_15d = df.copy()
    if 'rain_7d_deficit_mm' in df_15d.columns:
        # For 15 days, we might just scale up the 7d deficit or apply to 30d deficit
        df_15d['rain_7d_deficit_mm'] = df_15d['rain_7d_deficit_mm'] + (deficit_rate * 15)
        
    if 'sm_rootzone' in df_15d.columns:
        df_15d['sm_rootzone'] = df_15d['sm_rootzone'] * (1 + (deficit_rate / 100.0) * 15)
    if 'ndvi_anomaly' in df_15d.columns:
        df_15d['ndvi_anomaly'] = df_15d['ndvi_anomaly'] + (deficit_rate / 100.0) * 15
        
    risk_15d = model.predict(df_15d[feature_cols])
    
    # Output formatting
    output_df = pd.DataFrame({
        'cell_id': df['cell_id'],
        'lat': df['lat'],
        'lon': df['lon'],
        'current_risk': df['current_risk'],
        'risk_7d': np.clip(risk_7d, 0, 1),
        'risk_15d': np.clip(risk_15d, 0, 1)
    })
    
    # The projection_label must always say: 'Trend projection (not a forecast) — based on current deficit trajectory vs. climatology'
    output_df['projection_label'] = 'Trend projection (not a forecast) — based on current deficit trajectory vs. climatology'
    
    # Merge back any useful features for export (like date, region, raw features)
    export_df = pd.merge(df, output_df.drop(columns=['lat', 'lon']), on='cell_id', how='left')
    
    out_path = data_dir / "drought_projections.csv"
    export_df.to_csv(out_path, index=False)
    print(f"Saved projections to {out_path}")
    print("NOTE: These projections are NOT a learned forecast model. They are a simple trend extrapolation.")

if __name__ == "__main__":
    main()
