"""
train_leadtime_models.py
========================
Train two XGBoost models for lead-time drought prediction:
  Model 1: features(W) -> drought_risk(W+7 days)
  Model 2: features(W) -> drought_risk(W+15 days)

CRITICAL: Uses strict TEMPORAL train/test split (no random split).
  - Train on earlier ~75% of weeks
  - Test on most recent ~25% of weeks
  - Split is by date across ALL regions combined

Uses training data from build_lagged_training.py.
"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VALIDATION_DIR = DATA_DIR / "validation"

# Features to use (same 19 as the validated same-day model)
FEATURE_COLS = [
    'bare_ground_frac', 'cropland_frac', 'dry_spell_days', 'et_current_kg_m2',
    'lst_anomaly_c', 'lst_current_c', 'ndvi_anomaly', 'ndvi_current',
    'rain_30d_deficit_mm', 'rain_30d_mm', 'rain_60d_deficit_mm', 'rain_60d_mm',
    'rain_7d_deficit_mm', 'rain_7d_mm', 'rain_90d_deficit_mm', 'rain_90d_mm',
    'sm_rootzone', 'sm_surface', 'urban_built_frac'
]

TARGET_COL = 'target_risk_score'


def temporal_split(df, train_frac=0.75):
    """
    STRICT temporal split by feature_date.
    NO random shuffling - this would leak future information.
    """
    dates = sorted(df['feature_date'].unique())
    split_idx = int(len(dates) * train_frac)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    train = df[df['feature_date'].isin(train_dates)]
    test = df[df['feature_date'].isin(test_dates)]

    return train, test, train_dates, test_dates


def train_model(df, lead_days, model_name):
    """Train XGBoost for one lead horizon."""
    print(f"\n{'='*60}")
    print(f"TRAINING {lead_days}-DAY LEAD MODEL")
    print(f"{'='*60}")

    # Check available features
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        print(f"  WARNING: Missing features: {missing}")
    print(f"  Using {len(available_features)} features")

    # Drop rows with NaN target
    df = df.dropna(subset=[TARGET_COL])
    df = df.dropna(subset=available_features)
    print(f"  Valid rows after NaN drop: {len(df)}")

    if len(df) < 100:
        print(f"  ERROR: Too few rows ({len(df)}) to train meaningfully.")
        return None

    # Strict temporal split
    train, test, train_dates, test_dates = temporal_split(df)

    print(f"\n  TEMPORAL SPLIT (STRICT - no leakage):")
    print(f"    Train dates: {train_dates}")
    print(f"    Test dates:  {test_dates}")
    print(f"    Train rows:  {len(train)}")
    print(f"    Test rows:   {len(test)}")
    print(f"    Split boundary: train <= {train_dates[-1]} | test >= {test_dates[0]}")

    if len(test) < 50:
        print(f"  WARNING: Test set very small ({len(test)} rows). Metrics may be unstable.")

    X_train = train[available_features].values
    y_train = train[TARGET_COL].values
    X_test = test[available_features].values
    y_test = test[TARGET_COL].values

    n_jobs = max(1, (os.cpu_count() or 4) // 2)
    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=n_jobs
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate on temporal test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n  --- TEMPORAL Test Metrics (W+{lead_days}d) ---")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE:  {mae:.4f}")
    print(f"    R2:   {r2:.4f}")

    # Feature importance
    fi = dict(zip(available_features, model.feature_importances_.tolist()))
    print(f"\n  --- Feature Importances (Gain) ---")
    for feat, gain in sorted(fi.items(), key=lambda x: -x[1])[:10]:
        bar = '#' * min(int(gain * 50), 40)
        print(f"    {feat:<25} {gain:>7.4f}  {bar}")

    # Per-region metrics
    if 'region' in test.columns:
        print(f"\n  --- Per-Region Test Metrics ---")
        for region in sorted(test['region'].unique()):
            mask = test['region'] == region
            if mask.sum() < 10:
                continue
            y_r = y_test[mask.values]
            p_r = y_pred[mask.values]
            r2_r = r2_score(y_r, p_r) if len(set(y_r)) > 1 else float('nan')
            print(f"    {region:<20} R2={r2_r:.4f}  MAE={mean_absolute_error(y_r, p_r):.4f}  n={len(y_r)}")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"drought_lead{lead_days}d_model.json"
    model.save_model(str(model_path))
    print(f"\n  Saved model: {model_path}")

    # Save metadata
    meta = {
        "model_type": f"lead_{lead_days}d_drought_prediction",
        "lead_days": lead_days,
        "training_date": datetime.now().isoformat(),
        "features": available_features,
        "n_features": len(available_features),
        "temporal_split": {
            "train_dates": train_dates,
            "test_dates": test_dates,
            "train_rows": len(train),
            "test_rows": len(test),
            "boundary": f"train <= {train_dates[-1]} | test >= {test_dates[0]}"
        },
        "metrics_temporal_test": {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4)
        },
        "feature_importances_gain": {k: round(v, 4) for k, v in
                                      sorted(fi.items(), key=lambda x: -x[1])},
        "regions_in_training": train['region'].unique().tolist() if 'region' in train.columns else [],
        "honest_assessment": (
            f"R2={r2:.3f} on a strict temporal test set. "
            f"This is a {lead_days}-day lead prediction, which is genuinely harder "
            f"than same-day estimation. Even R2=0.5 would be a legitimately strong result "
            f"for a drought early-warning system."
        )
    }
    meta_path = MODELS_DIR / f"drought_lead{lead_days}d_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata: {meta_path}")

    return {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'test': test,
        'y_test': y_test,
        'y_pred': y_pred,
        'features': available_features,
    }


def backtest(result, lead_days):
    """
    Step 8: Real backtest.
    Of the weeks where drought severity increased within N days,
    did the model flag elevated risk in advance?
    """
    if result is None:
        return

    print(f"\n{'='*60}")
    print(f"BACKTEST: {lead_days}-DAY LEAD")
    print(f"{'='*60}")

    test = result['test'].copy()
    test['predicted_risk'] = result['y_pred']
    test['actual_risk'] = result['y_test']

    # Compare: did actual risk go up relative to baseline?
    # We define "severity increased" as actual_risk > 0.2 (at least D0)
    # and "model flagged" as predicted_risk > 0.15

    drought_actual = test['actual_risk'] > 0.2
    drought_flagged = test['predicted_risk'] > 0.15

    if drought_actual.sum() == 0:
        print("  No actual drought events in test period. Cannot compute detection rate.")
        return

    true_pos = (drought_actual & drought_flagged).sum()
    total_drought = drought_actual.sum()
    total_flagged = drought_flagged.sum()
    false_neg = (drought_actual & ~drought_flagged).sum()

    detection_rate = true_pos / total_drought if total_drought > 0 else 0

    print(f"  Definition: 'drought' = risk > 0.2, 'flagged' = predicted > 0.15")
    print(f"  Actual drought cells (test set): {total_drought}")
    print(f"  Model flagged cells:             {total_flagged}")
    print(f"  Correctly flagged (true positive): {true_pos}")
    print(f"  Missed (false negative):           {false_neg}")
    print(f"  DETECTION RATE: {detection_rate:.1%}")
    print()
    print(f"  Interpretation: Of cells that were actually in drought {lead_days} days later,")
    print(f"  the model flagged elevated risk in advance {detection_rate:.0%} of the time.")

    if detection_rate >= 0.6:
        print(f"  -> This supports the early-warning framing.")
    elif detection_rate >= 0.4:
        print(f"  -> Moderate skill. Useful as one input to a decision system, not standalone.")
    else:
        print(f"  -> Weak skill. The '{lead_days}-day prediction' framing is not well supported.")

    # Save backtest results
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    backtest_data = {
        "lead_days": lead_days,
        "drought_threshold": 0.2,
        "flag_threshold": 0.15,
        "actual_drought_cells": int(total_drought),
        "model_flagged_cells": int(total_flagged),
        "true_positives": int(true_pos),
        "false_negatives": int(false_neg),
        "detection_rate": round(detection_rate, 4),
    }
    bt_path = VALIDATION_DIR / f"backtest_lead{lead_days}d.json"
    with open(bt_path, 'w') as f:
        json.dump(backtest_data, f, indent=2)
    print(f"\n  Saved backtest: {bt_path}")


def main():
    for lead_days, input_file in [(7, "training_lead7d.csv"), (15, "training_lead15d.csv")]:
        path = DATA_DIR / input_file
        if not path.exists():
            print(f"Skipping W+{lead_days}d: {path} not found. Run build_lagged_training.py first.")
            continue

        df = pd.read_csv(path)
        print(f"\nLoaded {path.name}: {len(df)} rows, {len(df.columns)} columns")
        print(f"  Regions: {df['region'].unique().tolist() if 'region' in df.columns else '?'}")
        print(f"  Feature dates: {sorted(df['feature_date'].unique())}")

        result = train_model(df, lead_days, f"lead{lead_days}d")
        backtest(result, lead_days)


if __name__ == "__main__":
    main()
