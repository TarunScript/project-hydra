#!/usr/bin/env python3
"""
train_flood_model.py — Task 4: Train XGBoost Flood Risk Model

Trains an XGBoost classifier on the merged training_table.csv.

Per the implementation plan:
  - Model: XGBoost (tabular, binary classification)
  - Target: is_flood_any (1 = flood occurred that month, 0 = no flood)
  - Handles class imbalance via scale_pos_weight
  - Output: saved model + feature importance plot

Output:
  models/flood_model.json      — trained XGBoost model
  models/feature_importance.png
  models/model_report.txt
"""

import pandas as pd
import numpy as np
import joblib
import json
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix
)

try:
    import xgboost as xgb
except ImportError:
    print("Installing xgboost...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost as xgb

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "models"

# Feature columns (all GEE + district static, excluding identifiers and targets)
FEATURE_COLS = [
    # GEE Static
    "elevation", "slope", "flow_acc", "dist_to_river", "water_occurrence",
    # GEE Dynamic
    "rain_monthly_mm", "rain_7d_mm", "rain_3d_mm", "rain_1d_mm",
    "rain_daily_mean_mm", "rain_anomaly",
    "sm_surface", "sm_rootzone",
    "lst_day_k",
    "et_mm",
    "built_frac", "water_frac",
    # District Static
    "dfsi_score", "pct_flooded_area", "mean_flood_duration",
    "population", "historical_fatalities", "hist_flood_frequency",
    # Temporal
    "month",
]

TARGET_COL = "is_flood_any"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_training_table():
    path = FEATURES_DIR / "training_table.csv"
    if not path.exists():
        print(f"  ✗ training_table.csv not found — run build_training_table.py first")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"  ✓ Loaded training table: {df.shape}")
    return df


def prepare_features(df):
    """Select and validate feature columns."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]

    if missing:
        print(f"  ⚠ Missing feature columns (will skip): {missing}")

    X = df[available].copy()
    y = df[TARGET_COL].copy()

    # Drop rows with any remaining nulls
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"  ✓ Feature matrix: {X.shape} ({len(available)} features)")
    print(f"  ✓ Target: {y.sum()} flood / {(y == 0).sum()} no-flood rows")
    print(f"  Features used: {available}")

    return X, y, available


def compute_scale_pos_weight(y):
    """XGBoost scale_pos_weight = negative_count / positive_count."""
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    spw = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight = {spw:.2f} (n_neg={n_neg}, n_pos={n_pos})")
    return spw


def train_model(X_train, y_train, scale_pos_weight):
    """Train XGBoost with parameters from the implementation plan."""
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",      # AUC-PR better for imbalanced data
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("  Training XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=50,
    )
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """Print full evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No Flood", "Flood"])

    print("\n  ── Evaluation Results ──")
    print(f"  ROC-AUC:  {roc_auc:.4f}")
    print(f"  PR-AUC:   {pr_auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"\n  Classification Report:")
    print(report)

    return roc_auc, pr_auc, report


def save_outputs(model, feature_names, roc_auc, pr_auc, report, X_test, y_test):
    """Save model, importance plot, and report."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = MODELS_DIR / "flood_model.json"
    model.save_model(str(model_path))
    print(f"\n  ✓ Model saved: {model_path}")

    # Save feature importance
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    feat_imp_path = MODELS_DIR / "feature_importance.csv"
    feat_imp.to_csv(feat_imp_path, index=False)
    print(f"  ✓ Feature importance saved: {feat_imp_path}")

    print(f"\n  Top 10 features:")
    for _, row in feat_imp.head(10).iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"    {row['feature']:35s} {row['importance']:.4f} {bar}")

    # Plot feature importance
    if HAS_PLOT:
        fig, ax = plt.subplots(figsize=(10, 6))
        feat_imp.head(15).plot.barh(x="feature", y="importance", ax=ax, color="#2563eb")
        ax.set_title("XGBoost Feature Importance — Assam Flood Model", fontsize=13)
        ax.set_xlabel("Importance Score")
        ax.invert_yaxis()
        plt.tight_layout()
        plot_path = MODELS_DIR / "feature_importance.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Importance plot saved: {plot_path}")

    # Save text report
    report_text = f"""
Assam Flood Risk Model — Training Report
==========================================
Model: XGBoost
Target: is_flood_any (binary)
Features: {len(feature_names)}
Training date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Metrics on Hold-out Test Set (20%):
  ROC-AUC:  {roc_auc:.4f}
  PR-AUC:   {pr_auc:.4f}

{report}

Feature Importance (top 15):
{feat_imp.head(15).to_string(index=False)}
"""
    report_path = MODELS_DIR / "model_report.txt"
    report_path.write_text(report_text)
    print(f"  ✓ Report saved: {report_path}")


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Task 4: Train XGBoost Flood Risk Model                    ║")
    print("║  Region: Assam | 2015–2023                                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Load
    print("\n  Loading training table...")
    df = load_training_table()

    # Prepare
    print("\n  Preparing features...")
    X, y, feature_names = prepare_features(df)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # Scale pos weight
    spw = compute_scale_pos_weight(y_train)

    # Train
    print("\n" + "=" * 60)
    print("  Training...")
    print("=" * 60)
    model = train_model(X_train, y_train, spw)

    # Evaluate
    roc_auc, pr_auc, report = evaluate_model(model, X_test, y_test, feature_names)

    # Save
    save_outputs(model, feature_names, roc_auc, pr_auc, report, X_test, y_test)

    print("\n  ✓ Task 4 complete. Model ready for validation and inference.")


if __name__ == "__main__":
    main()
