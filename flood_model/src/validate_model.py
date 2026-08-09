#!/usr/bin/env python3
"""
validate_model.py — Task 5: Validate Against 2020 Assam Flood Events

Validates the trained model against the known 2020 Assam flood —
one of the worst floods in Assam's history (Jun–Sep 2020).

Checks:
  1. Model correctly predicts flood in 2020 Assam districts
  2. High-risk districts match known heavily-flooded districts
  3. ROC-AUC per year (should be high for 2017, 2020 — bad flood years)
  4. Spatial sanity check: Brahmaputra valley districts score higher

Output:
  models/validation_report.txt
  models/validation_plots/
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

try:
    import xgboost as xgb
except ImportError:
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

from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "models"
VAL_DIR = MODELS_DIR / "validation_plots"

# Known 2020 flood districts in Assam (from news reports / IFI-Impacts)
KNOWN_FLOOD_DISTRICTS_2020 = [
    "DHEMAJI", "LAKHIMPUR", "BISWANATH", "SONITPUR", "DARRANG",
    "BARPETA", "MORIGAON", "NAGAON", "JORHAT", "GOLAGHAT",
    "DHUBRI", "CHIRANG", "BONGAIGAON", "KOKRAJHAR", "BAKSA",
]

# Known low-risk districts (highlands / less flood-prone)
LOW_RISK_DISTRICTS = ["DIMA HASAO", "KARBI ANGLONG", "WEST KARBI ANGLONG"]

FEATURE_COLS = [
    "elevation", "slope", "flow_acc", "dist_to_river", "water_occurrence",
    "rain_monthly_mm", "rain_7d_mm", "rain_3d_mm", "rain_1d_mm",
    "rain_daily_mean_mm", "rain_anomaly",
    "sm_surface", "sm_rootzone",
    "lst_day_k", "et_mm", "built_frac", "water_frac",
    "dfsi_score", "pct_flooded_area", "mean_flood_duration",
    "population", "historical_fatalities", "hist_flood_frequency",
    "month",
]


def load_model_and_data():
    model_path = MODELS_DIR / "flood_model.json"
    if not model_path.exists():
        print("  ✗ flood_model.json not found — run train_flood_model.py first")
        sys.exit(1)

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    print(f"  ✓ Model loaded from {model_path}")

    df = pd.read_csv(FEATURES_DIR / "training_table.csv")
    print(f"  ✓ Training table: {df.shape}")
    return model, df


def predict_risk(model, df):
    """Add risk_score column to dataframe."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(df[available].median())
    df = df.copy()
    df["risk_score"] = model.predict_proba(X)[:, 1]
    return df


def validate_2020(df):
    """Check model correctly identifies 2020 Assam flood districts."""
    print("\n  ── Validation 1: 2020 Assam Flood ──")

    df_2020 = df[df["year"] == 2020].copy()
    if len(df_2020) == 0:
        print("  ⚠ No 2020 data found in training table")
        return {}

    # Average risk score per district per month in 2020 peak months (Jun-Sep)
    peak = df_2020[df_2020["month"].isin([6, 7, 8, 9])]
    if len(peak) == 0:
        print("  ⚠ No peak monsoon 2020 data")
        return {}

    district_risk = peak.groupby("district_name")["risk_score"].mean().sort_values(ascending=False)

    print(f"\n  Risk scores for known flood districts (2020 Jun-Sep):")
    high_risk_caught = 0
    for dist in KNOWN_FLOOD_DISTRICTS_2020:
        score = district_risk.get(dist, None)
        if score is not None:
            caught = "✓" if score > 0.5 else "⚠"
            if score > 0.5:
                high_risk_caught += 1
            print(f"    {caught} {dist:35s}: {score:.3f}")
        else:
            print(f"    ? {dist:35s}: not in data")

    pct = high_risk_caught / len(KNOWN_FLOOD_DISTRICTS_2020) * 100
    print(f"\n  Caught {high_risk_caught}/{len(KNOWN_FLOOD_DISTRICTS_2020)} known flood districts ({pct:.0f}%)")

    print(f"\n  Risk scores for low-risk districts:")
    for dist in LOW_RISK_DISTRICTS:
        score = district_risk.get(dist, None)
        if score is not None:
            status = "✓ low" if score < 0.4 else "⚠ high"
            print(f"    {status} {dist:35s}: {score:.3f}")

    print(f"\n  All districts ranked by risk (2020 Jun-Sep):")
    print(district_risk.to_string())

    return district_risk


def validate_by_year(df):
    """ROC-AUC by year — should be high for known bad flood years."""
    print("\n  ── Validation 2: ROC-AUC by Year ──")

    known_bad_years = [2017, 2019, 2020, 2022]  # known bad flood years in Assam
    results = {}

    for year in sorted(df["year"].unique()):
        year_df = df[df["year"] == year]
        if year_df["is_flood_any"].sum() < 5 or (year_df["is_flood_any"] == 0).sum() < 5:
            continue
        try:
            auc = roc_auc_score(year_df["is_flood_any"], year_df["risk_score"])
            pr_auc = average_precision_score(year_df["is_flood_any"], year_df["risk_score"])
            flag = "🚨" if year in known_bad_years else "  "
            print(f"    {flag} {year}: ROC-AUC={auc:.3f}  PR-AUC={pr_auc:.3f}")
            results[year] = {"roc_auc": auc, "pr_auc": pr_auc}
        except Exception as e:
            print(f"    {year}: ⚠ {e}")

    return results


def validate_spatial(df):
    """Sanity check: Brahmaputra valley cells should score higher than highlands."""
    print("\n  ── Validation 3: Spatial Sanity ──")

    # Low elevation (< 100m) = floodplain, high elevation (> 500m) = highlands
    if "elevation" not in df.columns:
        print("  ⚠ No elevation column")
        return

    low_elev = df[df["elevation"] < 100]["risk_score"].mean()
    mid_elev = df[(df["elevation"] >= 100) & (df["elevation"] < 500)]["risk_score"].mean()
    high_elev = df[df["elevation"] >= 500]["risk_score"].mean()

    print(f"  Mean risk by elevation:")
    print(f"    < 100m  (floodplain): {low_elev:.3f}")
    print(f"    100-500m (foothill):  {mid_elev:.3f}")
    print(f"    > 500m  (highland):   {high_elev:.3f}")

    if low_elev > high_elev:
        print("  ✓ Spatial check PASSED — floodplain > highland risk")
    else:
        print("  ⚠ Spatial check FAILED — check feature engineering")

    # High flow accumulation should mean higher risk
    high_flow = df[df["flow_acc"] > df["flow_acc"].quantile(0.75)]["risk_score"].mean()
    low_flow = df[df["flow_acc"] < df["flow_acc"].quantile(0.25)]["risk_score"].mean()
    print(f"\n  Mean risk by flow accumulation:")
    print(f"    High flow_acc (Q75+): {high_flow:.3f}")
    print(f"    Low flow_acc (Q25-):  {low_flow:.3f}")
    if high_flow > low_flow:
        print("  ✓ Flow check PASSED — high accumulation > low risk")
    else:
        print("  ⚠ Flow check FAILED — check flow_acc feature")


def plot_validation(df, district_risk_2020, yearly_aucs):
    """Generate validation plots."""
    if not HAS_PLOT:
        return
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: Risk score distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Flood Risk Model Validation — Assam 2015–2023", fontsize=13)

    # Risk score by actual label
    flood_scores = df[df["is_flood_any"] == 1]["risk_score"]
    no_flood_scores = df[df["is_flood_any"] == 0]["risk_score"]
    axes[0].hist(no_flood_scores, bins=50, alpha=0.6, color="#3b82f6", label="No Flood")
    axes[0].hist(flood_scores, bins=50, alpha=0.6, color="#ef4444", label="Flood")
    axes[0].set_title("Risk Score Distribution")
    axes[0].set_xlabel("Risk Score")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # AUC by year
    if yearly_aucs:
        years = list(yearly_aucs.keys())
        aucs = [yearly_aucs[y]["roc_auc"] for y in years]
        colors = ["#ef4444" if y in [2017, 2019, 2020, 2022] else "#3b82f6" for y in years]
        axes[1].bar(years, aucs, color=colors)
        axes[1].axhline(0.7, color="gray", linestyle="--", label="0.7 threshold")
        axes[1].set_title("ROC-AUC by Year\n(red = known bad flood years)")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("ROC-AUC")
        axes[1].set_ylim([0, 1])
        axes[1].legend()

    plt.tight_layout()
    path1 = VAL_DIR / "validation_overview.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"\n  ✓ Plot saved: {path1}")

    # Plot 2: 2020 district risk ranking
    if len(district_risk_2020) > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        colors = ["#ef4444" if d in KNOWN_FLOOD_DISTRICTS_2020 else "#64748b"
                  for d in district_risk_2020.index]
        district_risk_2020.plot.barh(ax=ax2, color=colors[::-1])
        ax2.axvline(0.5, color="black", linestyle="--", alpha=0.5, label="Threshold 0.5")
        ax2.set_title("2020 Flood Risk by District (Jun-Sep)\n(red = known flood districts)")
        ax2.set_xlabel("Mean Risk Score")
        ax2.invert_yaxis()
        ax2.legend()
        plt.tight_layout()
        path2 = VAL_DIR / "2020_district_risk.png"
        plt.savefig(path2, dpi=150, bbox_inches="tight")
        print(f"  ✓ Plot saved: {path2}")


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Task 5: Model Validation — 2020 Assam Flood               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    VAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    print("\n  Loading model and data...")
    model, df = load_model_and_data()

    # Predict risk scores
    print("\n  Generating risk scores...")
    df = predict_risk(model, df)
    print(f"  ✓ Risk scores added. Range: [{df['risk_score'].min():.3f}, {df['risk_score'].max():.3f}]")

    # Run validations
    district_risk_2020 = validate_2020(df)
    yearly_aucs = validate_by_year(df)
    validate_spatial(df)

    # Generate plots
    print("\n  Generating validation plots...")
    plot_validation(df, district_risk_2020, yearly_aucs)

    # Save report
    report = f"""
Assam Flood Model Validation Report
=====================================
Validation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Test: 2020 Assam Major Flood

2020 Known Flood District Risk Scores (Jun-Sep avg):
{district_risk_2020.to_string() if len(district_risk_2020) else 'No data'}

ROC-AUC by Year:
{pd.DataFrame(yearly_aucs).T.to_string() if yearly_aucs else 'No data'}
"""
    (MODELS_DIR / "validation_report.txt").write_text(report)
    print(f"\n  ✓ Validation report saved")
    print("\n  ✓ Task 5 complete.")


if __name__ == "__main__":
    main()
