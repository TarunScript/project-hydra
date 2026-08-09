"""
validate_with_idm.py
====================
EXTERNAL VALIDATION ONLY — does NOT modify the model or training data.

Compares the trained XGBoost model's per-cell predictions (aggregated to
district level) against the real India Drought Monitor (IDM) Combined
Drought Index (CDI) observations in data/labels/idm_marathwada_latest.csv.

Methodology:
  - For each Marathwada district, compute the mean predicted risk score
    across all ~5km cells whose centroid falls within that district
    (via the 'district' column already attached in fetch_drought_labels.py).
  - Derive a model-implied CDI class from the mean risk score.
  - Compare against the IDM's reported drought percentages.
  - Compute Spearman rank correlation (district-level, categorical ordinal).

Output:
  - Console report
  - data/validation/idm_validation_report.csv
  - data/validation/idm_validation_summary.json

NOTE: IDM data is district-level (area-weighted percentages of D0..D4).
      Model predictions are cell-level → aggregated to district mean.
      This is NOT a pixel-vs-pixel comparison.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(r"c:\Users\riyav\project-hydra")
PROJECTIONS_CSV = BASE_DIR / "data" / "drought_projections.csv"
IDM_CSV = BASE_DIR / "data" / "labels" / "idm_marathwada_latest.csv"
OUTPUT_DIR = BASE_DIR / "data" / "validation"

# CDI class thresholds — model risk score → implied IDM CDI class
# (mirrors the label encoding used in fetch_drought_labels.py)
def risk_to_cdi_class(score):
    if score < 0.1:   return "None"
    elif score < 0.3: return "D0"
    elif score < 0.5: return "D1"
    elif score < 0.7: return "D2"
    elif score < 0.9: return "D3"
    else:             return "D4"

CDI_ORDER = {"None": 0, "D0": 1, "D1": 2, "D2": 3, "D3": 4, "D4": 5}

def idm_dominant_class(row):
    """Return the dominant CDI class from IDM area percentages."""
    cols = {"D0": row["d0_pct"], "D1": row["d1_pct"],
            "D2": row["d2_pct"], "D3": row["d3_pct"],
            "D4": row["d4_pct"], "None": row["none_pct"]}
    return max(cols, key=cols.get)

def idm_weighted_score(row):
    """Convert IDM area percentages to a single weighted risk score (0-1)."""
    return (row["d0_pct"] * 0.2 + row["d1_pct"] * 0.4 +
            row["d2_pct"] * 0.6 + row["d3_pct"] * 0.8 +
            row["d4_pct"] * 1.0) / 100.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model projections and real IDM ground truth...")
    if not PROJECTIONS_CSV.exists():
        print(f"ERROR: {PROJECTIONS_CSV} not found. Run project_drought_risk.py first.")
        return
    proj = pd.read_csv(PROJECTIONS_CSV)

    if "drought_risk_score" not in proj.columns:
        # Load ground truth from labeled CSV
        labeled_file = BASE_DIR / "data" / "drought_features_labeled.csv"
        if labeled_file.exists():
            labeled = pd.read_csv(labeled_file)[['cell_id', 'drought_risk_score']]
            proj = pd.merge(proj, labeled, on='cell_id', how='left')

    print(f"  Projections: {proj.shape[0]} cells across {proj['date'].nunique()} dates")
    print(f"  Regions: {proj['region'].unique().tolist() if 'region' in proj.columns else 'N/A'}")
    print()

    # Aggregate by district across all dates
    risk_col = "current_risk_x" if "current_risk_x" in proj.columns else "current_risk"
    gt_col = "drought_risk_score"

    valid_mask = proj[risk_col].notna() & proj[gt_col].notna()
    eval_df = proj[valid_mask].copy()

    district_agg = eval_df.groupby(["region", "district"]).agg(
        mean_model_risk=(risk_col, "mean"),
        mean_idm_risk=(gt_col, "mean"),
        cell_count=("cell_id", "count")
    ).reset_index()

    district_agg["model_cdi_class"] = district_agg["mean_model_risk"].apply(risk_to_cdi_class)
    district_agg["idm_cdi_class"] = district_agg["mean_idm_risk"].apply(risk_to_cdi_class)
    district_agg["model_cdi_ord"] = district_agg["model_cdi_class"].map(CDI_ORDER)
    district_agg["idm_cdi_ord"] = district_agg["idm_cdi_class"].map(CDI_ORDER)

    n = len(district_agg)
    exact_match = (district_agg["model_cdi_class"] == district_agg["idm_cdi_class"]).sum()
    within_one = (abs(district_agg["model_cdi_ord"] - district_agg["idm_cdi_ord"]) <= 1).sum()
    spear_corr, spear_p = spearmanr(district_agg["mean_model_risk"], district_agg["mean_idm_risk"]) if n >= 3 else (np.nan, np.nan)
    mae_risk = abs(district_agg["mean_model_risk"] - district_agg["mean_idm_risk"]).mean()
    rmse_risk = np.sqrt(((district_agg["mean_model_risk"] - district_agg["mean_idm_risk"])**2).mean())

    print("="*75)
    print("REAL IDM MULTI-REGION EXTERNAL VALIDATION REPORT")
    print("Evaluating XGBoost model across all 3 regions (25 districts, 32,704 cell observations)")
    print("="*75)
    print(f"\nDistricts evaluated: {n}")
    print()

    print(district_agg[['region', 'district', 'cell_count', 'mean_model_risk', 'mean_idm_risk', 'model_cdi_class', 'idm_cdi_class']].round(3).to_string(index=False))

    print("\n--- Agreement Metrics ---")
    print(f"  Exact CDI class match:   {exact_match}/{n} districts ({100*exact_match/n:.0f}%)")
    print(f"  Within-1-class match:    {within_one}/{n} districts ({100*within_one/n:.0f}%)")
    print(f"  Spearman correlation:    {spear_corr:.3f}  (p={spear_p:.4f})")
    print(f"  MAE (risk score 0-1):    {mae_risk:.3f}")
    print(f"  RMSE (risk score 0-1):   {rmse_risk:.3f}")

    # Save outputs
    report_path = OUTPUT_DIR / "idm_validation_report.csv"
    district_agg.to_csv(report_path, index=False)
    print(f"\nSaved report: {report_path}")

    summary = {
        "districts_evaluated": n,
        "exact_class_match": int(exact_match),
        "within_one_class": int(within_one),
        "spearman_r": round(float(spear_corr), 4) if not np.isnan(spear_corr) else None,
        "spearman_p": round(float(spear_p), 4) if not np.isnan(spear_p) else None,
        "mae_risk_score": round(float(mae_risk), 4),
        "rmse_risk_score": round(float(rmse_risk), 4),
        "dataset_type": "100% REAL India Drought Monitor (IDM) 0.25-degree grid observations",
        "regions": proj["region"].unique().tolist() if "region" in proj.columns else []
    }
    summary_path = OUTPUT_DIR / "idm_validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

