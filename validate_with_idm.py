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

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("Loading model projections...")
    if not PROJECTIONS_CSV.exists():
        print(f"ERROR: {PROJECTIONS_CSV} not found. Run project_drought_risk.py first.")
        return
    proj = pd.read_csv(PROJECTIONS_CSV)

    print("Loading IDM observations...")
    if not IDM_CSV.exists():
        print(f"ERROR: {IDM_CSV} not found.")
        return
    idm = pd.read_csv(IDM_CSV)

    print(f"  Projections: {proj.shape[0]} cells across {proj['date'].nunique()} dates")
    print(f"  IDM: {len(idm)} districts")
    print()

    # -----------------------------------------------------------------------
    # Aggregate model predictions to district level
    # Use the most recent date's predictions
    # -----------------------------------------------------------------------
    # The 'district' column comes from fetch_drought_labels.py's
    # lat/lon → district mapping (pseudo-random but deterministic)
    if "district" not in proj.columns:
        print("ERROR: 'district' column not in projections CSV.")
        print("       Re-run fetch_drought_labels.py and project_drought_risk.py.")
        return

    latest_date = sorted(proj["date"].unique())[-1]
    print(f"Using most recent date for comparison: {latest_date}")
    proj_latest = proj[proj["date"] == latest_date].copy()

    # Use current_risk_x (the primary risk column from the merge)
    risk_col = "current_risk_x" if "current_risk_x" in proj_latest.columns else "current_risk"

    # Aggregate: mean risk score and cell count per district
    district_agg = (proj_latest
                    .groupby("district")[risk_col]
                    .agg(mean_model_risk="mean", cell_count="count")
                    .reset_index())
    district_agg.rename(columns={"district": "District"}, inplace=True)

    print("\n--- Model predictions aggregated to district ---")
    print(district_agg.to_string(index=False))

    # -----------------------------------------------------------------------
    # Process IDM data
    # -----------------------------------------------------------------------
    idm["idm_dominant_class"] = idm.apply(idm_dominant_class, axis=1)
    idm["idm_weighted_score"] = idm.apply(idm_weighted_score, axis=1)

    # Normalise district names for join
    district_agg["district_key"] = district_agg["District"].str.strip().str.title()
    idm["district_key"] = idm["district"].str.strip().str.title()

    # Handle Aurangabad / Chhatrapati Sambhajinagar rename
    idm["district_key"] = idm["district_key"].replace(
        {"Chhatrapati Sambhajinagar": "Aurangabad"}
    )

    # -----------------------------------------------------------------------
    # Join
    # -----------------------------------------------------------------------
    merged = pd.merge(district_agg, idm, on="district_key", how="inner")

    if merged.empty:
        print("\nWARNING: No districts matched between model and IDM data.")
        print(f"  Model districts: {district_agg['district_key'].tolist()}")
        print(f"  IDM districts:   {idm['district_key'].tolist()}")
        return

    merged["model_cdi_class"] = merged["mean_model_risk"].apply(risk_to_cdi_class)
    merged["model_cdi_ordinal"] = merged["model_cdi_class"].map(CDI_ORDER)
    merged["idm_cdi_ordinal"]   = merged["idm_dominant_class"].map(CDI_ORDER)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    n = len(merged)
    exact_match = (merged["model_cdi_class"] == merged["idm_dominant_class"]).sum()
    within_one  = (abs(merged["model_cdi_ordinal"] - merged["idm_cdi_ordinal"]) <= 1).sum()

    spear_corr, spear_p = spearmanr(
        merged["mean_model_risk"], merged["idm_weighted_score"]
    ) if n >= 3 else (float("nan"), float("nan"))

    mae_risk = abs(merged["mean_model_risk"] - merged["idm_weighted_score"]).mean()
    rmse_risk = np.sqrt(((merged["mean_model_risk"] - merged["idm_weighted_score"])**2).mean())

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("IDM EXTERNAL VALIDATION REPORT")
    print(f"Comparison date (model): {latest_date}")
    print(f"IDM data: latest weekly observation (data/labels/idm_marathwada_latest.csv)")
    print("="*70)
    print(f"\nDistricts matched: {n} / {len(idm)}")
    print()

    # Per-district table
    report_cols = ["District", "cell_count", "mean_model_risk", "model_cdi_class",
                   "idm_dominant_class", "idm_weighted_score",
                   "d0_pct", "d1_pct", "d2_pct", "d3_pct", "d4_pct"]
    report = merged[[c for c in report_cols if c in merged.columns]].copy()
    report["mean_model_risk"] = report["mean_model_risk"].round(3)
    report["idm_weighted_score"] = report["idm_weighted_score"].round(3)
    print(report.to_string(index=False))

    print("\n--- Agreement Metrics ---")
    print(f"  Exact CDI class match:   {exact_match}/{n} districts ({100*exact_match/n:.0f}%)")
    print(f"  Within-1-class match:    {within_one}/{n} districts ({100*within_one/n:.0f}%)")
    print(f"  Spearman correlation:    {spear_corr:.3f}  (p={spear_p:.3f})")
    print(f"  MAE (risk score 0-1):    {mae_risk:.3f}")
    print(f"  RMSE (risk score 0-1):   {rmse_risk:.3f}")

    print("\nIMPORTANT CAVEATS:")
    print("  1. Model labels are SYNTHETIC (derived from rain deficit + NDVI anomaly).")
    print("     High agreement may reflect feature correlation, not true drought skill.")
    print("  2. IDM date vs. model date may differ - IDM is 'latest week', model is", latest_date)
    print("  3. District assignment uses a pseudo-random lat/lon->district heuristic.")
    print("     For production, use a real district shapefile spatial join.")
    print("  4. This comparison is for DEMO/TRANSPARENCY purposes only.")
    print("     Do NOT interpret these metrics as model validation on independent data.")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    report_path = OUTPUT_DIR / "idm_validation_report.csv"
    merged.to_csv(report_path, index=False)
    print(f"\nSaved report: {report_path}")

    summary = {
        "comparison_date_model": latest_date,
        "idm_data_file": str(IDM_CSV),
        "districts_matched": n,
        "districts_total_idm": len(idm),
        "exact_class_match": int(exact_match),
        "within_one_class": int(within_one),
        "spearman_r": round(float(spear_corr), 4) if not np.isnan(spear_corr) else None,
        "spearman_p": round(float(spear_p), 4) if not np.isnan(spear_p) else None,
        "mae_risk_score": round(float(mae_risk), 4),
        "rmse_risk_score": round(float(rmse_risk), 4),
        "caveats": [
            "Synthetic labels — model trained on features, not real CDI.",
            "IDM is district-level; model predictions aggregated from cell-level.",
            "District assignment is pseudo-random heuristic, not shapefile-based.",
            "Temporal mismatch possible between IDM observation week and model date."
        ]
    }
    summary_path = OUTPUT_DIR / "idm_validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
