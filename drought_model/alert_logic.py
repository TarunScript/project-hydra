"""
alert_logic.py — Section 7 trigger logic, generalized so the flood model can
import the same functions (Person C's "Builds the alert trigger logic" task
in the Hour 10-16 column applies to both models, not just drought).

Risk level thresholds on the 0-1 risk_score (tune against validation data,
per the plan's caveat that these are "a starting proposal"):
    Low       risk_score < 0.25   -> no alert
    Moderate  0.25 <= score < 0.5 -> early advisory
    High      0.5  <= score < 0.75 -> prepare
    Severe    risk_score >= 0.75  -> evacuate / immediate action

For drought specifically, "days to event" doesn't map cleanly onto a single
moment (a drought is a slow-onset condition, not a discrete event like a
flood crest) — so days_to_event here is interpreted as "days until the
projected risk_score is expected to cross into the next-higher band",
estimated from forecast_projection.py's horizon outputs.
"""
from dataclasses import dataclass

RISK_THRESHOLDS = {
    "Low": (0.0, 0.25),
    "Moderate": (0.25, 0.5),
    "High": (0.5, 0.75),
    "Severe": (0.75, 1.01),
}

SUGGESTED_ACTIONS = {
    "Low": "No alert",
    "Moderate": "Early advisory — monitor conditions",
    "High": "Prepare — store water, reduce non-essential use, protect livestock water sources",
    "Severe": "Severe water scarcity risk — activate local contingency plan, prioritize drinking water",
}


def risk_score_to_level(score: float) -> str:
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return level
    return "Severe" if score >= 0.75 else "Low"


@dataclass
class DroughtAlert:
    region: str
    cell_id: str
    risk_score: float
    risk_level: str
    days_to_next_level: int | None
    suggested_action: str
    message: str


def estimate_days_to_next_level(projection_df, cell_id: str) -> int | None:
    """
    Given forecast_projection.py's per-cell, per-horizon output, find the
    smallest horizon_days at which risk_level increases beyond the current
    (horizon=0 / most recent observed) level. Returns None if no crossing is
    projected within the available horizons.
    """
    cell_rows = projection_df[projection_df["cell_id"] == cell_id].sort_values("horizon_days")
    if cell_rows.empty:
        return None
    current_level = risk_score_to_level(cell_rows.iloc[0]["risk_score"])
    current_rank = list(RISK_THRESHOLDS.keys()).index(current_level)

    for _, row in cell_rows.iterrows():
        level = risk_score_to_level(row["risk_score"])
        rank = list(RISK_THRESHOLDS.keys()).index(level)
        if rank > current_rank:
            return int(row["horizon_days"])
    return None


def build_alert(region: str, cell_id: str, risk_score: float, days_to_next_level: int = None) -> DroughtAlert:
    level = risk_score_to_level(risk_score)
    action = SUGGESTED_ACTIONS[level]
    if level == "Low":
        message = f"{region} (cell {cell_id}): conditions normal. No action needed."
    else:
        horizon_note = f" Risk may increase within ~{days_to_next_level} days." if days_to_next_level else ""
        message = f"{region} (cell {cell_id}): {level} drought risk.{horizon_note} {action}."
    return DroughtAlert(
        region=region,
        cell_id=cell_id,
        risk_score=risk_score,
        risk_level=level,
        days_to_next_level=days_to_next_level,
        suggested_action=action,
        message=message,
    )


def build_alerts_for_region(region: str, current_risk_df, projection_df=None) -> list:
    """
    current_risk_df: DataFrame with ['cell_id', 'risk_score'] for the most
        recent observed date.
    projection_df: optional output of forecast_projection.project_all_cells,
        used to estimate days_to_next_level.
    """
    alerts = []
    for _, row in current_risk_df.iterrows():
        days = None
        if projection_df is not None:
            days = estimate_days_to_next_level(projection_df, row["cell_id"])
        alerts.append(build_alert(region, row["cell_id"], row["risk_score"], days))
    return alerts
