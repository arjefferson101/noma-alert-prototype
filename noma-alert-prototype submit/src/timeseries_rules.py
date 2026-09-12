from __future__ import annotations

import numpy as np
import pandas as pd


def recommended_action_for_tier(tier: str) -> str:
    if tier.startswith("Tier 1"):
        return "Urgent escalation. EMS only if a clinician-defined opt-in care plan explicitly authorizes it; this prototype does not contact EMS."
    if tier.startswith("Tier 2"):
        return "Clinician review."
    if tier.startswith("Tier 3"):
        return "Patient plus opt-in patient-designated caregiver/contact check-in. Escalate to clinician review if the pattern persists or worsens over the illustrative interval."
    if tier == "Insufficient data":
        return "Insufficient recent sensor data. Check device connectivity or sensor placement; do not assume low risk from missing data."
    return "Continue monitoring."


def _tier(score: float, insufficient: bool) -> str:
    if insufficient:
        return "Insufficient data"
    if score >= 8:
        return "Tier 1 - urgent escalation"
    if score >= 4:
        return "Tier 2 - clinician review"
    if score >= 2:
        return "Tier 3 - patient/caregiver check-in"
    return "No alert"


def _reasons(row: pd.Series) -> str:
    if row["insufficient_data"]:
        return "Insufficient recent sensor data; the prototype avoids treating missing data as low risk."
    reasons = []
    if row["heart_rate_delta"] > 12:
        reasons.append(f"Heart rate {row['heart_rate_delta']:.0f} bpm above personal baseline")
    if row["respiratory_rate_delta"] > 2.5:
        reasons.append(f"Respiratory rate {row['respiratory_rate_delta']:.1f}/min above personal baseline")
    if row["temperature_delta"] > 0.5:
        reasons.append(f"Temperature {row['temperature_delta']:.1f} F above personal baseline")
    if row["abnormal_duration_min"] >= 20:
        reasons.append(f"Abnormal pattern persisted for {row['abnormal_duration_min']:.0f} minutes")
    if row["tier3_escalation_flag"]:
        reasons.append("Tier 3 pattern persisted or worsened over the illustrative escalation interval")
    if row["concordant_signals"] >= 2:
        reasons.append("Multiple physiologic signals are changing together")
    if row["low_movement_context"] and row["concordant_signals"] >= 1:
        reasons.append("Movement remains low during abnormal vitals")
    if row["movement_associated_hr"] and row["model2_score"] < 4:
        reasons.append("HR rise occurred with high movement and was down-weighted")
    return "; ".join(reasons[:6]) if reasons else "No sustained multivariable pattern."


def score_timeseries_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score = np.zeros(len(out))
    score += np.clip(out["heart_rate_delta"] / 10, 0, 3.2)
    score += np.clip(out["respiratory_rate_delta"] / 2.4, 0, 3.0)
    score += np.clip(out["temperature_delta"] / 0.45, 0, 3.0)
    score += np.where(out["heart_rate_slope30"] > 0.35, 1.0, 0)
    score += np.where(out["respiratory_rate_slope30"] > 0.08, 0.8, 0)
    score += np.where(out["temperature_slope30"] > 0.015, 0.8, 0)
    score += np.where(out["abnormal_duration_min"] >= 20, 1.0, 0)
    score += np.where(out["abnormal_duration_min"] >= 45, 1.0, 0)
    score += np.where(out["concordant_signals"] >= 2, 1.5, 0)
    score += np.where(out["concordant_signals"] >= 3, 1.5, 0)
    score += np.where(out["low_movement_context"] & (out["concordant_signals"] >= 2), 1.0, 0)
    score -= np.where(out["movement_associated_hr"] & (out["concordant_signals"] <= 1), 2.7, 0)
    score -= np.where(out["artifact_processed"], 1.2, 0)
    score = np.where(out["insufficient_data"], np.nan, np.clip(score, 0, 12))
    out["model2_score"] = score
    prior_score = out.groupby("patient_id")["model2_score"].shift(6)
    out["tier3_escalation_flag"] = (
        out["model2_score"].between(2, 4, inclusive="left")
        & (out["abnormal_duration_min"] >= 30)
        & ((out["model2_score"] >= prior_score.fillna(0) + 0.75) | (out["concordant_signals"] >= 2))
    )
    out.loc[out["tier3_escalation_flag"], "model2_score"] = np.maximum(out.loc[out["tier3_escalation_flag"], "model2_score"], 4.0)
    out["model2_tier"] = [_tier(s if not np.isnan(s) else 0, miss) for s, miss in zip(out["model2_score"], out["insufficient_data"])]
    out["model2_alert"] = out["model2_score"].fillna(-1) >= 2
    out["model2_recommended_action"] = out["model2_tier"].map(recommended_action_for_tier)
    out["model2_reason"] = out.apply(_reasons, axis=1)
    return out
