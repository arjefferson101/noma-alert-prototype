from __future__ import annotations

import pandas as pd

from .timeseries_rules import recommended_action_for_tier


def score_threshold_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw_alert = (
        (out["heart_rate"] >= 100)
        | (out["respiratory_rate"] >= 22)
        | (out["temperature"] >= 99.5)
    )
    out["model1_alert"] = raw_alert & (~out["insufficient_data"])
    out["model1_score"] = out["model1_alert"].astype(float)
    out["model1_tier"] = "No alert"
    out.loc[out["model1_alert"], "model1_tier"] = "Tier 2 - threshold alert"
    out.loc[out["insufficient_data"], "model1_tier"] = "Insufficient data"
    out["model1_recommended_action"] = out["model1_tier"].map(recommended_action_for_tier).fillna("Clinician review.")
    return out
