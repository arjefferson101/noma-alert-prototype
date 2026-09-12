from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import metric_table
from .timeseries_rules import recommended_action_for_tier


MODEL_FEATURES = [
    "heart_rate_delta",
    "respiratory_rate_delta",
    "temperature_delta",
    "heart_rate_slope30",
    "respiratory_rate_slope30",
    "temperature_slope30",
    "abnormal_duration_min",
    "concordant_signals",
    "movement",
    "low_movement_context",
    "movement_associated_hr",
]


def fit_statistical_model(train_df: pd.DataFrame) -> Pipeline:
    x = train_df[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train_df["deterioration_event"].astype(int)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=500, class_weight="balanced", random_state=11)),
    ])
    model.fit(x, y)
    return model


def choose_model3_threshold(
    validation_df: pd.DataFrame,
    model: Pipeline,
    min_sensitivity: float = 0.80,
    thresholds: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    if thresholds is None:
        thresholds = np.linspace(0.20, 0.95, 31)
    rows = []
    for threshold in thresholds:
        scored = score_statistical_model(validation_df, model, threshold=float(threshold))
        metrics = metric_table(scored, {"Model 3: logistic regression": ("model3_alert", "model3_score")}).iloc[0]
        rows.append({"threshold": float(threshold), **metrics.to_dict()})
    table = pd.DataFrame(rows)
    eligible = table[table["sensitivity"] >= min_sensitivity].copy()
    if eligible.empty:
        eligible = table.copy()
    eligible = eligible.sort_values(
        ["precision_ppv", "false_clinician_alert_episodes_per_patient_day", "clinician_alert_episodes_per_patient_day"],
        ascending=[False, True, True],
    )
    return float(eligible.iloc[0]["threshold"]), table


def score_statistical_model(df: pd.DataFrame, model: Pipeline, threshold: float = 0.45) -> pd.DataFrame:
    out = df.copy()
    x = out[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    out["model3_score"] = model.predict_proba(x)[:, 1]
    out.loc[out["insufficient_data"], "model3_score"] = np.nan
    out["model3_alert"] = out["model3_score"].fillna(-1) >= threshold
    out["model3_tier"] = "No alert"
    out.loc[(out["model3_score"] >= 0.20) & (out["model3_score"] < threshold), "model3_tier"] = "Tier 3 - patient/caregiver check-in"
    out.loc[(out["model3_score"] >= threshold) & (out["model3_score"] < 0.75), "model3_tier"] = "Tier 2 - clinician review"
    out.loc[out["model3_score"] >= 0.75, "model3_tier"] = "Tier 1 - urgent escalation"
    out.loc[out["insufficient_data"], "model3_tier"] = "Insufficient data"
    out["model3_recommended_action"] = out["model3_tier"].map(recommended_action_for_tier)
    return out


def calibration_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    valid = df.dropna(subset=["model3_score"])
    prob_true, prob_pred = calibration_curve(valid["deterioration_event"].astype(int), valid["model3_score"], n_bins=8, strategy="quantile")
    curve = pd.DataFrame({"mean_predicted_risk": prob_pred, "observed_event_rate": prob_true})
    brier = brier_score_loss(valid["deterioration_event"].astype(int), valid["model3_score"])
    return curve, float(brier)
