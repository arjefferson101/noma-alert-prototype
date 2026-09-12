from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import metric_table
from .statistical_model import choose_model3_threshold, fit_statistical_model, score_statistical_model


def learning_curve(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
    repeats: int = 4,
    seed: int = 33,
    threshold_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if threshold_grid is None:
        threshold_grid = np.linspace(0.20, 0.85, 8)
    ids = train_df["patient_id"].drop_duplicates().to_numpy()
    rows = []
    for frac in fractions:
        n = max(12, int(len(ids) * frac))
        for rep in range(repeats):
            sample_ids = rng.choice(ids, size=n, replace=False)
            model = fit_statistical_model(train_df[train_df["patient_id"].isin(sample_ids)])
            threshold, _ = choose_model3_threshold(validation_df, model, thresholds=threshold_grid)
            scored = score_statistical_model(test_df, model, threshold=threshold)
            mt = metric_table(scored, {"Model 3: logistic regression": ("model3_alert", "model3_score")}).iloc[0]
            rows.append({
                "training_fraction": frac,
                "training_patients": n,
                "repeat": rep + 1,
                "validation_threshold": threshold,
                "sensitivity": mt["sensitivity"],
                "precision_ppv": mt["precision_ppv"],
                "pr_auc": mt["pr_auc"],
                "false_positive_rate": mt["false_positive_rate"],
                "clinician_alert_episodes_per_patient_day": mt["clinician_alert_episodes_per_patient_day"],
            })
    return pd.DataFrame(rows)
