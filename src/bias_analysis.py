from __future__ import annotations

import pandas as pd

from .evaluation import metric_table, patient_bootstrap_ci


SPECS = {
    "Model 1: universal thresholds": ("model1_alert", "model1_score"),
    "Model 2: personalized rules": ("model2_alert", "model2_score"),
    "Model 3: logistic regression": ("model3_alert", "model3_score"),
}


def subgroup_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, g in df.groupby(group_col):
        mt = metric_table(g, SPECS)
        mt.insert(0, group_col, value)
        mt["patients"] = g["patient_id"].nunique()
        mt["events"] = int(g.groupby("patient_id")["deterioration_event"].max().sum())
        mt["uncertain_small_group"] = (mt["patients"] < 25) | (mt["events"] < 5)
        rows.append(mt)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def age_cohort_bootstrap_ci(df: pd.DataFrame, n_boot: int = 80, min_patients: int = 25, min_events: int = 5) -> pd.DataFrame:
    rows = []
    for cohort, g in df.groupby("age_cohort"):
        patients = g["patient_id"].nunique()
        events = int(g.groupby("patient_id")["deterioration_event"].max().sum())
        if patients < min_patients or events < min_events:
            continue
        ci = patient_bootstrap_ci(g, SPECS, n_boot=n_boot)
        ci.insert(0, "age_cohort", cohort)
        ci["patients"] = patients
        ci["events"] = events
        rows.append(ci)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
