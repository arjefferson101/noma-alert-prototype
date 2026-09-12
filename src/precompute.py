from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd

from .bias_analysis import SPECS, age_cohort_bootstrap_ci, subgroup_metrics
from .cohort_shift import cohort_shift_table
from .evaluation import metric_table, patient_bootstrap_ci, warning_lead_time
from .generate_data import save_synthetic_data
from .learning_curves import learning_curve
from .pipeline import (
    AGE_CI_PATH,
    ALERT_CI_PATH,
    ALERT_METRICS_PATH,
    CALIBRATION_PATH,
    COHORT_SHIFT_PATH,
    DATA_DIR,
    DATA_PATH,
    LEARNING_CURVE_PATH,
    MISSINGNESS_PATH,
    PATIENT_MONITOR_PATH,
    PATIENT_INDEX_PATH,
    PATIENT_TIMESERIES_DIR,
    RAW_PARQUET_PATH,
    SCORED_PARQUET_PATH,
    SUBGROUP_PATHS,
    build_scored_dataset,
    load_raw_data,
)
from .preprocessing import missingness_by_group
from .statistical_model import calibration_summary


META_PATH = DATA_DIR / "precompute_metadata.json"


def precompute_all(n_patients_a: int = 120, n_patients_b: int = 50, seed: int = 42) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw = save_synthetic_data(DATA_PATH, n_patients_a=n_patients_a, n_patients_b=n_patients_b, seed=seed)
    raw.to_parquet(RAW_PARQUET_PATH, index=False)
    for path in [
        SCORED_PARQUET_PATH,
        ALERT_METRICS_PATH,
        ALERT_CI_PATH,
        AGE_CI_PATH,
        LEARNING_CURVE_PATH,
        COHORT_SHIFT_PATH,
        CALIBRATION_PATH,
        MISSINGNESS_PATH,
        PATIENT_MONITOR_PATH,
        PATIENT_INDEX_PATH,
        *SUBGROUP_PATHS.values(),
    ]:
        path.unlink(missing_ok=True)
    if PATIENT_TIMESERIES_DIR.exists():
        shutil.rmtree(PATIENT_TIMESERIES_DIR)

    if hasattr(load_raw_data, "clear"):
        load_raw_data.clear()
    if hasattr(build_scored_dataset, "clear"):
        build_scored_dataset.clear()
    scored, splits, _ = build_scored_dataset()
    scored.to_parquet(SCORED_PARQUET_PATH, index=False)
    monitor_cols = [
        "patient_id",
        "timestamp",
        "age",
        "sex",
        "age_cohort",
        "discharge_category",
        "baseline_mobility",
        "heart_rate",
        "respiratory_rate",
        "temperature",
        "movement",
        "deterioration_event",
        "event_type",
        "model1_alert",
        "model1_tier",
        "model1_recommended_action",
        "model2_alert",
        "model2_tier",
        "model2_score",
        "model2_recommended_action",
        "model2_reason",
        "model3_alert",
        "model3_tier",
        "model3_score",
        "model3_recommended_action",
        "model3_threshold",
    ]
    monitor = scored[monitor_cols]
    monitor.to_parquet(PATIENT_MONITOR_PATH, index=False)
    patient_index = (
        monitor.groupby("patient_id", as_index=False)
        .agg(
            age=("age", "first"),
            age_cohort=("age_cohort", "first"),
            sex=("sex", "first"),
            baseline_mobility=("baseline_mobility", "first"),
            discharge_category=("discharge_category", "first"),
            deterioration_event=("deterioration_event", "max"),
        )
    )
    patient_index.to_parquet(PATIENT_INDEX_PATH, index=False)
    PATIENT_TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
    for patient_id, g in monitor.groupby("patient_id", sort=False):
        g.to_parquet(PATIENT_TIMESERIES_DIR / f"{patient_id}.parquet", index=False)

    test = scored[(scored["cohort"] == "A") & (scored["split"] == "test")]
    alert_metrics = metric_table(test, SPECS)
    for name, (alert_col, _) in SPECS.items():
        alert_metrics.loc[alert_metrics["system"] == name, "warning_lead_time_min"] = warning_lead_time(test, alert_col)
    alert_metrics.to_parquet(ALERT_METRICS_PATH, index=False)

    patient_bootstrap_ci(test, SPECS, n_boot=30).to_parquet(ALERT_CI_PATH, index=False)
    age_cohort_bootstrap_ci(test, n_boot=30).to_parquet(AGE_CI_PATH, index=False)
    missingness_by_group(test, "age_cohort").to_parquet(MISSINGNESS_PATH, index=False)

    for group_col, path in SUBGROUP_PATHS.items():
        subgroup_metrics(test, group_col).to_parquet(path, index=False)

    train = scored[(scored["cohort"] == "A") & (scored["split"] == "train")]
    validation = scored[(scored["cohort"] == "A") & (scored["split"] == "validation")]
    learning_curve(train, validation, test, repeats=2).to_parquet(LEARNING_CURVE_PATH, index=False)

    cohort_shift_table(scored).to_parquet(COHORT_SHIFT_PATH, index=False)
    curve, brier = calibration_summary(test)
    curve["brier_score"] = brier
    curve.to_parquet(CALIBRATION_PATH, index=False)

    metadata = {
        "n_patients_a_background": n_patients_a,
        "n_patients_b_background": n_patients_b,
        "seed": seed,
        "rows": int(len(scored)),
        "patients": int(scored["patient_id"].nunique()),
        "model3_threshold": float(scored["model3_threshold"].iloc[0]),
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "outputs": sorted(str(p.relative_to(DATA_DIR)) for p in DATA_DIR.glob("*.parquet")),
    }
    META_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(precompute_all(), indent=2))
