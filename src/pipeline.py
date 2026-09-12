from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

try:
    st = sys.modules.get("streamlit")
    if st is not None:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    else:
        get_script_run_ctx = None
except Exception:  # pragma: no cover - used when Streamlit is not installed.
    st = None
    get_script_run_ctx = None


def _cache_data(**kwargs):
    if st is None or get_script_run_ctx is None or get_script_run_ctx() is None:
        def decorator(func):
            return func
        return decorator
    return st.cache_data(**kwargs)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_PATH = DATA_DIR / "synthetic_patient_data.csv"
RAW_PARQUET_PATH = DATA_DIR / "synthetic_patient_data.parquet"
SCORED_PARQUET_PATH = DATA_DIR / "scored_patient_data.parquet"
PATIENT_MONITOR_PATH = DATA_DIR / "patient_monitor_data.parquet"
PATIENT_INDEX_PATH = DATA_DIR / "patient_monitor_index.parquet"
PATIENT_TIMESERIES_DIR = DATA_DIR / "patient_timeseries"
ALERT_METRICS_PATH = DATA_DIR / "alert_fatigue_metrics.parquet"
ALERT_CI_PATH = DATA_DIR / "alert_fatigue_bootstrap_ci.parquet"
AGE_CI_PATH = DATA_DIR / "age_cohort_bootstrap_ci.parquet"
LEARNING_CURVE_PATH = DATA_DIR / "learning_curve.parquet"
COHORT_SHIFT_PATH = DATA_DIR / "cohort_shift_metrics.parquet"
CALIBRATION_PATH = DATA_DIR / "model3_calibration.parquet"
MISSINGNESS_PATH = DATA_DIR / "missingness_by_age_cohort.parquet"
SUBGROUP_PATHS = {
    "age_cohort": DATA_DIR / "subgroup_age_cohort.parquet",
    "sex": DATA_DIR / "subgroup_sex.parquet",
    "baseline_mobility": DATA_DIR / "subgroup_baseline_mobility.parquet",
    "discharge_category": DATA_DIR / "subgroup_discharge_category.parquet",
}


@_cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    if RAW_PARQUET_PATH.exists():
        return pd.read_parquet(RAW_PARQUET_PATH)
    from .generate_data import save_synthetic_data

    if not DATA_PATH.exists():
        df = save_synthetic_data(DATA_PATH)
        df.to_parquet(RAW_PARQUET_PATH, index=False)
        return df
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


@_cache_data(show_spinner="Building scored synthetic dataset...")
def build_scored_dataset() -> tuple[pd.DataFrame, dict, object]:
    if SCORED_PARQUET_PATH.exists():
        scored = pd.read_parquet(SCORED_PARQUET_PATH)
        splits = {
            name: set(scored.loc[scored["split"] == name, "patient_id"].unique())
            for name in ["train", "validation", "test"]
        }
        return scored, splits, None

    from .evaluation import patient_split
    from .features import add_patient_baselines, add_time_series_features
    from .preprocessing import preprocess_sensor_data
    from .statistical_model import choose_model3_threshold, fit_statistical_model, score_statistical_model
    from .threshold_model import score_threshold_model
    from .timeseries_rules import score_timeseries_rules

    raw = load_raw_data()
    processed = preprocess_sensor_data(raw)
    with_baselines, _ = add_patient_baselines(processed)
    featured = add_time_series_features(with_baselines)
    featured = score_threshold_model(featured)
    featured = score_timeseries_rules(featured)

    split_source = featured[featured["cohort"] == "A"]
    splits = patient_split(split_source)
    train_ids = splits["train"]
    validation_ids = splits["validation"]
    model = fit_statistical_model(featured[featured["patient_id"].isin(train_ids) & (featured["cohort"] == "A")])
    threshold, threshold_table = choose_model3_threshold(
        featured[featured["patient_id"].isin(validation_ids) & (featured["cohort"] == "A")],
        model,
    )
    scored = score_statistical_model(featured, model, threshold=threshold)
    scored["model3_threshold"] = threshold
    scored.attrs["model3_threshold"] = threshold
    scored["split"] = "external_shift"
    for name, ids in splits.items():
        scored.loc[scored["patient_id"].isin(ids), "split"] = name
    return scored, splits, model


@_cache_data(show_spinner=False)
def load_scored_data() -> pd.DataFrame:
    if SCORED_PARQUET_PATH.exists():
        return pd.read_parquet(SCORED_PARQUET_PATH)
    scored, _, _ = build_scored_dataset()
    scored.to_parquet(SCORED_PARQUET_PATH, index=False)
    return scored


@_cache_data(show_spinner=False)
def load_patient_monitor_data() -> pd.DataFrame:
    if PATIENT_MONITOR_PATH.exists():
        return pd.read_parquet(PATIENT_MONITOR_PATH)
    scored = load_scored_data()
    cols = [
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
    return scored[cols]


@_cache_data(show_spinner=False)
def load_patient_index() -> pd.DataFrame:
    if PATIENT_INDEX_PATH.exists():
        return pd.read_parquet(PATIENT_INDEX_PATH)
    monitor = load_patient_monitor_data()
    return (
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


@_cache_data(show_spinner=False)
def load_patient_timeseries(patient_id: str) -> pd.DataFrame:
    path = PATIENT_TIMESERIES_DIR / f"{patient_id}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    monitor = load_patient_monitor_data()
    return monitor[monitor["patient_id"] == patient_id]


@_cache_data(show_spinner=False)
def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        from .precompute import precompute_all

        precompute_all()
    return pd.read_parquet(path)
