from __future__ import annotations

from pathlib import Path

import pandas as pd

from .evaluation import patient_split
from .features import add_patient_baselines, add_time_series_features
from .generate_data import save_synthetic_data
from .preprocessing import preprocess_sensor_data
from .statistical_model import choose_model3_threshold, fit_statistical_model, score_statistical_model
from .threshold_model import score_threshold_model
from .timeseries_rules import score_timeseries_rules


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_patient_data.csv"


def load_raw_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return save_synthetic_data(DATA_PATH)
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


def build_scored_dataset() -> tuple[pd.DataFrame, dict, object]:
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
