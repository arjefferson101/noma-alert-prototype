from __future__ import annotations

import pandas as pd

from src.evaluation import group_alert_episodes, metric_table, patient_split, warning_lead_time
from src.features import add_patient_baselines, add_time_series_features
from src.generate_data import generate_synthetic_data
from src.preprocessing import preprocess_sensor_data
from src.threshold_model import score_threshold_model
from src.timeseries_rules import score_timeseries_rules
from src.statistical_model import choose_model3_threshold, fit_statistical_model


def _featured_demo():
    raw = generate_synthetic_data(n_patients_a=12, n_patients_b=4, seed=123)
    processed = preprocess_sensor_data(raw)
    based, _ = add_patient_baselines(processed)
    featured = add_time_series_features(based)
    return score_timeseries_rules(score_threshold_model(featured))


def test_patient_split_keeps_patients_disjoint():
    df = generate_synthetic_data(n_patients_a=40, n_patients_b=0, seed=1)
    splits = patient_split(df)
    assert splits["train"].isdisjoint(splits["validation"])
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["validation"].isdisjoint(splits["test"])


def test_synthetic_generation_is_reproducible():
    a = generate_synthetic_data(n_patients_a=6, n_patients_b=2, seed=5)
    b = generate_synthetic_data(n_patients_a=6, n_patients_b=2, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_missing_data_is_not_confident_low_risk():
    scored = _featured_demo()
    missing = scored[scored["insufficient_data"]]
    assert not missing.empty
    assert (missing["model2_tier"] == "Insufficient data").all()


def test_preprocessing_uses_only_past_values_for_short_gaps():
    df = pd.DataFrame({
        "patient_id": ["p1"] * 5,
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="5min"),
        "age": [50] * 5,
        "sex": ["female"] * 5,
        "age_cohort": ["40-64"] * 5,
        "discharge_category": ["general_medical"] * 5,
        "baseline_mobility": ["medium"] * 5,
        "heart_rate": [70.0, None, None, 100.0, 101.0],
        "respiratory_rate": [16.0, None, None, 20.0, 20.0],
        "temperature": [98.0, None, None, 99.0, 99.0],
        "movement": [0.2, None, None, 0.8, 0.8],
        "deterioration_event": [False] * 5,
        "event_type": ["stable"] * 5,
        "event_phase": ["none"] * 5,
        "artifact_flag": [False] * 5,
        "cohort": ["A"] * 5,
    })
    processed = preprocess_sensor_data(df)
    assert processed.loc[1, "heart_rate"] == 70.0
    assert processed.loc[2, "heart_rate"] == 70.0
    assert processed.loc[1, "heart_rate"] != 100.0


def test_long_current_gap_is_marked_insufficient_without_backward_fill():
    rows = 9
    df = pd.DataFrame({
        "patient_id": ["p1"] * rows,
        "timestamp": pd.date_range("2026-01-01", periods=rows, freq="5min"),
        "age": [50] * rows,
        "sex": ["female"] * rows,
        "age_cohort": ["40-64"] * rows,
        "discharge_category": ["general_medical"] * rows,
        "baseline_mobility": ["medium"] * rows,
        "heart_rate": [70.0] + [None] * 7 + [90.0],
        "respiratory_rate": [16.0] + [None] * 7 + [18.0],
        "temperature": [98.0] + [None] * 7 + [98.5],
        "movement": [0.2] + [None] * 7 + [0.3],
        "deterioration_event": [False] * rows,
        "event_type": ["stable"] * rows,
        "event_phase": ["none"] * rows,
        "artifact_flag": [False] * rows,
        "cohort": ["A"] * rows,
    })
    processed = preprocess_sensor_data(df)
    assert processed.loc[7, "insufficient_data"]
    assert pd.isna(processed.loc[7, "heart_rate"])
    assert processed.loc[8, "heart_rate"] == 90.0


def test_sustained_multimodal_scores_above_movement_spike():
    scored = _featured_demo()
    movement = scored[scored["patient_id"] == "A_MOVEMENT_DEMO"]["model2_score"].max()
    deterioration = scored[scored["patient_id"] == "B_DETERIORATION_DEMO"]["model2_score"].max()
    assert deterioration > movement + 3


def test_threshold_alerts_treat_curated_cases_more_similarly_than_rules():
    scored = _featured_demo()
    movement_m1 = scored[scored["patient_id"] == "A_MOVEMENT_DEMO"]["model1_alert"].sum()
    deterioration_m1 = scored[scored["patient_id"] == "B_DETERIORATION_DEMO"]["model1_alert"].sum()
    movement_m2 = scored[scored["patient_id"] == "A_MOVEMENT_DEMO"]["model2_alert"].sum()
    deterioration_m2 = scored[scored["patient_id"] == "B_DETERIORATION_DEMO"]["model2_alert"].sum()
    assert movement_m1 > 0
    assert deterioration_m1 > 0
    assert deterioration_m2 > movement_m2


def test_alert_episode_grouping_combines_nearby_alerts():
    scored = _featured_demo()
    episodes = group_alert_episodes(scored, "model2_alert")
    assert {"patient_id", "start", "end", "event_captured"}.issubset(episodes.columns)


def test_metric_table_counts_alert_episodes_not_positive_timepoints():
    timestamps = pd.date_range("2026-01-01", periods=12, freq="5min")
    df = pd.DataFrame({
        "patient_id": ["p1"] * 12,
        "timestamp": timestamps,
        "deterioration_event": [False] * 12,
        "alert": [False, True, True, True, False, False, True, True, False, False, False, False],
        "score": [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
    })
    mt = metric_table(df, {"demo": ("alert", "score")}).iloc[0]
    assert mt["total_alert_episodes"] == 1
    assert mt["total_alerts"] == 1


def test_tier3_checkins_do_not_count_as_clinician_alert_burden():
    timestamps = pd.date_range("2026-01-01", periods=10, freq="5min")
    df = pd.DataFrame({
        "patient_id": ["p1"] * 10,
        "timestamp": timestamps,
        "deterioration_event": [False] * 10,
        "model2_alert": [False, True, True, False, False, True, True, False, False, False],
        "model2_score": [0, 2.5, 2.4, 0, 0, 4.2, 4.1, 0, 0, 0],
        "model2_tier": [
            "No alert",
            "Tier 3 - patient/caregiver check-in",
            "Tier 3 - patient/caregiver check-in",
            "No alert",
            "No alert",
            "Tier 2 - clinician review",
            "Tier 2 - clinician review",
            "No alert",
            "No alert",
            "No alert",
        ],
    })
    mt = metric_table(df, {"Model 2": ("model2_alert", "model2_score")}).iloc[0]
    assert mt["total_routed_episodes"] == 1
    assert mt["total_alert_episodes"] == 1
    assert mt["tier3_checkin_episodes"] == 1
    assert mt["clinician_alert_episodes_per_patient_day"] == mt["alert_episodes_per_patient_day"]


def test_warning_lead_time_ignores_unrelated_early_alerts():
    timestamps = pd.date_range("2026-01-01", periods=60, freq="5min")
    event = [False] * 60
    for i in range(48, 56):
        event[i] = True
    alert = [False] * 60
    alert[0] = True
    alert[42] = True
    df = pd.DataFrame({
        "patient_id": ["p1"] * 60,
        "timestamp": timestamps,
        "deterioration_event": event,
        "event_type": ["synthetic"] * 60,
        "alert": alert,
        "score": [int(x) for x in alert],
    })
    assert warning_lead_time(df, "alert") == 30.0


def test_model3_threshold_is_selected_from_validation_patients():
    scored = _featured_demo()
    train = scored[scored["patient_id"].str.startswith("A_")].copy()
    model = fit_statistical_model(train)
    threshold, table = choose_model3_threshold(scored, model, min_sensitivity=0.5)
    assert 0.20 <= threshold <= 0.95
    assert "threshold" in table.columns
