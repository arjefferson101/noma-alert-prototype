from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


AGE_COHORTS = [(18, 39, "18-39"), (40, 64, "40-64"), (65, 79, "65-79"), (80, 96, "80+")]
DISCHARGE_CATEGORIES = ["cardiac", "pulmonary", "infection", "post_surgical", "general_medical"]
SCENARIOS = [
    "stable",
    "movement_hr_spike",
    "sensor_artifact",
    "noisy",
    "slow_hr_rise",
    "rr_rise",
    "temp_rise",
    "sustained_abnormality",
    "multivariable_deterioration",
    "low_movement_abnormality",
    "temporary_resolve",
    "missing_data",
    "distinct_baseline",
]


def _age_cohort(age: int) -> str:
    for low, high, label in AGE_COHORTS:
        if low <= age <= high:
            return label
    return "80+"


def _patient_profile(rng: np.random.Generator, cohort: str) -> dict:
    if cohort == "B":
        age = int(np.clip(rng.normal(82, 8), 65, 96))
        baseline_mobility = rng.choice(["low", "medium"], p=[0.72, 0.28])
        hr = rng.normal(78, 9)
        rr = rng.normal(18.5, 2.4)
        temp = rng.normal(98.3, 0.35)
    else:
        age = int(rng.choice(
            [rng.integers(18, 40), rng.integers(40, 65), rng.integers(65, 80), rng.integers(80, 92)],
            p=[0.22, 0.38, 0.28, 0.12],
        ))
        baseline_mobility = rng.choice(["low", "medium", "high"], p=[0.25, 0.50, 0.25])
        hr = rng.normal(72 + (age > 75) * 4, 8)
        rr = rng.normal(16 + (age > 75) * 1.0, 1.8)
        temp = rng.normal(98.2, 0.3)

    mobility_mean = {"low": 0.11, "medium": 0.33, "high": 0.55}[baseline_mobility]
    return {
        "age": age,
        "age_cohort": _age_cohort(age),
        "sex": rng.choice(["female", "male"], p=[0.53, 0.47]),
        "baseline_mobility": baseline_mobility,
        "discharge_category": rng.choice(DISCHARGE_CATEGORIES),
        "base_hr": float(hr),
        "base_rr": float(rr),
        "base_temp": float(temp),
        "base_movement": float(mobility_mean),
    }


def _daily_activity(minutes: np.ndarray, base: float, rng: np.random.Generator) -> np.ndarray:
    hour = (minutes // 60) % 24
    awake = ((hour >= 7) & (hour <= 21)).astype(float)
    circadian = 0.08 * np.sin((hour - 7) / 24 * 2 * np.pi)
    movement = base * (0.55 + 0.75 * awake) + circadian + rng.normal(0, 0.06, len(minutes))
    return np.clip(movement, 0, 1)


def _apply_scenario(df: pd.DataFrame, scenario: str, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    event_start = int(rng.integers(n // 3, n - n // 4))
    duration = int(rng.integers(36, 96))
    idx = np.arange(event_start, min(n, event_start + duration))
    deterioration = scenario in {
        "slow_hr_rise",
        "rr_rise",
        "temp_rise",
        "sustained_abnormality",
        "multivariable_deterioration",
        "low_movement_abnormality",
    }

    if scenario == "movement_hr_spike":
        idx = np.arange(event_start, min(n, event_start + 10))
        pulse = np.sin(np.linspace(0, np.pi, len(idx)))
        df.loc[idx, "heart_rate"] += 34 * pulse
        df.loc[idx, "movement"] = np.clip(df.loc[idx, "movement"] + 0.55 * pulse, 0, 1)
        df.loc[idx, "event_type"] = "movement-associated HR spike"
    elif scenario == "sensor_artifact":
        spikes = rng.choice(n, size=4, replace=False)
        df.loc[spikes, "heart_rate"] += rng.choice([55, -35], size=4)
        df.loc[spikes, "artifact_flag"] = True
        df.loc[spikes, "event_type"] = "isolated sensor artifact"
    elif scenario == "noisy":
        df["heart_rate"] += rng.normal(0, 6, n)
        df["respiratory_rate"] += rng.normal(0, 1.8, n)
        df["event_type"] = "noisy non-deterioration"
    elif scenario == "slow_hr_rise":
        ramp = np.linspace(0, 34, len(idx))
        df.loc[idx, "heart_rate"] += ramp
        df.loc[idx, "deterioration_event"] = True
        df.loc[idx, "event_type"] = "slow HR rise"
    elif scenario == "rr_rise":
        df.loc[idx, "respiratory_rate"] += np.linspace(0, 8, len(idx))
        df.loc[idx, "deterioration_event"] = True
        df.loc[idx, "event_type"] = "gradual respiratory rise"
    elif scenario == "temp_rise":
        df.loc[idx, "temperature"] += np.linspace(0, 1.9, len(idx))
        df.loc[idx, "deterioration_event"] = True
        df.loc[idx, "event_type"] = "gradual temperature rise"
    elif scenario == "sustained_abnormality":
        df.loc[idx, "heart_rate"] += 22
        df.loc[idx, "respiratory_rate"] += 4.5
        df.loc[idx, "deterioration_event"] = True
        df.loc[idx, "event_type"] = "sustained abnormality"
    elif scenario == "multivariable_deterioration":
        ramp = np.linspace(0, 1, len(idx))
        df.loc[idx, "heart_rate"] += 35 * ramp
        df.loc[idx, "respiratory_rate"] += 8 * ramp
        df.loc[idx, "temperature"] += 1.8 * ramp
        df.loc[idx, "movement"] *= 0.35
        df.loc[idx, "deterioration_event"] = True
        df.loc[idx, "event_type"] = "multivariable deterioration"
    elif scenario == "low_movement_abnormality":
        df.loc[idx, "heart_rate"] += 26
        df.loc[idx, "respiratory_rate"] += 5
        df.loc[idx, "movement"] *= 0.25
        df.loc[idx, "deterioration_event"] = True
        df.loc[idx, "event_type"] = "abnormal vitals with low movement"
    elif scenario == "temporary_resolve":
        idx = np.arange(event_start, min(n, event_start + 22))
        pulse = np.sin(np.linspace(0, np.pi, len(idx)))
        df.loc[idx, "heart_rate"] += 24 * pulse
        df.loc[idx, "respiratory_rate"] += 3 * pulse
        df.loc[idx, "event_type"] = "temporary self-resolving abnormality"
    elif scenario == "missing_data":
        short = np.arange(event_start, min(n, event_start + 6))
        long = np.arange(min(n - 1, event_start + 65), min(n, event_start + 112))
        for col in ["heart_rate", "respiratory_rate", "temperature", "movement"]:
            df.loc[short, col] = np.nan
            df.loc[long, col] = np.nan
        df.loc[np.r_[short, long], "event_type"] = "missing sensor data"
    elif scenario == "distinct_baseline":
        df["heart_rate"] += rng.choice([-14, 16])
        df["respiratory_rate"] += rng.choice([-2, 3])
        df["event_type"] = "stable distinct baseline"

    if deterioration:
        df.loc[idx[: max(8, len(idx) // 4)], "event_phase"] = "early"
        df.loc[idx[max(8, len(idx) // 4) :], "event_phase"] = "active"
    return df


def _curated_patient(patient_id: str, pattern: str) -> pd.DataFrame:
    minutes = np.arange(0, 3 * 24 * 60, 5)
    rng = np.random.default_rng(100 if pattern == "movement" else 101)
    timestamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(minutes, unit="m")
    base_hr, base_rr, base_temp = 70.0, 16.0, 98.1
    movement = _daily_activity(minutes, 0.25, rng)
    df = pd.DataFrame({
        "patient_id": patient_id,
        "timestamp": timestamps,
        "age": 58,
        "sex": "female" if pattern == "movement" else "male",
        "age_cohort": "40-64",
        "discharge_category": "general_medical",
        "baseline_mobility": "medium",
        "heart_rate": base_hr + rng.normal(0, 1.6, len(minutes)),
        "respiratory_rate": base_rr + rng.normal(0, 0.5, len(minutes)),
        "temperature": base_temp + rng.normal(0, 0.08, len(minutes)),
        "movement": movement,
        "deterioration_event": False,
        "event_type": "curated movement spike" if pattern == "movement" else "curated deterioration",
        "event_phase": "none",
        "artifact_flag": False,
        "cohort": "A",
    })
    start = 250
    if pattern == "movement":
        idx = np.arange(start, start + 10)
        pulse = np.sin(np.linspace(0, np.pi, len(idx)))
        df.loc[idx, "heart_rate"] = base_hr + 37 * pulse
        df.loc[idx, "movement"] = np.clip(0.82 + 0.12 * pulse, 0, 1)
    else:
        idx = np.arange(start, start + 72)
        ramp = np.linspace(0, 1, len(idx))
        df.loc[idx, "heart_rate"] = base_hr + 35 * ramp + rng.normal(0, 1, len(idx))
        df.loc[idx, "respiratory_rate"] = base_rr + 8 * ramp + rng.normal(0, 0.4, len(idx))
        df.loc[idx, "temperature"] = base_temp + 1.7 * ramp + rng.normal(0, 0.05, len(idx))
        df.loc[idx, "movement"] = 0.06 + rng.normal(0, 0.02, len(idx))
        df.loc[idx, "deterioration_event"] = True
        df.loc[idx[:18], "event_phase"] = "early"
        df.loc[idx[18:], "event_phase"] = "active"
    return df


def generate_synthetic_data(n_patients_a: int = 120, n_patients_b: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [_curated_patient("A_MOVEMENT_DEMO", "movement"), _curated_patient("B_DETERIORATION_DEMO", "deterioration")]
    minutes = np.arange(0, 4 * 24 * 60, 5)
    timestamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(minutes, unit="m")

    for cohort, count in [("A", n_patients_a), ("B", n_patients_b)]:
        scenario_probs = np.array([0.30, 0.10, 0.08, 0.08, 0.055, 0.055, 0.055, 0.07, 0.06, 0.055, 0.06, 0.04, 0.05])
        scenario_probs = scenario_probs / scenario_probs.sum()
        if cohort == "B":
            scenario_probs = np.array([0.22, 0.05, 0.07, 0.10, 0.06, 0.075, 0.075, 0.09, 0.075, 0.075, 0.045, 0.07, 0.075])
            scenario_probs = scenario_probs / scenario_probs.sum()
        for i in range(count):
            profile = _patient_profile(rng, cohort)
            scenario = rng.choice(SCENARIOS, p=scenario_probs)
            pid = f"{cohort}_{i:04d}"
            movement = _daily_activity(minutes, profile["base_movement"], rng)
            df = pd.DataFrame({
                "patient_id": pid,
                "timestamp": timestamps,
                "age": profile["age"],
                "sex": profile["sex"],
                "age_cohort": profile["age_cohort"],
                "discharge_category": profile["discharge_category"],
                "baseline_mobility": profile["baseline_mobility"],
                "heart_rate": profile["base_hr"] + 8 * movement + rng.normal(0, 2.5, len(minutes)),
                "respiratory_rate": profile["base_rr"] + 1.2 * movement + rng.normal(0, 0.8, len(minutes)),
                "temperature": profile["base_temp"] + rng.normal(0, 0.12, len(minutes)),
                "movement": movement,
                "deterioration_event": False,
                "event_type": "stable",
                "event_phase": "none",
                "artifact_flag": False,
                "cohort": cohort,
            })
            frames.append(_apply_scenario(df, scenario, rng))

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


def save_synthetic_data(path: str | Path = "data/synthetic_patient_data.csv", **kwargs) -> pd.DataFrame:
    df = generate_synthetic_data(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    save_synthetic_data()
