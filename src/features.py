from __future__ import annotations

import numpy as np
import pandas as pd


SIGNALS = ["heart_rate", "respiratory_rate", "temperature", "movement"]


def _mad(x: pd.Series) -> float:
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    return max(1.4826 * mad, 0.1)


def add_patient_baselines(df: pd.DataFrame, baseline_hours: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    rows = []
    for pid, g in out.groupby("patient_id", sort=False):
        cutoff = g["timestamp"].min() + pd.Timedelta(hours=baseline_hours)
        base = g[(g["timestamp"] <= cutoff) & (~g["artifact_processed"]) & (~g["insufficient_data"])]
        if base.empty:
            base = g.head(max(12, min(len(g), 144)))
        row = {"patient_id": pid}
        for col in SIGNALS:
            row[f"baseline_{col}"] = float(np.nanmedian(base[col]))
            row[f"mad_{col}"] = _mad(base[col])
        rows.append(row)
    baselines = pd.DataFrame(rows)
    out = out.merge(baselines, on="patient_id", how="left")
    for col in SIGNALS:
        out[f"{col}_delta"] = out[col] - out[f"baseline_{col}"]
        out[f"{col}_z"] = out[f"{col}_delta"] / out[f"mad_{col}"].clip(lower=0.1)
    return out, baselines


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["patient_id", "timestamp"]).copy()
    pieces = []
    for _, g in out.groupby("patient_id", sort=False):
        g = g.copy()
        for col in ["heart_rate", "respiratory_rate", "temperature"]:
            g[f"{col}_roll15"] = g[col].rolling(3, min_periods=1).mean()
            g[f"{col}_roll30"] = g[col].rolling(6, min_periods=2).mean()
            g[f"{col}_roll60"] = g[col].rolling(12, min_periods=3).mean()
            g[f"{col}_sd30"] = g[col].rolling(6, min_periods=2).std().fillna(0)
            g[f"{col}_slope30"] = (g[col] - g[col].shift(6)) / 30
            g[f"{col}_change60"] = g[col] - g[col].shift(12)
        abnormal = (
            (g["heart_rate_delta"] > 18)
            | (g["respiratory_rate_delta"] > 3.5)
            | (g["temperature_delta"] > 0.9)
        )
        blocks = abnormal.ne(abnormal.shift(fill_value=False)).cumsum()
        g["abnormal_duration_min"] = abnormal.groupby(blocks).cumcount().add(1).mul(5).where(abnormal, 0)
        g["concordant_signals"] = (
            (g["heart_rate_delta"] > 12).astype(int)
            + (g["respiratory_rate_delta"] > 2.5).astype(int)
            + (g["temperature_delta"] > 0.6).astype(int)
        )
        g["low_movement_context"] = g["movement"] < (g["baseline_movement"] + 0.05).clip(upper=0.3)
        g["movement_associated_hr"] = (g["movement_delta"] > 0.25) & (g["heart_rate_delta"] > 16)
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)
