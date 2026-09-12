from __future__ import annotations

import numpy as np
import pandas as pd


VITALS = ["heart_rate", "respiratory_rate", "temperature", "movement"]


def preprocess_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    out["raw_missing"] = out[VITALS].isna().any(axis=1)
    out["implausible_flag"] = (
        (out["heart_rate"].lt(35) | out["heart_rate"].gt(190))
        | (out["respiratory_rate"].lt(6) | out["respiratory_rate"].gt(45))
        | (out["temperature"].lt(94) | out["temperature"].gt(104.5))
        | (out["movement"].lt(0) | out["movement"].gt(1.05))
    )
    for col in VITALS:
        out.loc[out["implausible_flag"], col] = np.nan

    pieces = []
    for _, g in out.groupby("patient_id", sort=False):
        g = g.copy()
        miss = g[VITALS].isna().any(axis=1)
        block = miss.ne(miss.shift(fill_value=False)).cumsum()
        g["missing_run"] = miss.groupby(block).cumcount().add(1).where(miss, 0)
        g["insufficient_data"] = g["missing_run"] >= 7
        for col in VITALS:
            g[col] = g[col].ffill(limit=3)
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    out["artifact_processed"] = out["artifact_flag"].astype(bool) | out["implausible_flag"].astype(bool)
    return out


def missingness_by_group(df: pd.DataFrame, group_col: str = "age_cohort") -> pd.DataFrame:
    return (
        df.groupby(group_col)
        .agg(patients=("patient_id", "nunique"), observations=("patient_id", "size"), missing_rate=("raw_missing", "mean"), insufficient_rate=("insufficient_data", "mean"))
        .reset_index()
    )
