from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PRE_EVENT_WINDOW_MINUTES = 120
POST_ONSET_GRACE_MINUTES = 30


def _empty_alert_episode_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["patient_id", "alert_col", "episode_id", "start", "end", "event_captured"])


def _group_episodes_from_mask(df: pd.DataFrame, mask: pd.Series, label: str, gap_minutes: int = 30) -> pd.DataFrame:
    rows = []
    work = df.sort_values(["patient_id", "timestamp"]).copy()
    work["_episode_mask"] = mask.reindex(work.index).fillna(False).astype(bool)
    for pid, g in work.groupby("patient_id"):
        active = g[g["_episode_mask"]]
        if active.empty:
            continue
        gap = active["timestamp"].diff().dt.total_seconds().div(60).fillna(9999)
        episode = (gap > gap_minutes).cumsum()
        active = active.assign(_episode=episode.to_numpy())
        for eid, e in active.groupby("_episode"):
            rows.append({
                "patient_id": pid,
                "alert_col": label,
                "episode_id": f"{pid}-{label}-{eid}",
                "start": e["timestamp"].min(),
                "end": e["timestamp"].max(),
                "event_captured": bool(e["deterioration_event"].any()),
            })
    return pd.DataFrame(rows) if rows else _empty_alert_episode_frame()


def group_alert_episodes(df: pd.DataFrame, alert_col: str, gap_minutes: int = 30) -> pd.DataFrame:
    return _group_episodes_from_mask(df, df[alert_col].fillna(False), alert_col, gap_minutes)


def routed_episode_masks(df: pd.DataFrame, alert_col: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    prefix = alert_col.removesuffix("_alert")
    tier_col = f"{prefix}_tier"
    alert_mask = df[alert_col].fillna(False).astype(bool)
    if tier_col not in df:
        return alert_mask, alert_mask, pd.Series(False, index=df.index)

    tiers = df[tier_col].fillna("")
    routed = tiers.str.startswith(("Tier 1", "Tier 2", "Tier 3"))
    clinician = tiers.str.startswith(("Tier 1", "Tier 2"))
    tier3 = tiers.str.startswith("Tier 3")
    return routed, clinician, tier3


def group_routed_episodes(df: pd.DataFrame, alert_col: str, route: str = "clinician", gap_minutes: int = 30) -> pd.DataFrame:
    routed, clinician, tier3 = routed_episode_masks(df, alert_col)
    masks = {"all": routed, "clinician": clinician, "tier3": tier3}
    if route not in masks:
        raise ValueError(f"Unknown route: {route}")
    return _group_episodes_from_mask(df, masks[route], f"{alert_col}_{route}", gap_minutes)


def group_deterioration_episodes(df: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    rows = []
    for pid, g in df.sort_values(["patient_id", "timestamp"]).groupby("patient_id"):
        active = g[g["deterioration_event"].fillna(False)]
        if active.empty:
            continue
        gap = active["timestamp"].diff().dt.total_seconds().div(60).fillna(9999)
        episode = (gap > gap_minutes).cumsum()
        active = active.assign(_episode=episode.to_numpy())
        for eid, e in active.groupby("_episode"):
            rows.append({
                "patient_id": pid,
                "event_episode_id": f"{pid}-event-{eid}",
                "event_onset": e["timestamp"].min(),
                "event_end": e["timestamp"].max(),
                "event_type": e["event_type"].iloc[0] if "event_type" in e else "synthetic deterioration",
            })
    return pd.DataFrame(rows)


def match_alerts_to_events(
    df: pd.DataFrame,
    alert_col: str,
    pre_event_window_minutes: int = PRE_EVENT_WINDOW_MINUTES,
    post_onset_grace_minutes: int = POST_ONSET_GRACE_MINUTES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return match_alerts_to_events_for_episodes(
        df,
        group_alert_episodes(df, alert_col),
        pre_event_window_minutes=pre_event_window_minutes,
        post_onset_grace_minutes=post_onset_grace_minutes,
    )


def match_alerts_to_events_for_episodes(
    df: pd.DataFrame,
    alerts: pd.DataFrame,
    pre_event_window_minutes: int = PRE_EVENT_WINDOW_MINUTES,
    post_onset_grace_minutes: int = POST_ONSET_GRACE_MINUTES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = group_deterioration_episodes(df)
    if alerts.empty:
        alerts = _empty_alert_episode_frame()
    if events.empty:
        events = pd.DataFrame(columns=["patient_id", "event_episode_id", "event_onset", "event_end", "event_type"])
    alerts = alerts.copy()
    events = events.copy()
    alerts["matched_event_episode_id"] = pd.NA
    alerts["lead_time_min"] = np.nan
    events["captured"] = False
    events["first_alert_start"] = pd.NaT

    for event_idx, event in events.iterrows():
        window_start = event["event_onset"] - pd.Timedelta(minutes=pre_event_window_minutes)
        window_end = event["event_onset"] + pd.Timedelta(minutes=post_onset_grace_minutes)
        candidates = alerts[
            (alerts["patient_id"] == event["patient_id"])
            & (alerts["start"] <= window_end)
            & (alerts["end"] >= window_start)
        ].sort_values("start")
        if candidates.empty:
            continue
        first = candidates.iloc[0]
        events.loc[event_idx, "captured"] = True
        events.loc[event_idx, "first_alert_start"] = first["start"]
        event_mask = alerts["episode_id"].isin(candidates["episode_id"])
        unmatched_mask = alerts["matched_event_episode_id"].isna()
        alerts.loc[event_mask & unmatched_mask, "matched_event_episode_id"] = event["event_episode_id"]
        alerts.loc[event_mask & unmatched_mask, "lead_time_min"] = (
            event["event_onset"] - alerts.loc[event_mask & unmatched_mask, "start"]
        ).dt.total_seconds().div(60).clip(lower=0, upper=pre_event_window_minutes)
    alerts["event_captured"] = alerts["matched_event_episode_id"].notna()
    return alerts, events


def patient_split(df: pd.DataFrame, seed: int = 7, train_frac: float = 0.6, val_frac: float = 0.2) -> dict:
    rng = np.random.default_rng(seed)
    labels = df.groupby("patient_id")["deterioration_event"].max().reset_index()
    train, val, test = [], [], []
    for _, part in labels.groupby("deterioration_event"):
        ids = part["patient_id"].to_numpy()
        rng.shuffle(ids)
        n_train = int(len(ids) * train_frac)
        n_val = int(len(ids) * val_frac)
        train.extend(ids[:n_train])
        val.extend(ids[n_train:n_train + n_val])
        test.extend(ids[n_train + n_val:])
    return {"train": set(train), "validation": set(val), "test": set(test)}


def metric_table(df: pd.DataFrame, specs: dict[str, tuple[str, str]]) -> pd.DataFrame:
    y = df["deterioration_event"].astype(int)
    days = (df.groupby("patient_id")["timestamp"].agg(lambda s: (s.max() - s.min()).total_seconds() / 86400).sum()) or 1
    rows = []
    for name, (alert_col, score_col) in specs.items():
        pred = df[alert_col].fillna(False).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        score = df[score_col].fillna(0) if score_col in df else pred
        routed_episodes, _ = match_alerts_to_events_for_episodes(df, group_routed_episodes(df, alert_col, "all"))
        clinician_episodes, event_episodes = match_alerts_to_events_for_episodes(df, group_routed_episodes(df, alert_col, "clinician"))
        tier3_episodes, _ = match_alerts_to_events_for_episodes(df, group_routed_episodes(df, alert_col, "tier3"))
        routed_episode_count = len(routed_episodes)
        clinician_episode_count = len(clinician_episodes)
        tier3_episode_count = len(tier3_episodes)
        false_clinician_episode_count = int((~clinician_episodes["event_captured"]).sum()) if clinician_episode_count else 0
        false_routed_episode_count = int((~routed_episodes["event_captured"]).sum()) if routed_episode_count else 0
        false_tier3_episode_count = int((~tier3_episodes["event_captured"]).sum()) if tier3_episode_count else 0
        matched_clinician_episodes = int(clinician_episodes["event_captured"].sum()) if clinician_episode_count else 0
        captured_events = int(event_episodes["captured"].sum()) if not event_episodes.empty else 0
        total_events = len(event_episodes)
        episode_sensitivity = captured_events / total_events if total_events else 0
        episode_ppv = matched_clinician_episodes / clinician_episode_count if clinician_episode_count else 0
        rows.append({
            "system": name,
            "sensitivity": episode_sensitivity,
            "observation_sensitivity": recall_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else 0,
            "precision_ppv": episode_ppv,
            "observation_precision_ppv": precision_score(y, pred, zero_division=0),
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0,
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": _safe_auc(y, score, "roc"),
            "pr_auc": _safe_auc(y, score, "pr"),
            "clinician_alert_episodes_per_patient_day": clinician_episode_count / days,
            "false_clinician_alert_episodes_per_patient_day": false_clinician_episode_count / days,
            "tier3_checkin_episodes_per_patient_day": tier3_episode_count / days,
            "false_tier3_checkin_episodes_per_patient_day": false_tier3_episode_count / days,
            "routed_episodes_per_patient_day": routed_episode_count / days,
            "false_routed_episodes_per_patient_day": false_routed_episode_count / days,
            "alert_episodes_per_patient_day": clinician_episode_count / days,
            "false_alert_episodes_per_patient_day": false_clinician_episode_count / days,
            "alerts_per_patient_day": clinician_episode_count / days,
            "false_alerts_per_patient_day": false_clinician_episode_count / days,
            "total_alerts": int(clinician_episode_count),
            "total_alert_episodes": int(clinician_episode_count),
            "false_alert_episodes": false_clinician_episode_count,
            "total_routed_episodes": int(routed_episode_count),
            "tier3_checkin_episodes": int(tier3_episode_count),
            "event_episodes": int(total_events),
            "captured_event_episodes": captured_events,
        })
    return pd.DataFrame(rows)


def _safe_auc(y, score, kind: str) -> float:
    if len(pd.Series(y).dropna().unique()) < 2:
        return float("nan")
    try:
        return roc_auc_score(y, score) if kind == "roc" else average_precision_score(y, score)
    except ValueError:
        return float("nan")


def patient_bootstrap_ci(df: pd.DataFrame, specs: dict[str, tuple[str, str]], n_boot: int = 120, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = df["patient_id"].drop_duplicates().to_numpy()
    rows = []
    metrics = [
        "sensitivity",
        "precision_ppv",
        "false_positive_rate",
        "clinician_alert_episodes_per_patient_day",
        "false_clinician_alert_episodes_per_patient_day",
        "tier3_checkin_episodes_per_patient_day",
        "routed_episodes_per_patient_day",
    ]
    for _ in range(n_boot):
        sample_ids = rng.choice(ids, size=len(ids), replace=True)
        sampled_frames = []
        for draw_idx, pid in enumerate(sample_ids):
            piece = df[df["patient_id"] == pid].copy()
            piece["patient_id"] = piece["patient_id"].astype(str) + f"__boot{draw_idx}"
            sampled_frames.append(piece)
        sample = pd.concat(sampled_frames, ignore_index=True)
        mt = metric_table(sample, specs)
        for _, row in mt.iterrows():
            for metric in metrics:
                rows.append({"system": row["system"], "metric": metric, "value": row[metric]})
    boot = pd.DataFrame(rows)
    return boot.groupby(["system", "metric"])["value"].quantile([0.025, 0.975]).unstack().reset_index().rename(columns={0.025: "ci_low", 0.975: "ci_high"})


def warning_lead_time(df: pd.DataFrame, alert_col: str) -> float:
    _, events = match_alerts_to_events_for_episodes(df, group_routed_episodes(df, alert_col, "clinician"))
    leads = (
        (events.loc[events["captured"], "event_onset"] - events.loc[events["captured"], "first_alert_start"])
        .dt.total_seconds()
        .div(60)
        .clip(lower=0, upper=PRE_EVENT_WINDOW_MINUTES)
        .dropna()
        .to_list()
    )
    return float(np.nanmean(leads)) if leads else float("nan")
