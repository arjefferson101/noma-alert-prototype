from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.pipeline import LEARNING_CURVE_PATH, load_table


st.set_page_config(page_title="How Much Data", layout="wide")


lc = load_table(LEARNING_CURVE_PATH)
st.title("How Much Data Do We Need?")
st.caption("Training size increases by patients, not rows, to avoid repeated-measures leakage.")

metric = st.selectbox("Metric", ["sensitivity", "precision_ppv", "pr_auc", "false_positive_rate", "clinician_alert_episodes_per_patient_day"])
summary = lc.groupby(["training_fraction", "training_patients"])[metric].agg(["mean", "std"]).reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=summary["training_patients"],
    y=summary["mean"],
    error_y=dict(type="data", array=summary["std"].fillna(0)),
    mode="lines+markers",
    name=metric,
))
fig.update_yaxes(title_text=metric)
fig.update_xaxes(title_text="training_patients")
fig.update_layout(height=500, margin=dict(l=10, r=10, t=25, b=10))
st.plotly_chart(fig, use_container_width=True)

last_gain = summary["mean"].iloc[-1] - summary["mean"].iloc[-2] if len(summary) > 1 else 0
st.info(
    "Business read: the simulated curve is still improving materially."
    if last_gain > 0.03
    else "Business read: the simulated curve is beginning to plateau; more representative patients may help less than better validation or feature design."
)
st.write("Model 3's classification threshold is selected on validation patients only, then frozen before test-set evaluation.")
st.dataframe(lc.round(3), use_container_width=True, hide_index=True)
