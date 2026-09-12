from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.pipeline import CALIBRATION_PATH, COHORT_SHIFT_PATH, load_table


st.set_page_config(page_title="Generalization", layout="wide")


table = load_table(COHORT_SHIFT_PATH)
st.title("Does It Generalize?")
st.caption("Cohort B is a synthetic nursing-home / long-term-care population with intentional domain shift.")

metric = st.selectbox("Metric", ["sensitivity", "precision_ppv", "false_positive_rate", "clinician_alert_episodes_per_patient_day", "false_clinician_alert_episodes_per_patient_day", "tier3_checkin_episodes_per_patient_day", "routed_episodes_per_patient_day", "pr_auc"])
fig = go.Figure()
for population in table["evaluation_population"].drop_duplicates():
    part = table[table["evaluation_population"] == population]
    fig.add_trace(go.Bar(x=part["system"], y=part[metric], name=population))
fig.update_layout(barmode="group")
fig.update_layout(height=500, margin=dict(l=10, r=10, t=25, b=10))
st.plotly_chart(fig, use_container_width=True)

st.info("More data is only valuable if it represents the people Noma intends to monitor. Free nursing-home trials could be useful for longitudinal wearability, sensor artifact, missingness, and workflow learning, but this synthetic demo separates that from any assumption that nursing-home data generalize to post-discharge patients.")

curve = load_table(CALIBRATION_PATH)
brier = float(curve["brier_score"].iloc[0])
st.subheader("Model 3 Calibration")
st.write(f"Synthetic Brier score: `{brier:.3f}`. Calibration asks whether groups scored near 0.70 risk experience the modeled event about 70% of the time.")
curve_plot = curve.drop(columns=["brier_score"])
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=curve_plot["mean_predicted_risk"], y=curve_plot["observed_event_rate"], mode="lines+markers", name="calibration"))
fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dash"))
fig2.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.dataframe(table.round(3), use_container_width=True, hide_index=True)
