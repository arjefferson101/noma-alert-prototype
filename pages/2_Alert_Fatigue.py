from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.pipeline import ALERT_CI_PATH, ALERT_METRICS_PATH, load_table


st.set_page_config(page_title="Alert Fatigue", layout="wide")


mt = load_table(ALERT_METRICS_PATH)
ci = load_table(ALERT_CI_PATH)
st.title("Alert Fatigue")
st.caption("Held-out Cohort A patients only. Primary burden is clinician-facing Tier 1/Tier 2 alert episodes. Tier 3 patient/caregiver check-ins are reported separately. Synthetic and illustrative.")

base_alerts = float(mt.loc[mt["system"].str.contains("universal"), "clinician_alert_episodes_per_patient_day"].iloc[0])
cols = st.columns(3)
for i, row in mt.iterrows():
    delta = (row["clinician_alert_episodes_per_patient_day"] - base_alerts) / base_alerts if base_alerts else 0
    cols[i].metric(row["system"].split(":")[0], f"{row['clinician_alert_episodes_per_patient_day']:.2f} clinician episodes/patient/day", f"{delta:.0%} vs Model 1")

plot_df = mt.melt(
    id_vars="system",
    value_vars=[
        "sensitivity",
        "precision_ppv",
        "clinician_alert_episodes_per_patient_day",
        "false_clinician_alert_episodes_per_patient_day",
        "tier3_checkin_episodes_per_patient_day",
        "routed_episodes_per_patient_day",
    ],
    var_name="metric",
    value_name="value",
)
metrics = plot_df["metric"].drop_duplicates().to_list()
fig = make_subplots(rows=3, cols=2, subplot_titles=metrics)
for idx, metric_name in enumerate(metrics):
    part = plot_df[plot_df["metric"] == metric_name]
    row = idx // 2 + 1
    col = idx % 2 + 1
    fig.add_trace(go.Bar(x=part["system"], y=part["value"], text=part["value"].round(2), textposition="auto", marker_color="#2962ff"), row=row, col=col)
fig.update_layout(height=560, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Patient-Level Bootstrap Intervals")
ci_plot = ci[ci["metric"].isin(["sensitivity", "precision_ppv", "clinician_alert_episodes_per_patient_day", "false_clinician_alert_episodes_per_patient_day", "tier3_checkin_episodes_per_patient_day", "routed_episodes_per_patient_day"])]
fig2 = go.Figure()
for metric_name in ci_plot["metric"].drop_duplicates():
    part = ci_plot[ci_plot["metric"] == metric_name]
    fig2.add_trace(go.Scatter(
        x=part["system"],
        y=part["ci_low"],
        error_y=dict(type="data", array=part["ci_high"] - part["ci_low"]),
        mode="markers",
        name=metric_name,
    ))
fig2.update_yaxes(title_text="95% CI low")
fig2.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.info("Warning lead time is matched to a specific synthetic deterioration episode. Event onset is the first timestamp in a contiguous deterioration episode; only alerts in the two-hour pre-onset window or 30-minute post-onset grace period are credited.")
st.write("Overall routed episodes remain in the table for product planning, but only Tier 1/Tier 2 episodes count as clinician-facing alert burden for Models 2 and 3.")
st.dataframe(mt.round(3), use_container_width=True, hide_index=True)
