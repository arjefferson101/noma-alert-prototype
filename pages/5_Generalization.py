from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.cohort_shift import cohort_shift_table
from src.pipeline import build_scored_dataset
from src.statistical_model import calibration_summary


st.set_page_config(page_title="Generalization", layout="wide")


@st.cache_data(show_spinner="Comparing cohorts...")
def data():
    scored, _, _ = build_scored_dataset()
    return scored, cohort_shift_table(scored)


scored, table = data()
st.title("Does It Generalize?")
st.caption("Cohort B is a synthetic nursing-home / long-term-care population with intentional domain shift.")

metric = st.selectbox("Metric", ["sensitivity", "precision_ppv", "false_positive_rate", "clinician_alert_episodes_per_patient_day", "false_clinician_alert_episodes_per_patient_day", "tier3_checkin_episodes_per_patient_day", "routed_episodes_per_patient_day", "pr_auc"])
fig = px.bar(table, x="system", y=metric, color="evaluation_population", barmode="group")
fig.update_layout(height=500, margin=dict(l=10, r=10, t=25, b=10))
st.plotly_chart(fig, use_container_width=True)

st.info("More data is only valuable if it represents the people Noma intends to monitor. Free nursing-home trials could be useful for longitudinal wearability, sensor artifact, missingness, and workflow learning, but this synthetic demo separates that from any assumption that nursing-home data generalize to post-discharge patients.")

test_a = scored[(scored["cohort"] == "A") & (scored["split"] == "test")]
curve, brier = calibration_summary(test_a)
st.subheader("Model 3 Calibration")
st.write(f"Synthetic Brier score: `{brier:.3f}`. Calibration asks whether groups scored near 0.70 risk experience the modeled event about 70% of the time.")
fig2 = px.line(curve, x="mean_predicted_risk", y="observed_event_rate", markers=True)
fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dash"))
fig2.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig2, use_container_width=True)
st.dataframe(table.round(3), use_container_width=True, hide_index=True)
