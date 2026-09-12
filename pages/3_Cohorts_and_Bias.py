from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.bias_analysis import age_cohort_bootstrap_ci, subgroup_metrics
from src.pipeline import build_scored_dataset
from src.preprocessing import missingness_by_group


st.set_page_config(page_title="Cohorts & Bias", layout="wide")


@st.cache_data(show_spinner="Running subgroup analysis...")
def data():
    scored, _, _ = build_scored_dataset()
    return scored[(scored["cohort"] == "A") & (scored["split"] == "test")]


df = data()
st.title("Does the Model Work Equally Well for Everyone?")
st.caption("Synthetic subgroup analysis. Small subgroups are labeled uncertain and should not be overinterpreted.")

group_col = st.selectbox("Subgroup", ["age_cohort", "sex", "baseline_mobility", "discharge_category"])
sg = subgroup_metrics(df, group_col)
metric = st.selectbox("Metric", ["sensitivity", "precision_ppv", "false_positive_rate", "clinician_alert_episodes_per_patient_day", "false_clinician_alert_episodes_per_patient_day", "tier3_checkin_episodes_per_patient_day", "routed_episodes_per_patient_day"])

fig = px.bar(sg, x=group_col, y=metric, color="system", barmode="group", hover_data=["patients", "events", "uncertain_small_group"])
fig.update_layout(height=500, margin=dict(l=10, r=10, t=25, b=10))
st.plotly_chart(fig, use_container_width=True)

if group_col == "age_cohort":
    st.subheader("Age-Cohort Bootstrap 95% Confidence Intervals")
    ci = age_cohort_bootstrap_ci(df)
    if ci.empty:
        st.write("No age cohort has enough synthetic patients and deterioration episodes for a patient-level bootstrap interval.")
    else:
        ci_metric = ci[ci["metric"] == metric]
        fig_ci = px.scatter(ci_metric, x="age_cohort", y="ci_low", color="system", error_y=ci_metric["ci_high"] - ci_metric["ci_low"], labels={"ci_low": "95% CI low"})
        fig_ci.update_layout(height=420, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig_ci, use_container_width=True)
        st.dataframe(ci.round(3), use_container_width=True, hide_index=True)

st.subheader("Missingness by Age Cohort")
st.dataframe(missingness_by_group(df, "age_cohort").round(3), use_container_width=True, hide_index=True)
st.subheader("Subgroup Metrics")
st.dataframe(sg.round(3), use_container_width=True, hide_index=True)
