from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.pipeline import AGE_CI_PATH, MISSINGNESS_PATH, SUBGROUP_PATHS, load_table


st.set_page_config(page_title="Cohorts & Bias", layout="wide")


st.title("Does the Model Work Equally Well for Everyone?")
st.caption("Synthetic subgroup analysis. Small subgroups are labeled uncertain and should not be overinterpreted.")

group_col = st.selectbox("Subgroup", ["age_cohort", "sex", "baseline_mobility", "discharge_category"])
sg = load_table(SUBGROUP_PATHS[group_col])
metric = st.selectbox("Metric", ["sensitivity", "precision_ppv", "false_positive_rate", "clinician_alert_episodes_per_patient_day", "false_clinician_alert_episodes_per_patient_day", "tier3_checkin_episodes_per_patient_day", "routed_episodes_per_patient_day"])

fig = go.Figure()
for system in sg["system"].drop_duplicates():
    part = sg[sg["system"] == system]
    fig.add_trace(go.Bar(
        x=part[group_col],
        y=part[metric],
        name=system,
        customdata=part[["patients", "events", "uncertain_small_group"]],
        hovertemplate=f"{group_col}: %{{x}}<br>{metric}: %{{y:.3f}}<br>patients: %{{customdata[0]}}<br>events: %{{customdata[1]}}<br>uncertain: %{{customdata[2]}}<extra></extra>",
    ))
fig.update_layout(barmode="group")
fig.update_layout(height=500, margin=dict(l=10, r=10, t=25, b=10))
st.plotly_chart(fig, use_container_width=True)

if group_col == "age_cohort":
    st.subheader("Age-Cohort Bootstrap 95% Confidence Intervals")
    ci = load_table(AGE_CI_PATH)
    if ci.empty:
        st.write("No age cohort has enough synthetic patients and deterioration episodes for a patient-level bootstrap interval.")
    else:
        ci_metric = ci[ci["metric"] == metric]
        fig_ci = go.Figure()
        for system in ci_metric["system"].drop_duplicates():
            part = ci_metric[ci_metric["system"] == system]
            fig_ci.add_trace(go.Scatter(
                x=part["age_cohort"],
                y=part["ci_low"],
                error_y=dict(type="data", array=part["ci_high"] - part["ci_low"]),
                mode="markers",
                name=system,
            ))
        fig_ci.update_yaxes(title_text="95% CI low")
        fig_ci.update_layout(height=420, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig_ci, use_container_width=True)
        st.dataframe(ci.round(3), use_container_width=True, hide_index=True)

st.subheader("Missingness by Age Cohort")
st.dataframe(load_table(MISSINGNESS_PATH).round(3), use_container_width=True, hide_index=True)
st.subheader("Subgroup Metrics")
st.dataframe(sg.round(3), use_container_width=True, hide_index=True)
