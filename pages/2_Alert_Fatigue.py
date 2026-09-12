from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.bias_analysis import SPECS
from src.evaluation import metric_table, patient_bootstrap_ci, warning_lead_time
from src.pipeline import build_scored_dataset


st.set_page_config(page_title="Alert Fatigue", layout="wide")


@st.cache_data(show_spinner="Calculating alert metrics...")
def results():
    scored, _, _ = build_scored_dataset()
    test = scored[(scored["cohort"] == "A") & (scored["split"] == "test")]
    mt = metric_table(test, SPECS)
    ci = patient_bootstrap_ci(test, SPECS, n_boot=30)
    for name, (alert_col, _) in SPECS.items():
        mt.loc[mt["system"] == name, "warning_lead_time_min"] = warning_lead_time(test, alert_col)
    return mt, ci


mt, ci = results()
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
fig = px.bar(plot_df, x="system", y="value", color="system", facet_col="metric", facet_col_wrap=2, text_auto=".2f")
fig.update_layout(height=560, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Patient-Level Bootstrap Intervals")
ci_plot = ci[ci["metric"].isin(["sensitivity", "precision_ppv", "clinician_alert_episodes_per_patient_day", "false_clinician_alert_episodes_per_patient_day", "tier3_checkin_episodes_per_patient_day", "routed_episodes_per_patient_day"])]
fig2 = px.scatter(ci_plot, x="system", y="ci_low", color="metric", error_y=ci_plot["ci_high"] - ci_plot["ci_low"], labels={"ci_low": "95% CI low"})
fig2.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.info("Warning lead time is matched to a specific synthetic deterioration episode. Event onset is the first timestamp in a contiguous deterioration episode; only alerts in the two-hour pre-onset window or 30-minute post-onset grace period are credited.")
st.write("Overall routed episodes remain in the table for product planning, but only Tier 1/Tier 2 episodes count as clinician-facing alert burden for Models 2 and 3.")
st.dataframe(mt.round(3), use_container_width=True, hide_index=True)
