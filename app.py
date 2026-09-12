from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="Noma Alert Triage Prototype", page_icon="N", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {background: #f7f9fb; border: 1px solid #e4e9ef; border-radius: 8px; padding: 14px;}
    .notice {background:#fff7e6; border:1px solid #f1d39a; border-radius:8px; padding:14px 16px; color:#3c2a05;}
    .card {background:#ffffff; border:1px solid #e3e8ee; border-radius:8px; padding:16px; margin:8px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Noma Alert Triage Prototype")
st.caption("Synthetic and illustrative startup prototype. Not clinically validated. No real patient data.")

st.markdown(
    """
    <div class="notice">
    <b>Synthetic demonstration only.</b> All patients, deterioration events, thresholds, risk scores,
    model outputs, confidence intervals, and results in this app are simulated for product exploration.
    This is not a clinical decision-support system and does not make medical recommendations.
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Product Question")
st.write(
    "Can a small startup reduce alert fatigue by replacing isolated population thresholds with "
    "patient-specific temporal context: baseline deviation, persistence, trend, multivariable agreement, "
    "movement context, and alert history?"
)

c1, c2, c3 = st.columns(3)
c1.metric("Model 1", "Universal thresholds", "hypothetical")
c2.metric("Model 2", "Personalized rules", "interpretable")
c3.metric("Model 3", "Logistic regression", "simple ML")

st.subheader("Use the pages in the sidebar")
st.write(
    "Start with **Patient Monitor** and select `A_MOVEMENT_DEMO` and `B_DETERIORATION_DEMO`. "
    "Those two synthetic cases show the core thesis: a brief HR spike during movement should not look "
    "the same as a sustained multivariable deterioration pattern at low movement."
)
