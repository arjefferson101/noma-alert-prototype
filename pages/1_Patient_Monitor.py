from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.pipeline import build_scored_dataset


st.set_page_config(page_title="Patient Monitor", layout="wide")


@st.cache_data(show_spinner="Generating synthetic patients and scores...")
def data():
    scored, _, _ = build_scored_dataset()
    return scored


df = data()
st.title("Patient Monitor")
st.caption("Synthetic and illustrative. Thresholds, risk tiers, and results are not clinically validated.")

demo_first = ["A_MOVEMENT_DEMO", "B_DETERIORATION_DEMO"]
patients = demo_first + [p for p in sorted(df["patient_id"].unique()) if p not in demo_first]
pid = st.selectbox("Synthetic patient", patients, index=0)
model = st.radio("Alert overlay", ["Model 1", "Model 2", "Model 3"], index=1, horizontal=True)
g = df[df["patient_id"] == pid].sort_values("timestamp")

latest = g.iloc[-1]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Current tier", latest[f"{ {'Model 1':'model1','Model 2':'model2','Model 3':'model3'}[model] }_tier"])
m2.metric("Age cohort", latest["age_cohort"])
m3.metric("Baseline mobility", latest["baseline_mobility"])
m4.metric("Known synthetic event", "yes" if g["deterioration_event"].any() else "no")
m5.metric("Model 3 threshold", f"{latest['model3_threshold']:.2f}")

prefix = {"Model 1": "model1", "Model 2": "model2", "Model 3": "model3"}[model]
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=("Heart rate", "Respiratory rate", "Temperature", "Movement"),
)
series = [
    ("heart_rate", "Heart rate", "#2962ff", 1),
    ("respiratory_rate", "Respiratory rate", "#00897b", 2),
    ("temperature", "Temperature", "#c62828", 3),
    ("movement", "Movement", "#6d4c41", 4),
]
for col, label, color, row in series:
    fig.add_trace(go.Scatter(x=g["timestamp"], y=g[col], name=label, line=dict(color=color, width=2)), row=row, col=1)

alerts = g[g[f"{prefix}_alert"].fillna(False)]
for col, _, _, row in series[:3]:
    fig.add_trace(
        go.Scatter(
            x=alerts["timestamp"],
            y=alerts[col],
            mode="markers",
            name=f"{model} alerts" if row == 1 else None,
            showlegend=row == 1,
            marker=dict(color="#111827", size=7, symbol="x"),
        ),
        row=row,
        col=1,
    )
events = g[g["deterioration_event"]]
if not events.empty:
    for row in range(1, 5):
        fig.add_vrect(x0=events["timestamp"].min(), x1=events["timestamp"].max(), fillcolor="#f59e0b", opacity=0.14, line_width=0, row=row, col=1)
fig.update_yaxes(title_text="bpm", row=1, col=1)
fig.update_yaxes(title_text="/min", row=2, col=1)
fig.update_yaxes(title_text="F", row=3, col=1)
fig.update_yaxes(title_text="0-1", row=4, col=1)
fig.update_layout(height=760, margin=dict(l=20, r=20, t=50, b=20), hovermode="x unified", legend_orientation="h")
st.plotly_chart(fig, use_container_width=True)

alert_rows = g[g[f"{prefix}_alert"].fillna(False)]
focus = alert_rows.iloc[0] if not alert_rows.empty else latest
st.subheader("Why This Alert?")
action_col = f"{prefix}_recommended_action"
if prefix == "model2":
    st.info(f"{focus['model2_tier']}: {focus['model2_reason']}")
elif prefix == "model1":
    st.info("Universal threshold alert: HR >= 100, RR >= 22, or temperature >= 99.5 F. These are hypothetical prototype thresholds chosen to demonstrate a sensitive monitoring workflow.")
else:
    st.info(f"{focus['model3_tier']}: logistic regression probability {focus['model3_score']:.2f}, using a threshold frozen from validation patients before test evaluation. Synthetic calibration only.")
st.success(f"Recommended action / routing: {focus[action_col]}")

st.dataframe(
    g[["timestamp", "heart_rate", "respiratory_rate", "temperature", "movement", "model1_tier", "model2_tier", "model2_recommended_action", "model2_reason"]].tail(80),
    use_container_width=True,
    hide_index=True,
)
