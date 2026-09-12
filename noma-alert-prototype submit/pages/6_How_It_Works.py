from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="How It Works", layout="wide")

st.title("How It Works")
st.caption("Synthetic demonstration only. Not a clinically validated decision-support system.")

st.markdown(
    """
```mermaid
flowchart LR
  A["Raw wearable measurements"] --> B["Artifact and missing-data processing"]
  B --> C["Patient-specific baseline"]
  C --> D["Rolling-window features"]
  D --> E["Interpretable risk score"]
  E --> F["Tier 3 / Tier 2 / Tier 1"]
```
"""
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Personalized Features")
    st.write("Baseline deviation compares a patient with their own initial stable period, not with a generic population average.")
    st.write("Persistence filtering asks whether an abnormal pattern lasts long enough to matter in the simulation.")
    st.write("Movement context down-weights brief HR spikes that happen with high activity.")
with c2:
    st.subheader("Statistical Safeguards")
    st.write("Train, validation, and test splits are by patient. Row-level random splits would leak the same person's repeated measurements into both train and test.")
    st.write("Metrics emphasize sensitivity, PPV, false-positive rate, PR-AUC, and clinician-facing alert episodes per patient/day, not accuracy.")
    st.write("Confidence intervals use patient-level bootstrap resampling because observations within a patient are correlated.")

st.warning("No automatic EMS dispatch is implemented. Any real clinical escalation would require prospective validation, clinical governance, and established protocols.")

st.subheader("Design Inspiration")
st.write(
    "Analogous continuous-monitoring devices, such as CGMs, motivated the prototype's focus on temporal trends, sensor artifacts, transient abnormalities, and contextual interpretation rather than isolated measurements. This is a product-design analogy only, not a claim that CGM evidence validates this synthetic Noma system."
)
