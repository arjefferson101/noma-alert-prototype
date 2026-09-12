# Noma Alert Triage Prototype

Polished, non-clinical Streamlit prototype for a hypothetical healthcare startup exploring alert-fatigue reduction after hospital discharge.

## The Problem

Continuous wearable monitoring can create too many threshold alerts. If clinicians see too many low-value notifications, meaningful deterioration signals can become harder to notice.

## Hypothesis

A sustained pattern across multiple physiological measures is more informative than an isolated threshold crossing. This prototype compares:

- **Model 1: Universal thresholds** using hypothetical population-level vital-sign cutoffs.
- **Model 2: Personalized interpretable rules** using patient baseline, trend, persistence, concordance, movement context, and missingness.
- **Model 3: Logistic regression** using the same engineered features to test whether simple ML adds enough value to justify complexity.

## Approach

The app generates completely synthetic longitudinal wearable data with heart rate, respiratory rate, temperature, movement, sensor artifacts, missing data, and simulated deterioration labels. It includes curated examples:

- `A_MOVEMENT_DEMO`: HR briefly rises to about 105 during high movement, then resolves.
- `B_DETERIORATION_DEMO`: HR, respiratory rate, and temperature rise together while movement remains low.

The pipeline is:

Raw wearable measurements -> artifact filtering -> patient-specific baseline -> rolling-window features -> risk score -> Tier 3 / Tier 2 / Tier 1.

Preprocessing is causal: missing values are only carried forward for short gaps using prior observations. The prototype does not use bidirectional interpolation or backward filling, because those would use future information that would not be available in a real-time alerting workflow. Longer current gaps are marked **Insufficient data**.

## Routing Logic

- **Tier 1:** urgent escalation. EMS only when a clinician-defined opt-in care plan explicitly authorizes it; this prototype never contacts EMS.
- **Tier 2:** clinician review.
- **Tier 3:** patient plus opt-in patient-designated caregiver/contact check-in. If the Tier 3 pattern persists or worsens over an illustrative interval, it escalates to Tier 2. The app does not actually contact anyone.
- **No alert:** continue monitoring.

## Statistical Safeguards

- **Patient-level splitting:** all rows from one patient stay entirely in train, validation, or test. Random row-level splitting would leak repeated observations from the same patient and inflate performance.
- **Class imbalance:** the dashboard reports sensitivity, specificity, PPV, false-positive rate, F1, ROC-AUC, PR-AUC, and clinician-facing alert episodes per patient/day instead of relying on accuracy.
- **Episode-level alert burden:** clinician alert burden is counted as Tier 1/Tier 2 clinician-facing alert episodes per patient/day and false clinician-facing alert episodes per patient/day, not individual positive timepoints. Tier 3 patient/caregiver check-in episodes are reported separately, and total routed episodes remain available for product planning.
- **Warning lead time:** event onset is the first timestamp in a contiguous synthetic deterioration episode. Alerts are credited only if matched to that episode inside a defined pre-event window, preventing unrelated earlier alerts from inflating lead time.
- **Validation-threshold selection:** Model 3's classification threshold is chosen on validation patients under a pre-specified high-sensitivity constraint, then frozen before test-set evaluation.
- **Confidence intervals:** key metrics use patient-level bootstrap resampling.
- **Subgroup evaluation:** results are shown by age cohort, sex, mobility, and discharge category, with small groups marked uncertain. Age-cohort bootstrap confidence intervals are shown where sample size permits.
- **Domain shift:** Model 3 is trained on synthetic post-discharge Cohort A and evaluated on both held-out Cohort A and shifted nursing-home / long-term-care Cohort B. Nursing-home data may help with longitudinal wearability and sensor-behavior learning, but cannot be assumed to generalize to post-discharge patients.
- **Calibration:** the app shows a synthetic calibration curve and Brier score for Model 3.

## Design Inspiration

Analogous continuous-monitoring devices such as CGMs motivated looking at temporal trends, sensor artifacts, transient abnormalities, and measurements in context rather than isolation. This is a design analogy only, not a claim that CGM clinical evidence validates the Noma prototype.

## Limitations

This is a startup challenge prototype, not a clinical product.

- All data are synthetic and illustrative.
- Thresholds and risk scores are hypothetical.
- Results do not demonstrate real-world effectiveness.
- The system is not clinically validated.
- Synthetic performance demonstrates the analysis framework, not evidence of clinical effectiveness.
- No automatic EMS dispatch is implemented.
- Any real escalation workflow would require prospective validation, physician oversight, operational protocols, and regulatory review.

## Running the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run tests:

```bash
pytest
```
