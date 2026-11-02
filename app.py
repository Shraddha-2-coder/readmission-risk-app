import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Readmission Risk Stratification", page_icon="🏥")


@st.cache_resource
def load_assets():
    art = joblib.load("readmission_model.joblib")
    with open("defaults.json") as f:
        defaults = json.load(f)
    return art, defaults


art, defaults = load_assets()
model, calibrator = art["model"], art["calibrator"]
threshold, feature_columns = art["threshold"], art["feature_columns"]

st.title("🏥 30-Day Readmission Risk Stratification")
st.warning("Administrative risk stratification tool for post-discharge care "
           "coordination. It is NOT a diagnostic tool and does not replace "
           "physician judgment.")

st.subheader("Patient details at discharge")
c1, c2 = st.columns(2)

DISPOSITIONS = {
    "Home": "1",
    "Home with health service": "6",
    "Skilled nursing facility": "3",
    "Short-term hospital": "2",
    "Rehab facility": "22",
}

with c1:
    age = st.number_input("Age (years)", 0, 100, 65)
    gender = st.selectbox("Gender", ["Female", "Male"])
    time_in_hospital = st.slider("Days in hospital", 1, 14, 4)
    num_medications = st.slider("Number of medications", 1, 80, 15)
    num_lab_procedures = st.slider("Number of lab procedures", 1, 130, 43)
    num_procedures = st.slider("Number of procedures", 0, 6, 1)
    number_diagnoses = st.slider("Number of diagnoses", 1, 16, 8)
    dx_group = st.selectbox("Primary diagnosis group", [
        "Circulatory", "Diabetes", "Respiratory", "Digestive", "Injury",
        "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"])

with c2:
    number_inpatient = st.number_input("Inpatient admissions, past 12 months", 0, 20, 0)
    number_emergency = st.number_input("Emergency visits, past 12 months", 0, 20, 0)
    number_outpatient = st.number_input("Outpatient visits, past 12 months", 0, 40, 0)
    disposition_label = st.selectbox("Discharged to", list(DISPOSITIONS.keys()))
    a1c = st.selectbox("HbA1c result", ["NotTested", "Norm", ">7", ">8"])
    glucose = st.selectbox("Serum glucose result", ["NotTested", "Norm", ">200", ">300"])
    insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
    heart_failure = st.checkbox("Heart failure")
    renal = st.checkbox("Kidney disease")
    copd = st.checkbox("COPD")

if st.button("Calculate risk", type="primary"):
    row = dict(defaults)
    age_num = min(age // 10, 9) * 10 + 5
    total_prior = number_inpatient + number_emergency + number_outpatient
    row.update({
        "gender": gender,
        "age_num": age_num,
        "time_in_hospital": time_in_hospital,
        "num_medications": num_medications,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "number_diagnoses": number_diagnoses,
        "primary_dx_group": dx_group,
        "number_inpatient": number_inpatient,
        "number_emergency": number_emergency,
        "number_outpatient": number_outpatient,
        "discharge_disposition_id": DISPOSITIONS[disposition_label],
        "A1Cresult": a1c,
        "max_glu_serum": glucose,
        "insulin": insulin,
        "has_diabetes_dx": 1,
        "has_heart_failure": int(heart_failure),
        "has_renal_disease": int(renal),
        "has_copd": int(copd),
        "comorbidity_score": 1 + int(heart_failure) + int(renal) + int(copd),
        "total_prior_visits": total_prior,
        "emergency_share": number_emergency / (total_prior + 1),
        "abnormal_a1c": int(a1c in [">7", ">8"]),
        "abnormal_glucose": int(glucose in [">200", ">300"]),
    })
    X_new = pd.DataFrame([row])[feature_columns]
    raw = model.predict_proba(X_new)[:, 1]
    risk = float(calibrator.predict(raw)[0])

    st.subheader("Result")
    st.metric("Estimated 30-day readmission risk", f"{risk:.1%}")
    st.caption("Average risk across all patients in the training data is about 11%.")

    if risk >= threshold:
        st.error(f"HIGH RISK (at or above the {threshold:.1%} action threshold). "
                 "Recommended: nurse phone call within 48 hours and a primary "
                 "care visit within 7 days.")
    else:
        st.success(f"Standard risk (below the {threshold:.1%} action threshold). "
                   "Routine discharge follow-up.")

    reasons = []
    if number_inpatient >= 2:
        reasons.append("2 or more inpatient admissions in the past year")
    if number_emergency >= 1:
        reasons.append("Emergency visits in the past year")
    if number_diagnoses >= 9:
        reasons.append("High number of diagnoses")
    if num_medications >= 20:
        reasons.append("Polypharmacy (20 or more medications)")
    if time_in_hospital >= 7:
        reasons.append("Long hospital stay")
    if heart_failure or renal or copd:
        reasons.append("Serious chronic condition recorded")
    if reasons:
        st.write("**Factors present that are commonly linked to higher risk:**")
        for r in reasons:
            st.write("- " + r)
    st.caption("These are statistical associations, not causes. "
               "A clinician makes the final decision.")
