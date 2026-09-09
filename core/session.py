"""
st.session_state schema and helpers — Designer Spec Section 11.
"""
import streamlit as st

DEFAULTS = {
    # Tab 1
    "patient_id": "",
    "demographics": {},          # dict of one-hot encoded demographic flags
    "demographics_complete": False,
    "comorbidities": {},         # dict of binary comorbidity flags
    "comorbidities_complete": False,
    "steadi_mode": "item",       # 'item' or 'total'
    "jhfrat_mode": "item",       # 'item' or 'total'

    # Tab 2 (STEADI)
    "steadi_items": {},          # raw item answers {id: bool}
    "steadi_score": None,
    "steadi_flags": {},          # 14 model columns
    "steadi_complete": False,

    # Tab 3 (JHFRAT)
    "jhfrat_prescreen_flag": False,
    "jhfrat_prescreen_reason": None,
    "jhfrat_domains": {},        # raw domain points {column: pts}
    "jhfrat_score": None,
    "jhfrat_flags": {},          # 10 model columns
    "jhfrat_complete": False,

    # Tab 4
    "selected_feature_set": "B",
    "prediction_probability": None,
    "prediction_ci": None,       # (low, high) uncertainty band
    "prediction_risk_tier": None,
    "prediction_decision_threshold": None,
    "feature_vector": None,      # 1-row DataFrame

    # Tab 5
    "shap_values": None,
    "shap_base_value": None,

    # Cohort / batch — deliberately excluded from reset_patient() below, so
    # switching tabs or starting a new single-patient assessment never loses
    # an uploaded cohort.
    "cohort_df": None,
    "cohort_results": None,
    "cohort_valid_df": None,
    "cohort_errors": None,
    "cohort_upload_name": None,
    "cohort_row_count": None,
}

_COHORT_KEYS = ("cohort_df", "cohort_results", "cohort_valid_df",
                "cohort_errors", "cohort_upload_name", "cohort_row_count")


def init_session_state():
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_patient():
    """Clear everything for the current patient (keep cohort data)."""
    for key, value in DEFAULTS.items():
        if key not in _COHORT_KEYS:
            st.session_state[key] = value


def progress_checklist() -> dict:
    patient_data_done = bool(
        st.session_state.get("demographics_complete") and
        st.session_state.get("comorbidities_complete") and
        st.session_state.get("steadi_complete")
    )
    return {
        "Home & Patient Data": patient_data_done,
        "Fall Risk Scores": bool(st.session_state.get("steadi_score") is not None),
        "In-Hospital Fall Risk": bool(st.session_state.get("prediction_probability") is not None),
    }
