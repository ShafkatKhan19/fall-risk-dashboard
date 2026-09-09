import pandas as pd
import streamlit as st

from core.config import DISCLAIMER_TEXT
from core.cohort import score_cohort_raw
from core.patient_form import (
    AGE_MAX,
    AGE_MIN,
    CLINICAL_HISTORY_FIELDS,
    MEDICATION_MOOD_FORM_MAP,
    RAW_FORM_COLUMNS,
    SEX_OPTIONS,
    STEADI_FORM_MAP,
    load_row_into_session,
)
from core.session import reset_patient
from core.validation import validate_patient_id, validate_raw_form_csv

st.session_state.setdefault("entry_view", "welcome")


def go_to(view):
    st.session_state["entry_view"] = view


# ===========================================================================
# WELCOME SCREEN
# ===========================================================================
if st.session_state["entry_view"] == "welcome":
    st.markdown(
        '''
        <div class="card-hero">
          <h1>Welcome to the Fall Risk Dashboard</h1>
          <p style="font-size:1.02rem; opacity:0.94; max-width:640px; margin-bottom:0;">
            This tool calculates an established fall risk score using information you
            provide. It also estimates the risk of an in-hospital fall and shows the
            contributing factors that influence the prediction.
          </p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.subheader("Get Started")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### :material/edit_note: Enter information manually")
        st.caption("Fill out one patient's data through a guided form.")
        if st.button("Enter data", type="primary", key="go_manual", use_container_width=True):
            go_to("manual")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### :material/upload_file: Upload a CSV file")
        st.caption("Score an entire cohort at once from a spreadsheet.")
        if st.button("Upload CSV", key="go_csv", use_container_width=True):
            go_to("csv")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.caption(f":material/info: {DISCLAIMER_TEXT}. Not intended for clinical decision-making or use in patient care.")
    st.info(
        "The associated study also evaluated the Johns Hopkins Fall Risk Assessment "
        "Tool (JHFRAT). JHFRAT calculation is not provided in this open-access "
        "dashboard. Researchers wishing to use JHFRAT should obtain research "
        "authorization directly from Johns Hopkins University."
    )
    st.stop()

# ===========================================================================
# FORM CHROME (shared by manual entry and CSV upload)
# ===========================================================================
st.title("Patient Data Entry")

view = st.session_state["entry_view"]
toggle = st.radio(
    "Input Method", ["Enter Manually", "Upload CSV"],
    index=0 if view == "manual" else 1,
    horizontal=True, label_visibility="collapsed",
)
go_to("manual" if toggle == "Enter Manually" else "csv")

# ===========================================================================
# MANUAL ENTRY
# ===========================================================================
if st.session_state["entry_view"] == "manual":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(":material/person: A. Patient Demographics")
    c1, c2, c3 = st.columns(3)
    with c1:
        study_id = st.text_input("Study ID", value=st.session_state.get("patient_id", ""),
                                  placeholder="Enter study ID")
    with c2:
        age_years = st.number_input(
            "Enter the patient's age in years.", min_value=AGE_MIN, max_value=AGE_MAX,
            value=st.session_state.get("form_age", AGE_MIN), step=1,
        )
    with c3:
        sex = st.selectbox("Select the patient's sex.", ["— Select —"] + list(SEX_OPTIONS.keys()))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(":material/medical_information: B. Clinical History")
    clinical_answers = {}
    cols = st.columns(2)
    half = len(CLINICAL_HISTORY_FIELDS) // 2 + len(CLINICAL_HISTORY_FIELDS) % 2
    for i, (csv_col, model_col, question) in enumerate(CLINICAL_HISTORY_FIELDS):
        target = cols[0] if i < half else cols[1]
        with target:
            answer = st.radio(question, ["Yes", "No"], index=1, horizontal=True, key=f"ch_{csv_col}")
            clinical_answers[csv_col] = answer == "Yes"
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(":material/directions_walk: C. Fall History and Functional Status")
    fall_30d_answer = st.radio(
        "Has a fall been recorded in the past 30 days?", ["Yes", "No"], index=1, horizontal=True,
        key="ch_fall_past_30d",
    ) == "Yes"
    steadi_c_answers = {}
    cols = st.columns(2)
    half = len(STEADI_FORM_MAP) // 2 + len(STEADI_FORM_MAP) % 2
    for i, (csv_col, item_id, question) in enumerate(STEADI_FORM_MAP):
        target = cols[0] if i < half else cols[1]
        with target:
            answer = st.radio(question, ["Yes", "No"], index=1, horizontal=True, key=f"st_{csv_col}")
            steadi_c_answers[csv_col] = answer == "Yes"
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(":material/medication: D. Medication and Mood Factors")
    steadi_d_answers = {}
    cols = st.columns(3)
    for i, (csv_col, item_id, question) in enumerate(MEDICATION_MOOD_FORM_MAP):
        with cols[i % 3]:
            answer = st.radio(question, ["Yes", "No"], index=1, horizontal=True, key=f"st_{csv_col}")
            steadi_d_answers[csv_col] = answer == "Yes"
    st.markdown("</div>", unsafe_allow_html=True)

    b1, b2 = st.columns([1, 1])
    with b1:
        clear_clicked = st.button(":material/refresh: Clear Form", use_container_width=True)
    with b2:
        save_clicked = st.button("Save & Continue →", type="primary", use_container_width=True)

    if clear_clicked:
        reset_patient()
        for key in list(st.session_state.keys()):
            if key.startswith(("ch_", "st_")):
                del st.session_state[key]
        st.rerun()

    if save_clicked:
        pid_ok, pid_err = validate_patient_id(study_id)
        errors = []
        if not pid_ok:
            errors.append(f"Study ID: {pid_err}")
        if sex == "— Select —":
            errors.append("Please select the patient's sex.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            row = {"study_id": study_id, "age_years": int(age_years), "sex": sex}
            row.update({csv_col: clinical_answers.get(csv_col, False) for csv_col, _, _ in CLINICAL_HISTORY_FIELDS})
            row["fall_past_30d"] = fall_30d_answer
            row.update(steadi_c_answers)
            row.update(steadi_d_answers)
            load_row_into_session(row, st.session_state)
            st.success("Patient data saved. Continue to **Fall Risk Scores** in the sidebar.")

# ===========================================================================
# CSV UPLOAD
# ===========================================================================
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(":material/upload_file: Upload CSV")
    st.caption(
        "Header row required, column names must exactly match the schema below "
        "(case-sensitive). One row = one patient. UTF-8, 50 MB max, 10,000 rows max."
    )
    with st.expander("Required columns"):
        st.code(", ".join(RAW_FORM_COLUMNS))
    uploaded = st.file_uploader("CSV file", type=["csv"], key="csv_uploader")
    st.markdown("</div>", unsafe_allow_html=True)

    MAX_CSV_ROWS = 10_000

    # A NEW file was just picked in this run — (re)validate it and stash the
    # result in session_state. Streamlit's file_uploader does NOT keep its
    # uploaded bytes across a page switch in a multipage app: navigating to
    # another tab and back always resets `uploaded` to None, even though the
    # user never "removed" anything. Everything below therefore reads from
    # session_state, not from `uploaded` directly, so a previously uploaded
    # and scored cohort keeps showing up here (and stays available to load a
    # patient from) no matter how many times you switch tabs.
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, nrows=MAX_CSV_ROWS + 1)
        except Exception:
            st.error("Could not read file. Confirm it's a valid UTF-8 CSV with a header row.")
            df = None

        if df is not None and len(df) > MAX_CSV_ROWS:
            st.error(f"CSV exceeds the {MAX_CSV_ROWS:,}-row limit. Split it into smaller files.")
            df = None

        if df is not None:
            valid_df, errors = validate_raw_form_csv(df)
            st.session_state["cohort_upload_name"] = uploaded.name
            st.session_state["cohort_row_count"] = len(df)
            st.session_state["cohort_valid_df"] = valid_df
            st.session_state["cohort_errors"] = errors
            # A fresh upload invalidates any previous batch-prediction results
            # for the old file — force "Run Batch Prediction" again.
            st.session_state["cohort_results"] = None

    valid_df = st.session_state.get("cohort_valid_df")
    errors = st.session_state.get("cohort_errors") or []
    scored = st.session_state.get("cohort_results")

    if valid_df is not None:
        upload_name = st.session_state.get("cohort_upload_name", "uploaded file")
        total_rows = st.session_state.get("cohort_row_count", len(valid_df))
        st.write(f"**{upload_name}** — {len(valid_df)} of {total_rows} rows loaded successfully.")

        if errors:
            st.error(f"{len(errors)} validation issue(s) found:")
            st.dataframe(pd.DataFrame(errors), use_container_width=True)

        if len(valid_df) > 0:
            st.write("Preview — first 5 valid rows:")
            st.dataframe(valid_df.head(5), use_container_width=True)

            if scored is None:
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner("Scoring cohort..."):
                        scored = score_cohort_raw(valid_df)
                    st.session_state["cohort_df"] = valid_df
                    st.session_state["cohort_results"] = scored
                    st.rerun()
            else:
                st.success(
                    f"Scored {len(scored)} patients. Open **Cohort Analytics** in the sidebar "
                    "for the full breakdown, or load one patient below to see their individual "
                    "Fall Risk Score and In-Hospital Fall Risk."
                )

                study_ids = scored["study_id"].astype(str).tolist()
                selected_id = st.selectbox("Load one patient's individual results", study_ids)
                if st.button("Load selected patient"):
                    row = valid_df[valid_df["study_id"].astype(str) == selected_id].iloc[0].to_dict()
                    load_row_into_session(row, st.session_state)
                    st.success(
                        f"Loaded **{selected_id}**. Open **Fall Risk Scores** or "
                        "**In-Hospital Fall Risk** in the sidebar."
                    )

                if st.button("Clear uploaded cohort", type="secondary"):
                    for key in ("cohort_df", "cohort_results", "cohort_valid_df",
                                "cohort_errors", "cohort_upload_name", "cohort_row_count"):
                        st.session_state[key] = None
                    st.rerun()
