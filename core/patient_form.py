"""
Unified single-page patient intake — matches "Information Required from the
Users.csv" exactly (27 columns): one form collects demographics, clinical
history, fall history/functional status, and medication/mood in one pass.
STEADI is entirely derived from this same form (Sections C/D map 1:1 onto the
12 STEADI items) rather than re-asked on a separate wizard tab, so nothing is
answered twice.

This module is the single source of truth for the raw-form <-> model-feature
mapping, used by both the manual-entry page and CSV batch upload so the two
input paths can never drift apart.
"""
from core.steadi import STEADI_ITEMS, flags_from_items, score_from_items

# ---------------------------------------------------------------------------
# Section A — Demographics
# ---------------------------------------------------------------------------
SEX_OPTIONS = {"Female": 0, "Male": 1}
AGE_MIN, AGE_MAX = 18, 120


def age_to_bucket_column(age: int) -> str:
    if age < 45:
        return "age_cat_18_44"
    if age < 65:
        return "age_cat_45_64"
    return "age_cat_65plus"


# ---------------------------------------------------------------------------
# Section B — Clinical History (11 comorbidity checkboxes; fall_30d lives in
# Section C alongside the rest of fall history, matching the source CSV).
# ---------------------------------------------------------------------------
CLINICAL_HISTORY_FIELDS = [
    ("hypertension", "dx_hypertension", "Has hypertension been diagnosed?"),
    ("diabetes", "dx_diabetes", "Has diabetes been diagnosed?"),
    ("heart_failure", "dx_heart_failure", "Has heart failure been diagnosed?"),
    ("gait_disorder", "dx_gait", "Is a gait disorder or persistent walking difficulty documented?"),
    ("anxiety_disorder", "dx_anxiety", "Is an anxiety disorder documented?"),
    ("cognitive_impairment", "dx_dementia", "Is cognitive impairment or dementia documented?"),
    ("stroke_history", "dx_stroke", "Is there a history of stroke?"),
    ("parkinsons_disease", "dx_parkinsons", "Has Parkinson's disease been diagnosed?"),
    ("arthritis", "dx_arthritis", "Is arthritis documented?"),
    ("vision_impairment", "dx_vision", "Does the patient have a documented vision impairment?"),
    ("hearing_impairment", "dx_hearing", "Does the patient have a documented hearing impairment?"),
]

# ---------------------------------------------------------------------------
# Section C — Fall History and Functional Status.
# fall_past_30d is a direct clinical feature (fall_30d). fall_past_year and
# everything else in this section is STEADI item s01-s09 in disguise — the
# CSV wording is patient-friendly, the underlying STEADI item is identical.
# ---------------------------------------------------------------------------
FALL_HISTORY_FIELDS = [
    ("fall_past_30d", "fall_30d", "Has a fall been recorded in the past 30 days?"),
]

# form_key -> steadi item id, in the exact order of "Information Required from the Users.csv"
STEADI_FORM_MAP = [
    ("fall_past_year", "s01", "Has a fall been recorded in the past 12 months?"),
    ("uses_cane_or_walker", "s02", "Does the patient currently use a cane or walker for mobility?"),
    ("unsteady_walking", "s03", "Is unsteady walking currently documented?"),
    ("holds_furniture_for_support", "s04", "Does the patient use furniture or nearby surfaces for walking support?"),
    ("concern_about_falling", "s05", "Does the patient report concern about falling?"),
    ("uses_arms_to_stand", "s06", "Does the patient use the arms or hands to rise from a chair?"),
    ("difficulty_with_curb_or_step", "s07", "Does the patient have difficulty stepping onto a curb or step?"),
    ("toileting_urgency", "s08", "Does urgency cause the patient to rush to the toilet?"),
    ("reduced_foot_sensation", "s09", "Is reduced sensation or numbness in the feet documented?"),
]

# ---------------------------------------------------------------------------
# Section D — Medication and Mood Factors (STEADI items s10-s12).
# ---------------------------------------------------------------------------
MEDICATION_MOOD_FORM_MAP = [
    ("dizziness_or_fatigue_medication", "s10",
     "Is the patient taking medication associated with dizziness or fatigue?"),
    ("sleep_or_mood_medication", "s11", "Is the patient taking medication for sleep or mood?"),
    ("low_mood", "s12", "Are low mood or depressive symptoms documented?"),
]

ALL_STEADI_FORM_MAP = STEADI_FORM_MAP + MEDICATION_MOOD_FORM_MAP

# Full raw-form column order, matching "Information Required from the Users.csv"
RAW_FORM_COLUMNS = (
    ["study_id", "age_years", "sex"]
    + [csv_col for csv_col, _, _ in CLINICAL_HISTORY_FIELDS]
    + [csv_col for csv_col, _, _ in FALL_HISTORY_FIELDS]
    + [csv_col for csv_col, _, _ in STEADI_FORM_MAP]
    + [csv_col for csv_col, _, _ in MEDICATION_MOOD_FORM_MAP]
)


def build_demographics(age: int, sex: str) -> dict:
    flags = {"age_cat_18_44": 0, "age_cat_45_64": 0, "age_cat_65plus": 0}
    flags[age_to_bucket_column(age)] = 1
    flags["gender_male"] = SEX_OPTIONS.get(sex, 0)
    return flags


def build_comorbidities(clinical_answers: dict, fall_30d_answer: bool) -> dict:
    """clinical_answers: {csv_col: bool} for CLINICAL_HISTORY_FIELDS."""
    flags = {model_col: int(bool(clinical_answers.get(csv_col, False)))
             for csv_col, model_col, _ in CLINICAL_HISTORY_FIELDS}
    flags["fall_30d"] = int(bool(fall_30d_answer))
    return flags


def build_steadi_answers(form_answers: dict) -> dict:
    """form_answers: {csv_col: bool} for every field in ALL_STEADI_FORM_MAP.
    Returns {steadi_item_id: bool} ready for core.steadi.flags_from_items / score_from_items.
    """
    return {item_id: bool(form_answers.get(csv_col, False)) for csv_col, item_id, _ in ALL_STEADI_FORM_MAP}


def full_row_to_features(row: dict) -> dict:
    """Convert one raw-form row (matching RAW_FORM_COLUMNS / the Information
    Required CSV schema) into every model-ready flag: demographics,
    comorbidities (incl. fall_30d), and the 14 STEADI-derived columns.
    """
    demographics = build_demographics(int(row["age_years"]), str(row["sex"]))
    comorbidities = build_comorbidities(
        {csv_col: row.get(csv_col, 0) for csv_col, _, _ in CLINICAL_HISTORY_FIELDS},
        row.get("fall_past_30d", 0),
    )
    steadi_answers = build_steadi_answers({csv_col: row.get(csv_col, 0) for csv_col, _, _ in ALL_STEADI_FORM_MAP})
    steadi_flags = flags_from_items(steadi_answers)

    features = {}
    features.update(demographics)
    features.update(comorbidities)
    features.update(steadi_flags)
    return features


def _question_for(csv_col: str) -> str:
    for group in (CLINICAL_HISTORY_FIELDS, FALL_HISTORY_FIELDS):
        for col, _model, question in group:
            if col == csv_col:
                return question
    for col, _item, question in ALL_STEADI_FORM_MAP:
        if col == csv_col:
            return question
    return {
        "study_id": "A de-identified study ID (letters and numbers only).",
        "age_years": "The patient's age in years.",
        "sex": "The patient's sex.",
    }.get(csv_col, "")


def _allowed_for(csv_col: str) -> str:
    if csv_col == "study_id":
        return "text, 1-20 letters/numbers"
    if csv_col == "age_years":
        return f"integer {AGE_MIN}-{AGE_MAX}"
    if csv_col == "sex":
        return " | ".join(SEX_OPTIONS)
    return "0 or 1"


def csv_column_reference():
    """A table describing every column the upload expects."""
    import pandas as pd

    return pd.DataFrame(
        {
            "column": RAW_FORM_COLUMNS,
            "allowed values": [_allowed_for(c) for c in RAW_FORM_COLUMNS],
            "question": [_question_for(c) for c in RAW_FORM_COLUMNS],
        }
    )


def build_csv_template() -> str:
    """Blank upload template: the header row plus one example row, so users can
    see the accepted values instead of guessing from the column names."""
    example = {c: "0" for c in RAW_FORM_COLUMNS}
    example["study_id"] = "EXAMPLE01"
    example["age_years"] = "78"
    example["sex"] = "Female"
    header = ",".join(RAW_FORM_COLUMNS)
    row = ",".join(example[c] for c in RAW_FORM_COLUMNS)
    return f"{header}\n{row}\n"


def build_csv_example() -> str:
    """The shipped 10-patient synthetic sample, for trying the batch flow."""
    from core.config import DATA_DIR

    path = DATA_DIR / "sample_patients.csv"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return build_csv_template()


def load_row_into_session(row: dict, session_state) -> None:
    """Populate session_state exactly as manual entry's "Save & Continue" does,
    from one raw-form row (a dict matching RAW_FORM_COLUMNS — e.g. one row of
    an uploaded CSV, or the manual-entry form's own answers). This is the single
    place both input paths go through, so Fall Risk Scores / In-Hospital Fall
    Risk work identically regardless of how the patient's data got in.
    """
    demographics = build_demographics(int(row["age_years"]), str(row["sex"]))
    comorbidities = build_comorbidities(
        {csv_col: row.get(csv_col, 0) for csv_col, _, _ in CLINICAL_HISTORY_FIELDS},
        row.get("fall_past_30d", 0),
    )
    steadi_answers = build_steadi_answers({csv_col: row.get(csv_col, 0) for csv_col, _, _ in ALL_STEADI_FORM_MAP})
    steadi_score = score_from_items(steadi_answers)
    steadi_flags = flags_from_items(steadi_answers)

    session_state["patient_id"] = str(row.get("study_id", ""))
    session_state["form_age"] = int(row["age_years"])
    session_state["demographics"] = demographics
    session_state["demographics_complete"] = True
    session_state["comorbidities"] = comorbidities
    session_state["comorbidities_complete"] = True
    session_state["steadi_items"] = steadi_answers
    session_state["steadi_score"] = steadi_score
    session_state["steadi_flags"] = steadi_flags
    session_state["steadi_complete"] = True
    session_state["selected_feature_set"] = "B"

    # A previously loaded/entered patient's prediction must not leak onto the
    # newly loaded one — clear it so the In-Hospital Fall Risk page recomputes.
    session_state["prediction_probability"] = None
    session_state["prediction_ci"] = None
    session_state["prediction_risk_tier"] = None
    session_state["prediction_decision_threshold"] = None
    session_state["feature_vector"] = None
    session_state["shap_values"] = None
    session_state["shap_base_value"] = None
