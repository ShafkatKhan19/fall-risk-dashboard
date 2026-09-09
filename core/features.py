"""
Exact ML feature schema, matching Designer Spec Section 9.

Column order matters: the model is trained expecting these exact columns in
this exact order. Never reorder without retraining.

Note: importing this module reads the model's explainability_config.json for
display names (see _bundle_display_names below).

This dashboard ships exactly ONE model: the real MIMIC-IV-trained
"Reduced Clinical + STEADI" bundle (Feature Set B). Earlier revisions also
carried synthetic placeholder RandomForests for a "Clinical only" (A) and a
"Clinical + JHFRAT" (C) feature set; those were unreachable from the UI, and
shipping unnecessary pickle files is both a deserialization risk and a way for
a synthetic placeholder to be mistaken for a validated model, so they are gone.
"""
import json
import os

from core.config import MODEL_DIR

# ---------------------------------------------------------------------------
# Reduced clinical predictors (16) — the clinical half of the deployed model.
# No race/marital columns: the trained bundle does not use them.
# ---------------------------------------------------------------------------
REDUCED_CLINICAL_COLS = [
    "age_cat_18_44",
    "age_cat_45_64",
    "age_cat_65plus",
    "gender_male",
    "fall_30d",
    "dx_hypertension",
    "dx_diabetes",
    "dx_heart_failure",
    "dx_gait",
    "dx_anxiety",
    "dx_dementia",
    "dx_stroke",
    "dx_parkinsons",
    "dx_arthritis",
    "dx_vision",
    "dx_hearing",
]

# ---------------------------------------------------------------------------
# STEADI-derived features (14) — Section 9.2 / 5.4
# ---------------------------------------------------------------------------
STEADI_DERIVED = [
    "steadi_low_risk",
    "steadi_at_risk",
    "steadi_s01_pts_(max_2)",
    "steadi_s02_pts_(max_2)",
    "steadi_s03_pts_(1pt)",
    "steadi_s04_pts_(1pt)",
    "steadi_s05_pts_(1pt)",
    "steadi_s06_pts_(1pt)",
    "steadi_s07_pts_(1pt)",
    "steadi_s08_pts_(1pt)",
    "steadi_s09_pts_(1pt)",
    "steadi_s10_pts_(1pt)",
    "steadi_s11_pts_(1pt)",
    "steadi_s12_pts_(1pt)",
]

# ---------------------------------------------------------------------------
# JHFRAT-derived features (10) — Section 9.3 / 6.5
# ---------------------------------------------------------------------------
JHFRAT_DERIVED = [
    "jhfrat_low",
    "jhfrat_moderate",
    "jhfrat_high",
    "jhfrat_d1_pts_age_0-3pts",
    "jhfrat_d2_pts_fall_history_0_or_5pts",
    "jhfrat_d3_pts_elimination_0-4pts",
    "jhfrat_d4_pts_medications_0-7pts",
    "jhfrat_d5_pts_equipment_0-3pts",
    "jhfrat_d6_pts_mobility_0-6pts",
    "jhfrat_d7_pts_cognition_0-7pts",
]

# The deployed model's exact feature contract — must equal
# bundle['feature_order'] in models/dashboard_v1/fall_risk_model_bundle.pkl.
# core/model.py asserts this at load time.
FEATURE_SET_B = REDUCED_CLINICAL_COLS + STEADI_DERIVED   # 30 features

FEATURE_SETS = {
    "B": {
        "name": "Clinical + STEADI (Trained Model)",
        "columns": FEATURE_SET_B,
        "requires": ["demographics", "comorbidities", "steadi"],
        "model_file": "dashboard_v1/fall_risk_model_bundle.pkl",
        "is_bundle": True,
    },
}

# ---------------------------------------------------------------------------
# Demographic field definitions — Section 4.2
# ---------------------------------------------------------------------------
AGE_GROUPS = {
    "18\u201344": "age_cat_18_44",
    "45\u201364": "age_cat_45_64",
    "65+": "age_cat_65plus",
}

GENDER_OPTIONS = {"Male": "gender_male", "Female": None}  # Female -> gender_male=0

RACE_OPTIONS = {
    "Asian": "race_asian",
    "Black": "race_black",
    "Hispanic": "race_hispanic",
    "Other": "race_other",
    "White": "race_white",
}

MARITAL_OPTIONS = {
    "Married": "marital_married",
    "Unknown": "marital_unknown",
    "Unmarried/Single": "marital_unmarried",
}

# ---------------------------------------------------------------------------
# Comorbidities — Section 4.3 (12 checkboxes -> 12 binary flags)
# ---------------------------------------------------------------------------
COMORBIDITY_LABELS = {
    "dx_hypertension": "Hypertension (high blood pressure)",
    "dx_diabetes": "Diabetes (any type)",
    "dx_heart_failure": "Heart Failure (congestive or systolic)",
    "dx_gait": "Gait Disorder (difficulty walking)",
    "dx_anxiety": "Anxiety Disorder",
    "dx_dementia": "Dementia / Cognitive Impairment",
    "dx_stroke": "Stroke (history of)",
    "dx_parkinsons": "Parkinson's Disease",
    "dx_arthritis": "Arthritis",
    "dx_vision": "Vision Impairment",
    "dx_hearing": "Hearing Impairment",
    "fall_30d": "Fall Occurred Within the Past 30 Days",
}

COMORBIDITY_COLUMNS = list(COMORBIDITY_LABELS.keys())

# Human-readable labels for *every* model column — used across Tab 4/5 and
# the cohort/assistant views so nothing is ever shown as a raw column name.
FEATURE_LABELS = {
    "age_cat_18_44": "Age 18\u201344",
    "age_cat_45_64": "Age 45\u201364",
    "age_cat_65plus": "Age 65+",
    "gender_male": "Male",
    "race_asian": "Race: Asian",
    "race_black": "Race: Black",
    "race_hispanic": "Race: Hispanic",
    "race_other": "Race: Other",
    "race_white": "Race: White",
    "marital_married": "Marital: Married",
    "marital_unknown": "Marital: Unknown",
    "marital_unmarried": "Marital: Unmarried/Single",
    "steadi_low_risk": "STEADI Low Risk",
    "steadi_at_risk": "STEADI At Risk",
    "jhfrat_low": "JHFRAT Low Risk",
    "jhfrat_moderate": "JHFRAT Moderate Risk",
    "jhfrat_high": "JHFRAT High Risk",
    "jhfrat_d1_pts_age_0-3pts": "JHFRAT \u2013 Age Domain",
    "jhfrat_d2_pts_fall_history_0_or_5pts": "JHFRAT \u2013 Prior Fall Domain",
    "jhfrat_d3_pts_elimination_0-4pts": "JHFRAT \u2013 Elimination Domain",
    "jhfrat_d4_pts_medications_0-7pts": "JHFRAT \u2013 Medications Domain",
    "jhfrat_d5_pts_equipment_0-3pts": "JHFRAT \u2013 Equipment Domain",
    "jhfrat_d6_pts_mobility_0-6pts": "JHFRAT \u2013 Mobility Domain",
    "jhfrat_d7_pts_cognition_0-7pts": "JHFRAT \u2013 Cognition Domain",
}
FEATURE_LABELS.update(COMORBIDITY_LABELS)
for i in range(1, 13):
    key = [c for c in STEADI_DERIVED if c.startswith(f"steadi_s{i:02d}_")]
    if key:
        FEATURE_LABELS[key[0]] = f"STEADI Item {i}"


def _bundle_display_names() -> dict:
    """Display names shipped alongside the model in explainability_config.json.

    These are the model authors' own labels and are far more informative than
    the generic "STEADI Item 11" fallbacks above — that file calls item 11
    "Sleep or mood medication". Showing a clinician "STEADI Item 11" as a
    reason for a risk estimate is not an explanation, so the bundle's names win
    wherever it defines one.
    """
    path = os.path.join(MODEL_DIR, "dashboard_v1", "explainability_config.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("feature_display_names", {}) or {}
    except (OSError, ValueError):
        return {}


FEATURE_LABELS.update(_bundle_display_names())


def label_for(column: str) -> str:
    return FEATURE_LABELS.get(column, column)
