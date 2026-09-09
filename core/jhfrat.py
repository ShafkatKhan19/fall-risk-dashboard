"""
JHFRAT scoring logic — Designer Spec Section 6.
"""
from core.config import JHFRAT_MODERATE_CUTOFF, JHFRAT_HIGH_CUTOFF

PRESCREEN_ITEMS = [
    {"id": "ps1", "text": "History of more than one fall within 6 months before admission"},
    {"id": "ps2", "text": "Patient has experienced a fall during this hospitalization"},
    {"id": "ps3", "text": "Patient is deemed high fall-risk per protocol (e.g., seizure precautions)"},
    {"id": "ps4", "text": "Patient is completely paralyzed or immobilized (triggers Low Fall Risk protocol instead)"},
]

# Domains D1-D5: single-select dropdowns. D6/D7: multi-select checklists.
DOMAIN_1_AGE = {
    "column": "jhfrat_d1_pts_age_0-3pts",
    "label": "Age Category",
    "options": {"Under 60": 0, "60\u201369": 1, "70\u201379": 2, "80+": 3},
}
DOMAIN_2_FALL_HISTORY = {
    "column": "jhfrat_d2_pts_fall_history_0_or_5pts",
    "label": "Prior Fall (6 months)",
    "options": {"No prior fall": 0, "One fall in past 6 months": 5},
}
DOMAIN_3_ELIMINATION = {
    "column": "jhfrat_d3_pts_elimination_0-4pts",
    "label": "Bladder/Bowel Function",
    "options": {"Normal": 0, "Incontinence": 2, "Urgency or frequency": 2, "Both": 4},
}
DOMAIN_4_MEDICATIONS = {
    "column": "jhfrat_d4_pts_medications_0-7pts",
    "label": "High-Risk Medications",
    "options": {
        "None": 0,
        "1 high-risk drug": 3,
        "2+ high-risk drugs": 5,
        "Sedated procedure within 24h": 7,
    },
}
DOMAIN_5_EQUIPMENT = {
    "column": "jhfrat_d5_pts_equipment_0-3pts",
    "label": "Tethering Equipment",
    "options": {"None": 0, "1 item": 1, "2 items": 2, "3 or more": 3},
}
DOMAIN_6_MOBILITY = {
    "column": "jhfrat_d6_pts_mobility_0-6pts",
    "label": "Mobility",
    "options": {
        "Needs assistance/supervision": 2,
        "Unsteady gait": 2,
        "Visual or auditory impairment": 2,
    },
}
DOMAIN_7_COGNITION = {
    "column": "jhfrat_d7_pts_cognition_0-7pts",
    "label": "Cognition",
    "options": {
        "Altered environmental awareness": 1,
        "Impulsive behavior": 2,
        "Lacks insight into own limitations": 4,
    },
}

DROPDOWN_DOMAINS = [
    DOMAIN_1_AGE, DOMAIN_2_FALL_HISTORY, DOMAIN_3_ELIMINATION,
    DOMAIN_4_MEDICATIONS, DOMAIN_5_EQUIPMENT,
]
MULTISELECT_DOMAINS = [DOMAIN_6_MOBILITY, DOMAIN_7_COGNITION]
ALL_DOMAINS = DROPDOWN_DOMAINS + MULTISELECT_DOMAINS

JHFRAT_MAX_SCORE = 3 + 5 + 4 + 7 + 3 + 6 + 7  # 35


def score_from_domains(domain_points: dict) -> int:
    """domain_points: {column: points}. Returns total score 0-35."""
    return sum(domain_points.get(d["column"], 0) for d in ALL_DOMAINS)


def flags_from_domains(domain_points: dict) -> dict:
    total = score_from_domains(domain_points)
    flags = {
        "jhfrat_low": int(total < JHFRAT_MODERATE_CUTOFF),
        "jhfrat_moderate": int(JHFRAT_MODERATE_CUTOFF <= total <= JHFRAT_HIGH_CUTOFF),
        "jhfrat_high": int(total > JHFRAT_HIGH_CUTOFF),
    }
    for d in ALL_DOMAINS:
        flags[d["column"]] = domain_points.get(d["column"], 0)
    return flags


def flags_from_total(total_score: int) -> dict:
    """
    Back-calculate flags from a pre-computed total (Mode B). Domain columns
    are distributed proportionally across domain max points so the feature
    vector stays internally consistent (same approach as STEADI Mode B).
    """
    flags = {
        "jhfrat_low": int(total_score < JHFRAT_MODERATE_CUTOFF),
        "jhfrat_moderate": int(JHFRAT_MODERATE_CUTOFF <= total_score <= JHFRAT_HIGH_CUTOFF),
        "jhfrat_high": int(total_score > JHFRAT_HIGH_CUTOFF),
    }
    domain_maxes = {
        "jhfrat_d1_pts_age_0-3pts": 3,
        "jhfrat_d2_pts_fall_history_0_or_5pts": 5,
        "jhfrat_d3_pts_elimination_0-4pts": 4,
        "jhfrat_d4_pts_medications_0-7pts": 7,
        "jhfrat_d5_pts_equipment_0-3pts": 3,
        "jhfrat_d6_pts_mobility_0-6pts": 6,
        "jhfrat_d7_pts_cognition_0-7pts": 7,
    }
    max_total = sum(domain_maxes.values())
    remaining = total_score
    for col, cap in domain_maxes.items():
        share = round(cap * (total_score / max_total)) if max_total else 0
        share = min(share, cap, remaining)
        flags[col] = share
        remaining -= share
    return flags


def risk_level(total_score: int) -> str:
    if total_score > JHFRAT_HIGH_CUTOFF:
        return "High Risk"
    if total_score >= JHFRAT_MODERATE_CUTOFF:
        return "Moderate Risk"
    return "Low Risk"
