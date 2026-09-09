"""
STEADI scoring logic — Designer Spec Section 5.
"""
from core.config import STEADI_AT_RISK_CUTOFF

STEADI_ITEMS = [
    {"id": "s01", "column": "steadi_s01_pts_(max_2)", "points": 2,
     "text": "Have you fallen in the past year?"},
    {"id": "s02", "column": "steadi_s02_pts_(max_2)", "points": 2,
     "text": "Do you use or have been advised to use a cane or walker?"},
    {"id": "s03", "column": "steadi_s03_pts_(1pt)", "points": 1,
     "text": "Do you sometimes feel unsteady when walking?"},
    {"id": "s04", "column": "steadi_s04_pts_(1pt)", "points": 1,
     "text": "Do you steady yourself by holding onto furniture when walking at home?"},
    {"id": "s05", "column": "steadi_s05_pts_(1pt)", "points": 1,
     "text": "Are you worried about falling?"},
    {"id": "s06", "column": "steadi_s06_pts_(1pt)", "points": 1,
     "text": "Do you need to push with your hands to stand up from a chair?"},
    {"id": "s07", "column": "steadi_s07_pts_(1pt)", "points": 1,
     "text": "Do you have trouble stepping up onto a curb?"},
    {"id": "s08", "column": "steadi_s08_pts_(1pt)", "points": 1,
     "text": "Do you often have to rush to the toilet?"},
    {"id": "s09", "column": "steadi_s09_pts_(1pt)", "points": 1,
     "text": "Have you lost some feeling in your feet?"},
    {"id": "s10", "column": "steadi_s10_pts_(1pt)", "points": 1,
     "text": "Do you take medicine that makes you feel light-headed or tired?"},
    {"id": "s11", "column": "steadi_s11_pts_(1pt)", "points": 1,
     "text": "Do you take medicine to help you sleep or improve your mood?"},
    {"id": "s12", "column": "steadi_s12_pts_(1pt)", "points": 1,
     "text": "Do you often feel sad or depressed?"},
]

STEADI_MAX_SCORE = sum(item["points"] for item in STEADI_ITEMS)  # 14


def score_from_items(answers: dict) -> int:
    """answers: {item_id: bool}. Returns total score 0-14."""
    total = 0
    for item in STEADI_ITEMS:
        if answers.get(item["id"], False):
            total += item["points"]
    return total


def flags_from_items(answers: dict) -> dict:
    """Build the 14 STEADI-derived model columns from item-by-item answers."""
    total = score_from_items(answers)
    flags = {
        "steadi_low_risk": int(total < STEADI_AT_RISK_CUTOFF),
        "steadi_at_risk": int(total >= STEADI_AT_RISK_CUTOFF),
    }
    for item in STEADI_ITEMS:
        flags[item["column"]] = item["points"] if answers.get(item["id"], False) else 0
    return flags


def flags_from_total(total_score: int) -> dict:
    """
    Back-calculate flags from a pre-computed total (Mode B).
    Individual item columns are set to proportional defaults (spec 5.2):
    we distribute points across items in spec order until the total is
    reached, which keeps the feature vector internally consistent without
    pretending to know which specific items were positive.

    Correctness note: this greedy pass never drops points. STEADI_ITEMS'
    point values are always {2, 2, 1, 1, ..., 1} (two 2-point items followed
    by ten 1-point items), so for any total_score in [0, 14] the ten 1-point
    items at the tail can always mop up whatever the two 2-point items
    couldn't consume — verified exhaustively for every total 0-14 in
    tests/test_core.py::test_flags_from_total_conserves_points.
    """
    flags = {
        "steadi_low_risk": int(total_score < STEADI_AT_RISK_CUTOFF),
        "steadi_at_risk": int(total_score >= STEADI_AT_RISK_CUTOFF),
    }
    remaining = total_score
    for item in STEADI_ITEMS:
        if remaining >= item["points"]:
            flags[item["column"]] = item["points"]
            remaining -= item["points"]
        else:
            flags[item["column"]] = 0
    return flags


def risk_level(total_score: int) -> str:
    return "At Risk" if total_score >= STEADI_AT_RISK_CUTOFF else "Low Risk"
