"""
Minimal smoke tests. Run with:  pytest tests/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features import FEATURE_SET_B, REDUCED_CLINICAL_COLS, STEADI_DERIVED
from core.steadi import score_from_items, flags_from_items, flags_from_total, STEADI_ITEMS
from core.jhfrat import score_from_domains, flags_from_domains, DOMAIN_2_FALL_HISTORY
from core.icd_lookup import map_icd_codes


def test_feature_set_sizes():
    assert len(REDUCED_CLINICAL_COLS) == 16
    assert len(STEADI_DERIVED) == 14
    assert len(FEATURE_SET_B) == 30  # matches fall_risk_model_bundle.pkl


def test_bundle_feature_order_matches_app_schema():
    """The single contract that makes every prediction meaningful: the order
    the app builds its feature vector in must equal the order the model was
    fit on. If these ever drift, predictions are silently wrong."""
    from core.model import load_model

    bundle = load_model("B")  # load_model itself raises on mismatch
    assert list(bundle["feature_order"]) == FEATURE_SET_B
    assert bundle["decision_threshold"] is not None
    assert bundle["shap_background"].shape[1] == len(FEATURE_SET_B)


def test_unsupported_features_are_flagged():
    """A patient with hearing impairment must be flagged as outside the
    model's training support. Hearing impairment appears in 0.006% of the
    training data, so the scaler turns a 'yes' into a ~130 sigma value and the
    prediction collapses to ~0%. The dashboard must say the estimate is
    unreliable rather than present that 0% as a finding."""
    import numpy as np

    from core.model import get_estimator
    from core.model_support import audit_feature_support

    row = {c: 0.0 for c in FEATURE_SET_B}
    row["age_cat_65plus"] = 1.0
    row["dx_hearing"] = 1.0
    X = np.array([[row[c] for c in FEATURE_SET_B]])

    flagged = audit_feature_support(get_estimator("B"), X, FEATURE_SET_B)
    columns = [f.column for f in flagged]
    assert "dx_hearing" in columns
    hearing = next(f for f in flagged if f.column == "dx_hearing")
    assert hearing.lowers_risk, "hearing impairment should be flagged as lowering risk"
    assert hearing.train_prevalence < 0.01


def test_supported_features_are_not_flagged():
    """Common inputs must NOT be flagged, or the warning becomes noise."""
    import numpy as np

    from core.model import get_estimator
    from core.model_support import audit_feature_support

    row = {c: 0.0 for c in FEATURE_SET_B}
    row["age_cat_65plus"] = 1.0
    row["dx_hypertension"] = 1.0   # 33% of training rows
    row["steadi_s11_pts_(1pt)"] = 1.0  # 44% of training rows
    X = np.array([[row[c] for c in FEATURE_SET_B]])

    flagged = audit_feature_support(get_estimator("B"), X, FEATURE_SET_B)
    assert flagged == [], f"common features should not be flagged, got {flagged}"


def test_flags_from_total_conserves_points():
    """flags_from_total must never drop points while back-distributing a
    pre-computed STEADI total across the 12 item columns (0-14 inclusive)."""
    for total in range(0, 15):
        flags = flags_from_total(total)
        allocated = sum(flags[item["column"]] for item in STEADI_ITEMS)
        assert allocated == total, f"total={total} allocated={allocated}"


def test_steadi_all_yes_scores_max():
    answers = {item["id"]: True for item in STEADI_ITEMS}
    assert score_from_items(answers) == 14
    flags = flags_from_items(answers)
    assert flags["steadi_at_risk"] == 1
    assert flags["steadi_low_risk"] == 0


def test_steadi_all_no_scores_zero():
    answers = {item["id"]: False for item in STEADI_ITEMS}
    assert score_from_items(answers) == 0


def test_jhfrat_scoring():
    points = {DOMAIN_2_FALL_HISTORY["column"]: 5}
    assert score_from_domains(points) == 5
    flags = flags_from_domains(points)
    assert flags["jhfrat_low"] == 1   # 5 < 6 -> low tier
    assert flags["jhfrat_moderate"] == 0
    assert flags["jhfrat_high"] == 0


def test_icd_lookup_known_and_unknown_codes():
    matched, unrecognized = map_icd_codes("I10, E11, ZZZ99")
    assert "dx_hypertension" in matched
    assert "dx_diabetes" in matched
    assert "ZZZ99" in unrecognized


if __name__ == "__main__":
    test_feature_set_sizes()
    test_bundle_feature_order_matches_app_schema()
    test_flags_from_total_conserves_points()
    test_steadi_all_yes_scores_max()
    test_steadi_all_no_scores_zero()
    test_jhfrat_scoring()
    test_icd_lookup_known_and_unknown_codes()
    print("All smoke tests passed.")
