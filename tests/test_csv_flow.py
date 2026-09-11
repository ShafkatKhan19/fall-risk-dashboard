"""
Regression tests for the CSV batch flow.

These cover a bug that made the dashboard look broken: clicking "Run Batch
Prediction" appeared to do nothing, needing several presses, and the scored
cohort never reached the Fall Risk Scores / In-Hospital Fall Risk tabs.

Cause: the upload-handling block ran on every rerun (Streamlit's file_uploader
stays truthy for as long as a file sits in the widget, not only on the run
where it was picked), and that block reset cohort_results to None. So the
results were wiped by the very rerun the button click triggered, immediately
after being computed.
"""
import os
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.validation import validate_raw_form_csv  # noqa: E402

PAGE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pages", "1_home_and_patient_data.py",
)
SAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sample_patients.csv",
)


def _app_with_validated_upload():
    """A page primed as if a CSV had been uploaded and validated on an earlier
    run — which is exactly the state in which the bug used to strike."""
    df = pd.read_csv(SAMPLE)
    valid, errors = validate_raw_form_csv(df)
    assert not errors

    at = AppTest.from_file(PAGE, default_timeout=300)
    at.session_state["entry_view"] = "csv"
    at.session_state["cohort_upload_token"] = ("file-1", "sample_patients.csv", 1105)
    at.session_state["cohort_upload_name"] = "sample_patients.csv"
    at.session_state["cohort_row_count"] = len(df)
    at.session_state["cohort_valid_df"] = valid
    at.session_state["cohort_errors"] = []
    return at.run(), valid


def test_batch_prediction_works_on_a_single_click():
    at, valid = _app_with_validated_upload()
    buttons = [b for b in at.button if b.label and "Run Batch Prediction" in b.label]
    assert len(buttons) == 1

    buttons[0].click().run()

    scored = at.session_state["cohort_results"]
    assert scored is not None, "one click must be enough to score the cohort"
    assert len(scored) == len(valid)


def test_scored_cohort_survives_later_reruns():
    """The results must not be wiped by subsequent reruns while the file is
    still sitting in the uploader widget."""
    at, _ = _app_with_validated_upload()
    [b for b in at.button if b.label and "Run Batch Prediction" in b.label][0].click().run()
    assert at.session_state["cohort_results"] is not None

    at.run()
    at.run()

    assert at.session_state["cohort_results"] is not None, (
        "cohort_results was cleared by a later rerun — the upload block is "
        "re-processing a file it has already handled"
    )


def test_batch_prediction_populates_the_individual_patient_tabs():
    """After scoring, the other tabs must have a patient to display rather than
    telling the user to go and fill in the form."""
    at, valid = _app_with_validated_upload()
    [b for b in at.button if b.label and "Run Batch Prediction" in b.label][0].click().run()

    assert at.session_state["patient_id"] == str(valid.iloc[0]["study_id"])
    assert at.session_state["steadi_complete"] is True
    assert at.session_state["demographics_complete"] is True
    assert at.session_state["steadi_score"] is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
