"""
Model loading, feature vector assembly, and prediction.

The dashboard ships exactly one model: the real MIMIC-IV-trained deployment
bundle at models/dashboard_v1/fall_risk_model_bundle.pkl (a dict with keys
model / feature_order / decision_threshold / shap_background / metadata —
see models/dashboard_v1/README_1.txt), saved with and loaded by joblib.

SECURITY NOTE: pickle/joblib deserialization can execute arbitrary code if the
file has been tampered with. The bundle is therefore hash-pinned in
core.config.MODEL_FILE_SHA256 and verified before it is ever deserialized
(see core/integrity.py). Only ever place model files here that you generated
or received directly from a trusted collaborator over a trusted channel —
never load a .pkl fetched from an untrusted or public location.
"""
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from core.config import risk_tier_from_probability
from core.features import FEATURE_SETS
from core.integrity import read_verified_bytes


@st.cache_resource(show_spinner=False)
def load_model(feature_set_key: str):
    spec = FEATURE_SETS[feature_set_key]
    bundle = joblib.load(io.BytesIO(read_verified_bytes(spec["model_file"])))

    # The feature vector this app builds and the order the model was fit on
    # must agree exactly, or every prediction is silently wrong. Fail loudly.
    if list(bundle["feature_order"]) != list(spec["columns"]):
        raise RuntimeError(
            "Model feature_order does not match the app's feature schema.\n"
            f"  bundle: {list(bundle['feature_order'])}\n"
            f"  app:    {list(spec['columns'])}"
        )
    return bundle


def get_estimator(feature_set_key: str):
    """Return the fitted predict_proba-capable estimator from the bundle."""
    return load_model(feature_set_key)["model"]


def get_decision_threshold(feature_set_key: str):
    """Return the validation-selected decision threshold shipped in the bundle."""
    return load_model(feature_set_key).get("decision_threshold")


def assemble_feature_vector(feature_set_key: str) -> pd.DataFrame:
    """
    Pull the relevant values out of st.session_state and assemble a single
    1-row DataFrame with columns in the exact order the model expects.
    """
    columns = FEATURE_SETS[feature_set_key]["columns"]
    values = {}
    values.update(st.session_state.get("demographics", {}))
    values.update(st.session_state.get("comorbidities", {}))
    values.update(st.session_state.get("steadi_flags", {}))

    row = {col: values.get(col, 0) for col in columns}
    return pd.DataFrame([row], columns=columns)


def predict_with_uncertainty(model, X: pd.DataFrame):
    """
    Point estimate = model.predict_proba(X)[0][1], matching spec 7.3.

    Uncertainty band: for a RandomForest, each tree casts its own vote: we
    take the spread across trees as a simple, honest confidence interval
    around the ensemble's averaged probability, rather than presenting a
    single number as if it were exact.
    """
    # The Feature-Set-B pipeline (StandardScaler + LogisticRegression) was
    # fit on a plain ndarray during training, so it has no recorded feature
    # names. Passing it a DataFrame still works but triggers a noisy sklearn
    # "fitted without feature names" UserWarning on every single call —
    # strip the column labels first since they add nothing here.
    X_for_predict = X.to_numpy() if hasattr(X, "to_numpy") else X
    proba = model.predict_proba(X_for_predict)[0][1]

    ci = None
    if hasattr(model, "estimators_"):
        # Individual trees were fit via the forest's internal bootstrap and
        # don't carry column names, so pass a plain array here to avoid a
        # harmless-but-noisy sklearn "fitted without feature names" warning.
        X_values = X.to_numpy() if hasattr(X, "to_numpy") else X
        tree_probs = np.array([
            est.predict_proba(X_values)[0][1] for est in model.estimators_
        ])
        low = float(np.percentile(tree_probs, 5))
        high = float(np.percentile(tree_probs, 95))
        ci = (max(0.0, low), min(1.0, high))

    return float(proba), ci


def run_prediction(feature_set_key: str):
    model = get_estimator(feature_set_key)
    X = assemble_feature_vector(feature_set_key)
    proba, ci = predict_with_uncertainty(model, X)
    tier = risk_tier_from_probability(proba)
    threshold = get_decision_threshold(feature_set_key)

    st.session_state["feature_vector"] = X
    st.session_state["prediction_probability"] = proba
    st.session_state["prediction_ci"] = ci
    st.session_state["prediction_risk_tier"] = tier
    st.session_state["prediction_decision_threshold"] = threshold
    st.session_state["selected_feature_set"] = feature_set_key
    return proba, ci, tier, X
