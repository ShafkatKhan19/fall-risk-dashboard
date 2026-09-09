"""
LIME explanation support for the real trained model bundle (Feature Set B).

Reference data and settings come straight from the training run's designer
delivery (see models/dashboard_v1/lime_reference_stats.json and
explainability_config.json): all reference rows are synthetic aggregate
samples, not real patient data, matching the privacy note in that file.
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from core.config import MODEL_DIR
from core.features import label_for

LIME_ASSET_DIR = os.path.join(MODEL_DIR, "dashboard_v1")


@st.cache_resource(show_spinner=False)
def load_lime_reference():
    path = os.path.join(LIME_ASSET_DIR, "lime_reference_stats.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def build_lime_explainer():
    from lime.lime_tabular import LimeTabularExplainer

    ref = load_lime_reference()
    training_data = np.array(ref["synthetic_reference_rows"], dtype=float)
    return LimeTabularExplainer(
        training_data=training_data,
        feature_names=ref["feature_order"],
        categorical_features=ref["categorical_feature_indices"],
        class_names=["No In-Hospital Fall", "In-Hospital Fall"],
        mode="classification",
        random_state=ref.get("random_state", 42),
        discretize_continuous=False,
    )


def _friendly_condition(condition: str, feature_order: list) -> str:
    """Swap the raw feature-name token in a LIME condition for its display label."""
    tokens = (
        condition.replace("<=", " ").replace(">=", " ").replace("<", " ").replace(">", " ").split()
    )
    raw_name = next((t for t in tokens if t in feature_order), None)
    if raw_name is None:
        return condition
    return condition.replace(raw_name, label_for(raw_name))


def explain_with_lime(feature_set_key: str, X_row: pd.DataFrame, num_features=10, num_samples=5000):
    """Return a list of (friendly_condition, weight) tuples for the positive class."""
    from core.model import load_model as _load_bundle

    bundle = _load_bundle(feature_set_key)
    feature_order = bundle["feature_order"]
    model = bundle["model"]

    explainer = build_lime_explainer()

    def predict_fn(data):
        # LIME already hands us a plain ndarray in feature_order; the model
        # was fit on ndarrays, so skip the DataFrame round-trip (it only adds
        # a spurious "fitted without feature names" sklearn warning).
        return model.predict_proba(data)

    row = X_row[feature_order].to_numpy()[0]
    exp = explainer.explain_instance(
        row, predict_fn, num_features=num_features, num_samples=num_samples, labels=(1,),
    )
    pairs = exp.as_list(label=1)
    return [(_friendly_condition(cond, feature_order), float(weight)) for cond, weight in pairs]
