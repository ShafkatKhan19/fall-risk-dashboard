"""
Cohort / batch analytics — an "AI-specialized" addition beyond the base
spec. Scores every valid row from a raw-form CSV upload (matching
"Information Required from the Users.csv") using the real trained model
(Feature Set B — Clinical + STEADI) and summarizes risk distribution and the
features driving risk across the whole uploaded cohort, not just one patient.
"""
import numpy as np
import pandas as pd

from core.config import risk_tier_from_probability
from core.features import FEATURE_SET_B, label_for
from core.model import get_decision_threshold, get_estimator, load_model
from core.patient_form import full_row_to_features


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = [full_row_to_features(row.to_dict()) for _, row in df.iterrows()]
    return pd.DataFrame(
        [{col: r.get(col, 0) for col in FEATURE_SET_B} for r in rows],
        columns=FEATURE_SET_B,
    ).astype(float)


def score_cohort_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Score a raw-form CSV (Information Required from the Users.csv schema)
    with the real trained Feature-Set-B model."""
    model = get_estimator("B")
    threshold = get_decision_threshold("B")
    X = _feature_matrix(df)
    probs = model.predict_proba(X.to_numpy())[:, 1]

    out = df.copy()
    out["predicted_probability"] = probs
    out["risk_tier"] = [risk_tier_from_probability(p) for p in probs]
    if threshold is not None:
        out["above_model_threshold"] = probs >= threshold
    return out


def cohort_mean_abs_shap(scored_df: pd.DataFrame, sample_size: int = 200):
    """Average |SHAP value| per feature across a sample of the scored cohort,
    using the real bundle's model-agnostic KernelExplainer background."""
    import shap

    bundle = load_model("B")
    feature_order = bundle["feature_order"]
    background = bundle["shap_background"]  # plain ndarray, matches the model's training input
    model = bundle["model"]

    X = _feature_matrix(scored_df)
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)

    def predict_positive_class(data):
        return model.predict_proba(data)[:, 1]

    explainer = shap.KernelExplainer(predict_positive_class, background)
    raw = explainer.shap_values(X[feature_order].to_numpy(), nsamples=100)
    values = np.array(raw)

    mean_abs = np.abs(values).mean(axis=0)
    ranked = sorted(zip(feature_order, mean_abs), key=lambda p: p[1], reverse=True)
    return [(label_for(col), val) for col, val in ranked]


def risk_tier_counts(scored_df: pd.DataFrame) -> pd.DataFrame:
    counts = scored_df["risk_tier"].value_counts().reindex(
        ["High Risk", "Moderate Risk", "Low Risk"], fill_value=0
    )
    counts.index.name = "Risk Tier"
    return counts.reset_index(name="Patients")
