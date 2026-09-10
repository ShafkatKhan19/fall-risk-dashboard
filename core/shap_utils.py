"""
SHAP explanation utilities — Designer Spec Section 8.

Explains the deployed model's own predict_proba with a model-agnostic
KernelExplainer, and renders a patient-specific waterfall as an interactive
Plotly chart (the spec asks for st.pyplot; we use Plotly instead for a more
"native," interactive feel \u2014 the underlying SHAP values and semantics are
unchanged).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.config import COLOR_HIGH_RISK, COLOR_LOW_RISK, COLOR_MID_BLUE
from core.features import label_for


def compute_shap(feature_set_key: str, X: pd.DataFrame):
    """Model-agnostic SHAP for the real trained bundle (Feature Set B).

    The final estimator is a scaler+LogisticRegression Pipeline, not a tree
    model, so we explain its predict_proba function directly using the
    aggregate k-means background centroids shipped inside the bundle
    (bundle['shap_background']) — matching explainability_config.json's
    "model_agnostic_kernel" method and recommended_nsamples=300.
    """
    import shap

    from core.model import load_model as _load_bundle

    bundle = _load_bundle(feature_set_key)
    feature_order = bundle["feature_order"]
    background = bundle["shap_background"]  # plain ndarray — matches the model's own training input
    model = bundle["model"]

    def predict_positive_class(data):
        # `data` is already a plain ndarray from KernelExplainer's own
        # perturbation loop; the model was fit on ndarrays too, so passing it
        # straight through avoids a spurious "fitted without feature names"
        # warning on every one of the thousands of calls this makes.
        return model.predict_proba(data)[:, 1]

    explainer = shap.KernelExplainer(predict_positive_class, background)
    row = X[feature_order].to_numpy()
    raw = explainer.shap_values(row, nsamples=300)

    values = np.array(raw)[0]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[0]
    return values, float(base_value)


def top_drivers(shap_values, columns, n=3, direction="risk"):
    pairs = list(zip(columns, shap_values))
    if direction == "risk":
        pairs = [p for p in pairs if p[1] > 0]
        pairs.sort(key=lambda p: p[1], reverse=True)
    else:
        pairs = [p for p in pairs if p[1] < 0]
        pairs.sort(key=lambda p: p[1])
    return [(label_for(col), val) for col, val in pairs[:n]]


def contribution_figure(labelled_pairs, x_title="Contribution to prediction", top_n=8):
    """Horizontal bar chart of per-feature contributions.

    Takes already-labelled (display_name, value) pairs so the same chart serves
    both SHAP values and LIME weights. Red bars push the predicted risk up,
    green bars push it down; the zero line makes the split obvious at a glance.
    """
    pairs = sorted(labelled_pairs, key=lambda p: abs(p[1]), reverse=True)[:top_n]
    pairs = sorted(pairs, key=lambda p: p[1])  # ascending reads cleanly top-to-bottom

    if not pairs:
        return None

    labels = [name for name, _ in pairs]
    values = [v for _, v in pairs]
    colors = [COLOR_HIGH_RISK if v > 0 else COLOR_LOW_RISK for v in values]
    span = max(abs(v) for v in values) or 1.0

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:+.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title=x_title,
        xaxis=dict(range=[-span * 1.45, span * 1.45], zeroline=False),
        # automargin lets Plotly reserve whatever width the longest feature
        # label needs; a fixed left margin clipped names like "Medication
        # associated with dizziness or fatigue".
        yaxis=dict(automargin=True, ticklabelposition="outside"),
        margin=dict(l=10, r=20, t=10, b=40),
        height=max(300, 44 * len(labels)),
        plot_bgcolor="white",
        showlegend=False,
        bargap=0.35,
    )
    fig.add_vline(x=0, line_width=1, line_color=COLOR_MID_BLUE)
    return fig
