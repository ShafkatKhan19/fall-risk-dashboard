import importlib.util
import logging
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

log = logging.getLogger(__name__)

# Checked once, without importing lime (which is slow and memory-hungry).
LIME_AVAILABLE = importlib.util.find_spec("lime") is not None

from core.config import (
    COLOR_HIGH_RISK, COLOR_LOW_RISK, COLOR_MODERATE_RISK, DISCLAIMER_TEXT, RISK_TIER_COLORS,
)
from core.model import get_estimator, run_prediction
from core.model_support import audit_feature_support
from core.report import build_pdf_report
from core.session import reset_patient
from core.shap_utils import compute_shap, contribution_figure, top_drivers

st.title("Estimated Risk of In-Hospital Fall")

demo_ok = st.session_state.get("demographics_complete", False)
comorb_ok = st.session_state.get("comorbidities_complete", False)
steadi_ok = st.session_state.get("steadi_complete", False)

if not (demo_ok and comorb_ok and steadi_ok):
    st.warning("Complete **Home & Patient Data** first — every field on that form feeds this prediction.")
    st.stop()

if st.session_state.get("prediction_probability") is None:
    with st.spinner("Running fall risk prediction..."):
        run_prediction("B")
    with st.spinner("Computing feature contributions..."):
        shap_values, base_value = compute_shap("B", st.session_state["feature_vector"])
        st.session_state["shap_values"] = shap_values
        st.session_state["shap_base_value"] = base_value

proba = st.session_state["prediction_probability"]
tier = st.session_state["prediction_risk_tier"]
fv = st.session_state["feature_vector"]
shap_values = st.session_state["shap_values"]

# ---------------------------------------------------------------------------
# Training-support check.
#
# Some inputs were almost never coded positive in the training data, so the
# scaler turns a "yes" into an extreme z-score and a single answer can collapse
# the estimate to ~0%. Telling the user the number is unreliable is the honest
# thing to do; showing a confident 0% for a patient who has several genuine
# risk factors is not.
# ---------------------------------------------------------------------------
unsupported = audit_feature_support(get_estimator("B"), fv.to_numpy(), list(fv.columns))
risk_lowering = [f for f in unsupported if f.lowers_risk]

if risk_lowering:
    names = ", ".join(f.label for f in risk_lowering)
    st.error(
        f"**This estimate is not reliable for this patient.** The model has almost no "
        f"training data for: **{names}**. Because of that, answering “yes” to these "
        f"pushes the estimate *down* rather than up, which is the opposite of the "
        f"clinical expectation. Treat the percentage below as uninformative for this "
        f"patient — see “Why this estimate is unreliable” for the detail.",
        icon=":material/warning:",
    )
    with st.expander("Why this estimate is unreliable"):
        st.write(
            "The model standardises each input against how often it appeared in the "
            "training data. An input that was almost never recorded gets divided by a "
            "very small standard deviation, so a single “yes” becomes an extreme value "
            "the model never saw during training."
        )
        st.dataframe(
            pd.DataFrame([{
                "Factor": f.label,
                "Present in training data": f"{f.train_prevalence * 100:.3f}% of patients",
                "Standard deviations from training average": f"{f.z_score:.1f}",
                "Effect on the estimate (log-odds)": f"{f.logit_contribution:+.2f}",
            } for f in risk_lowering]),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Fixing this requires retraining the model without these near-constant "
            "features (or without standardising binary indicators). It cannot be "
            "corrected in the dashboard, so the dashboard flags it instead."
        )

c1, c2 = st.columns([1, 1])
with c1:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": RISK_TIER_COLORS[tier]},
            "steps": [
                {"range": [0, 35], "color": "#EAF1EA"},
                {"range": [35, 60], "color": "#FDF1E7"},
                {"range": [60, 100], "color": "#FBEAEA"},
            ],
        },
        title={"text": "Predicted risk of in-hospital fall"},
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This prediction is based on the patient information.")

with c2, st.container(border=True):
    st.subheader("Risk level")
    if risk_lowering:
        # Never show a confident tier next to an estimate we have just told the
        # user is unreliable — "LOW RISK" is precisely the dangerous misread.
        cls, label = "moderate", "NOT RELIABLE"
    else:
        cls = {"Low Risk": "low", "Moderate Risk": "moderate", "High Risk": "high"}[tier]
        label = ("HIGH RISK" if tier == "High Risk"
                 else "MODERATE RISK" if tier == "Moderate Risk" else "LOW RISK")
    st.markdown(
        f'<div class="risk-badge {cls}"><span class="badge-dot {cls}"></span>{label}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The model cannot produce a trustworthy risk tier for this patient — "
        "see the note above."
        if risk_lowering else
        {
            "High Risk": "Higher predicted risk compared to other patients in this study.",
            "Moderate Risk": "Moderate predicted risk compared to other patients in this study.",
            "Low Risk": "Lower predicted risk compared to other patients in this study.",
        }[tier]
    )

st.divider()
st.header("Factors Contributing to This Risk")

up = top_drivers(shap_values, list(fv.columns), n=5, direction="risk")
down = top_drivers(shap_values, list(fv.columns), n=5, direction="protective")

def _render_shap_panel():
    st.markdown("**Factors contributing to this risk (SHAP)**")
    fig = contribution_figure(up + down, x_title="Contribution to prediction")
    if fig is None:
        st.info("No SHAP contributions to show for this patient.")
        return
    st.plotly_chart(fig, use_container_width=True, key="shap_chart",
                    config={"displayModeBar": False})
    with st.expander("Show these as a list"):
        for name, val in up:
            st.write(f":material/arrow_upward: {name}  **{val:+.3f}**")
        for name, val in down:
            st.write(f":material/arrow_downward: {name}  **{val:+.3f}**")


# LIME is an optional dependency: it is only 2 MB itself but pulls in
# matplotlib + scikit-image (~81 MB) that it needs solely for image
# explanations, which this dashboard never uses. That weight lands at runtime
# exactly when memory is tightest, so the hosted build omits it. When it is
# absent we show the SHAP panel full-width rather than a broken second column.
lime_up, lime_down = [], []
if LIME_AVAILABLE:
    c3, c4 = st.columns(2)
    with c3:
        _render_shap_panel()
    with c4:
        st.markdown("**How patient factors shaped the prediction (LIME)**")
        with st.spinner("Computing LIME explanation..."):
            lime_pairs = None
            try:
                from core.lime_utils import explain_with_lime
                lime_pairs = explain_with_lime("B", fv, num_features=8, num_samples=3000)
            except Exception:
                log.exception("LIME explanation failed")
                st.info("The second explanation method is unavailable for this patient.")
        if lime_pairs:
            lime_up = [(c, w) for c, w in lime_pairs if w > 0][:5]
            lime_down = [(c, w) for c, w in lime_pairs if w < 0][:5]
            lime_fig = contribution_figure(lime_up + lime_down, x_title="Impact on prediction")
            if lime_fig is not None:
                st.plotly_chart(lime_fig, use_container_width=True, key="lime_chart",
                                config={"displayModeBar": False})
                with st.expander("Show these as a list"):
                    for cond, w in lime_up:
                        st.write(f":material/arrow_upward: {cond}  **{w:+.3f}**")
                    for cond, w in lime_down:
                        st.write(f":material/arrow_downward: {cond}  **{w:+.3f}**")
            if not lime_up and not lime_down:
                st.info("No LIME contributions to show for this patient.")

    st.caption(
        "SHAP and LIME may rank factors differently — this is expected, since they use "
        "different local explanation techniques on the same prediction."
    )
else:
    _render_shap_panel()

st.success(
    ":material/check_circle: In-hospital fall risk prediction is derived from a trained "
    "classifier of a machine learning model."
)

st.caption(f":material/info: {DISCLAIMER_TEXT}")

b1, b2 = st.columns(2)
with b1:
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", st.session_state.get("patient_id") or "patient")[:40]
    pdf_bytes = build_pdf_report(
        patient_id=st.session_state.get("patient_id", ""),
        steadi_score=st.session_state.get("steadi_score"),
        proba=proba,
        tier=tier,
        threshold=st.session_state.get("prediction_decision_threshold"),
        shap_up=up,
        shap_down=down,
        lime_up=lime_up,
        lime_down=lime_down,
        unsupported=risk_lowering,
    )
    st.download_button(
        ":material/description: Download report (PDF)", data=pdf_bytes,
        file_name=f"fall_risk_report_{safe_id}.pdf", mime="application/pdf",
        use_container_width=True,
    )
with b2:
    if st.button(":material/restart_alt: Start a new assessment", use_container_width=True):
        reset_patient()
        st.session_state["entry_view"] = "welcome"
        st.rerun()

with st.expander("Feature vector submitted to the model"):
    st.dataframe(fv.T.rename(columns={0: "value"}), use_container_width=True)
