import plotly.express as px
import streamlit as st

from core.config import COLOR_HIGH_RISK, COLOR_MODERATE_RISK, COLOR_LOW_RISK
from core.cohort import cohort_mean_abs_shap, risk_tier_counts

st.title("Cohort Analytics")
st.caption(
    "Batch view across every patient in a CSV uploaded on Home & Patient Data. "
    "Separate from the patient-specific explanations on In-Hospital Fall Risk "
    "\u2014 this is the one place population-level patterns are shown, and only "
    "for the cohort you uploaded."
)

scored = st.session_state.get("cohort_results")
if scored is None:
    st.info(
        "Upload a CSV on **Patient Data Entry** and click **Run Batch Prediction** "
        "to populate this view."
    )
    st.stop()

st.subheader(f"Cohort overview \u2014 {len(scored)} patients")

c1, c2, c3 = st.columns(3)
counts = risk_tier_counts(scored)
tier_lookup = dict(zip(counts["Risk Tier"], counts["Patients"]))
c1.metric("High Risk", int(tier_lookup.get("High Risk", 0)))
c2.metric("Moderate Risk", int(tier_lookup.get("Moderate Risk", 0)))
c3.metric("Low Risk", int(tier_lookup.get("Low Risk", 0)))

fig = px.bar(
    counts, x="Risk Tier", y="Patients", color="Risk Tier",
    color_discrete_map={
        "High Risk": COLOR_HIGH_RISK,
        "Moderate Risk": COLOR_MODERATE_RISK,
        "Low Risk": COLOR_LOW_RISK,
    },
)
fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("What's driving risk across this cohort")
with st.spinner("Computing cohort-level feature importance..."):
    ranked = cohort_mean_abs_shap(scored)
labels = [r[0] for r in ranked[:12]]
values = [r[1] for r in ranked[:12]]
fig2 = px.bar(x=values[::-1], y=labels[::-1], orientation="h",
              labels={"x": "Mean |SHAP value|", "y": ""})
fig2.update_traces(marker_color=COLOR_MODERATE_RISK)
fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Scored patients")
st.dataframe(
    scored[["predicted_probability", "risk_tier"] +
           [c for c in scored.columns if c not in ("predicted_probability", "risk_tier")]],
    use_container_width=True,
)

csv = scored.to_csv(index=False)
st.download_button("\u2B07\uFE0F Download scored cohort (CSV)", data=csv,
                    file_name="scored_cohort.csv", mime="text/csv")
