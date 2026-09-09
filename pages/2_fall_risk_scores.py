import plotly.graph_objects as go
import streamlit as st

from core.config import COLOR_HIGH_RISK, COLOR_LOW_RISK
from core.steadi import STEADI_MAX_SCORE, STEADI_AT_RISK_CUTOFF, risk_level

st.title("Your Fall Risk Score")
st.caption("Scores are calculated from the information you provided.")

if st.session_state.get("steadi_score") is None:
    st.warning("Complete **Home & Patient Data** first — the STEADI score is calculated "
               "automatically from that form.")
    st.stop()

total = st.session_state["steadi_score"]
level = risk_level(total)
gauge_color = COLOR_LOW_RISK if level == "Low Risk" else COLOR_HIGH_RISK

with st.container(border=True):
    st.subheader("Fall Risk Score (STEADI)")
    st.caption("Based on CDC STEADI® tool")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total,
            number={"suffix": f" / {STEADI_MAX_SCORE}"},
            gauge={
                "axis": {"range": [0, STEADI_MAX_SCORE]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, STEADI_AT_RISK_CUTOFF], "color": "#EAF1EA"},
                    {"range": [STEADI_AT_RISK_CUTOFF, STEADI_MAX_SCORE], "color": "#FBEAEA"},
                ],
            },
        ))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("STEADI Total", f"{total} out of {STEADI_MAX_SCORE}")
        cls = "low" if level == "Low Risk" else "high"
        label = "AT RISK" if level != "Low Risk" else "LOW RISK"
        st.markdown(
            f'<div class="risk-badge {cls}"><span class="badge-dot {cls}"></span>{label}</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "STEADI and the STEADI logo are registered trademarks of the U.S. Department of "
        "Health and Human Services (HHS). © HHS 2024. Used with permission."
    )

st.info("Only STEADI is available in this open-access dashboard.")

with st.expander("STEADI item detail"):
    import pandas as pd
    from core.steadi import STEADI_ITEMS
    flags = st.session_state.get("steadi_flags", {})
    st.dataframe(
        pd.DataFrame({
            "Item": [it["text"] for it in STEADI_ITEMS],
            "Points": [flags.get(it["column"], 0) for it in STEADI_ITEMS],
        }),
        use_container_width=True, hide_index=True,
    )
