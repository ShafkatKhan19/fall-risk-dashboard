"""
In-Hospital Fall Risk Prediction Dashboard — entry point.

Run with:  streamlit run app.py
"""
import html

import streamlit as st

from core.config import APP_TITLE, APP_VERSION, DISCLAIMER_TEXT, STYLES_DIR
from core.session import init_session_state, progress_checklist

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()


def inject_theme():
    # STYLES_DIR is anchored to the project root, not the process CWD, so this
    # keeps working when Streamlit Cloud runs the app from the repo root.
    css = (STYLES_DIR / "theme.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_sidebar_chrome():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
              <div class="mark">:material/health_and_safety:</div>
              <div class="title">In-Hospital Fall Risk<br/>Prediction Study</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        checklist = progress_checklist()
        items_html = ""
        for label, done in checklist.items():
            cls = "done" if done else ""
            items_html += f'<li class="{cls}">{html.escape(str(label))}</li>'
        st.markdown(f'<ul class="stepper">{items_html}</ul>', unsafe_allow_html=True)

        st.markdown(f'<div class="sidebar-version">v{APP_VERSION}</div>', unsafe_allow_html=True)
        st.caption(f":material/info: {DISCLAIMER_TEXT}")


def render_footer():
    """A small, quiet, always-present footer note — same wording as
    core.config.DISCLAIMER_TEXT everywhere else, just understated here."""
    st.caption(f":material/info: {DISCLAIMER_TEXT}")


inject_theme()
render_sidebar_chrome()

pages = [
    st.Page("pages/1_home_and_patient_data.py", title="Home & Patient Data", icon=":material/home:", default=True),
    st.Page("pages/2_fall_risk_scores.py", title="Fall Risk Scores", icon=":material/monitoring:"),
    st.Page("pages/3_in_hospital_fall_risk.py", title="In-Hospital Fall Risk", icon=":material/emergency:"),
    st.Page("pages/7_cohort_analytics.py", title="Cohort Analytics", icon=":material/groups:"),
    st.Page("pages/8_ask_the_dashboard.py", title="Ask the Dashboard", icon=":material/chat:"),
]

nav = st.navigation(pages, position="sidebar")
nav.run()
render_footer()
