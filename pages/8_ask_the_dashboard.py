import os

import streamlit as st

from core.assistant import answer_question, build_context_summary, llm_available

st.title("Ask the Dashboard")
st.caption(
    "Ask about the current patient's scores, prediction, or what's driving their "
    "risk. Grounded only in what's already on Tabs 1-5 \u2014 no outside data."
)

if llm_available():
    st.success("LLM-backed answers are enabled (ANTHROPIC_API_KEY detected).", icon=":material/auto_awesome:")
else:
    st.info(
        "Running in rule-based mode. Set the `ANTHROPIC_API_KEY` environment "
        "variable to enable natural-language answers via the Claude API.", icon=":material/info:"
    )

with st.expander("What the assistant currently knows about this patient"):
    st.text(build_context_summary())

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for role, text in st.session_state["chat_history"]:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("e.g. Why is this patient high risk?")
if prompt:
    st.session_state["chat_history"].append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = answer_question(prompt)
        st.markdown(reply)
    st.session_state["chat_history"].append(("assistant", reply))
