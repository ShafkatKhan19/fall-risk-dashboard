"""
Alias entry point.

Streamlit Community Cloud's deploy form pre-fills "Main file path" with
`streamlit_app.py`. The real entry point of this project is `app.py`, so a
deploy left on the default value fails with "This file does not exist" and the
Deploy button stays disabled. This shim makes either filename work.

Run either of these locally:
    streamlit run app.py
    streamlit run streamlit_app.py
"""
import runpy
from pathlib import Path

# run_name="__main__" so app.py executes exactly as it would if it were the
# main script. Streamlit resolves st.Page() paths relative to the main
# script's directory, which is this same folder either way.
runpy.run_path(str(Path(__file__).resolve().parent / "app.py"), run_name="__main__")
