"""
Central configuration: color palette, risk thresholds, and app-wide constants.
Colors follow the Designer Specification (Section 12) exactly — this is the
one axis the spec pins down, so we don't deviate from it.
"""
from pathlib import Path

APP_VERSION = "0.1.0"
APP_TITLE = "In-Hospital Fall Risk Prediction Dashboard"

# ---------------------------------------------------------------------------
# Paths — anchored to this file, NOT to the current working directory.
# Streamlit Community Cloud runs the app with CWD set to the repository root,
# which is not necessarily the folder app.py lives in. CWD-relative paths
# ("models", "styles/theme.css") silently break there; these do not.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = str(PROJECT_ROOT / "models")
STYLES_DIR = PROJECT_ROOT / "styles"
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Color palette (Designer Spec, Section 12)
# ---------------------------------------------------------------------------
COLOR_DARK_BLUE = "#1F4E79"     # headings, sidebar
COLOR_MID_BLUE = "#2E75B6"      # accents, table headers
COLOR_HIGH_RISK = "#C00000"     # red
COLOR_MODERATE_RISK = "#ED7D31"  # orange
COLOR_LOW_RISK = "#375623"      # green
COLOR_BACKGROUND = "#F2F2F2"

# Extended tokens derived from the above for a more refined, "native" feel
# without contradicting the spec's chosen palette.
COLOR_INK = "#152B3D"           # near-black-blue for body text
COLOR_SURFACE = "#FFFFFF"
COLOR_SURFACE_MUTED = "#E8ECF1"
COLOR_BORDER = "#D6DEE6"
COLOR_HIGH_RISK_BG = "#FBEAEA"
COLOR_MODERATE_RISK_BG = "#FDF1E7"
COLOR_LOW_RISK_BG = "#EAF1EA"

# ---------------------------------------------------------------------------
# Risk tier thresholds (Section 7.4)
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "high": 0.60,       # >= 0.60
    "moderate": 0.35,   # 0.35 - 0.59
    # < 0.35 -> low
}

RISK_TIER_COLORS = {
    "High Risk": COLOR_HIGH_RISK,
    "Moderate Risk": COLOR_MODERATE_RISK,
    "Low Risk": COLOR_LOW_RISK,
}

RISK_TIER_ACTIONS = {
    "High Risk": "Immediate fall prevention protocol; escalate to nursing team",
    "Moderate Risk": "Enhanced monitoring; scheduled review within 24 hours",
    "Low Risk": "Standard fall precautions; routine monitoring",
}


def risk_tier_from_probability(p: float) -> str:
    if p >= RISK_THRESHOLDS["high"]:
        return "High Risk"
    if p >= RISK_THRESHOLDS["moderate"]:
        return "Moderate Risk"
    return "Low Risk"


# ---------------------------------------------------------------------------
# STEADI / JHFRAT thresholds
# ---------------------------------------------------------------------------
STEADI_AT_RISK_CUTOFF = 4       # >= 4 => at risk
JHFRAT_MODERATE_CUTOFF = 6      # 6-13 => moderate
JHFRAT_HIGH_CUTOFF = 13         # > 13 => high

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
DISCLAIMER_TEXT = "FOR RESEARCH USE ONLY — Not validated for clinical deployment"
# MODEL_DIR is defined at the top of this file, anchored to PROJECT_ROOT.

# ---------------------------------------------------------------------------
# Model file integrity pins
# ---------------------------------------------------------------------------
# pickle/joblib deserialization can execute arbitrary code, so every model
# artifact we load is hash-checked against this allowlist before it's
# unpickled. If you intentionally replace a model file, regenerate this dict
# (see models/print_model_hashes.py) — a mismatch here means either the file
# was swapped/corrupted, or this list is stale; investigate before trusting
# the new file's contents.
MODEL_FILE_SHA256 = {
    "dashboard_v1/fall_risk_model_bundle.pkl": "91eb24cdf8b8f9b06b29cfd5c45c7cc184527931e804f6a8c7ef383e8f43ed9e",
}
