# Fall Risk Dashboard

> **FOR RESEARCH USE ONLY — not validated for clinical deployment.**
> This dashboard is a research reproducibility tool. It has not been clinically
> validated, has no regulatory clearance, and must not be used to guide
> decisions about the care of real patients. See [Intended use](#intended-use).

Streamlit dashboard for the in-hospital fall risk study: single-form patient
intake, automatic CDC STEADI scoring, a trained in-hospital fall risk
prediction, and per-patient explanations of what drove that prediction.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

To run the tests: `pip install -r requirements-dev.txt && pytest tests/`

**Optional features.** `requirements.txt` deliberately omits two packages so
the hosted app fits inside Streamlit Community Cloud's ~1 GB memory limit:

| package | what it adds | cost |
|---|---|---|
| `lime` | the second explanation panel ("How patient factors shaped the prediction") | pulls matplotlib + scikit-image, ~81 MB |
| `anthropic` | LLM answers on "Ask the Dashboard" (needs `ANTHROPIC_API_KEY`) | ~12 MB |

The app detects both at runtime: without `lime` the SHAP panel simply renders
full-width, and without `anthropic` the assistant uses its built-in rule-based
answers. To get them locally: `pip install -r requirements-optional.txt`

## The model

One model ships with this app: the MIMIC-IV-trained **Reduced Clinical +
STEADI** bundle produced by `train_dashboard_reduced_model.py`.

| | |
|---|---|
| Feature set | Reduced Clinical (16) + STEADI-derived (14) = **30 features** |
| Estimator | `StandardScaler` + `LogisticRegression` pipeline |
| Imbalance method | NoSMOTE (class-weighted) |
| Decision threshold | 0.44960015 (selected on validation, not on the test set) |
| Artifact | `models/dashboard_v1/fall_risk_model_bundle.pkl` |

The bundle also carries the aggregate k-means SHAP background used for
explanations. `models/dashboard_v1/lime_reference_stats.json` holds
**synthetic** LIME reference rows generated from aggregate model-fit
distributions — no MIMIC-IV patient rows or identifiers are included in this
repository.

Earlier revisions also carried synthetic placeholder RandomForest models for a
"Clinical only" and a "Clinical + JHFRAT" feature set. They were unreachable
from the UI and have been removed: every unnecessary pickle is a
deserialization risk, and a synthetic placeholder sitting next to a real model
is a good way for someone to mistake one for the other.

JHFRAT is **not** provided in this open-access dashboard. Researchers wanting
to use JHFRAT should obtain authorization directly from Johns Hopkins
University.

## Project structure

```
app.py                        Entry point: page config, sidebar, navigation, footer
pages/
  1_home_and_patient_data.py  Welcome + the single intake form (or CSV upload)
  2_fall_risk_scores.py       STEADI score, derived from the intake form
  3_in_hospital_fall_risk.py  Prediction gauge + SHAP/LIME + PDF report
  7_cohort_analytics.py       Batch view over an uploaded CSV
  8_ask_the_dashboard.py      Conversational query over the current patient
core/
  config.py                   Paths, colors, thresholds, model hash pins
  features.py                 The exact 30-column feature contract
  patient_form.py             Raw intake form <-> model feature mapping
  steadi.py                   CDC STEADI scoring
  model.py                    Bundle loading + prediction
  integrity.py                SHA-256 verification before any unpickling
  shap_utils.py / lime_utils.py  Per-patient explanations
  cohort.py                   Batch scoring + cohort-level SHAP
  report.py                   PDF patient summary
  session.py / validation.py  Session schema, form + CSV validation
  assistant.py                "Ask the Dashboard" (LLM + rule-based fallback)
  icd_lookup.py / jhfrat.py   Retained helpers (not wired into the current UI)
models/dashboard_v1/          The deployed model bundle + explainability assets
data/sample_patients.csv      10 SYNTHETIC patients, valid for the CSV upload
styles/theme.css              Styling
```

## Data entry

Two paths, both producing the same result:

- **Manual** — one guided form (Sections A–D) covering demographics, clinical
  history, fall history/functional status, and medication/mood. STEADI is
  derived from Sections C/D automatically, so nothing is asked twice.
- **CSV upload** — one row per patient, columns exactly matching
  `data/sample_patients.csv` (the 27-column intake schema, *not* model feature
  names). After batch scoring you can load any individual patient from the
  cohort into the score and prediction pages.

Both paths funnel through `core.patient_form.load_row_into_session`, so the
two can't drift apart.

## Deploying to Streamlit Community Cloud

**Read [Intended use](#intended-use) first — a Community Cloud app is
internet-facing, and free-tier apps are public by default.**

1. Push this folder to a GitHub repo (this folder must be the repo root, so
   that `app.py` sits at the top level).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. **Create app** → pick the repo/branch → set **Main file path** to `app.py`.
   The form pre-fills `streamlit_app.py`; either works, because the repo ships
   a `streamlit_app.py` shim that runs `app.py`. If the field is red with
   "This file does not exist", the Deploy button stays disabled — that is the
   most common reason a deploy appears to hang without ever starting.
4. Under **Advanced settings**, choose Python **3.12** (recommended).

   This matters more than it looks. Community Cloud's default moved to Python
   **3.14**, and `scikit-learn==1.6.1` — the version this model was trained
   with — ships no 3.14 wheel. Pinning it there makes the installer compile
   scikit-learn from source, which needs Cython and a C/C++ toolchain and
   hangs the deploy indefinitely at "Processing dependencies".

   `requirements.txt` now guards against this with environment markers: on
   3.14+ it takes `scikit-learn>=1.7.2` instead, which has a prebuilt wheel.
   That is verified safe — predictions from this bundle are bit-identical
   under 1.6.1 and 1.9.1 for all 10 sample patients, because the pipeline only
   stores coefficient arrays. Choosing 3.12 simply keeps the exact training
   version, which is tidier.
5. Optional — to enable LLM answers on "Ask the Dashboard", add a secret in
   the app's **Settings → Secrets**:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   Without it the page falls back to rule-based answers and the app runs fine.
   Never commit this key to the repo.
6. If you do not want the app world-readable, set it to private and invite
   specific viewers under **Settings → Sharing**.

Notes:
- All asset paths are resolved relative to the project root (`core.config.PROJECT_ROOT`),
  not the working directory, so the app also works if it is deployed from a
  subdirectory.
- Community Cloud caps an app at roughly 1 GB of RAM, and this dependency tree
  is heavy (`shap` alone pulls numba + llvmlite, ~147 MB). That is why `lime`
  and `anthropic` are not in `requirements.txt` — see
  [Optional features](#quick-start). If the app still gets OOM-killed, the most
  likely culprit is cohort analytics: lower `sample_size` in
  `core.cohort.cohort_mean_abs_shap`, or reduce `nsamples` there.
- First build takes several minutes. If it never finishes, open **Manage app →
  logs** in Streamlit Cloud; a silent restart loop there usually means the
  container ran out of memory rather than a dependency actually failing.

## Security notes

- **Model integrity.** `pickle`/`joblib` deserialization can execute arbitrary
  code. The bundle is SHA-256 pinned in `core.config.MODEL_FILE_SHA256` and
  verified by `core/integrity.py` before it is ever unpickled. If you replace
  the model, regenerate that hash and review the file's provenance first.
- **Dependency pinning.** `requirements.txt` pins exact versions.
  `scikit-learn` in particular is pinned to **1.6.1**, the version the bundle
  was trained with — a mismatch makes scikit-learn warn that results may
  change.
- **Secrets.** `.streamlit/secrets.toml`, `.env`, and `*.key` are gitignored.
- **Uploads.** CSV uploads are capped (50 MB, 10,000 rows) and every field is
  validated before use.
- **Patient data.** `.gitignore` excludes everything in `data/` except the
  synthetic sample, so real data dropped there is never committed by accident.

## Intended use

This is a **research** tool, and the constraints below are not boilerplate:

- The model was trained and evaluated on MIMIC-IV retrospective data. It has
  **not** been prospectively validated, has no regulatory clearance, and has
  not been evaluated for safety or effectiveness in live clinical care.
- Its behavior on individual patients can be counterintuitive. For example,
  several coefficients (notably Parkinson's disease) are *protective* in this
  fit — plausibly a confounding artifact of the training data rather than a
  clinical truth. Do not read the outputs as clinical reasoning.
- Entering real, identifiable patient data into a hosted deployment sends that
  data to a third party. On Streamlit Community Cloud that is not covered by a
  HIPAA BAA. Use synthetic or properly de-identified data, or host it somewhere
  your institution has approved.
- Redistribution of MIMIC-IV-derived artifacts is governed by the PhysioNet
  credentialed-access data use agreement. Confirm with your IRB/PhysioNet
  before publishing the model bundle publicly.

## Known limitations

- STEADI Mode B (entering a pre-computed total instead of item answers)
  back-distributes points across items in a fixed order. Point totals are
  always conserved (`tests/test_core.py`), but the specific item attribution is
  an approximation, not a clinical back-calculation.
- `core/icd_lookup.py` and `core/jhfrat.py` are retained but not wired into the
  current UI.
