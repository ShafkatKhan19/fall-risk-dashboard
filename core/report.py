"""
PDF patient-summary report generation.
"""
from fpdf import FPDF, XPos, YPos

from core.config import DISCLAIMER_TEXT


def _ascii_safe(text: str) -> str:
    """fpdf2's built-in Helvetica font only supports latin-1; smart quotes,
    en/em dashes etc. from labels or free text would otherwise raise."""
    return str(text).encode("latin-1", errors="replace").decode("latin-1")


def build_pdf_report(
    patient_id: str,
    steadi_score,
    proba: float,
    tier: str,
    threshold,
    shap_up,
    shap_down,
    lime_up,
    lime_down,
    unsupported=(),
) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Fall Risk Dashboard - Patient Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.ln(2)

    def line(text):
        # multi_cell's own default leaves the cursor at the right edge of the
        # page rather than resetting to the left margin, which starves the
        # very next call of horizontal space — force it back explicitly.
        pdf.multi_cell(0, 7, _ascii_safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    line(f"Patient ID: {patient_id or '(not entered)'}")
    if steadi_score is not None:
        line(f"STEADI Score: {steadi_score} / 14")
    line(f"Predicted In-Hospital Fall Risk: {proba * 100:.1f}%")
    line(f"Risk Tier: {'NOT RELIABLE (see note below)' if unsupported else tier}")
    if threshold is not None:
        line(f"Model decision threshold: {threshold:.4f}")

    # The on-screen unreliability warning must travel with the PDF: a report
    # that leaves the app without it could be read as a real 0% finding.
    if unsupported:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        line("WARNING - THIS ESTIMATE IS NOT RELIABLE FOR THIS PATIENT")
        pdf.set_font("Helvetica", "", 10)
        line(
            "The model has almost no training data for the factors below, so answering "
            "'yes' to them pushes the estimate down instead of up. The percentage above "
            "is uninformative for this patient."
        )
        for f in unsupported:
            line(f"  - {f.label}: present in {f.train_prevalence * 100:.3f}% of training data")
        pdf.set_font("Helvetica", "", 11)

    if shap_up or shap_down:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Factors Contributing to This Risk (SHAP)", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for name, val in shap_up:
            line(f"  + {name} ({val:+.3f})")
        for name, val in shap_down:
            line(f"  - {name} ({val:+.3f})")

    if lime_up or lime_down:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "How Patient Factors Shaped the Prediction (LIME)", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for cond, w in lime_up:
            line(f"  + {cond} ({w:+.3f})")
        for cond, w in lime_down:
            line(f"  - {cond} ({w:+.3f})")

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0, 6, _ascii_safe(DISCLAIMER_TEXT + ". Not intended for clinical decision-making."),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    return bytes(pdf.output())
