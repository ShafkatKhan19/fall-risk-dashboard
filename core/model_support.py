"""
Training-support diagnostics for the deployed model.

WHY THIS EXISTS
---------------
The deployed pipeline is StandardScaler + LogisticRegression. The scaler
divides each feature by its standard deviation in the training data. Several
inputs this dashboard collects were almost never coded positive in MIMIC-IV —
hearing impairment appears in 0.006% of training rows, "steadies self on
furniture" in 0.010%, vision impairment in 0.045%. For those, the standard
deviation is tiny, so a patient answering "yes" is scaled to an enormous
z-score (up to ~130 standard deviations). Multiplied by a negative
coefficient, a single such answer can subtract ~11 from the log-odds and drive
the predicted probability to essentially zero.

The practical consequence: ticking a real risk factor can make the model
report LESS risk, and an elderly patient with vision or hearing impairment —
the population this tool is aimed at — can come back as 0%.

This module does not attempt to repair the model. It cannot: fixing this
properly means retraining (dropping near-zero-variance features, or not
standardising binary indicators), which needs the original training data.
What it does is detect, per patient, when the estimate is resting on inputs
the model has essentially no evidence for, so the interface can say so instead
of presenting a fabricated number as a finding.
"""
from dataclasses import dataclass

import numpy as np

from core.features import label_for

# A feature answered "yes" is treated as outside training support when fewer
# than this fraction of training rows had it. 2% is a deliberately generous
# line: below it, a logistic coefficient is estimated from so few positive
# examples that its sign is not trustworthy in either direction.
RARE_PREVALENCE = 0.02

# Only warn when such a feature is also materially moving the log-odds. A
# tiny contribution is not worth interrupting the user over.
MATERIAL_LOGIT = 0.5


@dataclass(frozen=True)
class UnsupportedFeature:
    column: str
    label: str
    entered_value: float
    train_prevalence: float
    z_score: float
    logit_contribution: float

    @property
    def lowers_risk(self) -> bool:
        return self.logit_contribution < 0


def _pipeline_parts(estimator):
    """Return (scaler, linear_model) if this is the scaler+LR pipeline we ship."""
    try:
        return estimator.named_steps["scaler"], estimator.named_steps["model"]
    except (AttributeError, KeyError):
        return None, None


def audit_feature_support(estimator, X, feature_order):
    """Flag entered values the trained model has essentially no evidence for.

    Returns a list of UnsupportedFeature, worst (most risk-lowering) first.
    Returns [] for any estimator that is not the scaler+linear pipeline, so
    swapping in a retrained model can never break this page.
    """
    scaler, linear = _pipeline_parts(estimator)
    if scaler is None or linear is None or not hasattr(linear, "coef_"):
        return []

    values = np.asarray(X, dtype=float).reshape(-1)
    means, scales = scaler.mean_, scaler.scale_
    coefs = linear.coef_[0]

    flagged = []
    for i, column in enumerate(feature_order):
        value = values[i]
        if value <= 0:
            continue  # the patient does not have this; nothing to warn about
        if means[i] >= RARE_PREVALENCE:
            continue  # the model saw plenty of these
        z = (value - means[i]) / scales[i]
        contribution = float(z * coefs[i])
        if abs(contribution) < MATERIAL_LOGIT:
            continue
        flagged.append(
            UnsupportedFeature(
                column=column,
                label=label_for(column),
                entered_value=float(value),
                train_prevalence=float(means[i]),
                z_score=float(z),
                logit_contribution=contribution,
            )
        )

    return sorted(flagged, key=lambda f: f.logit_contribution)
