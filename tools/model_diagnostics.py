"""
Model diagnostic report — run this to see exactly how the deployed model
responds to each input, and where it behaves implausibly.

    python tools/model_diagnostics.py

Written to hand to whoever trained the model. It reads only the shipped
bundle; it needs no MIMIC-IV data and changes nothing.

Background: the deployed pipeline is StandardScaler + LogisticRegression.
Standardising a binary indicator that was almost never positive in training
divides by a very small standard deviation, so a patient answering "yes"
becomes an extreme value the model never saw. Where that feature also carries
a negative coefficient, one "yes" can drive the predicted probability to zero.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features import FEATURE_SET_B, label_for  # noqa: E402
from core.model import get_estimator  # noqa: E402


def main():
    est = get_estimator("B")
    scaler = est.named_steps["scaler"]

    def prob(d):
        X = np.array([[float(d.get(c, 0)) for c in FEATURE_SET_B]])
        return est.predict_proba(X)[0][1]

    baseline = {c: 0 for c in FEATURE_SET_B}
    baseline["age_cat_65plus"] = 1
    baseline["steadi_low_risk"] = 1
    p0 = prob(baseline)

    print("=" * 100)
    print("SINGLE-FEATURE SENSITIVITY — deployed model")
    print("=" * 100)
    print(f"Baseline patient (65+, no risk factors recorded): {p0 * 100:.2f}%\n")
    print(f"{'factor turned ON':44s}{'train prev':>11s}{'z if ON':>9s}"
          f"{'new risk':>10s}{'change':>9s}")
    print("-" * 100)

    rows = []
    for i, c in enumerate(FEATURE_SET_B):
        if c.startswith("age_cat") or c in ("steadi_low_risk", "steadi_at_risk"):
            continue
        d = dict(baseline)
        d[c] = 2 if "max_2" in c else 1
        z = (d[c] - scaler.mean_[i]) / scaler.scale_[i]
        p = prob(d)
        rows.append((c, scaler.mean_[i], z, p, p - p0))

    implausible = []
    for c, mean, z, p, delta in sorted(rows, key=lambda r: r[4]):
        note = ""
        if delta < -0.01:
            note = "  <-- risk factor that LOWERS the estimate"
            implausible.append((c, mean, z, delta))
        print(f"{label_for(c)[:42]:44s}{mean * 100:10.3f}%{z:9.1f}"
              f"{p * 100:9.2f}%{delta * 100:+8.2f}%{note}")

    print()
    print("=" * 100)
    print("IMPLAUSIBLE DIRECTIONS")
    print("=" * 100)
    print("Every input on this form is there because it is a recognised fall risk")
    print("factor, so a negative effect is not clinically plausible for any of them.\n")
    for c, mean, z, delta in implausible:
        print(f"  {label_for(c):46s} present in {mean * 100:7.3f}% of training rows"
              f"   effect {delta * 100:+6.2f}%")

    print()
    print("=" * 100)
    print("SUGGESTED REMEDIES (all require retraining, with the original data)")
    print("=" * 100)
    print("  1. Drop near-zero-variance features before fitting")
    print("     (sklearn.feature_selection.VarianceThreshold). Anything present in")
    print("     well under 1% of rows cannot support a trustworthy coefficient.")
    print("  2. Do not standardise binary indicators. Leaving them as 0/1 removes")
    print("     the divide-by-tiny-sigma amplification entirely; only genuinely")
    print("     continuous features need scaling.")
    print("  3. If a feature must be kept despite thin support, constrain the sign")
    print("     (e.g. a monotonic or non-negative constraint) so a known risk")
    print("     factor cannot come out protective.")
    print("  4. Check the coding of these variables in the source data. Hearing")
    print("     impairment recorded in 0.006% of admissions reflects under-coding,")
    print("     not true prevalence, so the label itself is unreliable.")


if __name__ == "__main__":
    main()
