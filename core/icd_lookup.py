"""
ICD-9/ICD-10 -> comorbidity flag lookup, matching Designer Spec Section 4.3
(Mode 2: ICD Code Lookup).

Codes are matched by prefix, so a real code like "I11.0" matches the "I11"
entry below. This mirrors how the spec's example codes ("I10, I11, I12,
I13") are meant to be used as prefix families, not exact strings.
"""

ICD_TO_COMORBIDITY = {
    "dx_hypertension": {
        "icd10": ["I10", "I11", "I12", "I13"],
        "icd9": ["401", "402"],
    },
    "dx_diabetes": {
        "icd10": ["E10", "E11", "E13"],
        "icd9": ["250"],
    },
    "dx_heart_failure": {
        "icd10": ["I50"],
        "icd9": ["428"],
    },
    "dx_gait": {
        "icd10": ["R26"],
        "icd9": ["781.2", "781"],
    },
    "dx_anxiety": {
        "icd10": ["F40", "F41"],
        "icd9": ["300.0", "300"],
    },
    "dx_dementia": {
        "icd10": ["F00", "F01", "F02", "F03", "G30"],
        "icd9": ["290", "331.0", "331"],
    },
    "dx_stroke": {
        "icd10": ["I63", "I64", "Z86.73"],
        "icd9": ["433", "434"],
    },
    "dx_parkinsons": {
        "icd10": ["G20", "G21"],
        "icd9": ["332"],
    },
    "dx_arthritis": {
        "icd10": ["M05", "M06", "M15", "M16", "M17"],
        "icd9": ["714", "715"],
    },
    "dx_vision": {
        "icd10": ["H54", "H53"],
        "icd9": ["369", "368"],
    },
    "dx_hearing": {
        "icd10": ["H90", "H91"],
        "icd9": ["389"],
    },
    "fall_30d": {
        "icd10": ["W19", "Z87.39"],
        "icd9": ["E888.9", "E888"],
    },
}


def _normalize(code: str) -> str:
    return code.strip().upper().replace(" ", "")


def map_icd_codes(raw_input: str):
    """
    Parse a comma-separated string of ICD-9/10 codes and return:
      (matched_flags: set[str], unrecognized: list[str])
    """
    if not raw_input:
        return set(), []

    codes = [c for c in (c.strip() for c in raw_input.split(",")) if c]
    matched_flags = set()
    unrecognized = []

    for raw_code in codes:
        code = _normalize(raw_code)
        hit = False
        for flag, systems in ICD_TO_COMORBIDITY.items():
            for family in systems["icd10"] + systems["icd9"]:
                fam_norm = _normalize(family)
                if code.startswith(fam_norm):
                    matched_flags.add(flag)
                    hit = True
        if not hit:
            unrecognized.append(raw_code)

    return matched_flags, unrecognized
