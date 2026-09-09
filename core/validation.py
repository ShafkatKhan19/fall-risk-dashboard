"""
Validation logic — Designer Spec Sections 4.2 and 4.5.
"""
import pandas as pd

from core.patient_form import AGE_MAX, AGE_MIN, RAW_FORM_COLUMNS, SEX_OPTIONS


def validate_patient_id(value: str):
    if not value:
        return False, "Must be 1-20 alphanumeric characters"
    if len(value) > 20 or not value.isalnum():
        return False, "Must be 1-20 alphanumeric characters"
    return True, None


# The raw-form CSV columns that must contain 0/1 (binary yes/no answers).
_RAW_BINARY_COLUMNS = [c for c in RAW_FORM_COLUMNS if c not in ("study_id", "age_years", "sex")]


def validate_raw_form_csv(df: pd.DataFrame):
    """
    Validate a CSV matching "Information Required from the Users.csv" — one
    row per patient, raw yes/no answers (not model feature columns). Returns
    (valid_rows_df, error_rows: list[dict]).
    """
    errors = []
    missing_cols = [c for c in RAW_FORM_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append({
            "row": "–",
            "field": ", ".join(missing_cols),
            "message": "Required column(s) missing from header",
        })
        return df.iloc[0:0], errors

    valid_mask = pd.Series(True, index=df.index)

    for idx, row in df.iterrows():
        row_num = idx + 2
        row_ok = True

        study_id = row.get("study_id")
        if pd.isna(study_id) or not str(study_id).strip():
            errors.append({"row": row_num, "field": "study_id", "message": "Missing value"})
            row_ok = False
        elif not str(study_id).strip().isalnum() or len(str(study_id).strip()) > 20:
            errors.append({
                "row": row_num, "field": "study_id",
                "message": "Must be 1-20 alphanumeric characters",
            })
            row_ok = False

        age = row.get("age_years")
        if pd.isna(age):
            errors.append({"row": row_num, "field": "age_years", "message": "Missing value"})
            row_ok = False
        else:
            try:
                age_int = int(age)
                if not (AGE_MIN <= age_int <= AGE_MAX):
                    raise ValueError
            except (TypeError, ValueError):
                errors.append({
                    "row": row_num, "field": "age_years",
                    "message": f"Must be an integer between {AGE_MIN} and {AGE_MAX}",
                })
                row_ok = False

        sex = row.get("sex")
        if pd.isna(sex) or str(sex).strip() not in SEX_OPTIONS:
            errors.append({
                "row": row_num, "field": "sex",
                "message": f"Must be one of: {', '.join(SEX_OPTIONS)}",
            })
            row_ok = False

        for col in _RAW_BINARY_COLUMNS:
            val = row.get(col)
            if pd.isna(val):
                errors.append({"row": row_num, "field": col, "message": "Missing value"})
                row_ok = False
                continue
            try:
                if float(val) < 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append({"row": row_num, "field": col, "message": "Must be a non-negative number (0/1, or a fall count)"})
                row_ok = False

        if not row_ok:
            valid_mask.at[idx] = False

    return df[valid_mask], errors
