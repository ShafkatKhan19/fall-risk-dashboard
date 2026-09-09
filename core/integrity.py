"""
Integrity-checked file reads for pickle/joblib model artifacts.

pickle.load / joblib.load execute arbitrary Python during deserialization, so
anything that can overwrite a .pkl file under models/ can achieve code
execution on the server the moment the dashboard loads it. Every model file
is hash-pinned in core.config.MODEL_FILE_SHA256; this module refuses to hand
back file bytes that don't match the pinned hash.

This does NOT protect against a malicious file that was hashed and pinned
deliberately — it only detects unexpected changes (corruption, a bad copy, or
a swapped/tampered file) between when a model was reviewed and when it's
loaded.
"""
import hashlib
import os

from core.config import MODEL_DIR, MODEL_FILE_SHA256


class ModelIntegrityError(RuntimeError):
    pass


def read_verified_bytes(relative_path: str) -> bytes:
    """Read a file under MODEL_DIR and verify it against the pinned SHA-256.

    relative_path is the key used in MODEL_FILE_SHA256 (e.g.
    "model_A_clinical_RF.pkl" or "dashboard_v1/fall_risk_model_bundle.pkl"),
    using forward slashes regardless of OS.
    """
    expected = MODEL_FILE_SHA256.get(relative_path.replace(os.sep, "/"))
    if expected is None:
        raise ModelIntegrityError(
            f"No pinned hash for model file '{relative_path}'. Add it to "
            "core.config.MODEL_FILE_SHA256 after reviewing the file's "
            "provenance — refusing to load an unpinned model artifact."
        )

    full_path = os.path.join(MODEL_DIR, relative_path)
    with open(full_path, "rb") as f:
        data = f.read()

    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ModelIntegrityError(
            f"Integrity check failed for '{relative_path}': expected sha256 "
            f"{expected}, got {actual}. The file may be corrupted or was "
            "replaced unexpectedly. Refusing to deserialize it."
        )

    return data
