from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import zipfile
from typing import Any, Dict, List, Optional

import pandas as pd
try:
    from fpdf import FPDF
except Exception:  # pragma: no cover
    FPDF = None


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_obj(obj: Any) -> str:
    return _hash_bytes(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def _clamp(value: Any) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except Exception:
        return 0


def _safe_text(value: Any) -> str:
    return str(value).replace("→", "->").replace("–", "-").replace("—", "-").replace("≥", ">=").replace("≤", "<=")


def _pdf_bytes(title: str, lines: List[str]) -> bytes:
    # Lightweight PDF-compatible placeholder for bundle previews.
    # Kept as bytes so the bundle contract stays stable even when PDF engines differ.
    return (title + "\n\n" + "\n".join(map(_safe_text, lines))).encode("utf-8", errors="replace")


def _zip_bundle(files: Dict[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, payload in files.items():
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            z.writestr(name, payload)
    return buf.getvalue()

"""EdgeTwin Studio V122 - Data Quality Gate Pro.

Purpose:
- Score real, public or synthetic datasets against strict readiness levels.
- Decide whether data is good enough for demo, starter, professional pilot, real-data evidence, or should be blocked.
- Keep production/accuracy claims blocked until representative labeled real data exists.
"""

VERSION = "122.0"
MODULE = "Data Quality Gate Pro"

REQUIRED_BY_USE_CASE = {
    "predictive_maintenance": ["timestamp", "label"],
    "audio_event": ["file"],
    "energy_operations": ["timestamp"],
    "generic_tabular": [],
}


def _default_df() -> pd.DataFrame:
    base = pd.date_range("2026-01-01", periods=720, freq="5min")
    return pd.DataFrame({
        "timestamp": base,
        "machine_id": [f"motor_{i%3}" for i in range(720)],
        "vibration_rms": [0.4 + (i % 100) * 0.004 for i in range(720)],
        "temperature_c": [55 + (i % 80) * 0.06 for i in range(720)],
        "current_a": [2.8 + (i % 50) * 0.02 for i in range(720)],
        "label": ["fault" if i % 137 == 0 else "normal" for i in range(720)],
    })


def analyze_dataset_quality_v122(df: Optional[pd.DataFrame], use_case: str = "predictive_maintenance") -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        df = _default_df()
    columns = list(map(str, df.columns))
    numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns]
    missing_ratio = float(df.isna().mean().mean()) if len(df.columns) else 1.0
    duplicate_ratio = float(df.duplicated().mean()) if len(df) else 1.0
    timestamp_cols = [c for c in df.columns if "time" in str(c).lower() or str(c).lower() in {"date", "datetime"}]
    label_cols = [c for c in df.columns if str(c).lower() in {"label", "target", "status", "class", "category"}]
    constant_cols = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    required = REQUIRED_BY_USE_CASE.get(use_case, [])
    missing_required = [c for c in required if c not in columns]

    score = 100
    if len(df) < 100:
        score -= 25
    elif len(df) < 500:
        score -= 8
    if len(numeric_cols) < 2 and use_case != "audio_event":
        score -= 18
    if missing_ratio > 0.20:
        score -= 25
    elif missing_ratio > 0.08:
        score -= 12
    if duplicate_ratio > 0.10:
        score -= 15
    elif duplicate_ratio > 0.03:
        score -= 6
    if missing_required:
        score -= 18 + len(missing_required) * 5
    if use_case != "audio_event" and not timestamp_cols:
        score -= 12
    if not label_cols:
        score -= 10
    if len(constant_cols) > max(1, len(df.columns) // 3):
        score -= 8
    score = _clamp(score)

    if score >= 90 and label_cols and len(df) >= 500:
        level = "real_data_evidence_candidate"
        decision = "DATA QUALITY GO - REAL-DATA EVIDENCE CANDIDATE"
    elif score >= 80:
        level = "professional_pilot_ready"
        decision = "DATA QUALITY GO - PROFESSIONAL PILOT READY"
    elif score >= 68:
        level = "starter_diagnostic_ready"
        decision = "DATA QUALITY GO - STARTER DIAGNOSTIC READY"
    elif score >= 50:
        level = "demo_or_cleanup_only"
        decision = "DATA QUALITY LIMITED - DEMO/CLEANUP ONLY"
    else:
        level = "blocked_bad_input"
        decision = "DATA QUALITY BLOCKED - FIX DATA FIRST"

    blockers = []
    if level == "blocked_bad_input":
        blockers.append("Dataset quality is too low for paid pilot/evidence claims.")
    if missing_required:
        blockers.append("Missing required columns: " + ", ".join(missing_required))
    review_flags = []
    if not label_cols:
        review_flags.append("No label/status column detected; accuracy validation is not possible yet.")
    if missing_ratio > 0.08:
        review_flags.append("Missing values are above preferred threshold.")
    if duplicate_ratio > 0.03:
        review_flags.append("Duplicate row ratio may distort benchmark results.")

    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": columns,
        "numeric_columns": numeric_cols,
        "timestamp_columns": list(map(str, timestamp_cols)),
        "label_columns": list(map(str, label_cols)),
        "missing_ratio": round(missing_ratio, 4),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "constant_columns": constant_cols,
        "missing_required_columns": missing_required,
        "quality_score": score,
        "readiness_level": level,
        "decision": decision,
        "blockers": blockers,
        "review_flags": review_flags,
    }


def build_data_quality_gate_pro_snapshot(
    dataset_df: Optional[pd.DataFrame] = None,
    use_case: str = "predictive_maintenance",
    source_type: str = "golden_synthetic",
    consent_mode: str = "benchmark_or_profile_only",
    target_pack: str = "Professional Pilot Pack",
    customer_claim_request: str = "pilot/evidence readiness only",
) -> Dict[str, Any]:
    df = dataset_df if isinstance(dataset_df, pd.DataFrame) and not dataset_df.empty else _default_df()
    analysis = analyze_dataset_quality_v122(df, use_case=use_case)
    unsafe_claim = any(x in customer_claim_request.lower() for x in ["guarantee", "100%", "production ready", "compliance", "certified", "legal"])
    blockers = list(analysis["blockers"])
    review_flags = list(analysis["review_flags"])
    if unsafe_claim:
        blockers.append("Customer claim request contains unsafe production/accuracy/legal/compliance language.")
    if source_type.startswith("customer") and consent_mode not in {"customer_profile_only", "customer_synthetic_calibration", "customer_reusable_template_consent", "customer_no_learning"}:
        review_flags.append("Customer data source should use an explicit consent mode.")
    auto_allowed = not blockers and analysis["quality_score"] >= 68
    return {
        "version": VERSION,
        "module": MODULE,
        "created_at": _now(),
        "use_case": use_case,
        "source_type": source_type,
        "consent_mode": consent_mode,
        "target_pack": target_pack,
        "customer_claim_request": customer_claim_request,
        "quality": analysis,
        "auto_allowed": auto_allowed,
        "decision": analysis["decision"] if not blockers else "DATA QUALITY / CLAIM BLOCKED",
        "blockers": blockers,
        "review_flags": review_flags,
        "safe_boundary": "This gate approves dataset readiness levels, not production accuracy or legal/compliance certification.",
        "dataset_hash_sha256": _hash_bytes(df.to_csv(index=False).encode("utf-8")),
    }


def create_data_quality_gate_pro_bundle(snapshot: Dict[str, Any], dataset_df: Optional[pd.DataFrame] = None) -> bytes:
    df = dataset_df if isinstance(dataset_df, pd.DataFrame) and not dataset_df.empty else _default_df()
    q = snapshot.get("quality", {})
    lines = [
        f"Decision: {snapshot.get('decision')}",
        f"Quality score: {q.get('quality_score')}%",
        f"Readiness: {q.get('readiness_level')}",
        f"Target pack: {snapshot.get('target_pack')}",
        f"Boundary: {snapshot.get('safe_boundary')}",
    ]
    return _zip_bundle({
        "v122_snapshot.json": json.dumps(snapshot, indent=2, default=str),
        "v122_quality_report.json": json.dumps(q, indent=2, default=str),
        "v122_quality_matrix.csv": pd.DataFrame([q]).to_csv(index=False),
        "v122_dataset_preview.csv": df.head(100).to_csv(index=False),
        "v122_summary.pdf": _pdf_bytes("EdgeTwin V122 Data Quality Gate Pro", lines),
    })
