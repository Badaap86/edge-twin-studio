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

"""EdgeTwin Studio V123 - First Customer Intake Pack.

Purpose:
- Turn a prospect's first input into a guided intake, required uploads, missing questions and recommended pack.
- Keep customer work low while collecting enough data for a credible pilot/evidence pack.
"""

import data_quality_gate_pro as v122

VERSION = "123.0"
MODULE = "First Customer Intake Pack"


def _recommend_pack(problem: str, budget_eur: int, has_sample_data: bool, data_score: int, wants_custom: bool) -> Dict[str, Any]:
    text = (problem or "").lower()
    if wants_custom or any(k in text for k in ["custom", "multiple", "several", "hardware", "sensor", "bom"]):
        return {"pack": "Guided Custom Pack", "price_range_eur": [2500, 9500], "reason": "Customer needs guided custom scope."}
    if not has_sample_data or data_score < 65 or budget_eur < 1000:
        return {"pack": "Starter Diagnostic Pack", "price_range_eur": [499, 950], "reason": "Best for first data readiness and scope check."}
    if data_score >= 88 and budget_eur >= 3500:
        return {"pack": "Real-Data Evidence Pack", "price_range_eur": [3500, 7500], "reason": "Data appears strong enough for deeper evidence preparation."}
    return {"pack": "Professional Pilot Pack", "price_range_eur": [1500, 3500], "reason": "Best balance for a controlled paid pilot/evidence pack."}


def build_first_customer_intake_pack_snapshot(
    company: str = "Example Industrial Customer",
    contact_email: str = "customer@example.com",
    industry: str = "industrial maintenance",
    problem: str = "We have machine sensor data and want to know if it is ready for predictive maintenance.",
    desired_outcome: str = "pilot/evidence pack",
    budget_eur: int = 2500,
    urgency: str = "normal",
    has_sample_data: bool = True,
    wants_custom: bool = False,
    consent_mode: str = "customer_profile_only",
    dataset_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    dq = v122.build_data_quality_gate_pro_snapshot(
        dataset_df=dataset_df,
        use_case="predictive_maintenance",
        source_type="customer_sample" if has_sample_data else "no_sample_yet",
        consent_mode=consent_mode,
        target_pack="auto",
        customer_claim_request="pilot/evidence readiness only",
    )
    q = dq.get("quality", {})
    data_score = int(q.get("quality_score", 0))
    pack = _recommend_pack(problem, int(budget_eur), bool(has_sample_data), data_score, bool(wants_custom))
    required_uploads = [
        "1 sample CSV/Excel/log export with timestamps where possible",
        "short description of asset/machine/process",
        "known normal vs abnormal examples if available",
        "maintenance/failure history if available",
    ]
    missing_questions = []
    if not has_sample_data:
        missing_questions.append("Can you provide a small sample export before pricing a deeper evidence pack?")
    if not q.get("label_columns"):
        missing_questions.append("Do you have a status/label/failure column or maintenance history to identify abnormal events?")
    if not q.get("timestamp_columns"):
        missing_questions.append("Does the data contain timestamps or sampling frequency information?")
    if wants_custom:
        missing_questions.append("Which modules do you need: data quality, feature analysis, hardware/BOM, branding, extra use-case or reusable template?")
    blockers = list(dq.get("blockers", []))
    review_flags = list(dq.get("review_flags", []))
    if urgency in {"urgent", "high"} and pack["pack"] != "Starter Diagnostic Pack":
        review_flags.append("Urgent delivery should use a locked starter/professional scope to avoid custom creep.")
    decision = "CUSTOMER INTAKE READY" if not blockers else "INTAKE NEEDS DATA/CLAIM FIXES"
    return {
        "version": VERSION,
        "module": MODULE,
        "created_at": _now(),
        "company": company,
        "contact_email": contact_email,
        "industry": industry,
        "problem": problem,
        "desired_outcome": desired_outcome,
        "budget_eur": int(budget_eur),
        "urgency": urgency,
        "has_sample_data": bool(has_sample_data),
        "wants_custom": bool(wants_custom),
        "consent_mode": consent_mode,
        "recommended_pack": pack,
        "data_quality_snapshot": dq,
        "required_uploads": required_uploads,
        "missing_questions": missing_questions,
        "customer_next_step": "Upload sample data and confirm consent mode" if not blockers else "Fix blockers before paid evidence claims",
        "decision": decision,
        "blockers": blockers,
        "review_flags": review_flags,
        "safe_boundary": "Intake prepares a pack recommendation and evidence route; it does not promise production accuracy.",
    }


def create_first_customer_intake_pack_bundle(snapshot: Dict[str, Any]) -> bytes:
    lines = [
        f"Decision: {snapshot.get('decision')}",
        f"Company: {snapshot.get('company')}",
        f"Recommended pack: {snapshot.get('recommended_pack', {}).get('pack')}",
        f"Price range: {snapshot.get('recommended_pack', {}).get('price_range_eur')}",
        f"Next step: {snapshot.get('customer_next_step')}",
        f"Boundary: {snapshot.get('safe_boundary')}",
    ]
    questions = pd.DataFrame({"missing_question": snapshot.get("missing_questions", [])})
    uploads = pd.DataFrame({"required_upload": snapshot.get("required_uploads", [])})
    return _zip_bundle({
        "v123_snapshot.json": json.dumps(snapshot, indent=2, default=str),
        "v123_missing_questions.csv": questions.to_csv(index=False),
        "v123_required_uploads.csv": uploads.to_csv(index=False),
        "v123_customer_intake_summary.md": "\n".join(["# EdgeTwin Intake Summary", *[f"- {line}" for line in lines]]),
        "v123_summary.pdf": _pdf_bytes("EdgeTwin V123 First Customer Intake Pack", lines),
    })
