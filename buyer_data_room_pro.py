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

"""EdgeTwin Studio V124 - Buyer Data Room Pro.

Purpose:
- Assemble the tangible buyer-facing package after intake/quote/data-gate.
- Give the customer a clean room with scope, quote, evidence, limitations, safe claims and next steps.
"""

import first_customer_intake_pack as v123

VERSION = "124.0"
MODULE = "Buyer Data Room Pro"

SAFE_CLAIMS = [
    "EdgeTwin prepares pilot/evidence packs with data-quality checks, assumptions, limitations and next steps.",
    "EdgeTwin supports decision-making for controlled pilots; it does not guarantee production accuracy without validation.",
    "Real customer data requires consent mode and secure handling before reusable learning or benchmark calibration.",
]


def build_buyer_data_room_pro_snapshot(
    company: str = "Example Industrial Customer",
    contact_email: str = "customer@example.com",
    project_name: str = "EdgeTwin Pilot Evidence Pack",
    industry: str = "industrial maintenance",
    problem: str = "We have machine sensor data and want to know if it is ready for predictive maintenance.",
    budget_eur: int = 2500,
    has_sample_data: bool = True,
    wants_custom: bool = False,
    payment_status: str = "quote_ready",
    delivery_status: str = "intake_ready",
    dataset_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    intake = v123.build_first_customer_intake_pack_snapshot(
        company=company,
        contact_email=contact_email,
        industry=industry,
        problem=problem,
        budget_eur=int(budget_eur),
        has_sample_data=bool(has_sample_data),
        wants_custom=bool(wants_custom),
        dataset_df=dataset_df,
    )
    pack = intake.get("recommended_pack", {})
    data_quality = intake.get("data_quality_snapshot", {}).get("quality", {})
    evidence_score = _clamp((int(data_quality.get("quality_score", 0)) * 0.45) + 35 + (10 if has_sample_data else 0) + (5 if not wants_custom else 0))
    room_items = [
        {"item": "Quote / scope summary", "status": "ready", "customer_value": "Shows what is included and why the price is justified."},
        {"item": "Data quality result", "status": "ready" if has_sample_data else "waiting_for_sample", "customer_value": "Prevents garbage-in/garbage-out pilot claims."},
        {"item": "Trust and limitation notes", "status": "ready", "customer_value": "Explains what can and cannot be claimed safely."},
        {"item": "Delivery manifest", "status": delivery_status, "customer_value": "Shows what files will be delivered after unlock."},
        {"item": "Next-step plan", "status": "ready", "customer_value": "Gives a clear route from interest to pilot/evidence pack."},
    ]
    blockers = list(intake.get("blockers", []))
    review_flags = list(intake.get("review_flags", []))
    if payment_status not in {"quote_ready", "deposit_paid", "paid", "manual_paid_confirmed"}:
        review_flags.append("Payment is not ready/confirmed; keep final delivery locked.")
    if delivery_status not in {"intake_ready", "generating", "ready", "delivered"}:
        review_flags.append("Delivery status is not ready for customer download.")
    if evidence_score < 70:
        blockers.append("Evidence score too low for paid pilot/evidence room; collect better sample data first.")
    decision = "BUYER DATA ROOM READY" if not blockers else "BUYER DATA ROOM NEEDS FIXES"
    quote = {
        "recommended_pack": pack.get("pack"),
        "price_range_eur": pack.get("price_range_eur"),
        "deposit_recommendation_eur": max(250, int((pack.get("price_range_eur") or [1000])[0] * 0.4)) if isinstance(pack.get("price_range_eur"), list) else 500,
        "payment_status": payment_status,
        "final_delivery_unlock_condition": "paid/manual_paid_confirmed + delivery ready + no unsafe claims",
    }
    download_manifest = {
        "room_id": "BDR-" + _hash_obj({"company": company, "project": project_name})[:10].upper(),
        "files_expected": ["PDF report", "JSON snapshot", "CSV evidence tables", "Markdown summary", "secure download manifest"],
        "raw_customer_data_included": False,
        "expires_later_with_v104_signed_link": True,
    }
    return {
        "version": VERSION,
        "module": MODULE,
        "created_at": _now(),
        "company": company,
        "contact_email": contact_email,
        "project_name": project_name,
        "industry": industry,
        "problem": problem,
        "intake_snapshot": intake,
        "quote": quote,
        "evidence_score": evidence_score,
        "room_items": room_items,
        "safe_claims": SAFE_CLAIMS,
        "download_manifest": download_manifest,
        "decision": decision,
        "blockers": blockers,
        "review_flags": review_flags,
        "safe_boundary": "Buyer Data Room is an evidence/scope/delivery room, not a production guarantee or legal/compliance certification.",
    }


def build_v124_room_items_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(snapshot.get("room_items", []))


def create_buyer_data_room_pro_bundle(snapshot: Dict[str, Any]) -> bytes:
    lines = [
        f"Decision: {snapshot.get('decision')}",
        f"Company: {snapshot.get('company')}",
        f"Project: {snapshot.get('project_name')}",
        f"Evidence score: {snapshot.get('evidence_score')}%",
        f"Pack: {snapshot.get('quote', {}).get('recommended_pack')}",
        f"Price range: {snapshot.get('quote', {}).get('price_range_eur')}",
        f"Boundary: {snapshot.get('safe_boundary')}",
    ]
    return _zip_bundle({
        "v124_snapshot.json": json.dumps(snapshot, indent=2, default=str),
        "v124_room_items.csv": build_v124_room_items_table(snapshot).to_csv(index=False),
        "v124_safe_claims.md": "\n".join(["# Safe Claims", *[f"- {x}" for x in snapshot.get("safe_claims", [])]]),
        "v124_download_manifest.json": json.dumps(snapshot.get("download_manifest", {}), indent=2, default=str),
        "v124_customer_room_summary.md": "\n".join(["# EdgeTwin Buyer Data Room", *[f"- {line}" for line in lines]]),
        "v124_summary.pdf": _pdf_bytes("EdgeTwin V124 Buyer Data Room Pro", lines),
    })
