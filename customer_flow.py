"""EdgeTwin Studio V117 One Perfect Customer Flow.

Purpose:
- Collapse the powerful V80-V116 stack into one calm customer journey.
- Make the product self-selling without exposing the customer to 30+ founder tabs.
- Guide a lead from problem -> pack -> proof -> quote/payment -> delivery/download.
- Keep dangerous production/accuracy/legal/compliance guarantees blocked.

Boundary:
- V117 is a customer-flow and conversion orchestration layer.
- It can prepare pages, CTAs, quote/payment handoff text and delivery-status guidance.
- It does not process payments, sign contracts, certify compliance, or guarantee production accuracy.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import zipfile
from typing import Any, Dict, List

from fpdf import FPDF

import self_selling as v116

VERSION = "117.0"
MODULE = "One Perfect Customer Flow"

SAFE_BOUNDARY = (
    "EdgeTwin provides pilot/evidence and decision-support packs. It does not guarantee production accuracy, "
    "legal/compliance certification, safety certification, direct production readiness, or replacement of human review."
)

PAYMENT_STATUSES = ["demo_only", "quote_ready", "payment_link_ready", "deposit_paid", "paid", "confirmed", "failed", "refunded", "disputed"]
UPLOAD_STATUSES = ["not_needed", "needed", "requested", "received", "validated", "blocked"]
DELIVERY_STATUSES = ["locked", "intake_unlocked", "generating", "ready", "delivered", "blocked"]


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_text(value: Any) -> str:
    return str(value or "").replace("\u2192", "->").replace("\u2013", "-").replace("\u2014", "-")


def _sha_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def _clamp(value: Any, default: int = 0) -> int:
    try:
        v = int(value)
    except Exception:
        v = default
    return max(0, min(100, v))


def _payment_gate(payment_status: str, delivery_status: str, upload_status: str) -> Dict[str, Any]:
    ps = str(payment_status or "quote_ready").lower().strip()
    ds = str(delivery_status or "locked").lower().strip()
    us = str(upload_status or "needed").lower().strip()
    blockers: List[str] = []
    review_flags: List[str] = []

    if ps in {"failed", "refunded", "disputed"}:
        blockers.append(f"Payment status is {ps}; access and delivery must remain locked.")
        unlock = "locked"
        next_action = "Resolve payment/refund/dispute before unlocking anything."
    elif ps in {"paid", "confirmed"}:
        unlock = "full_delivery_allowed" if ds in {"ready", "delivered"} else "paid_generation_allowed"
        next_action = "Generate or release the secure delivery bundle."
    elif ps == "deposit_paid":
        unlock = "intake_upload_allowed"
        next_action = "Unlock intake/upload and start the delivery checklist; final download waits for full payment or approval."
    elif ps == "payment_link_ready":
        unlock = "checkout_ready"
        next_action = "Show payment link/checkout CTA before intake or download."
    elif ps == "quote_ready":
        unlock = "quote_only"
        next_action = "Show quote/scope and ask customer to approve or pay deposit."
    else:
        unlock = "demo_only"
        next_action = "Show demo/discovery CTA; no private delivery unlock."

    if us == "blocked":
        blockers.append("Upload/data status is blocked; do not sell pilot/evidence claims until data issue is fixed.")
    elif us in {"needed", "requested"} and ps in {"paid", "confirmed"}:
        review_flags.append("Payment is safe, but customer data/intake is still missing before final delivery.")

    if ds == "blocked":
        blockers.append("Delivery status is blocked; secure download must remain unavailable.")

    return {
        "payment_status": ps,
        "upload_status": us,
        "delivery_status": ds,
        "unlock_decision": unlock,
        "next_action": next_action,
        "blockers": blockers,
        "review_flags": review_flags,
    }


def _data_gate(data_readiness_score: int, needs_real_data: bool) -> Dict[str, Any]:
    score = _clamp(data_readiness_score, 70)
    if score >= 85:
        level = "real_data_evidence_candidate" if needs_real_data else "pilot_ready_candidate"
        message = "Data signal is strong enough for controlled pilot/evidence positioning, with the usual no-production-guarantee boundary."
    elif score >= 68:
        level = "professional_pilot_candidate"
        message = "Data signal is usable for a Professional Pilot Pack, but output must include missing-data and limitation notes."
    elif score >= 45:
        level = "starter_only"
        message = "Data signal is enough for discovery/starter diagnostics, not strong evidence or accuracy claims."
    else:
        level = "bad_input_blocker"
        message = "Data signal is too weak for paid pilot evidence. Route to discovery/data-readiness improvement."
    return {"score": score, "level": level, "customer_message": message}


def _build_flow_steps(pack: Dict[str, Any], payment_gate: Dict[str, Any], data_gate: Dict[str, Any], conversion: Dict[str, Any]) -> List[Dict[str, Any]]:
    pack_name = pack.get("name", "EdgeTwin Pack")
    unlock = payment_gate.get("unlock_decision")
    steps = [
        {
            "step": 1,
            "title": "Problem fit",
            "customer_sees": "A plain-language explanation of the problem EdgeTwin solves.",
            "system_does": "Summarizes the customer problem and routes it to the correct pack path.",
            "cta": "Describe your use-case",
            "status": "ready",
        },
        {
            "step": 2,
            "title": "Recommended pack",
            "customer_sees": f"{pack_name} with included deliverables and boundaries.",
            "system_does": "Uses V116 conversion logic and pack rules to choose the safest offer.",
            "cta": pack.get("primary_cta", "Request pack"),
            "status": "ready",
        },
        {
            "step": 3,
            "title": "Data readiness gate",
            "customer_sees": data_gate.get("customer_message"),
            "system_does": "Checks whether data is demo, starter, pilot or evidence-ready.",
            "cta": "Upload sample or continue with demo dataset",
            "status": "ready" if data_gate.get("level") != "bad_input_blocker" else "review_required",
        },
        {
            "step": 4,
            "title": "Proof before price",
            "customer_sees": "Trust Ledger, Data Quality Gate, benchmark/synthetic reliability and claim boundaries.",
            "system_does": "Shows why the price is tied to reduced uncertainty, not just a static file.",
            "cta": "Review proof cards",
            "status": "ready",
        },
        {
            "step": 5,
            "title": "Quote/payment handoff",
            "customer_sees": "Price range, scope lock, what is included and what is not included.",
            "system_does": "Routes to quote, payment link, deposit, or manual review depending on risk.",
            "cta": "Approve quote / pay deposit / request review",
            "status": "ready" if unlock in {"checkout_ready", "quote_only", "intake_upload_allowed", "paid_generation_allowed", "full_delivery_allowed"} else "locked",
        },
        {
            "step": 6,
            "title": "Delivery workbench",
            "customer_sees": "Intake/upload status, generation status and secure download readiness.",
            "system_does": "Uses payment/upload/delivery states to decide what can be unlocked.",
            "cta": payment_gate.get("next_action"),
            "status": "ready" if unlock in {"intake_upload_allowed", "paid_generation_allowed", "full_delivery_allowed"} else "locked",
        },
        {
            "step": 7,
            "title": "Safe next step",
            "customer_sees": "A clear next step without production, accuracy, legal or compliance guarantees.",
            "system_does": "Keeps risky claims blocked and exports customer-safe copy/bundles.",
            "cta": "Download bundle / book review / continue to next pack",
            "status": "ready" if not conversion.get("blockers") else "blocked",
        },
    ]
    return steps


def _score_flow(conversion_score: int, data_score: int, payment_status: str, delivery_status: str, blockers: List[str], review_flags: List[str]) -> int:
    payment_score = {
        "paid": 100, "confirmed": 100, "deposit_paid": 86, "payment_link_ready": 78,
        "quote_ready": 70, "demo_only": 62, "failed": 15, "refunded": 10, "disputed": 5,
    }.get(str(payment_status).lower(), 60)
    delivery_score = {"delivered": 100, "ready": 94, "generating": 82, "intake_unlocked": 75, "locked": 62, "blocked": 10}.get(str(delivery_status).lower(), 60)
    raw = 0.35 * conversion_score + 0.25 * data_score + 0.20 * payment_score + 0.20 * delivery_score
    raw -= len(blockers) * 18
    raw -= min(14, len(review_flags) * 4)
    return int(max(0, min(100, round(raw))))


def build_one_perfect_customer_flow_snapshot(
    project_name: str = "EdgeTwin Project",
    customer_segment: str = "industrial maintenance / machine operations teams",
    customer_problem: str = "We have machine or sensor data but do not know if it is ready for a predictive maintenance pilot.",
    selected_pack_key: str = "auto",
    budget_eur: int = 2500,
    data_readiness_score: int = 78,
    trust_score: int = 90,
    demo_score: int = 88,
    urgency: str = "normal",
    wants_custom: bool = False,
    needs_real_data: bool = False,
    payment_status: str = "payment_link_ready",
    upload_status: str = "requested",
    delivery_status: str = "locked",
) -> Dict[str, Any]:
    """Build the single customer journey that hides internal complexity and sells safely."""
    conversion = v116.build_self_selling_snapshot(
        project_name=project_name,
        customer_segment=customer_segment,
        customer_problem=customer_problem,
        selected_pack_key=selected_pack_key,
        budget_eur=budget_eur,
        data_readiness_score=data_readiness_score,
        trust_score=trust_score,
        demo_score=demo_score,
        urgency=urgency,
        wants_custom=wants_custom,
        needs_real_data=needs_real_data,
        payment_ready=payment_status in {"payment_link_ready", "deposit_paid", "paid", "confirmed"},
        download_ready=delivery_status in {"ready", "delivered"},
        draft_claim_text="EdgeTwin prepares controlled pilot/evidence packs with data checks, assumptions, limitations and next steps.",
    )
    selected_pack = conversion.get("selected_pack", {})
    dg = _data_gate(data_readiness_score, needs_real_data)
    pg = _payment_gate(payment_status, delivery_status, upload_status)
    blockers = list(conversion.get("blockers", [])) + list(pg.get("blockers", []))
    review_flags = list(conversion.get("review_flags", [])) + list(pg.get("review_flags", []))
    if dg.get("level") == "bad_input_blocker":
        blockers.append("Data readiness is too weak for paid pilot/evidence flow.")

    flow_score = _score_flow(
        conversion_score=int(conversion.get("conversion_score", 0)),
        data_score=int(dg.get("score", 0)),
        payment_status=payment_status,
        delivery_status=delivery_status,
        blockers=blockers,
        review_flags=review_flags,
    )
    if blockers:
        decision = "CUSTOMER FLOW BLOCKED - FIX DATA/PAYMENT/CLAIMS"
    elif flow_score >= 94:
        decision = "ONE PERFECT CUSTOMER FLOW READY"
    elif flow_score >= 86:
        decision = "LAUNCHABLE CUSTOMER FLOW READY"
    elif flow_score >= 74:
        decision = "DEMO CUSTOMER FLOW READY - IMPROVE PAYMENT/DELIVERY"
    else:
        decision = "CUSTOMER FLOW NEEDS WORK"

    steps = _build_flow_steps(selected_pack, pg, dg, conversion)
    customer_page = {
        "hero": "From messy industrial data idea to pilot-ready evidence pack.",
        "subhero": "Choose a pack, understand your data readiness, see proof before price, then unlock intake or delivery through a safe payment/download flow.",
        "primary_cta": selected_pack.get("primary_cta", "Request pack"),
        "secondary_cta": "Start free discovery" if selected_pack.get("name") != "Free Discovery Check" else "Review paid route",
        "price_range_eur": selected_pack.get("price_range_eur"),
        "safe_boundary": SAFE_BOUNDARY,
    }
    snapshot = {
        "version": VERSION,
        "module": MODULE,
        "created_at": _now(),
        "project_name": str(project_name),
        "decision": decision,
        "customer_flow_score": flow_score,
        "selected_pack_key": conversion.get("selected_pack_key"),
        "selected_pack": selected_pack,
        "conversion_v116_score": conversion.get("conversion_score"),
        "data_gate": dg,
        "payment_delivery_gate": pg,
        "customer_page": customer_page,
        "flow_steps": steps,
        "proof_cards": conversion.get("proof_cards", []),
        "pricing_justification": conversion.get("pricing_justification", {}),
        "objection_responses": conversion.get("objection_responses", []),
        "safe_customer_claim": conversion.get("customer_copy", {}).get("safe_claim", SAFE_BOUNDARY),
        "blockers": blockers,
        "review_flags": review_flags,
        "recommended_next_actions": _next_actions(decision, pg, dg, selected_pack),
        "hidden_internal_complexity": [
            "Trust Ledger", "Policy Approval", "Pricing Assurance", "Custom Pack Builder", "Quote Builder",
            "Payment Adapter", "Secure Downloads", "Order Fulfillment", "Synthetic/Real Benchmarks", "Claim Safety",
        ],
        "content_hash": _sha_json({"project": project_name, "score": flow_score, "pack": conversion.get("selected_pack_key"), "steps": steps}),
        "important_boundary": SAFE_BOUNDARY,
    }
    return snapshot


def _next_actions(decision: str, payment_gate: Dict[str, Any], data_gate: Dict[str, Any], pack: Dict[str, Any]) -> List[str]:
    if "BLOCKED" in decision:
        return [
            "Fix blockers before showing this flow to a paying customer.",
            "If data quality is weak, route to discovery/starter instead of pilot evidence.",
            "If payment is failed/refunded/disputed, keep delivery locked.",
        ]
    actions = [
        "Use this as the main customer-facing route instead of exposing many technical tabs.",
        "Show the proof cards before price so the customer understands what they are buying.",
        "Keep production/accuracy/legal/compliance guarantees out of customer copy.",
    ]
    unlock = payment_gate.get("unlock_decision")
    if unlock in {"checkout_ready", "quote_only"}:
        actions.append("Connect the CTA to V98 quote and V102/V99 payment-unlock logic.")
    if data_gate.get("level") in {"starter_only", "professional_pilot_candidate"}:
        actions.append("Use limitation notes and missing-input checklist in the bundle.")
    if pack.get("name") in {"Real-Data Evidence Pack", "Premium Custom Pack"}:
        actions.append("Keep final scope under founder/policy review before delivery release.")
    return actions


def build_v117_flow_table(snapshot: Dict[str, Any]):
    try:
        import pandas as pd
        return pd.DataFrame(snapshot.get("flow_steps", []))
    except Exception:
        return []


def build_v117_objection_table(snapshot: Dict[str, Any]):
    try:
        import pandas as pd
        return pd.DataFrame(snapshot.get("objection_responses", []))
    except Exception:
        return []


def _pdf_bytes(snapshot: Dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "EdgeTwin V117 One Perfect Customer Flow", ln=True)
    pdf.set_font("Arial", "", 10)
    lines = [
        f"Project: {_safe_text(snapshot.get('project_name'))}",
        f"Decision: {_safe_text(snapshot.get('decision'))}",
        f"Customer flow score: {snapshot.get('customer_flow_score')}%",
        f"Selected pack: {_safe_text(snapshot.get('selected_pack', {}).get('name'))}",
        f"Price range EUR: {snapshot.get('selected_pack', {}).get('price_range_eur')}",
        "",
        "Customer hero:",
        _safe_text(snapshot.get("customer_page", {}).get("hero")),
        "",
        "Flow steps:",
    ]
    for step in snapshot.get("flow_steps", []):
        lines.append(f"{step.get('step')}. {step.get('title')} - {step.get('status')}: {step.get('cta')}")
    lines += ["", "Boundary:", _safe_text(snapshot.get("important_boundary")), "", "Next actions:"]
    for action in snapshot.get("recommended_next_actions", []):
        lines.append(f"- {action}")
    for line in lines:
        line = _safe_text(line)
        if not line:
            pdf.ln(3)
            continue
        for i in range(0, len(line), 100):
            pdf.multi_cell(185, 6, line[i:i+100])
    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1", errors="replace")
    return bytes(out)


def _markdown_page(snapshot: Dict[str, Any]) -> str:
    page = snapshot.get("customer_page", {})
    pack = snapshot.get("selected_pack", {})
    lines = [
        f"# {page.get('hero', 'EdgeTwin Customer Flow')}",
        "",
        page.get("subhero", ""),
        "",
        f"**Recommended pack:** {pack.get('name', 'n/a')}",
        f"**Price range:** EUR {pack.get('price_range_eur', ['TBD','TBD'])[0]} - {pack.get('price_range_eur', ['TBD','TBD'])[1]}",
        "",
        "## What happens next",
    ]
    for step in snapshot.get("flow_steps", []):
        lines.append(f"{step.get('step')}. **{step.get('title')}** - {step.get('customer_sees')}  ")
        lines.append(f"   CTA: {step.get('cta')}")
    lines += ["", "## Important boundary", snapshot.get("important_boundary", SAFE_BOUNDARY)]
    return "\n".join(_safe_text(x) for x in lines)


def create_one_perfect_customer_flow_bundle(snapshot: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("v117_one_perfect_customer_flow_snapshot.json", json.dumps(snapshot, indent=2, ensure_ascii=False))
        zf.writestr("v117_customer_page.md", _markdown_page(snapshot))
        zf.writestr("v117_safe_boundary.txt", snapshot.get("important_boundary", SAFE_BOUNDARY))
        zf.writestr("v117_flow_steps.json", json.dumps(snapshot.get("flow_steps", []), indent=2, ensure_ascii=False))
        zf.writestr("v117_recommended_actions.json", json.dumps(snapshot.get("recommended_next_actions", []), indent=2, ensure_ascii=False))
        try:
            import pandas as pd
            zf.writestr("v117_flow_steps.csv", pd.DataFrame(snapshot.get("flow_steps", [])).to_csv(index=False))
            zf.writestr("v117_objections.csv", pd.DataFrame(snapshot.get("objection_responses", [])).to_csv(index=False))
        except Exception:
            pass
        zf.writestr("v117_customer_flow_summary.pdf", _pdf_bytes(snapshot))
    return buf.getvalue()


if __name__ == "__main__":
    snap = build_one_perfect_customer_flow_snapshot()
    print(json.dumps({"decision": snap["decision"], "score": snap["customer_flow_score"], "pack": snap["selected_pack"].get("name")}, indent=2))
