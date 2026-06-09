"""EdgeTwin Studio Product Release Candidate.

Purpose:
- Combine the customer-facing landing portal, guided custom builder, conversion copy,
  payment/delivery readiness and data/synthetic trust signals into one release candidate.
- Make the product easier to test with real prospects without exposing internal cockpit complexity.
- Keep boundaries honest: no production accuracy, legal/compliance certification or safety guarantees.

Boundary:
- Founder-led pack product release candidate, not full SaaS.
- Payment providers, private delivery endpoints and hosted deployment can be connected later.
- Live customer data still requires consent, privacy review and secure upload/storage policy.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import zipfile
from typing import Any, Dict, List

import pandas as pd
from fpdf import FPDF

import customer_facing_landing_portal as v119

VERSION = "120.0"
MODULE = "Pack Product Release Candidate"

SAFE_BOUNDARY = (
    "EdgeTwin is positioned as a founder-led pilot/evidence pack product. It supports data-readiness, "
    "custom pack configuration, quote/payment handoff and secure delivery preparation. It does not guarantee "
    "production accuracy, legal/compliance certification, safety certification or replacement of human review."
)

RC_READINESS_AREAS = [
    {
        "area": "customer_front_door",
        "label": "Customer-facing landing portal",
        "why_it_matters": "The customer understands the offer without seeing the internal technical cockpit.",
        "minimum": "Clear headline, pack cards, custom route, proof cards and safe CTA.",
    },
    {
        "area": "guided_custom",
        "label": "Guided custom pack builder",
        "why_it_matters": "The customer can configure custom needs while EdgeTwin controls scope and risk.",
        "minimum": "Module-based scope, price range, questions, review flags and blocker rules.",
    },
    {
        "area": "conversion",
        "label": "Self-selling conversion copy",
        "why_it_matters": "The product explains value, evidence and price logic before asking for payment.",
        "minimum": "Pain, proof, pack recommendation, objections and CTA.",
    },
    {
        "area": "claims",
        "label": "Claim safety policy",
        "why_it_matters": "Prevents unsafe promises like production-ready, 100% accuracy or compliance certified.",
        "minimum": "Unsafe claims blocked and safe rewrite available.",
    },
    {
        "area": "data",
        "label": "Synthetic/real data trust path",
        "why_it_matters": "Demo, benchmark and regression data must be measurable and not fake-looking random data.",
        "minimum": "Golden synthetic, benchmark harness, import wizard and calibration loop signals.",
    },
    {
        "area": "quote_payment_delivery",
        "label": "Quote, payment unlock and delivery handoff",
        "why_it_matters": "The customer journey must continue from interest to quote, paid/deposit status and delivery.",
        "minimum": "Quote handoff, payment-provider adapter contract, secure download/delivery status.",
    },
    {
        "area": "operator_control",
        "label": "Founder/operator control",
        "why_it_matters": "Standard work can be automated while risky exceptions stay under founder approval.",
        "minimum": "Review flags, blockers and no automatic legal/production commitments.",
    },
]

LAUNCH_CHECKLIST = [
    "Use the landing portal as the default customer front door.",
    "Offer only three visible routes first: Starter, Professional and Guided Custom.",
    "Keep Real-Data Evidence behind data-readiness and consent checks.",
    "Use payment links/manual confirmation before connecting live Stripe/Paddle webhooks.",
    "Do not expose raw storage paths or internal policy state to customers.",
    "Use secure/signed download logic for delivery bundles when hosted publicly.",
    "Collect the first prospect feedback before adding more engine features.",
]


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _hash_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _clamp_score(value: int | float) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except Exception:
        return 0


def _area_scores(portal: Dict[str, Any], data_score: int, payment_ready: bool, hosted_ready: bool) -> Dict[str, int]:
    portal_score = _clamp_score(portal.get("portal_score", 0))
    source = portal.get("source_scores", {}) if isinstance(portal.get("source_scores"), dict) else {}
    conversion_score = _clamp_score(source.get("v116_conversion", 0))
    flow_score = _clamp_score(source.get("v117_flow", 0))
    custom_score = _clamp_score(source.get("v118_custom_raw", source.get("v118_custom_portal_signal", 0)))

    quote_delivery = 72
    if payment_ready:
        quote_delivery += 10
    if hosted_ready:
        quote_delivery += 8
    quote_delivery = _clamp_score(quote_delivery)

    claims_score = 96
    if portal.get("blockers"):
        claims_score -= 16
    if len(portal.get("review_flags", [])) > 3:
        claims_score -= 5

    return {
        "customer_front_door": portal_score,
        "guided_custom": max(75, custom_score),
        "conversion": max(conversion_score, flow_score, 75),
        "claims": _clamp_score(claims_score),
        "data": _clamp_score(data_score),
        "quote_payment_delivery": quote_delivery,
        "operator_control": 94,
    }


def _decision(score: int, blockers: List[str], hosted_ready: bool, payment_ready: bool) -> str:
    if blockers:
        return "RC BLOCKED - FIX CLAIM/DATA/PAYMENT RISKS"
    if score >= 92 and hosted_ready and payment_ready:
        return "PUBLIC/PRIVATE PACK PORTAL RC READY"
    if score >= 88:
        return "FOUNDER-LED CUSTOMER TEST RC READY"
    if score >= 80:
        return "PRIVATE DEMO RC - NEEDS POLISH"
    return "INTERNAL REVIEW ONLY"


def build_product_release_candidate_snapshot(
    project_name: str = "EdgeTwin Studio",
    company: str = "Customer",
    industry: str = "Industrial / maintenance",
    customer_segment: str = "industrial maintenance and machine operations teams",
    customer_problem: str = "We have machine or sensor data but do not know if it is ready for a predictive maintenance pilot.",
    desired_outcome: str = "Pilot/evidence pack",
    budget_eur: int = 2500,
    data_readiness_score: int = 82,
    trust_score: int = 92,
    synthetic_reliability_score: int = 92,
    wants_custom: bool = True,
    has_real_data: bool = False,
    payment_ready: bool = False,
    hosted_ready: bool = False,
    public_mode: str = "private_demo_link",
) -> Dict[str, Any]:
    """Build a complete release-candidate view from the customer-facing stack."""
    payment_mode = "payment_link_ready" if payment_ready else "manual_quote_ready"
    portal = v119.build_customer_facing_landing_portal_snapshot(
        project_name=project_name,
        customer_segment=customer_segment,
        company=company,
        industry=industry,
        customer_problem=customer_problem,
        desired_outcome=desired_outcome,
        budget_eur=int(budget_eur),
        data_readiness_score=int(data_readiness_score),
        trust_score=int(trust_score),
        demo_score=int(round((int(synthetic_reliability_score) + int(trust_score)) / 2)),
        wants_custom=bool(wants_custom),
        has_real_data=bool(has_real_data),
        public_mode=public_mode,
        payment_mode=payment_mode,
    )

    data_score = _clamp_score(round((int(data_readiness_score) + int(synthetic_reliability_score)) / 2))
    area_scores = _area_scores(portal, data_score=data_score, payment_ready=payment_ready, hosted_ready=hosted_ready)
    weighted = {
        "customer_front_door": 0.18,
        "guided_custom": 0.14,
        "conversion": 0.15,
        "claims": 0.16,
        "data": 0.17,
        "quote_payment_delivery": 0.12,
        "operator_control": 0.08,
    }
    rc_score = _clamp_score(sum(area_scores[k] * weighted[k] for k in weighted))

    blockers: List[str] = list(portal.get("blockers", []) or [])
    if int(data_readiness_score) < 55:
        blockers.append("Data readiness is too low for paid pilot/evidence claims; offer discovery/data-cleanup route first.")
    if int(synthetic_reliability_score) < 70:
        blockers.append("Synthetic reliability is too low; recalibrate golden synthetic datasets before demo/regression use.")

    review_flags: List[str] = list(portal.get("review_flags", []) or [])
    if not hosted_ready:
        review_flags.append("Public/private hosting is not marked ready; use local/private demo or deploy V101 hosting kit first.")
    if not payment_ready:
        review_flags.append("Live payment handoff is not marked ready; use manual quote/payment link before automatic unlock.")
    if has_real_data:
        review_flags.append("Real customer data requires consent mode, secure storage/upload policy and no raw-data reuse without approval.")
    if wants_custom:
        review_flags.append("Guided custom route is enabled; final scope/pricing exceptions remain under policy/founder review.")

    rc_status = _decision(rc_score, blockers, hosted_ready=hosted_ready, payment_ready=payment_ready)

    customer_steps = [
        {"step": 1, "name": "Open landing portal", "customer_view": "Problem, proof and pack choices", "system_layer": "Landing portal"},
        {"step": 2, "name": "Choose standard or guided custom", "customer_view": "Pack cards or module builder", "system_layer": "Guided custom builder + landing portal"},
        {"step": 3, "name": "See scope and value", "customer_view": "Deliverables, limits, proof cards and price logic", "system_layer": "V116/V117"},
        {"step": 4, "name": "Quote/payment handoff", "customer_view": "Manual quote/payment link now; webhook later", "system_layer": "Quote + payment adapter"},
        {"step": 5, "name": "Intake or upload", "customer_view": "Only after proper unlock/consent", "system_layer": "Payment unlock + customer portal + order state"},
        {"step": 6, "name": "Evidence bundle delivery", "customer_view": "PDF/ZIP/manifest through secure delivery path", "system_layer": "Secure links + private delivery endpoint"},
    ]

    return {
        "version": VERSION,
        "module": MODULE,
        "created_at": _now(),
        "project_name": project_name,
        "company": company,
        "industry": industry,
        "customer_segment": customer_segment,
        "customer_problem": customer_problem,
        "desired_outcome": desired_outcome,
        "release_candidate_score": rc_score,
        "decision": rc_status,
        "recommended_route": portal.get("recommended_route"),
        "area_scores": area_scores,
        "area_table": [
            {
                **area,
                "score": area_scores.get(area["area"], 0),
                "status": "ready" if area_scores.get(area["area"], 0) >= 85 else "needs_attention",
            }
            for area in RC_READINESS_AREAS
        ],
        "customer_steps": customer_steps,
        "launch_checklist": LAUNCH_CHECKLIST,
        "blockers": blockers,
        "review_flags": review_flags,
        "safe_boundary": SAFE_BOUNDARY,
        "customer_safe_offer": {
            "headline": "Turn messy industrial ideas and data into a controlled pilot/evidence pack.",
            "primary_offer": "Professional Pilot Pack or Guided Custom Pack",
            "safe_claim": "EdgeTwin prepares data-readiness checks, trust notes, assumptions, limitations and concrete next steps for controlled pilots.",
            "not_included": "No production accuracy guarantee, no compliance/legal certification, no safety certification and no replacement of human review.",
            "cta": "Request quote / start guided custom setup",
        },
        "launch_mode": {
            "full_saas": False,
            "public_private_hosting_ready": bool(hosted_ready),
            "payment_handoff_ready": bool(payment_ready),
            "recommended_first_launch": "founder_led_private_link" if not hosted_ready else "private_public_pack_portal",
        },
        "source_scores": {
            "v119_portal": portal.get("portal_score"),
            "data_readiness": int(data_readiness_score),
            "synthetic_reliability": int(synthetic_reliability_score),
            "trust_score": int(trust_score),
        },
        "source_snapshots": {"v119": portal},
        "snapshot_hash": _hash_obj({
            "portal": portal.get("customer_problem_hash"),
            "area_scores": area_scores,
            "blockers": blockers,
            "review_flags": review_flags,
            "score": rc_score,
        }),
    }


def build_v120_area_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(snapshot.get("area_table", []))


def build_v120_customer_steps_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(snapshot.get("customer_steps", []))


def build_v120_flags_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for b in snapshot.get("blockers", []) or []:
        rows.append({"type": "blocker", "message": b})
    for f in snapshot.get("review_flags", []) or []:
        rows.append({"type": "review_flag", "message": f})
    return pd.DataFrame(rows or [{"type": "none", "message": "No blockers. Review standard launch checklist before customer test."}])


def _pdf_safe(value: Any) -> str:
    text = str(value or "")
    # fpdf latin-1 backend and long slash-joined words can break rendering.
    text = text.replace("→", "->").replace("—", "-").replace("–", "-")
    text = text.replace("/", "/ ")
    return text.encode("latin-1", "replace").decode("latin-1")


def _pdf(snapshot: Dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "EdgeTwin Product Release Candidate", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(190, 6, _pdf_safe(f"Project: {snapshot.get('project_name')} | Score: {snapshot.get('release_candidate_score')} | Decision: {snapshot.get('decision')}"))
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Customer-safe offer", ln=True)
    pdf.set_font("Arial", "", 10)
    offer = snapshot.get("customer_safe_offer", {})
    for key in ["headline", "primary_offer", "safe_claim", "not_included", "cta"]:
        pdf.multi_cell(190, 6, _pdf_safe(f"{key}: {offer.get(key, '')}"))
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Readiness areas", ln=True)
    pdf.set_font("Arial", "", 9)
    for row in snapshot.get("area_table", []):
        pdf.multi_cell(190, 5, _pdf_safe(f"- {row.get('label')}: {row.get('score')} ({row.get('status')})"))
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Flags", ln=True)
    pdf.set_font("Arial", "", 9)
    flags = list(snapshot.get("blockers", []) or []) + list(snapshot.get("review_flags", []) or [])
    for flag in flags[:12] or ["No blockers; review launch checklist before customer test."]:
        pdf.multi_cell(190, 5, _pdf_safe(f"- {flag}"))
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Boundary", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(190, 5, _pdf_safe(snapshot.get("safe_boundary", SAFE_BOUNDARY)))
    return bytes(pdf.output(dest="S"))


def create_product_release_candidate_bundle(snapshot: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("v120_release_candidate_snapshot.json", json.dumps(snapshot, indent=2, ensure_ascii=False, default=str))
        z.writestr("v120_area_scores.csv", build_v120_area_table(snapshot).to_csv(index=False))
        z.writestr("v120_customer_steps.csv", build_v120_customer_steps_table(snapshot).to_csv(index=False))
        z.writestr("v120_flags.csv", build_v120_flags_table(snapshot).to_csv(index=False))
        z.writestr("v120_launch_checklist.md", "\n".join([f"- {x}" for x in snapshot.get("launch_checklist", [])]))
        z.writestr("v120_customer_safe_offer.md", "\n".join([f"**{k}**: {v}" for k, v in snapshot.get("customer_safe_offer", {}).items()]))
        z.writestr("v120_release_candidate_summary.pdf", _pdf(snapshot))
    return buf.getvalue()
