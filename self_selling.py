"""EdgeTwin Studio V116 Self-Selling Conversion Engine.

Purpose:
- Turn the product/pack logic into a customer-convincing sales page and payment-ready journey.
- Explain value, proof, price justification, objections, next steps and safe claims automatically.
- Keep conversion focused on pilot/evidence packs, not production/accuracy/legal/compliance guarantees.

Boundary:
- V116 can generate customer-safe sales copy, proof cards, objection handling and CTA flows.
- It does not execute payments, sign contracts, guarantee accuracy, or certify compliance.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import re
import zipfile
from typing import Any, Dict, List, Optional

from fpdf import FPDF

VERSION = "116.0"
MODULE = "Self-Selling Conversion Engine"

PACK_LIBRARY: Dict[str, Dict[str, Any]] = {
    "free_discovery": {
        "name": "Free Discovery Check",
        "price_range_eur": [0, 0],
        "best_for": "Curious leads who need to understand the fit before paying.",
        "included": [
            "One short use-case fit check",
            "Suggested paid pack route",
            "Missing-input checklist",
            "Customer-safe next step",
        ],
        "primary_cta": "Start free discovery",
        "conversion_goal": "lead_capture",
    },
    "starter_diagnostic": {
        "name": "Starter Diagnostic Pack",
        "price_range_eur": [499, 950],
        "best_for": "Teams with a problem/data sample who need a first technical go/no-go.",
        "included": [
            "Client intake summary",
            "Data readiness check",
            "Initial trust/evidence notes",
            "PDF/ZIP diagnostic bundle",
            "Concrete next-step recommendation",
        ],
        "primary_cta": "Buy Starter Diagnostic",
        "conversion_goal": "low_friction_purchase",
    },
    "professional_pilot": {
        "name": "Professional Pilot Pack",
        "price_range_eur": [1500, 3500],
        "best_for": "Companies that want a serious pilot/evidence package for management or technical review.",
        "included": [
            "Everything in Starter",
            "Trust Ledger and limitation notes",
            "Pack-specific acceptance criteria",
            "Risk and blocker matrix",
            "Customer demo summary",
            "Founder-safe delivery plan",
        ],
        "primary_cta": "Request Professional Pilot Pack",
        "conversion_goal": "paid_pilot",
    },
    "real_data_evidence": {
        "name": "Real-Data Evidence Pack",
        "price_range_eur": [3500, 7500],
        "best_for": "Teams with representative real data who need evidence before production decisions.",
        "included": [
            "Real-data profile and quality gate",
            "Synthetic-to-real benchmark comparison where allowed",
            "Pilot readiness score",
            "Operational risk list",
            "Data improvement roadmap",
            "Production-handoff candidate notes, not guarantees",
        ],
        "primary_cta": "Request Real-Data Evidence Pack",
        "conversion_goal": "qualified_consultative_sale",
    },
    "premium_custom": {
        "name": "Premium Custom Pack",
        "price_range_eur": [7500, 15000],
        "best_for": "High-value use-cases that need custom modules, templates or domain-specific delivery.",
        "included": [
            "Custom pack builder configuration",
            "Custom deliverable map",
            "Reusable pack template candidate",
            "Founder review queue",
            "Scope/claim/payment guardrails",
            "Optional hardware/BOM reference notes",
        ],
        "primary_cta": "Build custom pack",
        "conversion_goal": "high_value_custom",
    },
}

PROOF_CARDS = [
    {
        "title": "Trust Ledger instead of black-box claims",
        "proof": "Every pack highlights assumptions, limitations, evidence status and what cannot be claimed yet.",
        "customer_benefit": "Management can see why a pilot is or is not ready, instead of guessing.",
    },
    {
        "title": "Data Quality Gate",
        "proof": "Synthetic, public benchmark and customer-approved datasets can be checked against the same readiness harness.",
        "customer_benefit": "Bad data is flagged early before money is wasted on a weak pilot.",
    },
    {
        "title": "Pack-to-delivery workflow",
        "proof": "Pack builder, quote, payment status, customer portal, secure download and fulfillment state machine are aligned.",
        "customer_benefit": "The buying experience feels like a modern digital product, not a long consulting process.",
    },
    {
        "title": "Safe claim boundaries",
        "proof": "The claim policy blocks production-ready, 100% accuracy, legal/compliance and liability guarantees.",
        "customer_benefit": "The output is useful and honest without creating false confidence.",
    },
]

OBJECTION_LIBRARY: Dict[str, Dict[str, str]] = {
    "too_expensive": {
        "customer_objection": "This is expensive for a report/bundle.",
        "safe_answer": "You are not buying a static report. You are buying a structured pilot-readiness process: intake, data quality gate, trust notes, risks, acceptance criteria and a delivery bundle that can prevent weeks of unclear internal work.",
    },
    "need_accuracy": {
        "customer_objection": "Can you guarantee accuracy?",
        "safe_answer": "Not before validation on representative labelled field data. EdgeTwin can show whether the data and use-case are ready for a controlled pilot and which validation steps are needed before production accuracy claims.",
    },
    "already_have_data_team": {
        "customer_objection": "We already have engineers/data people.",
        "safe_answer": "EdgeTwin is not meant to replace them. It gives them a structured starting point, data quality signal, risk list and evidence bundle so they can move faster.",
    },
    "send_sample_first": {
        "customer_objection": "Can we send a small sample first?",
        "safe_answer": "Yes. A small sample is ideal for a Starter Diagnostic or Professional Pilot route. The output will clearly state whether the sample is enough for the chosen next step.",
    },
    "production_ready": {
        "customer_objection": "Can we use this directly in production?",
        "safe_answer": "The pack is designed for pilot/evidence and decision support. Production deployment requires additional validation, security/privacy review and customer approval.",
    },
    "why_not_saas": {
        "customer_objection": "Why is this not a SaaS subscription?",
        "safe_answer": "The pack model is faster and lower-risk for early evidence. It gives a concrete deliverable first; SaaS or ongoing automation can come later if the pilot proves value.",
    },
}

RISKY_CLAIM_PATTERNS = [
    r"\b100\s*%\s*accurate\b",
    r"\bguarantee(?:d|s)?\b.*\baccuracy\b",
    r"\bproduction[-\s]?ready\b",
    r"\bcertif(?:y|ied|ication)\b",
    r"\bcompliance\s+(?:approved|certified|guaranteed)\b",
    r"\blegal\s+(?:approved|certified|guaranteed)\b",
    r"\breplace(?:s)?\s+human\s+review\b",
    r"\baccept\s+liability\b",
]

SAFE_BASE_CLAIM = (
    "EdgeTwin prepares controlled pilot/evidence packs with data-quality checks, trust notes, "
    "assumptions, limitations and concrete next steps. It does not guarantee production accuracy, "
    "legal/compliance certification or direct production readiness without additional validation and approval."
)


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_text(value: Any) -> str:
    text = str(value or "")
    return text.replace("\u2192", "->").replace("\u2013", "-").replace("\u2014", "-")


def _slug(value: Any) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "unknown")).strip("._")
    return s[:96] or "unknown"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _clamp_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = default
    return max(lo, min(hi, v))


def detect_risky_claims_v116(text: str) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []
    source = str(text or "")
    for pat in RISKY_CLAIM_PATTERNS:
        if re.search(pat, source, flags=re.IGNORECASE):
            findings.append({"pattern": pat, "severity": "block_or_review", "safe_replacement": SAFE_BASE_CLAIM})
    return findings


def recommend_pack_for_conversion_v116(
    budget_eur: int,
    data_readiness_score: int,
    urgency: str,
    wants_custom: bool,
    needs_real_data: bool,
) -> str:
    urgency_l = str(urgency or "normal").lower()
    if wants_custom or budget_eur >= 7500:
        return "premium_custom"
    if needs_real_data or data_readiness_score >= 82 or budget_eur >= 5000:
        return "real_data_evidence"
    if budget_eur >= 1500 or urgency_l in {"high", "urgent", "this week"}:
        return "professional_pilot"
    if budget_eur >= 499 or data_readiness_score >= 45:
        return "starter_diagnostic"
    return "free_discovery"


def build_conversion_journey_v116(pack_key: str, payment_ready: bool, download_ready: bool) -> List[Dict[str, Any]]:
    pack = PACK_LIBRARY.get(pack_key, PACK_LIBRARY["professional_pilot"])
    return [
        {"step": 1, "title": "Understand the problem", "customer_action": "Describe the use-case and upload/select sample data", "system_output": "Plain-language use-case summary and missing-input checklist", "status": "ready"},
        {"step": 2, "title": "See the recommended pack", "customer_action": "Review the suggested pack and scope", "system_output": pack["name"], "status": "ready"},
        {"step": 3, "title": "Trust before payment", "customer_action": "Review what is included, what is excluded and why the pack is priced this way", "system_output": "Proof cards + safe claim boundary + example deliverables", "status": "ready"},
        {"step": 4, "title": "Pay / request quote", "customer_action": "Use Stripe/Paddle/manual invoice route", "system_output": "Payment status can unlock intake/download through the payment, unlock and order-state layers", "status": "ready" if payment_ready else "manual_or_provider_needed"},
        {"step": 5, "title": "Receive delivery", "customer_action": "Download the bundle or complete intake/upload", "system_output": "Secure link, portal status and delivery audit path", "status": "ready" if download_ready else "prepared"},
    ]


def build_pricing_justification_v116(pack_key: str) -> Dict[str, Any]:
    pack = PACK_LIBRARY.get(pack_key, PACK_LIBRARY["professional_pilot"])
    lo, hi = pack["price_range_eur"]
    if hi == 0:
        anchor = "Free entry point to reduce buyer friction and qualify serious leads."
    elif hi <= 1000:
        anchor = "Priced as a low-friction diagnostic that replaces messy early investigation work."
    elif hi <= 3500:
        anchor = "Priced as a professional pilot-readiness bundle, not as a simple file export."
    elif hi <= 7500:
        anchor = "Priced around real-data evidence work where the customer needs confidence before larger decisions."
    else:
        anchor = "Priced as custom/pre-implementation work with reusable template value and higher founder/domain involvement."
    return {
        "pack": pack["name"],
        "price_range_eur": pack["price_range_eur"],
        "why_this_price_is_defensible": anchor,
        "customer_buys": [
            "speed to a decision",
            "structured evidence instead of ad-hoc analysis",
            "data quality clarity",
            "risk and limitation visibility",
            "downloadable management/technical bundle",
        ],
        "not_sold_as": [
            "production accuracy guarantee",
            "legal/compliance certification",
            "replacement for human review",
        ],
    }


def build_homepage_sections_v116(pack_key: str, customer_segment: str, pain_points: List[str]) -> List[Dict[str, Any]]:
    pack = PACK_LIBRARY.get(pack_key, PACK_LIBRARY["professional_pilot"])
    pains = [p for p in pain_points if str(p).strip()] or [
        "unclear data readiness",
        "slow pilot scoping",
        "too many unanswered risks before budget approval",
    ]
    return [
        {
            "section": "hero",
            "headline": "Turn messy industrial data ideas into a pilot-ready evidence pack.",
            "subheadline": f"For {customer_segment or 'machine, sensor and operations teams'} who need clarity before spending months on a weak pilot.",
            "cta": pack["primary_cta"],
        },
        {
            "section": "problem",
            "headline": "Most teams do not fail because they lack data. They fail because they do not know if the data is usable.",
            "bullets": pains[:5],
        },
        {
            "section": "solution",
            "headline": "EdgeTwin packages the first 80% of pilot-preparation into a structured deliverable.",
            "bullets": pack["included"],
        },
        {
            "section": "proof",
            "headline": "Built for trust, not black-box hype.",
            "bullets": [card["title"] for card in PROOF_CARDS],
        },
        {
            "section": "boundary",
            "headline": "Honest limits are part of the product.",
            "body": SAFE_BASE_CLAIM,
        },
    ]


def build_conversion_score_v116(
    trust_score: int,
    data_readiness_score: int,
    demo_score: int,
    proof_asset_count: int,
    risky_claim_count: int,
    pack_key: str,
    payment_ready: bool,
    download_ready: bool,
) -> int:
    score = 52
    score += min(15, max(0, trust_score - 60) // 3)
    score += min(12, max(0, data_readiness_score - 50) // 4)
    score += min(10, max(0, demo_score - 50) // 5)
    score += min(8, proof_asset_count * 2)
    if payment_ready:
        score += 5
    if download_ready:
        score += 4
    if pack_key in {"professional_pilot", "real_data_evidence"}:
        score += 4
    if pack_key == "free_discovery":
        score -= 5
    score -= risky_claim_count * 12
    return max(0, min(100, int(score)))


def build_self_selling_snapshot(
    project_name: str = "EdgeTwin_Project",
    customer_segment: str = "industrial machine and operations teams",
    customer_problem: str = "We have machine/sensor data but do not know if it is ready for a predictive maintenance pilot.",
    pain_points: Optional[List[str]] = None,
    selected_pack_key: str = "auto",
    budget_eur: int = 2500,
    data_readiness_score: int = 76,
    trust_score: int = 88,
    demo_score: int = 84,
    urgency: str = "normal",
    wants_custom: bool = False,
    needs_real_data: bool = False,
    proof_assets: Optional[List[str]] = None,
    draft_claim_text: str = "",
    payment_ready: bool = True,
    download_ready: bool = True,
) -> Dict[str, Any]:
    pain_points = pain_points or [
        "We do not know whether our data is usable.",
        "We need management-friendly evidence before approving a pilot.",
        "We want to avoid wasting months on the wrong data/use-case.",
    ]
    proof_assets = proof_assets or ["Trust Ledger", "Data Quality Gate", "Synthetic Reliability Lab", "Benchmark Harness"]
    if selected_pack_key == "auto" or selected_pack_key not in PACK_LIBRARY:
        selected_pack_key = recommend_pack_for_conversion_v116(
            budget_eur=budget_eur,
            data_readiness_score=data_readiness_score,
            urgency=urgency,
            wants_custom=wants_custom,
            needs_real_data=needs_real_data,
        )
    pack = PACK_LIBRARY[selected_pack_key]

    claim_findings = detect_risky_claims_v116(draft_claim_text)
    conversion_score = build_conversion_score_v116(
        trust_score=_clamp_int(trust_score, 88, 0, 100),
        data_readiness_score=_clamp_int(data_readiness_score, 76, 0, 100),
        demo_score=_clamp_int(demo_score, 84, 0, 100),
        proof_asset_count=len(proof_assets),
        risky_claim_count=len(claim_findings),
        pack_key=selected_pack_key,
        payment_ready=bool(payment_ready),
        download_ready=bool(download_ready),
    )

    blockers: List[str] = []
    review_flags: List[str] = []
    if claim_findings:
        blockers.append("Risky claim text found. Use safe rewrite before customer-facing release.")
    if data_readiness_score < 45:
        review_flags.append("Data readiness is low; sell discovery/starter route, not pilot evidence.")
    if trust_score < 70:
        review_flags.append("Trust score is not high enough for strong conversion copy.")
    if not payment_ready:
        review_flags.append("Payment provider route is prepared but not live; use manual/Payment Link workflow.")
    if not download_ready:
        review_flags.append("Delivery/download unlock is prepared but needs final endpoint/storage setup.")

    if blockers:
        decision = "CONVERSION BLOCKED - FIX CLAIMS BEFORE SELLING"
    elif conversion_score >= 88:
        decision = "SELF-SELLING FUNNEL READY"
    elif conversion_score >= 76:
        decision = "CUSTOMER CONVERSION READY - IMPROVE PROOF/CTA"
    elif conversion_score >= 62:
        decision = "DEMO FUNNEL READY - NOT STRONG ENOUGH FOR AUTO-SELL"
    else:
        decision = "NEEDS POSITIONING WORK"

    homepage_sections = build_homepage_sections_v116(selected_pack_key, customer_segment, pain_points)
    objections = list(OBJECTION_LIBRARY.values())
    journey = build_conversion_journey_v116(selected_pack_key, payment_ready, download_ready)
    pricing = build_pricing_justification_v116(selected_pack_key)

    customer_copy = {
        "one_line_pitch": "EdgeTwin turns messy industrial data ideas into controlled pilot/evidence packs with trust notes, data checks and next steps.",
        "safe_claim": SAFE_BASE_CLAIM,
        "main_cta": pack["primary_cta"],
        "secondary_cta": "Start with a free discovery check" if selected_pack_key != "free_discovery" else "Review recommended paid route",
        "price_anchor": pricing["why_this_price_is_defensible"],
    }

    snapshot = {
        "version": VERSION,
        "module": MODULE,
        "created_at": _now(),
        "project_name": str(project_name),
        "customer_segment": str(customer_segment),
        "customer_problem": str(customer_problem),
        "selected_pack_key": selected_pack_key,
        "selected_pack": pack,
        "budget_eur": int(budget_eur),
        "input_scores": {
            "data_readiness_score": _clamp_int(data_readiness_score, 76, 0, 100),
            "trust_score": _clamp_int(trust_score, 88, 0, 100),
            "demo_score": _clamp_int(demo_score, 84, 0, 100),
        },
        "conversion_score": conversion_score,
        "decision": decision,
        "proof_assets": proof_assets,
        "proof_cards": PROOF_CARDS,
        "homepage_sections": homepage_sections,
        "pricing_justification": pricing,
        "objection_responses": objections,
        "buyer_journey": journey,
        "customer_copy": customer_copy,
        "claim_findings": claim_findings,
        "blockers": blockers,
        "review_flags": review_flags,
        "recommended_next_actions": _next_actions(conversion_score, blockers, selected_pack_key),
        "sales_boundary": {
            "allowed": [
                "pilot/evidence pack",
                "data-quality check",
                "trust notes",
                "assumptions and limitations",
                "decision-support report",
                "quote/payment/download journey",
            ],
            "blocked_without_review": [
                "production accuracy guarantee",
                "legal/compliance certification",
                "liability acceptance",
                "replacement of human review",
                "automatic refunds/contracts",
            ],
        },
        "content_hash": _sha(json.dumps({"pack": selected_pack_key, "problem": customer_problem, "score": conversion_score}, sort_keys=True)),
    }
    return snapshot


def _next_actions(conversion_score: int, blockers: List[str], pack_key: str) -> List[str]:
    if blockers:
        return [
            "Replace risky customer-facing claims with the V116 safe claim.",
            "Run Claim Safety V108 before publishing the page/copy.",
            "Do not unlock payment flow until the blocked copy is fixed.",
        ]
    actions = [
        "Publish this as a simple pack landing page section, not as a complex technical cockpit.",
        "Show proof cards before the price so the amount feels justified.",
        "Place one primary CTA next to the selected pack and one secondary discovery CTA below it.",
    ]
    if conversion_score < 88:
        actions.append("Add stronger proof: sample PDF, screenshot, benchmark score or safe mini case study.")
    if pack_key in {"real_data_evidence", "premium_custom"}:
        actions.append("Keep final scope/payment under review for high-value or custom deals.")
    return actions


def build_v116_objection_table(snapshot: Dict[str, Any]):
    try:
        import pandas as pd
        return pd.DataFrame(snapshot.get("objection_responses", []))
    except Exception:
        return []


def build_v116_homepage_table(snapshot: Dict[str, Any]):
    try:
        import pandas as pd
        return pd.DataFrame(snapshot.get("homepage_sections", []))
    except Exception:
        return []


def _pdf_bytes(snapshot: Dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "EdgeTwin V116 Self-Selling Conversion Summary", ln=True)
    pdf.set_font("Arial", "", 10)
    lines = [
        f"Project: {_safe_text(snapshot.get('project_name'))}",
        f"Decision: {_safe_text(snapshot.get('decision'))}",
        f"Conversion score: {snapshot.get('conversion_score')}%",
        f"Selected pack: {_safe_text(snapshot.get('selected_pack', {}).get('name'))}",
        f"Price range EUR: {snapshot.get('selected_pack', {}).get('price_range_eur')}",
        "",
        "Safe claim:",
        _safe_text(snapshot.get('customer_copy', {}).get('safe_claim')),
        "",
        "Why customers pay:",
    ]
    for item in snapshot.get("pricing_justification", {}).get("customer_buys", []):
        lines.append(f"- {_safe_text(item)}")
    lines += ["", "Next actions:"]
    for item in snapshot.get("recommended_next_actions", []):
        lines.append(f"- {_safe_text(item)}")
    for line in lines:
        safe_line = _safe_text(line)
        # fpdf can fail on very long unbroken fragments in some environments; wrap defensively.
        if not safe_line:
            pdf.ln(3)
            continue
        chunks = [safe_line[i:i+100] for i in range(0, len(safe_line), 100)]
        for chunk in chunks:
            pdf.multi_cell(185, 6, chunk)
    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1", errors="replace")
    return bytes(out)


def create_self_selling_bundle(snapshot: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("v116_self_selling_conversion_snapshot.json", json.dumps(snapshot, indent=2, ensure_ascii=False))
        zf.writestr("v116_customer_landing_page_sections.json", json.dumps(snapshot.get("homepage_sections", []), indent=2, ensure_ascii=False))
        zf.writestr("v116_safe_customer_copy.md", _markdown_customer_copy(snapshot))
        zf.writestr("v116_objection_responses.json", json.dumps(snapshot.get("objection_responses", []), indent=2, ensure_ascii=False))
        zf.writestr("v116_buyer_journey.json", json.dumps(snapshot.get("buyer_journey", []), indent=2, ensure_ascii=False))
        zf.writestr("v116_conversion_summary.pdf", _pdf_bytes(snapshot))
    return buf.getvalue()


def _markdown_customer_copy(snapshot: Dict[str, Any]) -> str:
    pack = snapshot.get("selected_pack", {})
    lines = [
        f"# {pack.get('name', 'EdgeTwin Pack')}",
        "",
        snapshot.get("customer_copy", {}).get("one_line_pitch", ""),
        "",
        "## What you get",
    ]
    for item in pack.get("included", []):
        lines.append(f"- {item}")
    lines += [
        "",
        "## Price range",
        f"EUR {pack.get('price_range_eur', ['TBD', 'TBD'])[0]} - {pack.get('price_range_eur', ['TBD', 'TBD'])[1]}",
        "",
        "## Important boundary",
        snapshot.get("customer_copy", {}).get("safe_claim", SAFE_BASE_CLAIM),
        "",
        "## CTA",
        snapshot.get("customer_copy", {}).get("main_cta", "Request pack"),
    ]
    return "\n".join(_safe_text(x) for x in lines)


if __name__ == "__main__":
    snap = build_self_selling_snapshot()
    print(json.dumps({"decision": snap["decision"], "score": snap["conversion_score"], "pack": snap["selected_pack"]["name"]}, indent=2))
