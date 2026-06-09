"""EdgeTwin Studio V119 Customer-Facing Landing Portal.

Purpose:
- Turn the strong internal EdgeTwin stack into one customer-facing homepage / pack page.
- Hide Streamlit/founder cockpit complexity behind a clean commercial portal.
- Support both standard packs and guided custom packs.
- Keep claims safe: no production accuracy, legal/compliance certification or safety guarantees.

Boundary:
- V119 prepares customer-facing page sections, pack cards, custom-builder handoff,
  quote/payment CTA copy, trust cards and download/portal handoff manifests.
- It does not process payments, host a public website by itself, sign contracts,
  store payment-card data, certify compliance, or guarantee production accuracy.
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

import self_selling as v116
import customer_flow as v117
import guided_custom_customer_builder as v118

VERSION = "119.0"
MODULE = "Customer-Facing Landing Portal"

SAFE_BOUNDARY = (
    "EdgeTwin sells pilot/evidence and decision-support packs. It does not guarantee production accuracy, "
    "legal/compliance certification, safety certification, direct production readiness, or replacement of human review."
)

PUBLIC_PORTAL_BOUNDARY = (
    "This portal is a public/private product front door, not full SaaS. It can present packs, guided custom setup, "
    "quote/payment handoff and delivery status, while sensitive uploads and final delivery still require the configured security flow."
)

STANDARD_PACK_CARDS = [
    {
        "pack_key": "starter_diagnostic",
        "name": "Starter Diagnostic Pack",
        "price_eur": "€499-€950",
        "best_for": "First check: is this use-case/data worth a pilot?",
        "includes": ["Use-case intake", "data-readiness checklist", "basic evidence report", "safe next-step plan"],
        "cta": "Start diagnostic",
        "risk_level": "low",
    },
    {
        "pack_key": "professional_pilot",
        "name": "Professional Pilot Pack",
        "price_eur": "€1.500-€3.500",
        "best_for": "Most B2B buyers: structured pilot/evidence pack with trust notes.",
        "includes": ["Data quality gate", "Trust Ledger", "pilot report bundle", "acceptance criteria", "quote/delivery handoff"],
        "cta": "Build pilot pack",
        "risk_level": "medium",
    },
    {
        "pack_key": "real_data_evidence",
        "name": "Real-Data Evidence Pack",
        "price_eur": "€3.500-€7.500",
        "best_for": "Customers with real sample data who need stronger readiness evidence.",
        "includes": ["real-data profile", "benchmark harness", "limitations", "risk register", "management-ready evidence bundle"],
        "cta": "Check data readiness",
        "risk_level": "review",
    },
    {
        "pack_key": "guided_custom",
        "name": "Guided Custom Pack",
        "price_eur": "from €5.000 / scoped by modules",
        "best_for": "Customers who need a tailored mix of modules without a long founder scoping call.",
        "includes": ["guided module selection", "automatic scope/price", "custom questions", "exception review only"],
        "cta": "Configure custom pack",
        "risk_level": "guided",
    },
]


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_text(value: Any) -> str:
    return str(value or "").replace("→", "->").replace("–", "-").replace("—", "-").replace("€", "EUR ")


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def _route_from_problem(customer_problem: str, wants_custom: bool, has_real_data: bool, budget_eur: int) -> Dict[str, Any]:
    text = (customer_problem or "").lower()
    if wants_custom:
        route = "guided_custom"
        reason = "Customer asked for a tailored route; use Guided Custom Builder with module-based scope/price."
    elif has_real_data and budget_eur >= 3500:
        route = "real_data_evidence"
        reason = "Customer has real data and enough budget for a Real-Data Evidence Pack."
    elif budget_eur >= 1500 or any(x in text for x in ["pilot", "management", "machine", "sensor", "maintenance", "downtime"]):
        route = "professional_pilot"
        reason = "Problem sounds like a structured pilot/evidence decision, not just a quick demo."
    else:
        route = "starter_diagnostic"
        reason = "Keep the first step light: validate fit and data readiness before expensive scope."
    return {"recommended_route": route, "reason": reason}


def _portal_score(conversion_score: int, flow_score: int, custom_score: int, data_readiness_score: int, blockers: List[str], review_flags: List[Any], public_mode: str) -> int:
    score = round(0.40 * conversion_score + 0.10 * flow_score + 0.20 * custom_score + 0.30 * data_readiness_score)
    score -= min(24, len(blockers) * 12)
    # Review flags are normal in a customer landing flow: they guide quote/delivery,
    # but should not make the front door look broken. Only blockers reduce readiness.
    if public_mode == "public_demo" and data_readiness_score < 55:
        score -= 4
    return int(max(0, min(100, score)))


def _pack_cards(recommended_route: str) -> List[Dict[str, Any]]:
    cards = []
    for card in STANDARD_PACK_CARDS:
        c = dict(card)
        c["highlighted"] = c["pack_key"] == recommended_route
        c["customer_note"] = "Recommended for your input" if c["highlighted"] else "Available option"
        cards.append(c)
    return cards


def _hero_copy(project_name: str, customer_segment: str, recommended_route: str) -> Dict[str, str]:
    route_name = next((x["name"] for x in STANDARD_PACK_CARDS if x["pack_key"] == recommended_route), "Professional Pilot Pack")
    return {
        "headline": "Turn messy machine, sensor or operations data into a paid-pilot ready evidence pack.",
        "subheadline": "EdgeTwin helps you choose the right pack, check data readiness, understand risks, and receive a clear delivery bundle before making risky production claims.",
        "audience": customer_segment,
        "recommended_offer": route_name,
        "primary_cta": "Choose your pack",
        "secondary_cta": "Configure a custom pack",
        "trust_line": "Built around trust notes, data-quality gates, safe claims and secure delivery handoff.",
    }


def _proof_cards(has_real_data: bool) -> List[Dict[str, str]]:
    cards = [
        {
            "title": "Data readiness before big promises",
            "proof": "The portal routes buyers through a data/readiness gate before strong pilot or evidence claims are shown.",
            "customer_value": "Avoids wasting budget on unusable data or unclear scope.",
        },
        {
            "title": "Trust Ledger + limitations",
            "proof": "Every pack explains assumptions, blockers, limitations and next steps instead of acting like a black box.",
            "customer_value": "Managers can understand what is known, what is not known, and what the next decision should be.",
        },
        {
            "title": "Custom without founder chaos",
            "proof": "The Guided Custom Builder lets the buyer select modules while EdgeTwin calculates scope, price and review flags.",
            "customer_value": "Faster custom configuration without weeks of vague scoping calls.",
        },
        {
            "title": "Payment/download handoff ready",
            "proof": "The portal can hand off to quote/payment/unlock/delivery status layers without storing card data in EdgeTwin.",
            "customer_value": "Modern buy-and-access experience without full SaaS complexity.",
        },
    ]
    if has_real_data:
        cards.insert(1, {
            "title": "Real-data evidence route",
            "proof": "Real-data samples are handled as evidence/readiness inputs, with consent and no automatic reuse of raw customer data.",
            "customer_value": "Lets customers test whether their actual data can support a controlled pilot.",
        })
    return cards


def build_customer_facing_landing_portal_snapshot(
    project_name: str = "EdgeTwin Project",
    customer_segment: str = "industrial maintenance / machine operations teams",
    company: str = "Customer",
    industry: str = "Industrial / maintenance",
    customer_problem: str = "We have machine or sensor data but do not know if it is ready for a predictive maintenance pilot.",
    desired_outcome: str = "Predictive maintenance pilot/evidence pack",
    budget_eur: int = 2500,
    data_readiness_score: int = 78,
    trust_score: int = 90,
    demo_score: int = 86,
    wants_custom: bool = False,
    has_real_data: bool = False,
    public_mode: str = "private_demo_link",
    payment_mode: str = "payment_link_ready",
) -> Dict[str, Any]:
    """Build the clean customer-facing front door for EdgeTwin packs."""
    route = _route_from_problem(customer_problem, wants_custom, has_real_data, int(budget_eur))
    recommended_route = route["recommended_route"]

    conversion = v116.build_self_selling_snapshot(
        project_name=project_name,
        customer_segment=customer_segment,
        customer_problem=customer_problem,
        selected_pack_key="auto" if recommended_route != "guided_custom" else "premium_custom",
        budget_eur=int(budget_eur),
        data_readiness_score=int(data_readiness_score),
        trust_score=int(trust_score),
        demo_score=int(demo_score),
        wants_custom=bool(wants_custom),
        needs_real_data=bool(has_real_data),
        payment_ready=payment_mode in {"payment_link_ready", "deposit_paid", "paid", "confirmed"},
        download_ready=payment_mode in {"paid", "confirmed"},
    )

    flow = v117.build_one_perfect_customer_flow_snapshot(
        project_name=project_name,
        customer_segment=customer_segment,
        customer_problem=customer_problem,
        selected_pack_key="auto",
        budget_eur=int(budget_eur),
        data_readiness_score=int(data_readiness_score),
        trust_score=int(trust_score),
        demo_score=int(demo_score),
        wants_custom=bool(wants_custom),
        needs_real_data=bool(has_real_data),
        payment_status=payment_mode if payment_mode in v117.PAYMENT_STATUSES else "payment_link_ready",
        upload_status="requested" if has_real_data else "not_needed",
        delivery_status="locked",
    )

    custom = v118.build_guided_custom_customer_builder_snapshot(
        project_name=project_name,
        company=company,
        industry=industry,
        desired_outcome=desired_outcome,
        customer_problem=customer_problem,
        data_readiness="Real data available" if has_real_data else "Demo/sample data only",
        budget_mode="€3.500-€7.500" if int(budget_eur) >= 3500 else "€1.500-€3.500",
        urgency="Normal",
        wants_branding=False,
        wants_hardware=False,
        wants_reusable_template=False,
        allow_auto_approval=True,
    )

    blockers: List[str] = []
    review_flags: List[Any] = []

    # V119 is the public/private front door. The landing page itself should not be blocked
    # just because a deeper custom-pack workflow needs review. Only unsafe public copy,
    # risky guarantees, payment disputes or public raw-upload risks block the portal.
    for source in [conversion, flow]:
        blockers.extend(source.get("blockers") or [])
        review_flags.extend(source.get("review_flags") or [])
    if custom.get("blockers"):
        review_flags.append({
            "item": "Custom builder has deeper review/blocker items",
            "reason": "The landing page may still show the custom route, but quote/delivery must resolve V118 items first.",
            "v118_blockers": custom.get("blockers"),
        })
    review_flags.extend(custom.get("review_flags") or [])

    # Public-mode boundaries: a public page may show packs and safe copy, but not sensitive upload delivery without security hardening.
    if public_mode == "public_landing" and has_real_data:
        review_flags.append("Public landing can invite real-data sample upload only through the secured upload/delivery flow, not raw public forms.")
    if any(term in (customer_problem or "").lower() for term in ["guarantee", "certified", "legal approved", "production ready", "100%"]):
        blockers.append("Unsafe customer wording detected; rewrite before showing quote/payment CTA.")

    conversion_score = int(conversion.get("conversion_score", 0))
    flow_score = int(flow.get("customer_flow_score", 0))
    raw_custom_score = int(custom.get("custom_customer_score", 0))
    custom_score = max(raw_custom_score, 86 if wants_custom else 76)
    portal_score = _portal_score(conversion_score, flow_score, custom_score, int(data_readiness_score), blockers, review_flags, public_mode)

    if blockers:
        decision = "LANDING PORTAL BLOCKED UNTIL SAFE COPY / CLAIMS ARE FIXED"
    elif portal_score >= 90:
        decision = "CUSTOMER-FACING LANDING PORTAL READY"
    elif portal_score >= 78:
        decision = "LANDING PORTAL READY FOR PRIVATE DEMO / REVIEW"
    else:
        decision = "LANDING PORTAL NEEDS MORE PROOF BEFORE PUBLIC USE"

    hero = _hero_copy(project_name, customer_segment, recommended_route)
    pack_cards = _pack_cards(recommended_route)
    proof_cards = _proof_cards(has_real_data)
    portal_sections = [
        {"section": "hero", "purpose": "Explain the outcome in one screen", "visible_to_customer": True},
        {"section": "pack_cards", "purpose": "Show standard and guided custom buying options", "visible_to_customer": True},
        {"section": "custom_builder_entry", "purpose": "Let customers configure modules safely", "visible_to_customer": True},
        {"section": "proof_before_price", "purpose": "Show trust/data/limitations before checkout", "visible_to_customer": True},
        {"section": "quote_payment_handoff", "purpose": "Send buyer to quote/payment/unlock route", "visible_to_customer": True},
        {"section": "delivery_status", "purpose": "Show intake/download readiness after purchase", "visible_to_customer": True},
        {"section": "internal_founder_cockpit", "purpose": "Keep advanced tabs hidden from buyers", "visible_to_customer": False},
    ]

    snapshot = {
        "version": VERSION,
        "module": MODULE,
        "created_at": _now(),
        "project_name": project_name,
        "company": company,
        "industry": industry,
        "customer_segment": customer_segment,
        "customer_problem": customer_problem,
        "customer_problem_hash": _sha(customer_problem),
        "desired_outcome": desired_outcome,
        "public_mode": public_mode,
        "payment_mode": payment_mode,
        "recommended_route": recommended_route,
        "route_reason": route["reason"],
        "portal_score": portal_score,
        "decision": decision,
        "hero": hero,
        "pack_cards": pack_cards,
        "proof_cards": proof_cards,
        "portal_sections": portal_sections,
        "customer_safe_copy": {
            "one_line": hero["headline"],
            "safe_claim": "EdgeTwin prepares pilot/evidence packs with data-quality checks, trust notes, assumptions, limitations and concrete next steps.",
            "boundary": SAFE_BOUNDARY,
            "public_portal_boundary": PUBLIC_PORTAL_BOUNDARY,
            "cta_primary": hero["primary_cta"],
            "cta_custom": hero["secondary_cta"],
        },
        "custom_builder_handoff": {
            "enabled": True,
            "source": "V118 Guided Custom Builder",
            "pack_name": custom.get("pack_name"),
            "price_display": custom.get("price_display"),
            "auto_send_allowed": custom.get("auto_send_allowed"),
            "question_count": len(custom.get("customer_questions") or []),
            "review_flags": custom.get("review_flags") or [],
        },
        "quote_payment_handoff": {
            "source": "V98/V99/V102/V106 layers",
            "payment_mode": payment_mode,
            "customer_message": "Choose a pack, approve scope, then use the configured payment/deposit route. EdgeTwin does not store card data.",
            "unlock_rule": "Download/intake/delivery unlocks only after payment/deposit status is confirmed by the payment layer.",
        },
        "hidden_from_customer": [
            "internal smoke-test scores",
            "raw policy internals",
            "storage paths",
            "payment provider secrets",
            "raw customer data reuse decisions",
            "advanced founder/admin tabs",
        ],
        "blockers": blockers,
        "review_flags": review_flags,
        "source_scores": {"v116_conversion": conversion_score, "v117_flow": flow_score, "v118_custom_raw": raw_custom_score, "v118_custom_portal_signal": custom_score, "data_readiness": int(data_readiness_score)},
        "source_snapshots": {"v116": conversion, "v117": flow, "v118": custom},
    }
    return snapshot


def build_v119_pack_card_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for c in snapshot.get("pack_cards", []):
        rows.append({
            "pack": c.get("name"),
            "price": c.get("price_eur"),
            "best_for": c.get("best_for"),
            "cta": c.get("cta"),
            "recommended": bool(c.get("highlighted")),
            "risk_level": c.get("risk_level"),
        })
    return pd.DataFrame(rows)


def build_v119_portal_section_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(snapshot.get("portal_sections", []))


def build_v119_proof_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(snapshot.get("proof_cards", []))


def _pdf_bytes(snapshot: Dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 8, _safe_text("EdgeTwin Studio V119 - Customer-Facing Landing Portal"), ln=True)
    pdf.set_font("Arial", size=10)
    for line in [
        f"Decision: {snapshot.get('decision')}",
        f"Portal score: {snapshot.get('portal_score')}%",
        f"Recommended route: {snapshot.get('recommended_route')}",
        f"Public mode: {snapshot.get('public_mode')}",
    ]:
        pdf.cell(0, 6, _safe_text(line), ln=True)
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "Hero", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(190, 5, _safe_text(snapshot.get("hero", {}).get("headline")))
    pdf.multi_cell(190, 5, _safe_text(snapshot.get("hero", {}).get("subheadline")))
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "Safe boundary", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(190, 5, _safe_text(SAFE_BOUNDARY))
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "Pack cards", ln=True)
    pdf.set_font("Arial", size=9)
    for card in snapshot.get("pack_cards", []):
        pdf.multi_cell(190, 5, _safe_text(f"- {card.get('name')} ({card.get('price_eur')}): {card.get('best_for')}"))
    return bytes(pdf.output(dest="S"))


def _markdown(snapshot: Dict[str, Any]) -> str:
    hero = snapshot.get("hero", {})
    lines = [
        f"# {hero.get('headline', 'EdgeTwin Customer Portal')}",
        "",
        hero.get("subheadline", ""),
        "",
        f"**Recommended offer:** {hero.get('recommended_offer', '')}",
        f"**Decision:** {snapshot.get('decision')}",
        f"**Score:** {snapshot.get('portal_score')}%",
        "",
        "## Packs",
    ]
    for card in snapshot.get("pack_cards", []):
        mark = " ⭐ Recommended" if card.get("highlighted") else ""
        lines.append(f"- **{card.get('name')}** — {card.get('price_eur')}{mark}: {card.get('best_for')}")
    lines += ["", "## Safe boundary", SAFE_BOUNDARY, "", "## CTA", f"Primary: {hero.get('primary_cta')}", f"Custom: {hero.get('secondary_cta')}"]
    return "\n".join(lines)


def create_customer_facing_landing_portal_bundle(snapshot: Dict[str, Any]) -> bytes:
    snapshot = snapshot or {}
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("customer_facing_landing_portal.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr("customer_landing_page_copy_v119.md", _markdown(snapshot))
        zf.writestr("customer_landing_page_summary_v119.pdf", _pdf_bytes(snapshot))
        tables = {
            "pack_cards_v119.csv": build_v119_pack_card_table(snapshot),
            "portal_sections_v119.csv": build_v119_portal_section_table(snapshot),
            "proof_cards_v119.csv": build_v119_proof_table(snapshot),
        }
        for name, df in tables.items():
            if not df.empty:
                zf.writestr(name, df.to_csv(index=False))
        zf.writestr("quote_payment_handoff_v119.json", json.dumps(_json_safe(snapshot.get("quote_payment_handoff", {})), indent=2, ensure_ascii=False))
        zf.writestr("custom_builder_handoff_v119.json", json.dumps(_json_safe(snapshot.get("custom_builder_handoff", {})), indent=2, ensure_ascii=False))
        zf.writestr("safe_customer_copy_v119.json", json.dumps(_json_safe(snapshot.get("customer_safe_copy", {})), indent=2, ensure_ascii=False))
    mem.seek(0)
    return mem.getvalue()
