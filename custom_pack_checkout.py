"""EdgeTwin Studio Custom Pack Checkout Autopilot.

Purpose:
- Let customers build a custom pack and receive an automatic price/scope/payment handoff.
- Keep standard custom work automated by default.
- Only block unsafe promises, refunds/disputes, legal/compliance claims, production guarantees,
  or safety-critical hardware obligations.

This module does not process payment cards. Stripe/Paddle/manual payment providers are expected
outside this module. It prepares the checkout/order snapshot and unlock rules.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import zipfile
from typing import Any, Dict, List, Tuple

import pandas as pd
from fpdf import FPDF

MODULE = "Custom Pack Checkout Autopilot"

SAFE_BOUNDARY = (
    "EdgeTwin custom packs provide pilot/evidence and decision-support deliverables. "
    "They do not guarantee production accuracy, legal/compliance certification, safety certification, "
    "or removal of responsible human oversight for high-risk decisions without additional validation and formal agreement."
)

# Customer-facing module catalogue. Human review/call add-ons are intentionally excluded for now.
# Standard custom packs should be automated unless unsafe claims or high-risk scope appear.
MODULE_CATALOG = [
    {
        "module_id": "intake_scope",
        "label": "Use-case intake + scope lock",
        "price_eur": 350,
        "automation": "automatic",
        "customer_value": "Turns a loose request into a clear scope and next-step map.",
        "delivery": "Intake summary, scope boundary, assumptions and exclusions.",
    },
    {
        "module_id": "data_quality_gate",
        "label": "Data quality gate",
        "price_eur": 650,
        "automation": "automatic",
        "customer_value": "Shows whether the customer data is good enough for demo, pilot or evidence use.",
        "delivery": "Data readiness score, missing inputs, blockers and improvement plan.",
    },
    {
        "module_id": "feature_anomaly_analysis",
        "label": "Feature/anomaly analysis",
        "price_eur": 950,
        "automation": "automatic",
        "customer_value": "Screens the data for useful signals, unusual behaviour and pilot potential.",
        "delivery": "Feature summary, anomaly notes and evidence-ready charts/tables where possible.",
    },
    {
        "module_id": "trust_ledger",
        "label": "Trust ledger + limitations",
        "price_eur": 550,
        "automation": "automatic",
        "customer_value": "Explains what can and cannot be trusted yet.",
        "delivery": "Trust notes, assumptions, limitations, safe claims and risk flags.",
    },
    {
        "module_id": "customer_report_bundle",
        "label": "PDF/ZIP customer report bundle",
        "price_eur": 550,
        "automation": "automatic",
        "customer_value": "Gives the customer a tangible delivery package for internal decision-making.",
        "delivery": "PDF/ZIP/JSON/CSV/Markdown buyer-ready delivery bundle.",
    },
    {
        "module_id": "maintenance_alignment",
        "label": "Maintenance history alignment",
        "price_eur": 900,
        "automation": "automatic_if_data_available",
        "customer_value": "Connects sensor/machine data to known maintenance or failure events when provided.",
        "delivery": "Event alignment notes, missing history list and pilot-readiness comments.",
    },
    {
        "module_id": "extra_data_source",
        "label": "Extra data source / file type",
        "price_eur": 500,
        "automation": "automatic_for_standard_files",
        "customer_value": "Adds one extra standard source such as CSV, Excel, JSON or maintenance export.",
        "delivery": "Source mapping, basic validation and integration notes.",
    },
    {
        "module_id": "branding_language",
        "label": "Branding / language pack",
        "price_eur": 450,
        "automation": "automatic",
        "customer_value": "Makes the deliverable easier to share internally or with management.",
        "delivery": "Customer-facing wording, language/tone choice and branded summary sections.",
    },
    {
        "module_id": "extra_use_case_variant",
        "label": "Extra use-case variant",
        "price_eur": 1500,
        "automation": "automatic_when_non_safety_critical",
        "customer_value": "Adds one extra bounded scenario without turning the pack into unlimited consulting.",
        "delivery": "Second use-case route, evidence notes, risks and acceptance criteria.",
    },
    {
        "module_id": "hardware_bom_reference",
        "label": "Hardware/BOM reference module",
        "price_eur": 1500,
        "automation": "automatic_as_reference_only",
        "customer_value": "Gives practical reference guidance for sensors, gateway, data capture and deployment cautions.",
        "delivery": "Reference BOM notes and deployment checklist. Not certified engineering.",
    },
    {
        "module_id": "reusable_custom_template",
        "label": "Reusable custom pack template",
        "price_eur": 2500,
        "automation": "automatic_template_generation",
        "customer_value": "Turns one custom setup into a reusable template for future similar requests.",
        "delivery": "Reusable pack recipe, module config and repeatable report structure.",
    },
]

BASE_PACKS = {
    "Custom Lite Pack": {"base_price_eur": 950, "max_modules": 6, "deposit_pct": 1.0},
    "Custom Professional Pack": {"base_price_eur": 1500, "max_modules": 8, "deposit_pct": 1.0},
    "Custom Real-Data Evidence Pack": {"base_price_eur": 3500, "max_modules": 10, "deposit_pct": 0.5},
    "Premium Custom Pack": {"base_price_eur": 7500, "max_modules": 12, "deposit_pct": 0.5},
}

BLOCK_TERMS = [
    "100%", "guarantee", "guaranteed", "production-ready", "production ready", "certified",
    "legal approved", "compliance approved", "liability", "risk-free", "always predict",
    "replace human", "safety critical", "ce certified", "ai act compliant",
]


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def _catalog_lookup() -> Dict[str, Dict[str, Any]]:
    return {m["module_id"]: dict(m) for m in MODULE_CATALOG}


def _default_modules() -> List[str]:
    return [
        "intake_scope",
        "data_quality_gate",
        "feature_anomaly_analysis",
        "trust_ledger",
        "customer_report_bundle",
    ]


def _recommended_modules(problem: str, desired_outcome: str, data_readiness: str) -> List[str]:
    text = f"{problem} {desired_outcome} {data_readiness}".lower()
    mods = set(_default_modules())
    if any(x in text for x in ["maintenance", "failure", "downtime", "storing", "repair", "bearing", "motor"]):
        mods.add("maintenance_alignment")
    if any(x in text for x in ["excel", "json", "csv", "log", "plc", "history", "multiple", "several"]):
        mods.add("extra_data_source")
    if any(x in text for x in ["management", "board", "investor", "logo", "brand", "english", "dutch", "client"]):
        mods.add("branding_language")
    if any(x in text for x in ["energy", "security", "quality", "also", "ook", "second use", "another use"]):
        mods.add("extra_use_case_variant")
    if any(x in text for x in ["sensor", "gateway", "hardware", "bom", "node", "device"]):
        mods.add("hardware_bom_reference")
    if any(x in text for x in ["repeat", "template", "many", "reusable", "scale"]):
        mods.add("reusable_custom_template")
    return [m["module_id"] for m in MODULE_CATALOG if m["module_id"] in mods]


def _blocked_terms(text: str) -> List[str]:
    low = str(text or "").lower()
    return [term for term in BLOCK_TERMS if term in low]


def _select_base_pack(total_module_price: int, data_readiness: str, selected_count: int) -> str:
    if str(data_readiness).lower().startswith("real labelled") or total_module_price >= 5500:
        return "Custom Real-Data Evidence Pack"
    if total_module_price >= 6500 or selected_count >= 10:
        return "Premium Custom Pack"
    if total_module_price >= 2500 or selected_count >= 7:
        return "Custom Professional Pack"
    return "Custom Lite Pack"


def build_custom_pack_checkout_snapshot(
    company: str = "Customer",
    project_name: str = "EdgeTwin Custom Pack",
    problem: str = "We want to evaluate whether our machine data is ready for a pilot.",
    desired_outcome: str = "Predictive maintenance pilot",
    data_readiness: str = "Real data available",
    selected_module_ids: List[str] | None = None,
    payment_mode: str = "full_payment",
    provider: str = "Stripe/Paddle/manual-ready",
) -> Dict[str, Any]:
    """Create automatic custom-pack pricing, scope and payment handoff."""
    selected_module_ids = list(selected_module_ids or _recommended_modules(problem, desired_outcome, data_readiness))

    lookup = _catalog_lookup()
    selected_modules = [lookup[mid] for mid in selected_module_ids if mid in lookup]
    module_total = int(sum(int(m.get("price_eur", 0)) for m in selected_modules))
    base_pack = _select_base_pack(module_total, data_readiness, len(selected_modules))
    base_price = int(BASE_PACKS[base_pack]["base_price_eur"])

    # Avoid double counting the five foundation modules too harshly by crediting the standard base.
    foundation_credit = 1750 if base_pack != "Premium Custom Pack" else 2500
    subtotal = max(base_price, base_price + max(0, module_total - foundation_credit))

    blocked = []
    risky_terms = _blocked_terms(problem + " " + desired_outcome)
    if risky_terms:
        blocked.append({
            "reason": "Unsafe customer claim wording detected",
            "terms": risky_terms,
            "action": "Rewrite before payment/delivery. Do not promise production accuracy, certification or liability coverage.",
        })

    review_flags = []
    for m in selected_modules:
        automation = str(m.get("automation", "automatic"))
        if automation in {"automatic_when_non_safety_critical", "automatic_as_reference_only"}:
            review_flags.append({
                "module": m["label"],
                "reason": "Prepared automatically inside safe boundaries; blocked only if customer asks for safety/compliance/certified deployment promises.",
            })

    automatic_modules = [m for m in selected_modules if not str(m.get("automation", "")).startswith("optional_human")]
    auto_ready = not blocked
    deposit_pct = float(BASE_PACKS[base_pack]["deposit_pct"])
    if payment_mode == "deposit":
        amount_due_now = int(round(subtotal * deposit_pct))
    else:
        amount_due_now = subtotal

    checkout_status = "AUTO_CHECKOUT_READY" if auto_ready else "CHECKOUT_BLOCKED_UNTIL_SAFE_REWRITE"
    delivery_unlock = {
        "after_payment_confirmed": [
            "create_order",
            "unlock_intake_or_delivery",
            "generate_buyer_data_room",
            "prepare_secure_download_manifest",
        ] if auto_ready else [],
        "blocked_actions": [
            "refunds/disputes",
            "production accuracy guarantees",
            "legal/compliance/safety certification claims",
            "liability promises",
            "customer data deletion without controlled workflow",
        ],
    }

    value_summary = (
        f"{company} receives a {base_pack} with automatic scope, price, data-readiness route, "
        "safe claims and delivery handoff. If the data is not ready, the pack explains why, what is missing, "
        "and which lower-risk next step is recommended."
    )

    snapshot: Dict[str, Any] = {
        "module": MODULE,
        "created_at": _now(),
        "company": company,
        "project_name": project_name,
        "base_pack": base_pack,
        "problem": problem,
        "desired_outcome": desired_outcome,
        "data_readiness": data_readiness,
        "selected_modules": selected_modules,
        "automatic_modules": automatic_modules,
        "pricing": {
            "currency": "EUR",
            "base_price_eur": base_price,
            "module_total_eur": module_total,
            "foundation_credit_eur": foundation_credit,
            "subtotal_eur": subtotal,
            "payment_mode": payment_mode,
            "amount_due_now_eur": amount_due_now,
            "price_note": "Custom pricing is calculated from fixed module prices and safe scope boundaries.",
        },
        "provider": provider,
        "checkout_status": checkout_status,
        "auto_ready": auto_ready,
        "review_flags": review_flags,
        "blockers": blocked,
        "delivery_unlock": delivery_unlock,
        "safe_boundary": SAFE_BOUNDARY,
        "value_summary": value_summary,
        "customer_next_step": "Proceed to checkout" if auto_ready else "Rewrite unsafe claims or reduce scope before checkout",
    }
    snapshot["snapshot_hash"] = _sha(snapshot)
    return snapshot


def _pdf_bytes(snapshot: Dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 15)
    pdf.cell(0, 10, "EdgeTwin Custom Pack Checkout", ln=True)
    pdf.set_font("Arial", "", 10)
    lines = [
        f"Company: {snapshot.get('company')}",
        f"Project: {snapshot.get('project_name')}",
        f"Pack: {snapshot.get('base_pack')}",
        f"Status: {snapshot.get('checkout_status')}",
        f"Subtotal: EUR {snapshot.get('pricing', {}).get('subtotal_eur')}",
        f"Due now: EUR {snapshot.get('pricing', {}).get('amount_due_now_eur')}",
        "",
        "Value summary:",
        snapshot.get("value_summary", ""),
        "",
        "Selected modules:",
    ]
    for m in snapshot.get("selected_modules", []):
        lines.append(f"- {m.get('label')}: EUR {m.get('price_eur')} ({m.get('automation')})")
    lines.extend(["", "Safe boundary:", snapshot.get("safe_boundary", "")])
    for line in lines:
        pdf.set_x(10)
        pdf.multi_cell(190, 6, str(line).encode("latin-1", "replace").decode("latin-1"))
    out = pdf.output(dest="S")
    return bytes(out) if isinstance(out, bytearray) else (out if isinstance(out, bytes) else str(out).encode("latin-1", errors="replace"))


def create_custom_pack_checkout_bundle(snapshot: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("custom_pack_checkout_snapshot.json", json.dumps(snapshot, indent=2, ensure_ascii=False, default=str))
        z.writestr("custom_pack_checkout_summary.md", _markdown(snapshot))
        z.writestr("selected_modules.csv", pd.DataFrame(snapshot.get("selected_modules", [])).to_csv(index=False))
        z.writestr("review_flags.csv", pd.DataFrame(snapshot.get("review_flags", [])).to_csv(index=False))
        z.writestr("blockers.csv", pd.DataFrame(snapshot.get("blockers", [])).to_csv(index=False))
        z.writestr("custom_pack_checkout_summary.pdf", _pdf_bytes(snapshot))
    return buf.getvalue()


def _markdown(snapshot: Dict[str, Any]) -> str:
    pricing = snapshot.get("pricing", {})
    lines = [
        "# EdgeTwin Custom Pack Checkout",
        "",
        f"**Company:** {snapshot.get('company')}",
        f"**Project:** {snapshot.get('project_name')}",
        f"**Recommended pack:** {snapshot.get('base_pack')}",
        f"**Checkout status:** {snapshot.get('checkout_status')}",
        "",
        "## Price",
        f"- Subtotal: EUR {pricing.get('subtotal_eur')}",
        f"- Due now: EUR {pricing.get('amount_due_now_eur')}",
        f"- Payment mode: {pricing.get('payment_mode')}",
        "",
        "## Selected modules",
    ]
    for m in snapshot.get("selected_modules", []):
        lines.append(f"- {m.get('label')} — EUR {m.get('price_eur')} — {m.get('automation')}")
    lines.extend([
        "",
        "## Value Guarantee",
        "If the provided data is not ready for a controlled pilot, the customer receives a clear explanation of why, what is missing, what must be improved and which lower-risk next step is recommended.",
        "",
        "## Safe boundary",
        snapshot.get("safe_boundary", ""),
    ])
    if snapshot.get("blockers"):
        lines.append("\n## Blockers")
        for b in snapshot.get("blockers", []):
            lines.append(f"- {b}")
    return "\n".join(lines) + "\n"


def render_streamlit_tab(st) -> Tuple[Dict[str, Any] | None, bytes | None]:
    st.header("Custom Pack Checkout")
    st.caption("Automatic custom-pack pricing, safe scope and payment handoff. Standard custom packs stay automated unless unsafe claims or high-risk scope appear.")

    col1, col2 = st.columns(2)
    with col1:
        company = st.text_input("Company", value="Customer")
        project_name = st.text_input("Project name", value="EdgeTwin Custom Pack")
        desired_outcome = st.selectbox(
            "Desired outcome",
            ["Predictive maintenance pilot", "Sensor anomaly screening", "Data quality and feasibility check", "Operations evidence report", "Custom management report"],
            index=0,
        )
        data_readiness = st.selectbox(
            "Data readiness",
            ["No data yet", "Small sample available", "Real data available", "Real labelled data available", "Multiple data sources available"],
            index=2,
        )
    with col2:
        problem = st.text_area("Customer problem / use-case", value="We have machine data and want to know if it is ready for a controlled pilot.", height=140)
        payment_mode = st.radio("Payment mode", ["full_payment", "deposit"], horizontal=True)

    recommended = _recommended_modules(problem, desired_outcome, data_readiness)
    selected = []
    st.subheader("Modules")
    for m in MODULE_CATALOG:
        default = m["module_id"] in recommended
        checked = st.checkbox(f"{m['label']} — EUR {m['price_eur']}", value=default, key=f"checkout_mod_{m['module_id']}")
        st.caption(f"{m['customer_value']} Automation: {m['automation']}.")
        if checked:
            selected.append(m["module_id"])

    snapshot = build_custom_pack_checkout_snapshot(
        company=company,
        project_name=project_name,
        problem=problem,
        desired_outcome=desired_outcome,
        data_readiness=data_readiness,
        selected_module_ids=selected,
        payment_mode=payment_mode,
    )
    bundle = create_custom_pack_checkout_bundle(snapshot)

    st.subheader("Automatic quote")
    pricing = snapshot["pricing"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Recommended pack", snapshot["base_pack"])
    m2.metric("Subtotal", f"EUR {pricing['subtotal_eur']}")
    m3.metric("Due now", f"EUR {pricing['amount_due_now_eur']}")

    if snapshot["auto_ready"]:
        st.success("Automatic checkout ready inside safe policy boundaries.")
    else:
        st.error("Checkout blocked until unsafe claims/scope are fixed.")
        st.dataframe(pd.DataFrame(snapshot.get("blockers", [])), use_container_width=True)

    if snapshot.get("review_flags"):
        st.info("Some selected modules are automatically prepared with guardrails. They are only blocked if the customer asks for unsafe guarantees or certified/safety-critical promises.")
        st.dataframe(pd.DataFrame(snapshot["review_flags"]), use_container_width=True)

    st.markdown("### Customer value")
    st.write(snapshot["value_summary"])
    st.caption(SAFE_BOUNDARY)
    st.download_button("Download custom checkout bundle", data=bundle, file_name="edgetwin_custom_pack_checkout.zip", mime="application/zip")
    return snapshot, bundle
