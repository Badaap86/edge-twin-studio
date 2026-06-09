"""EdgeTwin Studio V118 Guided Custom Customer Builder.

Purpose:
- Put the Custom Pack Builder inside the customer-facing flow.
- Let customers configure what they need without turning custom work into founder chaos.
- Reuse the V97 custom-pack calculation, then add customer-facing questions, scope lock,
  price explanation, self-service/approval decisions and reusable-template guidance.

Boundary:
- V118 can recommend modules, calculate scope/price/risk and prepare a safe custom pack bundle.
- It does not process payments, sign contracts, certify compliance, guarantee production accuracy,
  or auto-approve high-risk custom promises.
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

VERSION = "118.0"
MODULE = "Guided Custom Customer Builder"

SAFE_BOUNDARY = (
    "EdgeTwin custom packs are pilot/evidence and decision-support deliverables. They do not guarantee "
    "production accuracy, legal/compliance certification, safety certification, or replacement of human review."
)

# Mirror the V97 commercial module language so the customer-facing route stays consistent.
MODULES = [
    {"module_id": "intake_scope", "label": "Use-case intake + scope lock", "category": "foundation", "price_eur": 350, "complexity": 1, "default_selected": True, "auto_approvable": True, "deliverable": "Structured intake summary, scope boundary and next-step map."},
    {"module_id": "data_quality_gate", "label": "Data quality gate", "category": "assurance", "price_eur": 650, "complexity": 2, "default_selected": True, "auto_approvable": True, "deliverable": "Data readiness score, missing inputs and limits before strong claims."},
    {"module_id": "feature_anomaly_analysis", "label": "Feature/anomaly analysis", "category": "engine", "price_eur": 950, "complexity": 3, "default_selected": True, "auto_approvable": True, "deliverable": "Core EdgeTwin analysis path with signal/data features and anomaly evidence."},
    {"module_id": "trust_ledger", "label": "Trust ledger + limitations", "category": "trust", "price_eur": 550, "complexity": 2, "default_selected": True, "auto_approvable": True, "deliverable": "Trust score, assumptions, blockers, safe claims and limitations."},
    {"module_id": "customer_report_bundle", "label": "PDF/ZIP customer report bundle", "category": "delivery", "price_eur": 550, "complexity": 1, "default_selected": True, "auto_approvable": True, "deliverable": "Customer-safe PDF/ZIP/JSON/CSV/Markdown delivery bundle."},
    {"module_id": "maintenance_alignment", "label": "Maintenance history alignment", "category": "advanced", "price_eur": 900, "complexity": 3, "default_selected": False, "auto_approvable": True, "deliverable": "Aligns available machine/sensor data with maintenance or failure history."},
    {"module_id": "extra_data_source", "label": "Extra data source/file type", "category": "data", "price_eur": 500, "complexity": 2, "default_selected": False, "auto_approvable": True, "deliverable": "Adds one extra standard source such as CSV, Excel, JSON or maintenance export."},
    {"module_id": "branding_language", "label": "Branding/language pack", "category": "delivery", "price_eur": 450, "complexity": 1, "default_selected": False, "auto_approvable": True, "deliverable": "Customer name/branding and selected language/tone for report outputs."},
    {"module_id": "hardware_bom_reference", "label": "Hardware/BOM reference module", "category": "hardware", "price_eur": 1250, "complexity": 4, "default_selected": False, "auto_approvable": False, "deliverable": "Sensor/gateway/BOM reference guidance with deployment cautions."},
    {"module_id": "extra_use_case_variant", "label": "Extra use-case variant", "category": "custom", "price_eur": 1400, "complexity": 4, "default_selected": False, "auto_approvable": False, "deliverable": "Adds one additional use-case route with its own risks and acceptance criteria."},
    {"module_id": "founder_review_call", "label": "Founder review call", "category": "service", "price_eur": 500, "complexity": 1, "default_selected": False, "auto_approvable": False, "deliverable": "Founder/operator review call with action list and upgrade recommendation."},
    {"module_id": "reusable_custom_template", "label": "Reusable custom pack template", "category": "productization", "price_eur": 2500, "complexity": 5, "default_selected": False, "auto_approvable": False, "deliverable": "Turns the custom configuration into a reusable future pack recipe."},
]

INDUSTRIES = ["Industrial / maintenance", "Manufacturing", "Energy / utilities", "Construction / assets", "Remote assets / forestry", "Security / monitoring", "Other / unknown"]
OUTCOMES = ["Predictive maintenance pilot", "Sensor anomaly screening", "Data quality and feasibility check", "Operations evidence report", "Hardware/sensor reference pack", "Custom management report"]
DATA_READINESS = ["No data yet", "Small sample available", "Real data available", "Real labelled data available", "Multiple data sources available"]
BUDGET_MODES = ["Under €1.000", "€1.000–€3.500", "€3.500–€7.500", "€7.500+", "Auto"]
URGENCIES = ["Normal", "Fast", "Very urgent"]

RISKY_TERMS = [
    "100%", "guarantee", "guaranteed", "production-ready", "production ready", "certified", "legal approved",
    "compliance approved", "liability", "replace human", "without validation", "always predict",
]


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def _module_lookup() -> Dict[str, Dict[str, Any]]:
    return {m["module_id"]: dict(m) for m in MODULES}


def _default_module_ids() -> List[str]:
    return [m["module_id"] for m in MODULES if m.get("default_selected")]


def _safe_pdf_text(value: Any) -> str:
    return str(value or "").encode("latin-1", "replace").decode("latin-1")


def _pdf_bytes(pdf: FPDF) -> bytes:
    out = pdf.output(dest="S")
    if isinstance(out, bytearray):
        return bytes(out)
    if isinstance(out, bytes):
        return out
    return str(out).encode("latin-1", errors="replace")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [_json_safe(x) for x in obj]
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    return obj


def _recommend_modules(customer_problem: str, industry: str, desired_outcome: str, selected_ids: List[str]) -> List[str]:
    text = f"{customer_problem} {industry} {desired_outcome}".lower()
    recommended = set(selected_ids or _default_module_ids())
    if any(k in text for k in ["maintenance", "failure", "downtime", "storing", "bearing", "motor", "machine"]):
        recommended.add("maintenance_alignment")
    if any(k in text for k in ["excel", "json", "csv", "log", "plc", "history", "historie", "multiple", "several"]):
        recommended.add("extra_data_source")
    if any(k in text for k in ["management", "board", "investor", "klant", "customer", "english", "dutch", "brand", "logo"]):
        recommended.add("branding_language")
    if any(k in text for k in ["sensor", "gateway", "hardware", "bom", "node", "rak", "edge device"]):
        recommended.add("hardware_bom_reference")
    if any(k in text for k in ["extra use", "also", "ook", "energy", "energie", "security", "forestry"]):
        recommended.add("extra_use_case_variant")
    if any(k in text for k in ["template", "repeat", "reusable", "herbruikbaar", "many customers"]):
        recommended.add("reusable_custom_template")
    return [m["module_id"] for m in MODULES if m["module_id"] in recommended]


def _risk_terms(customer_problem: str) -> List[str]:
    text = str(customer_problem or "").lower()
    return [term for term in RISKY_TERMS if term in text]


def _customer_questions(selected_modules: List[Dict[str, Any]], data_readiness: str) -> List[Dict[str, Any]]:
    mids = {m.get("module_id") for m in selected_modules}
    q = [
        {"question": "What is the main machine/asset/process you want EdgeTwin to evaluate?", "why_it_matters": "Locks the scope and prevents vague custom work.", "required_for": "all_custom_packs"},
        {"question": "Can you upload a small CSV/Excel sample or describe the columns you have?", "why_it_matters": "Enables the Data Quality Gate and quote confidence.", "required_for": "data_quality_gate"},
        {"question": "What outcome would make this pack worth paying for?", "why_it_matters": "Connects the deliverables to a business decision.", "required_for": "all_custom_packs"},
    ]
    if "maintenance_alignment" in mids:
        q.append({"question": "Do you have maintenance/failure history with dates or downtime notes?", "why_it_matters": "Needed to align signals with real events.", "required_for": "maintenance_alignment"})
    if "extra_data_source" in mids:
        q.append({"question": "Which extra file type/source should be included first?", "why_it_matters": "Keeps the custom parser scope bounded.", "required_for": "extra_data_source"})
    if "hardware_bom_reference" in mids:
        q.append({"question": "Is hardware guidance only a reference, or do you need certified deployment engineering?", "why_it_matters": "Certified engineering cannot be auto-approved.", "required_for": "hardware_bom_reference"})
    if "extra_use_case_variant" in mids:
        q.append({"question": "Which second use-case is most important, and what should be excluded?", "why_it_matters": "Prevents one custom pack from becoming unlimited consulting.", "required_for": "extra_use_case_variant"})
    if "branding_language" in mids:
        q.append({"question": "Which language, company name and report audience should be used?", "why_it_matters": "Makes the output manager/customer ready.", "required_for": "branding_language"})
    if str(data_readiness) in {"No data yet", "Small sample available"}:
        q.append({"question": "Can we start with a demo/synthetic flow while you prepare real data?", "why_it_matters": "Avoids overclaiming when real data is not ready.", "required_for": "low_data_readiness"})
    return q


def _score(v97: Dict[str, Any], blockers: List[str], review_flags: List[str], selected_modules: List[Dict[str, Any]]) -> int:
    base = int(v97.get("auto_build_score", 70) or 70)
    base += 8 if len(selected_modules) >= 5 else 2
    base -= min(25, len(review_flags) * 4)
    base -= min(45, len(blockers) * 15)
    return int(max(0, min(100, base)))


def build_guided_custom_customer_builder_snapshot(
    project_name: str = "EdgeTwin Project",
    company: str = "Customer",
    industry: str = "Industrial / maintenance",
    desired_outcome: str = "Predictive maintenance pilot",
    customer_problem: str = "We have machine data and want a custom pilot/evidence pack.",
    data_readiness: str = "Real data available",
    budget_mode: str = "€3.500–€7.500",
    urgency: str = "Normal",
    selected_module_ids: List[str] | None = None,
    wants_branding: bool = False,
    wants_hardware: bool = False,
    wants_reusable_template: bool = False,
    allow_auto_approval: bool = True,
) -> Dict[str, Any]:
    """Build a customer-facing custom configurator snapshot."""
    selected_module_ids = list(selected_module_ids or _default_module_ids())
    recommended_ids = _recommend_modules(customer_problem, industry, desired_outcome, selected_module_ids)
    if wants_branding and "branding_language" not in recommended_ids:
        recommended_ids.append("branding_language")
    if wants_hardware and "hardware_bom_reference" not in recommended_ids:
        recommended_ids.append("hardware_bom_reference")
    if wants_reusable_template and "reusable_custom_template" not in recommended_ids:
        recommended_ids.append("reusable_custom_template")

    # Use V97 as the price/scope engine so custom logic stays centralized.
    import core  # local import avoids app import cycles
    v97 = core.build_custom_pack_builder_v97_snapshot({
        "project_name": project_name,
        "company": company,
        "industry": industry,
        "desired_outcome": desired_outcome,
        "customer_note": customer_problem,
        "data_readiness": data_readiness,
        "budget_mode": budget_mode,
        "urgency": urgency,
        "selected_module_ids": recommended_ids,
        "wants_branding": wants_branding,
        "wants_hardware": wants_hardware,
        "wants_reusable_template": wants_reusable_template,
        "allow_auto_approval": allow_auto_approval,
        "snapshots": {"policy_approval_engine_v95": {"policy_approval_score": 92}, "pricing_assurance_os_v96": {"reliability_score": 84}},
    })

    selected_modules = list(v97.get("selected_modules") or [])
    risky = _risk_terms(customer_problem)
    blockers: List[str] = []
    review_flags: List[str] = list(v97.get("founder_review_items") or [])
    if risky:
        blockers.append("Customer wording contains risky guarantee/compliance/production language that must be rewritten before quote/send.")
    if v97.get("data_readiness_score", 0) < 55:
        blockers.append("Data readiness is too weak for automatic custom pilot/evidence approval.")
    if any(not bool(m.get("auto_approvable")) for m in selected_modules):
        review_flags.append({"item": "Advanced custom module selected", "reason": "Non-standard modules are prepared automatically but not blindly auto-approved."})

    auto_send_allowed = bool(v97.get("pack_config", {}).get("safe_to_auto_send")) and not blockers
    score = _score(v97, blockers, review_flags, selected_modules)
    if auto_send_allowed and score >= 88:
        decision = "GUIDED CUSTOM PACK AUTO-READY INSIDE POLICY"
        customer_visible_decision = "Your custom pack can be prepared automatically inside the safe policy boundary."
    elif blockers:
        decision = "CUSTOM REQUEST BLOCKED UNTIL SAFE REWRITE / DATA FIX"
        customer_visible_decision = "Your request is useful, but it needs a safety/data fix before EdgeTwin can prepare the quote automatically."
    else:
        decision = "GUIDED CUSTOM PACK READY - EXCEPTION REVIEW ONLY"
        customer_visible_decision = "Your custom pack is prepared; only the highlighted exceptions need founder/operator review."

    scope_lock = {
        "included": [m.get("label") for m in selected_modules],
        "excluded": [
            "production accuracy guarantees",
            "legal/compliance certification",
            "certified hardware/safety engineering",
            "unlimited extra use-cases or data sources",
            "customer data reuse without explicit consent",
        ],
        "change_rule": "Adding modules changes price/scope. Discounting should remove scope, not hidden work.",
    }
    questions = _customer_questions(selected_modules, data_readiness)

    snapshot = {
        "version": VERSION,
        "module": MODULE,
        "created_at": _now(),
        "project_name": project_name,
        "company": company,
        "industry": industry,
        "desired_outcome": desired_outcome,
        "customer_problem": customer_problem,
        "customer_problem_hash": _sha(customer_problem),
        "decision": decision,
        "customer_visible_decision": customer_visible_decision,
        "custom_customer_score": score,
        "auto_send_allowed": auto_send_allowed,
        "price_display": v97.get("price_display"),
        "min_price_eur": v97.get("min_price_eur"),
        "max_price_eur": v97.get("max_price_eur"),
        "founder_work_minutes": v97.get("founder_work_minutes"),
        "pack_name": v97.get("pack_name"),
        "data_readiness": data_readiness,
        "data_readiness_score": v97.get("data_readiness_score"),
        "budget_mode": budget_mode,
        "urgency": urgency,
        "selected_modules": selected_modules,
        "recommended_module_ids": recommended_ids,
        "pricing_lines": v97.get("pricing_lines", []),
        "deliverables": v97.get("deliverables", []),
        "acceptance_criteria": v97.get("acceptance_criteria", []),
        "reusable_template": v97.get("reusable_template", {}),
        "scope_lock": scope_lock,
        "customer_questions": questions,
        "review_flags": review_flags,
        "blockers": blockers,
        "risky_terms_found": risky,
        "auto_approval_boundary": {
            "allowed": "standard modules, safe claims, data readiness >= 55, policy score strong, quote under cap, no blockers",
            "review_only": "hardware/BOM, extra use-case, reusable template, founder call, high complexity, budget mismatch",
            "blocked": "production/accuracy/legal/compliance guarantees, low data readiness, unsafe customer wording",
        },
        "customer_page": {
            "intro": "Build your own EdgeTwin pack by selecting what you need. EdgeTwin calculates scope, price, risks and deliverables before you pay.",
            "safe_boundary": SAFE_BOUNDARY,
            "primary_cta": "Build my custom pack",
            "secondary_cta": "Start with a standard pack instead",
        },
        "customer_copy": {
            "safe_summary": f"{company} configured a {v97.get('pack_name')} for {desired_outcome}. EdgeTwin prepared scope, price and deliverables with clear limitations and no production guarantee.",
            "price_explanation": "The price is based on selected modules, data readiness, complexity and delivery urgency.",
            "next_step": "Auto-send if policy-approved; otherwise review only the highlighted exception items.",
        },
        "v97_source_snapshot": v97,
    }
    return snapshot


def build_v118_module_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for m in snapshot.get("selected_modules", []):
        rows.append({
            "module": m.get("label"),
            "category": m.get("category"),
            "price_eur": m.get("price_eur"),
            "complexity": m.get("complexity"),
            "auto_approvable": m.get("auto_approvable"),
            "deliverable": m.get("deliverable"),
        })
    return pd.DataFrame(rows)


def build_v118_pricing_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(snapshot.get("pricing_lines", []))


def build_v118_question_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(snapshot.get("customer_questions", []))


def create_guided_custom_customer_builder_bundle(snapshot: Dict[str, Any]) -> bytes:
    snapshot = snapshot or {}
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("guided_custom_customer_builder.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr("customer_custom_pack_summary_v118.md", "\n".join([
            f"# {snapshot.get('pack_name', 'EdgeTwin Custom Pack')} — {snapshot.get('company', 'Customer')}",
            "",
            f"**Decision:** {snapshot.get('decision')}",
            f"**Price:** {snapshot.get('price_display')}",
            f"**Founder work:** {snapshot.get('founder_work_minutes')} minutes",
            "",
            "## What is included",
            *[f"- {m.get('label')}: {m.get('deliverable')}" for m in snapshot.get('selected_modules', [])],
            "",
            "## Safe boundary",
            SAFE_BOUNDARY,
        ]))
        for name, rows in [
            ("selected_modules_v118.csv", snapshot.get("selected_modules", [])),
            ("pricing_lines_v118.csv", snapshot.get("pricing_lines", [])),
            ("deliverables_v118.csv", snapshot.get("deliverables", [])),
            ("customer_questions_v118.csv", snapshot.get("customer_questions", [])),
            ("review_flags_v118.csv", snapshot.get("review_flags", [])),
        ]:
            if rows:
                zf.writestr(name, pd.DataFrame(rows).to_csv(index=False))
        zf.writestr("scope_lock_v118.json", json.dumps(_json_safe(snapshot.get("scope_lock", {})), indent=2, ensure_ascii=False))
        zf.writestr("reusable_template_v118.json", json.dumps(_json_safe(snapshot.get("reusable_template", {})), indent=2, ensure_ascii=False))
        zf.writestr("auto_approval_boundary_v118.json", json.dumps(_json_safe(snapshot.get("auto_approval_boundary", {})), indent=2, ensure_ascii=False))

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 8, _safe_pdf_text("EdgeTwin Studio V118 - Guided Custom Customer Builder"), ln=True)
        pdf.set_font("Arial", size=10)
        for line in [
            f"Customer: {snapshot.get('company')}",
            f"Pack: {snapshot.get('pack_name')}",
            f"Decision: {snapshot.get('decision')}",
            f"Price: {snapshot.get('price_display')}",
            f"Founder work: {snapshot.get('founder_work_minutes')} minutes",
        ]:
            pdf.cell(0, 6, _safe_pdf_text(line), ln=True)
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "Selected modules", ln=True)
        pdf.set_font("Arial", size=9)
        for m in snapshot.get("selected_modules", [])[:12]:
            pdf.set_x(10)
            pdf.multi_cell(190, 5, _safe_pdf_text(f"- {m.get('label')}: {m.get('deliverable')}"))
        pdf.ln(2)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "Safe boundary", ln=True)
        pdf.set_font("Arial", size=9)
        pdf.set_x(10)
        pdf.multi_cell(190, 5, _safe_pdf_text(SAFE_BOUNDARY))
        zf.writestr("guided_custom_customer_builder.pdf", _pdf_bytes(pdf))
    mem.seek(0)
    return mem.getvalue()
