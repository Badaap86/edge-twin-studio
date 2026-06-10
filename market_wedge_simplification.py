"""EdgeTwin Studio Market Wedge + First Customer Simplification.

Purpose:
- Keep the product powerful internally, but make the first customer story simple.
- Focus first sales on rotating assets / bearing & motor-health readiness.
- Show only three top-level packs, while synthetic expansion, hardware profile,
  firmware starter and OMEGA-X remain controlled route options/add-ons.
- Add pre-sale data/privacy/claim hygiene so the product is easier and safer to buy.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import zipfile
from typing import Any, Dict, List, Tuple

import pandas as pd

MODULE = "Market Wedge + First Customer Simplification"

POSITIONING = (
    "EdgeTwin Studio is an industrial AI readiness & evidence platform that helps manufacturers "
    "decide whether machine, sensor and maintenance data is ready for a controlled AI or "
    "predictive-maintenance pilot."
)

SAFE_BOUNDARY = (
    "EdgeTwin provides readiness evidence, data-quality insight, assumptions, limitations and next steps. "
    "It does not provide production accuracy guarantees, legal approval, compliance certification, "
    "safety certification, or automated machine-control permission."
)

SYNTHETIC_POLICY = (
    "Synthetic data may expand scenario coverage, stress-test assumptions and improve pilot preparation. "
    "It must not replace real customer validation or be presented as customer-specific production proof."
)

FIRST_WEDGE_USE_CASES = {
    "bearing_wear": {
        "label": "Bearing wear / lager-slijtage readiness",
        "why": "Most practical first wedge: rotating assets are common, vibration/audio signals are understandable, and privacy risk is lower than security/person-monitoring use-cases.",
        "required": ["vibration/accelerometer", "asset ID", "timestamps", "maintenance/failure labels where available"],
        "recommended": ["audio", "temperature", "RPM/load context", "maintenance history"],
    },
    "motor_health": {
        "label": "Motor / pump / fan health readiness",
        "why": "Good mid-market entry use-case with clear downtime pain and familiar maintenance language.",
        "required": ["vibration or existing condition data", "asset ID", "timestamps", "maintenance context"],
        "recommended": ["audio", "temperature", "RPM/load", "work orders / CMMS history"],
    },
    "existing_industrial_export": {
        "label": "Existing SCADA / historian / CMMS export readiness",
        "why": "Fastest first deal route when the customer already has usable exports and no hardware installation is needed.",
        "required": ["time-series export", "asset mapping", "timestamps", "field descriptions"],
        "recommended": ["maintenance labels", "failure events", "operating context", "data owner"],
    },
}

ADVANCED_USE_CASES = {
    "facade_security": "Security / facade sentinel: keep as custom/experimental due to privacy, safety and liability complexity.",
    "fire_arson_risk": "Fire/arson-risk context: keep as custom/experimental due to safety/compliance sensitivity.",
    "omega_x_field_kit": "OMEGA-X Field Data Kit: keep as controlled add-on; private core IP remains private.",
}

TOP_LEVEL_PACKS = {
    "starter": {
        "name": "Starter Diagnostic Pack",
        "guide_price": "€750 founding-customer guide",
        "role": "Low-friction first check for data readiness, missing inputs and safest next step.",
        "best_for": "Customer is unsure whether their data is usable or wants a quick first screen.",
    },
    "professional": {
        "name": "Professional Pilot Evidence Pack",
        "guide_price": "€1.500 founding-customer guide; later €2.500+",
        "role": "Main hero offer for first customers: management-ready pilot-readiness evidence and go/conditional/no-go route.",
        "best_for": "Customer has a clear asset/use-case and some data or a realistic route to data.",
    },
    "real_data": {
        "name": "Real-Data Evidence Pack",
        "guide_price": "€3.500 founding-customer guide; later €5.500+",
        "role": "Stronger evidence when real data, labels/context and asset mapping are available.",
        "best_for": "Customer wants a deeper pre-pilot decision before larger AI/platform spend.",
    },
}

ROUTE_OPTIONS = {
    "synthetic_expansion": "Synthetic Evidence Expansion: add scenario/stress coverage when real data is limited; never replace real validation.",
    "hardware_profile": "Hardware Profile Matrix: recommend the smallest reliable sensor set per use-case.",
    "firmware_starter": "Firmware Starter: approved pilot/data-collection templates only, not random production code.",
    "field_data_kit": "Field Data Kit / OMEGA-X route: controlled data collection add-on when no real data exists; core OMEGA-X IP stays private.",
}

PRE_SALE_HYGIENE = {
    "data_minimisation": "Only request data needed for readiness assessment; avoid personal data where possible.",
    "pseudonymisation": "Pseudonymise asset/operator/customer identifiers where possible before upload.",
    "nda_confidentiality": "Use NDA/confidentiality wording before sensitive customer exports.",
    "dpa_needed": "Use DPA/processing terms if personal data can be present.",
    "deletion_policy": "Define retention and deletion confirmation before receiving files.",
    "synth_policy": SYNTHETIC_POLICY,
    "claim_boundary": SAFE_BOUNDARY,
}


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def recommend_pack(data_situation: str, has_real_data: bool, has_labels: bool, has_context: bool, wants_hardware: bool) -> Tuple[str, str, List[str]]:
    """Return pack key, headline route, and controlled add-ons."""
    add_ons: List[str] = []
    if has_real_data and has_labels and has_context:
        return "real_data", "Real-data evidence route", add_ons
    if has_real_data:
        add_ons.append("synthetic_expansion")
        if not has_labels or not has_context:
            add_ons.append("hardware_profile")
        return "professional", "Professional evidence route with limited-data controls", add_ons
    # no real data or only hardware
    add_ons.extend(["hardware_profile", "firmware_starter"])
    if wants_hardware or data_situation in {"no_real_data", "hardware_only"}:
        add_ons.append("field_data_kit")
    return "starter", "Dataset starter / data-collection route", add_ons


def build_market_wedge_snapshot(
    company: str = "Demo Industrial Customer",
    use_case_key: str = "bearing_wear",
    data_situation: str = "limited_real_data",
    has_real_data: bool = True,
    has_labels: bool = False,
    has_context: bool = False,
    wants_hardware: bool = False,
    first_customer_slot: bool = True,
    notes: str = "",
) -> Dict[str, Any]:
    pack_key, route, add_on_keys = recommend_pack(data_situation, has_real_data, has_labels, has_context, wants_hardware)
    pack = TOP_LEVEL_PACKS[pack_key]
    use_case = FIRST_WEDGE_USE_CASES.get(use_case_key, FIRST_WEDGE_USE_CASES["bearing_wear"])

    blockers: List[str] = []
    if use_case_key not in FIRST_WEDGE_USE_CASES:
        blockers.append("Use-case is not part of the first-customer wedge; treat as custom/experimental.")
    if data_situation in {"security", "fire"}:
        blockers.append("Security/fire route requires custom risk review before external sale.")
    if not has_real_data:
        blockers.append("No customer-specific production evidence is possible without real field data.")
    if has_real_data and not has_labels:
        blockers.append("Failure/wear labels are limited or missing; avoid known-failure prediction claims.")
    if wants_hardware:
        blockers.append("Hardware/firmware starter must stay pilot/data-collection only; no production/safety certification.")

    customer_story = (
        "We focus first on rotating assets and bearing/motor-health readiness. "
        "The customer sees three simple packs. Advanced capabilities remain controlled route options under the hood."
    )

    snapshot = {
        "module": MODULE,
        "created_at": _now(),
        "company": company,
        "positioning": POSITIONING,
        "customer_story": customer_story,
        "first_wedge_use_case": use_case["label"],
        "why_this_wedge": use_case["why"],
        "data_situation": data_situation,
        "recommended_pack": pack["name"],
        "guide_price": pack["guide_price"],
        "pack_role": pack["role"],
        "route": route,
        "route_options": [ROUTE_OPTIONS[k] for k in add_on_keys],
        "top_level_packs": TOP_LEVEL_PACKS,
        "advanced_use_cases_policy": ADVANCED_USE_CASES,
        "required_inputs": use_case["required"],
        "recommended_inputs": use_case["recommended"],
        "pre_sale_hygiene": PRE_SALE_HYGIENE,
        "blockers_or_limits": blockers,
        "safe_boundary": SAFE_BOUNDARY,
        "synthetic_policy": SYNTHETIC_POLICY,
        "first_customer_slot": bool(first_customer_slot),
        "next_steps": [
            "Keep external story to three packs: Starter, Professional Pilot Evidence, Real-Data Evidence.",
            "Lead first outreach with bearing/motor-health readiness, not broad predictive-maintenance platform claims.",
            "Use sample Demo Evidence Room to show what the customer receives before asking for payment.",
            "Use NDA/DPA/deletion checklist before sensitive file exchange.",
            "Keep hardware/firmware/OMEGA-X as controlled add-ons until proven in field tests.",
        ],
        "notes": notes,
    }
    snapshot["decision"] = "FIRST-CUSTOMER READY WEDGE" if not blockers or all("No customer-specific" in b or "labels" in b or "pilot/data" in b for b in blockers) else "CUSTOM / REVIEW BEFORE SALE"
    snapshot["snapshot_hash"] = _sha(snapshot)
    return snapshot


def build_tables(snapshot: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    packs = pd.DataFrame([
        {"Pack": v["name"], "Guide price": v["guide_price"], "Role": v["role"], "Best for": v["best_for"]}
        for v in snapshot.get("top_level_packs", {}).values()
    ])
    route = pd.DataFrame([
        {"Item": "Positioning", "Value": snapshot.get("positioning")},
        {"Item": "First wedge", "Value": snapshot.get("first_wedge_use_case")},
        {"Item": "Recommended pack", "Value": snapshot.get("recommended_pack")},
        {"Item": "Guide price", "Value": snapshot.get("guide_price")},
        {"Item": "Route", "Value": snapshot.get("route")},
        {"Item": "Decision", "Value": snapshot.get("decision")},
    ])
    add_ons = pd.DataFrame([{"Controlled route option": x} for x in snapshot.get("route_options", [])] or [{"Controlled route option": "No add-on needed first."}])
    hygiene = pd.DataFrame([{"Rule": k.replace("_", " ").title(), "Text": v} for k, v in snapshot.get("pre_sale_hygiene", {}).items()])
    limits = pd.DataFrame([{"Limit / blocker": x} for x in snapshot.get("blockers_or_limits", [])] or [{"Limit / blocker": "No hard blocker for first-customer readiness sale."}])
    return {"packs": packs, "route": route, "add_ons": add_ons, "hygiene": hygiene, "limits": limits}


def create_market_wedge_bundle(snapshot: Dict[str, Any]) -> bytes:
    tables = build_tables(snapshot)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(snapshot, indent=2, ensure_ascii=False, default=str))
        for name, df in tables.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False))
        zf.writestr("FIRST_CUSTOMER_POSITIONING.md", f"""# EdgeTwin first-customer positioning\n\n{snapshot.get('positioning')}\n\n## First wedge\n\n{snapshot.get('first_wedge_use_case')}\n\n{snapshot.get('why_this_wedge')}\n\n## Customer-facing offer\n\nShow three packs only:\n\n- Starter Diagnostic Pack\n- Professional Pilot Evidence Pack\n- Real-Data Evidence Pack\n\nSynthetic expansion, hardware profile, firmware starter and Field Data Kit remain controlled add-ons under the hood.\n\n## Safe boundary\n\n{snapshot.get('safe_boundary')}\n""")
        zf.writestr("SYNTHETIC_DATA_POLICY.md", f"# Synthetic data policy\n\n{SYNTHETIC_POLICY}\n")
        zf.writestr("DATA_PRIVACY_INTAKE_CHECKLIST.md", "# Data privacy intake checklist\n\n" + "\n".join(f"- {v}" for v in PRE_SALE_HYGIENE.values()))
    return buf.getvalue()


def render_streamlit_tab(st) -> Tuple[Dict[str, Any] | None, bytes | None]:
    st.header("First Customer Focus")
    st.caption("Keep the outside simple: three packs, one first wedge, controlled add-ons behind the scenes.")

    with st.form("market_wedge_form"):
        c1, c2 = st.columns(2)
        with c1:
            company = st.text_input("Customer / company", "Demo Industrial Customer", key="v141_company")
            use_case_key = st.selectbox("First-customer wedge", list(FIRST_WEDGE_USE_CASES.keys()), format_func=lambda k: FIRST_WEDGE_USE_CASES[k]["label"], index=0, key="v141_wedge")
            data_situation = st.selectbox("Data situation", ["good_real_data", "limited_real_data", "no_real_data", "hardware_only"], index=1, key="v141_data_situation")
            notes = st.text_area("Notes", "Focus first offer on bearing/motor-health readiness. Keep advanced options as controlled add-ons.", height=100, key="v141_notes")
        with c2:
            has_real_data = st.checkbox("Customer has real machine/sensor/maintenance data", value=True, key="v141_real")
            has_labels = st.checkbox("Customer has useful wear/failure/maintenance labels", value=False, key="v141_labels")
            has_context = st.checkbox("Customer has RPM/load/operating context", value=False, key="v141_context")
            wants_hardware = st.checkbox("Hardware/firmware route may be needed", value=False, key="v141_hardware")
            first_customer_slot = st.checkbox("Founding customer slot available", value=True, key="v141_founder_slot")
        submitted = st.form_submit_button("Build first-customer focus", use_container_width=True)

    if not submitted:
        st.info("This page is the commercial simplifier: it keeps EdgeTwin powerful inside, but easy to understand outside.")
        st.markdown("### Customer-facing top-level packs")
        st.dataframe(pd.DataFrame([
            {"Pack": v["name"], "Guide price": v["guide_price"], "Role": v["role"]}
            for v in TOP_LEVEL_PACKS.values()
        ]), use_container_width=True)
        st.warning(SAFE_BOUNDARY)
        return None, None

    snapshot = build_market_wedge_snapshot(
        company=company,
        use_case_key=use_case_key,
        data_situation=data_situation,
        has_real_data=has_real_data,
        has_labels=has_labels,
        has_context=has_context,
        wants_hardware=wants_hardware,
        first_customer_slot=first_customer_slot,
        notes=notes,
    )
    bundle = create_market_wedge_bundle(snapshot)
    tables = build_tables(snapshot)

    m1, m2, m3 = st.columns(3)
    m1.metric("Recommended pack", snapshot.get("recommended_pack"))
    m2.metric("Guide price", snapshot.get("guide_price"))
    m3.metric("Decision", snapshot.get("decision"))

    if snapshot.get("decision") == "FIRST-CUSTOMER READY WEDGE":
        st.success(snapshot.get("decision"))
    else:
        st.warning(snapshot.get("decision"))

    tabs = st.tabs(["Route", "3 packs", "Controlled add-ons", "Data/privacy", "Limits", "Download"])
    with tabs[0]:
        st.dataframe(tables["route"], use_container_width=True)
        st.info(snapshot.get("why_this_wedge"))
    with tabs[1]:
        st.dataframe(tables["packs"], use_container_width=True)
    with tabs[2]:
        st.dataframe(tables["add_ons"], use_container_width=True)
    with tabs[3]:
        st.dataframe(tables["hygiene"], use_container_width=True)
    with tabs[4]:
        st.dataframe(tables["limits"], use_container_width=True)
        st.warning(snapshot.get("safe_boundary"))
        st.info(snapshot.get("synthetic_policy"))
    with tabs[5]:
        st.download_button("Download first-customer focus bundle", bundle, file_name="edgetwin_first_customer_focus_bundle.zip", mime="application/zip", use_container_width=True)

    return snapshot, bundle
