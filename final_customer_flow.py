"""EdgeTwin Studio Ultimate Customer Flow.

Purpose:
- Give the customer one clean path from situation -> route -> pack -> evidence -> quote.
- Keep the product self-selling without production, legal, compliance or safety guarantees.
- Combine readiness, data coverage, hardware profile, firmware starter and evidence-room logic into one customer-safe summary.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import zipfile
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    import readiness_snapshot
except Exception:  # pragma: no cover
    readiness_snapshot = None

try:
    import data_coverage_engine
except Exception:  # pragma: no cover
    data_coverage_engine = None

try:
    import hardware_profile_matrix
except Exception:  # pragma: no cover
    hardware_profile_matrix = None

try:
    import demo_evidence_room
except Exception:  # pragma: no cover
    demo_evidence_room = None

MODULE = "Ultimate Customer Flow"

SAFE_BOUNDARY = (
    "EdgeTwin provides pilot-readiness evidence, data-quality insight, limitations and next steps. "
    "It does not provide production-accuracy guarantees, legal approval, compliance certification, "
    "safety certification or permission to run automated machine control without customer-specific validation."
)

DATA_SITUATIONS = {
    "good_real_data": "We have usable real data",
    "limited_real_data": "We have limited real data",
    "no_real_data": "We have no real data yet",
    "hardware_only": "We have hardware, but no clean dataset",
}

USE_CASES = {
    "bearing_wear": "Bearing wear / lager-slijtage",
    "machine_anomaly": "General machine anomaly",
    "facade_security": "Facade / site security sentinel",
    "fire_arson_risk": "Fire / arson-risk context",
    "existing_industrial_export": "Existing industrial data export",
}

SIGNALS = {
    "vibration_accel": "Vibration / accelerometer",
    "audio_i2s": "Audio / I2S microphone",
    "temperature": "Temperature",
    "rpm_load": "RPM / load context",
    "maintenance_labels": "Maintenance/failure labels",
    "radar": "Radar / presence/distance",
    "bme688_gas": "BME688 gas/environment",
    "pmu_power": "PMU / power health",
    "gps_asset": "GPS / asset/location context",
    "lora_gateway": "LoRa/gateway connectivity",
    "existing_export": "Existing PLC/SCADA/historian/CMMS export",
}

PACK_PRICE_GUIDE = {
    "Starter Diagnostic Pack": "€750 founding-customer guide",
    "Professional Pilot Pack": "€1.500 founding-customer guide",
    "Real-Data Evidence Pack": "€3.500 founding-customer guide",
    "Real Data + Synthetic Evidence Expansion": "€2.500–€5.500 guide",
    "Dataset Starter / Hardware Profile / Field Data Kit": "€950–€7.500+ depending on scope",
    "Hardware Profile + Firmware Starter Pack": "€750–€3.500 guide",
}


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def _safe_route(data_situation_key: str, has_real_data: bool, enough_labels: bool, customer_ready_to_collect: bool) -> Tuple[str, str, str]:
    if data_situation_key == "good_real_data" and has_real_data and enough_labels:
        return "real_data_evidence", "Real-Data Evidence Pack", "Run Data Quality Gate and prepare evidence room."
    if data_situation_key in {"limited_real_data", "good_real_data"} and has_real_data:
        return "synthetic_expansion", "Real Data + Synthetic Evidence Expansion", "Use real data as evidence basis; use synthetic data only for scenario coverage and stress testing."
    if data_situation_key == "hardware_only" or customer_ready_to_collect:
        return "firmware_starter", "Hardware Profile + Firmware Starter Pack", "Confirm approved hardware profile, generate starter firmware/config and collect EdgeTwin-compatible data."
    return "dataset_starter", "Dataset Starter / Hardware Profile / Field Data Kit", "Define what must be measured before any predictive-maintenance claim."


def _recommended_signals(use_case_key: str) -> Dict[str, List[str]]:
    fallback = {
        "required": ["vibration_accel", "audio_i2s"],
        "recommended": ["temperature", "rpm_load", "maintenance_labels"],
        "optional": ["lora_gateway"],
    }
    if hardware_profile_matrix is not None:
        matrix = getattr(hardware_profile_matrix, "USE_CASE_MATRIX", {})
        row = matrix.get(use_case_key)
        if row:
            return {
                "required": row.get("required", []),
                "recommended": row.get("recommended", []),
                "optional": row.get("optional", []),
            }
    if use_case_key == "facade_security":
        return {"required": ["radar", "audio_i2s", "vibration_accel", "pmu_power"], "recommended": ["bme688_gas", "lora_gateway"], "optional": ["gps_asset"]}
    if use_case_key == "fire_arson_risk":
        return {"required": ["bme688_gas", "temperature", "audio_i2s"], "recommended": ["pmu_power", "lora_gateway"], "optional": ["radar"]}
    if use_case_key == "existing_industrial_export":
        return {"required": ["existing_export", "maintenance_labels"], "recommended": ["rpm_load", "temperature"], "optional": ["vibration_accel", "audio_i2s"]}
    return fallback


def _coverage(required: List[str], recommended: List[str], optional: List[str], available_signals: List[str]) -> Dict[str, Any]:
    available = set(available_signals or [])
    req_hit = [x for x in required if x in available]
    rec_hit = [x for x in recommended if x in available]
    opt_hit = [x for x in optional if x in available]
    score = 0
    if required:
        score += int(55 * len(req_hit) / len(required))
    if recommended:
        score += int(35 * len(rec_hit) / len(recommended))
    if optional:
        score += int(10 * len(opt_hit) / len(optional))
    missing_required = [x for x in required if x not in available]
    missing_recommended = [x for x in recommended if x not in available]
    return {
        "score": min(100, score),
        "required_present": req_hit,
        "recommended_present": rec_hit,
        "optional_present": opt_hit,
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
    }


def _signal_names(keys: List[str]) -> List[str]:
    return [SIGNALS.get(k, k) for k in keys]


def build_ultimate_customer_flow_snapshot(
    company: str = "Demo Industrial Customer",
    data_situation_key: str = "limited_real_data",
    use_case_key: str = "bearing_wear",
    available_signals: List[str] | None = None,
    has_real_data: bool = True,
    enough_labels: bool = False,
    customer_ready_to_collect: bool = True,
    wants_firmware_starter: bool = True,
    wants_demo_room: bool = True,
    customer_notes: str = "",
) -> Dict[str, Any]:
    available_signals = available_signals or []
    route_key, recommended_pack, next_step = _safe_route(data_situation_key, has_real_data, enough_labels, customer_ready_to_collect)
    signals = _recommended_signals(use_case_key)
    coverage = _coverage(signals["required"], signals["recommended"], signals["optional"], available_signals)

    if route_key in {"firmware_starter", "dataset_starter"} and wants_firmware_starter:
        cta = "Request hardware profile + firmware starter"
    elif route_key == "synthetic_expansion":
        cta = "Request Professional Pilot Pack with synthetic scenario expansion"
    else:
        cta = "Request Real-Data Evidence Pack"

    trust_reasons = [
        "Real data remains the evidence basis whenever available.",
        "Synthetic data is used only for scenario expansion, stress testing and pilot preparation.",
        "Public benchmark data is used for demonstration/reference, not customer-specific proof.",
        "Approved firmware templates are pilot/data-collection starters, not production-certified control firmware.",
    ]

    blockers: List[str] = []
    if route_key == "real_data_evidence" and coverage["missing_required"]:
        blockers.append("Some required signals/fields are missing for a strong real-data evidence pack.")
    if not has_real_data and route_key != "firmware_starter":
        blockers.append("No customer-specific production evidence is possible without real field data.")
    if not enough_labels and use_case_key in {"bearing_wear", "machine_anomaly", "existing_industrial_export"}:
        blockers.append("Failure/wear/maintenance labels are limited or missing; do not claim known failure prediction yet.")

    hardware_preview = None
    if hardware_profile_matrix is not None:
        try:
            hardware_preview = hardware_profile_matrix.build_hardware_profile_snapshot(
                company=company,
                data_situation_key=data_situation_key,
                use_case_key=use_case_key,
                available_signals=available_signals,
                customer_confirms_hardware=False,
                wants_firmware_starter=wants_firmware_starter,
            )
        except Exception as exc:
            hardware_preview = {"error": str(exc)}

    readiness_preview = None
    if readiness_snapshot is not None:
        try:
            readiness_preview = readiness_snapshot.build_snapshot(
                company=company,
                use_case_key=use_case_key,
                available_signals=available_signals,
                real_data=has_real_data,
                timestamps=("existing_export" in available_signals or has_real_data),
                asset_ids=True,
                signal_quality=bool(available_signals),
                history=has_real_data,
                labels=enough_labels,
                operating_context=("rpm_load" in available_signals),
                privacy_ready=True,
                business_owner=True,
                notes=customer_notes,
            )
        except Exception as exc:
            readiness_preview = {"error": str(exc)}

    snapshot = {
        "module": MODULE,
        "created_at": _now(),
        "company": company,
        "data_situation_key": data_situation_key,
        "data_situation": DATA_SITUATIONS.get(data_situation_key, data_situation_key),
        "use_case_key": use_case_key,
        "use_case": USE_CASES.get(use_case_key, use_case_key),
        "available_signals": available_signals,
        "available_signal_names": _signal_names(available_signals),
        "required_signal_names": _signal_names(signals["required"]),
        "recommended_signal_names": _signal_names(signals["recommended"]),
        "optional_signal_names": _signal_names(signals["optional"]),
        "missing_required_signal_names": _signal_names(coverage["missing_required"]),
        "missing_recommended_signal_names": _signal_names(coverage["missing_recommended"]),
        "coverage_score": coverage["score"],
        "route_key": route_key,
        "recommended_pack": recommended_pack,
        "price_guide": PACK_PRICE_GUIDE.get(recommended_pack, "quote required"),
        "customer_cta": cta,
        "next_step": next_step,
        "blockers": blockers,
        "trust_reasons": trust_reasons,
        "hardware_profile_preview": hardware_preview,
        "readiness_preview": readiness_preview,
        "safe_boundary": SAFE_BOUNDARY,
        "customer_notes": customer_notes,
    }
    snapshot["decision"] = "READY FOR CUSTOMER NEXT STEP" if not blockers else "CONDITIONAL NEXT STEP - LIMIT CLAIMS"
    snapshot["snapshot_hash"] = _sha(snapshot)
    return snapshot


def build_tables(snapshot: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    route_rows = [
        {"Item": "Data situation", "Value": snapshot.get("data_situation")},
        {"Item": "Use-case", "Value": snapshot.get("use_case")},
        {"Item": "Coverage score", "Value": f"{snapshot.get('coverage_score')}%"},
        {"Item": "Recommended pack", "Value": snapshot.get("recommended_pack")},
        {"Item": "Price guide", "Value": snapshot.get("price_guide")},
        {"Item": "Next step", "Value": snapshot.get("next_step")},
        {"Item": "CTA", "Value": snapshot.get("customer_cta")},
    ]
    signal_rows = []
    for group, values in [
        ("Available", snapshot.get("available_signal_names", [])),
        ("Required", snapshot.get("required_signal_names", [])),
        ("Recommended", snapshot.get("recommended_signal_names", [])),
        ("Missing required", snapshot.get("missing_required_signal_names", [])),
        ("Missing recommended", snapshot.get("missing_recommended_signal_names", [])),
    ]:
        signal_rows.append({"Signal group": group, "Signals": ", ".join(values) if values else "None"})

    risk_rows = [{"Type": "Blocker / limitation", "Text": item} for item in snapshot.get("blockers", [])]
    if not risk_rows:
        risk_rows.append({"Type": "Status", "Text": "No hard blocker for this customer next step."})
    risk_rows += [{"Type": "Trust rule", "Text": item} for item in snapshot.get("trust_reasons", [])]

    return {
        "route": pd.DataFrame(route_rows),
        "signals": pd.DataFrame(signal_rows),
        "risks": pd.DataFrame(risk_rows),
    }


def create_bundle(snapshot: Dict[str, Any]) -> bytes:
    tables = build_tables(snapshot)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(snapshot, indent=2, ensure_ascii=False, default=str))
        zf.writestr("route_summary.csv", tables["route"].to_csv(index=False))
        zf.writestr("signal_matrix.csv", tables["signals"].to_csv(index=False))
        zf.writestr("risk_and_trust_notes.csv", tables["risks"].to_csv(index=False))
        zf.writestr(
            "CUSTOMER_SAFE_SUMMARY.md",
            f"# EdgeTwin customer route\n\n"
            f"Company: {snapshot.get('company')}\n\n"
            f"Decision: {snapshot.get('decision')}\n\n"
            f"Recommended pack: {snapshot.get('recommended_pack')}\n\n"
            f"Next step: {snapshot.get('next_step')}\n\n"
            f"Boundary: {snapshot.get('safe_boundary')}\n",
        )
    return buffer.getvalue()


def render_streamlit_tab(st) -> Tuple[Dict[str, Any] | None, bytes | None]:
    st.header("Ultimate Customer Flow")
    st.caption("One clean self-selling path: data situation → use-case → sensor/profile route → evidence pack → quote/payment request.")

    with st.form("ultimate_customer_flow_form"):
        c1, c2 = st.columns(2)
        with c1:
            company = st.text_input("Customer / company", "Demo Industrial Customer", key="ultimate_customer_company")
            data_situation_key = st.selectbox("Customer data situation", list(DATA_SITUATIONS.keys()), format_func=lambda k: DATA_SITUATIONS[k], index=1, key="ultimate_data_situation")
            use_case_key = st.selectbox("Use-case", list(USE_CASES.keys()), format_func=lambda k: USE_CASES[k], index=0, key="ultimate_use_case")
            customer_notes = st.text_area("Customer notes", "We want to know whether our data/hardware is ready for a controlled AI or predictive-maintenance pilot.", height=100, key="ultimate_notes")
        with c2:
            has_real_data = st.checkbox("Customer has some real data", value=True, key="ultimate_has_real")
            enough_labels = st.checkbox("Customer has useful maintenance/failure labels", value=False, key="ultimate_labels")
            customer_ready_to_collect = st.checkbox("Customer is ready to collect missing data", value=True, key="ultimate_collect")
            wants_firmware_starter = st.checkbox("Offer hardware/firmware starter when needed", value=True, key="ultimate_fw")
            wants_demo_room = st.checkbox("Show demo evidence room", value=True, key="ultimate_demo")

        st.markdown("**Available signals / sources**")
        available_signals = []
        cols = st.columns(2)
        for idx, (key, label) in enumerate(SIGNALS.items()):
            default = key in {"vibration_accel", "audio_i2s"} if use_case_key in {"bearing_wear", "machine_anomaly"} else False
            with cols[idx % 2]:
                if st.checkbox(label, value=default, key=f"ultimate_signal_{key}"):
                    available_signals.append(key)

        submitted = st.form_submit_button("Build ultimate customer route", use_container_width=True)

    if not submitted:
        st.info("Use this page as the main customer route: it prevents confusion and recommends the safest pack, hardware and evidence path.")
        return None, None

    snapshot = build_ultimate_customer_flow_snapshot(
        company=company,
        data_situation_key=data_situation_key,
        use_case_key=use_case_key,
        available_signals=available_signals,
        has_real_data=has_real_data,
        enough_labels=enough_labels,
        customer_ready_to_collect=customer_ready_to_collect,
        wants_firmware_starter=wants_firmware_starter,
        wants_demo_room=wants_demo_room,
        customer_notes=customer_notes,
    )
    bundle = create_bundle(snapshot)
    tables = build_tables(snapshot)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Coverage", f"{snapshot.get('coverage_score')}%")
    m2.metric("Route", snapshot.get("route_key"))
    m3.metric("Pack", snapshot.get("recommended_pack"))
    m4.metric("Decision", snapshot.get("decision"))

    if snapshot.get("blockers"):
        st.warning(snapshot.get("decision"))
    else:
        st.success(snapshot.get("decision"))

    tabs = st.tabs(["Route", "Signals", "Risks", "Hardware preview", "Download"])
    with tabs[0]:
        st.dataframe(tables["route"], use_container_width=True)
        st.info(snapshot.get("safe_boundary"))
    with tabs[1]:
        st.dataframe(tables["signals"], use_container_width=True)
    with tabs[2]:
        st.dataframe(tables["risks"], use_container_width=True)
    with tabs[3]:
        st.json(snapshot.get("hardware_profile_preview", {}))
    with tabs[4]:
        st.download_button("Download ultimate customer route bundle", bundle, file_name="edgetwin_ultimate_customer_route.zip", mime="application/zip", use_container_width=True)

    return snapshot, bundle
