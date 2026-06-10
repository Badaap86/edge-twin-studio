"""EdgeTwin Hardware Profile Matrix + Data Situation Selector.

Purpose:
- Help a customer choose the smallest reliable sensor/hardware route per use-case.
- Connect data situation -> sensor profile -> firmware starter -> evidence pack route.
- Keep hardware/code guidance bounded: pilot/data-collection templates only, not certified production firmware.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import textwrap
import zipfile
from typing import Any, Dict, List

import pandas as pd

MODULE = "Hardware Profile Matrix"

SAFE_HARDWARE_BOUNDARY = (
    "Hardware profiles and firmware starters are for controlled pilot data collection and readiness/evidence assessment. "
    "They are not production-certified devices, safety systems, security certifications, legal/compliance approvals, "
    "or guarantees of predictive-maintenance performance. Customer-specific validation requires real field data."
)

DATA_SITUATIONS = {
    "good_real_data": {
        "label": "We already have usable real data",
        "route": "Run Real-Data Evidence Pack first. New hardware may not be needed immediately.",
        "pack": "Real-Data Evidence Pack",
        "cta": "Upload sample/export for Data Quality Gate.",
    },
    "limited_real_data": {
        "label": "We have limited real data",
        "route": "Use real data as evidence basis and add Synthetic Evidence Expansion for safe scenario coverage.",
        "pack": "Real Data + Synthetic Evidence Expansion",
        "cta": "Use current data, then collect missing fields with a starter profile if needed.",
    },
    "no_real_data": {
        "label": "We do not have real data yet",
        "route": "Start with Dataset Starter Pack, hardware profile and firmware starter before predictive claims.",
        "pack": "Dataset Starter / Hardware Profile / Field Data Kit",
        "cta": "Confirm hardware profile and collect baseline data first.",
    },
    "hardware_only": {
        "label": "We have hardware, but no clean dataset",
        "route": "Confirm hardware/pinout, generate pilot firmware/config and log EdgeTwin-compatible data.",
        "pack": "Hardware Profile + Firmware Starter Pack",
        "cta": "Confirm board/sensors and generate starter bundle.",
    },
}

SENSOR_CATALOG = {
    "vibration_accel": "Vibration / accelerometer",
    "audio_i2s": "Audio / I2S microphone",
    "temperature": "Temperature",
    "rpm_load": "RPM / load context",
    "maintenance_labels": "Maintenance/failure labels",
    "radar": "Radar / presence/distance",
    "bme688_gas": "BME688 gas/environment",
    "pmu_power": "PMU / power health / dying-gasp",
    "gps_asset": "GPS / asset/location context",
    "lora_gateway": "LoRa/gateway connectivity",
    "nine_axis_orientation": "9-axis orientation/tamper context",
    "existing_export": "Existing PLC/SCADA/historian/CMMS export",
}

USE_CASE_MATRIX = {
    "bearing_wear": {
        "label": "Bearing wear / lager-slijtage",
        "plain_customer_problem": "We want to know if bearing wear can be detected before starting an expensive AI pilot.",
        "required": ["vibration_accel", "maintenance_labels"],
        "recommended": ["audio_i2s", "temperature", "rpm_load"],
        "optional": ["nine_axis_orientation", "lora_gateway"],
        "not_core": ["radar", "bme688_gas", "gps_asset"],
        "minimal_profile": "Bearing Wear Basic",
        "strong_profile": "Bearing Wear Pro",
        "firmware_profile_key": "bearing_wear_basic",
        "starter_hardware": "RAK3312/ESP32-S3 + vibration/accelerometer + ICS43434 audio; add temperature/RPM/load for stronger evidence.",
        "safe_claim": "Can prepare bearing-wear pilot readiness and data requirements; cannot prove customer-specific failure prediction without sufficient real machine data.",
    },
    "machine_anomaly": {
        "label": "General machine anomaly / motor-pump-fan",
        "plain_customer_problem": "We want to learn normal behaviour and flag unusual patterns for pilot evaluation.",
        "required": ["vibration_accel", "audio_i2s"],
        "recommended": ["temperature", "rpm_load", "maintenance_labels"],
        "optional": ["pmu_power", "lora_gateway", "nine_axis_orientation"],
        "not_core": ["radar", "bme688_gas"],
        "minimal_profile": "Machine Baseline Basic",
        "strong_profile": "Machine Baseline Pro",
        "firmware_profile_key": "machine_baseline_basic",
        "starter_hardware": "ESP32-S3/RAK3312 + vibration + I2S audio + CSV/LoRa logging; add temperature/load context where possible.",
        "safe_claim": "Can learn and document baseline/anomaly evidence; root cause needs labels and customer context.",
    },
    "facade_security": {
        "label": "Facade / site security sentinel",
        "plain_customer_problem": "We want to detect tampering, loitering, impact or suspicious activity near an asset/building.",
        "required": ["radar", "audio_i2s", "vibration_accel", "pmu_power"],
        "recommended": ["bme688_gas", "gps_asset", "lora_gateway"],
        "optional": ["nine_axis_orientation", "temperature"],
        "not_core": ["rpm_load", "maintenance_labels"],
        "minimal_profile": "Security Sentinel Basic",
        "strong_profile": "OMEGA-X Full Sentinel",
        "firmware_profile_key": "omega_x_sentinel",
        "starter_hardware": "OMEGA-X/RAK3312 style node with radar, audio, vibration, PMU, LoRa and optional BME688/GPS.",
        "safe_claim": "Can provide security decision-support and evidence logging; not a certified security/safety guarantee.",
    },
    "fire_arson_risk": {
        "label": "Fire / arson-risk context",
        "plain_customer_problem": "We want early environmental/fire-risk signals around an asset or facade.",
        "required": ["bme688_gas", "temperature", "audio_i2s"],
        "recommended": ["pmu_power", "lora_gateway"],
        "optional": ["radar", "gps_asset", "vibration_accel"],
        "not_core": ["rpm_load", "maintenance_labels"],
        "minimal_profile": "Environment/Fire Risk Basic",
        "strong_profile": "OMEGA-X Environment Sentinel",
        "firmware_profile_key": "environment_fire_basic",
        "starter_hardware": "ESP32-S3/RAK3312 + BME688 + audio + stable power + LoRa/gateway logging.",
        "safe_claim": "Can document environmental anomaly evidence; not fire certification or safety-system approval.",
    },
    "existing_industrial_export": {
        "label": "Existing industrial data export",
        "plain_customer_problem": "We already have PLC/SCADA/historian/CMMS exports and want to know if they are usable.",
        "required": ["existing_export", "maintenance_labels", "rpm_load"],
        "recommended": ["vibration_accel", "temperature"],
        "optional": ["audio_i2s", "lora_gateway"],
        "not_core": ["radar", "bme688_gas", "gps_asset"],
        "minimal_profile": "No New Hardware / Import Schema",
        "strong_profile": "Real-Data Evidence + Optional Field Kit",
        "firmware_profile_key": "no_firmware_needed",
        "starter_hardware": "No new hardware needed first. Map existing exports to EdgeTwin schema; add field kit only if key signals are missing.",
        "safe_claim": "Can evaluate data readiness and gaps; production claims depend on quality, labels and operating context.",
    },
}

FIRMWARE_STARTER_MAP = {
    "bearing_wear_basic": {
        "bundle_name": "Bearing Wear Basic Firmware Starter",
        "signals": ["vibration_accel", "audio_i2s"],
        "mode": "evidence_logging",
        "outputs": ["CSV feature log", "baseline summary", "EdgeTwin import schema"],
    },
    "machine_baseline_basic": {
        "bundle_name": "Machine Baseline Firmware Starter",
        "signals": ["vibration_accel", "audio_i2s", "temperature"],
        "mode": "baseline_learning",
        "outputs": ["CSV feature log", "normal/anomaly flags", "EdgeTwin import schema"],
    },
    "omega_x_sentinel": {
        "bundle_name": "OMEGA-X Sentinel Firmware Starter",
        "signals": ["radar", "audio_i2s", "vibration_accel", "bme688_gas", "pmu_power", "gps_asset", "lora_gateway"],
        "mode": "sentinel_and_evidence_logging",
        "outputs": ["LoRa alarm summary", "CSV evidence log", "baseline profile"],
    },
    "environment_fire_basic": {
        "bundle_name": "Environment/Fire Risk Firmware Starter",
        "signals": ["bme688_gas", "temperature", "audio_i2s", "pmu_power"],
        "mode": "environment_evidence_logging",
        "outputs": ["CSV environment log", "gas/temp trend summary", "EdgeTwin import schema"],
    },
    "no_firmware_needed": {
        "bundle_name": "Existing Export Import Schema",
        "signals": ["existing_export"],
        "mode": "import_only",
        "outputs": ["CSV/Excel schema", "column mapping checklist", "Data Quality Gate input"],
    },
}


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def _names(keys: List[str]) -> List[str]:
    return [SENSOR_CATALOG.get(k, k) for k in keys]


def _coverage_score(row: Dict[str, Any], available: List[str]) -> int:
    available_set = set(available)
    required = set(row.get("required", []))
    recommended = set(row.get("recommended", []))
    optional = set(row.get("optional", []))
    score = 0
    if required:
        score += int(55 * len(required & available_set) / len(required))
    if recommended:
        score += int(35 * len(recommended & available_set) / len(recommended))
    if optional:
        score += int(10 * len(optional & available_set) / len(optional))
    return min(100, score)


def build_sensor_matrix_table() -> pd.DataFrame:
    rows = []
    for key, item in USE_CASE_MATRIX.items():
        rows.append({
            "Use-case": item["label"],
            "Minimal profile": item["minimal_profile"],
            "Strong profile": item["strong_profile"],
            "Required": ", ".join(_names(item["required"])),
            "Recommended": ", ".join(_names(item["recommended"])),
            "Optional/context": ", ".join(_names(item["optional"])),
            "Not core": ", ".join(_names(item["not_core"])),
        })
    return pd.DataFrame(rows)


def build_hardware_profile_snapshot(
    company: str = "Customer",
    data_situation_key: str = "limited_real_data",
    use_case_key: str = "bearing_wear",
    available_signals: List[str] | None = None,
    customer_confirms_hardware: bool = False,
    wants_firmware_starter: bool = True,
    notes: str = "",
) -> Dict[str, Any]:
    available_signals = list(available_signals or [])
    situation = DATA_SITUATIONS.get(data_situation_key, DATA_SITUATIONS["limited_real_data"])
    use_case = USE_CASE_MATRIX.get(use_case_key, USE_CASE_MATRIX["bearing_wear"])

    coverage = _coverage_score(use_case, available_signals)
    required_missing = [s for s in use_case["required"] if s not in available_signals]
    recommended_missing = [s for s in use_case["recommended"] if s not in available_signals]
    firmware_key = use_case["firmware_profile_key"]
    firmware = FIRMWARE_STARTER_MAP.get(firmware_key, FIRMWARE_STARTER_MAP["bearing_wear_basic"])

    if data_situation_key == "good_real_data" and not required_missing:
        decision = "Use real-data evidence first"
    elif data_situation_key in {"no_real_data", "hardware_only"} or required_missing:
        decision = "Start with hardware/profile data collection"
    else:
        decision = "Use limited real data + collect missing context"

    starter_status = "not_needed_now"
    if firmware_key == "no_firmware_needed":
        starter_status = "import_schema_only"
    elif wants_firmware_starter and customer_confirms_hardware:
        starter_status = "ready_to_generate_starter"
    elif wants_firmware_starter:
        starter_status = "hardware_confirmation_needed"

    customer_summary = (
        f"{company}: for {use_case['label']}, EdgeTwin recommends {situation['pack']}. "
        f"Current signal coverage is {coverage}%. Next step: {decision}."
    )

    next_steps: List[str] = []
    if required_missing:
        next_steps.append("Add or confirm required signals: " + ", ".join(_names(required_missing)) + ".")
    if recommended_missing:
        next_steps.append("Recommended for stronger evidence: " + ", ".join(_names(recommended_missing)) + ".")
    if starter_status == "ready_to_generate_starter":
        next_steps.append("Generate the firmware starter/config and collect controlled baseline data.")
    elif starter_status == "hardware_confirmation_needed":
        next_steps.append("Confirm exact board, sensor modules and pinout before generating firmware.")
    elif starter_status == "import_schema_only":
        next_steps.append("Map existing data export to the EdgeTwin import schema.")
    next_steps.append("Run Data Quality Gate before any customer-specific prediction or production claim.")

    snapshot = {
        "module": MODULE,
        "created_at": _now(),
        "company": company,
        "data_situation_key": data_situation_key,
        "data_situation": situation,
        "use_case_key": use_case_key,
        "use_case": use_case,
        "available_signals": available_signals,
        "available_signal_names": _names(available_signals),
        "coverage_score": coverage,
        "required_missing": required_missing,
        "required_missing_names": _names(required_missing),
        "recommended_missing": recommended_missing,
        "recommended_missing_names": _names(recommended_missing),
        "decision": decision,
        "starter_status": starter_status,
        "firmware_starter": firmware,
        "customer_confirms_hardware": bool(customer_confirms_hardware),
        "wants_firmware_starter": bool(wants_firmware_starter),
        "safe_hardware_boundary": SAFE_HARDWARE_BOUNDARY,
        "customer_summary": customer_summary,
        "next_steps": next_steps,
        "notes": notes,
    }
    snapshot["snapshot_hash"] = _sha(snapshot)
    return snapshot


def build_recommended_route_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    use_case = snapshot.get("use_case", {})
    rows = [
        {"Category": "Data situation", "Recommendation": snapshot.get("data_situation", {}).get("label", "")},
        {"Category": "Recommended pack", "Recommendation": snapshot.get("data_situation", {}).get("pack", "")},
        {"Category": "Minimal hardware profile", "Recommendation": use_case.get("minimal_profile", "")},
        {"Category": "Strong hardware profile", "Recommendation": use_case.get("strong_profile", "")},
        {"Category": "Starter hardware", "Recommendation": use_case.get("starter_hardware", "")},
        {"Category": "Firmware starter", "Recommendation": snapshot.get("firmware_starter", {}).get("bundle_name", "")},
        {"Category": "Safe boundary", "Recommendation": use_case.get("safe_claim", "")},
    ]
    return pd.DataFrame(rows)


def create_hardware_profile_bundle(snapshot: Dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("hardware_profile_snapshot.json", json.dumps(snapshot, indent=2, ensure_ascii=False, default=str))
        z.writestr("hardware_profile_matrix.csv", build_sensor_matrix_table().to_csv(index=False))
        z.writestr("recommended_route.csv", build_recommended_route_table(snapshot).to_csv(index=False))
        z.writestr("safe_boundary.txt", SAFE_HARDWARE_BOUNDARY + "\n")
        z.writestr("customer_summary.md", build_customer_summary_markdown(snapshot))
        z.writestr("firmware_next_step.md", build_firmware_next_step_markdown(snapshot))
    return buffer.getvalue()


def build_customer_summary_markdown(snapshot: Dict[str, Any]) -> str:
    lines = [
        "# EdgeTwin Hardware Profile Recommendation",
        "",
        snapshot.get("customer_summary", ""),
        "",
        "## What this means",
        f"- Data route: **{snapshot.get('data_situation', {}).get('pack', '')}**",
        f"- Use-case: **{snapshot.get('use_case', {}).get('label', '')}**",
        f"- Signal coverage: **{snapshot.get('coverage_score', 0)}%**",
        f"- Decision: **{snapshot.get('decision', '')}**",
        "",
        "## Missing required signals",
    ]
    missing = snapshot.get("required_missing_names", [])
    lines.extend([f"- {item}" for item in missing] or ["- No required signal gap recorded."])
    lines += ["", "## Next steps"]
    lines.extend([f"- {item}" for item in snapshot.get("next_steps", [])])
    lines += ["", "## Safe boundary", SAFE_HARDWARE_BOUNDARY]
    return "\n".join(lines) + "\n"


def build_firmware_next_step_markdown(snapshot: Dict[str, Any]) -> str:
    firmware = snapshot.get("firmware_starter", {})
    starter_status = snapshot.get("starter_status", "")
    text = f"""
    # Firmware Starter Next Step

    Recommended starter: **{firmware.get('bundle_name', 'Unknown')}**  
    Mode: **{firmware.get('mode', 'Unknown')}**  
    Status: **{starter_status}**

    ## Signals covered
    {chr(10).join('- ' + SENSOR_CATALOG.get(s, s) for s in firmware.get('signals', []))}

    ## Outputs
    {chr(10).join('- ' + str(s) for s in firmware.get('outputs', []))}

    ## Important
    This starter is for controlled pilot data collection only. It is not production-certified firmware and does not create a production accuracy, legal, safety or compliance guarantee.
    """
    return textwrap.dedent(text).strip() + "\n"


def render_streamlit_tab(st) -> tuple[Dict[str, Any] | None, bytes | None]:
    st.header("Hardware Profile Matrix")
    st.write(
        "Choose the customer data situation and use-case. EdgeTwin recommends the smallest reliable sensor profile, "
        "what is missing, and whether a firmware starter is appropriate."
    )

    c1, c2 = st.columns([1.05, 1])
    with c1:
        company = st.text_input("Customer / project name", value="Founding customer", key="v134_company")
        data_situation_key = st.selectbox(
            "Customer data situation",
            list(DATA_SITUATIONS.keys()),
            format_func=lambda k: DATA_SITUATIONS[k]["label"],
            index=1,
            key="v134_data_situation",
        )
        use_case_key = st.selectbox(
            "Use-case",
            list(USE_CASE_MATRIX.keys()),
            format_func=lambda k: USE_CASE_MATRIX[k]["label"],
            index=0,
            key="v134_use_case",
        )
    with c2:
        available_signals = st.multiselect(
            "Signals/hardware/data already available",
            options=list(SENSOR_CATALOG.keys()),
            default=["vibration_accel", "audio_i2s"],
            format_func=lambda k: SENSOR_CATALOG.get(k, k),
            key="v134_available_signals",
        )
        customer_confirms_hardware = st.checkbox(
            "Customer confirms this hardware/profile will be used",
            value=False,
            key="v134_confirm_hardware",
        )
        wants_firmware_starter = st.checkbox(
            "Prepare firmware starter route if appropriate",
            value=True,
            key="v134_wants_firmware",
        )

    notes = st.text_area(
        "Notes",
        value="Use the smallest reliable sensor profile. Keep synthetic data as scenario expansion only.",
        height=80,
        key="v134_notes",
    )

    snapshot = build_hardware_profile_snapshot(
        company=company,
        data_situation_key=data_situation_key,
        use_case_key=use_case_key,
        available_signals=available_signals,
        customer_confirms_hardware=customer_confirms_hardware,
        wants_firmware_starter=wants_firmware_starter,
        notes=notes,
    )
    bundle = create_hardware_profile_bundle(snapshot)

    m1, m2, m3 = st.columns(3)
    m1.metric("Signal coverage", f"{snapshot['coverage_score']}%")
    m2.metric("Recommended pack", snapshot["data_situation"]["pack"])
    m3.metric("Starter status", snapshot["starter_status"].replace("_", " "))

    st.success(snapshot["customer_summary"])
    st.subheader("Recommended route")
    st.dataframe(build_recommended_route_table(snapshot), use_container_width=True)

    st.subheader("Hardware profile matrix")
    st.dataframe(build_sensor_matrix_table(), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Missing required signals")
        for item in snapshot.get("required_missing_names", []) or ["No required signal gap recorded."]:
            st.write(f"- {item}")
        st.markdown("### Recommended extras")
        for item in snapshot.get("recommended_missing_names", []) or ["No recommended gap recorded."]:
            st.write(f"- {item}")
    with col_b:
        st.markdown("### Next steps")
        for item in snapshot.get("next_steps", []):
            st.write(f"- {item}")
        st.markdown("### Safe claim boundary")
        st.info(snapshot.get("use_case", {}).get("safe_claim", SAFE_HARDWARE_BOUNDARY))

    st.download_button(
        "Download hardware profile recommendation",
        data=bundle,
        file_name="edgetwin_hardware_profile_recommendation.zip",
        mime="application/zip",
        use_container_width=True,
        key="v134_download_bundle",
    )
    st.caption(SAFE_HARDWARE_BOUNDARY)
    return snapshot, bundle
