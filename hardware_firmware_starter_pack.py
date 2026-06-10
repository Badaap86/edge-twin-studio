"""EdgeTwin Hardware Profile + Firmware Starter Pack.

Purpose:
- Recommend approved hardware profiles for pilot/data collection.
- Generate a bounded starter firmware/config bundle only after the customer confirms hardware.
- Keep the output safe: data collection and readiness evidence only, not certified production/safety firmware.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import textwrap
import zipfile
from typing import Any, Dict, List, Tuple

import pandas as pd

MODULE = "Hardware Profile + Firmware Starter Pack"

SAFE_FIRMWARE_BOUNDARY = (
    "Firmware starter files are provided for controlled pilot data collection and EdgeTwin readiness/evidence workflows. "
    "They are not production-certified, safety-certified, security-certified or compliance-certified firmware. "
    "Customer-specific validation, wiring verification, environmental testing and responsible human oversight remain required."
)

APPROVED_HARDWARE_PROFILES = {
    "rak3312_omega_x_field_kit": {
        "label": "RAK3312 OMEGA-X Field Data Kit",
        "best_for": "bearing wear, machine anomaly, facade/security, multi-sensor baseline learning",
        "board": "RAK3312 / ESP32-S3 + WisBlock base",
        "sensors": ["I2S microphone", "vibration / 9-axis", "BME688", "radar optional", "GPS optional", "FRAM optional", "LoRa SX1262"],
        "connectivity": ["serial_csv", "lora_summary", "sd_csv_optional", "gateway_mqtt_optional"],
        "default_sample_rate": 16000,
        "default_fft_size": 1024,
        "pin_notes": [
            "Confirm actual RAK base board pinout before flashing.",
            "I2S microphone: SCK/WS/SD must match the physical wiring.",
            "BME688/FRAM/PMU share I2C; verify addresses and pullups.",
            "Radar UART baud/protocol must be verified with raw-frame debug.",
        ],
    },
    "generic_esp32_i2s_vibration": {
        "label": "Generic ESP32 I2S + Vibration Starter",
        "best_for": "low-cost data collection proof-of-concept",
        "board": "ESP32 / ESP32-S3",
        "sensors": ["I2S microphone", "digital/analog vibration", "temperature optional"],
        "connectivity": ["serial_csv", "sd_csv_optional", "wifi_mqtt_optional"],
        "default_sample_rate": 16000,
        "default_fft_size": 1024,
        "pin_notes": [
            "Confirm board pin compatibility and bootstrapping pins.",
            "Avoid pins that affect boot mode.",
            "Use stable power supply before long logging tests.",
        ],
    },
    "csv_existing_system_export": {
        "label": "Existing System Export / No Firmware",
        "best_for": "customers with PLC/SCADA/historian/CMMS exports",
        "board": "No new hardware required",
        "sensors": ["existing sensor export", "maintenance/export files"],
        "connectivity": ["csv_excel_json_export"],
        "default_sample_rate": 0,
        "default_fft_size": 0,
        "pin_notes": ["Use EdgeTwin import schema instead of firmware."],
    },
}

USE_CASE_PROFILES = {
    "bearing_wear": {
        "label": "Bearing wear / rotating equipment",
        "required_signals": ["vibration", "audio optional", "temperature", "rpm/load if available", "maintenance/failure labels"],
        "features": ["rms", "crest_factor", "kurtosis", "spectral_peaks", "energy_trend", "temperature_context"],
        "warning": "Bearing claims require real customer data and operating context; synthetic data is scenario expansion only.",
    },
    "motor_anomaly": {
        "label": "Motor / pump / fan anomaly",
        "required_signals": ["vibration", "audio", "temperature", "current/load if available", "maintenance events"],
        "features": ["rms", "impulse_ratio", "entropy", "spectral_bands", "baseline_drift"],
        "warning": "Anomaly detection may find unusual behaviour, but root cause needs labels/context.",
    },
    "facade_security": {
        "label": "Facade / site security sentinel",
        "required_signals": ["audio", "radar", "vibration", "gas/environment", "power status", "GPS optional"],
        "features": ["impulse_ratio", "traffic_band", "shockwave_band", "crackle_band", "radar_presence", "gas_ratio"],
        "warning": "Security alerts are decision-support signals, not certified safety/security guarantees.",
    },
    "generic_machine_baseline": {
        "label": "Generic machine baseline learning",
        "required_signals": ["at least one sensor stream", "timestamps", "asset ID", "operating context if possible"],
        "features": ["rolling_baseline", "energy", "variance", "drift", "missing_data", "outlier_count"],
        "warning": "Baseline learning describes normal/abnormal patterns; it does not prove failure type without labels.",
    },
}


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def _slug(value: str) -> str:
    keep = []
    for ch in str(value).lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_"}:
            keep.append("_")
    return "".join(keep).strip("_") or "edgetwin_firmware"


def build_hardware_firmware_snapshot(
    company: str = "Customer",
    use_case_key: str = "bearing_wear",
    hardware_profile_key: str = "rak3312_omega_x_field_kit",
    customer_confirmed_hardware: bool = False,
    mode: str = "evidence_logging",
    connectivity: str = "serial_csv",
    sample_rate: int | None = None,
    fft_size: int | None = None,
    notes: str = "",
) -> Dict[str, Any]:
    profile = dict(APPROVED_HARDWARE_PROFILES.get(hardware_profile_key, APPROVED_HARDWARE_PROFILES["rak3312_omega_x_field_kit"]))
    use_case = dict(USE_CASE_PROFILES.get(use_case_key, USE_CASE_PROFILES["bearing_wear"]))
    sample_rate = int(sample_rate if sample_rate is not None else profile.get("default_sample_rate", 16000))
    fft_size = int(fft_size if fft_size is not None else profile.get("default_fft_size", 1024))

    status = "ready_to_generate" if customer_confirmed_hardware and hardware_profile_key != "csv_existing_system_export" else "confirmation_needed"
    if hardware_profile_key == "csv_existing_system_export":
        status = "no_firmware_needed"

    deliverables = [
        "hardware profile summary",
        "wiring/checklist notes",
        "EdgeTwin CSV schema",
        "safe-use notice",
    ]
    if status == "ready_to_generate":
        deliverables.extend(["config.h", "Arduino starter firmware", "firmware README"])
    elif status == "confirmation_needed":
        deliverables.append("draft config only after customer confirms hardware")
    else:
        deliverables.append("export/import schema instead of firmware")

    next_steps = {
        "ready_to_generate": [
            "Verify wiring and power before flashing.",
            "Flash starter firmware in a controlled test environment.",
            "Run 48-hour bench logging, then field baseline logging.",
            "Import CSV/log output into EdgeTwin Data Quality Gate.",
        ],
        "confirmation_needed": [
            "Customer confirms exact board, sensor modules and pinout.",
            "EdgeTwin locks the hardware profile and generates the starter bundle.",
            "No production/safety claim is made from template firmware alone.",
        ],
        "no_firmware_needed": [
            "Export CSV/Excel/JSON from existing systems.",
            "Map columns to EdgeTwin import schema.",
            "Run Data Quality Gate and evidence review.",
        ],
    }[status]

    snapshot = {
        "module": MODULE,
        "created_at": _now(),
        "company": company,
        "use_case_key": use_case_key,
        "use_case": use_case,
        "hardware_profile_key": hardware_profile_key,
        "hardware_profile": profile,
        "customer_confirmed_hardware": bool(customer_confirmed_hardware),
        "status": status,
        "mode": mode,
        "connectivity": connectivity,
        "sample_rate": sample_rate,
        "fft_size": fft_size,
        "deliverables": deliverables,
        "next_steps": next_steps,
        "safe_firmware_boundary": SAFE_FIRMWARE_BOUNDARY,
        "notes": notes,
    }
    snapshot["snapshot_hash"] = _sha(snapshot)
    return snapshot


def build_profile_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Profile": v["label"],
            "Best for": v["best_for"],
            "Board": v["board"],
            "Sensors": ", ".join(v["sensors"]),
        }
        for v in APPROVED_HARDWARE_PROFILES.values()
    ])


def build_csv_schema() -> str:
    rows = [
        ["timestamp_ms", "integer", "Required", "Device timestamp in milliseconds"],
        ["node_id", "string/integer", "Required", "Unique node/device ID"],
        ["asset_id", "string", "Recommended", "Machine/asset identifier"],
        ["use_case", "string", "Required", "Use-case label"],
        ["mode", "string", "Required", "evidence_logging / sentinel / calibration"],
        ["raw_energy", "float", "Recommended", "Audio/vibration energy proxy"],
        ["rms", "float", "Recommended", "Vibration/audio RMS"],
        ["crest_factor", "float", "Recommended", "Peak-to-RMS ratio"],
        ["kurtosis", "float", "Optional", "Impulsiveness indicator"],
        ["entropy", "float", "Optional", "Spectral entropy"],
        ["dominant_freq_hz", "float", "Optional", "Dominant frequency"],
        ["temperature_c", "float", "Optional", "Temperature context"],
        ["rpm", "float", "Optional", "Speed context if available"],
        ["load_pct", "float", "Optional", "Load context if available"],
        ["event_label", "string", "Optional", "normal / suspected_anomaly / maintenance / failure"],
        ["confidence", "float", "Optional", "Internal confidence, not production proof"],
    ]
    return "column,type,importance,notes\n" + "\n".join([",".join(map(str, row)) for row in rows]) + "\n"


def build_config_h(snapshot: Dict[str, Any]) -> str:
    profile_key = snapshot.get("hardware_profile_key", "rak3312_omega_x_field_kit")
    use_case_key = snapshot.get("use_case_key", "bearing_wear")
    sample_rate = int(snapshot.get("sample_rate") or 16000)
    fft_size = int(snapshot.get("fft_size") or 1024)
    mode = snapshot.get("mode", "evidence_logging")
    connectivity = snapshot.get("connectivity", "serial_csv")
    return textwrap.dedent(f"""
    #pragma once

    // EdgeTwin Firmware Starter Config
    // Generated for controlled pilot data collection only.
    // Not production/safety/security/compliance certified firmware.

    #define EDGETWIN_PROFILE "{profile_key}"
    #define EDGETWIN_USE_CASE "{use_case_key}"
    #define EDGETWIN_MODE "{mode}"
    #define EDGETWIN_CONNECTIVITY "{connectivity}"

    #define NODE_ID 101
    #define SAMPLE_RATE {sample_rate}
    #define FFT_SIZE {fft_size}

    // Default RAK3312 / OMEGA-X style pins. Verify before flashing.
    #define I2S_SCK 5
    #define I2S_WS 6
    #define I2S_SD 7
    #define VIB_PIN 17
    #define RADAR_RX 15
    #define RADAR_TX 16
    #define GPS_RX 34
    #define GPS_TX 33
    #define FRAM_I2C_ADDR 0x50

    // Logging interval for evidence mode.
    #define EVIDENCE_LOG_INTERVAL_MS 1000

    // Safety boundary: this firmware logs data for EdgeTwin readiness/evidence.
    // It must not be used as a certified production control or safety system.
    """).strip() + "\n"


def build_starter_ino(snapshot: Dict[str, Any]) -> str:
    use_case = snapshot.get("use_case", {})
    feature_comment = ", ".join(use_case.get("features", []))
    return textwrap.dedent(f"""
    /***********************************************************
     * EdgeTwin Firmware Starter
     * Purpose: controlled pilot data collection for EdgeTwin evidence packs.
     * Boundary: NOT production/safety/security/compliance certified firmware.
     * Features to collect for this use-case: {feature_comment}
     ***********************************************************/

    #include <Arduino.h>
    #include "config.h"

    struct EvidenceRow {{
      uint32_t timestampMs;
      float rawEnergy;
      float rms;
      float crestFactor;
      float entropy;
      float dominantFreqHz;
      float temperatureC;
      int vibrationState;
      const char* eventLabel;
    }};

    static uint32_t lastLogMs = 0;
    static float rollingBaseline = 0.0f;
    static uint32_t baselineSamples = 0;

    float readRawEnergyPlaceholder() {{
      // Replace with real I2S/audio/vibration feature extraction.
      // Keep this as a placeholder until hardware wiring is verified.
      int v = analogRead(A0);
      return (float)v;
    }}

    float safeRatio(float a, float b) {{
      if (fabsf(b) < 0.0001f) return 0.0f;
      return a / b;
    }}

    EvidenceRow collectEvidenceRow() {{
      EvidenceRow row;
      row.timestampMs = millis();
      row.rawEnergy = readRawEnergyPlaceholder();
      row.rms = row.rawEnergy;              // Replace with real RMS after sensor verification.
      row.crestFactor = safeRatio(row.rawEnergy, row.rms + 1.0f);
      row.entropy = 0.0f;                   // Replace with FFT entropy when enabled.
      row.dominantFreqHz = 0.0f;            // Replace with FFT peak frequency when enabled.
      row.temperatureC = NAN;               // Fill from BME688/temperature sensor if available.
      row.vibrationState = digitalRead(VIB_PIN);

      if (baselineSamples < 3600) {{
        rollingBaseline = rollingBaseline * 0.99f + row.rawEnergy * 0.01f;
        baselineSamples++;
        row.eventLabel = "baseline_learning";
      }} else if (row.rawEnergy > rollingBaseline * 2.5f) {{
        row.eventLabel = "suspected_anomaly";
      }} else {{
        row.eventLabel = "normal";
      }}
      return row;
    }}

    void printCsvHeader() {{
      Serial.println("timestamp_ms,node_id,asset_id,use_case,mode,raw_energy,rms,crest_factor,entropy,dominant_freq_hz,temperature_c,vibration_state,event_label,confidence");
    }}

    void printEvidenceCsv(const EvidenceRow& row) {{
      float confidence = (strcmp(row.eventLabel, "suspected_anomaly") == 0) ? 60.0f : 30.0f;
      Serial.printf("%lu,%d,%s,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%d,%s,%.1f\\n",
        (unsigned long)row.timestampMs,
        NODE_ID,
        "asset_001",
        EDGETWIN_USE_CASE,
        EDGETWIN_MODE,
        row.rawEnergy,
        row.rms,
        row.crestFactor,
        row.entropy,
        row.dominantFreqHz,
        row.temperatureC,
        row.vibrationState,
        row.eventLabel,
        confidence
      );
    }}

    void setup() {{
      Serial.begin(115200);
      delay(500);
      pinMode(VIB_PIN, INPUT_PULLDOWN);
      printCsvHeader();
      Serial.println("# EdgeTwin firmware starter active - pilot data collection only");
      Serial.println("# Verify wiring, power, sensor readings and logs before field use");
    }}

    void loop() {{
      uint32_t now = millis();
      if (now - lastLogMs >= EVIDENCE_LOG_INTERVAL_MS) {{
        lastLogMs = now;
        EvidenceRow row = collectEvidenceRow();
        printEvidenceCsv(row);
      }}
    }}
    """).strip() + "\n"


def build_wiring_checklist(snapshot: Dict[str, Any]) -> str:
    profile = snapshot.get("hardware_profile", {})
    lines = [
        "# Wiring / Hardware Confirmation Checklist",
        "",
        f"Hardware profile: {profile.get('label')}",
        f"Board: {profile.get('board')}",
        "",
        "## Confirm before flashing",
        "- Exact board model and revision confirmed.",
        "- Power supply is stable for long logging sessions.",
        "- I2C addresses verified where applicable.",
        "- UART/radar baud and raw protocol verified where applicable.",
        "- Sensor orientation/mounting documented.",
        "- Asset ID and location are known.",
        "- Customer understands this is pilot/data-collection firmware only.",
        "",
        "## Profile notes",
    ]
    lines.extend([f"- {x}" for x in profile.get("pin_notes", [])])
    return "\n".join(lines) + "\n"


def create_firmware_starter_bundle(snapshot: Dict[str, Any]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("hardware_firmware_snapshot.json", json.dumps(snapshot, indent=2, ensure_ascii=False))
        zf.writestr("SAFE_USE_NOTICE.txt", SAFE_FIRMWARE_BOUNDARY)
        zf.writestr("edgetwin_csv_schema.csv", build_csv_schema())
        zf.writestr("WIRING_CHECKLIST.md", build_wiring_checklist(snapshot))
        zf.writestr("README_FIRMWARE_STARTER.md", build_firmware_readme(snapshot))
        if snapshot.get("status") == "ready_to_generate":
            folder = _slug(snapshot.get("company", "customer")) + "_edgetwin_firmware_starter"
            zf.writestr(f"{folder}/config.h", build_config_h(snapshot))
            zf.writestr(f"{folder}/EdgeTwin_Firmware_Starter.ino", build_starter_ino(snapshot))
    return mem.getvalue()


def build_firmware_readme(snapshot: Dict[str, Any]) -> str:
    use_case = snapshot.get("use_case", {})
    profile = snapshot.get("hardware_profile", {})
    lines = [
        "# EdgeTwin Hardware Profile + Firmware Starter Pack",
        "",
        f"Customer: {snapshot.get('company')}",
        f"Use-case: {use_case.get('label')}",
        f"Hardware profile: {profile.get('label')}",
        f"Status: {snapshot.get('status')}",
        "",
        "## Safe boundary",
        SAFE_FIRMWARE_BOUNDARY,
        "",
        "## Required signals for this use-case",
    ]
    lines.extend([f"- {x}" for x in use_case.get("required_signals", [])])
    lines.extend(["", "## Features to collect or derive"])
    lines.extend([f"- {x}" for x in use_case.get("features", [])])
    lines.extend(["", "## Next steps"])
    lines.extend([f"- {x}" for x in snapshot.get("next_steps", [])])
    lines.extend(["", "## Important warning", use_case.get("warning", "")])
    return "\n".join(lines) + "\n"


def render_streamlit_tab(st) -> Tuple[Dict[str, Any] | None, bytes | None]:
    st.header("Hardware Profile + Firmware Starter")
    st.caption(
        "For customers with no/limited data: choose an approved hardware profile, confirm the hardware, "
        "and generate a pilot data-collection firmware starter bundle."
    )

    with st.form("hardware_firmware_starter_form"):
        c1, c2 = st.columns(2)
        with c1:
            company = st.text_input("Company", "Demo Customer")
            use_case_key = st.selectbox("Use-case", list(USE_CASE_PROFILES.keys()), format_func=lambda k: USE_CASE_PROFILES[k]["label"])
            hardware_profile_key = st.selectbox("Approved hardware profile", list(APPROVED_HARDWARE_PROFILES.keys()), format_func=lambda k: APPROVED_HARDWARE_PROFILES[k]["label"])
            customer_confirmed_hardware = st.checkbox("Customer confirms this exact hardware/pinout will be used", value=False)
        with c2:
            mode = st.selectbox("Mode", ["evidence_logging", "baseline_learning", "sentinel_pilot", "calibration_only"], index=0)
            connectivity = st.selectbox("Connectivity/output", ["serial_csv", "lora_summary", "sd_csv_optional", "gateway_mqtt_optional", "csv_excel_json_export"], index=0)
            sample_rate = st.number_input("Sample rate", min_value=0, max_value=48000, value=int(APPROVED_HARDWARE_PROFILES[hardware_profile_key].get("default_sample_rate", 16000)), step=1000)
            fft_size = st.number_input("FFT size", min_value=0, max_value=4096, value=int(APPROVED_HARDWARE_PROFILES[hardware_profile_key].get("default_fft_size", 1024)), step=256)
            notes = st.text_area("Notes", "", height=80)
        submitted = st.form_submit_button("Build firmware starter pack", use_container_width=True)

    st.subheader("Approved hardware profiles")
    st.dataframe(build_profile_table(), use_container_width=True)

    if not submitted:
        st.info("Firmware is only generated after hardware confirmation. Until then this remains a reference profile, not usable field firmware.")
        return None, None

    snapshot = build_hardware_firmware_snapshot(
        company=company,
        use_case_key=use_case_key,
        hardware_profile_key=hardware_profile_key,
        customer_confirmed_hardware=customer_confirmed_hardware,
        mode=mode,
        connectivity=connectivity,
        sample_rate=int(sample_rate),
        fft_size=int(fft_size),
        notes=notes,
    )
    bundle = create_firmware_starter_bundle(snapshot)

    if snapshot["status"] == "ready_to_generate":
        st.success("Starter firmware bundle generated for controlled pilot data collection.")
    elif snapshot["status"] == "no_firmware_needed":
        st.info("No firmware needed for this route. Use the CSV/import schema for existing system exports.")
    else:
        st.warning("Hardware not confirmed yet. EdgeTwin prepares the profile and checklist, but does not generate usable firmware as final.")

    st.write(snapshot["safe_firmware_boundary"])
    st.subheader("Next steps")
    for step in snapshot.get("next_steps", []):
        st.write(f"- {step}")

    st.download_button(
        "Download Hardware/Firmware Starter Bundle",
        data=bundle,
        file_name="edgetwin_hardware_firmware_starter_pack.zip",
        mime="application/zip",
        use_container_width=True,
    )
    return snapshot, bundle
