"""EdgeTwin Studio Readiness Snapshot.

Purpose:
- Give a 2-minute customer-facing readiness check before a paid pack.
- Recommend a safe route: real-data evidence, synthetic expansion, dataset starter, or hardware/firmware starter.
- Keep claims honest: no production accuracy, legal/compliance or safety guarantees.
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
    import data_coverage_engine
except Exception:  # pragma: no cover - keeps module usable in partial installs
    data_coverage_engine = None

try:
    import hardware_profile_matrix
except Exception:  # pragma: no cover
    hardware_profile_matrix = None

MODULE = "Readiness Snapshot"

SAFE_BOUNDARY = (
    "This snapshot is an early readiness indication only. It is not a production-accuracy guarantee, "
    "legal approval, compliance certification, safety certification, or replacement for customer-specific validation."
)

QUESTION_WEIGHTS: List[Tuple[str, str, int, str]] = [
    ("real_data", "Machine/sensor/maintenance data is available", 16, "Without real data, customer-specific proof is not possible yet."),
    ("timestamps", "Reliable timestamps are present", 12, "Without timestamps, trends and event matching become unreliable."),
    ("asset_ids", "Machine/asset IDs are present", 10, "Without asset IDs, data cannot be tied to the right machine/component."),
    ("signal_quality", "Signal quality looks usable", 10, "Noisy or interrupted signals reduce evidence quality."),
    ("history", "Enough history or measurement duration exists", 10, "Short history weakens trend and degradation conclusions."),
    ("labels", "Failure/wear/maintenance labels exist", 14, "Without labels, anomaly signals must not be treated as known failure types."),
    ("operating_context", "RPM/load/shift/product/context is available", 12, "Without context, normal operating changes can look like faults."),
    ("privacy_ready", "Data can be shared safely or pseudonymised", 8, "Privacy/security not ready means upload and processing should be minimised or delayed."),
    ("business_owner", "A technical/business owner can confirm the use-case", 8, "Without an owner, pack scope and acceptance criteria become vague."),
]

DEFAULT_USE_CASES = {
    "bearing_wear": "Bearing wear / lager-slijtage",
    "machine_anomaly": "General machine anomaly / motor-pump-fan",
    "facade_security": "Facade / site security sentinel",
    "fire_arson_risk": "Fire / arson-risk context",
    "existing_industrial_export": "Existing industrial data export",
}

DEFAULT_SIGNALS = {
    "vibration_accel": "Vibration / accelerometer",
    "audio_i2s": "Audio / I2S microphone",
    "temperature": "Temperature",
    "rpm_load": "RPM / load context",
    "maintenance_labels": "Maintenance/failure labels",
    "radar": "Radar / presence/distance",
    "bme688_gas": "BME688 gas/environment",
    "pmu_power": "PMU / power health",
    "gps_asset": "GPS / location context",
    "lora_gateway": "LoRa/gateway connectivity",
    "existing_export": "Existing PLC/SCADA/historian/CMMS export",
}


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "ja", "true", "1", "available", "ready"}
    return bool(value)


def _route_from_answers(score: int, answers: Dict[str, bool]) -> Tuple[str, str, str]:
    if not answers.get("real_data"):
        return "no_real_data", "Not ready yet", "Dataset Starter Pack / Hardware Profile / Field Data Kit"
    if score >= 78 and answers.get("timestamps") and answers.get("asset_ids") and answers.get("privacy_ready"):
        return "good_real_data", "Likely ready for real-data evidence review", "Real-Data Evidence Pack"
    if score >= 40:
        return "limited_real_data", "Partly ready", "Real Data + Synthetic Evidence Expansion"
    return "no_real_data", "Not ready yet", "Dataset Starter Pack / Hardware Profile / Field Data Kit"


def _data_coverage_score(answers: Dict[str, bool]) -> Dict[str, int]:
    real = 0
    real += 25 if answers.get("real_data") else 0
    real += 20 if answers.get("timestamps") else 0
    real += 15 if answers.get("asset_ids") else 0
    real += 15 if answers.get("signal_quality") else 0
    real += 15 if answers.get("history") else 0
    real += 10 if answers.get("privacy_ready") else 0

    label = 0
    label += 55 if answers.get("labels") else 0
    label += 25 if answers.get("maintenance_history", False) else 0
    label += 20 if answers.get("business_owner") else 0

    context = 0
    context += 55 if answers.get("operating_context") else 0
    context += 20 if answers.get("asset_ids") else 0
    context += 15 if answers.get("timestamps") else 0
    context += 10 if answers.get("business_owner") else 0

    synthetic_usefulness = 30
    if answers.get("real_data") and not answers.get("labels"):
        synthetic_usefulness += 25
    if answers.get("real_data") and not answers.get("operating_context"):
        synthetic_usefulness += 10
    if answers.get("signal_quality") and answers.get("timestamps"):
        synthetic_usefulness += 15
    if not answers.get("real_data"):
        synthetic_usefulness = 55  # useful for demo/planning, not proof

    return {
        "real_data_coverage": min(100, real),
        "label_coverage": min(100, label),
        "context_coverage": min(100, context),
        "synthetic_expansion_usefulness": min(100, synthetic_usefulness),
    }


def build_snapshot(
    company: str = "Customer",
    use_case_key: str = "bearing_wear",
    available_signals: List[str] | None = None,
    notes: str = "",
    **answers: Any,
) -> Dict[str, Any]:
    """Build a customer-safe 2-minute readiness snapshot."""
    clean_answers = {key: _as_bool(answers.get(key, False)) for key, _label, _weight, _reason in QUESTION_WEIGHTS}
    score = int(sum(weight for key, _label, weight, _reason in QUESTION_WEIGHTS if clean_answers.get(key)))
    route_key, status, recommended_pack = _route_from_answers(score, clean_answers)

    available_signals = available_signals or []
    use_case_label = DEFAULT_USE_CASES.get(use_case_key, use_case_key)

    present = [label for key, label, _weight, _reason in QUESTION_WEIGHTS if clean_answers.get(key)]
    missing = [label for key, label, _weight, _reason in QUESTION_WEIGHTS if not clean_answers.get(key)]
    risks = [reason for key, _label, _weight, reason in QUESTION_WEIGHTS if not clean_answers.get(key)]

    data_scores = _data_coverage_score(clean_answers)

    hardware_snapshot = None
    if hardware_profile_matrix is not None:
        try:
            hardware_snapshot = hardware_profile_matrix.build_hardware_profile_snapshot(
                company=company,
                data_situation_key=route_key if route_key in {"good_real_data", "limited_real_data", "no_real_data"} else "limited_real_data",
                use_case_key=use_case_key,
                available_signals=available_signals,
                customer_confirms_hardware=False,
                wants_firmware_starter=(route_key != "good_real_data"),
            )
        except Exception as exc:  # keep snapshot resilient
            hardware_snapshot = {"error": str(exc)}

    if route_key == "good_real_data":
        next_steps = [
            "Run Data Quality Gate on a real export/sample.",
            "Prepare a Real-Data Evidence Pack with assumptions, limitations and safe claims.",
            "Define pilot KPIs and a go / conditional go / no-go decision.",
        ]
    elif route_key == "limited_real_data":
        next_steps = [
            "Use real data as the evidence basis.",
            "Apply Synthetic Evidence Expansion only for scenario coverage and stress testing.",
            "Collect missing labels/context before any production-performance claim.",
        ]
    else:
        next_steps = [
            "Create a Dataset Starter Pack and measurement plan.",
            "Choose the smallest reliable hardware profile for the use-case.",
            "Use pilot firmware only for data collection/evidence logging, not machine control.",
        ]

    summary = (
        f"{company}: {status}. Recommended route: {recommended_pack}. "
        "EdgeTwin provides a safe next step without promising production accuracy."
    )

    snapshot = {
        "module": MODULE,
        "created_at": _now(),
        "company": company,
        "use_case_key": use_case_key,
        "use_case": use_case_label,
        "available_signals": available_signals,
        "answers": clean_answers,
        "score": score,
        "status": status,
        "route_key": route_key,
        "recommended_pack": recommended_pack,
        "data_coverage_score": data_scores,
        "present_inputs": present,
        "missing_inputs": missing,
        "risk_notes": risks,
        "hardware_profile_preview": hardware_snapshot,
        "next_steps": next_steps,
        "safe_boundary": SAFE_BOUNDARY,
        "safe_customer_summary": summary,
        "notes": notes,
    }
    snapshot["snapshot_hash"] = _sha(snapshot)
    return snapshot


def build_snapshot_tables(snapshot: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    answer_rows = []
    for key, label, weight, reason in QUESTION_WEIGHTS:
        ok = bool(snapshot.get("answers", {}).get(key))
        answer_rows.append({
            "Check": label,
            "Present": "Yes" if ok else "No",
            "Weight": weight,
            "Why it matters": "Ready signal" if ok else reason,
        })

    scores = snapshot.get("data_coverage_score", {})
    score_rows = [
        {"Coverage area": name.replace("_", " ").title(), "Score": value}
        for name, value in scores.items()
    ]

    next_rows = [{"Next step": step} for step in snapshot.get("next_steps", [])]
    return {
        "answers": pd.DataFrame(answer_rows),
        "coverage": pd.DataFrame(score_rows),
        "next_steps": pd.DataFrame(next_rows),
    }


def create_snapshot_bundle(snapshot: Dict[str, Any]) -> bytes:
    tables = build_snapshot_tables(snapshot)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("readiness_snapshot.json", json.dumps(snapshot, indent=2, ensure_ascii=False))
        for name, df in tables.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False))
        md = [
            "# EdgeTwin Readiness Snapshot",
            "",
            f"Customer: {snapshot.get('company')}",
            f"Use-case: {snapshot.get('use_case')}",
            f"Score: {snapshot.get('score')}%",
            f"Status: {snapshot.get('status')}",
            f"Recommended route: {snapshot.get('recommended_pack')}",
            "",
            "## Data coverage",
        ]
        for key, value in snapshot.get("data_coverage_score", {}).items():
            md.append(f"- {key.replace('_', ' ').title()}: {value}%")
        md.extend(["", "## Missing / weak inputs"])
        md.extend([f"- {x}" for x in snapshot.get("missing_inputs", [])] or ["- None flagged in this quick check."])
        md.extend(["", "## Next steps"])
        md.extend([f"- {x}" for x in snapshot.get("next_steps", [])])
        md.extend(["", "## Safe boundary", SAFE_BOUNDARY])
        zf.writestr("README_snapshot.md", "\n".join(md))
    return mem.getvalue()


def render_streamlit_tab(st):
    st.header("Readiness Snapshot")
    st.write(
        "A short customer-safe check that routes the buyer to the right next step before a paid evidence pack."
    )
    st.info(SAFE_BOUNDARY)

    col1, col2 = st.columns(2)
    with col1:
        company = st.text_input("Company / project", value="Demo customer", key="snapshot_company")
        use_case_key = st.selectbox(
            "Use-case",
            list(DEFAULT_USE_CASES.keys()),
            format_func=lambda k: DEFAULT_USE_CASES.get(k, k),
            key="snapshot_use_case",
        )
        signal_options = hardware_profile_matrix.SENSOR_CATALOG if hardware_profile_matrix is not None else DEFAULT_SIGNALS
        available_signals = st.multiselect(
            "Available signals / hardware",
            options=list(signal_options.keys()),
            default=["vibration_accel", "audio_i2s"] if "vibration_accel" in signal_options else [],
            format_func=lambda k: signal_options.get(k, k),
            key="snapshot_signals",
        )
    with col2:
        notes = st.text_area(
            "Optional notes",
            value="Customer wants to understand if the data is ready for a controlled pilot.",
            key="snapshot_notes",
        )

    st.subheader("2-minute readiness check")
    answer_values: Dict[str, bool] = {}
    cols = st.columns(3)
    for idx, (key, label, _weight, _reason) in enumerate(QUESTION_WEIGHTS):
        default = key in {"real_data", "timestamps", "asset_ids", "signal_quality", "privacy_ready", "business_owner"}
        with cols[idx % 3]:
            answer_values[key] = st.checkbox(label, value=default, key=f"snapshot_q_{key}")

    snapshot = build_snapshot(
        company=company,
        use_case_key=use_case_key,
        available_signals=available_signals,
        notes=notes,
        **answer_values,
    )
    bundle = create_snapshot_bundle(snapshot)

    st.subheader("Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Readiness score", f"{snapshot['score']}%")
    m2.metric("Status", snapshot["status"])
    m3.metric("Recommended route", snapshot["recommended_pack"])

    st.write(snapshot["safe_customer_summary"])

    tables = build_snapshot_tables(snapshot)
    t1, t2, t3 = st.tabs(["Coverage", "Checks", "Next steps"])
    with t1:
        st.dataframe(tables["coverage"], use_container_width=True)
    with t2:
        st.dataframe(tables["answers"], use_container_width=True)
    with t3:
        st.dataframe(tables["next_steps"], use_container_width=True)

    hardware_preview = snapshot.get("hardware_profile_preview") or {}
    if isinstance(hardware_preview, dict) and not hardware_preview.get("error"):
        st.subheader("Hardware / firmware route preview")
        st.write(f"**Minimal profile:** {hardware_preview.get('minimal_profile', 'n/a')}")
        st.write(f"**Strong profile:** {hardware_preview.get('strong_profile', 'n/a')}")
        st.write(f"**Firmware starter:** {hardware_preview.get('firmware_bundle', {}).get('bundle_name', 'n/a')}")
        st.caption(hardware_preview.get("safe_claim", ""))

    st.download_button(
        "Download readiness snapshot bundle",
        data=bundle,
        file_name="edgetwin_readiness_snapshot.zip",
        mime="application/zip",
        use_container_width=True,
        key="snapshot_download_bundle",
    )
    return snapshot, bundle
