"""EdgeTwin Studio Data Coverage Engine.

Purpose:
- Convert the customer's data situation into a safe next step.
- Give value when the customer has good data, limited data, or no data.
- Keep synthetic data honest: scenario expansion only, never customer-specific production proof.

This module is customer-facing and intentionally avoids production-accuracy guarantees.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import zipfile
from typing import Any, Dict, List, Tuple

import pandas as pd

MODULE = "Data Coverage Engine"

SAFE_SYNTHETIC_BOUNDARY = (
    "Real customer data is the evidence basis. Synthetic data may be used for scenario expansion, "
    "stress testing, missing-condition exploration and demo/regression only. Synthetic data does not "
    "replace customer-specific real-world validation and must not be presented as production proof."
)

DATA_ROUTES = {
    "good_real_data": {
        "label": "Good real data available",
        "recommended_pack": "Real-Data Evidence Pack",
        "route": "Use real customer data as the primary evidence basis.",
        "customer_value": "Move quickly toward a stronger pilot-readiness decision.",
    },
    "limited_real_data": {
        "label": "Limited real data available",
        "recommended_pack": "Real Data + Synthetic Evidence Expansion",
        "route": "Use real data as the basis and add synthetic scenarios only for coverage/stress testing.",
        "customer_value": "Avoid throwing away limited data while staying honest about uncertainty.",
    },
    "no_real_data": {
        "label": "No real data available yet",
        "recommended_pack": "Dataset Starter Pack / Hardware Profile / Field Data Kit",
        "route": "Create the measurement plan, schema, hardware/firmware starter and collection checklist first.",
        "customer_value": "Start collecting the right data instead of guessing or buying a premature AI pilot.",
    },
}

READINESS_QUESTIONS = [
    ("machine_sensor_data", "Machine/sensor data available", 18),
    ("timestamps", "Reliable timestamps", 14),
    ("asset_ids", "Asset/machine IDs", 12),
    ("maintenance_history", "Maintenance/failure history", 14),
    ("labels", "Failure/wear/event labels", 14),
    ("operating_context", "Operating context such as speed/load/shift/product", 12),
    ("enough_history", "Enough history or measurement duration", 10),
    ("privacy_ready", "Data can be shared safely/pseudonymised", 6),
]


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def _bool_score(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true", "ja", "available", "ready", "1"}
    return bool(value)


def build_2_minute_readiness_snapshot(
    company: str = "Customer",
    use_case: str = "predictive maintenance / bearing wear",
    machine_sensor_data: bool = True,
    timestamps: bool = True,
    asset_ids: bool = True,
    maintenance_history: bool = False,
    labels: bool = False,
    operating_context: bool = False,
    enough_history: bool = False,
    privacy_ready: bool = True,
    notes: str = "",
) -> Dict[str, Any]:
    """Build a fast customer-facing readiness snapshot and route recommendation."""
    answers = {
        "machine_sensor_data": _bool_score(machine_sensor_data),
        "timestamps": _bool_score(timestamps),
        "asset_ids": _bool_score(asset_ids),
        "maintenance_history": _bool_score(maintenance_history),
        "labels": _bool_score(labels),
        "operating_context": _bool_score(operating_context),
        "enough_history": _bool_score(enough_history),
        "privacy_ready": _bool_score(privacy_ready),
    }
    score = int(sum(weight for key, _label, weight in READINESS_QUESTIONS if answers.get(key)))

    missing = [label for key, label, _weight in READINESS_QUESTIONS if not answers.get(key)]
    present = [label for key, label, _weight in READINESS_QUESTIONS if answers.get(key)]

    if not answers["machine_sensor_data"]:
        route_key = "no_real_data"
        status = "Not ready yet"
    elif score >= 78 and answers["timestamps"] and answers["asset_ids"]:
        route_key = "good_real_data"
        status = "Likely ready for real-data evidence review"
    elif score >= 35:
        route_key = "limited_real_data"
        status = "Partly ready"
    else:
        route_key = "no_real_data"
        status = "Not ready yet"

    route = dict(DATA_ROUTES[route_key])

    risk_flags: List[str] = []
    if not answers["labels"]:
        risk_flags.append("No/weak labels: failure or wear claims must stay limited.")
    if not answers["maintenance_history"]:
        risk_flags.append("Missing maintenance history: root-cause interpretation will be limited.")
    if not answers["operating_context"]:
        risk_flags.append("Missing operating context: speed/load/shift/product effects may be confused with anomalies.")
    if not answers["privacy_ready"]:
        risk_flags.append("Data sharing/privacy is not ready: use pseudonymisation and minimisation first.")
    if not answers["enough_history"]:
        risk_flags.append("Short history: trend and degradation conclusions must remain cautious.")

    next_steps = {
        "good_real_data": [
            "Prepare a Real-Data Evidence Pack.",
            "Run Data Quality Gate on the provided export/sample.",
            "Document assumptions, limitations, safe claims and pilot KPI candidates.",
        ],
        "limited_real_data": [
            "Use the available real data as the evidence basis.",
            "Apply Synthetic Evidence Expansion only for scenario coverage and stress testing.",
            "Separate real-data findings from synthetic scenario notes in the report.",
        ],
        "no_real_data": [
            "Create a Dataset Starter Pack with required fields and labels.",
            "Choose an approved hardware/data-capture profile if live measurement is needed.",
            "Collect baseline data before any customer-specific predictive claim is considered.",
        ],
    }[route_key]

    safe_customer_summary = (
        f"{company} is currently assessed as: {status}. Recommended route: {route['recommended_pack']}. "
        "EdgeTwin can help define the right next step without promising production accuracy."
    )

    snapshot = {
        "module": MODULE,
        "created_at": _now(),
        "company": company,
        "use_case": use_case,
        "score": score,
        "status": status,
        "route_key": route_key,
        "route": route,
        "present_inputs": present,
        "missing_inputs": missing,
        "risk_flags": risk_flags,
        "next_steps": next_steps,
        "safe_synthetic_boundary": SAFE_SYNTHETIC_BOUNDARY,
        "safe_customer_summary": safe_customer_summary,
        "notes": notes,
    }
    snapshot["snapshot_hash"] = _sha(snapshot)
    return snapshot


def build_data_ladder_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, route in DATA_ROUTES.items():
        rows.append({
            "Data situation": route["label"],
            "Recommended route": route["recommended_pack"],
            "What EdgeTwin does": route["route"],
            "Customer value": route["customer_value"],
            "Selected": "Yes" if snapshot.get("route_key") == key else "No",
        })
    return pd.DataFrame(rows)


def build_missing_input_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({
        "Missing / weak input": snapshot.get("missing_inputs", []),
        "Why it matters": [
            _missing_reason(x) for x in snapshot.get("missing_inputs", [])
        ],
    })


def _missing_reason(label: str) -> str:
    low = str(label).lower()
    if "timestamps" in low:
        return "Without time alignment, trends and event matching become unreliable."
    if "asset" in low:
        return "Without asset IDs, data cannot be tied to the right machine/component."
    if "maintenance" in low:
        return "Without maintenance history, failure/wear interpretation remains limited."
    if "labels" in low:
        return "Without labels, anomaly signals cannot be safely treated as known failure types."
    if "context" in low:
        return "Without context, normal operating changes can look like faults."
    if "history" in low:
        return "Without enough history, degradation trends are weak."
    if "privacy" in low:
        return "Without safe sharing rules, upload/processing should be delayed or minimised."
    return "This input improves readiness and reduces pilot risk."


def create_data_coverage_bundle(snapshot: Dict[str, Any]) -> bytes:
    """Create a ZIP bundle for the customer/founder."""
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data_coverage_snapshot.json", json.dumps(snapshot, indent=2, ensure_ascii=False))
        zf.writestr("safe_synthetic_boundary.txt", SAFE_SYNTHETIC_BOUNDARY)
        zf.writestr("data_ladder.csv", build_data_ladder_table(snapshot).to_csv(index=False))
        zf.writestr("missing_inputs.csv", build_missing_input_table(snapshot).to_csv(index=False))
        md = [
            "# EdgeTwin Data Coverage Snapshot",
            "",
            f"Customer: {snapshot.get('company')}",
            f"Use-case: {snapshot.get('use_case')}",
            f"Readiness score: {snapshot.get('score')}%",
            f"Status: {snapshot.get('status')}",
            f"Recommended route: {snapshot.get('route', {}).get('recommended_pack')}",
            "",
            "## Safe boundary",
            SAFE_SYNTHETIC_BOUNDARY,
            "",
            "## Next steps",
        ]
        md.extend([f"- {x}" for x in snapshot.get("next_steps", [])])
        md.extend(["", "## Risk flags"])
        md.extend([f"- {x}" for x in snapshot.get("risk_flags", [])] or ["- No major flags from this quick check."])
        zf.writestr("DATA_COVERAGE_README.md", "\n".join(md))
    return mem.getvalue()


def render_streamlit_tab(st) -> Tuple[Dict[str, Any] | None, bytes | None]:
    st.header("Data Coverage Engine")
    st.caption(
        "A simple route for customers with good data, limited data, or no data. "
        "Synthetic data is kept honest: scenario expansion only, not production proof."
    )

    with st.form("data_coverage_engine_form"):
        c1, c2 = st.columns(2)
        with c1:
            company = st.text_input("Company", "Demo Customer")
            use_case = st.text_input("Use-case", "bearing wear / predictive maintenance")
            machine_sensor_data = st.checkbox("Machine/sensor data available", value=True)
            timestamps = st.checkbox("Reliable timestamps", value=True)
            asset_ids = st.checkbox("Asset/machine IDs", value=True)
            maintenance_history = st.checkbox("Maintenance/failure history", value=False)
        with c2:
            labels = st.checkbox("Failure/wear/event labels", value=False)
            operating_context = st.checkbox("Operating context such as speed/load/shift/product", value=False)
            enough_history = st.checkbox("Enough history / measurement duration", value=False)
            privacy_ready = st.checkbox("Data can be shared safely/pseudonymised", value=True)
            notes = st.text_area("Notes", "", height=90)
        submitted = st.form_submit_button("Build readiness snapshot", use_container_width=True)

    if not submitted:
        return None, None

    snapshot = build_2_minute_readiness_snapshot(
        company=company,
        use_case=use_case,
        machine_sensor_data=machine_sensor_data,
        timestamps=timestamps,
        asset_ids=asset_ids,
        maintenance_history=maintenance_history,
        labels=labels,
        operating_context=operating_context,
        enough_history=enough_history,
        privacy_ready=privacy_ready,
        notes=notes,
    )
    bundle = create_data_coverage_bundle(snapshot)

    m1, m2, m3 = st.columns(3)
    m1.metric("Readiness score", f"{snapshot['score']}%")
    m2.metric("Status", snapshot["status"])
    m3.metric("Recommended route", snapshot["route"]["recommended_pack"])

    st.success(snapshot["safe_customer_summary"])
    st.subheader("Data ladder")
    st.dataframe(build_data_ladder_table(snapshot), use_container_width=True)

    if snapshot.get("risk_flags"):
        st.subheader("Risk flags")
        for flag in snapshot["risk_flags"]:
            st.warning(flag)

    st.info(SAFE_SYNTHETIC_BOUNDARY)
    st.download_button(
        "Download Data Coverage Snapshot",
        data=bundle,
        file_name="edgetwin_data_coverage_snapshot.zip",
        mime="application/zip",
        use_container_width=True,
    )
    return snapshot, bundle
