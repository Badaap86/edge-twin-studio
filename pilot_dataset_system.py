"""EdgeTwin Studio Pilot Dataset System.

Purpose:
- Turn "no data" customers into a safe, paid starting route.
- Provide starter/benchmark/synthetic datasets for learning, demo and pilot preparation.
- Show a clear path from synthetic or benchmark data toward customer-specific real data.
- Keep the production boundary honest: starter/synthetic/benchmark data is not customer production proof.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import zipfile
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from synthetic_data_optimizer import generate_scenario_dataset
except Exception:  # pragma: no cover
    generate_scenario_dataset = None

MODULE = "Pilot Dataset System"

SAFE_DATASET_BOUNDARY = (
    "Starter, benchmark and synthetic datasets are intended for orientation, demo, internal buy-in, "
    "method testing, scenario coverage and pilot preparation. They are not customer-specific production "
    "validation and must not be presented as proof that an AI model works on the customer's real machines."
)

REAL_DATA_GOAL = (
    "The goal is to move every no-data or low-data customer toward customer-specific measured data: "
    "first baseline measurements, then operating-context data, then labeled maintenance/event data, "
    "and finally a real-data evidence pack."
)

REALISM_LEVELS = [
    {
        "level": "L0",
        "name": "Synthetic starter dataset",
        "realness": 15,
        "what_it_is": "Generated scenario data for learning, demos and workflow testing.",
        "allowed_use": "Orientation, feature explanation, internal demo, method dry-run.",
        "not_allowed": "Customer production validation or accuracy claim.",
        "next_move": "Add benchmark/lab data or start a real baseline collection plan.",
    },
    {
        "level": "L1",
        "name": "Public/benchmark dataset",
        "realness": 35,
        "what_it_is": "External lab or benchmark data with known labels, usually not from the customer's machine.",
        "allowed_use": "Benchmarking, sample evidence report, method comparison.",
        "not_allowed": "Assuming lab accuracy transfers to the customer's factory.",
        "next_move": "Map benchmark signals to the customer's target asset and sensor plan.",
    },
    {
        "level": "L2",
        "name": "Customer aggregate profile",
        "realness": 50,
        "what_it_is": "Schema/statistical profile from customer data without reusable raw rows.",
        "allowed_use": "Calibrate scenario ranges, data-quality expectations and collection plan.",
        "not_allowed": "Training reusable production models on hidden customer data.",
        "next_move": "Request a small raw sample or start a controlled measurement window.",
    },
    {
        "level": "L3",
        "name": "Limited customer sample",
        "realness": 65,
        "what_it_is": "Small raw export/sample from the customer's actual machine or historian.",
        "allowed_use": "Early data quality review, schema validation, missing signal detection.",
        "not_allowed": "Strong failure prediction or production-readiness conclusion.",
        "next_move": "Collect longer baseline and operating-context data.",
    },
    {
        "level": "L4",
        "name": "Customer baseline collection",
        "realness": 78,
        "what_it_is": "Time-boxed measurement of normal operation on the target asset.",
        "allowed_use": "Baseline normality, sensor placement validation, context-gap detection.",
        "not_allowed": "Fault-class proof if no real fault/maintenance labels exist.",
        "next_move": "Add maintenance events, fault observations and operating context.",
    },
    {
        "level": "L5",
        "name": "Labeled customer pilot dataset",
        "realness": 90,
        "what_it_is": "Customer data with asset IDs, timestamps, context and maintenance/fault labels.",
        "allowed_use": "Real-Data Evidence Pack and controlled pilot-readiness decision.",
        "not_allowed": "Unbounded production guarantee without field validation.",
        "next_move": "Run Real-Data Evidence Pack and define pilot validation boundaries.",
    },
    {
        "level": "L6",
        "name": "Verified field evidence",
        "realness": 100,
        "what_it_is": "Customer-specific data validated across time, assets, context and events.",
        "allowed_use": "Strongest basis for pilot-to-production decision support.",
        "not_allowed": "Replacing human maintenance/safety decisions or legal certification.",
        "next_move": "Decide on production pilot, partner handoff or continued evidence collection.",
    },
]

SCENARIOS = {
    "bearing_wear_motor_health": {
        "label": "Bearing wear / motor-health",
        "asset_examples": "motors, bearings, fans, pumps, conveyors, rotating machinery",
        "signals": ["vibration_rms", "kurtosis", "crest_factor", "fft_peak_hz", "temperature_c", "current_a", "rpm", "load_pct"],
        "starter_labels": ["normal", "early_wear", "imbalance", "misalignment", "looseness", "sensor_noise", "load_variation"],
        "recommended_sampling": "Vibration/audio: 1-20 kHz depending sensor/use-case; summary features every 1-10 seconds for first evidence packs.",
        "minimum_real_window": "Start with 7-14 days baseline normal operation; 30-90 days preferred when maintenance events are rare.",
        "best_first_pack": "Dataset Starter Pack -> Professional Pilot Evidence Pack -> Real-Data Evidence Pack",
    },
    "pump_fan_health": {
        "label": "Pump/fan health",
        "asset_examples": "pumps, fans, blowers, HVAC industrial rotating assets",
        "signals": ["vibration_rms", "temperature_c", "current_a", "pressure", "flow_rate", "rpm", "load_pct"],
        "starter_labels": ["normal", "cavitation_risk", "imbalance", "bearing_noise", "load_variation"],
        "recommended_sampling": "Summary signals 1-10 seconds; vibration raw bursts when possible.",
        "minimum_real_window": "14-30 days baseline plus maintenance/context logs.",
        "best_first_pack": "Dataset Starter Pack or Hardware Profile + Firmware Starter",
    },
}

REAL_DATA_UPGRADE_OPTIONS = [
    {
        "option": "Use any existing machine export first",
        "how_it_gets_more_real": "Even a messy historian/SCADA/CSV sample reveals schema, cadence, missingness and context gaps.",
        "minimum_input": "Small CSV/export with timestamp, asset ID and at least one sensor/feature column.",
        "risk_control": "Treat as data-quality evidence only; no production claim.",
    },
    {
        "option": "Create a 7-14 day baseline window",
        "how_it_gets_more_real": "Moves from generic starter data to normal operation data from the customer's actual asset.",
        "minimum_input": "Sensor stream or feature CSV plus asset ID and operating notes.",
        "risk_control": "Baseline is normality/context evidence, not fault proof without labels.",
    },
    {
        "option": "Attach maintenance and event labels",
        "how_it_gets_more_real": "Turns raw time-series into evidence by linking anomalies to inspections, work orders or faults.",
        "minimum_input": "CMMS/work order exports, inspection notes, fault timestamps or technician labels.",
        "risk_control": "Keep label uncertainty visible; do not treat vague notes as hard ground truth.",
    },
    {
        "option": "Collect operating context",
        "how_it_gets_more_real": "Separates real faults from normal changes in speed, load, product, shift or environment.",
        "minimum_input": "RPM/load/product/shift/ambient context if available.",
        "risk_control": "Missing context is a report limitation and may force Conditional Go.",
    },
    {
        "option": "Use aggregate profile calibration",
        "how_it_gets_more_real": "Synthetic/starter scenarios are calibrated to customer-level ranges without storing reusable raw rows.",
        "minimum_input": "Customer-approved aggregate stats/schema profile.",
        "risk_control": "Profile-only learning requires explicit consent and remains non-production evidence.",
    },
    {
        "option": "Deploy approved Field Data Kit route",
        "how_it_gets_more_real": "Creates fresh customer-specific measurements when no useful data exists.",
        "minimum_input": "Approved asset/use-case, sensor placement plan, collection duration and safe firmware template.",
        "risk_control": "Pilot/data-collection template only; no production-certified hardware claim.",
    },
    {
        "option": "Build a consented evidence library",
        "how_it_gets_more_real": "With written permission, anonymized aggregate learnings improve starter templates over time.",
        "minimum_input": "Explicit customer consent, anonymization/profiling rules and retention boundary.",
        "risk_control": "No raw cross-customer reuse unless contractually approved; protect IP and trade secrets.",
    },
]


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def _zip_bundle(files: Dict[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, payload in files.items():
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            z.writestr(name, payload)
    return buf.getvalue()


def _safe_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except Exception:
        return default


def _safe_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return default


def build_realism_ladder_df(selected_level: str = "L0") -> pd.DataFrame:
    rows = []
    for item in REALISM_LEVELS:
        row = dict(item)
        row["selected"] = "Yes" if item["level"] == selected_level else "No"
        rows.append(row)
    return pd.DataFrame(rows)


def build_real_data_upgrade_options_df() -> pd.DataFrame:
    return pd.DataFrame(REAL_DATA_UPGRADE_OPTIONS)


def build_scenario_coverage_map(scenario_key: str = "bearing_wear_motor_health", include_synthetic: bool = True) -> pd.DataFrame:
    spec = SCENARIOS.get(scenario_key, SCENARIOS["bearing_wear_motor_health"])
    rows: List[Dict[str, Any]] = []
    for idx, label in enumerate(spec["starter_labels"]):
        if label == "normal":
            source = "starter baseline + customer baseline required"
            production_value = "low until customer baseline exists"
        elif label in {"sensor_noise", "load_variation"}:
            source = "synthetic stress scenario + customer context required"
            production_value = "method stress only"
        else:
            source = "synthetic/benchmark scenario first; customer event labels required"
            production_value = "non-production until customer event validation"
        rows.append({
            "scenario": label,
            "coverage_source": source if include_synthetic else "customer real data required",
            "pilot_preparation_value": "high" if label != "normal" else "medium",
            "production_evidence_value": production_value,
            "next_real_data_need": _next_real_need(label),
            "safe_claim": "pilot preparation only" if label != "normal" else "baseline preparation only",
        })
    return pd.DataFrame(rows)


def _next_real_need(label: str) -> str:
    if label == "normal":
        return "7-14 day customer baseline, then 30+ days preferred."
    if label == "sensor_noise":
        return "Known sensor type, placement, noise profile and rejected samples."
    if label == "load_variation":
        return "RPM/load/product/shift context linked to timestamps."
    return "Maintenance inspection, work order, fault timestamp or technician label."


def generate_pilot_starter_dataset(
    scenario_key: str = "bearing_wear_motor_health",
    rows: int = 1500,
    seed: int = 142,
    target_realism: str = "starter_synthetic",
    include_real_bridge_columns: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create a safe starter dataset for pilot preparation.

    It deliberately marks provenance and production validity so the dataset cannot be confused
    with customer production proof.
    """
    rows = _safe_int(rows, 1500, 120, 50000)
    seed = _safe_int(seed, 142, 1, 2_000_000_000)
    rng = np.random.default_rng(seed)

    if generate_scenario_dataset is not None:
        try:
            df, manifest = generate_scenario_dataset(
                pack_key="rotating_machinery",
                rows=rows,
                seed=seed,
                noise_level=0.07,
                missing_rate=0.012,
                drift_strength=0.05,
                imbalance_factor=1.15,
                include_edge_cases=True,
            )
        except Exception:
            df, manifest = _fallback_rotating_dataset(rows, rng), {"source": "fallback"}
    else:
        df, manifest = _fallback_rotating_dataset(rows, rng), {"source": "fallback"}

    # Normalize customer-friendly bearing/motor-health columns where possible.
    rename = {
        "vibration_rms": "vibration_rms_g",
        "current_a": "motor_current_a",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "vibration_rms_g" not in df.columns:
        base = rng.normal(0.42, 0.05, rows)
        df["vibration_rms_g"] = np.clip(base, 0.05, 4.0)
    if "kurtosis" not in df.columns:
        label_series = df["label"].astype(str) if "label" in df.columns else pd.Series(["normal"] * len(df))
        df["kurtosis"] = np.where(label_series.str.contains("normal", case=False, na=False), rng.normal(3.1, 0.4, len(df)), rng.normal(5.2, 1.0, len(df))).clip(1.0, 15.0)
    if "crest_factor" not in df.columns:
        df["crest_factor"] = (2.2 + df["kurtosis"] * 0.22 + rng.normal(0, 0.25, len(df))).clip(1.2, 9.0)
    if "fft_peak_hz" not in df.columns:
        df["fft_peak_hz"] = rng.choice([50, 100, 150, 250, 320, 420], size=len(df), p=[0.45, 0.18, 0.12, 0.1, 0.08, 0.07]) + rng.normal(0, 2.5, len(df))
    if "rpm" not in df.columns:
        df["rpm"] = rng.choice([960, 1450, 2900], size=len(df), p=[0.25, 0.55, 0.20]) + rng.normal(0, 18, len(df))
    if "load_pct" not in df.columns:
        df["load_pct"] = rng.normal(68, 18, len(df)).clip(5, 120)
    if "temperature_c" not in df.columns:
        df["temperature_c"] = rng.normal(58, 6, len(df)).clip(15, 120)
    if "machine_id" not in df.columns and "asset_id" not in df.columns:
        df["machine_id"] = [f"motor_{(i % 4) + 1:02d}" for i in range(len(df))]
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="5min")

    if include_real_bridge_columns:
        df["dataset_provenance"] = target_realism
        df["customer_specific"] = False
        df["production_validation"] = False
        df["requires_customer_real_data"] = True
        df["recommended_next_real_step"] = "Collect 7-14 day customer baseline + maintenance/event labels."

    # Keep readable ordering.
    preferred = [
        "timestamp", "machine_id", "asset_id", "label", "is_anomaly", "vibration_rms_g", "kurtosis", "crest_factor",
        "fft_peak_hz", "temperature_c", "motor_current_a", "current_amp", "rpm", "load_pct", "dataset_provenance",
        "customer_specific", "production_validation", "requires_customer_real_data", "recommended_next_real_step",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    manifest = {
        "module": MODULE,
        "created_at": _now(),
        "scenario_key": scenario_key,
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "target_realism": target_realism,
        "safe_boundary": SAFE_DATASET_BOUNDARY,
        "real_data_goal": REAL_DATA_GOAL,
        "source_manifest": manifest,
        "hash_sha256": hashlib.sha256(df.to_csv(index=False).encode("utf-8", errors="replace")).hexdigest(),
    }
    return df, manifest


def _fallback_rotating_dataset(rows: int, rng: np.random.Generator) -> pd.DataFrame:
    labels = rng.choice(["normal", "early_wear", "imbalance", "misalignment", "looseness"], rows, p=[0.72, 0.10, 0.07, 0.06, 0.05])
    ts = pd.date_range("2026-01-01", periods=rows, freq="5min")
    vib = rng.normal(0.42, 0.08, rows)
    vib += np.where(labels == "early_wear", rng.normal(0.18, 0.04, rows), 0)
    vib += np.where(labels == "imbalance", rng.normal(0.28, 0.06, rows), 0)
    vib += np.where(labels == "misalignment", rng.normal(0.34, 0.08, rows), 0)
    vib += np.where(labels == "looseness", rng.normal(0.45, 0.12, rows), 0)
    temp = rng.normal(55, 5, rows) + np.where(labels != "normal", rng.normal(4, 2, rows), 0)
    current = rng.normal(3.6, 0.3, rows) + np.where(labels != "normal", rng.normal(0.4, 0.2, rows), 0)
    return pd.DataFrame({
        "timestamp": ts,
        "machine_id": [f"motor_{(i % 4) + 1:02d}" for i in range(rows)],
        "label": labels,
        "is_anomaly": labels != "normal",
        "vibration_rms_g": vib.clip(0.05, 4.0),
        "temperature_c": temp.clip(10, 130),
        "motor_current_a": current.clip(0.1, 20),
    })


def build_dataset_evidence_passport(
    customer: str = "Customer",
    scenario_key: str = "bearing_wear_motor_health",
    current_level: str = "L0",
    target_level: str = "L5",
    rows: int = 1500,
    has_customer_baseline: bool = False,
    has_labels: bool = False,
    has_operating_context: bool = False,
    has_maintenance_history: bool = False,
) -> Dict[str, Any]:
    spec = SCENARIOS.get(scenario_key, SCENARIOS["bearing_wear_motor_health"])
    level_map = {x["level"]: x for x in REALISM_LEVELS}
    current = level_map.get(current_level, level_map["L0"])
    target = level_map.get(target_level, level_map["L5"])

    score = int(current["realness"])
    if has_customer_baseline:
        score = max(score, 78)
    if has_labels and has_maintenance_history and has_operating_context:
        score = max(score, 90)
    elif has_labels or has_maintenance_history:
        score = max(score, 72)
    elif has_operating_context:
        score = max(score, 68)
    score = min(100, score)

    missing_to_real = []
    if not has_customer_baseline:
        missing_to_real.append("customer-specific baseline measurement")
    if not has_labels:
        missing_to_real.append("failure/wear/event labels")
    if not has_operating_context:
        missing_to_real.append("operating context: rpm/load/product/shift/environment")
    if not has_maintenance_history:
        missing_to_real.append("maintenance history or inspection notes")

    if score >= 90:
        recommended_route = "Real-Data Evidence Pack"
        decision = "Ready for real-data evidence review"
    elif score >= 65:
        recommended_route = "Professional Pilot Evidence Pack + Real-World Data Bridge"
        decision = "Promising, but still needs stronger customer evidence"
    else:
        recommended_route = "Dataset Starter Pack + Real-World Data Bridge"
        decision = "Starter/pilot preparation only"

    passport = {
        "module": MODULE,
        "created_at": _now(),
        "customer": customer,
        "scenario_key": scenario_key,
        "scenario_label": spec["label"],
        "asset_examples": spec["asset_examples"],
        "current_realism_level": current,
        "target_realism_level": target,
        "dataset_rows_planned": int(rows),
        "signals": spec["signals"],
        "starter_labels": spec["starter_labels"],
        "recommended_sampling": spec["recommended_sampling"],
        "minimum_real_window": spec["minimum_real_window"],
        "has_customer_baseline": bool(has_customer_baseline),
        "has_labels": bool(has_labels),
        "has_operating_context": bool(has_operating_context),
        "has_maintenance_history": bool(has_maintenance_history),
        "realism_score": score,
        "pilot_preparation_value": "high" if score >= 35 else "medium",
        "production_evidence_value": "low" if score < 78 else ("medium" if score < 90 else "stronger, still bounded"),
        "missing_to_reach_real_data": missing_to_real,
        "recommended_route": recommended_route,
        "decision": decision,
        "safe_boundary": SAFE_DATASET_BOUNDARY,
        "real_data_goal": REAL_DATA_GOAL,
        "claim_rule": "The dataset can support pilot preparation. Production validity requires customer-specific real-world validation.",
    }
    passport["passport_hash"] = _sha_obj(passport)
    return passport


def create_pilot_dataset_system_bundle(passport: Dict[str, Any], dataset_df: pd.DataFrame, manifest: Dict[str, Any]) -> bytes:
    coverage = build_scenario_coverage_map(passport.get("scenario_key", "bearing_wear_motor_health"))
    ladder = build_realism_ladder_df(passport.get("current_realism_level", {}).get("level", "L0"))
    upgrade = build_real_data_upgrade_options_df()
    report_lines = [
        "# EdgeTwin Pilot Dataset System",
        "",
        f"Customer: {passport.get('customer')}",
        f"Scenario: {passport.get('scenario_label')}",
        f"Decision: {passport.get('decision')}",
        f"Recommended route: {passport.get('recommended_route')}",
        f"Realism score: {passport.get('realism_score')} / 100",
        "",
        "## Safe boundary",
        passport.get("safe_boundary", SAFE_DATASET_BOUNDARY),
        "",
        "## To move closer to real data",
    ]
    for item in passport.get("missing_to_reach_real_data", []):
        report_lines.append(f"- {item}")
    report_lines.extend([
        "",
        "## Real-data goal",
        passport.get("real_data_goal", REAL_DATA_GOAL),
    ])
    return _zip_bundle({
        "pilot_dataset_system_passport.json": json.dumps(passport, indent=2, default=str),
        "pilot_starter_dataset.csv": dataset_df.to_csv(index=False),
        "pilot_starter_dataset_manifest.json": json.dumps(manifest, indent=2, default=str),
        "scenario_coverage_map.csv": coverage.to_csv(index=False),
        "realism_ladder.csv": ladder.to_csv(index=False),
        "real_data_upgrade_options.csv": upgrade.to_csv(index=False),
        "pilot_dataset_report.md": "\n".join(report_lines),
        "safe_dataset_policy.txt": SAFE_DATASET_BOUNDARY,
    })


def render_streamlit_tab(st) -> Tuple[Dict[str, Any] | None, bytes | None]:
    st.header("Pilot Dataset System")
    st.write(
        "For customers with no usable data yet: provide a safe starter dataset, scenario coverage and a clear route toward customer-specific real data."
    )
    st.info(SAFE_DATASET_BOUNDARY)

    c1, c2 = st.columns(2)
    with c1:
        customer = st.text_input("Customer / project", value="Founding customer - no-data route", key="pds_customer")
        scenario_key = st.selectbox(
            "First use-case focus",
            list(SCENARIOS.keys()),
            format_func=lambda k: SCENARIOS[k]["label"],
            key="pds_scenario",
        )
        rows = st.slider("Starter dataset rows", min_value=500, max_value=5000, value=1500, step=250, key="pds_rows")
        current_level = st.selectbox(
            "Current dataset realism level",
            [x["level"] for x in REALISM_LEVELS],
            index=0,
            format_func=lambda lvl: f"{lvl} - {next(x['name'] for x in REALISM_LEVELS if x['level'] == lvl)}",
            key="pds_current_level",
        )
    with c2:
        st.markdown("### Real-data bridge inputs")
        has_customer_baseline = st.checkbox("Customer baseline data exists", value=False, key="pds_baseline")
        has_labels = st.checkbox("Failure/wear/event labels exist", value=False, key="pds_labels")
        has_operating_context = st.checkbox("Operating context exists", value=False, key="pds_context")
        has_maintenance_history = st.checkbox("Maintenance / inspection history exists", value=False, key="pds_maintenance")

    spec = SCENARIOS[scenario_key]
    st.markdown("### What this pack prepares")
    cols = st.columns(3)
    cols[0].metric("Starter route", "Dataset Starter")
    cols[1].metric("Target route", "Real-Data Evidence")
    cols[2].metric("Production proof", "No")
    st.write(f"**Assets:** {spec['asset_examples']}")
    st.write(f"**Recommended sampling:** {spec['recommended_sampling']}")
    st.write(f"**Minimum real-data window:** {spec['minimum_real_window']}")

    passport = build_dataset_evidence_passport(
        customer=customer,
        scenario_key=scenario_key,
        current_level=current_level,
        rows=rows,
        has_customer_baseline=has_customer_baseline,
        has_labels=has_labels,
        has_operating_context=has_operating_context,
        has_maintenance_history=has_maintenance_history,
    )
    df, manifest = generate_pilot_starter_dataset(scenario_key=scenario_key, rows=rows)
    bundle = create_pilot_dataset_system_bundle(passport, df, manifest)

    st.markdown("### Dataset Evidence Passport")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Realism score", f"{passport['realism_score']}%")
    k2.metric("Pilot prep value", passport["pilot_preparation_value"].title())
    k3.metric("Production evidence", passport["production_evidence_value"].title())
    k4.metric("Route", passport["recommended_route"])

    if passport["missing_to_reach_real_data"]:
        st.markdown("### Missing to move closer to real data")
        for item in passport["missing_to_reach_real_data"]:
            st.warning(item)
    else:
        st.success("Strong real-data ingredients are present. Run a Real-Data Evidence Pack next.")

    tab1, tab2, tab3, tab4 = st.tabs(["Starter dataset", "Coverage map", "Realism ladder", "Upgrade options"])
    with tab1:
        st.dataframe(df.head(200), use_container_width=True)
        st.caption("Dataset is clearly marked as non-production validation and customer-specific=False.")
    with tab2:
        st.dataframe(build_scenario_coverage_map(scenario_key), use_container_width=True)
    with tab3:
        st.dataframe(build_realism_ladder_df(current_level), use_container_width=True)
    with tab4:
        st.dataframe(build_real_data_upgrade_options_df(), use_container_width=True)

    st.download_button(
        "Download Pilot Dataset System bundle",
        data=bundle,
        file_name="edgetwin_pilot_dataset_system_bundle.zip",
        mime="application/zip",
        use_container_width=True,
        key="pds_download_bundle",
    )

    return passport, bundle
