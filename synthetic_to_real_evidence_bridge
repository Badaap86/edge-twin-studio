"""EdgeTwin Synthetic-to-Real Evidence Bridge.

Purpose:
- Make the Pilot Dataset System stronger without pretending synthetic data is production proof.
- Treat deep learning / domain adaptation as an advanced, gated lane instead of the default V1 claim.
- Move customers from starter/benchmark/synthetic data toward customer-specific baseline and labeled field evidence.

This module is intentionally explainable and dependency-light. It creates the governance,
method selection, intake templates, metric registry and customer-safe bundle needed before
heavier deep-learning/domain-adaptation work is used in paid delivery.
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

try:  # Optional internal bridge; do not hard-fail if missing.
    import pilot_dataset_system as pds
except Exception:  # pragma: no cover
    pds = None

MODULE = "Synthetic-to-Real Evidence Bridge"

SAFE_BRIDGE_BOUNDARY = (
    "Deep-learning, transfer-learning and domain-adaptation methods may make starter or synthetic datasets "
    "more realistic for pilot preparation, but they do not create customer-specific production validation. "
    "Every adapted dataset still requires customer baseline data, operating context, maintenance/event labels "
    "and field validation before production claims are allowed."
)

WHY_NOT_DEFAULT = (
    "Advanced domain-adaptation and generative methods are valuable, but not safe as the V1 default sales claim: "
    "they need target-domain samples, careful leakage-free evaluation, explainable limits, compute/support effort, "
    "licensing checks and strict claim control. In V1 EdgeTwin should use them as a gated advanced lane on top of "
    "benchmarks, physics-informed scenarios, aggregate calibration and real baseline collection."
)

METHOD_LANES: List[Dict[str, Any]] = [
    {
        "lane": "V1 default",
        "method": "Benchmark + physics-informed starter scenarios",
        "what_it_does": "Uses public/benchmark bearing data and engineered fault scenarios for learning, demo and pilot design.",
        "real_data_needed": "None for demo; customer validation still required.",
        "best_for": "No-data customers, sample evidence reports, first sales conversations.",
        "customer_facing_claim": "Good for pilot preparation and scenario understanding; not production validation.",
        "risk": "Benchmark/lab bias if presented as real factory proof.",
        "recommended_v1": "Yes",
    },
    {
        "lane": "V1 default",
        "method": "Customer aggregate calibration",
        "what_it_does": "Narrows starter ranges using customer-approved statistics such as RMS range, RPM, load, sample rate and sensor location.",
        "real_data_needed": "Aggregate/profile only; raw rows not required.",
        "best_for": "Customer-like starter datasets without storing full raw customer data.",
        "customer_facing_claim": "Customer-calibrated starter dataset; still non-production evidence.",
        "risk": "Aggregates can hide important failure/context patterns.",
        "recommended_v1": "Yes",
    },
    {
        "lane": "V1 default",
        "method": "Limited real-sample feature alignment",
        "what_it_does": "Compares a small real sample to starter data and aligns feature distributions such as RMS, kurtosis, crest factor and spectral bands.",
        "real_data_needed": "Small customer CSV/sample, preferably normal operation.",
        "best_for": "Moving L1/L2 toward L3 without promising model performance.",
        "customer_facing_claim": "Better realism and data-gap visibility; not fault prediction proof.",
        "risk": "Small samples may not cover enough operating regimes.",
        "recommended_v1": "Yes",
    },
    {
        "lane": "Advanced gated",
        "method": "Transfer learning / light domain adaptation",
        "what_it_does": "Uses a model or feature representation trained on source/benchmark data and adapts it to limited customer target data.",
        "real_data_needed": "Target-domain baseline sample plus context; ideally some event labels.",
        "best_for": "Customers with some real data but not enough labels for a full model.",
        "customer_facing_claim": "Research/advanced pilot preparation only unless validated on held-out customer field data.",
        "risk": "Can look impressive while failing under different load/speed/assets if evaluation leaks data.",
        "recommended_v1": "Optional add-on, not default",
    },
    {
        "lane": "Advanced gated",
        "method": "Generative time-series model (GAN/diffusion/VAE)",
        "what_it_does": "Learns to generate realistic vibration/audio-like sequences or augment rare fault scenarios.",
        "real_data_needed": "Enough representative source data; target data required for calibration and validation.",
        "best_for": "Rare fault scenario expansion when real fault data is scarce or impossible to intentionally create.",
        "customer_facing_claim": "Scenario augmentation and stress testing only; generated data is not proof of real fault behavior.",
        "risk": "High compute/evaluation burden, privacy/memorization checks, hard-to-explain realism, possible false confidence.",
        "recommended_v1": "Later / premium lab lane",
    },
    {
        "lane": "Advanced gated",
        "method": "Adversarial domain adaptation / deep representation alignment",
        "what_it_does": "Tries to make source and target feature spaces indistinguishable across domains.",
        "real_data_needed": "Source + target data split by physical assets, conditions and time.",
        "best_for": "Mature customers with enough baseline data and strong evaluation discipline.",
        "customer_facing_claim": "Experimental advanced calibration; customer-specific validation required.",
        "risk": "Complex, less transparent, can overfit or hide failure modes; not ideal for early trust-building.",
        "recommended_v1": "No, roadmap only",
    },
]

REALISM_METRICS: List[Dict[str, Any]] = [
    {"metric": "RMS distribution similarity", "customer_facing": "Yes", "why": "Shows whether vibration energy ranges are similar."},
    {"metric": "Kurtosis distribution similarity", "customer_facing": "Yes", "why": "Important for impulsive bearing/wear signatures."},
    {"metric": "Crest factor distribution similarity", "customer_facing": "Yes", "why": "Useful for peak-to-energy behavior and early impact patterns."},
    {"metric": "FFT/PSD band overlap", "customer_facing": "Yes", "why": "Shows whether dominant frequency bands are represented."},
    {"metric": "RPM/load/context coverage", "customer_facing": "Yes", "why": "Prevents confusing normal operating changes with faults."},
    {"metric": "Missingness/noise pattern similarity", "customer_facing": "Yes", "why": "Real factory data is messy; clean data is often unrealistic."},
    {"metric": "Feature-space MMD / Wasserstein distance", "customer_facing": "Internal", "why": "Good internal guardrail; too technical for most buyers."},
    {"metric": "Nearest-neighbor/memorization check", "customer_facing": "Internal", "why": "Important when using generative models or customer samples."},
    {"metric": "Leakage-free split check", "customer_facing": "Internal + summary", "why": "Avoids over-optimistic scores from same-bearing/same-run leakage."},
]

DATASET_SHORTLIST: List[Dict[str, Any]] = [
    {
        "dataset": "CWRU Bearing Data",
        "role_in_edge_twin": "Feature explanation, basic bearing fault demo, sample evidence report template.",
        "strength": "Well-known, documented seeded-fault benchmark with high sample-rate vibration data.",
        "limitation": "Lab/seeded faults; known leakage/generalization risks if used naively.",
        "safe_use": "Internal benchmark and customer demo with explicit non-production caveat.",
    },
    {
        "dataset": "NASA/IMS Bearing",
        "role_in_edge_twin": "Degradation/run-to-failure readiness examples and health trend demos.",
        "strength": "Better for degradation and time evolution than static class demos.",
        "limitation": "Still not customer-specific field validation; check repository terms/attribution.",
        "safe_use": "Benchmark/prognostics method demo and evidence-template development.",
    },
    {
        "dataset": "FEMTO / PRONOSTIA Bearing",
        "role_in_edge_twin": "Accelerated-life / RUL-like pilot-preparation examples.",
        "strength": "Useful for run-to-failure and remaining-life discussion.",
        "limitation": "Accelerated lab conditions; not proof for a customer's asset.",
        "safe_use": "Advanced benchmark and dataset-passport example.",
    },
    {
        "dataset": "Paderborn Bearing",
        "role_in_edge_twin": "Cross-condition, transfer and robustness checks.",
        "strength": "Useful when testing variable operating conditions and domain shift.",
        "limitation": "Still requires license/source verification and customer validation.",
        "safe_use": "Internal robustness benchmark and customer-calibration research lane.",
    },
]

PRODUCT_PACKS: List[Dict[str, Any]] = [
    {
        "pack": "Pilot Dataset Starter",
        "price_founder": "€950–€1.500 excl. VAT",
        "price_later": "€1.500–€2.500 excl. VAT",
        "fit": "No-data customer; needs starter dataset + bridge plan.",
        "included": "Starter dataset, scenario map, dataset passport, sample report, data collection blueprint.",
        "not_included": "Production validation, custom model training, onsite installation, PLC/SCADA integration.",
    },
    {
        "pack": "Customer-Calibrated Starter Dataset",
        "price_founder": "€2.500–€4.500 excl. VAT",
        "price_later": "€3.500–€5.500 excl. VAT",
        "fit": "Customer can share aggregate stats or a limited normal sample.",
        "included": "Feature distribution comparison, customer-like calibration, limitations report, bridge plan.",
        "not_included": "Production model accuracy claims or reusable customer raw data without explicit consent.",
    },
    {
        "pack": "Baseline-to-Evidence Pack",
        "price_founder": "€2.500–€5.500 excl. VAT",
        "price_later": "€4.500–€7.500+ excl. VAT",
        "fit": "Customer wants to start real baseline collection on one asset/fault family.",
        "included": "Sensor/data plan, schema, label protocol, baseline quality check, upgrade path to Evidence Pack.",
        "not_included": "Onsite hardware installation unless sold separately as controlled Field Data Kit route.",
    },
]


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _zip_bundle(files: Dict[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in files.items():
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            zf.writestr(name, payload)
    return buf.getvalue()


def _hash_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def build_method_lane_df() -> pd.DataFrame:
    return pd.DataFrame(METHOD_LANES)


def build_metric_registry_df() -> pd.DataFrame:
    return pd.DataFrame(REALISM_METRICS)


def build_dataset_shortlist_df() -> pd.DataFrame:
    return pd.DataFrame(DATASET_SHORTLIST)


def build_product_pack_df() -> pd.DataFrame:
    return pd.DataFrame(PRODUCT_PACKS)


def build_customer_aggregate_intake_template() -> pd.DataFrame:
    rows = [
        ("asset_id", "Unique asset/machine identifier", "Required", "motor_line_01"),
        ("asset_type", "Motor, pump, fan, conveyor, gearbox, etc.", "Required", "electric_motor"),
        ("bearing_type_or_geometry", "Bearing type/geometry if available", "Recommended", "6205 / unknown"),
        ("rpm_min", "Minimum operating RPM", "Recommended", "900"),
        ("rpm_max", "Maximum operating RPM", "Recommended", "1500"),
        ("load_profile", "Typical load bands or product modes", "Recommended", "40-80% load"),
        ("sensor_type", "Accelerometer/audio/current/temp", "Required", "accelerometer"),
        ("sensor_location", "Where the sensor is or will be mounted", "Required", "bearing housing drive end"),
        ("sample_rate_hz", "Raw sample rate or feature interval", "Required", "10000"),
        ("normal_rms_range", "Normal vibration RMS range", "Recommended", "0.2-0.6 g"),
        ("kurtosis_range", "Observed/expected kurtosis range", "Optional", "2.5-4.0"),
        ("crest_factor_range", "Observed/expected crest factor range", "Optional", "2.0-4.5"),
        ("known_fault_types", "Known or suspected fault types", "Recommended", "bearing wear, imbalance"),
        ("maintenance_events", "Available work orders / inspections / replacement dates", "Recommended", "bearing changed 2026-04-12"),
        ("operating_context_available", "RPM/load/current/product/shift/temp context?", "Required", "rpm + motor current"),
        ("data_privacy_notes", "Audio/operator/device identifiers present?", "Required", "no operator data"),
    ]
    return pd.DataFrame(rows, columns=["field", "description", "importance", "example"])


def _norm_score(value: float, max_value: float) -> int:
    if max_value <= 0:
        return 0
    return int(max(0, min(100, round(100 * (1 - min(value, max_value) / max_value)))))


def compare_starter_to_real_profile(starter_df: pd.DataFrame, real_profile: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Light, explainable similarity comparison.

    real_profile is intentionally aggregate-only. Example:
    {"vibration_rms_g_mean": 0.55, "vibration_rms_g_std": 0.12, "rpm_mean": 1450, "load_pct_mean": 70}
    """
    real_profile = real_profile or {}
    metrics: List[Dict[str, Any]] = []
    score_parts: List[int] = []
    for col, max_delta in [("vibration_rms_g", 1.0), ("kurtosis", 5.0), ("crest_factor", 5.0), ("rpm", 1500.0), ("load_pct", 100.0)]:
        if col in starter_df.columns:
            synth_mean = float(pd.to_numeric(starter_df[col], errors="coerce").dropna().mean())
            target = real_profile.get(f"{col}_mean")
            if target is None:
                metrics.append({
                    "metric": f"{col} mean similarity",
                    "status": "target missing",
                    "score": None,
                    "note": "Ask customer for aggregate profile or small real sample.",
                })
            else:
                try:
                    delta = abs(synth_mean - float(target))
                    score = _norm_score(delta, max_delta)
                    score_parts.append(score)
                    metrics.append({
                        "metric": f"{col} mean similarity",
                        "status": "computed",
                        "starter_mean": round(synth_mean, 4),
                        "target_mean": float(target),
                        "absolute_delta": round(delta, 4),
                        "score": score,
                    })
                except Exception:
                    metrics.append({"metric": f"{col} mean similarity", "status": "invalid target", "score": None})
    overall = int(round(float(np.mean(score_parts)))) if score_parts else 0
    return {
        "overall_similarity_score": overall,
        "metrics": metrics,
        "interpretation": "Realism support only; similarity does not equal production validation.",
    }


def build_bridge_plan(
    customer: str,
    current_level: str = "L1",
    target_level: str = "L4",
    use_deep_learning_lane: bool = False,
    has_aggregate_profile: bool = False,
    has_limited_sample: bool = False,
    has_baseline_collection: bool = False,
    has_labels: bool = False,
) -> Dict[str, Any]:
    if has_labels and has_baseline_collection:
        route = "Real-Data Evidence Pack"
        decision = "Move to real-data evidence review. Synthetic is now supporting context only."
    elif has_baseline_collection:
        route = "Baseline-to-Evidence Pack"
        decision = "Good bridge progress. Add labels/context before stronger pilot-readiness claims."
    elif has_limited_sample or has_aggregate_profile:
        route = "Customer-Calibrated Starter Dataset"
        decision = "Use feature alignment and customer-like calibration, then collect baseline."
    else:
        route = "Pilot Dataset Starter"
        decision = "Start with benchmark/physics-informed starter data and a real-world collection plan."

    advanced_gate = {
        "allowed": bool(use_deep_learning_lane and (has_limited_sample or has_baseline_collection)),
        "reason": (
            "Allowed as advanced pilot-preparation lane because target-domain data exists. Still requires leakage-free validation."
            if use_deep_learning_lane and (has_limited_sample or has_baseline_collection)
            else "Not default. Use only after aggregate/limited real data exists and evaluation boundaries are clear."
        ),
        "safe_claim": "Advanced adaptation can improve realism; it cannot replace customer-specific field validation.",
    }

    steps = [
        {"step": 1, "name": "Select one asset and one fault family", "output": "asset/fault scope", "why": "Stops the no-data route from becoming open-ended."},
        {"step": 2, "name": "Use benchmark + physics-informed starter data", "output": "starter dataset + scenario map", "why": "Gives immediate learning/demo value without needing customer failure data."},
        {"step": 3, "name": "Collect aggregate profile", "output": "customer-like ranges", "why": "Makes starter data more realistic without storing raw rows."},
        {"step": 4, "name": "Request limited normal sample", "output": "feature alignment + data gaps", "why": "First true contact with real machine behavior."},
        {"step": 5, "name": "Run 7-14 day baseline", "output": "customer-specific normal data", "why": "Moves from preparation to real-world evidence foundation."},
        {"step": 6, "name": "Attach maintenance/events/context", "output": "labels + operating context", "why": "Turns raw signals into useful evidence."},
        {"step": 7, "name": "Run Evidence Pack", "output": "Go / Conditional Go / No-Go", "why": "Production path still requires customer-specific evidence boundaries."},
    ]

    plan = {
        "module": MODULE,
        "created_at": _now(),
        "customer": customer,
        "current_level": current_level,
        "target_level": target_level,
        "recommended_route": route,
        "decision": decision,
        "use_deep_learning_lane_requested": bool(use_deep_learning_lane),
        "advanced_lane_gate": advanced_gate,
        "why_not_default": WHY_NOT_DEFAULT,
        "safe_boundary": SAFE_BRIDGE_BOUNDARY,
        "bridge_steps": steps,
        "recommended_claim_footer": "Advisory evidence only. Not production validation. Customer-specific field validation required.",
    }
    plan["plan_hash"] = _hash_obj(plan)
    return plan


def create_bridge_bundle(plan: Dict[str, Any], starter_df: pd.DataFrame | None = None, real_profile: Dict[str, Any] | None = None) -> bytes:
    if starter_df is None:
        if pds is not None:
            starter_df, manifest = pds.generate_pilot_starter_dataset(rows=1500, seed=143, target_realism="benchmark_physics_informed_starter")
        else:
            rng = np.random.default_rng(143)
            starter_df = pd.DataFrame({
                "timestamp": pd.date_range("2026-01-01", periods=1500, freq="5min"),
                "machine_id": [f"motor_{i % 3 + 1}" for i in range(1500)],
                "label": rng.choice(["normal", "early_wear", "imbalance"], 1500, p=[0.8, 0.12, 0.08]),
                "vibration_rms_g": rng.normal(0.45, 0.09, 1500).clip(0.05, 3.0),
                "kurtosis": rng.normal(3.4, 0.9, 1500).clip(1, 12),
                "crest_factor": rng.normal(3.1, 0.7, 1500).clip(1, 9),
                "rpm": rng.normal(1450, 35, 1500).clip(200, 4000),
                "load_pct": rng.normal(68, 16, 1500).clip(0, 120),
            })
            manifest = {"source": "fallback_starter"}
    else:
        manifest = {"source": "provided_starter_df"}

    comparison = compare_starter_to_real_profile(starter_df, real_profile)
    lines = [
        "# EdgeTwin Synthetic-to-Real Evidence Bridge",
        "",
        f"Customer: {plan.get('customer')}",
        f"Recommended route: {plan.get('recommended_route')}",
        f"Decision: {plan.get('decision')}",
        "",
        "## Why advanced deep-learning/domain-adaptation is gated",
        plan.get("why_not_default", WHY_NOT_DEFAULT),
        "",
        "## Safe boundary",
        plan.get("safe_boundary", SAFE_BRIDGE_BOUNDARY),
        "",
        "## Bridge steps",
    ]
    for step in plan.get("bridge_steps", []):
        lines.append(f"{step.get('step')}. {step.get('name')} — {step.get('output')} ({step.get('why')})")
    lines.extend([
        "",
        "## Claim footer",
        plan.get("recommended_claim_footer"),
    ])

    return _zip_bundle({
        "synthetic_to_real_bridge_plan.json": json.dumps(plan, indent=2, default=str),
        "synthetic_to_real_bridge_report.md": "\n".join(lines),
        "method_lane_matrix.csv": build_method_lane_df().to_csv(index=False),
        "realism_metric_registry.csv": build_metric_registry_df().to_csv(index=False),
        "dataset_shortlist.csv": build_dataset_shortlist_df().to_csv(index=False),
        "product_pack_pricing.csv": build_product_pack_df().to_csv(index=False),
        "customer_aggregate_intake_template.csv": build_customer_aggregate_intake_template().to_csv(index=False),
        "starter_dataset_preview.csv": starter_df.to_csv(index=False),
        "starter_manifest.json": json.dumps(manifest, indent=2, default=str),
        "real_profile_similarity.json": json.dumps(comparison, indent=2, default=str),
        "safe_bridge_boundary.txt": SAFE_BRIDGE_BOUNDARY,
    })


def render_streamlit_tab(st) -> Tuple[Dict[str, Any], bytes]:
    st.header("Synthetic-to-Real Evidence Bridge")
    st.write("Advanced lane for making starter/synthetic datasets more realistic while keeping production-claim boundaries honest.")
    st.info(SAFE_BRIDGE_BOUNDARY)

    c1, c2 = st.columns(2)
    with c1:
        customer = st.text_input("Customer / project", value="Founding customer - synthetic-to-real bridge", key="v143_customer")
        current_level = st.selectbox("Current realism level", ["L1", "L2", "L3", "L4"], index=0, key="v143_current_level")
        target_level = st.selectbox("Target realism level", ["L3", "L4", "L5", "L6"], index=1, key="v143_target_level")
        use_deep_learning_lane = st.checkbox("Request advanced deep-learning / domain-adaptation lane", value=False, key="v143_dl_lane")
    with c2:
        has_aggregate_profile = st.checkbox("Customer aggregate profile available", value=False, key="v143_profile")
        has_limited_sample = st.checkbox("Limited real customer sample available", value=False, key="v143_sample")
        has_baseline_collection = st.checkbox("7-14 day baseline collection available", value=False, key="v143_baseline")
        has_labels = st.checkbox("Maintenance / fault / event labels available", value=False, key="v143_labels")

    plan = build_bridge_plan(
        customer=customer,
        current_level=current_level,
        target_level=target_level,
        use_deep_learning_lane=use_deep_learning_lane,
        has_aggregate_profile=has_aggregate_profile,
        has_limited_sample=has_limited_sample,
        has_baseline_collection=has_baseline_collection,
        has_labels=has_labels,
    )

    bundle = create_bridge_bundle(plan)

    st.markdown("### Decision")
    k1, k2, k3 = st.columns(3)
    k1.metric("Recommended route", plan["recommended_route"])
    k2.metric("Advanced lane", "Allowed" if plan["advanced_lane_gate"]["allowed"] else "Gated")
    k3.metric("Production validation", "No")
    st.write(plan["decision"])

    if plan["advanced_lane_gate"]["allowed"]:
        st.success(plan["advanced_lane_gate"]["reason"])
    else:
        st.warning(plan["advanced_lane_gate"]["reason"])

    tabs = st.tabs(["Why not default DL", "Method lanes", "Bridge steps", "Metrics", "Datasets", "Pricing", "Intake template", "Raw plan"])
    with tabs[0]:
        st.write(WHY_NOT_DEFAULT)
        st.markdown("**Important:** advanced methods are useful, but they need real target data and leakage-free evaluation before they should influence customer-facing conclusions.")
    with tabs[1]:
        st.dataframe(build_method_lane_df(), use_container_width=True)
    with tabs[2]:
        st.dataframe(pd.DataFrame(plan["bridge_steps"]), use_container_width=True)
    with tabs[3]:
        st.dataframe(build_metric_registry_df(), use_container_width=True)
    with tabs[4]:
        st.dataframe(build_dataset_shortlist_df(), use_container_width=True)
    with tabs[5]:
        st.dataframe(build_product_pack_df(), use_container_width=True)
    with tabs[6]:
        st.dataframe(build_customer_aggregate_intake_template(), use_container_width=True)
    with tabs[7]:
        st.json(plan)

    st.download_button(
        "Download Synthetic-to-Real Evidence Bridge bundle",
        data=bundle,
        file_name="edgetwin_synthetic_to_real_evidence_bridge_bundle.zip",
        mime="application/zip",
        use_container_width=True,
        key="v143_download_bundle",
    )
    return plan, bundle
