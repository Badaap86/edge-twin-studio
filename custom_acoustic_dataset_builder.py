"""EdgeTwin Custom Acoustic Dataset Builder.

Purpose:
- Add a configurable audio/acoustic dataset route without turning EdgeTwin into a guaranteed detection/security system.
- Let customers define their own sound event, site profile, microphone setup, privacy mode and false-positive risks.
- Build a safe synthetic-to-real acoustic bridge: synthetic/public event ideas -> real background profile -> limited site sample -> baseline -> labeled field evidence.

Important boundary:
Raw audio can contain speech or worker/location signals. This module is designed around
feature-first, privacy-safe dataset planning and evidence preparation, not people recognition,
voice ID, emotion inference, worker scoring, or security/safety guarantees.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import math
import zipfile
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

MODULE = "Custom Acoustic Dataset Builder"

SAFE_AUDIO_POLICY = (
    "Audio is used for dataset preparation, scenario analysis and field-evidence planning. "
    "Raw audio should stay local by default. Prefer features, spectrograms, event metadata and short consented samples. "
    "No voice identification, no speaker recognition, no worker scoring, no emotion inference, no surveillance guarantee and no safety/security certification. "
    "Customer-specific field validation is required before operational claims."
)

CLAIM_FOOTER = (
    "Advisory evidence only. Not production validation. Not a guaranteed detection, security or safety system. "
    "No person identification, no voice ID, no worker scoring. Customer-specific field validation required."
)

ACOUSTIC_REALISM_LADDER: List[Dict[str, Any]] = [
    {
        "level": "A0",
        "name": "Synthetic acoustic event",
        "value": "Fast demo and scenario thinking without customer audio.",
        "limitation": "Not site-realistic and not field evidence.",
        "next_step": "Blend with public/background profiles or ask for a site background sample.",
    },
    {
        "level": "A1",
        "name": "Public/environmental audio reference",
        "value": "Useful for general event categories and feature examples.",
        "limitation": "Different microphones, distances, weather and background noise than customer site.",
        "next_step": "Create a site profile and identify false-positive sources.",
    },
    {
        "level": "A2",
        "name": "Customer site profile",
        "value": "Adds environment, microphone, distance, weather and noise-floor assumptions.",
        "limitation": "Still assumption-driven without real site audio/features.",
        "next_step": "Collect a short privacy-safe background sample or feature extract.",
    },
    {
        "level": "A3",
        "name": "Limited real site background sample",
        "value": "First real noise-floor, echo, wind, machinery and day/night context.",
        "limitation": "Usually has no labeled target events yet.",
        "next_step": "Generate customer-like scenarios by mixing event profiles with real background features.",
    },
    {
        "level": "A4",
        "name": "7-14 day acoustic baseline",
        "value": "Captures real normal background, recurring false positives and operating patterns.",
        "limitation": "Baseline alone does not prove target-event detection.",
        "next_step": "Add consented/labeled events and false-positive review.",
    },
    {
        "level": "A5",
        "name": "Labeled acoustic pilot dataset",
        "value": "Real site events, labels, false positives and context for pilot-readiness evidence.",
        "limitation": "Labels still need review; not a guarantee of operational detection.",
        "next_step": "Run Evidence Pack and define pilot acceptance criteria.",
    },
    {
        "level": "A6",
        "name": "Verified acoustic field evidence",
        "value": "Highest evidence layer: site-specific, reviewed, labeled and bounded.",
        "limitation": "Still not a legal/safety/security certification by itself.",
        "next_step": "Use only for controlled pilot decisions and human-reviewed operational validation.",
    },
]

AUDIO_ROUTES: List[Dict[str, Any]] = [
    {
        "route": "Machine-health acoustic add-on",
        "primary_fit": "motors, bearings, pumps, fans, belts, cavitation, scraping/looseness",
        "default_positioning": "EdgeTwin machine-health evidence add-on",
        "risk_level": "Medium",
        "privacy_note": "Lower than terrain audio when mic is close to machine and raw audio is not stored.",
    },
    {
        "route": "Custom acoustic dataset route",
        "primary_fit": "process sounds, leaks, alarms, metal impact, custom industrial events",
        "default_positioning": "Custom Audio Dataset Builder",
        "risk_level": "Medium-High",
        "privacy_note": "Need data minimization, event features and clear raw-audio policy.",
    },
    {
        "route": "OMEGA-X acoustic field-data route",
        "primary_fit": "terrain/security/building site/forest/agri acoustic events",
        "default_positioning": "Custom field-evidence route, not a guaranteed security system",
        "risk_level": "High",
        "privacy_note": "Terrain audio can capture people, speech and activity patterns; strict policy required.",
    },
]

EVENT_LIBRARY: List[Dict[str, Any]] = [
    {"category": "machine_health", "event": "bearing_squeal_or_rattle", "bands": "1-8 kHz", "false_positives": "nearby machinery, loose panels, belts"},
    {"category": "machine_health", "event": "pump_cavitation", "bands": "broadband + high-frequency bursts", "false_positives": "valves, flow turbulence, air tools"},
    {"category": "machine_health", "event": "belt_slip", "bands": "tonal squeal / harmonics", "false_positives": "brakes, fans, compressed air"},
    {"category": "machine_health", "event": "scraping_metal", "bands": "broadband harsh/noisy", "false_positives": "maintenance work, conveyors, pallets"},
    {"category": "terrain_security", "event": "angle_grinder", "bands": "high-energy broadband + motor tone", "false_positives": "legitimate maintenance, power tools nearby"},
    {"category": "terrain_security", "event": "chainsaw", "bands": "engine harmonics + cutting bursts", "false_positives": "two-stroke engines, brush cutters"},
    {"category": "terrain_security", "event": "glass_break", "bands": "sharp impulse + high-frequency decay", "false_positives": "metal impact, dropped tools"},
    {"category": "terrain_security", "event": "fence_impact", "bands": "low-mid impulse + rattle", "false_positives": "wind, animals, gates"},
    {"category": "terrain_security", "event": "vehicle_approach", "bands": "low-frequency engine/tires", "false_positives": "public road traffic, tractors"},
    {"category": "terrain_security", "event": "drone_or_propeller", "bands": "narrowband harmonics", "false_positives": "fans, insects, small engines"},
    {"category": "process_audio", "event": "compressed_air_leak", "bands": "high-frequency hiss", "false_positives": "air tools, ventilation"},
    {"category": "process_audio", "event": "steam_leak", "bands": "broadband hiss + pressure context", "false_positives": "cleaning, valves, compressed air"},
    {"category": "process_audio", "event": "alarm_beeper", "bands": "repeating tonal signal", "false_positives": "forklift reverse beepers, phones"},
]

BACKGROUND_PROFILES: List[Dict[str, Any]] = [
    {"environment": "factory_hall", "backgrounds": "machines, fans, compressed air, forklifts, echo", "baseline_need": "shift/product mode coverage"},
    {"environment": "outdoor_industrial_site", "backgrounds": "wind, rain, road traffic, birds, distant machinery", "baseline_need": "day/night + weather coverage"},
    {"environment": "construction_site", "backgrounds": "tools, vehicles, impacts, generators, voices", "baseline_need": "workday/non-workday split"},
    {"environment": "forest_or_agri_field", "backgrounds": "wind, birds, insects, tractors, rain, distant road", "baseline_need": "weather and season profile"},
    {"environment": "warehouse", "backgrounds": "forklifts, pallets, HVAC, alarms, doors", "baseline_need": "operating hours and loading events"},
]

FEATURE_PLAN: List[Dict[str, Any]] = [
    {"feature": "Spectrogram / STFT", "purpose": "Main time-frequency view for events.", "customer_facing": "Yes"},
    {"feature": "Band energy", "purpose": "Measures energy in event-relevant frequency bands.", "customer_facing": "Yes"},
    {"feature": "Audio RMS / loudness", "purpose": "Overall acoustic energy and event strength.", "customer_facing": "Yes"},
    {"feature": "Noise floor / SNR", "purpose": "Shows if the target event is detectable above background.", "customer_facing": "Yes"},
    {"feature": "Spectral centroid / bandwidth / rolloff", "purpose": "Simple shape descriptors for harsh, high or low sounds.", "customer_facing": "Yes"},
    {"feature": "Zero-crossing rate", "purpose": "Useful for noisy/impulsive sound patterns.", "customer_facing": "Internal + summary"},
    {"feature": "MFCC summary", "purpose": "Useful representation for audio classification experiments.", "customer_facing": "Internal + summary"},
    {"feature": "Event duration and impulse count", "purpose": "Separates short impacts from continuous tools or machinery.", "customer_facing": "Yes"},
]

PRODUCT_PACKS: List[Dict[str, Any]] = [
    {
        "pack": "Acoustic Dataset Starter",
        "price_founder": "€950–€1.500 excl. VAT",
        "price_later": "€1.500–€2.500 excl. VAT",
        "fit": "Customer wants to test a custom audio event idea without existing audio data.",
        "included": "Use-case config, scenario map, feature plan, dataset passport, synthetic/background plan, claim/privacy boundary.",
        "not_included": "Guaranteed detection, raw-audio labeling at scale, onsite installation, production security monitoring.",
    },
    {
        "pack": "Customer-Calibrated Acoustic Dataset",
        "price_founder": "€2.500–€4.500 excl. VAT",
        "price_later": "€3.500–€5.500 excl. VAT",
        "fit": "Customer can provide site profile, background sample or feature extract.",
        "included": "Customer-like acoustic scenarios, false-positive map, background calibration, baseline plan, limitations report.",
        "not_included": "Long-term audio storage, person identification, worker monitoring, security certification.",
    },
    {
        "pack": "Acoustic Baseline-to-Evidence Pack",
        "price_founder": "€2.500–€5.500 excl. VAT",
        "price_later": "€4.500–€7.500+ excl. VAT",
        "fit": "Customer wants 7-14 day baseline and route to field evidence.",
        "included": "Baseline design, event/false-positive label protocol, privacy-safe data schema, first quality review, Evidence Pack route.",
        "not_included": "Hardware installation unless sold separately as OMEGA-X/Field Data Kit route.",
    },
    {
        "pack": "OMEGA-X Acoustic Field Data Route",
        "price_founder": "From €3.500–€7.500+ excl. VAT",
        "price_later": "Custom quote",
        "fit": "Terrain/security/remote-site audio needs controlled field-data collection.",
        "included": "Hardware profile, safe data-collection template, acoustic site plan, privacy boundary, validation route.",
        "not_included": "OMEGA-X core IP, production firmware, guaranteed security response, safety/compliance certification.",
    },
]


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _hash_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def _zip_bundle(files: Dict[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in files.items():
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            zf.writestr(name, payload)
    return buf.getvalue()


def build_realism_ladder_df() -> pd.DataFrame:
    return pd.DataFrame(ACOUSTIC_REALISM_LADDER)


def build_audio_routes_df() -> pd.DataFrame:
    return pd.DataFrame(AUDIO_ROUTES)


def build_event_library_df() -> pd.DataFrame:
    return pd.DataFrame(EVENT_LIBRARY)


def build_background_profiles_df() -> pd.DataFrame:
    return pd.DataFrame(BACKGROUND_PROFILES)


def build_feature_plan_df() -> pd.DataFrame:
    return pd.DataFrame(FEATURE_PLAN)


def build_product_pack_df() -> pd.DataFrame:
    return pd.DataFrame(PRODUCT_PACKS)


def recommend_route(use_case_type: str, raw_audio_storage: str, contains_people_risk: bool) -> Dict[str, Any]:
    if use_case_type == "Machine-health audio":
        route = "Machine-health acoustic add-on"
        risk = "Medium" if not contains_people_risk and raw_audio_storage != "Store raw audio" else "Medium-High"
    elif use_case_type in ["Terrain / security audio", "Forestry / agriculture audio", "Construction-site audio"]:
        route = "OMEGA-X acoustic field-data route"
        risk = "High"
    else:
        route = "Custom acoustic dataset route"
        risk = "Medium-High" if contains_people_risk or raw_audio_storage == "Store raw audio" else "Medium"

    privacy_gate = []
    if raw_audio_storage == "Store raw audio":
        privacy_gate.append("Raw audio storage requested: require explicit purpose, retention limit, access control and consent/legal basis review.")
    else:
        privacy_gate.append("Feature-first route preferred: raw audio does not need to leave the site by default.")
    if contains_people_risk:
        privacy_gate.append("People/speech risk present: no voice ID, no worker scoring, no emotion inference; consider local speech filtering and DPIA-style review.")
    else:
        privacy_gate.append("People/speech risk marked low, but keep privacy boundary in the deliverable.")

    return {
        "recommended_route": route,
        "risk_level": risk,
        "privacy_gate": privacy_gate,
        "safe_claim": CLAIM_FOOTER,
    }


def build_use_case_config(
    project: str,
    use_case_type: str,
    target_event: str,
    environment: str,
    microphone: str,
    distance_m: float,
    raw_audio_storage: str,
    contains_people_risk: bool,
    background_noise: str,
    false_positive_sources: str,
    sample_rate_hz: int,
    event_duration_s: float,
) -> Dict[str, Any]:
    route = recommend_route(use_case_type, raw_audio_storage, contains_people_risk)
    config = {
        "module": MODULE,
        "created_at": _now(),
        "project": project,
        "use_case_type": use_case_type,
        "target_event": target_event,
        "environment": environment,
        "microphone": microphone,
        "distance_m": float(distance_m),
        "raw_audio_storage": raw_audio_storage,
        "contains_people_or_speech_risk": bool(contains_people_risk),
        "background_noise": background_noise,
        "false_positive_sources": false_positive_sources,
        "sample_rate_hz": int(sample_rate_hz),
        "event_duration_s": float(event_duration_s),
        "route_recommendation": route,
        "safe_audio_policy": SAFE_AUDIO_POLICY,
        "claim_footer": CLAIM_FOOTER,
    }
    config["config_hash"] = _hash_obj(config)
    return config


def build_scenario_map(config: Dict[str, Any]) -> pd.DataFrame:
    target = config.get("target_event", "custom_event")
    env = config.get("environment", "site")
    distance = float(config.get("distance_m", 5.0))
    background = config.get("background_noise", "normal background")
    false_pos = config.get("false_positive_sources", "unknown")
    rows = [
        {
            "scenario": "clean_target_event",
            "event": target,
            "background": "minimal background",
            "distance_m": distance,
            "purpose": "Understand target signature before site noise.",
            "realism_level": "A0-A1",
        },
        {
            "scenario": "target_plus_customer_background",
            "event": target,
            "background": background,
            "distance_m": distance,
            "purpose": "Main synthetic-to-real scenario: event mixed with site-like background.",
            "realism_level": "A2-A3",
        },
        {
            "scenario": "low_snr_event",
            "event": target,
            "background": f"{background} + masking noise",
            "distance_m": distance * 1.5,
            "purpose": "Test weak/masked events and false negatives.",
            "realism_level": "A2-A4",
        },
        {
            "scenario": "weather_or_echo_variant",
            "event": target,
            "background": f"{env}: wind/rain/echo/reverb variant",
            "distance_m": distance,
            "purpose": "Test environment effects that often break clean audio demos.",
            "realism_level": "A2-A4",
        },
        {
            "scenario": "false_positive_challenge",
            "event": "no target event",
            "background": false_pos,
            "distance_m": distance,
            "purpose": "Map likely false positives before customer relies on alerts.",
            "realism_level": "A3-A5",
        },
        {
            "scenario": "baseline_only",
            "event": "no target event",
            "background": background,
            "distance_m": 0,
            "purpose": "Build normal profile and alert threshold assumptions.",
            "realism_level": "A4",
        },
    ]
    return pd.DataFrame(rows)


def build_false_positive_risk_map(config: Dict[str, Any]) -> pd.DataFrame:
    target = str(config.get("target_event", "custom_event")).lower()
    generic = [
        ("background masking", "Target event may be hidden by louder site noise.", "Collect baseline across shifts/weather and use SNR thresholds."),
        ("distance attenuation", "Event may be too weak at selected mic distance.", "Validate expected distance and microphone placement."),
        ("microphone variation", "Different microphones change spectral shape.", "Document mic model and avoid mixing uncontrolled devices."),
        ("privacy spillover", "Speech or people-related audio may be captured.", "Feature-first processing, speech filtering and no person analytics."),
    ]
    event_specific = []
    if "grinder" in target or "slijp" in target:
        event_specific.append(("legitimate tool use", "Maintenance tools may look like intrusion tool use.", "Add work schedule/context and human review route."))
    if "chainsaw" in target or "ketting" in target:
        event_specific.append(("similar engines", "Brush cutters or small engines can sound similar.", "Use duration, harmonic profile and site context."))
    if "glass" in target or "glas" in target:
        event_specific.append(("metal/drop impact", "Dropped tools or metal impacts may mimic sharp glass impulse.", "Use decay profile and secondary evidence if available."))
    if "bearing" in target or "lager" in target:
        event_specific.append(("nearby machine tones", "Other motors may create similar tonal bands.", "Use mic placement close to asset and load/RPM context."))
    if "leak" in target or "lek" in target or "lucht" in target:
        event_specific.append(("normal air tools/valves", "Compressed-air tools can mimic leaks.", "Collect normal pneumatic events as false-positive labels."))
    rows = event_specific + generic
    return pd.DataFrame(rows, columns=["risk", "why_it_matters", "mitigation"])


def build_audio_dataset_passport(config: Dict[str, Any]) -> Dict[str, Any]:
    scenario_df = build_scenario_map(config)
    privacy_mode = "feature-first" if config.get("raw_audio_storage") != "Store raw audio" else "raw-audio-review-required"
    production_score = 10
    pilot_score = 70
    if config.get("contains_people_or_speech_risk"):
        pilot_score -= 10
    if config.get("raw_audio_storage") == "Store raw audio":
        pilot_score -= 10
    if float(config.get("distance_m", 0)) > 25:
        pilot_score -= 5

    passport = {
        "dataset_type": "custom_acoustic_starter_or_bridge_dataset",
        "source_type": "synthetic/public/site-profile/limited-baseline depending on route",
        "use_case_type": config.get("use_case_type"),
        "target_event": config.get("target_event"),
        "environment": config.get("environment"),
        "sensor_type": config.get("microphone"),
        "sample_rate_hz": config.get("sample_rate_hz"),
        "scenario_coverage_count": int(len(scenario_df)),
        "label_quality": "planned; field labels required",
        "realism_score": "estimated after background/sample comparison",
        "production_evidence_score": production_score,
        "pilot_preparation_score": max(0, min(100, pilot_score)),
        "known_limitations": [
            "Synthetic/acoustic starter data is not proof of customer-site detection performance.",
            "Background, distance, microphone and weather can materially change performance.",
            "False positives must be tested with real site baseline data.",
        ],
        "allowed_use": ["pilot preparation", "scenario analysis", "feature planning", "baseline collection planning", "internal decision support"],
        "not_allowed_use": ["production validation", "security guarantee", "safety certification", "voice/person identification", "worker scoring"],
        "privacy_ip_status": privacy_mode,
        "customer_validation_required": True,
        "recommended_next_step": "Collect A3 limited background sample or run A4 7-14 day baseline before stronger evidence claims.",
        "claim_footer": CLAIM_FOOTER,
    }
    passport["passport_hash"] = _hash_obj(passport)
    return passport


def generate_acoustic_feature_dataset(config: Dict[str, Any], rows: int = 1000, seed: int = 144) -> pd.DataFrame:
    """Generate a privacy-safe acoustic feature dataset preview, not raw audio.

    This is only a feature-level starter/preview dataset. It intentionally does not create
    or store raw audio waveforms.
    """
    rng = np.random.default_rng(seed)
    target = str(config.get("target_event", "target_event"))
    env = str(config.get("environment", "site"))
    sr = int(config.get("sample_rate_hz", 16000))
    distance = max(0.5, float(config.get("distance_m", 8)))
    labels = rng.choice(["background", target, "false_positive"], size=rows, p=[0.72, 0.14, 0.14])
    distance_loss = 20 * math.log10(distance)

    base_noise = rng.normal(-52, 6, rows)
    event_boost = np.where(labels == target, rng.normal(18, 4, rows), 0)
    fp_boost = np.where(labels == "false_positive", rng.normal(14, 5, rows), 0)
    rms_db = base_noise + event_boost + fp_boost - distance_loss / 5
    snr_db = np.where(labels == "background", rng.normal(2, 2, rows), rng.normal(12, 5, rows)).clip(-10, 40)

    centroid_base = rng.normal(1800, 600, rows)
    if any(token in target.lower() for token in ["grinder", "slijp", "glass", "glas", "leak", "lek", "air"]):
        centroid_base += np.where(labels != "background", rng.normal(2500, 800, rows), 0)
    elif any(token in target.lower() for token in ["vehicle", "motor", "engine", "voertuig"]):
        centroid_base += np.where(labels != "background", rng.normal(-900, 300, rows), 0)
    else:
        centroid_base += np.where(labels != "background", rng.normal(600, 500, rows), 0)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=rows, freq="30s"),
        "environment": env,
        "target_event": target,
        "label": labels,
        "sample_rate_hz": sr,
        "audio_rms_db": np.round(rms_db, 2),
        "noise_floor_db": np.round(base_noise, 2),
        "snr_db": np.round(snr_db, 2),
        "spectral_centroid_hz": np.round(np.clip(centroid_base, 80, sr / 2 - 100), 1),
        "spectral_bandwidth_hz": np.round(rng.normal(1400, 500, rows).clip(100, sr / 2), 1),
        "zero_crossing_rate": np.round(rng.normal(0.08, 0.04, rows).clip(0, 1), 4),
        "event_duration_s": np.round(np.where(labels == "background", rng.normal(0.5, 0.2, rows), rng.normal(float(config.get("event_duration_s", 2.0)), 0.7, rows)).clip(0.05, 20), 2),
        "raw_audio_stored": config.get("raw_audio_storage") == "Store raw audio",
        "privacy_mode": "feature_first" if config.get("raw_audio_storage") != "Store raw audio" else "raw_audio_review_required",
    })
    return df


def build_baseline_collection_plan(config: Dict[str, Any]) -> Dict[str, Any]:
    environment = config.get("environment", "site")
    contains_people = bool(config.get("contains_people_or_speech_risk"))
    duration_days = 14 if environment in ["outdoor_industrial_site", "construction_site", "forest_or_agri_field"] or contains_people else 7
    if environment in ["forest_or_agri_field"]:
        duration_days = 14
    plan = {
        "scope": "one acoustic use-case, one site/asset zone, one microphone profile",
        "recommended_duration_days": duration_days,
        "minimum_context_fields": [
            "timestamp", "site_or_asset_id", "microphone_id", "microphone_location", "sample_rate_hz",
            "event_marker", "background_condition", "weather_or_shift", "known_maintenance_or_legitimate_activity", "operator/person-data flag",
        ],
        "recommended_storage": "features/spectrograms + short reviewed clips only if needed; raw audio local by default",
        "label_protocol": [
            "background", "target_event", "false_positive", "uncertain", "privacy_exclusion"
        ],
        "acceptance_for_next_pack": [
            "baseline covers typical operating periods", "false positives reviewed", "target event examples or controlled test plan exists", "privacy gate passed"
        ],
        "claim_boundary": CLAIM_FOOTER,
    }
    return plan


def create_acoustic_bundle(config: Dict[str, Any], feature_df: pd.DataFrame | None = None) -> bytes:
    feature_df = feature_df if feature_df is not None else generate_acoustic_feature_dataset(config, rows=1200)
    scenario_df = build_scenario_map(config)
    fp_df = build_false_positive_risk_map(config)
    passport = build_audio_dataset_passport(config)
    baseline_plan = build_baseline_collection_plan(config)

    report_lines = [
        "# EdgeTwin Custom Acoustic Dataset Builder",
        "",
        f"Project: {config.get('project')}",
        f"Use-case type: {config.get('use_case_type')}",
        f"Target event: {config.get('target_event')}",
        f"Recommended route: {config.get('route_recommendation', {}).get('recommended_route')}",
        "",
        "## Safe audio policy",
        SAFE_AUDIO_POLICY,
        "",
        "## What this prepares",
        "This bundle prepares acoustic scenario analysis, feature planning, false-positive review and baseline collection.",
        "",
        "## What this does not prove",
        "It does not prove production detection performance, security response reliability, safety compliance or person-related identification.",
        "",
        "## Recommended next step",
        passport.get("recommended_next_step"),
        "",
        "## Claim footer",
        CLAIM_FOOTER,
    ]

    return _zip_bundle({
        "custom_acoustic_config.json": json.dumps(config, indent=2, default=str),
        "custom_acoustic_report.md": "\n".join(report_lines),
        "audio_dataset_passport.json": json.dumps(passport, indent=2, default=str),
        "acoustic_scenario_map.csv": scenario_df.to_csv(index=False),
        "false_positive_risk_map.csv": fp_df.to_csv(index=False),
        "acoustic_feature_dataset_preview.csv": feature_df.to_csv(index=False),
        "baseline_collection_plan.json": json.dumps(baseline_plan, indent=2, default=str),
        "acoustic_realism_ladder.csv": build_realism_ladder_df().to_csv(index=False),
        "audio_routes.csv": build_audio_routes_df().to_csv(index=False),
        "event_library.csv": build_event_library_df().to_csv(index=False),
        "background_profiles.csv": build_background_profiles_df().to_csv(index=False),
        "feature_plan.csv": build_feature_plan_df().to_csv(index=False),
        "acoustic_product_pricing.csv": build_product_pack_df().to_csv(index=False),
        "safe_audio_policy.txt": SAFE_AUDIO_POLICY,
        "claim_footer.txt": CLAIM_FOOTER,
    })


def render_streamlit_tab(st) -> Tuple[Dict[str, Any], bytes]:
    st.header("Custom Acoustic Dataset Builder")
    st.write("Configure customer-specific audio use-cases as a safe dataset/evidence route, not as a guaranteed detection system.")
    st.info(SAFE_AUDIO_POLICY)

    c1, c2 = st.columns(2)
    with c1:
        project = st.text_input("Project / customer", value="Custom acoustic pilot", key="v144_project")
        use_case_type = st.selectbox(
            "Audio use-case type",
            ["Machine-health audio", "Custom industrial audio", "Terrain / security audio", "Construction-site audio", "Forestry / agriculture audio"],
            index=0,
            key="v144_use_case_type",
        )
        target_event = st.text_input("Target sound/event", value="bearing rattle / grinder / air leak", key="v144_target_event")
        environment = st.selectbox(
            "Environment",
            ["factory_hall", "outdoor_industrial_site", "construction_site", "forest_or_agri_field", "warehouse", "custom_site"],
            index=0,
            key="v144_environment",
        )
        microphone = st.selectbox(
            "Microphone / sensor profile",
            ["I2S MEMS microphone", "industrial microphone", "OMEGA-X acoustic node", "gateway microphone", "customer supplied microphone"],
            index=0,
            key="v144_microphone",
        )
    with c2:
        distance_m = st.number_input("Typical distance to sound source (m)", min_value=0.5, max_value=200.0, value=5.0, step=0.5, key="v144_distance")
        sample_rate_hz = st.selectbox("Sample rate", [8000, 16000, 22050, 32000, 44100, 48000], index=1, key="v144_sample_rate")
        event_duration_s = st.number_input("Typical event duration (s)", min_value=0.1, max_value=120.0, value=2.0, step=0.1, key="v144_duration")
        raw_audio_storage = st.selectbox("Raw audio policy", ["Do not store raw audio", "Store short reviewed clips only", "Store raw audio"], index=0, key="v144_raw_policy")
        contains_people_risk = st.checkbox("People/speech may be captured", value=False, key="v144_people_risk")

    background_noise = st.text_area("Background sounds / site noise", value="wind, machines, traffic, ventilation, forklifts, echo", key="v144_background")
    false_positive_sources = st.text_area("Likely false positives", value="maintenance tools, nearby machines, vehicles, animals, alarms", key="v144_false_positives")

    config = build_use_case_config(
        project=project,
        use_case_type=use_case_type,
        target_event=target_event,
        environment=environment,
        microphone=microphone,
        distance_m=float(distance_m),
        raw_audio_storage=raw_audio_storage,
        contains_people_risk=bool(contains_people_risk),
        background_noise=background_noise,
        false_positive_sources=false_positive_sources,
        sample_rate_hz=int(sample_rate_hz),
        event_duration_s=float(event_duration_s),
    )
    feature_df = generate_acoustic_feature_dataset(config, rows=800)
    bundle = create_acoustic_bundle(config, feature_df)
    passport = build_audio_dataset_passport(config)
    baseline_plan = build_baseline_collection_plan(config)

    st.markdown("### Route recommendation")
    k1, k2, k3 = st.columns(3)
    k1.metric("Recommended route", config["route_recommendation"]["recommended_route"])
    k2.metric("Risk level", config["route_recommendation"]["risk_level"])
    k3.metric("Production validation", "No")
    for item in config["route_recommendation"]["privacy_gate"]:
        st.write(f"- {item}")

    tabs = st.tabs(["Scenario map", "Dataset passport", "False positives", "Feature preview", "Baseline plan", "Routes", "Pricing", "Event library", "Raw config"])
    with tabs[0]:
        st.dataframe(build_scenario_map(config), use_container_width=True)
    with tabs[1]:
        st.json(passport)
    with tabs[2]:
        st.dataframe(build_false_positive_risk_map(config), use_container_width=True)
    with tabs[3]:
        st.dataframe(feature_df.head(200), use_container_width=True)
    with tabs[4]:
        st.json(baseline_plan)
    with tabs[5]:
        st.dataframe(build_audio_routes_df(), use_container_width=True)
        st.dataframe(build_realism_ladder_df(), use_container_width=True)
    with tabs[6]:
        st.dataframe(build_product_pack_df(), use_container_width=True)
    with tabs[7]:
        st.dataframe(build_event_library_df(), use_container_width=True)
        st.dataframe(build_background_profiles_df(), use_container_width=True)
        st.dataframe(build_feature_plan_df(), use_container_width=True)
    with tabs[8]:
        st.json(config)

    st.download_button(
        "Download Custom Acoustic Dataset Builder bundle",
        data=bundle,
        file_name="edgetwin_custom_acoustic_dataset_builder_bundle.zip",
        mime="application/zip",
        use_container_width=True,
        key="v144_download_bundle",
    )
    return config, bundle
