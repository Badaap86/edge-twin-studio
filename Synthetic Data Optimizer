
"""EdgeTwin V109 Synthetic Data Optimizer.

Purpose:
- Upgrade demo/synthetic data from simple random samples into scenario-based evidence data.
- Keep it local/offline and deterministic for repeatable demos/regression tests.
- Produce known ground truth so Data Quality Gates, Trust Ledger and Real-Data Bridge can be tested honestly.

Important boundary:
Synthetic data proves workflow, stress handling and expected behavior. It does not prove production accuracy.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

VERSION = "V109"
MODULE = "Synthetic Data Optimizer"


def _utc_now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = default
    return max(lo, min(hi, v))


def _safe_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    return max(lo, min(hi, v))


def _df_hash(df: pd.DataFrame) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "empty"
    payload = df.to_csv(index=False).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


SCENARIO_LIBRARY: Dict[str, Dict[str, Any]] = {
    "rotating_machinery": {
        "title": "Rotating Machinery Pack",
        "sensors": ["vibration_rms", "vibration_kurtosis", "temperature_c", "current_amp", "rpm", "acoustic_db"],
        "classes": ["normal", "bearing_wear", "imbalance", "overheating", "loose_mount"],
        "base_rate": {"normal": 0.64, "bearing_wear": 0.14, "imbalance": 0.10, "overheating": 0.07, "loose_mount": 0.05},
        "description": "Synthetic motor/rotating asset data with realistic overlap, drift, noise and known fault labels.",
    },
    "forestry_remote_asset": {
        "title": "Forestry & Remote Asset Pack",
        "sensors": ["vibration_rms", "temperature_c", "humidity_pct", "battery_v", "gps_jitter_m", "acoustic_db"],
        "classes": ["normal", "tamper", "vehicle_nearby", "weather_stress", "battery_drop"],
        "base_rate": {"normal": 0.58, "tamper": 0.09, "vehicle_nearby": 0.15, "weather_stress": 0.11, "battery_drop": 0.07},
        "description": "Synthetic remote asset / outdoor sensor data for noisy low-power field conditions.",
    },
    "energy_anomaly": {
        "title": "Energy / Operations Anomaly Pack",
        "sensors": ["current_amp", "power_kw", "temperature_c", "duty_cycle_pct", "load_pct", "runtime_minutes"],
        "classes": ["normal", "overload", "idle_waste", "thermal_drift", "startup_spike"],
        "base_rate": {"normal": 0.62, "overload": 0.11, "idle_waste": 0.12, "thermal_drift": 0.08, "startup_spike": 0.07},
        "description": "Synthetic operations/energy signals for finding waste, overload and thermal drift patterns.",
    },
}


def _choice_counts(total_rows: int, class_rates: Dict[str, float]) -> Dict[str, int]:
    # Deterministic class counts with all classes represented.
    classes = list(class_rates.keys())
    total = max(1, int(total_rows))
    raw = {c: max(1, int(round(total * float(class_rates[c])))) for c in classes}
    delta = total - sum(raw.values())
    if delta != 0:
        # Adjust normal/first class to hit exact total.
        first = classes[0]
        raw[first] = max(1, raw[first] + delta)
    # If adjustment created too many rows through min-1 constraints, trim largest class.
    while sum(raw.values()) > total:
        largest = max(raw, key=raw.get)
        if raw[largest] > 1:
            raw[largest] -= 1
        else:
            break
    while sum(raw.values()) < total:
        raw[classes[0]] += 1
    return raw


def _base_for_class(pack_key: str, class_name: str, rng: np.random.Generator, n: int) -> pd.DataFrame:
    # Shared latent variables create plausible correlations.
    load = rng.normal(55, 12, n).clip(10, 100)
    ambient = rng.normal(22, 7, n).clip(-10, 45)
    rpm = rng.normal(1450, 90, n).clip(600, 2300)
    runtime = rng.uniform(5, 480, n)

    if pack_key == "rotating_machinery":
        vibration = rng.normal(0.42, 0.08, n) + (load - 55) * 0.003
        kurt = rng.normal(2.7, 0.35, n)
        temp = ambient + 35 + load * 0.20 + rng.normal(0, 2.0, n)
        current = 2.0 + load * 0.035 + rng.normal(0, 0.15, n)
        acoustic = 58 + vibration * 12 + rng.normal(0, 1.8, n)
        rpm_arr = rpm + rng.normal(0, 12, n)
        if class_name == "bearing_wear":
            vibration += rng.normal(0.42, 0.10, n); kurt += rng.normal(2.0, 0.45, n); acoustic += rng.normal(6, 2, n)
        elif class_name == "imbalance":
            vibration += rng.normal(0.32, 0.11, n); rpm_arr += rng.normal(-80, 30, n); acoustic += rng.normal(4, 1.8, n)
        elif class_name == "overheating":
            temp += rng.normal(18, 5, n); current += rng.normal(0.55, 0.18, n)
        elif class_name == "loose_mount":
            vibration += rng.normal(0.25, 0.12, n); kurt += rng.normal(0.9, 0.35, n); acoustic += rng.normal(8, 2.5, n)
        return pd.DataFrame({
            "vibration_rms": vibration.clip(0.05, 3.0),
            "vibration_kurtosis": kurt.clip(1.0, 15.0),
            "temperature_c": temp.clip(-10, 130),
            "current_amp": current.clip(0.1, 18),
            "rpm": rpm_arr.clip(200, 4000),
            "acoustic_db": acoustic.clip(25, 120),
        })

    if pack_key == "forestry_remote_asset":
        humidity = rng.normal(64, 18, n).clip(15, 100)
        battery = rng.normal(3.92, 0.12, n).clip(3.1, 4.25)
        gps_jitter = rng.normal(8, 3, n).clip(1, 60)
        vibration = rng.normal(0.18, 0.06, n)
        temp = ambient + rng.normal(0, 1.5, n)
        acoustic = rng.normal(43, 5, n)
        if class_name == "tamper":
            vibration += rng.normal(0.8, 0.22, n); acoustic += rng.normal(18, 5, n); gps_jitter += rng.normal(7, 4, n)
        elif class_name == "vehicle_nearby":
            vibration += rng.normal(0.35, 0.14, n); acoustic += rng.normal(12, 4, n)
        elif class_name == "weather_stress":
            humidity += rng.normal(25, 8, n); temp += rng.normal(-5, 6, n); gps_jitter += rng.normal(10, 5, n)
        elif class_name == "battery_drop":
            battery -= rng.normal(0.45, 0.12, n); temp += rng.normal(-3, 4, n)
        return pd.DataFrame({
            "vibration_rms": vibration.clip(0.01, 3.0),
            "temperature_c": temp.clip(-35, 70),
            "humidity_pct": humidity.clip(0, 100),
            "battery_v": battery.clip(2.7, 4.3),
            "gps_jitter_m": gps_jitter.clip(0.5, 120),
            "acoustic_db": acoustic.clip(15, 125),
        })

    # energy_anomaly
    duty = rng.normal(55, 22, n).clip(0, 100)
    load_pct = (load + rng.normal(0, 8, n)).clip(0, 120)
    current = 1.2 + load_pct * 0.055 + rng.normal(0, 0.25, n)
    power = current * rng.normal(0.70, 0.08, n)
    temp = ambient + 20 + load_pct * 0.24 + rng.normal(0, 2, n)
    runtime_arr = runtime
    if class_name == "overload":
        load_pct += rng.normal(28, 8, n); current += rng.normal(2.5, 0.7, n); power += rng.normal(2.0, 0.6, n)
    elif class_name == "idle_waste":
        duty -= rng.normal(25, 8, n); current += rng.normal(1.0, 0.3, n); power += rng.normal(0.8, 0.25, n)
    elif class_name == "thermal_drift":
        temp += rng.normal(16, 5, n); current += rng.normal(0.6, 0.2, n)
    elif class_name == "startup_spike":
        runtime_arr = rng.uniform(1, 18, n); current += rng.normal(3.4, 0.8, n); power += rng.normal(2.8, 0.7, n)
    return pd.DataFrame({
        "current_amp": current.clip(0.1, 30),
        "power_kw": power.clip(0.05, 30),
        "temperature_c": temp.clip(-10, 140),
        "duty_cycle_pct": duty.clip(0, 100),
        "load_pct": load_pct.clip(0, 150),
        "runtime_minutes": runtime_arr.clip(1, 1000),
    })


def generate_scenario_dataset(
    pack_key: str = "rotating_machinery",
    rows: int = 1500,
    seed: int = 109,
    noise_level: float = 0.06,
    missing_rate: float = 0.015,
    drift_strength: float = 0.05,
    imbalance_factor: float = 1.0,
    include_edge_cases: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create deterministic scenario-based synthetic data with known labels and stressors."""
    if pack_key not in SCENARIO_LIBRARY:
        pack_key = "rotating_machinery"
    spec = SCENARIO_LIBRARY[pack_key]
    rows = _safe_int(rows, 1000, 120, 50000)
    seed = _safe_int(seed, 109, 1, 2_000_000_000)
    noise_level = _safe_float(noise_level, 0.06, 0.0, 0.60)
    missing_rate = _safe_float(missing_rate, 0.015, 0.0, 0.50)
    drift_strength = _safe_float(drift_strength, 0.05, 0.0, 0.75)
    imbalance_factor = _safe_float(imbalance_factor, 1.0, 0.2, 3.0)
    rng = np.random.default_rng(seed)

    base_rates = dict(spec["base_rate"])
    if imbalance_factor != 1.0:
        # Increase/decrease anomaly rates while keeping normal as residual.
        non_normal = [c for c in base_rates if c != "normal"]
        for c in non_normal:
            base_rates[c] = min(0.45, max(0.01, base_rates[c] * imbalance_factor))
        base_rates["normal"] = max(0.05, 1.0 - sum(base_rates[c] for c in non_normal))
        total_rate = sum(base_rates.values())
        base_rates = {k: v / total_rate for k, v in base_rates.items()}

    counts = _choice_counts(rows, base_rates)
    parts: List[pd.DataFrame] = []
    start = pd.Timestamp("2026-01-01 00:00:00")
    idx = 0
    for class_name, n in counts.items():
        part = _base_for_class(pack_key, class_name, rng, n)
        part["label"] = class_name
        part["is_anomaly"] = class_name != "normal"
        part["scenario_id"] = f"{pack_key}_{class_name}"
        part["machine_id"] = [f"asset_{(idx + i) % 8 + 1:02d}" for i in range(n)]
        idx += n
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)
    # Shuffle but then assign monotonic timestamps.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["timestamp"] = [start + pd.Timedelta(minutes=5 * i) for i in range(len(df))]

    sensor_cols = [c for c in spec["sensors"] if c in df.columns]
    # Inject proportional sensor noise.
    for c in sensor_cols:
        scale = max(float(df[c].std(ddof=0)), 1e-6)
        df[c] = df[c] + rng.normal(0, scale * noise_level, len(df))

    # Add gradual drift to one/two key columns for realism.
    if drift_strength > 0 and sensor_cols:
        drift = np.linspace(0, drift_strength, len(df))
        for c in sensor_cols[: min(2, len(sensor_cols))]:
            scale = max(float(df[c].std(ddof=0)), 1e-6)
            df[c] = df[c] + drift * scale

    # Add edge cases/outliers with ground truth marker.
    df["synthetic_edge_case"] = False
    if include_edge_cases and len(df) >= 50 and sensor_cols:
        outlier_count = max(3, int(len(df) * 0.012))
        outlier_idx = rng.choice(df.index.to_numpy(), size=min(outlier_count, len(df)), replace=False)
        for c in sensor_cols[:3]:
            scale = max(float(df[c].std(ddof=0)), 1e-6)
            df.loc[outlier_idx, c] = df.loc[outlier_idx, c] + rng.normal(3.0 * scale, 0.6 * scale, len(outlier_idx))
        df.loc[outlier_idx, "synthetic_edge_case"] = True

    # Inject missingness only in sensor columns; keep labels intact.
    missing_cells = 0
    if missing_rate > 0 and sensor_cols:
        mask = rng.random((len(df), len(sensor_cols))) < missing_rate
        missing_cells = int(mask.sum())
        for j, c in enumerate(sensor_cols):
            df.loc[mask[:, j], c] = np.nan

    # Reorder columns.
    front = ["timestamp", "machine_id", "label", "is_anomaly", "scenario_id", "synthetic_edge_case"]
    df = df[front + [c for c in df.columns if c not in front]]

    manifest = {
        "version": VERSION,
        "pack_key": pack_key,
        "pack_title": spec["title"],
        "description": spec["description"],
        "rows": int(len(df)),
        "sensor_columns": sensor_cols,
        "labels": sorted(df["label"].unique().tolist()),
        "class_counts": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
        "noise_level": noise_level,
        "missing_rate_requested": missing_rate,
        "missing_cells": int(missing_cells),
        "drift_strength": drift_strength,
        "imbalance_factor": imbalance_factor,
        "include_edge_cases": bool(include_edge_cases),
        "seed": int(seed),
        "dataset_hash": _df_hash(df),
        "ground_truth_available": True,
        "boundary": "Synthetic scenario data validates workflow and stress handling; it does not prove production accuracy.",
    }
    return df, manifest


def score_synthetic_dataset(df: pd.DataFrame, manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score synthetic data usefulness for demos/regression/gates without claiming field accuracy."""
    manifest = manifest or {}
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "synthetic_quality_score": 0,
            "decision": "BLOCK - EMPTY SYNTHETIC DATASET",
            "issues": [{"severity": "blocker", "issue": "Dataset is empty"}],
            "strengths": [],
            "recommendations": ["Generate a scenario dataset before running demos/regression tests."],
        }

    issues: List[Dict[str, str]] = []
    strengths: List[str] = []
    recommendations: List[str] = []
    score = 100

    rows = len(df)
    cols = len(df.columns)
    label_col = "label" if "label" in df.columns else None
    sensor_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {"is_anomaly"}]
    missing_pct = float(df.isna().mean().mean()) if rows and cols else 1.0

    if rows < 500:
        score -= 18; issues.append({"severity": "warning", "issue": "Small dataset; demos may look weak below 500 rows."})
        recommendations.append("Use at least 1,000-2,000 rows for stronger demos and regression tests.")
    elif rows >= 1000:
        strengths.append("Enough rows for credible demo/regression behavior.")

    if not label_col:
        score -= 28; issues.append({"severity": "blocker", "issue": "No label column; ground truth cannot be tested."})
    else:
        classes = df[label_col].nunique(dropna=True)
        counts = df[label_col].value_counts(normalize=True)
        if classes < 3:
            score -= 20; issues.append({"severity": "warning", "issue": "Too few classes; scenario coverage is limited."})
        else:
            strengths.append(f"Ground-truth labels present with {classes} classes.")
        if not counts.empty and counts.iloc[0] > 0.85:
            score -= 15; issues.append({"severity": "warning", "issue": "Class imbalance is high; model/demo may over-focus on majority class."})
            recommendations.append("Increase anomaly class coverage or generate a stress-balanced dataset variant.")

    if len(sensor_cols) < 4:
        score -= 15; issues.append({"severity": "warning", "issue": "Few numeric sensor columns; weak multi-sensor realism."})
    else:
        strengths.append(f"Multi-sensor coverage present ({len(sensor_cols)} numeric sensor columns).")

    if missing_pct == 0:
        score -= 6; issues.append({"severity": "info", "issue": "No missing values; real customer data is usually messier."})
        recommendations.append("Keep a small controlled missing rate to exercise data-quality gates.")
    elif missing_pct > 0.12:
        score -= 16; issues.append({"severity": "warning", "issue": "Missingness is high; this is useful as stress data but not ideal as default demo data."})
    else:
        strengths.append("Controlled missingness present for data-quality gate testing.")

    if "synthetic_edge_case" in df.columns and bool(df["synthetic_edge_case"].any()):
        strengths.append("Edge cases/outliers included with marker column.")
    else:
        score -= 8; issues.append({"severity": "info", "issue": "No explicit edge-case markers included."})

    if not manifest.get("ground_truth_available", False):
        score -= 8; issues.append({"severity": "info", "issue": "Manifest does not confirm ground truth availability."})
    if manifest.get("boundary"):
        strengths.append("Synthetic-data boundary/disclaimer is present.")
    else:
        score -= 5; recommendations.append("Add a clear boundary: synthetic data does not prove production accuracy.")

    score = int(max(0, min(100, round(score))))
    if score >= 92:
        decision = "SYNTHETIC GOLDEN TESTDATA READY"
    elif score >= 82:
        decision = "SYNTHETIC PILOT DATA READY"
    elif score >= 70:
        decision = "SYNTHETIC DATA NEEDS TUNING"
    else:
        decision = "BLOCK - SYNTHETIC DATA TOO WEAK"

    return {
        "version": VERSION,
        "rows": int(rows),
        "cols": int(cols),
        "sensor_column_count": int(len(sensor_cols)),
        "missing_pct": round(missing_pct * 100, 3),
        "class_counts": df[label_col].value_counts().to_dict() if label_col else {},
        "synthetic_quality_score": score,
        "decision": decision,
        "strengths": strengths,
        "issues": issues,
        "recommendations": recommendations,
        "boundary": "This score measures synthetic demo/regression usefulness, not production accuracy.",
    }


def build_synthetic_data_optimizer_v109_snapshot(
    project_name: str = "EdgeTwin Project",
    pack_key: str = "rotating_machinery",
    rows: int = 1500,
    seed: int = 109,
    noise_level: float = 0.06,
    missing_rate: float = 0.015,
    drift_strength: float = 0.05,
    imbalance_factor: float = 1.0,
    include_edge_cases: bool = True,
) -> Dict[str, Any]:
    df, manifest = generate_scenario_dataset(
        pack_key=pack_key,
        rows=rows,
        seed=seed,
        noise_level=noise_level,
        missing_rate=missing_rate,
        drift_strength=drift_strength,
        imbalance_factor=imbalance_factor,
        include_edge_cases=include_edge_cases,
    )
    quality = score_synthetic_dataset(df, manifest)
    score = int(quality.get("synthetic_quality_score", 0))
    if score >= 92:
        decision = "GOLDEN SYNTHETIC DATA GO"
    elif score >= 82:
        decision = "SYNTHETIC PILOT DATA GO"
    elif score >= 70:
        decision = "TUNE BEFORE CUSTOMER DEMO"
    else:
        decision = "BLOCK SYNTHETIC DATA"

    stress_profiles = [
        {"name": "clean_demo", "noise_level": 0.03, "missing_rate": 0.0, "drift_strength": 0.02, "use": "customer demo"},
        {"name": "realistic_messy", "noise_level": 0.08, "missing_rate": 0.02, "drift_strength": 0.05, "use": "data-quality gate"},
        {"name": "field_stress", "noise_level": 0.14, "missing_rate": 0.06, "drift_strength": 0.12, "use": "regression / robustness"},
        {"name": "bad_input_blocker", "noise_level": 0.22, "missing_rate": 0.18, "drift_strength": 0.20, "use": "blocker test"},
    ]
    validation_contract = {
        "can_prove": [
            "workflow works end-to-end",
            "Data Quality Gate catches missingness/imbalance/drift",
            "Trust Ledger and reports remain reproducible",
            "known labels allow regression checks",
        ],
        "cannot_prove": [
            "real field accuracy",
            "production readiness",
            "legal/compliance certification",
            "customer-specific failure prediction without customer validation",
        ],
        "bridge_to_real_data": "Use V109 synthetic data to harden flows, then require customer sample data before Real-Data Evidence claims.",
    }
    return {
        "version": VERSION,
        "module": MODULE,
        "created_at": _utc_now(),
        "project_name": project_name,
        "pack_key": manifest.get("pack_key"),
        "pack_title": manifest.get("pack_title"),
        "scenario_library": SCENARIO_LIBRARY,
        "manifest": manifest,
        "quality": quality,
        "stress_profiles": stress_profiles,
        "validation_contract": validation_contract,
        "synthetic_quality_score": score,
        "decision": decision,
        "auto_use_allowed": score >= 82,
        "customer_demo_allowed": score >= 92,
        "field_accuracy_claim_allowed": False,
        "production_claim_allowed": False,
        "sample_preview": json.loads(df.head(12).to_json(orient="records", date_format="iso")),
        "dataset_hash": manifest.get("dataset_hash"),
        "important_boundary": "Synthetic data is optimized for demos, regression and data-quality gates. It is a bridge to customer data, not a replacement for field validation.",
    }


__all__ = [
    "SCENARIO_LIBRARY",
    "generate_scenario_dataset",
    "score_synthetic_dataset",
    "build_synthetic_data_optimizer_v109_snapshot",
]
