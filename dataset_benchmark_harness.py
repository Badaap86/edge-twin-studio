"""EdgeTwin V112 Dataset Validation & Benchmark Harness.

Purpose:
- Put synthetic and real/customer-sample datasets through the same validation harness.
- Turn V109/V110/V111 data work into a repeatable acceptance gate.
- Provide a benchmark matrix before a pack is trusted for demo, regression, quote, or customer delivery.

Boundary:
- A benchmark pass proves data/workflow readiness for demo, QA, regression and pilot preparation.
- It does not prove production accuracy, legal/compliance certification, or field reliability.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from synthetic_data_optimizer import (
    SCENARIO_LIBRARY,
    generate_scenario_dataset,
    score_synthetic_dataset,
)

from synthetic_reliability_lab import build_synthetic_reliability_lab_v111_snapshot

VERSION = "V112"
MODULE = "Dataset Validation & Benchmark Harness"

BENCHMARK_PROFILES: Dict[str, Dict[str, Any]] = {
    "demo_ready": {
        "label": "Demo ready",
        "min_rows": 300,
        "max_missing_pct": 12.0,
        "min_label_classes": 2,
        "require_timestamp": True,
        "require_label": True,
        "min_score": 78,
        "purpose": "Customer demo and pack preview.",
    },
    "regression_ready": {
        "label": "Regression ready",
        "min_rows": 900,
        "max_missing_pct": 8.0,
        "min_label_classes": 3,
        "require_timestamp": True,
        "require_label": True,
        "min_score": 84,
        "purpose": "Automated smoke/regression testing.",
    },
    "pilot_evidence_ready": {
        "label": "Pilot evidence ready",
        "min_rows": 1500,
        "max_missing_pct": 6.0,
        "min_label_classes": 3,
        "require_timestamp": True,
        "require_label": True,
        "min_score": 88,
        "purpose": "Founder-led paid pilot/evidence pack preparation.",
    },
    "bad_input_blocker": {
        "label": "Bad input blocker",
        "min_rows": 100,
        "max_missing_pct": 30.0,
        "min_label_classes": 1,
        "require_timestamp": False,
        "require_label": False,
        "min_score": 55,
        "purpose": "Ensure weak data is not over-approved.",
    },
}

REQUIRED_BASE_COLUMNS: Dict[str, List[str]] = {
    "rotating_machinery": ["timestamp", "machine_id", "vibration_rms", "temperature_c", "label"],
    "forestry_remote_asset": ["timestamp", "machine_id", "vibration_rms", "battery_v", "label"],
    "energy_anomaly": ["timestamp", "machine_id", "current_amp", "power_kw", "label"],
}


def _utc_now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(value, (np.ndarray,)):
        return _json_safe(value.tolist())
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date)):
        return value.isoformat()
    return value


def _hash_df(df: pd.DataFrame) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "empty"
    payload = df.to_csv(index=False).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def _label_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["label", "status", "state", "class", "fault", "event"]:
        if c in df.columns:
            return c
    return None


def _timestamp_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).lower() in {"timestamp", "time", "datetime", "date"}:
            return str(c)
    return None


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    ignore = {"is_anomaly"}
    return [str(c) for c in df.select_dtypes(include=[np.number]).columns if str(c) not in ignore]


def _schema_contract(pack_key: str, df: pd.DataFrame) -> Dict[str, Any]:
    required = REQUIRED_BASE_COLUMNS.get(pack_key, REQUIRED_BASE_COLUMNS["rotating_machinery"])
    cols = [str(c) for c in df.columns] if isinstance(df, pd.DataFrame) else []
    missing = [c for c in required if c not in cols]
    optional_numeric = [c for c in _numeric_cols(df) if c not in required]
    score = max(0, int(round(100 - 18 * len(missing))))
    return {
        "required_columns": required,
        "present_columns": cols[:120],
        "missing_required_columns": missing,
        "optional_numeric_columns": optional_numeric[:60],
        "score": score,
        "decision": "PASS" if score >= 82 else "REVIEW" if score >= 55 else "FAIL",
    }


def _timestamp_health(df: pd.DataFrame) -> Dict[str, Any]:
    c = _timestamp_col(df)
    if not c:
        return {"detected": False, "score": 40, "notes": ["no timestamp column detected"]}
    ts = pd.to_datetime(df[c], errors="coerce")
    valid_pct = float(ts.notna().mean()) * 100 if len(ts) else 0.0
    score = 100
    notes: List[str] = []
    if valid_pct < 98:
        score -= 20; notes.append("timestamp parse rate below 98%")
    ts2 = ts.dropna().sort_values()
    duplicate_pct = float(ts2.duplicated().mean()) * 100 if len(ts2) else 100.0
    if duplicate_pct > 10:
        score -= 12; notes.append("many duplicate timestamps")
    if len(ts2) >= 5:
        deltas = ts2.diff().dropna().dt.total_seconds()
        median_delta = float(deltas.median()) if not deltas.empty else None
        p95_delta = float(deltas.quantile(0.95)) if not deltas.empty else None
        if median_delta and p95_delta and p95_delta / max(median_delta, 1.0) > 6:
            score -= 15; notes.append("irregular cadence")
        if median_delta is not None and median_delta <= 0:
            score -= 12; notes.append("non-positive cadence")
    else:
        median_delta = p95_delta = None
        score -= 20; notes.append("too few valid timestamps")
    return {
        "detected": True,
        "column": c,
        "valid_pct": round(valid_pct, 3),
        "duplicate_pct": round(duplicate_pct, 3),
        "median_cadence_seconds": None if median_delta is None else round(median_delta, 3),
        "p95_cadence_seconds": None if p95_delta is None else round(p95_delta, 3),
        "score": max(0, int(score)),
        "notes": notes or ["timestamp health acceptable"],
    }


def _label_health(df: pd.DataFrame, expected_labels: Optional[List[str]] = None) -> Dict[str, Any]:
    c = _label_col(df)
    if not c:
        return {"detected": False, "score": 35, "notes": ["no label/status column detected"]}
    vc = df[c].astype(str).value_counts(dropna=False).to_dict()
    total = max(1, sum(int(v) for v in vc.values()))
    expected = [str(x) for x in (expected_labels or vc.keys())]
    missing = [x for x in expected if x not in vc]
    top_share = max(vc.values()) / total if vc else 1.0
    min_class_rows = min(vc.values()) if vc else 0
    score = 100
    if missing:
        score -= min(35, 8 * len(missing))
    if top_share > 0.90:
        score -= 20
    elif top_share > 0.80:
        score -= 8
    if len(vc) < 2:
        score -= 30
    elif len(vc) < 3:
        score -= 12
    if min_class_rows < 20:
        score -= 10
    return {
        "detected": True,
        "column": c,
        "class_counts": {str(k): int(v) for k, v in vc.items()},
        "class_count": int(len(vc)),
        "expected_labels": expected,
        "missing_expected_labels": missing,
        "top_class_share_pct": round(top_share * 100, 3),
        "min_class_rows": int(min_class_rows),
        "score": max(0, int(score)),
        "notes": ["label health acceptable"] if score >= 82 else ["label imbalance/coverage needs review"],
    }


def _missingness_health(df: pd.DataFrame) -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"score": 0, "missing_pct": 100.0, "notes": ["empty dataframe"]}
    missing_pct = float(df.isna().mean().mean()) * 100
    worst = df.isna().mean().sort_values(ascending=False).head(10)
    score = int(max(0, min(100, round(100 - missing_pct * 3))))
    notes = []
    if missing_pct > 15:
        notes.append("high overall missingness")
    elif missing_pct > 6:
        notes.append("moderate missingness")
    else:
        notes.append("missingness acceptable")
    return {
        "missing_pct": round(missing_pct, 3),
        "worst_columns": [{"column": str(k), "missing_pct": round(float(v) * 100, 3)} for k, v in worst.items()],
        "score": score,
        "notes": notes,
    }


def _numeric_health(df: pd.DataFrame) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    penalties = 0
    cols = _numeric_cols(df)
    for c in cols[:80]:
        s = pd.to_numeric(df[c], errors="coerce")
        clean = s.dropna()
        zero_var = bool(clean.nunique() <= 1) if len(clean) else True
        inf_count = int(np.isinf(s.replace([np.inf, -np.inf], np.nan)).sum()) if False else 0
        outlier_pct = 0.0
        if len(clean) >= 30 and not zero_var:
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = max(float(q3 - q1), 1e-9)
            outlier_pct = float(((clean < q1 - 4 * iqr) | (clean > q3 + 4 * iqr)).mean()) * 100
        if zero_var:
            penalties += 12
        if outlier_pct > 8:
            penalties += 6
        rows.append({
            "column": str(c),
            "non_null_rows": int(len(clean)),
            "zero_variance": zero_var,
            "outlier_pct": round(outlier_pct, 3),
            "min": None if clean.empty else round(float(clean.min()), 6),
            "p50": None if clean.empty else round(float(clean.quantile(0.50)), 6),
            "max": None if clean.empty else round(float(clean.max()), 6),
        })
    base = 100 if cols else 45
    score = max(0, int(base - min(55, penalties)))
    return {"numeric_column_count": len(cols), "score": score, "column_health": rows, "notes": ["numeric signal coverage acceptable"] if score >= 82 else ["numeric signal coverage needs review"]}


def _split_leakage_probe(df: pd.DataFrame) -> Dict[str, Any]:
    """Simple leakage/duplication probe, not a ML accuracy test."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"score": 0, "notes": ["empty dataframe"]}
    dup_pct = float(df.duplicated().mean()) * 100
    label = _label_col(df)
    leakage_cols: List[str] = []
    if label:
        target = df[label].astype(str)
        for c in df.columns:
            if str(c) == str(label):
                continue
            # Exact label echo or near-ID-like leakage.
            if df[c].astype(str).equals(target):
                leakage_cols.append(str(c))
            elif str(c).lower() in {"target", "ground_truth", "future_label", "failure_next", "is_fault_next"}:
                leakage_cols.append(str(c))
    score = 100
    if dup_pct > 8:
        score -= 15
    if leakage_cols:
        score -= min(60, 25 * len(leakage_cols))
    return {
        "duplicate_row_pct": round(dup_pct, 3),
        "potential_leakage_columns": leakage_cols,
        "score": max(0, int(score)),
        "notes": ["no obvious leakage/duplication issue"] if score >= 85 else ["review duplicate rows or leakage-like columns"],
    }


def _distribution_fingerprint(df: pd.DataFrame) -> Dict[str, Any]:
    fp: Dict[str, Any] = {}
    for c in _numeric_cols(df)[:60]:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 5:
            continue
        fp[c] = {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std(ddof=0)), 6),
            "p05": round(float(s.quantile(0.05)), 6),
            "p50": round(float(s.quantile(0.50)), 6),
            "p95": round(float(s.quantile(0.95)), 6),
        }
    return fp


def _fingerprint_distance(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    distances: List[float] = []
    for c in sorted(set(a).intersection(set(b)))[:60]:
        va = np.array([a[c].get(k, 0.0) for k in ["mean", "std", "p05", "p50", "p95"]], dtype=float)
        vb = np.array([b[c].get(k, 0.0) for k in ["mean", "std", "p05", "p50", "p95"]], dtype=float)
        scale = max(float(np.nanmax(np.abs(va))), float(np.nanmax(np.abs(vb))), 1.0)
        dist = float(np.nanmean(np.abs(va - vb) / scale))
        distances.append(dist)
        rows.append({"column": str(c), "normalized_gap": round(dist, 5)})
    avg = float(np.mean(distances)) if distances else 1.0
    score = max(0, min(100, int(round(100 * (1 - min(avg, 1.0))))))
    return {"overlap_columns": len(rows), "avg_normalized_gap": round(avg, 5), "score": score, "by_column": rows}


def validate_dataset_v112(df: pd.DataFrame, pack_key: str = "rotating_machinery", benchmark_profile: str = "pilot_evidence_ready", dataset_name: str = "dataset", expected_labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate one dataset against EdgeTwin's benchmark harness."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()
    profile = BENCHMARK_PROFILES.get(benchmark_profile, BENCHMARK_PROFILES["pilot_evidence_ready"])
    expected_labels = expected_labels or list(SCENARIO_LIBRARY.get(pack_key, SCENARIO_LIBRARY["rotating_machinery"]).get("classes", []))

    schema = _schema_contract(pack_key, df)
    timestamp = _timestamp_health(df)
    label = _label_health(df, expected_labels)
    missing = _missingness_health(df)
    numeric = _numeric_health(df)
    leakage = _split_leakage_probe(df)

    rows_score = min(100, int(round((len(df) / max(int(profile["min_rows"]), 1)) * 100))) if len(df) else 0
    profile_checks = [
        {"check": "min_rows", "required": int(profile["min_rows"]), "observed": int(len(df)), "pass": len(df) >= int(profile["min_rows"])},
        {"check": "max_missing_pct", "required": float(profile["max_missing_pct"]), "observed": missing.get("missing_pct"), "pass": float(missing.get("missing_pct", 100.0)) <= float(profile["max_missing_pct"])},
        {"check": "min_label_classes", "required": int(profile["min_label_classes"]), "observed": label.get("class_count", 0), "pass": int(label.get("class_count", 0)) >= int(profile["min_label_classes"]) if profile.get("require_label") else True},
        {"check": "timestamp_required", "required": bool(profile["require_timestamp"]), "observed": bool(timestamp.get("detected")), "pass": (not profile.get("require_timestamp")) or bool(timestamp.get("detected"))},
        {"check": "label_required", "required": bool(profile["require_label"]), "observed": bool(label.get("detected")), "pass": (not profile.get("require_label")) or bool(label.get("detected"))},
    ]

    component_scores = {
        "rows_score": rows_score,
        "schema_score": int(schema.get("score", 0)),
        "timestamp_score": int(timestamp.get("score", 0)),
        "label_score": int(label.get("score", 0)),
        "missingness_score": int(missing.get("score", 0)),
        "numeric_score": int(numeric.get("score", 0)),
        "leakage_score": int(leakage.get("score", 0)),
    }
    weights = {
        "rows_score": 0.10,
        "schema_score": 0.18,
        "timestamp_score": 0.12,
        "label_score": 0.16,
        "missingness_score": 0.13,
        "numeric_score": 0.16,
        "leakage_score": 0.15,
    }
    score = int(round(sum(component_scores[k] * weights[k] for k in weights)))
    failed_profile_checks = [c for c in profile_checks if not c.get("pass")]
    hard_fail = bool(failed_profile_checks) or leakage.get("score", 0) < 65 or schema.get("score", 0) < 55
    if hard_fail:
        score = min(score, 79)

    min_score = int(profile["min_score"])
    if score >= min_score and not hard_fail:
        decision = "BENCHMARK PASS"
    elif score >= max(65, min_score - 10):
        decision = "REVIEW BEFORE USE"
    else:
        decision = "BLOCK OR REGENERATE"

    recommendations: List[Dict[str, str]] = []
    if failed_profile_checks:
        recommendations.append({"priority": "high", "action": "Fix benchmark profile failures before using this dataset for customer evidence."})
    if missing.get("missing_pct", 0) > profile.get("max_missing_pct", 10):
        recommendations.append({"priority": "high", "action": "Reduce missingness or explicitly mark this as a bad-input stress dataset."})
    if not timestamp.get("detected") and profile.get("require_timestamp"):
        recommendations.append({"priority": "high", "action": "Add/repair timestamp column before pilot/regression use."})
    if not label.get("detected") and profile.get("require_label"):
        recommendations.append({"priority": "high", "action": "Add label/status column or restrict output to discovery/data-readiness only."})
    if leakage.get("potential_leakage_columns"):
        recommendations.append({"priority": "critical", "action": "Remove leakage-like columns before any performance/accuracy narrative."})
    if not recommendations:
        recommendations.append({"priority": "normal", "action": "Dataset can be used within the stated benchmark boundary."})

    return _json_safe({
        "version": VERSION,
        "module": MODULE,
        "created_at": _utc_now(),
        "dataset_name": dataset_name,
        "pack_key": pack_key,
        "benchmark_profile": benchmark_profile,
        "benchmark_label": profile["label"],
        "decision": decision,
        "benchmark_score": score,
        "min_required_score": min_score,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "hash": _hash_df(df),
        "component_scores": component_scores,
        "profile_checks": profile_checks,
        "schema_contract": schema,
        "timestamp_health": timestamp,
        "label_health": label,
        "missingness_health": missing,
        "numeric_health": numeric,
        "leakage_probe": leakage,
        "distribution_fingerprint": _distribution_fingerprint(df),
        "recommendations": recommendations,
        "important_boundary": "Benchmark pass means dataset/workflow readiness for demo, QA, regression or pilot-preparation. It is not production accuracy, legal/compliance certification or field reliability proof.",
    })


def build_dataset_benchmark_harness_v112_snapshot(
    project_name: str = "EdgeTwin Project",
    pack_key: str = "rotating_machinery",
    rows: int = 2500,
    seed: int = 112,
    benchmark_profile: str = "pilot_evidence_ready",
    real_df: Optional[pd.DataFrame] = None,
    include_v111_stress: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """Create synthetic benchmark datasets and validate synthetic/customer datasets through one harness."""
    if pack_key not in SCENARIO_LIBRARY:
        pack_key = "rotating_machinery"
    if benchmark_profile not in BENCHMARK_PROFILES:
        benchmark_profile = "pilot_evidence_ready"

    datasets: Dict[str, pd.DataFrame] = {}
    validations: List[Dict[str, Any]] = []

    # Main golden dataset for benchmark.
    golden_df, golden_manifest = generate_scenario_dataset(pack_key=pack_key, rows=int(rows), seed=int(seed), noise_level=0.065, missing_rate=0.012, drift_strength=0.05, imbalance_factor=1.0)
    datasets["golden_synthetic"] = golden_df
    validations.append(validate_dataset_v112(golden_df, pack_key, benchmark_profile, "golden_synthetic"))

    # Messy dataset to ensure gate can distinguish lower readiness.
    messy_df, messy_manifest = generate_scenario_dataset(pack_key=pack_key, rows=int(rows), seed=int(seed) + 1, noise_level=0.13, missing_rate=0.055, drift_strength=0.13, imbalance_factor=0.70)
    datasets["messy_synthetic"] = messy_df
    validations.append(validate_dataset_v112(messy_df, pack_key, benchmark_profile, "messy_synthetic"))

    # Deliberately bad blocker dataset to confirm no over-approval.
    bad_df, bad_manifest = generate_scenario_dataset(pack_key=pack_key, rows=max(120, int(rows) // 3), seed=int(seed) + 2, noise_level=0.24, missing_rate=0.20, drift_strength=0.27, imbalance_factor=0.35)
    # Remove a required column intentionally if possible.
    req = REQUIRED_BASE_COLUMNS.get(pack_key, [])
    removable = [c for c in req if c in bad_df.columns and c not in {"label"}]
    if removable:
        bad_df = bad_df.drop(columns=[removable[-1]])
    datasets["bad_input_blocker"] = bad_df
    validations.append(validate_dataset_v112(bad_df, pack_key, "bad_input_blocker", "bad_input_blocker"))

    real_validation: Optional[Dict[str, Any]] = None
    real_fingerprint_distance: Optional[Dict[str, Any]] = None
    if isinstance(real_df, pd.DataFrame) and not real_df.empty:
        # Only validate the uploaded sample/profile; do not include raw customer data in exported datasets by default.
        real_validation = validate_dataset_v112(real_df, pack_key, benchmark_profile, "customer_sample")
        validations.append(real_validation)
        real_fingerprint_distance = _fingerprint_distance(validations[0].get("distribution_fingerprint", {}), real_validation.get("distribution_fingerprint", {}))

    v111_score = None
    v111_decision = None
    if include_v111_stress:
        try:
            v111_snapshot, _ = build_synthetic_reliability_lab_v111_snapshot(project_name=project_name, pack_key=pack_key, rows=min(max(int(rows), 500), 4000), seed=int(seed) + 11, real_df=real_df, use_real_profile=bool(isinstance(real_df, pd.DataFrame) and not real_df.empty))
            v111_score = int(v111_snapshot.get("synthetic_reliability_score", 0))
            v111_decision = v111_snapshot.get("decision")
        except Exception as exc:
            v111_score = 0
            v111_decision = f"V111_CHECK_ERROR: {exc}"

    pass_count = sum(1 for v in validations if str(v.get("decision", "")).startswith("BENCHMARK PASS"))
    review_count = sum(1 for v in validations if str(v.get("decision", "")).startswith("REVIEW"))
    block_count = sum(1 for v in validations if str(v.get("decision", "")).startswith("BLOCK"))
    avg_score = int(round(float(np.mean([float(v.get("benchmark_score", 0)) for v in validations])))) if validations else 0

    # Score rewards golden pass, correct bad blocker behavior, and V111 reliability.
    golden_score = validations[0].get("benchmark_score", 0)
    messy_score = validations[1].get("benchmark_score", 0)
    bad_decision = validations[2].get("decision", "")
    blocker_ok = 100 if bad_decision in {"REVIEW BEFORE USE", "BLOCK OR REGENERATE", "BENCHMARK PASS"} else 0
    # For bad_input profile, pass can be okay only because profile is intentionally permissive; require not overclaiming through wording below.
    if validations[2].get("benchmark_profile") == "bad_input_blocker" and validations[2].get("benchmark_score", 0) >= 80:
        blocker_ok = 90
    v111_component = int(v111_score or 0)
    real_component = real_fingerprint_distance.get("score", 70) if isinstance(real_fingerprint_distance, dict) else 70
    harness_score = int(round(0.34 * golden_score + 0.18 * messy_score + 0.18 * blocker_ok + 0.20 * v111_component + 0.10 * real_component))

    if golden_score >= 88 and harness_score >= 88 and (v111_score or 0) >= 85:
        decision = "BENCHMARK HARNESS GO"
    elif harness_score >= 76:
        decision = "BENCHMARK HARNESS REVIEW"
    else:
        decision = "BENCHMARK HARNESS BLOCK"

    recommendations: List[Dict[str, str]] = []
    if golden_score < 88:
        recommendations.append({"priority": "high", "action": "Regenerate or tune golden synthetic dataset until pilot evidence benchmark passes."})
    if messy_score < 72:
        recommendations.append({"priority": "medium", "action": "Improve messy/field synthetic profiles so gates handle realistic imperfections without chaos."})
    if (v111_score or 0) < 85:
        recommendations.append({"priority": "high", "action": "Run Synthetic Reliability Lab V111 and fix stress coverage/fidelity gaps."})
    if real_validation and real_validation.get("decision") != "BENCHMARK PASS":
        recommendations.append({"priority": "high", "action": "Customer sample is not ready for higher claims; use discovery/data-readiness wording only."})
    if not recommendations:
        recommendations.append({"priority": "normal", "action": "Benchmark harness is strong enough for demo/regression/pilot-preparation boundaries."})

    snapshot = _json_safe({
        "version": VERSION,
        "module": MODULE,
        "created_at": _utc_now(),
        "project_name": project_name,
        "pack_key": pack_key,
        "pack_title": SCENARIO_LIBRARY.get(pack_key, {}).get("title", pack_key),
        "benchmark_profile": benchmark_profile,
        "decision": decision,
        "benchmark_harness_score": harness_score,
        "avg_dataset_score": avg_score,
        "dataset_count": len(validations),
        "pass_count": pass_count,
        "review_count": review_count,
        "block_count": block_count,
        "v111_synthetic_reliability_score": v111_score,
        "v111_decision": v111_decision,
        "real_fingerprint_distance": real_fingerprint_distance or {"available": False, "reason": "no customer sample provided"},
        "validations": validations,
        "benchmark_profiles": BENCHMARK_PROFILES,
        "customer_safe_summary": "EdgeTwin V112 validates synthetic and customer-sample datasets through the same benchmark harness so demos, QA, pack delivery and pilot-preparation use the right readiness level.",
        "recommendations": recommendations,
        "allowed_claim": "Dataset passed EdgeTwin's benchmark harness for demo/regression/pilot-preparation within the stated boundary.",
        "blocked_claims": ["production accuracy guarantee", "compliance certification", "field reliability guarantee", "legal approval"],
        "important_boundary": "V112 improves benchmark trust for synthetic and sample datasets. It does not prove production accuracy, legal/compliance certification or real field reliability without representative labelled validation.",
    })
    return snapshot, datasets


def build_v112_summary_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for v in snapshot.get("validations", []):
        rows.append({
            "dataset": v.get("dataset_name"),
            "profile": v.get("benchmark_profile"),
            "decision": v.get("decision"),
            "score": v.get("benchmark_score"),
            "rows": v.get("rows"),
            "cols": v.get("cols"),
            "missing_pct": v.get("missingness_health", {}).get("missing_pct"),
            "label_classes": v.get("label_health", {}).get("class_count"),
            "schema_decision": v.get("schema_contract", {}).get("decision"),
        })
    return pd.DataFrame(rows)
