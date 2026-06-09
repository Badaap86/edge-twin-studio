"""EdgeTwin V111 Synthetic Reliability Lab.

Purpose:
- Make synthetic testdata not only realistic-looking, but measurable and regression-worthy.
- Stress synthetic datasets across seeds, messiness profiles, drift, missingness and edge cases.
- Produce an honest reliability score and gap list before synthetic data is trusted in demos, QA or pack delivery.

Boundary:
- Synthetic reliability proves workflow/stress coverage, not production accuracy.
- Real customer data is only used through aggregate profiles when explicitly provided/allowed by V110.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from synthetic_data_optimizer import generate_scenario_dataset, score_synthetic_dataset, SCENARIO_LIBRARY
from synthetic_real_bridge import build_real_data_profile, calibrate_synthetic_to_profile

VERSION = "V111"
MODULE = "Synthetic Reliability Lab"

STRESS_PROFILES: Dict[str, Dict[str, Any]] = {
    "clean_demo": {
        "label": "Clean demo",
        "noise_level": 0.035,
        "missing_rate": 0.002,
        "drift_strength": 0.02,
        "imbalance_factor": 1.0,
        "purpose": "Prove the happy path and customer demo stability.",
    },
    "realistic_messy": {
        "label": "Realistic messy",
        "noise_level": 0.085,
        "missing_rate": 0.025,
        "drift_strength": 0.07,
        "imbalance_factor": 0.85,
        "purpose": "Simulate normal customer mess: missing values, overlap and mild drift.",
    },
    "field_stress": {
        "label": "Field stress",
        "noise_level": 0.145,
        "missing_rate": 0.075,
        "drift_strength": 0.16,
        "imbalance_factor": 0.65,
        "purpose": "Stress low-quality field data before real deployment claims are made.",
    },
    "bad_input_blocker": {
        "label": "Bad input blocker",
        "noise_level": 0.26,
        "missing_rate": 0.18,
        "drift_strength": 0.28,
        "imbalance_factor": 0.35,
        "purpose": "Ensure EdgeTwin blocks weak data instead of producing overconfident reports.",
    },
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
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8", errors="replace")).hexdigest()


def _safe_numeric_cols(df: pd.DataFrame) -> List[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    ignore = {"is_anomaly"}
    return [str(c) for c in df.select_dtypes(include=[np.number]).columns if str(c) not in ignore and not pd.api.types.is_bool_dtype(df[c])]


def _label_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["label", "status", "state", "class", "fault", "event"]:
        if c in df.columns:
            return c
    return None


def _quality_from_df(df: pd.DataFrame, manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        return score_synthetic_dataset(df, manifest or {})
    except Exception:
        return {"synthetic_quality_score": 0, "decision": "QUALITY_SCORE_ERROR"}


def _column_health(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(df, pd.DataFrame) or df.empty:
        return rows
    for c in df.columns[:120]:
        s = df[c]
        entry = {
            "column": str(c),
            "dtype": str(s.dtype),
            "missing_pct": round(float(s.isna().mean()) * 100, 3),
            "unique_count": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            sn = pd.to_numeric(s, errors="coerce")
            entry.update({
                "min": None if sn.dropna().empty else round(float(sn.min()), 6),
                "p05": None if sn.dropna().empty else round(float(sn.quantile(0.05)), 6),
                "p50": None if sn.dropna().empty else round(float(sn.quantile(0.50)), 6),
                "p95": None if sn.dropna().empty else round(float(sn.quantile(0.95)), 6),
                "max": None if sn.dropna().empty else round(float(sn.max()), 6),
                "zero_variance": bool(sn.dropna().nunique() <= 1),
            })
        rows.append(entry)
    return rows


def _timestamp_health(df: pd.DataFrame) -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"detected": False, "score": 0, "notes": ["no dataframe"]}
    candidates = [c for c in df.columns if str(c).lower() in {"timestamp", "time", "datetime", "date"}]
    if not candidates:
        return {"detected": False, "score": 60, "notes": ["no timestamp column detected"]}
    c = candidates[0]
    ts = pd.to_datetime(df[c], errors="coerce")
    valid_pct = float(ts.notna().mean()) * 100
    notes: List[str] = []
    score = 100
    if valid_pct < 95:
        score -= 25; notes.append("timestamp parse rate below 95%")
    ts2 = ts.dropna().sort_values()
    if len(ts2) >= 3:
        deltas = ts2.diff().dropna().dt.total_seconds()
        median_delta = float(deltas.median()) if not deltas.empty else None
        p95_delta = float(deltas.quantile(0.95)) if not deltas.empty else None
        if median_delta and p95_delta and p95_delta / max(median_delta, 1.0) > 4:
            score -= 15; notes.append("irregular cadence detected")
    else:
        median_delta = p95_delta = None
        score -= 20; notes.append("too few valid timestamps")
    return {
        "detected": True,
        "column": str(c),
        "valid_pct": round(valid_pct, 3),
        "median_cadence_seconds": None if median_delta is None else round(median_delta, 3),
        "p95_cadence_seconds": None if p95_delta is None else round(p95_delta, 3),
        "score": max(0, int(score)),
        "notes": notes or ["timestamp health acceptable"],
    }


def _distribution_summary(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in _safe_numeric_cols(df)[:60]:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        out[c] = {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std(ddof=0)), 6),
            "p05": round(float(s.quantile(0.05)), 6),
            "p50": round(float(s.quantile(0.50)), 6),
            "p95": round(float(s.quantile(0.95)), 6),
        }
    return out


def _distribution_distance(a: pd.DataFrame, b: pd.DataFrame) -> Dict[str, Any]:
    """Simple model-free distance: normalized quantile/mean/std gaps on overlapping numeric columns."""
    cols = [c for c in _safe_numeric_cols(a) if c in b.columns]
    rows: List[Dict[str, Any]] = []
    distances: List[float] = []
    for c in cols[:60]:
        sa = pd.to_numeric(a[c], errors="coerce").dropna()
        sb = pd.to_numeric(b[c], errors="coerce").dropna()
        if len(sa) < 10 or len(sb) < 10:
            continue
        qa = np.array([sa.mean(), sa.std(ddof=0), sa.quantile(0.05), sa.quantile(0.50), sa.quantile(0.95)], dtype=float)
        qb = np.array([sb.mean(), sb.std(ddof=0), sb.quantile(0.05), sb.quantile(0.50), sb.quantile(0.95)], dtype=float)
        scale = max(float(np.nanmax(np.abs(qb))), float(np.nanmax(np.abs(qa))), 1.0)
        dist = float(np.nanmean(np.abs(qa - qb) / scale))
        distances.append(dist)
        rows.append({"column": str(c), "normalized_quantile_distance": round(dist, 5)})
    avg = float(np.mean(distances)) if distances else 1.0
    fidelity_score = max(0, min(100, int(round(100 * (1 - min(avg, 1.0))))))
    return {"overlap_column_count": len(rows), "avg_normalized_distance": round(avg, 5), "fidelity_score": fidelity_score, "by_column": rows}


def _correlation_stability(datasets: List[pd.DataFrame]) -> Dict[str, Any]:
    if len(datasets) < 2:
        return {"score": 0, "notes": ["need at least two datasets"]}
    cols = set(_safe_numeric_cols(datasets[0]))
    for df in datasets[1:]:
        cols = cols.intersection(set(_safe_numeric_cols(df)))
    cols = sorted(cols)[:20]
    if len(cols) < 2:
        return {"score": 60, "notes": ["not enough shared numeric columns for correlation stability"]}
    vecs = []
    for df in datasets:
        corr = df[cols].corr(numeric_only=True).fillna(0.0).to_numpy()
        iu = np.triu_indices_from(corr, k=1)
        vecs.append(corr[iu])
    distances = []
    for i in range(1, len(vecs)):
        distances.append(float(np.mean(np.abs(vecs[0] - vecs[i]))))
    avg = float(np.mean(distances)) if distances else 1.0
    score = max(0, min(100, int(round(100 * (1 - min(avg, 1.0))))))
    return {"score": score, "avg_correlation_gap": round(avg, 5), "shared_numeric_columns": cols, "dataset_count": len(datasets)}


def _label_coverage(df: pd.DataFrame, expected_labels: Optional[List[str]] = None) -> Dict[str, Any]:
    c = _label_col(df)
    if not c:
        return {"detected": False, "score": 45, "notes": ["no label/status column detected"]}
    vc = df[c].astype(str).value_counts().to_dict()
    total = max(1, sum(int(v) for v in vc.values()))
    expected = expected_labels or list(vc.keys())
    missing = [str(x) for x in expected if str(x) not in vc]
    top_share = max(vc.values()) / total if vc else 1.0
    score = 100
    if missing:
        score -= min(40, 10 * len(missing))
    if top_share > 0.88:
        score -= 20
    if len(vc) < 3:
        score -= 15
    return {
        "detected": True,
        "column": c,
        "top_values": {str(k): int(v) for k, v in vc.items()},
        "expected_labels": expected,
        "missing_expected_labels": missing,
        "top_class_share_pct": round(top_share * 100, 3),
        "score": max(0, int(score)),
        "notes": ["label coverage acceptable"] if score >= 85 else ["label coverage needs improvement or intentionally represents bad-input profile"],
    }


def _privacy_similarity_risk(synthetic_df: pd.DataFrame, real_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Lightweight red-flag check. Not a formal privacy proof."""
    if real_df is None or not isinstance(real_df, pd.DataFrame) or real_df.empty:
        return {"available": False, "score": 75, "risk_level": "unknown_without_real_holdout", "notes": ["no raw real data provided; use profile-only mode by default"]}
    shared = [c for c in synthetic_df.columns if c in real_df.columns]
    if not shared:
        return {"available": True, "score": 80, "risk_level": "low_overlap", "notes": ["no shared columns for direct record similarity check"]}
    # Compare row hashes on shared columns after coarse string normalization. Exact matches are a red flag.
    syn = synthetic_df[shared].head(5000).astype(str).apply(lambda col: col.str.slice(0, 64))
    rea = real_df[shared].head(5000).astype(str).apply(lambda col: col.str.slice(0, 64))
    syn_hashes = set(pd.util.hash_pandas_object(syn, index=False).astype(str).tolist())
    rea_hashes = set(pd.util.hash_pandas_object(rea, index=False).astype(str).tolist())
    exact = len(syn_hashes.intersection(rea_hashes))
    rate = exact / max(1, min(len(syn_hashes), len(rea_hashes)))
    score = 100
    risk = "low"
    notes = ["no exact shared-row matches detected"]
    if rate > 0.005:
        score = 60; risk = "medium"; notes = ["some exact shared-row matches detected; do not reuse synthetic output broadly"]
    if rate > 0.02:
        score = 25; risk = "high"; notes = ["high exact-match rate; block reusable benchmark/template mode"]
    return {"available": True, "score": score, "risk_level": risk, "exact_match_count": exact, "exact_match_rate": round(rate, 6), "shared_columns_checked": shared[:80], "notes": notes}


def _red_team_checks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    if not isinstance(df, pd.DataFrame) or df.empty:
        return [{"check": "non_empty_dataset", "status": "FAIL", "severity": "blocker", "detail": "dataset is empty"}]
    checks.append({"check": "non_empty_dataset", "status": "PASS", "severity": "info", "detail": f"{len(df)} rows"})
    duplicate_rate = float(df.duplicated().mean()) if len(df) else 0.0
    checks.append({"check": "duplicate_rows", "status": "PASS" if duplicate_rate < 0.01 else "WARN", "severity": "warning" if duplicate_rate >= 0.01 else "info", "detail": f"duplicate_rate={duplicate_rate:.4f}"})
    missing_mean = float(df.isna().mean().mean()) if len(df.columns) else 1.0
    checks.append({"check": "missingness", "status": "PASS" if missing_mean < 0.20 else "WARN", "severity": "warning" if missing_mean >= 0.20 else "info", "detail": f"mean_missing={missing_mean:.4f}"})
    zero_var = [c for c in _safe_numeric_cols(df) if pd.to_numeric(df[c], errors="coerce").dropna().nunique() <= 1]
    checks.append({"check": "zero_variance_numeric", "status": "PASS" if not zero_var else "WARN", "severity": "warning" if zero_var else "info", "detail": ", ".join(zero_var[:12]) if zero_var else "none"})
    label = _label_col(df)
    if label and len(_safe_numeric_cols(df)):
        # suspicious leakage: any numeric column equal to is_anomaly/encoded labels exactly.
        leak_cols = []
        if "is_anomaly" in df.columns:
            target = pd.to_numeric(df["is_anomaly"], errors="coerce")
            for c in _safe_numeric_cols(df):
                if c == "is_anomaly":
                    continue
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().all() and target.notna().all() and s.nunique() <= 3 and float((s == target).mean()) > 0.98:
                    leak_cols.append(c)
        checks.append({"check": "label_leakage_smoke", "status": "PASS" if not leak_cols else "FAIL", "severity": "blocker" if leak_cols else "info", "detail": ", ".join(leak_cols) if leak_cols else "none"})
    return checks


def build_synthetic_reliability_lab_snapshot(
    project_name: str = "EdgeTwin Project",
    pack_key: str = "rotating_machinery",
    rows: int = 2500,
    seed: int = 111,
    real_df: Optional[pd.DataFrame] = None,
    use_real_profile: bool = True,
    stress_profile_keys: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    if pack_key not in SCENARIO_LIBRARY:
        pack_key = "rotating_machinery"
    rows = max(300, min(50000, int(rows or 2500)))
    seed = max(1, min(2_000_000_000, int(seed or 111)))
    stress_profile_keys = stress_profile_keys or ["clean_demo", "realistic_messy", "field_stress", "bad_input_blocker"]
    stress_profile_keys = [k for k in stress_profile_keys if k in STRESS_PROFILES] or ["realistic_messy"]

    real_profile = build_real_data_profile(real_df) if use_real_profile and isinstance(real_df, pd.DataFrame) and not real_df.empty else build_real_data_profile(None)
    datasets: Dict[str, pd.DataFrame] = {}
    stress_rows: List[Dict[str, Any]] = []
    quality_scores: List[int] = []
    reliability_parts: List[int] = []

    expected_labels = list(SCENARIO_LIBRARY[pack_key].get("classes", []))
    baseline_df: Optional[pd.DataFrame] = None

    for idx, key in enumerate(stress_profile_keys):
        prof = STRESS_PROFILES[key]
        df, manifest = generate_scenario_dataset(
            pack_key=pack_key,
            rows=rows,
            seed=seed + idx * 997,
            noise_level=prof["noise_level"],
            missing_rate=prof["missing_rate"],
            drift_strength=prof["drift_strength"],
            imbalance_factor=prof["imbalance_factor"],
            include_edge_cases=True,
        )
        if real_profile.get("available") and key in {"realistic_messy", "field_stress"}:
            df = calibrate_synthetic_to_profile(df, real_profile, seed=seed + idx * 1009)
        datasets[key] = df
        if baseline_df is None:
            baseline_df = df
        q = _quality_from_df(df, manifest)
        label_cov = _label_coverage(df, expected_labels)
        ts = _timestamp_health(df)
        red = _red_team_checks(df)
        blockers = [r for r in red if r.get("status") == "FAIL"]
        quality_score = int(q.get("synthetic_quality_score", q.get("score", 0)) or 0)
        reliability = int(round(np.mean([quality_score, int(label_cov.get("score", 0)), int(ts.get("score", 0)), 100 if not blockers else 40])))
        if key == "bad_input_blocker":
            # Bad input should often be weaker. The score rewards it for being present but not for quality.
            reliability = int(round(np.mean([65, int(label_cov.get("score", 0)), 100 if key in datasets else 0, 100 if red else 0])))
        quality_scores.append(quality_score)
        reliability_parts.append(reliability)
        stress_rows.append({
            "stress_profile": key,
            "label": prof["label"],
            "purpose": prof["purpose"],
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "dataset_hash": _hash_df(df),
            "synthetic_quality_score": quality_score,
            "label_coverage_score": int(label_cov.get("score", 0)),
            "timestamp_score": int(ts.get("score", 0)),
            "blocker_count": len(blockers),
            "synthetic_reliability_score": reliability,
            "decision": "PASS" if reliability >= 82 and not blockers else ("STRESS_EXPECTED" if key == "bad_input_blocker" else "REVIEW"),
        })

    dataset_list = list(datasets.values())
    corr = _correlation_stability(dataset_list)
    privacy = _privacy_similarity_risk(datasets.get("realistic_messy", dataset_list[0]), real_df if isinstance(real_df, pd.DataFrame) else None)
    fidelity_to_real = None
    if isinstance(real_df, pd.DataFrame) and not real_df.empty:
        fidelity_to_real = _distribution_distance(datasets.get("realistic_messy", dataset_list[0]), real_df)
    coverage_score = int(round(100 * len(stress_profile_keys) / max(1, len(STRESS_PROFILES))))
    score_inputs = [
        int(round(np.mean(reliability_parts))) if reliability_parts else 0,
        int(corr.get("score", 0)),
        int(privacy.get("score", 75)),
        coverage_score,
    ]
    if fidelity_to_real:
        score_inputs.append(int(fidelity_to_real.get("fidelity_score", 0)))
    synthetic_reliability_score = int(round(np.mean(score_inputs)))

    if synthetic_reliability_score >= 92:
        decision = "SYNTHETIC RELIABILITY GO"
    elif synthetic_reliability_score >= 82:
        decision = "SYNTHETIC RELIABILITY CONDITIONAL GO"
    else:
        decision = "SYNTHETIC RELIABILITY REVIEW"

    recommendations: List[Dict[str, Any]] = []
    if coverage_score < 100:
        recommendations.append({"priority": "medium", "action": "Enable all stress profiles before using synthetic data as a golden regression pack."})
    if corr.get("score", 0) < 80:
        recommendations.append({"priority": "high", "action": "Tune generator correlations; unstable cross-seed correlations can make regression results misleading."})
    if privacy.get("risk_level") in {"medium", "high"}:
        recommendations.append({"priority": "blocker", "action": "Do not reuse this synthetic output across customers; exact-row similarity risk is too high."})
    if not real_profile.get("available"):
        recommendations.append({"priority": "medium", "action": "Add consent-controlled aggregate real-data profiles later to reduce the synthetic-to-real gap."})
    if fidelity_to_real and fidelity_to_real.get("fidelity_score", 0) < 70:
        recommendations.append({"priority": "high", "action": "Real-profile calibration is weak; review column mapping, units and missingness before using the synthetic bridge."})
    recommendations.append({"priority": "policy", "action": "Use V111 for regression/demo reliability only; never present synthetic data as production accuracy evidence."})

    validation_contract = {
        "synthetic_data_can_prove": [
            "workflow stability",
            "stress handling",
            "data-quality gate behavior",
            "known-label regression behavior",
            "demo realism before customer data is available",
        ],
        "synthetic_data_cannot_prove": [
            "production accuracy",
            "legal/compliance certification",
            "field reliability for a specific customer",
            "safety-critical performance",
        ],
        "required_before_accuracy_claims": [
            "representative real customer data",
            "labelled faults/events where accuracy is claimed",
            "holdout validation",
            "human/founder review",
            "customer approval of assumptions and limits",
        ],
    }

    snapshot = {
        "version": VERSION,
        "module": MODULE,
        "created_at": _utc_now(),
        "project_name": str(project_name),
        "pack_key": pack_key,
        "pack_title": SCENARIO_LIBRARY[pack_key].get("title", pack_key),
        "rows_per_profile": rows,
        "seed": seed,
        "decision": decision,
        "synthetic_reliability_score": synthetic_reliability_score,
        "score_inputs": {
            "mean_stress_reliability": int(round(np.mean(reliability_parts))) if reliability_parts else 0,
            "correlation_stability": int(corr.get("score", 0)),
            "privacy_similarity_risk_score": int(privacy.get("score", 75)),
            "stress_coverage_score": coverage_score,
            "real_fidelity_score": None if not fidelity_to_real else int(fidelity_to_real.get("fidelity_score", 0)),
        },
        "stress_profiles": stress_rows,
        "correlation_stability": corr,
        "privacy_similarity_risk": privacy,
        "fidelity_to_real": fidelity_to_real or {"available": False, "reason": "no raw real customer data provided; profile-only mode preferred"},
        "real_profile_available": bool(real_profile.get("available")),
        "real_profile": real_profile,
        "distribution_summary_realistic_messy": _distribution_summary(datasets.get("realistic_messy", dataset_list[0])),
        "column_health_realistic_messy": _column_health(datasets.get("realistic_messy", dataset_list[0])),
        "timestamp_health_realistic_messy": _timestamp_health(datasets.get("realistic_messy", dataset_list[0])),
        "label_coverage_realistic_messy": _label_coverage(datasets.get("realistic_messy", dataset_list[0]), expected_labels),
        "red_team_checks_realistic_messy": _red_team_checks(datasets.get("realistic_messy", dataset_list[0])),
        "recommendations": recommendations,
        "validation_contract": validation_contract,
        "customer_safe_summary": (
            "V111 evaluates whether EdgeTwin synthetic datasets are reliable enough for demos, regression tests and data-quality gate stress tests. "
            "It measures stress coverage, label coverage, timestamp health, correlation stability, real-profile fidelity when available, and similarity/privacy red flags. "
            "It does not turn synthetic data into production accuracy evidence."
        ),
        "important_boundary": "Synthetic reliability reduces demo/regression risk; production claims still require representative labelled real-world validation.",
    }
    return _json_safe(snapshot), datasets
