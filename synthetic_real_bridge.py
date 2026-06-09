
"""EdgeTwin V110 Synthetic→Real Bridge.

Purpose:
- Bring synthetic data closer to real customer data using aggregate profiles.
- Keep real customer data consent-controlled and minimized.
- Improve demos, regression and Data Quality Gates without claiming production accuracy.

Boundary:
- Default mode stores aggregate profiles only, not raw customer rows.
- Customer data can improve EdgeTwin only with explicit permission/purpose/retention rules.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from synthetic_data_optimizer import generate_scenario_dataset, score_synthetic_dataset

VERSION = "V110"
MODULE = "Synthetic→Real Bridge"

CONSENT_MODES = {
    "no_learning": {
        "label": "No learning",
        "raw_data_allowed": False,
        "profile_allowed": False,
        "cross_customer_reuse_allowed": False,
        "meaning": "Use customer data only for this order/delivery; do not store a reusable profile.",
    },
    "profile_only": {
        "label": "Profile only",
        "raw_data_allowed": False,
        "profile_allowed": True,
        "cross_customer_reuse_allowed": False,
        "meaning": "Store aggregate schema/statistics only; no raw rows and no cross-customer template reuse.",
    },
    "synthetic_calibration": {
        "label": "Synthetic calibration",
        "raw_data_allowed": False,
        "profile_allowed": True,
        "cross_customer_reuse_allowed": False,
        "meaning": "Use aggregate profile to tune synthetic generators for this/customer-specific pack outputs.",
    },
    "reusable_benchmark_template": {
        "label": "Reusable benchmark template",
        "raw_data_allowed": False,
        "profile_allowed": True,
        "cross_customer_reuse_allowed": True,
        "meaning": "Use anonymized aggregate profile to improve generic pack templates; requires explicit written permission.",
    },
}


def _utc_now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _hash_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()


def _df_fingerprint(df: pd.DataFrame) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "empty"
    sample = {
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing": {str(c): float(df[c].isna().mean()) for c in df.columns[:80]},
    }
    return _hash_text(json.dumps(sample, sort_keys=True, default=str))


def _numeric_profile(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(df, pd.DataFrame) or df.empty:
        return out
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    for c in numeric_cols[:80]:
        s = pd.to_numeric(df[c], errors="coerce")
        out[str(c)] = {
            "count": int(s.count()),
            "missing_pct": round(float(s.isna().mean()) * 100, 4),
            "mean": None if s.dropna().empty else round(float(s.mean()), 6),
            "std": None if s.dropna().empty else round(float(s.std(ddof=0)), 6),
            "min": None if s.dropna().empty else round(float(s.min()), 6),
            "p05": None if s.dropna().empty else round(float(s.quantile(0.05)), 6),
            "p50": None if s.dropna().empty else round(float(s.quantile(0.50)), 6),
            "p95": None if s.dropna().empty else round(float(s.quantile(0.95)), 6),
            "max": None if s.dropna().empty else round(float(s.max()), 6),
        }
    return out


def build_real_data_profile(real_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Build an aggregate, privacy-safer profile. Does not include raw rows."""
    if real_df is None or not isinstance(real_df, pd.DataFrame) or real_df.empty:
        return {
            "available": False,
            "rows": 0,
            "cols": 0,
            "columns": [],
            "numeric_profile": {},
            "timestamp_profile": {},
            "label_profile": {},
            "fingerprint": "none",
            "raw_rows_stored": False,
        }

    df = real_df.copy()
    cols = [str(c) for c in df.columns]
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    missing_pct = float(df.isna().mean().mean()) if len(df.columns) else 0.0

    timestamp_profile: Dict[str, Any] = {"detected": False}
    for candidate in ["timestamp", "time", "datetime", "date"]:
        matches = [c for c in df.columns if str(c).lower() == candidate]
        if matches:
            ts = pd.to_datetime(df[matches[0]], errors="coerce").dropna().sort_values()
            if len(ts) >= 3:
                deltas = ts.diff().dropna().dt.total_seconds()
                timestamp_profile = {
                    "detected": True,
                    "column": str(matches[0]),
                    "valid_pct": round(float(pd.to_datetime(df[matches[0]], errors="coerce").notna().mean()) * 100, 3),
                    "median_cadence_seconds": round(float(deltas.median()), 3) if not deltas.empty else None,
                    "p95_cadence_seconds": round(float(deltas.quantile(0.95)), 3) if not deltas.empty else None,
                }
            break

    label_profile: Dict[str, Any] = {"detected": False}
    for candidate in ["label", "status", "state", "class", "fault", "event"]:
        matches = [c for c in df.columns if str(c).lower() == candidate]
        if matches:
            vc = df[matches[0]].astype(str).value_counts().head(20)
            label_profile = {
                "detected": True,
                "column": str(matches[0]),
                "top_values": {str(k): int(v) for k, v in vc.to_dict().items()},
                "unique_count": int(df[matches[0]].nunique(dropna=True)),
            }
            break

    return {
        "available": True,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": cols[:120],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "numeric_column_count": int(len(numeric_cols)),
        "missing_pct": round(missing_pct * 100, 4),
        "numeric_profile": _numeric_profile(df),
        "timestamp_profile": timestamp_profile,
        "label_profile": label_profile,
        "fingerprint": _df_fingerprint(df),
        "raw_rows_stored": False,
        "profile_boundary": "Aggregate profile only; no raw customer rows included.",
    }


def _estimate_params_from_profile(profile: Dict[str, Any]) -> Dict[str, float]:
    if not profile or not profile.get("available"):
        return {"noise_level": 0.08, "missing_rate": 0.02, "drift_strength": 0.06, "imbalance_factor": 1.0}
    missing = float(profile.get("missing_pct", 2.0)) / 100.0
    # simple heuristic: messier data -> stronger stress profile
    noise = min(0.22, max(0.04, 0.05 + missing * 1.5))
    drift = 0.06
    ts = profile.get("timestamp_profile", {})
    if ts.get("detected") and ts.get("p95_cadence_seconds") and ts.get("median_cadence_seconds"):
        try:
            ratio = float(ts["p95_cadence_seconds"]) / max(float(ts["median_cadence_seconds"]), 1.0)
            if ratio > 3:
                drift += 0.04
        except Exception:
            pass
    imbalance = 1.0
    lp = profile.get("label_profile", {})
    if lp.get("detected") and lp.get("top_values"):
        counts = list(lp["top_values"].values())
        total = sum(counts) or 1
        if max(counts) / total > 0.82:
            imbalance = 0.75
    return {
        "noise_level": round(noise, 4),
        "missing_rate": round(min(0.20, max(0.0, missing)), 4),
        "drift_strength": round(min(0.25, max(0.02, drift)), 4),
        "imbalance_factor": round(imbalance, 4),
    }


def calibrate_synthetic_to_profile(
    synthetic_df: pd.DataFrame,
    real_profile: Dict[str, Any],
    seed: int = 110,
) -> pd.DataFrame:
    """Nudge synthetic numeric distributions toward aggregate real statistics when names overlap."""
    if not isinstance(synthetic_df, pd.DataFrame) or synthetic_df.empty:
        return pd.DataFrame()
    df = synthetic_df.copy()
    if not real_profile.get("available"):
        df["realism_calibration"] = "synthetic_baseline_no_real_profile"
        return df

    rng = np.random.default_rng(int(seed))
    real_numeric = real_profile.get("numeric_profile", {}) or {}
    overlap = [c for c in df.select_dtypes(include=[np.number]).columns if str(c) in real_numeric]
    for c in overlap:
        stats = real_numeric[str(c)]
        target_mean = stats.get("mean")
        target_std = stats.get("std")
        if target_mean is None or target_std is None or target_std == 0:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        mean = float(s.mean()) if s.notna().any() else 0.0
        std = float(s.std(ddof=0)) if s.notna().any() else 1.0
        if std <= 1e-9:
            std = 1.0
        calibrated = ((s - mean) / std) * float(target_std) + float(target_mean)
        jitter = rng.normal(0, max(abs(float(target_std)) * 0.015, 1e-6), len(df))
        df[c] = calibrated + jitter

    # apply real missingness profile to overlapping numeric cols only
    for c in overlap:
        miss_pct = float(real_numeric.get(str(c), {}).get("missing_pct", 0.0)) / 100.0
        if miss_pct > 0:
            mask = rng.random(len(df)) < min(miss_pct, 0.30)
            df.loc[mask, c] = np.nan

    df["realism_calibration"] = "aggregate_profile_calibrated" if overlap else "profile_available_no_column_overlap"
    return df


def build_learning_policy(consent_mode: str, purpose: str) -> Dict[str, Any]:
    if consent_mode not in CONSENT_MODES:
        consent_mode = "profile_only"
    mode = CONSENT_MODES[consent_mode]
    return {
        "mode": consent_mode,
        "label": mode["label"],
        "purpose": str(purpose or "pilot readiness / synthetic calibration"),
        "raw_customer_rows_stored_by_default": False,
        "raw_data_allowed": bool(mode["raw_data_allowed"]),
        "aggregate_profile_allowed": bool(mode["profile_allowed"]),
        "cross_customer_reuse_allowed": bool(mode["cross_customer_reuse_allowed"]),
        "requires_written_customer_permission": consent_mode in {"synthetic_calibration", "reusable_benchmark_template"},
        "retention_recommendation": "Keep raw customer files out of reusable product bundles; store only order-specific files under agreed retention. Prefer aggregate profiles for product learning.",
        "customer_safe_wording": "Customer data is used only for the agreed purpose. By default EdgeTwin stores aggregate profiles for calibration, not raw customer rows, unless a separate written agreement allows more.",
        "forbidden_without_extra_approval": [
            "mix raw customer data into generic demo datasets",
            "reuse identifiable customer data across customers",
            "claim production accuracy from synthetic calibration alone",
            "store personal data without lawful basis and retention plan",
        ],
    }


def build_synthetic_real_bridge_snapshot(
    project_name: str = "EdgeTwin Project",
    pack_key: str = "rotating_machinery",
    rows: int = 2500,
    seed: int = 110,
    consent_mode: str = "profile_only",
    purpose: str = "pilot readiness / synthetic calibration",
    real_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    real_profile = build_real_data_profile(real_df)
    params = _estimate_params_from_profile(real_profile)
    base_df, base_manifest = generate_scenario_dataset(
        pack_key=pack_key,
        rows=rows,
        seed=seed,
        noise_level=params["noise_level"],
        missing_rate=params["missing_rate"],
        drift_strength=params["drift_strength"],
        imbalance_factor=params["imbalance_factor"],
        include_edge_cases=True,
    )
    calibrated_df = calibrate_synthetic_to_profile(base_df, real_profile, seed=seed)
    quality = score_synthetic_dataset(calibrated_df, base_manifest)
    policy = build_learning_policy(consent_mode, purpose)

    realism_score = int(quality.get("synthetic_quality_score", 0))
    if real_profile.get("available"):
        realism_score = min(100, realism_score + 3)
    if consent_mode == "no_learning" and real_profile.get("available"):
        realism_score = max(0, realism_score - 8)
    if real_profile.get("available") and real_profile.get("numeric_column_count", 0) >= 4:
        data_bridge = "REAL PROFILE AVAILABLE"
    elif real_profile.get("available"):
        data_bridge = "LIMITED REAL PROFILE AVAILABLE"
    else:
        data_bridge = "NO REAL PROFILE - SYNTHETIC ONLY"

    blockers: List[str] = []
    if consent_mode == "reusable_benchmark_template":
        blockers.append("Requires explicit written permission before using aggregate profile across customers.")
    if real_profile.get("available") and real_profile.get("rows", 0) < 100:
        blockers.append("Customer sample is small; use only as weak calibration signal.")

    decision = "SYNTHETIC-REAL BRIDGE READY"
    if blockers:
        decision = "READY WITH CONSENT/REVIEW GUARDS"
    if realism_score < 80:
        decision = "TUNE BEFORE CUSTOMER DEMO"

    recommendations = [
        {"priority": 1, "action": "Keep synthetic golden datasets separate from raw customer data.", "why": "Prevents privacy/IP leakage and keeps demos reproducible."},
        {"priority": 2, "action": "Use aggregate customer profiles to tune noise, missingness, drift and schema realism.", "why": "Makes synthetic tests closer to field conditions without copying customer data."},
        {"priority": 3, "action": "Require written permission before reusing profiles across customers.", "why": "Protects customer trust and keeps EdgeTwin commercially safe."},
        {"priority": 4, "action": "Do not claim accuracy from synthetic-real calibration alone.", "why": "Production claims need representative labelled field validation."},
    ]
    if not real_profile.get("available"):
        recommendations.append({"priority": 5, "action": "Ask first pilot customers for a small CSV/Excel sample.", "why": "Even a small sample improves schema realism and demo credibility."})

    snapshot = {
        "version": VERSION,
        "module": MODULE,
        "created_at": _utc_now(),
        "project_name": project_name,
        "pack_key": pack_key,
        "base_manifest": base_manifest,
        "estimated_generation_params": params,
        "real_profile": real_profile,
        "consent": policy,
        "learning_policy": policy,
        "quality": quality,
        "realism_score": int(max(0, min(100, realism_score))),
        "decision": decision,
        "decision_card": {
            "data_bridge_status": data_bridge,
            "auto_use_allowed": int(realism_score) >= 82 and not blockers,
            "customer_demo_allowed": int(realism_score) >= 90,
            "real_data_learning_allowed": bool(policy.get("aggregate_profile_allowed")),
            "raw_data_reuse_allowed": False,
            "production_accuracy_claim_allowed": False,
            "founder_review_required": bool(blockers),
            "blockers": blockers,
        },
        "recommendations": recommendations,
        "customer_safe_summary": "V110 improves synthetic realism using scenario generation and, when allowed, aggregate customer-data profiles. It does not store raw customer rows in reusable bundles and does not create production accuracy/compliance guarantees.",
        "important_boundary": "Synthetic-real calibration improves demos/regression/data-quality gates. Production accuracy requires representative labelled customer field data and approval.",
        "calibrated_dataset_hash": _df_fingerprint(calibrated_df),
        "raw_customer_rows_in_bundle": False,
    }
    return snapshot, calibrated_df
