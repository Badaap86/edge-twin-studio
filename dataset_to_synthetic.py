"""EdgeTwin V115 Dataset-to-Synthetic Calibration Loop.

Purpose:
- Turn imported public/real/customer-approved dataset profiles into better synthetic scenario data.
- Try multiple synthetic parameter candidates, score them against quality, benchmark readiness,
  synthetic reliability and optional real/profile fidelity.
- Keep raw customer rows out of reusable bundles by default.

Boundary:
- V115 improves demo/regression/pilot-preparation synthetic data.
- It does not prove production accuracy, legal/compliance certification or real field reliability.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from synthetic_data_optimizer import (
    SCENARIO_LIBRARY,
    generate_scenario_dataset,
    score_synthetic_dataset,
)
from synthetic_real_bridge import build_real_data_profile, calibrate_synthetic_to_profile, build_learning_policy
from synthetic_reliability_lab import build_synthetic_reliability_lab_snapshot
from dataset_benchmark_harness import validate_dataset_v112, BENCHMARK_PROFILES

VERSION = "V115"
MODULE = "Dataset-to-Synthetic Calibration Loop"

CONSENT_MODES = {
    "benchmark_only": {
        "label": "Benchmark only",
        "raw_reuse_allowed": False,
        "profile_learning_allowed": True,
        "reusable_template_allowed": False,
        "note": "Use public/benchmark/profile signals for testing; do not redistribute raw source data.",
    },
    "customer_no_learning": {
        "label": "Customer delivery only / no learning",
        "raw_reuse_allowed": False,
        "profile_learning_allowed": False,
        "reusable_template_allowed": False,
        "note": "Use customer data only for the active delivery; do not improve reusable templates from it.",
    },
    "customer_profile_only": {
        "label": "Customer profile only",
        "raw_reuse_allowed": False,
        "profile_learning_allowed": True,
        "reusable_template_allowed": False,
        "note": "Use aggregate profile statistics only; raw customer rows remain separated.",
    },
    "synthetic_calibration_consent": {
        "label": "Synthetic calibration consent",
        "raw_reuse_allowed": False,
        "profile_learning_allowed": True,
        "reusable_template_allowed": True,
        "note": "Use aggregate profiles to improve synthetic scenario parameters/templates.",
    },
    "reusable_benchmark_template_consent": {
        "label": "Reusable benchmark template consent",
        "raw_reuse_allowed": False,
        "profile_learning_allowed": True,
        "reusable_template_allowed": True,
        "note": "Use aggregate patterns to improve reusable benchmark/golden synthetic templates; do not bundle raw source rows.",
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
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date)):
        return value.isoformat()
    return value


def _hash_df(df: pd.DataFrame) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "empty"
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8", errors="replace")).hexdigest()


def _safe_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    return max(lo, min(hi, v))


def _safe_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = default
    return max(lo, min(hi, v))


def _numeric_profile_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(df, pd.DataFrame) or df.empty:
        return out
    for c in df.select_dtypes(include=[np.number]).columns[:80]:
        if str(c) in {"is_anomaly"}:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 5:
            continue
        out[str(c)] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "p05": float(s.quantile(0.05)),
            "p50": float(s.quantile(0.50)),
            "p95": float(s.quantile(0.95)),
            "missing_pct": float(pd.to_numeric(df[c], errors="coerce").isna().mean()) * 100,
        }
    return out


def _profile_fidelity_score(synthetic_df: pd.DataFrame, reference_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Compare overlapping numeric distributions against an aggregate profile."""
    if not isinstance(synthetic_df, pd.DataFrame) or synthetic_df.empty:
        return {"available": False, "score": 0, "reason": "empty synthetic data"}
    ref_numeric = reference_profile.get("numeric_profile") if isinstance(reference_profile, dict) else None
    if not isinstance(ref_numeric, dict) or not ref_numeric:
        return {"available": False, "score": 72, "reason": "no reference numeric profile"}

    rows: List[Dict[str, Any]] = []
    distances: List[float] = []
    for c in synthetic_df.select_dtypes(include=[np.number]).columns[:80]:
        key = str(c)
        if key not in ref_numeric:
            continue
        s = pd.to_numeric(synthetic_df[c], errors="coerce").dropna()
        if len(s) < 10:
            continue
        ref = ref_numeric[key]
        syn_vec = np.array([s.mean(), s.std(ddof=0), s.quantile(0.05), s.quantile(0.50), s.quantile(0.95)], dtype=float)
        ref_vec = np.array([
            ref.get("mean", np.nan), ref.get("std", np.nan), ref.get("p05", np.nan), ref.get("p50", np.nan), ref.get("p95", np.nan)
        ], dtype=float)
        if np.isnan(ref_vec).all():
            continue
        scale = max(float(np.nanmax(np.abs(ref_vec))), float(np.nanmax(np.abs(syn_vec))), 1.0)
        dist = float(np.nanmean(np.abs(syn_vec - ref_vec) / scale))
        distances.append(dist)
        rows.append({"column": key, "normalized_profile_distance": round(dist, 5)})

    if not rows:
        return {"available": False, "score": 68, "reason": "no overlapping numeric columns"}
    avg = float(np.mean(distances))
    score = max(0, min(100, int(round(100 * (1 - min(avg, 1.0))))))
    return {
        "available": True,
        "score": score,
        "avg_normalized_distance": round(avg, 5),
        "overlap_column_count": len(rows),
        "by_column": rows,
    }


def _row_similarity_privacy_check(synthetic_df: pd.DataFrame, reference_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Cheap exact-row fingerprint check on overlapping columns to avoid accidental raw-row copying."""
    if not isinstance(reference_df, pd.DataFrame) or reference_df.empty or not isinstance(synthetic_df, pd.DataFrame) or synthetic_df.empty:
        return {"available": False, "score": 90, "risk": "low", "note": "no raw reference rows supplied"}
    common = [c for c in synthetic_df.columns if c in reference_df.columns][:40]
    if not common:
        return {"available": False, "score": 88, "risk": "low", "note": "no common columns for exact-row similarity"}
    syn = synthetic_df[common].head(5000).astype(str).agg("|".join, axis=1)
    ref = reference_df[common].head(5000).astype(str).agg("|".join, axis=1)
    ref_hashes = set(hashlib.sha256(x.encode("utf-8", errors="replace")).hexdigest() for x in ref)
    matches = sum(1 for x in syn if hashlib.sha256(x.encode("utf-8", errors="replace")).hexdigest() in ref_hashes)
    pct = 100 * matches / max(1, len(syn))
    if pct > 2.0:
        score, risk = 20, "high"
    elif pct > 0.2:
        score, risk = 60, "medium"
    else:
        score, risk = 96, "low"
    return {"available": True, "score": score, "risk": risk, "exact_match_pct": round(pct, 4), "common_column_count": len(common)}


def _candidate_grid(base_noise: float, base_missing: float, base_drift: float, base_imbalance: float, max_candidates: int) -> List[Dict[str, float]]:
    presets = [
        (base_noise, base_missing, base_drift, base_imbalance),
        (base_noise * 0.75, base_missing * 0.70, base_drift * 0.75, base_imbalance * 1.05),
        (base_noise * 1.15, base_missing * 1.25, base_drift * 1.20, base_imbalance * 0.90),
        (base_noise * 1.45, base_missing * 1.60, base_drift * 1.50, base_imbalance * 0.75),
        (base_noise * 0.95, base_missing * 1.40, base_drift * 1.10, base_imbalance * 1.25),
        (base_noise * 1.25, base_missing * 0.90, base_drift * 1.65, base_imbalance * 0.85),
        (base_noise * 0.55, base_missing * 0.40, base_drift * 0.45, base_imbalance * 1.00),
        (base_noise * 1.70, base_missing * 2.00, base_drift * 1.90, base_imbalance * 0.65),
    ]
    out = []
    seen = set()
    for noise, missing, drift, imbalance in presets:
        item = {
            "noise_level": round(_safe_float(noise, base_noise, 0.0, 0.60), 4),
            "missing_rate": round(_safe_float(missing, base_missing, 0.0, 0.50), 4),
            "drift_strength": round(_safe_float(drift, base_drift, 0.0, 0.75), 4),
            "imbalance_factor": round(_safe_float(imbalance, base_imbalance, 0.2, 3.0), 4),
        }
        key = tuple(item.values())
        if key not in seen:
            out.append(item); seen.add(key)
        if len(out) >= max_candidates:
            break
    return out


def build_dataset_to_synthetic_calibration_v115_snapshot(
    project_name: str = "EdgeTwin Project",
    pack_key: str = "rotating_machinery",
    rows: int = 2500,
    seed: int = 115,
    reference_df: Optional[pd.DataFrame] = None,
    consent_mode: str = "benchmark_only",
    benchmark_profile: str = "pilot_evidence_ready",
    max_candidates: int = 6,
    base_noise: float = 0.075,
    base_missing: float = 0.020,
    base_drift: float = 0.065,
    base_imbalance: float = 1.0,
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """Run a bounded calibration loop and return the best synthetic dataset candidate."""
    if pack_key not in SCENARIO_LIBRARY:
        pack_key = "rotating_machinery"
    if benchmark_profile not in BENCHMARK_PROFILES:
        benchmark_profile = "pilot_evidence_ready"
    if consent_mode not in CONSENT_MODES:
        consent_mode = "benchmark_only"
    rows = _safe_int(rows, 2500, 300, 50000)
    seed = _safe_int(seed, 115, 1, 2_000_000_000)
    max_candidates = _safe_int(max_candidates, 6, 2, 12)

    policy = CONSENT_MODES[consent_mode]
    reference_available = isinstance(reference_df, pd.DataFrame) and not reference_df.empty
    reference_profile = build_real_data_profile(reference_df) if reference_available and policy["profile_learning_allowed"] else build_real_data_profile(None)

    # Baseline candidate before calibration.
    baseline_df, baseline_manifest = generate_scenario_dataset(
        pack_key=pack_key,
        rows=rows,
        seed=seed,
        noise_level=base_noise,
        missing_rate=base_missing,
        drift_strength=base_drift,
        imbalance_factor=base_imbalance,
        include_edge_cases=True,
    )
    baseline_quality = score_synthetic_dataset(baseline_df, baseline_manifest)
    baseline_validation = validate_dataset_v112(baseline_df, pack_key, benchmark_profile, "baseline_synthetic_v115")
    baseline_fidelity = _profile_fidelity_score(baseline_df, reference_profile)

    candidates: List[Dict[str, Any]] = []
    candidate_datasets: Dict[str, pd.DataFrame] = {"baseline_synthetic_v115": baseline_df}
    best_df = baseline_df
    best_record: Dict[str, Any] = {}

    for idx, params in enumerate(_candidate_grid(base_noise, base_missing, base_drift, base_imbalance, max_candidates), start=1):
        df, manifest = generate_scenario_dataset(
            pack_key=pack_key,
            rows=rows,
            seed=seed + idx * 101,
            noise_level=params["noise_level"],
            missing_rate=params["missing_rate"],
            drift_strength=params["drift_strength"],
            imbalance_factor=params["imbalance_factor"],
            include_edge_cases=True,
        )
        # Calibrate only from aggregate profile and only if the consent/policy allows profile learning.
        if reference_profile.get("available") and policy["profile_learning_allowed"]:
            df = calibrate_synthetic_to_profile(df, reference_profile, seed=seed + idx * 211)
            manifest["calibrated_to_reference_profile"] = True
        else:
            manifest["calibrated_to_reference_profile"] = False

        quality = score_synthetic_dataset(df, manifest)
        validation = validate_dataset_v112(df, pack_key, benchmark_profile, f"candidate_{idx}_v115")
        fidelity = _profile_fidelity_score(df, reference_profile)
        privacy = _row_similarity_privacy_check(df, reference_df if reference_available else None)

        quality_score = int(quality.get("synthetic_quality_score", quality.get("score", 0)) or 0)
        benchmark_score = int(validation.get("benchmark_score", 0) or 0)
        fidelity_score = int(fidelity.get("score", 72) or 0)
        privacy_score = int(privacy.get("score", 90) or 0)
        # If no reference profile exists, fidelity is a lighter term and benchmark/quality dominate.
        if fidelity.get("available"):
            composite = int(round(0.34 * quality_score + 0.34 * benchmark_score + 0.22 * fidelity_score + 0.10 * privacy_score))
        else:
            composite = int(round(0.45 * quality_score + 0.40 * benchmark_score + 0.15 * privacy_score))

        record = {
            "candidate": f"candidate_{idx}",
            **params,
            "dataset_hash": _hash_df(df),
            "calibrated_to_reference_profile": bool(manifest.get("calibrated_to_reference_profile")),
            "synthetic_quality_score": quality_score,
            "benchmark_score": benchmark_score,
            "fidelity_score": fidelity_score,
            "privacy_score": privacy_score,
            "composite_score": composite,
            "benchmark_decision": validation.get("decision"),
            "privacy_risk": privacy.get("risk"),
            "fidelity_overlap_columns": fidelity.get("overlap_column_count", 0),
            "validation_blockers": validation.get("blockers", []),
        }
        candidates.append(record)
        candidate_datasets[f"candidate_{idx}_v115"] = df
        if not best_record or composite > int(best_record.get("composite_score", -1)):
            best_record = record
            best_df = df

    best_validation = validate_dataset_v112(best_df, pack_key, benchmark_profile, "best_calibrated_synthetic_v115")
    best_quality = score_synthetic_dataset(best_df, {"pack_key": pack_key, "rows": rows, **best_record})
    best_fidelity = _profile_fidelity_score(best_df, reference_profile)
    best_privacy = _row_similarity_privacy_check(best_df, reference_df if reference_available else None)

    try:
        v111_snapshot, _ = build_synthetic_reliability_lab_snapshot(
            project_name=project_name,
            pack_key=pack_key,
            rows=min(max(rows, 500), 4000),
            seed=seed + 1150,
            real_df=reference_df if reference_available and policy["profile_learning_allowed"] else None,
            use_real_profile=bool(reference_available and policy["profile_learning_allowed"]),
            stress_profile_keys=["clean_demo", "realistic_messy", "field_stress", "bad_input_blocker"],
        )
        v111_score = int(v111_snapshot.get("synthetic_reliability_score", 0) or 0)
        v111_decision = v111_snapshot.get("decision")
    except Exception as exc:
        v111_score = 0
        v111_decision = f"V111_ERROR: {exc}"

    baseline_score = int(round(0.50 * int(baseline_quality.get("synthetic_quality_score", 0) or 0) + 0.35 * int(baseline_validation.get("benchmark_score", 0) or 0) + 0.15 * int(baseline_fidelity.get("score", 72) or 0)))
    best_score = int(best_record.get("composite_score", 0) or 0)
    improvement = best_score - baseline_score
    final_score = int(round(0.55 * best_score + 0.25 * int(best_validation.get("benchmark_score", 0) or 0) + 0.20 * v111_score))

    blockers: List[str] = []
    review_flags: List[str] = []
    if best_privacy.get("risk") in {"medium", "high"}:
        blockers.append("synthetic output is too similar to reference rows; do not reuse or export as a reusable template")
    if consent_mode == "customer_no_learning" and reference_available:
        review_flags.append("customer_no_learning selected: use any reference/profile signal only for the active delivery, not reusable templates")
    if reference_available and not policy["profile_learning_allowed"]:
        review_flags.append("reference data supplied but consent mode does not allow profile learning; calibration used synthetic baseline only")
    if best_validation.get("decision") != "BENCHMARK PASS":
        review_flags.append("best synthetic candidate did not fully pass the selected V112 benchmark profile")
    if v111_score < 85:
        review_flags.append("V111 stress reliability signal below preferred threshold")

    golden_candidate = final_score >= 92 and not blockers and policy["reusable_template_allowed"]
    if final_score >= 92 and not blockers:
        decision = "CALIBRATED GOLDEN SYNTHETIC GO"
    elif final_score >= 82 and not blockers:
        decision = "CALIBRATED SYNTHETIC CONDITIONAL GO"
    else:
        decision = "CALIBRATION REVIEW REQUIRED"

    if consent_mode in {"customer_no_learning", "customer_profile_only"} and golden_candidate:
        golden_candidate = False
        review_flags.append("golden reusable template disabled by consent mode")

    recommendations: List[Dict[str, str]] = []
    if improvement <= 0:
        recommendations.append({"priority": "medium", "action": "Calibration did not improve the baseline; review reference mapping, units and overlapping columns."})
    if not reference_profile.get("available"):
        recommendations.append({"priority": "medium", "action": "Add public benchmark metadata or consent-controlled customer profiles to improve synthetic-to-real fidelity."})
    if best_fidelity.get("available") and best_fidelity.get("score", 0) < 75:
        recommendations.append({"priority": "high", "action": "Fidelity to reference profile is still weak; tune units/columns or use a closer scenario pack."})
    if best_validation.get("benchmark_score", 0) < 88:
        recommendations.append({"priority": "high", "action": "Keep this synthetic dataset for testing only; regenerate before using it as a golden demo/regression dataset."})
    if not recommendations:
        recommendations.append({"priority": "normal", "action": "Use the best candidate as controlled demo/regression/pilot-preparation synthetic data, not as production accuracy proof."})

    validation_contract = {
        "can_prove": [
            "synthetic generator is closer to selected public/real aggregate profiles",
            "known-label synthetic datasets can support regression and data-quality gate tests",
            "benchmark readiness improved or is measurable against V112/V111",
            "privacy/similarity risks are checked before reuse",
        ],
        "cannot_prove": [
            "production accuracy",
            "legal/compliance certification",
            "field reliability for a specific customer without representative labelled validation",
            "permission to reuse customer data beyond the selected consent mode",
        ],
    }

    candidate_table = sorted(candidates, key=lambda r: int(r.get("composite_score", 0)), reverse=True)
    snapshot = _json_safe({
        "version": VERSION,
        "module": MODULE,
        "created_at": _utc_now(),
        "project_name": project_name,
        "pack_key": pack_key,
        "pack_title": SCENARIO_LIBRARY.get(pack_key, {}).get("title", pack_key),
        "benchmark_profile": benchmark_profile,
        "consent_mode": consent_mode,
        "consent_policy": policy,
        "decision": decision,
        "calibration_score": final_score,
        "baseline_score": baseline_score,
        "best_candidate_score": best_score,
        "improvement_points": improvement,
        "golden_candidate_allowed": bool(golden_candidate),
        "raw_reference_rows_in_bundle": False,
        "reference_profile_available": bool(reference_profile.get("available")),
        "reference_profile_summary": {
            "rows": reference_profile.get("rows", 0),
            "cols": reference_profile.get("cols", 0),
            "numeric_column_count": len(reference_profile.get("numeric_profile", {}) or {}),
            "label_columns": reference_profile.get("label_columns", []),
        },
        "rows": rows,
        "seed": seed,
        "candidate_count": len(candidate_table),
        "best_candidate": best_record,
        "candidate_table": candidate_table,
        "best_validation": best_validation,
        "best_quality": best_quality,
        "best_fidelity": best_fidelity,
        "best_privacy_similarity": best_privacy,
        "v111_synthetic_reliability_score": v111_score,
        "v111_decision": v111_decision,
        "blockers": blockers,
        "review_flags": review_flags,
        "recommendations": recommendations,
        "learning_policy": build_learning_policy("synthetic_calibration" if policy["profile_learning_allowed"] else "no_learning", "dataset-to-synthetic calibration"),
        "validation_contract": validation_contract,
        "customer_safe_summary": "V115 tunes synthetic scenario data toward benchmark/real aggregate profiles and selects the best measured candidate for demo, regression and pilot-preparation use. It does not create production accuracy, legal or compliance guarantees.",
        "important_boundary": "Use calibrated synthetic data for testing, demo, benchmark and pilot-readiness. Do not claim real-world accuracy without representative labelled customer validation.",
    })
    datasets = {
        "baseline_synthetic_v115": baseline_df,
        "best_calibrated_synthetic_v115": best_df,
    }
    # Keep only the top 3 candidates in exported datasets to avoid huge bundles.
    for rec in candidate_table[:3]:
        name = f"{rec.get('candidate')}_v115"
        df = candidate_datasets.get(name)
        if isinstance(df, pd.DataFrame):
            datasets[name] = df
    return snapshot, datasets


def build_v115_candidate_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(snapshot.get("candidate_table", []))


def create_synthetic_calibration_v115_bundle(snapshot: Dict[str, Any], datasets: Optional[Dict[str, pd.DataFrame]] = None, project_name: str = "EdgeTwin_Project") -> bytes:
    bio = io.BytesIO()
    datasets = datasets or {}
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dataset_to_synthetic.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr("candidate_scores_v115.csv", build_v115_candidate_table(snapshot).to_csv(index=False))
        zf.writestr("recommendations_v115.csv", pd.DataFrame(snapshot.get("recommendations", [])).to_csv(index=False))
        zf.writestr("validation_contract_v115.json", json.dumps(_json_safe(snapshot.get("validation_contract", {})), indent=2, ensure_ascii=False))
        zf.writestr("best_validation_v115.json", json.dumps(_json_safe(snapshot.get("best_validation", {})), indent=2, ensure_ascii=False))
        zf.writestr("best_fidelity_v115.json", json.dumps(_json_safe(snapshot.get("best_fidelity", {})), indent=2, ensure_ascii=False))
        zf.writestr("consent_policy_v115.json", json.dumps(_json_safe(snapshot.get("consent_policy", {})), indent=2, ensure_ascii=False))
        for name, df in datasets.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                safe_name = str(name).replace("/", "_").replace("\\", "_")[:80]
                zf.writestr(f"{safe_name}.csv", df.to_csv(index=False))
        zf.writestr("README_V115_SYNTHETIC_CALIBRATION_LOOP.md", f"""# EdgeTwin Dataset-to-Synthetic Calibration Loop V115

Project: {project_name}
Decision: {snapshot.get('decision')}
Calibration score: {snapshot.get('calibration_score')}
Baseline score: {snapshot.get('baseline_score')}
Improvement: {snapshot.get('improvement_points')} points
Pack: {snapshot.get('pack_title')}
Consent mode: {snapshot.get('consent_mode')}
Golden candidate allowed: {snapshot.get('golden_candidate_allowed')}
Raw reference rows in bundle: {snapshot.get('raw_reference_rows_in_bundle')}

## What V115 does
- Generates multiple synthetic candidates with controlled noise, missingness, drift and imbalance.
- Uses aggregate public/real/customer-approved profiles when policy allows it.
- Scores candidates with synthetic quality, V112 benchmark readiness, fidelity and privacy/similarity checks.
- Selects the best measured candidate and produces a reusable manifest/bundle where allowed.

## What this does not prove
- Production accuracy.
- Legal/compliance certification.
- Permission to reuse raw customer data.
- Real field reliability without representative labelled validation.

Boundary: {snapshot.get('important_boundary')}
""")
    bio.seek(0)
    return bio.getvalue()
