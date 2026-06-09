from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import zipfile
from typing import Any, Dict, List, Optional

import pandas as pd
try:
    from fpdf import FPDF
except Exception:  # pragma: no cover
    FPDF = None


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_obj(obj: Any) -> str:
    return _hash_bytes(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def _clamp(value: Any) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except Exception:
        return 0


def _safe_text(value: Any) -> str:
    return str(value).replace("→", "->").replace("–", "-").replace("—", "-").replace("≥", ">=").replace("≤", "<=")


def _pdf_bytes(title: str, lines: List[str]) -> bytes:
    # Lightweight PDF-compatible placeholder for bundle previews.
    # Kept as bytes so the bundle contract stays stable even when PDF engines differ.
    return (title + "\n\n" + "\n".join(map(_safe_text, lines))).encode("utf-8", errors="replace")


def _zip_bundle(files: Dict[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, payload in files.items():
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            z.writestr(name, payload)
    return buf.getvalue()

"""EdgeTwin Studio V121 - Golden Dataset Library + Version Lock.

Purpose:
- Freeze approved synthetic/public/customer-consented benchmark datasets as versioned golden sets.
- Store hash, profile, expected benchmark score and expected output contract.
- Make regression tests repeatable so future EdgeTwin versions cannot silently drift.
"""

VERSION = "121.0"
MODULE = "Golden Dataset Library + Version Lock"

SCENARIOS = {
    "rotating_machinery": {"min_rows": 500, "required": ["timestamp", "machine_id", "vibration_rms", "temperature_c", "current_a", "label"]},
    "audio_event": {"min_rows": 200, "required": ["file", "fold", "category", "target"]},
    "energy_operations": {"min_rows": 300, "required": ["timestamp", "asset_id", "power_kw", "temperature_c", "label"]},
}


def build_sample_golden_dataset(scenario: str = "rotating_machinery", rows: int = 720) -> pd.DataFrame:
    scenario = scenario if scenario in SCENARIOS else "rotating_machinery"
    rows = max(60, int(rows))
    base = pd.date_range("2026-01-01", periods=rows, freq="5min")
    if scenario == "audio_event":
        cats = ["normal_motor", "bearing_noise", "impact", "air_leak", "background"]
        return pd.DataFrame({
            "file": [f"sample_{i:05d}.wav" for i in range(rows)],
            "fold": [(i % 5) + 1 for i in range(rows)],
            "category": [cats[i % len(cats)] for i in range(rows)],
            "target": [i % len(cats) for i in range(rows)],
            "duration_s": [5.0 for _ in range(rows)],
        })
    if scenario == "energy_operations":
        return pd.DataFrame({
            "timestamp": base,
            "asset_id": [f"pump_{(i % 4) + 1}" for i in range(rows)],
            "power_kw": [round(7.0 + (i % 48) * 0.03 + (2.2 if i % 137 == 0 else 0), 3) for i in range(rows)],
            "temperature_c": [round(44 + (i % 96) * 0.04 + (5 if i % 149 == 0 else 0), 2) for i in range(rows)],
            "label": ["anomaly" if i % 137 == 0 or i % 149 == 0 else "normal" for i in range(rows)],
        })
    return pd.DataFrame({
        "timestamp": base,
        "machine_id": [f"motor_{(i % 3) + 1}" for i in range(rows)],
        "vibration_rms": [round(0.38 + (i % 60) * 0.004 + (0.55 if i % 113 == 0 else 0), 4) for i in range(rows)],
        "temperature_c": [round(58 + (i % 72) * 0.05 + (6 if i % 157 == 0 else 0), 2) for i in range(rows)],
        "current_a": [round(3.1 + (i % 48) * 0.015 + (0.9 if i % 131 == 0 else 0), 3) for i in range(rows)],
        "label": ["fault" if i % 113 == 0 or i % 157 == 0 or i % 131 == 0 else "normal" for i in range(rows)],
    })


def _profile_df(df: pd.DataFrame) -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"rows": 0, "cols": 0, "columns": [], "numeric_columns": [], "missing_ratio": 1.0, "label_distribution": {}}
    label_col = "label" if "label" in df.columns else ("category" if "category" in df.columns else None)
    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": list(map(str, df.columns)),
        "numeric_columns": [str(c) for c in df.select_dtypes(include="number").columns],
        "missing_ratio": round(float(df.isna().mean().mean()), 4),
        "duplicate_rows": int(df.duplicated().sum()),
        "label_column": label_col,
        "label_distribution": (df[label_col].astype(str).value_counts().head(20).to_dict() if label_col else {}),
    }


def build_golden_dataset_library_snapshot(
    dataset_name: str = "EdgeTwin golden rotating machinery v1",
    scenario: str = "rotating_machinery",
    dataset_df: Optional[pd.DataFrame] = None,
    dataset_version: str = "1.0.0",
    source_type: str = "golden_synthetic",
    license_or_consent: str = "internal synthetic benchmark - reusable",
    expected_benchmark_score: int = 90,
    expected_quality_gate: str = "pilot_evidence_ready",
) -> Dict[str, Any]:
    scenario = scenario if scenario in SCENARIOS else "rotating_machinery"
    df = dataset_df if isinstance(dataset_df, pd.DataFrame) and not dataset_df.empty else build_sample_golden_dataset(scenario=scenario)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    profile = _profile_df(df)
    req = SCENARIOS[scenario]["required"]
    missing_required = [c for c in req if c not in df.columns]
    min_rows = SCENARIOS[scenario]["min_rows"]
    score = 100
    if missing_required:
        score -= 25 + len(missing_required) * 5
    if profile["rows"] < min_rows:
        score -= 18
    if profile["missing_ratio"] > 0.08:
        score -= 10
    if profile["duplicate_rows"] > max(3, profile["rows"] * 0.02):
        score -= 8
    if not profile.get("label_distribution"):
        score -= 10
    score = _clamp(score)
    decision = "GOLDEN DATASET VERSION LOCKED" if score >= 88 and not missing_required else "GOLDEN DATASET NEEDS REVIEW"
    contract = {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "scenario": scenario,
        "dataset_hash_sha256": _hash_bytes(csv_bytes),
        "expected_benchmark_score_min": int(expected_benchmark_score),
        "expected_quality_gate": expected_quality_gate,
        "expected_output_contract": [
            "profile can be reproduced from manifest hash",
            "benchmark score should remain within expected tolerance",
            "unsafe claims remain blocked",
            "dataset is not production accuracy proof",
        ],
    }
    return {
        "version": VERSION,
        "module": MODULE,
        "created_at": _now(),
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "scenario": scenario,
        "source_type": source_type,
        "license_or_consent": license_or_consent,
        "profile": profile,
        "version_lock_contract": contract,
        "library_score": score,
        "decision": decision,
        "missing_required_columns": missing_required,
        "review_flags": [] if score >= 88 and not missing_required else ["Golden dataset should be fixed before release regression use."],
        "safe_boundary": "Golden datasets are for demo, regression, benchmark and readiness checks. They do not prove customer production accuracy.",
    }


def create_golden_dataset_library_bundle(snapshot: Dict[str, Any], dataset_df: Optional[pd.DataFrame] = None) -> bytes:
    scenario = snapshot.get("scenario", "rotating_machinery")
    df = dataset_df if isinstance(dataset_df, pd.DataFrame) and not dataset_df.empty else build_sample_golden_dataset(scenario=scenario)
    lines = [
        f"Decision: {snapshot.get('decision')}",
        f"Library score: {snapshot.get('library_score')}%",
        f"Dataset: {snapshot.get('dataset_name')} v{snapshot.get('dataset_version')}",
        f"Hash: {snapshot.get('version_lock_contract', {}).get('dataset_hash_sha256')}",
        f"Boundary: {snapshot.get('safe_boundary')}",
    ]
    return _zip_bundle({
        "v121_snapshot.json": json.dumps(snapshot, indent=2, default=str),
        "v121_golden_dataset.csv": df.to_csv(index=False),
        "v121_version_lock_contract.json": json.dumps(snapshot.get("version_lock_contract", {}), indent=2, default=str),
        "v121_profile.json": json.dumps(snapshot.get("profile", {}), indent=2, default=str),
        "v121_summary.pdf": _pdf_bytes("EdgeTwin V121 Golden Dataset Library", lines),
    })
