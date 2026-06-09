"""EdgeTwin Studio V114 dataset library import wizard.

Purpose:
- Add datasets such as ESC-50, UrbanSound8K, FSD50K, synthetic golden sets and customer-consented samples
  into a local EdgeTwin Dataset Library without mixing raw files into customer bundles.
- Create a dataset manifest with license/consent, folder/file inventory, metadata profile and benchmark handoff.
- Keep public benchmark data, synthetic data and customer data separated by policy.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fpdf import FPDF

V114_VERSION = "114.0"
DATASET_LIBRARY_ROOT = Path(os.environ.get("EDGETWIN_DATASET_LIBRARY_ROOT", "storage/dataset_library"))


@dataclass(frozen=True)
class DatasetSourcePolicy:
    source_type: str
    display_name: str
    allowed_learning_modes: List[str]
    default_learning_mode: str
    reuse_policy: str
    commercial_guard: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: Any) -> str:
    txt = str(value or "unknown")
    txt = re.sub(r"[^a-zA-Z0-9_.-]+", "_", txt).strip("._")
    return txt[:96] or "unknown"


def _safe_text(value: Any) -> str:
    return str(value or "").replace("\u2192", "->").replace("\u2013", "-").replace("\u2014", "-").replace("_", " ")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _hash_file(path: Path, max_bytes: int = 2_000_000) -> str:
    """Hash first bytes only by default so huge audio libraries do not freeze local installs."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            remaining = max_bytes
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
        return h.hexdigest()
    except Exception:
        return "unreadable"


def ensure_dataset_library_dirs() -> Dict[str, str]:
    DATASET_LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    dirs = {
        "root": DATASET_LIBRARY_ROOT,
        "public": DATASET_LIBRARY_ROOT / "public",
        "synthetic": DATASET_LIBRARY_ROOT / "synthetic",
        "customer": DATASET_LIBRARY_ROOT / "customer_consent",
        "manifests": DATASET_LIBRARY_ROOT / "manifests",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in dirs.items()}


def get_dataset_source_policies_v114() -> Dict[str, Dict[str, Any]]:
    policies = [
        DatasetSourcePolicy(
            source_type="public_benchmark",
            display_name="Public benchmark dataset",
            allowed_learning_modes=["benchmark_only", "feature_regression", "synthetic_inspiration"],
            default_learning_mode="benchmark_only",
            reuse_policy="Keep raw public files out of paid customer bundles unless license review explicitly allows redistribution.",
            commercial_guard="Public datasets can improve pipelines and demos, but do not prove customer production accuracy.",
        ),
        DatasetSourcePolicy(
            source_type="synthetic_golden",
            display_name="Golden synthetic dataset",
            allowed_learning_modes=["regression", "demo", "stress_test", "data_quality_gate"],
            default_learning_mode="regression",
            reuse_policy="Reusable inside EdgeTwin packs because it is generated/test-owned, but keep manifests and hashes version-locked.",
            commercial_guard="Synthetic data is evidence for workflow robustness, not production accuracy.",
        ),
        DatasetSourcePolicy(
            source_type="customer_no_learning",
            display_name="Customer dataset - no learning permission",
            allowed_learning_modes=["single_customer_delivery_only"],
            default_learning_mode="single_customer_delivery_only",
            reuse_policy="Use only for that customer/order. Do not reuse rows, profiles or findings in reusable templates.",
            commercial_guard="Treat as confidential customer material; keep out of shared benchmarks/templates.",
        ),
        DatasetSourcePolicy(
            source_type="customer_profile_only",
            display_name="Customer dataset - profile-only consent",
            allowed_learning_modes=["profile_only", "synthetic_calibration"],
            default_learning_mode="profile_only",
            reuse_policy="Store aggregate profile/statistics only; keep raw rows separated and do not redistribute.",
            commercial_guard="Use aggregate profiles to improve data-quality rules and synthetic calibration, not raw data resale.",
        ),
        DatasetSourcePolicy(
            source_type="customer_reusable_template_consent",
            display_name="Customer dataset - reusable template consent",
            allowed_learning_modes=["profile_only", "synthetic_calibration", "reusable_benchmark_template"],
            default_learning_mode="synthetic_calibration",
            reuse_policy="Use only within the written consent scope. Prefer aggregate/profile/template learning over raw row reuse.",
            commercial_guard="Requires explicit written consent and clear scope before improving reusable packs/templates.",
        ),
    ]
    return {p.source_type: asdict(p) for p in policies}


def get_known_dataset_profiles_v114() -> Dict[str, Dict[str, Any]]:
    return {
        "esc50": {
            "display_name": "ESC-50 Environmental Sound Classification",
            "source_type": "public_benchmark",
            "domain": "audio_event_classification",
            "expected_metadata_files": ["meta/esc50.csv", "esc50.csv", "metadata.csv"],
            "expected_columns": ["filename", "fold", "target", "category", "esc10", "src_file", "take"],
            "expected_raw_extensions": [".wav"],
            "recommended_library_subdir": "public/esc50",
            "license_guard": "ESC-50 is commonly used as a non-commercial public benchmark. Keep raw audio out of paid customer bundles unless reviewed.",
        },
        "urbansound8k": {
            "display_name": "UrbanSound8K",
            "source_type": "public_benchmark",
            "domain": "urban_audio_event_classification",
            "expected_metadata_files": ["metadata/UrbanSound8K.csv", "UrbanSound8K.csv", "metadata.csv"],
            "expected_columns": ["slice_file_name", "fsID", "start", "end", "salience", "fold", "classID", "class"],
            "expected_raw_extensions": [".wav"],
            "recommended_library_subdir": "public/urbansound8k",
            "license_guard": "Verify source/clip licenses before paid redistribution or commercial bundle inclusion.",
        },
        "fsd50k": {
            "display_name": "FSD50K",
            "source_type": "public_benchmark",
            "domain": "large_audio_event_classification",
            "expected_metadata_files": ["ground_truth/dev.csv", "ground_truth/eval.csv", "metadata.csv"],
            "expected_columns": ["fname", "labels", "mids", "split"],
            "expected_raw_extensions": [".wav"],
            "recommended_library_subdir": "public/fsd50k",
            "license_guard": "Per-clip Creative Commons licenses must be tracked before redistribution or paid bundle inclusion.",
        },
        "synthetic_golden": {
            "display_name": "EdgeTwin Golden Synthetic Dataset",
            "source_type": "synthetic_golden",
            "domain": "scenario_regression",
            "expected_metadata_files": ["manifest.json", "metadata.csv", "dataset.csv"],
            "expected_columns": ["timestamp", "scenario", "label"],
            "expected_raw_extensions": [".csv", ".parquet", ".json"],
            "recommended_library_subdir": "synthetic/golden",
            "license_guard": "Generated/test-owned dataset. Version-lock manifests and hashes before using for release gates.",
        },
        "customer_sample": {
            "display_name": "Customer sample dataset",
            "source_type": "customer_profile_only",
            "domain": "customer_real_data_sample",
            "expected_metadata_files": ["metadata.csv", "dataset.csv", "sample.csv"],
            "expected_columns": ["timestamp"],
            "expected_raw_extensions": [".csv", ".xlsx", ".json", ".parquet", ".wav"],
            "recommended_library_subdir": "customer_consent/customer_sample",
            "license_guard": "Requires customer permission. Choose no-learning/profile-only/template-consent mode explicitly.",
        },
    }


def find_metadata_file(dataset_path: str | Path, known_dataset_id: str) -> Optional[Path]:
    p = Path(dataset_path)
    profile = get_known_dataset_profiles_v114().get(known_dataset_id, {})
    candidates = profile.get("expected_metadata_files", [])
    for rel in candidates:
        cand = p / rel
        if cand.exists() and cand.is_file():
            return cand
    # fallback: first csv close to root
    try:
        csvs = sorted([x for x in p.rglob("*.csv") if x.is_file()], key=lambda x: (len(x.parts), str(x)))
        return csvs[0] if csvs else None
    except Exception:
        return None


def scan_dataset_folder_v114(dataset_path: str | Path, known_dataset_id: str = "esc50", max_files: int = 5000) -> Dict[str, Any]:
    p = Path(dataset_path)
    profile = get_known_dataset_profiles_v114().get(known_dataset_id, get_known_dataset_profiles_v114()["customer_sample"])
    if not p.exists() or not p.is_dir():
        return {
            "path_exists": False,
            "dataset_path": str(p),
            "file_count": 0,
            "total_mb_scanned": 0.0,
            "extension_counts": {},
            "metadata_file": None,
            "raw_file_count": 0,
            "inventory_warning": "folder_not_found",
        }

    files: List[Path] = []
    for idx, fp in enumerate(p.rglob("*")):
        if idx >= max_files:
            break
        if fp.is_file():
            files.append(fp)

    extension_counts: Dict[str, int] = {}
    total_bytes = 0
    sample_hashes: List[Dict[str, str]] = []
    for fp in files:
        ext = fp.suffix.lower() or "no_ext"
        extension_counts[ext] = extension_counts.get(ext, 0) + 1
        try:
            total_bytes += fp.stat().st_size
        except Exception:
            pass

    for fp in files[:25]:
        sample_hashes.append({"relative_path": str(fp.relative_to(p)), "sha256_first_bytes": _hash_file(fp)})

    raw_exts = set(profile.get("expected_raw_extensions", []))
    raw_file_count = sum(1 for fp in files if fp.suffix.lower() in raw_exts)
    metadata_file = find_metadata_file(p, known_dataset_id)

    warning = "ok"
    if len(files) >= max_files:
        warning = "inventory_truncated_increase_max_files_for_full_scan"
    elif not metadata_file:
        warning = "metadata_not_found"
    elif raw_file_count == 0:
        warning = "raw_files_not_detected_or_metadata_only_import"

    return {
        "path_exists": True,
        "dataset_path": str(p),
        "file_count": int(len(files)),
        "total_mb_scanned": round(total_bytes / (1024 * 1024), 3),
        "extension_counts": extension_counts,
        "metadata_file": str(metadata_file) if metadata_file else None,
        "raw_file_count": int(raw_file_count),
        "sample_file_hashes": sample_hashes,
        "inventory_warning": warning,
    }


def read_metadata_profile_v114(metadata_path: Optional[str | Path], known_dataset_id: str = "esc50") -> Dict[str, Any]:
    if not metadata_path:
        return {"metadata_present": False, "rows": 0, "columns": [], "missing_expected_columns": [], "class_count": 0, "fold_count": 0}
    p = Path(metadata_path)
    if not p.exists() or not p.is_file():
        return {"metadata_present": False, "rows": 0, "columns": [], "missing_expected_columns": [], "class_count": 0, "fold_count": 0, "error": "metadata_file_not_found"}
    try:
        df = pd.read_csv(p)
    except Exception as exc:
        return {"metadata_present": False, "rows": 0, "columns": [], "missing_expected_columns": [], "class_count": 0, "fold_count": 0, "error": str(exc)}

    known = get_known_dataset_profiles_v114().get(known_dataset_id, {})
    expected = set(known.get("expected_columns", []))
    cols = list(df.columns)
    missing = sorted(expected.difference(cols)) if expected else []
    fold_col = next((c for c in ["fold", "split"] if c in df.columns), None)
    label_col = next((c for c in ["category", "class", "labels", "label", "target", "classID", "scenario", "status"] if c in df.columns), None)
    ts_col = next((c for c in ["timestamp", "time", "date", "datetime"] if c in df.columns), None)

    class_count = int(df[label_col].nunique()) if label_col else 0
    fold_count = int(df[fold_col].nunique()) if fold_col else 0
    missing_ratio = float(df.isna().mean().mean()) if len(df) else 0.0
    duplicate_ratio = float(df.duplicated().mean()) if len(df) else 0.0

    label_balance = "unknown"
    if label_col and len(df):
        vc = df[label_col].value_counts()
        if not vc.empty:
            ratio = float(vc.max() / max(vc.min(), 1))
            label_balance = "balanced" if ratio <= 2.0 else "imbalanced"

    return {
        "metadata_present": True,
        "metadata_path": str(p),
        "rows": int(len(df)),
        "columns": cols,
        "missing_expected_columns": missing,
        "class_count": class_count,
        "fold_count": fold_count,
        "label_column": label_col,
        "fold_column": fold_col,
        "timestamp_column": ts_col,
        "missing_ratio": round(missing_ratio, 4),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "label_balance": label_balance,
        "metadata_schema_hash": _sha256_text("|".join(cols)),
    }


def build_dataset_import_manifest_v114(
    dataset_name: str,
    known_dataset_id: str = "esc50",
    dataset_path: str | Path = "",
    source_type: Optional[str] = None,
    learning_mode: Optional[str] = None,
    intended_use: str = "benchmark_regression",
    customer_consent_reference: str = "",
    project_id: str = "unknown",
) -> Dict[str, Any]:
    ensure_dataset_library_dirs()
    known_profiles = get_known_dataset_profiles_v114()
    known = known_profiles.get(known_dataset_id, known_profiles["customer_sample"])
    source_type = source_type or known.get("source_type", "public_benchmark")
    policies = get_dataset_source_policies_v114()
    policy = policies.get(source_type, policies["public_benchmark"])
    learning_mode = learning_mode or policy.get("default_learning_mode")

    scan = scan_dataset_folder_v114(dataset_path, known_dataset_id) if dataset_path else {
        "path_exists": False,
        "dataset_path": str(dataset_path),
        "file_count": 0,
        "total_mb_scanned": 0.0,
        "extension_counts": {},
        "metadata_file": None,
        "raw_file_count": 0,
        "inventory_warning": "no_folder_selected",
    }
    metadata = read_metadata_profile_v114(scan.get("metadata_file"), known_dataset_id)

    blockers: List[str] = []
    review_flags: List[str] = []
    score = 50

    if scan.get("path_exists"):
        score += 10
    else:
        review_flags.append("dataset_folder_not_registered_yet")

    if metadata.get("metadata_present"):
        score += 15
    else:
        review_flags.append("metadata_missing_or_unreadable")

    if not metadata.get("missing_expected_columns"):
        score += 10
    else:
        review_flags.append("metadata_missing_expected_columns")

    if metadata.get("class_count", 0) >= 5:
        score += 8
    elif metadata.get("class_count", 0) > 0:
        score += 4
    else:
        review_flags.append("low_or_unknown_label_coverage")

    if metadata.get("fold_count", 0) >= 3:
        score += 6
    elif known_dataset_id in {"esc50", "urbansound8k", "fsd50k"}:
        review_flags.append("public_benchmark_split_or_fold_not_detected")

    if metadata.get("missing_ratio", 0) <= 0.05:
        score += 4
    else:
        review_flags.append("high_missing_ratio")

    if metadata.get("duplicate_ratio", 0) > 0.05:
        review_flags.append("duplicate_metadata_rows_detected")

    if learning_mode not in policy.get("allowed_learning_modes", []):
        blockers.append("learning_mode_not_allowed_for_source_type")
        score = min(score, 70)

    if source_type.startswith("customer") and not customer_consent_reference:
        blockers.append("customer_consent_reference_required")
        score = min(score, 68)

    if source_type == "public_benchmark" and intended_use in {"paid_bundle_raw_redistribution", "commercial_training", "production_accuracy_claim"}:
        blockers.append("public_dataset_license_or_accuracy_review_required")
        score = min(score, 69)

    if intended_use == "production_accuracy_claim":
        blockers.append("dataset_import_cannot_create_production_accuracy_claim")
        score = min(score, 69)

    if blockers:
        decision = "IMPORT REGISTERED - BLOCKED FOR REQUESTED USE"
    elif score >= 90:
        decision = "DATASET LIBRARY READY"
    elif score >= 75:
        decision = "DATASET LIBRARY READY WITH REVIEW NOTES"
    else:
        decision = "DATASET IMPORT NEEDS CLEANUP"

    manifest = {
        "version": V114_VERSION,
        "created_at": _now(),
        "project_id": str(project_id),
        "dataset_name": str(dataset_name or known.get("display_name")),
        "known_dataset_id": known_dataset_id,
        "dataset_uid": _safe_id(f"{known_dataset_id}_{dataset_name}_{_sha256_text(str(dataset_path))[:10]}"),
        "source_type": source_type,
        "learning_mode": learning_mode,
        "intended_use": intended_use,
        "customer_consent_reference": customer_consent_reference,
        "known_profile": known,
        "source_policy": policy,
        "folder_inventory": scan,
        "metadata_profile": metadata,
        "score": int(max(0, min(100, score))),
        "decision": decision,
        "review_flags": review_flags,
        "blockers": blockers,
        "storage_rule": "Register path + manifest. Do not copy large raw datasets into SQLite. Keep raw public/customer files out of paid bundles unless allowed.",
        "benchmark_handoff": {
            "v112_profile": "demo/regression/pilot_evidence depending on source and consent",
            "v113_public_adapter": known_dataset_id in {"esc50", "urbansound8k", "fsd50k"},
            "v110_synthetic_real_bridge": source_type.startswith("customer") and learning_mode in {"profile_only", "synthetic_calibration", "reusable_benchmark_template"},
            "v111_synthetic_reliability": source_type == "synthetic_golden",
        },
        "recommended_next_steps": build_dataset_import_next_steps_v114(source_type, learning_mode, known_dataset_id),
    }
    return manifest


def build_dataset_import_next_steps_v114(source_type: str, learning_mode: str, known_dataset_id: str) -> List[str]:
    steps = [
        "Keep raw files in the dataset library folder; store only path/manifest/hash in EdgeTwin metadata.",
        "Run V112 Dataset Benchmark Harness after registration.",
        "Use dataset cards and license/consent notes in the Trust Ledger / Buyer Data Room.",
    ]
    if source_type == "public_benchmark":
        steps += [
            "Use public data for benchmark/demo/regression work, not customer production accuracy claims.",
            "Do not redistribute raw files in paid customer bundles without license review.",
        ]
    elif source_type == "synthetic_golden":
        steps += [
            "Version-lock the synthetic manifest and dataset hash before release/regression use.",
            "Run V111 reliability checks before treating it as a golden set.",
        ]
    elif source_type.startswith("customer"):
        steps += [
            "Confirm consent scope before any learning/reuse beyond the single customer delivery.",
            "Prefer profile-only or synthetic calibration over raw row reuse.",
            "Keep customer data separated from public/synthetic libraries.",
        ]
    if known_dataset_id == "esc50":
        steps.append("Expected ESC-50 structure: storage/dataset_library/public/esc50/meta/esc50.csv and audio/*.wav.")
    return steps


def save_dataset_manifest_v114(manifest: Dict[str, Any]) -> Dict[str, Any]:
    dirs = ensure_dataset_library_dirs()
    manifest_dir = Path(dirs["manifests"])
    uid = _safe_id(manifest.get("dataset_uid", "dataset"))
    path = manifest_dir / f"{uid}.manifest.v114.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    out = dict(manifest)
    out["manifest_path"] = str(path)
    return out


def render_dataset_import_markdown_v114(manifest: Dict[str, Any]) -> str:
    lines = [
        "# EdgeTwin V114 Dataset Library Import Manifest",
        "",
        f"Dataset: **{manifest.get('dataset_name')}**",
        f"Known profile: `{manifest.get('known_dataset_id')}`",
        f"Source type: `{manifest.get('source_type')}`",
        f"Learning mode: `{manifest.get('learning_mode')}`",
        f"Decision: **{manifest.get('decision')}**",
        f"Score: **{manifest.get('score')}%**",
        "",
        "## Folder inventory",
        f"- Path exists: {manifest.get('folder_inventory', {}).get('path_exists')}",
        f"- File count scanned: {manifest.get('folder_inventory', {}).get('file_count')}",
        f"- Raw file count: {manifest.get('folder_inventory', {}).get('raw_file_count')}",
        f"- Metadata file: {manifest.get('folder_inventory', {}).get('metadata_file')}",
        "",
        "## Metadata profile",
        f"- Rows: {manifest.get('metadata_profile', {}).get('rows')}",
        f"- Columns: {manifest.get('metadata_profile', {}).get('columns')}",
        f"- Classes: {manifest.get('metadata_profile', {}).get('class_count')}",
        f"- Folds/splits: {manifest.get('metadata_profile', {}).get('fold_count')}",
        f"- Missing expected columns: {manifest.get('metadata_profile', {}).get('missing_expected_columns')}",
        "",
        "## Review flags",
    ]
    lines += [f"- {x}" for x in manifest.get("review_flags", [])] or ["- none"]
    lines += ["", "## Blockers"]
    lines += [f"- {x}" for x in manifest.get("blockers", [])] or ["- none"]
    lines += ["", "## Next steps"]
    lines += [f"- {x}" for x in manifest.get("recommended_next_steps", [])]
    lines += ["", "## Storage rule", manifest.get("storage_rule", "")]
    return "\n".join(lines)


def _make_pdf(manifest: Dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "EdgeTwin V114 Dataset Library Import", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    lines = [
        f"Dataset: {manifest.get('dataset_name')}",
        f"Source type: {manifest.get('source_type')}",
        f"Learning mode: {manifest.get('learning_mode')}",
        f"Decision: {manifest.get('decision')}",
        f"Score: {manifest.get('score')}%",
        f"Folder: {manifest.get('folder_inventory', {}).get('dataset_path')}",
        f"Files scanned: {manifest.get('folder_inventory', {}).get('file_count')}",
        f"Metadata rows: {manifest.get('metadata_profile', {}).get('rows')}",
        "",
        "Blockers:",
    ]
    lines += [f"- {x}" for x in manifest.get("blockers", [])] or ["- none"]
    lines += ["", "Review flags:"]
    lines += [f"- {x}" for x in manifest.get("review_flags", [])] or ["- none"]
    lines += ["", "Next steps:"]
    lines += [f"- {x}" for x in manifest.get("recommended_next_steps", [])]
    for line in lines:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(180, 6, _safe_text(line)[:500])
    return bytes(pdf.output(dest="S"))


def create_dataset_import_v114_bundle(manifest: Dict[str, Any], project_name: str = "EdgeTwin_Project") -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dataset_import_manifest_v114.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        zf.writestr("DATASET_IMPORT_CARD_V114.md", render_dataset_import_markdown_v114(manifest))
        zf.writestr("dataset_import_card_v114.pdf", _make_pdf(manifest))
        zf.writestr("dataset_source_policies_v114.json", json.dumps(get_dataset_source_policies_v114(), indent=2, ensure_ascii=False))
        zf.writestr("known_dataset_profiles_v114.json", json.dumps(get_known_dataset_profiles_v114(), indent=2, ensure_ascii=False))
        summary_df = pd.DataFrame([{k: manifest.get(k) for k in ["dataset_name", "known_dataset_id", "source_type", "learning_mode", "intended_use", "score", "decision"]}])
        zf.writestr("dataset_import_summary_v114.csv", summary_df.to_csv(index=False))
        zf.writestr("README_V114.txt", "Place datasets in storage/dataset_library/{public,synthetic,customer_consent}. Register via manifest. Do not copy raw datasets into SQLite or paid bundles unless policy allows it.\n")
    return out.getvalue()
