"""EdgeTwin Studio V113 public benchmark dataset adapter.

Purpose:
- Let EdgeTwin use public/open datasets such as ESC-50 as benchmark/demo/reference inputs.
- Keep public benchmark use separate from customer-data validation and production accuracy claims.
- Capture license/commercial-use risk before a public dataset is used in a paid pack or delivery bundle.
- Convert dataset metadata into an EdgeTwin-compatible benchmark card, mapping template and readiness decision.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fpdf import FPDF

V113_VERSION = "113.0"


@dataclass(frozen=True)
class PublicBenchmarkSpec:
    dataset_id: str
    display_name: str
    domain: str
    primary_use: str
    expected_metadata_columns: List[str]
    split_strategy: str
    license_note: str
    commercial_guard: str
    edgetwin_value: List[str]
    not_sufficient_for: List[str]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any) -> str:
    return str(value or "").replace("\u2192", "->").replace("\u2013", "-").replace("\u2014", "-").replace("_", " ")


def get_public_benchmark_catalog_v113() -> Dict[str, Dict[str, Any]]:
    """Return public dataset specs known to the adapter.

    This is deliberately conservative: it does not download or redistribute any dataset.
    It only creates a dataset card and validation contract for datasets the user already has rights to use.
    """
    specs = [
        PublicBenchmarkSpec(
            dataset_id="esc50",
            display_name="ESC-50 Environmental Sound Classification",
            domain="audio_event_classification",
            primary_use="Audio feature extraction, sound-event classification benchmark and acoustic pipeline regression.",
            expected_metadata_columns=["filename", "fold", "target", "category", "esc10", "src_file", "take"],
            split_strategy="Use the official 5 folds. Do not mix fragments from the same source across train/test.",
            license_note="ESC-50 is available under a Creative Commons Attribution Non-Commercial license; ESC-10 subset is CC BY according to the official README.",
            commercial_guard="Do not redistribute full ESC-50 audio in paid customer bundles unless the license and use are reviewed. Prefer metadata cards, feature summaries, internal benchmarks and links/citations.",
            edgetwin_value=[
                "benchmark audio preprocessing and feature extraction",
                "validate fold-aware classification workflows",
                "stress-test sound-event labels before customer audio arrives",
                "seed synthetic audio-event scenario profiles",
            ],
            not_sufficient_for=[
                "customer-specific machinery accuracy",
                "production detection guarantee",
                "legal/compliance certification",
                "commercial redistribution of non-commercial audio",
            ],
        ),
        PublicBenchmarkSpec(
            dataset_id="urbansound8k",
            display_name="UrbanSound8K",
            domain="urban_audio_event_classification",
            primary_use="Urban sound benchmark for siren, drilling, engine and street-noise style events.",
            expected_metadata_columns=["slice_file_name", "fsID", "start", "end", "salience", "fold", "classID", "class"],
            split_strategy="Use pre-sorted folds and avoid leakage across related Freesound sources.",
            license_note="License can vary by source; verify before commercial redistribution.",
            commercial_guard="Use for internal benchmarking unless license review confirms allowed commercial use.",
            edgetwin_value=[
                "urban/security sound-event reference",
                "audio robustness tests with variable sampling/audio conditions",
                "siren/engine/drilling scenario inspiration",
            ],
            not_sufficient_for=[
                "industrial vibration accuracy",
                "customer-specific site reliability",
                "safety guarantee",
            ],
        ),
        PublicBenchmarkSpec(
            dataset_id="fsd50k",
            display_name="FSD50K Freesound Dataset 50k",
            domain="large_audio_event_classification",
            primary_use="Large multi-label sound-event benchmark and label taxonomy reference.",
            expected_metadata_columns=["fname", "labels", "mids", "split"],
            split_strategy="Use published development/evaluation splits and preserve multi-label semantics.",
            license_note="FSD50K is released under Creative Commons licenses that can differ per clip.",
            commercial_guard="Per-clip license review is required before redistribution or paid bundle inclusion.",
            edgetwin_value=[
                "larger sound taxonomy reference",
                "multi-label audio event stress tests",
                "scenario inspiration for synthetic audio events",
            ],
            not_sufficient_for=[
                "single-site field validation",
                "commercial redistribution without per-clip license review",
                "production accuracy guarantee",
            ],
        ),
    ]
    return {s.dataset_id: asdict(s) for s in specs}


def _metadata_profile(meta_df: Optional[pd.DataFrame], dataset_id: str) -> Dict[str, Any]:
    if meta_df is None or not isinstance(meta_df, pd.DataFrame) or meta_df.empty:
        return {
            "metadata_present": False,
            "rows": 0,
            "columns": [],
            "missing_expected_columns": [],
            "fold_count": 0,
            "class_count": 0,
            "balance_warning": "metadata_not_uploaded",
        }

    catalog = get_public_benchmark_catalog_v113()
    expected = set(catalog.get(dataset_id, {}).get("expected_metadata_columns", []))
    cols = list(meta_df.columns)
    missing = sorted(expected.difference(set(cols))) if expected else []

    fold_col = "fold" if "fold" in meta_df.columns else None
    category_col = None
    for candidate in ["category", "class", "labels", "target", "classID"]:
        if candidate in meta_df.columns:
            category_col = candidate
            break

    class_count = int(meta_df[category_col].nunique()) if category_col else 0
    fold_count = int(meta_df[fold_col].nunique()) if fold_col else 0

    balance_warning = "unknown"
    if category_col:
        counts = meta_df[category_col].value_counts()
        if not counts.empty:
            ratio = float(counts.max() / max(counts.min(), 1))
            balance_warning = "balanced_enough" if ratio <= 2.0 else "imbalanced_classes"

    return {
        "metadata_present": True,
        "rows": int(len(meta_df)),
        "columns": cols,
        "missing_expected_columns": missing,
        "fold_count": fold_count,
        "class_count": class_count,
        "class_column_used": category_col,
        "fold_column_used": fold_col,
        "balance_warning": balance_warning,
    }


def _score_public_benchmark(spec: Dict[str, Any], profile: Dict[str, Any], intended_use: str) -> Dict[str, Any]:
    score = 55
    blockers: List[str] = []
    review_flags: List[str] = []

    if profile.get("metadata_present"):
        score += 15
    else:
        review_flags.append("metadata_not_uploaded_use_catalog_card_only")

    if not profile.get("missing_expected_columns"):
        score += 10
    else:
        review_flags.append("metadata_missing_expected_columns")

    if profile.get("fold_count", 0) >= 5:
        score += 8
    elif profile.get("fold_count", 0) > 0:
        score += 4
    else:
        review_flags.append("no_fold_strategy_detected")

    if profile.get("class_count", 0) >= 10:
        score += 7
    elif profile.get("class_count", 0) > 0:
        score += 3

    if profile.get("balance_warning") == "balanced_enough":
        score += 5
    elif profile.get("balance_warning") == "imbalanced_classes":
        review_flags.append("imbalanced_classes")

    license_note = spec.get("license_note", "")
    commercial_guard = spec.get("commercial_guard", "")

    if intended_use in {"paid_customer_bundle", "commercial_model_training", "redistribution"}:
        if "Non-Commercial" in license_note or "per-clip" in commercial_guard or "license review" in commercial_guard:
            blockers.append("license_review_required_before_commercial_use_or_redistribution")
            score = min(score, 74)

    if intended_use == "production_accuracy_claim":
        blockers.append("public_benchmark_dataset_cannot_support_customer_production_accuracy_claim")
        score = min(score, 69)

    if blockers:
        decision = "REVIEW_OR_BLOCKED_FOR_COMMERCIAL_USE"
    elif score >= 90:
        decision = "PUBLIC BENCHMARK READY"
    elif score >= 75:
        decision = "PUBLIC BENCHMARK READY WITH NOTES"
    else:
        decision = "CATALOG CARD ONLY - NEED METADATA/LICENSE REVIEW"

    return {
        "public_benchmark_score": int(max(0, min(100, score))),
        "decision": decision,
        "review_flags": review_flags,
        "blockers": blockers,
        "safe_use": [
            "internal benchmark",
            "demo pipeline validation",
            "feature extraction regression",
            "synthetic scenario inspiration",
        ],
        "unsafe_without_review": [
            "paid redistribution of raw audio",
            "customer production accuracy claims",
            "legal/compliance certification claims",
        ],
    }


def build_public_benchmark_v113_snapshot(
    dataset_id: str = "esc50",
    metadata_df: Optional[pd.DataFrame] = None,
    intended_use: str = "internal_benchmark",
    project_id: str = "unknown",
) -> Dict[str, Any]:
    catalog = get_public_benchmark_catalog_v113()
    if dataset_id not in catalog:
        dataset_id = "esc50"
    spec = catalog[dataset_id]
    profile = _metadata_profile(metadata_df, dataset_id)
    scoring = _score_public_benchmark(spec, profile, intended_use)

    mapping_template = {
        "audio_file_column": "filename" if dataset_id == "esc50" else "slice_file_name_or_fname",
        "label_column": "category_or_class_or_labels",
        "fold_column": "fold_or_split",
        "source_column": "src_file_or_fsID_optional",
        "edgetwin_features": ["rms", "zcr", "spectral_centroid", "band_energy", "mfcc_summary"],
        "recommended_benchmark_profiles": ["demo", "regression", "public_audio_reference"],
    }

    return {
        "version": V113_VERSION,
        "created_at": _now(),
        "project_id": str(project_id),
        "dataset_id": dataset_id,
        "intended_use": intended_use,
        "spec": spec,
        "metadata_profile": profile,
        "mapping_template": mapping_template,
        "scoring": scoring,
        "edge_twin_positioning": "Public datasets are benchmark/demo/reference fuel. They improve pipeline confidence but do not replace customer data validation.",
        "next_steps": [
            "keep public benchmark data separate from customer data",
            "store license notes in the buyer data room / trust ledger",
            "use folds/splits exactly as provided",
            "use results for regression and demo confidence, not production guarantees",
            "use customer data with consent for real-data evidence packs",
        ],
    }


def _make_pdf(snapshot: Dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "EdgeTwin V113 Public Benchmark Dataset Card", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    lines = [
        f"Dataset: {snapshot.get('spec', {}).get('display_name')}",
        f"Decision: {snapshot.get('scoring', {}).get('decision')}",
        f"Score: {snapshot.get('scoring', {}).get('public_benchmark_score')}%",
        f"Intended use: {snapshot.get('intended_use')}",
        f"Rows in metadata: {snapshot.get('metadata_profile', {}).get('rows')}",
        f"Classes: {snapshot.get('metadata_profile', {}).get('class_count')}",
        f"Folds/splits: {snapshot.get('metadata_profile', {}).get('fold_count')}",
        "",
        "Safe use:",
    ]
    for item in snapshot.get("scoring", {}).get("safe_use", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Unsafe without review:")
    for item in snapshot.get("scoring", {}).get("unsafe_without_review", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("License note:")
    lines.append(snapshot.get("spec", {}).get("license_note", ""))
    lines.append("")
    lines.append("Positioning:")
    lines.append(snapshot.get("edge_twin_positioning", ""))

    for line in lines:
        pdf.set_x(pdf.l_margin)
        txt = _safe_text(line)[:500]
        pdf.multi_cell(180, 6, txt)
    return bytes(pdf.output(dest="S"))


def create_public_benchmark_v113_bundle(snapshot: Dict[str, Any], project_name: str = "EdgeTwin_Project") -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("public_benchmark_snapshot_v113.json", json.dumps(snapshot, indent=2, ensure_ascii=False))
        zf.writestr("PUBLIC_BENCHMARK_CARD_V113.md", render_public_benchmark_card_markdown(snapshot))
        zf.writestr("public_benchmark_dataset_card_v113.pdf", _make_pdf(snapshot))
        mapping_df = pd.DataFrame([snapshot.get("mapping_template", {})])
        zf.writestr("edgetwin_public_benchmark_mapping_template_v113.csv", mapping_df.to_csv(index=False))
        scoring_df = pd.DataFrame([snapshot.get("scoring", {})])
        zf.writestr("public_benchmark_scoring_v113.csv", scoring_df.to_csv(index=False))
        zf.writestr("README_V113.txt", "Public benchmark datasets are for benchmark/demo/reference use. Do not treat them as customer production accuracy proof.\n")
    return out.getvalue()


def render_public_benchmark_card_markdown(snapshot: Dict[str, Any]) -> str:
    spec = snapshot.get("spec", {})
    scoring = snapshot.get("scoring", {})
    profile = snapshot.get("metadata_profile", {})
    lines = [
        "# EdgeTwin V113 Public Benchmark Dataset Card",
        "",
        f"Dataset: **{spec.get('display_name')}**",
        f"Domain: `{spec.get('domain')}`",
        f"Decision: **{scoring.get('decision')}**",
        f"Score: **{scoring.get('public_benchmark_score')}%**",
        f"Intended use: `{snapshot.get('intended_use')}`",
        "",
        "## Metadata profile",
        f"- Rows: {profile.get('rows')}",
        f"- Classes: {profile.get('class_count')}",
        f"- Folds/splits: {profile.get('fold_count')}",
        f"- Missing expected columns: {profile.get('missing_expected_columns')}",
        "",
        "## Safe use",
    ]
    lines += [f"- {x}" for x in scoring.get("safe_use", [])]
    lines += ["", "## Unsafe without review"]
    lines += [f"- {x}" for x in scoring.get("unsafe_without_review", [])]
    lines += ["", "## License/commercial guard", spec.get("commercial_guard", "")]
    lines += ["", "## EdgeTwin positioning", snapshot.get("edge_twin_positioning", "")]
    return "\n".join(lines)
