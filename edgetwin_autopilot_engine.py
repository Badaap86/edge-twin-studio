"""
EdgeTwin v152 Autopilot Approval Engine

Scope:
- Industrial data-readiness and evidence-strength assessment.
- Customer-safe synthetic scenario support.
- Founder approval: Green / Yellow / Red.
- Claim guardrails to prevent unsafe commercial or technical promises.

This module does NOT certify safety, compliance, cybersecurity, CE, CRA, DPP or machine condition.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional
import json
import re
import zipfile

import numpy as np
import pandas as pd


PROHIBITED_CLAIMS = [
    "guaranteed failure prediction",
    "guaranteed predictive maintenance",
    "machine will fail",
    "machine is safe",
    "certified compliant",
    "legal compliance guaranteed",
    "ce certified",
    "cra certified",
    "dpp compliant guaranteed",
    "cybersecurity certified",
    "100% accurate",
    "maintenance can be skipped",
    "maintenance can be postponed safely",
    "root cause is certain",
    "no further validation needed",
]

SAFE_CLAIMS = [
    "EdgeTwin assesses whether the provided data is strong enough for a first AI, IoT or predictive-maintenance readiness decision.",
    "EdgeTwin identifies missing evidence, data-quality gaps, unclear context and practical next steps.",
    "EdgeTwin supports Go / Conditional Go / No-Go pilot decision-making based on the information provided.",
    "EdgeTwin does not replace a maintenance engineer, safety auditor, cybersecurity company, notified body, legal advisor or formal certification process.",
]

STANDARD_DISCLAIMER = (
    "Readiness and evidence-gap assessment only. EdgeTwin does not provide legal advice, financial advice, "
    "cybersecurity certification, CE approval, CRA/DPP certification, safety certification, penetration testing, "
    "machine diagnosis or guaranteed predictive-maintenance performance. Final operational, legal, safety and "
    "technical decisions remain the responsibility of the customer and qualified specialists."
)


@dataclass
class DataProfile:
    rows: int
    columns: int
    numeric_columns: List[str]
    text_columns: List[str]
    timestamp_column: Optional[str]
    possible_label_columns: List[str]
    missing_percent: float
    duplicate_rows: int
    time_span_days: Optional[float]
    suspected_sampling_gaps: int
    constant_numeric_columns: List[str]
    high_missing_columns: List[str]
    suspicious_numeric_columns: List[str]


def _score_rows(rows: int) -> int:
    if rows >= 100000:
        return 100
    if rows >= 25000:
        return 90
    if rows >= 5000:
        return 75
    if rows >= 1000:
        return 60
    if rows >= 250:
        return 40
    if rows >= 50:
        return 25
    if rows > 0:
        return 10
    return 0


def _score_missing(missing_percent: float) -> int:
    if missing_percent <= 1:
        return 100
    if missing_percent <= 5:
        return 85
    if missing_percent <= 10:
        return 70
    if missing_percent <= 20:
        return 50
    if missing_percent <= 35:
        return 30
    return 10


def _detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    preferred = ["timestamp", "datetime", "date", "time", "ts"]
    for col in df.columns:
        name = str(col).strip().lower()
        if any(term == name or term in name for term in preferred):
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            if parsed.notna().mean() >= 0.60:
                return str(col)
    for col in df.columns:
        if df[col].dtype == object:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            if parsed.notna().mean() >= 0.75:
                return str(col)
    return None


def _detect_label_columns(df: pd.DataFrame) -> List[str]:
    terms = ["label", "status", "state", "fault", "failure", "alarm", "event", "class", "condition"]
    return [str(c) for c in df.columns if any(t in str(c).lower() for t in terms)]


def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    if df is None or df.empty:
        return DataProfile(
            rows=0,
            columns=0,
            numeric_columns=[],
            text_columns=[],
            timestamp_column=None,
            possible_label_columns=[],
            missing_percent=100.0,
            duplicate_rows=0,
            time_span_days=None,
            suspected_sampling_gaps=0,
            constant_numeric_columns=[],
            high_missing_columns=[],
            suspicious_numeric_columns=[],
        )

    rows, cols = df.shape
    numeric_columns = [str(c) for c in df.select_dtypes(include=[np.number]).columns]
    text_columns = [str(c) for c in df.select_dtypes(exclude=[np.number]).columns]
    timestamp_column = _detect_timestamp_column(df)
    possible_label_columns = _detect_label_columns(df)
    missing_percent = round(float(df.isna().mean().mean() * 100), 2) if rows and cols else 100.0
    duplicate_rows = int(df.duplicated().sum()) if rows else 0

    time_span_days = None
    suspected_sampling_gaps = 0
    if timestamp_column:
        times = pd.to_datetime(df[timestamp_column], errors="coerce", utc=True).dropna().sort_values()
        if len(times) >= 2:
            time_span_days = round(float((times.iloc[-1] - times.iloc[0]).total_seconds() / 86400), 3)
            diffs = times.diff().dropna().dt.total_seconds()
            if len(diffs) >= 10:
                median_gap = float(diffs.median())
                if median_gap > 0:
                    suspected_sampling_gaps = int((diffs > median_gap * 10).sum())

    constant_numeric_columns = []
    suspicious_numeric_columns = []
    for col in numeric_columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            suspicious_numeric_columns.append(col)
            continue
        if s.nunique() <= 1:
            constant_numeric_columns.append(col)
        if len(s) >= 10:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outlier_ratio = ((s < q1 - 6 * iqr) | (s > q3 + 6 * iqr)).mean()
                if outlier_ratio > 0.05:
                    suspicious_numeric_columns.append(col)

    high_missing_columns = [str(c) for c in df.columns if float(df[c].isna().mean()) >= 0.25]

    return DataProfile(
        rows=int(rows),
        columns=int(cols),
        numeric_columns=numeric_columns,
        text_columns=text_columns,
        timestamp_column=timestamp_column,
        possible_label_columns=possible_label_columns,
        missing_percent=missing_percent,
        duplicate_rows=duplicate_rows,
        time_span_days=time_span_days,
        suspected_sampling_gaps=suspected_sampling_gaps,
        constant_numeric_columns=constant_numeric_columns,
        high_missing_columns=high_missing_columns,
        suspicious_numeric_columns=suspicious_numeric_columns,
    )


def analyze_readiness(df: pd.DataFrame, context: Dict[str, Any], maintenance_log: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    profile = profile_dataframe(df)
    maintenance_profile = profile_dataframe(maintenance_log) if maintenance_log is not None and not maintenance_log.empty else None

    rows_score = _score_rows(profile.rows)
    completeness_score = _score_missing(profile.missing_percent)
    timestamp_score = 100 if profile.timestamp_column else 25 if profile.rows else 0
    numeric_signal_score = min(100, len(profile.numeric_columns) * 18) if profile.rows else 0
    label_score = 90 if profile.possible_label_columns else 35 if profile.rows >= 1000 else 15

    context_score = 0
    if context.get("machine_type"):
        context_score += 15
    if context.get("business_problem"):
        context_score += 15
    if context.get("decision_goal"):
        context_score += 15
    if context.get("downtime_cost"):
        context_score += 10
    if context.get("data_period"):
        context_score += 10
    if context.get("has_failure_history"):
        context_score += 15
    if maintenance_profile:
        context_score += 20
    context_score = min(100, context_score)

    penalty = 0
    if profile.suspected_sampling_gaps:
        penalty += min(15, profile.suspected_sampling_gaps * 2)
    if profile.duplicate_rows > max(10, profile.rows * 0.03):
        penalty += 10
    if profile.high_missing_columns:
        penalty += min(15, len(profile.high_missing_columns) * 3)
    if profile.constant_numeric_columns:
        penalty += min(10, len(profile.constant_numeric_columns) * 2)

    data_quality_score = int(max(0, round(rows_score * 0.30 + completeness_score * 0.30 + timestamp_score * 0.20 + numeric_signal_score * 0.20 - penalty)))
    maintenance_context_score = int(round(context_score * 0.70 + label_score * 0.30))
    evidence_strength_score = int(round(data_quality_score * 0.55 + maintenance_context_score * 0.45))

    if evidence_strength_score >= 78 and label_score >= 70 and timestamp_score >= 80:
        decision = "GO"
        decision_label = "Go for a controlled pilot-readiness next step"
        risk_level = "Low"
    elif evidence_strength_score >= 55:
        decision = "CONDITIONAL_GO"
        decision_label = "Conditional Go: useful for readiness and trend exploration, not for production claims"
        risk_level = "Medium"
    else:
        decision = "NO_GO"
        decision_label = "No-Go / Improve-first: collect better evidence before investing in AI or predictive maintenance"
        risk_level = "High"

    missing_evidence = []
    if profile.rows < 1000:
        missing_evidence.append("More historical rows/samples are needed before strong readiness conclusions can be made.")
    if not profile.timestamp_column:
        missing_evidence.append("A reliable timestamp/date column is missing or not parseable.")
    if not profile.possible_label_columns:
        missing_evidence.append("Failure/event/condition labels are missing or not clearly named.")
    if not maintenance_profile:
        missing_evidence.append("Maintenance logs, repair history or event notes are missing.")
    if not context.get("machine_type"):
        missing_evidence.append("Machine/process type is not clearly described.")
    if not context.get("business_problem"):
        missing_evidence.append("The business problem or decision is not clearly defined.")
    if profile.high_missing_columns:
        missing_evidence.append("High missing-data columns detected: " + ", ".join(profile.high_missing_columns[:8]))
    if profile.suspected_sampling_gaps:
        missing_evidence.append(f"Possible time-series sampling gaps detected: {profile.suspected_sampling_gaps}.")
    if profile.constant_numeric_columns:
        missing_evidence.append("Constant numeric columns may not be useful: " + ", ".join(profile.constant_numeric_columns[:8]))

    allowed_conclusion = {
        "GO": "The data appears suitable for a controlled readiness/pilot planning step, with human validation before production use.",
        "CONDITIONAL_GO": "The data can support an exploratory readiness report, but missing evidence must stay visible.",
        "NO_GO": "The data is not strong enough for responsible predictive-maintenance claims; improve evidence first.",
    }[decision]

    if decision == "GO":
        safe_next_steps = [
            "Run a fixed-scope pilot-readiness pack for one machine, line or use-case.",
            "Validate assumptions with a maintenance/reliability specialist before production deployment.",
            "Keep safe-claim language: readiness and evidence strength, not guaranteed failure prediction.",
        ]
    elif decision == "CONDITIONAL_GO":
        safe_next_steps = [
            "Deliver a readiness report with warnings and missing-evidence list.",
            "Request missing labels/logs/context before recommending a larger pilot.",
            "Use synthetic scenarios only for exploration, demonstration or test planning.",
        ]
    else:
        safe_next_steps = [
            "Do not sell this as predictive maintenance yet.",
            "Ask the customer to collect timestamps, labels, maintenance logs and baseline data.",
            "Offer an Improve-First Quick Scan or data collection plan instead of a premium pilot.",
        ]

    snapshot = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "context": context,
        "data_profile": asdict(profile),
        "maintenance_profile": asdict(maintenance_profile) if maintenance_profile else None,
        "scores": {
            "rows_score": rows_score,
            "completeness_score": completeness_score,
            "timestamp_score": timestamp_score,
            "numeric_signal_score": numeric_signal_score,
            "label_score": label_score,
            "context_score": context_score,
            "data_quality_score": data_quality_score,
            "maintenance_context_score": maintenance_context_score,
            "evidence_strength_score": evidence_strength_score,
        },
        "decision": decision,
        "decision_label": decision_label,
        "risk_level": risk_level,
        "allowed_conclusion": allowed_conclusion,
        "missing_evidence": missing_evidence,
        "safe_next_steps": safe_next_steps,
        "safe_claims": SAFE_CLAIMS,
        "claims_to_avoid": PROHIBITED_CLAIMS,
        "disclaimer": STANDARD_DISCLAIMER,
    }
    snapshot["approval"] = founder_approval(snapshot)
    return snapshot


def founder_approval(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    scores = snapshot.get("scores", {})
    decision = snapshot.get("decision", "NO_GO")
    missing = snapshot.get("missing_evidence", [])
    profile = snapshot.get("data_profile", {})
    risk = snapshot.get("risk_level", "High")

    blockers = []
    warnings = []

    if profile.get("rows", 0) == 0:
        blockers.append("No dataset available.")
    if scores.get("evidence_strength_score", 0) < 40:
        blockers.append("Evidence strength too low for a paid pilot-readiness recommendation.")
    if not profile.get("timestamp_column"):
        warnings.append("Timestamp evidence is weak or missing.")
    if not profile.get("possible_label_columns"):
        warnings.append("Failure/event labels are missing.")
    if len(missing) >= 5:
        warnings.append("Many evidence gaps detected; keep report conservative.")
    if decision == "NO_GO":
        warnings.append("Only deliver an Improve-First or data collection plan.")

    if blockers:
        status = "RED"
        safe_to_deliver = "No"
        founder_action = "Block delivery or request more data."
    elif risk == "Medium" or warnings:
        status = "YELLOW"
        safe_to_deliver = "Conditional"
        founder_action = "Deliver only with warnings, missing-evidence list and conservative language."
    else:
        status = "GREEN"
        safe_to_deliver = "Yes"
        founder_action = "Safe to deliver as readiness/evidence report, not as certification."

    return {
        "status": status,
        "safe_to_deliver": safe_to_deliver,
        "founder_action": founder_action,
        "blockers": blockers,
        "warnings": warnings,
        "allowed_conclusion": snapshot.get("allowed_conclusion", ""),
        "blocked_claims": PROHIBITED_CLAIMS,
    }


def scan_claims(text: str) -> Dict[str, Any]:
    text_l = (text or "").lower()
    hits = []
    for claim in PROHIBITED_CLAIMS:
        pattern = re.escape(claim.lower()).replace("\\ ", r"\s+")
        if re.search(pattern, text_l):
            hits.append(claim)
    return {
        "safe": len(hits) == 0,
        "blocked_claims_found": hits,
        "recommendation": "OK" if not hits else "Rewrite with readiness/evidence wording only.",
    }


def generate_synthetic_scenarios(df: pd.DataFrame, n_rows: int = 300, scenario_strength: float = 0.18, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    if df is None or df.empty:
        base = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="min"),
            "vibration_rms": rng.normal(1.0, 0.08, n_rows).clip(0),
            "temperature_c": rng.normal(42.0, 1.5, n_rows),
            "current_a": rng.normal(8.0, 0.5, n_rows),
        })
        drift = np.linspace(0, scenario_strength, n_rows)
        base["synthetic_vibration_rms_stress"] = base["vibration_rms"] * (1 + drift)
        base["synthetic_temperature_c_stress"] = base["temperature_c"] + drift * 12
        base["scenario_label"] = np.where(np.arange(n_rows) > n_rows * 0.7, "Synthetic_Stress_Example", "Synthetic_Baseline_Example")
        return base

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    timestamp_col = _detect_timestamp_column(df)
    if not numeric_cols:
        return generate_synthetic_scenarios(pd.DataFrame(), n_rows=n_rows, scenario_strength=scenario_strength, random_state=random_state)

    output = pd.DataFrame()
    if timestamp_col:
        times = pd.to_datetime(df[timestamp_col], errors="coerce").dropna()
        start = times.min() if len(times) else pd.Timestamp("2026-01-01")
        output["timestamp"] = pd.date_range(start=start, periods=n_rows, freq="min")
    else:
        output["timestamp"] = pd.date_range("2026-01-01", periods=n_rows, freq="min")

    for col in numeric_cols[:12]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        mu = float(s.mean())
        sd = float(s.std()) if float(s.std()) > 0 else max(abs(mu) * 0.05, 0.01)
        sampled = rng.normal(mu, sd, n_rows)
        drift = np.linspace(0, scenario_strength, n_rows)
        output[f"{col}_synthetic_baseline"] = sampled
        output[f"{col}_synthetic_stress"] = sampled * (1 + drift)

    output["scenario_label"] = np.where(np.arange(n_rows) > n_rows * 0.7, "Synthetic_Stress_Example", "Synthetic_Baseline_Example")
    return output


def build_markdown_report(snapshot: Dict[str, Any]) -> str:
    context = snapshot.get("context", {})
    scores = snapshot.get("scores", {})
    profile = snapshot.get("data_profile", {})
    approval = snapshot.get("approval", {})

    lines = [
        "# EdgeTwin Industrial Data Readiness Evidence Pack",
        "",
        f"Generated: {snapshot.get('generated_at', '')}",
        "",
        "## Executive Summary",
        f"Project / use-case: {context.get('project_name', 'Untitled project')}",
        f"Machine / process type: {context.get('machine_type', 'Not provided')}",
        f"Decision goal: {context.get('decision_goal', 'Not provided')}",
        f"Recommendation: {snapshot.get('decision_label', 'Unknown')}",
        f"Risk level: {snapshot.get('risk_level', 'Unknown')}",
        f"Founder approval status: {approval.get('status', 'Unknown')} — safe to deliver: {approval.get('safe_to_deliver', 'Unknown')}",
        "",
        "## Readiness Scores",
        f"- Data quality score: {scores.get('data_quality_score', 0)}%",
        f"- Maintenance/context score: {scores.get('maintenance_context_score', 0)}%",
        f"- Evidence strength score: {scores.get('evidence_strength_score', 0)}%",
        "",
        "## Data Inventory",
        f"- Rows: {profile.get('rows', 0)}",
        f"- Columns: {profile.get('columns', 0)}",
        f"- Numeric columns: {', '.join(profile.get('numeric_columns', [])[:12]) or 'None detected'}",
        f"- Timestamp column: {profile.get('timestamp_column') or 'Not detected'}",
        f"- Possible label columns: {', '.join(profile.get('possible_label_columns', [])) or 'Not detected'}",
        f"- Missing data: {profile.get('missing_percent', 100)}%",
        f"- Possible sampling gaps: {profile.get('suspected_sampling_gaps', 0)}",
        "",
        "## Allowed Conclusion",
        snapshot.get("allowed_conclusion", ""),
        "",
        "## Missing Evidence / Gaps",
    ]

    gaps = snapshot.get("missing_evidence", [])
    lines.extend([f"- {g}" for g in gaps] if gaps else ["- No major evidence gaps detected in the quickscan scope."])

    lines += ["", "## Recommended Next Steps"]
    lines.extend([f"- {s}" for s in snapshot.get("safe_next_steps", [])])

    lines += ["", "## Safe Customer-Facing Claims"]
    lines.extend([f"- {s}" for s in snapshot.get("safe_claims", [])])

    lines += ["", "## Claims To Avoid"]
    lines.extend([f"- {s}" for s in snapshot.get("claims_to_avoid", [])])

    if approval.get("warnings"):
        lines += ["", "## Founder Review Warnings"]
        lines.extend([f"- {w}" for w in approval.get("warnings", [])])

    if approval.get("blockers"):
        lines += ["", "## Founder Review Blockers"]
        lines.extend([f"- {b}" for b in approval.get("blockers", [])])

    lines += ["", "## Disclaimer", snapshot.get("disclaimer", STANDARD_DISCLAIMER)]
    return "\n".join(lines)


def build_bundle_zip(snapshot: Dict[str, Any], original_df: Optional[pd.DataFrame] = None, synthetic_df: Optional[pd.DataFrame] = None) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("EdgeTwin_Evidence_Pack_Report.md", build_markdown_report(snapshot))
        z.writestr("edgetwin_snapshot.json", json.dumps(snapshot, indent=2, ensure_ascii=False))
        z.writestr("README_SCOPE.txt", STANDARD_DISCLAIMER)
        if original_df is not None and not original_df.empty:
            z.writestr("data/original_input_preview.csv", original_df.head(5000).to_csv(index=False))
        if synthetic_df is not None and not synthetic_df.empty:
            z.writestr("data/customer_safe_synthetic_scenarios.csv", synthetic_df.to_csv(index=False))
    return buffer.getvalue()
