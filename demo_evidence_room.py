"""EdgeTwin Studio Demo Evidence Room.

Purpose:
- Show customers what an evidence pack looks like before they buy.
- Use demo/synthetic example content only.
- Make the output management-ready: scorecard, assumptions, limitations, claims and next steps.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import zipfile
from typing import Any, Dict, List

import pandas as pd

MODULE = "Demo Evidence Room"

DEMO_BOUNDARY = (
    "This is a demonstration evidence room using example/synthetic content. It shows the delivery format, "
    "not customer-specific production performance. Customer-specific evidence requires customer-specific data."
)

SAFE_CLAIMS = [
    "EdgeTwin can assess whether the provided data appears ready for a controlled AI / predictive-maintenance pilot.",
    "EdgeTwin can document data quality, assumptions, limitations, risks and recommended next steps.",
    "EdgeTwin can use synthetic scenarios for coverage and stress testing when clearly separated from real-data evidence.",
]

BLOCKED_CLAIMS = [
    "Production accuracy is guaranteed.",
    "The system is safety certified or compliance certified.",
    "AI replaces human engineering review or maintenance decision-making.",
    "Synthetic data proves customer-specific production performance.",
]


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def build_demo_scorecard(use_case: str = "Bearing wear / lager-slijtage", data_situation: str = "Limited real data") -> pd.DataFrame:
    rows = [
        ("Signal availability", 70, "Vibration/audio available; RPM/load optional but recommended."),
        ("Timestamp quality", 82, "Example timestamps are consistent enough for pilot-readiness demonstration."),
        ("Asset linkage", 75, "Machine/component linkage exists but should be verified by customer."),
        ("Label coverage", 35, "Failure/wear labels are weak; claims must remain limited."),
        ("Operating context", 45, "Load/RPM/shift context is incomplete."),
        ("Privacy/minimisation", 90, "Demo pack uses no personal data."),
        ("Synthetic scenario usefulness", 78, "Useful for stress testing gaps, not proof."),
    ]
    return pd.DataFrame(rows, columns=["Evidence area", "Score", "Finding"])


def build_demo_decision(scorecard: pd.DataFrame) -> Dict[str, Any]:
    avg_score = int(scorecard["Score"].mean())
    if avg_score >= 78:
        decision = "Go for controlled real-data evidence review"
    elif avg_score >= 50:
        decision = "Conditional Go"
    else:
        decision = "No-Go / collect better data first"
    return {
        "overall_score": avg_score,
        "decision": decision,
        "decision_reason": (
            "This demo example is strong enough to show the EdgeTwin workflow, but labels/context remain the limiting factors."
        ),
    }


def build_demo_evidence_room(
    customer: str = "Demo customer",
    use_case: str = "Bearing wear / lager-slijtage",
    data_situation: str = "Limited real data",
    recommended_route: str = "Real Data + Synthetic Evidence Expansion",
) -> Dict[str, Any]:
    scorecard = build_demo_scorecard(use_case, data_situation)
    decision = build_demo_decision(scorecard)
    evidence = {
        "module": MODULE,
        "created_at": _now(),
        "customer": customer,
        "use_case": use_case,
        "data_situation": data_situation,
        "recommended_route": recommended_route,
        "decision": decision,
        "scorecard": scorecard.to_dict(orient="records"),
        "assumptions": [
            "Demo content is synthetic/example content and not customer-specific proof.",
            "Real customer data must be checked before operational pilot claims.",
            "Labels, operating context and maintenance records increase evidence strength.",
        ],
        "limitations": [
            "Weak labels limit failure-type claims.",
            "Missing RPM/load context can make normal operating variation look anomalous.",
            "Synthetic scenarios are useful for stress testing but cannot replace real validation.",
        ],
        "safe_claims": SAFE_CLAIMS,
        "blocked_claims": BLOCKED_CLAIMS,
        "recommended_next_steps": [
            "Run a Readiness Snapshot for the real customer situation.",
            "If real data exists, run Data Quality Gate and prepare a Professional Pilot Pack.",
            "If real data is limited, separate real-data evidence from synthetic scenario expansion.",
            "If no data exists, start with Hardware Profile + Firmware Starter or Field Data Kit.",
        ],
        "demo_boundary": DEMO_BOUNDARY,
    }
    evidence["evidence_hash"] = _sha(evidence)
    return evidence


def evidence_to_tables(evidence: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    return {
        "scorecard": pd.DataFrame(evidence.get("scorecard", [])),
        "safe_claims": pd.DataFrame({"Safe claim": evidence.get("safe_claims", [])}),
        "blocked_claims": pd.DataFrame({"Blocked claim": evidence.get("blocked_claims", [])}),
        "limitations": pd.DataFrame({"Limitation": evidence.get("limitations", [])}),
        "next_steps": pd.DataFrame({"Next step": evidence.get("recommended_next_steps", [])}),
    }


def create_demo_evidence_bundle(evidence: Dict[str, Any]) -> bytes:
    tables = evidence_to_tables(evidence)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("demo_evidence_room.json", json.dumps(evidence, indent=2, ensure_ascii=False))
        for name, df in tables.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False))
        md = [
            "# EdgeTwin Demo Evidence Room",
            "",
            f"Customer: {evidence.get('customer')}",
            f"Use-case: {evidence.get('use_case')}",
            f"Data situation: {evidence.get('data_situation')}",
            f"Recommended route: {evidence.get('recommended_route')}",
            f"Decision: {evidence.get('decision', {}).get('decision')}",
            f"Overall score: {evidence.get('decision', {}).get('overall_score')}%",
            "",
            "## Boundary",
            DEMO_BOUNDARY,
            "",
            "## Safe claims",
        ]
        md.extend([f"- {x}" for x in evidence.get("safe_claims", [])])
        md.extend(["", "## Blocked claims"])
        md.extend([f"- {x}" for x in evidence.get("blocked_claims", [])])
        md.extend(["", "## Recommended next steps"])
        md.extend([f"- {x}" for x in evidence.get("recommended_next_steps", [])])
        zf.writestr("README_demo_evidence_room.md", "\n".join(md))
    return mem.getvalue()


def render_streamlit_tab(st):
    st.header("Demo Evidence Room")
    st.write(
        "Show customers what they receive before they buy: scorecard, limitations, safe claims and next steps."
    )
    st.info(DEMO_BOUNDARY)

    col1, col2 = st.columns(2)
    with col1:
        customer = st.text_input("Demo customer", value="Demo manufacturing team", key="demo_room_customer")
        use_case = st.selectbox(
            "Demo use-case",
            [
                "Bearing wear / lager-slijtage",
                "General machine anomaly",
                "Facade / site security sentinel",
                "Fire / arson-risk context",
                "Existing industrial data export",
            ],
            key="demo_room_use_case",
        )
    with col2:
        data_situation = st.selectbox(
            "Data situation",
            ["Good real data", "Limited real data", "No real data yet", "Hardware only / no clean dataset"],
            index=1,
            key="demo_room_data_situation",
        )
        recommended_route = st.selectbox(
            "Recommended route shown in demo",
            [
                "Real-Data Evidence Pack",
                "Real Data + Synthetic Evidence Expansion",
                "Dataset Starter Pack / Hardware Profile / Field Data Kit",
                "Hardware Profile + Firmware Starter Pack",
            ],
            index=1,
            key="demo_room_route",
        )

    evidence = build_demo_evidence_room(customer, use_case, data_situation, recommended_route)
    bundle = create_demo_evidence_bundle(evidence)
    decision = evidence["decision"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Demo evidence score", f"{decision['overall_score']}%")
    m2.metric("Decision", decision["decision"])
    m3.metric("Route", recommended_route)

    tables = evidence_to_tables(evidence)
    t1, t2, t3, t4, t5 = st.tabs(["Scorecard", "Limitations", "Safe claims", "Blocked claims", "Next steps"])
    with t1:
        st.dataframe(tables["scorecard"], use_container_width=True)
    with t2:
        st.dataframe(tables["limitations"], use_container_width=True)
    with t3:
        st.dataframe(tables["safe_claims"], use_container_width=True)
    with t4:
        st.dataframe(tables["blocked_claims"], use_container_width=True)
    with t5:
        st.dataframe(tables["next_steps"], use_container_width=True)

    st.download_button(
        "Download demo evidence room bundle",
        data=bundle,
        file_name="edgetwin_demo_evidence_room.zip",
        mime="application/zip",
        use_container_width=True,
        key="demo_room_download_bundle",
    )
    return evidence, bundle
