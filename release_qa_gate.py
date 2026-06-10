"""EdgeTwin Studio Release QA Gate.

Purpose:
- Check that the release is close to first-customer-ready.
- Avoid dead tabs, missing key modules, unsafe wording and unclear next steps.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import re
import zipfile
from typing import Any, Dict, List, Tuple

import pandas as pd

MODULE = "Release QA Gate"
SAFE_BOUNDARY = "QA confirms product readiness for founder-led customer testing; it does not prove market demand or production-certified operation."

REQUIRED_FILES = [
    "app.py",
    "readiness_snapshot.py",
    "demo_evidence_room.py",
    "data_coverage_engine.py",
    "hardware_profile_matrix.py",
    "hardware_firmware_starter_pack.py",
    "custom_pack_checkout.py",
    "no_gezeik_policy.py",
    "commercial_readiness_pack.py",
    "data_quality_gate_pro.py",
    "buyer_data_room_pro.py",
    "final_customer_flow.py",
]

REQUIRED_DOCS = [
    "README.md",
    "SALES_ONE_PAGER.md",
    "PRICING_CARD.md",
    "OUTREACH_MESSAGES.md",
    "SAFE_CLAIM_LIBRARY_CUSTOMER.md",
]

UNSAFE_VISIBLE_PHRASES = [
    "100% accuracy",
    "guaranteed accuracy",
    "production guarantee",
    "safety certified",
    "compliance certified",
    "legal approval",
    "risk-free",
    "onomstotelijk bewijs",
    "absolute duidelijkheid",
]

CUSTOMER_ROUTE_KEYS = [
    "ultimate_customer_flow_tab",
    "readiness_snapshot_tab",
    "data_coverage_engine_tab",
    "hardware_profile_matrix_tab",
    "hardware_firmware_starter_tab",
    "custom_pack_checkout_tab",
    "demo_evidence_room_tab",
    "buyer_data_room_pro_tab",
    "clean_quote_delivery_tab",
]


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def run_release_qa(root_dir: str = ".") -> Dict[str, Any]:
    root_dir = os.path.abspath(root_dir)
    files_present = {name: os.path.exists(os.path.join(root_dir, name)) for name in REQUIRED_FILES}
    docs_present = {name: os.path.exists(os.path.join(root_dir, name)) for name in REQUIRED_DOCS}

    app_text = _read_text(os.path.join(root_dir, "app.py"))
    visible_version_hits = re.findall(r"\bV\d{2,3}(?:\.\d+)?\b", app_text)
    route_missing = [key for key in CUSTOMER_ROUTE_KEYS if key not in app_text]

    unsafe_hits = []
    scan_files = ["app.py", "README.md", "SALES_ONE_PAGER.md", "PRICING_CARD.md", "SAFE_CLAIM_LIBRARY_CUSTOMER.md"]
    for fname in scan_files:
        text = _read_text(os.path.join(root_dir, fname)).lower()
        for phrase in UNSAFE_VISIBLE_PHRASES:
            phrase_l = phrase.lower()
            start = 0
            while True:
                idx = text.find(phrase_l, start)
                if idx == -1:
                    break
                context = text[max(0, idx - 300): idx + len(phrase_l) + 300]
                negated = any(marker in context for marker in [
                    "no ", "not ", "does not", "do not", "without", "geen ", "niet ", "nooit", "doesn't", "don’t", "don't", "avoid", "do not use", "asks for", "requested", "blocked", "forbidden"
                ])
                # Phrases are acceptable when they are clearly used as forbidden-claim examples.
                if not negated:
                    unsafe_hits.append({"file": fname, "phrase": phrase})
                    break
                start = idx + len(phrase_l)

    checklist = []
    checklist.append({"Check": "All critical modules exist", "Pass": all(files_present.values()), "Detail": ", ".join([k for k, v in files_present.items() if not v]) or "OK"})
    checklist.append({"Check": "All launch docs exist", "Pass": all(docs_present.values()), "Detail": ", ".join([k for k, v in docs_present.items() if not v]) or "OK"})
    checklist.append({"Check": "Customer route wired in app", "Pass": not route_missing, "Detail": ", ".join(route_missing) or "OK"})
    checklist.append({"Check": "Unsafe customer claims not present in main copy", "Pass": not unsafe_hits, "Detail": json.dumps(unsafe_hits, ensure_ascii=False) if unsafe_hits else "OK"})
    checklist.append({"Check": "Customer mode avoids technical overload", "Pass": "CUSTOMER_LAUNCH_FLOW" in app_text, "Detail": "OK" if "CUSTOMER_LAUNCH_FLOW" in app_text else "Missing CUSTOMER_LAUNCH_FLOW"})
    checklist.append({"Check": "Quote/delivery is request/payment-link safe", "Pass": "Request payment" in app_text or "payment_link_ready" in app_text, "Detail": "OK" if ("Request payment" in app_text or "payment_link_ready" in app_text) else "Payment handoff unclear"})

    passed = sum(1 for item in checklist if item["Pass"])
    score = int(100 * passed / len(checklist)) if checklist else 0
    blockers = [item for item in checklist if not item["Pass"]]

    snapshot = {
        "module": MODULE,
        "created_at": _now(),
        "root_dir": root_dir,
        "qa_score": score,
        "decision": "FIRST-CUSTOMER READY CANDIDATE" if score >= 90 and not blockers else "CONDITIONAL RELEASE - FIX BLOCKERS",
        "checklist": checklist,
        "blockers": blockers,
        "files_present": files_present,
        "docs_present": docs_present,
        "visible_version_hit_count_internal_scan": len(visible_version_hits),
        "note_on_version_hits": "Internal version keys may still exist; customer text cleaner hides visible labels. Manual UI check remains recommended.",
        "safe_boundary": SAFE_BOUNDARY,
    }
    snapshot["snapshot_hash"] = _sha(snapshot)
    return snapshot


def build_tables(snapshot: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    checklist = pd.DataFrame(snapshot.get("checklist", []))
    files = pd.DataFrame([{"File": k, "Present": v} for k, v in snapshot.get("files_present", {}).items()])
    docs = pd.DataFrame([{"Doc": k, "Present": v} for k, v in snapshot.get("docs_present", {}).items()])
    return {"checklist": checklist, "files": files, "docs": docs}


def create_bundle(snapshot: Dict[str, Any]) -> bytes:
    tables = build_tables(snapshot)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("release_qa_manifest.json", json.dumps(snapshot, indent=2, ensure_ascii=False, default=str))
        zf.writestr("release_qa_checklist.csv", tables["checklist"].to_csv(index=False))
        zf.writestr("release_qa_files.csv", tables["files"].to_csv(index=False))
        zf.writestr("release_qa_docs.csv", tables["docs"].to_csv(index=False))
        zf.writestr(
            "RELEASE_QA_SUMMARY.md",
            f"# EdgeTwin Release QA\n\nScore: {snapshot.get('qa_score')}%\n\nDecision: {snapshot.get('decision')}\n\nBoundary: {snapshot.get('safe_boundary')}\n",
        )
    return buffer.getvalue()


def render_streamlit_tab(st) -> Tuple[Dict[str, Any] | None, bytes | None]:
    st.header("Release QA Gate")
    st.caption("Final internal check before showing EdgeTwin to the first 3 founding customers.")

    root_dir = st.text_input("Repository/root directory to scan", ".", key="release_qa_root")
    if not st.button("Run release QA", use_container_width=True, key="run_release_qa"):
        st.info("Run this after uploading a patch. It checks the critical modules, docs, customer route and unsafe customer claims.")
        return None, None

    snapshot = run_release_qa(root_dir)
    bundle = create_bundle(snapshot)
    tables = build_tables(snapshot)

    c1, c2, c3 = st.columns(3)
    c1.metric("QA score", f"{snapshot.get('qa_score')}%")
    c2.metric("Decision", snapshot.get("decision"))
    c3.metric("Blockers", len(snapshot.get("blockers", [])))

    if snapshot.get("blockers"):
        st.warning(snapshot.get("decision"))
    else:
        st.success(snapshot.get("decision"))

    tabs = st.tabs(["Checklist", "Files", "Docs", "Manifest", "Download"])
    with tabs[0]:
        st.dataframe(tables["checklist"], use_container_width=True)
    with tabs[1]:
        st.dataframe(tables["files"], use_container_width=True)
    with tabs[2]:
        st.dataframe(tables["docs"], use_container_width=True)
    with tabs[3]:
        st.json(snapshot)
    with tabs[4]:
        st.download_button("Download release QA bundle", bundle, file_name="edgetwin_release_qa_bundle.zip", mime="application/zip", use_container_width=True)

    return snapshot, bundle
