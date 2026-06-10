import io
import json
import re
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    px = None

try:
    import core
except Exception as exc:
    core = None
    CORE_IMPORT_ERROR = exc
else:
    CORE_IMPORT_ERROR = None

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EdgeTwin Studio",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# EDGE TWIN STUDIO V145 — WEBSITE SHELL
# Complete replacement app.py
# Keeps your existing core.py backend, but gives the app a cleaner
# SaaS / customer-portal style interface.
# ============================================================


# ============================================================
# SAFE BACKEND BOOT
# ============================================================

def backend_available() -> bool:
    return core is not None and CORE_IMPORT_ERROR is None


if backend_available():
    try:
        core.init_db()
    except Exception as exc:
        st.error(f"Backend database could not start: {exc}")
        st.stop()
else:
    st.error("EdgeTwin backend could not be imported. Make sure core.py is in the same folder as app.py.")
    st.exception(CORE_IMPORT_ERROR)
    st.stop()


# ============================================================
# CUSTOMER TEXT CLEANER
# ============================================================

def clean_visible_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value
    text = re.sub(r"\b[Vv]\d+(?:\.\d+)?\s*(?:[-–—/]\s*[Vv]?\d+(?:\.\d+)?)*\b", "", text)
    replacements = {
        "turns/ into": "turns into",
        "/server-side": "server-side",
        "→/webhook": "→ webhook",
        "our- commerce": "our commerce",
        "EdgeTwin Studio -": "EdgeTwin Studio",
        "EdgeTwin Studio —": "EdgeTwin Studio",
        "Customer Portal Lite": "Customer Portal",
        "Secure Download Links Bundle": "Secure Access Bundle",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[ \t]*[—-][ \t]*(?=\n|$)", "", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip() if text.strip() else value


# ============================================================
# STATE
# ============================================================

def init_state() -> None:
    defaults = {
        "project_id": str(uuid.uuid4()),
        "project_name": "Bearing_Readiness_Project",
        "workspace_mode": "Customer Mode",
        "active_customer_page": "Home",
        "active_founder_page": "Founder Overview",
        "selected_plan": "Founder Test Mode",
        "dataset": pd.DataFrame(),
        "raw_upload_name": "",
        "last_demo_summary": {},
        "fusion_df": pd.DataFrame(),
        "fusion_manifest": {},
        "fusion_doctor": {},
        "fusion_training_df": pd.DataFrame(),
        "fusion_bundle": None,
        "enterprise_bundle": None,
        "auto_pilot_result": None,
        "auto_pilot_bundle": None,
        "auto_pilot_config": None,
        "hardware_result": None,
        "optimizer_result": None,
        "optimizer_bundle": None,
        "trust_gate": None,
        "trust_bundle": None,
        "reliability_v2": None,
        "reliability_v2_bundle": None,
        "professional_report_snapshot": None,
        "professional_report_bundle": None,
        "deployment_plan": None,
        "deployment_bundle": None,
        "last_error": None,
        "intake": {
            "company": "",
            "asset": "Rotating equipment / motor / bearing system",
            "use_case": "Check whether my machine data is ready for a safe predictive-maintenance pilot.",
            "signals": ["Vibration", "Audio"],
            "target": "Bearing wear / motor health readiness",
            "data_status": "I have some data, but I am not sure if it is usable",
            "decision_goal": "Decide whether a paid pilot is safe and useful",
        },
        "demo_choice": "Predictive Maintenance Demo",
        "signal_type": "Vibration / IMU",
        "sr": 4000,
        "base_f": 50.0,
        "harm_r": 0.2,
        "imp_r": 2.0,
        "noise_l": 0.1,
        "current_label": "Baseline_Normal",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# STYLE
# ============================================================

def install_css() -> None:
    st.markdown(
        """
<style>
:root {
    --et-card: rgba(255,255,255,0.055);
    --et-card-soft: rgba(255,255,255,0.032);
    --et-line: rgba(150,160,190,0.24);
    --et-text-soft: rgba(235,238,247,0.72);
    --et-text-faint: rgba(235,238,247,0.54);
    --et-accent: #7aa2ff;
    --et-good: #44d38a;
    --et-warn: #ffc857;
    --et-bad: #ff6b6b;
    --et-radius-lg: 26px;
    --et-radius-md: 18px;
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 4rem;
    max-width: 1220px;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(150,160,190,0.16);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    letter-spacing: -0.02em;
}

.et-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
}

.et-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    font-weight: 800;
    font-size: 1.1rem;
    letter-spacing: -0.02em;
}

.et-logo {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: radial-gradient(circle at 30% 20%, rgba(255,255,255,0.42), rgba(122,162,255,0.34) 35%, rgba(80,80,120,0.18));
    border: 1px solid rgba(255,255,255,0.18);
}

.et-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.32rem 0.72rem;
    border-radius: 999px;
    border: 1px solid var(--et-line);
    background: rgba(255,255,255,0.035);
    color: var(--et-text-soft);
    font-size: 0.82rem;
    margin: 0.14rem 0.18rem 0.14rem 0;
    white-space: nowrap;
}

.et-hero {
    padding: 2.4rem 2rem;
    border: 1px solid var(--et-line);
    border-radius: var(--et-radius-lg);
    background:
        radial-gradient(circle at top left, rgba(122,162,255,0.20), transparent 34%),
        radial-gradient(circle at bottom right, rgba(68,211,138,0.10), transparent 28%),
        linear-gradient(135deg, rgba(255,255,255,0.075), rgba(255,255,255,0.025));
    margin: 0.65rem 0 1.1rem 0;
    box-shadow: 0 18px 60px rgba(0,0,0,0.18);
}

.et-kicker {
    color: var(--et-accent);
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 800;
    margin-bottom: 0.75rem;
}

.et-hero h1 {
    font-size: clamp(2rem, 4vw, 3.55rem);
    line-height: 1.02;
    letter-spacing: -0.055em;
    margin: 0 0 0.85rem 0;
    max-width: 860px;
}

.et-hero p {
    font-size: 1.08rem;
    max-width: 790px;
    color: var(--et-text-soft);
    margin: 0 0 1.1rem 0;
}

.et-card {
    border: 1px solid var(--et-line);
    border-radius: var(--et-radius-md);
    background: var(--et-card);
    padding: 1.15rem 1.2rem;
    min-height: 138px;
    margin-bottom: 1rem;
}

.et-card-tight {
    border: 1px solid var(--et-line);
    border-radius: 16px;
    background: var(--et-card-soft);
    padding: 0.95rem 1rem;
    margin-bottom: 0.75rem;
}

.et-card h3,
.et-card-tight h3 {
    margin: 0 0 0.45rem 0;
    font-size: 1.08rem;
    letter-spacing: -0.02em;
}

.et-card p,
.et-card-tight p {
    margin: 0;
    color: var(--et-text-soft);
    font-size: 0.94rem;
}

.et-muted {
    color: var(--et-text-soft);
    font-size: 0.93rem;
}

.et-faint {
    color: var(--et-text-faint);
    font-size: 0.86rem;
}

.et-divider {
    height: 1px;
    background: rgba(150,160,190,0.18);
    margin: 1.25rem 0;
}

.et-step {
    display: grid;
    grid-template-columns: 44px 1fr;
    gap: 0.9rem;
    align-items: start;
    border: 1px solid var(--et-line);
    border-radius: 16px;
    background: var(--et-card-soft);
    padding: 0.9rem 1rem;
    margin-bottom: 0.65rem;
}

.et-step-number {
    width: 34px;
    height: 34px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(122,162,255,0.34);
    background: rgba(122,162,255,0.12);
    color: #cfe0ff;
    font-weight: 800;
}

.et-step-title {
    font-weight: 800;
    margin-bottom: 0.12rem;
}

.et-status-good { color: var(--et-good); font-weight: 800; }
.et-status-warn { color: var(--et-warn); font-weight: 800; }
.et-status-bad { color: var(--et-bad); font-weight: 800; }

div[data-testid="stMetric"] {
    border: 1px solid var(--et-line);
    border-radius: 17px;
    padding: 0.8rem 0.85rem;
    background: rgba(255,255,255,0.035);
}

.stButton > button {
    border-radius: 13px !important;
    font-weight: 750 !important;
}

.stDownloadButton > button {
    border-radius: 13px !important;
    font-weight: 750 !important;
}

[data-testid="stTabs"] button {
    font-weight: 700;
}

.et-mini-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
}

.et-mini-table td {
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(150,160,190,0.14);
}

.et-mini-table td:first-child {
    color: var(--et-text-soft);
}

.et-mini-table td:last-child {
    text-align: right;
    font-weight: 800;
}
</style>
        """,
        unsafe_allow_html=True,
    )


install_css()


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_call(fn_name: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    if not hasattr(core, fn_name):
        return default
    try:
        return getattr(core, fn_name)(*args, **kwargs)
    except Exception as exc:
        st.session_state.last_error = f"{fn_name}: {exc}"
        return default


def get_pricing_plans_safe() -> List[str]:
    plans = safe_call("get_pricing_plans", default=[])
    if isinstance(plans, list) and plans:
        return plans
    return ["Founder Test Mode", "Starter", "Pilot", "Enterprise"]


def as_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    try:
        return pd.DataFrame(value)
    except Exception:
        return pd.DataFrame()


def has_dataset() -> bool:
    return isinstance(st.session_state.dataset, pd.DataFrame) and not st.session_state.dataset.empty


def score_status(score: Any) -> Tuple[str, str]:
    try:
        score_int = int(float(score))
    except Exception:
        return "Unknown", "warn"
    if score_int >= 80:
        return "Ready for controlled pilot", "good"
    if score_int >= 60:
        return "Needs a controlled pilot plan", "warn"
    return "Not ready yet", "bad"


def metric_value(obj: Dict[str, Any], keys: List[str], default: Any = 0) -> Any:
    for key in keys:
        if isinstance(obj, dict) and key in obj and obj[key] is not None:
            return obj[key]
    return default


def render_topbar() -> None:
    user = st.session_state.get("user", {}) or {}
    username = user.get("username", "workspace")
    st.markdown(
        f"""
<div class="et-topbar">
    <div class="et-brand"><div class="et-logo">⚙️</div><div>EdgeTwin Studio</div></div>
    <div><span class="et-pill">Workspace: {clean_visible_text(st.session_state.workspace_mode)}</span><span class="et-pill">User: {username}</span></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
<div class="et-hero">
    <div class="et-kicker">Industrial AI readiness & evidence platform</div>
    <h1>Know if your machine data is ready before you invest in AI.</h1>
    <p>
        EdgeTwin Studio turns vibration, audio, maintenance and sensor data into a clear pilot-readiness view:
        what is usable, what is missing, what is risky, and what the next safe step should be.
    </p>
    <span class="et-pill">Bearing wear</span>
    <span class="et-pill">Motor health</span>
    <span class="et-pill">Rotating equipment</span>
    <span class="et-pill">Dataset readiness</span>
    <span class="et-pill">Evidence packs</span>
    <span class="et-pill">Safe AI claims</span>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_value_cards() -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
<div class="et-card">
    <h3>1. Start simple</h3>
    <p>Describe the machine, the available signals and the decision you need to make.</p>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="et-card">
    <h3>2. Check the data</h3>
    <p>Upload real data or run a guided demo to inspect quality, balance, readiness and risk.</p>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div class="et-card">
    <h3>3. Export evidence</h3>
    <p>Create a professional evidence pack with limitations, assumptions and safe next steps.</p>
</div>
            """,
            unsafe_allow_html=True,
        )


def render_flow_steps(active: int = 1) -> None:
    steps = [
        ("Describe", "Machine, use-case, signals and decision goal."),
        ("Analyze", "Demo data, uploaded data or generated pilot dataset."),
        ("Review", "Readiness verdict, risks, missing data and trust checks."),
        ("Export", "Evidence pack, report bundle and next-step advice."),
    ]
    for idx, (title, body) in enumerate(steps, start=1):
        marker = "✓" if idx < active else str(idx)
        st.markdown(
            f"""
<div class="et-step">
    <div class="et-step-number">{marker}</div>
    <div>
        <div class="et-step-title">{title}</div>
        <div class="et-muted">{body}</div>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )


def render_dataset_preview(df: pd.DataFrame, title: str = "Dataset preview") -> None:
    if df is None or df.empty:
        st.info("No dataset loaded yet.")
        return
    st.markdown(f"### {title}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Numeric columns", len(df.select_dtypes(include="number").columns))
    st.dataframe(df.head(100), use_container_width=True)


# ============================================================
# BACKEND FLOW HELPERS
# ============================================================

def reset_generated_outputs() -> None:
    for key in [
        "fusion_bundle",
        "enterprise_bundle",
        "auto_pilot_bundle",
        "optimizer_bundle",
        "trust_bundle",
        "reliability_v2_bundle",
        "professional_report_bundle",
        "deployment_bundle",
    ]:
        st.session_state[key] = None


def run_demo_flow(demo_name: str) -> bool:
    result = safe_call("run_demo_project", demo_name, default=None)
    if not isinstance(result, dict):
        st.error("Demo could not run. Check whether core.run_demo_project exists and supports this demo name.")
        return False

    demo = result.get("demo", {}) or {}
    project_title = demo.get("title") or demo_name or "EdgeTwin_Demo_Project"
    st.session_state.project_name = str(project_title).replace(" ", "_").replace("/", "_")
    st.session_state.last_demo_summary = result.get("commercial_summary", {}) or {}
    st.session_state.fusion_df = as_dataframe(result.get("fusion_df", pd.DataFrame()))
    st.session_state.fusion_manifest = result.get("manifest", {}) or {}
    st.session_state.fusion_doctor = result.get("doctor", {}) or {}
    st.session_state.fusion_training_df = as_dataframe(result.get("training_df", pd.DataFrame()))
    st.session_state.dataset = st.session_state.fusion_training_df.copy()
    st.session_state.hardware_result = result.get("hardware")

    bundle_result = safe_call(
        "create_sensor_fusion_export_bundle",
        st.session_state.project_name,
        st.session_state.fusion_df,
        st.session_state.fusion_manifest,
        st.session_state.last_demo_summary,
        result.get("reliability"),
        st.session_state.hardware_result,
        default=None,
    )

    if isinstance(bundle_result, tuple) and len(bundle_result) >= 3:
        st.session_state.fusion_bundle = bundle_result[0]
        st.session_state.fusion_doctor = bundle_result[1] or st.session_state.fusion_doctor
        st.session_state.fusion_training_df = as_dataframe(bundle_result[2])
        if not st.session_state.fusion_training_df.empty:
            st.session_state.dataset = st.session_state.fusion_training_df.copy()

    st.session_state.enterprise_bundle = safe_call(
        "create_enterprise_bundle",
        st.session_state.project_name,
        st.session_state.dataset,
        st.session_state.fusion_doctor,
        st.session_state.hardware_result,
        default=None,
    )
    return True


def run_auto_pilot_flow(config: Dict[str, Any]) -> bool:
    result = safe_call("run_auto_pilot_project", config, default=None)
    if not isinstance(result, dict):
        st.error("Auto pilot could not run. Check core.run_auto_pilot_project.")
        return False

    st.session_state.auto_pilot_result = result
    st.session_state.auto_pilot_config = config
    st.session_state.project_name = f"{config.get('use_case_type', 'Custom').replace(' ', '_').replace('/', '_')}_Pilot"
    st.session_state.fusion_df = as_dataframe(result.get("fusion_df", pd.DataFrame()))
    st.session_state.fusion_manifest = result.get("manifest", {}) or {}
    st.session_state.fusion_doctor = result.get("doctor", {}) or {}
    st.session_state.fusion_training_df = as_dataframe(result.get("training_df", pd.DataFrame()))
    st.session_state.dataset = st.session_state.fusion_training_df.copy()
    st.session_state.hardware_result = result.get("hardware")
    st.session_state.last_demo_summary = result.get("commercial_summary", {}) or {}
    st.session_state.auto_pilot_bundle = safe_call("create_auto_pilot_bundle", st.session_state.project_name, result, default=None)

    bundle_result = safe_call(
        "create_sensor_fusion_export_bundle",
        st.session_state.project_name,
        st.session_state.fusion_df,
        st.session_state.fusion_manifest,
        st.session_state.last_demo_summary,
        result.get("reliability"),
        st.session_state.hardware_result,
        default=None,
    )
    if isinstance(bundle_result, tuple) and len(bundle_result) >= 3:
        st.session_state.fusion_bundle = bundle_result[0]
        st.session_state.fusion_doctor = bundle_result[1] or st.session_state.fusion_doctor
        st.session_state.fusion_training_df = as_dataframe(bundle_result[2])
        if not st.session_state.fusion_training_df.empty:
            st.session_state.dataset = st.session_state.fusion_training_df.copy()

    st.session_state.enterprise_bundle = safe_call(
        "create_enterprise_bundle",
        st.session_state.project_name,
        st.session_state.dataset,
        st.session_state.fusion_doctor,
        st.session_state.hardware_result,
        default=None,
    )
    return True


def run_readiness_checks() -> None:
    df = st.session_state.dataset if has_dataset() else pd.DataFrame()

    if df.empty:
        st.warning("Load or generate a dataset first.")
        return

    doctor = safe_call("fusion_dataset_doctor", df, default=None)
    if isinstance(doctor, dict):
        st.session_state.fusion_doctor = doctor

    optimizer = safe_call("run_smart_dataset_optimizer", df, default=None)
    if isinstance(optimizer, dict):
        st.session_state.optimizer_result = optimizer
        st.session_state.optimizer_bundle = safe_call(
            "create_optimizer_bundle",
            st.session_state.project_name,
            optimizer,
            df,
            default=None,
        )

    trust = safe_call("build_trust_gate", st.session_state.project_name, df, st.session_state.fusion_doctor, default=None)
    if isinstance(trust, dict):
        st.session_state.trust_gate = trust
        st.session_state.trust_bundle = safe_call(
            "create_trust_bundle",
            st.session_state.project_name,
            trust,
            default=None,
        )

    reliability = safe_call(
        "build_reliability_engine_v2",
        st.session_state.project_name,
        df,
        st.session_state.fusion_doctor,
        st.session_state.trust_gate,
        default=None,
    )
    if isinstance(reliability, dict):
        st.session_state.reliability_v2 = reliability
        st.session_state.reliability_v2_bundle = safe_call(
            "create_reliability_v2_bundle",
            st.session_state.project_name,
            reliability,
            default=None,
        )

    deployment = safe_call(
        "build_deployment_plan",
        st.session_state.project_name,
        df,
        st.session_state.fusion_doctor,
        st.session_state.hardware_result,
        default=None,
    )
    if isinstance(deployment, dict):
        st.session_state.deployment_plan = deployment
        st.session_state.deployment_bundle = safe_call(
            "create_deployment_bundle",
            st.session_state.project_name,
            deployment,
            default=None,
        )

    report = safe_call(
        "build_professional_report_snapshot",
        project_name=st.session_state.project_name,
        customer_name=st.session_state.intake.get("company") or "Customer",
        prepared_by="EdgeTwin Studio",
        dataset_df=df,
        doctor=st.session_state.fusion_doctor,
        reliability_v2=st.session_state.reliability_v2,
        real_bridge=None,
        trust_gate=st.session_state.trust_gate,
        hardware=st.session_state.hardware_result,
        deployment=st.session_state.deployment_plan,
        package_level="Pilot readiness evidence pack",
        default=None,
    )
    if isinstance(report, dict):
        st.session_state.professional_report_snapshot = report
        st.session_state.professional_report_bundle = safe_call(
            "create_professional_report_bundle",
            st.session_state.project_name,
            report,
            df,
            default=None,
        )

    st.success("Readiness checks completed.")


def read_uploaded_file(uploaded_file: Any) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(data))
    if name.endswith(".json"):
        return pd.read_json(io.BytesIO(data))
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(data))
    raise ValueError("Supported formats: CSV, XLSX, JSON, Parquet")


def downloadable_bytes(value: Any) -> Optional[bytes]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    return None


# ============================================================
# AUTH
# ============================================================

def render_auth_screen() -> None:
    st.markdown(
        """
<div class="et-hero">
    <div class="et-kicker">EdgeTwin Studio</div>
    <h1>Industrial AI readiness without the chaos.</h1>
    <p>Login or create a local workspace to start checking whether machine data is ready for a safe pilot.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", type="primary", use_container_width=True, key="login_button"):
            user = safe_call("authenticate_user", username, password, default=None)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Login failed. Check username and password.")

    with right:
        st.markdown("### Create account")
        new_username = st.text_input("New username", key="reg_user")
        new_password = st.text_input("New password", type="password", key="reg_pass")
        if st.button("Create account", use_container_width=True, key="create_account_button"):
            user = safe_call("create_user", new_username, new_password, default=None)
            if user:
                st.session_state.user = user
                st.success("Account created.")
                st.rerun()
            else:
                st.error("Username already exists, username is empty, or password is too short. Use at least 6 characters.")


if "user" not in st.session_state:
    render_auth_screen()
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

def save_current_project() -> None:
    settings = {
        "project_name": st.session_state.project_name,
        "workspace_mode": st.session_state.workspace_mode,
        "selected_plan": st.session_state.selected_plan,
        "intake": st.session_state.intake,
        "last_demo_summary": st.session_state.last_demo_summary,
        "fusion_manifest": st.session_state.fusion_manifest,
        "fusion_doctor": st.session_state.fusion_doctor,
        "hardware_result": st.session_state.hardware_result,
        "optimizer_result": st.session_state.optimizer_result,
        "trust_gate": st.session_state.trust_gate,
        "reliability_v2": st.session_state.reliability_v2,
        "deployment_plan": st.session_state.deployment_plan,
        "professional_report_snapshot": st.session_state.professional_report_snapshot,
        "raw_upload_name": st.session_state.raw_upload_name,
    }
    safe_call(
        "save_project",
        st.session_state.project_id,
        st.session_state.user["id"],
        st.session_state.project_name,
        st.session_state.dataset if has_dataset() else pd.DataFrame(),
        settings,
        default=None,
    )
    st.sidebar.success("Project saved.")


def load_selected_project(project_name: str, projects: pd.DataFrame) -> None:
    proj_id = projects.loc[projects["name"] == project_name, "id"].iloc[0]
    loaded = safe_call("load_project", proj_id, st.session_state.user["id"], default=None)
    if not loaded:
        st.sidebar.error("Project could not be loaded.")
        return

    settings = loaded.get("settings", {}) or {}
    st.session_state.project_id = proj_id
    st.session_state.project_name = loaded.get("name", project_name)
    st.session_state.dataset = as_dataframe(loaded.get("dataset", pd.DataFrame()))
    st.session_state.fusion_training_df = st.session_state.dataset.copy()
    st.session_state.workspace_mode = settings.get("workspace_mode", st.session_state.workspace_mode)
    st.session_state.selected_plan = settings.get("selected_plan", st.session_state.selected_plan)
    st.session_state.intake = settings.get("intake", st.session_state.intake)
    st.session_state.last_demo_summary = settings.get("last_demo_summary", {})
    st.session_state.fusion_manifest = settings.get("fusion_manifest", {})
    st.session_state.fusion_doctor = settings.get("fusion_doctor", {})
    st.session_state.hardware_result = settings.get("hardware_result")
    st.session_state.optimizer_result = settings.get("optimizer_result")
    st.session_state.trust_gate = settings.get("trust_gate")
    st.session_state.reliability_v2 = settings.get("reliability_v2")
    st.session_state.deployment_plan = settings.get("deployment_plan")
    st.session_state.professional_report_snapshot = settings.get("professional_report_snapshot")
    st.session_state.raw_upload_name = settings.get("raw_upload_name", "")
    reset_generated_outputs()
    st.sidebar.success("Project loaded.")
    st.rerun()


def render_sidebar() -> None:
    st.sidebar.markdown("# ⚙️ EdgeTwin")
    st.sidebar.caption("Industrial AI readiness portal")
    st.sidebar.success(f"Logged in as {st.session_state.user.get('username', 'user')}")

    st.session_state.workspace_mode = st.sidebar.radio(
        "Workspace mode",
        ["Customer Mode", "Founder Mode"],
        index=0 if st.session_state.workspace_mode == "Customer Mode" else 1,
        key="workspace_mode_radio",
    )

    st.session_state.project_name = st.sidebar.text_input(
        "Project name",
        value=st.session_state.project_name,
        key="sidebar_project_name",
    )

    if st.session_state.workspace_mode == "Founder Mode":
        plans = get_pricing_plans_safe()
        st.session_state.selected_plan = st.sidebar.selectbox(
            "Access plan",
            plans,
            index=plans.index(st.session_state.selected_plan) if st.session_state.selected_plan in plans else 0,
            key="sidebar_selected_plan",
        )
    else:
        st.sidebar.caption("Customer Mode shows the clean pilot-readiness route.")

    if st.sidebar.button("Save project", use_container_width=True, key="sidebar_save_project"):
        save_current_project()

    projects = safe_call("get_user_projects", st.session_state.user["id"], default=pd.DataFrame())
    projects = as_dataframe(projects)
    if not projects.empty and "name" in projects.columns and "id" in projects.columns:
        choice = st.sidebar.selectbox("Load project", ["-"] + projects["name"].astype(str).tolist(), key="sidebar_load_project")
        if choice != "-" and st.sidebar.button("Load selected", use_container_width=True, key="sidebar_load_selected"):
            load_selected_project(choice, projects)

    st.sidebar.markdown("---")

    if st.session_state.workspace_mode == "Customer Mode":
        customer_pages = ["Home", "Intake", "Demo Pilot", "Upload Data", "Readiness", "Evidence Pack"]
        st.session_state.active_customer_page = st.sidebar.radio(
            "Navigation",
            customer_pages,
            index=customer_pages.index(st.session_state.active_customer_page) if st.session_state.active_customer_page in customer_pages else 0,
            key="customer_navigation",
        )
    else:
        founder_pages = ["Founder Overview", "Projects & Storage", "Advanced Signals", "Backend Check"]
        st.session_state.active_founder_page = st.sidebar.radio(
            "Founder navigation",
            founder_pages,
            index=founder_pages.index(st.session_state.active_founder_page) if st.session_state.active_founder_page in founder_pages else 0,
            key="founder_navigation",
        )

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True, key="sidebar_logout"):
        del st.session_state.user
        st.rerun()


render_sidebar()
render_topbar()


# ============================================================
# CUSTOMER PAGES
# ============================================================

def page_home() -> None:
    render_hero()
    render_value_cards()

    st.markdown("### Your pilot-readiness path")
    cols = st.columns([1.25, 1])
    with cols[0]:
        render_flow_steps(active=1)
    with cols[1]:
        st.markdown(
            """
<div class="et-card">
    <h3>What EdgeTwin does not claim</h3>
    <p>
        EdgeTwin does not promise 100% prediction, safety certification or guaranteed production performance.
        It gives evidence, risk visibility and a safer decision path before bigger AI spending.
    </p>
</div>
            """,
            unsafe_allow_html=True,
        )
        if has_dataset():
            score = metric_value(st.session_state.fusion_doctor, ["overall_score", "readiness_score"], 0)
            status, status_class = score_status(score)
            st.markdown(
                f"""
<div class="et-card-tight">
    <h3>Current workspace</h3>
    <table class="et-mini-table">
        <tr><td>Project</td><td>{st.session_state.project_name}</td></tr>
        <tr><td>Dataset rows</td><td>{len(st.session_state.dataset)}</td></tr>
        <tr><td>Readiness</td><td><span class="et-status-{status_class}">{score}%</span></td></tr>
        <tr><td>Status</td><td>{status}</td></tr>
    </table>
</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Start with Intake, Demo Pilot or Upload Data.")


def page_intake() -> None:
    st.markdown("# Start intake")
    st.caption("Keep this simple. The goal is to understand the asset, data and decision before running checks.")

    render_flow_steps(active=1)
    st.markdown("---")

    intake = dict(st.session_state.intake)
    c1, c2 = st.columns(2)
    with c1:
        intake["company"] = st.text_input("Company / customer name", value=intake.get("company", ""))
        intake["asset"] = st.text_input("Machine or asset", value=intake.get("asset", "Rotating equipment / motor / bearing system"))
        intake["target"] = st.text_input("Target problem", value=intake.get("target", "Bearing wear / motor health readiness"))
    with c2:
        intake["data_status"] = st.selectbox(
            "Current data status",
            [
                "I have some data, but I am not sure if it is usable",
                "I have sensor data and maintenance history",
                "I only have maintenance history",
                "I do not have useful data yet",
                "I want to start with a demo or synthetic pilot dataset",
            ],
            index=0,
        )
        intake["signals"] = st.multiselect(
            "Available signals",
            ["Vibration", "Audio", "Temperature", "Current", "Maintenance logs", "Environmental", "Radar", "Gas / air quality", "Other"],
            default=intake.get("signals", ["Vibration", "Audio"]),
        )
        intake["decision_goal"] = st.selectbox(
            "Decision goal",
            [
                "Decide whether a paid pilot is safe and useful",
                "Check if my real data is good enough",
                "Prepare a management evidence pack",
                "Find missing sensors or data gaps",
                "Compare synthetic and real data readiness",
            ],
            index=0,
        )

    intake["use_case"] = st.text_area(
        "Use-case description",
        value=intake.get("use_case", "Check whether my machine data is ready for a safe predictive-maintenance pilot."),
        height=110,
    )

    st.session_state.intake = intake

    if st.button("Save intake", type="primary", use_container_width=True):
        st.success("Intake saved. Next step: run Demo Pilot or upload real data.")


def page_demo_pilot() -> None:
    st.markdown("# Demo pilot")
    st.caption("Generate a complete example flow so the customer can understand the product without technical setup.")

    render_flow_steps(active=2)
    st.markdown("---")

    demo_names = safe_call("get_demo_projects", default=[])
    if not isinstance(demo_names, list) or not demo_names:
        demo_names = ["Predictive Maintenance Demo", "Smart Forestry Threat", "Motor Health Demo"]

    default_index = demo_names.index(st.session_state.demo_choice) if st.session_state.demo_choice in demo_names else 0
    st.session_state.demo_choice = st.selectbox("Choose demo scenario", demo_names, index=default_index)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Run demo pilot", type="primary", use_container_width=True):
            with st.spinner("Running EdgeTwin demo pilot..."):
                ok = run_demo_flow(st.session_state.demo_choice)
            if ok:
                st.success("Demo pilot generated.")
    with c2:
        if st.button("Run one-click custom pilot", use_container_width=True):
            config = {
                "use_case_type": st.session_state.intake.get("target", "Bearing wear / motor health readiness"),
                "asset_type": st.session_state.intake.get("asset", "Rotating equipment"),
                "signal_types": st.session_state.intake.get("signals", ["Vibration", "Audio"]),
                "customer_problem": st.session_state.intake.get("use_case", "Check pilot readiness."),
                "decision_goal": st.session_state.intake.get("decision_goal", "Decide whether a paid pilot is safe and useful"),
            }
            with st.spinner("Building one-click pilot..."):
                ok = run_auto_pilot_flow(config)
            if ok:
                st.success("One-click pilot generated.")

    if st.session_state.last_demo_summary:
        st.markdown("### Pilot result")
        summary = st.session_state.last_demo_summary
        c1, c2, c3 = st.columns(3)
        c1.metric("Readiness", f"{summary.get('overall_score', metric_value(st.session_state.fusion_doctor, ['overall_score'], 0))}%")
        c2.metric("Reliability", f"{summary.get('reliability_score', 0)}%")
        c3.metric("Recommended board", summary.get("recommended_board", "Unknown"))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Problem")
            st.write(summary.get("problem", st.session_state.intake.get("use_case", "")))
            st.markdown("#### Recommended next step")
            st.write(summary.get("cta", "Run readiness checks and export the evidence pack."))
        with c2:
            st.markdown("#### Output")
            st.write(summary.get("output", "Dataset, audit, readiness view and export bundle generated."))
            st.markdown("#### Solution")
            st.write(summary.get("solution", "EdgeTwin created a structured pilot-readiness evidence flow."))

        render_dataset_preview(st.session_state.dataset, "Generated dataset preview")


def page_upload_data() -> None:
    st.markdown("# Upload data")
    st.caption("Upload real customer data. Keep claims careful: this checks readiness, not guaranteed production performance.")

    render_flow_steps(active=2)
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV, Excel, JSON or Parquet", type=["csv", "xlsx", "xls", "json", "parquet"])
    if uploaded is not None:
        try:
            df = read_uploaded_file(uploaded)
            st.session_state.dataset = df.copy()
            st.session_state.fusion_training_df = df.copy()
            st.session_state.raw_upload_name = uploaded.name
            st.success(f"Uploaded {uploaded.name}: {len(df)} rows, {len(df.columns)} columns.")
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

    if has_dataset():
        render_dataset_preview(st.session_state.dataset, "Current uploaded dataset")
        if st.button("Run readiness checks on this data", type="primary", use_container_width=True):
            with st.spinner("Checking data quality, trust and pilot readiness..."):
                run_readiness_checks()


def page_readiness() -> None:
    st.markdown("# Readiness")
    st.caption("A simple view of whether the data looks usable for a controlled industrial AI pilot.")

    render_flow_steps(active=3)
    st.markdown("---")

    if not has_dataset():
        st.warning("No dataset yet. Generate a demo pilot or upload data first.")
        return

    if st.button("Run / refresh readiness checks", type="primary", use_container_width=True):
        with st.spinner("Running checks..."):
            run_readiness_checks()

    doctor = st.session_state.fusion_doctor or {}
    trust = st.session_state.trust_gate or {}
    reliability = st.session_state.reliability_v2 or {}

    overall = metric_value(doctor, ["overall_score", "readiness_score", "score"], 0)
    diversity = metric_value(doctor, ["diversity_score"], 0)
    balance = metric_value(doctor, ["balance_score"], 0)
    separation = metric_value(doctor, ["separation_score"], 0)
    status, status_class = score_status(overall)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Readiness", f"{overall}%")
    c2.metric("Diversity", f"{diversity}%")
    c3.metric("Balance", f"{balance}%")
    c4.metric("Separation", f"{separation}%")

    st.markdown(
        f"""
<div class="et-card-tight">
    <h3>Verdict</h3>
    <p><span class="et-status-{status_class}">{status}</span></p>
</div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Issues and advice")
        advice = doctor.get("advice", []) if isinstance(doctor, dict) else []
        if advice:
            for item in advice:
                sev = item.get("severity", "info")
                msg = item.get("message", "")
                if sev == "high":
                    st.error(msg)
                elif sev == "medium":
                    st.warning(msg)
                else:
                    st.info(msg)
        else:
            st.info("No detailed advice yet. Run readiness checks to generate recommendations.")

    with c2:
        st.markdown("### Trust and reliability")
        trust_score = metric_value(trust, ["trust_score", "score", "overall_score"], "Not checked")
        reliability_score = metric_value(reliability, ["reliability_score", "overall_score", "score"], "Not checked")
        st.markdown(
            f"""
<div class="et-card-tight">
    <table class="et-mini-table">
        <tr><td>Trust score</td><td>{trust_score}</td></tr>
        <tr><td>Reliability score</td><td>{reliability_score}</td></tr>
        <tr><td>Rows checked</td><td>{len(st.session_state.dataset)}</td></tr>
        <tr><td>Source</td><td>{st.session_state.raw_upload_name or 'Generated / demo data'}</td></tr>
    </table>
</div>
            """,
            unsafe_allow_html=True,
        )

    numeric_df = st.session_state.dataset.select_dtypes(include="number")
    if px is not None and not numeric_df.empty:
        st.markdown("### Numeric data overview")
        selected = st.selectbox("Column to visualize", numeric_df.columns.tolist())
        fig = px.histogram(st.session_state.dataset, x=selected, nbins=40)
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=330)
        st.plotly_chart(fig, use_container_width=True)


def page_evidence_pack() -> None:
    st.markdown("# Evidence pack")
    st.caption("Export the customer-facing proof package: readiness, limitations, risks and next safe steps.")

    render_flow_steps(active=4)
    st.markdown("---")

    if not has_dataset():
        st.warning("No dataset yet. Generate or upload data first.")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Build evidence pack", type="primary", use_container_width=True):
            with st.spinner("Building report and evidence bundles..."):
                run_readiness_checks()
    with c2:
        st.info("Evidence packs are useful for management review, pilot scoping and safe sales conversations.")

    st.markdown("### Available downloads")

    downloads = [
        ("Sensor fusion evidence bundle", st.session_state.fusion_bundle, "sensor_fusion_evidence_bundle.zip"),
        ("Enterprise evidence bundle", st.session_state.enterprise_bundle, "enterprise_evidence_bundle.zip"),
        ("Trust bundle", st.session_state.trust_bundle, "trust_bundle.zip"),
        ("Reliability bundle", st.session_state.reliability_v2_bundle, "reliability_bundle.zip"),
        ("Professional report bundle", st.session_state.professional_report_bundle, "professional_report_bundle.zip"),
        ("Deployment plan bundle", st.session_state.deployment_bundle, "deployment_plan_bundle.zip"),
        ("Optimizer bundle", st.session_state.optimizer_bundle, "optimizer_bundle.zip"),
        ("Auto pilot bundle", st.session_state.auto_pilot_bundle, "auto_pilot_bundle.zip"),
    ]

    any_download = False
    for label, value, filename in downloads:
        data = downloadable_bytes(value)
        if data:
            any_download = True
            st.download_button(
                label=f"Download {label}",
                data=data,
                file_name=f"{st.session_state.project_name}_{filename}",
                mime="application/zip",
                use_container_width=True,
                key=f"download_{filename}",
            )

    if not any_download:
        st.info("No bundles available yet. Click 'Build evidence pack' first.")

    if st.session_state.professional_report_snapshot:
        st.markdown("### Report snapshot")
        st.json(st.session_state.professional_report_snapshot)
    elif st.session_state.last_demo_summary:
        st.markdown("### Commercial summary")
        st.json(st.session_state.last_demo_summary)


# ============================================================
# FOUNDER PAGES
# ============================================================

def page_founder_overview() -> None:
    st.markdown("# Founder overview")
    st.caption("Internal control area. Customers should stay in Customer Mode.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Project", st.session_state.project_name)
    c2.metric("Rows", len(st.session_state.dataset) if has_dataset() else 0)
    c3.metric("Mode", st.session_state.workspace_mode)
    c4.metric("Plan", st.session_state.selected_plan)

    st.markdown("### Internal status")
    st.json(
        {
            "has_dataset": has_dataset(),
            "raw_upload_name": st.session_state.raw_upload_name,
            "has_doctor": bool(st.session_state.fusion_doctor),
            "has_trust_gate": bool(st.session_state.trust_gate),
            "has_reliability": bool(st.session_state.reliability_v2),
            "has_report": bool(st.session_state.professional_report_snapshot),
            "last_error": st.session_state.last_error,
        }
    )


def page_projects_storage() -> None:
    st.markdown("# Projects & storage")
    st.caption("Save, load and inspect project data.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save current project", type="primary", use_container_width=True):
            save_current_project()
    with c2:
        if st.button("New empty project", use_container_width=True):
            st.session_state.project_id = str(uuid.uuid4())
            st.session_state.project_name = "New_EdgeTwin_Project"
            st.session_state.dataset = pd.DataFrame()
            reset_generated_outputs()
            st.success("New project started.")

    projects = safe_call("get_user_projects", st.session_state.user["id"], default=pd.DataFrame())
    projects = as_dataframe(projects)
    if not projects.empty:
        st.markdown("### Saved projects")
        st.dataframe(projects, use_container_width=True)
    else:
        st.info("No saved projects yet.")

    if has_dataset():
        render_dataset_preview(st.session_state.dataset, "Current project dataset")


def page_advanced_signals() -> None:
    st.markdown("# Advanced signals")
    st.caption("Founder-only synthetic signal sandbox for fast demos and testing.")

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.signal_type = st.radio("Signal type", ["Audio / Acoustic", "Vibration / IMU"], index=0 if st.session_state.signal_type == "Audio / Acoustic" else 1)
        st.session_state.sr = 16000 if st.session_state.signal_type == "Audio / Acoustic" else 4000
        st.session_state.current_label = st.text_input("Dataset label", value=st.session_state.current_label)
    with c2:
        st.session_state.base_f = st.slider("Base frequency", 0.0, 1000.0, float(st.session_state.base_f), 5.0)
        st.session_state.harm_r = st.slider("Harmonics", 0.0, 2.0, float(st.session_state.harm_r), 0.05)
        st.session_state.imp_r = st.slider("Impact rate", 0.0, 50.0, float(st.session_state.imp_r), 0.5)
        st.session_state.noise_l = st.slider("Noise", 0.0, 1.0, float(st.session_state.noise_l), 0.02)

    if st.button("Generate signal sample", type="primary", use_container_width=True):
        signal = safe_call(
            "generate_universal_signal",
            sr=st.session_state.sr,
            base_f=st.session_state.base_f,
            harm_r=st.session_state.harm_r,
            imp_r=st.session_state.imp_r,
            noise_l=st.session_state.noise_l,
            default=None,
        )
        if signal is None:
            st.error("Signal generation failed. Check core.generate_universal_signal signature.")
        else:
            try:
                features = safe_call("extract_signal_features", signal, st.session_state.sr, default={}) or {}
                row = dict(features)
                row["Label"] = st.session_state.current_label
                row["sample_rate"] = st.session_state.sr
                row["created_at"] = datetime.utcnow().isoformat()
                new_row = pd.DataFrame([row])
                st.session_state.dataset = pd.concat([st.session_state.dataset, new_row], ignore_index=True)
                st.success("Signal sample added to dataset.")
            except Exception as exc:
                st.error(f"Feature extraction failed: {exc}")

    if has_dataset():
        render_dataset_preview(st.session_state.dataset, "Generated feature dataset")


def page_backend_check() -> None:
    st.markdown("# Backend check")
    st.caption("Small sanity check without touching customer-facing claims.")

    checks = []
    for fn_name in [
        "init_db",
        "create_user",
        "authenticate_user",
        "get_user_projects",
        "save_project",
        "load_project",
        "get_demo_projects",
        "run_demo_project",
        "run_auto_pilot_project",
        "fusion_dataset_doctor",
        "create_sensor_fusion_export_bundle",
        "create_enterprise_bundle",
    ]:
        checks.append({"function": fn_name, "available": hasattr(core, fn_name)})

    st.dataframe(pd.DataFrame(checks), use_container_width=True)

    if st.button("Run lightweight backend check", type="primary", use_container_width=True):
        results = []
        try:
            plans = get_pricing_plans_safe()
            results.append({"check": "Pricing plans", "result": "PASS", "detail": ", ".join(map(str, plans[:4]))})
        except Exception as exc:
            results.append({"check": "Pricing plans", "result": "FAIL", "detail": str(exc)})

        try:
            demos = safe_call("get_demo_projects", default=[])
            results.append({"check": "Demo list", "result": "PASS" if demos else "WARN", "detail": str(demos[:3] if isinstance(demos, list) else demos)})
        except Exception as exc:
            results.append({"check": "Demo list", "result": "FAIL", "detail": str(exc)})

        try:
            df = st.session_state.dataset if has_dataset() else pd.DataFrame({"rms": [0.1, 0.2, 0.9, 1.1], "kurtosis": [2.5, 2.8, 5.2, 5.6], "Label": ["Normal", "Normal", "Fault", "Fault"]})
            doctor = safe_call("fusion_dataset_doctor", df, default={})
            results.append({"check": "Dataset doctor", "result": "PASS" if isinstance(doctor, dict) else "WARN", "detail": str(type(doctor))})
        except Exception as exc:
            results.append({"check": "Dataset doctor", "result": "FAIL", "detail": str(exc)})

        st.dataframe(pd.DataFrame(results), use_container_width=True)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)


# ============================================================
# ROUTER
# ============================================================

if st.session_state.workspace_mode == "Customer Mode":
    page = st.session_state.active_customer_page
    if page == "Home":
        page_home()
    elif page == "Intake":
        page_intake()
    elif page == "Demo Pilot":
        page_demo_pilot()
    elif page == "Upload Data":
        page_upload_data()
    elif page == "Readiness":
        page_readiness()
    elif page == "Evidence Pack":
        page_evidence_pack()
else:
    page = st.session_state.active_founder_page
    if page == "Founder Overview":
        page_founder_overview()
    elif page == "Projects & Storage":
        page_projects_storage()
    elif page == "Advanced Signals":
        page_advanced_signals()
    elif page == "Backend Check":
        page_backend_check()
