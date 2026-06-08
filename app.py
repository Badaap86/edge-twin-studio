import uuid
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

import core

warnings.filterwarnings("ignore")

st.set_page_config(page_title="EdgeTwin Studio V45", layout="wide", initial_sidebar_state="expanded")

core.init_db()


# ============================================================
# STATE
# ============================================================

def init_state():
    defaults = {
        "project_id": str(uuid.uuid4()),
        "project_name": "Smart_Forestry_Demo_Project",
        "dataset": pd.DataFrame(),
        "fusion_df": pd.DataFrame(),
        "fusion_manifest": {},
        "fusion_doctor": {},
        "fusion_training_df": pd.DataFrame(),
        "fusion_bundle": None,
        "enterprise_bundle": None,
        "hardware_result": None,
        "last_demo_summary": {},
        "selected_template": "Smart Forestry Threat",
        "sr": 16000,
        "base_f": 50.0,
        "harm_r": 0.2,
        "imp_r": 2.0,
        "noise_l": 0.1,
        "current_label": "Baseline_Normal",
        "auto_pilot_result": None,
        "auto_pilot_bundle": None,
        "auto_pilot_config": None,
        "optimizer_result": None,
        "optimizer_bundle": None,
        "trust_gate": None,
        "trust_bundle": None,
        "real_bridge_result": None,
        "real_bridge_bundle": None,
        "reliability_v2": None,
        "reliability_v2_bundle": None,
        "deployment_plan": None,
        "deployment_bundle": None,
        "professional_report_bundle": None,
        "professional_report_snapshot": None,
        "monetization_snapshot": None,
        "monetization_bundle": None,
        "hardening_snapshot": None,
        "hardening_bundle": None,
        "beta_launch_snapshot": None,
        "beta_launch_bundle": None,
        "api_automation_snapshot": None,
        "api_automation_bundle": None,
        "api_simulation_response": None,
        "pack_marketplace_snapshot": None,
        "pack_marketplace_bundle": None,
        "custom_pack_definition": None,
        "marketplace_generated_dataset": pd.DataFrame(),
        "normality_result": None,
        "normality_bundle": None,
        "edge_impulse_snapshot": None,
        "edge_impulse_bundle": None,
        "edge_impulse_classifier_snapshot": None,
        "edge_impulse_classifier_bundle": None,
        "release_success_snapshot": None,
        "release_success_bundle": None,
        "golden_demo_result": None,
        "golden_demo_bundle": None,
        "closed_beta_kit": None,
        "closed_beta_bundle": None,
        "paid_license_snapshot": None,
        "paid_license_bundle": None,
        "field_validation_snapshot": None,
        "field_validation_bundle": None,
        "field_validation_df": pd.DataFrame(),
        "edge_deployment_starter_snapshot": None,
        "edge_deployment_starter_bundle": None,
        "scalability_snapshot": None,
        "scalability_bundle": None,
        "operational_control_snapshot": None,
        "operational_control_bundle": None,
        "observability_snapshot": None,
        "observability_bundle": None,
        "last_observability_event": None,
        "customer_assurance_snapshot": None,
        "customer_assurance_bundle": None,
        "onboarding_snapshot": None,
        "onboarding_bundle": None,
        "guided_success_bundle": None,
        "security_v41_snapshot": None,
        "workspace_lifecycle_snapshot": None,
        "workspace_lifecycle_bundle": None,
        "admin_usage_snapshot": None,
        "admin_usage_bundle": None,
        "commercial_license_certificate": None,
        "commercial_license_bundle": None,
        "customer_delivery_snapshot": None,
        "customer_delivery_bundle": None,
        "customer_success_snapshot": None,
        "customer_success_bundle": None,
        "pricing_offer_snapshot": None,
        "pricing_offer_bundle": None,
        "paid_pilot_v45_snapshot": None,
        "paid_pilot_v45_bundle": None,
        "field_evidence_v2_snapshot": None,
        "field_evidence_v2_bundle": None,
        "product_readiness_v40_snapshot": None,
        "product_readiness_v40_bundle": None,
        "security_hardening_v41_snapshot": None,
        "security_hardening_v41_bundle": None,
        "last_admin_export_event": None,
        "last_operator_note_event": None,
        "selected_plan": "Founder Test Mode",
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


# ============================================================
# HELPERS
# ============================================================

def render_metric_cards(doctor):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Diversity", f"{doctor.get('diversity_score', 0)}%")
    c2.metric("Balance", f"{doctor.get('balance_score', 0)}%")
    c3.metric("Separation", f"{doctor.get('separation_score', 0)}%")
    c4.metric("Overall", f"{doctor.get('overall_score', 0)}%")


def render_reliability_cards(reliability):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Synthetic Realism", f"{reliability.get('synthetic_realism_score', 0)}%")
    c2.metric("Field Readiness", f"{reliability.get('field_readiness_score', 0)}%")
    c3.metric("Reliability", f"{reliability.get('reliability_score', 0)}%")
    c4.metric("Dataset Risk", reliability.get("dataset_risk", "Unknown"))


def render_doctor(doctor):
    for item in doctor.get("advice", []):
        sev = item.get("severity", "info")
        msg = item.get("message", "")
        if sev == "high":
            st.error(msg)
        elif sev == "medium":
            st.warning(msg)
        else:
            st.info(msg)


def reset_generated_bundles():
    st.session_state.fusion_bundle = None
    st.session_state.enterprise_bundle = None
    st.session_state.auto_pilot_bundle = None
    st.session_state.optimizer_bundle = None
    st.session_state.trust_bundle = None
    st.session_state.real_bridge_bundle = None
    st.session_state.reliability_v2_bundle = None
    st.session_state.deployment_bundle = None
    st.session_state.professional_report_bundle = None
    st.session_state.monetization_bundle = None
    st.session_state.hardening_bundle = None
    st.session_state.beta_launch_bundle = None
    st.session_state.api_automation_bundle = None
    st.session_state.pack_marketplace_bundle = None
    st.session_state.normality_bundle = None
    st.session_state.edge_impulse_bundle = None
    st.session_state.edge_impulse_classifier_bundle = None
    st.session_state.release_success_bundle = None
    st.session_state.golden_demo_bundle = None
    st.session_state.closed_beta_bundle = None
    st.session_state.paid_license_bundle = None
    st.session_state.field_validation_bundle = None
    st.session_state.edge_deployment_starter_bundle = None
    st.session_state.scalability_bundle = None
    st.session_state.commercial_license_bundle = None
    st.session_state.customer_delivery_bundle = None
    st.session_state.customer_success_bundle = None
    st.session_state.pricing_offer_bundle = None
    st.session_state.paid_pilot_v45_bundle = None
    st.session_state.field_evidence_v2_bundle = None
    st.session_state.admin_usage_bundle = None
    st.session_state.observability_bundle = None
    st.session_state.customer_assurance_bundle = None
    st.session_state.onboarding_bundle = None
    st.session_state.guided_success_bundle = None
    st.session_state.workspace_lifecycle_bundle = None
    st.session_state.product_readiness_v40_bundle = None
    st.session_state.security_hardening_v41_bundle = None


def run_demo(demo_name):
    result = core.run_demo_project(demo_name)
    demo = result["demo"]

    st.session_state.project_name = demo["title"].replace(" ", "_")
    st.session_state.selected_template = demo["template"]
    st.session_state.fusion_df = result["fusion_df"]
    st.session_state.fusion_manifest = result["manifest"]
    st.session_state.fusion_doctor = result["doctor"]
    st.session_state.fusion_training_df = result["training_df"]
    st.session_state.hardware_result = result["hardware"]
    st.session_state.last_demo_summary = result["commercial_summary"]

    bundle, doctor, training_df = core.create_sensor_fusion_export_bundle(
        st.session_state.project_name,
        st.session_state.fusion_df,
        st.session_state.fusion_manifest,
        st.session_state.last_demo_summary,
        result.get("reliability"),
        st.session_state.hardware_result,
    )

    st.session_state.fusion_bundle = bundle
    st.session_state.dataset = training_df.copy()
    st.session_state.fusion_training_df = training_df.copy()
    st.session_state.fusion_doctor = doctor

    st.session_state.enterprise_bundle = core.create_enterprise_bundle(
        st.session_state.project_name,
        st.session_state.dataset,
        doctor,
        st.session_state.hardware_result,
    )


def run_auto_pilot(config):
    result = core.run_auto_pilot_project(config)
    st.session_state.auto_pilot_result = result
    st.session_state.auto_pilot_config = config
    st.session_state.project_name = f"{config.get('use_case_type', 'Custom').replace(' ', '_').replace('/', '_')}_Auto_Pilot"
    st.session_state.fusion_df = result["fusion_df"]
    st.session_state.fusion_manifest = result["manifest"]
    st.session_state.fusion_doctor = result["doctor"]
    st.session_state.fusion_training_df = result["training_df"]
    st.session_state.hardware_result = result["hardware"]
    st.session_state.dataset = result["training_df"].copy()
    st.session_state.last_demo_summary = result["commercial_summary"]
    st.session_state.auto_pilot_bundle = core.create_auto_pilot_bundle(st.session_state.project_name, result)

    fusion_bundle, doctor, training_df = core.create_sensor_fusion_export_bundle(
        st.session_state.project_name,
        result["fusion_df"],
        result["manifest"],
        result["commercial_summary"],
        result["reliability"],
        result["hardware"],
    )

    st.session_state.fusion_bundle = fusion_bundle
    st.session_state.fusion_doctor = doctor
    st.session_state.fusion_training_df = training_df.copy()
    st.session_state.enterprise_bundle = core.create_enterprise_bundle(
        st.session_state.project_name,
        st.session_state.dataset,
        doctor,
        st.session_state.hardware_result,
    )


# ============================================================
# AUTH SCREEN
# ============================================================

if "user" not in st.session_state:
    st.title("EdgeTwin Studio")
    st.caption("Powered by the OMEGA-X Synthetic Sensor Engine")
    st.markdown("### Turn sensor ideas into Edge AI-ready datasets, reliability reports and hardware advice.")

    login_tab, register_tab = st.tabs(["Login", "Create account"])

    with login_tab:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", type="primary", key="login_button"):
            user = core.authenticate_user(username, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Login failed. Check username and password.")

    with register_tab:
        username = st.text_input("New username", key="reg_user")
        password = st.text_input("New password", type="password", key="reg_pass")

        if st.button("Create account", key="create_account_button"):
            user = core.create_user(username, password)
            if user:
                st.session_state.user = user
                st.success("Account created.")
                st.rerun()
            else:
                st.error("Username already exists, username is empty, or password is too short. Use at least 6 characters.")

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("EdgeTwin Studio")
st.sidebar.caption("Powered by OMEGA-X Engine")
st.sidebar.success(f"Logged in as {st.session_state.user['username']}")

st.session_state.selected_plan = st.sidebar.selectbox(
    "Access plan",
    core.get_pricing_plans(),
    index=core.get_pricing_plans().index(st.session_state.selected_plan) if st.session_state.selected_plan in core.get_pricing_plans() else 0,
    key="sidebar_selected_plan_v24",
)
st.sidebar.caption("V24 local plan simulator. Payments are not connected yet.")

st.session_state.project_name = st.sidebar.text_input(
    "Project name",
    st.session_state.project_name,
    key="sidebar_project_name",
)

if st.sidebar.button("Save project", use_container_width=True, key="sidebar_save_project"):
    settings = {
        "selected_template": st.session_state.selected_template,
        "last_demo_summary": st.session_state.last_demo_summary,
        "fusion_manifest": st.session_state.fusion_manifest,
        "fusion_doctor": st.session_state.fusion_doctor,
        "auto_pilot_config": st.session_state.auto_pilot_config,
        "hardware_result": st.session_state.hardware_result,
        "optimizer_result": st.session_state.optimizer_result,
        "trust_gate": st.session_state.trust_gate,
        "real_bridge_summary": core.compact_bridge_summary(st.session_state.real_bridge_result) if st.session_state.real_bridge_result else {},
        "sr": st.session_state.sr,
        "selected_plan": st.session_state.selected_plan,
        "monetization_snapshot": st.session_state.monetization_snapshot,
        "release_success_snapshot": st.session_state.release_success_snapshot,
        "closed_beta_kit": st.session_state.closed_beta_kit,
        "paid_license_snapshot": st.session_state.paid_license_snapshot,
        "field_validation_snapshot": st.session_state.field_validation_snapshot,
        "field_evidence_v2_snapshot": st.session_state.field_evidence_v2_snapshot,
        "edge_deployment_starter_snapshot": st.session_state.edge_deployment_starter_snapshot,
        "pricing_offer_snapshot": st.session_state.pricing_offer_snapshot,
        "paid_pilot_v45_snapshot": st.session_state.paid_pilot_v45_snapshot,
        "observability_snapshot": st.session_state.observability_snapshot,
        "customer_assurance_snapshot": st.session_state.customer_assurance_snapshot,
        "workspace_lifecycle_snapshot": st.session_state.workspace_lifecycle_snapshot,
        "product_readiness_v40_snapshot": st.session_state.product_readiness_v40_snapshot,
        "customer_delivery_snapshot": st.session_state.customer_delivery_snapshot,
        "customer_success_snapshot": st.session_state.customer_success_snapshot,
    }

    core.save_project(
        st.session_state.project_id,
        st.session_state.user["id"],
        st.session_state.project_name,
        st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        settings,
    )

    st.sidebar.success("Project saved.")

projects = core.get_user_projects(st.session_state.user["id"])

if len(projects) > 0:
    choice = st.sidebar.selectbox(
        "Load project",
        ["-"] + projects["name"].tolist(),
        key="sidebar_load_project_select",
    )

    if choice != "-" and st.sidebar.button("Load selected", use_container_width=True, key="sidebar_load_selected"):
        proj_id = projects.loc[projects["name"] == choice, "id"].iloc[0]
        loaded = core.load_project(proj_id, st.session_state.user["id"])

        if loaded:
            settings = loaded.get("settings", {}) or {}
            st.session_state.project_id = proj_id
            st.session_state.project_name = loaded["name"]
            st.session_state.dataset = loaded["dataset"]
            st.session_state.fusion_training_df = loaded["dataset"].copy()
            st.session_state.selected_template = settings.get("selected_template", st.session_state.selected_template)
            st.session_state.last_demo_summary = settings.get("last_demo_summary", {})
            st.session_state.fusion_manifest = settings.get("fusion_manifest", {})
            st.session_state.fusion_doctor = settings.get("fusion_doctor", {})
            st.session_state.auto_pilot_config = settings.get("auto_pilot_config")
            st.session_state.hardware_result = settings.get("hardware_result")
            st.session_state.optimizer_result = settings.get("optimizer_result")
            st.session_state.trust_gate = settings.get("trust_gate")
            st.session_state.real_bridge_result = None
            st.session_state.selected_plan = settings.get("selected_plan", st.session_state.selected_plan)
            st.session_state.monetization_snapshot = settings.get("monetization_snapshot")
            st.session_state.release_success_snapshot = settings.get("release_success_snapshot")
            st.session_state.closed_beta_kit = settings.get("closed_beta_kit")
            st.session_state.paid_license_snapshot = settings.get("paid_license_snapshot")
            st.session_state.field_validation_snapshot = settings.get("field_validation_snapshot")
            st.session_state.field_evidence_v2_snapshot = settings.get("field_evidence_v2_snapshot")
            st.session_state.edge_deployment_starter_snapshot = settings.get("edge_deployment_starter_snapshot")
            st.session_state.pricing_offer_snapshot = settings.get("pricing_offer_snapshot")
            st.session_state.paid_pilot_v45_snapshot = settings.get("paid_pilot_v45_snapshot")
            st.session_state.security_hardening_v41_snapshot = settings.get("security_hardening_v41_snapshot")
            st.session_state.commercial_license_certificate = settings.get("commercial_license_certificate")
            st.session_state.customer_delivery_snapshot = settings.get("customer_delivery_snapshot")
            st.session_state.customer_success_snapshot = settings.get("customer_success_snapshot")
            st.session_state.observability_snapshot = settings.get("observability_snapshot")
            st.session_state.customer_assurance_snapshot = settings.get("customer_assurance_snapshot")
            st.session_state.workspace_lifecycle_snapshot = settings.get("workspace_lifecycle_snapshot")
            st.session_state.product_readiness_v40_snapshot = settings.get("product_readiness_v40_snapshot")
            st.session_state.sr = int(settings.get("sr", st.session_state.sr))
            st.session_state.fusion_df = pd.DataFrame()
            st.session_state.auto_pilot_result = None
            reset_generated_bundles()
            st.sidebar.success("Project loaded. Re-run export if you want fresh ZIP/PDF bundles.")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Canvas settings")

signal_type = st.sidebar.radio(
    "Signal type",
    ["Audio / Acoustic", "Vibration / IMU"],
    key="sidebar_signal_type",
)

st.session_state.sr = 4000 if signal_type == "Vibration / IMU" else 16000

st.session_state.current_label = st.sidebar.text_input(
    "Dataset label",
    st.session_state.current_label,
    key="sidebar_dataset_label",
)

st.session_state.base_f = st.sidebar.slider(
    "Base frequency",
    0.0,
    1000.0,
    float(st.session_state.base_f),
    5.0,
    key="sidebar_base_frequency",
)

st.session_state.harm_r = st.sidebar.slider(
    "Harmonics",
    0.0,
    2.0,
    float(st.session_state.harm_r),
    0.05,
    key="sidebar_harmonics",
)

st.session_state.imp_r = st.sidebar.slider(
    "Impact rate",
    0.0,
    50.0,
    float(st.session_state.imp_r),
    0.5,
    key="sidebar_impact_rate",
)

st.session_state.noise_l = st.sidebar.slider(
    "Noise",
    0.0,
    1.0,
    float(st.session_state.noise_l),
    0.02,
    key="sidebar_noise",
)

if st.sidebar.button("Logout", use_container_width=True, key="sidebar_logout"):
    del st.session_state.user
    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("EdgeTwin Studio V45")
st.caption("Self-Selling Demo • Use Case Wizard • Smart Optimizer • Synthetic-to-Real Bridge • Reliability Engine 2.0 • Trust Center • Deployment Planner • Reports 2.0 • Product Hardening • Beta Launch Readiness • SaaS-light Monetization Gate • API Automation • Industry Pack Marketplace • Normal vs Abnormal Baseline Engine • Edge Impulse Export • Edge Impulse Classifier Export • Release Success Gate • Golden Demo Suite • Closed Beta Launch Kit • Paid Export Gate • Real Field Validation • Edge Deployment Starter • Storage/Scalability • Operational Control Center • Error Observatory • Customer Assurance & Data Governance • Guided Success Onboarding • Customer Workspace & Lifecycle • Admin Dashboard & Usage Tracking • Commercial License Certificate • Field Evidence 2.0 • Full Product Readiness Gate • Security & Access Control Hardening • Customer Delivery Portal • Customer Success Feedback Loop • Pricing Validation & Offer Builder • Paid Pilot Launch Readiness")

home, wizard_tab, fusion_tab, audit_tab, optimizer_tab, real_bridge_tab, trust_tab, deployment_tab, reports_tab, hardening_tab, beta_launch_tab, monetization_tab, api_tab, marketplace_tab, normality_tab, edge_impulse_tab, ei_classifier_tab, success_gate_tab, golden_demo_tab, closed_beta_tab, paid_export_tab, field_validation_tab, field_evidence_v2_tab, product_readiness_tab, security_v41_tab, edge_starter_tab, scalability_tab, operational_tab, observability_tab, governance_tab, onboarding_tab, workspace_tab, admin_tab, license_cert_tab, delivery_tab, customer_success_tab, pricing_offer_tab, paid_pilot_v45_tab, canvas_tab, packs_tab, hardware_tab = st.tabs(
    [
        "🏠 Self-Selling Demo",
        "🧭 Use Case Wizard",
        "🧬 Sensor Fusion Studio",
        "🩺 Enterprise Audit",
        "🧪 Smart Optimizer",
        "🔗 Real Bridge",
        "🛡️ Trust Center",
        "🚀 Deployment Planner",
        "📑 Reports 2.0",
        "🧰 Product Hardening",
        "🧲 Beta Launch",
        "💳 Monetization Gate",
        "🔌 API Automation",
        "🛒 Pack Marketplace",
        "⚖️ Normality Engine",
        "📤 Edge Impulse Export",
        "🎯 EI Classifier Export",
        "✅ Success Gate",
        "🏆 Golden Demo",
        "🚪 Closed Beta",
        "🔐 Paid Export",
        "🌍 Field Validation",
        "📡 Field Evidence 2.0",
        "🏁 Product Ready V40",
        "🔑 Security V41",
        "🧩 Edge Starter",
        "📚 Storage/Scale",
        "🕹️ Control Center",
        "🛰️ Error Observatory",
        "🔒 Customer Assurance",
        "🧭 Guided Success",
        "🏢 Workspace",
        "📊 Admin/Usage",
        "📜 License Cert",
        "📦 Delivery Portal",
        "💬 Customer Success",
        "💶 Pricing Offer",
        "🤝 Paid Pilot V45",
        "📈 Signal Canvas",
        "📦 Industry Packs",
        "🧱 Hardware Architect",
    ]
)


# ============================================================
# HOME / SELF SELLING DEMO
# ============================================================

with home:
    st.header("Choose the customer problem")
    st.write(
        "One click loads the scenario, generates the fusion dataset, runs the Dataset Doctor, "
        "creates the report bundle and selects a hardware direction."
    )

    demo_names = core.get_demo_projects()
    cols = st.columns(4)

    for idx, name in enumerate(demo_names):
        demo = core.get_demo_project(name)

        with cols[idx % 4]:
            st.markdown(f"### {demo['title']}")
            st.caption(demo["problem"])
            st.write(demo["solution"])

            if st.button(f"Run {demo['title']}", key=f"run_demo_{name}", type="primary", use_container_width=True):
                with st.spinner("Running demo flow..."):
                    run_demo(name)
                st.success("Demo generated. Scroll down for the result.")

    st.markdown("---")

    if st.session_state.last_demo_summary:
        summary = st.session_state.last_demo_summary

        st.subheader("Demo result")

        c1, c2, c3 = st.columns([1.2, 1.2, 1])

        with c1:
            st.markdown("#### Problem")
            st.write(summary.get("problem", ""))
            st.markdown("#### Solution")
            st.write(summary.get("solution", ""))

        with c2:
            st.markdown("#### Output")
            st.write(summary.get("output", ""))
            st.markdown("#### Recommended next step")
            st.write(summary.get("cta", ""))

        with c3:
            st.metric("Readiness", f"{summary.get('overall_score', 0)}%")
            st.metric("Reliability", f"{summary.get('reliability_score', 0)}%")
            st.metric("Hardware", summary.get("recommended_board", "Unknown"))

        if st.session_state.fusion_doctor:
            render_metric_cards(st.session_state.fusion_doctor)
            render_doctor(st.session_state.fusion_doctor)

        d1, d2, d3 = st.columns(3)

        if st.session_state.fusion_bundle:
            d1.download_button(
                "Download Professional Fusion Bundle",
                st.session_state.fusion_bundle,
                file_name=f"{st.session_state.project_name}_fusion_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="demo_download_professional_fusion_bundle_v191",
            )

        if st.session_state.enterprise_bundle:
            d2.download_button(
                "Download Enterprise Bundle",
                st.session_state.enterprise_bundle,
                file_name=f"{st.session_state.project_name}_enterprise_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="demo_download_enterprise_bundle_v191",
            )

        if isinstance(st.session_state.fusion_training_df, pd.DataFrame) and len(st.session_state.fusion_training_df) > 0:
            d3.download_button(
                "Download Training CSV",
                st.session_state.fusion_training_df.to_csv(index=False),
                file_name=f"{st.session_state.project_name}_training.csv",
                mime="text/csv",
                use_container_width=True,
                key="demo_download_training_csv_v191",
            )

        if isinstance(st.session_state.fusion_df, pd.DataFrame) and len(st.session_state.fusion_df) > 0:
            st.subheader("Generated fusion dataset preview")
            st.dataframe(st.session_state.fusion_df.head(25), use_container_width=True)

            if "Label" in st.session_state.fusion_df.columns:
                fig = px.histogram(st.session_state.fusion_df, x="Label", title="Label distribution")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run one of the demo cards above. This is the self-selling front door of the product.")


# ============================================================
# V19.1 USE CASE WIZARD / AUTO PILOT GENERATOR
# ============================================================

with wizard_tab:
    st.header("Use Case Wizard + Auto Pilot Generator")
    st.write(
        "This is the customer-facing flow: the customer chooses the problem, sensors and output level. "
        "EdgeTwin Studio then generates the pilot dataset, reliability estimate, hardware direction and downloadable bundle."
    )

    use_case_types = core.get_use_case_types()

    use_case = st.selectbox(
        "1. What do you want to solve?",
        use_case_types,
        key="wizard_use_case_type",
    )

    defaults = core.get_use_case_defaults(use_case)

    st.info(defaults.get("problem_hint", ""))

    c1, c2 = st.columns([1.25, 1])

    with c1:
        project_goal = st.text_area(
            "2. Describe the customer problem",
            value=defaults.get("problem_hint", ""),
            height=110,
            key=f"wizard_goal_{use_case}",
        )

        selected_sensors = st.multiselect(
            "3. Which sensors are available or planned?",
            core.get_sensor_options(),
            default=defaults.get("recommended_sensors", []),
            key=f"wizard_sensors_{use_case}",
        )

        label_default = ", ".join(defaults.get("default_classes", []))

        labels_text = st.text_area(
            "4. Which classes/labels should the pilot detect?",
            value=label_default,
            height=90,
            key=f"wizard_labels_{use_case}",
        )

    with c2:
        env_options = core.get_environment_options()
        environment = st.selectbox(
            "Environment",
            env_options,
            index=env_options.index(defaults.get("environment", "Custom")) if defaults.get("environment", "Custom") in env_options else 0,
            key=f"wizard_environment_{use_case}",
        )

        samples = st.number_input(
            "Samples",
            min_value=100,
            max_value=10000,
            value=500,
            step=100,
            key=f"wizard_samples_{use_case}",
        )

        priority_options = ["balanced", "low_power", "performance", "gateway"]
        priority = st.radio(
            "Optimization priority",
            priority_options,
            index=priority_options.index(defaults.get("priority", "balanced")),
            horizontal=False,
            key=f"wizard_priority_{use_case}",
        )

        output_level = st.selectbox(
            "Output level",
            core.get_output_levels(),
            index=1,
            key=f"wizard_output_level_{use_case}",
        )

        has_real_data = st.checkbox(
            "Customer has real sensor data already",
            value=False,
            key=f"wizard_has_real_data_{use_case}",
            help="V19.1 records this in the reliability estimate. V21 will use uploads to build synthetic variants around real signal fingerprints.",
        )

    with st.expander("Optional: upload example real files for early inspection", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload WAV/CSV files",
            type=["wav", "csv"],
            accept_multiple_files=True,
            key=f"wizard_real_upload_{use_case}",
        )

        if uploaded_files:
            feature_rows = []
            for up in uploaded_files:
                features = core.extract_features_from_bytes(up.getvalue(), up.name, defaults.get("sample_rate", 16000))
                features["Filename"] = up.name
                feature_rows.append(features)
            real_df = pd.DataFrame(feature_rows)
            st.write("Early real-file feature inspection")
            st.dataframe(real_df, use_container_width=True)
            st.caption("These uploads are inspected now. In V21 they will become the basis for Synthetic-to-Real generation.")
            has_real_data = True

    config = core.build_use_case_config(
        use_case_type=use_case,
        project_goal=project_goal,
        selected_sensors=selected_sensors,
        environment=environment,
        labels_text=labels_text,
        samples=samples,
        has_real_data=has_real_data,
        output_level=output_level,
        priority=priority,
    )

    st.markdown("---")

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Template", config.get("template"))
    cc2.metric("Sensors", len(config.get("selected_sensors", [])))
    cc3.metric("Classes", len(config.get("classes", [])))
    cc4.metric("Sample rate", f"{config.get('sample_rate', 0)} Hz")

    if st.button("Generate Auto Pilot Bundle", type="primary", use_container_width=True, key="wizard_generate_auto_pilot"):
        with st.spinner("Generating auto pilot package..."):
            run_auto_pilot(config)
        st.success("Auto Pilot generated. The dataset is also loaded into Enterprise Audit.")

    if st.session_state.auto_pilot_result:
        result = st.session_state.auto_pilot_result
        summary = result["commercial_summary"]
        doctor = result["doctor"]
        reliability = result["reliability"]
        hardware = result["hardware"]

        st.subheader("Auto Pilot Result")

        r1, r2, r3 = st.columns([1.2, 1.2, 1])

        with r1:
            st.markdown("#### Problem")
            st.write(summary.get("problem", ""))
            st.markdown("#### Solution")
            st.write(summary.get("solution", ""))

        with r2:
            st.markdown("#### Output")
            st.write(summary.get("output", ""))
            st.markdown("#### Next step")
            st.write(summary.get("cta", ""))

        with r3:
            st.metric("Readiness", f"{doctor.get('overall_score', 0)}%")
            st.metric("Reliability", f"{reliability.get('reliability_score', 0)}%")
            st.metric("Hardware", hardware.get("recommendation", "Unknown"))

        render_metric_cards(doctor)
        render_reliability_cards(reliability)

        if reliability.get("dataset_risk") == "High":
            st.error(reliability.get("verdict", ""))
        elif reliability.get("dataset_risk") == "Medium":
            st.warning(reliability.get("verdict", ""))
        else:
            st.success(reliability.get("verdict", ""))

        st.caption("Reliability is a pilot estimate based on dataset structure, class balance, label separation and sensor coverage. Field validation is required before production deployment.")

        render_doctor(doctor)

        st.info(hardware.get("reason", ""))

        b1, b2, b3, b4 = st.columns(4)

        if st.session_state.auto_pilot_bundle:
            b1.download_button(
                "Download Auto Pilot Bundle",
                st.session_state.auto_pilot_bundle,
                file_name=f"{st.session_state.project_name}_auto_pilot_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="wizard_download_auto_pilot_bundle",
            )

        if st.session_state.fusion_bundle:
            b2.download_button(
                "Download Fusion Bundle",
                st.session_state.fusion_bundle,
                file_name=f"{st.session_state.project_name}_fusion_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="wizard_download_fusion_bundle",
            )

        if st.session_state.enterprise_bundle:
            b3.download_button(
                "Download Enterprise Bundle",
                st.session_state.enterprise_bundle,
                file_name=f"{st.session_state.project_name}_enterprise_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="wizard_download_enterprise_bundle",
            )

        if isinstance(result["training_df"], pd.DataFrame) and len(result["training_df"]) > 0:
            b4.download_button(
                "Download Training CSV",
                result["training_df"].to_csv(index=False),
                file_name=f"{st.session_state.project_name}_training.csv",
                mime="text/csv",
                use_container_width=True,
                key="wizard_download_training_csv",
            )

        st.subheader("Generated pilot dataset preview")
        st.dataframe(result["fusion_df"].head(50), use_container_width=True)

        if "Label" in result["fusion_df"].columns:
            fig = px.histogram(result["fusion_df"], x="Label", title="Auto Pilot label distribution")
            st.plotly_chart(fig, use_container_width=True)

        ranking = pd.DataFrame(hardware.get("ranking", []))
        if len(ranking) > 0:
            fig_hw = px.bar(ranking, x="board", y="adjusted_score", title="Hardware ranking")
            st.plotly_chart(fig_hw, use_container_width=True)


# ============================================================
# SENSOR FUSION
# ============================================================

with fusion_tab:
    st.header("Sensor Fusion Studio")

    templates = core.get_fusion_templates()

    template = st.selectbox(
        "Fusion template",
        templates,
        index=templates.index(st.session_state.selected_template) if st.session_state.selected_template in templates else 0,
        key="fusion_template_select",
    )

    st.session_state.selected_template = template
    t_data = core.get_fusion_template(template)

    st.info(t_data.get("description", ""))

    defaults = t_data.get("defaults", {})

    c1, c2, c3, c4, c5 = st.columns(5)

    audio = c1.slider("Audio", 0, 100, int(defaults.get("audio", 50)), key=f"fusion_audio_{template}")
    vibration = c2.slider("Vibration", 0, 100, int(defaults.get("vibration", 50)), key=f"fusion_vibration_{template}")
    gas = c3.slider("Gas/Env", 0, 100, int(defaults.get("gas", 20)), key=f"fusion_gas_{template}")
    radar = c4.slider("Radar", 0, 100, int(defaults.get("radar", 50)), key=f"fusion_radar_{template}")
    gps = c5.slider("GPS/Zone", 0, 100, int(defaults.get("gps", 50)), key=f"fusion_gps_{template}")

    result = core.calculate_fusion_score(audio, vibration, gas, radar, gps, template)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Fusion Score", f"{result['fusion_score']:.1f}")
    m2.metric("Health Score", f"{result['health_score']:.1f}")
    m3.metric("Confidence", f"{result['confidence']:.1f}")
    m4.metric("Level", result["level"])

    st.write(f"**Event:** {result['event']}")
    st.write(f"**Recommended action:** {result['recommended_action']}")

    st.markdown("---")

    samples = st.number_input("Samples", min_value=50, max_value=10000, value=500, step=50, key="fusion_samples")
    balanced = st.checkbox("Balanced classes", value=True, key="fusion_balanced_classes")

    if st.button("Generate Fusion Dataset", type="primary", key="fusion_generate_dataset"):
        with st.spinner("Generating fusion dataset..."):
            fusion_df, manifest = core.generate_sensor_fusion_dataset(
                template,
                samples=samples,
                base_audio=audio,
                base_vibration=vibration,
                base_gas=gas,
                base_radar=radar,
                base_gps=gps,
                balanced_classes=balanced,
            )

            doctor = core.fusion_dataset_doctor(fusion_df, template)
            training_df = core.create_fusion_training_dataframe(fusion_df, template)
            reliability = core.calculate_reliability_score(doctor, has_real_data=False, selected_sensors=["Audio", "Vibration", "Radar", "Gas / Environment", "GPS / Zone"])
            hw = core.hardware_auto_architect(max(1, len(training_df.columns) - 1), st.session_state.sr, "balanced")

            commercial_summary = {
                "problem": "Customer needs a faster route from sensor idea to Edge AI-ready dataset.",
                "solution": f"Use the {template} fusion template to generate a multi-sensor training dataset.",
                "output": "Full fusion CSV, training features, PDF report and dataset doctor advice.",
                "cta": "Upload real sensor files or request a custom industry pack for a pilot.",
            }

            bundle, doctor, training_df = core.create_sensor_fusion_export_bundle(
                st.session_state.project_name,
                fusion_df,
                manifest,
                commercial_summary,
                reliability,
                hw,
            )

            st.session_state.fusion_df = fusion_df
            st.session_state.fusion_manifest = manifest
            st.session_state.fusion_doctor = doctor
            st.session_state.fusion_training_df = training_df
            st.session_state.fusion_bundle = bundle
            st.session_state.hardware_result = hw
            st.session_state.dataset = training_df.copy()

        st.success("Fusion dataset generated and training features added to Enterprise Audit.")

    if isinstance(st.session_state.fusion_df, pd.DataFrame) and len(st.session_state.fusion_df) > 0:
        render_metric_cards(st.session_state.fusion_doctor)
        render_doctor(st.session_state.fusion_doctor)

        st.dataframe(st.session_state.fusion_df.head(50), use_container_width=True)

        if st.session_state.fusion_bundle:
            st.download_button(
                "Download Professional Fusion Bundle",
                st.session_state.fusion_bundle,
                file_name=f"{st.session_state.project_name}_fusion_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="fusion_tab_download_professional_bundle_v191",
            )


# ============================================================
# ENTERPRISE AUDIT
# ============================================================

with audit_tab:
    st.header("Enterprise Audit")
    st.write("This audits the current training feature dataset and creates a professional enterprise bundle.")

    if isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset) > 0:
        st.dataframe(st.session_state.dataset.head(50), use_container_width=True)

        if "Label" in st.session_state.dataset.columns:
            numeric_cols = [c for c in st.session_state.dataset.columns if c != "Label" and pd.api.types.is_numeric_dtype(st.session_state.dataset[c])]

            if len(numeric_cols) >= 1:
                X = st.session_state.dataset[numeric_cols]
                y = st.session_state.dataset["Label"]

                audit = core.dataset_doctor(X, y)

                render_metric_cards(audit)
                render_doctor(audit)

                hw = core.hardware_auto_architect(num_features=len(numeric_cols), sr=st.session_state.sr, target="balanced")
                st.session_state.hardware_result = hw
                st.info(hw["reason"])

                if len(y.unique()) >= 2 and y.value_counts().min() >= 2:
                    try:
                        clf = RandomForestClassifier(n_estimators=120, random_state=42)
                        clf.fit(X, y)

                        perm = permutation_importance(clf, X, y, n_repeats=5, random_state=42)

                        imp_df = pd.DataFrame({"Feature": numeric_cols, "Importance": perm.importances_mean}).sort_values("Importance", ascending=False)

                        st.subheader("Feature importance")
                        st.dataframe(imp_df, use_container_width=True)

                        fig = px.bar(imp_df, x="Feature", y="Importance", title="Permutation importance")
                        st.plotly_chart(fig, use_container_width=True)

                        cv = min(5, int(y.value_counts().min()))
                        preds = cross_val_predict(clf, X, y, cv=cv)
                        labels = list(y.unique())
                        cm = confusion_matrix(y, preds, labels=labels)
                        fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True, title="Cross-validation confusion matrix")
                        st.plotly_chart(fig_cm, use_container_width=True)

                    except Exception as e:
                        st.warning(f"Model validation skipped: {e}")

                bundle = core.create_enterprise_bundle(st.session_state.project_name, st.session_state.dataset, audit, hw)
                st.session_state.enterprise_bundle = bundle

                st.download_button(
                    "Download Enterprise Bundle",
                    bundle,
                    file_name=f"{st.session_state.project_name}_enterprise_bundle.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="audit_download_enterprise_bundle_v191",
                )

            else:
                st.warning("No numeric feature columns found.")

        else:
            st.warning("Dataset needs a Label column.")

    else:
        st.info("Run a demo, use the wizard, or generate a fusion dataset first.")

    uploaded = st.file_uploader("Or upload your own CSV for audit", type=["csv"], key="audit_upload_csv_v191")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.dataset = df
        reset_generated_bundles()
        st.success("CSV loaded into Enterprise Audit.")
        st.rerun()



# ============================================================
# V20 SMART DATASET OPTIMIZER
# ============================================================

with optimizer_tab:
    st.header("Smart Dataset Optimizer")
    st.write(
        "V20 maakt EdgeTwin sterker: de app geeft niet alleen kritiek, maar kan de dataset ook automatisch verbeteren "
        "voor een betere pilot-start. Gebruik dit als voorbereiding, niet als vervanging van echte veldvalidatie."
    )

    if isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset) > 0 and "Label" in st.session_state.dataset.columns:
        report = core.smart_dataset_optimizer_report(st.session_state.dataset)

        if report.get("status") == "ok":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Score", f"{report.get('current_score', 0)}%")
            c2.metric("Rows", report.get("rows", 0))
            c3.metric("Weakest Class", report.get("weakest_class", "-"))
            c4.metric("Target / Class", report.get("target_per_class", 0))

            st.subheader("Optimizer diagnosis")
            for item in report.get("recommended_actions", []):
                sev = item.get("severity", "info")
                msg = item.get("message", "")
                if sev == "high":
                    st.error(msg)
                elif sev == "medium":
                    st.warning(msg)
                else:
                    st.info(msg)

            left, right = st.columns([1, 1])
            with left:
                st.markdown("#### Class counts")
                counts_df = pd.DataFrame(
                    [{"Label": k, "Samples": v} for k, v in report.get("class_counts", {}).items()]
                )
                st.dataframe(counts_df, use_container_width=True)

            with right:
                st.markdown("#### Feature issues")
                st.write(f"Redundant features: {len(report.get('redundant_features', []))}")
                st.write(f"Low variance features: {len(report.get('low_variance_features', []))}")
                if report.get("redundant_features"):
                    st.caption(", ".join(report.get("redundant_features", [])[:10]))

            st.markdown("---")
            st.subheader("Choose improvements")

            o1, o2 = st.columns(2)
            with o1:
                balance_classes = st.checkbox("Balance classes / add weak-class samples", value=True, key="optimizer_balance_v20")
                improve_separation = st.checkbox("Improve label separation", value=True, key="optimizer_separation_v20")
                add_noise = st.checkbox("Add realistic noise variants", value=False, key="optimizer_noise_v20")
                reduce_features = st.checkbox("Reduce redundant / low-variance features", value=False, key="optimizer_reduce_v20")
            with o2:
                target_per_class = st.number_input(
                    "Target samples per class",
                    min_value=1,
                    max_value=50000,
                    value=int(report.get("target_per_class", 50) or 50),
                    step=10,
                    key="optimizer_target_per_class_v20",
                )
                noise_strength = st.slider("Noise strength", 0.0, 0.20, 0.03, 0.01, key="optimizer_noise_strength_v20")
                separation_strength = st.slider("Separation strength", 0.0, 0.30, 0.08, 0.01, key="optimizer_separation_strength_v20")
                st.caption("Houd separation/noise laag voor geloofwaardige pilotdata.")

            actions = []
            if balance_classes:
                actions.append("balance_classes")
            if improve_separation:
                actions.append("improve_label_separation")
            if add_noise:
                actions.append("add_realistic_noise")
            if reduce_features:
                actions.append("reduce_redundant_features")

            if st.button("Run Smart Dataset Optimizer", type="primary", use_container_width=True, key="optimizer_run_v20"):
                before_df = st.session_state.dataset.copy()
                with st.spinner("Optimizing dataset..."):
                    result = core.run_smart_dataset_optimizer(
                        before_df,
                        actions=actions,
                        target_per_class=target_per_class,
                        noise_strength=noise_strength,
                        separation_strength=separation_strength,
                    )
                    st.session_state.dataset = result["optimized_df"].copy()
                    st.session_state.fusion_training_df = result["optimized_df"].copy()
                    st.session_state.optimizer_result = result
                    st.session_state.optimizer_bundle = core.create_optimizer_bundle(
                        st.session_state.project_name,
                        before_df,
                        result,
                    )
                    st.session_state.enterprise_bundle = None
                    st.session_state.fusion_bundle = None
                    st.session_state.auto_pilot_bundle = None
                st.success("Dataset optimized and loaded into Enterprise Audit.")

            if st.session_state.optimizer_result:
                result = st.session_state.optimizer_result
                before = result.get("before_report", {})
                after = result.get("after_report", {})

                st.markdown("---")
                st.subheader("Optimization result")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Before Score", f"{before.get('current_score', 0)}%")
                r2.metric("After Score", f"{after.get('current_score', 0)}%")
                r3.metric("Rows After", len(result.get("optimized_df", pd.DataFrame())))
                r4.metric("Features After", after.get("feature_count", 0))

                for ch in result.get("changes", []):
                    st.success(ch.get("message", ""))

                st.dataframe(result.get("optimized_df", pd.DataFrame()).head(50), use_container_width=True)

                d1, d2 = st.columns(2)
                d1.download_button(
                    "Download Optimized CSV",
                    st.session_state.dataset.to_csv(index=False),
                    file_name=f"{st.session_state.project_name}_optimized_dataset.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="optimizer_download_csv_v20",
                )
                if st.session_state.optimizer_bundle:
                    d2.download_button(
                        "Download Optimizer Bundle",
                        st.session_state.optimizer_bundle,
                        file_name=f"{st.session_state.project_name}_optimizer_bundle.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key="optimizer_download_bundle_v20",
                    )

            st.info(core.RELIABILITY_DISCLAIMER)
        else:
            st.warning(report.get("recommended_actions", [{}])[0].get("message", "Dataset cannot be optimized yet."))
    else:
        st.info("Run a demo, use the wizard, generate a fusion dataset, or upload a CSV with a Label column first.")


# ============================================================
# V21 SYNTHETIC-TO-REAL BRIDGE
# ============================================================

with real_bridge_tab:
    st.header("Synthetic-to-Real Bridge")
    st.write(
        "V21 maakt EdgeTwin betrouwbaarder: upload echte WAV/CSV sensorbestanden, maak een OMEGA-X Signal Fingerprint, "
        "genereer real-based variants en vergelijk hoe dicht synthetic data bij echte velddata ligt."
    )

    st.info(
        "Gebruik dit als geloofwaardigheidslaag: echte data -> fingerprint -> synthetic variants -> similarity score -> real samples needed."
    )

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        uploaded_real_files = st.file_uploader(
            "Upload real WAV/CSV files",
            type=["wav", "csv"],
            accept_multiple_files=True,
            key="real_bridge_upload_v21",
            help="CSV mag time,value of één numerieke signaalkolom bevatten. WAV wordt automatisch naar mono fingerprint omgezet.",
        )

        label_mode = st.radio(
            "Label handling",
            ["Use one label for all uploads", "Infer label from filename"],
            horizontal=True,
            key="real_bridge_label_mode_v21",
        )

        bridge_label = st.text_input(
            "Label for uploaded files",
            value=st.session_state.current_label,
            key="real_bridge_label_v21",
            help="Bij meerdere classes kun je best duidelijke filenames gebruiken, of later per class apart uploaden.",
        )

    with col_b:
        sr_hint = st.number_input(
            "CSV sample-rate hint",
            min_value=100,
            max_value=48000,
            value=int(st.session_state.sr),
            step=100,
            key="real_bridge_sr_hint_v21",
        )
        variants_per_file = st.number_input(
            "Variants per real file",
            min_value=1,
            max_value=500,
            value=25,
            step=5,
            key="real_bridge_variants_v21",
        )
        jitter_strength = st.slider(
            "Variant realism jitter",
            min_value=0.01,
            max_value=0.25,
            value=0.08,
            step=0.01,
            key="real_bridge_jitter_v21",
            help="Lager = dichter bij echte fingerprint. Hoger = meer variatie, maar minder veilig.",
        )
        compare_current_dataset = st.checkbox(
            "Compare with current synthetic/project dataset",
            value=isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset) > 0,
            key="real_bridge_compare_current_v21",
        )

    if uploaded_real_files:
        preview_rows = []
        for up in uploaded_real_files:
            label = core.label_from_filename(up.name) if label_mode == "Infer label from filename" else bridge_label
            preview_rows.append({"Filename": up.name, "Label": label, "Size KB": round(len(up.getvalue()) / 1024, 1)})
        st.subheader("Upload preview")
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

    if st.button("Run Synthetic-to-Real Bridge", type="primary", use_container_width=True, key="real_bridge_run_v21"):
        if not uploaded_real_files:
            st.warning("Upload eerst minimaal één WAV/CSV bestand.")
        else:
            file_specs = []
            for up in uploaded_real_files:
                label = core.label_from_filename(up.name) if label_mode == "Infer label from filename" else bridge_label
                file_specs.append({"filename": up.name, "bytes": up.getvalue(), "label": label})

            existing_df = st.session_state.dataset if compare_current_dataset and isinstance(st.session_state.dataset, pd.DataFrame) else None
            with st.spinner("Building real signal fingerprints and synthetic variants..."):
                result = core.run_synthetic_to_real_bridge(
                    file_specs,
                    existing_synthetic_df=existing_df,
                    variants_per_file=int(variants_per_file),
                    jitter_strength=float(jitter_strength),
                    sr_hint=int(sr_hint),
                )
                st.session_state.real_bridge_result = result

                bridge_df = result.get("bridge_training_df", pd.DataFrame())
                if isinstance(bridge_df, pd.DataFrame) and len(bridge_df) > 0:
                    st.session_state.dataset = bridge_df.copy()
                    st.session_state.fusion_training_df = bridge_df.copy()
                    st.session_state.fusion_doctor = result.get("doctor", {})
                    numeric_cols = [c for c in bridge_df.columns if c != "Label" and pd.api.types.is_numeric_dtype(bridge_df[c])]
                    st.session_state.hardware_result = core.hardware_auto_architect(
                        max(1, len(numeric_cols)),
                        int(sr_hint),
                        "performance" if int(sr_hint) >= 8000 else "balanced",
                    )
                    st.session_state.reliability_v2 = core.build_reliability_engine_v2(
                        bridge_df,
                        doctor=result.get("doctor", {}),
                        reliability=result.get("reliability", {}),
                        hardware_result=st.session_state.hardware_result,
                        bridge_result=result,
                        selected_sensors=["Audio", "Vibration"],
                        has_real_data=True,
                    )
                    result["reliability_v2"] = st.session_state.reliability_v2
                    st.session_state.real_bridge_bundle = core.create_synthetic_to_real_bridge_bundle(
                        st.session_state.project_name,
                        result,
                    )
                    st.session_state.enterprise_bundle = None
                    st.session_state.trust_bundle = None
                    st.session_state.reliability_v2_bundle = None
                    st.session_state.deployment_bundle = None
                    st.session_state.professional_report_bundle = None
                    st.session_state.monetization_bundle = None
            if st.session_state.real_bridge_result.get("status") == "ok":
                st.success("Synthetic-to-Real Bridge completed and loaded into Enterprise Audit.")
            else:
                st.error("No valid real files could be fingerprinted. Check CSV/WAV format.")

    if st.session_state.real_bridge_result:
        result = st.session_state.real_bridge_result
        summary = result.get("summary", {})
        similarity = result.get("similarity", {})
        reliability = result.get("reliability", {})
        sample_plan = result.get("sample_plan", {})

        st.markdown("---")
        st.subheader("Bridge result")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Real files", summary.get("real_files", 0))
        m2.metric("Generated variants", summary.get("generated_variants", 0))
        m3.metric("Synthetic-to-Real", f"{similarity.get('similarity_score', 0)}%")
        m4.metric("Reliability", f"{reliability.get('reliability_score', 0)}%")

        rel2 = result.get("reliability_v2") or core.build_reliability_engine_v2(
            result.get("bridge_training_df", pd.DataFrame()),
            doctor=result.get("doctor", {}),
            reliability=reliability,
            hardware_result=st.session_state.hardware_result,
            bridge_result=result,
            selected_sensors=["Audio", "Vibration"],
            has_real_data=True,
        )
        st.session_state.reliability_v2 = rel2

        st.markdown("#### Reliability Engine 2.0")
        rr1, rr2, rr3, rr4 = st.columns(4)
        rr1.metric("Readiness Stage", rel2.get("readiness_stage", "Unknown"))
        rr2.metric("Production Risk", rel2.get("production_risk_level", "Unknown"))
        rr3.metric("Trust Score", f"{rel2.get('trust_score_v2', 0)}%")
        rr4.metric("Real Samples Needed", rel2.get("total_real_samples_needed", 0))

        if rel2.get("go_no_go") == "GO":
            st.success(rel2.get("decision", ""))
        elif rel2.get("go_no_go") == "CONDITIONAL":
            st.warning(rel2.get("decision", ""))
        else:
            st.error(rel2.get("decision", ""))

        class_risks = pd.DataFrame(rel2.get("class_risks", []))
        if len(class_risks) > 0:
            with st.expander("Per-class risk / real samples needed", expanded=True):
                st.dataframe(class_risks, use_container_width=True)

        sensor_scores = pd.DataFrame(rel2.get("sensor_value_scores", []))
        if len(sensor_scores) > 0:
            with st.expander("Sensor Value Score", expanded=False):
                st.dataframe(sensor_scores, use_container_width=True)

        risk = similarity.get("dataset_risk", "Unknown")
        verdict = similarity.get("verdict", "")
        if risk == "High":
            st.error(verdict)
        elif risk == "Medium":
            st.warning(verdict)
        else:
            st.success(verdict)

        st.caption(core.BRIDGE_DISCLAIMER)

        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Real samples needed")
            st.write(f"Target per class: **{sample_plan.get('target_per_class', 0)}**")
            st.write(f"Additional recommended total: **{sample_plan.get('total_needed', 0)}**")
            needed_df = pd.DataFrame(
                [{"Label": k, "Needed": v} for k, v in (sample_plan.get("needed_by_class", {}) or {}).items()]
            )
            if len(needed_df) > 0:
                st.dataframe(needed_df, use_container_width=True)

        with right:
            st.markdown("#### Comparable features")
            common = similarity.get("common_features", [])
            st.write(f"Common numeric features: **{len(common)}**")
            if common:
                st.caption(", ".join(common[:12]))
            if similarity.get("weak_labels"):
                st.warning("Weak labels: " + ", ".join(similarity.get("weak_labels", [])))

        fp_df = result.get("fingerprint_df", pd.DataFrame())
        var_df = result.get("variant_df", pd.DataFrame())
        train_df = result.get("bridge_training_df", pd.DataFrame())

        with st.expander("Real signal fingerprints", expanded=True):
            st.dataframe(fp_df.head(50), use_container_width=True)

        with st.expander("Real-based synthetic variants", expanded=False):
            st.dataframe(var_df.head(50), use_container_width=True)

        with st.expander("Bridge training dataset loaded into audit", expanded=False):
            st.dataframe(train_df.head(50), use_container_width=True)

        d1, d2 = st.columns(2)
        if isinstance(train_df, pd.DataFrame) and len(train_df) > 0:
            d1.download_button(
                "Download Bridge Training CSV",
                train_df.to_csv(index=False),
                file_name=f"{st.session_state.project_name}_bridge_training_dataset.csv",
                mime="text/csv",
                use_container_width=True,
                key="real_bridge_download_training_v21",
            )
        if st.session_state.real_bridge_bundle:
            d2.download_button(
                "Download Synthetic-to-Real Bundle",
                st.session_state.real_bridge_bundle,
                file_name=f"{st.session_state.project_name}_synthetic_to_real_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="real_bridge_download_bundle_v21",
            )
    else:
        st.info("Upload real WAV/CSV data to create the V21 bridge. This is the strongest trust feature before paid pilot bundles.")


# ============================================================
# V20.1 TRUST CENTER / COMMERCIAL READINESS
# ============================================================

with trust_tab:
    st.header("Trust Center + Reliability Engine 2.0")
    st.write(
        "V21.1 is de betrouwbare verkooplaag: naast demo/pilot readiness berekent de app nu ook "
        "per-class risico, sensor value, production risk en hoeveel echte samples nog nodig zijn."
    )

    if isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset) > 0:
        default_real_data = bool(st.session_state.real_bridge_result)
        if isinstance(st.session_state.auto_pilot_config, dict):
            default_real_data = default_real_data or bool(st.session_state.auto_pilot_config.get("has_real_data", False))

        has_real_data_for_gate = st.checkbox(
            "This project includes real field data",
            value=default_real_data,
            key="trust_has_real_data_v201",
            help="Laat dit alleen aan staan als er echte WAV/CSV velddata is gebruikt of geupload.",
        )

        doctor_for_gate = st.session_state.fusion_doctor if isinstance(st.session_state.fusion_doctor, dict) and st.session_state.fusion_doctor else None

        reliability_for_gate = None
        if isinstance(st.session_state.real_bridge_result, dict):
            reliability_for_gate = st.session_state.real_bridge_result.get("reliability")

        if reliability_for_gate is None and isinstance(st.session_state.auto_pilot_result, dict):
            reliability_for_gate = st.session_state.auto_pilot_result.get("reliability")

        if reliability_for_gate is None and doctor_for_gate:
            reliability_for_gate = core.calculate_reliability_score(
                doctor_for_gate,
                has_real_data=has_real_data_for_gate,
                selected_sensors=(st.session_state.fusion_manifest or {}).get("selected_sensors", []),
            )

        trust_gate = core.build_trust_gate(
            st.session_state.dataset,
            doctor=doctor_for_gate,
            reliability=reliability_for_gate,
            hardware_result=st.session_state.hardware_result,
            has_real_data=has_real_data_for_gate,
        )
        st.session_state.trust_gate = trust_gate

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trust Level", trust_gate.get("trust_level", "Unknown"))
        c2.metric("Data Quality", f"{trust_gate.get('data_quality_score', 0)}%")
        c3.metric("Reliability", f"{trust_gate.get('reliability_score', 0)}%")
        c4.metric("Hardware Fit", f"{trust_gate.get('hardware_fit_score', 0)}%")

        st.subheader("Decision gate")
        go_no_go = trust_gate.get("go_no_go", "Unknown")
        if "No-go" in go_no_go:
            st.error(go_no_go)
        elif "internal" in go_no_go.lower() or "demo" in go_no_go.lower():
            st.warning(go_no_go)
        else:
            st.success(go_no_go)

        st.info(trust_gate.get("production_status", ""))

        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Commercial packaging")
            st.write(f"**Suggested package:** {trust_gate.get('commercial_package', 'Unknown')}")
            st.write(f"**Suggested price range:** {trust_gate.get('suggested_price_range', 'Unknown')}")
            st.write(f"**Rows:** {trust_gate.get('rows', 0)}")
            st.write(f"**Classes:** {trust_gate.get('class_count', 0)}")
            st.write(f"**Weakest class:** {trust_gate.get('weakest_class', '-')}")

        with right:
            st.markdown("#### Red flags")
            flags = trust_gate.get("red_flags", [])
            if flags:
                for flag in flags:
                    st.warning(flag)
            else:
                st.success("No major red flags detected.")

        st.markdown("#### Recommended next steps")
        for step in trust_gate.get("next_steps", []):
            st.write(f"- {step}")

        selected_sensors_for_rel = []
        if isinstance(st.session_state.fusion_manifest, dict):
            selected_sensors_for_rel = st.session_state.fusion_manifest.get("selected_sensors", []) or []
        if not selected_sensors_for_rel and isinstance(st.session_state.auto_pilot_config, dict):
            selected_sensors_for_rel = st.session_state.auto_pilot_config.get("selected_sensors", []) or []

        rel2 = core.build_reliability_engine_v2(
            st.session_state.dataset,
            doctor=doctor_for_gate,
            reliability=reliability_for_gate,
            hardware_result=st.session_state.hardware_result,
            bridge_result=st.session_state.real_bridge_result,
            selected_sensors=selected_sensors_for_rel,
            has_real_data=has_real_data_for_gate,
        )
        st.session_state.reliability_v2 = rel2

        st.markdown("---")
        st.subheader("Reliability Engine 2.0")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Trust Score V2", f"{rel2.get('trust_score_v2', 0)}%")
        v2.metric("Readiness Stage", rel2.get("readiness_stage", "Unknown"))
        v3.metric("Production Risk", rel2.get("production_risk_level", "Unknown"))
        v4.metric("Real Samples Needed", rel2.get("total_real_samples_needed", 0))

        decision = rel2.get("decision", "")
        if rel2.get("go_no_go") == "GO":
            st.success(decision)
        elif rel2.get("go_no_go") == "CONDITIONAL":
            st.warning(decision)
        else:
            st.error(decision)

        cl, srcol = st.columns([1.25, 1])
        with cl:
            st.markdown("#### Per-class risk")
            class_risks = pd.DataFrame(rel2.get("class_risks", []))
            if len(class_risks) > 0:
                st.dataframe(class_risks, use_container_width=True)
            else:
                st.info("No class risk table available yet.")

        with srcol:
            st.markdown("#### Sensor Value Score")
            sensor_scores = pd.DataFrame(rel2.get("sensor_value_scores", []))
            if len(sensor_scores) > 0:
                st.dataframe(sensor_scores, use_container_width=True)
            else:
                st.info("No sensor value scores available yet.")

        with st.expander("Safe commercial claim language", expanded=False):
            st.markdown("**Allowed to say**")
            for claim in rel2.get("allowed_claims", []):
                st.write(f"- {claim}")
            st.markdown("**Do not claim yet**")
            for claim in rel2.get("blocked_claims", []):
                st.write(f"- {claim}")

        st.caption(core.RELIABILITY_DISCLAIMER)

        btrust, brel = st.columns(2)
        if btrust.button("Generate Trust Bundle", type="primary", use_container_width=True, key="trust_generate_bundle_v201"):
            st.session_state.trust_bundle = core.create_trust_bundle(
                st.session_state.project_name,
                st.session_state.dataset,
                trust_gate,
            )
            st.success("Trust Bundle generated.")

        if brel.button("Generate Reliability 2.0 Bundle", type="primary", use_container_width=True, key="rel_v2_generate_bundle_v211"):
            st.session_state.reliability_v2_bundle = core.create_reliability_v2_bundle(
                st.session_state.project_name,
                st.session_state.dataset,
                rel2,
            )
            st.success("Reliability 2.0 Bundle generated.")

        dtrust, drel = st.columns(2)
        if st.session_state.trust_bundle:
            dtrust.download_button(
                "Download Trust Bundle",
                st.session_state.trust_bundle,
                file_name=f"{st.session_state.project_name}_trust_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="trust_download_bundle_v201",
            )
        if st.session_state.reliability_v2_bundle:
            drel.download_button(
                "Download Reliability 2.0 Bundle",
                st.session_state.reliability_v2_bundle,
                file_name=f"{st.session_state.project_name}_reliability_v2_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="rel_v2_download_bundle_v211",
            )
    else:
        st.info("Generate or upload a dataset first. Then this tab tells you if it is demo-ready, pilot-ready or not billable yet.")



# ============================================================
# V22 DEPLOYMENT PLANNER
# ============================================================

with deployment_tab:
    st.header("Hardware BOM & Deployment Planner")
    st.write(
        "Turn the current dataset/use-case into a practical field-pilot plan: BOM, power budget, "
        "communication choice, enclosure target, maintenance interval, risks and validation steps."
    )

    defaults = core.infer_deployment_defaults(
        st.session_state.dataset,
        st.session_state.fusion_manifest,
        st.session_state.hardware_result,
    )

    c1, c2 = st.columns([1.2, 1])

    with c1:
        dep_environment = st.selectbox(
            "Deployment environment",
            core.get_environment_options(),
            index=core.get_environment_options().index(defaults.get("environment")) if defaults.get("environment") in core.get_environment_options() else 0,
            key="deployment_environment_v22",
        )

        dep_sensors = st.multiselect(
            "Sensors in deployment",
            core.get_sensor_options(),
            default=[s for s in defaults.get("selected_sensors", []) if s in core.get_sensor_options()],
            key="deployment_sensors_v22",
        )

        deployment_scale = st.selectbox(
            "Deployment scale",
            ["Pilot: 1-5 nodes", "Small rollout: 6-25 nodes", "Larger rollout: 25+ nodes"],
            index=0,
            key="deployment_scale_v22",
        )

        communication = st.selectbox(
            "Communication mode",
            ["LoRa / LoRaWAN", "WiFi / MQTT", "LTE / NB-IoT", "Wired Ethernet"],
            index=["LoRa / LoRaWAN", "WiFi / MQTT", "LTE / NB-IoT", "Wired Ethernet"].index(defaults.get("communication", "WiFi / MQTT")),
            key="deployment_comm_v22",
        )

    with c2:
        power_source = st.selectbox(
            "Power source",
            ["Mains + small backup", "Battery only", "Solar + battery", "Vehicle / machine power"],
            index=["Mains + small backup", "Battery only", "Solar + battery", "Vehicle / machine power"].index(defaults.get("power_source", "Mains + small backup")),
            key="deployment_power_v22",
        )

        enclosure_target = st.selectbox(
            "Enclosure target",
            ["Indoor / IP40", "Outdoor / IP65", "Outdoor harsh / IP67", "Industrial / IP67 + vibration mount"],
            index=["Indoor / IP40", "Outdoor / IP65", "Outdoor harsh / IP67", "Industrial / IP67 + vibration mount"].index(defaults.get("enclosure_target", "Outdoor / IP65")),
            key="deployment_enclosure_v22",
        )

        autonomy_days = st.number_input(
            "Autonomy target / backup days",
            min_value=1,
            max_value=90,
            value=int(defaults.get("autonomy_days", 7)),
            step=1,
            key="deployment_autonomy_v22",
        )

        dep_priority = st.radio(
            "Deployment priority",
            ["balanced", "low_power", "performance", "gateway"],
            index=["balanced", "low_power", "performance", "gateway"].index(defaults.get("priority", "balanced")),
            horizontal=True,
            key="deployment_priority_v22",
        )

        dep_sample_rate = st.number_input(
            "Sample rate for planning",
            min_value=1000,
            max_value=48000,
            value=int(defaults.get("sample_rate", st.session_state.sr)),
            step=1000,
            key="deployment_sample_rate_v22",
        )

    st.caption(core.DEPLOYMENT_DISCLAIMER)

    if st.button("Generate Deployment Plan", type="primary", use_container_width=True, key="deployment_generate_v22"):
        with st.spinner("Building deployment plan..."):
            plan = core.build_deployment_plan(
                project_name=st.session_state.project_name,
                dataset_df=st.session_state.dataset,
                manifest=st.session_state.fusion_manifest,
                hardware_result=st.session_state.hardware_result,
                selected_sensors=dep_sensors,
                environment=dep_environment,
                deployment_scale=deployment_scale,
                autonomy_days=autonomy_days,
                communication=communication,
                power_source=power_source,
                enclosure_target=enclosure_target,
                maintenance_profile="standard",
                priority=dep_priority,
                sample_rate=dep_sample_rate,
            )
            st.session_state.deployment_plan = plan
            st.session_state.deployment_bundle = core.create_deployment_bundle(
                st.session_state.project_name,
                plan,
                st.session_state.dataset,
            )
            st.session_state.hardware_result = {
                "recommendation": plan.get("hardware", {}).get("recommended_board"),
                "ranking": st.session_state.hardware_result.get("ranking", []) if isinstance(st.session_state.hardware_result, dict) else [],
                "reason": plan.get("hardware", {}).get("reason", ""),
            }
        st.success("Deployment plan generated.")

    if st.session_state.deployment_plan:
        plan = st.session_state.deployment_plan
        hw = plan.get("hardware", {})
        edge = plan.get("edge_settings", {})
        power = plan.get("power_budget", {})
        cost = plan.get("cost_estimate", {})

        st.subheader("Deployment Summary")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Readiness", plan.get("readiness", "Unknown"))
        d2.metric("Go / No-Go", plan.get("go_no_go", "Unknown"))
        d3.metric("Hardware", hw.get("recommended_board", "Unknown"))
        d4.metric("Nodes", plan.get("node_count_estimate", 0))

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Latency", f"{edge.get('estimated_latency_ms', 0)} ms")
        p2.metric("RAM", f"{edge.get('estimated_ram_kb', 0)} KB")
        p3.metric("Avg current", f"{power.get('average_current_ma', 0)} mA")
        p4.metric("Battery", f"{power.get('recommended_battery_mAh', 0)} mAh")

        st.info(plan.get("communication_plan", {}).get("payload_policy", ""))

        if plan.get("go_no_go") == "NO-GO":
            st.error("Deployment is not ready yet. Fix high-risk hardware, power or data issues first.")
        elif plan.get("go_no_go") == "CONDITIONAL":
            st.warning("Deployment-prep ready, but validate the listed risks before customer promises.")
        else:
            st.success("Good candidate for a controlled field pilot. Still validate before production rollout.")

        st.subheader("Budgetary BOM")
        bom_df = pd.DataFrame(plan.get("bom", []))
        if len(bom_df):
            st.dataframe(bom_df, use_container_width=True)
            fig_cost = px.bar(bom_df, x="component", y="line_max_eur", color="category", title="BOM cost estimate by item")
            st.plotly_chart(fig_cost, use_container_width=True)

        st.subheader("Power & Maintenance")
        st.write(power.get("notes", ""))
        st.write(f"**Estimated maintenance interval:** {power.get('maintenance_interval_estimate', 'Unknown')}")

        st.subheader("Deployment Risks")
        for risk in plan.get("deployment_risks", []):
            sev = risk.get("severity", "info")
            msg = f"{risk.get('risk')} — {risk.get('mitigation')}"
            if sev == "high":
                st.error(msg)
            elif sev == "medium":
                st.warning(msg)
            else:
                st.info(msg)

        st.subheader("Validation Plan")
        for step in plan.get("validation_plan", []):
            st.write(f"- {step}")

        if st.session_state.deployment_bundle:
            st.download_button(
                "Download Deployment Bundle",
                st.session_state.deployment_bundle,
                file_name=f"{st.session_state.project_name}_deployment_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="deployment_download_bundle_v22",
            )
    else:
        st.info("Generate a deployment plan after running a demo, wizard, real bridge or audit. This turns the dataset into a field-pilot plan.")


# ============================================================
# V23 PROFESSIONAL REPORTS 2.0
# ============================================================

with reports_tab:
    st.header("Professional Reports 2.0 • SaaS-light Monetization Gate")
    st.write(
        "Build a customer-ready report that combines dataset quality, Reliability Engine 2.0, Trust Center, "
        "Synthetic-to-Real status, hardware direction and deployment planning into one paid-consultancy style bundle."
    )

    has_dataset = isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset) > 0
    has_deployment = isinstance(st.session_state.deployment_plan, dict) and bool(st.session_state.deployment_plan)
    has_reliability = isinstance(st.session_state.reliability_v2, dict) and bool(st.session_state.reliability_v2)
    has_trust = isinstance(st.session_state.trust_gate, dict) and bool(st.session_state.trust_gate)
    has_bridge = isinstance(st.session_state.real_bridge_result, dict) and bool(st.session_state.real_bridge_result)

    status_cols = st.columns(5)
    status_cols[0].metric("Dataset", "Yes" if has_dataset else "No")
    status_cols[1].metric("Reliability V2", "Yes" if has_reliability else "No")
    status_cols[2].metric("Trust", "Yes" if has_trust else "No")
    status_cols[3].metric("Real Bridge", "Yes" if has_bridge else "No")
    status_cols[4].metric("Deployment", "Yes" if has_deployment else "No")

    if not has_dataset:
        st.warning("Generate or upload a dataset first. Reports 2.0 works best after the Wizard, Real Bridge, Trust Center and Deployment Planner have been run.")

    c1, c2 = st.columns([1.15, 1])
    with c1:
        report_type = st.selectbox(
            "Report type",
            core.get_professional_report_types(),
            index=0,
            key="reports_type_v23",
        )
        customer_name = st.text_input("Customer / organization name", value="Customer", key="reports_customer_v23")
        customer_problem = st.text_area(
            "Customer problem / context",
            value=st.session_state.last_demo_summary.get("problem", "Customer wants to move from sensor idea to a validated Edge AI pilot with less trial-and-error."),
            height=100,
            key="reports_problem_v23",
        )
    with c2:
        package_level = st.selectbox(
            "Package level",
            ["Starter Pilot", "Professional Pilot", "Enterprise Deployment", "Real-Data Pilot"],
            index=1,
            key="reports_package_level_v23",
        )
        audience = st.selectbox(
            "Audience",
            ["Executive / decision maker", "Technical team", "Mixed business + technical", "Investor / partner"],
            index=2,
            key="reports_audience_v23",
        )
        prepared_by = st.text_input("Prepared by", value="EdgeTwin Studio / OMEGA-X Engine", key="reports_prepared_by_v23")
        include_dataset_snapshot = st.checkbox("Include dataset CSV snapshot in ZIP", value=True, key="reports_include_dataset_v23")
        include_bom = st.checkbox("Include deployment BOM when available", value=True, key="reports_include_bom_v23")

    st.caption(core.REPORTS_DISCLAIMER)

    if st.button("Generate Professional Report Bundle", type="primary", use_container_width=True, key="reports_generate_bundle_v23"):
        with st.spinner("Building professional report package..."):
            snapshot = core.build_professional_report_snapshot(
                project_name=st.session_state.project_name,
                dataset_df=st.session_state.dataset,
                manifest=st.session_state.fusion_manifest,
                doctor=st.session_state.fusion_doctor,
                reliability_v2=st.session_state.reliability_v2,
                trust_gate=st.session_state.trust_gate,
                deployment_plan=st.session_state.deployment_plan,
                hardware_result=st.session_state.hardware_result,
                commercial_summary=st.session_state.last_demo_summary,
                real_bridge_result=st.session_state.real_bridge_result,
                report_type=report_type,
                package_level=package_level,
                customer_name=customer_name,
                customer_problem=customer_problem,
                audience=audience,
                prepared_by=prepared_by,
            )
            bundle = core.create_professional_report_bundle(
                st.session_state.project_name,
                snapshot,
                st.session_state.dataset if include_dataset_snapshot else None,
                st.session_state.deployment_plan if include_bom else None,
            )
            st.session_state.professional_report_snapshot = snapshot
            st.session_state.professional_report_bundle = bundle
        st.success("Professional report bundle generated.")

    if st.session_state.professional_report_snapshot:
        snap = st.session_state.professional_report_snapshot
        exec_summary = snap.get("executive_summary", {})
        readiness = snap.get("readiness", {})
        dataset_quality = snap.get("dataset_quality", {})
        commercial = snap.get("commercial_positioning", {})

        st.subheader("Report Preview")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Decision", readiness.get("go_no_go", "Unknown"))
        p2.metric("Readiness", readiness.get("stage", "Unknown"))
        p3.metric("Trust", f"{readiness.get('trust_score', 0)}%")
        p4.metric("Suggested package", commercial.get("suggested_package", "Unknown"))

        st.markdown("#### Executive summary")
        st.write(exec_summary.get("summary", ""))
        st.write(f"**Decision:** {exec_summary.get('decision', '')}")
        st.write(f"**Recommended next step:** {exec_summary.get('next_step', '')}")

        st.markdown("#### Dataset quality")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Rows", dataset_quality.get("rows", 0))
        q2.metric("Labels", dataset_quality.get("labels", 0))
        q3.metric("Features", dataset_quality.get("features", 0))
        q4.metric("Quality", f"{dataset_quality.get('overall_score', 0)}%")

        if commercial.get("allowed_claims"):
            st.markdown("#### Safe commercial claims")
            for claim in commercial.get("allowed_claims", [])[:5]:
                st.info(claim)

        if commercial.get("blocked_claims"):
            st.markdown("#### Claims to avoid")
            for claim in commercial.get("blocked_claims", [])[:5]:
                st.warning(claim)

        if st.session_state.professional_report_bundle:
            st.download_button(
                "Download Professional Report Bundle",
                st.session_state.professional_report_bundle,
                file_name=f"{st.session_state.project_name}_professional_report_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="reports_download_bundle_v23",
            )
    else:
        st.info("Run this after the Wizard/Real Bridge/Trust Center/Deployment Planner for the strongest paid report output.")




# ============================================================
# V24.1 PRODUCT HARDENING & VALIDATION SUITE
# ============================================================

with hardening_tab:
    st.header("Product Hardening & Validation Suite")
    st.write(
        "This is the internal readiness gate before paid pilots: dataset safety, launch blockers, release checklist, "
        "benchmark sanity tests and safe-to-sell guidance. Use this before you send anything to a real customer."
    )

    has_real_data = bool(st.session_state.real_bridge_result)
    dataset_df = st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset rows", len(dataset_df))
    c2.metric("Real bridge", "Yes" if has_real_data else "No")
    c3.metric("Deployment plan", "Yes" if st.session_state.deployment_plan else "No")
    c4.metric("Reports 2.0", "Yes" if st.session_state.professional_report_snapshot else "No")

    st.info(
        "V24.1 is intentionally strict. It may block paid claims even when the demo looks good. "
        "That is good: it protects trust and helps EdgeTwin feel professional."
    )

    if st.button("Run Product Hardening Scan", type="primary", use_container_width=True, key="hardening_run_scan_v241"):
        with st.spinner("Running product-readiness validation..."):
            st.session_state.hardening_snapshot = core.run_product_hardening_suite(
                st.session_state.project_name,
                dataset_df,
                doctor=st.session_state.fusion_doctor,
                reliability_v2=st.session_state.reliability_v2,
                trust_gate=st.session_state.trust_gate,
                deployment_plan=st.session_state.deployment_plan,
                professional_report_snapshot=st.session_state.professional_report_snapshot,
                monetization_snapshot=st.session_state.monetization_snapshot,
                has_real_data=has_real_data,
            )
            st.session_state.hardening_bundle = None
            st.session_state.beta_launch_bundle = None
        st.success("Hardening scan completed.")

    if st.session_state.hardening_snapshot:
        snap = st.session_state.hardening_snapshot
        st.subheader("Readiness result")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Product Readiness", f"{snap.get('product_readiness_score', 0)}%")
        r2.metric("Level", snap.get("readiness_level", "Unknown"))
        r3.metric("Launch Risk", snap.get("launch_risk", "Unknown"))
        r4.metric("Safe to sell", snap.get("safe_to_sell", "Unknown"))

        score = int(snap.get("product_readiness_score", 0))
        msg = snap.get("recommended_next_action", "")
        if score >= 88:
            st.success(msg)
        elif score >= 74:
            st.warning(msg)
        else:
            st.error(msg)

        st.caption(snap.get("disclaimer", ""))

        st.subheader("Score breakdown")
        breakdown = pd.DataFrame([snap.get("score_breakdown", {})])
        st.dataframe(breakdown, use_container_width=True)

        st.subheader("Blockers")
        blockers = pd.DataFrame(snap.get("blockers", []))
        if len(blockers):
            st.dataframe(blockers, use_container_width=True)
        else:
            st.success("No critical/high blockers detected by the V24.1 scan.")

        dscan = snap.get("dataset_scan", {}) or {}
        st.subheader("Dataset safety")
        ds1, ds2 = st.columns([1, 2])
        with ds1:
            st.json(dscan.get("summary", {}))
        with ds2:
            issues = pd.DataFrame(dscan.get("issues", []))
            if len(issues):
                st.dataframe(issues, use_container_width=True)

        st.subheader("Release checklist")
        checklist = pd.DataFrame(snap.get("release_checklist", []))
        if len(checklist):
            st.dataframe(checklist, use_container_width=True)

        st.subheader("Internal benchmark sanity tests")
        bench = pd.DataFrame(snap.get("benchmark_cases", []))
        if len(bench):
            st.dataframe(bench, use_container_width=True)

        with st.expander("Hardening rules and known limitations", expanded=False):
            st.markdown("#### Hardening rules")
            for item in snap.get("hardening_rules", []):
                st.write(f"- {item}")
            st.markdown("#### Known limitations")
            for item in snap.get("known_limitations", []):
                st.write(f"- {item}")

        if st.button("Create Product Hardening Bundle", type="primary", use_container_width=True, key="hardening_create_bundle_v241"):
            st.session_state.hardening_bundle = core.create_product_hardening_bundle(
                st.session_state.project_name,
                snap,
                dataset_df,
            )
            st.success("Product hardening bundle created.")

        if st.session_state.hardening_bundle:
            st.download_button(
                "Download Product Hardening Bundle",
                st.session_state.hardening_bundle,
                file_name=f"{st.session_state.project_name}_product_hardening_v24_1.zip",
                mime="application/zip",
                use_container_width=True,
                key="hardening_download_bundle_v241",
            )
    else:
        st.info("Run the hardening scan after generating a dataset and preferably after Trust Center, Real Bridge, Deployment Planner and Reports 2.0.")



# ============================================================
# V24.2 BETA LAUNCH READINESS
# ============================================================

with beta_launch_tab:
    st.header("Beta Launch Readiness")
    st.write(
        "This turns the product into something you can safely show in a controlled beta: landing-page copy, "
        "demo script, package cards, feedback questions, watermark rules and launch blockers."
    )

    bc1, bc2 = st.columns([1.15, 1])
    with bc1:
        target_segment = st.selectbox(
            "Target beta customer",
            core.get_beta_target_segments(),
            key="beta_target_segment_v242",
        )
    with bc2:
        beta_mode = st.selectbox(
            "Beta mode",
            ["Private demo", "Founder-led beta", "Controlled paid beta"],
            index=1,
            key="beta_mode_v242",
        )

    segment_profile = core.get_beta_target_segment(target_segment)
    st.info(segment_profile.get("pain", ""))
    st.caption(f"Suggested first paid offer: {segment_profile.get('best_paid_offer', 'Professional Pilot Bundle')}")

    snapshot = core.build_beta_launch_snapshot(
        st.session_state.project_name,
        target_segment=target_segment,
        beta_mode=beta_mode,
        dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        trust_gate=st.session_state.trust_gate,
        reliability_v2=st.session_state.reliability_v2,
        real_bridge_result=st.session_state.real_bridge_result,
        deployment_plan=st.session_state.deployment_plan,
        professional_report_snapshot=st.session_state.professional_report_snapshot,
        hardening_snapshot=st.session_state.hardening_snapshot,
        monetization_snapshot=st.session_state.monetization_snapshot,
    )
    st.session_state.beta_launch_snapshot = snapshot

    m1, m2, m3 = st.columns(3)
    m1.metric("Beta readiness", f"{snapshot.get('beta_readiness_score', 0)}%")
    m2.metric("Readiness", snapshot.get("readiness", "Unknown"))
    m3.metric("Target", snapshot.get("target_segment", "Unknown"))

    if snapshot.get("readiness") == "Paid beta ready":
        st.success(snapshot.get("launch_action", ""))
    elif snapshot.get("readiness") == "Founder-led beta ready":
        st.warning(snapshot.get("launch_action", ""))
    else:
        st.error(snapshot.get("launch_action", ""))

    st.subheader("Customer-facing positioning")
    st.markdown(f"### {snapshot.get('safe_public_tagline', '')}")
    st.write(snapshot.get("safe_positioning", ""))

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Demo script")
        for item in snapshot.get("demo_script", []):
            st.write(f"- {item}")
    with c2:
        st.markdown("#### Beta rules")
        for item in snapshot.get("beta_rules", []):
            st.write(f"- {item}")

    st.subheader("Launch checklist")
    checklist_df = pd.DataFrame(snapshot.get("launch_checklist", []))
    if len(checklist_df):
        st.dataframe(checklist_df, use_container_width=True)

    blockers = snapshot.get("blockers", []) or []
    if blockers:
        st.subheader("Launch blockers")
        for item in blockers:
            st.error(f"{item.get('area')} / {item.get('check')}: {item.get('next_step')}")
    else:
        st.success("No high/critical beta launch blockers detected by V24.2.")

    st.subheader("Package cards")
    package_df = pd.DataFrame(snapshot.get("package_cards", []))
    if len(package_df):
        st.dataframe(package_df, use_container_width=True)

    st.subheader("Landing-page copy")
    landing_df = pd.DataFrame(snapshot.get("landing_page_sections", []))
    if len(landing_df):
        st.dataframe(landing_df, use_container_width=True)

    st.subheader("Beta feedback questions")
    feedback_df = pd.DataFrame(snapshot.get("feedback_questions", []))
    if len(feedback_df):
        st.dataframe(feedback_df, use_container_width=True)

    with st.expander("Watermark / paid unlock policy", expanded=False):
        st.json(snapshot.get("watermark_policy", {}))
        st.markdown("#### Paid unlock preparation")
        for item in snapshot.get("paid_unlock_preparation", []):
            st.write(f"- {item}")

    if st.button("Create Beta Launch Bundle", type="primary", use_container_width=True, key="beta_launch_create_bundle_v242"):
        st.session_state.beta_launch_bundle = core.create_beta_launch_bundle(
            st.session_state.project_name,
            snapshot,
            st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        st.success("Beta Launch bundle created.")

    if st.session_state.beta_launch_bundle:
        st.download_button(
            "Download Beta Launch Bundle",
            st.session_state.beta_launch_bundle,
            file_name=f"{st.session_state.project_name}_beta_launch_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="beta_launch_download_bundle_v242",
        )

    st.caption(snapshot.get("disclaimer", ""))


# ============================================================
# V24 SAAS-LIGHT / MONETIZATION GATE
# ============================================================

with monetization_tab:
    st.header("SaaS-light & Monetization Gate")
    st.write(
        "This keeps monetization simple: plan levels, locked premium exports, package suggestions and safe pricing guidance. "
        "No payment provider is connected yet; this is the access/pricing layer you can later connect to Stripe."
    )

    plan_names = core.get_pricing_plans()
    st.session_state.selected_plan = st.selectbox(
        "Simulate customer plan",
        plan_names,
        index=plan_names.index(st.session_state.selected_plan) if st.session_state.selected_plan in plan_names else 0,
        key="monetization_selected_plan_v24",
    )

    plan = core.get_pricing_plan(st.session_state.selected_plan)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Monthly price", "Custom" if plan.get("monthly_price_eur") is None else f"€{plan.get('monthly_price_eur', 0)}")
    p2.metric("Projects", plan.get("project_limit", 0))
    p3.metric("Bundles/month", plan.get("monthly_bundle_limit", 0))
    p4.metric("Real uploads", plan.get("real_upload_limit", 0))
    st.info(plan.get("description", ""))
    st.caption(f"Best for: {plan.get('best_for', '')}")

    state_like = {
        "fusion_bundle": bool(st.session_state.fusion_bundle),
        "enterprise_bundle": bool(st.session_state.enterprise_bundle),
        "optimizer_result": bool(st.session_state.optimizer_result),
        "trust_gate": bool(st.session_state.trust_gate),
        "real_bridge_result": bool(st.session_state.real_bridge_result),
        "reliability_v2": bool(st.session_state.reliability_v2),
        "deployment_plan": bool(st.session_state.deployment_plan),
        "professional_report_snapshot": bool(st.session_state.professional_report_snapshot),
    }

    snapshot = core.build_monetization_snapshot(
        st.session_state.project_name,
        selected_plan=st.session_state.selected_plan,
        dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        trust_gate=st.session_state.trust_gate,
        reliability_v2=st.session_state.reliability_v2,
        real_bridge_result=st.session_state.real_bridge_result,
        deployment_plan=st.session_state.deployment_plan,
        professional_report_snapshot=st.session_state.professional_report_snapshot,
        state_like=state_like,
        usage={"bundles_used": 0, "real_uploads": 0},
    )

    st.session_state.monetization_snapshot = snapshot
    rec = snapshot.get("package_recommendation", {})

    st.subheader("Recommended package")
    r1, r2, r3 = st.columns(3)
    r1.metric("Package", rec.get("suggested_package", "Unknown"))
    r2.metric("Plan", rec.get("suggested_plan", "Unknown"))
    r3.metric("Price range", rec.get("suggested_price_range", "Unknown"))
    st.success(rec.get("reason", ""))

    st.subheader("Available outputs on this project")
    available_access = pd.DataFrame(snapshot.get("available_access", []))
    if len(available_access):
        st.dataframe(available_access, use_container_width=True)
    else:
        st.info("No outputs are available yet. Run the Wizard or a Demo first.")

    st.subheader("Full export access matrix")
    matrix = pd.DataFrame(snapshot.get("access_matrix", []))
    if len(matrix):
        st.dataframe(matrix, use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Safe sales positioning")
        for item in snapshot.get("safe_sales_positioning", []):
            st.write(f"- {item}")
    with c2:
        st.markdown("#### Next monetization steps")
        for item in snapshot.get("next_steps", []):
            st.write(f"- {item}")

    st.subheader("Price ladder")
    ladder = pd.DataFrame(snapshot.get("price_ladder", []))
    if len(ladder):
        st.dataframe(ladder, use_container_width=True)

    if st.button("Create Monetization Gate Bundle", type="primary", use_container_width=True, key="monetization_create_bundle_v24"):
        st.session_state.monetization_bundle = core.create_monetization_bundle(
            st.session_state.project_name,
            snapshot,
            st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        st.success("Monetization bundle created.")

    if st.session_state.monetization_bundle:
        st.download_button(
            "Download Monetization Gate Bundle",
            st.session_state.monetization_bundle,
            file_name=f"{st.session_state.project_name}_monetization_gate_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="monetization_download_bundle_v24",
        )

    st.warning("V24 does not take payments yet. It defines access rules and package logic first. That is safer before adding Stripe.")


# ============================================================
# V25 API AUTOMATION
# ============================================================

with api_tab:
    st.header("API Automation")
    st.write(
        "This is the integration layer: API endpoint blueprint, access rules, sample requests, safe responses and a downloadable API bundle. "
        "It is designed for private beta and enterprise integrations before exposing a public production API."
    )

    if "api_key" in st.session_state.user:
        api_key = st.session_state.user.get("api_key", "")
        shown_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 14 else "Available"
        st.info(f"Current account API key: {shown_key}. Keep full keys private and rotate them before production launch.")

    plan_names = core.get_pricing_plans()
    api_plan = st.selectbox(
        "Simulate API access plan",
        plan_names,
        index=plan_names.index(st.session_state.selected_plan) if st.session_state.selected_plan in plan_names else 0,
        key="api_selected_plan_v25",
    )
    st.session_state.selected_plan = api_plan

    api_mode = st.radio(
        "API launch mode",
        ["Local blueprint", "Private beta API", "Enterprise/on-premise"],
        horizontal=True,
        key="api_launch_mode_v25",
    )

    state_like = {
        "fusion_bundle": bool(st.session_state.fusion_bundle),
        "enterprise_bundle": bool(st.session_state.enterprise_bundle),
        "optimizer_result": bool(st.session_state.optimizer_result),
        "trust_gate": bool(st.session_state.trust_gate),
        "real_bridge_result": bool(st.session_state.real_bridge_result),
        "reliability_v2": bool(st.session_state.reliability_v2),
        "deployment_plan": bool(st.session_state.deployment_plan),
        "professional_report_snapshot": bool(st.session_state.professional_report_snapshot),
        "hardening_snapshot": bool(st.session_state.hardening_snapshot),
        "beta_launch_snapshot": bool(st.session_state.beta_launch_snapshot),
        "monetization_snapshot": bool(st.session_state.monetization_snapshot),
    }

    snapshot = core.build_api_automation_snapshot(
        st.session_state.project_name,
        selected_plan=api_plan,
        api_mode=api_mode,
        dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        trust_gate=st.session_state.trust_gate,
        reliability_v2=st.session_state.reliability_v2,
        real_bridge_result=st.session_state.real_bridge_result,
        deployment_plan=st.session_state.deployment_plan,
        hardening_snapshot=st.session_state.hardening_snapshot,
        beta_launch_snapshot=st.session_state.beta_launch_snapshot,
        monetization_snapshot=st.session_state.monetization_snapshot,
        state_like=state_like,
    )
    st.session_state.api_automation_snapshot = snapshot

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("API readiness", f"{snapshot.get('api_readiness_score', 0)}%")
    a2.metric("Access", snapshot.get("access_level", "Unknown"))
    a3.metric("Mode", snapshot.get("api_mode", "Unknown"))
    a4.metric("Integration risk", snapshot.get("integration_risk", "Unknown"))

    if snapshot.get("api_readiness_score", 0) >= 80:
        st.success(snapshot.get("api_verdict", ""))
    elif snapshot.get("api_readiness_score", 0) >= 60:
        st.warning(snapshot.get("api_verdict", ""))
    else:
        st.error(snapshot.get("api_verdict", ""))

    st.subheader("Endpoint catalog")
    endpoints_df = pd.DataFrame(snapshot.get("endpoint_catalog", []))
    if len(endpoints_df):
        st.dataframe(endpoints_df, use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### API safety rules")
        for item in snapshot.get("api_safety_rules", []):
            st.write(f"- {item}")
    with c2:
        st.markdown("#### Implementation checklist")
        for item in snapshot.get("implementation_checklist", [])[:10]:
            st.write(f"- {item}")

    st.subheader("Sample request / response simulator")
    endpoint_names = [e.get("endpoint") for e in snapshot.get("endpoint_catalog", [])]
    selected_endpoint = st.selectbox(
        "Endpoint",
        endpoint_names,
        index=0 if endpoint_names else None,
        key="api_endpoint_select_v25",
    ) if endpoint_names else None

    if selected_endpoint:
        sample_payload = core.get_api_sample_payload(selected_endpoint)
        payload_text = st.text_area(
            "JSON payload",
            value=core.safe_json_dumps(sample_payload, indent=2),
            height=220,
            key="api_payload_text_v25",
        )
        if st.button("Simulate API response", type="primary", use_container_width=True, key="api_simulate_response_v25"):
            try:
                payload = core.parse_json_payload(payload_text)
                st.session_state.api_simulation_response = core.simulate_api_endpoint(
                    selected_endpoint,
                    payload,
                    dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
                    project_name=st.session_state.project_name,
                    selected_plan=api_plan,
                )
            except Exception as e:
                st.session_state.api_simulation_response = {"ok": False, "error": str(e)}

        if st.session_state.api_simulation_response:
            st.json(st.session_state.api_simulation_response)

    st.subheader("API documentation preview")
    st.markdown(core.generate_api_docs_markdown(snapshot))

    if st.button("Create API Automation Bundle", type="primary", use_container_width=True, key="api_create_bundle_v25"):
        st.session_state.api_automation_bundle = core.create_api_automation_bundle(
            st.session_state.project_name,
            snapshot,
            st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        st.success("API Automation bundle created.")

    if st.session_state.api_automation_bundle:
        st.download_button(
            "Download API Automation Bundle",
            st.session_state.api_automation_bundle,
            file_name=f"{st.session_state.project_name}_api_automation_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="api_download_bundle_v25",
        )

    st.warning("V25 is API-ready architecture, not a public production API yet. Keep first integrations private, logged and rate-limited.")


# ============================================================
# V26 INDUSTRY PACK MARKETPLACE / PACK BUILDER
# ============================================================

with marketplace_tab:
    st.header("Industry Pack Marketplace & Pack Builder")
    st.write(
        "This turns EdgeTwin Studio into a reusable pack system: packaged use-case knowledge, default labels, sensors, "
        "pricing position, export rights and dataset-generation settings. Keep it curated first: deep packs beat many shallow packs."
    )

    st.info("V26 is a marketplace blueprint and internal pack builder. Payments/licensing are simulated locally; use this to validate which packs are worth selling first.")

    mc1, mc2, mc3 = st.columns([1, 1, 1])
    with mc1:
        marketplace_plan = st.selectbox(
            "Customer plan",
            core.get_pricing_plans(),
            index=core.get_pricing_plans().index(st.session_state.selected_plan) if st.session_state.selected_plan in core.get_pricing_plans() else 0,
            key="marketplace_plan_v26",
        )
    with mc2:
        target_market = st.selectbox(
            "Target market",
            ["Predictive maintenance", "Security / tamper", "Forestry / remote area", "Agriculture", "Remote assets", "Industrial gateway", "Custom"],
            key="marketplace_target_market_v26",
        )
    with mc3:
        pack_depth = st.radio("Pack strategy", ["Curated paid packs", "Custom internal pack", "Founder test"], horizontal=False, key="marketplace_strategy_v26")

    catalog = core.get_marketplace_pack_catalog()
    catalog_df = pd.DataFrame(catalog)

    st.subheader("Curated marketplace catalog")
    st.dataframe(catalog_df, use_container_width=True)

    state_like = {
        "has_dataset": isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset) > 0,
        "has_real_bridge": bool(st.session_state.real_bridge_result),
        "has_reliability_v2": bool(st.session_state.reliability_v2),
        "has_deployment_plan": bool(st.session_state.deployment_plan),
        "has_professional_report": bool(st.session_state.professional_report_snapshot),
        "has_api_plan": bool(st.session_state.api_automation_snapshot),
    }

    recommendations = core.recommend_marketplace_packs(
        target_market=target_market,
        selected_plan=marketplace_plan,
        dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        trust_gate=st.session_state.trust_gate,
        reliability_v2=st.session_state.reliability_v2,
        real_bridge_result=st.session_state.real_bridge_result,
        deployment_plan=st.session_state.deployment_plan,
        state_like=state_like,
    )

    st.subheader("Recommended packs")
    rec_df = pd.DataFrame(recommendations.get("recommendations", []))
    if len(rec_df):
        st.dataframe(rec_df, use_container_width=True)
        fig = px.bar(rec_df, x="pack_id", y="fit_score", title="Marketplace pack fit score")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No recommendations available yet.")

    st.markdown("---")
    st.subheader("Generate pack dataset")
    pack_names = [p.get("pack_id") for p in catalog]
    selected_pack_id = st.selectbox("Pack", pack_names, key="marketplace_selected_pack_id_v26")
    samples_per_class = st.number_input("Samples per class", 10, 2000, 80, 10, key="marketplace_samples_per_class_v26")

    selected_pack = core.get_marketplace_pack(selected_pack_id)
    if selected_pack:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Tier", selected_pack.get("tier", "Unknown"))
        p2.metric("Min plan", selected_pack.get("minimum_plan", "Unknown"))
        p3.metric("Suggested price", selected_pack.get("price_range", "Unknown"))
        p4.metric("Classes", len(selected_pack.get("classes", [])))
        st.write(selected_pack.get("description", ""))
        st.caption("Sensors: " + ", ".join(selected_pack.get("sensors", [])))

    if st.button("Generate Marketplace Pack Dataset", type="primary", use_container_width=True, key="marketplace_generate_dataset_v26"):
        with st.spinner("Generating marketplace pack dataset..."):
            df, manifest = core.generate_marketplace_pack_dataset(selected_pack_id, samples_per_class=samples_per_class)
            st.session_state.dataset = df
            st.session_state.marketplace_generated_dataset = df.copy()
            st.session_state.fusion_manifest = manifest
            st.session_state.pack_marketplace_snapshot = core.build_pack_marketplace_snapshot(
                st.session_state.project_name,
                selected_pack_id=selected_pack_id,
                selected_plan=marketplace_plan,
                target_market=target_market,
                dataset_df=df,
                trust_gate=st.session_state.trust_gate,
                reliability_v2=st.session_state.reliability_v2,
                real_bridge_result=st.session_state.real_bridge_result,
                deployment_plan=st.session_state.deployment_plan,
                recommendations=recommendations,
            )
            reset_generated_bundles()
        st.success("Marketplace pack dataset generated and loaded into Enterprise Audit.")

    st.markdown("---")
    with st.expander("Build a custom pack definition", expanded=False):
        custom_name = st.text_input("Custom pack name", value="Custom Audio Vibration Pack", key="marketplace_custom_name_v26")
        custom_description = st.text_area(
            "Description",
            value="Reusable Edge AI pilot pack for a focused audio/vibration sensor use-case.",
            height=80,
            key="marketplace_custom_description_v26",
        )
        custom_use_case = st.selectbox("Use-case type", core.get_use_case_types(), key="marketplace_custom_use_case_v26")
        custom_sensors = st.multiselect(
            "Sensors",
            core.get_sensor_options(),
            default=["Audio", "Vibration"],
            key="marketplace_custom_sensors_v26",
        )
        custom_labels = st.text_area(
            "Classes / labels",
            value="Normal, Warning, Event, Critical",
            height=80,
            key="marketplace_custom_labels_v26",
        )
        custom_sr = st.number_input("Sample rate", 1000, 48000, 16000, 1000, key="marketplace_custom_sr_v26")
        custom_price_tier = st.selectbox("Commercial tier", ["Free Demo", "Starter", "Professional", "Real-Data Pilot", "Enterprise"], index=2, key="marketplace_custom_price_tier_v26")
        license_model = st.selectbox("License model", ["Single project", "Reusable pack", "Team pack", "Enterprise/on-premise"], key="marketplace_custom_license_v26")

        custom_def = core.build_custom_marketplace_pack(
            name=custom_name,
            description=custom_description,
            use_case_type=custom_use_case,
            sensors=custom_sensors,
            labels_text=custom_labels,
            sample_rate=custom_sr,
            price_tier=custom_price_tier,
            license_model=license_model,
        )
        validation = core.validate_marketplace_pack_definition(custom_def)
        st.session_state.custom_pack_definition = custom_def

        v1, v2, v3 = st.columns(3)
        v1.metric("Pack quality", f"{validation.get('quality_score', 0)}%")
        v2.metric("Commercial tier", custom_def.get("price_tier", "Unknown"))
        v3.metric("Labels", len(custom_def.get("classes", [])))

        if validation.get("quality_score", 0) >= 75:
            st.success(validation.get("verdict", ""))
        elif validation.get("quality_score", 0) >= 55:
            st.warning(validation.get("verdict", ""))
        else:
            st.error(validation.get("verdict", ""))

        for issue in validation.get("issues", []):
            st.write(f"- {issue}")

        if st.button("Generate Custom Pack Dataset", type="primary", use_container_width=True, key="marketplace_generate_custom_pack_v26"):
            with st.spinner("Generating custom pack dataset..."):
                df, manifest = core.generate_custom_marketplace_pack_dataset(custom_def, samples_per_class=samples_per_class)
                st.session_state.dataset = df
                st.session_state.marketplace_generated_dataset = df.copy()
                st.session_state.fusion_manifest = manifest
                st.session_state.pack_marketplace_snapshot = core.build_pack_marketplace_snapshot(
                    st.session_state.project_name,
                    selected_pack_id=custom_def.get("pack_id"),
                    selected_plan=marketplace_plan,
                    target_market=target_market,
                    dataset_df=df,
                    trust_gate=st.session_state.trust_gate,
                    reliability_v2=st.session_state.reliability_v2,
                    real_bridge_result=st.session_state.real_bridge_result,
                    deployment_plan=st.session_state.deployment_plan,
                    recommendations=recommendations,
                    custom_pack_definition=custom_def,
                    custom_pack_validation=validation,
                )
                reset_generated_bundles()
            st.success("Custom pack dataset generated and loaded into Enterprise Audit.")

    if st.session_state.pack_marketplace_snapshot:
        snapshot = st.session_state.pack_marketplace_snapshot
        st.markdown("---")
        st.subheader("Marketplace readiness snapshot")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Marketplace score", f"{snapshot.get('marketplace_readiness_score', 0)}%")
        s2.metric("Pack status", snapshot.get("pack_status", "Unknown"))
        s3.metric("Commercial fit", snapshot.get("commercial_fit", "Unknown"))
        s4.metric("Minimum plan", snapshot.get("minimum_plan", "Unknown"))

        if snapshot.get("marketplace_readiness_score", 0) >= 80:
            st.success(snapshot.get("verdict", ""))
        elif snapshot.get("marketplace_readiness_score", 0) >= 60:
            st.warning(snapshot.get("verdict", ""))
        else:
            st.error(snapshot.get("verdict", ""))

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### What this pack should include")
            for item in snapshot.get("pack_assets", []):
                st.write(f"- {item}")
        with c2:
            st.markdown("#### Launch risks")
            for item in snapshot.get("launch_risks", []):
                st.write(f"- {item}")

        if st.button("Create Marketplace Pack Bundle", type="primary", use_container_width=True, key="marketplace_create_bundle_v26"):
            st.session_state.pack_marketplace_bundle = core.create_marketplace_pack_bundle(
                st.session_state.project_name,
                snapshot,
                st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            )
            st.success("Marketplace Pack bundle created.")

    if st.session_state.pack_marketplace_bundle:
        st.download_button(
            "Download Marketplace Pack Bundle",
            st.session_state.pack_marketplace_bundle,
            file_name=f"{st.session_state.project_name}_marketplace_pack_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="marketplace_download_bundle_v26",
        )

    st.warning("Start with 3 deep packs: Predictive Maintenance, Acoustic Tamper, and Audio+Vibration Fusion. A small trusted catalog is stronger than a huge shallow marketplace.")


# ============================================================
# V26.1 NORMAL VS ABNORMAL BASELINE ENGINE
# ============================================================

with normality_tab:
    st.header("Normal vs Abnormal Baseline Engine • Edge Impulse Export • Edge Impulse Classifier Export")
    st.write(
        "This is the practical layer for customers: define what normal looks like, then score which samples are normal-like, "
        "watch-level, abnormal or critical abnormal. It helps turn sensor data into an understandable pilot threshold plan."
    )

    st.info("Use this as pilot preparation. A clean normal/baseline class is the most important input. Production thresholds still need field validation.")

    if not isinstance(st.session_state.dataset, pd.DataFrame) or len(st.session_state.dataset) == 0:
        st.warning("Generate or upload a dataset first. Best sources: Use Case Wizard, Real Bridge, Marketplace Pack or Industry Pack.")
    elif "Label" not in st.session_state.dataset.columns:
        st.warning("The current dataset needs a Label column before normality analysis can run.")
    else:
        df = st.session_state.dataset.copy()
        labels = sorted([str(x) for x in df["Label"].dropna().unique().tolist()])
        detected_labels, detection_notes = core.detect_normal_labels(df)
        feature_cols = core.get_normality_feature_columns(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Samples", len(df))
        c2.metric("Labels", len(labels))
        c3.metric("Detected normal labels", len(detected_labels))
        c4.metric("Usable features", len(feature_cols))

        for note in detection_notes:
            st.caption(note)

        left, right = st.columns([1.2, 1])
        with left:
            normal_labels = st.multiselect(
                "Which labels should count as normal / baseline?",
                labels,
                default=[x for x in detected_labels if x in labels],
                key="normality_selected_normal_labels_v261",
                help="For machine health this is usually Healthy/Normal/Baseline. For security it is Normal_Background or Normal.",
            )
            st.caption("Tip: if this is wrong, the whole normal/abnormal decision becomes unreliable. Always confirm the baseline labels.")
        with right:
            sensitivity = st.slider(
                "Baseline tolerance",
                0.5,
                2.0,
                1.0,
                0.05,
                key="normality_sensitivity_v261",
                help="Lower = stricter abnormal detection. Higher = more tolerant baseline.",
            )
            apply_scored = st.checkbox("Load scored dataset back into Enterprise Audit after analysis", value=False, key="normality_apply_scored_v261")

        with st.expander("Normality feature columns", expanded=False):
            st.write(feature_cols)

        if st.button("Analyze Normal vs Abnormal", type="primary", use_container_width=True, key="normality_run_v261"):
            with st.spinner("Building baseline profile and scoring normality..."):
                result = core.run_normality_baseline_engine(
                    df,
                    normal_labels=normal_labels,
                    sensitivity=sensitivity,
                    project_context={
                        "project_name": st.session_state.project_name,
                        "selected_template": st.session_state.selected_template,
                    },
                )
                st.session_state.normality_result = result
                st.session_state.normality_bundle = None
                if apply_scored and result.get("valid") and isinstance(result.get("scored_dataset"), pd.DataFrame):
                    st.session_state.dataset = result["scored_dataset"].copy()
                    reset_generated_bundles()
            st.success("Normal vs Abnormal analysis completed.")

    if st.session_state.normality_result:
        result = st.session_state.normality_result
        summary = result.get("summary", {})
        profile = result.get("baseline_profile", {})

        st.markdown("---")
        st.subheader("Normality result")

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Normality Score", f"{summary.get('normality_score', 0)}%")
        n2.metric("Abnormal Rate", f"{summary.get('abnormal_rate_pct', 0)}%")
        n3.metric("Baseline Confidence", f"{summary.get('baseline_confidence', 0)}%")
        n4.metric("Decision", summary.get("decision", "Unknown"))

        decision = summary.get("decision", "Unknown")
        if decision == "GO":
            st.success(summary.get("verdict", ""))
        elif decision == "NO-GO":
            st.error(summary.get("verdict", ""))
        else:
            st.warning(summary.get("verdict", ""))

        st.caption("Normal labels: " + ", ".join(profile.get("normal_labels", [])))
        for issue in profile.get("issues", []):
            st.warning(issue)

        p1, p2 = st.columns([1, 1])
        with p1:
            st.markdown("#### Per-label normality")
            per_label = pd.DataFrame(summary.get("per_label_summary", []))
            if len(per_label):
                st.dataframe(per_label, use_container_width=True)
                fig = px.bar(per_label, x="label", y="mean_normality_score", title="Mean normality score per label")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No per-label summary available.")
        with p2:
            st.markdown("#### Top deviation features")
            top_dev = pd.DataFrame(summary.get("top_deviation_features", []))
            if len(top_dev):
                st.dataframe(top_dev, use_container_width=True)
                fig2 = px.bar(top_dev, x="feature", y="avg_deviation_z", title="Top deviation features")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No deviation features available.")

        st.markdown("#### Recommended next steps")
        for step in summary.get("recommended_next_steps", []):
            st.write(f"- {step}")

        scored_df = result.get("scored_dataset")
        if isinstance(scored_df, pd.DataFrame) and len(scored_df):
            st.markdown("#### Scored dataset preview")
            st.dataframe(scored_df.head(80), use_container_width=True)
            st.download_button(
                "Download scored normality CSV",
                scored_df.to_csv(index=False),
                file_name=f"{st.session_state.project_name}_normality_scored_dataset.csv",
                mime="text/csv",
                use_container_width=True,
                key="normality_download_scored_csv_v261",
            )

        if st.button("Create Normality Bundle", type="primary", use_container_width=True, key="normality_create_bundle_v261"):
            st.session_state.normality_bundle = core.create_normality_baseline_bundle(
                st.session_state.project_name,
                st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
                result,
            )
            st.success("Normality bundle created.")

    if st.session_state.normality_bundle:
        st.download_button(
            "Download Normality Bundle",
            st.session_state.normality_bundle,
            file_name=f"{st.session_state.project_name}_normality_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="normality_download_bundle_v261",
        )

    st.warning("This engine explains normal-like versus abnormal relative to a selected baseline. It does not replace real field validation or safety certification.")


# ============================================================
# V26.2 EDGE IMPULSE ANOMALY EXPORT
# ============================================================

with edge_impulse_tab:
    st.header("Edge Impulse Anomaly Export")
    st.write(
        "Prepare the current EdgeTwin dataset for an Edge Impulse anomaly workflow. "
        "This is designed for the practical K-means style flow: learn normal behavior first, then flag deviations."
    )

    st.info(
        "Best use: run the Normality Engine first, confirm the true normal/baseline labels, then export the normal-only training CSV and all-label evaluation CSV."
    )

    if not isinstance(st.session_state.dataset, pd.DataFrame) or len(st.session_state.dataset) == 0:
        st.warning("Generate, upload or bridge a dataset first. Best sources: Real Bridge, Normality Engine, Auto Pilot or Marketplace Pack.")
    elif "Label" not in st.session_state.dataset.columns:
        st.warning("The current dataset needs a Label column before Edge Impulse export can run.")
    else:
        df = st.session_state.dataset.copy()
        labels = sorted([str(x) for x in df["Label"].dropna().unique().tolist()])
        detected_labels, detection_notes = core.detect_normal_labels(df)
        previous_normal = []
        if isinstance(st.session_state.normality_result, dict):
            previous_normal = st.session_state.normality_result.get("baseline_profile", {}).get("normal_labels", []) or []
        default_normal = [x for x in previous_normal if x in labels] or [x for x in detected_labels if x in labels]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Samples", len(df))
        c2.metric("Labels", len(labels))
        c3.metric("Detected normal labels", len(default_normal))
        c4.metric("Numeric features", len(core.get_edge_impulse_feature_columns(df)))

        for note in detection_notes[:3]:
            st.caption(note)

        left, right = st.columns([1.2, 1])
        with left:
            normal_labels = st.multiselect(
                "Normal / baseline labels for anomaly training",
                labels,
                default=default_normal,
                key="ei_normal_labels_v262",
                help="For K-means anomaly detection, train mostly on normal behavior. Abnormal labels are best used for validation/evaluation.",
            )
            workflow = st.selectbox(
                "Export workflow",
                [
                    "K-means anomaly baseline",
                    "Classification + anomaly layer",
                    "Feature CSV only",
                ],
                index=0,
                key="ei_workflow_v262",
            )
        with right:
            suggested_k = core.suggest_edge_impulse_k(len(df[df["Label"].astype(str).isin(normal_labels)]), len(normal_labels)) if normal_labels else 8
            k_clusters = st.number_input(
                "Suggested K clusters",
                min_value=4,
                max_value=128,
                value=int(suggested_k),
                step=4,
                key="ei_k_clusters_v262",
                help="Edge Impulse still lets you tune this. This is only a starting recommendation.",
            )
            max_axes = st.slider("Max anomaly axes", 2, 12, 6, 1, key="ei_max_axes_v262")
            include_derived = st.checkbox(
                "Include EdgeTwin derived scores as export columns",
                value=False,
                key="ei_include_derived_v262",
                help="Usually keep this off for Edge Impulse training; use raw signal/features where possible. Derived scores are useful for reports and evaluation.",
            )

        if st.button("Prepare Edge Impulse Export", type="primary", use_container_width=True, key="ei_prepare_export_v262"):
            with st.spinner("Preparing Edge Impulse anomaly export package..."):
                snapshot = core.build_edge_impulse_anomaly_export_snapshot(
                    project_name=st.session_state.project_name,
                    dataset_df=df,
                    normal_labels=normal_labels,
                    workflow=workflow,
                    k_clusters=int(k_clusters),
                    max_axes=int(max_axes),
                    normality_result=st.session_state.normality_result,
                    include_derived_scores=include_derived,
                )
                st.session_state.edge_impulse_snapshot = snapshot
                st.session_state.edge_impulse_bundle = core.create_edge_impulse_anomaly_bundle(st.session_state.project_name, snapshot)
            st.success("Edge Impulse export prepared.")

    if st.session_state.edge_impulse_snapshot:
        snapshot = st.session_state.edge_impulse_snapshot
        summary = snapshot.get("summary", {})
        files = snapshot.get("files", {})

        st.markdown("---")
        st.subheader("Edge Impulse export result")

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Export Readiness", f"{summary.get('export_readiness_score', 0)}%")
        e2.metric("Normal Training Samples", summary.get("normal_training_samples", 0))
        e3.metric("Recommended K", summary.get("recommended_k", 0))
        e4.metric("Decision", summary.get("decision", "Unknown"))

        decision = summary.get("decision", "Unknown")
        if decision == "GO":
            st.success(summary.get("verdict", ""))
        elif decision == "NO-GO":
            st.error(summary.get("verdict", ""))
        else:
            st.warning(summary.get("verdict", ""))

        st.markdown("#### Recommended anomaly axes")
        axes_df = pd.DataFrame(summary.get("recommended_axes", []))
        if len(axes_df):
            st.dataframe(axes_df, use_container_width=True)
        else:
            st.info("No recommended axes available yet.")

        st.markdown("#### Edge Impulse steps")
        for step in summary.get("edge_impulse_steps", []):
            st.write(f"- {step}")

        st.markdown("#### Warnings")
        for warning in summary.get("warnings", []):
            st.warning(warning)

        d1, d2, d3, d4 = st.columns(4)
        if isinstance(files.get("normal_training_csv"), pd.DataFrame):
            d1.download_button(
                "Download EI normal training CSV",
                files["normal_training_csv"].to_csv(index=False),
                file_name=f"{st.session_state.project_name}_edge_impulse_normal_training.csv",
                mime="text/csv",
                use_container_width=True,
                key="ei_download_normal_csv_v262",
            )
        if isinstance(files.get("evaluation_csv"), pd.DataFrame):
            d2.download_button(
                "Download EI evaluation CSV",
                files["evaluation_csv"].to_csv(index=False),
                file_name=f"{st.session_state.project_name}_edge_impulse_evaluation.csv",
                mime="text/csv",
                use_container_width=True,
                key="ei_download_eval_csv_v262",
            )
        d3.download_button(
            "Download EI instructions",
            files.get("instructions_md", ""),
            file_name=f"{st.session_state.project_name}_edge_impulse_instructions.md",
            mime="text/markdown",
            use_container_width=True,
            key="ei_download_instructions_v262",
        )
        if st.session_state.edge_impulse_bundle:
            d4.download_button(
                "Download EI Export Bundle",
                st.session_state.edge_impulse_bundle,
                file_name=f"{st.session_state.project_name}_edge_impulse_anomaly_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="ei_download_bundle_v262",
            )

        if isinstance(files.get("normal_training_csv"), pd.DataFrame):
            st.markdown("#### Normal training CSV preview")
            st.dataframe(files["normal_training_csv"].head(50), use_container_width=True)

    st.warning("Use EdgeTwin as pilot preparation. Edge Impulse K-means thresholds and axes still need real-field validation before production deployment.")


# ============================================================
# V26.3 EDGE IMPULSE CLASSIFIER EXPORT
# ============================================================

with ei_classifier_tab:
    st.header("Edge Impulse Classifier Export")
    st.write(
        "Prepare labelled EdgeTwin datasets for supervised Edge Impulse classification. "
        "Use this when you want the model to distinguish classes such as Normal, Bearing_Wear, Drilling, Impact or Critical_Tamper."
    )

    st.info(
        "Use V26.2 for K-means anomaly baseline. Use this tab for labelled event classification with train/test CSV exports and feature recommendations."
    )

    if not isinstance(st.session_state.dataset, pd.DataFrame) or len(st.session_state.dataset) == 0:
        st.warning("Generate, upload or bridge a dataset first. Good sources: Real Bridge, Auto Pilot, Normality Engine or Marketplace Pack.")
    elif "Label" not in st.session_state.dataset.columns:
        st.warning("The current dataset needs a Label column before classifier export can run.")
    else:
        df = st.session_state.dataset.copy()
        labels = sorted([str(x) for x in df["Label"].dropna().unique().tolist()])
        feature_cols = core.get_edge_impulse_classifier_feature_columns(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Samples", len(df))
        c2.metric("Labels", len(labels))
        c3.metric("Classifier features", len(feature_cols))
        c4.metric("Smallest class", int(df["Label"].value_counts().min()) if len(labels) else 0)

        left, right = st.columns([1.2, 1])
        with left:
            workflow = st.selectbox(
                "Classifier workflow",
                [
                    "Supervised event classification",
                    "Machine health classification",
                    "Security/tamper classification",
                    "Classifier + anomaly validation",
                ],
                index=0,
                key="ei_classifier_workflow_v263",
            )
            min_samples = st.number_input(
                "Minimum samples per class for GO decision",
                min_value=5,
                max_value=200,
                value=20,
                step=5,
                key="ei_classifier_min_samples_v263",
            )
        with right:
            test_size = st.slider("Holdout/test split", 0.10, 0.40, 0.20, 0.05, key="ei_classifier_test_size_v263")
            max_features = st.slider("Max exported features", 3, 30, 12, 1, key="ei_classifier_max_features_v263")
            include_derived = st.checkbox(
                "Include EdgeTwin derived scores",
                value=False,
                key="ei_classifier_include_derived_v263",
                help="Usually keep this off for model training. Derived scores are useful for evaluation/reporting, but raw features are safer for deployment.",
            )

        if st.button("Prepare Classifier Export", type="primary", use_container_width=True, key="ei_classifier_prepare_v263"):
            with st.spinner("Preparing Edge Impulse classifier export package..."):
                snapshot = core.build_edge_impulse_classifier_export_snapshot(
                    project_name=st.session_state.project_name,
                    dataset_df=df,
                    workflow=workflow,
                    test_size=float(test_size),
                    include_derived_scores=include_derived,
                    min_samples_per_class=int(min_samples),
                    max_features=int(max_features),
                )
                st.session_state.edge_impulse_classifier_snapshot = snapshot
                st.session_state.edge_impulse_classifier_bundle = core.create_edge_impulse_classifier_bundle(st.session_state.project_name, snapshot)
            st.success("Edge Impulse classifier export prepared.")

    if st.session_state.edge_impulse_classifier_snapshot:
        snapshot = st.session_state.edge_impulse_classifier_snapshot
        summary = snapshot.get("summary", {})
        files = snapshot.get("files", {})

        st.markdown("---")
        st.subheader("Classifier export result")

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Readiness", f"{summary.get('export_readiness_score', 0)}%")
        e2.metric("Train samples", summary.get("train_samples", 0))
        e3.metric("Test samples", summary.get("test_samples", 0))
        e4.metric("Decision", summary.get("decision", "Unknown"))

        decision = summary.get("decision", "Unknown")
        if decision == "GO":
            st.success(summary.get("verdict", ""))
        elif decision == "NO-GO":
            st.error(summary.get("verdict", ""))
        else:
            st.warning(summary.get("verdict", ""))

        class_df = files.get("class_summary_csv")
        if isinstance(class_df, pd.DataFrame) and len(class_df):
            st.markdown("#### Class readiness")
            st.dataframe(class_df, use_container_width=True)

        ranking_df = pd.DataFrame(summary.get("recommended_features", []))
        if len(ranking_df):
            st.markdown("#### Feature ranking for classifier")
            st.dataframe(ranking_df[[c for c in ["feature", "classifier_value_score", "reason"] if c in ranking_df.columns]], use_container_width=True)

        st.markdown("#### Edge Impulse steps")
        for step in summary.get("edge_impulse_steps", []):
            st.write(f"- {step}")

        for warning in summary.get("warnings", []):
            st.warning(warning)

        d1, d2, d3, d4 = st.columns(4)
        if isinstance(files.get("train_csv"), pd.DataFrame):
            d1.download_button(
                "Download classifier train CSV",
                files["train_csv"].to_csv(index=False),
                file_name=f"{st.session_state.project_name}_ei_classifier_train.csv",
                mime="text/csv",
                use_container_width=True,
                key="ei_classifier_download_train_v263",
            )
        if isinstance(files.get("test_csv"), pd.DataFrame):
            d2.download_button(
                "Download classifier test CSV",
                files["test_csv"].to_csv(index=False),
                file_name=f"{st.session_state.project_name}_ei_classifier_test.csv",
                mime="text/csv",
                use_container_width=True,
                key="ei_classifier_download_test_v263",
            )
        d3.download_button(
            "Download classifier instructions",
            files.get("instructions_md", ""),
            file_name=f"{st.session_state.project_name}_ei_classifier_instructions.md",
            mime="text/markdown",
            use_container_width=True,
            key="ei_classifier_download_instructions_v263",
        )
        if st.session_state.edge_impulse_classifier_bundle:
            d4.download_button(
                "Download Classifier Export Bundle",
                st.session_state.edge_impulse_classifier_bundle,
                file_name=f"{st.session_state.project_name}_edge_impulse_classifier_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="ei_classifier_download_bundle_v263",
            )

        if isinstance(files.get("train_csv"), pd.DataFrame):
            st.markdown("#### Training CSV preview")
            st.dataframe(files["train_csv"].head(50), use_container_width=True)

    st.warning("Classifier exports are for pilot preparation. Real field validation and Edge Impulse training metrics are still required before deployment claims.")



# ============================================================
# RELEASE SUCCESS GATE / ONE-CLICK CUSTOMER READINESS
# ============================================================

with success_gate_tab:
    st.header("Release Success Gate • Golden Demo Suite • Closed Beta Launch Kit • Paid Export Gate • Real Field Validation • Edge Deployment Starter • Storage/Scalability • Operational Control Center • Production Safety & Error Observatory")
    st.write(
        "This is the final one-click gate before showing EdgeTwin Studio output to a beta customer. "
        "It combines data quality, trust evidence, hardening, reports, Edge Impulse export readiness, "
        "deployment planning and monetization positioning into one honest release decision."
    )

    intended_offer = st.selectbox(
        "Intended offer level",
        ["Free demo", "Private beta", "Paid pilot", "Real-data pilot", "Enterprise review"],
        index=2,
        key="success_gate_intended_offer_v264",
    )

    customer_type = st.selectbox(
        "Target customer type",
        ["Solo maker / hobby", "Small business", "Industrial pilot team", "Enterprise innovation team", "Internal OMEGA-X validation"],
        index=2,
        key="success_gate_customer_type_v264",
    )

    auto_refresh = st.checkbox(
        "Auto-build missing internal evidence where possible",
        value=True,
        key="success_gate_auto_refresh_v264",
        help="Creates lightweight internal snapshots for hardening/beta/monetization when they are missing. It does not fake real-data evidence.",
    )

    if st.button("Run Release Success Gate", type="primary", use_container_width=True, key="run_success_gate_v264"):
        with st.spinner("Running final release readiness checks..."):
            if auto_refresh:
                if not st.session_state.hardening_snapshot:
                    st.session_state.hardening_snapshot = core.run_product_hardening_suite(
                        st.session_state.project_name,
                        dataset_df=st.session_state.dataset,
                        trust_gate=st.session_state.trust_gate,
                        deployment_plan=st.session_state.deployment_plan,
                        professional_report_snapshot=st.session_state.professional_report_snapshot,
                        monetization_snapshot=st.session_state.monetization_snapshot,
                        has_real_data=bool(st.session_state.real_bridge_result),
                    )
                if not st.session_state.beta_launch_snapshot:
                    st.session_state.beta_launch_snapshot = core.build_beta_launch_snapshot(
                        st.session_state.project_name,
                        dataset_df=st.session_state.dataset,
                        trust_gate=st.session_state.trust_gate,
                        real_bridge_result=st.session_state.real_bridge_result,
                        deployment_plan=st.session_state.deployment_plan,
                        professional_report_snapshot=st.session_state.professional_report_snapshot,
                        hardening_snapshot=st.session_state.hardening_snapshot,
                        monetization_snapshot=st.session_state.monetization_snapshot,
                    )
                if not st.session_state.monetization_snapshot:
                    st.session_state.monetization_snapshot = core.build_monetization_snapshot(
                        st.session_state.project_name,
                        selected_plan=st.session_state.selected_plan,
                        dataset_df=st.session_state.dataset,
                        trust_gate=st.session_state.trust_gate,
                        reliability_v2=st.session_state.reliability_v2,
                        real_bridge_result=st.session_state.real_bridge_result,
                    )

            snapshot = core.build_release_success_gate_snapshot(
                project_name=st.session_state.project_name,
                dataset_df=st.session_state.dataset,
                intended_offer=intended_offer,
                customer_type=customer_type,
                selected_plan=st.session_state.selected_plan,
                trust_gate=st.session_state.trust_gate,
                reliability_v2=st.session_state.reliability_v2,
                real_bridge_result=st.session_state.real_bridge_result,
                deployment_plan=st.session_state.deployment_plan,
                professional_report_snapshot=st.session_state.professional_report_snapshot,
                hardening_snapshot=st.session_state.hardening_snapshot,
                beta_launch_snapshot=st.session_state.beta_launch_snapshot,
                monetization_snapshot=st.session_state.monetization_snapshot,
                normality_result=st.session_state.normality_result,
                edge_impulse_snapshot=st.session_state.edge_impulse_snapshot,
                edge_impulse_classifier_snapshot=st.session_state.edge_impulse_classifier_snapshot,
            )
            st.session_state.release_success_snapshot = snapshot
            st.session_state.release_success_bundle = core.create_release_success_gate_bundle(
                st.session_state.project_name,
                snapshot,
                st.session_state.dataset,
            )
        st.success("Release Success Gate completed.")

    snapshot = st.session_state.release_success_snapshot
    if snapshot:
        summary = snapshot.get("summary", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Success Score", f"{summary.get('success_score', 0)}%")
        c2.metric("Decision", summary.get("decision", "Unknown"))
        c3.metric("Commercial Status", summary.get("commercial_status", "Unknown"))
        c4.metric("Risk Level", summary.get("risk_level", "Unknown"))

        decision = summary.get("decision", "")
        if decision == "GO":
            st.success(summary.get("verdict", ""))
        elif decision == "CONDITIONAL GO":
            st.warning(summary.get("verdict", ""))
        else:
            st.error(summary.get("verdict", ""))

        st.markdown("#### Score breakdown")
        breakdown = pd.DataFrame(snapshot.get("score_breakdown", []))
        if len(breakdown) > 0:
            st.dataframe(breakdown, use_container_width=True)
            fig = px.bar(breakdown, x="area", y="score", title="Release success score breakdown")
            st.plotly_chart(fig, use_container_width=True)

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("#### Must-fix blockers")
            blockers = snapshot.get("blockers", [])
            if blockers:
                for item in blockers:
                    st.error(item)
            else:
                st.success("No must-fix blockers detected by the Success Gate.")

            st.markdown("#### Recommended next actions")
            for item in snapshot.get("recommended_actions", []):
                st.info(item)

        with b2:
            st.markdown("#### Evidence coverage")
            evidence = pd.DataFrame(snapshot.get("evidence", []))
            if len(evidence) > 0:
                st.dataframe(evidence, use_container_width=True)

            st.markdown("#### Safe customer wording")
            st.write(summary.get("safe_customer_wording", ""))

        st.markdown("#### Claims check")
        claims_df = pd.DataFrame(snapshot.get("claims_check", []))
        if len(claims_df) > 0:
            st.dataframe(claims_df, use_container_width=True)

        if st.session_state.release_success_bundle:
            st.download_button(
                "Download Release Success Bundle",
                st.session_state.release_success_bundle,
                file_name=f"{st.session_state.project_name}_release_success_gate_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_release_success_bundle_v264",
            )
    else:
        st.info("Run the Success Gate after generating at least one dataset. For a paid pilot, also run Trust Center, Reports 2.0, Hardening and one Edge Impulse export route.")



# ============================================================
# V27 GOLDEN DEMO SUITE / CUSTOMER PAIN PROOF MATRIX
# ============================================================

with golden_demo_tab:
    st.header("Golden Demo Suite + Customer Pain Proof Matrix")
    st.write(
        "This is the one-click proof flow. It runs a complete EdgeTwin pilot chain and then explains which real customer problems were solved, "
        "which evidence was generated, and whether the package is safe to show as a demo or paid pilot candidate."
    )

    scenario_ids = core.get_golden_demo_scenarios()
    scenario_titles = {sid: core.get_golden_demo_scenario(sid).get("title", sid) for sid in scenario_ids}

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        selected_scenario = st.selectbox(
            "Golden demo scenario",
            scenario_ids,
            format_func=lambda x: scenario_titles.get(x, x),
            key="golden_demo_scenario_v27",
        )
    with c2:
        golden_offer = st.selectbox(
            "Intended offer",
            ["Free demo", "Private beta", "Paid pilot", "Real-data pilot", "Enterprise review"],
            index=2,
            key="golden_demo_offer_v27",
        )
    with c3:
        golden_plan = st.selectbox(
            "Plan context",
            ["Free Demo", "Starter", "Professional Pilot", "Real-Data Pilot", "Enterprise", "Founder Test Mode"],
            index=5,
            key="golden_demo_plan_v27",
        )

    scenario = core.get_golden_demo_scenario(selected_scenario)
    st.info(scenario.get("customer_problem", ""))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Use case", scenario.get("use_case_type", "Custom"))
    s2.metric("Sensors", len(scenario.get("selected_sensors", [])))
    s3.metric("Classes", len(scenario.get("classes", [])))
    s4.metric("Sample rate", f"{scenario.get('sample_rate', 0)} Hz")

    with st.expander("What this proves", expanded=False):
        st.write("Target customer:", scenario.get("target_customer", ""))
        st.write("Sales angle:", scenario.get("sales_angle", ""))
        st.write("Normal labels:", ", ".join(scenario.get("normal_labels", [])))
        st.write("Classes:", ", ".join(scenario.get("classes", [])))
        st.warning("Golden demos prove the end-to-end pilot-preparation flow. They do not replace real field validation.")

    run_col, load_col = st.columns([2, 1])
    with run_col:
        if st.button("Run Golden Demo Suite", type="primary", use_container_width=True, key="run_golden_demo_suite_v27"):
            with st.spinner("Running full end-to-end pilot proof flow..."):
                result = core.run_golden_demo_suite(
                    selected_scenario,
                    selected_plan=golden_plan,
                    intended_offer=golden_offer,
                    include_optimizer=True,
                    include_edge_impulse=True,
                )
                st.session_state.golden_demo_result = result
                st.session_state.golden_demo_bundle = core.create_golden_demo_bundle(
                    result.get("summary", {}).get("project_name", st.session_state.project_name),
                    result,
                )
                # Load the result back into the main workspace so all existing tabs can inspect it.
                st.session_state.project_name = result.get("summary", {}).get("project_name", st.session_state.project_name)
                st.session_state.dataset = result.get("dataset", pd.DataFrame())
                artifacts = result.get("artifacts", {})
                st.session_state.fusion_df = result.get("fusion_dataset", pd.DataFrame())
                st.session_state.fusion_manifest = artifacts.get("pilot", {}).get("manifest", {}) if isinstance(artifacts.get("pilot"), dict) else {}
                st.session_state.fusion_doctor = artifacts.get("pilot", {}).get("doctor", {}) if isinstance(artifacts.get("pilot"), dict) else {}
                st.session_state.fusion_training_df = result.get("dataset", pd.DataFrame())
                st.session_state.hardware_result = artifacts.get("pilot", {}).get("hardware") if isinstance(artifacts.get("pilot"), dict) else None
                st.session_state.trust_gate = artifacts.get("trust_gate")
                st.session_state.reliability_v2 = artifacts.get("reliability_v2")
                st.session_state.normality_result = artifacts.get("normality_result")
                st.session_state.deployment_plan = artifacts.get("deployment_plan")
                st.session_state.professional_report_snapshot = artifacts.get("professional_report_snapshot")
                st.session_state.hardening_snapshot = artifacts.get("hardening_snapshot")
                st.session_state.beta_launch_snapshot = artifacts.get("beta_launch_snapshot")
                st.session_state.monetization_snapshot = artifacts.get("monetization_snapshot")
                st.session_state.edge_impulse_snapshot = artifacts.get("edge_impulse_snapshot")
                st.session_state.edge_impulse_classifier_snapshot = artifacts.get("edge_impulse_classifier_snapshot")
                st.session_state.release_success_snapshot = artifacts.get("release_success_snapshot")
            st.success("Golden Demo Suite completed and loaded into the workspace.")

    with load_col:
        if st.session_state.golden_demo_bundle:
            st.download_button(
                "Download Golden Demo Bundle",
                st.session_state.golden_demo_bundle,
                file_name=f"{st.session_state.project_name}_golden_demo_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_golden_demo_bundle_v27",
            )

    result = st.session_state.golden_demo_result
    if result:
        summary = result.get("summary", {})
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Golden Demo Score", f"{summary.get('golden_demo_score', 0)}%")
        r2.metric("Decision", summary.get("decision", "Unknown"))
        r3.metric("Proof Matrix", f"{summary.get('proof_matrix_score', 0)}%")
        r4.metric("Release Success", f"{summary.get('release_success_score', 0)}%")

        if summary.get("decision") == "GO":
            st.success(summary.get("customer_wording", ""))
        elif summary.get("decision") == "CONDITIONAL GO":
            st.warning(summary.get("customer_wording", ""))
        else:
            st.error(summary.get("customer_wording", ""))

        st.markdown("#### End-to-end proof steps")
        steps_df = pd.DataFrame(result.get("steps", []))
        if len(steps_df) > 0:
            st.dataframe(steps_df, use_container_width=True)
            fig = px.bar(steps_df, x="step", y="score", color="status", title="Golden demo end-to-end step scores")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Customer Pain → EdgeTwin Proof Matrix")
        proof_df = pd.DataFrame(result.get("pain_proof_matrix", []))
        if len(proof_df) > 0:
            st.dataframe(proof_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Dataset preview loaded into workspace")
            dataset = result.get("dataset", pd.DataFrame())
            if isinstance(dataset, pd.DataFrame) and len(dataset) > 0:
                st.dataframe(dataset.head(50), use_container_width=True)
        with c2:
            st.markdown("#### What to say safely")
            st.write(summary.get("customer_wording", ""))
            st.warning(summary.get("disclaimer", ""))

    else:
        st.info("Run one Golden Demo to prove the full chain: customer problem → pilot package → proof matrix → release decision.")



# ============================================================
# V28 CLOSED BETA LAUNCH KIT
# ============================================================

with closed_beta_tab:
    st.header("Closed Beta Launch Kit • Paid Export Gate • Real Field Validation • Edge Deployment Starter • Storage/Scalability • Operational Control Center • Production Safety & Error Observatory")
    st.write(
        "Use this after the Golden Demo and Success Gate. It prepares a controlled beta plan, invite email, demo script, "
        "feedback questions, safe claims and package recommendation so you can test EdgeTwin with a small number of trusted users."
    )

    beta_c1, beta_c2 = st.columns([1.15, 1])

    with beta_c1:
        target_segment = st.selectbox(
            "Target beta segment",
            core.get_closed_beta_segments(),
            key="closed_beta_target_segment_v28",
        )
        segment_info = core.get_closed_beta_segment(target_segment)
        st.info(segment_info.get("pain", ""))
        beta_goal = st.text_area(
            "Beta goal",
            value="Validate whether EdgeTwin Studio helps a real customer move from sensor idea to pilot-ready package faster.",
            height=95,
            key="closed_beta_goal_v28",
        )

    with beta_c2:
        selected_offer = st.selectbox(
            "Offer to test",
            [
                "Free Feedback Beta",
                "Starter Pilot Bundle",
                "Professional Pilot Bundle",
                "Real-Data Pilot Bundle",
                "Enterprise Review Candidate",
            ],
            index=2,
            key="closed_beta_offer_v28",
        )
        max_beta_users = st.number_input(
            "Max beta users",
            min_value=1,
            max_value=25,
            value=5,
            step=1,
            key="closed_beta_max_users_v28",
        )
        st.markdown("#### Segment promise")
        st.write(segment_info.get("promise", ""))
        st.caption(f"Best matching demo: {segment_info.get('best_demo', 'Unknown')}")

    st.markdown("---")
    st.caption("Best practice: run a Golden Demo first, then run the Success Gate, then create this Closed Beta Kit.")

    if st.button("Create Closed Beta Launch Kit", type="primary", use_container_width=True, key="create_closed_beta_kit_v28"):
        with st.spinner("Building closed beta launch kit..."):
            kit = core.build_closed_beta_launch_kit(
                project_name=st.session_state.project_name,
                target_segment=target_segment,
                selected_offer=selected_offer,
                beta_goal=beta_goal,
                max_beta_users=max_beta_users,
                dataset_df=st.session_state.dataset,
                golden_demo_result=st.session_state.golden_demo_result,
                release_success_snapshot=st.session_state.release_success_snapshot,
                hardening_snapshot=st.session_state.hardening_snapshot,
                monetization_snapshot=st.session_state.monetization_snapshot,
                professional_report_snapshot=st.session_state.professional_report_snapshot,
            )
            st.session_state.closed_beta_kit = kit
            st.session_state.closed_beta_bundle = core.create_closed_beta_launch_bundle(st.session_state.project_name, kit)
        st.success("Closed Beta Launch Kit created.")

    kit = st.session_state.closed_beta_kit
    if kit:
        summary = kit.get("summary", {})
        package = summary.get("package_recommendation", {}) or {}
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Beta Readiness", f"{summary.get('beta_readiness_score', 0)}%")
        k2.metric("Decision", summary.get("decision", "Unknown"))
        k3.metric("Max Users", summary.get("max_beta_users", 0))
        k4.metric("Package", package.get("recommended_package", "Unknown"))

        decision = summary.get("decision", "")
        if decision == "READY FOR CONTROLLED BETA":
            st.success(package.get("positioning", ""))
        elif decision == "LIMITED BETA ONLY":
            st.warning(package.get("positioning", ""))
        else:
            st.error(package.get("positioning", ""))

        if st.session_state.closed_beta_bundle:
            st.download_button(
                "Download Closed Beta Launch Bundle",
                st.session_state.closed_beta_bundle,
                file_name=f"{st.session_state.project_name}_closed_beta_launch_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_closed_beta_bundle_v28",
            )

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("#### Evidence")
            evidence_df = pd.DataFrame(kit.get("evidence", []))
            if len(evidence_df) > 0:
                st.dataframe(evidence_df, use_container_width=True)
            if kit.get("blockers"):
                st.markdown("#### Blockers")
                for item in kit.get("blockers", []):
                    st.error(item)
            if kit.get("warnings"):
                st.markdown("#### Warnings")
                for item in kit.get("warnings", []):
                    st.warning(item)
        with e2:
            st.markdown("#### Invite email")
            st.text_area("Copy/paste beta invite", kit.get("invite_email", ""), height=300, key="closed_beta_invite_preview_v28")

        st.markdown("#### Demo script")
        demo_df = pd.DataFrame(kit.get("demo_script", []))
        if len(demo_df) > 0:
            st.dataframe(demo_df, use_container_width=True)

        st.markdown("#### Feedback questions")
        feedback_df = pd.DataFrame(kit.get("feedback_questions", []))
        if len(feedback_df) > 0:
            st.dataframe(feedback_df, use_container_width=True)

        st.markdown("#### 4-week beta plan")
        week_df = pd.DataFrame(kit.get("week_plan", []))
        if len(week_df) > 0:
            st.dataframe(week_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Safe claims")
            for claim in kit.get("safe_claims", []):
                st.success(claim)
        with c2:
            st.markdown("#### Claims to avoid")
            for claim in kit.get("claims_to_avoid", []):
                st.warning(claim)
    else:
        st.info("Create a Closed Beta Launch Kit after running a Golden Demo. This helps you approach first testers without overpromising.")


# ============================================================
# V29 PAID EXPORT & LICENSE GATE
# ============================================================

with paid_export_tab:
    st.header("Paid Export & License Gate")
    st.write(
        "Manual paid-delivery control for beta/commercial bundles. This does not process payments yet; "
        "it creates entitlement metadata, delivery scope, safe-use terms and locked/unlocked export evidence."
    )

    c1, c2 = st.columns([1.15, 1])
    with c1:
        customer_name = st.text_input("Customer / company name", "Beta Customer", key="paid_customer_name_v29")
        customer_email = st.text_input("Customer email", "customer@example.com", key="paid_customer_email_v29")
        package_options = list(getattr(core, "PAID_PACKAGE_EXPORTS", {"Professional Pilot Bundle": []}).keys())
        requested_package = st.selectbox("Requested package", package_options, index=package_options.index("Professional Pilot Bundle") if "Professional Pilot Bundle" in package_options else 0, key="paid_requested_package_v29")
    with c2:
        plan_options = core.get_paid_license_plans() if hasattr(core, "get_paid_license_plans") else core.get_pricing_plans()
        selected_paid_plan = st.selectbox("Selected customer plan", plan_options, index=plan_options.index(st.session_state.selected_plan) if st.session_state.selected_plan in plan_options else 0, key="paid_selected_plan_v29")
        checkout_mode = st.selectbox("Checkout / delivery mode", ["Manual invoice", "Founder internal", "Stripe later", "Partner delivery"], index=0, key="paid_checkout_mode_v29")
        watermark_free = st.checkbox("Watermark free demo exports", value=True, key="paid_watermark_v29")

    st.caption(core.PAID_LICENSE_DISCLAIMER if hasattr(core, "PAID_LICENSE_DISCLAIMER") else "Manual paid export control. Not a payment processor.")

    if st.button("Create Paid Export License Gate", type="primary", use_container_width=True, key="create_paid_license_gate_v29"):
        snap = core.build_paid_export_license_gate(
            st.session_state.project_name,
            customer_name=customer_name,
            customer_email=customer_email,
            requested_package=requested_package,
            selected_plan=selected_paid_plan,
            checkout_mode=checkout_mode,
            watermark_free_exports=watermark_free,
            dataset_df=st.session_state.dataset,
            monetization_snapshot=st.session_state.monetization_snapshot,
            release_success_snapshot=st.session_state.release_success_snapshot,
            professional_report_snapshot=st.session_state.professional_report_snapshot,
            closed_beta_kit=st.session_state.closed_beta_kit,
        )
        st.session_state.paid_license_snapshot = snap
        st.session_state.paid_license_bundle = core.create_paid_export_license_bundle(st.session_state.project_name, snap, st.session_state.dataset)
        st.success("Paid Export License Gate created.")

    if st.session_state.paid_license_snapshot:
        snap = st.session_state.paid_license_snapshot
        summary = snap.get("summary", {}) or {}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Commercial readiness", f"{summary.get('commercial_readiness_score', 0)}%")
        m2.metric("Decision", summary.get("decision", "Unknown"))
        m3.metric("License status", summary.get("license_status", "Unknown"))
        m4.metric("Price position", summary.get("price_position", "Unknown"))
        st.info(summary.get("customer_message", ""))

        if st.session_state.paid_license_bundle:
            st.download_button(
                "Download Paid Export License Bundle",
                st.session_state.paid_license_bundle,
                file_name=f"{st.session_state.project_name}_paid_export_license_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_paid_license_bundle_v29",
            )

        a1, a2 = st.columns(2)
        with a1:
            st.markdown("#### Entitlement manifest")
            st.json(snap.get("entitlement_manifest", {}))
            if snap.get("must_fix_before_charging"):
                st.markdown("#### Must fix before charging")
                for item in snap.get("must_fix_before_charging", []):
                    st.error(item)
        with a2:
            st.markdown("#### Access matrix")
            access_df = pd.DataFrame(snap.get("access_matrix", []))
            if len(access_df) > 0:
                st.dataframe(access_df, use_container_width=True)
            st.markdown("#### Receipt preview")
            st.text_area("Delivery note", snap.get("receipt_preview", ""), height=260, key="paid_receipt_preview_v29")
    else:
        st.info("Create this gate before sending paid bundle files. Founder Test Mode is internal only, not a real customer entitlement.")


# ============================================================
# V30 REAL FIELD VALIDATION PACK
# ============================================================

with field_validation_tab:
    st.header("Real Field Validation Pack")
    st.write(
        "Use this when you have real WAV/CSV evidence from a machine, site or remote asset. "
        "It checks field evidence coverage, normal/event labels, field-vs-synthetic similarity and what still must be collected."
    )

    scenario_options = core.get_field_validation_scenarios()
    c1, c2 = st.columns([1.1, 1])
    with c1:
        scenario = st.selectbox("Validation scenario", scenario_options, key="field_validation_scenario_v30")
        field_environment = st.selectbox("Field environment", core.get_environment_options(), index=2 if "Industrial" in core.get_environment_options() else 0, key="field_environment_v30")
        device_setup = st.text_area("Device / sensor setup", "ESP32-S3 / RAK node with audio and vibration sensors. Feature-level validation only.", height=90, key="field_device_setup_v30")
    with c2:
        planned_days = st.number_input("Planned / completed field days", 1, 60, 7, 1, key="field_planned_days_v30")
        minimum_files = st.number_input("Minimum real files target", 1, 500, 20, 1, key="field_min_files_v30")
        use_current_dataset_as_field = st.checkbox("Use current dataset as temporary field evidence", value=False, key="field_use_current_dataset_v30")

    with st.expander("Upload real field WAV/CSV files", expanded=False):
        st.caption("Labels are inferred from filenames when possible. Use names like normal_01.wav, bearing_wear_02.csv, drilling_03.wav.")
        uploaded_field_files = st.file_uploader("Real field files", type=["wav", "csv"], accept_multiple_files=True, key="field_validation_uploads_v30")
        fallback_label = st.text_input("Fallback label if filename is unclear", "Normal_Field", key="field_fallback_label_v30")

        def _infer_field_label(name, fallback):
            txt = str(name or "").lower()
            if any(k in txt for k in ["normal", "healthy", "baseline", "background", "idle", "calm"]):
                return "Normal_Field"
            if any(k in txt for k in ["bearing", "wear", "failure", "fault", "unbalance", "misalignment"]):
                return "Bearing_Wear_Field"
            if any(k in txt for k in ["drill", "grind", "cut", "tool", "tamper", "impact"]):
                return "Tamper_Event_Field"
            if any(k in txt for k in ["chain", "vehicle", "human", "threat"]):
                return "Remote_Event_Field"
            return fallback

        if uploaded_field_files:
            rows = []
            for up in uploaded_field_files:
                try:
                    features = core.extract_features_from_bytes(up.read(), up.name, st.session_state.sr)
                    if "error" not in features:
                        features = dict(features)
                        features["Label"] = _infer_field_label(up.name, fallback_label)
                        features["Filename"] = up.name
                        rows.append(features)
                    else:
                        st.warning(f"{up.name}: {features.get('error')}")
                except Exception as exc:
                    st.warning(f"{up.name}: {exc}")
            if rows:
                st.session_state.field_validation_df = pd.DataFrame(rows)
                st.success(f"Loaded {len(rows)} field feature rows.")
                st.dataframe(st.session_state.field_validation_df.head(30), use_container_width=True)

    field_df = st.session_state.field_validation_df
    if use_current_dataset_as_field and isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset) > 0:
        field_df = st.session_state.dataset.copy()
        st.info("Using current dataset as temporary field evidence. For real paid claims, upload actual field files.")

    st.caption(core.FIELD_VALIDATION_DISCLAIMER if hasattr(core, "FIELD_VALIDATION_DISCLAIMER") else "Field validation is not production certification.")

    if st.button("Build Real Field Validation Pack", type="primary", use_container_width=True, key="build_field_validation_pack_v30"):
        snap = core.build_real_field_validation_pack(
            st.session_state.project_name,
            scenario,
            field_environment,
            device_setup,
            planned_days,
            minimum_files,
            field_df=field_df,
            synthetic_dataset_df=st.session_state.dataset,
            reliability_v2=st.session_state.reliability_v2,
            normality_result=st.session_state.normality_result,
            deployment_plan=st.session_state.deployment_plan,
            edge_impulse_snapshot=st.session_state.edge_impulse_snapshot,
            edge_impulse_classifier_snapshot=st.session_state.edge_impulse_classifier_snapshot,
            release_success_snapshot=st.session_state.release_success_snapshot,
            license_gate_snapshot=st.session_state.paid_license_snapshot,
        )
        st.session_state.field_validation_snapshot = snap
        st.session_state.field_validation_bundle = core.create_real_field_validation_bundle(st.session_state.project_name, snap, field_df, st.session_state.dataset)
        st.success("Real Field Validation Pack created.")

    if st.session_state.field_validation_snapshot:
        snap = st.session_state.field_validation_snapshot
        summary = snap.get("summary", {}) or {}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Field evidence", f"{summary.get('field_evidence_score', 0)}%")
        m2.metric("Decision", summary.get("decision", "Unknown"))
        m3.metric("Real files", summary.get("real_file_count", 0))
        m4.metric("Labels", summary.get("label_count", 0))
        st.info(summary.get("customer_message", ""))
        if st.session_state.field_validation_bundle:
            st.download_button(
                "Download Real Field Validation Bundle",
                st.session_state.field_validation_bundle,
                file_name=f"{st.session_state.project_name}_real_field_validation_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_field_validation_bundle_v30",
            )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Evidence items")
            e_df = pd.DataFrame(snap.get("evidence_items", []))
            if len(e_df) > 0:
                st.dataframe(e_df, use_container_width=True)
            for item in snap.get("blockers", []):
                st.error(item)
            for item in snap.get("must_collect_before_production", []):
                st.warning(item)
        with c2:
            st.markdown("#### Label summary")
            label_df = pd.DataFrame(snap.get("field_label_summary", []))
            if len(label_df) > 0:
                st.dataframe(label_df, use_container_width=True)
            st.markdown("#### Field-vs-synthetic")
            st.json(snap.get("field_vs_synthetic", {}))
    else:
        st.info("Upload field files or use the current dataset temporarily, then build the validation pack.")



# ============================================================
# V39 FIELD EVIDENCE 2.0 / ACCEPTANCE GATE
# ============================================================

with field_evidence_v2_tab:
    st.header("Field Evidence 2.0 • Acceptance Gate")
    st.write(
        "This turns real field files and generated evidence into a stricter acceptance package: coverage, label evidence, "
        "synthetic-vs-field drift, normal-baseline quality, Edge Impulse readiness, customer assurance and paid-delivery blockers."
    )

    c1, c2, c3 = st.columns(3)
    evidence_goal = c1.selectbox("Validation goal", core.get_field_evidence_v2_goals(), key="v39_evidence_goal")
    acceptance_level = c2.selectbox("Acceptance strictness", core.get_field_evidence_v2_acceptance_levels(), index=1, key="v39_acceptance_level")
    target_customer = c3.selectbox("Target customer type", ["Predictive Maintenance", "Security / Tamper", "Remote Asset / Forestry", "Edge AI Integrator", "Internal Founder Test"], key="v39_target_customer")

    c4, c5 = st.columns(2)
    site_name = c4.text_input("Site / machine / asset name", "Pilot Site A", key="v39_site_name")
    field_days = c5.number_input("Field evidence days", 1, 180, 14, 1, key="v39_field_days")

    sensor_stack = st.multiselect(
        "Field sensor stack",
        core.get_sensor_options(),
        default=["Audio", "Vibration"] if "Audio" in core.get_sensor_options() else core.get_sensor_options()[:2],
        key="v39_sensor_stack",
    )

    use_current_dataset_as_field_v2 = st.checkbox(
        "Use current dataset as temporary field evidence",
        value=False,
        key="v39_use_current_dataset_as_field",
        help="Good for internal testing only. Paid claims should use real uploaded field files.",
    )

    field_v2_df = st.session_state.field_validation_df
    if use_current_dataset_as_field_v2 and isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset) > 0:
        field_v2_df = st.session_state.dataset.copy()
        st.warning("Using current dataset as temporary field evidence. Do not present this as real field validation to a paying customer.")

    if isinstance(field_v2_df, pd.DataFrame) and len(field_v2_df) > 0:
        st.subheader("Field evidence preview")
        st.dataframe(field_v2_df.head(30), use_container_width=True)
    else:
        st.info("Use the Real Field Validation tab to upload WAV/CSV files first, or temporarily use the current dataset for internal testing.")

    st.caption(core.FIELD_EVIDENCE_V2_DISCLAIMER if hasattr(core, "FIELD_EVIDENCE_V2_DISCLAIMER") else "Field evidence is not production certification.")

    if st.button("Build Field Evidence 2.0 Gate", type="primary", use_container_width=True, key="v39_build_field_evidence_gate"):
        snap = core.build_field_evidence_v2_gate(
            project_name=st.session_state.project_name,
            field_df=field_v2_df,
            synthetic_df=st.session_state.dataset,
            evidence_goal=evidence_goal,
            acceptance_level=acceptance_level,
            site_name=site_name,
            field_days=field_days,
            sensor_stack=sensor_stack,
            target_customer=target_customer,
            reliability_v2=st.session_state.reliability_v2,
            normality_result=st.session_state.normality_result,
            edge_impulse_snapshot=st.session_state.edge_impulse_snapshot,
            edge_impulse_classifier_snapshot=st.session_state.edge_impulse_classifier_snapshot,
            deployment_plan=st.session_state.deployment_plan,
            field_validation_snapshot=st.session_state.field_validation_snapshot,
            customer_assurance_snapshot=st.session_state.customer_assurance_snapshot,
            commercial_license_certificate=st.session_state.commercial_license_certificate,
        )
        st.session_state.field_evidence_v2_snapshot = snap
        st.session_state.field_evidence_v2_bundle = core.create_field_evidence_v2_bundle(
            st.session_state.project_name,
            snap,
            field_df=field_v2_df,
            synthetic_df=st.session_state.dataset,
        )
        st.success("Field Evidence 2.0 Gate created.")

    if st.session_state.field_evidence_v2_snapshot:
        snap = st.session_state.field_evidence_v2_snapshot
        s = snap.get("summary", {}) or {}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Evidence Score", f"{s.get('field_evidence_v2_score', 0)}%")
        m2.metric("Decision", s.get("decision", "Unknown"))
        m3.metric("Evidence Grade", s.get("evidence_grade", "Unknown"))
        m4.metric("Real rows", s.get("field_rows", 0))
        st.info(s.get("customer_status_line", ""))

        if st.session_state.field_evidence_v2_bundle:
            st.download_button(
                "Download Field Evidence 2.0 Bundle",
                st.session_state.field_evidence_v2_bundle,
                file_name=f"{st.session_state.project_name}_field_evidence_v2_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_field_evidence_v2_bundle",
            )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Acceptance checklist")
            checklist_df = pd.DataFrame(snap.get("acceptance_checklist", []))
            if len(checklist_df) > 0:
                st.dataframe(checklist_df, use_container_width=True)
            for item in snap.get("blockers", []):
                st.error(item)
            for item in snap.get("warnings", []):
                st.warning(item)
        with c2:
            st.subheader("Label coverage")
            label_df = pd.DataFrame(snap.get("label_evidence", []))
            if len(label_df) > 0:
                st.dataframe(label_df, use_container_width=True)
            st.subheader("Drift / similarity")
            st.json(snap.get("field_vs_synthetic_drift", {}))

        with st.expander("Safe claims and claims to avoid", expanded=False):
            st.markdown("#### Safe claims")
            for item in snap.get("safe_customer_claims", []):
                st.write(f"- {item}")
            st.markdown("#### Avoid")
            for item in snap.get("claims_to_avoid", []):
                st.write(f"- {item}")
    else:
        st.info("Build this gate after you have at least a basic dataset and preferably uploaded real field files.")

# ============================================================
# V31 EDGE DEPLOYMENT STARTER KIT
# ============================================================

with edge_starter_tab:
    st.header("Edge Deployment Starter Kit")
    st.write(
        "Turns the pilot evidence into an implementation starter: feature contract, MQTT schema, firmware/gateway stubs, "
        "CSV validator and Node-RED starter flow. This is still pilot starter code, not production firmware certification."
    )

    hw_options = core.get_available_hardware() if hasattr(core, "get_available_hardware") else ["ESP32-S3"]
    default_board = st.session_state.hardware_result.get("recommendation") if isinstance(st.session_state.hardware_result, dict) else "ESP32-S3"
    c1, c2, c3 = st.columns(3)
    target_board = c1.selectbox("Target board", hw_options, index=hw_options.index(default_board) if default_board in hw_options else 0, key="edge_starter_board_v31")
    communication = c2.selectbox("Communication", ["WiFi / MQTT", "LoRa / LoRaWAN", "LTE / NB-IoT", "USB serial", "Local CSV logging"], index=0, key="edge_starter_comm_v31")
    inference_mode = c3.selectbox("Inference mode", ["auto", "Edge Impulse classifier", "Edge Impulse anomaly", "TFLite Micro", "gateway scoring"], index=0, key="edge_starter_mode_v31")
    c4, c5 = st.columns(2)
    starter_sr = c4.number_input("Sample rate", 1000, 48000, int(st.session_state.sr), 1000, key="edge_starter_sr_v31")
    starter_fft = c5.selectbox("FFT size", [512, 1024, 2048, 4096], index=1, key="edge_starter_fft_v31")

    st.caption(core.EDGE_STARTER_DISCLAIMER if hasattr(core, "EDGE_STARTER_DISCLAIMER") else "Pilot implementation starter only. Validate before deployment.")

    if st.button("Build Edge Deployment Starter Kit", type="primary", use_container_width=True, key="build_edge_starter_v31"):
        snap = core.build_edge_deployment_starter_kit(
            st.session_state.project_name,
            dataset_df=st.session_state.dataset,
            manifest=st.session_state.fusion_manifest,
            hardware_result=st.session_state.hardware_result,
            deployment_plan=st.session_state.deployment_plan,
            reliability_v2=st.session_state.reliability_v2,
            normality_result=st.session_state.normality_result,
            edge_impulse_snapshot=st.session_state.edge_impulse_snapshot,
            edge_impulse_classifier_snapshot=st.session_state.edge_impulse_classifier_snapshot,
            field_validation_snapshot=st.session_state.field_validation_snapshot,
            target_board=target_board,
            communication=communication,
            inference_mode=inference_mode,
            sample_rate=starter_sr,
            fft_size=starter_fft,
        )
        st.session_state.edge_deployment_starter_snapshot = snap
        st.session_state.edge_deployment_starter_bundle = core.create_edge_deployment_starter_bundle(st.session_state.project_name, snap, st.session_state.dataset)
        st.success("Edge Deployment Starter Kit created.")

    if st.session_state.edge_deployment_starter_snapshot:
        snap = st.session_state.edge_deployment_starter_snapshot
        summary = snap.get("summary", {}) or {}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Starter score", f"{summary.get('starter_score', 0)}%")
        m2.metric("Decision", summary.get("decision", "Unknown"))
        m3.metric("Target", summary.get("target_board", "Unknown"))
        m4.metric("Mode", summary.get("inference_mode", "Unknown"))
        st.info(summary.get("customer_message", ""))
        if st.session_state.edge_deployment_starter_bundle:
            st.download_button(
                "Download Edge Deployment Starter Bundle",
                st.session_state.edge_deployment_starter_bundle,
                file_name=f"{st.session_state.project_name}_edge_deployment_starter_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_edge_starter_bundle_v31",
            )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Evidence")
            e_df = pd.DataFrame(snap.get("evidence", []))
            if len(e_df) > 0:
                st.dataframe(e_df, use_container_width=True)
            for item in snap.get("blockers", []):
                st.error(item)
            for item in snap.get("warnings", []):
                st.warning(item)
        with c2:
            st.markdown("#### Feature contract")
            fc_df = pd.DataFrame(snap.get("feature_contract", []))
            if len(fc_df) > 0:
                st.dataframe(fc_df, use_container_width=True)
            st.markdown("#### MQTT schema")
            st.json(snap.get("mqtt_schema", {}))
    else:
        st.info("Generate a dataset, deployment plan and Edge Impulse export first for a stronger starter kit.")


# ============================================================
# V31.2 STORAGE / SCALABILITY RECOVERY
# ============================================================

with scalability_tab:
    st.header("Storage & Scalability Recovery")
    st.write(
        "V31.1 moves large DataFrames out of SQLite into local file storage. This keeps the beta app lighter and prepares "
        "a later move to S3/MinIO + PostgreSQL without changing the product logic."
    )

    plan_options = core.get_pricing_plans()
    scale_plan = st.selectbox("Plan for sample-limit check", plan_options, index=plan_options.index(st.session_state.selected_plan) if st.session_state.selected_plan in plan_options else 0, key="scale_plan_v31_2")
    requested_samples = st.number_input("Requested generation samples", 0, 200000, int(len(st.session_state.dataset)) if isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset) > 0 else 500, 100, key="scale_requested_samples_v31_2")
    sample_check = core.validate_sample_request(scale_plan, requested_samples)

    if sample_check.get("was_capped"):
        st.warning(sample_check.get("message"))
    else:
        st.success(sample_check.get("message"))

    if st.button("Run Storage / Scalability Check", type="primary", use_container_width=True, key="run_scalability_check_v31_2"):
        snap = core.get_scalability_readiness_snapshot(st.session_state.dataset, scale_plan)
        st.session_state.scalability_snapshot = snap
        try:
            user_projects = core.get_user_projects(st.session_state.user["id"])
        except Exception:
            user_projects = pd.DataFrame()
        st.session_state.scalability_bundle = core.create_scalability_storage_bundle(st.session_state.project_name, snap, user_projects)
        st.success("Storage / Scalability check completed.")

    if st.session_state.scalability_snapshot:
        snap = st.session_state.scalability_snapshot
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scalability score", f"{snap.get('scalability_score', 0)}%")
        c2.metric("Dataset rows", snap.get("dataset_rows", 0))
        c3.metric("Plan limit", snap.get("sample_limit", 0))
        c4.metric("Storage mode", snap.get("storage", {}).get("mode", "Unknown"))
        st.info(snap.get("recommendation", ""))
        if st.session_state.scalability_bundle:
            st.download_button(
                "Download Storage / Scalability Bundle",
                st.session_state.scalability_bundle,
                file_name=f"{st.session_state.project_name}_storage_scalability_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_scalability_bundle_v31_2",
            )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Issues")
            for item in snap.get("issues", []):
                sev = item.get("severity", "info")
                if sev == "high": st.error(item.get("message", ""))
                elif sev == "medium": st.warning(item.get("message", ""))
                else: st.info(item.get("message", ""))
        with c2:
            st.markdown("#### Storage status")
            st.json(snap.get("storage", {}))
    else:
        st.info("Run this check after saving/loading projects or generating larger datasets.")


# ============================================================
# V32 OPERATIONAL CONTROL CENTER
# ============================================================

with operational_tab:
    st.header("Operational Control Center")
    st.write(
        "This is the command center for running EdgeTwin Studio as a serious beta/product: storage, release readiness, "
        "paid-export status, field validation, deployment starter, API readiness and must-fix blockers in one place."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        control_plan = st.selectbox(
            "Operating plan",
            core.get_pricing_plans(),
            index=core.get_pricing_plans().index(st.session_state.selected_plan) if st.session_state.selected_plan in core.get_pricing_plans() else 0,
            key="operational_plan_v32",
        )
    with c2:
        operating_mode = st.selectbox(
            "Operating mode",
            ["Internal demo", "Closed beta", "Paid pilot", "Real-data pilot", "Enterprise review"],
            index=1,
            key="operational_mode_v32",
        )
    with c3:
        support_level = st.selectbox(
            "Support level",
            ["Solo founder", "Assisted beta", "Enterprise handoff"],
            index=0,
            key="operational_support_v32",
        )

    open_notes = st.text_area(
        "Open operator notes / known issues",
        value="No critical known issue recorded. Validate with real data before production claims.",
        height=80,
        key="operational_notes_v32",
    )

    if st.button("Run Operational Control Check", type="primary", use_container_width=True, key="run_operational_control_v32"):
        try:
            user_projects = core.get_user_projects(st.session_state.user["id"])
        except Exception:
            user_projects = pd.DataFrame()
        snapshot = core.build_operational_control_center_snapshot(
            project_name=st.session_state.project_name,
            dataset_df=st.session_state.dataset,
            plan_name=control_plan,
            operating_mode=operating_mode,
            support_level=support_level,
            user_projects=user_projects,
            storage_snapshot=st.session_state.scalability_snapshot,
            hardening_snapshot=st.session_state.hardening_snapshot,
            release_success_snapshot=st.session_state.release_success_snapshot,
            beta_launch_snapshot=st.session_state.beta_launch_snapshot,
            paid_license_snapshot=st.session_state.paid_license_snapshot,
            field_validation_snapshot=st.session_state.field_validation_snapshot,
            edge_deployment_starter_snapshot=st.session_state.edge_deployment_starter_snapshot,
            professional_report_snapshot=st.session_state.professional_report_snapshot,
            api_automation_snapshot=st.session_state.api_automation_snapshot,
            monetization_snapshot=st.session_state.monetization_snapshot,
            trust_gate=st.session_state.trust_gate,
            reliability_v2=st.session_state.reliability_v2,
            normality_result=st.session_state.normality_result,
            edge_impulse_snapshot=st.session_state.edge_impulse_snapshot,
            edge_impulse_classifier_snapshot=st.session_state.edge_impulse_classifier_snapshot,
            open_notes=open_notes,
        )
        st.session_state.operational_control_snapshot = snapshot
        st.session_state.operational_control_bundle = core.create_operational_control_center_bundle(
            st.session_state.project_name,
            snapshot,
            dataset_df=st.session_state.dataset,
            projects_df=user_projects,
        )
        st.success("Operational Control Check completed.")

    if st.session_state.operational_control_snapshot:
        snap = st.session_state.operational_control_snapshot
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Operational score", f"{snap.get('operational_score', 0)}%")
        c2.metric("Decision", snap.get("decision", "Unknown"))
        c3.metric("Risk", snap.get("risk_level", "Unknown"))
        c4.metric("Mode", snap.get("operating_mode", "Unknown"))

        decision = snap.get("decision", "")
        if decision == "GO":
            st.success(snap.get("operator_summary", ""))
        elif decision == "CONDITIONAL GO":
            st.warning(snap.get("operator_summary", ""))
        else:
            st.error(snap.get("operator_summary", ""))

        if st.session_state.operational_control_bundle:
            st.download_button(
                "Download Operational Control Bundle",
                st.session_state.operational_control_bundle,
                file_name=f"{st.session_state.project_name}_operational_control_bundle_v32.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_operational_control_bundle_v32",
            )

        st.markdown("#### Control panels")
        panels = pd.DataFrame(snap.get("control_panels", []))
        if len(panels) > 0:
            st.dataframe(panels, use_container_width=True)

        lcol, rcol = st.columns(2)
        with lcol:
            st.markdown("#### Must-fix blockers")
            blockers = snap.get("must_fix_blockers", [])
            if blockers:
                for item in blockers:
                    sev = item.get("severity", "info")
                    msg = item.get("message", "")
                    if sev == "high":
                        st.error(msg)
                    elif sev == "medium":
                        st.warning(msg)
                    else:
                        st.info(msg)
            else:
                st.success("No must-fix blockers detected for the selected operating mode.")

            st.markdown("#### Next actions")
            for item in snap.get("next_actions", []):
                st.write(f"- {item}")

        with rcol:
            st.markdown("#### Daily operator checklist")
            checklist = pd.DataFrame(snap.get("daily_operator_checklist", []))
            if len(checklist) > 0:
                st.dataframe(checklist, use_container_width=True)
            st.markdown("#### Safe status line")
            st.info(snap.get("safe_status_line", ""))

        with st.expander("Launch runbook", expanded=False):
            for section in snap.get("launch_runbook", []):
                st.markdown(f"**{section.get('title', '')}**")
                for step in section.get("steps", []):
                    st.write(f"- {step}")

        with st.expander("Full operational snapshot", expanded=False):
            st.json(snap)
    else:
        st.info("Run this after generating at least one dataset and the key readiness checks. It is the operator view before demo, beta or paid pilot delivery.")



# ============================================================
# V33 PRODUCTION SAFETY & ERROR OBSERVATORY
# ============================================================

with observability_tab:
    st.header("Production Safety & Error Observatory")
    st.write(
        "V33 is the safety layer for closed beta and paid pilots: record incidents, check missing outputs, "
        "monitor export readiness and create an operator bundle when something must be fixed before a customer sees it."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        obs_mode = st.selectbox(
            "Release mode",
            ["Internal demo", "Closed beta", "Paid pilot", "Real-data pilot", "Enterprise review"],
            index=1,
            key="observability_mode_v33",
        )
    with c2:
        obs_severity = st.selectbox(
            "Manual event severity",
            ["info", "warning", "error", "critical"],
            index=0,
            key="observability_manual_severity_v33",
        )
    with c3:
        obs_flow = st.selectbox(
            "Flow / area",
            ["general", "auth", "dataset", "real_bridge", "export", "report", "edge_impulse", "deployment", "payment", "api", "storage"],
            index=0,
            key="observability_flow_v33",
        )

    manual_message = st.text_area(
        "Manual incident / note",
        value="",
        height=80,
        placeholder="Example: Enterprise Bundle export failed after loading a real CSV file.",
        key="observability_manual_message_v33",
    )

    if st.button("Log manual observability event", use_container_width=True, key="log_observability_event_v33"):
        if manual_message.strip():
            event = core.log_observability_event(
                event_type="manual_note",
                severity=obs_severity,
                message=manual_message.strip(),
                flow=obs_flow,
                project_name=st.session_state.project_name,
                user_id=st.session_state.user.get("id"),
                context={"mode": obs_mode, "plan": st.session_state.selected_plan},
            )
            st.session_state.last_observability_event = event
            st.success("Observability event logged.")
        else:
            st.warning("Write a short message before logging an event.")

    if st.button("Run Production Safety Check", type="primary", use_container_width=True, key="run_observability_check_v33"):
        snapshot = core.build_production_safety_observatory_snapshot(
            project_name=st.session_state.project_name,
            dataset_df=st.session_state.dataset,
            release_mode=obs_mode,
            selected_plan=st.session_state.selected_plan,
            user_id=st.session_state.user.get("id"),
            operational_snapshot=st.session_state.operational_control_snapshot,
            hardening_snapshot=st.session_state.hardening_snapshot,
            release_success_snapshot=st.session_state.release_success_snapshot,
            trust_gate=st.session_state.trust_gate,
            reliability_v2=st.session_state.reliability_v2,
            professional_report_snapshot=st.session_state.professional_report_snapshot,
            field_validation_snapshot=st.session_state.field_validation_snapshot,
            edge_deployment_starter_snapshot=st.session_state.edge_deployment_starter_snapshot,
            monetization_snapshot=st.session_state.monetization_snapshot,
            api_automation_snapshot=st.session_state.api_automation_snapshot,
            edge_impulse_snapshot=st.session_state.edge_impulse_snapshot,
            edge_impulse_classifier_snapshot=st.session_state.edge_impulse_classifier_snapshot,
            bundle_presence={
                "fusion_bundle": st.session_state.fusion_bundle is not None,
                "enterprise_bundle": st.session_state.enterprise_bundle is not None,
                "professional_report_bundle": st.session_state.professional_report_bundle is not None,
                "deployment_bundle": st.session_state.deployment_bundle is not None,
                "hardening_bundle": st.session_state.hardening_bundle is not None,
                "release_success_bundle": st.session_state.release_success_bundle is not None,
                "operational_control_bundle": st.session_state.operational_control_bundle is not None,
                "field_validation_bundle": st.session_state.field_validation_bundle is not None,
                "edge_impulse_bundle": st.session_state.edge_impulse_bundle is not None,
                "edge_impulse_classifier_bundle": st.session_state.edge_impulse_classifier_bundle is not None,
            },
        )
        st.session_state.observability_snapshot = snapshot
        st.session_state.observability_bundle = core.create_production_safety_observatory_bundle(
            st.session_state.project_name,
            snapshot,
            dataset_df=st.session_state.dataset,
        )
        st.success("Production Safety Check completed.")

    if st.session_state.last_observability_event:
        with st.expander("Last logged event", expanded=False):
            st.json(st.session_state.last_observability_event)

    if st.session_state.observability_snapshot:
        snap = st.session_state.observability_snapshot
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Safety score", f"{snap.get('safety_score', 0)}%")
        c2.metric("Decision", snap.get("decision", "Unknown"))
        c3.metric("Incident risk", snap.get("incident_risk", "Unknown"))
        c4.metric("Recent events", snap.get("recent_event_count", 0))

        if snap.get("decision") == "GO":
            st.success(snap.get("operator_summary", ""))
        elif snap.get("decision") == "CONDITIONAL GO":
            st.warning(snap.get("operator_summary", ""))
        else:
            st.error(snap.get("operator_summary", ""))

        if st.session_state.observability_bundle:
            st.download_button(
                "Download Production Safety Bundle",
                st.session_state.observability_bundle,
                file_name=f"{st.session_state.project_name}_production_safety_bundle_v33.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_production_safety_bundle_v33",
            )

        st.markdown("#### Safety checks")
        checks_df = pd.DataFrame(snap.get("safety_checks", []))
        if len(checks_df) > 0:
            st.dataframe(checks_df, use_container_width=True)

        lcol, rcol = st.columns(2)
        with lcol:
            st.markdown("#### Must-fix before customer")
            for item in snap.get("must_fix", []):
                sev = item.get("severity", "info")
                msg = item.get("message", "")
                if sev in ["critical", "high", "error"]:
                    st.error(msg)
                elif sev in ["medium", "warning"]:
                    st.warning(msg)
                else:
                    st.info(msg)
        with rcol:
            st.markdown("#### Next actions")
            for action in snap.get("next_actions", []):
                st.write(f"- {action}")
            st.markdown("#### Recent events")
            recent_df = pd.DataFrame(snap.get("recent_events", []))
            if len(recent_df) > 0:
                st.dataframe(recent_df, use_container_width=True)
            else:
                st.caption("No recorded observability events yet.")

        with st.expander("Full observability snapshot", expanded=False):
            st.json(snap)
    else:
        st.info("Run this before showing a demo, delivering a paid pilot bundle, or after any export/runtime problem.")




# ============================================================
# V34 CUSTOMER ASSURANCE & DATA GOVERNANCE CENTER
# ============================================================

with governance_tab:
    st.header("Customer Assurance & Data Governance Center")
    st.write(
        "V34 prepares EdgeTwin for real customer trust: data categories, upload warnings, retention policy, "
        "safe customer claims, customer-data reuse controls and an assurance bundle for private beta or paid pilots."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        release_mode = st.selectbox(
            "Release mode",
            ["Internal demo", "Private beta", "Paid pilot", "Real-data pilot", "Enterprise review"],
            index=1,
            key="governance_release_mode_v34",
        )
        retention_policy = st.selectbox(
            "Retention policy",
            core.get_governance_retention_options(),
            index=1,
            key="governance_retention_policy_v34",
            help="For early beta, delete_after_export or customer_controlled_retention is safest."
        )
    with c2:
        uses_customer_uploads = st.checkbox(
            "Customer may upload real WAV/CSV data",
            value=bool(st.session_state.real_bridge_result or st.session_state.field_validation_snapshot),
            key="governance_customer_uploads_v34",
        )
        allows_model_reuse = st.checkbox(
            "Allow customer data reuse for shared model improvement",
            value=False,
            key="governance_model_reuse_v34",
            help="Keep this OFF by default. Reuse should only happen with explicit opt-in."
        )
    with c3:
        customer_data_training = st.checkbox(
            "Use customer data for future training packs",
            value=False,
            key="governance_customer_training_v34",
            help="Keep this OFF unless a written opt-in exists."
        )
        st.metric("Active dataset rows", len(st.session_state.dataset) if isinstance(st.session_state.dataset, pd.DataFrame) else 0)

    st.info("Default safe position: customer data stays customer-owned, is not reused for shared model improvement without opt-in, and EdgeTwin outputs are pilot-preparation evidence — not production certification.")

    if st.button("Run Customer Assurance Check", type="primary", use_container_width=True, key="run_customer_assurance_v34"):
        with st.spinner("Building customer assurance snapshot..."):
            snapshot = core.create_customer_assurance_snapshot(
                project_name=st.session_state.project_name,
                dataset_df=st.session_state.dataset,
                selected_plan=st.session_state.selected_plan,
                release_mode=release_mode,
                retention_policy=retention_policy,
                uses_customer_uploads=uses_customer_uploads,
                allows_model_reuse=allows_model_reuse,
                customer_data_used_for_training=customer_data_training,
                manifest=st.session_state.fusion_manifest,
                trust_snapshot=st.session_state.trust_gate,
                reliability_snapshot=st.session_state.reliability_v2 or (st.session_state.auto_pilot_result or {}).get("reliability") if st.session_state.auto_pilot_result else st.session_state.reliability_v2,
                hardening_snapshot=st.session_state.hardening_snapshot,
                observability_snapshot=st.session_state.observability_snapshot,
                scalability_snapshot=st.session_state.scalability_snapshot,
                field_validation_snapshot=st.session_state.field_validation_snapshot,
            )
            st.session_state.customer_assurance_snapshot = snapshot
            st.session_state.customer_assurance_bundle = core.create_customer_assurance_bundle(
                st.session_state.project_name,
                snapshot,
                st.session_state.dataset,
            )
        st.success("Customer Assurance Check completed.")

    if st.session_state.customer_assurance_snapshot:
        snap = st.session_state.customer_assurance_snapshot
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Governance Score", f"{snap.get('governance_score', 0)}%")
        m2.metric("Decision", snap.get("decision", "Unknown"))
        m3.metric("Dataset Risk", (snap.get("dataset_governance", {}) or {}).get("risk_level", "Unknown"))
        m4.metric("Safe to Share", "Yes" if snap.get("safe_to_share_with_customer") else "Review")

        if snap.get("decision") == "CUSTOMER-ASSURANCE READY":
            st.success(snap.get("customer_assurance_line", ""))
        elif str(snap.get("decision", "")).startswith("CONDITIONAL"):
            st.warning(snap.get("customer_assurance_line", ""))
        else:
            st.error(snap.get("customer_assurance_line", ""))

        if st.session_state.customer_assurance_bundle:
            st.download_button(
                "Download Customer Assurance Bundle",
                st.session_state.customer_assurance_bundle,
                file_name=f"{st.session_state.project_name}_customer_assurance_v34.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_customer_assurance_bundle_v34",
            )

        tabs = st.tabs(["Controls", "Issues", "Data Categories", "Claims"])
        with tabs[0]:
            st.subheader("Assurance controls")
            st.dataframe(pd.DataFrame(snap.get("controls", [])), use_container_width=True)
        with tabs[1]:
            st.subheader("Governance issues / must-fix")
            issues = pd.DataFrame(snap.get("issues", []))
            if len(issues) > 0:
                st.dataframe(issues, use_container_width=True)
            else:
                st.success("No governance issues detected.")
        with tabs[2]:
            st.subheader("Detected data categories")
            cats = pd.DataFrame((snap.get("dataset_governance", {}) or {}).get("categories", []))
            if len(cats) > 0:
                st.dataframe(cats, use_container_width=True)
            else:
                st.info("No data categories detected yet.")
        with tabs[3]:
            st.subheader("Allowed claims")
            for item in snap.get("claims_allowed", []):
                st.success(item)
            st.subheader("Claims to avoid")
            for item in snap.get("claims_to_avoid", []):
                st.warning(item)
            st.subheader("Customer upload notice")
            st.info(snap.get("customer_upload_notice", ""))

        with st.expander("Full customer assurance snapshot", expanded=False):
            st.json(snap)
    else:
        st.info("Run the Customer Assurance Check after generating a dataset and before sharing real-data upload flows with beta users or customers.")


# ============================================================
# V35 CUSTOMER ONBOARDING & GUIDED SUCCESS FLOW
# ============================================================

with onboarding_tab:
    st.header("Customer Onboarding & Guided Success Flow")
    st.write(
        "V35 turns the full EdgeTwin platform into a guided customer journey. Instead of asking a customer to understand every tab, "
        "it recommends the safest path, required evidence, right bundle, demo script and next actions for their use-case."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        customer_segment = st.selectbox(
            "Customer segment",
            core.get_onboarding_customer_segments(),
            index=0,
            key="onboarding_segment_v35",
        )
        use_case_type = st.selectbox(
            "Primary use-case",
            core.get_use_case_types(),
            index=0,
            key="onboarding_use_case_v35",
        )
    with c2:
        data_status = st.selectbox(
            "Customer data status",
            core.get_onboarding_data_status_options(),
            index=1,
            key="onboarding_data_status_v35",
        )
        technical_level = st.selectbox(
            "Customer technical level",
            core.get_onboarding_technical_levels(),
            index=1,
            key="onboarding_technical_level_v35",
        )
    with c3:
        desired_outcome = st.selectbox(
            "Desired outcome",
            core.get_onboarding_outcomes(),
            index=1,
            key="onboarding_desired_outcome_v35",
        )
        sales_mode = st.selectbox(
            "Sales mode",
            ["Internal demo", "First customer demo", "Private beta", "Paid pilot", "Enterprise review"],
            index=2,
            key="onboarding_sales_mode_v35",
        )

    st.info("The safest customer journey is usually: guided demo -> use-case wizard -> reliability/trust -> deployment/report -> customer assurance -> success gate.")

    if st.button("Build Guided Success Plan", type="primary", use_container_width=True, key="run_guided_success_v35"):
        with st.spinner("Building customer success plan..."):
            snapshot = core.create_guided_success_snapshot(
                project_name=st.session_state.project_name,
                customer_segment=customer_segment,
                use_case_type=use_case_type,
                data_status=data_status,
                technical_level=technical_level,
                desired_outcome=desired_outcome,
                sales_mode=sales_mode,
                selected_plan=st.session_state.selected_plan,
                dataset_df=st.session_state.dataset,
                manifest=st.session_state.fusion_manifest,
                trust_snapshot=st.session_state.trust_gate,
                reliability_snapshot=st.session_state.reliability_v2 or ((st.session_state.auto_pilot_result or {}).get("reliability") if st.session_state.auto_pilot_result else None),
                deployment_snapshot=st.session_state.deployment_plan,
                report_snapshot=st.session_state.professional_report_snapshot,
                assurance_snapshot=st.session_state.customer_assurance_snapshot,
                success_snapshot=st.session_state.release_success_snapshot,
                operational_snapshot=st.session_state.operational_control_snapshot,
                observability_snapshot=st.session_state.observability_snapshot,
                governance_snapshot=st.session_state.customer_assurance_snapshot,
            )
            st.session_state.onboarding_snapshot = snapshot
            st.session_state.onboarding_bundle = core.create_guided_success_bundle(
                st.session_state.project_name,
                snapshot,
                st.session_state.dataset,
            )
            st.session_state.guided_success_bundle = st.session_state.onboarding_bundle
        st.success("Guided Success Plan created.")

    if st.session_state.onboarding_snapshot:
        snap = st.session_state.onboarding_snapshot
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Onboarding Score", f"{snap.get('onboarding_score', 0)}%")
        m2.metric("Decision", snap.get("decision", "Unknown"))
        m3.metric("Recommended Package", snap.get("recommended_package", "Unknown"))
        m4.metric("Next Milestone", snap.get("next_milestone", "Unknown"))

        decision = snap.get("decision", "")
        if decision == "READY FOR GUIDED CUSTOMER SESSION":
            st.success(snap.get("customer_success_line", ""))
        elif decision.startswith("CONDITIONAL"):
            st.warning(snap.get("customer_success_line", ""))
        else:
            st.error(snap.get("customer_success_line", ""))

        if st.session_state.onboarding_bundle:
            st.download_button(
                "Download Guided Success Bundle",
                st.session_state.onboarding_bundle,
                file_name=f"{st.session_state.project_name}_guided_success_v35.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_guided_success_bundle_v35",
            )

        tabs = st.tabs(["Journey", "Evidence", "Script", "Blockers", "Customer Copy"])
        with tabs[0]:
            st.subheader("Recommended customer journey")
            st.dataframe(pd.DataFrame(snap.get("journey_steps", [])), use_container_width=True)
            st.subheader("Next actions")
            for item in snap.get("next_actions", []):
                st.write(f"- {item}")
        with tabs[1]:
            st.subheader("Evidence coverage")
            st.dataframe(pd.DataFrame(snap.get("evidence_coverage", [])), use_container_width=True)
            st.subheader("Success metrics")
            st.dataframe(pd.DataFrame(snap.get("success_metrics", [])), use_container_width=True)
        with tabs[2]:
            st.subheader("Customer demo script")
            for line in snap.get("demo_script", []):
                st.write(f"- {line}")
        with tabs[3]:
            st.subheader("Blockers / must-fix")
            blockers = pd.DataFrame(snap.get("blockers", []))
            if len(blockers) > 0:
                st.dataframe(blockers, use_container_width=True)
            else:
                st.success("No high-priority onboarding blockers detected.")
        with tabs[4]:
            st.subheader("Safe customer-facing copy")
            st.info(snap.get("customer_facing_copy", ""))
            st.subheader("Claims to avoid")
            for item in snap.get("claims_to_avoid", []):
                st.warning(item)

        with st.expander("Full guided success snapshot", expanded=False):
            st.json(snap)

    else:
        st.info("Run this before sending a beta invite, doing a live demo, or offering a paid pilot bundle.")


# ============================================================
# V36 CUSTOMER WORKSPACE / PROJECT LIFECYCLE
# ============================================================

with workspace_tab:
    st.header("Customer Workspace & Project Lifecycle")
    st.write(
        "V36 turns EdgeTwin into a project workspace: it tracks where a customer is in the journey, "
        "what evidence is already available, which deliverables are missing, and what the next safe action should be."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        workspace_customer = st.text_input("Customer / workspace name", value="Private Beta Customer", key="workspace_customer_v36")
        lifecycle_stage = st.selectbox("Current lifecycle stage", core.get_workspace_lifecycle_stages(), index=1, key="workspace_stage_v36")
    with c2:
        lifecycle_goal = st.selectbox("Target outcome", core.get_workspace_target_goals(), index=2, key="workspace_goal_v36")
        workspace_owner = st.selectbox("Owner role", ["Founder/operator", "Customer engineer", "Integrator", "Consultant", "Internal champion"], index=0, key="workspace_owner_v36")
    with c3:
        delivery_window = st.selectbox("Delivery window", ["Today", "This week", "2 weeks", "30 days", "No deadline yet"], index=2, key="workspace_delivery_window_v36")
        risk_tolerance = st.selectbox("Customer risk tolerance", ["Low", "Medium", "High / exploratory"], index=1, key="workspace_risk_tolerance_v36")

    st.info("Use this after Guided Success. It becomes the operator cockpit for each customer/workspace: stage, evidence, deliverables, risks and next actions.")

    if st.button("Build Workspace Lifecycle Plan", type="primary", use_container_width=True, key="run_workspace_lifecycle_v36"):
        with st.spinner("Building workspace lifecycle plan..."):
            snapshot = core.create_workspace_lifecycle_snapshot(
                project_name=st.session_state.project_name,
                customer_name=workspace_customer,
                lifecycle_stage=lifecycle_stage,
                target_goal=lifecycle_goal,
                owner_role=workspace_owner,
                delivery_window=delivery_window,
                risk_tolerance=risk_tolerance,
                selected_plan=st.session_state.selected_plan,
                dataset_df=st.session_state.dataset,
                manifest=st.session_state.fusion_manifest,
                onboarding_snapshot=st.session_state.onboarding_snapshot,
                assurance_snapshot=st.session_state.customer_assurance_snapshot,
                trust_snapshot=st.session_state.trust_gate,
                reliability_snapshot=st.session_state.reliability_v2 or ((st.session_state.auto_pilot_result or {}).get("reliability") if st.session_state.auto_pilot_result else None),
                deployment_snapshot=st.session_state.deployment_plan,
                report_snapshot=st.session_state.professional_report_snapshot,
                success_snapshot=st.session_state.release_success_snapshot,
                operational_snapshot=st.session_state.operational_control_snapshot,
                observability_snapshot=st.session_state.observability_snapshot,
                paid_license_snapshot=st.session_state.paid_license_snapshot,
                field_validation_snapshot=st.session_state.field_validation_snapshot,
                edge_export_snapshot=st.session_state.edge_impulse_snapshot,
                classifier_export_snapshot=st.session_state.edge_impulse_classifier_snapshot,
            )
            st.session_state.workspace_lifecycle_snapshot = snapshot
            st.session_state.workspace_lifecycle_bundle = core.create_workspace_lifecycle_bundle(
                st.session_state.project_name,
                snapshot,
                st.session_state.dataset,
            )
        st.success("Workspace Lifecycle Plan created.")

    if st.session_state.workspace_lifecycle_snapshot:
        snap = st.session_state.workspace_lifecycle_snapshot
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Workspace Score", f"{snap.get('workspace_score', 0)}%")
        m2.metric("Decision", snap.get("decision", "Unknown"))
        m3.metric("Stage", snap.get("lifecycle_stage", "Unknown"))
        m4.metric("Next Action", snap.get("next_action", "Unknown"))

        decision = snap.get("decision", "")
        if decision.startswith("READY"):
            st.success(snap.get("operator_status_line", ""))
        elif decision.startswith("CONDITIONAL"):
            st.warning(snap.get("operator_status_line", ""))
        else:
            st.error(snap.get("operator_status_line", ""))

        if st.session_state.workspace_lifecycle_bundle:
            st.download_button(
                "Download Workspace Lifecycle Bundle",
                st.session_state.workspace_lifecycle_bundle,
                file_name=f"{st.session_state.project_name}_workspace_lifecycle_v36.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_workspace_lifecycle_bundle_v36",
            )

        tabs_ws = st.tabs(["Timeline", "Deliverables", "Evidence", "Risks", "Hand-off"])
        with tabs_ws[0]:
            st.subheader("Lifecycle timeline")
            st.dataframe(pd.DataFrame(snap.get("timeline", [])), use_container_width=True)
            st.subheader("Next actions")
            for item in snap.get("next_actions", []):
                st.write(f"- {item}")
        with tabs_ws[1]:
            st.subheader("Deliverable checklist")
            st.dataframe(pd.DataFrame(snap.get("deliverables", [])), use_container_width=True)
        with tabs_ws[2]:
            st.subheader("Evidence map")
            st.dataframe(pd.DataFrame(snap.get("evidence_map", [])), use_container_width=True)
        with tabs_ws[3]:
            st.subheader("Workspace risks")
            risks = pd.DataFrame(snap.get("risks", []))
            if len(risks) > 0:
                st.dataframe(risks, use_container_width=True)
            else:
                st.success("No major lifecycle risks detected.")
        with tabs_ws[4]:
            st.subheader("Customer hand-off summary")
            st.info(snap.get("customer_handoff_summary", ""))
            st.subheader("Internal operator notes")
            for item in snap.get("operator_notes", []):
                st.write(f"- {item}")
            st.subheader("Claims to avoid")
            for item in snap.get("claims_to_avoid", []):
                st.warning(item)

        with st.expander("Full workspace lifecycle snapshot", expanded=False):
            st.json(snap)
    else:
        st.info("Run this once per customer/workspace after you have at least a demo or wizard dataset. It becomes the project lifecycle record.")



# ============================================================
# V37 ADMIN DASHBOARD / USAGE TRACKING
# ============================================================

with admin_tab:
    st.header("Admin Dashboard & Usage Tracking • Commercial License Certificate")
    st.write(
        "V37 gives the operator a lightweight admin/customer-success dashboard: project history, ready bundles, "
        "export events, plan limits, usage signals and next actions before a beta or paid-pilot handoff."
    )

    c1, c2, c3 = st.columns(3)
    operating_mode = c1.selectbox(
        "Operating mode",
        ["Internal QA", "Closed beta", "Paid pilot", "Customer handoff", "Public SaaS prep"],
        index=1,
        key="admin_operating_mode_v37",
    )
    customer_success_stage = c2.selectbox(
        "Customer success stage",
        ["Internal QA", "First demo", "Beta onboarding", "Paid pilot", "Field validation", "Renewal / expansion"],
        index=1,
        key="admin_success_stage_v37",
    )
    log_bundle_type = c3.selectbox(
        "Bundle/event to log",
        core.EXPORT_BUNDLE_TYPES if hasattr(core, "EXPORT_BUNDLE_TYPES") else ["Professional Report Bundle"],
        key="admin_bundle_type_v37",
    )

    operator_note = st.text_area(
        "Operator note / customer-success note",
        value="",
        height=90,
        key="admin_operator_note_v37",
        help="Use this for customer feedback, delivery notes, blockers, or what you promised during a demo.",
    )

    active_bundles = {
        "Fusion Bundle": st.session_state.fusion_bundle is not None,
        "Enterprise Bundle": st.session_state.enterprise_bundle is not None,
        "Auto Pilot Bundle": st.session_state.auto_pilot_bundle is not None,
        "Optimizer Bundle": st.session_state.optimizer_bundle is not None,
        "Trust Bundle": st.session_state.trust_bundle is not None,
        "Real Bridge Bundle": st.session_state.real_bridge_bundle is not None,
        "Reliability 2.0 Bundle": st.session_state.reliability_v2_bundle is not None,
        "Deployment Bundle": st.session_state.deployment_bundle is not None,
        "Professional Report Bundle": st.session_state.professional_report_bundle is not None,
        "Monetization Bundle": st.session_state.monetization_bundle is not None,
        "Hardening Bundle": st.session_state.hardening_bundle is not None,
        "Beta Launch Bundle": st.session_state.beta_launch_bundle is not None,
        "API Automation Bundle": st.session_state.api_automation_bundle is not None,
        "Marketplace Bundle": st.session_state.pack_marketplace_bundle is not None,
        "Normality Bundle": st.session_state.normality_bundle is not None,
        "Edge Impulse Anomaly Bundle": st.session_state.edge_impulse_bundle is not None,
        "Edge Impulse Classifier Bundle": st.session_state.edge_impulse_classifier_bundle is not None,
        "Release Success Bundle": st.session_state.release_success_bundle is not None,
        "Golden Demo Bundle": st.session_state.golden_demo_bundle is not None,
        "Closed Beta Bundle": st.session_state.closed_beta_bundle is not None,
        "Paid Export Bundle": st.session_state.paid_license_bundle is not None,
        "Field Validation Bundle": st.session_state.field_validation_bundle is not None,
        "Edge Starter Bundle": st.session_state.edge_deployment_starter_bundle is not None,
        "Scalability Bundle": st.session_state.scalability_bundle is not None,
        "Operational Control Bundle": st.session_state.operational_control_bundle is not None,
        "Observability Bundle": st.session_state.observability_bundle is not None,
        "Customer Assurance Bundle": st.session_state.customer_assurance_bundle is not None,
        "Guided Success Bundle": st.session_state.onboarding_bundle is not None,
        "Workspace Lifecycle Bundle": st.session_state.workspace_lifecycle_bundle is not None,
        "Commercial License Certificate": st.session_state.commercial_license_bundle is not None,
    }

    a1, a2, a3 = st.columns(3)
    if a1.button("Log selected export event", use_container_width=True, key="admin_log_export_v37"):
        st.session_state.last_admin_export_event = core.record_export_event(
            st.session_state.user.get("id"),
            st.session_state.project_id,
            st.session_state.project_name,
            log_bundle_type,
            st.session_state.selected_plan,
            status="prepared_or_delivered",
            notes=operator_note,
        )
        st.success("Export event logged.")

    if a2.button("Record operator note", use_container_width=True, key="admin_record_note_v37"):
        st.session_state.last_operator_note_event = core.record_operator_note(
            st.session_state.user.get("id"),
            st.session_state.project_id,
            st.session_state.project_name,
            note_type=customer_success_stage,
            severity="info",
            note=operator_note,
        )
        st.success("Operator note recorded.")

    if a3.button("Build Admin Usage Dashboard", type="primary", use_container_width=True, key="run_admin_usage_v37"):
        projects_df = core.get_user_projects(st.session_state.user["id"])
        snap = core.create_admin_usage_dashboard_snapshot(
            user=st.session_state.user,
            project_name=st.session_state.project_name,
            project_id=st.session_state.project_id,
            plan_name=st.session_state.selected_plan,
            dataset_df=st.session_state.dataset,
            projects_df=projects_df,
            active_bundles=active_bundles,
            operating_mode=operating_mode,
            customer_success_stage=customer_success_stage,
            operator_notes_text=operator_note,
        )
        st.session_state.admin_usage_snapshot = snap
        export_df = core.get_export_events(st.session_state.user["id"])
        notes_df = core.get_operator_notes(st.session_state.user["id"])
        st.session_state.admin_usage_bundle = core.create_admin_usage_dashboard_bundle(
            st.session_state.project_name,
            snap,
            export_events_df=export_df,
            projects_df=projects_df,
            notes_df=notes_df,
            dataset_df=st.session_state.dataset,
        )
        st.success("Admin Usage Dashboard created.")

    if st.session_state.admin_usage_snapshot:
        snap = st.session_state.admin_usage_snapshot
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Admin Score", f"{snap.get('admin_score', 0)}%")
        m2.metric("Decision", snap.get("decision", "Unknown"))
        m3.metric("Ready bundles", snap.get("ready_bundles_in_session", 0))
        m4.metric("Exports logged", snap.get("exports_logged", 0))

        decision = snap.get("decision", "")
        if decision == "ADMIN-READY":
            st.success(snap.get("summary", ""))
        elif decision == "CONDITIONAL":
            st.warning(snap.get("summary", ""))
        else:
            st.error(snap.get("summary", ""))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Utilization")
            st.json(snap.get("utilization", {}))
            st.subheader("Issues")
            issues = snap.get("issues", [])
            if issues:
                for item in issues:
                    sev = item.get("severity", "info")
                    msg = item.get("message", "")
                    if sev == "high":
                        st.error(msg)
                    elif sev == "medium":
                        st.warning(msg)
                    else:
                        st.info(msg)
            else:
                st.success("No admin issues detected.")
        with c2:
            st.subheader("Recommended actions")
            for item in snap.get("recommended_actions", []):
                st.write(f"- {item}")
            st.subheader("Plan limits")
            st.json(snap.get("limits", {}))

        if snap.get("active_bundles"):
            st.subheader("Active bundles in this session")
            st.dataframe(pd.DataFrame(snap.get("active_bundles", [])), use_container_width=True)

        export_df = core.get_export_events(st.session_state.user["id"])
        if len(export_df) > 0:
            st.subheader("Recent export events")
            st.dataframe(export_df.head(50), use_container_width=True)

        notes_df = core.get_operator_notes(st.session_state.user["id"])
        if len(notes_df) > 0:
            st.subheader("Recent operator notes")
            st.dataframe(notes_df.head(50), use_container_width=True)

        if st.session_state.admin_usage_bundle:
            st.download_button(
                "Download Admin Usage Bundle",
                st.session_state.admin_usage_bundle,
                file_name=f"{st.session_state.project_name}_admin_usage_v37.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_admin_usage_bundle_v37",
            )
    else:
        st.info("Build the Admin Usage Dashboard after generating or loading a project. Use it before beta calls, paid pilots and handoffs.")


# ============================================================
# V38 COMMERCIAL LICENSE CERTIFICATE / PAID EXPORT LOCKING
# ============================================================

with license_cert_tab:
    st.header("Commercial License Certificate & Paid Export Locking")
    st.write(
        "V38 turns the paid export gate into a customer-facing delivery certificate: scope, unlocked exports, locked exports, "
        "safe-use terms, certificate hash, validity and delivery conditions. Use this before sending a paid pilot bundle."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        cert_customer_name = st.text_input("Customer name", "Beta Customer", key="v38_cert_customer_name")
        cert_customer_org = st.text_input("Customer organization", "Customer Organization", key="v38_cert_customer_org")
        cert_customer_email = st.text_input("Customer email", "customer@example.com", key="v38_cert_customer_email")
    with c2:
        cert_plan_options = core.get_pricing_plans()
        cert_plan = st.selectbox(
            "Licensed plan",
            cert_plan_options,
            index=cert_plan_options.index(st.session_state.selected_plan) if st.session_state.selected_plan in cert_plan_options else 0,
            key="v38_cert_plan",
        )
        cert_package_options = list(getattr(core, "PAID_PACKAGE_EXPORTS", {"Professional Pilot Bundle": []}).keys())
        cert_package = st.selectbox(
            "Licensed package",
            cert_package_options,
            index=cert_package_options.index("Professional Pilot Bundle") if "Professional Pilot Bundle" in cert_package_options else 0,
            key="v38_cert_package",
        )
    with c3:
        scope_options = core.get_license_scope_options() if hasattr(core, "get_license_scope_options") else ["Paid pilot preparation"]
        cert_scope = st.selectbox("License scope", scope_options, index=2 if len(scope_options) > 2 else 0, key="v38_cert_scope")
        validity_options = core.get_license_validity_options() if hasattr(core, "get_license_validity_options") else ["30 days"]
        cert_validity = st.selectbox("Validity", validity_options, index=1 if len(validity_options) > 1 else 0, key="v38_cert_validity")
        payment_reference = st.text_input("Payment / invoice reference", "Manual invoice / not connected", key="v38_payment_reference")

    c4, c5 = st.columns(2)
    watermark_previews = c4.checkbox("Watermark review/free snapshots", value=True, key="v38_watermark_previews")
    include_dataset_snapshot = c5.checkbox("Include dataset snapshot in certificate bundle", value=True, key="v38_include_dataset_snapshot")

    st.caption(core.COMMERCIAL_LICENSE_DISCLAIMER if hasattr(core, "COMMERCIAL_LICENSE_DISCLAIMER") else "Commercial certificate. Not production certification.")

    if st.button("Issue Commercial License Certificate", type="primary", use_container_width=True, key="v38_issue_certificate"):
        cert = core.build_commercial_license_certificate(
            st.session_state.project_name,
            customer_name=cert_customer_name,
            customer_email=cert_customer_email,
            customer_organization=cert_customer_org,
            selected_plan=cert_plan,
            requested_package=cert_package,
            license_scope=cert_scope,
            validity_label=cert_validity,
            payment_reference=payment_reference,
            dataset_df=st.session_state.dataset,
            paid_license_snapshot=st.session_state.paid_license_snapshot,
            release_success_snapshot=st.session_state.release_success_snapshot,
            customer_assurance_snapshot=st.session_state.customer_assurance_snapshot,
            workspace_lifecycle_snapshot=st.session_state.workspace_lifecycle_snapshot,
            admin_usage_snapshot=st.session_state.admin_usage_snapshot,
            watermark_previews=watermark_previews,
            include_dataset_snapshot=include_dataset_snapshot,
        )
        st.session_state.commercial_license_certificate = cert
        st.session_state.commercial_license_bundle = core.create_commercial_license_certificate_bundle(
            st.session_state.project_name,
            cert,
            st.session_state.dataset,
        )
        if hasattr(core, "record_export_event"):
            core.record_export_event(
                st.session_state.user.get("id"),
                st.session_state.project_id,
                st.session_state.project_name,
                "Commercial License Certificate",
                cert_plan,
                status=cert.get("certificate_status", "created"),
                notes=cert.get("customer_status_line", ""),
            )
        st.success("Commercial License Certificate created.")

    if st.session_state.commercial_license_certificate:
        cert = st.session_state.commercial_license_certificate
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("License Score", f"{cert.get('license_score', 0)}%")
        m2.metric("Decision", cert.get("decision", "Unknown"))
        m3.metric("Status", cert.get("certificate_status", "Unknown"))
        m4.metric("Unlocked", len(cert.get("unlocked_exports", [])))

        status = cert.get("certificate_status", "")
        if status == "ACTIVE":
            st.success(cert.get("customer_status_line", ""))
        elif status in ["REVIEW_REQUIRED", "INTERNAL_REVIEW"]:
            st.warning(cert.get("customer_status_line", ""))
        else:
            st.error(cert.get("customer_status_line", ""))

        if st.session_state.commercial_license_bundle:
            st.download_button(
                "Download Commercial License Certificate Bundle",
                st.session_state.commercial_license_bundle,
                file_name=f"{st.session_state.project_name}_commercial_license_certificate_v38.zip",
                mime="application/zip",
                use_container_width=True,
                key="v38_download_certificate_bundle",
            )

        a1, a2 = st.columns(2)
        with a1:
            st.subheader("Certificate identity")
            st.json({
                "certificate_id": cert.get("certificate_id"),
                "certificate_hash": cert.get("certificate_hash"),
                "created_at": cert.get("created_at"),
                "valid_until": cert.get("commercial_scope", {}).get("valid_until"),
            })
            st.subheader("Blockers")
            blockers = cert.get("blockers", [])
            if blockers:
                for item in blockers:
                    st.error(item)
            else:
                st.success("No hard blockers detected.")
            st.subheader("Warnings")
            warnings_list = cert.get("warnings", [])
            if warnings_list:
                for item in warnings_list:
                    st.warning(item)
            else:
                st.success("No warnings detected.")
        with a2:
            st.subheader("Commercial scope")
            st.json(cert.get("commercial_scope", {}))
            st.subheader("Evidence")
            st.json(cert.get("evidence", {}))

        if cert.get("access_matrix"):
            st.subheader("Export lock/unlock matrix")
            st.dataframe(pd.DataFrame(cert.get("access_matrix", [])), use_container_width=True)

        with st.expander("Safe-use terms and delivery conditions", expanded=False):
            st.markdown("#### Safe-use terms")
            for item in cert.get("safe_use_terms", []):
                st.write(f"- {item}")
            st.markdown("#### Delivery conditions")
            for item in cert.get("delivery_conditions", []):
                st.write(f"- {item}")
    else:
        st.info("Issue this only after the Paid Export Gate, Customer Assurance and Workspace checks are good enough for delivery.")



# ============================================================
# V40 FULL PRODUCT READINESS GATE
# ============================================================

with product_readiness_tab:
    st.header("V40 Full Product Readiness Gate • Security & Access Control Hardening • Customer Delivery Portal • Customer Success Feedback Loop")
    st.write(
        "This is the final consolidated readiness gate before showing EdgeTwin Studio to serious beta users or paid pilot customers. "
        "It checks technical readiness, trust, reliability, governance, operations, field evidence and commercial delivery safety."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Active rows", len(st.session_state.dataset) if isinstance(st.session_state.dataset, pd.DataFrame) else 0)
    c2.metric("Plan", st.session_state.selected_plan)
    c3.metric("Project", st.session_state.project_name[:24])

    st.warning(
        "V40 is a product-readiness decision-support gate. It does not certify production deployment. "
        "Field validation remains required before production or safety-critical use."
    )

    run_v40 = st.button("Run Full Product Readiness Gate", type="primary", use_container_width=True, key="v40_run_product_readiness_gate")
    if run_v40:
        snapshot = core.build_full_product_readiness_gate(
            project_name=st.session_state.project_name,
            dataset_df=st.session_state.dataset,
            selected_plan=st.session_state.selected_plan,
            doctor=st.session_state.fusion_doctor,
            reliability_v2=st.session_state.reliability_v2,
            trust_gate=st.session_state.trust_gate,
            hardening_snapshot=st.session_state.hardening_snapshot,
            release_success_snapshot=st.session_state.release_success_snapshot,
            operational_control_snapshot=st.session_state.operational_control_snapshot,
            observability_snapshot=st.session_state.observability_snapshot,
            customer_assurance_snapshot=st.session_state.customer_assurance_snapshot,
            onboarding_snapshot=st.session_state.onboarding_snapshot,
            workspace_lifecycle_snapshot=st.session_state.workspace_lifecycle_snapshot,
            admin_usage_snapshot=st.session_state.admin_usage_snapshot,
            commercial_license_certificate=st.session_state.commercial_license_certificate,
            field_evidence_v2_snapshot=st.session_state.field_evidence_v2_snapshot,
            deployment_plan=st.session_state.deployment_plan,
            professional_report_snapshot=st.session_state.professional_report_snapshot,
            edge_impulse_snapshot=st.session_state.edge_impulse_snapshot,
            edge_impulse_classifier_snapshot=st.session_state.edge_impulse_classifier_snapshot,
            monetization_snapshot=st.session_state.monetization_snapshot,
            scalability_snapshot=st.session_state.scalability_snapshot,
        )
        st.session_state.product_readiness_v40_snapshot = snapshot
        st.session_state.product_readiness_v40_bundle = core.create_full_product_readiness_bundle(
            st.session_state.project_name,
            snapshot,
            st.session_state.dataset,
        )
        st.success("V40 product readiness gate completed.")

    if st.session_state.product_readiness_v40_snapshot:
        snap = st.session_state.product_readiness_v40_snapshot
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("V40 Readiness", f"{snap.get('final_product_readiness_score', 0)}%")
        m2.metric("Decision", snap.get("decision", "Unknown"))
        m3.metric("Status", snap.get("commercial_status", "Unknown"))
        m4.metric("Labels", snap.get("label_count", 0))

        decision = snap.get("decision", "")
        if decision == "GO":
            st.success(snap.get("customer_summary", ""))
        elif decision == "CONDITIONAL GO":
            st.warning(snap.get("customer_summary", ""))
        else:
            st.error(snap.get("customer_summary", ""))

        if st.session_state.product_readiness_v40_bundle:
            st.download_button(
                "Download V40 Product Readiness Bundle",
                st.session_state.product_readiness_v40_bundle,
                file_name=f"{st.session_state.project_name}_v40_product_readiness_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="v40_download_product_readiness_bundle",
            )

        tabs_v40 = st.tabs(["Components", "Blockers", "Warnings", "Safe Claims", "Next Actions"])
        with tabs_v40[0]:
            st.dataframe(pd.DataFrame(snap.get("components", [])), use_container_width=True)
        with tabs_v40[1]:
            blockers = snap.get("blockers", [])
            if blockers:
                for item in blockers:
                    st.error(item)
            else:
                st.success("No must-fix blockers detected by the V40 gate.")
        with tabs_v40[2]:
            warnings = snap.get("warnings", [])
            if warnings:
                for item in warnings:
                    st.warning(item)
            else:
                st.success("No warnings detected.")
        with tabs_v40[3]:
            st.markdown("#### Safe claims")
            for item in snap.get("safe_claims", []):
                st.write(f"- {item}")
            st.markdown("#### Claims to avoid")
            for item in snap.get("claims_to_avoid", []):
                st.write(f"- {item}")
        with tabs_v40[4]:
            for item in snap.get("next_actions", []):
                st.info(item)
    else:
        st.info("Run the V40 gate after you have generated a dataset and, ideally, run Reliability, Trust, Hardening, Governance, Field Evidence and License checks.")


# ============================================================
# V42 CUSTOMER DELIVERY PORTAL / DELIVERABLES CENTER
# ============================================================

with delivery_tab:
    st.header("Customer Delivery Portal / Deliverables Center")
    st.write(
        "V42 prepares a single customer handoff manifest: what is ready, what is missing, which claims are safe, "
        "and whether the package may be delivered as demo, pilot, real-data review or enterprise handoff."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        delivery_customer_name = st.text_input("Customer name", "Beta Customer", key="v42_delivery_customer_name")
        delivery_customer_org = st.text_input("Customer organization", "Customer Organization", key="v42_delivery_customer_org")
        delivery_customer_email = st.text_input("Customer email", "customer@example.com", key="v42_delivery_customer_email")
    with c2:
        delivery_levels = core.get_delivery_package_levels() if hasattr(core, "get_delivery_package_levels") else ["Professional Pilot Delivery"]
        delivery_level = st.selectbox(
            "Delivery level",
            delivery_levels,
            index=delivery_levels.index("Professional Pilot Delivery") if "Professional Pilot Delivery" in delivery_levels else 0,
            key="v42_delivery_level",
        )
        st.metric("Plan", st.session_state.selected_plan)
        st.metric("Dataset rows", len(st.session_state.dataset) if isinstance(st.session_state.dataset, pd.DataFrame) else 0)
    with c3:
        include_dataset_snapshot_delivery = st.checkbox("Include dataset snapshot", value=True, key="v42_include_dataset_snapshot")
        delivery_notes = st.text_area("Internal handoff notes", "Controlled beta/pilot handoff. Review blockers before delivery.", height=120, key="v42_delivery_notes")

    bundle_flags = {
        "Auto Pilot Bundle": st.session_state.auto_pilot_bundle is not None,
        "Fusion Bundle": st.session_state.fusion_bundle is not None,
        "Enterprise Bundle": st.session_state.enterprise_bundle is not None,
        "Optimizer Bundle": st.session_state.optimizer_bundle is not None,
        "Trust Bundle": st.session_state.trust_bundle is not None,
        "Trust Snapshot": st.session_state.trust_gate is not None,
        "Real Bridge Bundle": st.session_state.real_bridge_bundle is not None,
        "Real Bridge Snapshot": st.session_state.real_bridge_result is not None,
        "Reliability Bundle": st.session_state.reliability_v2_bundle is not None,
        "Reliability Snapshot": st.session_state.reliability_v2 is not None,
        "Deployment Bundle": st.session_state.deployment_bundle is not None,
        "Professional Report": st.session_state.professional_report_bundle is not None,
        "Customer Assurance Bundle": st.session_state.customer_assurance_bundle is not None,
        "Guided Success Bundle": st.session_state.guided_success_bundle is not None,
        "Golden Demo Bundle": st.session_state.golden_demo_bundle is not None,
        "Product Readiness Bundle": st.session_state.product_readiness_v40_bundle is not None,
        "Security Bundle": st.session_state.security_hardening_v41_bundle is not None if "security_hardening_v41_bundle" in st.session_state else False,
        "Field Evidence Bundle": st.session_state.field_evidence_v2_bundle is not None,
        "Field Validation Bundle": st.session_state.field_validation_bundle is not None,
        "Normality Bundle": st.session_state.normality_bundle is not None,
        "Edge Impulse Anomaly Bundle": st.session_state.edge_impulse_bundle is not None,
        "Edge Impulse Classifier Bundle": st.session_state.edge_impulse_classifier_bundle is not None,
        "Commercial License Certificate": st.session_state.commercial_license_bundle is not None,
        "Paid License Bundle": st.session_state.paid_license_bundle is not None,
        "Operational Control Bundle": st.session_state.operational_control_bundle is not None,
        "Production Safety Bundle": st.session_state.observability_bundle is not None,
        "Admin Usage Bundle": st.session_state.admin_usage_bundle is not None,
        "Workspace Lifecycle Bundle": st.session_state.workspace_lifecycle_bundle is not None,
        "Edge Deployment Starter": st.session_state.edge_deployment_starter_bundle is not None,
    }

    if st.button("Build Customer Delivery Portal", type="primary", use_container_width=True, key="v42_build_customer_delivery_portal"):
        snapshot = core.build_customer_delivery_portal(
            project_name=st.session_state.project_name,
            customer_name=delivery_customer_name,
            customer_organization=delivery_customer_org,
            customer_email=delivery_customer_email,
            selected_plan=st.session_state.selected_plan,
            delivery_level=delivery_level,
            dataset_df=st.session_state.dataset,
            bundle_flags=bundle_flags,
            professional_report_snapshot=st.session_state.professional_report_snapshot,
            customer_assurance_snapshot=st.session_state.customer_assurance_snapshot,
            commercial_license_certificate=st.session_state.commercial_license_certificate,
            product_readiness_snapshot=st.session_state.product_readiness_v40_snapshot,
            field_evidence_snapshot=st.session_state.field_evidence_v2_snapshot,
            security_snapshot=st.session_state.security_hardening_v41_snapshot if "security_hardening_v41_snapshot" in st.session_state else None,
            workspace_lifecycle_snapshot=st.session_state.workspace_lifecycle_snapshot,
            admin_usage_snapshot=st.session_state.admin_usage_snapshot,
            notes=delivery_notes,
        )
        st.session_state.customer_delivery_snapshot = snapshot
        dataset_for_delivery = st.session_state.dataset if include_dataset_snapshot_delivery else pd.DataFrame()
        st.session_state.customer_delivery_bundle = core.create_customer_delivery_portal_bundle(
            st.session_state.project_name,
            snapshot,
            dataset_for_delivery,
        )
        if hasattr(core, "record_export_event"):
            core.record_export_event(
                st.session_state.user.get("id"),
                st.session_state.project_id,
                st.session_state.project_name,
                "Customer Delivery Portal",
                st.session_state.selected_plan,
                status=snapshot.get("decision", "created"),
                notes=snapshot.get("customer_status_line", ""),
            )
        st.success("Customer Delivery Portal built.")

    if st.session_state.customer_delivery_snapshot:
        snap = st.session_state.customer_delivery_snapshot
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Delivery Score", f"{snap.get('delivery_score', 0)}%")
        m2.metric("Decision", snap.get("decision", "Unknown"))
        m3.metric("Status", snap.get("portal_status", "Unknown"))
        m4.metric("Required Ready", f"{sum(1 for r in snap.get('required_status', []) if r.get('ready'))}/{len(snap.get('required_status', []))}")

        decision = snap.get("decision", "")
        if decision == "GO":
            st.success(snap.get("customer_status_line", ""))
        elif decision == "CONDITIONAL GO":
            st.warning(snap.get("customer_status_line", ""))
        else:
            st.error(snap.get("customer_status_line", ""))

        if st.session_state.customer_delivery_bundle:
            st.download_button(
                "Download Customer Delivery Bundle",
                st.session_state.customer_delivery_bundle,
                file_name=f"{st.session_state.project_name}_customer_delivery_v42.zip",
                mime="application/zip",
                use_container_width=True,
                key="v42_download_customer_delivery_bundle",
            )

        tabs_delivery = st.tabs(["Deliverables", "Blockers", "Checklist", "Claims", "Handoff"])
        with tabs_delivery[0]:
            st.dataframe(pd.DataFrame(snap.get("deliverables", [])), use_container_width=True)
        with tabs_delivery[1]:
            blockers = snap.get("blockers", [])
            if blockers:
                for item in blockers:
                    st.error(item)
            else:
                st.success("No hard customer-delivery blockers detected.")
            warnings_list = snap.get("warnings", [])
            if warnings_list:
                st.markdown("#### Warnings")
                for item in warnings_list:
                    st.warning(item)
        with tabs_delivery[2]:
            st.dataframe(pd.DataFrame(snap.get("delivery_checklist", [])), use_container_width=True)
        with tabs_delivery[3]:
            st.markdown("#### Safe claims")
            for item in snap.get("safe_claims", []):
                st.write(f"- {item}")
            st.markdown("#### Claims to avoid")
            for item in snap.get("claims_to_avoid", []):
                st.write(f"- {item}")
        with tabs_delivery[4]:
            for item in snap.get("handoff_steps", []):
                st.info(item)
            with st.expander("Delivery metadata", expanded=False):
                st.json({
                    "customer": snap.get("customer", {}),
                    "delivery_level": snap.get("delivery_level"),
                    "license_status": snap.get("license_status"),
                    "price_position": snap.get("price_position"),
                    "disclaimer": snap.get("disclaimer"),
                })
    else:
        st.info("Build this after generating the relevant reports, assurance, license certificate and readiness gates. It becomes the customer handoff page.")



# ============================================================
# V43 CUSTOMER SUCCESS FEEDBACK LOOP
# ============================================================

with customer_success_tab:
    st.header("Customer Success Feedback Loop / Post-Delivery Learning")
    st.write(
        "V43 closes the loop after delivery: collect structured feedback, score customer success, "
        "detect renewal/upgrade potential, identify blockers, and create a clear follow-up plan."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        cs_customer_name = st.text_input("Customer name", "Beta Customer", key="v43_cs_customer_name")
        cs_customer_org = st.text_input("Organization", "Customer Organization", key="v43_cs_customer_org")
        cs_contact_email = st.text_input("Contact email", "customer@example.com", key="v43_cs_contact_email")
    with c2:
        cs_stage = st.selectbox(
            "Success stage",
            core.get_customer_success_stages() if hasattr(core, "get_customer_success_stages") else ["First demo follow-up"],
            index=1 if hasattr(core, "get_customer_success_stages") and len(core.get_customer_success_stages()) > 1 else 0,
            key="v43_cs_stage",
        )
        feedback_channel = st.selectbox(
            "Feedback channel",
            core.get_customer_feedback_channels() if hasattr(core, "get_customer_feedback_channels") else ["Call / meeting"],
            key="v43_feedback_channel",
        )
        requested_next_step = st.selectbox(
            "Requested next step",
            ["Needs more evidence", "Wants paid pilot", "Wants real-data review", "Wants technical call", "Not ready yet", "Enterprise/procurement review"],
            index=1,
            key="v43_requested_next_step",
        )
    with c3:
        follow_up_days = st.number_input("Follow-up in days", min_value=1, max_value=60, value=7, step=1, key="v43_follow_up_days")
        include_dataset_success = st.checkbox("Include dataset snapshot", value=False, key="v43_include_dataset_success")
        st.metric("Plan", st.session_state.selected_plan)

    st.markdown("#### Feedback scoring")
    s1, s2, s3, s4, s5 = st.columns(5)
    customer_satisfaction = s1.slider("Satisfaction", 0, 100, 75, 5, key="v43_customer_satisfaction")
    stakeholder_confidence = s2.slider("Stakeholder confidence", 0, 100, 70, 5, key="v43_stakeholder_confidence")
    technical_fit = s3.slider("Technical fit", 0, 100, 78, 5, key="v43_technical_fit")
    evidence_clarity = s4.slider("Evidence clarity", 0, 100, 72, 5, key="v43_evidence_clarity")
    price_acceptance = s5.slider("Price acceptance", 0, 100, 60, 5, key="v43_price_acceptance")

    friction_text = st.text_area(
        "Friction points / objections, one per line",
        "Needs more real field data\nWants clearer hardware cost estimate",
        height=100,
        key="v43_friction_points",
    )
    success_notes = st.text_area(
        "Customer success notes",
        "Customer understood the pilot workflow and wants to see a stronger real-data evidence path.",
        height=100,
        key="v43_success_notes",
    )

    if st.button("Build Customer Success Feedback Loop", type="primary", use_container_width=True, key="v43_build_customer_success"):
        snapshot = core.build_customer_success_feedback_loop(
            project_name=st.session_state.project_name,
            customer_name=cs_customer_name,
            customer_organization=cs_customer_org,
            customer_email=cs_contact_email,
            selected_plan=st.session_state.selected_plan,
            success_stage=cs_stage,
            feedback_channel=feedback_channel,
            requested_next_step=requested_next_step,
            customer_satisfaction=customer_satisfaction,
            stakeholder_confidence=stakeholder_confidence,
            technical_fit=technical_fit,
            evidence_clarity=evidence_clarity,
            price_acceptance=price_acceptance,
            friction_points=[x.strip() for x in friction_text.splitlines() if x.strip()],
            follow_up_days=int(follow_up_days),
            notes=success_notes,
            customer_delivery_snapshot=st.session_state.customer_delivery_snapshot,
            product_readiness_snapshot=st.session_state.product_readiness_v40_snapshot,
            field_evidence_snapshot=st.session_state.field_evidence_v2_snapshot,
            commercial_license_certificate=st.session_state.commercial_license_certificate,
        )
        st.session_state.customer_success_snapshot = snapshot
        dataset_for_success = st.session_state.dataset if include_dataset_success else pd.DataFrame()
        st.session_state.customer_success_bundle = core.create_customer_success_feedback_bundle(
            st.session_state.project_name,
            snapshot,
            dataset_for_success,
        )
        if hasattr(core, "record_export_event"):
            core.record_export_event(
                st.session_state.user.get("id"),
                st.session_state.project_id,
                st.session_state.project_name,
                "Customer Success Feedback Loop",
                st.session_state.selected_plan,
                status=snapshot.get("decision", "created"),
                notes=snapshot.get("customer_status_line", ""),
            )
        st.success("Customer Success Feedback Loop built.")

    if st.session_state.customer_success_snapshot:
        snap = st.session_state.customer_success_snapshot
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Success Score", f"{snap.get('success_score', 0)}%")
        m2.metric("Decision", snap.get("decision", "Unknown"))
        m3.metric("Churn Risk", snap.get("churn_risk", "Unknown"))
        m4.metric("Upgrade Potential", snap.get("upgrade_potential", "Unknown"))

        decision = snap.get("decision", "")
        if decision == "GO":
            st.success(snap.get("customer_status_line", ""))
        elif decision == "CONDITIONAL GO":
            st.warning(snap.get("customer_status_line", ""))
        else:
            st.error(snap.get("customer_status_line", ""))

        if st.session_state.customer_success_bundle:
            st.download_button(
                "Download Customer Success Bundle",
                st.session_state.customer_success_bundle,
                file_name=f"{st.session_state.project_name}_customer_success_v43.zip",
                mime="application/zip",
                use_container_width=True,
                key="v43_download_customer_success_bundle",
            )

        success_tabs = st.tabs(["Feedback", "Actions", "Risks", "Commercial", "Full snapshot"])
        with success_tabs[0]:
            st.dataframe(pd.DataFrame(snap.get("feedback_scores", [])), use_container_width=True)
            st.markdown("#### Friction points")
            for item in snap.get("friction_points", []):
                st.warning(item)
        with success_tabs[1]:
            st.dataframe(pd.DataFrame(snap.get("follow_up_plan", [])), use_container_width=True)
            st.markdown("#### Next actions")
            for action in snap.get("next_actions", []):
                st.info(action)
        with success_tabs[2]:
            blockers = snap.get("blockers", [])
            if blockers:
                for item in blockers:
                    st.error(item)
            else:
                st.success("No hard customer-success blockers detected.")
            for item in snap.get("warnings", []):
                st.warning(item)
        with success_tabs[3]:
            st.markdown("#### Recommended commercial move")
            st.info(snap.get("recommended_commercial_move", ""))
            st.markdown("#### Safe claims")
            for item in snap.get("safe_claims", []):
                st.write(f"- {item}")
            st.markdown("#### Claims to avoid")
            for item in snap.get("claims_to_avoid", []):
                st.write(f"- {item}")
        with success_tabs[4]:
            st.json(snap)
    else:
        st.info("Use this after a demo, delivery portal, beta call, or paid-pilot handoff to turn feedback into a concrete action plan.")



# ============================================================
# V44 PRICING VALIDATION / OFFER BUILDER
# ============================================================

with pricing_offer_tab:
    st.header("Pricing Validation & Offer Builder • Paid Pilot Launch Readiness")
    st.write(
        "V44 turns the current product evidence into a safe commercial offer: what to sell, "
        "what to include, what to lock, what price range is fair, and which claims are safe."
    )

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        offer_segment = st.selectbox(
            "Customer segment",
            core.get_offer_customer_segments() if hasattr(core, "get_offer_customer_segments") else ["Predictive Maintenance Team"],
            key="v44_offer_segment",
        )
        offer_use_case = st.selectbox(
            "Use-case",
            core.get_use_case_types() if hasattr(core, "get_use_case_types") else ["Predictive Maintenance"],
            key="v44_offer_use_case",
        )
        offer_data_status = st.selectbox(
            "Data status",
            core.get_offer_data_statuses() if hasattr(core, "get_offer_data_statuses") else ["Synthetic-only dataset ready"],
            index=1 if hasattr(core, "get_offer_data_statuses") and len(core.get_offer_data_statuses()) > 1 else 0,
            key="v44_offer_data_status",
        )
    with pc2:
        requested_delivery_goal = st.selectbox(
            "Requested delivery goal",
            core.get_offer_delivery_goals() if hasattr(core, "get_offer_delivery_goals") else ["Professional pilot bundle"],
            index=2 if hasattr(core, "get_offer_delivery_goals") and len(core.get_offer_delivery_goals()) > 2 else 0,
            key="v44_requested_delivery_goal",
        )
        price_sensitivity = st.selectbox(
            "Price sensitivity",
            core.get_price_sensitivity_levels() if hasattr(core, "get_price_sensitivity_levels") else ["Medium"],
            index=1,
            key="v44_price_sensitivity",
        )
        customer_budget_range = st.text_input("Customer budget range", "Unknown / not discussed", key="v44_budget_range")
    with pc3:
        include_dataset_offer = st.checkbox("Include dataset snapshot", value=False, key="v44_include_dataset_snapshot")
        st.metric("Current plan", st.session_state.selected_plan)
        st.metric("Dataset rows", len(st.session_state.dataset) if isinstance(st.session_state.dataset, pd.DataFrame) else 0)

    offer_notes = st.text_area(
        "Internal offer notes",
        "Keep the promise honest: pilot preparation, not production guarantee.",
        height=90,
        key="v44_offer_notes",
    )

    if st.button("Build Pricing Validation Offer", type="primary", use_container_width=True, key="v44_build_pricing_offer"):
        snapshot = core.build_pricing_validation_offer(
            project_name=st.session_state.project_name,
            customer_segment=offer_segment,
            use_case=offer_use_case,
            data_status=offer_data_status,
            requested_delivery_goal=requested_delivery_goal,
            selected_plan=st.session_state.selected_plan,
            price_sensitivity=price_sensitivity,
            customer_budget_range=customer_budget_range,
            dataset_df=st.session_state.dataset,
            trust_gate=st.session_state.trust_gate,
            reliability_v2=st.session_state.reliability_v2,
            product_readiness_snapshot=st.session_state.product_readiness_v40_snapshot,
            field_evidence_snapshot=st.session_state.field_evidence_v2_snapshot,
            customer_delivery_snapshot=st.session_state.customer_delivery_snapshot,
            customer_success_snapshot=st.session_state.customer_success_snapshot,
            commercial_license_certificate=st.session_state.commercial_license_certificate,
            governance_snapshot=st.session_state.customer_assurance_snapshot,
            security_snapshot=st.session_state.security_v41_snapshot,
            notes=offer_notes,
        )
        st.session_state.pricing_offer_snapshot = snapshot
        dataset_for_offer = st.session_state.dataset if include_dataset_offer else pd.DataFrame()
        st.session_state.pricing_offer_bundle = core.create_pricing_validation_offer_bundle(
            st.session_state.project_name,
            snapshot,
            dataset_for_offer,
        )
        if hasattr(core, "record_export_event"):
            core.record_export_event(
                st.session_state.user.get("id"),
                st.session_state.project_id,
                st.session_state.project_name,
                "Pricing Validation Offer",
                st.session_state.selected_plan,
                status=snapshot.get("decision", "created"),
                notes=snapshot.get("customer_facing_offer", {}).get("safe_status_line", ""),
            )
        st.success("Pricing Validation Offer built.")

    if st.session_state.pricing_offer_snapshot:
        snap = st.session_state.pricing_offer_snapshot
        scores = snap.get("scores", {})
        om1, om2, om3, om4 = st.columns(4)
        om1.metric("Offer Score", f"{scores.get('offer_score', 0)}%")
        om2.metric("Decision", snap.get("decision", "Unknown"))
        om3.metric("Recommended Tier", snap.get("recommended_tier", "Unknown"))
        om4.metric("Price Range", snap.get("price_range", "Unknown"))

        decision = snap.get("decision", "")
        status_line = snap.get("customer_facing_offer", {}).get("safe_status_line", "")
        if decision == "GO":
            st.success(status_line)
        elif decision == "CONDITIONAL GO":
            st.warning(status_line)
        else:
            st.error(status_line)

        if st.session_state.pricing_offer_bundle:
            st.download_button(
                "Download Pricing Offer Bundle",
                st.session_state.pricing_offer_bundle,
                file_name=f"{st.session_state.project_name}_pricing_offer_v44.zip",
                mime="application/zip",
                use_container_width=True,
                key="v44_download_pricing_offer_bundle",
            )

        offer_tabs = st.tabs(["Offer", "Deliverables", "Evidence", "Price ladder", "Claims", "Upsell path", "Full snapshot"])
        with offer_tabs[0]:
            st.markdown("#### Customer-facing offer")
            st.json(snap.get("customer_facing_offer", {}))
            st.markdown("#### Value drivers")
            for item in snap.get("value_drivers", []):
                st.info(item)
        with offer_tabs[1]:
            st.dataframe(pd.DataFrame(snap.get("included_deliverables", [])), use_container_width=True)
            st.markdown("#### Not included")
            for item in snap.get("excluded_deliverables", []):
                st.warning(item)
        with offer_tabs[2]:
            st.dataframe(pd.DataFrame(snap.get("evidence_components", [])), use_container_width=True)
            blockers = snap.get("blockers", [])
            if blockers:
                st.markdown("#### Blockers")
                for item in blockers:
                    st.error(item)
            warnings_list = snap.get("warnings", [])
            if warnings_list:
                st.markdown("#### Warnings")
                for item in warnings_list:
                    st.warning(item)
        with offer_tabs[3]:
            st.dataframe(pd.DataFrame(snap.get("offer_matrix", [])), use_container_width=True)
        with offer_tabs[4]:
            st.markdown("#### Safe claims")
            for item in snap.get("safe_claims", []):
                st.write(f"- {item}")
            st.markdown("#### Claims to avoid")
            for item in snap.get("claims_to_avoid", []):
                st.write(f"- {item}")
        with offer_tabs[5]:
            for idx, item in enumerate(snap.get("upsell_path", []), start=1):
                st.info(f"{idx}. {item}")
        with offer_tabs[6]:
            st.json(snap)
    else:
        st.info("Use this before quoting a price or offering a paid pilot. It keeps the offer honest, scoped and commercially strong.")


# ============================================================
# V45 PAID PILOT LAUNCH READINESS
# ============================================================

with paid_pilot_v45_tab:
    st.header("Paid Pilot Launch Readiness V45")
    st.write(
        "This gate decides whether the current offer can safely become a paid pilot. "
        "It connects pricing, license/certificate, delivery, security, governance, field evidence and product readiness into one revenue-ready decision."
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        v45_customer_name = st.text_input("Customer contact name", "Beta Customer", key="v45_customer_name")
        v45_customer_org = st.text_input("Customer organization", "Customer Organization", key="v45_customer_org")
        v45_customer_email = st.text_input("Customer email", "customer@example.com", key="v45_customer_email")
        v45_pilot_use_case = st.text_input(
            "Paid pilot use-case",
            st.session_state.pricing_offer_snapshot.get("use_case", "Predictive Maintenance") if st.session_state.pricing_offer_snapshot else "Predictive Maintenance",
            key="v45_pilot_use_case",
        )
    with p2:
        default_tier = st.session_state.pricing_offer_snapshot.get("recommended_tier", "Professional Pilot Bundle") if st.session_state.pricing_offer_snapshot else "Professional Pilot Bundle"
        v45_offer_tier = st.selectbox(
            "Offer tier",
            ["Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Mini Audit", "Real-Data Pilot Bundle", "Enterprise Custom Review"],
            index=["Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Mini Audit", "Real-Data Pilot Bundle", "Enterprise Custom Review"].index(default_tier) if default_tier in ["Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Mini Audit", "Real-Data Pilot Bundle", "Enterprise Custom Review"] else 1,
            key="v45_offer_tier",
        )
        v45_payment_status = st.selectbox(
            "Payment/commercial status",
            ["Not discussed", "Quote requested", "Quote accepted", "Invoice sent", "Paid / invoice confirmed"],
            index=1,
            key="v45_payment_status",
        )
        v45_contract_status = st.selectbox(
            "Scope / contract status",
            ["Not started", "Draft scope", "Scope reviewed", "Scope accepted", "MSA/NDA required"],
            index=1,
            key="v45_contract_status",
        )
    with p3:
        v45_delivery_window = st.selectbox(
            "Delivery window",
            ["1 business day", "2-3 business days", "1 week", "2 weeks", "Custom"],
            index=2,
            key="v45_delivery_window",
        )
        v45_acceptance_mode = st.selectbox(
            "Acceptance mode",
            ["Internal review only", "Customer demo acceptance", "Paid pilot acceptance", "Field evidence acceptance"],
            index=2,
            key="v45_acceptance_mode",
        )
        v45_include_dataset = st.checkbox("Include dataset snapshot in bundle", value=False, key="v45_include_dataset")
        st.metric("Dataset rows", len(st.session_state.dataset) if isinstance(st.session_state.dataset, pd.DataFrame) else 0)

    v45_notes = st.text_area(
        "Internal close notes",
        "Keep scope tight. Sell pilot preparation and evidence, not production guarantees.",
        height=90,
        key="v45_notes",
    )

    if st.button("Build Paid Pilot Launch Pack", type="primary", use_container_width=True, key="v45_build_paid_pilot"):
        snapshot = core.build_paid_pilot_launch_readiness(
            project_name=st.session_state.project_name,
            customer_name=v45_customer_name,
            customer_organization=v45_customer_org,
            customer_email=v45_customer_email,
            use_case=v45_pilot_use_case,
            offer_tier=v45_offer_tier,
            selected_plan=st.session_state.selected_plan,
            payment_status=v45_payment_status,
            contract_status=v45_contract_status,
            delivery_window=v45_delivery_window,
            acceptance_mode=v45_acceptance_mode,
            dataset_df=st.session_state.dataset,
            pricing_offer_snapshot=st.session_state.pricing_offer_snapshot,
            commercial_license_certificate=st.session_state.commercial_license_certificate,
            customer_delivery_snapshot=st.session_state.customer_delivery_snapshot,
            product_readiness_snapshot=st.session_state.product_readiness_v40_snapshot,
            field_evidence_snapshot=st.session_state.field_evidence_v2_snapshot,
            security_snapshot=st.session_state.security_hardening_v41_snapshot,
            governance_snapshot=st.session_state.customer_assurance_snapshot,
            observability_snapshot=st.session_state.observability_snapshot,
            admin_usage_snapshot=st.session_state.admin_usage_snapshot,
            customer_success_snapshot=st.session_state.customer_success_snapshot,
            notes=v45_notes,
        )
        st.session_state.paid_pilot_v45_snapshot = snapshot
        dataset_for_bundle = st.session_state.dataset if v45_include_dataset else pd.DataFrame()
        st.session_state.paid_pilot_v45_bundle = core.create_paid_pilot_launch_bundle(
            st.session_state.project_name,
            snapshot,
            dataset_for_bundle,
        )
        if hasattr(core, "record_export_event"):
            core.record_export_event(
                st.session_state.user.get("id"),
                st.session_state.project_id,
                st.session_state.project_name,
                "Paid Pilot Launch Pack V45",
                st.session_state.selected_plan,
                status=snapshot.get("decision", "created"),
                notes=snapshot.get("safe_status_line", ""),
            )
        st.success("Paid Pilot Launch Pack built.")

    if st.session_state.paid_pilot_v45_snapshot:
        snap = st.session_state.paid_pilot_v45_snapshot
        scores = snap.get("scores", {})
        a, b, c, d = st.columns(4)
        a.metric("Paid Pilot Score", f"{scores.get('paid_pilot_score', 0)}%")
        b.metric("Decision", snap.get("decision", "Unknown"))
        c.metric("Revenue Status", snap.get("revenue_status", "Unknown"))
        d.metric("Offer Tier", snap.get("offer_tier", "Unknown"))

        if snap.get("decision") == "GO":
            st.success(snap.get("safe_status_line", ""))
        elif snap.get("decision") == "CONDITIONAL GO":
            st.warning(snap.get("safe_status_line", ""))
        else:
            st.error(snap.get("safe_status_line", ""))

        if st.session_state.paid_pilot_v45_bundle:
            st.download_button(
                "Download Paid Pilot Launch Bundle",
                st.session_state.paid_pilot_v45_bundle,
                file_name=f"{st.session_state.project_name}_paid_pilot_v45.zip",
                mime="application/zip",
                use_container_width=True,
                key="v45_download_paid_pilot_bundle",
            )

        v45_tabs = st.tabs(["Readiness", "Before Invoice", "Before Delivery", "Acceptance", "Email Draft", "Claims", "Full Snapshot"])
        with v45_tabs[0]:
            st.dataframe(pd.DataFrame(snap.get("evidence_checks", [])), use_container_width=True)
            for item in snap.get("blockers", []):
                st.error(item)
            for item in snap.get("warnings", []):
                st.warning(item)
        with v45_tabs[1]:
            for item in snap.get("required_before_invoice", []):
                st.info(item)
        with v45_tabs[2]:
            for item in snap.get("required_before_delivery", []):
                st.info(item)
        with v45_tabs[3]:
            st.dataframe(pd.DataFrame(snap.get("acceptance_criteria", [])), use_container_width=True)
        with v45_tabs[4]:
            st.code(snap.get("customer_email_draft", ""), language="text")
        with v45_tabs[5]:
            st.markdown("#### Safe claims")
            for item in snap.get("safe_claims", []):
                st.write(f"- {item}")
            st.markdown("#### Claims to avoid")
            for item in snap.get("claims_to_avoid", []):
                st.write(f"- {item}")
        with v45_tabs[6]:
            st.json(snap)
    else:
        st.info("Use this after the Pricing Offer. V45 decides whether it is safe to move from quote/demo to a real paid pilot delivery.")

# ============================================================
# SIGNAL CANVAS
# ============================================================

with canvas_tab:
    st.header("Signal Canvas")

    d_live = core.generate_universal_signal(2.0, st.session_state.sr, st.session_state.base_f, st.session_state.harm_r, st.session_state.imp_r, st.session_state.noise_l)

    c1, c2 = st.columns(2)

    fig_sig = go.Figure(go.Scatter(x=d_live["t"][:3000], y=d_live["sig"][:3000], mode="lines"))
    fig_sig.update_layout(height=320, title="Synthetic signal", margin=dict(l=0, r=0, t=40, b=0))
    c1.plotly_chart(fig_sig, use_container_width=True)

    fig_fft = go.Figure(go.Scatter(x=d_live["fft_f"], y=d_live["fft_v"], mode="lines"))
    fig_fft.update_layout(height=320, title="FFT", xaxis_range=[0, 1500 if st.session_state.sr == 4000 else 4000], margin=dict(l=0, r=0, t=40, b=0))
    c2.plotly_chart(fig_fft, use_container_width=True)

    features = core.extract_signal_features(d_live["sig"], st.session_state.sr, st.session_state.current_label)

    st.subheader("Extracted features")
    st.json(features)

    if st.button("Add current signal features to audit dataset", key="canvas_add_features_v191"):
        new_row = pd.DataFrame([features])

        if isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset) > 0:
            st.session_state.dataset = pd.concat([st.session_state.dataset, new_row], ignore_index=True)
        else:
            st.session_state.dataset = new_row

        reset_generated_bundles()
        st.success("Added to Enterprise Audit dataset.")


# ============================================================
# INDUSTRY PACKS
# ============================================================

with packs_tab:
    st.header("Industry Packs")

    pack_name = st.selectbox("Choose pack", core.get_industry_packs(), key="industry_pack_select_v191")
    pack = core.get_industry_pack(pack_name)

    st.info(pack.get("description", ""))

    per_class = st.number_input("Samples per class", 10, 1000, 50, 10, key="industry_pack_samples_per_class_v191")

    if st.button("Generate Industry Pack Dataset", type="primary", key="industry_pack_generate_v191"):
        with st.spinner("Generating industry pack..."):
            df, manifest = core.generate_industry_pack_dataset(pack_name, per_class)
            st.session_state.dataset = df
            st.session_state.fusion_manifest = manifest
            reset_generated_bundles()

        st.success("Industry pack generated and loaded into Enterprise Audit.")

    if isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset) > 0:
        st.dataframe(st.session_state.dataset.head(50), use_container_width=True)

        st.download_button(
            "Download current dataset CSV",
            st.session_state.dataset.to_csv(index=False),
            file_name=f"{pack_name.replace(' ', '_')}_dataset.csv",
            mime="text/csv",
            key="industry_pack_download_dataset_v191",
        )


# ============================================================
# HARDWARE ARCHITECT
# ============================================================

with hardware_tab:
    st.header("Hardware Architect")

    target = st.radio("Optimization target", ["balanced", "low_power", "performance", "gateway"], horizontal=True, key="hardware_target_v191")

    default_features = 8

    if isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset.columns):
        default_features = int(max(1, len([c for c in st.session_state.dataset.columns if c != "Label"])))

    num_features = st.number_input("Number of features", 1, 100, default_features, key="hardware_num_features_v191")
    hw_sr = st.number_input("Sample rate", 1000, 48000, int(st.session_state.sr), 1000, key="hardware_sample_rate_v191")

    if st.button("Run Hardware Architect", type="primary", key="hardware_run_architect_v191"):
        st.session_state.hardware_result = core.hardware_auto_architect(num_features, hw_sr, target)

    if st.session_state.hardware_result:
        st.success(st.session_state.hardware_result["reason"])

        ranking = pd.DataFrame(st.session_state.hardware_result["ranking"])

        st.dataframe(ranking, use_container_width=True)

        fig = px.bar(ranking, x="board", y="adjusted_score", title="Hardware ranking")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hardware catalog")
    st.dataframe(core.get_hardware_catalog(), use_container_width=True)
