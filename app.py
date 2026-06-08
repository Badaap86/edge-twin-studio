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

st.set_page_config(page_title="EdgeTwin Studio V80.1", layout="wide", initial_sidebar_state="expanded")

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
        "proposal_sow_snapshot": None,
        "proposal_sow_bundle": None,
        "quote_to_cash_snapshot": None,
        "quote_to_cash_bundle": None,
        "lead_intake_v48_snapshot": None,
        "lead_intake_v48_bundle": None,
        "founder_ops_v49_snapshot": None,
        "founder_ops_v49_bundle": None,
        "customer_mode_v50_snapshot": None,
        "customer_mode_v50_bundle": None,
        "customer_ui_v51_snapshot": None,
        "customer_ui_v51_bundle": None,
        "field_learning_v52_snapshot": None,
        "field_learning_v52_bundle": None,
        "launch_experience_v53_snapshot": None,
        "launch_experience_v53_bundle": None,
        "launch_assets_v54_snapshot": None,
        "launch_assets_v54_bundle": None,
        "first_customer_beta_v55_snapshot": None,
        "first_customer_beta_v55_bundle": None,
        "real_upload_v56_snapshot": None,
        "real_upload_v56_bundle": None,
        "real_upload_v56_features_df": pd.DataFrame(),
        "checkout_v57_snapshot": None,
        "checkout_v57_bundle": None,
        "cloud_architecture_v58_snapshot": None,
        "cloud_architecture_v58_bundle": None,
        "hardware_reference_v59_snapshot": None,
        "hardware_reference_v59_bundle": None,
        "commercial_release_v60_snapshot": None,
        "commercial_release_v60_bundle": None,
        "launch_stabilization_v60_1_snapshot": None,
        "launch_stabilization_v60_1_bundle": None,
        "traction_proof_v61_snapshot": None,
        "traction_proof_v61_bundle": None,
        "roi_value_v62_snapshot": None,
        "roi_value_v62_bundle": None,
        "case_study_v63_snapshot": None,
        "case_study_v63_bundle": None,
        "buyer_dataroom_v64_snapshot": None,
        "buyer_dataroom_v64_bundle": None,
        "ip_moat_v65_snapshot": None,
        "ip_moat_v65_bundle": None,
        "continuous_improvement_v66_snapshot": None,
        "continuous_improvement_v66_bundle": None,
        "reliability_calibration_v67_snapshot": None,
        "reliability_calibration_v67_bundle": None,
        "automation_orchestrator_v68_snapshot": None,
        "automation_orchestrator_v68_bundle": None,
        "zero_touch_v69_snapshot": None,
        "zero_touch_v69_bundle": None,
        "outcome_assurance_v70_snapshot": None,
        "outcome_assurance_v70_bundle": None,
        "customer_support_v71_snapshot": None,
        "customer_support_v71_bundle": None,
        "customer_status_v72_snapshot": None,
        "customer_status_v72_bundle": None,
        "customer_journey_v73_snapshot": None,
        "customer_journey_v73_bundle": None,
        "quality_guardian_v74_snapshot": None,
        "quality_guardian_v74_bundle": None,
        "deliverable_qa_v75_snapshot": None,
        "deliverable_qa_v75_bundle": None,
        "product_consolidation_v76_snapshot": None,
        "product_consolidation_v76_bundle": None,
        "smart_intake_v77_snapshot": None,
        "smart_intake_v77_bundle": None,
        "one_click_pilot_v78_snapshot": None,
        "one_click_pilot_v78_bundle": None,
        "pilot_factory_v79_snapshot": None,
        "pilot_factory_v79_bundle": None,
        "trust_ledger_v80_snapshot": None,
        "trust_ledger_v80_bundle": None,
        "workspace_mode_v50": "Customer Mode",
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

PERSISTED_EXTRA_SESSION_KEYS = [
    "admin_usage_snapshot",
    "api_automation_snapshot",
    "beta_launch_snapshot",
    "checkout_v57_snapshot",
    "cloud_architecture_v58_snapshot",
    "commercial_license_certificate",
    "commercial_release_v60_snapshot",
    "edge_impulse_classifier_snapshot",
    "edge_impulse_snapshot",
    "first_customer_beta_v55_snapshot",
    "golden_demo_result",
    "hardening_snapshot",
    "hardware_reference_v59_snapshot",
    "launch_assets_v54_snapshot",
    "launch_experience_v53_snapshot",
    "normality_result",
    "onboarding_snapshot",
    "operational_control_snapshot",
    "pack_marketplace_snapshot",
    "professional_report_snapshot",
    "real_upload_v56_snapshot",
    "scalability_snapshot",
    "security_hardening_v41_snapshot",
    "security_v41_snapshot",
]


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
    st.session_state.proposal_sow_bundle = None
    st.session_state.quote_to_cash_bundle = None
    st.session_state.lead_intake_v48_bundle = None
    st.session_state.founder_ops_v49_bundle = None
    st.session_state.customer_mode_v50_bundle = None
    st.session_state.customer_ui_v51_bundle = None
    st.session_state.field_learning_v52_bundle = None
    st.session_state.real_upload_v56_bundle = None
    st.session_state.field_evidence_v2_bundle = None
    st.session_state.admin_usage_bundle = None
    st.session_state.observability_bundle = None
    st.session_state.customer_assurance_bundle = None
    st.session_state.onboarding_bundle = None
    st.session_state.guided_success_bundle = None
    st.session_state.workspace_lifecycle_bundle = None
    st.session_state.product_readiness_v40_bundle = None
    st.session_state.security_hardening_v41_bundle = None
    st.session_state.first_customer_beta_v55_bundle = None
    st.session_state.launch_stabilization_v60_1_bundle = None
    st.session_state.traction_proof_v61_bundle = None
    st.session_state.roi_value_v62_bundle = None
    st.session_state.case_study_v63_bundle = None
    st.session_state.buyer_dataroom_v64_bundle = None
    st.session_state.ip_moat_v65_bundle = None
    st.session_state.continuous_improvement_v66_bundle = None
    st.session_state.reliability_calibration_v67_bundle = None
    st.session_state.automation_orchestrator_v68_bundle = None
    st.session_state.zero_touch_v69_bundle = None
    st.session_state.outcome_assurance_v70_bundle = None
    st.session_state.customer_support_v71_bundle = None
    st.session_state.customer_status_v72_bundle = None
    st.session_state.customer_journey_v73_bundle = None
    st.session_state.quality_guardian_v74_bundle = None
    st.session_state.deliverable_qa_v75_bundle = None
    st.session_state.product_consolidation_v76_bundle = None
    st.session_state.smart_intake_v77_bundle = None
    st.session_state.one_click_pilot_v78_bundle = None
    st.session_state.pilot_factory_v79_bundle = None
    st.session_state.trust_ledger_v80_bundle = None


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

workspace_mode_v50 = st.sidebar.radio(
    "Workspace mode",
    ["Customer Mode", "Founder Mode"],
    index=0 if st.session_state.workspace_mode_v50 == "Customer Mode" else 1,
    key="workspace_mode_v50",
    horizontal=False,
)
is_founder_mode = workspace_mode_v50 == "Founder Mode"

if is_founder_mode:
    st.session_state.selected_plan = st.sidebar.selectbox(
        "Access plan",
        core.get_pricing_plans(),
        index=core.get_pricing_plans().index(st.session_state.selected_plan) if st.session_state.selected_plan in core.get_pricing_plans() else 0,
        key="sidebar_selected_plan_v24",
    )
    st.sidebar.caption("Founder-only local plan simulator. Payments are not connected yet.")
else:
    st.sidebar.caption("Customer Mode hides advanced/admin tools and shows the clean pilot route.")

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
        "proposal_sow_snapshot": st.session_state.proposal_sow_snapshot,
        "quote_to_cash_snapshot": st.session_state.quote_to_cash_snapshot,
        "lead_intake_v48_snapshot": st.session_state.lead_intake_v48_snapshot,
        "founder_ops_v49_snapshot": st.session_state.founder_ops_v49_snapshot,
        "customer_mode_v50_snapshot": st.session_state.customer_mode_v50_snapshot,
        "customer_ui_v51_snapshot": st.session_state.customer_ui_v51_snapshot,
        "field_learning_v52_snapshot": st.session_state.field_learning_v52_snapshot,
        "workspace_mode_v50": st.session_state.workspace_mode_v50,
        "observability_snapshot": st.session_state.observability_snapshot,
        "customer_assurance_snapshot": st.session_state.customer_assurance_snapshot,
        "workspace_lifecycle_snapshot": st.session_state.workspace_lifecycle_snapshot,
        "product_readiness_v40_snapshot": st.session_state.product_readiness_v40_snapshot,
        "customer_delivery_snapshot": st.session_state.customer_delivery_snapshot,
        "customer_success_snapshot": st.session_state.customer_success_snapshot,
        "launch_stabilization_v60_1_snapshot": st.session_state.launch_stabilization_v60_1_snapshot,
        "traction_proof_v61_snapshot": st.session_state.traction_proof_v61_snapshot,
        "roi_value_v62_snapshot": st.session_state.roi_value_v62_snapshot,
        "case_study_v63_snapshot": st.session_state.case_study_v63_snapshot,
        "buyer_dataroom_v64_snapshot": st.session_state.buyer_dataroom_v64_snapshot,
        "ip_moat_v65_snapshot": st.session_state.ip_moat_v65_snapshot,
        "continuous_improvement_v66_snapshot": st.session_state.continuous_improvement_v66_snapshot,
        "reliability_calibration_v67_snapshot": st.session_state.reliability_calibration_v67_snapshot,
        "automation_orchestrator_v68_snapshot": st.session_state.automation_orchestrator_v68_snapshot,
        "zero_touch_v69_snapshot": st.session_state.zero_touch_v69_snapshot,
        "outcome_assurance_v70_snapshot": st.session_state.outcome_assurance_v70_snapshot,
        "customer_support_v71_snapshot": st.session_state.customer_support_v71_snapshot,
        "customer_status_v72_snapshot": st.session_state.customer_status_v72_snapshot,
        "customer_journey_v73_snapshot": st.session_state.customer_journey_v73_snapshot,
        "quality_guardian_v74_snapshot": st.session_state.quality_guardian_v74_snapshot,
        "deliverable_qa_v75_snapshot": st.session_state.deliverable_qa_v75_snapshot,
        "product_consolidation_v76_snapshot": st.session_state.product_consolidation_v76_snapshot,
        "smart_intake_v77_snapshot": st.session_state.smart_intake_v77_snapshot,
        "one_click_pilot_v78_snapshot": st.session_state.one_click_pilot_v78_snapshot,
        "pilot_factory_v79_snapshot": st.session_state.pilot_factory_v79_snapshot,
        "trust_ledger_v80_snapshot": st.session_state.trust_ledger_v80_snapshot,
    }

    for key in PERSISTED_EXTRA_SESSION_KEYS:
        if key in st.session_state:
            settings[key] = st.session_state.get(key)

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
            st.session_state.proposal_sow_snapshot = settings.get("proposal_sow_snapshot")
            st.session_state.quote_to_cash_snapshot = settings.get("quote_to_cash_snapshot")
            st.session_state.lead_intake_v48_snapshot = settings.get("lead_intake_v48_snapshot")
            st.session_state.founder_ops_v49_snapshot = settings.get("founder_ops_v49_snapshot")
            st.session_state.customer_mode_v50_snapshot = settings.get("customer_mode_v50_snapshot")
            st.session_state.customer_ui_v51_snapshot = settings.get("customer_ui_v51_snapshot")
            st.session_state.field_learning_v52_snapshot = settings.get("field_learning_v52_snapshot")
            st.session_state.workspace_mode_v50 = settings.get("workspace_mode_v50", st.session_state.workspace_mode_v50)
            st.session_state.security_hardening_v41_snapshot = settings.get("security_hardening_v41_snapshot")
            st.session_state.commercial_license_certificate = settings.get("commercial_license_certificate")
            st.session_state.customer_delivery_snapshot = settings.get("customer_delivery_snapshot")
            st.session_state.customer_success_snapshot = settings.get("customer_success_snapshot")
            st.session_state.observability_snapshot = settings.get("observability_snapshot")
            st.session_state.customer_assurance_snapshot = settings.get("customer_assurance_snapshot")
            st.session_state.workspace_lifecycle_snapshot = settings.get("workspace_lifecycle_snapshot")
            st.session_state.product_readiness_v40_snapshot = settings.get("product_readiness_v40_snapshot")
            st.session_state.launch_stabilization_v60_1_snapshot = settings.get("launch_stabilization_v60_1_snapshot")
            st.session_state.traction_proof_v61_snapshot = settings.get("traction_proof_v61_snapshot")
            st.session_state.roi_value_v62_snapshot = settings.get("roi_value_v62_snapshot")
            st.session_state.case_study_v63_snapshot = settings.get("case_study_v63_snapshot")
            st.session_state.buyer_dataroom_v64_snapshot = settings.get("buyer_dataroom_v64_snapshot")
            st.session_state.ip_moat_v65_snapshot = settings.get("ip_moat_v65_snapshot")
            st.session_state.continuous_improvement_v66_snapshot = settings.get("continuous_improvement_v66_snapshot")
            st.session_state.reliability_calibration_v67_snapshot = settings.get("reliability_calibration_v67_snapshot")
            st.session_state.automation_orchestrator_v68_snapshot = settings.get("automation_orchestrator_v68_snapshot")
            st.session_state.zero_touch_v69_snapshot = settings.get("zero_touch_v69_snapshot")
            st.session_state.outcome_assurance_v70_snapshot = settings.get("outcome_assurance_v70_snapshot")
            st.session_state.customer_support_v71_snapshot = settings.get("customer_support_v71_snapshot")
            st.session_state.customer_status_v72_snapshot = settings.get("customer_status_v72_snapshot")
            st.session_state.customer_journey_v73_snapshot = settings.get("customer_journey_v73_snapshot")
            st.session_state.quality_guardian_v74_snapshot = settings.get("quality_guardian_v74_snapshot")
            st.session_state.deliverable_qa_v75_snapshot = settings.get("deliverable_qa_v75_snapshot")
            st.session_state.product_consolidation_v76_snapshot = settings.get("product_consolidation_v76_snapshot")
            st.session_state.smart_intake_v77_snapshot = settings.get("smart_intake_v77_snapshot")
            st.session_state.one_click_pilot_v78_snapshot = settings.get("one_click_pilot_v78_snapshot")
            st.session_state.pilot_factory_v79_snapshot = settings.get("pilot_factory_v79_snapshot")
            st.session_state.trust_ledger_v80_snapshot = settings.get("trust_ledger_v80_snapshot")
            for key in PERSISTED_EXTRA_SESSION_KEYS:
                if key in settings:
                    st.session_state[key] = settings.get(key, st.session_state.get(key))

            st.session_state.sr = int(settings.get("sr", st.session_state.sr))
            st.session_state.fusion_df = pd.DataFrame()
            st.session_state.auto_pilot_result = None
            reset_generated_bundles()
            st.sidebar.success("Project loaded. Re-run export if you want fresh ZIP/PDF bundles.")
            st.rerun()

if is_founder_mode:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Advanced signal settings")

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
else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Customer flow")
    st.sidebar.caption("Advanced signal sliders, admin screens and engineering tools are hidden in Customer Mode.")

if st.sidebar.button("Logout", use_container_width=True, key="sidebar_logout"):
    del st.session_state.user
    st.rerun()


# ============================================================
# HEADER
# ============================================================


# ============================================================
# HEADER / ORGANIZED NAVIGATION V45.2
# ============================================================

st.title("EdgeTwin Studio V80")
st.caption(
    "Commercial Release Candidate: simple customer mode, full founder control, privacy-safe learning, real upload intake, checkout readiness, cloud migration planning and hardware reference proof. "
    "V80 adds Trust Ledger and Decision Traceability: one evidence receipt explaining what data, gates, risks, approvals and claims support each customer deliverable."
)

st.markdown("""
<style>
.v51-hero {
    padding: 1.05rem 1.15rem;
    border: 1px solid rgba(120,120,120,0.25);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015));
    margin-bottom: 1rem;
}
.v51-card {
    padding: 1rem;
    border: 1px solid rgba(120,120,120,0.22);
    border-radius: 16px;
    min-height: 150px;
    background: rgba(255,255,255,0.03);
}
.v51-badge {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    margin: 0.12rem 0.18rem 0.12rem 0;
    border-radius: 999px;
    border: 1px solid rgba(120,120,120,0.28);
    font-size: 0.86rem;
}
.v51-step {
    padding: 0.65rem 0.75rem;
    border-left: 4px solid rgba(120,120,120,0.55);
    margin: 0.4rem 0;
    border-radius: 10px;
    background: rgba(255,255,255,0.025);
}
.v51-small {
    opacity: 0.78;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

FOUNDER_NAV_GROUPS = {'1. Start & Guided Flow': ['pilot_factory_v79_tab', 'one_click_pilot_v78_tab', 'smart_intake_v77_tab', 'home', 'launch_experience_v53_tab', 'launch_assets_v54_tab', 'first_customer_beta_v55_tab', 'onboarding_tab', 'golden_demo_tab', 'workspace_tab'], '2. Build Pilot': ['pilot_factory_v79_tab', 'one_click_pilot_v78_tab', 'wizard_tab', 'fusion_tab', 'canvas_tab', 'packs_tab', 'marketplace_tab', 'optimizer_tab'], '3. Validate & Trust': ['audit_tab', 'reliability_calibration_v67_tab', 'real_upload_v56_tab', 'real_bridge_tab', 'field_learning_v52_tab', 'normality_tab', 'trust_tab', 'field_validation_tab', 'field_evidence_v2_tab', 'success_gate_tab'], '4. Deploy & Export': ['deployment_tab', 'edge_impulse_tab', 'ei_classifier_tab', 'edge_starter_tab', 'hardware_reference_v59_tab', 'reports_tab', 'api_tab'], '5. Sell & Deliver': ['pilot_factory_v79_tab', 'one_click_pilot_v78_tab', 'deliverable_qa_v75_tab', 'customer_journey_v73_tab', 'customer_status_v72_tab', 'outcome_assurance_v70_tab', 'customer_support_v71_tab', 'ip_moat_v65_tab', 'buyer_dataroom_v64_tab', 'case_study_v63_tab', 'roi_value_v62_tab', 'traction_proof_v61_tab', 'founder_ops_v49_tab', 'lead_intake_v48_tab', 'pricing_offer_tab', 'proposal_sow_tab', 'checkout_v57_tab', 'quote_to_cash_tab', 'monetization_tab', 'paid_export_tab', 'license_cert_tab', 'paid_pilot_v45_tab', 'delivery_tab', 'customer_success_tab', 'closed_beta_tab', 'beta_launch_tab'], '6. Operator & Admin': ['pilot_factory_v79_tab', 'one_click_pilot_v78_tab', 'smart_intake_v77_tab', 'product_consolidation_v76_tab', 'deliverable_qa_v75_tab', 'quality_guardian_v74_tab', 'customer_journey_v73_tab', 'zero_touch_v69_tab', 'customer_status_v72_tab', 'customer_support_v71_tab', 'outcome_assurance_v70_tab', 'automation_orchestrator_v68_tab', 'continuous_improvement_v66_tab', 'launch_stabilization_v60_1_tab', 'commercial_release_v60_tab', 'product_readiness_tab', 'cloud_architecture_v58_tab', 'hardening_tab', 'governance_tab', 'scalability_tab', 'operational_tab', 'observability_tab', 'admin_tab', 'hardware_tab']}
CUSTOMER_NAV_GROUPS = {'1. Start': ['pilot_factory_v79_tab', 'one_click_pilot_v78_tab', 'smart_intake_v77_tab', 'product_consolidation_v76_tab', 'customer_journey_v73_tab', 'zero_touch_v69_tab', 'customer_home_v50_tab', 'automation_orchestrator_v68_tab', 'launch_experience_v53_tab', 'launch_assets_v54_tab', 'first_customer_beta_v55_tab'], '2. Create pilot': ['pilot_factory_v79_tab', 'one_click_pilot_v78_tab', 'wizard_tab', 'real_upload_v56_tab'], '3. Review readiness': ['customer_review_v50_tab', 'reliability_calibration_v67_tab', 'field_learning_v52_tab'], '4. Download / handoff': ['pilot_factory_v79_tab', 'one_click_pilot_v78_tab', 'deliverable_qa_v75_tab', 'customer_status_v72_tab', 'outcome_assurance_v70_tab', 'customer_support_v71_tab', 'reports_tab', 'delivery_tab'], '5. Request proposal': ['roi_value_v62_tab', 'lead_intake_v48_tab', 'pricing_offer_tab', 'proposal_sow_tab', 'checkout_v57_tab']}
NAV_GROUPS = CUSTOMER_NAV_GROUPS if st.session_state.workspace_mode_v50 == 'Customer Mode' else FOUNDER_NAV_GROUPS
NAV_LABELS = {'pilot_factory_v79_tab': '🏭 Pilot Factory V79', 'home': '🏠 Self-Selling Demo', 'wizard_tab': '🧭 Use Case Wizard', 'fusion_tab': '🧬 Sensor Fusion Studio', 'audit_tab': '🩺 Enterprise Audit', 'optimizer_tab': '🧪 Smart Optimizer', 'real_bridge_tab': '🔗 Real Bridge', 'field_learning_v52_tab': '🔐 Privacy Learning V52', 'launch_experience_v53_tab': '🚀 Launch Experience V53', 'launch_assets_v54_tab': '🌐 Launch Assets V54', 'first_customer_beta_v55_tab': '🧪 First Customer Beta V55', 'real_upload_v56_tab': '📥 Real Upload V56', 'trust_tab': '🛡️ Trust Center', 'deployment_tab': '🚀 Deployment Planner', 'reports_tab': '📑 Reports 2.0', 'hardening_tab': '🧰 Product Hardening', 'beta_launch_tab': '🧲 Beta Launch', 'monetization_tab': '💳 Monetization Gate', 'api_tab': '🔌 API Automation', 'marketplace_tab': '🛒 Pack Marketplace', 'normality_tab': '⚖️ Normality Engine', 'edge_impulse_tab': '📤 Edge Impulse Anomaly Export', 'ei_classifier_tab': '🎯 EI Classifier Export', 'success_gate_tab': '✅ Success Gate', 'golden_demo_tab': '🏆 Golden Demo', 'closed_beta_tab': '🚪 Closed Beta', 'paid_export_tab': '🔐 Paid Export', 'field_validation_tab': '🌍 Field Validation', 'field_evidence_v2_tab': '📡 Field Evidence 2.0', 'edge_starter_tab': '🧩 Edge Starter', 'scalability_tab': '📚 Storage/Scale', 'operational_tab': '🕹️ Control Center', 'observability_tab': '🛰️ Error Observatory', 'governance_tab': '🔒 Customer Assurance', 'onboarding_tab': '🧭 Guided Success', 'workspace_tab': '🏢 Workspace', 'admin_tab': '📊 Admin/Usage', 'license_cert_tab': '📜 License Cert', 'product_readiness_tab': '🏁 Product Ready V40', 'delivery_tab': '📦 Delivery Portal', 'customer_success_tab': '💬 Customer Success', 'lead_intake_v48_tab': '🎯 Lead Intake V48', 'pricing_offer_tab': '💶 Pricing Offer', 'proposal_sow_tab': '📝 Proposal / SOW', 'checkout_v57_tab': '🛒 Checkout V57', 'cloud_architecture_v58_tab': '☁️ Cloud Architecture V58', 'hardware_reference_v59_tab': '🧪 Hardware Reference V59', 'traction_proof_v61_tab': '📈 Traction Proof V61', 'roi_value_v62_tab': '💰 ROI Value V62', 'reliability_calibration_v67_tab': '🎚️ Reliability Calibration V67', 'zero_touch_v69_tab': '🧭 Zero-Touch Value V69', 'outcome_assurance_v70_tab': '✅ Outcome Assurance V70', 'customer_support_v71_tab': '🧑‍💻 Support Autopilot V71', 'customer_journey_v73_tab': '🧭 Customer Journey V73', 'one_click_pilot_v78_tab': '⚡ One-Click Pilot V78', 'smart_intake_v77_tab': '🧠 Smart Intake V77', 'product_consolidation_v76_tab': '🧩 Product Consolidation V76', 'deliverable_qa_v75_tab': '📦 Deliverable QA V75', 'quality_guardian_v74_tab': '🛡️ Quality Guardian V74', 'customer_status_v72_tab': '📍 Customer Status V72', 'automation_orchestrator_v68_tab': '🤖 Automation V68', 'continuous_improvement_v66_tab': '♻️ Improvement Flywheel V66', 'ip_moat_v65_tab': '🧬 IP & Moat V65', 'buyer_dataroom_v64_tab': '🗂️ Buyer Data Room V64', 'case_study_v63_tab': '🧾 Case Study V63', 'launch_stabilization_v60_1_tab': '🧊 Launch Stabilizer V60.1', 'commercial_release_v60_tab': '🚦 Commercial Release V60', 'quote_to_cash_tab': '🧾 Quote-to-Cash V47', 'paid_pilot_v45_tab': '🤝 Paid Pilot V45', 'canvas_tab': '📈 Signal Canvas', 'packs_tab': '📦 Industry Packs', 'hardware_tab': '🧱 Hardware Architect'}
NAV_LABELS.update({'customer_home_v50_tab': '✨ Customer Start', 'customer_review_v50_tab': '✅ Readiness & Next Step'})
NAV_HINTS = {
    "1. Start & Guided Flow": "For demos, guided onboarding, golden demo proof and project lifecycle overview.",
    "2. Build Pilot": "Generate datasets, choose sensors/use-cases, create packs and optimize weak datasets.",
    "3. Validate & Trust": "Calibrate reliability, check real-data bridge, normal/abnormal baseline, field evidence and readiness gates.",
    "4. Deploy & Export": "Prepare hardware/deployment plans, reports, Edge Impulse exports and API automation.",
    "5. Sell & Deliver": "Build the ROI/value case, track proof, protect the moat, show one customer status, manage founder workload, qualify leads, create pricing offers, paid pilot checks, licenses, delivery and follow-up.",
    "6. Operator & Admin": "Control deliverable QA, quality guardian self-tests, zero-touch customer value, continuous improvement, launch stability, product health, cloud architecture, governance, storage, errors, admin usage, hardening and hardware catalog.",
    "1. Start": "A calm customer-facing front door with three simple cards, a zero-touch value route, launch-ready copy and no engineering overload.",
    "2. Create pilot": "Generate the pilot package while the advanced engine stays hidden behind the route.",
    "3. Review readiness": "Show customer-safe status badges, calibrated readiness, privacy-safe learning options, gaps and the honest next step.",
    "4. Download / handoff": "Give the customer one clear project status, deliverable QA, relevant reports, bundles, handoff outputs and self-service support answers.",
    "5. Request proposal": "Show the value case, qualify the request, build pricing and create a proposal/SOW when the lead is ready."
}


# V80 navigation: keep the trust ledger visible in the simplified customer route and founder cockpit.
def _v80_add_nav(group_name, page_key, position=0):
    try:
        pages = NAV_GROUPS.get(group_name)
        if isinstance(pages, list) and page_key not in pages:
            pages.insert(position, page_key)
    except Exception:
        pass

NAV_LABELS["trust_ledger_v80_tab"] = "📒 Trust Ledger V80"
_v80_add_nav("1. Start", "trust_ledger_v80_tab", 0)
_v80_add_nav("4. Download / handoff", "trust_ledger_v80_tab", 0)
_v80_add_nav("5. Sell & Deliver", "trust_ledger_v80_tab", 0)
_v80_add_nav("6. Operator & Admin", "trust_ledger_v80_tab", 0)

st.sidebar.markdown("---")
st.sidebar.subheader("Workspace navigation")
nav_group = st.sidebar.radio(
    "Section",
    list(NAV_GROUPS.keys()),
    index=0,
    key=f"workspace_nav_group_v50_{st.session_state.workspace_mode_v50}",
)
nav_options = NAV_GROUPS.get(nav_group, [])
nav_page_key = st.sidebar.selectbox(
    "Open tool",
    nav_options,
    format_func=lambda k: NAV_LABELS.get(k, k),
    key=f"workspace_nav_page_v50_{st.session_state.workspace_mode_v50}_{nav_group}",
)

st.markdown(f"### {NAV_LABELS.get(nav_page_key, nav_page_key)}")
if st.session_state.workspace_mode_v50 == "Customer Mode":
    nav_hint_text = NAV_HINTS.get(nav_group, "Choose the next customer step from the sidebar.")
    st.markdown(
        f'<div class="v51-hero"><b>Simple customer route.</b><br><span class="v51-small">{nav_hint_text}</span></div>',
        unsafe_allow_html=True,
    )
else:
    st.info(NAV_HINTS.get(nav_group, "Choose a workflow step from the sidebar."))
with st.expander("Why V51 stays simple for customers", expanded=False):
    st.write(
        "Customer Mode is intentionally calm: the buyer sees the route, readiness and next step. "
        "The full Dataset Doctor, Reliability Engine, Synthetic-to-Real Bridge, trust gates, reports, pricing and delivery logic still run behind the scenes. "
        "Founder Mode keeps the complete operating system available for you without overwhelming customers."
    )

def render_home():
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

def render_wizard_tab():
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

def render_fusion_tab():
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

def render_audit_tab():
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

def render_optimizer_tab():
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

def render_real_bridge_tab():
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

def render_trust_tab():
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

def render_deployment_tab():
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

def render_reports_tab():
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

def render_hardening_tab():
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

def render_beta_launch_tab():
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

def render_monetization_tab():
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

def render_api_tab():
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

def render_marketplace_tab():
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

def render_normality_tab():
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

def render_edge_impulse_tab():
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

def render_ei_classifier_tab():
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

def render_success_gate_tab():
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

def render_golden_demo_tab():
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

def render_closed_beta_tab():
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

def render_paid_export_tab():
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

def render_field_validation_tab():
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

def render_field_evidence_v2_tab():
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

def render_edge_starter_tab():
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

def render_scalability_tab():
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

def render_operational_tab():
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

def render_observability_tab():
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

def render_governance_tab():
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

def render_onboarding_tab():
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

def render_workspace_tab():
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

def render_admin_tab():
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

def render_license_cert_tab():
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

def render_product_readiness_tab():
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

def render_delivery_tab():
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

def render_customer_success_tab():
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

def render_pricing_offer_tab():
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
# V46 PROPOSAL & STATEMENT OF WORK GENERATOR
# ============================================================

def render_proposal_sow_tab():
    st.header("Proposal & Statement of Work Generator V46")
    st.write(
        "V46 turns the selected offer into a customer-ready proposal and scoped statement of work. "
        "It keeps expectations clear: what is included, what is excluded, what evidence exists, "
        "what milestones apply, and what must be accepted before delivery."
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        customer_name = st.text_input("Customer contact name", "Beta Customer", key="v46_customer_name")
        customer_org = st.text_input("Customer organization", "Customer Organization", key="v46_customer_org")
        customer_email = st.text_input("Customer email", "customer@example.com", key="v46_customer_email")
        proposal_use_case = st.text_input(
            "Proposal use-case",
            st.session_state.pricing_offer_snapshot.get("use_case", "Predictive Maintenance") if st.session_state.pricing_offer_snapshot else "Predictive Maintenance",
            key="v46_use_case",
        )
    with s2:
        default_tier = st.session_state.pricing_offer_snapshot.get("recommended_tier", "Professional Pilot Bundle") if st.session_state.pricing_offer_snapshot else "Professional Pilot Bundle"
        proposal_tier = st.selectbox(
            "Proposed package",
            ["Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Mini Audit", "Real-Data Pilot Bundle", "Enterprise Custom Review"],
            index=["Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Mini Audit", "Real-Data Pilot Bundle", "Enterprise Custom Review"].index(default_tier) if default_tier in ["Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Mini Audit", "Real-Data Pilot Bundle", "Enterprise Custom Review"] else 1,
            key="v46_proposal_tier",
        )
        proposal_valid_days = st.number_input("Proposal valid for days", 7, 90, 14, 1, key="v46_valid_days")
        delivery_window = st.selectbox("Target delivery window", ["2-3 business days", "1 week", "2 weeks", "Custom"], index=1, key="v46_delivery_window")
    with s3:
        scope_mode = st.selectbox(
            "Scope mode",
            ["Fixed-scope paid pilot", "Evidence-building starter", "Real-data audit", "Enterprise/custom review"],
            index=0,
            key="v46_scope_mode",
        )
        include_dataset = st.checkbox("Include dataset snapshot in SOW bundle", value=False, key="v46_include_dataset")
        st.metric("Dataset rows", len(st.session_state.dataset) if isinstance(st.session_state.dataset, pd.DataFrame) else 0)

    assumptions = st.text_area(
        "Customer assumptions / boundaries",
        "Customer provides truthful use-case context. Field validation remains required before production deployment.",
        height=90,
        key="v46_assumptions",
    )
    notes = st.text_area(
        "Internal proposal notes",
        "Keep scope tight. Do not promise guaranteed accuracy, ROI, safety certification or production readiness.",
        height=90,
        key="v46_notes",
    )

    if st.button("Build Proposal & SOW", type="primary", use_container_width=True, key="v46_build_proposal_sow"):
        snapshot = core.build_customer_proposal_sow(
            project_name=st.session_state.project_name,
            customer_name=customer_name,
            customer_organization=customer_org,
            customer_email=customer_email,
            use_case=proposal_use_case,
            proposed_package=proposal_tier,
            selected_plan=st.session_state.selected_plan,
            scope_mode=scope_mode,
            proposal_valid_days=int(proposal_valid_days),
            delivery_window=delivery_window,
            dataset_df=st.session_state.dataset,
            pricing_offer_snapshot=st.session_state.pricing_offer_snapshot,
            paid_pilot_snapshot=st.session_state.paid_pilot_v45_snapshot,
            customer_delivery_snapshot=st.session_state.customer_delivery_snapshot,
            commercial_license_certificate=st.session_state.commercial_license_certificate,
            field_evidence_snapshot=st.session_state.field_evidence_v2_snapshot,
            product_readiness_snapshot=st.session_state.product_readiness_v40_snapshot,
            security_snapshot=st.session_state.security_hardening_v41_snapshot,
            governance_snapshot=st.session_state.customer_assurance_snapshot,
            assumptions=assumptions,
            notes=notes,
        )
        st.session_state.proposal_sow_snapshot = snapshot
        dataset_for_bundle = st.session_state.dataset if include_dataset else pd.DataFrame()
        st.session_state.proposal_sow_bundle = core.create_customer_proposal_sow_bundle(
            st.session_state.project_name,
            snapshot,
            dataset_for_bundle,
        )
        if hasattr(core, "record_export_event"):
            core.record_export_event(
                st.session_state.user.get("id"),
                st.session_state.project_id,
                st.session_state.project_name,
                "Proposal & SOW V46",
                st.session_state.selected_plan,
                status=snapshot.get("decision", "created"),
                notes=snapshot.get("safe_status_line", ""),
            )
        st.success("Proposal & SOW bundle built.")

    if st.session_state.proposal_sow_snapshot:
        snap = st.session_state.proposal_sow_snapshot
        scores = snap.get("scores", {})
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Proposal Score", f"{scores.get('proposal_score', 0)}%")
        p2.metric("Decision", snap.get("decision", "Unknown"))
        p3.metric("Package", snap.get("proposed_package", "Unknown"))
        p4.metric("Validity", f"{snap.get('proposal_valid_days', 0)} days")

        decision = snap.get("decision", "")
        if decision == "GO":
            st.success(snap.get("safe_status_line", ""))
        elif decision == "CONDITIONAL GO":
            st.warning(snap.get("safe_status_line", ""))
        else:
            st.error(snap.get("safe_status_line", ""))

        if st.session_state.proposal_sow_bundle:
            st.download_button(
                "Download Proposal & SOW Bundle",
                st.session_state.proposal_sow_bundle,
                file_name=f"{st.session_state.project_name}_proposal_sow_v46.zip",
                mime="application/zip",
                use_container_width=True,
                key="v46_download_proposal_sow_bundle",
            )

        proposal_tabs = st.tabs(["Proposal", "Scope", "Milestones", "Acceptance", "Risks", "Claims", "Email", "Full snapshot"])
        with proposal_tabs[0]:
            st.markdown("#### Customer proposal text")
            st.text_area("Proposal", snap.get("proposal_text", ""), height=280, key="v46_proposal_text_preview")
        with proposal_tabs[1]:
            st.markdown("#### Included deliverables")
            st.dataframe(pd.DataFrame(snap.get("included_deliverables", [])), use_container_width=True)
            st.markdown("#### Excluded deliverables")
            for item in snap.get("excluded_deliverables", []):
                st.warning(item)
        with proposal_tabs[2]:
            st.dataframe(pd.DataFrame(snap.get("milestones", [])), use_container_width=True)
            st.markdown("#### Responsibilities")
            st.dataframe(pd.DataFrame(snap.get("responsibilities", [])), use_container_width=True)
        with proposal_tabs[3]:
            st.dataframe(pd.DataFrame(snap.get("acceptance_criteria", [])), use_container_width=True)
            st.markdown("#### Required before signature / delivery")
            for item in snap.get("required_before_signature", []):
                st.info(item)
            for item in snap.get("required_before_delivery", []):
                st.info(item)
        with proposal_tabs[4]:
            blockers = snap.get("blockers", [])
            if blockers:
                for item in blockers:
                    st.error(item)
            else:
                st.success("No hard SOW blockers detected.")
            for item in snap.get("warnings", []):
                st.warning(item)
        with proposal_tabs[5]:
            st.markdown("#### Safe claims")
            for item in snap.get("safe_claims", []):
                st.write(f"- {item}")
            st.markdown("#### Claims to avoid")
            for item in snap.get("claims_to_avoid", []):
                st.write(f"- {item}")
        with proposal_tabs[6]:
            st.text_area("Customer email draft", snap.get("customer_email_draft", ""), height=260, key="v46_email_draft_preview")
        with proposal_tabs[7]:
            st.json(snap)
    else:
        st.info("Use this after Pricing Offer V44 and before Paid Pilot V45 delivery. It creates a clear customer proposal and SOW so the scope cannot drift.")


# ============================================================
# V45 PAID PILOT LAUNCH READINESS
# ============================================================

def render_paid_pilot_v45_tab():
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

def render_canvas_tab():
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

def render_packs_tab():
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

def render_hardware_tab():
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




# ============================================================
# V47 QUOTE-TO-CASH / INVOICE READINESS CENTER
# ============================================================

def render_quote_to_cash_tab():
    st.header("Quote-to-Cash & Invoice Readiness V47")
    st.write(
        "This gate turns a scoped paid pilot offer into a clean commercial handoff: quote, invoice readiness, payment checklist, line items, safe terms and delivery gating. "
        "It does not process payments and is not legal/tax/accounting advice."
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        customer_name = st.text_input("Customer contact name", "Beta Customer", key="v47_customer_name")
        customer_org = st.text_input("Customer organization", "Customer Organization", key="v47_customer_org")
        customer_email = st.text_input("Customer billing email", "customer@example.com", key="v47_customer_email")
        offer_tier_default = "Professional Pilot Bundle"
        if st.session_state.pricing_offer_snapshot:
            offer_tier_default = st.session_state.pricing_offer_snapshot.get("recommended_tier", offer_tier_default)
        offer_tier = st.selectbox(
            "Offer tier",
            ["Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Mini Audit", "Real-Data Pilot Bundle", "Enterprise Custom Review"],
            index=["Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Mini Audit", "Real-Data Pilot Bundle", "Enterprise Custom Review"].index(offer_tier_default) if offer_tier_default in ["Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Mini Audit", "Real-Data Pilot Bundle", "Enterprise Custom Review"] else 1,
            key="v47_offer_tier",
        )
    with p2:
        currency = st.selectbox("Currency", ["EUR", "USD", "GBP", "HUF"], index=0, key="v47_currency")
        suggested_price = 499.0
        if st.session_state.pricing_offer_snapshot:
            pr = st.session_state.pricing_offer_snapshot.get("price_range", {})
            suggested_price = float(pr.get("mid", pr.get("high", pr.get("low", suggested_price))) or suggested_price) if isinstance(pr, dict) else suggested_price
        quoted_price = st.number_input("Quoted pilot price", min_value=0.0, max_value=50000.0, value=float(suggested_price), step=50.0, key="v47_quoted_price")
        vat_rate = st.number_input("VAT/tax estimate %", min_value=0.0, max_value=50.0, value=0.0, step=1.0, key="v47_vat_rate")
        payment_terms = st.selectbox("Payment terms", ["Due on receipt", "Net 7", "Net 14", "50% upfront / 50% before delivery", "Custom"], index=0, key="v47_payment_terms")
    with p3:
        payment_status = st.selectbox("Payment status", ["Not discussed", "Quote requested", "Quote accepted", "Invoice sent", "Paid / invoice confirmed"], index=1, key="v47_payment_status")
        invoice_details_status = st.selectbox("Invoice details", ["Missing", "Requested", "Received", "Verified"], index=1, key="v47_invoice_details_status")
        legal_scope_status = st.selectbox("Scope/SOW status", ["Not started", "Draft scope", "Scope reviewed", "Scope accepted", "Signed / accepted"], index=1, key="v47_legal_scope_status")
        delivery_status = st.selectbox("Delivery status", ["Not delivered yet", "Delivery bundle ready", "Delivered"], index=0, key="v47_delivery_status")
        license_status = st.selectbox("License/certificate status", ["Not issued", "Draft issued", "Issued", "Signed / accepted"], index=0, key="v47_license_status")

    with st.expander("Purchase order / procurement options", expanded=False):
        customer_po_required = st.checkbox("Customer requires PO before invoice", value=False, key="v47_po_required")
        customer_po_status = st.selectbox("PO status", ["Not required", "Requested", "Received", "Approved"], index=0, key="v47_po_status")

    if st.button("Build Quote-to-Cash Readiness Pack", type="primary", use_container_width=True, key="v47_build_quote_to_cash"):
        snap = core.build_quote_to_cash_invoice_readiness(
            customer_name=customer_name,
            customer_org=customer_org,
            customer_email=customer_email,
            offer_tier=offer_tier,
            currency=currency,
            quoted_price=quoted_price,
            vat_rate=vat_rate,
            payment_terms=payment_terms,
            payment_status=payment_status,
            invoice_details_status=invoice_details_status,
            legal_scope_status=legal_scope_status,
            delivery_status=delivery_status,
            license_status=license_status,
            customer_po_required=customer_po_required,
            customer_po_status=customer_po_status,
            paid_pilot_snapshot=st.session_state.paid_pilot_v45_snapshot,
            proposal_sow_snapshot=st.session_state.proposal_sow_snapshot,
            pricing_offer_snapshot=st.session_state.pricing_offer_snapshot,
            delivery_snapshot=st.session_state.customer_delivery_snapshot,
            license_certificate=st.session_state.commercial_license_certificate,
        )
        st.session_state.quote_to_cash_snapshot = snap
        st.session_state.quote_to_cash_bundle = core.create_quote_to_cash_invoice_bundle(
            st.session_state.project_name,
            snap,
            st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        st.success("Quote-to-Cash readiness pack built.")

    if st.session_state.quote_to_cash_snapshot:
        snap = st.session_state.quote_to_cash_snapshot
        a, b, c, d = st.columns(4)
        a.metric("Quote-to-Cash Score", f"{snap.get('quote_to_cash_score', 0)}%")
        b.metric("Decision", snap.get("decision", "Unknown"))
        c.metric("Revenue status", snap.get("revenue_status", "Unknown"))
        d.metric("Total", f"{snap.get('currency', 'EUR')} {snap.get('total_amount', 0)}")

        if snap.get("decision") == "GO":
            st.success(snap.get("revenue_status", ""))
        elif snap.get("decision") == "CONDITIONAL GO":
            st.warning(snap.get("revenue_status", ""))
        else:
            st.error(snap.get("revenue_status", ""))

        if st.session_state.quote_to_cash_bundle:
            st.download_button(
                "Download Quote-to-Cash Bundle V47",
                st.session_state.quote_to_cash_bundle,
                file_name=f"{st.session_state.project_name}_quote_to_cash_v47.zip",
                mime="application/zip",
                use_container_width=True,
                key="v47_download_quote_to_cash_bundle",
            )

        tabs = st.tabs(["Line items", "Checks", "Blockers", "Terms", "Customer email", "Full snapshot"])
        with tabs[0]:
            st.dataframe(pd.DataFrame(snap.get("line_items", [])), use_container_width=True)
        with tabs[1]:
            st.dataframe(pd.DataFrame(snap.get("ready_checks", [])), use_container_width=True)
            st.markdown("#### Payment checklist")
            for item in snap.get("payment_checklist", []):
                st.info(item)
        with tabs[2]:
            blockers = snap.get("blockers", [])
            if blockers:
                for item in blockers:
                    st.error(item)
            else:
                st.success("No hard invoice blockers detected.")
            for item in snap.get("warnings", []):
                st.warning(item)
        with tabs[3]:
            for item in snap.get("quote_terms", []):
                st.write(f"- {item}")
            st.caption(snap.get("disclaimer", ""))
        with tabs[4]:
            st.text_area("Customer email draft", snap.get("customer_email_draft", ""), height=280, key="v47_email_preview")
        with tabs[5]:
            st.json(snap)
    else:
        st.info("Use this after Pricing Offer V44 and Proposal/SOW V46. V47 prepares quote, invoice readiness and payment checklist without processing payments.")


# ============================================================
# V48 CUSTOMER LEAD INTAKE / QUALIFICATION CENTER
# ============================================================

def render_lead_intake_v48_tab():
    st.header("Customer Lead Intake & Qualification V48")
    st.write(
        "This turns an interested customer into a clean next action: qualify the use case, check readiness, flag risks, "
        "prepare discovery questions and route the lead to demo, proposal, paid pilot or nurture."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        customer_name = st.text_input("Contact name", "Beta Customer", key="v48_contact_name")
        customer_org = st.text_input("Organization", "Customer Organization", key="v48_customer_org")
        customer_email = st.text_input("Email", "customer@example.com", key="v48_customer_email")
        industry = st.selectbox(
            "Industry / domain",
            ["Manufacturing", "Construction / Site Security", "Smart Forestry", "Agriculture", "Utilities / Remote Assets", "Research / Lab", "Other"],
            index=0,
            key="v48_industry",
        )
    with c2:
        use_case = st.selectbox(
            "Requested use case",
            ["Predictive maintenance", "Anomaly detection", "Security / intrusion", "Tamper detection", "Machine health", "Environmental monitoring", "Custom sensor pilot"],
            index=0,
            key="v48_use_case",
        )
        urgency = st.selectbox(
            "Urgency",
            ["Exploring only", "Pilot in 1-3 months", "Pilot this month", "Urgent operational pain"],
            index=1,
            key="v48_urgency",
        )
        budget_range = st.selectbox(
            "Budget signal",
            ["Unknown", "Below €500", "€500-€1,500", "€1,500-€5,000", "€5,000+"],
            index=2,
            key="v48_budget_range",
        )
        decision_status = st.selectbox(
            "Decision status",
            ["Just researching", "Technical evaluator", "Budget owner involved", "Decision maker involved"],
            index=1,
            key="v48_decision_status",
        )
    with c3:
        data_status = st.selectbox(
            "Real data status",
            ["No data yet", "Can collect data", "Sample data available", "Labeled dataset available"],
            index=1,
            key="v48_data_status",
        )
        sensor_status = st.selectbox(
            "Sensor / hardware status",
            ["Unknown", "Need hardware advice", "Sensors selected", "Hardware already installed"],
            index=1,
            key="v48_sensor_status",
        )
        deployment_environment = st.selectbox(
            "Deployment environment",
            ["Indoor machine", "Outdoor asset", "Remote/off-grid", "Vehicle/mobile", "Mixed/unknown"],
            index=0,
            key="v48_deployment_environment",
        )
        technical_owner = st.selectbox(
            "Technical owner availability",
            ["Unknown", "No technical owner", "Part-time technical owner", "Dedicated technical owner"],
            index=2,
            key="v48_technical_owner",
        )

    with st.expander("Risk and scope checks", expanded=False):
        safety_critical = st.checkbox("Use case could affect safety-critical decisions", value=False, key="v48_safety_critical")
        regulated_context = st.checkbox("Regulated/compliance-heavy environment", value=False, key="v48_regulated_context")
        needs_installation = st.checkbox("Customer expects on-site installation", value=False, key="v48_needs_installation")
        wants_guaranteed_accuracy = st.checkbox("Customer asks for guaranteed accuracy / production certainty", value=False, key="v48_guaranteed_accuracy")
        requested_outcome = st.text_area(
            "Customer requested outcome",
            "We want to know if this sensor idea can become a reliable Edge AI pilot and what hardware/data we need.",
            height=110,
            key="v48_requested_outcome",
        )

    if st.button("Build Lead Qualification Pack", type="primary", use_container_width=True, key="v48_build_lead_intake"):
        snap = core.build_customer_lead_intake_qualification(
            customer_name=customer_name,
            customer_org=customer_org,
            customer_email=customer_email,
            industry=industry,
            use_case=use_case,
            urgency=urgency,
            budget_range=budget_range,
            decision_status=decision_status,
            data_status=data_status,
            sensor_status=sensor_status,
            deployment_environment=deployment_environment,
            technical_owner=technical_owner,
            requested_outcome=requested_outcome,
            safety_critical=safety_critical,
            regulated_context=regulated_context,
            needs_installation=needs_installation,
            wants_guaranteed_accuracy=wants_guaranteed_accuracy,
            pricing_offer_snapshot=st.session_state.pricing_offer_snapshot,
            proposal_sow_snapshot=st.session_state.proposal_sow_snapshot,
            quote_to_cash_snapshot=st.session_state.quote_to_cash_snapshot,
            paid_pilot_snapshot=st.session_state.paid_pilot_v45_snapshot,
        )
        st.session_state.lead_intake_v48_snapshot = snap
        st.session_state.lead_intake_v48_bundle = core.create_customer_lead_intake_bundle(
            st.session_state.project_name,
            snap,
            st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        st.success("Lead qualification pack built.")

    if st.session_state.lead_intake_v48_snapshot:
        snap = st.session_state.lead_intake_v48_snapshot
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Qualification", f"{snap.get('qualification_score', 0)}%")
        m2.metric("Fit", snap.get("fit_level", "Unknown"))
        m3.metric("Route", snap.get("recommended_route", "Unknown"))
        m4.metric("Commercial stage", snap.get("commercial_stage", "Unknown"))

        if snap.get("fit_level") in ["High-fit paid pilot", "Good-fit pilot"]:
            st.success(snap.get("operator_verdict", ""))
        elif snap.get("fit_level") == "Nurture / needs clarity":
            st.warning(snap.get("operator_verdict", ""))
        else:
            st.error(snap.get("operator_verdict", ""))

        if st.session_state.lead_intake_v48_bundle:
            st.download_button(
                "Download Lead Intake Bundle V48",
                st.session_state.lead_intake_v48_bundle,
                file_name=f"{st.session_state.project_name}_lead_intake_v48.zip",
                mime="application/zip",
                use_container_width=True,
                key="v48_download_lead_intake_bundle",
            )

        tabs = st.tabs(["Next actions", "Discovery questions", "Risks", "Customer email", "Routing", "Full snapshot"])
        with tabs[0]:
            for item in snap.get("next_actions", []):
                st.info(item)
            st.markdown("#### Customer required inputs")
            for item in snap.get("customer_required_inputs", []):
                st.write(f"- {item}")
        with tabs[1]:
            for item in snap.get("discovery_questions", []):
                st.write(f"- {item}")
        with tabs[2]:
            blockers = snap.get("blockers", [])
            warnings = snap.get("warnings", [])
            if blockers:
                for item in blockers:
                    st.error(item)
            else:
                st.success("No hard lead blockers detected.")
            for item in warnings:
                st.warning(item)
            st.markdown("#### Claims to avoid")
            for item in snap.get("claims_to_avoid", []):
                st.caption(f"- {item}")
        with tabs[3]:
            st.text_area("Customer email draft", snap.get("customer_email_draft", ""), height=320, key="v48_email_preview")
        with tabs[4]:
            st.json(snap.get("routing", {}))
            st.markdown("#### Sales follow-up checklist")
            for item in snap.get("sales_followup_checklist", []):
                st.write(f"- {item}")
        with tabs[5]:
            st.json(snap)
    else:
        st.info("Use this before Pricing Offer/SOW. V48 helps avoid bad-fit leads and turns serious leads into a clean next step.")


# ============================================================
# V49 FOUNDER OPS / FOLLOW-UP AUTOMATION CENTER
# ============================================================

def render_founder_ops_v49_tab():
    st.header("Founder Ops & Customer Follow-up V49")
    st.write(
        "This page turns customer interest into a practical founder task queue: what to do now, what to automate, "
        "what to ask the customer, and what not to spend time on. Built for one busy founder, not a big sales team."
    )

    snapshot_status = {
        "Lead Intake V48": bool(st.session_state.lead_intake_v48_snapshot),
        "Pricing Offer V44": bool(st.session_state.pricing_offer_snapshot),
        "Proposal/SOW V46": bool(st.session_state.proposal_sow_snapshot),
        "Quote-to-Cash V47": bool(st.session_state.quote_to_cash_snapshot),
        "Paid Pilot V45": bool(st.session_state.paid_pilot_v45_snapshot),
        "Delivery Portal": bool(st.session_state.customer_delivery_snapshot),
        "Customer Success": bool(st.session_state.customer_success_snapshot),
    }
    st.caption("Connected workflow signals")
    sc = st.columns(len(snapshot_status))
    for idx, (name, ok) in enumerate(snapshot_status.items()):
        sc[idx].metric(name, "Ready" if ok else "Missing")

    c1, c2, c3 = st.columns(3)
    with c1:
        customer_name = st.text_input("Customer contact", "Beta Customer", key="v49_customer_name")
        customer_org = st.text_input("Organization", "Customer Organization", key="v49_customer_org")
        customer_email = st.text_input("Customer email", "customer@example.com", key="v49_customer_email")
        current_stage = st.selectbox(
            "Current stage",
            ["New lead", "Discovery", "Pricing", "Proposal/SOW", "Quote sent", "Awaiting payment", "Paid pilot", "Delivery", "Post-delivery follow-up"],
            index=1,
            key="v49_current_stage",
        )
    with c2:
        founder_hours = st.slider("Founder hours available this week", 1, 40, 8, key="v49_founder_hours")
        max_customer_hours = st.slider("Max custom hours for this customer this week", 1, 20, 3, key="v49_max_customer_hours")
        urgency = st.selectbox("Customer urgency", ["Low", "Normal", "High", "Critical"], index=1, key="v49_customer_urgency")
        customer_responsiveness = st.selectbox(
            "Customer responsiveness",
            ["Unknown", "Slow", "Normal", "Fast"],
            index=2,
            key="v49_customer_responsiveness",
        )
    with c3:
        payment_status = st.selectbox(
            "Payment status",
            ["Not discussed", "Quote requested", "Invoice details missing", "Invoice ready", "Awaiting payment", "Paid", "Overdue"],
            index=0,
            key="v49_payment_status",
        )
        data_status = st.selectbox(
            "Data/status",
            ["No data", "Customer can collect", "Sample data received", "Labeled data received", "Dataset ready"],
            index=1,
            key="v49_data_status",
        )
        next_deadline = st.selectbox(
            "Next deadline pressure",
            ["None", "This week", "48 hours", "Today"],
            index=0,
            key="v49_deadline_pressure",
        )
        support_load = st.selectbox("Support load", ["Low", "Normal", "High", "Too high"], index=1, key="v49_support_load")

    with st.expander("Founder workload and risk flags", expanded=False):
        waiting_on_customer = st.checkbox("Waiting on customer input", value=True, key="v49_waiting_customer")
        needs_follow_up = st.checkbox("Needs follow-up email", value=True, key="v49_needs_followup")
        unclear_scope = st.checkbox("Scope is still unclear", value=False, key="v49_unclear_scope")
        high_custom_work = st.checkbox("Customer is pulling toward too much custom work", value=False, key="v49_high_custom_work")
        unsafe_expectation = st.checkbox("Customer expects guarantees / production certainty", value=False, key="v49_unsafe_expectation")
        internal_notes = st.text_area(
            "Internal notes",
            "Keep the next step small, paid, and clearly scoped. Avoid unpaid custom engineering.",
            height=100,
            key="v49_internal_notes",
        )
        outstanding_inputs_text = st.text_area(
            "Outstanding customer inputs, one per line",
            "Representative sample data or permission to generate synthetic pilot data\nNamed technical contact\nAcceptance criteria for first pilot",
            height=110,
            key="v49_outstanding_inputs",
        )

    if st.button("Build Founder Ops Plan", type="primary", use_container_width=True, key="v49_build_founder_ops"):
        outstanding_inputs = [line.strip() for line in outstanding_inputs_text.splitlines() if line.strip()]
        snap = core.build_founder_ops_followup_center(
            customer_name=customer_name,
            customer_org=customer_org,
            customer_email=customer_email,
            current_stage=current_stage,
            founder_hours_available=founder_hours,
            max_customer_hours=max_customer_hours,
            urgency=urgency,
            customer_responsiveness=customer_responsiveness,
            payment_status=payment_status,
            data_status=data_status,
            next_deadline=next_deadline,
            support_load=support_load,
            waiting_on_customer=waiting_on_customer,
            needs_follow_up=needs_follow_up,
            unclear_scope=unclear_scope,
            high_custom_work=high_custom_work,
            unsafe_expectation=unsafe_expectation,
            outstanding_customer_inputs=outstanding_inputs,
            internal_notes=internal_notes,
            lead_intake_snapshot=st.session_state.lead_intake_v48_snapshot,
            pricing_offer_snapshot=st.session_state.pricing_offer_snapshot,
            proposal_sow_snapshot=st.session_state.proposal_sow_snapshot,
            quote_to_cash_snapshot=st.session_state.quote_to_cash_snapshot,
            paid_pilot_snapshot=st.session_state.paid_pilot_v45_snapshot,
            delivery_snapshot=st.session_state.customer_delivery_snapshot,
            customer_success_snapshot=st.session_state.customer_success_snapshot,
            observability_snapshot=st.session_state.observability_snapshot,
        )
        st.session_state.founder_ops_v49_snapshot = snap
        st.session_state.founder_ops_v49_bundle = core.create_founder_ops_followup_bundle(
            st.session_state.project_name,
            snap,
            st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        st.success("Founder Ops plan built.")

    if st.session_state.founder_ops_v49_snapshot:
        snap = st.session_state.founder_ops_v49_snapshot
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ops score", f"{snap.get('ops_score', 0)}%")
        m2.metric("Mode", snap.get("operator_mode", "Unknown"))
        m3.metric("Overload risk", snap.get("overload_risk", "Unknown"))
        m4.metric("Do-now tasks", len(snap.get("do_now_queue", [])))

        if snap.get("overload_risk") == "Low":
            st.success(snap.get("founder_verdict", ""))
        elif snap.get("overload_risk") == "Medium":
            st.warning(snap.get("founder_verdict", ""))
        else:
            st.error(snap.get("founder_verdict", ""))

        if st.session_state.founder_ops_v49_bundle:
            st.download_button(
                "Download Founder Ops Bundle V49",
                st.session_state.founder_ops_v49_bundle,
                file_name=f"{st.session_state.project_name}_founder_ops_v49.zip",
                mime="application/zip",
                use_container_width=True,
                key="v49_download_founder_ops_bundle",
            )

        tabs = st.tabs(["Do now", "Task queue", "Emails", "Weekly plan", "Automate/stop", "Risks", "Full snapshot"])
        with tabs[0]:
            for item in snap.get("do_now_queue", []):
                st.info(f"{item.get('priority', '')}: {item.get('task', '')} — {item.get('time_box', '')}")
        with tabs[1]:
            st.dataframe(pd.DataFrame(snap.get("task_queue", [])), use_container_width=True)
        with tabs[2]:
            st.text_area("Customer follow-up email", snap.get("customer_followup_email", ""), height=260, key="v49_customer_email_preview")
            st.text_area("Internal operator note", snap.get("internal_operator_note", ""), height=180, key="v49_internal_note_preview")
        with tabs[3]:
            st.dataframe(pd.DataFrame(snap.get("weekly_plan", [])), use_container_width=True)
        with tabs[4]:
            st.markdown("#### Automate next")
            for item in snap.get("automation_opportunities", []):
                st.write(f"- {item}")
            st.markdown("#### Stop doing")
            for item in snap.get("stop_doing_list", []):
                st.warning(item)
        with tabs[5]:
            blockers = snap.get("blockers", [])
            warnings = snap.get("warnings", [])
            if blockers:
                for item in blockers:
                    st.error(item)
            else:
                st.success("No hard founder-ops blockers detected.")
            for item in warnings:
                st.warning(item)
            st.caption(snap.get("disclaimer", ""))
        with tabs[6]:
            st.json(snap)
    else:
        st.info("Use this after Lead Intake or any customer conversation. V49 helps you protect your time and keep next actions controlled.")


# ============================================================
# V50 CUSTOMER MODE / SIMPLIFIED CUSTOMER ROUTE
# ============================================================


def _v51_badge(text):
    return f"<span class=\"v51-badge\">{text}</span>"


def _build_v51_customer_snapshot():
    dataset_df = st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame()
    snap = core.build_customer_ui_v51_summary(
        project_name=st.session_state.project_name,
        dataset_df=dataset_df,
        auto_pilot_result=st.session_state.auto_pilot_result,
        fusion_doctor=st.session_state.fusion_doctor,
        trust_gate=st.session_state.trust_gate,
        pricing_offer_snapshot=st.session_state.pricing_offer_snapshot,
        proposal_sow_snapshot=st.session_state.proposal_sow_snapshot,
        paid_pilot_snapshot=st.session_state.paid_pilot_v45_snapshot,
        delivery_snapshot=st.session_state.customer_delivery_snapshot,
        lead_intake_snapshot=st.session_state.lead_intake_v48_snapshot,
    )
    st.session_state.customer_ui_v51_snapshot = snap
    # Keep V50 snapshot populated for backwards compatibility with saved projects/bundles.
    st.session_state.customer_mode_v50_snapshot = snap.get("base_v50_snapshot", snap)
    return snap


def _render_v51_status_badges(snap):
    badges = snap.get("status_badges", [])
    if badges:
        st.markdown(" ".join(_v51_badge(str(b)) for b in badges), unsafe_allow_html=True)


def _render_v51_card(title, body, footer=""):
    footer_html = f"<br><br><span class=\"v51-small\">{footer}</span>" if footer else ""
    st.markdown(f"<div class=\"v51-card\"><h4>{title}</h4><p>{body}</p>{footer_html}</div>", unsafe_allow_html=True)

def _build_v50_customer_snapshot():
    dataset_df = st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame()
    snap = core.build_customer_mode_v50_summary(
        project_name=st.session_state.project_name,
        dataset_df=dataset_df,
        auto_pilot_result=st.session_state.auto_pilot_result,
        fusion_doctor=st.session_state.fusion_doctor,
        trust_gate=st.session_state.trust_gate,
        pricing_offer_snapshot=st.session_state.pricing_offer_snapshot,
        proposal_sow_snapshot=st.session_state.proposal_sow_snapshot,
        paid_pilot_snapshot=st.session_state.paid_pilot_v45_snapshot,
        delivery_snapshot=st.session_state.customer_delivery_snapshot,
        lead_intake_snapshot=st.session_state.lead_intake_v48_snapshot,
    )
    st.session_state.customer_mode_v50_snapshot = snap
    return snap



def render_customer_home_v50_tab():
    st.header("Customer Start — Clean Pilot Route V51")
    snap = _build_v51_customer_snapshot()

    _render_v51_status_badges(snap)
    st.markdown(
        "<div class=\"v51-hero\"><h3>From sensor idea to pilot package — without the technical chaos.</h3>"
        "<p>The customer sees a simple route. EdgeTwin still runs the full OMEGA-X engine behind it: dataset generation, feature extraction, reliability checks, trust gates, hardware advice and customer-safe deliverables.</p></div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customer clarity", f"{snap.get('customer_clarity_score', 0)}%")
    m2.metric("Readiness", snap.get("customer_status", "Needs setup"))
    m3.metric("Dataset rows", snap.get("dataset_rows", 0))
    m4.metric("Next step", snap.get("primary_next_step", "Choose problem"))

    verdict = snap.get("customer_verdict", "")
    if snap.get("customer_clarity_score", 0) >= 80:
        st.success(verdict)
    elif snap.get("customer_clarity_score", 0) >= 55:
        st.warning(verdict)
    else:
        st.info(verdict)

    st.subheader("Simple customer journey")
    c1, c2, c3 = st.columns(3)
    with c1:
        _render_v51_card(
            "1. Choose the problem",
            "The customer describes the machine, sensor idea, environment and what should be detected.",
            "No engineering settings needed at the front door."
        )
    with c2:
        _render_v51_card(
            "2. Generate the pilot",
            "EdgeTwin creates the pilot dataset, features, first reliability view and hardware direction.",
            "Advanced checks stay under the hood."
        )
    with c3:
        _render_v51_card(
            "3. Review & hand off",
            "The customer gets a clear readiness status, honest limitations, downloads and proposal route.",
            "Pilot-ready does not mean production-certified."
        )

    st.subheader("Where this project is now")
    for step in snap.get("customer_route", []):
        status = step.get("status", "Pending")
        prefix = "✅" if status == "Done" else "➡️" if status == "Next" else "○"
        st.markdown(
            f"<div class=\"v51-step\"><b>{prefix} {step.get('step', '')}. {step.get('title', '')}</b> "
            f"<span class=\"v51-small\">— {step.get('description', '')}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Customer sees")
        for item in snap.get("customer_visible_outputs", []):
            st.write(f"✓ {item}")
    with right:
        st.markdown("#### Hidden but still active")
        for item in snap.get("hidden_advanced_tools", []):
            st.caption(f"• {item}")

    with st.expander("Advanced details", expanded=False):
        st.write("These are internal/customer-success details. Keep this collapsed during normal buyer demos.")
        st.json({k: v for k, v in snap.items() if k not in ["base_v50_snapshot"]})

    if st.button("Build Customer UI Bundle V51", type="primary", use_container_width=True, key="v51_build_customer_ui_bundle"):
        st.session_state.customer_ui_v51_bundle = core.create_customer_ui_v51_bundle(
            st.session_state.project_name,
            snap,
            st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        st.success("Customer UI bundle created.")

    if st.session_state.customer_ui_v51_bundle:
        st.download_button(
            "Download Customer UI Bundle V51",
            st.session_state.customer_ui_v51_bundle,
            file_name=f"{st.session_state.project_name}_customer_ui_v51.zip",
            mime="application/zip",
            use_container_width=True,
            key="v51_download_customer_ui_bundle",
        )


def render_customer_review_v50_tab():
    st.header("Readiness & Next Step — Customer View V51")
    snap = _build_v51_customer_snapshot()

    _render_v51_status_badges(snap)
    c1, c2, c3 = st.columns(3)
    c1.metric("Readiness", f"{snap.get('customer_clarity_score', 0)}%")
    c2.metric("Status", snap.get("customer_status", "Not ready"))
    c3.metric("Risk level", snap.get("risk_level", "Unknown"))

    st.markdown("#### Recommended next step")
    st.info(snap.get("primary_next_step_long", "Create the pilot package first."))

    ready_col, need_col = st.columns(2)
    with ready_col:
        st.subheader("Ready")
        ready_items = snap.get("ready_items", [])
        if ready_items:
            for item in ready_items:
                st.success(item)
        else:
            st.info("Nothing customer-facing is ready yet. Start with Create pilot.")
    with need_col:
        st.subheader("Still needed")
        missing_items = snap.get("missing_items", [])
        if missing_items:
            for item in missing_items:
                st.warning(item)
        else:
            st.success("No major customer-facing gaps detected for a controlled pilot discussion.")

    st.subheader("Customer-safe summary")
    st.text_area("Summary", snap.get("safe_customer_summary", ""), height=180, key="v51_safe_customer_summary")

    with st.expander("Advanced details", expanded=False):
        st.markdown("#### UI simplification rules")
        for rule in snap.get("ui_simplification_rules", []):
            st.write(f"- {rule}")
        st.markdown("#### Engine still active behind the clean UI")
        for item in snap.get("engine_kept_active", []):
            st.write(f"- {item}")

    st.caption(snap.get("disclaimer", ""))



# ============================================================
# V53 — Launch-Ready Customer Experience
# ============================================================

def render_launch_experience_v53_tab():
    st.header("Launch-Ready Customer Experience V53")
    st.write(
        "Create customer-facing copy, safe status badges, demo script and launch-readiness evidence. "
        "This does not weaken the engine; it translates the full EdgeTwin workflow into a calmer buyer experience."
    )

    dataset_df = st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame()
    c1, c2 = st.columns([1.15, 1])
    with c1:
        target_segment = st.selectbox(
            "Target customer segment",
            [
                "Industrial maintenance team",
                "Security / tamper integrator",
                "Remote asset operator",
                "Forestry / agriculture operator",
                "Edge AI consultant / system integrator",
                "Custom technical team",
            ],
            key="v53_target_segment",
        )
        customer_problem = st.text_area(
            "Customer problem statement",
            value=(st.session_state.last_demo_summary or {}).get(
                "problem",
                "The customer wants to validate an Edge AI sensor pilot but lacks clear data, labels, reliability evidence and hardware direction.",
            ),
            height=110,
            key="v53_customer_problem",
        )
        desired_outcome = st.selectbox(
            "Desired customer outcome",
            ["Pilot package", "Real-data readiness report", "Proposal request", "Paid pilot decision", "Deployment planning"],
            key="v53_desired_outcome",
        )
    with c2:
        proof_level = st.selectbox(
            "Current proof level",
            ["Synthetic pilot", "Real data uploaded", "Field evidence available"],
            key="v53_proof_level",
        )
        cta_mode = st.selectbox(
            "Primary call to action",
            ["Request proposal", "Download pilot bundle", "Book discovery", "Start pilot route"],
            key="v53_cta_mode",
        )
        include_price_hint = st.checkbox("Include price/package positioning", value=True, key="v53_include_price_hint")
        include_privacy_badge = st.checkbox("Include privacy-safe learning badge", value=True, key="v53_include_privacy_badge")
        include_field_validation_notice = st.checkbox("Show field validation notice", value=True, key="v53_include_field_validation_notice")

    if len(dataset_df) == 0:
        st.info("No dataset loaded yet. V53 can still create launch copy, but a generated pilot dataset makes the experience stronger.")
    else:
        st.success(f"Dataset detected: {len(dataset_df)} rows, {len(dataset_df.columns)} columns.")

    if st.button("Build Launch-Ready Experience V53", type="primary", use_container_width=True, key="v53_build_launch_experience"):
        snap = core.build_launch_ready_customer_experience_v53(
            project_name=st.session_state.project_name,
            dataset_df=dataset_df,
            target_segment=target_segment,
            customer_problem=customer_problem,
            desired_outcome=desired_outcome,
            proof_level=proof_level,
            cta_mode=cta_mode,
            include_price_hint=include_price_hint,
            include_privacy_badge=include_privacy_badge,
            include_field_validation_notice=include_field_validation_notice,
            customer_ui_snapshot=st.session_state.customer_ui_v51_snapshot,
            field_learning_snapshot=st.session_state.field_learning_v52_snapshot,
            pricing_offer_snapshot=st.session_state.pricing_offer_snapshot,
            proposal_sow_snapshot=st.session_state.proposal_sow_snapshot,
        )
        st.session_state.launch_experience_v53_snapshot = snap
        st.session_state.launch_experience_v53_bundle = core.create_launch_ready_customer_experience_v53_bundle(
            st.session_state.project_name,
            snap,
            dataset_df,
        )
        st.success("Launch-ready customer experience created.")

    snap = st.session_state.launch_experience_v53_snapshot
    if snap:
        m1, m2, m3 = st.columns(3)
        m1.metric("Launch score", f"{snap.get('launch_score', 0)}%")
        m2.metric("Decision", snap.get("decision", "Unknown"))
        m3.metric("CTA", snap.get("primary_cta", "Unknown"))

        if snap.get("decision") == "GO":
            st.success(snap.get("launch_status", "Launch-ready"))
        elif snap.get("decision") == "CONDITIONAL GO":
            st.warning(snap.get("launch_status", "Conditional launch"))
        else:
            st.error(snap.get("launch_status", "Not ready"))

        st.markdown("#### Customer status badges")
        badges = snap.get("status_badges", [])
        if badges:
            st.markdown(" ".join([f'<span class="v51-badge">{b}</span>' for b in badges]), unsafe_allow_html=True)

        st.markdown("#### Launch copy")
        st.text_area("Landing / demo copy", snap.get("landing_copy", ""), height=240, key="v53_landing_copy")

        st.markdown("#### Simple customer route")
        cols = st.columns(3)
        for i, step in enumerate(snap.get("customer_steps", [])):
            with cols[i % 3]:
                st.markdown(f"### {step.get('step')}. {step.get('title')}")
                st.write(step.get("copy", ""))

        st.markdown("#### Recommended next step")
        st.info(snap.get("recommended_next_step", "Create the pilot package and review readiness."))

        bcol, wcol = st.columns(2)
        with bcol:
            st.subheader("Blockers")
            blockers = snap.get("blockers", [])
            if blockers:
                for item in blockers:
                    st.error(item)
            else:
                st.success("No major launch blockers detected.")
        with wcol:
            st.subheader("Warnings")
            warnings = snap.get("warnings", [])
            if warnings:
                for item in warnings:
                    st.warning(item)
            else:
                st.success("No major launch warnings detected.")

        with st.expander("Demo script", expanded=False):
            for item in snap.get("demo_script", []):
                st.write(f"- {item}")
        with st.expander("Safe claims and claims to avoid", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Safe claims")
                for item in snap.get("safe_claims", []):
                    st.success(item)
            with c2:
                st.markdown("#### Avoid")
                for item in snap.get("claims_to_avoid", []):
                    st.warning(item)
        with st.expander("FAQ", expanded=False):
            for item in snap.get("faq", []):
                st.markdown(f"**{item.get('question')}**")
                st.write(item.get("answer", ""))

        if st.session_state.launch_experience_v53_bundle:
            st.download_button(
                "Download Launch-Ready Customer Experience Bundle V53",
                st.session_state.launch_experience_v53_bundle,
                file_name=f"{st.session_state.project_name}_launch_ready_customer_experience_v53.zip",
                mime="application/zip",
                use_container_width=True,
                key="v53_download_launch_experience_bundle",
            )

    st.caption("V53 is a customer-experience and launch-copy layer. It keeps the full engine behind the scenes and does not certify production performance.")


# ============================================================
# V52 — Privacy-Safe Field Learning System
# ============================================================

def render_field_learning_v52_tab():
    st.header("Privacy-Safe Field Learning V52")
    st.write(
        "Use real customer field data without making privacy or trust mistakes. "
        "By default, customer data stays private. EdgeTwin can learn only from safe extracted features when explicit opt-in is selected."
    )

    dataset_df = st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame()

    c1, c2 = st.columns([1.2, 1])
    with c1:
        consent_mode = st.selectbox(
            "Data permission mode",
            ["Private only", "Feature learning allowed", "Raw data permission"],
            index=0,
            key="v52_consent_mode",
            help="Private only is the default. Feature learning is the recommended opt-in. Raw data permission should be rare and contractual."
        )
        customer_sector = st.selectbox(
            "Sector / environment",
            ["Industrial maintenance", "Building security", "Forestry / outdoor", "Agriculture", "Logistics", "Custom"],
            key="v52_customer_sector",
        )
        use_case_type = st.text_input(
            "Use-case label",
            value=str(st.session_state.auto_pilot_config.get("use_case_type", "Custom pilot")) if st.session_state.auto_pilot_config else "Custom pilot",
            key="v52_use_case_type",
        )
        retention_days = st.number_input("Retention days", min_value=7, max_value=3650, value=180, step=30, key="v52_retention_days")

    with c2:
        contains_audio = st.checkbox("Dataset may include raw audio or audio filenames", value=False, key="v52_contains_audio")
        contains_gps = st.checkbox("Dataset may include GPS/location data", value=False, key="v52_contains_gps")
        contains_machine_ids = st.checkbox("Dataset may include machine/customer IDs", value=False, key="v52_contains_machine_ids")
        contains_personal_data = st.checkbox("Dataset may include personal/employee/customer data", value=False, key="v52_contains_personal_data")
        allow_cross_customer_learning = st.checkbox(
            "Allow cross-customer aggregate feature learning",
            value=(consent_mode == "Feature learning allowed"),
            key="v52_allow_cross_customer_learning",
        )

    st.caption(
        "Recommended default: Private only. Best scalable option: Feature learning allowed, because EdgeTwin learns from aggregate feature patterns without copying raw customer files into the global engine."
    )

    if len(dataset_df) == 0:
        st.info("No dataset loaded yet. Generate a pilot dataset or upload real customer data first. V52 can still create the privacy policy/checklist.")
    else:
        st.success(f"Dataset detected: {len(dataset_df)} rows, {len(dataset_df.columns)} columns.")

    if st.button("Build Privacy-Safe Learning Plan V52", type="primary", use_container_width=True, key="v52_build_privacy_learning"):
        snap = core.build_privacy_safe_field_learning_v52(
            project_name=st.session_state.project_name,
            dataset_df=dataset_df,
            consent_mode=consent_mode,
            customer_sector=customer_sector,
            use_case_type=use_case_type,
            retention_days=int(retention_days),
            contains_audio=contains_audio,
            contains_gps=contains_gps,
            contains_machine_ids=contains_machine_ids,
            contains_personal_data=contains_personal_data,
            allow_cross_customer_learning=allow_cross_customer_learning,
        )
        st.session_state.field_learning_v52_snapshot = snap
        st.session_state.field_learning_v52_bundle = core.create_privacy_safe_field_learning_v52_bundle(
            st.session_state.project_name,
            snap,
            dataset_df,
        )
        st.success("Privacy-safe learning plan created.")

    snap = st.session_state.field_learning_v52_snapshot
    if snap:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Privacy score", f"{snap.get('privacy_score', 0)}%")
        m2.metric("Learning mode", snap.get("learning_mode", "Unknown"))
        m3.metric("Risk", snap.get("risk_level", "Unknown"))
        m4.metric("Rows usable", snap.get("feature_rows_available", 0))

        if snap.get("risk_level") == "High":
            st.error(snap.get("verdict", "High privacy risk. Keep data private until consent and minimization are solved."))
        elif snap.get("risk_level") == "Medium":
            st.warning(snap.get("verdict", "Use feature-only learning and keep raw data private."))
        else:
            st.success(snap.get("verdict", "Privacy posture looks acceptable for controlled feature learning."))

        st.subheader("Allowed vs blocked")
        allowed_col, blocked_col = st.columns(2)
        with allowed_col:
            st.markdown("#### Allowed actions")
            for item in snap.get("allowed_actions", []):
                st.success(item)
        with blocked_col:
            st.markdown("#### Blocked by default")
            for item in snap.get("blocked_actions", []):
                st.warning(item)

        st.subheader("Safe learning output")
        st.write(snap.get("learning_summary", ""))
        features = snap.get("safe_feature_candidates", [])
        if features:
            st.dataframe(pd.DataFrame(features), use_container_width=True)
        else:
            st.info("No safe numeric feature columns found yet. Generate/extract features first.")

        with st.expander("Consent text for Proposal / SOW", expanded=False):
            st.text_area("Customer consent clause", snap.get("customer_consent_clause", ""), height=180, key="v52_consent_clause_text")

        with st.expander("Advanced privacy details", expanded=False):
            st.markdown("#### Data minimization plan")
            for item in snap.get("data_minimization_plan", []):
                st.write(f"- {item}")
            st.markdown("#### Retention and deletion")
            for item in snap.get("retention_and_deletion_plan", []):
                st.write(f"- {item}")
            st.markdown("#### Audit log events")
            for item in snap.get("audit_log_events", []):
                st.write(f"- {item}")
            st.json({k: v for k, v in snap.items() if k not in ["safe_feature_candidates", "feature_library_preview"]})

        if st.session_state.field_learning_v52_bundle:
            st.download_button(
                "Download Privacy-Safe Field Learning Bundle V52",
                st.session_state.field_learning_v52_bundle,
                file_name=f"{st.session_state.project_name}_privacy_safe_field_learning_v52.zip",
                mime="application/zip",
                use_container_width=True,
                key="v52_download_privacy_learning_bundle",
            )

    st.caption(
        "V52 is a workflow and data-minimization layer. It is not legal advice and does not automatically make customer data anonymous. Real anonymisation requires careful removal of identifiers and re-identification risk checks."
    )



# ============================================================
# V54 — Public Launch Page & Sales Assets
# ============================================================

def render_launch_assets_v54_tab():
    st.header("Public Launch Page & Sales Assets V54")
    st.write(
        "Create simple customer-facing website copy, offer cards, outreach email, LinkedIn message and launch checklist. "
        "This helps you explain the product without manually rewriting the story for every lead."
    )

    dataset_df = st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame()

    c1, c2 = st.columns([1.12, 1])
    with c1:
        market_focus = st.selectbox(
            "Market focus",
            [
                "Predictive maintenance / machine health",
                "Acoustic tamper / construction security",
                "Remote asset monitoring",
                "Forestry / agriculture monitoring",
                "Edge AI consultant toolkit",
                "Custom sensor fusion pilot",
            ],
            key="v54_market_focus",
        )
        buyer_persona = st.selectbox(
            "Buyer persona",
            [
                "Maintenance manager / technical lead",
                "Security integrator / operations manager",
                "IoT product manager",
                "Founder / innovation lead",
                "System integrator / consultant",
                "Engineering team",
            ],
            key="v54_buyer_persona",
        )
        primary_offer = st.selectbox(
            "Primary offer",
            ["Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Pilot", "Enterprise / Custom Pilot"],
            index=1,
            key="v54_primary_offer",
        )
        price_anchor = st.text_input("Price anchor / range", value="€750 – €2,500 per pilot", key="v54_price_anchor")
    with c2:
        primary_cta = st.selectbox(
            "Primary CTA",
            ["Request a pilot proposal", "Book a discovery call", "Generate a pilot package", "Upload real sensor data"],
            key="v54_primary_cta",
        )
        launch_channel = st.selectbox(
            "Launch channel",
            ["Direct outreach / LinkedIn", "Landing page", "Private beta invite", "Partner/system integrator intro", "Customer demo call"],
            key="v54_launch_channel",
        )
        proof_level = st.selectbox(
            "Proof level",
            ["Pilot package available", "Real customer data uploaded", "Real field evidence available"],
            key="v54_proof_level",
        )
        include_pricing = st.checkbox("Include price positioning", value=True, key="v54_include_pricing")
        include_calendar_cta = st.checkbox("Include calendar/discovery CTA", value=True, key="v54_include_calendar_cta")

    if len(dataset_df) > 0:
        st.success(f"Dataset detected: {len(dataset_df)} rows, {len(dataset_df.columns)} columns.")
    else:
        st.info("No dataset loaded yet. V54 can still create launch assets, but a generated pilot dataset makes the offer stronger.")

    if st.button("Build Public Launch Assets V54", type="primary", use_container_width=True, key="v54_build_launch_assets"):
        snap = core.build_public_launch_assets_v54(
            project_name=st.session_state.project_name,
            dataset_df=dataset_df,
            market_focus=market_focus,
            buyer_persona=buyer_persona,
            primary_offer=primary_offer,
            price_anchor=price_anchor,
            primary_cta=primary_cta,
            launch_channel=launch_channel,
            proof_level=proof_level,
            include_pricing=include_pricing,
            include_calendar_cta=include_calendar_cta,
            launch_experience_snapshot=st.session_state.launch_experience_v53_snapshot,
            pricing_offer_snapshot=st.session_state.pricing_offer_snapshot,
            proposal_sow_snapshot=st.session_state.proposal_sow_snapshot,
            field_learning_snapshot=st.session_state.field_learning_v52_snapshot,
            product_readiness_snapshot=st.session_state.product_readiness_v40_snapshot,
        )
        st.session_state.launch_assets_v54_snapshot = snap
        st.session_state.launch_assets_v54_bundle = core.create_public_launch_assets_v54_bundle(
            st.session_state.project_name,
            snap,
            dataset_df,
        )
        st.success("Public launch assets created.")

    snap = st.session_state.launch_assets_v54_snapshot
    if snap:
        m1, m2, m3 = st.columns(3)
        m1.metric("Launch assets score", f"{snap.get('launch_asset_score', 0)}%")
        m2.metric("Decision", snap.get("decision", "Unknown"))
        m3.metric("Primary offer", snap.get("primary_offer", "Unknown"))

        if snap.get("decision") == "GO":
            st.success(snap.get("launch_status", "Ready for outreach"))
        elif snap.get("decision") == "CONDITIONAL GO":
            st.warning(snap.get("launch_status", "Use careful outreach"))
        else:
            st.error(snap.get("launch_status", "Not ready"))

        st.subheader("Website / landing copy")
        st.markdown(f"### {snap.get('headline', '')}")
        st.write(snap.get("subheadline", ""))
        st.info(snap.get("positioning", ""))

        st.subheader("Problem → EdgeTwin response")
        ps = pd.DataFrame(snap.get("problem_solution_rows", []))
        if len(ps) > 0:
            st.dataframe(ps, use_container_width=True)

        st.subheader("Offer cards")
        offer_df = pd.DataFrame(snap.get("offer_cards", []))
        if len(offer_df) > 0:
            st.dataframe(offer_df, use_container_width=True)

        with st.expander("Outreach email", expanded=False):
            st.text_area("Email draft", snap.get("outreach_email", ""), height=260, key="v54_outreach_email")
        with st.expander("LinkedIn message", expanded=False):
            st.text_area("LinkedIn draft", snap.get("linkedin_message", ""), height=130, key="v54_linkedin_message")
        with st.expander("Demo agenda and launch checklist", expanded=False):
            st.markdown("#### Demo agenda")
            for item in snap.get("demo_agenda", []):
                st.write(f"- {item}")
            st.markdown("#### Launch checklist")
            st.dataframe(pd.DataFrame(snap.get("launch_checklist", [])), use_container_width=True)

        c_safe, c_avoid = st.columns(2)
        with c_safe:
            st.markdown("#### Safe claims")
            for item in snap.get("safe_claims", []):
                st.success(item)
        with c_avoid:
            st.markdown("#### Claims to avoid")
            for item in snap.get("claims_to_avoid", []):
                st.warning(item)

        if snap.get("blockers"):
            st.subheader("Blockers")
            for item in snap.get("blockers", []):
                st.error(item)
        if snap.get("warnings"):
            st.subheader("Warnings")
            for item in snap.get("warnings", []):
                st.warning(item)

        if st.session_state.launch_assets_v54_bundle:
            st.download_button(
                "Download Public Launch Assets Bundle V54",
                st.session_state.launch_assets_v54_bundle,
                file_name=f"{st.session_state.project_name}_public_launch_assets_v54.zip",
                mime="application/zip",
                use_container_width=True,
                key="v54_download_public_launch_assets_bundle",
            )

    st.caption("V54 creates launch and sales assets only. Keep all claims pilot-focused and require field validation before production deployment.")


# ============================================================
# V55 — First Customer Beta Script / Test Run Planner
# ============================================================

def render_first_customer_beta_v55_tab():
    st.header("First Customer Beta Script V55")
    st.write(
        "Build a structured first-customer beta/demo plan so every conversation follows the same safe route: "
        "problem, pilot package, readiness, boundaries, next step and feedback."
    )

    dataset_df = st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame()

    c1, c2 = st.columns([1.1, 1])
    with c1:
        target_segment = st.selectbox(
            "Target customer segment",
            [
                "Predictive maintenance / machine health",
                "Security / tamper integrator",
                "Remote asset / forestry pilot",
                "Edge AI consultant / integrator",
                "Custom industrial sensor pilot",
            ],
            key="v55_target_segment",
        )
        beta_goal = st.selectbox(
            "Main beta goal",
            [
                "Validate customer understanding",
                "Validate paid pilot interest",
                "Validate real-data upload flow",
                "Validate Edge Impulse export usefulness",
                "Validate report and proposal clarity",
            ],
            key="v55_beta_goal",
        )
        customer_data_status = st.selectbox(
            "Customer data status",
            ["No data yet", "Synthetic pilot only", "Has example CSV/WAV", "Has real field dataset", "Unknown"],
            key="v55_customer_data_status",
        )
        demo_duration = st.slider("Demo / beta call duration", 20, 90, 45, 5, key="v55_demo_duration")
    with c2:
        technical_maturity = st.selectbox(
            "Customer technical maturity",
            ["Non-technical decision maker", "Mixed business/engineering", "Engineering team", "AI/ML team"],
            index=1,
            key="v55_technical_maturity",
        )
        commercial_stage = st.selectbox(
            "Commercial stage",
            ["First conversation", "Demo booked", "Pricing discussion", "Proposal/SOW stage", "Paid pilot candidate"],
            key="v55_commercial_stage",
        )
        include_follow_up = st.checkbox("Create follow-up email", value=True, key="v55_include_follow_up")
        strict_scope_control = st.checkbox("Strict scope control", value=True, key="v55_strict_scope_control")

    if len(dataset_df) > 0:
        st.success(f"Dataset detected: {len(dataset_df)} rows. The beta script can reference current project evidence.")
    else:
        st.info("No dataset loaded. The beta script will focus on discovery and demo flow rather than evidence delivery.")

    if st.button("Build First Customer Beta Script V55", type="primary", use_container_width=True, key="v55_build_beta_script"):
        snap = core.build_first_customer_beta_script_v55(
            project_name=st.session_state.project_name,
            dataset_df=dataset_df,
            target_segment=target_segment,
            beta_goal=beta_goal,
            customer_data_status=customer_data_status,
            demo_duration=demo_duration,
            technical_maturity=technical_maturity,
            commercial_stage=commercial_stage,
            include_follow_up=include_follow_up,
            strict_scope_control=strict_scope_control,
            launch_assets_snapshot=st.session_state.launch_assets_v54_snapshot,
            launch_experience_snapshot=st.session_state.launch_experience_v53_snapshot,
            field_learning_snapshot=st.session_state.field_learning_v52_snapshot,
            pricing_offer_snapshot=st.session_state.pricing_offer_snapshot,
            proposal_sow_snapshot=st.session_state.proposal_sow_snapshot,
            product_readiness_snapshot=st.session_state.product_readiness_v40_snapshot,
        )
        st.session_state.first_customer_beta_v55_snapshot = snap
        st.session_state.first_customer_beta_v55_bundle = core.create_first_customer_beta_script_v55_bundle(
            st.session_state.project_name,
            snap,
            dataset_df,
        )
        st.success("First customer beta script created.")

    snap = st.session_state.first_customer_beta_v55_snapshot
    if snap:
        m1, m2, m3 = st.columns(3)
        m1.metric("Beta script score", f"{snap.get('beta_script_score', 0)}%")
        m2.metric("Decision", snap.get("decision", "Unknown"))
        m3.metric("Beta goal", snap.get("beta_goal", "Unknown"))

        if snap.get("decision") == "GO":
            st.success(snap.get("beta_status", "Ready for first beta call"))
        elif snap.get("decision") == "CONDITIONAL GO":
            st.warning(snap.get("beta_status", "Use with careful positioning"))
        else:
            st.error(snap.get("beta_status", "Not ready"))

        st.subheader("Call agenda")
        for row in snap.get("call_agenda", []):
            st.write(f"**{row.get('minute')} min — {row.get('section')}**: {row.get('goal')}")

        st.subheader("Live demo script")
        for step in snap.get("demo_steps", []):
            st.write(f"**{step.get('step')}. {step.get('title')}**")
            st.caption(step.get("operator_note", ""))
            st.write(step.get("customer_message", ""))

        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("#### Feedback questions")
            for item in snap.get("feedback_questions", []):
                st.info(item)
        with c_b:
            st.markdown("#### Success metrics")
            for item in snap.get("success_metrics", []):
                st.success(item)

        with st.expander("Founder checklist", expanded=False):
            st.dataframe(pd.DataFrame(snap.get("founder_checklist", [])), use_container_width=True)
        with st.expander("Customer required inputs", expanded=False):
            st.dataframe(pd.DataFrame(snap.get("customer_required_inputs", [])), use_container_width=True)
        with st.expander("Follow-up email", expanded=False):
            st.text_area("Email draft", snap.get("follow_up_email", ""), height=240, key="v55_follow_up_email")

        c_safe, c_avoid = st.columns(2)
        with c_safe:
            st.markdown("#### Safe claims")
            for item in snap.get("safe_claims", []):
                st.success(item)
        with c_avoid:
            st.markdown("#### Claims to avoid")
            for item in snap.get("claims_to_avoid", []):
                st.warning(item)

        if snap.get("blockers"):
            st.subheader("Blockers")
            for item in snap.get("blockers", []):
                st.error(item)
        if snap.get("warnings"):
            st.subheader("Warnings")
            for item in snap.get("warnings", []):
                st.warning(item)

        st.info(snap.get("recommended_next_step", ""))

        if st.session_state.first_customer_beta_v55_bundle:
            st.download_button(
                "Download First Customer Beta Script Bundle V55",
                st.session_state.first_customer_beta_v55_bundle,
                file_name=f"{st.session_state.project_name}_first_customer_beta_script_v55.zip",
                mime="application/zip",
                use_container_width=True,
                key="v55_download_first_customer_beta_script_bundle",
            )

    st.caption("V55 is a demo/beta execution script. Keep the conversation pilot-focused and use customer feedback to improve the product, not to promise production accuracy.")


# ============================================================
# V56 — Real Upload Experience 2.0 / Customer Data Intake
# ============================================================

def render_real_upload_v56_tab():
    st.header("Real Upload Experience V56")
    st.write(
        "Upload real WAV/CSV field files in a controlled, privacy-first way. "
        "EdgeTwin inspects the files, extracts safe feature-level evidence and tells you what is usable, missing or risky before downstream trust/report flows."
    )

    c1, c2 = st.columns([1.2, 1])
    with c1:
        use_case = st.selectbox(
            "Upload use-case context",
            ["Predictive Maintenance", "Security / Tamper", "Smart Forestry / Remote Area", "Remote Asset Monitoring", "Custom Sensor Fusion"],
            key="v56_upload_use_case",
        )
        expected_labels_text = st.text_area(
            "Expected labels/classes (optional)",
            value="Normal, Warning, Event, Critical",
            height=80,
            key="v56_expected_labels_text",
            help="Used only to check whether uploaded files cover the customer classes."
        )
        shared_label = st.text_input(
            "Default label for uploaded files if unknown",
            value="Unlabeled",
            key="v56_default_upload_label",
        )
    with c2:
        learning_mode = st.radio(
            "Data learning mode",
            ["Private only", "Feature learning allowed", "Raw data permission"],
            index=0,
            key="v56_learning_mode",
            help="Private only is the safest default. Feature learning uses derived features only. Raw data permission should be rare and require written consent."
        )
        st.caption("Raw uploaded files are not added to any global learning bundle by default.")

    uploaded_files = st.file_uploader(
        "Upload WAV/CSV field files",
        type=["wav", "csv"],
        accept_multiple_files=True,
        key="v56_real_upload_files",
    )

    upload_records = []
    feature_rows = []
    if uploaded_files:
        st.subheader("File inspection preview")
        preview_rows = []
        for up in uploaded_files:
            content = up.getvalue()
            file_type = up.name.split(".")[-1].lower() if "." in up.name else "unknown"
            features = core.extract_features_from_bytes(content, up.name, st.session_state.sr)
            label = shared_label.strip().replace(" ", "_") or "Unlabeled"
            rec = {
                "filename": up.name,
                "file_type": file_type,
                "size_bytes": len(content),
                "label": label,
                "features": features,
            }
            if isinstance(features, dict) and "error" in features:
                rec["error"] = features.get("error")
            else:
                safe_row = {"Label": label, "Filename": up.name, "FileType": file_type}
                for k, v in (features or {}).items():
                    if k != "error":
                        safe_row[k] = v
                feature_rows.append(safe_row)
            upload_records.append(rec)
            preview_rows.append({
                "filename": up.name,
                "type": file_type,
                "size_mb": round(len(content) / (1024 * 1024), 3),
                "label": label,
                "status": "ERROR" if rec.get("error") else "OK",
                "message": rec.get("error", "Feature inspection completed."),
            })
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

    expected_labels = [x.strip().replace(" ", "_") for x in expected_labels_text.replace(";", ",").split(",") if x.strip()]

    if st.button("Build Real Upload Intake Plan V56", type="primary", use_container_width=True, key="v56_build_upload_intake"):
        feature_df = pd.DataFrame(feature_rows)
        snapshot = core.inspect_real_upload_records_v56(
            upload_records=upload_records,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            use_case=use_case,
            learning_mode=learning_mode,
            expected_labels=expected_labels,
        )
        bundle = core.create_real_upload_experience_v56_bundle(
            st.session_state.project_name,
            snapshot,
            feature_df=feature_df,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        st.session_state.real_upload_v56_snapshot = snapshot
        st.session_state.real_upload_v56_bundle = bundle
        st.session_state.real_upload_v56_features_df = feature_df
        st.success("Real upload intake plan generated.")

    snap = st.session_state.real_upload_v56_snapshot
    if snap:
        st.subheader("Upload readiness result")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Upload score", f"{snap.get('upload_score', 0)}%")
        m2.metric("Decision", snap.get("decision", "Unknown"))
        m3.metric("Files", snap.get("uploaded_file_count", 0))
        m4.metric("Feature rows", snap.get("feature_rows_count", 0))
        if snap.get("decision") == "GO":
            st.success(snap.get("status_text", ""))
        elif snap.get("decision") == "CONDITIONAL GO":
            st.warning(snap.get("status_text", ""))
        else:
            st.error(snap.get("status_text", ""))

        if snap.get("privacy_flags"):
            st.subheader("Privacy flags")
            for item in snap.get("privacy_flags", []):
                st.warning(item)

        if snap.get("blockers"):
            st.subheader("Blockers")
            for item in snap.get("blockers", []):
                st.error(item)
        if snap.get("warnings"):
            st.subheader("Warnings")
            for item in snap.get("warnings", []):
                st.warning(item)

        st.subheader("Next actions")
        for item in snap.get("next_actions", []):
            st.info(item)

        with st.expander("Inspected files and privacy policy", expanded=False):
            st.dataframe(pd.DataFrame(snap.get("inspected_files", [])), use_container_width=True)
            st.json(snap.get("privacy_policy", {}))

        feature_df = st.session_state.real_upload_v56_features_df
        if isinstance(feature_df, pd.DataFrame) and len(feature_df) > 0:
            st.subheader("Safe extracted feature rows")
            st.dataframe(feature_df.head(50), use_container_width=True)
            if st.button("Load extracted feature rows into Enterprise Audit", use_container_width=True, key="v56_load_features_to_audit"):
                st.session_state.dataset = feature_df.copy()
                st.success("Safe feature rows loaded into Enterprise Audit dataset.")

        if st.session_state.real_upload_v56_bundle:
            st.download_button(
                "Download Real Upload Experience Bundle V56",
                st.session_state.real_upload_v56_bundle,
                file_name=f"{st.session_state.project_name}_real_upload_experience_v56.zip",
                mime="application/zip",
                use_container_width=True,
                key="v56_download_real_upload_bundle",
            )

    st.caption("V56 keeps raw customer uploads private by default and converts usable evidence into safer feature-level rows for downstream readiness and trust checks.")



# ============================================================
# V57 — Checkout & Paid Download Readiness / Payment Prep
# ============================================================

def render_checkout_v57_tab():
    st.header("Checkout & Paid Download Readiness V57")
    st.write(
        "Prepare a controlled checkout/invoice step for paid pilot bundles. "
        "V57 does not process payments directly; it checks whether scope, license, privacy and delivery conditions are ready before payment or paid download unlock."
    )

    c1, c2 = st.columns([1.2, 1])
    with c1:
        customer_name = st.text_input("Customer / organization name", value="Customer", key="v57_customer_name")
        customer_email = st.text_input("Customer email", value="", key="v57_customer_email")
        package_name = st.selectbox("Package", core.get_checkout_packages_v57(), index=1, key="v57_package_name")
        payment_provider = st.selectbox(
            "Payment provider mode",
            ["Manual invoice first", "Stripe/live checkout", "Payment link placeholder", "Enterprise procurement"],
            index=0,
            key="v57_payment_provider",
        )
        payment_method = st.selectbox(
            "Payment method",
            ["Manual invoice / bank transfer", "Stripe card checkout", "Payment link", "PO / enterprise invoice"],
            index=0,
            key="v57_payment_method",
        )
        tax_mode = st.selectbox(
            "Tax/VAT handling",
            ["EU B2B reverse charge / manual review", "Domestic VAT manual review", "Outside EU manual review", "Tax not configured yet"],
            index=0,
            key="v57_tax_mode",
        )
    with c2:
        invoice_details_complete = st.checkbox("Invoice/customer details complete", value=False, key="v57_invoice_details_complete")
        sow_scope_accepted = st.checkbox("SOW/scope accepted", value=bool(st.session_state.get("proposal_sow_snapshot")), key="v57_sow_scope_accepted")
        license_certificate_ready = st.checkbox("License/certificate ready", value=bool(st.session_state.get("license_cert_snapshot")), key="v57_license_ready")
        delivery_bundle_ready = st.checkbox("Delivery bundle ready", value=bool(st.session_state.get("delivery_snapshot")), key="v57_delivery_ready")
        privacy_notice_ready = st.checkbox("Privacy/customer data notice ready", value=True, key="v57_privacy_notice_ready")
        refund_policy_acknowledged = st.checkbox("Refund/cancellation policy acknowledged", value=False, key="v57_refund_ack")
        st.info("For first paid pilots, manual invoice/payment link is safer than fully automated checkout. Stripe can come later after scope and terms are stable.")

    if st.button("Build Checkout Readiness V57", type="primary", use_container_width=True, key="v57_build_checkout"):
        snapshot = core.build_checkout_readiness_v57(
            project_name=st.session_state.project_name,
            customer_name=customer_name,
            customer_email=customer_email,
            package_name=package_name,
            payment_method=payment_method,
            invoice_details_complete=invoice_details_complete,
            sow_scope_accepted=sow_scope_accepted,
            license_certificate_ready=license_certificate_ready,
            delivery_bundle_ready=delivery_bundle_ready,
            privacy_notice_ready=privacy_notice_ready,
            refund_policy_acknowledged=refund_policy_acknowledged,
            tax_mode=tax_mode,
            payment_provider=payment_provider,
            selected_plan=st.session_state.selected_plan,
            pricing_offer_snapshot=st.session_state.get("pricing_offer_snapshot"),
            paid_pilot_snapshot=st.session_state.get("paid_pilot_v45_snapshot"),
            delivery_snapshot=st.session_state.get("delivery_snapshot"),
        )
        st.session_state.checkout_v57_snapshot = snapshot
        st.session_state.checkout_v57_bundle = core.create_checkout_readiness_v57_bundle(
            st.session_state.project_name,
            snapshot,
            st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        decision = snapshot.get("decision")
        if decision == "GO":
            st.success("Checkout/invoice step is ready.")
        elif decision == "CONDITIONAL GO":
            st.warning("Checkout is close, but clean up the warnings first.")
        else:
            st.error("Do not send checkout/invoice yet. Fix blockers first.")

    if st.session_state.checkout_v57_snapshot:
        snap = st.session_state.checkout_v57_snapshot
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Checkout score", f"{snap.get('checkout_score', 0)}%")
        m2.metric("Decision", snap.get("decision", "Unknown"))
        m3.metric("Package", snap.get("package_name", "Unknown"))
        m4.metric("Price", f"{snap.get('currency', 'EUR')} {snap.get('price', 0):,.0f}")

        st.subheader("Customer checkout copy")
        st.write(snap.get("customer_checkout_copy", ""))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Blockers")
            for item in snap.get("blockers", []) or ["No hard blockers recorded."]:
                st.error(item) if snap.get("blockers") else st.success(item)
            st.markdown("#### Internal next actions")
            for item in snap.get("internal_next_actions", []):
                st.write(f"- {item}")
        with c2:
            st.markdown("#### Warnings")
            for item in snap.get("warnings", []) or ["No warnings recorded."]:
                st.warning(item) if snap.get("warnings") else st.success(item)
            st.markdown("#### Deliverables unlocked by scope")
            for item in snap.get("deliverables", []):
                st.write(f"- {item}")

        st.subheader("Invoice line items")
        st.dataframe(pd.DataFrame(snap.get("line_items", [])), use_container_width=True)

        with st.expander("Safe claims / claims to avoid", expanded=False):
            st.markdown("**Safe claims**")
            for item in snap.get("safe_claims", []):
                st.write(f"- {item}")
            st.markdown("**Avoid**")
            for item in snap.get("claims_to_avoid", []):
                st.write(f"- {item}")

        if st.session_state.checkout_v57_bundle:
            st.download_button(
                "Download Checkout Readiness Bundle V57",
                st.session_state.checkout_v57_bundle,
                file_name=f"{st.session_state.project_name}_checkout_readiness_v57.zip",
                mime="application/zip",
                use_container_width=True,
                key="v57_download_checkout_bundle",
            )

    st.caption("V57 prepares payment/download gating. It does not replace tax/legal review and does not process payments inside this local app.")



# ============================================================
# V58 — Cloud Production Architecture & Migration Planner
# ============================================================

def render_cloud_architecture_v58_tab():
    st.header("Cloud Production Architecture & Migration Planner V58")
    st.write(
        "Plan the safe move from local/private beta to paid pilots, public SaaS or enterprise/on-prem. "
        "This does not force heavy cloud infrastructure too early; it tells you when SQLite/local storage is enough and when PostgreSQL/object storage/FastAPI become necessary."
    )

    dataset_rows = len(st.session_state.dataset) if isinstance(st.session_state.dataset, pd.DataFrame) else 0

    c1, c2 = st.columns([1.2, 1])
    with c1:
        target_stage = st.selectbox("Target stage", core.get_cloud_stage_options_v58(), index=1, key="v58_target_stage")
        expected_monthly_users = st.number_input("Expected monthly users", 1, 10000, 10, 5, key="v58_expected_users")
        expected_monthly_projects = st.number_input("Expected monthly projects", 1, 50000, 25, 10, key="v58_expected_projects")
        expected_max_dataset_rows = st.number_input("Expected max dataset rows per project", 100, 1000000, max(25000, int(dataset_rows or 0)), 1000, key="v58_expected_rows")
        data_sensitivity = st.selectbox(
            "Data sensitivity",
            [
                "Customer sensor data, no personal audio expected",
                "Audio uploads may contain personal/environmental content",
                "Industrial proprietary machine data",
                "Location/GPS or site-security sensitive data",
                "Enterprise confidential / regulated review needed",
            ],
            index=0,
            key="v58_data_sensitivity",
        )
    with c2:
        current_frontend = st.selectbox("Current frontend", ["Streamlit app", "Streamlit Cloud app", "Custom web frontend"], index=0, key="v58_frontend")
        current_backend = st.selectbox("Current backend", ["Streamlit monolith / core.py", "FastAPI private beta", "FastAPI + workers", "Enterprise/on-prem services"], index=0, key="v58_backend")
        current_database = st.selectbox("Current database", ["SQLite metadata", "PostgreSQL/Supabase/Neon", "Enterprise DB", "Not configured"], index=0, key="v58_database")
        current_storage = st.selectbox("Current storage", ["Local file storage", "S3/R2/MinIO object storage", "Enterprise object storage", "Not configured"], index=0, key="v58_storage")
        payment_mode = st.selectbox("Payment mode", ["Manual invoice first", "Stripe checkout planned", "Stripe/live checkout", "Enterprise procurement"], index=0, key="v58_payment")
        wants_public_signup = st.checkbox("Public signup planned", value=False, key="v58_public_signup")
        needs_real_uploads = st.checkbox("Real customer uploads needed", value=True, key="v58_real_uploads")
        needs_api = st.checkbox("Public/partner API needed", value=False, key="v58_needs_api")
        needs_stripe = st.checkbox("Stripe / paid checkout needed soon", value=False, key="v58_needs_stripe")

    if st.button("Build Cloud Production Plan V58", type="primary", use_container_width=True, key="v58_build_architecture"):
        snapshot = core.build_cloud_production_architecture_v58(
            project_name=st.session_state.project_name,
            target_stage=target_stage,
            expected_monthly_users=expected_monthly_users,
            expected_monthly_projects=expected_monthly_projects,
            expected_max_dataset_rows=expected_max_dataset_rows,
            data_sensitivity=data_sensitivity,
            current_frontend=current_frontend,
            current_backend=current_backend,
            current_database=current_database,
            current_storage=current_storage,
            payment_mode=payment_mode,
            wants_public_signup=wants_public_signup,
            needs_real_uploads=needs_real_uploads,
            needs_api=needs_api,
            needs_stripe=needs_stripe,
            selected_plan=st.session_state.selected_plan,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        st.session_state.cloud_architecture_v58_snapshot = snapshot
        st.session_state.cloud_architecture_v58_bundle = core.create_cloud_production_architecture_v58_bundle(
            st.session_state.project_name,
            snapshot,
            st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        if snapshot.get("decision") == "GO":
            st.success("Architecture is acceptable for the selected stage.")
        elif snapshot.get("decision") == "CONDITIONAL GO":
            st.warning("Architecture is usable with controlled limits and a migration plan.")
        else:
            st.error("Architecture is not ready for the selected target stage.")

    if st.session_state.cloud_architecture_v58_snapshot:
        snap = st.session_state.cloud_architecture_v58_snapshot
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Architecture score", f"{snap.get('architecture_score', 0)}%")
        m2.metric("Decision", snap.get("decision", "Unknown"))
        m3.metric("Target", snap.get("target_stage", "Unknown"))
        m4.metric("Rows/project", f"{snap.get('expected_max_dataset_rows', 0):,}")

        st.subheader("Readiness")
        st.write(snap.get("readiness", ""))
        st.info(snap.get("profile_priority", ""))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Blockers")
            for item in snap.get("blockers", []) or ["No hard blockers recorded for selected stage."]:
                st.error(item) if snap.get("blockers") else st.success(item)
            st.markdown("#### Immediate next actions")
            for item in snap.get("immediate_next_actions", []):
                st.write(f"- {item}")
        with c2:
            st.markdown("#### Warnings")
            for item in snap.get("warnings", []) or ["No warnings recorded."]:
                st.warning(item) if snap.get("warnings") else st.success(item)

        st.subheader("Recommended architecture layers")
        st.dataframe(pd.DataFrame(snap.get("architecture_layers", [])), use_container_width=True)

        st.subheader("Migration phases")
        st.dataframe(pd.DataFrame(snap.get("migration_phases", [])), use_container_width=True)

        with st.expander("Safe claims / claims to avoid", expanded=False):
            st.markdown("**Safe claims**")
            for item in snap.get("safe_claims", []):
                st.write(f"- {item}")
            st.markdown("**Avoid**")
            for item in snap.get("claims_to_avoid", []):
                st.write(f"- {item}")

        if st.session_state.cloud_architecture_v58_bundle:
            st.download_button(
                "Download Cloud Architecture Bundle V58",
                st.session_state.cloud_architecture_v58_bundle,
                file_name=f"{st.session_state.project_name}_cloud_architecture_v58.zip",
                mime="application/zip",
                use_container_width=True,
                key="v58_download_cloud_architecture_bundle",
            )

    st.caption("V58 is an architecture and migration planner. It does not replace security/legal review, but it prevents moving to public SaaS too early with local-only infrastructure.")


# ============================================================
# V59 — Hardware Reference Demo / Edge Node Proof Kit
# ============================================================

def render_hardware_reference_v59_tab():
    st.header("Hardware Reference Demo / Edge Node Proof Kit V59")
    st.write(
        "Use this when you want a minimal, controlled hardware proof without turning your life into months of manual field testing. "
        "It plans the EdgeTwin → Edge Impulse/export → edge node → controlled demo loop."
    )

    profiles = core.get_hardware_reference_profiles_v59()
    demo_type = st.selectbox("Reference demo type", profiles, key="v59_demo_type")
    default_profile = core.HARDWARE_REFERENCE_PROFILES_V59.get(demo_type, {})

    c1, c2 = st.columns(2)
    with c1:
        board = st.selectbox(
            "Reference board",
            ["Auto", "ESP32-S3", "RAK4631 / nRF52840", "STM32U5", "STM32H7", "Raspberry Pi Zero 2 W", "Generic Linux Gateway"],
            key="v59_board",
        )
        sensors = st.multiselect(
            "Sensor stack",
            core.get_sensor_options(),
            default=default_profile.get("default_sensors", ["Audio", "Vibration"]),
            key="v59_sensors",
        )
        expected_demo_days = st.number_input(
            "Expected controlled demo days",
            min_value=1,
            max_value=30,
            value=int(default_profile.get("minimum_demo_days", 2)),
            step=1,
            key="v59_demo_days",
        )
    with c2:
        target_environment = st.text_input(
            "Target environment",
            value=default_profile.get("target_environment", "controlled bench / field reference area"),
            key="v59_target_environment",
        )
        edge_route = st.selectbox(
            "Edge route",
            ["Auto", "Edge Impulse anomaly/K-means", "Edge Impulse classifier", "TinyML export", "LoRa event scoring + gateway", "Hybrid anomaly + classifier"],
            key="v59_edge_route",
        )
        has_real_uploads = st.checkbox("Real uploads available", value=bool(st.session_state.get("real_upload_v56_snapshot")), key="v59_has_real_uploads")
        has_ei_export = st.checkbox("Edge Impulse export prepared", value=bool(st.session_state.get("edge_impulse_bundle") or st.session_state.get("edge_impulse_classifier_bundle")), key="v59_has_ei_export")
        has_deployment_plan = st.checkbox("Deployment plan prepared", value=bool(st.session_state.get("deployment_plan") or st.session_state.get("edge_deployment_starter_snapshot")), key="v59_has_deployment")
        has_privacy_plan = st.checkbox("Privacy-safe learning plan prepared", value=bool(st.session_state.get("field_learning_v52_snapshot")), key="v59_has_privacy_plan")

    if st.button("Build Hardware Reference Demo Plan V59", type="primary", use_container_width=True, key="v59_build_reference_demo"):
        snap = core.build_hardware_reference_demo_v59(
            project_name=st.session_state.project_name,
            demo_type=demo_type,
            board=board,
            sensors=sensors,
            target_environment=target_environment,
            expected_demo_days=expected_demo_days,
            edge_impulse_route=edge_route,
            has_real_uploads=has_real_uploads,
            has_edge_impulse_export=has_ei_export,
            has_deployment_plan=has_deployment_plan,
            has_privacy_learning_plan=has_privacy_plan,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        st.session_state.hardware_reference_v59_snapshot = snap
        st.session_state.hardware_reference_v59_bundle = core.create_hardware_reference_demo_v59_bundle(
            st.session_state.project_name,
            snap,
            st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        if snap.get("decision") == "GO":
            st.success("Hardware reference demo plan is GO.")
        elif snap.get("decision") == "CONDITIONAL GO":
            st.warning("Hardware reference demo plan is CONDITIONAL GO.")
        else:
            st.error("Hardware reference demo plan is NO-GO.")

    snap = st.session_state.get("hardware_reference_v59_snapshot")
    if snap:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Reference score", f"{snap.get('hardware_reference_score', 0)}%")
        m2.metric("Decision", snap.get("decision", "Unknown"))
        m3.metric("Board", snap.get("board", "Unknown"))
        m4.metric("Dataset rows", snap.get("dataset_rows", 0))

        st.info(snap.get("readiness", ""))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Next actions")
            for item in snap.get("next_actions", []):
                st.write(f"- {item}")
            st.subheader("Blockers / warnings")
            for item in snap.get("blockers", []):
                st.error(item)
            for item in snap.get("warnings", []):
                st.warning(item)
        with c2:
            st.subheader("Controlled demo steps")
            st.dataframe(pd.DataFrame(snap.get("demo_steps", [])), use_container_width=True)

        with st.expander("BOM, data collection plan and safe claims", expanded=False):
            st.write("**Reference BOM**")
            st.dataframe(pd.DataFrame(snap.get("bom", [])), use_container_width=True)
            st.write("**Data collection plan**")
            st.dataframe(pd.DataFrame(snap.get("data_collection_plan", [])), use_container_width=True)
            st.write("**Safe claims**")
            for item in snap.get("safe_claims", []):
                st.write(f"- {item}")
            st.write("**Claims to avoid**")
            for item in snap.get("claims_to_avoid", []):
                st.write(f"- {item}")

        if st.session_state.hardware_reference_v59_bundle:
            st.download_button(
                "Download Hardware Reference Demo Bundle V59",
                st.session_state.hardware_reference_v59_bundle,
                file_name=f"{st.session_state.project_name}_hardware_reference_v59.zip",
                mime="application/zip",
                use_container_width=True,
                key="v59_download_hardware_reference_bundle",
            )

    st.caption("V59 is intentionally a minimal reference proof layer: enough to prove the pipeline, not enough to claim production certification.")




# ============================================================
# V61 — Traction & Proof Dashboard / Buyer Evidence System
# ============================================================

def render_traction_proof_v61_tab():
    st.header("Traction & Proof Dashboard V61")
    st.write(
        "Track the evidence that makes EdgeTwin valuable: real leads, demos, proposals, paid pilots, real-data uploads, "
        "customer feedback and buyer-readiness signals. This is not a valuation tool; it is a proof discipline."
    )

    with st.expander("Why this matters", expanded=True):
        st.markdown(
            """
            Features alone do not create strategic value. Proof does. V61 helps you move from *great product* to
            *credible business evidence*: who is interested, who pays, what data arrives, what value is proven,
            and what still needs to be collected before bigger launch or partner conversations.
            """
        )

    c1, c2 = st.columns(2)
    with c1:
        period_name = st.text_input("Evidence period", value="Current beta period", key="v61_period_name")
        launch_goal = st.selectbox(
            "Current goal",
            ["First paid pilots", "Private beta", "Paid pilot launch", "Public SaaS preview", "Strategic partner / buyer readiness"],
            index=0,
            key="v61_launch_goal",
        )
        leads_total = st.number_input("Total targeted leads", min_value=0, value=0, step=1, key="v61_leads_total")
        qualified_leads = st.number_input("Qualified leads", min_value=0, value=0, step=1, key="v61_qualified_leads")
        demo_calls = st.number_input("Demo / discovery calls", min_value=0, value=0, step=1, key="v61_demo_calls")
        beta_test_runs = st.number_input("Beta test runs", min_value=0, value=0, step=1, key="v61_beta_runs")
        proposals_sent = st.number_input("Proposals / SOWs sent", min_value=0, value=0, step=1, key="v61_proposals_sent")
        paid_pilots_sold = st.number_input("Paid pilots sold", min_value=0, value=0, step=1, key="v61_paid_pilots")
    with c2:
        pilots_delivered = st.number_input("Pilots delivered", min_value=0, value=0, step=1, key="v61_pilots_delivered")
        real_data_uploads = st.number_input("Real-data uploads", min_value=0, value=int(1 if isinstance(st.session_state.get("real_upload_v56_features_df"), pd.DataFrame) and len(st.session_state.get("real_upload_v56_features_df")) > 0 else 0), step=1, key="v61_real_uploads")
        privacy_opt_ins = st.number_input("Privacy-safe feature learning opt-ins", min_value=0, value=0, step=1, key="v61_privacy_optins")
        generated_datasets = st.number_input("Generated datasets / bundles", min_value=0, value=int(1 if isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset) > 0 else 0), step=1, key="v61_generated_datasets")
        customer_feedback_score = st.slider("Average customer feedback score", 0.0, 5.0, 0.0, 0.1, key="v61_feedback_score")
        pipeline_value_eur = st.number_input("Pipeline value estimate (€)", min_value=0.0, value=0.0, step=100.0, key="v61_pipeline_value")
        collected_revenue_eur = st.number_input("Collected revenue (€)", min_value=0.0, value=0.0, step=100.0, key="v61_collected_revenue")
        avg_hours_saved_per_pilot = st.number_input("Estimated hours saved per pilot", min_value=0.0, value=0.0, step=1.0, key="v61_hours_saved")

    a1, a2, a3 = st.columns(3)
    with a1:
        reference_customer_available = st.checkbox("Reference customer / anonymized case possible", value=False, key="v61_reference_customer")
    with a2:
        repeatable_niche_found = st.checkbox("Repeatable wedge niche found", value=False, key="v61_repeatable_niche")
    with a3:
        partner_or_buyer_inquiries = st.number_input("Partner/buyer inquiries", min_value=0, value=0, step=1, key="v61_partner_inquiries")

    evidence_notes = st.text_area(
        "Evidence notes",
        value="",
        height=90,
        key="v61_evidence_notes",
        help="Add honest notes: who reacted, what confused them, what they would pay for, and what proof is still missing.",
    )

    if st.button("Build Traction & Proof Dashboard V61", type="primary", use_container_width=True, key="v61_build_traction_proof"):
        snapshot = core.build_traction_proof_dashboard_v61(
            project_name=st.session_state.project_name,
            period_name=period_name,
            launch_goal=launch_goal,
            leads_total=leads_total,
            qualified_leads=qualified_leads,
            demo_calls=demo_calls,
            beta_test_runs=beta_test_runs,
            proposals_sent=proposals_sent,
            paid_pilots_sold=paid_pilots_sold,
            pilots_delivered=pilots_delivered,
            real_data_uploads=real_data_uploads,
            privacy_opt_ins=privacy_opt_ins,
            generated_datasets=generated_datasets,
            customer_feedback_score=customer_feedback_score,
            pipeline_value_eur=pipeline_value_eur,
            collected_revenue_eur=collected_revenue_eur,
            avg_hours_saved_per_pilot=avg_hours_saved_per_pilot,
            reference_customer_available=reference_customer_available,
            repeatable_niche_found=repeatable_niche_found,
            partner_or_buyer_inquiries=partner_or_buyer_inquiries,
            evidence_notes=evidence_notes,
            evidence_snapshots={
                "customer_mode_v50": st.session_state.get("customer_mode_v50_snapshot"),
                "customer_ui_v51": st.session_state.get("customer_ui_v51_snapshot"),
                "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
                "launch_experience_v53": st.session_state.get("launch_experience_v53_snapshot"),
                "launch_assets_v54": st.session_state.get("launch_assets_v54_snapshot"),
                "first_customer_beta_v55": st.session_state.get("first_customer_beta_v55_snapshot"),
                "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
                "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
                "cloud_architecture_v58": st.session_state.get("cloud_architecture_v58_snapshot"),
                "hardware_reference_v59": st.session_state.get("hardware_reference_v59_snapshot"),
                "commercial_release_v60": st.session_state.get("commercial_release_v60_snapshot"),
                "launch_stabilization_v60_1": st.session_state.get("launch_stabilization_v60_1_snapshot"),
            },
        )
        st.session_state.traction_proof_v61_snapshot = snapshot
        st.session_state.traction_proof_v61_bundle = core.create_traction_proof_v61_bundle(st.session_state.project_name, snapshot)

        if snapshot.get("decision") == "PROOF BUILDING GO":
            st.success("V61 says you have early commercial proof. Now repeat the niche and document results.")
        elif snapshot.get("decision") == "KEEP SELLING CONTROLLED PILOTS":
            st.warning("V61 sees first revenue signal. Keep selling controlled pilots and collect proof.")
        else:
            st.info("V61 says the next job is evidence: customer calls, proposals, uploads and paid pilot proof.")

    snapshot = st.session_state.get("traction_proof_v61_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Traction score", f"{snapshot.get('traction_score', 0)}%")
        m2.metric("Buyer readiness", f"{snapshot.get('buyer_readiness_score', 0)}%")
        m3.metric("Revenue signal", f"{snapshot.get('revenue_signal_score', 0)}%")
        m4.metric("Stage", snapshot.get("proof_stage", "Unknown"))

        st.info(snapshot.get("recommended_next_step", ""))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Must collect next")
            for item in snapshot.get("must_collect_next", []) or ["No immediate proof gaps recorded."]:
                st.write(f"- {item}")
            st.subheader("Next 30 days")
            for item in snapshot.get("next_30_days", []):
                st.write(f"- {item}")
        with c2:
            st.subheader("Proof metrics")
            st.dataframe(pd.DataFrame(snapshot.get("proof_metrics", [])), use_container_width=True)

        st.subheader("Conversion funnel")
        funnel_df = pd.DataFrame(snapshot.get("conversion_funnel", []))
        st.dataframe(funnel_df, use_container_width=True)
        if len(funnel_df) > 0 and "rate_percent" in funnel_df.columns:
            fig = px.bar(funnel_df, x="step", y="rate_percent", title="Evidence funnel conversion rates")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Evidence assets / buyer data room", expanded=False):
            st.dataframe(pd.DataFrame(snapshot.get("evidence_assets", [])), use_container_width=True)
            st.markdown("**Buyer/partner data room checklist**")
            for item in snapshot.get("buyer_data_room", []):
                st.write(f"- {item}")

        with st.expander("Safe claims / claims to avoid", expanded=False):
            st.markdown("**Safe claims**")
            for item in snapshot.get("safe_claims", []):
                st.write(f"- {item}")
            st.markdown("**Claims to avoid**")
            for item in snapshot.get("claims_to_avoid", []):
                st.write(f"- {item}")

        if st.session_state.get("traction_proof_v61_bundle"):
            st.download_button(
                "Download Traction & Proof Bundle V61",
                st.session_state.traction_proof_v61_bundle,
                file_name=f"{st.session_state.project_name}_traction_proof_v61.zip",
                mime="application/zip",
                use_container_width=True,
                key="v61_download_traction_proof_bundle",
            )

    st.caption("V61 is for evidence discipline. It does not guarantee valuation, investment, acquisition or production accuracy.")

# ============================================================
# V60.1 — Launch Stabilization & Strategic Moat Gate
# ============================================================

def render_launch_stabilization_v60_1_tab():
    st.header("Launch Stabilization & Strategic Moat Gate V60.1")
    st.write(
        "This gate freezes the product into a safer public-beta / paid-pilot shape and checks whether EdgeTwin is building strategic value, not just features. "
        "It is not a valuation promise or acquisition guarantee; it is a practical readiness and moat checklist."
    )

    c1, c2 = st.columns(2)
    with c1:
        release_target = st.selectbox(
            "Stabilization target",
            ["Private beta", "Paid pilot launch", "Public demo preview", "Strategic partner / buyer demo"],
            index=1,
            key="v60_1_release_target",
        )
        full_test_passed = st.checkbox("Latest full test/check-up passed", value=False, key="v60_1_full_test_passed")
        no_critical_errors = st.checkbox("No known critical Streamlit/backend errors", value=True, key="v60_1_no_critical_errors")
        customer_mode_clean = st.checkbox("Customer Mode is clean and simple", value=True, key="v60_1_customer_mode_clean")
        founder_controls_hidden = st.checkbox("Founder/admin controls are hidden from customers", value=True, key="v60_1_founder_hidden")
        safe_claims_locked = st.checkbox("Safe claims/disclaimers are locked", value=True, key="v60_1_safe_claims_locked")
    with c2:
        privacy_terms_ready = st.checkbox("Privacy/consent wording is ready", value=bool(st.session_state.get("field_learning_v52_snapshot")), key="v60_1_privacy_ready")
        payment_lock_safe = st.checkbox("Payment/download locks are safe", value=bool(st.session_state.get("checkout_v57_snapshot")), key="v60_1_payment_lock")
        storage_backup_plan = st.checkbox("Storage/back-up plan documented", value=True, key="v60_1_storage_backup")
        cloud_migration_plan = st.checkbox("Cloud migration plan documented", value=bool(st.session_state.get("cloud_architecture_v58_snapshot")), key="v60_1_cloud_plan")
        hardware_reference_evidence = st.checkbox("Hardware reference proof path exists", value=bool(st.session_state.get("hardware_reference_v59_snapshot")), key="v60_1_hardware_reference")
        legal_review_started = st.checkbox("Legal/terms review started", value=False, key="v60_1_legal_started")

    with st.expander("Strategic moat / buyer-value assumptions", expanded=False):
        moat_notes = st.text_area(
            "What makes EdgeTwin hard to copy?",
            value="Integrated sensor-pilot workflow, trust gates, privacy-safe field learning, paid-pilot delivery system, hardware reference proof and founder operations layer.",
            height=110,
            key="v60_1_moat_notes",
        )
        buyer_target = st.selectbox(
            "Possible strategic buyer category",
            ["Edge AI platform", "Industrial IoT company", "Sensor hardware vendor", "Predictive maintenance vendor", "Security/remote monitoring company", "Consultancy/SI", "Not targeting buyers yet"],
            index=0,
            key="v60_1_buyer_target",
        )

    if st.button("Build Launch Stabilization Plan V60.1", type="primary", use_container_width=True, key="v60_1_build_launch_stabilization"):
        snapshot = core.build_launch_stabilization_and_moat_v60_1(
            project_name=st.session_state.project_name,
            release_target=release_target,
            full_test_passed=full_test_passed,
            no_critical_errors=no_critical_errors,
            customer_mode_clean=customer_mode_clean,
            founder_controls_hidden=founder_controls_hidden,
            safe_claims_locked=safe_claims_locked,
            privacy_terms_ready=privacy_terms_ready,
            payment_lock_safe=payment_lock_safe,
            storage_backup_plan=storage_backup_plan,
            cloud_migration_plan=cloud_migration_plan,
            hardware_reference_evidence=hardware_reference_evidence,
            legal_review_started=legal_review_started,
            buyer_target=buyer_target,
            moat_notes=moat_notes,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            evidence_snapshots={
                "commercial_release_v60": st.session_state.get("commercial_release_v60_snapshot"),
                "customer_mode_v50": st.session_state.get("customer_mode_v50_snapshot"),
                "customer_ui_v51": st.session_state.get("customer_ui_v51_snapshot"),
                "privacy_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
                "launch_experience_v53": st.session_state.get("launch_experience_v53_snapshot"),
                "launch_assets_v54": st.session_state.get("launch_assets_v54_snapshot"),
                "first_customer_beta_v55": st.session_state.get("first_customer_beta_v55_snapshot"),
                "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
                "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
                "cloud_architecture_v58": st.session_state.get("cloud_architecture_v58_snapshot"),
                "hardware_reference_v59": st.session_state.get("hardware_reference_v59_snapshot"),
                "pricing_offer": st.session_state.get("pricing_offer_snapshot"),
                "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
                "founder_ops_v49": st.session_state.get("founder_ops_v49_snapshot"),
                "delivery": st.session_state.get("customer_delivery_snapshot"),
            },
        )
        st.session_state.launch_stabilization_v60_1_snapshot = snapshot
        st.session_state.launch_stabilization_v60_1_bundle = core.create_launch_stabilization_v60_1_bundle(
            st.session_state.project_name,
            snapshot,
            st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        if snapshot.get("decision") == "STABLE BETA GO":
            st.success("V60.1 says this build is stable enough for controlled beta / paid-pilot conversations.")
        elif snapshot.get("decision") == "CONTROLLED LAUNCH":
            st.warning("V60.1 says controlled launch only: fix the listed items before wider exposure.")
        else:
            st.error("V60.1 says stabilize first before presenting this as launch-ready.")

    snapshot = st.session_state.get("launch_stabilization_v60_1_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stability", f"{snapshot.get('stability_score', 0)}%")
        m2.metric("Moat", f"{snapshot.get('moat_score', 0)}%")
        m3.metric("Buyer signal", f"{snapshot.get('buyer_signal_score', 0)}%")
        m4.metric("Decision", snapshot.get("decision", "Unknown"))

        st.info(snapshot.get("recommended_next_step", ""))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Must-fix before wider launch")
            for item in snapshot.get("must_fix", []) or ["No launch-blocking items recorded."]:
                st.error(item) if snapshot.get("must_fix") else st.success(item)
            st.subheader("Stabilization sprint")
            for item in snapshot.get("stabilization_sprint", []):
                st.write(f"- {item}")
        with c2:
            st.subheader("Strategic moat evidence")
            st.dataframe(pd.DataFrame(snapshot.get("moat_evidence", [])), use_container_width=True)

        with st.expander("Buyer / partner data-room checklist", expanded=False):
            for item in snapshot.get("data_room_checklist", []):
                st.write(f"- {item}")

        with st.expander("Safe positioning / claims to avoid", expanded=False):
            st.markdown("**Safe external positioning**")
            for item in snapshot.get("safe_positioning", []):
                st.write(f"- {item}")
            st.markdown("**Claims to avoid**")
            for item in snapshot.get("claims_to_avoid", []):
                st.write(f"- {item}")

        if st.session_state.launch_stabilization_v60_1_bundle:
            st.download_button(
                "Download Launch Stabilization Bundle V60.1",
                st.session_state.launch_stabilization_v60_1_bundle,
                file_name=f"{st.session_state.project_name}_launch_stabilization_v60_1.zip",
                mime="application/zip",
                use_container_width=True,
                key="v60_1_download_launch_stabilization_bundle",
            )

    st.caption("V60.1 is about stability, defensibility and strategic value. It does not guarantee acquisition interest, valuation or legal readiness.")

# ============================================================
# V60 — Commercial Release Candidate / Final Launch Gate
# ============================================================

def render_commercial_release_v60_tab():
    st.header("Commercial Release Candidate / Final Launch Gate V60")
    st.write(
        "This is the final commercial gate before showing EdgeTwin as a launch-ready paid-pilot product. "
        "It combines product readiness, security, delivery, pricing, checkout, privacy-safe learning, real upload intake, cloud architecture and hardware reference proof."
    )

    c1, c2 = st.columns(2)
    with c1:
        release_target = st.selectbox(
            "Release target",
            ["Private beta", "Paid pilot launch", "Public SaaS preview", "Enterprise/on-prem evaluation"],
            index=1,
            key="v60_release_target",
        )
        customer_mode_ready = st.checkbox("Customer Mode is clear and tested", value=True, key="v60_customer_mode_ready")
        founder_mode_ready = st.checkbox("Founder Mode/operator controls are ready", value=True, key="v60_founder_mode_ready")
        pricing_ready = st.checkbox("Pricing/offer/SOW/checkout flow is ready", value=bool(st.session_state.get("pricing_offer_snapshot") or st.session_state.get("checkout_v57_snapshot")), key="v60_pricing_ready")
        delivery_ready = st.checkbox("Customer delivery bundle/process is ready", value=bool(st.session_state.get("delivery_snapshot") or st.session_state.get("paid_pilot_v45_snapshot")), key="v60_delivery_ready")
    with c2:
        real_data_ready = st.checkbox("Real upload / privacy-safe learning path is ready", value=bool(st.session_state.get("real_upload_v56_snapshot") or st.session_state.get("field_learning_v52_snapshot")), key="v60_real_data_ready")
        hardware_reference_ready = st.checkbox("Hardware reference proof path is ready", value=bool(st.session_state.get("hardware_reference_v59_snapshot")), key="v60_hardware_ready")
        cloud_plan_ready = st.checkbox("Cloud migration plan is documented", value=bool(st.session_state.get("cloud_architecture_v58_snapshot")), key="v60_cloud_ready")
        legal_review_done = st.checkbox("Legal/terms review done", value=False, key="v60_legal_done")
        payment_provider_connected = st.checkbox("Real payment provider connected", value=False, key="v60_payment_connected")

    if st.button("Build Commercial Release Candidate V60", type="primary", use_container_width=True, key="v60_build_release_candidate"):
        snapshot = core.build_commercial_release_candidate_v60(
            project_name=st.session_state.project_name,
            release_target=release_target,
            customer_mode_ready=customer_mode_ready,
            founder_mode_ready=founder_mode_ready,
            pricing_ready=pricing_ready,
            delivery_ready=delivery_ready,
            real_data_ready=real_data_ready,
            hardware_reference_ready=hardware_reference_ready,
            cloud_plan_ready=cloud_plan_ready,
            legal_review_done=legal_review_done,
            payment_provider_connected=payment_provider_connected,
            selected_plan=st.session_state.get("selected_plan", "Founder Test Mode"),
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            evidence_snapshots={
                "product_readiness": st.session_state.get("product_readiness_snapshot"),
                "security_v41": st.session_state.get("security_v41_snapshot"),
                "customer_assurance": st.session_state.get("governance_snapshot"),
                "guided_success": st.session_state.get("onboarding_snapshot") or st.session_state.get("guided_success_snapshot"),
                "launch_experience_v53": st.session_state.get("launch_experience_v53_snapshot"),
                "pricing_offer": st.session_state.get("pricing_offer_snapshot"),
                "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
                "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
                "delivery": st.session_state.get("delivery_snapshot"),
                "paid_pilot_v45": st.session_state.get("paid_pilot_v45_snapshot"),
                "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
                "privacy_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
                "cloud_architecture_v58": st.session_state.get("cloud_architecture_v58_snapshot"),
                "hardware_reference_v59": st.session_state.get("hardware_reference_v59_snapshot"),
                "founder_ops_v49": st.session_state.get("founder_ops_v49_snapshot"),
                "observability": st.session_state.get("observability_snapshot"),
            },
        )
        st.session_state.commercial_release_v60_snapshot = snapshot
        st.session_state.commercial_release_v60_bundle = core.create_commercial_release_candidate_v60_bundle(
            st.session_state.project_name,
            snapshot,
            st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        )
        if snapshot.get("decision") == "GO":
            st.success("Commercial Release Candidate is GO for the selected target.")
        elif snapshot.get("decision") == "CONDITIONAL GO":
            st.warning("Commercial Release Candidate is CONDITIONAL GO. Fix the listed blockers before broader launch.")
        else:
            st.error("Commercial Release Candidate is NO-GO. Keep this internal until blockers are fixed.")

    snapshot = st.session_state.get("commercial_release_v60_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Release score", f"{snapshot.get('commercial_release_score', 0)}%")
        m2.metric("Decision", snapshot.get("decision", "Unknown"))
        m3.metric("Release target", snapshot.get("release_target", "Unknown"))
        m4.metric("Launch stage", snapshot.get("launch_stage", "Unknown"))

        st.info(snapshot.get("recommended_next_step", ""))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Must-fix before launch")
            for item in snapshot.get("must_fix", []) or ["No launch-blocking issues recorded."]:
                st.error(item) if snapshot.get("must_fix") else st.success(item)
            st.subheader("Recommended actions")
            for item in snapshot.get("recommended_actions", []):
                st.write(f"- {item}")
        with c2:
            st.subheader("Release evidence")
            st.dataframe(pd.DataFrame(snapshot.get("evidence_matrix", [])), use_container_width=True)

        with st.expander("Safe launch claims / claims to avoid", expanded=False):
            st.markdown("**Safe claims**")
            for item in snapshot.get("safe_claims", []):
                st.write(f"- {item}")
            st.markdown("**Claims to avoid**")
            for item in snapshot.get("claims_to_avoid", []):
                st.write(f"- {item}")

        if st.session_state.commercial_release_v60_bundle:
            st.download_button(
                "Download Commercial Release Candidate Bundle V60",
                st.session_state.commercial_release_v60_bundle,
                file_name=f"{st.session_state.project_name}_commercial_release_candidate_v60.zip",
                mime="application/zip",
                use_container_width=True,
                key="v60_download_commercial_release_candidate_bundle",
            )

    st.caption("V60 is a commercial release gate, not a legal certification. Use it to decide demo, paid pilot, public preview or keep-internal status.")


# ============================================================
# V62 — ROI & Value Proof Center / Customer Business Case
# ============================================================

def render_roi_value_v62_tab():
    st.header("ROI & Value Proof Center V62")
    st.write(
        "Turn the technical pilot output into a customer business case: time saved, cost avoided, risk reduced, "
        "payback logic and safe value claims. This helps justify pilot pricing without overpromising production accuracy."
    )

    c1, c2 = st.columns([1.15, 1])
    with c1:
        customer_segment = st.selectbox(
            "Customer segment",
            ["Industrial maintenance", "Security / anti-tamper", "Remote assets / forestry", "IoT consultant", "OEM / hardware company", "Custom"],
            key="v62_customer_segment",
        )
        use_case = st.text_input(
            "Use-case / pain point",
            value="Reduce time and uncertainty when starting an Edge AI sensor pilot",
            key="v62_use_case",
        )
        pilot_package = st.selectbox(
            "Offer/package",
            ["Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Pilot", "Enterprise / Custom Pilot"],
            index=1,
            key="v62_pilot_package",
        )
        expected_pilot_price = st.number_input("Pilot price / quote EUR", min_value=0.0, max_value=100000.0, value=1500.0, step=100.0, key="v62_pilot_price")
        hardware_budget = st.number_input("Estimated hardware / test budget EUR", min_value=0.0, max_value=100000.0, value=350.0, step=50.0, key="v62_hardware_budget")

    with c2:
        hours_without = st.number_input("Estimated hours without EdgeTwin", min_value=0.0, max_value=2000.0, value=80.0, step=5.0, key="v62_hours_without")
        hours_with = st.number_input("Estimated hours with EdgeTwin", min_value=0.0, max_value=2000.0, value=24.0, step=5.0, key="v62_hours_with")
        hourly_cost = st.number_input("Blended hourly cost EUR", min_value=0.0, max_value=1000.0, value=75.0, step=5.0, key="v62_hourly_cost")
        current_delay_weeks = st.number_input("Estimated pilot delay avoided / weeks", min_value=0.0, max_value=52.0, value=3.0, step=0.5, key="v62_delay_weeks")
        risk_reduction_score = st.slider("Risk/uncertainty reduction estimate", 0, 100, 70, key="v62_risk_reduction")
        real_data_available = st.checkbox("Customer has real data or can upload sample files", value=bool(st.session_state.get("real_upload_v56_snapshot")), key="v62_real_data_available")
        privacy_learning_allowed = st.checkbox("Privacy-safe feature learning is allowed", value=False, key="v62_privacy_learning_allowed")

    with st.expander("Evidence snapshots used by V62", expanded=False):
        evidence_snapshots = {
            "traction_proof_v61": st.session_state.get("traction_proof_v61_snapshot"),
            "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
            "privacy_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
            "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
            "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
            "hardware_reference_v59": st.session_state.get("hardware_reference_v59_snapshot"),
            "commercial_release_v60": st.session_state.get("commercial_release_v60_snapshot"),
        }
        st.json({k: bool(v) for k, v in evidence_snapshots.items()})

    evidence_notes = st.text_area(
        "Optional evidence notes",
        value="Use this case as a controlled paid-pilot value estimate, not a production guarantee.",
        height=90,
        key="v62_evidence_notes",
    )

    if st.button("Build ROI & Value Case V62", type="primary", use_container_width=True, key="v62_build_roi_value_case"):
        snapshot = core.build_roi_value_case_v62(
            project_name=st.session_state.project_name,
            customer_segment=customer_segment,
            use_case=use_case,
            pilot_package=pilot_package,
            estimated_hours_without_edgetwin=hours_without,
            estimated_hours_with_edgetwin=hours_with,
            blended_hourly_cost_eur=hourly_cost,
            expected_pilot_price_eur=expected_pilot_price,
            hardware_budget_eur=hardware_budget,
            current_delay_weeks=current_delay_weeks,
            risk_reduction_score=risk_reduction_score,
            real_data_available=real_data_available,
            privacy_learning_allowed=privacy_learning_allowed,
            evidence_snapshots=evidence_snapshots,
            evidence_notes=evidence_notes,
        )
        st.session_state.roi_value_v62_snapshot = snapshot
        st.session_state.roi_value_v62_bundle = core.create_roi_value_v62_bundle(st.session_state.project_name, snapshot)
        if snapshot.get("decision") == "GO":
            st.success("V62 says the business case is strong enough for a paid-pilot offer.")
        elif snapshot.get("decision") == "CONDITIONAL GO":
            st.warning("V62 says the value case is usable, but you need more evidence or tighter assumptions.")
        else:
            st.error("V62 says the value case is weak. Do not sell this as premium yet.")

    snapshot = st.session_state.get("roi_value_v62_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Value score", f"{snapshot.get('value_proof_score', 0)}%")
        m2.metric("Decision", snapshot.get("decision", "Unknown"))
        m3.metric("Time saved", f"{snapshot.get('time_saved_hours', 0)} h")
        m4.metric("ROI estimate", f"{snapshot.get('roi_percent', 0)}%")

        st.info(snapshot.get("recommended_next_step", ""))
        st.subheader("Customer business case")
        st.write(snapshot.get("customer_business_case", ""))

        a, b = st.columns(2)
        with a:
            st.subheader("Economic summary")
            st.dataframe(pd.DataFrame(snapshot.get("economic_summary", [])), use_container_width=True)
            st.subheader("Proof needed next")
            for item in snapshot.get("proof_needed_next", []) or ["No major proof gaps recorded."]:
                st.write(f"- {item}")
        with b:
            st.subheader("Customer questions")
            for item in snapshot.get("customer_questions", []):
                st.write(f"- {item}")
            st.subheader("Value narrative")
            for item in snapshot.get("value_narrative", []):
                st.write(f"- {item}")

        with st.expander("Safe claims / claims to avoid", expanded=False):
            st.markdown("**Safe value claims**")
            for item in snapshot.get("safe_value_claims", []):
                st.write(f"- {item}")
            st.markdown("**Claims to avoid**")
            for item in snapshot.get("claims_to_avoid", []):
                st.write(f"- {item}")

        if st.session_state.get("roi_value_v62_bundle"):
            st.download_button(
                "Download ROI & Value Proof Bundle V62",
                st.session_state.roi_value_v62_bundle,
                file_name=f"{st.session_state.project_name}_roi_value_proof_v62.zip",
                mime="application/zip",
                use_container_width=True,
                key="v62_download_roi_value_bundle",
            )

    st.caption("V62 provides a business-case estimate for pilot sales. It is not financial, legal, investment or production-performance advice.")


# ============================================================
# V63 — Case Study & Customer Proof Pack / Buyer Evidence Story
# ============================================================

def render_case_study_v63_tab():
    st.header("Case Study & Customer Proof Pack V63")
    st.write(
        "Turn pilot outcomes, ROI assumptions and customer feedback into safe proof assets: an anonymised case study, "
        "testimonial request, buyer evidence summary and follow-up story. This helps EdgeTwin prove value without overclaiming."
    )

    c1, c2 = st.columns([1.15, 1])
    with c1:
        customer_segment = st.text_input("Customer segment", value="Industrial maintenance / operations team", key="v63_customer_segment")
        use_case = st.text_area(
            "Use-case story",
            value="The customer wants to reduce uncertainty and time before starting an Edge AI sensor pilot.",
            height=90,
            key="v63_use_case_story",
        )
        problem_before = st.text_area(
            "Before EdgeTwin: customer pain",
            value="Manual sensor-data preparation, unclear labels, uncertain hardware direction and no clear pilot-readiness report.",
            height=90,
            key="v63_problem_before",
        )
        outcome_summary = st.text_area(
            "Outcome summary",
            value="EdgeTwin generated a pilot package with dataset, reliability checks, hardware direction, report and commercial handoff.",
            height=90,
            key="v63_outcome_summary",
        )
    with c2:
        pilot_stage = st.selectbox(
            "Pilot stage",
            ["Demo completed", "Proposal sent", "Paid pilot accepted", "Pilot delivered", "Reference approved"],
            index=1,
            key="v63_pilot_stage",
        )
        publish_mode = st.selectbox(
            "Publishing mode",
            ["Internal only", "Anonymous public case study", "Named public case study with written approval"],
            index=1,
            key="v63_publish_mode",
        )
        customer_rating = st.slider("Customer feedback rating", 0, 10, 7, key="v63_customer_rating")
        real_data_used = st.checkbox("Real customer data was used", value=bool(st.session_state.get("real_upload_v56_snapshot")), key="v63_real_data_used")
        written_permission = st.checkbox("Written permission for external use", value=False, key="v63_written_permission")
        anonymise_customer = st.checkbox("Anonymise customer/company details", value=True, key="v63_anonymise_customer")
        include_roi = st.checkbox("Include ROI/value proof", value=bool(st.session_state.get("roi_value_v62_snapshot")), key="v63_include_roi")

    with st.expander("Evidence snapshots used by V63", expanded=False):
        evidence = {
            "ROI Value V62": bool(st.session_state.get("roi_value_v62_snapshot")),
            "Traction Proof V61": bool(st.session_state.get("traction_proof_v61_snapshot")),
            "Real Upload V56": bool(st.session_state.get("real_upload_v56_snapshot")),
            "Privacy Learning V52": bool(st.session_state.get("field_learning_v52_snapshot")),
            "Proposal / SOW": bool(st.session_state.get("proposal_sow_snapshot")),
            "Checkout V57": bool(st.session_state.get("checkout_v57_snapshot")),
            "Delivery Portal": bool(st.session_state.get("customer_delivery_snapshot")),
            "Hardware Reference V59": bool(st.session_state.get("hardware_reference_v59_snapshot")),
        }
        st.dataframe(pd.DataFrame([{"Evidence": k, "Available": v} for k, v in evidence.items()]), use_container_width=True)

    if st.button("Build Case Study Proof Pack V63", type="primary", use_container_width=True, key="v63_build_case_study"):
        snapshot = core.build_case_study_proof_pack_v63(
            project_name=st.session_state.project_name,
            customer_segment=customer_segment,
            use_case=use_case,
            problem_before=problem_before,
            outcome_summary=outcome_summary,
            pilot_stage=pilot_stage,
            publish_mode=publish_mode,
            customer_rating=customer_rating,
            real_data_used=real_data_used,
            written_permission=written_permission,
            anonymise_customer=anonymise_customer,
            include_roi=include_roi,
            evidence_snapshots={
                "roi_value_v62": st.session_state.get("roi_value_v62_snapshot"),
                "traction_proof_v61": st.session_state.get("traction_proof_v61_snapshot"),
                "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
                "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
                "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
                "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
                "delivery_portal": st.session_state.get("customer_delivery_snapshot"),
                "hardware_reference_v59": st.session_state.get("hardware_reference_v59_snapshot"),
            },
        )
        st.session_state.case_study_v63_snapshot = snapshot
        st.session_state.case_study_v63_bundle = core.create_case_study_v63_bundle(st.session_state.project_name, snapshot)

    snapshot = st.session_state.get("case_study_v63_snapshot")
    if snapshot:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Proof score", f"{snapshot.get('proof_score', 0)}%")
        c2.metric("Publish readiness", f"{snapshot.get('publish_readiness_score', 0)}%")
        c3.metric("Buyer evidence", f"{snapshot.get('buyer_evidence_score', 0)}%")
        c4.metric("Decision", snapshot.get("decision", "Unknown"))

        decision = snapshot.get("decision")
        if decision == "PUBLIC READY":
            st.success(snapshot.get("recommended_next_step", ""))
        elif decision == "INTERNAL READY":
            st.warning(snapshot.get("recommended_next_step", ""))
        else:
            st.error(snapshot.get("recommended_next_step", ""))

        st.subheader("Customer-safe case study draft")
        st.write(snapshot.get("public_case_study_draft", ""))

        st.subheader("Buyer evidence summary")
        st.write(snapshot.get("buyer_evidence_summary", ""))

        t1, t2 = st.tabs(["Proof assets", "Safe claims & blocked claims"])
        with t1:
            st.dataframe(pd.DataFrame(snapshot.get("proof_assets", [])), use_container_width=True)
            st.markdown("#### Testimonial request email")
            st.code(snapshot.get("testimonial_request_email", ""))
            st.markdown("#### LinkedIn / outreach proof snippet")
            st.code(snapshot.get("short_social_proof", ""))
        with t2:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Safe claims")
                for item in snapshot.get("safe_claims", []):
                    st.success(item)
            with col_b:
                st.markdown("#### Claims to avoid")
                for item in snapshot.get("claims_to_avoid", []):
                    st.warning(item)

        if st.session_state.get("case_study_v63_bundle"):
            st.download_button(
                "Download Case Study Proof Pack V63",
                st.session_state.case_study_v63_bundle,
                file_name=f"{st.session_state.project_name}_case_study_proof_v63.zip",
                mime="application/zip",
                use_container_width=True,
                key="v63_download_case_study_bundle",
            )

    st.caption("V63 creates proof assets for sales, reference calls and future buyer/partner evidence. It does not claim guaranteed outcomes, public rights or acquisition value.")




# ============================================================
# V64 BUYER DATA ROOM / STRATEGIC PARTNER READINESS
# ============================================================

def render_buyer_dataroom_v64_tab():
    st.header("Buyer Data Room V64")
    st.write(
        "Turn EdgeTwin proof into a structured partner/buyer evidence room. "
        "This does not promise a buyout; it shows what proof is ready, what is missing, and what should stay confidential."
    )

    c1, c2 = st.columns([1.15, 1])
    with c1:
        strategic_goal = st.selectbox(
            "Strategic goal",
            [
                "Prepare for strategic partnership",
                "Prepare for reseller/OEM conversation",
                "Prepare for investor/data-room review",
                "Prepare for future acquisition interest",
                "Internal maturity check only",
            ],
            index=0,
            key="v64_strategic_goal",
        )
        target_buyer_type = st.selectbox(
            "Likely interested party",
            [
                "Industrial IoT / edge AI platform",
                "Predictive maintenance vendor",
                "Sensor hardware / OEM company",
                "Cloud/AI developer platform",
                "Security/remote monitoring company",
                "Consultancy / systems integrator",
                "Unknown / not defined yet",
            ],
            index=0,
            key="v64_target_buyer_type",
        )
        moat_summary = st.text_area(
            "Why EdgeTwin could be strategically valuable",
            value="EdgeTwin automates the sensor-pilot chain: use-case intake, synthetic/real data checks, reliability gates, hardware direction, reports, proposal, checkout readiness, delivery and learning feedback.",
            height=115,
            key="v64_moat_summary",
        )
    with c2:
        has_paid_pilots = st.checkbox("At least one paid pilot exists", value=bool(st.session_state.get("paid_pilot_v45_snapshot")), key="v64_has_paid_pilots")
        has_real_uploads = st.checkbox("Real customer data uploads exist", value=bool(st.session_state.get("real_upload_v56_snapshot")), key="v64_has_real_uploads")
        has_case_study = st.checkbox("At least one case study/proof pack exists", value=bool(st.session_state.get("case_study_v63_snapshot")), key="v64_has_case_study")
        has_revenue_signal = st.checkbox("Revenue/pipeline evidence exists", value=bool(st.session_state.get("traction_proof_v61_snapshot") or st.session_state.get("roi_value_v62_snapshot")), key="v64_has_revenue_signal")
        has_privacy_controls = st.checkbox("Privacy-safe learning controls documented", value=bool(st.session_state.get("field_learning_v52_snapshot")), key="v64_has_privacy_controls")
        has_cloud_plan = st.checkbox("Cloud/production migration plan documented", value=bool(st.session_state.get("cloud_architecture_v58_snapshot")), key="v64_has_cloud_plan")
        has_hardware_reference = st.checkbox("Hardware reference proof plan exists", value=bool(st.session_state.get("hardware_reference_v59_snapshot")), key="v64_has_hardware_reference")
        codebase_documented = st.checkbox("Core product modules and limits are documented", value=True, key="v64_codebase_documented")

    with st.expander("Evidence already available in this workspace", expanded=False):
        evidence = {
            "Traction Proof V61": bool(st.session_state.get("traction_proof_v61_snapshot")),
            "ROI Value V62": bool(st.session_state.get("roi_value_v62_snapshot")),
            "Case Study V63": bool(st.session_state.get("case_study_v63_snapshot")),
            "Privacy Learning V52": bool(st.session_state.get("field_learning_v52_snapshot")),
            "Real Upload V56": bool(st.session_state.get("real_upload_v56_snapshot")),
            "Checkout V57": bool(st.session_state.get("checkout_v57_snapshot")),
            "Cloud Architecture V58": bool(st.session_state.get("cloud_architecture_v58_snapshot")),
            "Hardware Reference V59": bool(st.session_state.get("hardware_reference_v59_snapshot")),
            "Commercial Release V60": bool(st.session_state.get("commercial_release_v60_snapshot")),
            "Launch Stabilizer V60.1": bool(st.session_state.get("launch_stabilization_v60_1_snapshot")),
        }
        st.dataframe(pd.DataFrame([{"Evidence": k, "Available": v} for k, v in evidence.items()]), use_container_width=True)

    if st.button("Build Buyer Data Room V64", type="primary", use_container_width=True, key="v64_build_buyer_dataroom"):
        snapshot = core.build_buyer_dataroom_v64(
            project_name=st.session_state.project_name,
            strategic_goal=strategic_goal,
            target_buyer_type=target_buyer_type,
            moat_summary=moat_summary,
            flags={
                "has_paid_pilots": has_paid_pilots,
                "has_real_uploads": has_real_uploads,
                "has_case_study": has_case_study,
                "has_revenue_signal": has_revenue_signal,
                "has_privacy_controls": has_privacy_controls,
                "has_cloud_plan": has_cloud_plan,
                "has_hardware_reference": has_hardware_reference,
                "codebase_documented": codebase_documented,
            },
            evidence_snapshots={
                "traction_proof_v61": st.session_state.get("traction_proof_v61_snapshot"),
                "roi_value_v62": st.session_state.get("roi_value_v62_snapshot"),
                "case_study_v63": st.session_state.get("case_study_v63_snapshot"),
                "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
                "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
                "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
                "cloud_architecture_v58": st.session_state.get("cloud_architecture_v58_snapshot"),
                "hardware_reference_v59": st.session_state.get("hardware_reference_v59_snapshot"),
                "commercial_release_v60": st.session_state.get("commercial_release_v60_snapshot"),
                "launch_stabilization_v60_1": st.session_state.get("launch_stabilization_v60_1_snapshot"),
            },
        )
        st.session_state.buyer_dataroom_v64_snapshot = snapshot
        st.session_state.buyer_dataroom_v64_bundle = core.create_buyer_dataroom_v64_bundle(st.session_state.project_name, snapshot)

    snapshot = st.session_state.get("buyer_dataroom_v64_snapshot")
    if snapshot:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Data-room readiness", f"{snapshot.get('data_room_readiness_score', 0)}%")
        c2.metric("Strategic value signal", f"{snapshot.get('strategic_value_score', 0)}%")
        c3.metric("Proof completeness", f"{snapshot.get('proof_completeness_score', 0)}%")
        c4.metric("Decision", snapshot.get("decision", "Unknown"))

        decision = snapshot.get("decision")
        if decision == "PARTNER READY":
            st.success(snapshot.get("recommended_next_step", ""))
        elif decision == "BUILD MORE PROOF":
            st.warning(snapshot.get("recommended_next_step", ""))
        else:
            st.info(snapshot.get("recommended_next_step", ""))

        t1, t2, t3 = st.tabs(["Data room", "Moat & proof", "Sensitive items"])
        with t1:
            st.subheader("Suggested data-room folders")
            st.dataframe(pd.DataFrame(snapshot.get("data_room_folders", [])), use_container_width=True)
            st.subheader("30-day proof plan")
            for item in snapshot.get("next_30_day_plan", []):
                st.info(item)
        with t2:
            st.subheader("Strategic narrative")
            st.write(snapshot.get("strategic_narrative", ""))
            st.subheader("Buyer/partner questions to be ready for")
            for item in snapshot.get("buyer_questions", []):
                st.write(f"- {item}")
            st.subheader("Moat checklist")
            st.dataframe(pd.DataFrame(snapshot.get("moat_checklist", [])), use_container_width=True)
        with t3:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Shareable with care")
                for item in snapshot.get("shareable_with_care", []):
                    st.success(item)
            with col_b:
                st.markdown("#### Do not share publicly")
                for item in snapshot.get("do_not_share_publicly", []):
                    st.warning(item)

        st.subheader("Claims to avoid")
        for item in snapshot.get("claims_to_avoid", []):
            st.warning(item)

        if st.session_state.get("buyer_dataroom_v64_bundle"):
            st.download_button(
                "Download Buyer Data Room V64",
                st.session_state.buyer_dataroom_v64_bundle,
                file_name=f"{st.session_state.project_name}_buyer_dataroom_v64.zip",
                mime="application/zip",
                use_container_width=True,
                key="v64_download_buyer_dataroom_bundle",
            )

    st.caption("V64 helps structure strategic evidence. It is not legal, tax, investment, valuation or acquisition advice, and it does not guarantee buyer interest.")




# ============================================================
# V65 — IP & Moat Registry / Defensibility Center
# ============================================================

def render_ip_moat_v65_tab():
    st.header("IP & Moat Registry / Defensibility Center V65")
    st.write(
        "This internal founder page helps organize what makes EdgeTwin strategically defensible: product workflow, engine logic, privacy-safe learning, proof loops, data rights, tests, and transfer readiness. "
        "It is not legal/IP/patent advice, but it helps avoid chaos before partner, investor or buyer conversations."
    )

    c1, c2 = st.columns([1.25, 1])
    with c1:
        product_summary = st.text_area(
            "Strategic product summary",
            value="EdgeTwin Studio turns sensor/edge-AI ideas into pilot-ready packages: dataset generation, real-upload intake, reliability/trust checks, privacy-safe learning, hardware guidance, proposal/checkout/delivery workflows and proof tracking.",
            height=120,
            key="v65_product_summary",
        )
        unique_assets_text = st.text_area(
            "Unique assets / moat notes",
            value="End-to-end Edge AI pilot workflow\nPrivacy-safe field learning\nDataset Doctor + reliability gates\nReal upload intake\nProposal-to-delivery automation\nBuyer proof and data-room workflow",
            height=140,
            key="v65_unique_assets",
        )
    with c2:
        st.markdown("#### Protection / transfer checks")
        owns_code = st.checkbox("Codebase ownership is tracked", value=True, key="v65_owns_code")
        architecture_documented = st.checkbox("Architecture is documented enough to explain", value=True, key="v65_architecture_documented")
        open_source_reviewed = st.checkbox("Open-source/dependency review started", value=False, key="v65_open_source_reviewed")
        secrets_removed = st.checkbox("Secrets/customer identifiers removed from shareable artifacts", value=True, key="v65_secrets_removed")
        data_rights_policy = st.checkbox("Data rights / retention policy documented", value=bool(st.session_state.get("field_learning_v52_snapshot")), key="v65_data_rights_policy")
        customer_consent_workflow = st.checkbox("Customer consent workflow exists", value=bool(st.session_state.get("field_learning_v52_snapshot")), key="v65_customer_consent_workflow")
        transfer_docs = st.checkbox("Operator/transfer docs started", value=False, key="v65_transfer_docs")
        reproducible_tests = st.checkbox("Smoke/comprehensive tests are reproducible", value=True, key="v65_reproducible_tests")

    e1, e2, e3 = st.columns(3)
    with e1:
        privacy_safe_learning = st.checkbox("Privacy-safe learning present", value=bool(st.session_state.get("field_learning_v52_snapshot")), key="v65_privacy_safe_learning")
        real_upload_flow = st.checkbox("Real upload flow present", value=bool(st.session_state.get("real_upload_v56_snapshot")), key="v65_real_upload_flow")
        traction_evidence = st.checkbox("Traction proof evidence present", value=bool(st.session_state.get("traction_proof_v61_snapshot")), key="v65_traction_evidence")
    with e2:
        roi_proof = st.checkbox("ROI/value proof present", value=bool(st.session_state.get("roi_value_v62_snapshot")), key="v65_roi_proof")
        case_study_proof = st.checkbox("Case study proof present", value=bool(st.session_state.get("case_study_v63_snapshot")), key="v65_case_study_proof")
        buyer_dataroom = st.checkbox("Buyer data-room structure present", value=bool(st.session_state.get("buyer_dataroom_v64_snapshot")), key="v65_buyer_dataroom")
    with e3:
        hardware_reference = st.checkbox("Hardware reference proof/plan present", value=bool(st.session_state.get("hardware_reference_v59_snapshot")), key="v65_hardware_reference")
        cloud_plan = st.checkbox("Cloud migration plan present", value=bool(st.session_state.get("cloud_architecture_v58_snapshot")), key="v65_cloud_plan")
        brand_assets = st.checkbox("Brand/product naming is consistent", value=True, key="v65_brand_assets")

    flags = {
        "owns_code": owns_code,
        "architecture_documented": architecture_documented,
        "open_source_reviewed": open_source_reviewed,
        "secrets_removed": secrets_removed,
        "data_rights_policy": data_rights_policy,
        "customer_consent_workflow": customer_consent_workflow,
        "transfer_docs": transfer_docs,
        "reproducible_tests": reproducible_tests,
        "privacy_safe_learning": privacy_safe_learning,
        "real_upload_flow": real_upload_flow,
        "traction_evidence": traction_evidence,
        "roi_proof": roi_proof,
        "case_study_proof": case_study_proof,
        "buyer_dataroom": buyer_dataroom,
        "hardware_reference": hardware_reference,
        "cloud_plan": cloud_plan,
        "brand_assets": brand_assets,
        "storage_documented": True,
        "founder_ops": bool(st.session_state.get("founder_ops_v49_snapshot")),
        "commercial_packages": bool(st.session_state.get("pricing_offer_snapshot")) or bool(st.session_state.get("proposal_sow_snapshot")),
        "safe_claims": True,
        "known_limits": True,
        "operator_handoff": transfer_docs,
    }

    evidence_snapshots = {
        "Privacy Learning V52": bool(st.session_state.get("field_learning_v52_snapshot")),
        "Real Upload V56": bool(st.session_state.get("real_upload_v56_snapshot")),
        "Cloud Architecture V58": bool(st.session_state.get("cloud_architecture_v58_snapshot")),
        "Hardware Reference V59": bool(st.session_state.get("hardware_reference_v59_snapshot")),
        "Traction V61": bool(st.session_state.get("traction_proof_v61_snapshot")),
        "ROI V62": bool(st.session_state.get("roi_value_v62_snapshot")),
        "Case Study V63": bool(st.session_state.get("case_study_v63_snapshot")),
        "Buyer Data Room V64": bool(st.session_state.get("buyer_dataroom_v64_snapshot")),
    }

    if st.button("Build IP & Moat Registry V65", type="primary", use_container_width=True, key="v65_build_ip_moat"):
        snapshot = core.build_ip_moat_registry_v65(
            st.session_state.project_name,
            product_summary,
            unique_assets_text,
            flags=flags,
            evidence_snapshots=evidence_snapshots,
        )
        st.session_state.ip_moat_v65_snapshot = snapshot
        st.session_state.ip_moat_v65_bundle = core.create_ip_moat_registry_v65_bundle(st.session_state.project_name, snapshot)
        if snapshot.get("decision") == "STRATEGIC ASSET READY":
            st.success(snapshot.get("recommended_next_step", ""))
        elif snapshot.get("decision") == "PROTECT AND PROVE":
            st.warning(snapshot.get("recommended_next_step", ""))
        else:
            st.info(snapshot.get("recommended_next_step", ""))

    snapshot = st.session_state.get("ip_moat_v65_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategic asset", f"{snapshot.get('strategic_asset_score', 0)}%")
        m2.metric("IP clarity", f"{snapshot.get('ip_clarity_score', 0)}%")
        m3.metric("Defensibility", f"{snapshot.get('defensibility_score', 0)}%")
        m4.metric("Transfer readiness", f"{snapshot.get('transfer_readiness_score', 0)}%")

        decision = snapshot.get("decision", "")
        if decision == "STRATEGIC ASSET READY":
            st.success(decision)
        elif decision == "PROTECT AND PROVE":
            st.warning(decision)
        else:
            st.info(decision)

        tabs = st.tabs(["Asset register", "Ownership & risks", "Moat narrative", "Do not share"])
        with tabs[0]:
            st.subheader("Strategic asset register")
            st.dataframe(pd.DataFrame(snapshot.get("asset_register", [])), use_container_width=True)
            st.subheader("Trade-secret candidates")
            for item in snapshot.get("trade_secret_candidates", []):
                st.info(item)
        with tabs[1]:
            st.subheader("Ownership checklist")
            st.dataframe(pd.DataFrame(snapshot.get("ownership_checklist", [])), use_container_width=True)
            st.subheader("Must-fix before serious external sharing")
            for item in snapshot.get("must_fix", []):
                st.warning(item)
            st.subheader("Open-source review notes")
            for item in snapshot.get("open_source_review_notes", []):
                st.write(f"- {item}")
        with tabs[2]:
            st.subheader("Safe external positioning")
            for item in snapshot.get("safe_external_positioning", []):
                st.success(item)
            st.subheader("Next 30-day plan")
            for item in snapshot.get("next_30_day_plan", []):
                st.info(item)
        with tabs[3]:
            st.subheader("Do not share publicly")
            for item in snapshot.get("do_not_share_publicly", []):
                st.error(item)
            st.subheader("Claims to avoid")
            for item in snapshot.get("claims_to_avoid", []):
                st.warning(item)

        if st.session_state.get("ip_moat_v65_bundle"):
            st.download_button(
                "Download IP & Moat Registry V65",
                st.session_state.ip_moat_v65_bundle,
                file_name=f"{st.session_state.project_name}_ip_moat_v65.zip",
                mime="application/zip",
                use_container_width=True,
                key="v65_download_ip_moat_bundle",
            )

    st.caption("V65 is an internal strategic organizer only. It is not legal, patent, IP, valuation, investment or acquisition advice.")



# ============================================================
# V66 — Continuous Improvement & Quality Flywheel
# ============================================================

def render_continuous_improvement_v66_tab():
    st.header("Continuous Improvement & Quality Flywheel V66")
    st.write(
        "V66 is the internal system for making EdgeTwin better without creating feature chaos. "
        "It turns bugs, customer friction, proof gaps, founder workload and reliability signals into a focused improvement backlog."
    )

    c1, c2 = st.columns([1.1, 1])
    with c1:
        release_target = st.selectbox(
            "Improvement target",
            ["Private beta", "Paid pilot launch", "Public SaaS preview", "Strategic partner readiness"],
            index=1,
            key="v66_release_target",
        )
        customer_notes = st.text_area(
            "Customer friction / feedback notes",
            value="Customers need a simpler route, clear next step, safe claims and confidence that real data stays private.",
            height=110,
            key="v66_customer_notes",
        )
        founder_constraints = st.text_area(
            "Founder constraints",
            value="Too much to do, limited time, avoid unpaid custom work, keep the engine strong but the customer UI simple.",
            height=90,
            key="v66_founder_constraints",
        )

    with c2:
        test_pass_rate = st.slider("Test pass rate", 0, 100, 92, key="v66_test_pass_rate")
        open_bugs = st.number_input("Open visible bugs", min_value=0, max_value=50, value=2, step=1, key="v66_open_bugs")
        customer_friction = st.slider("Customer friction", 0, 100, 32, help="Lower is better.", key="v66_customer_friction")
        support_load_hours = st.number_input("Founder support load hours/week", min_value=0.0, max_value=80.0, value=6.0, step=1.0, key="v66_support_load")

    m1, m2, m3 = st.columns(3)
    ui_clarity = m1.slider("Customer UI clarity", 0, 100, 82, key="v66_ui_clarity")
    engine_confidence = m2.slider("Engine confidence", 0, 100, 84, key="v66_engine_confidence")
    cloud_readiness = m3.slider("Cloud readiness", 0, 100, 62, key="v66_cloud_readiness")
    m4, m5, m6 = st.columns(3)
    field_evidence_strength = m4.slider("Field evidence strength", 0, 100, 52, key="v66_field_evidence")
    proof_strength = m5.slider("Traction/proof strength", 0, 100, 58, key="v66_proof_strength")
    privacy_confidence = m6.slider("Privacy confidence", 0, 100, 94, key="v66_privacy_confidence")

    with st.expander("Foundation checks", expanded=True):
        f1, f2, f3 = st.columns(3)
        recent_full_test = f1.checkbox("Recent full test done", value=True, key="v66_recent_full_test")
        no_critical_errors = f1.checkbox("No critical errors", value=True, key="v66_no_critical_errors")
        customer_mode_live = f1.checkbox("Customer Mode live", value=True, key="v66_customer_mode_live")
        real_upload_flow = f2.checkbox("Real upload flow live", value=bool(st.session_state.get("real_upload_v56_snapshot")), key="v66_real_upload_flow")
        privacy_safe_learning = f2.checkbox("Privacy-safe learning live", value=True, key="v66_privacy_safe_learning")
        payment_lock_ready = f2.checkbox("Payment/download gates ready", value=bool(st.session_state.get("checkout_v57_snapshot")), key="v66_payment_lock_ready")
        launch_gate_exists = f3.checkbox("Commercial launch gate exists", value=True, key="v66_launch_gate_exists")
        hardware_reference_planned = f3.checkbox("Hardware reference planned", value=True, key="v66_hardware_reference")
        terms_review_started = f3.checkbox("Terms/privacy review started", value=False, key="v66_terms_review")

    evidence_snapshots = {
        "V50 Customer Mode": bool(st.session_state.get("customer_mode_v50_snapshot")),
        "V51 Customer UI": bool(st.session_state.get("customer_ui_v51_snapshot")),
        "V52 Privacy Learning": bool(st.session_state.get("field_learning_v52_snapshot")),
        "V56 Real Upload": bool(st.session_state.get("real_upload_v56_snapshot")),
        "V57 Checkout": bool(st.session_state.get("checkout_v57_snapshot")),
        "V59 Hardware Reference": bool(st.session_state.get("hardware_reference_v59_snapshot")),
        "V60 Commercial Release": bool(st.session_state.get("commercial_release_v60_snapshot")),
        "V61 Traction Proof": bool(st.session_state.get("traction_proof_v61_snapshot")),
        "V62 ROI Value": bool(st.session_state.get("roi_value_v62_snapshot")),
        "V63 Case Study": bool(st.session_state.get("case_study_v63_snapshot")),
        "V64 Buyer Data Room": bool(st.session_state.get("buyer_dataroom_v64_snapshot")),
        "V65 IP & Moat": bool(st.session_state.get("ip_moat_v65_snapshot")),
    }

    quality_inputs = {
        "test_pass_rate": test_pass_rate,
        "open_bugs": open_bugs,
        "customer_friction": customer_friction,
        "ui_clarity": ui_clarity,
        "engine_confidence": engine_confidence,
        "cloud_readiness": cloud_readiness,
        "support_load_hours": support_load_hours,
        "field_evidence_strength": field_evidence_strength,
        "proof_strength": proof_strength,
        "privacy_confidence": privacy_confidence,
        "foundation_flags": {
            "recent_full_test": recent_full_test,
            "no_critical_errors": no_critical_errors,
            "customer_mode_live": customer_mode_live,
            "real_upload_flow": real_upload_flow,
            "privacy_safe_learning": privacy_safe_learning,
            "payment_lock_ready": payment_lock_ready,
            "launch_gate_exists": launch_gate_exists,
            "hardware_reference_planned": hardware_reference_planned,
            "terms_review_started": terms_review_started,
        },
    }

    if st.button("Build Continuous Improvement Plan V66", type="primary", use_container_width=True, key="v66_build_improvement_plan"):
        snapshot = core.build_continuous_improvement_flywheel_v66(
            project_name=st.session_state.project_name,
            release_target=release_target,
            quality_inputs=quality_inputs,
            evidence_snapshots=evidence_snapshots,
            customer_notes=customer_notes,
            founder_constraints=founder_constraints,
        )
        st.session_state.continuous_improvement_v66_snapshot = snapshot
        st.session_state.continuous_improvement_v66_bundle = core.create_continuous_improvement_v66_bundle(st.session_state.project_name, snapshot)
        st.success("Continuous improvement plan generated.")

    snapshot = st.session_state.get("continuous_improvement_v66_snapshot")
    if snapshot:
        decision = snapshot.get("decision", "")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ultimate product score", f"{snapshot.get('ultimate_product_score', 0)}%")
        k2.metric("Quality", f"{snapshot.get('quality_score', 0)}%")
        k3.metric("Foundation", f"{snapshot.get('foundation_score', 0)}%")
        k4.metric("Evidence", f"{snapshot.get('evidence_coverage', 0)}%")

        if "PUBLIC" in decision:
            st.success(decision)
        elif "CONTROLLED" in decision or "HARDEN" in decision:
            st.warning(decision)
        else:
            st.info(decision)
        st.write(snapshot.get("recommended_next_step", ""))

        tabs = st.tabs(["Backlog", "Customer simplicity", "Engine reliability", "Founder focus", "30-day sprint"])
        with tabs[0]:
            st.subheader("Prioritized improvement backlog")
            st.dataframe(pd.DataFrame(snapshot.get("improvement_backlog", [])), use_container_width=True)
            if snapshot.get("blockers"):
                st.subheader("Blockers")
                for item in snapshot.get("blockers", []):
                    st.warning(item)
        with tabs[1]:
            st.subheader("Customer simplification plan")
            for item in snapshot.get("customer_simplification_plan", []):
                st.success(item)
        with tabs[2]:
            st.subheader("Engine reliability plan")
            for item in snapshot.get("engine_reliability_plan", []):
                st.info(item)
        with tabs[3]:
            st.subheader("Founder automation plan")
            for item in snapshot.get("founder_automation_plan", []):
                st.write(f"- {item}")
            st.subheader("Do not do next")
            for item in snapshot.get("dont_do_next", []):
                st.error(item)
        with tabs[4]:
            st.subheader("30-day sprint")
            for item in snapshot.get("thirty_day_sprint", []):
                st.info(item)
            st.subheader("Success metrics")
            st.dataframe(pd.DataFrame(snapshot.get("success_metrics", [])), use_container_width=True)

        if st.session_state.get("continuous_improvement_v66_bundle"):
            st.download_button(
                "Download Continuous Improvement Bundle V66",
                st.session_state.continuous_improvement_v66_bundle,
                file_name=f"{st.session_state.project_name}_continuous_improvement_v66.zip",
                mime="application/zip",
                use_container_width=True,
                key="v66_download_improvement_bundle",
            )

    st.caption("V66 is an internal improvement and quality planning tool. It is not legal, valuation, certification, investment or acquisition advice.")




# ============================================================
# V67 — Reliability Calibration & Benchmarking Center
# ============================================================

def render_reliability_calibration_v67_tab():
    st.header("Reliability Calibration & Benchmarking V67")
    st.write(
        "V67 turns synthetic datasets, real uploads, privacy-safe learning and field evidence into a clearer reliability decision. "
        "It keeps customer-facing confidence honest: pilot-ready is not production-certified."
    )

    c1, c2 = st.columns([1.1, 1])
    with c1:
        calibration_target = st.selectbox(
            "Calibration target",
            ["Paid pilot", "Customer demo", "Hardware reference", "Public beta", "Enterprise/on-prem evaluation"],
            index=0,
            key="v67_calibration_target",
        )
        use_case_risk = st.selectbox(
            "Use-case risk level",
            ["Low", "Medium", "High", "Safety-critical / restricted"],
            index=1,
            key="v67_use_case_risk",
        )
        benchmark_notes = st.text_area(
            "Calibration notes / customer context",
            value="Use synthetic data for a fast start, real upload features for matching, and field validation before production claims.",
            height=105,
            key="v67_benchmark_notes",
        )
    with c2:
        real_sample_count = st.number_input("Real labelled samples available", min_value=0, max_value=100000, value=0, step=10, key="v67_real_samples")
        target_confidence = st.slider("Target confidence for pilot", 50, 99, 85, key="v67_target_confidence")
        false_alarm_tolerance = st.selectbox("False alarm tolerance", ["Low", "Medium", "High"], index=1, key="v67_false_alarm_tolerance")
        missing_cost = st.selectbox("Cost of missed detection", ["Low", "Medium", "High", "Severe"], index=1, key="v67_missing_cost")

    with st.expander("Evidence source status", expanded=True):
        e1, e2, e3 = st.columns(3)
        dataset_generated = e1.checkbox(
            "Pilot dataset generated",
            value=isinstance(st.session_state.get("dataset"), pd.DataFrame) and len(st.session_state.get("dataset")) > 0,
            key="v67_dataset_generated",
        )
        real_upload_done = e1.checkbox("Real upload inspected", value=bool(st.session_state.get("real_upload_v56_snapshot")), key="v67_real_upload_done")
        privacy_plan_done = e1.checkbox("Privacy plan exists", value=bool(st.session_state.get("field_learning_v52_snapshot")), key="v67_privacy_plan_done")
        synthetic_to_real_done = e2.checkbox("Synthetic-to-real bridge checked", value=bool(st.session_state.get("real_bridge_result")), key="v67_bridge_done")
        normality_checked = e2.checkbox("Normal/abnormal baseline checked", value=bool(st.session_state.get("normality_snapshot")), key="v67_normality_done")
        field_evidence_done = e2.checkbox("Field evidence or hardware reference", value=bool(st.session_state.get("field_evidence_v2_snapshot") or st.session_state.get("hardware_reference_v59_snapshot")), key="v67_field_evidence_done")
        trusted_claims = e3.checkbox("Safe claims/disclaimers active", value=True, key="v67_safe_claims")
        customer_acceptance_defined = e3.checkbox("Customer acceptance criteria defined", value=bool(st.session_state.get("proposal_sow_snapshot") or st.session_state.get("paid_pilot_v45_snapshot")), key="v67_acceptance_defined")
        full_test_recent = e3.checkbox("Recent full/smoke test passed", value=True, key="v67_full_test_recent")

    evidence_flags = {
        "dataset_generated": dataset_generated,
        "real_upload_done": real_upload_done,
        "privacy_plan_done": privacy_plan_done,
        "synthetic_to_real_done": synthetic_to_real_done,
        "normality_checked": normality_checked,
        "field_evidence_done": field_evidence_done,
        "trusted_claims": trusted_claims,
        "customer_acceptance_defined": customer_acceptance_defined,
        "full_test_recent": full_test_recent,
    }

    dataset_df = st.session_state.get("dataset") if isinstance(st.session_state.get("dataset"), pd.DataFrame) else pd.DataFrame()
    related_snapshots = {
        "fusion_doctor": st.session_state.get("fusion_doctor") or {},
        "trust_gate": st.session_state.get("trust_gate") or {},
        "real_upload_v56": st.session_state.get("real_upload_v56_snapshot") or {},
        "privacy_learning_v52": st.session_state.get("field_learning_v52_snapshot") or {},
        "field_evidence_v2": st.session_state.get("field_evidence_v2_snapshot") or {},
        "hardware_reference_v59": st.session_state.get("hardware_reference_v59_snapshot") or {},
        "proposal_sow": st.session_state.get("proposal_sow_snapshot") or {},
        "paid_pilot": st.session_state.get("paid_pilot_v45_snapshot") or {},
    }

    if st.button("Build Reliability Calibration Plan V67", type="primary", use_container_width=True, key="v67_build_calibration"):
        snapshot = core.build_reliability_calibration_v67(
            project_name=st.session_state.project_name,
            dataset_df=dataset_df,
            calibration_target=calibration_target,
            use_case_risk=use_case_risk,
            real_sample_count=int(real_sample_count),
            target_confidence=int(target_confidence),
            false_alarm_tolerance=false_alarm_tolerance,
            missing_cost=missing_cost,
            evidence_flags=evidence_flags,
            related_snapshots=related_snapshots,
            notes=benchmark_notes,
        )
        st.session_state.reliability_calibration_v67_snapshot = snapshot
        st.session_state.reliability_calibration_v67_bundle = core.create_reliability_calibration_v67_bundle(st.session_state.project_name, snapshot)
        st.success("Reliability calibration plan generated.")

    snapshot = st.session_state.get("reliability_calibration_v67_snapshot")
    if snapshot:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Calibration score", f"{snapshot.get('calibration_score', 0)}%")
        k2.metric("Evidence grade", snapshot.get("evidence_grade", "Unknown"))
        k3.metric("Data quality", f"{snapshot.get('data_quality_score', 0)}%")
        k4.metric("Decision", snapshot.get("decision", "Unknown"))

        decision = snapshot.get("decision", "")
        if "GO" in decision and "NO" not in decision:
            st.success(snapshot.get("recommended_next_step", ""))
        elif "NEEDS" in decision or "CONTROLLED" in decision:
            st.warning(snapshot.get("recommended_next_step", ""))
        else:
            st.error(snapshot.get("recommended_next_step", ""))

        tabs = st.tabs(["Benchmark", "Calibration plan", "Gaps", "Safe claims"])
        with tabs[0]:
            st.subheader("Reliability benchmark matrix")
            st.dataframe(pd.DataFrame(snapshot.get("benchmark_matrix", [])), use_container_width=True)
            st.subheader("Evidence components")
            st.dataframe(pd.DataFrame(snapshot.get("score_components", [])), use_container_width=True)
        with tabs[1]:
            st.subheader("Calibration actions")
            for item in snapshot.get("calibration_actions", []):
                st.info(item)
            st.subheader("Acceptance criteria")
            for item in snapshot.get("acceptance_criteria", []):
                st.write(f"- {item}")
        with tabs[2]:
            st.subheader("Proof gaps")
            for item in snapshot.get("proof_gaps", []):
                st.warning(item)
            st.subheader("Blocked claims until validated")
            for item in snapshot.get("blocked_claims", []):
                st.error(item)
        with tabs[3]:
            st.subheader("Safe customer-facing claims")
            for item in snapshot.get("safe_claims", []):
                st.success(item)
            st.subheader("Claims to avoid")
            for item in snapshot.get("claims_to_avoid", []):
                st.warning(item)

        if st.session_state.get("reliability_calibration_v67_bundle"):
            st.download_button(
                "Download Reliability Calibration Bundle V67",
                st.session_state.reliability_calibration_v67_bundle,
                file_name=f"{st.session_state.project_name}_reliability_calibration_v67.zip",
                mime="application/zip",
                use_container_width=True,
                key="v67_download_calibration_bundle",
            )

    st.caption("V67 improves reliability evidence and benchmarking. It is not a certification, legal opinion or guarantee of production accuracy.")



def render_automation_orchestrator_v68_tab():
    st.header("Automation Orchestrator V68")
    st.write(
        "This checks how much of the EdgeTwin customer journey is already automatic, what still needs founder approval, "
        "and which next action should run before a paid pilot or customer handoff."
    )

    c1, c2 = st.columns([1.15, 1])
    with c1:
        automation_goal = st.selectbox(
            "Automation goal",
            ["Controlled paid pilot", "Customer self-service demo", "Founder-assisted delivery", "Public SaaS preview", "Enterprise/on-prem evaluation"],
            key="v68_automation_goal",
        )
        notes = st.text_area(
            "Founder notes / constraints",
            value="Keep customer flow simple, automate checks and bundles, keep payment/privacy/final claims under founder approval.",
            height=90,
            key="v68_notes",
        )
    with c2:
        customer_self_service = st.checkbox("Allow customer self-service route", value=True, key="v68_customer_self_service")
        auto_generate_bundles = st.checkbox("Auto-generate draft bundles after checks", value=True, key="v68_auto_generate_bundles")
        auto_followup_queue = st.checkbox("Create automatic follow-up task queue", value=True, key="v68_auto_followup_queue")
        require_human_approval = st.checkbox("Require founder approval for payment/delivery/claims", value=True, key="v68_require_human_approval")
        allow_payment_without_review = st.checkbox("Allow payment without founder review", value=False, key="v68_allow_payment_without_review")
        allow_raw_data_global_learning = st.checkbox("Allow raw customer data for global learning", value=False, key="v68_allow_raw_data_global_learning")

    snapshots = {
        "auto_pilot": st.session_state.get("auto_pilot_result"),
        "customer_mode_v50": st.session_state.get("customer_mode_v50_snapshot"),
        "customer_ui_v51": st.session_state.get("customer_ui_v51_snapshot"),
        "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
        "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
        "reliability_calibration_v67": st.session_state.get("reliability_calibration_v67_snapshot"),
        "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
        "pricing_offer": st.session_state.get("pricing_offer_snapshot"),
        "roi_value_v62": st.session_state.get("roi_value_v62_snapshot"),
        "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
        "quote_to_cash": st.session_state.get("quote_to_cash_snapshot"),
        "customer_delivery": st.session_state.get("customer_delivery_snapshot"),
        "founder_ops_v49": st.session_state.get("founder_ops_v49_snapshot"),
        "customer_success": st.session_state.get("customer_success_snapshot"),
        "traction_proof_v61": st.session_state.get("traction_proof_v61_snapshot"),
        "case_study_v63": st.session_state.get("case_study_v63_snapshot"),
        "buyer_dataroom_v64": st.session_state.get("buyer_dataroom_v64_snapshot"),
        "ip_moat_v65": st.session_state.get("ip_moat_v65_snapshot"),
    }

    if st.button("Build Automation Orchestrator Plan V68", type="primary", use_container_width=True, key="v68_build_automation_plan"):
        snapshot = core.build_automation_orchestrator_v68(
            project_name=st.session_state.project_name,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            snapshots=snapshots,
            automation_goal=automation_goal,
            customer_self_service=customer_self_service,
            auto_generate_bundles=auto_generate_bundles,
            auto_followup_queue=auto_followup_queue,
            require_human_approval=require_human_approval,
            allow_payment_without_review=allow_payment_without_review,
            allow_raw_data_global_learning=allow_raw_data_global_learning,
            notes=notes,
        )
        st.session_state.automation_orchestrator_v68_snapshot = snapshot
        st.session_state.automation_orchestrator_v68_bundle = core.create_automation_orchestrator_v68_bundle(st.session_state.project_name, snapshot)
        st.success("Automation plan generated. Review the next-action queue and manual approval gates below.")

    snapshot = st.session_state.get("automation_orchestrator_v68_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Automation", f"{snapshot.get('automation_score')}%")
        m2.metric("Coverage", f"{snapshot.get('coverage_score')}%")
        m3.metric("Safety", f"{snapshot.get('safety_score')}%")
        m4.metric("Decision", snapshot.get("decision"))

        decision = str(snapshot.get("decision", ""))
        if "GO" in decision and "NO" not in decision:
            st.success(snapshot.get("next_best_action", ""))
        elif "PARTIAL" in decision:
            st.warning(snapshot.get("next_best_action", ""))
        else:
            st.error(snapshot.get("next_best_action", ""))

        tabs = st.tabs(["Next-action queue", "Automation map", "Human gates", "Customer simplicity"])
        with tabs[0]:
            st.subheader("Automatic next-action queue")
            st.dataframe(pd.DataFrame(snapshot.get("automation_queue", [])), use_container_width=True)
            st.caption("This is the practical do-next order. It reduces your manual thinking, but does not bypass founder approval for risky actions.")
        with tabs[1]:
            st.subheader("Connected automation components")
            st.dataframe(pd.DataFrame(snapshot.get("components", [])), use_container_width=True)
            st.subheader("Trigger map")
            st.dataframe(pd.DataFrame(snapshot.get("trigger_map", [])), use_container_width=True)
        with tabs[2]:
            st.subheader("Manual approval steps that should stay manual")
            for item in snapshot.get("manual_approval_steps", []):
                st.warning(item)
            st.subheader("Do not automate yet")
            for item in snapshot.get("do_not_automate_yet", []):
                st.error(item)
        with tabs[3]:
            st.subheader("Customer simplification rules")
            for item in snapshot.get("customer_simplification", []):
                st.success(item)
            st.subheader("Founder automation rules")
            for item in snapshot.get("founder_automation", []):
                st.info(item)

        if st.session_state.get("automation_orchestrator_v68_bundle"):
            st.download_button(
                "Download Automation Orchestrator Bundle V68",
                st.session_state.automation_orchestrator_v68_bundle,
                file_name=f"{st.session_state.project_name}_automation_orchestrator_v68.zip",
                mime="application/zip",
                use_container_width=True,
                key="v68_download_automation_bundle",
            )

    st.caption("V68 is assisted automation by design. Payment, privacy consent, paid delivery, legal/scope decisions and production-readiness claims still require human approval.")


# ============================================================
# V69 — Zero-Touch Customer Value Concierge
# ============================================================

def render_zero_touch_v69_tab():
    st.header("Zero-Touch Customer Value Concierge V69")
    st.write(
        "V69 is designed for your time problem and the customer's clarity problem: it turns the full EdgeTwin engine into one simple, guided route. "
        "The customer sees what they get, why it is worth paying for, what is ready, what is not ready, and what happens next."
    )

    c1, c2 = st.columns([1.1, 1])
    with c1:
        customer_goal = st.text_area(
            "Customer goal / problem",
            value="We want to know whether our sensor use-case is suitable for an Edge AI pilot, what data is needed, and what a realistic next step is.",
            height=95,
            key="v69_customer_goal",
        )
        offer_level = st.selectbox(
            "Offer level to guide toward",
            ["Starter Pilot", "Professional Pilot", "Real-Data Pilot", "Enterprise / Custom Pilot"],
            index=1,
            key="v69_offer_level",
        )
        customer_complexity = st.selectbox(
            "Customer complexity",
            ["Low", "Medium", "High", "Safety-critical / restricted"],
            index=1,
            key="v69_customer_complexity",
        )
        customer_data_status = st.selectbox(
            "Customer data status",
            ["No real data yet", "Example files available", "Labelled real data available", "Field validated data available"],
            index=1 if st.session_state.get("real_upload_v56_snapshot") else 0,
            key="v69_customer_data_status",
        )
    with c2:
        founder_time_budget = st.slider("Founder time budget per lead", 0, 240, 30, step=5, key="v69_founder_time_budget")
        allow_customer_self_service = st.checkbox("Allow customer self-service route", value=True, key="v69_allow_self_service")
        auto_prepare_outputs = st.checkbox("Auto-prepare draft outputs", value=True, key="v69_auto_prepare_outputs")
        require_approval_for_paid_steps = st.checkbox("Require founder approval before payment/delivery", value=True, key="v69_founder_approval")
        hide_advanced_details = st.checkbox("Hide advanced engine details from customer", value=True, key="v69_hide_advanced")
        show_value_for_money = st.checkbox("Show clear value-for-money explanation", value=True, key="v69_show_value")

    snapshots = {
        "customer_mode_v50": st.session_state.get("customer_mode_v50_snapshot"),
        "customer_ui_v51": st.session_state.get("customer_ui_v51_snapshot"),
        "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
        "launch_experience_v53": st.session_state.get("launch_experience_v53_snapshot"),
        "first_customer_beta_v55": st.session_state.get("first_customer_beta_v55_snapshot"),
        "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
        "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
        "commercial_release_v60": st.session_state.get("commercial_release_v60_snapshot"),
        "traction_proof_v61": st.session_state.get("traction_proof_v61_snapshot"),
        "roi_value_v62": st.session_state.get("roi_value_v62_snapshot"),
        "case_study_v63": st.session_state.get("case_study_v63_snapshot"),
        "buyer_dataroom_v64": st.session_state.get("buyer_dataroom_v64_snapshot"),
        "ip_moat_v65": st.session_state.get("ip_moat_v65_snapshot"),
        "continuous_improvement_v66": st.session_state.get("continuous_improvement_v66_snapshot"),
        "reliability_calibration_v67": st.session_state.get("reliability_calibration_v67_snapshot"),
        "automation_orchestrator_v68": st.session_state.get("automation_orchestrator_v68_snapshot"),
        "auto_pilot": st.session_state.get("auto_pilot_result"),
        "pricing_offer": st.session_state.get("pricing_offer_snapshot"),
        "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
        "customer_delivery": st.session_state.get("customer_delivery_snapshot"),
    }

    if st.button("Build Zero-Touch Value Route V69", type="primary", use_container_width=True, key="v69_build_zero_touch"):
        snapshot = core.build_zero_touch_customer_value_v69(
            project_name=st.session_state.project_name,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            snapshots=snapshots,
            customer_goal=customer_goal,
            offer_level=offer_level,
            customer_complexity=customer_complexity,
            customer_data_status=customer_data_status,
            founder_time_budget_minutes=founder_time_budget,
            allow_customer_self_service=allow_customer_self_service,
            auto_prepare_outputs=auto_prepare_outputs,
            require_approval_for_paid_steps=require_approval_for_paid_steps,
            hide_advanced_details=hide_advanced_details,
            show_value_for_money=show_value_for_money,
        )
        st.session_state.zero_touch_v69_snapshot = snapshot
        st.session_state.zero_touch_v69_bundle = core.create_zero_touch_customer_value_v69_bundle(st.session_state.project_name, snapshot)
        st.success("Zero-touch customer value route generated.")

    snapshot = st.session_state.get("zero_touch_v69_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Zero-touch", f"{snapshot.get('zero_touch_score')}%")
        m2.metric("Customer clarity", f"{snapshot.get('customer_clarity_score')}%")
        m3.metric("Value proof", f"{snapshot.get('value_proof_score')}%")
        m4.metric("Decision", snapshot.get("decision"))

        decision = str(snapshot.get("decision", ""))
        if "SELF-SERVICE" in decision:
            st.success(snapshot.get("recommended_next_step", ""))
        elif "ASSISTED" in decision or "CONTROLLED" in decision:
            st.warning(snapshot.get("recommended_next_step", ""))
        else:
            st.error(snapshot.get("recommended_next_step", ""))

        tabs = st.tabs(["Customer route", "Value for money", "Founder time shield", "Hidden engine", "Approval gates"])
        with tabs[0]:
            st.subheader("Simple customer route")
            for item in snapshot.get("customer_route", []):
                st.markdown(f"- **{item.get('step')}** — {item.get('customer_text')}")
            st.subheader("Customer-facing status badges")
            for badge in snapshot.get("customer_badges", []):
                st.markdown(f'<span class="v51-badge">{badge}</span>', unsafe_allow_html=True)
        with tabs[1]:
            st.subheader("What the customer gets")
            for item in snapshot.get("customer_value_items", []):
                st.success(item)
            st.subheader("Value-for-money explanation")
            st.write(snapshot.get("value_for_money_explanation", ""))
        with tabs[2]:
            st.subheader("Founder time shield")
            st.metric("Estimated founder time", f"{snapshot.get('estimated_founder_minutes')} min")
            for item in snapshot.get("founder_time_savers", []):
                st.info(item)
            st.subheader("Do not spend time on")
            for item in snapshot.get("do_not_spend_time_on", []):
                st.warning(item)
        with tabs[3]:
            st.subheader("Hidden engine checks")
            st.dataframe(pd.DataFrame(snapshot.get("hidden_engine_checks", [])), use_container_width=True)
            st.caption("These are kept behind the scenes so the customer gets clarity, not engineering overload.")
        with tabs[4]:
            st.subheader("Approval gates")
            for item in snapshot.get("approval_gates", []):
                st.error(item)
            st.subheader("Blocked promises")
            for item in snapshot.get("blocked_promises", []):
                st.warning(item)

        if st.session_state.get("zero_touch_v69_bundle"):
            st.download_button(
                "Download Zero-Touch Value Bundle V69",
                st.session_state.zero_touch_v69_bundle,
                file_name=f"{st.session_state.project_name}_zero_touch_value_v69.zip",
                mime="application/zip",
                use_container_width=True,
                key="v69_download_zero_touch_bundle",
            )

    st.caption("V69 reduces founder workload and customer confusion. It does not remove human review for paid delivery, privacy consent, legal/scope decisions or production-readiness claims.")



# ============================================================
# V70 — Outcome Assurance & Customer Success Autopilot
# ============================================================

def render_outcome_assurance_v70_tab():
    st.header("Outcome Assurance & Customer Success Autopilot V70")
    st.write(
        "V70 makes the customer delivery simple and clear: what they receive, why it matters, what is still not production-ready, "
        "and what follow-up can be automated so the founder does not become the support desk."
    )

    c1, c2 = st.columns([1.15, 1])
    with c1:
        delivery_stage = st.selectbox(
            "Delivery stage",
            ["Before payment", "After payment", "Ready for delivery", "Post-delivery follow-up"],
            index=2,
            key="v70_delivery_stage",
        )
        package_type = st.selectbox(
            "Package type",
            ["Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Pilot Bundle", "Enterprise / Custom Pilot"],
            index=1,
            key="v70_package_type",
        )
        customer_success_goal = st.text_area(
            "Customer success goal",
            value="Customer understands the pilot output, what is included, what is not included, and the next validation step.",
            height=95,
            key="v70_customer_success_goal",
        )
    with c2:
        support_level = st.selectbox(
            "Support model",
            ["Self-service with founder approval gates", "Light assisted support", "Founder manual support"],
            index=0,
            key="v70_support_level",
        )
        max_founder_minutes = st.slider("Max founder minutes per delivery", 5, 120, 20, 5, key="v70_founder_minutes")
        require_acceptance = st.checkbox("Require customer acceptance checklist", value=True, key="v70_require_acceptance")
        auto_prepare_followup = st.checkbox("Prepare automated follow-up sequence", value=True, key="v70_auto_followup")

    snapshots = {
        "auto_pilot": st.session_state.get("auto_pilot_result"),
        "fusion_doctor": st.session_state.get("fusion_doctor"),
        "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
        "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
        "reliability_calibration_v67": st.session_state.get("reliability_calibration_v67_snapshot"),
        "zero_touch_v69": st.session_state.get("zero_touch_v69_snapshot"),
        "automation_orchestrator_v68": st.session_state.get("automation_orchestrator_v68_snapshot"),
        "roi_value_v62": st.session_state.get("roi_value_v62_snapshot"),
        "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
        "quote_to_cash": st.session_state.get("quote_to_cash_snapshot"),
        "customer_delivery": st.session_state.get("customer_delivery_snapshot"),
        "case_study_v63": st.session_state.get("case_study_v63_snapshot"),
        "hardware_reference_v59": st.session_state.get("hardware_reference_v59_snapshot"),
        "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
    }

    if st.button("Build Outcome Assurance Plan V70", type="primary", use_container_width=True, key="v70_build_outcome_assurance"):
        snapshot = core.build_outcome_assurance_v70(
            project_name=st.session_state.project_name,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            snapshots=snapshots,
            delivery_stage=delivery_stage,
            package_type=package_type,
            customer_success_goal=customer_success_goal,
            support_level=support_level,
            require_acceptance=require_acceptance,
            auto_prepare_followup=auto_prepare_followup,
            max_founder_minutes=max_founder_minutes,
        )
        st.session_state.outcome_assurance_v70_snapshot = snapshot
        st.session_state.outcome_assurance_v70_bundle = core.create_outcome_assurance_v70_bundle(st.session_state.project_name, snapshot)
        st.success("Outcome Assurance V70 generated.")

    snapshot = st.session_state.get("outcome_assurance_v70_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Outcome assurance", f"{snapshot.get('outcome_assurance_score', 0)}%")
        m2.metric("Delivery clarity", f"{snapshot.get('delivery_clarity_score', 0)}%")
        m3.metric("Evidence", f"{snapshot.get('outcome_evidence_score', 0)}%")
        m4.metric("Founder time", f"{snapshot.get('estimated_founder_minutes', 0)} min")

        decision = snapshot.get("decision", "")
        if "GO" in decision and "HOLD" not in decision:
            st.success(f"Decision: {decision}")
        else:
            st.warning(f"Decision: {decision}")
        st.info(snapshot.get("recommended_next_step", ""))

        tabs = st.tabs(["Customer receives", "Acceptance", "FAQ", "Quality gates", "Founder shield"])
        with tabs[0]:
            st.subheader("What the customer receives")
            for item in snapshot.get("what_customer_receives", []):
                st.write(f"- {item}")
            st.subheader("Safe customer claims")
            for item in snapshot.get("safe_customer_claims", []):
                st.success(item)
        with tabs[1]:
            st.subheader("Acceptance criteria")
            for item in snapshot.get("acceptance_criteria", []):
                st.write(f"- {item}")
            st.subheader("Customer next steps")
            for item in snapshot.get("customer_next_steps", []):
                st.write(f"- {item}")
        with tabs[2]:
            st.subheader("Support FAQ")
            st.dataframe(pd.DataFrame(snapshot.get("support_faq", [])), use_container_width=True)
            if snapshot.get("automated_followup_sequence"):
                st.subheader("Automated follow-up sequence")
                st.dataframe(pd.DataFrame(snapshot.get("automated_followup_sequence", [])), use_container_width=True)
        with tabs[3]:
            st.subheader("Quality gates")
            st.dataframe(pd.DataFrame(snapshot.get("quality_gates", [])), use_container_width=True)
        with tabs[4]:
            st.subheader("Founder time shield")
            for item in snapshot.get("founder_do_not_touch", []):
                st.warning(item)
            st.subheader("Blocked promises")
            for item in snapshot.get("blocked_promises", []):
                st.error(item)

        if st.session_state.get("outcome_assurance_v70_bundle"):
            st.download_button(
                "Download Outcome Assurance Bundle V70",
                st.session_state.outcome_assurance_v70_bundle,
                file_name=f"{st.session_state.project_name}_outcome_assurance_v70.zip",
                mime="application/zip",
                use_container_width=True,
                key="v70_download_outcome_assurance_bundle",
            )

    st.caption("V70 keeps the customer experience simple while protecting delivery quality, founder time, privacy boundaries and honest pilot-only claims.")



# ============================================================
# V71 — Customer Support Autopilot / Self-Service Deflection Center
# ============================================================

def render_customer_support_v71_tab():
    st.header("Customer Support Autopilot V71")
    st.write(
        "Turns common customer questions into controlled self-service answers, clear next steps and safe escalation gates. "
        "Goal: customers understand what they paid for, while founder time stays protected."
    )

    c1, c2 = st.columns(2)
    with c1:
        customer_stage = st.selectbox(
            "Customer stage",
            [
                "Before pilot generation",
                "After pilot generation",
                "After readiness review",
                "After proposal / quote",
                "After payment / checkout readiness",
                "After delivery / handoff",
            ],
            index=5,
            key="v71_customer_stage",
        )
        issue_category = st.selectbox(
            "Question / support category",
            [
                "General: what do I get?",
                "Next step confusion",
                "Upload/data question",
                "Report/readiness question",
                "Pricing/checkout question",
                "Privacy/data-use question",
                "Technical bug or missing output",
                "Production accuracy/safety question",
            ],
            index=0,
            key="v71_issue_category",
        )
        customer_question = st.text_area(
            "Customer question",
            value="What did I receive, what should I do next, and what is included in the pilot package?",
            height=110,
            key="v71_customer_question",
        )
    with c2:
        urgency = st.selectbox("Urgency", ["Low", "Normal", "High", "Urgent / customer blocked"], index=1, key="v71_urgency")
        customer_confidence = st.slider("Customer confidence / understanding", 0, 10, 7, 1, key="v71_customer_confidence")
        self_service_enabled = st.checkbox("Allow self-service answer first", value=True, key="v71_self_service_enabled")
        customer_has_downloaded = st.checkbox("Customer has downloaded/reviewed bundle", value=False, key="v71_customer_has_downloaded")
        customer_has_uploaded_data = st.checkbox("Customer has uploaded real data", value=bool(st.session_state.get("real_upload_v56_snapshot")), key="v71_customer_has_uploaded")
        max_founder_minutes = st.slider("Max founder minutes allowed", 0, 120, 10, 5, key="v71_max_founder_minutes")

    snapshots = {
        "outcome_assurance_v70": st.session_state.get("outcome_assurance_v70_snapshot"),
        "zero_touch_v69": st.session_state.get("zero_touch_v69_snapshot"),
        "automation_orchestrator_v68": st.session_state.get("automation_orchestrator_v68_snapshot"),
        "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
        "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
        "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
        "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
        "customer_delivery": st.session_state.get("customer_delivery_snapshot"),
    }

    if st.button("Build Support Autopilot Plan V71", type="primary", use_container_width=True, key="v71_build_support_autopilot"):
        snapshot = core.build_customer_support_autopilot_v71(
            project_name=st.session_state.project_name,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            snapshots=snapshots,
            customer_stage=customer_stage,
            issue_category=issue_category,
            customer_question=customer_question,
            customer_confidence=customer_confidence,
            urgency=urgency,
            self_service_enabled=self_service_enabled,
            max_founder_minutes=max_founder_minutes,
            customer_has_downloaded=customer_has_downloaded,
            customer_has_uploaded_data=customer_has_uploaded_data,
        )
        st.session_state.customer_support_v71_snapshot = snapshot
        st.session_state.customer_support_v71_bundle = core.create_customer_support_autopilot_v71_bundle(st.session_state.project_name, snapshot)
        st.success("Customer Support Autopilot V71 generated.")

    snapshot = st.session_state.get("customer_support_v71_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Support deflection", f"{snapshot.get('support_deflection_score', 0)}%")
        m2.metric("Customer clarity", f"{snapshot.get('customer_clarity_score', 0)}%")
        m3.metric("Escalation risk", f"{snapshot.get('escalation_risk_score', 0)}%")
        m4.metric("Founder time", f"{snapshot.get('estimated_founder_minutes', 0)} min")

        decision = snapshot.get("decision", "")
        if "RESOLVED" in decision:
            st.success(f"Decision: {decision}")
        elif "APPROVAL" in decision:
            st.warning(f"Decision: {decision}")
        else:
            st.error(f"Decision: {decision}")

        st.info(snapshot.get("recommended_next_step", ""))
        st.markdown("#### Controlled customer answer")
        st.write(snapshot.get("customer_safe_answer", ""))

        tabs = st.tabs(["Value answer", "FAQ", "Support queue", "Escalation gates", "Founder shield"])
        with tabs[0]:
            st.subheader("Why this is worth paying for")
            for item in snapshot.get("value_for_money_explanation", []):
                st.write(f"- {item}")
            st.subheader("Customer next steps")
            for item in snapshot.get("next_steps", []):
                st.write(f"- {item}")
            st.subheader("Required inputs")
            for item in snapshot.get("required_inputs", []):
                st.write(f"- {item}")
        with tabs[1]:
            st.dataframe(pd.DataFrame(snapshot.get("faq_articles", [])), use_container_width=True)
            st.subheader("Auto-reply sequence")
            st.dataframe(pd.DataFrame(snapshot.get("auto_reply_sequence", [])), use_container_width=True)
        with tabs[2]:
            st.dataframe(pd.DataFrame(snapshot.get("support_queue", [])), use_container_width=True)
            st.subheader("Support metrics")
            st.dataframe(pd.DataFrame(snapshot.get("support_metrics", [])), use_container_width=True)
        with tabs[3]:
            for item in snapshot.get("escalation_conditions", []):
                st.warning(item)
            st.subheader("Blocked actions")
            for item in snapshot.get("blocked_actions", []):
                st.error(item)
        with tabs[4]:
            st.subheader("Founder do-not-touch list")
            for item in snapshot.get("founder_do_not_touch", []):
                st.warning(item)
            st.subheader("Safe claims")
            for item in snapshot.get("safe_claims", []):
                st.success(item)
            st.subheader("Claims to avoid")
            for item in snapshot.get("claims_to_avoid", []):
                st.error(item)

        if st.session_state.get("customer_support_v71_bundle"):
            st.download_button(
                "Download Customer Support Autopilot Bundle V71",
                st.session_state.customer_support_v71_bundle,
                file_name=f"{st.session_state.project_name}_customer_support_v71.zip",
                mime="application/zip",
                use_container_width=True,
                key="v71_download_customer_support_bundle",
            )

    st.caption("V71 reduces founder support workload with controlled answers, customer-safe FAQs and escalation gates. It does not automate legal, privacy, payment or production-safety decisions.")



# ============================================================
# V72 — Customer Status Portal / Project State & Next-Step Tracker
# ============================================================

def render_customer_status_v72_tab():
    st.header("Customer Status Portal V72")
    st.write(
        "V72 gives the customer one calm status screen: what is ready, what is blocked, what they need to do next, "
        "what EdgeTwin is preparing, and when founder approval is required. This reduces repeated support questions."
    )

    c1, c2 = st.columns(2)
    with c1:
        project_stage = st.selectbox(
            "Project stage",
            [
                "Intake",
                "Pilot generation",
                "Readiness review",
                "Proposal / SOW",
                "Checkout / payment readiness",
                "Delivery / handoff",
                "Post-delivery follow-up",
            ],
            index=5,
            key="v72_project_stage",
        )
        customer_plan = st.selectbox(
            "Customer package",
            ["Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Pilot Bundle", "Enterprise / Custom Pilot"],
            index=1,
            key="v72_customer_plan",
        )
        customer_question = st.text_area(
            "Main customer status question",
            value="Where are we now, what is ready, what do I need to do next, and what did I pay for?",
            height=105,
            key="v72_customer_question",
        )
    with c2:
        payment_status = st.selectbox(
            "Payment / commercial status",
            ["Not requested", "Quote ready", "Waiting for customer approval", "Payment/invoice ready", "Paid / approved", "Manual review needed"],
            index=1,
            key="v72_payment_status",
        )
        privacy_status = st.selectbox(
            "Privacy/data status",
            ["Private only", "Feature learning opt-in", "Raw data permission requested", "Missing privacy decision"],
            index=0,
            key="v72_privacy_status",
        )
        next_milestone = st.text_input(
            "Next visible milestone",
            value="Review pilot readiness and confirm the next validation step.",
            key="v72_next_milestone",
        )
        customer_inputs_complete = st.checkbox("Customer inputs complete", value=False, key="v72_customer_inputs_complete")
        delivery_bundle_ready = st.checkbox("Delivery bundle ready", value=bool(st.session_state.get("customer_delivery_bundle") or st.session_state.get("outcome_assurance_v70_bundle")), key="v72_delivery_bundle_ready")
        founder_review_required = st.checkbox("Founder approval required before next step", value=True, key="v72_founder_review_required")
        support_self_service_ready = st.checkbox("Self-service support answers ready", value=bool(st.session_state.get("customer_support_v71_snapshot")), key="v72_support_ready")

    snapshots = {
        "zero_touch_v69": st.session_state.get("zero_touch_v69_snapshot"),
        "outcome_assurance_v70": st.session_state.get("outcome_assurance_v70_snapshot"),
        "customer_support_v71": st.session_state.get("customer_support_v71_snapshot"),
        "reliability_calibration_v67": st.session_state.get("reliability_calibration_v67_snapshot"),
        "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
        "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
        "roi_value_v62": st.session_state.get("roi_value_v62_snapshot"),
        "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
        "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
        "customer_delivery": st.session_state.get("customer_delivery_snapshot"),
        "automation_orchestrator_v68": st.session_state.get("automation_orchestrator_v68_snapshot"),
    }

    dataset_df = st.session_state.dataset if isinstance(st.session_state.get("dataset"), pd.DataFrame) else pd.DataFrame()

    if st.button("Build Customer Status Portal V72", type="primary", use_container_width=True, key="v72_build_customer_status"):
        snapshot = core.build_customer_status_portal_v72(
            project_name=st.session_state.project_name,
            dataset_df=dataset_df,
            snapshots=snapshots,
            project_stage=project_stage,
            customer_plan=customer_plan,
            customer_question=customer_question,
            payment_status=payment_status,
            privacy_status=privacy_status,
            next_milestone=next_milestone,
            customer_inputs_complete=customer_inputs_complete,
            delivery_bundle_ready=delivery_bundle_ready,
            founder_review_required=founder_review_required,
            support_self_service_ready=support_self_service_ready,
        )
        st.session_state.customer_status_v72_snapshot = snapshot
        st.session_state.customer_status_v72_bundle = core.create_customer_status_portal_v72_bundle(st.session_state.project_name, snapshot)
        st.success("Customer Status Portal V72 generated.")

    snapshot = st.session_state.get("customer_status_v72_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Status clarity", f"{snapshot.get('status_clarity_score', 0)}%")
        m2.metric("Customer confidence", f"{snapshot.get('customer_confidence_score', 0)}%")
        m3.metric("Founder time risk", snapshot.get("founder_time_risk", "Unknown"))
        m4.metric("Decision", snapshot.get("decision", "Unknown"))

        decision = snapshot.get("decision", "")
        if "READY" in decision or "CLEAR" in decision:
            st.success(snapshot.get("recommended_next_step", ""))
        elif "INPUT" in decision or "REVIEW" in decision:
            st.warning(snapshot.get("recommended_next_step", ""))
        else:
            st.error(snapshot.get("recommended_next_step", ""))

        st.markdown("#### Customer-safe status summary")
        st.info(snapshot.get("customer_visible_summary", ""))
        for badge in snapshot.get("status_badges", []):
            st.markdown(f'<span class="v51-badge">{badge}</span>', unsafe_allow_html=True)

        tabs = st.tabs(["One-screen status", "Timeline", "Blockers", "Customer actions", "Founder shield"])
        with tabs[0]:
            st.dataframe(pd.DataFrame(snapshot.get("status_cards", [])), use_container_width=True)
            st.subheader("Value delivered")
            for item in snapshot.get("value_delivered", []):
                st.write(f"- {item}")
        with tabs[1]:
            st.dataframe(pd.DataFrame(snapshot.get("timeline", [])), use_container_width=True)
        with tabs[2]:
            for item in snapshot.get("blockers", []):
                st.warning(item)
            st.subheader("Blocked promises")
            for item in snapshot.get("blocked_promises", []):
                st.error(item)
        with tabs[3]:
            st.subheader("Customer next actions")
            for item in snapshot.get("customer_next_actions", []):
                st.write(f"- {item}")
            st.subheader("Required inputs")
            for item in snapshot.get("required_customer_inputs", []):
                st.write(f"- {item}")
        with tabs[4]:
            st.subheader("Founder actions")
            for item in snapshot.get("founder_actions", []):
                st.info(item)
            st.subheader("Do not handle manually")
            for item in snapshot.get("do_not_handle_manually", []):
                st.warning(item)
            st.subheader("Safe claims")
            for item in snapshot.get("safe_claims", []):
                st.success(item)

        if st.session_state.get("customer_status_v72_bundle"):
            st.download_button(
                "Download Customer Status Portal Bundle V72",
                st.session_state.customer_status_v72_bundle,
                file_name=f"{st.session_state.project_name}_customer_status_v72.zip",
                mime="application/zip",
                use_container_width=True,
                key="v72_download_customer_status_bundle",
            )

    st.caption("V72 is a customer-facing status portal. It keeps the project simple and transparent without promising production performance, legal compliance or fully automated paid delivery.")

# ============================================================
# V73 — Unified Customer Journey / One-Page Value Cockpit
# ============================================================

def render_customer_journey_v73_tab():
    st.header("Unified Customer Journey V73")
    st.write(
        "V73 turns EdgeTwin into one calm customer cockpit: value, status, readiness, missing inputs, next step and escalation gates. "
        "The customer sees a simple path, while the deeper OMEGA-X engine checks stay behind the scenes."
    )

    snapshots = {
        "zero_touch_v69": st.session_state.get("zero_touch_v69_snapshot"),
        "outcome_assurance_v70": st.session_state.get("outcome_assurance_v70_snapshot"),
        "customer_support_v71": st.session_state.get("customer_support_v71_snapshot"),
        "customer_status_v72": st.session_state.get("customer_status_v72_snapshot"),
        "reliability_calibration_v67": st.session_state.get("reliability_calibration_v67_snapshot"),
        "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
        "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
        "roi_value_v62": st.session_state.get("roi_value_v62_snapshot"),
        "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
        "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
        "delivery": st.session_state.get("customer_delivery_snapshot"),
    }

    c1, c2, c3 = st.columns(3)
    with c1:
        use_case = st.text_input("Customer use-case", value="Predictive maintenance / acoustic-vibration pilot", key="v73_use_case")
        customer_stage = st.selectbox(
            "Customer stage",
            ["Just exploring", "Pilot dataset needed", "Readiness review", "Proposal requested", "Paid pilot / delivery", "Post-delivery follow-up"],
            index=2,
            key="v73_customer_stage",
        )
    with c2:
        customer_data_status = st.selectbox(
            "Customer data status",
            ["No data yet", "Synthetic pilot only", "Some real WAV/CSV samples", "Real data uploaded and inspected", "Field/hardware evidence available"],
            index=1,
            key="v73_customer_data_status",
        )
        commercial_status = st.selectbox(
            "Commercial status",
            ["Free demo", "Qualified lead", "Proposal requested", "Quote/SOW ready", "Payment pending", "Paid / approved"],
            index=1,
            key="v73_commercial_status",
        )
    with c3:
        privacy_status = st.selectbox(
            "Privacy mode",
            ["Private only", "Feature learning allowed", "Raw data permission requested", "Missing privacy decision"],
            index=0,
            key="v73_privacy_status",
        )
        founder_touch_target = st.slider("Founder time target per customer (minutes)", 0, 120, 15, 5, key="v73_founder_touch_target")

    st.markdown("---")
    if st.button("Build Unified Customer Journey V73", type="primary", use_container_width=True, key="v73_build_customer_journey"):
        snapshot = core.build_unified_customer_journey_v73(
            project_name=st.session_state.project_name,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            snapshots=snapshots,
            use_case=use_case,
            customer_stage=customer_stage,
            customer_data_status=customer_data_status,
            commercial_status=commercial_status,
            privacy_status=privacy_status,
            founder_touch_target_minutes=int(founder_touch_target),
        )
        st.session_state.customer_journey_v73_snapshot = snapshot
        st.session_state.customer_journey_v73_bundle = core.create_unified_customer_journey_v73_bundle(st.session_state.project_name, snapshot)
        st.success("Unified Customer Journey V73 generated.")

    snapshot = st.session_state.get("customer_journey_v73_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Journey clarity", f"{snapshot.get('journey_clarity_score', 0)}%")
        m2.metric("Value clarity", f"{snapshot.get('value_clarity_score', 0)}%")
        m3.metric("Trust gate", f"{snapshot.get('trust_gate_score', 0)}%")
        m4.metric("Founder time risk", snapshot.get("founder_time_risk", "Unknown"))

        if snapshot.get("decision") == "ZERO-TOUCH CUSTOMER GO":
            st.success(snapshot.get("recommended_next_step"))
        elif snapshot.get("decision") in ["ASSISTED CUSTOMER GO", "CUSTOMER INPUT NEEDED"]:
            st.warning(snapshot.get("recommended_next_step"))
        else:
            st.error(snapshot.get("recommended_next_step"))

        st.markdown("### Customer-visible cockpit")
        st.info(snapshot.get("customer_one_page_summary", ""))

        badges = snapshot.get("status_badges", [])
        if badges:
            st.markdown(" ".join([f'<span class="v51-badge">{b}</span>' for b in badges]), unsafe_allow_html=True)

        tabs = st.tabs(["Journey", "Value", "Next actions", "Hidden engine checks", "Founder shield"])
        with tabs[0]:
            journey_df = pd.DataFrame(snapshot.get("journey_steps", []))
            if len(journey_df) > 0:
                st.dataframe(journey_df, use_container_width=True)
            st.subheader("Customer stage summary")
            for item in snapshot.get("customer_visible_status", []):
                st.write(f"- {item}")
        with tabs[1]:
            st.subheader("What the customer pays for")
            for item in snapshot.get("value_for_money_points", []):
                st.success(item)
            st.subheader("Safe value claims")
            for item in snapshot.get("safe_value_claims", []):
                st.info(item)
        with tabs[2]:
            st.subheader("Customer next actions")
            for item in snapshot.get("customer_next_actions", []):
                st.write(f"- {item}")
            st.subheader("Missing inputs")
            for item in snapshot.get("missing_inputs", []):
                st.warning(item)
        with tabs[3]:
            st.subheader("Engine checks running behind the simple route")
            checks_df = pd.DataFrame(snapshot.get("hidden_engine_checks", []))
            if len(checks_df) > 0:
                st.dataframe(checks_df, use_container_width=True)
            st.caption("Customers do not need to understand every internal module. They need the result, the evidence level and the next step.")
        with tabs[4]:
            st.subheader("Founder time shield")
            for item in snapshot.get("founder_time_shield", []):
                st.info(item)
            st.subheader("Escalation gates")
            for item in snapshot.get("escalation_gates", []):
                st.warning(item)
            st.subheader("Blocked promises")
            for item in snapshot.get("blocked_promises", []):
                st.error(item)

        if st.session_state.get("customer_journey_v73_bundle"):
            st.download_button(
                "Download Unified Customer Journey Bundle V73",
                st.session_state.customer_journey_v73_bundle,
                file_name=f"{st.session_state.project_name}_customer_journey_v73.zip",
                mime="application/zip",
                use_container_width=True,
                key="v73_download_customer_journey_bundle",
            )

    st.caption("V73 is a customer simplicity layer. It does not remove the deeper engine; it hides complexity and keeps founder approval gates for payment, privacy, production claims and delivery release.")


# ============================================================
# V74 — End-to-End Quality Guardian / Self-Test & Failure Recovery
# ============================================================

def render_quality_guardian_v74_tab():
    st.header("End-to-End Quality Guardian V74")
    st.write(
        "V74 is the reliability layer for the whole product: it checks the critical customer/founder route, "
        "looks for missing evidence/gates, and creates a safe recovery plan before a customer sees a broken or overpromised flow."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        launch_target = st.selectbox(
            "Launch target",
            ["Private beta", "Paid pilot launch", "Public SaaS preview", "Enterprise/on-prem evaluation"],
            index=1,
            key="v74_launch_target",
        )
    with c2:
        automation_level = st.selectbox(
            "Automation level",
            ["Assisted automation", "Zero-touch customer route", "Founder-reviewed delivery", "Public self-service"],
            index=1,
            key="v74_automation_level",
        )
    with c3:
        test_depth = st.selectbox(
            "Self-test depth",
            ["Core gates", "Full customer journey", "Commercial + delivery", "Launch-critical"],
            index=3,
            key="v74_test_depth",
        )

    st.caption("V74 does not promise production accuracy. It protects the pilot route, payment/delivery gates, privacy defaults and support/founder-time boundaries.")

    snapshots = {
        "customer_journey_v73": st.session_state.get("customer_journey_v73_snapshot"),
        "customer_status_v72": st.session_state.get("customer_status_v72_snapshot"),
        "customer_support_v71": st.session_state.get("customer_support_v71_snapshot"),
        "outcome_assurance_v70": st.session_state.get("outcome_assurance_v70_snapshot"),
        "zero_touch_v69": st.session_state.get("zero_touch_v69_snapshot"),
        "automation_orchestrator_v68": st.session_state.get("automation_orchestrator_v68_snapshot"),
        "reliability_calibration_v67": st.session_state.get("reliability_calibration_v67_snapshot"),
        "continuous_improvement_v66": st.session_state.get("continuous_improvement_v66_snapshot"),
        "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
        "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
        "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
        "commercial_release_v60": st.session_state.get("commercial_release_v60_snapshot"),
        "launch_stabilization_v60_1": st.session_state.get("launch_stabilization_v60_1_snapshot"),
        "pricing_offer": st.session_state.get("pricing_offer_snapshot"),
        "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
        "customer_delivery": st.session_state.get("customer_delivery_snapshot"),
        "cloud_architecture_v58": st.session_state.get("cloud_architecture_v58_snapshot"),
        "hardware_reference_v59": st.session_state.get("hardware_reference_v59_snapshot"),
    }
    bundle_presence = {
        "auto_pilot_bundle": st.session_state.get("auto_pilot_bundle") is not None,
        "fusion_bundle": st.session_state.get("fusion_bundle") is not None,
        "enterprise_bundle": st.session_state.get("enterprise_bundle") is not None,
        "customer_journey_v73_bundle": st.session_state.get("customer_journey_v73_bundle") is not None,
        "customer_status_v72_bundle": st.session_state.get("customer_status_v72_bundle") is not None,
        "outcome_assurance_v70_bundle": st.session_state.get("outcome_assurance_v70_bundle") is not None,
        "customer_support_v71_bundle": st.session_state.get("customer_support_v71_bundle") is not None,
        "checkout_v57_bundle": st.session_state.get("checkout_v57_bundle") is not None,
        "proposal_sow_bundle": st.session_state.get("proposal_sow_bundle") is not None,
        "pricing_offer_bundle": st.session_state.get("pricing_offer_bundle") is not None,
    }
    expected_pages = list(_PAGE_RENDERERS.keys()) if "_PAGE_RENDERERS" in globals() else []

    if st.button("Run End-to-End Quality Guardian V74", type="primary", use_container_width=True, key="v74_run_quality_guardian"):
        snapshot = core.build_end_to_end_quality_guardian_v74(
            project_name=st.session_state.project_name,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            snapshots=snapshots,
            bundle_presence=bundle_presence,
            launch_target=launch_target,
            automation_level=automation_level,
            test_depth=test_depth,
            expected_pages=expected_pages,
            workspace_mode=st.session_state.workspace_mode_v50,
        )
        st.session_state.quality_guardian_v74_snapshot = snapshot
        st.session_state.quality_guardian_v74_bundle = core.create_quality_guardian_v74_bundle(st.session_state.project_name, snapshot)
        st.success("Quality Guardian V74 self-test generated.")

    snapshot = st.session_state.get("quality_guardian_v74_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Quality score", f"{snapshot.get('quality_score', 0)}%")
        m2.metric("Route integrity", f"{snapshot.get('route_integrity_score', 0)}%")
        m3.metric("Safety gates", f"{snapshot.get('safety_gate_score', 0)}%")
        m4.metric("Decision", snapshot.get("decision", "Unknown"))

        if snapshot.get("decision") == "SELF-TEST PASS":
            st.success(snapshot.get("recommended_next_step", "Continue."))
        elif snapshot.get("decision") == "CONDITIONAL PASS":
            st.warning(snapshot.get("recommended_next_step", "Fix the listed items before a broader launch."))
        else:
            st.error(snapshot.get("recommended_next_step", "Do not expose this route externally yet."))

        st.markdown("#### System status badges")
        st.markdown(" ".join([f'<span class="v51-badge">{badge}</span>' for badge in snapshot.get("status_badges", [])]), unsafe_allow_html=True)

        checks_df = pd.DataFrame(snapshot.get("quality_checks", []))
        if len(checks_df) > 0:
            st.subheader("Quality checks")
            st.dataframe(checks_df, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Must fix first")
            for item in snapshot.get("must_fix_first", []):
                st.warning(item)
            st.subheader("Auto-recovery actions")
            for item in snapshot.get("auto_recovery_actions", []):
                st.write(f"- {item}")
        with col_b:
            st.subheader("Do not expose yet")
            for item in snapshot.get("do_not_expose_yet", []):
                st.error(item)
            st.subheader("Founder approval gates")
            for item in snapshot.get("founder_approval_gates", []):
                st.write(f"- {item}")

        with st.expander("Customer-safe launch copy", expanded=False):
            st.write(snapshot.get("customer_safe_copy", ""))
            st.markdown("**Safe claims**")
            for item in snapshot.get("safe_claims", []):
                st.write(f"- {item}")
            st.markdown("**Blocked promises**")
            for item in snapshot.get("blocked_promises", []):
                st.write(f"- {item}")

        with st.expander("Failure recovery playbook", expanded=False):
            for step in snapshot.get("failure_recovery_playbook", []):
                st.write(f"- {step}")

        if st.session_state.get("quality_guardian_v74_bundle"):
            st.download_button(
                "Download Quality Guardian Bundle V74",
                st.session_state.quality_guardian_v74_bundle,
                file_name=f"{st.session_state.project_name}_quality_guardian_v74.zip",
                mime="application/zip",
                use_container_width=True,
                key="v74_download_quality_guardian_bundle",
            )
    else:
        st.info("Run the Quality Guardian. Use it before broad demos, paid pilot delivery or any public launch attempt.")

    st.caption("V74 is a self-test and recovery planning layer. It improves reliability, but it is not a substitute for production monitoring, external security/legal review, field validation or cloud hardening.")



# ============================================================
# V75 — Deliverable QA & Value Lock / Customer Output Assurance
# ============================================================

def render_deliverable_qa_v75_tab():
    st.header("Deliverable QA & Value Lock V75")
    st.write(
        "V75 is the last quality gate before a customer receives a bundle, report or paid handoff. "
        "It checks if the output is clear, useful, safe, privacy-aware and worth the money — and blocks weak delivery when evidence is missing."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        package_type = st.selectbox(
            "Package type",
            ["Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Pilot Bundle", "Enterprise / Custom Pilot"],
            index=1,
            key="v75_package_type",
        )
    with c2:
        delivery_stage = st.selectbox(
            "Delivery stage",
            ["Pre-demo", "Pre-proposal", "Pre-payment", "Pre-delivery", "Post-delivery review"],
            index=3,
            key="v75_delivery_stage",
        )
    with c3:
        customer_visibility = st.selectbox(
            "Customer visibility",
            ["Customer summary only", "Customer + founder details", "Founder review only"],
            index=0,
            key="v75_customer_visibility",
        )

    snapshots = {
        "quality_guardian_v74": st.session_state.get("quality_guardian_v74_snapshot"),
        "customer_journey_v73": st.session_state.get("customer_journey_v73_snapshot"),
        "customer_status_v72": st.session_state.get("customer_status_v72_snapshot"),
        "customer_support_v71": st.session_state.get("customer_support_v71_snapshot"),
        "outcome_assurance_v70": st.session_state.get("outcome_assurance_v70_snapshot"),
        "zero_touch_v69": st.session_state.get("zero_touch_v69_snapshot"),
        "automation_orchestrator_v68": st.session_state.get("automation_orchestrator_v68_snapshot"),
        "reliability_calibration_v67": st.session_state.get("reliability_calibration_v67_snapshot"),
        "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
        "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
        "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
        "pricing_offer": st.session_state.get("pricing_offer_snapshot"),
        "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
        "delivery": st.session_state.get("customer_delivery_snapshot"),
        "roi_value_v62": st.session_state.get("roi_value_v62_snapshot"),
    }
    bundle_presence = {
        "auto_pilot_bundle": st.session_state.get("auto_pilot_bundle") is not None,
        "fusion_bundle": st.session_state.get("fusion_bundle") is not None,
        "enterprise_bundle": st.session_state.get("enterprise_bundle") is not None,
        "professional_report_bundle": st.session_state.get("professional_report_bundle") is not None,
        "proposal_sow_bundle": st.session_state.get("proposal_sow_bundle") is not None,
        "checkout_v57_bundle": st.session_state.get("checkout_v57_bundle") is not None,
        "customer_status_v72_bundle": st.session_state.get("customer_status_v72_bundle") is not None,
        "outcome_assurance_v70_bundle": st.session_state.get("outcome_assurance_v70_bundle") is not None,
        "quality_guardian_v74_bundle": st.session_state.get("quality_guardian_v74_bundle") is not None,
    }

    if st.button("Run Deliverable QA & Value Lock V75", type="primary", use_container_width=True, key="v75_run_deliverable_qa"):
        snapshot = core.build_deliverable_qa_value_lock_v75(
            project_name=st.session_state.project_name,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            snapshots=snapshots,
            bundle_presence=bundle_presence,
            package_type=package_type,
            delivery_stage=delivery_stage,
            customer_visibility=customer_visibility,
        )
        st.session_state.deliverable_qa_v75_snapshot = snapshot
        st.session_state.deliverable_qa_v75_bundle = core.create_deliverable_qa_v75_bundle(st.session_state.project_name, snapshot)
        st.success("Deliverable QA V75 generated.")

    snapshot = st.session_state.get("deliverable_qa_v75_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Delivery QA", f"{snapshot.get('delivery_quality_score', 0)}%")
        m2.metric("Value clarity", f"{snapshot.get('value_clarity_score', 0)}%")
        m3.metric("Evidence", f"{snapshot.get('evidence_score', 0)}%")
        m4.metric("Decision", snapshot.get("decision", "Unknown"))

        if snapshot.get("decision") == "DELIVERY APPROVED":
            st.success(snapshot.get("recommended_next_step", "Delivery can proceed."))
        elif snapshot.get("decision") == "FOUNDER REVIEW":
            st.warning(snapshot.get("recommended_next_step", "Founder review is needed."))
        else:
            st.error(snapshot.get("recommended_next_step", "Block delivery until gaps are fixed."))

        st.markdown("#### Customer-safe value summary")
        st.markdown(" ".join([f'<span class="v51-badge">{badge}</span>' for badge in snapshot.get("status_badges", [])]), unsafe_allow_html=True)
        st.write(snapshot.get("customer_value_summary", ""))

        checks_df = pd.DataFrame(snapshot.get("qa_checks", []))
        if len(checks_df) > 0:
            st.subheader("QA checks")
            st.dataframe(checks_df, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Fix before delivery")
            for item in snapshot.get("fix_before_delivery", []):
                st.warning(item)
            st.subheader("Customer gets")
            for item in snapshot.get("customer_receives", []):
                st.write(f"- {item}")
        with col_b:
            st.subheader("Do not deliver yet")
            for item in snapshot.get("do_not_deliver_yet", []):
                st.error(item)
            st.subheader("Founder time shield")
            for item in snapshot.get("founder_time_shield", []):
                st.write(f"- {item}")

        with st.expander("Acceptance criteria", expanded=False):
            for item in snapshot.get("acceptance_criteria", []):
                st.write(f"- {item}")
        with st.expander("Safe claims and blocked promises", expanded=False):
            st.markdown("**Safe claims**")
            for item in snapshot.get("safe_claims", []):
                st.write(f"- {item}")
            st.markdown("**Blocked promises**")
            for item in snapshot.get("blocked_promises", []):
                st.write(f"- {item}")

        if st.session_state.get("deliverable_qa_v75_bundle"):
            st.download_button(
                "Download Deliverable QA Bundle V75",
                st.session_state.deliverable_qa_v75_bundle,
                file_name=f"{st.session_state.project_name}_deliverable_qa_v75.zip",
                mime="application/zip",
                use_container_width=True,
                key="v75_download_deliverable_qa_bundle",
            )
    else:
        st.info("Run V75 before you deliver a paid pilot package. It protects the customer from unclear outputs and protects you from manual support/rework.")

    st.caption("V75 approves pilot deliverables only when value, evidence, privacy, payment/scope and customer clarity are good enough. It still does not replace field validation or legal/security review.")

# ============================================================
# ORGANIZED PAGE DISPATCHER V45.2
# ============================================================


# ============================================================
# V76 — Product Consolidation & Flow Simplifier
# ============================================================

def render_product_consolidation_v76_tab():
    st.header("Product Consolidation & Flow Simplifier V76")
    st.write(
        "V76 makes EdgeTwin feel like one calm product instead of many separate tools. "
        "The engine stays powerful, but customers see a clear route and founder-only modules stay behind the cockpit."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        customer_route_style = st.selectbox(
            "Customer route style",
            ["Guided 5-step route", "Single-page cockpit", "Assisted upload-first route", "Proposal-first route"],
            index=0,
            key="v76_customer_route_style",
        )
        launch_target = st.selectbox(
            "Launch target",
            ["Private beta", "Paid pilot launch", "Public SaaS preview", "Enterprise/on-prem evaluation"],
            index=1,
            key="v76_launch_target",
        )
    with c2:
        founder_visibility = st.selectbox(
            "Founder visibility",
            ["Full founder cockpit", "Only blockers + next actions", "Commercial cockpit first", "Reliability cockpit first"],
            index=1,
            key="v76_founder_visibility",
        )
        max_customer_steps = st.slider("Maximum customer-visible steps", 3, 8, 5, 1, key="v76_max_customer_steps")
    with c3:
        complexity_tolerance = st.selectbox(
            "Customer complexity tolerance",
            ["Very low", "Low", "Medium", "Technical buyer"],
            index=1,
            key="v76_complexity_tolerance",
        )
        hide_advanced_by_default = st.checkbox("Hide advanced modules by default", value=True, key="v76_hide_advanced")
        keep_founder_tools = st.checkbox("Keep founder tools available", value=True, key="v76_keep_founder_tools")

    snapshots = {
        "customer_journey_v73": st.session_state.get("customer_journey_v73_snapshot"),
        "deliverable_qa_v75": st.session_state.get("deliverable_qa_v75_snapshot"),
        "quality_guardian_v74": st.session_state.get("quality_guardian_v74_snapshot"),
        "customer_status_v72": st.session_state.get("customer_status_v72_snapshot"),
        "customer_support_v71": st.session_state.get("customer_support_v71_snapshot"),
        "outcome_assurance_v70": st.session_state.get("outcome_assurance_v70_snapshot"),
        "zero_touch_v69": st.session_state.get("zero_touch_v69_snapshot"),
        "automation_orchestrator_v68": st.session_state.get("automation_orchestrator_v68_snapshot"),
        "reliability_calibration_v67": st.session_state.get("reliability_calibration_v67_snapshot"),
        "continuous_improvement_v66": st.session_state.get("continuous_improvement_v66_snapshot"),
        "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
        "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
        "pricing_offer": st.session_state.get("pricing_offer_snapshot"),
        "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
        "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
        "roi_value_v62": st.session_state.get("roi_value_v62_snapshot"),
        "cloud_architecture_v58": st.session_state.get("cloud_architecture_v58_snapshot"),
        "hardware_reference_v59": st.session_state.get("hardware_reference_v59_snapshot"),
        "commercial_release_v60": st.session_state.get("commercial_release_v60_snapshot"),
        "product_readiness_v40": st.session_state.get("product_readiness_v40_snapshot"),
    }
    bundle_presence = {
        "auto_pilot_bundle": st.session_state.get("auto_pilot_bundle") is not None,
        "fusion_bundle": st.session_state.get("fusion_bundle") is not None,
        "enterprise_bundle": st.session_state.get("enterprise_bundle") is not None,
        "customer_journey_v73_bundle": st.session_state.get("customer_journey_v73_bundle") is not None,
        "customer_status_v72_bundle": st.session_state.get("customer_status_v72_bundle") is not None,
        "outcome_assurance_v70_bundle": st.session_state.get("outcome_assurance_v70_bundle") is not None,
        "customer_support_v71_bundle": st.session_state.get("customer_support_v71_bundle") is not None,
        "deliverable_qa_v75_bundle": st.session_state.get("deliverable_qa_v75_bundle") is not None,
        "quality_guardian_v74_bundle": st.session_state.get("quality_guardian_v74_bundle") is not None,
        "pricing_offer_bundle": st.session_state.get("pricing_offer_bundle") is not None,
        "proposal_sow_bundle": st.session_state.get("proposal_sow_bundle") is not None,
    }

    st.caption("V76 consolidates the experience; it does not weaken the engine or remove founder controls.")

    if st.button("Build Product Consolidation Plan V76", type="primary", use_container_width=True, key="v76_build_consolidation_plan"):
        snapshot = core.build_product_consolidation_flow_simplifier_v76(
            project_name=st.session_state.project_name,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            snapshots=snapshots,
            bundle_presence=bundle_presence,
            customer_route_style=customer_route_style,
            founder_visibility=founder_visibility,
            launch_target=launch_target,
            complexity_tolerance=complexity_tolerance,
            max_customer_steps=int(max_customer_steps),
            hide_advanced_by_default=bool(hide_advanced_by_default),
            keep_founder_tools=bool(keep_founder_tools),
        )
        st.session_state.product_consolidation_v76_snapshot = snapshot
        st.session_state.product_consolidation_v76_bundle = core.create_product_consolidation_v76_bundle(st.session_state.project_name, snapshot)
        st.success("Product Consolidation Plan V76 generated.")

    snapshot = st.session_state.get("product_consolidation_v76_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Simplicity", f"{snapshot.get('simplicity_score', 0)}%")
        m2.metric("Engine coverage", f"{snapshot.get('engine_coverage_score', 0)}%")
        m3.metric("Founder automation", f"{snapshot.get('founder_automation_score', 0)}%")
        m4.metric("Chaos risk", snapshot.get("chaos_risk", "Unknown"))

        decision = snapshot.get("decision", "")
        if "CLEAR" in decision or "GO" in decision:
            st.success(snapshot.get("recommended_next_step"))
        elif "CONSOLIDATE" in decision or "FOUNDER" in decision:
            st.warning(snapshot.get("recommended_next_step"))
        else:
            st.error(snapshot.get("recommended_next_step"))

        badges = snapshot.get("status_badges", [])
        if badges:
            st.markdown(" ".join([f'<span class="v51-badge">{b}</span>' for b in badges]), unsafe_allow_html=True)

        st.markdown("### Customer one-route summary")
        st.info(snapshot.get("customer_one_route_summary", ""))

        tabs = st.tabs(["Customer route", "Founder cockpit", "Hide from customer", "Quality gates", "Backlog"])
        with tabs[0]:
            route_df = pd.DataFrame(snapshot.get("customer_route", []))
            if len(route_df) > 0:
                st.dataframe(route_df, use_container_width=True)
            st.subheader("Customer-visible promises")
            for item in snapshot.get("customer_visible_value", []):
                st.success(item)
        with tabs[1]:
            cockpit_df = pd.DataFrame(snapshot.get("founder_cockpit_groups", []))
            if len(cockpit_df) > 0:
                st.dataframe(cockpit_df, use_container_width=True)
            st.subheader("Founder time shield")
            for item in snapshot.get("founder_time_shield", []):
                st.info(item)
        with tabs[2]:
            st.subheader("Modules hidden from customer by default")
            for item in snapshot.get("hide_from_customer", []):
                st.write(f"- {item}")
            st.subheader("Do not remove from engine")
            for item in snapshot.get("do_not_remove_from_engine", []):
                st.warning(item)
        with tabs[3]:
            gates_df = pd.DataFrame(snapshot.get("quality_gates", []))
            if len(gates_df) > 0:
                st.dataframe(gates_df, use_container_width=True)
            st.subheader("Blocked promises")
            for item in snapshot.get("blocked_promises", []):
                st.error(item)
        with tabs[4]:
            st.subheader("Consolidation actions")
            for item in snapshot.get("consolidation_actions", []):
                st.write(f"- {item}")
            st.subheader("Do not do next")
            for item in snapshot.get("do_not_do_next", []):
                st.warning(item)

        if st.session_state.get("product_consolidation_v76_bundle"):
            st.download_button(
                "Download Product Consolidation Bundle V76",
                st.session_state.product_consolidation_v76_bundle,
                file_name=f"{st.session_state.project_name}_product_consolidation_v76.zip",
                mime="application/zip",
                use_container_width=True,
                key="v76_download_product_consolidation_bundle",
            )

    st.caption("V76 is the anti-chaos layer: one customer route, full founder cockpit, and all deeper reliability/privacy/commercial gates still protected behind the scenes.")


# ============================================================
# V77 — Smart Intake Router / Minimal Customer Input Engine
# ============================================================

def render_smart_intake_v77_tab():
    st.header("Smart Intake Router V77")
    st.write(
        "V77 reduces customer friction: the customer gives a small amount of information, "
        "and EdgeTwin recommends the safest route, required data, package level and founder approval gates."
    )

    st.markdown(
        '<div class="v51-hero"><b>Goal:</b> fewer customer questions, more automatic routing, no weaker reliability gates.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        customer_problem = st.text_area(
            "Customer problem in one sentence",
            value="We want to know if our sensor/use-case is suitable for an Edge AI pilot.",
            height=90,
            key="v77_customer_problem",
        )
        industry = st.selectbox(
            "Industry / context",
            ["Predictive maintenance", "Construction / asset security", "Remote forestry / outdoor", "Manufacturing", "Energy / utilities", "Custom / unknown"],
            index=0,
            key="v77_industry",
        )
        urgency = st.selectbox(
            "Urgency",
            ["Explore idea", "Pilot this month", "Active operational pain", "Enterprise evaluation"],
            index=1,
            key="v77_urgency",
        )
    with c2:
        data_status = st.selectbox(
            "Data status",
            ["No data yet", "Some WAV/CSV samples", "Labelled real data", "Live node data", "Unknown"],
            index=1,
            key="v77_data_status",
        )
        sensor_stack = st.multiselect(
            "Known sensors",
            ["Vibration / IMU", "Audio / Acoustic", "Temperature", "Gas / Environment", "Radar", "GPS / Zone", "LoRa / Gateway", "Unknown"],
            default=["Vibration / IMU", "Audio / Acoustic"],
            key="v77_sensor_stack",
        )
        privacy_preference = st.selectbox(
            "Privacy preference",
            ["Private only", "Feature learning allowed", "Raw data permission needed later"],
            index=0,
            key="v77_privacy_preference",
        )
    with c3:
        desired_output = st.selectbox(
            "Desired output",
            ["Quick feasibility", "Pilot bundle", "Real-data analysis", "Edge Impulse export", "Paid pilot proposal", "Enterprise/on-prem evaluation"],
            index=1,
            key="v77_desired_output",
        )
        budget_band = st.selectbox(
            "Budget signal",
            ["Unknown", "Low / testing", "Starter pilot", "Professional pilot", "Enterprise budget"],
            index=2,
            key="v77_budget_band",
        )
        customer_self_service = st.checkbox("Customer should self-serve as much as possible", value=True, key="v77_customer_self_service")
        founder_time_limit = st.selectbox("Founder time target", ["Near-zero touch", "Under 30 minutes", "Under 2 hours", "Hands-on custom"], index=0, key="v77_founder_time_limit")

    snapshots = {
        "product_consolidation_v76": st.session_state.get("product_consolidation_v76_snapshot"),
        "deliverable_qa_v75": st.session_state.get("deliverable_qa_v75_snapshot"),
        "quality_guardian_v74": st.session_state.get("quality_guardian_v74_snapshot"),
        "customer_journey_v73": st.session_state.get("customer_journey_v73_snapshot"),
        "customer_status_v72": st.session_state.get("customer_status_v72_snapshot"),
        "customer_support_v71": st.session_state.get("customer_support_v71_snapshot"),
        "outcome_assurance_v70": st.session_state.get("outcome_assurance_v70_snapshot"),
        "zero_touch_v69": st.session_state.get("zero_touch_v69_snapshot"),
        "automation_orchestrator_v68": st.session_state.get("automation_orchestrator_v68_snapshot"),
        "reliability_calibration_v67": st.session_state.get("reliability_calibration_v67_snapshot"),
        "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
        "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
        "pricing_offer": st.session_state.get("pricing_offer_snapshot"),
        "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
        "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
    }

    if st.button("Build Smart Intake Route V77", type="primary", use_container_width=True, key="v77_build_smart_intake"):
        snapshot = core.build_smart_intake_router_v77(
            project_name=st.session_state.project_name,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            customer_problem=customer_problem,
            industry=industry,
            data_status=data_status,
            sensor_stack=sensor_stack,
            urgency=urgency,
            desired_output=desired_output,
            budget_band=budget_band,
            privacy_preference=privacy_preference,
            customer_self_service=bool(customer_self_service),
            founder_time_limit=founder_time_limit,
            snapshots=snapshots,
        )
        st.session_state.smart_intake_v77_snapshot = snapshot
        st.session_state.smart_intake_v77_bundle = core.create_smart_intake_v77_bundle(st.session_state.project_name, snapshot)
        st.success("Smart Intake Route V77 generated.")

    snapshot = st.session_state.get("smart_intake_v77_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Intake clarity", f"{snapshot.get('intake_clarity_score', 0)}%")
        m2.metric("Automation fit", f"{snapshot.get('automation_fit_score', 0)}%")
        m3.metric("Founder touch risk", snapshot.get("founder_touch_risk", "Unknown"))
        m4.metric("Decision", snapshot.get("decision", "Unknown"))

        decision = snapshot.get("decision", "")
        if "SELF-SERVICE" in decision or "GO" in decision:
            st.success(snapshot.get("recommended_next_step", ""))
        elif "FOUNDER" in decision or "CONDITIONAL" in decision:
            st.warning(snapshot.get("recommended_next_step", ""))
        else:
            st.error(snapshot.get("recommended_next_step", ""))

        badges = snapshot.get("status_badges", [])
        if badges:
            st.markdown(" ".join([f'<span class="v51-badge">{b}</span>' for b in badges]), unsafe_allow_html=True)

        st.markdown("### Customer-facing answer")
        st.info(snapshot.get("customer_facing_answer", ""))

        tabs = st.tabs(["Recommended route", "Auto config", "Missing inputs", "Approval gates", "Hidden engine"])
        with tabs[0]:
            route_df = pd.DataFrame(snapshot.get("recommended_route", []))
            if len(route_df) > 0:
                st.dataframe(route_df, use_container_width=True)
            st.subheader("Customer next actions")
            for item in snapshot.get("customer_next_actions", []):
                st.success(item)
        with tabs[1]:
            st.json(snapshot.get("auto_config", {}))
            st.subheader("Recommended package")
            st.write(snapshot.get("recommended_package", ""))
        with tabs[2]:
            for item in snapshot.get("missing_inputs", []):
                st.warning(item)
            st.subheader("Do not ask customer yet")
            for item in snapshot.get("do_not_ask_yet", []):
                st.info(item)
        with tabs[3]:
            for item in snapshot.get("founder_approval_gates", []):
                st.warning(item)
            st.subheader("Blocked promises")
            for item in snapshot.get("blocked_promises", []):
                st.error(item)
        with tabs[4]:
            hidden_df = pd.DataFrame(snapshot.get("hidden_engine_checks", []))
            if len(hidden_df) > 0:
                st.dataframe(hidden_df, use_container_width=True)
            st.subheader("Recommended internal modules")
            for item in snapshot.get("recommended_internal_modules", []):
                st.write(f"- {item}")

        if st.session_state.get("smart_intake_v77_bundle"):
            st.download_button(
                "Download Smart Intake Bundle V77",
                st.session_state.smart_intake_v77_bundle,
                file_name=f"{st.session_state.project_name}_smart_intake_v77.zip",
                mime="application/zip",
                use_container_width=True,
                key="v77_download_smart_intake_bundle",
            )

    st.caption("V77 asks less from the customer while keeping reliability, privacy, payment and delivery gates active behind the scenes.")


# ============================================================
# V78 — One-Click Pilot Assembler / Autonomous Fulfillment Preview
# ============================================================

def render_one_click_pilot_v78_tab():
    st.header("One-Click Pilot Assembler V78")
    st.write(
        "V78 turns the simplified intake route into one customer-ready pilot package. "
        "The customer sees one clear outcome; EdgeTwin quietly checks data, privacy, reliability, value, proposal and delivery gates."
    )

    st.markdown("""
    <div class="v51-hero">
      <h3>One clear package, deep checks underneath</h3>
      <p>The goal is near-zero founder time: EdgeTwin assembles what can be delivered, blocks unsafe promises, and shows the customer exactly what they get for their money.</p>
    </div>
    """, unsafe_allow_html=True)

    smart = st.session_state.get("smart_intake_v77_snapshot") or {}
    if smart:
        st.success(f"Using Smart Intake V77 route: {smart.get('recommended_package', 'recommended package')} — {smart.get('decision', 'route available')}")
    else:
        st.info("Tip: run Smart Intake V77 first for the strongest auto-route. V78 can still create a safe package from the current project state.")

    c1, c2, c3 = st.columns(3)
    with c1:
        package_level = st.selectbox(
            "Package level",
            ["Auto from Smart Intake", "Starter Pilot Bundle", "Professional Pilot Bundle", "Real-Data Pilot Bundle", "Enterprise / custom evaluation"],
            index=0,
            key="v78_package_level",
        )
        customer_problem = st.text_area(
            "Customer problem summary",
            value=smart.get("customer_problem", "Detect abnormal vibration/acoustic patterns and create a pilot-ready Edge AI package."),
            height=95,
            key="v78_customer_problem",
        )
    with c2:
        data_route = st.selectbox(
            "Data route",
            ["Auto from Smart Intake", "Generate synthetic pilot dataset", "Use uploaded real data features", "Hybrid synthetic + real samples", "Discovery needed first"],
            index=0,
            key="v78_data_route",
        )
        allow_customer_download = st.checkbox("Allow customer-ready bundle if QA passes", value=True, key="v78_allow_customer_download")
        include_advanced_details = st.checkbox("Put technical details under Advanced only", value=True, key="v78_advanced_details")
    with c3:
        include_proposal = st.checkbox("Include proposal/SOW summary", value=True, key="v78_include_proposal")
        include_checkout = st.checkbox("Include checkout/invoice readiness summary", value=True, key="v78_include_checkout")
        founder_review_required = st.checkbox("Founder approval before paid delivery", value=True, key="v78_founder_review_required")

    snapshots = {
        "smart_intake_v77": st.session_state.get("smart_intake_v77_snapshot"),
        "product_consolidation_v76": st.session_state.get("product_consolidation_v76_snapshot"),
        "deliverable_qa_v75": st.session_state.get("deliverable_qa_v75_snapshot"),
        "quality_guardian_v74": st.session_state.get("quality_guardian_v74_snapshot"),
        "customer_journey_v73": st.session_state.get("customer_journey_v73_snapshot"),
        "customer_status_v72": st.session_state.get("customer_status_v72_snapshot"),
        "customer_support_v71": st.session_state.get("customer_support_v71_snapshot"),
        "outcome_assurance_v70": st.session_state.get("outcome_assurance_v70_snapshot"),
        "zero_touch_v69": st.session_state.get("zero_touch_v69_snapshot"),
        "automation_orchestrator_v68": st.session_state.get("automation_orchestrator_v68_snapshot"),
        "reliability_calibration_v67": st.session_state.get("reliability_calibration_v67_snapshot"),
        "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
        "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
        "roi_value_v62": st.session_state.get("roi_value_v62_snapshot"),
        "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
        "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
        "pricing_offer": st.session_state.get("pricing_offer_snapshot"),
    }

    if st.button("Assemble One-Click Pilot Package V78", type="primary", use_container_width=True, key="v78_assemble_one_click_package"):
        snapshot = core.build_one_click_pilot_package_v78(
            project_name=st.session_state.project_name,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            smart_intake_snapshot=smart,
            customer_problem=customer_problem,
            package_level=package_level,
            data_route=data_route,
            include_proposal=bool(include_proposal),
            include_checkout=bool(include_checkout),
            include_advanced_details=bool(include_advanced_details),
            allow_customer_download=bool(allow_customer_download),
            founder_review_required=bool(founder_review_required),
            snapshots=snapshots,
        )
        st.session_state.one_click_pilot_v78_snapshot = snapshot
        st.session_state.one_click_pilot_v78_bundle = core.create_one_click_pilot_v78_bundle(st.session_state.project_name, snapshot)
        st.success("One-Click Pilot Package V78 assembled.")

    snapshot = st.session_state.get("one_click_pilot_v78_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Package readiness", f"{snapshot.get('package_readiness_score', 0)}%")
        m2.metric("Customer clarity", f"{snapshot.get('customer_clarity_score', 0)}%")
        m3.metric("Founder time", snapshot.get("founder_time_risk", "Unknown"))
        m4.metric("Decision", snapshot.get("decision", "Unknown"))

        decision = snapshot.get("decision", "")
        if "READY" in decision or "GO" in decision:
            st.success(snapshot.get("recommended_next_step", ""))
        elif "CONDITIONAL" in decision:
            st.warning(snapshot.get("recommended_next_step", ""))
        else:
            st.error(snapshot.get("recommended_next_step", ""))

        badges = snapshot.get("status_badges", [])
        if badges:
            st.markdown(" ".join([f'<span class="v51-badge">{b}</span>' for b in badges]), unsafe_allow_html=True)

        st.markdown("### Customer-ready message")
        st.info(snapshot.get("customer_message", ""))

        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            st.subheader("Package contents")
            contents_df = pd.DataFrame(snapshot.get("package_contents", []))
            if len(contents_df) > 0:
                st.dataframe(contents_df, use_container_width=True)
        with col_b:
            st.subheader("What the customer pays for")
            for item in snapshot.get("value_points", []):
                st.success(item)

        tabs = st.tabs(["Delivery gates", "Blocked automation", "Founder shield", "Advanced manifest"])
        with tabs[0]:
            gates_df = pd.DataFrame(snapshot.get("delivery_gates", []))
            if len(gates_df) > 0:
                st.dataframe(gates_df, use_container_width=True)
            st.subheader("Missing before delivery")
            for item in snapshot.get("missing_before_delivery", []):
                st.warning(item)
        with tabs[1]:
            for item in snapshot.get("blocked_automation", []):
                st.error(item)
            st.subheader("Safe automation allowed")
            for item in snapshot.get("safe_automation_allowed", []):
                st.info(item)
        with tabs[2]:
            for item in snapshot.get("founder_time_shield", []):
                st.success(item)
            st.subheader("Founder approval only when")
            for item in snapshot.get("founder_approval_only_when", []):
                st.warning(item)
        with tabs[3]:
            st.json(snapshot.get("package_manifest", {}))
            st.subheader("Internal modules used")
            for item in snapshot.get("internal_modules_used", []):
                st.write(f"- {item}")

        if st.session_state.get("one_click_pilot_v78_bundle"):
            st.download_button(
                "Download One-Click Pilot Bundle V78",
                st.session_state.one_click_pilot_v78_bundle,
                file_name=f"{st.session_state.project_name}_one_click_pilot_v78.zip",
                mime="application/zip",
                use_container_width=True,
                key="v78_download_one_click_pilot_bundle",
            )

    st.caption("V78 keeps the customer experience simple while preserving the deep EdgeTwin quality, privacy, reliability and delivery gates behind the scenes.")



# ============================================================
# V79 — Pilot Factory Control Tower / Lifecycle State Machine
# ============================================================

def render_pilot_factory_v79_tab():
    st.header("Pilot Factory Control Tower V79")
    st.write(
        "V79 keeps the whole EdgeTwin journey aligned as one lifecycle: intake, data, readiness, privacy, proposal, checkout, delivery and support. "
        "The customer sees one simple status and next step; Founder Mode keeps the approval gates visible."
    )

    existing_one_click = st.session_state.get("one_click_pilot_v78_snapshot") or {}
    existing_intake = st.session_state.get("smart_intake_v77_snapshot") or {}

    c1, c2, c3 = st.columns(3)
    with c1:
        current_stage = st.selectbox(
            "Current lifecycle stage",
            [
                "New lead",
                "Intake routed",
                "Data uploaded/generated",
                "Readiness reviewed",
                "Proposal requested",
                "Checkout/invoice ready",
                "Delivery QA ready",
                "Delivered / follow-up",
            ],
            index=1 if existing_intake else 0,
            key="v79_current_stage",
        )
        desired_outcome = st.selectbox(
            "Desired customer outcome",
            ["Pilot bundle", "Real-data pilot", "Proposal/SOW", "Paid pilot delivery", "Hardware reference proof"],
            index=0,
            key="v79_desired_outcome",
        )
    with c2:
        automation_level = st.selectbox(
            "Automation level",
            ["Assisted automation", "Self-service preview", "Founder approval required", "Hold until gates pass"],
            index=0,
            key="v79_automation_level",
        )
        customer_data_status = st.selectbox(
            "Customer data status",
            ["No data yet", "Synthetic only", "Real data uploaded", "Mixed synthetic + real", "Sensitive/raw data case"],
            index=2 if st.session_state.get("real_upload_v56_snapshot") else 1,
            key="v79_customer_data_status",
        )
    with c3:
        paid_delivery_requested = st.checkbox("Paid delivery requested", value=bool(st.session_state.get("checkout_v57_snapshot")), key="v79_paid_delivery_requested")
        allow_customer_self_service = st.checkbox("Allow customer self-service where safe", value=True, key="v79_allow_self_service")
        strict_quality_mode = st.checkbox("Strict quality mode", value=True, key="v79_strict_quality_mode")

    risk_inputs = st.multiselect(
        "Risk flags",
        [
            "Safety-critical use-case",
            "Raw audio/location/customer identifiers",
            "Customer asks for guaranteed production accuracy",
            "Payment/scope not confirmed",
            "No real data yet",
            "Enterprise/on-prem request",
        ],
        default=[] if existing_one_click else ["No real data yet"],
        key="v79_risk_inputs",
    )

    snapshots = {
        "smart_intake_v77": st.session_state.get("smart_intake_v77_snapshot"),
        "one_click_pilot_v78": st.session_state.get("one_click_pilot_v78_snapshot"),
        "deliverable_qa_v75": st.session_state.get("deliverable_qa_v75_snapshot"),
        "quality_guardian_v74": st.session_state.get("quality_guardian_v74_snapshot"),
        "customer_status_v72": st.session_state.get("customer_status_v72_snapshot"),
        "outcome_assurance_v70": st.session_state.get("outcome_assurance_v70_snapshot"),
        "customer_support_v71": st.session_state.get("customer_support_v71_snapshot"),
        "reliability_calibration_v67": st.session_state.get("reliability_calibration_v67_snapshot"),
        "privacy_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
        "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
        "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
        "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
    }

    if st.button("Build Pilot Factory Control Plan V79", type="primary", use_container_width=True, key="v79_build_factory"):
        snapshot = core.build_pilot_factory_v79(
            project_name=st.session_state.project_name,
            dataset_df=st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
            snapshots=snapshots,
            current_stage=current_stage,
            desired_outcome=desired_outcome,
            automation_level=automation_level,
            customer_data_status=customer_data_status,
            paid_delivery_requested=paid_delivery_requested,
            allow_customer_self_service=allow_customer_self_service,
            strict_quality_mode=strict_quality_mode,
            risk_inputs=risk_inputs,
        )
        st.session_state.pilot_factory_v79_snapshot = snapshot
        st.session_state.pilot_factory_v79_bundle = core.create_pilot_factory_v79_bundle(st.session_state.project_name, snapshot)
        st.success("Pilot Factory Control Plan V79 created.")

    snapshot = st.session_state.get("pilot_factory_v79_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Factory readiness", f"{snapshot.get('factory_readiness_score', 0)}%")
        m2.metric("Autonomy", f"{snapshot.get('autonomy_score', 0)}%")
        m3.metric("Risk control", f"{snapshot.get('risk_control_score', 0)}%")
        m4.metric("Decision", snapshot.get("decision", "Unknown"))

        badges = " ".join([f'<span class="v51-badge">{b}</span>' for b in snapshot.get("status_badges", [])])
        st.markdown(badges, unsafe_allow_html=True)
        st.markdown(f"### Customer status")
        st.write(snapshot.get("customer_status_message", ""))
        st.markdown(f"### Next best action")
        st.info(snapshot.get("next_best_action", ""))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Lifecycle state machine")
            st.dataframe(pd.DataFrame(snapshot.get("lifecycle", [])), use_container_width=True)
        with c2:
            st.subheader("Automation queue")
            st.dataframe(pd.DataFrame(snapshot.get("automation_queue", [])), use_container_width=True)

        with st.expander("Founder approval gates", expanded=True):
            for item in snapshot.get("founder_approval_gates", []):
                st.write(f"- {item}")
        with st.expander("Blocked until fixed", expanded=False):
            blocked = snapshot.get("blocked_until_fixed", [])
            if blocked:
                for item in blocked:
                    st.warning(item)
            else:
                st.success("No hard blockers detected for the selected automation level.")
        with st.expander("Advanced — internal modules connected", expanded=False):
            st.json(snapshot.get("module_connection_map", {}))

        if st.session_state.get("pilot_factory_v79_bundle"):
            st.download_button(
                "Download Pilot Factory Bundle V79",
                st.session_state.pilot_factory_v79_bundle,
                file_name=f"{st.session_state.project_name}_pilot_factory_v79.zip",
                mime="application/zip",
                use_container_width=True,
                key="v79_download_factory_bundle",
            )

    st.caption("V79 does not remove the deep engine. It turns it into a controlled lifecycle so customers see clarity and you only approve high-risk gates.")


# ============================================================
# V80 — Trust Ledger & Decision Traceability
# ============================================================

def render_trust_ledger_v80_tab():
    st.header("Trust Ledger & Decision Traceability V80")
    st.write(
        "V80 creates one customer-safe evidence receipt for the whole pilot: what data was used, "
        "which gates passed, what is still blocked, what can be claimed safely and where founder approval is required."
    )

    with st.expander("Why this matters", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.markdown("""
        <div class="v51-card"><b>For the customer</b><br><span class="v51-small">Clear proof of what they get, why it is trustworthy and what still needs validation.</span></div>
        """, unsafe_allow_html=True)
        c2.markdown("""
        <div class="v51-card"><b>For you</b><br><span class="v51-small">Less explaining, fewer loose promises, more automatic handoff discipline.</span></div>
        """, unsafe_allow_html=True)
        c3.markdown("""
        <div class="v51-card"><b>For strategic value</b><br><span class="v51-small">A traceable proof layer that makes the product more serious, transferable and defensible.</span></div>
        """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        include_customer_view = st.checkbox("Include customer-safe summary", value=True, key="v80_include_customer_view")
        include_founder_view = st.checkbox("Include founder/internal gates", value=True, key="v80_include_founder_view")
    with c2:
        strict_traceability = st.checkbox("Strict traceability mode", value=True, key="v80_strict_traceability")
        require_field_validation_note = st.checkbox("Always include field-validation note", value=True, key="v80_require_field_note")
    with c3:
        release_context = st.selectbox(
            "Ledger context",
            ["Customer handoff", "Paid pilot delivery", "Founder review", "Partner/buyer evidence"],
            index=0,
            key="v80_release_context",
        )
        visibility = st.radio("Visibility", ["Customer-safe", "Founder internal", "Both"], index=0, key="v80_visibility", horizontal=False)

    context = {
        "project_name": st.session_state.project_name,
        "dataset_df": st.session_state.dataset if isinstance(st.session_state.dataset, pd.DataFrame) else pd.DataFrame(),
        "fusion_doctor": st.session_state.get("fusion_doctor"),
        "hardware_result": st.session_state.get("hardware_result"),
        "auto_pilot_result": st.session_state.get("auto_pilot_result"),
        "field_learning_v52": st.session_state.get("field_learning_v52_snapshot"),
        "real_upload_v56": st.session_state.get("real_upload_v56_snapshot"),
        "reliability_calibration_v67": st.session_state.get("reliability_calibration_v67_snapshot"),
        "zero_touch_v69": st.session_state.get("zero_touch_v69_snapshot"),
        "customer_status_v72": st.session_state.get("customer_status_v72_snapshot"),
        "customer_journey_v73": st.session_state.get("customer_journey_v73_snapshot"),
        "quality_guardian_v74": st.session_state.get("quality_guardian_v74_snapshot"),
        "deliverable_qa_v75": st.session_state.get("deliverable_qa_v75_snapshot"),
        "product_consolidation_v76": st.session_state.get("product_consolidation_v76_snapshot"),
        "smart_intake_v77": st.session_state.get("smart_intake_v77_snapshot"),
        "one_click_pilot_v78": st.session_state.get("one_click_pilot_v78_snapshot"),
        "pilot_factory_v79": st.session_state.get("pilot_factory_v79_snapshot"),
        "pricing_offer": st.session_state.get("pricing_offer_snapshot"),
        "proposal_sow": st.session_state.get("proposal_sow_snapshot"),
        "checkout_v57": st.session_state.get("checkout_v57_snapshot"),
        "commercial_release_v60": st.session_state.get("commercial_release_v60_snapshot"),
        "privacy_required": require_field_validation_note,
        "strict_traceability": strict_traceability,
        "include_customer_view": include_customer_view,
        "include_founder_view": include_founder_view,
        "release_context": release_context,
        "visibility": visibility,
    }

    if st.button("Build Trust Ledger V80", type="primary", use_container_width=True, key="v80_build_trust_ledger"):
        snapshot = core.build_trust_ledger_v80(context)
        st.session_state.trust_ledger_v80_snapshot = snapshot
        st.session_state.trust_ledger_v80_bundle = core.create_trust_ledger_v80_bundle(st.session_state.project_name, snapshot)
        st.success("Trust Ledger V80 created.")

    snapshot = st.session_state.get("trust_ledger_v80_snapshot")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trust ledger", f"{snapshot.get('trust_ledger_score', 0)}%")
        m2.metric("Traceability", f"{snapshot.get('traceability_score', 0)}%")
        m3.metric("Customer clarity", f"{snapshot.get('customer_clarity_score', 0)}%")
        m4.metric("Decision", snapshot.get("decision", "Unknown"))

        badges = " ".join([f'<span class="v51-badge">{b}</span>' for b in snapshot.get("status_badges", [])])
        st.markdown(badges, unsafe_allow_html=True)
        st.markdown("### Customer-safe explanation")
        st.write(snapshot.get("customer_safe_summary", ""))
        st.markdown("### Next best action")
        st.info(snapshot.get("next_best_action", ""))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Evidence ledger")
            st.dataframe(pd.DataFrame(snapshot.get("evidence_ledger", [])), use_container_width=True)
        with c2:
            st.subheader("Gate status")
            st.dataframe(pd.DataFrame(snapshot.get("gate_status", [])), use_container_width=True)

        with st.expander("What can be shared with the customer", expanded=True):
            for item in snapshot.get("customer_shareable", []):
                st.write(f"- {item}")
        with st.expander("Founder approval required", expanded=False):
            approvals = snapshot.get("founder_approval_required", [])
            if approvals:
                for item in approvals:
                    st.warning(item)
            else:
                st.success("No extra founder approval required for the selected context.")
        with st.expander("Claims to avoid", expanded=False):
            for item in snapshot.get("claims_to_avoid", []):
                st.error(item)
        with st.expander("Advanced — raw ledger JSON", expanded=False):
            st.json(snapshot)

        if st.session_state.get("trust_ledger_v80_bundle"):
            st.download_button(
                "Download Trust Ledger Bundle V80",
                st.session_state.trust_ledger_v80_bundle,
                file_name=f"{st.session_state.project_name}_trust_ledger_v80.zip",
                mime="application/zip",
                use_container_width=True,
                key="v80_download_trust_ledger_bundle",
            )

    st.caption("V80 does not promise guaranteed accuracy. It makes the reasoning, gates, data status and allowed claims traceable before customer handoff or paid delivery.")

_PAGE_RENDERERS = {
    'trust_ledger_v80_tab': render_trust_ledger_v80_tab,
    'pilot_factory_v79_tab': render_pilot_factory_v79_tab,
    'one_click_pilot_v78_tab': render_one_click_pilot_v78_tab,
    'smart_intake_v77_tab': render_smart_intake_v77_tab,
    'product_consolidation_v76_tab': render_product_consolidation_v76_tab,
    'deliverable_qa_v75_tab': render_deliverable_qa_v75_tab,
    'quality_guardian_v74_tab': render_quality_guardian_v74_tab,
    'customer_journey_v73_tab': render_customer_journey_v73_tab,
    'customer_status_v72_tab': render_customer_status_v72_tab,
    'customer_support_v71_tab': render_customer_support_v71_tab,
    'outcome_assurance_v70_tab': render_outcome_assurance_v70_tab,
    'zero_touch_v69_tab': render_zero_touch_v69_tab,
    'automation_orchestrator_v68_tab': render_automation_orchestrator_v68_tab,
    'reliability_calibration_v67_tab': render_reliability_calibration_v67_tab,
    'continuous_improvement_v66_tab': render_continuous_improvement_v66_tab,
    'ip_moat_v65_tab': render_ip_moat_v65_tab,
    'buyer_dataroom_v64_tab': render_buyer_dataroom_v64_tab,
    'case_study_v63_tab': render_case_study_v63_tab,
    'roi_value_v62_tab': render_roi_value_v62_tab,
    'traction_proof_v61_tab': render_traction_proof_v61_tab,
    'launch_stabilization_v60_1_tab': render_launch_stabilization_v60_1_tab,
    'customer_home_v50_tab': render_customer_home_v50_tab,
    'customer_review_v50_tab': render_customer_review_v50_tab,
    'launch_experience_v53_tab': render_launch_experience_v53_tab,
    'launch_assets_v54_tab': render_launch_assets_v54_tab,
    'first_customer_beta_v55_tab': render_first_customer_beta_v55_tab,
    'real_upload_v56_tab': render_real_upload_v56_tab,
    'home': render_home,
    'wizard_tab': render_wizard_tab,
    'fusion_tab': render_fusion_tab,
    'audit_tab': render_audit_tab,
    'optimizer_tab': render_optimizer_tab,
    'real_bridge_tab': render_real_bridge_tab,
    'field_learning_v52_tab': render_field_learning_v52_tab,
    'trust_tab': render_trust_tab,
    'deployment_tab': render_deployment_tab,
    'reports_tab': render_reports_tab,
    'hardening_tab': render_hardening_tab,
    'beta_launch_tab': render_beta_launch_tab,
    'monetization_tab': render_monetization_tab,
    'api_tab': render_api_tab,
    'marketplace_tab': render_marketplace_tab,
    'normality_tab': render_normality_tab,
    'edge_impulse_tab': render_edge_impulse_tab,
    'ei_classifier_tab': render_ei_classifier_tab,
    'success_gate_tab': render_success_gate_tab,
    'golden_demo_tab': render_golden_demo_tab,
    'closed_beta_tab': render_closed_beta_tab,
    'paid_export_tab': render_paid_export_tab,
    'field_validation_tab': render_field_validation_tab,
    'field_evidence_v2_tab': render_field_evidence_v2_tab,
    'edge_starter_tab': render_edge_starter_tab,
    'scalability_tab': render_scalability_tab,
    'operational_tab': render_operational_tab,
    'observability_tab': render_observability_tab,
    'governance_tab': render_governance_tab,
    'onboarding_tab': render_onboarding_tab,
    'workspace_tab': render_workspace_tab,
    'admin_tab': render_admin_tab,
    'license_cert_tab': render_license_cert_tab,
    'commercial_release_v60_tab': render_commercial_release_v60_tab,
    'product_readiness_tab': render_product_readiness_tab,
    'cloud_architecture_v58_tab': render_cloud_architecture_v58_tab,
    'hardware_reference_v59_tab': render_hardware_reference_v59_tab,
    'delivery_tab': render_delivery_tab,
    'customer_success_tab': render_customer_success_tab,
    'founder_ops_v49_tab': render_founder_ops_v49_tab,
    'lead_intake_v48_tab': render_lead_intake_v48_tab,
    'pricing_offer_tab': render_pricing_offer_tab,
    'proposal_sow_tab': render_proposal_sow_tab,
    'checkout_v57_tab': render_checkout_v57_tab,
    'quote_to_cash_tab': render_quote_to_cash_tab,
    'paid_pilot_v45_tab': render_paid_pilot_v45_tab,
    'canvas_tab': render_canvas_tab,
    'packs_tab': render_packs_tab,
    'hardware_tab': render_hardware_tab,
}

render_func = _PAGE_RENDERERS.get(nav_page_key)
if render_func is None:
    st.error("Selected page is not available in this build.")
else:
    render_func()
