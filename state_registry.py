"""EdgeTwin Studio V81 state registry.

Purpose:
- Keep project persistence centralized instead of manually saving/loading many scattered session keys.
- Persist customer/product decision state, but avoid generated ZIP/PDF bundles and large DataFrames.
- Make save/load safer as EdgeTwin grows beyond a simple MVP.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

try:
    import pandas as pd
except Exception:  # keep non-pandas tooling alive
    pd = None

STATE_REGISTRY_VERSION = "90.0"

DATAFRAME_STATE_KEYS = {'fusion_training_df', 'real_upload_v56_features_df', 'marketplace_generated_dataset', 'dataset', 'field_validation_df', 'fusion_df'}
GENERATED_BUNDLE_KEYS = ['fusion_bundle', 'enterprise_bundle', 'auto_pilot_bundle', 'optimizer_bundle', 'trust_bundle', 'real_bridge_bundle', 'reliability_v2_bundle', 'deployment_bundle', 'professional_report_bundle', 'monetization_bundle', 'hardening_bundle', 'beta_launch_bundle', 'api_automation_bundle', 'pack_marketplace_bundle', 'normality_bundle', 'edge_impulse_bundle', 'edge_impulse_classifier_bundle', 'release_success_bundle', 'golden_demo_bundle', 'closed_beta_bundle', 'paid_license_bundle', 'field_validation_bundle', 'edge_deployment_starter_bundle', 'scalability_bundle', 'operational_control_bundle', 'observability_bundle', 'customer_assurance_bundle', 'onboarding_bundle', 'guided_success_bundle', 'workspace_lifecycle_bundle', 'admin_usage_bundle', 'commercial_license_bundle', 'customer_delivery_bundle', 'customer_success_bundle', 'pricing_offer_bundle', 'paid_pilot_v45_bundle', 'proposal_sow_bundle', 'quote_to_cash_bundle', 'lead_intake_v48_bundle', 'founder_ops_v49_bundle', 'customer_mode_v50_bundle', 'customer_ui_v51_bundle', 'field_learning_v52_bundle', 'launch_experience_v53_bundle', 'launch_assets_v54_bundle', 'first_customer_beta_v55_bundle', 'real_upload_v56_bundle', 'checkout_v57_bundle', 'cloud_architecture_v58_bundle', 'hardware_reference_v59_bundle', 'commercial_release_v60_bundle', 'launch_stabilization_v60_1_bundle', 'traction_proof_v61_bundle', 'roi_value_v62_bundle', 'case_study_v63_bundle', 'buyer_dataroom_v64_bundle', 'ip_moat_v65_bundle', 'continuous_improvement_v66_bundle', 'reliability_calibration_v67_bundle', 'automation_orchestrator_v68_bundle', 'zero_touch_v69_bundle', 'outcome_assurance_v70_bundle', 'customer_support_v71_bundle', 'customer_status_v72_bundle', 'customer_journey_v73_bundle', 'quality_guardian_v74_bundle', 'deliverable_qa_v75_bundle', 'product_consolidation_v76_bundle', 'smart_intake_v77_bundle', 'one_click_pilot_v78_bundle', 'pilot_factory_v79_bundle', 'trust_ledger_v80_bundle', 'autonomy_controller_v82_bundle', 'release_guard_v83_bundle', 'ultimate_product_v90_bundle', 'field_evidence_v2_bundle', 'product_readiness_v40_bundle', 'security_hardening_v41_bundle']
VOLATILE_STATE_KEYS = {'last_operator_note_event', 'last_observability_event', 'last_admin_export_event', 'api_simulation_response'}
PROJECT_MANAGED_KEYS = {'project_name', 'project_id'}
PERSISTED_STATE_KEYS = ['admin_usage_snapshot', 'api_automation_snapshot', 'auto_pilot_config', 'auto_pilot_result', 'automation_orchestrator_v68_snapshot', 'base_f', 'beta_launch_snapshot', 'buyer_dataroom_v64_snapshot', 'case_study_v63_snapshot', 'checkout_v57_snapshot', 'closed_beta_kit', 'cloud_architecture_v58_snapshot', 'commercial_license_certificate', 'commercial_release_v60_snapshot', 'continuous_improvement_v66_snapshot', 'current_label', 'custom_pack_definition', 'customer_assurance_snapshot', 'customer_delivery_snapshot', 'customer_journey_v73_snapshot', 'customer_mode_v50_snapshot', 'customer_status_v72_snapshot', 'customer_success_snapshot', 'customer_support_v71_snapshot', 'customer_ui_v51_snapshot', 'deliverable_qa_v75_snapshot', 'deployment_plan', 'edge_deployment_starter_snapshot', 'edge_impulse_classifier_snapshot', 'edge_impulse_snapshot', 'field_evidence_v2_snapshot', 'field_learning_v52_snapshot', 'field_validation_snapshot', 'first_customer_beta_v55_snapshot', 'founder_ops_v49_snapshot', 'fusion_doctor', 'fusion_manifest', 'golden_demo_result', 'hardening_snapshot', 'hardware_reference_v59_snapshot', 'hardware_result', 'harm_r', 'imp_r', 'ip_moat_v65_snapshot', 'last_demo_summary', 'launch_assets_v54_snapshot', 'launch_experience_v53_snapshot', 'launch_stabilization_v60_1_snapshot', 'lead_intake_v48_snapshot', 'monetization_snapshot', 'noise_l', 'normality_result', 'observability_snapshot', 'onboarding_snapshot', 'one_click_pilot_v78_snapshot', 'operational_control_snapshot', 'optimizer_result', 'outcome_assurance_v70_snapshot', 'pack_marketplace_snapshot', 'paid_license_snapshot', 'paid_pilot_v45_snapshot', 'pilot_factory_v79_snapshot', 'pricing_offer_snapshot', 'product_consolidation_v76_snapshot', 'product_readiness_v40_snapshot', 'professional_report_snapshot', 'proposal_sow_snapshot', 'quality_guardian_v74_snapshot', 'quote_to_cash_snapshot', 'real_upload_v56_snapshot', 'release_success_snapshot', 'reliability_calibration_v67_snapshot', 'reliability_v2', 'roi_value_v62_snapshot', 'scalability_snapshot', 'security_hardening_v41_snapshot', 'security_v41_snapshot', 'selected_plan', 'selected_template', 'smart_intake_v77_snapshot', 'sr', 'traction_proof_v61_snapshot', 'trust_gate', 'trust_ledger_v80_snapshot', 'workspace_lifecycle_snapshot', 'workspace_mode_v50', 'zero_touch_v69_snapshot', 'autonomy_controller_v82_snapshot', 'release_guard_v83_snapshot', 'ultimate_product_v90_snapshot']

# Backward-compatible name used by app.py V80.1.
PERSISTED_EXTRA_SESSION_KEYS = PERSISTED_STATE_KEYS


def _state_get(state: Any, key: str, default: Any = None) -> Any:
    if state is None:
        return default
    getter = getattr(state, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            pass
    return getattr(state, key, default)


def _state_set(state: Any, key: str, value: Any) -> None:
    try:
        state[key] = value
        return
    except Exception:
        pass
    try:
        setattr(state, key, value)
    except Exception:
        return


def _is_dataframe(value: Any) -> bool:
    return pd is not None and isinstance(value, pd.DataFrame)


def _is_large_generated_artifact(key: str, value: Any) -> bool:
    if key.endswith("_bundle") or key in GENERATED_BUNDLE_KEYS:
        return True
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    return False


def should_persist_key(key: str, value: Any = None) -> bool:
    if key in PROJECT_MANAGED_KEYS:
        return False
    if key in DATAFRAME_STATE_KEYS:
        return False
    if key in VOLATILE_STATE_KEYS:
        return False
    if key.endswith("_bundle") or key in GENERATED_BUNDLE_KEYS:
        return False
    if value is not None:
        if _is_dataframe(value):
            return False
        if _is_large_generated_artifact(key, value):
            return False
    return key in PERSISTED_STATE_KEYS


def collect_persisted_settings(
    session_state: Any,
    compact_bridge_summary_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Collect only lightweight project settings/snapshots for SQLite metadata.

    Large DataFrames are handled by storage.py, and generated ZIP/PDF bundles are recreated on demand.
    """
    settings: Dict[str, Any] = {
        "_state_registry_version": STATE_REGISTRY_VERSION,
    }

    for key in PERSISTED_STATE_KEYS:
        value = _state_get(session_state, key, None)
        if value is None:
            continue
        if should_persist_key(key, value):
            settings[key] = value

    bridge_result = _state_get(session_state, "real_bridge_result", None)
    if bridge_result is not None:
        if callable(compact_bridge_summary_fn):
            try:
                settings["real_bridge_summary"] = compact_bridge_summary_fn(bridge_result)
            except Exception:
                settings["real_bridge_summary"] = {}
        elif isinstance(bridge_result, dict):
            settings["real_bridge_summary"] = {
                k: bridge_result.get(k)
                for k in ["status", "decision", "score", "summary"]
                if k in bridge_result
            }

    return settings


def restore_persisted_settings(session_state: Any, settings: Optional[Dict[str, Any]]) -> List[str]:
    """Restore registered lightweight state from a loaded project.

    Returns the list of keys restored, which can be shown in Founder Mode/debug views later.
    """
    if not isinstance(settings, dict):
        return []

    restored: List[str] = []
    for key in PERSISTED_STATE_KEYS:
        if key not in settings:
            continue
        value = settings.get(key)
        if should_persist_key(key, value):
            _state_set(session_state, key, value)
            restored.append(key)

    return restored


def get_registry_status(session_state: Any = None, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    known = set(PERSISTED_STATE_KEYS)
    saved = set(settings.keys()) if isinstance(settings, dict) else set()
    present = set()
    if session_state is not None:
        for key in known:
            if _state_get(session_state, key, None) is not None:
                present.add(key)

    return {
        "state_registry_version": STATE_REGISTRY_VERSION,
        "autonomy_controller_v82_supported": "autonomy_controller_v82_snapshot" in PERSISTED_STATE_KEYS,
        "release_guard_v83_supported": "release_guard_v83_snapshot" in PERSISTED_STATE_KEYS,
        "ultimate_product_v90_supported": "ultimate_product_v90_snapshot" in PERSISTED_STATE_KEYS,
        "persisted_key_count": len(PERSISTED_STATE_KEYS),
        "generated_bundle_key_count": len(GENERATED_BUNDLE_KEYS),
        "dataframe_key_count": len(DATAFRAME_STATE_KEYS),
        "volatile_key_count": len(VOLATILE_STATE_KEYS),
        "present_persisted_key_count": len(present),
        "saved_persisted_key_count": len(saved.intersection(known)),
        "not_saved_by_design": sorted(DATAFRAME_STATE_KEYS.union(set(GENERATED_BUNDLE_KEYS)).union(VOLATILE_STATE_KEYS).union(PROJECT_MANAGED_KEYS)),
        "registry_policy": "Persist lightweight project state only; rebuild bundles and store DataFrames through storage.py.",
    }
