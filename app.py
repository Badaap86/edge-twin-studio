import io
import json
import uuid
import zipfile
import warnings

import numpy as np
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

st.set_page_config(page_title="EdgeTwin Studio V18", layout="wide", initial_sidebar_state="expanded")

core.init_db()


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
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


def render_metric_cards(doctor):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Diversity", f"{doctor.get('diversity_score', 0)}%")
    c2.metric("Balance", f"{doctor.get('balance_score', 0)}%")
    c3.metric("Separation", f"{doctor.get('separation_score', 0)}%")
    c4.metric("Overall", f"{doctor.get('overall_score', 0)}%")


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
    )
    st.session_state.fusion_bundle = bundle
    st.session_state.dataset = training_df.copy()
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
    st.markdown("### Turn sensor ideas into Edge AI-ready datasets, reports and hardware advice.")

    login_tab, register_tab = st.tabs(["Login", "Create account"])

    with login_tab:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", type="primary"):
            user = core.authenticate_user(username, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Login failed. Check username and password.")

    with register_tab:
        username = st.text_input("New username", key="reg_user")
        password = st.text_input("New password", type="password", key="reg_pass")
        if st.button("Create account"):
            user = core.create_user(username, password)
            if user:
                st.session_state.user = user
                st.success("Account created.")
                st.rerun()
            else:
                st.error("Username already exists.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("EdgeTwin Studio")
st.sidebar.caption("Powered by OMEGA-X Engine")
st.sidebar.success(f"Logged in as {st.session_state.user['username']}")

st.session_state.project_name = st.sidebar.text_input("Project name", st.session_state.project_name)

if st.sidebar.button("Save project", use_container_width=True):
    settings = {
        "selected_template": st.session_state.selected_template,
        "last_demo_summary": st.session_state.last_demo_summary,
        "fusion_manifest": st.session_state.fusion_manifest,
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
    choice = st.sidebar.selectbox("Load project", ["-"] + projects["name"].tolist())
    if choice != "-" and st.sidebar.button("Load selected", use_container_width=True):
        proj_id = projects.loc[projects["name"] == choice, "id"].iloc[0]
        loaded = core.load_project(proj_id, st.session_state.user["id"])
        if loaded:
            st.session_state.project_name = loaded["name"]
            st.session_state.dataset = loaded["dataset"]
            st.sidebar.success("Project loaded.")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Canvas settings")
st.session_state.sr = 4000 if st.sidebar.radio("Signal type", ["Audio / Acoustic", "Vibration / IMU"]) == "Vibration / IMU" else 16000
st.session_state.current_label = st.sidebar.text_input("Dataset label", st.session_state.current_label)
st.session_state.base_f = st.sidebar.slider("Base frequency", 0.0, 1000.0, float(st.session_state.base_f), 5.0)
st.session_state.harm_r = st.sidebar.slider("Harmonics", 0.0, 2.0, float(st.session_state.harm_r), 0.05)
st.session_state.imp_r = st.sidebar.slider("Impact rate", 0.0, 50.0, float(st.session_state.imp_r), 0.5)
st.session_state.noise_l = st.sidebar.slider("Noise", 0.0, 1.0, float(st.session_state.noise_l), 0.02)

if st.sidebar.button("Logout", use_container_width=True):
    del st.session_state.user
    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("EdgeTwin Studio V18")
st.caption("Self-Selling Demo Mode • Sensor Fusion • Dataset Doctor • Hardware Architect • Professional Reports")


home, fusion_tab, audit_tab, canvas_tab, packs_tab, hardware_tab = st.tabs([
    "🏠 Self-Selling Demo",
    "🧬 Sensor Fusion Studio",
    "🩺 Enterprise Audit",
    "📈 Signal Canvas",
    "📦 Industry Packs",
    "🧱 Hardware Architect",
])


# ============================================================
# HOME / SELF SELLING DEMO
# ============================================================

with home:
    st.header("Choose the customer problem")
    st.write("One click loads the scenario, generates the fusion dataset, runs the Dataset Doctor, creates the report bundle and selects a hardware direction.")

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
            st.metric("Hardware", summary.get("recommended_board", "Unknown"))
            st.metric("Template", summary.get("template", "Unknown"))

        if st.session_state.fusion_doctor:
            render_metric_cards(st.session_state.fusion_doctor)
            render_doctor(st.session_state.fusion_doctor)

        d1, d2, d3 = st.columns(3)
        if st.session_state.fusion_bundle:
            d1.download_button("Download Professional Fusion Bundle", st.session_state.fusion_bundle, file_name=f"{st.session_state.project_name}_fusion_bundle.zip", mime="application/zip", use_container_width=True)
        if st.session_state.enterprise_bundle:
            d2.download_button("Download Enterprise Bundle", st.session_state.enterprise_bundle, file_name=f"{st.session_state.project_name}_enterprise_bundle.zip", mime="application/zip", use_container_width=True)
        if isinstance(st.session_state.fusion_training_df, pd.DataFrame) and len(st.session_state.fusion_training_df) > 0:
            d3.download_button("Download Training CSV", st.session_state.fusion_training_df.to_csv(index=False), file_name=f"{st.session_state.project_name}_training.csv", mime="text/csv", use_container_width=True)

        if isinstance(st.session_state.fusion_df, pd.DataFrame) and len(st.session_state.fusion_df) > 0:
            st.subheader("Generated fusion dataset preview")
            st.dataframe(st.session_state.fusion_df.head(25), use_container_width=True)
            if "Label" in st.session_state.fusion_df.columns:
                fig = px.histogram(st.session_state.fusion_df, x="Label", title="Label distribution")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run one of the demo cards above. This is the new self-selling front door of the product.")


# ============================================================
# SENSOR FUSION
# ============================================================

with fusion_tab:
    st.header("Sensor Fusion Studio")
    template = st.selectbox("Fusion template", core.get_fusion_templates(), index=core.get_fusion_templates().index(st.session_state.selected_template) if st.session_state.selected_template in core.get_fusion_templates() else 0)
    st.session_state.selected_template = template
    t_data = core.get_fusion_template(template)
    st.info(t_data.get("description", ""))

    defaults = t_data.get("defaults", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    audio = c1.slider("Audio", 0, 100, int(defaults.get("audio", 50)))
    vibration = c2.slider("Vibration", 0, 100, int(defaults.get("vibration", 50)))
    gas = c3.slider("Gas/Env", 0, 100, int(defaults.get("gas", 20)))
    radar = c4.slider("Radar", 0, 100, int(defaults.get("radar", 50)))
    gps = c5.slider("GPS/Zone", 0, 100, int(defaults.get("gps", 50)))

    result = core.calculate_fusion_score(audio, vibration, gas, radar, gps, template)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fusion Score", f"{result['fusion_score']:.1f}")
    m2.metric("Health Score", f"{result['health_score']:.1f}")
    m3.metric("Confidence", f"{result['confidence']:.1f}")
    m4.metric("Level", result["level"])
    st.write(f"**Event:** {result['event']}  ")
    st.write(f"**Recommended action:** {result['recommended_action']}")

    st.markdown("---")
    samples = st.number_input("Samples", min_value=50, max_value=10000, value=500, step=50)
    balanced = st.checkbox("Balanced classes", value=True)
    if st.button("Generate Fusion Dataset", type="primary"):
        with st.spinner("Generating fusion dataset..."):
            fusion_df, manifest = core.generate_sensor_fusion_dataset(template, samples=samples, base_audio=audio, base_vibration=vibration, base_gas=gas, base_radar=radar, base_gps=gps, balanced_classes=balanced)
            doctor = core.fusion_dataset_doctor(fusion_df, template)
            training_df = core.create_fusion_training_dataframe(fusion_df, template)
            commercial_summary = {
                "problem": "Customer needs a faster route from sensor idea to Edge AI-ready dataset.",
                "solution": f"Use the {template} fusion template to generate a multi-sensor training dataset.",
                "output": "Full fusion CSV, training features, PDF report and dataset doctor advice.",
                "cta": "Upload real sensor files or request a custom industry pack for a pilot.",
            }
            bundle, doctor, training_df = core.create_sensor_fusion_export_bundle(st.session_state.project_name, fusion_df, manifest, commercial_summary)
            st.session_state.fusion_df = fusion_df
            st.session_state.fusion_manifest = manifest
            st.session_state.fusion_doctor = doctor
            st.session_state.fusion_training_df = training_df
            st.session_state.fusion_bundle = bundle
            st.session_state.dataset = training_df.copy()
        st.success("Fusion dataset generated and training features added to Enterprise Audit.")

    if isinstance(st.session_state.fusion_df, pd.DataFrame) and len(st.session_state.fusion_df) > 0:
        render_metric_cards(st.session_state.fusion_doctor)
        render_doctor(st.session_state.fusion_doctor)
        st.dataframe(st.session_state.fusion_df.head(50), use_container_width=True)
        st.download_button("Download Professional Fusion Bundle", st.session_state.fusion_bundle, file_name=f"{st.session_state.project_name}_fusion_bundle.zip", mime="application/zip", use_container_width=True)


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

                        preds = cross_val_predict(clf, X, y, cv=min(5, y.value_counts().min()))
                        cm = confusion_matrix(y, preds, labels=list(y.unique()))
                        fig_cm = px.imshow(cm, x=list(y.unique()), y=list(y.unique()), text_auto=True, title="Cross-validation confusion matrix")
                        st.plotly_chart(fig_cm, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Model validation skipped: {e}")

                bundle = core.create_enterprise_bundle(st.session_state.project_name, st.session_state.dataset, audit, hw)
                st.session_state.enterprise_bundle = bundle
                st.download_button("Download Enterprise Bundle", bundle, file_name=f"{st.session_state.project_name}_enterprise_bundle.zip", mime="application/zip", use_container_width=True)
            else:
                st.warning("No numeric feature columns found.")
        else:
            st.warning("Dataset needs a Label column.")
    else:
        st.info("Run a demo or generate a fusion dataset first.")

    uploaded = st.file_uploader("Or upload your own CSV for audit", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.dataset = df
        st.success("CSV loaded into Enterprise Audit.")
        st.rerun()


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

    if st.button("Add current signal features to audit dataset"):
        new_row = pd.DataFrame([features])
        if isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset) > 0:
            st.session_state.dataset = pd.concat([st.session_state.dataset, new_row], ignore_index=True)
        else:
            st.session_state.dataset = new_row
        st.success("Added to Enterprise Audit dataset.")


# ============================================================
# INDUSTRY PACKS
# ============================================================

with packs_tab:
    st.header("Industry Packs")
    pack_name = st.selectbox("Choose pack", core.get_industry_packs())
    pack = core.get_industry_pack(pack_name)
    st.info(pack.get("description", ""))
    per_class = st.number_input("Samples per class", 10, 1000, 50, 10)
    if st.button("Generate Industry Pack Dataset", type="primary"):
        with st.spinner("Generating industry pack..."):
            df, manifest = core.generate_industry_pack_dataset(pack_name, per_class)
            st.session_state.dataset = df
        st.success("Industry pack generated and loaded into Enterprise Audit.")
    if isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset) > 0:
        st.dataframe(st.session_state.dataset.head(50), use_container_width=True)
        st.download_button("Download current dataset CSV", st.session_state.dataset.to_csv(index=False), file_name=f"{pack_name.replace(' ', '_')}_dataset.csv", mime="text/csv")


# ============================================================
# HARDWARE ARCHITECT
# ============================================================

with hardware_tab:
    st.header("Hardware Architect")
    target = st.radio("Optimization target", ["balanced", "low_power", "performance", "gateway"], horizontal=True)
    num_features = st.number_input("Number of features", 1, 100, int(max(1, len(st.session_state.dataset.columns) - 1)) if isinstance(st.session_state.dataset, pd.DataFrame) and len(st.session_state.dataset.columns) else 8)
    hw_sr = st.number_input("Sample rate", 1000, 48000, int(st.session_state.sr), 1000)
    if st.button("Run Hardware Architect", type="primary"):
        st.session_state.hardware_result = core.hardware_auto_architect(num_features, hw_sr, target)
    if st.session_state.hardware_result:
        st.success(st.session_state.hardware_result["reason"])
        ranking = pd.DataFrame(st.session_state.hardware_result["ranking"])
        st.dataframe(ranking, use_container_width=True)
        fig = px.bar(ranking, x="board", y="adjusted_score", title="Hardware ranking")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Hardware catalog")
    st.dataframe(core.get_hardware_catalog(), use_container_width=True)
