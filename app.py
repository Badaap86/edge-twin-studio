import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io
import json
import zipfile
import warnings

from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

import core


warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="OMEGA-X Enterprise Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

core.init_db()


# ============================================================
# HELPERS
# ============================================================

def safe_dataframe_from_dataset():
    if "dataset" not in st.session_state:
        st.session_state.dataset = pd.DataFrame()

    if not isinstance(st.session_state.dataset, pd.DataFrame):
        st.session_state.dataset = pd.DataFrame(st.session_state.dataset)

    return st.session_state.dataset


def set_param_value(key, value):
    value = float(value)
    st.session_state[key] = value
    st.session_state[f"{key}_slider_widget"] = value
    st.session_state[f"{key}_number_widget"] = value


def queue_param_update(values):
    st.session_state["_pending_param_update"] = values


def apply_pending_param_updates():
    if "_pending_param_update" in st.session_state:
        values = st.session_state["_pending_param_update"]

        for key, value in values.items():
            set_param_value(key, value)

        del st.session_state["_pending_param_update"]


def sync_from_slider(key):
    value = float(st.session_state[f"{key}_slider_widget"])
    st.session_state[key] = value
    st.session_state[f"{key}_number_widget"] = value


def sync_from_number(key):
    value = float(st.session_state[f"{key}_number_widget"])
    st.session_state[key] = value
    st.session_state[f"{key}_slider_widget"] = value


def sidebar_slider_with_number(label, min_value, max_value, default_value, step, key):
    if key not in st.session_state:
        st.session_state[key] = float(default_value)

    current_value = float(st.session_state[key])

    if f"{key}_slider_widget" not in st.session_state:
        st.session_state[f"{key}_slider_widget"] = current_value

    if f"{key}_number_widget" not in st.session_state:
        st.session_state[f"{key}_number_widget"] = current_value

    st.sidebar.markdown(f"**{label}**")

    col_slider, col_number = st.sidebar.columns([3, 1])

    with col_slider:
        st.slider(
            label=f"{label} slider",
            min_value=float(min_value),
            max_value=float(max_value),
            step=float(step),
            key=f"{key}_slider_widget",
            on_change=sync_from_slider,
            args=(key,),
            label_visibility="collapsed"
        )

    with col_number:
        st.number_input(
            label=f"{label} value",
            min_value=float(min_value),
            max_value=float(max_value),
            step=float(step),
            key=f"{key}_number_widget",
            on_change=sync_from_number,
            args=(key,),
            label_visibility="collapsed"
        )

    st.session_state[key] = float(st.session_state[f"{key}_number_widget"])
    return float(st.session_state[key])


def make_zip_from_signal_files(files, manifest):
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        for item in files:
            zf.writestr(
                item["filename"],
                item["dataframe"].to_csv(index=False)
            )

        zf.writestr(
            "manifest.json",
            json.dumps(manifest, indent=2)
        )

    return zip_buf.getvalue()


def extract_features_to_dataset(uploaded_files, label, sample_rate):
    rows = []

    for f in uploaded_files:
        file_bytes = f.read()
        feats = core.extract_features_from_bytes(file_bytes, f.name, sample_rate)

        if "error" not in feats:
            feats["Label"] = label
            rows.append(feats)

    if rows:
        new_df = pd.DataFrame(rows)
        st.session_state.dataset = pd.concat(
            [safe_dataframe_from_dataset(), new_df],
            ignore_index=True
        )

    return len(rows)


def build_hardware_table(num_features, sample_rate, selected_boards=None):
    b_dat = []

    boards = selected_boards if selected_boards else core.get_available_hardware()

    for board in boards:
        ram, l_fft, l_feat, l_inf = core.estimate_edge_load(
            board,
            num_features,
            sample_rate
        )

        latency = l_fft + l_feat + l_inf
        score = core.calculate_deployment_score(board, latency, ram)
        profile = core.get_hardware_profile(board) if hasattr(core, "get_hardware_profile") else {}

        b_dat.append({
            "Board": board,
            "CPU": profile.get("cpu", "") if profile else "",
            "Role": profile.get("role", "") if profile else "",
            "Power": profile.get("power_class", "") if profile else "",
            "Score": score,
            "Latency": latency,
            "RAM": ram,
            "FFT_ms": l_fft,
            "Feature_ms": l_feat,
            "Inference_ms": l_inf,
            "Gateway Fit": profile.get("gateway_fit", "No") if profile else "No",
            "Recommended For": profile.get("recommended_for", "") if profile else "",
        })

    return b_dat


def create_enterprise_bundle(
    project_name,
    dataset_df,
    div_score,
    bal_score,
    sep_score,
    top_feats,
    b_dat,
    best_board,
    doctor=None,
    architect=None
):
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "edge_dataset.csv",
            dataset_df.to_csv(index=False)
        )

        pdf_bytes = core.generate_pdf_report(
            project_name,
            len(dataset_df),
            dataset_df["Label"].nunique() if "Label" in dataset_df.columns else 0,
            div_score,
            bal_score,
            sep_score,
            top_feats,
            b_dat,
            best_board
        )

        zf.writestr("audit_report.pdf", pdf_bytes)

        meta = {
            "project": project_name,
            "user": st.session_state.user["username"],
            "features": [c for c in dataset_df.columns if c != "Label"],
            "metrics": {
                "diversity": div_score,
                "balance": bal_score,
                "separation": sep_score,
            },
            "hardware_recommendation": best_board,
            "top_features": [
                {"feature": f, "importance": float(s)}
                for f, s in top_feats[:5]
            ],
            "dataset_doctor": doctor,
            "hardware_architect": architect,
        }

        zf.writestr(
            "metadata.json",
            json.dumps(meta, indent=2)
        )

    return zip_buf.getvalue()


def make_fusion_zip(fusion_df, manifest):
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sensor_fusion_dataset.csv", fusion_df.to_csv(index=False))
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    return zip_buf.getvalue()


def add_fusion_to_audit_dataset(fusion_df):
    numeric_cols = [
        "Label",
        "AudioScore",
        "VibrationScore",
        "GasScore",
        "RadarScore",
        "GPSZoneScore",
        "FusionScore",
        "HealthScore",
        "Confidence",
    ]

    clean_cols = [c for c in numeric_cols if c in fusion_df.columns]
    clean_df = fusion_df[clean_cols].copy()

    st.session_state.dataset = pd.concat(
        [safe_dataframe_from_dataset(), clean_df],
        ignore_index=True
    )


# ============================================================
# LOGIN / REGISTER
# ============================================================

if "user" not in st.session_state:
    st.title("🔒 OMEGA-X Enterprise Studio")
    st.caption("Industrial Edge AI Dataset Engineering • Synthetic Data • TinyML Audit • Deployment Intelligence")

    st.info(
        "Log in of maak een account aan. "
        "Daarna krijg je toegang tot je eigen projecten, datasets en exportbundels."
    )

    tab_login, tab_register = st.tabs(["Inloggen", "Nieuw Account"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Gebruikersnaam")
            password = st.text_input("Wachtwoord", type="password")

            submitted = st.form_submit_button("Log In", type="primary")

            if submitted:
                user = core.authenticate_user(username, password)

                if user:
                    st.session_state.user = user
                    st.session_state.dataset = pd.DataFrame()
                    st.session_state.project_id = "proj_" + str(np.random.randint(1000, 9999))
                    st.success("Ingelogd.")
                    st.rerun()
                else:
                    st.error("Ongeldige inloggegevens.")

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("Kies gebruikersnaam")
            new_password = st.text_input("Kies wachtwoord", type="password")

            submitted = st.form_submit_button("Account aanmaken")

            if submitted:
                if len(new_username) < 3:
                    st.warning("Gebruikersnaam moet minimaal 3 tekens hebben.")
                elif len(new_password) < 5:
                    st.warning("Wachtwoord moet minimaal 5 tekens hebben.")
                else:
                    res = core.create_user(new_username, new_password)

                    if res:
                        st.success("Account aangemaakt. Ga nu naar Inloggen.")
                    else:
                        st.error("Gebruikersnaam bestaat al.")

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "dataset" not in st.session_state:
    st.session_state.dataset = pd.DataFrame()

if "project_id" not in st.session_state:
    st.session_state.project_id = "proj_" + str(np.random.randint(1000, 9999))

if "project_name" not in st.session_state:
    st.session_state.project_name = "New_Enterprise_AI"

apply_pending_param_updates()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("👤 Account")
st.sidebar.write(f"Ingelogd als: **{st.session_state.user['username']}**")
st.sidebar.caption("Jouw API-key voor latere headless enterprise/API-versie:")
st.sidebar.code(st.session_state.user["api_key"], language="bash")

if st.sidebar.button("Uitloggen", use_container_width=True):
    for key in ["user", "dataset", "project_id"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🗂️ Projecten")

project_name = st.sidebar.text_input(
    "Project Name",
    st.session_state.get("project_name", "New_Enterprise_AI")
)

st.session_state.project_name = project_name

col_save, col_load = st.sidebar.columns(2)

if col_save.button("💾 Save"):
    settings = {
        "base_f": st.session_state.get("base_f_slider", 50.0),
        "harm_r": st.session_state.get("harm_r_slider", 0.0),
        "imp_r": st.session_state.get("imp_r_slider", 0.0),
        "noise_l": st.session_state.get("noise_l_slider", 0.1),
        "project_name": project_name,
    }

    core.save_project(
        st.session_state.project_id,
        st.session_state.user["id"],
        project_name,
        safe_dataframe_from_dataset(),
        settings
    )

    st.sidebar.success("Opgeslagen.")

user_projects = core.get_user_projects(st.session_state.user["id"])

if not user_projects.empty:
    selected_project_name = st.sidebar.selectbox(
        "Load Project",
        user_projects["name"].tolist()
    )

    if col_load.button("📂 Load"):
        proj_id = user_projects[user_projects["name"] == selected_project_name].iloc[0]["id"]

        loaded = core.load_project(
            proj_id,
            st.session_state.user["id"]
        )

        if loaded:
            st.session_state.project_id = proj_id
            st.session_state.project_name = loaded["name"]
            st.session_state.dataset = loaded["dataset"]

            settings = loaded.get("settings", {})

            queue_param_update({
                "base_f_slider": float(settings.get("base_f", 50.0)),
                "harm_r_slider": float(settings.get("harm_r", 0.0)),
                "imp_r_slider": float(settings.get("imp_r", 0.0)),
                "noise_l_slider": float(settings.get("noise_l", 0.1)),
            })

            st.sidebar.success(f"Loaded {loaded['name']}")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Studio Parameters")

modality = st.sidebar.radio(
    "Data Profiling",
    ["Vibration (4 kHz)", "Audio (16 kHz)"]
)

sr = 4000 if "Vibration" in modality else 16000

current_class = st.sidebar.text_input(
    "Dataset Label",
    "Baseline_Normal"
)

base_f = sidebar_slider_with_number(
    label="Base Freq (Hz)",
    min_value=0.0,
    max_value=1000.0,
    default_value=50.0,
    step=1.0,
    key="base_f_slider"
)

harm_r = sidebar_slider_with_number(
    label="Harmonics",
    min_value=0.0,
    max_value=2.0,
    default_value=0.0,
    step=0.01,
    key="harm_r_slider"
)

imp_r = sidebar_slider_with_number(
    label="Impacts (Hz)",
    min_value=0.0,
    max_value=50.0,
    default_value=0.0,
    step=0.1,
    key="imp_r_slider"
)

noise_l = sidebar_slider_with_number(
    label="Noise",
    min_value=0.0,
    max_value=1.0,
    default_value=0.1,
    step=0.01,
    key="noise_l_slider"
)


# ============================================================
# MAIN UI
# ============================================================

st.title("⚡ OMEGA-X Enterprise Studio V17")
st.caption(
    "Synthetic Sensor Data • Digital Twin Aging • Sensor Fusion • AI Dataset Doctor • Hardware Auto Architect"
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Canvas",
    "📦 Dataset Generator",
    "🧬 Digital Twin Aging",
    "🧪 Deep Cloner",
    "🧩 Sensor Fusion Studio",
    "🤖 Enterprise Audit",
    "🏗️ Hardware Architect"
])


# ============================================================
# TAB 1 — CANVAS
# ============================================================

with tab1:
    st.header(f"Live DSP Canvas: `{current_class}`")

    data_live = core.generate_universal_signal(
        duration=2.0,
        sr=sr,
        base_f=base_f,
        harm_r=harm_r,
        imp_r=imp_r,
        noise_l=noise_l,
        normalize=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Time Domain")
        fig_sig = go.Figure()
        fig_sig.add_trace(
            go.Scatter(
                x=data_live["t"][:2000],
                y=data_live["sig"][:2000],
                mode="lines",
                name="Signal"
            )
        )
        fig_sig.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=25, b=0)
        )
        st.plotly_chart(fig_sig, use_container_width=True)

    with c2:
        st.subheader("Frequency Domain")
        fig_fft = go.Figure()
        fig_fft.add_trace(
            go.Scatter(
                x=data_live["fft_f"],
                y=data_live["fft_v"],
                mode="lines",
                name="FFT"
            )
        )
        fig_fft.update_layout(
            height=320,
            xaxis_range=[0, 1500 if sr == 4000 else 4000],
            margin=dict(l=0, r=0, t=25, b=0)
        )
        st.plotly_chart(fig_fft, use_container_width=True)

    st.markdown("---")
    st.subheader("Current Signal Parameters")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Sample Rate", f"{sr} Hz")
    p2.metric("Base Frequency", f"{base_f:.1f} Hz")
    p3.metric("Harmonic Ratio", f"{harm_r:.2f}")
    p4.metric("Impact Rate", f"{imp_r:.1f} Hz")


# ============================================================
# TAB 2 — DATASET GENERATOR + INDUSTRY PACKS
# ============================================================

with tab2:
    st.header("📦 Dataset Generator")

    gen_tab_manual, gen_tab_packs = st.tabs([
        "Manual Generator",
        "Industry Packs"
    ])

    with gen_tab_manual:
        st.subheader("Manual Synthetic Dataset")

        batch_size = st.number_input(
            "Samples to generate",
            min_value=10,
            max_value=5000,
            value=100,
            step=50
        )

        if st.button("🚀 Generate Manual Dataset", type="primary"):
            zip_buf = io.BytesIO()
            manifest = {
                "project": project_name,
                "type": "manual_synthetic_dataset",
                "sample_rate": sr,
                "label": current_class,
                "samples": int(batch_size),
                "parameters": {
                    "base_f": base_f,
                    "harm_r": harm_r,
                    "imp_r": imp_r,
                    "noise_l": noise_l,
                }
            }

            with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
                for i in range(int(batch_size)):
                    jitter_base = max(0.0, base_f + np.random.normal(0, max(base_f * 0.02, 0.2)))
                    jitter_harm = max(0.0, harm_r + np.random.normal(0, 0.04))
                    jitter_imp = max(0.0, imp_r + np.random.normal(0, max(imp_r * 0.04, 0.1)))
                    jitter_noise = max(0.001, noise_l * np.random.uniform(0.9, 1.15))

                    d = core.generate_universal_signal(
                        2.0,
                        sr,
                        jitter_base,
                        jitter_harm,
                        jitter_imp,
                        jitter_noise
                    )

                    df = pd.DataFrame({
                        "time": d["t"],
                        "value": d["sig"]
                    })

                    zf.writestr(
                        f"{current_class}_{i:04d}.csv",
                        df.to_csv(index=False)
                    )

                zf.writestr(
                    "manifest.json",
                    json.dumps(manifest, indent=2)
                )

            st.success("Dataset klaar.")
            st.download_button(
                "📦 Download Manual Dataset ZIP",
                data=zip_buf.getvalue(),
                file_name=f"{current_class}_manual_dataset.zip",
                mime="application/zip",
                use_container_width=True
            )

    with gen_tab_packs:
        st.subheader("Industry Packs")

        pack_names = core.get_industry_packs()

        selected_pack = st.selectbox(
            "Select Industry Pack",
            pack_names
        )

        pack = core.get_industry_pack(selected_pack)

        if pack:
            st.info(pack["description"])
            st.write(f"**Sample rate:** {pack['sample_rate']} Hz")
            st.write("**Classes:**")
            st.write(", ".join(pack["classes"].keys()))

        samples_per_class = st.number_input(
            "Samples per class",
            min_value=10,
            max_value=2000,
            value=100,
            step=50
        )

        if st.button("🏭 Generate Industry Pack", type="primary"):
            files, manifest = core.generate_industry_pack_dataset(
                selected_pack,
                int(samples_per_class)
            )

            if "error" in manifest:
                st.error(manifest["error"])
            else:
                zip_data = make_zip_from_signal_files(files, manifest)

                st.success(
                    f"{manifest['total_files']} files gegenereerd voor {selected_pack}."
                )

                st.download_button(
                    "📦 Download Industry Pack ZIP",
                    data=zip_data,
                    file_name=f"{selected_pack.replace(' ', '_')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )


# ============================================================
# TAB 3 — DIGITAL TWIN AGING ENGINE
# ============================================================

with tab3:
    st.header("🧬 Digital Twin Aging Engine")
    st.caption(
        "Maak uit een gezond signaal automatisch verouderingsfases: Healthy, Wear 25, Wear 50, Wear 75 en Failure."
    )

    aging_mode = st.radio(
        "Aging source",
        [
            "Use current sliders",
            "Upload normal CSV/WAV and reverse engineer"
        ]
    )

    aging_base_params = {
        "base_f": base_f,
        "harm_r": harm_r,
        "imp_r": imp_r,
        "noise_l": noise_l
    }

    if aging_mode == "Upload normal CSV/WAV and reverse engineer":
        up_aging = st.file_uploader(
            "Upload normal/healthy signal",
            type=["csv", "wav"],
            key="aging_upload"
        )

        if up_aging:
            try:
                if up_aging.name.lower().endswith(".csv"):
                    df_up = pd.read_csv(up_aging)
                    sig = df_up.iloc[:, 1].astype(float).values
                    a_sr = sr
                else:
                    a_sr, wav_data = wavfile.read(io.BytesIO(up_aging.read()))
                    sig = (wav_data.mean(axis=1) if len(wav_data.shape) > 1 else wav_data).astype(float)

                sig = sig - np.mean(sig)
                phys = core.reverse_engineer_physics(sig, a_sr)

                aging_base_params = {
                    "base_f": phys["base_f"],
                    "harm_r": phys["harm_r"],
                    "imp_r": phys["imp_r"],
                    "noise_l": phys["noise"]
                }

                st.success("Healthy signal reverse engineered.")

                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Base Freq", f"{aging_base_params['base_f']:.1f} Hz")
                a2.metric("Harmonics", f"{aging_base_params['harm_r']:.2f}")
                a3.metric("Impacts", f"{aging_base_params['imp_r']:.1f} Hz")
                a4.metric("Noise", f"{aging_base_params['noise_l']:.3f}")

            except Exception as e:
                st.error(f"Upload analyse mislukt: {e}")

    st.markdown("---")

    stages = core.generate_aging_stages(aging_base_params)
    stage_df = pd.DataFrame([
        {
            "Stage": label,
            "Base Freq": params["base_f"],
            "Harmonics": params["harm_r"],
            "Impacts": params["imp_r"],
            "Noise": params["noise_l"],
        }
        for label, params in stages.items()
    ])

    st.subheader("Synthetic Aging Stages")
    st.dataframe(stage_df, use_container_width=True)

    samples_per_stage = st.number_input(
        "Samples per aging stage",
        min_value=10,
        max_value=2000,
        value=100,
        step=50
    )

    aging_sr = st.selectbox(
        "Aging sample rate",
        [4000, 16000],
        index=0 if sr == 4000 else 1
    )

    if st.button("🧬 Generate Digital Twin Aging Dataset", type="primary"):
        files, manifest = core.generate_predictive_maintenance_aging_dataset(
            base_params=aging_base_params,
            samples_per_stage=int(samples_per_stage),
            sr=int(aging_sr)
        )

        zip_data = make_zip_from_signal_files(files, manifest)

        st.success(
            f"{manifest['total_files']} aging files gegenereerd."
        )

        st.download_button(
            "📦 Download Digital Twin Aging Dataset",
            data=zip_data,
            file_name=f"{project_name}_digital_twin_aging.zip",
            mime="application/zip",
            use_container_width=True
        )


# ============================================================
# TAB 4 — DEEP CLONER
# ============================================================

with tab4:
    st.header("🧪 Deep Spectral Cloner")
    st.caption(
        "Upload een echt signaal. De cloner haalt dominante frequentie, harmonischen, impact-rate en noise estimate eruit."
    )

    up_clone = st.file_uploader(
        "Upload Signal",
        type=["csv", "wav"],
        key="clone_upload"
    )

    if up_clone:
        try:
            if up_clone.name.lower().endswith(".csv"):
                df_c = pd.read_csv(up_clone)
                sig_c = df_c.iloc[:, 1].astype(float).values
                actual_sr = sr
            else:
                actual_sr, wav_data = wavfile.read(io.BytesIO(up_clone.read()))
                sig_c = (wav_data.mean(axis=1) if len(wav_data.shape) > 1 else wav_data).astype(float)

            sig_c = sig_c - np.mean(sig_c)
            phys = core.reverse_engineer_physics(sig_c, actual_sr)

            st.success("DSP-profiel geëxtraheerd.")

            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            c_s1.metric("Dominant Freq", f"{phys['base_f']:.1f} Hz")
            c_s2.metric("Harmonic Energy", f"{phys['harm_r']:.2f}")
            c_s3.metric("Impact Rate", f"{phys['imp_r']:.1f} Hz")
            c_s4.metric("Noise Estimate", f"{phys['noise']:.3f}")

            if st.button("🔄 Sync Sliders", type="primary"):
                queue_param_update({
                    "base_f_slider": float(np.clip(phys["base_f"], 0.0, 1000.0)),
                    "harm_r_slider": float(np.clip(phys["harm_r"], 0.0, 2.0)),
                    "imp_r_slider": float(np.clip(phys["imp_r"], 0.0, 50.0)),
                    "noise_l_slider": float(np.clip(phys["noise"], 0.0, 1.0)),
                })
                st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")


# ============================================================
# TAB 5 — SENSOR FUSION STUDIO
# ============================================================

with tab5:
    st.header("🧩 Sensor Fusion Studio")
    st.caption(
        "Combineer audio, vibration, gas, radar en GPS/zone-context tot één fused risk/health score."
    )

    fusion_templates = core.get_fusion_templates()
    selected_template = st.selectbox(
        "Fusion scenario template",
        fusion_templates
    )

    template = core.get_fusion_template(selected_template)

    if template:
        st.info(template["description"])
        st.write(f"**Mode:** {template.get('mode', 'threat')}")
        st.write("**Weights:**")
        st.json(template["weights"])

    defaults = template.get("defaults", {}) if template else {}

    st.markdown("---")
    st.subheader("Live Fusion Inputs")

    f1, f2, f3, f4, f5 = st.columns(5)

    with f1:
        audio_score = st.slider(
            "Audio score",
            0,
            100,
            int(defaults.get("audio", 50))
        )

    with f2:
        vibration_score = st.slider(
            "Vibration score",
            0,
            100,
            int(defaults.get("vibration", 50))
        )

    with f3:
        gas_score = st.slider(
            "Gas / environmental score",
            0,
            100,
            int(defaults.get("gas", 20))
        )

    with f4:
        radar_score = st.slider(
            "Radar movement score",
            0,
            100,
            int(defaults.get("radar", 50))
        )

    with f5:
        gps_score = st.slider(
            "GPS / zone risk score",
            0,
            100,
            int(defaults.get("gps", 50))
        )

    fusion_result = core.calculate_fusion_score(
        audio_score=audio_score,
        vibration_score=vibration_score,
        gas_score=gas_score,
        radar_score=radar_score,
        gps_score=gps_score,
        template_name=selected_template
    )

    st.markdown("---")
    st.subheader("Fusion Result")

    r1, r2, r3, r4, r5 = st.columns(5)

    r1.metric("Fusion Score", f"{fusion_result['fusion_score']:.1f}%")
    r2.metric("Health Score", f"{fusion_result['health_score']:.1f}%")
    r3.metric("Confidence", f"{fusion_result['confidence']:.1f}%")
    r4.metric("Level", fusion_result["level"])
    r5.metric("Event", fusion_result["event"])

    if fusion_result["level"] == "CRITICAL":
        st.error(f"Recommended action: {fusion_result['recommended_action']}")
    elif fusion_result["level"] == "HIGH":
        st.warning(f"Recommended action: {fusion_result['recommended_action']}")
    elif fusion_result["level"] == "ELEVATED":
        st.info(f"Recommended action: {fusion_result['recommended_action']}")
    else:
        st.success(f"Recommended action: {fusion_result['recommended_action']}")

    sensor_df = pd.DataFrame([
        {"Sensor": "Audio", "Score": audio_score},
        {"Sensor": "Vibration", "Score": vibration_score},
        {"Sensor": "Gas / Environment", "Score": gas_score},
        {"Sensor": "Radar", "Score": radar_score},
        {"Sensor": "GPS / Zone", "Score": gps_score},
    ])

    fig_sensor = px.bar(
        sensor_df,
        x="Sensor",
        y="Score",
        range_y=[0, 100],
        title="Sensor Contribution Scores"
    )
    fig_sensor.update_layout(height=320)
    st.plotly_chart(fig_sensor, use_container_width=True)

    st.markdown("---")
    st.subheader("Generate Multi-Sensor Fusion Dataset")

    d1, d2 = st.columns(2)

    with d1:
        fusion_samples = st.number_input(
            "Fusion samples",
            min_value=50,
            max_value=10000,
            value=500,
            step=50
        )

    with d2:
        include_variants = st.checkbox(
            "Include scenario variants",
            value=True
        )

    if st.button("🧩 Generate Fusion Dataset", type="primary", use_container_width=True):
        fusion_df, fusion_manifest = core.generate_sensor_fusion_dataset(
            template_name=selected_template,
            samples=int(fusion_samples),
            base_audio=audio_score,
            base_vibration=vibration_score,
            base_gas=gas_score,
            base_radar=radar_score,
            base_gps=gps_score,
            include_scenario_variants=include_variants
        )

        st.session_state["last_fusion_df"] = fusion_df
        st.session_state["last_fusion_manifest"] = fusion_manifest

        st.success(f"{len(fusion_df)} fusion samples gegenereerd.")

    if "last_fusion_df" in st.session_state:
        fusion_df = st.session_state["last_fusion_df"]
        fusion_manifest = st.session_state["last_fusion_manifest"]

        st.dataframe(fusion_df.head(100), use_container_width=True)

        label_counts = fusion_df["Label"].value_counts().reset_index()
        label_counts.columns = ["Label", "Count"]

        fig_labels = px.bar(
            label_counts,
            x="Label",
            y="Count",
            title="Fusion Dataset Label Distribution"
        )
        fig_labels.update_layout(height=320)
        st.plotly_chart(fig_labels, use_container_width=True)

        zip_data = make_fusion_zip(fusion_df, fusion_manifest)

        st.download_button(
            "📦 Download Sensor Fusion Dataset ZIP",
            data=zip_data,
            file_name=f"{project_name}_sensor_fusion_dataset.zip",
            mime="application/zip",
            use_container_width=True
        )

        if st.button("➕ Add Fusion Dataset to Enterprise Audit", use_container_width=True):
            add_fusion_to_audit_dataset(fusion_df)
            st.success("Fusion dataset toegevoegd aan Enterprise Audit pipeline.")


# ============================================================
# TAB 6 — ENTERPRISE AUDIT + DATASET DOCTOR
# ============================================================

with tab6:
    st.header("🤖 Enterprise AI Intelligence")
    st.caption(
        "Upload CSV/WAV data, extraheer features, controleer ML-readiness en genereer een Edge Impulse bundle."
    )

    raw_files = st.file_uploader(
        "Upload Data (CSV/WAV)",
        type=["csv", "wav"],
        accept_multiple_files=True,
        key="audit_upload"
    )

    label_for_upload = st.selectbox(
        "Label",
        [current_class, "Anomaly", "Test", "Normal", "Fault", "Wear"]
    )

    if raw_files and st.button("Extract to Pipeline", type="primary"):
        count = extract_features_to_dataset(
            raw_files,
            label_for_upload,
            sr
        )

        if count > 0:
            st.success(f"{count} samples toegevoegd aan dataset.")
        else:
            st.warning("Geen geldige features gevonden.")

    m_df = safe_dataframe_from_dataset()

    if len(m_df) == 0:
        st.warning("Nog geen dataset in pipeline. Upload eerst data of genereer samples.")
    else:
        st.subheader("Current Feature Dataset")
        st.dataframe(m_df, use_container_width=True, height=180)

        if "Label" not in m_df.columns:
            st.error("Dataset bevat geen Label-kolom.")
        else:
            feature_cols = [c for c in m_df.columns if c != "Label"]
            X = m_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

            X = X.fillna(0)
            y = m_df["Label"]
            label_counts = y.value_counts()

            div_score, bal_score, sep_score = core.calculate_audit_scores(X, y)

            st.markdown("---")
            st.subheader("📊 AI Readiness Scores")

            s1, s2, s3, s4 = st.columns(4)
            overall = int((div_score * 0.35) + (bal_score * 0.30) + (sep_score * 0.35))

            s1.metric("Diversity", f"{div_score}%")
            s2.metric("Balance", f"{bal_score}%")
            s3.metric("Separation", f"{sep_score}%")
            s4.metric("Overall", f"{overall}%")

            st.markdown("---")
            st.subheader("🩺 AI Dataset Doctor")

            top_feats = []

            doctor = core.dataset_doctor(X, y)

            d1, d2 = st.columns([1, 2])

            with d1:
                st.metric("Doctor Overall Score", f"{doctor['overall_score']}%")

            with d2:
                for item in doctor["advice"]:
                    sev = item["severity"]
                    msg = item["message"]

                    if sev == "high":
                        st.error(msg)
                    elif sev == "medium":
                        st.warning(msg)
                    elif sev == "low":
                        st.info(msg)
                    else:
                        st.success(msg)

            st.markdown("---")
            st.subheader("🧠 Model Matrices & Permutation Importance")

            cm1, cm2, cm3 = st.columns(3)

            with cm1:
                st.write("**Feature Redundancy**")

                if len(feature_cols) > 1:
                    fig_corr = px.imshow(
                        X.corr().abs(),
                        text_auto=".1f",
                        color_continuous_scale="Blues",
                        title="Correlation"
                    )
                    fig_corr.update_layout(
                        height=360,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Meer features nodig voor correlatie.")

            rf = None

            with cm2:
                st.write("**Confusion Matrix**")

                if len(y.unique()) >= 2 and label_counts.min() >= 2:
                    try:
                        rf = RandomForestClassifier(
                            n_estimators=50,
                            random_state=42
                        ).fit(X, y)

                        pred = cross_val_predict(
                            rf,
                            X,
                            y,
                            cv=min(3, label_counts.min())
                        )

                        labels = sorted(y.unique())
                        cm = confusion_matrix(y, pred, labels=labels)

                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            color_continuous_scale="Greens",
                            x=[f"Pred: {l}" for l in labels],
                            y=[f"True: {l}" for l in labels]
                        )
                        fig_cm.update_layout(
                            height=360,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )

                        st.plotly_chart(fig_cm, use_container_width=True)

                    except Exception as e:
                        st.error(f"Confusion matrix mislukt: {e}")
                else:
                    st.warning("Minimaal 2 labels met elk 2+ samples nodig.")

            with cm3:
                st.write("**Permutation Importance**")

                if rf is not None:
                    try:
                        imp = permutation_importance(
                            rf,
                            X,
                            y,
                            n_repeats=5,
                            random_state=42
                        )

                        imp_df = pd.DataFrame({
                            "Feature": feature_cols,
                            "Importance": (imp.importances_mean * 100).round(1)
                        }).sort_values("Importance", ascending=False)

                        top_feats = list(zip(imp_df["Feature"], imp_df["Importance"]))

                        fig_imp = px.bar(
                            imp_df.sort_values("Importance", ascending=True),
                            x="Importance",
                            y="Feature",
                            orientation="h"
                        )
                        fig_imp.update_layout(
                            height=360,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )

                        st.plotly_chart(fig_imp, use_container_width=True)

                    except Exception as e:
                        st.error(f"Importance mislukt: {e}")
                else:
                    st.info("Trainbare dataset nodig.")

            st.markdown("---")
            st.subheader("⚙️ Deployment Validation")

            b_dat = build_hardware_table(len(feature_cols), sr)
            b_df = pd.DataFrame(b_dat)
            best_board = max(b_dat, key=lambda x: x["Score"])["Board"] if b_dat else "Unknown"

            fig_hw = px.bar(
                b_df,
                x="Board",
                y="Score",
                color="Score",
                text="Score",
                color_continuous_scale="RdYlGn",
                range_y=[0, 100],
                hover_data=["CPU", "Role", "Power", "Latency", "RAM"]
            )
            fig_hw.update_layout(
                height=330,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig_hw, use_container_width=True)
            st.success(f"Recommended hardware: **{best_board}**")

            st.markdown("---")
            st.subheader("📄 Enterprise Edge Impulse Bundle")

            if st.button("📦 Download Enterprise Bundle", type="primary", use_container_width=True):
                if len(y.unique()) >= 2:
                    architect = core.hardware_auto_architect(
                        num_features=len(feature_cols),
                        sr=sr,
                        target="balanced"
                    )

                    bundle = create_enterprise_bundle(
                        project_name=project_name,
                        dataset_df=m_df,
                        div_score=div_score,
                        bal_score=bal_score,
                        sep_score=sep_score,
                        top_feats=top_feats,
                        b_dat=b_dat,
                        best_board=best_board,
                        doctor=doctor,
                        architect=architect
                    )

                    st.download_button(
                        "✅ Download ZIP Bundle",
                        data=bundle,
                        file_name=f"OMEGA_Enterprise_Bundle_{st.session_state.user['username']}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                else:
                    st.error("Minimaal 2 labels nodig voor enterprise bundle.")

            if st.button("🗑️ Clear Feature Dataset"):
                st.session_state.dataset = pd.DataFrame()
                st.rerun()


# ============================================================
# TAB 7 — HARDWARE AUTO ARCHITECT V17
# ============================================================

with tab7:
    st.header("🏗️ Hardware Auto Architect V17")
    st.caption(
        "Kies automatisch hardware, vergelijk specifieke boards, of voer custom klant-hardware in."
    )

    m_df = safe_dataframe_from_dataset()

    if len(m_df) > 0 and "Label" in m_df.columns:
        num_features_default = len([c for c in m_df.columns if c != "Label"])
    else:
        num_features_default = 7

    mode = st.radio(
        "Hardware mode",
        [
            "Auto recommendation",
            "Compare selected boards",
            "Custom hardware"
        ],
        horizontal=True
    )

    st.markdown("---")

    if mode == "Auto recommendation":
        st.subheader("Automatic Node + Gateway Recommendation")

        a1, a2, a3 = st.columns(3)

        with a1:
            architect_features = st.number_input(
                "Number of features",
                min_value=1,
                max_value=256,
                value=int(num_features_default),
                key="auto_arch_features"
            )

        with a2:
            architect_sr = st.selectbox(
                "Sample rate",
                [4000, 8000, 16000, 22050, 44100],
                index=0 if sr == 4000 else 2,
                key="auto_arch_sr"
            )

        with a3:
            architect_target = st.selectbox(
                "Optimization target",
                ["balanced", "low_power", "performance", "gateway"],
                key="auto_arch_target"
            )

        architect = core.hardware_auto_architect(
            num_features=int(architect_features),
            sr=int(architect_sr),
            target=architect_target
        )

        st.success(architect["reason"])

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Recommended Board", architect["recommendation"])
        n2.metric("Node", architect.get("node_recommendation", "Unknown"))
        n3.metric("Gateway", architect.get("gateway_recommendation", "Unknown"))
        n4.metric("FFT", architect.get("fft_recommendation", 1024))

        ranking_df = pd.DataFrame(architect["ranking"])

        if not ranking_df.empty:
            st.subheader("Hardware Ranking")
            st.dataframe(ranking_df, use_container_width=True)

            fig_rank = px.bar(
                ranking_df,
                x="board",
                y="adjusted_score",
                color="adjusted_score",
                text="adjusted_score",
                hover_data=["cpu", "role", "power_class", "latency_ms", "ram_kb", "gateway_fit"],
                color_continuous_scale="RdYlGn"
            )
            fig_rank.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_rank, use_container_width=True)

        st.markdown("---")
        st.subheader("Suggested Edge Architecture")

        rec_node = architect.get("node_recommendation", architect["recommendation"])
        rec_gateway = architect.get("gateway_recommendation", "Linux Gateway / Raspberry Pi 4")

        st.write(f"**Recommended node:** {rec_node}")
        st.write(f"**Recommended gateway:** {rec_gateway}")
        st.write(f"**Sampling:** {architect_sr} Hz")
        st.write(f"**FFT:** {architect.get('fft_recommendation', 1024)}")
        st.write("**Communication:** LoRa / WiFi / BLE / MQTT afhankelijk van project.")
        st.write("**Architecture note:** Gebruik MCU-node voor lokale feature extraction en gateway voor opslag, dashboards, API en model-updates.")

    elif mode == "Compare selected boards":
        st.subheader("Compare Selected Hardware")

        all_boards = core.get_available_hardware()

        c1, c2, c3 = st.columns(3)

        with c1:
            compare_features = st.number_input(
                "Number of features",
                min_value=1,
                max_value=256,
                value=int(num_features_default),
                key="compare_features"
            )

        with c2:
            compare_sr = st.selectbox(
                "Sample rate",
                [4000, 8000, 16000, 22050, 44100],
                index=0 if sr == 4000 else 2,
                key="compare_sr"
            )

        with c3:
            compare_target = st.selectbox(
                "Optimization target",
                ["balanced", "low_power", "performance", "gateway"],
                key="compare_target"
            )

        selected_boards = st.multiselect(
            "Select boards to compare",
            all_boards,
            default=all_boards[:6]
        )

        if selected_boards:
            architect = core.hardware_auto_architect(
                num_features=int(compare_features),
                sr=int(compare_sr),
                target=compare_target,
                selected_boards=selected_boards
            )

            st.success(architect["reason"])

            ranking_df = pd.DataFrame(architect["ranking"])

            st.dataframe(ranking_df, use_container_width=True)

            fig_compare = px.bar(
                ranking_df,
                x="board",
                y="adjusted_score",
                color="adjusted_score",
                text="adjusted_score",
                hover_data=["cpu", "role", "power_class", "latency_ms", "ram_kb", "gateway_fit"],
                color_continuous_scale="RdYlGn"
            )
            fig_compare.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            st.markdown("---")
            st.subheader("Selected Hardware Advice")
            st.write(f"**Best selected board:** {architect['recommendation']}")
            st.write(f"**Node recommendation:** {architect.get('node_recommendation', 'Unknown')}")
            st.write(f"**Gateway recommendation:** {architect.get('gateway_recommendation', 'Linux Gateway / Raspberry Pi 4')}")
        else:
            st.warning("Selecteer minimaal één board.")

    else:
        st.subheader("Custom Customer Hardware")
        st.caption(
            "Gebruik dit als een klant eigen hardware heeft die niet in de database staat."
        )

        c1, c2 = st.columns(2)

        with c1:
            custom_name = st.text_input(
                "Hardware name",
                "Customer Custom MCU"
            )

            custom_ram = st.number_input(
                "Available RAM (KB)",
                min_value=16,
                max_value=8192000,
                value=320,
                step=16
            )

            custom_features = st.number_input(
                "Number of features",
                min_value=1,
                max_value=256,
                value=int(num_features_default)
            )

            custom_sr = st.selectbox(
                "Sample rate",
                [4000, 8000, 16000, 22050, 44100],
                index=0 if sr == 4000 else 2
            )

        with c2:
            st.write("Speed factors")
            st.caption("Lager = sneller. ESP32-S3 FFT factor is ongeveer 0.00008.")

            custom_fft_factor = st.number_input(
                "FFT speed factor",
                min_value=0.000001,
                max_value=0.005000,
                value=0.000100,
                step=0.000001,
                format="%.6f"
            )

            custom_feature_factor = st.number_input(
                "Feature speed factor",
                min_value=0.01,
                max_value=5.0,
                value=0.30,
                step=0.01
            )

            custom_inference_ms = st.number_input(
                "Estimated inference latency (ms)",
                min_value=0.01,
                max_value=100.0,
                value=2.0,
                step=0.1
            )

        if st.button("Evaluate Custom Hardware", type="primary", use_container_width=True):
            result = core.estimate_custom_hardware_load(
                hardware_name=custom_name,
                ram_kb_available=custom_ram,
                fft_speed_factor=custom_fft_factor,
                feature_speed_factor=custom_feature_factor,
                inference_ms=custom_inference_ms,
                feat_n=int(custom_features),
                sr=int(custom_sr)
            )

            st.markdown("---")
            st.subheader("Custom Hardware Result")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Score", f"{result['score']:.1f}%")
            r2.metric("Verdict", result["verdict"])
            r3.metric("Latency", f"{result['latency_ms']:.1f} ms")
            r4.metric("RAM Fit", "Yes" if result["fits_ram"] else "No")

            st.dataframe(pd.DataFrame([result]), use_container_width=True)

            if result["score"] >= 85:
                st.success("Deze hardware lijkt uitstekend geschikt voor deze workload.")
            elif result["score"] >= 70:
                st.info("Deze hardware lijkt geschikt, maar test latency en RAM op echte hardware.")
            elif result["score"] >= 50:
                st.warning("Deze hardware is bruikbaar voor prototype, maar mogelijk niet productiegeschikt.")
            else:
                st.error("Deze hardware wordt niet aanbevolen voor deze workload.")

    st.markdown("---")
    st.subheader("Hardware Catalog")

    if hasattr(core, "get_hardware_catalog"):
        catalog_df = core.get_hardware_catalog()
        st.dataframe(catalog_df, use_container_width=True)
    else:
        st.info("Hardware catalog niet beschikbaar in core.py.")
