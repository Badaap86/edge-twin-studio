import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import scipy.signal as signal
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

warnings.filterwarnings('ignore')
st.set_page_config(page_title="OMEGA-X SaaS Studio", layout="wide")
st.title("⚡ OMEGA-X SaaS Studio (V14.0)")

# Initialize Database
core.init_db()

# ============================================================
# SaaS PROJECT MANAGEMENT
# ============================================================
if "dataset" not in st.session_state: st.session_state.dataset = pd.DataFrame()
if "project_id" not in st.session_state: st.session_state.project_id = "proj_" + str(np.random.randint(1000, 9999))

with st.sidebar:
    st.header("🗂️ Cloud Projects (SQLite)")
    proj_name = st.text_input("Project Name", "New_Enterprise_AI")
    
    col_a, col_b = st.columns(2)
    if col_a.button("💾 Save to DB"):
        setts = {"base_f": st.session_state.get("base_f_slider", 50.0)} # Simplified for brevity
        core.save_project(st.session_state.project_id, proj_name, st.session_state.dataset, setts)
        st.success("Saved!")
        
    db_projects = core.get_all_projects()
    if not db_projects.empty:
        sel_proj = st.selectbox("Load Project", db_projects["name"].tolist())
        if col_b.button("📂 Load"):
            proj_id = db_projects[db_projects["name"] == sel_proj].iloc[0]["id"]
            loaded = core.load_project(proj_id)
            if loaded:
                st.session_state.project_id = proj_id
                st.session_state.dataset = loaded["dataset"]
                st.success(f"Loaded {sel_proj}!")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Studio parameters")
modality = st.sidebar.radio("Data Profiling", ["Vibration (4 kHz)", "Audio (16 kHz)"])
sr = 4000 if "Vibration" in modality else 16000
current_class = st.sidebar.text_input("Dataset Label", "Baseline_Normal")

base_f = st.sidebar.slider("Base Freq (Hz)", 0.0, 1000.0, st.session_state.get("base_f_slider", 50.0), key="base_f_slider")
harm_r = st.sidebar.slider("Harmonics", 0.0, 2.0, st.session_state.get("harm_r_slider", 0.0), key="harm_r_slider")
imp_r = st.sidebar.slider("Impacts (Hz)", 0.0, 50.0, st.session_state.get("imp_r_slider", 0.0), key="imp_r_slider")
noise_l = st.sidebar.slider("Noise", 0.0, 1.0, st.session_state.get("noise_l_slider", 0.1), key="noise_l_slider")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Canvas", "📦 Generator", "🧪 Cloner", "🤖 ML Enterprise Audit"])

with tab1:
    d_live = core.generate_universal_signal(2.0, sr, base_f, harm_r, imp_r, noise_l)
    c1, c2 = st.columns(2)
    c1.plotly_chart(go.Figure(go.Scatter(x=d_live["t"][:2000], y=d_live["sig"][:2000])).update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0)), use_container_width=True)
    c2.plotly_chart(go.Figure(go.Scatter(x=d_live["fft_f"], y=d_live["fft_v"])).update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0)), use_container_width=True)

with tab2:
    st.header("📦 Batch Generator")
    b_size = st.number_input("Samples", 10, 5000, 100, 50)
    if st.button("🚀 Export Dataset"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "a", zipfile.ZIP_DEFLATED) as zf:
            for i in range(b_size):
                d = core.generate_universal_signal(2.0, sr, base_f, harm_r, imp_r, noise_l)
                zf.writestr(f"{current_class}_{i:04d}.csv", pd.DataFrame({'t': d["t"], 'v': d["sig"]}).to_csv(index=False))
        st.download_button("📦 Download .ZIP", data=buf.getvalue(), file_name=f"{current_class}.zip", mime="application/zip")

with tab3:
    st.header("🧪 Deep Spectral Cloner")
    up = st.file_uploader("Upload Signal", type=['csv', 'wav'])
    if up:
        try:
            a_sr = sr 
            if up.name.endswith('.csv'): s_c = pd.read_csv(up).iloc[:, 1].values
            else:
                a_sr, w_d = wavfile.read(io.BytesIO(up.read()))
                s_c = w_d.mean(axis=1) if len(w_d.shape) > 1 else w_d
            
            s_c = s_c.astype(float) - np.mean(s_c.astype(float))
            # FIX: Original Deep Physics Extraction
            phys = core.reverse_engineer_physics(s_c, a_sr)
            
            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            c_s1.metric("Dominant Freq", f"{phys['base_f']:.1f} Hz")
            c_s2.metric("Harmonic Energy", f"{phys['harm_r']:.2f}")
            c_s3.metric("Impact Rate", f"{phys['imp_r']:.1f} Hz")
            c_s4.metric("Noise Estimate", f"{phys['noise']:.3f}")
            
            if st.button("🔄 Sync Sliders"):
                st.session_state.base_f_slider = float(np.clip(phys['base_f'], 0.0, 1000.0))
                st.session_state.harm_r_slider = float(np.clip(phys['harm_r'], 0.0, 2.0))
                st.session_state.imp_r_slider = float(np.clip(phys['imp_r'], 0.0, 50.0))
                st.session_state.noise_l_slider = float(np.clip(phys['noise'], 0.0, 1.0))
                st.rerun()
        except Exception as e: st.error(f"Error: {e}")

with tab4:
    st.header("🤖 Enterprise AI Intelligence")
    raw_f = st.file_uploader("Upload Data (CSV/WAV)", type=["csv", "wav"], accept_multiple_files=True)
    l_box = st.selectbox("Label", [current_class, "Anomaly", "Test"])
    
    if raw_f and st.button("Extract", type="primary"):
        rows = []
        for f in raw_f:
            feats = core.extract_features_from_bytes(f.read(), f.name, sr)
            if "error" not in feats:
                feats["Label"] = l_box
                rows.append(feats)
        if rows: st.session_state.dataset = pd.concat([st.session_state.dataset, pd.DataFrame(rows)], ignore_index=True)

    m_df = st.session_state.dataset
    if len(m_df) > 0:
        f_cols = [c for c in m_df.columns if c != "Label"]
        X = m_df[f_cols].replace([np.inf, -np.inf], 0).fillna(0)
        y = m_df["Label"]
        v_c = y.value_counts()
        
        # --- HEALTH ---
        st.subheader("📊 Dataset Health")
        col1, col2, col3 = st.columns(3)
        div_score, bal_score, sep_score = core.calculate_audit_scores(X, y)
        col1.metric("Diversity", f"{div_score}%")
        col2.metric("Balance", f"{bal_score}%")
        col3.metric("Separation", f"{sep_score}%")
        st.markdown("---")
        
        # --- MULTI-BOARD ---
        st.subheader("⚙️ Deployment Validation")
        brds = ["ESP32-S3", "STM32L4", "RAK4631", "Cortex-M0+"]
        b_dat = []
        for b in brds:
            ram, l_fft, l_feat, l_inf = core.estimate_edge_load(b, len(f_cols), sr)
            tot_l = l_fft + l_feat + l_inf
            b_dat.append({"Board": b, "Score": core.calculate_deployment_score(b, tot_l, ram), "Latency": tot_l, "RAM": ram})
        
        st.plotly_chart(px.bar(pd.DataFrame(b_dat), x="Board", y="Score", color="Score", text="Score", color_continuous_scale="RdYlGn", range_y=[0,100]).update_layout(height=250), use_container_width=True)
        best_board = max(b_dat, key=lambda x: x['Score'])['Board'] if b_dat else "Unknown"

        st.markdown("---")

        # --- MATRICES & PERMUTATION ---
        st.subheader("🧠 Model Matrices & Permutation")
        cm1, cm2, cm3 = st.columns(3)
        top_feats = []
        
        with cm1: st.plotly_chart(px.imshow(X.corr().abs(), text_auto=".1f", color_continuous_scale="Blues", title="Feature Redundancy").update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0)), use_container_width=True)
            
        with cm2:
            if len(y.unique()) >= 2 and v_c.min() >= 2:
                rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
                pred = cross_val_predict(rf, X, y, cv=min(3, v_c.min()))
                st.plotly_chart(px.imshow(confusion_matrix(y, pred), text_auto=True, color_continuous_scale="Greens", title="Confusion Matrix").update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0)), use_container_width=True)
            else: st.warning("Minimaal 2 klassen met 2+ samples nodig.")

        with cm3:
            if len(y.unique()) >= 2 and v_c.min() >= 2:
                imp = permutation_importance(rf, X, y, n_repeats=5, random_state=42)
                imp_df = pd.DataFrame({"F": f_cols, "Imp": (imp.importances_mean * 100).round(1)}).sort_values("Imp", ascending=False)
                top_feats = list(zip(imp_df["F"], imp_df["Imp"]))
                st.plotly_chart(px.bar(imp_df.sort_values("Imp", ascending=True), x="Imp", y="F", orientation='h', title="Permutation Importance").update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0)), use_container_width=True)

        # --- ULTIMATE BUNDLE EXPORT ---
        st.markdown("---")
        st.subheader("📄 Enterprise Export Bundle")

        if st.button("📦 Download Edge Impulse Bundle", type="primary", use_container_width=True):
            if len(y.unique()) >= 2:
                z_buf = io.BytesIO()
                with zipfile.ZipFile(z_buf, "a", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("edge_dataset.csv", m_df.to_csv(index=False))
                    pdf_bytes = core.generate_pdf_report(proj_name, len(X), len(y.unique()), div_score, bal_score, sep_score, top_feats, b_dat, best_board)
                    zf.writestr("audit_report.pdf", pdf_bytes)
                    
                    meta = {
                        "project": proj_name, "features": f_cols,
                        "metrics": {"diversity": div_score, "balance": bal_score, "separation": sep_score},
                        "hardware_recommendation": best_board,
                        "top_features": [{"feature": f, "importance": s} for f, s in top_feats[:5]]
                    }
                    zf.writestr("metadata.json", json.dumps(meta, indent=2))
                st.download_button("✅ Download Bundle (.ZIP)", data=z_buf.getvalue(), file_name=f"OMEGA_Bundle_{proj_name}.zip", mime="application/zip", use_container_width=True)
            else: st.error("Kan Bundle niet genereren zonder getraind model.")
