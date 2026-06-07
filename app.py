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
st.set_page_config(page_title="OMEGA-X Enterprise Studio", layout="wide")
st.title("⚡ OMEGA-X Enterprise Studio (V13.0)")

# ============================================================
# STATE & UI
# ============================================================
if "dataset" not in st.session_state: st.session_state.dataset = pd.DataFrame()
if "project_name" not in st.session_state: st.session_state.project_name = "Enterprise_AI_V1"

st.sidebar.header("🗂️ Workspace")
st.session_state.project_name = st.sidebar.text_input("Project Name", st.session_state.project_name)

st.sidebar.header("🎛️ Studio parameters")
modality = st.sidebar.radio("Data Profiling", ["Vibration (4 kHz)", "Audio (16 kHz)"])
sr = 4000 if "Vibration" in modality else 16000
current_class = st.sidebar.text_input("Dataset Label", "Baseline_Normal")

# FIX: Sliders hebben nu expliciete keys voor state-syncing
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
    st.header("🧪 Reverse Engineer")
    up = st.file_uploader("Upload Signal", type=['csv', 'wav'])
    if up:
        try:
            feats = core.extract_features_from_bytes(up.read(), up.name, sr)
            if "error" not in feats:
                st.success("DSP geëxtraheerd.")
                c_s1, c_s2, c_s3, c_s4 = st.columns(4)
                
                # Approximate values back to sliders (Simplified mapping)
                ext_base = feats.get("SpectralCentroid", 50.0) / 2.0 
                ext_harm = feats.get("SpectralFlatness", 0.0) * 2.0
                impact_rate = feats.get("ZCR", 0.0) * 100
                noise_est = feats.get("RMS", 0.1)

                c_s1.metric("Est. Base Freq", f"{ext_base:.1f} Hz")
                c_s2.metric("Harmonic Indicator", f"{ext_harm:.2f}")
                c_s3.metric("Impact/ZCR", f"{impact_rate:.1f} Hz")
                c_s4.metric("RMS/Noise", f"{noise_est:.3f}")
                
                if st.button("🔄 Sync Sliders"):
                    st.session_state.base_f_slider = float(np.clip(ext_base, 0.0, 1000.0))
                    st.session_state.harm_r_slider = float(np.clip(ext_harm, 0.0, 2.0))
                    st.session_state.imp_r_slider = float(np.clip(impact_rate, 0.0, 50.0))
                    st.session_state.noise_l_slider = float(np.clip(noise_est, 0.0, 1.0))
                    st.rerun()
            else: st.error("Extractie fout.")
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
        
        # --- 1. HEALTH ---
        st.subheader("📊 Dataset Health")
        col1, col2, col3 = st.columns(3)
        div_score, bal_score, sep_score = core.calculate_audit_scores(X, y)
        col1.metric("Diversity", f"{div_score}%")
        col2.metric("Balance", f"{bal_score}%")
        col3.metric("Separation", f"{sep_score}%")
        st.markdown("---")
        
        # --- 2. MULTI-BOARD ---
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

        # --- 3. MATRICES & PERMUTATION ---
        st.subheader("🧠 Model Matrices & Permutation")
        cm1, cm2, cm3 = st.columns(3)
        
        top_feats = []
        with cm1:
            st.plotly_chart(px.imshow(X.corr().abs(), text_auto=".1f", color_continuous_scale="Blues", title="Feature Redundancy").update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0)), use_container_width=True)
            
        with cm2:
            if len(y.unique()) >= 2 and v_c.min() >= 2:
                rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
                pred = cross_val_predict(rf, X, y, cv=min(3, v_c.min()))
                st.plotly_chart(px.imshow(confusion_matrix(y, pred), text_auto=True, color_continuous_scale="Greens", title="Confusion Matrix").update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0)), use_container_width=True)
            else: st.warning("Voeg data toe voor Confusion Matrix")

        with cm3:
            if len(y.unique()) >= 2 and v_c.min() >= 2:
                imp = permutation_importance(rf, X, y, n_repeats=5, random_state=42)
                imp_df = pd.DataFrame({"F": f_cols, "Imp": (imp.importances_mean * 100).round(1)}).sort_values("Imp", ascending=False)
                top_feats = list(zip(imp_df["F"], imp_df["Imp"]))
                st.plotly_chart(px.bar(imp_df.sort_values("Imp", ascending=True), x="Imp", y="F", orientation='h', title="Permutation Importance").update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0)), use_container_width=True)

        # --- 4. ULTIMATE BUNDLE EXPORT ---
        st.markdown("---")
        st.subheader("📄 Enterprise Export Bundle")

        if st.button("📦 Download Edge Impulse Bundle", type="primary", use_container_width=True):
            if len(y.unique()) >= 2:
                z_buf = io.BytesIO()
                with zipfile.ZipFile(z_buf, "a", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("edge_dataset.csv", m_df.to_csv(index=False))
                    pdf_bytes = core.generate_pdf_report(st.session_state.project_name, len(X), len(y.unique()), div_score, bal_score, sep_score, top_feats, b_dat, best_board)
                    zf.writestr("audit_report.pdf", pdf_bytes)
                    
                    meta = {
                        "project": st.session_state.project_name,
                        "features": f_cols,
                        "metrics": {"diversity_score": div_score, "balance_score": bal_score, "separation_score": sep_score},
                        "hardware_recommendation": best_board,
                        "top_features": [{"feature": f, "importance": s} for f, s in top_feats[:5]]
                    }
                    zf.writestr("metadata.json", json.dumps(meta, indent=2))
                st.download_button("✅ Download Bundle (.ZIP)", data=z_buf.getvalue(), file_name=f"OMEGA_Bundle_{st.session_state.project_name}.zip", mime="application/zip", use_container_width=True)
            else: st.error("Kan Bundle niet genereren zonder getraind model (minimaal 2 klassen nodig).")
