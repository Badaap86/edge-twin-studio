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
from scipy.stats import kurtosis
from scipy.io import wavfile
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings('ignore')

# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(page_title="OMEGA-X Enterprise Studio", layout="wide", initial_sidebar_state="expanded")
st.title("⚡ OMEGA-X Enterprise Studio (V10.0)")
st.caption("Universal Synthesis • Edge Profiling • Advanced ML Intelligence")

# ============================================================
# REST-API READY CORE FUNCTIONS
# ============================================================
def calculate_fft(signal_data, sample_rate):
    window = np.hanning(len(signal_data))
    fft_values = np.abs(np.fft.rfft(signal_data * window))
    fft_freqs = np.fft.rfftfreq(len(signal_data), 1 / sample_rate)
    return fft_freqs, fft_values

def calculate_spectrogram(signal_data, sample_rate):
    f_spec, t_spec, Sxx = signal.spectrogram(signal_data, sample_rate, nperseg=256, noverlap=128)
    return f_spec, t_spec, Sxx

def get_audio_features(sig, f_f, v_f):
    eps = 1e-10 
    zcr = ((sig[:-1] * sig[1:]) < 0).sum() / len(sig)
    centroid = np.sum(f_f * v_f) / (np.sum(v_f) + eps)
    
    cum_energy = np.cumsum(v_f)
    tot_energy = cum_energy[-1]
    rolloff = 0
    if tot_energy > 0:
        rolloff_idx = np.where(cum_energy >= 0.85 * tot_energy)[0][0]
        rolloff = f_f[rolloff_idx]
        
    geom_mean = np.exp(np.mean(np.log(v_f + eps)))
    arith_mean = np.mean(v_f) + eps
    flatness = geom_mean / arith_mean
    
    return zcr, centroid, rolloff, flatness

def generate_universal_signal(duration, sample_rate, base_freq, harmonic_ratio, impact_rate, noise_level, normalize=True):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sig = np.zeros_like(t)
    
    if base_freq > 0:
        sig += np.sin(2 * np.pi * base_freq * t)
    if harmonic_ratio > 0:
        for h in range(2, 6):
            sig += (harmonic_ratio / h) * np.sin(2 * np.pi * (base_freq * h) * t)
    if impact_rate > 0:
        impact_interval = 1.0 / impact_rate
        for i_t in np.arange(0, duration, impact_interval):
            jitter = np.random.uniform(-0.02, 0.02)
            idx = int((i_t + jitter) * sample_rate)
            if 0 <= idx < len(t):
                decay = np.exp(-25 * (t[idx:] - i_t))
                burst = np.random.normal(0, 1, len(decay)) * decay
                sig[idx:] += 2.5 * burst
                
    sig += np.random.normal(0, noise_level, len(t))
    if normalize:
        max_val = np.max(np.abs(sig))
        if max_val > 0: sig = sig / max_val
            
    f, v = calculate_fft(sig, sample_rate)
    return {"t": t, "sig": sig, "fft_f": f, "fft_v": v, "sr": sample_rate}

def estimate_edge_load(hw_choice, num_features, sample_rate, duration=1.0):
    window_size = min(int(sample_rate * duration), 8192)
    fft_n = 1024 if sample_rate <= 4000 else 2048
    ram_buffer_kb = (window_size * 4) / 1024 
    ram_fft_kb = (fft_n * 4 * 2) / 1024 
    ram_total_kb = ram_buffer_kb + ram_fft_kb + 2.0 
    
    if "ESP32-S3" in hw_choice: return "Xtensa LX7 (SIMD)", ram_total_kb, (fft_n * np.log2(fft_n)) * 0.00008, num_features * 0.2, 1.5
    elif "WisBlock" in hw_choice: return "ARM Cortex-M4F", ram_total_kb, (fft_n * np.log2(fft_n)) * 0.00015, num_features * 0.4, 3.0
    else: return "ARM Cortex-M0+", ram_total_kb, (fft_n * np.log2(fft_n)) * 0.0008, num_features * 1.5, 12.0

def calculate_deployment_score(hw_choice, latency, ram_kb, target_latency=20.0, target_ram=120.0):
    lat_score = max(0, 100 - (latency / target_latency) * 50)
    ram_score = max(0, 100 - (ram_kb / target_ram) * 50)
    base_score = (lat_score * 0.7) + (ram_score * 0.3)
    
    if "ESP32-S3" in hw_choice: return min(100, base_score + 15)
    elif "WisBlock" in hw_choice: return min(100, base_score + 5)
    else: return max(0, base_score - 20)
# ============================================================
# STATE & WORKSPACE INITIALIZATION
# ============================================================
if "hw_target_val" not in st.session_state: st.session_state.hw_target_val = "ESP32-S3 (LilyGO / Vector AI)"
if "modality_val" not in st.session_state: st.session_state.modality_val = "Vibration / IMU (4 kHz)"
if "normalize_val" not in st.session_state: st.session_state.normalize_val = True

if "base_f_slider" not in st.session_state: st.session_state.base_f_slider = 50.0
if "harm_r_slider" not in st.session_state: st.session_state.harm_r_slider = 0.0
if "imp_r_slider" not in st.session_state: st.session_state.imp_r_slider = 0.0
if "noise_l_slider" not in st.session_state: st.session_state.noise_l_slider = 0.1

if "master_training_dataset" not in st.session_state: st.session_state.master_training_dataset = pd.DataFrame()
if "project_name" not in st.session_state: st.session_state.project_name = "Forest_Guardian_V1"

st.sidebar.header("🗂️ Workspace")
st.session_state.project_name = st.sidebar.text_input("Project Name", st.session_state.project_name)

# --- SAVE WORKSPACE ---
project_state = {
    "version": "OMEGA-X V10.0",
    "project_name": st.session_state.project_name,
    "global_settings": {"hardware_target": st.session_state.hw_target_val, "modality": st.session_state.modality_val, "normalize": st.session_state.normalize_val},
    "sliders": {"base_f": st.session_state.base_f_slider, "harm_r": st.session_state.harm_r_slider, "imp_r": st.session_state.imp_r_slider, "noise_l": st.session_state.noise_l_slider},
    "dataset": st.session_state.master_training_dataset.to_dict(orient="records")
}
st.sidebar.download_button("💾 Save Workspace (.json)", data=json.dumps(project_state, indent=4), file_name=f"omega_ws_{st.session_state.project_name}.json", mime="application/json", use_container_width=True)

# --- LOAD WORKSPACE ---
uploaded_ws = st.sidebar.file_uploader("📂 Load Workspace", type=["json"])
if uploaded_ws:
    try:
        loaded = json.load(uploaded_ws)
        st.session_state.project_name = loaded.get("project_name", "Loaded_Project")
        st.session_state.hw_target_val = loaded.get("global_settings", {}).get("hardware_target", "ESP32-S3 (LilyGO / Vector AI)")
        st.session_state.modality_val = loaded.get("global_settings", {}).get("modality", "Vibration / IMU (4 kHz)")
        st.session_state.normalize_val = loaded.get("global_settings", {}).get("normalize", True)
        st.session_state.base_f_slider = loaded["sliders"]["base_f"]
        st.session_state.harm_r_slider = loaded["sliders"]["harm_r"]
        st.session_state.imp_r_slider = loaded["sliders"]["imp_r"]
        st.session_state.noise_l_slider = loaded["sliders"]["noise_l"]
        st.session_state.master_training_dataset = pd.DataFrame(loaded.get("dataset", []))
        st.sidebar.success("✅ Workspace Loaded!")
    except Exception as e: st.sidebar.error("Invalid Workspace")

st.sidebar.markdown("---")
hardware_target = st.sidebar.selectbox("Deployment Board", ["ESP32-S3 (LilyGO / Vector AI)", "nRF52840 (RAK WisBlock M4F)", "Generic Cortex-M0+"], key="hw_target_val")
st.sidebar.markdown("---")
modality = st.sidebar.radio("Data Profiling", ["Vibration / IMU (4 kHz)", "Acoustic / Audio (16 kHz)"], key="modality_val")
sr = 4000 if "Vibration" in modality else 16000

current_class = st.sidebar.text_input("Dataset Label", "Baseline_Normal")
base_f = st.sidebar.slider("Base Frequency (Hz)", 0.0, 1000.0, key="base_f_slider", step=5.0)
harm_r = st.sidebar.slider("Harmonic Ratio", 0.0, 2.0, key="harm_r_slider", step=0.05)
imp_r = st.sidebar.slider("Transient Impact Rate (Hz)", 0.0, 50.0, key="imp_r_slider", step=0.5)
noise_l = st.sidebar.slider("Noise Floor (SNR)", 0.0, 1.0, key="noise_l_slider", step=0.02)
do_normalize = st.sidebar.checkbox("Normalize Amplitude", key="normalize_val")

data_live = generate_universal_signal(duration=2.0, sample_rate=sr, base_freq=base_f, harmonic_ratio=harm_r, impact_rate=imp_r, noise_level=noise_l, normalize=do_normalize)
# ============================================================
# TABS 1, 2, 3
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 1. Studio Canvas", "📦 2. Batch Generator", "🧪 3. Reverse Engineer", "🤖 4. AutoML & Deployment"])

with tab1:
    st.header(f"Live DSP Canvas: `{current_class}`")
    c_sig, c_fft = st.columns(2)
    fig_sig = go.Figure(go.Scatter(x=data_live["t"][:2000], y=data_live["sig"][:2000], mode="lines", line=dict(color='#1f77b4')))
    c_sig.plotly_chart(fig_sig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0)), use_container_width=True)
    fig_fft = go.Figure(go.Scatter(x=data_live["fft_f"], y=data_live["fft_v"], mode="lines", line=dict(color='#1f77b4')))
    c_fft.plotly_chart(fig_fft.update_layout(height=300, xaxis_range=[0, 1500 if sr == 4000 else 4000], margin=dict(l=0, r=0, t=30, b=0)), use_container_width=True)

with tab2:
    st.header(f"📦 Audit-Proof Dataset Production")
    batch_size = st.number_input("Samples to generate", 10, 5000, 100, 50)
    if st.button(f"🚀 Produce '{current_class}' Dataset", type="primary"):
        my_bar = st.progress(0, text="Generating API-ready batch...")
        zip_buf = io.BytesIO()
        manifest = {"project_name": st.session_state.project_name, "generator_version": "OMEGA-X 10.0", "global_settings": {"sample_rate": sr, "duration_seconds": 2.0, "normalization_applied": do_normalize}, "files": []}
        
        with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zipf:
            for i in range(batch_size):
                j_b, j_h, j_i, j_n = max(0, base_f + np.random.normal(0, base_f*0.02)), max(0, harm_r + np.random.normal(0, 0.05)), max(0, imp_r + np.random.normal(0, imp_r*0.05)), max(0.001, noise_l * np.random.uniform(0.9, 1.1))
                d = generate_universal_signal(2.0, sr, j_b, j_h, j_i, j_n, do_normalize)
                fname = f"{current_class.lower()}_{i:04d}.csv"
                zipf.writestr(fname, pd.DataFrame({'time': d["t"], 'value': d["sig"]}).to_csv(index=False))
                manifest["files"].append({"filename": fname, "label": current_class, "parameters": {"base_freq": round(j_b, 2), "harmonic_ratio": round(j_h, 3), "impact_rate": round(j_i, 2), "noise_floor": round(j_n, 3)}})
                my_bar.progress((i + 1) / batch_size)
            zipf.writestr("dataset_manifest.json", json.dumps(manifest, indent=4))
        my_bar.empty()
        st.success(f"✅ Download Ready! ({batch_size} files + manifest.json)")
        st.download_button("📦 Download .ZIP", data=zip_buf.getvalue(), file_name=f"dataset_{current_class}.zip", mime="application/zip")

with tab3:
    st.header("🧪 Reverse Engineer (Deep Spectral Cloner)")
    up_clone = st.file_uploader("Upload Signal", type=['csv', 'wav'], key="clone_up")
    if up_clone:
        try:
            actual_sr = sr 
            if up_clone.name.endswith('.csv'):
                sig_c = pd.read_csv(up_clone).iloc[:, 1].values
            elif up_clone.name.endswith('.wav'):
                actual_sr, wav_data = wavfile.read(io.BytesIO(up_clone.read()))
                sig_c = wav_data.mean(axis=1) if len(wav_data.shape) > 1 else wav_data
                sig_c = sig_c.astype(float)
                if np.max(np.abs(sig_c)) > 0: sig_c = sig_c / np.max(np.abs(sig_c)) 
            
            sig_c = sig_c - np.mean(sig_c)
            f_c, v_c = calculate_fft(sig_c, actual_sr)
            dom_idx = np.argmax(v_c[1:]) + 1
            ext_base = f_c[dom_idx]
            ext_harm = sum(v_c[np.argmin(np.abs(f_c - (h * ext_base)))] for h in range(2, 6)) / max(v_c[dom_idx], 1e-6)
            peaks, _ = signal.find_peaks(np.abs(sig_c), height=np.mean(np.abs(sig_c)) + 2.5 * np.std(sig_c), distance=actual_sr/100)
            
            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            c_s1.metric("Detected Base Freq", f"{ext_base:.1f} Hz")
            c_s2.metric("Harmonic Energy", f"{ext_harm:.2f}")
            c_s3.metric("Impact Rate", f"{len(peaks) / (len(sig_c) / actual_sr):.1f} Hz")
            c_s4.metric("Noise Estimate", f"{np.median(np.abs(sig_c)):.3f}")

            if st.button("🔄 Sync to Studio Sliders", type="primary"):
                st.session_state.base_f_slider = float(np.clip(ext_base, 0.0, 1000.0))
                st.session_state.harm_r_slider = float(np.clip(ext_harm, 0.0, 2.0))
                st.session_state.imp_r_slider = float(np.clip(len(peaks) / (len(sig_c) / actual_sr), 0.0, 50.0))
                st.session_state.noise_l_slider = float(np.clip(np.median(np.abs(sig_c)), 0.0, 1.0))
                st.rerun()
        except Exception as e: st.error(f"Error: {e}")
# ============================================================
# TAB 4: AUTOML, DEPLOYMENT & ENTERPRISE AUDIT (V10.0)
# ============================================================
with tab4:
    st.header("🤖 Enterprise AI Intelligence & Deployment")

    raw_files = st.file_uploader("Upload Data (CSV/WAV) voor Extractie", type=["csv", "wav"], accept_multiple_files=True)
    lbl = st.selectbox("Wijs Label toe", [current_class, "Anomaly", "Baseline", "Test_Class"])
    
    if raw_files and st.button("Extract & Add to Pipeline", type="primary"):
        rows = []
        for file in raw_files:
            try:
                actual_sr = sr
                if file.name.endswith('.csv'):
                    sig = pd.read_csv(file).iloc[:, 1].astype(float).values
                elif file.name.endswith('.wav'):
                    actual_sr, wav_data = wavfile.read(io.BytesIO(file.read()))
                    sig = (wav_data.mean(axis=1) if len(wav_data.shape) > 1 else wav_data).astype(float)
                
                f_f, v_f = calculate_fft(sig - np.mean(sig), actual_sr)
                rms_val = np.sqrt(np.mean(sig**2))
                zcr, centroid, rolloff, flatness = get_audio_features(sig, f_f, v_f)
                
                rows.append({"Label": lbl, "RMS": rms_val, "Kurtosis": kurtosis(sig), "CrestFactor": np.max(np.abs(sig)) / max(rms_val, 1e-6), "ZCR": zcr, "SpectralCentroid": centroid, "SpectralRolloff": rolloff, "SpectralFlatness": flatness})
            except Exception: pass
        if rows:
            st.session_state.master_training_dataset = pd.concat([st.session_state.master_training_dataset, pd.DataFrame(rows)], ignore_index=True)
            st.success(f"Geëxtraheerd: {len(rows)} signatures!")

    st.divider()
    m_df = st.session_state.master_training_dataset
    
    if len(m_df) > 0:
        feature_cols = [col for col in m_df.columns if col != "Label"]
        X = m_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
        y = m_df["Label"]
        lbl_cnts = y.value_counts()
        
        # --- 1. ENTERPRISE STATUS DASHBOARD ---
        st.subheader("📊 Dataset Readiness Audit")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        div_score = min(100, int((np.mean(pdist(X_scaled)) / 4.0) * 100)) if len(X_scaled) > 1 else 0
        bal_score = 100 if len(lbl_cnts) >= 2 and (lbl_cnts.min() / lbl_cnts.max()) > 0.5 else 50
        sep_score = int((silhouette_score(X_scaled, y) + 1) * 50) if len(y.unique()) >= 2 and len(X) > 5 else 0
        
        mcu, ram_req, ms_fft, ms_feat, ms_inf = estimate_edge_load(hardware_target, len(feature_cols), sr)
        total_lat = ms_fft + ms_feat + ms_inf
        deploy_score = calculate_deployment_score(hardware_target, total_lat, ram_req)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Dataset Diversity", f"{div_score}%")
        col_m2.metric("Dataset Balance", f"{bal_score}%")
        col_m3.metric("Label Separation", f"{sep_score}%")
        col_m4.metric("Deployment Readiness", f"{deploy_score}%")
        
        overall_ready = all(s > 80 for s in [div_score, bal_score, sep_score, deploy_score])
        st.info(f"**Overall Status:** {'🟢 PRODUCTION READY' if overall_ready else '🟡 OPTIMALISATIE VEREIST'}")
        
        c_i1, c_i2 = st.columns([1, 1])
        
        # --- 2. AUTO FEATURE PRUNING ---
        with c_i1:
            st.subheader("✂️ Auto Feature Pruning")
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            
            st.write(f"**Current Feature Set:** {len(feature_cols)} features")
            if to_drop:
                st.warning(f"Overtollige/Redundante features: `{', '.join(to_drop)}`")
                _, _, _, ms_feat_new, _ = estimate_edge_load(hardware_target, len(feature_cols) - len(to_drop), sr)
                lat_reduc_pct = ((ms_feat - ms_feat_new) / total_lat) * 100 if total_lat > 0 else 0
                st.success(f"**Advies:** Verwijder deze uit C++. Latency reductie: **{lat_reduc_pct:.1f}%**")
            else:
                st.success("🟢 Feature set is wiskundig optimaal.")

        # --- 3. HARDWARE PROFILER ---
        with c_i2:
            st.subheader("⚙️ Edge Hardware Profiler")
            st.markdown(f"**Target MCU:** `{mcu}`")
            st.write(f"- RAM Allocatie: **{ram_req:.1f} KB**")
            st.write(f"- FFT Processing: **{ms_fft:.2f} ms**")
            st.write(f"- Feature Extraction: **{ms_feat:.1f} ms**")
            st.write(f"- ML Inference: **{ms_inf:.1f} ms**")
            st.progress(deploy_score / 100.0, text=f"Deployment Totaal Score: {deploy_score}%")

        st.markdown("---")

        # --- 4. PERMUTATION IMPORTANCE & CONFUSION MATRIX ---
        if len(lbl_cnts) >= 2 and all(count >= 5 for count in lbl_cnts):
            c_ml1, c_ml2 = st.columns([1, 1])
            
            try:
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                rf.fit(X, y)
                
                with c_ml1:
                    st.subheader("🎯 Feature Importance (Permutation)")
                    perm_imp = permutation_importance(rf, X, y, n_repeats=5, random_state=42)
                    imp_df = pd.DataFrame({"Feature": feature_cols, "Belang": (perm_imp.importances_mean * 100).round(1)}).sort_values(by="Belang", ascending=False)
                    fig_imp = px.bar(imp_df, x="Belang", y="Feature", orientation='h')
                    fig_imp.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_imp, use_container_width=True)

                with c_ml2:
                    st.subheader("🔍 Cross-Validated Confusion Matrix")
                    pred = cross_val_predict(rf, X, y, cv=min(5, len(m_df)//2))
                    labels = sorted(y.unique())
                    cm = confusion_matrix(y, pred, labels=labels)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', x=[f"Pred: {l}" for l in labels], y=[f"True: {l}" for l in labels])
                    fig_cm.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_cm, use_container_width=True)

            except Exception as e:
                st.error(f"Matrix simulatie faalde: {e}")
        else:
            st.info("Voeg minimaal 2 labels toe (met elk 5+ samples) om Importance & Confusion Matrix te genereren.")

        # --- EXPORT PIPELINE ---
        st.markdown("### 🚀 Dataset Export")
        c_ex1, c_ex2 = st.columns([1, 3])
        c_ex1.download_button("⚡ Download CSV (Edge Impulse)", data=m_df.to_csv(index=False), file_name="edge_impulse.csv", mime="text/csv", use_container_width=True)
        if c_ex1.button("🗑️ Clear Pipeline", use_container_width=True):
            st.session_state.master_training_dataset = pd.DataFrame()
            st.rerun()
        c_ex2.dataframe(m_df, use_container_width=True, height=150)
