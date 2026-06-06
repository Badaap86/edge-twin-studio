import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.signal as signal
import io
import json
import zipfile
import time
from scipy.stats import kurtosis

# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(page_title="OMEGA-X Edge Data Platform", layout="wide", initial_sidebar_state="expanded")
st.title("⚡ OMEGA-X Edge Data Platform (V4.1)")
st.caption("Multi-Modal Synthetic Generation • Anomaly Detection • Feature Extraction")

# ============================================================
# CORE DSP (DIGITAL SIGNAL PROCESSING) FUNCTIONS
# ============================================================
def calculate_fft(signal_data, sample_rate):
    window = np.hanning(len(signal_data))
    fft_values = np.abs(np.fft.rfft(signal_data * window))
    fft_freqs = np.fft.rfftfreq(len(signal_data), 1 / sample_rate)
    return fft_freqs, fft_values

def calculate_spectrogram(signal_data, sample_rate):
    f_spec, t_spec, Sxx = signal.spectrogram(signal_data, sample_rate, nperseg=256, noverlap=128)
    return f_spec, t_spec, Sxx

# --- AUDIO AI FEATURE EXTRACTORS ---
def get_zero_crossing_rate(sig):
    return ((sig[:-1] * sig[1:]) < 0).sum() / len(sig)

def get_spectral_centroid(fft_f, fft_v):
    if np.sum(fft_v) == 0: return 0
    return np.sum(fft_f * fft_v) / np.sum(fft_v)

def get_spectral_rolloff(fft_f, fft_v, percentile=0.85):
    cumulative_energy = np.cumsum(fft_v)
    total_energy = cumulative_energy[-1]
    if total_energy == 0: return 0
    rolloff_idx = np.where(cumulative_energy >= percentile * total_energy)[0][0]
    return fft_f[rolloff_idx]

# ============================================================
# SENSOR 1: VIBRATION (KINEMATIC PHYSICS ENGINE)
# ============================================================
@st.cache_data
def generate_vibration_data(condition, severity, rpm, duration=2.0, apply_randomness=False):
    sample_rate = 4000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    act_rpm = rpm * np.random.normal(1.0, 0.02) if apply_randomness else rpm
    act_sev = np.clip(severity * np.random.normal(1.0, 0.10) if apply_randomness else severity, 0.01, 1.0)
    
    f_1x = act_rpm / 60.0
    phase_1x = 2 * np.pi * f_1x * t
    vibration = 0.2 * np.sin(phase_1x)
    
    noise_scale = np.random.uniform(0.85, 1.15) if apply_randomness else 1.0
    vibration += 0.05 * np.sin(2 * np.pi * 50 * t) + (np.random.normal(0, 0.1 * noise_scale, len(t)) * (1 + act_sev))
    
    bpfo = f_1x * 3.58
    res_freq = np.random.normal(1200.0, 15.0) if apply_randomness else 1200.0

    if condition == "Unbalance":
        vibration += 1.5 * act_sev * np.sin(phase_1x)
    elif condition == "Mechanical Looseness":
        for h in range(1, 6): vibration += (1.2 / h) * act_sev * np.sin(h * phase_1x)
        vibration += np.random.normal(0, 0.1, len(t)) * act_sev * 2.0
    elif condition == "BPFO (Outer Race)":
        impacts = np.maximum(0, np.cos(2 * np.pi * bpfo * t)) ** 20 * (0.8 + 0.4 * np.random.rand(len(t)))
        vibration += 2.0 * act_sev * np.sin(2 * np.pi * res_freq * t) * impacts * (1 + 0.6 * np.cos(phase_1x))
    elif condition == "Unknown Anomaly (Random)":
        # Simuleert een willekeurige, onbekende verstoring voor Anomaly Detection
        vibration += act_sev * np.random.normal(0, 2.0, len(t)) * np.sin(2 * np.pi * np.random.uniform(10, 500) * t)

    f, v = calculate_fft(vibration, sample_rate)
    return {"t": t, "sig": vibration, "fft_f": f, "fft_v": v, "rpm": act_rpm, "sev": act_sev, "bpfo": bpfo, "res": res_freq, "sr": sample_rate}

# ============================================================
# SENSOR 2: ACOUSTIC (AUDIO PHYSICS ENGINE)
# ============================================================
@st.cache_data
def generate_acoustic_data(condition, severity, duration=2.0, apply_randomness=False):
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    act_sev = np.clip(severity * np.random.normal(1.0, 0.10) if apply_randomness else severity, 0.01, 1.0)
    
    # Base layer: Ambient forest/wind noise
    audio = np.random.normal(0, 0.05, len(t))
    
    if condition == "Chainsaw":
        f0 = 80 + (np.random.normal(0, 5) if apply_randomness else 0)
        saw = signal.sawtooth(2 * np.pi * f0 * t)
        mod = 1 + 0.5 * np.sin(2 * np.pi * 3 * t)
        grit = np.random.normal(0, 0.2, len(t))
        audio += act_sev * (saw * mod + grit)
    elif condition == "Gunshot":
        impact_t = 0.5 + (np.random.uniform(-0.2, 0.2) if apply_randomness else 0)
        idx = int(impact_t * sample_rate)
        if idx < len(t):
            decay = np.exp(-15 * (t[idx:] - impact_t))
            burst = np.random.normal(0, 1, len(decay)) * decay
            audio[idx:] += act_sev * 4.0 * burst
    elif condition == "Engine Drone":
        f_drone = 50 + (np.random.normal(0, 2) if apply_randomness else 0)
        audio += act_sev * 0.8 * np.sin(2 * np.pi * f_drone * t)
        audio += act_sev * 0.4 * np.sin(2 * np.pi * (f_drone * 1.5) * t)
    elif condition == "Unknown Anomaly (Random)":
        # Simuleert ongeclassificeerde herrie (krakende takken, machines, ruis)
        sweep = signal.chirp(t, f0=100, f1=2000, t1=duration, method='logarithmic')
        broadband = np.random.uniform(-1, 1, len(t))
        envelope = np.abs(np.sin(2 * np.pi * np.random.uniform(0.5, 5) * t))
        audio += act_sev * 2.0 * (sweep * 0.5 + broadband * 0.5) * envelope

    audio = np.clip(audio, -1.0, 1.0)
    f, v = calculate_fft(audio, sample_rate)
    return {"t": t, "sig": audio, "fft_f": f, "fft_v": v, "sev": act_sev, "sr": sample_rate}

# ============================================================
# GLOBAL UI & MODALITY STATE
# ============================================================
st.sidebar.header("📡 1. Sensor Modality")
modality = st.sidebar.radio("Selecteer het type ML Sensor", ["Vibration (Accelerometer)", "Acoustic (Microphone)"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 2. Generator Controls")

if modality == "Vibration (Accelerometer)":
    if "rpm" not in st.session_state: st.session_state.rpm = 1500
    rpm = st.sidebar.slider("Basis RPM", 600, 3000, value=st.session_state.rpm, step=10)
    st.session_state.rpm = rpm
    cond_options = ["Baseline (Normal)", "Unbalance", "Mechanical Looseness", "BPFO (Outer Race)", "Unknown Anomaly (Random)"]
else:
    cond_options = ["Baseline (Ambient/Normal)", "Chainsaw", "Gunshot", "Engine Drone", "Unknown Anomaly (Random)"]

if "severity" not in st.session_state: st.session_state.severity = 80
severity_percent = st.sidebar.slider("Signal Severity (%)", 0, 100, value=st.session_state.severity, step=5)
severity = severity_percent / 100.0
st.session_state.severity = severity_percent

condition = st.sidebar.selectbox("Gesimuleerde Conditie", cond_options)
apply_randomness = st.sidebar.checkbox("Actieve Dataset Jitter (Realism)", value=True)

if "master_training_dataset" not in st.session_state: st.session_state.master_training_dataset = pd.DataFrame()
if "current_modality" not in st.session_state: st.session_state.current_modality = modality

if st.session_state.current_modality != modality:
    st.session_state.master_training_dataset = pd.DataFrame()
    st.session_state.current_modality = modality

# ============================================================
# ACTIVE DATASET GENERATION
# ============================================================
if modality == "Vibration (Accelerometer)":
    data_live = generate_vibration_data(condition, severity, rpm, apply_randomness=apply_randomness)
    data_ref = generate_vibration_data("Baseline (Normal)", 0.0, rpm, apply_randomness=False)
else:
    data_live = generate_acoustic_data(condition, severity, apply_randomness=apply_randomness)
    data_ref = generate_acoustic_data("Baseline (Ambient/Normal)", 0.0, apply_randomness=False)

# ============================================================
# TAB CONFIGURATION
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 1. Synthetic Generator", "📦 2. Batch Generator", "🔍 3. Upload Analyzer", "🧪 4. Data Multiplier", "🤖 5. ML Pipeline"])

# ============================================================
# TAB 1: SYNTHETIC GENERATOR
# ============================================================
with tab1:
    st.header(f"Synthetic {modality.split(' ')[0]} Simulator")
    c_sig, c_fft = st.columns(2)

    with c_sig:
        st.subheader("Time Domain Signal")
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(x=data_ref["t"][:1000], y=data_ref["sig"][:1000], mode="lines", name="Baseline (Normal)", line=dict(color='#2ca02c', dash='dot')))
        fig_sig.add_trace(go.Scatter(x=data_live["t"][:1000], y=data_live["sig"][:1000], mode="lines", name=condition, line=dict(color='#d62728')))
        fig_sig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_sig, use_container_width=True)

    with c_fft:
        st.subheader("Frequency Domain (FFT)")
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(x=data_ref["fft_f"], y=data_ref["fft_v"], mode="lines", name="Baseline (Normal)", line=dict(color='#2ca02c', dash='dot')))
        fig_fft.add_trace(go.Scatter(x=data_live["fft_f"], y=data_live["fft_v"], mode="lines", name=condition, line=dict(color='#d62728')))
        x_max = 1500 if modality == "Vibration (Accelerometer)" else 4000
        fig_fft.update_layout(height=300, xaxis_range=[0, x_max], margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_fft, use_container_width=True)

    st.subheader("Spectrogram (Time/Frequency Evolution)")
    f_spec, t_spec, Sxx = calculate_spectrogram(data_live["sig"], data_live["sr"])
    max_f = np.where(f_spec <= x_max)[0][-1] if len(f_spec) > 0 else len(f_spec)
    fig_spec = go.Figure(data=go.Heatmap(z=10 * np.log10(Sxx[:max_f, :] + 1e-10), x=t_spec, y=f_spec[:max_f], colorscale="Viridis"))
    fig_spec.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_spec, use_container_width=True)

# ============================================================
# TAB 2: BATCH GENERATOR
# ============================================================
with tab2:
    st.header(f"📦 {modality.split(' ')[0]} Batch Generator")
    profile = st.selectbox("Dataset Size Profile", ["Quick Test (20/cond)", "Research (100/cond)", "Production (500/cond)"])
    batch_size = int(profile.split('(')[1].split('/')[0])
    
    if st.button(f"🚀 Generate Dataset ({batch_size * len(cond_options)} files)", type="primary"):
        my_bar = st.progress(0, text="Generating datasets...")
        zip_buf = io.BytesIO()
        metadata = []
        
        with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zipf:
            for idx, cond in enumerate(cond_options):
                sev_base = 0.0 if "Baseline" in cond else severity
                folder = cond.replace(' ', '_').replace('(', '').replace(')', '').lower()
                
                for i in range(batch_size):
                    if modality == "Vibration (Accelerometer)":
                        d = generate_vibration_data(cond, sev_base, rpm, apply_randomness=apply_randomness)
                        metadata.append({"file": f"{folder}_{i:03d}.csv", "label": cond, "rpm": round(d["rpm"],2)})
                    else:
                        d = generate_acoustic_data(cond, sev_base, apply_randomness=apply_randomness)
                        metadata.append({"file": f"{folder}_{i:03d}.csv", "label": cond})
                        
                    zipf.writestr(f"{folder}/{folder}_{i:03d}.csv", pd.DataFrame({'time': d["t"], 'value': d["sig"]}).to_csv(index=False))
                my_bar.progress((idx + 1) / len(cond_options))
                
            zipf.writestr("metadata.json", json.dumps(metadata, indent=4))
        
        my_bar.empty()
        st.success("✅ Batch generation complete!")
        st.download_button("📦 Download Dataset (.ZIP)", data=zip_buf.getvalue(), file_name=f"omega_x_{modality[:3].lower()}_batch.zip", mime="application/zip")

# ============================================================
# TAB 3 & 4: ANALYZER & CLONER
# ============================================================
with tab4:
    st.header("🧪 Physics-Aware Clone Engine")
    st.write(f"Upload a small `{modality.split(' ')[0]}` sample to extract its physical signature and generate clones.")
    
    up_clone = st.file_uploader("Upload reference CSV", type=['csv'], key="clone_up")
    if up_clone:
        try:
            df_c = pd.read_csv(up_clone)
            if df_c.shape[1] >= 2:
                sig_c = df_c.iloc[:, 1].values
                sr_c = 16000 if modality == "Acoustic (Microphone)" else 4000
                f_c, v_c = calculate_fft(sig_c - np.mean(sig_c), sr_c)
                
                if modality == "Vibration (Accelerometer)":
                    crest = np.max(np.abs(sig_c)) / max(np.sqrt(np.mean(sig_c**2)), 1e-6)
                    kurt = pd.Series(sig_c).kurt()
                    if crest > 4.0 and kurt > 3.0: det_cond = "BPFO (Outer Race)"
                    elif crest > 2.5: det_cond = "Mechanical Looseness"
                    elif np.sqrt(np.mean(sig_c**2)) > 0.15: det_cond = "Unbalance"
                    else: det_cond = "Baseline (Normal)"
                    st.info(f"🧬 Detected Vibration Profile: **{det_cond}**")
                else:
                    zcr = get_zero_crossing_rate(sig_c)
                    if zcr > 0.2: det_cond = "Chainsaw"
                    elif np.max(np.abs(sig_c)) > 0.8: det_cond = "Gunshot"
                    else: det_cond = "Baseline (Ambient/Normal)"
                    st.info(f"🧬 Detected Acoustic Profile: **{det_cond}** (ZCR: {zcr:.3f})")

                clones = st.slider("Clones to Generate", 50, 500, 100)
                if st.button("Clone & Multiply", type="primary"):
                    st.success("Logica werkt! Implementeer ZIP export hier zoals in Tab 2.")
        except Exception as e: st.error(f"Error: {e}")

# ============================================================
# TAB 5: MULTI-MODAL ML PIPELINE (CLASSIFICATION & ANOMALY)
# ============================================================
with tab5:
    st.header("🤖 Multi-Modal Feature Pipeline")
    st.write("Verzamel data voor Supervised Classification (Specifieke defecten) of Unsupervised Anomaly Detection (Alleen normaal vs. onbekend).")

    # --- NIEUW: ML STRATEGY KEUZE ---
    ml_strategy = st.radio("Edge Impulse ML Strategy:", ["Supervised Classification (Multiple Labels)", "Unsupervised Anomaly Detection (Normal vs. Rest)"])

    raw_files = st.file_uploader("Upload Raw CSVs", type=["csv"], accept_multiple_files=True)
    
    if raw_files:
        if ml_strategy == "Unsupervised Anomaly Detection (Normal vs. Rest)":
            st.info("💡 Voor Anomaly Detection train je uitsluitend op de 'Baseline (Normal)'. Gebruik de Anomaly labels alleen als Test Data.")
            lbl_options = ["Baseline (Normal)", "Test Data: Anomaly"]
        else:
            lbl_options = cond_options
            
        lbl = st.selectbox("Assign Label", lbl_options)
        
        if st.button("Extract Features", type="primary"):
            rows = []
            for file in raw_files:
                try:
                    df = pd.read_csv(file)
                    sig = df.iloc[:, 1].astype(float).values
                    sr_feat = 16000 if modality == "Acoustic (Microphone)" else 4000
                    f_f, v_f = calculate_fft(sig - np.mean(sig), sr_feat)
                    
                    rms_val = np.sqrt(np.mean(sig**2))
                    dom_f = f_f[np.argmax(v_f[1:]) + 1]
                    
                    feat_dict = {"Label": lbl}
                    if modality == "Vibration (Accelerometer)":
                        fund_amp = v_f[np.argmax(v_f[1:]) + 1]
                        harm_sum = sum(v_f[np.argmin(np.abs(f_f - (h * dom_f)))] for h in range(2, 6))
                        
                        feat_dict.update({
                            "RMS": rms_val,
                            "Kurtosis": kurtosis(sig),
                            "CrestFactor": np.max(np.abs(sig)) / max(rms_val, 1e-6),
                            "DominantFreq": dom_f,
                            "HarmonicRatio": harm_sum / max(fund_amp, 1e-6)
                        })
                    else: # AUDIO FEATURES
                        feat_dict.update({
                            "RMS": rms_val,
                            "ZeroCrossingRate": get_zero_crossing_rate(sig),
                            "SpectralCentroid": get_spectral_centroid(f_f, v_f),
                            "SpectralRolloff": get_spectral_rolloff(f_f, v_f),
                            "PeakAmplitude": np.max(v_f)
                        })
                    rows.append(feat_dict)
                except Exception: pass
            
            if rows:
                st.session_state.master_training_dataset = pd.concat([st.session_state.master_training_dataset, pd.DataFrame(rows)], ignore_index=True)
                st.success(f"✅ Extracted features from {len(rows)} files!")

    st.divider()
    m_df = st.session_state.master_training_dataset
    if len(m_df) > 0:
        st.subheader(f"📊 Dataset Quality Engine ({modality.split(' ')[0]})")
        
        lbl_cnts = m_df["Label"].value_counts()
        q_score = 100
        variance_penalty = 0
        
        # Balance & Size Check aangescherpt voor Anomaly Detection
        if ml_strategy == "Unsupervised Anomaly Detection (Normal vs. Rest)":
            if "Baseline (Normal)" not in lbl_cnts:
                q_score -= 80
                st.error("🔴 LET OP: Een Anomaly model vereist een grote 'Baseline (Normal)' dataset om op te trainen.")
            elif lbl_cnts.get("Baseline (Normal)", 0) < 100:
                q_score -= 20
        else:
            if len(lbl_cnts) < 2: q_score -= 50
            elif (lbl_cnts.min() / lbl_cnts.max()) < 0.3: q_score -= 30
            
        if len(m_df) < 50: q_score -= 30
        
        # Feature Health Check
        health_list = []
        for col in m_df.columns:
            if col != "Label":
                mean_v, std_v = m_df[col].mean(), m_df[col].std()
                cv = std_v / max(abs(mean_v), 1e-6)
                if pd.isna(std_v) or std_v == 0: 
                    stat, variance_penalty = "🔴 CRITICAL", variance_penalty + 10
                elif cv < 0.05: 
                    stat, variance_penalty = "🟡 LOW", variance_penalty + 3
                else: stat = "🟢 GOOD"
                health_list.append({"Feature": col, "Variance": stat, "CV": f"{cv:.3f}"})
                
        q_score = max(0, min(100, int(q_score - variance_penalty)))
        
        col_m1, col_m2 = st.columns([1, 2])
        col_m1.metric("Enterprise Quality Score", f"{q_score}%")
        col_m2.dataframe(pd.DataFrame(health_list), use_container_width=True, height=150)
        
        st.dataframe(m_df, use_container_width=True)
        
        st.markdown("### 🚀 Export to Edge Impulse")
        c_ex1, c_ex2, c_ex3 = st.columns(3)
        
        json_meta = json.dumps({"sensor": modality, "strategy": ml_strategy, "samples": len(m_df), "quality": q_score}, indent=4)
        csv_str = m_df.to_csv(index=False)
        
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("edge_impulse_dataset.csv", csv_str)
            zf.writestr("metadata.json", json_meta)
            
        c_ex1.download_button("📦 Download Production .ZIP", data=zip_buf.getvalue(), file_name="omega_x_dataset.zip", mime="application/zip", use_container_width=True)
        c_ex2.download_button("⚡ Download Raw .CSV", data=csv_str, file_name="edge_impulse.csv", mime="text/csv", use_container_width=True)
        if c_ex3.button("🗑️ Clear Pipeline", use_container_width=True):
            st.session_state.master_training_dataset = pd.DataFrame()
            st.rerun()
