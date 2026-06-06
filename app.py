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
st.set_page_config(page_title="TinyML Data Accelerator V3.5", layout="wide", initial_sidebar_state="expanded")
st.title("⚡ TinyML Data Accelerator (OMEGA-X V3.5)")
st.caption("Synthetic Data Generation • Dataset Augmentation • Feature Extraction • Edge AI Development")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def calculate_fft(signal_data, sample_rate):
    window = np.hanning(len(signal_data))
    fft_values = np.abs(np.fft.rfft(signal_data * window))
    fft_freqs = np.fft.rfftfreq(len(signal_data), 1 / sample_rate)
    return fft_freqs, fft_values

def calculate_spectrogram(signal_data, sample_rate):
    f_spec, t_spec, Sxx = signal.spectrogram(signal_data, sample_rate, nperseg=256, noverlap=128)
    return f_spec, t_spec, Sxx

# ============================================================
# SYNTHETIC VIBRATION ENGINE V3
# ============================================================
@st.cache_data
def generate_vibration_data(condition, severity, rpm, duration=2.0, apply_randomness=False):
    sample_rate = 4000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    actual_rpm = rpm * np.random.normal(1.0, 0.02) if apply_randomness else rpm
    actual_severity = severity * np.random.normal(1.0, 0.10) if apply_randomness else severity
    actual_severity = np.clip(actual_severity, 0.01, 1.0)
    
    f_1x = actual_rpm / 60.0
    phase_1x = 2 * np.pi * f_1x * t
    vibration = 0.2 * np.sin(phase_1x)
    mains_noise = 0.05 * np.sin(2 * np.pi * 50 * t)
    
    noise_scale = np.random.uniform(0.85, 1.15) if apply_randomness else 1.0
    base_noise = np.random.normal(0, 0.1 * noise_scale, len(t))
    vibration += mains_noise + (base_noise * (1 + actual_severity))
    
    bpfo_frequency = f_1x * 3.58
    resonance_frequency = np.random.normal(1200.0, 15.0) if apply_randomness else 1200.0

    if condition == "Unbalance":
        vibration += 1.5 * actual_severity * np.sin(phase_1x)
    elif condition == "Mechanical Looseness":
        for harmonic in range(1, 6):
            vibration += (1.2 / harmonic) * actual_severity * np.sin(harmonic * phase_1x)
        vibration += base_noise * actual_severity * 2.0
    elif condition == "BPFO (Outer Race)":
        phase_fault = 2 * np.pi * bpfo_frequency * t
        impact_variation = 0.8 + 0.4 * np.random.rand(len(t))
        impact_envelope = np.maximum(0, np.cos(phase_fault)) ** 20 * impact_variation
        load_zone_modulation = 1 + 0.6 * np.cos(phase_1x)
        defect_signal = np.sin(2 * np.pi * resonance_frequency * t) * impact_envelope * load_zone_modulation
        vibration += 2.0 * actual_severity * defect_signal

    fft_freqs, fft_values = calculate_fft(vibration, sample_rate)
    return t, vibration, fft_freqs, fft_values, bpfo_frequency, resonance_frequency, actual_rpm, actual_severity, sample_rate

# ============================================================
# SESSION STATE & SIDEBAR
# ============================================================
if "rpm" not in st.session_state: st.session_state.rpm = 1500
if "severity" not in st.session_state: st.session_state.severity = 80
if "master_training_dataset" not in st.session_state: st.session_state.master_training_dataset = pd.DataFrame()

st.sidebar.header("⚙️ Generator Controls")
rpm = st.sidebar.slider("RPM", min_value=600, max_value=3000, value=st.session_state.rpm, step=10)
severity_percent = st.sidebar.slider("Severity (%)", min_value=0, max_value=100, value=st.session_state.severity, step=5)
severity = severity_percent / 100.0
condition = st.sidebar.selectbox("Condition", ["Healthy", "Unbalance", "Mechanical Looseness", "BPFO (Outer Race)"])
show_healthy_overlay = st.sidebar.checkbox("Show Healthy Reference", value=True)
apply_randomness = st.sidebar.checkbox("Apply Dataset Randomization", value=True)

st.session_state.rpm = rpm
st.session_state.severity = severity_percent

# ============================================================
# GENERATE ACTIVE DATASET
# ============================================================
t, vibration, fft_freqs, fft_values, bpfo_frequency, resonance_frequency, actual_rpm, actual_severity, sample_rate = generate_vibration_data(condition, severity, rpm, apply_randomness=apply_randomness)
t_ref, vibration_ref, fft_freqs_ref, fft_values_ref, _, _, _, _, _ = generate_vibration_data("Healthy", 0.0, rpm, apply_randomness=False)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 1. Synthetic Generator", "📦 2. Batch Generator", "🔍 3. Upload Analyzer", "🧪 4. Data Multiplier", "🤖 5. TinyML Trainer"])

# ============================================================
# TAB 1: SYNTHETIC GENERATOR
# ============================================================
with tab1:
    st.header("Synthetic Condition Simulator")
    col_signal, col_fft = st.columns(2)

    with col_signal:
        st.subheader("Time Signal")
        fig_signal = go.Figure()
        if show_healthy_overlay:
            fig_signal.add_trace(go.Scatter(x=t[:800], y=vibration_ref[:800], mode="lines", name="Healthy Reference", line=dict(color='#2ca02c', dash='dot')))
        fig_signal.add_trace(go.Scatter(x=t[:800], y=vibration[:800], mode="lines", name=condition, line=dict(color='#d62728')))
        fig_signal.update_layout(height=350, xaxis_title="Time (s)", yaxis_title="Amplitude")
        st.plotly_chart(fig_signal, use_container_width=True)

    with col_fft:
        st.subheader("FFT Spectrum")
        fig_fft = go.Figure()
        if show_healthy_overlay:
            fig_fft.add_trace(go.Scatter(x=fft_freqs_ref, y=fft_values_ref, mode="lines", name="Healthy Reference", line=dict(color='#2ca02c', dash='dot')))
        fig_fft.add_trace(go.Scatter(x=fft_freqs, y=fft_values, mode="lines", name=condition, line=dict(color='#d62728')))
        fig_fft.update_layout(height=350, xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
        st.plotly_chart(fig_fft, use_container_width=True)

    st.subheader("Spectrogram (Time / Frequency)")
    f_spec, t_spec, Sxx = calculate_spectrogram(vibration, sample_rate)
    max_freq_idx = np.where(f_spec <= 1500)[0][-1] if len(f_spec) > 0 else len(f_spec)
    fig_spec = go.Figure(data=go.Heatmap(z=10 * np.log10(Sxx[:max_freq_idx, :] + 1e-12), x=t_spec, y=f_spec[:max_freq_idx], colorscale="Viridis"))
    fig_spec.update_layout(height=350, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
    st.plotly_chart(fig_spec, use_container_width=True)

    st.markdown("---")
    st.subheader("Physics Dashboard")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("RPM", f"{actual_rpm:.0f}")
    col_b.metric("1× RPM", f"{actual_rpm/60:.2f} Hz")
    col_c.metric("BPFO", f"{bpfo_frequency:.2f} Hz")
    col_d.metric("Resonance", f"{resonance_frequency:.1f} Hz")

# ============================================================
# TAB 2: BATCH GENERATOR
# ============================================================
with tab2:
    st.header("📦 Batch Dataset Generator")
    profile = st.selectbox("Dataset Profile", ["Quick Test (20/cond)", "Research (100/cond)", "Production (500/cond)"])
    files_per_condition = int(profile.split('(')[1].split('/')[0])
    total_files = files_per_condition * 4

    st.info(f"Configuration: {total_files} total files across 4 conditions. Randomization: {'ON' if apply_randomness else 'OFF'}.")

    if st.button("🚀 Generate Balanced Dataset", type="primary"):
        my_bar = st.progress(0, text="Generating physics-driven datasets...")
        zip_buffer = io.BytesIO()
        metadata_list = []
        conditions = ["Healthy", "Unbalance", "Mechanical Looseness", "BPFO (Outer Race)"]
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, cond in enumerate(conditions):
                huidige_sev = 0.0 if cond == "Healthy" else severity
                folder_naam = cond.replace(' ', '_').replace('(', '').replace(')', '').lower()
                
                for i in range(files_per_condition):
                    t_b, z_b, _, _, bpfo_b, res_b, rpm_b, sev_b, _ = generate_vibration_data(cond, huidige_sev, rpm, apply_randomness=apply_randomness)
                    filename = f"{folder_naam}/{folder_naam}_{i:03d}.csv"
                    df = pd.DataFrame({'time': t_b, 'vibration': z_b})
                    zip_file.writestr(filename, df.to_csv(index=False))
                    metadata_list.append({"file": filename, "condition": cond, "rpm": round(rpm_b, 2), "severity": round(sev_b, 3)})
                    
                my_bar.progress((idx + 1) / 4, text=f"Data generated for: {cond}")
            
            zip_file.writestr("metadata.json", json.dumps(metadata_list, indent=4))
            
        time.sleep(0.5)
        my_bar.empty()
        st.success(f"✅ Production batch successfully generated ({total_files} files)!")
        st.download_button("📦 Download .ZIP Archive", data=zip_buffer.getvalue(), file_name="tinyml_balanced_dataset.zip", mime="application/zip")

# ============================================================
# TAB 3: UPLOAD ANALYZER
# ============================================================
with tab3:
    st.header("🔍 Upload Analyzer")
    uploaded_file = st.file_uploader("Upload CSV file (time, vibration)", type=["csv"], key="analyzer_upload")

    if uploaded_file:
        try:
            df_up = pd.read_csv(uploaded_file)
            if "time" in df_up.columns and "vibration" in df_up.columns:
                time_data, vib_data = df_up["time"].values, df_up["vibration"].values
                sr_est = int(1.0 / np.mean(np.diff(time_data)))
                
                rms_val = np.sqrt(np.mean(vib_data**2))
                kurt_val = pd.Series(vib_data).kurt()
                crest_val = np.max(np.abs(vib_data)) / rms_val if rms_val > 0 else 0
                
                fft_freqs_up, fft_vals_up = calculate_fft(vib_data - np.mean(vib_data), sr_est)
                dom_idx = np.argmax(fft_vals_up[5:]) + 5
                dom_freq = fft_freqs_up[dom_idx]
                
                st.success("File analyzed successfully!")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Dominant Freq", f"{dom_freq:.1f} Hz")
                c2.metric("RMS", f"{rms_val:.3f}")
                c3.metric("Kurtosis", f"{kurt_val:.2f}")
                c4.metric("Crest Factor", f"{crest_val:.2f}")
                
                health_score = max(0, min(100, 100 - (15 if crest_val > 6 else 0) - (20 if kurt_val > 8 else 0)))
                st.metric("Machine Health Score", f"{health_score}%")
                st.progress(health_score / 100)
            else:
                st.error("CSV must contain 'time' and 'vibration' columns.")
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================
# TAB 4: DATA MULTIPLIER (PHYSICS-AWARE CLONE ENGINE)
# ============================================================
with tab4:
    st.header("🧪 Data Multiplier (Physics-Aware Clone Engine)")
    st.write("Upload a small sample. We extract the physical signature (Frequency, Kurtosis, Crest, Harmonics) and generate a robust dataset.")
    
    upload_mult = st.file_uploader("Upload reference CSV", type=['csv'], key="multiplier_upload")
    
    if upload_mult:
        try:
            df_mult = pd.read_csv(upload_mult)
            if df_mult.shape[1] >= 2:
                t_mult = df_mult.iloc[:, 0].values
                z_mult = df_mult.iloc[:, 1].values
                sr_mult = int(1.0 / (t_mult[1] - t_mult[0])) * 1000 if (t_mult[1] - t_mult[0]) < 1 else 4000
                
                rms_mult = np.sqrt(np.mean(z_mult**2))
                kurt_mult = pd.Series(z_mult).kurt()
                crest_mult = np.max(np.abs(z_mult)) / rms_mult if rms_mult > 0 else 0
                
                fft_f, fft_v = calculate_fft(z_mult - np.mean(z_mult), sr_mult)
                dom_idx = np.argmax(fft_v[5:]) + 5
                dom_f = fft_f[dom_idx]
                extracted_rpm = dom_f * 60
                
                # --- NIEUW: Harmonic Ratio Berekening ---
                fundamental_amp = fft_v[dom_idx]
                harmonic_amp_sum = 0
                for h in range(2, 6): # 2x, 3x, 4x, 5x harmonischen
                    h_target = h * dom_f
                    h_idx = np.argmin(np.abs(fft_f - h_target))
                    harmonic_amp_sum += fft_v[h_idx]
                
                harmonic_ratio = harmonic_amp_sum / max(abs(fundamental_amp), 1e-6)
                
                # --- SLIMME AUTODETECTIE CONDITIE ---
                if crest_mult > 4.0 and kurt_mult > 3.0:
                    detected_cond = "BPFO (Outer Race)"
                elif harmonic_ratio > 0.4 or (crest_mult > 2.5 and kurt_mult > 0.5):
                    detected_cond = "Mechanical Looseness"
                elif rms_mult > 0.15:
                    detected_cond = "Unbalance"
                else:
                    detected_cond = "Healthy"
                
                st.success(f"Signature Extracted! Dominant Freq: {dom_f:.1f} Hz | Derived RPM: {extracted_rpm:.0f}")
                st.info(f"🧬 Detected Profile: **{detected_cond}** (Kurt: {kurt_mult:.1f}, Crest: {crest_mult:.1f}, Harmonic Ratio: {harmonic_ratio:.2f})")
                
                clone_count = st.slider("Clones to Generate", 50, 500, 100)
                if st.button(f"Clone & Multiply ({detected_cond})", type="primary"):
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zipf:
                        for i in range(clone_count):
                            t_k, z_k, _, _, _, _, _, _, _ = generate_vibration_data(detected_cond, severity=0.8, rpm=extracted_rpm, apply_randomness=True)
                            df_kloon = pd.DataFrame({'time': t_k, 'vibration': z_k})
                            zipf.writestr(f"synthetic_clone_{detected_cond.replace(' ', '_')}_{i:03d}.csv", df_kloon.to_csv(index=False))
                    st.success("Cloning complete!")
                    st.download_button("📦 Download Synthetic Clones (.ZIP)", data=zip_buf.getvalue(), file_name="synthetic_clones.zip", mime="application/zip")
            else:
                st.error("Need 2 columns (time, vibration).")
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================
# TAB 5: TINYML TRAINER (Master Dataset Builder & Quality)
# ============================================================
with tab5:
    st.header("🤖 TinyML Feature Extractor & Pipeline")
    st.write("Extract features from raw vibration CSVs and compile them into a unified ML training dataset.")

    training_files = st.file_uploader("Upload raw CSV files to process", type=["csv"], accept_multiple_files=True, key="trainer_upload")

    if training_files:
        st.success(f"{len(training_files)} files ready for processing.")
        condition_label = st.selectbox("Assign Label to these files", ["Healthy", "Unbalance", "Mechanical Looseness", "BPFO (Outer Race)"])

        if st.button("Extract Features & Add to Master Dataset", type="primary"):
            training_rows = []
            for file in training_files:
                try:
                    df = pd.read_csv(file)
                    vib_arr = df.iloc[:, 1].astype(float).values
                    rms_val = np.sqrt(np.mean(vib_arr**2))
                    
                    fft_f, fft_v = calculate_fft(vib_arr - np.mean(vib_arr), 4000)
                    dom_idx = np.argmax(fft_v[1:]) + 1
                    dom_f = fft_f[dom_idx]
                    
                    # Harmonic Ratio Extractor
                    fundamental_amp = fft_v[dom_idx]
                    harmonic_amp_sum = 0
                    for h in range(2, 6):
                        h_target = h * dom_f
                        h_idx = np.argmin(np.abs(fft_f - h_target))
                        harmonic_amp_sum += fft_v[h_idx]
                    harmonic_ratio = harmonic_amp_sum / max(abs(fundamental_amp), 1e-6)
                    
                    training_rows.append({
                        "RMS": rms_val,
                        "STD": np.std(vib_arr),
                        "Kurtosis": kurtosis(vib_arr),
                        "CrestFactor": np.max(np.abs(vib_arr)) / max(abs(rms_val), 1e-6),
                        "DominantFrequency": dom_f,
                        "PeakAmplitude": np.max(fft_v),
                        "HarmonicRatio": harmonic_ratio, # NIEUWE FEATURE
                        "Label": condition_label
                    })
                except Exception as e:
                    st.warning(f"Could not process {file.name}")

            if training_rows:
                st.session_state.master_training_dataset = pd.concat([st.session_state.master_training_dataset, pd.DataFrame(training_rows)], ignore_index=True)
                st.success(f"✅ Extracted features from {len(training_rows)} files and added them to the Master Dataset!")

    st.divider()
    
    master_df = st.session_state.master_training_dataset
    if len(master_df) > 0:
        
        st.subheader("📊 Dataset Statistics & Quality")
        label_counts = master_df["Label"].value_counts()
        
        # QUALITY ALGORITHM BASE
        quality_score = 100
        num_classes = len(label_counts)
        total_samples = len(master_df)
        
        # Class Penalty
        if num_classes < 2:
            quality_score -= 50
            msg_classes = "🔴 Minimaal 2 klassen nodig voor AI training."
        elif num_classes < 4:
            quality_score -= 10
            msg_classes = "🟡 Overweeg alle defect-statussen toe te voegen."
        else:
            msg_classes = "🟢 Uitstekende variatie in klassen."
            
        # Balance Penalty
        if num_classes > 1:
            balance_ratio = label_counts.min() / label_counts.max()
            if balance_ratio < 0.2:
                quality_score -= 30
                msg_balance = "🔴 Dataset is zwaar uit balans (Imbalanced)."
            elif balance_ratio < 0.6:
                quality_score -= 15
                msg_balance = "🟡 Dataset is enigszins uit balans."
            else:
                msg_balance = "🟢 Dataset is perfect in balans."
        else:
            msg_balance = "🔴 Balans NVT (slechts 1 klasse)."
            
        # Size Penalty
        if total_samples < 50:
            quality_score -= 30
            msg_size = "🔴 Te weinig samples voor robuuste training."
        elif total_samples < 200:
            quality_score -= 10
            msg_size = "🟡 Sample grootte is acceptabel, groter is beter."
        else:
            msg_size = "🟢 Goede totale sample grootte."
            
        # --- FEATURE HEALTH (ROBUUSTE VARIANCE CHECK) ---
        feature_cols = ["RMS", "STD", "Kurtosis", "CrestFactor", "DominantFrequency", "PeakAmplitude", "HarmonicRatio"]
        health_status_list = []
        variance_penalty = 0
        
        for col in feature_cols:
            if col in master_df.columns:
                mean_val = master_df[col].mean()
                std_val = master_df[col].std()
                
                # Robuuste Coëfficiënt van Variatie
                cv = std_val / max(abs(mean_val), 1e-6)
                
                if pd.isna(std_val) or std_val == 0:
                    status = "🔴 CRITICAL (Zero Variance)"
                    variance_penalty += 10
                elif cv < 0.05:
                    status = "🟡 LOW"
                    variance_penalty += 3
                else:
                    status = "🟢 GOOD"
                    
                health_status_list.append({"Feature": col, "Variance Status": status, "CV Ratio": f"{cv:.3f}"})
            
        quality_score -= variance_penalty
        quality_score = max(0, min(100, int(quality_score)))

        col_q1, col_q2 = st.columns([1, 2])
        with col_q1:
            st.metric("Dataset Quality Score", f"{quality_score}%")
            if quality_score >= 85: st.success("🟢 Ready for Edge Impulse!")
            elif quality_score >= 50: st.warning("🟡 Bruikbaar, maar let op de waarschuwingen.")
            else: st.error("🔴 Niet klaar voor training. Pas dataset aan.")
        with col_q2:
            st.markdown(f"**Dataset Structuur Analyse:**\n- {msg_classes}\n- {msg_balance}\n- {msg_size}")

        st.bar_chart(label_counts)

        # FEATURE HEALTH TABEL
        st.markdown("#### 🧬 Feature Health Analysis (Variance Check)")
        st.write("Modellen hebben variantie in de data nodig om te generaliseren. Geen variantie = Data Leakage / Memorization.")
        st.dataframe(pd.DataFrame(health_status_list), use_container_width=True)

        # DATASET TABEL
        st.subheader("📋 Master Training Dataset")
        st.dataframe(master_df, use_container_width=True)
        
        # EXPORT PIPELINE
        st.markdown("### 🚀 Export Pipeline")
        col_dl1, col_dl2, col_clr = st.columns(3)
        
        metadata_dict = {
            "total_samples": total_samples,
            "labels": label_counts.to_dict(),
            "quality_score": quality_score,
            "feature_health_penalties": variance_penalty,
            "generated_by": "TinyML Data Accelerator (OMEGA-X V3.5)"
        }
        json_meta_str = json.dumps(metadata_dict, indent=4)
        csv_master_str = master_df.to_csv(index=False)
        
        zip_buf_master = io.BytesIO()
        with zipfile.ZipFile(zip_buf_master, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("master_tinyml_dataset.csv", csv_master_str)
            zipf.writestr("dataset_metadata.json", json_meta_str)
        
        with col_dl1:
            st.download_button(
                "📦 Download Production Dataset (.ZIP)", 
                data=zip_buf_master.getvalue(), 
                file_name="omega_x_master_dataset_v3_5.zip", 
                mime="application/zip", 
                use_container_width=True
            )
        with col_dl2:
            st.download_button(
                "⚡ Export Edge Impulse Dataset (.CSV)", 
                data=csv_master_str, 
                file_name="edge_impulse_dataset.csv", 
                mime="text/csv", 
                use_container_width=True
            )
        with col_clr:
            if st.button("🗑️ Clear Master Dataset", use_container_width=True):
                st.session_state.master_training_dataset = pd.DataFrame()
                st.rerun()
    else:
        st.info("No samples collected yet. Upload raw files and process them to build your dataset.")
