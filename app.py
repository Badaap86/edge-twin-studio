import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
import json
import zipfile

# --- CONFIGURATIE & COMMERCIËLE POSITIONERING ---
st.set_page_config(page_title="TinyML Data Accelerator", layout="wide")
st.title("⚡ Edge AI Acceleration Tool: Kinematic Vibration Simulator")
st.subheader("Verkort je Edge AI proof-of-concept van weken naar minuten.")
st.markdown("Genereer fysisch verklaarbare trainingsdata en omzeil het *Cold Start*-probleem voordat je echte hardware-defecten hebt gemeten.")

@st.cache_data
def generate_vibration_data(condition, severity, rpm, duration=2.0, apply_randomness=False):
    sample_rate = 4000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    actueel_rpm = rpm * np.random.normal(1.0, 0.02) if apply_randomness else rpm
    actuele_severity = severity * np.random.normal(1.0, 0.05) if apply_randomness else severity
    actuele_severity = max(0.01, min(1.0, actuele_severity))
    
    f_1x = actueel_rpm / 60.0
    fase_1x = 2 * np.pi * f_1x * t
    
    z_totaal = 0.2 * np.sin(fase_1x)
    netruis = 0.05 * np.sin(2 * np.pi * 50 * t)
    
    noise_mult = np.random.uniform(0.8, 1.2) if apply_randomness else 1.0
    base_noise = np.random.normal(0, 0.1 * noise_mult, len(t))
    z_totaal += netruis + (base_noise * (1 + actuele_severity))
    
    calc_bpfo = f_1x * 3.58
    calc_res = np.random.normal(1200.0, 15.0) if apply_randomness else 1200.0
    
    if condition == 'Unbalance':
        z_totaal += (1.5 * actuele_severity) * np.sin(fase_1x)
    elif condition == 'Mechanical Looseness':
        for i in range(1, 6):
            z_totaal += (1.2 / i) * actuele_severity * np.sin(i * fase_1x)
        z_totaal += base_noise * (actuele_severity * 2.0)
    elif condition == 'BPFO (Outer Race)':
        fase_fout = 2 * np.pi * calc_bpfo * t
        impact_variatie = 0.8 + 0.4 * np.random.rand(len(t))
        impact_envelop = np.maximum(0, np.cos(fase_fout))**20 * impact_variatie
        load_zone_modulatie = 1 + 0.6 * np.cos(fase_1x)
        defect_signaal = np.sin(2 * np.pi * calc_res * t) * impact_envelop * load_zone_modulatie
        z_totaal += (2.0 * actuele_severity) * defect_signaal

    window = np.hanning(len(t))
    fft_waarden = np.abs(np.fft.rfft(z_totaal * window))
    fft_frequenties = np.fft.rfftfreq(len(t), 1/sample_rate)
    
    return t, z_totaal, fft_frequenties, fft_waarden, calc_bpfo, calc_res, actueel_rpm, actuele_severity

# --- INTERFACE INDELING ---
st.sidebar.header("1. Visualiseer & Valideer")

# Twee-weg koppeling (Two-way bind) voor RPM en Severity
if "rpm" not in st.session_state: st.session_state["rpm"] = 1500
if "sev" not in st.session_state: st.session_state["sev"] = 80

def sync_rpm_slider(): st.session_state["rpm"] = st.session_state["rpm_slider_widget"]
def sync_rpm_num(): st.session_state["rpm"] = st.session_state["rpm_num_widget"]
def sync_sev_slider(): st.session_state["sev"] = st.session_state["sev_slider_widget"]
def sync_sev_num(): st.session_state["sev"] = st.session_state["sev_num_widget"]

st.session_state["rpm_slider_widget"] = st.session_state["rpm"]
st.session_state["rpm_num_widget"] = st.session_state["rpm"]
st.session_state["sev_slider_widget"] = st.session_state["sev"]
st.session_state["sev_num_widget"] = st.session_state["sev"]

st.sidebar.markdown("**Basis RPM**")
st.sidebar.number_input("Typ RPM", 600, 3000, key="rpm_num_widget", on_change=sync_rpm_num, label_visibility="collapsed")
st.sidebar.slider("Schuif RPM", 600, 3000, step=10, key="rpm_slider_widget", on_change=sync_rpm_slider, label_visibility="collapsed")
rpm = st.session_state["rpm"]

st.sidebar.markdown("**Severity (%)**")
st.sidebar.number_input("Typ Severity", 0, 100, key="sev_num_widget", on_change=sync_sev_num, label_visibility="collapsed")
st.sidebar.slider("Schuif Severity", 0, 100, step=5, key="sev_slider_widget", on_change=sync_sev_slider, label_visibility="collapsed")
severity_pct = st.session_state["sev"]
severity = severity_pct / 100.0

st.sidebar.markdown("---")
condition = st.sidebar.selectbox("Defect Type", ["Healthy", "Unbalance", "Mechanical Looseness", "BPFO (Outer Race)"])

t, z_data, freqs, fft_z, bpfo_hz, res_hz, _, _ = generate_vibration_data(condition, severity, rpm)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Live Tijdsignaal")
    fig_time = go.Figure(go.Scatter(x=t[:800], y=z_data[:800], mode='lines', line=dict(color='#1f77b4')))
    fig_time.update_layout(xaxis_title="Tijd (s)", yaxis_title="g", height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_time, use_container_width=True)

with col2:
    st.subheader("Live FFT Spectrum")
    fig_fft = go.Figure(go.Scatter(x=freqs, y=fft_z, mode='lines', line=dict(color='#d62728')))
    if condition == 'BPFO (Outer Race)': fig_fft.update_layout(xaxis_range=[1000, 1400])
    else: fig_fft.update_layout(xaxis_range=[0, min(rpm/60 * 10, 500)])
    fig_fft.update_layout(xaxis_title="Hz", yaxis_title="Amplitude", height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_fft, use_container_width=True)

# --- METRICS & BATCH BILDER ---
st.markdown("---")
st.header("2. Genereer Trainingsdata (Batch Export)")
st.write("Genereer een gebalanceerde dataset met realistische variaties (±2% RPM jitter, variabele ruisvloer) om overfitting in je Edge AI model te voorkomen.")

met_col1, met_col2, met_col3, met_col4 = st.columns(4)
met_col1.metric("1× RPM (Draaggolf)", f"{rpm/60:.2f} Hz")
met_col2.metric("Berekende BPFO", f"{bpfo_hz:.2f} Hz")
met_col3.metric("Structurele Resonantie", f"{res_hz:.1f} Hz")
met_col4.metric("Workflow Status", "Edge Impulse Ready")

batch_size = st.slider("Aantal CSV bestanden per conditie", min_value=10, max_value=250, value=50, step=10)

if st.button(f"Genereer {batch_size} datasets voor '{condition}'"):
    with st.spinner("Genereren van fysica-gedreven datasets..."):
        zip_buffer = io.BytesIO()
        metadata_list = []
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for i in range(batch_size):
                t_b, z_b, _, _, bpfo_b, res_b, rpm_b, sev_b = generate_vibration_data(condition, severity, rpm, apply_randomness=True)
                filename = f"{condition.replace(' ', '_').lower()}_{i:03d}.csv"
                df = pd.DataFrame({'timestamp_ms': t_b * 1000, 'accZ': z_b})
                zip_file.writestr(filename, df.to_csv(index=False))
                
                metadata_list.append({
                    "file": filename, "condition": condition, "base_rpm": rpm, "actual_rpm": round(rpm_b, 2),
                    "target_severity": severity, "actual_severity": round(sev_b, 3), "calculated_bpfo_hz": round(bpfo_b, 2), "resonance_hz": round(res_b, 2)
                })
            zip_file.writestr("metadata.json", json.dumps(metadata_list, indent=4))
        
        st.success("Batch succesvol gegenereerd!")
        st.download_button(label="📦 Download .ZIP Archief (CSV + JSON)", data=zip_buffer.getvalue(), file_name=f"dataset_{condition.replace(' ', '_').lower()}_batch.zip", mime="application/zip")
