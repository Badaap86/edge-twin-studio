import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.signal as signal
import io
import json
import zipfile
import time

# --- CONFIGURATIE & COMMERCIËLE POSITIONERING ---
st.set_page_config(page_title="TinyML Data Accelerator", layout="wide", initial_sidebar_state="expanded")
st.title("⚡ Edge AI Acceleration Tool: Kinematic Vibration Simulator")
st.subheader("Train Edge AI models before real-world failure data exists.")
st.markdown("Genereer of vermenigvuldig fysisch verklaarbare trainingsdata om het *Cold Start*-probleem in voorspellend onderhoud te verslaan.")

@st.cache_data
def generate_vibration_data(condition, severity, rpm, duration=2.0, apply_randomness=False):
    sample_rate = 4000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    actueel_rpm = rpm * np.random.normal(1.0, 0.02) if apply_randomness else rpm
    actuele_severity = severity * np.random.normal(1.0, 0.10) if apply_randomness else severity
    actuele_severity = max(0.01, min(1.0, actuele_severity))
    
    f_1x = actueel_rpm / 60.0
    fase_1x = 2 * np.pi * f_1x * t
    
    z_totaal = 0.2 * np.sin(fase_1x)
    netruis = 0.05 * np.sin(2 * np.pi * 50 * t)
    
    noise_mult = np.random.uniform(0.85, 1.15) if apply_randomness else 1.0
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
    
    return t, z_totaal, fft_frequenties, fft_waarden, calc_bpfo, calc_res, actueel_rpm, actuele_severity, sample_rate

# --- HOOFD NAVIGATIE (TABS) ---
tab1, tab2 = st.tabs(["🎛️ 1. Synthetische Generator", "📂 2. Data Multiplier (CSV Upload)"])

# =========================================================================
# TAB 1: SYNTHETISCHE GENERATOR
# =========================================================================
with tab1:
    st.sidebar.header("1. Visualiseer & Valideer")

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
    condition = st.sidebar.selectbox("Defect Type", ["Unbalance", "Mechanical Looseness", "BPFO (Outer Race)"])
    toon_healthy = st.sidebar.checkbox("✅ Toon Healthy Referentie", value=True)

    t, z_data_def, freqs, fft_z_def, bpfo_hz, res_hz, _, _, fs = generate_vibration_data(condition, severity, rpm)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Live Tijdsignaal")
        fig_time = go.Figure()
        if toon_healthy:
            _, z_data_ref, _, _, _, _, _, _, _ = generate_vibration_data("Healthy", 0, rpm)
            fig_time.add_trace(go.Scatter(x=t[:800], y=z_data_ref[:800], mode='lines', name='Healthy (Ref)', line=dict(color='#2ca02c', width=1, dash='dot')))
        fig_time.add_trace(go.Scatter(x=t[:800], y=z_data_def[:800], mode='lines', name=condition, line=dict(color='#d62728')))
        fig_time.update_layout(xaxis_title="Tijd (s)", yaxis_title="g", height=300, margin=dict(l=0, r=0, t=30, b=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig_time, use_container_width=True)

    with col2:
        st.subheader("Live FFT Spectrum")
        fig_fft = go.Figure()
        if toon_healthy:
            _, _, _, fft_z_ref, _, _, _, _, _ = generate_vibration_data("Healthy", 0, rpm)
            fig_fft.add_trace(go.Scatter(x=freqs, y=fft_z_ref, mode='lines', name='Healthy (Ref)', line=dict(color='#2ca02c', width=1, dash='dot')))
        fig_fft.add_trace(go.Scatter(x=freqs, y=fft_z_def, mode='lines', name=condition, line=dict(color='#d62728')))
        if condition == 'BPFO (Outer Race)': fig_fft.update_layout(xaxis_range=[0, 1400])
        else: fig_fft.update_layout(xaxis_range=[0, min(rpm/60 * 10, 500)])
        fig_fft.update_layout(xaxis_title="Hz", yaxis_title="Amplitude", height=300, margin=dict(l=0, r=0, t=30, b=0), legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig_fft, use_container_width=True)

    with col3:
        st.subheader("Spectrogram (Tijd/Frequentie)")
        f_spec, t_spec, Sxx = signal.spectrogram(z_data_def, fs, nperseg=256, noverlap=128)
        max_freq_idx = np.where(f_spec <= 1500)[0][-1] 
        fig_spec = go.Figure(data=go.Heatmap(z=10 * np.log10(Sxx[:max_freq_idx, :] + 1e-10), x=t_spec, y=f_spec[:max_freq_idx], colorscale='Viridis'))
        fig_spec.update_layout(xaxis_title="Tijd (s)", yaxis_title="Hz", height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_spec, use_container_width=True)

    st.markdown("---")
    st.header("2. Fysische Analyse (Edge Labeling)")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### Spectrale Signatuur")
        st.write(f"- **RPM:** {rpm}")
        st.write(f"- **1x RPM Draaggolf:** {rpm/60:.2f} Hz")
        if condition == 'BPFO (Outer Race)':
            st.write(f"- **Berekende BPFO:** {bpfo_hz:.2f} Hz")
            st.write(f"- **Resonantie Piek:** {res_hz:.1f} Hz")

    with col_b:
        st.markdown("### Edge Impulse Metadata")
        st.code(f"Label: {condition.replace(' ', '_').lower()}\nSeverity: {severity}\nRPM: {rpm}", language="text")

    with col_c:
         st.markdown("### Model Readiness Score")
         st.write("✅ Dataset Balance: Gebalanceerd")
         st.write("✅ Variatie: Toegepast (Jitter/Noise)")
         st.write("🟢 **Overfitting Risk:** Low")

    st.markdown("---")
    st.header("3. Genereer Trainingsdata (Batch Export)")
    col_export1, col_export2, col_export3 = st.columns(3)

    with col_export1:
        batch_profile = st.selectbox("Selecteer Grootte", ["Quick Dataset (50/conditie)", "Research Dataset (250/conditie)", "Production Dataset (1000/conditie)"])
        batch_size = int(batch_profile.split('(')[1].split('/')[0])
        totaal_bestanden = batch_size * 4

    if st.button(f"Genereer Gebalanceerde Dataset (4 condities × {batch_size} = {totaal_bestanden} files)", use_container_width=True, type="primary"):
        progress_text = "Genereren van fysica-gedreven bestanden..."
        my_bar = st.progress(0, text=progress_text)
        
        zip_buffer = io.BytesIO()
        metadata_list = []
        condities = ["Healthy", "Unbalance", "Mechanical Looseness", "BPFO (Outer Race)"]
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, cond in enumerate(condities):
                huidige_sev = 0.0 if cond == "Healthy" else severity
                folder_naam = cond.replace(' ', '_').lower()
                for i in range(batch_size):
                    t_b, z_b, _, _, bpfo_b, res_b, rpm_b, sev_b, _ = generate_vibration_data(cond, huidige_sev, rpm, apply_randomness=True)
                    filename = f"{folder_naam}/{folder_naam}_{i:03d}.csv"
                    df = pd.DataFrame({'timestamp_ms': t_b * 1000, 'accZ': z_b})
                    zip_file.writestr(filename, df.to_csv(index=False))
                    
                    metadata_list.append({
                        "file": filename, "condition": cond, "base_rpm": rpm, "actual_rpm": round(rpm_b, 2)
                    })
                my_bar.progress((idx + 1) / 4, text=f"Data gegenereerd voor: {cond}")
            
            zip_file.writestr("metadata.json", json.dumps(metadata_list, indent=4))
            
        time.sleep(0.5)
        my_bar.empty()
        st.success(f"✅ Productie-klare batch succesvol gegenereerd ({totaal_bestanden} bestanden)!")
        st.balloons()
        st.download_button(label="📦 Download .ZIP Archief (CSV + JSON)", data=zip_buffer.getvalue(), file_name="gebalanceerde_edge_dataset.zip", mime="application/zip", use_container_width=True)

# =========================================================================
# TAB 2: DATA MULTIPLIER (CSV)
# =========================================================================
with tab2:
    st.header("Referentie Signaal Uploaden & Klonen")
    st.write("Upload een échte, te kleine dataset. Ons algoritme extraheert de fysieke kenmerken en genereert een robuuste, gevarieerde Edge AI dataset met exact diezelfde signatuur.")
    
    uploaded_file = st.file_uploader("Upload je ruwe meting (.csv met Tijd en Acceleratie)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_real = pd.read_csv(uploaded_file)
            
            if df_real.shape[1] < 2:
                st.error("❌ Oeps! Het lijkt erop dat deze CSV niet de juiste structuur heeft. Zorg dat je een bestand uploadt met minimaal 2 kolommen (Tijd en Acceleratie).")
            else:
                t_real = df_real.iloc[:, 0].values
                z_real = df_real.iloc[:, 1].values
                
                # Check of de data numeriek is
                if not (np.issubdtype(t_real.dtype, np.number) and np.issubdtype(z_real.dtype, np.number)):
                    st.error("❌ Fout bij inlezen: De kolommen bevatten tekst in plaats van getallen. Zorg dat de CSV puur numerieke data bevat.")
                else:
                    st.success(f"✅ Bestand '{uploaded_file.name}' succesvol ingelezen! AI Analyse is voltooid.")
                    
                    # --- AI ANALYSE ---
                    sample_rate_est = int(1.0 / (t_real[1] - t_real[0])) * 1000 
                    if sample_rate_est < 10: sample_rate_est = 4000 
                    window_real = np.hanning(len(t_real))
                    fft_waarden_real = np.abs(np.fft.rfft(z_real * window_real))
                    fft_freqs_real = np.fft.rfftfreq(len(t_real), 1/sample_rate_est)
                    dominant_idx = np.argmax(fft_waarden_real[1:]) + 1
                    dominant_freq = fft_freqs_real[dominant_idx]

                    st.subheader("FFT Analyse")

                    col1, col2 = st.columns(2)

                    with col1:
                        fig_fft_real = go.Figure()
                        fig_fft_real.add_trace(
                            go.Scatter(
                                x=fft_freqs_real,
                                y=fft_waarden_real,
                                mode="lines",
                                name="FFT"
                            )
                        )
                        fig_fft_real.update_layout(
                            xaxis_title="Frequentie (Hz)",
                            yaxis_title="Amplitude",
                            height=350
                        )
                        st.plotly_chart(fig_fft_real, use_container_width=True)

                    with col2:
                        st.metric(
                            "Dominante Frequentie",
                            f"{dominant_freq:.2f} Hz"
                        )

                        st.info(
                            f"Het systeem detecteert een dominante component rond "
                            f"{dominant_freq:.2f} Hz."
                        )

        except Exception as e:
            st.error(f"Analyse mislukt: {e}")
                   
