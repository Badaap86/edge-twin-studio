import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.signal as signal
import io
import json
import zipfile
import time

# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="TinyML Data Accelerator V3",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ TinyML Data Accelerator V3")
st.caption(
    "Synthetic Data Generation • Dataset Augmentation • "
    "Fault Analysis • Edge AI Development"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_fft(signal_data, sample_rate):

    window = np.hanning(len(signal_data))

    fft_values = np.abs(
        np.fft.rfft(
            signal_data * window
        )
    )

    fft_freqs = np.fft.rfftfreq(
        len(signal_data),
        1 / sample_rate
    )

    return fft_freqs, fft_values


def calculate_spectrogram(signal_data, sample_rate):

    f_spec, t_spec, Sxx = signal.spectrogram(
        signal_data,
        sample_rate,
        nperseg=256,
        noverlap=128
    )

    return f_spec, t_spec, Sxx


def estimate_sample_rate(time_vector):

    try:

        dt = np.mean(
            np.diff(time_vector)
        )

        if dt <= 0:
            return 4000

        fs = int(1.0 / dt)

        if fs < 10:
            return 4000

        return fs

    except:

        return 4000


def rms(signal_data):

    return np.sqrt(
        np.mean(
            signal_data ** 2
        )
    )


# ============================================================
# SYNTHETIC VIBRATION ENGINE V3
# ============================================================

@st.cache_data
def generate_vibration_data(
    condition,
    severity,
    rpm,
    duration=2.0,
    apply_randomness=False
):

    sample_rate = 4000

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False
    )

    actual_rpm = rpm

    if apply_randomness:

        actual_rpm = rpm * np.random.normal(
            1.0,
            0.02
        )

    actual_severity = severity

    if apply_randomness:

        actual_severity = severity * np.random.normal(
            1.0,
            0.10
        )

    actual_severity = np.clip(
        actual_severity,
        0.01,
        1.0
    )

    f_1x = actual_rpm / 60.0

    phase_1x = (
        2
        * np.pi
        * f_1x
        * t
    )

    vibration = (
        0.2
        * np.sin(phase_1x)
    )

    mains_noise = (
        0.05
        * np.sin(
            2
            * np.pi
            * 50
            * t
        )
    )

    noise_scale = 1.0

    if apply_randomness:

        noise_scale = np.random.uniform(
            0.85,
            1.15
        )

    base_noise = np.random.normal(
        0,
        0.1 * noise_scale,
        len(t)
    )

    vibration += mains_noise

    vibration += (
        base_noise
        * (1 + actual_severity)
    )

    bpfo_frequency = f_1x * 3.58

    resonance_frequency = 1200.0

    if apply_randomness:

        resonance_frequency = np.random.normal(
            1200.0,
            15.0
        )

    # ========================================================
    # HEALTHY
    # ========================================================

    if condition == "Healthy":

        pass

    # ========================================================
    # UNBALANCE
    # ========================================================

    elif condition == "Unbalance":

        vibration += (
            1.5
            * actual_severity
            * np.sin(
                phase_1x
            )
        )

    # ========================================================
    # MECHANICAL LOOSENESS
    # ========================================================

    elif condition == "Mechanical Looseness":

        for harmonic in range(1, 6):

            vibration += (
                (1.2 / harmonic)
                * actual_severity
                * np.sin(
                    harmonic
                    * phase_1x
                )
            )

        vibration += (
            base_noise
            * actual_severity
            * 2.0
        )

    # ========================================================
    # BPFO OUTER RACE
    # ========================================================

    elif condition == "BPFO (Outer Race)":

        phase_fault = (
            2
            * np.pi
            * bpfo_frequency
            * t
        )

        impact_variation = (
            0.8
            + 0.4
            * np.random.rand(
                len(t)
            )
        )

        impact_envelope = (
            np.maximum(
                0,
                np.cos(
                    phase_fault
                )
            ) ** 20
        )

        impact_envelope *= (
            impact_variation
        )

        load_zone_modulation = (
            1
            + 0.6
            * np.cos(
                phase_1x
            )
        )

        defect_signal = (
            np.sin(
                2
                * np.pi
                * resonance_frequency
                * t
            )
            * impact_envelope
            * load_zone_modulation
        )

        vibration += (
            2.0
            * actual_severity
            * defect_signal
        )

    fft_freqs, fft_values = calculate_fft(
        vibration,
        sample_rate
    )

    return (
        t,
        vibration,
        fft_freqs,
        fft_values,
        bpfo_frequency,
        resonance_frequency,
        actual_rpm,
        actual_severity,
        sample_rate
    )


# ============================================================
# SESSION STATE
# ============================================================

if "rpm" not in st.session_state:

    st.session_state.rpm = 1500

if "severity" not in st.session_state:

    st.session_state.severity = 80
    # ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Generator Controls")

rpm = st.sidebar.slider(
    "RPM",
    min_value=600,
    max_value=3000,
    value=st.session_state.rpm,
    step=10
)

severity_percent = st.sidebar.slider(
    "Severity (%)",
    min_value=0,
    max_value=100,
    value=st.session_state.severity,
    step=5
)

severity = severity_percent / 100.0

condition = st.sidebar.selectbox(
    "Condition",
    [
        "Healthy",
        "Unbalance",
        "Mechanical Looseness",
        "BPFO (Outer Race)"
    ]
)

show_healthy_overlay = st.sidebar.checkbox(
    "Show Healthy Reference",
    value=True
)

apply_randomness = st.sidebar.checkbox(
    "Apply Dataset Randomization",
    value=True
)

st.session_state.rpm = rpm
st.session_state.severity = severity_percent

st.sidebar.markdown("---")

st.sidebar.info(
    "TinyML Data Accelerator V3\n\n"
    "Synthetic Data + Fault Analysis + Dataset Augmentation"
)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Synthetic Generator",
        "📦 Batch Generator",
        "🔍 Upload Analyzer",
        "🧪 Data Multiplier"
    ]
)

# ============================================================
# GENERATE ACTIVE DATASET
# ============================================================

(
    t,
    vibration,
    fft_freqs,
    fft_values,
    bpfo_frequency,
    resonance_frequency,
    actual_rpm,
    actual_severity,
    sample_rate
) = generate_vibration_data(
    condition=condition,
    severity=severity,
    rpm=rpm,
    apply_randomness=apply_randomness
)

# Healthy reference

(
    t_ref,
    vibration_ref,
    fft_freqs_ref,
    fft_values_ref,
    _,
    _,
    _,
    _,
    _
) = generate_vibration_data(
    condition="Healthy",
    severity=0.0,
    rpm=rpm,
    apply_randomness=False
)

# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.header("Synthetic Condition Simulator")

    st.write(
        "Generate physically explainable vibration datasets "
        "for TinyML, Edge AI and Predictive Maintenance."
    )

    col_signal, col_fft = st.columns(2)

    # --------------------------------------------------------
    # TIME DOMAIN
    # --------------------------------------------------------

    with col_signal:

        st.subheader("Time Signal")

        fig_signal = go.Figure()

        if show_healthy_overlay:

            fig_signal.add_trace(
                go.Scatter(
                    x=t[:800],
                    y=vibration_ref[:800],
                    mode="lines",
                    name="Healthy Reference"
                )
            )

        fig_signal.add_trace(
            go.Scatter(
                x=t[:800],
                y=vibration[:800],
                mode="lines",
                name=condition
            )
        )

        fig_signal.update_layout(
            height=400,
            xaxis_title="Time (s)",
            yaxis_title="Amplitude"
        )

        st.plotly_chart(
            fig_signal,
            use_container_width=True
        )

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    with col_fft:

        st.subheader("FFT Spectrum")

        fig_fft = go.Figure()

        if show_healthy_overlay:

            fig_fft.add_trace(
                go.Scatter(
                    x=fft_freqs_ref,
                    y=fft_values_ref,
                    mode="lines",
                    name="Healthy Reference"
                )
            )

        fig_fft.add_trace(
            go.Scatter(
                x=fft_freqs,
                y=fft_values,
                mode="lines",
                name=condition
            )
        )

        fig_fft.update_layout(
            height=400,
            xaxis_title="Frequency (Hz)",
            yaxis_title="Amplitude"
        )

        st.plotly_chart(
            fig_fft,
            use_container_width=True
        )

    st.markdown("---")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "RPM",
        f"{actual_rpm:.0f}"
    )

    metric2.metric(
        "1× RPM",
        f"{actual_rpm/60:.2f} Hz"
    )

    metric3.metric(
        "BPFO",
        f"{bpfo_frequency:.2f} Hz"
    )

    metric4.metric(
        "Resonance",
        f"{resonance_frequency:.1f} Hz"
    )

    st.markdown("---")
        # ========================================================
    # SPECTROGRAM
    # ========================================================

    st.subheader("Spectrogram (Time / Frequency)")

    f_spec, t_spec, Sxx = calculate_spectrogram(
        vibration,
        sample_rate
    )

    fig_spec = go.Figure(
        data=go.Heatmap(
            z=10 * np.log10(
                Sxx + 1e-12
            ),
            x=t_spec,
            y=f_spec,
            colorscale="Viridis"
        )
    )

    fig_spec.update_layout(
        height=450,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)"
    )

    st.plotly_chart(
        fig_spec,
        use_container_width=True
    )

    st.markdown("---")

    # ========================================================
    # PHYSICS DASHBOARD
    # ========================================================

    st.subheader("Physics Dashboard")

    col_a, col_b, col_c, col_d = st.columns(4)

    col_a.metric(
        "RPM",
        f"{actual_rpm:.0f}"
    )

    col_b.metric(
        "1× RPM",
        f"{actual_rpm/60:.2f} Hz"
    )

    col_c.metric(
        "BPFO",
        f"{bpfo_frequency:.2f} Hz"
    )

    col_d.metric(
        "Resonance",
        f"{resonance_frequency:.1f} Hz"
    )

    st.markdown("---")

    # ========================================================
    # MODEL READINESS SCORE
    # ========================================================

    st.subheader("Model Readiness")

    readiness_score = 50

    if apply_randomness:
        readiness_score += 20

    if show_healthy_overlay:
        readiness_score += 10

    if severity_percent > 0:
        readiness_score += 10

    if condition != "Healthy":
        readiness_score += 10

    readiness_score = min(
        readiness_score,
        100
    )

    st.progress(
        readiness_score
    )

    readiness_col1, readiness_col2, readiness_col3 = st.columns(3)

    readiness_col1.metric(
        "Readiness Score",
        f"{readiness_score}%"
    )

    readiness_col2.metric(
        "Randomization",
        "Enabled" if apply_randomness else "Disabled"
    )

    readiness_col3.metric(
        "Condition",
        condition
    )

    if readiness_score >= 90:

        st.success(
            "Dataset appears suitable for TinyML training."
        )

    elif readiness_score >= 70:

        st.warning(
            "Dataset is usable but could benefit from more variation."
        )

    else:

        st.error(
            "Dataset variation may be insufficient."
        )

    st.markdown("---")

    # ========================================================
    # FAULT FINGERPRINT
    # ========================================================

    st.subheader("Expected Fault Fingerprint")

    if condition == "Healthy":

        fingerprint = pd.DataFrame(
            {
                "Feature": [
                    "1× RPM",
                    "Harmonics",
                    "Impacts",
                    "Resonance"
                ],
                "Expected":
                [
                    "Low",
                    "Low",
                    "None",
                    "Low"
                ]
            }
        )

    elif condition == "Unbalance":

        fingerprint = pd.DataFrame(
            {
                "Feature": [
                    "1× RPM",
                    "Harmonics",
                    "Impacts",
                    "Resonance"
                ],
                "Expected":
                [
                    "Very High",
                    "Low",
                    "None",
                    "Low"
                ]
            }
        )

    elif condition == "Mechanical Looseness":

        fingerprint = pd.DataFrame(
            {
                "Feature": [
                    "1× RPM",
                    "2× RPM",
                    "3× RPM",
                    "4× RPM"
                ],
                "Expected":
                [
                    "High",
                    "High",
                    "Medium",
                    "Medium"
                ]
            }
        )

    else:

        fingerprint = pd.DataFrame(
            {
                "Feature": [
                    "BPFO",
                    "Impacts",
                    "Sidebands",
                    "Resonance"
                ],
                "Expected":
                [
                    "High",
                    "High",
                    "Present",
                    "Strong"
                ]
            }
        )

    st.dataframe(
        fingerprint,
        use_container_width=True
    )

    st.markdown("---")

    st.info(
        "Synthetic signal generated using DSP-based "
        "kinematic fault modelling."
    )
    
