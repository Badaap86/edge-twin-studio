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
    
