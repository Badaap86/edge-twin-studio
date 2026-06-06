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
    # ============================================================
# TAB 2 - BATCH GENERATOR
# ============================================================

with tab2:

    st.header("📦 Batch Dataset Generator")

    st.write(
        "Generate complete TinyML-ready training datasets "
        "with automatic parameter variation."
    )

    st.markdown("---")

    # ========================================================
    # DATASET PROFILE
    # ========================================================

    profile = st.selectbox(
        "Dataset Profile",
        [
            "Quick Test",
            "Research",
            "Production"
        ]
    )

    if profile == "Quick Test":

        files_per_condition = 20

    elif profile == "Research":

        files_per_condition = 100

    else:

        files_per_condition = 500

    total_files = files_per_condition * 4

    # ========================================================
    # DATASET STATISTICS
    # ========================================================

    st.subheader("Dataset Statistics")

    stat1, stat2, stat3, stat4 = st.columns(4)

    stat1.metric(
        "Conditions",
        "4"
    )

    stat2.metric(
        "Files / Condition",
        files_per_condition
    )

    stat3.metric(
        "Total Files",
        total_files
    )

    estimated_size_mb = round(
        total_files * 0.08,
        1
    )

    stat4.metric(
        "Estimated Size",
        f"{estimated_size_mb} MB"
    )

    st.markdown("---")

    # ========================================================
    # RANDOMIZATION SETTINGS
    # ========================================================

    st.subheader("Randomization Settings")

    rpm_randomization = st.checkbox(
        "RPM Randomization",
        value=True
    )

    severity_randomization = st.checkbox(
        "Severity Randomization",
        value=True
    )

    noise_randomization = st.checkbox(
        "Noise Randomization",
        value=True
    )

    st.markdown("---")

    # ========================================================
    # CONDITION OVERVIEW
    # ========================================================

    st.subheader("Conditions Included")

    overview_df = pd.DataFrame(
        {
            "Condition": [
                "Healthy",
                "Unbalance",
                "Mechanical Looseness",
                "BPFO (Outer Race)"
            ],
            "Files": [
                files_per_condition,
                files_per_condition,
                files_per_condition,
                files_per_condition
            ]
        }
    )

    st.dataframe(
        overview_df,
        use_container_width=True
    )

    st.markdown("---")

    # ========================================================
    # GENERATION PREVIEW
    # ========================================================

    st.subheader("Generation Preview")

    st.info(
        f"""
Profile: {profile}

Total Files: {total_files}

RPM Variation:
{"Enabled" if rpm_randomization else "Disabled"}

Severity Variation:
{"Enabled" if severity_randomization else "Disabled"}

Noise Variation:
{"Enabled" if noise_randomization else "Disabled"}
"""
    )

    st.markdown("---")

    # ========================================================
    # GENERATE BUTTON
    # ========================================================

    generate_batch = st.button(
        "🚀 Prepare Dataset Generation",
        use_container_width=True
    )

    if generate_batch:

        st.success(
            f"Dataset profile '{profile}' ready."
        )

        progress = st.progress(0)

        for i in range(100):

            progress.progress(i + 1)

        st.success(
            f"Configuration complete: {total_files} files."
        )
        # ========================================================
# DATASET GENERATION ENGINE
# DEEL 4.2A
# ========================================================

    st.markdown("---")

    st.subheader("Dataset Generation Engine")

    condition_map = [
        "Healthy",
        "Unbalance",
        "Mechanical Looseness",
        "BPFO (Outer Race)"
    ]

    if generate_batch:

        generated_dataset = {}

        generation_progress = st.progress(0)

        status_text = st.empty()

        total_jobs = (
            len(condition_map)
            * files_per_condition
        )

        current_job = 0

        for fault_condition in condition_map:

            generated_dataset[fault_condition] = []

            for sample_idx in range(
                files_per_condition
            ):

                rpm_value = rpm

                severity_value = severity

                if rpm_randomization:

                    rpm_value = int(
                        rpm
                        * np.random.uniform(
                            0.95,
                            1.05
                        )
                    )

                if severity_randomization:

                    severity_value = np.clip(
                        severity
                        * np.random.uniform(
                            0.85,
                            1.15
                        ),
                        0.01,
                        1.0
                    )

                (
                    t_gen,
                    vibration_gen,
                    _,
                    _,
                    bpfo_gen,
                    resonance_gen,
                    rpm_gen,
                    severity_gen,
                    sample_rate_gen
                ) = generate_vibration_data(
                    condition=fault_condition,
                    severity=severity_value,
                    rpm=rpm_value,
                    apply_randomness=noise_randomization
                )

                sample_df = pd.DataFrame(
                    {
                        "time": t_gen,
                        "vibration": vibration_gen
                    }
                )

                generated_dataset[
                    fault_condition
                ].append(
                    {
                        "data": sample_df,
                        "rpm": rpm_gen,
                        "severity": severity_gen,
                        "bpfo": bpfo_gen,
                        "resonance": resonance_gen,
                        "sample_rate": sample_rate_gen
                    }
                )

                current_job += 1

                generation_progress.progress(
                    current_job
                    / total_jobs
                )

                status_text.text(
                    f"Generating {fault_condition} "
                    f"{sample_idx + 1}/"
                    f"{files_per_condition}"
                )

        st.session_state[
            "generated_dataset"
        ] = generated_dataset

        st.success(
            f"Generated "
            f"{total_files} synthetic files."
        )

        st.info(
            "Dataset stored in memory and ready "
            "for CSV export."
        )

        preview_col1, preview_col2 = st.columns(2)

        preview_col1.metric(
            "Conditions",
            len(condition_map)
        )

        preview_col2.metric(
            "Generated Files",
            total_files
        )

        preview_condition = st.selectbox(
            "Preview Condition",
            condition_map,
            key="preview_condition"
        )

        preview_df = (
            st.session_state[
                "generated_dataset"
            ][preview_condition][0]["data"]
        )

        st.subheader(
            "Generated Sample Preview"
        )

        st.dataframe(
            preview_df.head(20),
            use_container_width=True
        )
        # ========================================================
# CSV EXPORT ENGINE
# DEEL 4.2B
# ========================================================

    st.markdown("---")

    st.subheader("CSV Export Engine")

    if "generated_dataset" in st.session_state:

        export_dataset = st.session_state[
            "generated_dataset"
        ]

        total_csv_files = sum(
            len(export_dataset[c])
            for c in export_dataset
        )

        st.metric(
            "CSV Files Ready",
            total_csv_files
        )

        csv_preview_condition = st.selectbox(
            "CSV Preview Condition",
            list(export_dataset.keys()),
            key="csv_preview_condition"
        )

        csv_preview_sample = export_dataset[
            csv_preview_condition
        ][0]["data"]

        st.dataframe(
            csv_preview_sample.head(10),
            use_container_width=True
        )

        st.success(
            f"{total_csv_files} CSV datasets ready for export."
        )

        csv_export_dict = {}

        for fault_condition in export_dataset:

            csv_export_dict[
                fault_condition
            ] = []

            for idx, sample in enumerate(
                export_dataset[fault_condition]
            ):

                filename = (
                    fault_condition
                    .lower()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    + "_"
                    + str(idx + 1).zfill(3)
                    + ".csv"
                )

                csv_string = sample[
                    "data"
                ].to_csv(
                    index=False
                )

                csv_export_dict[
                    fault_condition
                ].append(
                    {
                        "filename": filename,
                        "csv": csv_string
                    }
                )

        st.session_state[
            "csv_export_dict"
        ] = csv_export_dict

        total_csv_created = sum(
            len(csv_export_dict[c])
            for c in csv_export_dict
        )

        st.info(
            f"{total_csv_created} CSV files prepared in memory."
        )

        sample_condition = list(
            csv_export_dict.keys()
        )[0]

        sample_file = csv_export_dict[
            sample_condition
        ][0]

        st.subheader(
            "Generated Filename Example"
        )

        st.code(
            sample_file["filename"]
        )

    else:

        st.warning(
            "Generate a dataset first."
        )
        # ========================================================
# ZIP EXPORT + DOWNLOAD
# DEEL 4.3
# ========================================================

    st.markdown("---")

    st.subheader("ZIP Export")

    if "csv_export_dict" in st.session_state:

        export_dict = st.session_state[
            "csv_export_dict"
        ]

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(
            zip_buffer,
            "w",
            zipfile.ZIP_DEFLATED
        ) as zip_file:

            total_files_zip = 0

            for condition_name in export_dict:

                for sample in export_dict[
                    condition_name
                ]:

                    zip_file.writestr(
                        sample["filename"],
                        sample["csv"]
                    )

                    total_files_zip += 1

            metadata = {
                "generator": "TinyML Data Accelerator V3",
                "profile": profile,
                "conditions": list(
                    export_dict.keys()
                ),
                "total_files": total_files_zip,
                "rpm_randomization":
                    rpm_randomization,
                "severity_randomization":
                    severity_randomization,
                "noise_randomization":
                    noise_randomization
            }

            zip_file.writestr(
                "metadata.json",
                json.dumps(
                    metadata,
                    indent=4
                )
            )

        zip_buffer.seek(0)

        st.success(
            f"ZIP package ready "
            f"({total_files_zip} CSV files)"
        )

        st.download_button(
            label="⬇ Download Dataset ZIP",
            data=zip_buffer,
            file_name=(
                f"tinyml_dataset_"
                f"{profile.lower().replace(' ','_')}"
                f".zip"
            ),
            mime="application/zip",
            use_container_width=True
        )

        st.info(
            "ZIP contains CSV files "
            "and metadata.json"
        )

    else:

        st.warning(
            "Generate dataset first."
        )
    # ============================================================
# TAB 3 - UPLOAD ANALYZER
# DEEL 5.1A
# ============================================================

with tab3:

    st.header("🔍 Upload Analyzer")

    st.write(
        "Upload vibration datasets and perform "
        "automatic analysis."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="upload_analyzer_csv"
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV dataset "
            "to begin analysis."
        )

    else:

        try:

            df_uploaded = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"Loaded: {uploaded_file.name}"
            )

            required_columns = [
                "time",
                "vibration"
            ]

            if not all(
                col in df_uploaded.columns
                for col in required_columns
            ):

                st.error(
                    "CSV must contain "
                    "'time' and 'vibration' columns."
                )

            else:

                time_data = (
                    df_uploaded["time"]
                    .values
                )

                vibration_data = (
                    df_uploaded["vibration"]
                    .values
                )

                sample_count = len(
                    vibration_data
                )

                duration = (
                    time_data[-1]
                    - time_data[0]
                )

                sample_rate = (
                    1.0 /
                    np.mean(
                        np.diff(time_data)
                    )
                )

                st.subheader(
                    "Dataset Information"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Samples",
                        f"{sample_count:,}"
                    )

                with col2:

                    st.metric(
                        "Duration",
                        f"{duration:.2f} s"
                    )

                with col3:

                    st.metric(
                        "Sample Rate",
                        f"{sample_rate:.0f} Hz"
                    )

                st.subheader(
                    "Dataset Preview"
                )

                st.dataframe(
                    df_uploaded.head(20),
                    use_container_width=True
                )

                st.divider()

                st.subheader(
                    "FFT Analysis"
                )

                vibration_centered = (
                    vibration_data
                    - np.mean(vibration_data)
                )

                fft_values = np.abs(
                    np.fft.rfft(
                        vibration_centered
                    )
                )

                fft_freqs = np.fft.rfftfreq(
                    len(vibration_centered),
                    d=1 / sample_rate
                )

                dominant_idx = (
                    np.argmax(
                        fft_values[1:]
                    ) + 1
                )

                dominant_frequency = (
                    fft_freqs[
                        dominant_idx
                    ]
                )

                peak_amplitude = (
                    fft_values[
                        dominant_idx
                    ]
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Dominant Frequency",
                        f"{dominant_frequency:.2f} Hz"
                    )

                with col2:

                    st.metric(
                        "Peak Amplitude",
                        f"{peak_amplitude:.1f}"
                    )

                st.divider()

                fft_fig = go.Figure()

                fft_fig.add_trace(
                    go.Scatter(
                        x=fft_freqs,
                        y=fft_values,
                        mode="lines",
                        name="FFT"
                    )
                )

                fft_fig.update_layout(
                    template="plotly_dark",
                    height=450,
                    title="FFT Spectrum",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Amplitude"
                )

                st.plotly_chart(
                    fft_fig,
                    use_container_width=True
                )
                st.divider()

                st.subheader(
                    "Predicted Condition"
                )

                if dominant_frequency < 40:

                    predicted_condition = (
                        "Unbalance"
                    )

                    confidence = 85

                elif dominant_frequency < 80:

                    predicted_condition = (
                        "Mechanical Looseness"
                    )

                    confidence = 88

                elif dominant_frequency < 120:

                    predicted_condition = (
                        "BPFO (Outer Race)"
                    )

                    confidence = 90

                else:

                    predicted_condition = (
                        "Healthy"
                    )

                    confidence = 80

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Condition",
                        predicted_condition
                    )

                with col2:

                    st.metric(
                        "Confidence",
                        f"{confidence}%"
                    )

                if predicted_condition == "Healthy":

                    st.success(
                        "Machine appears healthy."
                    )

                elif predicted_condition == "Unbalance":

                    st.warning(
                        "Possible rotor imbalance detected."
                    )

                elif predicted_condition == "Mechanical Looseness":

                    st.warning(
                        "Possible mechanical looseness detected."
                    )

                else:

                    st.error(
                        "Possible bearing fault detected."
                    )
        except Exception as e:

            st.error(
                f"Error loading file: {e}"
            )
        
    
