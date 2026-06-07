import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.stats import kurtosis
from scipy.io import wavfile
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from fpdf import FPDF
import io
import sqlite3
import json
import datetime
import uuid
import bcrypt
import secrets


# ============================================================
# DATABASE ENGINE - V16.1 SAAS FOUNDATION
# ============================================================

DB_NAME = "omega_v16.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            api_key TEXT UNIQUE,
            created_at TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            name TEXT,
            created_at TEXT,
            dataset TEXT,
            settings TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS api_usage (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            endpoint TEXT,
            created_at TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def create_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    try:
        user_id = str(uuid.uuid4())
        api_key = "omg_" + secrets.token_hex(16)
        pwd_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        c.execute(
            """
            INSERT INTO users (id, username, password_hash, api_key, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, username, pwd_hash, api_key, str(datetime.datetime.now())),
        )

        conn.commit()

        return {
            "id": user_id,
            "username": username,
            "api_key": api_key,
        }

    except sqlite3.IntegrityError:
        return None

    finally:
        conn.close()


def authenticate_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute(
        """
        SELECT id, username, password_hash, api_key
        FROM users
        WHERE username = ?
        """,
        (username,),
    )

    row = c.fetchone()
    conn.close()

    if row and bcrypt.checkpw(password.encode("utf-8"), row[2].encode("utf-8")):
        return {
            "id": row[0],
            "username": row[1],
            "api_key": row[3],
        }

    return None


def verify_api_key(api_key):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute(
        """
        SELECT id, username
        FROM users
        WHERE api_key = ?
        """,
        (api_key,),
    )

    row = c.fetchone()
    conn.close()

    if row:
        return {
            "id": row[0],
            "username": row[1],
        }

    return None


def log_api_usage(user_id, endpoint):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute(
        """
        INSERT INTO api_usage (id, user_id, endpoint, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (str(uuid.uuid4()), user_id, endpoint, str(datetime.datetime.now())),
    )

    conn.commit()
    conn.close()


def get_user_usage(user_id):
    conn = sqlite3.connect(DB_NAME)

    df = pd.read_sql_query(
        """
        SELECT endpoint, created_at
        FROM api_usage
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        conn,
        params=(user_id,),
    )

    conn.close()
    return df


def save_project(proj_id, user_id, name, dataset_df, settings_dict):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute(
        """
        REPLACE INTO projects (id, user_id, name, created_at, dataset, settings)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            proj_id,
            user_id,
            name,
            str(datetime.datetime.now()),
            dataset_df.to_json(orient="records"),
            json.dumps(settings_dict),
        ),
    )

    conn.commit()
    conn.close()


def load_project(proj_id, user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute(
        """
        SELECT name, dataset, settings
        FROM projects
        WHERE id = ? AND user_id = ?
        """,
        (proj_id, user_id),
    )

    row = c.fetchone()
    conn.close()

    if row:
        return {
            "name": row[0],
            "dataset": pd.read_json(io.StringIO(row[1])),
            "settings": json.loads(row[2]),
        }

    return None


def get_user_projects(user_id):
    conn = sqlite3.connect(DB_NAME)

    df = pd.read_sql_query(
        """
        SELECT id, name, created_at
        FROM projects
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        conn,
        params=(user_id,),
    )

    conn.close()
    return df


# ============================================================
# DSP ENGINE
# ============================================================

def calculate_fft(sig, sr):
    sig = np.asarray(sig, dtype=float)

    if len(sig) == 0:
        return np.array([]), np.array([])

    window = np.hanning(len(sig))
    fft_v = np.abs(np.fft.rfft(sig * window))
    fft_f = np.fft.rfftfreq(len(sig), 1 / sr)

    return fft_f, fft_v


def get_audio_features(sig, f_f, v_f):
    sig = np.asarray(sig, dtype=float)
    eps = 1e-10

    if len(sig) < 2 or len(f_f) == 0 or len(v_f) == 0:
        return 0.0, 0.0, 0.0, 0.0

    zcr = ((sig[:-1] * sig[1:]) < 0).sum() / len(sig)
    centroid = np.sum(f_f * v_f) / (np.sum(v_f) + eps)

    cum_e = np.cumsum(v_f)
    tot_e = cum_e[-1] if len(cum_e) > 0 else 0

    if tot_e > 0:
        idxs = np.where(cum_e >= 0.85 * tot_e)[0]
        rolloff = f_f[idxs[0]] if len(idxs) > 0 else 0.0
    else:
        rolloff = 0.0

    flatness = np.exp(np.mean(np.log(v_f + eps))) / (np.mean(v_f) + eps)

    return float(zcr), float(centroid), float(rolloff), float(flatness)


def reverse_engineer_physics(sig, sr):
    sig = np.asarray(sig, dtype=float)

    if len(sig) < 4:
        return {
            "base_f": 0.0,
            "harm_r": 0.0,
            "imp_r": 0.0,
            "noise": 0.0,
        }

    sig = sig - np.mean(sig)
    f_f, v_f = calculate_fft(sig, sr)

    if len(v_f) < 2:
        return {
            "base_f": 0.0,
            "harm_r": 0.0,
            "imp_r": 0.0,
            "noise": float(np.median(np.abs(sig))),
        }

    dom_idx = np.argmax(v_f[1:]) + 1
    ext_base = float(f_f[dom_idx])

    ext_harm = sum(
        v_f[np.argmin(np.abs(f_f - (h * ext_base)))]
        for h in range(2, 6)
    ) / max(v_f[dom_idx], 1e-6)

    peaks, _ = signal.find_peaks(
        np.abs(sig),
        height=np.mean(np.abs(sig)) + 2.5 * np.std(sig),
        distance=max(1, int(sr / 100)),
    )

    duration = len(sig) / sr if sr > 0 else 1.0
    impact_rate = len(peaks) / max(duration, 1e-6)

    return {
        "base_f": float(ext_base),
        "harm_r": float(ext_harm),
        "imp_r": float(impact_rate),
        "noise": float(np.median(np.abs(sig))),
    }


def extract_features_from_bytes(file_bytes, filename, sr=16000):
    try:
        filename = filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
            if df.shape[1] < 2:
                return {"error": "CSV must contain at least two columns: time and value."}
            sig = df.iloc[:, 1].astype(float).values

        elif filename.endswith(".wav"):
            s_ext, w_d = wavfile.read(io.BytesIO(file_bytes))
            sr = s_ext
            sig = (w_d.mean(axis=1) if len(w_d.shape) > 1 else w_d).astype(float)

        else:
            return {"error": "Unsupported file type. Use CSV or WAV."}

        sig = np.asarray(sig, dtype=float)
        sig = sig - np.mean(sig)

        f_f, v_f = calculate_fft(sig, sr)

        rms = float(np.sqrt(np.mean(sig ** 2)))
        zcr, cent, roll, flat = get_audio_features(sig, f_f, v_f)

        return {
            "RMS": rms,
            "Kurtosis": float(kurtosis(sig)),
            "CrestFactor": float(np.max(np.abs(sig)) / max(rms, 1e-6)),
            "ZCR": zcr,
            "SpectralCentroid": cent,
            "SpectralRolloff": roll,
            "SpectralFlatness": flat,
        }

    except Exception as e:
        return {"error": str(e)}


def generate_universal_signal(duration, sr, base_f, harm_r, imp_r, noise_l, normalize=True):
    duration = float(duration)
    sr = int(sr)

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    if base_f > 0:
        sig = np.sin(2 * np.pi * base_f * t)
    else:
        sig = np.zeros_like(t)

    if harm_r > 0 and base_f > 0:
        for h in range(2, 6):
            sig += (harm_r / h) * np.sin(2 * np.pi * (base_f * h) * t)

    if imp_r > 0:
        step = 1.0 / imp_r

        for i_t in np.arange(0, duration, step):
            idx = int((i_t + np.random.uniform(-0.02, 0.02)) * sr)

            if 0 <= idx < len(t):
                decay = np.exp(-25 * (t[idx:] - i_t))
                sig[idx:] += 2.5 * np.random.normal(0, 1, len(decay)) * decay

    sig += np.random.normal(0, noise_l, len(t))

    max_val = np.max(np.abs(sig)) if len(sig) else 0

    if normalize and max_val > 0:
        sig = sig / max_val

    f, v = calculate_fft(sig, sr)

    return {
        "t": t,
        "sig": sig,
        "fft_f": f,
        "fft_v": v,
        "sr": sr,
    }


# ============================================================
# EXPANDED HARDWARE PROFILER - V16.1
# ============================================================

HARDWARE_PROFILES = {
    "ESP32-S3": {
        "cpu": "Xtensa LX7 dual-core",
        "role": "Edge AI Node",
        "power_class": "Medium",
        "fft_mult": 0.00008,
        "feat_mult": 0.20,
        "inf_mult": 1.50,
        "target_ram": 320,
        "recommended_for": "Audio TinyML, vibration classification, WiFi/BLE edge nodes.",
        "gateway_fit": "No",
    },
    "ESP32-P4": {
        "cpu": "RISC-V high-performance MCU",
        "role": "High-performance Edge AI Node",
        "power_class": "Medium",
        "fft_mult": 0.000045,
        "feat_mult": 0.12,
        "inf_mult": 1.00,
        "target_ram": 768,
        "recommended_for": "Heavy TinyML, higher sample rates, camera/audio edge intelligence.",
        "gateway_fit": "Limited",
    },
    "ESP32-C6": {
        "cpu": "RISC-V WiFi 6 / BLE / Thread",
        "role": "Connected Low-cost Node",
        "power_class": "Low-Medium",
        "fft_mult": 0.00022,
        "feat_mult": 0.55,
        "inf_mult": 4.50,
        "target_ram": 320,
        "recommended_for": "Lightweight sensor inference, connected IoT nodes.",
        "gateway_fit": "No",
    },
    "RAK4631 / nRF52840": {
        "cpu": "ARM Cortex-M4F",
        "role": "Ultra-low-power LoRa Sensor Node",
        "power_class": "Very Low",
        "fft_mult": 0.00015,
        "feat_mult": 0.40,
        "inf_mult": 3.00,
        "target_ram": 256,
        "recommended_for": "LoRaWAN, battery-powered vibration, low-power remote sensing.",
        "gateway_fit": "LoRa node / relay",
    },
    "nRF5340": {
        "cpu": "Dual-core ARM Cortex-M33",
        "role": "Advanced BLE/Edge AI Node",
        "power_class": "Low",
        "fft_mult": 0.00010,
        "feat_mult": 0.25,
        "inf_mult": 2.00,
        "target_ram": 512,
        "recommended_for": "BLE sensor fusion, secure low-power edge inference.",
        "gateway_fit": "BLE gateway possible",
    },
    "nRF54L15": {
        "cpu": "ARM Cortex-M33 next-gen low power",
        "role": "Next-gen Ultra-low-power AI Node",
        "power_class": "Very Low",
        "fft_mult": 0.00009,
        "feat_mult": 0.22,
        "inf_mult": 1.80,
        "target_ram": 256,
        "recommended_for": "Future low-power BLE/TinyML deployments.",
        "gateway_fit": "BLE node",
    },
    "STM32L4": {
        "cpu": "ARM Cortex-M4F",
        "role": "Industrial Low-power Node",
        "power_class": "Low",
        "fft_mult": 0.00012,
        "feat_mult": 0.30,
        "inf_mult": 2.50,
        "target_ram": 320,
        "recommended_for": "Industrial low-power vibration and condition monitoring.",
        "gateway_fit": "No",
    },
    "STM32U5": {
        "cpu": "ARM Cortex-M33 ultra-low-power",
        "role": "Secure Industrial Low-power Node",
        "power_class": "Very Low",
        "fft_mult": 0.00009,
        "feat_mult": 0.22,
        "inf_mult": 1.80,
        "target_ram": 512,
        "recommended_for": "Secure industrial sensing, low-power TinyML.",
        "gateway_fit": "No",
    },
    "STM32H7": {
        "cpu": "ARM Cortex-M7 high-performance",
        "role": "High-performance Industrial Node",
        "power_class": "Medium-High",
        "fft_mult": 0.000035,
        "feat_mult": 0.10,
        "inf_mult": 0.80,
        "target_ram": 1024,
        "recommended_for": "High-speed vibration, audio DSP, industrial inference.",
        "gateway_fit": "Limited",
    },
    "RP2040": {
        "cpu": "Dual-core ARM Cortex-M0+",
        "role": "Low-cost Sensor Node",
        "power_class": "Medium",
        "fft_mult": 0.00080,
        "feat_mult": 1.50,
        "inf_mult": 12.00,
        "target_ram": 264,
        "recommended_for": "Low-cost simple feature extraction, not ideal for heavy FFT.",
        "gateway_fit": "No",
    },
    "RP2350": {
        "cpu": "Dual-core Cortex-M33 / RISC-V",
        "role": "Low-cost Improved Edge Node",
        "power_class": "Medium",
        "fft_mult": 0.00032,
        "feat_mult": 0.80,
        "inf_mult": 6.00,
        "target_ram": 520,
        "recommended_for": "Budget TinyML, improved over RP2040.",
        "gateway_fit": "No",
    },
    "Teensy 4.1": {
        "cpu": "ARM Cortex-M7 600 MHz",
        "role": "Fast DSP Prototyping Node",
        "power_class": "Medium-High",
        "fft_mult": 0.000025,
        "feat_mult": 0.08,
        "inf_mult": 0.70,
        "target_ram": 1024,
        "recommended_for": "Very fast audio/vibration DSP prototyping.",
        "gateway_fit": "No",
    },
    "Arduino Portenta H7": {
        "cpu": "Dual-core STM32H747",
        "role": "Professional Edge AI Node",
        "power_class": "Medium-High",
        "fft_mult": 0.000025,
        "feat_mult": 0.08,
        "inf_mult": 0.70,
        "target_ram": 8192,
        "recommended_for": "Professional prototyping, industrial AI, mixed workloads.",
        "gateway_fit": "Limited",
    },
    "Raspberry Pi Zero 2 W": {
        "cpu": "Quad-core ARM Cortex-A53",
        "role": "Light Linux Gateway",
        "power_class": "Medium",
        "fft_mult": 0.000015,
        "feat_mult": 0.05,
        "inf_mult": 0.40,
        "target_ram": 512000,
        "recommended_for": "Light gateway, MQTT, local preprocessing, dashboard bridge.",
        "gateway_fit": "Yes",
    },
    "Raspberry Pi 4": {
        "cpu": "Quad-core ARM Cortex-A72",
        "role": "Linux Gateway / Edge Server",
        "power_class": "High",
        "fft_mult": 0.000006,
        "feat_mult": 0.02,
        "inf_mult": 0.15,
        "target_ram": 2048000,
        "recommended_for": "Gateway, local server, MQTT, dashboards, heavier ML.",
        "gateway_fit": "Yes",
    },
    "Raspberry Pi 5": {
        "cpu": "Quad-core ARM Cortex-A76",
        "role": "High-performance Linux Gateway",
        "power_class": "High",
        "fft_mult": 0.000003,
        "feat_mult": 0.01,
        "inf_mult": 0.08,
        "target_ram": 4096000,
        "recommended_for": "Fast gateway, edge server, local model training/inference.",
        "gateway_fit": "Yes",
    },
    "Jetson Nano": {
        "cpu": "ARM + NVIDIA GPU",
        "role": "GPU Edge AI Gateway",
        "power_class": "High",
        "fft_mult": 0.000004,
        "feat_mult": 0.015,
        "inf_mult": 0.05,
        "target_ram": 4096000,
        "recommended_for": "GPU inference, camera/audio fusion, heavier AI workloads.",
        "gateway_fit": "Yes",
    },
    "Generic Linux Gateway": {
        "cpu": "x86/ARM Linux",
        "role": "Industrial Gateway / Server",
        "power_class": "High",
        "fft_mult": 0.000003,
        "feat_mult": 0.01,
        "inf_mult": 0.05,
        "target_ram": 8192000,
        "recommended_for": "Industrial gateway, API server, database, dashboards, model orchestration.",
        "gateway_fit": "Yes",
    },
}


def get_available_hardware():
    return list(HARDWARE_PROFILES.keys())


def get_hardware_profile(hw_name):
    return HARDWARE_PROFILES.get(hw_name)


def get_hardware_catalog():
    rows = []

    for name, profile in HARDWARE_PROFILES.items():
        rows.append({
            "board": name,
            "cpu": profile.get("cpu", ""),
            "role": profile.get("role", ""),
            "power_class": profile.get("power_class", ""),
            "target_ram_kb": profile.get("target_ram", 0),
            "recommended_for": profile.get("recommended_for", ""),
            "gateway_fit": profile.get("gateway_fit", "No"),
        })

    return pd.DataFrame(rows)


def estimate_edge_load(hw_name, feat_n, sr, duration=1.0):
    profile = HARDWARE_PROFILES.get(hw_name, HARDWARE_PROFILES["RAK4631 / nRF52840"])

    fft_n = 1024 if sr <= 4000 else 2048

    ram = (
        (min(int(sr * duration), 8192) * 4)
        + (fft_n * 8)
        + 2048
    ) / 1024

    fft_ms = (fft_n * np.log2(fft_n)) * profile["fft_mult"]
    feature_ms = feat_n * profile["feat_mult"]
    inference_ms = profile["inf_mult"]

    return float(ram), float(fft_ms), float(feature_ms), float(inference_ms)


def calculate_deployment_score(hw_name, latency, ram_kb):
    profile = HARDWARE_PROFILES.get(hw_name, HARDWARE_PROFILES["RAK4631 / nRF52840"])

    latency_score = max(0, 100 - (latency / 20.0) * 50)
    ram_score = max(0, 100 - (ram_kb / max(profile["target_ram"] / 2, 1)) * 50)

    base = (latency_score * 0.7) + (ram_score * 0.3)

    if profile["fft_mult"] < 0.00001:
        base += 15
    elif profile["fft_mult"] < 0.00005:
        base += 12
    elif profile["fft_mult"] < 0.00010:
        base += 10
    elif profile["fft_mult"] < 0.00020:
        base += 5

    if profile.get("power_class", "").lower().startswith("very low"):
        base += 3

    return float(min(100, max(0, base)))


def estimate_custom_hardware_load(
    hardware_name,
    ram_kb_available,
    fft_speed_factor,
    feature_speed_factor,
    inference_ms,
    feat_n,
    sr,
    duration=1.0
):
    fft_n = 1024 if sr <= 4000 else 2048

    ram_required = (
        (min(int(sr * duration), 8192) * 4)
        + (fft_n * 8)
        + 2048
    ) / 1024

    fft_ms = (fft_n * np.log2(fft_n)) * float(fft_speed_factor)
    feature_ms = feat_n * float(feature_speed_factor)
    total_latency = fft_ms + feature_ms + float(inference_ms)

    latency_score = max(0, 100 - (total_latency / 20.0) * 50)
    ram_score = max(0, 100 - (ram_required / max(ram_kb_available, 1)) * 100)

    score = (latency_score * 0.7) + (ram_score * 0.3)

    if ram_required > ram_kb_available:
        score = min(score, 45)

    result = {
        "board": hardware_name,
        "score": float(min(100, max(0, score))),
        "latency_ms": float(total_latency),
        "ram_required_kb": float(ram_required),
        "ram_available_kb": float(ram_kb_available),
        "fft_ms": float(fft_ms),
        "feature_ms": float(feature_ms),
        "inference_ms": float(inference_ms),
        "fits_ram": bool(ram_required <= ram_kb_available),
    }

    if result["score"] >= 85:
        verdict = "Excellent fit"
    elif result["score"] >= 70:
        verdict = "Good fit"
    elif result["score"] >= 50:
        verdict = "Prototype fit"
    else:
        verdict = "Not recommended"

    result["verdict"] = verdict

    return result


def hardware_auto_architect(num_features, sr, target="balanced", selected_boards=None):
    boards = selected_boards if selected_boards else get_available_hardware()
    results = []

    for board in boards:
        if board not in HARDWARE_PROFILES:
            continue

        profile = HARDWARE_PROFILES[board]

        ram, l_fft, l_feat, l_inf = estimate_edge_load(board, num_features, sr)
        latency = l_fft + l_feat + l_inf
        score = calculate_deployment_score(board, latency, ram)

        if target == "low_power":
            power_bonus = 0
            if profile["power_class"] == "Very Low":
                power_bonus = 12
            elif profile["power_class"] == "Low":
                power_bonus = 8
            elif profile["power_class"] == "Low-Medium":
                power_bonus = 5

            adjusted = score + power_bonus - (latency * 0.20)

        elif target == "performance":
            adjusted = score - (latency * 1.0)

        elif target == "gateway":
            gateway_bonus = 20 if profile.get("gateway_fit") == "Yes" else 0
            adjusted = score + gateway_bonus - (latency * 0.35)

        else:
            adjusted = score - (latency * 0.50)

        results.append(
            {
                "board": board,
                "cpu": profile.get("cpu", ""),
                "role": profile.get("role", ""),
                "power_class": profile.get("power_class", ""),
                "score": float(score),
                "adjusted_score": float(adjusted),
                "latency_ms": float(latency),
                "ram_kb": float(ram),
                "notes": profile.get("recommended_for", ""),
                "gateway_fit": profile.get("gateway_fit", "No"),
            }
        )

    results = sorted(results, key=lambda x: x["adjusted_score"], reverse=True)

    if not results:
        return {
            "recommendation": "Unknown",
            "reason": "No hardware profiles available.",
            "ranking": [],
            "node_recommendation": "Unknown",
            "gateway_recommendation": "Unknown",
        }

    best = results[0]

    node_candidates = [r for r in results if "Gateway" not in r["role"] and r["gateway_fit"] != "Yes"]
    gateway_candidates = [r for r in results if r["gateway_fit"] == "Yes"]

    best_node = node_candidates[0] if node_candidates else best
    best_gateway = gateway_candidates[0] if gateway_candidates else None

    if best_gateway:
        gateway_name = best_gateway["board"]
    else:
        gateway_name = "Linux Gateway / Raspberry Pi 4"

    fft_recommendation = 1024 if sr <= 4000 else 2048

    reason = (
        f"Beste keuze voor target '{target}' is {best['board']} "
        f"met {best['latency_ms']:.1f} ms latency en {best['ram_kb']:.1f} KB RAM."
    )

    return {
        "recommendation": best["board"],
        "node_recommendation": best_node["board"],
        "gateway_recommendation": gateway_name,
        "sample_rate": sr,
        "fft_recommendation": fft_recommendation,
        "reason": reason,
        "ranking": results,
    }


# ============================================================
# UNIVERSAL INDUSTRY PACKS
# ============================================================

INDUSTRY_PACKS = {
    "Rotating Machinery Pack": {
        "description": "Universeel predictive-maintenance pack voor motoren, pompen, ventilatoren, compressoren, generatoren en lagers.",
        "sample_rate": 4000,
        "classes": {
            "Normal": {"base_f": 50.0, "harm_r": 0.05, "imp_r": 0.0, "noise_l": 0.08},
            "Unbalance": {"base_f": 50.0, "harm_r": 0.30, "imp_r": 0.0, "noise_l": 0.12},
            "Misalignment": {"base_f": 50.0, "harm_r": 0.55, "imp_r": 1.0, "noise_l": 0.14},
            "Mechanical_Looseness": {"base_f": 50.0, "harm_r": 0.90, "imp_r": 4.0, "noise_l": 0.17},
            "Bearing_Wear": {"base_f": 50.0, "harm_r": 0.35, "imp_r": 14.0, "noise_l": 0.18},
            "Critical_Failure": {"base_f": 50.0, "harm_r": 1.35, "imp_r": 35.0, "noise_l": 0.34},
        },
    },
    "Predictive Maintenance Aging Pack": {
        "description": "Universeel aging/digital-twin pack: gezond, lichte slijtage, middelmatige slijtage, zware slijtage en falen.",
        "sample_rate": 4000,
        "classes": {
            "Healthy": {"base_f": 50.0, "harm_r": 0.05, "imp_r": 0.0, "noise_l": 0.07},
            "Wear_10": {"base_f": 50.2, "harm_r": 0.12, "imp_r": 2.0, "noise_l": 0.09},
            "Wear_25": {"base_f": 50.5, "harm_r": 0.25, "imp_r": 6.0, "noise_l": 0.12},
            "Wear_50": {"base_f": 51.0, "harm_r": 0.55, "imp_r": 14.0, "noise_l": 0.18},
            "Wear_75": {"base_f": 51.8, "harm_r": 0.95, "imp_r": 25.0, "noise_l": 0.26},
            "Failure": {"base_f": 52.5, "harm_r": 1.45, "imp_r": 38.0, "noise_l": 0.36},
        },
    },
    "Acoustic Event Detection Pack": {
        "description": "Universeel audio-event pack voor TinyML sound detection: background, engine, tools, impact, alarm en menselijke activiteit.",
        "sample_rate": 16000,
        "classes": {
            "Background": {"base_f": 0.0, "harm_r": 0.0, "imp_r": 0.0, "noise_l": 0.08},
            "Engine": {"base_f": 45.0, "harm_r": 0.65, "imp_r": 0.0, "noise_l": 0.20},
            "Cutting_Tool": {"base_f": 190.0, "harm_r": 1.35, "imp_r": 6.0, "noise_l": 0.30},
            "Impact": {"base_f": 0.0, "harm_r": 0.0, "imp_r": 18.0, "noise_l": 0.18},
            "Alarm": {"base_f": 850.0, "harm_r": 0.45, "imp_r": 0.0, "noise_l": 0.10},
            "Human_Activity": {"base_f": 120.0, "harm_r": 0.25, "imp_r": 4.0, "noise_l": 0.16},
        },
    },
    "Security & Tamper Pack": {
        "description": "Universeel security/tamper pack voor bouwplaatsen, containers, gevels, machines en remote assets.",
        "sample_rate": 16000,
        "classes": {
            "Normal_Background": {"base_f": 0.0, "harm_r": 0.0, "imp_r": 0.0, "noise_l": 0.08},
            "Impact_Tamper": {"base_f": 0.0, "harm_r": 0.0, "imp_r": 25.0, "noise_l": 0.20},
            "Grinding": {"base_f": 180.0, "harm_r": 1.50, "imp_r": 0.0, "noise_l": 0.32},
            "Drilling": {"base_f": 130.0, "harm_r": 1.10, "imp_r": 8.0, "noise_l": 0.28},
            "Cutting": {"base_f": 260.0, "harm_r": 1.80, "imp_r": 12.0, "noise_l": 0.36},
            "Climbing_Or_Handling": {"base_f": 35.0, "harm_r": 0.25, "imp_r": 5.0, "noise_l": 0.15},
            "Vehicle_Nearby": {"base_f": 45.0, "harm_r": 0.60, "imp_r": 0.0, "noise_l": 0.22},
        },
    },
    "Environmental Monitoring Pack": {
        "description": "Universeel buitenomgeving-pack voor wind, regen, onweer, dieren, mensen, voertuigen en machines.",
        "sample_rate": 16000,
        "classes": {
            "Quiet_Background": {"base_f": 0.0, "harm_r": 0.0, "imp_r": 0.0, "noise_l": 0.06},
            "Wind": {"base_f": 8.0, "harm_r": 0.12, "imp_r": 0.0, "noise_l": 0.18},
            "Rain": {"base_f": 0.0, "harm_r": 0.0, "imp_r": 30.0, "noise_l": 0.22},
            "Thunder_Or_Low_Rumble": {"base_f": 25.0, "harm_r": 0.35, "imp_r": 1.5, "noise_l": 0.25},
            "Animal_Or_Bird": {"base_f": 450.0, "harm_r": 0.35, "imp_r": 3.0, "noise_l": 0.12},
            "Human_Activity": {"base_f": 95.0, "harm_r": 0.25, "imp_r": 4.0, "noise_l": 0.15},
            "Vehicle": {"base_f": 45.0, "harm_r": 0.55, "imp_r": 0.0, "noise_l": 0.22},
            "Machinery": {"base_f": 80.0, "harm_r": 0.85, "imp_r": 3.0, "noise_l": 0.25},
        },
    },
    "Forestry & Remote Asset Pack": {
        "description": "Pack voor bosbouw, landbouw, afgelegen assets en remote site monitoring.",
        "sample_rate": 16000,
        "classes": {
            "Forest_Normal": {"base_f": 0.0, "harm_r": 0.0, "imp_r": 0.0, "noise_l": 0.08},
            "Chainsaw": {"base_f": 85.0, "harm_r": 1.20, "imp_r": 0.0, "noise_l": 0.28},
            "Offroad_Vehicle": {"base_f": 42.0, "harm_r": 0.70, "imp_r": 1.0, "noise_l": 0.24},
            "Human_Movement": {"base_f": 70.0, "harm_r": 0.25, "imp_r": 5.0, "noise_l": 0.14},
            "Branch_Crack": {"base_f": 0.0, "harm_r": 0.0, "imp_r": 8.0, "noise_l": 0.15},
            "Rain_And_Wind": {"base_f": 12.0, "harm_r": 0.12, "imp_r": 22.0, "noise_l": 0.26},
            "Remote_Machinery": {"base_f": 60.0, "harm_r": 0.90, "imp_r": 2.0, "noise_l": 0.24},
        },
    },
}


def get_industry_packs():
    return list(INDUSTRY_PACKS.keys())


def get_industry_pack(pack_name):
    return INDUSTRY_PACKS.get(pack_name)


def generate_industry_pack_dataset(pack_name, samples_per_class=100):
    pack = get_industry_pack(pack_name)

    if not pack:
        return [], {"error": f"Unknown industry pack: {pack_name}"}

    sr = pack["sample_rate"]
    files = []

    for label, params in pack["classes"].items():
        for i in range(samples_per_class):
            p = {
                "base_f": max(
                    0.0,
                    params["base_f"] + np.random.normal(0, max(params["base_f"] * 0.02, 0.5)),
                ),
                "harm_r": max(
                    0.0,
                    params["harm_r"] + np.random.normal(0, 0.05),
                ),
                "imp_r": max(
                    0.0,
                    params["imp_r"] + np.random.normal(0, max(params["imp_r"] * 0.04, 0.2)),
                ),
                "noise_l": max(
                    0.001,
                    params["noise_l"] * np.random.uniform(0.85, 1.20),
                ),
            }

            d = generate_universal_signal(
                duration=2.0,
                sr=sr,
                base_f=p["base_f"],
                harm_r=p["harm_r"],
                imp_r=p["imp_r"],
                noise_l=p["noise_l"],
                normalize=True,
            )

            df = pd.DataFrame({"time": d["t"], "value": d["sig"]})

            files.append(
                {
                    "filename": f"{label}_{i:04d}.csv",
                    "label": label,
                    "dataframe": df,
                    "params": p,
                }
            )

    manifest = {
        "pack_name": pack_name,
        "description": pack["description"],
        "sample_rate": sr,
        "samples_per_class": samples_per_class,
        "classes": list(pack["classes"].keys()),
        "total_files": len(files),
    }

    return files, manifest


# ============================================================
# SYNTHETIC AGING ENGINE
# ============================================================

def apply_aging_profile(base_params, aging_percent):
    aging = max(0.0, min(100.0, float(aging_percent))) / 100.0

    aged = dict(base_params)

    aged["harm_r"] = float(base_params.get("harm_r", 0.0)) * (1.0 + 2.8 * aging)
    aged["imp_r"] = float(base_params.get("imp_r", 0.0)) + (aging ** 1.6) * 35.0
    aged["noise_l"] = float(base_params.get("noise_l", 0.05)) * (1.0 + 2.2 * aging)
    aged["base_f"] = float(base_params.get("base_f", 0.0)) * (1.0 + 0.03 * aging)

    return aged


def generate_aging_stages(base_params):
    stages = {
        "Healthy_0": 0,
        "Wear_25": 25,
        "Wear_50": 50,
        "Wear_75": 75,
        "Failure_100": 100,
    }

    return {
        label: apply_aging_profile(base_params, aging)
        for label, aging in stages.items()
    }


def generate_predictive_maintenance_aging_dataset(base_params, samples_per_stage=100, sr=4000):
    stages = generate_aging_stages(base_params)
    files = []

    for label, params in stages.items():
        for i in range(samples_per_stage):
            p = {
                "base_f": max(
                    0.0,
                    params["base_f"] + np.random.normal(0, max(params["base_f"] * 0.015, 0.3)),
                ),
                "harm_r": max(
                    0.0,
                    params["harm_r"] + np.random.normal(0, 0.04),
                ),
                "imp_r": max(
                    0.0,
                    params["imp_r"] + np.random.normal(0, max(params["imp_r"] * 0.04, 0.1)),
                ),
                "noise_l": max(
                    0.001,
                    params["noise_l"] * np.random.uniform(0.9, 1.15),
                ),
            }

            d = generate_universal_signal(
                duration=2.0,
                sr=sr,
                base_f=p["base_f"],
                harm_r=p["harm_r"],
                imp_r=p["imp_r"],
                noise_l=p["noise_l"],
                normalize=True,
            )

            df = pd.DataFrame({"time": d["t"], "value": d["sig"]})

            files.append(
                {
                    "filename": f"{label}_{i:04d}.csv",
                    "label": label,
                    "dataframe": df,
                    "params": p,
                }
            )

    manifest = {
        "engine": "OMEGA-X Synthetic Aging Engine",
        "sample_rate": sr,
        "samples_per_stage": samples_per_stage,
        "stages": list(stages.keys()),
        "total_files": len(files),
        "base_params": base_params,
    }

    return files, manifest


# ============================================================
# ML AUDIT ENGINE
# ============================================================

def calculate_audit_scores(X_df, y_series):
    if len(X_df) == 0:
        return 0, 0, 0

    v_c = y_series.value_counts()

    try:
        X_scaled = StandardScaler().fit_transform(X_df)
    except Exception:
        return 0, 0, 0

    if len(X_scaled) > 1:
        div = min(100, int((np.mean(pdist(X_scaled)) / 4.0) * 100))
    else:
        div = 0

    if len(v_c) >= 2 and v_c.max() > 0 and (v_c.min() / v_c.max()) > 0.5:
        bal = 100
    else:
        bal = 50

    if len(y_series.unique()) >= 2 and v_c.min() >= 2:
        try:
            sep = int((silhouette_score(X_scaled, y_series) + 1) * 50)
        except Exception:
            sep = 0
    else:
        sep = 0

    return div, bal, sep


def dataset_doctor(X_df, y_series, feature_importance=None):
    advice = []
    severity = []

    div, bal, sep = calculate_audit_scores(X_df, y_series)
    label_counts = y_series.value_counts()

    if len(X_df) < 50:
        advice.append("Dataset is erg klein. Voeg minimaal 50-100 samples toe voor een betrouwbare eerste test.")
        severity.append("high")
    elif len(X_df) < 200:
        advice.append("Dataset is bruikbaar voor prototype, maar nog klein voor productie. Richt op 200+ samples.")
        severity.append("medium")

    if len(label_counts) < 2:
        advice.append("Er is maar één label aanwezig. Voeg minimaal één extra klasse toe voor classificatie.")
        severity.append("high")
    elif label_counts.max() > 0 and label_counts.min() / label_counts.max() < 0.5:
        weakest = label_counts.idxmin()
        advice.append(f"Klasse '{weakest}' heeft te weinig samples. Voeg extra voorbeelden toe voor betere balans.")
        severity.append("high")

    if div < 50:
        advice.append("Dataset-diversiteit is laag. Voeg jitter toe in frequentie, ruis, impact-rate en amplitude.")
        severity.append("high")
    elif div < 75:
        advice.append("Dataset-diversiteit is redelijk, maar kan beter. Maak meer variaties per label.")
        severity.append("medium")

    if len(label_counts) >= 2:
        if sep < 50:
            advice.append("Label separation is laag. Klassen lijken te veel op elkaar in feature-space.")
            severity.append("high")
        elif sep < 75:
            advice.append("Label separation is middelmatig. Extra onderscheidende features kunnen helpen.")
            severity.append("medium")

    for col in X_df.columns:
        std = X_df[col].std()
        mean = X_df[col].mean()

        if pd.isna(std) or std == 0:
            advice.append(f"Feature '{col}' heeft nul variantie en kan waarschijnlijk worden verwijderd.")
            severity.append("medium")
        elif abs(std / max(abs(mean), 1e-6)) < 0.03:
            advice.append(f"Feature '{col}' heeft lage variantie. Controleer of deze echt nuttig is.")
            severity.append("low")

    if len(X_df.columns) > 1:
        corr = X_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        redundant = [column for column in upper.columns if any(upper[column] > 0.95)]

        if redundant:
            advice.append(f"Redundante features gevonden: {', '.join(redundant)}. Overweeg pruning voor lagere latency.")
            severity.append("medium")

    if feature_importance:
        weak_features = [f for f, s in feature_importance if s <= 0.1]

        if weak_features:
            advice.append(f"Lage feature impact: {', '.join(weak_features[:5])}. Deze kunnen mogelijk weg.")
            severity.append("low")

    if not advice:
        advice.append("Dataset ziet er gezond uit voor een eerste ML-training.")
        severity.append("info")

    return {
        "diversity_score": div,
        "balance_score": bal,
        "separation_score": sep,
        "overall_score": int((div * 0.35) + (bal * 0.30) + (sep * 0.35)),
        "advice": [{"severity": s, "message": a} for s, a in zip(severity, advice)],
    }


# ============================================================
# PDF REPORT ENGINE
# ============================================================

def generate_pdf_report(proj_name, num_samples, num_classes, div, bal, sep, top_features, b_dat, best_board):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(200, 10, txt="OMEGA-X Enterprise Audit", ln=True, align="C")

    pdf.set_font("Arial", "I", 12)
    pdf.cell(200, 10, txt=f"Project ID: {proj_name}", ln=True, align="C")

    pdf.ln(10)

    overall_status = "PRODUCTION READY" if all(s > 80 for s in [div, bal, sep]) else "OPTIMIZATION REQUIRED"

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt=f"Overall Status: {overall_status}", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, txt=f"Total Samples: {num_samples} | Unique Classes: {num_classes}", ln=True)
    pdf.cell(200, 8, txt=f"Dataset Diversity: {div}% | Class Balance: {bal}% | Label Separation: {sep}%", ln=True)

    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Top Features (Permutation Importance):", ln=True)

    pdf.set_font("Arial", "", 11)

    if top_features:
        for f, score in top_features[:5]:
            pdf.cell(200, 7, txt=f"- {f}: {score:.1f}% impact", ln=True)
    else:
        pdf.cell(200, 7, txt="Not enough data to calculate feature importance.", ln=True)

    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt=f"Hardware Recommendation: {best_board}", ln=True)

    pdf.set_font("Arial", "", 11)

    for d in b_dat:
        pdf.cell(
            200,
            7,
            txt=f"- {d['Board']}: Score {d['Score']:.0f}% | Latency: {d['Latency']:.1f} ms | RAM: {d.get('RAM', 0):.1f} KB",
            ln=True,
        )

    return pdf.output(dest="S").encode("latin1")
