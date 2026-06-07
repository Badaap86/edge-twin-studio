import io
import json
import uuid
import secrets
import sqlite3
import zipfile
import datetime

import bcrypt
import numpy as np
import pandas as pd
import scipy.signal as signal
from fpdf import FPDF
from scipy.io import wavfile
from scipy.stats import kurtosis
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

DB_NAME = "edgetwin_v19.db"


# ============================================================
# DATABASE
# ============================================================

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            api_key TEXT UNIQUE,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            name TEXT,
            created_at TEXT,
            dataset TEXT,
            settings TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS api_usage (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            endpoint TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def create_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        user_id = str(uuid.uuid4())
        api_key = "ets_" + secrets.token_hex(16)
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        c.execute(
            "INSERT INTO users (id, username, password_hash, api_key, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, password_hash, api_key, str(datetime.datetime.now())),
        )
        conn.commit()
        return {"id": user_id, "username": username, "api_key": api_key}
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, username, password_hash, api_key FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    if bcrypt.checkpw(password.encode("utf-8"), row[2].encode("utf-8")):
        return {"id": row[0], "username": row[1], "api_key": row[3]}
    return None


def save_project(proj_id, user_id, name, dataset_df, settings_dict):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    dataset_json = dataset_df.to_json(orient="records") if isinstance(dataset_df, pd.DataFrame) else "[]"
    c.execute(
        """
        REPLACE INTO projects (id, user_id, name, created_at, dataset, settings)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (proj_id, user_id, name, str(datetime.datetime.now()), dataset_json, json.dumps(settings_dict)),
    )
    conn.commit()
    conn.close()


def load_project(proj_id, user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name, dataset, settings FROM projects WHERE id = ? AND user_id = ?", (proj_id, user_id))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "name": row[0],
        "dataset": pd.read_json(io.StringIO(row[1])) if row[1] else pd.DataFrame(),
        "settings": json.loads(row[2]) if row[2] else {},
    }


def get_user_projects(user_id):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(
        "SELECT id, name, created_at FROM projects WHERE user_id = ? ORDER BY created_at DESC",
        conn,
        params=(user_id,),
    )
    conn.close()
    return df


# ============================================================
# DSP / SIGNAL ENGINE
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
    total = cum_e[-1] if len(cum_e) else 0.0
    if total > 0:
        roll_idx = np.where(cum_e >= 0.85 * total)[0]
        rolloff = f_f[roll_idx[0]] if len(roll_idx) else 0.0
    else:
        rolloff = 0.0
    flatness = np.exp(np.mean(np.log(v_f + eps))) / (np.mean(v_f) + eps)
    return float(zcr), float(centroid), float(rolloff), float(flatness)


def generate_universal_signal(duration, sr, base_f, harm_r, imp_r, noise_l, normalize=True):
    duration = float(duration)
    sr = int(sr)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.sin(2 * np.pi * base_f * t) if base_f > 0 else np.zeros_like(t)

    if harm_r > 0 and base_f > 0:
        for h in range(2, 6):
            sig += (harm_r / h) * np.sin(2 * np.pi * (base_f * h) * t)

    if imp_r > 0:
        for i_t in np.arange(0, duration, 1.0 / max(imp_r, 1e-6)):
            idx = int((i_t + np.random.uniform(-0.02, 0.02)) * sr)
            if 0 <= idx < len(t):
                decay = np.exp(-25 * (t[idx:] - i_t))
                sig[idx:] += 2.5 * np.random.normal(0, 1, len(decay)) * decay

    sig += np.random.normal(0, noise_l, len(t))

    max_val = np.max(np.abs(sig)) if len(sig) else 0.0
    if normalize and max_val > 0:
        sig = sig / max_val

    f, v = calculate_fft(sig, sr)

    return {"t": t, "sig": sig, "fft_f": f, "fft_v": v, "sr": sr}


def extract_signal_features(sig, sr, label=None):
    sig = np.asarray(sig, dtype=float)
    sig = sig - np.mean(sig)
    f_f, v_f = calculate_fft(sig, sr)
    rms = float(np.sqrt(np.mean(sig ** 2))) if len(sig) else 0.0
    zcr, centroid, rolloff, flatness = get_audio_features(sig, f_f, v_f)
    out = {
        "RMS": rms,
        "Std": float(np.std(sig)) if len(sig) else 0.0,
        "Kurtosis": float(kurtosis(sig)) if len(sig) > 3 else 0.0,
        "CrestFactor": float(np.max(np.abs(sig)) / max(rms, 1e-6)) if len(sig) else 0.0,
        "ZCR": zcr,
        "SpectralCentroid": centroid,
        "SpectralRolloff": rolloff,
        "SpectralFlatness": flatness,
    }
    if label is not None:
        out = {"Label": label, **out}
    return out


def extract_features_from_bytes(file_bytes, filename, sr=16000):
    try:
        lower = filename.lower()
        if lower.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
            if df.shape[1] < 2:
                return {"error": "CSV must contain at least two columns."}
            sig = df.iloc[:, 1].astype(float).values
        elif lower.endswith(".wav"):
            sr, wav_data = wavfile.read(io.BytesIO(file_bytes))
            sig = (wav_data.mean(axis=1) if len(wav_data.shape) > 1 else wav_data).astype(float)
        else:
            return {"error": "Unsupported file type. Use CSV or WAV."}
        return extract_signal_features(sig, sr)
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# INDUSTRY PACKS / AGING
# ============================================================

INDUSTRY_PACKS = {
    "Rotating Machinery Pack": {
        "description": "Motoren, pompen, ventilatoren, compressoren, generatoren en lagers.",
        "sample_rate": 4000,
        "classes": {
            "Normal": {"base_f": 50, "harm_r": 0.05, "imp_r": 0, "noise_l": 0.08},
            "Unbalance": {"base_f": 50, "harm_r": 0.30, "imp_r": 0, "noise_l": 0.12},
            "Misalignment": {"base_f": 50, "harm_r": 0.55, "imp_r": 1, "noise_l": 0.14},
            "Bearing_Wear": {"base_f": 50, "harm_r": 0.35, "imp_r": 14, "noise_l": 0.18},
            "Critical_Failure": {"base_f": 52, "harm_r": 1.35, "imp_r": 35, "noise_l": 0.34},
        },
    },
    "Security & Tamper Pack": {
        "description": "Bouwplaatsen, containers, gevels, machines en remote assets.",
        "sample_rate": 16000,
        "classes": {
            "Normal_Background": {"base_f": 0, "harm_r": 0, "imp_r": 0, "noise_l": 0.08},
            "Impact_Tamper": {"base_f": 0, "harm_r": 0, "imp_r": 25, "noise_l": 0.20},
            "Grinding": {"base_f": 180, "harm_r": 1.50, "imp_r": 0, "noise_l": 0.32},
            "Drilling": {"base_f": 130, "harm_r": 1.10, "imp_r": 8, "noise_l": 0.28},
            "Vehicle_Nearby": {"base_f": 45, "harm_r": 0.60, "imp_r": 0, "noise_l": 0.22},
        },
    },
    "Forestry & Remote Asset Pack": {
        "description": "Bosbouw, landbouw, afgelegen assets en remote site monitoring.",
        "sample_rate": 16000,
        "classes": {
            "Forest_Normal": {"base_f": 0, "harm_r": 0, "imp_r": 0, "noise_l": 0.08},
            "Chainsaw": {"base_f": 85, "harm_r": 1.20, "imp_r": 0, "noise_l": 0.28},
            "Offroad_Vehicle": {"base_f": 42, "harm_r": 0.70, "imp_r": 1, "noise_l": 0.24},
            "Human_Movement": {"base_f": 70, "harm_r": 0.25, "imp_r": 5, "noise_l": 0.14},
            "Rain_And_Wind": {"base_f": 12, "harm_r": 0.12, "imp_r": 22, "noise_l": 0.26},
        },
    },
}


def get_industry_packs():
    return list(INDUSTRY_PACKS.keys())


def get_industry_pack(pack_name):
    return INDUSTRY_PACKS.get(pack_name)


def generate_industry_pack_dataset(pack_name, samples_per_class=50):
    pack = get_industry_pack(pack_name)
    if not pack:
        return pd.DataFrame(), {"error": "Unknown pack"}
    rows = []
    sr = pack["sample_rate"]
    for label, params in pack["classes"].items():
        for _ in range(int(samples_per_class)):
            p = {
                "base_f": max(0, params["base_f"] + np.random.normal(0, max(params["base_f"] * 0.02, 0.5))),
                "harm_r": max(0, params["harm_r"] + np.random.normal(0, 0.05)),
                "imp_r": max(0, params["imp_r"] + np.random.normal(0, max(params["imp_r"] * 0.04, 0.2))),
                "noise_l": max(0.001, params["noise_l"] * np.random.uniform(0.85, 1.20)),
            }
            d = generate_universal_signal(2.0, sr, p["base_f"], p["harm_r"], p["imp_r"], p["noise_l"])
            rows.append(extract_signal_features(d["sig"], sr, label))
    df = pd.DataFrame(rows)
    manifest = {
        "pack_name": pack_name,
        "sample_rate": sr,
        "samples_per_class": samples_per_class,
        "classes": list(pack["classes"].keys()),
    }
    return df, manifest


def apply_aging_profile(base_params, aging_percent):
    aging = np.clip(float(aging_percent), 0, 100) / 100.0
    return {
        "base_f": float(base_params.get("base_f", 50)) * (1 + 0.03 * aging),
        "harm_r": float(base_params.get("harm_r", 0.05)) * (1 + 2.8 * aging),
        "imp_r": float(base_params.get("imp_r", 0)) + (aging ** 1.6) * 35,
        "noise_l": float(base_params.get("noise_l", 0.05)) * (1 + 2.2 * aging),
    }


# ============================================================
# FUSION / DEMO ENGINE
# ============================================================

FUSION_TEMPLATES = {
    "Smart Forestry Threat": {
        "description": "Kettingzaag, voertuig, menselijke activiteit en remote-zone risico.",
        "mode": "threat",
        "weights": {"audio": 0.35, "vibration": 0.15, "gas": 0.05, "radar": 0.25, "gps": 0.20},
        "defaults": {"audio": 85, "vibration": 30, "gas": 10, "radar": 75, "gps": 80},
        "classes": ["Normal", "Human_Activity", "Vehicle", "Chainsaw", "Critical_Threat"],
    },
    "Construction Site Tamper": {
        "description": "Grinding, cutting, impact, handling en voertuigactiviteit.",
        "mode": "threat",
        "weights": {"audio": 0.30, "vibration": 0.30, "gas": 0.05, "radar": 0.25, "gps": 0.10},
        "defaults": {"audio": 75, "vibration": 80, "gas": 5, "radar": 70, "gps": 40},
        "classes": ["Normal", "Handling", "Impact", "Tool_Use", "Critical_Tamper"],
    },
    "Predictive Maintenance Fusion": {
        "description": "Machine health analyse met audio, vibratie, gas/temperatuur en asset context.",
        "mode": "health",
        "weights": {"audio": 0.25, "vibration": 0.45, "gas": 0.20, "radar": 0.05, "gps": 0.05},
        "defaults": {"audio": 35, "vibration": 85, "gas": 40, "radar": 0, "gps": 5},
        "classes": ["Healthy", "Early_Wear", "Wear", "Failure_Risk", "Critical_Failure"],
    },
    "Remote Asset Security": {
        "description": "Containers, machines, landbouwassets en afgelegen infrastructuur.",
        "mode": "threat",
        "weights": {"audio": 0.25, "vibration": 0.25, "gas": 0.10, "radar": 0.30, "gps": 0.10},
        "defaults": {"audio": 55, "vibration": 65, "gas": 10, "radar": 80, "gps": 50},
        "classes": ["Normal", "Approach", "Handling", "Tamper", "Critical_Theft_Risk"],
    },
}

DEMO_PROJECTS = {
    "Smart Forestry Demo": {
        "title": "Smart Forestry Demo",
        "problem": "Remote forests are hard to monitor and illegal activity can happen before people notice.",
        "solution": "Create an Edge AI-ready sensor fusion dataset for chainsaw, vehicle and human activity detection.",
        "output": "Fusion dataset, training features, PDF report, dataset doctor advice and hardware recommendation.",
        "template": "Smart Forestry Threat",
        "samples": 500,
        "industry_pack": "Forestry & Remote Asset Pack",
        "hardware_target": "low_power",
        "cta": "Use this demo to pitch forestry, agriculture, remote asset security and anti-theft monitoring.",
    },
    "Construction Security Demo": {
        "title": "Construction Security Demo",
        "problem": "Construction sites suffer from tool theft, tampering and after-hours intrusion.",
        "solution": "Generate a multi-sensor tamper dataset with audio, vibration, radar and location risk.",
        "output": "Tamper/security training data, audit report and ESP32/STM32 deployment advice.",
        "template": "Construction Site Tamper",
        "samples": 500,
        "industry_pack": "Security & Tamper Pack",
        "hardware_target": "balanced",
        "cta": "Use this demo for site security, facade protection, containers and high-value equipment.",
    },
    "Predictive Maintenance Demo": {
        "title": "Predictive Maintenance Demo",
        "problem": "Machine failure is expensive, but real failure data is often rare or unavailable.",
        "solution": "Create synthetic machine health data with vibration, audio and environmental indicators.",
        "output": "Health/failure-risk dataset, training features, readiness score and hardware recommendation.",
        "template": "Predictive Maintenance Fusion",
        "samples": 500,
        "industry_pack": "Rotating Machinery Pack",
        "hardware_target": "performance",
        "cta": "Use this demo for pumps, motors, bearings, compressors and industrial predictive maintenance.",
    },
    "Remote Asset Demo": {
        "title": "Remote Asset Demo",
        "problem": "Remote machines and assets are vulnerable because nobody is nearby to verify events.",
        "solution": "Build a sensor fusion dataset for approach, handling, tamper and theft-risk detection.",
        "output": "Remote security fusion data, report bundle and low-power gateway/node advice.",
        "template": "Remote Asset Security",
        "samples": 500,
        "industry_pack": "Security & Tamper Pack",
        "hardware_target": "low_power",
        "cta": "Use this demo for farms, containers, trailers, machinery yards and remote infrastructure.",
    },
}


def get_demo_projects():
    return list(DEMO_PROJECTS.keys())


def get_demo_project(name):
    return DEMO_PROJECTS.get(name)


def get_fusion_templates():
    return list(FUSION_TEMPLATES.keys())


def get_fusion_template(template_name):
    return FUSION_TEMPLATES.get(template_name)


def calculate_fusion_score(audio_score, vibration_score, gas_score, radar_score, gps_score, template_name="Smart Forestry Threat"):
    template = get_fusion_template(template_name) or FUSION_TEMPLATES["Smart Forestry Threat"]
    weights = template["weights"]
    values = {
        "audio": float(np.clip(audio_score, 0, 100)),
        "vibration": float(np.clip(vibration_score, 0, 100)),
        "gas": float(np.clip(gas_score, 0, 100)),
        "radar": float(np.clip(radar_score, 0, 100)),
        "gps": float(np.clip(gps_score, 0, 100)),
    }
    fusion = float(np.clip(sum(values[k] * weights.get(k, 0.0) for k in values), 0, 100))
    health = float(np.clip(100 - fusion, 0, 100))
    confidence = float(np.clip(100 - np.std(list(values.values())) * 0.75, 0, 100))
    if fusion < 20:
        level = "LOW"
    elif fusion < 45:
        level = "ELEVATED"
    elif fusion < 70:
        level = "HIGH"
    else:
        level = "CRITICAL"
    if template.get("mode") == "health":
        if health > 80:
            event, action = "Healthy", "Continue monitoring."
        elif health > 60:
            event, action = "Early wear", "Schedule inspection."
        elif health > 35:
            event, action = "Failure risk", "Plan maintenance soon."
        else:
            event, action = "Critical failure risk", "Immediate maintenance recommended."
    else:
        if fusion < 20:
            event, action = "Normal", "No action required."
        elif fusion < 45:
            event, action = "Suspicious activity", "Increase monitoring."
        elif fusion < 70:
            event, action = "Likely event", "Send alert / verify situation."
        else:
            event, action = "Critical threat", "Immediate response recommended."
    return {
        "fusion_score": fusion,
        "health_score": health,
        "confidence": confidence,
        "level": level,
        "event": event,
        "recommended_action": action,
        "sensor_values": values,
        "weights": weights,
    }


def fusion_score_target_for_label(label, classes):
    if label not in classes:
        return 50.0
    idx = classes.index(label)
    low = (idx / len(classes)) * 100
    high = ((idx + 1) / len(classes)) * 100
    return float((low + high) / 2.0)


def fusion_label_from_score(score, template_name):
    template = get_fusion_template(template_name) or FUSION_TEMPLATES["Smart Forestry Threat"]
    classes = template.get("classes", ["Normal", "Elevated", "High", "Critical"])
    idx = int(np.floor((float(np.clip(score, 0, 100)) / 100.0) * len(classes)))
    return classes[min(max(idx, 0), len(classes) - 1)]


def generate_balanced_sensor_values_for_target(template_name, target_score):
    template = get_fusion_template(template_name) or FUSION_TEMPLATES["Smart Forestry Threat"]
    weights = template["weights"]
    base = float(np.clip(target_score, 0, 100))
    return (
        np.clip(base + np.random.normal(0, 12) + weights.get("audio", 0.2) * 15, 0, 100),
        np.clip(base + np.random.normal(0, 12) + weights.get("vibration", 0.2) * 15, 0, 100),
        np.clip(base + np.random.normal(0, 10) + weights.get("gas", 0.2) * 10, 0, 100),
        np.clip(base + np.random.normal(0, 13) + weights.get("radar", 0.2) * 15, 0, 100),
        np.clip(base + np.random.normal(0, 10) + weights.get("gps", 0.2) * 10, 0, 100),
    )


def generate_sensor_fusion_sample(template_name, audio_score, vibration_score, gas_score, radar_score, gps_score, jitter=True):
    if jitter:
        audio_score = np.clip(audio_score + np.random.normal(0, 5), 0, 100)
        vibration_score = np.clip(vibration_score + np.random.normal(0, 5), 0, 100)
        gas_score = np.clip(gas_score + np.random.normal(0, 4), 0, 100)
        radar_score = np.clip(radar_score + np.random.normal(0, 6), 0, 100)
        gps_score = np.clip(gps_score + np.random.normal(0, 4), 0, 100)
    result = calculate_fusion_score(audio_score, vibration_score, gas_score, radar_score, gps_score, template_name)
    label = fusion_label_from_score(result["fusion_score"], template_name)
    return {
        "Label": label,
        "Template": template_name,
        "AudioScore": float(audio_score),
        "VibrationScore": float(vibration_score),
        "GasScore": float(gas_score),
        "RadarScore": float(radar_score),
        "GPSZoneScore": float(gps_score),
        "FusionScore": float(result["fusion_score"]),
        "HealthScore": float(result["health_score"]),
        "Confidence": float(result["confidence"]),
        "Level": result["level"],
        "Event": result["event"],
        "RecommendedAction": result["recommended_action"],
    }


def get_fusion_training_columns(template_name=None):
    template = get_fusion_template(template_name) if template_name else None
    mode = template.get("mode", "threat") if template else "threat"
    cols = ["Label", "AudioScore", "VibrationScore", "GasScore", "RadarScore", "GPSZoneScore", "Confidence"]
    cols.append("HealthScore" if mode == "health" else "FusionScore")
    return cols


def create_fusion_training_dataframe(fusion_df, template_name=None):
    cols = get_fusion_training_columns(template_name)
    return fusion_df[[c for c in cols if c in fusion_df.columns]].copy()


def generate_sensor_fusion_dataset(
    template_name,
    samples=500,
    base_audio=50,
    base_vibration=50,
    base_gas=20,
    base_radar=50,
    base_gps=50,
    include_scenario_variants=True,
    balanced_classes=True,
):
    template = get_fusion_template(template_name) or FUSION_TEMPLATES["Smart Forestry Threat"]
    classes = template.get("classes", ["Normal", "Elevated", "High", "Critical"])
    defaults = template.get("defaults", {})
    base_audio = defaults.get("audio", base_audio)
    base_vibration = defaults.get("vibration", base_vibration)
    base_gas = defaults.get("gas", base_gas)
    base_radar = defaults.get("radar", base_radar)
    base_gps = defaults.get("gps", base_gps)
    rows = []
    samples = int(samples)
    if balanced_classes and classes:
        per_class = max(1, samples // len(classes))
        plan = []
        for label in classes:
            plan.extend([label] * per_class)
        while len(plan) < samples:
            plan.append(np.random.choice(classes))
        np.random.shuffle(plan)
        for label in plan[:samples]:
            target = fusion_score_target_for_label(label, classes)
            vals = generate_balanced_sensor_values_for_target(template_name, target)
            row = generate_sensor_fusion_sample(template_name, *vals, jitter=True)
            row["Label"] = label
            rows.append(row)
    else:
        for _ in range(samples):
            multiplier = 1.0
            if include_scenario_variants:
                multiplier = np.random.choice([
                    np.random.uniform(0.1, 0.35),
                    np.random.uniform(0.35, 0.65),
                    np.random.uniform(0.65, 0.9),
                    np.random.uniform(0.9, 1.15),
                ])
            rows.append(generate_sensor_fusion_sample(
                template_name,
                base_audio * multiplier,
                base_vibration * multiplier,
                base_gas * multiplier,
                base_radar * multiplier,
                base_gps * multiplier,
                jitter=True,
            ))
    df = pd.DataFrame(rows)
    manifest = {
        "engine": "EdgeTwin Studio V19 - Powered by OMEGA-X Engine",
        "template": template_name,
        "description": template.get("description", ""),
        "samples": samples,
        "balanced_classes": bool(balanced_classes),
        "sensors": ["audio", "vibration", "gas", "radar", "gps"],
        "columns": list(df.columns),
        "recommended_training_features": get_fusion_training_columns(template_name),
    }
    return df, manifest


def run_demo_project(demo_name):
    demo = get_demo_project(demo_name)
    if not demo:
        raise ValueError("Unknown demo project")
    fusion_df, manifest = generate_sensor_fusion_dataset(demo["template"], samples=demo["samples"], balanced_classes=True)
    doctor = fusion_dataset_doctor(fusion_df, demo["template"])
    training_df = create_fusion_training_dataframe(fusion_df, demo["template"])
    sr = 16000 if ("Forestry" in demo["template"] or "Security" in demo["template"] or "Tamper" in demo["template"]) else 4000
    hw = hardware_auto_architect(max(1, len(training_df.columns) - 1), sr, demo["hardware_target"])
    reliability = calculate_reliability_score(doctor, has_real_data=False, selected_sensors=["audio", "vibration", "radar", "gps"])
    commercial_summary = {
        "demo_name": demo_name,
        "title": demo["title"],
        "problem": demo["problem"],
        "solution": demo["solution"],
        "output": demo["output"],
        "cta": demo["cta"],
        "template": demo["template"],
        "recommended_board": hw["recommendation"],
        "overall_score": doctor["overall_score"],
        "reliability_score": reliability["reliability_score"],
    }
    return {
        "demo": demo,
        "fusion_df": fusion_df,
        "manifest": manifest,
        "doctor": doctor,
        "training_df": training_df,
        "hardware": hw,
        "reliability": reliability,
        "commercial_summary": commercial_summary,
    }


# ============================================================
# V19 USE CASE WIZARD / AUTO PILOT GENERATOR
# ============================================================

USE_CASE_TYPES = {
    "Predictive Maintenance": {
        "template": "Predictive Maintenance Fusion",
        "recommended_sensors": ["Audio", "Vibration", "Gas / Environment"],
        "default_classes": ["Healthy", "Early_Wear", "Wear", "Failure_Risk", "Critical_Failure"],
        "environment": "Industrial",
        "sample_rate": 4000,
        "priority": "performance",
        "problem_hint": "Detect machine wear or failure risk before costly breakdowns happen.",
    },
    "Security / Tamper": {
        "template": "Construction Site Tamper",
        "recommended_sensors": ["Audio", "Vibration", "Radar", "GPS / Zone"],
        "default_classes": ["Normal", "Handling", "Impact", "Tool_Use", "Critical_Tamper"],
        "environment": "Construction site",
        "sample_rate": 16000,
        "priority": "balanced",
        "problem_hint": "Detect theft, tool use, tamper, impact or after-hours intrusion.",
    },
    "Smart Forestry / Remote Area": {
        "template": "Smart Forestry Threat",
        "recommended_sensors": ["Audio", "Radar", "GPS / Zone", "Vibration"],
        "default_classes": ["Normal", "Human_Activity", "Vehicle", "Chainsaw", "Critical_Threat"],
        "environment": "Forest / remote outdoor",
        "sample_rate": 16000,
        "priority": "low_power",
        "problem_hint": "Detect chainsaw, vehicle, human activity and remote-zone risk.",
    },
    "Remote Asset Monitoring": {
        "template": "Remote Asset Security",
        "recommended_sensors": ["Radar", "Vibration", "Audio", "GPS / Zone"],
        "default_classes": ["Normal", "Approach", "Handling", "Tamper", "Critical_Theft_Risk"],
        "environment": "Remote asset",
        "sample_rate": 16000,
        "priority": "low_power",
        "problem_hint": "Protect containers, trailers, farm assets, remote machines or infrastructure.",
    },
    "Environmental Risk": {
        "template": "Smart Forestry Threat",
        "recommended_sensors": ["Gas / Environment", "Audio", "Radar", "GPS / Zone"],
        "default_classes": ["Calm", "Weather", "Movement", "Machinery", "Environmental_Risk"],
        "environment": "Outdoor",
        "sample_rate": 16000,
        "priority": "balanced",
        "problem_hint": "Detect weather, environmental anomalies, movement and machinery activity.",
    },
    "Custom Sensor Fusion": {
        "template": "Remote Asset Security",
        "recommended_sensors": ["Audio", "Vibration", "Radar"],
        "default_classes": ["Normal", "Warning", "Event", "Critical"],
        "environment": "Custom",
        "sample_rate": 16000,
        "priority": "balanced",
        "problem_hint": "Build a custom sensor fusion pilot for your own detection problem.",
    },
}

SENSOR_OPTIONS = ["Audio", "Vibration", "Radar", "Gas / Environment", "GPS / Zone", "Temperature", "Humidity", "IMU / Movement"]
ENVIRONMENT_OPTIONS = ["Indoor", "Outdoor", "Industrial", "Construction site", "Forest / remote outdoor", "Remote asset", "Vehicle / mobile asset", "Agriculture", "Custom"]
OUTPUT_LEVELS = ["Dataset only", "Professional Pilot Bundle", "Enterprise Deployment Bundle"]


def get_use_case_types():
    return list(USE_CASE_TYPES.keys())


def get_use_case_defaults(use_case_type):
    return USE_CASE_TYPES.get(use_case_type, USE_CASE_TYPES["Custom Sensor Fusion"])


def get_sensor_options():
    return SENSOR_OPTIONS


def get_environment_options():
    return ENVIRONMENT_OPTIONS


def get_output_levels():
    return OUTPUT_LEVELS


def parse_labels_text(labels_text, fallback=None):
    fallback = fallback or ["Normal", "Warning", "Critical"]
    if not labels_text:
        return fallback
    raw = labels_text.replace(";", ",").replace("\n", ",").split(",")
    labels = []
    for item in raw:
        clean = item.strip().replace(" ", "_")
        if clean and clean not in labels:
            labels.append(clean)
    return labels if len(labels) >= 2 else fallback


def build_use_case_config(
    use_case_type,
    project_goal,
    selected_sensors,
    environment,
    labels_text,
    samples,
    has_real_data=False,
    output_level="Professional Pilot Bundle",
    priority="balanced",
):
    defaults = get_use_case_defaults(use_case_type)
    selected_sensors = selected_sensors or defaults["recommended_sensors"]
    labels = parse_labels_text(labels_text, defaults["default_classes"])
    return {
        "use_case_type": use_case_type,
        "project_goal": project_goal or defaults["problem_hint"],
        "selected_sensors": selected_sensors,
        "environment": environment or defaults["environment"],
        "classes": labels,
        "samples": int(samples),
        "has_real_data": bool(has_real_data),
        "output_level": output_level,
        "priority": priority or defaults["priority"],
        "template": defaults["template"],
        "sample_rate": defaults["sample_rate"],
        "created_at": str(datetime.datetime.now()),
    }


def sensor_weights_from_selection(selected_sensors, use_case_type="Custom Sensor Fusion"):
    defaults = get_use_case_defaults(use_case_type)
    template = get_fusion_template(defaults["template"]) or FUSION_TEMPLATES["Remote Asset Security"]
    base = template.get("weights", {"audio": 0.25, "vibration": 0.25, "gas": 0.10, "radar": 0.30, "gps": 0.10}).copy()
    map_names = {
        "Audio": "audio",
        "Vibration": "vibration",
        "Radar": "radar",
        "Gas / Environment": "gas",
        "GPS / Zone": "gps",
        "Temperature": "gas",
        "Humidity": "gas",
        "IMU / Movement": "vibration",
    }
    active = {map_names[s] for s in selected_sensors if s in map_names}
    weights = {}
    for key in ["audio", "vibration", "gas", "radar", "gps"]:
        weights[key] = base.get(key, 0.0) if key in active else 0.0
    total = sum(weights.values())
    if total <= 0:
        return base
    return {k: v / total for k, v in weights.items()}


def generate_custom_fusion_dataset(config):
    classes = config.get("classes", ["Normal", "Warning", "Critical"])
    samples = int(config.get("samples", 500))
    weights = sensor_weights_from_selection(config.get("selected_sensors", []), config.get("use_case_type", "Custom Sensor Fusion"))
    template_name = config.get("template", "Remote Asset Security")
    mode = (get_fusion_template(template_name) or {}).get("mode", "threat")
    rows = []
    per_class = max(1, samples // len(classes))
    plan = []
    for label in classes:
        plan.extend([label] * per_class)
    while len(plan) < samples:
        plan.append(np.random.choice(classes))
    np.random.shuffle(plan)
    for label in plan[:samples]:
        target = fusion_score_target_for_label(label, classes)
        base = float(np.clip(target, 0, 100))
        audio = np.clip(base + np.random.normal(0, 14) + weights.get("audio", 0.0) * 18, 0, 100)
        vibration = np.clip(base + np.random.normal(0, 14) + weights.get("vibration", 0.0) * 18, 0, 100)
        gas = np.clip(base + np.random.normal(0, 12) + weights.get("gas", 0.0) * 15, 0, 100)
        radar = np.clip(base + np.random.normal(0, 15) + weights.get("radar", 0.0) * 18, 0, 100)
        gps = np.clip(base + np.random.normal(0, 12) + weights.get("gps", 0.0) * 15, 0, 100)
        values = {"audio": audio, "vibration": vibration, "gas": gas, "radar": radar, "gps": gps}
        fusion = float(np.clip(sum(values[k] * weights.get(k, 0.0) for k in values), 0, 100))
        health = float(np.clip(100 - fusion, 0, 100))
        confidence = float(np.clip(100 - np.std(list(values.values())) * 0.70, 0, 100))
        if fusion < 20:
            level = "LOW"
        elif fusion < 45:
            level = "ELEVATED"
        elif fusion < 70:
            level = "HIGH"
        else:
            level = "CRITICAL"
        if mode == "health":
            event = "Health state"
            action = "Use field validation data before production deployment."
        else:
            event = "Detected scenario"
            action = "Verify with real-world pilot data before production deployment."
        rows.append({
            "Label": label,
            "Template": template_name,
            "UseCaseType": config.get("use_case_type", "Custom"),
            "Environment": config.get("environment", "Custom"),
            "AudioScore": float(audio),
            "VibrationScore": float(vibration),
            "GasScore": float(gas),
            "RadarScore": float(radar),
            "GPSZoneScore": float(gps),
            "FusionScore": fusion,
            "HealthScore": health,
            "Confidence": confidence,
            "Level": level,
            "Event": event,
            "RecommendedAction": action,
        })
    df = pd.DataFrame(rows)
    manifest = {
        "engine": "EdgeTwin Studio V19 Auto Pilot Generator",
        "template": template_name,
        "use_case_type": config.get("use_case_type"),
        "project_goal": config.get("project_goal"),
        "environment": config.get("environment"),
        "selected_sensors": config.get("selected_sensors", []),
        "classes": classes,
        "samples": samples,
        "has_real_data": config.get("has_real_data", False),
        "output_level": config.get("output_level"),
        "priority": config.get("priority"),
        "weights": weights,
        "columns": list(df.columns),
        "recommended_training_features": get_fusion_training_columns(template_name),
    }
    return df, manifest


def calculate_reliability_score(doctor, has_real_data=False, selected_sensors=None):
    selected_sensors = selected_sensors or []
    overall = float(doctor.get("overall_score", 0))
    separation = float(doctor.get("separation_score", 0))
    balance = float(doctor.get("balance_score", 0))
    sensor_bonus = min(12, max(0, len(selected_sensors) - 1) * 3)
    real_bonus = 18 if has_real_data else 0
    synthetic_realism = int(np.clip((overall * 0.55) + (separation * 0.25) + sensor_bonus + 8, 0, 100))
    field_readiness = int(np.clip((overall * 0.45) + (balance * 0.15) + real_bonus + sensor_bonus, 0, 100))
    reliability = int(np.clip((synthetic_realism * 0.45) + (field_readiness * 0.55), 0, 100))
    if has_real_data:
        needed = 10 if field_readiness >= 80 else 20
    else:
        needed = 25 if field_readiness >= 70 else 50
    if reliability >= 82:
        risk = "Low-Medium"
        verdict = "Strong pilot package. Still validate with field data before production."
    elif reliability >= 65:
        risk = "Medium"
        verdict = "Good prototype package. Add real data to improve production confidence."
    else:
        risk = "High"
        verdict = "Useful for concepting, but more real data and optimization are recommended."
    return {
        "synthetic_realism_score": synthetic_realism,
        "field_readiness_score": field_readiness,
        "reliability_score": reliability,
        "dataset_risk": risk,
        "recommended_real_samples_needed": needed,
        "verdict": verdict,
    }


def run_auto_pilot_project(config):
    fusion_df, manifest = generate_custom_fusion_dataset(config)
    doctor = fusion_dataset_doctor(fusion_df, manifest.get("template"))
    training_df = create_fusion_training_dataframe(fusion_df, manifest.get("template"))
    if "UseCaseType" in fusion_df.columns:
        extra_cols = ["UseCaseType", "Environment"]
        keep_cols = [c for c in training_df.columns] + [c for c in extra_cols if c in fusion_df.columns]
        # Keep training dataframe clean for ML, but manifest/report retain context.
        training_df = training_df[[c for c in training_df.columns if c in training_df.columns]].copy()
    hw = hardware_auto_architect(max(1, len(training_df.columns) - 1), manifest.get("sample_rate", config.get("sample_rate", 16000)), config.get("priority", "balanced"))
    reliability = calculate_reliability_score(doctor, config.get("has_real_data", False), config.get("selected_sensors", []))
    commercial_summary = {
        "title": f"{config.get('use_case_type', 'Custom')} Auto Pilot",
        "problem": config.get("project_goal", "Customer needs an Edge AI sensor pilot."),
        "solution": "The wizard generated a sensor-fusion dataset, dataset audit, reliability estimate and hardware direction.",
        "output": "Auto Pilot Bundle with CSV datasets, PDF report, metadata, hardware advice and next steps.",
        "cta": "Next step: upload real sensor data or run a field pilot to increase production reliability.",
        "template": manifest.get("template"),
        "recommended_board": hw.get("recommendation", "Unknown"),
        "overall_score": doctor.get("overall_score", 0),
        "reliability_score": reliability.get("reliability_score", 0),
    }
    return {
        "config": config,
        "fusion_df": fusion_df,
        "manifest": manifest,
        "doctor": doctor,
        "training_df": training_df,
        "hardware": hw,
        "reliability": reliability,
        "commercial_summary": commercial_summary,
    }


# ============================================================
# HARDWARE ARCHITECT
# ============================================================

HARDWARE_PROFILES = {
    "ESP32-S3": {"cpu": "Xtensa LX7 dual-core", "role": "Edge AI Node", "power_class": "Medium", "fft_mult": 0.00008, "feat_mult": 0.20, "inf_mult": 1.50, "target_ram": 320, "gateway_fit": "No", "notes": "Audio TinyML, vibration classification, WiFi/BLE edge nodes."},
    "RAK4631 / nRF52840": {"cpu": "ARM Cortex-M4F", "role": "Ultra-low-power LoRa Node", "power_class": "Very Low", "fft_mult": 0.00015, "feat_mult": 0.40, "inf_mult": 3.00, "target_ram": 256, "gateway_fit": "LoRa node", "notes": "LoRaWAN and battery-powered remote sensing."},
    "STM32U5": {"cpu": "ARM Cortex-M33", "role": "Secure Industrial Low-power Node", "power_class": "Very Low", "fft_mult": 0.00009, "feat_mult": 0.22, "inf_mult": 1.80, "target_ram": 512, "gateway_fit": "No", "notes": "Secure industrial sensing and low-power TinyML."},
    "STM32H7": {"cpu": "ARM Cortex-M7", "role": "High-performance Industrial Node", "power_class": "Medium-High", "fft_mult": 0.000035, "feat_mult": 0.10, "inf_mult": 0.80, "target_ram": 1024, "gateway_fit": "Limited", "notes": "High-speed vibration, audio DSP and industrial inference."},
    "Raspberry Pi Zero 2 W": {"cpu": "Quad-core ARM Cortex-A53", "role": "Light Linux Gateway", "power_class": "Medium", "fft_mult": 0.000015, "feat_mult": 0.05, "inf_mult": 0.40, "target_ram": 512000, "gateway_fit": "Yes", "notes": "Light MQTT gateway and dashboard bridge."},
    "Raspberry Pi 5": {"cpu": "Quad-core ARM Cortex-A76", "role": "High-performance Linux Gateway", "power_class": "High", "fft_mult": 0.000003, "feat_mult": 0.01, "inf_mult": 0.08, "target_ram": 4096000, "gateway_fit": "Yes", "notes": "Fast gateway, local model training and dashboards."},
    "Generic Linux Gateway": {"cpu": "x86/ARM Linux", "role": "Industrial Gateway / Server", "power_class": "High", "fft_mult": 0.000003, "feat_mult": 0.01, "inf_mult": 0.05, "target_ram": 8192000, "gateway_fit": "Yes", "notes": "API server, database, dashboards and model orchestration."},
}


def get_available_hardware():
    return list(HARDWARE_PROFILES.keys())


def get_hardware_catalog():
    rows = []
    for name, p in HARDWARE_PROFILES.items():
        rows.append({"board": name, "cpu": p["cpu"], "role": p["role"], "power_class": p["power_class"], "target_ram_kb": p["target_ram"], "gateway_fit": p["gateway_fit"], "notes": p["notes"]})
    return pd.DataFrame(rows)


def estimate_edge_load(hw_name, feat_n, sr, duration=1.0):
    p = HARDWARE_PROFILES.get(hw_name, HARDWARE_PROFILES["RAK4631 / nRF52840"])
    fft_n = 1024 if sr <= 4000 else 2048
    ram = ((min(int(sr * duration), 8192) * 4) + (fft_n * 8) + 2048) / 1024
    fft_ms = (fft_n * np.log2(fft_n)) * p["fft_mult"]
    feat_ms = feat_n * p["feat_mult"]
    inf_ms = p["inf_mult"]
    return float(ram), float(fft_ms), float(feat_ms), float(inf_ms)


def calculate_deployment_score(hw_name, latency, ram_kb):
    p = HARDWARE_PROFILES.get(hw_name, HARDWARE_PROFILES["RAK4631 / nRF52840"])
    lat_score = max(0, 100 - (latency / 20.0) * 50)
    ram_score = max(0, 100 - (ram_kb / max(p["target_ram"] / 2, 1)) * 50)
    score = lat_score * 0.7 + ram_score * 0.3
    if p["fft_mult"] < 0.00001:
        score += 15
    elif p["fft_mult"] < 0.00005:
        score += 12
    elif p["fft_mult"] < 0.00010:
        score += 10
    elif p["fft_mult"] < 0.00020:
        score += 5
    if p["power_class"] == "Very Low":
        score += 3
    return float(np.clip(score, 0, 100))


def hardware_auto_architect(num_features, sr, target="balanced", selected_boards=None):
    boards = selected_boards or get_available_hardware()
    rows = []
    for board in boards:
        if board not in HARDWARE_PROFILES:
            continue
        p = HARDWARE_PROFILES[board]
        ram, fft_ms, feat_ms, inf_ms = estimate_edge_load(board, num_features, sr)
        latency = fft_ms + feat_ms + inf_ms
        score = calculate_deployment_score(board, latency, ram)
        adjusted = score
        if target == "low_power":
            adjusted += {"Very Low": 12, "Low": 8, "Medium": 3}.get(p["power_class"], 0) - latency * 0.20
        elif target == "performance":
            adjusted -= latency * 1.0
        elif target == "gateway":
            adjusted += 20 if p["gateway_fit"] == "Yes" else 0
            adjusted -= latency * 0.35
        else:
            adjusted -= latency * 0.50
        rows.append({"board": board, "cpu": p["cpu"], "role": p["role"], "power_class": p["power_class"], "score": score, "adjusted_score": adjusted, "latency_ms": latency, "ram_kb": ram, "gateway_fit": p["gateway_fit"], "notes": p["notes"]})
    rows = sorted(rows, key=lambda r: r["adjusted_score"], reverse=True)
    best = rows[0] if rows else {"board": "Unknown"}
    return {"recommendation": best["board"], "ranking": rows, "reason": f"Best fit for {target}: {best['board']} with {best.get('latency_ms', 0):.1f} ms estimated latency."}


# ============================================================
# DATASET DOCTOR / EXPORTS
# ============================================================

def calculate_audit_scores(X_df, y_series):
    if X_df is None or len(X_df) == 0 or y_series is None or len(y_series) == 0:
        return 0, 0, 0
    try:
        X_scaled = StandardScaler().fit_transform(X_df)
    except Exception:
        return 0, 0, 0
    label_counts = y_series.value_counts()
    div = min(100, int((np.mean(pdist(X_scaled)) / 4.0) * 100)) if len(X_scaled) > 1 else 0
    bal = 100 if len(label_counts) >= 2 and label_counts.max() > 0 and (label_counts.min() / label_counts.max()) > 0.5 else 50
    try:
        sep = int((silhouette_score(X_scaled, y_series) + 1) * 50) if len(y_series.unique()) >= 2 and label_counts.min() >= 2 else 0
    except Exception:
        sep = 0
    return div, bal, sep


def fusion_dataset_doctor(fusion_df, template_name=None):
    if fusion_df is None or len(fusion_df) == 0 or "Label" not in fusion_df.columns:
        return {"overall_score": 0, "advice": [{"severity": "high", "message": "Geen geldige fusion dataset aanwezig."}]}
    df = fusion_df.copy()
    label_counts = df["Label"].value_counts()
    advice, severity = [], []
    if len(label_counts) < 2:
        advice.append("Dataset heeft maar een klasse. Genereer balanced classes.")
        severity.append("high")
    elif label_counts.min() / max(label_counts.max(), 1) < 0.65:
        advice.append("Class balance is zwak. Gebruik balanced generation of verhoog sample count.")
        severity.append("high")
    else:
        advice.append("Class balance ziet er goed uit.")
        severity.append("info")
    sensor_cols = ["AudioScore", "VibrationScore", "GasScore", "RadarScore", "GPSZoneScore"]
    missing = [c for c in sensor_cols if c not in df.columns]
    if missing:
        advice.append(f"Ontbrekende sensor-kolommen: {', '.join(missing)}.")
        severity.append("high")
    else:
        low_var = df[sensor_cols].std(numeric_only=True)
        low_var = low_var[low_var < 3].index.tolist()
        if low_var:
            advice.append(f"Lage sensor-variatie bij: {', '.join(low_var)}.")
            severity.append("medium")
        else:
            advice.append("Sensorvariatie ziet er gezond uit.")
            severity.append("info")
    training_df = create_fusion_training_dataframe(df, template_name)
    numeric = [c for c in training_df.columns if c != "Label"]
    div, bal, sep = calculate_audit_scores(training_df[numeric], training_df["Label"]) if numeric else (0, 0, 0)
    if sep < 50 and len(label_counts) >= 2:
        advice.append("Label separation is laag. Maak scenario's sterker verschillend.")
        severity.append("high")
    elif sep < 75 and len(label_counts) >= 2:
        advice.append("Label separation is middelmatig. Dit is bruikbaar voor demo, maar optimalisatie kan beter.")
        severity.append("medium")
    else:
        advice.append("Label separation is goed bruikbaar.")
        severity.append("info")
    score = int(div * 0.30 + bal * 0.30 + sep * 0.40)
    return {
        "diversity_score": div,
        "balance_score": bal,
        "separation_score": sep,
        "overall_score": score,
        "label_counts": label_counts.to_dict(),
        "recommended_training_features": get_fusion_training_columns(template_name),
        "advice": [{"severity": s, "message": a} for s, a in zip(severity, advice)],
    }


def dataset_doctor(X_df, y_series):
    div, bal, sep = calculate_audit_scores(X_df, y_series)
    advice, severity = [], []
    if len(X_df) < 100:
        advice.append("Dataset is nog klein. Richt op 200+ samples voor een sterkere pilot.")
        severity.append("medium")
    counts = y_series.value_counts()
    if len(counts) < 2:
        advice.append("Er is maar een label aanwezig.")
        severity.append("high")
    elif counts.min() / max(counts.max(), 1) < 0.5:
        advice.append("Class balance is zwak. Voeg samples toe aan de kleinste klasse.")
        severity.append("high")
    if sep < 50:
        advice.append("Label separation is laag. Klassen lijken te veel op elkaar.")
        severity.append("high")
    elif sep < 75:
        advice.append("Label separation is middelmatig. Extra onderscheidende features kunnen helpen.")
        severity.append("medium")
    if not advice:
        advice.append("Dataset ziet er gezond uit voor een eerste ML-training.")
        severity.append("info")
    return {
        "diversity_score": div,
        "balance_score": bal,
        "separation_score": sep,
        "overall_score": int(div * 0.35 + bal * 0.30 + sep * 0.35),
        "advice": [{"severity": s, "message": a} for s, a in zip(severity, advice)],
    }


def clean_pdf_text(value):
    text = str(value)
    replacements = {"—": "-", "–": "-", "•": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "→": "->", "✅": "", "⚠️": "", "❌": "", "🧩": "", "🩺": "", "_": " "}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def safe_pdf_output(pdf):
    out = pdf.output(dest="S")
    if isinstance(out, bytes):
        return out
    if isinstance(out, bytearray):
        return bytes(out)
    return str(out).encode("latin1", errors="ignore")


def safe_pdf_cell(pdf, text, height=7, bold=False):
    pdf.set_x(10)
    pdf.set_font("Arial", "B" if bold else "", 10)
    pdf.cell(190, height, txt=clean_pdf_text(text)[:120], ln=True)


def safe_pdf_multicell(pdf, text, height=6):
    pdf.set_x(10)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(190, height, txt=clean_pdf_text(text))


def generate_fusion_pdf_report(project_name, fusion_df, manifest, doctor, commercial_summary=None, reliability=None, hardware_result=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Studio Pilot Report", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("Powered by OMEGA-X Synthetic Sensor Engine"), ln=True, align="C")
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)
    if commercial_summary:
        safe_pdf_cell(pdf, "Executive Summary", 8, True)
        safe_pdf_multicell(pdf, f"Problem: {commercial_summary.get('problem', '')}")
        safe_pdf_multicell(pdf, f"Solution: {commercial_summary.get('solution', '')}")
        safe_pdf_multicell(pdf, f"Output: {commercial_summary.get('output', '')}")
        pdf.ln(4)
    safe_pdf_cell(pdf, "Dataset Readiness", 8, True)
    safe_pdf_cell(pdf, f"Template: {manifest.get('template', 'Unknown')}")
    safe_pdf_cell(pdf, f"Samples: {manifest.get('samples', len(fusion_df))}")
    safe_pdf_cell(pdf, f"Diversity: {doctor.get('diversity_score', 0)}%")
    safe_pdf_cell(pdf, f"Balance: {doctor.get('balance_score', 0)}%")
    safe_pdf_cell(pdf, f"Separation: {doctor.get('separation_score', 0)}%")
    safe_pdf_cell(pdf, f"Overall: {doctor.get('overall_score', 0)}%")
    if reliability:
        pdf.ln(4)
        safe_pdf_cell(pdf, "Reliability Estimate", 8, True)
        safe_pdf_cell(pdf, f"Synthetic Realism: {reliability.get('synthetic_realism_score', 0)}%")
        safe_pdf_cell(pdf, f"Field Readiness: {reliability.get('field_readiness_score', 0)}%")
        safe_pdf_cell(pdf, f"Reliability: {reliability.get('reliability_score', 0)}%")
        safe_pdf_cell(pdf, f"Dataset Risk: {reliability.get('dataset_risk', 'Unknown')}")
        safe_pdf_cell(pdf, f"Recommended real samples needed: {reliability.get('recommended_real_samples_needed', 0)}")
        safe_pdf_multicell(pdf, reliability.get("verdict", ""))
    if hardware_result:
        pdf.ln(4)
        safe_pdf_cell(pdf, "Hardware Direction", 8, True)
        safe_pdf_cell(pdf, f"Recommended board: {hardware_result.get('recommendation', 'Unknown')}")
        safe_pdf_multicell(pdf, hardware_result.get("reason", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Label Distribution", 8, True)
    for label, count in doctor.get("label_counts", {}).items():
        safe_pdf_cell(pdf, f"- {label}: {count}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Dataset Doctor Advice", 8, True)
    for item in doctor.get("advice", []):
        safe_pdf_multicell(pdf, f"[{item.get('severity', 'info').upper()}] {item.get('message', '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Recommended Training Features", 8, True)
    for col in doctor.get("recommended_training_features", []):
        safe_pdf_cell(pdf, f"- {col}")
    if commercial_summary:
        pdf.ln(4)
        safe_pdf_cell(pdf, "Recommended Next Step", 8, True)
        safe_pdf_multicell(pdf, commercial_summary.get("cta", "Upload real sensor files or request a custom industry pack."))
    return safe_pdf_output(pdf)


def create_sensor_fusion_export_bundle(project_name, fusion_df, manifest, commercial_summary=None, reliability=None, hardware_result=None):
    template_name = manifest.get("template")
    doctor = fusion_dataset_doctor(fusion_df, template_name)
    training_df = create_fusion_training_dataframe(fusion_df, template_name)
    pdf_bytes = generate_fusion_pdf_report(project_name, fusion_df, manifest, doctor, commercial_summary, reliability, hardware_result)
    bundle_manifest = dict(manifest)
    bundle_manifest["doctor"] = doctor
    bundle_manifest["commercial_summary"] = commercial_summary or {}
    bundle_manifest["reliability"] = reliability or {}
    bundle_manifest["hardware"] = hardware_result or {}
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sensor_fusion_full_dataset.csv", fusion_df.to_csv(index=False))
        zf.writestr("sensor_fusion_training_features.csv", training_df.to_csv(index=False))
        zf.writestr("manifest.json", json.dumps(bundle_manifest, indent=2))
        zf.writestr("fusion_report.pdf", pdf_bytes)
    return zip_buf.getvalue(), doctor, training_df


def create_auto_pilot_bundle(project_name, pilot_result):
    fusion_df = pilot_result["fusion_df"]
    training_df = pilot_result["training_df"]
    manifest = pilot_result["manifest"]
    doctor = pilot_result["doctor"]
    reliability = pilot_result["reliability"]
    hardware_result = pilot_result["hardware"]
    summary = pilot_result["commercial_summary"]
    pdf_bytes = generate_fusion_pdf_report(project_name, fusion_df, manifest, doctor, summary, reliability, hardware_result)
    readme = f"""EdgeTwin Studio V19 Auto Pilot Bundle

Project: {project_name}
Use case: {manifest.get('use_case_type')}
Template: {manifest.get('template')}
Samples: {manifest.get('samples')}
Recommended hardware: {hardware_result.get('recommendation')}
Reliability score: {reliability.get('reliability_score')}%

This bundle is intended for Edge AI pilot preparation. Validate with real field data before production deployment.
"""
    full_manifest = dict(manifest)
    full_manifest["doctor"] = doctor
    full_manifest["reliability"] = reliability
    full_manifest["hardware"] = hardware_result
    full_manifest["commercial_summary"] = summary
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("pilot_report.pdf", pdf_bytes)
        zf.writestr("pilot_full_dataset.csv", fusion_df.to_csv(index=False))
        zf.writestr("pilot_training_features.csv", training_df.to_csv(index=False))
        zf.writestr("hardware_advice.json", json.dumps(hardware_result, indent=2))
        zf.writestr("manifest.json", json.dumps(full_manifest, indent=2))
        zf.writestr("README.txt", readme)
    return zip_buf.getvalue()


def create_enterprise_bundle(project_name, dataset_df, audit_result, hardware_result=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Enterprise Audit", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)
    safe_pdf_cell(pdf, "Readiness Scores", 8, True)
    safe_pdf_cell(pdf, f"Diversity: {audit_result.get('diversity_score', 0)}%")
    safe_pdf_cell(pdf, f"Balance: {audit_result.get('balance_score', 0)}%")
    safe_pdf_cell(pdf, f"Separation: {audit_result.get('separation_score', 0)}%")
    safe_pdf_cell(pdf, f"Overall: {audit_result.get('overall_score', 0)}%")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Advice", 8, True)
    for item in audit_result.get("advice", []):
        safe_pdf_multicell(pdf, f"[{item.get('severity', 'info').upper()}] {item.get('message', '')}")
    if hardware_result:
        pdf.ln(4)
        safe_pdf_cell(pdf, "Hardware Recommendation", 8, True)
        safe_pdf_cell(pdf, f"Recommended board: {hardware_result.get('recommendation', 'Unknown')}")
        safe_pdf_multicell(pdf, hardware_result.get("reason", ""))
    metadata = {"project_name": project_name, "created_at": str(datetime.datetime.now()), "audit": audit_result, "hardware": hardware_result or {}}
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("edge_dataset.csv", dataset_df.to_csv(index=False))
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        zf.writestr("audit_report.pdf", safe_pdf_output(pdf))
    return zip_buf.getvalue()
