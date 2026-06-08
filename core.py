import io
import json
import uuid
import secrets
import sqlite3
import zipfile
import datetime

try:
    import bcrypt
except ImportError:  # Streamlit Cloud/local fallback when bcrypt is not installed
    bcrypt = None

import hashlib
import hmac
import os
import re

import numpy as np
import pandas as pd
from fpdf import FPDF
from scipy.io import wavfile
from scipy.stats import kurtosis
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    import storage as et_storage
except Exception:  # keep local MVP running even if storage.py is not present
    et_storage = None

DB_NAME = "edgetwin_v19.db"
RELIABILITY_DISCLAIMER = (
    "Reliability is a pilot estimate based on synthetic dataset structure, class balance, "
    "label separation and sensor coverage. Field validation is required before production deployment."
)


# ============================================================
# SMALL UTILITIES
# ============================================================

def _json_safe(obj):
    """Convert numpy/pandas objects into JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if pd.isna(obj) if not isinstance(obj, (dict, list, tuple, str, bytes)) else False:
        return None
    return obj


def _now():
    return str(datetime.datetime.now())


def hash_password(password):
    """Use bcrypt when available, otherwise PBKDF2-SHA256 fallback."""
    password = password or ""
    if bcrypt is not None:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    salt = os.urandom(16)
    iterations = 260000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password, stored_hash):
    password = password or ""
    stored_hash = stored_hash or ""
    if stored_hash.startswith("$2") and bcrypt is not None:
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    if stored_hash.startswith("pbkdf2_sha256$"):
        try:
            _, iter_s, salt_hex, digest_hex = stored_hash.split("$", 3)
            digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iter_s))
            return hmac.compare_digest(digest.hex(), digest_hex)
        except Exception:
            return False
    return False



# ============================================================
# V31.1 STORAGE / SCALABILITY HELPERS
# ============================================================

PROJECT_STORAGE_COLUMNS = {
    "updated_at": "TEXT",
    "dataset_path": "TEXT",
    "dataset_hash": "TEXT",
    "dataset_format": "TEXT",
    "storage_mode": "TEXT",
    "dataset_rows": "INTEGER DEFAULT 0",
    "dataset_cols": "INTEGER DEFAULT 0",
}

PLAN_SAMPLE_LIMITS = {
    "Free Demo": 500,
    "Starter Pilot": 2500,
    "Professional Pilot": 10000,
    "Real-Data Pilot": 25000,
    "Enterprise": 100000,
    "Founder Test Mode": 100000,
}


def _ensure_project_storage_columns(cursor):
    """Add V31.1 storage metadata columns without breaking old SQLite databases."""
    try:
        cursor.execute("PRAGMA table_info(projects)")
        existing = {row[1] for row in cursor.fetchall()}
        for col, col_type in PROJECT_STORAGE_COLUMNS.items():
            if col not in existing:
                cursor.execute(f"ALTER TABLE projects ADD COLUMN {col} {col_type}")
    except Exception:
        # init_db should never crash just because an old DB cannot be migrated.
        pass


def dataset_hash(dataset_df):
    if et_storage is not None and isinstance(dataset_df, pd.DataFrame):
        try:
            return et_storage.dataframe_hash(dataset_df)
        except Exception:
            pass
    if not isinstance(dataset_df, pd.DataFrame) or dataset_df.empty:
        return "empty"
    return hashlib.sha256(dataset_df.to_csv(index=False).encode("utf-8", errors="replace")).hexdigest()


def save_dataset_to_storage(project_id, dataset_df, kind="dataset"):
    if et_storage is None or not isinstance(dataset_df, pd.DataFrame):
        return None
    return et_storage.save_dataframe(dataset_df, project_id, kind=kind)


def load_dataset_from_storage(path, fmt=None):
    if et_storage is None or not path:
        return pd.DataFrame()
    return et_storage.load_dataframe(path, fmt)


def get_storage_status():
    status = {
        "mode": "sqlite_json_legacy",
        "storage_root": "not configured",
        "file_count": 0,
        "total_mb": 0,
        "parquet_preferred": False,
        "fallback_format": "sqlite_json",
        "future_cloud_target": "PostgreSQL + S3/MinIO later",
    }
    if et_storage is not None:
        try:
            status.update(et_storage.get_storage_status())
        except Exception as exc:
            status["error"] = str(exc)
    status["database"] = DB_NAME
    status["database_role"] = "metadata/users/projects only; large datasets should live in file/object storage"
    return status


def get_sample_limit_for_plan(plan_name):
    return int(PLAN_SAMPLE_LIMITS.get(plan_name, PLAN_SAMPLE_LIMITS["Free Demo"]))


def validate_sample_request(plan_name, requested_samples):
    requested = int(requested_samples or 0)
    limit = get_sample_limit_for_plan(plan_name)
    allowed = min(max(requested, 0), limit)
    return {
        "plan": plan_name,
        "requested_samples": requested,
        "allowed_samples": allowed,
        "limit": limit,
        "was_capped": requested > limit,
        "message": "OK" if requested <= limit else f"Sample request capped to {limit} for {plan_name}.",
    }


def get_scalability_readiness_snapshot(dataset_df=None, plan_name="Founder Test Mode"):
    rows = int(len(dataset_df)) if isinstance(dataset_df, pd.DataFrame) else 0
    cols = int(len(dataset_df.columns)) if isinstance(dataset_df, pd.DataFrame) else 0
    storage_status = get_storage_status()
    limit = get_sample_limit_for_plan(plan_name)
    issues = []
    if storage_status.get("mode") == "sqlite_json_legacy":
        issues.append({"severity": "high", "message": "File storage layer is not available; large datasets may be stored in SQLite JSON fallback."})
    if rows > limit:
        issues.append({"severity": "medium", "message": f"Current dataset has {rows} rows, above the configured plan limit of {limit}."})
    if rows > 25000:
        issues.append({"severity": "medium", "message": "Large dataset detected. Prefer parquet/object storage and avoid full in-memory audits."})
    if rows == 0:
        issues.append({"severity": "info", "message": "No active dataset loaded yet."})
    if not issues:
        issues.append({"severity": "info", "message": "Storage/scalability checks look acceptable for local beta use."})
    score = 100
    for item in issues:
        if item["severity"] == "high":
            score -= 35
        elif item["severity"] == "medium":
            score -= 15
        else:
            score -= 0
    score = int(np.clip(score, 0, 100))
    return {
        "scalability_score": score,
        "storage": storage_status,
        "dataset_rows": rows,
        "dataset_cols": cols,
        "dataset_hash": dataset_hash(dataset_df) if isinstance(dataset_df, pd.DataFrame) else "empty",
        "plan": plan_name,
        "sample_limit": limit,
        "issues": issues,
        "recommendation": "Ready for local/private beta" if score >= 80 else "Use carefully; harden before multi-user SaaS",
        "next_step": "PostgreSQL + S3/MinIO when moving from closed beta to public SaaS.",
    }

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
    _ensure_project_storage_columns(c)
    conn.commit()
    conn.close()


def create_user(username, password):
    username = (username or "").strip()
    password = password or ""
    if not username or len(password) < 6:
        return None

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        user_id = str(uuid.uuid4())
        api_key = "ets_" + secrets.token_hex(16)
        password_hash = hash_password(password)
        c.execute(
            "INSERT INTO users (id, username, password_hash, api_key, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, password_hash, api_key, _now()),
        )
        conn.commit()
        return {"id": user_id, "username": username, "api_key": api_key}
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def authenticate_user(username, password):
    username = (username or "").strip()
    password = password or ""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, username, password_hash, api_key FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    if verify_password(password, row[2]):
        return {"id": row[0], "username": row[1], "api_key": row[3]}
    return None


def save_project(proj_id, user_id, name, dataset_df, settings_dict):
    """Save project metadata in SQLite and large datasets in V31.1 file storage.

    Backward compatible: if file storage fails, the old JSON-in-SQLite path is used.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    _ensure_project_storage_columns(c)

    dataset_json = "[]"
    dataset_path = ""
    dataset_format = "none"
    storage_mode = "empty"
    ds_hash = "empty"
    rows = int(len(dataset_df)) if isinstance(dataset_df, pd.DataFrame) else 0
    cols = int(len(dataset_df.columns)) if isinstance(dataset_df, pd.DataFrame) else 0

    if isinstance(dataset_df, pd.DataFrame) and rows > 0:
        ds_hash = dataset_hash(dataset_df)
        try:
            meta = save_dataset_to_storage(proj_id, dataset_df, kind="dataset")
            if meta:
                dataset_path = meta.get("path", "")
                dataset_format = meta.get("format", "unknown")
                ds_hash = meta.get("hash", ds_hash)
                rows = int(meta.get("rows", rows))
                cols = int(meta.get("cols", cols))
                storage_mode = "file_storage_v31_1"
            else:
                raise RuntimeError("storage layer unavailable")
        except Exception:
            dataset_json = dataset_df.to_json(orient="records")
            storage_mode = "sqlite_json_fallback"
            dataset_format = "json_records"

    safe_settings = json.dumps(_json_safe(settings_dict), ensure_ascii=False)
    now = _now()
    c.execute(
        "SELECT created_at FROM projects WHERE id = ? AND user_id = ?",
        (proj_id, user_id),
    )
    existing = c.fetchone()
    created_at = existing[0] if existing and existing[0] else now

    c.execute(
        """
        REPLACE INTO projects (
            id, user_id, name, created_at, updated_at, dataset, settings,
            dataset_path, dataset_hash, dataset_format, storage_mode, dataset_rows, dataset_cols
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            proj_id, user_id, name, created_at, now, dataset_json, safe_settings,
            dataset_path, ds_hash, dataset_format, storage_mode, rows, cols,
        ),
    )
    conn.commit()
    conn.close()


def load_project(proj_id, user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    _ensure_project_storage_columns(c)
    c.execute(
        """
        SELECT name, dataset, settings, dataset_path, dataset_hash, dataset_format,
               storage_mode, dataset_rows, dataset_cols, updated_at
        FROM projects WHERE id = ? AND user_id = ?
        """,
        (proj_id, user_id),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None

    name, dataset_json, settings_json, dataset_path, ds_hash, dataset_format, storage_mode, dataset_rows, dataset_cols, updated_at = row

    dataset = pd.DataFrame()
    if dataset_path:
        dataset = load_dataset_from_storage(dataset_path, dataset_format)
    if (dataset is None or dataset.empty) and dataset_json:
        try:
            dataset = pd.read_json(io.StringIO(dataset_json))
        except Exception:
            dataset = pd.DataFrame()
    if dataset is None:
        dataset = pd.DataFrame()

    try:
        settings = json.loads(settings_json) if settings_json else {}
    except Exception:
        settings = {}

    storage_info = {
        "dataset_path": dataset_path or "",
        "dataset_hash": ds_hash or dataset_hash(dataset),
        "dataset_format": dataset_format or ("json_records" if dataset_json else "none"),
        "storage_mode": storage_mode or ("sqlite_json_legacy" if dataset_json else "empty"),
        "dataset_rows": int(dataset_rows or len(dataset)),
        "dataset_cols": int(dataset_cols or len(dataset.columns)),
        "updated_at": updated_at,
    }
    return {
        "name": name,
        "dataset": dataset,
        "settings": settings,
        "storage": storage_info,
    }


def get_user_projects(user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    _ensure_project_storage_columns(c)
    df = pd.read_sql_query(
        """
        SELECT id, name, created_at, updated_at, storage_mode, dataset_rows, dataset_cols, dataset_format
        FROM projects WHERE user_id = ? ORDER BY updated_at DESC, created_at DESC
        """,
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


def generate_balanced_sensor_values_for_target(template_name, target_score, base_values=None):
    """Create balanced class samples while still letting UI sliders influence the sensor profile."""
    template = get_fusion_template(template_name) or FUSION_TEMPLATES["Smart Forestry Threat"]
    weights = template["weights"]
    defaults = template.get("defaults", {})
    base_values = base_values or defaults
    target = float(np.clip(target_score, 0, 100))

    def mix(sensor_key, noise=12):
        base = float(base_values.get(sensor_key, defaults.get(sensor_key, target)))
        return np.clip((target * 0.68) + (base * 0.32) + np.random.normal(0, noise) + weights.get(sensor_key, 0.2) * 8, 0, 100)

    return (
        mix("audio", 12),
        mix("vibration", 12),
        mix("gas", 10),
        mix("radar", 13),
        mix("gps", 10),
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


SENSOR_SCORE_COLUMNS = {
    "Audio": "AudioScore",
    "Vibration": "VibrationScore",
    "Radar": "RadarScore",
    "Gas / Environment": "GasScore",
    "GPS / Zone": "GPSZoneScore",
    "Temperature": "GasScore",
    "Humidity": "GasScore",
    "IMU / Movement": "VibrationScore",
}

SENSOR_KEY_MAP = {
    "Audio": "audio",
    "Vibration": "vibration",
    "Radar": "radar",
    "Gas / Environment": "gas",
    "GPS / Zone": "gps",
    "Temperature": "gas",
    "Humidity": "gas",
    "IMU / Movement": "vibration",
}


def active_sensor_keys(selected_sensors):
    return {SENSOR_KEY_MAP[s] for s in selected_sensors or [] if s in SENSOR_KEY_MAP}


def get_fusion_training_columns(template_name=None, selected_sensors=None):
    template = get_fusion_template(template_name) if template_name else None
    mode = template.get("mode", "threat") if template else "threat"
    if selected_sensors:
        sensor_cols = []
        for s in selected_sensors:
            col = SENSOR_SCORE_COLUMNS.get(s)
            if col and col not in sensor_cols:
                sensor_cols.append(col)
    else:
        sensor_cols = ["AudioScore", "VibrationScore", "GasScore", "RadarScore", "GPSZoneScore"]
    cols = ["Label", *sensor_cols, "Confidence"]
    cols.append("HealthScore" if mode == "health" else "FusionScore")
    return cols


def create_fusion_training_dataframe(fusion_df, template_name=None, selected_sensors=None):
    cols = get_fusion_training_columns(template_name, selected_sensors)
    return fusion_df[[c for c in cols if c in fusion_df.columns]].copy()


def generate_sensor_fusion_dataset(
    template_name,
    samples=500,
    base_audio=None,
    base_vibration=None,
    base_gas=None,
    base_radar=None,
    base_gps=None,
    include_scenario_variants=True,
    balanced_classes=True,
):
    template = get_fusion_template(template_name) or FUSION_TEMPLATES["Smart Forestry Threat"]
    classes = template.get("classes", ["Normal", "Elevated", "High", "Critical"])
    defaults = template.get("defaults", {})

    base_values = {
        "audio": defaults.get("audio", 50) if base_audio is None else base_audio,
        "vibration": defaults.get("vibration", 50) if base_vibration is None else base_vibration,
        "gas": defaults.get("gas", 20) if base_gas is None else base_gas,
        "radar": defaults.get("radar", 50) if base_radar is None else base_radar,
        "gps": defaults.get("gps", 50) if base_gps is None else base_gps,
    }

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
            vals = generate_balanced_sensor_values_for_target(template_name, target, base_values=base_values)
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
                base_values["audio"] * multiplier,
                base_values["vibration"] * multiplier,
                base_values["gas"] * multiplier,
                base_values["radar"] * multiplier,
                base_values["gps"] * multiplier,
                jitter=True,
            ))
    df = pd.DataFrame(rows)
    manifest = {
        "engine": "EdgeTwin Studio V20 - Powered by OMEGA-X Engine",
        "template": template_name,
        "description": template.get("description", ""),
        "samples": samples,
        "balanced_classes": bool(balanced_classes),
        "base_sensor_values": base_values,
        "sensors": ["audio", "vibration", "gas", "radar", "gps"],
        "columns": list(df.columns),
        "recommended_training_features": get_fusion_training_columns(template_name),
        "reliability_disclaimer": RELIABILITY_DISCLAIMER,
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
    reliability = calculate_reliability_score(doctor, has_real_data=False, selected_sensors=["Audio", "Vibration", "Radar", "GPS / Zone"])
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
# V19.1 USE CASE WIZARD / AUTO PILOT GENERATOR
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
        "created_at": _now(),
    }


def sensor_weights_from_selection(selected_sensors, use_case_type="Custom Sensor Fusion"):
    defaults = get_use_case_defaults(use_case_type)
    template = get_fusion_template(defaults["template"]) or FUSION_TEMPLATES["Remote Asset Security"]
    base = template.get("weights", {"audio": 0.25, "vibration": 0.25, "gas": 0.10, "radar": 0.30, "gps": 0.10}).copy()
    active = active_sensor_keys(selected_sensors)
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
    selected_sensors = config.get("selected_sensors", [])
    active_keys = active_sensor_keys(selected_sensors)
    weights = sensor_weights_from_selection(selected_sensors, config.get("use_case_type", "Custom Sensor Fusion"))
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

        values = {}
        for key, noise in [("audio", 14), ("vibration", 14), ("gas", 12), ("radar", 15), ("gps", 12)]:
            if key in active_keys:
                values[key] = float(np.clip(base + np.random.normal(0, noise) + weights.get(key, 0.0) * 18, 0, 100))
            else:
                values[key] = 0.0

        fusion = float(np.clip(sum(values[k] * weights.get(k, 0.0) for k in values), 0, 100))
        health = float(np.clip(100 - fusion, 0, 100))
        active_vals = [values[k] for k in active_keys] or list(values.values())
        confidence = float(np.clip(100 - np.std(active_vals) * 0.70, 0, 100))

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
            "AudioScore": values["audio"],
            "VibrationScore": values["vibration"],
            "GasScore": values["gas"],
            "RadarScore": values["radar"],
            "GPSZoneScore": values["gps"],
            "ActiveSensors": ", ".join(selected_sensors),
            "FusionScore": fusion,
            "HealthScore": health,
            "Confidence": confidence,
            "Level": level,
            "Event": event,
            "RecommendedAction": action,
        })
    df = pd.DataFrame(rows)
    manifest = {
        "engine": "EdgeTwin Studio V20 Auto Pilot Generator",
        "template": template_name,
        "use_case_type": config.get("use_case_type"),
        "project_goal": config.get("project_goal"),
        "environment": config.get("environment"),
        "selected_sensors": selected_sensors,
        "active_sensor_keys": sorted(list(active_keys)),
        "classes": classes,
        "samples": samples,
        "sample_rate": config.get("sample_rate", 16000),
        "has_real_data": config.get("has_real_data", False),
        "output_level": config.get("output_level"),
        "priority": config.get("priority"),
        "weights": weights,
        "columns": list(df.columns),
        "recommended_training_features": get_fusion_training_columns(template_name, selected_sensors),
        "reliability_disclaimer": RELIABILITY_DISCLAIMER,
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
        "disclaimer": RELIABILITY_DISCLAIMER,
    }


def run_auto_pilot_project(config):
    fusion_df, manifest = generate_custom_fusion_dataset(config)
    selected_sensors = manifest.get("selected_sensors", [])
    doctor = fusion_dataset_doctor(fusion_df, manifest.get("template"), selected_sensors=selected_sensors)
    training_df = create_fusion_training_dataframe(fusion_df, manifest.get("template"), selected_sensors=selected_sensors)
    hw = hardware_auto_architect(max(1, len(training_df.columns) - 1), manifest.get("sample_rate", config.get("sample_rate", 16000)), config.get("priority", "balanced"))
    reliability = calculate_reliability_score(doctor, config.get("has_real_data", False), selected_sensors)
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
        adjusted = float(score)
        if target == "low_power":
            adjusted += {"Very Low": 12, "Low": 8, "Medium": 3}.get(p["power_class"], 0) - latency * 0.20
        elif target == "performance":
            adjusted -= latency * 1.0
        elif target == "gateway":
            adjusted += 20 if p["gateway_fit"] == "Yes" else 0
            adjusted -= latency * 0.35
        else:
            adjusted -= latency * 0.50
        rows.append({"board": board, "cpu": p["cpu"], "role": p["role"], "power_class": p["power_class"], "score": float(score), "adjusted_score": float(adjusted), "latency_ms": float(latency), "ram_kb": float(ram), "gateway_fit": p["gateway_fit"], "notes": p["notes"]})
    rows = sorted(rows, key=lambda r: r["adjusted_score"], reverse=True)
    best = rows[0] if rows else {"board": "Unknown"}
    return {"recommendation": best["board"], "ranking": rows, "reason": f"Best fit for {target}: {best['board']} with {best.get('latency_ms', 0):.1f} ms estimated latency."}


# ============================================================
# DATASET DOCTOR / EXPORTS
# ============================================================

def calculate_audit_scores(X_df, y_series, max_audit_samples=1500):
    if X_df is None or len(X_df) == 0 or y_series is None or len(y_series) == 0:
        return 0, 0, 0

    X_df = X_df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    y_series = pd.Series(y_series).reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    if len(X_df) > max_audit_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_df), max_audit_samples, replace=False)
        X_df = X_df.iloc[idx].reset_index(drop=True)
        y_series = y_series.iloc[idx].reset_index(drop=True)

    try:
        X_scaled = StandardScaler().fit_transform(X_df)
    except Exception:
        return 0, 0, 0

    label_counts = y_series.value_counts()
    div = min(100, int((np.mean(pdist(X_scaled)) / 4.0) * 100)) if len(X_scaled) > 1 else 0
    bal = 100 if len(label_counts) >= 2 and label_counts.max() > 0 and (label_counts.min() / label_counts.max()) > 0.5 else 50
    try:
        sep = int((silhouette_score(X_scaled, y_series) + 1) * 50) if len(y_series.unique()) >= 2 and label_counts.min() >= 2 and len(y_series.unique()) < len(y_series) else 0
    except Exception:
        sep = 0
    return int(div), int(bal), int(sep)


def fusion_dataset_doctor(fusion_df, template_name=None, selected_sensors=None):
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

    if selected_sensors:
        sensor_cols = [SENSOR_SCORE_COLUMNS[s] for s in selected_sensors if s in SENSOR_SCORE_COLUMNS]
        sensor_cols = list(dict.fromkeys(sensor_cols))
    else:
        sensor_cols = ["AudioScore", "VibrationScore", "GasScore", "RadarScore", "GPSZoneScore"]

    missing = [c for c in sensor_cols if c not in df.columns]
    if missing:
        advice.append(f"Ontbrekende sensor-kolommen: {', '.join(missing)}.")
        severity.append("high")
    else:
        low_var = df[sensor_cols].std(numeric_only=True)
        low_var = low_var[low_var < 3].index.tolist()
        if low_var:
            advice.append(f"Lage actieve sensor-variatie bij: {', '.join(low_var)}.")
            severity.append("medium")
        else:
            advice.append("Actieve sensorvariatie ziet er gezond uit.")
            severity.append("info")

    training_df = create_fusion_training_dataframe(df, template_name, selected_sensors=selected_sensors)
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
        "label_counts": {str(k): int(v) for k, v in label_counts.to_dict().items()},
        "recommended_training_features": get_fusion_training_columns(template_name, selected_sensors=selected_sensors),
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



# ============================================================
# V20 SMART DATASET OPTIMIZER
# ============================================================

def _numeric_feature_columns(df, label_col="Label"):
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return []
    return [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]


def find_redundant_features(df, label_col="Label", corr_threshold=0.97):
    numeric_cols = _numeric_feature_columns(df, label_col)
    if len(numeric_cols) < 2:
        return []
    clean = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    corr = clean.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    redundant = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    return redundant


def find_low_variance_features(df, label_col="Label", min_std=1e-9):
    numeric_cols = _numeric_feature_columns(df, label_col)
    if not numeric_cols:
        return []
    stds = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).std(numeric_only=True)
    return stds[stds <= min_std].index.tolist()


def smart_dataset_optimizer_report(df, label_col="Label"):
    """Analyze a dataset and propose concrete V20 optimizer actions."""
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return {
            "status": "empty",
            "current_score": 0,
            "target_per_class": 0,
            "recommended_actions": [{"severity": "high", "message": "Geen dataset aanwezig om te optimaliseren."}],
        }
    if label_col not in df.columns:
        return {
            "status": "missing_label",
            "current_score": 0,
            "target_per_class": 0,
            "recommended_actions": [{"severity": "high", "message": "Dataset heeft een Label-kolom nodig."}],
        }

    numeric_cols = _numeric_feature_columns(df, label_col)
    if not numeric_cols:
        return {
            "status": "no_numeric_features",
            "current_score": 0,
            "target_per_class": 0,
            "recommended_actions": [{"severity": "high", "message": "Dataset heeft numerieke feature-kolommen nodig."}],
        }

    y = df[label_col].astype(str)
    counts = y.value_counts()
    audit = dataset_doctor(df[numeric_cols], y)
    max_count = int(counts.max()) if len(counts) else 0
    min_count = int(counts.min()) if len(counts) else 0
    target_per_class = max(max_count, 50) if len(counts) >= 2 else max_count
    weak_classes = counts[counts < target_per_class].to_dict()
    redundant = find_redundant_features(df, label_col)
    low_var = find_low_variance_features(df, label_col)

    # Simple class distance overview for explainability.
    closest_pair = None
    try:
        means = df.assign(**{label_col: y}).groupby(label_col)[numeric_cols].mean()
        if len(means) >= 2:
            scaler = StandardScaler()
            scaled_means = pd.DataFrame(scaler.fit_transform(means), index=means.index, columns=means.columns)
            distances = []
            labels = list(scaled_means.index)
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    dist = float(np.linalg.norm(scaled_means.loc[labels[i]].values - scaled_means.loc[labels[j]].values))
                    distances.append((labels[i], labels[j], dist))
            if distances:
                closest_pair = sorted(distances, key=lambda x: x[2])[0]
    except Exception:
        closest_pair = None

    actions = []
    if len(counts) < 2:
        actions.append({"severity": "high", "action": "add_classes", "message": "Voeg minimaal twee labels/classes toe."})
    elif min_count / max(max_count, 1) < 0.70:
        actions.append({"severity": "high", "action": "balance_classes", "message": "Class balance is zwak. Balance classes of voeg weak-class samples toe."})
    else:
        actions.append({"severity": "info", "action": "balance_ok", "message": "Class balance is bruikbaar."})

    if audit.get("separation_score", 0) < 65:
        actions.append({"severity": "medium", "action": "improve_label_separation", "message": "Label separation kan beter. Versterk verschil tussen klassen."})
    else:
        actions.append({"severity": "info", "action": "separation_ok", "message": "Label separation is goed bruikbaar voor een pilot."})

    if redundant:
        actions.append({"severity": "medium", "action": "reduce_redundant_features", "message": f"Redundante features gevonden: {', '.join(redundant[:6])}."})
    if low_var:
        actions.append({"severity": "medium", "action": "remove_low_variance_features", "message": f"Lage-variatie features gevonden: {', '.join(low_var[:6])}."})
    if len(df) < 200:
        actions.append({"severity": "medium", "action": "add_samples", "message": "Dataset is klein. Voeg meer samples toe voor een sterkere pilot."})
    if closest_pair:
        actions.append({"severity": "info", "action": "closest_classes", "message": f"Meest vergelijkbare classes: {closest_pair[0]} vs {closest_pair[1]}."})

    return {
        "status": "ok",
        "rows": int(len(df)),
        "numeric_features": numeric_cols,
        "feature_count": int(len(numeric_cols)),
        "class_counts": {str(k): int(v) for k, v in counts.to_dict().items()},
        "weak_classes": {str(k): int(v) for k, v in weak_classes.items()},
        "weakest_class": str(counts.idxmin()) if len(counts) else None,
        "majority_class": str(counts.idxmax()) if len(counts) else None,
        "target_per_class": int(target_per_class),
        "imbalance_ratio": float(min_count / max(max_count, 1)) if max_count else 0.0,
        "redundant_features": redundant,
        "low_variance_features": low_var,
        "closest_class_pair": closest_pair,
        "current_audit": audit,
        "current_score": int(audit.get("overall_score", 0)),
        "recommended_actions": actions,
        "disclaimer": RELIABILITY_DISCLAIMER,
    }


def _augment_rows_for_class(class_df, numeric_cols, n_needed, noise_strength=0.03, random_state=42):
    if n_needed <= 0 or len(class_df) == 0:
        return pd.DataFrame(columns=class_df.columns)
    rng = np.random.default_rng(random_state)
    sampled = class_df.sample(n=n_needed, replace=True, random_state=random_state).copy().reset_index(drop=True)
    for col in numeric_cols:
        values = class_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        std = float(values.std())
        scale = std if std > 1e-9 else max(abs(float(values.mean())), 1.0)
        sampled[col] = sampled[col].astype(float) + rng.normal(0, scale * float(noise_strength), size=len(sampled))
    return sampled


def run_smart_dataset_optimizer(
    df,
    actions=None,
    label_col="Label",
    target_per_class=None,
    noise_strength=0.03,
    separation_strength=0.08,
    random_state=42,
):
    """Apply safe, explainable dataset improvements for pilot datasets."""
    actions = actions or []
    before_df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    before_report = smart_dataset_optimizer_report(before_df, label_col=label_col)
    if before_report.get("status") != "ok":
        return {
            "optimized_df": before_df,
            "before_report": before_report,
            "after_report": before_report,
            "changes": [{"type": "skipped", "message": "Optimizer skipped because dataset is not valid."}],
        }

    out = before_df.copy().reset_index(drop=True)
    numeric_cols = before_report.get("numeric_features", [])
    changes = []
    target_per_class = int(target_per_class or before_report.get("target_per_class", 0) or 0)

    if "balance_classes" in actions and target_per_class > 0:
        parts = [out]
        counts = out[label_col].astype(str).value_counts()
        added_total = 0
        for label, count in counts.items():
            n_needed = max(0, target_per_class - int(count))
            if n_needed > 0:
                class_df = out[out[label_col].astype(str) == str(label)].copy()
                added = _augment_rows_for_class(class_df, numeric_cols, n_needed, noise_strength=noise_strength, random_state=random_state + added_total + len(str(label)))
                parts.append(added)
                added_total += len(added)
        out = pd.concat(parts, ignore_index=True) if parts else out
        changes.append({"type": "balance_classes", "message": f"Added {added_total} synthetic weak-class samples up to target {target_per_class} per class."})

    if "add_realistic_noise" in actions and numeric_cols:
        rng = np.random.default_rng(random_state + 17)
        noisy = out.copy()
        for col in numeric_cols:
            values = out[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            std = float(values.std())
            scale = std if std > 1e-9 else max(abs(float(values.mean())), 1.0)
            noisy[col] = noisy[col].astype(float) + rng.normal(0, scale * float(noise_strength), size=len(noisy))
        out = pd.concat([out, noisy], ignore_index=True)
        changes.append({"type": "add_realistic_noise", "message": f"Added {len(noisy)} noisy variants to improve robustness."})

    if "improve_label_separation" in actions and numeric_cols and out[label_col].nunique() >= 2:
        global_mean = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).mean()
        adjusted = out.copy()
        for label in adjusted[label_col].astype(str).unique():
            mask = adjusted[label_col].astype(str) == label
            class_mean = adjusted.loc[mask, numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).mean()
            direction = class_mean - global_mean
            adjusted.loc[mask, numeric_cols] = adjusted.loc[mask, numeric_cols].astype(float) + (direction * float(separation_strength))
        out = adjusted
        changes.append({"type": "improve_label_separation", "message": "Shifted class feature centers slightly apart for better pilot separation."})

    removed_features = []
    if "reduce_redundant_features" in actions:
        removed_features = list(dict.fromkeys(find_redundant_features(out, label_col) + find_low_variance_features(out, label_col)))
        if removed_features:
            out = out.drop(columns=[c for c in removed_features if c in out.columns])
        changes.append({"type": "reduce_redundant_features", "message": f"Removed {len(removed_features)} redundant/low-variance features.", "features": removed_features})

    # Keep clean row order and JSON-friendly dtypes where possible.
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index(drop=True)
    after_report = smart_dataset_optimizer_report(out, label_col=label_col)
    return {
        "optimized_df": out,
        "before_report": before_report,
        "after_report": after_report,
        "changes": changes,
        "removed_features": removed_features,
        "created_at": _now(),
    }


def create_optimizer_bundle(project_name, before_df, optimized_result):
    optimized_df = optimized_result.get("optimized_df", pd.DataFrame())
    before_report = optimized_result.get("before_report", {})
    after_report = optimized_result.get("after_report", {})
    changes = optimized_result.get("changes", [])

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Smart Dataset Optimizer", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)
    safe_pdf_cell(pdf, "Optimization Summary", 8, True)
    safe_pdf_cell(pdf, f"Rows before: {len(before_df) if isinstance(before_df, pd.DataFrame) else 0}")
    safe_pdf_cell(pdf, f"Rows after: {len(optimized_df) if isinstance(optimized_df, pd.DataFrame) else 0}")
    safe_pdf_cell(pdf, f"Score before: {before_report.get('current_score', 0)}%")
    safe_pdf_cell(pdf, f"Score after: {after_report.get('current_score', 0)}%")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Applied Changes", 8, True)
    for ch in changes:
        safe_pdf_multicell(pdf, f"- {ch.get('message', '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Validation Note", 8, True)
    safe_pdf_multicell(pdf, RELIABILITY_DISCLAIMER)

    manifest = {
        "project_name": project_name,
        "created_at": _now(),
        "before_report": before_report,
        "after_report": after_report,
        "changes": changes,
        "disclaimer": RELIABILITY_DISCLAIMER,
    }
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        if isinstance(before_df, pd.DataFrame):
            zf.writestr("dataset_before_optimizer.csv", before_df.to_csv(index=False))
        zf.writestr("dataset_after_optimizer.csv", optimized_df.to_csv(index=False))
        zf.writestr("optimizer_manifest.json", json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False))
        zf.writestr("optimizer_report.pdf", safe_pdf_output(pdf))
    return zip_buf.getvalue()

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
        safe_pdf_multicell(pdf, reliability.get("disclaimer", RELIABILITY_DISCLAIMER))
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
    selected_sensors = manifest.get("selected_sensors")
    doctor = fusion_dataset_doctor(fusion_df, template_name, selected_sensors=selected_sensors)
    training_df = create_fusion_training_dataframe(fusion_df, template_name, selected_sensors=selected_sensors)
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
        zf.writestr("manifest.json", json.dumps(_json_safe(bundle_manifest), indent=2, ensure_ascii=False))
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
    readme = f"""EdgeTwin Studio V20 Auto Pilot Bundle

Project: {project_name}
Use case: {manifest.get('use_case_type')}
Template: {manifest.get('template')}
Samples: {manifest.get('samples')}
Selected sensors: {', '.join(manifest.get('selected_sensors', []))}
Recommended hardware: {hardware_result.get('recommendation')}
Reliability score: {reliability.get('reliability_score')}%

This bundle is intended for Edge AI pilot preparation.
{RELIABILITY_DISCLAIMER}
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
        zf.writestr("hardware_advice.json", json.dumps(_json_safe(hardware_result), indent=2, ensure_ascii=False))
        zf.writestr("manifest.json", json.dumps(_json_safe(full_manifest), indent=2, ensure_ascii=False))
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
    pdf.ln(4)
    safe_pdf_cell(pdf, "Validation Note", 8, True)
    safe_pdf_multicell(pdf, RELIABILITY_DISCLAIMER)

    metadata = {"project_name": project_name, "created_at": _now(), "audit": audit_result, "hardware": hardware_result or {}, "disclaimer": RELIABILITY_DISCLAIMER}
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("edge_dataset.csv", dataset_df.to_csv(index=False))
        zf.writestr("metadata.json", json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False))
        zf.writestr("audit_report.pdf", safe_pdf_output(pdf))
    return zip_buf.getvalue()


# ============================================================
# V20.1 TRUST / COMMERCIAL HARDENING
# ============================================================

def build_trust_gate(dataset_df, doctor=None, reliability=None, hardware_result=None, has_real_data=False, label_col="Label"):
    """Create a customer-facing trust gate: what this output is safe to claim, and what still needs validation."""
    red_flags = []
    next_steps = []

    if not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0:
        return {
            "status": "empty",
            "trust_level": "Not ready",
            "go_no_go": "No-go",
            "production_status": "No dataset available.",
            "data_quality_score": 0,
            "reliability_score": 0,
            "hardware_fit_score": 0,
            "commercial_package": "Demo only",
            "suggested_price_range": "Free / internal testing",
            "red_flags": ["No dataset is loaded."],
            "next_steps": ["Generate a demo, use the wizard, or upload a CSV with a Label column."],
            "disclaimer": RELIABILITY_DISCLAIMER,
        }

    df = dataset_df.copy()
    numeric_cols = _numeric_feature_columns(df, label_col)
    if label_col not in df.columns:
        red_flags.append("Dataset has no Label column.")
    if len(numeric_cols) == 0:
        red_flags.append("Dataset has no numeric feature columns.")

    if doctor is None or not isinstance(doctor, dict) or not doctor:
        if label_col in df.columns and numeric_cols:
            try:
                doctor = dataset_doctor(df[numeric_cols], df[label_col].astype(str))
            except Exception:
                doctor = {"overall_score": 0, "separation_score": 0, "balance_score": 0, "diversity_score": 0, "advice": []}
        else:
            doctor = {"overall_score": 0, "separation_score": 0, "balance_score": 0, "diversity_score": 0, "advice": []}

    if reliability is None or not isinstance(reliability, dict) or not reliability:
        reliability = calculate_reliability_score(doctor, has_real_data=has_real_data, selected_sensors=[])

    rows = int(len(df))
    class_count = 0
    min_class = 0
    weakest_class = None
    if label_col in df.columns:
        counts = df[label_col].astype(str).value_counts()
        class_count = int(len(counts))
        min_class = int(counts.min()) if len(counts) else 0
        weakest_class = str(counts.idxmin()) if len(counts) else None
        if class_count < 2:
            red_flags.append("Less than two classes are available.")
        if min_class < 20:
            red_flags.append(f"Weakest class has only {min_class} samples.")

    data_quality = int(np.clip(doctor.get("overall_score", 0), 0, 100))
    separation = int(np.clip(doctor.get("separation_score", 0), 0, 100))
    reliability_score = int(np.clip(reliability.get("reliability_score", 0), 0, 100))

    hardware_fit = 0
    hardware_name = "Unknown"
    if isinstance(hardware_result, dict) and hardware_result:
        hardware_name = hardware_result.get("recommendation", "Unknown")
        ranking = hardware_result.get("ranking", []) or []
        if ranking:
            try:
                top = float(ranking[0].get("adjusted_score", ranking[0].get("score", 0)))
                hardware_fit = int(np.clip(top, 0, 100))
            except Exception:
                hardware_fit = 65
        else:
            hardware_fit = 65

    if rows < 100:
        red_flags.append("Dataset is still small for a serious pilot.")
    if separation < 55:
        red_flags.append("Class separation is weak; labels may overlap.")
    if not has_real_data:
        red_flags.append("No real field data is linked yet.")

    # Customer-facing stage gates. Keep these honest: synthetic output can be billable, but not production proof.
    if data_quality >= 82 and reliability_score >= 78 and min_class >= 50 and class_count >= 2:
        trust_level = "Pilot-ready"
        go_no_go = "Go for pilot preparation"
    elif data_quality >= 68 and reliability_score >= 62 and class_count >= 2:
        trust_level = "Demo / prototype-ready"
        go_no_go = "Go for demo, improve before paid pilot delivery"
    elif class_count >= 2 and numeric_cols:
        trust_level = "Concept-ready"
        go_no_go = "Use internally, optimize before customer delivery"
    else:
        trust_level = "Not ready"
        go_no_go = "No-go"

    if has_real_data and reliability_score >= 82 and data_quality >= 80:
        production_status = "Field-validation ready, but not automatically production-approved."
    elif has_real_data:
        production_status = "Real data present, but more validation/tuning is recommended."
    else:
        production_status = "Pilot estimate only. Production deployment requires real field validation."

    if not has_real_data:
        next_steps.append("Upload or collect real WAV/CSV samples per class for Synthetic-to-Real validation.")
    if min_class and min_class < 50:
        next_steps.append(f"Add more samples to weakest class: {weakest_class}.")
    if separation < 70:
        next_steps.append("Improve label separation with stronger class definitions or optimizer variants.")
    if hardware_fit < 60:
        next_steps.append("Run Hardware Architect again with correct sample rate, feature count and power target.")
    if not next_steps:
        next_steps.append("Package as a paid pilot bundle and validate with customer field data.")

    if trust_level == "Pilot-ready" and has_real_data:
        package = "Real Data Pilot Bundle"
        price = "€499 - €999+"
    elif trust_level == "Pilot-ready":
        package = "Professional Pilot Bundle"
        price = "€149 - €499"
    elif trust_level == "Demo / prototype-ready":
        package = "Starter / Beta Pilot Bundle"
        price = "€49 - €199"
    elif trust_level == "Concept-ready":
        package = "Internal Demo / Free Lead Magnet"
        price = "Free - €49"
    else:
        package = "Not billable yet"
        price = "Fix before selling"

    confidence_gap = int(np.clip(100 - reliability_score, 0, 100))

    return {
        "status": "ok",
        "trust_level": trust_level,
        "go_no_go": go_no_go,
        "production_status": production_status,
        "data_quality_score": data_quality,
        "separation_score": separation,
        "reliability_score": reliability_score,
        "hardware_fit_score": hardware_fit,
        "hardware_recommendation": hardware_name,
        "rows": rows,
        "class_count": class_count,
        "min_class_samples": min_class,
        "weakest_class": weakest_class,
        "has_real_data": bool(has_real_data),
        "confidence_gap": confidence_gap,
        "commercial_package": package,
        "suggested_price_range": price,
        "red_flags": red_flags,
        "next_steps": next_steps,
        "disclaimer": RELIABILITY_DISCLAIMER,
    }


def create_trust_bundle(project_name, dataset_df, trust_gate):
    """Export a small customer-trust bundle with a trust snapshot PDF and JSON."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Trust & Readiness Snapshot", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)

    safe_pdf_cell(pdf, "Decision Gate", 8, True)
    safe_pdf_cell(pdf, f"Trust level: {trust_gate.get('trust_level', 'Unknown')}")
    safe_pdf_cell(pdf, f"Go / No-Go: {trust_gate.get('go_no_go', 'Unknown')}")
    safe_pdf_multicell(pdf, f"Production status: {trust_gate.get('production_status', '')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Scores", 8, True)
    safe_pdf_cell(pdf, f"Data quality: {trust_gate.get('data_quality_score', 0)}%")
    safe_pdf_cell(pdf, f"Label separation: {trust_gate.get('separation_score', 0)}%")
    safe_pdf_cell(pdf, f"Reliability estimate: {trust_gate.get('reliability_score', 0)}%")
    safe_pdf_cell(pdf, f"Hardware fit: {trust_gate.get('hardware_fit_score', 0)}%")
    safe_pdf_cell(pdf, f"Recommended hardware: {trust_gate.get('hardware_recommendation', 'Unknown')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Commercial Packaging", 8, True)
    safe_pdf_cell(pdf, f"Suggested package: {trust_gate.get('commercial_package', 'Unknown')}")
    safe_pdf_cell(pdf, f"Suggested price range: {trust_gate.get('suggested_price_range', 'Unknown')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Red Flags", 8, True)
    for item in trust_gate.get("red_flags", []) or ["No major red flags detected."]:
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Next Steps", 8, True)
    for item in trust_gate.get("next_steps", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Validation Note", 8, True)
    safe_pdf_multicell(pdf, trust_gate.get("disclaimer", RELIABILITY_DISCLAIMER))

    zip_buf = io.BytesIO()
    metadata = {
        "project_name": project_name,
        "created_at": _now(),
        "trust_gate": trust_gate,
    }
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("trust_snapshot.pdf", safe_pdf_output(pdf))
        zf.writestr("trust_gate.json", json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr("current_dataset.csv", dataset_df.to_csv(index=False))
        zf.writestr("README.txt", f"""EdgeTwin Studio V20.1 Trust Bundle

Project: {project_name}
Trust level: {trust_gate.get('trust_level')}
Go/No-Go: {trust_gate.get('go_no_go')}
Suggested package: {trust_gate.get('commercial_package')}
Suggested price range: {trust_gate.get('suggested_price_range')}

{trust_gate.get('disclaimer', RELIABILITY_DISCLAIMER)}
""")
    return zip_buf.getvalue()


# ============================================================
# V21 SYNTHETIC-TO-REAL BRIDGE
# ============================================================

BRIDGE_DISCLAIMER = (
    "Synthetic-to-Real Bridge compares real WAV/CSV signal fingerprints with generated variants. "
    "A high similarity score supports pilot confidence, but production deployment still requires field validation."
)


def label_from_filename(filename):
    """Create a practical label hint from a filename for quick upload flows."""
    name = os.path.basename(str(filename or "sample")).rsplit(".", 1)[0]
    clean = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
    parts = [p for p in clean.split("_") if p]
    # If a filename starts with a class prefix like BearingWear_001.wav, keep the first meaningful block.
    if len(parts) >= 2 and any(ch.isdigit() for ch in parts[-1]):
        parts = parts[:-1]
    label = "_".join(parts[:3]) if parts else "Real_Sample"
    return label or "Real_Sample"


def _read_signal_from_bytes(file_bytes, filename, sr_hint=16000):
    """Read a one-dimensional signal from WAV or CSV bytes."""
    filename = str(filename or "sample")
    lower = filename.lower()
    if lower.endswith(".wav"):
        sr, wav_data = wavfile.read(io.BytesIO(file_bytes))
        sig = wav_data.mean(axis=1) if getattr(wav_data, "ndim", 1) > 1 else wav_data
        sig = np.asarray(sig, dtype=float)
        # Normalize common integer PCM so features stay comparable.
        if np.max(np.abs(sig)) > 1.5:
            sig = sig / max(np.max(np.abs(sig)), 1.0)
        return sig, int(sr)

    if lower.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            # Try coercing text columns if the CSV was parsed as object.
            coerced = df.apply(pd.to_numeric, errors="coerce")
            numeric = coerced.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        if numeric.shape[1] == 0:
            raise ValueError("CSV must contain at least one numeric signal column.")
        if numeric.shape[1] >= 2:
            # Common format: time,value. Use second numeric column as signal.
            sig = numeric.iloc[:, 1].dropna().values
        else:
            sig = numeric.iloc[:, 0].dropna().values
        return np.asarray(sig, dtype=float), int(sr_hint or 16000)

    raise ValueError("Unsupported file type. Upload WAV or CSV.")


def _band_energy(f, v, low, high):
    mask = (f >= low) & (f < high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(v[mask] ** 2))


def extract_real_signal_fingerprint(file_bytes, filename, label="Real_Sample", sr_hint=16000, max_samples=120000):
    """Extract a compact OMEGA-X Signal Fingerprint from a real WAV/CSV file."""
    try:
        sig, sr = _read_signal_from_bytes(file_bytes, filename, sr_hint=sr_hint)
        sig = np.asarray(sig, dtype=float)
        sig = sig[np.isfinite(sig)]
        if len(sig) < 16:
            raise ValueError("Signal is too short for fingerprint extraction.")

        # Keep the app responsive on large uploads while preserving the broad signal shape.
        if len(sig) > max_samples:
            step = int(np.ceil(len(sig) / max_samples))
            sig = sig[::step]

        sig = sig - np.mean(sig)
        duration = float(len(sig) / max(sr, 1))
        f, v = calculate_fft(sig, sr)
        base_features = extract_signal_features(sig, sr, label=None)

        if len(v) > 1:
            dom_idx = int(np.argmax(v[1:]) + 1)
        else:
            dom_idx = 0
        dominant_freq = float(f[dom_idx]) if len(f) else 0.0
        peak_mag = float(v[dom_idx]) if len(v) else 0.0
        total_energy = float(np.sum(v ** 2) + 1e-12)

        nyq = sr / 2.0
        low_energy = _band_energy(f, v, 0, min(250, nyq))
        mid_energy = _band_energy(f, v, min(250, nyq), min(2000, nyq))
        high_energy = _band_energy(f, v, min(2000, nyq), nyq + 1)
        band_ratio = float((mid_energy + high_energy) / max(low_energy, 1e-12))
        noise_floor = float(np.median(v)) if len(v) else 0.0

        abs_sig = np.abs(sig)
        threshold = float(np.mean(abs_sig) + 2.5 * np.std(abs_sig)) if len(abs_sig) else 0.0
        impulse_count = int(np.sum(abs_sig > threshold)) if threshold > 0 else 0
        impulse_density = float(impulse_count / max(duration, 1e-6))

        harmonic_energy = 0.0
        fundamental_energy = peak_mag ** 2
        if dominant_freq > 0 and len(f) > 0:
            for h in [2, 3, 4, 5]:
                target = dominant_freq * h
                if target < nyq:
                    idx = int(np.argmin(np.abs(f - target)))
                    harmonic_energy += float(v[idx] ** 2)
        harmonic_ratio = float(harmonic_energy / max(fundamental_energy, 1e-12))

        out = {
            "Label": str(label or "Real_Sample").replace(" ", "_"),
            "Filename": filename,
            "Source": "Real_Field_Data",
            "SampleRate": int(sr),
            "DurationSec": duration,
            "Samples": int(len(sig)),
            **base_features,
            "DominantFreq": dominant_freq,
            "PeakMagnitude": peak_mag,
            "LowBandEnergy": float(low_energy / total_energy),
            "MidBandEnergy": float(mid_energy / total_energy),
            "HighBandEnergy": float(high_energy / total_energy),
            "BandEnergyRatio": band_ratio,
            "NoiseFloor": noise_floor,
            "ImpulseDensity": impulse_density,
            "HarmonicRatio": harmonic_ratio,
            "FingerprintVersion": "OMEGA-X-FP-v1",
            "error": "",
        }
        return out
    except Exception as exc:
        return {
            "Label": str(label or "Real_Sample").replace(" ", "_"),
            "Filename": filename,
            "Source": "Real_Field_Data",
            "error": str(exc),
        }


def build_real_fingerprint_dataframe(file_specs, sr_hint=16000):
    """Build a dataframe of OMEGA-X fingerprints from uploaded file specs."""
    rows = []
    for spec in file_specs or []:
        filename = spec.get("filename", "sample")
        label = spec.get("label") or label_from_filename(filename)
        file_bytes = spec.get("bytes", b"")
        rows.append(extract_real_signal_fingerprint(file_bytes, filename, label=label, sr_hint=sr_hint))
    return pd.DataFrame(rows)


def _bridge_numeric_features(df):
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return []
    excluded = {"SampleRate", "Samples", "DurationSec"}
    return [
        c for c in df.columns
        if c not in excluded
        and c not in {"Label", "Filename", "Source", "VariantOf", "FingerprintVersion", "error"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def generate_variants_from_real_fingerprints(fingerprint_df, variants_per_file=25, jitter_strength=0.08, random_state=42):
    """Generate feature-level synthetic variants around real signal fingerprints."""
    if not isinstance(fingerprint_df, pd.DataFrame) or len(fingerprint_df) == 0:
        return pd.DataFrame()
    valid = fingerprint_df.copy()
    if "error" in valid.columns:
        valid = valid[valid["error"].fillna("") == ""].copy()
    if len(valid) == 0:
        return pd.DataFrame()

    numeric_cols = _bridge_numeric_features(valid)
    rng = np.random.default_rng(random_state)
    rows = []
    variants_per_file = int(max(1, variants_per_file))

    class_stds = {}
    for label, group in valid.groupby("Label"):
        class_stds[str(label)] = group[numeric_cols].std(numeric_only=True).replace(0, np.nan).fillna(0).to_dict()

    for _, row in valid.iterrows():
        label = str(row.get("Label", "Real_Sample"))
        for i in range(variants_per_file):
            new = row.to_dict()
            new["Source"] = "Real_Fingerprint_Variant"
            new["VariantOf"] = row.get("Filename", "unknown")
            new["Filename"] = f"variant_{i+1:03d}_of_{row.get('Filename', 'sample')}"
            new["error"] = ""
            for col in numeric_cols:
                val = float(row.get(col, 0) or 0)
                cls_std = float(class_stds.get(label, {}).get(col, 0) or 0)
                scale = max(abs(val) * float(jitter_strength), cls_std * 0.35, 1e-6)
                updated = val + rng.normal(0, scale)
                if col not in {"ZCR", "SpectralFlatness"}:
                    updated = max(0.0, updated) if val >= 0 else updated
                else:
                    updated = float(np.clip(updated, 0, 1))
                new[col] = float(updated)
            rows.append(new)
    return pd.DataFrame(rows)


def calculate_synthetic_real_similarity(real_df, synthetic_df, label_col="Label"):
    """Score how close synthetic/variant feature distributions are to real fingerprints."""
    if not isinstance(real_df, pd.DataFrame) or len(real_df) == 0:
        return {
            "status": "no_real_data",
            "similarity_score": 0,
            "verdict": "No real data available for comparison.",
            "weak_labels": [],
        }
    if not isinstance(synthetic_df, pd.DataFrame) or len(synthetic_df) == 0:
        return {
            "status": "no_synthetic_data",
            "similarity_score": 0,
            "verdict": "No synthetic/variant data available for comparison.",
            "weak_labels": [],
        }

    real_valid = real_df.copy()
    if "error" in real_valid.columns:
        real_valid = real_valid[real_valid["error"].fillna("") == ""].copy()
    if len(real_valid) == 0:
        return {
            "status": "invalid_real_data",
            "similarity_score": 0,
            "verdict": "Uploaded files could not be fingerprinted.",
            "weak_labels": [],
        }

    # Compare true signal features only; metadata such as SampleRate/Samples/DurationSec should not inflate the score.
    real_num = set(_bridge_numeric_features(real_valid))
    synth_num = set(_bridge_numeric_features(synthetic_df))
    common = sorted([c for c in real_num.intersection(synth_num) if c in real_valid.columns and c in synthetic_df.columns])
    if not common:
        return {
            "status": "no_common_features",
            "similarity_score": 0,
            "verdict": "Real and synthetic datasets do not share comparable numeric features.",
            "weak_labels": [],
        }

    r = real_valid[common].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    s = synthetic_df[common].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    r_mean, s_mean = r.mean(), s.mean()
    r_std, s_std = r.std().replace(0, np.nan).fillna(0), s.std().replace(0, np.nan).fillna(0)
    scale = (r_std + s_std + (r_mean.abs() * 0.05) + 1e-6)
    mean_dist = float(np.mean(np.abs(r_mean - s_mean) / scale))
    std_dist = float(np.mean(np.abs(r_std - s_std) / (scale + 1e-6)))
    raw = np.exp(-0.55 * mean_dist - 0.30 * std_dist)

    real_labels = set(real_valid[label_col].astype(str)) if label_col in real_valid.columns else set()
    synth_labels = set(synthetic_df[label_col].astype(str)) if label_col in synthetic_df.columns else set()
    coverage = len(real_labels.intersection(synth_labels)) / max(len(real_labels), 1) if real_labels else 1.0
    score = int(np.clip(raw * 100 * (0.75 + 0.25 * coverage), 0, 100))

    counts = real_valid[label_col].astype(str).value_counts() if label_col in real_valid.columns else pd.Series(dtype=int)
    weak_labels = [str(k) for k, v in counts.items() if int(v) < 3]

    if score >= 82:
        verdict = "Strong synthetic-to-real alignment for pilot preparation. Continue with field validation."
        risk = "Low-Medium"
    elif score >= 65:
        verdict = "Usable alignment for prototype/pilot work. Add more real samples per class."
        risk = "Medium"
    else:
        verdict = "Weak alignment. Synthetic variants need more real examples or better class definitions."
        risk = "High"

    return {
        "status": "ok",
        "similarity_score": score,
        "mean_distribution_gap": round(mean_dist, 4),
        "std_distribution_gap": round(std_dist, 4),
        "class_coverage": round(float(coverage), 3),
        "common_features": common,
        "weak_labels": weak_labels,
        "dataset_risk": risk,
        "verdict": verdict,
        "disclaimer": BRIDGE_DISCLAIMER,
    }


def estimate_real_samples_needed(real_df, similarity, label_col="Label"):
    """Estimate remaining real samples needed per class for a stronger paid pilot."""
    if not isinstance(real_df, pd.DataFrame) or len(real_df) == 0 or label_col not in real_df.columns:
        return {"target_per_class": 20, "needed_by_class": {}, "total_needed": 0}
    sim = int(similarity.get("similarity_score", 0) if isinstance(similarity, dict) else 0)
    if sim >= 82:
        target = 10
    elif sim >= 65:
        target = 20
    else:
        target = 35
    counts = real_df[label_col].astype(str).value_counts().to_dict()
    needed = {str(label): int(max(0, target - int(count))) for label, count in counts.items()}
    return {
        "target_per_class": int(target),
        "current_by_class": {str(k): int(v) for k, v in counts.items()},
        "needed_by_class": needed,
        "total_needed": int(sum(needed.values())),
    }


def run_synthetic_to_real_bridge(
    file_specs,
    existing_synthetic_df=None,
    variants_per_file=25,
    jitter_strength=0.08,
    sr_hint=16000,
    label_col="Label",
):
    """Complete V21 bridge flow: real fingerprints -> variants -> similarity -> pilot training dataframe."""
    fingerprint_df = build_real_fingerprint_dataframe(file_specs, sr_hint=sr_hint)
    valid_fingerprints = fingerprint_df.copy()
    if "error" in valid_fingerprints.columns:
        valid_fingerprints = valid_fingerprints[valid_fingerprints["error"].fillna("") == ""].copy()

    variant_df = generate_variants_from_real_fingerprints(
        valid_fingerprints,
        variants_per_file=variants_per_file,
        jitter_strength=jitter_strength,
    )

    compare_target = variant_df
    if isinstance(existing_synthetic_df, pd.DataFrame) and len(existing_synthetic_df) > 0:
        # Compare uploaded real fingerprints against the current project dataset as well as generated variants when possible.
        compare_target = pd.concat([existing_synthetic_df.copy(), variant_df.copy()], ignore_index=True, sort=False)

    similarity = calculate_synthetic_real_similarity(valid_fingerprints, compare_target, label_col=label_col)
    sample_plan = estimate_real_samples_needed(valid_fingerprints, similarity, label_col=label_col)

    bridge_training_df = pd.concat([valid_fingerprints, variant_df], ignore_index=True, sort=False)
    if len(bridge_training_df) > 0:
        bridge_training_df = bridge_training_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    numeric_cols = _numeric_feature_columns(bridge_training_df, label_col) if len(bridge_training_df) else []
    if len(bridge_training_df) > 0 and label_col in bridge_training_df.columns and numeric_cols:
        doctor = dataset_doctor(bridge_training_df[numeric_cols], bridge_training_df[label_col].astype(str))
    else:
        doctor = {"overall_score": 0, "advice": [{"severity": "high", "message": "No valid bridge training dataset created."}]}

    reliability = calculate_reliability_score(doctor, has_real_data=True, selected_sensors=["Audio", "Vibration"])
    # Blend reliability with actual synthetic-real similarity so the score stays honest.
    if similarity.get("status") == "ok":
        blended = int(np.clip((reliability.get("reliability_score", 0) * 0.55) + (similarity.get("similarity_score", 0) * 0.45), 0, 100))
        reliability["reliability_score"] = blended
        reliability["synthetic_real_similarity_score"] = similarity.get("similarity_score", 0)
        reliability["dataset_risk"] = similarity.get("dataset_risk", reliability.get("dataset_risk", "Medium"))
        reliability["verdict"] = similarity.get("verdict", reliability.get("verdict", ""))
        reliability["disclaimer"] = BRIDGE_DISCLAIMER

    result = {
        "status": "ok" if len(valid_fingerprints) > 0 else "no_valid_files",
        "fingerprint_df": fingerprint_df,
        "valid_fingerprint_df": valid_fingerprints,
        "variant_df": variant_df,
        "bridge_training_df": bridge_training_df,
        "similarity": similarity,
        "sample_plan": sample_plan,
        "doctor": doctor,
        "reliability": reliability,
        "created_at": _now(),
        "summary": {
            "real_files": int(len(valid_fingerprints)),
            "generated_variants": int(len(variant_df)),
            "classes": int(valid_fingerprints[label_col].nunique()) if label_col in valid_fingerprints.columns and len(valid_fingerprints) else 0,
            "similarity_score": int(similarity.get("similarity_score", 0)),
            "reliability_score": int(reliability.get("reliability_score", 0)),
            "dataset_risk": similarity.get("dataset_risk", "Unknown"),
            "recommended_real_samples_needed": int(sample_plan.get("total_needed", 0)),
        },
        "disclaimer": BRIDGE_DISCLAIMER,
    }
    return result


def compact_bridge_summary(result):
    """Small JSON-safe summary for project settings."""
    if not isinstance(result, dict):
        return {}
    return _json_safe({
        "status": result.get("status"),
        "created_at": result.get("created_at"),
        "summary": result.get("summary", {}),
        "similarity": result.get("similarity", {}),
        "sample_plan": result.get("sample_plan", {}),
        "reliability": result.get("reliability", {}),
    })


def create_synthetic_to_real_bridge_bundle(project_name, bridge_result):
    """Export the V21 bridge package."""
    bridge_result = bridge_result or {}
    fingerprint_df = bridge_result.get("fingerprint_df", pd.DataFrame())
    valid_df = bridge_result.get("valid_fingerprint_df", pd.DataFrame())
    variant_df = bridge_result.get("variant_df", pd.DataFrame())
    training_df = bridge_result.get("bridge_training_df", pd.DataFrame())
    similarity = bridge_result.get("similarity", {})
    sample_plan = bridge_result.get("sample_plan", {})
    reliability = bridge_result.get("reliability", {})
    doctor = bridge_result.get("doctor", {})
    summary = bridge_result.get("summary", {})
    reliability_v2 = bridge_result.get("reliability_v2", {})

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Synthetic-to-Real Bridge", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)

    safe_pdf_cell(pdf, "Bridge Summary", 8, True)
    safe_pdf_cell(pdf, f"Real files analyzed: {summary.get('real_files', len(valid_df) if isinstance(valid_df, pd.DataFrame) else 0)}")
    safe_pdf_cell(pdf, f"Generated variants: {summary.get('generated_variants', len(variant_df) if isinstance(variant_df, pd.DataFrame) else 0)}")
    safe_pdf_cell(pdf, f"Synthetic-to-real similarity: {similarity.get('similarity_score', 0)}%")
    safe_pdf_cell(pdf, f"Reliability estimate: {reliability.get('reliability_score', 0)}%")
    safe_pdf_cell(pdf, f"Dataset risk: {similarity.get('dataset_risk', 'Unknown')}")
    safe_pdf_multicell(pdf, similarity.get("verdict", ""))
    if reliability_v2:
        pdf.ln(4)
        safe_pdf_cell(pdf, "Reliability Engine 2.0", 8, True)
        safe_pdf_cell(pdf, f"Trust Score V2: {reliability_v2.get('trust_score_v2', 0)}%")
        safe_pdf_cell(pdf, f"Readiness stage: {reliability_v2.get('readiness_stage', 'Unknown')}")
        safe_pdf_cell(pdf, f"Production risk: {reliability_v2.get('production_risk_level', 'Unknown')}")
        safe_pdf_multicell(pdf, reliability_v2.get("decision", ""))
    pdf.ln(4)

    safe_pdf_cell(pdf, "Real Samples Needed", 8, True)
    safe_pdf_cell(pdf, f"Target per class: {sample_plan.get('target_per_class', 0)}")
    safe_pdf_cell(pdf, f"Total additional recommended samples: {sample_plan.get('total_needed', 0)}")
    for label, needed in (sample_plan.get("needed_by_class", {}) or {}).items():
        safe_pdf_cell(pdf, f"- {label}: {needed}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Dataset Doctor", 8, True)
    safe_pdf_cell(pdf, f"Overall score: {doctor.get('overall_score', 0)}%")
    for item in doctor.get("advice", []):
        safe_pdf_multicell(pdf, f"[{item.get('severity', 'info').upper()}] {item.get('message', '')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Validation Note", 8, True)
    safe_pdf_multicell(pdf, BRIDGE_DISCLAIMER)

    metadata = compact_bridge_summary(bridge_result)
    if reliability_v2:
        metadata["reliability_v2"] = reliability_v2
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("synthetic_to_real_report.pdf", safe_pdf_output(pdf))
        if isinstance(fingerprint_df, pd.DataFrame):
            zf.writestr("real_signal_fingerprints.csv", fingerprint_df.to_csv(index=False))
        if isinstance(variant_df, pd.DataFrame):
            zf.writestr("real_based_synthetic_variants.csv", variant_df.to_csv(index=False))
        if isinstance(training_df, pd.DataFrame):
            zf.writestr("bridge_training_dataset.csv", training_df.to_csv(index=False))
        zf.writestr("bridge_manifest.json", json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False))
        if reliability_v2:
            zf.writestr("reliability_engine_v2.json", json.dumps(_json_safe(reliability_v2), indent=2, ensure_ascii=False))
        zf.writestr("README.txt", f"""EdgeTwin Studio V21.1 Synthetic-to-Real Bridge Bundle

Project: {project_name}
Real files analyzed: {summary.get('real_files', 0)}
Generated variants: {summary.get('generated_variants', 0)}
Similarity score: {similarity.get('similarity_score', 0)}%
Reliability estimate: {reliability.get('reliability_score', 0)}%
Dataset risk: {similarity.get('dataset_risk', 'Unknown')}

{BRIDGE_DISCLAIMER}
""")
    return zip_buf.getvalue()


# ============================================================
# V21.1 RELIABILITY ENGINE 2.0
# ============================================================

def _risk_level_from_score(score):
    score = float(np.clip(score, 0, 100))
    if score >= 75:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def _sensor_key_from_name(name):
    text = str(name).lower()
    if "audio" in text or "sound" in text or "acoustic" in text or text in {"rms", "zcr"} or "spectral" in text or "rolloff" in text or "flatness" in text:
        return "Audio"
    if "vibration" in text or "imu" in text or "accel" in text or "kurtosis" in text or "crest" in text or "std" == text:
        return "Vibration"
    if "gas" in text or "temp" in text or "humidity" in text or "env" in text:
        return "Gas / Environment"
    if "radar" in text or "presence" in text or "distance" in text:
        return "Radar"
    if "gps" in text or "zone" in text or "location" in text:
        return "GPS / Zone"
    if "fusion" in text or "health" in text or "confidence" in text:
        return "Fusion / Context"
    return "Signal Features"


def calculate_class_risk_report(dataset_df, label_col="Label", has_real_data=False, target_per_class=None, max_rows=1800):
    """Per-label risk table: sample weakness + separation weakness -> real sample plan."""
    if not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0 or label_col not in dataset_df.columns:
        return []

    df = dataset_df.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=[label_col])
    numeric_cols = _numeric_feature_columns(df, label_col)
    if not numeric_cols:
        counts = df[label_col].astype(str).value_counts()
        target = int(target_per_class or (25 if has_real_data else 50))
        return [
            {
                "label": str(label),
                "samples": int(count),
                "sample_score": int(np.clip((count / max(target, 1)) * 100, 0, 100)),
                "separation_score": 0,
                "risk_score": int(np.clip(100 - ((count / max(target, 1)) * 60), 0, 100)),
                "risk_level": _risk_level_from_score(100 - ((count / max(target, 1)) * 60)),
                "recommended_real_samples_needed": int(max(0, target - int(count))),
                "main_issue": "No numeric features available for separation analysis.",
            }
            for label, count in counts.items()
        ]

    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)

    y = df[label_col].astype(str)
    X = df[numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    counts_full = dataset_df[label_col].astype(str).value_counts()
    target = int(target_per_class or (20 if has_real_data else 50))
    try:
        Xs = StandardScaler().fit_transform(X)
    except Exception:
        Xs = X.values.astype(float)

    labels = list(pd.Series(y).dropna().unique())
    centroids = {}
    intra = {}
    for label in labels:
        mask = (y.values == label)
        if mask.sum() <= 0:
            continue
        points = Xs[mask]
        c = points.mean(axis=0)
        centroids[label] = c
        intra[label] = float(np.mean(np.linalg.norm(points - c, axis=1))) if len(points) else 0.0

    rows = []
    for label in labels:
        count = int(counts_full.get(label, 0))
        sample_score = int(np.clip((count / max(target, 1)) * 100, 0, 100))
        other = [k for k in centroids if k != label]
        if other:
            nearest = min(float(np.linalg.norm(centroids[label] - centroids[o])) for o in other)
            sep_ratio = nearest / max(intra.get(label, 0.0), 1e-6)
            separation_score = int(np.clip((sep_ratio / 2.0) * 100, 0, 100))
        else:
            separation_score = 0
        risk_score = int(np.clip(100 - (sample_score * 0.45 + separation_score * 0.55), 0, 100))
        if count < max(5, target * 0.25):
            issue = "Too few real/synthetic examples for this class."
        elif separation_score < 45:
            issue = "This class overlaps strongly with another class."
        elif sample_score < 70:
            issue = "Class has usable structure but still needs more examples."
        else:
            issue = "Class looks usable for pilot preparation."
        rows.append({
            "label": str(label),
            "samples": count,
            "sample_score": sample_score,
            "separation_score": separation_score,
            "risk_score": risk_score,
            "risk_level": _risk_level_from_score(risk_score),
            "recommended_real_samples_needed": int(max(0, target - count)),
            "main_issue": issue,
        })
    return sorted(rows, key=lambda r: (r["risk_score"], -r["samples"]), reverse=True)


def calculate_sensor_value_scores(dataset_df, selected_sensors=None, label_col="Label"):
    """Estimate which sensors/features are actually contributing signal in the current dataset."""
    selected_sensors = selected_sensors or []
    if not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0:
        return []
    numeric_cols = _numeric_feature_columns(dataset_df, label_col)
    if not numeric_cols:
        return []

    df = dataset_df.copy().replace([np.inf, -np.inf], np.nan)
    groups = {}
    for col in numeric_cols:
        sensor = _sensor_key_from_name(col)
        groups.setdefault(sensor, []).append(col)

    # Ensure selected but missing sensors are visible as low value instead of silently ignored.
    for sensor in selected_sensors:
        sensor_name = str(sensor)
        if sensor_name not in groups and sensor_name in ["Audio", "Vibration", "Radar", "Gas / Environment", "GPS / Zone"]:
            groups.setdefault(sensor_name, [])

    label_strength = {}
    if label_col in df.columns and df[label_col].nunique() >= 2:
        y_codes = pd.Categorical(df[label_col].astype(str)).codes.astype(float)
        for col in numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values.astype(float)
            if np.std(vals) < 1e-9 or np.std(y_codes) < 1e-9:
                label_strength[col] = 0.0
            else:
                try:
                    label_strength[col] = abs(float(np.corrcoef(vals, y_codes)[0, 1]))
                except Exception:
                    label_strength[col] = 0.0
    else:
        label_strength = {c: 0.0 for c in numeric_cols}

    rows = []
    for sensor, cols in groups.items():
        if not cols:
            rows.append({
                "sensor": sensor,
                "value_score": 0,
                "feature_count": 0,
                "risk": "Missing / not used",
                "reason": "No active numeric feature columns found for this sensor.",
            })
            continue
        variances = []
        missing_rates = []
        strengths = []
        for col in cols:
            vals = pd.to_numeric(df[col], errors="coerce")
            missing_rates.append(float(vals.isna().mean()))
            vals = vals.fillna(0)
            variances.append(float(np.nanstd(vals)))
            strengths.append(float(label_strength.get(col, 0.0)))
        variance_score = 100 if np.mean(variances) > 1e-6 else 20
        strength_score = int(np.clip(np.mean(strengths) * 160, 0, 100)) if strengths else 0
        missing_score = int(np.clip(100 - np.mean(missing_rates) * 100, 0, 100))
        value_score = int(np.clip(strength_score * 0.60 + variance_score * 0.20 + missing_score * 0.20, 0, 100))
        if value_score >= 70:
            risk = "High value"
            reason = "This sensor/feature group appears useful for separating labels."
        elif value_score >= 40:
            risk = "Medium value"
            reason = "This sensor/feature group may help, but needs validation or better features."
        else:
            risk = "Low value"
            reason = "This sensor/feature group adds weak separation in the current dataset."
        rows.append({
            "sensor": sensor,
            "value_score": value_score,
            "feature_count": int(len(cols)),
            "risk": risk,
            "reason": reason,
        })
    return sorted(rows, key=lambda r: r["value_score"], reverse=True)


def build_reliability_engine_v2(
    dataset_df,
    doctor=None,
    reliability=None,
    hardware_result=None,
    bridge_result=None,
    selected_sensors=None,
    has_real_data=False,
    label_col="Label",
):
    """V21.1: honest commercial readiness engine with class risk, sensor value and production-risk language."""
    selected_sensors = selected_sensors or []
    df = dataset_df.copy() if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    numeric_cols = _numeric_feature_columns(df, label_col) if len(df) else []

    if doctor is None or not isinstance(doctor, dict) or not doctor:
        if len(df) and label_col in df.columns and numeric_cols:
            try:
                doctor = dataset_doctor(df[numeric_cols], df[label_col].astype(str))
            except Exception:
                doctor = {"overall_score": 0, "separation_score": 0, "balance_score": 0, "diversity_score": 0, "advice": []}
        else:
            doctor = {"overall_score": 0, "separation_score": 0, "balance_score": 0, "diversity_score": 0, "advice": []}

    bridge_result = bridge_result if isinstance(bridge_result, dict) else {}
    if reliability is None or not isinstance(reliability, dict) or not reliability:
        reliability = calculate_reliability_score(doctor, has_real_data=has_real_data, selected_sensors=selected_sensors)

    similarity = bridge_result.get("similarity", {}) if isinstance(bridge_result, dict) else {}
    sample_plan = bridge_result.get("sample_plan", {}) if isinstance(bridge_result, dict) else {}
    similarity_score = int(np.clip(similarity.get("similarity_score", reliability.get("synthetic_real_similarity_score", 0)), 0, 100))
    target_per_class = sample_plan.get("target_per_class") if isinstance(sample_plan, dict) else None

    class_risks = calculate_class_risk_report(df, label_col=label_col, has_real_data=has_real_data, target_per_class=target_per_class)
    sensor_scores = calculate_sensor_value_scores(df, selected_sensors=selected_sensors, label_col=label_col)

    data_quality = int(np.clip(doctor.get("overall_score", 0), 0, 100))
    separation = int(np.clip(doctor.get("separation_score", 0), 0, 100))
    base_reliability = int(np.clip(reliability.get("reliability_score", 0), 0, 100))
    worst_class_risk = max([r.get("risk_score", 100) for r in class_risks], default=100)
    avg_sensor_value = int(np.mean([r.get("value_score", 0) for r in sensor_scores])) if sensor_scores else 0

    hardware_fit = 0
    hardware_recommendation = "Unknown"
    if isinstance(hardware_result, dict) and hardware_result:
        hardware_recommendation = hardware_result.get("recommendation", "Unknown")
        ranking = hardware_result.get("ranking", []) or []
        if ranking:
            try:
                hardware_fit = int(np.clip(float(ranking[0].get("adjusted_score", ranking[0].get("score", 0))), 0, 100))
            except Exception:
                hardware_fit = 65
        else:
            hardware_fit = 65

    real_data_factor = 100 if has_real_data else 45
    similarity_component = similarity_score if has_real_data and similarity_score > 0 else 45
    trust_score = int(np.clip(
        data_quality * 0.25 +
        separation * 0.17 +
        base_reliability * 0.20 +
        similarity_component * 0.15 +
        avg_sensor_value * 0.10 +
        hardware_fit * 0.08 +
        real_data_factor * 0.05 -
        max(0, worst_class_risk - 65) * 0.12,
        0,
        100,
    ))

    total_needed = int(sum([r.get("recommended_real_samples_needed", 0) for r in class_risks]))
    if isinstance(sample_plan, dict) and sample_plan.get("total_needed") is not None:
        total_needed = max(total_needed, int(sample_plan.get("total_needed", 0)))

    red_flags = []
    if len(df) < 100:
        red_flags.append("Dataset is small; keep as concept/demo until more samples are available.")
    if not has_real_data:
        red_flags.append("No real field data linked; do not claim production reliability.")
    if separation < 55:
        red_flags.append("Label separation is weak; some classes may overlap.")
    if worst_class_risk >= 75:
        red_flags.append("At least one class is high-risk and needs more samples or clearer definition.")
    if avg_sensor_value < 35 and sensor_scores:
        red_flags.append("Sensor value is weak; current sensors/features may not explain the labels well.")
    if hardware_fit and hardware_fit < 55:
        red_flags.append("Hardware fit is weak; review board, sample rate and feature count.")

    if has_real_data and trust_score >= 84 and worst_class_risk < 55 and similarity_score >= 78:
        readiness_stage = "Pilot-ready with real-data validation"
        production_risk = "Medium"
        go_no_go = "GO"
        decision = "Good for a paid real-data pilot bundle. Still require field validation before production deployment."
    elif trust_score >= 72 and worst_class_risk < 70:
        readiness_stage = "Pilot-prep ready"
        production_risk = "Medium-High" if not has_real_data else "Medium"
        go_no_go = "CONDITIONAL"
        decision = "Good for a paid pilot-preparation bundle, but add real samples and fix weak classes before stronger claims."
    elif trust_score >= 55:
        readiness_stage = "Demo / prototype-ready"
        production_risk = "High"
        go_no_go = "CONDITIONAL"
        decision = "Useful for demos and early customer discovery. Do not sell as a validated pilot yet."
    else:
        readiness_stage = "Concept-only"
        production_risk = "Very High"
        go_no_go = "NO-GO"
        decision = "Not billable as a serious pilot yet. Improve data quality, class separation and real-data coverage first."

    allowed_claims = [
        "Automatically generates an Edge AI pilot-preparation package.",
        "Provides dataset audit, class-risk analysis and hardware direction.",
        "Estimates readiness and highlights what still needs field validation.",
    ]
    if has_real_data:
        allowed_claims.append("Uses uploaded real WAV/CSV files to create signal fingerprints and real-based variants.")
    blocked_claims = [
        "Do not claim production-ready without customer field validation.",
        "Do not claim guaranteed model accuracy from synthetic data alone.",
        "Do not claim hardware certification or safety approval.",
    ]

    next_steps = []
    high_risk_labels = [r["label"] for r in class_risks if r.get("risk_level") == "High"]
    if high_risk_labels:
        next_steps.append("Collect or generate stronger examples for high-risk labels: " + ", ".join(high_risk_labels[:5]) + ".")
    if total_needed > 0:
        next_steps.append(f"Collect approximately {total_needed} additional real samples across weak classes.")
    if avg_sensor_value < 50 and sensor_scores:
        next_steps.append("Review sensor selection and feature extraction; weak sensors should be removed or improved.")
    if similarity_score and similarity_score < 70:
        next_steps.append("Improve Synthetic-to-Real similarity by uploading more representative real files.")
    if not next_steps:
        next_steps.append("Package as a paid pilot bundle and validate with a short controlled field test.")

    return _json_safe({
        "engine": "EdgeTwin Studio V21.1 Reliability Engine 2.0",
        "created_at": _now(),
        "trust_score_v2": trust_score,
        "readiness_stage": readiness_stage,
        "production_risk_level": production_risk,
        "go_no_go": go_no_go,
        "decision": decision,
        "data_quality_score": data_quality,
        "separation_score": separation,
        "base_reliability_score": base_reliability,
        "synthetic_real_similarity_score": similarity_score,
        "avg_sensor_value_score": avg_sensor_value,
        "hardware_fit_score": hardware_fit,
        "hardware_recommendation": hardware_recommendation,
        "worst_class_risk_score": int(worst_class_risk),
        "class_risks": class_risks,
        "sensor_value_scores": sensor_scores,
        "total_real_samples_needed": int(total_needed),
        "has_real_data": bool(has_real_data),
        "red_flags": red_flags,
        "next_steps": next_steps,
        "allowed_claims": allowed_claims,
        "blocked_claims": blocked_claims,
        "disclaimer": RELIABILITY_DISCLAIMER if not has_real_data else BRIDGE_DISCLAIMER,
    })


def create_reliability_v2_bundle(project_name, dataset_df, reliability_v2):
    """Export Reliability Engine 2.0 report + JSON + current dataset."""
    reliability_v2 = reliability_v2 or {}
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Reliability Engine 2.0", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)

    safe_pdf_cell(pdf, "Executive Readiness", 8, True)
    safe_pdf_cell(pdf, f"Trust Score V2: {reliability_v2.get('trust_score_v2', 0)}%")
    safe_pdf_cell(pdf, f"Readiness stage: {reliability_v2.get('readiness_stage', 'Unknown')}")
    safe_pdf_cell(pdf, f"Production risk: {reliability_v2.get('production_risk_level', 'Unknown')}")
    safe_pdf_cell(pdf, f"Go / No-Go: {reliability_v2.get('go_no_go', 'Unknown')}")
    safe_pdf_multicell(pdf, reliability_v2.get("decision", ""))
    pdf.ln(4)

    safe_pdf_cell(pdf, "Score Breakdown", 8, True)
    safe_pdf_cell(pdf, f"Data quality: {reliability_v2.get('data_quality_score', 0)}%")
    safe_pdf_cell(pdf, f"Separation: {reliability_v2.get('separation_score', 0)}%")
    safe_pdf_cell(pdf, f"Base reliability: {reliability_v2.get('base_reliability_score', 0)}%")
    safe_pdf_cell(pdf, f"Synthetic-real similarity: {reliability_v2.get('synthetic_real_similarity_score', 0)}%")
    safe_pdf_cell(pdf, f"Average sensor value: {reliability_v2.get('avg_sensor_value_score', 0)}%")
    safe_pdf_cell(pdf, f"Hardware fit: {reliability_v2.get('hardware_fit_score', 0)}%")
    safe_pdf_cell(pdf, f"Real samples still needed: {reliability_v2.get('total_real_samples_needed', 0)}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Highest Class Risks", 8, True)
    for row in (reliability_v2.get("class_risks", []) or [])[:8]:
        safe_pdf_multicell(pdf, f"{row.get('label')}: {row.get('risk_level')} risk, samples={row.get('samples')}, needed={row.get('recommended_real_samples_needed')} - {row.get('main_issue')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Sensor Value", 8, True)
    for row in (reliability_v2.get("sensor_value_scores", []) or [])[:8]:
        safe_pdf_multicell(pdf, f"{row.get('sensor')}: {row.get('value_score')}% - {row.get('risk')}. {row.get('reason')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Safe Commercial Claims", 8, True)
    for claim in reliability_v2.get("allowed_claims", []):
        safe_pdf_multicell(pdf, f"Allowed: {claim}")
    for claim in reliability_v2.get("blocked_claims", []):
        safe_pdf_multicell(pdf, f"Blocked: {claim}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Next Steps", 8, True)
    for step in reliability_v2.get("next_steps", []):
        safe_pdf_multicell(pdf, f"- {step}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Validation Note", 8, True)
    safe_pdf_multicell(pdf, reliability_v2.get("disclaimer", RELIABILITY_DISCLAIMER))

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("reliability_engine_v2_report.pdf", safe_pdf_output(pdf))
        zf.writestr("reliability_engine_v2.json", json.dumps(_json_safe(reliability_v2), indent=2, ensure_ascii=False))
        if isinstance(dataset_df, pd.DataFrame):
            zf.writestr("current_dataset_snapshot.csv", dataset_df.to_csv(index=False))
        zf.writestr("README.txt", f"""EdgeTwin Studio V21.1 Reliability Engine 2.0 Bundle

Project: {project_name}
Trust Score V2: {reliability_v2.get('trust_score_v2', 0)}%
Readiness stage: {reliability_v2.get('readiness_stage', 'Unknown')}
Production risk: {reliability_v2.get('production_risk_level', 'Unknown')}
Go / No-Go: {reliability_v2.get('go_no_go', 'Unknown')}

{reliability_v2.get('disclaimer', RELIABILITY_DISCLAIMER)}
""")
    return zip_buf.getvalue()


# ============================================================
# V22 HARDWARE BOM & DEPLOYMENT PLANNER
# ============================================================

DEPLOYMENT_DISCLAIMER = (
    "Deployment plans are budgetary pilot estimates. Verify exact component availability, pricing, "
    "certification, enclosure rating, power budget and installation constraints before production rollout."
)

SENSOR_BOM_CATALOG = {
    "Audio": {
        "component": "Digital I2S microphone / acoustic front-end",
        "example": "ICS-43434 / INMP441 class microphone",
        "unit_min_eur": 3,
        "unit_max_eur": 15,
        "avg_ma": 3.0,
        "notes": "Use wind/rain protection for outdoor acoustic pilots.",
    },
    "Vibration": {
        "component": "MEMS accelerometer / vibration sensor",
        "example": "ADXL345/ICM-42688/industrial accelerometer class",
        "unit_min_eur": 4,
        "unit_max_eur": 35,
        "avg_ma": 2.5,
        "notes": "Mechanical mounting quality strongly affects vibration data.",
    },
    "IMU / Movement": {
        "component": "6-axis / 9-axis IMU",
        "example": "ICM-20948 / BMI270 class IMU",
        "unit_min_eur": 5,
        "unit_max_eur": 25,
        "avg_ma": 2.5,
        "notes": "Useful for tamper, movement and orientation context.",
    },
    "Radar": {
        "component": "mmWave / presence radar module",
        "example": "LD2410 / LD2450 class radar",
        "unit_min_eur": 5,
        "unit_max_eur": 25,
        "avg_ma": 80.0,
        "notes": "Good for presence/context; tune zones carefully.",
    },
    "Gas / Environment": {
        "component": "Environmental / gas sensor",
        "example": "BME688 / SGP40 class sensor",
        "unit_min_eur": 10,
        "unit_max_eur": 35,
        "avg_ma": 2.0,
        "notes": "Useful as context; avoid overclaiming gas classification without validation.",
    },
    "Temperature": {
        "component": "Temperature sensor",
        "example": "SHT31 / BME280 class sensor",
        "unit_min_eur": 3,
        "unit_max_eur": 15,
        "avg_ma": 0.5,
        "notes": "Good context for drift, enclosure and battery health.",
    },
    "Humidity": {
        "component": "Humidity sensor",
        "example": "SHT31 / BME280 class sensor",
        "unit_min_eur": 3,
        "unit_max_eur": 15,
        "avg_ma": 0.5,
        "notes": "Good for environmental context and enclosure condensation risk.",
    },
    "GPS / Zone": {
        "component": "GNSS / zone module",
        "example": "u-blox class GNSS / geofence logic",
        "unit_min_eur": 8,
        "unit_max_eur": 35,
        "avg_ma": 30.0,
        "notes": "Use duty-cycling; GPS can dominate power budget.",
    },
}

COMMUNICATION_CATALOG = {
    "LoRa / LoRaWAN": {
        "node_extra_min_eur": 8,
        "node_extra_max_eur": 35,
        "gateway_min_eur": 80,
        "gateway_max_eur": 350,
        "avg_ma_tx": 120,
        "avg_ma_idle": 2,
        "notes": "Best for low-power remote events and small payloads. Not suitable for frequent raw audio upload.",
    },
    "WiFi / MQTT": {
        "node_extra_min_eur": 0,
        "node_extra_max_eur": 10,
        "gateway_min_eur": 0,
        "gateway_max_eur": 150,
        "avg_ma_tx": 180,
        "avg_ma_idle": 25,
        "notes": "Good where site WiFi exists; higher power and less remote-friendly.",
    },
    "LTE / NB-IoT": {
        "node_extra_min_eur": 18,
        "node_extra_max_eur": 80,
        "gateway_min_eur": 0,
        "gateway_max_eur": 0,
        "avg_ma_tx": 250,
        "avg_ma_idle": 8,
        "notes": "Best for independent remote assets. Include SIM/data plan and power headroom.",
    },
    "Wired Ethernet": {
        "node_extra_min_eur": 8,
        "node_extra_max_eur": 35,
        "gateway_min_eur": 0,
        "gateway_max_eur": 120,
        "avg_ma_tx": 70,
        "avg_ma_idle": 35,
        "notes": "Best for industrial fixed installations when cabling is available.",
    },
}

ENCLOSURE_CATALOG = {
    "Indoor / IP40": {"min_eur": 5, "max_eur": 25, "risk": "Low", "notes": "For lab/indoor pilots only."},
    "Outdoor / IP65": {"min_eur": 15, "max_eur": 70, "risk": "Medium", "notes": "Minimum serious outdoor target; add cable glands and acoustic membrane if audio is used."},
    "Outdoor harsh / IP67": {"min_eur": 35, "max_eur": 140, "risk": "Medium-High", "notes": "Recommended for forestry, agriculture and exposed remote assets."},
    "Industrial / IP67 + vibration mount": {"min_eur": 45, "max_eur": 180, "risk": "Medium-High", "notes": "Use secure mounting, strain relief, service access and vibration-safe connectors."},
}

POWER_SOURCE_CATALOG = {
    "Mains + small backup": {"base_min_eur": 15, "base_max_eur": 80, "maintenance": "6-12 months visual check", "notes": "Best when asset power is available; backup covers short outages."},
    "Battery only": {"base_min_eur": 20, "base_max_eur": 120, "maintenance": "Depends on autonomy target", "notes": "Keep duty-cycle low and avoid raw-data transmission."},
    "Solar + battery": {"base_min_eur": 45, "base_max_eur": 250, "maintenance": "Seasonal check recommended", "notes": "Good for remote sites; size panel for worst season, shading and cold."},
    "Vehicle / machine power": {"base_min_eur": 12, "base_max_eur": 80, "maintenance": "Check during machine service", "notes": "Use protection against transients, reverse polarity and noise."},
}


def _normalize_sensor_names(selected_sensors=None, manifest=None, dataset_df=None):
    selected_sensors = selected_sensors or []
    out = []
    for s in selected_sensors:
        if not s:
            continue
        clean = str(s).strip()
        if clean and clean not in out:
            out.append(clean)
    if not out and isinstance(manifest, dict):
        for s in manifest.get("selected_sensors", manifest.get("sensors", [])) or []:
            name = str(s).strip()
            mapping = {"audio": "Audio", "vibration": "Vibration", "radar": "Radar", "gas": "Gas / Environment", "gps": "GPS / Zone"}
            name = mapping.get(name.lower(), name)
            if name and name not in out:
                out.append(name)
    if not out and isinstance(dataset_df, pd.DataFrame) and len(dataset_df.columns):
        col_text = " ".join([str(c).lower() for c in dataset_df.columns])
        candidates = [
            ("Audio", ["audio", "spectral", "zcr"]),
            ("Vibration", ["vibration", "rms", "kurtosis", "crest"]),
            ("Radar", ["radar"]),
            ("Gas / Environment", ["gas", "temperature", "humidity"]),
            ("GPS / Zone", ["gps", "zone"]),
        ]
        for name, keys in candidates:
            if any(k in col_text for k in keys) and name not in out:
                out.append(name)
    return out or ["Audio", "Vibration"]


def _estimate_board_power_ma(board_name):
    board_name = str(board_name or "").lower()
    if "rak4631" in board_name or "nrf52840" in board_name:
        return {"active_ma": 18, "sleep_ma": 0.08, "notes": "Very low-power node; keep DSP windows short."}
    if "esp32" in board_name:
        return {"active_ma": 95, "sleep_ma": 0.20, "notes": "Good DSP/TinyML balance; WiFi peaks need power headroom."}
    if "stm32u5" in board_name:
        return {"active_ma": 28, "sleep_ma": 0.03, "notes": "Strong low-power industrial MCU option."}
    if "stm32h7" in board_name:
        return {"active_ma": 120, "sleep_ma": 1.00, "notes": "High DSP performance but higher power."}
    if "zero" in board_name:
        return {"active_ma": 280, "sleep_ma": 80, "notes": "Linux gateway class; not ideal for long battery-only nodes."}
    if "raspberry pi 5" in board_name:
        return {"active_ma": 900, "sleep_ma": 300, "notes": "Gateway/server class; use stable mains power."}
    if "linux" in board_name:
        return {"active_ma": 1200, "sleep_ma": 500, "notes": "Industrial gateway/server; not battery-node class."}
    return {"active_ma": 80, "sleep_ma": 0.5, "notes": "Generic estimate; validate with measurements."}


def infer_deployment_defaults(dataset_df=None, manifest=None, hardware_result=None):
    manifest = manifest if isinstance(manifest, dict) else {}
    use_case = manifest.get("use_case_type") or manifest.get("template") or "Custom Edge AI pilot"
    env = manifest.get("environment", "Custom")
    sensors = _normalize_sensor_names(manifest=manifest, dataset_df=dataset_df)
    sr = manifest.get("sample_rate") or 16000
    try:
        sr = int(sr)
    except Exception:
        sr = 16000
    recommendation = "ESP32-S3"
    if isinstance(hardware_result, dict) and hardware_result.get("recommendation"):
        recommendation = hardware_result.get("recommendation")
    if "Forest" in str(env) or "remote" in str(env).lower() or "Forestry" in str(use_case):
        comm = "LoRa / LoRaWAN"
        enclosure = "Outdoor harsh / IP67"
        power = "Solar + battery"
    elif "Industrial" in str(env) or "Predictive" in str(use_case):
        comm = "WiFi / MQTT"
        enclosure = "Industrial / IP67 + vibration mount"
        power = "Mains + small backup"
    elif "Construction" in str(env) or "Security" in str(use_case) or "Tamper" in str(use_case):
        comm = "LTE / NB-IoT"
        enclosure = "Outdoor / IP65"
        power = "Battery only"
    else:
        comm = "WiFi / MQTT"
        enclosure = "Outdoor / IP65"
        power = "Mains + small backup"
    return {
        "use_case": use_case,
        "environment": env,
        "selected_sensors": sensors,
        "sample_rate": sr,
        "recommended_board": recommendation,
        "communication": comm,
        "enclosure_target": enclosure,
        "power_source": power,
        "deployment_scale": "Pilot: 1-5 nodes",
        "autonomy_days": 7,
        "maintenance_profile": "standard",
        "priority": "balanced",
    }


def _deployment_fft_size(sr, priority="balanced"):
    sr = int(sr or 16000)
    if sr <= 4000:
        return 1024
    if priority == "low_power":
        return 1024
    return 2048


def _add_bom_row(rows, category, component, recommendation, qty, unit_min, unit_max, notes, required=True):
    qty = int(max(1, qty)) if required else int(max(0, qty))
    rows.append(_json_safe({
        "category": category,
        "component": component,
        "recommendation": recommendation,
        "qty": qty,
        "unit_min_eur": float(unit_min),
        "unit_max_eur": float(unit_max),
        "line_min_eur": float(qty * unit_min),
        "line_max_eur": float(qty * unit_max),
        "required": bool(required),
        "notes": notes,
    }))


def build_deployment_plan(
    project_name,
    dataset_df=None,
    manifest=None,
    hardware_result=None,
    selected_sensors=None,
    environment="Custom",
    deployment_scale="Pilot: 1-5 nodes",
    autonomy_days=7,
    communication="WiFi / MQTT",
    power_source="Mains + small backup",
    enclosure_target="Outdoor / IP65",
    maintenance_profile="standard",
    priority="balanced",
    sample_rate=None,
    node_count=None,
):
    """Create a deployment-ready pilot plan with BOM, power budget, comms plan and risks."""
    manifest = manifest if isinstance(manifest, dict) else {}
    df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    sensors = _normalize_sensor_names(selected_sensors=selected_sensors, manifest=manifest, dataset_df=df)
    use_case = manifest.get("use_case_type") or manifest.get("template") or "Custom Edge AI pilot"
    sr = int(sample_rate or manifest.get("sample_rate", 16000) or 16000)
    if node_count is None:
        if "1-5" in deployment_scale:
            node_count = 3
        elif "6-25" in deployment_scale:
            node_count = 12
        elif "25" in deployment_scale:
            node_count = 30
        else:
            node_count = 3
    node_count = int(max(1, node_count))

    node_board_pool = ["ESP32-S3", "RAK4631 / nRF52840", "STM32U5", "STM32H7"]
    gateway_board_names = ["Raspberry Pi Zero 2 W", "Raspberry Pi 5", "Generic Linux Gateway"]
    if hardware_result is None or not isinstance(hardware_result, dict) or not hardware_result:
        numeric_cols = [c for c in df.columns if c != "Label" and pd.api.types.is_numeric_dtype(df[c])] if len(df) else []
        selected_pool = None if priority == "gateway" else node_board_pool
        hardware_result = hardware_auto_architect(max(1, len(numeric_cols) or 8), sr, priority, selected_boards=selected_pool)
    board = hardware_result.get("recommendation", "ESP32-S3")
    if priority != "gateway" and board in gateway_board_names:
        # Deployment Planner should recommend a node-class board unless the user is explicitly planning a gateway.
        numeric_cols = [c for c in df.columns if c != "Label" and pd.api.types.is_numeric_dtype(df[c])] if len(df) else []
        hardware_result = hardware_auto_architect(max(1, len(numeric_cols) or 8), sr, priority, selected_boards=node_board_pool)
        board = hardware_result.get("recommendation", "ESP32-S3")
    ranking = hardware_result.get("ranking", []) or []
    hardware_fit = 65
    if ranking:
        try:
            hardware_fit = int(np.clip(float(ranking[0].get("adjusted_score", ranking[0].get("score", 65))), 0, 100))
        except Exception:
            hardware_fit = 65

    numeric_cols = [c for c in df.columns if c != "Label" and pd.api.types.is_numeric_dtype(df[c])] if len(df) else []
    feature_count = int(max(1, len(numeric_cols) or 8))
    fft_size = _deployment_fft_size(sr, priority)
    window_s = 1.0 if sr <= 8000 else 0.5
    est_ram, fft_ms, feat_ms, inf_ms = estimate_edge_load(board, feature_count, sr, duration=window_s)
    latency_ms = float(fft_ms + feat_ms + inf_ms)

    bom_rows = []
    board_profile = HARDWARE_PROFILES.get(board, {})
    board_min, board_max = (8, 25)
    if "RAK4631" in board:
        board_min, board_max = (25, 70)
    elif "ESP32" in board:
        board_min, board_max = (8, 35)
    elif "STM32U5" in board:
        board_min, board_max = (20, 80)
    elif "STM32H7" in board:
        board_min, board_max = (25, 95)
    elif "Raspberry Pi Zero" in board:
        board_min, board_max = (20, 60)
    elif "Raspberry Pi 5" in board:
        board_min, board_max = (75, 150)
    elif "Linux" in board:
        board_min, board_max = (150, 900)
    _add_bom_row(bom_rows, "Compute", "Edge node compute board", board, node_count, board_min, board_max, board_profile.get("notes", "Validate board availability and pin compatibility."))

    sensor_current = 0.0
    used_sensor_keys = []
    for s in sensors:
        key = s if s in SENSOR_BOM_CATALOG else None
        if key is None and s == "Gas/Env":
            key = "Gas / Environment"
        if key is None:
            continue
        item = SENSOR_BOM_CATALOG[key]
        used_sensor_keys.append(key)
        sensor_current += float(item.get("avg_ma", 0))
        _add_bom_row(bom_rows, "Sensor", item["component"], item["example"], node_count, item["unit_min_eur"], item["unit_max_eur"], item["notes"])

    comm = COMMUNICATION_CATALOG.get(communication, COMMUNICATION_CATALOG["WiFi / MQTT"])
    if comm["node_extra_max_eur"] > 0:
        _add_bom_row(bom_rows, "Communication", "Node communication module / radio support", communication, node_count, comm["node_extra_min_eur"], comm["node_extra_max_eur"], comm["notes"])
    if comm["gateway_max_eur"] > 0 and communication in ["LoRa / LoRaWAN", "WiFi / MQTT", "Wired Ethernet"]:
        gateway_qty = 1 if node_count <= 25 else max(1, int(np.ceil(node_count / 25)))
        _add_bom_row(bom_rows, "Gateway", "Site gateway / bridge", "Raspberry Pi / industrial gateway class", gateway_qty, comm["gateway_min_eur"], comm["gateway_max_eur"], "Gateway cost depends on range, enclosure and backhaul.")

    enc = ENCLOSURE_CATALOG.get(enclosure_target, ENCLOSURE_CATALOG["Outdoor / IP65"])
    _add_bom_row(bom_rows, "Mechanical", "Enclosure and cable glands", enclosure_target, node_count, enc["min_eur"], enc["max_eur"], enc["notes"])
    _add_bom_row(bom_rows, "Mechanical", "Mounting kit / bracket / strain relief", "Use-case specific mounting", node_count, 5, 35, "Mounting quality is critical for vibration and outdoor reliability.")
    power = POWER_SOURCE_CATALOG.get(power_source, POWER_SOURCE_CATALOG["Mains + small backup"])
    _add_bom_row(bom_rows, "Power", "Power supply / battery / charge hardware", power_source, node_count, power["base_min_eur"], power["base_max_eur"], power["notes"])
    _add_bom_row(bom_rows, "Validation", "Pilot installation and test spares", "Spare sensors, SD card, cables, fasteners", max(1, int(np.ceil(node_count * 0.15))), 15, 80, "Keep spares for first field test.", required=True)

    total_min = float(sum(r["line_min_eur"] for r in bom_rows))
    total_max = float(sum(r["line_max_eur"] for r in bom_rows))

    board_power = _estimate_board_power_ma(board)
    # Simple duty-cycle model. Keep conservative because this is for pilot planning.
    active_duty = 0.20 if priority == "performance" else 0.10 if priority == "balanced" else 0.04
    if communication == "WiFi / MQTT":
        active_duty += 0.04
    elif communication == "LTE / NB-IoT":
        active_duty += 0.03
    elif communication == "LoRa / LoRaWAN":
        active_duty += 0.01
    active_duty = float(np.clip(active_duty, 0.02, 0.60))
    active_ma = float(board_power["active_ma"] + sensor_current + comm.get("avg_ma_tx", 0) * 0.08)
    sleep_ma = float(board_power["sleep_ma"] + max(0.05, sensor_current * 0.10) + comm.get("avg_ma_idle", 0) * 0.15)
    avg_ma = float(active_ma * active_duty + sleep_ma * (1 - active_duty))
    required_mAh = float(avg_ma * 24 * float(autonomy_days) * 1.35)
    recommended_battery_mAh = int(np.ceil(required_mAh / 500.0) * 500)

    if power_source in ["Mains + small backup", "Vehicle / machine power"]:
        maintenance_interval = "6-12 months"
    elif power_source == "Solar + battery":
        maintenance_interval = "3-6 months during first pilot season, then 6-12 months if stable"
    else:
        if recommended_battery_mAh <= 5000:
            maintenance_interval = "2-6 weeks, depending on duty-cycle and temperature"
        elif recommended_battery_mAh <= 20000:
            maintenance_interval = "1-3 months, depending on duty-cycle and temperature"
        else:
            maintenance_interval = "Large battery required; review duty-cycle or power source"

    risks = []
    if "Audio" in used_sensor_keys and "Outdoor" in enclosure_target:
        risks.append({"severity": "medium", "risk": "Outdoor audio quality", "mitigation": "Use acoustic membrane/wind protection and collect field noise examples."})
    if "Vibration" in used_sensor_keys:
        risks.append({"severity": "medium", "risk": "Mounting affects vibration signatures", "mitigation": "Define a repeatable mounting method and include it in the field test."})
    if communication == "LoRa / LoRaWAN" and sr >= 16000:
        risks.append({"severity": "medium", "risk": "Raw audio cannot be sent over LoRa", "mitigation": "Send features/events only; store raw clips locally only for validation."})
    if communication == "LTE / NB-IoT":
        risks.append({"severity": "medium", "risk": "Cellular coverage and SIM cost", "mitigation": "Run coverage test and budget monthly connectivity per node."})
    if power_source == "Battery only" and avg_ma > 15:
        risks.append({"severity": "high", "risk": "Battery-only autonomy may be short", "mitigation": "Lower duty-cycle, use event-triggered sampling or add external power/solar."})
    if hardware_fit < 55:
        risks.append({"severity": "high", "risk": "Hardware fit is weak", "mitigation": "Review board choice, sample rate, feature count and FFT size before field pilot."})
    if len(df) and "Label" in df.columns and df["Label"].nunique() < 2:
        risks.append({"severity": "high", "risk": "Dataset has too few classes", "mitigation": "Add realistic normal and event examples before training."})
    if not risks:
        risks.append({"severity": "info", "risk": "No critical deployment blockers detected", "mitigation": "Proceed with a controlled pilot and document field results."})

    if hardware_fit >= 70 and latency_ms < 50 and avg_ma < 80 and not any(r["severity"] == "high" for r in risks):
        go_no_go = "GO"
        readiness = "Deployment pilot-ready"
    elif hardware_fit >= 55 and latency_ms < 120:
        go_no_go = "CONDITIONAL"
        readiness = "Deployment-prep ready"
    else:
        go_no_go = "NO-GO"
        readiness = "Needs hardware/power review"

    validation_plan = [
        "Bench-test sensor wiring, sample rate, FFT window and event pipeline.",
        "Record normal baseline data for at least one operating cycle.",
        "Collect controlled event samples for every target class.",
        "Compare field fingerprints with generated synthetic fingerprints.",
        "Review false positives/false negatives before production claims.",
        "Freeze hardware BOM only after field-test power and data quality are measured.",
    ]
    if "Vibration" in used_sensor_keys:
        validation_plan.append("Repeat vibration tests after remounting to quantify mounting sensitivity.")
    if "Audio" in used_sensor_keys:
        validation_plan.append("Test wind, rain, vehicle and human background audio separately.")

    next_steps = [
        "Build one bench node with the recommended BOM.",
        "Run 24-72 hour baseline logging before customer demo claims.",
        "Use Real Bridge with field WAV/CSV to improve synthetic-real similarity.",
        "Update Reliability Engine 2.0 after the first field dataset.",
    ]

    telemetry = {
        "event_payload": ["timestamp", "node_id", "predicted_label", "confidence", "reliability_stage", "battery", "temperature"],
        "recommended_mode": "features/events first; raw snippets only for validation when bandwidth allows",
        "storage": "local CSV/JSON ring buffer or SD card for field debug",
        "mqtt_topics": ["edgetwin/<project>/node/<id>/event", "edgetwin/<project>/node/<id>/health", "edgetwin/<project>/node/<id>/debug"],
    }

    return _json_safe({
        "engine": "EdgeTwin Studio V22 Hardware BOM & Deployment Planner",
        "created_at": _now(),
        "project_name": project_name,
        "use_case": use_case,
        "environment": environment or manifest.get("environment", "Custom"),
        "deployment_scale": deployment_scale,
        "node_count_estimate": node_count,
        "selected_sensors": sensors,
        "communication": communication,
        "power_source": power_source,
        "enclosure_target": enclosure_target,
        "maintenance_profile": maintenance_profile,
        "priority": priority,
        "readiness": readiness,
        "go_no_go": go_no_go,
        "hardware": {
            "recommended_board": board,
            "hardware_fit_score": hardware_fit,
            "role": board_profile.get("role", "Edge node"),
            "power_class": board_profile.get("power_class", "Unknown"),
            "reason": hardware_result.get("reason", ""),
        },
        "edge_settings": {
            "sample_rate_hz": sr,
            "fft_size": fft_size,
            "window_seconds": window_s,
            "feature_count": feature_count,
            "estimated_latency_ms": round(latency_ms, 2),
            "estimated_ram_kb": round(float(est_ram), 2),
            "recommended_feature_policy": "Use extracted features for telemetry; avoid continuous raw signal upload unless on wired/WiFi gateway.",
        },
        "power_budget": {
            "active_duty_estimate": round(active_duty, 3),
            "active_current_ma": round(active_ma, 2),
            "sleep_current_ma": round(sleep_ma, 2),
            "average_current_ma": round(avg_ma, 2),
            "autonomy_days": float(autonomy_days),
            "recommended_battery_mAh": recommended_battery_mAh,
            "maintenance_interval_estimate": maintenance_interval,
            "notes": board_power.get("notes", "") + " " + power.get("notes", ""),
        },
        "cost_estimate": {
            "min_total_eur": round(total_min, 2),
            "max_total_eur": round(total_max, 2),
            "per_node_min_eur": round(total_min / max(node_count, 1), 2),
            "per_node_max_eur": round(total_max / max(node_count, 1), 2),
            "confidence": "Budgetary estimate only",
        },
        "bom": bom_rows,
        "communication_plan": {
            "mode": communication,
            "notes": comm.get("notes", ""),
            "payload_policy": "Send class/confidence/health events first; only upload raw data during validation windows.",
        },
        "telemetry_plan": telemetry,
        "deployment_risks": risks,
        "validation_plan": validation_plan,
        "next_steps": next_steps,
        "disclaimer": DEPLOYMENT_DISCLAIMER,
    })


def create_deployment_bundle(project_name, deployment_plan, dataset_df=None):
    """Export V22 deployment planner bundle with PDF, BOM CSV and JSON manifest."""
    deployment_plan = deployment_plan or {}
    bom_df = pd.DataFrame(deployment_plan.get("bom", []))

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Deployment Planner", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.cell(0, 8, txt=clean_pdf_text("V22 Hardware BOM & Field Pilot Plan"), ln=True, align="C")
    pdf.ln(8)

    safe_pdf_cell(pdf, "Executive Deployment Summary", 8, True)
    safe_pdf_cell(pdf, f"Readiness: {deployment_plan.get('readiness', 'Unknown')}")
    safe_pdf_cell(pdf, f"Go / No-Go: {deployment_plan.get('go_no_go', 'Unknown')}")
    safe_pdf_cell(pdf, f"Use case: {deployment_plan.get('use_case', 'Unknown')}")
    safe_pdf_cell(pdf, f"Environment: {deployment_plan.get('environment', 'Unknown')}")
    safe_pdf_cell(pdf, f"Node count estimate: {deployment_plan.get('node_count_estimate', 0)}")
    safe_pdf_multicell(pdf, deployment_plan.get("disclaimer", DEPLOYMENT_DISCLAIMER))
    pdf.ln(4)

    hw = deployment_plan.get("hardware", {}) or {}
    edge = deployment_plan.get("edge_settings", {}) or {}
    power = deployment_plan.get("power_budget", {}) or {}
    cost = deployment_plan.get("cost_estimate", {}) or {}
    safe_pdf_cell(pdf, "Hardware & Edge Settings", 8, True)
    safe_pdf_cell(pdf, f"Recommended board: {hw.get('recommended_board', 'Unknown')}")
    safe_pdf_cell(pdf, f"Hardware fit score: {hw.get('hardware_fit_score', 0)}%")
    safe_pdf_cell(pdf, f"Sample rate: {edge.get('sample_rate_hz', 0)} Hz")
    safe_pdf_cell(pdf, f"FFT size: {edge.get('fft_size', 0)}")
    safe_pdf_cell(pdf, f"Estimated latency: {edge.get('estimated_latency_ms', 0)} ms")
    safe_pdf_cell(pdf, f"Estimated RAM: {edge.get('estimated_ram_kb', 0)} KB")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Power Budget", 8, True)
    safe_pdf_cell(pdf, f"Power source: {deployment_plan.get('power_source', 'Unknown')}")
    safe_pdf_cell(pdf, f"Average current estimate: {power.get('average_current_ma', 0)} mA")
    safe_pdf_cell(pdf, f"Recommended battery: {power.get('recommended_battery_mAh', 0)} mAh")
    safe_pdf_cell(pdf, f"Maintenance interval: {power.get('maintenance_interval_estimate', 'Unknown')}")
    safe_pdf_multicell(pdf, power.get("notes", ""))
    pdf.ln(4)

    safe_pdf_cell(pdf, "Budgetary Cost Estimate", 8, True)
    safe_pdf_cell(pdf, f"Total estimate: EUR {cost.get('min_total_eur', 0)} - EUR {cost.get('max_total_eur', 0)}")
    safe_pdf_cell(pdf, f"Per-node estimate: EUR {cost.get('per_node_min_eur', 0)} - EUR {cost.get('per_node_max_eur', 0)}")
    safe_pdf_cell(pdf, cost.get("confidence", "Budgetary estimate only"))
    pdf.ln(4)

    safe_pdf_cell(pdf, "Top BOM Items", 8, True)
    for row in (deployment_plan.get("bom", []) or [])[:12]:
        safe_pdf_multicell(pdf, f"{row.get('category')}: {row.get('component')} x{row.get('qty')} - EUR {row.get('line_min_eur')}-{row.get('line_max_eur')} ({row.get('recommendation')})")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Deployment Risks", 8, True)
    for row in deployment_plan.get("deployment_risks", []):
        safe_pdf_multicell(pdf, f"[{row.get('severity', 'info').upper()}] {row.get('risk')} - {row.get('mitigation')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Validation Plan", 8, True)
    for step in deployment_plan.get("validation_plan", []):
        safe_pdf_multicell(pdf, f"- {step}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Next Steps", 8, True)
    for step in deployment_plan.get("next_steps", []):
        safe_pdf_multicell(pdf, f"- {step}")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("deployment_plan.pdf", safe_pdf_output(pdf))
        zf.writestr("deployment_plan.json", json.dumps(_json_safe(deployment_plan), indent=2, ensure_ascii=False))
        if len(bom_df):
            zf.writestr("hardware_bom.csv", bom_df.to_csv(index=False))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df):
            zf.writestr("dataset_snapshot.csv", dataset_df.to_csv(index=False))
        zf.writestr("README.txt", f"""EdgeTwin Studio V22 Deployment Planner Bundle

Project: {project_name}
Readiness: {deployment_plan.get('readiness', 'Unknown')}
Go / No-Go: {deployment_plan.get('go_no_go', 'Unknown')}
Recommended board: {(deployment_plan.get('hardware') or {}).get('recommended_board', 'Unknown')}
Estimated total cost: EUR {(deployment_plan.get('cost_estimate') or {}).get('min_total_eur', 0)} - EUR {(deployment_plan.get('cost_estimate') or {}).get('max_total_eur', 0)}

{deployment_plan.get('disclaimer', DEPLOYMENT_DISCLAIMER)}
""")
    return zip_buf.getvalue()



# ============================================================
# V23 PROFESSIONAL REPORTS 2.0
# ============================================================

REPORTS_DISCLAIMER = (
    "Professional Reports 2.0 summarizes pilot-readiness evidence. It is not a production guarantee, "
    "certification, safety approval or replacement for field validation with real deployment data."
)


def get_professional_report_types():
    return [
        "Executive Pilot Report",
        "Technical Validation Report",
        "Deployment Decision Report",
        "Sales / Partner Pitch Report",
    ]


def _professional_dataset_quality(dataset_df, doctor=None, manifest=None):
    manifest = manifest or {}
    if not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0:
        return {
            "rows": 0,
            "labels": 0,
            "features": 0,
            "overall_score": 0,
            "diversity_score": 0,
            "balance_score": 0,
            "separation_score": 0,
            "label_counts": {},
            "recommended_training_features": manifest.get("recommended_training_features", []),
        }

    df = dataset_df.copy()
    if doctor and isinstance(doctor, dict) and doctor.get("overall_score") is not None:
        base = dict(doctor)
    elif "Label" in df.columns:
        numeric_cols = [c for c in df.columns if c != "Label" and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            base = dataset_doctor(df[numeric_cols], df["Label"])
        else:
            base = {"overall_score": 0, "diversity_score": 0, "balance_score": 0, "separation_score": 0, "advice": []}
    else:
        base = {"overall_score": 0, "diversity_score": 0, "balance_score": 0, "separation_score": 0, "advice": []}

    label_counts = df["Label"].value_counts().to_dict() if "Label" in df.columns else {}
    feature_cols = [c for c in df.columns if c != "Label"]
    return _json_safe({
        "rows": int(len(df)),
        "labels": int(df["Label"].nunique()) if "Label" in df.columns else 0,
        "features": int(len(feature_cols)),
        "overall_score": int(base.get("overall_score", 0)),
        "diversity_score": int(base.get("diversity_score", 0)),
        "balance_score": int(base.get("balance_score", 0)),
        "separation_score": int(base.get("separation_score", 0)),
        "label_counts": label_counts,
        "advice": base.get("advice", []),
        "recommended_training_features": base.get("recommended_training_features", manifest.get("recommended_training_features", feature_cols[:12])),
    })


def _compact_real_bridge_for_report(real_bridge_result):
    if not isinstance(real_bridge_result, dict) or not real_bridge_result:
        return {
            "used": False,
            "similarity_score": 0,
            "real_files": 0,
            "summary": "No real-data bridge evidence is attached to this report.",
        }
    summary = compact_bridge_summary(real_bridge_result)
    return _json_safe({
        "used": True,
        "similarity_score": summary.get("synthetic_real_similarity_score", real_bridge_result.get("similarity_score", 0)),
        "real_files": summary.get("real_files", len(real_bridge_result.get("real_features_df", [])) if isinstance(real_bridge_result.get("real_features_df"), list) else 0),
        "weak_labels": summary.get("weak_labels", []),
        "summary": "Real WAV/CSV files were used to create a signal fingerprint and synthetic-to-real comparison.",
    })


def _readiness_from_inputs(dataset_quality, reliability_v2=None, trust_gate=None, deployment_plan=None, real_bridge=None):
    reliability_v2 = reliability_v2 or {}
    trust_gate = trust_gate or {}
    deployment_plan = deployment_plan or {}
    real_bridge = real_bridge or {}

    trust_score = int(reliability_v2.get("trust_score_v2", 0) or trust_gate.get("trust_score", 0) or dataset_quality.get("overall_score", 0))
    stage = reliability_v2.get("readiness_stage") or trust_gate.get("readiness_stage") or deployment_plan.get("readiness") or "Demo / prototype-ready"
    production_risk = reliability_v2.get("production_risk_level") or trust_gate.get("production_risk") or "High"
    go_no_go = reliability_v2.get("go_no_go") or trust_gate.get("go_no_go") or deployment_plan.get("go_no_go") or "CONDITIONAL"

    if dataset_quality.get("rows", 0) == 0:
        stage = "Concept-only"
        production_risk = "Very High"
        go_no_go = "NO-GO"
    elif not real_bridge.get("used") and trust_score >= 80:
        # Prevent overclaiming on synthetic-only packages.
        stage = "Pilot-prep ready"
        production_risk = "Medium-High"
        go_no_go = "CONDITIONAL"

    return _json_safe({
        "trust_score": int(np.clip(trust_score, 0, 100)),
        "stage": stage,
        "production_risk": production_risk,
        "go_no_go": go_no_go,
        "has_real_data": bool(real_bridge.get("used", False) or reliability_v2.get("has_real_data", False)),
        "real_similarity_score": int(real_bridge.get("similarity_score", reliability_v2.get("synthetic_real_similarity_score", 0) or 0)),
    })


def _commercial_positioning_from_readiness(readiness, package_level, reliability_v2=None):
    reliability_v2 = reliability_v2 or {}
    score = int(readiness.get("trust_score", 0))
    has_real = bool(readiness.get("has_real_data", False))
    go = readiness.get("go_no_go", "CONDITIONAL")

    if go == "NO-GO" or score < 55:
        suggested = "Internal concept / demo only"
        price_range = "EUR 0 - 99"
    elif has_real and score >= 78:
        suggested = "Real-Data Pilot Bundle"
        price_range = "EUR 499 - 1,500+"
    elif score >= 70:
        suggested = "Professional Pilot Bundle"
        price_range = "EUR 149 - 499"
    else:
        suggested = "Starter Pilot Bundle"
        price_range = "EUR 49 - 199"

    allowed = reliability_v2.get("allowed_claims") or [
        "Generates an Edge AI pilot-preparation package from sensor/use-case inputs.",
        "Provides dataset quality, readiness, hardware direction and next-step recommendations.",
        "Helps reduce trial-and-error before training/deployment in tools such as Edge Impulse or TinyML pipelines.",
    ]
    blocked = reliability_v2.get("blocked_claims") or [
        "Do not claim production-ready without field validation.",
        "Do not claim guaranteed model accuracy from synthetic data alone.",
        "Do not claim certification, safety approval or hardware compliance.",
    ]
    return _json_safe({
        "requested_package_level": package_level,
        "suggested_package": suggested,
        "suggested_price_range": price_range,
        "allowed_claims": allowed,
        "blocked_claims": blocked,
    })


def build_professional_report_snapshot(
    project_name,
    dataset_df=None,
    manifest=None,
    doctor=None,
    reliability_v2=None,
    trust_gate=None,
    deployment_plan=None,
    hardware_result=None,
    commercial_summary=None,
    real_bridge_result=None,
    report_type="Executive Pilot Report",
    package_level="Professional Pilot",
    customer_name="Customer",
    customer_problem="",
    audience="Mixed business + technical",
    prepared_by="EdgeTwin Studio / OMEGA-X Engine",
):
    """Build one JSON-safe V23 snapshot used by the professional PDF and ZIP bundle."""
    manifest = manifest or {}
    commercial_summary = commercial_summary or {}
    reliability_v2 = reliability_v2 or {}
    trust_gate = trust_gate or {}
    deployment_plan = deployment_plan or {}
    hardware_result = hardware_result or {}

    dataset_quality = _professional_dataset_quality(dataset_df, doctor, manifest)
    bridge = _compact_real_bridge_for_report(real_bridge_result)
    readiness = _readiness_from_inputs(dataset_quality, reliability_v2, trust_gate, deployment_plan, bridge)
    commercial = _commercial_positioning_from_readiness(readiness, package_level, reliability_v2)

    problem = customer_problem or commercial_summary.get("problem") or manifest.get("project_goal") or "Customer wants to move from a sensor idea to an Edge AI pilot with less trial-and-error."
    solution = commercial_summary.get("solution") or "EdgeTwin Studio generated a dataset package, audit, readiness view, hardware direction and deployment next steps."

    decision_text = {
        "GO": "Good candidate for a controlled paid field pilot. Still validate before production rollout.",
        "CONDITIONAL": "Usable for pilot preparation, with clear conditions before stronger customer promises.",
        "NO-GO": "Not ready for a paid pilot claim yet. Improve data quality, real-data evidence or deployment assumptions first.",
    }.get(readiness.get("go_no_go"), "Use as pilot preparation with validation conditions.")

    next_steps = []
    if reliability_v2.get("next_steps"):
        next_steps.extend(reliability_v2.get("next_steps", [])[:5])
    if deployment_plan.get("next_steps"):
        next_steps.extend(deployment_plan.get("next_steps", [])[:5])
    if not next_steps:
        if not readiness.get("has_real_data"):
            next_steps.append("Upload representative real WAV/CSV files to improve Synthetic-to-Real confidence.")
        next_steps.append("Run a short controlled field pilot and compare predicted labels with real events.")
        next_steps.append("Use the hardware/deployment plan as a budgetary starting point, not final certification.")

    executive_summary = {
        "problem": problem,
        "solution": solution,
        "summary": f"{project_name} is currently assessed as {readiness.get('stage')} with a Trust Score of {readiness.get('trust_score')}%. The recommended decision is {readiness.get('go_no_go')} for the requested {package_level} package.",
        "decision": decision_text,
        "next_step": next_steps[0] if next_steps else "Validate with field data before production deployment.",
    }

    hw_recommendation = hardware_result.get("recommendation") or (deployment_plan.get("hardware") or {}).get("recommended_board", "Unknown")
    hardware_summary = {
        "recommendation": hw_recommendation,
        "reason": hardware_result.get("reason") or (deployment_plan.get("hardware") or {}).get("reason", "Run Hardware Architect or Deployment Planner for a stronger hardware direction."),
        "ranking": hardware_result.get("ranking", []),
    }

    deployment_summary = {
        "available": bool(deployment_plan),
        "readiness": deployment_plan.get("readiness", "Not generated"),
        "go_no_go": deployment_plan.get("go_no_go", "Unknown"),
        "communication": (deployment_plan.get("communication_plan") or {}).get("mode", deployment_plan.get("communication", "Unknown")),
        "power_source": deployment_plan.get("power_source", "Unknown"),
        "enclosure": deployment_plan.get("enclosure_target", "Unknown"),
        "cost_estimate": deployment_plan.get("cost_estimate", {}),
        "risks": deployment_plan.get("deployment_risks", []),
        "validation_plan": deployment_plan.get("validation_plan", []),
    }

    return _json_safe({
        "engine": "EdgeTwin Studio V23 Professional Reports 2.0",
        "created_at": _now(),
        "project_name": project_name,
        "customer_name": customer_name,
        "prepared_by": prepared_by,
        "report_type": report_type,
        "audience": audience,
        "package_level": package_level,
        "template": manifest.get("template", manifest.get("pack_name", "Unknown")),
        "executive_summary": executive_summary,
        "dataset_quality": dataset_quality,
        "readiness": readiness,
        "reliability_v2": reliability_v2,
        "real_bridge": bridge,
        "trust_gate": trust_gate,
        "hardware": hardware_summary,
        "deployment": deployment_summary,
        "commercial_positioning": commercial,
        "next_steps": list(dict.fromkeys([str(s) for s in next_steps]))[:10],
        "disclaimer": REPORTS_DISCLAIMER,
    })


def _pdf_section_title(pdf, title):
    pdf.ln(3)
    safe_pdf_cell(pdf, title, 8, True)


def _pdf_kv(pdf, key, value):
    safe_pdf_cell(pdf, f"{key}: {value}")


def generate_professional_report_pdf(snapshot):
    snapshot = snapshot or {}
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title page
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 12, txt=clean_pdf_text("EdgeTwin Studio"), ln=True, align="C")
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, txt=clean_pdf_text("Professional Edge AI Pilot Report"), ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("Powered by the OMEGA-X Engine"), ln=True, align="C")
    pdf.ln(10)

    _pdf_kv(pdf, "Project", snapshot.get("project_name", "Unknown"))
    _pdf_kv(pdf, "Customer", snapshot.get("customer_name", "Customer"))
    _pdf_kv(pdf, "Report type", snapshot.get("report_type", "Executive Pilot Report"))
    _pdf_kv(pdf, "Audience", snapshot.get("audience", "Mixed business + technical"))
    _pdf_kv(pdf, "Prepared by", snapshot.get("prepared_by", "EdgeTwin Studio / OMEGA-X Engine"))
    _pdf_kv(pdf, "Created", snapshot.get("created_at", _now()))
    pdf.ln(8)
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", REPORTS_DISCLAIMER))

    # Executive summary
    pdf.add_page()
    exec_summary = snapshot.get("executive_summary", {}) or {}
    readiness = snapshot.get("readiness", {}) or {}
    dataset_quality = snapshot.get("dataset_quality", {}) or {}
    commercial = snapshot.get("commercial_positioning", {}) or {}

    _pdf_section_title(pdf, "1. Executive Summary")
    safe_pdf_multicell(pdf, exec_summary.get("summary", ""))
    pdf.ln(2)
    safe_pdf_multicell(pdf, f"Problem: {exec_summary.get('problem', '')}")
    safe_pdf_multicell(pdf, f"Solution: {exec_summary.get('solution', '')}")
    safe_pdf_multicell(pdf, f"Decision: {exec_summary.get('decision', '')}")
    safe_pdf_multicell(pdf, f"Recommended next step: {exec_summary.get('next_step', '')}")

    _pdf_section_title(pdf, "2. Readiness Decision")
    _pdf_kv(pdf, "Trust Score", f"{readiness.get('trust_score', 0)}%")
    _pdf_kv(pdf, "Readiness stage", readiness.get("stage", "Unknown"))
    _pdf_kv(pdf, "Go / No-Go", readiness.get("go_no_go", "Unknown"))
    _pdf_kv(pdf, "Production risk", readiness.get("production_risk", "Unknown"))
    _pdf_kv(pdf, "Real data used", "Yes" if readiness.get("has_real_data") else "No")
    _pdf_kv(pdf, "Synthetic-to-real similarity", f"{readiness.get('real_similarity_score', 0)}%")

    _pdf_section_title(pdf, "3. Dataset Quality")
    _pdf_kv(pdf, "Rows", dataset_quality.get("rows", 0))
    _pdf_kv(pdf, "Labels", dataset_quality.get("labels", 0))
    _pdf_kv(pdf, "Features", dataset_quality.get("features", 0))
    _pdf_kv(pdf, "Diversity", f"{dataset_quality.get('diversity_score', 0)}%")
    _pdf_kv(pdf, "Balance", f"{dataset_quality.get('balance_score', 0)}%")
    _pdf_kv(pdf, "Separation", f"{dataset_quality.get('separation_score', 0)}%")
    _pdf_kv(pdf, "Overall", f"{dataset_quality.get('overall_score', 0)}%")

    _pdf_section_title(pdf, "4. Label Distribution")
    label_counts = dataset_quality.get("label_counts", {}) or {}
    if label_counts:
        for label, count in list(label_counts.items())[:15]:
            safe_pdf_multicell(pdf, f"- {label}: {count}")
    else:
        safe_pdf_multicell(pdf, "No label distribution available.")

    reliability = snapshot.get("reliability_v2", {}) or {}
    if reliability:
        _pdf_section_title(pdf, "5. Reliability Engine 2.0")
        _pdf_kv(pdf, "Trust Score V2", f"{reliability.get('trust_score_v2', 0)}%")
        _pdf_kv(pdf, "Base reliability", f"{reliability.get('base_reliability_score', 0)}%")
        _pdf_kv(pdf, "Avg sensor value", f"{reliability.get('avg_sensor_value_score', 0)}%")
        _pdf_kv(pdf, "Hardware fit", f"{reliability.get('hardware_fit_score', 0)}%")
        _pdf_kv(pdf, "Real samples needed", reliability.get("total_real_samples_needed", 0))
        safe_pdf_multicell(pdf, reliability.get("decision", ""))

        class_risks = reliability.get("class_risks", []) or []
        if class_risks:
            _pdf_section_title(pdf, "6. Highest Class Risks")
            for row in class_risks[:10]:
                safe_pdf_multicell(pdf, f"{row.get('label')}: {row.get('risk_level')} risk - samples={row.get('samples')}, needed={row.get('recommended_real_samples_needed')}. {row.get('main_issue')}")

        sensor_scores = reliability.get("sensor_value_scores", []) or []
        if sensor_scores:
            _pdf_section_title(pdf, "7. Sensor Value Scores")
            for row in sensor_scores[:10]:
                safe_pdf_multicell(pdf, f"{row.get('sensor')}: {row.get('value_score')}% - {row.get('risk')}. {row.get('reason')}")

    real_bridge = snapshot.get("real_bridge", {}) or {}
    _pdf_section_title(pdf, "8. Synthetic-to-Real Evidence")
    _pdf_kv(pdf, "Real bridge used", "Yes" if real_bridge.get("used") else "No")
    _pdf_kv(pdf, "Similarity score", f"{real_bridge.get('similarity_score', 0)}%")
    _pdf_kv(pdf, "Real files", real_bridge.get("real_files", 0))
    safe_pdf_multicell(pdf, real_bridge.get("summary", ""))

    hardware = snapshot.get("hardware", {}) or {}
    deployment = snapshot.get("deployment", {}) or {}
    _pdf_section_title(pdf, "9. Hardware & Deployment Direction")
    _pdf_kv(pdf, "Recommended hardware", hardware.get("recommendation", "Unknown"))
    safe_pdf_multicell(pdf, hardware.get("reason", ""))
    _pdf_kv(pdf, "Deployment readiness", deployment.get("readiness", "Not generated"))
    _pdf_kv(pdf, "Communication", deployment.get("communication", "Unknown"))
    _pdf_kv(pdf, "Power source", deployment.get("power_source", "Unknown"))
    _pdf_kv(pdf, "Enclosure", deployment.get("enclosure", "Unknown"))
    cost = deployment.get("cost_estimate", {}) or {}
    if cost:
        _pdf_kv(pdf, "Budget estimate", f"EUR {cost.get('min_total_eur', 0)} - EUR {cost.get('max_total_eur', 0)}")

    risks = deployment.get("risks", []) or []
    if risks:
        _pdf_section_title(pdf, "10. Deployment Risks")
        for row in risks[:10]:
            safe_pdf_multicell(pdf, f"[{row.get('severity', 'info').upper()}] {row.get('risk')} - {row.get('mitigation')}")

    _pdf_section_title(pdf, "11. Commercial Positioning")
    _pdf_kv(pdf, "Suggested package", commercial.get("suggested_package", "Unknown"))
    _pdf_kv(pdf, "Suggested price range", commercial.get("suggested_price_range", "Unknown"))
    safe_pdf_cell(pdf, "Allowed claims", 8, True)
    for claim in commercial.get("allowed_claims", [])[:8]:
        safe_pdf_multicell(pdf, f"- {claim}")
    safe_pdf_cell(pdf, "Claims to avoid", 8, True)
    for claim in commercial.get("blocked_claims", [])[:8]:
        safe_pdf_multicell(pdf, f"- {claim}")

    _pdf_section_title(pdf, "12. Recommended Next Steps")
    for step in snapshot.get("next_steps", []):
        safe_pdf_multicell(pdf, f"- {step}")

    _pdf_section_title(pdf, "Validation Note")
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", REPORTS_DISCLAIMER))
    return safe_pdf_output(pdf)


def create_professional_report_bundle(project_name, snapshot, dataset_df=None, deployment_plan=None):
    """Export V23 professional report bundle with PDF, JSON, optional dataset and BOM."""
    snapshot = snapshot or {}
    pdf_bytes = generate_professional_report_pdf(snapshot)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("professional_report_v23.pdf", pdf_bytes)
        zf.writestr("professional_report_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr("executive_summary.json", json.dumps(_json_safe(snapshot.get("executive_summary", {})), indent=2, ensure_ascii=False))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df):
            zf.writestr("dataset_snapshot.csv", dataset_df.to_csv(index=False))
        if isinstance(deployment_plan, dict) and deployment_plan:
            bom = pd.DataFrame(deployment_plan.get("bom", []))
            if len(bom):
                zf.writestr("hardware_bom.csv", bom.to_csv(index=False))
            zf.writestr("deployment_summary.json", json.dumps(_json_safe(deployment_plan), indent=2, ensure_ascii=False))
        zf.writestr("README.txt", f"""EdgeTwin Studio V23 Professional Report Bundle

Project: {project_name}
Report type: {snapshot.get('report_type', 'Professional Report')}
Customer: {snapshot.get('customer_name', 'Customer')}
Readiness: {(snapshot.get('readiness') or {}).get('stage', 'Unknown')}
Go / No-Go: {(snapshot.get('readiness') or {}).get('go_no_go', 'Unknown')}
Suggested package: {(snapshot.get('commercial_positioning') or {}).get('suggested_package', 'Unknown')}

{snapshot.get('disclaimer', REPORTS_DISCLAIMER)}
""")
    return zip_buf.getvalue()

# ============================================================
# V24 - SAAS-LIGHT & MONETIZATION GATE
# ============================================================

MONETIZATION_DISCLAIMER = (
    "Monetization Gate is a commercial access and packaging guide for the current prototype. "
    "It does not process payments yet. Use it to test pricing, package levels and locked exports before connecting Stripe or another payment provider."
)

PRICING_PLANS = {
    "Free Demo": {
        "monthly_price_eur": 0,
        "description": "Self-selling demo access with limited previews and watermarked/basic outputs.",
        "project_limit": 1,
        "monthly_bundle_limit": 0,
        "real_upload_limit": 0,
        "api_calls_month": 0,
        "allowed_exports": ["training_csv_preview"],
        "locked_exports": ["auto_pilot_bundle", "fusion_bundle", "enterprise_bundle", "real_bridge_bundle", "deployment_bundle", "professional_report_bundle"],
        "best_for": "Visitors, demos, first trust-building page.",
    },
    "Starter Pilot": {
        "monthly_price_eur": 49,
        "one_time_price_eur": 49,
        "description": "Basic synthetic pilot package for one simple audio/vibration use-case.",
        "project_limit": 3,
        "monthly_bundle_limit": 3,
        "real_upload_limit": 0,
        "api_calls_month": 0,
        "allowed_exports": ["training_csv", "fusion_bundle", "basic_report"],
        "locked_exports": ["enterprise_bundle", "real_bridge_bundle", "deployment_bundle", "professional_report_bundle", "api_access"],
        "best_for": "Small prototype users who need a first dataset and basic audit.",
    },
    "Professional Pilot": {
        "monthly_price_eur": 149,
        "one_time_price_eur": 199,
        "description": "Full pilot bundle with dataset doctor, optimizer, reliability and professional report.",
        "project_limit": 10,
        "monthly_bundle_limit": 20,
        "real_upload_limit": 3,
        "api_calls_month": 0,
        "allowed_exports": ["training_csv", "fusion_bundle", "enterprise_bundle", "optimizer_bundle", "trust_bundle", "professional_report_bundle"],
        "locked_exports": ["api_access", "on_premise", "white_label"],
        "best_for": "Companies preparing a serious first pilot without custom engineering support.",
    },
    "Real-Data Pilot": {
        "monthly_price_eur": 299,
        "one_time_price_eur": 799,
        "description": "Real WAV/CSV upload, signal fingerprinting, synthetic-to-real bridge and deployment preparation.",
        "project_limit": 20,
        "monthly_bundle_limit": 40,
        "real_upload_limit": 25,
        "api_calls_month": 500,
        "allowed_exports": ["training_csv", "fusion_bundle", "enterprise_bundle", "optimizer_bundle", "trust_bundle", "real_bridge_bundle", "reliability_v2_bundle", "deployment_bundle", "professional_report_bundle"],
        "locked_exports": ["on_premise", "white_label"],
        "best_for": "Businesses with real sensor files who need a stronger pilot-ready package.",
    },
    "Enterprise": {
        "monthly_price_eur": None,
        "one_time_price_eur": None,
        "description": "Custom review, team usage, API/on-premise options and higher assurance deployment support.",
        "project_limit": 999,
        "monthly_bundle_limit": 999,
        "real_upload_limit": 999,
        "api_calls_month": 100000,
        "allowed_exports": ["training_csv", "fusion_bundle", "enterprise_bundle", "optimizer_bundle", "trust_bundle", "real_bridge_bundle", "reliability_v2_bundle", "deployment_bundle", "professional_report_bundle", "api_access", "on_premise", "white_label"],
        "locked_exports": [],
        "best_for": "Teams, regulated pilots, customers needing custom review or integration.",
    },
    "Founder Test Mode": {
        "monthly_price_eur": 0,
        "description": "Internal unlocked mode for building, demos and manual testing.",
        "project_limit": 999,
        "monthly_bundle_limit": 999,
        "real_upload_limit": 999,
        "api_calls_month": 999999,
        "allowed_exports": ["training_csv", "training_csv_preview", "fusion_bundle", "basic_report", "enterprise_bundle", "optimizer_bundle", "trust_bundle", "real_bridge_bundle", "reliability_v2_bundle", "deployment_bundle", "professional_report_bundle", "api_access"],
        "locked_exports": [],
        "best_for": "You, while building and testing EdgeTwin Studio.",
    },
}

EXPORT_PRODUCTS = {
    "training_csv_preview": {"label": "Training CSV preview", "minimum_plan": "Free Demo", "commercial_value": "Lead magnet / demo trust"},
    "training_csv": {"label": "Full Training CSV", "minimum_plan": "Starter Pilot", "commercial_value": "First usable dataset output"},
    "fusion_bundle": {"label": "Professional Fusion Bundle", "minimum_plan": "Starter Pilot", "commercial_value": "Dataset + manifest + PDF"},
    "enterprise_bundle": {"label": "Enterprise Audit Bundle", "minimum_plan": "Professional Pilot", "commercial_value": "Stronger audit output"},
    "optimizer_bundle": {"label": "Smart Optimizer Bundle", "minimum_plan": "Professional Pilot", "commercial_value": "Before/after improvement proof"},
    "trust_bundle": {"label": "Trust Center Bundle", "minimum_plan": "Professional Pilot", "commercial_value": "Go/No-Go and safe claims"},
    "real_bridge_bundle": {"label": "Synthetic-to-Real Bundle", "minimum_plan": "Real-Data Pilot", "commercial_value": "Real upload + signal fingerprint evidence"},
    "reliability_v2_bundle": {"label": "Reliability 2.0 Bundle", "minimum_plan": "Real-Data Pilot", "commercial_value": "Per-class and per-sensor risk"},
    "deployment_bundle": {"label": "Deployment Planner Bundle", "minimum_plan": "Real-Data Pilot", "commercial_value": "BOM + power + field plan"},
    "professional_report_bundle": {"label": "Professional Report 2.0 Bundle", "minimum_plan": "Professional Pilot", "commercial_value": "Consultancy-style paid report"},
    "api_access": {"label": "API Access", "minimum_plan": "Enterprise", "commercial_value": "Integration / automation"},
    "on_premise": {"label": "On-premise License", "minimum_plan": "Enterprise", "commercial_value": "High-ticket deployment"},
}

PACKAGE_PRICE_LADDER = [
    {"package": "Demo / Lead Magnet", "price_range": "Free", "when_to_offer": "When no dataset or no real data exists yet.", "promise": "Show the workflow and create trust."},
    {"package": "Starter Pilot Bundle", "price_range": "EUR 49 - 99", "when_to_offer": "Synthetic dataset, decent balance, simple use-case.", "promise": "First dataset and basic pilot direction."},
    {"package": "Professional Pilot Bundle", "price_range": "EUR 199 - 499", "when_to_offer": "Good Trust Score, usable audit and professional report.", "promise": "Pilot preparation package with risk analysis."},
    {"package": "Real-Data Pilot Bundle", "price_range": "EUR 799 - 1,500", "when_to_offer": "Customer uploads real WAV/CSV and needs synthetic-to-real evidence.", "promise": "Real-data-based pilot package, not production guarantee."},
    {"package": "Enterprise Custom Review", "price_range": "EUR 2,500 - 10,000+", "when_to_offer": "Production-sensitive customer, multiple assets, team/API/deployment needs.", "promise": "Custom validation review and deployment support."},
]


def get_pricing_plans():
    return list(PRICING_PLANS.keys())


def get_pricing_plan(plan_name):
    return PRICING_PLANS.get(plan_name, PRICING_PLANS["Free Demo"])


def get_export_products():
    return EXPORT_PRODUCTS.copy()


def plan_allows_export(plan_name, export_key):
    plan = get_pricing_plan(plan_name)
    return export_key in plan.get("allowed_exports", [])


def evaluate_export_access(plan_name, export_key, usage=None):
    usage = usage or {}
    plan = get_pricing_plan(plan_name)
    product = EXPORT_PRODUCTS.get(export_key, {"label": export_key, "minimum_plan": "Unknown", "commercial_value": ""})
    allowed = plan_allows_export(plan_name, export_key)

    reasons = []
    if not allowed:
        reasons.append(f"Requires {product.get('minimum_plan', 'a higher plan')}.")

    bundles_used = int(usage.get("bundles_used", 0))
    bundle_limit = plan.get("monthly_bundle_limit", 0)
    if allowed and bundle_limit is not None and bundles_used >= int(bundle_limit):
        allowed = False
        reasons.append("Monthly bundle limit reached.")

    real_uploads = int(usage.get("real_uploads", 0))
    real_limit = plan.get("real_upload_limit", 0)
    if export_key in ["real_bridge_bundle", "reliability_v2_bundle"] and allowed and real_limit is not None and real_uploads > int(real_limit):
        allowed = False
        reasons.append("Real upload limit reached.")

    return {
        "export_key": export_key,
        "label": product.get("label", export_key),
        "allowed": bool(allowed),
        "minimum_plan": product.get("minimum_plan", "Unknown"),
        "commercial_value": product.get("commercial_value", ""),
        "reason": "Allowed" if allowed else " ".join(reasons) if reasons else "Locked on this plan.",
    }


def detect_available_outputs(state_like=None, dataset_df=None):
    """Return a conservative list of outputs available from the current project state."""
    outputs = []
    if isinstance(dataset_df, pd.DataFrame) and len(dataset_df):
        outputs.append("training_csv")
    else:
        outputs.append("training_csv_preview")
    state_like = state_like or {}
    keys = {
        "fusion_bundle": "fusion_bundle",
        "enterprise_bundle": "enterprise_bundle",
        "optimizer_result": "optimizer_bundle",
        "trust_gate": "trust_bundle",
        "real_bridge_result": "real_bridge_bundle",
        "reliability_v2": "reliability_v2_bundle",
        "deployment_plan": "deployment_bundle",
        "professional_report_snapshot": "professional_report_bundle",
    }
    for state_key, export_key in keys.items():
        if state_like.get(state_key):
            outputs.append(export_key)
    return list(dict.fromkeys(outputs))


def recommend_package_level(dataset_df=None, trust_gate=None, reliability_v2=None, real_bridge_result=None, deployment_plan=None, professional_report_snapshot=None):
    has_dataset = isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0
    has_real = bool(real_bridge_result)
    has_deployment = bool(deployment_plan)
    has_report = bool(professional_report_snapshot)

    trust_score = 0
    stage = "Unknown"
    if isinstance(trust_gate, dict) and trust_gate:
        trust_score = int(trust_gate.get("trust_score", trust_gate.get("overall_score", 0)) or 0)
        stage = trust_gate.get("stage", trust_gate.get("readiness_stage", "Unknown"))
    if isinstance(reliability_v2, dict) and reliability_v2:
        trust_score = max(trust_score, int(reliability_v2.get("trust_score_v2", 0) or 0))

    if not has_dataset:
        return {
            "suggested_package": "Demo / Lead Magnet",
            "suggested_plan": "Free Demo",
            "suggested_price_range": "Free",
            "reason": "No usable dataset is loaded yet. Use the demo flow to create trust before charging.",
        }
    if has_real and has_deployment and trust_score >= 65:
        return {
            "suggested_package": "Real-Data Pilot Bundle",
            "suggested_plan": "Real-Data Pilot",
            "suggested_price_range": "EUR 799 - 1,500",
            "reason": "Real data, reliability scoring and deployment planning create a high-value pilot package.",
        }
    if has_real:
        return {
            "suggested_package": "Real-Data Mini Audit / Pilot Bundle",
            "suggested_plan": "Real-Data Pilot",
            "suggested_price_range": "EUR 299 - 799",
            "reason": "Real WAV/CSV analysis is more valuable than pure synthetic generation, even before full deployment planning.",
        }
    if has_report or trust_score >= 65 or stage in ["Pilot-ready", "Demo-ready"]:
        return {
            "suggested_package": "Professional Pilot Bundle",
            "suggested_plan": "Professional Pilot",
            "suggested_price_range": "EUR 199 - 499",
            "reason": "The project has enough structure, audit and trust evidence for a paid pilot-preparation package.",
        }
    return {
        "suggested_package": "Starter Pilot Bundle",
        "suggested_plan": "Starter Pilot",
        "suggested_price_range": "EUR 49 - 99",
        "reason": "The project is useful as a first dataset/export, but needs stronger trust/reliability proof before higher pricing.",
    }


def build_monetization_snapshot(
    project_name,
    selected_plan="Founder Test Mode",
    dataset_df=None,
    trust_gate=None,
    reliability_v2=None,
    real_bridge_result=None,
    deployment_plan=None,
    professional_report_snapshot=None,
    state_like=None,
    usage=None,
):
    plan = get_pricing_plan(selected_plan)
    state_like = state_like or {}
    if real_bridge_result:
        state_like = dict(state_like)
        state_like["real_bridge_result"] = real_bridge_result
    if deployment_plan:
        state_like = dict(state_like)
        state_like["deployment_plan"] = deployment_plan
    if professional_report_snapshot:
        state_like = dict(state_like)
        state_like["professional_report_snapshot"] = professional_report_snapshot
    if trust_gate:
        state_like = dict(state_like)
        state_like["trust_gate"] = trust_gate
    if reliability_v2:
        state_like = dict(state_like)
        state_like["reliability_v2"] = reliability_v2

    available_outputs = detect_available_outputs(state_like, dataset_df)
    access_matrix = [evaluate_export_access(selected_plan, k, usage) for k in EXPORT_PRODUCTS.keys()]
    available_access = [evaluate_export_access(selected_plan, k, usage) for k in available_outputs]
    recommendation = recommend_package_level(dataset_df, trust_gate, reliability_v2, real_bridge_result, deployment_plan, professional_report_snapshot)

    rows = int(len(dataset_df)) if isinstance(dataset_df, pd.DataFrame) else 0
    features = int(max(0, len([c for c in dataset_df.columns if c != "Label"]))) if isinstance(dataset_df, pd.DataFrame) and len(dataset_df.columns) else 0
    labels = int(dataset_df["Label"].nunique()) if isinstance(dataset_df, pd.DataFrame) and "Label" in dataset_df.columns else 0

    blocked = [a for a in available_access if not a.get("allowed")]
    unlocked = [a for a in available_access if a.get("allowed")]
    revenue_mode = "pay_per_bundle_first"
    if selected_plan in ["Professional Pilot", "Real-Data Pilot", "Enterprise"]:
        revenue_mode = "subscription_or_bundle"
    if selected_plan == "Free Demo":
        revenue_mode = "lead_generation"

    next_steps = []
    if selected_plan == "Free Demo":
        next_steps.append("Keep downloads limited and show upgrade CTA for Professional Pilot Bundle.")
    if not real_bridge_result:
        next_steps.append("Offer Real-Data Pilot upgrade when the customer has WAV/CSV files.")
    if not deployment_plan:
        next_steps.append("Run Deployment Planner before charging enterprise-level prices.")
    if not professional_report_snapshot:
        next_steps.append("Generate Reports 2.0 before sending a paid customer package.")
    next_steps.append("Do not promise production readiness unless field validation is completed.")

    snapshot = {
        "engine": "EdgeTwin Studio V24 SaaS-light & Monetization Gate",
        "created_at": _now(),
        "project_name": project_name,
        "selected_plan": selected_plan,
        "plan": plan,
        "revenue_mode": revenue_mode,
        "dataset_summary": {"rows": rows, "features": features, "labels": labels},
        "available_outputs": available_outputs,
        "available_access": available_access,
        "access_matrix": access_matrix,
        "blocked_available_outputs": blocked,
        "unlocked_available_outputs": unlocked,
        "package_recommendation": recommendation,
        "price_ladder": PACKAGE_PRICE_LADDER,
        "next_steps": next_steps,
        "safe_sales_positioning": [
            "Sell pilot preparation, not production guarantees.",
            "Charge more when real data, reliability evidence and deployment planning are included.",
            "Use Free Demo as lead generation; keep valuable ZIP/PDF exports behind paid tiers.",
            "For solo-founder execution, prefer pay-per-bundle before full SaaS subscriptions.",
        ],
        "disclaimer": MONETIZATION_DISCLAIMER,
    }
    return _json_safe(snapshot)


def generate_monetization_pdf(snapshot):
    snapshot = snapshot or {}
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt=clean_pdf_text("EdgeTwin Studio Monetization Gate"), ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("V24 SaaS-light & Commercial Access Plan"), ln=True, align="C")
    pdf.ln(8)

    safe_pdf_cell(pdf, "Project", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("project_name", "Unknown"))
    safe_pdf_cell(pdf, "Selected plan", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("selected_plan", "Unknown"))
    safe_pdf_cell(pdf, "Revenue mode", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("revenue_mode", "Unknown"))
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", MONETIZATION_DISCLAIMER))

    rec = snapshot.get("package_recommendation", {}) or {}
    pdf.ln(4)
    safe_pdf_cell(pdf, "Recommended Commercial Package", 8, True)
    safe_pdf_cell(pdf, f"Package: {rec.get('suggested_package', 'Unknown')}")
    safe_pdf_cell(pdf, f"Plan: {rec.get('suggested_plan', 'Unknown')}")
    safe_pdf_cell(pdf, f"Price range: {rec.get('suggested_price_range', 'Unknown')}")
    safe_pdf_multicell(pdf, f"Reason: {rec.get('reason', '')}")

    ds = snapshot.get("dataset_summary", {}) or {}
    pdf.ln(4)
    safe_pdf_cell(pdf, "Dataset Summary", 8, True)
    safe_pdf_cell(pdf, f"Rows: {ds.get('rows', 0)}")
    safe_pdf_cell(pdf, f"Features: {ds.get('features', 0)}")
    safe_pdf_cell(pdf, f"Labels: {ds.get('labels', 0)}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Available Output Access", 8, True)
    for row in snapshot.get("available_access", [])[:20]:
        status = "UNLOCKED" if row.get("allowed") else "LOCKED"
        safe_pdf_multicell(pdf, f"[{status}] {row.get('label')} - {row.get('reason')}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Price Ladder", 8, True)
    for row in snapshot.get("price_ladder", [])[:10]:
        safe_pdf_multicell(pdf, f"{row.get('package')}: {row.get('price_range')} - {row.get('when_to_offer')}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Next Steps", 8, True)
    for step in snapshot.get("next_steps", [])[:10]:
        safe_pdf_multicell(pdf, f"- {step}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Safe Sales Positioning", 8, True)
    for item in snapshot.get("safe_sales_positioning", [])[:10]:
        safe_pdf_multicell(pdf, f"- {item}")
    return safe_pdf_output(pdf)


def create_monetization_bundle(project_name, monetization_snapshot, dataset_df=None):
    snapshot = monetization_snapshot or {}
    pdf_bytes = generate_monetization_pdf(snapshot)
    access = pd.DataFrame(snapshot.get("access_matrix", []))
    price_ladder = pd.DataFrame(snapshot.get("price_ladder", []))
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("monetization_gate_v24.pdf", pdf_bytes)
        zf.writestr("monetization_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        if len(access):
            zf.writestr("export_access_matrix.csv", access.to_csv(index=False))
        if len(price_ladder):
            zf.writestr("price_ladder.csv", price_ladder.to_csv(index=False))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df):
            zf.writestr("dataset_snapshot.csv", dataset_df.head(2000).to_csv(index=False))
        rec = snapshot.get("package_recommendation", {}) or {}
        zf.writestr("README.txt", f"""EdgeTwin Studio V24 Monetization Gate Bundle

Project: {project_name}
Selected plan: {snapshot.get('selected_plan', 'Unknown')}
Suggested package: {rec.get('suggested_package', 'Unknown')}
Suggested price range: {rec.get('suggested_price_range', 'Unknown')}

This is a SaaS-light commercial planning bundle. Payments are not processed in V24.
Use this to test plan limits, locked exports and paid bundle positioning.

{snapshot.get('disclaimer', MONETIZATION_DISCLAIMER)}
""")
    return zip_buf.getvalue()




# ============================================================
# V24.1 PRODUCT HARDENING & VALIDATION SUITE
# ============================================================

PRODUCT_HARDENING_VERSION = "EdgeTwin Studio V24.1 - Product Hardening & Validation Suite"
MAX_SAFE_UPLOAD_MB = 50
MAX_SAFE_ROWS_FOR_UI = 50000
MIN_PUBLIC_DEMO_ROWS = 100
MIN_PAID_PILOT_ROWS = 250
MIN_REAL_DATA_PILOT_ROWS = 20

HARDENING_DISCLAIMER = (
    "This hardening scan is an engineering and product-readiness check. It does not certify production safety, "
    "regulatory compliance or model performance. Use it to reduce launch risk before paid pilots."
)


def _severity_score(severity):
    return {"pass": 0, "info": 1, "low": 2, "medium": 4, "high": 7, "critical": 10}.get(str(severity).lower(), 3)


def _status_from_score(score):
    score = int(np.clip(score, 0, 100))
    if score >= 88:
        return "Beta-ready"
    if score >= 74:
        return "Controlled pilot-ready"
    if score >= 60:
        return "Internal demo-ready"
    if score >= 40:
        return "Needs hardening"
    return "Not ready"


def _launch_risk_from_score(score):
    score = int(np.clip(score, 0, 100))
    if score >= 88:
        return "Low-Medium"
    if score >= 74:
        return "Medium"
    if score >= 60:
        return "Medium-High"
    return "High"


def validate_project_name(project_name):
    raw = str(project_name or "").strip()
    issues = []
    if not raw:
        issues.append({"area": "Project", "check": "Project name", "severity": "medium", "status": "warn", "message": "Project name is empty."})
    if len(raw) > 80:
        issues.append({"area": "Project", "check": "Project name length", "severity": "low", "status": "warn", "message": "Project name is long; keep customer-facing exports concise."})
    bad_chars = [c for c in raw if c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']]
    if bad_chars:
        issues.append({"area": "Project", "check": "Filename safety", "severity": "medium", "status": "warn", "message": "Project name contains characters that are unsafe for filenames."})
    if not issues:
        issues.append({"area": "Project", "check": "Project name", "severity": "pass", "status": "pass", "message": "Project name is usable for reports and bundles."})
    return issues


def validate_uploaded_file_metadata(filename, size_bytes=None, allowed_ext=None, max_mb=MAX_SAFE_UPLOAD_MB):
    allowed_ext = allowed_ext or [".csv", ".wav"]
    filename = str(filename or "").strip()
    issues = []
    lower = filename.lower()
    if not filename:
        issues.append({"area": "Upload", "check": "Filename", "severity": "high", "status": "fail", "message": "Upload has no filename."})
    elif not any(lower.endswith(ext) for ext in allowed_ext):
        issues.append({"area": "Upload", "check": "File extension", "severity": "high", "status": "fail", "message": f"Unsupported file type. Allowed: {', '.join(allowed_ext)}."})
    else:
        issues.append({"area": "Upload", "check": "File extension", "severity": "pass", "status": "pass", "message": "File extension is allowed."})
    if size_bytes is not None:
        try:
            mb = float(size_bytes) / (1024 * 1024)
            if mb > max_mb:
                issues.append({"area": "Upload", "check": "File size", "severity": "high", "status": "fail", "message": f"File is {mb:.1f} MB. Limit is {max_mb} MB for safe MVP operation."})
            else:
                issues.append({"area": "Upload", "check": "File size", "severity": "pass", "status": "pass", "message": f"File size is safe ({mb:.1f} MB)."})
        except Exception:
            issues.append({"area": "Upload", "check": "File size", "severity": "medium", "status": "warn", "message": "Could not inspect upload size."})
    return issues


def sanitize_customer_dataframe(df, label_col="Label", max_rows=MAX_SAFE_ROWS_FOR_UI):
    """Return a cleaned dataframe plus non-destructive warnings. Designed for customer uploads."""
    warnings = []
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(), [{"area": "Dataset", "check": "DataFrame", "severity": "critical", "status": "fail", "message": "Input is not a dataframe."}]
    clean = df.copy()
    clean.columns = [str(c).strip().replace(" ", "_")[:80] for c in clean.columns]
    if len(clean) > max_rows:
        warnings.append({"area": "Dataset", "check": "Row cap", "severity": "medium", "status": "warn", "message": f"Dataset has {len(clean)} rows. UI/validation should sample or cap at {max_rows} rows."})
    # Remove fully empty rows/columns, but keep user intent otherwise.
    before_rows, before_cols = clean.shape
    clean = clean.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if clean.shape != (before_rows, before_cols):
        warnings.append({"area": "Dataset", "check": "Empty rows/columns", "severity": "low", "status": "warn", "message": "Fully empty rows or columns were removed for safer analysis."})
    # Convert obvious numeric columns.
    for col in clean.columns:
        if col == label_col:
            continue
        if clean[col].dtype == object:
            converted = pd.to_numeric(clean[col], errors="coerce")
            if converted.notna().mean() >= 0.80:
                clean[col] = converted
    return clean, warnings


def dataset_safety_scan(dataset_df, label_col="Label"):
    issues = []
    summary = {
        "rows": 0,
        "columns": 0,
        "numeric_features": 0,
        "labels": 0,
        "missing_ratio": 0.0,
        "duplicate_ratio": 0.0,
        "min_class_count": 0,
        "max_class_count": 0,
        "weakest_class": None,
    }

    if not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0:
        issues.append({"area": "Dataset", "check": "Dataset loaded", "severity": "critical", "status": "fail", "message": "No dataset is loaded. Generate or upload a dataset before paid export."})
        return {"summary": summary, "issues": issues, "score": 0}

    df = dataset_df.copy()
    rows, cols = df.shape
    numeric_cols = _numeric_feature_columns(df, label_col) if isinstance(df, pd.DataFrame) else []
    summary.update({"rows": int(rows), "columns": int(cols), "numeric_features": int(len(numeric_cols))})

    if rows < MIN_PUBLIC_DEMO_ROWS:
        issues.append({"area": "Dataset", "check": "Sample count", "severity": "high", "status": "fail", "message": f"Only {rows} rows. Aim for at least {MIN_PUBLIC_DEMO_ROWS} rows even for public demos."})
    elif rows < MIN_PAID_PILOT_ROWS:
        issues.append({"area": "Dataset", "check": "Sample count", "severity": "medium", "status": "warn", "message": f"{rows} rows is usable for demo, but paid pilot bundles should usually have {MIN_PAID_PILOT_ROWS}+ rows."})
    else:
        issues.append({"area": "Dataset", "check": "Sample count", "severity": "pass", "status": "pass", "message": f"Dataset has {rows} rows."})

    if label_col not in df.columns:
        issues.append({"area": "Dataset", "check": "Label column", "severity": "critical", "status": "fail", "message": "Dataset has no Label column. Audit, ML validation and bundle claims will be weak."})
    else:
        counts = df[label_col].astype(str).value_counts()
        summary["labels"] = int(len(counts))
        if len(counts):
            summary["min_class_count"] = int(counts.min())
            summary["max_class_count"] = int(counts.max())
            summary["weakest_class"] = str(counts.idxmin())
        if len(counts) < 2:
            issues.append({"area": "Dataset", "check": "Class count", "severity": "critical", "status": "fail", "message": "Dataset has fewer than two classes."})
        elif counts.min() < 10:
            issues.append({"area": "Dataset", "check": "Weak class", "severity": "high", "status": "fail", "message": f"Weakest class has only {int(counts.min())} samples."})
        elif counts.min() / max(counts.max(), 1) < 0.35:
            issues.append({"area": "Dataset", "check": "Class balance", "severity": "high", "status": "fail", "message": "Class imbalance is severe. Paid reliability claims should be blocked."})
        elif counts.min() / max(counts.max(), 1) < 0.60:
            issues.append({"area": "Dataset", "check": "Class balance", "severity": "medium", "status": "warn", "message": "Class balance is acceptable for demo, but weak for paid pilot confidence."})
        else:
            issues.append({"area": "Dataset", "check": "Class balance", "severity": "pass", "status": "pass", "message": "Class distribution is reasonably balanced."})

    if len(numeric_cols) < 3:
        issues.append({"area": "Dataset", "check": "Numeric features", "severity": "high", "status": "fail", "message": "Too few numeric feature columns. Add extracted audio/vibration features."})
    else:
        issues.append({"area": "Dataset", "check": "Numeric features", "severity": "pass", "status": "pass", "message": f"Found {len(numeric_cols)} numeric feature columns."})

    missing_ratio = float(df.isna().mean().mean()) if rows and cols else 0.0
    duplicate_ratio = float(df.duplicated().mean()) if rows else 0.0
    summary["missing_ratio"] = round(missing_ratio, 4)
    summary["duplicate_ratio"] = round(duplicate_ratio, 4)

    if missing_ratio > 0.15:
        issues.append({"area": "Dataset", "check": "Missing values", "severity": "high", "status": "fail", "message": f"Missing value ratio is high ({missing_ratio:.1%})."})
    elif missing_ratio > 0.02:
        issues.append({"area": "Dataset", "check": "Missing values", "severity": "medium", "status": "warn", "message": f"Missing value ratio is {missing_ratio:.1%}; clean before paid customer export."})
    else:
        issues.append({"area": "Dataset", "check": "Missing values", "severity": "pass", "status": "pass", "message": "Missing values are low."})

    if duplicate_ratio > 0.20:
        issues.append({"area": "Dataset", "check": "Duplicate rows", "severity": "medium", "status": "warn", "message": f"Duplicate row ratio is {duplicate_ratio:.1%}; synthetic diversity may be too low."})
    else:
        issues.append({"area": "Dataset", "check": "Duplicate rows", "severity": "pass", "status": "pass", "message": "Duplicate row ratio is acceptable."})

    if len(numeric_cols):
        numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        inf_or_nan = float(numeric.isna().mean().mean())
        if inf_or_nan > 0.05:
            issues.append({"area": "Dataset", "check": "Numeric safety", "severity": "high", "status": "fail", "message": f"Numeric features contain too many NaN/inf values ({inf_or_nan:.1%})."})
        low_var = numeric.std(numeric_only=True).fillna(0)
        low_var_cols = low_var[low_var < 1e-9].index.tolist()
        if low_var_cols:
            issues.append({"area": "Dataset", "check": "Low-variance features", "severity": "medium", "status": "warn", "message": f"Low-variance features detected: {', '.join(low_var_cols[:8])}."})

    penalty = sum(_severity_score(i.get("severity")) for i in issues if i.get("severity") != "pass")
    score = int(np.clip(100 - penalty * 5, 0, 100))
    return {"summary": _json_safe(summary), "issues": _json_safe(issues), "score": score}


def build_release_readiness_checklist(has_real_data=False, has_deployment_plan=False, has_professional_report=False, has_monetization_gate=False, has_trust_gate=False):
    checks = []
    def add(category, check, ok, severity_if_missing, message_ok, message_missing):
        checks.append({
            "category": category,
            "check": check,
            "status": "pass" if ok else "missing",
            "severity": "pass" if ok else severity_if_missing,
            "message": message_ok if ok else message_missing,
        })

    add("Trust", "Trust Center generated", bool(has_trust_gate), "medium", "Trust gate exists.", "Run Trust Center before customer delivery.")
    add("Evidence", "Real data bridge", bool(has_real_data), "medium", "Real data evidence is present.", "No real-data bridge evidence yet; avoid production-readiness claims.")
    add("Deployment", "Deployment Planner", bool(has_deployment_plan), "medium", "Deployment plan exists.", "Run Deployment Planner before enterprise or hardware-related claims.")
    add("Reporting", "Reports 2.0", bool(has_professional_report), "medium", "Professional report snapshot exists.", "Generate Reports 2.0 before paid customer delivery.")
    add("Commercial", "Monetization Gate", bool(has_monetization_gate), "low", "Monetization gate exists.", "Run Monetization Gate before testing paid exports.")
    add("Claims", "Safe production wording", True, "critical", "Reports include field-validation disclaimers.", "Missing production disclaimer.")
    add("Solo-founder", "Pay-per-bundle first", True, "low", "Business model stays lightweight.", "Avoid heavy SaaS before validation.")
    return checks


def build_internal_benchmark_cases():
    """Small synthetic benchmark cases that prove the hardening scanner reacts sensibly."""
    cases = []
    # Good balanced case
    rows = []
    for label, offset in [("Normal", 0.0), ("Wear", 2.0), ("Critical", 5.0)]:
        for _ in range(80):
            rows.append({"Label": label, "RMS": np.random.normal(1 + offset, 0.15), "Kurtosis": np.random.normal(2 + offset, 0.20), "CrestFactor": np.random.normal(3 + offset, 0.25)})
    cases.append({"name": "Balanced separated dataset", "expected": "pass", "scan": dataset_safety_scan(pd.DataFrame(rows))})

    # Bad one-class case
    bad_rows = [{"Label": "Normal", "RMS": 1.0, "Kurtosis": 2.0, "CrestFactor": 3.0} for _ in range(40)]
    cases.append({"name": "One-class tiny dataset", "expected": "fail", "scan": dataset_safety_scan(pd.DataFrame(bad_rows))})

    # Missing label case
    no_label = pd.DataFrame({"RMS": np.random.normal(1, 0.1, 120), "Kurtosis": np.random.normal(2, 0.1, 120)})
    cases.append({"name": "Missing Label column", "expected": "fail", "scan": dataset_safety_scan(no_label)})

    out = []
    for case in cases:
        scan = case["scan"]
        critical = any(i.get("severity") == "critical" for i in scan.get("issues", []))
        high = any(i.get("severity") == "high" for i in scan.get("issues", []))
        observed = "fail" if critical or high or scan.get("score", 0) < 60 else "pass"
        out.append({
            "name": case["name"],
            "expected": case["expected"],
            "observed": observed,
            "passed": observed == case["expected"],
            "score": scan.get("score", 0),
            "critical_or_high_issues": int(sum(1 for i in scan.get("issues", []) if i.get("severity") in ["critical", "high"])),
        })
    return out


def run_product_hardening_suite(
    project_name,
    dataset_df=None,
    doctor=None,
    reliability_v2=None,
    trust_gate=None,
    deployment_plan=None,
    professional_report_snapshot=None,
    monetization_snapshot=None,
    has_real_data=False,
):
    dataset_scan = dataset_safety_scan(dataset_df)
    project_issues = validate_project_name(project_name)
    checklist = build_release_readiness_checklist(
        has_real_data=has_real_data,
        has_deployment_plan=bool(deployment_plan),
        has_professional_report=bool(professional_report_snapshot),
        has_monetization_gate=bool(monetization_snapshot),
        has_trust_gate=bool(trust_gate),
    )
    benchmark_cases = build_internal_benchmark_cases()

    doctor_score = int((doctor or {}).get("overall_score", 0)) if isinstance(doctor, dict) else 0
    rel_score = int((reliability_v2 or {}).get("trust_score_v2", (reliability_v2 or {}).get("reliability_score", 0))) if isinstance(reliability_v2, dict) else 0
    trust_score = int((trust_gate or {}).get("data_quality_score", 0)) if isinstance(trust_gate, dict) else 0
    dataset_score = int(dataset_scan.get("score", 0))
    checklist_penalty = sum(_severity_score(c.get("severity")) for c in checklist if c.get("severity") != "pass")
    project_penalty = sum(_severity_score(i.get("severity")) for i in project_issues if i.get("severity") != "pass")
    benchmark_pass_rate = int(round(100 * (sum(1 for c in benchmark_cases if c.get("passed")) / max(len(benchmark_cases), 1))))

    evidence_score = 0
    evidence_score += 20 if has_real_data else 0
    evidence_score += 20 if deployment_plan else 0
    evidence_score += 20 if professional_report_snapshot else 0
    evidence_score += 15 if monetization_snapshot else 0
    evidence_score += 15 if trust_gate else 0
    evidence_score += 10 if benchmark_pass_rate >= 100 else 0
    evidence_score = int(np.clip(evidence_score, 0, 100))

    # Weighted to reward safety and evidence, not only pretty dataset numbers.
    product_score = int(np.clip(
        dataset_score * 0.30 +
        max(doctor_score, dataset_score) * 0.15 +
        rel_score * 0.20 +
        evidence_score * 0.20 +
        benchmark_pass_rate * 0.10 -
        checklist_penalty * 1.5 -
        project_penalty,
        0,
        100,
    ))

    all_issues = []
    all_issues.extend(project_issues)
    all_issues.extend(dataset_scan.get("issues", []))
    all_issues.extend(checklist)
    blockers = [i for i in all_issues if i.get("severity") in ["critical", "high"] and i.get("status") != "pass"]

    safe_to_sell = "No"
    if product_score >= 74 and not any(i.get("severity") == "critical" for i in blockers):
        safe_to_sell = "Controlled paid pilot"
    elif product_score >= 60:
        safe_to_sell = "Demo / founder-led beta only"

    if product_score >= 88 and has_real_data and professional_report_snapshot and deployment_plan:
        recommended_next_action = "Start a small paid beta with strict field-validation wording."
    elif product_score >= 74:
        recommended_next_action = "Use for controlled paid pilot bundles, but keep production claims blocked."
    elif product_score >= 60:
        recommended_next_action = "Use for demos and feedback calls; fix blockers before paid delivery."
    else:
        recommended_next_action = "Do not sell yet. Generate a dataset, run Trust Center, Reports 2.0 and hardening fixes first."

    snapshot = {
        "engine": PRODUCT_HARDENING_VERSION,
        "created_at": _now(),
        "project_name": project_name,
        "product_readiness_score": product_score,
        "readiness_level": _status_from_score(product_score),
        "launch_risk": _launch_risk_from_score(product_score),
        "safe_to_sell": safe_to_sell,
        "recommended_next_action": recommended_next_action,
        "score_breakdown": {
            "dataset_safety_score": dataset_score,
            "doctor_score": doctor_score,
            "reliability_or_trust_score": rel_score,
            "evidence_score": evidence_score,
            "benchmark_pass_rate": benchmark_pass_rate,
            "checklist_penalty": checklist_penalty,
        },
        "dataset_scan": dataset_scan,
        "project_checks": project_issues,
        "release_checklist": checklist,
        "benchmark_cases": benchmark_cases,
        "blockers": blockers,
        "hardening_rules": [
            "No paid reliability claim when the dataset has critical validation failures.",
            "No production-ready claim without real field data and documented deployment validation.",
            "Real-data uploads must remain customer-owned and should be processed with clear retention rules.",
            "Paid bundles should include PDF, JSON manifest, CSV snapshot and explicit limitations.",
            "Keep SaaS-light/pay-per-bundle until repeat usage is proven.",
        ],
        "known_limitations": [
            "Synthetic-to-real similarity is still an engineering estimate until benchmarked with many real datasets.",
            "Hardware latency and battery estimates are planning estimates, not certified measurements.",
            "Reports are pilot-preparation outputs, not safety certifications.",
            "Public deployment still needs hosting security, backups, rate limits and payment-provider integration.",
        ],
        "disclaimer": HARDENING_DISCLAIMER,
    }
    return _json_safe(snapshot)


def generate_hardening_pdf(snapshot):
    snapshot = snapshot or {}
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt=clean_pdf_text("EdgeTwin Studio Product Hardening Report"), ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("V24.1 Validation Suite"), ln=True, align="C")
    pdf.ln(8)

    safe_pdf_cell(pdf, "Executive Hardening Summary", 8, True)
    safe_pdf_cell(pdf, f"Project: {snapshot.get('project_name', 'Unknown')}")
    safe_pdf_cell(pdf, f"Product readiness: {snapshot.get('product_readiness_score', 0)}%")
    safe_pdf_cell(pdf, f"Readiness level: {snapshot.get('readiness_level', 'Unknown')}")
    safe_pdf_cell(pdf, f"Launch risk: {snapshot.get('launch_risk', 'Unknown')}")
    safe_pdf_cell(pdf, f"Safe to sell: {snapshot.get('safe_to_sell', 'Unknown')}")
    safe_pdf_multicell(pdf, f"Recommended action: {snapshot.get('recommended_next_action', '')}")
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", HARDENING_DISCLAIMER))

    breakdown = snapshot.get("score_breakdown", {}) or {}
    pdf.ln(4)
    safe_pdf_cell(pdf, "Score Breakdown", 8, True)
    for k, v in breakdown.items():
        safe_pdf_cell(pdf, f"{str(k).replace('_', ' ').title()}: {v}")

    ds = snapshot.get("dataset_scan", {}).get("summary", {}) if isinstance(snapshot.get("dataset_scan"), dict) else {}
    pdf.ln(4)
    safe_pdf_cell(pdf, "Dataset Safety Summary", 8, True)
    for key in ["rows", "columns", "numeric_features", "labels", "min_class_count", "weakest_class", "missing_ratio", "duplicate_ratio"]:
        safe_pdf_cell(pdf, f"{key.replace('_', ' ').title()}: {ds.get(key, '')}")

    blockers = snapshot.get("blockers", []) or []
    pdf.ln(4)
    safe_pdf_cell(pdf, "Blockers", 8, True)
    if blockers:
        for item in blockers[:15]:
            safe_pdf_multicell(pdf, f"[{item.get('severity', '').upper()}] {item.get('area', item.get('category', ''))} / {item.get('check', '')}: {item.get('message', '')}")
    else:
        safe_pdf_multicell(pdf, "No critical/high blockers detected by V24.1 hardening scan.")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Release Checklist", 8, True)
    for item in snapshot.get("release_checklist", [])[:20]:
        safe_pdf_multicell(pdf, f"[{item.get('status', '').upper()}] {item.get('category', '')} - {item.get('check', '')}: {item.get('message', '')}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Internal Benchmark Cases", 8, True)
    for case in snapshot.get("benchmark_cases", [])[:10]:
        safe_pdf_multicell(pdf, f"{case.get('name')}: expected {case.get('expected')}, observed {case.get('observed')}, passed={case.get('passed')}, score={case.get('score')}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Known Limitations", 8, True)
    for item in snapshot.get("known_limitations", [])[:12]:
        safe_pdf_multicell(pdf, f"- {item}")
    return safe_pdf_output(pdf)


def create_product_hardening_bundle(project_name, hardening_snapshot, dataset_df=None):
    snapshot = hardening_snapshot or {}
    pdf_bytes = generate_hardening_pdf(snapshot)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("product_hardening_report_v24_1.pdf", pdf_bytes)
        zf.writestr("hardening_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        issues = []
        if isinstance(snapshot.get("dataset_scan"), dict):
            issues.extend(snapshot.get("dataset_scan", {}).get("issues", []))
        issues.extend(snapshot.get("project_checks", []) or [])
        issues.extend(snapshot.get("release_checklist", []) or [])
        if issues:
            zf.writestr("hardening_issues.csv", pd.DataFrame(issues).to_csv(index=False))
        if snapshot.get("benchmark_cases"):
            zf.writestr("internal_benchmark_cases.csv", pd.DataFrame(snapshot.get("benchmark_cases", [])).to_csv(index=False))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df):
            zf.writestr("dataset_snapshot.csv", dataset_df.head(3000).to_csv(index=False))
        zf.writestr("README.txt", f"""EdgeTwin Studio V24.1 Product Hardening Bundle

Project: {project_name}
Product readiness score: {snapshot.get('product_readiness_score', 0)}%
Readiness level: {snapshot.get('readiness_level', 'Unknown')}
Launch risk: {snapshot.get('launch_risk', 'Unknown')}
Safe to sell: {snapshot.get('safe_to_sell', 'Unknown')}

Recommended next action:
{snapshot.get('recommended_next_action', '')}

This bundle is for internal validation and controlled beta preparation.
{snapshot.get('disclaimer', HARDENING_DISCLAIMER)}
""")
    return zip_buf.getvalue()


# ============================================================
# V24.2 BETA LAUNCH READINESS
# ============================================================

BETA_LAUNCH_VERSION = "EdgeTwin Studio V24.2 - Beta Launch Readiness"
BETA_LAUNCH_DISCLAIMER = (
    "Beta launch readiness is a commercial and operational readiness estimate. "
    "It does not certify model performance, field safety or production suitability. "
    "Use controlled beta wording and require real-world validation before deployment claims."
)

BETA_TARGET_SEGMENTS = {
    "Predictive maintenance teams": {
        "pain": "Need a faster way to plan vibration/acoustic Edge AI pilots without waiting for perfect failure data.",
        "promise": "Generate a pilot package with dataset, reliability score, weak classes, hardware direction and field-test plan.",
        "best_demo": "Predictive Maintenance Demo",
        "best_paid_offer": "Real-Data Pilot Bundle",
    },
    "Construction / asset security": {
        "pain": "Need to detect tamper, drilling, grinding, impact and asset handling with limited labeled data.",
        "promise": "Turn a security use-case into a sensor-fusion pilot bundle and professional risk report.",
        "best_demo": "Construction Security Demo",
        "best_paid_offer": "Professional Pilot Bundle",
    },
    "Remote asset / forestry operators": {
        "pain": "Need low-power remote monitoring for chainsaw, vehicle, movement or theft risk without constant connectivity.",
        "promise": "Generate a remote monitoring pilot plan with sensor choices, LoRa/gateway direction and reliability limits.",
        "best_demo": "Smart Forestry Demo",
        "best_paid_offer": "Deployment Pilot Bundle",
    },
    "Embedded / TinyML developers": {
        "pain": "Need structured sensor data and feature guidance before moving to Edge Impulse, TinyML or firmware work.",
        "promise": "Create clean training features, audit reports, hardware fit scores and export bundles for fast prototyping.",
        "best_demo": "Custom Sensor Fusion",
        "best_paid_offer": "Professional Pilot Bundle",
    },
}


def get_beta_target_segments():
    return list(BETA_TARGET_SEGMENTS.keys())


def get_beta_target_segment(segment):
    return BETA_TARGET_SEGMENTS.get(segment, BETA_TARGET_SEGMENTS["Predictive maintenance teams"])


def _bool_score(value, points):
    return int(points if value else 0)


def build_beta_package_cards():
    return [
        {
            "package": "Free Demo",
            "price_hint": "€0",
            "for": "First impression / lead generation",
            "includes": "Demo scenarios, limited preview, watermarked outputs, no real-data bridge export.",
            "locked_reason": "Pro reports, real-data bridge and deployment bundles stay locked.",
            "claim_level": "Concept/demo only",
        },
        {
            "package": "Starter Pilot Bundle",
            "price_hint": "€49 - €99",
            "for": "Small teams testing a first sensor idea",
            "includes": "Synthetic dataset, basic audit, CSV/JSON, basic PDF summary.",
            "locked_reason": "No production readiness, no deep real-data bridge.",
            "claim_level": "Prototype preparation",
        },
        {
            "package": "Professional Pilot Bundle",
            "price_hint": "€149 - €299",
            "for": "Companies preparing a serious pilot",
            "includes": "Dataset, Dataset Doctor, Reliability Engine, Trust Center, professional report and ZIP bundle.",
            "locked_reason": "Production claims require real field validation.",
            "claim_level": "Pilot preparation",
        },
        {
            "package": "Real-Data Pilot Bundle",
            "price_hint": "€499 - €999",
            "for": "Customers with WAV/CSV sensor recordings",
            "includes": "Signal Fingerprint, Synthetic-to-Real Bridge, similarity score, weak label risks and real-samples-needed advice.",
            "locked_reason": "Still not a safety certification; field trial required.",
            "claim_level": "Controlled pilot evidence",
        },
        {
            "package": "Enterprise Review",
            "price_hint": "€1,500+",
            "for": "Higher-value clients needing deployment decisions",
            "includes": "Professional report, deployment planner, hardware BOM, risks, field validation plan and founder review.",
            "locked_reason": "Use founder-led delivery until repeatable playbook is proven.",
            "claim_level": "Decision-support package",
        },
    ]


def build_beta_feedback_questions(target_segment="Predictive maintenance teams"):
    segment = get_beta_target_segment(target_segment)
    return [
        {"question": "What sensor/use-case are you trying to validate?", "why_it_matters": "Confirms target problem and industry pack fit."},
        {"question": "Do you already have WAV/CSV sensor data?", "why_it_matters": "Determines if Real Bridge should be the main paid offer."},
        {"question": "Which output would you pay for: dataset, report, hardware plan, or real-data audit?", "why_it_matters": "Tests willingness to pay for bundle layers."},
        {"question": "Was the reliability/risk wording clear and trustworthy?", "why_it_matters": "Trust is the core commercial moat."},
        {"question": "Which score or recommendation would you need before starting a field pilot?", "why_it_matters": "Guides V25/V26 prioritization."},
        {"question": f"For {target_segment}, does this solve the pain: {segment.get('pain')}?", "why_it_matters": "Checks message-market fit."},
        {"question": "Would you share this report with a technical manager or buyer?", "why_it_matters": "Validates report quality and sales usefulness."},
        {"question": "What price feels reasonable for the generated pilot bundle?", "why_it_matters": "Validates price ladder before Stripe."},
    ]


def build_beta_landing_page_sections(target_segment="Predictive maintenance teams"):
    seg = get_beta_target_segment(target_segment)
    return [
        {
            "section": "Hero",
            "headline": "Start your Edge AI sensor pilot automatically.",
            "body": "EdgeTwin Studio turns a sensor use-case into a pilot-ready package: dataset, audit, reliability score, hardware advice and professional report.",
        },
        {
            "section": "Who it is for",
            "headline": target_segment,
            "body": seg.get("pain", "Teams that need a faster path from sensor idea to Edge AI pilot."),
        },
        {
            "section": "What you get",
            "headline": "Dataset + risk audit + hardware plan + report",
            "body": "Generate synthetic or real-data-based sensor datasets with clear weak-class, sensor-value and field-readiness advice.",
        },
        {
            "section": "Why trust it",
            "headline": "Built around honest readiness scoring",
            "body": "EdgeTwin separates demo-ready, pilot-ready and production-risk wording so customers do not overclaim synthetic results.",
        },
        {
            "section": "Next step",
            "headline": "Run a controlled beta pilot",
            "body": f"Recommended first offer for this segment: {seg.get('best_paid_offer', 'Professional Pilot Bundle')}.",
        },
    ]


def build_beta_launch_checklist(has_dataset=False, has_trust=False, has_real_bridge=False, has_deployment=False, has_report=False, has_hardening=False, has_monetization=False):
    rows = []
    def add(area, check, ok, severity, next_step):
        rows.append({
            "area": area,
            "check": check,
            "status": "ready" if ok else "not_ready",
            "severity_if_missing": "pass" if ok else severity,
            "next_step": "Ready" if ok else next_step,
        })
    add("Product", "Dataset exists", has_dataset, "critical", "Run demo, wizard, industry pack or real bridge first.")
    add("Trust", "Trust Center generated", has_trust, "high", "Run Trust Center before showing paid output.")
    add("Evidence", "Real Bridge used when real data exists", has_real_bridge, "medium", "For higher-value beta, upload real WAV/CSV files.")
    add("Deployment", "Deployment Planner generated", has_deployment, "medium", "Run Deployment Planner for hardware-oriented buyers.")
    add("Sales", "Reports 2.0 generated", has_report, "high", "Generate Professional Report before beta call.")
    add("Safety", "Product Hardening scan generated", has_hardening, "high", "Run Product Hardening to catch blockers.")
    add("Monetization", "Pricing/access logic generated", has_monetization, "medium", "Run Monetization Gate before paid beta tests.")
    add("Positioning", "Use pilot-only claims", True, "critical", "Keep production claims blocked without field validation.")
    add("Solo-founder", "Founder-led beta first", True, "low", "Avoid public self-serve until 3-5 beta calls validate messaging.")
    return rows


def build_beta_launch_snapshot(
    project_name,
    target_segment="Predictive maintenance teams",
    beta_mode="Founder-led beta",
    dataset_df=None,
    trust_gate=None,
    reliability_v2=None,
    real_bridge_result=None,
    deployment_plan=None,
    professional_report_snapshot=None,
    hardening_snapshot=None,
    monetization_snapshot=None,
):
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    has_dataset = len(dataset_df) > 0
    has_trust = isinstance(trust_gate, dict) and len(trust_gate) > 0
    has_reliability = isinstance(reliability_v2, dict) and len(reliability_v2) > 0
    has_real_bridge = isinstance(real_bridge_result, dict) and len(real_bridge_result) > 0
    has_deployment = isinstance(deployment_plan, dict) and len(deployment_plan) > 0
    has_report = isinstance(professional_report_snapshot, dict) and len(professional_report_snapshot) > 0
    has_hardening = isinstance(hardening_snapshot, dict) and len(hardening_snapshot) > 0
    has_monetization = isinstance(monetization_snapshot, dict) and len(monetization_snapshot) > 0

    checklist = build_beta_launch_checklist(has_dataset, has_trust, has_real_bridge, has_deployment, has_report, has_hardening, has_monetization)
    missing_penalty = sum(_severity_score(r.get("severity_if_missing")) for r in checklist if r.get("status") != "ready")

    dataset_scan = dataset_safety_scan(dataset_df) if has_dataset else {"score": 0, "issues": [], "summary": {}}
    dataset_score = int(dataset_scan.get("score", 0))
    trust_score = int((trust_gate or {}).get("data_quality_score", 0)) if has_trust else 0
    reliability_score = int((reliability_v2 or {}).get("trust_score_v2", (reliability_v2 or {}).get("reliability_score", 0))) if has_reliability else 0
    hardening_score = int((hardening_snapshot or {}).get("product_readiness_score", 0)) if has_hardening else 0
    monetization_score = 85 if has_monetization else 35
    evidence_score = 0
    evidence_score += _bool_score(has_dataset, 20)
    evidence_score += _bool_score(has_trust, 15)
    evidence_score += _bool_score(has_reliability, 15)
    evidence_score += _bool_score(has_real_bridge, 20)
    evidence_score += _bool_score(has_deployment, 10)
    evidence_score += _bool_score(has_report, 10)
    evidence_score += _bool_score(has_hardening, 10)
    evidence_score = int(np.clip(evidence_score, 0, 100))

    beta_score = int(np.clip(
        dataset_score * 0.18 +
        max(trust_score, reliability_score) * 0.22 +
        hardening_score * 0.20 +
        monetization_score * 0.10 +
        evidence_score * 0.22 +
        8 - missing_penalty * 1.15,
        0,
        100,
    ))

    blockers = [r for r in checklist if r.get("status") != "ready" and r.get("severity_if_missing") in ["critical", "high"]]
    if beta_score >= 82 and not blockers:
        readiness = "Paid beta ready"
        launch_action = "Invite 3-5 controlled beta users and sell founder-led pilot bundles with strict validation wording."
    elif beta_score >= 68:
        readiness = "Founder-led beta ready"
        launch_action = "Run demo calls and founder-led beta, but fix high-severity blockers before paid self-serve."
    elif beta_score >= 50:
        readiness = "Demo-only beta"
        launch_action = "Use for demos and feedback only; generate Trust Center, Reports 2.0 and Hardening before charging."
    else:
        readiness = "Not launch ready"
        launch_action = "Do not launch. Generate a dataset, report, trust snapshot and hardening scan first."

    seg = get_beta_target_segment(target_segment)
    package_cards = build_beta_package_cards()
    feedback_questions = build_beta_feedback_questions(target_segment)
    landing_sections = build_beta_landing_page_sections(target_segment)

    demo_script = [
        "1. Start with the customer problem, not the technology.",
        f"2. Use segment pain: {seg.get('pain')}",
        f"3. Run recommended demo/use-case: {seg.get('best_demo')}",
        "4. Show Dataset Doctor, Trust Center and Reliability Engine before exports.",
        "5. Explain that pilot-ready is not production-ready.",
        "6. Open Reports 2.0 and show the professional PDF bundle.",
        "7. Offer the smallest paid next step: a controlled pilot bundle or real-data audit.",
    ]

    watermark_policy = {
        "free_demo": "Watermark PDFs and limit CSV rows to preview size.",
        "starter": "Allow CSV/JSON and basic report, but lock Real Bridge and Deployment Bundle.",
        "professional": "Unlock Trust, Reliability and Reports 2.0 bundle.",
        "real_data_pilot": "Unlock Signal Fingerprint, Synthetic-to-Real Bridge and per-class real samples needed.",
        "enterprise": "Founder-led delivery with deployment planner, BOM and limitations review.",
    }

    paid_unlock_preparation = [
        "Keep all premium export buttons behind plan checks before adding Stripe.",
        "Store export events with user_id, project_id, package, timestamp and bundle type.",
        "Use watermarked PDFs for Free Demo outputs.",
        "Never unlock production wording based on payment; only evidence can improve readiness.",
        "Start with manual invoices or private beta codes before public self-serve payments.",
    ]

    snapshot = {
        "engine": BETA_LAUNCH_VERSION,
        "created_at": _now(),
        "project_name": project_name,
        "target_segment": target_segment,
        "beta_mode": beta_mode,
        "beta_readiness_score": beta_score,
        "readiness": readiness,
        "launch_action": launch_action,
        "blockers": blockers,
        "score_breakdown": {
            "dataset_score": dataset_score,
            "trust_score": trust_score,
            "reliability_score": reliability_score,
            "hardening_score": hardening_score,
            "monetization_score": monetization_score,
            "evidence_score": evidence_score,
            "missing_penalty": missing_penalty,
        },
        "target_segment_profile": seg,
        "landing_page_sections": landing_sections,
        "package_cards": package_cards,
        "demo_script": demo_script,
        "feedback_questions": feedback_questions,
        "launch_checklist": checklist,
        "watermark_policy": watermark_policy,
        "paid_unlock_preparation": paid_unlock_preparation,
        "beta_rules": [
            "Start founder-led: talk to beta users before public self-serve.",
            "Sell output bundles, not a complicated SaaS subscription, until repeat usage is proven.",
            "Use real-data bridge as the premium trust layer.",
            "Keep production claims blocked without field validation.",
            "Every paid bundle must include limitations, next steps and a field-validation recommendation.",
        ],
        "safe_public_tagline": "Start your Edge AI sensor pilot automatically.",
        "safe_positioning": "EdgeTwin Studio helps prepare and de-risk Edge AI sensor pilots; it does not certify production performance.",
        "disclaimer": BETA_LAUNCH_DISCLAIMER,
    }
    return _json_safe(snapshot)


def generate_beta_launch_pdf(snapshot):
    snapshot = snapshot or {}
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt=clean_pdf_text("EdgeTwin Studio Beta Launch Readiness"), ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("V24.2 - Launch Kit"), ln=True, align="C")
    pdf.ln(8)

    safe_pdf_cell(pdf, "Launch Summary", 8, True)
    safe_pdf_cell(pdf, f"Project: {snapshot.get('project_name', 'Unknown')}")
    safe_pdf_cell(pdf, f"Target segment: {snapshot.get('target_segment', 'Unknown')}")
    safe_pdf_cell(pdf, f"Beta mode: {snapshot.get('beta_mode', 'Unknown')}")
    safe_pdf_cell(pdf, f"Beta readiness score: {snapshot.get('beta_readiness_score', 0)}%")
    safe_pdf_cell(pdf, f"Readiness: {snapshot.get('readiness', 'Unknown')}")
    safe_pdf_multicell(pdf, f"Launch action: {snapshot.get('launch_action', '')}")
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", BETA_LAUNCH_DISCLAIMER))

    pdf.ln(4)
    safe_pdf_cell(pdf, "Customer-facing Positioning", 8, True)
    safe_pdf_cell(pdf, f"Tagline: {snapshot.get('safe_public_tagline', '')}")
    safe_pdf_multicell(pdf, snapshot.get("safe_positioning", ""))

    pdf.ln(4)
    safe_pdf_cell(pdf, "Landing Page Sections", 8, True)
    for section in snapshot.get("landing_page_sections", [])[:8]:
        safe_pdf_multicell(pdf, f"{section.get('section')}: {section.get('headline')} - {section.get('body')}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Demo Script", 8, True)
    for item in snapshot.get("demo_script", [])[:10]:
        safe_pdf_multicell(pdf, f"- {item}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Package Cards", 8, True)
    for card in snapshot.get("package_cards", [])[:10]:
        safe_pdf_multicell(pdf, f"{card.get('package')} ({card.get('price_hint')}): {card.get('includes')} Claim level: {card.get('claim_level')}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Launch Checklist", 8, True)
    for item in snapshot.get("launch_checklist", [])[:15]:
        safe_pdf_multicell(pdf, f"[{item.get('status')}] {item.get('area')} - {item.get('check')}: {item.get('next_step')}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Beta Feedback Questions", 8, True)
    for item in snapshot.get("feedback_questions", [])[:12]:
        safe_pdf_multicell(pdf, f"- {item.get('question')} ({item.get('why_it_matters')})")

    blockers = snapshot.get("blockers", []) or []
    pdf.ln(4)
    safe_pdf_cell(pdf, "Blockers", 8, True)
    if blockers:
        for item in blockers[:12]:
            safe_pdf_multicell(pdf, f"[{item.get('severity_if_missing', '').upper()}] {item.get('area')} - {item.get('check')}: {item.get('next_step')}")
    else:
        safe_pdf_multicell(pdf, "No high/critical beta launch blockers detected by V24.2.")
    return safe_pdf_output(pdf)


def create_beta_launch_bundle(project_name, beta_snapshot, dataset_df=None):
    snapshot = beta_snapshot or {}
    pdf_bytes = generate_beta_launch_pdf(snapshot)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("beta_launch_readiness_report_v24_2.pdf", pdf_bytes)
        zf.writestr("beta_launch_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        if snapshot.get("package_cards"):
            zf.writestr("package_cards.csv", pd.DataFrame(snapshot.get("package_cards", [])).to_csv(index=False))
        if snapshot.get("launch_checklist"):
            zf.writestr("launch_checklist.csv", pd.DataFrame(snapshot.get("launch_checklist", [])).to_csv(index=False))
        if snapshot.get("feedback_questions"):
            zf.writestr("beta_feedback_questions.csv", pd.DataFrame(snapshot.get("feedback_questions", [])).to_csv(index=False))
        if snapshot.get("landing_page_sections"):
            zf.writestr("landing_page_copy.csv", pd.DataFrame(snapshot.get("landing_page_sections", [])).to_csv(index=False))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df):
            zf.writestr("dataset_preview_for_beta.csv", dataset_df.head(1000).to_csv(index=False))
        zf.writestr("README.txt", f"""EdgeTwin Studio V24.2 Beta Launch Readiness Bundle

Project: {project_name}
Target segment: {snapshot.get('target_segment', 'Unknown')}
Beta readiness score: {snapshot.get('beta_readiness_score', 0)}%
Readiness: {snapshot.get('readiness', 'Unknown')}

Launch action:
{snapshot.get('launch_action', '')}

Safe tagline:
{snapshot.get('safe_public_tagline', '')}

Safe positioning:
{snapshot.get('safe_positioning', '')}

{snapshot.get('disclaimer', BETA_LAUNCH_DISCLAIMER)}
""")
    return zip_buf.getvalue()


# ============================================================
# V25 API AUTOMATION
# ============================================================

API_AUTOMATION_VERSION = "EdgeTwin Studio V25 - API Automation"
API_AUTOMATION_DISCLAIMER = (
    "API Automation is an integration-readiness layer for controlled beta and enterprise workflows. "
    "It does not certify model performance, production safety or deployment suitability. "
    "All API outputs must keep pilot/field-validation wording unless validated with real deployment evidence."
)

API_ENDPOINT_CATALOG = {
    "/v1/signal/analyze": {
        "name": "Analyze Signal",
        "method": "POST",
        "minimum_plan": "Professional",
        "purpose": "Extract vibration/audio signal features or inspect uploaded signal metadata.",
        "returns": "RMS, Std, Kurtosis, CrestFactor, ZCR and spectral features.",
        "risk": "Low",
    },
    "/v1/dataset/generate": {
        "name": "Generate Pilot Dataset",
        "method": "POST",
        "minimum_plan": "Professional",
        "purpose": "Generate a use-case based pilot dataset and manifest.",
        "returns": "Dataset preview, label counts, manifest and recommended training columns.",
        "risk": "Medium",
    },
    "/v1/dataset/audit": {
        "name": "Audit Dataset",
        "method": "POST",
        "minimum_plan": "Starter",
        "purpose": "Run Dataset Doctor on numeric features and labels.",
        "returns": "Diversity, balance, separation, overall score and advice.",
        "risk": "Low",
    },
    "/v1/reliability/score": {
        "name": "Reliability Score",
        "method": "POST",
        "minimum_plan": "Professional",
        "purpose": "Calculate reliability, field readiness and risk levels.",
        "returns": "Reliability scores, risk verdict and real-samples-needed guidance.",
        "risk": "Medium",
    },
    "/v1/hardware/advice": {
        "name": "Hardware Advice",
        "method": "POST",
        "minimum_plan": "Professional",
        "purpose": "Recommend edge/gateway hardware based on feature count, sample rate and target.",
        "returns": "Recommended board, ranking, latency/RAM estimates and reason.",
        "risk": "Low",
    },
    "/v1/deployment/plan": {
        "name": "Deployment Plan",
        "method": "POST",
        "minimum_plan": "Enterprise",
        "purpose": "Create deployment direction with BOM, communications, maintenance and field validation plan.",
        "returns": "Deployment plan summary and implementation risks.",
        "risk": "Medium-High",
    },
    "/v1/report/generate": {
        "name": "Generate Report",
        "method": "POST",
        "minimum_plan": "Enterprise",
        "purpose": "Generate a professional report bundle from current evidence snapshots.",
        "returns": "Report manifest and bundle-ready metadata. Actual PDF generation should be async in production.",
        "risk": "Medium",
    },
}

API_PLAN_ORDER = ["Free Demo", "Starter", "Professional", "Real-Data Pilot", "Enterprise", "Founder Test Mode"]


def safe_json_dumps(obj, indent=2):
    return json.dumps(_json_safe(obj), indent=indent, ensure_ascii=False)


def parse_json_payload(payload_text):
    if not payload_text or not str(payload_text).strip():
        return {}
    return json.loads(payload_text)


def _plan_rank(plan_name):
    if plan_name not in API_PLAN_ORDER:
        return 0
    return API_PLAN_ORDER.index(plan_name)


def _plan_allows(minimum_plan, selected_plan):
    if selected_plan == "Founder Test Mode":
        return True
    return _plan_rank(selected_plan) >= _plan_rank(minimum_plan)


def get_api_endpoint_catalog():
    rows = []
    for endpoint, meta in API_ENDPOINT_CATALOG.items():
        row = dict(meta)
        row["endpoint"] = endpoint
        rows.append(row)
    return rows


def validate_api_key(api_key):
    """Return user dict for a valid key. This is local SQLite validation, not production auth."""
    if not api_key:
        return None
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, username, api_key FROM users WHERE api_key = ?", (api_key,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "api_key": row[2]}


def record_api_usage(user_id, endpoint):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO api_usage (id, user_id, endpoint, created_at) VALUES (?, ?, ?, ?)",
        (str(uuid.uuid4()), user_id, endpoint, _now()),
    )
    conn.commit()
    conn.close()


def get_api_usage_count(user_id, since_iso=None):
    conn = sqlite3.connect(DB_NAME)
    if since_iso:
        df = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM api_usage WHERE user_id = ? AND created_at >= ?",
            conn,
            params=(user_id, since_iso),
        )
    else:
        df = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM api_usage WHERE user_id = ?",
            conn,
            params=(user_id,),
        )
    conn.close()
    return int(df["n"].iloc[0]) if len(df) else 0


def get_api_sample_payload(endpoint):
    samples = {
        "/v1/signal/analyze": {
            "sample_rate": 16000,
            "label": "Normal",
            "signal": [0.0, 0.12, -0.04, 0.19, -0.08, 0.03, 0.01, -0.02],
            "note": "For production, send either signal values or upload a WAV/CSV through a controlled file endpoint.",
        },
        "/v1/dataset/generate": {
            "use_case_type": "Predictive Maintenance",
            "project_goal": "Detect early machine wear from vibration/audio features.",
            "selected_sensors": ["Audio", "Vibration"],
            "environment": "Industrial",
            "labels_text": "Healthy, Early_Wear, Wear, Critical_Failure",
            "samples": 300,
            "has_real_data": False,
            "output_level": "Professional Pilot Bundle",
            "priority": "performance",
        },
        "/v1/dataset/audit": {
            "use_current_project_dataset": True,
            "label_column": "Label",
            "note": "API beta can audit the current project dataset or a JSON records dataset.",
        },
        "/v1/reliability/score": {
            "has_real_data": False,
            "selected_sensors": ["Audio", "Vibration"],
            "use_current_project_dataset": True,
        },
        "/v1/hardware/advice": {
            "num_features": 8,
            "sample_rate": 16000,
            "target": "balanced",
        },
        "/v1/deployment/plan": {
            "use_case_type": "Remote Asset Monitoring",
            "environment": "Remote asset",
            "selected_sensors": ["Audio", "Vibration", "Radar"],
            "sample_rate": 16000,
            "target": "low_power",
            "nodes": 5,
        },
        "/v1/report/generate": {
            "report_type": "professional_pilot_report",
            "include_dataset_snapshot": True,
            "include_deployment_plan": True,
            "claim_level": "pilot preparation",
        },
    }
    return samples.get(endpoint, {})


def _df_from_payload_or_current(payload, dataset_df=None):
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    records = payload.get("records") or payload.get("dataset")
    if records:
        return pd.DataFrame(records)
    if payload.get("use_current_project_dataset", True):
        return dataset_df.copy()
    return pd.DataFrame()


def simulate_api_endpoint(endpoint, payload=None, dataset_df=None, project_name="API_Project", selected_plan="Founder Test Mode"):
    payload = payload or {}
    meta = API_ENDPOINT_CATALOG.get(endpoint)
    if not meta:
        return {"ok": False, "error": "Unknown endpoint", "endpoint": endpoint}
    if not _plan_allows(meta.get("minimum_plan", "Enterprise"), selected_plan):
        return {
            "ok": False,
            "error": "Plan does not allow this endpoint",
            "endpoint": endpoint,
            "minimum_plan": meta.get("minimum_plan"),
            "selected_plan": selected_plan,
        }

    try:
        if endpoint == "/v1/signal/analyze":
            sr = int(payload.get("sample_rate", 16000))
            sig = payload.get("signal")
            if not sig:
                d = generate_universal_signal(1.0, sr, 80, 0.2, 1.0, 0.05)
                sig = d["sig"]
            features = extract_signal_features(sig, sr, payload.get("label"))
            return {"ok": True, "endpoint": endpoint, "features": _json_safe(features), "disclaimer": RELIABILITY_DISCLAIMER}

        if endpoint == "/v1/dataset/generate":
            config = build_use_case_config(
                use_case_type=payload.get("use_case_type", "Predictive Maintenance"),
                project_goal=payload.get("project_goal", "Generate an Edge AI pilot dataset."),
                selected_sensors=payload.get("selected_sensors", ["Audio", "Vibration"]),
                environment=payload.get("environment", "Industrial"),
                labels_text=payload.get("labels_text", "Healthy, Early_Wear, Wear, Critical_Failure"),
                samples=int(min(max(payload.get("samples", 300), 50), 2000)),
                has_real_data=bool(payload.get("has_real_data", False)),
                output_level=payload.get("output_level", "Professional Pilot Bundle"),
                priority=payload.get("priority", "balanced"),
            )
            result = run_auto_pilot_project(config)
            return {
                "ok": True,
                "endpoint": endpoint,
                "manifest": _json_safe(result.get("manifest", {})),
                "doctor": _json_safe(result.get("doctor", {})),
                "reliability": _json_safe(result.get("reliability", {})),
                "hardware": {"recommendation": result.get("hardware", {}).get("recommendation"), "reason": result.get("hardware", {}).get("reason")},
                "dataset_preview": _json_safe(result.get("training_df", pd.DataFrame()).head(10)),
                "disclaimer": RELIABILITY_DISCLAIMER,
            }

        if endpoint == "/v1/dataset/audit":
            df = _df_from_payload_or_current(payload, dataset_df)
            label_col = payload.get("label_column", "Label")
            if df.empty or label_col not in df.columns:
                return {"ok": False, "endpoint": endpoint, "error": "Dataset needs a Label column or records payload."}
            numeric = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
            if not numeric:
                return {"ok": False, "endpoint": endpoint, "error": "No numeric feature columns found."}
            audit = dataset_doctor(df[numeric], df[label_col])
            return {"ok": True, "endpoint": endpoint, "audit": _json_safe(audit), "rows": int(len(df)), "features": numeric}

        if endpoint == "/v1/reliability/score":
            df = _df_from_payload_or_current(payload, dataset_df)
            if df.empty or "Label" not in df.columns:
                doctor = {"overall_score": 0, "separation_score": 0, "balance_score": 0, "advice": []}
            else:
                if all(c in df.columns for c in ["AudioScore", "VibrationScore", "GasScore", "RadarScore", "GPSZoneScore"]):
                    doctor = fusion_dataset_doctor(df, payload.get("template"), selected_sensors=payload.get("selected_sensors"))
                else:
                    numeric = [c for c in df.columns if c != "Label" and pd.api.types.is_numeric_dtype(df[c])]
                    doctor = dataset_doctor(df[numeric], df["Label"]) if numeric else {"overall_score": 0, "separation_score": 0, "balance_score": 0, "advice": []}
            reliability = calculate_reliability_score(doctor, bool(payload.get("has_real_data", False)), payload.get("selected_sensors", []))
            return {"ok": True, "endpoint": endpoint, "reliability": _json_safe(reliability), "doctor_summary": _json_safe(doctor), "disclaimer": RELIABILITY_DISCLAIMER}

        if endpoint == "/v1/hardware/advice":
            hw = hardware_auto_architect(
                int(payload.get("num_features", 8)),
                int(payload.get("sample_rate", 16000)),
                payload.get("target", "balanced"),
            )
            return {"ok": True, "endpoint": endpoint, "hardware": _json_safe(hw)}

        if endpoint == "/v1/deployment/plan":
            # Avoid heavy app coupling: produce a practical skeleton using available hardware advice.
            hw = hardware_auto_architect(
                int(payload.get("num_features", 8)),
                int(payload.get("sample_rate", 16000)),
                payload.get("target", "balanced"),
            )
            plan = {
                "project_name": project_name,
                "use_case_type": payload.get("use_case_type", "Custom"),
                "environment": payload.get("environment", "Custom"),
                "selected_sensors": payload.get("selected_sensors", []),
                "nodes": int(payload.get("nodes", 1)),
                "recommended_hardware": hw.get("recommendation"),
                "communication_direction": "LoRa or LTE for remote assets; WiFi/MQTT or Ethernet for indoor/gateway deployments.",
                "field_validation_plan": ["Collect real baseline data", "Collect event samples", "Run Reliability Engine", "Do not claim production readiness before field validation."],
                "deployment_risks": ["Insufficient real samples", "Sensor mounting variation", "Power budget uncertainty", "Connectivity gaps"],
                "disclaimer": API_AUTOMATION_DISCLAIMER,
            }
            return {"ok": True, "endpoint": endpoint, "deployment_plan": _json_safe(plan)}

        if endpoint == "/v1/report/generate":
            return {
                "ok": True,
                "endpoint": endpoint,
                "report_status": "blueprint_ready",
                "recommended_pattern": "Generate PDFs asynchronously in production and return a signed download URL.",
                "report_type": payload.get("report_type", "professional_pilot_report"),
                "claim_level": payload.get("claim_level", "pilot preparation"),
                "disclaimer": API_AUTOMATION_DISCLAIMER,
            }

        return {"ok": False, "endpoint": endpoint, "error": "Endpoint not implemented in simulator."}
    except Exception as exc:
        return {"ok": False, "endpoint": endpoint, "error": str(exc)}


def build_api_automation_snapshot(
    project_name,
    selected_plan="Founder Test Mode",
    api_mode="Local blueprint",
    dataset_df=None,
    trust_gate=None,
    reliability_v2=None,
    real_bridge_result=None,
    deployment_plan=None,
    hardening_snapshot=None,
    beta_launch_snapshot=None,
    monetization_snapshot=None,
    state_like=None,
):
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    state_like = state_like or {}
    plan = get_pricing_plan(selected_plan) if "get_pricing_plan" in globals() else {}
    catalog = []
    for row in get_api_endpoint_catalog():
        minimum = row.get("minimum_plan", "Enterprise")
        allowed = _plan_allows(minimum, selected_plan)
        row = dict(row)
        row["allowed_on_selected_plan"] = bool(allowed)
        row["status"] = "available" if allowed else "locked"
        catalog.append(row)

    has_dataset = isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0 and "Label" in dataset_df.columns
    has_trust = bool(trust_gate)
    has_reliability = bool(reliability_v2)
    has_real_bridge = bool(real_bridge_result)
    has_deployment = bool(deployment_plan)
    has_hardening = bool(hardening_snapshot)
    has_beta = bool(beta_launch_snapshot)
    has_monetization = bool(monetization_snapshot)

    evidence_score = 0
    evidence_score += 15 if has_dataset else 0
    evidence_score += 15 if has_trust else 0
    evidence_score += 15 if has_reliability else 0
    evidence_score += 15 if has_real_bridge else 0
    evidence_score += 10 if has_deployment else 0
    evidence_score += 15 if has_hardening else 0
    evidence_score += 8 if has_beta else 0
    evidence_score += 7 if has_monetization else 0
    plan_score = 100 if selected_plan in ["Founder Test Mode", "Enterprise"] else (75 if selected_plan in ["Real-Data Pilot", "Professional"] else 45)
    mode_score = {"Local blueprint": 60, "Private beta API": 75, "Enterprise/on-premise": 85}.get(api_mode, 60)
    api_docs_score = 82
    hardening_score = int(hardening_snapshot.get("product_readiness_score", 0)) if has_hardening else 35

    api_score = int(np.clip(evidence_score * 0.34 + plan_score * 0.16 + mode_score * 0.16 + api_docs_score * 0.14 + hardening_score * 0.20, 0, 100))
    locked = [r for r in catalog if not r.get("allowed_on_selected_plan")]
    available = [r for r in catalog if r.get("allowed_on_selected_plan")]
    if api_score >= 82 and has_hardening:
        risk = "Low-Medium"
        verdict = "API layer is ready for controlled private beta or enterprise blueprint use. Keep rate limits and validation wording."
        access_level = "Private beta ready"
    elif api_score >= 65:
        risk = "Medium"
        verdict = "API layer is useful as a blueprint and internal simulator. Fix hardening and evidence gaps before paid integrations."
        access_level = "Internal/API blueprint"
    else:
        risk = "High"
        verdict = "Do not expose a public API yet. Use documentation and simulator only until dataset, trust and hardening evidence exists."
        access_level = "Documentation only"

    implementation_checklist = [
        "Keep API keys hashed or stored securely before production.",
        "Add request size limits for WAV/CSV/JSON payloads.",
        "Add per-user and per-endpoint rate limiting.",
        "Log request metadata, not raw sensitive sensor data by default.",
        "Return trace_id/request_id for every response.",
        "Make report/PDF generation asynchronous for large jobs.",
        "Use signed download URLs for generated bundles.",
        "Validate all CSV schemas before running audits.",
        "Never return production-ready claims from synthetic-only evidence.",
        "Create admin usage dashboard before public launch.",
        "Rotate API keys and support revocation.",
        "Put API behind HTTPS/reverse proxy when deployed.",
    ]
    api_safety_rules = [
        "Every response must include a pilot/field-validation disclaimer for reliability and deployment outputs.",
        "Synthetic-only requests cannot unlock production-ready wording.",
        "Real-data endpoints should require explicit customer consent and retention policy.",
        "Large files should be processed asynchronously, not inside a single blocking request.",
        "Premium endpoints should be gated by plan and usage limits.",
        "Safety-critical use-cases require human review before deployment recommendations are used.",
    ]
    sample_requests = {r["endpoint"]: get_api_sample_payload(r["endpoint"]) for r in catalog}

    snapshot = {
        "engine": API_AUTOMATION_VERSION,
        "created_at": _now(),
        "project_name": project_name,
        "selected_plan": selected_plan,
        "api_mode": api_mode,
        "api_readiness_score": api_score,
        "access_level": access_level,
        "integration_risk": risk,
        "api_verdict": verdict,
        "plan_limits": plan,
        "endpoint_catalog": catalog,
        "available_endpoints": available,
        "locked_endpoints": locked,
        "score_breakdown": {
            "evidence_score": evidence_score,
            "plan_score": plan_score,
            "mode_score": mode_score,
            "api_docs_score": api_docs_score,
            "hardening_score": hardening_score,
        },
        "evidence_flags": {
            "has_dataset": has_dataset,
            "has_trust": has_trust,
            "has_reliability_v2": has_reliability,
            "has_real_bridge": has_real_bridge,
            "has_deployment_plan": has_deployment,
            "has_hardening": has_hardening,
            "has_beta_launch": has_beta,
            "has_monetization": has_monetization,
        },
        "api_safety_rules": api_safety_rules,
        "implementation_checklist": implementation_checklist,
        "sample_requests": sample_requests,
        "recommended_next_step": "Use this API layer in private beta first. Add FastAPI/Stripe only after hardening, rate limiting and usage logging are proven.",
        "safe_positioning": "EdgeTwin Studio API automates pilot preparation, dataset audit, reliability estimates and hardware advice. It does not certify production deployment.",
        "disclaimer": API_AUTOMATION_DISCLAIMER,
    }
    return _json_safe(snapshot)


def generate_api_docs_markdown(snapshot):
    snapshot = snapshot or {}
    lines = []
    lines.append("### EdgeTwin Studio API Blueprint")
    lines.append("")
    lines.append(f"**Mode:** {snapshot.get('api_mode', 'Unknown')}  ")
    lines.append(f"**Access level:** {snapshot.get('access_level', 'Unknown')}  ")
    lines.append(f"**Readiness:** {snapshot.get('api_readiness_score', 0)}%  ")
    lines.append("")
    lines.append("**Base URL for future deployment:** `/api` or your private domain.")
    lines.append("")
    lines.append("**Authentication:** `Authorization: Bearer <api_key>`")
    lines.append("")
    lines.append("| Endpoint | Method | Minimum plan | Purpose |")
    lines.append("|---|---:|---|---|")
    for e in snapshot.get("endpoint_catalog", []):
        lines.append(f"| `{e.get('endpoint')}` | {e.get('method')} | {e.get('minimum_plan')} | {e.get('purpose')} |")
    lines.append("")
    lines.append("**Important:** all reliability/deployment responses must include field-validation disclaimers.")
    return "\n".join(lines)


def generate_fastapi_skeleton(snapshot):
    return '''"""
EdgeTwin Studio V25 FastAPI skeleton
For private beta only. Add production auth, rate limits, file limits, HTTPS and async jobs before public use.
"""

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

import core

app = FastAPI(title="EdgeTwin Studio API", version="v25-private-beta")

class APIRequest(BaseModel):
    payload: Dict[str, Any] = {}


def require_api_key(authorization: Optional[str]):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    api_key = authorization.replace("Bearer ", "", 1).strip()
    user = core.validate_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user


@app.post("/v1/{group}/{action}")
def dispatch(group: str, action: str, request: APIRequest, authorization: Optional[str] = Header(default=None)):
    user = require_api_key(authorization)
    endpoint = f"/v1/{group}/{action}"
    response = core.simulate_api_endpoint(endpoint, request.payload, selected_plan="Enterprise")
    core.record_api_usage(user["id"], endpoint)
    if not response.get("ok"):
        raise HTTPException(status_code=400, detail=response)
    return response
'''


def generate_api_automation_pdf(snapshot):
    snapshot = snapshot or {}
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt=clean_pdf_text("EdgeTwin Studio API Automation"), ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("V25 - Integration Blueprint"), ln=True, align="C")
    pdf.ln(8)
    safe_pdf_cell(pdf, "API Readiness Summary", 8, True)
    safe_pdf_cell(pdf, f"Project: {snapshot.get('project_name', 'Unknown')}")
    safe_pdf_cell(pdf, f"Selected plan: {snapshot.get('selected_plan', 'Unknown')}")
    safe_pdf_cell(pdf, f"API mode: {snapshot.get('api_mode', 'Unknown')}")
    safe_pdf_cell(pdf, f"Readiness score: {snapshot.get('api_readiness_score', 0)}%")
    safe_pdf_cell(pdf, f"Access level: {snapshot.get('access_level', 'Unknown')}")
    safe_pdf_cell(pdf, f"Integration risk: {snapshot.get('integration_risk', 'Unknown')}")
    safe_pdf_multicell(pdf, snapshot.get("api_verdict", ""))
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", API_AUTOMATION_DISCLAIMER))

    pdf.ln(4)
    safe_pdf_cell(pdf, "Endpoint Catalog", 8, True)
    for e in snapshot.get("endpoint_catalog", [])[:15]:
        safe_pdf_multicell(pdf, f"{e.get('method')} {e.get('endpoint')} - {e.get('name')} ({e.get('minimum_plan')}): {e.get('purpose')}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "API Safety Rules", 8, True)
    for item in snapshot.get("api_safety_rules", [])[:12]:
        safe_pdf_multicell(pdf, f"- {item}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Implementation Checklist", 8, True)
    for item in snapshot.get("implementation_checklist", [])[:15]:
        safe_pdf_multicell(pdf, f"- {item}")
    return safe_pdf_output(pdf)


def create_api_automation_bundle(project_name, api_snapshot, dataset_df=None):
    snapshot = api_snapshot or {}
    zip_buf = io.BytesIO()
    pdf_bytes = generate_api_automation_pdf(snapshot)
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("api_automation_report_v25.pdf", pdf_bytes)
        zf.writestr("api_automation_snapshot.json", safe_json_dumps(snapshot, indent=2))
        zf.writestr("api_docs.md", generate_api_docs_markdown(snapshot))
        zf.writestr("fastapi_private_beta_skeleton.py", generate_fastapi_skeleton(snapshot))
        zf.writestr("endpoint_catalog.csv", pd.DataFrame(snapshot.get("endpoint_catalog", [])).to_csv(index=False))
        sample_requests = snapshot.get("sample_requests", {})
        for endpoint, payload in sample_requests.items():
            safe_name = endpoint.strip("/").replace("/", "_") or "root"
            zf.writestr(f"sample_request_{safe_name}.json", safe_json_dumps(payload, indent=2))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df):
            zf.writestr("dataset_preview_for_api.csv", dataset_df.head(1000).to_csv(index=False))
        zf.writestr("README.txt", f"""EdgeTwin Studio V25 API Automation Bundle

Project: {project_name}
API readiness score: {snapshot.get('api_readiness_score', 0)}%
Access level: {snapshot.get('access_level', 'Unknown')}
Integration risk: {snapshot.get('integration_risk', 'Unknown')}

Recommended next step:
{snapshot.get('recommended_next_step', '')}

This bundle is intended for private beta/API blueprint use. Do not expose a public API without authentication hardening, rate limiting, file limits, HTTPS, usage logging and async job handling.

{snapshot.get('disclaimer', API_AUTOMATION_DISCLAIMER)}
""")
    return zip_buf.getvalue()


# ============================================================
# V26 - INDUSTRY PACK MARKETPLACE / PACK BUILDER
# ============================================================

PACK_MARKETPLACE_DISCLAIMER = (
    "Industry Pack Marketplace is a curated commercial packaging layer for Edge AI pilot preparation. "
    "Packs provide reusable templates, labels, signal settings, reports and deployment guidance. "
    "They do not certify production performance; field validation with customer data remains required."
)

MARKETPLACE_PACKS = {
    "bearing_wear_predictive_maintenance": {
        "name": "Bearing Wear Predictive Maintenance Pack",
        "tier": "Professional",
        "minimum_plan": "Professional Pilot",
        "price_range": "EUR 199 - 499",
        "target_market": "Predictive maintenance",
        "description": "Focused vibration/audio pilot pack for motors, bearings, pumps and rotating machinery.",
        "sensors": ["Vibration", "Audio", "Temperature"],
        "classes": ["Normal", "Unbalance", "Misalignment", "Bearing_Wear", "Critical_Failure"],
        "sample_rate": 4000,
        "base_pack": "Rotating Machinery Pack",
        "commercial_angle": "Reduce trial-and-error in early predictive maintenance pilots.",
        "evidence_needed": "Real vibration samples per class, operating RPM/context and maintenance history.",
    },
    "acoustic_tamper_security": {
        "name": "Acoustic Tamper Security Pack",
        "tier": "Professional",
        "minimum_plan": "Professional Pilot",
        "price_range": "EUR 199 - 499",
        "target_market": "Security / tamper",
        "description": "Audio/vibration pack for construction sites, containers, facades and asset tamper detection.",
        "sensors": ["Audio", "Vibration", "Radar", "GPS / Zone"],
        "classes": ["Normal_Background", "Impact_Tamper", "Grinding", "Drilling", "Vehicle_Nearby"],
        "sample_rate": 16000,
        "base_pack": "Security & Tamper Pack",
        "commercial_angle": "Generate a credible first pilot for intrusion, tool-use and tamper events.",
        "evidence_needed": "Real site ambience, drilling/grinding/impact examples and false-positive scenarios.",
    },
    "forestry_remote_threat": {
        "name": "Smart Forestry Remote Threat Pack",
        "tier": "Real-Data Pilot",
        "minimum_plan": "Real-Data Pilot",
        "price_range": "EUR 799 - 1,500",
        "target_market": "Forestry / remote area",
        "description": "Remote outdoor audio/radar/GPS fusion pack for chainsaw, vehicle and human-activity pilots.",
        "sensors": ["Audio", "Radar", "GPS / Zone", "Vibration"],
        "classes": ["Forest_Normal", "Chainsaw", "Offroad_Vehicle", "Human_Movement", "Rain_And_Wind"],
        "sample_rate": 16000,
        "base_pack": "Forestry & Remote Asset Pack",
        "commercial_angle": "Turn a remote monitoring idea into a field-testable pilot package.",
        "evidence_needed": "Local forest ambience, weather samples, vehicle/chainsaw recordings and zone context.",
    },
    "audio_vibration_fusion_core": {
        "name": "Audio + Vibration Fusion Core Pack",
        "tier": "Starter/Professional",
        "minimum_plan": "Starter Pilot",
        "price_range": "EUR 49 - 299",
        "target_market": "Remote assets",
        "description": "Universal audio/vibration fusion pack for early concept validation across machines, assets and security.",
        "sensors": ["Audio", "Vibration"],
        "classes": ["Normal", "Suspicious_Audio", "Suspicious_Vibration", "Confirmed_Event", "Critical_Event"],
        "sample_rate": 16000,
        "base_pack": None,
        "commercial_angle": "A simple, reusable starter pack for customers who do not know their exact labels yet.",
        "evidence_needed": "At least a few representative normal/event samples before production claims.",
    },
    "agricultural_machinery_monitoring": {
        "name": "Agricultural Machinery Monitoring Pack",
        "tier": "Professional",
        "minimum_plan": "Professional Pilot",
        "price_range": "EUR 199 - 699",
        "target_market": "Agriculture",
        "description": "Audio/vibration pack for tractors, attachments, pumps and farm machinery pilot monitoring.",
        "sensors": ["Vibration", "Audio", "GPS / Zone", "Temperature"],
        "classes": ["Normal", "Rough_Running", "Loose_Attachment", "Impact", "Failure_Risk"],
        "sample_rate": 4000,
        "base_pack": "Rotating Machinery Pack",
        "commercial_angle": "Help farms and machinery yards start condition/security monitoring without a custom engineering project.",
        "evidence_needed": "Machine type, RPM/use mode, normal samples and rough-running/impact examples.",
    },
}


def _slugify_pack_name(value):
    value = str(value or "custom_pack").lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "custom_pack"


def get_marketplace_pack(pack_id):
    return MARKETPLACE_PACKS.get(pack_id)


def get_marketplace_pack_catalog():
    rows = []
    for pack_id, pack in MARKETPLACE_PACKS.items():
        rows.append({
            "pack_id": pack_id,
            "name": pack.get("name"),
            "tier": pack.get("tier"),
            "minimum_plan": pack.get("minimum_plan"),
            "price_range": pack.get("price_range"),
            "target_market": pack.get("target_market"),
            "sensors": ", ".join(pack.get("sensors", [])),
            "classes": ", ".join(pack.get("classes", [])),
            "sample_rate": pack.get("sample_rate"),
            "description": pack.get("description"),
        })
    return rows


def _marketplace_pack_fit(pack, target_market, selected_plan, dataset_df=None, trust_gate=None, reliability_v2=None, real_bridge_result=None, deployment_plan=None, state_like=None):
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    state_like = state_like or {}
    score = 45
    reasons = []
    if str(pack.get("target_market", "")).lower().split(" /")[0] in str(target_market or "").lower():
        score += 22
        reasons.append("Matches selected target market.")
    if len(dataset_df) > 0:
        score += 8
        reasons.append("A current dataset exists for preview/audit.")
    if trust_gate:
        score += 6
        reasons.append("Trust Center evidence is available.")
    if reliability_v2:
        score += 8
        reasons.append("Reliability Engine 2.0 evidence is available.")
    if real_bridge_result:
        score += 10
        reasons.append("Real-data bridge evidence increases paid-pack credibility.")
    elif pack.get("minimum_plan") == "Real-Data Pilot":
        score -= 12
        reasons.append("Real-data pack should be validated with uploads before paid claims.")
    if deployment_plan:
        score += 5
        reasons.append("Deployment plan/BOM exists.")
    allowed = True
    access = evaluate_export_access(selected_plan, "professional_report_bundle")
    if selected_plan == "Free Demo" and pack.get("minimum_plan") != "Free Demo":
        allowed = False
        score -= 8
        reasons.append("Current plan should show this as locked/upgrade pack.")
    if pack.get("minimum_plan") == "Real-Data Pilot" and selected_plan in ["Free Demo", "Starter Pilot", "Professional Pilot"]:
        allowed = False
        score -= 4
        reasons.append("Requires Real-Data Pilot or higher for full value.")
    return int(np.clip(score, 0, 100)), allowed, reasons


def recommend_marketplace_packs(target_market="Predictive maintenance", selected_plan="Founder Test Mode", dataset_df=None, trust_gate=None, reliability_v2=None, real_bridge_result=None, deployment_plan=None, state_like=None):
    rows = []
    for pack_id, pack in MARKETPLACE_PACKS.items():
        score, allowed, reasons = _marketplace_pack_fit(pack, target_market, selected_plan, dataset_df, trust_gate, reliability_v2, real_bridge_result, deployment_plan, state_like)
        rows.append({
            "pack_id": pack_id,
            "name": pack.get("name"),
            "fit_score": score,
            "status": "Unlocked" if allowed else "Locked / upgrade",
            "minimum_plan": pack.get("minimum_plan"),
            "price_range": pack.get("price_range"),
            "target_market": pack.get("target_market"),
            "why": " ".join(reasons[:3]) if reasons else "General reusable pack.",
        })
    rows = sorted(rows, key=lambda r: r["fit_score"], reverse=True)
    return {
        "target_market": target_market,
        "selected_plan": selected_plan,
        "recommendations": rows,
        "recommended_first_pack": rows[0]["pack_id"] if rows else None,
    }


def _params_for_marketplace_label(label, idx, total, use_case=""):
    sev = idx / max(total - 1, 1)
    name = str(label).lower()
    base = 40 + 35 * sev
    if "normal" in name or "healthy" in name or "calm" in name:
        return {"base_f": 45 if "machine" in use_case.lower() or "maintenance" in use_case.lower() else 0, "harm_r": 0.04, "imp_r": 0.0, "noise_l": 0.07}
    if "bearing" in name or "wear" in name or "failure" in name or "misalignment" in name or "unbalance" in name:
        return {"base_f": base, "harm_r": 0.18 + sev * 1.0, "imp_r": sev * 28, "noise_l": 0.10 + sev * 0.25}
    if "drill" in name:
        return {"base_f": 130, "harm_r": 1.1, "imp_r": 8 + sev * 12, "noise_l": 0.25}
    if "grind" in name or "cut" in name or "scrap" in name:
        return {"base_f": 180, "harm_r": 1.45, "imp_r": sev * 8, "noise_l": 0.32}
    if "impact" in name or "tamper" in name or "critical" in name:
        return {"base_f": 0, "harm_r": 0.1 + sev, "imp_r": 10 + sev * 35, "noise_l": 0.15 + sev * 0.25}
    if "vehicle" in name or "tractor" in name or "offroad" in name:
        return {"base_f": 42, "harm_r": 0.55 + sev * 0.3, "imp_r": sev * 3, "noise_l": 0.18 + sev * 0.1}
    if "chainsaw" in name:
        return {"base_f": 85, "harm_r": 1.20, "imp_r": sev * 2, "noise_l": 0.28}
    return {"base_f": base, "harm_r": 0.08 + sev * 0.75, "imp_r": sev * 20, "noise_l": 0.08 + sev * 0.22}


def generate_marketplace_pack_dataset(pack_id, samples_per_class=80):
    pack = get_marketplace_pack(pack_id)
    if not pack:
        return pd.DataFrame(), {"error": "Unknown marketplace pack", "pack_id": pack_id}
    base_pack = pack.get("base_pack")
    if base_pack and base_pack in INDUSTRY_PACKS:
        df, manifest = generate_industry_pack_dataset(base_pack, samples_per_class=samples_per_class)
        manifest["marketplace_pack_id"] = pack_id
        manifest["marketplace_pack_name"] = pack.get("name")
        manifest["marketplace_tier"] = pack.get("tier")
        manifest["minimum_plan"] = pack.get("minimum_plan")
        manifest["price_range"] = pack.get("price_range")
        manifest["disclaimer"] = PACK_MARKETPLACE_DISCLAIMER
        return df, manifest
    rows = []
    labels = pack.get("classes", ["Normal", "Warning", "Critical"])
    sr = int(pack.get("sample_rate", 16000))
    for idx, label in enumerate(labels):
        params = _params_for_marketplace_label(label, idx, len(labels), pack.get("target_market", ""))
        for _ in range(int(samples_per_class)):
            p = {
                "base_f": max(0, params["base_f"] + np.random.normal(0, max(abs(params["base_f"]) * 0.025, 0.7))),
                "harm_r": max(0, params["harm_r"] + np.random.normal(0, 0.06)),
                "imp_r": max(0, params["imp_r"] + np.random.normal(0, max(params["imp_r"] * 0.05, 0.25))),
                "noise_l": max(0.001, params["noise_l"] * np.random.uniform(0.85, 1.22)),
            }
            d = generate_universal_signal(2.0, sr, p["base_f"], p["harm_r"], p["imp_r"], p["noise_l"])
            row = extract_signal_features(d["sig"], sr, label)
            row["PackID"] = pack_id
            rows.append(row)
    df = pd.DataFrame(rows)
    manifest = {
        "engine": "EdgeTwin Studio V26 Marketplace Pack Generator",
        "marketplace_pack_id": pack_id,
        "marketplace_pack_name": pack.get("name"),
        "samples_per_class": samples_per_class,
        "classes": labels,
        "sample_rate": sr,
        "sensors": pack.get("sensors", []),
        "minimum_plan": pack.get("minimum_plan"),
        "price_range": pack.get("price_range"),
        "disclaimer": PACK_MARKETPLACE_DISCLAIMER,
    }
    return df, manifest


def build_custom_marketplace_pack(name, description, use_case_type, sensors, labels_text, sample_rate=16000, price_tier="Professional", license_model="Reusable pack"):
    labels = parse_labels_text(labels_text, fallback=["Normal", "Warning", "Critical"])
    price_map = {
        "Free Demo": ("Free Demo", "Free"),
        "Starter": ("Starter Pilot", "EUR 49 - 99"),
        "Professional": ("Professional Pilot", "EUR 199 - 499"),
        "Real-Data Pilot": ("Real-Data Pilot", "EUR 799 - 1,500"),
        "Enterprise": ("Enterprise", "Custom / EUR 2,500+"),
    }
    min_plan, price = price_map.get(price_tier, ("Professional Pilot", "EUR 199 - 499"))
    pack_id = "custom_" + _slugify_pack_name(name)
    defaults = get_use_case_defaults(use_case_type)
    return {
        "pack_id": pack_id,
        "name": name.strip() or "Custom Marketplace Pack",
        "description": description.strip() or "Custom Edge AI pilot pack.",
        "use_case_type": use_case_type,
        "target_market": use_case_type,
        "template": defaults.get("template"),
        "sensors": sensors or defaults.get("recommended_sensors", ["Audio", "Vibration"]),
        "classes": labels,
        "sample_rate": int(sample_rate),
        "price_tier": price_tier,
        "minimum_plan": min_plan,
        "price_range": price,
        "license_model": license_model,
        "commercial_angle": "Reusable customer-specific pack for repeatable Edge AI pilot generation.",
        "evidence_needed": "Representative real samples, known false-positive cases and field validation notes.",
    }


def validate_marketplace_pack_definition(pack_def):
    issues = []
    score = 100
    if not pack_def.get("name") or len(pack_def.get("name", "")) < 5:
        issues.append("Pack name is too short.")
        score -= 15
    if len(pack_def.get("classes", [])) < 3:
        issues.append("Use at least 3 classes for a commercially useful pack.")
        score -= 20
    if len(pack_def.get("classes", [])) > 8:
        issues.append("Too many classes for a first pack; consider a narrower paid pack.")
        score -= 8
    if len(pack_def.get("sensors", [])) < 2:
        issues.append("Use at least two relevant sensors or explain why single-sensor is enough.")
        score -= 12
    if "Audio" not in pack_def.get("sensors", []) and "Vibration" not in pack_def.get("sensors", []):
        issues.append("For EdgeTwin's current strength, include Audio and/or Vibration.")
        score -= 18
    if int(pack_def.get("sample_rate", 0)) < 1000:
        issues.append("Sample rate is too low for most audio/vibration packs.")
        score -= 12
    if len(pack_def.get("description", "")) < 30:
        issues.append("Description should explain target customer and use-case value.")
        score -= 10
    if not issues:
        issues.append("Pack definition is clean enough for internal beta packaging.")
    score = int(np.clip(score, 0, 100))
    if score >= 80:
        verdict = "Strong custom pack candidate. Add real-data bridge evidence before high-ticket sales."
    elif score >= 60:
        verdict = "Usable beta pack, but improve focus, sensors or label design before selling seriously."
    else:
        verdict = "Not ready as a paid pack yet. Narrow the use-case and improve labels/sensors."
    return {"quality_score": score, "issues": issues, "verdict": verdict}


def generate_custom_marketplace_pack_dataset(pack_def, samples_per_class=80):
    rows = []
    labels = pack_def.get("classes", ["Normal", "Warning", "Critical"])
    sr = int(pack_def.get("sample_rate", 16000))
    for idx, label in enumerate(labels):
        params = _params_for_marketplace_label(label, idx, len(labels), pack_def.get("use_case_type", ""))
        for _ in range(int(samples_per_class)):
            p = {
                "base_f": max(0, params["base_f"] + np.random.normal(0, max(abs(params["base_f"]) * 0.025, 0.7))),
                "harm_r": max(0, params["harm_r"] + np.random.normal(0, 0.06)),
                "imp_r": max(0, params["imp_r"] + np.random.normal(0, max(params["imp_r"] * 0.05, 0.25))),
                "noise_l": max(0.001, params["noise_l"] * np.random.uniform(0.85, 1.22)),
            }
            d = generate_universal_signal(2.0, sr, p["base_f"], p["harm_r"], p["imp_r"], p["noise_l"])
            row = extract_signal_features(d["sig"], sr, label)
            row["PackID"] = pack_def.get("pack_id", "custom_pack")
            rows.append(row)
    df = pd.DataFrame(rows)
    manifest = {
        "engine": "EdgeTwin Studio V26 Custom Marketplace Pack Generator",
        "custom_pack": pack_def,
        "samples_per_class": samples_per_class,
        "classes": labels,
        "sample_rate": sr,
        "sensors": pack_def.get("sensors", []),
        "minimum_plan": pack_def.get("minimum_plan"),
        "price_range": pack_def.get("price_range"),
        "disclaimer": PACK_MARKETPLACE_DISCLAIMER,
    }
    return df, manifest


def build_pack_marketplace_snapshot(project_name, selected_pack_id, selected_plan="Founder Test Mode", target_market="Custom", dataset_df=None, trust_gate=None, reliability_v2=None, real_bridge_result=None, deployment_plan=None, recommendations=None, custom_pack_definition=None, custom_pack_validation=None):
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    pack = custom_pack_definition or get_marketplace_pack(selected_pack_id) or {}
    audit = {}
    if len(dataset_df) and "Label" in dataset_df.columns:
        numeric = [c for c in dataset_df.columns if c != "Label" and pd.api.types.is_numeric_dtype(dataset_df[c])]
        if numeric:
            audit = dataset_doctor(dataset_df[numeric], dataset_df["Label"])
    score = 35
    risks = []
    assets = [
        "pack_manifest.json with labels, sensors, sample-rate and commercial positioning",
        "generated training dataset CSV",
        "dataset audit / doctor summary",
        "safe claims and validation disclaimer",
        "recommended real-data collection plan",
    ]
    if audit:
        score += min(25, audit.get("overall_score", 0) * 0.25)
    else:
        risks.append("No auditable dataset snapshot yet.")
    if trust_gate:
        score += 8
        assets.append("Trust Center snapshot")
    else:
        risks.append("Trust Center has not been run yet.")
    if reliability_v2:
        score += 10
        assets.append("Reliability Engine 2.0 per-class/per-sensor evidence")
    else:
        risks.append("Reliability 2.0 evidence is missing.")
    if real_bridge_result:
        score += 12
        assets.append("Synthetic-to-Real Bridge evidence")
    else:
        risks.append("Real-data bridge is missing; avoid production claims.")
    if deployment_plan:
        score += 7
        assets.append("Deployment/BOM guidance")
    else:
        risks.append("Deployment plan/BOM not attached yet.")
    if custom_pack_validation:
        score = score * 0.75 + custom_pack_validation.get("quality_score", 0) * 0.25
    minimum_plan = pack.get("minimum_plan", "Professional Pilot")
    if selected_plan == "Free Demo" and minimum_plan != "Free Demo":
        risks.append("Selected plan should show this pack as locked/upgrade.")
        score -= 8
    score = int(np.clip(score, 0, 100))
    if score >= 82:
        status = "Marketplace beta-ready"
        commercial_fit = "Strong"
        verdict = "This pack is ready for a curated beta offer, with clear field-validation limits."
    elif score >= 65:
        status = "Internal beta-ready"
        commercial_fit = "Medium"
        verdict = "This pack can be tested internally or with friendly beta users before paid launch."
    else:
        status = "Not ready to sell"
        commercial_fit = "Weak"
        verdict = "Improve audit evidence, reliability scoring and real-data validation before selling this pack."
    return {
        "project_name": project_name,
        "created_at": _now(),
        "selected_pack_id": selected_pack_id,
        "pack_name": pack.get("name", selected_pack_id),
        "selected_plan": selected_plan,
        "target_market": target_market,
        "minimum_plan": minimum_plan,
        "price_range": pack.get("price_range", "Unknown"),
        "marketplace_readiness_score": score,
        "pack_status": status,
        "commercial_fit": commercial_fit,
        "verdict": verdict,
        "pack_assets": assets,
        "launch_risks": risks[:10] or ["No major launch risks detected for beta positioning."],
        "pack_definition": pack,
        "custom_pack_validation": custom_pack_validation or {},
        "recommendations": recommendations or {},
        "dataset_audit": audit,
        "safe_sales_claim": "This pack accelerates Edge AI pilot preparation for a focused sensor use-case.",
        "unsafe_claims_to_avoid": [
            "Do not claim production accuracy without real field validation.",
            "Do not claim the pack replaces customer data collection.",
            "Do not claim certified safety/security performance.",
        ],
        "disclaimer": PACK_MARKETPLACE_DISCLAIMER,
    }


def compact_pack_marketplace_summary(snapshot):
    if not snapshot:
        return {}
    return {
        "pack_name": snapshot.get("pack_name"),
        "score": snapshot.get("marketplace_readiness_score"),
        "status": snapshot.get("pack_status"),
        "minimum_plan": snapshot.get("minimum_plan"),
        "price_range": snapshot.get("price_range"),
    }


def generate_marketplace_pack_pdf(snapshot):
    snapshot = snapshot or {}
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt=clean_pdf_text("EdgeTwin Studio Industry Pack Marketplace"), ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("V26 - Pack Builder & Commercial Catalog"), ln=True, align="C")
    pdf.ln(8)
    safe_pdf_cell(pdf, "Pack Summary", 8, True)
    safe_pdf_cell(pdf, f"Project: {snapshot.get('project_name', 'Unknown')}")
    safe_pdf_cell(pdf, f"Pack: {snapshot.get('pack_name', 'Unknown')}")
    safe_pdf_cell(pdf, f"Target market: {snapshot.get('target_market', 'Unknown')}")
    safe_pdf_cell(pdf, f"Minimum plan: {snapshot.get('minimum_plan', 'Unknown')}")
    safe_pdf_cell(pdf, f"Price range: {snapshot.get('price_range', 'Unknown')}")
    safe_pdf_cell(pdf, f"Marketplace readiness: {snapshot.get('marketplace_readiness_score', 0)}%")
    safe_pdf_cell(pdf, f"Pack status: {snapshot.get('pack_status', 'Unknown')}")
    safe_pdf_multicell(pdf, snapshot.get("verdict", ""))

    pdf.ln(4)
    safe_pdf_cell(pdf, "Pack Assets", 8, True)
    for item in snapshot.get("pack_assets", [])[:12]:
        safe_pdf_multicell(pdf, f"- {item}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Launch Risks", 8, True)
    for item in snapshot.get("launch_risks", [])[:12]:
        safe_pdf_multicell(pdf, f"- {item}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Safe Sales Claim", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("safe_sales_claim", ""))
    safe_pdf_cell(pdf, "Claims to Avoid", 8, True)
    for item in snapshot.get("unsafe_claims_to_avoid", []):
        safe_pdf_multicell(pdf, f"- {item}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Validation Disclaimer", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", PACK_MARKETPLACE_DISCLAIMER))
    return safe_pdf_output(pdf)


def create_marketplace_pack_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    zip_buf = io.BytesIO()
    pdf_bytes = generate_marketplace_pack_pdf(snapshot)
    pack_def = snapshot.get("pack_definition", {})
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("marketplace_pack_report_v26.pdf", pdf_bytes)
        zf.writestr("marketplace_snapshot.json", safe_json_dumps(snapshot, indent=2))
        zf.writestr("pack_manifest.json", safe_json_dumps(pack_def, indent=2))
        zf.writestr("marketplace_catalog.csv", pd.DataFrame(get_marketplace_pack_catalog()).to_csv(index=False))
        recs = snapshot.get("recommendations", {}).get("recommendations", []) if isinstance(snapshot.get("recommendations"), dict) else []
        zf.writestr("pack_recommendations.csv", pd.DataFrame(recs).to_csv(index=False))
        if len(dataset_df):
            zf.writestr("generated_pack_dataset.csv", dataset_df.to_csv(index=False))
            if "Label" in dataset_df.columns:
                zf.writestr("label_distribution.csv", dataset_df["Label"].value_counts().reset_index().to_csv(index=False))
        zf.writestr("README.txt", f"""EdgeTwin Studio V26 Marketplace Pack Bundle

Project: {project_name}
Pack: {snapshot.get('pack_name', 'Unknown')}
Marketplace readiness: {snapshot.get('marketplace_readiness_score', 0)}%
Status: {snapshot.get('pack_status', 'Unknown')}
Minimum plan: {snapshot.get('minimum_plan', 'Unknown')}
Price range: {snapshot.get('price_range', 'Unknown')}

Recommended next step:
Use this pack internally or with beta users first. Add real-data bridge evidence before high-ticket or production-sensitive claims.

{snapshot.get('disclaimer', PACK_MARKETPLACE_DISCLAIMER)}
""")
    return zip_buf.getvalue()

# ============================================================
# V26.1 NORMAL VS ABNORMAL BASELINE ENGINE
# ============================================================

NORMAL_LABEL_KEYWORDS = [
    "normal", "healthy", "baseline", "calm", "idle", "ok", "good", "forest_normal", "background"
]


def get_normality_feature_columns(dataset_df, label_col="Label"):
    if dataset_df is None or not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0:
        return []
    exclude = {label_col, "Template", "UseCaseType", "Environment", "Event", "RecommendedAction", "Level", "Filename", "Source"}
    cols = []
    for col in dataset_df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(dataset_df[col]):
            vals = pd.to_numeric(dataset_df[col], errors="coerce")
            if vals.notna().sum() >= max(5, min(20, len(dataset_df) // 10)) and float(vals.std(skipna=True) or 0) > 1e-12:
                cols.append(col)
    return cols


def detect_normal_labels(dataset_df, label_col="Label"):
    if dataset_df is None or not isinstance(dataset_df, pd.DataFrame) or label_col not in dataset_df.columns:
        return [], ["No Label column found, so no normal class can be detected automatically."]

    labels = [str(x) for x in dataset_df[label_col].dropna().unique().tolist()]
    detected = []
    for label in labels:
        cleaned = label.lower().replace(" ", "_")
        if any(keyword in cleaned for keyword in NORMAL_LABEL_KEYWORDS):
            detected.append(label)

    notes = []
    if detected:
        notes.append("Detected normal/baseline labels from label names.")
        return detected, notes

    # Fallback heuristics for existing EdgeTwin generated datasets.
    try:
        if "HealthScore" in dataset_df.columns:
            means = dataset_df.groupby(label_col)["HealthScore"].mean(numeric_only=True).sort_values(ascending=False)
            if len(means):
                detected = [str(means.index[0])]
                notes.append("No explicit normal label found. Used highest average HealthScore as baseline candidate.")
                return detected, notes
        if "FusionScore" in dataset_df.columns:
            means = dataset_df.groupby(label_col)["FusionScore"].mean(numeric_only=True).sort_values(ascending=True)
            if len(means):
                detected = [str(means.index[0])]
                notes.append("No explicit normal label found. Used lowest average FusionScore as baseline candidate.")
                return detected, notes
    except Exception:
        pass

    counts = dataset_df[label_col].astype(str).value_counts()
    if len(counts):
        detected = [str(counts.index[0])]
        notes.append("No explicit normal label found. Used largest class as fallback baseline. Review this before using commercially.")
        return detected, notes

    return [], ["No usable labels found for baseline detection."]


def _robust_scale_frame(df, profile):
    cols = profile.get("feature_columns", [])
    if not cols:
        return pd.DataFrame(index=df.index), pd.Series(np.zeros(len(df)), index=df.index)
    med = pd.Series(profile.get("median", {}), dtype=float).reindex(cols).fillna(0.0)
    scale = pd.Series(profile.get("scale", {}), dtype=float).reindex(cols).replace(0, np.nan).fillna(1.0)
    numeric = df[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    z = (numeric - med) / scale
    distance = z.abs().clip(0, 12).mean(axis=1)
    return z, distance


def build_normal_baseline_profile(dataset_df, normal_labels=None, label_col="Label", sensitivity=1.0):
    if dataset_df is None or not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0:
        return {"valid": False, "reason": "No dataset available.", "issues": ["Load or generate a dataset first."]}
    if label_col not in dataset_df.columns:
        return {"valid": False, "reason": "Dataset needs a Label column.", "issues": ["Add a Label column before running normality analysis."]}

    df = dataset_df.copy()
    df[label_col] = df[label_col].astype(str)
    feature_cols = get_normality_feature_columns(df, label_col=label_col)
    if len(feature_cols) < 2:
        return {
            "valid": False,
            "reason": "Not enough numeric signal/fusion features for normality analysis.",
            "issues": ["At least 2 numeric feature columns are recommended."],
            "feature_columns": feature_cols,
        }

    detected, notes = detect_normal_labels(df, label_col=label_col)
    normal_labels = [str(x) for x in (normal_labels or detected) if str(x) in set(df[label_col].unique())]
    if not normal_labels:
        normal_labels = detected

    normal_df = df[df[label_col].isin(normal_labels)].copy()
    issues = []
    if len(normal_df) < 20:
        issues.append("Baseline class has fewer than 20 samples. Normal/abnormal thresholds are weak.")
    if len(normal_df) < 5:
        return {
            "valid": False,
            "reason": "Not enough baseline samples.",
            "issues": issues + ["Generate or upload more normal/baseline samples."],
            "normal_labels": normal_labels,
            "feature_columns": feature_cols,
        }

    Xn = normal_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    med = Xn.median(skipna=True)
    q25 = Xn.quantile(0.25)
    q75 = Xn.quantile(0.75)
    iqr = (q75 - q25).replace(0, np.nan)
    std = Xn.std(skipna=True).replace(0, np.nan)
    scale = iqr.fillna(std).fillna(1.0)
    sensitivity = float(np.clip(sensitivity, 0.5, 2.0))
    scale = scale * sensitivity

    baseline_ratio = len(normal_df) / max(len(df), 1)
    baseline_confidence = int(np.clip(35 + min(35, len(normal_df) / 4) + min(20, len(feature_cols) * 2) + (10 if baseline_ratio >= 0.15 else 0), 0, 100))
    if any("fallback" in n.lower() or "no explicit" in n.lower() for n in notes):
        baseline_confidence = int(max(0, baseline_confidence - 15))
        issues.append("Normal baseline was inferred instead of explicitly provided. Confirm the selected baseline labels.")

    profile = {
        "valid": True,
        "label_col": label_col,
        "normal_labels": normal_labels,
        "detected_normal_labels": detected,
        "notes": notes,
        "issues": issues,
        "feature_columns": feature_cols,
        "baseline_samples": int(len(normal_df)),
        "total_samples": int(len(df)),
        "baseline_ratio": float(baseline_ratio),
        "baseline_confidence": int(baseline_confidence),
        "median": med.to_dict(),
        "q25": q25.to_dict(),
        "q75": q75.to_dict(),
        "scale": scale.to_dict(),
        "created_at": _now(),
    }
    return _json_safe(profile)


def score_normality_against_baseline(dataset_df, baseline_profile):
    if dataset_df is None or not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0:
        return pd.DataFrame()
    if not baseline_profile or not baseline_profile.get("valid"):
        return dataset_df.copy()

    df = dataset_df.copy()
    z, distance = _robust_scale_frame(df, baseline_profile)
    # A robust distance of ~0-1 is normal-like, 2-3 warning/abnormal, 4+ critical.
    normality_score = (100 - distance * 22).clip(0, 100).round(1)
    abnormality_score = (100 - normality_score).round(1)

    def status(score):
        score = float(score)
        if score >= 80:
            return "Normal-like"
        if score >= 65:
            return "Watch"
        if score >= 45:
            return "Abnormal"
        return "Critical abnormal"

    df["NormalityScore"] = normality_score.values
    df["AbnormalityScore"] = abnormality_score.values
    df["NormalityStatus"] = [status(v) for v in normality_score.values]
    df["BaselineDistance"] = distance.round(3).values

    if len(z.columns):
        df["TopDeviationFeature"] = z.abs().idxmax(axis=1).values
        df["TopDeviationZ"] = z.abs().max(axis=1).round(3).values
    else:
        df["TopDeviationFeature"] = "Unknown"
        df["TopDeviationZ"] = 0.0
    return df


def summarize_normality_result(scored_df, baseline_profile, label_col="Label"):
    if scored_df is None or not isinstance(scored_df, pd.DataFrame) or len(scored_df) == 0:
        return {"valid": False, "verdict": "No scored dataset available."}

    abnormal_mask = scored_df["NormalityStatus"].isin(["Abnormal", "Critical abnormal"]) if "NormalityStatus" in scored_df.columns else pd.Series(False, index=scored_df.index)
    critical_mask = scored_df["NormalityStatus"].eq("Critical abnormal") if "NormalityStatus" in scored_df.columns else pd.Series(False, index=scored_df.index)
    avg_score = float(scored_df.get("NormalityScore", pd.Series([0])).mean())
    abnormal_rate = float(abnormal_mask.mean() * 100)
    critical_rate = float(critical_mask.mean() * 100)

    baseline_confidence = int(baseline_profile.get("baseline_confidence", 0)) if baseline_profile else 0
    if baseline_confidence < 45:
        decision = "NO-GO"
        verdict = "Baseline is too weak. Add clearer normal/reference data before using this commercially."
    elif critical_rate >= 30:
        decision = "CONDITIONAL"
        verdict = "Many samples strongly deviate from the normal baseline. Useful for anomaly discovery, but validate labels and field data."
    elif abnormal_rate >= 35:
        decision = "CONDITIONAL"
        verdict = "Dataset contains a meaningful abnormal region. Good for pilot design if normal labels are confirmed."
    elif avg_score >= 75 and abnormal_rate < 20:
        decision = "GO"
        verdict = "Normal baseline looks coherent. Good for demo/pilot normal-vs-abnormal explanation."
    else:
        decision = "CONDITIONAL"
        verdict = "Normal/abnormal split is usable, but more real baseline data would improve confidence."

    per_label = []
    if label_col in scored_df.columns and "NormalityScore" in scored_df.columns:
        grouped = scored_df.groupby(label_col)
        for label, g in grouped:
            abn = g["NormalityStatus"].isin(["Abnormal", "Critical abnormal"]).mean() * 100
            crit = g["NormalityStatus"].eq("Critical abnormal").mean() * 100
            mean_score = g["NormalityScore"].mean()
            if str(label) in baseline_profile.get("normal_labels", []):
                role = "Baseline / normal"
            elif mean_score >= 75:
                role = "Normal-like / overlapping"
            elif mean_score >= 55:
                role = "Transition / warning"
            else:
                role = "Abnormal / event-like"
            per_label.append({
                "label": str(label),
                "samples": int(len(g)),
                "mean_normality_score": round(float(mean_score), 1),
                "abnormal_rate_pct": round(float(abn), 1),
                "critical_rate_pct": round(float(crit), 1),
                "role": role,
            })
    per_label = sorted(per_label, key=lambda x: x.get("mean_normality_score", 0))

    top_features = []
    if "TopDeviationFeature" in scored_df.columns:
        for feat, count in scored_df["TopDeviationFeature"].astype(str).value_counts().head(10).items():
            subset = scored_df[scored_df["TopDeviationFeature"].astype(str) == feat]
            top_features.append({
                "feature": feat,
                "count": int(count),
                "avg_deviation_z": round(float(subset.get("TopDeviationZ", pd.Series([0])).mean()), 2),
                "meaning": normality_feature_explanation(feat),
            })

    claims_allowed = [
        "Can explain which samples look normal-like versus abnormal relative to the selected baseline.",
        "Can help identify weak/overlapping labels and top deviating features for a pilot.",
        "Can support field-test planning by showing where more normal/reference data is needed.",
    ]
    claims_avoid = [
        "Do not claim production anomaly detection is guaranteed without field validation.",
        "Do not claim the inferred baseline is correct if normal labels were only guessed.",
        "Do not use as a safety-critical automatic decision system without independent validation.",
    ]

    return _json_safe({
        "valid": True,
        "normality_score": int(np.clip(avg_score, 0, 100)),
        "abnormal_rate_pct": round(abnormal_rate, 1),
        "critical_rate_pct": round(critical_rate, 1),
        "baseline_confidence": baseline_confidence,
        "decision": decision,
        "verdict": verdict,
        "per_label_summary": per_label,
        "top_deviation_features": top_features,
        "claims_allowed": claims_allowed,
        "claims_to_avoid": claims_avoid,
        "recommended_next_steps": normality_next_steps(baseline_profile, abnormal_rate, critical_rate),
    })


def normality_feature_explanation(feature_name):
    f = str(feature_name).lower()
    if "rms" in f or "std" in f:
        return "Signal energy/amplitude differs from normal baseline."
    if "kurt" in f or "crest" in f or "imp" in f:
        return "Impact/spike behavior differs from normal baseline."
    if "centroid" in f or "rolloff" in f or "flat" in f or "freq" in f:
        return "Frequency content differs from normal baseline."
    if "audio" in f:
        return "Acoustic sensor score deviates from normal context."
    if "vibration" in f or "health" in f:
        return "Vibration/health signal deviates from normal operating pattern."
    if "radar" in f or "gps" in f or "zone" in f:
        return "Context/presence/location signal deviates from normal conditions."
    return "Feature deviates most from the selected normal baseline."


def normality_next_steps(baseline_profile, abnormal_rate, critical_rate):
    steps = []
    if not baseline_profile or not baseline_profile.get("valid"):
        return ["Create a clean normal/baseline class before running this analysis."]
    if baseline_profile.get("baseline_confidence", 0) < 65:
        steps.append("Collect more normal/reference samples in the real operating environment.")
    if any("fallback" in x.lower() or "inferred" in x.lower() or "no explicit" in x.lower() for x in baseline_profile.get("notes", [])):
        steps.append("Confirm which labels are truly normal before using this output in a paid report.")
    if abnormal_rate < 10:
        steps.append("Add stronger abnormal/event examples. The dataset may be too normal-like.")
    if critical_rate > 35:
        steps.append("Review critical abnormal samples; they may be too extreme or unrealistic for pilot training.")
    steps.append("Validate thresholds with real field data before production deployment.")
    return steps


def run_normality_baseline_engine(dataset_df, normal_labels=None, label_col="Label", sensitivity=1.0, project_context=None):
    profile = build_normal_baseline_profile(dataset_df, normal_labels=normal_labels, label_col=label_col, sensitivity=sensitivity)
    if not profile.get("valid"):
        return _json_safe({
            "valid": False,
            "baseline_profile": profile,
            "scored_dataset": pd.DataFrame(),
            "summary": {"valid": False, "verdict": profile.get("reason", "Normality engine could not run.")},
            "created_at": _now(),
        })
    scored_df = score_normality_against_baseline(dataset_df, profile)
    summary = summarize_normality_result(scored_df, profile, label_col=label_col)
    return {
        "valid": True,
        "engine": "EdgeTwin Studio V26.1 Normal vs Abnormal Baseline Engine",
        "baseline_profile": profile,
        "summary": summary,
        "scored_dataset": scored_df,
        "project_context": project_context or {},
        "created_at": _now(),
        "disclaimer": RELIABILITY_DISCLAIMER,
    }


def generate_normality_pdf_report(project_name, normality_result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Normal vs Abnormal Report", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("Powered by OMEGA-X Baseline Intelligence"), ln=True, align="C")
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)

    summary = normality_result.get("summary", {}) if normality_result else {}
    profile = normality_result.get("baseline_profile", {}) if normality_result else {}

    safe_pdf_cell(pdf, "Executive Summary", 8, True)
    safe_pdf_cell(pdf, f"Decision: {summary.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Normality score: {summary.get('normality_score', 0)}%")
    safe_pdf_cell(pdf, f"Abnormal rate: {summary.get('abnormal_rate_pct', 0)}%")
    safe_pdf_cell(pdf, f"Critical abnormal rate: {summary.get('critical_rate_pct', 0)}%")
    safe_pdf_cell(pdf, f"Baseline confidence: {summary.get('baseline_confidence', 0)}%")
    safe_pdf_multicell(pdf, summary.get("verdict", ""))
    pdf.ln(4)

    safe_pdf_cell(pdf, "Baseline Definition", 8, True)
    safe_pdf_cell(pdf, f"Normal labels: {', '.join(profile.get('normal_labels', []))}")
    safe_pdf_cell(pdf, f"Baseline samples: {profile.get('baseline_samples', 0)} / {profile.get('total_samples', 0)}")
    safe_pdf_cell(pdf, f"Feature columns: {len(profile.get('feature_columns', []))}")
    for note in profile.get("notes", []):
        safe_pdf_multicell(pdf, f"Note: {note}")
    for issue in profile.get("issues", []):
        safe_pdf_multicell(pdf, f"Issue: {issue}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Per-label Normality", 8, True)
    for row in summary.get("per_label_summary", [])[:12]:
        safe_pdf_multicell(pdf, f"{row.get('label')}: score {row.get('mean_normality_score')}%, abnormal {row.get('abnormal_rate_pct')}%, role: {row.get('role')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Top Deviating Features", 8, True)
    for row in summary.get("top_deviation_features", [])[:8]:
        safe_pdf_multicell(pdf, f"{row.get('feature')}: avg z {row.get('avg_deviation_z')} - {row.get('meaning')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Recommended Next Steps", 8, True)
    for step in summary.get("recommended_next_steps", []):
        safe_pdf_multicell(pdf, f"- {step}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Claims to Avoid", 8, True)
    for claim in summary.get("claims_to_avoid", []):
        safe_pdf_multicell(pdf, f"- {claim}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Validation Note", 8, True)
    safe_pdf_multicell(pdf, RELIABILITY_DISCLAIMER)
    return safe_pdf_output(pdf)


def create_normality_baseline_bundle(project_name, dataset_df, normality_result):
    pdf_bytes = generate_normality_pdf_report(project_name, normality_result)
    scored_df = normality_result.get("scored_dataset", pd.DataFrame()) if isinstance(normality_result, dict) else pd.DataFrame()
    if not isinstance(scored_df, pd.DataFrame):
        scored_df = pd.DataFrame(scored_df)
    metadata = dict(normality_result or {})
    metadata.pop("scored_dataset", None)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("normality_report.pdf", pdf_bytes)
        zf.writestr("normality_result.json", json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False))
        zf.writestr("normality_scored_dataset.csv", scored_df.to_csv(index=False))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr("original_dataset_snapshot.csv", dataset_df.to_csv(index=False))
        zf.writestr("README.txt", """EdgeTwin Studio V26.1 Normal vs Abnormal Baseline Bundle\n\nThis bundle explains what looks normal-like versus abnormal relative to the selected baseline class.\nIt is intended for pilot preparation and threshold planning. Validate with real field data before production deployment.\n""")
    return zip_buf.getvalue()



# ============================================================
# V26.2 EDGE IMPULSE ANOMALY EXPORT
# ============================================================

EDGE_IMPULSE_EXPORT_NOTE = (
    "EdgeTwin prepares normal/baseline feature exports and guidance for Edge Impulse anomaly workflows. "
    "Use real field data to validate axes, K value and anomaly threshold before production deployment."
)

DERIVED_EDGETWIN_COLUMNS = {
    "FusionScore", "HealthScore", "Confidence", "NormalityScore", "AbnormalityScore", "BaselineDistance",
    "NormalityStatus", "TopDeviationFeature", "Template", "Event", "RecommendedAction", "Level",
    "UseCaseType", "Environment"
}


def get_edge_impulse_feature_columns(dataset_df, include_derived_scores=False):
    if not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0:
        return []
    cols = []
    for col in dataset_df.columns:
        if col == "Label":
            continue
        if not pd.api.types.is_numeric_dtype(dataset_df[col]):
            continue
        if not include_derived_scores and col in DERIVED_EDGETWIN_COLUMNS:
            continue
        if dataset_df[col].isna().all():
            continue
        if float(dataset_df[col].std(skipna=True) or 0) <= 1e-12:
            continue
        cols.append(col)
    # Prefer physically interpretable features first.
    priority = [
        "RMS", "Std", "Kurtosis", "CrestFactor", "ZCR", "SpectralCentroid", "SpectralRolloff", "SpectralFlatness",
        "AudioScore", "VibrationScore", "GasScore", "RadarScore", "GPSZoneScore"
    ]
    return sorted(cols, key=lambda c: (priority.index(c) if c in priority else 999, c))


def suggest_edge_impulse_k(normal_sample_count, normal_label_count=1):
    normal_sample_count = int(max(0, normal_sample_count or 0))
    normal_label_count = int(max(1, normal_label_count or 1))
    if normal_sample_count < 50:
        return 8
    if normal_sample_count < 150:
        return 16
    if normal_sample_count < 500:
        return 32
    # More normal variation can justify more clusters, but cap it to keep embedded inference practical.
    return int(min(64, max(32, normal_label_count * 16)))


def _axis_deviation_scores(dataset_df, normal_labels, feature_cols, normality_result=None):
    if not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0 or not feature_cols:
        return []
    df = dataset_df.copy()
    labels = df["Label"].astype(str) if "Label" in df.columns else pd.Series(["Unknown"] * len(df))
    normal_mask = labels.isin([str(x) for x in normal_labels]) if normal_labels else pd.Series([False] * len(df), index=df.index)
    normal_df = df.loc[normal_mask, feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    other_df = df.loc[~normal_mask, feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(normal_df) < 2:
        return []
    rows = []
    for col in feature_cols:
        n_mean = float(normal_df[col].mean())
        n_std = float(normal_df[col].std() or 1e-9)
        variance = float(normal_df[col].std() or 0)
        if len(other_df) > 0:
            o_mean = float(other_df[col].mean())
            separation = abs(o_mean - n_mean) / max(n_std, 1e-9)
        else:
            separation = variance / max(abs(n_mean), 1.0)
        score = float(np.clip(separation * 25 + min(variance, 100) * 0.5, 0, 100))
        rows.append({
            "axis": col,
            "score": round(score, 2),
            "normal_mean": round(n_mean, 6),
            "normal_std": round(n_std, 6),
            "reason": "High normal-vs-event separation" if separation >= 1.5 else "Useful normal variation axis",
        })
    # If normality engine exists, boost known top-deviation features.
    try:
        top_dev = (normality_result or {}).get("summary", {}).get("top_deviation_features", [])
        boost = {row.get("feature"): float(row.get("avg_deviation_z", 0)) for row in top_dev}
        for row in rows:
            row["score"] = round(min(100, row["score"] + min(20, boost.get(row["axis"], 0) * 4)), 2)
    except Exception:
        pass
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def build_edge_impulse_instructions(project_name, summary):
    axes = [row.get("axis") for row in summary.get("recommended_axes", [])]
    axes_text = ", ".join(axes) if axes else "use Edge Impulse 'Select suggested axes' after feature generation"
    return f"""# EdgeTwin Studio -> Edge Impulse Anomaly Export

Project: {project_name}
Workflow: {summary.get('workflow')}
Export readiness: {summary.get('export_readiness_score')}%
Recommended K clusters: {summary.get('recommended_k')}
Recommended anomaly axes: {axes_text}

## Intended use
This bundle prepares a normal/baseline dataset and an evaluation dataset for an Edge Impulse anomaly workflow.
Use it for pilot preparation, not as a production guarantee.

## Files
- `edge_impulse_normal_training.csv`: baseline/normal samples only. Use this to build the normal behavior model.
- `edge_impulse_evaluation_all_labels.csv`: all labels/classes for sanity-checking anomaly behavior.
- `edge_impulse_recommended_axes.json`: suggested axes/features for K-means anomaly detection.
- `edge_impulse_export_manifest.json`: export metadata and safety notes.

## Suggested Edge Impulse workflow
1. Create or open an Edge Impulse project.
2. Upload data through Data acquisition / CSV Wizard or your preferred Edge Impulse uploader.
3. Build an impulse that extracts useful time-series/spectral features for your sensor type.
4. Add the Anomaly Detection (K-means) learning block.
5. Train the anomaly block mainly on normal/baseline samples.
6. Select axes/features. Start with: {axes_text}.
7. Start with K={summary.get('recommended_k')}, then tune K and threshold using real validation data.
8. Test with the evaluation CSV and, more importantly, fresh field data.
9. Only deploy after threshold, false positives and missed events are validated in the real environment.

## Safety notes
- Synthetic data can accelerate pilots, but should not replace real field validation.
- If normal/baseline data is weak, anomaly detection will be unstable.
- Keep abnormal/event data for validation; do not train the normal model on mixed abnormal behavior.
- Production use requires real-world threshold tuning, device testing and deployment monitoring.
"""


def build_edge_impulse_anomaly_export_snapshot(
    project_name,
    dataset_df,
    normal_labels,
    workflow="K-means anomaly baseline",
    k_clusters=None,
    max_axes=6,
    normality_result=None,
    include_derived_scores=False,
):
    if not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0:
        return {"valid": False, "summary": {"decision": "NO-GO", "verdict": "No dataset available."}, "files": {}}
    if "Label" not in dataset_df.columns:
        return {"valid": False, "summary": {"decision": "NO-GO", "verdict": "Dataset needs a Label column."}, "files": {}}

    df = dataset_df.copy()
    df["Label"] = df["Label"].astype(str)
    normal_labels = [str(x) for x in (normal_labels or [])]
    feature_cols = get_edge_impulse_feature_columns(df, include_derived_scores=include_derived_scores)
    normal_mask = df["Label"].isin(normal_labels) if normal_labels else pd.Series([False] * len(df), index=df.index)
    normal_df = df.loc[normal_mask].copy()
    eval_df = df.copy()

    warnings = []
    if not normal_labels:
        warnings.append("No normal/baseline labels selected. Edge Impulse anomaly export is not reliable.")
    if len(normal_df) < 30:
        warnings.append("Very few normal samples. Collect more baseline data before tuning anomaly thresholds.")
    if len(feature_cols) < 2:
        warnings.append("Too few usable numeric features. Edge Impulse anomaly axes will be weak.")
    if len(set(normal_labels)) > 3:
        warnings.append("Multiple baseline labels selected. Confirm they all represent healthy/normal behavior.")

    axes_ranked = _axis_deviation_scores(df, normal_labels, feature_cols, normality_result=normality_result)
    recommended_axes = axes_ranked[: int(max(2, max_axes or 6))]
    if not k_clusters:
        k_clusters = suggest_edge_impulse_k(len(normal_df), len(normal_labels))

    # Keep export columns simple and Edge Impulse friendly.
    export_cols = ["Label"] + feature_cols
    normal_export = normal_df[[c for c in export_cols if c in normal_df.columns]].copy()
    eval_cols = export_cols + [c for c in ["NormalityScore", "AbnormalityScore", "NormalityStatus", "BaselineDistance"] if c in eval_df.columns]
    eval_export = eval_df[[c for c in eval_cols if c in eval_df.columns]].copy()

    # Edge Impulse examples often use lowercase label; provide a compatible alias while preserving original Label.
    if "label" not in normal_export.columns:
        normal_export.insert(0, "label", normal_export["Label"].astype(str))
    if "label" not in eval_export.columns:
        eval_export.insert(0, "label", eval_export["Label"].astype(str))

    readiness = 45
    readiness += min(25, len(normal_df) / 4)
    readiness += min(20, len(feature_cols) * 3)
    readiness += 10 if normality_result else 0
    readiness -= 20 if not normal_labels else 0
    readiness -= 20 if len(normal_df) < 30 else 0
    readiness = int(np.clip(readiness, 0, 100))
    if readiness >= 80:
        decision = "GO"
        verdict = "Good Edge Impulse anomaly export candidate. Validate K, axes and threshold with real field data."
    elif readiness >= 55:
        decision = "CONDITIONAL"
        verdict = "Usable for a private pilot, but more normal baseline data or better features are recommended."
    else:
        decision = "NO-GO"
        verdict = "Not ready for reliable anomaly export. Strengthen baseline data and feature quality first."

    summary = {
        "workflow": workflow,
        "export_readiness_score": readiness,
        "decision": decision,
        "verdict": verdict,
        "normal_labels": normal_labels,
        "normal_training_samples": int(len(normal_export)),
        "evaluation_samples": int(len(eval_export)),
        "feature_columns": feature_cols,
        "recommended_axes": recommended_axes,
        "recommended_k": int(k_clusters),
        "warnings": warnings,
        "edge_impulse_steps": [
            "Upload the normal-training CSV or raw normal samples into Edge Impulse Data acquisition.",
            "Create an impulse with a suitable processing block for your sensor signal.",
            "Add Anomaly Detection (K-means) and train mainly on normal/baseline behavior.",
            "Use the recommended axes as a starting point, then compare with Edge Impulse suggested axes.",
            "Tune K and threshold using real abnormal/evaluation samples.",
            "Deploy only after field validation on the actual hardware and environment.",
        ],
        "safety_note": EDGE_IMPULSE_EXPORT_NOTE,
    }
    manifest = {
        "engine": "EdgeTwin Studio V26.2 Edge Impulse Export",
        "project_name": project_name,
        "created_at": str(datetime.datetime.now()),
        "summary": summary,
        "source_dataset_rows": int(len(df)),
        "source_columns": list(df.columns),
    }
    instructions = build_edge_impulse_instructions(project_name, summary)
    return {
        "valid": decision != "NO-GO",
        "summary": summary,
        "manifest": manifest,
        "files": {
            "normal_training_csv": normal_export,
            "evaluation_csv": eval_export,
            "recommended_axes_json": recommended_axes,
            "instructions_md": instructions,
        },
    }


def generate_edge_impulse_pdf_report(project_name, snapshot):
    summary = (snapshot or {}).get("summary", {})
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin -> Edge Impulse Anomaly Export", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)
    safe_pdf_cell(pdf, "Export Summary", 8, True)
    safe_pdf_cell(pdf, f"Workflow: {summary.get('workflow', 'Unknown')}")
    safe_pdf_cell(pdf, f"Export readiness: {summary.get('export_readiness_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {summary.get('decision', 'Unknown')}")
    safe_pdf_multicell(pdf, summary.get("verdict", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Baseline", 8, True)
    safe_pdf_cell(pdf, f"Normal labels: {', '.join(summary.get('normal_labels', []))}")
    safe_pdf_cell(pdf, f"Normal training samples: {summary.get('normal_training_samples', 0)}")
    safe_pdf_cell(pdf, f"Evaluation samples: {summary.get('evaluation_samples', 0)}")
    safe_pdf_cell(pdf, f"Recommended K: {summary.get('recommended_k', 0)}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Recommended Anomaly Axes", 8, True)
    for row in summary.get("recommended_axes", [])[:10]:
        safe_pdf_multicell(pdf, f"- {row.get('axis')}: score {row.get('score')} ({row.get('reason')})")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Edge Impulse Steps", 8, True)
    for step in summary.get("edge_impulse_steps", []):
        safe_pdf_multicell(pdf, f"- {step}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Warnings", 8, True)
    for warning in summary.get("warnings", []) or ["No critical warnings."]:
        safe_pdf_multicell(pdf, f"- {warning}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Validation Note", 8, True)
    safe_pdf_multicell(pdf, EDGE_IMPULSE_EXPORT_NOTE)
    return safe_pdf_output(pdf)


def create_edge_impulse_anomaly_bundle(project_name, snapshot):
    files = (snapshot or {}).get("files", {})
    manifest = (snapshot or {}).get("manifest", {})
    pdf_bytes = generate_edge_impulse_pdf_report(project_name, snapshot)
    normal_df = files.get("normal_training_csv", pd.DataFrame())
    eval_df = files.get("evaluation_csv", pd.DataFrame())
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("edge_impulse_export_report.pdf", pdf_bytes)
        if isinstance(normal_df, pd.DataFrame):
            zf.writestr("edge_impulse_normal_training.csv", normal_df.to_csv(index=False))
        if isinstance(eval_df, pd.DataFrame):
            zf.writestr("edge_impulse_evaluation_all_labels.csv", eval_df.to_csv(index=False))
        zf.writestr("edge_impulse_recommended_axes.json", json.dumps(_json_safe(files.get("recommended_axes_json", [])), indent=2, ensure_ascii=False))
        zf.writestr("edge_impulse_export_manifest.json", json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False))
        zf.writestr("edge_impulse_instructions.md", files.get("instructions_md", ""))
        zf.writestr("README.txt", """EdgeTwin Studio V26.2 Edge Impulse Anomaly Export Bundle

Use this bundle to prepare a normal/baseline anomaly workflow in Edge Impulse.
Train anomaly detection primarily on normal behavior, then validate threshold and K value with real field data.
This is pilot preparation and does not certify production performance.
""")
    return zip_buf.getvalue()

# ============================================================
# V26.3 EDGE IMPULSE CLASSIFIER EXPORT
# ============================================================

EDGE_IMPULSE_CLASSIFIER_NOTE = (
    "EdgeTwin prepares labelled feature exports for Edge Impulse supervised classification workflows. "
    "Use stratified train/test splits, validate with real field data and avoid claiming production readiness from synthetic-only tests."
)


def get_edge_impulse_classifier_feature_columns(dataset_df, include_derived_scores=False):
    """Return numeric features suitable for supervised/classification CSV export."""
    return get_edge_impulse_feature_columns(dataset_df, include_derived_scores=include_derived_scores)


def rank_classifier_features(dataset_df, feature_cols=None, label_col="Label", top_n=12):
    """Simple ANOVA-like feature ranking without requiring heavy training.

    Score is based on between-class mean separation versus within-class variance.
    This is intentionally explainable for pilot reports.
    """
    if not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0 or label_col not in dataset_df.columns:
        return []
    df = dataset_df.copy()
    feature_cols = feature_cols or get_edge_impulse_classifier_feature_columns(df)
    labels = df[label_col].astype(str)
    rows = []
    for col in feature_cols:
        try:
            series = pd.to_numeric(df[col], errors="coerce")
            global_mean = float(series.mean())
            between = 0.0
            within = 0.0
            usable_classes = 0
            class_means = {}
            for label, grp in df.groupby(labels):
                vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                if len(vals) < 2:
                    continue
                usable_classes += 1
                mean = float(vals.mean())
                var = float(vals.var() or 0.0)
                class_means[str(label)] = round(mean, 6)
                between += len(vals) * ((mean - global_mean) ** 2)
                within += max(len(vals) - 1, 1) * max(var, 1e-12)
            if usable_classes < 2:
                score = 0.0
                reason = "Not enough class coverage"
            else:
                raw = between / max(within, 1e-12)
                score = float(np.clip(raw * 18, 0, 100))
                reason = "Strong class separation" if score >= 70 else ("Moderate class separation" if score >= 35 else "Weak but potentially useful")
            rows.append({
                "feature": col,
                "classifier_value_score": round(score, 2),
                "reason": reason,
                "class_means": class_means,
            })
        except Exception as exc:
            rows.append({
                "feature": col,
                "classifier_value_score": 0.0,
                "reason": f"Skipped: {exc}",
                "class_means": {},
            })
    rows = sorted(rows, key=lambda r: r.get("classifier_value_score", 0), reverse=True)
    return rows[:int(top_n)]


def _stratified_split_dataframe(df, label_col="Label", test_size=0.2, seed=42):
    """Deterministic stratified split that handles small classes gracefully."""
    if not isinstance(df, pd.DataFrame) or len(df) == 0 or label_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), ["Dataset is empty or missing Label column."]
    rng = np.random.default_rng(int(seed))
    train_parts = []
    test_parts = []
    warnings_out = []
    test_size = float(np.clip(test_size, 0.05, 0.5))
    for label, grp in df.groupby(df[label_col].astype(str), sort=False):
        grp = grp.copy()
        idx = np.array(grp.index.tolist())
        rng.shuffle(idx)
        n = len(idx)
        if n < 2:
            train_parts.append(grp)
            warnings_out.append(f"Class '{label}' has only {n} sample; it was kept in train only.")
            continue
        n_test = int(round(n * test_size))
        n_test = max(1, min(n_test, n - 1))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        train_parts.append(df.loc[train_idx])
        test_parts.append(df.loc[test_idx])
        if n < 10:
            warnings_out.append(f"Class '{label}' has only {n} samples; add more real samples before trusting validation metrics.")
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    # Shuffle final order for realistic upload/export.
    if len(train_df):
        train_df = train_df.sample(frac=1, random_state=int(seed)).reset_index(drop=True)
    if len(test_df):
        test_df = test_df.sample(frac=1, random_state=int(seed)+1).reset_index(drop=True)
    return train_df, test_df, warnings_out


def build_edge_impulse_classifier_instructions(project_name, summary):
    labels = summary.get("labels", [])
    features = [x.get("feature") for x in summary.get("recommended_features", [])[:8]]
    return f"""# Edge Impulse Classifier Export - {project_name}

## Purpose
This bundle prepares a labelled dataset for a supervised Edge Impulse classification workflow.
Use it for classes such as normal, bearing wear, drilling, impact, vehicle, critical tamper or failure risk.

## Included files
- `edge_impulse_classifier_train.csv`: stratified training split.
- `edge_impulse_classifier_test.csv`: stratified holdout/evaluation split.
- `edge_impulse_classifier_full.csv`: full labelled feature dataset.
- `edge_impulse_classifier_feature_ranking.json`: recommended feature ranking.
- `edge_impulse_classifier_manifest.json`: export metadata and safety notes.

## Labels/classes
{', '.join(map(str, labels))}

## Recommended feature columns
{', '.join([str(x) for x in features]) if features else 'No feature recommendation available yet.'}

## Suggested workflow
1. Create or open an Edge Impulse project.
2. Import the CSV files with the Studio uploader, CSV Wizard or CLI uploader.
3. Make sure the `Label` column is treated as the class label.
4. Build an impulse for classification. For tabular features, keep the exported numeric feature columns. For raw audio projects, use raw WAV upload and spectral/MFE/MFCC blocks instead.
5. Train a classifier and inspect confusion matrix, per-class precision/recall and false positives.
6. Test with the holdout CSV and, more importantly, real field data.
7. Only move toward deployment after field validation on the target hardware/environment.

## Important warning
{EDGE_IMPULSE_CLASSIFIER_NOTE}
"""


def build_edge_impulse_classifier_export_snapshot(
    project_name,
    dataset_df,
    workflow="Supervised event classification",
    test_size=0.2,
    include_derived_scores=False,
    seed=42,
    min_samples_per_class=10,
    max_features=12,
):
    if not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0:
        empty_summary = {
            "export_readiness_score": 0,
            "decision": "NO-GO",
            "verdict": "No dataset available for classifier export.",
            "warnings": ["Generate, upload or bridge a dataset first."],
        }
        return {"summary": empty_summary, "files": {}}
    if "Label" not in dataset_df.columns:
        empty_summary = {
            "export_readiness_score": 0,
            "decision": "NO-GO",
            "verdict": "Dataset needs a Label column for supervised classification.",
            "warnings": ["Add a Label column before exporting."],
        }
        return {"summary": empty_summary, "files": {}}

    df = dataset_df.copy().replace([np.inf, -np.inf], np.nan)
    df["Label"] = df["Label"].astype(str)
    feature_cols = get_edge_impulse_classifier_feature_columns(df, include_derived_scores=include_derived_scores)
    feature_cols = feature_cols[: max(1, int(max_features))]
    export_cols = ["Label"] + feature_cols
    export_df = df[export_cols].dropna().copy()
    warnings_out = []

    labels = sorted(export_df["Label"].dropna().unique().tolist()) if len(export_df) else []
    counts = export_df["Label"].value_counts().to_dict() if len(export_df) else {}
    low_classes = [str(k) for k, v in counts.items() if int(v) < int(min_samples_per_class)]

    if len(feature_cols) < 2:
        warnings_out.append("Fewer than 2 usable numeric features found. Add signal/fusion features before classifier export.")
    if len(labels) < 2:
        warnings_out.append("Supervised classification needs at least 2 labels/classes.")
    if low_classes:
        warnings_out.append("Low-sample classes need more data: " + ", ".join(low_classes))

    train_df, test_df, split_warnings = _stratified_split_dataframe(export_df, "Label", test_size, seed)
    warnings_out.extend(split_warnings)

    feature_ranking = rank_classifier_features(export_df, feature_cols=feature_cols, top_n=max_features)
    audit = dataset_doctor(export_df[feature_cols], export_df["Label"]) if len(feature_cols) and len(labels) >= 1 else {"overall_score": 0, "separation_score": 0, "balance_score": 0, "diversity_score": 0, "advice": []}

    balance_ratio = 0.0
    if counts:
        values = list(counts.values())
        balance_ratio = min(values) / max(max(values), 1)
    readiness = 0
    readiness += min(25, len(export_df) / 8)
    readiness += 20 if len(labels) >= 2 else 0
    readiness += 20 if all(v >= min_samples_per_class for v in counts.values()) and counts else 0
    readiness += min(15, len(feature_cols) * 2)
    readiness += min(20, float(audit.get("separation_score", 0)) * 0.20)
    if balance_ratio >= 0.5:
        readiness += 10
    elif balance_ratio >= 0.25:
        readiness += 4
    readiness = int(np.clip(readiness, 0, 100))

    if readiness >= 78 and not low_classes:
        decision = "GO"
        verdict = "Good classifier export candidate for pilot training. Validate with real holdout/field data before deployment."
    elif readiness >= 55:
        decision = "CONDITIONAL"
        verdict = "Usable for prototype classifier training, but improve weak classes/features before paid pilot claims."
    else:
        decision = "NO-GO"
        verdict = "Not ready for supervised classifier export. Add labels, samples or stronger features first."

    class_rows = []
    for label in labels:
        n = int(counts.get(label, 0))
        class_rows.append({
            "label": label,
            "samples": n,
            "status": "OK" if n >= min_samples_per_class else "LOW_SAMPLE",
            "recommended_minimum": int(min_samples_per_class),
            "additional_samples_needed": int(max(0, int(min_samples_per_class) - n)),
        })
    class_summary_df = pd.DataFrame(class_rows)

    summary = {
        "project_name": project_name,
        "workflow": workflow,
        "export_readiness_score": readiness,
        "decision": decision,
        "verdict": verdict,
        "labels": labels,
        "label_counts": counts,
        "total_samples": int(len(export_df)),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "test_size": float(test_size),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "recommended_features": feature_ranking,
        "audit": audit,
        "balance_ratio": round(float(balance_ratio), 4),
        "warnings": warnings_out,
        "edge_impulse_steps": [
            "Upload the train CSV as labelled training data.",
            "Upload the test/evaluation CSV separately or reserve it for manual testing.",
            "Treat the Label column as the class label.",
            "Train a classifier and inspect the confusion matrix per class.",
            "Use real field data before claiming pilot or production performance.",
        ],
        "safety_note": EDGE_IMPULSE_CLASSIFIER_NOTE,
    }
    instructions = build_edge_impulse_classifier_instructions(project_name, summary)
    manifest = {
        "engine": "EdgeTwin Studio V26.3 Edge Impulse Classifier Export",
        "created_at": _now(),
        "summary": summary,
        "feature_ranking": feature_ranking,
        "class_summary": class_rows,
        "files": [
            "edge_impulse_classifier_train.csv",
            "edge_impulse_classifier_test.csv",
            "edge_impulse_classifier_full.csv",
            "edge_impulse_classifier_feature_ranking.json",
            "edge_impulse_classifier_manifest.json",
            "edge_impulse_classifier_instructions.md",
        ],
    }
    return {
        "summary": summary,
        "files": {
            "train_csv": train_df,
            "test_csv": test_df,
            "full_csv": export_df,
            "class_summary_csv": class_summary_df,
            "feature_ranking_json": feature_ranking,
            "manifest": manifest,
            "instructions_md": instructions,
        },
    }


def generate_edge_impulse_classifier_pdf_report(project_name, snapshot):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Edge Impulse Classifier Export", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)
    summary = snapshot.get("summary", {}) if isinstance(snapshot, dict) else {}
    safe_pdf_cell(pdf, "Export Summary", 8, True)
    safe_pdf_cell(pdf, f"Readiness: {summary.get('export_readiness_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {summary.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Samples: {summary.get('total_samples', 0)} total / {summary.get('train_samples', 0)} train / {summary.get('test_samples', 0)} test")
    safe_pdf_cell(pdf, f"Labels: {len(summary.get('labels', []))}")
    safe_pdf_cell(pdf, f"Features: {summary.get('feature_count', 0)}")
    safe_pdf_multicell(pdf, summary.get("verdict", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Class Summary", 8, True)
    for label, count in (summary.get("label_counts", {}) or {}).items():
        safe_pdf_cell(pdf, f"- {label}: {count} samples")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Recommended Features", 8, True)
    for row in summary.get("recommended_features", [])[:10]:
        safe_pdf_multicell(pdf, f"{row.get('feature')}: {row.get('classifier_value_score')} - {row.get('reason')}")
    if summary.get("warnings"):
        pdf.ln(4)
        safe_pdf_cell(pdf, "Warnings", 8, True)
        for warning in summary.get("warnings", []):
            safe_pdf_multicell(pdf, f"- {warning}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Edge Impulse Steps", 8, True)
    for step in summary.get("edge_impulse_steps", []):
        safe_pdf_multicell(pdf, f"- {step}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Safety Note", 8, True)
    safe_pdf_multicell(pdf, EDGE_IMPULSE_CLASSIFIER_NOTE)
    return safe_pdf_output(pdf)


def create_edge_impulse_classifier_bundle(project_name, snapshot):
    pdf_bytes = generate_edge_impulse_classifier_pdf_report(project_name, snapshot)
    files = snapshot.get("files", {}) if isinstance(snapshot, dict) else {}
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("edge_impulse_classifier_report.pdf", pdf_bytes)
        for key, filename in [
            ("train_csv", "edge_impulse_classifier_train.csv"),
            ("test_csv", "edge_impulse_classifier_test.csv"),
            ("full_csv", "edge_impulse_classifier_full.csv"),
            ("class_summary_csv", "edge_impulse_classifier_class_summary.csv"),
        ]:
            obj = files.get(key)
            if isinstance(obj, pd.DataFrame):
                zf.writestr(filename, obj.to_csv(index=False))
        zf.writestr("edge_impulse_classifier_feature_ranking.json", json.dumps(_json_safe(files.get("feature_ranking_json", [])), indent=2, ensure_ascii=False))
        zf.writestr("edge_impulse_classifier_manifest.json", json.dumps(_json_safe(files.get("manifest", {})), indent=2, ensure_ascii=False))
        zf.writestr("edge_impulse_classifier_instructions.md", files.get("instructions_md", ""))
    return zip_buf.getvalue()



# ============================================================
# V26.4 RELEASE SUCCESS GATE / CUSTOMER READINESS
# ============================================================

RELEASE_SUCCESS_NOTE = (
    "The Release Success Gate is a product-readiness and customer-delivery check. "
    "It does not certify a safety-critical deployment and it does not replace field validation, "
    "Edge Impulse training metrics, hardware testing or customer acceptance testing."
)


def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _snapshot_score(snapshot, *keys, default=0):
    if not isinstance(snapshot, dict):
        return default
    cur = snapshot
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return _safe_int(cur, default)


def _release_dataset_health(dataset_df):
    result = {
        "score": 0,
        "samples": 0,
        "feature_count": 0,
        "labels": [],
        "label_counts": {},
        "issues": [],
    }
    if not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0:
        result["issues"].append("No dataset loaded.")
        return result
    df = dataset_df.copy().replace([np.inf, -np.inf], np.nan)
    result["samples"] = int(len(df))
    numeric_cols = [c for c in df.columns if c != "Label" and pd.api.types.is_numeric_dtype(df[c])]
    result["feature_count"] = int(len(numeric_cols))
    score = 0
    if len(df) >= 100:
        score += 25
    elif len(df) >= 30:
        score += 14
    else:
        score += 5
        result["issues"].append("Dataset is small; paid pilot evidence should be stronger.")
    if "Label" in df.columns:
        labels = sorted(df["Label"].astype(str).dropna().unique().tolist())
        counts = df["Label"].astype(str).value_counts().to_dict()
        result["labels"] = labels
        result["label_counts"] = {str(k): int(v) for k, v in counts.items()}
        if len(labels) >= 2:
            score += 25
        else:
            result["issues"].append("Dataset has fewer than 2 labels/classes.")
        if counts:
            min_count = min(counts.values())
            max_count = max(counts.values())
            if min_count >= 20:
                score += 16
            elif min_count >= 10:
                score += 10
            else:
                result["issues"].append("At least one class has fewer than 10 samples.")
            if min_count / max(max_count, 1) >= 0.50:
                score += 12
            else:
                result["issues"].append("Class balance is weak.")
    else:
        result["issues"].append("Dataset has no Label column.")
    if len(numeric_cols) >= 4:
        score += 15
    elif len(numeric_cols) >= 2:
        score += 9
    else:
        result["issues"].append("Dataset has too few numeric features for a strong pilot.")
    missing_ratio = float(df[numeric_cols].isna().mean().mean()) if numeric_cols else 1.0
    if missing_ratio <= 0.01:
        score += 7
    elif missing_ratio <= 0.05:
        score += 3
        result["issues"].append("Dataset contains some missing numeric values.")
    else:
        result["issues"].append("Dataset contains many missing numeric values.")
    result["score"] = int(np.clip(score, 0, 100))
    return result


def build_release_success_gate_snapshot(
    project_name,
    dataset_df,
    intended_offer="Paid pilot",
    customer_type="Industrial pilot team",
    selected_plan="Founder Test Mode",
    trust_gate=None,
    reliability_v2=None,
    real_bridge_result=None,
    deployment_plan=None,
    professional_report_snapshot=None,
    hardening_snapshot=None,
    beta_launch_snapshot=None,
    monetization_snapshot=None,
    normality_result=None,
    edge_impulse_snapshot=None,
    edge_impulse_classifier_snapshot=None,
):
    dataset_health = _release_dataset_health(dataset_df)

    trust_score = _snapshot_score(trust_gate, "trust_score", default=0)
    if trust_score == 0:
        trust_score = _snapshot_score(reliability_v2, "trust_score_v2", default=0)
    reliability_score = _snapshot_score(reliability_v2, "trust_score_v2", default=0)
    if reliability_score == 0:
        reliability_score = _snapshot_score(reliability_v2, "reliability_score", default=0)

    hardening_score = _snapshot_score(hardening_snapshot, "product_readiness_score", default=0)
    beta_score = _snapshot_score(beta_launch_snapshot, "beta_readiness_score", default=0)
    monetization_score = _snapshot_score(monetization_snapshot, "monetization_readiness_score", default=0)
    report_score = 85 if isinstance(professional_report_snapshot, dict) and professional_report_snapshot else 0
    deployment_score = 82 if isinstance(deployment_plan, dict) and deployment_plan else 0

    anomaly_score = _snapshot_score(edge_impulse_snapshot, "summary", "export_readiness_score", default=0)
    classifier_score = _snapshot_score(edge_impulse_classifier_snapshot, "summary", "export_readiness_score", default=0)
    ei_score = int(np.clip(max(anomaly_score, classifier_score) * 0.75 + min(anomaly_score, classifier_score) * 0.25, 0, 100)) if max(anomaly_score, classifier_score) else 0

    real_bridge_score = 0
    if isinstance(real_bridge_result, dict) and real_bridge_result:
        real_bridge_score = _snapshot_score(real_bridge_result, "summary", "similarity_score", default=0)
        if real_bridge_score == 0:
            real_bridge_score = 70
    normality_score = 0
    if isinstance(normality_result, dict) and normality_result:
        normality_score = _snapshot_score(normality_result, "summary", "baseline_quality_score", default=0)
        if normality_score == 0:
            normality_score = _snapshot_score(normality_result, "baseline_quality_score", default=65)

    evidence = [
        {"evidence": "Dataset loaded", "available": dataset_health["samples"] > 0, "score": dataset_health["score"], "why_it_matters": "Every customer bundle starts with a valid dataset."},
        {"evidence": "Trust / Reliability generated", "available": bool(trust_gate or reliability_v2), "score": max(trust_score, reliability_score), "why_it_matters": "Paid output needs honest risk and readiness scoring."},
        {"evidence": "Real-data bridge", "available": bool(real_bridge_result), "score": real_bridge_score, "why_it_matters": "Real data increases credibility and reduces synthetic-only risk."},
        {"evidence": "Normality baseline", "available": bool(normality_result), "score": normality_score, "why_it_matters": "Customers need to know what normal vs abnormal means."},
        {"evidence": "Deployment plan", "available": bool(deployment_plan), "score": deployment_score, "why_it_matters": "Hardware/BOM/field risks make the bundle actionable."},
        {"evidence": "Professional report", "available": bool(professional_report_snapshot), "score": report_score, "why_it_matters": "The customer must receive decision-ready documentation."},
        {"evidence": "Product hardening", "available": bool(hardening_snapshot), "score": hardening_score, "why_it_matters": "Prevents crashes, unsafe claims and bad files before launch."},
        {"evidence": "Beta launch readiness", "available": bool(beta_launch_snapshot), "score": beta_score, "why_it_matters": "Makes the demo understandable without heavy manual explanation."},
        {"evidence": "Monetization gate", "available": bool(monetization_snapshot), "score": monetization_score, "why_it_matters": "Keeps free/paid value clear and avoids underpricing."},
        {"evidence": "Edge Impulse export route", "available": bool(edge_impulse_snapshot or edge_impulse_classifier_snapshot), "score": ei_score, "why_it_matters": "Gives the customer a practical next step toward model training."},
    ]

    data_score = dataset_health["score"]
    trust_area = max(trust_score, reliability_score)
    product_area = int(np.clip(hardening_score * 0.60 + beta_score * 0.25 + monetization_score * 0.15, 0, 100)) if any([hardening_score, beta_score, monetization_score]) else 0
    delivery_area = int(np.clip(report_score * 0.35 + deployment_score * 0.25 + ei_score * 0.25 + normality_score * 0.15, 0, 100))
    evidence_area = int(np.clip(sum(1 for e in evidence if e["available"]) / max(len(evidence), 1) * 100, 0, 100))

    success_score = int(np.clip(
        data_score * 0.24 +
        trust_area * 0.22 +
        product_area * 0.20 +
        delivery_area * 0.22 +
        evidence_area * 0.12,
        0, 100,
    ))

    blockers = []
    blockers.extend(dataset_health["issues"])
    if dataset_health["samples"] == 0:
        blockers.append("Generate or upload a dataset before any customer demo.")
    if intended_offer in ["Paid pilot", "Real-data pilot", "Enterprise review"] and not (trust_gate or reliability_v2):
        blockers.append("Run Trust Center / Reliability Engine before paid claims.")
    if intended_offer in ["Paid pilot", "Real-data pilot", "Enterprise review"] and not professional_report_snapshot:
        blockers.append("Create Reports 2.0 output before charging for a professional bundle.")
    if intended_offer in ["Real-data pilot", "Enterprise review"] and not real_bridge_result:
        blockers.append("Real-data offer selected but no Real Bridge evidence exists.")
    if intended_offer in ["Paid pilot", "Real-data pilot", "Enterprise review"] and hardening_score < 60:
        blockers.append("Product Hardening score is too low or missing for paid customer delivery.")
    if not (edge_impulse_snapshot or edge_impulse_classifier_snapshot):
        blockers.append("No Edge Impulse export route prepared yet.")

    critical_blockers = []
    for b in blockers:
        if any(term in b.lower() for term in ["no dataset", "generate or upload", "no label", "fewer than 2", "too low or missing"]):
            critical_blockers.append(b)

    if success_score >= 82 and len(critical_blockers) == 0 and len(blockers) <= 1:
        decision = "GO"
        commercial_status = "Paid pilot candidate"
        risk_level = "Controlled"
        verdict = "Strong enough for a controlled beta or paid pilot conversation, as long as the report keeps field-validation warnings visible."
    elif success_score >= 64 and len(critical_blockers) == 0:
        decision = "CONDITIONAL GO"
        commercial_status = "Private beta / discounted pilot"
        risk_level = "Medium"
        verdict = "Usable for a private beta or discounted pilot, but fix the listed blockers before charging full professional pricing."
    else:
        decision = "NO-GO"
        commercial_status = "Internal demo only"
        risk_level = "High"
        verdict = "Do not sell this output yet. Use it internally, fix blockers, then re-run the Success Gate."

    recommended_actions = []
    if dataset_health["samples"] < 100:
        recommended_actions.append("Increase dataset size to at least 100 samples for demos and 200+ for stronger pilot confidence.")
    if not (trust_gate or reliability_v2):
        recommended_actions.append("Run Trust Center / Reliability Engine 2.0 before showing the output to customers.")
    if not real_bridge_result and intended_offer in ["Real-data pilot", "Enterprise review"]:
        recommended_actions.append("Upload real WAV/CSV files and run Synthetic-to-Real Bridge before offering real-data pricing.")
    if not normality_result:
        recommended_actions.append("Run Normality Engine so the product can explain what is normal and what is abnormal.")
    if not deployment_plan:
        recommended_actions.append("Run Deployment Planner to add BOM, communication and field-test risks.")
    if not professional_report_snapshot:
        recommended_actions.append("Create Reports 2.0 bundle before professional customer delivery.")
    if hardening_score < 70:
        recommended_actions.append("Run/fix Product Hardening until no critical blockers remain.")
    if not (edge_impulse_snapshot or edge_impulse_classifier_snapshot):
        recommended_actions.append("Prepare at least one Edge Impulse export: anomaly for normal-vs-abnormal or classifier for labelled events.")
    if not recommended_actions:
        recommended_actions.append("Use this as a controlled customer demo/pilot package. Keep field-validation disclaimers in every report.")

    score_breakdown = [
        {"area": "Dataset health", "score": data_score, "weight": "24%"},
        {"area": "Trust/Reliability", "score": trust_area, "weight": "22%"},
        {"area": "Product hardening/commercial", "score": product_area, "weight": "20%"},
        {"area": "Delivery/export readiness", "score": delivery_area, "weight": "22%"},
        {"area": "Evidence coverage", "score": evidence_area, "weight": "12%"},
    ]

    claims_check = [
        {"claim": "Can prepare a sensor AI pilot package", "status": "Allowed", "reason": "Supported by generated dataset, report and export bundles."},
        {"claim": "Can reduce early trial-and-error", "status": "Allowed", "reason": "The app structures labels, features, risks and hardware direction."},
        {"claim": "Production-ready model guaranteed", "status": "Avoid", "reason": "Requires real-world validation and Edge Impulse/model metrics."},
        {"claim": "Synthetic data replaces real field data", "status": "Avoid", "reason": "Synthetic data is for pilot preparation and augmentation, not final proof."},
        {"claim": "Can export toward Edge Impulse workflows", "status": "Allowed", "reason": "Anomaly/classifier export bundles are generated when prepared."},
    ]

    safe_customer_wording = (
        "EdgeTwin Studio can generate a pilot-preparation package for your audio/vibration Edge AI use-case: "
        "dataset, audit, reliability/readiness checks, hardware direction, report and Edge Impulse export guidance. "
        "Final model performance and production deployment must be validated with real field data."
    )

    summary = {
        "project_name": project_name,
        "created_at": _now(),
        "engine": "EdgeTwin Studio V26.4 Release Success Gate",
        "intended_offer": intended_offer,
        "customer_type": customer_type,
        "selected_plan": selected_plan,
        "success_score": success_score,
        "decision": decision,
        "commercial_status": commercial_status,
        "risk_level": risk_level,
        "verdict": verdict,
        "safe_customer_wording": safe_customer_wording,
        "release_note": RELEASE_SUCCESS_NOTE,
    }

    return {
        "summary": summary,
        "dataset_health": dataset_health,
        "score_breakdown": score_breakdown,
        "evidence": evidence,
        "blockers": blockers,
        "critical_blockers": critical_blockers,
        "recommended_actions": recommended_actions,
        "claims_check": claims_check,
        "source_scores": {
            "trust_score": trust_score,
            "reliability_score": reliability_score,
            "hardening_score": hardening_score,
            "beta_score": beta_score,
            "monetization_score": monetization_score,
            "real_bridge_score": real_bridge_score,
            "normality_score": normality_score,
            "edge_impulse_anomaly_score": anomaly_score,
            "edge_impulse_classifier_score": classifier_score,
            "deployment_score": deployment_score,
            "report_score": report_score,
        },
    }


def generate_release_success_gate_pdf(project_name, snapshot):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Release Success Gate", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)
    summary = (snapshot or {}).get("summary", {})
    safe_pdf_cell(pdf, "Release Decision", 8, True)
    safe_pdf_cell(pdf, f"Success score: {summary.get('success_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {summary.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Commercial status: {summary.get('commercial_status', 'Unknown')}")
    safe_pdf_cell(pdf, f"Risk level: {summary.get('risk_level', 'Unknown')}")
    safe_pdf_multicell(pdf, summary.get("verdict", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Score Breakdown", 8, True)
    for row in (snapshot or {}).get("score_breakdown", []):
        safe_pdf_cell(pdf, f"{row.get('area')}: {row.get('score')}% ({row.get('weight')})")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Must-fix Blockers", 8, True)
    blockers = (snapshot or {}).get("blockers", [])
    if blockers:
        for item in blockers:
            safe_pdf_multicell(pdf, f"- {item}")
    else:
        safe_pdf_multicell(pdf, "No must-fix blockers detected by the Success Gate.")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Recommended Actions", 8, True)
    for item in (snapshot or {}).get("recommended_actions", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Safe Customer Wording", 8, True)
    safe_pdf_multicell(pdf, summary.get("safe_customer_wording", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Claims Check", 8, True)
    for row in (snapshot or {}).get("claims_check", []):
        safe_pdf_multicell(pdf, f"{row.get('status')}: {row.get('claim')} - {row.get('reason')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Important Note", 8, True)
    safe_pdf_multicell(pdf, RELEASE_SUCCESS_NOTE)
    return safe_pdf_output(pdf)


def create_release_success_gate_bundle(project_name, snapshot, dataset_df=None):
    pdf_bytes = generate_release_success_gate_pdf(project_name, snapshot)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("release_success_gate_report.pdf", pdf_bytes)
        zf.writestr("release_success_gate_snapshot.json", json.dumps(_json_safe(snapshot or {}), indent=2, ensure_ascii=False))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr("dataset_snapshot.csv", dataset_df.head(5000).to_csv(index=False))
        for key, filename in [
            ("score_breakdown", "score_breakdown.csv"),
            ("evidence", "evidence_coverage.csv"),
            ("claims_check", "claims_check.csv"),
        ]:
            rows = (snapshot or {}).get(key, [])
            if rows:
                zf.writestr(filename, pd.DataFrame(rows).to_csv(index=False))
        blockers = (snapshot or {}).get("blockers", [])
        actions = (snapshot or {}).get("recommended_actions", [])
        zf.writestr("release_action_list.txt", "Must-fix blockers:\n" + "\n".join(f"- {b}" for b in blockers) + "\n\nRecommended actions:\n" + "\n".join(f"- {a}" for a in actions))
        zf.writestr("README.txt", f"""EdgeTwin Studio V26.4 Release Success Gate Bundle

Project: {project_name}
Decision: {(snapshot or {}).get('summary', {}).get('decision', 'Unknown')}
Success score: {(snapshot or {}).get('summary', {}).get('success_score', 0)}%
Commercial status: {(snapshot or {}).get('summary', {}).get('commercial_status', 'Unknown')}

{RELEASE_SUCCESS_NOTE}
""")
    return zip_buf.getvalue()


# ============================================================
# V27 GOLDEN DEMO SUITE / CUSTOMER PAIN PROOF MATRIX
# ============================================================

GOLDEN_DEMO_NOTE = (
    "The Golden Demo Suite is an end-to-end pilot-preparation proof flow. "
    "It demonstrates that EdgeTwin Studio can move from a customer problem to dataset, audit, "
    "normality, reliability, hardware/deployment direction, Edge Impulse export guidance, report and Success Gate. "
    "It is not a production certification and must be validated with real field data before deployment claims."
)

GOLDEN_DEMO_SCENARIOS = {
    "predictive_maintenance_golden": {
        "title": "Predictive Maintenance Golden Demo",
        "customer_problem": "Machine teams want early warning for bearing wear, imbalance and failure risk, but they lack labelled vibration/audio data and do not know how much real data is still needed.",
        "use_case_type": "Predictive Maintenance",
        "environment": "Industrial",
        "selected_sensors": ["Vibration", "Audio", "Gas / Environment"],
        "classes": ["Healthy", "Early_Wear", "Wear", "Failure_Risk", "Critical_Failure"],
        "normal_labels": ["Healthy"],
        "samples": 650,
        "priority": "performance",
        "sample_rate": 4000,
        "deployment_scale": "Pilot: 1-5 nodes",
        "communication": "WiFi / MQTT",
        "power_source": "Mains + small backup",
        "enclosure_target": "Industrial / IP54",
        "target_customer": "Maintenance manager / reliability engineer",
        "sales_angle": "Reduce early trial-and-error before a vibration/audio predictive-maintenance pilot.",
    },
    "acoustic_tamper_golden": {
        "title": "Acoustic Tamper Golden Demo",
        "customer_problem": "Security teams need to detect drilling, grinding, impact and tamper events before theft damage escalates, but real labelled event data is rare.",
        "use_case_type": "Security / Tamper",
        "environment": "Construction site",
        "selected_sensors": ["Audio", "Vibration", "Radar", "GPS / Zone"],
        "classes": ["Normal", "Handling", "Impact", "Tool_Use", "Critical_Tamper"],
        "normal_labels": ["Normal"],
        "samples": 650,
        "priority": "balanced",
        "sample_rate": 16000,
        "deployment_scale": "Pilot: 1-5 nodes",
        "communication": "LTE / NB-IoT",
        "power_source": "Battery only",
        "enclosure_target": "Outdoor / IP65",
        "target_customer": "Construction security / asset protection team",
        "sales_angle": "Prepare an audio/vibration tamper pilot with honest risk and hardware guidance.",
    },
    "remote_forestry_golden": {
        "title": "Remote Forestry / Asset Golden Demo",
        "customer_problem": "Remote assets and forests are hard to monitor because events happen far away, data is sparse and power/connectivity choices are risky.",
        "use_case_type": "Smart Forestry / Remote Area",
        "environment": "Forest / remote outdoor",
        "selected_sensors": ["Audio", "Vibration", "Radar", "GPS / Zone"],
        "classes": ["Normal", "Human_Activity", "Vehicle", "Chainsaw", "Critical_Threat"],
        "normal_labels": ["Normal"],
        "samples": 650,
        "priority": "low_power",
        "sample_rate": 16000,
        "deployment_scale": "Pilot: 1-5 nodes",
        "communication": "LoRa / LoRaWAN",
        "power_source": "Battery only",
        "enclosure_target": "Outdoor / IP65",
        "target_customer": "Forestry / agriculture / remote asset owner",
        "sales_angle": "Turn remote monitoring ideas into a practical low-power pilot plan.",
    },
}


def get_golden_demo_scenarios():
    return list(GOLDEN_DEMO_SCENARIOS.keys())


def get_golden_demo_scenario(scenario_id):
    return GOLDEN_DEMO_SCENARIOS.get(scenario_id, GOLDEN_DEMO_SCENARIOS["predictive_maintenance_golden"])


def _compact_step(name, status, score=0, evidence="", risk="", output=""):
    return {
        "step": name,
        "status": status,
        "score": int(np.clip(_safe_int(score, 0), 0, 100)),
        "evidence": evidence,
        "risk": risk,
        "output": output,
    }


def _decision_from_score(score):
    score = _safe_int(score, 0)
    if score >= 82:
        return "GO"
    if score >= 62:
        return "CONDITIONAL GO"
    return "NO-GO"


def build_customer_pain_proof_matrix(scenario, dataset_df=None, doctor=None, reliability_v2=None, trust_gate=None, normality_result=None, deployment_plan=None, edge_impulse_snapshot=None, edge_impulse_classifier_snapshot=None, professional_report_snapshot=None, release_success_snapshot=None):
    """Translate technical evidence into customer-pain proof points."""
    df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    rows = int(len(df)) if len(df) else 0
    labels = int(df["Label"].nunique()) if len(df) and "Label" in df.columns else 0
    data_score = int((doctor or {}).get("overall_score", 0)) if isinstance(doctor, dict) else 0
    rel_score = int((reliability_v2 or {}).get("trust_score_v2", (reliability_v2 or {}).get("reliability_score", 0))) if isinstance(reliability_v2, dict) else 0
    normality_score = int(((normality_result or {}).get("summary", {}) or {}).get("normality_score", 0)) if isinstance(normality_result, dict) else 0
    deployment_go = (deployment_plan or {}).get("go_no_go", "Missing") if isinstance(deployment_plan, dict) else "Missing"
    anomaly_score = int(((edge_impulse_snapshot or {}).get("summary", {}) or {}).get("export_readiness_score", 0)) if isinstance(edge_impulse_snapshot, dict) else 0
    classifier_score = int(((edge_impulse_classifier_snapshot or {}).get("summary", {}) or {}).get("export_readiness_score", 0)) if isinstance(edge_impulse_classifier_snapshot, dict) else 0
    report_score = int(((professional_report_snapshot or {}).get("readiness", {}) or {}).get("trust_score", 0)) if isinstance(professional_report_snapshot, dict) else 0
    success_score = int(((release_success_snapshot or {}).get("summary", {}) or {}).get("success_score", 0)) if isinstance(release_success_snapshot, dict) else 0

    def status(score, hard=False):
        if hard and score <= 0:
            return "Missing"
        if score >= 75:
            return "Strong proof"
        if score >= 50:
            return "Partial proof"
        return "Weak / needs work"

    return [
        {
            "customer_pain": "Too little real or labelled data to start a pilot",
            "why_it_matters": "Without a first dataset, teams cannot judge labels, features, hardware or model direction.",
            "edgetwin_proof": f"Generated {rows} pilot rows across {labels} classes.",
            "evidence_source": "Auto Pilot Generator + generated training dataset",
            "proof_status": status(data_score),
            "customer_value": "Gives the customer a structured starting point instead of a blank page.",
        },
        {
            "customer_pain": "Unclear labels/classes and weak class separation",
            "why_it_matters": "Bad labels produce bad models and unreliable demos.",
            "edgetwin_proof": f"Dataset Doctor overall score: {data_score}%.",
            "evidence_source": "Dataset Doctor / Enterprise Audit",
            "proof_status": status(data_score),
            "customer_value": "Shows which labels/classes need more samples or stronger separation.",
        },
        {
            "customer_pain": "Not knowing what normal vs abnormal looks like",
            "why_it_matters": "Anomaly detection needs a trustworthy normal baseline.",
            "edgetwin_proof": f"Normality baseline score: {normality_score}%.",
            "evidence_source": "Normality Engine",
            "proof_status": status(normality_score, hard=True),
            "customer_value": "Prepares the customer for anomaly/K-means workflows and baseline validation.",
        },
        {
            "customer_pain": "No honest reliability/readiness view",
            "why_it_matters": "Teams need to know whether an output is demo-ready, pilot-ready or still too risky.",
            "edgetwin_proof": f"Reliability/Trust score: {rel_score}%.",
            "evidence_source": "Reliability Engine 2.0 + Trust Center",
            "proof_status": status(rel_score),
            "customer_value": "Prevents overclaiming and makes the pilot decision safer.",
        },
        {
            "customer_pain": "Wrong or unclear hardware choice",
            "why_it_matters": "Wrong board, enclosure, communication or power plan can kill the field pilot.",
            "edgetwin_proof": f"Deployment planner decision: {deployment_go}.",
            "evidence_source": "Hardware BOM & Deployment Planner",
            "proof_status": "Strong proof" if deployment_go == "GO" else "Partial proof" if deployment_go == "CONDITIONAL" else "Weak / needs work",
            "customer_value": "Turns data planning into a practical node/gateway/BOM/field-test plan.",
        },
        {
            "customer_pain": "Difficult route into Edge Impulse / TinyML workflows",
            "why_it_matters": "Customers often get stuck between data, features and model workflow setup.",
            "edgetwin_proof": f"Anomaly export score: {anomaly_score}%, classifier export score: {classifier_score}%.",
            "evidence_source": "Edge Impulse Anomaly + Classifier Export",
            "proof_status": status(max(anomaly_score, classifier_score), hard=True),
            "customer_value": "Provides training/evaluation CSVs and instructions for anomaly or classifier routes.",
        },
        {
            "customer_pain": "No professional report for internal buy-in",
            "why_it_matters": "A pilot often needs management approval, not only technical files.",
            "edgetwin_proof": f"Professional report readiness score: {report_score}%.",
            "evidence_source": "Reports 2.0",
            "proof_status": status(report_score, hard=True),
            "customer_value": "Makes the output look like a decision package instead of a loose dataset export.",
        },
        {
            "customer_pain": "No clear go/no-go decision",
            "why_it_matters": "Teams need a safe answer: show it, sell it, or keep improving it.",
            "edgetwin_proof": f"Release Success score: {success_score}%.",
            "evidence_source": "Release Success Gate",
            "proof_status": status(success_score, hard=True),
            "customer_value": "Converts many technical checks into one customer-delivery decision.",
        },
    ]


def summarize_golden_steps(step_rows):
    if not step_rows:
        return {"overall_score": 0, "decision": "NO-GO", "pass_count": 0, "warn_count": 0, "fail_count": 0}
    scores = [_safe_int(r.get("score", 0)) for r in step_rows]
    pass_count = sum(1 for r in step_rows if str(r.get("status", "")).upper() in ["PASS", "GO"])
    warn_count = sum(1 for r in step_rows if "WARN" in str(r.get("status", "")).upper() or "CONDITIONAL" in str(r.get("status", "")).upper())
    fail_count = sum(1 for r in step_rows if "FAIL" in str(r.get("status", "")).upper() or "NO-GO" in str(r.get("status", "")).upper())
    score = int(np.clip(np.mean(scores) - fail_count * 6, 0, 100))
    return {
        "overall_score": score,
        "decision": _decision_from_score(score),
        "pass_count": int(pass_count),
        "warn_count": int(warn_count),
        "fail_count": int(fail_count),
    }


def run_golden_demo_suite(scenario_id="predictive_maintenance_golden", selected_plan="Founder Test Mode", intended_offer="Paid pilot", include_optimizer=True, include_edge_impulse=True):
    """Run the V27 one-click proof flow from customer problem to delivery decision."""
    scenario = get_golden_demo_scenario(scenario_id)
    project_name = scenario.get("title", "Golden Demo").replace(" ", "_").replace("/", "_")

    labels_text = ", ".join(scenario.get("classes", []))
    config = build_use_case_config(
        use_case_type=scenario.get("use_case_type", "Custom Sensor Fusion"),
        project_goal=scenario.get("customer_problem", "Customer needs an Edge AI pilot."),
        selected_sensors=scenario.get("selected_sensors", []),
        environment=scenario.get("environment", "Custom"),
        labels_text=labels_text,
        samples=int(scenario.get("samples", 500)),
        has_real_data=False,
        output_level="Enterprise Deployment Bundle",
        priority=scenario.get("priority", "balanced"),
    )
    config["sample_rate"] = int(scenario.get("sample_rate", config.get("sample_rate", 16000)))

    steps = []
    artifacts = {}

    pilot = run_auto_pilot_project(config)
    training_df = pilot.get("training_df", pd.DataFrame()).copy()
    fusion_df = pilot.get("fusion_df", pd.DataFrame()).copy()
    manifest = pilot.get("manifest", {})
    doctor = pilot.get("doctor", {})
    hardware = pilot.get("hardware", {})
    reliability = pilot.get("reliability", {})
    artifacts["pilot"] = pilot
    steps.append(_compact_step(
        "Auto Pilot Generator",
        "PASS" if len(training_df) > 0 else "FAIL",
        doctor.get("overall_score", 0),
        f"Generated {len(training_df)} training rows and {len(fusion_df)} full fusion rows.",
        "No dataset generated." if len(training_df) == 0 else "",
        "pilot_training_features.csv",
    ))

    working_df = training_df.copy()
    optimizer_result = None
    if include_optimizer and isinstance(working_df, pd.DataFrame) and len(working_df) > 0:
        optimizer_result = run_smart_dataset_optimizer(
            working_df,
            actions=["balance_classes", "improve_label_separation"],
            label_col="Label",
            noise_strength=0.02,
            separation_strength=0.04,
        )
        opt_df = optimizer_result.get("optimized_df", working_df)
        if isinstance(opt_df, pd.DataFrame) and len(opt_df) > 0:
            working_df = opt_df
        after_score = ((optimizer_result.get("after_report", {}) or {}).get("current_score", 0))
        before_score = ((optimizer_result.get("before_report", {}) or {}).get("current_score", 0))
        artifacts["optimizer_result"] = optimizer_result
        steps.append(_compact_step(
            "Smart Dataset Optimizer",
            "PASS" if after_score >= before_score else "WARN",
            max(after_score, before_score),
            f"Before score {before_score}%, after score {after_score}%, rows now {len(working_df)}.",
            "Optimizer did not improve score; inspect class separation manually." if after_score < before_score else "",
            "optimized_dataset.csv",
        ))

    numeric_cols = [c for c in working_df.columns if c != "Label" and pd.api.types.is_numeric_dtype(working_df[c])] if isinstance(working_df, pd.DataFrame) else []
    if len(working_df) and "Label" in working_df.columns and numeric_cols:
        doctor2 = dataset_doctor(working_df[numeric_cols], working_df["Label"].astype(str))
    else:
        doctor2 = doctor
    reliability_v2 = build_reliability_engine_v2(
        working_df,
        doctor=doctor2,
        reliability=reliability,
        hardware_result=hardware,
        selected_sensors=scenario.get("selected_sensors", []),
        has_real_data=False,
    )
    artifacts["reliability_v2"] = reliability_v2
    rel_score = int(reliability_v2.get("trust_score_v2", reliability_v2.get("reliability_score", 0)))
    steps.append(_compact_step(
        "Reliability Engine 2.0",
        "PASS" if rel_score >= 65 else "WARN" if rel_score >= 45 else "FAIL",
        rel_score,
        f"Trust/reliability score {rel_score}%, production risk {reliability_v2.get('production_risk_level', 'Unknown')}.",
        "Synthetic-only evidence; real field data still required.",
        "reliability_v2_snapshot.json",
    ))

    trust_gate = build_trust_gate(working_df, doctor=doctor2, reliability=reliability, hardware_result=hardware, has_real_data=False)
    artifacts["trust_gate"] = trust_gate
    trust_score = int(trust_gate.get("data_quality_score", 0))
    steps.append(_compact_step(
        "Trust Center",
        "PASS" if trust_gate.get("go_no_go") in ["Go", "Conditional"] else "WARN",
        trust_score,
        f"Trust level: {trust_gate.get('trust_level', 'Unknown')}; package: {trust_gate.get('commercial_package', 'Unknown')}.",
        "; ".join(trust_gate.get("red_flags", [])[:2]),
        "trust_gate_snapshot.json",
    ))

    normality = run_normality_baseline_engine(
        working_df,
        normal_labels=scenario.get("normal_labels", []),
        sensitivity=1.0,
        project_context={"scenario": scenario.get("title"), "normal_labels": scenario.get("normal_labels", [])},
    )
    artifacts["normality_result"] = normality
    norm_score = int((normality.get("summary", {}) or {}).get("normality_score", 0)) if isinstance(normality, dict) else 0
    steps.append(_compact_step(
        "Normality Engine",
        "PASS" if norm_score >= 65 else "WARN" if norm_score >= 40 else "FAIL",
        norm_score,
        f"Normal baseline labels: {', '.join(scenario.get('normal_labels', []))}; normality score {norm_score}%.",
        "Normal baseline needs real field validation.",
        "normality_scored_dataset.csv",
    ))

    deployment = build_deployment_plan(
        project_name,
        dataset_df=working_df,
        manifest=manifest,
        hardware_result=hardware,
        selected_sensors=scenario.get("selected_sensors", []),
        environment=scenario.get("environment", "Custom"),
        deployment_scale=scenario.get("deployment_scale", "Pilot: 1-5 nodes"),
        autonomy_days=7,
        communication=scenario.get("communication", "WiFi / MQTT"),
        power_source=scenario.get("power_source", "Mains + small backup"),
        enclosure_target=scenario.get("enclosure_target", "Outdoor / IP65"),
        priority=scenario.get("priority", "balanced"),
        sample_rate=scenario.get("sample_rate", config.get("sample_rate", 16000)),
    )
    artifacts["deployment_plan"] = deployment
    deployment_score = int(((deployment.get("hardware") or {}).get("hardware_fit_score", 0)))
    steps.append(_compact_step(
        "Deployment Planner",
        "PASS" if deployment.get("go_no_go") == "GO" else "WARN" if deployment.get("go_no_go") == "CONDITIONAL" else "FAIL",
        deployment_score,
        f"Board: {(deployment.get('hardware') or {}).get('recommended_board', 'Unknown')}; communication: {deployment.get('communication')}.",
        "; ".join([r.get("risk", "") for r in deployment.get("deployment_risks", []) if r.get("severity") == "high"][:2]),
        "deployment_plan.json + bom.csv",
    ))

    ei_anomaly = None
    ei_classifier = None
    if include_edge_impulse:
        ei_anomaly = build_edge_impulse_anomaly_export_snapshot(
            project_name,
            working_df,
            normal_labels=scenario.get("normal_labels", []),
            normality_result=normality,
            include_derived_scores=False,
        )
        ei_classifier = build_edge_impulse_classifier_export_snapshot(
            project_name,
            working_df,
            include_derived_scores=False,
            min_samples_per_class=10,
            max_features=12,
        )
        artifacts["edge_impulse_snapshot"] = ei_anomaly
        artifacts["edge_impulse_classifier_snapshot"] = ei_classifier
        a_score = int(((ei_anomaly.get("summary", {}) or {}).get("export_readiness_score", 0)))
        c_score = int(((ei_classifier.get("summary", {}) or {}).get("export_readiness_score", 0)))
        steps.append(_compact_step(
            "Edge Impulse Export",
            "PASS" if max(a_score, c_score) >= 70 else "WARN" if max(a_score, c_score) >= 50 else "FAIL",
            max(a_score, c_score),
            f"Anomaly export {a_score}%, classifier export {c_score}%.",
            "Edge Impulse training metrics still required after upload.",
            "edge_impulse_anomaly/classifier CSV bundles",
        ))

    report_snapshot = build_professional_report_snapshot(
        project_name,
        dataset_df=working_df,
        manifest=manifest,
        doctor=doctor2,
        reliability_v2=reliability_v2,
        trust_gate=trust_gate,
        deployment_plan=deployment,
        hardware_result=hardware,
        commercial_summary=pilot.get("commercial_summary", {}),
        real_bridge_result=None,
        package_level="Professional Pilot",
        customer_name=scenario.get("target_customer", "Customer"),
        customer_problem=scenario.get("customer_problem", ""),
    )
    artifacts["professional_report_snapshot"] = report_snapshot
    report_score = int((report_snapshot.get("readiness", {}) or {}).get("trust_score", 0))
    steps.append(_compact_step(
        "Reports 2.0",
        "PASS" if report_score >= 65 else "WARN",
        report_score,
        f"Professional report snapshot created with readiness {report_score}%.",
        "Report is pilot-preparation, not production certification.",
        "professional_report_bundle.zip",
    ))

    monetization = build_monetization_snapshot(
        project_name,
        selected_plan=selected_plan,
        dataset_df=working_df,
        trust_gate=trust_gate,
        reliability_v2=reliability_v2,
        deployment_plan=deployment,
        professional_report_snapshot=report_snapshot,
    )
    hardening = run_product_hardening_suite(
        project_name,
        dataset_df=working_df,
        doctor=doctor2,
        reliability_v2=reliability_v2,
        trust_gate=trust_gate,
        deployment_plan=deployment,
        professional_report_snapshot=report_snapshot,
        monetization_snapshot=monetization,
        has_real_data=False,
    )
    beta = build_beta_launch_snapshot(
        project_name,
        target_segment=scenario.get("target_customer", "Pilot customer"),
        dataset_df=working_df,
        trust_gate=trust_gate,
        reliability_v2=reliability_v2,
        deployment_plan=deployment,
        professional_report_snapshot=report_snapshot,
        hardening_snapshot=hardening,
        monetization_snapshot=monetization,
    )
    artifacts["monetization_snapshot"] = monetization
    artifacts["hardening_snapshot"] = hardening
    artifacts["beta_launch_snapshot"] = beta
    hardening_score = int(hardening.get("product_readiness_score", 0))
    steps.append(_compact_step(
        "Product Hardening + Beta Launch",
        "PASS" if hardening_score >= 70 else "WARN" if hardening_score >= 50 else "FAIL",
        hardening_score,
        f"Hardening score {hardening_score}%; beta readiness {beta.get('beta_readiness_score', 0)}%.",
        "; ".join(hardening.get("launch_blockers", [])[:2]) if isinstance(hardening.get("launch_blockers"), list) else "",
        "hardening_snapshot.json + beta_launch_snapshot.json",
    ))

    release = build_release_success_gate_snapshot(
        project_name=project_name,
        dataset_df=working_df,
        intended_offer=intended_offer,
        customer_type=scenario.get("target_customer", "Industrial pilot team"),
        selected_plan=selected_plan,
        trust_gate=trust_gate,
        reliability_v2=reliability_v2,
        real_bridge_result=None,
        deployment_plan=deployment,
        professional_report_snapshot=report_snapshot,
        hardening_snapshot=hardening,
        beta_launch_snapshot=beta,
        monetization_snapshot=monetization,
        normality_result=normality,
        edge_impulse_snapshot=ei_anomaly,
        edge_impulse_classifier_snapshot=ei_classifier,
    )
    artifacts["release_success_snapshot"] = release
    success_score = int((release.get("summary", {}) or {}).get("success_score", 0))
    steps.append(_compact_step(
        "Release Success Gate",
        (release.get("summary", {}) or {}).get("decision", "NO-GO"),
        success_score,
        (release.get("summary", {}) or {}).get("commercial_status", "Unknown"),
        "; ".join(release.get("blockers", [])[:2]),
        "release_success_gate_bundle.zip",
    ))

    step_summary = summarize_golden_steps(steps)
    pain_matrix = build_customer_pain_proof_matrix(
        scenario,
        dataset_df=working_df,
        doctor=doctor2,
        reliability_v2=reliability_v2,
        trust_gate=trust_gate,
        normality_result=normality,
        deployment_plan=deployment,
        edge_impulse_snapshot=ei_anomaly,
        edge_impulse_classifier_snapshot=ei_classifier,
        professional_report_snapshot=report_snapshot,
        release_success_snapshot=release,
    )
    proof_score = int(np.clip(np.mean([85 if r.get("proof_status") == "Strong proof" else 60 if r.get("proof_status") == "Partial proof" else 25 for r in pain_matrix]), 0, 100)) if pain_matrix else 0
    final_score = int(np.clip(step_summary.get("overall_score", 0) * 0.65 + proof_score * 0.35, 0, 100))
    final_decision = _decision_from_score(final_score)

    customer_wording = (
        f"EdgeTwin Studio prepared an end-to-end {scenario.get('title')} pilot package for {scenario.get('target_customer')}: "
        "dataset, audit, normal/abnormal baseline, reliability check, hardware/deployment direction, "
        "Edge Impulse export guidance, professional report and release decision. Field validation remains required before production deployment."
    )

    summary = {
        "engine": "EdgeTwin Studio V27 Golden Demo Suite",
        "created_at": _now(),
        "scenario_id": scenario_id,
        "title": scenario.get("title"),
        "project_name": project_name,
        "customer_problem": scenario.get("customer_problem"),
        "target_customer": scenario.get("target_customer"),
        "sales_angle": scenario.get("sales_angle"),
        "samples": int(len(working_df)) if isinstance(working_df, pd.DataFrame) else 0,
        "labels": sorted(working_df["Label"].astype(str).unique().tolist()) if isinstance(working_df, pd.DataFrame) and "Label" in working_df.columns else [],
        "normal_labels": scenario.get("normal_labels", []),
        "golden_demo_score": final_score,
        "proof_matrix_score": proof_score,
        "decision": final_decision,
        "customer_wording": customer_wording,
        "release_decision": (release.get("summary", {}) or {}).get("decision", "Unknown"),
        "release_success_score": success_score,
        "disclaimer": GOLDEN_DEMO_NOTE,
    }

    return {
        "summary": summary,
        "scenario": scenario,
        "config": config,
        "steps": steps,
        "step_summary": step_summary,
        "pain_proof_matrix": pain_matrix,
        "dataset": working_df,
        "fusion_dataset": fusion_df,
        "artifacts": artifacts,
    }


def generate_golden_demo_pdf_report(project_name, golden_result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Golden Demo Proof Report", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)
    summary = (golden_result or {}).get("summary", {})
    safe_pdf_cell(pdf, "Executive Decision", 8, True)
    safe_pdf_cell(pdf, f"Scenario: {summary.get('title', 'Unknown')}")
    safe_pdf_cell(pdf, f"Golden demo score: {summary.get('golden_demo_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {summary.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Release Success: {summary.get('release_success_score', 0)}% / {summary.get('release_decision', 'Unknown')}")
    safe_pdf_multicell(pdf, summary.get("customer_problem", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Safe Customer Wording", 8, True)
    safe_pdf_multicell(pdf, summary.get("customer_wording", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "End-to-End Step Results", 8, True)
    for row in (golden_result or {}).get("steps", []):
        safe_pdf_multicell(pdf, f"{row.get('step')}: {row.get('status')} / {row.get('score')}% - {row.get('evidence')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Customer Pain -> EdgeTwin Proof", 8, True)
    for row in (golden_result or {}).get("pain_proof_matrix", []):
        safe_pdf_multicell(pdf, f"Pain: {row.get('customer_pain')}")
        safe_pdf_multicell(pdf, f"Proof: {row.get('edgetwin_proof')} ({row.get('proof_status')})")
        safe_pdf_multicell(pdf, f"Value: {row.get('customer_value')}")
        pdf.ln(2)
    pdf.ln(2)
    safe_pdf_cell(pdf, "Important Note", 8, True)
    safe_pdf_multicell(pdf, GOLDEN_DEMO_NOTE)
    return safe_pdf_output(pdf)


def create_golden_demo_bundle(project_name, golden_result):
    pdf_bytes = generate_golden_demo_pdf_report(project_name, golden_result)
    zip_buf = io.BytesIO()
    result = golden_result or {}
    summary = result.get("summary", {})
    artifacts = result.get("artifacts", {}) if isinstance(result.get("artifacts"), dict) else {}
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("golden_demo_proof_report.pdf", pdf_bytes)
        zf.writestr("golden_demo_summary.json", json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
        zf.writestr("golden_demo_full_result.json", json.dumps(_json_safe({k:v for k,v in result.items() if k not in ["dataset", "fusion_dataset"]}), indent=2, ensure_ascii=False))
        if isinstance(result.get("dataset"), pd.DataFrame) and len(result.get("dataset")) > 0:
            zf.writestr("golden_demo_training_dataset.csv", result["dataset"].to_csv(index=False))
        if isinstance(result.get("fusion_dataset"), pd.DataFrame) and len(result.get("fusion_dataset")) > 0:
            zf.writestr("golden_demo_fusion_dataset.csv", result["fusion_dataset"].to_csv(index=False))
        if result.get("steps"):
            zf.writestr("end_to_end_steps.csv", pd.DataFrame(result.get("steps", [])).to_csv(index=False))
        if result.get("pain_proof_matrix"):
            zf.writestr("customer_pain_proof_matrix.csv", pd.DataFrame(result.get("pain_proof_matrix", [])).to_csv(index=False))
        # Include compact copies of the most valuable generated evidence.
        for key in ["reliability_v2", "trust_gate", "deployment_plan", "professional_report_snapshot", "hardening_snapshot", "beta_launch_snapshot", "monetization_snapshot", "release_success_snapshot"]:
            if key in artifacts:
                zf.writestr(f"evidence_{key}.json", json.dumps(_json_safe(artifacts.get(key)), indent=2, ensure_ascii=False))
        if isinstance(artifacts.get("normality_result"), dict):
            normality = artifacts.get("normality_result")
            scored = normality.get("scored_dataset")
            zf.writestr("evidence_normality_summary.json", json.dumps(_json_safe({k:v for k,v in normality.items() if k != "scored_dataset"}), indent=2, ensure_ascii=False))
            if isinstance(scored, pd.DataFrame) and len(scored) > 0:
                zf.writestr("normality_scored_dataset.csv", scored.to_csv(index=False))
        for key, prefix in [("edge_impulse_snapshot", "edge_impulse_anomaly"), ("edge_impulse_classifier_snapshot", "edge_impulse_classifier")]:
            snap = artifacts.get(key)
            if isinstance(snap, dict):
                zf.writestr(f"evidence_{prefix}_summary.json", json.dumps(_json_safe(snap.get("summary", {})), indent=2, ensure_ascii=False))
                files = snap.get("files", {})
                if isinstance(files, dict):
                    for fkey, filename in [("normal_training_csv", f"{prefix}_normal_training.csv"), ("evaluation_csv", f"{prefix}_evaluation.csv"), ("train_csv", f"{prefix}_train.csv"), ("test_csv", f"{prefix}_test.csv"), ("full_csv", f"{prefix}_full.csv")]:
                        obj = files.get(fkey)
                        if isinstance(obj, pd.DataFrame) and len(obj) > 0:
                            zf.writestr(filename, obj.to_csv(index=False))
        zf.writestr("README.txt", f"""EdgeTwin Studio V27 Golden Demo Bundle

Scenario: {summary.get('title', 'Unknown')}
Decision: {summary.get('decision', 'Unknown')}
Golden demo score: {summary.get('golden_demo_score', 0)}%
Release success score: {summary.get('release_success_score', 0)}%

This bundle proves the end-to-end pilot-preparation flow. It remains a pilot-preparation package, not production certification.

{GOLDEN_DEMO_NOTE}
""")
    return zip_buf.getvalue()


# ============================================================
# V28 CLOSED BETA LAUNCH KIT
# ============================================================

CLOSED_BETA_NOTE = (
    "Closed beta is intended for controlled validation with a small number of trusted testers. "
    "It is not a production certification, safety certification or guaranteed model performance statement. "
    "Field validation, customer review and deployment responsibility remain required before production use."
)

CLOSED_BETA_SEGMENTS = {
    "Predictive Maintenance Team": {
        "who": "Maintenance managers, machine builders, industrial IoT teams and reliability engineers.",
        "pain": "They want to start vibration/acoustic monitoring but lack clean labels, failure examples and a hardware path.",
        "promise": "Prepare a pilot package for machine-health Edge AI: dataset, normal baseline, reliability check, hardware/deployment direction and Edge Impulse exports.",
        "best_demo": "Predictive Maintenance Golden Demo",
    },
    "Security / Tamper Integrator": {
        "who": "Construction security, container monitoring, asset protection and intrusion/tamper detection teams.",
        "pain": "They need to detect drilling, grinding, impact and handling without spending weeks building first datasets.",
        "promise": "Prepare an acoustic/vibration security pilot package with event labels, normality baseline, Edge Impulse classifier/anomaly exports and report evidence.",
        "best_demo": "Acoustic Tamper Golden Demo",
    },
    "Remote Asset / Forestry Pilot": {
        "who": "Forestry, agriculture, machinery yards, remote infrastructure and off-grid asset operators.",
        "pain": "They need fast remote monitoring pilots but are unsure about sensors, power, communications and normal/abnormal conditions.",
        "promise": "Prepare a remote asset pilot package with sensor-fusion data, deployment plan, field validation steps and customer-ready report output.",
        "best_demo": "Remote Forestry / Asset Golden Demo",
    },
    "Edge AI Consultant / Integrator": {
        "who": "Consultants and system integrators who need faster first-pass pilot packages for clients.",
        "pain": "They lose time turning vague customer problems into datasets, pilot plans, reports and hardware options.",
        "promise": "Use EdgeTwin as a pre-consultancy automation layer that creates a structured pilot package and evidence bundle.",
        "best_demo": "Any Golden Demo matching the client vertical",
    },
}


def get_closed_beta_segments():
    return list(CLOSED_BETA_SEGMENTS.keys())


def get_closed_beta_segment(segment_name):
    return CLOSED_BETA_SEGMENTS.get(segment_name, CLOSED_BETA_SEGMENTS["Predictive Maintenance Team"])


def _extract_score_from_snapshot(snapshot, keys):
    if not isinstance(snapshot, dict):
        return 0
    for key in keys:
        cur = snapshot
        ok = True
        for part in key.split('.'):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            try:
                return int(float(cur))
            except Exception:
                pass
    return 0


def _closed_beta_decision(score, blockers):
    if blockers:
        return "NOT READY"
    if score >= 82:
        return "READY FOR CONTROLLED BETA"
    if score >= 65:
        return "LIMITED BETA ONLY"
    return "INTERNAL DEMO ONLY"


def _beta_package_recommendation(score, selected_offer):
    offer = selected_offer or "Professional Pilot Bundle"
    if score >= 85:
        return {
            "recommended_package": "Paid Pilot Candidate",
            "pricing_range": "€199 - €799 depending on real-data usage and manual review",
            "positioning": "Use as a controlled paid pilot package for a small number of trusted technical customers.",
            "minimum_plan": "Professional / Real-Data Pilot",
        }
    if score >= 70:
        return {
            "recommended_package": "Private Beta / Discounted Pilot",
            "pricing_range": "€49 - €249 or free for strong feedback/testimonial commitment",
            "positioning": "Use with clear beta wording and limited customer promises.",
            "minimum_plan": "Starter / Professional",
        }
    return {
        "recommended_package": "Internal Demo / Founder Test",
        "pricing_range": "Do not charge yet; use internally or with friendly testers only",
        "positioning": "Improve blockers before presenting this as a paid product.",
        "minimum_plan": "Founder Test Mode",
    }


def build_closed_beta_launch_kit(
    project_name,
    target_segment="Predictive Maintenance Team",
    selected_offer="Professional Pilot Bundle",
    beta_goal="Validate whether EdgeTwin Studio helps a real customer move from sensor idea to pilot-ready package faster.",
    max_beta_users=5,
    dataset_df=None,
    golden_demo_result=None,
    release_success_snapshot=None,
    hardening_snapshot=None,
    monetization_snapshot=None,
    professional_report_snapshot=None,
):
    segment = get_closed_beta_segment(target_segment)
    df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    rows = int(len(df)) if isinstance(df, pd.DataFrame) else 0
    labels = []
    if isinstance(df, pd.DataFrame) and "Label" in df.columns and len(df) > 0:
        labels = sorted(df["Label"].astype(str).unique().tolist())

    golden_score = _extract_score_from_snapshot(golden_demo_result, ["summary.golden_demo_score", "golden_demo_score"])
    release_score = _extract_score_from_snapshot(release_success_snapshot, ["summary.success_score", "summary.release_success_score", "release_success_score"])
    hardening_score = _extract_score_from_snapshot(hardening_snapshot, ["product_readiness_score", "summary.product_readiness_score"])
    report_score = _extract_score_from_snapshot(professional_report_snapshot, ["report_score", "summary.report_score"])

    evidence = []
    blockers = []
    warnings = []

    if rows >= 200 and len(labels) >= 2:
        evidence.append({"area": "Dataset", "status": "OK", "detail": f"{rows} rows across {len(labels)} labels."})
    elif rows > 0:
        warnings.append("Dataset exists but is small or has limited label coverage.")
        evidence.append({"area": "Dataset", "status": "WARN", "detail": f"{rows} rows across {len(labels)} labels."})
    else:
        blockers.append("No dataset available for beta demonstration.")
        evidence.append({"area": "Dataset", "status": "BLOCKED", "detail": "No dataset loaded."})

    if golden_score >= 70:
        evidence.append({"area": "Golden Demo", "status": "OK", "detail": f"Golden demo score {golden_score}%."})
    elif golden_score > 0:
        warnings.append("Golden Demo score is not strong enough for a confident paid beta.")
        evidence.append({"area": "Golden Demo", "status": "WARN", "detail": f"Golden demo score {golden_score}%."})
    else:
        warnings.append("Golden Demo has not been run yet. Run it before approaching serious beta customers.")
        evidence.append({"area": "Golden Demo", "status": "MISSING", "detail": "No Golden Demo result detected."})

    if release_score >= 70:
        evidence.append({"area": "Release Gate", "status": "OK", "detail": f"Release success score {release_score}%."})
    elif release_score > 0:
        warnings.append("Release Success Gate is conditional; keep beta scope narrow.")
        evidence.append({"area": "Release Gate", "status": "WARN", "detail": f"Release success score {release_score}%."})
    else:
        warnings.append("Release Success Gate not detected. Use Success Gate before selling access.")
        evidence.append({"area": "Release Gate", "status": "MISSING", "detail": "No release snapshot detected."})

    if hardening_score >= 70:
        evidence.append({"area": "Hardening", "status": "OK", "detail": f"Product hardening score {hardening_score}%."})
    elif hardening_score > 0:
        warnings.append("Hardening score is not ideal; avoid broad beta launch.")
        evidence.append({"area": "Hardening", "status": "WARN", "detail": f"Product hardening score {hardening_score}%."})
    else:
        warnings.append("Product Hardening was not detected. Use V24.1 before external beta.")
        evidence.append({"area": "Hardening", "status": "MISSING", "detail": "No hardening snapshot detected."})

    base_scores = [s for s in [golden_score, release_score, hardening_score, report_score] if s > 0]
    evidence_score = int(np.mean(base_scores)) if base_scores else 35
    dataset_score = 90 if rows >= 500 and len(labels) >= 3 else 75 if rows >= 200 and len(labels) >= 2 else 45 if rows > 0 else 0
    beta_score = int(np.clip(evidence_score * 0.65 + dataset_score * 0.25 + min(10, int(max_beta_users)) * 1.0, 0, 100))
    if blockers:
        beta_score = min(beta_score, 55)
    decision = _closed_beta_decision(beta_score, blockers)
    package = _beta_package_recommendation(beta_score, selected_offer)

    onboarding_checklist = [
        {"step": "Use one clear vertical", "owner": "Founder", "status": "Required", "detail": f"Start with {target_segment}; do not pitch every industry at once."},
        {"step": "Run Golden Demo before call", "owner": "Founder", "status": "Required", "detail": "Use the matching Golden Demo and keep the output bundle ready."},
        {"step": "Explain beta scope", "owner": "Founder", "status": "Required", "detail": "Pilot-preparation tool, not production certification."},
        {"step": "Collect customer use-case", "owner": "Customer", "status": "Required", "detail": "Ask sensors, labels/classes, target hardware and available real data."},
        {"step": "Generate pilot bundle", "owner": "EdgeTwin", "status": "Required", "detail": "Dataset, reliability, normality, hardware plan, EI export and report."},
        {"step": "Gather feedback", "owner": "Customer", "status": "Required", "detail": "Use the included beta feedback form within 7 days."},
        {"step": "Decide next action", "owner": "Founder + Customer", "status": "Required", "detail": "No-go, second beta, paid pilot, or real-data upload pilot."},
    ]

    feedback_questions = [
        {"category": "Problem fit", "question": "Does EdgeTwin clearly understand your sensor/use-case problem?", "type": "1-5 score"},
        {"category": "Output value", "question": "Which output was most valuable: dataset, reliability, hardware plan, Edge Impulse export or report?", "type": "text"},
        {"category": "Trust", "question": "Did the risk/reliability wording feel honest enough for a technical pilot?", "type": "1-5 score"},
        {"category": "Usability", "question": "Could a non-AI specialist understand what to do next?", "type": "1-5 score"},
        {"category": "Payment", "question": "Would you pay for this pilot bundle? What price range feels fair?", "type": "text"},
        {"category": "Missing", "question": "What output would you need before using this in a real pilot?", "type": "text"},
        {"category": "Real data", "question": "Can you provide real WAV/CSV samples for a stronger Synthetic-to-Real Bridge?", "type": "yes/no + notes"},
    ]

    success_metrics = [
        {"metric": "Time-to-understanding", "target": "Customer understands product value within 10 minutes", "why_it_matters": "Shows accessibility."},
        {"metric": "Bundle usefulness", "target": "Customer rates report/bundle 4/5 or higher", "why_it_matters": "Shows output value."},
        {"metric": "Real-data interest", "target": "At least 30% of beta users offer real WAV/CSV data", "why_it_matters": "Validates V21 bridge."},
        {"metric": "Payment signal", "target": "At least 1 beta user accepts paid pilot or LOI", "why_it_matters": "Validates pricing."},
        {"metric": "Support load", "target": "Founder spends less than 45 minutes per beta user", "why_it_matters": "Protects solo-founder execution."},
    ]

    week_plan = [
        {"week": "Week 0", "focus": "Prepare", "actions": "Run Golden Demo, Success Gate and Closed Beta Kit; pick 3-5 trusted testers."},
        {"week": "Week 1", "focus": "Demo", "actions": "Show one vertical-specific demo; collect use-case details; generate first bundles."},
        {"week": "Week 2", "focus": "Feedback", "actions": "Review feedback form, pain points, missing outputs and pricing reaction."},
        {"week": "Week 3", "focus": "Real data", "actions": "Ask for WAV/CSV samples from the strongest tester; run Real Bridge."},
        {"week": "Week 4", "focus": "Decision", "actions": "Decide: improve, paid pilot, LOI, or stop segment."},
    ]

    invite_email = f"""Subject: Private beta invitation: EdgeTwin Studio pilot package for {target_segment}

Hi [Name],

I am preparing a small closed beta for EdgeTwin Studio, a tool that helps turn vibration/acoustic/sensor use-cases into an Edge AI pilot package: dataset, normal/abnormal baseline, reliability check, hardware/deployment direction, Edge Impulse export guidance and a professional report.

Why I think it may fit your situation:
{segment.get('pain')}

What the beta would test:
{beta_goal}

Important: this is pilot-preparation software, not production certification. The goal is to reduce trial-and-error and create a clearer route toward a real field pilot.

Would you be open to a short demo and feedback session?

Best,
[Your name]
"""

    demo_script = [
        {"minute": "0-2", "section": "Problem", "talk_track": f"Many {target_segment} customers struggle because {segment.get('pain')}"},
        {"minute": "2-5", "section": "Golden Demo", "talk_track": f"Show {segment.get('best_demo')} and the customer pain proof matrix."},
        {"minute": "5-8", "section": "Trust", "talk_track": "Show Reliability Engine, Normality Engine, and the clear pilot-ready vs production-ready distinction."},
        {"minute": "8-11", "section": "Outputs", "talk_track": "Show ZIP/PDF bundle, Edge Impulse export, deployment plan and report."},
        {"minute": "11-15", "section": "Feedback", "talk_track": "Ask what would make this valuable enough for a paid pilot or real-data upload."},
    ]

    safe_claims = [
        "EdgeTwin Studio helps prepare Edge AI sensor pilots faster.",
        "It creates datasets, audits, reliability estimates, hardware direction and reports for pilot planning.",
        "It helps identify normal vs abnormal patterns and weak labels/classes.",
        "It can export structured datasets for Edge Impulse anomaly/classifier workflows.",
        "Field validation remains required before production deployment.",
    ]
    claims_to_avoid = [
        "Do not claim production-ready accuracy without real field validation.",
        "Do not claim certification or safety approval.",
        "Do not claim synthetic data fully replaces real data.",
        "Do not promise zero false alarms.",
        "Do not sell broad public SaaS until closed beta evidence is collected.",
    ]

    summary = {
        "engine": "EdgeTwin Studio V28 Closed Beta Launch Kit",
        "created_at": _now(),
        "project_name": project_name,
        "target_segment": target_segment,
        "selected_offer": selected_offer,
        "beta_goal": beta_goal,
        "max_beta_users": int(max_beta_users),
        "beta_readiness_score": beta_score,
        "decision": decision,
        "rows": rows,
        "labels": labels,
        "golden_demo_score": golden_score,
        "release_success_score": release_score,
        "hardening_score": hardening_score,
        "package_recommendation": package,
        "segment_promise": segment.get("promise"),
        "closed_beta_note": CLOSED_BETA_NOTE,
    }

    return {
        "summary": summary,
        "segment": segment,
        "evidence": evidence,
        "blockers": blockers,
        "warnings": warnings,
        "onboarding_checklist": onboarding_checklist,
        "feedback_questions": feedback_questions,
        "success_metrics": success_metrics,
        "week_plan": week_plan,
        "invite_email": invite_email,
        "demo_script": demo_script,
        "safe_claims": safe_claims,
        "claims_to_avoid": claims_to_avoid,
        "dataset_snapshot": df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(),
    }


def generate_closed_beta_pdf_report(project_name, beta_kit):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Closed Beta Launch Kit", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)
    summary = (beta_kit or {}).get("summary", {})
    safe_pdf_cell(pdf, "Beta Decision", 8, True)
    safe_pdf_cell(pdf, f"Target segment: {summary.get('target_segment', 'Unknown')}")
    safe_pdf_cell(pdf, f"Readiness score: {summary.get('beta_readiness_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {summary.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Recommended package: {(summary.get('package_recommendation') or {}).get('recommended_package', 'Unknown')}")
    safe_pdf_multicell(pdf, f"Positioning: {(summary.get('package_recommendation') or {}).get('positioning', '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Goal", 8, True)
    safe_pdf_multicell(pdf, summary.get("beta_goal", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Evidence", 8, True)
    for row in (beta_kit or {}).get("evidence", []):
        safe_pdf_multicell(pdf, f"[{row.get('status')}] {row.get('area')}: {row.get('detail')}")
    blockers = (beta_kit or {}).get("blockers", [])
    warnings = (beta_kit or {}).get("warnings", [])
    if blockers:
        pdf.ln(4)
        safe_pdf_cell(pdf, "Blockers", 8, True)
        for b in blockers:
            safe_pdf_multicell(pdf, f"- {b}")
    if warnings:
        pdf.ln(4)
        safe_pdf_cell(pdf, "Warnings", 8, True)
        for w in warnings:
            safe_pdf_multicell(pdf, f"- {w}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Demo Script", 8, True)
    for row in (beta_kit or {}).get("demo_script", []):
        safe_pdf_multicell(pdf, f"{row.get('minute')} - {row.get('section')}: {row.get('talk_track')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Success Metrics", 8, True)
    for row in (beta_kit or {}).get("success_metrics", []):
        safe_pdf_multicell(pdf, f"{row.get('metric')}: {row.get('target')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Safe Claims", 8, True)
    for c in (beta_kit or {}).get("safe_claims", []):
        safe_pdf_multicell(pdf, f"- {c}")
    pdf.ln(2)
    safe_pdf_cell(pdf, "Claims to Avoid", 8, True)
    for c in (beta_kit or {}).get("claims_to_avoid", []):
        safe_pdf_multicell(pdf, f"- {c}")
    pdf.ln(2)
    safe_pdf_cell(pdf, "Important Note", 8, True)
    safe_pdf_multicell(pdf, CLOSED_BETA_NOTE)
    return safe_pdf_output(pdf)


def create_closed_beta_launch_bundle(project_name, beta_kit):
    beta_kit = beta_kit or {}
    pdf_bytes = generate_closed_beta_pdf_report(project_name, beta_kit)
    summary = beta_kit.get("summary", {})
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("closed_beta_launch_report.pdf", pdf_bytes)
        zf.writestr("closed_beta_summary.json", json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
        zf.writestr("closed_beta_full_plan.json", json.dumps(_json_safe({k:v for k,v in beta_kit.items() if k != "dataset_snapshot"}), indent=2, ensure_ascii=False))
        zf.writestr("invite_email.md", beta_kit.get("invite_email", ""))
        for key, filename in [
            ("evidence", "beta_evidence.csv"),
            ("onboarding_checklist", "onboarding_checklist.csv"),
            ("feedback_questions", "feedback_questions.csv"),
            ("success_metrics", "success_metrics.csv"),
            ("week_plan", "four_week_beta_plan.csv"),
            ("demo_script", "demo_script.csv"),
        ]:
            rows = beta_kit.get(key, [])
            if isinstance(rows, list) and rows:
                zf.writestr(filename, pd.DataFrame(rows).to_csv(index=False))
        if isinstance(beta_kit.get("dataset_snapshot"), pd.DataFrame) and len(beta_kit.get("dataset_snapshot")) > 0:
            zf.writestr("dataset_snapshot.csv", beta_kit["dataset_snapshot"].to_csv(index=False))
        zf.writestr("safe_claims.txt", "\n".join(beta_kit.get("safe_claims", [])))
        zf.writestr("claims_to_avoid.txt", "\n".join(beta_kit.get("claims_to_avoid", [])))
        zf.writestr("README.txt", f"""EdgeTwin Studio V28 Closed Beta Launch Kit

Decision: {summary.get('decision', 'Unknown')}
Beta readiness score: {summary.get('beta_readiness_score', 0)}%
Target segment: {summary.get('target_segment', 'Unknown')}
Recommended package: {(summary.get('package_recommendation') or {}).get('recommended_package', 'Unknown')}

Use this bundle to run a small controlled beta with trusted testers. Keep scope narrow, collect feedback, and avoid production claims.

{CLOSED_BETA_NOTE}
""")
    return zip_buf.getvalue()


# ============================================================
# V29 PAID EXPORT & LICENSE GATE
# ============================================================

PAID_LICENSE_VERSION = "EdgeTwin Studio V29 - Paid Export & License Gate"
PAID_LICENSE_DISCLAIMER = (
    "Paid Export & License Gate controls commercial packaging, entitlement metadata and safe-use delivery notes. "
    "It does not process payments directly yet. Connect Stripe or another payment provider only after beta validation and security hardening."
)

PAID_PACKAGE_EXPORTS = {
    "Free Demo Preview": ["training_csv_preview"],
    "Starter Pilot Bundle": ["training_csv", "fusion_bundle", "basic_report"],
    "Professional Pilot Bundle": ["training_csv", "fusion_bundle", "enterprise_bundle", "optimizer_bundle", "trust_bundle", "professional_report_bundle"],
    "Real-Data Pilot Bundle": ["training_csv", "fusion_bundle", "enterprise_bundle", "optimizer_bundle", "trust_bundle", "real_bridge_bundle", "reliability_v2_bundle", "deployment_bundle", "professional_report_bundle"],
    "Enterprise Review Candidate": ["training_csv", "fusion_bundle", "enterprise_bundle", "optimizer_bundle", "trust_bundle", "real_bridge_bundle", "reliability_v2_bundle", "deployment_bundle", "professional_report_bundle", "api_access", "on_premise"],
}

PAID_PRICE_POSITIONS = {
    "Free Demo Preview": "Free / lead magnet",
    "Starter Pilot Bundle": "EUR 49 - 99",
    "Professional Pilot Bundle": "EUR 199 - 499",
    "Real-Data Pilot Bundle": "EUR 799 - 1,500",
    "Enterprise Review Candidate": "EUR 2,500 - 10,000+ / custom",
}


def get_paid_license_plans():
    return get_pricing_plans()


def _stable_license_fingerprint(project_name, customer_email, selected_plan, requested_package):
    raw = f"{project_name}|{customer_email}|{selected_plan}|{requested_package}|{_now()[:10]}"
    # Not a cryptographic license system; good enough for beta/manual delivery receipts.
    import hashlib
    digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest().upper()
    return f"ETL-{digest[:4]}-{digest[4:8]}-{digest[8:12]}"


def _plan_rank_for_license(plan_name):
    order = ["Free Demo", "Starter Pilot", "Professional Pilot", "Real-Data Pilot", "Enterprise", "Founder Test Mode"]
    if plan_name not in order:
        return 0
    return order.index(plan_name)


def _package_min_plan(package):
    mapping = {
        "Free Demo Preview": "Free Demo",
        "Starter Pilot Bundle": "Starter Pilot",
        "Professional Pilot Bundle": "Professional Pilot",
        "Real-Data Pilot Bundle": "Real-Data Pilot",
        "Enterprise Review Candidate": "Enterprise",
    }
    return mapping.get(package, "Professional Pilot")


def build_paid_export_license_gate(
    project_name,
    customer_name="Beta Customer",
    customer_email="customer@example.com",
    requested_package="Professional Pilot Bundle",
    selected_plan="Founder Test Mode",
    checkout_mode="Manual invoice",
    watermark_free_exports=True,
    dataset_df=None,
    monetization_snapshot=None,
    release_success_snapshot=None,
    professional_report_snapshot=None,
    closed_beta_kit=None,
):
    requested_exports = PAID_PACKAGE_EXPORTS.get(requested_package, PAID_PACKAGE_EXPORTS["Professional Pilot Bundle"])
    minimum_plan = _package_min_plan(requested_package)
    plan_rank = _plan_rank_for_license(selected_plan)
    min_rank = _plan_rank_for_license(minimum_plan)
    plan_ok = selected_plan == "Founder Test Mode" or plan_rank >= min_rank

    access_rows = []
    unlocked = []
    locked = []
    for export_key in requested_exports:
        access = evaluate_export_access(selected_plan, export_key)
        # Founder Test Mode can unlock for testing, but it must stay marked internal.
        if selected_plan == "Founder Test Mode":
            access["allowed"] = True
            access["reason"] = "Founder/internal test access. Do not treat as paid customer entitlement."
        if not plan_ok and selected_plan != "Founder Test Mode":
            access["allowed"] = False
            access["reason"] = f"Requested package requires at least {minimum_plan}."
        access_rows.append(access)
        if access.get("allowed"):
            unlocked.append(access)
        else:
            locked.append(access)

    rows = int(len(dataset_df)) if isinstance(dataset_df, pd.DataFrame) else 0
    labels = int(dataset_df["Label"].nunique()) if isinstance(dataset_df, pd.DataFrame) and "Label" in dataset_df.columns else 0
    has_dataset = rows > 0
    has_report = bool(professional_report_snapshot)

    release_score = 0
    release_decision = "Unknown"
    if isinstance(release_success_snapshot, dict) and release_success_snapshot:
        rel_sum = release_success_snapshot.get("summary", release_success_snapshot)
        release_score = int(rel_sum.get("release_success_score", rel_sum.get("score", 0)) or 0)
        release_decision = rel_sum.get("decision", rel_sum.get("status", "Unknown"))

    beta_score = 0
    if isinstance(closed_beta_kit, dict) and closed_beta_kit:
        beta_score = int((closed_beta_kit.get("summary") or {}).get("beta_readiness_score", 0) or 0)

    base_score = 35
    if has_dataset:
        base_score += 15
    if labels >= 2:
        base_score += 10
    if has_report:
        base_score += 10
    if release_score >= 70:
        base_score += 12
    elif release_score >= 50:
        base_score += 6
    if beta_score >= 70:
        base_score += 8
    if plan_ok:
        base_score += 10
    if locked:
        base_score -= min(18, len(locked) * 4)
    commercial_readiness = int(np.clip(base_score, 0, 100))

    must_fix = []
    if not has_dataset:
        must_fix.append("Generate or upload a usable dataset before selling a paid export.")
    if requested_package in ["Professional Pilot Bundle", "Real-Data Pilot Bundle", "Enterprise Review Candidate"] and not has_report:
        must_fix.append("Generate Reports 2.0 before selling this package level.")
    if requested_package in ["Real-Data Pilot Bundle", "Enterprise Review Candidate"]:
        if not any(x.get("export_key") == "real_bridge_bundle" and x.get("allowed") for x in unlocked):
            must_fix.append("Real-Data package needs Synthetic-to-Real evidence or a lower package level.")
    if locked and selected_plan != "Founder Test Mode":
        must_fix.append("Selected plan does not unlock all requested package exports.")
    if selected_plan == "Founder Test Mode" and checkout_mode != "Founder internal":
        must_fix.append("Founder Test Mode is internal only; switch to a customer plan before real paid delivery.")

    if commercial_readiness >= 82 and not must_fix and selected_plan != "Founder Test Mode":
        decision = "PAID EXPORT READY"
        license_status = "ACTIVE_FOR_DELIVERY"
        customer_message = "This package can be delivered as a paid pilot-preparation export with safe-use terms."
    elif commercial_readiness >= 62 and has_dataset:
        decision = "MANUAL REVIEW BEFORE PAID EXPORT"
        license_status = "REVIEW_REQUIRED"
        customer_message = "This package is close, but should be reviewed before charging or delivered as a controlled beta."
    else:
        decision = "DO NOT SELL YET"
        license_status = "INTERNAL_ONLY"
        customer_message = "Keep this as an internal/demo package until blockers are fixed."

    license_key = _stable_license_fingerprint(project_name, customer_email, selected_plan, requested_package)
    watermark_policy = "Watermarked previews only" if selected_plan == "Free Demo" and watermark_free_exports else "No watermark for unlocked paid exports"
    if selected_plan == "Founder Test Mode":
        watermark_policy = "Internal founder/testing outputs - not a customer license"

    entitlement_manifest = {
        "license_key": license_key,
        "license_status": license_status,
        "project_name": project_name,
        "customer_name": customer_name,
        "customer_email": customer_email,
        "selected_plan": selected_plan,
        "requested_package": requested_package,
        "minimum_required_plan": minimum_plan,
        "checkout_mode": checkout_mode,
        "created_at": _now(),
        "unlocked_export_keys": [x.get("export_key") for x in unlocked],
        "locked_export_keys": [x.get("export_key") for x in locked],
        "watermark_policy": watermark_policy,
        "price_position": PAID_PRICE_POSITIONS.get(requested_package, "Custom"),
        "delivery_scope": "Pilot preparation and decision support. Not production certification.",
    }

    safe_use_terms = [
        "EdgeTwin Studio outputs are for pilot preparation, dataset planning and decision support.",
        "Production deployment requires real-world field validation and customer-side acceptance testing.",
        "Synthetic data and synthetic-to-real variants do not replace production certification.",
        "Do not use this package as a safety-critical guarantee or automatic compliance certificate.",
        "Customer data ownership and retention terms must be agreed before using real customer uploads.",
    ]

    receipt_preview = f"""EdgeTwin Studio Delivery Note / License Receipt

Customer: {customer_name} <{customer_email}>
Project: {project_name}
Package: {requested_package}
Plan: {selected_plan}
License key: {license_key}
Status: {license_status}
Price position: {PAID_PRICE_POSITIONS.get(requested_package, 'Custom')}
Unlocked exports: {', '.join(entitlement_manifest['unlocked_export_keys']) or 'None'}
Locked exports: {', '.join(entitlement_manifest['locked_export_keys']) or 'None'}

Scope: Pilot preparation, dataset audit, reliability guidance and export support.
Not included: production certification, guaranteed model accuracy, regulatory approval or safety-critical validation.
"""

    return _json_safe({
        "engine": PAID_LICENSE_VERSION,
        "disclaimer": PAID_LICENSE_DISCLAIMER,
        "summary": {
            "commercial_readiness_score": commercial_readiness,
            "decision": decision,
            "license_status": license_status,
            "customer_message": customer_message,
            "price_position": PAID_PRICE_POSITIONS.get(requested_package, "Custom"),
            "requested_package": requested_package,
            "selected_plan": selected_plan,
            "minimum_required_plan": minimum_plan,
            "release_decision": release_decision,
        },
        "entitlement_manifest": entitlement_manifest,
        "access_matrix": access_rows,
        "unlocked_exports": unlocked,
        "locked_exports": locked,
        "must_fix_before_charging": must_fix,
        "safe_use_terms": safe_use_terms,
        "receipt_preview": receipt_preview,
        "dataset_summary": {"rows": rows, "labels": labels},
    })


def generate_paid_license_pdf(project_name, license_snapshot):
    snapshot = license_snapshot or {}
    summary = snapshot.get("summary", {}) or {}
    manifest = snapshot.get("entitlement_manifest", {}) or {}
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt=clean_pdf_text("EdgeTwin Studio Paid Export License Gate"), ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("V29 Commercial Delivery & Entitlement Manifest"), ln=True, align="C")
    pdf.ln(8)
    safe_pdf_cell(pdf, "Decision", 8, True)
    safe_pdf_cell(pdf, f"Commercial readiness: {summary.get('commercial_readiness_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {summary.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"License status: {summary.get('license_status', 'Unknown')}")
    safe_pdf_multicell(pdf, summary.get("customer_message", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Entitlement", 8, True)
    for key in ["license_key", "customer_name", "customer_email", "project_name", "requested_package", "selected_plan", "minimum_required_plan", "checkout_mode", "price_position", "watermark_policy"]:
        safe_pdf_multicell(pdf, f"{key.replace('_',' ').title()}: {manifest.get(key, '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Unlocked Exports", 8, True)
    for row in snapshot.get("unlocked_exports", []):
        safe_pdf_multicell(pdf, f"- {row.get('label', row.get('export_key'))}: {row.get('commercial_value', '')}")
    pdf.ln(2)
    safe_pdf_cell(pdf, "Locked Exports", 8, True)
    locked = snapshot.get("locked_exports", [])
    if locked:
        for row in locked:
            safe_pdf_multicell(pdf, f"- {row.get('label', row.get('export_key'))}: {row.get('reason', '')}")
    else:
        safe_pdf_multicell(pdf, "No locked exports for this package/plan.")
    must_fix = snapshot.get("must_fix_before_charging", [])
    if must_fix:
        pdf.ln(4)
        safe_pdf_cell(pdf, "Must Fix Before Charging", 8, True)
        for item in must_fix:
            safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Safe-use Terms", 8, True)
    for item in snapshot.get("safe_use_terms", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Important Disclaimer", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", PAID_LICENSE_DISCLAIMER))
    return safe_pdf_output(pdf)


def create_paid_export_license_bundle(project_name, license_snapshot, dataset_df=None):
    snapshot = license_snapshot or {}
    pdf_bytes = generate_paid_license_pdf(project_name, snapshot)
    zip_buf = io.BytesIO()
    manifest = snapshot.get("entitlement_manifest", {}) or {}
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("paid_export_license_gate_v29.pdf", pdf_bytes)
        zf.writestr("entitlement_manifest.json", json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False))
        zf.writestr("license_gate_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr("customer_receipt_preview.txt", snapshot.get("receipt_preview", ""))
        if snapshot.get("access_matrix"):
            zf.writestr("export_access_matrix.csv", pd.DataFrame(snapshot.get("access_matrix", [])).to_csv(index=False))
        if snapshot.get("unlocked_exports"):
            zf.writestr("unlocked_exports.csv", pd.DataFrame(snapshot.get("unlocked_exports", [])).to_csv(index=False))
        if snapshot.get("locked_exports"):
            zf.writestr("locked_exports.csv", pd.DataFrame(snapshot.get("locked_exports", [])).to_csv(index=False))
        if snapshot.get("must_fix_before_charging"):
            zf.writestr("must_fix_before_charging.txt", "\n".join(snapshot.get("must_fix_before_charging", [])))
        zf.writestr("safe_use_terms.txt", "\n".join(snapshot.get("safe_use_terms", [])))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            # Include only a snapshot; paid production exports remain handled by the product-specific bundle buttons.
            out_df = dataset_df.head(2000).copy()
            if manifest.get("watermark_policy") == "Watermarked previews only":
                out_df["EdgeTwin_Watermark"] = "PREVIEW_ONLY_NOT_FOR_PRODUCTION"
            zf.writestr("licensed_dataset_snapshot.csv", out_df.to_csv(index=False))
        zf.writestr("README.txt", f"""EdgeTwin Studio V29 Paid Export License Bundle

Project: {project_name}
Package: {manifest.get('requested_package', 'Unknown')}
Plan: {manifest.get('selected_plan', 'Unknown')}
License key: {manifest.get('license_key', 'Unknown')}
Status: {manifest.get('license_status', 'Unknown')}

This bundle documents what the customer is entitled to receive and under which safe-use terms.
It is not a payment processor and not a production certification.

{snapshot.get('disclaimer', PAID_LICENSE_DISCLAIMER)}
""")
    return zip_buf.getvalue()


# ============================================================
# V30 REAL FIELD VALIDATION PACK
# ============================================================

FIELD_VALIDATION_VERSION = "EdgeTwin Studio V30 - Real Field Validation Pack"
FIELD_VALIDATION_DISCLAIMER = (
    "Real Field Validation Pack is a pilot evidence and validation planning layer. It is not production certification, "
    "not a guaranteed model-accuracy statement and not a safety-critical approval. Production deployment requires "
    "customer-side field validation, acceptance testing and documented deployment controls."
)

FIELD_VALIDATION_SCENARIOS = {
    "Predictive Maintenance Field Pilot": {
        "normal_keywords": ["normal", "healthy", "baseline", "idle"],
        "event_keywords": ["wear", "failure", "fault", "critical", "unbalance", "misalignment", "bearing"],
        "minimum_files": 20,
        "minimum_days": 7,
        "recommended_real_per_label": 30,
        "customer_pain": "The customer needs proof that machine normal behavior and failure-risk behavior can be separated in real field conditions.",
    },
    "Acoustic / Vibration Tamper Field Pilot": {
        "normal_keywords": ["normal", "background", "baseline", "quiet"],
        "event_keywords": ["tamper", "impact", "drilling", "grinding", "cutting", "tool", "critical"],
        "minimum_files": 20,
        "minimum_days": 5,
        "recommended_real_per_label": 25,
        "customer_pain": "The customer needs proof that normal site noise can be separated from impact/tool/tamper events.",
    },
    "Remote Asset / Forestry Field Pilot": {
        "normal_keywords": ["normal", "forest", "baseline", "calm", "background"],
        "event_keywords": ["chainsaw", "vehicle", "human", "threat", "critical", "movement"],
        "minimum_files": 25,
        "minimum_days": 10,
        "recommended_real_per_label": 30,
        "customer_pain": "The customer needs proof that remote-site normal conditions are separated from vehicles, human activity or machinery events.",
    },
}


def get_field_validation_scenarios():
    return list(FIELD_VALIDATION_SCENARIOS.keys())


def _label_matches_any(label, keywords):
    txt = str(label).lower()
    return any(k.lower() in txt for k in keywords)


def _field_label_summary(field_df, scenario):
    if not isinstance(field_df, pd.DataFrame) or len(field_df) == 0 or "Label" not in field_df.columns:
        return []
    cfg = FIELD_VALIDATION_SCENARIOS.get(scenario, {})
    normal_kw = cfg.get("normal_keywords", ["normal", "baseline", "healthy"])
    event_kw = cfg.get("event_keywords", ["event", "critical", "fault"])
    rows = []
    for label, count in field_df["Label"].astype(str).value_counts().items():
        if _label_matches_any(label, normal_kw):
            role = "normal/baseline"
        elif _label_matches_any(label, event_kw):
            role = "event/abnormal"
        else:
            role = "unknown/needs review"
        rows.append({"label": label, "samples": int(count), "detected_role": role})
    return rows


def _numeric_common_columns(df_a, df_b):
    if not isinstance(df_a, pd.DataFrame) or not isinstance(df_b, pd.DataFrame):
        return []
    cols = []
    for c in df_a.columns:
        if c in df_b.columns and c != "Label":
            if pd.api.types.is_numeric_dtype(df_a[c]) and pd.api.types.is_numeric_dtype(df_b[c]):
                cols.append(c)
    return cols


def _estimate_field_vs_synthetic_distance(field_df, synthetic_df):
    cols = _numeric_common_columns(field_df, synthetic_df)
    if len(cols) < 2 or len(field_df) == 0 or len(synthetic_df) == 0:
        return {"available": False, "distance_score": 0, "verdict": "Not enough common numeric columns for field-vs-synthetic comparison.", "common_columns": cols}
    try:
        synth = synthetic_df[cols].replace([np.inf, -np.inf], np.nan).dropna()
        real = field_df[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(synth) < 5 or len(real) < 2:
            return {"available": False, "distance_score": 0, "verdict": "Not enough clean rows for comparison.", "common_columns": cols}
        mu = synth.mean()
        sd = synth.std().replace(0, 1.0)
        z = ((real.mean() - mu).abs() / sd).mean()
        distance_score = int(np.clip(100 - (float(z) * 22), 0, 100))
        if distance_score >= 75:
            verdict = "Field feature profile is reasonably aligned with the current dataset."
        elif distance_score >= 50:
            verdict = "Field feature profile partly matches the current dataset, but more real data/tuning is recommended."
        else:
            verdict = "Field feature profile differs strongly from the current dataset. Synthetic assumptions need review."
        return {"available": True, "distance_score": distance_score, "mean_z_distance": float(z), "verdict": verdict, "common_columns": cols}
    except Exception as e:
        return {"available": False, "distance_score": 0, "verdict": f"Comparison failed: {e}", "common_columns": cols}


def build_real_field_validation_pack(
    project_name,
    validation_use_case,
    field_environment,
    device_setup,
    planned_days,
    minimum_real_files,
    field_df=None,
    synthetic_dataset_df=None,
    reliability_v2=None,
    normality_result=None,
    deployment_plan=None,
    edge_impulse_snapshot=None,
    edge_impulse_classifier_snapshot=None,
    release_success_snapshot=None,
    license_gate_snapshot=None,
):
    field_df = field_df if isinstance(field_df, pd.DataFrame) else pd.DataFrame()
    synthetic_dataset_df = synthetic_dataset_df if isinstance(synthetic_dataset_df, pd.DataFrame) else pd.DataFrame()
    cfg = FIELD_VALIDATION_SCENARIOS.get(validation_use_case, next(iter(FIELD_VALIDATION_SCENARIOS.values())))
    min_files = int(max(minimum_real_files or 0, cfg.get("minimum_files", 20)))
    planned_days = int(planned_days or 0)
    real_file_count = int(len(field_df))
    label_summary = _field_label_summary(field_df, validation_use_case)
    normal_count = sum(1 for r in label_summary if r.get("detected_role") == "normal/baseline")
    event_count = sum(1 for r in label_summary if r.get("detected_role") == "event/abnormal")
    unknown_count = sum(1 for r in label_summary if r.get("detected_role") == "unknown/needs review")
    label_count = len(label_summary)

    reliability_summary = reliability_v2.get("summary", {}) if isinstance(reliability_v2, dict) else {}
    rel_score = int(reliability_summary.get("trust_score_v2", reliability_summary.get("reliability_score", 0)) or 0)
    normality_score = 0
    if isinstance(normality_result, dict):
        normality_score = int((normality_result.get("summary", {}) or {}).get("normality_readiness_score", normality_result.get("normality_score", 0)) or 0)
    release_score = 0
    release_decision = "Unknown"
    if isinstance(release_success_snapshot, dict):
        release_score = int((release_success_snapshot.get("summary", {}) or {}).get("release_success_score", 0) or 0)
        release_decision = (release_success_snapshot.get("summary", {}) or {}).get("decision", "Unknown")

    distance = _estimate_field_vs_synthetic_distance(field_df, synthetic_dataset_df)
    distance_score = int(distance.get("distance_score", 0)) if distance.get("available") else 35 if real_file_count > 0 else 0

    coverage_score = 0
    coverage_score += min(30, int((real_file_count / max(min_files, 1)) * 30))
    coverage_score += min(20, label_count * 8)
    coverage_score += 18 if normal_count >= 1 else 0
    coverage_score += 16 if event_count >= 1 else 0
    coverage_score += min(10, planned_days)
    coverage_score -= 8 if unknown_count > 0 else 0
    coverage_score = int(np.clip(coverage_score, 0, 100))

    integration_score = 0
    integration_score += 18 if isinstance(reliability_v2, dict) and reliability_v2 else 0
    integration_score += 18 if isinstance(normality_result, dict) and normality_result else 0
    integration_score += 16 if isinstance(deployment_plan, dict) and deployment_plan else 0
    integration_score += 12 if isinstance(edge_impulse_snapshot, dict) and edge_impulse_snapshot else 0
    integration_score += 12 if isinstance(edge_impulse_classifier_snapshot, dict) and edge_impulse_classifier_snapshot else 0
    integration_score += 12 if isinstance(release_success_snapshot, dict) and release_success_snapshot else 0
    integration_score += 6 if isinstance(license_gate_snapshot, dict) and license_gate_snapshot else 0
    integration_score = int(np.clip(integration_score, 0, 100))

    field_evidence_score = int(np.clip(coverage_score * 0.45 + distance_score * 0.25 + integration_score * 0.20 + min(rel_score, 100) * 0.10, 0, 100))

    blockers = []
    must_collect = []
    if real_file_count < min_files:
        blockers.append(f"Only {real_file_count} real field files found; target is {min_files}+.")
        must_collect.append(f"Collect at least {max(0, min_files-real_file_count)} additional real field files.")
    if normal_count < 1:
        blockers.append("No clear normal/baseline field label detected.")
        must_collect.append("Capture clean normal/baseline data under expected operating conditions.")
    if event_count < 1:
        must_collect.append("Capture or verify at least one abnormal/event class for evaluation, without contaminating normal anomaly baseline training.")
    if planned_days < int(cfg.get("minimum_days", 7)):
        must_collect.append(f"Run the field pilot for at least {cfg.get('minimum_days', 7)} days to cover environmental variation.")
    if not distance.get("available"):
        must_collect.append("Keep common feature columns between synthetic and real datasets for field-vs-synthetic similarity evidence.")
    elif distance_score < 55:
        blockers.append("Field-vs-synthetic feature profile is weak; synthetic assumptions need tuning.")
    if release_decision == "NO-GO":
        blockers.append("Release Success Gate is NO-GO; resolve launch blockers before paid delivery.")

    if field_evidence_score >= 78 and len(blockers) == 0:
        decision = "FIELD VALIDATION READY"
        validation_stage = "Paid pilot evidence candidate"
        customer_message = "Strong enough for a paid pilot evidence package. Still not production certification."
    elif field_evidence_score >= 55:
        decision = "LIMITED FIELD EVIDENCE"
        validation_stage = "Private beta / conditional paid pilot"
        customer_message = "Useful field-validation package, but more real data or baseline coverage is needed before stronger claims."
    else:
        decision = "INTERNAL VALIDATION ONLY"
        validation_stage = "Internal testing / demo only"
        customer_message = "Not enough real field evidence yet. Use this as a collection plan, not as paid proof."

    evidence_items = [
        {"evidence": "Real field file count", "status": "pass" if real_file_count >= min_files else "weak", "value": real_file_count, "target": min_files},
        {"evidence": "Normal/baseline labels", "status": "pass" if normal_count >= 1 else "missing", "value": normal_count, "target": "1+"},
        {"evidence": "Event/abnormal labels", "status": "pass" if event_count >= 1 else "limited", "value": event_count, "target": "1+ for evaluation"},
        {"evidence": "Field-vs-synthetic similarity", "status": "pass" if distance_score >= 70 else "review", "value": distance_score, "target": "70+"},
        {"evidence": "Reliability Engine 2.0 present", "status": "pass" if reliability_v2 else "missing", "value": rel_score, "target": "available"},
        {"evidence": "Normality Engine present", "status": "pass" if normality_result else "missing", "value": normality_score, "target": "available"},
        {"evidence": "Deployment Planner present", "status": "pass" if deployment_plan else "missing", "value": bool(deployment_plan), "target": "available"},
        {"evidence": "Edge Impulse exports present", "status": "pass" if (edge_impulse_snapshot or edge_impulse_classifier_snapshot) else "missing", "value": bool(edge_impulse_snapshot or edge_impulse_classifier_snapshot), "target": "available"},
    ]

    field_test_plan = [
        {"step": 1, "action": "Capture baseline normal data", "details": "Record normal operating conditions across quiet/noisy and low/high load periods."},
        {"step": 2, "action": "Capture event/evaluation data", "details": "Collect abnormal/tamper/fault examples for evaluation without mixing them into normal anomaly training."},
        {"step": 3, "action": "Run EdgeTwin Real Bridge", "details": "Create signal fingerprints and synthetic-to-real variants from real WAV/CSV data."},
        {"step": 4, "action": "Run Normality Engine", "details": "Confirm what is normal, warning and abnormal using baseline distance and top deviation features."},
        {"step": 5, "action": "Export to Edge Impulse", "details": "Use normal-only data for K-means anomaly baseline and labelled data for classifier testing."},
        {"step": 6, "action": "Document go/no-go", "details": "Use Reliability 2.0, Success Gate and this Field Validation Pack before paid or production claims."},
    ]

    safe_claim = (
        "This package supports a real-field Edge AI pilot decision by combining real sensor evidence, dataset quality checks, "
        "normal/abnormal baseline analysis, hardware/deployment guidance and export-ready files. It does not certify production performance."
    )
    claims_to_avoid = [
        "Do not claim guaranteed production accuracy from this pack alone.",
        "Do not claim the synthetic dataset fully replaces real validation data.",
        "Do not claim safety-critical readiness without customer-side certification and acceptance tests.",
        "Do not train anomaly K-means on abnormal/event data as if it were normal baseline data.",
    ]

    return _json_safe({
        "engine": FIELD_VALIDATION_VERSION,
        "disclaimer": FIELD_VALIDATION_DISCLAIMER,
        "summary": {
            "field_evidence_score": field_evidence_score,
            "coverage_score": coverage_score,
            "field_vs_synthetic_score": distance_score,
            "integration_score": integration_score,
            "reliability_score": rel_score,
            "normality_score": normality_score,
            "release_success_score": release_score,
            "decision": decision,
            "validation_stage": validation_stage,
            "customer_message": customer_message,
            "real_file_count": real_file_count,
            "label_count": label_count,
            "normal_label_count": normal_count,
            "event_label_count": event_count,
        },
        "project_name": project_name,
        "validation_use_case": validation_use_case,
        "field_environment": field_environment,
        "device_setup": device_setup,
        "planned_days": planned_days,
        "customer_pain": cfg.get("customer_pain", "Customer needs real-world evidence before a serious pilot."),
        "field_label_summary": label_summary,
        "field_vs_synthetic": distance,
        "evidence_items": evidence_items,
        "field_test_plan": field_test_plan,
        "blockers": blockers,
        "must_collect_before_production": must_collect,
        "safe_customer_claim": safe_claim,
        "claims_to_avoid": claims_to_avoid,
        "recommended_real_per_label": cfg.get("recommended_real_per_label", 30),
        "created_at": _now(),
    })


def generate_real_field_validation_pdf(project_name, snapshot):
    snapshot = snapshot or {}
    summary = snapshot.get("summary", {}) or {}
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt=clean_pdf_text("EdgeTwin Studio Real Field Validation Pack"), ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("V30 Field Evidence & Pilot Validation Gate"), ln=True, align="C")
    pdf.ln(8)
    safe_pdf_cell(pdf, "Project", 8, True)
    safe_pdf_cell(pdf, f"Project: {project_name}")
    safe_pdf_cell(pdf, f"Use case: {snapshot.get('validation_use_case', '')}")
    safe_pdf_cell(pdf, f"Environment: {snapshot.get('field_environment', '')}")
    safe_pdf_multicell(pdf, f"Device setup: {snapshot.get('device_setup', '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Field Validation Decision", 8, True)
    for key in ["field_evidence_score", "coverage_score", "field_vs_synthetic_score", "integration_score", "reliability_score", "normality_score"]:
        safe_pdf_cell(pdf, f"{key.replace('_',' ').title()}: {summary.get(key, 0)}")
    safe_pdf_cell(pdf, f"Decision: {summary.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Stage: {summary.get('validation_stage', 'Unknown')}")
    safe_pdf_multicell(pdf, summary.get("customer_message", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Customer Pain", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("customer_pain", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Evidence Coverage", 8, True)
    for row in snapshot.get("evidence_items", []):
        safe_pdf_multicell(pdf, f"- {row.get('evidence')}: {row.get('status')} | value={row.get('value')} target={row.get('target')}")
    if snapshot.get("blockers"):
        pdf.ln(4)
        safe_pdf_cell(pdf, "Blockers", 8, True)
        for item in snapshot.get("blockers", []):
            safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Must Collect Before Production", 8, True)
    for item in snapshot.get("must_collect_before_production", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Field Test Plan", 8, True)
    for row in snapshot.get("field_test_plan", []):
        safe_pdf_multicell(pdf, f"{row.get('step')}. {row.get('action')}: {row.get('details')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Safe Customer Claim", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("safe_customer_claim", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Claims To Avoid", 8, True)
    for item in snapshot.get("claims_to_avoid", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Disclaimer", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", FIELD_VALIDATION_DISCLAIMER))
    return safe_pdf_output(pdf)


def create_real_field_validation_bundle(project_name, snapshot, field_df=None, synthetic_dataset_df=None):
    snapshot = snapshot or {}
    field_df = field_df if isinstance(field_df, pd.DataFrame) else pd.DataFrame()
    synthetic_dataset_df = synthetic_dataset_df if isinstance(synthetic_dataset_df, pd.DataFrame) else pd.DataFrame()
    pdf_bytes = generate_real_field_validation_pdf(project_name, snapshot)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("real_field_validation_v30.pdf", pdf_bytes)
        zf.writestr("field_validation_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        if snapshot.get("evidence_items"):
            zf.writestr("evidence_coverage.csv", pd.DataFrame(snapshot.get("evidence_items", [])).to_csv(index=False))
        if snapshot.get("field_test_plan"):
            zf.writestr("field_test_plan.csv", pd.DataFrame(snapshot.get("field_test_plan", [])).to_csv(index=False))
        if snapshot.get("field_label_summary"):
            zf.writestr("field_label_summary.csv", pd.DataFrame(snapshot.get("field_label_summary", [])).to_csv(index=False))
        if len(field_df) > 0:
            zf.writestr("real_field_uploaded_features.csv", field_df.to_csv(index=False))
        if len(synthetic_dataset_df) > 0:
            zf.writestr("synthetic_dataset_snapshot.csv", synthetic_dataset_df.head(2000).to_csv(index=False))
        zf.writestr("must_collect_before_production.txt", "\n".join(snapshot.get("must_collect_before_production", [])))
        zf.writestr("claims_to_avoid.txt", "\n".join(snapshot.get("claims_to_avoid", [])))
        zf.writestr("README.txt", f"""EdgeTwin Studio V30 Real Field Validation Pack

Project: {project_name}
Decision: {(snapshot.get('summary', {}) or {}).get('decision', 'Unknown')}
Field evidence score: {(snapshot.get('summary', {}) or {}).get('field_evidence_score', 0)}%
Use case: {snapshot.get('validation_use_case', 'Unknown')}

This bundle is meant to support a real-world pilot decision.
It is not production certification and does not guarantee final model accuracy.

{snapshot.get('disclaimer', FIELD_VALIDATION_DISCLAIMER)}
""")
    return zip_buf.getvalue()

# ============================================================
# V31 EDGE DEPLOYMENT STARTER KIT
# ============================================================

EDGE_STARTER_DISCLAIMER = (
    "The Edge Deployment Starter Kit is a pilot-start package. Generated firmware/config files are templates, "
    "not certified production firmware. Validate sensors, wiring, power, safety, regulatory requirements and field behavior before deployment."
)

DEPLOYMENT_STARTER_PROFILES = {
    "ESP32-S3": {"firmware_file": "firmware/esp32_s3_edgetwin_starter.ino", "framework": "Arduino / ESP-IDF compatible starter", "best_for": "Audio/vibration TinyML nodes, WiFi/MQTT pilots and Edge Impulse deployment experiments.", "transport": ["WiFi / MQTT", "BLE debug", "Serial"]},
    "RAK4631 / nRF52840": {"firmware_file": "firmware/rak4631_lora_edgetwin_starter.ino", "framework": "Arduino / Zephyr-style starter", "best_for": "Low-power LoRaWAN feature/event telemetry where raw audio is not transmitted.", "transport": ["LoRa / LoRaWAN", "BLE debug", "Serial"]},
    "STM32U5": {"firmware_file": "firmware/stm32u5_edgetwin_starter.cpp", "framework": "STM32Cube / C++ starter", "best_for": "Secure low-power industrial sensing and TinyML feature extraction.", "transport": ["UART", "I2C/SPI sensor bus", "MQTT through gateway"]},
    "STM32H7": {"firmware_file": "firmware/stm32h7_dsp_edgetwin_starter.cpp", "framework": "STM32Cube / CMSIS-DSP starter", "best_for": "Higher-speed vibration/audio DSP and industrial inference experiments.", "transport": ["Ethernet / MQTT", "UART", "Gateway bridge"]},
    "Raspberry Pi Zero 2 W": {"firmware_file": "python_gateway/pi_zero2_gateway_starter.py", "framework": "Python gateway starter", "best_for": "Light local gateway, MQTT bridge and field validation scripts.", "transport": ["WiFi / MQTT", "USB serial", "Local CSV logging"]},
    "Raspberry Pi 5": {"firmware_file": "python_gateway/pi5_gateway_starter.py", "framework": "Python/Linux gateway starter", "best_for": "Local validation, dashboards, heavier preprocessing and gateway orchestration.", "transport": ["Ethernet / MQTT", "WiFi / MQTT", "Local API"]},
    "Generic Linux Gateway": {"firmware_file": "python_gateway/linux_gateway_starter.py", "framework": "Python/Linux service starter", "best_for": "Industrial gateway/server pilots, API integration and local orchestration.", "transport": ["Ethernet / MQTT", "REST API", "Database logging"]},
}


def _safe_slug(value, fallback="edgetwin_project"):
    import re
    text = str(value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def _numeric_feature_columns_for_deployment(dataset_df, max_features=12):
    df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    if len(df) == 0:
        return ["RMS", "Std", "Kurtosis", "CrestFactor", "ZCR", "SpectralCentroid", "SpectralRolloff", "SpectralFlatness"][:max_features]
    numeric = [c for c in df.columns if c != "Label" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return ["RMS", "Std", "Kurtosis", "CrestFactor"][:max_features]
    preferred = ["RMS", "Std", "Kurtosis", "CrestFactor", "ZCR", "SpectralCentroid", "SpectralRolloff", "SpectralFlatness", "AudioScore", "VibrationScore", "FusionScore", "HealthScore", "Confidence"]
    ordered = []
    for c in preferred:
        if c in numeric and c not in ordered:
            ordered.append(c)
    remaining = [c for c in numeric if c not in ordered]
    try:
        variances = df[remaining].var(numeric_only=True).sort_values(ascending=False)
        remaining = list(variances.index)
    except Exception:
        pass
    return (ordered + remaining)[:max_features]


def _detect_inference_mode(edge_impulse_snapshot=None, edge_impulse_classifier_snapshot=None, normality_result=None):
    if isinstance(edge_impulse_classifier_snapshot, dict) and edge_impulse_classifier_snapshot:
        return "classification"
    if isinstance(edge_impulse_snapshot, dict) and edge_impulse_snapshot:
        return "anomaly_kmeans"
    if isinstance(normality_result, dict) and normality_result:
        return "normality_threshold"
    return "feature_logging_only"


def _starter_mqtt_schema(project_slug, features):
    return {
        "topic_event": f"edgetwin/{project_slug}/node/{{node_id}}/event",
        "topic_health": f"edgetwin/{project_slug}/node/{{node_id}}/health",
        "payload_event": {
            "timestamp": "ISO-8601 string",
            "project": project_slug,
            "node_id": "string",
            "firmware_version": "edgetwin-starter-v31",
            "mode": "classification | anomaly_kmeans | normality_threshold | feature_logging_only",
            "predicted_label": "string or null",
            "confidence": "0..1 float or null",
            "anomaly_score": "float or null",
            "normality_score": "0..100 float or null",
            "features": {f: "float" for f in features[:10]},
            "battery_v": "float or null",
            "temperature_c": "float or null",
        },
        "payload_health": {
            "timestamp": "ISO-8601 string",
            "project": project_slug,
            "node_id": "string",
            "uptime_s": "integer",
            "battery_v": "float",
            "rssi": "integer or null",
            "free_ram": "integer or null",
            "status": "ok | warning | fault",
        },
    }


def _cpp_feature_stub(project_slug, config, features, board, mode, communication):
    sr = int(config.get("sample_rate_hz", 16000))
    fft_size = int(config.get("fft_size", 1024))
    window_ms = int(float(config.get("window_seconds", 1.0)) * 1000)
    feature_array = ", ".join(['"{}"'.format(f) for f in features])
    return """/*
  EdgeTwin Studio V31 Edge Deployment Starter Kit
  Project: {project_slug}
  Target board: {board}
  Inference mode: {mode}
  Communication: {communication}

  This is pilot starter code, not production-certified firmware.
  Replace sensor_read_sample(), run_model_stub() and publish_event_stub() with your hardware/model implementation.
*/

#include <Arduino.h>
#include <math.h>

#define EDGETWIN_SAMPLE_RATE {sr}
#define EDGETWIN_FFT_SIZE {fft_size}
#define EDGETWIN_WINDOW_MS {window_ms}
#define EDGETWIN_FEATURE_COUNT {feature_count}

const char* FEATURE_NAMES[EDGETWIN_FEATURE_COUNT] = {{{feature_array}}};
float features[EDGETWIN_FEATURE_COUNT];

float sensor_read_sample() {{
  // TODO: replace with ADC/I2S/IMU sample read.
  return 0.0f;
}}

void compute_basic_features(float* buffer, int n) {{
  float sum = 0.0f;
  float sumsq = 0.0f;
  float maxabs = 0.0f;
  for (int i = 0; i < n; i++) {{ sum += buffer[i]; }}
  float mean = sum / max(n, 1);
  for (int i = 0; i < n; i++) {{
    float centered = buffer[i] - mean;
    sumsq += centered * centered;
    float a = fabs(centered);
    if (a > maxabs) maxabs = a;
  }}
  float rms = sqrt(sumsq / max(n, 1));
  for (int i = 0; i < EDGETWIN_FEATURE_COUNT; i++) features[i] = 0.0f;
  for (int i = 0; i < EDGETWIN_FEATURE_COUNT; i++) {{
    String name = String(FEATURE_NAMES[i]);
    if (name == "RMS") features[i] = rms;
    else if (name == "Std") features[i] = rms;
    else if (name == "CrestFactor") features[i] = maxabs / max(rms, 0.000001f);
    // TODO: add FFT/kurtosis/ZCR/spectral features using CMSIS-DSP, arduinoFFT or Edge Impulse DSP blocks.
  }}
}}

float run_model_stub(float* feature_values, int feature_count, const char** out_label) {{
  // TODO: replace with Edge Impulse classifier/anomaly inference or your own TinyML model.
  *out_label = "pilot_stub";
  return 0.0f;
}}

void publish_event_stub(const char* label, float score) {{
  Serial.print("{{\"project\":\"{project_slug}\",\"label\":\"");
  Serial.print(label);
  Serial.print("\",\"score\":");
  Serial.print(score, 4);
  Serial.println("}}");
}}

void setup() {{
  Serial.begin(115200);
  delay(1000);
  Serial.println("EdgeTwin deployment starter booting...");
}}

void loop() {{
  static float buffer[EDGETWIN_FFT_SIZE];
  for (int i = 0; i < EDGETWIN_FFT_SIZE; i++) {{
    buffer[i] = sensor_read_sample();
    delayMicroseconds(1000000 / EDGETWIN_SAMPLE_RATE);
  }}
  compute_basic_features(buffer, EDGETWIN_FFT_SIZE);
  const char* label = "unknown";
  float score = run_model_stub(features, EDGETWIN_FEATURE_COUNT, &label);
  publish_event_stub(label, score);
}}
""".format(project_slug=project_slug, board=board, mode=mode, communication=communication, sr=sr, fft_size=fft_size, window_ms=window_ms, feature_count=len(features), feature_array=feature_array)


def _python_gateway_stub(project_slug, config, features, mode):
    cols = ", ".join([repr(f) for f in features])
    config_txt = safe_json_dumps(config, indent=2)
    return f'''"""
EdgeTwin Studio V31 Python Gateway Starter
Project: {project_slug}
Mode: {mode}
Pilot starter only. Replace read_sensor_window() and run_model() with real sensor/model code.
"""
import json
import time
from datetime import datetime, timezone

FEATURE_COLUMNS = [{cols}]
CONFIG = {config_txt}


def read_sensor_window():
    return {{name: 0.0 for name in FEATURE_COLUMNS}}


def run_model(features):
    return {{"label": "pilot_stub", "confidence": None, "anomaly_score": None}}


def build_event(features, result):
    return {{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": "{project_slug}",
        "node_id": "gateway-001",
        "firmware_version": "edgetwin-starter-v31",
        "mode": "{mode}",
        "predicted_label": result.get("label"),
        "confidence": result.get("confidence"),
        "anomaly_score": result.get("anomaly_score"),
        "features": features,
    }}


def main():
    while True:
        features = read_sensor_window()
        result = run_model(features)
        print(json.dumps(build_event(features, result)))
        time.sleep(float(CONFIG.get("window_seconds", 1.0)))


if __name__ == "__main__":
    main()
'''


def _validation_script(features):
    cols = ", ".join([repr(f) for f in features])
    return f'''"""Validate an EdgeTwin deployment CSV before flashing/field testing.
Usage: python validate_edge_deployment_csv.py pilot_training_features.csv
"""
import sys
import pandas as pd

REQUIRED_FEATURES = [{cols}]


def main(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        print("FAIL missing features:", missing)
        raise SystemExit(2)
    if "Label" not in df.columns:
        print("WARN: no Label column; classifier training may not be possible.")
    numeric_ok = True
    for c in REQUIRED_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[c]):
            print("FAIL non-numeric feature:", c)
            numeric_ok = False
    if df[REQUIRED_FEATURES].isna().any().any():
        print("FAIL NaN values detected in required features.")
        numeric_ok = False
    print("Rows:", len(df))
    print("Required features:", len(REQUIRED_FEATURES))
    if "Label" in df.columns:
        print("Labels:")
        print(df["Label"].value_counts())
    print("PASS" if numeric_ok else "FAIL")
    raise SystemExit(0 if numeric_ok else 2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_edge_deployment_csv.py <csv>")
        raise SystemExit(1)
    main(sys.argv[1])
'''


def _node_red_flow_stub(project_slug, mqtt_schema):
    return [
        {"id": "edgetwin_mqtt_in", "type": "mqtt in", "z": "edgetwin_flow", "name": "EdgeTwin event input", "topic": mqtt_schema.get("topic_event", f"edgetwin/{project_slug}/node/+/event"), "qos": "0", "datatype": "json", "broker": "mqtt_broker_placeholder", "x": 150, "y": 120, "wires": [["edgetwin_debug", "edgetwin_alert_filter"]]},
        {"id": "edgetwin_alert_filter", "type": "switch", "z": "edgetwin_flow", "name": "Anomaly / critical?", "property": "payload.predicted_label", "rules": [{"t": "regex", "v": "anomaly|critical|tamper|failure", "vt": "str", "case": False}], "x": 390, "y": 120, "wires": [["edgetwin_alert_debug"]]},
        {"id": "edgetwin_debug", "type": "debug", "z": "edgetwin_flow", "name": "All events", "active": True, "tosidebar": True, "x": 390, "y": 180, "wires": []},
        {"id": "edgetwin_alert_debug", "type": "debug", "z": "edgetwin_flow", "name": "Alerts", "active": True, "tosidebar": True, "x": 620, "y": 120, "wires": []},
    ]


def build_edge_deployment_starter_kit(project_name, dataset_df=None, manifest=None, hardware_result=None, deployment_plan=None, reliability_v2=None, normality_result=None, edge_impulse_snapshot=None, edge_impulse_classifier_snapshot=None, field_validation_snapshot=None, target_board=None, communication=None, inference_mode="auto", sample_rate=None, fft_size=None):
    df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    manifest = manifest if isinstance(manifest, dict) else {}
    deployment_plan = deployment_plan if isinstance(deployment_plan, dict) else {}
    hardware = hardware_result if isinstance(hardware_result, dict) else {}
    project_slug = _safe_slug(project_name)
    if not hardware and deployment_plan.get("hardware"):
        hardware = {"recommendation": (deployment_plan.get("hardware") or {}).get("recommended_board")}
    board = target_board or hardware.get("recommendation") or (deployment_plan.get("hardware") or {}).get("recommended_board") or "ESP32-S3"
    if board not in DEPLOYMENT_STARTER_PROFILES:
        if "RAK" in str(board): board = "RAK4631 / nRF52840"
        elif "STM32H7" in str(board): board = "STM32H7"
        elif "STM32" in str(board): board = "STM32U5"
        elif "Pi 5" in str(board): board = "Raspberry Pi 5"
        elif "Pi" in str(board): board = "Raspberry Pi Zero 2 W"
        else: board = "ESP32-S3"
    profile = DEPLOYMENT_STARTER_PROFILES.get(board, DEPLOYMENT_STARTER_PROFILES["ESP32-S3"])
    edge_settings = deployment_plan.get("edge_settings", {}) if isinstance(deployment_plan.get("edge_settings"), dict) else {}
    sr = int(sample_rate or edge_settings.get("sample_rate_hz") or manifest.get("sample_rate") or 16000)
    fft = int(fft_size or edge_settings.get("fft_size") or _deployment_fft_size(sr, manifest.get("priority", "balanced")))
    window_s = float(edge_settings.get("window_seconds", 1.0 if sr <= 8000 else 0.5))
    comm = communication or deployment_plan.get("communication") or infer_deployment_defaults(df, manifest, hardware).get("communication")
    features = _numeric_feature_columns_for_deployment(df, max_features=12)
    mode = _detect_inference_mode(edge_impulse_snapshot, edge_impulse_classifier_snapshot, normality_result) if inference_mode == "auto" else inference_mode
    config = _json_safe({"project": project_slug, "version": "edgetwin-v31-starter", "target_board": board, "framework": profile.get("framework"), "sample_rate_hz": sr, "fft_size": fft, "window_seconds": window_s, "feature_columns": features, "inference_mode": mode, "communication": comm, "payload_policy": "Send features/events first; do not stream raw audio/vibration continuously unless bandwidth and consent allow it.", "field_validation_required": True})
    mqtt_schema = _starter_mqtt_schema(project_slug, features)
    assets = {"firmware_code": _cpp_feature_stub(project_slug, config, features, board, mode, comm), "python_gateway": _python_gateway_stub(project_slug, config, features, mode), "validation_script": _validation_script(features), "node_red_flow": _node_red_flow_stub(project_slug, mqtt_schema)}
    evidence = []
    def add_evidence(name, ok, weight, message): evidence.append({"evidence": name, "status": "pass" if ok else "missing", "weight": weight, "message": message})
    add_evidence("Dataset loaded", len(df) > 0, 15, "A dataset is required to define feature contract and labels.")
    add_evidence("Label column", len(df) > 0 and "Label" in df.columns, 10, "Classifier/anomaly evaluation needs label context.")
    add_evidence("Deployment plan", bool(deployment_plan), 15, "BOM/power/comms plan makes the starter kit practical.")
    add_evidence("Reliability evidence", bool(reliability_v2), 15, "Reliability Engine 2.0 should be attached before paid pilot delivery.")
    add_evidence("Normality/anomaly evidence", bool(normality_result) or bool(edge_impulse_snapshot), 15, "Normal-vs-abnormal baseline supports anomaly deployment.")
    add_evidence("Edge Impulse route", bool(edge_impulse_snapshot) or bool(edge_impulse_classifier_snapshot), 10, "Export route makes next training/deployment step clearer.")
    add_evidence("Field validation", bool(field_validation_snapshot), 20, "Real field validation is the strongest evidence before serious deployment claims.")
    score = int(np.clip(sum(row["weight"] for row in evidence if row["status"] == "pass"), 0, 100))
    blockers, warnings = [], []
    if len(df) == 0: blockers.append("No dataset loaded. Generate a pilot dataset or upload real field data first.")
    if len(df) > 0 and "Label" not in df.columns: warnings.append("Dataset has no Label column. Feature logging can proceed, but classifier export is weak.")
    if not deployment_plan: warnings.append("Deployment Planner was not run. BOM/power/comms values are inferred and weaker.")
    if not (edge_impulse_snapshot or edge_impulse_classifier_snapshot): warnings.append("No Edge Impulse export snapshot attached. Starter kit includes stubs, but model integration remains manual.")
    if not field_validation_snapshot: warnings.append("No real field validation attached. Do not claim production readiness.")
    if comm == "LoRa / LoRaWAN" and sr >= 16000: warnings.append("LoRa is not suitable for raw audio streaming. Send features/events only.")
    if blockers:
        decision, stage, customer_message = "NO-GO", "Internal only", "Create a dataset and deployment evidence before sharing this starter kit externally."
    elif score >= 75:
        decision, stage, customer_message = "GO", "Pilot implementation starter-ready", "This package is suitable as a controlled pilot implementation starter, with field validation still required."
    elif score >= 50:
        decision, stage, customer_message = "CONDITIONAL GO", "Private beta starter-ready", "This package can support a private technical beta, but missing evidence should be collected before paid deployment claims."
    else:
        decision, stage, customer_message = "NO-GO", "Demo-only starter", "Use this internally for demos until reliability, deployment and field evidence are attached."
    feature_contract = []
    if len(df) > 0:
        for f in features:
            try:
                feature_contract.append({"feature": f, "dtype": str(df[f].dtype) if f in df.columns else "float", "min": float(df[f].min()) if f in df.columns else None, "max": float(df[f].max()) if f in df.columns else None, "mean": float(df[f].mean()) if f in df.columns else None, "required": True})
            except Exception:
                feature_contract.append({"feature": f, "dtype": "float", "min": None, "max": None, "mean": None, "required": True})
    else:
        feature_contract = [{"feature": f, "dtype": "float", "min": None, "max": None, "mean": None, "required": True} for f in features]
    integration_steps = [
        {"step": 1, "action": "Flash or run starter", "details": f"Use {profile.get('firmware_file')} for {board}."},
        {"step": 2, "action": "Replace sensor read stub", "details": "Connect real I2S/IMU/ADC/CSV gateway input and verify sample rate."},
        {"step": 3, "action": "Align features", "details": "Keep feature names identical to EdgeTwin/Edge Impulse training exports."},
        {"step": 4, "action": "Attach model", "details": "Replace run_model_stub with Edge Impulse/TFLite/classifier/anomaly code."},
        {"step": 5, "action": "Publish telemetry", "details": "Use MQTT schema and send features/events first."},
        {"step": 6, "action": "Run field validation", "details": "Collect normal baseline and controlled events before production claims."},
    ]
    safe_claims = ["Generated a pilot starter kit with feature contract, telemetry schema and implementation stubs.", "Designed to accelerate controlled Edge AI pilot implementation.", "Requires hardware verification and field validation before production use."]
    claims_to_avoid = ["Do not claim production-ready firmware from generated stubs.", "Do not claim guaranteed model accuracy without field validation.", "Do not claim safety/security certification from this package."]
    files = ["edge_deployment_starter_report_v31.pdf", "deployment_config.json", "feature_contract.csv", "mqtt_payload_schema.json", profile.get("firmware_file"), "python/validate_edge_deployment_csv.py", "node_red/edgetwin_mqtt_flow.json", "README.md"]
    return _json_safe({"engine": "EdgeTwin Studio V31 Edge Deployment Starter Kit", "created_at": _now(), "project_name": project_name, "project_slug": project_slug, "summary": {"starter_score": score, "decision": decision, "stage": stage, "target_board": board, "inference_mode": mode, "communication": comm, "customer_message": customer_message}, "config": config, "profile": profile, "feature_contract": feature_contract, "mqtt_schema": mqtt_schema, "evidence": evidence, "blockers": blockers, "warnings": warnings, "integration_steps": integration_steps, "safe_claims": safe_claims, "claims_to_avoid": claims_to_avoid, "generated_files": files, "disclaimer": EDGE_STARTER_DISCLAIMER, "assets": assets})


def generate_edge_deployment_starter_pdf(project_name, snapshot):
    snapshot = snapshot or {}; summary = snapshot.get("summary", {}) or {}
    pdf = FPDF(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18); pdf.cell(0, 10, txt="EdgeTwin Deployment Starter Kit", ln=True, align="C")
    pdf.set_font("Arial", "I", 11); pdf.cell(0, 8, txt=clean_pdf_text("V31 - Pilot implementation starter package"), ln=True, align="C"); pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C"); pdf.ln(8)
    safe_pdf_cell(pdf, "Starter Decision", 8, True)
    for label, key in [("Starter score", "starter_score"), ("Decision", "decision"), ("Stage", "stage"), ("Target board", "target_board"), ("Inference mode", "inference_mode"), ("Communication", "communication")]:
        val = summary.get(key, 0 if key == "starter_score" else "Unknown")
        suffix = "%" if key == "starter_score" else ""
        safe_pdf_cell(pdf, f"{label}: {val}{suffix}")
    safe_pdf_multicell(pdf, summary.get("customer_message", "")); pdf.ln(4)
    safe_pdf_cell(pdf, "Evidence", 8, True)
    for row in snapshot.get("evidence", []): safe_pdf_multicell(pdf, f"- {row.get('evidence')}: {row.get('status')} | {row.get('message')}")
    if snapshot.get("blockers"):
        pdf.ln(4); safe_pdf_cell(pdf, "Blockers", 8, True)
        for item in snapshot.get("blockers", []): safe_pdf_multicell(pdf, f"- {item}")
    if snapshot.get("warnings"):
        pdf.ln(4); safe_pdf_cell(pdf, "Warnings", 8, True)
        for item in snapshot.get("warnings", []): safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4); safe_pdf_cell(pdf, "Feature Contract", 8, True)
    for row in snapshot.get("feature_contract", [])[:14]: safe_pdf_multicell(pdf, f"- {row.get('feature')} ({row.get('dtype')}) required={row.get('required')}")
    pdf.ln(4); safe_pdf_cell(pdf, "Integration Steps", 8, True)
    for row in snapshot.get("integration_steps", []): safe_pdf_multicell(pdf, f"{row.get('step')}. {row.get('action')}: {row.get('details')}")
    pdf.ln(4); safe_pdf_cell(pdf, "Safe Claims", 8, True)
    for item in snapshot.get("safe_claims", []): safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4); safe_pdf_cell(pdf, "Claims To Avoid", 8, True)
    for item in snapshot.get("claims_to_avoid", []): safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4); safe_pdf_cell(pdf, "Disclaimer", 8, True); safe_pdf_multicell(pdf, snapshot.get("disclaimer", EDGE_STARTER_DISCLAIMER))
    return safe_pdf_output(pdf)


def create_edge_deployment_starter_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}; assets = snapshot.get("assets", {}) or {}; profile = snapshot.get("profile", {}) or {}; dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    pdf_bytes = generate_edge_deployment_starter_pdf(project_name, snapshot)
    feature_df = pd.DataFrame(snapshot.get("feature_contract", [])); zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("edge_deployment_starter_report_v31.pdf", pdf_bytes)
        zf.writestr("deployment_starter_snapshot.json", safe_json_dumps({k:v for k,v in snapshot.items() if k != "assets"}, indent=2))
        zf.writestr("deployment_config.json", safe_json_dumps(snapshot.get("config", {}), indent=2))
        zf.writestr("mqtt_payload_schema.json", safe_json_dumps(snapshot.get("mqtt_schema", {}), indent=2))
        if len(feature_df) > 0: zf.writestr("feature_contract.csv", feature_df.to_csv(index=False))
        firmware_file = profile.get("firmware_file", "firmware/edgetwin_starter.ino")
        if str(firmware_file).startswith("python_gateway"):
            zf.writestr(firmware_file, assets.get("python_gateway", "")); zf.writestr("firmware/edge_node_feature_stub.ino", assets.get("firmware_code", ""))
        else:
            zf.writestr(firmware_file, assets.get("firmware_code", "")); zf.writestr("python_gateway/gateway_starter.py", assets.get("python_gateway", ""))
        zf.writestr("python/validate_edge_deployment_csv.py", assets.get("validation_script", ""))
        zf.writestr("node_red/edgetwin_mqtt_flow.json", safe_json_dumps(assets.get("node_red_flow", []), indent=2))
        if len(dataset_df) > 0:
            features = [r.get("feature") for r in snapshot.get("feature_contract", []) if r.get("feature") in dataset_df.columns]
            cols = (["Label"] if "Label" in dataset_df.columns else []) + features
            if cols: zf.writestr("dataset_feature_contract_snapshot.csv", dataset_df[cols].head(2000).to_csv(index=False))
        next_steps = "\n".join([f"{row.get('step')}. {row.get('action')} - {row.get('details')}" for row in snapshot.get("integration_steps", [])])
        zf.writestr("README.md", f"""# EdgeTwin Studio V31 Edge Deployment Starter Kit

Project: {project_name}
Decision: {(snapshot.get('summary', {}) or {}).get('decision', 'Unknown')}
Starter score: {(snapshot.get('summary', {}) or {}).get('starter_score', 0)}%
Target board: {(snapshot.get('summary', {}) or {}).get('target_board', 'Unknown')}
Inference mode: {(snapshot.get('summary', {}) or {}).get('inference_mode', 'Unknown')}

## What this bundle contains
- deployment_config.json: feature/sample-rate/board configuration
- feature_contract.csv: required feature names for training and firmware alignment
- mqtt_payload_schema.json: event/health telemetry schema
- firmware or gateway starter code
- Python CSV validator
- Node-RED MQTT starter flow
- PDF report with evidence, warnings and safe claims

## Important
{snapshot.get('disclaimer', EDGE_STARTER_DISCLAIMER)}

## Next steps
{next_steps}
""")
    return zip_buf.getvalue()


# ============================================================
# V31.2 STORAGE / SCALABILITY BUNDLE
# ============================================================

def generate_scalability_storage_pdf(project_name, snapshot):
    snapshot = snapshot or {}
    storage = snapshot.get("storage", {}) or {}
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Storage & Scalability Report", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("V31.2 - Recovery and scale-readiness checkpoint"), ln=True, align="C")
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)
    safe_pdf_cell(pdf, "Scalability Snapshot", 8, True)
    safe_pdf_cell(pdf, f"Scalability score: {snapshot.get('scalability_score', 0)}%")
    safe_pdf_cell(pdf, f"Dataset rows: {snapshot.get('dataset_rows', 0)}")
    safe_pdf_cell(pdf, f"Dataset cols: {snapshot.get('dataset_cols', 0)}")
    safe_pdf_cell(pdf, f"Plan: {snapshot.get('plan', 'Unknown')}")
    safe_pdf_cell(pdf, f"Sample limit: {snapshot.get('sample_limit', 0)}")
    safe_pdf_multicell(pdf, f"Recommendation: {snapshot.get('recommendation', '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Storage Status", 8, True)
    for key in ["mode", "storage_root", "projects_dir", "exports_dir", "file_count", "total_mb", "database", "database_role", "future_cloud_target"]:
        safe_pdf_multicell(pdf, f"{key}: {storage.get(key, '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Issues", 8, True)
    for item in snapshot.get("issues", []):
        safe_pdf_multicell(pdf, f"[{item.get('severity', 'info').upper()}] {item.get('message', '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Next Step", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("next_step", "PostgreSQL + S3/MinIO when moving from closed beta to public SaaS."))
    return safe_pdf_output(pdf)


def create_scalability_storage_bundle(project_name, snapshot, projects_df=None):
    snapshot = snapshot or {}
    projects_df = projects_df if isinstance(projects_df, pd.DataFrame) else pd.DataFrame()
    pdf_bytes = generate_scalability_storage_pdf(project_name, snapshot)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("storage_scalability_report_v31_2.pdf", pdf_bytes)
        zf.writestr("storage_scalability_snapshot.json", safe_json_dumps(snapshot, indent=2))
        if snapshot.get("issues"):
            zf.writestr("storage_scalability_issues.csv", pd.DataFrame(snapshot.get("issues", [])).to_csv(index=False))
        if len(projects_df) > 0:
            zf.writestr("project_storage_index.csv", projects_df.to_csv(index=False))
        zf.writestr("README.md", f"""# EdgeTwin Studio V31.2 Storage & Scalability Bundle

Project: {project_name}
Scalability score: {snapshot.get('scalability_score', 0)}%
Storage mode: {(snapshot.get('storage', {}) or {}).get('mode', 'Unknown')}
Dataset rows: {snapshot.get('dataset_rows', 0)}

This bundle documents whether the current local beta storage setup is safe enough for continued testing.
It does not replace production infrastructure hardening. Move to PostgreSQL + S3/MinIO before public multi-user SaaS.
""")
    return zip_buf.getvalue()


# ============================================================
# V32 OPERATIONAL CONTROL CENTER
# ============================================================

OPERATIONAL_CONTROL_DISCLAIMER = (
    "Operational readiness is a release-management estimate. It does not certify production safety. "
    "Use real field validation, customer acceptance and engineering review before production deployment."
)


def _component_status(name, present=False, score=None, required_for_paid=False, notes=""):
    if score is not None:
        if score >= 80:
            status = "Ready"
        elif score >= 60:
            status = "Needs attention"
        else:
            status = "Weak"
    else:
        status = "Ready" if present else "Missing"
    return {
        "component": name,
        "present": bool(present),
        "score": int(score) if score is not None else None,
        "status": status,
        "required_for_paid": bool(required_for_paid),
        "notes": notes,
    }


def _extract_score(snapshot, *keys, default=None):
    if not isinstance(snapshot, dict):
        return default
    for key in keys:
        val = snapshot.get(key)
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
    return default


def build_operational_control_center_snapshot(
    project_name,
    dataset_df=None,
    plan_name="Founder Test Mode",
    operating_mode="Closed beta",
    support_level="Solo founder",
    user_projects=None,
    storage_snapshot=None,
    hardening_snapshot=None,
    release_success_snapshot=None,
    beta_launch_snapshot=None,
    paid_license_snapshot=None,
    field_validation_snapshot=None,
    edge_deployment_starter_snapshot=None,
    professional_report_snapshot=None,
    api_automation_snapshot=None,
    monetization_snapshot=None,
    trust_gate=None,
    reliability_v2=None,
    normality_result=None,
    edge_impulse_snapshot=None,
    edge_impulse_classifier_snapshot=None,
    open_notes="",
):
    """Build a single operator view across product, launch, storage and delivery readiness."""
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    user_projects = user_projects if isinstance(user_projects, pd.DataFrame) else pd.DataFrame()
    rows = int(len(dataset_df))
    cols = int(len(dataset_df.columns)) if rows >= 0 else 0
    labels = []
    if "Label" in dataset_df.columns:
        try:
            labels = [str(x) for x in dataset_df["Label"].dropna().unique().tolist()]
        except Exception:
            labels = []

    if not storage_snapshot:
        storage_snapshot = get_scalability_readiness_snapshot(dataset_df, plan_name)

    storage_score = _extract_score(storage_snapshot, "scalability_score", default=0) or 0
    hardening_score = _extract_score(hardening_snapshot, "product_readiness_score", "hardening_score", default=None)
    release_score = _extract_score(release_success_snapshot, "release_success_score", "success_score", default=None)
    beta_score = _extract_score(beta_launch_snapshot, "beta_readiness_score", default=None)
    api_score = _extract_score(api_automation_snapshot, "api_readiness_score", default=None)
    paid_score = _extract_score(paid_license_snapshot, "paid_export_score", "license_score", default=None)
    field_score = _extract_score(field_validation_snapshot, "field_validation_score", "validation_score", default=None)
    starter_score = _extract_score(edge_deployment_starter_snapshot, "starter_score", "deployment_starter_score", default=None)
    trust_score = _extract_score(trust_gate, "trust_score", "overall_score", default=None)
    reliability_score = _extract_score(reliability_v2, "trust_score_v2", "reliability_score", default=None)
    normality_score = _extract_score(normality_result, "baseline_quality_score", "normality_score", default=None)

    control_panels = [
        _component_status("Dataset loaded", rows > 0, 100 if rows > 0 else 0, True, f"{rows} rows / {cols} columns / {len(labels)} labels"),
        _component_status("Storage & scalability", True, storage_score, True, (storage_snapshot or {}).get("recommendation", "")),
        _component_status("Product hardening", bool(hardening_snapshot), hardening_score, True, "Safety scan, limitations and release checklist."),
        _component_status("Release Success Gate", bool(release_success_snapshot), release_score, True, "GO/NO-GO release checkpoint."),
        _component_status("Trust Center", bool(trust_gate), trust_score, True, "Customer-safe readiness and risk framing."),
        _component_status("Reliability Engine 2.0", bool(reliability_v2), reliability_score, True, "Per-class/sensor reliability evidence."),
        _component_status("Normality Engine", bool(normality_result), normality_score, False, "Normal vs abnormal baseline evidence."),
        _component_status("Professional Reports 2.0", bool(professional_report_snapshot), None, True, "Consultancy-style customer output."),
        _component_status("Beta Launch Kit", bool(beta_launch_snapshot), beta_score, False, "Invite, feedback and beta plan."),
        _component_status("Monetization Gate", bool(monetization_snapshot), None, False, "Package/plan positioning."),
        _component_status("Paid Export Gate", bool(paid_license_snapshot), paid_score, "Paid" in str(operating_mode), "License and locked-export readiness."),
        _component_status("Real Field Validation", bool(field_validation_snapshot), field_score, "Real-data" in str(operating_mode) or "Enterprise" in str(operating_mode), "Real-world validation evidence."),
        _component_status("Edge Deployment Starter", bool(edge_deployment_starter_snapshot), starter_score, "Paid" in str(operating_mode) or "Enterprise" in str(operating_mode), "Firmware/gateway handoff starter kit."),
        _component_status("API Automation", bool(api_automation_snapshot), api_score, False, "Private-beta API readiness."),
        _component_status("Edge Impulse Anomaly Export", bool(edge_impulse_snapshot), None, False, "Normal/anomaly workflow export."),
        _component_status("Edge Impulse Classifier Export", bool(edge_impulse_classifier_snapshot), None, False, "Supervised classifier workflow export."),
    ]

    must_fix = []
    if rows == 0:
        must_fix.append({"severity": "high", "message": "No active dataset loaded. Generate or upload a dataset before any beta/paid delivery."})
    if "Label" not in dataset_df.columns and rows > 0:
        must_fix.append({"severity": "high", "message": "Dataset has no Label column. Audit, classifier export and report evidence will be weak."})
    if rows > 0 and len(labels) < 2:
        must_fix.append({"severity": "high", "message": "Dataset has fewer than two labels/classes. It is not enough for a credible classifier/pilot package."})
    if storage_score < 60:
        must_fix.append({"severity": "high", "message": "Storage/scalability score is weak. Fix storage before growing dataset size or beta usage."})
    elif storage_score < 80:
        must_fix.append({"severity": "medium", "message": "Storage/scalability needs attention before multi-user/public use."})

    paid_like = str(operating_mode).lower() in {"paid pilot", "real-data pilot", "enterprise review"}
    if not release_success_snapshot:
        must_fix.append({"severity": "medium" if not paid_like else "high", "message": "Release Success Gate has not been run for the current state."})
    if not hardening_snapshot:
        must_fix.append({"severity": "medium", "message": "Product Hardening check has not been run. Do this before customer demos."})
    if paid_like and not professional_report_snapshot:
        must_fix.append({"severity": "high", "message": "Professional Report snapshot is missing. Paid pilots need a polished customer-facing report."})
    if paid_like and not paid_license_snapshot:
        must_fix.append({"severity": "medium", "message": "Paid Export Gate is missing. Paid delivery should define what is unlocked and licensed."})
    if str(operating_mode).lower() in {"real-data pilot", "enterprise review"} and not field_validation_snapshot:
        must_fix.append({"severity": "high", "message": "Real Field Validation Pack is missing for real-data/enterprise positioning."})
    if str(operating_mode).lower() in {"paid pilot", "real-data pilot", "enterprise review"} and not edge_deployment_starter_snapshot:
        must_fix.append({"severity": "medium", "message": "Edge Deployment Starter Kit is missing. Add it for stronger handoff/deployment value."})

    panel_scores = []
    for p in control_panels:
        if p.get("score") is not None:
            panel_scores.append(float(p["score"]))
        else:
            panel_scores.append(100.0 if p.get("present") else (45.0 if not p.get("required_for_paid") else 25.0))
    base_score = float(np.mean(panel_scores)) if panel_scores else 0.0
    for b in must_fix:
        if b.get("severity") == "high":
            base_score -= 13
        elif b.get("severity") == "medium":
            base_score -= 6
    if rows > 0 and len(labels) >= 2:
        base_score += 3
    operational_score = int(np.clip(base_score, 0, 100))

    high_count = sum(1 for b in must_fix if b.get("severity") == "high")
    medium_count = sum(1 for b in must_fix if b.get("severity") == "medium")
    if high_count == 0 and operational_score >= 82:
        decision = "GO"
        risk_level = "Low-Medium"
        operator_summary = "Controlled beta or selected paid-pilot delivery is acceptable if claims stay conservative and field validation is respected."
    elif high_count == 0 and operational_score >= 65:
        decision = "CONDITIONAL GO"
        risk_level = "Medium"
        operator_summary = "Usable for controlled demos/private beta. Fix medium issues before charging serious money or widening access."
    else:
        decision = "NO-GO"
        risk_level = "High"
        operator_summary = "Do not use for paid customer delivery yet. Resolve high-severity blockers first."

    next_actions = []
    if rows == 0:
        next_actions.append("Generate a golden demo or Auto Pilot dataset first.")
    if not hardening_snapshot:
        next_actions.append("Run Product Hardening and export the hardening bundle.")
    if not release_success_snapshot:
        next_actions.append("Run Release Success Gate for the current project state.")
    if not professional_report_snapshot:
        next_actions.append("Create a Professional Reports 2.0 bundle for customer-facing proof.")
    if not paid_license_snapshot and paid_like:
        next_actions.append("Create the Paid Export / License Gate bundle before paid delivery.")
    if not field_validation_snapshot and str(operating_mode).lower() in {"real-data pilot", "enterprise review"}:
        next_actions.append("Run Real Field Validation Pack with real uploaded data.")
    if not edge_deployment_starter_snapshot and paid_like:
        next_actions.append("Build Edge Deployment Starter Kit for deployment handoff.")
    if not next_actions:
        next_actions = [
            "Keep beta access limited and document every customer feedback point.",
            "Use the safe status line in demos and avoid production-ready claims without field validation.",
            "Back up storage and database before sharing with external testers.",
        ]

    daily_operator_checklist = [
        {"check": "Run smoke tests", "status": "Required", "why": "Catch broken imports/export functions before demos."},
        {"check": "Confirm dataset labels", "status": "Required", "why": "Bad labels create bad reliability claims."},
        {"check": "Run Success Gate", "status": "Required", "why": "Central GO/NO-GO for the current project."},
        {"check": "Export fresh report bundle", "status": "Recommended", "why": "Customer-facing evidence must match current data."},
        {"check": "Backup storage/database", "status": "Recommended", "why": "Protect local beta data and project history."},
        {"check": "Record known limitations", "status": "Required", "why": "Trustworthy positioning and safer sales."},
    ]

    launch_runbook = [
        {"title": "Before demo", "steps": [
            "Run Golden Demo or load the customer-specific project.",
            "Run Trust Center, Reliability 2.0, Product Hardening and Success Gate.",
            "Open Operational Control Center and confirm GO/CONDITIONAL GO.",
        ]},
        {"title": "During demo", "steps": [
            "Start from the customer pain, not the feature list.",
            "Show dataset, normal/abnormal baseline, reliability, hardware and report output.",
            "Use conservative wording: pilot-ready estimate, not production certification.",
        ]},
        {"title": "After demo", "steps": [
            "Export the relevant bundle and record feedback.",
            "Ask for one real WAV/CSV sample set if available.",
            "Decide whether the next step is beta test, paid pilot or field validation.",
        ]},
    ]

    safe_status_line = (
        f"EdgeTwin Studio operational status: {decision} for {operating_mode}. "
        "This supports pilot preparation and decision support; production deployment still requires field validation."
    )

    return {
        "version": "V32 Operational Control Center",
        "project_name": project_name,
        "created_at": _now(),
        "plan": plan_name,
        "operating_mode": operating_mode,
        "support_level": support_level,
        "operational_score": operational_score,
        "decision": decision,
        "risk_level": risk_level,
        "operator_summary": operator_summary,
        "safe_status_line": safe_status_line,
        "dataset": {
            "rows": rows,
            "cols": cols,
            "hash": dataset_hash(dataset_df) if isinstance(dataset_df, pd.DataFrame) else "empty",
            "has_label": "Label" in dataset_df.columns,
            "labels": labels,
        },
        "storage_snapshot": storage_snapshot,
        "control_panels": control_panels,
        "must_fix_blockers": must_fix,
        "next_actions": next_actions,
        "daily_operator_checklist": daily_operator_checklist,
        "launch_runbook": launch_runbook,
        "project_index_rows": int(len(user_projects)),
        "open_operator_notes": str(open_notes or ""),
        "disclaimer": OPERATIONAL_CONTROL_DISCLAIMER,
    }


def generate_operational_control_center_pdf(project_name, snapshot):
    snapshot = snapshot or {}
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Operational Control Center", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("V32 - Operator view for beta, paid pilot and product control"), ln=True, align="C")
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)

    safe_pdf_cell(pdf, "Operational Decision", 8, True)
    safe_pdf_cell(pdf, f"Score: {snapshot.get('operational_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {snapshot.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Risk level: {snapshot.get('risk_level', 'Unknown')}")
    safe_pdf_cell(pdf, f"Operating mode: {snapshot.get('operating_mode', 'Unknown')}")
    safe_pdf_multicell(pdf, snapshot.get("operator_summary", ""))
    pdf.ln(4)

    ds = snapshot.get("dataset", {}) or {}
    safe_pdf_cell(pdf, "Dataset", 8, True)
    safe_pdf_cell(pdf, f"Rows: {ds.get('rows', 0)}")
    safe_pdf_cell(pdf, f"Columns: {ds.get('cols', 0)}")
    safe_pdf_cell(pdf, f"Has label: {ds.get('has_label', False)}")
    safe_pdf_multicell(pdf, f"Labels: {', '.join(ds.get('labels', []))}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Control Panels", 8, True)
    for panel in snapshot.get("control_panels", []):
        score = panel.get("score")
        score_txt = "n/a" if score is None else f"{score}%"
        safe_pdf_multicell(pdf, f"- {panel.get('component')}: {panel.get('status')} ({score_txt}) - {panel.get('notes', '')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Must-Fix Blockers", 8, True)
    blockers = snapshot.get("must_fix_blockers", [])
    if not blockers:
        safe_pdf_multicell(pdf, "No must-fix blockers detected for the selected operating mode.")
    for item in blockers:
        safe_pdf_multicell(pdf, f"[{item.get('severity', 'info').upper()}] {item.get('message', '')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Next Actions", 8, True)
    for item in snapshot.get("next_actions", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Safe Status Line", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("safe_status_line", ""))
    pdf.ln(4)

    safe_pdf_cell(pdf, "Disclaimer", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", OPERATIONAL_CONTROL_DISCLAIMER))
    return safe_pdf_output(pdf)


def create_operational_control_center_bundle(project_name, snapshot, dataset_df=None, projects_df=None):
    snapshot = snapshot or {}
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    projects_df = projects_df if isinstance(projects_df, pd.DataFrame) else pd.DataFrame()
    pdf_bytes = generate_operational_control_center_pdf(project_name, snapshot)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("operational_control_report_v32.pdf", pdf_bytes)
        zf.writestr("operational_control_snapshot.json", safe_json_dumps(snapshot, indent=2))
        if snapshot.get("control_panels"):
            zf.writestr("control_panels.csv", pd.DataFrame(snapshot.get("control_panels", [])).to_csv(index=False))
        if snapshot.get("must_fix_blockers"):
            zf.writestr("must_fix_blockers.csv", pd.DataFrame(snapshot.get("must_fix_blockers", [])).to_csv(index=False))
        if snapshot.get("daily_operator_checklist"):
            zf.writestr("daily_operator_checklist.csv", pd.DataFrame(snapshot.get("daily_operator_checklist", [])).to_csv(index=False))
        if len(projects_df) > 0:
            zf.writestr("project_index.csv", projects_df.to_csv(index=False))
        if len(dataset_df) > 0:
            zf.writestr("dataset_snapshot.csv", dataset_df.head(5000).to_csv(index=False))
        zf.writestr("README.md", f"""# EdgeTwin Studio V32 Operational Control Bundle

Project: {project_name}
Decision: {snapshot.get('decision', 'Unknown')}
Operational score: {snapshot.get('operational_score', 0)}%
Risk level: {snapshot.get('risk_level', 'Unknown')}
Operating mode: {snapshot.get('operating_mode', 'Unknown')}

## Safe status line
{snapshot.get('safe_status_line', '')}

## Important
{snapshot.get('disclaimer', OPERATIONAL_CONTROL_DISCLAIMER)}

Use this bundle as the operator checkpoint before demos, closed beta, paid pilot delivery or enterprise review.
""")
    return zip_buf.getvalue()


# ============================================================
# V33 PRODUCTION SAFETY & ERROR OBSERVATORY
# ============================================================

OBSERVABILITY_DISCLAIMER = (
    "This observability layer is an operational safety aid. It does not replace external QA, security review, "
    "field validation or production monitoring. Customer-facing claims must remain limited to pilot preparation unless validated in the field."
)


def ensure_observability_table():
    """Create a lightweight local event table for beta operations."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS observability_events (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                user_id TEXT,
                project_name TEXT,
                event_type TEXT,
                severity TEXT,
                flow TEXT,
                message TEXT,
                context TEXT
            )
            """
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def log_observability_event(event_type="manual_note", severity="info", message="", flow="general", project_name="", user_id=None, context=None):
    ensure_observability_table()
    event = {
        "id": str(uuid.uuid4()),
        "created_at": _now(),
        "user_id": str(user_id or ""),
        "project_name": str(project_name or ""),
        "event_type": str(event_type or "manual_note"),
        "severity": str(severity or "info").lower(),
        "flow": str(flow or "general"),
        "message": str(message or ""),
        "context": _json_safe(context or {}),
    }
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO observability_events
            (id, created_at, user_id, project_name, event_type, severity, flow, message, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["id"], event["created_at"], event["user_id"], event["project_name"],
                event["event_type"], event["severity"], event["flow"], event["message"],
                safe_json_dumps(event["context"], indent=0),
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        # Keep UI alive even if event logging fails.
        event["logging_error"] = True
    return event


def get_recent_observability_events(limit=50, project_name=None, user_id=None):
    ensure_observability_table()
    try:
        conn = sqlite3.connect(DB_NAME)
        query = "SELECT id, created_at, user_id, project_name, event_type, severity, flow, message, context FROM observability_events"
        params = []
        where = []
        if project_name:
            where.append("project_name = ?")
            params.append(str(project_name))
        if user_id:
            where.append("user_id = ?")
            params.append(str(user_id))
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        if len(df) == 0:
            return []
        rows = df.to_dict(orient="records")
        for row in rows:
            try:
                row["context"] = json.loads(row.get("context") or "{}")
            except Exception:
                row["context"] = {}
        return _json_safe(rows)
    except Exception:
        return []


def _snapshot_score(snapshot, *keys, default=None):
    """Robust score extractor used by Release Gate and Error Observatory.

    Supports both patterns used across versions:
    - nested path: _snapshot_score(obj, "summary", "success_score", default=0)
    - alternative keys: _snapshot_score(obj, "trust_score", "score", "overall_score")

    The previous V33 observability helper accidentally overwrote the V26.4 helper
    and removed the `default=` argument. This unified version keeps both flows safe.
    """
    if not isinstance(snapshot, dict):
        return default

    def to_int(value):
        try:
            if value is None:
                return None
            return int(float(value))
        except Exception:
            return None

    # 1) Try full nested path first.
    cur = snapshot
    nested_ok = bool(keys)
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur.get(key)
        else:
            nested_ok = False
            break
    if nested_ok:
        val = to_int(cur)
        if val is not None:
            return val

    # 2) Treat keys as alternative top-level score names.
    for key in keys:
        if key in snapshot:
            val = to_int(snapshot.get(key))
            if val is not None:
                return val

    # 3) Search common nested containers, especially summary blocks.
    for container_key in ["summary", "scores", "source_scores", "readiness", "result"]:
        container = snapshot.get(container_key)
        if isinstance(container, dict):
            for key in keys:
                if key in container:
                    val = to_int(container.get(key))
                    if val is not None:
                        return val

    return default


def _component_status(name, present=False, score=None, required=False, note=""):
    if score is not None:
        status = "pass" if score >= 75 else "warn" if score >= 55 else "fail"
    else:
        status = "pass" if present else "fail" if required else "warn"
    return {
        "component": name,
        "status": status,
        "score": score,
        "required": bool(required),
        "notes": note,
    }


def build_production_safety_observatory_snapshot(
    project_name,
    dataset_df=None,
    release_mode="Closed beta",
    selected_plan="Founder Test Mode",
    user_id=None,
    operational_snapshot=None,
    hardening_snapshot=None,
    release_success_snapshot=None,
    trust_gate=None,
    reliability_v2=None,
    professional_report_snapshot=None,
    field_validation_snapshot=None,
    edge_deployment_starter_snapshot=None,
    monetization_snapshot=None,
    api_automation_snapshot=None,
    edge_impulse_snapshot=None,
    edge_impulse_classifier_snapshot=None,
    bundle_presence=None,
):
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    bundle_presence = bundle_presence or {}
    rows = int(len(dataset_df))
    cols = int(len(dataset_df.columns))
    has_label = "Label" in dataset_df.columns
    labels = [str(x) for x in dataset_df["Label"].dropna().unique().tolist()] if has_label else []

    recent_events = get_recent_observability_events(limit=40, project_name=project_name, user_id=user_id)
    severity_penalty = 0
    for ev in recent_events:
        sev = str(ev.get("severity", "info")).lower()
        if sev in {"critical"}:
            severity_penalty += 25
        elif sev in {"error", "high"}:
            severity_penalty += 15
        elif sev in {"warning", "medium"}:
            severity_penalty += 7

    checks = []
    checks.append(_component_status("Dataset loaded", present=rows > 0 and cols > 0, required=True, note=f"{rows} rows / {cols} columns"))
    checks.append(_component_status("Label column", present=has_label, required=True, note=f"{len(labels)} labels detected"))
    checks.append(_component_status("Minimum class coverage", present=(len(labels) >= 2), required=True, note="At least two classes are needed for supervised pilot checks."))

    hardening_score = _snapshot_score(hardening_snapshot, "hardening_score", "product_readiness_score", "readiness_score")
    release_score = _snapshot_score(release_success_snapshot, "release_success_score", "success_score")
    operational_score = _snapshot_score(operational_snapshot, "operational_score")
    trust_score = _snapshot_score(trust_gate, "trust_score", "score", "overall_score")
    reliability_score = _snapshot_score(reliability_v2, "trust_score_v2", "reliability_score", "overall_score")
    report_score = _snapshot_score(professional_report_snapshot, "report_score", "professional_report_score", "score")
    field_score = _snapshot_score(field_validation_snapshot, "field_validation_score", "validation_score", "score")
    deployment_score = _snapshot_score(edge_deployment_starter_snapshot, "edge_starter_score", "deployment_starter_score", "score")
    api_score = _snapshot_score(api_automation_snapshot, "api_readiness_score", "score")

    checks.extend([
        _component_status("Product hardening", present=hardening_snapshot is not None, score=hardening_score, required=True, note="Hardening should be run before any external beta."),
        _component_status("Release Success Gate", present=release_success_snapshot is not None, score=release_score, required=True, note="Release gate should be run before customer delivery."),
        _component_status("Operational Control Center", present=operational_snapshot is not None, score=operational_score, required=True, note="Operator check should be current."),
        _component_status("Trust / Reliability evidence", present=(trust_gate is not None or reliability_v2 is not None), score=(reliability_score or trust_score), required=True, note="Reliability evidence must be visible in reports."),
        _component_status("Professional report", present=professional_report_snapshot is not None or bundle_presence.get("professional_report_bundle", False), score=report_score, required=True, note="Paid pilots need a professional report bundle."),
        _component_status("Field validation", present=field_validation_snapshot is not None, score=field_score, required=(release_mode in ["Real-data pilot", "Enterprise review"]), note="Required for real-data/enterprise claims."),
        _component_status("Edge deployment starter", present=edge_deployment_starter_snapshot is not None, score=deployment_score, required=(release_mode in ["Paid pilot", "Real-data pilot", "Enterprise review"]), note="Needed when advising physical deployment."),
        _component_status("Monetization/package policy", present=monetization_snapshot is not None, required=(release_mode in ["Paid pilot", "Real-data pilot", "Enterprise review"]), note="Paid delivery needs clear package boundaries."),
        _component_status("API readiness", present=api_automation_snapshot is not None, score=api_score, required=False, note="Only required if selling API/private integration."),
        _component_status("Edge Impulse anomaly export", present=edge_impulse_snapshot is not None or bundle_presence.get("edge_impulse_bundle", False), required=False, note="Useful for anomaly/K-means workflow."),
        _component_status("Edge Impulse classifier export", present=edge_impulse_classifier_snapshot is not None or bundle_presence.get("edge_impulse_classifier_bundle", False), required=False, note="Useful for supervised workflow."),
    ])

    must_fix = []
    for chk in checks:
        if chk["required"] and chk["status"] == "fail":
            must_fix.append({"severity": "critical", "message": f"Missing required check: {chk['component']}. {chk.get('notes', '')}"})
        elif chk["required"] and chk["status"] == "warn":
            must_fix.append({"severity": "warning", "message": f"Weak required check: {chk['component']}. {chk.get('notes', '')}"})

    for ev in recent_events[:10]:
        sev = str(ev.get("severity", "info")).lower()
        if sev in {"critical", "error", "high"}:
            must_fix.append({"severity": sev, "message": f"Recent {sev} event in {ev.get('flow', 'general')}: {ev.get('message', '')}"})

    # Score: components + recent incident penalty.
    base_score = 100
    for chk in checks:
        if chk["required"] and chk["status"] == "fail":
            base_score -= 22
        elif chk["required"] and chk["status"] == "warn":
            base_score -= 10
        elif (not chk["required"]) and chk["status"] == "fail":
            base_score -= 3
    base_score -= min(45, severity_penalty)
    safety_score = int(np.clip(base_score, 0, 100))

    if safety_score >= 82 and not [m for m in must_fix if str(m.get("severity")).lower() in {"critical", "error", "high"}]:
        decision = "GO"
        incident_risk = "Low-Medium"
        summary = "Operational safety looks acceptable for the selected release mode. Keep claims limited to pilot preparation and keep monitoring incidents."
    elif safety_score >= 60:
        decision = "CONDITIONAL GO"
        incident_risk = "Medium"
        summary = "Usable with caution. Fix the listed items before paid/enterprise delivery, and avoid production-readiness claims."
    else:
        decision = "NO-GO"
        incident_risk = "High"
        summary = "Do not show or sell this flow yet. Resolve critical gaps, rerun hardening/release checks, and verify exports."

    next_actions = []
    if rows == 0:
        next_actions.append("Generate or load a dataset before running customer-facing checks.")
    if hardening_snapshot is None:
        next_actions.append("Run Product Hardening and fix blockers.")
    if release_success_snapshot is None:
        next_actions.append("Run Release Success Gate after generating the latest outputs.")
    if operational_snapshot is None:
        next_actions.append("Run Operational Control Center after readiness checks.")
    if professional_report_snapshot is None and not bundle_presence.get("professional_report_bundle", False):
        next_actions.append("Generate Reports 2.0 bundle for paid pilot presentation.")
    if release_mode in ["Real-data pilot", "Enterprise review"] and field_validation_snapshot is None:
        next_actions.append("Run Real Field Validation before mentioning real-data confidence.")
    if not next_actions:
        next_actions = ["Export the safety bundle, save the project, and use this snapshot as the release checkpoint."]

    return {
        "version": "V33 Production Safety & Error Observatory",
        "project_name": project_name,
        "created_at": _now(),
        "release_mode": release_mode,
        "selected_plan": selected_plan,
        "safety_score": safety_score,
        "decision": decision,
        "incident_risk": incident_risk,
        "operator_summary": summary,
        "dataset": {"rows": rows, "cols": cols, "has_label": has_label, "labels": labels, "hash": dataset_hash(dataset_df) if rows else "empty"},
        "recent_event_count": len(recent_events),
        "recent_events": recent_events,
        "safety_checks": checks,
        "must_fix": must_fix,
        "next_actions": next_actions,
        "bundle_presence": bundle_presence,
        "safe_customer_line": "EdgeTwin Studio can prepare a pilot package and expose risks, but production deployment still requires real field validation.",
        "claims_to_avoid": [
            "Do not claim production-ready reliability from synthetic-only data.",
            "Do not claim certified safety performance without field validation and external QA.",
            "Do not guarantee anomaly thresholds before testing on real deployment data.",
        ],
        "disclaimer": OBSERVABILITY_DISCLAIMER,
    }


def generate_production_safety_observatory_pdf(project_name, snapshot):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Studio Production Safety Observatory", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)
    safe_pdf_cell(pdf, "Release Decision", 8, True)
    safe_pdf_cell(pdf, f"Safety score: {snapshot.get('safety_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {snapshot.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Incident risk: {snapshot.get('incident_risk', 'Unknown')}")
    safe_pdf_multicell(pdf, snapshot.get("operator_summary", ""))
    pdf.ln(4)
    ds = snapshot.get("dataset", {}) or {}
    safe_pdf_cell(pdf, "Dataset", 8, True)
    safe_pdf_cell(pdf, f"Rows: {ds.get('rows', 0)}")
    safe_pdf_cell(pdf, f"Columns: {ds.get('cols', 0)}")
    safe_pdf_cell(pdf, f"Has label column: {ds.get('has_label', False)}")
    safe_pdf_multicell(pdf, f"Labels: {', '.join(ds.get('labels', []))}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Safety Checks", 8, True)
    for chk in snapshot.get("safety_checks", []):
        score = chk.get("score")
        score_txt = "n/a" if score is None else f"{score}%"
        safe_pdf_multicell(pdf, f"- {chk.get('component')}: {chk.get('status')} ({score_txt}) - {chk.get('notes', '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Must-Fix Items", 8, True)
    items = snapshot.get("must_fix", [])
    if not items:
        safe_pdf_multicell(pdf, "No critical must-fix items detected for the selected mode.")
    for item in items:
        safe_pdf_multicell(pdf, f"[{item.get('severity', 'info').upper()}] {item.get('message', '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Next Actions", 8, True)
    for item in snapshot.get("next_actions", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Safe Customer Line", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("safe_customer_line", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Disclaimer", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", OBSERVABILITY_DISCLAIMER))
    return safe_pdf_output(pdf)


def create_production_safety_observatory_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    pdf_bytes = generate_production_safety_observatory_pdf(project_name, snapshot)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("production_safety_report_v33.pdf", pdf_bytes)
        zf.writestr("production_safety_snapshot.json", safe_json_dumps(snapshot, indent=2))
        if snapshot.get("safety_checks"):
            zf.writestr("safety_checks.csv", pd.DataFrame(snapshot.get("safety_checks", [])).to_csv(index=False))
        if snapshot.get("must_fix"):
            zf.writestr("must_fix_items.csv", pd.DataFrame(snapshot.get("must_fix", [])).to_csv(index=False))
        if snapshot.get("recent_events"):
            zf.writestr("recent_observability_events.csv", pd.DataFrame(snapshot.get("recent_events", [])).to_csv(index=False))
        if len(dataset_df) > 0:
            zf.writestr("dataset_snapshot.csv", dataset_df.head(5000).to_csv(index=False))
        zf.writestr("README.md", f"""# EdgeTwin Studio V33 Production Safety Bundle

Project: {project_name}
Decision: {snapshot.get('decision', 'Unknown')}
Safety score: {snapshot.get('safety_score', 0)}%
Incident risk: {snapshot.get('incident_risk', 'Unknown')}

## Operator summary
{snapshot.get('operator_summary', '')}

## Safe customer line
{snapshot.get('safe_customer_line', '')}

## Important
{snapshot.get('disclaimer', OBSERVABILITY_DISCLAIMER)}

Use this bundle as the final operational/error checkpoint before demos, beta users, paid pilots or enterprise review.
""")
    return zip_buf.getvalue()


# ============================================================
# V34 CUSTOMER ASSURANCE & DATA GOVERNANCE CENTER
# ============================================================

GOVERNANCE_DISCLAIMER = (
    "EdgeTwin Studio governance outputs are product-readiness and customer-assurance aids. "
    "They are not legal advice. Uploaded customer data, privacy obligations, security controls "
    "and production deployment approval must be reviewed by the responsible organization."
)

GOVERNANCE_RETENTION_OPTIONS = [
    "session_only",
    "delete_after_export",
    "retain_30_days_private_beta",
    "customer_controlled_retention",
]

DATA_CATEGORY_HINTS = {
    "signal_features_only": "Feature-level numeric dataset without raw personal content.",
    "raw_sensor_signal": "Raw WAV/CSV sensor data may contain operational or environmental information.",
    "customer_asset_metadata": "Project names, locations, asset labels or deployment notes may reveal business context.",
    "potential_audio_privacy": "Audio uploads can accidentally contain voices or identifiable background information.",
    "industrial_confidential": "Machine condition, failure patterns and asset context can be commercially sensitive.",
}


def get_governance_retention_options():
    return list(GOVERNANCE_RETENTION_OPTIONS)


def _governance_score_penalty(severity):
    return {"critical": 35, "high": 25, "medium": 12, "low": 5, "info": 0}.get(str(severity).lower(), 0)


def _contains_audio_sensor_hint(dataset_df=None, project_name="", manifest=None):
    text = " ".join([str(project_name or ""), safe_json_dumps(manifest or {}, indent=None) if 'safe_json_dumps' in globals() else str(manifest or {})]).lower()
    if "audio" in text or "acoustic" in text or "wav" in text:
        return True
    if isinstance(dataset_df, pd.DataFrame):
        cols = " ".join([str(c).lower() for c in dataset_df.columns])
        if "audio" in cols or "zcr" in cols or "spectral" in cols:
            return True
    return False


def classify_dataset_governance_risk(dataset_df=None, project_name="", manifest=None, has_real_uploads=False):
    rows = int(len(dataset_df)) if isinstance(dataset_df, pd.DataFrame) else 0
    cols = int(len(dataset_df.columns)) if isinstance(dataset_df, pd.DataFrame) else 0
    has_audio = _contains_audio_sensor_hint(dataset_df, project_name, manifest)
    has_labels = isinstance(dataset_df, pd.DataFrame) and "Label" in dataset_df.columns
    categories = []
    issues = []

    if rows == 0:
        categories.append({"category": "no_active_dataset", "risk": "Low", "reason": "No active dataset is loaded."})
        issues.append({"severity": "info", "message": "No dataset loaded. Governance risk is low, but customer assurance evidence is incomplete."})
    else:
        categories.append({"category": "signal_features_only", "risk": "Low-Medium", "reason": DATA_CATEGORY_HINTS["signal_features_only"]})
        categories.append({"category": "industrial_confidential", "risk": "Medium", "reason": DATA_CATEGORY_HINTS["industrial_confidential"]})
        if has_labels:
            categories.append({"category": "label_taxonomy", "risk": "Medium", "reason": "Labels/classes may expose failure modes or security events."})
        else:
            issues.append({"severity": "medium", "message": "Dataset has no Label column, so assurance reports cannot explain class-level risk."})

    if has_real_uploads:
        categories.append({"category": "raw_sensor_signal", "risk": "Medium-High", "reason": DATA_CATEGORY_HINTS["raw_sensor_signal"]})
        issues.append({"severity": "medium", "message": "Real customer uploads detected or selected. Retention and deletion policy must be explicit."})
    if has_audio:
        categories.append({"category": "potential_audio_privacy", "risk": "High", "reason": DATA_CATEGORY_HINTS["potential_audio_privacy"]})
        issues.append({"severity": "high", "message": "Audio/acoustic use case detected. Warn customers not to upload recordings with voices or personal information."})
    if rows > 25000:
        issues.append({"severity": "medium", "message": "Large dataset detected. Use file/object storage and avoid keeping unnecessary raw uploads."})
    if rows > 0 and rows < 50:
        issues.append({"severity": "low", "message": "Very small dataset. Governance risk is low, but evidence quality may be weak."})
    if not issues:
        issues.append({"severity": "info", "message": "No major governance blockers detected for private beta use."})

    highest = "Low"
    order = {"Low": 1, "Low-Medium": 2, "Medium": 3, "Medium-High": 4, "High": 5}
    for c in categories:
        if order.get(c.get("risk", "Low"), 1) > order.get(highest, 1):
            highest = c.get("risk", highest)
    return {"rows": rows, "cols": cols, "has_audio_hint": has_audio, "has_labels": bool(has_labels), "risk_level": highest, "categories": categories, "issues": issues}


def create_customer_assurance_snapshot(
    project_name,
    dataset_df=None,
    selected_plan="Founder Test Mode",
    release_mode="Private beta",
    retention_policy="delete_after_export",
    uses_customer_uploads=False,
    allows_model_reuse=False,
    customer_data_used_for_training=False,
    manifest=None,
    trust_snapshot=None,
    reliability_snapshot=None,
    hardening_snapshot=None,
    observability_snapshot=None,
    scalability_snapshot=None,
    field_validation_snapshot=None,
):
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    manifest = manifest or {}
    governance = classify_dataset_governance_risk(dataset_df, project_name, manifest, has_real_uploads=uses_customer_uploads)
    controls = []
    issues = list(governance.get("issues", []))

    def add_control(name, status, evidence, severity="info"):
        controls.append({"control": name, "status": status, "evidence": evidence, "severity": severity})
        if status in ["Missing", "Weak", "Needs review"] and severity in ["critical", "high", "medium"]:
            issues.append({"severity": severity, "message": f"{name}: {evidence}"})

    add_control(
        "Customer data ownership statement",
        "Present",
        "Reports state that customer data remains customer-owned and field validation is required.",
        "info",
    )
    if retention_policy in GOVERNANCE_RETENTION_OPTIONS:
        add_control("Retention policy", "Present", f"Selected retention policy: {retention_policy}.", "info")
    else:
        add_control("Retention policy", "Missing", "No valid retention policy selected.", "high")

    if uses_customer_uploads:
        add_control("Upload consent notice", "Needs review", "Customer upload flow must show explicit consent/retention notice before paid beta.", "medium")
    else:
        add_control("Upload consent notice", "Present", "No real upload selected for this snapshot; synthetic/demo flow is lower risk.", "info")

    if customer_data_used_for_training or allows_model_reuse:
        add_control("Customer data reuse", "Needs review", "Customer data reuse/model improvement must be opt-in and written clearly.", "high")
    else:
        add_control("Customer data reuse", "Present", "Default position: do not reuse customer data for shared models without explicit opt-in.", "info")

    if trust_snapshot:
        add_control("Trust/Readiness evidence", "Present", "Trust Center snapshot is available.", "info")
    else:
        add_control("Trust/Readiness evidence", "Weak", "Run Trust Center before sharing assurance bundle.", "medium")

    if reliability_snapshot:
        add_control("Reliability evidence", "Present", "Reliability/score evidence is available.", "info")
    else:
        add_control("Reliability evidence", "Weak", "Run Reliability Engine 2.0 or Reports 2.0 before paid pilot.", "medium")

    if hardening_snapshot:
        add_control("Product hardening evidence", "Present", "Hardening snapshot is available.", "info")
    else:
        add_control("Product hardening evidence", "Weak", "Run Product Hardening before closed beta or paid pilot.", "medium")

    if observability_snapshot:
        add_control("Error/incident observability", "Present", "Production Safety Observatory snapshot is available.", "info")
    else:
        add_control("Error/incident observability", "Weak", "Run Error Observatory before sending to customer testers.", "medium")

    if scalability_snapshot:
        add_control("Storage/scalability evidence", "Present", "Storage/scalability snapshot is available.", "info")
    else:
        add_control("Storage/scalability evidence", "Needs review", "Run Storage/Scale check before larger customer datasets.", "low")

    if uses_customer_uploads and not field_validation_snapshot:
        add_control("Field validation evidence", "Weak", "Real-data claims should be backed by Field Validation snapshot.", "medium")
    elif field_validation_snapshot:
        add_control("Field validation evidence", "Present", "Field Validation Pack snapshot is available.", "info")

    score = 100
    for item in issues:
        score -= _governance_score_penalty(item.get("severity"))
    # cap score if risky choices are selected
    if allows_model_reuse or customer_data_used_for_training:
        score = min(score, 70)
    if uses_customer_uploads and retention_policy == "session_only":
        score += 5
    score = int(np.clip(score, 0, 100))

    if score >= 85:
        decision = "CUSTOMER-ASSURANCE READY"
        safe_to_share = True
    elif score >= 68:
        decision = "CONDITIONAL - REVIEW BEFORE CUSTOMER"
        safe_to_share = False
    else:
        decision = "NO-GO FOR CUSTOMER DATA"
        safe_to_share = False

    must_fix = [i for i in issues if i.get("severity") in ["critical", "high"]]
    next_actions = []
    if not trust_snapshot:
        next_actions.append("Run Trust Center and attach the readiness snapshot.")
    if not reliability_snapshot:
        next_actions.append("Run Reliability Engine 2.0 or Reports 2.0 before quoting reliability claims.")
    if uses_customer_uploads:
        next_actions.append("Show upload consent, retention and deletion wording before customer uploads real WAV/CSV files.")
    if allows_model_reuse or customer_data_used_for_training:
        next_actions.append("Make data reuse opt-in only; never reuse customer data silently.")
    if not observability_snapshot:
        next_actions.append("Run Error Observatory to show how incidents will be tracked.")
    if not next_actions:
        next_actions.append("Export this assurance bundle and attach it to the private beta/pilot handoff.")

    return {
        "version": "V34 Customer Assurance & Data Governance Center",
        "project_name": project_name,
        "created_at": _now(),
        "selected_plan": selected_plan,
        "release_mode": release_mode,
        "retention_policy": retention_policy,
        "uses_customer_uploads": bool(uses_customer_uploads),
        "allows_model_reuse": bool(allows_model_reuse),
        "customer_data_used_for_training": bool(customer_data_used_for_training),
        "governance_score": score,
        "decision": decision,
        "safe_to_share_with_customer": safe_to_share,
        "dataset_governance": governance,
        "controls": controls,
        "issues": issues,
        "must_fix": must_fix,
        "next_actions": next_actions,
        "customer_assurance_line": "Customer data is treated as confidential pilot input; EdgeTwin outputs are pilot-preparation evidence, not production certification.",
        "customer_upload_notice": "Only upload data you are authorized to process. Avoid recordings with voices, personal data or unrelated confidential information unless your organization has approved it.",
        "claims_allowed": [
            "EdgeTwin can prepare pilot datasets, reliability evidence, hardware advice and export bundles.",
            "EdgeTwin can help identify dataset, label, sensor and deployment risks before field validation.",
            "EdgeTwin can support Edge Impulse/TinyML preparation with normality and classifier exports.",
        ],
        "claims_to_avoid": [
            "Do not claim legal/compliance certification.",
            "Do not claim production readiness without real field validation.",
            "Do not imply customer data will be reused for model improvement unless explicit opt-in exists.",
            "Do not accept sensitive audio or personal data without customer approval and retention controls.",
        ],
        "disclaimer": GOVERNANCE_DISCLAIMER,
    }


def generate_customer_assurance_pdf(project_name, snapshot):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Studio Customer Assurance", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)
    safe_pdf_cell(pdf, "Assurance Decision", 8, True)
    safe_pdf_cell(pdf, f"Governance score: {snapshot.get('governance_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {snapshot.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Retention policy: {snapshot.get('retention_policy', 'Unknown')}")
    safe_pdf_multicell(pdf, snapshot.get("customer_assurance_line", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Dataset Governance", 8, True)
    gov = snapshot.get("dataset_governance", {}) or {}
    safe_pdf_cell(pdf, f"Rows: {gov.get('rows', 0)}")
    safe_pdf_cell(pdf, f"Columns: {gov.get('cols', 0)}")
    safe_pdf_cell(pdf, f"Risk level: {gov.get('risk_level', 'Unknown')}")
    for cat in gov.get("categories", []):
        safe_pdf_multicell(pdf, f"- {cat.get('category')}: {cat.get('risk')} - {cat.get('reason', '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Controls", 8, True)
    for ctrl in snapshot.get("controls", []):
        safe_pdf_multicell(pdf, f"- {ctrl.get('control')}: {ctrl.get('status')} - {ctrl.get('evidence', '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Must-Fix Items", 8, True)
    items = snapshot.get("must_fix", [])
    if not items:
        safe_pdf_multicell(pdf, "No high-severity customer assurance blockers detected.")
    for item in items:
        safe_pdf_multicell(pdf, f"[{item.get('severity', 'info').upper()}] {item.get('message', '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Customer Upload Notice", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("customer_upload_notice", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Allowed Claims", 8, True)
    for item in snapshot.get("claims_allowed", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Claims to Avoid", 8, True)
    for item in snapshot.get("claims_to_avoid", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Disclaimer", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", GOVERNANCE_DISCLAIMER))
    return safe_pdf_output(pdf)


def create_customer_assurance_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    pdf_bytes = generate_customer_assurance_pdf(project_name, snapshot)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("customer_assurance_report_v34.pdf", pdf_bytes)
        zf.writestr("customer_assurance_snapshot.json", safe_json_dumps(snapshot, indent=2))
        if snapshot.get("controls"):
            zf.writestr("assurance_controls.csv", pd.DataFrame(snapshot.get("controls", [])).to_csv(index=False))
        if snapshot.get("issues"):
            zf.writestr("governance_issues.csv", pd.DataFrame(snapshot.get("issues", [])).to_csv(index=False))
        gov = snapshot.get("dataset_governance", {}) or {}
        if gov.get("categories"):
            zf.writestr("data_categories.csv", pd.DataFrame(gov.get("categories", [])).to_csv(index=False))
        if len(dataset_df) > 0:
            zf.writestr("dataset_snapshot_head.csv", dataset_df.head(1000).to_csv(index=False))
        zf.writestr("CUSTOMER_UPLOAD_NOTICE.md", f"""# Customer Upload Notice

{snapshot.get('customer_upload_notice', '')}

## Data reuse
Customer data is not reused for shared model improvement unless explicit opt-in is configured and documented.

## Disclaimer
{snapshot.get('disclaimer', GOVERNANCE_DISCLAIMER)}
""")
        zf.writestr("README.md", f"""# EdgeTwin Studio V34 Customer Assurance Bundle

Project: {project_name}
Decision: {snapshot.get('decision', 'Unknown')}
Governance score: {snapshot.get('governance_score', 0)}%
Retention policy: {snapshot.get('retention_policy', 'Unknown')}

## Customer assurance line
{snapshot.get('customer_assurance_line', '')}

## Next actions
""" + "\n".join([f"- {x}" for x in snapshot.get("next_actions", [])]) + f"""

## Important
{snapshot.get('disclaimer', GOVERNANCE_DISCLAIMER)}
""")
    return zip_buf.getvalue()


# ============================================================
# V35 CUSTOMER ONBOARDING & GUIDED SUCCESS FLOW
# ============================================================

GUIDED_SUCCESS_DISCLAIMER = (
    "Guided Success is a customer onboarding and pilot-delivery aid. It does not certify production safety, "
    "legal compliance or deployment fitness without field validation and customer approval."
)

ONBOARDING_CUSTOMER_SEGMENTS = [
    "Predictive Maintenance Team",
    "Security / Tamper Integrator",
    "Remote Asset / Forestry Operator",
    "Edge AI Consultant / System Integrator",
    "Small Industrial Business",
    "Research / Education / Lab",
]

ONBOARDING_DATA_STATUS = [
    "No data yet",
    "Some synthetic/demo data",
    "Small real WAV/CSV sample set",
    "Enough real baseline data",
    "Real normal + abnormal data available",
]

ONBOARDING_TECH_LEVELS = ["Non-technical", "Technical operator", "Engineer", "AI/ML specialist"]
ONBOARDING_OUTCOMES = [
    "Understand if pilot is worth starting",
    "Generate pilot bundle",
    "Prepare Edge Impulse export",
    "Create management-ready report",
    "Plan real field validation",
    "Prepare paid pilot delivery",
]


def get_onboarding_customer_segments():
    return list(ONBOARDING_CUSTOMER_SEGMENTS)


def get_onboarding_data_status_options():
    return list(ONBOARDING_DATA_STATUS)


def get_onboarding_technical_levels():
    return list(ONBOARDING_TECH_LEVELS)


def get_onboarding_outcomes():
    return list(ONBOARDING_OUTCOMES)


def _has_snapshot(s):
    return bool(s) and isinstance(s, dict)


def _readiness_value(snapshot, keys=None, default=0):
    if not isinstance(snapshot, dict):
        return default
    keys = keys or ["score", "overall_score", "trust_score", "trust_score_v2", "release_success_score", "operational_score", "governance_score"]
    for key in keys:
        val = snapshot.get(key)
        if isinstance(val, (int, float, np.integer, np.floating)):
            return int(val)
    return default


def create_guided_success_snapshot(
    project_name,
    customer_segment="Predictive Maintenance Team",
    use_case_type="Predictive Maintenance",
    data_status="Some synthetic/demo data",
    technical_level="Technical operator",
    desired_outcome="Generate pilot bundle",
    sales_mode="Private beta",
    selected_plan="Founder Test Mode",
    dataset_df=None,
    manifest=None,
    trust_snapshot=None,
    reliability_snapshot=None,
    deployment_snapshot=None,
    report_snapshot=None,
    assurance_snapshot=None,
    success_snapshot=None,
    operational_snapshot=None,
    observability_snapshot=None,
    governance_snapshot=None,
):
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    manifest = manifest or {}
    rows = int(len(dataset_df))
    cols = int(len(dataset_df.columns))
    labels = sorted([str(x) for x in dataset_df["Label"].dropna().unique().tolist()]) if "Label" in dataset_df.columns else []

    evidence_items = [
        {"evidence": "Dataset loaded", "present": rows > 0, "why_it_matters": "Customer can see concrete generated or uploaded data."},
        {"evidence": "Labels/classes available", "present": len(labels) >= 2, "why_it_matters": "Pilot needs at least two states/classes or a clear normal baseline."},
        {"evidence": "Trust Center / readiness", "present": _has_snapshot(trust_snapshot), "why_it_matters": "Avoids overpromising and gives a readiness position."},
        {"evidence": "Reliability estimate", "present": _has_snapshot(reliability_snapshot), "why_it_matters": "Explains pilot confidence and remaining validation need."},
        {"evidence": "Deployment plan", "present": _has_snapshot(deployment_snapshot), "why_it_matters": "Connects data work to real hardware and field deployment."},
        {"evidence": "Professional report", "present": _has_snapshot(report_snapshot), "why_it_matters": "Gives customer a management-ready deliverable."},
        {"evidence": "Customer assurance", "present": _has_snapshot(assurance_snapshot or governance_snapshot), "why_it_matters": "Explains how customer uploads and claims are handled."},
        {"evidence": "Success Gate", "present": _has_snapshot(success_snapshot), "why_it_matters": "Final go/no-go control before demo or paid pilot."},
        {"evidence": "Operational Control", "present": _has_snapshot(operational_snapshot), "why_it_matters": "Shows operator discipline and launch oversight."},
        {"evidence": "Error Observatory", "present": _has_snapshot(observability_snapshot), "why_it_matters": "Shows issues can be logged and managed."},
    ]
    present_count = sum(1 for x in evidence_items if x["present"])
    evidence_score = int((present_count / max(1, len(evidence_items))) * 100)

    data_score_map = {
        "No data yet": 25,
        "Some synthetic/demo data": 55,
        "Small real WAV/CSV sample set": 70,
        "Enough real baseline data": 82,
        "Real normal + abnormal data available": 92,
    }
    data_score = data_score_map.get(data_status, 50)
    dataset_score = 0
    if rows > 0:
        dataset_score += 25
    if rows >= 200:
        dataset_score += 20
    if rows >= 1000:
        dataset_score += 10
    if len(labels) >= 2:
        dataset_score += 25
    if len(labels) >= 4:
        dataset_score += 10
    if cols >= 5:
        dataset_score += 10
    dataset_score = int(np.clip(dataset_score, 0, 100))

    reliability_score = _readiness_value(reliability_snapshot, ["reliability_score", "trust_score_v2", "score"], 0)
    trust_score = _readiness_value(trust_snapshot, ["trust_score", "score", "readiness_score"], 0)
    success_score = _readiness_value(success_snapshot, ["release_success_score", "success_score", "score"], 0)
    governance_score = _readiness_value(assurance_snapshot or governance_snapshot, ["governance_score", "score"], 0)
    operational_score = _readiness_value(operational_snapshot, ["operational_score", "score"], 0)

    score = int(np.clip(
        dataset_score * 0.22 + data_score * 0.18 + evidence_score * 0.28 +
        max(reliability_score, trust_score) * 0.14 + max(success_score, operational_score) * 0.10 + governance_score * 0.08,
        0, 100
    ))

    blockers = []
    if rows == 0:
        blockers.append({"severity": "high", "message": "No active dataset. Run a demo, wizard, industry pack or upload real data first."})
    if len(labels) < 2:
        blockers.append({"severity": "high", "message": "Dataset has fewer than two labels/classes. Build a clearer normal/abnormal or classifier setup."})
    if sales_mode in ["Paid pilot", "Enterprise review"] and not _has_snapshot(assurance_snapshot or governance_snapshot):
        blockers.append({"severity": "high", "message": "Customer assurance is missing. Run V34 before paid/enterprise customer data flows."})
    if desired_outcome in ["Prepare Edge Impulse export", "Prepare paid pilot delivery"] and not _has_snapshot(success_snapshot):
        blockers.append({"severity": "medium", "message": "Release Success Gate snapshot is missing. Run Success Gate before delivery."})
    if data_status == "No data yet" and sales_mode in ["Paid pilot", "Enterprise review"]:
        blockers.append({"severity": "medium", "message": "Paid pilot with no data is weak. Generate a demo bundle or collect baseline samples first."})

    if score >= 82 and not any(b.get("severity") == "high" for b in blockers):
        decision = "READY FOR GUIDED CUSTOMER SESSION"
    elif score >= 62:
        decision = "CONDITIONAL - COMPLETE MISSING EVIDENCE"
    else:
        decision = "NO-GO - PREPARE FOUNDATION FIRST"

    if "Real" in data_status:
        recommended_package = "Real-Data Pilot Bundle"
    elif sales_mode in ["Paid pilot", "Enterprise review"]:
        recommended_package = "Professional / Enterprise Pilot Bundle"
    elif desired_outcome == "Understand if pilot is worth starting":
        recommended_package = "Starter Pilot Bundle"
    else:
        recommended_package = "Professional Pilot Bundle"

    if desired_outcome == "Prepare Edge Impulse export":
        next_milestone = "Run Edge Impulse Anomaly/Classifer Export"
    elif "field" in desired_outcome.lower() or "real" in data_status.lower():
        next_milestone = "Run Real Field Validation"
    elif sales_mode in ["Paid pilot", "Enterprise review"]:
        next_milestone = "Export customer assurance + paid pilot bundle"
    else:
        next_milestone = "Run Golden Demo + Success Gate"

    # Customer journey deliberately references UI tabs by business outcome, not raw technical jargon only.
    journey_steps = [
        {"step": 1, "stage": "Choose customer problem", "tab": "Self-Selling Demo / Use Case Wizard", "output": "Use-case, sensors, labels and project goal."},
        {"step": 2, "stage": "Generate pilot evidence", "tab": "Auto Pilot / Industry Pack / Real Bridge", "output": "Dataset, features and initial bundle."},
        {"step": 3, "stage": "Check reliability", "tab": "Smart Optimizer / Reliability / Trust Center", "output": "Weak-class, readiness and risk evidence."},
        {"step": 4, "stage": "Plan deployment", "tab": "Deployment Planner / Edge Starter", "output": "Hardware, BOM, power, communication and field plan."},
        {"step": 5, "stage": "Prepare customer proof", "tab": "Reports 2.0 / Customer Assurance / Success Gate", "output": "Safe report, claims, go/no-go and next actions."},
        {"step": 6, "stage": "Deliver or iterate", "tab": "Closed Beta / Paid Export / Operational Control", "output": "Demo session, paid bundle or beta feedback loop."},
    ]

    demo_script = [
        f"Start with the customer pain: {use_case_type} for {customer_segment}.",
        "Show that EdgeTwin does not promise magic: it creates pilot-preparation evidence and highlights risks.",
        "Run or show the generated dataset/classes, then explain normal vs abnormal or classifier route.",
        "Show reliability/trust/readiness scores and what real samples are still needed.",
        "Show hardware/deployment direction so the pilot feels practical, not theoretical.",
        "End with the exact next step: demo bundle, real-data upload, field validation, or paid pilot.",
    ]

    next_actions = []
    missing = [x["evidence"] for x in evidence_items if not x["present"]]
    if missing:
        next_actions.append("Complete missing evidence: " + ", ".join(missing[:4]) + ("..." if len(missing) > 4 else ""))
    if rows == 0:
        next_actions.append("Generate a golden demo or use-case wizard pilot dataset before any customer session.")
    if "real" in data_status.lower() and not _has_snapshot(assurance_snapshot or governance_snapshot):
        next_actions.append("Run Customer Assurance before accepting or discussing real customer data uploads.")
    if not next_actions:
        next_actions.append("Export the Guided Success Bundle and use the included demo script for the next customer conversation.")

    success_metrics = [
        {"metric": "Time to first useful pilot package", "target": "< 15 minutes in guided demo"},
        {"metric": "Customer understands next real-data requirement", "target": "Yes, per-class/sample guidance explained"},
        {"metric": "No overclaiming", "target": "Pilot-ready vs production-ready distinction shown"},
        {"metric": "Concrete deliverable exported", "target": "ZIP/PDF/CSV bundle delivered or queued"},
        {"metric": "Next step agreed", "target": "Demo, real-data upload, field validation or paid pilot"},
    ]

    customer_facing_copy = (
        f"EdgeTwin Studio helps {customer_segment.lower()} turn a {use_case_type.lower()} sensor idea into a guided Edge AI pilot package: "
        "dataset, risk checks, reliability evidence, hardware/deployment advice and exportable reports. "
        "The output is intended to accelerate pilot preparation; real field validation is required before production use."
    )

    return {
        "version": "V35 Customer Onboarding & Guided Success Flow",
        "created_at": _now(),
        "project_name": project_name,
        "customer_segment": customer_segment,
        "use_case_type": use_case_type,
        "data_status": data_status,
        "technical_level": technical_level,
        "desired_outcome": desired_outcome,
        "sales_mode": sales_mode,
        "selected_plan": selected_plan,
        "dataset_rows": rows,
        "dataset_cols": cols,
        "labels": labels,
        "dataset_score": dataset_score,
        "data_score": data_score,
        "evidence_score": evidence_score,
        "onboarding_score": score,
        "decision": decision,
        "recommended_package": recommended_package,
        "next_milestone": next_milestone,
        "journey_steps": journey_steps,
        "evidence_coverage": evidence_items,
        "demo_script": demo_script,
        "success_metrics": success_metrics,
        "blockers": blockers,
        "next_actions": next_actions,
        "customer_success_line": "Guided customer session is safe when evidence, assurance and limitations are visible before any paid/real-data promise.",
        "customer_facing_copy": customer_facing_copy,
        "claims_to_avoid": [
            "Do not claim production readiness without field validation.",
            "Do not claim EdgeTwin replaces engineering review for safety-critical deployment.",
            "Do not imply customer data will be reused without explicit opt-in.",
            "Do not promise perfect anomaly detection from synthetic-only data.",
        ],
        "disclaimer": GUIDED_SUCCESS_DISCLAIMER,
    }


def generate_guided_success_pdf(project_name, snapshot):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Studio Guided Success Plan", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)
    safe_pdf_cell(pdf, "Guided Success Decision", 8, True)
    safe_pdf_cell(pdf, f"Onboarding score: {snapshot.get('onboarding_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {snapshot.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Recommended package: {snapshot.get('recommended_package', 'Unknown')}")
    safe_pdf_cell(pdf, f"Next milestone: {snapshot.get('next_milestone', 'Unknown')}")
    safe_pdf_multicell(pdf, snapshot.get("customer_facing_copy", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Recommended Journey", 8, True)
    for step in snapshot.get("journey_steps", []):
        safe_pdf_multicell(pdf, f"{step.get('step')}. {step.get('stage')} - {step.get('tab')} -> {step.get('output')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Evidence Coverage", 8, True)
    for item in snapshot.get("evidence_coverage", []):
        status = "OK" if item.get("present") else "MISSING"
        safe_pdf_multicell(pdf, f"[{status}] {item.get('evidence')} - {item.get('why_it_matters', '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Demo Script", 8, True)
    for item in snapshot.get("demo_script", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Blockers", 8, True)
    blockers = snapshot.get("blockers", [])
    if not blockers:
        safe_pdf_multicell(pdf, "No high-priority guided success blockers detected.")
    for b in blockers:
        safe_pdf_multicell(pdf, f"[{b.get('severity', 'info').upper()}] {b.get('message', '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Next Actions", 8, True)
    for item in snapshot.get("next_actions", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Disclaimer", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", GUIDED_SUCCESS_DISCLAIMER))
    return safe_pdf_output(pdf)


def create_guided_success_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    pdf_bytes = generate_guided_success_pdf(project_name, snapshot)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("guided_success_plan_v35.pdf", pdf_bytes)
        zf.writestr("guided_success_snapshot.json", safe_json_dumps(snapshot, indent=2))
        if snapshot.get("journey_steps"):
            zf.writestr("guided_journey_steps.csv", pd.DataFrame(snapshot.get("journey_steps", [])).to_csv(index=False))
        if snapshot.get("evidence_coverage"):
            zf.writestr("evidence_coverage.csv", pd.DataFrame(snapshot.get("evidence_coverage", [])).to_csv(index=False))
        if snapshot.get("success_metrics"):
            zf.writestr("success_metrics.csv", pd.DataFrame(snapshot.get("success_metrics", [])).to_csv(index=False))
        if snapshot.get("blockers"):
            zf.writestr("onboarding_blockers.csv", pd.DataFrame(snapshot.get("blockers", [])).to_csv(index=False))
        if len(dataset_df) > 0:
            zf.writestr("dataset_snapshot_head.csv", dataset_df.head(1000).to_csv(index=False))
        zf.writestr("CUSTOMER_DEMO_SCRIPT.md", "# Customer Demo Script\n\n" + "\n".join([f"- {x}" for x in snapshot.get("demo_script", [])]))
        zf.writestr("README.md", f"""# EdgeTwin Studio V35 Guided Success Bundle

Project: {project_name}
Decision: {snapshot.get('decision', 'Unknown')}
Onboarding score: {snapshot.get('onboarding_score', 0)}%
Recommended package: {snapshot.get('recommended_package', 'Unknown')}
Next milestone: {snapshot.get('next_milestone', 'Unknown')}

## Customer-facing copy
{snapshot.get('customer_facing_copy', '')}

## Next actions
""" + "\n".join([f"- {x}" for x in snapshot.get("next_actions", [])]) + f"""

## Disclaimer
{snapshot.get('disclaimer', GUIDED_SUCCESS_DISCLAIMER)}
""")
    return zip_buf.getvalue()

# ============================================================
# V36 CUSTOMER WORKSPACE / PROJECT LIFECYCLE
# ============================================================

WORKSPACE_LIFECYCLE_DISCLAIMER = (
    "Workspace status is an operational planning aid. It does not certify production readiness, safety, "
    "legal compliance or model performance. Field validation and customer approval are required before deployment."
)

WORKSPACE_LIFECYCLE_STAGES = [
    "Idea / discovery",
    "Demo prepared",
    "Pilot package generated",
    "Real data bridge",
    "Field validation",
    "Paid pilot delivery",
    "Deployment preparation",
    "Post-pilot review",
]

WORKSPACE_TARGET_GOALS = [
    "Qualify opportunity",
    "Run customer demo",
    "Deliver pilot bundle",
    "Prepare Edge Impulse workflow",
    "Validate with real field data",
    "Prepare paid pilot",
    "Prepare deployment handoff",
]


def get_workspace_lifecycle_stages():
    return list(WORKSPACE_LIFECYCLE_STAGES)


def get_workspace_target_goals():
    return list(WORKSPACE_TARGET_GOALS)


def _workspace_present(snapshot):
    return isinstance(snapshot, dict) and bool(snapshot)


def _workspace_status_from_score(score, high_risk=False):
    if high_risk:
        return "NO-GO - FIX LIFECYCLE BLOCKERS"
    if score >= 82:
        return "READY FOR CUSTOMER WORKSPACE HANDOFF"
    if score >= 62:
        return "CONDITIONAL - COMPLETE WORKSPACE EVIDENCE"
    return "NO-GO - BUILD CORE EVIDENCE FIRST"


def create_workspace_lifecycle_snapshot(
    project_name,
    customer_name="Private Beta Customer",
    lifecycle_stage="Demo prepared",
    target_goal="Deliver pilot bundle",
    owner_role="Founder/operator",
    delivery_window="2 weeks",
    risk_tolerance="Medium",
    selected_plan="Founder Test Mode",
    dataset_df=None,
    manifest=None,
    onboarding_snapshot=None,
    assurance_snapshot=None,
    trust_snapshot=None,
    reliability_snapshot=None,
    deployment_snapshot=None,
    report_snapshot=None,
    success_snapshot=None,
    operational_snapshot=None,
    observability_snapshot=None,
    paid_license_snapshot=None,
    field_validation_snapshot=None,
    edge_export_snapshot=None,
    classifier_export_snapshot=None,
):
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    manifest = manifest or {}
    rows = int(len(dataset_df))
    cols = int(len(dataset_df.columns))
    labels = sorted([str(x) for x in dataset_df["Label"].dropna().unique().tolist()]) if "Label" in dataset_df.columns else []

    evidence_map = [
        {"area": "Dataset", "present": rows > 0, "status": "OK" if rows > 0 else "Missing", "proof": f"{rows} rows / {cols} columns"},
        {"area": "Labels/classes", "present": len(labels) >= 2, "status": "OK" if len(labels) >= 2 else "Weak", "proof": ", ".join(labels[:8]) if labels else "No labels"},
        {"area": "Guided onboarding", "present": _workspace_present(onboarding_snapshot), "status": "OK" if _workspace_present(onboarding_snapshot) else "Missing", "proof": str((onboarding_snapshot or {}).get("decision", "Not run"))},
        {"area": "Customer assurance", "present": _workspace_present(assurance_snapshot), "status": "OK" if _workspace_present(assurance_snapshot) else "Missing", "proof": str((assurance_snapshot or {}).get("decision", "Not run"))},
        {"area": "Trust / reliability", "present": _workspace_present(trust_snapshot) or _workspace_present(reliability_snapshot), "status": "OK" if (_workspace_present(trust_snapshot) or _workspace_present(reliability_snapshot)) else "Missing", "proof": f"trust={_readiness_value(trust_snapshot, ['trust_score','score'], 0)} reliability={_readiness_value(reliability_snapshot, ['reliability_score','trust_score_v2','score'], 0)}"},
        {"area": "Deployment", "present": _workspace_present(deployment_snapshot), "status": "OK" if _workspace_present(deployment_snapshot) else "Missing", "proof": str((deployment_snapshot or {}).get("recommendation", (deployment_snapshot or {}).get("decision", "Not run")))},
        {"area": "Professional report", "present": _workspace_present(report_snapshot), "status": "OK" if _workspace_present(report_snapshot) else "Missing", "proof": str((report_snapshot or {}).get("decision", "Not run"))},
        {"area": "Release success gate", "present": _workspace_present(success_snapshot), "status": "OK" if _workspace_present(success_snapshot) else "Missing", "proof": str((success_snapshot or {}).get("decision", "Not run"))},
        {"area": "Operational control", "present": _workspace_present(operational_snapshot), "status": "OK" if _workspace_present(operational_snapshot) else "Missing", "proof": str((operational_snapshot or {}).get("decision", "Not run"))},
        {"area": "Error observability", "present": _workspace_present(observability_snapshot), "status": "OK" if _workspace_present(observability_snapshot) else "Missing", "proof": str((observability_snapshot or {}).get("decision", "Not run"))},
        {"area": "Paid/license state", "present": _workspace_present(paid_license_snapshot), "status": "OK" if _workspace_present(paid_license_snapshot) else "Optional", "proof": str((paid_license_snapshot or {}).get("decision", "Not run"))},
        {"area": "Field validation", "present": _workspace_present(field_validation_snapshot), "status": "OK" if _workspace_present(field_validation_snapshot) else "Optional", "proof": str((field_validation_snapshot or {}).get("decision", "Not run"))},
        {"area": "Edge Impulse exports", "present": _workspace_present(edge_export_snapshot) or _workspace_present(classifier_export_snapshot), "status": "OK" if (_workspace_present(edge_export_snapshot) or _workspace_present(classifier_export_snapshot)) else "Optional", "proof": "Anomaly/classifier export available" if (_workspace_present(edge_export_snapshot) or _workspace_present(classifier_export_snapshot)) else "Not prepared"},
    ]

    required_areas = {"Dataset", "Labels/classes", "Guided onboarding", "Customer assurance", "Trust / reliability", "Release success gate", "Operational control", "Error observability"}
    required_present = sum(1 for x in evidence_map if x["area"] in required_areas and x["present"])
    required_total = len(required_areas)
    optional_present = sum(1 for x in evidence_map if x["area"] not in required_areas and x["present"])
    evidence_score = int(np.clip((required_present / required_total) * 82 + min(optional_present * 4, 18), 0, 100))

    dataset_score = 0
    if rows > 0: dataset_score += 25
    if rows >= 200: dataset_score += 20
    if rows >= 1000: dataset_score += 10
    if len(labels) >= 2: dataset_score += 25
    if len(labels) >= 4: dataset_score += 10
    if cols >= 5: dataset_score += 10
    dataset_score = int(np.clip(dataset_score, 0, 100))

    stage_weights = {
        "Idea / discovery": 35,
        "Demo prepared": 55,
        "Pilot package generated": 70,
        "Real data bridge": 76,
        "Field validation": 84,
        "Paid pilot delivery": 86,
        "Deployment preparation": 90,
        "Post-pilot review": 92,
    }
    stage_score = stage_weights.get(lifecycle_stage, 55)

    confidence_scores = [
        _readiness_value(onboarding_snapshot, ["onboarding_score", "score"], 0),
        _readiness_value(assurance_snapshot, ["governance_score", "score"], 0),
        max(_readiness_value(trust_snapshot, ["trust_score", "score"], 0), _readiness_value(reliability_snapshot, ["reliability_score", "trust_score_v2", "score"], 0)),
        _readiness_value(success_snapshot, ["release_success_score", "success_score", "score"], 0),
        _readiness_value(operational_snapshot, ["operational_score", "score"], 0),
        _readiness_value(observability_snapshot, ["safety_score", "score"], 0),
    ]
    confidence_score = int(np.mean([s for s in confidence_scores if s > 0])) if any(s > 0 for s in confidence_scores) else 0

    score = int(np.clip(dataset_score * 0.22 + evidence_score * 0.34 + stage_score * 0.16 + confidence_score * 0.28, 0, 100))

    deliverables = [
        {"deliverable": "Pilot dataset / features", "required": True, "ready": rows > 0, "owner": owner_role},
        {"deliverable": "Reliability / trust evidence", "required": True, "ready": _workspace_present(trust_snapshot) or _workspace_present(reliability_snapshot), "owner": owner_role},
        {"deliverable": "Customer assurance notice", "required": True, "ready": _workspace_present(assurance_snapshot), "owner": owner_role},
        {"deliverable": "Professional report", "required": target_goal in ["Deliver pilot bundle", "Prepare paid pilot", "Prepare deployment handoff"], "ready": _workspace_present(report_snapshot), "owner": owner_role},
        {"deliverable": "Deployment / hardware plan", "required": target_goal in ["Prepare deployment handoff", "Validate with real field data"], "ready": _workspace_present(deployment_snapshot), "owner": owner_role},
        {"deliverable": "Edge Impulse export", "required": target_goal == "Prepare Edge Impulse workflow", "ready": _workspace_present(edge_export_snapshot) or _workspace_present(classifier_export_snapshot), "owner": owner_role},
        {"deliverable": "Field validation plan", "required": target_goal in ["Validate with real field data", "Prepare deployment handoff"], "ready": _workspace_present(field_validation_snapshot), "owner": owner_role},
        {"deliverable": "Paid license / entitlement snapshot", "required": target_goal in ["Prepare paid pilot", "Prepare deployment handoff"], "ready": _workspace_present(paid_license_snapshot), "owner": owner_role},
    ]

    risks = []
    if rows == 0:
        risks.append({"severity": "high", "risk": "No dataset loaded", "mitigation": "Run a golden demo, wizard, industry pack or real bridge before customer delivery."})
    if len(labels) < 2:
        risks.append({"severity": "high", "risk": "No clear classes/baseline", "mitigation": "Create normal/abnormal labels or supervised classes."})
    if target_goal in ["Prepare paid pilot", "Prepare deployment handoff"] and not _workspace_present(assurance_snapshot):
        risks.append({"severity": "high", "risk": "Customer assurance missing", "mitigation": "Run Customer Assurance before paid/enterprise delivery."})
    if target_goal == "Prepare Edge Impulse workflow" and not (_workspace_present(edge_export_snapshot) or _workspace_present(classifier_export_snapshot)):
        risks.append({"severity": "medium", "risk": "Edge Impulse export not prepared", "mitigation": "Run anomaly and/or classifier export before hand-off."})
    if risk_tolerance == "Low" and score < 82:
        risks.append({"severity": "medium", "risk": "Customer risk tolerance is low", "mitigation": "Only present as guided demo/private beta until missing evidence is complete."})
    if delivery_window in ["Today", "This week"] and any(d.get("required") and not d.get("ready") for d in deliverables):
        risks.append({"severity": "medium", "risk": "Short delivery window with missing deliverables", "mitigation": "Reduce scope to demo bundle or conditional pilot package."})

    high_risk = any(r.get("severity") == "high" for r in risks)
    decision = _workspace_status_from_score(score, high_risk=high_risk)

    if high_risk:
        next_action = "Fix high-risk blockers"
    elif target_goal == "Run customer demo":
        next_action = "Export Guided Success + demo bundle"
    elif target_goal == "Deliver pilot bundle":
        next_action = "Export report + success gate bundle"
    elif target_goal == "Prepare Edge Impulse workflow":
        next_action = "Export EI anomaly/classifier bundle"
    elif target_goal == "Validate with real field data":
        next_action = "Run field validation plan"
    elif target_goal == "Prepare paid pilot":
        next_action = "Run paid license + customer assurance"
    elif target_goal == "Prepare deployment handoff":
        next_action = "Run deployment + edge starter handoff"
    else:
        next_action = "Complete qualification evidence"

    timeline = [
        {"order": 1, "stage": "Discovery", "status": "done" if lifecycle_stage != "Idea / discovery" else "current", "expected_output": "Problem, sensors, desired outcome"},
        {"order": 2, "stage": "Demo / pilot generation", "status": "done" if rows > 0 else "next", "expected_output": "Dataset, labels/classes, first bundle"},
        {"order": 3, "stage": "Evidence & trust", "status": "done" if (_workspace_present(trust_snapshot) or _workspace_present(reliability_snapshot)) else "next", "expected_output": "Reliability, risks, samples needed"},
        {"order": 4, "stage": "Customer assurance", "status": "done" if _workspace_present(assurance_snapshot) else "next", "expected_output": "Upload notice, ownership, allowed claims"},
        {"order": 5, "stage": "Delivery package", "status": "done" if _workspace_present(report_snapshot) or _workspace_present(success_snapshot) else "next", "expected_output": "Report, Success Gate, handoff bundle"},
        {"order": 6, "stage": "Validation / deployment", "status": "done" if _workspace_present(field_validation_snapshot) or _workspace_present(deployment_snapshot) else "future", "expected_output": "Field plan, BOM, edge starter"},
    ]

    next_actions = []
    for r in risks:
        if r.get("severity") == "high":
            next_actions.append(r.get("mitigation"))
    if not next_actions:
        missing_required = [d["deliverable"] for d in deliverables if d.get("required") and not d.get("ready")]
        if missing_required:
            next_actions.append("Complete required deliverables: " + ", ".join(missing_required[:4]))
        else:
            next_actions.append("Export the Workspace Lifecycle Bundle and use it as the customer/project control record.")
    if target_goal == "Prepare paid pilot" and not _workspace_present(paid_license_snapshot):
        next_actions.append("Run Paid Export / License Gate before sending paid bundle.")
    if target_goal == "Validate with real field data" and not _workspace_present(field_validation_snapshot):
        next_actions.append("Run Real Field Validation to define baseline and sample collection plan.")

    customer_handoff_summary = (
        f"Workspace for {customer_name}: {project_name} is at '{lifecycle_stage}' aiming for '{target_goal}'. "
        f"Current decision: {decision}. The next safe action is: {next_action}. "
        "EdgeTwin outputs support pilot preparation; production deployment requires field validation and customer approval."
    )
    operator_notes = [
        "Keep one workspace bundle per customer milestone so decisions are traceable.",
        "Do not jump to paid/deployment delivery if assurance, success gate or reliability evidence is missing.",
        "Use real-data uploads only with clear customer data ownership and retention statements.",
        "For low-risk-tolerance customers, show limitations before pricing or deployment claims.",
    ]

    return {
        "version": "V36 Customer Workspace & Project Lifecycle",
        "created_at": _now(),
        "project_name": project_name,
        "customer_name": customer_name,
        "lifecycle_stage": lifecycle_stage,
        "target_goal": target_goal,
        "owner_role": owner_role,
        "delivery_window": delivery_window,
        "risk_tolerance": risk_tolerance,
        "selected_plan": selected_plan,
        "dataset_rows": rows,
        "dataset_cols": cols,
        "labels": labels,
        "manifest_template": manifest.get("template", manifest.get("pack_name", "Unknown")),
        "dataset_score": dataset_score,
        "evidence_score": evidence_score,
        "stage_score": stage_score,
        "confidence_score": confidence_score,
        "workspace_score": score,
        "decision": decision,
        "next_action": next_action,
        "timeline": timeline,
        "deliverables": deliverables,
        "evidence_map": evidence_map,
        "risks": risks,
        "next_actions": next_actions,
        "customer_handoff_summary": customer_handoff_summary,
        "operator_status_line": "Workspace is safe to progress only when required deliverables, limitations and customer assurance are visible.",
        "operator_notes": operator_notes,
        "claims_to_avoid": [
            "Do not claim this workspace certifies production readiness.",
            "Do not imply synthetic-only data is sufficient for safety-critical deployment.",
            "Do not send paid/deployment deliverables without customer assurance and success gate evidence.",
            "Do not reuse customer data beyond the agreed retention/reuse policy.",
        ],
        "disclaimer": WORKSPACE_LIFECYCLE_DISCLAIMER,
    }


def generate_workspace_lifecycle_pdf(project_name, snapshot):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Studio Workspace Lifecycle Plan", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.cell(0, 8, txt=clean_pdf_text(f"Customer: {snapshot.get('customer_name', 'Unknown')}"), ln=True, align="C")
    pdf.ln(8)
    safe_pdf_cell(pdf, "Workspace Decision", 8, True)
    safe_pdf_cell(pdf, f"Workspace score: {snapshot.get('workspace_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {snapshot.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Lifecycle stage: {snapshot.get('lifecycle_stage', 'Unknown')}")
    safe_pdf_cell(pdf, f"Target goal: {snapshot.get('target_goal', 'Unknown')}")
    safe_pdf_cell(pdf, f"Next action: {snapshot.get('next_action', 'Unknown')}")
    safe_pdf_multicell(pdf, snapshot.get("customer_handoff_summary", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Lifecycle Timeline", 8, True)
    for item in snapshot.get("timeline", []):
        safe_pdf_multicell(pdf, f"{item.get('order')}. {item.get('stage')} [{item.get('status')}] - {item.get('expected_output')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Deliverables", 8, True)
    for item in snapshot.get("deliverables", []):
        status = "READY" if item.get("ready") else "MISSING"
        req = "required" if item.get("required") else "optional"
        safe_pdf_multicell(pdf, f"[{status}] {item.get('deliverable')} ({req}) - owner: {item.get('owner')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Risks", 8, True)
    risks = snapshot.get("risks", [])
    if not risks:
        safe_pdf_multicell(pdf, "No major workspace lifecycle risks detected.")
    for risk in risks:
        safe_pdf_multicell(pdf, f"[{risk.get('severity', 'info').upper()}] {risk.get('risk')} - Mitigation: {risk.get('mitigation')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Next Actions", 8, True)
    for item in snapshot.get("next_actions", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Disclaimer", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", WORKSPACE_LIFECYCLE_DISCLAIMER))
    return safe_pdf_output(pdf)


def create_workspace_lifecycle_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    pdf_bytes = generate_workspace_lifecycle_pdf(project_name, snapshot)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("workspace_lifecycle_plan_v36.pdf", pdf_bytes)
        zf.writestr("workspace_lifecycle_snapshot.json", safe_json_dumps(snapshot, indent=2))
        if snapshot.get("timeline"):
            zf.writestr("lifecycle_timeline.csv", pd.DataFrame(snapshot.get("timeline", [])).to_csv(index=False))
        if snapshot.get("deliverables"):
            zf.writestr("deliverable_checklist.csv", pd.DataFrame(snapshot.get("deliverables", [])).to_csv(index=False))
        if snapshot.get("evidence_map"):
            zf.writestr("workspace_evidence_map.csv", pd.DataFrame(snapshot.get("evidence_map", [])).to_csv(index=False))
        if snapshot.get("risks"):
            zf.writestr("workspace_risks.csv", pd.DataFrame(snapshot.get("risks", [])).to_csv(index=False))
        if len(dataset_df) > 0:
            zf.writestr("dataset_snapshot_head.csv", dataset_df.head(1000).to_csv(index=False))
        zf.writestr("CUSTOMER_HANDOFF_SUMMARY.md", f"""# Customer Handoff Summary

{snapshot.get('customer_handoff_summary', '')}

## Next action
{snapshot.get('next_action', 'Unknown')}

## Next actions
""" + "\n".join([f"- {x}" for x in snapshot.get("next_actions", [])]) + f"""

## Disclaimer
{snapshot.get('disclaimer', WORKSPACE_LIFECYCLE_DISCLAIMER)}
""")
        zf.writestr("README.md", f"""# EdgeTwin Studio V36 Workspace Lifecycle Bundle

Project: {project_name}
Customer: {snapshot.get('customer_name', 'Unknown')}
Decision: {snapshot.get('decision', 'Unknown')}
Workspace score: {snapshot.get('workspace_score', 0)}%
Lifecycle stage: {snapshot.get('lifecycle_stage', 'Unknown')}
Target goal: {snapshot.get('target_goal', 'Unknown')}
Next action: {snapshot.get('next_action', 'Unknown')}

This bundle is intended as the customer/project lifecycle control record.

## Disclaimer
{snapshot.get('disclaimer', WORKSPACE_LIFECYCLE_DISCLAIMER)}
""")
    return zip_buf.getvalue()



# ============================================================
# V37 ADMIN DASHBOARD / USAGE TRACKING
# ============================================================

ADMIN_USAGE_DISCLAIMER = (
    "Admin/usage metrics are local operational indicators for beta and paid-pilot management. "
    "They are not financial accounting records and should be connected to a payment provider, audit log and backup system before public SaaS launch."
)

EXPORT_BUNDLE_TYPES = [
    "Fusion Bundle",
    "Enterprise Bundle",
    "Auto Pilot Bundle",
    "Optimizer Bundle",
    "Trust Bundle",
    "Real Bridge Bundle",
    "Reliability 2.0 Bundle",
    "Deployment Bundle",
    "Professional Report Bundle",
    "Monetization Bundle",
    "Hardening Bundle",
    "Beta Launch Bundle",
    "API Automation Bundle",
    "Marketplace Bundle",
    "Normality Bundle",
    "Edge Impulse Anomaly Bundle",
    "Edge Impulse Classifier Bundle",
    "Release Success Bundle",
    "Golden Demo Bundle",
    "Closed Beta Bundle",
    "Paid Export Bundle",
    "Field Validation Bundle",
    "Edge Starter Bundle",
    "Scalability Bundle",
    "Operational Control Bundle",
    "Observability Bundle",
    "Customer Assurance Bundle",
    "Guided Success Bundle",
    "Workspace Lifecycle Bundle",
]


def _ensure_admin_usage_tables():
    """Create lightweight admin/usage tables without breaking old SQLite installs."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS export_events (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            project_id TEXT,
            project_name TEXT,
            bundle_type TEXT,
            plan_name TEXT,
            status TEXT,
            notes TEXT,
            created_at TEXT
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS operator_notes (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            project_id TEXT,
            project_name TEXT,
            note_type TEXT,
            severity TEXT,
            note TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def record_export_event(user_id, project_id, project_name, bundle_type, plan_name="Founder Test Mode", status="prepared", notes=""):
    """Record a bundle/export event for local beta admin tracking."""
    _ensure_admin_usage_tables()
    bundle_type = str(bundle_type or "Unknown Bundle")[:120]
    plan_name = str(plan_name or "Unknown")[:80]
    status = str(status or "prepared")[:40]
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO export_events (id, user_id, project_id, project_name, bundle_type, plan_name, status, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (str(uuid.uuid4()), user_id, project_id, str(project_name or "")[:160], bundle_type, plan_name, status, str(notes or "")[:1000], _now()),
    )
    conn.commit()
    conn.close()
    return {"status": "recorded", "bundle_type": bundle_type, "plan_name": plan_name}


def record_operator_note(user_id, project_id, project_name, note_type="general", severity="info", note=""):
    """Record an operator/customer-success note for the current project."""
    _ensure_admin_usage_tables()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO operator_notes (id, user_id, project_id, project_name, note_type, severity, note, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (str(uuid.uuid4()), user_id, project_id, str(project_name or "")[:160], str(note_type or "general")[:80], str(severity or "info")[:40], str(note or "")[:2000], _now()),
    )
    conn.commit()
    conn.close()
    return {"status": "recorded", "note_type": note_type, "severity": severity}


def get_export_events(user_id=None, limit=250):
    _ensure_admin_usage_tables()
    conn = sqlite3.connect(DB_NAME)
    if user_id:
        df = pd.read_sql_query(
            "SELECT * FROM export_events WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            conn,
            params=(user_id, int(limit)),
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM export_events ORDER BY created_at DESC LIMIT ?",
            conn,
            params=(int(limit),),
        )
    conn.close()
    return df


def get_operator_notes(user_id=None, limit=250):
    _ensure_admin_usage_tables()
    conn = sqlite3.connect(DB_NAME)
    if user_id:
        df = pd.read_sql_query(
            "SELECT * FROM operator_notes WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            conn,
            params=(user_id, int(limit)),
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM operator_notes ORDER BY created_at DESC LIMIT ?",
            conn,
            params=(int(limit),),
        )
    conn.close()
    return df


def _admin_plan_limits(plan_name):
    plan_name = str(plan_name or "Free Demo")
    # Keep this defensive because older pricing dictionaries may use slightly different names.
    bundle_limits = {
        "Free Demo": 1,
        "Starter Pilot": 5,
        "Starter": 5,
        "Professional Pilot": 25,
        "Professional": 25,
        "Real-Data Pilot": 50,
        "Enterprise": 500,
        "Founder Test Mode": 9999,
    }
    api_limits = {
        "Free Demo": 0,
        "Starter Pilot": 50,
        "Starter": 50,
        "Professional Pilot": 500,
        "Professional": 500,
        "Real-Data Pilot": 1500,
        "Enterprise": 10000,
        "Founder Test Mode": 99999,
    }
    return {
        "plan_name": plan_name,
        "sample_limit": get_sample_limit_for_plan(plan_name),
        "bundle_limit": int(bundle_limits.get(plan_name, 5)),
        "api_call_limit": int(api_limits.get(plan_name, 50)),
    }


def create_admin_usage_dashboard_snapshot(
    user=None,
    project_name="EdgeTwin_Project",
    project_id="",
    plan_name="Founder Test Mode",
    dataset_df=None,
    projects_df=None,
    active_bundles=None,
    operating_mode="Closed beta",
    customer_success_stage="Internal QA",
    operator_notes_text="",
):
    """Build the V37 admin/customer-success dashboard snapshot."""
    _ensure_admin_usage_tables()
    user = user or {}
    user_id = user.get("id", "") if isinstance(user, dict) else ""
    username = user.get("username", "unknown") if isinstance(user, dict) else "unknown"
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    projects_df = projects_df if isinstance(projects_df, pd.DataFrame) else pd.DataFrame()
    active_bundles = active_bundles or {}

    export_df = get_export_events(user_id if user_id else None, limit=500)
    notes_df = get_operator_notes(user_id if user_id else None, limit=250)

    active_bundle_rows = []
    for name, present in active_bundles.items():
        active_bundle_rows.append({
            "bundle_type": str(name),
            "ready_in_session": bool(present),
            "status": "Ready" if present else "Not prepared",
        })

    limits = _admin_plan_limits(plan_name)
    rows = int(len(dataset_df))
    cols = int(len(dataset_df.columns))
    project_count = int(len(projects_df))
    exports_logged = int(len(export_df))
    notes_count = int(len(notes_df))
    api_calls = get_api_usage_count(user_id) if user_id else 0
    ready_bundles = int(sum(1 for x in active_bundle_rows if x["ready_in_session"]))

    utilization = {
        "sample_usage_pct": int(np.clip((rows / max(limits["sample_limit"], 1)) * 100, 0, 999)),
        "bundle_usage_pct": int(np.clip((exports_logged / max(limits["bundle_limit"], 1)) * 100, 0, 999)),
        "api_usage_pct": int(np.clip((api_calls / max(limits["api_call_limit"], 1)) * 100, 0, 999)) if limits["api_call_limit"] > 0 else 0,
    }

    issues = []
    if rows == 0:
        issues.append({"severity": "medium", "message": "No active dataset loaded. Admin view is useful, but customer delivery is not ready yet."})
    if rows > limits["sample_limit"]:
        issues.append({"severity": "high", "message": f"Dataset rows exceed the selected plan sample limit ({rows} > {limits['sample_limit']})."})
    if exports_logged > limits["bundle_limit"] and plan_name != "Founder Test Mode":
        issues.append({"severity": "medium", "message": "Logged exports exceed the local plan bundle limit. Review paid unlock/upgrade status."})
    if ready_bundles == 0:
        issues.append({"severity": "medium", "message": "No export bundles are ready in this session. Generate at least one deliverable before customer handoff."})
    if project_count == 0:
        issues.append({"severity": "info", "message": "No saved project history yet. Save projects so admin tracking can prove continuity."})
    if not operator_notes_text and notes_count == 0:
        issues.append({"severity": "info", "message": "No operator notes recorded yet. Add notes for beta/customer-success traceability."})

    score = 100
    for item in issues:
        if item["severity"] == "high":
            score -= 35
        elif item["severity"] == "medium":
            score -= 15
    if ready_bundles >= 3:
        score += 5
    if project_count >= 1:
        score += 5
    if notes_count >= 1 or operator_notes_text:
        score += 3
    score = int(np.clip(score, 0, 100))

    if score >= 85 and not any(i["severity"] == "high" for i in issues):
        decision = "ADMIN-READY"
        summary = "Admin/usage controls look suitable for closed beta and paid-pilot supervision."
    elif score >= 65:
        decision = "CONDITIONAL"
        summary = "Admin/usage controls are usable, but fix the listed items before relying on it for customer delivery."
    else:
        decision = "NO-GO"
        summary = "Admin/usage readiness is weak. Use internally only until blockers are resolved."

    if exports_logged > 0 and "bundle_type" in export_df.columns:
        by_bundle = export_df.groupby("bundle_type").size().reset_index(name="count").sort_values("count", ascending=False).to_dict(orient="records")
    else:
        by_bundle = []

    project_rows = []
    if len(projects_df) > 0:
        for _, row in projects_df.head(50).iterrows():
            project_rows.append({
                "id": str(row.get("id", "")),
                "name": str(row.get("name", "")),
                "updated_at": str(row.get("updated_at", row.get("created_at", ""))),
                "storage_mode": str(row.get("storage_mode", "unknown")),
                "dataset_rows": int(row.get("dataset_rows", 0) or 0),
                "dataset_cols": int(row.get("dataset_cols", 0) or 0),
            })

    recommended_actions = []
    if rows == 0:
        recommended_actions.append("Run a Golden Demo or Auto Pilot flow before offering a customer bundle.")
    if ready_bundles == 0:
        recommended_actions.append("Generate at least one report/export bundle and record the export event.")
    if project_count == 0:
        recommended_actions.append("Save the project to create traceability for future beta/customer delivery.")
    if not recommended_actions:
        recommended_actions.append("Review the dashboard before each customer call and record delivered bundle events.")
        recommended_actions.append("Keep paid-pilot delivery tied to a project, plan and bundle log.")

    return {
        "version": "V37 Admin Dashboard & Usage Tracking",
        "created_at": _now(),
        "project_name": project_name,
        "project_id": str(project_id or ""),
        "user": {"id": user_id, "username": username},
        "plan_name": plan_name,
        "operating_mode": operating_mode,
        "customer_success_stage": customer_success_stage,
        "admin_score": score,
        "decision": decision,
        "summary": summary,
        "limits": limits,
        "utilization": utilization,
        "dataset": {"rows": rows, "cols": cols, "hash": dataset_hash(dataset_df) if len(dataset_df) else "empty", "has_label": "Label" in dataset_df.columns},
        "project_count": project_count,
        "api_calls_logged": api_calls,
        "exports_logged": exports_logged,
        "operator_notes_count": notes_count,
        "ready_bundles_in_session": ready_bundles,
        "active_bundles": active_bundle_rows,
        "exports_by_bundle": by_bundle,
        "project_rows": project_rows,
        "issues": issues,
        "recommended_actions": recommended_actions,
        "operator_notes_text": str(operator_notes_text or ""),
        "disclaimer": ADMIN_USAGE_DISCLAIMER,
    }


def generate_admin_usage_dashboard_pdf(project_name, snapshot):
    snapshot = snapshot or {}
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Admin Dashboard & Usage Report", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("V37 - Customer success, usage and delivery tracking"), ln=True, align="C")
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(8)

    safe_pdf_cell(pdf, "Admin Readiness", 8, True)
    safe_pdf_cell(pdf, f"Score: {snapshot.get('admin_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {snapshot.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Plan: {snapshot.get('plan_name', 'Unknown')}")
    safe_pdf_cell(pdf, f"Operating mode: {snapshot.get('operating_mode', 'Unknown')}")
    safe_pdf_multicell(pdf, snapshot.get("summary", ""))
    pdf.ln(4)

    ds = snapshot.get("dataset", {}) or {}
    safe_pdf_cell(pdf, "Dataset / Project", 8, True)
    safe_pdf_cell(pdf, f"Dataset rows: {ds.get('rows', 0)}")
    safe_pdf_cell(pdf, f"Dataset columns: {ds.get('cols', 0)}")
    safe_pdf_cell(pdf, f"Saved projects: {snapshot.get('project_count', 0)}")
    safe_pdf_cell(pdf, f"Ready bundles in session: {snapshot.get('ready_bundles_in_session', 0)}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Usage", 8, True)
    safe_pdf_cell(pdf, f"Exports logged: {snapshot.get('exports_logged', 0)}")
    safe_pdf_cell(pdf, f"API calls logged: {snapshot.get('api_calls_logged', 0)}")
    safe_pdf_cell(pdf, f"Operator notes: {snapshot.get('operator_notes_count', 0)}")
    util = snapshot.get("utilization", {}) or {}
    safe_pdf_cell(pdf, f"Sample usage: {util.get('sample_usage_pct', 0)}%")
    safe_pdf_cell(pdf, f"Bundle usage: {util.get('bundle_usage_pct', 0)}%")
    safe_pdf_cell(pdf, f"API usage: {util.get('api_usage_pct', 0)}%")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Issues", 8, True)
    issues = snapshot.get("issues", [])
    if not issues:
        safe_pdf_multicell(pdf, "No admin blockers detected.")
    for item in issues:
        safe_pdf_multicell(pdf, f"[{item.get('severity', 'info').upper()}] {item.get('message', '')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Recommended Actions", 8, True)
    for item in snapshot.get("recommended_actions", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Disclaimer", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", ADMIN_USAGE_DISCLAIMER))
    return safe_pdf_output(pdf)


def create_admin_usage_dashboard_bundle(project_name, snapshot, export_events_df=None, projects_df=None, notes_df=None, dataset_df=None):
    snapshot = snapshot or {}
    export_events_df = export_events_df if isinstance(export_events_df, pd.DataFrame) else pd.DataFrame()
    projects_df = projects_df if isinstance(projects_df, pd.DataFrame) else pd.DataFrame()
    notes_df = notes_df if isinstance(notes_df, pd.DataFrame) else pd.DataFrame()
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    pdf_bytes = generate_admin_usage_dashboard_pdf(project_name, snapshot)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("admin_usage_report_v37.pdf", pdf_bytes)
        zf.writestr("admin_usage_snapshot.json", safe_json_dumps(snapshot, indent=2))
        if snapshot.get("active_bundles"):
            zf.writestr("active_bundles.csv", pd.DataFrame(snapshot.get("active_bundles", [])).to_csv(index=False))
        if snapshot.get("issues"):
            zf.writestr("admin_issues.csv", pd.DataFrame(snapshot.get("issues", [])).to_csv(index=False))
        if snapshot.get("exports_by_bundle"):
            zf.writestr("exports_by_bundle.csv", pd.DataFrame(snapshot.get("exports_by_bundle", [])).to_csv(index=False))
        if len(export_events_df) > 0:
            zf.writestr("export_events.csv", export_events_df.to_csv(index=False))
        if len(projects_df) > 0:
            zf.writestr("project_index.csv", projects_df.to_csv(index=False))
        if len(notes_df) > 0:
            zf.writestr("operator_notes.csv", notes_df.to_csv(index=False))
        if len(dataset_df) > 0:
            zf.writestr("dataset_snapshot.csv", dataset_df.head(5000).to_csv(index=False))
        zf.writestr("README.md", f"""# EdgeTwin Studio V37 Admin Dashboard Bundle

Project: {project_name}
Decision: {snapshot.get('decision', 'Unknown')}
Admin score: {snapshot.get('admin_score', 0)}%
Plan: {snapshot.get('plan_name', 'Unknown')}

## Purpose
This bundle captures local beta/customer-success usage state: projects, export events, ready bundles, notes, limits and blockers.

## Important
{snapshot.get('disclaimer', ADMIN_USAGE_DISCLAIMER)}
""")
    return zip_buf.getvalue()


# ============================================================
# V38 COMMERCIAL LICENSE CERTIFICATE / PAID EXPORT LOCKING
# ============================================================

COMMERCIAL_LICENSE_VERSION = "EdgeTwin Studio V38 - Commercial License Certificate & Paid Export Locking"
COMMERCIAL_LICENSE_DISCLAIMER = (
    "This certificate documents commercial entitlement, delivery scope and safe-use boundaries for an EdgeTwin Studio export. "
    "It is not a payment processor, not production certification, not a model accuracy guarantee and not regulatory approval. "
    "Production deployment requires real-world validation, customer acceptance testing and documented deployment controls."
)

LICENSE_SCOPE_OPTIONS = [
    "Demo preview only",
    "Internal pilot preparation",
    "Paid pilot preparation",
    "Real-data pilot preparation",
    "Enterprise review / custom handoff",
]

LICENSE_VALIDITY_OPTIONS = {
    "14 days": 14,
    "30 days": 30,
    "90 days": 90,
    "180 days": 180,
    "1 year": 365,
    "Perpetual snapshot": 0,
}


def get_license_scope_options():
    return LICENSE_SCOPE_OPTIONS.copy()


def get_license_validity_options():
    return list(LICENSE_VALIDITY_OPTIONS.keys())


def _license_valid_until(validity_label):
    days = int(LICENSE_VALIDITY_OPTIONS.get(validity_label, 30))
    if days <= 0:
        return "Perpetual snapshot license - tied to generated bundle hash and safe-use terms"
    try:
        return (datetime.datetime.now() + datetime.timedelta(days=days)).date().isoformat()
    except Exception:
        return "Unknown"


def _safe_license_token(*parts):
    raw = "|".join(str(p or "") for p in parts) + "|" + _now()
    digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest().upper()
    return f"ETC-{digest[:5]}-{digest[5:10]}-{digest[10:15]}-{digest[15:20]}"


def _certificate_hash(payload):
    try:
        raw = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False).encode("utf-8", errors="ignore")
    except Exception:
        raw = str(payload).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def _paid_snapshot_summary(paid_license_snapshot):
    snap = paid_license_snapshot or {}
    summary = snap.get("summary", {}) or {}
    manifest = snap.get("entitlement_manifest", {}) or {}
    return {
        "paid_gate_present": bool(snap),
        "paid_gate_decision": summary.get("decision", "Not run"),
        "paid_gate_status": summary.get("license_status", "Unknown"),
        "paid_gate_score": int(summary.get("commercial_readiness_score", 0) or 0),
        "paid_gate_license_key": manifest.get("license_key", ""),
        "paid_gate_package": manifest.get("requested_package", summary.get("requested_package", "Unknown")),
        "paid_gate_plan": manifest.get("selected_plan", summary.get("selected_plan", "Unknown")),
    }


def build_commercial_license_certificate(
    project_name,
    customer_name="Beta Customer",
    customer_email="customer@example.com",
    customer_organization="Customer Organization",
    selected_plan="Professional Pilot",
    requested_package="Professional Pilot Bundle",
    license_scope="Paid pilot preparation",
    validity_label="30 days",
    payment_reference="Manual invoice / not connected",
    dataset_df=None,
    paid_license_snapshot=None,
    release_success_snapshot=None,
    customer_assurance_snapshot=None,
    workspace_lifecycle_snapshot=None,
    admin_usage_snapshot=None,
    watermark_previews=True,
    include_dataset_snapshot=True,
):
    """Create a customer-facing commercial certificate for paid/manual delivery.

    This is stricter than V29: V29 creates an entitlement gate; V38 creates a signed delivery certificate
    with scope, expiry, hash, unlocked exports, locked exports, safe terms and final delivery decision.
    """
    rows = int(len(dataset_df)) if isinstance(dataset_df, pd.DataFrame) else 0
    cols = int(len(dataset_df.columns)) if isinstance(dataset_df, pd.DataFrame) else 0
    labels = []
    if isinstance(dataset_df, pd.DataFrame) and "Label" in dataset_df.columns:
        labels = [str(x) for x in dataset_df["Label"].dropna().unique().tolist()]

    paid = _paid_snapshot_summary(paid_license_snapshot)
    release_decision = (release_success_snapshot or {}).get("decision", "Not run") if isinstance(release_success_snapshot, dict) else "Not run"
    assurance_decision = (customer_assurance_snapshot or {}).get("decision", "Not run") if isinstance(customer_assurance_snapshot, dict) else "Not run"
    workspace_decision = (workspace_lifecycle_snapshot or {}).get("decision", "Not run") if isinstance(workspace_lifecycle_snapshot, dict) else "Not run"
    admin_decision = (admin_usage_snapshot or {}).get("decision", "Not run") if isinstance(admin_usage_snapshot, dict) else "Not run"

    requested_exports = PAID_PACKAGE_EXPORTS.get(requested_package, PAID_PACKAGE_EXPORTS.get("Professional Pilot Bundle", []))
    access_matrix = []
    unlocked = []
    locked = []
    for export_key in requested_exports:
        access = evaluate_export_access(selected_plan, export_key)
        if selected_plan == "Founder Test Mode":
            access["allowed"] = True
            access["reason"] = "Founder/internal testing only. Not a real customer license."
        if access.get("allowed"):
            unlocked.append(access)
        else:
            locked.append(access)
        access_matrix.append(access)

    blockers = []
    warnings = []
    if rows <= 0:
        blockers.append("No active dataset snapshot is attached to this license certificate.")
    if selected_plan == "Founder Test Mode":
        blockers.append("Founder Test Mode is internal only. Use a real customer plan before paid delivery.")
    if not paid.get("paid_gate_present"):
        warnings.append("Paid Export Gate has not been created yet. V38 can certify scope, but V29 entitlement evidence is missing.")
    elif paid.get("paid_gate_status") not in ["ACTIVE_FOR_DELIVERY", "REVIEW_REQUIRED"]:
        warnings.append(f"Paid Export Gate status is {paid.get('paid_gate_status')}; review before delivery.")
    if "NO-GO" in str(release_decision).upper():
        blockers.append("Release Success Gate is NO-GO. Do not issue a paid customer certificate yet.")
    if "NO-GO" in str(assurance_decision).upper():
        blockers.append("Customer Assurance is NO-GO. Fix governance/privacy blockers before customer data delivery.")
    if len(unlocked) == 0:
        blockers.append("No exports are unlocked for the selected plan/package combination.")
    if customer_email and "@" not in customer_email:
        warnings.append("Customer email does not look valid; certificate delivery metadata may be incomplete.")

    score = 100
    score -= 25 * len(blockers)
    score -= 8 * len(warnings)
    if paid.get("paid_gate_present"):
        score = min(score, 75 + int(paid.get("paid_gate_score", 0)) // 5)
    if release_decision in ["GO", "CUSTOMER-READY", "ADMIN-READY"] or "GO" in str(release_decision):
        score += 5
    if assurance_decision and "READY" in str(assurance_decision):
        score += 5
    score = int(np.clip(score, 0, 100))

    if blockers:
        decision = "DO NOT ISSUE"
        certificate_status = "BLOCKED"
        customer_status_line = "This project is not ready for paid customer license delivery yet. Fix blockers first."
    elif score >= 85 and not locked:
        decision = "ISSUE LICENSE"
        certificate_status = "ACTIVE"
        customer_status_line = "This package is approved for controlled paid pilot-preparation delivery under the listed terms."
    elif score >= 70:
        decision = "ISSUE WITH REVIEW"
        certificate_status = "REVIEW_REQUIRED"
        customer_status_line = "This package can be issued after manual review of locked exports, warnings and delivery scope."
    else:
        decision = "INTERNAL ONLY"
        certificate_status = "INTERNAL_REVIEW"
        customer_status_line = "Keep this certificate internal until readiness evidence improves."

    certificate_id = _safe_license_token(project_name, customer_email, selected_plan, requested_package, license_scope)
    bundle_fingerprint_payload = {
        "project_name": project_name,
        "customer_name": customer_name,
        "customer_email": customer_email,
        "customer_organization": customer_organization,
        "selected_plan": selected_plan,
        "requested_package": requested_package,
        "license_scope": license_scope,
        "validity_label": validity_label,
        "dataset_hash": dataset_hash(dataset_df) if isinstance(dataset_df, pd.DataFrame) else "empty",
        "unlocked_exports": [x.get("export_key") for x in unlocked],
        "locked_exports": [x.get("export_key") for x in locked],
    }
    certificate_hash = _certificate_hash(bundle_fingerprint_payload)

    safe_use_terms = [
        "Licensed output is for pilot preparation, dataset review, feasibility assessment and decision support only.",
        "No production model accuracy, safety, regulatory or compliance guarantee is provided by this certificate.",
        "Production deployment requires real-world field validation, customer acceptance testing and documented operational controls.",
        "Synthetic and synthetic-to-real data are planning aids and do not replace validated field evidence.",
        "Customer-owned uploaded data must follow the agreed retention, privacy and reuse policy.",
        "Do not resell, redistribute or publish paid bundles outside the licensed customer/workspace without written permission.",
    ]

    delivery_conditions = [
        "Deliver the certificate together with the matching bundle files and entitlement manifest.",
        "Keep a copy of certificate_hash and certificate_id in the project/admin notes.",
        "Use watermarked previews for Free Demo or review-only deliveries.",
        "Re-issue the certificate when dataset, report, plan, scope or customer identity changes materially.",
    ]

    watermark_policy = "Watermark preview snapshots" if watermark_previews and selected_plan in ["Free Demo", "Founder Test Mode"] else "No preview watermark required for active paid exports"
    if certificate_status != "ACTIVE":
        watermark_policy = "Review watermark recommended until certificate becomes ACTIVE"

    certificate = {
        "version": COMMERCIAL_LICENSE_VERSION,
        "created_at": _now(),
        "certificate_id": certificate_id,
        "certificate_hash": certificate_hash,
        "certificate_status": certificate_status,
        "decision": decision,
        "license_score": score,
        "customer_status_line": customer_status_line,
        "customer": {
            "name": customer_name,
            "email": customer_email,
            "organization": customer_organization,
        },
        "project": {
            "project_name": project_name,
            "dataset_rows": rows,
            "dataset_cols": cols,
            "labels": labels,
            "dataset_hash": dataset_hash(dataset_df) if isinstance(dataset_df, pd.DataFrame) else "empty",
        },
        "commercial_scope": {
            "selected_plan": selected_plan,
            "requested_package": requested_package,
            "license_scope": license_scope,
            "validity": validity_label,
            "valid_until": _license_valid_until(validity_label),
            "payment_reference": payment_reference,
            "price_position": PAID_PRICE_POSITIONS.get(requested_package, "Custom"),
            "watermark_policy": watermark_policy,
            "include_dataset_snapshot": bool(include_dataset_snapshot),
        },
        "evidence": {
            "paid_gate": paid,
            "release_success_decision": release_decision,
            "customer_assurance_decision": assurance_decision,
            "workspace_decision": workspace_decision,
            "admin_decision": admin_decision,
        },
        "access_matrix": access_matrix,
        "unlocked_exports": unlocked,
        "locked_exports": locked,
        "blockers": blockers,
        "warnings": warnings,
        "safe_use_terms": safe_use_terms,
        "delivery_conditions": delivery_conditions,
        "disclaimer": COMMERCIAL_LICENSE_DISCLAIMER,
    }
    return _json_safe(certificate)


def generate_commercial_license_certificate_pdf(project_name, certificate):
    cert = certificate or {}
    customer = cert.get("customer", {}) or {}
    scope = cert.get("commercial_scope", {}) or {}
    project = cert.get("project", {}) or {}
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt=clean_pdf_text("EdgeTwin Commercial License Certificate"), ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("V38 - Paid Export Locking & Customer Delivery Certificate"), ln=True, align="C")
    pdf.ln(8)
    safe_pdf_cell(pdf, "Certificate Status", 8, True)
    safe_pdf_cell(pdf, f"Decision: {cert.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Status: {cert.get('certificate_status', 'Unknown')}")
    safe_pdf_cell(pdf, f"License score: {cert.get('license_score', 0)}%")
    safe_pdf_cell(pdf, f"Certificate ID: {cert.get('certificate_id', '')}")
    safe_pdf_multicell(pdf, f"Certificate hash: {cert.get('certificate_hash', '')}")
    safe_pdf_multicell(pdf, cert.get("customer_status_line", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Customer & Project", 8, True)
    safe_pdf_cell(pdf, f"Customer: {customer.get('name', '')}")
    safe_pdf_cell(pdf, f"Organization: {customer.get('organization', '')}")
    safe_pdf_cell(pdf, f"Email: {customer.get('email', '')}")
    safe_pdf_cell(pdf, f"Project: {project.get('project_name', project_name)}")
    safe_pdf_cell(pdf, f"Dataset rows/cols: {project.get('dataset_rows', 0)} / {project.get('dataset_cols', 0)}")
    safe_pdf_multicell(pdf, f"Dataset hash: {project.get('dataset_hash', 'empty')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Commercial Scope", 8, True)
    for key in ["selected_plan", "requested_package", "license_scope", "validity", "valid_until", "payment_reference", "price_position", "watermark_policy"]:
        safe_pdf_multicell(pdf, f"{key.replace('_',' ').title()}: {scope.get(key, '')}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Unlocked Exports", 8, True)
    if cert.get("unlocked_exports"):
        for row in cert.get("unlocked_exports", []):
            safe_pdf_multicell(pdf, f"- {row.get('label', row.get('export_key'))}: {row.get('commercial_value', '')}")
    else:
        safe_pdf_multicell(pdf, "No unlocked exports.")
    pdf.ln(2)
    safe_pdf_cell(pdf, "Locked Exports", 8, True)
    if cert.get("locked_exports"):
        for row in cert.get("locked_exports", []):
            safe_pdf_multicell(pdf, f"- {row.get('label', row.get('export_key'))}: {row.get('reason', '')}")
    else:
        safe_pdf_multicell(pdf, "No locked exports for this certificate scope.")
    if cert.get("blockers"):
        pdf.ln(4)
        safe_pdf_cell(pdf, "Blockers", 8, True)
        for item in cert.get("blockers", []):
            safe_pdf_multicell(pdf, f"- {item}")
    if cert.get("warnings"):
        pdf.ln(4)
        safe_pdf_cell(pdf, "Warnings", 8, True)
        for item in cert.get("warnings", []):
            safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Safe-use Terms", 8, True)
    for item in cert.get("safe_use_terms", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Delivery Conditions", 8, True)
    for item in cert.get("delivery_conditions", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Disclaimer", 8, True)
    safe_pdf_multicell(pdf, cert.get("disclaimer", COMMERCIAL_LICENSE_DISCLAIMER))
    return safe_pdf_output(pdf)


def create_commercial_license_certificate_bundle(project_name, certificate, dataset_df=None):
    cert = certificate or {}
    pdf_bytes = generate_commercial_license_certificate_pdf(project_name, cert)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("commercial_license_certificate_v38.pdf", pdf_bytes)
        zf.writestr("commercial_license_certificate.json", safe_json_dumps(cert, indent=2))
        if cert.get("access_matrix"):
            zf.writestr("export_access_matrix.csv", pd.DataFrame(cert.get("access_matrix", [])).to_csv(index=False))
        if cert.get("unlocked_exports"):
            zf.writestr("unlocked_exports.csv", pd.DataFrame(cert.get("unlocked_exports", [])).to_csv(index=False))
        if cert.get("locked_exports"):
            zf.writestr("locked_exports.csv", pd.DataFrame(cert.get("locked_exports", [])).to_csv(index=False))
        if cert.get("blockers"):
            zf.writestr("blockers.txt", "\n".join(cert.get("blockers", [])))
        if cert.get("warnings"):
            zf.writestr("warnings.txt", "\n".join(cert.get("warnings", [])))
        zf.writestr("safe_use_terms.txt", "\n".join(cert.get("safe_use_terms", [])))
        zf.writestr("delivery_conditions.txt", "\n".join(cert.get("delivery_conditions", [])))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0 and cert.get("commercial_scope", {}).get("include_dataset_snapshot", True):
            out_df = dataset_df.head(5000).copy()
            if "watermark" in str(cert.get("commercial_scope", {}).get("watermark_policy", "")).lower():
                out_df["EdgeTwin_License_Watermark"] = f"{cert.get('certificate_status', 'REVIEW')}_{cert.get('certificate_id', '')}"
            zf.writestr("licensed_dataset_snapshot.csv", out_df.to_csv(index=False))
        zf.writestr("README.md", f"""# EdgeTwin Studio V38 Commercial License Certificate

Project: {project_name}
Certificate ID: {cert.get('certificate_id', 'Unknown')}
Status: {cert.get('certificate_status', 'Unknown')}
Decision: {cert.get('decision', 'Unknown')}
Hash: {cert.get('certificate_hash', 'Unknown')}

This bundle documents paid/manual delivery entitlement, export locking, safe-use terms and delivery scope.
It is not a payment processor, production certification, safety guarantee or regulatory approval.

{cert.get('disclaimer', COMMERCIAL_LICENSE_DISCLAIMER)}
""")
    return zip_buf.getvalue()


# ============================================================
# V39 FIELD EVIDENCE 2.0 / ACCEPTANCE GATE
# ============================================================

FIELD_EVIDENCE_V2_VERSION = "V39 Field Evidence 2.0"
FIELD_EVIDENCE_V2_DISCLAIMER = (
    "Field Evidence 2.0 is a controlled pilot-evidence gate. It can support paid pilot decisions, "
    "but it is not production certification, safety approval, regulatory approval or a guarantee of model accuracy."
)

FIELD_EVIDENCE_V2_GOALS = [
    "Paid pilot acceptance",
    "Private beta evidence",
    "Internal validation",
    "Edge Impulse readiness",
    "Production-readiness precheck",
]

FIELD_EVIDENCE_V2_ACCEPTANCE_LEVELS = [
    "Light beta",
    "Paid pilot",
    "Strict enterprise",
]


def get_field_evidence_v2_goals():
    return FIELD_EVIDENCE_V2_GOALS.copy()


def get_field_evidence_v2_acceptance_levels():
    return FIELD_EVIDENCE_V2_ACCEPTANCE_LEVELS.copy()


def _v39_score(snapshot, keys, default=0):
    if not isinstance(snapshot, dict):
        return default
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        cur = snapshot
        ok = True
        for part in str(key).split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            try:
                return int(cur or 0)
            except Exception:
                return default
    return default


def _v39_decision(snapshot, keys, default="Not run"):
    if not isinstance(snapshot, dict):
        return default
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        cur = snapshot
        ok = True
        for part in str(key).split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur:
            return str(cur)
    return default


def _v39_profile_df(df, name="dataset"):
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return {
            "name": name,
            "rows": 0,
            "cols": 0,
            "has_label": False,
            "label_count": 0,
            "labels": [],
            "numeric_cols": [],
            "missing_ratio": 0.0,
            "duplicate_rows": 0,
        }
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "Label"]
    labels = []
    if "Label" in df.columns:
        labels = [str(x) for x in df["Label"].dropna().unique().tolist()]
    missing_ratio = float(df.isna().mean().mean()) if len(df.columns) else 0.0
    duplicate_rows = int(df.duplicated().sum()) if len(df) else 0
    return {
        "name": name,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "has_label": "Label" in df.columns,
        "label_count": int(len(labels)),
        "labels": labels,
        "numeric_cols": numeric_cols,
        "missing_ratio": round(missing_ratio, 4),
        "duplicate_rows": duplicate_rows,
        "dataset_hash": dataset_hash(df),
    }


def _v39_label_evidence(field_df, synthetic_df=None):
    rows = []
    if not isinstance(field_df, pd.DataFrame) or len(field_df) == 0 or "Label" not in field_df.columns:
        return rows
    field_counts = field_df["Label"].astype(str).value_counts().to_dict()
    synthetic_counts = {}
    if isinstance(synthetic_df, pd.DataFrame) and len(synthetic_df) > 0 and "Label" in synthetic_df.columns:
        synthetic_counts = synthetic_df["Label"].astype(str).value_counts().to_dict()
    normal_keywords = ["normal", "healthy", "baseline", "background", "idle", "calm"]
    for label, count in field_counts.items():
        synthetic_match = int(synthetic_counts.get(label, 0))
        is_normal = any(k in label.lower() for k in normal_keywords)
        if count >= 50:
            coverage = "strong"
        elif count >= 20:
            coverage = "usable"
        elif count >= 5:
            coverage = "weak"
        else:
            coverage = "very weak"
        rows.append({
            "label": label,
            "field_samples": int(count),
            "synthetic_samples_same_label": synthetic_match,
            "baseline_like": bool(is_normal),
            "coverage": coverage,
            "recommended_next_action": "Collect more field samples" if count < 20 else "Use for pilot evidence, validate in real deployment",
        })
    return rows


def _v39_field_vs_synthetic_drift(field_df, synthetic_df):
    if not isinstance(field_df, pd.DataFrame) or len(field_df) == 0:
        return {"status": "no_field_data", "similarity_score": 0, "drift_level": "Unknown", "message": "No field data available."}
    if not isinstance(synthetic_df, pd.DataFrame) or len(synthetic_df) == 0:
        return {"status": "no_synthetic_reference", "similarity_score": 35, "drift_level": "Unknown", "message": "No synthetic reference dataset available."}
    common = [c for c in field_df.columns if c in synthetic_df.columns and c != "Label" and pd.api.types.is_numeric_dtype(field_df[c]) and pd.api.types.is_numeric_dtype(synthetic_df[c])]
    if not common:
        return {"status": "no_common_numeric_features", "similarity_score": 20, "drift_level": "High", "message": "No common numeric features to compare."}
    f = field_df[common].replace([np.inf, -np.inf], np.nan).fillna(0)
    s = synthetic_df[common].replace([np.inf, -np.inf], np.nan).fillna(0)
    f_mean = f.mean()
    s_mean = s.mean()
    f_std = f.std().replace(0, 1.0)
    s_std = s.std().replace(0, 1.0)
    mean_delta = ((f_mean - s_mean).abs() / (s_std.abs() + 1e-6)).clip(0, 10)
    std_delta = ((f_std - s_std).abs() / (s_std.abs() + 1e-6)).clip(0, 10)
    drift_raw = float((mean_delta.mean() * 0.65) + (std_delta.mean() * 0.35))
    similarity = int(np.clip(100 - drift_raw * 18, 0, 100))
    if similarity >= 80:
        drift_level = "Low"
    elif similarity >= 60:
        drift_level = "Medium"
    else:
        drift_level = "High"
    top = []
    for col in common:
        top.append({
            "feature": col,
            "mean_delta_z": float(round(mean_delta.get(col, 0), 4)),
            "std_delta_ratio": float(round(std_delta.get(col, 0), 4)),
        })
    top = sorted(top, key=lambda x: x["mean_delta_z"] + x["std_delta_ratio"], reverse=True)[:8]
    return {
        "status": "compared",
        "common_features": common,
        "similarity_score": similarity,
        "drift_level": drift_level,
        "drift_raw": round(drift_raw, 4),
        "top_drift_features": top,
        "message": "Field and synthetic distributions are close enough for pilot use." if similarity >= 70 else "Field/synthetic drift is significant; collect more real data or tune synthetic generation.",
    }


def build_field_evidence_v2_gate(
    project_name,
    field_df=None,
    synthetic_df=None,
    evidence_goal="Paid pilot acceptance",
    acceptance_level="Paid pilot",
    site_name="Pilot Site A",
    field_days=14,
    sensor_stack=None,
    target_customer="Predictive Maintenance",
    reliability_v2=None,
    normality_result=None,
    edge_impulse_snapshot=None,
    edge_impulse_classifier_snapshot=None,
    deployment_plan=None,
    field_validation_snapshot=None,
    customer_assurance_snapshot=None,
    commercial_license_certificate=None,
):
    sensor_stack = sensor_stack or []
    field_profile = _v39_profile_df(field_df, "field_evidence")
    synthetic_profile = _v39_profile_df(synthetic_df, "synthetic_reference")
    label_evidence = _v39_label_evidence(field_df, synthetic_df)
    drift = _v39_field_vs_synthetic_drift(field_df, synthetic_df)

    strictness = {"Light beta": 0, "Paid pilot": 1, "Strict enterprise": 2}.get(acceptance_level, 1)
    min_rows = [20, 50, 150][strictness]
    min_labels = [1, 2, 3][strictness]
    min_days = [3, 7, 14][strictness]
    min_similarity = [45, 60, 70][strictness]

    field_rows = field_profile["rows"]
    label_count = field_profile["label_count"]
    has_normal = any(row.get("baseline_like") for row in label_evidence)
    real_validation_score = _v39_score(field_validation_snapshot, ["summary.field_evidence_score", "field_evidence_score"], 0)
    reliability_score = _v39_score(reliability_v2, ["summary.trust_score_v2", "trust_score_v2", "reliability_score"], 0)
    normality_score = _v39_score(normality_result, ["summary.normality_score", "normality_score"], 0)
    assurance_score = _v39_score(customer_assurance_snapshot, ["summary.governance_score", "governance_score", "customer_assurance_score"], 0)
    license_score = _v39_score(commercial_license_certificate, ["license_score"], 0)
    deployment_score = _v39_score(deployment_plan, ["summary.deployment_score", "deployment_score"], 0)
    ei_anomaly_decision = _v39_decision(edge_impulse_snapshot, ["decision", "summary.decision"], "Not run")
    ei_classifier_decision = _v39_decision(edge_impulse_classifier_snapshot, ["decision", "summary.decision"], "Not run")

    acceptance_checklist = [
        {"check": "Real field rows", "target": min_rows, "actual": field_rows, "passed": field_rows >= min_rows},
        {"check": "Field labels/classes", "target": min_labels, "actual": label_count, "passed": label_count >= min_labels},
        {"check": "Normal/baseline evidence", "target": "present", "actual": "present" if has_normal else "missing", "passed": has_normal},
        {"check": "Field evidence days", "target": min_days, "actual": int(field_days or 0), "passed": int(field_days or 0) >= min_days},
        {"check": "Field/synthetic similarity", "target": min_similarity, "actual": drift.get("similarity_score", 0), "passed": int(drift.get("similarity_score", 0)) >= min_similarity},
        {"check": "Sensor stack declared", "target": "1+ sensors", "actual": len(sensor_stack), "passed": len(sensor_stack) >= 1},
        {"check": "Customer assurance", "target": "recommended", "actual": assurance_score, "passed": assurance_score >= 60 or acceptance_level == "Light beta"},
        {"check": "Deployment evidence", "target": "recommended", "actual": deployment_score, "passed": deployment_score >= 50 or evidence_goal in ["Internal validation", "Edge Impulse readiness"]},
    ]

    blockers = []
    warnings = []
    for item in acceptance_checklist:
        if not item.get("passed"):
            if item["check"] in ["Real field rows", "Normal/baseline evidence", "Field/synthetic similarity"] and strictness >= 1:
                blockers.append(f"{item['check']} is below target: {item.get('actual')} vs {item.get('target')}.")
            else:
                warnings.append(f"{item['check']} is incomplete: {item.get('actual')} vs {item.get('target')}.")
    if field_profile["missing_ratio"] > 0.05:
        warnings.append("Field dataset contains missing values above 5%; clean the upload before relying on evidence.")
    if field_profile["duplicate_rows"] > max(2, field_rows * 0.05):
        warnings.append("Field dataset has many duplicate rows; verify that evidence is not artificially repeated.")
    if acceptance_level == "Strict enterprise" and license_score < 70:
        warnings.append("Commercial license certificate is weak or missing for strict enterprise delivery.")
    if "NO" in str(ei_anomaly_decision).upper() and "NO" in str(ei_classifier_decision).upper():
        warnings.append("Both Edge Impulse exports appear NO-GO or missing; validate export path before customer handoff.")

    base_score = 0
    base_score += min(30, int(field_rows / max(min_rows, 1) * 30))
    base_score += min(15, int(label_count / max(min_labels, 1) * 15))
    base_score += 10 if has_normal else 0
    base_score += min(15, int((field_days or 0) / max(min_days, 1) * 15))
    base_score += int(drift.get("similarity_score", 0) * 0.15)
    base_score += min(8, int(len(sensor_stack) * 2))
    base_score += int(max(real_validation_score, reliability_score, normality_score) * 0.05)
    base_score += int(max(assurance_score, deployment_score) * 0.03)
    score = int(np.clip(base_score - 18 * len(blockers) - 6 * len(warnings), 0, 100))

    if blockers:
        decision = "NO-GO"
        grade = "Insufficient field evidence"
        customer_status_line = "Do not present this as paid field-validated evidence yet. Fix blockers first."
    elif score >= 85:
        decision = "FIELD-EVIDENCE READY"
        grade = "Strong pilot evidence"
        customer_status_line = "Field evidence is strong enough for a controlled paid pilot-preparation handoff."
    elif score >= 68:
        decision = "CONDITIONAL GO"
        grade = "Usable with review"
        customer_status_line = "Field evidence can support a controlled pilot discussion, but warnings must be reviewed."
    else:
        decision = "INTERNAL ONLY"
        grade = "Needs more evidence"
        customer_status_line = "Keep this internal until more real field evidence is collected."

    safe_claims = [
        "EdgeTwin generated a field-evidence snapshot for pilot preparation and decision support.",
        "The package identifies real-data coverage, weak labels, drift and next collection steps.",
        "The output can help prepare Edge Impulse/TinyML workflows, but still requires deployment validation.",
    ]
    if decision == "FIELD-EVIDENCE READY":
        safe_claims.append("This project has enough field evidence for a controlled paid pilot-preparation handoff under the stated scope.")

    claims_to_avoid = [
        "Do not claim production accuracy or guaranteed fault detection from this gate alone.",
        "Do not claim regulatory, safety or insurance approval.",
        "Do not claim the synthetic data fully replaces field testing.",
        "Do not claim customer hardware is production-certified without independent validation.",
    ]

    next_actions = []
    if field_rows < min_rows:
        next_actions.append(f"Collect at least {min_rows - field_rows} additional field rows/files for {acceptance_level} evidence.")
    weak_labels = [r for r in label_evidence if r.get("field_samples", 0) < 20]
    for row in weak_labels[:5]:
        next_actions.append(f"Collect more field samples for weak label: {row.get('label')}")
    if not has_normal:
        next_actions.append("Capture clean normal/baseline data before anomaly or readiness claims.")
    if int(drift.get("similarity_score", 0)) < min_similarity:
        next_actions.append("Tune Synthetic-to-Real generation using the uploaded field fingerprint or collect more diverse real data.")
    if not next_actions:
        next_actions.append("Proceed to controlled pilot handoff and document field acceptance criteria with the customer.")

    field_trial_plan = [
        {"phase": "Baseline capture", "duration": "1-3 days", "goal": "Capture clean normal/baseline behavior across operating conditions."},
        {"phase": "Event capture", "duration": "3-14 days", "goal": "Collect labelled warning/fault/tamper/event evidence without mixing labels."},
        {"phase": "Edge export validation", "duration": "1-2 days", "goal": "Confirm Edge Impulse anomaly/classifier export and feature columns."},
        {"phase": "Pilot acceptance review", "duration": "1 day", "goal": "Review blockers, warnings, safe claims and final bundle scope."},
    ]

    snapshot = {
        "version": FIELD_EVIDENCE_V2_VERSION,
        "created_at": _now(),
        "project_name": project_name,
        "site_name": site_name,
        "evidence_goal": evidence_goal,
        "acceptance_level": acceptance_level,
        "target_customer": target_customer,
        "field_days": int(field_days or 0),
        "sensor_stack": sensor_stack,
        "summary": {
            "field_evidence_v2_score": score,
            "decision": decision,
            "evidence_grade": grade,
            "customer_status_line": customer_status_line,
            "field_rows": field_rows,
            "field_labels": label_count,
            "field_synthetic_similarity": int(drift.get("similarity_score", 0)),
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
        },
        "profiles": {"field": field_profile, "synthetic": synthetic_profile},
        "label_evidence": label_evidence,
        "field_vs_synthetic_drift": drift,
        "acceptance_checklist": acceptance_checklist,
        "evidence_dependencies": {
            "real_validation_score": real_validation_score,
            "reliability_score": reliability_score,
            "normality_score": normality_score,
            "customer_assurance_score": assurance_score,
            "deployment_score": deployment_score,
            "commercial_license_score": license_score,
            "edge_impulse_anomaly_decision": ei_anomaly_decision,
            "edge_impulse_classifier_decision": ei_classifier_decision,
        },
        "blockers": blockers,
        "warnings": warnings,
        "next_actions": next_actions,
        "field_trial_plan": field_trial_plan,
        "safe_customer_claims": safe_claims,
        "claims_to_avoid": claims_to_avoid,
        "disclaimer": FIELD_EVIDENCE_V2_DISCLAIMER,
    }
    return _json_safe(snapshot)


def generate_field_evidence_v2_pdf(project_name, snapshot):
    snap = snapshot or {}
    summary = snap.get("summary", {}) or {}
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt=clean_pdf_text("EdgeTwin Field Evidence 2.0 Gate"), ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, txt=clean_pdf_text("V39 - Real Field Validation Acceptance Evidence"), ln=True, align="C")
    pdf.ln(8)
    safe_pdf_cell(pdf, "Executive Status", 8, True)
    safe_pdf_cell(pdf, f"Project: {project_name}")
    safe_pdf_cell(pdf, f"Decision: {summary.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Evidence score: {summary.get('field_evidence_v2_score', 0)}%")
    safe_pdf_cell(pdf, f"Evidence grade: {summary.get('evidence_grade', 'Unknown')}")
    safe_pdf_multicell(pdf, summary.get("customer_status_line", ""))
    pdf.ln(4)
    safe_pdf_cell(pdf, "Field Evidence", 8, True)
    safe_pdf_cell(pdf, f"Site: {snap.get('site_name', '')}")
    safe_pdf_cell(pdf, f"Goal: {snap.get('evidence_goal', '')}")
    safe_pdf_cell(pdf, f"Acceptance level: {snap.get('acceptance_level', '')}")
    safe_pdf_cell(pdf, f"Field days: {snap.get('field_days', 0)}")
    safe_pdf_cell(pdf, f"Sensors: {', '.join(snap.get('sensor_stack', []))}")
    safe_pdf_cell(pdf, f"Field rows/classes: {summary.get('field_rows', 0)} / {summary.get('field_labels', 0)}")
    safe_pdf_cell(pdf, f"Field-synthetic similarity: {summary.get('field_synthetic_similarity', 0)}%")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Acceptance Checklist", 8, True)
    for item in snap.get("acceptance_checklist", []):
        status = "PASS" if item.get("passed") else "FAIL"
        safe_pdf_multicell(pdf, f"[{status}] {item.get('check')}: actual {item.get('actual')} / target {item.get('target')}")
    if snap.get("blockers"):
        pdf.ln(4)
        safe_pdf_cell(pdf, "Blockers", 8, True)
        for item in snap.get("blockers", []):
            safe_pdf_multicell(pdf, f"- {item}")
    if snap.get("warnings"):
        pdf.ln(4)
        safe_pdf_cell(pdf, "Warnings", 8, True)
        for item in snap.get("warnings", []):
            safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Next Actions", 8, True)
    for item in snap.get("next_actions", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Safe Customer Claims", 8, True)
    for item in snap.get("safe_customer_claims", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Claims To Avoid", 8, True)
    for item in snap.get("claims_to_avoid", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_cell(pdf, "Disclaimer", 8, True)
    safe_pdf_multicell(pdf, snap.get("disclaimer", FIELD_EVIDENCE_V2_DISCLAIMER))
    return safe_pdf_output(pdf)


def create_field_evidence_v2_bundle(project_name, snapshot, field_df=None, synthetic_df=None):
    snap = snapshot or {}
    pdf_bytes = generate_field_evidence_v2_pdf(project_name, snap)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("field_evidence_v2_report.pdf", pdf_bytes)
        zf.writestr("field_evidence_v2_snapshot.json", safe_json_dumps(snap, indent=2))
        if snap.get("acceptance_checklist"):
            zf.writestr("acceptance_checklist.csv", pd.DataFrame(snap.get("acceptance_checklist", [])).to_csv(index=False))
        if snap.get("label_evidence"):
            zf.writestr("label_evidence.csv", pd.DataFrame(snap.get("label_evidence", [])).to_csv(index=False))
        drift = snap.get("field_vs_synthetic_drift", {}) or {}
        if drift.get("top_drift_features"):
            zf.writestr("top_drift_features.csv", pd.DataFrame(drift.get("top_drift_features", [])).to_csv(index=False))
        if isinstance(field_df, pd.DataFrame) and len(field_df) > 0:
            zf.writestr("field_evidence_dataset_snapshot.csv", field_df.head(10000).to_csv(index=False))
        if isinstance(synthetic_df, pd.DataFrame) and len(synthetic_df) > 0:
            zf.writestr("synthetic_reference_snapshot.csv", synthetic_df.head(10000).to_csv(index=False))
        zf.writestr("next_actions.txt", "\n".join(snap.get("next_actions", [])))
        zf.writestr("safe_customer_claims.txt", "\n".join(snap.get("safe_customer_claims", [])))
        zf.writestr("claims_to_avoid.txt", "\n".join(snap.get("claims_to_avoid", [])))
        zf.writestr("README.md", f"""# EdgeTwin Studio V39 Field Evidence 2.0 Bundle

Project: {project_name}
Decision: {snap.get('summary', {}).get('decision', 'Unknown')}
Evidence score: {snap.get('summary', {}).get('field_evidence_v2_score', 0)}%

This bundle documents controlled field-evidence readiness for pilot preparation. It is not production certification.

{snap.get('disclaimer', FIELD_EVIDENCE_V2_DISCLAIMER)}
""")
    return zip_buf.getvalue()


# ============================================================
# V40 FULL PRODUCT READINESS GATE
# ============================================================

def _v40_num(value, default=0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _v40_get_score(snapshot, keys=None, default=0):
    keys = keys or [
        "score", "readiness_score", "trust_score", "trust_score_v2", "release_success_score",
        "operational_score", "safety_score", "governance_score", "workspace_score",
        "admin_readiness_score", "acceptance_score", "field_evidence_score",
        "onboarding_score", "scalability_score", "license_score", "customer_assurance_score",
    ]
    if not isinstance(snapshot, dict):
        return default
    for key in keys:
        if key in snapshot:
            return int(max(0, min(100, _v40_num(snapshot.get(key), default))))
    return default


def _v40_component(name, snapshot, weight, required=False, score_keys=None, evidence_hint=""):
    score = _v40_get_score(snapshot, score_keys, default=0)
    present = isinstance(snapshot, dict) and len(snapshot) > 0
    if not present:
        status = "missing_required" if required else "missing_optional"
        decision = "NO-GO" if required else "CONDITIONAL"
        message = evidence_hint or f"{name} evidence is missing."
        score = 0
    elif score >= 85:
        status = "strong"
        decision = "GO"
        message = f"{name} looks strong enough for the current readiness gate."
    elif score >= 70:
        status = "acceptable"
        decision = "CONDITIONAL"
        message = f"{name} is usable, but should be reviewed before paid delivery."
    elif score >= 50:
        status = "weak"
        decision = "CONDITIONAL"
        message = f"{name} is present but weak. Improve before claiming product readiness."
    else:
        status = "blocker" if required else "weak_optional"
        decision = "NO-GO" if required else "CONDITIONAL"
        message = f"{name} is below readiness threshold."
    return {
        "component": name,
        "present": bool(present),
        "required": bool(required),
        "weight": float(weight),
        "score": int(score),
        "weighted_score": float(score) * float(weight),
        "status": status,
        "decision": decision,
        "message": message,
    }


def build_full_product_readiness_gate(
    project_name="EdgeTwin_Project",
    dataset_df=None,
    selected_plan="Founder Test Mode",
    doctor=None,
    reliability_v2=None,
    trust_gate=None,
    hardening_snapshot=None,
    release_success_snapshot=None,
    operational_control_snapshot=None,
    observability_snapshot=None,
    customer_assurance_snapshot=None,
    onboarding_snapshot=None,
    workspace_lifecycle_snapshot=None,
    admin_usage_snapshot=None,
    commercial_license_certificate=None,
    field_evidence_v2_snapshot=None,
    deployment_plan=None,
    professional_report_snapshot=None,
    edge_impulse_snapshot=None,
    edge_impulse_classifier_snapshot=None,
    monetization_snapshot=None,
    scalability_snapshot=None,
):
    """Final V40 product-readiness gate.

    This does not claim production validation. It checks whether EdgeTwin is ready
    for controlled closed beta / paid pilot delivery with safe customer claims.
    """
    rows = int(len(dataset_df)) if isinstance(dataset_df, pd.DataFrame) else 0
    cols = int(len(dataset_df.columns)) if isinstance(dataset_df, pd.DataFrame) else 0
    labels = []
    if isinstance(dataset_df, pd.DataFrame) and "Label" in dataset_df.columns:
        try:
            labels = [str(x) for x in dataset_df["Label"].dropna().unique().tolist()]
        except Exception:
            labels = []
    label_count = len(labels)

    if doctor is None and isinstance(dataset_df, pd.DataFrame) and "Label" in dataset_df.columns and len(dataset_df) > 1:
        try:
            numeric = [c for c in dataset_df.columns if c != "Label" and pd.api.types.is_numeric_dtype(dataset_df[c])]
            if numeric:
                doctor = dataset_doctor(dataset_df[numeric], dataset_df["Label"])
        except Exception:
            doctor = None

    dataset_score = 0
    dataset_issues = []
    if rows <= 0:
        dataset_issues.append({"severity": "high", "message": "No active dataset is loaded."})
    if label_count < 2:
        dataset_issues.append({"severity": "high", "message": "Dataset needs at least two labels/classes for meaningful readiness."})
    if rows < 100:
        dataset_issues.append({"severity": "medium", "message": "Dataset is small. Use demo/private beta only."})
    if rows >= 100 and label_count >= 2:
        dataset_score = 70
        if rows >= 500:
            dataset_score += 10
        if label_count >= 4:
            dataset_score += 5
        if isinstance(doctor, dict):
            dataset_score = int(dataset_score * 0.45 + _v40_get_score(doctor, ["overall_score"], 0) * 0.55)
    dataset_score = int(max(0, min(100, dataset_score)))

    components = []
    components.append({
        "component": "Dataset Foundation",
        "present": rows > 0,
        "required": True,
        "weight": 0.11,
        "score": dataset_score,
        "weighted_score": dataset_score * 0.11,
        "status": "strong" if dataset_score >= 85 else "acceptable" if dataset_score >= 70 else "blocker",
        "decision": "GO" if dataset_score >= 80 else "CONDITIONAL" if dataset_score >= 60 else "NO-GO",
        "message": f"Dataset has {rows} rows, {cols} columns and {label_count} labels/classes.",
    })
    components.append(_v40_component("Reliability Engine 2.0", reliability_v2, 0.11, True, ["trust_score_v2", "reliability_score", "overall_reliability", "score"], "Reliability V2 evidence is required before product-ready claims."))
    components.append(_v40_component("Trust Center", trust_gate, 0.08, True, ["trust_score", "score", "readiness_score"], "Trust Center result is required."))
    components.append(_v40_component("Product Hardening", hardening_snapshot, 0.10, True, ["product_readiness_score", "hardening_score", "score", "readiness_score"], "Hardening snapshot is required before beta/paid delivery."))
    components.append(_v40_component("Release Success Gate", release_success_snapshot, 0.10, True, ["release_success_score", "score"], "Release Success Gate must be run before customer delivery."))
    components.append(_v40_component("Operational Control", operational_control_snapshot, 0.08, True, ["operational_score", "score"], "Operational Control snapshot is required."))
    components.append(_v40_component("Error Observatory / Safety", observability_snapshot, 0.07, True, ["safety_score", "score"], "Error Observatory is required for customer-facing beta."))
    components.append(_v40_component("Customer Assurance & Governance", customer_assurance_snapshot, 0.08, True, ["governance_score", "customer_assurance_score", "score"], "Customer data assurance is required when customers upload real files."))
    components.append(_v40_component("Guided Success / Onboarding", onboarding_snapshot, 0.06, True, ["onboarding_score", "score"], "Guided Success flow reduces customer confusion and support load."))
    components.append(_v40_component("Workspace Lifecycle", workspace_lifecycle_snapshot, 0.05, False, ["workspace_score", "score"], "Workspace lifecycle is recommended for paid pilots."))
    components.append(_v40_component("Admin / Usage Tracking", admin_usage_snapshot, 0.05, False, ["admin_readiness_score", "score"], "Admin usage tracking is recommended before multiple testers."))
    components.append(_v40_component("Commercial License Certificate", commercial_license_certificate, 0.04, False, ["license_score", "score"], "License certificate is recommended for paid delivery."))
    components.append(_v40_component("Field Evidence 2.0", field_evidence_v2_snapshot, 0.05, False, ["acceptance_score", "field_evidence_score", "score"], "Field evidence is required before production-oriented claims."))
    components.append(_v40_component("Deployment Planner", deployment_plan, 0.04, False, ["deployment_score", "score"], "Deployment plan improves customer decision confidence."))
    components.append(_v40_component("Professional Reports 2.0", professional_report_snapshot, 0.04, False, ["report_score", "score", "readiness_score"], "Professional report is recommended for paid pilot bundles."))
    components.append(_v40_component("Edge Impulse Export Coverage", edge_impulse_snapshot or edge_impulse_classifier_snapshot, 0.03, False, ["export_score", "readiness_score", "score"], "Edge Impulse export is recommended for TinyML continuation."))
    components.append(_v40_component("Monetization / Paid Export Logic", monetization_snapshot, 0.03, False, ["monetization_score", "score", "readiness_score"], "Paid export logic is recommended before charging customers."))
    components.append(_v40_component("Storage / Scalability", scalability_snapshot, 0.03, False, ["scalability_score", "score"], "Storage/scalability is required before larger datasets or multiple customers."))

    total_weight = sum(c["weight"] for c in components) or 1.0
    final_score = int(max(0, min(100, sum(c["weighted_score"] for c in components) / total_weight)))

    blockers = []
    warnings = []
    for item in dataset_issues:
        if item["severity"] == "high":
            blockers.append(item["message"])
        else:
            warnings.append(item["message"])
    for c in components:
        if c["required"] and (not c["present"] or c["decision"] == "NO-GO"):
            blockers.append(c["message"])
        elif c["decision"] == "CONDITIONAL":
            warnings.append(c["message"])

    has_paid_delivery_evidence = bool(commercial_license_certificate) and bool(customer_assurance_snapshot) and bool(release_success_snapshot)
    has_field_evidence = bool(field_evidence_v2_snapshot)
    has_real_or_field = bool(has_field_evidence)

    if blockers:
        decision = "NO-GO"
        commercial_status = "Internal only"
        readiness_level = "Not product-ready"
    elif final_score >= 88 and has_paid_delivery_evidence and has_field_evidence:
        decision = "GO"
        commercial_status = "Paid pilot ready"
        readiness_level = "Product-ready for controlled paid pilots"
    elif final_score >= 82 and has_paid_delivery_evidence:
        decision = "GO"
        commercial_status = "Closed beta / paid review candidate"
        readiness_level = "Product-ready for controlled closed beta"
    elif final_score >= 72:
        decision = "CONDITIONAL GO"
        commercial_status = "Private beta only"
        readiness_level = "Beta-ready with conditions"
    else:
        decision = "NO-GO"
        commercial_status = "Internal demo only"
        readiness_level = "Not product-ready"

    safe_claims = [
        "EdgeTwin Studio can prepare Edge AI pilot packages with dataset, audit, reliability, hardware and reporting support.",
        "Outputs are intended for concept validation, demo preparation and controlled pilot planning.",
        "Reliability scores are decision-support estimates, not production certification.",
        "Field validation is required before operational or safety-critical deployment.",
    ]
    if has_real_or_field:
        safe_claims.append("This project includes real/field evidence that can support stronger pilot-readiness discussion.")
    claims_to_avoid = [
        "Do not claim guaranteed production accuracy.",
        "Do not claim the synthetic dataset replaces real-world validation.",
        "Do not claim safety certification or regulatory approval.",
        "Do not promise autonomous deployment decisions without customer field testing.",
    ]
    next_actions = []
    if blockers:
        next_actions.extend(blockers[:5])
    else:
        if not has_field_evidence:
            next_actions.append("Collect or upload real field data and run Field Evidence 2.0 before stronger paid-pilot claims.")
        if not commercial_license_certificate:
            next_actions.append("Issue a Commercial License Certificate before paid bundle delivery.")
        if not admin_usage_snapshot:
            next_actions.append("Run Admin/Usage tracking before adding multiple beta testers.")
        if not scalability_snapshot:
            next_actions.append("Run Storage/Scalability check before accepting large customer datasets.")
        if not next_actions:
            next_actions.append("Proceed with controlled beta/pilot delivery using the safe claims and limitations in this bundle.")

    return {
        "version": "V40 Full Product Readiness Gate",
        "project_name": project_name,
        "created_at": _now(),
        "selected_plan": selected_plan,
        "dataset_rows": rows,
        "dataset_cols": cols,
        "labels": labels,
        "label_count": label_count,
        "final_product_readiness_score": final_score,
        "decision": decision,
        "commercial_status": commercial_status,
        "readiness_level": readiness_level,
        "components": components,
        "blockers": blockers,
        "warnings": warnings[:12],
        "safe_claims": safe_claims,
        "claims_to_avoid": claims_to_avoid,
        "next_actions": next_actions,
        "customer_summary": f"{project_name}: {readiness_level} ({decision}) with score {final_score}/100. Field validation remains required before production deployment.",
        "internal_summary": "V40 is the consolidated readiness gate: it checks technical, trust, governance, operational, commercial and field-evidence readiness.",
    }


def generate_full_product_readiness_pdf(project_name, snapshot):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Studio V40 Product Readiness Gate", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 7, txt=clean_pdf_text("Powered by OMEGA-X Engine - controlled pilot readiness, not production certification"), ln=True, align="C")
    pdf.ln(6)

    safe_pdf_cell(pdf, "Executive Readiness Decision", 8, True)
    safe_pdf_cell(pdf, f"Project: {project_name}")
    safe_pdf_cell(pdf, f"Score: {snapshot.get('final_product_readiness_score', 0)}/100")
    safe_pdf_cell(pdf, f"Decision: {snapshot.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Commercial status: {snapshot.get('commercial_status', 'Unknown')}")
    safe_pdf_multicell(pdf, snapshot.get("customer_summary", ""))
    pdf.ln(3)

    safe_pdf_cell(pdf, "Dataset", 8, True)
    safe_pdf_cell(pdf, f"Rows: {snapshot.get('dataset_rows', 0)}")
    safe_pdf_cell(pdf, f"Columns: {snapshot.get('dataset_cols', 0)}")
    safe_pdf_cell(pdf, f"Labels/classes: {snapshot.get('label_count', 0)}")

    pdf.ln(3)
    safe_pdf_cell(pdf, "Component Readiness", 8, True)
    for comp in snapshot.get("components", []):
        safe_pdf_multicell(pdf, f"- {comp.get('component')}: {comp.get('score')}% | {comp.get('decision')} | {comp.get('status')}")

    if snapshot.get("blockers"):
        pdf.ln(3)
        safe_pdf_cell(pdf, "Must-fix blockers", 8, True)
        for item in snapshot.get("blockers", []):
            safe_pdf_multicell(pdf, f"- {item}")

    if snapshot.get("warnings"):
        pdf.ln(3)
        safe_pdf_cell(pdf, "Warnings", 8, True)
        for item in snapshot.get("warnings", []):
            safe_pdf_multicell(pdf, f"- {item}")

    pdf.ln(3)
    safe_pdf_cell(pdf, "Safe customer claims", 8, True)
    for item in snapshot.get("safe_claims", []):
        safe_pdf_multicell(pdf, f"- {item}")

    pdf.ln(3)
    safe_pdf_cell(pdf, "Claims to avoid", 8, True)
    for item in snapshot.get("claims_to_avoid", []):
        safe_pdf_multicell(pdf, f"- {item}")

    pdf.ln(3)
    safe_pdf_cell(pdf, "Next actions", 8, True)
    for item in snapshot.get("next_actions", []):
        safe_pdf_multicell(pdf, f"- {item}")

    return safe_pdf_output(pdf)


def create_full_product_readiness_bundle(project_name, snapshot, dataset_df=None):
    zip_buf = io.BytesIO()
    components_df = pd.DataFrame(snapshot.get("components", []))
    blockers_df = pd.DataFrame([{"blocker": x} for x in snapshot.get("blockers", [])])
    warnings_df = pd.DataFrame([{"warning": x} for x in snapshot.get("warnings", [])])
    claims_df = pd.DataFrame(
        [{"type": "safe_claim", "claim": x} for x in snapshot.get("safe_claims", [])] +
        [{"type": "avoid", "claim": x} for x in snapshot.get("claims_to_avoid", [])]
    )
    pdf_bytes = generate_full_product_readiness_pdf(project_name, snapshot)
    readme = f"""EdgeTwin Studio V40 Product Readiness Bundle

Project: {project_name}
Decision: {snapshot.get('decision')}
Score: {snapshot.get('final_product_readiness_score')}/100
Commercial status: {snapshot.get('commercial_status')}

This is a product-readiness decision-support bundle. It does not certify production deployment.
Field validation remains required before production or safety-critical use.
"""
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("product_readiness_v40_report.pdf", pdf_bytes)
        zf.writestr("product_readiness_v40_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr("component_readiness_matrix.csv", components_df.to_csv(index=False))
        zf.writestr("blockers.csv", blockers_df.to_csv(index=False))
        zf.writestr("warnings.csv", warnings_df.to_csv(index=False))
        zf.writestr("safe_claims_matrix.csv", claims_df.to_csv(index=False))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr("dataset_snapshot.csv", dataset_df.head(5000).to_csv(index=False))
        zf.writestr("README.txt", readme)
    return zip_buf.getvalue()


# ============================================================
# V41 SECURITY & ACCESS CONTROL HARDENING
# ============================================================

SECURITY_ROLES = {
    "Founder / Admin": {
        "description": "Full local operator access. Intended for the product owner during private beta.",
        "risk": "medium",
        "allowed_actions": ["generate", "audit", "export", "admin", "license", "delete_local"],
    },
    "Customer Admin": {
        "description": "Customer-side admin who can create projects, upload data and download paid bundles.",
        "risk": "medium",
        "allowed_actions": ["generate", "audit", "export", "license"],
    },
    "Engineer": {
        "description": "Technical user who can generate datasets, upload files and inspect analysis results.",
        "risk": "low-medium",
        "allowed_actions": ["generate", "audit", "export_review"],
    },
    "Viewer / Reviewer": {
        "description": "Read-only reviewer for reports, scores and decision gates.",
        "risk": "low",
        "allowed_actions": ["view", "download_report_only"],
    },
}

SECURITY_FILE_POLICIES = {
    "strict_beta": {
        "allowed_extensions": ["csv", "wav"],
        "max_upload_mb": 50,
        "description": "Recommended for first customer beta. Only CSV/WAV sensor files are allowed.",
    },
    "professional_pilot": {
        "allowed_extensions": ["csv", "wav", "json", "zip"],
        "max_upload_mb": 150,
        "description": "Allows bundled project evidence while still blocking unknown file types.",
    },
    "enterprise_review": {
        "allowed_extensions": ["csv", "wav", "json", "zip", "txt", "md"],
        "max_upload_mb": 500,
        "description": "Larger review package for trusted enterprise pilots.",
    },
}


def get_security_roles():
    return list(SECURITY_ROLES.keys())


def get_security_file_policies():
    return list(SECURITY_FILE_POLICIES.keys())


def _security_issue(severity, message, control="general"):
    return {"severity": severity, "control": control, "message": message}


def assess_security_access_control(
    user=None,
    plan_name="Founder Test Mode",
    dataset_df=None,
    role="Founder / Admin",
    file_policy="strict_beta",
    public_demo_mode=False,
    enforce_plan_limits=True,
    require_login=True,
    require_customer_assurance=True,
    signed_exports=True,
    allow_audio_uploads=True,
    retention_days=30,
):
    """Create a practical V41 access/security snapshot for closed-beta and paid-pilot use.

    This is intentionally a product safety gate, not a formal security certification.
    It checks whether the app is being operated with sane limits, roles, upload policy,
    governance warnings and bundle controls before external customer use.
    """
    role_info = SECURITY_ROLES.get(role, SECURITY_ROLES["Founder / Admin"])
    policy = SECURITY_FILE_POLICIES.get(file_policy, SECURITY_FILE_POLICIES["strict_beta"])
    rows = int(len(dataset_df)) if isinstance(dataset_df, pd.DataFrame) else 0
    cols = int(len(dataset_df.columns)) if isinstance(dataset_df, pd.DataFrame) else 0
    plan_limit = get_sample_limit_for_plan(plan_name) if "get_sample_limit_for_plan" in globals() else 500
    issues = []
    controls = []

    def add_control(name, status, evidence, severity="info"):
        controls.append({"control": name, "status": status, "severity": severity, "evidence": evidence})

    if require_login:
        add_control("Authenticated access", "enabled", "Login/account gate is required before application access.")
    else:
        issues.append(_security_issue("high", "Login is not required. Do not expose customer data or paid exports in this mode.", "auth"))
        add_control("Authenticated access", "disabled", "Public access is enabled.", "high")

    if public_demo_mode:
        issues.append(_security_issue("medium", "Public demo mode should only expose synthetic/demo data and watermarked outputs.", "public_demo"))
        add_control("Public demo protection", "conditional", "Public demo mode selected. Paid exports should remain locked.", "medium")
    else:
        add_control("Public demo protection", "private", "Private/operator/customer workspace mode selected.")

    if role == "Founder / Admin" and public_demo_mode:
        issues.append(_security_issue("high", "Founder/Admin role should not be used for public customer demos.", "role"))
    add_control("Role scope", role, f"Allowed actions: {', '.join(role_info.get('allowed_actions', []))}", "info")

    if enforce_plan_limits:
        add_control("Plan limits", "enabled", f"Plan '{plan_name}' limit is {plan_limit} samples.")
        if rows > plan_limit:
            issues.append(_security_issue("medium", f"Current dataset has {rows} rows and exceeds the {plan_name} sample limit of {plan_limit}.", "plan_limits"))
    else:
        issues.append(_security_issue("medium", "Plan limits are not enforced. Keep this mode internal only.", "plan_limits"))
        add_control("Plan limits", "disabled", "Generation/download caps are advisory only.", "medium")

    if policy.get("max_upload_mb", 0) > 250 and plan_name not in {"Enterprise", "Founder Test Mode"}:
        issues.append(_security_issue("medium", "Upload policy allows large files for a non-enterprise plan.", "upload"))
    add_control("Upload policy", file_policy, f"Allowed: {', '.join(policy.get('allowed_extensions', []))}; max {policy.get('max_upload_mb')} MB")

    if allow_audio_uploads:
        issues.append(_security_issue("info", "Audio uploads may contain privacy-sensitive content. Show customer notice and retention policy.", "privacy"))
        add_control("Audio privacy notice", "required", "Audio/WAV uploads are allowed; customer notice should be visible.", "info")
    else:
        add_control("Audio privacy notice", "not_required", "Audio uploads disabled for this workspace.")

    if not require_customer_assurance:
        issues.append(_security_issue("medium", "Customer assurance/governance confirmation is disabled before upload/export.", "governance"))
        add_control("Customer assurance", "disabled", "Governance confirmation skipped.", "medium")
    else:
        add_control("Customer assurance", "enabled", "Customer upload notice, ownership and safe-use terms are required.")

    if not signed_exports:
        issues.append(_security_issue("medium", "Exports are not signed/certified. Paid bundles should include certificate/hash metadata.", "exports"))
        add_control("Signed export/certificate", "disabled", "Commercial certificate not attached.", "medium")
    else:
        add_control("Signed export/certificate", "enabled", "Paid exports should include certificate ID/hash and license scope.")

    if int(retention_days or 0) > 90 and plan_name not in {"Enterprise", "Founder Test Mode"}:
        issues.append(_security_issue("medium", "Retention period is long for non-enterprise pilots. Prefer shorter retention unless contracted.", "retention"))
    elif int(retention_days or 0) <= 0:
        issues.append(_security_issue("high", "Retention period must be explicit before accepting customer uploads.", "retention"))
    add_control("Retention policy", f"{retention_days} days", "Customer data retention is explicitly configured.")

    # Basic dependency/platform reminders.
    controls.append({
        "control": "Secrets handling",
        "status": "required_for_cloud",
        "severity": "medium",
        "evidence": "Use Streamlit secrets/environment variables for API keys/payment secrets. Do not hardcode production secrets.",
    })
    controls.append({
        "control": "Transport security",
        "status": "required_for_public_use",
        "severity": "medium",
        "evidence": "Use HTTPS and trusted hosting before public paid access.",
    })

    score = 100
    for item in issues:
        sev = item.get("severity")
        if sev == "high":
            score -= 28
        elif sev == "medium":
            score -= 12
        elif sev == "warning":
            score -= 8
    if require_login and enforce_plan_limits and signed_exports and require_customer_assurance:
        score += 5
    score = int(np.clip(score, 0, 100))

    high = [x for x in issues if x.get("severity") == "high"]
    medium = [x for x in issues if x.get("severity") == "medium"]
    if high:
        decision = "NO-GO"
        status = "Internal only until access/security blockers are fixed"
    elif score >= 85 and len(medium) <= 1:
        decision = "GO"
        status = "Ready for controlled closed beta / paid pilot access"
    elif score >= 68:
        decision = "CONDITIONAL GO"
        status = "Usable for controlled beta with operator review"
    else:
        decision = "NO-GO"
        status = "Security hardening required before customer access"

    safe_claims = [
        "Access scope, upload policy and retention have been reviewed for this pilot workspace.",
        "Paid exports can be delivered with license/certificate metadata and safe-use terms.",
        "Customer data handling is treated as pilot-governance controlled, not open public upload.",
    ]
    claims_to_avoid = [
        "Do not claim the platform is fully certified or enterprise-security audited.",
        "Do not promise production deployment safety based only on synthetic or beta data.",
        "Do not reuse customer uploaded data for product improvement without explicit permission.",
    ]
    next_actions = []
    if high or medium:
        next_actions.extend([x.get("message") for x in high + medium][:6])
    else:
        next_actions.extend([
            "Keep V41 controls enabled for all beta/customer workspaces.",
            "Use commercial license certificates for paid bundle delivery.",
            "Run Field Evidence and Product Ready gates before production-positioned claims.",
        ])

    return {
        "version": "V41 Security & Access Control Hardening",
        "created_at": _now() if "_now" in globals() else str(datetime.datetime.now()),
        "decision": decision,
        "security_score": score,
        "status": status,
        "role": role,
        "role_description": role_info.get("description", ""),
        "plan": plan_name,
        "dataset_rows": rows,
        "dataset_cols": cols,
        "sample_limit": plan_limit,
        "file_policy": {"name": file_policy, **policy},
        "settings": {
            "public_demo_mode": bool(public_demo_mode),
            "enforce_plan_limits": bool(enforce_plan_limits),
            "require_login": bool(require_login),
            "require_customer_assurance": bool(require_customer_assurance),
            "signed_exports": bool(signed_exports),
            "allow_audio_uploads": bool(allow_audio_uploads),
            "retention_days": int(retention_days or 0),
        },
        "controls": controls,
        "issues": issues,
        "safe_claims": safe_claims,
        "claims_to_avoid": claims_to_avoid,
        "next_actions": next_actions,
        "disclaimer": "V41 is a practical access/security readiness gate, not a formal penetration test or security certification.",
    }


def generate_security_hardening_pdf(project_name, snapshot):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Studio V41 Security & Access Control", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(6)
    safe_pdf_cell(pdf, "Decision Summary", 8, True)
    safe_pdf_cell(pdf, f"Decision: {snapshot.get('decision')}")
    safe_pdf_cell(pdf, f"Security Score: {snapshot.get('security_score')}/100")
    safe_pdf_cell(pdf, f"Status: {snapshot.get('status')}")
    safe_pdf_cell(pdf, f"Role: {snapshot.get('role')}")
    safe_pdf_cell(pdf, f"Plan: {snapshot.get('plan')}")
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", ""))

    pdf.ln(4)
    safe_pdf_cell(pdf, "Access & Upload Controls", 8, True)
    for item in snapshot.get("controls", []):
        safe_pdf_multicell(pdf, f"[{str(item.get('severity','info')).upper()}] {item.get('control')}: {item.get('status')} - {item.get('evidence')}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Issues", 8, True)
    if snapshot.get("issues"):
        for item in snapshot.get("issues", []):
            safe_pdf_multicell(pdf, f"[{str(item.get('severity','info')).upper()}] {item.get('message')}")
    else:
        safe_pdf_multicell(pdf, "No blocking access/security issues detected for controlled beta use.")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Safe Claims", 8, True)
    for item in snapshot.get("safe_claims", []):
        safe_pdf_multicell(pdf, f"- {item}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Claims To Avoid", 8, True)
    for item in snapshot.get("claims_to_avoid", []):
        safe_pdf_multicell(pdf, f"- {item}")

    pdf.ln(4)
    safe_pdf_cell(pdf, "Next Actions", 8, True)
    for item in snapshot.get("next_actions", []):
        safe_pdf_multicell(pdf, f"- {item}")

    return safe_pdf_output(pdf)


def create_security_hardening_bundle(project_name, snapshot, dataset_df=None):
    zip_buf = io.BytesIO()
    controls_df = pd.DataFrame(snapshot.get("controls", []))
    issues_df = pd.DataFrame(snapshot.get("issues", []))
    claims_df = pd.DataFrame(
        [{"type": "safe_claim", "claim": x} for x in snapshot.get("safe_claims", [])] +
        [{"type": "avoid", "claim": x} for x in snapshot.get("claims_to_avoid", [])]
    )
    pdf_bytes = generate_security_hardening_pdf(project_name, snapshot)
    readme = f"""EdgeTwin Studio V41 Security & Access Control Bundle

Project: {project_name}
Decision: {snapshot.get('decision')}
Security score: {snapshot.get('security_score')}/100
Role: {snapshot.get('role')}
Plan: {snapshot.get('plan')}

This bundle is a practical security/access readiness snapshot for private beta and paid pilot delivery.
It is not a formal penetration test, ISO certification or legal security audit.
"""
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("security_access_v41_report.pdf", pdf_bytes)
        zf.writestr("security_access_v41_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr("security_controls.csv", controls_df.to_csv(index=False))
        zf.writestr("security_issues.csv", issues_df.to_csv(index=False))
        zf.writestr("security_claims.csv", claims_df.to_csv(index=False))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr("dataset_snapshot.csv", dataset_df.head(5000).to_csv(index=False))
        zf.writestr("README.txt", readme)
    return zip_buf.getvalue()


# ============================================================
# V42 CUSTOMER DELIVERY PORTAL / DELIVERABLES CENTER
# ============================================================

DELIVERY_PORTAL_DISCLAIMER = (
    "Customer delivery packages are pilot-preparation deliverables. They are not production certification, "
    "safety certification or a guarantee of model performance. Field validation and customer acceptance remain required."
)

DELIVERY_PACKAGE_LEVELS = {
    "Review / Demo Delivery": {
        "required": ["Professional Report", "Customer Assurance", "Guided Success"],
        "recommended": ["Golden Demo", "Product Readiness Gate"],
        "price_position": "free-review or internal demo",
    },
    "Starter Pilot Delivery": {
        "required": ["Pilot Dataset", "Dataset Audit", "Professional Report", "Customer Assurance"],
        "recommended": ["Trust Center", "Deployment Plan", "License Certificate"],
        "price_position": "starter paid bundle",
    },
    "Professional Pilot Delivery": {
        "required": ["Pilot Dataset", "Dataset Audit", "Reliability 2.0", "Professional Report", "Deployment Plan", "Customer Assurance", "License Certificate"],
        "recommended": ["Edge Impulse Export", "Normality Engine", "Product Readiness Gate", "Admin Usage"],
        "price_position": "professional paid pilot bundle",
    },
    "Real-Data Pilot Delivery": {
        "required": ["Real Bridge", "Reliability 2.0", "Field Evidence", "Customer Assurance", "Professional Report", "License Certificate"],
        "recommended": ["Normality Engine", "Edge Impulse Export", "Deployment Plan", "Product Readiness Gate"],
        "price_position": "real-data paid pilot bundle",
    },
    "Enterprise Review Delivery": {
        "required": ["Real Bridge", "Reliability 2.0", "Field Evidence", "Product Readiness Gate", "Security Gate", "Customer Assurance", "Commercial License Certificate"],
        "recommended": ["Operational Control", "Error Observatory", "Admin Usage", "Workspace Lifecycle"],
        "price_position": "enterprise/custom review",
    },
}


def get_delivery_package_levels():
    return list(DELIVERY_PACKAGE_LEVELS.keys())


def _delivery_bool(bundle_flags, *names):
    for name in names:
        if bool(bundle_flags.get(name)):
            return True
    return False


def build_customer_delivery_portal(
    project_name,
    customer_name="Beta Customer",
    customer_organization="Customer Organization",
    customer_email="customer@example.com",
    selected_plan="Founder Test Mode",
    delivery_level="Professional Pilot Delivery",
    dataset_df=None,
    bundle_flags=None,
    professional_report_snapshot=None,
    customer_assurance_snapshot=None,
    commercial_license_certificate=None,
    product_readiness_snapshot=None,
    field_evidence_snapshot=None,
    security_snapshot=None,
    workspace_lifecycle_snapshot=None,
    admin_usage_snapshot=None,
    notes="",
):
    """Build a single customer-facing delivery status/manifest.

    The portal is intentionally conservative: it checks evidence and deliverables before a customer handoff.
    """
    bundle_flags = bundle_flags or {}
    level_cfg = DELIVERY_PACKAGE_LEVELS.get(delivery_level, DELIVERY_PACKAGE_LEVELS["Professional Pilot Delivery"])
    rows = int(len(dataset_df)) if isinstance(dataset_df, pd.DataFrame) else 0
    cols = int(len(dataset_df.columns)) if isinstance(dataset_df, pd.DataFrame) else 0
    has_label = bool(isinstance(dataset_df, pd.DataFrame) and "Label" in dataset_df.columns)

    deliverable_map = {
        "Pilot Dataset": rows > 0,
        "Dataset Audit": _delivery_bool(bundle_flags, "Enterprise Bundle", "Fusion Bundle", "Auto Pilot Bundle") or has_label,
        "Professional Report": _delivery_bool(bundle_flags, "Professional Report", "Enterprise Bundle"),
        "Deployment Plan": _delivery_bool(bundle_flags, "Deployment Bundle", "Edge Deployment Starter"),
        "Trust Center": _delivery_bool(bundle_flags, "Trust Bundle") or bool(bundle_flags.get("Trust Snapshot")),
        "Reliability 2.0": _delivery_bool(bundle_flags, "Reliability Bundle") or bool(bundle_flags.get("Reliability Snapshot")),
        "Real Bridge": _delivery_bool(bundle_flags, "Real Bridge Bundle") or bool(bundle_flags.get("Real Bridge Snapshot")),
        "Field Evidence": _delivery_bool(bundle_flags, "Field Evidence Bundle", "Field Validation Bundle") or bool(field_evidence_snapshot),
        "Normality Engine": _delivery_bool(bundle_flags, "Normality Bundle"),
        "Edge Impulse Export": _delivery_bool(bundle_flags, "Edge Impulse Anomaly Bundle", "Edge Impulse Classifier Bundle"),
        "Customer Assurance": _delivery_bool(bundle_flags, "Customer Assurance Bundle") or bool(customer_assurance_snapshot),
        "Guided Success": _delivery_bool(bundle_flags, "Guided Success Bundle"),
        "Golden Demo": _delivery_bool(bundle_flags, "Golden Demo Bundle"),
        "Product Readiness Gate": _delivery_bool(bundle_flags, "Product Readiness Bundle") or bool(product_readiness_snapshot),
        "Security Gate": _delivery_bool(bundle_flags, "Security Bundle") or bool(security_snapshot),
        "Commercial License Certificate": _delivery_bool(bundle_flags, "Commercial License Certificate") or bool(commercial_license_certificate),
        "License Certificate": _delivery_bool(bundle_flags, "Commercial License Certificate", "Paid License Bundle") or bool(commercial_license_certificate),
        "Operational Control": _delivery_bool(bundle_flags, "Operational Control Bundle"),
        "Error Observatory": _delivery_bool(bundle_flags, "Production Safety Bundle"),
        "Admin Usage": _delivery_bool(bundle_flags, "Admin Usage Bundle") or bool(admin_usage_snapshot),
        "Workspace Lifecycle": _delivery_bool(bundle_flags, "Workspace Lifecycle Bundle") or bool(workspace_lifecycle_snapshot),
    }

    required = level_cfg.get("required", [])
    recommended = level_cfg.get("recommended", [])
    required_status = []
    recommended_status = []
    blockers = []
    warnings = []

    for item in required:
        ready = bool(deliverable_map.get(item, False))
        required_status.append({"deliverable": item, "required": True, "ready": ready, "status": "READY" if ready else "MISSING"})
        if not ready:
            blockers.append(f"Required deliverable missing: {item}")

    for item in recommended:
        ready = bool(deliverable_map.get(item, False))
        recommended_status.append({"deliverable": item, "required": False, "ready": ready, "status": "READY" if ready else "RECOMMENDED"})
        if not ready:
            warnings.append(f"Recommended deliverable not ready yet: {item}")

    if rows == 0:
        blockers.append("No active dataset is loaded for customer delivery.")
    elif rows < 100 and delivery_level not in ["Review / Demo Delivery"]:
        warnings.append("Dataset is small for a paid pilot handoff; customer should treat this as pilot preparation only.")
    if not has_label and rows > 0:
        warnings.append("Dataset has no Label column; supervised/classifier delivery may be limited.")

    assurance_score = 0
    if isinstance(customer_assurance_snapshot, dict):
        assurance_score = int(customer_assurance_snapshot.get("governance_score", customer_assurance_snapshot.get("assurance_score", 0)) or 0)
        if assurance_score and assurance_score < 65:
            warnings.append("Customer assurance/governance score is below ideal delivery level.")

    product_score = 0
    if isinstance(product_readiness_snapshot, dict):
        product_score = int(product_readiness_snapshot.get("final_product_readiness_score", product_readiness_snapshot.get("product_readiness_score", 0)) or 0)
        if product_score and product_score < 65:
            warnings.append("Product readiness gate is below ideal customer delivery level.")

    license_status = "not_issued"
    if isinstance(commercial_license_certificate, dict):
        license_status = commercial_license_certificate.get("certificate_status", "issued")
        if delivery_level not in ["Review / Demo Delivery"] and license_status not in ["ACTIVE", "REVIEW_REQUIRED"]:
            warnings.append("Commercial license certificate exists but is not active/review-ready.")
    elif delivery_level not in ["Review / Demo Delivery", "Starter Pilot Delivery"]:
        blockers.append("Commercial license certificate is required before professional/enterprise delivery.")

    required_ready = sum(1 for row in required_status if row["ready"])
    required_total = max(1, len(required_status))
    recommended_ready = sum(1 for row in recommended_status if row["ready"])
    recommended_total = max(1, len(recommended_status))
    base_score = int((required_ready / required_total) * 70 + (recommended_ready / recommended_total) * 20)
    data_bonus = 10 if rows >= 100 and has_label else (5 if rows > 0 else 0)
    score = int(np.clip(base_score + data_bonus, 0, 100))
    if blockers:
        score = min(score, 64)
    elif warnings:
        score = min(max(score, 65), 84)

    if score >= 85 and not blockers:
        decision = "GO"
        portal_status = "CUSTOMER DELIVERY READY"
        customer_status_line = "This package is ready for controlled customer delivery as a pilot-preparation handoff."
    elif score >= 65 and not any("No active dataset" in b for b in blockers):
        decision = "CONDITIONAL GO"
        portal_status = "DELIVERY REVIEW REQUIRED"
        customer_status_line = "This package can be reviewed with the customer, but listed blockers/warnings must be handled before paid or formal delivery."
    else:
        decision = "NO-GO"
        portal_status = "NOT READY FOR CUSTOMER DELIVERY"
        customer_status_line = "This package should not be delivered to a customer until the blockers are fixed."

    delivery_checklist = [
        {"step": "Confirm customer scope and use-case", "status": "required"},
        {"step": "Confirm uploaded data ownership and retention policy", "status": "required"},
        {"step": "Generate/report all required bundles for selected delivery level", "status": "required"},
        {"step": "Review safe claims and claims to avoid", "status": "required"},
        {"step": "Issue license certificate for paid/professional delivery", "status": "required_for_paid"},
        {"step": "Send delivery bundle and request acceptance feedback", "status": "handoff"},
    ]

    safe_claims = [
        "EdgeTwin Studio prepares Edge AI sensor pilot packages with datasets, audits, reliability estimates and hardware/deployment guidance.",
        "The delivery is suitable for concept, demo or pilot-preparation according to the selected package level.",
        "Field validation is required before production deployment.",
    ]
    claims_to_avoid = [
        "Do not claim production certification.",
        "Do not claim guaranteed model accuracy from synthetic data alone.",
        "Do not claim safety-critical readiness without formal validation and customer acceptance.",
    ]

    handoff_steps = [
        "Share the Customer Delivery Bundle with the customer.",
        "Walk through the executive summary, risk notes and safe-use terms.",
        "Confirm whether the next step is beta feedback, paid pilot, field validation or custom review.",
        "Log the delivery event in Admin/Usage after handoff.",
    ]

    all_deliverables = required_status + recommended_status
    # Add optional deliverables that are already available but not part of the selected level.
    for name, ready in deliverable_map.items():
        if ready and name not in {r["deliverable"] for r in all_deliverables}:
            all_deliverables.append({"deliverable": name, "required": False, "ready": True, "status": "OPTIONAL_READY"})

    return {
        "version": "V42 Customer Delivery Portal",
        "created_at": _now(),
        "project_name": project_name,
        "customer": {
            "name": customer_name,
            "organization": customer_organization,
            "email": customer_email,
        },
        "selected_plan": selected_plan,
        "delivery_level": delivery_level,
        "price_position": level_cfg.get("price_position"),
        "dataset_rows": rows,
        "dataset_cols": cols,
        "dataset_has_label": has_label,
        "delivery_score": score,
        "decision": decision,
        "portal_status": portal_status,
        "customer_status_line": customer_status_line,
        "license_status": license_status,
        "product_readiness_score": product_score,
        "customer_assurance_score": assurance_score,
        "required_status": required_status,
        "recommended_status": recommended_status,
        "deliverables": all_deliverables,
        "blockers": blockers,
        "warnings": warnings,
        "delivery_checklist": delivery_checklist,
        "safe_claims": safe_claims,
        "claims_to_avoid": claims_to_avoid,
        "handoff_steps": handoff_steps,
        "notes": notes,
        "disclaimer": DELIVERY_PORTAL_DISCLAIMER,
    }


def create_customer_delivery_portal_bundle(project_name, delivery_snapshot, dataset_df=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Customer Delivery Portal", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 7, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(6)

    safe_pdf_cell(pdf, "Delivery Decision", 8, True)
    safe_pdf_cell(pdf, f"Score: {delivery_snapshot.get('delivery_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {delivery_snapshot.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Status: {delivery_snapshot.get('portal_status', 'Unknown')}")
    safe_pdf_multicell(pdf, delivery_snapshot.get("customer_status_line", ""))
    pdf.ln(3)

    safe_pdf_cell(pdf, "Customer", 8, True)
    customer = delivery_snapshot.get("customer", {})
    safe_pdf_cell(pdf, f"Name: {customer.get('name', '')}")
    safe_pdf_cell(pdf, f"Organization: {customer.get('organization', '')}")
    safe_pdf_cell(pdf, f"Email: {customer.get('email', '')}")
    safe_pdf_cell(pdf, f"Delivery level: {delivery_snapshot.get('delivery_level')}")
    safe_pdf_cell(pdf, f"Plan: {delivery_snapshot.get('selected_plan')}")
    pdf.ln(3)

    safe_pdf_cell(pdf, "Blockers", 8, True)
    blockers = delivery_snapshot.get("blockers", [])
    if blockers:
        for item in blockers:
            safe_pdf_multicell(pdf, f"- {item}")
    else:
        safe_pdf_multicell(pdf, "No hard delivery blockers detected.")
    pdf.ln(3)

    safe_pdf_cell(pdf, "Warnings", 8, True)
    warnings = delivery_snapshot.get("warnings", [])
    if warnings:
        for item in warnings:
            safe_pdf_multicell(pdf, f"- {item}")
    else:
        safe_pdf_multicell(pdf, "No warnings detected.")
    pdf.ln(3)

    safe_pdf_cell(pdf, "Required Deliverables", 8, True)
    for row in delivery_snapshot.get("required_status", []):
        safe_pdf_cell(pdf, f"- {row.get('deliverable')}: {row.get('status')}")
    pdf.ln(3)

    safe_pdf_cell(pdf, "Safe Claims", 8, True)
    for item in delivery_snapshot.get("safe_claims", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(3)

    safe_pdf_cell(pdf, "Claims To Avoid", 8, True)
    for item in delivery_snapshot.get("claims_to_avoid", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(3)

    safe_pdf_cell(pdf, "Next Handoff Steps", 8, True)
    for item in delivery_snapshot.get("handoff_steps", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(3)
    safe_pdf_multicell(pdf, delivery_snapshot.get("disclaimer", DELIVERY_PORTAL_DISCLAIMER))

    deliverables_df = pd.DataFrame(delivery_snapshot.get("deliverables", []))
    checklist_df = pd.DataFrame(delivery_snapshot.get("delivery_checklist", []))
    claims_df = pd.DataFrame({
        "safe_claims": pd.Series(delivery_snapshot.get("safe_claims", [])),
        "claims_to_avoid": pd.Series(delivery_snapshot.get("claims_to_avoid", [])),
    })

    readme = f"""EdgeTwin Studio V42 Customer Delivery Portal

Project: {project_name}
Customer: {delivery_snapshot.get('customer', {}).get('organization', '')}
Delivery level: {delivery_snapshot.get('delivery_level')}
Decision: {delivery_snapshot.get('decision')}
Score: {delivery_snapshot.get('delivery_score')}%

{delivery_snapshot.get('customer_status_line')}

Important: {delivery_snapshot.get('disclaimer', DELIVERY_PORTAL_DISCLAIMER)}
"""

    metadata = {
        "project_name": project_name,
        "created_at": _now(),
        "delivery_snapshot": delivery_snapshot,
    }

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("customer_delivery_report.pdf", safe_pdf_output(pdf))
        zf.writestr("delivery_manifest.json", json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False))
        zf.writestr("deliverables_status.csv", deliverables_df.to_csv(index=False))
        zf.writestr("delivery_checklist.csv", checklist_df.to_csv(index=False))
        zf.writestr("claims_matrix.csv", claims_df.to_csv(index=False))
        zf.writestr("README.txt", readme)
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr("dataset_snapshot.csv", dataset_df.head(5000).to_csv(index=False))
    return zip_buf.getvalue()


# ============================================================
# V43 CUSTOMER SUCCESS FEEDBACK LOOP / POST-DELIVERY LEARNING
# ============================================================

CUSTOMER_SUCCESS_STAGES = [
    "First demo follow-up",
    "Private beta check-in",
    "Paid pilot qualification",
    "Real-data review follow-up",
    "Post-delivery handoff",
    "Renewal / expansion review",
]

CUSTOMER_FEEDBACK_CHANNELS = [
    "Call / meeting",
    "Email feedback",
    "In-app feedback",
    "Technical workshop",
    "Procurement / management review",
    "Field test debrief",
]


def get_customer_success_stages():
    return CUSTOMER_SUCCESS_STAGES


def get_customer_feedback_channels():
    return CUSTOMER_FEEDBACK_CHANNELS


def _numeric_score(value, default=0):
    try:
        return float(value)
    except Exception:
        return float(default)


def build_customer_success_feedback_loop(
    project_name,
    customer_name="Beta Customer",
    customer_organization="Customer Organization",
    customer_email="customer@example.com",
    selected_plan="Founder Test Mode",
    success_stage="First demo follow-up",
    feedback_channel="Call / meeting",
    requested_next_step="Wants paid pilot",
    customer_satisfaction=75,
    stakeholder_confidence=70,
    technical_fit=75,
    evidence_clarity=70,
    price_acceptance=60,
    friction_points=None,
    follow_up_days=7,
    notes="",
    customer_delivery_snapshot=None,
    product_readiness_snapshot=None,
    field_evidence_snapshot=None,
    commercial_license_certificate=None,
):
    friction_points = friction_points or []
    scores = {
        "customer_satisfaction": _numeric_score(customer_satisfaction),
        "stakeholder_confidence": _numeric_score(stakeholder_confidence),
        "technical_fit": _numeric_score(technical_fit),
        "evidence_clarity": _numeric_score(evidence_clarity),
        "price_acceptance": _numeric_score(price_acceptance),
    }
    evidence_bonus = 0
    evidence_sources = []
    if customer_delivery_snapshot:
        evidence_bonus += 5
        evidence_sources.append("Customer Delivery Portal available")
    if product_readiness_snapshot:
        evidence_bonus += 5
        evidence_sources.append("V40 Product Readiness available")
    if field_evidence_snapshot:
        evidence_bonus += 8
        evidence_sources.append("Field Evidence 2.0 available")
    if commercial_license_certificate:
        evidence_bonus += 4
        evidence_sources.append("Commercial License Certificate available")

    base_score = (
        scores["customer_satisfaction"] * 0.24
        + scores["stakeholder_confidence"] * 0.20
        + scores["technical_fit"] * 0.22
        + scores["evidence_clarity"] * 0.20
        + scores["price_acceptance"] * 0.14
    )
    friction_penalty = min(18, len(friction_points) * 4)
    success_score = int(np.clip(base_score + evidence_bonus - friction_penalty, 0, 100))

    blockers = []
    warnings = []
    if scores["customer_satisfaction"] < 45:
        blockers.append("Customer satisfaction is low; do not push paid delivery before resolving objections.")
    if scores["technical_fit"] < 50:
        blockers.append("Technical fit is weak; clarify use-case, sensors and required outputs before next commercial step.")
    if scores["evidence_clarity"] < 55:
        warnings.append("Customer needs clearer evidence. Generate/refresh Reports 2.0, Field Evidence or Delivery Portal before asking for commitment.")
    if scores["price_acceptance"] < 45:
        warnings.append("Price acceptance is low. Use starter/professional pilot positioning and emphasize concrete deliverables.")
    if requested_next_step in {"Wants paid pilot", "Wants real-data review"} and not customer_delivery_snapshot:
        warnings.append("Customer wants a next step, but no Customer Delivery Portal snapshot is attached yet.")

    if success_score >= 78 and not blockers:
        decision = "GO"
        churn_risk = "Low"
        upgrade_potential = "High" if scores["price_acceptance"] >= 65 else "Medium"
    elif success_score >= 58 and not blockers:
        decision = "CONDITIONAL GO"
        churn_risk = "Medium"
        upgrade_potential = "Medium"
    else:
        decision = "NO-GO"
        churn_risk = "High"
        upgrade_potential = "Low"

    if requested_next_step == "Wants paid pilot" and decision != "NO-GO":
        commercial_move = "Offer a controlled paid pilot bundle with explicit scope, safe claims and field-validation next steps."
    elif requested_next_step == "Wants real-data review" and decision != "NO-GO":
        commercial_move = "Offer Real-Data Pilot / Synthetic-to-Real review as the next paid step."
    elif requested_next_step == "Needs more evidence":
        commercial_move = "Do not sell hard yet. Generate more evidence: field validation, reliability report and delivery portal."
    elif requested_next_step == "Enterprise/procurement review":
        commercial_move = "Prepare enterprise review package: assurance, license certificate, delivery portal and safe-use terms."
    else:
        commercial_move = "Schedule a technical follow-up and use objections to improve the pilot package."

    follow_up_plan = [
        {"day": 0, "owner": "Operator", "action": "Send customer summary and safe status line."},
        {"day": min(2, int(follow_up_days)), "owner": "Operator", "action": "Answer top objections and share missing evidence plan."},
        {"day": int(follow_up_days), "owner": "Operator", "action": "Follow up on next-step decision and collect structured feedback."},
    ]
    if scores["evidence_clarity"] < 70:
        follow_up_plan.insert(1, {"day": 1, "owner": "EdgeTwin", "action": "Refresh report, delivery portal, field evidence or real-data bridge snapshot."})
    if scores["price_acceptance"] < 55:
        follow_up_plan.append({"day": int(follow_up_days) + 2, "owner": "Operator", "action": "Offer lower-risk starter/professional pilot scope instead of enterprise scope."})

    feedback_scores = [
        {"metric": "Customer satisfaction", "score": int(scores["customer_satisfaction"]), "meaning": "Did the customer clearly see value?"},
        {"metric": "Stakeholder confidence", "score": int(scores["stakeholder_confidence"]), "meaning": "Can the buyer defend this internally?"},
        {"metric": "Technical fit", "score": int(scores["technical_fit"]), "meaning": "Does the EdgeTwin workflow match the use-case?"},
        {"metric": "Evidence clarity", "score": int(scores["evidence_clarity"]), "meaning": "Are reports/scores/proof understandable?"},
        {"metric": "Price acceptance", "score": int(scores["price_acceptance"]), "meaning": "Is the customer open to the proposed package level?"},
    ]

    status_line = (
        "Customer success loop is strong enough for the next controlled commercial step."
        if decision == "GO" else
        "Customer success loop is usable, but resolve warnings before pushing a paid step."
        if decision == "CONDITIONAL GO" else
        "Customer success loop is not ready for paid delivery; resolve blockers first."
    )

    safe_claims = [
        "EdgeTwin can prepare a structured pilot package and clarify next validation steps.",
        "Feedback is converted into a follow-up plan, blockers and commercial next-step guidance.",
        "Paid delivery should remain scoped to pilot preparation unless field evidence supports broader claims.",
    ]
    claims_to_avoid = [
        "Do not claim the customer is production-ready based only on a positive demo.",
        "Do not claim guaranteed ROI without field validation and customer-side cost data.",
        "Do not treat customer interest as a signed commercial commitment.",
    ]

    return {
        "version": "V43 Customer Success Feedback Loop",
        "project_name": str(project_name),
        "created_at": _now() if '_now' in globals() else str(datetime.datetime.now()),
        "customer": {
            "name": str(customer_name),
            "organization": str(customer_organization),
            "email": str(customer_email),
        },
        "selected_plan": selected_plan,
        "success_stage": success_stage,
        "feedback_channel": feedback_channel,
        "requested_next_step": requested_next_step,
        "success_score": success_score,
        "decision": decision,
        "churn_risk": churn_risk,
        "upgrade_potential": upgrade_potential,
        "customer_status_line": status_line,
        "feedback_scores": feedback_scores,
        "friction_points": friction_points,
        "evidence_sources": evidence_sources or ["No attached delivery/product/field evidence snapshots"],
        "evidence_bonus": evidence_bonus,
        "blockers": blockers,
        "warnings": warnings,
        "follow_up_plan": follow_up_plan,
        "next_actions": [x["action"] for x in follow_up_plan],
        "recommended_commercial_move": commercial_move,
        "safe_claims": safe_claims,
        "claims_to_avoid": claims_to_avoid,
        "notes": str(notes or ""),
        "disclaimer": "Customer feedback supports commercial prioritization; it is not proof of production readiness or guaranteed deployment success.",
    }


def create_customer_success_feedback_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    zip_buf = io.BytesIO()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Customer Success Feedback Loop", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(6)
    safe_pdf_cell(pdf, "Success Snapshot", 8, True)
    safe_pdf_cell(pdf, f"Success score: {snapshot.get('success_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {snapshot.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Churn risk: {snapshot.get('churn_risk', 'Unknown')}")
    safe_pdf_cell(pdf, f"Upgrade potential: {snapshot.get('upgrade_potential', 'Unknown')}")
    safe_pdf_multicell(pdf, snapshot.get("customer_status_line", ""))
    pdf.ln(3)
    safe_pdf_cell(pdf, "Recommended Commercial Move", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("recommended_commercial_move", ""))
    pdf.ln(3)
    safe_pdf_cell(pdf, "Friction Points", 8, True)
    for item in snapshot.get("friction_points", []) or ["No friction points recorded."]:
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Follow-up Plan", 8, True)
    for item in snapshot.get("follow_up_plan", []):
        safe_pdf_multicell(pdf, f"Day {item.get('day')}: {item.get('action')} ({item.get('owner')})")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Claims to Avoid", 8, True)
    for item in snapshot.get("claims_to_avoid", []):
        safe_pdf_multicell(pdf, f"- {item}")

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("customer_success_feedback_report.pdf", safe_pdf_output(pdf))
        zf.writestr("customer_success_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr("feedback_scores.csv", pd.DataFrame(snapshot.get("feedback_scores", [])).to_csv(index=False))
        zf.writestr("follow_up_plan.csv", pd.DataFrame(snapshot.get("follow_up_plan", [])).to_csv(index=False))
        zf.writestr("friction_points.csv", pd.DataFrame({"friction_point": snapshot.get("friction_points", [])}).to_csv(index=False))
        zf.writestr("README.txt", "EdgeTwin Studio V43 Customer Success Feedback Loop. Use this after demos, delivery, beta calls or paid-pilot handoffs.\n")
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr("dataset_snapshot.csv", dataset_df.head(2000).to_csv(index=False))
    return zip_buf.getvalue()


# ============================================================
# V44 PRICING VALIDATION / OFFER BUILDER
# ============================================================

OFFER_CUSTOMER_SEGMENTS = [
    "Predictive Maintenance Team",
    "Security / Tamper Integrator",
    "Remote Asset / Forestry Operator",
    "Edge AI Consultant / Integrator",
    "Machine Builder / OEM",
    "Research / Education",
    "Enterprise Innovation Team",
]

OFFER_DELIVERY_GOALS = [
    "Free demo / evaluation",
    "Starter synthetic pilot",
    "Professional pilot bundle",
    "Real-data pilot bundle",
    "Enterprise custom review",
    "On-premise / API discussion",
]

OFFER_DATA_STATUS = [
    "No data yet",
    "Synthetic-only dataset ready",
    "Some real WAV/CSV available",
    "Field validation data available",
    "Production-sensitive customer data",
]

PRICE_SENSITIVITY_LEVELS = ["Low", "Medium", "High"]


def get_offer_customer_segments():
    return OFFER_CUSTOMER_SEGMENTS.copy()


def get_offer_delivery_goals():
    return OFFER_DELIVERY_GOALS.copy()


def get_offer_data_statuses():
    return OFFER_DATA_STATUS.copy()


def get_price_sensitivity_levels():
    return PRICE_SENSITIVITY_LEVELS.copy()


def _score_from_snapshot(snapshot, keys=None, default=0):
    if not isinstance(snapshot, dict) or not snapshot:
        return int(default)
    keys = keys or [
        "score", "trust_score", "trust_score_v2", "readiness_score", "product_readiness_score",
        "delivery_score", "success_score", "field_evidence_score", "security_score", "governance_score",
        "operational_score", "workspace_score", "admin_score", "license_score", "assurance_score",
    ]
    for key in keys:
        val = snapshot.get(key)
        if isinstance(val, (int, float)):
            return int(max(0, min(100, val)))
    return int(default)


def _decision_from_score(score, go=82, conditional=62):
    score = int(score or 0)
    if score >= go:
        return "GO"
    if score >= conditional:
        return "CONDITIONAL GO"
    return "NO-GO"


def build_pricing_validation_offer(
    project_name="EdgeTwin_Project",
    customer_segment="Predictive Maintenance Team",
    use_case="Predictive Maintenance",
    data_status="Synthetic-only dataset ready",
    requested_delivery_goal="Professional pilot bundle",
    selected_plan="Professional Pilot",
    price_sensitivity="Medium",
    customer_budget_range="Unknown / not discussed",
    dataset_df=None,
    trust_gate=None,
    reliability_v2=None,
    product_readiness_snapshot=None,
    field_evidence_snapshot=None,
    customer_delivery_snapshot=None,
    customer_success_snapshot=None,
    commercial_license_certificate=None,
    governance_snapshot=None,
    security_snapshot=None,
    notes="",
):
    """Create a conservative pricing/offer recommendation.

    This does not process payments. It helps the founder decide what to offer,
    what to include, what to lock, and which claims are safe.
    """
    rows = int(len(dataset_df)) if isinstance(dataset_df, pd.DataFrame) else 0
    cols = int(len(dataset_df.columns)) if isinstance(dataset_df, pd.DataFrame) else 0
    has_dataset = rows > 0
    has_real_data = data_status in ["Some real WAV/CSV available", "Field validation data available", "Production-sensitive customer data"] or bool(reliability_v2) or bool(field_evidence_snapshot)
    has_field_evidence = data_status == "Field validation data available" or bool(field_evidence_snapshot)
    has_governance = bool(governance_snapshot)
    has_license = bool(commercial_license_certificate)
    has_delivery = bool(customer_delivery_snapshot)

    trust_score = max(
        _score_from_snapshot(trust_gate, ["trust_score", "overall_score"], 0),
        _score_from_snapshot(reliability_v2, ["trust_score_v2", "reliability_score", "trust_score"], 0),
    )
    product_score = _score_from_snapshot(product_readiness_snapshot, ["product_readiness_score", "readiness_score", "score"], 0)
    field_score = _score_from_snapshot(field_evidence_snapshot, ["field_evidence_score", "field_validation_score", "score"], 0)
    delivery_score = _score_from_snapshot(customer_delivery_snapshot, ["delivery_score", "score"], 0)
    success_score = _score_from_snapshot(customer_success_snapshot, ["success_score", "score"], 0)
    governance_score = _score_from_snapshot(governance_snapshot, ["governance_score", "assurance_score", "score"], 0)
    security_score = _score_from_snapshot(security_snapshot, ["security_score", "score"], 0)

    evidence_components = []
    if has_dataset:
        evidence_components.append({"component": "Dataset available", "score": 70, "status": "present"})
    else:
        evidence_components.append({"component": "Dataset available", "score": 0, "status": "missing"})
    evidence_components.extend([
        {"component": "Trust / reliability evidence", "score": trust_score, "status": "present" if trust_score else "missing"},
        {"component": "Product readiness", "score": product_score, "status": "present" if product_score else "missing"},
        {"component": "Field evidence", "score": field_score, "status": "present" if field_score else "missing/optional"},
        {"component": "Customer delivery proof", "score": delivery_score, "status": "present" if delivery_score else "missing/optional"},
        {"component": "Governance/assurance", "score": governance_score, "status": "present" if governance_score else "missing/optional"},
        {"component": "Security/access", "score": security_score, "status": "present" if security_score else "missing/optional"},
    ])

    base_score = 35
    if has_dataset:
        base_score += 12
    if trust_score:
        base_score += min(18, trust_score * 0.18)
    if product_score:
        base_score += min(14, product_score * 0.14)
    if has_real_data:
        base_score += 8
    if has_field_evidence:
        base_score += min(12, max(4, field_score * 0.12 if field_score else 6))
    if has_governance:
        base_score += 5
    if has_license:
        base_score += 5
    if has_delivery:
        base_score += 4
    if success_score:
        base_score += min(4, success_score * 0.04)

    blockers = []
    warnings_list = []

    if not has_dataset:
        blockers.append("No active dataset is available. Do not offer a paid pilot bundle yet.")
    if "Real-data" in requested_delivery_goal and not has_real_data:
        blockers.append("Requested offer is real-data based, but no real-data evidence is available yet.")
    if "Enterprise" in requested_delivery_goal and not has_governance:
        warnings_list.append("Enterprise/custom review should include customer assurance, retention and data-ownership language.")
    if "On-premise" in requested_delivery_goal and selected_plan != "Enterprise":
        warnings_list.append("On-premise/API discussions should normally be positioned as Enterprise/custom only.")
    if trust_score and trust_score < 60:
        warnings_list.append("Trust/reliability evidence is weak; keep the offer at demo/starter level.")
    if has_real_data and not has_governance:
        warnings_list.append("Real data is involved. Add Customer Assurance/Data Governance before paid delivery.")
    if price_sensitivity == "High":
        warnings_list.append("Customer is price-sensitive. Use a smaller starter scope and upsell after evidence improves.")

    score = int(np.clip(base_score - len(blockers) * 22 - len(warnings_list) * 4, 0, 100))
    decision = _decision_from_score(score, go=82, conditional=60)

    # Package recommendation
    if blockers:
        recommended_tier = "Free Demo / Internal Follow-up"
        recommended_plan = "Free Demo"
        price_range = "Free - EUR 49"
        one_liner = "Use a limited demo or evidence-building session before selling a paid pilot."
    elif has_field_evidence and (trust_score >= 70 or product_score >= 70):
        recommended_tier = "Real-Data Pilot Bundle"
        recommended_plan = "Real-Data Pilot"
        price_range = "EUR 799 - 1,500"
        one_liner = "Offer a real-data pilot package with field-evidence review and clear production limitations."
    elif has_real_data:
        recommended_tier = "Real-Data Mini Audit"
        recommended_plan = "Real-Data Pilot"
        price_range = "EUR 299 - 799"
        one_liner = "Offer a smaller real-data review before a full paid pilot."
    elif trust_score >= 65 or product_score >= 65 or requested_delivery_goal == "Professional pilot bundle":
        recommended_tier = "Professional Pilot Bundle"
        recommended_plan = "Professional Pilot"
        price_range = "EUR 199 - 499"
        one_liner = "Offer a professional synthetic pilot package with reliability, report and hardware direction."
    elif has_dataset:
        recommended_tier = "Starter Pilot Bundle"
        recommended_plan = "Starter Pilot"
        price_range = "EUR 49 - 149"
        one_liner = "Offer a starter synthetic dataset/audit bundle with clear limitations."
    else:
        recommended_tier = "Free Demo / Lead Magnet"
        recommended_plan = "Free Demo"
        price_range = "Free"
        one_liner = "Show the workflow, collect the use-case and build evidence first."

    if customer_segment in ["Enterprise Innovation Team", "Machine Builder / OEM"] and decision != "NO-GO" and has_governance:
        if recommended_tier in ["Professional Pilot Bundle", "Real-Data Pilot Bundle"]:
            warnings_list.append("This customer segment may support enterprise pricing if multiple assets/sites are included.")

    included_deliverables = [
        {"deliverable": "Dataset snapshot / training CSV", "include": has_dataset, "minimum_tier": "Starter Pilot"},
        {"deliverable": "Dataset Doctor / quality audit", "include": has_dataset, "minimum_tier": "Starter Pilot"},
        {"deliverable": "Reliability / Trust summary", "include": bool(trust_score), "minimum_tier": "Professional Pilot"},
        {"deliverable": "Professional report", "include": recommended_tier not in ["Free Demo / Lead Magnet", "Free Demo / Internal Follow-up"], "minimum_tier": "Professional Pilot"},
        {"deliverable": "Hardware/deployment direction", "include": recommended_tier in ["Professional Pilot Bundle", "Real-Data Mini Audit", "Real-Data Pilot Bundle"], "minimum_tier": "Professional Pilot"},
        {"deliverable": "Synthetic-to-real / real-data review", "include": has_real_data, "minimum_tier": "Real-Data Pilot"},
        {"deliverable": "Field evidence acceptance check", "include": has_field_evidence, "minimum_tier": "Real-Data Pilot"},
        {"deliverable": "Commercial license certificate", "include": has_license or recommended_tier not in ["Free Demo / Lead Magnet", "Free Demo / Internal Follow-up"], "minimum_tier": "Professional Pilot"},
    ]

    excluded_deliverables = [
        "Production guarantee",
        "Safety certification",
        "Guaranteed model accuracy",
        "Autonomous production deployment without field validation",
    ]
    if not has_real_data:
        excluded_deliverables.append("Real-data similarity proof")
    if not has_field_evidence:
        excluded_deliverables.append("Field-evidence acceptance proof")

    offer_matrix = []
    for item in PACKAGE_PRICE_LADDER:
        pkg = item.get("package")
        fit = "Low"
        reason = item.get("when_to_offer", "")
        if recommended_tier.startswith("Starter") and "Starter" in pkg:
            fit = "High"
        elif recommended_tier.startswith("Professional") and "Professional" in pkg:
            fit = "High"
        elif recommended_tier.startswith("Real-Data") and "Real-Data" in pkg:
            fit = "High"
        elif "Enterprise" in requested_delivery_goal and "Enterprise" in pkg:
            fit = "Medium-High" if has_governance else "Medium"
        elif "Demo" in pkg and not has_dataset:
            fit = "High"
        elif "Demo" in pkg:
            fit = "Medium"
        offer_matrix.append({
            "package": pkg,
            "fit": fit,
            "price_range": item.get("price_range"),
            "when_to_offer": reason,
            "promise": item.get("promise"),
        })

    safe_claims = [
        "EdgeTwin can generate a structured Edge AI pilot package for this use-case.",
        "The offer includes dataset, audit, reliability/risk language and next-step recommendations within the selected scope.",
        "The output is intended for pilot preparation and decision support.",
        "Production deployment requires field validation and customer-side engineering review.",
    ]
    if has_real_data:
        safe_claims.append("Real uploaded data can be used to strengthen similarity, fingerprinting and evidence review.")
    if has_field_evidence:
        safe_claims.append("Field evidence is available for a stronger pilot acceptance discussion, within the stated limitations.")

    claims_to_avoid = [
        "Do not claim production-ready or certified unless field validation and customer acceptance are complete.",
        "Do not guarantee model accuracy, safety or hardware performance from synthetic data alone.",
        "Do not imply uploaded audio/sensor data is reused for other customers without explicit permission.",
        "Do not sell Enterprise/on-premise promises as a fixed low-price bundle.",
    ]

    value_drivers = [
        "Reduces early trial-and-error by producing a clear pilot package.",
        "Clarifies labels/classes, normal-vs-abnormal baseline and weak evidence areas.",
        "Creates professional reporting that can be shared internally by the customer.",
        "Links data quality, reliability and hardware/deployment direction in one workflow.",
    ]

    if recommended_tier in ["Real-Data Mini Audit", "Real-Data Pilot Bundle"]:
        value_drivers.append("Uses real WAV/CSV evidence to make the pilot more credible than synthetic-only preparation.")

    upsell_path = []
    if recommended_tier in ["Free Demo / Lead Magnet", "Free Demo / Internal Follow-up"]:
        upsell_path = ["Collect use-case details", "Generate starter dataset", "Offer Starter or Professional Pilot Bundle"]
    elif recommended_tier.startswith("Starter"):
        upsell_path = ["Generate Professional Report", "Add Reliability 2.0", "Offer Professional Pilot Bundle"]
    elif recommended_tier.startswith("Professional"):
        upsell_path = ["Ask for real WAV/CSV samples", "Run Synthetic-to-Real Bridge", "Offer Real-Data Pilot Bundle"]
    elif recommended_tier.startswith("Real-Data"):
        upsell_path = ["Run controlled field validation", "Add deployment planner/BOM", "Offer Enterprise custom review"]
    else:
        upsell_path = ["Define enterprise scope", "Set custom review terms", "Prepare API/on-premise discussion"]

    customer_facing_offer = {
        "headline": f"{recommended_tier} for {use_case}",
        "one_liner": one_liner,
        "price_range": price_range,
        "scope": f"Segment: {customer_segment}. Data status: {data_status}. Delivery goal: {requested_delivery_goal}.",
        "what_customer_gets": [d["deliverable"] for d in included_deliverables if d.get("include")],
        "what_is_not_included": excluded_deliverables,
        "safe_status_line": f"Recommended offer: {recommended_tier} ({price_range}). Decision: {decision}. {one_liner}",
    }

    return {
        "version": "V44 Pricing Validation & Offer Builder",
        "created_at": _now(),
        "project_name": project_name,
        "customer_segment": customer_segment,
        "use_case": use_case,
        "data_status": data_status,
        "requested_delivery_goal": requested_delivery_goal,
        "selected_plan": selected_plan,
        "price_sensitivity": price_sensitivity,
        "customer_budget_range": customer_budget_range,
        "dataset_rows": rows,
        "dataset_cols": cols,
        "has_dataset": has_dataset,
        "has_real_data": has_real_data,
        "has_field_evidence": has_field_evidence,
        "scores": {
            "offer_score": score,
            "trust_score": trust_score,
            "product_readiness_score": product_score,
            "field_evidence_score": field_score,
            "delivery_score": delivery_score,
            "success_score": success_score,
            "governance_score": governance_score,
            "security_score": security_score,
        },
        "decision": decision,
        "recommended_tier": recommended_tier,
        "recommended_plan": recommended_plan,
        "price_range": price_range,
        "one_liner": one_liner,
        "blockers": blockers,
        "warnings": warnings_list,
        "included_deliverables": included_deliverables,
        "excluded_deliverables": excluded_deliverables,
        "offer_matrix": offer_matrix,
        "evidence_components": evidence_components,
        "value_drivers": value_drivers,
        "upsell_path": upsell_path,
        "safe_claims": safe_claims,
        "claims_to_avoid": claims_to_avoid,
        "customer_facing_offer": customer_facing_offer,
        "internal_notes": str(notes or ""),
        "disclaimer": "Pricing guidance is an internal commercial recommendation. It is not a payment processor, legal agreement, production guarantee or safety certification.",
    }


def create_pricing_validation_offer_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    zip_buf = io.BytesIO()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Pricing Validation & Offer Builder", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(6)
    safe_pdf_cell(pdf, "Offer Recommendation", 8, True)
    safe_pdf_cell(pdf, f"Offer score: {snapshot.get('scores', {}).get('offer_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {snapshot.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Recommended tier: {snapshot.get('recommended_tier', 'Unknown')}")
    safe_pdf_cell(pdf, f"Recommended plan: {snapshot.get('recommended_plan', 'Unknown')}")
    safe_pdf_cell(pdf, f"Price range: {snapshot.get('price_range', 'Unknown')}")
    safe_pdf_multicell(pdf, snapshot.get("one_liner", ""))
    pdf.ln(3)
    safe_pdf_cell(pdf, "Customer-Facing Offer", 8, True)
    cfo = snapshot.get("customer_facing_offer", {})
    safe_pdf_multicell(pdf, cfo.get("safe_status_line", ""))
    pdf.ln(3)
    safe_pdf_cell(pdf, "Included Deliverables", 8, True)
    for item in snapshot.get("included_deliverables", []):
        if item.get("include"):
            safe_pdf_multicell(pdf, f"- {item.get('deliverable')} ({item.get('minimum_tier')})")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Blockers / Warnings", 8, True)
    for item in snapshot.get("blockers", []) or ["No hard pricing blockers detected."]:
        safe_pdf_multicell(pdf, f"- {item}")
    for item in snapshot.get("warnings", []):
        safe_pdf_multicell(pdf, f"- Warning: {item}")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Safe Claims", 8, True)
    for item in snapshot.get("safe_claims", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Claims to Avoid", 8, True)
    for item in snapshot.get("claims_to_avoid", []):
        safe_pdf_multicell(pdf, f"- {item}")

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("pricing_offer_report.pdf", safe_pdf_output(pdf))
        zf.writestr("pricing_offer_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr("offer_matrix.csv", pd.DataFrame(snapshot.get("offer_matrix", [])).to_csv(index=False))
        zf.writestr("included_deliverables.csv", pd.DataFrame(snapshot.get("included_deliverables", [])).to_csv(index=False))
        zf.writestr("evidence_components.csv", pd.DataFrame(snapshot.get("evidence_components", [])).to_csv(index=False))
        zf.writestr("value_drivers.csv", pd.DataFrame({"value_driver": snapshot.get("value_drivers", [])}).to_csv(index=False))
        zf.writestr("upsell_path.csv", pd.DataFrame({"next_step": snapshot.get("upsell_path", [])}).to_csv(index=False))
        zf.writestr("safe_claims.csv", pd.DataFrame({"safe_claim": snapshot.get("safe_claims", [])}).to_csv(index=False))
        zf.writestr("claims_to_avoid.csv", pd.DataFrame({"claim_to_avoid": snapshot.get("claims_to_avoid", [])}).to_csv(index=False))
        zf.writestr("README.txt", "EdgeTwin Studio V44 Pricing Validation & Offer Builder. This is internal offer guidance, not a payment processor or legal contract.\n")
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr("dataset_snapshot.csv", dataset_df.head(2000).to_csv(index=False))
    return zip_buf.getvalue()


# ============================================================
# V45 PAID PILOT LAUNCH READINESS / FIRST REVENUE GATE
# ============================================================

def _v45_nested_score(snapshot, possible_keys=None, default=0):
    if not isinstance(snapshot, dict):
        return int(default)
    possible_keys = possible_keys or []
    for key in possible_keys:
        if key in snapshot and isinstance(snapshot.get(key), (int, float)):
            return int(snapshot.get(key))
    scores = snapshot.get("scores", {}) if isinstance(snapshot.get("scores", {}), dict) else {}
    for key in possible_keys:
        if key in scores and isinstance(scores.get(key), (int, float)):
            return int(scores.get(key))
    for field in ["score", "overall_score", "trust_score", "product_readiness_score", "delivery_score", "offer_score", "security_score", "governance_score", "admin_score", "safety_score"]:
        if field in snapshot and isinstance(snapshot.get(field), (int, float)):
            return int(snapshot.get(field))
        if field in scores and isinstance(scores.get(field), (int, float)):
            return int(scores.get(field))
    return int(default)


def _v45_component(name, present, score=None, weight=10, required=False, note=""):
    score_val = int(np.clip(score if score is not None else (100 if present else 0), 0, 100))
    contribution = (score_val / 100.0) * float(weight) if present else 0.0
    status = "PASS" if present and score_val >= 70 else ("WEAK" if present else ("MISSING" if required else "OPTIONAL"))
    return {
        "component": name,
        "present": bool(present),
        "score": score_val,
        "weight": float(weight),
        "contribution": round(contribution, 2),
        "required": bool(required),
        "status": status,
        "note": note,
    }


def build_paid_pilot_launch_readiness(
    project_name,
    customer_name="Beta Customer",
    customer_organization="Customer Organization",
    customer_email="customer@example.com",
    use_case="Predictive Maintenance",
    offer_tier="Professional Pilot Bundle",
    selected_plan="Professional Pilot",
    payment_status="Quote requested",
    contract_status="Draft scope",
    delivery_window="1 week",
    acceptance_mode="Paid pilot acceptance",
    dataset_df=None,
    pricing_offer_snapshot=None,
    commercial_license_certificate=None,
    customer_delivery_snapshot=None,
    product_readiness_snapshot=None,
    field_evidence_snapshot=None,
    security_snapshot=None,
    governance_snapshot=None,
    observability_snapshot=None,
    admin_usage_snapshot=None,
    customer_success_snapshot=None,
    notes="",
):
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    rows = int(len(dataset_df))
    cols = int(len(dataset_df.columns))
    has_dataset = rows > 0
    has_pricing = isinstance(pricing_offer_snapshot, dict) and bool(pricing_offer_snapshot)
    has_license = isinstance(commercial_license_certificate, dict) and bool(commercial_license_certificate)
    has_delivery = isinstance(customer_delivery_snapshot, dict) and bool(customer_delivery_snapshot)
    has_product = isinstance(product_readiness_snapshot, dict) and bool(product_readiness_snapshot)
    has_field = isinstance(field_evidence_snapshot, dict) and bool(field_evidence_snapshot)
    has_security = isinstance(security_snapshot, dict) and bool(security_snapshot)
    has_governance = isinstance(governance_snapshot, dict) and bool(governance_snapshot)
    has_observability = isinstance(observability_snapshot, dict) and bool(observability_snapshot)
    has_admin = isinstance(admin_usage_snapshot, dict) and bool(admin_usage_snapshot)
    has_success = isinstance(customer_success_snapshot, dict) and bool(customer_success_snapshot)

    offer_score = _v45_nested_score(pricing_offer_snapshot, ["offer_score"], 0)
    delivery_score = _v45_nested_score(customer_delivery_snapshot, ["delivery_score", "customer_delivery_score"], 0)
    product_score = _v45_nested_score(product_readiness_snapshot, ["product_readiness_score"], 0)
    field_score = _v45_nested_score(field_evidence_snapshot, ["field_evidence_score", "field_acceptance_score"], 0)
    security_score = _v45_nested_score(security_snapshot, ["security_score", "security_hardening_score"], 0)
    governance_score = _v45_nested_score(governance_snapshot, ["governance_score", "customer_assurance_score"], 0)
    observability_score = _v45_nested_score(observability_snapshot, ["safety_score", "observability_score"], 0)
    admin_score = _v45_nested_score(admin_usage_snapshot, ["admin_readiness_score", "admin_score"], 0)
    success_score = _v45_nested_score(customer_success_snapshot, ["customer_success_score", "satisfaction_score"], 0)
    license_score = 100 if has_license else 0

    evidence_checks = [
        _v45_component("Dataset / project artifact exists", has_dataset, 100 if has_dataset else 0, weight=8, required=True, note=f"{rows} rows, {cols} columns"),
        _v45_component("Pricing offer is built", has_pricing, offer_score, weight=14, required=True, note="V44 offer scope and price guidance"),
        _v45_component("Commercial license/certificate exists", has_license, license_score, weight=12, required=True, note="V38 certificate/scope guard"),
        _v45_component("Customer delivery portal exists", has_delivery, delivery_score, weight=11, required=True, note="V42 handoff readiness"),
        _v45_component("V40 product readiness gate", has_product, product_score, weight=10, required=True, note="Full product readiness evidence"),
        _v45_component("V41 security/access hardening", has_security, security_score, weight=9, required=True, note="Access, upload and claim controls"),
        _v45_component("Customer assurance/governance", has_governance, governance_score, weight=8, required=True, note="Upload notice, ownership and retention"),
        _v45_component("Observability/error safety", has_observability, observability_score, weight=7, required=False, note="Incident visibility before customer use"),
        _v45_component("Admin/usage tracking", has_admin, admin_score, weight=6, required=False, note="Usage/export record keeping"),
        _v45_component("Field evidence available", has_field, field_score, weight=8, required=offer_tier in ["Real-Data Pilot Bundle", "Enterprise Custom Review"], note="Required for stronger real-data promises"),
        _v45_component("Customer success feedback context", has_success, success_score, weight=4, required=False, note="Useful after demo or beta call"),
    ]

    base_score = sum(item["contribution"] for item in evidence_checks)
    payment_bonus = {
        "Not discussed": -8,
        "Quote requested": 0,
        "Quote accepted": 8,
        "Invoice sent": 10,
        "Paid / invoice confirmed": 14,
    }.get(payment_status, 0)
    contract_bonus = {
        "Not started": -12,
        "Draft scope": 0,
        "Scope reviewed": 6,
        "Scope accepted": 12,
        "MSA/NDA required": -2,
    }.get(contract_status, 0)

    score = int(np.clip(base_score + payment_bonus + contract_bonus, 0, 100))

    blockers = []
    warnings_list = []
    if not has_dataset:
        blockers.append("No active dataset/project artifact is loaded. Generate a pilot package before paid delivery.")
    if not has_pricing:
        blockers.append("No V44 Pricing Offer exists. Build the offer before moving to paid pilot.")
    if not has_license:
        blockers.append("No commercial license/certificate exists. Create V38 License Cert before paid delivery.")
    if not has_delivery:
        blockers.append("No customer delivery portal snapshot exists. Prepare V42 delivery before handoff.")
    if not has_product:
        warnings_list.append("V40 Product Readiness has not been run. Paid pilot can proceed only with careful manual review.")
    if not has_security:
        blockers.append("V41 Security/Access gate missing. Do not accept customer data/payment before access/upload controls are reviewed.")
    if not has_governance:
        blockers.append("Customer assurance/governance is missing. Upload/data ownership language must be clear before paid real-data work.")
    if offer_tier in ["Real-Data Pilot Bundle", "Enterprise Custom Review"] and not has_field:
        warnings_list.append("Real-data/enterprise tier selected without V39 Field Evidence 2.0. Keep claims limited until field evidence is reviewed.")
    if payment_status == "Not discussed":
        warnings_list.append("Payment has not been discussed. Use as quote-preparation, not delivery approval.")
    if contract_status == "Not started":
        blockers.append("Scope/contract has not started. Confirm scope before invoice or delivery.")
    if contract_status == "MSA/NDA required":
        warnings_list.append("MSA/NDA required. Do not accept sensitive files until paperwork is handled.")
    if "Enterprise" in offer_tier and selected_plan not in ["Enterprise", "Founder Test Mode"]:
        warnings_list.append("Enterprise offer selected while current plan is not Enterprise/Founder Test Mode.")

    hard_blockers = len(blockers)
    if score >= 84 and hard_blockers == 0 and contract_status in ["Scope accepted", "Scope reviewed"] and payment_status in ["Quote accepted", "Invoice sent", "Paid / invoice confirmed"]:
        decision = "GO"
        readiness_level = "PAID PILOT READY"
        revenue_status = "Delivery-ready" if payment_status == "Paid / invoice confirmed" else "Invoice/quote-ready"
    elif score >= 66 and hard_blockers <= 2:
        decision = "CONDITIONAL GO"
        readiness_level = "CONDITIONAL PAID PILOT"
        revenue_status = "Quote-ready with conditions"
    else:
        decision = "NO-GO"
        readiness_level = "NOT PAID-PILOT READY"
        revenue_status = "Internal preparation only"

    required_before_invoice = []
    if not has_pricing:
        required_before_invoice.append("Build V44 Pricing Offer and confirm recommended tier/price range.")
    if contract_status in ["Not started", "Draft scope"]:
        required_before_invoice.append("Confirm paid pilot scope, exclusions and acceptance criteria with customer.")
    if not has_license:
        required_before_invoice.append("Generate V38 Commercial License Certificate.")
    if not has_governance:
        required_before_invoice.append("Generate Customer Assurance/Governance notice and retention policy.")
    if payment_status in ["Not discussed", "Quote requested"]:
        required_before_invoice.append("Send quote/offer and get acceptance before delivery.")
    if not required_before_invoice:
        required_before_invoice.append("Invoice/quote prerequisites look acceptable for a controlled paid pilot.")

    required_before_delivery = []
    if not has_delivery:
        required_before_delivery.append("Generate V42 Customer Delivery Bundle.")
    if not has_security:
        required_before_delivery.append("Run V41 Security & Access Control Hardening.")
    if not has_product:
        required_before_delivery.append("Run V40 Full Product Readiness Gate.")
    if offer_tier in ["Real-Data Pilot Bundle", "Enterprise Custom Review"] and not has_field:
        required_before_delivery.append("Run V39 Field Evidence 2.0 before making real-data evidence claims.")
    if payment_status not in ["Invoice sent", "Paid / invoice confirmed", "Quote accepted"]:
        required_before_delivery.append("Do not release paid deliverables until quote/payment status is confirmed.")
    if not required_before_delivery:
        required_before_delivery.append("Delivery prerequisites look acceptable for controlled paid handoff.")

    acceptance_criteria = [
        {"criterion": "Customer receives agreed ZIP/PDF/report bundle", "required": True, "status": "ready" if has_delivery else "pending"},
        {"criterion": "Scope clearly excludes production guarantee/safety certification", "required": True, "status": "ready" if has_license else "pending"},
        {"criterion": "Dataset/report includes reliability limitations and field-validation language", "required": True, "status": "ready" if has_product or has_pricing else "pending"},
        {"criterion": "Customer confirms next-step field validation requirements", "required": True, "status": "ready" if contract_status in ["Scope reviewed", "Scope accepted"] else "pending"},
        {"criterion": "Payment/quote status matches delivery level", "required": True, "status": "ready" if payment_status in ["Quote accepted", "Invoice sent", "Paid / invoice confirmed"] else "pending"},
    ]

    safe_claims = [
        "This is a paid Edge AI pilot-preparation package, not a production certification.",
        "The deliverables can include dataset, audit, reliability/risk language, hardware/deployment guidance and next-step recommendations within the agreed scope.",
        "Field validation remains required before production deployment or safety-critical use.",
        "Customer data ownership, retention and reuse must follow the agreed customer assurance policy.",
    ]
    if has_field:
        safe_claims.append("Field evidence can be discussed as supporting pilot evidence within the stated limitations.")

    claims_to_avoid = [
        "Do not promise guaranteed accuracy, guaranteed ROI, production readiness or safety certification.",
        "Do not imply synthetic data replaces real field validation.",
        "Do not release paid deliverables outside the licensed scope.",
        "Do not reuse customer-uploaded data for other customers without explicit permission.",
    ]

    customer_email_draft = f"""Subject: EdgeTwin Studio paid pilot package for {use_case}

Hi {customer_name},

Thanks for discussing the {use_case} pilot with us.

Based on the current scope, the recommended package is: {offer_tier}.
Current readiness status: {readiness_level} ({decision}).
Delivery window: {delivery_window}.

What this pilot package is intended to provide:
- Structured Edge AI pilot preparation
- Dataset/audit and reliability-oriented reporting within scope
- Hardware/deployment direction where included
- Clear next steps for field validation

Important limitation:
This is pilot preparation and decision support, not a production guarantee or safety certification. Field validation is required before production deployment.

Next step:
{required_before_invoice[0] if required_before_invoice else 'Confirm invoice and delivery details.'}

Kind regards,
EdgeTwin Studio
"""

    internal_close_plan = [
        "Confirm customer problem, target sensor setup and expected deliverables.",
        "Confirm offer tier, exclusions and price range from V44.",
        "Generate license/certificate and delivery bundle before paid handoff.",
        "Only release paid deliverables when quote/payment status is confirmed.",
        "After delivery, collect V43 customer success feedback and update evidence.",
    ]

    safe_status_line = f"{readiness_level}: {offer_tier} for {customer_organization}. Score {score}%. Decision: {decision}. Revenue status: {revenue_status}."

    return {
        "version": "V45 Paid Pilot Launch Readiness / First Revenue Gate",
        "created_at": _now(),
        "project_name": project_name,
        "customer": {
            "name": customer_name,
            "organization": customer_organization,
            "email": customer_email,
        },
        "use_case": use_case,
        "offer_tier": offer_tier,
        "selected_plan": selected_plan,
        "payment_status": payment_status,
        "contract_status": contract_status,
        "delivery_window": delivery_window,
        "acceptance_mode": acceptance_mode,
        "dataset_rows": rows,
        "dataset_cols": cols,
        "scores": {
            "paid_pilot_score": score,
            "offer_score": offer_score,
            "delivery_score": delivery_score,
            "license_score": license_score,
            "product_readiness_score": product_score,
            "field_evidence_score": field_score,
            "security_score": security_score,
            "governance_score": governance_score,
            "observability_score": observability_score,
            "admin_score": admin_score,
            "customer_success_score": success_score,
        },
        "decision": decision,
        "readiness_level": readiness_level,
        "revenue_status": revenue_status,
        "safe_status_line": safe_status_line,
        "evidence_checks": evidence_checks,
        "blockers": blockers,
        "warnings": warnings_list,
        "required_before_invoice": required_before_invoice,
        "required_before_delivery": required_before_delivery,
        "acceptance_criteria": acceptance_criteria,
        "safe_claims": safe_claims,
        "claims_to_avoid": claims_to_avoid,
        "customer_email_draft": customer_email_draft,
        "internal_close_plan": internal_close_plan,
        "internal_notes": str(notes or ""),
        "disclaimer": "V45 is a paid-pilot readiness gate. It is not a payment processor, legal contract, production guarantee or safety certification.",
    }


def create_paid_pilot_launch_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    zip_buf = io.BytesIO()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Paid Pilot Launch Readiness", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(6)
    safe_pdf_cell(pdf, "Paid Pilot Decision", 8, True)
    safe_pdf_cell(pdf, f"Score: {snapshot.get('scores', {}).get('paid_pilot_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {snapshot.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Readiness: {snapshot.get('readiness_level', 'Unknown')}")
    safe_pdf_cell(pdf, f"Revenue status: {snapshot.get('revenue_status', 'Unknown')}")
    safe_pdf_cell(pdf, f"Offer tier: {snapshot.get('offer_tier', 'Unknown')}")
    safe_pdf_multicell(pdf, snapshot.get("safe_status_line", ""))
    pdf.ln(3)
    safe_pdf_cell(pdf, "Customer", 8, True)
    customer = snapshot.get("customer", {})
    safe_pdf_cell(pdf, f"Name: {customer.get('name', '')}")
    safe_pdf_cell(pdf, f"Organization: {customer.get('organization', '')}")
    safe_pdf_cell(pdf, f"Email: {customer.get('email', '')}")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Blockers / Warnings", 8, True)
    for item in snapshot.get("blockers", []) or ["No hard blockers detected."]:
        safe_pdf_multicell(pdf, f"- {item}")
    for item in snapshot.get("warnings", []):
        safe_pdf_multicell(pdf, f"- Warning: {item}")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Required Before Invoice", 8, True)
    for item in snapshot.get("required_before_invoice", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Required Before Delivery", 8, True)
    for item in snapshot.get("required_before_delivery", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Safe Claims", 8, True)
    for item in snapshot.get("safe_claims", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Claims to Avoid", 8, True)
    for item in snapshot.get("claims_to_avoid", []):
        safe_pdf_multicell(pdf, f"- {item}")

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("paid_pilot_launch_report.pdf", safe_pdf_output(pdf))
        zf.writestr("paid_pilot_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr("evidence_checks.csv", pd.DataFrame(snapshot.get("evidence_checks", [])).to_csv(index=False))
        zf.writestr("acceptance_criteria.csv", pd.DataFrame(snapshot.get("acceptance_criteria", [])).to_csv(index=False))
        zf.writestr("required_before_invoice.csv", pd.DataFrame({"required_before_invoice": snapshot.get("required_before_invoice", [])}).to_csv(index=False))
        zf.writestr("required_before_delivery.csv", pd.DataFrame({"required_before_delivery": snapshot.get("required_before_delivery", [])}).to_csv(index=False))
        zf.writestr("customer_email_draft.txt", snapshot.get("customer_email_draft", ""))
        zf.writestr("safe_claims.csv", pd.DataFrame({"safe_claim": snapshot.get("safe_claims", [])}).to_csv(index=False))
        zf.writestr("claims_to_avoid.csv", pd.DataFrame({"claim_to_avoid": snapshot.get("claims_to_avoid", [])}).to_csv(index=False))
        zf.writestr("internal_close_plan.csv", pd.DataFrame({"step": snapshot.get("internal_close_plan", [])}).to_csv(index=False))
        zf.writestr("README.txt", "EdgeTwin Studio V45 Paid Pilot Launch Readiness. This is a readiness/close pack, not a payment processor or legal contract.\n")
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr("dataset_snapshot.csv", dataset_df.head(2000).to_csv(index=False))
    return zip_buf.getvalue()


# ============================================================
# V46 CUSTOMER PROPOSAL & STATEMENT OF WORK GENERATOR
# ============================================================

def _v46_score(snapshot, keys=None, default=0):
    if not isinstance(snapshot, dict) or not snapshot:
        return int(default)
    keys = keys or [
        "proposal_score", "offer_score", "paid_pilot_score", "delivery_score", "license_score",
        "product_readiness_score", "security_score", "governance_score", "field_evidence_score",
        "trust_score", "trust_score_v2", "score",
    ]
    if "scores" in snapshot and isinstance(snapshot.get("scores"), dict):
        for key in keys:
            val = snapshot["scores"].get(key)
            if isinstance(val, (int, float)):
                return int(np.clip(val, 0, 100))
    for key in keys:
        val = snapshot.get(key)
        if isinstance(val, (int, float)):
            return int(np.clip(val, 0, 100))
    return int(default)


def build_customer_proposal_sow(
    project_name="EdgeTwin_Project",
    customer_name="Beta Customer",
    customer_organization="Customer Organization",
    customer_email="customer@example.com",
    use_case="Predictive Maintenance",
    proposed_package="Professional Pilot Bundle",
    selected_plan="Professional Pilot",
    scope_mode="Fixed-scope paid pilot",
    proposal_valid_days=14,
    delivery_window="1 week",
    dataset_df=None,
    pricing_offer_snapshot=None,
    paid_pilot_snapshot=None,
    customer_delivery_snapshot=None,
    commercial_license_certificate=None,
    field_evidence_snapshot=None,
    product_readiness_snapshot=None,
    security_snapshot=None,
    governance_snapshot=None,
    assumptions="",
    notes="",
):
    """Build a customer-ready proposal/SOW snapshot without pretending to be a legal contract."""
    rows = int(len(dataset_df)) if isinstance(dataset_df, pd.DataFrame) else 0
    cols = int(len(dataset_df.columns)) if isinstance(dataset_df, pd.DataFrame) else 0
    has_dataset = rows > 0
    has_offer = isinstance(pricing_offer_snapshot, dict) and bool(pricing_offer_snapshot)
    has_paid = isinstance(paid_pilot_snapshot, dict) and bool(paid_pilot_snapshot)
    has_delivery = isinstance(customer_delivery_snapshot, dict) and bool(customer_delivery_snapshot)
    has_license = isinstance(commercial_license_certificate, dict) and bool(commercial_license_certificate)
    has_field = isinstance(field_evidence_snapshot, dict) and bool(field_evidence_snapshot)
    has_product = isinstance(product_readiness_snapshot, dict) and bool(product_readiness_snapshot)
    has_security = isinstance(security_snapshot, dict) and bool(security_snapshot)
    has_governance = isinstance(governance_snapshot, dict) and bool(governance_snapshot)

    offer_score = _v46_score(pricing_offer_snapshot, ["offer_score"], 0)
    paid_score = _v46_score(paid_pilot_snapshot, ["paid_pilot_score"], 0)
    delivery_score = _v46_score(customer_delivery_snapshot, ["delivery_score"], 0)
    license_score = _v46_score(commercial_license_certificate, ["license_score"], 0)
    product_score = _v46_score(product_readiness_snapshot, ["product_readiness_score", "readiness_score"], 0)
    field_score = _v46_score(field_evidence_snapshot, ["field_evidence_score", "field_validation_score"], 0)
    security_score = _v46_score(security_snapshot, ["security_score"], 0)
    governance_score = _v46_score(governance_snapshot, ["governance_score", "assurance_score"], 0)

    blockers = []
    warnings_list = []
    if not has_dataset:
        blockers.append("No active dataset is available. Create or upload a dataset before sending a formal SOW.")
    if not has_offer:
        warnings_list.append("Pricing Offer V44 has not been generated yet. Price/scope may be weaker.")
    if not has_license and proposed_package not in ["Starter Pilot Bundle"]:
        warnings_list.append("Commercial License Certificate is recommended before paid delivery.")
    if proposed_package in ["Real-Data Pilot Bundle", "Enterprise Custom Review"] and not has_governance:
        blockers.append("Real-data/enterprise proposals require Customer Assurance / Data Governance language.")
    if proposed_package in ["Real-Data Pilot Bundle", "Enterprise Custom Review"] and not has_field:
        warnings_list.append("Real-data/enterprise proposal should include field evidence or clearly state that evidence collection is in scope.")
    if has_paid and paid_score < 60:
        warnings_list.append("Paid Pilot V45 score is weak. Keep the proposal conditional or reduce scope.")
    if has_security and security_score < 60:
        warnings_list.append("Security/access check is weak. Avoid customer data uploads until hardened.")

    base = 38
    base += 10 if has_dataset else 0
    base += min(12, offer_score * 0.12) if offer_score else (3 if has_offer else 0)
    base += min(14, paid_score * 0.14) if paid_score else (2 if has_paid else 0)
    base += min(10, delivery_score * 0.10) if delivery_score else (2 if has_delivery else 0)
    base += min(8, license_score * 0.08) if license_score else (2 if has_license else 0)
    base += min(10, product_score * 0.10) if product_score else (2 if has_product else 0)
    base += min(6, security_score * 0.06) if security_score else (1 if has_security else 0)
    base += min(6, governance_score * 0.06) if governance_score else (1 if has_governance else 0)
    base += min(6, field_score * 0.06) if field_score else (1 if has_field else 0)
    score = int(np.clip(base - len(blockers) * 24 - len(warnings_list) * 4, 0, 100))
    if score >= 82:
        decision = "GO"
        readiness = "Customer proposal ready"
    elif score >= 62:
        decision = "CONDITIONAL GO"
        readiness = "Proposal usable with listed conditions"
    else:
        decision = "NO-GO"
        readiness = "Do not send yet"

    included_deliverables = [
        {"deliverable": "Edge AI pilot-preparation proposal", "included": True, "notes": "Customer-facing summary and scope."},
        {"deliverable": "Statement of Work", "included": True, "notes": "Scope, milestones, responsibilities and acceptance criteria."},
        {"deliverable": "Dataset snapshot / training CSV", "included": has_dataset, "notes": "Limited to agreed pilot scope."},
        {"deliverable": "Dataset Doctor / quality audit", "included": has_dataset, "notes": "Pilot-readiness evidence, not production certification."},
        {"deliverable": "Reliability / Trust summary", "included": has_offer or has_product or has_paid, "notes": "Uses existing readiness gates where available."},
        {"deliverable": "Hardware/deployment direction", "included": proposed_package in ["Professional Pilot Bundle", "Real-Data Mini Audit", "Real-Data Pilot Bundle", "Enterprise Custom Review"], "notes": "Guidance only unless separately contracted."},
        {"deliverable": "Field evidence review", "included": proposed_package in ["Real-Data Pilot Bundle", "Enterprise Custom Review"] or has_field, "notes": "Only if real field data is available or collection is included."},
        {"deliverable": "Commercial license certificate", "included": has_license or proposed_package != "Starter Pilot Bundle", "notes": "Defines permitted delivery scope."},
    ]

    excluded_deliverables = [
        "Production guarantee or guaranteed model accuracy",
        "Safety certification or regulatory certification",
        "Autonomous production deployment without field validation",
        "Unlimited revisions outside the agreed scope",
        "Reuse of customer-uploaded data for other customers without explicit permission",
    ]
    if proposed_package not in ["Enterprise Custom Review"]:
        excluded_deliverables.append("On-premise deployment, custom integrations or enterprise procurement support")

    milestones = [
        {"phase": "1. Scope confirmation", "owner": "Both", "output": "Confirmed use-case, sensors, labels/classes and deliverables", "target": "Before start"},
        {"phase": "2. Pilot package generation", "owner": "EdgeTwin", "output": "Dataset/audit/report/export bundle within agreed scope", "target": delivery_window},
        {"phase": "3. Review & limitations", "owner": "Both", "output": "Review findings, risks, weak labels and real-sample needs", "target": "Delivery review"},
        {"phase": "4. Next-step plan", "owner": "Both", "output": "Field validation or next paid phase recommendation", "target": "After review"},
    ]
    if proposed_package in ["Real-Data Pilot Bundle", "Enterprise Custom Review"]:
        milestones.insert(1, {"phase": "2. Customer data intake", "owner": "Customer", "output": "Real WAV/CSV files and context under agreed retention/reuse policy", "target": "Before analysis"})

    responsibilities = [
        {"party": "EdgeTwin Studio", "responsibility": "Generate agreed pilot-preparation deliverables and clearly state limitations."},
        {"party": "Customer", "responsibility": "Provide accurate use-case context, sensor details, labels/classes and any real data needed."},
        {"party": "Both", "responsibility": "Review whether the output is demo-ready, pilot-ready, or needs more field evidence."},
    ]

    acceptance_criteria = [
        {"criterion": "Customer receives agreed ZIP/PDF/CSV/JSON deliverables", "required": True, "status": "ready" if decision in ["GO", "CONDITIONAL GO"] else "pending"},
        {"criterion": "Report states reliability limitations and field validation requirement", "required": True, "status": "ready"},
        {"criterion": "Scope excludes production guarantee and safety certification", "required": True, "status": "ready"},
        {"criterion": "Customer confirms next-step validation requirements", "required": True, "status": "pending"},
    ]

    required_before_signature = []
    if not has_offer:
        required_before_signature.append("Generate V44 Pricing Offer before sending final proposal.")
    if proposed_package in ["Real-Data Pilot Bundle", "Enterprise Custom Review"] and not has_governance:
        required_before_signature.append("Add customer data ownership, retention and reuse terms before accepting real data.")
    if not required_before_signature:
        required_before_signature.append("Proposal can be sent as a scoped pilot-preparation offer.")

    required_before_delivery = []
    if not has_license and proposed_package != "Starter Pilot Bundle":
        required_before_delivery.append("Generate Commercial License Certificate before delivery.")
    if not has_delivery:
        required_before_delivery.append("Run Customer Delivery Portal before final handoff.")
    if not has_product:
        required_before_delivery.append("Run V40 Product Readiness before paid delivery.")
    if not required_before_delivery:
        required_before_delivery.append("Delivery prerequisites look acceptable for controlled handoff.")

    safe_claims = [
        "EdgeTwin Studio provides Edge AI pilot-preparation deliverables, not production certification.",
        "The proposal can include dataset, audit, reliability/risk language, hardware/deployment guidance and next-step field validation planning within the agreed scope.",
        "Field validation remains required before production deployment or safety-critical use.",
        "Customer data ownership, retention and reuse follow the agreed customer assurance policy.",
    ]
    claims_to_avoid = [
        "Do not promise guaranteed accuracy, guaranteed ROI, production readiness or safety certification.",
        "Do not imply synthetic data replaces real field validation.",
        "Do not include unlimited support, revisions or integrations unless explicitly priced.",
        "Do not reuse customer-uploaded data outside the agreed scope.",
    ]

    proposal_text = f"""EdgeTwin Studio Proposal & Statement of Work

Customer: {customer_organization}
Contact: {customer_name} <{customer_email}>
Project: {project_name}
Use-case: {use_case}
Proposed package: {proposed_package}
Scope mode: {scope_mode}
Selected plan: {selected_plan}
Proposal validity: {proposal_valid_days} days
Target delivery window: {delivery_window}

Summary:
EdgeTwin Studio will provide a scoped Edge AI pilot-preparation package for the stated use-case. The package is intended to help the customer move from sensor idea/use-case to a structured pilot dataset, audit, reliability/risk summary, hardware/deployment guidance and next-step field validation plan where included.

Important limitation:
This proposal covers pilot preparation and decision support. It does not include production guarantee, safety certification, guaranteed model accuracy, or autonomous production deployment.

Decision: {decision}
Readiness: {readiness}
"""

    customer_email_draft = f"""Subject: EdgeTwin Studio proposal for {use_case}

Hi {customer_name},

Thanks for discussing the {use_case} pilot with us.

Attached is a scoped EdgeTwin Studio proposal/SOW for: {proposed_package}.
Current proposal status: {readiness} ({decision}).
Target delivery window: {delivery_window}.

The proposed work is focused on Edge AI pilot preparation: dataset/audit/reporting, reliability/risk language, hardware/deployment direction and next-step validation planning within the agreed scope.

Important limitation: this is not a production guarantee or safety certification. Field validation remains required before production deployment.

Kind regards,
EdgeTwin Studio
"""

    safe_status_line = f"{readiness}: {proposed_package} for {customer_organization}. Proposal score {score}%. Decision: {decision}."
    return {
        "version": "V46 Customer Proposal & Statement of Work Generator",
        "created_at": _now(),
        "project_name": project_name,
        "customer": {"name": customer_name, "organization": customer_organization, "email": customer_email},
        "use_case": use_case,
        "proposed_package": proposed_package,
        "selected_plan": selected_plan,
        "scope_mode": scope_mode,
        "proposal_valid_days": int(proposal_valid_days),
        "delivery_window": delivery_window,
        "dataset_rows": rows,
        "dataset_cols": cols,
        "scores": {
            "proposal_score": score,
            "offer_score": offer_score,
            "paid_pilot_score": paid_score,
            "delivery_score": delivery_score,
            "license_score": license_score,
            "product_readiness_score": product_score,
            "field_evidence_score": field_score,
            "security_score": security_score,
            "governance_score": governance_score,
        },
        "decision": decision,
        "readiness_level": readiness,
        "safe_status_line": safe_status_line,
        "included_deliverables": included_deliverables,
        "excluded_deliverables": excluded_deliverables,
        "milestones": milestones,
        "responsibilities": responsibilities,
        "acceptance_criteria": acceptance_criteria,
        "required_before_signature": required_before_signature,
        "required_before_delivery": required_before_delivery,
        "blockers": blockers,
        "warnings": warnings_list,
        "safe_claims": safe_claims,
        "claims_to_avoid": claims_to_avoid,
        "proposal_text": proposal_text,
        "customer_email_draft": customer_email_draft,
        "assumptions": str(assumptions or ""),
        "internal_notes": str(notes or ""),
        "disclaimer": "This SOW generator supports commercial scoping. It is not legal advice, a payment processor, production guarantee or safety certification.",
    }


def create_customer_proposal_sow_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    zip_buf = io.BytesIO()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Proposal & Statement of Work", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.ln(6)
    safe_pdf_cell(pdf, "Proposal Decision", 8, True)
    safe_pdf_cell(pdf, f"Score: {snapshot.get('scores', {}).get('proposal_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {snapshot.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Readiness: {snapshot.get('readiness_level', 'Unknown')}")
    safe_pdf_cell(pdf, f"Package: {snapshot.get('proposed_package', 'Unknown')}")
    safe_pdf_multicell(pdf, snapshot.get("safe_status_line", ""))
    pdf.ln(3)
    safe_pdf_cell(pdf, "Customer", 8, True)
    customer = snapshot.get("customer", {})
    safe_pdf_cell(pdf, f"Name: {customer.get('name', '')}")
    safe_pdf_cell(pdf, f"Organization: {customer.get('organization', '')}")
    safe_pdf_cell(pdf, f"Email: {customer.get('email', '')}")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Scope Summary", 8, True)
    safe_pdf_multicell(pdf, snapshot.get("proposal_text", "")[:2500])
    pdf.ln(3)
    safe_pdf_cell(pdf, "Blockers / Warnings", 8, True)
    for item in snapshot.get("blockers", []) or ["No hard blockers detected."]:
        safe_pdf_multicell(pdf, f"- {item}")
    for item in snapshot.get("warnings", []):
        safe_pdf_multicell(pdf, f"- Warning: {item}")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Required Before Signature", 8, True)
    for item in snapshot.get("required_before_signature", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Required Before Delivery", 8, True)
    for item in snapshot.get("required_before_delivery", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Safe Claims", 8, True)
    for item in snapshot.get("safe_claims", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(3)
    safe_pdf_cell(pdf, "Claims to Avoid", 8, True)
    for item in snapshot.get("claims_to_avoid", []):
        safe_pdf_multicell(pdf, f"- {item}")

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("proposal_sow_report.pdf", safe_pdf_output(pdf))
        zf.writestr("proposal_sow_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr("proposal_text.txt", snapshot.get("proposal_text", ""))
        zf.writestr("customer_email_draft.txt", snapshot.get("customer_email_draft", ""))
        zf.writestr("included_deliverables.csv", pd.DataFrame(snapshot.get("included_deliverables", [])).to_csv(index=False))
        zf.writestr("milestones.csv", pd.DataFrame(snapshot.get("milestones", [])).to_csv(index=False))
        zf.writestr("responsibilities.csv", pd.DataFrame(snapshot.get("responsibilities", [])).to_csv(index=False))
        zf.writestr("acceptance_criteria.csv", pd.DataFrame(snapshot.get("acceptance_criteria", [])).to_csv(index=False))
        zf.writestr("required_before_signature.csv", pd.DataFrame({"required_before_signature": snapshot.get("required_before_signature", [])}).to_csv(index=False))
        zf.writestr("required_before_delivery.csv", pd.DataFrame({"required_before_delivery": snapshot.get("required_before_delivery", [])}).to_csv(index=False))
        zf.writestr("safe_claims.csv", pd.DataFrame({"safe_claim": snapshot.get("safe_claims", [])}).to_csv(index=False))
        zf.writestr("claims_to_avoid.csv", pd.DataFrame({"claim_to_avoid": snapshot.get("claims_to_avoid", [])}).to_csv(index=False))
        zf.writestr("README.txt", "EdgeTwin Studio V46 Proposal & SOW bundle. This is commercial scoping support, not legal advice, payment processing, production guarantee or safety certification.\n")
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr("dataset_snapshot.csv", dataset_df.head(2000).to_csv(index=False))
    return zip_buf.getvalue()

# ============================================================
# V47 QUOTE-TO-CASH / INVOICE READINESS CENTER
# ============================================================

V47_CURRENCY_SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£", "HUF": "Ft"}


def _safe_money(value, fallback=0.0):
    try:
        return float(value)
    except Exception:
        return float(fallback)


def build_quote_to_cash_invoice_readiness(
    customer_name="Beta Customer",
    customer_org="Customer Organization",
    customer_email="customer@example.com",
    offer_tier="Professional Pilot Bundle",
    currency="EUR",
    quoted_price=499.0,
    vat_rate=0.0,
    payment_terms="Due on receipt",
    payment_status="Quote requested",
    invoice_details_status="Missing",
    legal_scope_status="Draft scope",
    delivery_status="Not delivered yet",
    license_status="Not issued",
    customer_po_required=False,
    customer_po_status="Not required",
    paid_pilot_snapshot=None,
    proposal_sow_snapshot=None,
    pricing_offer_snapshot=None,
    delivery_snapshot=None,
    license_certificate=None,
):
    """Create a practical quote-to-cash readiness snapshot.

    This does not process payments or create legal/accounting advice. It is an operator gate
    that helps decide whether a paid pilot is quote-ready, invoice-ready or delivery-ready.
    """
    paid_pilot_snapshot = paid_pilot_snapshot or {}
    proposal_sow_snapshot = proposal_sow_snapshot or {}
    pricing_offer_snapshot = pricing_offer_snapshot or {}
    delivery_snapshot = delivery_snapshot or {}
    license_certificate = license_certificate or {}

    price = _safe_money(quoted_price, 0)
    vat = max(0.0, _safe_money(vat_rate, 0.0))
    vat_amount = round(price * (vat / 100.0), 2)
    total = round(price + vat_amount, 2)
    symbol = V47_CURRENCY_SYMBOLS.get(currency, currency + " ")

    blockers = []
    warnings = []
    ready_checks = []

    def check(name, ok, required=True, message_if_missing=""):
        item = {"check": name, "status": "PASS" if ok else ("BLOCKER" if required else "WARNING"), "required": bool(required)}
        ready_checks.append(item)
        if not ok and required:
            blockers.append(message_if_missing or name)
        elif not ok:
            warnings.append(message_if_missing or name)

    has_pricing = bool(pricing_offer_snapshot) or price > 0
    has_sow = bool(proposal_sow_snapshot) or legal_scope_status in {"Scope reviewed", "Scope accepted", "Signed / accepted"}
    has_paid_gate = bool(paid_pilot_snapshot)
    has_delivery = bool(delivery_snapshot)
    has_license = bool(license_certificate) or license_status in {"Draft issued", "Issued", "Signed / accepted"}

    check("Pricing offer exists", has_pricing, True, "Create or confirm a V44 pricing offer before invoicing.")
    check("Proposal/SOW scope exists", has_sow, True, "Create or review the V46 Proposal/SOW before invoicing.")
    check("Paid pilot gate exists", has_paid_gate or payment_status in {"Quote accepted", "Invoice sent", "Paid / invoice confirmed"}, False, "Run V45 Paid Pilot gate before delivery.")
    check("Customer invoice details present", invoice_details_status in {"Received", "Verified"}, True, "Collect and verify customer invoice details before sending invoice.")
    check("Legal/commercial scope accepted", legal_scope_status in {"Scope accepted", "Signed / accepted"}, True, "Get scope/SOW accepted before taking payment or delivery.")
    check("License/certificate ready", has_license, False, "Issue commercial license certificate before final delivery.")
    check("Delivery portal ready", has_delivery or delivery_status in {"Delivery bundle ready", "Delivered"}, False, "Build V42 Customer Delivery bundle before final hand-off.")
    if customer_po_required:
        check("Customer PO received", customer_po_status in {"Received", "Approved"}, True, "Customer requires a PO; receive/approve it before invoicing.")
    else:
        check("PO not required or waived", True, False, "")

    if price <= 0:
        blockers.append("Quoted price must be above zero for a paid pilot.")
    if "enterprise" in offer_tier.lower() and legal_scope_status != "Signed / accepted":
        warnings.append("Enterprise/custom offers should normally have signed scope and explicit acceptance criteria.")
    if vat > 0 and invoice_details_status != "Verified":
        warnings.append("VAT/tax fields should be verified before sending the invoice.")

    score = 100
    score -= len(blockers) * 18
    score -= len(warnings) * 7
    if payment_status == "Paid / invoice confirmed":
        score += 8
    elif payment_status == "Invoice sent":
        score += 4
    elif payment_status == "Not discussed":
        score -= 8
    score = int(np.clip(score, 0, 100))

    if blockers:
        decision = "NO-GO"
        revenue_status = "Not invoice-ready"
    elif score >= 85 and payment_status in {"Quote accepted", "Invoice sent", "Paid / invoice confirmed"}:
        decision = "GO"
        revenue_status = "Invoice/delivery-ready" if payment_status != "Paid / invoice confirmed" else "Paid delivery-ready"
    elif score >= 70:
        decision = "CONDITIONAL GO"
        revenue_status = "Quote-ready; fix warnings before delivery"
    else:
        decision = "NO-GO"
        revenue_status = "Commercial preparation required"

    line_items = [
        {
            "item": offer_tier,
            "description": "EdgeTwin Studio paid pilot package: dataset/audit/reliability/hardware/report delivery within agreed SOW scope.",
            "quantity": 1,
            "unit_price": round(price, 2),
            "currency": currency,
            "subtotal": round(price, 2),
        },
        {
            "item": "VAT / tax",
            "description": f"Tax estimate at {vat}% if applicable. Verify locally before issuing formal invoice.",
            "quantity": 1,
            "unit_price": vat_amount,
            "currency": currency,
            "subtotal": vat_amount,
        },
    ]

    payment_checklist = [
        "Confirm customer organization legal name, billing email and billing address.",
        "Confirm offer tier, price, currency, VAT/tax handling and payment terms.",
        "Attach Proposal/SOW or accepted scope reference to the payment request.",
        "Send invoice/payment request only after invoice details and scope are accepted.",
        "Do not deliver premium/final bundles before payment confirmation unless intentionally agreed.",
        "Store delivery certificate, license certificate and acceptance criteria with the project.",
    ]

    quote_terms = [
        "This quote covers Edge AI pilot preparation and decision-support deliverables only.",
        "Production deployment, safety certification, on-site installation and legal compliance validation are excluded unless explicitly agreed.",
        "Synthetic and real-data-derived outputs require field validation before production use.",
        "Uploaded customer data remains customer-owned unless a separate written agreement states otherwise.",
        "Any prices, VAT/tax treatment and payment terms should be verified before formal invoicing.",
    ]

    customer_email_draft = f"""Subject: EdgeTwin Studio paid pilot quote - {offer_tier}

Hi {customer_name},

Based on the current scope, the recommended paid pilot package is: {offer_tier}.

Estimated pilot price: {symbol}{price:,.2f}
Estimated VAT/tax: {symbol}{vat_amount:,.2f}
Estimated total: {symbol}{total:,.2f}
Payment terms: {payment_terms}

The pilot includes Edge AI pilot-preparation outputs such as dataset/audit/reliability review, hardware/deployment direction and customer-ready reporting within the agreed scope. It does not certify production readiness; field validation remains required before production deployment.

Before we invoice or deliver final bundles, we should confirm the billing details, accepted scope/SOW and any purchase order requirements.

Best regards,
EdgeTwin Studio
"""

    safe_claims = [
        "We can prepare a paid Edge AI pilot package with clearly scoped deliverables.",
        "We can provide dataset, reliability, hardware and reporting support for pilot decision-making.",
        "Final delivery is gated by accepted scope, invoice details and safe-use conditions.",
    ]
    claims_to_avoid = [
        "Do not claim payment guarantees production readiness.",
        "Do not claim the invoice/certificate is legal or tax advice.",
        "Do not deliver safety-critical or production claims without field validation and explicit agreement.",
    ]

    return {
        "version": "V47 Quote-to-Cash & Invoice Readiness Center",
        "created_at": _now(),
        "customer": {"name": customer_name, "organization": customer_org, "email": customer_email},
        "offer_tier": offer_tier,
        "currency": currency,
        "quoted_price": round(price, 2),
        "vat_rate": vat,
        "vat_amount": vat_amount,
        "total_amount": total,
        "payment_terms": payment_terms,
        "payment_status": payment_status,
        "invoice_details_status": invoice_details_status,
        "legal_scope_status": legal_scope_status,
        "delivery_status": delivery_status,
        "license_status": license_status,
        "customer_po_required": bool(customer_po_required),
        "customer_po_status": customer_po_status,
        "quote_to_cash_score": score,
        "decision": decision,
        "revenue_status": revenue_status,
        "ready_checks": ready_checks,
        "blockers": blockers,
        "warnings": warnings,
        "line_items": line_items,
        "payment_checklist": payment_checklist,
        "quote_terms": quote_terms,
        "customer_email_draft": customer_email_draft,
        "safe_claims": safe_claims,
        "claims_to_avoid": claims_to_avoid,
        "inputs_present": {
            "pricing_offer_snapshot": bool(pricing_offer_snapshot),
            "proposal_sow_snapshot": bool(proposal_sow_snapshot),
            "paid_pilot_snapshot": bool(paid_pilot_snapshot),
            "delivery_snapshot": bool(delivery_snapshot),
            "license_certificate": bool(license_certificate),
        },
        "disclaimer": "V47 supports quote-to-cash readiness and commercial hygiene. It is not legal, tax, accounting or payment-processing advice.",
    }


def create_quote_to_cash_invoice_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Quote-to-Cash Readiness", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.cell(0, 8, txt=clean_pdf_text(snapshot.get("version", "V47")), ln=True, align="C")
    pdf.ln(6)

    safe_pdf_cell(pdf, "Revenue Decision", 8, True)
    safe_pdf_cell(pdf, f"Score: {snapshot.get('quote_to_cash_score', 0)}%")
    safe_pdf_cell(pdf, f"Decision: {snapshot.get('decision', 'Unknown')}")
    safe_pdf_cell(pdf, f"Revenue status: {snapshot.get('revenue_status', 'Unknown')}")
    safe_pdf_cell(pdf, f"Offer tier: {snapshot.get('offer_tier', 'Unknown')}")
    safe_pdf_cell(pdf, f"Total: {snapshot.get('currency', 'EUR')} {snapshot.get('total_amount', 0)}")
    pdf.ln(4)

    cust = snapshot.get("customer", {})
    safe_pdf_cell(pdf, "Customer", 8, True)
    safe_pdf_cell(pdf, f"Name: {cust.get('name', '')}")
    safe_pdf_cell(pdf, f"Organization: {cust.get('organization', '')}")
    safe_pdf_cell(pdf, f"Email: {cust.get('email', '')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Line Items", 8, True)
    for item in snapshot.get("line_items", []):
        safe_pdf_multicell(pdf, f"- {item.get('item')}: {item.get('currency')} {item.get('subtotal')} | {item.get('description')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Blockers / Warnings", 8, True)
    blockers = snapshot.get("blockers", [])
    warnings = snapshot.get("warnings", [])
    if blockers:
        for item in blockers:
            safe_pdf_multicell(pdf, f"[BLOCKER] {item}")
    else:
        safe_pdf_multicell(pdf, "No hard invoice blockers detected.")
    for item in warnings:
        safe_pdf_multicell(pdf, f"[WARNING] {item}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Payment Checklist", 8, True)
    for item in snapshot.get("payment_checklist", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Safe Terms", 8, True)
    for item in snapshot.get("quote_terms", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", ""))

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("quote_to_cash_readiness.pdf", safe_pdf_output(pdf))
        zf.writestr("quote_to_cash_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr("invoice_line_items.csv", pd.DataFrame(snapshot.get("line_items", [])).to_csv(index=False))
        zf.writestr("ready_checks.csv", pd.DataFrame(snapshot.get("ready_checks", [])).to_csv(index=False))
        zf.writestr("payment_checklist.csv", pd.DataFrame({"item": snapshot.get("payment_checklist", [])}).to_csv(index=False))
        zf.writestr("safe_quote_terms.csv", pd.DataFrame({"term": snapshot.get("quote_terms", [])}).to_csv(index=False))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr("dataset_snapshot.csv", dataset_df.head(1000).to_csv(index=False))
        zf.writestr("README.txt", "EdgeTwin Studio V47 Quote-to-Cash bundle. This is commercial readiness support, not payment processing, legal, tax or accounting advice.\n")
    return zip_buf.getvalue()



# ============================================================
# V48 CUSTOMER LEAD INTAKE / QUALIFICATION CENTER
# ============================================================

def _v48_score_value(value, table, default=0):
    try:
        return int(table.get(value, default))
    except Exception:
        return int(default)


def build_customer_lead_intake_qualification(
    customer_name="Beta Customer",
    customer_org="Customer Organization",
    customer_email="customer@example.com",
    industry="Manufacturing",
    use_case="Predictive maintenance",
    urgency="Pilot in 1-3 months",
    budget_range="€500-€1,500",
    decision_status="Technical evaluator",
    data_status="Can collect data",
    sensor_status="Need hardware advice",
    deployment_environment="Indoor machine",
    technical_owner="Part-time technical owner",
    requested_outcome="",
    safety_critical=False,
    regulated_context=False,
    needs_installation=False,
    wants_guaranteed_accuracy=False,
    pricing_offer_snapshot=None,
    proposal_sow_snapshot=None,
    quote_to_cash_snapshot=None,
    paid_pilot_snapshot=None,
):
    """Qualify an incoming lead and decide the safest next commercial/product step.

    This is intentionally conservative: it rewards clear data, budget, decision access and technical owner,
    while flagging safety-critical or guaranteed-accuracy expectations before a paid pilot is sold.
    """
    pricing_offer_snapshot = pricing_offer_snapshot or {}
    proposal_sow_snapshot = proposal_sow_snapshot or {}
    quote_to_cash_snapshot = quote_to_cash_snapshot or {}
    paid_pilot_snapshot = paid_pilot_snapshot or {}

    urgency_score = _v48_score_value(urgency, {
        "Exploring only": 4,
        "Pilot in 1-3 months": 12,
        "Pilot this month": 16,
        "Urgent operational pain": 14,
    })
    budget_score = _v48_score_value(budget_range, {
        "Unknown": 3,
        "Below €500": 4,
        "€500-€1,500": 12,
        "€1,500-€5,000": 16,
        "€5,000+": 18,
    })
    decision_score = _v48_score_value(decision_status, {
        "Just researching": 4,
        "Technical evaluator": 10,
        "Budget owner involved": 15,
        "Decision maker involved": 18,
    })
    data_score = _v48_score_value(data_status, {
        "No data yet": 3,
        "Can collect data": 11,
        "Sample data available": 16,
        "Labeled dataset available": 20,
    })
    sensor_score = _v48_score_value(sensor_status, {
        "Unknown": 3,
        "Need hardware advice": 9,
        "Sensors selected": 13,
        "Hardware already installed": 16,
    })
    owner_score = _v48_score_value(technical_owner, {
        "Unknown": 3,
        "No technical owner": 2,
        "Part-time technical owner": 10,
        "Dedicated technical owner": 14,
    })

    use_case_bonus = 4 if use_case in {"Predictive maintenance", "Anomaly detection", "Machine health", "Tamper detection", "Security / intrusion"} else 2
    environment_bonus = 4 if deployment_environment in {"Indoor machine", "Outdoor asset", "Remote/off-grid"} else 2

    warnings = []
    blockers = []

    if safety_critical:
        warnings.append("Safety-critical context detected: keep the offer as a pilot/decision-support package until field validation and responsibilities are explicit.")
    if regulated_context:
        warnings.append("Regulated/compliance-heavy context detected: require explicit scope, data handling and compliance owner before delivery.")
    if needs_installation:
        warnings.append("Customer expects on-site installation: keep installation outside scope unless you have a clear partner/process.")
    if wants_guaranteed_accuracy:
        blockers.append("Customer expects guaranteed accuracy/production certainty. Reframe before proposal; EdgeTwin should not sell guarantees without field validation.")
    if budget_range in {"Unknown", "Below €500"} and urgency not in {"Exploring only"}:
        warnings.append("Urgency exists but budget signal is weak. Qualify budget before spending founder time.")
    if data_status == "No data yet" and sensor_status == "Unknown":
        warnings.append("No data and unknown sensors: start with demo/hardware discovery, not a custom paid pilot yet.")
    if technical_owner in {"Unknown", "No technical owner"}:
        warnings.append("No clear technical owner: request a named contact before deeper pilot work.")

    score = urgency_score + budget_score + decision_score + data_score + sensor_score + owner_score + use_case_bonus + environment_bonus
    score -= len(warnings) * 5
    score -= len(blockers) * 25
    if pricing_offer_snapshot:
        score += 3
    if proposal_sow_snapshot:
        score += 3
    if paid_pilot_snapshot:
        score += 3
    score = int(np.clip(score, 0, 100))

    if blockers:
        fit_level = "Not ready / reframe first"
        recommended_route = "Reframe expectations before proposal"
        operator_verdict = "Do not move this lead into a paid pilot until the unsafe/guaranteed claims are corrected."
    elif score >= 80:
        fit_level = "High-fit paid pilot"
        recommended_route = "Move to Proposal/SOW + Pricing Offer"
        operator_verdict = "Strong lead. Prepare V46 Proposal/SOW and V44 Pricing Offer, then use V47 before invoicing."
    elif score >= 65:
        fit_level = "Good-fit pilot"
        recommended_route = "Run demo + collect missing data"
        operator_verdict = "Good lead, but qualify missing data/scope before invoice."
    elif score >= 45:
        fit_level = "Nurture / needs clarity"
        recommended_route = "Send discovery questions and lightweight demo"
        operator_verdict = "Do not spend heavy custom time yet. Use demo and discovery to improve the lead."
    else:
        fit_level = "Low-fit for now"
        recommended_route = "Nurture or decline custom work"
        operator_verdict = "This lead is not ready for a paid pilot without much clearer scope, data, owner or budget."

    commercial_stage = "Lead qualified"
    if quote_to_cash_snapshot and quote_to_cash_snapshot.get("decision") in {"GO", "CONDITIONAL GO"}:
        commercial_stage = "Commercial handoff started"
    elif proposal_sow_snapshot:
        commercial_stage = "Proposal/SOW drafted"
    elif pricing_offer_snapshot:
        commercial_stage = "Pricing drafted"
    elif paid_pilot_snapshot:
        commercial_stage = "Paid pilot gate available"

    discovery_questions = [
        f"Which exact asset/process should the {use_case.lower()} pilot monitor first?",
        "What does a normal condition look like, and what abnormal events matter most?",
        "Do you already have CSV/WAV/sensor logs, and are they labeled?",
        "Which sensors are already installed or approved for the pilot?",
        "Who owns technical validation and who signs off on the pilot result?",
        "What would make the pilot valuable enough to continue after the first delivery?",
        "Are there safety, compliance, privacy or installation constraints we must exclude from scope?",
    ]

    customer_required_inputs = [
        "One clear asset/process for the first pilot scope.",
        "At least one example of normal and abnormal behavior, or permission to generate a synthetic pilot dataset.",
        "Sensor/hardware constraints or permission to recommend hardware.",
        "Named technical contact and decision/budget contact.",
        "Acceptance criteria for what the pilot must prove.",
    ]
    if data_status in {"Sample data available", "Labeled dataset available"}:
        customer_required_inputs.insert(0, "Upload representative CSV/WAV/sensor samples for real-data review.")
    if needs_installation:
        customer_required_inputs.append("Confirm whether installation is handled by customer/partner; default EdgeTwin scope excludes installation.")

    next_actions = []
    if fit_level == "High-fit paid pilot":
        next_actions = [
            "Create V46 Proposal/SOW with explicit included/excluded deliverables.",
            "Create V44 Pricing Offer and confirm paid pilot tier.",
            "Run V47 Quote-to-Cash before invoice or delivery.",
        ]
    elif fit_level == "Good-fit pilot":
        next_actions = [
            "Run self-selling demo or Use Case Wizard for this use case.",
            "Collect missing data/sensor/owner details using the discovery questions.",
            "Then create pricing/proposal only if scope and acceptance criteria are clear.",
        ]
    elif fit_level == "Nurture / needs clarity":
        next_actions = [
            "Send discovery email and ask for data/scope/owner details.",
            "Avoid custom engineering until budget, data and decision access improve.",
            "Offer a lightweight demo or Real-Data Mini Audit if suitable.",
        ]
    else:
        next_actions = [
            "Reframe unsafe/unclear expectations before any paid offer.",
            "Decline guaranteed-production claims and keep pilot language honest.",
            "Only continue if the customer accepts field-validation and scope boundaries.",
        ]

    sales_followup_checklist = [
        "Log lead source, customer problem and requested outcome.",
        "Confirm data availability and hardware status.",
        "Confirm budget/decision status before custom work.",
        "Choose route: demo, mini audit, paid pilot, or nurture.",
        "Do not promise production accuracy, safety certification or legal compliance.",
    ]

    safe_claims = [
        "We can assess whether your sensor idea is suitable for an Edge AI pilot.",
        "We can generate pilot datasets, reliability checks, hardware direction and customer-ready reports.",
        "We can define a paid pilot scope with clear acceptance criteria and exclusions.",
    ]
    claims_to_avoid = [
        "Do not guarantee production accuracy or uptime from a pilot.",
        "Do not claim certification, safety validation or regulatory compliance unless separately proven.",
        "Do not include on-site installation, custom hardware manufacturing or legal/tax advice by default.",
    ]

    customer_email_draft = f"""Subject: EdgeTwin Studio pilot intake - next step for {customer_org}

Hi {customer_name},

Thanks for sharing the use case around {use_case.lower()} in {industry}.

Based on the first intake, the recommended route is: {recommended_route}.
Fit level: {fit_level}
Qualification score: {score}%

To keep the pilot useful and safe, I would first like to confirm:
1. The exact asset/process for the first pilot scope
2. Any available CSV/WAV/sensor samples or examples of normal vs abnormal behavior
3. The sensors/hardware you already have or plan to use
4. The technical owner and decision owner for the pilot
5. The acceptance criteria for a successful first delivery

Important: EdgeTwin Studio prepares pilot-ready decision-support deliverables. It does not certify production readiness or guarantee accuracy without field validation.

Best regards,
EdgeTwin Studio
"""

    routing = {
        "recommended_route": recommended_route,
        "open_next_in_app": "Proposal / SOW" if fit_level == "High-fit paid pilot" else ("Use Case Wizard" if fit_level == "Good-fit pilot" else "Self-Selling Demo"),
        "pricing_allowed": fit_level in {"High-fit paid pilot", "Good-fit pilot"} and not blockers,
        "invoice_allowed": bool(quote_to_cash_snapshot and quote_to_cash_snapshot.get("decision") == "GO"),
        "custom_work_allowed": fit_level in {"High-fit paid pilot", "Good-fit pilot"} and budget_range not in {"Unknown", "Below €500"},
    }

    return {
        "version": "V48 Customer Lead Intake & Qualification Center",
        "created_at": _now(),
        "customer": {"name": customer_name, "organization": customer_org, "email": customer_email},
        "industry": industry,
        "use_case": use_case,
        "urgency": urgency,
        "budget_range": budget_range,
        "decision_status": decision_status,
        "data_status": data_status,
        "sensor_status": sensor_status,
        "deployment_environment": deployment_environment,
        "technical_owner": technical_owner,
        "requested_outcome": requested_outcome,
        "risk_flags": {
            "safety_critical": bool(safety_critical),
            "regulated_context": bool(regulated_context),
            "needs_installation": bool(needs_installation),
            "wants_guaranteed_accuracy": bool(wants_guaranteed_accuracy),
        },
        "qualification_score": score,
        "fit_level": fit_level,
        "recommended_route": recommended_route,
        "commercial_stage": commercial_stage,
        "operator_verdict": operator_verdict,
        "blockers": blockers,
        "warnings": warnings,
        "discovery_questions": discovery_questions,
        "customer_required_inputs": customer_required_inputs,
        "next_actions": next_actions,
        "sales_followup_checklist": sales_followup_checklist,
        "safe_claims": safe_claims,
        "claims_to_avoid": claims_to_avoid,
        "customer_email_draft": customer_email_draft,
        "routing": routing,
        "connected_snapshots": {
            "pricing_offer_snapshot": bool(pricing_offer_snapshot),
            "proposal_sow_snapshot": bool(proposal_sow_snapshot),
            "quote_to_cash_snapshot": bool(quote_to_cash_snapshot),
            "paid_pilot_snapshot": bool(paid_pilot_snapshot),
        },
        "disclaimer": "V48 supports lead qualification and safe sales routing. It is not legal advice, not a guarantee of revenue, and not production validation.",
    }


def create_customer_lead_intake_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Lead Intake Qualification", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.cell(0, 8, txt=clean_pdf_text(snapshot.get("version", "V48")), ln=True, align="C")
    pdf.ln(6)

    safe_pdf_cell(pdf, "Qualification Result", 8, True)
    safe_pdf_cell(pdf, f"Score: {snapshot.get('qualification_score', 0)}%")
    safe_pdf_cell(pdf, f"Fit level: {snapshot.get('fit_level', 'Unknown')}")
    safe_pdf_cell(pdf, f"Recommended route: {snapshot.get('recommended_route', 'Unknown')}")
    safe_pdf_multicell(pdf, f"Verdict: {snapshot.get('operator_verdict', '')}")
    pdf.ln(4)

    customer = snapshot.get("customer", {})
    safe_pdf_cell(pdf, "Customer", 8, True)
    safe_pdf_cell(pdf, f"Name: {customer.get('name', '')}")
    safe_pdf_cell(pdf, f"Organization: {customer.get('organization', '')}")
    safe_pdf_cell(pdf, f"Email: {customer.get('email', '')}")
    safe_pdf_cell(pdf, f"Industry: {snapshot.get('industry', '')}")
    safe_pdf_cell(pdf, f"Use case: {snapshot.get('use_case', '')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Next Actions", 8, True)
    for item in snapshot.get("next_actions", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Discovery Questions", 8, True)
    for item in snapshot.get("discovery_questions", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Risks", 8, True)
    blockers = snapshot.get("blockers", [])
    warnings = snapshot.get("warnings", [])
    if blockers:
        for item in blockers:
            safe_pdf_multicell(pdf, f"[BLOCKER] {item}")
    else:
        safe_pdf_multicell(pdf, "No hard lead blockers detected.")
    for item in warnings:
        safe_pdf_multicell(pdf, f"[WARNING] {item}")
    pdf.ln(4)
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", ""))

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("lead_intake_qualification.pdf", safe_pdf_output(pdf))
        zf.writestr("lead_intake_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr("discovery_questions.csv", pd.DataFrame({"question": snapshot.get("discovery_questions", [])}).to_csv(index=False))
        zf.writestr("next_actions.csv", pd.DataFrame({"action": snapshot.get("next_actions", [])}).to_csv(index=False))
        zf.writestr("customer_required_inputs.csv", pd.DataFrame({"input": snapshot.get("customer_required_inputs", [])}).to_csv(index=False))
        zf.writestr("sales_followup_checklist.csv", pd.DataFrame({"item": snapshot.get("sales_followup_checklist", [])}).to_csv(index=False))
        zf.writestr("customer_email_draft.txt", snapshot.get("customer_email_draft", ""))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr("dataset_snapshot.csv", dataset_df.head(1000).to_csv(index=False))
        zf.writestr("README.txt", "EdgeTwin Studio V48 Lead Intake bundle. Use this to qualify leads and route them safely before pricing/proposal/invoice steps. Not legal advice or production validation.\n")
    return zip_buf.getvalue()


# ============================================================
# V49 FOUNDER OPS / FOLLOW-UP AUTOMATION CENTER
# ============================================================

def _v49_bool_score(value, points):
    return points if bool(value) else 0


def build_founder_ops_followup_center(
    customer_name="Beta Customer",
    customer_org="Customer Organization",
    customer_email="customer@example.com",
    current_stage="Discovery",
    founder_hours_available=8,
    max_customer_hours=3,
    urgency="Normal",
    customer_responsiveness="Normal",
    payment_status="Not discussed",
    data_status="Customer can collect",
    next_deadline="None",
    support_load="Normal",
    waiting_on_customer=True,
    needs_follow_up=True,
    unclear_scope=False,
    high_custom_work=False,
    unsafe_expectation=False,
    outstanding_customer_inputs=None,
    internal_notes="",
    lead_intake_snapshot=None,
    pricing_offer_snapshot=None,
    proposal_sow_snapshot=None,
    quote_to_cash_snapshot=None,
    paid_pilot_snapshot=None,
    delivery_snapshot=None,
    customer_success_snapshot=None,
    observability_snapshot=None,
):
    """Build a founder-friendly operating plan for customer follow-up and workload control."""
    outstanding_customer_inputs = outstanding_customer_inputs or []

    stage_order = {
        "New lead": 5,
        "Discovery": 15,
        "Pricing": 30,
        "Proposal/SOW": 45,
        "Quote sent": 60,
        "Awaiting payment": 70,
        "Paid pilot": 80,
        "Delivery": 90,
        "Post-delivery follow-up": 95,
    }
    urgency_load = {"Low": 0, "Normal": 5, "High": 12, "Critical": 20}.get(urgency, 5)
    deadline_load = {"None": 0, "This week": 7, "48 hours": 14, "Today": 22}.get(next_deadline, 0)
    support_load_score = {"Low": 0, "Normal": 6, "High": 14, "Too high": 24}.get(support_load, 6)
    response_bonus = {"Unknown": 0, "Slow": -5, "Normal": 4, "Fast": 8}.get(customer_responsiveness, 0)

    readiness_score = stage_order.get(current_stage, 15)
    readiness_score += _v49_bool_score(lead_intake_snapshot, 8)
    readiness_score += _v49_bool_score(pricing_offer_snapshot, 7)
    readiness_score += _v49_bool_score(proposal_sow_snapshot, 7)
    readiness_score += _v49_bool_score(quote_to_cash_snapshot, 7)
    readiness_score += _v49_bool_score(paid_pilot_snapshot, 5)
    readiness_score += _v49_bool_score(delivery_snapshot, 5)
    readiness_score += _v49_bool_score(customer_success_snapshot, 4)
    readiness_score += response_bonus

    overload_points = urgency_load + deadline_load + support_load_score
    if founder_hours_available <= 4:
        overload_points += 22
    elif founder_hours_available <= 8:
        overload_points += 12
    if max_customer_hours <= 2:
        overload_points += 8
    if high_custom_work:
        overload_points += 18
    if unclear_scope:
        overload_points += 12
    if len(outstanding_customer_inputs) >= 4:
        overload_points += 8
    if unsafe_expectation:
        overload_points += 30

    blockers = []
    warnings = []
    if unsafe_expectation:
        blockers.append("Customer expects guarantees or production certainty. Reframe before spending more time.")
    if high_custom_work and payment_status not in {"Invoice ready", "Awaiting payment", "Paid"}:
        blockers.append("Customer is pulling toward custom work before payment/scope is controlled.")
    if unclear_scope:
        warnings.append("Scope is unclear. Do not create custom deliverables until the first pilot boundary is written down.")
    if waiting_on_customer and outstanding_customer_inputs:
        warnings.append("You are waiting on customer inputs. Send one clear request instead of doing more internal work.")
    if support_load in {"High", "Too high"}:
        warnings.append("Support load is high. Time-box support and convert repeated answers into templates.")
    if payment_status in {"Invoice details missing", "Overdue"}:
        warnings.append(f"Payment status needs attention: {payment_status}.")

    ops_score = int(np.clip(readiness_score - overload_points - len(blockers) * 18 - len(warnings) * 4 + 35, 0, 100))

    if blockers:
        overload_risk = "High"
        operator_mode = "Protect time / reframe"
        founder_verdict = "Do not do more custom work yet. First reframe expectations, scope and payment boundaries."
    elif overload_points >= 45 or ops_score < 45:
        overload_risk = "High"
        operator_mode = "Reduce workload"
        founder_verdict = "Workload risk is high. Only do one next action and push missing work back to the customer."
    elif overload_points >= 25 or ops_score < 70:
        overload_risk = "Medium"
        operator_mode = "Controlled follow-up"
        founder_verdict = "This is manageable if you time-box it and avoid extra unpaid custom work."
    else:
        overload_risk = "Low"
        operator_mode = "Proceed"
        founder_verdict = "This workflow is under control. Continue with the next commercial or delivery step."

    task_queue = []
    def add_task(priority, task, owner="Founder", time_box="15 min", status="Open", gate=""):
        task_queue.append({
            "priority": priority,
            "task": task,
            "owner": owner,
            "time_box": time_box,
            "status": status,
            "gate": gate,
        })

    if needs_follow_up or waiting_on_customer:
        add_task("P0", "Send one clear customer follow-up with requested inputs and next deadline.", "Founder", "10 min", gate="Customer input")
    if not lead_intake_snapshot:
        add_task("P1", "Run Lead Intake V48 before deeper custom work.", "Founder", "10 min", gate="Qualification")
    if current_stage in {"New lead", "Discovery"} and not pricing_offer_snapshot:
        add_task("P1", "Qualify budget and create Pricing Offer only after scope/data are clear.", "Founder", "20 min", gate="Budget/scope")
    if pricing_offer_snapshot and not proposal_sow_snapshot:
        add_task("P1", "Create Proposal/SOW V46 with included and excluded deliverables.", "Founder", "25 min", gate="Scope")
    if proposal_sow_snapshot and not quote_to_cash_snapshot:
        add_task("P1", "Run Quote-to-Cash V47 before invoice/payment handoff.", "Founder", "15 min", gate="Invoice readiness")
    if payment_status in {"Invoice details missing", "Quote requested"}:
        add_task("P0", "Request invoice details, PO/procurement requirements and billing contact.", "Customer", "5 min", gate="Billing info")
    if payment_status == "Awaiting payment":
        add_task("P0", "Send polite payment reminder before releasing delivery package.", "Founder", "8 min", gate="Payment")
    if payment_status == "Paid" and not delivery_snapshot:
        add_task("P0", "Generate Delivery Portal bundle and send customer handoff.", "Founder", "20 min", gate="Delivery")
    if data_status in {"No data", "Customer can collect"}:
        add_task("P1", "Ask for sample data or confirm synthetic-pilot approach.", "Customer", "5 min", gate="Data")
    if unsafe_expectation:
        add_task("P0", "Reframe: pilot-ready is not production-certified or guaranteed accuracy.", "Founder", "10 min", gate="Safe claims")
    if high_custom_work:
        add_task("P0", "Move custom requests into paid scope or backlog, not free support.", "Founder", "10 min", gate="Scope control")

    # Always include a founder-protection task.
    add_task("P2", "Update project notes and save current app state after the next customer action.", "Founder", "5 min", gate="Admin hygiene")

    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    task_queue = sorted(task_queue, key=lambda x: (priority_order.get(x["priority"], 9), x["time_box"]))
    do_now_queue = [t for t in task_queue if t["priority"] == "P0"][:4]
    if not do_now_queue and task_queue:
        do_now_queue = task_queue[:2]

    weekly_plan = [
        {"block": "Block 1", "focus": "One customer-facing follow-up", "max_time": "30 min", "rule": "Send, then wait. Do not build extra custom work."},
        {"block": "Block 2", "focus": "Commercial gate", "max_time": "45 min", "rule": "Pricing/SOW/Quote only if scope and data are clear."},
        {"block": "Block 3", "focus": "Delivery/admin", "max_time": "45 min", "rule": "Generate bundle, save project, log next action."},
    ]
    if founder_hours_available <= 4:
        weekly_plan = weekly_plan[:1]
        weekly_plan[0]["rule"] = "Only one next action this week. Customer must provide inputs before more work."

    automation_opportunities = [
        "Reuse customer email drafts from V48/V49 instead of writing every reply from scratch.",
        "Use Proposal/SOW V46 and Quote-to-Cash V47 as gates before invoice or delivery.",
        "Convert repeated customer questions into templates inside the Delivery Portal / Customer Success flow.",
        "Batch admin work once per week instead of reacting all day.",
    ]
    if observability_snapshot:
        automation_opportunities.append("Use Error Observatory signals to prioritize only product issues that affect customer delivery.")

    stop_doing_list = [
        "Do not perform unpaid custom engineering before scope, price and acceptance criteria are clear.",
        "Do not chase unclear leads more than once without customer input.",
        "Do not promise production accuracy, safety certification or uptime from a pilot.",
        "Do not manually rewrite the same emails; use the generated drafts and adjust lightly.",
    ]
    if overload_risk == "High":
        stop_doing_list.insert(0, "Stop adding new features for this customer until the next commercial/customer input gate is cleared.")

    missing_inputs_text = "\n".join([f"- {x}" for x in outstanding_customer_inputs]) or "- No specific missing inputs listed."
    customer_followup_email = f"""Subject: Next step for {customer_org} EdgeTwin pilot

Hi {customer_name},

To keep the EdgeTwin pilot efficient and clearly scoped, the best next step is to close the missing inputs below:

{missing_inputs_text}

Current stage: {current_stage}
Recommended next action: {do_now_queue[0]['task'] if do_now_queue else 'Confirm the next pilot step'}

Once these are confirmed, I can move the project through the correct gate: lead qualification, pricing, proposal/SOW, quote/payment readiness, or delivery.

Important: EdgeTwin prepares pilot-ready decision-support deliverables. Production readiness, guaranteed accuracy, safety certification and regulatory compliance require separate field validation and are not promised by default.

Best regards,
EdgeTwin Studio
"""

    internal_operator_note = f"""Founder Ops V49 note
Customer: {customer_org} / {customer_name}
Stage: {current_stage}
Operator mode: {operator_mode}
Overload risk: {overload_risk}
Ops score: {ops_score}%
Founder rule: {founder_verdict}

Internal notes:
{internal_notes}
"""

    return {
        "version": "V49 Founder Ops & Customer Follow-up Center",
        "created_at": _now(),
        "customer": {"name": customer_name, "organization": customer_org, "email": customer_email},
        "current_stage": current_stage,
        "founder_hours_available": founder_hours_available,
        "max_customer_hours": max_customer_hours,
        "urgency": urgency,
        "customer_responsiveness": customer_responsiveness,
        "payment_status": payment_status,
        "data_status": data_status,
        "next_deadline": next_deadline,
        "support_load": support_load,
        "risk_flags": {
            "waiting_on_customer": bool(waiting_on_customer),
            "needs_follow_up": bool(needs_follow_up),
            "unclear_scope": bool(unclear_scope),
            "high_custom_work": bool(high_custom_work),
            "unsafe_expectation": bool(unsafe_expectation),
        },
        "ops_score": ops_score,
        "operator_mode": operator_mode,
        "overload_risk": overload_risk,
        "founder_verdict": founder_verdict,
        "blockers": blockers,
        "warnings": warnings,
        "outstanding_customer_inputs": outstanding_customer_inputs,
        "do_now_queue": do_now_queue,
        "task_queue": task_queue,
        "weekly_plan": weekly_plan,
        "automation_opportunities": automation_opportunities,
        "stop_doing_list": stop_doing_list,
        "customer_followup_email": customer_followup_email,
        "internal_operator_note": internal_operator_note,
        "connected_snapshots": {
            "lead_intake_v48": bool(lead_intake_snapshot),
            "pricing_offer": bool(pricing_offer_snapshot),
            "proposal_sow": bool(proposal_sow_snapshot),
            "quote_to_cash": bool(quote_to_cash_snapshot),
            "paid_pilot": bool(paid_pilot_snapshot),
            "delivery": bool(delivery_snapshot),
            "customer_success": bool(customer_success_snapshot),
            "observability": bool(observability_snapshot),
        },
        "disclaimer": "V49 is an operational planning aid for founder workload and follow-up. It is not legal, financial, tax or project-management advice.",
    }


def create_founder_ops_followup_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="EdgeTwin Founder Ops Plan", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, txt=clean_pdf_text(f"Project: {project_name}"), ln=True, align="C")
    pdf.cell(0, 8, txt=clean_pdf_text(snapshot.get("version", "V49")), ln=True, align="C")
    pdf.ln(6)

    safe_pdf_cell(pdf, "Founder Ops Result", 8, True)
    safe_pdf_cell(pdf, f"Ops score: {snapshot.get('ops_score', 0)}%")
    safe_pdf_cell(pdf, f"Operator mode: {snapshot.get('operator_mode', 'Unknown')}")
    safe_pdf_cell(pdf, f"Overload risk: {snapshot.get('overload_risk', 'Unknown')}")
    safe_pdf_multicell(pdf, f"Verdict: {snapshot.get('founder_verdict', '')}")
    pdf.ln(4)

    customer = snapshot.get("customer", {})
    safe_pdf_cell(pdf, "Customer", 8, True)
    safe_pdf_cell(pdf, f"Name: {customer.get('name', '')}")
    safe_pdf_cell(pdf, f"Organization: {customer.get('organization', '')}")
    safe_pdf_cell(pdf, f"Email: {customer.get('email', '')}")
    safe_pdf_cell(pdf, f"Current stage: {snapshot.get('current_stage', '')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Do Now Queue", 8, True)
    for item in snapshot.get("do_now_queue", []):
        safe_pdf_multicell(pdf, f"- {item.get('priority', '')}: {item.get('task', '')} ({item.get('time_box', '')})")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Risks", 8, True)
    if snapshot.get("blockers"):
        for item in snapshot.get("blockers", []):
            safe_pdf_multicell(pdf, f"[BLOCKER] {item}")
    else:
        safe_pdf_multicell(pdf, "No hard founder-ops blockers detected.")
    for item in snapshot.get("warnings", []):
        safe_pdf_multicell(pdf, f"[WARNING] {item}")
    pdf.ln(4)

    safe_pdf_cell(pdf, "Stop Doing", 8, True)
    for item in snapshot.get("stop_doing_list", []):
        safe_pdf_multicell(pdf, f"- {item}")
    pdf.ln(4)
    safe_pdf_multicell(pdf, snapshot.get("disclaimer", ""))

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("founder_ops_plan.pdf", safe_pdf_output(pdf))
        zf.writestr("founder_ops_snapshot.json", json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr("task_queue.csv", pd.DataFrame(snapshot.get("task_queue", [])).to_csv(index=False))
        zf.writestr("do_now_queue.csv", pd.DataFrame(snapshot.get("do_now_queue", [])).to_csv(index=False))
        zf.writestr("weekly_plan.csv", pd.DataFrame(snapshot.get("weekly_plan", [])).to_csv(index=False))
        zf.writestr("automation_opportunities.csv", pd.DataFrame({"automation": snapshot.get("automation_opportunities", [])}).to_csv(index=False))
        zf.writestr("stop_doing_list.csv", pd.DataFrame({"stop_doing": snapshot.get("stop_doing_list", [])}).to_csv(index=False))
        zf.writestr("customer_followup_email.txt", snapshot.get("customer_followup_email", ""))
        zf.writestr("internal_operator_note.txt", snapshot.get("internal_operator_note", ""))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr("dataset_snapshot.csv", dataset_df.head(1000).to_csv(index=False))
        zf.writestr("README.txt", "EdgeTwin Studio V49 Founder Ops bundle. Use this to protect founder time, route customer follow-up and avoid unpaid custom chaos. Not legal/financial/tax advice.\n")
    return zip_buf.getvalue()



# ============================================================
# V50 CUSTOMER MODE / SIMPLIFIED CUSTOMER ROUTE
# ============================================================

def _v50_has_rows(dataset_df):
    return isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0


def _v50_snapshot_score(snapshot, key='overall_score'):
    if not isinstance(snapshot, dict):
        return 0
    for candidate in [key, 'customer_clarity_score', 'overall_score', 'readiness_score', 'trust_score', 'pilot_readiness_score', 'pricing_score', 'proposal_score', 'delivery_score', 'reliability_score']:
        val = snapshot.get(candidate)
        if isinstance(val, (int, float)):
            return max(0, min(100, int(round(float(val)))))
    return 0


def build_customer_mode_v50_summary(
    project_name,
    dataset_df=None,
    auto_pilot_result=None,
    fusion_doctor=None,
    trust_gate=None,
    pricing_offer_snapshot=None,
    proposal_sow_snapshot=None,
    paid_pilot_snapshot=None,
    delivery_snapshot=None,
    lead_intake_snapshot=None,
):
    """Create a customer-safe route summary for the V50 simplified UI.

    V50 is intentionally not a new technical engine. It is a UX/product gate that
    hides advanced/admin complexity from buyers while keeping Founder Mode intact.
    """
    dataset_rows = int(len(dataset_df)) if _v50_has_rows(dataset_df) else 0
    has_dataset = dataset_rows > 0
    has_pilot = bool(auto_pilot_result) or has_dataset
    has_doctor = isinstance(fusion_doctor, dict) and bool(fusion_doctor)
    has_trust = isinstance(trust_gate, dict) and bool(trust_gate)
    has_lead = isinstance(lead_intake_snapshot, dict) and bool(lead_intake_snapshot)
    has_price = isinstance(pricing_offer_snapshot, dict) and bool(pricing_offer_snapshot)
    has_sow = isinstance(proposal_sow_snapshot, dict) and bool(proposal_sow_snapshot)
    has_paid_pilot = isinstance(paid_pilot_snapshot, dict) and bool(paid_pilot_snapshot)
    has_delivery = isinstance(delivery_snapshot, dict) and bool(delivery_snapshot)

    doctor_score = _v50_snapshot_score(fusion_doctor) if has_doctor else 0
    trust_score = _v50_snapshot_score(trust_gate, 'trust_score') if has_trust else 0
    proposal_score = _v50_snapshot_score(proposal_sow_snapshot, 'proposal_score') if has_sow else 0
    pricing_score = _v50_snapshot_score(pricing_offer_snapshot, 'pricing_score') if has_price else 0
    delivery_score = _v50_snapshot_score(delivery_snapshot, 'delivery_score') if has_delivery else 0

    # Weighted for customer clarity, not model accuracy.
    score = 15
    if has_lead:
        score += 10
    if has_pilot:
        score += 22
    if has_dataset:
        score += 8
    if has_doctor:
        score += min(15, round(doctor_score * 0.15))
    if has_trust:
        score += min(15, round(trust_score * 0.15))
    if has_price:
        score += 8
    if has_sow:
        score += 12
    if has_paid_pilot:
        score += 5
    if has_delivery:
        score += 10
    score = max(0, min(100, int(score)))

    route_defs = [
        ('1', 'Choose problem', 'Select the customer use-case and describe the expected outcome.', has_lead or bool(auto_pilot_result)),
        ('2', 'Generate pilot', 'Create a pilot dataset, features, hardware direction and first export package.', has_pilot),
        ('3', 'Review trust', 'Check reliability, limitations, safe claims and field-validation needs.', has_doctor or has_trust),
        ('4', 'Download / handoff', 'Prepare the customer-facing report, delivery bundle or export package.', has_delivery),
        ('5', 'Request proposal', 'Move to pricing and Proposal/SOW only when scope and expectations are clear.', has_price or has_sow or has_paid_pilot),
    ]
    first_pending_seen = False
    customer_route = []
    for num, title, desc, done in route_defs:
        if done:
            status = 'Done'
        elif not first_pending_seen:
            status = 'Next'
            first_pending_seen = True
        else:
            status = 'Pending'
        customer_route.append({'step': num, 'title': title, 'description': desc, 'status': status})

    next_steps = [step for step in customer_route if step['status'] == 'Next']
    current_step = next_steps[0]['title'] if next_steps else 'Ready for controlled customer handoff'
    next_action_short = current_step

    ready_items = []
    missing_items = []
    if has_pilot:
        ready_items.append('Pilot package/data foundation exists.')
    else:
        missing_items.append('Create a pilot package through the Use Case Wizard.')
    if has_doctor or has_trust:
        ready_items.append('Readiness/trust signals are available for customer review.')
    else:
        missing_items.append('Run a readiness/trust check before customer delivery.')
    if has_price:
        ready_items.append('Pricing offer exists or can be reviewed.')
    else:
        missing_items.append('Create pricing only after customer scope and data status are clear.')
    if has_sow:
        ready_items.append('Proposal/SOW exists with scope and exclusions.')
    else:
        missing_items.append('Create Proposal/SOW before paid pilot commitment.')
    if has_delivery:
        ready_items.append('Delivery portal/handoff snapshot exists.')

    if score >= 85:
        customer_status = 'Clear route'
        risk_level = 'Low'
        verdict = 'Customer Mode is clear enough for a controlled paid-pilot conversation.'
    elif score >= 60:
        customer_status = 'Almost clear'
        risk_level = 'Medium'
        verdict = 'Customer Mode is usable, but one or two customer-facing gaps should be closed before delivery.'
    else:
        customer_status = 'Needs setup'
        risk_level = 'High'
        verdict = 'Start with the guided pilot route before showing pricing, proposal or delivery outputs.'

    visible_outputs = [
        'Problem/use-case explanation',
        'Pilot dataset and feature summary',
        'Readiness and reliability summary',
        'Safe limitations and field-validation message',
        'Report/proposal/delivery outputs when ready',
    ]
    hidden_tools = [
        'Storage internals and database scale tools',
        'Security/admin/error observability screens',
        'Raw signal sliders and engineering tuning',
        'Founder workload and internal follow-up notes',
        'Payment/invoice gates unless Founder Mode is active',
    ]

    safe_summary = f"""Project: {project_name}
Customer status: {customer_status}
Current step: {current_step}

EdgeTwin Studio can prepare a pilot-ready sensor AI package with dataset generation, feature extraction, readiness checks, hardware direction and customer deliverables.

Important limitation: this is a pilot preparation workflow. Production readiness, guaranteed accuracy, uptime, safety certification and regulatory compliance require separate field validation and are not promised by default.
"""

    return {
        'version': 'V50 Customer Mode + Founder Mode',
        'created_at': _now(),
        'project_name': project_name,
        'customer_clarity_score': score,
        'customer_status': customer_status,
        'risk_level': risk_level,
        'customer_verdict': verdict,
        'dataset_rows': dataset_rows,
        'current_step': current_step,
        'next_action_short': next_action_short,
        'customer_route': customer_route,
        'ready_items': ready_items,
        'missing_items': missing_items,
        'customer_visible_outputs': visible_outputs,
        'hidden_advanced_tools': hidden_tools,
        'connected_snapshots': {
            'lead_intake_v48': has_lead,
            'auto_pilot_or_dataset': has_pilot,
            'fusion_doctor': has_doctor,
            'trust_gate': has_trust,
            'pricing_offer': has_price,
            'proposal_sow': has_sow,
            'paid_pilot': has_paid_pilot,
            'delivery': has_delivery,
        },
        'scores': {
            'doctor_score': doctor_score,
            'trust_score': trust_score,
            'pricing_score': pricing_score,
            'proposal_score': proposal_score,
            'delivery_score': delivery_score,
        },
        'safe_customer_summary': safe_summary,
        'disclaimer': 'V50 is a customer-facing UX simplification and workflow guide. It is not a guarantee of model accuracy, production readiness, legal compliance, safety certification or payment readiness.',
    }


def create_customer_mode_v50_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 10, txt='EdgeTwin Customer Route V50', ln=True, align='C')
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 8, txt=clean_pdf_text(f'Project: {project_name}'), ln=True, align='C')
    pdf.cell(0, 8, txt=clean_pdf_text(snapshot.get('version', 'V50')), ln=True, align='C')
    pdf.ln(6)

    safe_pdf_cell(pdf, 'Customer Route Result', 8, True)
    safe_pdf_cell(pdf, f"Customer clarity: {snapshot.get('customer_clarity_score', 0)}%")
    safe_pdf_cell(pdf, f"Status: {snapshot.get('customer_status', 'Unknown')}")
    safe_pdf_cell(pdf, f"Risk level: {snapshot.get('risk_level', 'Unknown')}")
    safe_pdf_multicell(pdf, f"Verdict: {snapshot.get('customer_verdict', '')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, '5-Step Route', 8, True)
    for step in snapshot.get('customer_route', []):
        safe_pdf_multicell(pdf, f"{step.get('step', '')}. {step.get('title', '')} [{step.get('status', '')}] - {step.get('description', '')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, 'Still Needed', 8, True)
    for item in snapshot.get('missing_items', []):
        safe_pdf_multicell(pdf, f'- {item}')
    if not snapshot.get('missing_items'):
        safe_pdf_multicell(pdf, 'No major customer-facing gaps detected.')
    pdf.ln(4)

    safe_pdf_cell(pdf, 'Safe Customer Summary', 8, True)
    safe_pdf_multicell(pdf, snapshot.get('safe_customer_summary', ''))
    pdf.ln(3)
    safe_pdf_multicell(pdf, snapshot.get('disclaimer', ''))

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'a', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('customer_route_v50.pdf', safe_pdf_output(pdf))
        zf.writestr('customer_mode_snapshot.json', json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr('customer_route.csv', pd.DataFrame(snapshot.get('customer_route', [])).to_csv(index=False))
        zf.writestr('ready_items.csv', pd.DataFrame({'ready_item': snapshot.get('ready_items', [])}).to_csv(index=False))
        zf.writestr('missing_items.csv', pd.DataFrame({'missing_item': snapshot.get('missing_items', [])}).to_csv(index=False))
        zf.writestr('safe_customer_summary.txt', snapshot.get('safe_customer_summary', ''))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr('dataset_preview.csv', dataset_df.head(1000).to_csv(index=False))
        zf.writestr('README.txt', 'EdgeTwin Studio V50 Customer Route bundle. Use Customer Mode for clean demos and buyer-facing pilot flow. Founder Mode keeps advanced internal operations hidden.\n')
    return zip_buf.getvalue()

# ============================================================
# V51 — Customer Simplicity & UI Polish
# ============================================================

def build_customer_ui_v51_summary(
    project_name,
    dataset_df=None,
    auto_pilot_result=None,
    fusion_doctor=None,
    trust_gate=None,
    pricing_offer_snapshot=None,
    proposal_sow_snapshot=None,
    paid_pilot_snapshot=None,
    delivery_snapshot=None,
    lead_intake_snapshot=None,
):
    """Build a calmer customer-facing UX summary without weakening the engine.

    V51 is deliberately a presentation/workflow layer. It keeps V50's customer route
    and all existing EdgeTwin checks intact, but turns them into simpler status badges,
    three visible cards and one primary next step.
    """
    base = build_customer_mode_v50_summary(
        project_name=project_name,
        dataset_df=dataset_df,
        auto_pilot_result=auto_pilot_result,
        fusion_doctor=fusion_doctor,
        trust_gate=trust_gate,
        pricing_offer_snapshot=pricing_offer_snapshot,
        proposal_sow_snapshot=proposal_sow_snapshot,
        paid_pilot_snapshot=paid_pilot_snapshot,
        delivery_snapshot=delivery_snapshot,
        lead_intake_snapshot=lead_intake_snapshot,
    )

    score = int(base.get('customer_clarity_score', 0))
    dataset_rows = int(base.get('dataset_rows', 0))
    has_dataset = dataset_rows > 0
    has_trust = bool(base.get('connected_snapshots', {}).get('trust_gate')) or bool(base.get('connected_snapshots', {}).get('fusion_doctor'))
    has_proposal = bool(base.get('connected_snapshots', {}).get('proposal_sow'))
    has_price = bool(base.get('connected_snapshots', {}).get('pricing_offer'))
    has_delivery = bool(base.get('connected_snapshots', {}).get('delivery'))

    badges = []
    if score >= 85:
        badges.append('Pilot conversation ready')
    elif score >= 60:
        badges.append('Almost pilot-ready')
    else:
        badges.append('Setup needed')
    badges.append('Pilot only — field validation required')
    if has_dataset:
        badges.append('Dataset generated')
    else:
        badges.append('Needs pilot dataset')
    if has_trust:
        badges.append('Readiness checked')
    else:
        badges.append('Needs readiness check')
    if has_price or has_proposal:
        badges.append('Proposal path available')
    else:
        badges.append('Scope before price')

    if not has_dataset:
        primary_short = 'Create pilot'
        primary_long = 'Start with Create pilot: choose the use-case, sensors, labels and data status, then generate the pilot package.'
    elif not has_trust:
        primary_short = 'Review readiness'
        primary_long = 'Run the readiness/trust review before sending outputs to a customer. Keep the field-validation limitation visible.'
    elif not (has_price or has_proposal):
        primary_short = 'Prepare proposal'
        primary_long = 'The pilot foundation is ready enough for a controlled discussion. Qualify the lead, then create pricing and Proposal/SOW.'
    elif not has_delivery:
        primary_short = 'Prepare handoff'
        primary_long = 'Pricing/proposal exists. Prepare the customer delivery bundle and make sure scope, exclusions and acceptance criteria are clear.'
    else:
        primary_short = 'Controlled handoff'
        primary_long = 'The customer route is clear. Use this for a controlled paid-pilot handoff, not as a production guarantee.'

    customer_cards = [
        {
            'title': 'Choose problem',
            'body': 'Customer explains the machine, environment, sensor idea and detection goal.',
            'visible': 'Plain-language intake',
            'hidden_engine': 'Use-case defaults, sensor assumptions, class generation rules',
        },
        {
            'title': 'Generate pilot',
            'body': 'EdgeTwin creates the pilot dataset, features, reliability signals and first export outputs.',
            'visible': 'One guided pilot action',
            'hidden_engine': 'Signal generation, feature extraction, Dataset Doctor, hardware direction',
        },
        {
            'title': 'Review & handoff',
            'body': 'Customer sees readiness, limitations, downloads and proposal route without admin overload.',
            'visible': 'Readiness + next step',
            'hidden_engine': 'Trust gates, safe claims, pricing, SOW, delivery and founder controls',
        },
    ]

    engine_kept_active = [
        'Dataset Doctor and class-balance checks',
        'Reliability/Trust scoring and safe-claim guidance',
        'Synthetic-to-real and normality workflow when data exists',
        'Hardware/deployment planner and Edge Impulse exports',
        'Pricing, Proposal/SOW, delivery and founder-ops gates in Founder Mode',
    ]
    ui_rules = [
        'Show one primary next step instead of many technical choices.',
        'Keep risky claims visible: pilot-ready is not production-certified.',
        'Move engineering details into Advanced details.',
        'Do not remove engine checks; hide them until they help the customer decision.',
        'Use Founder Mode for pricing, payment gates, storage, security and admin work.',
    ]

    safe_summary = f"""Project: {project_name}
Customer status: {base.get('customer_status', 'Unknown')}
Readiness: {score}%
Primary next step: {primary_short}

EdgeTwin Studio prepares a customer-safe pilot route: choose the problem, generate the pilot package, review readiness, then move to handoff or proposal only when scope and limitations are clear.

The simplified V51 interface does not reduce the technical engine. Dataset generation, feature extraction, reliability checks, trust gates, hardware direction and delivery controls remain active behind the clean customer view.

Important limitation: this is a pilot preparation workflow. Production deployment, guaranteed accuracy, uptime, safety certification and regulatory compliance require separate field validation and are not promised by default.
"""

    result = dict(base)
    result.update({
        'version': 'V51 Customer Simplicity & UI Polish',
        'ui_layer': 'Customer-facing simplified route with Founder Mode preserved',
        'status_badges': badges,
        'primary_next_step': primary_short,
        'primary_next_step_long': primary_long,
        'customer_cards': customer_cards,
        'engine_kept_active': engine_kept_active,
        'ui_simplification_rules': ui_rules,
        'safe_customer_summary': safe_summary,
        'base_v50_snapshot': base,
        'disclaimer': 'V51 simplifies the customer interface only. It does not guarantee model accuracy, production readiness, legal compliance, safety certification, payment readiness or field performance.',
    })
    return result


def create_customer_ui_v51_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 10, txt='EdgeTwin Customer UI V51', ln=True, align='C')
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 8, txt=clean_pdf_text(f'Project: {project_name}'), ln=True, align='C')
    pdf.cell(0, 8, txt=clean_pdf_text(snapshot.get('version', 'V51')), ln=True, align='C')
    pdf.ln(6)

    safe_pdf_cell(pdf, 'Customer View Result', 8, True)
    safe_pdf_cell(pdf, f"Readiness: {snapshot.get('customer_clarity_score', 0)}%")
    safe_pdf_cell(pdf, f"Status: {snapshot.get('customer_status', 'Unknown')}")
    safe_pdf_cell(pdf, f"Risk level: {snapshot.get('risk_level', 'Unknown')}")
    safe_pdf_multicell(pdf, f"Primary next step: {snapshot.get('primary_next_step_long', '')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, 'Status Badges', 8, True)
    safe_pdf_multicell(pdf, ', '.join(snapshot.get('status_badges', [])))
    pdf.ln(4)

    safe_pdf_cell(pdf, 'Simple Customer Cards', 8, True)
    for card in snapshot.get('customer_cards', []):
        safe_pdf_multicell(pdf, f"{card.get('title', '')}: {card.get('body', '')}")
        safe_pdf_multicell(pdf, f"Visible: {card.get('visible', '')}")
        safe_pdf_multicell(pdf, f"Hidden engine: {card.get('hidden_engine', '')}")
        pdf.ln(2)

    safe_pdf_cell(pdf, 'Engine Kept Active', 8, True)
    for item in snapshot.get('engine_kept_active', []):
        safe_pdf_multicell(pdf, f'- {item}')
    pdf.ln(4)

    safe_pdf_cell(pdf, 'Safe Customer Summary', 8, True)
    safe_pdf_multicell(pdf, snapshot.get('safe_customer_summary', ''))
    pdf.ln(3)
    safe_pdf_multicell(pdf, snapshot.get('disclaimer', ''))

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'a', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('customer_ui_v51.pdf', safe_pdf_output(pdf))
        zf.writestr('customer_ui_v51_snapshot.json', json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr('customer_cards.csv', pd.DataFrame(snapshot.get('customer_cards', [])).to_csv(index=False))
        zf.writestr('status_badges.csv', pd.DataFrame({'badge': snapshot.get('status_badges', [])}).to_csv(index=False))
        zf.writestr('engine_kept_active.csv', pd.DataFrame({'engine_component': snapshot.get('engine_kept_active', [])}).to_csv(index=False))
        zf.writestr('ui_simplification_rules.csv', pd.DataFrame({'rule': snapshot.get('ui_simplification_rules', [])}).to_csv(index=False))
        zf.writestr('safe_customer_summary.txt', snapshot.get('safe_customer_summary', ''))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            zf.writestr('dataset_preview.csv', dataset_df.head(1000).to_csv(index=False))
        zf.writestr('README.txt', 'EdgeTwin Studio V51 Customer UI bundle. V51 simplifies the customer-facing route while preserving the full EdgeTwin/OMEGA-X engine behind it.\n')
    return zip_buf.getvalue()

# ============================================================
# V52 — Privacy-Safe Field Learning System
# ============================================================

def _v52_contains_any(columns, patterns):
    cols = [str(c).lower() for c in columns]
    return any(any(p in c for p in patterns) for c in cols)


def _v52_safe_numeric_feature_summary(dataset_df, max_features=40):
    """Return aggregate numeric feature summaries only; never raw rows."""
    if not isinstance(dataset_df, pd.DataFrame) or len(dataset_df) == 0:
        return []
    rows = []
    numeric_cols = [c for c in dataset_df.columns if pd.api.types.is_numeric_dtype(dataset_df[c])]
    sensitive_patterns = ['gps', 'lat', 'lon', 'lng', 'location', 'address', 'name', 'email', 'phone', 'customer', 'operator', 'employee', 'user', 'id']
    for col in numeric_cols[:max_features]:
        low = str(col).lower()
        sensitivity_flag = 'potential_identifier' if any(p in low for p in sensitive_patterns) else 'feature_candidate'
        series = pd.to_numeric(dataset_df[col], errors='coerce').dropna()
        if len(series) == 0:
            continue
        rows.append({
            'feature': str(col),
            'count': int(series.count()),
            'mean': float(series.mean()),
            'std': float(series.std()) if len(series) > 1 else 0.0,
            'min': float(series.min()),
            'max': float(series.max()),
            'safe_status': sensitivity_flag,
            'learning_use': 'blocked_until_review' if sensitivity_flag != 'feature_candidate' else 'aggregate_only',
        })
    return rows


def build_privacy_safe_field_learning_v52(
    project_name,
    dataset_df=None,
    consent_mode='Private only',
    customer_sector='Custom',
    use_case_type='Custom pilot',
    retention_days=180,
    contains_audio=False,
    contains_gps=False,
    contains_machine_ids=False,
    contains_personal_data=False,
    allow_cross_customer_learning=False,
):
    """Build a privacy-first plan for learning from real customer field data.

    The default is private project use only. Global/cross-customer learning is only
    recommended for aggregate feature statistics when there is explicit opt-in.
    This function intentionally avoids storing raw rows in the returned learning library.
    """
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    columns = list(dataset_df.columns)
    rows = int(len(dataset_df))
    cols = int(len(columns))
    consent_mode = consent_mode or 'Private only'

    detected_audio = bool(contains_audio or _v52_contains_any(columns, ['audio', 'wav', 'filename', 'file_name', 'raw_signal']))
    detected_gps = bool(contains_gps or _v52_contains_any(columns, ['gps', 'lat', 'lon', 'lng', 'location', 'address']))
    detected_ids = bool(contains_machine_ids or _v52_contains_any(columns, ['machine_id', 'device_id', 'customer_id', 'project_id', 'serial', 'asset_id']))
    detected_personal = bool(contains_personal_data or _v52_contains_any(columns, ['name', 'email', 'phone', 'employee', 'operator', 'user']))

    risk_points = 0
    if consent_mode == 'Raw data permission':
        risk_points += 30
    elif consent_mode == 'Feature learning allowed':
        risk_points += 10
    if detected_audio:
        risk_points += 18
    if detected_gps:
        risk_points += 20
    if detected_ids:
        risk_points += 18
    if detected_personal:
        risk_points += 30
    if retention_days > 365:
        risk_points += 10
    if allow_cross_customer_learning and consent_mode != 'Feature learning allowed':
        risk_points += 15

    privacy_score = max(0, min(100, 100 - risk_points))
    if privacy_score >= 78:
        risk_level = 'Low'
    elif privacy_score >= 55:
        risk_level = 'Medium'
    else:
        risk_level = 'High'

    if consent_mode == 'Private only':
        learning_mode = 'Project-private learning only'
    elif consent_mode == 'Feature learning allowed' and allow_cross_customer_learning:
        learning_mode = 'Opt-in aggregate feature learning'
    elif consent_mode == 'Feature learning allowed':
        learning_mode = 'Feature extraction allowed, global learning disabled'
    else:
        learning_mode = 'Raw-data permission requested — restricted review required'

    safe_features = _v52_safe_numeric_feature_summary(dataset_df)
    usable_features = [f for f in safe_features if f.get('safe_status') == 'feature_candidate']

    allowed_actions = []
    blocked_actions = []
    if consent_mode == 'Private only':
        allowed_actions += [
            'Use data only inside this customer project.',
            'Generate project-specific features, readiness checks and reports.',
            'Delete/export project data on customer request.',
        ]
        blocked_actions += [
            'Do not use this customer data for global templates.',
            'Do not copy raw files into cross-customer training libraries.',
            'Do not publish or reuse customer-specific patterns outside the project.',
        ]
    elif consent_mode == 'Feature learning allowed':
        allowed_actions += [
            'Extract aggregate numeric features for privacy-safe learning.',
            'Use de-identified feature statistics to improve templates and thresholds.',
            'Keep raw files/private identifiers out of the global learning library.',
        ]
        if allow_cross_customer_learning:
            allowed_actions.append('Use aggregate feature patterns across customers after minimization checks.')
        else:
            blocked_actions.append('Cross-customer learning remains disabled until the opt-in box is selected.')
        blocked_actions += [
            'Do not store raw audio/GPS/customer files in global training by default.',
            'Do not include customer names, machine IDs, exact location or operator data in learning exports.',
        ]
    else:
        allowed_actions += [
            'Use raw data only for the explicitly scoped project purpose.',
            'Run restricted review before any raw-data transfer or long retention.',
        ]
        blocked_actions += [
            'Do not treat raw permission as automatic global training permission.',
            'Do not keep raw data longer than the agreed retention period.',
            'Do not use raw audio/location/person data without written scope and deletion terms.',
        ]

    if detected_audio:
        blocked_actions.append('Raw audio is blocked from global learning unless there is a separate written permission and review.')
    if detected_gps:
        blocked_actions.append('Exact GPS/location data must be generalized or removed before feature learning.')
    if detected_ids or detected_personal:
        blocked_actions.append('Customer/device/person identifiers must be removed before any cross-customer learning.')

    data_minimization_plan = [
        'Prefer feature-only learning: RMS, kurtosis, crest factor, band energy, spectral shape and class-level statistics.',
        'Remove customer/project names, exact locations, serial numbers, device IDs and operator/person identifiers.',
        'Store only aggregate feature summaries in the global learning library.',
        'Keep raw files inside the private project vault unless a separate written permission exists.',
        'Use small dataset previews for reports; never export complete raw customer data inside global bundles.',
    ]

    retention_and_deletion_plan = [
        f'Default retention for this project setting: {int(retention_days)} days.',
        'Customer can request export/delete of project data.',
        'Learning library should retain aggregate feature summaries, not raw customer files.',
        'If consent is withdrawn, stop future learning use and remove project-linked metadata where possible.',
    ]

    audit_log_events = [
        'Record data permission mode selected.',
        'Record feature extraction date and project ID.',
        'Record whether cross-customer aggregate learning was enabled.',
        'Record generated bundles/reports and deletion/export requests.',
    ]

    if risk_level == 'Low' and consent_mode == 'Feature learning allowed' and usable_features:
        verdict = 'Good candidate for opt-in aggregate feature learning. Keep raw data private and use only minimized feature statistics.'
    elif consent_mode == 'Private only':
        verdict = 'Safest mode: data stays private to this customer project. EdgeTwin can still improve this customer pilot without global reuse.'
    elif risk_level == 'High':
        verdict = 'High privacy risk. Keep this data private until identifiers, raw audio/location/person risks and consent terms are solved.'
    else:
        verdict = 'Use cautiously: feature-only learning may be possible, but identifiers/raw data risks must be minimized first.'

    learning_summary = f"""V52 learning mode: {learning_mode}
Project: {project_name}
Sector/use-case: {customer_sector} / {use_case_type}
Dataset: {rows} rows, {cols} columns
Usable aggregate feature candidates: {len(usable_features)}

Recommended approach: keep customer data private by default. If the customer explicitly opts in, use only de-identified aggregate feature statistics to improve EdgeTwin templates, thresholds and realism. Raw files, exact locations, customer identities and personal data should not enter the global learning engine by default.
"""

    consent_clause = f"""Optional anonymised feature learning clause for Proposal/SOW:

By default, Customer project data is used only for the Customer's own pilot project. Customer may optionally allow EdgeTwin Studio to use de-identified and aggregated feature statistics derived from the project, such as signal feature distributions and class-level summary metrics, to improve templates, reliability scoring and pilot-generation quality.

This optional learning does not grant permission to publish, sell, or reuse raw customer files, exact locations, customer names, machine identifiers, personal data, raw audio, images, or confidential operational information outside the agreed project scope. Customer may request export or deletion according to the agreed retention terms. Production deployment and compliance validation remain separate from this pilot workflow.
"""

    return {
        'version': 'V52 Privacy-Safe Field Learning System',
        'project_name': project_name,
        'customer_sector': customer_sector,
        'use_case_type': use_case_type,
        'consent_mode': consent_mode,
        'allow_cross_customer_learning': bool(allow_cross_customer_learning),
        'learning_mode': learning_mode,
        'privacy_score': int(privacy_score),
        'risk_level': risk_level,
        'verdict': verdict,
        'dataset_rows': rows,
        'dataset_cols': cols,
        'feature_rows_available': len(usable_features),
        'detected_risks': {
            'raw_audio_or_files': detected_audio,
            'gps_or_location': detected_gps,
            'machine_or_customer_ids': detected_ids,
            'personal_data': detected_personal,
        },
        'safe_feature_candidates': usable_features,
        'feature_library_preview': usable_features[:20],
        'allowed_actions': allowed_actions,
        'blocked_actions': blocked_actions,
        'data_minimization_plan': data_minimization_plan,
        'retention_and_deletion_plan': retention_and_deletion_plan,
        'audit_log_events': audit_log_events,
        'learning_summary': learning_summary,
        'customer_consent_clause': consent_clause,
        'recommended_next_step': 'Use Feature learning allowed only when the customer explicitly opts in; otherwise keep data project-private.',
        'disclaimer': 'V52 is not legal advice and does not automatically anonymise data. Treat raw audio, location, identifiers and personal data as restricted unless separately reviewed and agreed.',
    }


def create_privacy_safe_field_learning_v52_bundle(project_name, snapshot, dataset_df=None):
    snapshot = snapshot or {}
    dataset_df = dataset_df if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 10, txt='EdgeTwin Privacy-Safe Field Learning V52', ln=True, align='C')
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 8, txt=clean_pdf_text(f'Project: {project_name}'), ln=True, align='C')
    pdf.cell(0, 8, txt=clean_pdf_text(snapshot.get('version', 'V52')), ln=True, align='C')
    pdf.ln(6)

    safe_pdf_cell(pdf, 'Privacy Learning Result', 8, True)
    safe_pdf_cell(pdf, f"Privacy score: {snapshot.get('privacy_score', 0)}%")
    safe_pdf_cell(pdf, f"Learning mode: {snapshot.get('learning_mode', 'Unknown')}")
    safe_pdf_cell(pdf, f"Risk level: {snapshot.get('risk_level', 'Unknown')}")
    safe_pdf_multicell(pdf, f"Verdict: {snapshot.get('verdict', '')}")
    pdf.ln(4)

    safe_pdf_cell(pdf, 'Allowed Actions', 8, True)
    for item in snapshot.get('allowed_actions', []):
        safe_pdf_multicell(pdf, f'- {item}')
    pdf.ln(3)

    safe_pdf_cell(pdf, 'Blocked By Default', 8, True)
    for item in snapshot.get('blocked_actions', []):
        safe_pdf_multicell(pdf, f'- {item}')
    pdf.ln(3)

    safe_pdf_cell(pdf, 'Data Minimization Plan', 8, True)
    for item in snapshot.get('data_minimization_plan', []):
        safe_pdf_multicell(pdf, f'- {item}')
    pdf.ln(3)

    safe_pdf_cell(pdf, 'Consent Clause', 8, True)
    safe_pdf_multicell(pdf, snapshot.get('customer_consent_clause', ''))
    pdf.ln(3)
    safe_pdf_multicell(pdf, snapshot.get('disclaimer', ''))

    safe_features_df = pd.DataFrame(snapshot.get('safe_feature_candidates', []))
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'a', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('privacy_safe_field_learning_v52.pdf', safe_pdf_output(pdf))
        zf.writestr('privacy_learning_snapshot.json', json.dumps(_json_safe(snapshot), indent=2, ensure_ascii=False))
        zf.writestr('safe_feature_library_preview.csv', safe_features_df.to_csv(index=False))
        zf.writestr('allowed_actions.csv', pd.DataFrame({'allowed_action': snapshot.get('allowed_actions', [])}).to_csv(index=False))
        zf.writestr('blocked_actions.csv', pd.DataFrame({'blocked_action': snapshot.get('blocked_actions', [])}).to_csv(index=False))
        zf.writestr('data_minimization_plan.csv', pd.DataFrame({'minimization_step': snapshot.get('data_minimization_plan', [])}).to_csv(index=False))
        zf.writestr('retention_and_deletion_plan.csv', pd.DataFrame({'retention_step': snapshot.get('retention_and_deletion_plan', [])}).to_csv(index=False))
        zf.writestr('customer_consent_clause.txt', snapshot.get('customer_consent_clause', ''))
        zf.writestr('learning_summary.txt', snapshot.get('learning_summary', ''))
        if isinstance(dataset_df, pd.DataFrame) and len(dataset_df) > 0:
            # Deliberately no raw rows here; schema-only export keeps bundle safe.
            schema = pd.DataFrame({'column': list(dataset_df.columns), 'dtype': [str(dataset_df[c].dtype) for c in dataset_df.columns]})
            zf.writestr('dataset_schema_only.csv', schema.to_csv(index=False))
        zf.writestr('README.txt', 'EdgeTwin Studio V52 Privacy-Safe Field Learning bundle. This bundle contains consent, minimization and aggregate feature-learning outputs. It intentionally does not include raw customer data rows.\n')
    return zip_buf.getvalue()

