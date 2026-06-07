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

# ============================================================
# DATABASE ENGINE (SQLite SaaS Foundation)
# ============================================================
def init_db():
    conn = sqlite3.connect('omega_saas.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS projects
                 (id TEXT PRIMARY KEY, name TEXT, created_at TEXT, dataset TEXT, settings TEXT)''')
    conn.commit()
    conn.close()

def save_project(proj_id, name, dataset_df, settings_dict):
    conn = sqlite3.connect('omega_saas.db')
    c = conn.cursor()
    c.execute("REPLACE INTO projects (id, name, created_at, dataset, settings) VALUES (?, ?, ?, ?, ?)",
              (proj_id, name, str(datetime.datetime.now()), dataset_df.to_json(orient='records'), json.dumps(settings_dict)))
    conn.commit()
    conn.close()

def load_project(proj_id):
    conn = sqlite3.connect('omega_saas.db')
    c = conn.cursor()
    c.execute("SELECT name, dataset, settings FROM projects WHERE id=?", (proj_id,))
    row = c.fetchone()
    conn.close()
    if row: return {"name": row[0], "dataset": pd.read_json(io.StringIO(row[1])), "settings": json.loads(row[2])}
    return None

def get_all_projects():
    conn = sqlite3.connect('omega_saas.db')
    df = pd.read_sql_query("SELECT id, name, created_at FROM projects", conn)
    conn.close()
    return df

# ============================================================
# DSP ENGINE (HERSTELDE DEEP CLONER)
# ============================================================
def calculate_fft(sig, sr):
    window = np.hanning(len(sig))
    fft_v = np.abs(np.fft.rfft(sig * window))
    fft_f = np.fft.rfftfreq(len(sig), 1 / sr)
    return fft_f, fft_v

def get_audio_features(sig, f_f, v_f):
    eps = 1e-10 
    zcr = ((sig[:-1] * sig[1:]) < 0).sum() / len(sig)
    centroid = np.sum(f_f * v_f) / (np.sum(v_f) + eps)
    cum_e = np.cumsum(v_f)
    tot_e = cum_e[-1]
    rolloff = f_f[np.where(cum_e >= 0.85 * tot_e)[0][0]] if tot_e > 0 else 0
    flatness = np.exp(np.mean(np.log(v_f + eps))) / (np.mean(v_f) + eps)
    return float(zcr), float(centroid), float(rolloff), float(flatness)

def reverse_engineer_physics(sig, sr):
    f_f, v_f = calculate_fft(sig, sr)
    dom_idx = np.argmax(v_f[1:]) + 1
    ext_base = f_f[dom_idx]
    ext_harm = sum(v_f[np.argmin(np.abs(f_f - (h * ext_base)))] for h in range(2, 6)) / max(v_f[dom_idx], 1e-6)
    peaks, _ = signal.find_peaks(np.abs(sig), height=np.mean(np.abs(sig)) + 2.5 * np.std(sig), distance=sr/100)
    return {"base_f": ext_base, "harm_r": ext_harm, "imp_r": len(peaks) / (len(sig) / sr), "noise": np.median(np.abs(sig))}

def extract_features_from_bytes(file_bytes, filename, sr=16000):
    try:
        if filename.endswith('.csv'): sig = pd.read_csv(io.BytesIO(file_bytes)).iloc[:, 1].astype(float).values
        else:
            s_ext, w_d = wavfile.read(io.BytesIO(file_bytes))
            sr = s_ext
            sig = (w_d.mean(axis=1) if len(w_d.shape) > 1 else w_d).astype(float)
        sig = sig - np.mean(sig)
        f_f, v_f = calculate_fft(sig, sr)
        rms = float(np.sqrt(np.mean(sig**2)))
        zcr, cent, roll, flat = get_audio_features(sig, f_f, v_f)
        return {
            "RMS": rms, "Kurtosis": float(kurtosis(sig)), 
            "CrestFactor": float(np.max(np.abs(sig)) / max(rms, 1e-6)), 
            "ZCR": zcr, "SpectralCentroid": cent, 
            "SpectralRolloff": roll, "SpectralFlatness": flat
        }
    except Exception as e: return {"error": str(e)}

def generate_universal_signal(duration, sr, base_f, harm_r, imp_r, noise_l, normalize=True):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.sin(2 * np.pi * base_f * t) if base_f > 0 else np.zeros_like(t)
    if harm_r > 0 and base_f > 0:
        for h in range(2, 6): sig += (harm_r / h) * np.sin(2 * np.pi * (base_f * h) * t)
    if imp_r > 0:
        for i_t in np.arange(0, duration, 1.0 / imp_r):
            idx = int((i_t + np.random.uniform(-0.02, 0.02)) * sr)
            if 0 <= idx < len(t):
                decay = np.exp(-25 * (t[idx:] - i_t))
                sig[idx:] += 2.5 * np.random.normal(0, 1, len(decay)) * decay
    sig += np.random.normal(0, noise_l, len(t))
    if normalize and np.max(np.abs(sig)) > 0: sig /= np.max(np.abs(sig))
    f, v = calculate_fft(sig, sr)
    return {"t": t, "sig": sig, "fft_f": f, "fft_v": v, "sr": sr}

# ============================================================
# UNIVERSAL EDGE PROFILER (V14.5)
# ============================================================

# Database met hardware profielen
# Elk profiel heeft een baseline factor voor FFT, Feature Extraction, en Inference snelheid.
HARDWARE_PROFILES = {
    "Xtensa LX7 (ESP32-S3)": {"fft_mult": 0.00008, "feat_mult": 0.2, "inf_mult": 1.5, "target_ram": 320},
    "Xtensa LX6 (ESP32)":    {"fft_mult": 0.00010, "feat_mult": 0.25, "inf_mult": 2.0, "target_ram": 320},
    "ARM Cortex-M7 (STM32H7)": {"fft_mult": 0.00005, "feat_mult": 0.1, "inf_mult": 0.8, "target_ram": 512},
    "ARM Cortex-M4F (nRF52840 / RAK4631)": {"fft_mult": 0.00015, "feat_mult": 0.4, "inf_mult": 3.0, "target_ram": 256},
    "ARM Cortex-M3 (STM32F1)": {"fft_mult": 0.00030, "feat_mult": 0.8, "inf_mult": 6.0, "target_ram": 64},
    "ARM Cortex-M0+ (RP2040)": {"fft_mult": 0.00080, "feat_mult": 1.5, "inf_mult": 12.0, "target_ram": 264},
    "RISC-V (BL602)":        {"fft_mult": 0.00020, "feat_mult": 0.5, "inf_mult": 4.0, "target_ram": 276}
}

def get_available_hardware():
    return list(HARDWARE_PROFILES.keys())

def estimate_edge_load(hw_name, feat_n, sr, duration=1.0):
    """Berekent de belasting op basis van universele profielen."""
    # Fallback profiel als de naam niet wordt gevonden
    profile = HARDWARE_PROFILES.get(hw_name, HARDWARE_PROFILES["ARM Cortex-M4F (nRF52840 / RAK4631)"])
    
    fft_n = 1024 if sr <= 4000 else 2048
    ram = ((min(int(sr * duration), 8192) * 4) + (fft_n * 8) + 2048) / 1024 
    
    l_fft = (fft_n * np.log2(fft_n)) * profile["fft_mult"]
    l_feat = feat_n * profile["feat_mult"]
    l_inf = profile["inf_mult"]
    
    return ram, l_fft, l_feat, l_inf

def calculate_deployment_score(hw_name, latency, ram_kb):
    profile = HARDWARE_PROFILES.get(hw_name, HARDWARE_PROFILES["ARM Cortex-M4F (nRF52840 / RAK4631)"])
    target_ram = profile["target_ram"]
    
    # Doel latency is arbitrair gezet op 20ms voor een perfecte score.
    lat_score = max(0, 100 - (latency / 20.0) * 50)
    ram_score = max(0, 100 - (ram_kb / (target_ram / 2)) * 50) # Bij 50% RAM gebruik begint de penalty
    
    base = (lat_score * 0.7) + (ram_score * 0.3)
    
    # Kleine bonus voor efficiëntere architecturen
    if profile["fft_mult"] < 0.00010: base += 10 # Bijv. Cortex-M7 of LX7
    elif profile["fft_mult"] < 0.00020: base += 5  # Bijv. Cortex-M4F
    
    return min(100, max(0, base))

# ============================================================
# ML AUDIT & PDF ENGINE
# ============================================================
def calculate_audit_scores(X_df, y_series):
    v_c = y_series.value_counts()
    X_scaled = StandardScaler().fit_transform(X_df)
    div = min(100, int((np.mean(pdist(X_scaled)) / 4.0) * 100)) if len(X_scaled) > 1 else 0
    bal = 100 if len(v_c) >= 2 and (v_c.min() / v_c.max()) > 0.5 else 50
    sep = int((silhouette_score(X_scaled, y_series) + 1) * 50) if len(y_series.unique()) >= 2 and v_c.min() >= 2 else 0
    return div, bal, sep

def generate_pdf_report(proj_name, num_samples, num_classes, div, bal, sep, top_features, b_dat, best_board):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(200, 10, txt="OMEGA-X Enterprise Audit", ln=True, align="C")
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(200, 10, txt=f"Project ID: {proj_name}", ln=True, align="C")
    pdf.ln(10)
    
    overall_status = "PRODUCTION READY" if all(s > 80 for s in [div, bal, sep]) else "OPTIMIZATION REQUIRED"
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Overall Status: {overall_status}", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 8, txt=f"Total Samples: {num_samples} | Unique Classes: {num_classes}", ln=True)
    pdf.cell(200, 8, txt=f"Dataset Diversity: {div}% | Class Balance: {bal}% | Label Separation: {sep}%", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Top Features (Permutation Importance):", ln=True)
    pdf.set_font("Arial", '', 11)
    if top_features:
        for f, score in top_features[:5]: 
            pdf.cell(200, 7, txt=f"- {f}: {score:.1f}% impact", ln=True)
    else: 
        pdf.cell(200, 7, txt="Not enough data to calculate feature importance.", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Hardware Recommendation: {best_board}", ln=True)
    pdf.set_font("Arial", '', 11)
    for d in b_dat: 
        pdf.cell(200, 7, txt=f"- {d['Board']}: Score {d['Score']:.0f}% (Lat: {d['Latency']:.1f}ms)", ln=True)
        
    return pdf.output(dest='S').encode('latin1')
