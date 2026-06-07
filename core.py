# core.py
import numpy as np
import scipy.signal as signal

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
    return zcr, centroid, rolloff, flatness

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

def estimate_edge_load(hw, feat_n, sr, duration=1.0):
    fft_n = 1024 if sr <= 4000 else 2048
    ram = ((min(int(sr * duration), 8192) * 4) + (fft_n * 8) + 2048) / 1024 
    if "ESP32-S3" in hw: return ram, (fft_n * np.log2(fft_n)) * 0.00008, feat_n * 0.2, 1.5
    elif "STM32L4" in hw: return ram, (fft_n * np.log2(fft_n)) * 0.00012, feat_n * 0.3, 2.5
    elif "RAK4631" in hw: return ram, (fft_n * np.log2(fft_n)) * 0.00015, feat_n * 0.4, 3.0
    else: return ram, (fft_n * np.log2(fft_n)) * 0.0008, feat_n * 1.5, 12.0

def calculate_deployment_score(hw, latency, ram_kb):
    base = (max(0, 100 - (latency / 20.0) * 50) * 0.7) + (max(0, 100 - (ram_kb / 120.0) * 50) * 0.3)
    if "ESP32-S3" in hw: return min(100, base + 15)
    elif "STM32L4" in hw: return min(100, base + 10)
    elif "RAK4631" in hw: return min(100, base + 5)
    return max(0, base - 20)
