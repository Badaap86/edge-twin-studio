# api.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import core

app = FastAPI(title="OMEGA-X Enterprise REST API")

class GenerateReq(BaseModel):
    base_f: float
    harm_r: float
    imp_r: float
    noise_l: float
    sr: int = 16000

class DeployReq(BaseModel):
    hardware: str
    num_features: int
    sr: int = 16000

@app.post("/api/generate")
def generate_dataset(req: GenerateReq):
    res = core.generate_universal_signal(2.0, req.sr, req.base_f, req.harm_r, req.imp_r, req.noise_l)
    return {"status": "success", "samples": len(res["sig"])}

@app.post("/api/extract")
async def extract_features(file: UploadFile = File(...)):
    # Simuleert snelle feature extractie van geüploade bestanden via de API
    return {"status": "success", "features": {"ZCR": 0.02, "SpectralCentroid": 1250, "RMS": 0.15}}

@app.post("/api/deployment")
def check_deployment(req: DeployReq):
    ram, l_fft, l_feat, l_inf = core.estimate_edge_load(req.hardware, req.num_features, req.sr)
    tot_lat = l_fft + l_feat + l_inf
    score = core.calculate_deployment_score(req.hardware, tot_lat, ram)
    return {"hardware": req.hardware, "deployment_ready_score": score, "latency_ms": tot_lat, "ram_kb": ram}
