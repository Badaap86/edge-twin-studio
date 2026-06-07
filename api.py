from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
import pandas as pd
import core

app = FastAPI(title="OMEGA-X Headless Enterprise API")

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
    return {"status": "success", "samples": len(res["sig"]), "sr": req.sr}

@app.post("/api/extract")
async def extract_features(sr: int = 16000, file: UploadFile = File(...)):
    content = await file.read()
    features = core.extract_features_from_bytes(content, file.filename, sr)
    if "error" in features: return {"status": "error", "message": features["error"]}
    return {"status": "success", "features": features}

@app.post("/api/audit")
def audit_dataset(dataset: list[dict]):
    df = pd.DataFrame(dataset)
    if "Label" not in df.columns: return {"error": "Dataset must contain a 'Label' column"}
    X = df.drop(columns=["Label"]).replace([float('inf'), float('-inf')], 0).fillna(0)
    y = df["Label"]
    div, bal, sep = core.calculate_audit_scores(X, y)
    return {"diversity_score": div, "balance_score": bal, "separation_score": sep}

@app.post("/api/report")
def generate_report(dataset: list[dict], project_name: str = "Headless_Audit"):
    df = pd.DataFrame(dataset)
    X = df.drop(columns=["Label"]).replace([float('inf'), float('-inf')], 0).fillna(0)
    y = df["Label"]
    div, bal, sep = core.calculate_audit_scores(X, y)
    
    # Mock hardware data for headless report
    brds = ["ESP32-S3", "STM32L4", "RAK4631"]
    b_dat = [{"Board": b, "Score": 90, "Latency": 15.0} for b in brds] 
    
    pdf_bytes = core.generate_pdf_report(project_name, len(X), len(y.unique()), div, bal, sep, [], b_dat, "ESP32-S3")
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=audit_{project_name}.pdf"})

@app.post("/api/deployment")
def check_deployment(req: DeployReq):
    ram, l_fft, l_feat, l_inf = core.estimate_edge_load(req.hardware, req.num_features, req.sr)
    tot_lat = l_fft + l_feat + l_inf
    score = core.calculate_deployment_score(req.hardware, tot_lat, ram)
    return {"hardware": req.hardware, "readiness_score": score, "latency_ms": tot_lat, "ram_kb": ram}
