from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import Response
from pydantic import BaseModel
import pandas as pd
import core

# --- ECHTE SAAS SECURITY LAYER ---
api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)

async def get_current_user(api_key: str = Security(api_key_header)):
    user = core.verify_api_key(api_key)
    if not user:
        raise HTTPException(status_code=403, detail="Ongeldige API Key. Controleer je OMEGA-X Dashboard.")
    return user

app = FastAPI(title="OMEGA-X SaaS API (V15)")

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

@app.post("/api/generate", tags=["DSP"])
def generate_dataset(req: GenerateReq, user: dict = Depends(get_current_user)):
    res = core.generate_universal_signal(2.0, req.sr, req.base_f, req.harm_r, req.imp_r, req.noise_l)
    return {"status": "success", "samples": len(res["sig"]), "sr": req.sr, "user_id": user["id"]}

@app.post("/api/extract", tags=["DSP"])
async def extract_features(sr: int = 16000, file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    content = await file.read()
    features = core.extract_features_from_bytes(content, file.filename, sr)
    if "error" in features: return {"status": "error", "message": features["error"]}
    return {"status": "success", "features": features}

@app.post("/api/audit", tags=["ML Intelligence"])
def audit_dataset(dataset: list[dict], user: dict = Depends(get_current_user)):
    df = pd.DataFrame(dataset)
    if "Label" not in df.columns: return {"error": "Dataset must contain a 'Label' column"}
    X = df.drop(columns=["Label"]).replace([float('inf'), float('-inf')], 0).fillna(0)
    div, bal, sep = core.calculate_audit_scores(X, df["Label"])
    return {"diversity": div, "balance": bal, "separation": sep}

@app.post("/api/report", tags=["Enterprise"])
def generate_report(dataset: list[dict], project_name: str = "Headless_Audit", user: dict = Depends(get_current_user)):
    df = pd.DataFrame(dataset)
    X = df.drop(columns=["Label"]).replace([float('inf'), float('-inf')], 0).fillna(0)
    y = df["Label"]
    div, bal, sep = core.calculate_audit_scores(X, y)
    
    brds = core.get_available_hardware()
    b_dat = []
    for b in brds:
        ram, l_fft, l_f, l_inf = core.estimate_edge_load(b, len(X.columns), 16000)
        t_lat = l_fft + l_f + l_inf
        score = core.calculate_deployment_score(b, t_lat, ram)
        b_dat.append({"Board": b, "Score": score, "Latency": t_lat})
        
    best_board = max(b_dat, key=lambda x: x['Score'])['Board']
    pdf_bytes = core.generate_pdf_report(project_name, len(X), len(y.unique()), div, bal, sep, [], b_dat, best_board)
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=audit_{project_name}.pdf"})
