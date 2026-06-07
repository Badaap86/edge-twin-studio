from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import zipfile
import io
import json
import core


# ============================================================
# API CONFIG
# ============================================================

core.init_db()

app = FastAPI(
    title="OMEGA-X SaaS API",
    version="V16.0",
    description="Industrial Edge AI Dataset Engineering API"
)


# ============================================================
# SECURITY
# ============================================================

api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)


async def get_current_user(api_key: str = Security(api_key_header)):
    user = core.verify_api_key(api_key)

    if not user:
        raise HTTPException(
            status_code=403,
            detail="Ongeldige API key. Controleer je OMEGA-X dashboard."
        )

    return user


# ============================================================
# REQUEST MODELS
# ============================================================

class GenerateReq(BaseModel):
    base_f: float
    harm_r: float
    imp_r: float
    noise_l: float
    sr: int = 16000
    duration: float = 2.0


class DeployReq(BaseModel):
    hardware: str
    num_features: int
    sr: int = 16000


class ArchitectReq(BaseModel):
    num_features: int
    sr: int = 16000
    target: str = "balanced"


class AuditReq(BaseModel):
    dataset: list[dict]


class ReportReq(BaseModel):
    dataset: list[dict]
    project_name: str = "Headless_Audit"


class IndustryPackReq(BaseModel):
    pack_name: str
    samples_per_class: int = 100


class AgingReq(BaseModel):
    base_f: float = 50.0
    harm_r: float = 0.05
    imp_r: float = 0.0
    noise_l: float = 0.08
    sr: int = 4000
    samples_per_stage: int = 100


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "name": "OMEGA-X SaaS API",
        "version": "V16.0",
        "status": "online"
    }


@app.get("/api/me")
def me(user: dict = Depends(get_current_user)):
    core.log_api_usage(user["id"], "/api/me")

    return {
        "status": "success",
        "user": user
    }


# ============================================================
# DSP GENERATION
# ============================================================

@app.post("/api/generate", tags=["DSP"])
def generate_signal(req: GenerateReq, user: dict = Depends(get_current_user)):
    core.log_api_usage(user["id"], "/api/generate")

    res = core.generate_universal_signal(
        duration=req.duration,
        sr=req.sr,
        base_f=req.base_f,
        harm_r=req.harm_r,
        imp_r=req.imp_r,
        noise_l=req.noise_l,
        normalize=True
    )

    return {
        "status": "success",
        "sample_rate": req.sr,
        "duration": req.duration,
        "samples": len(res["sig"]),
        "params": {
            "base_f": req.base_f,
            "harm_r": req.harm_r,
            "imp_r": req.imp_r,
            "noise_l": req.noise_l
        }
    }


@app.post("/api/generate-csv", tags=["DSP"])
def generate_signal_csv(req: GenerateReq, user: dict = Depends(get_current_user)):
    core.log_api_usage(user["id"], "/api/generate-csv")

    res = core.generate_universal_signal(
        duration=req.duration,
        sr=req.sr,
        base_f=req.base_f,
        harm_r=req.harm_r,
        imp_r=req.imp_r,
        noise_l=req.noise_l,
        normalize=True
    )

    df = pd.DataFrame({
        "time": res["t"],
        "value": res["sig"]
    })

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=omega_signal.csv"
        }
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@app.post("/api/extract", tags=["DSP"])
async def extract_features(
    sr: int = 16000,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    core.log_api_usage(user["id"], "/api/extract")

    content = await file.read()
    features = core.extract_features_from_bytes(content, file.filename, sr)

    if "error" in features:
        return {
            "status": "error",
            "message": features["error"]
        }

    return {
        "status": "success",
        "filename": file.filename,
        "features": features
    }


@app.post("/api/reverse-engineer", tags=["DSP"])
async def reverse_engineer(
    sr: int = 16000,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    core.log_api_usage(user["id"], "/api/reverse-engineer")

    content = await file.read()

    try:
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            sig = df.iloc[:, 1].astype(float).values
        else:
            return {
                "status": "error",
                "message": "Reverse engineer API ondersteunt momenteel CSV. Gebruik /api/extract voor WAV."
            }

        phys = core.reverse_engineer_physics(sig, sr)

        return {
            "status": "success",
            "filename": file.filename,
            "physics": phys
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================
# DATASET AUDIT
# ============================================================

@app.post("/api/audit", tags=["ML Intelligence"])
def audit_dataset(req: AuditReq, user: dict = Depends(get_current_user)):
    core.log_api_usage(user["id"], "/api/audit")

    df = pd.DataFrame(req.dataset)

    if "Label" not in df.columns:
        return {
            "status": "error",
            "message": "Dataset must contain a 'Label' column."
        }

    X = df.drop(columns=["Label"]).replace([float("inf"), float("-inf")], 0).fillna(0)
    y = df["Label"]

    div, bal, sep = core.calculate_audit_scores(X, y)

    return {
        "status": "success",
        "metrics": {
            "diversity": div,
            "balance": bal,
            "separation": sep,
            "overall": int((div * 0.35) + (bal * 0.30) + (sep * 0.35))
        }
    }


@app.post("/api/doctor", tags=["ML Intelligence"])
def dataset_doctor(req: AuditReq, user: dict = Depends(get_current_user)):
    core.log_api_usage(user["id"], "/api/doctor")

    df = pd.DataFrame(req.dataset)

    if "Label" not in df.columns:
        return {
            "status": "error",
            "message": "Dataset must contain a 'Label' column."
        }

    X = df.drop(columns=["Label"]).replace([float("inf"), float("-inf")], 0).fillna(0)
    y = df["Label"]

    doctor = core.dataset_doctor(X, y)

    return {
        "status": "success",
        "doctor": doctor
    }


# ============================================================
# HARDWARE DEPLOYMENT
# ============================================================

@app.get("/api/hardware", tags=["Hardware Validation"])
def list_hardware(user: dict = Depends(get_current_user)):
    core.log_api_usage(user["id"], "/api/hardware")

    return {
        "status": "success",
        "hardware": core.get_available_hardware()
    }


@app.post("/api/deployment", tags=["Hardware Validation"])
def check_deployment(req: DeployReq, user: dict = Depends(get_current_user)):
    core.log_api_usage(user["id"], "/api/deployment")

    ram, l_fft, l_feat, l_inf = core.estimate_edge_load(
        req.hardware,
        req.num_features,
        req.sr
    )

    latency = l_fft + l_feat + l_inf
    score = core.calculate_deployment_score(req.hardware, latency, ram)

    return {
        "status": "success",
        "hardware": req.hardware,
        "readiness_score": score,
        "latency_ms": latency,
        "ram_kb": ram,
        "breakdown": {
            "fft_ms": l_fft,
            "feature_ms": l_feat,
            "inference_ms": l_inf
        }
    }


@app.post("/api/architect", tags=["Hardware Validation"])
def hardware_architect(req: ArchitectReq, user: dict = Depends(get_current_user)):
    core.log_api_usage(user["id"], "/api/architect")

    result = core.hardware_auto_architect(
        num_features=req.num_features,
        sr=req.sr,
        target=req.target
    )

    return {
        "status": "success",
        "architect": result
    }


# ============================================================
# INDUSTRY PACKS
# ============================================================

@app.get("/api/industry-packs", tags=["Industry Packs"])
def industry_packs(user: dict = Depends(get_current_user)):
    core.log_api_usage(user["id"], "/api/industry-packs")

    packs = []

    for name in core.get_industry_packs():
        pack = core.get_industry_pack(name)
        packs.append({
            "name": name,
            "description": pack["description"],
            "sample_rate": pack["sample_rate"],
            "classes": list(pack["classes"].keys())
        })

    return {
        "status": "success",
        "packs": packs
    }


@app.post("/api/generate-pack", tags=["Industry Packs"])
def generate_pack(req: IndustryPackReq, user: dict = Depends(get_current_user)):
    core.log_api_usage(user["id"], "/api/generate-pack")

    files, manifest = core.generate_industry_pack_dataset(
        pack_name=req.pack_name,
        samples_per_class=req.samples_per_class
    )

    if "error" in manifest:
        return {
            "status": "error",
            "message": manifest["error"]
        }

    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        for item in files:
            zf.writestr(
                item["filename"],
                item["dataframe"].to_csv(index=False)
            )

        zf.writestr(
            "manifest.json",
            json.dumps(manifest, indent=2)
        )

    return Response(
        content=zip_buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={req.pack_name.replace(' ', '_')}.zip"
        }
    )


@app.post("/api/generate-aging", tags=["Industry Packs"])
def generate_aging(req: AgingReq, user: dict = Depends(get_current_user)):
    core.log_api_usage(user["id"], "/api/generate-aging")

    base_params = {
        "base_f": req.base_f,
        "harm_r": req.harm_r,
        "imp_r": req.imp_r,
        "noise_l": req.noise_l
    }

    files, manifest = core.generate_predictive_maintenance_aging_dataset(
        base_params=base_params,
        samples_per_stage=req.samples_per_stage,
        sr=req.sr
    )

    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        for item in files:
            zf.writestr(
                item["filename"],
                item["dataframe"].to_csv(index=False)
            )

        zf.writestr(
            "manifest.json",
            json.dumps(manifest, indent=2)
        )

    return Response(
        content=zip_buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=omega_aging_dataset.zip"
        }
    )


# ============================================================
# PDF REPORT
# ============================================================

@app.post("/api/report", tags=["Enterprise"])
def generate_report(req: ReportReq, user: dict = Depends(get_current_user)):
    core.log_api_usage(user["id"], "/api/report")

    df = pd.DataFrame(req.dataset)

    if "Label" not in df.columns:
        return {
            "status": "error",
            "message": "Dataset must contain a 'Label' column."
        }

    X = df.drop(columns=["Label"]).replace([float("inf"), float("-inf")], 0).fillna(0)
    y = df["Label"]

    div, bal, sep = core.calculate_audit_scores(X, y)

    b_dat = []

    for board in core.get_available_hardware():
        ram, l_fft, l_feat, l_inf = core.estimate_edge_load(
            board,
            len(X.columns),
            16000
        )

        latency = l_fft + l_feat + l_inf
        score = core.calculate_deployment_score(board, latency, ram)

        b_dat.append({
            "Board": board,
            "Score": score,
            "Latency": latency,
            "RAM": ram
        })

    best_board = max(b_dat, key=lambda x: x["Score"])["Board"] if b_dat else "Unknown"

    pdf_bytes = core.generate_pdf_report(
        proj_name=req.project_name,
        num_samples=len(X),
        num_classes=len(y.unique()),
        div=div,
        bal=bal,
        sep=sep,
        top_features=[],
        b_dat=b_dat,
        best_board=best_board
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=audit_{req.project_name}.pdf"
        }
    )


# ============================================================
# USAGE
# ============================================================

@app.get("/api/usage", tags=["SaaS"])
def usage(user: dict = Depends(get_current_user)):
    usage_df = core.get_user_usage(user["id"])

    if usage_df.empty:
        return {
            "status": "success",
            "usage": []
        }

    return {
        "status": "success",
        "usage": usage_df.to_dict(orient="records")
    }
