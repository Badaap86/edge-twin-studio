"""EdgeTwin Studio V31.1 local storage layer.

Purpose:
- Keep SQLite light by storing large DataFrames as files.
- Prefer Parquet when pyarrow/fastparquet is available.
- Fall back to compressed pickle so local installs do not break.
- Keep the API small so this can later be swapped for S3/MinIO.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

STORAGE_ROOT = Path(os.environ.get("EDGETWIN_STORAGE_ROOT", "storage"))
PROJECTS_DIR = STORAGE_ROOT / "projects"
EXPORTS_DIR = STORAGE_ROOT / "exports"


def _safe_id(value: str) -> str:
    value = str(value or "unknown")
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._")
    return value[:96] or "unknown"


def ensure_storage_dirs() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def dataframe_hash(df: pd.DataFrame) -> str:
    """Create a deterministic hash for a DataFrame without keeping a JSON copy in SQLite."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "empty"
    # CSV hashing is simple, stable, and works without optional parquet engines.
    payload = df.to_csv(index=False).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def project_dir(project_id: str) -> Path:
    ensure_storage_dirs()
    d = PROJECTS_DIR / _safe_id(project_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_dataframe(df: pd.DataFrame, project_id: str, kind: str = "dataset") -> Dict[str, Any]:
    ensure_storage_dirs()
    kind = _safe_id(kind)
    pdir = project_dir(project_id)
    rows = int(len(df)) if isinstance(df, pd.DataFrame) else 0
    cols = int(len(df.columns)) if isinstance(df, pd.DataFrame) else 0
    digest = dataframe_hash(df)

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()

    parquet_path = pdir / f"{kind}.parquet"
    pickle_path = pdir / f"{kind}.pkl.gz"
    meta_path = pdir / f"{kind}.meta.json"

    fmt = "parquet"
    path = parquet_path
    try:
        df.to_parquet(parquet_path, index=False)
        # If a previous fallback exists, leave it alone; metadata points to parquet.
    except Exception as exc:
        fmt = "pickle_gzip"
        path = pickle_path
        with gzip.open(pickle_path, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "project_id": str(project_id),
        "kind": kind,
        "path": str(path),
        "format": fmt,
        "rows": rows,
        "cols": cols,
        "hash": digest,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_dataframe(path: Optional[str], fmt: Optional[str] = None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    fmt = (fmt or p.suffix.lstrip(".")).lower()
    try:
        if fmt == "parquet" or p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        if fmt in {"pickle_gzip", "pkl", "pickle"} or p.name.endswith(".pkl.gz"):
            with gzip.open(p, "rb") as f:
                return pickle.load(f)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def get_storage_status() -> Dict[str, Any]:
    ensure_storage_dirs()
    files = list(STORAGE_ROOT.rglob("*")) if STORAGE_ROOT.exists() else []
    file_count = sum(1 for p in files if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files if p.is_file())
    return {
        "storage_root": str(STORAGE_ROOT),
        "projects_dir": str(PROJECTS_DIR),
        "exports_dir": str(EXPORTS_DIR),
        "file_count": file_count,
        "total_mb": round(total_bytes / (1024 * 1024), 3),
        "mode": "local_file_storage",
        "parquet_preferred": True,
        "fallback_format": "pickle_gzip",
        "future_cloud_target": "S3/MinIO compatible object storage",
    }
