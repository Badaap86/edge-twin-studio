"""EdgeTwin Studio V31.2 local storage layer.

Purpose:
- Keep SQLite light by storing large DataFrames as files.
- Prefer Parquet when pyarrow/fastparquet is available.
- Fall back to compressed pickle so local installs do not break.
- Use atomic writes so interrupted saves do not corrupt existing data.
- Keep the API small so this can later be swapped for S3/MinIO.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

STORAGE_VERSION = "31.2"

STORAGE_ROOT = Path(os.environ.get("EDGETWIN_STORAGE_ROOT", "storage"))
PROJECTS_DIR = STORAGE_ROOT / "projects"
EXPORTS_DIR = STORAGE_ROOT / "exports"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    value = str(value or "unknown")
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._")
    return value[:96] or "unknown"


def ensure_storage_dirs() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_replace(temp_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.replace(target_path)


def _atomic_write_bytes(target_path: Path, data: bytes) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=str(target_path.parent),
        prefix=f".{target_path.name}.",
        suffix=".tmp",
    ) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    _atomic_replace(tmp_path, target_path)


def _atomic_write_text(target_path: Path, text: str) -> None:
    _atomic_write_bytes(target_path, text.encode("utf-8"))


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

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()

    rows = int(len(df))
    cols = int(len(df.columns))
    digest = dataframe_hash(df)

    parquet_path = pdir / f"{kind}.parquet"
    pickle_path = pdir / f"{kind}.pkl.gz"
    meta_path = pdir / f"{kind}.meta.json"

    fmt = "parquet"
    path = parquet_path
    error_message = None

    try:
        tmp_parquet = parquet_path.with_name(f".{parquet_path.name}.tmp")
        df.to_parquet(tmp_parquet, index=False)
        _atomic_replace(tmp_parquet, parquet_path)

        # If Parquet now works, remove stale fallback so metadata and files stay clean.
        if pickle_path.exists():
            pickle_path.unlink()

    except Exception as exc:
        fmt = "pickle_gzip"
        path = pickle_path
        error_message = str(exc)

        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=str(pickle_path.parent),
            prefix=f".{pickle_path.name}.",
            suffix=".tmp",
        ) as tmp:
            with gzip.GzipFile(fileobj=tmp, mode="wb") as gz:
                pickle.dump(df, gz, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)

        _atomic_replace(tmp_path, pickle_path)

        # Clean broken/incomplete parquet if fallback was used.
        if parquet_path.exists() and parquet_path.stat().st_size == 0:
            parquet_path.unlink()

    size_bytes = path.stat().st_size if path.exists() else 0

    meta = {
        "storage_version": STORAGE_VERSION,
        "created_at": _utc_now(),
        "project_id": str(project_id),
        "safe_project_id": _safe_id(project_id),
        "kind": kind,
        "path": str(path),
        "absolute_path": str(path.resolve()),
        "format": fmt,
        "rows": rows,
        "cols": cols,
        "hash": digest,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
        "parquet_error_if_fallback": error_message,
    }

    _atomic_write_text(meta_path, json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def load_dataframe(path: Optional[str], fmt: Optional[str] = None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()

    p = Path(path)

    if not p.exists() or not p.is_file():
        return pd.DataFrame()

    fmt = (fmt or "").lower().strip()

    try:
        if fmt == "parquet" or p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)

        # Pickle is only used for files EdgeTwin itself wrote as local fallback.
        if fmt == "pickle_gzip" or p.name.endswith(".pkl.gz"):
            with gzip.open(p, "rb") as f:
                loaded = pickle.load(f)

            if isinstance(loaded, pd.DataFrame):
                return loaded

            return pd.DataFrame()

    except Exception:
        return pd.DataFrame()

    return pd.DataFrame()


def load_dataframe_from_meta(meta_path: str | Path) -> pd.DataFrame:
    p = Path(meta_path)

    if not p.exists() or not p.is_file():
        return pd.DataFrame()

    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
        return load_dataframe(meta.get("path"), meta.get("format"))
    except Exception:
        return pd.DataFrame()


def list_project_storage(project_id: str) -> Dict[str, Any]:
    pdir = project_dir(project_id)
    metas = []

    for meta_path in sorted(pdir.glob("*.meta.json")):
        try:
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            continue

    return {
        "project_id": str(project_id),
        "safe_project_id": _safe_id(project_id),
        "project_dir": str(pdir),
        "items": metas,
        "item_count": len(metas),
    }


def get_storage_status() -> Dict[str, Any]:
    ensure_storage_dirs()

    files = list(STORAGE_ROOT.rglob("*")) if STORAGE_ROOT.exists() else []
    file_count = sum(1 for p in files if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files if p.is_file())

    project_count = 0
    if PROJECTS_DIR.exists():
        project_count = sum(1 for p in PROJECTS_DIR.iterdir() if p.is_dir())

    return {
        "storage_version": STORAGE_VERSION,
        "storage_root": str(STORAGE_ROOT),
        "projects_dir": str(PROJECTS_DIR),
        "exports_dir": str(EXPORTS_DIR),
        "project_count": project_count,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 3),
        "mode": "local_file_storage",
        "parquet_preferred": True,
        "fallback_format": "pickle_gzip",
        "future_cloud_target": "S3/MinIO compatible object storage",
    }
