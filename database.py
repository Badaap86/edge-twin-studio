"""EdgeTwin Studio database layer.

Purpose:
- Keep a small local SQLite metadata database.
- Store project metadata, events, and key/value settings.
- Do NOT store large DataFrames here.
- Large datasets should stay in storage.py as Parquet/pickle files.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DB_PATH = Path(os.environ.get("EDGETWIN_DB_PATH", "storage/edgetwin_metadata.sqlite3"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_db_dir(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return "{}"


def _json_loads(value: Optional[str], default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    ensure_db_dir(db_path)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def init_db(db_path: Path = DB_PATH) -> Path:
    """Initialize local metadata database."""
    ensure_db_dir(db_path)

    with get_connection(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                project_name TEXT,
                status TEXT DEFAULT 'active',
                created_at TEXT,
                updated_at TEXT,
                metadata_json TEXT
            )
            """
        )

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS project_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT,
                event_type TEXT,
                message TEXT,
                created_at TEXT,
                metadata_json TEXT
            )
            """
        )

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS app_status (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
            """
        )

        con.execute(
            """
            INSERT OR REPLACE INTO app_status(key, value, updated_at)
            VALUES (?, ?, ?)
            """,
            ("database_mode", "local_sqlite_metadata", utc_now()),
        )

    return db_path


def init_database(db_path: Path = DB_PATH) -> Path:
    """Alias for older code that may call init_database()."""
    return init_db(db_path)


def upsert_project(
    project_id: str,
    project_name: Optional[str] = None,
    status: str = "active",
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Path = DB_PATH,
) -> Dict[str, Any]:
    """Create or update a project metadata record."""
    init_db(db_path)

    project_id = str(project_id or "unknown")
    project_name = str(project_name or project_id)
    status = str(status or "active")
    metadata = metadata or {}
    now = utc_now()

    with get_connection(db_path) as con:
        existing = con.execute(
            "SELECT created_at FROM projects WHERE project_id = ?",
            (project_id,),
        ).fetchone()

        created_at = existing["created_at"] if existing else now

        con.execute(
            """
            INSERT OR REPLACE INTO projects (
                project_id,
                project_name,
                status,
                created_at,
                updated_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                project_name,
                status,
                created_at,
                now,
                _json_dumps(metadata),
            ),
        )

    return {
        "project_id": project_id,
        "project_name": project_name,
        "status": status,
        "created_at": created_at,
        "updated_at": now,
        "metadata": metadata,
    }


def get_project(project_id: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """Load one project metadata record."""
    init_db(db_path)

    with get_connection(db_path) as con:
        row = con.execute(
            """
            SELECT project_id, project_name, status, created_at, updated_at, metadata_json
            FROM projects
            WHERE project_id = ?
            """,
            (str(project_id),),
        ).fetchone()

    if not row:
        return None

    return {
        "project_id": row["project_id"],
        "project_name": row["project_name"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "metadata": _json_loads(row["metadata_json"], {}),
    }


def list_projects(
    status: Optional[str] = None,
    limit: int = 100,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """List project metadata records."""
    init_db(db_path)

    limit = max(1, min(int(limit or 100), 1000))

    with get_connection(db_path) as con:
        if status:
            rows = con.execute(
                """
                SELECT project_id, project_name, status, created_at, updated_at, metadata_json
                FROM projects
                WHERE status = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (str(status), limit),
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT project_id, project_name, status, created_at, updated_at, metadata_json
                FROM projects
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    return [
        {
            "project_id": row["project_id"],
            "project_name": row["project_name"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": _json_loads(row["metadata_json"], {}),
        }
        for row in rows
    ]


def archive_project(project_id: str, db_path: Path = DB_PATH) -> bool:
    """Soft-archive a project instead of deleting it."""
    init_db(db_path)

    with get_connection(db_path) as con:
        cur = con.execute(
            """
            UPDATE projects
            SET status = ?, updated_at = ?
            WHERE project_id = ?
            """,
            ("archived", utc_now(), str(project_id)),
        )

    return cur.rowcount > 0


def add_event(
    project_id: str,
    event_type: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Path = DB_PATH,
) -> Dict[str, Any]:
    """Add a lightweight audit/event record."""
    init_db(db_path)

    project_id = str(project_id or "unknown")
    event_type = str(event_type or "event")
    message = str(message or "")
    metadata = metadata or {}
    now = utc_now()

    with get_connection(db_path) as con:
        cur = con.execute(
            """
            INSERT INTO project_events (
                project_id,
                event_type,
                message,
                created_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                project_id,
                event_type,
                message,
                now,
                _json_dumps(metadata),
            ),
        )
        event_id = int(cur.lastrowid)

    return {
        "id": event_id,
        "project_id": project_id,
        "event_type": event_type,
        "message": message,
        "created_at": now,
        "metadata": metadata,
    }


def list_events(
    project_id: Optional[str] = None,
    limit: int = 100,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """List recent project events."""
    init_db(db_path)

    limit = max(1, min(int(limit or 100), 1000))

    with get_connection(db_path) as con:
        if project_id:
            rows = con.execute(
                """
                SELECT id, project_id, event_type, message, created_at, metadata_json
                FROM project_events
                WHERE project_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(project_id), limit),
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT id, project_id, event_type, message, created_at, metadata_json
                FROM project_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    return [
        {
            "id": int(row["id"]),
            "project_id": row["project_id"],
            "event_type": row["event_type"],
            "message": row["message"],
            "created_at": row["created_at"],
            "metadata": _json_loads(row["metadata_json"], {}),
        }
        for row in rows
    ]


def set_kv(key: str, value: Any, db_path: Path = DB_PATH) -> Dict[str, Any]:
    """Store a small JSON-safe key/value setting."""
    init_db(db_path)

    key = str(key or "").strip()
    if not key:
        raise ValueError("key is required")

    now = utc_now()

    with get_connection(db_path) as con:
        existing = con.execute(
            "SELECT created_at FROM kv_store WHERE key = ?",
            (key,),
        ).fetchone()

        created_at = existing["created_at"] if existing else now

        con.execute(
            """
            INSERT OR REPLACE INTO kv_store (
                key,
                value_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                key,
                _json_dumps(value),
                created_at,
                now,
            ),
        )

    return {
        "key": key,
        "value": value,
        "created_at": created_at,
        "updated_at": now,
    }


def get_kv(key: str, default: Any = None, db_path: Path = DB_PATH) -> Any:
    """Load a small JSON-safe key/value setting."""
    init_db(db_path)

    with get_connection(db_path) as con:
        row = con.execute(
            "SELECT value_json FROM kv_store WHERE key = ?",
            (str(key),),
        ).fetchone()

    if not row:
        return default

    return _json_loads(row["value_json"], default)


def delete_kv(key: str, db_path: Path = DB_PATH) -> bool:
    """Delete a key/value setting."""
    init_db(db_path)

    with get_connection(db_path) as con:
        cur = con.execute(
            "DELETE FROM kv_store WHERE key = ?",
            (str(key),),
        )

    return cur.rowcount > 0


def get_database_status(db_path: Path = DB_PATH) -> Dict[str, Any]:
    """Return lightweight database health/status information."""
    init_db(db_path)

    with get_connection(db_path) as con:
        project_count = con.execute("SELECT COUNT(*) AS c FROM projects").fetchone()["c"]
        event_count = con.execute("SELECT COUNT(*) AS c FROM project_events").fetchone()["c"]
        kv_count = con.execute("SELECT COUNT(*) AS c FROM kv_store").fetchone()["c"]

    size_bytes = db_path.stat().st_size if db_path.exists() else 0

    return {
        "db_path": str(db_path),
        "mode": "local_sqlite_metadata",
        "project_count": int(project_count),
        "event_count": int(event_count),
        "kv_count": int(kv_count),
        "size_bytes": int(size_bytes),
        "size_mb": round(size_bytes / (1024 * 1024), 3),
        "large_dataframe_policy": "store large dataframes in storage.py, not SQLite",
        "updated_at": utc_now(),
    }


# Backwards-compatible aliases
save_project = upsert_project
load_project = get_project
save_setting = set_kv
load_setting = get_kv


if __name__ == "__main__":
    init_db()
    print(json.dumps(get_database_status(), indent=2))
