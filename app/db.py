"""SQLite storage for tile classification results."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "storage" / "tiles.db"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS results (
    content_sha256 TEXT PRIMARY KEY,
    original_filename TEXT,
    tile_path TEXT,
    status TEXT,
    predicted_class TEXT,
    confidence REAL,
    probabilities_json TEXT,
    model_name TEXT,
    preprocess_version TEXT,
    error TEXT,
    created_at TEXT
)
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(CREATE_SQL)
        conn.commit()


def get_by_hash(content_sha256: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM results WHERE content_sha256 = ?",
            (content_sha256,),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def insert(row: dict[str, Any]) -> dict[str, Any]:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO results (
                content_sha256,
                original_filename,
                tile_path,
                status,
                predicted_class,
                confidence,
                probabilities_json,
                model_name,
                preprocess_version,
                error,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["content_sha256"],
                row["original_filename"],
                row["tile_path"],
                row["status"],
                row["predicted_class"],
                row["confidence"],
                row["probabilities_json"],
                row["model_name"],
                row["preprocess_version"],
                row["error"],
                row["created_at"],
            ),
        )
        conn.commit()
    stored = get_by_hash(row["content_sha256"])
    if stored is None:
        raise RuntimeError("insert succeeded but row is missing")
    return stored


init_db()
