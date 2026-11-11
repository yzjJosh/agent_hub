"""Time-based cleanup of presentation rows."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone

from app.lib.shared import paths

__all__ = ["sweep_presentations"]


def sweep_presentations(days: int, data_dir: str | None = None) -> list[str]:
    """Remove presentation rows older than the window.

    Args:
        days: A presentation older than this many days is removed.
        data_dir: Folder holding the records file.
            Defaults to the DATA_DIR environment value, or /app/data.

    Returns:
        Ids of the removed presentation rows.
    """
    data_dir = data_dir or paths.data_dir()
    db_path = os.path.join(data_dir, "records.db")
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        if conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='presentations'"
        ).fetchone() is None:
            return []
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        stale = conn.execute(
            "SELECT id FROM presentations WHERE presented_at < ?", (cutoff,)
        ).fetchall()
        ids = [row[0] for row in stale]
        if ids:
            conn.executemany(
                "DELETE FROM presentations WHERE id = ?", [(i,) for i in ids]
            )
            conn.commit()
        return ids
    finally:
        conn.close()
