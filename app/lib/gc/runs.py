from __future__ import annotations

import os
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone

from app.lib.db import migration
from app.lib.shared import paths

__all__ = ["sweep_runs"]


def sweep_runs(days: int, data_dir: str | None = None) -> list[str]:
    """Remove run folders for old finished tasks and folders no task points at.

    Args:
        days: A task that ended more than this many days ago is removed.
        data_dir: Folder holding the records file and the run folders.
            Defaults to the DATA_DIR environment value, or /app/data.

    Returns:
        Full paths of the removed folders.
    """
    data_dir = data_dir or paths.data_dir()
    migration.migrate_all_sync(data_dir)
    db_path = os.path.join(data_dir, "records.db")
    removed: list[str] = []

    live_dirs: set[str] = set()
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path, timeout=10)
        try:
            conn.execute("PRAGMA busy_timeout=10000")
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            stale = conn.execute(
                "SELECT run_dir FROM task_records"
                " WHERE ended_at IS NOT NULL AND ended_at < ?",
                (cutoff,),
            ).fetchall()
            for (run_dir,) in stale:
                path = os.path.join(data_dir, run_dir)
                shutil.rmtree(path, ignore_errors=True)
                conn.execute(
                    "DELETE FROM agent_records"
                    " WHERE task_id IN (SELECT task_id FROM task_records WHERE run_dir = ?)",
                    (run_dir,),
                )
                conn.execute(
                    "DELETE FROM tool_records"
                    " WHERE task_id IN (SELECT task_id FROM task_records WHERE run_dir = ?)",
                    (run_dir,),
                )
                conn.execute("DELETE FROM task_records WHERE run_dir = ?", (run_dir,))
                removed.append(path)
            conn.commit()
            live_dirs = {row[0] for row in conn.execute("SELECT run_dir FROM task_records")}
        finally:
            conn.close()

    for root_name in ("runs", "tool_runs"):
        root = os.path.join(data_dir, root_name)
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if os.path.isdir(full) and f"{root_name}/{entry}" not in live_dirs:
                shutil.rmtree(full, ignore_errors=True)
                removed.append(full)
    return removed
