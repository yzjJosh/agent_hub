import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.lib.gc.runs import sweep_runs

_OLD_SCHEMA = """
CREATE TABLE tool_records (
  task_id TEXT PRIMARY KEY, tool_name TEXT NOT NULL, cache_key TEXT,
  params TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
  started_at TEXT NOT NULL, ended_at TEXT, run_dir TEXT NOT NULL
);
"""

_NOW_SCHEMA = """
CREATE TABLE task_records (
  task_id TEXT PRIMARY KEY, parent_task_id TEXT, status TEXT NOT NULL,
  error TEXT, started_at TEXT NOT NULL, ended_at TEXT, run_dir TEXT NOT NULL
);
CREATE TABLE tool_records (
  task_id TEXT PRIMARY KEY, tool_name TEXT NOT NULL, cache_key TEXT, params TEXT NOT NULL
);
CREATE TABLE agent_records (
  task_id TEXT PRIMARY KEY, name TEXT NOT NULL
);
"""


def _seed(data_dir: Path) -> None:
    (data_dir / "runs").mkdir()
    conn = sqlite3.connect(data_dir / "records.db")
    conn.executescript(_NOW_SCHEMA)
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=31)).isoformat()
    new = (now - timedelta(days=1)).isoformat()
    for tid, ended in [("old1", old), ("new1", new), ("running1", None)]:
        status = "running" if ended is None else "completed"
        conn.execute(
            "INSERT INTO task_records (task_id, parent_task_id, status, error, started_at, ended_at, run_dir)"
            " VALUES (?, NULL, ?, NULL, ?, ?, ?)",
            (tid, status, now.isoformat(), ended, f"runs/{tid}"),
        )
        conn.execute("INSERT INTO tool_records VALUES (?, 'tool', NULL, '{}')", (tid,))
        conn.execute("INSERT INTO agent_records VALUES (?, 'res')", (tid,))
    conn.commit()
    conn.close()
    for tid in ("old1", "new1", "running1", "orphan"):
        d = data_dir / "runs" / tid
        d.mkdir(exist_ok=True)
        (d / "log").write_text("x")


def _seed_legacy(data_dir: Path) -> None:
    (data_dir / "tool_runs").mkdir()
    conn = sqlite3.connect(data_dir / "tool.db")
    conn.executescript(_OLD_SCHEMA)
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=31)).isoformat()
    new = (now - timedelta(days=1)).isoformat()
    for tid, ended in [("old1", old), ("new1", new), ("running1", None)]:
        status = "running" if ended is None else "completed"
        conn.execute(
            "INSERT INTO tool_records VALUES (?,?,?,?,?,?,?,?,?)",
            (tid, "tool", None, "{}", status, None, now.isoformat(), ended, f"tool_runs/{tid}"),
        )
    conn.commit()
    conn.close()
    for tid in ("old1", "new1", "running1"):
        d = data_dir / "tool_runs" / tid
        d.mkdir(exist_ok=True)
        (d / "log").write_text("x")


def _ids(data_dir: Path, table: str) -> set[str]:
    conn = sqlite3.connect(data_dir / "records.db")
    ids = {r[0] for r in conn.execute(f"SELECT task_id FROM {table}")}
    conn.close()
    return ids


class TestSweepRuns:
    def test_deletes_stale_rows_and_dirs(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        removed = sweep_runs(30, str(tmp_path))
        assert str(tmp_path / "runs" / "old1") in removed
        assert not os.path.exists(tmp_path / "runs" / "old1")
        assert os.path.exists(tmp_path / "runs" / "new1")
        assert os.path.exists(tmp_path / "runs" / "running1")
        assert _ids(tmp_path, "task_records") == {"new1", "running1"}
        assert _ids(tmp_path, "tool_records") == {"new1", "running1"}

    def test_sweeps_agent_rows_child_first(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        sweep_runs(30, data_dir=str(tmp_path))
        assert _ids(tmp_path, "agent_records") == {"new1", "running1"}
        assert _ids(tmp_path, "task_records") == {"new1", "running1"}

    def test_orphan_dir_removed(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        removed = sweep_runs(30, str(tmp_path))
        assert str(tmp_path / "runs" / "orphan") in removed
        assert not os.path.exists(tmp_path / "runs" / "orphan")

    def test_no_db_no_dirs(self, tmp_path: Path) -> None:
        assert sweep_runs(30, str(tmp_path)) == []

    def test_days_override_keeps_younger(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        removed = sweep_runs(45, str(tmp_path))
        assert removed == [str(tmp_path / "runs" / "orphan")]
        assert os.path.exists(tmp_path / "runs" / "old1")
        assert _ids(tmp_path, "task_records") == {"old1", "new1", "running1"}

    def test_legacy_only_dir_migrates_first_and_deletes_nothing_live(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        removed = sweep_runs(30, str(tmp_path))
        assert (tmp_path / "records.db").exists()
        assert str(tmp_path / "runs" / "old1") in removed
        assert os.path.exists(tmp_path / "runs" / "new1")
        assert _ids(tmp_path, "task_records") == {"new1", "running1"}
        conn = sqlite3.connect(tmp_path / "tool.db")
        assert conn.execute("SELECT count(*) FROM tool_records").fetchone()[0] == 3
        conn.close()
