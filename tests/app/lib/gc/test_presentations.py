import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.lib.gc.presentations import sweep_presentations


def _make_db(db_path: Path, rows: list[tuple[str, str]]) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS presentations ("
        "  id TEXT PRIMARY KEY, presented_at TEXT NOT NULL,"
        "  url_path TEXT NOT NULL, files TEXT NOT NULL, task_id TEXT)"
    )
    for rid, ts in rows:
        conn.execute(
            "INSERT INTO presentations (id, presented_at, url_path, files) VALUES (?, ?, ?, ?)",
            (rid, ts, f"/view/{rid}", '[]'),
        )
    conn.commit()
    conn.close()


def test_removes_rows_older_than_window(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=31)).isoformat()
    recent = (now - timedelta(days=1)).isoformat()
    db = tmp_path / "records.db"
    _make_db(db, [("a" * 16, old), ("b" * 16, recent)])
    removed = sweep_presentations(30, str(tmp_path))
    assert removed == ["a" * 16]
    conn = sqlite3.connect(db)
    ids = {r[0] for r in conn.execute("SELECT id FROM presentations")}
    conn.close()
    assert ids == {"b" * 16}


def test_keeps_all_when_nothing_stale(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(days=1)).isoformat()
    db = tmp_path / "records.db"
    _make_db(db, [("a" * 16, recent)])
    assert sweep_presentations(30, str(tmp_path)) == []


def test_missing_db_returns_empty(tmp_path: Path) -> None:
    assert sweep_presentations(30, str(tmp_path)) == []


def test_missing_table_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "records.db").touch()
    assert sweep_presentations(30, str(tmp_path)) == []


def test_custom_days(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=2)).isoformat()
    recent = (now - timedelta(days=1)).isoformat()
    db = tmp_path / "records.db"
    _make_db(db, [("a" * 16, old), ("b" * 16, recent)])
    removed = sweep_presentations(1, str(tmp_path))
    assert "a" * 16 in removed
