import json
import shutil
import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.lib.db.migration import migrate_all, migrate_all_sync

@pytest.fixture(autouse=True)
def _isolated_resources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RESOURCES_DIR", str(tmp_path / "resources"))


_OLD_SCHEMA = """
CREATE TABLE tool_records (
  task_id TEXT PRIMARY KEY, tool_name TEXT NOT NULL, cache_key TEXT,
  params TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
  started_at TEXT NOT NULL, ended_at TEXT, run_dir TEXT NOT NULL
);
CREATE INDEX idx_cache ON tool_records(tool_name, cache_key, started_at DESC);
CREATE INDEX idx_started ON tool_records(started_at);
"""

# This matches the tool_records table the store creates.
_FRESH_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_records (
  task_id   TEXT PRIMARY KEY,
  tool_name TEXT NOT NULL,
  cache_key TEXT,
  params    TEXT NOT NULL,
  render_type TEXT
);
"""


def _seed_legacy(data_dir: Path, n_rows: int = 3) -> None:
    (data_dir / "tool_runs").mkdir()
    conn = sqlite3.connect(data_dir / "tool.db")
    conn.executescript(_OLD_SCHEMA)
    now = datetime.now(timezone.utc)
    for i in range(n_rows):
        tid = f"t{i}"
        conn.execute(
            "INSERT INTO tool_records VALUES (?,?,?,?,?,?,?,?,?)",
            (tid, "tool", None, "{}", "completed", None, now.isoformat(), now.isoformat(), f"tool_runs/{tid}"),
        )
        d = data_dir / "tool_runs" / tid
        d.mkdir()
        (d / "log").write_text("x")
    conn.commit()
    conn.close()


def _read_run_dirs(db: str) -> set[str]:
    conn = sqlite3.connect(db)
    dirs = {r[0] for r in conn.execute("SELECT run_dir FROM task_records")}
    conn.close()
    return dirs


def _move_runs_dirs(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir(exist_ok=True)
    for i in range(3):
        shutil.move(str(tmp_path / "tool_runs" / f"t{i}"), str(tmp_path / "runs" / f"t{i}"))


def _run_concurrent(fn: Callable[[], None], n: int = 4) -> list[BaseException]:
    errors: list[BaseException] = []

    def run() -> None:
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


class TestMigrateAll:
    def test_full_sequence_from_legacy_only(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        legacy_before = (tmp_path / "tool.db").read_bytes()
        migrate_all_sync(str(tmp_path))
        assert (tmp_path / "records.db").exists()
        assert (tmp_path / "runs" / "t0").exists()
        assert not (tmp_path / "tool_runs").exists()
        assert (tmp_path / "tool.db").read_bytes() == legacy_before
        conn = sqlite3.connect(tmp_path / "records.db")
        assert conn.execute("SELECT count(*) FROM task_records").fetchone()[0] == 3
        conn.close()
        conn = sqlite3.connect(tmp_path / "tool.db")
        assert conn.execute("SELECT count(*) FROM tool_records").fetchone()[0] == 3
        conn.close()

    def test_no_legacy_creates_no_records_db(self, tmp_path: Path) -> None:
        migrate_all_sync(str(tmp_path))
        assert not (tmp_path / "records.db").exists()

    def test_splits_tables(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        migrate_all_sync(str(tmp_path))
        conn = sqlite3.connect(tmp_path / "records.db")
        task_cols = {r[1] for r in conn.execute("PRAGMA table_info(task_records)")}
        tool_cols = {r[1] for r in conn.execute("PRAGMA table_info(tool_records)")}
        n_task = conn.execute("SELECT count(*) FROM task_records").fetchone()[0]
        n_tool = conn.execute("SELECT count(*) FROM tool_records").fetchone()[0]
        parent = conn.execute("SELECT parent_task_id FROM task_records LIMIT 1").fetchone()[0]
        cache_cols = {r[2] for r in conn.execute("PRAGMA index_info(idx_cache)")}
        conn.close()
        assert {"task_id", "parent_task_id", "status", "error", "started_at", "ended_at", "run_dir"} <= task_cols
        assert tool_cols == {"task_id", "tool_name", "cache_key", "params", "render_type"}
        assert (n_task, n_tool) == (3, 3)
        assert parent is None
        assert cache_cols == {"tool_name", "cache_key"}

    def test_moves_dirs_and_repoints_rows(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        migrate_all_sync(str(tmp_path))
        assert (tmp_path / "runs" / "t0").exists()
        assert not (tmp_path / "tool_runs").exists()
        assert _read_run_dirs(str(tmp_path / "records.db")) == {f"runs/t{i}" for i in range(3)}

    def test_row_for_missing_dir_is_not_repointed(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        shutil.rmtree(tmp_path / "tool_runs" / "t1")
        migrate_all_sync(str(tmp_path))
        dirs = _read_run_dirs(str(tmp_path / "records.db"))
        assert "runs/t0" in dirs
        assert "tool_runs/t1" in dirs

    def test_self_heals_after_crash_between_move_and_update(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        _move_runs_dirs(tmp_path)
        migrate_all_sync(str(tmp_path))
        assert _read_run_dirs(str(tmp_path / "records.db")) == {f"runs/t{i}" for i in range(3)}

    def test_collision_keeps_runs_dir_and_leaves_row(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        (tmp_path / "runs" / "t0").mkdir(parents=True)
        (tmp_path / "runs" / "t0" / "result").write_text("winner")
        migrate_all_sync(str(tmp_path))
        assert (tmp_path / "runs" / "t0" / "result").read_text() == "winner"
        assert (tmp_path / "tool_runs" / "t0").exists()
        dirs = _read_run_dirs(str(tmp_path / "records.db"))
        assert "tool_runs/t0" in dirs and "runs/t1" in dirs

    def test_idempotent_second_call(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        migrate_all_sync(str(tmp_path))
        migrate_all_sync(str(tmp_path))
        conn = sqlite3.connect(tmp_path / "records.db")
        assert conn.execute("SELECT count(*) FROM task_records").fetchone()[0] == 3
        conn.close()

    def test_empty_records_db_with_legacy_is_replaced(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        sqlite3.connect(tmp_path / "records.db").close()
        migrate_all_sync(str(tmp_path))
        conn = sqlite3.connect(tmp_path / "records.db")
        assert conn.execute("SELECT count(*) FROM task_records").fetchone()[0] == 3
        conn.close()
        assert _read_run_dirs(str(tmp_path / "records.db")) == {f"runs/t{i}" for i in range(3)}
        assert not (tmp_path / "tool_runs").exists()

    def test_empty_records_db_crash_between_move_and_update(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        sqlite3.connect(tmp_path / "records.db").close()
        _move_runs_dirs(tmp_path)
        migrate_all_sync(str(tmp_path))
        assert _read_run_dirs(str(tmp_path / "records.db")) == {f"runs/t{i}" for i in range(3)}

    def test_empty_records_db_without_legacy_does_not_crash(self, tmp_path: Path) -> None:
        sqlite3.connect(tmp_path / "records.db").close()
        (tmp_path / "runs" / "t0").mkdir(parents=True)
        (tmp_path / "runs" / "t0" / "log").write_text("x")
        migrate_all_sync(str(tmp_path))
        assert (tmp_path / "runs" / "t0" / "log").read_text() == "x"

    def test_concurrent_first_migration(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        errors = _run_concurrent(lambda: migrate_all_sync(str(tmp_path)))
        assert not errors
        conn = sqlite3.connect(tmp_path / "records.db")
        assert conn.execute("SELECT count(*) FROM task_records").fetchone()[0] == 3
        tool_cols = {r[1] for r in conn.execute("PRAGMA table_info(tool_records)")}
        conn.close()
        assert tool_cols == {"task_id", "tool_name", "cache_key", "params", "render_type"}

    async def test_async_entry_runs_the_migration(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        await migrate_all(str(tmp_path))
        assert (tmp_path / "runs" / "t0").exists()
        assert _read_run_dirs(str(tmp_path / "records.db")) == {f"runs/t{i}" for i in range(3)}


def _seed_legacy_json(data_dir: Path, rid: str, files: list[str], created_at: str = "2026-09-01T00:00:00+00:00") -> None:
    h = data_dir / "handoff"
    h.mkdir(parents=True, exist_ok=True)
    (h / f"{rid}.json").write_text(
        json.dumps({"id": rid, "created_at": created_at, "files": files}), encoding="utf-8",
    )


def _seed_present_to_user_task(
    data_dir: Path, task_id: str, started_at: str, result_url: str, params: dict | None = None,
) -> None:
    conn = sqlite3.connect(data_dir / "records.db")
    conn.execute("CREATE TABLE IF NOT EXISTS task_records ("
                 "  task_id TEXT PRIMARY KEY, parent_task_id TEXT, status TEXT NOT NULL,"
                 "  error TEXT, started_at TEXT NOT NULL, ended_at TEXT, run_dir TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS tool_records ("
                 "  task_id TEXT PRIMARY KEY, tool_name TEXT NOT NULL, cache_key TEXT, params TEXT NOT NULL)")
    conn.execute(
        "INSERT OR REPLACE INTO task_records VALUES (?,?,?,?,?,?,?)",
        (task_id, None, "completed", None, started_at, started_at, f"runs/{task_id}"),
    )
    conn.execute(
        "INSERT OR REPLACE INTO tool_records VALUES (?,?,?,?)",
        (task_id, "present_to_user", None, json.dumps(params or {})),
    )
    conn.commit()
    conn.close()
    run_dir = data_dir / "runs" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result").write_text(
        f"Presented to the user at {result_url} · link valid until the file is garbage-collected (30 days)\n",
        encoding="utf-8",
    )


def _read_presentations(db: str) -> list[tuple]:
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT id, presented_at, url_path, files, task_id FROM presentations ORDER BY id").fetchall()
    conn.close()
    return rows


class TestPresentationMigration:
    def test_json_to_table(self, tmp_path: Path) -> None:
        (tmp_path / "records.db").touch()
        _seed_legacy_json(tmp_path, "a" * 16, ["handoffs/x.md", "upload/pic.png"])
        migrate_all_sync(str(tmp_path))
        rows = _read_presentations(str(tmp_path / "records.db"))
        assert len(rows) == 1
        rid, ts, url_path, files_json, task_id = rows[0]
        assert rid == "a" * 16
        assert ts == "2026-09-01T00:00:00+00:00"
        assert url_path == "/view/presentation/aaaaaaaaaaaaaaaa"
        assert json.loads(files_json) == ["presentations/x.md", "upload/pic.png"]
        assert task_id is None
        assert not (tmp_path / "handoff" / (f"{'a'*16}.json")).exists()

    def test_json_folder_removed_when_empty(self, tmp_path: Path) -> None:
        (tmp_path / "records.db").touch()
        _seed_legacy_json(tmp_path, "a" * 16, ["upload/pic.png"])
        migrate_all_sync(str(tmp_path))
        assert not (tmp_path / "handoff").exists()

    def test_malformed_json_deleted_no_row(self, tmp_path: Path) -> None:
        (tmp_path / "records.db").touch()
        h = tmp_path / "handoff"
        h.mkdir()
        (h / "bad1234567890ab.json").write_text("{not json", encoding="utf-8")
        migrate_all_sync(str(tmp_path))
        rows = _read_presentations(str(tmp_path / "records.db"))
        assert rows == []
        assert not (h / "bad1234567890ab.json").exists()

    def test_task_row_backfill_solo(self, tmp_path: Path) -> None:
        (tmp_path / "records.db").touch()
        _seed_present_to_user_task(
            tmp_path, "task-solo-1", "2026-09-01T12:00:00+00:00",
            "http://agent.example.com:8315/view/upload/pic.png",
        )
        migrate_all_sync(str(tmp_path))
        rows = _read_presentations(str(tmp_path / "records.db"))
        assert len(rows) == 1
        rid, ts, url_path, files_json, task_id = rows[0]
        assert rid == "task-solo-1"
        assert ts == "2026-09-01T12:00:00+00:00"
        assert url_path == "/view/upload/pic.png"
        assert json.loads(files_json) == ["upload/pic.png"]
        assert task_id == "task-solo-1"

    def test_task_row_backfill_multi_with_rewrite(self, tmp_path: Path) -> None:
        (tmp_path / "records.db").touch()
        _seed_present_to_user_task(
            tmp_path, "task-multi-1", "2026-09-01T12:00:00+00:00",
            "http://agent.example.com:8315/view/handoff/abcdef1234567890",
            {"files": ["upload/a.png", "upload/b.md"], "content": ""},
        )
        migrate_all_sync(str(tmp_path))
        rows = _read_presentations(str(tmp_path / "records.db"))
        assert len(rows) == 1
        rid, _ts, url_path, files_json, task_id = rows[0]
        assert rid == "abcdef1234567890"
        assert url_path == "/view/presentation/abcdef1234567890"
        assert json.loads(files_json) == ["upload/a.png", "upload/b.md"]
        assert task_id == "task-multi-1"

    def test_task_row_backfill_legacy_grammar(self, tmp_path: Path) -> None:
        (tmp_path / "records.db").touch()
        _seed_present_to_user_task(
            tmp_path, "task-legacy-1", "2026-09-01T12:00:00+00:00",
            "http://paste.example.com/upload/turkey-bat",
        )
        migrate_all_sync(str(tmp_path))
        rows = _read_presentations(str(tmp_path / "records.db"))
        assert len(rows) == 1
        rid, _ts, url_path, files_json, _task_id = rows[0]
        assert rid == "task-legacy-1"
        assert url_path == "/upload/turkey-bat"
        assert json.loads(files_json) == ["upload/turkey-bat"]

    def test_dedup_by_id(self, tmp_path: Path) -> None:
        (tmp_path / "records.db").touch()
        _seed_legacy_json(tmp_path, "a" * 16, ["upload/pic.png"])
        _seed_present_to_user_task(
            tmp_path, "task-1", "2026-09-01T12:00:00+00:00",
            "http://agent.example.com:8315/view/handoff/aaaaaaaaaaaaaaaa",
            {"files": ["upload/pic.png"], "content": ""},
        )
        migrate_all_sync(str(tmp_path))
        rows = _read_presentations(str(tmp_path / "records.db"))
        assert len(rows) == 1
        assert rows[0][4] is None

    def test_idempotent(self, tmp_path: Path) -> None:
        (tmp_path / "records.db").touch()
        _seed_legacy_json(tmp_path, "a" * 16, ["upload/pic.png"])
        migrate_all_sync(str(tmp_path))
        count1 = len(_read_presentations(str(tmp_path / "records.db")))
        migrate_all_sync(str(tmp_path))
        count2 = len(_read_presentations(str(tmp_path / "records.db")))
        assert count1 == count2 == 1

    def test_content_folder_move(self, tmp_path: Path) -> None:
        (tmp_path / "records.db").touch()
        res = tmp_path / "resources"
        legacy_dir = res / "handoffs"
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "x.md").write_text("hello", encoding="utf-8")
        migrate_all_sync(str(tmp_path))
        assert (res / "presentations" / "x.md").exists()
        assert not legacy_dir.exists()

    def test_content_folder_move_merge_into_existing(self, tmp_path: Path) -> None:
        (tmp_path / "records.db").touch()
        res = tmp_path / "resources"
        legacy_dir = res / "handoffs"
        presentations = res / "presentations"
        legacy_dir.mkdir(parents=True)
        presentations.mkdir(parents=True)
        (legacy_dir / "x.md").write_text("hello", encoding="utf-8")
        (presentations / "x.md").write_text("existing", encoding="utf-8")
        migrate_all_sync(str(tmp_path))
        assert (presentations / "x.md").read_text() == "existing"
        assert not legacy_dir.exists()

    def test_missing_result_file_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "records.db").touch()
        _seed_present_to_user_task(
            tmp_path, "task-noresult", "2026-09-01T12:00:00+00:00",
            "http://agent.example.com:8315/view/upload/pic.png",
        )
        (tmp_path / "runs" / "task-noresult" / "result").unlink()
        migrate_all_sync(str(tmp_path))
        rows = _read_presentations(str(tmp_path / "records.db"))
        assert rows == []

    def test_adds_title_column_to_existing_table(self, tmp_path: Path) -> None:
        (tmp_path / "records.db").touch()
        conn = sqlite3.connect(tmp_path / "records.db")
        conn.executescript(
            "CREATE TABLE presentations ("
            "  id TEXT PRIMARY KEY, presented_at TEXT NOT NULL, url_path TEXT NOT NULL,"
            "  files TEXT NOT NULL, task_id TEXT);"
        )
        conn.close()
        migrate_all_sync(str(tmp_path))
        migrate_all_sync(str(tmp_path))
        conn = sqlite3.connect(tmp_path / "records.db")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(presentations)")}
        conn.close()
        assert "title" in cols

    def test_legacy_folder_rewrite_in_files(self, tmp_path: Path) -> None:
        (tmp_path / "records.db").touch()
        _seed_legacy_json(tmp_path, "b" * 16, ["handoffs/token.md", "upload/pic.png"])
        migrate_all_sync(str(tmp_path))
        rows = _read_presentations(str(tmp_path / "records.db"))
        assert json.loads(rows[0][3]) == ["presentations/token.md", "upload/pic.png"]


class TestAgentRecordsMigration:
    def test_agent_records_created_on_preexisting_db(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "records.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE task_records (task_id TEXT PRIMARY KEY, parent_task_id TEXT,"
            " status TEXT NOT NULL, error TEXT, started_at TEXT NOT NULL, ended_at TEXT, run_dir TEXT NOT NULL)"
        )
        conn.execute("CREATE TABLE tool_records (task_id TEXT PRIMARY KEY, tool_name TEXT NOT NULL, cache_key TEXT, params TEXT NOT NULL)")
        conn.commit()
        conn.close()

        migrate_all_sync(str(data_dir))
        # idempotent second pass
        migrate_all_sync(str(data_dir))

        conn = sqlite3.connect(db_path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        agent_cols = {r[1] for r in conn.execute("PRAGMA table_info(agent_records)")}
        conn.close()
        assert "agent_records" in tables
        assert agent_cols == {"task_id", "name"}


class TestToolRecordsRenderTypeMigration:
    """The migration step that adds the render kind column."""

    def _tool_records_columns(self, db_path: Path) -> list[str]:
        conn = sqlite3.connect(db_path)
        names = [r[1] for r in conn.execute("PRAGMA table_info(tool_records)")]
        conn.close()
        return names

    def test_adds_column_once_across_double_run(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        migrate_all_sync(str(tmp_path))
        migrate_all_sync(str(tmp_path))
        names = self._tool_records_columns(tmp_path / "records.db")
        assert names.count("render_type") == 1

    def test_legacy_rows_come_up_null(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        migrate_all_sync(str(tmp_path))
        conn = sqlite3.connect(tmp_path / "records.db")
        values = [r[0] for r in conn.execute("SELECT render_type FROM tool_records")]
        conn.close()
        assert values == [None, None, None]

    def test_stored_value_survives_rerun(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        migrate_all_sync(str(tmp_path))
        conn = sqlite3.connect(tmp_path / "records.db")
        conn.execute("UPDATE tool_records SET render_type = 'markdown' WHERE task_id = 't0'")
        conn.commit()
        conn.close()
        migrate_all_sync(str(tmp_path))
        conn = sqlite3.connect(tmp_path / "records.db")
        value = conn.execute(
            "SELECT render_type FROM tool_records WHERE task_id = 't0'").fetchone()[0]
        conn.close()
        assert value == "markdown"

    def test_missing_tool_records_table_is_noop(self, tmp_path: Path) -> None:
        # A DB file with a sibling table and no tool_records.
        conn = sqlite3.connect(tmp_path / "records.db")
        conn.execute("CREATE TABLE sibling (x)")
        conn.commit()
        conn.close()
        migrate_all_sync(str(tmp_path))
        migrate_all_sync(str(tmp_path))
        conn = sqlite3.connect(tmp_path / "records.db")
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        assert "tool_records" not in tables

    def test_fresh_schema_and_migrated_schema_agree(self, tmp_path: Path) -> None:
        _seed_legacy(tmp_path)
        migrate_all_sync(str(tmp_path))
        migrated = set(self._tool_records_columns(tmp_path / "records.db"))
        fresh = sqlite3.connect(tmp_path / "fresh.db")
        fresh.executescript(_FRESH_SCHEMA)
        fresh_cols = {r[1] for r in fresh.execute("PRAGMA table_info(tool_records)")}
        fresh.close()
        assert migrated == fresh_cols
