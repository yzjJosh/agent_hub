from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import shutil
import sqlite3
from urllib.parse import unquote, urlsplit

from app.lib.shared.paths import resources_dir

_logger = logging.getLogger(__name__)

__all__ = ["migrate_all", "migrate_all_sync"]

_LOCK_NAME = ".records.lock"
_DB_NAME = "records.db"
_LEGACY_DB_NAME = "tool.db"
_TASK_COLS = ("status", "error", "started_at", "ended_at", "run_dir")

_PRESENTATIONS_DDL = """
CREATE TABLE IF NOT EXISTS presentations (
  id           TEXT PRIMARY KEY,
  presented_at TEXT NOT NULL,
  url_path     TEXT NOT NULL,
  files        TEXT NOT NULL,
  task_id      TEXT,
  title        TEXT
);
CREATE INDEX IF NOT EXISTS idx_presentation_at ON presentations(presented_at);
"""

_AGENT_RECORDS_DDL = """
CREATE TABLE IF NOT EXISTS agent_records (
  task_id  TEXT PRIMARY KEY REFERENCES task_records(task_id),
  name     TEXT NOT NULL
);
"""

_PRESENT_TO_USER = "present_to_user"
_RESULT_URL_PREFIX = "Presented to the user at "


def _has_tables(db_path: str) -> bool:
    conn = sqlite3.connect(db_path, timeout=10, isolation_level=None)
    try:
        return conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone() is not None
    finally:
        conn.close()


def _ensure_database(db_path: str, legacy_path: str | None = None) -> None:
    """Copy the old tool.db onto records.db when records.db is missing or empty."""
    if os.path.exists(db_path) and _has_tables(db_path):
        return
    if legacy_path is None or not os.path.exists(legacy_path):
        return
    tmp = db_path + ".copy-tmp"
    shutil.copy2(legacy_path, tmp)
    os.replace(tmp, db_path)


def _migrate_schema(db_path: str) -> None:
    """Split an old wide tool_records table into task_records and a slim tool_records."""
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path, timeout=10, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "tool_records" not in tables:
            return
        cols = {r[1] for r in conn.execute("PRAGMA table_info(tool_records)")}
        if "status" not in cols:
            return
        conn.execute("BEGIN")
        conn.execute(
            "CREATE TABLE task_records ("
            "  task_id TEXT PRIMARY KEY,"
            "  parent_task_id TEXT REFERENCES task_records(task_id),"
            "  status TEXT NOT NULL, error TEXT, started_at TEXT NOT NULL,"
            "  ended_at TEXT, run_dir TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO task_records (task_id, parent_task_id, status, error, started_at, ended_at, run_dir)"
            " SELECT task_id, NULL, status, error, started_at, ended_at, run_dir FROM tool_records"
        )
        conn.execute("DROP INDEX IF EXISTS idx_cache")
        conn.execute("DROP INDEX IF EXISTS idx_started")
        for col in _TASK_COLS:
            conn.execute(f"ALTER TABLE tool_records DROP COLUMN {col}")
        conn.execute("CREATE INDEX idx_cache ON tool_records(tool_name, cache_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_task_started ON task_records(started_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_task_parent ON task_records(parent_task_id)")
        conn.execute("COMMIT")
    finally:
        conn.close()


def _migrate_runs_dir(data_dir: str, db_path: str) -> None:
    """Move old tool_runs folders into runs and point the rows at the new place."""
    legacy_root = os.path.join(data_dir, "tool_runs")
    runs_root = os.path.join(data_dir, "runs")
    os.makedirs(runs_root, exist_ok=True)
    if os.path.isdir(legacy_root):
        for entry in sorted(os.listdir(legacy_root)):
            src = os.path.join(legacy_root, entry)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(runs_root, entry)
            if os.path.exists(dst):
                _logger.warning(
                    "run dir collision for '%s'; keeping runs/%s, leaving tool_runs/%s in place",
                    entry, entry, entry,
                )
                continue
            shutil.move(src, dst)
        if not os.listdir(legacy_root):
            os.rmdir(legacy_root)
    if not os.path.exists(db_path):
        return
    legacy_names = set(os.listdir(legacy_root)) if os.path.isdir(legacy_root) else set()
    moved = set(os.listdir(runs_root)) - legacy_names
    conn = sqlite3.connect(db_path, timeout=10, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_records'").fetchone() is None:
            return
        conn.execute("BEGIN")
        for name in sorted(moved):
            conn.execute(
                "UPDATE task_records SET run_dir = ? WHERE run_dir = ?",
                (f"runs/{name}", f"tool_runs/{name}"),
            )
        conn.execute("COMMIT")
    finally:
        conn.close()


def _rewrite_url_path(url_path: str) -> str:
    """Apply the handoff to presentation renames to a URL path."""
    return url_path.replace("/view/handoff/", "/view/presentation/").replace("handoffs/", "presentations/")


def _rewrite_files(files: list[str]) -> list[str]:
    """Apply the content folder rename to a file list."""
    return [f.replace("handoffs/", "presentations/") for f in files]


def _extract_url_path(full_url: str) -> str:
    """The path part of a full URL, with the leading slash kept."""
    return urlsplit(full_url).path


def _load_legacy_record(path: str) -> dict | None:
    """Read one handoff JSON record, or None when the record is malformed."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, json.JSONDecodeError):
        _logger.warning("malformed handoff record %s, deleting without a row", path)
        return None
    if not isinstance(rec, dict):
        _logger.warning("malformed handoff record %s, deleting without a row", path)
        return None
    files = rec.get("files")
    created_at = rec.get("created_at")
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files) or not isinstance(created_at, str):
        _logger.warning("malformed handoff record %s, deleting without a row", path)
        return None
    return rec


def _migrate_legacy_records(data_dir: str, conn: sqlite3.Connection) -> None:
    """Move handoff JSON records into the presentations table."""
    legacy_dir = os.path.join(data_dir, "handoff")
    if not os.path.isdir(legacy_dir):
        return
    moved: list[str] = []
    for name in sorted(os.listdir(legacy_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(legacy_dir, name)
        rec = _load_legacy_record(path)
        if rec is None:
            os.remove(path)
            continue
        rid = name.removesuffix(".json")
        conn.execute(
            "INSERT OR IGNORE INTO presentations (id, presented_at, url_path, files, task_id)"
            " VALUES (?, ?, ?, ?, NULL)",
            (rid, rec["created_at"], f"/view/presentation/{rid}", json.dumps(_rewrite_files(rec["files"]), ensure_ascii=False)),
        )
        moved.append(path)
    for path in moved:
        os.remove(path)
    if not os.listdir(legacy_dir):
        os.rmdir(legacy_dir)


def _result_url_path(result_path: str) -> str | None:
    """The rewritten presentation URL path in a run result file, or None."""
    if not os.path.isfile(result_path):
        return None
    try:
        with open(result_path, "r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
    except OSError:
        return None
    if not first_line.startswith(_RESULT_URL_PREFIX):
        return None
    full_url = first_line[len(_RESULT_URL_PREFIX):].split(" · ", 1)[0].strip()
    return _rewrite_url_path(_extract_url_path(full_url))


def _params_files(params_json: str | None) -> list[str]:
    """The file list from stored tool params, or an empty list when absent."""
    if not params_json:
        return []
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError:
        return []
    files = params.get("files") if isinstance(params, dict) else None
    if isinstance(files, list) and all(isinstance(f, str) for f in files):
        return files
    return []


def _backfill_task_rows(data_dir: str, conn: sqlite3.Connection) -> None:
    """Copy finished present_to_user task rows into the presentations table."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"task_records", "tool_records"} <= tables:
        return
    rows = conn.execute(
        "SELECT t.task_id, t.started_at, t.run_dir, tr.params"
        " FROM task_records t JOIN tool_records tr ON tr.task_id = t.task_id"
        " WHERE tr.tool_name = ? AND t.status = 'completed'",
        (_PRESENT_TO_USER,),
    ).fetchall()
    existing = {r[0] for r in conn.execute("SELECT id FROM presentations")}
    for task_id, started_at, run_dir, params_json in rows:
        url_path = _result_url_path(os.path.join(data_dir, run_dir, "result"))
        if url_path is None:
            continue
        if url_path.startswith("/view/presentation/"):
            rid = url_path.rsplit("/", 1)[1]
            files = _params_files(params_json)
        else:
            rid = task_id
            resource_path = url_path[len("/view/"):] if url_path.startswith("/view/") else url_path.lstrip("/")
            files = [unquote(resource_path)]
        if rid in existing:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO presentations (id, presented_at, url_path, files, task_id)"
            " VALUES (?, ?, ?, ?, ?)",
            (rid, started_at, url_path, json.dumps(files, ensure_ascii=False), task_id),
        )
        existing.add(rid)


def _move_presentations_folder() -> None:
    """Move the handoffs content folder into the presentations folder."""
    root = resources_dir()
    src = os.path.join(root, "handoffs")
    dst = os.path.join(root, "presentations")
    if not os.path.isdir(src):
        return
    if os.path.isdir(dst):
        for name in sorted(os.listdir(src)):
            s = os.path.join(src, name)
            d = os.path.join(dst, name)
            if os.path.exists(d):
                _logger.warning("presentation folder collision for '%s'; keeping existing", name)
                if os.path.isdir(s):
                    shutil.rmtree(s)
                else:
                    os.remove(s)
                continue
            shutil.move(s, d)
    elif os.path.exists(dst):
        _logger.warning("cannot move handoffs into %s because it is not a folder; leaving handoffs in place", dst)
        return
    else:
        os.rename(src, dst)
        return
    if not os.listdir(src):
        os.rmdir(src)


def _migrate_presentations(data_dir: str, db_path: str) -> None:
    """Migrate handoff records and task rows into the presentations table."""
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path, timeout=10, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        conn.executescript(_PRESENTATIONS_DDL)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(presentations)")}
        if "title" not in cols:
            conn.execute("ALTER TABLE presentations ADD COLUMN title TEXT")
        _migrate_legacy_records(data_dir, conn)
        _backfill_task_rows(data_dir, conn)
    finally:
        conn.close()
    _move_presentations_folder()


def _migrate_agent_records(_data_dir: str, db_path: str) -> None:
    """Create the agent_records table when it is missing."""
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path, timeout=10, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        conn.executescript(_AGENT_RECORDS_DDL)
    finally:
        conn.close()


def _migrate_tool_records_render_type(_data_dir: str, db_path: str) -> None:
    """Add the tool-declared render kind column when it is missing."""
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path, timeout=10, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        table_names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "tool_records" not in table_names:
            return
        cols = {r[1] for r in conn.execute("PRAGMA table_info(tool_records)")}
        if "render_type" not in cols:
            conn.execute("ALTER TABLE tool_records ADD COLUMN render_type TEXT")
    finally:
        conn.close()


def migrate_all_sync(data_dir: str) -> None:
    """Bring the data folder up to date, without an event loop.

    Only one process does this at a time. This is for code that runs outside
    the server, such as the garbage collector command. Never call it on the
    event loop, use migrate_all there.

    Args:
        data_dir: Folder holding the records file and the run folders.
    """
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, _DB_NAME)
    legacy_path = os.path.join(data_dir, _LEGACY_DB_NAME)
    lock_path = os.path.join(data_dir, _LOCK_NAME)
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            _ensure_database(db_path, legacy_path)
            _migrate_schema(db_path)
            _migrate_runs_dir(data_dir, db_path)
            _migrate_presentations(data_dir, db_path)
            _migrate_agent_records(data_dir, db_path)
            _migrate_tool_records_render_type(data_dir, db_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


async def migrate_all(data_dir: str) -> None:
    """Bring the data folder up to date.

    Only one process does this at a time.

    Args:
        data_dir: Folder holding the records file and the run folders.
    """
    await asyncio.to_thread(migrate_all_sync, data_dir)
