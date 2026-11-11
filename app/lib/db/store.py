from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import aiosqlite

from app.lib.db import migration
from app.lib.shared import paths

__all__ = [
    "PRESENTATION_WINDOW_DAYS",
    "Presentation",
    "RUNS_DIR",
    "TaskRecord",
    "TaskView",
    "agent_store",
    "ancestor_views",
    "fingerprint",
    "finished_since_views",
    "init_store",
    "keyset_tasks_for_tool",
    "presentation_store",
    "running_views",
    "started_since_views",
    "store_is_open",
    "subtask_views",
    "task_store",
    "task_view",
    "task_view_by_prefix",
    "total_count",
    "tool_stats",
    "tool_store",
]

_DATA_DIR = paths.data_dir()
_DB_PATH = os.path.join(_DATA_DIR, "records.db")

RUNS_DIR = "runs"

PRESENTATION_WINDOW_DAYS = 30


@dataclass
class Presentation:
    id: str
    presented_at: str
    url_path: str
    files: list[str]
    task_id: str | None
    title: str | None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_records (
  task_id        TEXT PRIMARY KEY,
  parent_task_id TEXT REFERENCES task_records(task_id),
  status         TEXT NOT NULL,
  error          TEXT,
  started_at     TEXT NOT NULL,
  ended_at       TEXT,
  run_dir        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_started ON task_records(started_at);
CREATE INDEX IF NOT EXISTS idx_task_parent  ON task_records(parent_task_id);
CREATE TABLE IF NOT EXISTS tool_records (
  task_id   TEXT PRIMARY KEY,
  tool_name TEXT NOT NULL,
  cache_key TEXT,
  params    TEXT NOT NULL,
  render_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_cache ON tool_records(tool_name, cache_key);
CREATE TABLE IF NOT EXISTS agent_records (
  task_id  TEXT PRIMARY KEY REFERENCES task_records(task_id),
  name     TEXT NOT NULL
);
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

_conn: aiosqlite.Connection | None = None


def _db() -> aiosqlite.Connection:
    if _conn is None:
        raise RuntimeError("record store is not open")
    return _conn


async def _open_shared() -> None:
    global _conn
    if _conn is not None:
        return
    data_dir = os.path.dirname(_DB_PATH)
    await asyncio.to_thread(os.makedirs, data_dir, exist_ok=True)
    await migration.migrate_all(data_dir)
    _conn = await aiosqlite.connect(_DB_PATH)
    await _conn.execute("PRAGMA journal_mode=WAL")
    await _conn.execute("PRAGMA busy_timeout=5000")
    await _conn.executescript(_SCHEMA)
    await _conn.commit()


async def _close_shared() -> None:
    global _conn
    if _conn is not None:
        await _conn.close()
        _conn = None


@dataclass
class TaskRecord:
    task_id: str
    parent_task_id: str | None
    status: str
    error: str | None
    started_at: str
    ended_at: str | None
    run_dir: str


@dataclass
class TaskView(TaskRecord):
    tool_name: str | None
    params: dict | None
    agent_name: str | None
    render_type: str | None


@dataclass
class _ToolRecord:
    task_id: str
    tool_name: str
    cache_key: str | None
    params: dict


def _task_row(row: aiosqlite.Row) -> TaskRecord:
    return TaskRecord(
        task_id=row["task_id"],
        parent_task_id=row["parent_task_id"],
        status=row["status"],
        error=row["error"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        run_dir=row["run_dir"],
    )


def _tool_row(row: aiosqlite.Row) -> _ToolRecord:
    return _ToolRecord(
        task_id=row["task_id"],
        tool_name=row["tool_name"],
        cache_key=row["cache_key"],
        params=json.loads(row["params"]),
    )


def _presentation_row(row: aiosqlite.Row) -> Presentation:
    return Presentation(
        id=row["id"],
        presented_at=row["presented_at"],
        url_path=row["url_path"],
        files=json.loads(row["files"]),
        task_id=row["task_id"],
        title=row["title"],
    )


_VIEW_SELECT = (
    "SELECT t.task_id, t.parent_task_id, t.status, t.error, t.started_at, t.ended_at, t.run_dir,"
    " tr.tool_name, tr.params, tr.render_type,"
    " ar.name AS agent_name"
    " FROM task_records t"
    " LEFT JOIN tool_records tr ON tr.task_id = t.task_id"
    " LEFT JOIN agent_records ar ON ar.task_id = t.task_id"
)


def _view_row(row: aiosqlite.Row) -> TaskView:
    return TaskView(
        task_id=row["task_id"],
        parent_task_id=row["parent_task_id"],
        status=row["status"],
        error=row["error"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        run_dir=row["run_dir"],
        tool_name=row["tool_name"],
        params=json.loads(row["params"]) if row["params"] is not None else None,
        agent_name=row["agent_name"],
        render_type=row["render_type"],
    )


class _SharedRecordStore:
    async def open(self) -> None:
        await _open_shared()

    async def close(self) -> None:
        await _close_shared()


class _TaskRecordStore(_SharedRecordStore):
    """Reads and writes task records. One row per task."""

    @property
    def data_dir(self) -> str:
        return os.path.dirname(_DB_PATH)

    async def insert_running(self, task_id: str, parent_task_id: str | None, started_at: str, run_dir: str) -> None:
        db = _db()
        await db.execute(
            "INSERT INTO task_records (task_id, parent_task_id, status, started_at, run_dir)"
            " VALUES (?, ?, 'running', ?, ?)",
            (task_id, parent_task_id, started_at, run_dir),
        )
        await db.commit()

    async def finish(self, task_id: str, status: str, *, error: str | None = None, ended_at: str | None = None) -> None:
        db = _db()
        await db.execute(
            "UPDATE task_records SET status = ?, error = ?, ended_at = ? WHERE task_id = ?",
            (status, error, ended_at, task_id),
        )
        await db.commit()

    async def delete(self, task_id: str) -> None:
        db = _db()
        await db.execute("DELETE FROM task_records WHERE task_id = ?", (task_id,))
        await db.commit()

    async def get(self, task_id: str) -> TaskRecord | None:
        db = _db()
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM task_records WHERE task_id = ?", (task_id,))
        row = await cursor.fetchone()
        return _task_row(row) if row is not None else None

    async def list(self) -> list[TaskRecord]:
        db = _db()
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM task_records ORDER BY started_at")
        return [_task_row(r) for r in await cursor.fetchall()]

    async def list_subtasks(self, parent_task_id: str) -> list[TaskRecord]:
        db = _db()
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM task_records WHERE parent_task_id = ? ORDER BY started_at, rowid",
            (parent_task_id,),
        )
        return [_task_row(r) for r in await cursor.fetchall()]

    async def mark_orphans_failed(self, error: str) -> int:
        db = _db()
        cursor = await db.execute(
            "UPDATE task_records SET status = 'failed', error = ?, ended_at = started_at"
            " WHERE status = 'running'",
            (error,),
        )
        await db.commit()
        return cursor.rowcount


class _ToolRecordStore(_SharedRecordStore):
    """Reads and writes tool records. One row per task."""

    @property
    def data_dir(self) -> str:
        return os.path.dirname(_DB_PATH)

    async def insert(self, task_id: str, tool_name: str, cache_key: str | None, params: dict) -> None:
        db = _db()
        await db.execute(
            "INSERT INTO tool_records (task_id, tool_name, cache_key, params) VALUES (?, ?, ?, ?)",
            (task_id, tool_name, cache_key, json.dumps(params, ensure_ascii=False)),
        )
        await db.commit()

    async def get(self, task_id: str) -> _ToolRecord | None:
        db = _db()
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM tool_records WHERE task_id = ?", (task_id,))
        row = await cursor.fetchone()
        return _tool_row(row) if row is not None else None

    async def find_cache(self, tool_name: str, cache_key: str, min_started_at: str) -> TaskRecord | None:
        db = _db()
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT t.task_id, t.parent_task_id, t.status, t.error, t.started_at, t.ended_at, t.run_dir"
            " FROM task_records t JOIN tool_records tr ON tr.task_id = t.task_id"
            " WHERE tr.tool_name = ? AND tr.cache_key = ? AND t.status = 'completed'"
            " AND t.started_at >= ? ORDER BY t.started_at DESC LIMIT 1",
            (tool_name, cache_key, min_started_at),
        )
        row = await cursor.fetchone()
        return _task_row(row) if row is not None else None

    async def set_render_type(self, task_id: str, token: str) -> None:
        db = _db()
        await db.execute(
            "UPDATE tool_records SET render_type = ? WHERE task_id = ?",
            (token, task_id),
        )
        await db.commit()


class _AgentRecordStore(_SharedRecordStore):
    """Writes agent records. One row per agent run."""

    async def insert(self, task_id: str, name: str) -> None:
        db = _db()
        await db.execute(
            "INSERT INTO agent_records (task_id, name) VALUES (?, ?)",
            (task_id, name),
        )
        await db.commit()


class _PresentationRecordStore(_SharedRecordStore):
    """Reads and writes presentation records. One row per presentation."""

    async def insert(
        self,
        id: str,
        presented_at: str,
        url_path: str,
        files: list[str],
        task_id: str | None = None,
        title: str | None = None,
    ) -> None:
        db = _db()
        await db.execute(
            "INSERT OR IGNORE INTO presentations (id, presented_at, url_path, files, task_id, title)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (id, presented_at, url_path, json.dumps(files, ensure_ascii=False), task_id, title),
        )
        await db.commit()

    async def get(self, id: str) -> Presentation | None:
        db = _db()
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM presentations WHERE id = ?", (id,))
        row = await cursor.fetchone()
        return _presentation_row(row) if row is not None else None

    async def list_newest(self, window_days: int = PRESENTATION_WINDOW_DAYS) -> list[Presentation]:
        db = _db()
        db.row_factory = aiosqlite.Row
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        cursor = await db.execute(
            "SELECT * FROM presentations WHERE presented_at >= ?"
            " ORDER BY presented_at DESC, id DESC",
            (cutoff,),
        )
        return [_presentation_row(r) for r in await cursor.fetchall()]

    async def newest_at(self) -> str | None:
        db = _db()
        cursor = await db.execute("SELECT MAX(presented_at) FROM presentations")
        row = await cursor.fetchone()
        return row[0]

    async def delete_older(self, window_days: int = PRESENTATION_WINDOW_DAYS) -> int:
        db = _db()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        cursor = await db.execute("DELETE FROM presentations WHERE presented_at < ?", (cutoff,))
        await db.commit()
        return cursor.rowcount


task_store = _TaskRecordStore()
tool_store = _ToolRecordStore()
presentation_store = _PresentationRecordStore()
agent_store = _AgentRecordStore()


def init_store(db_path: str | None = None) -> _TaskRecordStore:
    """Point the stores at a database file. The connection stays closed.

    Args:
        db_path: Database file path. Defaults to records.db in the data folder.

    Returns:
        The task store.
    """
    global _DB_PATH, task_store, tool_store, presentation_store, agent_store
    if db_path is not None:
        _DB_PATH = db_path
    task_store = _TaskRecordStore()
    tool_store = _ToolRecordStore()
    presentation_store = _PresentationRecordStore()
    agent_store = _AgentRecordStore()
    return task_store


def store_is_open() -> bool:
    """True when the shared database connection is open."""
    return _conn is not None


async def tool_stats() -> list[dict]:
    """Run counts by status, the newest start time, and the earliest running start, one dict per tool."""
    db = _db()
    db.row_factory = aiosqlite.Row
    cursor = await db.execute(
        "SELECT tr.tool_name,"
        " SUM(t.status = 'running') AS running, SUM(t.status = 'completed') AS completed,"
        " SUM(t.status = 'failed') AS failed, SUM(t.status = 'cancelled') AS cancelled,"
        " MAX(t.started_at) AS last_activity,"
        " MIN(CASE WHEN t.status = 'running' THEN t.started_at END) AS running_since"
        " FROM tool_records tr JOIN task_records t ON t.task_id = tr.task_id"
        " GROUP BY tr.tool_name"
    )
    return [
        {
            "tool_name": row["tool_name"],
            "running": row["running"] or 0,
            "completed": row["completed"] or 0,
            "failed": row["failed"] or 0,
            "cancelled": row["cancelled"] or 0,
            "last_activity": row["last_activity"],
            "running_since": row["running_since"],
        }
        for row in await cursor.fetchall()
    ]


async def keyset_tasks_for_tool(tool_name: str, limit: int, cursor: str | None) -> tuple[list[TaskView], str | None]:
    """One page of a tool's tasks, newest first.

    Args:
        tool_name: Name of the tool.
        limit: How many tasks the page holds.
        cursor: Where to pick up from, or None to start at the newest.

    Returns:
        The page of tasks and the cursor for the next page, or None when the page is the last.
    """
    db = _db()
    db.row_factory = aiosqlite.Row
    if cursor is None:
        where = "tr.tool_name = ?"
        args: list = [tool_name]
    else:
        c_started, c_task_id = cursor.split("|", 1)
        where = "tr.tool_name = ? AND (t.started_at, t.task_id) < (?, ?)"
        args = [tool_name, c_started, c_task_id]
    args.append(limit + 1)
    cursor_obj = await db.execute(
        f"{_VIEW_SELECT} WHERE {where} ORDER BY t.started_at DESC, t.task_id DESC LIMIT ?", args
    )
    rows = await cursor_obj.fetchall()
    has_more = len(rows) > limit
    views = [_view_row(r) for r in rows[:limit]]
    next_cursor = f"{views[-1].started_at}|{views[-1].task_id}" if (has_more and views) else None
    return views, next_cursor


async def task_view(task_id: str) -> TaskView | None:
    """The full view of one task, or None when the id is unknown."""
    db = _db()
    db.row_factory = aiosqlite.Row
    cursor = await db.execute(f"{_VIEW_SELECT} WHERE t.task_id = ?", (task_id,))
    row = await cursor.fetchone()
    return _view_row(row) if row is not None else None


async def subtask_views(parent_task_id: str) -> list[TaskView]:
    """The full views of a task's direct subtasks, oldest first."""
    db = _db()
    db.row_factory = aiosqlite.Row
    cursor = await db.execute(
        f"{_VIEW_SELECT} WHERE t.parent_task_id = ? ORDER BY t.started_at, t.task_id", (parent_task_id,)
    )
    return [_view_row(r) for r in await cursor.fetchall()]


_MAX_ANCESTOR_DEPTH = 64


async def task_view_by_prefix(prefix: str) -> TaskView | None:
    """The full view of the task whose id starts with the prefix.

    Args:
        prefix: A task id or a prefix of one.

    Returns:
        The view when exactly one task matches, or None when zero or several
        match.
    """
    db = _db()
    db.row_factory = aiosqlite.Row
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    cursor = await db.execute(f"{_VIEW_SELECT} WHERE t.task_id LIKE ? ESCAPE '\\'", (escaped + "%",))
    rows = await cursor.fetchmany(2)
    if len(rows) != 1:
        return None
    return _view_row(rows[0])


async def ancestor_views(task_id: str) -> list[TaskView]:
    """The task's ancestor chain, the root run first and the direct parent last.

    The walk follows parent links and stops at a top-level task, a missing row,
    a cycle, or the depth cap.

    Args:
        task_id: The task whose ancestors to list.

    Returns:
        The ancestors from the root down to the direct parent, or an empty
        list when the task is top-level or unknown.
    """
    view = await task_view(task_id)
    if view is None or view.parent_task_id is None:
        return []
    chain: list[TaskView] = []
    seen: set[str] = {task_id}
    current = view
    while current.parent_task_id is not None and len(chain) < _MAX_ANCESTOR_DEPTH:
        parent_id = current.parent_task_id
        if parent_id in seen:
            break
        seen.add(parent_id)
        parent = await task_view(parent_id)
        if parent is None:
            break
        chain.append(parent)
        current = parent
    chain.reverse()
    return chain


async def fingerprint() -> tuple[str | None, str | None, int, str | None]:
    """The newest start time, the newest end time, the running count, and the newest presentation time."""
    db = _db()
    cursor = await db.execute(
        "SELECT MAX(started_at), MAX(ended_at), SUM(status = 'running') FROM task_records"
    )
    row = await cursor.fetchone()
    cursor2 = await db.execute("SELECT MAX(presented_at) FROM presentations")
    row2 = await cursor2.fetchone()
    return row[0], row[1], (row[2] or 0), row2[0]


async def running_views() -> list[TaskView]:
    """All running tasks, oldest first."""
    db = _db()
    db.row_factory = aiosqlite.Row
    cursor = await db.execute(f"{_VIEW_SELECT} WHERE t.status = 'running' ORDER BY t.started_at, t.task_id")
    return [_view_row(r) for r in await cursor.fetchall()]


async def started_since_views(since_gen: str) -> list[TaskView]:
    """Tasks that started after the given time, oldest first."""
    db = _db()
    db.row_factory = aiosqlite.Row
    cursor = await db.execute(
        f"{_VIEW_SELECT} WHERE t.started_at > ? ORDER BY t.started_at, t.task_id", (since_gen,)
    )
    return [_view_row(r) for r in await cursor.fetchall()]


async def finished_since_views(since_gen: str) -> list[TaskView]:
    """Tasks that finished after the given time, oldest first."""
    db = _db()
    db.row_factory = aiosqlite.Row
    cursor = await db.execute(
        f"{_VIEW_SELECT} WHERE t.status != 'running' AND t.ended_at IS NOT NULL AND t.ended_at > ?"
        " ORDER BY t.ended_at, t.task_id",
        (since_gen,),
    )
    return [_view_row(r) for r in await cursor.fetchall()]


async def total_count() -> int:
    """Total number of task records."""
    db = _db()
    cursor = await db.execute("SELECT COUNT(*) FROM task_records")
    return (await cursor.fetchone())[0]
