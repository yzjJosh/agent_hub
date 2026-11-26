"""SQLite record stores and task queries for runs, tools, agents, presentations, and context errors."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import aiosqlite

from app.lib.db import migration
from app.lib.shared import paths

__all__ = [
    "CONTEXT_ERROR_WINDOW_DAYS",
    "PRESENTATION_WINDOW_DAYS",
    "Presentation",
    "RUNS_DIR",
    "StoreFingerprint",
    "TaskPage",
    "TaskRecord",
    "TaskView",
    "ToolStats",
    "agent_store",
    "context_error_store",
    "count_tasks",
    "find_task_view_by_prefix",
    "get_store_fingerprint",
    "get_task_view",
    "init_store",
    "is_store_open",
    "list_ancestor_views",
    "list_finished_since_views",
    "list_running_views",
    "list_started_since_views",
    "list_subtask_views",
    "list_tool_stats",
    "list_tool_tasks",
    "presentation_store",
    "task_store",
    "tool_store",
]

_DB_NAME = "records.db"
_BUSY_TIMEOUT_MILLISECONDS = 5000
_SWEEP_BUSY_TIMEOUT_MILLISECONDS = 10000
_SWEEP_CONNECT_TIMEOUT_SECONDS = 10
_MAX_ANCESTOR_DEPTH = 64

RUNS_DIR = "runs"

PRESENTATION_WINDOW_DAYS = 30
CONTEXT_ERROR_WINDOW_DAYS = 30

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
CREATE TABLE IF NOT EXISTS context_error_cache (
  normalized_error    TEXT PRIMARY KEY,
  is_context_overflow INTEGER NOT NULL,
  classified_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_context_error_at ON context_error_cache(classified_at);
"""

_VIEW_SELECT = (
    "SELECT t.task_id, t.parent_task_id, t.status, t.error, t.started_at, t.ended_at, t.run_dir,"
    " tr.tool_name, tr.params, tr.render_type,"
    " ar.name AS agent_name"
    " FROM task_records t"
    " LEFT JOIN tool_records tr ON tr.task_id = t.task_id"
    " LEFT JOIN agent_records ar ON ar.task_id = t.task_id"
)


@dataclass(frozen=True)
class Presentation:
    """One saved presentation record and its file list."""

    id: str
    presented_at: str
    url_path: str
    files: list[str]
    task_id: str | None
    title: str | None


@dataclass(frozen=True)
class TaskRecord:
    """One persisted task record."""

    task_id: str
    parent_task_id: str | None
    status: str
    error: str | None
    started_at: str
    ended_at: str | None
    run_dir: str


@dataclass(frozen=True)
class TaskView(TaskRecord):
    """One task record joined with its optional tool and agent details."""

    tool_name: str | None
    params: dict | None
    agent_name: str | None
    render_type: str | None


@dataclass(frozen=True)
class StoreFingerprint:
    """Latest task and presentation markers for change polling."""

    latest_started_at: str | None
    latest_ended_at: str | None
    running_count: int
    latest_presented_at: str | None


@dataclass(frozen=True)
class TaskPage:
    """One page of task views and the cursor for the next page."""

    tasks: list[TaskView]
    next_cursor: str | None


@dataclass(frozen=True)
class ToolStats:
    """Run counts and activity timestamps for one tool."""

    tool_name: str
    running: int
    completed: int
    failed: int
    cancelled: int
    last_activity: str | None
    running_since: str | None


@dataclass(frozen=True)
class _ToolRecord:
    task_id: str
    tool_name: str
    cache_key: str | None
    params: dict


class _StoreState:
    def __init__(self) -> None:
        self.db_path: str = os.path.join(paths.data_dir(), _DB_NAME)
        self.connection: aiosqlite.Connection | None = None

    @property
    def data_dir(self) -> str:
        return os.path.dirname(self.db_path)

    @property
    def is_open(self) -> bool:
        return self.connection is not None

    def require_connection(self) -> aiosqlite.Connection:
        if self.connection is None:
            raise RuntimeError("record store is not open")
        return self.connection

    async def open(self) -> None:
        if self.connection is not None:
            return
        data_dir = self.data_dir
        await asyncio.to_thread(os.makedirs, data_dir, exist_ok=True)
        await migration.migrate_all(data_dir)
        connection = await aiosqlite.connect(self.db_path)
        try:
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MILLISECONDS}")
            await connection.executescript(_SCHEMA)
            await connection.commit()
        except BaseException:
            await connection.close()
            raise
        self.connection = connection

    async def close(self) -> None:
        if self.connection is not None:
            await self.connection.close()
            self.connection = None


def _sweep_table_sync(
    table: str, key_column: str, timestamp_column: str, days: int, data_dir: str
) -> list[str]:
    if not (table.isidentifier() and key_column.isidentifier() and timestamp_column.isidentifier()):
        raise ValueError("table, key_column, and timestamp_column must be valid SQL identifiers")
    db_path = os.path.join(data_dir, _DB_NAME)
    if not os.path.exists(db_path):
        return []
    connection = sqlite3.connect(db_path, timeout=_SWEEP_CONNECT_TIMEOUT_SECONDS)
    try:
        connection.execute(f"PRAGMA busy_timeout={_SWEEP_BUSY_TIMEOUT_MILLISECONDS}")
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if table_exists is None:
            return []
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        stale_rows = connection.execute(
            f"SELECT {key_column} FROM {table} WHERE {timestamp_column} < ?", (cutoff,)
        ).fetchall()
        keys = [row[0] for row in stale_rows]
        if keys:
            connection.executemany(f"DELETE FROM {table} WHERE {key_column} = ?", [(key,) for key in keys])
            connection.commit()
        return keys
    finally:
        connection.close()


def _build_task_record(row: aiosqlite.Row) -> TaskRecord:
    return TaskRecord(
        task_id=row["task_id"],
        parent_task_id=row["parent_task_id"],
        status=row["status"],
        error=row["error"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        run_dir=row["run_dir"],
    )


def _build_tool_record(row: aiosqlite.Row) -> _ToolRecord:
    return _ToolRecord(
        task_id=row["task_id"],
        tool_name=row["tool_name"],
        cache_key=row["cache_key"],
        params=json.loads(row["params"]),
    )


def _build_presentation(row: aiosqlite.Row) -> Presentation:
    return Presentation(
        id=row["id"],
        presented_at=row["presented_at"],
        url_path=row["url_path"],
        files=json.loads(row["files"]),
        task_id=row["task_id"],
        title=row["title"],
    )


def _build_task_view(row: aiosqlite.Row) -> TaskView:
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
    def __init__(self, state: _StoreState) -> None:
        self._state = state

    @property
    def db_path(self) -> str:
        return self._state.db_path

    @property
    def data_dir(self) -> str:
        return self._state.data_dir

    async def open(self) -> None:
        await self._state.open()

    async def close(self) -> None:
        await self._state.close()


class _TaskRecordStore(_SharedRecordStore):
    """Reads and writes task records. One row per task."""

    async def insert_running(self, task_id: str, parent_task_id: str | None, started_at: str, run_dir: str) -> None:
        db = self._state.require_connection()
        await db.execute(
            "INSERT INTO task_records (task_id, parent_task_id, status, started_at, run_dir)"
            " VALUES (?, ?, 'running', ?, ?)",
            (task_id, parent_task_id, started_at, run_dir),
        )
        await db.commit()

    async def finish(
        self,
        task_id: str,
        status: str,
        *,
        error: str | None = None,
        ended_at: str | None = None,
    ) -> None:
        db = self._state.require_connection()
        await db.execute(
            "UPDATE task_records SET status = ?, error = ?, ended_at = ? WHERE task_id = ?",
            (status, error, ended_at, task_id),
        )
        await db.commit()

    async def delete(self, task_id: str) -> None:
        db = self._state.require_connection()
        await db.execute("DELETE FROM task_records WHERE task_id = ?", (task_id,))
        await db.commit()

    async def get(self, task_id: str) -> TaskRecord | None:
        db = self._state.require_connection()
        cursor = await db.execute("SELECT * FROM task_records WHERE task_id = ?", (task_id,))
        row = await cursor.fetchone()
        return _build_task_record(row) if row is not None else None

    async def list_subtasks(self, parent_task_id: str) -> list[TaskRecord]:
        db = self._state.require_connection()
        cursor = await db.execute(
            "SELECT * FROM task_records WHERE parent_task_id = ? ORDER BY started_at, rowid",
            (parent_task_id,),
        )
        return [_build_task_record(row) for row in await cursor.fetchall()]

    async def list(self) -> list[TaskRecord]:
        db = self._state.require_connection()
        cursor = await db.execute("SELECT * FROM task_records ORDER BY started_at")
        return [_build_task_record(row) for row in await cursor.fetchall()]

    async def mark_orphans_failed(self, error: str) -> int:
        db = self._state.require_connection()
        cursor = await db.execute(
            "UPDATE task_records SET status = 'failed', error = ?, ended_at = started_at"
            " WHERE status = 'running'",
            (error,),
        )
        await db.commit()
        return cursor.rowcount


class _ToolRecordStore(_SharedRecordStore):
    """Reads and writes tool records. One row per task."""

    async def insert(self, task_id: str, tool_name: str, cache_key: str | None, params: dict) -> None:
        db = self._state.require_connection()
        await db.execute(
            "INSERT INTO tool_records (task_id, tool_name, cache_key, params) VALUES (?, ?, ?, ?)",
            (task_id, tool_name, cache_key, json.dumps(params, ensure_ascii=False)),
        )
        await db.commit()

    async def get(self, task_id: str) -> _ToolRecord | None:
        db = self._state.require_connection()
        cursor = await db.execute("SELECT * FROM tool_records WHERE task_id = ?", (task_id,))
        row = await cursor.fetchone()
        return _build_tool_record(row) if row is not None else None

    async def find_cache(self, tool_name: str, cache_key: str, min_started_at: str) -> TaskRecord | None:
        db = self._state.require_connection()
        cursor = await db.execute(
            "SELECT t.task_id, t.parent_task_id, t.status, t.error, t.started_at, t.ended_at, t.run_dir"
            " FROM task_records t JOIN tool_records tr ON tr.task_id = t.task_id"
            " WHERE tr.tool_name = ? AND tr.cache_key = ? AND t.status = 'completed'"
            " AND t.started_at >= ? ORDER BY t.started_at DESC LIMIT 1",
            (tool_name, cache_key, min_started_at),
        )
        row = await cursor.fetchone()
        return _build_task_record(row) if row is not None else None

    async def set_render_type(self, task_id: str, token: str) -> None:
        db = self._state.require_connection()
        await db.execute(
            "UPDATE tool_records SET render_type = ? WHERE task_id = ?",
            (token, task_id),
        )
        await db.commit()


class _AgentRecordStore(_SharedRecordStore):
    """Writes agent records. One row per agent run."""

    async def insert(self, task_id: str, name: str) -> None:
        db = self._state.require_connection()
        await db.execute(
            "INSERT INTO agent_records (task_id, name) VALUES (?, ?)",
            (task_id, name),
        )
        await db.commit()


class _PresentationRecordStore(_SharedRecordStore):
    """Reads and writes presentation records. One row per presentation."""

    async def insert(
        self,
        presentation_id: str,
        presented_at: str,
        url_path: str,
        files: list[str],
        task_id: str | None = None,
        title: str | None = None,
    ) -> None:
        db = self._state.require_connection()
        await db.execute(
            "INSERT OR IGNORE INTO presentations (id, presented_at, url_path, files, task_id, title)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (presentation_id, presented_at, url_path, json.dumps(files, ensure_ascii=False), task_id, title),
        )
        await db.commit()

    async def get(self, presentation_id: str) -> Presentation | None:
        db = self._state.require_connection()
        cursor = await db.execute("SELECT * FROM presentations WHERE id = ?", (presentation_id,))
        row = await cursor.fetchone()
        return _build_presentation(row) if row is not None else None

    async def list_newest(self, window_days: int = PRESENTATION_WINDOW_DAYS) -> list[Presentation]:
        db = self._state.require_connection()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        cursor = await db.execute(
            "SELECT * FROM presentations WHERE presented_at >= ?"
            " ORDER BY presented_at DESC, id DESC",
            (cutoff,),
        )
        return [_build_presentation(row) for row in await cursor.fetchall()]

    async def get_newest_presented_at(self) -> str | None:
        db = self._state.require_connection()
        cursor = await db.execute("SELECT MAX(presented_at) FROM presentations")
        row = await cursor.fetchone()
        return row[0] if row is not None else None

    async def delete_older(self, window_days: int = PRESENTATION_WINDOW_DAYS) -> int:
        db = self._state.require_connection()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        cursor = await db.execute("DELETE FROM presentations WHERE presented_at < ?", (cutoff,))
        await db.commit()
        return cursor.rowcount

    def sweep_sync(
        self,
        days: int = PRESENTATION_WINDOW_DAYS,
        data_dir: str | None = None,
    ) -> list[str]:
        """Remove presentation rows older than the day window."""
        target_dir = data_dir if data_dir is not None else self._state.data_dir
        return _sweep_table_sync("presentations", "id", "presented_at", days, target_dir)


class _ContextErrorRecordStore(_SharedRecordStore):
    """Reads and writes cached context error rows."""

    async def get(
        self,
        normalized_error: str,
        window_days: int = CONTEXT_ERROR_WINDOW_DAYS,
    ) -> bool | None:
        """Return the cached overflow verdict when fresh, or None."""
        db = self._state.require_connection()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        cursor = await db.execute(
            "SELECT is_context_overflow FROM context_error_cache"
            " WHERE normalized_error = ? AND classified_at >= ?",
            (normalized_error, cutoff),
        )
        row = await cursor.fetchone()
        return bool(row[0]) if row is not None else None

    async def put(
        self,
        normalized_error: str,
        is_context_overflow: bool,
        classified_at: str,
    ) -> None:
        """Save or update a cached overflow verdict."""
        db = self._state.require_connection()
        await db.execute(
            "INSERT INTO context_error_cache (normalized_error, is_context_overflow, classified_at)"
            " VALUES (?, ?, ?)"
            " ON CONFLICT(normalized_error) DO UPDATE SET"
            " is_context_overflow = excluded.is_context_overflow,"
            " classified_at = excluded.classified_at",
            (normalized_error, int(is_context_overflow), classified_at),
        )
        await db.commit()

    def sweep_sync(
        self,
        days: int = CONTEXT_ERROR_WINDOW_DAYS,
        data_dir: str | None = None,
    ) -> list[str]:
        """Remove cached context error rows older than the day window."""
        target_dir = data_dir if data_dir is not None else self._state.data_dir
        return _sweep_table_sync(
            "context_error_cache",
            "normalized_error",
            "classified_at",
            days,
            target_dir,
        )


_state = _StoreState()

task_store = _TaskRecordStore(_state)
tool_store = _ToolRecordStore(_state)
presentation_store = _PresentationRecordStore(_state)
agent_store = _AgentRecordStore(_state)
context_error_store = _ContextErrorRecordStore(_state)


def init_store(db_path: str | None = None) -> _TaskRecordStore:
    """Point the stores at a database file while keeping the connection closed.

    Args:
        db_path: Database file path, or None to use records.db in the data folder.

    Returns:
        The task store.
    """
    _state.db_path = db_path if db_path is not None else os.path.join(paths.data_dir(), _DB_NAME)
    return task_store


def is_store_open() -> bool:
    """Return True when the shared database connection is open."""
    return _state.is_open


async def list_tool_stats() -> list[ToolStats]:
    """List run counts and activity timestamps for each tool."""
    db = _state.require_connection()
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
        ToolStats(
            tool_name=row["tool_name"],
            running=row["running"] or 0,
            completed=row["completed"] or 0,
            failed=row["failed"] or 0,
            cancelled=row["cancelled"] or 0,
            last_activity=row["last_activity"],
            running_since=row["running_since"],
        )
        for row in await cursor.fetchall()
    ]


async def list_tool_tasks(tool_name: str, limit: int, cursor: str | None) -> TaskPage:
    """List one page of tasks for a tool, newest first.

    Args:
        tool_name: Name of the tool.
        limit: Maximum number of tasks on the page.
        cursor: Position marker from the previous page, or None to start at the newest.

    Returns:
        The task page and the cursor for the next page, or None on the last page.
    """
    db = _state.require_connection()
    if cursor is None:
        where_clause = "tr.tool_name = ?"
        query_args: list[str | int] = [tool_name]
    else:
        cursor_started_at, cursor_task_id = cursor.split("|", 1)
        where_clause = "tr.tool_name = ? AND (t.started_at, t.task_id) < (?, ?)"
        query_args = [tool_name, cursor_started_at, cursor_task_id]
    query_args.append(limit + 1)
    query_cursor = await db.execute(
        f"{_VIEW_SELECT} WHERE {where_clause} ORDER BY t.started_at DESC, t.task_id DESC LIMIT ?",
        query_args,
    )
    rows = list(await query_cursor.fetchall())
    has_more = len(rows) > limit
    views = [_build_task_view(row) for row in rows[:limit]]
    next_cursor = f"{views[-1].started_at}|{views[-1].task_id}" if (has_more and views) else None
    return TaskPage(tasks=views, next_cursor=next_cursor)


async def get_task_view(task_id: str) -> TaskView | None:
    """Get the full view of one task, or None when the id is unknown."""
    db = _state.require_connection()
    cursor = await db.execute(f"{_VIEW_SELECT} WHERE t.task_id = ?", (task_id,))
    row = await cursor.fetchone()
    return _build_task_view(row) if row is not None else None


async def list_subtask_views(parent_task_id: str) -> list[TaskView]:
    """List full views of a task's direct subtasks, oldest first."""
    db = _state.require_connection()
    cursor = await db.execute(
        f"{_VIEW_SELECT} WHERE t.parent_task_id = ? ORDER BY t.started_at, t.task_id",
        (parent_task_id,),
    )
    return [_build_task_view(row) for row in await cursor.fetchall()]


async def find_task_view_by_prefix(prefix: str) -> TaskView | None:
    """Find the task whose id starts with the given prefix.

    Args:
        prefix: A task id or a prefix of one.

    Returns:
        The view when one task matches, or None when zero or several match.
    """
    db = _state.require_connection()
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    cursor = await db.execute(f"{_VIEW_SELECT} WHERE t.task_id LIKE ? ESCAPE '\\'", (escaped + "%",))
    rows = list(await cursor.fetchmany(2))
    if len(rows) != 1:
        return None
    return _build_task_view(rows[0])


async def list_ancestor_views(task_id: str) -> list[TaskView]:
    """List a task's ancestors from root down to direct parent.

    Args:
        task_id: The task whose ancestors to list.

    Returns:
        The ancestors from the root down to the direct parent, or an empty
        list when the task is top-level or unknown.
    """
    view = await get_task_view(task_id)
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
        parent = await get_task_view(parent_id)
        if parent is None:
            break
        chain.append(parent)
        current = parent
    chain.reverse()
    return chain


async def get_store_fingerprint() -> StoreFingerprint:
    """Get the latest task and presentation activity markers."""
    db = _state.require_connection()
    task_cursor = await db.execute(
        "SELECT MAX(started_at), MAX(ended_at), SUM(status = 'running') FROM task_records"
    )
    task_row = await task_cursor.fetchone()
    presentation_cursor = await db.execute("SELECT MAX(presented_at) FROM presentations")
    presentation_row = await presentation_cursor.fetchone()
    return StoreFingerprint(
        latest_started_at=task_row[0] if task_row is not None else None,
        latest_ended_at=task_row[1] if task_row is not None else None,
        running_count=(task_row[2] or 0) if task_row is not None else 0,
        latest_presented_at=presentation_row[0] if presentation_row is not None else None,
    )


async def list_running_views() -> list[TaskView]:
    """List all running tasks, oldest first."""
    db = _state.require_connection()
    cursor = await db.execute(f"{_VIEW_SELECT} WHERE t.status = 'running' ORDER BY t.started_at, t.task_id")
    return [_build_task_view(row) for row in await cursor.fetchall()]


async def list_started_since_views(since_iso: str) -> list[TaskView]:
    """List tasks that started after the given ISO timestamp, oldest first."""
    db = _state.require_connection()
    cursor = await db.execute(
        f"{_VIEW_SELECT} WHERE t.started_at > ? ORDER BY t.started_at, t.task_id",
        (since_iso,),
    )
    return [_build_task_view(row) for row in await cursor.fetchall()]


async def list_finished_since_views(since_iso: str) -> list[TaskView]:
    """List tasks that finished after the given ISO timestamp, oldest first."""
    db = _state.require_connection()
    cursor = await db.execute(
        f"{_VIEW_SELECT} WHERE t.status != 'running' AND t.ended_at IS NOT NULL AND t.ended_at > ?"
        " ORDER BY t.ended_at, t.task_id",
        (since_iso,),
    )
    return [_build_task_view(row) for row in await cursor.fetchall()]


async def count_tasks() -> int:
    """Count all persisted task records."""
    db = _state.require_connection()
    cursor = await db.execute("SELECT COUNT(*) FROM task_records")
    row = await cursor.fetchone()
    return int(row[0]) if row is not None else 0
