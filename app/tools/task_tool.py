from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import ClassVar, override

import aiofiles

import app.lib.db.store as store_module
from app.lib.db.store import TaskRecord
from app.lib.task import Task, TaskStatus, task_manager
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult, read_run_mcp_text
from app.lib.tool.truncation import DEFAULT_TOOL_OUTPUT_LEN_LIMIT, MIN_TOOL_OUTPUT_LEN_LIMIT

__all__ = ["task_tool"]

_NOT_FOUND = "No task found with id '{task_id}'."
_MIN_WAIT_TIMEOUT = 10.0


def _uuid(task_id: str) -> uuid.UUID | None:
    """Return the id as a UUID, or None when it is not one."""
    try:
        return uuid.UUID(task_id)
    except ValueError:
        return None


async def _read_window(path: str, offset: int, max_char: int) -> str:
    """Return part of a file's text, starting at offset."""
    async with aiofiles.open(path, "r", encoding="utf-8") as fh:
        content = await fh.read()
    return content[offset : offset + max_char]


async def _live(task_id: str) -> Task | None:
    """Return the running task with this id, or None when there is none."""
    live_id = _uuid(task_id)
    return await task_manager.get(live_id) if live_id else None


async def _await_terminal(live: Task, timeout: float | None = None) -> bool:
    """Wait for a running task to finish.

    Returns:
        True when the time ran out and the task is still running.
    """
    try:
        await asyncio.wait_for(live.result(), timeout)
    except TimeoutError:
        return live.status is TaskStatus.RUNNING
    except Exception:
        pass
    return False


def _still_running_message(task_id: str, started_at: str) -> str:
    """Return the message saying the task is still running."""
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds()
    return f"Task {task_id} is still running ({elapsed:.1f}s since {started_at}); repoll for the result."


async def _status(record: TaskRecord) -> str:
    """Return the status report for a task."""
    tool = await store_module.tool_store.get(record.task_id)
    subtasks = await store_module.task_store.list_subtasks(record.task_id)
    lines = [
        f"task_id: {record.task_id}",
        f"tool: {tool.tool_name if tool is not None else '-'}",
        f"status: {record.status}",
        f"started_at: {record.started_at}",
        f"ended_at: {record.ended_at or '-'}",
        f"subtasks: {', '.join(s.task_id for s in subtasks) or '-'}",
    ]
    if record.status in ("failed", "cancelled") and record.error:
        lines.append(f"error: {record.error}")
    return "\n".join(lines)


async def _result(
    record: TaskRecord, task_id: str, wait: bool, wait_timeout: float | None, offset: int, max_char: int
) -> str:
    """Return part of the task's result, or the error it ended with."""
    if wait:
        live = await _live(task_id)
        if live is not None and live.status is TaskStatus.RUNNING:
            if await _await_terminal(live, wait_timeout):
                return _still_running_message(task_id, record.started_at)
        record = await store_module.task_store.get(task_id)
        if record is None:
            return _NOT_FOUND.format(task_id=task_id)
    if record.status == "completed":
        run_path = os.path.join(store_module.task_store.data_dir, record.run_dir)
        text = await read_run_mcp_text(run_path)
        return text[offset : offset + max_char]
    if record.status in ("failed", "cancelled"):
        return record.error or f"Task {task_id} {record.status}."
    return f"Task {task_id} is still running."


async def _logs(record: TaskRecord, offset: int, max_char: int, tail: int | None) -> str:
    """Return part of the task's log, or its last lines."""
    log_path = os.path.join(store_module.task_store.data_dir, record.run_dir, "log")
    try:
        if tail is not None:
            async with aiofiles.open(log_path, "r", encoding="utf-8") as fh:
                content = await fh.read()
            lines = content.splitlines()
            return "\n".join(lines[-tail:]) if tail > 0 else ""
        return await _read_window(log_path, offset, max_char)
    except FileNotFoundError:
        return ""


async def _cancel(task_id: str, record: TaskRecord) -> str:
    """Cancel a running task and its subtasks, or say it already finished."""
    live = await _live(task_id)
    if live is not None and live.status is TaskStatus.RUNNING:
        await task_manager.cancel(task_id)
        record = await store_module.task_store.get(task_id)
        if record is None:
            return _NOT_FOUND.format(task_id=task_id)
        return f"Task {task_id} is now {record.status}."
    return f"Task {task_id} is already {record.status}."


class _TaskTool(Tool):
    name: ClassVar[str] = "task"
    tracked: ClassVar[bool] = False
    description: ClassVar[str] = (
        "Query or control a stateful tool run.\n"
        "\n"
        "Actions:\n"
        "  status   Show the run's status report.\n"
        "  result   Read the run's result.\n"
        "  logs     Read the run's log (streams while running).\n"
        "  cancel   Cancel a running task.\n"
        "\n"
        "Parameters not applicable to an action are ignored."
    )
    summary: ClassVar[str] = "Query or control a running tool task"
    params: ClassVar[tuple[Param, ...]] = (
        Param(
            "action",
            str,
            enum=("status", "result", "logs", "cancel"),
            description="Which action to run. Required by: all.",
        ),
        Param(
            "task_id",
            str,
            description=(
                "The run's id (from a tool's ack message or truncation note). "
                "Required by: all. Not used by: none."
            ),
        ),
        Param(
            "wait",
            bool,
            False,
            description=(
                "Block until completion before reading. Used by: result. "
                "Not used by: status, logs, cancel."
            ),
        ),
        Param(
            "wait_timeout",
            float | None,
            None,
            min=_MIN_WAIT_TIMEOUT,
            description=(
                "Maximum seconds to block before reporting the run still running. "
                "null = no limit. Used by: result, only when wait is true. Not used "
                "by: status, logs, cancel, or result when wait is false."
            ),
        ),
        Param(
            "offset",
            int,
            0,
            description=(
                "Character offset into the content. Used by: result, logs "
                "(when tail is not set). Not used by: status, cancel."
            ),
        ),
        Param(
            "max_char",
            int,
            DEFAULT_TOOL_OUTPUT_LEN_LIMIT,
            min=MIN_TOOL_OUTPUT_LEN_LIMIT,
            description=(
                "Maximum characters returned. Used by: result, logs (when tail is "
                "not set). Not used by: status, cancel."
            ),
        ),
        Param(
            "tail",
            int | None,
            None,
            description="Number of trailing log lines to return. Used by: logs only.\n    Not used by: status, result, cancel.",
        ),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        action = params["action"]
        task_id = params["task_id"]
        wait = params.get("wait", False)
        wait_timeout = params.get("wait_timeout")
        offset = params.get("offset", 0)
        max_char = params.get("max_char", DEFAULT_TOOL_OUTPUT_LEN_LIMIT)
        tail = params.get("tail")
        record = await store_module.task_store.get(task_id)
        if record is None:
            raise ToolMessageError(_NOT_FOUND.format(task_id=task_id))
        if action == "status":
            return ToolResult(text=await _status(record))
        if action == "result":
            return ToolResult(text=await _result(record, task_id, wait, wait_timeout, offset, max_char))
        if action == "logs":
            return ToolResult(text=await _logs(record, offset, max_char, tail))
        return ToolResult(text=await _cancel(task_id, record))


task_tool = _TaskTool()
