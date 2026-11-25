"""Endpoints for console tools, tasks, run files, cancellation, and change polling."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

import app.lib.db.store as store_module
from app.console.errors import error_response
from app.console.result_api import result_files_for
from app.console.windows import (
    OFFSET_MAX_BYTES,
    backward_window,
    forward_window,
    parse_int,
    parse_max_bytes,
    parse_offset,
)
from app.lib.db.store import TaskView
from app.lib.shared.timefmt import GeneratedTimestamps, build_generated_timestamps, iso_to_local
from app.lib.task import TaskStatus, task_manager
from app.mcp.registry import registered_tools, tool_class

__all__ = [
    "cancel_task",
    "download_task_log",
    "download_task_result",
    "get_changes",
    "get_task_detail",
    "get_task_log",
    "get_task_result",
    "list_tool_tasks",
    "list_tools",
]

_EMPTY_TOOL_STATS = {
    "running": 0,
    "completed": 0,
    "failed": 0,
    "cancelled": 0,
    "last_activity": None,
    "running_since": None,
}
_TASK_PAGE_DEFAULT_LIMIT = 200
_TASK_PAGE_MAX_LIMIT = 500
_LOG_FILE = "log"
_RESULT_FILE = "result"
_LOG_WINDOW_DEFAULT_MAX_BYTES = 1048576
_RESULT_WINDOW_DEFAULT_MAX_BYTES = 524288
_BACKWARD_LINES_MIN = 1
_BACKWARD_LINES_MAX = 5000

_PENDING_CANCELS: set[asyncio.Task[None]] = set()


@dataclass(frozen=True)
class _RunFileSizes:
    """Byte sizes of a run's log and result files."""

    log_bytes: int
    result_bytes: int | None


@dataclass(frozen=True)
class _FileSlice:
    """Total size and bytes read from one stretch of a file."""

    total_bytes: int
    content_bytes: bytes


def _serialize_task(view: TaskView) -> dict:
    return {
        "task_id": view.task_id,
        "tool": view.tool_name,
        "agent_name": view.agent_name,
        "status": view.status,
        "started_at": view.started_at,
        "started_at_local": iso_to_local(view.started_at),
        "ended_at": view.ended_at,
        "ended_at_local": iso_to_local(view.ended_at),
        "error": view.error,
        "parent_task_id": view.parent_task_id,
        "params": view.params,
    }


def _is_rerunnable(view: TaskView) -> bool:
    if view.tool_name is None:
        return False
    return view.tool_name in registered_tools()


def _format_generated_timestamps(timestamps: GeneratedTimestamps) -> dict[str, str]:
    return {
        "generated_at": timestamps.utc_iso,
        "generated_at_local": timestamps.local_display,
    }


def _format_tool_stats(stats: dict) -> dict:
    return {
        "running": stats["running"],
        "completed": stats["completed"],
        "failed": stats["failed"],
        "cancelled": stats["cancelled"],
        "last_activity": stats["last_activity"],
        "last_activity_local": iso_to_local(stats["last_activity"]),
        "running_since": stats["running_since"],
    }


def _parse_task_page_limit(raw_value: str | None) -> int | None:
    value = parse_int(raw_value, _TASK_PAGE_DEFAULT_LIMIT)
    if value is None or value < 1 or value > _TASK_PAGE_MAX_LIMIT:
        return None
    return value


def _is_valid_cursor(cursor: str) -> bool:
    left, separator, right = cursor.partition("|")
    return bool(separator and left and right)


def _measure_run_files(data_dir: str, run_dir: str) -> _RunFileSizes:
    log_path = os.path.join(data_dir, run_dir, _LOG_FILE)
    result_path = os.path.join(data_dir, run_dir, _RESULT_FILE)
    try:
        log_bytes = os.path.getsize(log_path)
    except OSError:
        log_bytes = 0
    try:
        result_bytes: int | None = os.path.getsize(result_path)
    except OSError:
        result_bytes = None
    return _RunFileSizes(log_bytes=log_bytes, result_bytes=result_bytes)


def _read_file_slice(file_path: str, offset: int, max_bytes: int) -> _FileSlice:
    with open(file_path, "rb") as file_handle:
        total_bytes = os.fstat(file_handle.fileno()).st_size
        if offset > 0:
            file_handle.seek(offset)
        content_bytes = file_handle.read(max_bytes)
    return _FileSlice(total_bytes=total_bytes, content_bytes=content_bytes)


def _require_readable_stat(file_path: str) -> os.stat_result:
    with open(file_path, "rb") as file_handle:
        return os.fstat(file_handle.fileno())


def _parse_backward_lines(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None
    value = parse_int(raw_value, 0)
    if value is None:
        return None
    return min(max(value, _BACKWARD_LINES_MIN), _BACKWARD_LINES_MAX)


async def _run_with_task_file(
    task_id: str,
    file_name: str,
    missing_code: str,
    missing_message: str,
    build_response: Callable[[str], Awaitable[Response]],
) -> Response:
    record = await store_module.task_store.get(task_id)
    if record is None:
        return error_response(404, "not_found", f"No task found with id '{task_id}'.")
    file_path = os.path.join(store_module.task_store.data_dir, record.run_dir, file_name)
    if not await asyncio.to_thread(os.path.exists, file_path):
        return error_response(404, missing_code, missing_message)
    try:
        return await build_response(file_path)
    except OSError:
        return error_response(404, missing_code, missing_message)


async def _read_forward_run_window(
    task_id: str,
    file_name: str,
    missing_code: str,
    missing_message: str,
    query_params: dict[str, str],
    default_max_bytes: int,
) -> Response:
    offset = parse_offset(query_params.get("offset"))
    max_bytes = parse_max_bytes(query_params.get("max_bytes"), default_max_bytes)
    if offset is None or max_bytes is None:
        return error_response(
            400,
            "bad_param",
            f"offset must be an integer between 0 and {OFFSET_MAX_BYTES} and max_bytes a positive integer.",
        )

    async def _build_forward_response(file_path: str) -> Response:
        file_slice = await asyncio.to_thread(_read_file_slice, file_path, offset, max_bytes)
        return JSONResponse(await forward_window(file_slice.content_bytes, offset, file_slice.total_bytes))

    return await _run_with_task_file(task_id, file_name, missing_code, missing_message, _build_forward_response)


def _handle_cancel_worker_done(worker: asyncio.Task[None]) -> None:
    _PENDING_CANCELS.discard(worker)
    if not worker.cancelled():
        exception = worker.exception()
        if exception is not None:
            print(
                f"[tasks_api._start_cancel_worker] cancel worker {worker.get_name()} failed: {exception!r}",
                file=sys.stderr,
                flush=True,
            )


def _start_cancel_worker(task_id: str) -> None:
    worker = asyncio.create_task(task_manager.cancel(task_id), name=f"cancel-{task_id}")
    _PENDING_CANCELS.add(worker)
    worker.add_done_callback(_handle_cancel_worker_done)


def _build_fingerprint(
    max_started: str | None,
    max_ended: str | None,
    running_count: int,
    max_presented: str | None,
) -> str:
    return f"{max_started or ''}|{max_ended or ''}|{running_count}|{max_presented or ''}"


async def _build_stats_block(running_total: int) -> dict:
    roster = set(registered_tools())
    stats_by_tool = {row["tool_name"]: row for row in await store_module.tool_stats()}
    total_count = await store_module.total_count()
    tools = {
        tool_name: _format_tool_stats(stats_by_tool.get(tool_name, _EMPTY_TOOL_STATS))
        for tool_name in roster
    }
    return {"total": total_count, "running": running_total, "tools": tools}


async def list_tools(_request: Request) -> JSONResponse:
    """List tracked tools with their summary text and run counts.

    Args:
        _request: The incoming HTTP request.

    Returns:
        The tool list and current timestamps.
    """
    roster = {tool_name: cls.summary for tool_name, cls in registered_tools().items() if cls.tracked is True}
    stats_by_tool = {row["tool_name"]: row for row in await store_module.tool_stats()}
    timestamps = build_generated_timestamps()
    tools = [
        {
            "name": tool_name,
            "description": summary,
            **_format_tool_stats(stats_by_tool.get(tool_name, _EMPTY_TOOL_STATS)),
        }
        for tool_name, summary in roster.items()
    ]
    return JSONResponse(
        {
            **_format_generated_timestamps(timestamps),
            "tools": tools,
        }
    )


async def list_tool_tasks(request: Request) -> JSONResponse:
    """List one page of tasks for a single tool.

    Args:
        request: The incoming HTTP request.

    Returns:
        The page of tasks, or an error reply.
    """
    limit = _parse_task_page_limit(request.query_params.get("limit"))
    if limit is None:
        return error_response(400, "bad_param", "limit must be an integer between 1 and 500.")
    cursor = request.query_params.get("cursor")
    if cursor is not None and not _is_valid_cursor(cursor):
        return error_response(400, "bad_param", "cursor must be '<started_at>|<task_id>'.")
    tool_name = request.path_params["tool"]
    cls = tool_class(tool_name)
    if cls is None:
        return error_response(404, "not_found", f"Unknown tool '{tool_name}'.")
    views, next_cursor = await store_module.keyset_tasks_for_tool(tool_name, limit, cursor)
    timestamps = build_generated_timestamps()
    return JSONResponse(
        {
            "description": cls.summary,
            "tasks": [_serialize_task(view) for view in views],
            "next_cursor": next_cursor,
            **_format_generated_timestamps(timestamps),
        }
    )


async def get_task_detail(request: Request) -> JSONResponse:
    """Get full details for one task, its subtasks, and its run files.

    Args:
        request: The incoming HTTP request.

    Returns:
        The task detail payload, or an error reply.
    """
    task_id = request.path_params["task_id"]
    view = await store_module.task_view(task_id)
    if view is None:
        view = await store_module.task_view_by_prefix(task_id)
    if view is None:
        return error_response(404, "not_found", f"No task found with id '{task_id}'.")
    subtasks = [_serialize_task(child) for child in await store_module.subtask_views(view.task_id)]
    ancestors = [
        {
            "task_id": ancestor.task_id,
            "tool": ancestor.tool_name,
            "agent_name": ancestor.agent_name,
            "status": ancestor.status,
        }
        for ancestor in await store_module.ancestor_views(view.task_id)
    ]
    file_sizes = await asyncio.to_thread(_measure_run_files, store_module.task_store.data_dir, view.run_dir)
    result_files = await result_files_for(store_module.task_store.data_dir, view.run_dir)
    task_payload = _serialize_task(view)
    task_payload["rerunnable"] = _is_rerunnable(view)
    return JSONResponse(
        {
            "task": task_payload,
            "subtasks": subtasks,
            "ancestors": ancestors,
            "log_size": file_sizes.log_bytes,
            "result_size": file_sizes.result_bytes,
            "result_files": result_files,
        }
    )


async def get_task_log(request: Request) -> Response:
    """Read a forward or backward window of a task log.

    Args:
        request: The incoming HTTP request.

    Returns:
        The log window payload, or an error reply.
    """
    task_id = request.path_params["task_id"]
    query_params = dict(request.query_params)
    if "lines" in query_params:
        line_count = _parse_backward_lines(query_params.get("lines"))
        end_offset = parse_offset(query_params.get("end_offset"))
        if line_count is None or end_offset is None:
            return error_response(
                400,
                "bad_param",
                f"lines must be an integer and end_offset an integer between 0 and {OFFSET_MAX_BYTES}.",
            )

        async def _build_backward_response(file_path: str) -> Response:
            await asyncio.to_thread(_require_readable_stat, file_path)
            return JSONResponse(await backward_window(file_path, line_count, end_offset))

        return await _run_with_task_file(
            task_id, _LOG_FILE, "log_not_found", "No log output for this task.", _build_backward_response
        )
    return await _read_forward_run_window(
        task_id,
        _LOG_FILE,
        "log_not_found",
        "No log output for this task.",
        query_params,
        _LOG_WINDOW_DEFAULT_MAX_BYTES,
    )


async def get_task_result(request: Request) -> Response:
    """Read a forward window of a task result file.

    Args:
        request: The incoming HTTP request.

    Returns:
        The result window payload, or an error reply.
    """
    return await _read_forward_run_window(
        request.path_params["task_id"],
        _RESULT_FILE,
        "result_not_found",
        "No result stored for this task.",
        dict(request.query_params),
        _RESULT_WINDOW_DEFAULT_MAX_BYTES,
    )


async def download_task_log(request: Request) -> Response:
    """Send a task log file as a download.

    Args:
        request: The incoming HTTP request.

    Returns:
        The log file response, or an error reply.
    """
    task_id = request.path_params["task_id"]

    async def _build_file_response(file_path: str) -> Response:
        stat_result = await asyncio.to_thread(_require_readable_stat, file_path)
        return FileResponse(file_path, media_type="text/plain", filename=f"{task_id}_log", stat_result=stat_result)

    return await _run_with_task_file(
        task_id, _LOG_FILE, "log_not_found", "No log output for this task.", _build_file_response
    )


async def download_task_result(request: Request) -> Response:
    """Send a task result file as a download.

    Args:
        request: The incoming HTTP request.

    Returns:
        The result file response, or an error reply.
    """
    task_id = request.path_params["task_id"]

    async def _build_file_response(file_path: str) -> Response:
        stat_result = await asyncio.to_thread(_require_readable_stat, file_path)
        return FileResponse(
            file_path,
            media_type="application/octet-stream",
            filename=f"{task_id}_result",
            stat_result=stat_result,
        )

    return await _run_with_task_file(
        task_id, _RESULT_FILE, "result_not_found", "No result stored for this task.", _build_file_response
    )


async def cancel_task(request: Request) -> Response:
    """Start cancelling a running task.

    Args:
        request: The incoming HTTP request.

    Returns:
        An accepted reply when cancellation starts, or an error reply.
    """
    task_id = request.path_params["task_id"]
    record = await store_module.task_store.get(task_id)
    if record is None:
        return error_response(404, "not_found", f"No task found with id '{task_id}'.")
    if record.status != "running":
        return JSONResponse(
            {"error": f"Task {task_id} is {record.status}.", "code": "already_terminal", "status": record.status},
            status_code=409,
        )
    live_task = None
    try:
        live_task = await task_manager.get(uuid.UUID(task_id))
    except ValueError:
        pass
    if live_task is None or live_task.status is not TaskStatus.RUNNING:
        return JSONResponse(
            {"error": f"No live task for {task_id}.", "code": "no_live_task", "status": record.status},
            status_code=409,
        )
    _start_cancel_worker(task_id)
    return JSONResponse({"task_id": task_id, "status": "running"}, status_code=202)


async def get_changes(request: Request) -> Response:
    """Report tasks and counts that changed since the previous poll.

    Args:
        request: The incoming HTTP request.

    Returns:
        The change summary payload, or a 304 reply when nothing changed.
    """
    previous_fingerprint = request.query_params.get("since_fp")
    previous_generated_at = request.query_params.get("since_gen")
    if (previous_fingerprint is None) != (previous_generated_at is None):
        previous_fingerprint = None
        previous_generated_at = None
    max_started, max_ended, running_count, max_presented = await store_module.fingerprint()
    fingerprint = _build_fingerprint(max_started, max_ended, running_count, max_presented)
    if previous_fingerprint is not None and previous_fingerprint == fingerprint:
        return Response(status_code=304)
    running = await store_module.running_views()
    started = await store_module.started_since_views(previous_generated_at) if previous_generated_at else []
    finished = await store_module.finished_since_views(previous_generated_at) if previous_generated_at else []
    timestamps = build_generated_timestamps()
    return JSONResponse(
        {
            "fingerprint": fingerprint,
            **_format_generated_timestamps(timestamps),
            "running": [_serialize_task(view) for view in running],
            "started": [_serialize_task(view) for view in started],
            "finished": [_serialize_task(view) for view in finished],
            "stats": await _build_stats_block(running_count),
        }
    )
