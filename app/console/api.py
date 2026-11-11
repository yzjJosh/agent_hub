from __future__ import annotations

import asyncio
import os
import sys
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route, Router
from starlette.types import ASGIApp, Receive, Send, Scope

import app.lib.db.store as store_module
from app.console import toolcall
from app.console.conversation_api import conversation_stream_impl, conversation_window_impl
from app.console.errors import error_response
from app.console.render_api import render_message_impl
from app.console.result_api import result_files_for, result_feed_impl, result_view_impl
from app.console.thumb_api import thumb_impl, thumbs_impl
from app.console.view_api import retry_view_impl, view_impl
from app.console.vnc import vnc_reachable
from app.console.windows import OFFSET_MAX_BYTES, backward_window, forward_window, parse_int, parse_max_bytes, parse_offset
from app.lib.db.store import (
    Presentation,
    TaskView,
    ancestor_views,
    keyset_tasks_for_tool,
    store_is_open,
    subtask_views,
    task_view,
    task_view_by_prefix,
    tool_stats,
)
from app.lib.shared import activity
from app.lib.shared.paths import resolve_path, resolve_path_sync, resources_dir
from app.lib.render import cache, thumbs
from app.lib.render.cache import thumb_status_sync
from app.lib.render.classify import classify, classify_sync, kind_label
from app.lib.task import TaskStatus, task_manager
from app.lib.shared.timefmt import LOCAL_TIME_FORMAT, iso_to_local
from app.mcp.registry import registered_tools, tool_class
from app.resources.resources import dav_base_url

__all__ = [
    "API_ROUTE_METHODS",
    "ConsolePathSanitizer",
    "ConsoleSecurityMiddleware",
    "STATIC_DIR",
    "api_not_found",
    "build_api_routes",
    "dav_download",
]

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_SKIP_PREFIXES = ("/mcp", "/dav", "/v1")
_NON_SPA_PREFIXES = ("/api", "/assets", "/dav", "/mcp", "/v1", "/view", "/vnc")
_CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self'; frame-src 'self'; "
    "connect-src 'self'; base-uri 'self'"
)
_CSP_NOVNC = _CSP.replace("img-src 'self'", "img-src 'self' data:")
_VNC_CACHE_MAX_AGE = 3600
_VERSION = "1.0.0"
_ZERO = {"running": 0, "completed": 0, "failed": 0, "cancelled": 0, "last_activity": None, "running_since": None}
_TASK_PAGE_DEFAULT_LIMIT = 200
_TASK_PAGE_MAX_LIMIT = 500
_LOG_FILE = "log"
_RESULT_FILE = "result"
_LOG_WINDOW_DEFAULT_MAX_BYTES = 1048576
_RESULT_WINDOW_DEFAULT_MAX_BYTES = 524288
_BACKWARD_LINES_MIN = 1
_BACKWARD_LINES_MAX = 5000
_API_METHODS = ("GET", "HEAD", "POST", "PUT", "DELETE", "PATCH")
_GET_ONLY_METHODS = ("GET", "HEAD")
# Every API route lists all of these methods, so a request that uses a method
# an endpoint does not take still gets a JSON error from that endpoint.
API_ROUTE_METHODS = _API_METHODS + ("OPTIONS", "CONNECT", "TRACE")

_PENDING_CANCELS: set[asyncio.Task[None]] = set()


def _is_spa_path(path: str) -> bool:
    """Check whether the web page answers a path.

    Args:
        path: The request path.

    Returns:
        True when the page answers it.
    """
    if path in ("/websockify", "/api"):
        return False
    return not path.startswith(tuple(prefix + "/" for prefix in _NON_SPA_PREFIXES))


def _is_vnc_static_path(path: str) -> bool:
    """Check whether a path is a screen sharing file.

    Args:
        path: The request path.

    Returns:
        True when the path sits inside the screen sharing folder.
    """
    return path.startswith("/vnc/")


def _is_skip_path(path: str) -> bool:
    """Check whether a path belongs to another service.

    A path that only starts with the same letters does not count.

    Args:
        path: The request path.

    Returns:
        True when another service owns the path.
    """
    return path in _SKIP_PREFIXES or path.startswith(tuple(prefix + "/" for prefix in _SKIP_PREFIXES))


def _utc_now_iso() -> str:
    """Get the time right now in UTC.

    Returns:
        The time as text.
    """
    return datetime.now(timezone.utc).isoformat()


def _local_now() -> str:
    """Get the time right now on this machine.

    Returns:
        The time as text, with no time zone name.
    """
    return datetime.now().strftime(LOCAL_TIME_FORMAT)


def _generated_stamps() -> tuple[str, str]:
    """Get the time right now in UTC and on this machine.

    Returns:
        The UTC time and the machine time.
    """
    return _utc_now_iso(), _local_now()


def _task_to_dict(view: TaskView) -> dict:
    """Turn a task into the JSON the page reads.

    Args:
        view: The task to turn into JSON.

    Returns:
        The task as JSON.
    """
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
    """Check whether a task can be re-run from the console.

    Args:
        view: The task to check.

    Returns:
        True when the task is a registered, tracked tool.
    """
    if view.tool_name is None:
        return False
    return view.tool_name in registered_tools()


def _parse_limit(raw: str | None) -> int | None:
    """Read how many tasks a page should hold.

    Args:
        raw: The value from the request, or None.

    Returns:
        The count, or None when the value is not a whole number from 1 to 500.
    """
    if raw is None:
        return _TASK_PAGE_DEFAULT_LIMIT
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 1 or value > _TASK_PAGE_MAX_LIMIT:
        return None
    return value


def _is_valid_cursor(cursor: str) -> bool:
    """Check whether a page marker looks right.

    Args:
        cursor: The marker from the request.

    Returns:
        True when the marker has both of its parts.
    """
    left, sep, right = cursor.partition("|")
    return bool(sep and left and right)


class ConsolePathSanitizer:
    """Takes line breaks out of a request path before it is routed."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path = scope["path"]
            if "\n" in path or "\r" in path:
                clean = path.replace("\n", "").replace("\r", "")
                scope["path"] = clean
                scope["raw_path"] = clean.encode("utf-8")
        await self.app(scope, receive, send)


class ConsoleSecurityMiddleware(BaseHTTPMiddleware):
    """Adds safety and caching headers to console replies.

    Replies from the other services are left alone. Screen sharing pages may
    also show pictures that are written into the page itself.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        path = request.url.path
        if _is_skip_path(path):
            return response
        is_vnc_static = _is_vnc_static_path(path)
        csp = _CSP_NOVNC if is_vnc_static else _CSP
        response.headers.setdefault("Content-Security-Policy", csp)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if is_vnc_static:
            response.headers.setdefault("Cache-Control", f"max-age={_VNC_CACHE_MAX_AGE}")
        elif path.startswith("/assets") or _is_spa_path(path):
            response.headers.setdefault("Cache-Control", "no-cache")
        if path.startswith("/api") or _is_spa_path(path):
            response.headers.setdefault("X-Frame-Options", "DENY")
        return response


async def api_not_found(request: Request) -> JSONResponse:
    """Answer an API path that has no endpoint.

    Every method gets the same not found reply.

    Args:
        request: The incoming request.

    Returns:
        The not found reply.
    """
    return error_response(404, "not_found", f"No API endpoint at '{request.url.path}'.")


def _method_guard(
    impl: Callable[[Request], Awaitable[Response]], allowed: tuple[str, ...]
) -> Callable[[Request], Awaitable[Response]]:
    """Let only some methods reach an endpoint.

    Any other method gets a reply that says the method is not allowed and
    lists the methods that are.

    Args:
        impl: The endpoint to guard.
        allowed: The methods the endpoint takes.

    Returns:
        The guarded endpoint.
    """

    async def guarded(request: Request) -> Response:
        if request.method not in allowed:
            return error_response(
                405,
                "method_not_allowed",
                f"Method '{request.method}' is not allowed at '{request.url.path}'.",
                headers={"allow": ", ".join(allowed)},
            )
        return await impl(request)

    return guarded


async def _health(_request: Request) -> JSONResponse:
    if not store_is_open():
        return error_response(503, "internal", "Record store is not open.")
    return JSONResponse({"status": "ok", "version": _VERSION, "dav_base_url": dav_base_url()})


async def _tools_impl() -> JSONResponse:
    """List the tools the console shows.

    Returns:
        Each tool with its name, its words, and its run counts.
    """
    roster = {name: cls.summary for name, cls in registered_tools().items() if cls.tracked is True}
    stats = {s["tool_name"]: s for s in await tool_stats()}
    generated, generated_local = _generated_stamps()
    tools = [
        {
            "name": name,
            "description": desc,
            "running": stats.get(name, _ZERO)["running"],
            "completed": stats.get(name, _ZERO)["completed"],
            "failed": stats.get(name, _ZERO)["failed"],
            "cancelled": stats.get(name, _ZERO)["cancelled"],
            "last_activity": stats.get(name, _ZERO)["last_activity"],
            "last_activity_local": iso_to_local(stats.get(name, _ZERO)["last_activity"]),
            "running_since": stats.get(name, _ZERO)["running_since"],
        }
        for name, desc in roster.items()
    ]
    return JSONResponse({"generated_at": generated, "generated_at_local": generated_local, "tools": tools})


async def _tool_tasks_impl(tool: str, limit: int, cursor: str | None) -> JSONResponse:
    """List one page of a tool's tasks.

    Args:
        tool: The tool name.
        limit: How many tasks the page holds.
        cursor: Where the page starts, or None for the first page.

    Returns:
        The page of tasks, or an error reply when the tool is unknown.
    """
    cls = tool_class(tool)
    if cls is None:
        return error_response(404, "not_found", f"Unknown tool '{tool}'.")
    views, next_cursor = await keyset_tasks_for_tool(tool, limit, cursor)
    generated, generated_local = _generated_stamps()
    return JSONResponse(
        {
            "description": cls.summary,
            "tasks": [_task_to_dict(v) for v in views],
            "next_cursor": next_cursor,
            "generated_at": generated,
            "generated_at_local": generated_local,
        }
    )


def _stat_log_result(data_dir: str, run_dir: str) -> tuple[int, int | None]:
    """Get how big a run's log and result files are.

    Args:
        data_dir: Where run folders are kept.
        run_dir: The run's folder name.

    Returns:
        The log size, 0 when there is no log, and the result size, None when
        there is no result.
    """
    log_path = os.path.join(data_dir, run_dir, _LOG_FILE)
    result_path = os.path.join(data_dir, run_dir, _RESULT_FILE)
    try:
        log_size = os.path.getsize(log_path)
    except OSError:
        log_size = 0
    try:
        result_size = os.path.getsize(result_path)
    except OSError:
        result_size = None
    return log_size, result_size


async def _task_detail_impl(task_id: str) -> JSONResponse:
    """Get everything the console shows about one task.

    Args:
        task_id: Which task to look at.

    Returns:
        The task, its subtasks, its file sizes, and its shared files, or an
        error reply.
    """
    view = await task_view(task_id)
    if view is None:
        view = await task_view_by_prefix(task_id)
    if view is None:
        return error_response(404, "not_found", f"No task found with id '{task_id}'.")
    children = [_task_to_dict(c) for c in await subtask_views(view.task_id)]
    ancestors = [
        {"task_id": v.task_id, "tool": v.tool_name, "agent_name": v.agent_name, "status": v.status}
        for v in await ancestor_views(view.task_id)
    ]
    log_size, result_size = await asyncio.to_thread(
        _stat_log_result, store_module.task_store.data_dir, view.run_dir
    )
    result_files = await result_files_for(store_module.task_store.data_dir, view.run_dir)
    task_dict = _task_to_dict(view)
    task_dict["rerunnable"] = _is_rerunnable(view)
    return JSONResponse(
        {"task": task_dict, "subtasks": children, "ancestors": ancestors,
         "log_size": log_size, "result_size": result_size, "result_files": result_files}
    )


def _file_size(path: str) -> int:
    """Get how big a file is.

    Args:
        path: Where the file is on disk.

    Returns:
        The size in bytes, or 0 when there is no file.
    """
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _read_slice(path: str, offset: int, max_bytes: int) -> bytes:
    """Read one stretch of a file.

    Args:
        path: Where the file is on disk.
        offset: The first byte to read.
        max_bytes: How many bytes to read at most.

    Returns:
        The bytes that were read.
    """
    with open(path, "rb") as fh:
        if offset > 0:
            fh.seek(offset)
        return fh.read(max_bytes)


def _require_readable(path: str) -> os.stat_result:
    """Get the facts about a file that can be opened.

    Args:
        path: Where the file is on disk.

    Returns:
        The file facts.

    Raises:
        OSError: If the file cannot be opened.
    """
    with open(path, "rb") as fh:
        return os.fstat(fh.fileno())


def _parse_lines(raw: str | None) -> int | None:
    """Read how many lines to send back.

    A number outside 1 to 5000 is pulled back into that range.

    Args:
        raw: The value from the request, or None.

    Returns:
        The line count, or None when the value is missing or is not a whole
        number.
    """
    if raw is None:
        return None
    value = parse_int(raw, 0)
    if value is None:
        return None
    return min(max(value, _BACKWARD_LINES_MIN), _BACKWARD_LINES_MAX)


async def _with_run_file(
    task_id: str,
    file_name: str,
    missing_code: str,
    missing_message: str,
    handle: Callable[[str], Awaitable[Response]],
) -> Response:
    """Work with one of a task's run files.

    Args:
        task_id: Which task to look at.
        file_name: The name of the run file.
        missing_code: The error name to send when the file is not there.
        missing_message: The error words to send when the file is not there.
        handle: What to do with the file.

    Returns:
        What handle sends back, or an error reply when the task or the file is
        missing.
    """
    record = await store_module.task_store.get(task_id)
    if record is None:
        return error_response(404, "not_found", f"No task found with id '{task_id}'.")
    path = os.path.join(store_module.task_store.data_dir, record.run_dir, file_name)
    if not await asyncio.to_thread(os.path.exists, path):
        return error_response(404, missing_code, missing_message)
    try:
        return await handle(path)
    except OSError:
        return error_response(404, missing_code, missing_message)


async def _forward_run_window(
    task_id: str, file_name: str, missing_code: str, missing_message: str, params: dict, default_max_bytes: int
) -> Response:
    """Read a stretch of a run file that starts at an offset.

    Args:
        task_id: Which task to look at.
        file_name: The name of the run file.
        missing_code: The error name to send when the file is not there.
        missing_message: The error words to send when the file is not there.
        params: The query values from the request.
        default_max_bytes: How many bytes to read when the request does not say.

    Returns:
        The text that was read, or an error reply.
    """
    offset = parse_offset(params.get("offset"))
    max_bytes = parse_max_bytes(params.get("max_bytes"), default_max_bytes)
    if offset is None or max_bytes is None:
        return error_response(
            400,
            "bad_param",
            f"offset must be an integer between 0 and {OFFSET_MAX_BYTES} and max_bytes a positive integer.",
        )

    async def _forward(path: str) -> Response:
        size = await asyncio.to_thread(_file_size, path)
        data = await asyncio.to_thread(_read_slice, path, offset, max_bytes)
        return JSONResponse(await forward_window(data, offset, size))

    return await _with_run_file(task_id, file_name, missing_code, missing_message, _forward)


async def _task_log_impl(task_id: str, params: dict) -> Response:
    """Read part of a task's log.

    Asking for lines reads back from a point in the file. Otherwise it reads
    forward from an offset.

    Args:
        task_id: Which task to look at.
        params: The query values from the request.

    Returns:
        The text that was read, or an error reply.
    """
    if "lines" in params:
        lines = _parse_lines(params.get("lines"))
        end_offset = parse_offset(params.get("end_offset"))
        if lines is None or end_offset is None:
            return error_response(
                400,
                "bad_param",
                f"lines must be an integer and end_offset an integer between 0 and {OFFSET_MAX_BYTES}.",
            )

        async def _backward(path: str) -> Response:
            await asyncio.to_thread(_require_readable, path)
            return JSONResponse(await backward_window(path, lines, end_offset))

        return await _with_run_file(task_id, _LOG_FILE, "log_not_found", "No log output for this task.", _backward)
    return await _forward_run_window(task_id, _LOG_FILE, "log_not_found", "No log output for this task.", params, _LOG_WINDOW_DEFAULT_MAX_BYTES)


async def _task_result_impl(task_id: str, params: dict) -> Response:
    """Read part of a task's result file.

    Args:
        task_id: Which task to look at.
        params: The query values from the request.

    Returns:
        The text that was read, or an error reply.
    """
    return await _forward_run_window(task_id, _RESULT_FILE, "result_not_found", "No result stored for this task.", params, _RESULT_WINDOW_DEFAULT_MAX_BYTES)


def _cancel_worker_done(worker: asyncio.Task[None]) -> None:
    """Forget a stop job that finished.

    A job that failed has its error logged.

    Args:
        worker: The stop job that finished.
    """
    _PENDING_CANCELS.discard(worker)
    if not worker.cancelled():
        exception = worker.exception()
        if exception is not None:
            print(f"[api._start_cancel] cancel worker {worker.get_name()} failed: {exception!r}", file=sys.stderr, flush=True)


def _start_cancel(task_id: str) -> None:
    """Start stopping a task in the background.

    Args:
        task_id: Which task to stop.
    """
    worker = asyncio.create_task(task_manager.cancel(task_id), name=f"cancel-{task_id}")
    _PENDING_CANCELS.add(worker)
    worker.add_done_callback(_cancel_worker_done)


async def _task_cancel_impl(task_id: str) -> Response:
    """Ask a running task to stop.

    Args:
        task_id: Which task to stop.

    Returns:
        The reply saying the stop began, or an error reply when the task is
        missing or is not running.
    """
    record = await store_module.task_store.get(task_id)
    if record is None:
        return error_response(404, "not_found", f"No task found with id '{task_id}'.")
    if record.status != "running":
        return JSONResponse(
            {"error": f"Task {task_id} is {record.status}.", "code": "already_terminal", "status": record.status},
            status_code=409,
        )
    live = None
    try:
        live = await task_manager.get(uuid.UUID(task_id))
    except ValueError:
        pass
    if live is None or live.status is not TaskStatus.RUNNING:
        return JSONResponse(
            {"error": f"No live task for {task_id}.", "code": "no_live_task", "status": record.status},
            status_code=409,
        )
    _start_cancel(task_id)
    return JSONResponse({"task_id": task_id, "status": "running"}, status_code=202)


async def _task_cancel(request: Request) -> Response:
    """Handle a request to stop a task."""
    return await _task_cancel_impl(request.path_params["task_id"])


async def _tool_schema(request: Request) -> Response:
    """Handle a request for one tool's parameters."""
    return toolcall.tool_schema_impl(request.path_params["tool"])


async def _tool_call(request: Request) -> Response:
    """Handle a request to run one tool."""
    return await toolcall.tool_call_impl(request, request.path_params["tool"])


def _fingerprint_str(
    max_started: str | None, max_ended: str | None, running_count: int, max_presented: str | None
) -> str:
    """Build the mark that changes when tasks or presentations change.

    Args:
        max_started: The newest start time.
        max_ended: The newest end time.
        running_count: How many tasks are running.
        max_presented: The newest presentation time.

    Returns:
        The mark as text.
    """
    return f"{max_started or ''}|{max_ended or ''}|{running_count}|{max_presented or ''}"


async def _stats_block(running_total: int) -> dict:
    """Count the tasks of every tool.

    Args:
        running_total: How many tasks are running.

    Returns:
        The totals and the counts for each tool.
    """
    roster = set(registered_tools())
    by_tool = {s["tool_name"]: s for s in await store_module.tool_stats()}
    total = await store_module.total_count()
    tools = {}
    for name in roster:
        s = by_tool.get(name, _ZERO)
        tools[name] = {
            "running": s["running"],
            "completed": s["completed"],
            "failed": s["failed"],
            "cancelled": s["cancelled"],
            "last_activity": s["last_activity"],
            "last_activity_local": iso_to_local(s["last_activity"]),
            "running_since": s["running_since"],
        }
    return {"total": total, "running": running_total, "tools": tools}


async def _changes_impl(since_fp: str | None, since_gen: str | None) -> Response:
    """Report what changed since the last check.

    When nothing changed the reply says so and carries no tasks.

    Args:
        since_fp: The mark from the last check, or None.
        since_gen: The time of the last check, or None.

    Returns:
        The running, started, and finished tasks with the counts, or the reply
        that says nothing changed.
    """
    max_started, max_ended, running_count, max_presented = await store_module.fingerprint()
    fingerprint = _fingerprint_str(max_started, max_ended, running_count, max_presented)
    if since_fp is not None and since_fp == fingerprint:
        return Response(status_code=304)
    running = await store_module.running_views()
    started = await store_module.started_since_views(since_gen) if since_gen else []
    finished = await store_module.finished_since_views(since_gen) if since_gen else []
    generated, generated_local = _generated_stamps()
    return JSONResponse(
        {
            "fingerprint": fingerprint,
            "generated_at": generated,
            "generated_at_local": generated_local,
            "running": [_task_to_dict(v) for v in running],
            "started": [_task_to_dict(v) for v in started],
            "finished": [_task_to_dict(v) for v in finished],
            "stats": await _stats_block(running_count),
        }
    )


async def _task_log_download(request: Request) -> Response:
    """Send a task's log as a file to save."""
    task_id = request.path_params["task_id"]

    async def _respond(path: str) -> Response:
        stat = await asyncio.to_thread(_require_readable, path)
        return FileResponse(path, media_type="text/plain", filename=f"{task_id}_log", stat_result=stat)

    return await _with_run_file(task_id, _LOG_FILE, "log_not_found", "No log output for this task.", _respond)


async def _task_result_download(request: Request) -> Response:
    """Send a task's result as a file to save."""
    task_id = request.path_params["task_id"]

    async def _respond(path: str) -> Response:
        stat = await asyncio.to_thread(_require_readable, path)
        return FileResponse(path, media_type="application/octet-stream", filename=f"{task_id}_result", stat_result=stat)

    return await _with_run_file(task_id, _RESULT_FILE, "result_not_found", "No result stored for this task.", _respond)


def _parent_dir(rel: str) -> str | None:
    """Get the folder that holds a folder.

    Args:
        rel: A path inside the shared folder.

    Returns:
        The path of the folder above it, empty when that is the top folder, or
        None when the path is already the top folder.
    """
    if not rel:
        return None
    parts = rel.split("/")
    return "/".join(parts[:-1]) if len(parts) > 1 else ""


def _subtree_stats(full: str) -> tuple[int, int]:
    """Count the files under a folder and add up their sizes.

    Files that go away while counting are skipped.

    Args:
        full: Where the folder is on disk.

    Returns:
        How many files there are and how many bytes they take.
    """
    count = 0
    size = 0
    for dirpath, _dirnames, filenames in os.walk(full):
        for filename in filenames:
            try:
                file_size = os.path.getsize(os.path.join(dirpath, filename))
            except OSError:
                continue
            count += 1
            size += file_size
    return count, size


def _dir_entry(base: str, entry: os.DirEntry) -> dict | None:
    """Describe one file or folder for the page, without its thumbnail state.

    Args:
        base: Where the shared folder is on disk.
        entry: The file or folder to describe.

    Returns:
        The file or folder as JSON, or None when it is gone or sits outside
        the shared folder.
    """
    try:
        st = entry.stat()
    except OSError:
        return None
    rel = os.path.relpath(entry.path, base).replace(os.sep, "/")
    if resolve_path_sync(base, rel) is None:
        return None
    mtime = int(st.st_mtime)
    item = {
        "name": entry.name,
        "path": rel,
        "type": "dir" if entry.is_dir() else "file",
        "mtime": mtime,
        "mtime_local": datetime.fromtimestamp(mtime).strftime(LOCAL_TIME_FORMAT),
    }
    if entry.is_dir():
        item["file_count"], item["total_size"] = _subtree_stats(entry.path)
    else:
        item["size"] = st.st_size
    return item


def _file_fact(full: str, rel: str, mtime: int) -> dict:
    """Say one file's type and how its thumbnail is doing.

    The caller must already run off the event loop.

    Args:
        full: Where the file is on disk.
        rel: A path inside the shared folder.
        mtime: When the file last changed.

    Returns:
        The type name and the thumbnail state with its address.
    """
    kind = classify_sync(full)
    engine = thumbs.thumb_engine(kind)
    status = "none" if engine is None else thumb_status_sync(cache.key_for(rel, mtime))
    return {
        "kind": kind_label(kind, os.path.basename(rel), None),
        "thumb": {"status": status, "url": cache.thumb_url(rel)},
    }


def _scan_dir(root: str, rel: str) -> tuple[str, list[dict]] | None:
    """Describe everything inside one folder.

    Args:
        root: Where the shared folder is on disk.
        rel: A path inside that folder, empty for the folder itself.

    Returns:
        The tidy path of the folder, plus each entry with its type and
        thumbnail state, or None when there is no such folder.
    """
    target = resolve_path_sync(root, rel)
    if target is None or not os.path.isdir(target):
        return None
    base = os.path.realpath(root)
    scanned = []
    with os.scandir(target) as it:
        for entry in it:
            item = _dir_entry(base, entry)
            if item is not None:
                if item["type"] == "file":
                    item.update(_file_fact(entry.path, item["path"], item["mtime"]))
                scanned.append(item)
    listed = os.path.relpath(target, base).replace(os.sep, "/")
    return ("" if listed == "." else listed), scanned


async def _list_dir_listing(root: str, rel: str) -> dict | None:
    """List what sits inside one folder.

    The path in the answer is the tidy path of the folder that was listed.

    Args:
        root: Where the shared folder is on disk.
        rel: A path inside that folder, empty for the folder itself.

    Returns:
        The folder path, the folder above it, and what sits inside, or None
        when there is no such folder.
    """
    scanned = await asyncio.to_thread(_scan_dir, root, rel)
    if scanned is None:
        return None
    listed, entries = scanned
    entries.sort(key=lambda e: (e["type"] != "dir", e["name"]))
    return {"path": listed, "parent": _parent_dir(listed), "entries": entries}


async def _resources_impl(path: str | None) -> Response:
    """List what sits inside one shared folder.

    Args:
        path: A path inside the shared folder, or None for the top folder.

    Returns:
        What sits inside the folder, or an error reply.
    """
    rel = (path or "").strip("/")
    root = resources_dir()
    # A path with a zero byte in it is not allowed.
    if "\x00" in rel or await resolve_path(root, rel) is None:
        return error_response(400, "bad_param", f"Invalid path: '{path}'.")
    listing = await _list_dir_listing(root, rel)
    if listing is None:
        return error_response(404, "not_found", f"No such directory: '{rel or '/'}'.")
    return JSONResponse(listing)


_SEARCH_NODE_CAP = 5000
_SEARCH_DEFAULT_LIMIT = 200
_SEARCH_MAX_LIMIT = 1000


def _search_match(rel: str, entry: os.DirEntry, ancestors: list[str]) -> dict | None:
    """Describe one search hit.

    Only a file carries a size, a type, and a thumbnail.

    Args:
        rel: The path of the hit inside the shared folder.
        entry: The file or folder that matched.
        ancestors: The names of the folders above it.

    Returns:
        The hit as JSON, or None when it is gone.
    """
    try:
        st = entry.stat()
    except OSError:
        return None
    mtime = int(st.st_mtime)
    is_dir = entry.is_dir()
    item = {
        "path": rel,
        "name": entry.name,
        "type": "dir" if is_dir else "file",
        "mtime": mtime,
        "mtime_local": datetime.fromtimestamp(mtime).strftime(LOCAL_TIME_FORMAT),
        "matched_ancestor_paths": list(ancestors),
    }
    if not is_dir:
        item["size"] = st.st_size
        item.update(_file_fact(entry.path, rel, mtime))
    return item


def _search_tree(root: str, q: str, limit: int, node_cap: int = _SEARCH_NODE_CAP) -> tuple[list[dict], bool]:
    """Find files and folders whose path holds a piece of text.

    Anything outside the shared folder is skipped. The search stops once it
    has looked at enough places.

    Args:
        root: Where the shared folder is on disk.
        q: The text to look for.
        limit: How many hits to send back at most.
        node_cap: How many places to look at most.

    Returns:
        The hits and whether the search stopped early.
    """
    needle = q.lower()
    base = os.path.realpath(root)
    matches: list[dict] = []
    truncated = False
    visited = 0
    queue: deque[tuple[str, list[str]]] = deque([("", [])])
    while queue:
        if visited >= node_cap:
            truncated = True
            break
        rel, ancestors = queue.popleft()
        full = base if rel == "" else os.path.join(base, rel)
        visited += 1
        try:
            with os.scandir(full) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            if visited >= node_cap:
                truncated = True
                break
            child_rel = entry.name if rel == "" else f"{rel}/{entry.name}"
            if resolve_path_sync(base, child_rel) is None:
                continue
            visited += 1
            if needle in child_rel.lower():
                if len(matches) < limit:
                    item = _search_match(child_rel, entry, ancestors)
                    if item is not None:
                        matches.append(item)
                else:
                    truncated = True
            if entry.is_dir():
                queue.append((child_rel, [*ancestors, entry.name]))
    return matches, truncated


def _parse_search_limit(raw: str | None) -> int | None:
    """Read how many search hits to send back.

    Args:
        raw: The value from the request, or None.

    Returns:
        The count, or None when the value is not a whole number from 1 to 1000.
    """
    if raw is None:
        return _SEARCH_DEFAULT_LIMIT
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 1 or value > _SEARCH_MAX_LIMIT:
        return None
    return value


async def _resources_search_impl(q: str, limit: int) -> JSONResponse:
    """Search the shared folder.

    Args:
        q: The text to look for.
        limit: How many hits to send back at most.

    Returns:
        The hits and whether the search stopped early.
    """
    matches, truncated = await asyncio.to_thread(_search_tree, resources_dir(), q, limit)
    return JSONResponse({"matches": matches, "truncated": truncated})


async def _resources(request: Request) -> Response:
    return await _resources_impl(request.query_params.get("path"))


async def _resources_search(request: Request) -> Response:
    q = request.query_params.get("q")
    if not q:
        return error_response(400, "bad_param", "q must be a non-empty substring.")
    limit = _parse_search_limit(request.query_params.get("limit"))
    if limit is None:
        return error_response(400, "bad_param", "limit must be an integer between 1 and 1000.")
    return await _resources_search_impl(q, limit)


async def _dav_download_impl(path: str | None) -> Response:
    """Send one shared file for the browser to save.

    Args:
        path: A path inside the shared folder, or None.

    Returns:
        The file to save, or an error reply when the path is missing, is not
        allowed, or has no file.
    """
    if path is None or not path.strip():
        return error_response(400, "bad_param", "path must be a share-relative file path.")
    rel = path.strip("/")
    if "\x00" in rel:
        return error_response(400, "bad_param", f"Invalid path: '{path}'.")
    full = await resolve_path(resources_dir(), rel)
    if full is None or not await asyncio.to_thread(os.path.isfile, full):
        return error_response(404, "not_found", f"No such file: '{rel}'.")
    return FileResponse(full, filename=os.path.basename(rel))


async def dav_download(request: Request) -> Response:
    """Send a shared file so the browser saves it instead of showing it.

    This route must be added before the /dav service.

    Args:
        request: The incoming request.

    Returns:
        The file to save, or an error reply.
    """
    path = request.query_params.get("path")
    if request.method == "GET" and path is not None:
        await activity.record(path.strip("/"))
    return await _dav_download_impl(path)


async def _vnc_status(_request: Request) -> JSONResponse:
    try:
        reachable = await vnc_reachable()
    except Exception:
        return error_response(500, "internal", "VNC probe failed.")
    return JSONResponse({"reachable": reachable})


async def _view(request: Request) -> Response:
    """Handle a request to view a shared file."""
    path = request.path_params["path"]
    if request.method == "GET":
        await activity.record(path)
    return await view_impl(path)


async def _retry_view(request: Request) -> Response:
    """Handle a request to convert a shared file again."""
    return await retry_view_impl(request.path_params["path"])


async def _thumb(request: Request) -> Response:
    """Handle a request for one file's small picture."""
    return await thumb_impl(request.query_params.get("path"))


async def _thumbs(request: Request) -> Response:
    """Handle a request to make the small pictures of a folder."""
    return await thumbs_impl(request.query_params.get("path"))


async def _tools(_request: Request) -> JSONResponse:
    """Handle a request for the tool list."""
    return await _tools_impl()


async def _tool_tasks(request: Request) -> JSONResponse:
    """Handle a request for one tool's tasks."""
    limit = _parse_limit(request.query_params.get("limit"))
    if limit is None:
        return error_response(400, "bad_param", "limit must be an integer between 1 and 500.")
    cursor = request.query_params.get("cursor")
    if cursor is not None and not _is_valid_cursor(cursor):
        return error_response(400, "bad_param", "cursor must be '<started_at>|<task_id>'.")
    return await _tool_tasks_impl(request.path_params["tool"], limit, cursor)


async def _task_detail(request: Request) -> JSONResponse:
    """Handle a request for one task."""
    return await _task_detail_impl(request.path_params["task_id"])


async def _conversation_stream(request: Request) -> Response:
    """Stream an agent run's conversation."""
    return await conversation_stream_impl(request.path_params["task_id"])


async def _conversation_window(request: Request) -> Response:
    """Read a window of an agent run's conversation."""
    return await conversation_window_impl(request.path_params["task_id"], dict(request.query_params))


async def _task_log(request: Request) -> Response:
    """Handle a request for a task's log."""
    return await _task_log_impl(request.path_params["task_id"], dict(request.query_params))


async def _task_result(request: Request) -> Response:
    """Handle a request for a task's result."""
    return await _task_result_impl(request.path_params["task_id"], dict(request.query_params))


async def _task_result_view(request: Request) -> Response:
    """Handle a request to view a task's result."""
    return await result_view_impl(request.path_params["task_id"], dict(request.query_params))


async def _task_result_feed(request: Request) -> Response:
    """Handle a request for a task's result feed."""
    return await result_feed_impl(request.path_params["task_id"], dict(request.query_params))


async def _render_message(request: Request) -> Response:
    """Render one message's bytes through the generic pipeline."""
    return await render_message_impl(request)


async def _changes(request: Request) -> Response:
    """Handle a request for what changed."""
    since_fp = request.query_params.get("since_fp")
    since_gen = request.query_params.get("since_gen")
    if (since_fp is None) != (since_gen is None):
        since_fp, since_gen = None, None
    return await _changes_impl(since_fp, since_gen)


def _presented_file_facts(root: str, rel: str) -> tuple[str, os.stat_result] | None:
    """Find where a presented file is and read its facts from disk.

    Args:
        root: Where the shared folder is on disk.
        rel: The path of the file inside it.

    Returns:
        Where the file is on disk and its facts, or None when it is gone.
    """
    full = resolve_path_sync(root, rel)
    if full is None or not os.path.isfile(full):
        return None
    return full, os.stat(full)


async def _presented_file(root: str, rel: str) -> dict:
    """Describe one file a presentation carries, with the facts read from disk.

    Args:
        root: Where the shared folder is on disk.
        rel: The path of the file inside it.

    Returns:
        The file as JSON.
    """
    name = os.path.basename(rel)
    facts = await asyncio.to_thread(_presented_file_facts, root, rel)
    if facts is not None:
        full, st = facts
        mtime = int(st.st_mtime)
        kind = await classify(full)
        return {
            "path": rel,
            "name": name,
            "size": st.st_size,
            "mtime": mtime,
            "present": True,
            "kind": kind_label(kind, name, None),
            "thumb": {"status": await cache.thumb_status(cache.key_for(rel, mtime)), "url": cache.thumb_url(rel)},
        }
    kind = await classify(os.path.join(root, rel))
    return {
        "path": rel,
        "name": name,
        "size": None,
        "mtime": None,
        "present": False,
        "kind": kind_label(kind, name, None),
        "thumb": {"status": "none", "url": cache.thumb_url(rel)},
    }


async def _presented_row(root: str, p: Presentation) -> dict:
    """Describe one presentation and how many of its files are still there.

    Args:
        root: Where the shared folder is on disk.
        p: The presentation to describe.

    Returns:
        The presentation as JSON.
    """
    files = [await _presented_file(root, rel) for rel in p.files]
    present_count = sum(1 for f in files if f["present"])
    total_size = sum(f["size"] for f in files if f["size"] is not None)
    if present_count == len(p.files):
        state = "active"
    elif present_count == 0:
        state = "expired"
    else:
        state = "partial"
    return {
        "id": p.id,
        "title": p.title,
        "presented_at": p.presented_at,
        "presented_at_local": iso_to_local(p.presented_at),
        "url": dav_base_url() + p.url_path,
        "kind": "solo" if len(p.files) == 1 else "package",
        "files": files,
        "file_count": len(p.files),
        "present_count": present_count,
        "total_size": total_size,
        "state": state,
    }


async def _presentations_rows(presentations: list[Presentation]) -> list[dict]:
    """Describe a list of presentations with the facts read from disk.

    Args:
        presentations: The presentations to describe, newest first.

    Returns:
        The presentations as JSON.
    """
    root = resources_dir()
    return [await _presented_row(root, p) for p in presentations]


async def _presentations_impl() -> Response:
    """List the presentations newest first, within the shared window.

    Returns:
        The presentations with their files, or an error reply.
    """
    if not store_is_open():
        return error_response(503, "internal", "Record store is not open.")
    presentations = await store_module.presentation_store.list_newest()
    rows = await _presentations_rows(presentations)
    generated, generated_local = _generated_stamps()
    return JSONResponse({"generated_at": generated, "generated_at_local": generated_local, "presentations": rows})


async def _presentations(_request: Request) -> Response:
    """Handle a request for the presentation list."""
    return await _presentations_impl()


def build_api_routes() -> Router:
    """Build the routes under /api.

    The catch all route must stay last. The view retry route takes only POST
    and comes before the view route, so a shared file named retry can still be
    viewed.

    Returns:
        The router that holds the routes.
    """
    routes: list[Route] = [
        Route("/health", _method_guard(_health, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tools", _method_guard(_tools, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tools/{tool}/tasks", _method_guard(_tool_tasks, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tools/{tool}/schema", _method_guard(_tool_schema, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tools/{tool}/call", _method_guard(_tool_call, ("POST",)), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}", _method_guard(_task_detail, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/conversation/stream", _method_guard(_conversation_stream, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/conversation", _method_guard(_conversation_window, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/log", _method_guard(_task_log, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/result", _method_guard(_task_result, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/log/download", _method_guard(_task_log_download, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/result/download", _method_guard(_task_result_download, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/result/view", _method_guard(_task_result_view, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/result/feed", _method_guard(_task_result_feed, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/changes", _method_guard(_changes, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/cancel", _method_guard(_task_cancel, ("POST",)), methods=API_ROUTE_METHODS),
        Route("/resources", _method_guard(_resources, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/resources/search", _method_guard(_resources_search, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/vnc/status", _method_guard(_vnc_status, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/view/{path:path}/retry", _method_guard(_retry_view, ("POST",)), methods=("POST",)),
        Route("/view/{path:path}", _method_guard(_view, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/resources/thumb", _method_guard(_thumb, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/resources/thumbs", _method_guard(_thumbs, ("POST",)), methods=API_ROUTE_METHODS),
        Route("/presentations", _method_guard(_presentations, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/render/message", _method_guard(_render_message, ("POST",)), methods=API_ROUTE_METHODS),
        Route("/{path:path}", api_not_found, methods=API_ROUTE_METHODS),
    ]
    return Router(routes=routes)
