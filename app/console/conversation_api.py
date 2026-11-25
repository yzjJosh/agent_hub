"""Endpoints for an agent run's conversation file."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator

from starlette.responses import JSONResponse, Response, StreamingResponse

import app.lib.db.store as store_module
from app.console.errors import error_response
from app.console.windows import OFFSET_MAX_BYTES, forward_window, parse_max_bytes, parse_offset
from app.lib.agent.conversation_log import tail_lines

__all__ = ["conversation_snapshot_impl", "conversation_stream_impl", "conversation_window_impl"]

_CONV_FILE = "conversation.jsonl"
_WINDOW_DEFAULT_MAX_BYTES = 524288
_DELTA_EVENT_TYPES = ("thinking", "content")
_NO_STORE_HEADERS = {"Cache-Control": "no-store"}


async def _agent_conversation_file(task_id: str) -> tuple[str, str] | Response:
    """Resolve the conversation file path for an agent task."""
    view = await store_module.task_view(task_id)
    if view is None:
        return error_response(404, "not_found", f"No task found with id '{task_id}'.")
    if view.agent_name is None:
        return error_response(404, "not_agent", "This task is not an agent run.")
    path = os.path.join(store_module.task_store.data_dir, view.run_dir, _CONV_FILE)
    return task_id, path


async def _conversation_path(task_id: str) -> tuple[str, str] | Response:
    """Get an existing conversation file for an agent task."""
    result = await _agent_conversation_file(task_id)
    if isinstance(result, Response):
        return result
    _, path = result
    if not await asyncio.to_thread(os.path.isfile, path):
        return error_response(404, "conversation_not_found", "No conversation stored for this agent run.")
    return result


def _delta_text(ev: dict) -> str:
    value = ev.get("text")
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _append_snapshot_event(events: list[dict], ev: dict) -> None:
    ev_type = ev.get("type")
    if ev_type in _DELTA_EVENT_TYPES:
        text = _delta_text(ev)
        if events and events[-1].get("type") == ev_type and events[-1].get("attempt") == ev.get("attempt"):
            events[-1]["text"] = (events[-1].get("text") or "") + text
            return
        entry = dict(ev)
        entry["text"] = text
        events.append(entry)
        return
    events.append(dict(ev))


def _read_snapshot_sync(path: str) -> dict:
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return {"offset": 0, "size": 0, "event_count": 0, "events": []}
    size = len(raw)
    last_nl = raw.rfind(b"\n")
    if last_nl == -1:
        return {"offset": 0, "size": size, "event_count": 0, "events": []}
    aligned_offset = last_nl + 1
    events: list[dict] = []
    event_count = 0
    for raw_line in raw[:aligned_offset].split(b"\n"):
        trimmed = raw_line.strip()
        if not trimmed:
            continue
        try:
            ev = json.loads(trimmed.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        event_count += 1
        _append_snapshot_event(events, ev)
    return {"offset": aligned_offset, "size": size, "event_count": event_count, "events": events}


async def conversation_snapshot_impl(task_id: str) -> Response:
    """Read a compacted snapshot of an agent run's conversation events.

    Adjacent thinking or content pieces from the same attempt are joined, and
    the reply gives the byte offset after the last complete line.

    Args:
        task_id: Which task to look at.

    Returns:
        The snapshot with its byte offset and event count, or an error reply.
    """
    result = await _agent_conversation_file(task_id)
    if isinstance(result, Response):
        return result
    _, path = result
    snapshot = await asyncio.to_thread(_read_snapshot_sync, path)
    return JSONResponse(snapshot, headers=_NO_STORE_HEADERS)


async def conversation_stream_impl(task_id: str, params: dict | None = None) -> Response:
    """Stream an agent run's conversation file as chunked text.

    Every complete line is sent as it appears, and the stream closes when the
    task leaves running and the file has been read to the end.

    Args:
        task_id: Which task to look at.
        params: The query values from the request, or None.

    Returns:
        The chunked stream, or an error reply.
    """
    result = await _conversation_path(task_id)
    if isinstance(result, Response):
        return result
    _tid, path = result
    offset = parse_offset((params or {}).get("offset"))
    if offset is None:
        return error_response(
            400,
            "bad_param",
            f"offset must be an integer between 0 and {OFFSET_MAX_BYTES}.",
        )

    async def _stop() -> bool:
        view = await store_module.task_view(_tid)
        return view is None or view.status != "running"

    async def _lines() -> AsyncGenerator[str, None]:
        async for line in tail_lines(path, _stop, offset=offset):
            yield line + "\n"

    return StreamingResponse(
        _lines(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache, no-store"},
    )


async def conversation_window_impl(task_id: str, params: dict) -> Response:
    """Read a window of an agent run's conversation file.

    Args:
        task_id: Which task to look at.
        params: The query values from the request.

    Returns:
        The window, or an error reply.
    """
    result = await _conversation_path(task_id)
    if isinstance(result, Response):
        return result
    _, path = result
    offset = parse_offset(params.get("offset"))
    max_bytes = parse_max_bytes(params.get("max_bytes"), _WINDOW_DEFAULT_MAX_BYTES)
    if offset is None or max_bytes is None:
        return error_response(
            400,
            "bad_param",
            f"offset must be an integer between 0 and {OFFSET_MAX_BYTES} and max_bytes a positive integer.",
        )

    def _read() -> tuple[int, bytes]:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        with open(path, "rb") as fh:
            if offset > 0:
                fh.seek(offset)
            return size, fh.read(max_bytes)

    size, data = await asyncio.to_thread(_read)
    return JSONResponse(await forward_window(data, offset, size))
