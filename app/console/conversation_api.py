"""Endpoints for an agent run's conversation file."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator

from starlette.responses import JSONResponse, Response, StreamingResponse

import app.lib.db.store as store_module
from app.console.errors import error_response
from app.console.windows import OFFSET_MAX_BYTES, forward_window, parse_max_bytes, parse_offset
from app.lib.agent.engine import tail_lines

__all__ = ["conversation_stream_impl", "conversation_window_impl"]

_CONV_FILE = "conversation.jsonl"
_WINDOW_DEFAULT_MAX_BYTES = 524288


async def _conversation_path(task_id: str) -> tuple[str, str] | Response:
    """Get the conversation file for an agent task.

    Args:
        task_id: Which task to look at.

    Returns:
        The task id and the path to its conversation file, or an error reply.
    """
    view = await store_module.task_view(task_id)
    if view is None:
        return error_response(404, "not_found", f"No task found with id '{task_id}'.")
    if view.agent_name is None:
        return error_response(404, "not_agent", "This task is not an agent run.")
    path = os.path.join(store_module.task_store.data_dir, view.run_dir, _CONV_FILE)
    if not os.path.isfile(path):
        return error_response(404, "conversation_not_found", "No conversation stored for this agent run.")
    return task_id, path


async def conversation_stream_impl(task_id: str) -> Response:
    """Stream an agent run's conversation file as chunked text.

    Every complete line is sent as it appears, and the stream closes when the
    task leaves running and the file has been read to the end.

    Args:
        task_id: Which task to look at.

    Returns:
        The chunked stream, or an error reply.
    """
    result = await _conversation_path(task_id)
    if isinstance(result, Response):
        return result
    _tid, path = result

    async def _stop() -> bool:
        view = await store_module.task_view(_tid)
        return view is None or view.status != "running"

    async def _lines() -> AsyncGenerator[str, None]:
        async for line in tail_lines(path, _stop):
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
