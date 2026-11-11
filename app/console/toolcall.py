"""Endpoints for running one tool by hand."""
from __future__ import annotations

import json
import re
import sys
from typing import get_args, get_origin

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

import app.lib.db.store as store_module
from app.console.errors import error_response
from app.lib.shared.timefmt import iso_to_local
from app.lib.task import ImmediateTask, Task
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult, format_staleness
from app.mcp.registry import tool_class

__all__ = ["tool_call_impl", "tool_schema_impl"]

_MAX_BODY = 10 * 2**20


def _unknown_tool(tool: str) -> JSONResponse:
    return error_response(404, "not_found", f"Unknown tool '{tool}'.")


def _unwrapped(t: type) -> type:
    args = get_args(t)
    if not args or get_origin(t) is list:
        return t
    non_null = [a for a in args if a is not type(None)]
    return non_null[0] if len(non_null) == 1 else t


def _type_name(t: type) -> str:
    if t is str:
        return "string"
    if t is bool:
        return "boolean"
    if t is int:
        return "integer"
    if t is float:
        return "number"
    if get_origin(t) is list:
        return "array"
    raise ValueError(f"unsupported type: {t!r}")


def _scalar_type(p: Param) -> str:
    t = _unwrapped(p.type)
    try:
        return _type_name(t)
    except ValueError:
        raise ValueError(f"unsupported Param.type for {p.name}: {t!r}") from None


def _item_type(p: Param) -> str | None:
    t = _unwrapped(p.type)
    if get_origin(t) is not list:
        return None
    items = get_args(t)
    if len(items) != 1:
        return None
    try:
        return _type_name(items[0])
    except ValueError:
        return None


def _normalize_description(text: str) -> str:
    """Clean up one parameter description.

    Args:
        text: The description to clean.

    Returns:
        The cleaned description on one line.
    """
    return re.sub(r"\s+", " ", text.replace("``", "")).strip()


def _normalize_full_description(text: str) -> str:
    """Clean up a tool's description.

    Line breaks and the spacing that lines entries up are kept.

    Args:
        text: The description to clean.

    Returns:
        The cleaned description.
    """
    lines = [ln.rstrip() for ln in text.replace("``", "").replace("\r\n", "\n").split("\n")]
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line[:1].isspace():
            out.append(line)
            i += 1
            continue
        run: list[str] = []
        while i < len(lines) and lines[i][:1].isspace():
            run.append(lines[i])
            i += 1
        base = min(len(ln) - len(ln.lstrip()) for ln in run)
        for ln in run:
            if len(ln) - len(ln.lstrip()) > base:
                joined = re.sub(r" {2,}", " ", ln.strip())
                if out:
                    out[-1] += " " + joined
                else:
                    out.append(ln)
            else:
                out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip("\n")


def _param_json(p: Param) -> dict:
    return {
        "name": p.name,
        "type": _scalar_type(p),
        "item_type": _item_type(p),
        "required": p.required,
        "default": None if p.required else p.default,
        "enum": list(p.enum) if p.enum is not None else None,
        "kind": p.kind,
        "min": p.min,
        "max": p.max,
        "step": p.step,
        "description": _normalize_description(p.description),
    }


def _exposed_tool_or_error(tool: str) -> type[Tool] | JSONResponse:
    """Look up a tool the console is allowed to show.

    Args:
        tool: The tool name.

    Returns:
        The tool, or an error reply when the tool is unknown or is not shown
        in the console.
    """
    cls = tool_class(tool)
    if cls is None:
        return _unknown_tool(tool)
    if not cls.tracked:
        return error_response(403, "not_exposed", f"Tool '{tool}' is not exposed in the console.")
    return cls


def tool_schema_impl(tool: str) -> Response:
    """Describe one tool and its parameters for the console.

    Args:
        tool: The tool name.

    Returns:
        The tool's name, words, settings, and parameters, or an error reply.
    """
    cls = _exposed_tool_or_error(tool)
    if isinstance(cls, JSONResponse):
        return cls
    return JSONResponse(
        {
            "tool": tool,
            "summary": cls.summary,
            "description": _normalize_full_description(cls.description),
            "is_slow": cls.is_slow,
            "staleness": format_staleness(cls.staleness) if cls.staleness is not None else None,
            "cached": cls.staleness is not None,
            "rules": list(cls.rules),
            "params": [_param_json(p) for p in cls.params],
        }
    )


def _is_json_media_type(request: Request) -> bool:
    return request.headers.get("content-type", "").split(";", 1)[0].strip().lower() == "application/json"


def _origin_host_mismatch(request: Request) -> bool:
    origin = request.headers.get("origin")
    if origin is None:
        return False
    try:
        host = origin.split("://", 1)[1].split("/", 1)[0].lower()
    except IndexError:
        return True
    # The console is reached on a port, and the origin carries that port, so
    # the port has to match too.
    return host != request.url.netloc.lower()


async def _read_json_capped(request: Request, cap: int) -> dict | str | None:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > cap:
            return None
    try:
        data = json.loads(bytes(body))
    except (ValueError, UnicodeDecodeError):
        return "invalid"
    return "not_object" if not isinstance(data, dict) else data


async def _run_response(task_id: str) -> Response:
    """Build the reply for a call whose run started.

    Args:
        task_id: The id of the run.

    Returns:
        The task id and when it started, or an error reply.
    """
    view = await store_module.task_view(task_id)
    if view is None:
        return error_response(500, "internal", "The tool call did not start a run.")
    body = {"task_id": task_id, "status": view.status,
            "started_at": view.started_at, "started_at_local": iso_to_local(view.started_at)}
    return JSONResponse(body, status_code=202)


async def _task_response(task: Task[ToolResult]) -> Response:
    """Build the reply for a call from the handle it returned.

    A call that started a run, new or reused, answers 202. A bad param set
    answers 400 with the tool's own words.

    Args:
        task: The handle the tool returned.

    Returns:
        The reply to send.
    """
    if isinstance(task, ImmediateTask) and not task.has_stored_run:
        try:
            await task.result()
        except ToolMessageError as exc:
            return error_response(400, "bad_param", str(exc))
    return await _run_response(str(task.id))


async def tool_call_impl(request: Request, tool: str) -> Response:
    """Run one tool by hand.

    A bad param set answers 400 with the tool's own words. Everything else
    started, or reused, a run and answers 202.

    Args:
        request: The incoming request.
        tool: The tool name.

    Returns:
        The task id of the run, or an error reply.
    """
    if not _is_json_media_type(request) or _origin_host_mismatch(request):
        return error_response(415, "unsupported_media_type", "The request must be JSON with an Origin matching this host.")
    cls = _exposed_tool_or_error(tool)
    if isinstance(cls, JSONResponse):
        return cls

    body = await _read_json_capped(request, _MAX_BODY)
    if body is None:
        return error_response(413, "body_too_large", "Request body too large (max 10 MB).")
    if body == "invalid":
        return error_response(400, "bad_param", "The request body must be valid JSON.")
    if body == "not_object":
        return error_response(400, "bad_param", "The request body must be a JSON object of params.")

    try:
        return await _task_response(await cls().run(body))
    except Exception as exc:
        print(f"[toolcall] {tool}: {exc!r}", file=sys.stderr, flush=True)
        return error_response(500, "internal", "The tool call failed.")
