"""Middleware, system routes, and router wiring for the console API."""

from __future__ import annotations

import inspect
import os
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, Router
from starlette.types import ASGIApp, Receive, Scope, Send

from app.console import toolcall
from app.console.conversation_api import (
    get_conversation_snapshot,
    get_conversation_window,
    stream_conversation,
)
from app.console.errors import error_response
from app.console.presentations_api import list_presentations
from app.console.render_api import render_message
from app.console.resources_api import list_resources, search_resources
from app.console.result_api import get_result_feed, get_result_view
from app.console.tasks_api import (
    cancel_task,
    download_task_log,
    download_task_result,
    get_changes,
    get_task_detail,
    get_task_log,
    get_task_result,
    list_tool_tasks,
    list_tools,
)
from app.console.thumb_api import get_thumbnail, queue_directory_thumbnails
from app.console.view_api import get_resource_view, retry_office_view
from app.console.vnc import vnc_reachable
from app.lib.db.store import is_store_open
from app.resources.resources import dav_base_url

__all__ = [
    "API_ROUTE_METHODS",
    "ConsolePathSanitizer",
    "ConsoleSecurityMiddleware",
    "STATIC_DIR",
    "api_not_found",
    "build_api_routes",
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
_API_METHODS = ("GET", "HEAD", "POST", "PUT", "DELETE", "PATCH")
_GET_ONLY_METHODS = ("GET", "HEAD")
API_ROUTE_METHODS = _API_METHODS + ("OPTIONS", "CONNECT", "TRACE")


def _is_spa_path(path: str) -> bool:
    if path in ("/websockify", "/api"):
        return False
    return not path.startswith(tuple(prefix + "/" for prefix in _NON_SPA_PREFIXES))


def _is_vnc_static_path(path: str) -> bool:
    return path.startswith("/vnc/")


def _is_skip_path(path: str) -> bool:
    return path in _SKIP_PREFIXES or path.startswith(tuple(prefix + "/" for prefix in _SKIP_PREFIXES))


class ConsolePathSanitizer:
    """Remove line breaks from a request path before routing."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path = scope["path"]
            if "\n" in path or "\r" in path:
                clean_path = path.replace("\n", "").replace("\r", "")
                scope["path"] = clean_path
                scope["raw_path"] = clean_path.encode("utf-8")
        await self.app(scope, receive, send)


class ConsoleSecurityMiddleware(BaseHTTPMiddleware):
    """Add security and cache headers to console responses."""

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
    """Answer an API path that has no matching endpoint.

    Args:
        request: The incoming HTTP request.

    Returns:
        The JSON 404 response.
    """
    return error_response(404, "not_found", f"No API endpoint at '{request.url.path}'.")


def _guard_http_methods(
    endpoint: Callable[[Request], Response | Awaitable[Response]],
    allowed_methods: tuple[str, ...],
) -> Callable[[Request], Awaitable[Response]]:
    async def guarded(request: Request) -> Response:
        if request.method not in allowed_methods:
            return error_response(
                405,
                "method_not_allowed",
                f"Method '{request.method}' is not allowed at '{request.url.path}'.",
                headers={"allow": ", ".join(allowed_methods)},
            )
        result = endpoint(request)
        if inspect.isawaitable(result):
            return await result
        return result

    return guarded


async def _get_health(_request: Request) -> JSONResponse:
    if not is_store_open():
        return error_response(503, "internal", "Record store is not open.")
    return JSONResponse({"status": "ok", "version": _VERSION, "dav_base_url": dav_base_url()})


async def _get_vnc_status(_request: Request) -> JSONResponse:
    try:
        is_reachable = await vnc_reachable()
    except Exception:
        return error_response(500, "internal", "VNC probe failed.")
    return JSONResponse({"reachable": is_reachable})


def build_api_routes() -> Router:
    """Build the router for all endpoints under /api.

    Returns:
        The router that holds all API routes.
    """
    routes: list[Route] = [
        Route("/health", _guard_http_methods(_get_health, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tools", _guard_http_methods(list_tools, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tools/{tool}/tasks", _guard_http_methods(list_tool_tasks, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tools/{tool}/schema", _guard_http_methods(toolcall.get_tool_schema, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tools/{tool}/call", _guard_http_methods(toolcall.call_tool, ("POST",)), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}", _guard_http_methods(get_task_detail, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/conversation/stream", _guard_http_methods(stream_conversation, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/conversation/snapshot", _guard_http_methods(get_conversation_snapshot, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/conversation", _guard_http_methods(get_conversation_window, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/log", _guard_http_methods(get_task_log, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/result", _guard_http_methods(get_task_result, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/log/download", _guard_http_methods(download_task_log, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/result/download", _guard_http_methods(download_task_result, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/result/view", _guard_http_methods(get_result_view, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/result/feed", _guard_http_methods(get_result_feed, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/changes", _guard_http_methods(get_changes, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/tasks/{task_id}/cancel", _guard_http_methods(cancel_task, ("POST",)), methods=API_ROUTE_METHODS),
        Route("/resources", _guard_http_methods(list_resources, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/resources/search", _guard_http_methods(search_resources, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/vnc/status", _guard_http_methods(_get_vnc_status, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/view/{path:path}/retry", _guard_http_methods(retry_office_view, ("POST",)), methods=("POST",)),
        Route("/view/{path:path}", _guard_http_methods(get_resource_view, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/resources/thumb", _guard_http_methods(get_thumbnail, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/resources/thumbs", _guard_http_methods(queue_directory_thumbnails, ("POST",)), methods=API_ROUTE_METHODS),
        Route("/presentations", _guard_http_methods(list_presentations, _GET_ONLY_METHODS), methods=API_ROUTE_METHODS),
        Route("/render/message", _guard_http_methods(render_message, ("POST",)), methods=API_ROUTE_METHODS),
        Route("/{path:path}", api_not_found, methods=API_ROUTE_METHODS),
    ]
    return Router(routes=routes)
