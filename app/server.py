import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from a2wsgi import WSGIMiddleware
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Route, WebSocketRoute
from starlette.staticfiles import StaticFiles

from app.console.api import (
    API_ROUTE_METHODS,
    ConsolePathSanitizer,
    ConsoleSecurityMiddleware,
    STATIC_DIR,
    api_not_found,
    build_api_routes,
    dav_download,
)
from app.console.vnc import websockify_proxy
from app.lib.db.store import init_store
from app.lib.render import thumbs
from app.lib.shared import paths
from app.lib.task import logcapture
from app.mcp.registry import mcp
from app.resources.resources import create_dav_app
from app.tts_server import create_openai_tts_app
from app.viewer.pages import view_doc, view_doc_asset, view_file, view_presentation
from app.viewer.result_page import view_result_page

__all__ = ["build_app", "mcp"]

_RESOURCES_DIR = paths.resources_dir()


@asynccontextmanager
async def _lifespan(mcp_app: Starlette) -> AsyncIterator[None]:
    """Get the server ready before it serves, and clean it up after it stops.

    The same mcp_app object must be used for the served routes and for this
    lifespan. Any other object leaves /mcp broken, and it answers 500.

    Args:
        mcp_app: The MCP app this server serves.
    """
    inner = mcp_app.lifespan
    logcapture.install_log_tee()
    record_store = None
    try:
        record_store = init_store()
        await record_store.open()
        marked = await record_store.mark_orphans_failed("server restarted before task completion")
        if marked:
            print(f"[startup] marked {marked} orphaned running task record(s) as failed", flush=True)
        async with inner(mcp_app):
            yield
    finally:
        await thumbs.stop_thumbnails()
        if record_store is not None:
            await record_store.close()
        logcapture.restore_log_tee()


def _build_mcp_app() -> Starlette:
    return mcp.http_app(path="/mcp")


_NOVNC_DIR = "/usr/share/novnc"
_INDEX_HTML = os.path.join(STATIC_DIR, "index.html")


async def _root_redirect(_request: Request) -> RedirectResponse:
    return RedirectResponse("/tools", status_code=302)


async def _websockify_not_found(_request: Request) -> PlainTextResponse:
    return PlainTextResponse("Not Found", status_code=404)


async def _serve_index(_request: Request) -> FileResponse:
    return FileResponse(_INDEX_HTML)


def build_app() -> Starlette:
    """Build the app that serves MCP, WebDAV, TTS, and the Agent Hub console.

    Route order matters. The catch all route that serves the web page must stay
    last, so every mount and route added before it still wins. /dav/dl must stay
    before the /dav mount, the plain /api route before the /api mount, and the
    five /view routes must keep the order they are added in. The plain
    /websockify route is what makes a request that is not a WebSocket get a 404
    there instead of the web page. Every /api route lists all standard methods,
    so a bad method on a known endpoint gets a JSON 405 that names the methods
    it does take, and anything under /api with no endpoint gets a JSON 404.

    ConsolePathSanitizer must stay the outermost middleware. It takes stray line
    breaks out of the request path, and no route matches a path that still has
    them.

    Returns:
        The app to serve.
    """
    mcp_app = _build_mcp_app()
    outer = Starlette(routes=list(mcp_app.routes), lifespan=lambda _app: _lifespan(mcp_app))
    outer.add_middleware(ConsoleSecurityMiddleware)
    outer.add_middleware(ConsolePathSanitizer)
    outer.add_route("/dav/dl", dav_download, methods=("GET", "HEAD"))
    outer.mount("/dav", WSGIMiddleware(create_dav_app(_RESOURCES_DIR)))
    outer.mount("/v1", create_openai_tts_app())
    outer.add_route("/api", api_not_found, methods=API_ROUTE_METHODS)
    outer.mount("/api", build_api_routes())
    outer.mount("/assets", StaticFiles(directory=STATIC_DIR))
    outer.mount("/vnc", StaticFiles(directory=_NOVNC_DIR))
    outer.routes.append(Route("/view/presentation/{id}", view_presentation))
    outer.routes.append(Route("/view/doc/{key}/{asset:path}", view_doc_asset))
    outer.routes.append(Route("/view/doc/{key}", view_doc))
    outer.routes.append(Route("/view/result/{task_id}", view_result_page))
    outer.routes.append(Route("/view/{path:path}", view_file))
    outer.routes.append(Route("/", _root_redirect))
    outer.routes.append(Route("/websockify", _websockify_not_found))
    outer.routes.append(WebSocketRoute("/websockify", websockify_proxy))
    outer.routes.append(Route("/{path:path}", _serve_index))
    return outer

