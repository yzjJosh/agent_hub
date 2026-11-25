"""Endpoints for the small pictures of resource files."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from app.console.errors import error_response
from app.lib.shared.paths import resolve_path, resources_dir
from app.lib.render import cache, thumb_engine, thumbs
from app.lib.render.classify import classify

__all__ = ["get_thumbnail", "queue_directory_thumbnails"]


@dataclass(frozen=True)
class _ResolvedResourcePath:
    """Normalized resource path and its resolved disk location."""

    relative_path: str
    full_path: str


async def _resolve_resource_query(request: Request, *, is_directory: bool) -> _ResolvedResourcePath | Response:
    raw_path = request.query_params.get("path")
    if raw_path is None or "\x00" in raw_path:
        return error_response(400, "bad_param", "path is required.")
    relative_path = raw_path.strip("/")
    full_path = await resolve_path(resources_dir(), relative_path)
    check = os.path.isdir if is_directory else os.path.isfile
    if full_path is None or not await asyncio.to_thread(check, full_path):
        if is_directory:
            return error_response(404, "not_found", f"No such directory: '{relative_path or '/'}'.")
        return error_response(404, "not_found", f"No such file: '{relative_path}'.")
    return _ResolvedResourcePath(relative_path=relative_path, full_path=full_path)


async def get_thumbnail(request: Request) -> Response:
    """Send the small picture of a resource file.

    A file whose picture is not made yet gets one lined up and comes back as
    not ready, so the caller can ask again. A failed document picture is
    lined up again once its page is published again, but a failed media
    picture is left alone. Only a missing file comes back as not found.

    Args:
        request: The incoming HTTP request.

    Returns:
        The picture, or an error reply.
    """
    resolved = await _resolve_resource_query(request, is_directory=False)
    if isinstance(resolved, Response):
        return resolved
    rel = resolved.relative_path
    full = resolved.full_path
    st = await asyncio.to_thread(os.stat, full)
    key = cache.key_for(rel, int(st.st_mtime))
    thumb_path = os.path.join(cache.render_root(), key, "thumb.png")
    if await asyncio.to_thread(os.path.exists, thumb_path):
        return FileResponse(thumb_path, media_type="image/png")
    kind = await classify(full)
    status = await cache.thumb_status(key)
    if status == "pending":
        thumbs.start_thumbnail_job(key, kind, rel)
    # A document whose page is published again is worth one more try, so a
    # stale failure mark is re-queued. Media is not retried on every poll.
    elif status == "failed" and thumb_engine(kind) == "doc" and await cache.read_doc(key) is not None:
        thumbs.start_thumbnail_job(key, kind, rel)
    return error_response(409, "not_ready", f"No thumbnail yet for '{rel}'.")


async def queue_directory_thumbnails(request: Request) -> Response:
    """Line up the small pictures for one resource folder.

    Args:
        request: The incoming HTTP request.

    Returns:
        How many pictures were lined up, or an error reply.
    """
    resolved = await _resolve_resource_query(request, is_directory=True)
    if isinstance(resolved, Response):
        return resolved
    queued = await thumbs.start_dir_thumbnails(resolved.relative_path)
    return JSONResponse({"queued": queued})
