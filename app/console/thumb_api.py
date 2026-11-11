"""Endpoints for the small pictures of shared files."""

from __future__ import annotations

import asyncio
import os

from starlette.responses import FileResponse, JSONResponse, Response

from app.console.errors import error_response
from app.lib.shared.paths import resolve_path, resources_dir
from app.lib.render import cache, thumb_engine, thumbs
from app.lib.render.classify import classify

__all__ = ["thumb_impl", "thumbs_impl"]


async def thumb_impl(path: str | None) -> Response:
    """Send the small picture of a shared file.

    A file whose picture is not made yet gets one lined up and comes back as
    not ready, so the caller can ask again. A failed document picture is
    lined up again once its page is published again, but a failed media
    picture is left alone. Only a missing file comes back as not found.

    Args:
        path: A path inside the shared folder.

    Returns:
        The picture, or an error reply.
    """
    if path is None or "\x00" in path:
        return error_response(400, "bad_param", "path is required.")
    rel = path.strip("/")
    full = await resolve_path(resources_dir(), rel)
    if full is None or not await asyncio.to_thread(os.path.isfile, full):
        return error_response(404, "not_found", f"No such file: '{rel}'.")
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


async def thumbs_impl(path: str | None) -> Response:
    """Line up the small pictures for one shared folder.

    Args:
        path: A folder inside the shared folder.

    Returns:
        How many pictures were lined up, or an error reply.
    """
    if path is None or "\x00" in path:
        return error_response(400, "bad_param", "path is required.")
    rel = path.strip("/")
    full = await resolve_path(resources_dir(), rel)
    if full is None or not await asyncio.to_thread(os.path.isdir, full):
        return error_response(404, "not_found", f"No such directory: '{rel or '/'}'.")
    queued = await thumbs.start_dir_thumbnails(rel)
    return JSONResponse({"queued": queued})
