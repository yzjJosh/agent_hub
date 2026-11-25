"""Endpoints for viewing one resource file."""

from __future__ import annotations

import asyncio
import os

from starlette.responses import JSONResponse, Response

from app.console.errors import error_response
from app.lib.render import cache, office, thumbs
from app.lib.render.classify import classify
from app.lib.render.descriptor import (OFFICE_KINDS, build_descriptor,
                                       converting_descriptor, get_resource_path, resolve_view)

__all__ = ["retry_view_impl", "view_impl"]


async def _resolve_full(path: str) -> str | Response:
    """Turn a resource path into a full path on disk.

    Args:
        path: A path inside the resources folder.

    Returns:
        The full path. An error reply comes back when the path is not allowed
        or there is no file there.
    """
    full = await resolve_view(path)
    if full is None:
        return error_response(400, "bad_param", f"Invalid path: '{path}'.")
    if not await asyncio.to_thread(os.path.isfile, full):
        return error_response(404, "not_found", f"No such file: '{path}'.")
    return full


async def view_impl(path: str) -> Response:
    """Describe one resource file for the viewer.

    Args:
        path: A path inside the resources folder.

    Returns:
        The description of the file, or an error reply.
    """
    full = await _resolve_full(path)
    if isinstance(full, Response):
        return full
    return JSONResponse(await build_descriptor(path, full))


async def retry_view_impl(path: str) -> Response:
    """Convert one resource office file again.

    Args:
        path: A path inside the resources folder.

    Returns:
        The description of the file while it converts, or an error reply.
    """
    full = await _resolve_full(path)
    if isinstance(full, Response):
        return full
    kind = await classify(full)
    if kind not in OFFICE_KINDS:
        return error_response(400, "bad_param", f"'{path}' is not a convertible Office file.")
    resource_path = get_resource_path(full, path)
    stat = await asyncio.to_thread(os.stat, full)
    key = cache.key_for(resource_path, int(stat.st_mtime))
    if await office.retry_office_conversion(resource_path, key, full):
        thumbs.start_thumbnail_job(key, kind, resource_path)
    return JSONResponse(converting_descriptor(resource_path, os.path.basename(resource_path),
                                              stat.st_size, int(stat.st_mtime), kind))
