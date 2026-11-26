"""Endpoints for viewing one resource file."""

from __future__ import annotations

import asyncio
import os

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.console.errors import error_response
from app.lib.render import cache, office, thumbs
from app.lib.render.classify import classify
from app.lib.render.descriptor import (OFFICE_KINDS, build_converting_descriptor,
                                       build_descriptor, get_resource_path, resolve_view)
from app.lib.shared import activity

__all__ = ["get_resource_view", "retry_office_view"]


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


async def get_resource_view(request: Request) -> Response:
    """Describe one resource file for the viewer.

    Args:
        request: The incoming HTTP request.

    Returns:
        The description of the file, or an error reply.
    """
    path = request.path_params["path"]
    if request.method == "GET":
        await activity.record(path)
    full = await _resolve_full(path)
    if isinstance(full, Response):
        return full
    return JSONResponse(await build_descriptor(path, full))


async def retry_office_view(request: Request) -> Response:
    """Convert one resource office file again.

    Args:
        request: The incoming HTTP request.

    Returns:
        The description of the file while it converts, or an error reply.
    """
    path = request.path_params["path"]
    full = await _resolve_full(path)
    if isinstance(full, Response):
        return full
    kind = await classify(full)
    if kind not in OFFICE_KINDS:
        return error_response(400, "bad_param", f"'{path}' is not a convertible Office file.")
    resource_path = get_resource_path(full, path)
    stat = await asyncio.to_thread(os.stat, full)
    key = cache.build_cache_key(resource_path, int(stat.st_mtime))
    if await office.retry_office_conversion(resource_path, key, full):
        thumbs.start_thumbnail_job(key, kind, resource_path)
    return JSONResponse(build_converting_descriptor(resource_path, os.path.basename(resource_path),
                                                    stat.st_size, int(stat.st_mtime), kind))
