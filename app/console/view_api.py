"""Endpoints for viewing one shared file."""

from __future__ import annotations

import asyncio
import os

from starlette.responses import JSONResponse, Response

from app.console.errors import error_response
from app.lib.render import cache, office
from app.lib.render.classify import classify
from app.lib.render.descriptor import (OFFICE_KINDS, build_descriptor,
                                       converting_descriptor, resolve_view, share_path_for)

__all__ = ["retry_view_impl", "view_impl"]


async def _resolve_full(path: str) -> str | Response:
    """Turn a shared path into a full path on disk.

    Args:
        path: A path inside the shared folder.

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
    """Describe one shared file for the viewer.

    Args:
        path: A path inside the shared folder.

    Returns:
        The description of the file, or an error reply.
    """
    full = await _resolve_full(path)
    if isinstance(full, Response):
        return full
    return JSONResponse(await build_descriptor(path, full))


async def retry_view_impl(path: str) -> Response:
    """Convert one shared office file again.

    Args:
        path: A path inside the shared folder.

    Returns:
        The description of the file while it converts, or an error reply.
    """
    full = await _resolve_full(path)
    if isinstance(full, Response):
        return full
    kind = await classify(full)
    if kind not in OFFICE_KINDS:
        return error_response(400, "bad_param", f"'{path}' is not a convertible Office file.")
    share_path = share_path_for(full, path)
    stat = await asyncio.to_thread(os.stat, full)
    key = cache.key_for(share_path, int(stat.st_mtime))
    await office.retry_office_conversion(share_path, key, full)
    return JSONResponse(converting_descriptor(share_path, os.path.basename(share_path),
                                              stat.st_size, int(stat.st_mtime), kind))
