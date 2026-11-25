"""Endpoint for listing recent file presentations."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

import app.lib.db.store as store_module
from app.console.errors import error_response
from app.lib.db.store import Presentation, store_is_open
from app.lib.render import cache
from app.lib.render.classify import classify, kind_label
from app.lib.shared.paths import resolve_path_sync, resources_dir
from app.lib.shared.timefmt import build_generated_timestamps, iso_to_local
from app.resources.resources import dav_base_url

__all__ = ["list_presentations"]


@dataclass(frozen=True)
class _PresentedFileInfo:
    """Resolved path and file stats for a presented file."""

    full_path: str
    stat_result: os.stat_result


def _inspect_presented_file(root_dir: str, relative_path: str) -> _PresentedFileInfo | None:
    full_path = resolve_path_sync(root_dir, relative_path)
    if full_path is None or not os.path.isfile(full_path):
        return None
    return _PresentedFileInfo(full_path=full_path, stat_result=os.stat(full_path))


async def _describe_presented_file(root_dir: str, relative_path: str) -> dict:
    file_name = os.path.basename(relative_path)
    file_info = await asyncio.to_thread(_inspect_presented_file, root_dir, relative_path)
    if file_info is not None:
        modified_time = int(file_info.stat_result.st_mtime)
        kind = await classify(file_info.full_path)
        thumb_status = await cache.thumb_status(cache.key_for(relative_path, modified_time))
        return {
            "path": relative_path,
            "name": file_name,
            "size": file_info.stat_result.st_size,
            "mtime": modified_time,
            "present": True,
            "kind": kind_label(kind, file_name, None),
            "thumb": {"status": thumb_status, "url": cache.thumb_url(relative_path)},
        }
    kind = await classify(os.path.join(root_dir, relative_path))
    return {
        "path": relative_path,
        "name": file_name,
        "size": None,
        "mtime": None,
        "present": False,
        "kind": kind_label(kind, file_name, None),
        "thumb": {"status": "none", "url": cache.thumb_url(relative_path)},
    }


async def _describe_presentation(root_dir: str, presentation: Presentation) -> dict:
    files = [await _describe_presented_file(root_dir, relative_path) for relative_path in presentation.files]
    present_count = sum(1 for file_item in files if file_item["present"])
    total_bytes = sum(file_item["size"] for file_item in files if file_item["size"] is not None)
    if present_count == len(presentation.files):
        state = "active"
    elif present_count == 0:
        state = "expired"
    else:
        state = "partial"
    return {
        "id": presentation.id,
        "title": presentation.title,
        "presented_at": presentation.presented_at,
        "presented_at_local": iso_to_local(presentation.presented_at),
        "url": dav_base_url() + presentation.url_path,
        "kind": "solo" if len(presentation.files) == 1 else "package",
        "files": files,
        "file_count": len(presentation.files),
        "present_count": present_count,
        "total_size": total_bytes,
        "state": state,
    }


async def list_presentations(_request: Request) -> Response:
    """List recent presentations newest first with current file state.

    Args:
        _request: The incoming HTTP request.

    Returns:
        The presentations list and current timestamps, or an error reply.
    """
    if not store_is_open():
        return error_response(503, "internal", "Record store is not open.")
    presentations = await store_module.presentation_store.list_newest()
    root_dir = resources_dir()
    rows = [await _describe_presentation(root_dir, presentation) for presentation in presentations]
    timestamps = build_generated_timestamps()
    return JSONResponse(
        {
            "generated_at": timestamps.utc_iso,
            "generated_at_local": timestamps.local_display,
            "presentations": rows,
        }
    )
