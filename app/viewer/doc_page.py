"""Cached rendered document page and sibling asset endpoints."""

from __future__ import annotations

import asyncio
import mimetypes
import os
import re
from dataclasses import dataclass

from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, PlainTextResponse, Response

from app.lib.render import cache
from app.viewer.file_page import view_file
from app.viewer.view_chrome import STATIC_HEAD, delegate_miss

__all__ = ["view_doc", "view_doc_asset"]

_DOC_KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _DocAsset:
    """Resolved file path and media type for a saved document asset."""

    path: str
    media_type: str


def _compose_doc_page(fragment: str, *, is_thumbnail: bool) -> str:
    override = (
        '<style id="vp-thumb-override">.vp-doccol{max-width:none;margin:0;padding:16px;}</style>'
        if is_thumbnail
        else ""
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        "<title>Document</title>"
        '<script>document.documentElement.setAttribute("data-theme","light");</script>'
        f"{STATIC_HEAD}{override}</head>"
        '<body class="view-page doc-page"><main class="vp-main"><div class="vp-doccol">'
        f"{fragment}</div></main></body></html>"
    )


def _build_plain_not_found() -> Response:
    return PlainTextResponse("Not Found", status_code=404)


def _resolve_doc_asset_sync(key: str, asset_name: str) -> _DocAsset | None:
    key_directory = os.path.realpath(os.path.join(cache.get_render_root(), key))
    if not os.path.isdir(key_directory):
        return None
    full_path = os.path.realpath(os.path.join(key_directory, asset_name))
    if full_path == key_directory or not full_path.startswith(key_directory + os.sep):
        return None
    if not os.path.isfile(full_path):
        return None
    media_type = mimetypes.guess_type(full_path)[0] or "application/octet-stream"
    return _DocAsset(path=full_path, media_type=media_type)


async def view_doc(request: Request) -> Response:
    """Show a saved document page.

    Args:
        request: The request being answered.

    Returns:
        The page, or a message when there is no such page.
    """
    key = request.path_params["key"]
    is_valid_key = _DOC_KEY_PATTERN.fullmatch(key) is not None
    entry = await cache.read_doc(key) if is_valid_key else None
    if entry is None:
        return await delegate_miss(request, f"doc/{key}", _build_plain_not_found(), view_file)
    is_thumbnail = request.query_params.get("thumb") == "1"
    return HTMLResponse(_compose_doc_page(entry.html, is_thumbnail=is_thumbnail))


async def view_doc_asset(request: Request) -> Response:
    """Send a file that belongs to a saved document page.

    Args:
        request: The request being answered.

    Returns:
        The file, or a message when there is no such file.
    """
    key = request.path_params["key"]
    asset_name = request.path_params["asset"]
    is_valid_key = _DOC_KEY_PATTERN.fullmatch(key) is not None
    asset = await asyncio.to_thread(_resolve_doc_asset_sync, key, asset_name) if is_valid_key else None
    if asset is None:
        return await delegate_miss(request, f"doc/{key}/{asset_name}", _build_plain_not_found(), view_file)
    return FileResponse(asset.path, media_type=asset.media_type)
