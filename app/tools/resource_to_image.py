from __future__ import annotations

import asyncio
import os
from typing import ClassVar, override

from app.lib.shared.paths import resolve_path_sync, resources_dir
from app.lib.render import RenderKind, cache, classify, thumb_engine, thumbs
from app.lib.tool.base import Param, Tool, ToolResult
from app.resources.resources import dav_base_url

__all__ = ["resource_to_image_tool"]

_THUMB_FILENAME_FALLBACK = "thumb.png"
_RETRY_LINE = (
    "While the thumbnail is still generating, this URL returns HTTP 409 — "
    "retry until it returns the 1024x640 PNG."
)


def _not_found_error(path: str) -> str:
    """Return the error text for a path that is not in the share."""
    return (
        f"Resource not found: {path}. The resource path must already "
        "exist in this server's resources share. Call the send_or_receive_file "
        "tool for instructions to upload the file, then retry this tool with "
        "the uploaded file's resource path (e.g. 'upload/<filename>')."
    )


def _folder_error(path: str) -> str:
    """Return the error text for a path that is a folder."""
    return (
        f"'{path}' is a folder, not a file. Pass the resource path of a "
        "single file instead, e.g. 'upload/<filename>'."
    )


def _source_download_line(path: str) -> str:
    """Return the command that downloads the file itself."""
    return f"  curl -o '/tmp/{os.path.basename(path)}' {dav_base_url()}/dav/{path}"


def _not_previewable_error(path: str, kind: RenderKind) -> str:
    """Return the error text for a file type that has no preview thumbnail."""
    return (
        f"'{path}' cannot be previewed as an image: file type '{kind.value}' "
        "has no preview thumbnail. The file itself remains downloadable:\n"
        f"{_source_download_line(path)}"
    )


def _failed_error(path: str) -> str:
    """Return the error text for a thumbnail that could not be made.

    A thumbnail that failed is never made again.
    """
    return (
        f"Preview thumbnail for '{path}' has failed to generate and is not "
        "retried automatically. The source file remains downloadable:\n"
        f"{_source_download_line(path)}"
    )


def _thumb_filename(path: str) -> str:
    """Return the name to save the thumbnail as."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{stem}.png" if stem else _THUMB_FILENAME_FALLBACK


def _success_message(path: str, url: str, ready: bool) -> str:
    """Return the message with the thumbnail URL and how to download it.

    When the thumbnail is not done yet, the message says to try the URL again.
    """
    lines = [
        f"Preview thumbnail URL for {path}:",
        f"  {url}",
        "Download it with:",
        f"  curl -o '/tmp/{_thumb_filename(path)}' {url}",
    ]
    if not ready:
        lines.append(_RETRY_LINE)
    return "\n".join(lines)


def _thumb_target(path: str) -> tuple[str, int]:
    """Find where a file in the share is and when it last changed.

    Args:
        path: Path of the file in the share.

    Returns:
        Where the file is on disk and when it last changed.

    Raises:
        ValueError: If the path is not in the share or is a folder.
    """
    target = resolve_path_sync(resources_dir(), path)
    if target is None or not (os.path.isfile(target) or os.path.isdir(target)):
        raise ValueError(_not_found_error(path))
    if os.path.isdir(target):
        raise ValueError(_folder_error(path))
    return target, int(os.stat(target).st_mtime)


async def _resolve_thumb(path: str) -> tuple[RenderKind, str, str]:
    """Look up the preview thumbnail for a file in the share.

    Args:
        path: Path of the file in the share.

    Returns:
        The file type, the thumbnail key, and how far along the thumbnail is.

    Raises:
        ValueError: If the path is not in the share or is a folder.
    """
    target, mtime = await asyncio.to_thread(_thumb_target, path)
    kind = await classify(target)
    key = cache.key_for(path, mtime)
    return kind, key, await cache.thumb_status(key)


class _ResourceToImageTool(Tool):
    name: ClassVar[str] = "resource_to_image"
    tracked: ClassVar[bool] = False
    description: ClassVar[str] = (
        "Return the preview thumbnail image URL for an existing resource. "
        "The original file is not affected."
    )
    summary: ClassVar[str] = "URL of the resource file's 1024x640 preview thumbnail"
    params: ClassVar[tuple[Param, ...]] = (
        Param(
            "resource_path",
            str,
            kind="resource",
            description="Path of the file relative to the resource root, e.g. 'upload/report.pdf'.",
        ),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        resource_path: str = params["resource_path"]
        kind, key, status = await _resolve_thumb(resource_path)
        if thumb_engine(kind) is None:
            raise ValueError(_not_previewable_error(resource_path, kind))
        if status == "failed":
            raise ValueError(_failed_error(resource_path))
        # A thumbnail that is already being made must not be asked for again.
        if status == "pending":
            thumbs.start_thumbnail_job(key, kind, resource_path)
        url = dav_base_url() + cache.thumb_url(resource_path)
        return ToolResult(text=_success_message(resource_path, url, ready=(status == "done")))


resource_to_image_tool = _ResourceToImageTool()
