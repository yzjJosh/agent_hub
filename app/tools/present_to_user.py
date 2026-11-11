from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import ClassVar, override
from urllib.parse import quote

import app.lib.db.store as store_module
from app.lib.shared.paths import resolve_path_sync, resources_dir
from app.lib.render import EXT_TO_KIND, RenderKind, cache, classify, language_extension, thumbs
from app.lib.text_type.detector import identify, kind_for_label, language_for_label
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult
from app.resources.resources import dav_base_url

__all__ = ["present_to_user_tool"]

_logger = logging.getLogger(__name__)

_KIND_EXTENSIONS_TARGETS = frozenset({
    RenderKind.MARKDOWN, RenderKind.CSV, RenderKind.JSON, RenderKind.JSONL,
    RenderKind.YAML, RenderKind.XML, RenderKind.HTML,
})


def _build_kind_extensions() -> dict[RenderKind, str]:
    """Return the file extension to use for each text kind."""
    table: dict[RenderKind, str] = {}
    for ext, kind in EXT_TO_KIND.items():
        if kind in _KIND_EXTENSIONS_TARGETS and kind not in table:
            table[kind] = ext
    return table


_KIND_EXTENSIONS: dict[RenderKind, str] = _build_kind_extensions()

# The file extension for each code label that has no extension of its own.
_CODE_EXTENSIONS: dict[str, str] = {
    "c": "c",
    "cpp": "cpp",
    "cs": "cs",
    "erlang": "erl",
    "haskell": "hs",
    "java": "java",
    "kotlin": "kt",
    "lua": "lua",
    "php": "php",
    "r": "r",
    "ruby": "rb",
    "rust": "rs",
    "scala": "scala",
    "swift": "swift",
    "toml": "toml",
    "typescript": "ts",
}


def _content_extension(label: str) -> str | None:
    """Return the file extension for a detected label, or None when it is unknown."""
    kind = kind_for_label(label)
    if kind is RenderKind.CODE:
        return language_extension(language_for_label(label)) or _CODE_EXTENSIONS.get(label)
    if kind is RenderKind.TEXT:
        return "txt" if label == "txt" else None
    return _KIND_EXTENSIONS.get(kind)


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


def _write_atomic(target: str, content: str) -> None:
    """Write text to a file so a reader never sees a half written file."""
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    staging = os.path.join(
        directory,
        f".staging-{os.path.basename(target)}-{os.getpid()}-{threading.get_ident()}",
    )
    try:
        with open(staging, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(staging, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(staging)
        raise


def _validate_files(files: list[str]) -> list[tuple[str, str]]:
    """Check that every file is in the share.

    This blocks, so run it in a thread.

    Args:
        files: Paths of files in the share.

    Returns:
        The given path and the full path for each file.

    Raises:
        ValueError: If a path is not in the share or is a folder.
    """
    resolved: list[tuple[str, str]] = []
    for path in files:
        target = resolve_path_sync(resources_dir(), path)
        if target is None or not (os.path.isfile(target) or os.path.isdir(target)):
            raise ValueError(_not_found_error(path))
        resolved.append((path, target))
    for path, target in resolved:
        if os.path.isdir(target):
            raise ValueError(_folder_error(path))
    return resolved


def _resolve_and_stat(root: str, rel: str) -> tuple[str, int]:
    """Find where a presented file is and when it last changed."""
    full = resolve_path_sync(root, rel)
    return full, int(os.stat(full).st_mtime)


async def _thumb_jobs(presented: list[str]) -> list[tuple[str, RenderKind, str]]:
    """Return the key, the file type, and the path for each presented file."""
    root = resources_dir()
    jobs: list[tuple[str, RenderKind, str]] = []
    for rel in presented:
        full, mtime = await asyncio.to_thread(_resolve_and_stat, root, rel)
        jobs.append((cache.key_for(rel, mtime), await classify(full), rel))
    return jobs


class _PresentToUserTool(Tool):
    name: ClassVar[str] = "present_to_user"
    tracked: ClassVar[bool] = False
    description: ClassVar[str] = (
        "Present files and/or text content to the user via this server's /view "
        "surface. Use to deliver a result or artifact to the user."
    )
    summary: ClassVar[str] = "Present files or text to the user via the /view surface"
    params: ClassVar[tuple[Param, ...]] = (
        Param(
            "title",
            str,
            description=(
                "A short human-facing name for this presentation, shown in "
                "the console's Presented by Agent list."
            ),
        ),
        Param(
            "files",
            list[str] | None,
            None,
            kind="resource_list",
            description=(
                "List of file paths relative to the resource root, e.g. 'upload/cat.jpg'. "
                "At least one of files or content is required."
            ),
        ),
        Param(
            "content",
            str | None,
            None,
            kind="longtext",
            description=(
                "Text to present. Type-detected locally and materialized as a file "
                "of the detected type in the resource, so it renders with the matching viewer "
                "treatment. Written as plain text when detection is unavailable. May be an "
                "empty string when files are provided. At least one of files or content is required."
            ),
        ),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        title: str = params["title"]
        files: list[str] = params.get("files") or []
        content: str = params.get("content") or ""
        if not files and not content:
            raise ToolMessageError("Error: at least one of files or content is required")

        resolved = await asyncio.to_thread(_validate_files, files)

        token = uuid.uuid4().hex[:16]
        presented = [path for path, _target in resolved]
        if content:
            label = await identify(content.encode("utf-8"))
            content_path = f"presentations/{token}.{_content_extension(label) or 'md'}"
            await asyncio.to_thread(
                _write_atomic,
                os.path.join(resources_dir(), content_path),
                content,
            )
            presented.append(content_path)

        base = dav_base_url()
        if len(presented) == 1:
            encoded = quote(presented[0], safe="/")
            url_path = f"/view/{encoded}"
            result = (
                f"Presented to the user at {base}{url_path} "
                "· link valid until the file is garbage-collected (30 days)"
            )
            try:
                await store_module.presentation_store.insert(
                    token,
                    datetime.now(timezone.utc).isoformat(),
                    url_path,
                    presented,
                    title=title,
                )
            except Exception:
                _logger.warning("failed to write presentation row for solo presentation %s", token)
        else:
            url_path = f"/view/presentation/{token}"
            result = (
                f"Presented to the user at {base}{url_path} "
                f"· {len(presented)} file(s) "
                "· links valid until the files are garbage-collected (30 days)"
            )
            await store_module.presentation_store.insert(
                token,
                datetime.now(timezone.utc).isoformat(),
                url_path,
                presented,
                title=title,
            )

        for key, kind, rel in await _thumb_jobs(presented):
            thumbs.start_thumbnail_job(key, kind, rel)
        return ToolResult(text=result)


present_to_user_tool = _PresentToUserTool()
