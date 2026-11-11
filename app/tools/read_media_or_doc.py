from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import timedelta
from typing import ClassVar, override

from app.lib.resolver.resolver import ResolveError, resolve_input
from app.lib.shared.format import answer_or_content
from app.lib.shared.paths import resolve_path, resolve_path_sync, resources_dir
from app.lib.tool.base import Param, Tool, ToolResult

__all__ = ["read_media_or_doc_tool"]


def _hash_file(target: str) -> str:
    """Return a hash of a file's bytes.

    This blocks, so run it in a thread.
    """
    with open(target, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _resolve_resource_file(resource_path: str) -> str:
    """Find a file of the resource on disk.

    This blocks, so run it in a thread.

    Args:
        resource_path: Path of the file in the resource.

    Returns:
        The path of the file on disk.

    Raises:
        ValueError: If the path is a folder or is not in the resource.
    """
    target = resolve_path_sync(resources_dir(), resource_path)
    if target is not None and os.path.isdir(target):
        raise ValueError(
            f"'{resource_path}' is a folder, not a file. Pass the resource path of "
            "a single media or document file instead, e.g. 'upload/<filename>'."
        )
    if target is None or not os.path.isfile(target):
        raise ValueError(
            f"Resource not found: {resource_path}. The resource path must already "
            "exist in this server's resources share. Call the send_or_receive_file "
            "tool for instructions to upload the file, then retry this tool with "
            "the uploaded file's resource path (e.g. 'upload/<filename>')."
        )
    return target


class _ReadMediaOrDocTool(Tool):
    name: ClassVar[str] = "read_media_or_doc"
    is_result_long: ClassVar[bool] = True
    staleness: ClassVar[timedelta] = timedelta(hours=24)
    description: ClassVar[str] = (
        "Read a media or document file from this server's resource and return its "
        "content as text. Media is described or transcribed, and text files come "
        "back verbatim. Use for files already in the resource. For content at a "
        "URL, use web_fetch instead. If the file is not in the resource, call "
        "send_or_receive_file first for upload instructions."
    )
    summary: ClassVar[str] = "Read media or documents from the resource to text"
    params: ClassVar[tuple[Param, ...]] = (
        Param(
            "resource_path",
            str,
            kind="resource",
            description="Path of the file relative to the resource root, e.g. 'upload/lecture.mp4'.",
        ),
        Param(
            "prompt",
            str,
            "",
            kind="longtext",
            description=(
                "Set ONLY when you want the LLM to extract a specific answer or summary "
                "from the content instead of returning the full raw content. For example, "
                "use prompt='What is the price of the product?' to extract just the price "
                "from a document. Leave empty to receive the complete raw content, such as "
                "a description of media, a transcription of audio, or a text file verbatim."
            ),
        ),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        resource_path: str = params["resource_path"]
        prompt: str = params.get("prompt", "")

        target = await asyncio.to_thread(_resolve_resource_file, resource_path)
        try:
            content, processed = await resolve_input(target, context=prompt, verbose=True)
        except ResolveError as e:
            raise RuntimeError(f"Error converting '{resource_path}' to text: {e}") from e
        return await answer_or_content(prompt, processed, content, f"'{resource_path}'")

    @override
    async def cache_key(self, params: dict) -> str:
        resource_path: str = params["resource_path"]
        target = await resolve_path(resources_dir(), resource_path)
        file_hash = resource_path
        if target is not None and await asyncio.to_thread(os.path.isfile, target):
            try:
                file_hash = await asyncio.to_thread(_hash_file, target)
            except OSError:
                file_hash = resource_path
        payload = json.dumps(
            {
                "resource_path": resource_path,
                "file_hash": file_hash,
                "prompt": params.get("prompt", ""),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


read_media_or_doc_tool = _ReadMediaOrDocTool()
