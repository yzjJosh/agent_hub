from __future__ import annotations

"""Fetch content from a URL."""

import json
import os
from typing import ClassVar, override

from app.lib.resolver.resolver import BINARY_PASSTHROUGH_TAG, ResolveError, resolve_input
from app.lib.tool.base import Param, Tool, ToolResult
from app.lib.shared.format import answer_or_content
from app.lib.shared.url import validate_url
from app.resources.resources import binary_download_notice

__all__ = ["web_fetch_tool"]


async def _fetch_impl(url: str, prompt: str) -> ToolResult:
    """Fetch a URL and give back its content.

    Args:
        url: The address to fetch.
        prompt: A question to answer about the content. Empty asks for the content itself.

    Returns:
        The content as text, or the saved file plus a download note when it is binary
        and there is no prompt.

    Raises:
        ValueError: If the url is not allowed.
        RuntimeError: If the fetch fails.
    """
    error = validate_url(url)
    if error:
        raise ValueError(error)
    binary_passthrough = not prompt
    try:
        content, processed = await resolve_input(url, context=prompt, verbose=True,
                                                 binary_passthrough=binary_passthrough)
    except ResolveError as e:
        raise RuntimeError(f"Error fetching {url}: {e}") from e
    if content.startswith(BINARY_PASSTHROUGH_TAG):
        return await _binary_result(content[len(BINARY_PASSTHROUGH_TAG):])
    result = await answer_or_content(prompt, processed, content, url)
    if prompt and not processed:
        return result
    text = result.text
    if text != content:
        return ToolResult(text=text, render_type="markdown")
    if _looks_like_json(text):
        return ToolResult(text=text, render_type="json")
    return result


def _looks_like_json(text: str) -> bool:
    """Whether the text parses as a JSON document.

    Args:
        text: The result body to test.

    Returns:
        True when the text is one whole JSON object or array, False
        otherwise.
    """
    if text.lstrip()[:1] not in "{[":
        return False
    try:
        json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return True


async def _binary_result(local_path: str) -> ToolResult:
    """Build the result for a binary file.

    Args:
        local_path: The downloaded file on disk.

    Returns:
        A note on how to download the saved copy and its resource path.
    """
    share_rel = f"downloads/{os.path.basename(local_path)}"
    text = await binary_download_notice(share_rel)
    return ToolResult(text=text, files=(share_rel,))


class _WebFetchTool(Tool):
    name: ClassVar[str] = "web_fetch"
    is_result_long: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Fetch content from a URL: webpages, PDFs, images, video, audio, and other "
        "downloadable files. Use for content at a URL. For files already in "
        "this server's resource, use read_media_or_doc instead."
    )
    summary: ClassVar[str] = "Fetch a URL: page, PDF, image, audio, doc"
    params: ClassVar[tuple[Param, ...]] = (
        Param(
            "url",
            str,
            kind="url",
            description="The URL to fetch. Can be a webpage, PDF, image, video, audio, or any downloadable file.",
        ),
        Param(
            "prompt",
            str,
            "",
            kind="longtext",
            description=(
                "Set ONLY when you want the LLM to extract a specific answer or summary from "
                "the content instead of returning the full raw content. For example, use "
                "prompt='What is the price of the product?' to extract just the price from a "
                "product page. Leave empty to receive the complete raw page or file content, "
                "such as for reading an article or browsing a full webpage. For a binary file, "
                "leaving it empty stores a download command as the run result and the bytes "
                "are saved under downloads/ (use read_media_or_doc on that path to describe "
                "or transcribe it). A non-empty prompt always returns a text answer instead."
            ),
        ),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        return await _fetch_impl(params["url"], params.get("prompt", ""))


web_fetch_tool = _WebFetchTool()
