"""Take a screenshot of a webpage and save it in this server's share."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import ClassVar, override

from app.lib.browser.browser_manager import BrowserManager
from app.lib.shared.paths import resources_dir
from app.lib.tool.base import Param, Tool, ToolResult
from app.lib.shared.url import validate_url
from app.resources.resources import dav_base_url

__all__ = ["web_screenshot_tool"]

_VERBOSE = True
_OUTPUT_DIR = "screenshots"
_NAVIGATION_TIMEOUT_MS = 30_000
_RENDER_WAIT_TIMEOUT_MS = 5_000


def _output_path() -> str:
    """Return where to save the screenshot."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(resources_dir(), _OUTPUT_DIR, f"screenshot_{timestamp}.jpg")


def _success_message(output_path: str) -> str:
    """Return the message with the saved path and how to download the file."""
    resource_path = os.path.relpath(output_path, resources_dir())
    return (
        f"Screenshot saved to resource path: {resource_path}\n"
        "Download it with:\n"
        f"  curl -o /tmp/{os.path.basename(output_path)} {dav_base_url()}/dav/{resource_path}\n"
        "See the send_or_receive_file tool for file transfer details."
    )


async def _screenshot_impl(url: str, full_page: bool) -> tuple[str, str]:
    """Take a screenshot of a webpage and save it.

    Args:
        url: The page to take a picture of.
        full_page: True for the whole page, False for only the part you can see.

    Returns:
        A message with the saved path and how to download the file, plus the resource
        path of the saved screenshot.

    Raises:
        ValueError: If the url is not allowed.
        RuntimeError: If the page could not be opened or captured.
    """
    error = validate_url(url)
    if error:
        raise ValueError(error)
    output_file = _output_path()
    async with BrowserManager(verbose=_VERBOSE) as mgr:
        target_id = await mgr.new_page(verbose=_VERBOSE)
        if not await mgr.navigate(
            target_id,
            url,
            wait_until="domcontentloaded",
            timeout=_NAVIGATION_TIMEOUT_MS,
            verbose=_VERBOSE,
        ):
            raise RuntimeError(f"Screenshot failed: could not navigate to {url}")
        await mgr.wait_for_load(
            target_id, state="load", timeout=_RENDER_WAIT_TIMEOUT_MS, verbose=_VERBOSE
        )
        await mgr.wait_for_load(
            target_id, state="networkidle", timeout=_RENDER_WAIT_TIMEOUT_MS, verbose=_VERBOSE
        )
        await asyncio.to_thread(os.makedirs, os.path.dirname(output_file), exist_ok=True)
        saved = await mgr.screenshot(
            target_id, output_file, full_page=full_page, verbose=_VERBOSE
        )
    if saved is None:
        raise RuntimeError(f"Screenshot failed: screenshot capture failed for {url}")
    return _success_message(output_file), os.path.relpath(output_file, resources_dir())


class _WebScreenshotTool(Tool):
    name: ClassVar[str] = "web_screenshot"
    result_render_type: ClassVar[str] = "text"
    description: ClassVar[str] = (
        "Take a screenshot of a webpage. Use for capturing a page's rendered "
        "content as an image resource."
    )
    summary: ClassVar[str] = "Screenshot of a rendered webpage"
    params: ClassVar[tuple[Param, ...]] = (
        Param("url", str, kind="url", description="The URL of the webpage to screenshot."),
        Param(
            "full_page",
            bool,
            True,
            description="Whether to capture the full scrollable page or only the browser's viewport.",
        ),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        text, share_rel = await _screenshot_impl(params["url"], params.get("full_page", True))
        return ToolResult(text=text, files=(share_rel,))


web_screenshot_tool = _WebScreenshotTool()
