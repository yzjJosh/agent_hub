from __future__ import annotations

"""Drive a browser to a goal with the vision-only role.

Carries out a plain-language task in the server's managed browser and
hands back the status, the summary, and the files that came down.
"""

__all__ = ["browser_use_tool"]

import os
from datetime import timedelta
from typing import ClassVar, override

from app.agents.browser_use.models import BrowserUseResult
from app.agents.browser_use.runner import run_browser_use
from app.lib.shared.paths import resources_dir
from app.lib.task.task_manager import task_manager
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult


def _result_markdown(result: BrowserUseResult) -> str:
    """The status, the summary, and the files as a markdown result."""
    lines = ["## Browser run", "", f"**Status:** {result.status}"]
    if result.summary:
        lines.append("")
        lines.append(result.summary)
    if result.files:
        lines.append("")
        lines.append("**Files**")
        lines.extend(f"- {file}" for file in result.files)
        lines.append("")
        lines.append("See the send_or_receive_file tool for file transfer details.")
    return "\n".join(lines)


class _BrowserUseTool(Tool):
    name: ClassVar[str] = "browser_use"
    is_slow: ClassVar[bool] = True
    staleness: ClassVar[timedelta | None] = None
    result_render_type: ClassVar[str] = "markdown"
    description: ClassVar[str] = (
        "Run a vision-only agentic browser session in the server's managed "
        "browser, carrying out a plain-language task over several steps. Use "
        "for goals that need to open pages, click, type, scroll, and download "
        "files. Use web_fetch to read a page's contents, web_screenshot for a "
        "picture of a page, and the task tool to watch a run that has started. "
        "Use send_or_receive_file to fetch the files the run downloads."
    )
    summary: ClassVar[str] = "Vision-only agentic browser session returning a markdown result"
    params: ClassVar[tuple[Param, ...]] = (
        Param(
            "prompt",
            str,
            kind="longtext",
            description="The complete task, including any site, account, or expected output file.",
        ),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        prompt = params["prompt"]
        if not prompt.strip():
            raise ToolMessageError("prompt must not be empty")
        task = task_manager.active_task()
        task_id = str(task.id)
        resources = resources_dir()
        downloads_dir = os.path.join(resources, "browser_use", task_id)
        result = await run_browser_use(prompt, downloads_dir, resources)
        return ToolResult(text=_result_markdown(result), files=result.files)


browser_use_tool = _BrowserUseTool()
