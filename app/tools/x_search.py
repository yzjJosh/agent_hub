from __future__ import annotations

import asyncio
import shutil
from typing import ClassVar, override

from app.lib.cli.cli_runner import run_cli
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult

__all__ = ["x_search_tool"]


async def _run_bird(query: str, count: int) -> str:
    """Search X for posts.

    Args:
        query: What to look for.
        count: How many results to ask for.

    Returns:
        The results as text, or a line saying there were none.

    Raises:
        RuntimeError: If the search fails.
    """
    args = ["bird", "search", query, "--count", str(count)]
    result = await run_cli(args, "bird search")
    if result.returncode != 0:
        raise RuntimeError(f"Error running bird search: {result.stderr}")
    return result.stdout or "No results from X search."


class _XSearchTool(Tool):
    name: ClassVar[str] = "x_search"
    description: ClassVar[str] = (
        "Search X/Twitter. Use ONLY when you specifically need social "
        "media/X/Twitter content, opinions, or real-time reactions. Not for "
        "general web searches."
    )
    summary: ClassVar[str] = "Search X / Twitter"
    params: ClassVar[tuple[Param, ...]] = (
        Param(
            "query",
            str,
            description=(
                "Free-text search query. Leading/trailing whitespace is\n"
                "stripped. A blank query returns a tool error without running "
                "the CLI."
            ),
        ),
        Param("count", int, 5, description="Number of results to request."),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        query = params["query"].strip()
        if not query:
            raise ToolMessageError("query must not be empty")
        if await asyncio.to_thread(shutil.which, "bird") is None:
            raise ToolMessageError("Error: bird CLI not found.")
        return ToolResult(text=await _run_bird(query, params["count"]))


x_search_tool = _XSearchTool()
