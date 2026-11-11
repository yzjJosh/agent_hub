from __future__ import annotations

"""Search Google for a query."""

import json
import re
import sys
from typing import ClassVar, override
from urllib.parse import quote_plus

from app.lib.browser.analyze_page import BrowserError, analyze_page
from app.lib.shared.format import format_results
from app.lib.shared.rate_limiter import RateLimiter
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult

__all__ = ["google_search_tool"]

_VERBOSE = True
_MIN_DELAY = 1.0
_ANTI_BOT_RE = re.compile(r"^https?://(?:[a-z0-9-]+\.)*google\.[a-z.]{2,6}/sorry/", re.I)
_RESULT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "snippet": {"type": "string"},
                },
                "required": ["title", "url", "snippet"],
            },
        },
    },
    "required": ["results"],
}

_limiter = RateLimiter(_MIN_DELAY)


def _walk(node: dict | list | str) -> dict | None:
    """Find the web results inside a page tree.

    Args:
        node: One node of the tree.

    Returns:
        The part of the tree that holds the web results, or None when it is not there.
    """
    if not isinstance(node, dict):
        return None
    if node.get("role") == "main":
        children = node.get("children", [])
        for idx, child in enumerate(children):
            if (
                isinstance(child, dict)
                and child.get("role") == "heading"
                and child.get("name") == "Web results"
            ):
                return {"role": "main", "children": children[idx + 1:]}
    for child in node.get("children", []):
        result = _walk(child)
        if result is not None:
            return result
    return None


def _find_web_results_section(tree: dict) -> dict | None:
    """Return the part of the page tree that holds the web results, or None."""
    return _walk(tree)


def _build_search_url(query: str, freshness: str | None) -> str:
    """Return the search URL for a query."""
    url = f"https://www.google.com/search?q={quote_plus(query)}"
    if freshness in ("d", "w", "m", "y"):
        url += f"&tbs=qdr:{freshness}"
    return url


def _build_prompt(query: str) -> str:
    """Return the question to ask about the search results page."""
    return (
        "You are given the accessibility tree of a Google search results page. "
        f"The user searched for: \"{query}\".\n\n"
        "First, check if this is a CAPTCHA or error page. If it is, return {\"results\": []}.\n\n"
        "Otherwise, extract ALL organic web search results shown on this page into the "
        "\"results\" array. For each result, provide:\n"
        "- title: The page title\n"
        "- url: The full URL\n"
        "- snippet: A brief description/snippet of the page content\n\n"
        "Only include REAL organic search results. Skip:\n"
        "- Navigation links (Images, Videos, News, etc.)\n"
        "- 'People also ask' sections\n"
        "- Related searches\n"
        "- Advertisements\n"
        "- YouTube chapter timestamps (deduplicate by video)\n"
        "- Social media sub-links like 'X answers' or comment anchors\n\n"
        "Return ONLY a JSON object with a \"results\" array, nothing else."
    )


def _parse_llm_response(response: str) -> list[dict[str, str]] | None:
    """Turn the model's answer into a list of results, or None when it cannot be read."""
    output = response.strip()
    start = output.find("{")
    end = output.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(output[start:end])
    except json.JSONDecodeError:
        return None
    results = data.get("results")
    return results if isinstance(results, list) else None


def _select_web_results(tree: dict) -> dict:
    """Return the part of the page tree that holds the web results, or the whole tree."""
    return _find_web_results_section(tree) or tree


async def _browser_search(
    query: str,
    freshness: str | None = None,
) -> list[dict[str, str]]:
    """Open the search page and read the results off it.

    Returns:
        The results, which may be empty when nothing matched.

    Raises:
        RuntimeError: If the page could not be read, or the answer could not be read.
    """
    search_url = _build_search_url(query, freshness)
    prompt = _build_prompt(query)

    try:
        response, final_url = await analyze_page(
            url=search_url,
            prompt=prompt,
            transform_snapshot=_select_web_results,
            output_json_schema=_RESULT_JSON_SCHEMA,
            verbose=_VERBOSE,
        )
    except BrowserError as e:
        print(f"[google_search] ERROR: {e}", file=sys.stderr)
        raise RuntimeError(f"google_search failed: {e}") from e

    if _ANTI_BOT_RE.match(final_url):
        print(
            "[google_search] ERROR: blocked by Google anti-bot page",
            file=sys.stderr,
        )
        raise RuntimeError("google_search failed: blocked by Google anti-bot page")

    results = _parse_llm_response(response)
    if results is None:
        print("[google_search] ERROR: Could not parse LLM output as JSON", file=sys.stderr)
        raise RuntimeError("google_search failed: could not parse LLM output as JSON")

    if _VERBOSE:
        print(f"[google_search] Found {len(results)} results", flush=True)
    return results


class _GoogleSearchTool(Tool):
    name: ClassVar[str] = "google_search"
    result_render_type: ClassVar[str] = "markdown"
    description: ClassVar[str] = (
        "Search Google for a web query. Free and unlimited, but slower than "
        "brave_search. Use as the fallback when brave_search's quota is exceeded."
    )
    summary: ClassVar[str] = "Web search via Google"
    params: ClassVar[tuple[Param, ...]] = (
        Param("query", str, description="Search query string."),
        Param(
            "freshness",
            str | None,
            None,
            enum=("d", "w", "m", "y"),
            description=(
                "Time filter. d is 24 hours, w is 7 days, "
                "m is 1 month, y is 1 year."
            ),
        ),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        query = params["query"]
        if not query.strip():
            raise ToolMessageError("query must not be empty")
        freshness = params.get("freshness")
        await _limiter.acquire()
        results = await _browser_search(query, freshness)
        if not results:
            return ToolResult(text="")
        return ToolResult(text=format_results("Google", query, results, "snippet"))


google_search_tool = _GoogleSearchTool()
