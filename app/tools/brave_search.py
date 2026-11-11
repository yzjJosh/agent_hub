from __future__ import annotations

from typing import ClassVar, override

import httpx

from app.lib.shared import config
from app.lib.shared.format import format_results
from app.lib.shared.rate_limiter import RateLimiter
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult

__all__ = ["brave_search_tool"]

_FRESHNESS_MAP = {"d": "pd", "w": "pw", "m": "pm", "y": "py"}
_API_URL = "https://api.search.brave.com/res/v1/web/search"
_DEFAULT_COUNT = 10
_TIMEOUT = 30
_MIN_DELAY = 1.1


_limiter = RateLimiter(_MIN_DELAY)


def _safe_json(resp: httpx.Response) -> dict | None:
    """Return the response body as JSON, or None when it is not JSON."""
    try:
        return resp.json()
    except ValueError:
        return None


def _extract_results(response_data: dict | None) -> list[dict[str, str]]:
    """Return the search results found in a response."""
    if not isinstance(response_data, dict):
        return []

    web = response_data.get("web")
    if not isinstance(web, dict):
        return []

    raw_results = web.get("results", [])
    if not isinstance(raw_results, list):
        return []

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("description", ""),
        }
        for item in raw_results
        if isinstance(item, dict)
    ]


async def _request(params: dict[str, str | int], api_key: str) -> list[dict[str, str]]:
    """Ask Brave for search results.

    Returns:
        The results, which may be empty when nothing matched.

    Raises:
        RuntimeError: If the request fails or the answer cannot be used.
    """
    resp: httpx.Response | None = None
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                _API_URL,
                headers={
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
                params=params,
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Brave Search request failed: {exc}") from exc

    if resp is None:
        raise RuntimeError("Brave Search request failed: no response")

    raw = _safe_json(resp)
    if raw is None:
        raise RuntimeError("Brave Search request failed: invalid JSON response")

    if isinstance(raw, dict) and "error" in raw:
        raise RuntimeError(f"Brave Search API error: {raw['error']}")

    return _extract_results(raw)


class _BraveSearchTool(Tool):
    name: ClassVar[str] = "brave_search"
    result_render_type: ClassVar[str] = "markdown"
    description: ClassVar[str] = (
        "Search the web via the Brave Search API. Faster than google_search but "
        "limited to 2000 queries per month. Use it first, and fall back to "
        "google_search when the quota is exceeded."
    )
    summary: ClassVar[str] = "Web search via the Brave API"
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
        api_key = config.require_env("BRAVE_SEARCH_API_KEY", "needed by brave_search")

        request_params: dict[str, str | int] = {"q": query.strip(), "count": _DEFAULT_COUNT}
        if freshness:
            request_params["freshness"] = _FRESHNESS_MAP[freshness]

        await _limiter.acquire()
        results = await _request(request_params, api_key)
        if not results:
            return ToolResult(text="")
        return ToolResult(text=format_results("Brave", query, results, "description"))


brave_search_tool = _BraveSearchTool()
