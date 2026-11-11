from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.tools.google_search
from app.lib.browser.analyze_page import BrowserError
from app.tools.google_search import google_search_tool
from tests.app.conftest import is_llm_available, skip_if_no_chrome
from tests.app.lib.tool.conftest import run_text

pytestmark = pytest.mark.usefixtures("spaced_clock")

_SEARCH_RESULT = ('{"results": []}', "https://www.google.com/search?q=test")


@pytest.fixture(autouse=True)
def reset_limiter() -> None:
    """Give each test a fresh copy of the module."""
    importlib.reload(app.tools.google_search)


class TestSearchParsing:
    """Tests for turning search results into markdown."""

    @pytest.mark.parametrize(
        "llm_response,expected",
        [
            (
                '{"results": [{"title": "X", "url": "http://x.com", "snippet": "desc"}]}',
                "## Google Search Results for: test query\n\n"
                "Found 1 results:\n\n"
                "1. **X**\n"
                "   URL: http://x.com\n"
                "   desc\n",
            ),
            (
                'noise {"results": [{"title": "Y", "url": "http://y.com", "snippet": "snip"}]} noise',
                "## Google Search Results for: test query\n\n"
                "Found 1 results:\n\n"
                "1. **Y**\n"
                "   URL: http://y.com\n"
                "   snip\n",
            ),
        ],
    )
    async def test_search_formats_markdown(self, llm_response: str, expected: str) -> None:
        """Search results come back as markdown, even when the JSON has extra text around it."""
        with patch(
            "app.tools.google_search.analyze_page",
            return_value=(llm_response, "https://www.google.com/search?q=test"),
        ):
            result = await run_text(google_search_tool, "test query")

        assert result == expected

    @pytest.mark.parametrize("llm_response", ["not json", '{"items": []}'])
    async def test_search_invalid_json_returns_failed(self, llm_response: str) -> None:
        """Output that is not JSON, or has no results list, gives back a failed tool message."""
        with patch(
            "app.tools.google_search.analyze_page",
            return_value=(llm_response, "https://www.google.com/search?q=test"),
        ):
            result = await run_text(google_search_tool, "test query")

        assert result == (
            "Tool 'google_search' failed: google_search failed: could not parse LLM output as JSON"
        )

    async def test_search_browser_error_returns_failed(self) -> None:
        """A browser error gives back a failed tool message."""
        with patch(
            "app.tools.google_search.analyze_page",
            side_effect=BrowserError("page blocked"),
        ):
            result = await run_text(google_search_tool, "test query")

        assert result == "Tool 'google_search' failed: google_search failed: page blocked"

    async def test_search_anti_bot_page_returns_failed(self) -> None:
        """Landing on Google's anti-bot page gives back a failed tool message."""
        with patch(
            "app.tools.google_search.analyze_page",
            return_value=('{"results": []}', "https://www.google.com/sorry/index?continue=abc"),
        ):
            result = await run_text(google_search_tool, "test query")

        assert result == (
            "Tool 'google_search' failed: google_search failed: "
            "blocked by Google anti-bot page"
        )

    async def test_search_anti_bot_page_alternate_host_returns_failed(self) -> None:
        """The anti-bot page on a different Google host also gives back a failed tool message."""
        with patch(
            "app.tools.google_search.analyze_page",
            return_value=('{"results": []}', "https://sorry.google.com/sorry/index?continue=abc"),
        ):
            result = await run_text(google_search_tool, "test query")

        assert result == (
            "Tool 'google_search' failed: google_search failed: "
            "blocked by Google anti-bot page"
        )

    async def test_search_empty_results_returns_empty(self) -> None:
        """No results gives back an empty string."""
        with patch(
            "app.tools.google_search.analyze_page",
            return_value=('{"results": []}', "https://www.google.com/search?q=test"),
        ):
            result = await run_text(google_search_tool, "test query")

        assert result == ""


class TestSearchThrottle:
    """Tests for the wait between searches."""

    async def test_back_to_back_searches_wait_one_second(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A search made right after another one waits a full second first."""
        monkeypatch.setattr("app.lib.shared.rate_limiter.time", MagicMock(monotonic=lambda: 1000.0))
        with (
            patch("app.tools.google_search.analyze_page", return_value=_SEARCH_RESULT),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            await run_text(google_search_tool, "first")
            mock_sleep.assert_not_called()
            await run_text(google_search_tool, "second")

        mock_sleep.assert_awaited_once_with(1.0)

    async def test_search_after_one_second_does_not_wait(self) -> None:
        """A search made after a second has passed goes right away."""
        with (
            patch("app.tools.google_search.analyze_page", return_value=_SEARCH_RESULT),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            await run_text(google_search_tool, "first")
            await run_text(google_search_tool, "second")

        mock_sleep.assert_not_called()


class TestSearchValidation:
    """Tests for bad search inputs."""

    async def test_search_empty_query(self) -> None:
        """An empty query is rejected with the tool's own words."""
        result = await run_text(google_search_tool, "")

        assert result == "query must not be empty"

    async def test_search_invalid_freshness(self) -> None:
        """A bad freshness value gives back an error message."""
        result = await run_text(google_search_tool, "test", freshness="x")
        assert result == "Error: freshness: Input should be 'd', 'w', 'm' or 'y'."


@pytest.mark.skipif(not is_llm_available(), reason="LLM server not available")
@skip_if_no_chrome
@pytest.mark.live
@pytest.mark.timeout(600)
class TestSearchLive:
    """Live web search tests."""

    async def test_search_returns_markdown(self) -> None:
        """A live web search gives back markdown, or an empty string when the page has no results."""
        result = await run_text(google_search_tool, "python async")

        assert isinstance(result, str), f"unexpected return type: {type(result)}"
        if result:
            assert result.startswith("## Google Search Results for: python async")
            assert "URL:" in result
