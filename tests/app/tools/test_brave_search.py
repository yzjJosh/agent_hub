from __future__ import annotations

import importlib
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import app.tools.brave_search
from app.tools.brave_search import brave_search_tool
from tests.app.lib.tool.conftest import run_text

pytestmark = pytest.mark.usefixtures("spaced_clock")


def _mock_client(response: MagicMock) -> MagicMock:
    """Make a fake client that gives back this response."""
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _mock_response(json_data: dict | None) -> MagicMock:
    """Make a fake response that gives back this JSON."""
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture(autouse=True)
def reset_limiter() -> None:
    """Give each test a fresh copy of the module."""
    importlib.reload(app.tools.brave_search)


@pytest.fixture
def mock_response() -> MagicMock:
    """A fake response holding two search results."""
    return _mock_response({
        "web": {
            "results": [
                {
                    "title": "Result One",
                    "url": "https://example.com/1",
                    "description": "First result description",
                },
                {
                    "title": "Result Two",
                    "url": "https://example.com/2",
                    "description": "Second result description",
                },
            ]
        }
    })


@pytest.fixture
def mock_client(mock_response: MagicMock) -> MagicMock:
    """A fake client that gives back search results."""
    return _mock_client(mock_response)


class TestSearch:

    async def test_empty_query(self) -> None:
        assert await run_text(brave_search_tool, "") == "query must not be empty"

    async def test_whitespace_only_query(self) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            assert await run_text(brave_search_tool, "   ") == "query must not be empty"

    async def test_invalid_freshness(self) -> None:
        assert (
            await run_text(brave_search_tool, "test", freshness="x")
            == "Error: freshness: Input should be 'd', 'w', 'm' or 'y'."
        )

    @pytest.mark.parametrize("value", [None, "", "  "], ids=["unset", "empty", "blank"])
    async def test_missing_api_key(self, value: str | None, monkeypatch: pytest.MonkeyPatch,
                                   capsys: pytest.CaptureFixture[str]) -> None:
        """An unset key fails the call and names the variable on stderr."""
        if value is None:
            monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        else:
            monkeypatch.setenv("BRAVE_SEARCH_API_KEY", value)
        assert await run_text(brave_search_tool, "python") == (
            "Tool 'brave_search' failed: BRAVE_SEARCH_API_KEY is not set: needed by brave_search"
        )
        assert "[config] BRAVE_SEARCH_API_KEY is not set: needed by brave_search\n" in capsys.readouterr().err

    async def test_http_error(self, mock_client: MagicMock) -> None:
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.brave_search.httpx.AsyncClient", return_value=mock_client):
                assert await run_text(brave_search_tool, "python") == (
                    "Tool 'brave_search' failed: Brave Search request failed: connection refused"
                )

    async def test_api_error_response(self) -> None:
        client = _mock_client(_mock_response({"error": {"code": 102, "message": "Account not found"}}))
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.brave_search.httpx.AsyncClient", return_value=client):
                assert await run_text(brave_search_tool, "python") == (
                    "Tool 'brave_search' failed: Brave Search API error: "
                    "{'code': 102, 'message': 'Account not found'}"
                )

    async def test_invalid_json_response(self) -> None:
        resp = _mock_response(None)
        resp.json.side_effect = ValueError("not json")
        client = _mock_client(resp)
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.brave_search.httpx.AsyncClient", return_value=client):
                assert await run_text(brave_search_tool, "python") == (
                    "Tool 'brave_search' failed: Brave Search request failed: invalid JSON response"
                )

    async def test_empty_web_results(self) -> None:
        client = _mock_client(_mock_response({"web": {"results": []}}))
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.brave_search.httpx.AsyncClient", return_value=client):
                assert await run_text(brave_search_tool, "python") == ""

    async def test_missing_web_key(self) -> None:
        client = _mock_client(_mock_response({}))
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.brave_search.httpx.AsyncClient", return_value=client):
                assert await run_text(brave_search_tool, "python") == ""

    async def test_successful_search(self, mock_client: MagicMock) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.brave_search.httpx.AsyncClient", return_value=mock_client):
                result = await run_text(brave_search_tool, "python")
                assert "## Brave Search Results for: python" in result
                assert "Found 2 results:" in result
                assert "1. **Result One**" in result
                assert "URL: https://example.com/1" in result
                assert "First result description" in result
                assert "2. **Result Two**" in result

    async def test_freshness_mapping(self, mock_client: MagicMock) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.brave_search.httpx.AsyncClient", return_value=mock_client):
                await run_text(brave_search_tool, "python", freshness="d")
                mock_client.get.assert_called_once()
                assert mock_client.get.call_args[1]["params"]["freshness"] == "pd"

    @pytest.mark.parametrize("freshness,expected", [
        ("d", "pd"),
        ("w", "pw"),
        ("m", "pm"),
        ("y", "py"),
    ])
    async def test_all_freshness_values(self, mock_client: MagicMock, freshness: str, expected: str) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.brave_search.httpx.AsyncClient", return_value=mock_client):
                await run_text(brave_search_tool, "test", freshness=freshness)
                assert mock_client.get.call_args[1]["params"]["freshness"] == expected

    async def test_request_headers(self, mock_client: MagicMock) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.brave_search.httpx.AsyncClient", return_value=mock_client):
                await run_text(brave_search_tool, "python")
                headers = mock_client.get.call_args[1]["headers"]
                assert headers["X-Subscription-Token"] == "test-key"
                assert headers["Accept-Encoding"] == "gzip"

    async def test_request_timeout(self, mock_client: MagicMock) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.brave_search.httpx.AsyncClient", return_value=mock_client):
                await run_text(brave_search_tool, "python")
                assert mock_client.get.call_args[1]["timeout"] == 30

    async def test_query_stripped(self, mock_client: MagicMock) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.brave_search.httpx.AsyncClient", return_value=mock_client):
                await run_text(brave_search_tool, "  python  ")
                assert mock_client.get.call_args[1]["params"]["q"] == "python"

    async def test_missing_fields_in_result(self) -> None:
        client = _mock_client(_mock_response({
            "web": {
                "results": [
                    {"title": "Only Title"},
                    {"url": "https://example.com"},
                    {},
                ]
            }
        }))
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.brave_search.httpx.AsyncClient", return_value=client):
                result = await run_text(brave_search_tool, "test")
                assert "Only Title" in result
                assert "URL: https://example.com" in result

    async def test_non_dict_items_in_results(self) -> None:
        client = _mock_client(_mock_response({
            "web": {
                "results": [
                    {"title": "Valid", "url": "https://example.com", "description": "desc"},
                    "invalid string",
                    42,
                ]
            }
        }))
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.brave_search.httpx.AsyncClient", return_value=client):
                result = await run_text(brave_search_tool, "test")
                assert "Valid" in result
                assert "invalid string" not in result

    async def test_http_error_response(self) -> None:
        resp = _mock_response(None)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403 Forbidden", request=MagicMock(), response=MagicMock(status_code=403)
        )
        client = _mock_client(resp)
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.brave_search.httpx.AsyncClient", return_value=client):
                assert (await run_text(brave_search_tool, "python")).startswith("Tool 'brave_search' failed: ")
