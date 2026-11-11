from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.tools.x_search import x_search_tool
from tests.app.lib.tool.conftest import run_text


def _result(stdout: str = "", stderr: str = "", returncode: int = 0) -> SimpleNamespace:
    """Make a fake command result with this output, error text, and exit code."""
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class TestSearch:

    @pytest.mark.parametrize("count,expected_count_arg", [
        (5, "5"),
        (3, "3"),
    ])
    async def test_successful_search(self, count: int, expected_count_arg: str) -> None:
        """A good search gives back the command output and asks for the count you passed."""
        with patch("app.tools.x_search.shutil.which", return_value="/usr/bin/bird"), \
             patch("app.tools.x_search.run_cli", new=AsyncMock(
                 return_value=_result("user1: first tweet\nuser2: second tweet"))) as cli:
            result = await run_text(x_search_tool, "python", count=count)
        assert result == "user1: first tweet\nuser2: second tweet"
        assert cli.call_args.args == (["bird", "search", "python", "--count", expected_count_arg], "bird search")

    async def test_default_count(self) -> None:
        """With no count given, the search asks for 5 results."""
        with patch("app.tools.x_search.shutil.which", return_value="/usr/bin/bird"), \
             patch("app.tools.x_search.run_cli", new=AsyncMock(return_value=_result("ok"))) as cli:
            await run_text(x_search_tool, "python")
        assert cli.call_args.args == (["bird", "search", "python", "--count", "5"], "bird search")

    @pytest.mark.parametrize("query", ["", "   "])
    async def test_empty_query_rejected(self, query: str) -> None:
        """A blank query is rejected with the tool's own words."""
        assert await run_text(x_search_tool, query) == "query must not be empty"

    async def test_query_is_stripped(self) -> None:
        """Spaces around the query are trimmed off."""
        with patch("app.tools.x_search.shutil.which", return_value="/usr/bin/bird"), \
             patch("app.tools.x_search.run_cli", new=AsyncMock(return_value=_result("ok"))) as cli:
            await run_text(x_search_tool, "  python  ")
        assert cli.call_args.args == (["bird", "search", "python", "--count", "5"], "bird search")

    async def test_missing_binary(self) -> None:
        """A missing bird program gives back a not found error."""
        with patch("app.tools.x_search.shutil.which", return_value=None):
            assert await run_text(x_search_tool, "python") == "Error: bird CLI not found."

    async def test_timeout(self) -> None:
        """A search that times out gives back a failed tool message with the timeout text."""
        with patch("app.tools.x_search.shutil.which", return_value="/usr/bin/bird"), \
             patch("app.tools.x_search.run_cli", new=AsyncMock(
                 return_value=_result("", "bird search timed out after 60 seconds.", -1))):
            result = await run_text(x_search_tool, "python")
        assert result == (
            "Tool 'x_search' failed: Error running bird search: "
            "bird search timed out after 60 seconds."
        )

    @pytest.mark.parametrize("stderr,expected", [
        ("rate limited", "Tool 'x_search' failed: Error running bird search: rate limited"),
        ("", "Tool 'x_search' failed: Error running bird search: "),
    ])
    async def test_nonzero_exit(self, stderr: str, expected: str) -> None:
        """A search that exits with an error gives back a failed tool message with the error text."""
        with patch("app.tools.x_search.shutil.which", return_value="/usr/bin/bird"), \
             patch("app.tools.x_search.run_cli", new=AsyncMock(return_value=_result("", stderr, 1))):
            assert await run_text(x_search_tool, "python") == expected

    async def test_empty_stdout_success_message(self) -> None:
        """A search with no output gives back the no results message."""
        with patch("app.tools.x_search.shutil.which", return_value="/usr/bin/bird"), \
             patch("app.tools.x_search.run_cli", new=AsyncMock(return_value=_result("", "", 0))):
            result = await run_text(x_search_tool, "python")
        assert result == "No results from X search."

    async def test_unexpected_exception(self) -> None:
        """A search that fails to start gives back a failed tool message with the error text."""
        with patch("app.tools.x_search.shutil.which", return_value="/usr/bin/bird"), \
             patch("app.tools.x_search.run_cli", new=AsyncMock(return_value=_result("", "spawn failed", -1))):
            result = await run_text(x_search_tool, "python")
        assert result == "Tool 'x_search' failed: Error running bird search: spawn failed"
