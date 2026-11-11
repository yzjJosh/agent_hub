from __future__ import annotations

from unittest.mock import patch

import pytest

from app.lib.tool.base import ToolResult
from app.lib.shared.format import answer_or_content, format_results, html_tree_to_markdown


class TestAnswerOrContent:
    """Tests for answering a prompt or giving back the content."""

    async def test_no_prompt_returns_content(self) -> None:
        """With no prompt, the content comes back as text with no declared kind."""
        result = await answer_or_content("", False, "Plain text.", "upload/a.mp4")

        assert isinstance(result, ToolResult)
        assert result.text == "Plain text."
        assert result.files is None
        assert result.render_type is None

    async def test_prompted_unprocessed_content_gets_answer(self) -> None:
        """A prompt on fresh content comes back as the answer."""
        with patch("app.lib.shared.format.reason") as mock_reason:
            mock_reason.return_value = ("The answer.", 0)
            result = await answer_or_content("summarize", False, "Plain text.", "upload/a.mp4")

        assert result.text == "The answer."
        mock_reason.assert_called_once_with("summarize", context="Plain text.")

    async def test_prompted_answer_declares_markdown(self) -> None:
        """An LLM answer comes back declared as markdown."""
        with patch("app.lib.shared.format.reason") as mock_reason:
            mock_reason.return_value = ("## Summary\n\nThe answer.", 0)
            result = await answer_or_content("summarize", False, "Plain text.", "upload/a.mp4")

        assert result.text == "## Summary\n\nThe answer."
        assert result.render_type == "markdown"

    async def test_prompted_processed_content_comes_back_raw(self) -> None:
        """Content that was already worked on comes back as is even with a prompt."""
        with patch("app.lib.shared.format.reason") as mock_reason:
            result = await answer_or_content("summarize", True, "Processed text.", "upload/a.mp4")

        assert result.text == "Processed text."
        mock_reason.assert_not_called()

    async def test_answer_failure_raises(self) -> None:
        """A failed answer raises with the source name in the message."""
        with patch("app.lib.shared.format.reason") as mock_reason:
            mock_reason.return_value = ("boom", 1)
            with pytest.raises(RuntimeError, match="Error extracting from upload/a.mp4: boom"):
                await answer_or_content("summarize", False, "Plain text.", "upload/a.mp4")


class TestFormatResults:
    """Tests for the search results formatter."""

    def test_formats_markdown(self) -> None:
        """Many results turn into a numbered markdown list."""
        results = [
            {"title": "Result One", "url": "https://example.com/1", "description": "First description"},
            {"title": "Result Two", "url": "https://example.com/2", "description": "Second description"},
        ]

        expected = (
            "## Brave Search Results for: python\n"
            "\n"
            "Found 2 results:\n"
            "\n"
            "1. **Result One**\n"
            "   URL: https://example.com/1\n"
            "   First description\n"
            "\n"
            "2. **Result Two**\n"
            "   URL: https://example.com/2\n"
            "   Second description\n"
        )

        assert format_results("Brave", "python", results, "description") == expected

    def test_missing_fields_omitted(self) -> None:
        """A missing title or description comes out as empty text."""
        results = [{"url": "https://example.com"}]

        expected = (
            "## Google Search Results for: test\n"
            "\n"
            "Found 1 results:\n"
            "\n"
            "1. ****\n"
            "   URL: https://example.com\n"
        )

        assert format_results("Google", "test", results, "snippet") == expected

    def test_empty_results(self) -> None:
        """No results gives back an empty string."""
        assert format_results("Brave", "python", [], "description") == ""


async def test_page_tree_as_markdown() -> None:
    """A page tree is returned as readable markdown."""
    result = await html_tree_to_markdown({"role": "heading", "name": "A title"})
    assert result == "# A title"
