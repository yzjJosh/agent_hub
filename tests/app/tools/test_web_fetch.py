import json
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

from app.lib.resolver.resolver import BINARY_PASSTHROUGH_TAG
from app.lib.tool.base import ToolResult
from app.lib.tool.truncation import DEFAULT_TOOL_OUTPUT_LEN_LIMIT
from app.tools.web_fetch import web_fetch_tool
from tests.app.conftest import skip_if_no_chrome
from tests.app.lib.tool.conftest import assert_max_char_wire_contract, run_text

_OVERFLOW = 5_000


class TestMaxCharDocstring:
    """Tests for the max_char description text."""

    async def test_docstring_uses_resolved_values(self) -> None:
        """The wire gives the real default and the real smallest value, and names no constant."""
        await assert_max_char_wire_contract(web_fetch_tool, "web_fetch")


@skip_if_no_chrome
class TestWebFetchLive:

    @pytest.mark.live
    async def test_web_fetch_html_page(self, test_server_url: str) -> None:
        """Fetching a web page returns the page content."""

        result = await run_text(web_fetch_tool, test_server_url)
        assert "Example Domain" in result

    @pytest.mark.live
    async def test_web_fetch_with_prompt(
        self, test_server_url: str
    ) -> None:
        """Fetching with a prompt returns an answer about the page."""

        result = await run_text(web_fetch_tool, test_server_url, prompt="summarize the page")
        assert "download" in result.lower()

    @pytest.mark.live
    async def test_web_fetch_404(self, test_server_url: str) -> None:
        """A missing page returns an error message."""

        result = await run_text(web_fetch_tool, f"{test_server_url}/nonexistent-page-404")
        assert "Error fetching" in result


class TestUrlValidation:
    """Tests for checking the URL."""

    @patch("app.tools.web_fetch.resolve_input")
    async def test_empty_url_returns_error(self, mock_resolve: MagicMock) -> None:
        """An empty URL returns an error."""
        result = await run_text(web_fetch_tool, "")

        assert "Invalid URL" in result
        assert "missing a scheme" in result
        mock_resolve.assert_not_called()

    @patch("app.tools.web_fetch.resolve_input")
    async def test_missing_scheme_returns_error(self, mock_resolve: MagicMock) -> None:
        """A URL with no scheme returns an error."""
        result = await run_text(web_fetch_tool, "example.com")

        assert "Invalid URL" in result
        assert "missing a scheme" in result
        mock_resolve.assert_not_called()

    @patch("app.tools.web_fetch.resolve_input")
    async def test_non_http_scheme_passes_through(self, mock_resolve: MagicMock) -> None:
        """A URL with a scheme other than http is accepted."""
        mock_resolve.return_value = ("ok", False)

        result = await run_text(web_fetch_tool, "ftp://example.com/file")

        assert result == "ok"
        mock_resolve.assert_called_once()

    @patch("app.tools.web_fetch.resolve_input")
    async def test_missing_hostname_returns_error(self, mock_resolve: MagicMock) -> None:
        """A URL with no host name returns an error."""
        result = await run_text(web_fetch_tool, "http://")

        assert "Invalid URL" in result
        assert "missing a hostname" in result
        mock_resolve.assert_not_called()

    @patch("app.tools.web_fetch.resolve_input")
    async def test_valid_url_passes_through(self, mock_resolve: MagicMock) -> None:
        """A good URL is accepted and the page is fetched."""
        mock_resolve.return_value = ("ok", False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert result == "ok"
        mock_resolve.assert_called_once()


class TestMarkdownConversion:
    """Tests for turning a page tree into markdown."""

    @patch("app.tools.web_fetch.resolve_input")
    async def test_heading_converts_to_markdown_heading(self, mock_resolve: MagicMock) -> None:
        """A heading turns into a markdown heading and the output gets smaller."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "heading",
                    "name": "Example Domain",
                    "children": [{"text": "Example Domain"}]
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")
        json_size = len(json.dumps(tree))

        assert "# Example Domain" in result
        assert '"role"' not in result
        assert len(result) < json_size

    @patch("app.tools.web_fetch.resolve_input")
    async def test_link_converts_to_markdown_link(self, mock_resolve: MagicMock) -> None:
        """A link turns into a markdown link and the output gets smaller."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "link",
                    "name": "Download",
                    "url": "/download/test.bin",
                    "children": [{"text": "Download"}]
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")
        json_size = len(json.dumps(tree))

        assert "[Download](/download/test.bin)" in result
        assert len(result) < json_size

    @patch("app.tools.web_fetch.resolve_input")
    async def test_nested_content_is_not_indented(self, mock_resolve: MagicMock) -> None:
        """Nested content starts at column 0 with no tabs and the output gets smaller."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "generic",
                    "children": [
                        {
                            "role": "heading",
                            "name": "Subsection",
                            "children": [{"text": "Subsection"}]
                        },
                        {
                            "role": "paragraph",
                            "children": [{"text": "Nested content"}]
                        }
                    ]
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")
        json_size = len(json.dumps(tree))

        assert result == "### Subsection\nNested content"
        assert "\t" not in result
        assert len(result) < json_size

    @patch("app.tools.web_fetch.resolve_input")
    async def test_full_page_converts_to_compact_markdown(
        self, mock_resolve: MagicMock
    ) -> None:
        """A whole page turns into markdown and the output gets smaller."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "heading",
                    "name": "Example Domain",
                    "children": [{"text": "Example Domain"}]
                },
                {
                    "role": "paragraph",
                    "children": [{"text": "This is a test page for browser tests."}]
                },
                {
                    "role": "link",
                    "name": "Download",
                    "url": "/download/test.bin",
                    "children": [{"text": "Download"}]
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")
        json_size = len(json.dumps(tree))

        assert "# Example Domain" in result
        assert "This is a test page for browser tests." in result
        assert "[Download](/download/test.bin)" in result
        assert '"role"' not in result
        assert len(result) < json_size

    @patch("app.tools.web_fetch.resolve_input")
    async def test_non_json_content_passes_through(self, mock_resolve: MagicMock) -> None:
        """Plain text comes back unchanged."""
        mock_resolve.return_value = ("Plain text content", False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert result == "Plain text content"

    @patch("app.tools.web_fetch.resolve_input")
    async def test_raw_json_dict_passes_through_unchanged(
        self, mock_resolve: MagicMock
    ) -> None:
        """JSON that is not a page tree comes back unchanged."""
        raw_json = json.dumps(
            {"current_condition": [{"temp_C": "25", "weatherDesc": [{"value": "Sunny"}]}]}
        )
        mock_resolve.return_value = (raw_json, False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert result == raw_json
        assert "temp_C" in result

    @patch("app.tools.web_fetch.resolve_input")
    async def test_empty_html_tree_not_returned_as_json(
        self, mock_resolve: MagicMock
    ) -> None:
        """A page tree with no children turns into empty text, not JSON."""
        tree = json.dumps({"role": "document"})
        mock_resolve.return_value = (tree, False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert result == ""
        assert not result.startswith("{")

    @patch("app.tools.web_fetch.resolve_input")
    async def test_image_converts_to_markdown_image(self, mock_resolve: MagicMock) -> None:
        """An image turns into a markdown image that keeps the image URL."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "image",
                    "name": "A cat",
                    "src": "https://example.com/cat.png"
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")
        json_size = len(json.dumps(tree))

        assert "![A cat](https://example.com/cat.png)" in result
        assert len(result) < json_size

    @patch("app.tools.web_fetch.resolve_input")
    async def test_image_without_alt_keeps_src(self, mock_resolve: MagicMock) -> None:
        """An image with no alt text still shows its URL."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "image",
                    "src": "https://example.com/photo.jpg"
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert result.strip() == "![](https://example.com/photo.jpg)"

    @patch("app.tools.web_fetch.resolve_input")
    async def test_button_converts_to_plain_text(self, mock_resolve: MagicMock) -> None:
        """A button turns into plain text and the output gets smaller."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "button",
                    "name": "Submit",
                    "children": [{"text": "Submit"}]
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")
        json_size = len(json.dumps(tree))

        assert "Submit" in result
        assert len(result) < json_size

    @patch("app.tools.web_fetch.resolve_input")
    async def test_listitem_converts_to_bullet(self, mock_resolve: MagicMock) -> None:
        """A list item turns into a bullet and the output gets smaller."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "listitem",
                    "name": "Item one",
                    "children": [{"text": "Item one"}]
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")
        json_size = len(json.dumps(tree))

        assert "- Item one" in result
        assert len(result) < json_size

    @patch("app.tools.web_fetch.resolve_input")
    async def test_heading_depth_capped_at_six(self, mock_resolve: MagicMock) -> None:
        """A heading never gets more than 6 hashes, no matter how deep it sits."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "generic",
                    "children": [
                        {
                            "role": "generic",
                            "children": [
                                {
                                    "role": "generic",
                                    "children": [
                                        {
                                            "role": "generic",
                                            "children": [
                                                {
                                                    "role": "generic",
                                                    "children": [
                                                        {
                                                            "role": "generic",
                                                            "children": [
                                                                {
                                                                    "role": "heading",
                                                                    "name": "Deep",
                                                                    "children": [{"text": "Deep"}]
                                                                }
                                                            ]
                                                        }
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")
        json_size = len(json.dumps(tree))

        assert "###### Deep" in result
        assert len(result) < json_size

    @patch("app.tools.web_fetch.resolve_input")
    async def test_table_converts_to_gfm_pipe_table(self, mock_resolve: MagicMock) -> None:
        """A table turns into a pipe table with blank lines around it and escaped pipes."""
        tree = {
            "role": "document",
            "children": [
                {"role": "paragraph", "children": [{"text": "Before"}]},
                {
                    "role": "table",
                    "children": [
                        {
                            "role": "row",
                            "children": [
                                {"role": "cell", "children": [{"text": "Col A"}]},
                                {"role": "cell", "children": [{"text": "Col B"}]},
                            ],
                        },
                        {
                            "role": "row",
                            "children": [
                                {"role": "cell", "children": [{"text": "1|2"}]},
                                {"role": "cell", "children": [{"text": "3"}]},
                            ],
                        },
                    ],
                },
                {"role": "paragraph", "children": [{"text": "After"}]},
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert result == (
            "Before\n\n| Col A | Col B |\n| --- | --- |\n| 1\\|2 | 3 |\n\nAfter"
        )

    @patch("app.tools.web_fetch.resolve_input")
    async def test_list_before_paragraph_gets_blank_line(
        self, mock_resolve: MagicMock
    ) -> None:
        """A list followed by a paragraph gets a blank line between them."""
        tree = {
            "role": "document",
            "children": [
                {"role": "paragraph", "children": [{"text": "Top"}]},
                {
                    "role": "list",
                    "children": [
                        {"role": "listitem", "name": "item", "children": [{"text": "item"}]}
                    ],
                },
                {"role": "paragraph", "children": [{"text": "After"}]},
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert result == "Top\n- item\n\nAfter"

    @patch("app.tools.web_fetch.resolve_input")
    async def test_list_before_heading_gets_no_blank_line(
        self, mock_resolve: MagicMock
    ) -> None:
        """A list followed by a heading needs no blank line between them."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "list",
                    "children": [
                        {"role": "listitem", "name": "item", "children": [{"text": "item"}]}
                    ],
                },
                {"role": "heading", "name": "Next", "children": [{"text": "Next"}]},
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert result == "- item\n## Next"

    @patch("app.tools.web_fetch.resolve_input")
    async def test_nested_list_item_indented_two_spaces(
        self, mock_resolve: MagicMock
    ) -> None:
        """A list item inside a nested list is indented with exactly two spaces."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "list",
                    "children": [
                        {
                            "role": "listitem",
                            "name": "one",
                            "children": [
                                {"text": "one"},
                                {
                                    "role": "list",
                                    "children": [
                                        {
                                            "role": "listitem",
                                            "name": "sub",
                                            "children": [{"text": "sub"}],
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert result == "- one\n  - sub"
        assert "\n  - sub" in result
        assert "\t" not in result

    @patch("app.tools.web_fetch.resolve_input")
    async def test_output_has_no_tabs(self, mock_resolve: MagicMock) -> None:
        """No line of a multi-block document carries a tab."""
        tree = {
            "role": "document",
            "children": [
                {"role": "heading", "name": "H", "children": [{"text": "H"}]},
                {"role": "paragraph", "children": [{"text": "Text"}]},
                {
                    "role": "list",
                    "children": [
                        {"role": "listitem", "name": "a", "children": [{"text": "a"}]}
                    ],
                },
                {
                    "role": "table",
                    "children": [
                        {
                            "role": "row",
                            "children": [{"role": "cell", "children": [{"text": "x"}]}],
                        }
                    ],
                },
                {
                    "role": "link",
                    "name": "L",
                    "url": "/l",
                    "children": [{"text": "L"}],
                },
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert "\t" not in result

    @patch("app.tools.web_fetch.resolve_input")
    async def test_link_name_with_brackets_is_escaped(self, mock_resolve: MagicMock) -> None:
        """A link name with brackets keeps the link syntax intact."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "link",
                    "name": "a[b]c",
                    "url": "/x",
                    "children": [{"text": "a[b]c"}],
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert result == "[a\\[b\\]c](/x)"

    @patch("app.tools.web_fetch.resolve_input")
    async def test_long_plain_text_comes_back_whole(self, mock_resolve: MagicMock) -> None:
        """Plain text longer than 25000 characters comes back whole, with no note."""
        long_text = "x" * (DEFAULT_TOOL_OUTPUT_LEN_LIMIT + _OVERFLOW)
        mock_resolve.return_value = (long_text, False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert result == long_text
        assert "[Truncated" not in result

    @patch("app.tools.web_fetch.resolve_input")
    async def test_short_plain_text_not_truncated(self, mock_resolve: MagicMock) -> None:
        """Text under 25000 characters comes back whole."""
        short_text = "short content"
        mock_resolve.return_value = (short_text, False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert result == short_text

    @patch("app.tools.web_fetch.resolve_input")
    async def test_long_markdown_comes_back_whole(self, mock_resolve: MagicMock) -> None:
        """Markdown output longer than 25000 characters comes back whole, with no note."""
        large_tree = {
            "role": "document",
            "children": [
                {
                    "role": "paragraph",
                    "children": [{"text": "p" * (DEFAULT_TOOL_OUTPUT_LEN_LIMIT + _OVERFLOW)}]
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(large_tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert result.startswith("p" * DEFAULT_TOOL_OUTPUT_LEN_LIMIT)
        assert "[Truncated" not in result

    @patch("app.lib.shared.format.reason")
    @patch("app.tools.web_fetch.resolve_input")
    async def test_llm_called_when_prompt_provided(
        self, mock_resolve: MagicMock, mock_reason: MagicMock
    ) -> None:
        """A prompt turns the page into an answer."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "paragraph",
                    "children": [{"text": "This is the page content."}]
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)
        mock_reason.return_value = ("Summary of page.", 0)

        result = await run_text(web_fetch_tool, "http://example.com", prompt="summarize")

        assert "Summary of page." == result
        mock_reason.assert_called_once()

    @patch("app.lib.shared.format.reason")
    @patch("app.tools.web_fetch.resolve_input")
    async def test_llm_failure_fails_fetch(
        self, mock_resolve: MagicMock, mock_reason: MagicMock
    ) -> None:
        """A failed answer returns an error."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "paragraph",
                    "children": [{"text": "This is the page content."}]
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)
        mock_reason.return_value = ("backend unavailable", 1)

        result = await run_text(web_fetch_tool, "http://example.com", prompt="summarize")

        assert result == (
            "Tool 'web_fetch' failed: "
            "Error extracting from http://example.com: backend unavailable"
        )

    @patch("app.tools.web_fetch.resolve_input")
    async def test_llm_skipped_when_processed_true(self, mock_resolve: MagicMock) -> None:
        """Content that is already worked on comes back as markdown even with a prompt."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "paragraph",
                    "children": [{"text": "Already processed content."}]
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), True)

        result = await run_text(web_fetch_tool, "http://example.com", prompt="summarize")

        assert "Already processed content." in result
        assert len(result) < len(json.dumps(tree))

    @patch("app.tools.web_fetch.resolve_input")
    async def test_llm_not_called_without_prompt(self, mock_resolve: MagicMock) -> None:
        """With no prompt, the page content comes back as markdown."""
        tree = {
            "role": "document",
            "children": [
                {
                    "role": "paragraph",
                    "children": [{"text": "Regular content."}]
                }
            ]
        }
        mock_resolve.return_value = (json.dumps(tree), False)

        result = await run_text(web_fetch_tool, "http://example.com")

        assert "Regular content." in result


class TestPromptParamDocs:
    """The prompt usage guidance lives on the prompt param, not the tool description."""

    def test_prompt_param_carries_usage_guidance(self) -> None:
        doc = next(p.description for p in web_fetch_tool.params if p.name == "prompt")
        assert "specific answer or summary" in doc
        assert "What is the price of the product?" in doc
        assert "reading an article or browsing a full webpage" in doc

    def test_prompt_empty_default_gets_bare_optional_note(self) -> None:
        """The '' default adds the bare Optional note. The wire text is the base plus the note."""
        p = next(p for p in web_fetch_tool.params if p.name == "prompt")
        assert p.wire_description == p.description + " Optional."

    def test_tool_description_has_no_prompt_guidance(self) -> None:
        assert "Set the prompt parameter" not in web_fetch_tool.description
        assert "prompt='What is the price" not in web_fetch_tool.description

class TestBinaryPassthrough:
    """Tests for fetching a binary file."""

    @patch("app.tools.web_fetch.resolve_input")
    async def test_binary_fetch_returns_toolresult(
        self,
        mock_resolve: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A binary fetch with no prompt returns the file bytes and a download note."""
        res = tmp_path / "resources"
        (res / "downloads").mkdir(parents=True)
        local = res / "downloads" / "a.pdf"
        local.write_bytes(b"%PDF-1.4" + b"\x00" * 8)
        monkeypatch.setenv("RESOURCES_DIR", str(res))
        mock_resolve.return_value = (BINARY_PASSTHROUGH_TAG + str(local), False)

        result = await web_fetch_tool.call({"url": "https://example.com/a.pdf"})

        assert isinstance(result, ToolResult)
        assert result.files == ("downloads/a.pdf",)
        assert result.render_type is None
        assert "Binary file with size" in result.text
        assert "downloads/a.pdf" in result.text
        mock_resolve.assert_called_once_with(
            "https://example.com/a.pdf", context="", verbose=True, binary_passthrough=True
        )

    @patch("app.lib.shared.format.reason")
    @patch("app.tools.web_fetch.resolve_input")
    async def test_prompted_fetch_returns_str(
        self,
        mock_resolve: MagicMock,
        mock_reason: MagicMock,
    ) -> None:
        """A fetch with a prompt returns text, not a file."""
        mock_resolve.return_value = ("plain text content", False)
        mock_reason.return_value = ("The answer.", 0)

        result = await web_fetch_tool.call({"url": "http://example.com", "prompt": "summarize"})

        assert isinstance(result, ToolResult)
        assert result.files is None
        assert result.text == "The answer."
        assert result.render_type == "markdown"
        mock_resolve.assert_called_once_with(
            "http://example.com", context="summarize", verbose=True, binary_passthrough=False
        )


class TestDeclaredRenderType:
    """The wrapper stamps the kind the final text actually has."""

    @patch("app.tools.web_fetch.resolve_input")
    async def test_converted_tree_declares_markdown(self, mock_resolve: MagicMock) -> None:
        tree = {
            "role": "document",
            "children": [{"role": "heading", "name": "H", "children": [{"text": "H"}]}],
        }
        mock_resolve.return_value = (json.dumps(tree), False)
        result = await web_fetch_tool.call({"url": "http://example.com", "prompt": ""})
        assert isinstance(result, ToolResult)
        assert result.render_type == "markdown"
        assert "# H" in result.text

    @patch("app.tools.web_fetch.resolve_input")
    async def test_verbatim_json_declares_json(self, mock_resolve: MagicMock) -> None:
        raw_json = json.dumps({"current_condition": [{"temp_C": "25"}]})
        mock_resolve.return_value = (raw_json, False)
        result = await web_fetch_tool.call({"url": "http://example.com", "prompt": ""})
        assert result.render_type == "json"
        assert result.text == raw_json

    @patch("app.tools.web_fetch.resolve_input")
    async def test_plain_text_stays_undeclared(self, mock_resolve: MagicMock) -> None:
        mock_resolve.return_value = ("Plain text content", False)
        result = await web_fetch_tool.call({"url": "http://example.com", "prompt": ""})
        assert result.render_type is None
        assert result.text == "Plain text content"

    @patch("app.tools.web_fetch.resolve_input")
    async def test_scalar_text_stays_undeclared(self, mock_resolve: MagicMock) -> None:
        mock_resolve.return_value = ("5", False)
        result = await web_fetch_tool.call({"url": "http://example.com", "prompt": ""})
        assert result.render_type is None
        assert result.text == "5"

    @patch("app.lib.shared.format.reason")
    @patch("app.tools.web_fetch.resolve_input")
    async def test_llm_answer_declares_markdown_even_over_a_converted_tree(
        self, mock_resolve: MagicMock, mock_reason: MagicMock
    ) -> None:
        tree = {
            "role": "document",
            "children": [{"role": "paragraph", "children": [{"text": "page body"}]}],
        }
        mock_resolve.return_value = (json.dumps(tree), False)
        mock_reason.return_value = ("An answer.", 0)
        result = await web_fetch_tool.call({"url": "http://example.com", "prompt": "summarize"})
        assert result.text == "An answer."
        assert result.render_type == "markdown"
