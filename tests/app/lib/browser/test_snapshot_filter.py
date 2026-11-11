"""Tests for turning a web page into a trimmed tree."""

import json
import pytest
from app.lib.browser.snapshot_filter import html_to_json, filter_html_file
from typing import Self
from pathlib import Path


class TestRoleBasedFiltering:
    """Tests for dropping parts of a page by what they are."""

    async def test_separator_role_filtered(self, tmp_path: Path) -> None:
        """Divider lines are dropped and text is kept."""
        p = tmp_path / "sep.html"
        p.write_text("<html><body><hr><p>Content</p></body></html>")
        result = await filter_html_file(str(p))
        result_str = json.dumps(result)
        assert "Content" in result_str
        assert "separator" not in result_str

    async def test_navigation_short_text_filtered(self, tmp_path: Path) -> None:
        """A menu with little text is dropped."""
        p = tmp_path / "nav.html"
        p.write_text("<html><body><nav>Go</nav><p>Content</p></body></html>")
        result = await filter_html_file(str(p))
        result_str = json.dumps(result)
        assert "navigation" not in result_str

    async def test_link_with_url_preserved(self, tmp_path: Path) -> None:
        """A link with an address is kept even inside a dropped menu."""
        p = tmp_path / "link.html"
        p.write_text("<html><body><nav><a href='/test'>Go</a></nav></body></html>")
        result = await filter_html_file(str(p))
        result_str = json.dumps(result)
        assert "link" in result_str
        assert "/test" in result_str

    async def test_navigation_substantial_text_preserved(self, tmp_path: Path) -> None:
        """A menu with a lot of text is kept."""
        p = tmp_path / "nav_long.html"
        p.write_text(
            "<html><body><nav>This is a very long navigation text that exceeds "
            "the substantial text threshold for filtering</nav></body></html>"
        )
        result = await filter_html_file(str(p))
        result_str = json.dumps(result)
        assert "navigation" in result_str

    async def test_listitem_short_filtered(self, tmp_path: Path) -> None:
        """A list item with a lot of text is kept."""
        p = tmp_path / "li.html"
        p.write_text("<html><body><ul><li>X</li><li>Longer item text here</li></ul></body></html>")
        result = await filter_html_file(str(p))
        result_str = json.dumps(result)
        assert "Longer item" in result_str

    async def test_list_short_filtered(self, tmp_path: Path) -> None:
        """A list with little text is dropped."""
        p = tmp_path / "list.html"
        p.write_text("<html><body><ul>X</ul></body></html>")
        result = await filter_html_file(str(p))
        result_str = json.dumps(result) if result else ""
        assert "list" not in result_str


class TestDuplicateAndMerge:
    """Tests for repeated text and joined text."""

    async def test_duplicate_text_with_content_preserved(self, tmp_path: Path) -> None:
        """Repeated text in headings is kept."""
        p = tmp_path / "dup.html"
        p.write_text(
            "<html><body><h1>Hello World</h1><h1>Hello World</h1></body></html>"
        )
        result = await filter_html_file(str(p))
        result_str = json.dumps(result)
        assert result_str.count("heading") == 2

    async def test_links_same_text_different_urls_preserved(self, tmp_path: Path) -> None:
        """Two links with the same text but different addresses are both kept."""
        p = tmp_path / "links.html"
        p.write_text(
            "<html><body>"
            "<a href='/page1'>Learn More</a>"
            "<a href='/page2'>Learn More</a>"
            "</body></html>"
        )
        result = await filter_html_file(str(p))
        assert "/page1" in json.dumps(result)
        assert "/page2" in json.dumps(result)
        link_nodes = [c for c in result.get("children", []) if c.get("role") == "link"]
        assert len(link_nodes) == 2

    async def test_empty_text_not_duplicated(self, tmp_path: Path) -> None:
        """Text still comes through when the page also has empty parts."""
        p = tmp_path / "empty.html"
        p.write_text(
            "<html><body><span></span><span></span><p>Content</p></body></html>"
        )
        result = await filter_html_file(str(p))
        result_str = json.dumps(result)
        assert "Content" in result_str

    async def test_consecutive_text_merged(self, tmp_path: Path) -> None:
        """Words split by spaces come back as one piece of text."""
        p = tmp_path / "merge.html"
        p.write_text("<html><body><p>Hello World</p></body></html>")
        result = await filter_html_file(str(p))
        result_str = json.dumps(result)
        assert "Hello World" in result_str

    async def test_text_merging_across_elements(self, tmp_path: Path) -> None:
        """All the text inside one part comes back joined."""
        p = tmp_path / "merge2.html"
        p.write_text(
            "<html><body><p>Hello World More text</p></body></html>"
        )
        result = await filter_html_file(str(p))
        result_str = json.dumps(result)
        assert "Hello World More text" in result_str

    async def test_element_no_text_handled(self, tmp_path: Path) -> None:
        """An empty part does not stop the rest of the page coming through."""
        p = tmp_path / "notext.html"
        p.write_text("<html><body><div></div><p>Content</p></body></html>")
        result = await filter_html_file(str(p))
        result_str = json.dumps(result)
        assert "Content" in result_str


class TestHiddenElementFiltering:
    """Tests for dropping hidden parts of a page."""

    async def test_display_none_with_space(self: Self) -> None:
        """Parts set to display none are dropped."""
        html = '<html><body><div><p style="display: none">Hidden</p><p>Visible</p></div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Hidden" not in result_str
        assert "Visible" in result_str

    async def test_display_none_no_space(self: Self) -> None:
        """Parts set to display none with no space are dropped."""
        html = '<html><body><div><p style="display:none">Hidden</p><p>Visible</p></div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Hidden" not in result_str
        assert "Visible" in result_str

    async def test_display_none_uppercase(self: Self) -> None:
        """Parts set to display none in capitals are dropped."""
        html = '<html><body><div><p style="DISPLAY:NONE">Hidden</p><p>Visible</p></div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Hidden" not in result_str
        assert "Visible" in result_str

    async def test_display_none_with_other_properties(self: Self) -> None:
        """Parts set to display none among other styles are dropped."""
        html = '<html><body><div><p style="display: none; color: red">Hidden</p><p>Visible</p></div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Hidden" not in result_str
        assert "Visible" in result_str

    async def test_display_none_with_important(self: Self) -> None:
        """Parts set to display none as important are dropped."""
        html = '<html><body><div><p style="display: none !important">Hidden</p><p>Visible</p></div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Hidden" not in result_str
        assert "Visible" in result_str

    async def test_visibility_hidden(self: Self) -> None:
        """Parts set to visibility hidden are dropped."""
        html = '<html><body><div><p style="visibility: hidden">Hidden</p><p>Visible</p></div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Hidden" not in result_str
        assert "Visible" in result_str

    async def test_visibility_hidden_no_space(self: Self) -> None:
        """Parts set to visibility hidden with no space are dropped."""
        html = '<html><body><div><p style="visibility:hidden">Hidden</p><p>Visible</p></div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Hidden" not in result_str
        assert "Visible" in result_str

    async def test_aria_hidden_true(self: Self) -> None:
        """Parts marked hidden for screen readers are dropped."""
        html = '<html><body><div><p aria-hidden="true">Hidden</p><p>Visible</p></div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Hidden" not in result_str
        assert "Visible" in result_str

    async def test_aria_hidden_without_style(self: Self) -> None:
        """Parts marked hidden with no style of their own are dropped."""
        html = '<html><body><div><span aria-hidden="true">Hidden icon</span><p>Visible</p></div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Hidden icon" not in result_str
        assert "Visible" in result_str

    async def test_children_of_hidden_element_removed(self: Self) -> None:
        """Everything inside a hidden part is dropped too."""
        html = '<html><body><div style="display: none"><p>Child1</p><p>Child2</p></div><p>Visible</p></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Child1" not in result_str
        assert "Child2" not in result_str
        assert "Visible" in result_str

    async def test_visible_elements_preserved(self: Self) -> None:
        """Parts that are not hidden are kept."""
        html = '<html><body><div><p>Paragraph</p><a href="#">Link</a><span>Span</span></div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Paragraph" in result_str
        assert "Link" in result_str
        assert "Span" in result_str

    async def test_empty_style_attribute(self: Self) -> None:
        """Parts with an empty style are kept."""
        html = '<html><body><div style="">Visible</div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Visible" in result_str

    async def test_none_style_attribute(self: Self) -> None:
        """Parts with no style are kept."""
        html = '<html><body><div>Visible</div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Visible" in result_str

    async def test_style_with_other_properties_only(self: Self) -> None:
        """Parts with other styles are kept."""
        html = '<html><body><div style="color: red; font-size: 14px">Visible</div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Visible" in result_str

    async def test_display_block_preserved(self: Self) -> None:
        """Parts set to display block are kept."""
        html = '<html><body><div style="display: block">Visible</div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Visible" in result_str

    async def test_display_flex_preserved(self: Self) -> None:
        """Parts set to display flex are kept."""
        html = '<html><body><div style="display: flex">Visible</div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Visible" in result_str

    async def test_aria_hidden_false_preserved(self: Self) -> None:
        """Parts marked as not hidden for screen readers are kept."""
        html = '<html><body><div aria-hidden="false">Visible</div></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Visible" in result_str

    async def test_mixed_hidden_and_visible(self: Self) -> None:
        """Only the parts that show stay in the tree."""
        html = '''<html><body>
            <div style="display:none">Hidden display none</div>
            <div style="visibility:hidden">Hidden visibility</div>
            <div aria-hidden="true">Hidden aria</div>
            <h1>Visible Title</h1>
            <p>Visible paragraph</p>
        </body></html>'''
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Hidden display none" not in result_str
        assert "Hidden visibility" not in result_str
        assert "Hidden aria" not in result_str
        assert "Visible Title" in result_str
        assert "Visible paragraph" in result_str

    async def test_no_hidden_elements_still_works(self: Self) -> None:
        """A page with nothing hidden still turns into a tree."""
        html = '<html><body><h1>Title</h1><p>Content</p></body></html>'
        result = await html_to_json(html)
        result_str = json.dumps(result)
        assert "Title" in result_str
        assert "Content" in result_str


class TestHTMLConversion:
    """Tests for turning a page into a tree."""

    async def test_html_to_json_basic(self: Self) -> None:
        """A simple page turns into a tree with a heading in it."""
        html = "<html><body><h1>Title</h1><p>Paragraph</p></body></html>"
        result = await html_to_json(html)
        assert result["role"] == "generic"  # body tag
        assert len(result["children"]) == 2
        assert result["children"][0]["role"] == "heading"
        assert result["children"][0]["name"] == "Title"

    async def test_html_to_json_metadata(self: Self) -> None:
        """A part is named from its alt, title, or label text."""
        html = """
        <img src="test.jpg" alt="Alt Text">
        <div title="Title Text">Content</div>
        <button aria-label="Label Text">Click</button>
        """
        result = await html_to_json(html)
        children = result["children"]
        assert children[0]["name"] == "Alt Text"
        assert children[1]["name"] == "Title Text"
        assert children[2]["name"] == "Label Text"

    async def test_html_to_json_complex_roles(self: Self) -> None:
        """Parts get what they are from their class names and tags."""
        html = """
        <div class="top-bar">Banner</div>
        <div class="sidebar">Aside</div>
        <div class="post">Article</div>
        <section>Region</section>
        """
        result = await html_to_json(html)
        roles = [c.get("role") for c in result["children"]]
        assert "banner" in roles
        assert "complementary" in roles
        assert "article" in roles
        assert "region" in roles

    async def test_html_to_json_cleaning(self: Self) -> None:
        """Script and style blocks are dropped."""
        html = "<html><body><script>alert(1)</script><style>body{}</style><p>Keep</p></body></html>"
        result = await html_to_json(html)
        assert len(result["children"]) == 1
        assert result["children"][0]["role"] == "paragraph"

    async def test_html_to_json_comments(self: Self) -> None:
        """Page comments are dropped."""
        html = "<body><!-- comment --><p>Text</p></body>"
        result = await html_to_json(html)
        assert len(result["children"]) == 1
        assert result["children"][0]["role"] == "paragraph"

    async def test_html_to_json_roles(self: Self) -> None:
        """Parts get what they are from class or role, and a link keeps its address."""
        html = """
        <div class="header">Banner</div>
        <div role="navigation">Nav</div>
        <a href="/test">Link Text</a>
        """
        result = await html_to_json(html)
        children = result["children"]
        roles = [c.get("role") for c in children]
        assert "banner" in roles
        assert "navigation" in roles
        assert "link" in roles

        link_node = next(c for c in children if c.get("role") == "link")
        assert link_node["name"] == "Link Text"
        assert link_node["url"] == "/test"


class TestFilterHTMLFile:
    """Tests for filtering a whole page file."""

    async def test_filter_html_file(self, tmp_path: Path) -> None:
        """A whole page file turns into a document tree."""
        d = tmp_path / "sub"
        d.mkdir()
        p = d / "test.html"
        p.write_text(
            "<html><body><nav>Short</nav>"
            "<main><h1>Substantial Title for Testing</h1></main>"
            "</body></html>"
        )
        result = await filter_html_file(str(p))
        assert result["role"] == "document"
        assert len(result["children"]) == 2
        assert "Substantial" in json.dumps(result)

    async def test_filter_html_file_verbose(self, tmp_path: Path) -> None:
        """The tree still comes back with logging turned on."""
        p = tmp_path / "test_v.html"
        p.write_text("<html><body><h1>Title</h1></body></html>")
        result = await filter_html_file(str(p), verbose=True)
        assert "Title" in json.dumps(result)

    async def test_filter_html_file_multiple_roots(self, tmp_path: Path) -> None:
        """Two top parts are wrapped in one document."""
        p = tmp_path / "test2.html"
        p.write_text("<html><body><h1>Title 1</h1><h1>Title 2</h1></body></html>")
        result = await filter_html_file(str(p))
        assert result["role"] == "document"
        assert len(result["children"]) == 2

    async def test_filter_html_file_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A missing file gives back nothing and says so."""
        result = await filter_html_file("/non/existent/path", verbose=True)
        assert result is None
        captured = capsys.readouterr()
        assert "HTML extraction failed" in captured.err

    async def test_hierarchy_preservation(self, tmp_path: Path) -> None:
        """Parts stay inside their parent."""
        p = tmp_path / "hierarchy.html"
        p.write_text(
            "<html><body>"
            "<main>"
            "<h1>Title</h1>"
            "<article><p>Body</p></article>"
            "</main>"
            "</body></html>"
        )
        result = await filter_html_file(str(p))
        assert result["role"] == "main"
        assert len(result["children"]) == 2


class TestXiaohongshuLiveFiltering:
    """Tests for filtering a saved real page."""

    async def test_xiaohongshu_hidden_modals_removed(self, xiaohongshu_html: str) -> None:
        """Hidden pop ups are dropped from a saved real page."""
        result = await filter_html_file(xiaohongshu_html)
        result_str = json.dumps(result, ensure_ascii=False)

        assert "我要申诉" not in result_str
        assert "温馨提示" not in result_str

    async def test_xiaohongshu_visible_content_preserved(self, xiaohongshu_html: str) -> None:
        """A saved real page still has content after filtering."""
        result = await filter_html_file(xiaohongshu_html)
        result_str = json.dumps(result, ensure_ascii=False)

        assert len(result_str) > 100


class TestRednoteDumpFiles:
    """Tests for filtering saved page dumps."""

    async def test_rednote_dump_html_no_comments(self, rednote_dump_html: str) -> None:
        """A saved dump keeps its note text and its loading mark."""
        result = await filter_html_file(rednote_dump_html, verbose=True)

        result_str = json.dumps(result, ensure_ascii=False)

        assert "Cupertino" in result_str

        assert "加载中" in result_str
