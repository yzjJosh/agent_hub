"""Tests for the markdown engine."""

from markdown_it import MarkdownIt
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

import pytest
import yaml

from app.lib.render.markdown_engine import markdown_builder


def test_markdown_gfm_like_with_linkify_renders_gfm() -> None:
    """The markdown builder renders tables, strikethrough, and task lists."""
    md = markdown_builder()
    assert "<table>" in md.render("| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert md.render("~~x~~").find("<s>") != -1
    assert md.render("- [x] done\n- [ ] todo\n").count("<li") == 2


def test_bare_url_becomes_new_tab_anchor() -> None:
    """A bare web link in the text renders as an anchor that opens in a new tab."""
    md = markdown_builder()
    expected = '<a href="https://example.com/path?q=1" target="_blank" rel="noopener">'
    assert expected in md.render("see https://example.com/path?q=1 now")


def test_fenced_code_keeps_bare_url_plain() -> None:
    """A bare web link inside a fenced code block renders without an anchor."""
    md = markdown_builder()
    out = md.render("```\nhttps://example.com\n```\n")
    assert "https://example.com" in out
    assert "<a" not in out


def test_explicit_link_is_not_double_linked() -> None:
    """An explicit markdown link renders exactly one anchor that opens in a new tab."""
    md = markdown_builder()
    out = md.render("[text](https://example.com)")
    assert out.count('<a href="https://example.com"') == 1
    assert 'target="_blank"' in out


@pytest.mark.parametrize(
    ("text", "href"),
    [
        ("www.example.com", "http://www.example.com"),
        ("example.com", "http://example.com"),
        ("foo@bar.com", "mailto:foo@bar.com"),
    ],
)
def test_bare_web_address_and_email_become_new_tab_anchors(text: str, href: str) -> None:
    """A bare web address or email renders as an anchor that opens in a new tab."""
    md = markdown_builder()
    out = md.render(f"see {text} now")
    assert f'<a href="{href}" target="_blank" rel="noopener">{text}</a>' in out


@pytest.mark.parametrize("text", ["v1.2.3", "example.zz", "a.b.c.", "localhost:8080"])
def test_dotted_text_that_is_not_a_link_stays_plain(text: str) -> None:
    """A version or dotted string that is not a link renders without an anchor."""
    md = markdown_builder()
    out = md.render(f"see {text} now")
    assert text in out
    assert "<a" not in out


def test_pygments_and_yaml_importable() -> None:
    assert isinstance(markdown_builder(), MarkdownIt)
    assert yaml.safe_load("a: 1") == {"a": 1}

    assert get_lexer_by_name("python").aliases[0] == "python"
    assert get_lexer_by_name("json").aliases[0] == "json"
    assert HtmlFormatter(style="default").get_style_defs(".code-block")
    assert HtmlFormatter(style="github-dark").get_style_defs(".code-block")
