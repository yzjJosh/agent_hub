"""Tests for the shared view page chrome."""
from __future__ import annotations

from app.viewer.view_chrome import esc, fmt_size, page, theme_btn, top_bar


def test_esc_html_and_quotes() -> None:
    assert esc('a"b<c>&') == "a&quot;b&lt;c&gt;&amp;"


def test_fmt_size_units() -> None:
    assert fmt_size(999) == "999 B"
    assert fmt_size(2048) == "2 KB"
    assert fmt_size(1572864) == "1.5 MB"


def test_top_bar_brand_and_actions() -> None:
    bar = top_bar(theme_btn())
    assert bar.startswith('<header class="vp-topbar"><a class="vp-brand" href="/tools" title="Home">')
    assert 'data-vp-theme' in bar


def test_page_solo_shell_is_cold_visitor_safe() -> None:
    html = page("t", top_bar(theme_btn()), '<main class="vp-main"></main>',
                footer=False, body_cls="view-page vp-solo")
    assert html.startswith('<!doctype html>')
    assert '/assets/css/tokens.css' in html and '/assets/css/view.css' in html
    assert 'view-page.js' in html and 'class="view-page vp-solo"' in html
