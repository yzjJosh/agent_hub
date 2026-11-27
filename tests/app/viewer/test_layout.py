"""Tests for shared viewer HTML layout builders and fallback file routing."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from app.viewer.layout import (
    VIEWER_STYLESHEET_LINKS,
    build_page,
    build_solo_download_cluster,
    build_theme_button,
    build_top_bar,
    escape_html,
    format_byte_size,
    serve_fallback_file,
)
from tests.app.conftest import configure_resource_env


class _RecordingFileOpener:
    """Capture fallback requests delegated to a file viewer handler."""

    def __init__(self, body: str = "Opened file") -> None:
        self.requests: list[Request] = []
        self._body: str = body

    async def open_file(self, request: Request) -> Response:
        """Record the delegated request and return a plain text response."""
        self.requests.append(request)
        return PlainTextResponse(self._body, status_code=200)


def _build_http_request(path: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "headers": [],
        "query_string": b"",
    }
    return Request(scope)


def test_viewer_stylesheet_links_includes_tokens_and_view_css() -> None:
    """Include both tokens and view stylesheets in the shared head links."""
    assert '<link rel="stylesheet" href="/assets/css/tokens.css">' in VIEWER_STYLESHEET_LINKS
    assert '<link rel="stylesheet" href="/assets/css/view.css">' in VIEWER_STYLESHEET_LINKS


def test_escape_html_escapes_tags_and_quotes() -> None:
    """Escape HTML special characters and double quotes."""
    assert escape_html('a"b<c>&') == "a&quot;b&lt;c&gt;&amp;"


@pytest.mark.parametrize(
    ("byte_count", "expected"),
    [
        (0, "0 B"),
        (999, "999 B"),
        (1023, "1023 B"),
        (1024, "1 KB"),
        (2048, "2 KB"),
        (1048575, "1024 KB"),
        (1048576, "1.0 MB"),
        (1572864, "1.5 MB"),
    ],
)
def test_format_byte_size_units(byte_count: int, expected: str) -> None:
    """Format byte counts in bytes, kilobytes, and megabytes."""
    assert format_byte_size(byte_count) == expected


def test_build_top_bar_includes_brand_and_actions() -> None:
    """Render the brand home link and supplied action buttons inside the header bar."""
    bar_html = build_top_bar(build_theme_button())
    assert bar_html.startswith('<header class="vp-topbar"><a class="vp-brand" href="/tools" title="Home">')
    assert "data-vp-theme" in bar_html
    assert bar_html.endswith("</div></header>")


def test_build_solo_download_cluster_renders_download_link() -> None:
    """Render the solo download cluster with the provided download link."""
    cluster_html = build_solo_download_cluster("/api/tasks/abc123/result/download")
    assert cluster_html == (
        '<div class="vp-solo-cluster"><a class="vp-solo-dl" href="/api/tasks/abc123/result/download" '
        'title="Download" aria-label="Download"><span data-vp-icon="download"></span></a></div>'
    )


def test_build_page_solo_shell_omits_footer_and_sets_body_class() -> None:
    """Render a standalone solo page without the retention footer."""
    page_html = build_page(
        "Solo Title",
        build_top_bar(build_theme_button()),
        '<main class="vp-main"></main>',
        should_include_footer=False,
        body_class="view-page vp-solo",
    )
    assert page_html.startswith("<!doctype html>")
    assert "/assets/css/tokens.css" in page_html
    assert "/assets/css/view.css" in page_html
    assert "view-page.js" in page_html
    assert 'class="view-page vp-solo"' in page_html
    assert "vp-footer" not in page_html


def test_build_page_includes_footer_by_default() -> None:
    """Include the retention footer when should_include_footer is left at its default."""
    page_html = build_page("Default Title", "", '<main class="vp-main"></main>')
    assert 'class="vp-footer"' in page_html
    assert 'class="view-page"' in page_html


@pytest.mark.parametrize(
    ("request_path", "literal_path"),
    [
        ("/view/result/missing", "result/missing"),
        ("/view/result/../secret", "../secret"),
    ],
)
async def test_serve_fallback_file_returns_missing_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request_path: str,
    literal_path: str,
) -> None:
    """Return the caller missing response when no resource file matches."""
    configure_resource_env(tmp_path, monkeypatch)
    missing_response = PlainTextResponse("Expired", status_code=200)
    opener = _RecordingFileOpener()
    response = await serve_fallback_file(
        _build_http_request(request_path),
        literal_path,
        missing_response,
        opener.open_file,
    )
    assert response is missing_response
    assert opener.requests == []


async def test_serve_fallback_file_returns_not_found_for_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a plain 404 response when the matching resource path is a directory."""
    resources_dir, _ = configure_resource_env(tmp_path, monkeypatch)
    (resources_dir / "result" / "folder").mkdir(parents=True)
    missing_response = PlainTextResponse("Expired", status_code=200)
    opener = _RecordingFileOpener()
    response = await serve_fallback_file(
        _build_http_request("/view/result/folder"),
        "result/folder",
        missing_response,
        opener.open_file,
    )
    assert response.status_code == 404
    assert response.body == b"Not Found"
    assert opener.requests == []


async def test_serve_fallback_file_delegates_to_open_file_for_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate to open_file with a rewritten request when a matching resource file exists."""
    resources_dir, _ = configure_resource_env(tmp_path, monkeypatch)
    target_file = resources_dir / "result" / "notes.txt"
    target_file.parent.mkdir(parents=True)
    target_file.write_text("hello", encoding="utf-8")
    missing_response = PlainTextResponse("Expired", status_code=200)
    opener = _RecordingFileOpener()
    response = await serve_fallback_file(
        _build_http_request("/view/result/notes.txt"),
        "result/notes.txt",
        missing_response,
        opener.open_file,
    )
    assert response.status_code == 200
    assert response.body == b"Opened file"
    assert len(opener.requests) == 1
    assert opener.requests[0].url.path == "/view/result/notes.txt"
    assert opener.requests[0].path_params == {"path": "result/notes.txt"}
