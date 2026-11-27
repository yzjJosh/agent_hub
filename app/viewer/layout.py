"""Shared HTML layout builders and fallback routing for viewer pages."""

from __future__ import annotations

import asyncio
import html
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from app.lib.render.descriptor import resolve_view

__all__ = [
    "VIEWER_STYLESHEET_LINKS",
    "build_page",
    "build_solo_download_cluster",
    "build_theme_button",
    "build_top_bar",
    "escape_html",
    "format_byte_size",
    "serve_fallback_file",
]

VIEWER_STYLESHEET_LINKS = (
    '<link rel="stylesheet" href="/assets/css/tokens.css">'
    '<link rel="stylesheet" href="/assets/css/view.css">'
)

_BYTES_PER_KILOBYTE = 1024
_BYTES_PER_MEGABYTE = 1024 * 1024

_INITIAL_THEME_SCRIPT = (
    '<script>(function(){var r=document.documentElement,t="light",s=null;'
    'try{t=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";}catch(e){t="light";}'
    'try{s=JSON.parse(localStorage.getItem("agent-console-theme"));}catch(e){s=null;}'
    'if(s&&(s.theme==="dark"||s.theme==="light")&&s.scheme===t){t=s.theme;}'
    'r.setAttribute("data-theme",t);})();</script>'
)

_BRAND_MARKUP = (
    '<span class="vp-brand-mark">'
    '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 5 4 3-4 3"/>'
    '<path d="M9 12h4"/></svg></span>'
    '<span class="vp-brand-name">Agent Hub</span>'
    '<span class="vp-brand-sub">· file viewer</span>'
)

_FOOTER_NOTE = "Served by Agent Hub · Files are retained for 30 days and may be removed without notice."
_FOOTER_NOTE_MOBILE = "Served by Agent Hub · 30-day retention"


@dataclass(frozen=True)
class _FallbackPathStatus:
    """Whether a resolved fallback path exists and whether it is a directory."""

    is_present: bool
    is_directory: bool


def _probe_fallback_path_sync(full_path: str) -> _FallbackPathStatus:
    return _FallbackPathStatus(
        is_present=os.path.exists(full_path),
        is_directory=os.path.isdir(full_path),
    )


def escape_html(text: str) -> str:
    """Escape text so it is safe to embed in HTML.

    Args:
        text: Text to escape.

    Returns:
        Escaped HTML text.
    """
    return html.escape(str(text), quote=True)


def build_top_bar(*actions: str) -> str:
    """Build the viewer page header bar with the given action controls.

    Args:
        *actions: HTML fragments for buttons placed on the right side of the bar.

    Returns:
        Header bar HTML string.
    """
    return (
        f'<header class="vp-topbar"><a class="vp-brand" href="/tools" title="Home">{_BRAND_MARKUP}</a>'
        f'<div class="vp-actions">{"".join(actions)}</div></header>'
    )


def _build_footer() -> str:
    return (
        f'<footer class="vp-footer"><span class="vp-footer-note">{_FOOTER_NOTE}</span>'
        f'<span class="vp-footer-note-mobile">{_FOOTER_NOTE_MOBILE}</span></footer>'
    )


def build_theme_button() -> str:
    """Build the button that switches between light and dark themes.

    Returns:
        Theme toggle button HTML string.
    """
    return '<button class="vp-icon-btn" data-vp-theme title="Switch theme"></button>'


def build_solo_download_cluster(download_href: str) -> str:
    """Build the floating download button cluster for solo viewer pages.

    Args:
        download_href: Escaped URL for the download link.

    Returns:
        Download cluster HTML string.
    """
    return (
        f'<div class="vp-solo-cluster"><a class="vp-solo-dl" href="{download_href}" '
        'title="Download" aria-label="Download"><span data-vp-icon="download"></span></a></div>'
    )


def build_page(
    title: str,
    top_html: str,
    main_html: str,
    *,
    should_include_footer: bool = True,
    body_class: str = "view-page",
) -> str:
    """Build a complete HTML document for a viewer page.

    Args:
        title: Page title shown in the browser tab.
        top_html: Header bar or floating action markup placed above the main area.
        main_html: Main content HTML for the page body.
        should_include_footer: Whether to append the retention footer at the bottom.
        body_class: CSS classes for the body element.

    Returns:
        Complete HTML document string.
    """
    footer_html = _build_footer() if should_include_footer else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        f"<title>{escape_html(title)}</title>{_INITIAL_THEME_SCRIPT}{VIEWER_STYLESHEET_LINKS}</head>"
        f'<body class="{body_class}">{top_html}{main_html}{footer_html}'
        '<script type="module" src="/assets/js/view-page.js"></script></body></html>'
    )


def format_byte_size(byte_count: int) -> str:
    """Format a byte count as a short readable size string.

    Args:
        byte_count: Number of bytes.

    Returns:
        Formatted size text in bytes, kilobytes, or megabytes.
    """
    if byte_count < _BYTES_PER_KILOBYTE:
        return f"{byte_count} B"
    if byte_count < _BYTES_PER_MEGABYTE:
        return f"{round(byte_count / _BYTES_PER_KILOBYTE)} KB"
    return f"{byte_count / _BYTES_PER_MEGABYTE:.1f} MB"


async def serve_fallback_file(
    request: Request,
    literal_path: str,
    missing_response: Response,
    open_file: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Route an unmatched viewer URL to a matching resource file when one exists.

    Args:
        request: Incoming HTTP request.
        literal_path: Resource path corresponding to the requested viewer URL.
        missing_response: Response returned when no matching resource path exists.
        open_file: Async file viewer endpoint called when a matching file exists.

    Returns:
        File viewer response, plain 404 response for directories, or missing_response.
    """
    full_path = await resolve_view(literal_path)
    if full_path is None:
        return missing_response
    path_status = await asyncio.to_thread(_probe_fallback_path_sync, full_path)
    if not path_status.is_present:
        return missing_response
    if path_status.is_directory:
        return PlainTextResponse("Not Found", status_code=404)
    view_path = "/view/" + literal_path
    scope = dict(
        request.scope,
        path=view_path,
        raw_path=view_path.encode("utf-8"),
        path_params={"path": literal_path},
    )
    return await open_file(Request(scope))
