"""Solo resource file viewer page and shared viewer mode fragments."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, Response

from app.lib.render.descriptor import build_descriptor, resolve_view
from app.viewer.layout import (
    build_page,
    build_solo_download_cluster,
    build_theme_button,
    build_top_bar,
    escape_html,
)

__all__ = [
    "build_download_href",
    "render_convert_failed_html",
    "render_converting_html",
    "render_expired_page",
    "render_pdf_container_html",
    "render_text_well_html",
    "view_file",
]

_TEXT_WELL_MODES = ("doc", "code", "table", "plain")


@dataclass(frozen=True)
class _FilePathStatus:
    """Whether a resolved path on disk is a directory or an existing path."""

    is_directory: bool
    is_present: bool


def _probe_file_path_sync(full_path: str) -> _FilePathStatus:
    return _FilePathStatus(
        is_directory=os.path.isdir(full_path),
        is_present=os.path.exists(full_path),
    )


def build_download_href(resource_path: str) -> str:
    """Build the download link for a resource file.

    Args:
        resource_path: Path inside the resources folder.

    Returns:
        The link to download the file.
    """
    return f"/dav/dl?path={escape_html(quote(resource_path, safe='/'))}"


def render_expired_page() -> Response:
    """Build the page shown when a file link is no longer valid.

    Returns:
        An HTML response with the expired link notice.
    """
    top_html = build_top_bar(build_theme_button())
    main_html = (
        '<main class="vp-main vp-expired">'
        '<div class="vp-expired-inner">'
        '<span class="vp-expired-ic" data-vp-icon="alert"></span>'
        '<h1 class="vp-expired-title">This link is no longer valid</h1>'
        '<p class="vp-expired-sub">The file may have been garbage-collected — files are retained '
        'for 30 days and then removed. Ask the sender for a fresh link.</p>'
        '<button class="btn ghost" data-vp-back>Go back</button>'
        "</div></main>"
    )
    return HTMLResponse(build_page("Link no longer valid", top_html, main_html))


def render_text_well_html(
    descriptor: dict,
    *,
    extra_class: str = "",
    extra_attributes: str = "",
    prefix_html: str = "",
) -> str | None:
    """Build the text or raw well markup when the descriptor uses a text well mode.

    Args:
        descriptor: File descriptor dictionary.
        extra_class: Extra CSS class suffix for the outer well element.
        extra_attributes: Extra HTML attributes for the outer well element.
        prefix_html: Optional HTML placed inside the well before the content.

    Returns:
        The well HTML string, or None when the mode is not a text or raw well.
    """
    mode = descriptor["mode"]
    if mode in _TEXT_WELL_MODES:
        return (
            f'<div class="vp-well{extra_class} vp-well-{mode}"{extra_attributes}>'
            f'{prefix_html}{descriptor["fragment"]}</div>'
        )
    if mode == "raw":
        inner_html = descriptor["fragment"] if descriptor["fragment"] is not None else escape_html(descriptor["note"])
        return f'<div class="vp-well{extra_class} vp-well-raw"{extra_attributes}>{prefix_html}{inner_html}</div>'
    return None


def render_pdf_container_html(descriptor: dict) -> str:
    """Build the inline PDF viewer mount container.

    Args:
        descriptor: File descriptor dictionary containing the PDF source URL.

    Returns:
        The HTML container element for the PDF viewer.
    """
    return f'<div class="vp-frame vp-pdf" data-pdf-src="{escape_html(descriptor["url"])}"></div>'


def render_converting_html() -> str:
    """Build the converting spinner placeholder element.

    Returns:
        The HTML placeholder element shown while a file converts.
    """
    return '<div class="vp-converting"></div>'


def render_convert_failed_html(descriptor: dict, download_href: str) -> str:
    """Build the conversion error block with retry and download controls.

    Args:
        descriptor: File descriptor dictionary containing the error and path.
        download_href: Download link for the original file.

    Returns:
        The HTML block describing the conversion failure.
    """
    return (
        '<div class="vp-convert-failed"><p>Couldn\'t convert this file</p>'
        f'<p class="vp-err">{escape_html(descriptor["error"])}</p>'
        '<div class="vp-state-actions">'
        f'<button class="btn ghost" data-vp-retry="/api/view/{escape_html(descriptor["path"])}/retry">Retry</button>'
        f'<a class="btn ghost" href="{download_href}">Download</a></div></div>'
    )


def _render_solo_native_html(descriptor: dict) -> str:
    kind = descriptor["kind"]
    if kind == "image":
        return f'<div class="img-stage vp-native"><img src="{descriptor["url"]}" alt="" data-vp-zoom></div>'
    if kind == "video":
        return f'<div class="vp-native"><video controls src="{descriptor["url"]}"></video></div>'
    return f'<div class="vp-native"><audio controls src="{descriptor["url"]}"></audio></div>'


def _render_unsupported_card_html(descriptor: dict, download_href: str) -> str:
    return (
        '<div class="vp-unsupported"><div class="vp-state-card">'
        '<span class="vp-state-ic" data-vp-icon="box"></span>'
        f'<h1 class="vp-state-title">{escape_html(descriptor["name"])}</h1>'
        '<p class="vp-state-sub">This file type can\'t be previewed — download it to view.</p>'
        '<div class="vp-state-actions">'
        f'<a class="btn ghost" href="{download_href}">Download</a>'
        '<a class="btn ghost" data-vp-back>Go back</a></div></div></div>'
    )


def _render_file_body(descriptor: dict) -> str:
    well_html = render_text_well_html(descriptor)
    if well_html is not None:
        return well_html
    mode = descriptor["mode"]
    if mode == "native":
        return _render_solo_native_html(descriptor)
    if mode == "frame":
        if descriptor["kind"] == "html":
            return (
                '<div class="vp-native"><iframe class="vp-frame" '
                f'src="{descriptor["url"]}" sandbox="allow-same-origin"></iframe></div>'
            )
        return f'<div class="vp-native">{render_pdf_container_html(descriptor)}</div>'
    if mode == "converting":
        return render_converting_html()
    download_href = build_download_href(descriptor["path"])
    if mode == "convert_failed":
        return render_convert_failed_html(descriptor, download_href)
    return _render_unsupported_card_html(descriptor, download_href)


async def view_file(request: Request) -> Response:
    """Show one resource file on a page of its own.

    The file fills the whole page. The only button is Download.

    Args:
        request: The request being answered.

    Returns:
        The page, or a message when the file is gone.
    """
    resource_path = request.path_params["path"]
    full_path = await resolve_view(resource_path)
    if full_path is None:
        return PlainTextResponse("Not Found", status_code=404)
    path_status = await asyncio.to_thread(_probe_file_path_sync, full_path)
    if path_status.is_directory:
        return PlainTextResponse("Not Found", status_code=404)
    if not path_status.is_present:
        return render_expired_page()
    descriptor = await build_descriptor(resource_path, full_path)
    download_href = build_download_href(descriptor["path"])
    top_cluster = build_solo_download_cluster(download_href)
    main_html = f'<main class="vp-main">{_render_file_body(descriptor)}</main>'
    return HTMLResponse(
        build_page(
            descriptor["name"],
            top_cluster,
            main_html,
            should_include_footer=False,
            body_class="view-page vp-solo",
        )
    )
