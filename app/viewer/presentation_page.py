"""Presentation viewer page showing a presented body and file grid."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

import app.lib.db.store as store_module
from app.lib.render import cache
from app.lib.render.descriptor import build_descriptor, resolve_view
from app.viewer.file_page import (
    build_download_href,
    render_convert_failed_html,
    render_converting_html,
    render_expired_page,
    render_pdf_container_html,
    render_text_well_html,
    view_file,
)
from app.viewer.layout import (
    build_page,
    build_theme_button,
    build_top_bar,
    escape_html,
    format_byte_size,
    serve_fallback_file,
)

__all__ = ["view_presentation"]

_PRESENTATION_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
_CONTENT_DIRECTORIES = ("presentations", "handoffs")


@dataclass(frozen=True)
class _PresentationFile:
    """Availability and display metadata for one presented file."""

    resource_path: str
    is_present: bool
    size: int | None
    label: str
    thumbnail_status: str
    descriptor: dict | None


def _find_content_path(file_paths: list[str], presentation_id: str) -> str | None:
    for resource_path in file_paths:
        is_in_content_directory = os.path.dirname(resource_path) in _CONTENT_DIRECTORIES
        file_stem = os.path.splitext(os.path.basename(resource_path))[0]
        if is_in_content_directory and file_stem == presentation_id:
            return resource_path
    return None


def _stat_regular_file_sync(full_path: str) -> os.stat_result | None:
    if not os.path.isfile(full_path):
        return None
    return os.stat(full_path)


async def _load_presentation_file(resource_path: str) -> _PresentationFile:
    full_path = await resolve_view(resource_path)
    file_stat = await asyncio.to_thread(_stat_regular_file_sync, full_path) if full_path is not None else None
    if file_stat is None or full_path is None:
        return _PresentationFile(
            resource_path=resource_path,
            is_present=False,
            size=None,
            label="file",
            thumbnail_status="none",
            descriptor=None,
        )
    descriptor = await build_descriptor(resource_path, full_path)
    thumbnail_key = cache.build_cache_key(resource_path, int(file_stat.st_mtime))
    return _PresentationFile(
        resource_path=resource_path,
        is_present=True,
        size=file_stat.st_size,
        label=descriptor["kind_label"],
        thumbnail_status=await cache.get_thumbnail_status(thumbnail_key),
        descriptor=descriptor,
    )


def _find_content_descriptor(files: list[_PresentationFile], content_path: str) -> dict | None:
    for file_item in files:
        if file_item.resource_path == content_path and file_item.is_present:
            return file_item.descriptor
    return None


def _render_presentation_toolbar(descriptor: dict, download_href: str) -> str:
    open_href = f"/view/{escape_html(quote(descriptor['path'], safe='/'))}"
    return (
        '<div class="well-topbar">'
        f'<span class="chip">{escape_html(descriptor["kind_label"])}</span>'
        f'<span class="well-note">{format_byte_size(descriptor["size"])}</span>'
        '<span class="well-actions">'
        f'<a class="vp-icon-btn" href="{open_href}" target="_blank" '
        'title="Open full-screen" aria-label="Open full-screen">'
        '<span data-vp-icon="open"></span></a>'
        f'<a class="vp-icon-btn" href="{download_href}" title="Download" '
        'aria-label="Download"><span data-vp-icon="download"></span></a>'
        "</span></div>"
    )


def _render_missing_body_section(content_path: str) -> str:
    return (
        f'<div class="vp-well vp-presentation-body" data-path="{escape_html(content_path)}">'
        '<div class="vp-body-gone">'
        '<span class="vp-state-ic" data-vp-icon="box"></span>'
        '<p class="vp-body-gone-title">The message content is no longer available</p>'
        '<p class="vp-body-gone-sub">Files are retained for 30 days, then removed.</p>'
        "</div></div>"
    )


def _render_presentation_native_inner(descriptor: dict) -> str:
    kind = descriptor["kind"]
    if kind == "image":
        return f'<div class="img-stage"><img src="{descriptor["url"]}" alt=""></div>'
    if kind == "video":
        return f'<div class="media-stage"><video controls src="{descriptor["url"]}"></video></div>'
    return f'<div class="audio-card" data-src="{descriptor["url"]}" data-name="{escape_html(descriptor["name"])}"></div>'


def _render_presentation_body_section(content_path: str, descriptor: dict | None) -> str | None:
    if descriptor is None:
        return _render_missing_body_section(content_path)
    mode = descriptor["mode"]
    if mode == "unsupported":
        return None
    download_href = build_download_href(content_path)
    extra_attributes = f' data-path="{escape_html(content_path)}" data-dl-href="{download_href}"'
    toolbar = _render_presentation_toolbar(descriptor, download_href)
    well_html = render_text_well_html(
        descriptor,
        extra_class=" vp-presentation-body",
        extra_attributes=extra_attributes,
        prefix_html=toolbar,
    )
    if well_html is not None:
        return well_html
    if mode == "native":
        inner_html = _render_presentation_native_inner(descriptor)
    elif mode == "frame":
        if descriptor["kind"] == "html":
            inner_html = f'<iframe class="frame-body" src="{descriptor["url"]}" sandbox="allow-same-origin"></iframe>'
        else:
            inner_html = render_pdf_container_html(descriptor)
    elif mode == "converting":
        inner_html = render_converting_html()
    elif mode == "convert_failed":
        inner_html = render_convert_failed_html(descriptor, download_href)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return f'<div class="vp-well vp-presentation-body"{extra_attributes}>{toolbar}{inner_html}</div>'


def _render_presentation_card(file_item: _PresentationFile) -> str:
    file_name = os.path.basename(file_item.resource_path)
    if not file_item.is_present or file_item.size is None:
        return (
            '<div class="thumb-card vp-card-gone"><div class="thumb-1610">'
            '<span class="thumb-glyph" data-vp-icon="box"></span></div>'
            f'<div class="thumb-name">{escape_html(file_name)}</div>'
            '<div class="thumb-meta">no longer available</div></div>'
        )
    download_href = build_download_href(file_item.resource_path)
    view_href = f"/view/{escape_html(quote(file_item.resource_path, safe='/'))}"
    thumbnail_html = (
        f'<img class="thumb-img" src="{cache.build_thumbnail_url(file_item.resource_path)}">'
        if file_item.thumbnail_status == "done"
        else '<span class="thumb-glyph" data-vp-icon="box"></span>'
    )
    return (
        f'<div class="thumb-card"><div class="thumb-1610">{thumbnail_html}</div>'
        f'<div class="thumb-name">{escape_html(file_name)}</div>'
        f'<div class="thumb-meta">{format_byte_size(file_item.size)} · {escape_html(file_item.label)}</div>'
        f'<a class="vp-card-open" href="{view_href}" aria-label="Open {escape_html(file_name)}"></a>'
        f'<div class="thumb-actions"><a class="vp-icon-btn" href="{download_href}" title="Download">'
        '<span data-vp-icon="download"></span></a></div></div>'
    )


def _format_presented_time(presented_at: str) -> str:
    try:
        return datetime.fromisoformat(presented_at).strftime("%b %d %H:%M")
    except (KeyError, TypeError, ValueError):
        return presented_at


def _render_presentation_header(
    title: str,
    presented_time: str,
    grid_count: int,
    present_grid_count: int,
) -> str:
    availability_note = (
        f" · {present_grid_count} of {grid_count} files available"
        if grid_count > 0 and present_grid_count < grid_count
        else ""
    )
    if title:
        heading = f'<h1 class="vp-presentation-title">{escape_html(title)}</h1>'
        sender = " by your agent"
    elif grid_count:
        heading = f'<h1 class="vp-presentation-head">{grid_count} files from your agent</h1>'
        sender = ""
    else:
        heading = '<h1 class="vp-presentation-head">Files from your agent</h1>'
        sender = ""
    return f'{heading}<div class="vp-presentation-sub">Shared {escape_html(presented_time)}{sender}{availability_note}</div>'


def _render_files_section(grid_cards: list[str], grid_count: int, *, has_body: bool) -> str:
    if has_body and grid_count > 0:
        return (
            f'<div class="vp-feed-files">FILES · {grid_count}</div>'
            f'<div class="thumb-grid after-divider">{"".join(grid_cards)}</div>'
        )
    if has_body:
        return ""
    return f'<div class="thumb-grid">{"".join(grid_cards)}</div>'


async def view_presentation(request: Request) -> Response:
    """Show every file in one presentation on one page.

    Args:
        request: The request being answered.

    Returns:
        The page, or a message when every file is gone.
    """
    presentation_id = request.path_params["id"]
    if not _PRESENTATION_ID_PATTERN.fullmatch(presentation_id):
        return await serve_fallback_file(request, f"presentation/{presentation_id}", render_expired_page(), view_file)
    presentation = await store_module.presentation_store.get(presentation_id)
    if presentation is None:
        return await serve_fallback_file(request, f"presentation/{presentation_id}", render_expired_page(), view_file)
    files = [await _load_presentation_file(resource_path) for resource_path in presentation.files]
    present_count = sum(1 for file_item in files if file_item.is_present)
    if present_count == 0:
        return render_expired_page()
    content_path = _find_content_path(presentation.files, presentation_id)
    content_descriptor = _find_content_descriptor(files, content_path) if content_path else None
    body_html = _render_presentation_body_section(content_path, content_descriptor) if content_path else None
    has_body = body_html is not None
    grid_cards = [
        _render_presentation_card(file_item)
        for file_item in files
        if not (has_body and file_item.resource_path == content_path)
    ]
    grid_count = len(presentation.files) - 1 if has_body else len(presentation.files)
    present_grid_count = present_count - 1 if has_body and content_descriptor is not None else present_count
    presented_time = _format_presented_time(presentation.presented_at)
    title = (presentation.title or "").strip()
    header_html = _render_presentation_header(title, presented_time, grid_count, present_grid_count)
    files_html = _render_files_section(grid_cards, grid_count, has_body=has_body)
    top_html = build_top_bar(build_theme_button())
    main_html = (
        '<main class="vp-main vp-presentation"><div class="vp-doccol">'
        f'{header_html}{body_html or ""}{files_html}</div></main>'
    )
    return HTMLResponse(build_page(title or "Files from your agent", top_html, main_html))
