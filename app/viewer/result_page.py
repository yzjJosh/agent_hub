"""Standalone viewer page for a task run result."""

from __future__ import annotations

import os
import re

from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

import app.lib.db.store as store_module
from app.lib.render.classify import RenderKind
from app.lib.render.detect import TEXT_FAMILY_KINDS
from app.lib.render.result_feed import build_result_feed
from app.viewer.file_page import view_file
from app.viewer.layout import (
    build_page,
    build_solo_download_cluster,
    build_theme_button,
    build_top_bar,
    escape_html,
    format_byte_size,
    serve_fallback_file,
)

__all__ = ["view_result"]

_TEXT_KIND_VALUES = {kind.value for kind in TEXT_FAMILY_KINDS}
_RAW_CONTROL_NOTE_PATTERN = re.compile(r"(Switch to |Use )Raw or Download")


def _render_result_expired_page() -> Response:
    top_html = build_top_bar(build_theme_button())
    main_html = (
        '<main class="vp-expired"><div class="vp-expired-inner">'
        '<span class="vp-expired-ic" data-vp-icon="alert"></span>'
        '<h1 class="vp-expired-title">This result is no longer available</h1>'
        '<p class="vp-expired-sub">The run result may have been garbage-collected — run results are '
        "retained for 30 days and then removed.</p>"
        '<button class="btn ghost" data-vp-back>Go back</button>'
        "</div></main>"
    )
    return HTMLResponse(build_page("Result no longer available", top_html, main_html))


def _parse_kind_query(raw_kind: str | None) -> RenderKind | None:
    if raw_kind and raw_kind in _TEXT_KIND_VALUES:
        return RenderKind(raw_kind)
    return None


def _rewrite_raw_control_note(note: str | None) -> str | None:
    if note is None:
        return None
    return _RAW_CONTROL_NOTE_PATTERN.sub("Download the result", note)


def _render_result_body(body_descriptor: dict, task_id: str, file_count: int) -> str:
    if body_descriptor["size"] == 0:
        subtitle = "This run stored files only." if file_count else "This run stored an empty result."
        return (
            '<div class="feed-state"><span class="vp-state-ic" data-vp-icon="box"></span>'
            '<h1 class="vp-state-title">No result content</h1>'
            f'<p class="vp-state-sub">{subtitle}</p></div>'
        )
    if body_descriptor["mode"] == "legacy":
        return (
            '<div class="vp-body-legacy"><div class="feed-state">'
            '<span class="vp-state-ic" data-vp-icon="box"></span>'
            f'<h1 class="vp-state-title">result.bin · {escape_html(format_byte_size(body_descriptor["size"]))}</h1>'
            '<p class="vp-legacy-label">Legacy binary result</p>'
            '<p class="vp-state-sub">This run predates the text-body migration — the stored '
            "body is binary. Use Download.</p></div>"
            f'<a class="btn ghost" href="/api/tasks/{escape_html(task_id)}/result/download">Download</a>'
            "</div>"
        )
    if body_descriptor["mode"] in ("doc", "code", "table", "plain"):
        return f'<div class="vp-well vp-well-{body_descriptor["mode"]}">{body_descriptor["fragment"]}</div>'
    note = _rewrite_raw_control_note(body_descriptor.get("note"))
    parts = [f'<div class="res-note">{escape_html(note)}</div>'] if note else []
    if body_descriptor.get("fragment") is not None:
        parts.append(body_descriptor["fragment"])
    return f'<div class="vp-well vp-well-raw">{"".join(parts)}</div>'


async def view_result(request: Request) -> Response:
    """Render the standalone viewer page for a task run result.

    Args:
        request: Incoming HTTP request.

    Returns:
        HTML page showing the result body, or an expired notice when the result is gone.
    """
    task_id = request.path_params["task_id"]
    task_view = await store_module.get_task_view(task_id)
    if task_view is None:
        return await serve_fallback_file(request, f"result/{task_id}", _render_result_expired_page(), view_file)
    run_directory = os.path.join(store_module.task_store.data_dir, task_view.run_dir)
    requested_kind = _parse_kind_query(request.query_params.get("kind"))
    declared_kind = RenderKind(task_view.render_type) if task_view.render_type is not None else None
    try:
        feed = await build_result_feed(
            task_id,
            run_directory,
            kind=requested_kind,
            declared_kind=declared_kind,
            full=True,
        )
    except FileNotFoundError:
        return _render_result_expired_page()
    top_cluster = build_solo_download_cluster(f"/api/tasks/{escape_html(task_id)}/result/download")
    main_html = f'<main class="vp-main">{_render_result_body(feed["body"], task_id, len(feed["files"]))}</main>'
    return HTMLResponse(
        build_page(
            f"result · {task_id[:8]}",
            top_cluster,
            main_html,
            should_include_footer=False,
            body_class="view-page vp-solo",
        )
    )
