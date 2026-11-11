"""The page that shows what a task run gave back."""

from __future__ import annotations

import os
import re

import app.lib.db.store as store_module
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

from app.lib.render import result_feed as rf
from app.lib.render.classify import RenderKind
from app.lib.render.detect import TEXT_FAMILY_KINDS
from app.viewer.pages import view_file
from app.viewer.view_chrome import delegate_miss, esc, fmt_size, page, theme_btn, top_bar

__all__ = ["view_result_page"]

_TEXT_TOKENS = {k.value for k in TEXT_FAMILY_KINDS}

# The full screen has no Raw control, so a body note that points at it points
# at Download instead.
_RAW_NOTE = re.compile(r"(Switch to |Use )Raw or Download")


def _result_expired_page() -> Response:
    """Build the page shown when a run result is gone.

    Returns:
        The page.
    """
    top = top_bar(theme_btn())
    main = (
        '<main class="vp-expired"><div class="vp-expired-inner">'
        '<span class="vp-expired-ic" data-vp-icon="alert"></span>'
        '<h1 class="vp-expired-title">This result is no longer available</h1>'
        '<p class="vp-expired-sub">The run result may have been garbage-collected — run results are '
        'retained for 30 days and then removed.</p>'
        '<button class="btn ghost" data-vp-back>Go back</button>'
        '</div></main>'
    )
    return HTMLResponse(page("Result no longer available", top, main))


def _kind_from_query(raw: str | None) -> RenderKind | None:
    """Pick out the body kind from the page's query string.

    Only a valid text family token is applied. Anything else is ignored, so a
    mistyped link still shows the result instead of an error page.

    Args:
        raw: The value of the kind query parameter, or None.

    Returns:
        The kind to show the body as, or None to work it out.
    """
    if raw and raw in _TEXT_TOKENS:
        return RenderKind(raw)
    return None


def _page_note(note: str | None) -> str | None:
    """Rewrite a body note for the full screen, which has no Raw control.

    Args:
        note: The note the body carries, or None.

    Returns:
        The note with Raw pointed at Download, or None.
    """
    if note is None:
        return None
    return _RAW_NOTE.sub("Download the result", note)


def _body_block(d: dict, task_id: str, file_count: int) -> str:
    """Build the feed's body block, which has no head.

    Args:
        d: The description of the body.
        task_id: The run the body came from.
        file_count: How many files the run stored.

    Returns:
        The HTML for the body block.
    """
    if d["size"] == 0:
        sub = "This run stored files only." if file_count else "This run stored an empty result."
        return (f'<div class="feed-state"><span class="vp-state-ic" data-vp-icon="box"></span>'
                f'<h1 class="vp-state-title">No result content</h1>'
                f'<p class="vp-state-sub">{sub}</p></div>')
    if d["mode"] == "legacy":
        return (f'<div class="vp-body-legacy"><div class="feed-state">'
                f'<span class="vp-state-ic" data-vp-icon="box"></span>'
                f'<h1 class="vp-state-title">result.bin · {esc(fmt_size(d["size"]))}</h1>'
                f'<p class="vp-legacy-label">Legacy binary result</p>'
                f'<p class="vp-state-sub">This run predates the text-body migration — the stored '
                f'body is binary. Use Download.</p></div>'
                f'<a class="btn ghost" href="/api/tasks/{esc(task_id)}/result/download">Download</a>'
                f'</div>')
    if d["mode"] in ("doc", "code", "table", "plain"):
        return f'<div class="vp-well vp-well-{d["mode"]}">{d["fragment"]}</div>'
    note = _page_note(d.get("note"))
    parts = [f'<div class="res-note">{esc(note)}</div>'] if note else []
    if d.get("fragment") is not None:
        parts.append(d["fragment"])
    return f'<div class="vp-well vp-well-raw">{"".join(parts)}</div>'


async def view_result_page(request: Request) -> Response:
    """Show what a task run gave back, on a page of its own.

    The body the run saved fills the page under a single Download pill — the
    run's files stay in the console feed. A kind in the query string shows
    the body that way, and a kind the tool declared itself outranks it.

    Args:
        request: The request being answered.

    Returns:
        The page, or a message when the result is gone.
    """
    task_id = request.path_params["task_id"]
    view = await store_module.task_view(task_id)
    if view is None:
        return await delegate_miss(request, f"result/{task_id}", _result_expired_page(), view_file)
    run_dir = os.path.join(store_module.task_store.data_dir, view.run_dir)
    kind = _kind_from_query(request.query_params.get("kind"))
    declared = RenderKind(view.render_type) if view.render_type is not None else None
    try:
        feed = await rf.build_result_feed(task_id, run_dir, kind=kind,
                                          declared_kind=declared, full=True)
    except FileNotFoundError:
        return _result_expired_page()
    chrome = (
        f'<div class="vp-solo-cluster"><a class="vp-solo-dl" '
        f'href="/api/tasks/{esc(task_id)}/result/download" title="Download" aria-label="Download">'
        f'<span data-vp-icon="download"></span></a></div>'
    )
    main = f'<main class="vp-main">{_body_block(feed["body"], task_id, len(feed["files"]))}</main>'
    return HTMLResponse(page(f"result · {task_id[:8]}", chrome, main, footer=False,
                             body_cls="view-page vp-solo"))
