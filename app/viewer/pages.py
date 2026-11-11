"""The pages someone sees when they open a shared link."""

from __future__ import annotations

import asyncio
import mimetypes
import os
import re
from datetime import datetime
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, PlainTextResponse, Response

import app.lib.db.store as store_module
from app.lib.render import cache
from app.lib.render.descriptor import build_descriptor, resolve_view
from app.viewer.view_chrome import (
    STATIC_HEAD as _STATIC_HEAD,
    delegate_miss as _delegate_miss,
    esc as _esc,
    fmt_size as _fmt_size,
    page as _page,
    theme_btn as _theme_btn,
    top_bar as _top_bar,
)

__all__ = ["view_doc", "view_doc_asset", "view_file", "view_presentation"]


def _dl_href(path: str) -> str:
    """Build the download link for a file.

    Args:
        path: A path inside the shared folder.

    Returns:
        The link to download the file.
    """
    return f"/dav/dl?path={_esc(quote(path, safe='/'))}"


def _expired_page() -> Response:
    # The bar holds no copy or download button.
    top = _top_bar(_theme_btn())
    main = (
        '<main class="vp-main vp-expired">'
        '<div class="vp-expired-inner">'
        '<span class="vp-expired-ic" data-vp-icon="alert"></span>'
        '<h1 class="vp-expired-title">This link is no longer valid</h1>'
        '<p class="vp-expired-sub">The file may have been garbage-collected — files are retained '
        'for 30 days and then removed. Ask the sender for a fresh link.</p>'
        '<button class="btn ghost" data-vp-back>Go back</button>'
        "</div></main>"
    )
    return HTMLResponse(_page("Link no longer valid", top, main))


async def view_file(request: Request) -> Response:
    """Show one shared file on a page of its own.

    The file fills the whole page. The only button is Download.

    Args:
        request: The request being answered.

    Returns:
        The page, or a message when the file is gone.
    """
    path = request.path_params["path"]
    full = await resolve_view(path)
    if full is None:
        return PlainTextResponse("Not Found", status_code=404)
    if await asyncio.to_thread(os.path.isdir, full):
        return PlainTextResponse("Not Found", status_code=404)
    if not await asyncio.to_thread(os.path.exists, full):
        return _expired_page()
    descriptor = await build_descriptor(path, full)
    chrome = (
        f'<div class="vp-solo-cluster"><a class="vp-solo-dl" href="{_dl_href(descriptor["path"])}" '
        'title="Download" aria-label="Download"><span data-vp-icon="download"></span></a></div>'
    )
    main = f'<main class="vp-main">{_file_body(descriptor)}</main>'
    return HTMLResponse(_page(descriptor["name"], chrome, main, footer=False,
                              body_cls="view-page vp-solo"))


def _file_body(d: dict) -> str:
    """Build the part of the page that shows the file.

    Args:
        d: The description of the file.

    Returns:
        The HTML for that part of the page.
    """
    mode = d["mode"]
    if mode in ("doc", "code", "table", "plain"):
        return f'<div class="vp-well vp-well-{mode}">{d["fragment"]}</div>'
    if mode == "raw":
        # A file that is not text has no page, so a note is shown instead.
        inner = d["fragment"] if d["fragment"] is not None else _esc(d["note"])
        return f'<div class="vp-well vp-well-raw">{inner}</div>'
    if mode == "native":
        if d["kind"] == "image":
            return f'<div class="img-stage vp-native"><img src="{d["url"]}" alt="" data-vp-zoom></div>'
        if d["kind"] == "video":
            return f'<div class="vp-native"><video controls src="{d["url"]}"></video></div>'
        return f'<div class="vp-native"><audio controls src="{d["url"]}"></audio></div>'
    if mode == "frame":
        if d["kind"] == "html":
            return (f'<div class="vp-native"><iframe class="vp-frame" '
                    f'src="{d["url"]}" sandbox="allow-same-origin"></iframe></div>')
        # The pdf viewer mounts into this container.
        return f'<div class="vp-native"><div class="vp-frame vp-pdf" data-pdf-src="{_esc(d["url"])}"></div></div>'
    if mode == "converting":
        return '<div class="vp-converting"></div>'
    if mode == "convert_failed":
        return (f'<div class="vp-convert-failed"><p>Couldn\'t convert this file</p>'
                f'<p class="vp-err">{_esc(d["error"])}</p>'
                f'<div class="vp-state-actions">'
                f'<button class="btn ghost" data-vp-retry="/api/view/{_esc(d["path"])}/retry">Retry</button>'
                f'<a class="btn ghost" href="{_dl_href(d["path"])}">Download</a></div></div>')
    return (f'<div class="vp-unsupported"><div class="vp-state-card">'
            f'<span class="vp-state-ic" data-vp-icon="box"></span>'
            f'<h1 class="vp-state-title">{_esc(d["name"])}</h1>'
            f'<p class="vp-state-sub">This file type can\'t be previewed — download it to view.</p>'
            f'<div class="vp-state-actions">'
            f'<a class="btn ghost" href="{_dl_href(d["path"])}">Download</a>'
            f'<a class="btn ghost" data-vp-back>Go back</a></div></div></div>')


_HEX16 = re.compile(r"^[0-9a-f]{16}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# The folders the content file is stored in. New rows use presentations,
# legacy rows use handoffs.
_CONTENT_DIRS = ("presentations", "handoffs")


def _content_match(files: list[str], rid: str) -> str | None:
    """Find the file that is the presentation's own content.

    A file is the content when it sits in one of the content folders and
    its name, without extension, is the presentation's id. The first such
    file in shared order wins.

    Args:
        files: The files in the order they were shared.
        rid: The presentation's id.

    Returns:
        The path of the content file, or None when no file is the content.
    """
    for share_rel in files:
        if os.path.dirname(share_rel) in _CONTENT_DIRS and \
                os.path.splitext(os.path.basename(share_rel))[0] == rid:
            return share_rel
    return None


def _presentation_toolbar(d: dict, dl_href: str) -> str:
    """Build the slim toolbar row of the presentation body.

    Args:
        d: The description of the content file.
        dl_href: The download link for the content file.

    Returns:
        The HTML for the toolbar row.
    """
    open_href = f"/view/{_esc(quote(d['path'], safe='/'))}"
    return (
        f'<div class="well-topbar">'
        f'<span class="chip">{_esc(d["kind_label"])}</span>'
        f'<span class="well-note">{_fmt_size(d["size"])}</span>'
        '<span class="well-actions">'
        f'<a class="vp-icon-btn" href="{open_href}" target="_blank" '
        'title="Open full-screen" aria-label="Open full-screen">'
        '<span data-vp-icon="open"></span></a>'
        f'<a class="vp-icon-btn" href="{dl_href}" title="Download" '
        'aria-label="Download"><span data-vp-icon="download"></span></a>'
        "</span></div>"
    )


def _presentation_body_section(content_path: str, d: dict | None) -> str | None:
    """Build the section that shows the presentation's content.

    Args:
        content_path: The path of the content file.
        d: The description of the file, or None when the file is gone.

    Returns:
        The HTML for the section, or None when the file's type has no body.
    """
    if d is None:
        return (f'<div class="vp-well vp-presentation-body" data-path="{_esc(content_path)}">'
                '<div class="vp-body-gone">'
                '<span class="vp-state-ic" data-vp-icon="box"></span>'
                '<p class="vp-body-gone-title">The message content is no longer available</p>'
                '<p class="vp-body-gone-sub">Files are retained for 30 days, then removed.</p>'
                "</div></div>")
    mode = d["mode"]
    if mode == "unsupported":
        return None
    dl_href = _dl_href(content_path)
    attrs = f' data-path="{_esc(content_path)}" data-dl-href="{dl_href}"'
    toolbar = _presentation_toolbar(d, dl_href)
    if mode in ("doc", "code", "table", "plain"):
        well = f'<div class="vp-well vp-presentation-body vp-well-{mode}"{attrs}>'
        return f"{well}{toolbar}{d['fragment']}</div>"
    if mode == "raw":
        inner = d["fragment"] if d["fragment"] is not None else _esc(d["note"])
        return f'<div class="vp-well vp-presentation-body vp-well-raw"{attrs}>{toolbar}{inner}</div>'
    if mode == "native":
        if d["kind"] == "image":
            inner = f'<div class="img-stage"><img src="{d["url"]}" alt=""></div>'
        elif d["kind"] == "video":
            inner = f'<div class="media-stage"><video controls src="{d["url"]}"></video></div>'
        else:
            inner = f'<div class="audio-card" data-src="{d["url"]}" data-name="{_esc(d["name"])}"></div>'
        return f'<div class="vp-well vp-presentation-body"{attrs}>{toolbar}{inner}</div>'
    if mode == "frame":
        if d["kind"] == "html":
            inner = f'<iframe class="frame-body" src="{d["url"]}" sandbox="allow-same-origin"></iframe>'
        else:
            # The pdf viewer mounts into this container.
            inner = f'<div class="vp-frame vp-pdf" data-pdf-src="{_esc(d["url"])}"></div>'
        return f'<div class="vp-well vp-presentation-body"{attrs}>{toolbar}{inner}</div>'
    if mode == "converting":
        return f'<div class="vp-well vp-presentation-body"{attrs}>{toolbar}' \
            f'<div class="vp-converting"></div></div>'
    if mode == "convert_failed":
        return (f'<div class="vp-well vp-presentation-body"{attrs}>{toolbar}'
                f'<div class="vp-convert-failed"><p>Couldn\'t convert this file</p>'
                f'<p class="vp-err">{_esc(d["error"])}</p>'
                f'<div class="vp-state-actions">'
                f'<button class="btn ghost" data-vp-retry="/api/view/{_esc(d["path"])}/retry">Retry</button>'
                f'<a class="btn ghost" href="{dl_href}">Download</a></div></div></div>')
    raise ValueError(f"Unknown mode: {mode}")


def _presentation_card(share_rel: str, is_present: bool, size: int | None, label: str, thumb_status: str) -> str:
    dl = f"/dav/dl?path={_esc(quote(share_rel, safe='/'))}"
    view = f"/view/{_esc(quote(share_rel, safe='/'))}"
    name = os.path.basename(share_rel)
    if not is_present:
        return (f'<div class="thumb-card vp-card-gone"><div class="thumb-1610">'
                f'<span class="thumb-glyph" data-vp-icon="box"></span></div>'
                f'<div class="thumb-name">{_esc(name)}</div>'
                f'<div class="thumb-meta">no longer available</div></div>')
    thumb = (f'<img class="thumb-img" src="{cache.thumb_url(share_rel)}">'
             if thumb_status == "done" else '<span class="thumb-glyph" data-vp-icon="box"></span>')
    # The whole card is a link that opens the file. The download icon sits on
    # top of it and can still be clicked.
    return (
        f'<div class="thumb-card"><div class="thumb-1610">{thumb}</div>'
        f'<div class="thumb-name">{_esc(name)}</div>'
        f'<div class="thumb-meta">{_fmt_size(size)} · {_esc(label)}</div>'
        f'<a class="vp-card-open" href="{view}" aria-label="Open {_esc(name)}"></a>'
        f'<div class="thumb-actions"><a class="vp-icon-btn" href="{dl}" title="Download">'
        f'<span data-vp-icon="download"></span></a></div></div>'
    )


async def _presentation_card_data(share_rel: str) -> dict:
    full = await resolve_view(share_rel)
    if full is None or not await asyncio.to_thread(os.path.isfile, full):
        return {"is_present": False, "size": None, "label": "file", "thumb_status": "none"}
    st = await asyncio.to_thread(os.stat, full)
    descriptor = await build_descriptor(share_rel, full)
    return {
        "is_present": True,
        "size": st.st_size,
        "label": descriptor["kind_label"],
        "thumb_status": await cache.thumb_status(cache.key_for(share_rel, int(st.st_mtime))),
    }


async def _content_descriptor(files: list[tuple[str, dict]], content_path: str) -> dict | None:
    """Describe the presentation's content file.

    Args:
        files: The files with their card data, in shared order.
        content_path: The path of the content file.

    Returns:
        The description of the content file, or None when it is gone.
    """
    for share_rel, data in files:
        if share_rel == content_path and data["is_present"]:
            full = await resolve_view(share_rel)
            if full is not None:
                return await build_descriptor(share_rel, full)
            break
    return None


def _presentation_header(title: str, shared: str, m: int, alive_grid: int) -> str:
    """Build the header with the title or the generic heading, and the sub line.

    Args:
        title: The agent's title, or the empty string for rows without one.
        shared: The shared time as text.
        m: How many files sit in the grid.
        alive_grid: How many of the grid files are still here.

    Returns:
        The HTML for the header.
    """
    avail = f" · {alive_grid} of {m} files available" if m > 0 and alive_grid < m else ""
    if title:
        head = f'<h1 class="vp-presentation-title">{_esc(title)}</h1>'
        sender = " by your agent"
    elif m:
        head = f'<h1 class="vp-presentation-head">{m} files from your agent</h1>'
        sender = ""
    else:
        head = '<h1 class="vp-presentation-head">Files from your agent</h1>'
        sender = ""
    return f'{head}<div class="vp-presentation-sub">Shared {_esc(shared)}{sender}{avail}</div>'


async def view_presentation(request: Request) -> Response:
    """Show every file in one presentation on one page.

    Args:
        request: The request being answered.

    Returns:
        The page, or a message when every file is gone.
    """
    rid = request.path_params["id"]
    if not _HEX16.fullmatch(rid):
        return await _delegate_miss(request, f"presentation/{rid}", _expired_page(), view_file)
    presentation = await store_module.presentation_store.get(rid)
    if presentation is None:
        return await _delegate_miss(request, f"presentation/{rid}", _expired_page(), view_file)
    alive = 0
    files: list[tuple[str, dict]] = []
    # The cards keep the order the files were shared in.
    for share_rel in presentation.files:
        data = await _presentation_card_data(share_rel)
        if data["is_present"]:
            alive += 1
        files.append((share_rel, data))
    if alive == 0:
        return _expired_page()
    content_path = _content_match(presentation.files, rid)
    content_desc = await _content_descriptor(files, content_path) if content_path else None
    body = _presentation_body_section(content_path, content_desc) if content_path else None
    # When the body shows, the content is out of the grid and the counts.
    body_shown = body is not None
    grid = [
        _presentation_card(share_rel, **data)
        for share_rel, data in files
        if not (body_shown and share_rel == content_path)
    ]
    m = len(presentation.files) - 1 if body_shown else len(presentation.files)
    alive_grid = alive - 1 if body_shown and content_desc is not None else alive
    # The saved time may not be a real date. It is then shown as it is.
    created_at = presentation.presented_at
    try:
        shared = datetime.fromisoformat(created_at).strftime("%b %d %H:%M")
    except (KeyError, TypeError, ValueError):
        shared = created_at
    title = (presentation.title or "").strip()
    header = _presentation_header(title, shared, m, alive_grid)
    if body_shown and m > 0:
        files_section = (f'<div class="vp-feed-files">FILES · {m}</div>'
                         f'<div class="thumb-grid after-divider">{"".join(grid)}</div>')
    elif body_shown:
        files_section = ""
    else:
        files_section = f'<div class="thumb-grid">{"".join(grid)}</div>'
    # The bar holds no copy or download button.
    top = _top_bar(_theme_btn())
    main = (f'<main class="vp-main vp-presentation"><div class="vp-doccol">'
            f'{header}{body or ""}{files_section}</div></main>')
    return HTMLResponse(_page(title or "Files from your agent", top, main))


def _doc_compose(fragment: str, thumb: bool) -> str:
    override = '<style id="vp-thumb-override">.vp-doccol{max-width:none;margin:0;padding:16px;}</style>' if thumb else ""
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        '<title>Document</title>'
        '<script>document.documentElement.setAttribute("data-theme","light");</script>'
        f"{_STATIC_HEAD}{override}</head>"
        f'<body class="view-page doc-page"><main class="vp-main"><div class="vp-doccol">'
        f'{fragment}</div></main></body></html>'
    )


def _plain_404() -> Response:
    return PlainTextResponse("Not Found", status_code=404)


async def view_doc(request: Request) -> Response:
    """Show a saved document page.

    Args:
        request: The request being answered.

    Returns:
        The page, or a message when there is no such page.
    """
    key = request.path_params["key"]
    entry = await cache.read_doc(key) if _HEX64.fullmatch(key) else None
    if entry is None:
        return await _delegate_miss(request, f"doc/{key}", _plain_404(), view_file)
    doc_html, _meta = entry
    thumb = request.query_params.get("thumb") == "1"
    return HTMLResponse(_doc_compose(doc_html, thumb))


def _resolve_doc_asset(key: str, asset: str) -> tuple[str, str] | None:
    """Find a file that belongs to a saved document page.

    Args:
        key: The name the document is saved under.
        asset: The name of the file to find.

    Returns:
        Where the file is and what type it is, or None when there is no such
        file.
    """
    key_dir = os.path.realpath(os.path.join(cache.render_root(), key))
    if not os.path.isdir(key_dir):
        return None
    target = os.path.realpath(os.path.join(key_dir, asset))
    if target == key_dir or not target.startswith(key_dir + os.sep):
        return None
    if not os.path.isfile(target):
        return None
    ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
    return target, ctype


async def view_doc_asset(request: Request) -> Response:
    """Send a file that belongs to a saved document page.

    Args:
        request: The request being answered.

    Returns:
        The file, or a message when there is no such file.
    """
    key = request.path_params["key"]
    asset = request.path_params["asset"]
    is_key = _HEX64.fullmatch(key) is not None
    resolved = await asyncio.to_thread(_resolve_doc_asset, key, asset) if is_key else None
    if resolved is None:
        return await _delegate_miss(request, f"doc/{key}/{asset}", _plain_404(), view_file)
    target, ctype = resolved
    return FileResponse(target, media_type=ctype)
