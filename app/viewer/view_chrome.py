"""The bits every viewer page is built from."""

from __future__ import annotations

import asyncio
import html as _html
import os
from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from app.lib.render.descriptor import resolve_view

__all__ = ["STATIC_HEAD", "delegate_miss", "esc", "fmt_size", "page", "theme_btn", "top_bar"]

STATIC_HEAD = (
    '<link rel="stylesheet" href="/assets/css/tokens.css">'
    '<link rel="stylesheet" href="/assets/css/view.css">'
)

# Picks the theme before the page is drawn. A saved pick wins while the
# reader's own setting is the one it was made under, otherwise that setting.
_PRE_STATE_SCRIPT = (
    '<script>(function(){var r=document.documentElement,t="light",s=null;'
    'try{t=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";}catch(e){t="light";}'
    'try{s=JSON.parse(localStorage.getItem("agent-console-theme"));}catch(e){s=null;}'
    'if(s&&(s.theme==="dark"||s.theme==="light")&&s.scheme===t){t=s.theme;}'
    'r.setAttribute("data-theme",t);})();</script>'
)

_BRAND = (
    '<span class="vp-brand-mark">'
    '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 5 4 3-4 3"/>'
    '<path d="M9 12h4"/></svg></span>'
    '<span class="vp-brand-name">Agent Hub</span>'
    '<span class="vp-brand-sub">· file viewer</span>'
)

_FOOTER_NOTE = "Served by Agent Hub · Files are retained for 30 days and may be removed without notice."
_FOOTER_NOTE_MOBILE = "Served by Agent Hub · 30-day retention"


def esc(s: str) -> str:
    """Make a value safe to drop into a page.

    Args:
        s: The value to make safe.

    Returns:
        The safe text.
    """
    return _html.escape(str(s), quote=True)


def top_bar(*actions: str) -> str:
    """Build the bar at the top of a page.

    Args:
        *actions: The buttons to put in the bar.

    Returns:
        The HTML for the bar.
    """
    return f'<header class="vp-topbar"><a class="vp-brand" href="/tools" title="Home">{_BRAND}</a>' \
           f'<div class="vp-actions">{"".join(actions)}</div></header>'


def _footer() -> str:
    return f'<footer class="vp-footer"><span class="vp-footer-note">{_FOOTER_NOTE}</span>' \
           f'<span class="vp-footer-note-mobile">{_FOOTER_NOTE_MOBILE}</span></footer>'


def theme_btn() -> str:
    """Build the button that switches the theme.

    Returns:
        The HTML for the button.
    """
    return '<button class="vp-icon-btn" data-vp-theme title="Switch theme"></button>'


def page(title: str, top: str, main: str, footer: bool = True, body_cls: str = "view-page") -> str:
    """Build a whole viewer page.

    Args:
        title: The name shown in the browser tab.
        top: The bar at the top of the page.
        main: The body of the page.
        footer: True to put a footer at the bottom.
        body_cls: The style name for the page body.

    Returns:
        The HTML for the page.
    """
    foot = _footer() if footer else ""
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        f"<title>{esc(title)}</title>{_PRE_STATE_SCRIPT}{STATIC_HEAD}</head>"
        f'<body class="{body_cls}">{top}{main}{foot}'
        '<script type="module" src="/assets/js/view-page.js"></script></body></html>'
    )


def fmt_size(n: int) -> str:
    """Turn a byte count into a short size people can read.

    Args:
        n: How many bytes.

    Returns:
        The size as text.
    """
    if n < 1024:
        return f"{n} B"
    if n < 1048576:
        return f"{round(n / 1024)} KB"
    return f"{n / 1048576:.1f} MB"


async def delegate_miss(
    request: Request, literal_path: str, missing_response: Response,
    open_file: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Answer a request whose link did not match anything.

    A path that names a real file opens that file. A path that names a folder
    is not found. Anything else gets the caller's own answer.

    Args:
        request: The request being answered.
        literal_path: The path the link asked for.
        missing_response: The answer to give when there is no such path.
        open_file: What opens a file page.

    Returns:
        The answer to send back.
    """
    full = await resolve_view(literal_path)
    if full is None:
        return missing_response
    if not await asyncio.to_thread(os.path.exists, full):
        return missing_response
    if await asyncio.to_thread(os.path.isdir, full):
        return PlainTextResponse("Not Found", status_code=404)
    view_path = "/view/" + literal_path
    scope = dict(request.scope, path=view_path, raw_path=view_path.encode("utf-8"),
                 path_params={"path": literal_path})
    return await open_file(Request(scope))
