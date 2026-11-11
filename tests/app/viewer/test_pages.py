"""Tests for the view pages."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

import app.lib.db.store as store_module
import app.lib.render.cache as cache
import app.lib.render.office as office
from app.server import build_app
from tests.app.conftest import point_share_env


@pytest.fixture
async def view_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[httpx.AsyncClient, Path]]:
    """A client for the app, plus the share dir it reads from."""
    share, data = point_share_env(tmp_path, monkeypatch)
    monkeypatch.setattr(store_module, "_DB_PATH", str(data / "records.db"))
    await store_module.task_store.open()
    app = build_app()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c, share
    await store_module.task_store.close()


def _write(share: Path, rel: str, content: str | bytes = b"") -> str:
    p = share / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        content = content.encode()
    p.write_bytes(content)
    return rel


async def _insert_presentation(rid: str, files: list[str], title: str | None = None,
                               presented_at: str = "2026-09-01T00:00:00+00:00") -> None:
    await store_module.presentation_store.insert(rid, presented_at, f"/view/presentation/{rid}",
                                                 files, title=title)


async def test_view_file_page_solo_full_screen(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The single file page has no top bar and no footer, only a Download pill."""
    client, share = view_client
    _write(share, "upload/notes.md", "# Hi\n\nbody")
    r = await client.get("/view/upload/notes.md")
    assert r.status_code == 200
    html = r.text
    assert "vp-solo" in html
    assert "vp-topbar" not in html
    assert "vp-footer" not in html
    assert "Agent Hub" not in html  # no brand name
    assert "data-vp-theme" not in html and "data-vp-copy" not in html
    assert "vp-filename" not in html and "vp-meta" not in html
    # only the Download pill is shown
    assert "vp-solo-label" not in html
    assert 'href="/dav/dl?path=upload/notes.md"' in html
    assert 'target="_blank"' not in html  # the download opens in the same tab
    assert "path=upload/notes.md" in html  # the share path in the Download link
    assert 'href="/assets/css/view.css"' in html
    assert '<script type="module" src="/assets/js/view-page.js">' in html
    # the page falls back to the system theme
    assert "agent-console-theme" in html
    assert 'matchMedia("(prefers-color-scheme: dark)")' in html


async def test_view_page_pre_state_theme_script(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The pre-paint script follows the system setting, and a saved pick made
    under that setting wins over it."""
    client, share = view_client
    _write(share, "upload/notes.md", "# Hi")
    r = await client.get("/view/upload/notes.md")
    assert r.status_code == 200
    html = r.text
    assert 'try{t=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";}catch(e){t="light";}' in html
    assert 'try{s=JSON.parse(localStorage.getItem("agent-console-theme"));}catch(e){s=null;}' in html
    assert 'if(s&&(s.theme==="dark"||s.theme==="light")&&s.scheme===t){t=s.theme;}' in html


async def test_view_solo_pdf_renders_the_inline_canvas_container(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A pdf file page carries the canvas container, no iframe, and the client script."""
    client, share = view_client
    _write(share, "upload/p.pdf", b"%PDF-1.4 fake")
    r = await client.get("/view/upload/p.pdf")
    assert r.status_code == 200
    html = r.text
    assert '<div class="vp-frame vp-pdf" data-pdf-src="/dav/upload/p.pdf?ephemeral=1"></div>' in html
    assert "<iframe" not in html
    assert 'src="/assets/js/view-page.js"' in html  # the script that imports the pdf viewer
    assert 'href="/dav/dl?path=upload/p.pdf"' in html  # the Download pill stays


async def test_view_presentation_uses_cached_pdf(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A PowerPoint preview serves its converted slides to the PDF viewer."""
    client, share = view_client
    rel = _write(share, "upload/deck.pptx", b"PK\x03\x04 fake pptx")
    key = cache.key_for(rel, int((share / rel).stat().st_mtime))
    pdf = b"%PDF-1.7\nslide"
    await cache.write_doc(key, "", {cache.OFFICE_PDF_ASSET: pdf}, {"mode": "frame"})

    page = await client.get(f"/view/{rel}")
    assert page.status_code == 200
    assert (f'<div class="vp-frame vp-pdf" data-pdf-src="/view/doc/{key}/'
            f'{cache.OFFICE_PDF_ASSET}"></div>') in page.text
    assert f'href="/dav/dl?path={rel}"' in page.text
    asset = await client.get(f"/view/doc/{key}/{cache.OFFICE_PDF_ASSET}")
    assert asset.status_code == 200
    assert asset.headers["content-type"] == "application/pdf"
    assert asset.content == pdf


async def test_view_solo_html_keeps_the_sandboxed_iframe(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """An html file page is still a sandboxed iframe, without the pdf container."""
    client, share = view_client
    _write(share, "upload/p.html", "<p>page</p>")
    r = await client.get("/view/upload/p.html")
    assert r.status_code == 200
    html = r.text
    assert '<iframe class="vp-frame" src="/dav/upload/p.html?ephemeral=1" sandbox="allow-same-origin"></iframe>' in html
    assert "vp-pdf" not in html


@pytest.mark.parametrize(
    "path",
    ["/assets/js/pdf-viewer.js", "/assets/js/pdfjs/pdf.min.mjs",
     "/assets/js/pdfjs/pdf.worker.min.mjs"],
)
async def test_pdf_viewer_and_vendor_assets_served(
        view_client: tuple[httpx.AsyncClient, Path], path: str) -> None:
    """The pdf viewer module and the vendored pdf.js files serve as javascript."""
    client, _ = view_client
    r = await client.get(path)
    assert r.status_code == 200
    assert "text/javascript" in r.headers["content-type"]


async def test_pdf_viewer_asset_content(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The served pdf viewer builds the canvas viewer, and its css and script carry their parts."""
    client, _ = view_client
    js = (await client.get("/assets/js/pdf-viewer.js")).text
    assert "export async function mountPdfViewer(container, davUrl)" in js
    assert "workerSrc" in js
    assert "IntersectionObserver" in js
    assert "./pdfjs/pdf.min.mjs" in js
    assert "/assets/js/pdfjs/pdf.worker.min.mjs" in js
    assert "data-pdf-src" in js  # the auto-init target
    assert "new pdfjsLib.AnnotationLayer(" in js  # the per-page annotation layer
    assert "getAnnotations()" in js
    assert "goToDestination" in js  # internal link navigation
    assert "addLinkAttributes" in js  # external link navigation
    css = (await client.get("/assets/css/view.css")).text
    assert ".vp-frame.vp-pdf{display:flex;flex-direction:column;overflow:hidden}" in css
    assert ".pv-toolbar{" in css and ".pv-pages{" in css and ".pv-error{" in css
    assert ".pv-ann section{" in css  # the annotation layer overlays the canvas
    view_page = (await client.get("/assets/js/view-page.js")).text
    assert 'from "./pdf-viewer.js"' in view_page


async def test_view_file_unsupported_state_card(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """An unsupported file shows a card with its name, a Download, and a Go back."""
    client, share = view_client
    _write(share, "upload/blob.bin", b"\x00\x01binary")
    r = await client.get("/view/upload/blob.bin")
    assert r.status_code == 200
    html = r.text
    assert "vp-unsupported" in html and "vp-state-card" in html
    assert "blob.bin" in html
    assert 'href="/dav/dl?path=upload/blob.bin"' in html
    assert "data-vp-back" in html  # the Go back control


async def test_presentation_page_keeps_chrome(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The presentation page has a top bar with the brand and the theme, plus a footer."""
    client, share = view_client
    _write(share, "upload/a.png", b"\x89PNG")
    await _insert_presentation("a" * 16, ["upload/a.png"])
    r = await client.get(f"/view/presentation/{'a'*16}")
    assert r.status_code == 200
    assert "vp-topbar" in r.text and "vp-footer" in r.text
    assert "Agent Hub" in r.text
    assert "data-vp-theme" in r.text and "data-vp-copy" not in r.text
    assert "vp-solo" not in r.text


async def test_view_missing_file_is_expired_page(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    client, _ = view_client
    r = await client.get("/view/never/existed.md")
    assert r.status_code == 200
    assert "This link is no longer valid" in r.text
    assert "no longer valid" in r.text
    # the expired page has no Download link
    assert "href=\"/dav/" not in r.text


async def test_view_existing_folder_is_plain_404(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    client, share = view_client
    (share / "somefolder").mkdir()
    r = await client.get("/view/somefolder")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/plain")


def _key_for(share: Path, rel: str) -> str:
    st = (share / rel).stat()
    return cache.key_for(rel, int(st.st_mtime))


async def test_presentation_page_lists_files(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    client, share = view_client
    _write(share, "upload/a.png", b"\x89PNG")
    _write(share, "upload/b.md", b"# hi")
    await _insert_presentation("b" * 16, ["upload/a.png", "upload/b.md"])
    r = await client.get(f"/view/presentation/{'b'*16}")
    assert r.status_code == 200
    assert "2 files from your agent" in r.text
    assert 'href="/view/upload/a.png"' in r.text  # the open link on each card
    assert 'href="/dav/dl?path=upload/a.png"' in r.text  # the Download link on each card
    assert 'target="_blank"' not in r.text  # the download opens in the same tab
    assert "markdown" in r.text  # the file kind on the card


def _card_html(page: str, index: int) -> str:
    """Give back the markup of one card."""
    seg = page.split('<div class="thumb-card')[1:][index]
    return '<div class="thumb-card' + seg.split("</div></div>")[0] + "</div></div>"


async def test_presentation_card_is_whole_card_link(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A live card is one link over the whole card, with Download beside it."""
    client, share = view_client
    _write(share, "upload/a.png", b"\x89PNG")
    rid = "a2" * 8
    await _insert_presentation(rid, ["upload/a.png"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert '<a class="vp-card-open" href="/view/upload/a.png" aria-label="Open a.png"></a>' in html
    assert ">Open</a>" not in html
    assert 'class="btn ghost" href="/view/' not in html
    # the Download link comes after the card link
    assert 'href="/dav/dl?path=upload/a.png"' in html
    assert html.index('class="vp-card-open"') < html.index('href="/dav/dl?path=upload/a.png"')


async def test_presentation_gone_card_is_non_interactive(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A card for a missing file has no links, and a live card beside it still does."""
    client, share = view_client
    _write(share, "upload/alive.png", b"\x89PNG")
    rid = "b2" * 8
    await _insert_presentation(rid, ["upload/alive.png", "upload/gone.png"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    live = _card_html(r.text, 0)
    gone = _card_html(r.text, 1)
    assert 'href="/view/upload/alive.png"' in live
    assert "vp-card-gone" in gone
    assert "<a" not in gone
    assert "thumb-actions" not in gone


async def test_presentation_card_open_is_native_anchor(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The card open control is a plain link, not a button."""
    client, share = view_client
    _write(share, "upload/a.png", b"\x89PNG")
    rid = "c2" * 8
    await _insert_presentation(rid, ["upload/a.png"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert '<a class="vp-card-open" href="/view/upload/a.png"' in html
    assert 'role="button"' not in html
    assert "tabindex" not in html


async def test_presentation_focus_ring_is_on_card(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The stylesheet draws the focus ring on the card and hides the link outline."""
    client, _ = view_client
    r = await client.get("/assets/css/view.css")
    assert r.status_code == 200
    css = r.text
    assert ".thumb-grid .thumb-card:focus-within{outline:2px solid var(--accent);outline-offset:1px}" in css
    assert ".vp-card-open:focus-visible{outline:none}" in css


def _thumb_actions_rule(css: str) -> str:
    """Give back the card actions rule from the stylesheet."""
    prefix = ".thumb-grid .thumb-actions{"
    for line in css.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError("served view.css is missing the .thumb-grid .thumb-actions rule")


async def test_presentation_download_actions_reset_inherited_offsets(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The card actions rule resets left and bottom."""
    client, _ = view_client
    r = await client.get("/assets/css/view.css")
    assert r.status_code == 200
    rule = _thumb_actions_rule(r.text)
    assert "left:auto" in rule
    assert "bottom:auto" in rule


async def test_presentation_thumb_src_url_encoded(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A thumbnail source for a path with special characters is escaped."""
    client, share = view_client
    rel = "upload/photo & more.png"
    _write(share, rel, b"\x89PNG")
    key_dir = os.path.join(cache.render_root(), _key_for(share, rel))
    os.makedirs(key_dir, exist_ok=True)
    with open(os.path.join(key_dir, "thumb.png"), "wb") as fh:
        fh.write(b"\x89PNG")
    await _insert_presentation("d" * 16, [rel])
    r = await client.get(f"/view/presentation/{'d'*16}")
    assert r.status_code == 200
    assert f'src="{cache.thumb_url(rel)}"' in r.text
    assert "path=upload/photo%20%26%20more.png" in r.text


async def test_presentation_missing_id_is_expired(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    client, _ = view_client
    r = await client.get("/view/presentation/" + "f" * 16)
    assert r.status_code == 200
    assert "This link is no longer valid" in r.text


async def test_presentation_non_hex_id_delegate_to_file(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    client, share = view_client
    _write(share, "presentation/plain", "# delegated")
    r = await client.get("/view/presentation/plain")
    assert r.status_code == 200
    assert "delegated" in r.text
    assert "presentation/plain" in r.text


async def test_presentation_all_gone_is_expired(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    client, _ = view_client
    await _insert_presentation("c" * 16, ["upload/gone.png"])  # file never created
    r = await client.get(f"/view/presentation/{'c'*16}")
    assert r.status_code == 200
    assert "This link is no longer valid" in r.text


async def test_presentation_open_href_url_encoded(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A card open link for a path with special characters is escaped."""
    client, share = view_client
    rel = "upload/a & b.png"
    _write(share, rel, b"\x89PNG")
    await _insert_presentation("e" * 16, [rel])
    r = await client.get(f"/view/presentation/{'e'*16}")
    assert r.status_code == 200
    assert 'href="/view/upload/a%20%26%20b.png"' in r.text
    assert 'href="/dav/dl?path=upload/a%20%26%20b.png"' in r.text


async def test_view_media_src_url_encoded(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A media page source for a path with special characters is escaped."""
    client, share = view_client
    _write(share, "upload/pic #1.png", b"\x89PNG")
    r = await client.get("/view/upload/pic%20%231.png")
    assert r.status_code == 200
    assert 'src="/dav/upload/pic%20%231.png?ephemeral=1"' in r.text
    assert 'src="/dav/upload/pic #1.png"' not in r.text


async def test_doc_page_composed_and_thumb_override(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    client, share = view_client
    _write(share, "upload/n.md", "# composed")
    key = _key_for(share, "upload/n.md")
    await cache.write_doc(key, '<div class="md-doc"><h1>composed</h1></div>', None,
                          {"mode": "doc", "provenance": None})
    r = await client.get(f"/view/doc/{key}")
    assert r.status_code == 200
    assert "composed" in r.text
    assert "thumb" not in r.text
    r2 = await client.get(f"/view/doc/{key}?thumb=1")
    assert r2.status_code == 200
    assert "vp-thumb-override" in r2.text  # the thumbnail layout


async def test_doc_missing_key_is_plain_404(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    client, _ = view_client
    r = await client.get("/view/doc/" + "0" * 64)
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/plain")


async def test_doc_non_hex_key_delegate_to_file(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    client, share = view_client
    _write(share, "doc/reports/q1", "# delegated doc")
    r = await client.get("/view/doc/reports/q1")
    assert r.status_code == 200
    assert "delegated doc" in r.text
    assert "doc/reports/q1" in r.text


async def test_doc_asset_serves_sibling(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    client, _ = view_client
    key = "0" * 64
    key_dir = os.path.join(cache.render_root(), key)
    os.makedirs(key_dir, exist_ok=True)
    with open(os.path.join(key_dir, "img.png"), "wb") as fh:
        fh.write(b"\x89PNG asset")
    r = await client.get(f"/view/doc/{key}/img.png")
    assert r.status_code == 200
    assert r.content == b"\x89PNG asset"
    assert "image/png" in r.headers["content-type"]


async def test_doc_asset_traversal_rejected(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    client, _ = view_client
    key = "0" * 64
    r = await client.get(f"/view/doc/{key}/..%2f..%2fsecret")
    assert r.status_code == 404


async def test_presentation_body_titled_markdown(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A titled row renders the title, the body well, and the files divider."""
    client, share = view_client
    rid = "e5d3049aa28546de"
    _write(share, f"presentations/{rid}.md", "# Hi\n\nbody text")
    _write(share, "upload/attach.png", b"\x89PNG")
    await _insert_presentation(rid, [f"presentations/{rid}.md", "upload/attach.png"],
                               title="Weekend swim lessons")
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    # header
    assert '<h1 class="vp-presentation-title">Weekend swim lessons</h1>' in html
    assert "Shared Sep 01 00:00 by your agent" in html
    assert "<title>Weekend swim lessons</title>" in html
    assert "Opened from a link" not in html  # the sub-line keeps no retention sentence
    # body section
    assert f'<div class="vp-well vp-presentation-body vp-well-doc" ' \
           f'data-path="presentations/{rid}.md" ' \
           f'data-dl-href="/dav/dl?path=presentations/{rid}.md">' in html
    assert '<span class="chip">markdown</span>' in html
    assert f'<a class="vp-icon-btn" href="/view/presentations/{rid}.md" target="_blank"' in html
    # the content file is not a card
    assert f'<div class="thumb-name">{rid}.md</div>' not in html
    # files section
    assert '<div class="vp-feed-files">FILES · 1</div>' in html
    assert '<div class="thumb-grid after-divider">' in html
    assert '<div class="thumb-name">attach.png</div>' in html
    # the content fragment renders in the body
    assert '<div class="md-doc">' in html and "body text" in html


async def test_presentation_body_plain_text(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A plain-text content file renders the body well like the file viewer does."""
    client, share = view_client
    rid = "9f8e7d6c5b4a0312"
    _write(share, f"presentations/{rid}.txt", "just a note\nsecond line")
    await _insert_presentation(rid, [f"presentations/{rid}.txt"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert f'<div class="vp-well vp-presentation-body vp-well-plain" ' \
           f'data-path="presentations/{rid}.txt" ' \
           f'data-dl-href="/dav/dl?path=presentations/{rid}.txt">' in html
    assert 'class="plain-pre">' in html
    assert "just a note" in html


async def test_presentation_body_null_title_content_only(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A null-title row with only the content gets the count-less heading."""
    client, share = view_client
    rid = "a1b2c3d4e5f60718"
    _write(share, f"handoffs/{rid}.md", "# legacy body")
    await _insert_presentation(rid, [f"handoffs/{rid}.md"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert '<h1 class="vp-presentation-head">Files from your agent</h1>' in html
    assert "vp-presentation-title" not in html
    assert "Shared Sep 01 00:00" in html
    assert " by your agent" not in html
    assert "<title>Files from your agent</title>" in html
    assert f'data-path="handoffs/{rid}.md"' in html  # the body still renders
    assert "vp-feed-files" not in html  # nothing to attach
    assert "thumb-grid" not in html


async def test_presentation_body_null_title_with_grid(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A null-title row with content and files counts grid files in the heading."""
    client, share = view_client
    rid = "b3c4d5e6f7a80912"
    _write(share, f"presentations/{rid}.md", "# body")
    _write(share, "upload/one.png", b"\x89PNG")
    _write(share, "upload/two.png", b"\x89PNG")
    await _insert_presentation(rid, ["upload/one.png", f"presentations/{rid}.md", "upload/two.png"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert '<h1 class="vp-presentation-head">2 files from your agent</h1>' in html
    assert f'data-path="presentations/{rid}.md"' in html
    assert '<div class="vp-feed-files">FILES · 2</div>' in html


async def test_presentation_blank_title_treated_as_null(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    client, share = view_client
    rid = "c5d6e7f809123456"
    _write(share, "upload/a.png", b"\x89PNG")
    await _insert_presentation(rid, ["upload/a.png"], title="   ")
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert "vp-presentation-title" not in html
    assert '<h1 class="vp-presentation-head">1 files from your agent</h1>' in html
    assert " by your agent" not in html
    assert "<title>Files from your agent</title>" in html


async def test_presentation_no_match_no_body(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """Files that never match the stem rule keep today's header-plus-grid layout."""
    client, share = view_client
    rid = "d7e8f90a1b2c3d4e"
    _write(share, "upload/a.png", b"\x89PNG")
    _write(share, "upload/b.png", b"\x89PNG")
    await _insert_presentation(rid, ["upload/a.png", "upload/b.png"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert '<h1 class="vp-presentation-head">2 files from your agent</h1>' in html
    assert "vp-presentation-body" not in html
    assert "vp-feed-files" not in html
    assert '<div class="thumb-grid">' in html


async def test_presentation_unsupported_content_falls_back_to_card(
        view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """An unsupported content file stays a counted grid card, no body, no divider."""
    client, share = view_client
    rid = "f1e2d3c4b5a69788"
    _write(share, "upload/other.png", b"\x89PNG")
    _write(share, f"presentations/{rid}.bin", b"\x00\x01binary")
    await _insert_presentation(rid, ["upload/other.png", f"presentations/{rid}.bin"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert "vp-presentation-body" not in html
    assert "vp-feed-files" not in html
    assert f'<div class="thumb-name">{rid}.bin</div>' in html  # the fallback card
    assert '<h1 class="vp-presentation-head">2 files from your agent</h1>' in html  # counted


async def test_presentation_unsupported_content_gone_counts_grid(
        view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The availability note counts grid files, fallback card included."""
    client, share = view_client
    rid = "0a1b2c3d4e5f6a7b"
    _write(share, f"presentations/{rid}.bin", b"\x00\x01binary")
    await _insert_presentation(rid, ["upload/gone.png", f"presentations/{rid}.bin"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert "vp-presentation-body" not in html
    assert "1 of 2 files available" in html  # the fallback card is a grid file


async def test_presentation_content_gone_state(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A gone content file renders the body gone state. Alive files keep the page."""
    client, share = view_client
    rid = "1c2d3e4f5a6b7c8d"
    _write(share, "upload/other.png", b"\x89PNG")
    await _insert_presentation(rid, [f"presentations/{rid}.md", "upload/other.png"],
                               title="Gone content")
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert "The message content is no longer available" in html
    assert "Files are retained for 30 days, then removed." in html
    assert f'data-path="presentations/{rid}.md"' in html
    assert "well-topbar" not in html  # the gone state has no toolbar
    assert "This link is no longer valid" not in html  # not the expired page
    assert '<div class="vp-feed-files">FILES · 1</div>' in html
    assert "files available" not in html  # the lone grid file is alive


async def test_presentation_partial_grid_gone_availability_note(
        view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The sub-line notes grid files only. The gone content is not in the count."""
    client, share = view_client
    rid = "2d3e4f5a6b7c8d9e"
    _write(share, f"presentations/{rid}.md", "# body")
    _write(share, "upload/alive.png", b"\x89PNG")
    await _insert_presentation(rid, [f"presentations/{rid}.md", "upload/alive.png",
                                     "upload/gone.png"], title="Partial")
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert "Shared Sep 01 00:00 by your agent · 1 of 2 files available" in html
    assert '<div class="vp-feed-files">FILES · 2</div>' in html
    assert "vp-card-gone" in html and "no longer available" in html


async def test_presentation_all_gone_including_content_is_expired(
        view_client: tuple[httpx.AsyncClient, Path]) -> None:
    client, _ = view_client
    rid = "3e4f5a6b7c8d9e0f"
    await _insert_presentation(rid, [f"presentations/{rid}.md", "upload/gone.png"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    assert "This link is no longer valid" in r.text


async def test_presentation_wrong_stem_in_content_folder_is_not_content(
        view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A file in a content folder whose stem is not the id stays a card."""
    client, share = view_client
    rid = "4e5f6a7b8c9d0e1f"
    _write(share, "presentations/notes.md", "# notes")
    await _insert_presentation(rid, ["presentations/notes.md"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert "vp-presentation-body" not in html
    assert '<div class="thumb-name">notes.md</div>' in html
    assert '<h1 class="vp-presentation-head">1 files from your agent</h1>' in html


async def test_presentation_id_stem_outside_content_folder_is_not_content(
        view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A file whose stem is the id but sits outside a content folder stays a card."""
    client, share = view_client
    rid = "5f6a7b8c9d0e1f20"
    _write(share, f"upload/{rid}.md", "# not content")
    await _insert_presentation(rid, [f"upload/{rid}.md"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert "vp-presentation-body" not in html
    assert f'<div class="thumb-name">{rid}.md</div>' in html
    assert '<h1 class="vp-presentation-head">1 files from your agent</h1>' in html


async def test_presentation_first_of_multiple_content_wins(
        view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """Two matching content files. The first shared one becomes the body."""
    client, share = view_client
    rid = "6a7b8c9d0e1f2031"
    _write(share, f"presentations/{rid}.md", "# first")
    _write(share, f"handoffs/{rid}.html", "<p>second</p>")
    await _insert_presentation(rid, [f"presentations/{rid}.md", f"handoffs/{rid}.html"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert f'data-path="presentations/{rid}.md"' in html
    assert f'data-path="handoffs/{rid}.html"' not in html
    assert f'<div class="thumb-name">{rid}.html</div>' in html  # the second stays a card
    assert "first" in html


async def test_presentation_body_frame_html_is_sandboxed(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """An html content file is a sandboxed frame in the body."""
    client, share = view_client
    rid = "7b8c9d0e1f203142"
    _write(share, f"presentations/{rid}.html", "<p>page</p>")
    await _insert_presentation(rid, [f"presentations/{rid}.html"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert (f'<iframe class="frame-body" src="/dav/presentations/{rid}.html?ephemeral=1" '
            'sandbox="allow-same-origin"></iframe>') in html
    assert f'data-path="presentations/{rid}.html"' in html


async def test_presentation_body_frame_pdf_is_inline(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A pdf content file is an inline canvas container in the body, no iframe."""
    client, share = view_client
    rid = "8c9d0e1f20314253"
    _write(share, f"presentations/{rid}.pdf", b"%PDF-1.4 fake")
    await _insert_presentation(rid, [f"presentations/{rid}.pdf"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert f'<div class="vp-frame vp-pdf" data-pdf-src="/dav/presentations/{rid}.pdf?ephemeral=1"></div>' in html
    assert "<iframe" not in html
    assert f'data-path="presentations/{rid}.pdf"' in html


async def test_presentation_body_pptx_uses_pdf_viewer(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A PowerPoint content file shows its converted PDF in the shared viewer."""
    client, share = view_client
    rid = "0e1f203142536475"
    rel = _write(share, f"presentations/{rid}.pptx", b"PK\x03\x04 fake pptx")
    key = cache.key_for(rel, int((share / rel).stat().st_mtime))
    await cache.write_doc(key, "", {cache.OFFICE_PDF_ASSET: b"%PDF-1.7"}, {"mode": "frame"})
    await _insert_presentation(rid, [rel])

    page = await client.get(f"/view/presentation/{rid}")
    assert page.status_code == 200
    assert (f'<div class="vp-frame vp-pdf" data-pdf-src="/view/doc/{key}/'
            f'{cache.OFFICE_PDF_ASSET}"></div>') in page.text


async def test_presentation_body_native_image_carries_no_zoom(
        view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """An image content file is a contained stage in the body, with no zoom."""
    client, share = view_client
    rid = "9d0e1f2031425364"
    _write(share, f"presentations/{rid}.png", b"\x89PNG")
    await _insert_presentation(rid, [f"presentations/{rid}.png"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert f'<div class="img-stage"><img src="/dav/presentations/{rid}.png?ephemeral=1" alt=""></div>' in html
    assert "data-vp-zoom" not in html


async def test_presentation_body_native_video(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A video content file is a media stage in the body."""
    client, share = view_client
    rid = "0e1f203142536475"
    _write(share, f"presentations/{rid}.mp4", b"\x00\x00\x00\x20ftypisom")
    await _insert_presentation(rid, [f"presentations/{rid}.mp4"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert (f'<div class="media-stage"><video controls src="/dav/presentations/{rid}.mp4?ephemeral=1">'
            "</video></div>") in html
    assert "well-topbar" in html


async def test_presentation_body_native_audio_is_a_card(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """An audio content file is an audio card in the body."""
    client, share = view_client
    rid = "1f20314253647586"
    _write(share, f"presentations/{rid}.mp3", b"ID3\x03\x00\x00\x00")
    await _insert_presentation(rid, [f"presentations/{rid}.mp3"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert (f'<div class="audio-card" data-src="/dav/presentations/{rid}.mp3?ephemeral=1" '
            f'data-name="{rid}.mp3"></div>') in html
    assert "well-topbar" in html


async def test_presentation_body_raw_shows_the_note(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A content file that does not read as text shows its note, with the toolbar."""
    client, share = view_client
    rid = "2031425364758697"
    _write(share, f"presentations/{rid}.txt", b"\x00\x01not text")
    await _insert_presentation(rid, [f"presentations/{rid}.txt"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert "vp-well-raw" in html
    assert "No preview available for this file." in html
    assert "well-topbar" in html


async def test_presentation_body_converting_keeps_the_toolbar(
        view_client: tuple[httpx.AsyncClient, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """A content file mid-conversion shows the spinner with the toolbar and no task id."""
    client, share = view_client

    async def fake_start(_path: str, _key: str, _full_path: str) -> None:
        return None

    monkeypatch.setattr(office, "start_office_conversion", fake_start)
    rid = "31425364758697a8"
    _write(share, f"presentations/{rid}.docx", "PK\x03\x04 fake docx")
    await _insert_presentation(rid, [f"presentations/{rid}.docx"])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert '<div class="vp-converting"></div>' in html
    assert "well-topbar" in html
    assert f'data-path="presentations/{rid}.docx"' in html


async def test_presentation_body_convert_failed(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A failed conversion shows the error with Retry and Download in the body."""
    client, share = view_client
    rid = "425364758697a8b9"
    rel = f"presentations/{rid}.docx"
    _write(share, rel, "PK\x03\x04 fake docx")
    st = (share / rel).stat()
    await cache.mark_failed(cache.key_for(rel, int(st.st_mtime)), "convert", "boom")
    await _insert_presentation(rid, [rel])
    r = await client.get(f"/view/presentation/{rid}")
    assert r.status_code == 200
    html = r.text
    assert f'data-vp-retry="/api/view/presentations/{rid}.docx/retry"' in html
    assert f'href="/dav/dl?path=presentations/{rid}.docx"' in html
    assert "boom" in html


async def test_presentation_body_css(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The body-scoped rules exist and the pinned existing mechanisms are intact."""
    client, _ = view_client
    r = await client.get("/assets/css/view.css")
    assert r.status_code == 200
    css = r.text
    assert ".vp-presentation-title{font-size:var(--text-xl);font-weight:600;" \
           "line-height:1.3;letter-spacing:-.01em}" in css
    assert ".vp-presentation-body{margin-top:18px}" in css
    assert ".well-actions{margin-left:auto;display:flex;align-items:center;gap:6px;flex:none}" in css
    assert ".frame-body{display:block;width:100%;height:480px;border:none;background:#fff}" in css
    assert ".frame-body{height:360px}" in css  # the mobile frame height
    assert ".vp-presentation-body .vp-pdf{height:480px}" in css  # the pdf body frame cap
    assert ".vp-presentation-body .vp-pdf{height:360px}" in css  # the mobile pdf body frame height
    assert ".vp-presentation-body .img-stage{height:auto;min-height:0;max-height:520px;" \
           "border:none;border-radius:0}" in css
    assert ".vp-presentation-body .img-stage > img{width:auto;height:auto;max-height:520px}" in css
    assert ".vp-presentation-body .audio-card{align-self:center;margin-top:20px}" in css
    assert ".thumb-grid.after-divider{margin-top:0}" in css
    assert ".vp-body-gone{display:flex;flex-direction:column;align-items:center;gap:6px;" \
           "text-align:center;padding:44px 24px}" in css
    assert ".vp-body-gone .vp-state-ic{opacity:.6;margin-bottom:6px}" in css
    assert ".vp-body-gone-title{font-size:var(--text-base);font-weight:600}" in css
    assert ".vp-body-gone-sub{font-size:var(--text-base);line-height:1.5;color:var(--text-secondary)}" in css
    # the mobile stage keeps following small images, over the 300px floor
    assert ".vp-presentation-body .img-stage{min-height:0}" in css
    # the wide-table scroll mechanism stays
    assert ".vp-well-doc .md-doc{flex:1;min-height:0;overflow:auto;padding:28px 32px}" in css
    # the gone-card name token stays
    assert ".vp-card-gone .thumb-name{color:var(--text-secondary)}" in css
    # no table-level overflow rule is added
    assert ".md-doc table{" in css
    assert "overflow-x" not in css.split(".md-doc table{")[1].split("}")[0]


async def test_presentation_css_does_not_touch_existing_rules(
        view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The shared rules the body must not re-scope are byte-identical to before."""
    client, _ = view_client
    r = await client.get("/assets/css/view.css")
    assert r.status_code == 200
    css = r.text
    assert ".vp-presentation-head{font-size:var(--text-base);font-weight:600;letter-spacing:-.01em}" in css
    assert ".vp-presentation-sub{font-size:var(--text-sm);line-height:1.5;color:var(--text-tertiary);margin-top:5px}" in css
    assert ".img-stage{position:relative;flex:1;min-height:0;display:flex;overflow:auto;background:#0a0d12}" in css
    assert ".media-stage img,.media-stage video{display:block;max-width:100%;max-height:90vh;" \
           "width:auto;height:auto;object-fit:contain}" in css


async def test_presentation_body_client_js(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The client script resolves file paths from elements, not just the URL."""
    client, _ = view_client
    r = await client.get("/assets/js/view-page.js")
    assert r.status_code == 200
    js = r.text
    assert "function _filePath(el)" in js
    assert 'location.pathname.startsWith("/view/presentation/")' in js
    assert 'el.closest(".vp-presentation-body")' in js
    assert "section.dataset.dlHref" in js
    assert "The message content is no longer available" in js
    assert "function _renderPresentationBody(d, path, dlHref)" in js
