"""Tests for the solo resource file viewer page."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

import app.lib.render.cache as cache
from tests.app.viewer.conftest import compute_cache_key, write_resource_file


async def test_view_file_page_solo_full_screen(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The single file page has no top bar and no footer, only a Download pill."""
    client, resources_dir = view_client
    write_resource_file(resources_dir, "upload/notes.md", "# Hi\n\nbody")
    response = await client.get("/view/upload/notes.md")
    assert response.status_code == 200
    html = response.text
    assert "vp-solo" in html
    assert "vp-topbar" not in html
    assert "vp-footer" not in html
    assert "Agent Hub" not in html
    assert "data-vp-theme" not in html and "data-vp-copy" not in html
    assert "vp-filename" not in html and "vp-meta" not in html
    assert "vp-solo-label" not in html
    assert 'href="/dav/dl?path=upload/notes.md"' in html
    assert 'target="_blank"' not in html
    assert "path=upload/notes.md" in html
    assert 'href="/assets/css/view.css"' in html
    assert '<script type="module" src="/assets/js/view-page.js">' in html
    assert "agent-console-theme" in html
    assert 'matchMedia("(prefers-color-scheme: dark)")' in html


async def test_view_page_pre_state_theme_script(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The pre-paint script follows the system setting, and a saved pick under that setting wins."""
    client, resources_dir = view_client
    write_resource_file(resources_dir, "upload/notes.md", "# Hi")
    response = await client.get("/view/upload/notes.md")
    assert response.status_code == 200
    html = response.text
    assert 'try{t=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";}catch(e){t="light";}' in html
    assert 'try{s=JSON.parse(localStorage.getItem("agent-console-theme"));}catch(e){s=null;}' in html
    assert 'if(s&&(s.theme==="dark"||s.theme==="light")&&s.scheme===t){t=s.theme;}' in html


async def test_view_solo_pdf_renders_the_inline_canvas_container(
    view_client: tuple[httpx.AsyncClient, Path],
) -> None:
    """A PDF file page carries the canvas container, no iframe, and the client script."""
    client, resources_dir = view_client
    write_resource_file(resources_dir, "upload/p.pdf", b"%PDF-1.4 fake")
    response = await client.get("/view/upload/p.pdf")
    assert response.status_code == 200
    html = response.text
    assert '<div class="vp-frame vp-pdf" data-pdf-src="/dav/upload/p.pdf?ephemeral=1"></div>' in html
    assert "<iframe" not in html
    assert 'src="/assets/js/view-page.js"' in html
    assert 'href="/dav/dl?path=upload/p.pdf"' in html


async def test_view_presentation_uses_cached_pdf(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A PowerPoint preview serves its converted slides to the PDF viewer."""
    client, resources_dir = view_client
    relative_path = write_resource_file(resources_dir, "upload/deck.pptx", b"PK\x03\x04 fake pptx")
    cache_key = compute_cache_key(resources_dir, relative_path)
    pdf_bytes = b"%PDF-1.7\nslide"
    await cache.write_doc(cache_key, "", {cache.OFFICE_PDF_ASSET: pdf_bytes}, {"mode": "frame"})

    page_response = await client.get(f"/view/{relative_path}")
    assert page_response.status_code == 200
    assert (
        f'<div class="vp-frame vp-pdf" data-pdf-src="/view/doc/{cache_key}/'
        f'{cache.OFFICE_PDF_ASSET}"></div>'
    ) in page_response.text
    assert f'href="/dav/dl?path={relative_path}"' in page_response.text
    asset_response = await client.get(f"/view/doc/{cache_key}/{cache.OFFICE_PDF_ASSET}")
    assert asset_response.status_code == 200
    assert asset_response.headers["content-type"] == "application/pdf"
    assert asset_response.content == pdf_bytes


async def test_view_solo_html_keeps_the_sandboxed_iframe(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """An HTML file page is a sandboxed iframe without the PDF container."""
    client, resources_dir = view_client
    write_resource_file(resources_dir, "upload/p.html", "<p>page</p>")
    response = await client.get("/view/upload/p.html")
    assert response.status_code == 200
    html = response.text
    assert '<iframe class="vp-frame" src="/dav/upload/p.html?ephemeral=1" sandbox="allow-same-origin"></iframe>' in html
    assert "vp-pdf" not in html


@pytest.mark.parametrize(
    "asset_path",
    [
        "/assets/js/pdf-viewer.js",
        "/assets/js/pdfjs/pdf.min.mjs",
        "/assets/js/pdfjs/pdf.worker.min.mjs",
    ],
)
async def test_pdf_viewer_and_vendor_assets_served(
    view_client: tuple[httpx.AsyncClient, Path],
    asset_path: str,
) -> None:
    """The PDF viewer module and vendored PDF.js files serve as JavaScript."""
    client, _ = view_client
    response = await client.get(asset_path)
    assert response.status_code == 200
    assert "text/javascript" in response.headers["content-type"]


async def test_pdf_viewer_asset_content(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The served PDF viewer builds the canvas viewer, and its CSS and script carry their parts."""
    client, _ = view_client
    viewer_js = (await client.get("/assets/js/pdf-viewer.js")).text
    assert "export async function mountPdfViewer(container, davUrl)" in viewer_js
    assert "workerSrc" in viewer_js
    assert "IntersectionObserver" in viewer_js
    assert "./pdfjs/pdf.min.mjs" in viewer_js
    assert "/assets/js/pdfjs/pdf.worker.min.mjs" in viewer_js
    assert "data-pdf-src" in viewer_js
    assert "new pdfjsLib.AnnotationLayer(" in viewer_js
    assert "getAnnotations()" in viewer_js
    assert "goToDestination" in viewer_js
    assert "addLinkAttributes" in viewer_js
    viewer_css = (await client.get("/assets/css/view.css")).text
    assert ".vp-frame.vp-pdf{display:flex;flex-direction:column;overflow:hidden}" in viewer_css
    assert ".pv-toolbar{" in viewer_css and ".pv-pages{" in viewer_css and ".pv-error{" in viewer_css
    assert ".pv-ann section{" in viewer_css
    view_page_js = (await client.get("/assets/js/view-page.js")).text
    assert 'from "./pdf-viewer.js"' in view_page_js


async def test_view_file_unsupported_state_card(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """An unsupported file shows a card with its name, Download, and Go back."""
    client, resources_dir = view_client
    write_resource_file(resources_dir, "upload/blob.bin", b"\x00\x01binary")
    response = await client.get("/view/upload/blob.bin")
    assert response.status_code == 200
    html = response.text
    assert "vp-unsupported" in html and "vp-state-card" in html
    assert "blob.bin" in html
    assert 'href="/dav/dl?path=upload/blob.bin"' in html
    assert "data-vp-back" in html


async def test_view_missing_file_is_expired_page(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A missing resource file renders the expired link page with no download link."""
    client, _ = view_client
    response = await client.get("/view/never/existed.md")
    assert response.status_code == 200
    assert "This link is no longer valid" in response.text
    assert "no longer valid" in response.text
    assert 'href="/dav/' not in response.text


async def test_view_existing_folder_is_plain_404(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """Requesting a directory path returns a plain text 404 response."""
    client, resources_dir = view_client
    (resources_dir / "somefolder").mkdir()
    response = await client.get("/view/somefolder")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/plain")


async def test_view_media_src_url_encoded(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A media page source for a path with special characters is percent-encoded."""
    client, resources_dir = view_client
    write_resource_file(resources_dir, "upload/pic #1.png", b"\x89PNG")
    response = await client.get("/view/upload/pic%20%231.png")
    assert response.status_code == 200
    assert 'src="/dav/upload/pic%20%231.png?ephemeral=1"' in response.text
    assert 'src="/dav/upload/pic #1.png"' not in response.text
