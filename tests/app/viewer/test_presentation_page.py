"""Tests for the presentation viewer page."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

import app.lib.db.store as store_module
import app.lib.render.cache as cache
import app.lib.render.office as office
from tests.app.viewer.conftest import compute_cache_key, write_resource_file


async def _insert_presentation(
    presentation_id: str,
    files: list[str],
    title: str | None = None,
    presented_at: str = "2026-09-01T00:00:00+00:00",
) -> None:
    await store_module.presentation_store.insert(
        presentation_id,
        presented_at,
        f"/view/presentation/{presentation_id}",
        files,
        title=title,
    )


def _extract_card_html(page_html: str, index: int) -> str:
    segment = page_html.split('<div class="thumb-card')[1:][index]
    return '<div class="thumb-card' + segment.split("</div></div>")[0] + "</div></div>"


def _find_thumb_actions_rule(css: str) -> str:
    prefix = ".thumb-grid .thumb-actions{"
    for line in css.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError("served view.css is missing the .thumb-grid .thumb-actions rule")


async def test_presentation_page_keeps_chrome(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The presentation page has a top bar with the brand and theme button, plus a footer."""
    client, resources_dir = view_client
    write_resource_file(resources_dir, "upload/a.png", b"\x89PNG")
    presentation_id = "a" * 16
    await _insert_presentation(presentation_id, ["upload/a.png"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    assert "vp-topbar" in response.text and "vp-footer" in response.text
    assert "Agent Hub" in response.text
    assert "data-vp-theme" in response.text and "data-vp-copy" not in response.text
    assert "vp-solo" not in response.text


async def test_presentation_page_lists_files(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The presentation page lists each presented file with open and download links."""
    client, resources_dir = view_client
    write_resource_file(resources_dir, "upload/a.png", b"\x89PNG")
    write_resource_file(resources_dir, "upload/b.md", b"# hi")
    presentation_id = "b" * 16
    await _insert_presentation(presentation_id, ["upload/a.png", "upload/b.md"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    assert "2 files from your agent" in response.text
    assert 'href="/view/upload/a.png"' in response.text
    assert 'href="/dav/dl?path=upload/a.png"' in response.text
    assert 'target="_blank"' not in response.text
    assert "markdown" in response.text


async def test_presentation_card_is_whole_card_link(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A present card is one link over the whole card, with Download beside it."""
    client, resources_dir = view_client
    write_resource_file(resources_dir, "upload/a.png", b"\x89PNG")
    presentation_id = "a2" * 8
    await _insert_presentation(presentation_id, ["upload/a.png"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert '<a class="vp-card-open" href="/view/upload/a.png" aria-label="Open a.png"></a>' in html
    assert ">Open</a>" not in html
    assert 'class="btn ghost" href="/view/' not in html
    assert 'href="/dav/dl?path=upload/a.png"' in html
    assert html.index('class="vp-card-open"') < html.index('href="/dav/dl?path=upload/a.png"')


async def test_presentation_gone_card_is_non_interactive(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A card for a missing file has no links, and a present card beside it still does."""
    client, resources_dir = view_client
    write_resource_file(resources_dir, "upload/alive.png", b"\x89PNG")
    presentation_id = "b2" * 8
    await _insert_presentation(presentation_id, ["upload/alive.png", "upload/gone.png"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    present_card = _extract_card_html(response.text, 0)
    missing_card = _extract_card_html(response.text, 1)
    assert 'href="/view/upload/alive.png"' in present_card
    assert "vp-card-gone" in missing_card
    assert "<a" not in missing_card
    assert "thumb-actions" not in missing_card


async def test_presentation_card_open_is_native_anchor(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The card open control is a plain link, not a button."""
    client, resources_dir = view_client
    write_resource_file(resources_dir, "upload/a.png", b"\x89PNG")
    presentation_id = "c2" * 8
    await _insert_presentation(presentation_id, ["upload/a.png"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert '<a class="vp-card-open" href="/view/upload/a.png"' in html
    assert 'role="button"' not in html
    assert "tabindex" not in html


async def test_presentation_focus_ring_is_on_card(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The stylesheet draws the focus ring on the card and hides the link outline."""
    client, _ = view_client
    response = await client.get("/assets/css/view.css")
    assert response.status_code == 200
    css = response.text
    assert ".thumb-grid .thumb-card:focus-within{outline:2px solid var(--accent);outline-offset:1px}" in css
    assert ".vp-card-open:focus-visible{outline:none}" in css


async def test_presentation_download_actions_reset_inherited_offsets(
    view_client: tuple[httpx.AsyncClient, Path],
) -> None:
    """The card actions rule resets left and bottom."""
    client, _ = view_client
    response = await client.get("/assets/css/view.css")
    assert response.status_code == 200
    rule = _find_thumb_actions_rule(response.text)
    assert "left:auto" in rule
    assert "bottom:auto" in rule


async def test_presentation_thumb_src_url_encoded(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A thumbnail source for a path with special characters is escaped."""
    client, resources_dir = view_client
    relative_path = "upload/photo & more.png"
    write_resource_file(resources_dir, relative_path, b"\x89PNG")
    key_dir = os.path.join(cache.render_root(), compute_cache_key(resources_dir, relative_path))
    os.makedirs(key_dir, exist_ok=True)
    with open(os.path.join(key_dir, "thumb.png"), "wb") as thumb_file:
        thumb_file.write(b"\x89PNG")
    presentation_id = "d" * 16
    await _insert_presentation(presentation_id, [relative_path])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    assert f'src="{cache.thumb_url(relative_path)}"' in response.text
    assert "path=upload/photo%20%26%20more.png" in response.text


async def test_presentation_missing_id_is_expired(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """An unknown presentation id renders the expired link page."""
    client, _ = view_client
    response = await client.get("/view/presentation/" + "f" * 16)
    assert response.status_code == 200
    assert "This link is no longer valid" in response.text


async def test_presentation_non_hex_id_delegate_to_file(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A non-hex presentation path delegates to the matching resource file."""
    client, resources_dir = view_client
    write_resource_file(resources_dir, "presentation/plain", "# delegated")
    response = await client.get("/view/presentation/plain")
    assert response.status_code == 200
    assert "delegated" in response.text
    assert "presentation/plain" in response.text


async def test_presentation_all_gone_is_expired(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A presentation whose files are all missing renders the expired link page."""
    client, _ = view_client
    presentation_id = "c" * 16
    await _insert_presentation(presentation_id, ["upload/gone.png"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    assert "This link is no longer valid" in response.text


async def test_presentation_open_href_url_encoded(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A card open link for a path with special characters is escaped."""
    client, resources_dir = view_client
    relative_path = "upload/a & b.png"
    write_resource_file(resources_dir, relative_path, b"\x89PNG")
    presentation_id = "e" * 16
    await _insert_presentation(presentation_id, [relative_path])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    assert 'href="/view/upload/a%20%26%20b.png"' in response.text
    assert 'href="/dav/dl?path=upload/a%20%26%20b.png"' in response.text


async def test_presentation_body_titled_markdown(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A titled row renders the title, the body well, and the files divider."""
    client, resources_dir = view_client
    presentation_id = "e5d3049aa28546de"
    write_resource_file(resources_dir, f"presentations/{presentation_id}.md", "# Hi\n\nbody text")
    write_resource_file(resources_dir, "upload/attach.png", b"\x89PNG")
    await _insert_presentation(
        presentation_id,
        [f"presentations/{presentation_id}.md", "upload/attach.png"],
        title="Weekend swim lessons",
    )
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert '<h1 class="vp-presentation-title">Weekend swim lessons</h1>' in html
    assert "Shared Sep 01 00:00 by your agent" in html
    assert "<title>Weekend swim lessons</title>" in html
    assert "Opened from a link" not in html
    assert (
        f'<div class="vp-well vp-presentation-body vp-well-doc" '
        f'data-path="presentations/{presentation_id}.md" '
        f'data-dl-href="/dav/dl?path=presentations/{presentation_id}.md">'
    ) in html
    assert '<span class="chip">markdown</span>' in html
    assert f'<a class="vp-icon-btn" href="/view/presentations/{presentation_id}.md" target="_blank"' in html
    assert f'<div class="thumb-name">{presentation_id}.md</div>' not in html
    assert '<div class="vp-feed-files">FILES · 1</div>' in html
    assert '<div class="thumb-grid after-divider">' in html
    assert '<div class="thumb-name">attach.png</div>' in html
    assert '<div class="md-doc">' in html and "body text" in html


async def test_presentation_body_plain_text(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A plain-text content file renders the body well like the file viewer does."""
    client, resources_dir = view_client
    presentation_id = "9f8e7d6c5b4a0312"
    write_resource_file(resources_dir, f"presentations/{presentation_id}.txt", "just a note\nsecond line")
    await _insert_presentation(presentation_id, [f"presentations/{presentation_id}.txt"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert (
        f'<div class="vp-well vp-presentation-body vp-well-plain" '
        f'data-path="presentations/{presentation_id}.txt" '
        f'data-dl-href="/dav/dl?path=presentations/{presentation_id}.txt">'
    ) in html
    assert 'class="plain-pre">' in html
    assert "just a note" in html


async def test_presentation_body_null_title_content_only(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A null-title row with only the content gets the count-less heading."""
    client, resources_dir = view_client
    presentation_id = "a1b2c3d4e5f60718"
    write_resource_file(resources_dir, f"handoffs/{presentation_id}.md", "# body")
    await _insert_presentation(presentation_id, [f"handoffs/{presentation_id}.md"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert '<h1 class="vp-presentation-head">Files from your agent</h1>' in html
    assert "vp-presentation-title" not in html
    assert "Shared Sep 01 00:00" in html
    assert " by your agent" not in html
    assert "<title>Files from your agent</title>" in html
    assert f'data-path="handoffs/{presentation_id}.md"' in html
    assert "vp-feed-files" not in html
    assert "thumb-grid" not in html


async def test_presentation_body_null_title_with_grid(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A null-title row with content and files counts grid files in the heading."""
    client, resources_dir = view_client
    presentation_id = "b3c4d5e6f7a80912"
    write_resource_file(resources_dir, f"presentations/{presentation_id}.md", "# body")
    write_resource_file(resources_dir, "upload/one.png", b"\x89PNG")
    write_resource_file(resources_dir, "upload/two.png", b"\x89PNG")
    await _insert_presentation(
        presentation_id,
        ["upload/one.png", f"presentations/{presentation_id}.md", "upload/two.png"],
    )
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert '<h1 class="vp-presentation-head">2 files from your agent</h1>' in html
    assert f'data-path="presentations/{presentation_id}.md"' in html
    assert '<div class="vp-feed-files">FILES · 2</div>' in html


async def test_presentation_blank_title_treated_as_null(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A whitespace-only title uses the generic heading."""
    client, resources_dir = view_client
    presentation_id = "c5d6e7f809123456"
    write_resource_file(resources_dir, "upload/a.png", b"\x89PNG")
    await _insert_presentation(presentation_id, ["upload/a.png"], title="   ")
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert "vp-presentation-title" not in html
    assert '<h1 class="vp-presentation-head">1 files from your agent</h1>' in html
    assert " by your agent" not in html
    assert "<title>Files from your agent</title>" in html


async def test_presentation_no_match_no_body(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """Files that never match the stem rule keep the header-plus-grid layout."""
    client, resources_dir = view_client
    presentation_id = "d7e8f90a1b2c3d4e"
    write_resource_file(resources_dir, "upload/a.png", b"\x89PNG")
    write_resource_file(resources_dir, "upload/b.png", b"\x89PNG")
    await _insert_presentation(presentation_id, ["upload/a.png", "upload/b.png"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert '<h1 class="vp-presentation-head">2 files from your agent</h1>' in html
    assert "vp-presentation-body" not in html
    assert "vp-feed-files" not in html
    assert '<div class="thumb-grid">' in html


async def test_presentation_unsupported_content_falls_back_to_card(
    view_client: tuple[httpx.AsyncClient, Path],
) -> None:
    """An unsupported content file stays a counted grid card with no body and no divider."""
    client, resources_dir = view_client
    presentation_id = "f1e2d3c4b5a69788"
    write_resource_file(resources_dir, "upload/other.png", b"\x89PNG")
    write_resource_file(resources_dir, f"presentations/{presentation_id}.bin", b"\x00\x01binary")
    await _insert_presentation(presentation_id, ["upload/other.png", f"presentations/{presentation_id}.bin"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert "vp-presentation-body" not in html
    assert "vp-feed-files" not in html
    assert f'<div class="thumb-name">{presentation_id}.bin</div>' in html
    assert '<h1 class="vp-presentation-head">2 files from your agent</h1>' in html


async def test_presentation_unsupported_content_gone_counts_grid(
    view_client: tuple[httpx.AsyncClient, Path],
) -> None:
    """The availability note counts grid files, including the fallback card."""
    client, resources_dir = view_client
    presentation_id = "0a1b2c3d4e5f6a7b"
    write_resource_file(resources_dir, f"presentations/{presentation_id}.bin", b"\x00\x01binary")
    await _insert_presentation(presentation_id, ["upload/gone.png", f"presentations/{presentation_id}.bin"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert "vp-presentation-body" not in html
    assert "1 of 2 files available" in html


async def test_presentation_content_gone_state(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A missing content file renders the missing body state while remaining files keep the page."""
    client, resources_dir = view_client
    presentation_id = "1c2d3e4f5a6b7c8d"
    write_resource_file(resources_dir, "upload/other.png", b"\x89PNG")
    await _insert_presentation(
        presentation_id,
        [f"presentations/{presentation_id}.md", "upload/other.png"],
        title="Gone content",
    )
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert "The message content is no longer available" in html
    assert "Files are retained for 30 days, then removed." in html
    assert f'data-path="presentations/{presentation_id}.md"' in html
    assert "well-topbar" not in html
    assert "This link is no longer valid" not in html
    assert '<div class="vp-feed-files">FILES · 1</div>' in html
    assert "files available" not in html


async def test_presentation_partial_grid_gone_availability_note(
    view_client: tuple[httpx.AsyncClient, Path],
) -> None:
    """The sub-line counts grid files only and omits the content file from the count."""
    client, resources_dir = view_client
    presentation_id = "2d3e4f5a6b7c8d9e"
    write_resource_file(resources_dir, f"presentations/{presentation_id}.md", "# body")
    write_resource_file(resources_dir, "upload/alive.png", b"\x89PNG")
    await _insert_presentation(
        presentation_id,
        [f"presentations/{presentation_id}.md", "upload/alive.png", "upload/gone.png"],
        title="Partial",
    )
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert "Shared Sep 01 00:00 by your agent · 1 of 2 files available" in html
    assert '<div class="vp-feed-files">FILES · 2</div>' in html
    assert "vp-card-gone" in html and "no longer available" in html


async def test_presentation_all_gone_including_content_is_expired(
    view_client: tuple[httpx.AsyncClient, Path],
) -> None:
    """When both content and grid files are missing, the expired page is shown."""
    client, _ = view_client
    presentation_id = "3e4f5a6b7c8d9e0f"
    await _insert_presentation(presentation_id, [f"presentations/{presentation_id}.md", "upload/gone.png"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    assert "This link is no longer valid" in response.text


async def test_presentation_wrong_stem_in_content_folder_is_not_content(
    view_client: tuple[httpx.AsyncClient, Path],
) -> None:
    """A file in a content folder whose stem is not the id stays a card."""
    client, resources_dir = view_client
    presentation_id = "4e5f6a7b8c9d0e1f"
    write_resource_file(resources_dir, "presentations/notes.md", "# notes")
    await _insert_presentation(presentation_id, ["presentations/notes.md"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert "vp-presentation-body" not in html
    assert '<div class="thumb-name">notes.md</div>' in html
    assert '<h1 class="vp-presentation-head">1 files from your agent</h1>' in html


async def test_presentation_id_stem_outside_content_folder_is_not_content(
    view_client: tuple[httpx.AsyncClient, Path],
) -> None:
    """A file whose stem is the id outside a content folder stays a card."""
    client, resources_dir = view_client
    presentation_id = "5f6a7b8c9d0e1f20"
    write_resource_file(resources_dir, f"upload/{presentation_id}.md", "# not content")
    await _insert_presentation(presentation_id, [f"upload/{presentation_id}.md"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert "vp-presentation-body" not in html
    assert f'<div class="thumb-name">{presentation_id}.md</div>' in html
    assert '<h1 class="vp-presentation-head">1 files from your agent</h1>' in html


async def test_presentation_first_of_multiple_content_wins(
    view_client: tuple[httpx.AsyncClient, Path],
) -> None:
    """When two content files match, the first presented file becomes the body."""
    client, resources_dir = view_client
    presentation_id = "6a7b8c9d0e1f2031"
    write_resource_file(resources_dir, f"presentations/{presentation_id}.md", "# first")
    write_resource_file(resources_dir, f"handoffs/{presentation_id}.html", "<p>second</p>")
    await _insert_presentation(
        presentation_id,
        [f"presentations/{presentation_id}.md", f"handoffs/{presentation_id}.html"],
    )
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert f'data-path="presentations/{presentation_id}.md"' in html
    assert f'data-path="handoffs/{presentation_id}.html"' not in html
    assert f'<div class="thumb-name">{presentation_id}.html</div>' in html
    assert "first" in html


async def test_presentation_body_frame_html_is_sandboxed(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """An HTML content file renders as a sandboxed iframe in the body."""
    client, resources_dir = view_client
    presentation_id = "7b8c9d0e1f203142"
    write_resource_file(resources_dir, f"presentations/{presentation_id}.html", "<p>page</p>")
    await _insert_presentation(presentation_id, [f"presentations/{presentation_id}.html"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert (
        f'<iframe class="frame-body" src="/dav/presentations/{presentation_id}.html?ephemeral=1" '
        'sandbox="allow-same-origin"></iframe>'
    ) in html
    assert f'data-path="presentations/{presentation_id}.html"' in html


async def test_presentation_body_frame_pdf_is_inline(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A PDF content file renders as an inline canvas container with no iframe."""
    client, resources_dir = view_client
    presentation_id = "8c9d0e1f20314253"
    write_resource_file(resources_dir, f"presentations/{presentation_id}.pdf", b"%PDF-1.4 fake")
    await _insert_presentation(presentation_id, [f"presentations/{presentation_id}.pdf"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert f'<div class="vp-frame vp-pdf" data-pdf-src="/dav/presentations/{presentation_id}.pdf?ephemeral=1"></div>' in html
    assert "<iframe" not in html
    assert f'data-path="presentations/{presentation_id}.pdf"' in html


async def test_presentation_body_pptx_uses_pdf_viewer(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A PowerPoint content file shows its converted PDF in the viewer."""
    client, resources_dir = view_client
    presentation_id = "0e1f203142536475"
    relative_path = write_resource_file(resources_dir, f"presentations/{presentation_id}.pptx", b"PK\x03\x04 fake pptx")
    cache_key = compute_cache_key(resources_dir, relative_path)
    await cache.write_doc(cache_key, "", {cache.OFFICE_PDF_ASSET: b"%PDF-1.7"}, {"mode": "frame"})
    await _insert_presentation(presentation_id, [relative_path])

    page_response = await client.get(f"/view/presentation/{presentation_id}")
    assert page_response.status_code == 200
    assert (
        f'<div class="vp-frame vp-pdf" data-pdf-src="/view/doc/{cache_key}/'
        f'{cache.OFFICE_PDF_ASSET}"></div>'
    ) in page_response.text


async def test_presentation_body_native_image_carries_no_zoom(
    view_client: tuple[httpx.AsyncClient, Path],
) -> None:
    """An image content file renders in a contained stage with no zoom attribute."""
    client, resources_dir = view_client
    presentation_id = "9d0e1f2031425364"
    write_resource_file(resources_dir, f"presentations/{presentation_id}.png", b"\x89PNG")
    await _insert_presentation(presentation_id, [f"presentations/{presentation_id}.png"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert f'<div class="img-stage"><img src="/dav/presentations/{presentation_id}.png?ephemeral=1" alt=""></div>' in html
    assert "data-vp-zoom" not in html


async def test_presentation_body_native_video(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A video content file renders in a media stage with the toolbar."""
    client, resources_dir = view_client
    presentation_id = "0e1f203142536475"
    write_resource_file(resources_dir, f"presentations/{presentation_id}.mp4", b"\x00\x00\x00\x20ftypisom")
    await _insert_presentation(presentation_id, [f"presentations/{presentation_id}.mp4"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert (
        f'<div class="media-stage"><video controls src="/dav/presentations/{presentation_id}.mp4?ephemeral=1">'
        "</video></div>"
    ) in html
    assert "well-topbar" in html


async def test_presentation_body_native_audio_is_a_card(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """An audio content file renders as an audio card with the toolbar."""
    client, resources_dir = view_client
    presentation_id = "1f20314253647586"
    write_resource_file(resources_dir, f"presentations/{presentation_id}.mp3", b"ID3\x03\x00\x00\x00")
    await _insert_presentation(presentation_id, [f"presentations/{presentation_id}.mp3"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert (
        f'<div class="audio-card" data-src="/dav/presentations/{presentation_id}.mp3?ephemeral=1" '
        f'data-name="{presentation_id}.mp3"></div>'
    ) in html
    assert "well-topbar" in html


async def test_presentation_body_raw_shows_the_note(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A content file that cannot be read as text shows its note with the toolbar."""
    client, resources_dir = view_client
    presentation_id = "2031425364758697"
    write_resource_file(resources_dir, f"presentations/{presentation_id}.txt", b"\x00\x01not text")
    await _insert_presentation(presentation_id, [f"presentations/{presentation_id}.txt"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert "vp-well-raw" in html
    assert "No preview available for this file." in html
    assert "well-topbar" in html


async def test_presentation_body_converting_keeps_the_toolbar(
    view_client: tuple[httpx.AsyncClient, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A content file mid-conversion shows the spinner with the toolbar."""
    client, resources_dir = view_client

    async def fake_start(_path: str, _key: str, _full_path: str) -> None:
        return None

    monkeypatch.setattr(office, "start_office_conversion", fake_start)
    presentation_id = "31425364758697a8"
    write_resource_file(resources_dir, f"presentations/{presentation_id}.docx", "PK\x03\x04 fake docx")
    await _insert_presentation(presentation_id, [f"presentations/{presentation_id}.docx"])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert '<div class="vp-converting"></div>' in html
    assert "well-topbar" in html
    assert f'data-path="presentations/{presentation_id}.docx"' in html


async def test_presentation_body_convert_failed(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A failed conversion shows the error with Retry and Download in the body."""
    client, resources_dir = view_client
    presentation_id = "425364758697a8b9"
    relative_path = f"presentations/{presentation_id}.docx"
    write_resource_file(resources_dir, relative_path, "PK\x03\x04 fake docx")
    await cache.mark_failed(compute_cache_key(resources_dir, relative_path), "convert", "boom")
    await _insert_presentation(presentation_id, [relative_path])
    response = await client.get(f"/view/presentation/{presentation_id}")
    assert response.status_code == 200
    html = response.text
    assert f'data-vp-retry="/api/view/presentations/{presentation_id}.docx/retry"' in html
    assert f'href="/dav/dl?path=presentations/{presentation_id}.docx"' in html
    assert "boom" in html


async def test_presentation_body_css(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The body-scoped rules exist and the pinned existing rules are intact."""
    client, _ = view_client
    response = await client.get("/assets/css/view.css")
    assert response.status_code == 200
    css = response.text
    assert (
        ".vp-presentation-title{font-size:var(--text-xl);font-weight:600;"
        "line-height:1.3;letter-spacing:-.01em}"
    ) in css
    assert ".vp-presentation-body{margin-top:18px}" in css
    assert ".well-actions{margin-left:auto;display:flex;align-items:center;gap:6px;flex:none}" in css
    assert ".frame-body{display:block;width:100%;height:480px;border:none;background:#fff}" in css
    assert ".frame-body{height:360px}" in css
    assert ".vp-presentation-body .vp-pdf{height:480px}" in css
    assert ".vp-presentation-body .vp-pdf{height:360px}" in css
    assert (
        ".vp-presentation-body .img-stage{height:auto;min-height:0;max-height:520px;"
        "border:none;border-radius:0}"
    ) in css
    assert ".vp-presentation-body .img-stage > img{width:auto;height:auto;max-height:520px}" in css
    assert ".vp-presentation-body .audio-card{align-self:center;margin-top:20px}" in css
    assert ".thumb-grid.after-divider{margin-top:0}" in css
    assert (
        ".vp-body-gone{display:flex;flex-direction:column;align-items:center;gap:6px;"
        "text-align:center;padding:44px 24px}"
    ) in css
    assert ".vp-body-gone .vp-state-ic{opacity:.6;margin-bottom:6px}" in css
    assert ".vp-body-gone-title{font-size:var(--text-base);font-weight:600}" in css
    assert ".vp-body-gone-sub{font-size:var(--text-base);line-height:1.5;color:var(--text-secondary)}" in css
    assert ".vp-presentation-body .img-stage{min-height:0}" in css
    assert ".vp-well-doc .md-doc{flex:1;min-height:0;overflow:auto;padding:28px 32px}" in css
    assert ".vp-card-gone .thumb-name{color:var(--text-secondary)}" in css
    assert ".md-doc table{" in css
    assert "overflow-x" not in css.split(".md-doc table{")[1].split("}")[0]


async def test_presentation_css_does_not_touch_existing_rules(
    view_client: tuple[httpx.AsyncClient, Path],
) -> None:
    """The shared rules the body must not re-scope stay intact."""
    client, _ = view_client
    response = await client.get("/assets/css/view.css")
    assert response.status_code == 200
    css = response.text
    assert ".vp-presentation-head{font-size:var(--text-base);font-weight:600;letter-spacing:-.01em}" in css
    assert ".vp-presentation-sub{font-size:var(--text-sm);line-height:1.5;color:var(--text-tertiary);margin-top:5px}" in css
    assert ".img-stage{position:relative;flex:1;min-height:0;display:flex;overflow:auto;background:#0a0d12}" in css
    assert (
        ".media-stage img,.media-stage video{display:block;max-width:100%;max-height:90vh;"
        "width:auto;height:auto;object-fit:contain}"
    ) in css


async def test_presentation_body_client_js(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """The client script resolves file paths from elements, not only the URL."""
    client, _ = view_client
    response = await client.get("/assets/js/view-page.js")
    assert response.status_code == 200
    js = response.text
    assert "function _filePath(el)" in js
    assert 'location.pathname.startsWith("/view/presentation/")' in js
    assert 'el.closest(".vp-presentation-body")' in js
    assert "section.dataset.dlHref" in js
    assert "The message content is no longer available" in js
    assert "function _renderPresentationBody(d, path, dlHref)" in js
