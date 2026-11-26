"""Tests for the cached document page and sibling asset routes."""

from __future__ import annotations

import os
from pathlib import Path

import httpx

import app.lib.render.cache as cache
from tests.app.viewer.conftest import compute_cache_key, write_resource_file


async def test_doc_page_composed_and_thumb_override(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A cached document renders its HTML and applies the thumbnail layout override when requested."""
    client, resources_dir = view_client
    write_resource_file(resources_dir, "upload/n.md", "# composed")
    cache_key = compute_cache_key(resources_dir, "upload/n.md")
    await cache.write_doc(
        cache_key,
        '<div class="md-doc"><h1>composed</h1></div>',
        None,
        {"mode": "doc", "provenance": None},
    )
    response = await client.get(f"/view/doc/{cache_key}")
    assert response.status_code == 200
    assert "composed" in response.text
    assert "thumb" not in response.text
    thumbnail_response = await client.get(f"/view/doc/{cache_key}?thumb=1")
    assert thumbnail_response.status_code == 200
    assert "vp-thumb-override" in thumbnail_response.text


async def test_doc_missing_key_is_plain_404(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A missing cached document key returns a plain text 404 response."""
    client, _ = view_client
    response = await client.get("/view/doc/" + "0" * 64)
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/plain")


async def test_doc_non_hex_key_delegate_to_file(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A non-hex document route delegates to the matching resource file."""
    client, resources_dir = view_client
    write_resource_file(resources_dir, "doc/reports/q1", "# delegated doc")
    response = await client.get("/view/doc/reports/q1")
    assert response.status_code == 200
    assert "delegated doc" in response.text
    assert "doc/reports/q1" in response.text


async def test_doc_asset_serves_sibling(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A document sibling asset returns its bytes with the guessed media type."""
    client, _ = view_client
    cache_key = "0" * 64
    key_directory = os.path.join(cache.render_root(), cache_key)
    os.makedirs(key_directory, exist_ok=True)
    with open(os.path.join(key_directory, "img.png"), "wb") as asset_file:
        asset_file.write(b"\x89PNG asset")
    response = await client.get(f"/view/doc/{cache_key}/img.png")
    assert response.status_code == 200
    assert response.content == b"\x89PNG asset"
    assert "image/png" in response.headers["content-type"]


async def test_doc_asset_traversal_rejected(view_client: tuple[httpx.AsyncClient, Path]) -> None:
    """A traversal attempt in a document asset path returns 404."""
    client, _ = view_client
    cache_key = "0" * 64
    response = await client.get(f"/view/doc/{cache_key}/..%2f..%2fsecret")
    assert response.status_code == 404
