"""Tests for the message render endpoint."""

from __future__ import annotations

import httpx

_KB = 1024
_MB = _KB * _KB
_NO_PREVIEW = "No preview available for this content."


async def test_render_message_json(api_client: httpx.AsyncClient) -> None:
    body = b'{"answer": "Paris"}'
    res = await api_client.post("/api/render/message", content=body)
    assert res.status_code == 200
    data = res.json()
    assert data["kind"] == "json"
    assert data["size"] == len(body)
    assert "Paris" in data["fragment"]


async def test_render_message_markdown(api_client: httpx.AsyncClient) -> None:
    body = b"# Heading\n\nSome text."
    res = await api_client.post("/api/render/message", content=body)
    assert res.status_code == 200
    data = res.json()
    assert data["kind"] == "markdown"
    assert "<h1>" in data["fragment"]


async def test_render_message_over_cap(api_client: httpx.AsyncClient) -> None:
    body = b"x" * (_MB + 1)
    res = await api_client.post("/api/render/message", content=body)
    assert res.status_code == 200
    data = res.json()
    assert data["fragment"] is None
    assert "Too large" in data["note"]


async def test_render_message_too_large_rejected(api_client: httpx.AsyncClient) -> None:
    body = b"x" * (4 * _MB + 1)
    res = await api_client.post("/api/render/message", content=body)
    assert res.status_code == 400
    assert res.json()["code"] == "bad_param"


async def test_render_message_binary_gets_note(api_client: httpx.AsyncClient) -> None:
    body = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n"
    res = await api_client.post("/api/render/message", content=body)
    assert res.status_code == 200
    data = res.json()
    assert data["kind"] == "pdf"
    assert data["fragment"] is None
    assert data["note"] == _NO_PREVIEW
