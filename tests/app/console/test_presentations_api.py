"""Tests for the console presentations listing endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

import app.lib.db.store as store_module
from tests.app.conftest import write_resource_file
from tests.app.console.conftest import console_env


async def test_presentations_empty(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/api/presentations")
    assert response.status_code == 200
    body = response.json()
    assert body["presentations"] == []
    assert "generated_at" in body
    assert "generated_at_local" in body


async def test_presentations_returns_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with console_env(tmp_path, monkeypatch) as (client, resources):
        write_resource_file(resources, "upload/a.png", b"\x89PNG fake")
        await store_module.presentation_store.insert(
            "a" * 16,
            (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            "/view/presentation/aaaaaaaaaaaaaaaa",
            ["upload/a.png"],
        )
        response = await client.get("/api/presentations")
        assert response.status_code == 200
        body = response.json()
        assert len(body["presentations"]) == 1
        presentation = body["presentations"][0]
        assert presentation["id"] == "a" * 16
        assert presentation["state"] == "active"
        assert presentation["file_count"] == 1
        assert presentation["present_count"] == 1
        assert presentation["kind"] == "solo"
        assert presentation["files"][0]["present"] is True
        assert presentation["url"].startswith("http://")
        assert presentation["url"].endswith("/view/presentation/aaaaaaaaaaaaaaaa")


async def test_presentations_title_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with console_env(tmp_path, monkeypatch) as (client, resources):
        write_resource_file(resources, "upload/a.png", b"\x89PNG fake")
        await store_module.presentation_store.insert(
            "a" * 16,
            (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
            "/view/presentation/aaaaaaaaaaaaaaaa",
            ["upload/a.png"],
            title="Weekly report",
        )
        await store_module.presentation_store.insert(
            "b" * 16,
            (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            "/view/presentation/bbbbbbbbbbbbbbbb",
            ["upload/a.png"],
        )
        response = await client.get("/api/presentations")
        body = response.json()
        titles = {item["id"]: item["title"] for item in body["presentations"]}
        assert titles == {"a" * 16: "Weekly report", "b" * 16: None}


async def test_presentations_partial_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with console_env(tmp_path, monkeypatch) as (client, resources):
        write_resource_file(resources, "upload/a.png", b"\x89PNG fake")
        await store_module.presentation_store.insert(
            "b" * 16,
            (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            "/view/presentation/bbbbbbbbbbbbbbbb",
            ["upload/a.png", "upload/gone.png"],
        )
        response = await client.get("/api/presentations")
        body = response.json()
        presentation = body["presentations"][0]
        assert presentation["state"] == "partial"
        assert presentation["present_count"] == 1
        assert presentation["file_count"] == 2
        assert presentation["kind"] == "package"


async def test_presentations_30day_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with console_env(tmp_path, monkeypatch) as (client, _resources):
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        await store_module.presentation_store.insert(
            "c" * 16,
            old_timestamp,
            "/view/presentation/cccccccccccccccc",
            ["upload/a.png"],
        )
        response = await client.get("/api/presentations")
        body = response.json()
        assert body["presentations"] == []


async def test_presentations_method_guard(api_client: httpx.AsyncClient) -> None:
    response = await api_client.post("/api/presentations")
    assert response.status_code == 405
