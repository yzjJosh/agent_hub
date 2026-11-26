"""Shared fixtures and file helpers for viewer page tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

import app.lib.render.cache as cache
from app.server import build_app
from tests.app.conftest import configure_resource_env, store_infra


@pytest.fixture
async def view_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[httpx.AsyncClient, Path]]:
    """Build an async HTTP client for the app and its resources directory."""
    resources_dir, data_dir = configure_resource_env(tmp_path, monkeypatch)
    async for _ in store_infra(data_dir):
        app = build_app()
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            yield client, resources_dir


def write_resource_file(resources_dir: Path, relative_path: str, content: str | bytes = b"") -> str:
    """Write text or bytes into the test resources directory and return its relative path."""
    target_path = resources_dir / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode() if isinstance(content, str) else content
    target_path.write_bytes(payload)
    return relative_path


def compute_cache_key(resources_dir: Path, relative_path: str) -> str:
    """Compute the render cache key for a file in the test resources directory."""
    file_stat = (resources_dir / relative_path).stat()
    return cache.build_cache_key(relative_path, int(file_stat.st_mtime))
