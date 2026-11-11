"""Shared setup and helpers for the console api tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport
from starlette.applications import Starlette
from starlette.routing import Mount

import app.lib.db.store as store_module
from app.console.api import ConsolePathSanitizer, ConsoleSecurityMiddleware, build_api_routes
from app.lib.db.store import TaskRecord, TaskView
from tests.app.conftest import point_share_env

POLL_SLEEP_S = 0.01


@asynccontextmanager
async def console_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[httpx.AsyncClient, Path]]:
    """Set up the console api for a test and give back the client and the share folder."""
    share, data = point_share_env(tmp_path, monkeypatch)
    store_module.init_store(str(data / "records.db"))
    await store_module.task_store.open()
    app = Starlette(routes=[Mount("/api", build_api_routes())])
    app.add_middleware(ConsoleSecurityMiddleware)
    app.add_middleware(ConsolePathSanitizer)
    client = httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t")
    try:
        yield client, share
    finally:
        await client.aclose()
        await store_module.task_store.close()


@pytest.fixture
async def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[httpx.AsyncClient]:
    """A test client for the console api."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    store_module.init_store(str(tmp_path / "data" / "records.db"))
    await store_module.task_store.open()
    try:
        app = Starlette(routes=[Mount("/api", build_api_routes())])
        app.add_middleware(ConsoleSecurityMiddleware)
        app.add_middleware(ConsolePathSanitizer)
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        await store_module.task_store.close()


async def wait_record(task_id: str, status: str) -> TaskRecord:
    """Wait for the task to reach the given status, then give back its record."""
    for _ in range(500):
        record = await store_module.task_store.get(task_id)
        if record is not None and record.status == status:
            return record
        await asyncio.sleep(POLL_SLEEP_S)
    raise AssertionError(f"task {task_id} did not reach status {status!r}")


async def wait_rows(min_count: int) -> int:
    """Wait until the store holds at least min_count rows, then give back the count."""
    for _ in range(300):
        total = await store_module.total_count()
        if total >= min_count:
            return total
        await asyncio.sleep(POLL_SLEEP_S)
    raise AssertionError(f"expected at least {min_count} row(s)")


async def wait_subtasks(parent_id: str) -> list[TaskView]:
    """Wait until subtask rows show up under the parent, then give them back."""
    for _ in range(500):
        kids = await store_module.subtask_views(parent_id)
        if kids:
            return kids
        await asyncio.sleep(POLL_SLEEP_S)
    raise AssertionError(f"no subtask rows appeared under {parent_id}")


