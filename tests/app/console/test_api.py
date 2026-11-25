"""Tests for console API routing, middleware, health, and VNC status endpoints."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport
from starlette.applications import Starlette
from starlette.routing import Mount

import app.console.api as api_module
import app.lib.db.store as store_module
from app.console.api import ConsoleSecurityMiddleware, build_api_routes
from app.mcp.registry import registered_tools
from tests.app.conftest import require_novnc

def test_api_py_carries_no_tool_description_map() -> None:
    """The api files hold no copy of any tool description."""
    console_dir = Path(api_module.STATIC_DIR).parent
    combined_source = "\n".join(
        (console_dir / file_name).read_text()
        for file_name in ("api.py", "tasks_api.py")
    )
    assert "_TOOL_DESCRIPTIONS" not in combined_source
    assert all(cls.description not in combined_source for cls in registered_tools().values())


async def test_unknown_api_path_returns_json_404(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/api/definitely-not-real")
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "not_found"
    assert "error" in body


async def test_api_response_has_security_headers(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/api/definitely-not-real")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "same-origin"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]


async def test_health_ok(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "1.0.0"
    assert body["dav_base_url"]


async def test_health_503_when_store_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    store_module.init_store(str(tmp_path / "data" / "records.db"))
    await store_module.task_store.close()
    app = Starlette(routes=[Mount("/api", build_api_routes())])
    app.add_middleware(ConsoleSecurityMiddleware)
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["code"] == "internal"


async def test_health_wrong_method_returns_405(api_client: httpx.AsyncClient) -> None:
    """A method an endpoint does not take answers 405.

    The reply names the allowed methods and carries the security headers.
    """
    response = await api_client.post("/api/health")
    assert response.status_code == 405
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["allow"] == "GET, HEAD"
    assert response.json() == {"error": "Method 'POST' is not allowed at '/api/health'.", "code": "method_not_allowed"}
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "same-origin"
    assert "default-src 'self'" in response.headers["content-security-policy"]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/health"),
        ("OPTIONS", "/api/health"),
        ("PUT", "/api/tools"),
        ("PATCH", "/api/tools/abc/tasks"),
        ("DELETE", "/api/tasks/abc"),
        ("POST", "/api/tasks/abc/log"),
        ("POST", "/api/tasks/abc/result"),
        ("DELETE", "/api/tasks/abc/log/download"),
        ("DELETE", "/api/tasks/abc/result/download"),
        ("DELETE", "/api/resources"),
        ("POST", "/api/resources/search"),
        ("DELETE", "/api/changes"),
        ("POST", "/api/vnc/status"),
    ],
)
async def test_wrong_method_on_known_endpoint_returns_405(
    api_client: httpx.AsyncClient, method: str, path: str
) -> None:
    """Every known endpoint answers a method it does not take with 405 and allows GET and HEAD."""
    response = await api_client.request(method, path)
    assert response.status_code == 405
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["allow"] == "GET, HEAD"
    assert response.json() == {"error": f"Method '{method}' is not allowed at '{path}'.", "code": "method_not_allowed"}


async def test_unknown_api_path_wrong_method_stays_404(api_client: httpx.AsyncClient) -> None:
    """An unknown path answers 404, not 405."""
    response = await api_client.post("/api/definitely-not-real")
    assert response.status_code == 404
    assert response.json() == {"error": "No API endpoint at '/api/definitely-not-real'.", "code": "not_found"}


@pytest.mark.parametrize("method", ["OPTIONS", "TRACE", "PUT"])
async def test_unknown_api_path_any_method_returns_404(api_client: httpx.AsyncClient, method: str) -> None:
    """An unknown path answers 404 for every method, with no Allow header."""
    response = await api_client.request(method, "/api/nope")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"error": "No API endpoint at '/api/nope'.", "code": "not_found"}
    assert "allow" not in response.headers


@pytest.mark.parametrize("newline", ["%0a", "%0d", "%0d%0a"])
async def test_encoded_newline_in_api_path_returns_json_404(api_client: httpx.AsyncClient, newline: str) -> None:
    """A path with an encoded newline is cleaned up and answers the JSON 404 with security headers."""
    response = await api_client.get(f"/api/tasks/a{newline}B/result/download")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"error": "No task found with id 'aB'.", "code": "not_found"}
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in response.headers["content-security-policy"]
    clean = await api_client.get("/api/tasks/a%2fb/result/download")
    assert clean.status_code == 404
    assert clean.json()["code"] == "not_found"


async def test_vnc_status(api_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def reachable() -> bool:
        return True

    monkeypatch.setattr(api_module, "vnc_reachable", reachable)
    response = await api_client.get("/api/vnc/status")
    assert response.status_code == 200
    assert response.json() == {"reachable": True}

    async def boom() -> bool:
        raise RuntimeError("probe failed")

    monkeypatch.setattr(api_module, "vnc_reachable", boom)
    response = await api_client.get("/api/vnc/status")
    assert response.status_code == 500
    assert response.json()["code"] == "internal"


@pytest.mark.live
@pytest.mark.timeout(30)
async def test_preview_vnc_status(api_client: httpx.AsyncClient) -> None:
    """The real screen viewer probe reports it is reachable."""
    require_novnc()
    response = await api_client.get("/api/vnc/status")
    assert response.status_code == 200
    assert response.json() == {"reachable": True}


@pytest.mark.timeout(30)
async def test_console_stub_fake_call_starts_a_running_task(console_stub: SimpleNamespace) -> None:
    """A tool call on the stub origin records a running task without running the tool."""
    async with httpx.AsyncClient(base_url=console_stub.base_url, timeout=10) as client:
        response = await client.post("/api/tools/rednote_research/call", json={"query": "e2e"})
        assert response.status_code == 202
        assert response.json()["status"] == "running"
        task_id = response.json()["task_id"]
        assert console_stub.last_call_id() == task_id
        detail = (await client.get("/api/tasks/" + task_id)).json()
        assert detail["task"]["status"] == "running"
        assert detail["task"]["params"] == {"query": "e2e"}
        running = (await client.get("/api/changes")).json()["running"]
        assert [task["task_id"] for task in running] == [task_id]
        response = await client.post("/api/tools/ghost-tool/call", json={})
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"
        response = await client.post(
            "/api/tools/rednote_research/call",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "bad_param"


@pytest.mark.timeout(30)
async def test_console_stub_force_status_knob(console_stub: SimpleNamespace) -> None:
    """A forced status answers the matching endpoint, and leaves the rest untouched."""
    console_stub.force_status(r"/api/changes", 500, code="stub_down", message="stub outage")
    async with httpx.AsyncClient(base_url=console_stub.base_url, timeout=10) as client:
        response = await client.get("/api/changes")
        assert response.status_code == 500
        assert response.json() == {"error": "stub outage", "code": "stub_down"}
        assert (await client.get("/api/tools")).status_code == 200


@pytest.mark.timeout(30)
async def test_console_stub_pin_body_knob(console_stub: SimpleNamespace) -> None:
    """A pinned body answers the matching endpoint with the fixed payload."""
    body = {"status": "ok", "version": "stub", "dav_base_url": console_stub.base_url}
    console_stub.pin_body(r"/api/health", body)
    async with httpx.AsyncClient(base_url=console_stub.base_url, timeout=10) as client:
        assert (await client.get("/api/health")).json() == body
        assert (await client.get("/api/tools")).status_code == 200


@pytest.mark.timeout(30)
async def test_console_stub_delay_knob(console_stub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """A delay waits before the first matching reply only when first_only is set."""
    delay_seconds = 0.4
    held: list[float] = []
    real_sleep = asyncio.sleep

    async def spy_sleep(seconds: float, result: None = None) -> None:
        if seconds == delay_seconds:
            held.append(seconds)
        return await real_sleep(seconds, result)

    monkeypatch.setattr(asyncio, "sleep", spy_sleep)
    console_stub.delay(r"/api/tools", delay_seconds, first_only=True)
    async with httpx.AsyncClient(base_url=console_stub.base_url, timeout=10) as client:
        started_at = time.monotonic()
        assert (await client.get("/api/tools")).status_code == 200
        elapsed = time.monotonic() - started_at
        assert held == [delay_seconds]
        assert (await client.get("/api/tools")).status_code == 200
        assert held == [delay_seconds]
    assert elapsed >= delay_seconds


@pytest.mark.timeout(30)
async def test_console_stub_seed_and_finish_drive_changes(console_stub: SimpleNamespace) -> None:
    """Seeding and finishing tasks change what the changes endpoint reports, and a match is a 304."""
    await console_stub.seed_task(
        "stub-a1",
        "rednote_research",
        "2026-08-23T01:00:00+00:00",
        "completed",
        ended_at="2026-08-23T01:00:05+00:00",
        params={"query": "one"},
    )
    await console_stub.seed_task(
        "stub-r1",
        "rednote_research",
        "2026-08-23T03:00:00+00:00",
        params={"query": "live"},
    )
    async with httpx.AsyncClient(base_url=console_stub.base_url, timeout=10) as client:
        first = (await client.get("/api/changes")).json()
        assert [task["task_id"] for task in first["running"]] == ["stub-r1"]
        fingerprint = first["fingerprint"]
        response = await client.get(
            "/api/changes",
            params={"since_fp": fingerprint, "since_gen": first["generated_at"]},
        )
        assert response.status_code == 304
        await console_stub.finish("stub-r1", "completed", ended_at="2030-01-01T00:00:00+00:00")
        response = await client.get(
            "/api/changes",
            params={"since_fp": fingerprint, "since_gen": first["generated_at"]},
        )
        assert response.status_code == 200
        assert response.json()["running"] == []
        assert [task["task_id"] for task in response.json()["finished"]] == ["stub-r1"]
