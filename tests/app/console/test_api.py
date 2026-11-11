from __future__ import annotations

import asyncio
import io
import os
import re
import time
from collections.abc import Coroutine
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport
from starlette.applications import Starlette
from starlette.responses import FileResponse
from starlette.routing import Mount

import app.console.api as api_module
import app.lib.db.store as store_module
from app.lib.render import cache
from app.lib.shared import activity
from app.console.api import ConsoleSecurityMiddleware, build_api_routes, dav_download
from app.lib.task import task_manager
from app.mcp.registry import registered_tools, tool_class
from tests.app.conftest import (
    E2E_ENDED_AT,
    E2E_PNG,
    E2E_RESULT_BIG,
    E2E_RUNNING,
    E2E_STARTED_AT,
    log_lines,
    require_novnc,
    seed_task,
    write_run_file,
    write_share_file,
)
from tests.app.console.conftest import console_env, wait_record

_API_SOURCE = (Path(api_module.STATIC_DIR).parent / "api.py").read_text()
_RESULT_BYTES_0797 = 793191


def test_api_py_carries_no_tool_description_map() -> None:
    """The api file holds no copy of any tool description."""
    assert "_TOOL_DESCRIPTIONS" not in _API_SOURCE
    assert all(cls.description not in _API_SOURCE for cls in registered_tools().values())


async def _seed_brave(
    task_id: str, started_at: str, q: int, status: str | None, *, error: str | None = None, ended_at: str | None = None
) -> None:
    await store_module.task_store.insert_running(task_id, None, started_at, f"runs/{task_id}")
    await store_module.tool_store.insert(task_id, "brave_search", None, {"q": q})
    if status is not None:
        await store_module.task_store.finish(task_id, status, error=error, ended_at=ended_at)


async def test_unknown_api_path_returns_json_404(api_client: httpx.AsyncClient) -> None:
    r = await api_client.get("/api/definitely-not-real")
    assert r.status_code == 404
    body = r.json()
    assert body["code"] == "not_found"
    assert "error" in body


async def test_api_response_has_security_headers(api_client: httpx.AsyncClient) -> None:
    r = await api_client.get("/api/definitely-not-real")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "same-origin"
    assert r.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in r.headers["content-security-policy"]


async def test_health_ok(api_client: httpx.AsyncClient) -> None:
    r = await api_client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "1.0.0"
    assert body["dav_base_url"]


async def test_health_503_when_store_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    store_module.init_store(str(tmp_path / "data" / "records.db"))
    await store_module.task_store.close()
    app = Starlette(routes=[Mount("/api", build_api_routes())])
    app.add_middleware(ConsoleSecurityMiddleware)
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/health")
    assert r.status_code == 503
    assert r.json()["code"] == "internal"


async def test_health_wrong_method_returns_405(api_client: httpx.AsyncClient) -> None:
    """A method an endpoint does not take answers 405.

    The reply names the allowed methods and carries the security headers.
    """
    r = await api_client.post("/api/health")
    assert r.status_code == 405
    assert r.headers["content-type"].startswith("application/json")
    assert r.headers["allow"] == "GET, HEAD"
    assert r.json() == {"error": "Method 'POST' is not allowed at '/api/health'.", "code": "method_not_allowed"}
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "same-origin"
    assert "default-src 'self'" in r.headers["content-security-policy"]


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
    r = await api_client.request(method, path)
    assert r.status_code == 405
    assert r.headers["content-type"].startswith("application/json")
    assert r.headers["allow"] == "GET, HEAD"
    assert r.json() == {"error": f"Method '{method}' is not allowed at '{path}'.", "code": "method_not_allowed"}


async def test_unknown_api_path_wrong_method_stays_404(api_client: httpx.AsyncClient) -> None:
    """An unknown path answers 404, not 405."""
    r = await api_client.post("/api/definitely-not-real")
    assert r.status_code == 404
    assert r.json() == {"error": "No API endpoint at '/api/definitely-not-real'.", "code": "not_found"}


@pytest.mark.parametrize("method", ["OPTIONS", "TRACE", "PUT"])
async def test_unknown_api_path_any_method_returns_404(api_client: httpx.AsyncClient, method: str) -> None:
    """An unknown path answers 404 for every method, with no Allow header."""
    r = await api_client.request(method, "/api/nope")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == {"error": "No API endpoint at '/api/nope'.", "code": "not_found"}
    assert "allow" not in r.headers


@pytest.mark.parametrize("newline", ["%0a", "%0d", "%0d%0a"])
async def test_encoded_newline_in_api_path_returns_json_404(api_client: httpx.AsyncClient, newline: str) -> None:
    """A path with an encoded newline is cleaned up and answers the JSON 404 with security headers."""
    r = await api_client.get(f"/api/tasks/a{newline}B/result/download")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == {"error": "No task found with id 'aB'.", "code": "not_found"}
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in r.headers["content-security-policy"]
    clean = await api_client.get("/api/tasks/a%2fb/result/download")
    assert clean.status_code == 404
    assert clean.json()["code"] == "not_found"


@pytest.fixture
def dav_download_app() -> Starlette:
    """A small app with only the download route."""
    app = Starlette()
    app.add_route("/dav/dl", dav_download, methods=("GET", "HEAD"))
    return app


def _write(share: Path, rel: str, content: str) -> Path:
    p = share / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


@pytest.mark.parametrize(
    ("rel", "content_disposition"),
    [
        ("upload/x.txt", 'attachment; filename="x.txt"'),
        ("sub dir/my file.txt", "attachment; filename*=utf-8''my%20file.txt"),
    ],
)
async def test_dav_download_serves_file_as_attachment(
    rel: str, content_disposition: str, dav_share: Path, dav_download_app: Starlette
) -> None:
    """A download gives back the file bytes as an attachment named after the file."""
    _write(dav_share, rel, "hello download")
    async with httpx.AsyncClient(transport=ASGITransport(app=dav_download_app), base_url="http://t") as c:
        r = await c.get("/dav/dl", params={"path": rel})
    assert r.status_code == 200
    assert r.text == "hello download"
    assert r.headers["content-disposition"] == content_disposition


async def test_dav_download_strips_leading_slash(dav_share: Path, dav_download_app: Starlette) -> None:
    """A leading slash on the path is ignored."""
    _write(dav_share, "upload/x.txt", "hello download")
    async with httpx.AsyncClient(transport=ASGITransport(app=dav_download_app), base_url="http://t") as c:
        r = await c.get("/dav/dl", params={"path": "/upload/x.txt"})
    assert r.status_code == 200
    assert r.text == "hello download"


async def test_dav_download_missing_param_is_400(dav_download_app: Starlette) -> None:
    """A missing or empty path answers 400 bad_param, never a folder listing."""
    async with httpx.AsyncClient(transport=ASGITransport(app=dav_download_app), base_url="http://t") as c:
        r_absent = await c.get("/dav/dl")
        r_empty = await c.get("/dav/dl", params={"path": ""})
    assert r_absent.status_code == 400
    assert r_absent.json()["code"] == "bad_param"
    assert r_empty.status_code == 400
    assert r_empty.json()["code"] == "bad_param"


@pytest.mark.parametrize("path", ["upload/missing.txt", "../../etc/passwd", "..", "a/../../b"])
async def test_dav_download_missing_or_traversal_is_404(
    path: str, dav_share: Path, dav_download_app: Starlette
) -> None:
    """A missing file and every path that climbs out of the share answer 404."""
    _write(dav_share, "upload/real.txt", "control")
    async with httpx.AsyncClient(transport=ASGITransport(app=dav_download_app), base_url="http://t") as c:
        control = await c.get("/dav/dl", params={"path": "upload/real.txt"})
        assert control.status_code == 200  # a real file downloads, so the 404s below are real misses
        r = await c.get("/dav/dl", params={"path": path})
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


@pytest.mark.parametrize(("method", "is_recorded"), [("GET", True), ("HEAD", False)])
async def test_dav_download_records_interest_on_get(
    method: str, is_recorded: bool, dav_share: Path, dav_download_app: Starlette
) -> None:
    """A download renews retention, a HEAD request does not."""
    path = _write(dav_share, "upload/old.txt", "old")
    old = time.time() - 40 * 86400
    os.utime(path, (old, old))
    async with httpx.AsyncClient(transport=ASGITransport(app=dav_download_app), base_url="http://t") as c:
        r = await c.request(method, "/dav/dl", params={"path": "upload/old.txt"})
    assert r.status_code == 200
    assert (activity.last_used_sync(str(path), path.stat()) > old) is is_recorded


async def test_tool_tasks_keyset(api_client: httpx.AsyncClient) -> None:
    """Local times are exact strings in the container time zone, in summer and in winter."""
    await _seed_brave("a", "2026-08-23T01:00:00+00:00", 1, "completed", ended_at="2026-08-23T01:00:05+00:00")
    await _seed_brave("b", "2026-08-23T02:00:00+00:00", 2, "failed", error="boom", ended_at="2026-08-23T02:00:05+00:00")
    await _seed_brave("c", "2026-08-23T03:00:00+00:00", 3, None)
    await _seed_brave("d", "2026-01-15T08:30:00+00:00", 4, "cancelled", ended_at="2026-01-15T08:30:05+00:00")

    r = await api_client.get("/api/tools/brave_search/tasks", params={"limit": 2})
    assert r.status_code == 200
    body = r.json()
    assert [t["task_id"] for t in body["tasks"]] == ["c", "b"]
    assert body["next_cursor"] == "2026-08-23T02:00:00+00:00|b"
    assert body["tasks"][0]["tool"] == "brave_search"
    assert body["tasks"][0]["params"] == {"q": 3}
    assert body["tasks"][0]["started_at"] == "2026-08-23T03:00:00+00:00"
    assert body["tasks"][0]["started_at_local"] == "2026-08-22 20:00:00"
    assert body["tasks"][1]["status"] == "failed"
    assert body["tasks"][1]["ended_at"] == "2026-08-23T02:00:05+00:00"
    assert body["tasks"][1]["ended_at_local"] == "2026-08-22 19:00:05"
    r2 = await api_client.get(
        "/api/tools/brave_search/tasks", params={"limit": 2, "cursor": body["next_cursor"]}
    )
    assert [t["task_id"] for t in r2.json()["tasks"]] == ["a", "d"]
    assert r2.json()["next_cursor"] is None
    assert {t["status"] for t in r2.json()["tasks"]} == {"completed", "cancelled"}
    assert r2.json()["tasks"][1]["ended_at"] == "2026-01-15T08:30:05+00:00"
    assert r2.json()["tasks"][1]["ended_at_local"] == "2026-01-15 00:30:05"


async def test_tool_tasks_unknown_tool(api_client: httpx.AsyncClient) -> None:
    r = await api_client.get("/api/tools/does_not_exist/tasks")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


async def test_tool_tasks_bad_limit(api_client: httpx.AsyncClient) -> None:
    r = await api_client.get("/api/tools/brave_search/tasks", params={"limit": 9999})
    assert r.status_code == 400
    assert r.json()["code"] == "bad_param"


async def test_tool_tasks_corrupt_cursor(api_client: httpx.AsyncClient) -> None:
    r = await api_client.get("/api/tools/brave_search/tasks", params={"cursor": "garbage-no-pipe"})
    assert r.status_code == 400
    body = r.json()
    assert set(body) == {"error", "code"}
    assert body["code"] == "bad_param"
    r = await api_client.get("/api/tools/brave_search/tasks", params={"cursor": ""})
    assert r.status_code == 400
    assert r.json()["code"] == "bad_param"


@pytest.mark.parametrize("name", ["brave_search", "github"])
async def test_tool_tasks_description_is_summary(api_client: httpx.AsyncClient, name: str) -> None:
    r = await api_client.get(f"/api/tools/{name}/tasks")
    assert r.status_code == 200
    assert r.json()["description"] == tool_class(name).summary


async def test_tool_tasks_description_not_exposed(api_client: httpx.AsyncClient) -> None:
    """A tool the console does not show still has its summary in the task list."""
    r = await api_client.get("/api/tools/task/tasks")
    assert r.status_code == 200
    assert r.json()["description"] == tool_class("task").summary


async def test_tools_roster_and_stats(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("a", None, "2026-08-23T01:00:00+00:00", "runs/a")
    await store_module.tool_store.insert("a", "brave_search", None, {"q": 1})
    await store_module.task_store.finish("a", "completed", ended_at="2026-08-23T01:00:05+00:00")
    await store_module.task_store.insert_running("b", None, "2026-08-23T02:00:00+00:00", "runs/b")
    await store_module.tool_store.insert("b", "brave_search", None, {"q": 2})
    await store_module.task_store.finish("b", "completed", ended_at="2026-08-23T02:00:05+00:00")
    await store_module.task_store.insert_running("c", None, "2026-08-23T03:00:00+00:00", "runs/c")
    await store_module.tool_store.insert("c", "brave_search", None, {"q": 3})

    r = await api_client.get("/api/tools")
    assert r.status_code == 200
    body = r.json()
    assert body["generated_at"].endswith("+00:00")
    assert "generated_at_local" in body
    by_name = {t["name"]: t for t in body["tools"]}
    # only tracked tools are listed
    tracked = {name for name, cls in registered_tools().items() if cls.tracked is True}
    assert set(by_name) == tracked
    bs = by_name["brave_search"]
    assert bs["description"] == tool_class("brave_search").summary
    assert (bs["completed"], bs["running"], bs["failed"], bs["cancelled"]) == (2, 1, 0, 0)
    assert bs["last_activity"] == "2026-08-23T03:00:00+00:00"
    assert bs["running_since"] == "2026-08-23T03:00:00+00:00"
    assert "task" not in by_name
    assert "send_or_receive_file" not in by_name
    assert "present_to_user" not in by_name


async def test_not_exposed_tools_absent_from_roster(api_client: httpx.AsyncClient) -> None:
    """The list holds the 16 tracked tools and leaves out task, send_or_receive_file, present_to_user and resource_to_image."""
    tracked = {name for name, cls in registered_tools().items() if cls.tracked is True}
    assert len(tracked) == 16  # 20 tools are registered and 4 are left out
    names = {t["name"] for t in (await api_client.get("/api/tools")).json()["tools"]}
    assert names == tracked
    assert (
        "task" not in names
        and "send_or_receive_file" not in names
        and "present_to_user" not in names
        and "resource_to_image" not in names
    )


async def test_task_detail_with_subtask(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("parent", None, "2026-08-23T01:00:00+00:00", "runs/parent")
    await store_module.tool_store.insert("parent", "brave_search", None, {"q": 1})
    await store_module.task_store.insert_running("child", "parent", "2026-08-23T01:00:01+00:00", "runs/child")
    await store_module.tool_store.insert("child", "web_fetch", None, {"url": "x"})
    data_dir = store_module.task_store.data_dir
    os.makedirs(os.path.join(data_dir, "runs/parent"), exist_ok=True)
    with open(os.path.join(data_dir, "runs/parent/log"), "w", encoding="utf-8") as fh:
        fh.write("hello\n")

    r = await api_client.get("/api/tasks/parent")
    assert r.status_code == 200
    body = r.json()
    assert body["task"]["task_id"] == "parent"
    assert body["task"]["tool"] == "brave_search"
    assert body["task"]["params"] == {"q": 1}
    assert "subtasks" not in body["task"]
    assert body["log_size"] == 6
    assert body["result_size"] is None
    assert len(body["subtasks"]) == 1
    assert body["subtasks"][0]["task_id"] == "child"
    assert body["subtasks"][0]["tool"] == "web_fetch"
    assert "subtasks" not in body["subtasks"][0]


async def test_task_detail_not_found(api_client: httpx.AsyncClient) -> None:
    r = await api_client.get("/api/tasks/does-not-exist")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


async def test_log_forward_and_backward(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("t1", None, "2026-08-23T01:00:00+00:00", "runs/t1")
    await store_module.tool_store.insert("t1", "brave_search", None, {"q": 1})
    log_dir = os.path.join(store_module.task_store.data_dir, "runs/t1")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "log"), "w", encoding="utf-8") as fh:
        fh.write("l1\nl2\nl3\n")
    b = (await api_client.get("/api/tasks/t1/log", params={"lines": 200})).json()
    assert b["content"] == "l1\nl2\nl3\n" and b["total_lines"] == 3 and b["offset"] == 0
    assert b["head_reached"] is True
    f = (await api_client.get("/api/tasks/t1/log", params={"offset": 0})).json()
    assert f["content"] == "l1\nl2\nl3\n" and f["content_bytes"] == 9 and f["size"] == 9 and f["truncated"] is False


async def test_log_not_found_no_record(api_client: httpx.AsyncClient) -> None:
    r = await api_client.get("/api/tasks/nope/log")
    assert r.status_code == 404 and r.json()["code"] == "not_found"


async def test_log_not_found_missing_file(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("t2", None, "2026-08-23T01:00:00+00:00", "runs/t2")
    await store_module.tool_store.insert("t2", "brave_search", None, {"q": 1})
    r = await api_client.get("/api/tasks/t2/log")
    assert r.status_code == 404 and r.json()["code"] == "log_not_found"


async def test_result_window(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("t1", None, "2026-08-23T01:00:00+00:00", "runs/t1")
    await store_module.tool_store.insert("t1", "brave_search", None, {"q": 1})
    await store_module.task_store.finish("t1", "completed", ended_at="2026-08-23T01:00:05+00:00")
    run_dir = os.path.join(store_module.task_store.data_dir, "runs/t1")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result"), "w", encoding="utf-8") as fh:
        fh.write("RESULT-CONTENT")
    b = (await api_client.get("/api/tasks/t1/result", params={"offset": 0})).json()
    assert b["content"] == "RESULT-CONTENT" and b["content_bytes"] == 14 and b["size"] == 14 and b["truncated"] is False
    b2 = (await api_client.get("/api/tasks/t1/result", params={"offset": 0, "max_bytes": 4})).json()
    assert b2["content"] == "RESU" and b2["truncated"] is True


async def test_result_not_found_when_running(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("t2", None, "2026-08-23T01:00:00+00:00", "runs/t2")
    await store_module.tool_store.insert("t2", "brave_search", None, {"q": 1})
    r = await api_client.get("/api/tasks/t2/result")
    assert r.status_code == 404 and r.json()["code"] == "result_not_found"


async def test_result_not_found_unknown(api_client: httpx.AsyncClient) -> None:
    r = await api_client.get("/api/tasks/does-not-exist/result")
    assert r.status_code == 404 and r.json()["code"] == "not_found"


@pytest.mark.parametrize(
    ("url", "offset"),
    [
        ("/api/tasks/t1/log", "9223372036854775808"),  # 2**63
        ("/api/tasks/t1/log", "999999999999999999999999999999"),  # 30 digits
        ("/api/tasks/t1/result", "9223372036854775808"),  # 2**63
        ("/api/tasks/t1/result", "999999999999999999999999999999"),  # 30 digits
    ],
)
async def test_offset_beyond_bound_returns_bad_param(
    api_client: httpx.AsyncClient, url: str, offset: str
) -> None:
    """An offset of 2**63 or more answers 400 bad_param."""
    r = await api_client.get(url, params={"offset": offset})
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == {
        "error": "offset must be an integer between 0 and 1099511627776 and max_bytes a positive integer.",
        "code": "bad_param",
    }
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"


async def test_offset_int64_danger_zone_returns_bad_param(api_client: httpx.AsyncClient) -> None:
    """An offset just under 2**63 answers 400 bad_param, not a missing file 404."""
    r = await api_client.get("/api/tasks/t1/log", params={"offset": "9223372036854775806"})
    assert r.status_code == 400
    assert r.json()["code"] == "bad_param"


async def test_offset_just_above_bound_returns_bad_param(api_client: httpx.AsyncClient) -> None:
    r = await api_client.get("/api/tasks/t1/log", params={"offset": "1099511627777"})  # 2**40 + 1
    assert r.status_code == 400
    assert r.json()["code"] == "bad_param"


async def test_offset_past_eof_returns_empty_window(api_client: httpx.AsyncClient) -> None:
    """An offset past the end of the file answers 200 with empty content."""
    await _seed_brave("t1", "2026-08-23T01:00:00+00:00", 1, None)
    run_dir = os.path.join(store_module.task_store.data_dir, "runs/t1")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "log"), "w", encoding="utf-8") as fh:
        fh.write("l1\nl2\nl3\n")
    r = await api_client.get("/api/tasks/t1/log", params={"offset": 4294967296})
    assert r.status_code == 200
    body = r.json()
    assert body["offset"] == 4294967296
    assert body["content"] == "" and body["content_bytes"] == 0 and body["truncated"] is False


async def test_end_offset_beyond_bound_returns_bad_param(api_client: httpx.AsyncClient) -> None:
    """A huge end_offset answers 400 bad_param."""
    r = await api_client.get(
        "/api/tasks/t1/log", params={"lines": 5, "end_offset": "999999999999999999999999999999"}
    )
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == {
        "error": "lines must be an integer and end_offset an integer between 0 and 1099511627776.",
        "code": "bad_param",
    }


async def test_log_download(api_client: httpx.AsyncClient) -> None:
    await _seed_brave("t1", "2026-08-23T01:00:00+00:00", 1, None)
    run_dir = os.path.join(store_module.task_store.data_dir, "runs/t1")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "log"), "w", encoding="utf-8") as fh:
        fh.write("LOG-LINES\n")
    r = await api_client.get("/api/tasks/t1/log/download")
    assert r.status_code == 200
    assert r.text == "LOG-LINES\n"
    assert r.headers["content-type"].startswith("text/plain")
    assert "attachment" in r.headers["content-disposition"] and "t1_log" in r.headers["content-disposition"]


async def test_log_download_missing_file(api_client: httpx.AsyncClient) -> None:
    await _seed_brave("t2", "2026-08-23T01:00:00+00:00", 1, None)
    r = await api_client.get("/api/tasks/t2/log/download")
    assert r.status_code == 404 and r.json()["code"] == "log_not_found"


async def test_result_download_range(api_client: httpx.AsyncClient) -> None:
    await _seed_brave("t1", "2026-08-23T01:00:00+00:00", 1, "completed", ended_at="2026-08-23T01:00:05+00:00")
    run_dir = os.path.join(store_module.task_store.data_dir, "runs/t1")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result"), "w", encoding="utf-8") as fh:
        fh.write("0123456789")
    r = await api_client.get("/api/tasks/t1/result/download")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/octet-stream")
    assert "t1_result" in r.headers["content-disposition"]
    r2 = await api_client.get("/api/tasks/t1/result/download", headers={"Range": "bytes=2-5"})
    assert r2.status_code == 206
    assert r2.text == "2345"


@pytest.mark.parametrize(
    ("url", "params", "file_name", "code"),
    [
        ("/api/tasks/t1/log", {"offset": 0}, "log", "log_not_found"),
        ("/api/tasks/t1/log", {"lines": 5, "end_offset": 0}, "log", "log_not_found"),
        ("/api/tasks/t1/result", {"offset": 0}, "result", "result_not_found"),
        ("/api/tasks/t1/log/download", None, "log", "log_not_found"),
        ("/api/tasks/t1/result/download", None, "result", "result_not_found"),
    ],
)
async def test_run_file_vanished_after_exists_check(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    params: dict | None,
    file_name: str,
    code: str,
) -> None:
    """A run file that goes away while it is being read answers the JSON 404, not a 500."""
    await _seed_brave("t1", "2026-08-23T01:00:00+00:00", 1, None)
    run_dir = os.path.join(store_module.task_store.data_dir, "runs/t1")
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, file_name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("data\n")

    real_open = open

    def vanish_open(file: str, *args: str, **kwargs: str) -> io.BinaryIO:
        if file == path:
            os.remove(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(api_module, "open", vanish_open, raising=False)
    r = await api_client.get(url, params=params)
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["code"] == code
    clean = await api_client.get(url, params=params)
    assert (r.status_code, r.json()) == (clean.status_code, clean.json())


async def test_download_passes_prechecked_stat_to_file_response(
    api_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both downloads hand the file size they already read to the response."""
    await _seed_brave("t1", "2026-08-23T01:00:00+00:00", 1, "completed", ended_at="2026-08-23T01:00:05+00:00")
    run_dir = os.path.join(store_module.task_store.data_dir, "runs/t1")
    os.makedirs(run_dir, exist_ok=True)
    log_bytes = b"LOG-LINES\n"
    result_bytes = b"RESULT-CONTENT"
    with open(os.path.join(run_dir, "log"), "wb") as fh:
        fh.write(log_bytes)
    with open(os.path.join(run_dir, "result"), "wb") as fh:
        fh.write(result_bytes)

    stats: list[os.stat_result] = []
    real_file_response = api_module.FileResponse

    def capturing_file_response(*args: str, **kwargs: str | os.stat_result) -> FileResponse:
        stats.append(kwargs["stat_result"])
        return real_file_response(*args, **kwargs)

    monkeypatch.setattr(api_module, "FileResponse", capturing_file_response)

    r = await api_client.get("/api/tasks/t1/log/download")
    assert r.status_code == 200
    assert r.text == log_bytes.decode()
    r = await api_client.get("/api/tasks/t1/result/download")
    assert r.status_code == 200
    assert all(isinstance(st, os.stat_result) for st in stats)
    assert [st.st_size for st in stats] == [len(log_bytes), len(result_bytes)]


async def test_changes_baseline_and_304(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("r1", None, "2026-08-23T01:00:00+00:00", "runs/r1")
    await store_module.tool_store.insert("r1", "brave_search", None, {"q": 1})
    await store_module.task_store.insert_running("d1", None, "2026-08-23T00:00:00+00:00", "runs/d1")
    await store_module.tool_store.insert("d1", "brave_search", None, {"q": 2})
    await store_module.task_store.finish("d1", "completed", ended_at="2026-08-23T00:00:05+00:00")

    body = (await api_client.get("/api/changes")).json()
    assert body["fingerprint"].endswith("|1|")
    assert [t["task_id"] for t in body["running"]] == ["r1"]
    assert body["started"] == [] and body["finished"] == []
    assert body["stats"]["running"] == 1
    assert body["stats"]["tools"]["brave_search"]["completed"] == 1
    assert body["stats"]["tools"]["brave_search"]["running_since"] == "2026-08-23T01:00:00+00:00"
    r2 = await api_client.get(
        "/api/changes", params={"since_fp": body["fingerprint"], "since_gen": body["generated_at"]}
    )
    assert r2.status_code == 304


async def test_changes_started_and_finished(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("x1", None, "2026-08-23T00:00:00+00:00", "runs/x1")
    await store_module.tool_store.insert("x1", "brave_search", None, {"q": 1})
    await store_module.task_store.finish("x1", "completed", ended_at="2026-08-23T00:00:05+00:00")
    body = (await api_client.get(
        "/api/changes", params={"since_fp": "no-match", "since_gen": "2026-08-22T23:00:00+00:00"}
    )).json()
    assert [t["task_id"] for t in body["started"]] == ["x1"]
    assert [t["task_id"] for t in body["finished"]] == ["x1"]
    assert body["running"] == []


async def test_changes_running_to_finished(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("n1", None, "2026-08-23T00:00:00+00:00", "runs/n1")
    await store_module.tool_store.insert("n1", "brave_search", None, {"q": 1})
    body = (await api_client.get(
        "/api/changes", params={"since_fp": "no-match", "since_gen": "2026-08-22T23:00:00+00:00"}
    )).json()
    assert [t["task_id"] for t in body["started"]] == ["n1"]
    assert [t["task_id"] for t in body["running"]] == ["n1"]
    assert body["running"][0]["status"] == "running"
    await store_module.task_store.finish("n1", "completed", ended_at=datetime.now(timezone.utc).isoformat())
    body2 = (await api_client.get(
        "/api/changes", params={"since_fp": body["fingerprint"], "since_gen": body["generated_at"]}
    )).json()
    assert [t["task_id"] for t in body2["finished"]] == ["n1"]
    assert body2["finished"][0]["status"] == "completed"
    assert body2["running"] == []


async def test_presentations_empty(api_client: httpx.AsyncClient) -> None:
    r = await api_client.get("/api/presentations")
    assert r.status_code == 200
    body = r.json()
    assert body["presentations"] == []
    assert "generated_at" in body
    assert "generated_at_local" in body


async def test_presentations_returns_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with console_env(tmp_path, monkeypatch) as (client, share):
        write_share_file(share, "upload/a.png", b"\x89PNG fake")
        await store_module.presentation_store.insert(
            "a" * 16, (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            "/view/presentation/aaaaaaaaaaaaaaaa", ["upload/a.png"],
        )
        r = await client.get("/api/presentations")
        assert r.status_code == 200
        body = r.json()
        assert len(body["presentations"]) == 1
        p = body["presentations"][0]
        assert p["id"] == "a" * 16
        assert p["state"] == "active"
        assert p["file_count"] == 1
        assert p["present_count"] == 1
        assert p["kind"] == "solo"
        assert p["files"][0]["present"] is True
        assert p["url"].startswith("http://")
        assert p["url"].endswith("/view/presentation/aaaaaaaaaaaaaaaa")


async def test_presentations_title_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with console_env(tmp_path, monkeypatch) as (client, share):
        write_share_file(share, "upload/a.png", b"\x89PNG fake")
        await store_module.presentation_store.insert(
            "a" * 16, (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
            "/view/presentation/aaaaaaaaaaaaaaaa", ["upload/a.png"], title="Weekly report",
        )
        await store_module.presentation_store.insert(
            "b" * 16, (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            "/view/presentation/bbbbbbbbbbbbbbbb", ["upload/a.png"],
        )
        r = await client.get("/api/presentations")
        body = r.json()
        titles = {p["id"]: p["title"] for p in body["presentations"]}
        assert titles == {"a" * 16: "Weekly report", "b" * 16: None}


async def test_presentations_partial_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with console_env(tmp_path, monkeypatch) as (client, share):
        write_share_file(share, "upload/a.png", b"\x89PNG fake")
        await store_module.presentation_store.insert(
            "b" * 16, (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            "/view/presentation/bbbbbbbbbbbbbbbb", ["upload/a.png", "upload/gone.png"],
        )
        r = await client.get("/api/presentations")
        body = r.json()
        p = body["presentations"][0]
        assert p["state"] == "partial"
        assert p["present_count"] == 1
        assert p["file_count"] == 2
        assert p["kind"] == "package"


async def test_presentations_30day_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with console_env(tmp_path, monkeypatch) as (client, _share):
        old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        await store_module.presentation_store.insert(
            "c" * 16, old, "/view/presentation/cccccccccccccccc", ["upload/a.png"],
        )
        r = await client.get("/api/presentations")
        body = r.json()
        assert body["presentations"] == []


async def test_presentations_method_guard(api_client: httpx.AsyncClient) -> None:
    r = await api_client.post("/api/presentations")
    assert r.status_code == 405


async def test_cancel_unknown_task(api_client: httpx.AsyncClient) -> None:
    r = await api_client.post("/api/tasks/does-not-exist/cancel")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


@pytest.mark.parametrize("method", ["GET", "HEAD", "PUT", "DELETE", "PATCH"])
async def test_cancel_wrong_method_returns_405(api_client: httpx.AsyncClient, method: str) -> None:
    """A method other than POST on cancel answers 405 and allows only POST."""
    r = await api_client.request(method, "/api/tasks/abc/cancel")
    assert r.status_code == 405
    assert r.headers["content-type"].startswith("application/json")
    assert r.headers["allow"] == "POST"
    assert r.headers["x-frame-options"] == "DENY"
    if method != "HEAD":
        assert r.json() == {
            "error": f"Method '{method}' is not allowed at '/api/tasks/abc/cancel'.",
            "code": "method_not_allowed",
        }


async def test_cancel_already_terminal(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("d", None, "2026-08-23T01:00:00+00:00", "runs/d")
    await store_module.task_store.finish("d", "completed", ended_at="2026-08-23T01:00:05+00:00")
    r = await api_client.post("/api/tasks/d/cancel")
    assert r.status_code == 409
    body = r.json()
    assert body["code"] == "already_terminal"
    assert body["status"] == "completed"


async def test_cancel_no_live_task(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("n", None, "2026-08-23T01:00:00+00:00", "runs/n")
    r = await api_client.post("/api/tasks/n/cancel")
    assert r.status_code == 409
    body = r.json()
    assert body["code"] == "no_live_task"
    assert body["status"] == "running"


async def test_cancel_live_task_returns_202(api_client: httpx.AsyncClient) -> None:
    async def slow() -> None:
        await asyncio.sleep(30)

    task = await task_manager.create_task(slow)
    await wait_record(str(task.id), "running")

    r = await api_client.post(f"/api/tasks/{task.id}/cancel")
    assert r.status_code == 202
    assert r.json() == {"task_id": str(task.id), "status": "running"}

    record = await wait_record(str(task.id), "cancelled")
    assert record.status == "cancelled"


async def test_cancel_worker_exception_retrieved(
    api_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel job that raises has its error picked up, and the call still answers 202."""
    async def boom(_task_id: str) -> None:
        raise RuntimeError("cancel blew up")

    monkeypatch.setattr(task_manager, "cancel", boom)

    async def slow() -> None:
        await asyncio.sleep(30)

    live = await task_manager.create_task(slow)
    await wait_record(str(live.id), "running")

    retrieved: list[asyncio.Task] = []
    created: list[asyncio.Task] = []

    class _SpyTask(asyncio.Task):
        def exception(self) -> BaseException | None:
            if self.get_name().startswith("cancel-"):
                retrieved.append(self)
            return super().exception()

    def _spy_factory(_loop: asyncio.AbstractEventLoop, coro: Coroutine, **kwargs: str) -> asyncio.Task:
        worker = _SpyTask(coro, **kwargs)
        created.append(worker)
        return worker

    loop = asyncio.get_running_loop()
    loop.set_task_factory(_spy_factory)
    try:
        r = await api_client.post(f"/api/tasks/{live.id}/cancel")
        assert r.status_code == 202
        assert r.json() == {"task_id": str(live.id), "status": "running"}
        worker = next(t for t in created if t.get_name() == f"cancel-{live.id}")
        done, _pending = await asyncio.wait({worker}, timeout=2)
        assert worker in done
        assert worker in retrieved
    finally:
        loop.set_task_factory(None)
    await live.cancel()
    await wait_record(str(live.id), "cancelled")


def _make_share(root: Path) -> None:
    (root / "downloads").mkdir(parents=True)
    (root / "downloads" / "cat.jpg").write_bytes(b"x" * 100)
    (root / "uploads").mkdir()
    (root / "uploads" / "a.txt").write_text("hello")
    (root / "readme.txt").write_text("hi")


async def test_resources_root_listing(api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    share = tmp_path / "share"
    _make_share(share)
    monkeypatch.setenv("RESOURCES_DIR", str(share))

    r = await api_client.get("/api/resources")
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == ""
    assert body["parent"] is None
    assert [e["name"] for e in body["entries"]] == ["downloads", "uploads", "readme.txt"]
    d = body["entries"][0]
    assert d["type"] == "dir" and d["path"] == "downloads"
    assert d["file_count"] == 1 and d["total_size"] == 100
    assert "size" not in d
    f = body["entries"][2]
    assert f["type"] == "file" and f["path"] == "readme.txt" and f["size"] == 2
    assert isinstance(f["mtime"], int) and f["mtime"] > 0
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", f["mtime_local"])
    assert f["kind"] == "text"
    assert set(f["thumb"]) == {"status", "url"}
    assert f["thumb"]["status"] == "pending"
    assert f["thumb"]["url"] == "/api/resources/thumb?path=readme.txt"


async def test_resources_subdir_listing(api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    share = tmp_path / "share"
    _make_share(share)
    monkeypatch.setenv("RESOURCES_DIR", str(share))

    r = await api_client.get("/api/resources", params={"path": "downloads/"})
    body = r.json()
    assert body["path"] == "downloads"
    assert body["parent"] == ""
    e = body["entries"][0]
    assert set(e) == {"name", "path", "type", "mtime", "mtime_local", "size", "kind", "thumb"}
    assert e["name"] == "cat.jpg"
    assert e["path"] == "downloads/cat.jpg"
    assert e["type"] == "file"
    assert e["size"] == 100
    assert e["kind"] == "image"
    assert e["thumb"]["status"] == "pending"
    assert e["thumb"]["url"] == "/api/resources/thumb?path=downloads/cat.jpg"


async def test_resources_dot_segments_echo_normalized(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path with dot parts comes back cleaned up, naming the folder that was listed."""
    share = tmp_path / "share"
    _make_share(share)
    monkeypatch.setenv("RESOURCES_DIR", str(share))

    r = await api_client.get("/api/resources", params={"path": "downloads/./.."})
    assert r.status_code == 200
    body = r.json()
    assert (body["path"], body["parent"]) == ("", None)
    assert {e["name"] for e in body["entries"]} == {"downloads", "uploads", "readme.txt"}

    r = await api_client.get("/api/resources", params={"path": "uploads/../downloads/./"})
    body = r.json()
    assert (body["path"], body["parent"]) == ("downloads", "")
    assert [e["name"] for e in body["entries"]] == ["cat.jpg"]


async def test_resources_missing_and_traversal(api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    share = tmp_path / "share"
    share.mkdir()
    monkeypatch.setenv("RESOURCES_DIR", str(share))
    r = await api_client.get("/api/resources", params={"path": "nope"})
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"
    r = await api_client.get("/api/resources", params={"path": "../outside"})
    assert r.status_code == 400
    assert r.json()["code"] == "bad_param"


@pytest.mark.parametrize("raw_query", ["path=%00", "path=downloads%00x"])
async def test_resources_nul_path_returns_bad_param(api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_query: str) -> None:
    """A NUL byte in the path answers the JSON 400 bad_param."""
    share = tmp_path / "share"
    _make_share(share)
    monkeypatch.setenv("RESOURCES_DIR", str(share))
    r = await api_client.get(f"/api/resources?{raw_query}")
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert set(body) == {"error", "code"}
    assert body["code"] == "bad_param"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"


async def test_resources_listing_skips_vanished_entries(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken link in the share is skipped and the rest still lists."""
    share = tmp_path / "share"
    _make_share(share)
    (share / "broken-top").symlink_to("missing-target")
    (share / "downloads" / "broken-nested").symlink_to("missing-target")
    monkeypatch.setenv("RESOURCES_DIR", str(share))

    r = await api_client.get("/api/resources")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert [e["name"] for e in body["entries"]] == ["downloads", "uploads", "readme.txt"]
    d = body["entries"][0]
    assert d["file_count"] == 1 and d["total_size"] == 100


async def test_resources_listing_skips_symlinks_out_of_share(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Links that point outside the share are left out of the listing."""
    share = tmp_path / "share"
    outside = tmp_path / "outside"
    share.mkdir()
    (outside / "target-dir").mkdir(parents=True)
    (outside / "target-dir" / "leak.bin").write_bytes(b"z" * 500)
    (outside / "leak-file.bin").write_bytes(b"z" * 500)
    (share / "keep.txt").write_text("hello")
    (share / "escape-dir").symlink_to(outside / "target-dir")
    (share / "escape-file").symlink_to(outside / "leak-file.bin")
    monkeypatch.setenv("RESOURCES_DIR", str(share))

    r = await api_client.get("/api/resources")
    assert r.status_code == 200
    body = r.json()
    assert [e["name"] for e in body["entries"]] == ["keep.txt"]
    assert body["entries"][0]["size"] == 5


async def test_resources_search(api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    share = tmp_path / "share"
    (share / "downloads" / "deep").mkdir(parents=True)
    (share / "downloads" / "deep" / "coffee-notes.txt").write_text("notes")
    (share / "downloads" / "cat.jpg").write_bytes(b"x" * 50)
    (share / "README.md").write_text("# hi")
    monkeypatch.setenv("RESOURCES_DIR", str(share))

    r = await api_client.get("/api/resources/search", params={"q": "CAT"})
    assert r.status_code == 200
    body = r.json()
    assert body["truncated"] is False
    assert [m["path"] for m in body["matches"]] == ["downloads/cat.jpg"]
    m = body["matches"][0]
    assert m["name"] == "cat.jpg"
    assert m["type"] == "file"
    assert m["size"] == 50
    assert m["matched_ancestor_paths"] == ["downloads"]
    assert m["kind"] == "image"
    assert set(m["thumb"]) == {"status", "url"}
    assert m["thumb"]["status"] == "pending"


async def test_resources_search_skips_vanished_entries(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken link that matches the search is skipped and the rest still comes back."""
    share = tmp_path / "share"
    share.mkdir()
    (share / "green.txt").write_text("notes")
    (share / "broken-link").symlink_to("missing-target")
    monkeypatch.setenv("RESOURCES_DIR", str(share))

    r = await api_client.get("/api/resources/search", params={"q": "e"})
    assert r.status_code == 200
    body = r.json()
    assert body["truncated"] is False
    assert [m["path"] for m in body["matches"]] == ["green.txt"]


async def test_resources_search_matches_dirs_and_descendants(api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    share = tmp_path / "share"
    (share / "a" / "b" / "c").mkdir(parents=True)
    (share / "a" / "b" / "c" / "f.txt").write_text("x")
    monkeypatch.setenv("RESOURCES_DIR", str(share))

    body = (await api_client.get("/api/resources/search", params={"q": "a/b"})).json()
    assert body["truncated"] is False
    assert [m["path"] for m in body["matches"]] == ["a/b", "a/b/c", "a/b/c/f.txt"]
    by_path = {m["path"]: m for m in body["matches"]}
    assert by_path["a/b"]["type"] == "dir"
    assert "size" not in by_path["a/b"]
    assert by_path["a/b"]["matched_ancestor_paths"] == ["a"]
    assert by_path["a/b/c"]["matched_ancestor_paths"] == ["a", "b"]
    assert by_path["a/b/c/f.txt"]["matched_ancestor_paths"] == ["a", "b", "c"]
    assert by_path["a/b/c/f.txt"]["size"] == 1
    assert "kind" not in by_path["a/b"]
    assert "thumb" not in by_path["a/b"]
    assert by_path["a/b/c/f.txt"]["kind"] == "text"
    assert by_path["a/b/c/f.txt"]["thumb"]["status"] == "pending"


async def test_resources_listing_done_thumb(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file whose thumbnail is already rendered comes back as done."""
    share = tmp_path / "share"
    (share / "media_gen").mkdir(parents=True)
    (share / "media_gen" / "a.png").write_bytes(b"x" * 100)
    monkeypatch.setenv("RESOURCES_DIR", str(share))
    mtime = int((share / "media_gen" / "a.png").stat().st_mtime)
    thumb_dir = Path(os.environ["DATA_DIR"]) / "render" / cache.key_for("media_gen/a.png", mtime)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    (thumb_dir / "thumb.png").write_bytes(b"png")

    try:
        body = (await api_client.get("/api/resources", params={"path": "media_gen"})).json()
    finally:
        (thumb_dir / "thumb.png").unlink()
        thumb_dir.rmdir()
    e = body["entries"][0]
    assert e["kind"] == "image"
    assert e["thumb"]["status"] == "done"
    assert e["thumb"]["url"] == "/api/resources/thumb?path=media_gen/a.png"


async def test_resources_listing_no_engine_thumb(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kind that gets no thumbnail engine comes back as none."""
    share = tmp_path / "share"
    share.mkdir()
    (share / "song.mp3").write_bytes(b"x")
    monkeypatch.setenv("RESOURCES_DIR", str(share))

    body = (await api_client.get("/api/resources")).json()
    e = body["entries"][0]
    assert e["kind"] == "audio"
    assert e["thumb"]["status"] == "none"


def _make_cap_share(root: Path) -> None:
    for i in range(5000):
        (root / f"x{i:05d}").mkdir()
    (root / "a.txt").write_text("x")


async def test_resources_search_limit_truncates(api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    share = tmp_path / "share"
    share.mkdir()
    (share / "f0.log").write_text("x")
    (share / "f1.log").write_text("x")
    (share / "f2.log").write_text("x")
    monkeypatch.setenv("RESOURCES_DIR", str(share))

    body = (await api_client.get("/api/resources/search", params={"q": "f", "limit": 2})).json()
    assert [m["path"] for m in body["matches"]] == ["f0.log", "f1.log"]
    assert body["truncated"] is True


async def test_resources_search_node_cap(api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    share = tmp_path / "share"
    share.mkdir()
    _make_cap_share(share)
    monkeypatch.setenv("RESOURCES_DIR", str(share))

    body = (await api_client.get("/api/resources/search", params={"q": "a"})).json()
    assert body["truncated"] is True
    assert [m["path"] for m in body["matches"]] == ["a.txt"]


async def test_resources_search_skips_symlinks_out_of_share(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Links that point outside the share are not matched and not followed."""
    share = tmp_path / "share"
    outside = tmp_path / "outside"
    share.mkdir()
    outside.mkdir()
    (share / "passwd-notes.txt").write_text("in share")
    (outside / "passwd").write_text("outside")
    (share / "zz_out").symlink_to(outside)
    (share / "passwd_link").symlink_to(outside / "passwd")
    monkeypatch.setenv("RESOURCES_DIR", str(share))

    r = await api_client.get("/api/resources/search", params={"q": "passwd"})
    assert r.status_code == 200
    body = r.json()
    assert body["truncated"] is False
    assert [m["path"] for m in body["matches"]] == ["passwd-notes.txt"]


def _make_capped_escape_share(root: Path, outside: Path) -> None:
    for i in range(4999):
        (root / f"y{i:05d}").write_text("x")
    (root / "aa_out").symlink_to(outside)


async def test_resources_search_escape_does_not_consume_node_cap(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link out of the share does not count toward the walk limit."""
    share = tmp_path / "share"
    outside = tmp_path / "outside"
    share.mkdir()
    outside.mkdir()
    (outside / "nomatch_target.txt").write_text("x")
    _make_capped_escape_share(share, outside)
    monkeypatch.setenv("RESOURCES_DIR", str(share))

    body = (await api_client.get("/api/resources/search", params={"q": "nomatch"})).json()
    assert body["truncated"] is False
    assert body["matches"] == []


async def test_resources_search_bad_params(api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    share = tmp_path / "share"
    share.mkdir()
    monkeypatch.setenv("RESOURCES_DIR", str(share))
    r = await api_client.get("/api/resources/search", params={"q": ""})
    assert r.status_code == 400
    assert r.json()["code"] == "bad_param"
    r = await api_client.get("/api/resources/search")
    assert r.status_code == 400
    r = await api_client.get("/api/resources/search", params={"q": "x", "limit": 0})
    assert r.status_code == 400


async def test_vnc_status(api_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def reachable() -> bool:
        return True

    monkeypatch.setattr(api_module, "vnc_reachable", reachable)
    r = await api_client.get("/api/vnc/status")
    assert r.status_code == 200
    assert r.json() == {"reachable": True}

    async def boom() -> bool:
        raise RuntimeError("probe failed")

    monkeypatch.setattr(api_module, "vnc_reachable", boom)
    r = await api_client.get("/api/vnc/status")
    assert r.status_code == 500
    assert r.json()["code"] == "internal"


@pytest.mark.live
@pytest.mark.timeout(30)
async def test_preview_vnc_status(api_client: httpx.AsyncClient) -> None:
    """The real screen viewer probe reports it is reachable."""
    require_novnc()
    r = await api_client.get("/api/vnc/status")
    assert r.status_code == 200
    assert r.json() == {"reachable": True}


async def test_l2_renders_rednote_rows(api_client: httpx.AsyncClient) -> None:
    """Four seeded runs fit one page with no next cursor.

    An unknown tool answers 404.
    """
    await seed_task("e2e-rednote-running", "rednote_research")
    await seed_task("e2e-rednote-done-1", "rednote_research", "2026-08-23T00:00:00+00:00", "completed", ended_at="2026-08-23T00:00:05+00:00")
    await seed_task("e2e-rednote-done-2", "rednote_research", "2026-08-22T23:00:00+00:00", "completed", ended_at="2026-08-22T23:00:05+00:00")
    await seed_task("e2e-rednote-failed", "rednote_research", "2026-08-22T22:00:00+00:00", "failed", error="boom", ended_at="2026-08-22T22:00:05+00:00")
    body = (await api_client.get("/api/tools/rednote_research/tasks", params={"limit": 200})).json()
    assert len(body["tasks"]) == 4
    assert body["next_cursor"] is None
    assert {t["status"] for t in body["tasks"]} == {"running", "completed", "failed"}
    r = await api_client.get("/api/tools/no_such_tool/tasks")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


async def test_detail_renders_running_fixture(api_client: httpx.AsyncClient) -> None:
    """A running parent shows its status and its two subtasks.

    Cancel answers 409 and an unknown id answers 404.
    """
    await seed_task(E2E_RUNNING, "rednote_research")
    await seed_task("e2e-running-sub-1", "rednote_research", "2026-08-23T01:00:01+00:00", parent_task_id=E2E_RUNNING)
    await seed_task("e2e-running-sub-2", "rednote_research", "2026-08-23T01:00:02+00:00", parent_task_id=E2E_RUNNING)
    body = (await api_client.get(f"/api/tasks/{E2E_RUNNING}")).json()
    assert body["task"]["status"] == "running"
    assert len(body["subtasks"]) == 2
    assert all(s["parent_task_id"] == E2E_RUNNING for s in body["subtasks"])
    r = await api_client.post(f"/api/tasks/{E2E_RUNNING}/cancel")
    assert r.status_code == 409
    assert r.json()["code"] == "no_live_task"
    r = await api_client.get("/api/tasks/does-not-exist")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


_RESULT_LINE_BYTES = 107
_RESULT_LINE_PAD = 89


def _result_content(size: int) -> str:
    """Make exactly size bytes of result text out of 107 byte lines."""
    line = "e2e-result 000000" + "a" * _RESULT_LINE_PAD + "\n"
    return line * (size // _RESULT_LINE_BYTES)


async def test_result_fixture_windows(api_client: httpx.AsyncClient) -> None:
    """A 793,191 byte result is cut short in a 4096 byte window and downloads whole."""
    await seed_task(E2E_RESULT_BIG, "generate_image", E2E_STARTED_AT, "completed", ended_at=E2E_ENDED_AT)
    write_run_file(store_module.task_store.data_dir, E2E_RESULT_BIG, "result", _result_content(_RESULT_BYTES_0797))
    b = (await api_client.get(f"/api/tasks/{E2E_RESULT_BIG}/result", params={"offset": 0, "max_bytes": 4096})).json()
    assert b["size"] == _RESULT_BYTES_0797
    assert b["truncated"] is True
    assert len(b["content"].encode()) <= 4096
    d = await api_client.get(f"/api/tasks/{E2E_RESULT_BIG}/result/download")
    assert d.status_code == 200
    assert len(d.content) == _RESULT_BYTES_0797


async def _reassemble_result_windows(c: httpx.AsyncClient, task_id: str) -> tuple[bytes, int]:
    """Read the result window by window.

    Returns:
        The joined bytes and how many windows it took.
    """
    acc = b""
    off = 0
    windows = 0
    for _ in range(64):
        b = (await c.get(f"/api/tasks/{task_id}/result", params={"offset": off, "max_bytes": 524288})).json()
        acc += b["content"].encode("utf-8")
        off = b["offset"] + b["content_bytes"]
        windows += 1
        if not b["truncated"] or off >= b["size"]:
            break
    return acc, windows


async def test_result_multiwindow_reassembly(api_client: httpx.AsyncClient) -> None:
    """Reading the 793,191 byte result window by window rebuilds it byte for byte."""
    await seed_task(E2E_RESULT_BIG, "generate_image", E2E_STARTED_AT, "completed", ended_at=E2E_ENDED_AT)
    write_run_file(store_module.task_store.data_dir, E2E_RESULT_BIG, "result", _result_content(_RESULT_BYTES_0797))
    d = await api_client.get(f"/api/tasks/{E2E_RESULT_BIG}/result/download")
    assert d.status_code == 200
    acc, windows = await _reassemble_result_windows(api_client, E2E_RESULT_BIG)
    assert windows >= 2
    assert len(acc) == _RESULT_BYTES_0797
    assert acc == d.content
_LOG238_ID = "e2e-log-238"
_LOG238_LINES = 238

async def test_log_fixture_backward_window(api_client: httpx.AsyncClient) -> None:
    """A 238 line log counts 238 lines and does not reach the head.

    Reading it forward is not cut short.
    """
    await seed_task(_LOG238_ID, "rednote_research")
    write_run_file(store_module.task_store.data_dir, _LOG238_ID, "log", log_lines(_LOG238_LINES))
    b = (await api_client.get(f"/api/tasks/{_LOG238_ID}/log", params={"lines": 200, "end_offset": 0})).json()
    assert b["total_lines"] == _LOG238_LINES
    assert b["head_reached"] is False
    assert b["content_bytes"] > 0
    f = (await api_client.get(f"/api/tasks/{_LOG238_ID}/log", params={"offset": 0, "max_bytes": 1048576})).json()
    assert f["truncated"] is False
_TOP_FOLDERS = ("github", "media_gen", "tts", "upload")
_FLAKE_FILE_COUNT = 108
_DOWNLOAD_FILE_COUNT = 12


def _make_top_level_share(share: Path) -> None:
    """Write a share with six top level folders."""
    for i in range(_FLAKE_FILE_COUNT):
        write_share_file(share, f"debug_flake/flake_{i:03d}.txt", b"flake")
    for i in range(_DOWNLOAD_FILE_COUNT):
        write_share_file(share, f"downloads/dl_{i:02d}.png", b"png")
    for folder in _TOP_FOLDERS:
        write_share_file(share, f"{folder}/placeholder.txt", b"placeholder")


async def test_resources_renders_top_level(api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The six top level folders list with their exact file counts."""
    share = tmp_path / "share"
    _make_top_level_share(share)
    monkeypatch.setenv("RESOURCES_DIR", str(share))
    body = (await api_client.get("/api/resources")).json()
    names = [e["name"] for e in body["entries"]]
    assert names == ["debug_flake", "downloads", "github", "media_gen", "tts", "upload"]
    top = {e["name"]: e for e in body["entries"]}
    assert top["debug_flake"]["file_count"] == _FLAKE_FILE_COUNT
    assert top["downloads"]["file_count"] == _DOWNLOAD_FILE_COUNT
    sub = (await api_client.get("/api/resources", params={"path": "downloads"})).json()
    assert len(sub["entries"]) == _DOWNLOAD_FILE_COUNT


_RETIRED_TOOL = "retired_tool"
_RETIRED_TID = "retired-result-1"
_RETIRED_FILE = "media_gen/gen_001.png"
_RETIRED_RESULT = "Saved: media_gen/gen_001.png\n"
_BEFORE_E2E_RUNS = "2026-08-22T23:00:00+00:00"


async def test_unregistered_tool_run_stays_readable(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run of a tool that is no longer registered is listed and viewable with its files.

    It cannot run again, and the tool has no page of its own.
    """
    share = tmp_path / "share"
    write_share_file(share, _RETIRED_FILE, E2E_PNG)
    monkeypatch.setenv("RESOURCES_DIR", str(share))
    await seed_task(_RETIRED_TID, _RETIRED_TOOL, E2E_STARTED_AT, "completed",
                    params={"prompt": "a tea room"}, ended_at=E2E_ENDED_AT)
    write_run_file(store_module.task_store.data_dir, _RETIRED_TID, "result", _RETIRED_RESULT)
    write_run_file(store_module.task_store.data_dir, _RETIRED_TID, "files", _RETIRED_FILE + "\n")
    changes = (await api_client.get(
        "/api/changes", params={"since_fp": "no-match", "since_gen": _BEFORE_E2E_RUNS}
    )).json()
    assert [(t["task_id"], t["tool"]) for t in changes["finished"]] == [(_RETIRED_TID, _RETIRED_TOOL)]
    detail = (await api_client.get(f"/api/tasks/{_RETIRED_TID}")).json()
    assert detail["task"]["tool"] == _RETIRED_TOOL
    assert detail["task"]["status"] == "completed"
    assert detail["task"]["rerunnable"] is False
    assert detail["result_files"] == [_RETIRED_FILE]
    result = (await api_client.get(f"/api/tasks/{_RETIRED_TID}/result")).json()
    assert result["content"] == _RETIRED_RESULT
    feed = (await api_client.get(f"/api/tasks/{_RETIRED_TID}/result/feed")).json()
    assert [(f["path"], f["present"]) for f in feed["files"]] == [(_RETIRED_FILE, True)]
    r = await api_client.get(f"/api/tools/{_RETIRED_TOOL}/tasks")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"



async def test_detail_includes_agent_name_and_rerunnable(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("t-agent", None, "2026-08-23T01:00:00+00:00", "runs/t-agent")
    await store_module.agent_store.insert("t-agent", "planner")
    body = (await api_client.get("/api/tasks/t-agent")).json()
    assert body["task"]["agent_name"] == "planner"
    assert body["task"]["rerunnable"] is False
    assert body["task"]["tool"] is None


async def test_detail_tool_is_rerunnable(api_client: httpx.AsyncClient) -> None:
    tool_name = next(iter(registered_tools()))
    await store_module.task_store.insert_running("t-tool", None, "2026-08-23T01:00:00+00:00", "runs/t-tool")
    await store_module.tool_store.insert("t-tool", tool_name, None, {})
    body = (await api_client.get("/api/tasks/t-tool")).json()
    assert body["task"]["agent_name"] is None
    assert body["task"]["rerunnable"] is True
    assert body["task"]["tool"] == tool_name


async def test_detail_flat_subtasks_no_nesting(api_client: httpx.AsyncClient) -> None:
    now = "2026-08-23T01:00:00+00:00"
    await store_module.task_store.insert_running("parent", None, now, "runs/parent")
    await store_module.tool_store.insert("parent", "brave_search", None, {"query": "x"})
    await store_module.task_store.insert_running("child1", "parent", now, "runs/child1")
    await store_module.task_store.insert_running("child2", "parent", now, "runs/child2")
    await store_module.task_store.insert_running("grandchild", "child1", now, "runs/grandchild")
    body = (await api_client.get("/api/tasks/parent")).json()
    subtask_ids = [s["task_id"] for s in body["subtasks"]]
    assert subtask_ids == ["child1", "child2"]
    assert not any("subtasks" in s for s in body["subtasks"])


async def test_detail_ancestors(api_client: httpx.AsyncClient) -> None:
    now = "2026-08-23T01:00:00+00:00"
    await store_module.task_store.insert_running("root", None, now, "runs/root")
    await store_module.tool_store.insert("root", "brave_search", None, {"query": "x"})
    await store_module.task_store.insert_running("mid", "root", now, "runs/mid")
    await store_module.task_store.insert_running("leaf", "mid", now, "runs/leaf")
    await store_module.agent_store.insert("mid", "collector")
    body = (await api_client.get("/api/tasks/leaf")).json()
    assert len(body["ancestors"]) == 2
    assert body["ancestors"][0]["task_id"] == "root"
    assert body["ancestors"][1]["task_id"] == "mid"
    assert body["ancestors"][1]["agent_name"] == "collector"
    assert body["ancestors"][0]["status"] == "running"
    top = (await api_client.get("/api/tasks/root")).json()
    assert top["ancestors"] == []


async def test_changes_payload_includes_agent_name(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("chg-agent", None, "2026-08-23T01:00:00+00:00", "runs/chg-agent")
    await store_module.agent_store.insert("chg-agent", "drafter")
    body = (await api_client.get("/api/changes")).json()
    tasks = body["running"] + body["started"] + body["finished"]
    found = [t for t in tasks if t["task_id"] == "chg-agent"]
    assert found and found[0]["agent_name"] == "drafter"


async def test_detail_short_id_unique_prefix(api_client: httpx.AsyncClient) -> None:
    full = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
    await store_module.task_store.insert_running(full, None, "2026-08-23T01:00:00+00:00", "runs/" + full)
    body = (await api_client.get("/api/tasks/3f2504e0")).json()
    assert body["task"]["task_id"] == full


async def test_detail_short_id_ambiguous_returns_404(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("3f2504e0-1111-1111-1111-111111111111", None, "2026-08-23T01:00:00+00:00", "runs/a")
    await store_module.task_store.insert_running("3f2504e0-2222-2222-2222-222222222222", None, "2026-08-23T01:00:00+00:00", "runs/b")
    res = await api_client.get("/api/tasks/3f2504e0")
    assert res.status_code == 404
    assert res.json()["code"] == "not_found"


async def test_detail_unknown_id_returns_404(api_client: httpx.AsyncClient) -> None:
    res = await api_client.get("/api/tasks/00000000")
    assert res.status_code == 404
    assert res.json()["code"] == "not_found"



@pytest.mark.timeout(30)
async def test_console_stub_fake_call_starts_a_running_task(console_stub: SimpleNamespace) -> None:
    """A tool call on the stub origin records a running task without running the tool."""
    async with httpx.AsyncClient(base_url=console_stub.base_url, timeout=10) as c:
        r = await c.post("/api/tools/rednote_research/call", json={"query": "e2e"})
        assert r.status_code == 202
        assert r.json()["status"] == "running"
        task_id = r.json()["task_id"]
        assert console_stub.last_call_id() == task_id
        detail = (await c.get("/api/tasks/" + task_id)).json()
        assert detail["task"]["status"] == "running"
        assert detail["task"]["params"] == {"query": "e2e"}
        running = (await c.get("/api/changes")).json()["running"]
        assert [t["task_id"] for t in running] == [task_id]
        r = await c.post("/api/tools/ghost-tool/call", json={})
        assert r.status_code == 404
        assert r.json()["code"] == "not_found"
        r = await c.post("/api/tools/rednote_research/call", content=b"not json",
                         headers={"Content-Type": "application/json"})
        assert r.status_code == 400
        assert r.json()["code"] == "bad_param"


@pytest.mark.timeout(30)
async def test_console_stub_force_status_knob(console_stub: SimpleNamespace) -> None:
    """A forced status answers the matching endpoint, and leaves the rest untouched."""
    console_stub.force_status(r"/api/changes", 500, code="stub_down", message="stub outage")
    async with httpx.AsyncClient(base_url=console_stub.base_url, timeout=10) as c:
        r = await c.get("/api/changes")
        assert r.status_code == 500
        assert r.json() == {"error": "stub outage", "code": "stub_down"}
        assert (await c.get("/api/tools")).status_code == 200


@pytest.mark.timeout(30)
async def test_console_stub_pin_body_knob(console_stub: SimpleNamespace) -> None:
    """A pinned body answers the matching endpoint with the fixed payload."""
    body = {"status": "ok", "version": "stub", "dav_base_url": console_stub.base_url}
    console_stub.pin_body(r"/api/health", body)
    async with httpx.AsyncClient(base_url=console_stub.base_url, timeout=10) as c:
        assert (await c.get("/api/health")).json() == body
        assert (await c.get("/api/tools")).status_code == 200


@pytest.mark.timeout(30)
async def test_console_stub_delay_knob(console_stub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """A delay waits before the first matching reply only when first_only is set."""
    delay_s = 0.4
    held: list[float] = []
    real_sleep = asyncio.sleep

    async def spy_sleep(seconds: float, result: None = None) -> None:
        if seconds == delay_s:
            held.append(seconds)
        return await real_sleep(seconds, result)

    monkeypatch.setattr(asyncio, "sleep", spy_sleep)
    console_stub.delay(r"/api/tools", delay_s, first_only=True)
    async with httpx.AsyncClient(base_url=console_stub.base_url, timeout=10) as c:
        t0 = time.monotonic()
        assert (await c.get("/api/tools")).status_code == 200
        first = time.monotonic() - t0
        assert held == [delay_s]
        assert (await c.get("/api/tools")).status_code == 200
        assert held == [delay_s]  # the second reply is not held
    assert first >= delay_s


@pytest.mark.timeout(30)
async def test_console_stub_seed_and_finish_drive_changes(console_stub: SimpleNamespace) -> None:
    """Seeding and finishing tasks change what the changes endpoint reports, and a match is a 304."""
    await console_stub.seed_task("stub-a1", "rednote_research", "2026-08-23T01:00:00+00:00", "completed",
                                 ended_at="2026-08-23T01:00:05+00:00", params={"query": "one"})
    await console_stub.seed_task("stub-r1", "rednote_research", "2026-08-23T03:00:00+00:00", params={"query": "live"})
    async with httpx.AsyncClient(base_url=console_stub.base_url, timeout=10) as c:
        first = (await c.get("/api/changes")).json()
        assert [t["task_id"] for t in first["running"]] == ["stub-r1"]
        fingerprint = first["fingerprint"]
        r = await c.get("/api/changes", params={"since_fp": fingerprint, "since_gen": first["generated_at"]})
        assert r.status_code == 304
        await console_stub.finish("stub-r1", "completed", ended_at="2030-01-01T00:00:00+00:00")
        r = await c.get("/api/changes", params={"since_fp": fingerprint, "since_gen": first["generated_at"]})
        assert r.status_code == 200
        assert r.json()["running"] == []
        assert [t["task_id"] for t in r.json()["finished"]] == ["stub-r1"]
