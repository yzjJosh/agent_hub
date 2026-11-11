"""Tests for the console tool schema and tool call endpoints."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, override

import httpx
import pytest
from httpx import ASGITransport
from mcp.types import TextContent
from starlette.applications import Starlette
from starlette.routing import Mount

import app.console.api as api_module
import app.console.toolcall as toolcall_module
import app.lib.db.store as store_module
import app.mcp.registry as registry_module
from app.console.api import ConsolePathSanitizer, ConsoleSecurityMiddleware, build_api_routes
from app.lib.tool.base import Param, Tool, ToolResult
from app.mcp.registry import mcp, registered_tools
from app.tools.present_to_user import present_to_user_tool
from app.tools.send_or_receive_file import send_or_receive_file_tool
from app.tools.web_fetch import web_fetch_tool
from tests.app.console.conftest import wait_record, wait_subtasks


def test_call_handler_runs_tools_itself_and_writes_no_rows() -> None:
    """The call handler reaches a tool through the tool class and never through MCP.

    The call handler is async and the schema handler is not.
    """
    assert asyncio.iscoroutinefunction(toolcall_module.tool_call_impl)
    assert not asyncio.iscoroutinefunction(toolcall_module.tool_schema_impl)
    source = (Path(api_module.STATIC_DIR).parent / "toolcall.py").read_text()
    assert ".call_tool(" not in source
    assert "FastMCP" not in source
    assert "run_start_slot" not in source
    assert "RunStart" not in source
    assert "ValidationError" not in source
    assert ".invoke(" not in source
    assert ".start(" not in source
    assert "create_task" not in source
    assert "task_manager" not in source
    assert "task_store.insert" not in source


class _FakeBlockingTool(Tool):
    """A tracked blocking fake tool with one param of each shape."""

    name: ClassVar[str] = "fake_blocking"
    description: ClassVar[str] = "A fast tracked blocking fake tool."
    summary: ClassVar[str] = "Fake blocking tool"
    rules: ClassVar[tuple[str, ...]] = ("Fake rule one", "Fake rule two")
    params: ClassVar[tuple[Param, ...]] = (
        Param("q", str, description="The query string."),
        Param("limit", int, 10, min=1, max=100, step=1, description="How many results to return."),
        Param("reference_images", list[str] | None, None, kind="resource_list",
              description="Existing files in the share that guide the call."),
        Param("tags", list[str] | None, None, description="Labels to filter the results by."),
        Param("mode", str, "image", enum=("image", "video"),
              description="Which mode to run in."),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text="fake result: " + params["q"])


class _FakeAsyncTool(Tool):
    """A tracked async fake tool."""

    name: ClassVar[str] = "fake_async"
    is_slow: ClassVar[bool] = True
    description: ClassVar[str] = "A tracked async fake tool."
    summary: ClassVar[str] = "Fake async tool"
    params: ClassVar[tuple[Param, ...]] = (Param("q", str, description="The query string."),)

    @override
    async def call(self, params: dict) -> ToolResult:
        await asyncio.sleep(0.05)
        return ToolResult(text="async result: " + params["q"])


class _FakeFailingRunTool(Tool):
    """A tracked async fake tool whose run fails."""

    name: ClassVar[str] = "fake_failing_run"
    description: ClassVar[str] = "A tracked fake tool whose run fails."
    summary: ClassVar[str] = "Fake failing run tool"
    params: ClassVar[tuple[Param, ...]] = (Param("q", str, description="The query string."),)

    @override
    async def call(self, _params: dict) -> ToolResult:
        await asyncio.sleep(0.05)
        raise RuntimeError("run boom")


class _FakeUntrackedBase(Tool):
    """Shared base for untracked fake tools, which never persist a run row."""

    tracked: ClassVar[bool] = False
    params: ClassVar[tuple[Param, ...]] = ()


class _FakeUntrackedTool(_FakeUntrackedBase):
    """An untracked fake tool that answers right away."""

    name: ClassVar[str] = "fake_untracked"
    description: ClassVar[str] = "An untracked fake tool that answers directly."
    summary: ClassVar[str] = "Fake untracked tool"

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="untracked result")


class _FakeFullDescTool(Tool):
    """A tracked fake tool with a messy multi line description."""

    name: ClassVar[str] = "fake_full_desc"
    description: ClassVar[str] = (
        "A ``backticked`` intro with trailing spaces   \r\n"
        "\r\n"
        "\r\n"
        "Actions:\r\n"
        "  alpha    One two\r\n"
        "           continued  line\r\n"
        "  beta     Three\r\n"
        "\r\n"
        "Tail."
    )
    summary: ClassVar[str] = "Fake full-desc tool"
    params: ClassVar[tuple[Param, ...]] = (Param("q", str, description="The query string."),)

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="full desc result")


class _FakeCachedTool(Tool):
    """A tracked fake tool that reuses its result for 24 hours."""

    name: ClassVar[str] = "fake_cached"
    staleness: ClassVar[timedelta] = timedelta(hours=24)
    description: ClassVar[str] = "A tracked staleness fake tool."
    summary: ClassVar[str] = "Fake cached tool"
    params: ClassVar[tuple[Param, ...]] = (Param("q", str, description="The query string."),)

    @override
    async def cache_key(self, params: dict) -> str:
        return f"key:{params['q']}"

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text="fresh result: " + params["q"])


class _FakeAsyncCachedTool(Tool):
    """A tracked async fake tool that reuses its result for 24 hours."""

    name: ClassVar[str] = "fake_async_cached"
    is_slow: ClassVar[bool] = True
    staleness: ClassVar[timedelta] = timedelta(hours=24)
    description: ClassVar[str] = "A tracked async staleness fake tool."
    summary: ClassVar[str] = "Fake async cached tool"
    params: ClassVar[tuple[Param, ...]] = (Param("q", str, description="The query string."),)

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text="async cached: " + params["q"])


class _FakeRaisingTool(Tool):
    """A tracked fake tool whose work raises."""

    name: ClassVar[str] = "fake_raising"
    description: ClassVar[str] = "A tracked fake tool whose call raises."
    summary: ClassVar[str] = "Fake raising tool"
    params: ClassVar[tuple[Param, ...]] = (Param("q", str, description="The query string."),)

    @override
    async def call(self, _params: dict) -> ToolResult:
        raise RuntimeError("raising boom")


class _FakeOuterTool(Tool):
    """A tracked blocking fake tool that runs another tool inside its own work."""

    name: ClassVar[str] = "fake_outer"
    description: ClassVar[str] = "A tracked blocking fake that nests a subtask inside its call."
    summary: ClassVar[str] = "Fake outer tool"
    params: ClassVar[tuple[Param, ...]] = (Param("q", str, description="The query string."),)

    def __init__(self) -> None:
        self._inner = _FakeBlockingTool()

    @override
    async def call(self, params: dict) -> ToolResult:
        await self._inner.result_text(await self._inner.run({"q": params["q"]}))
        return ToolResult(text="outer done: " + params["q"])


@pytest.fixture
async def toolcall_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[SimpleNamespace]:
    """A test client for the console api, with a tool map carrying the fake tools."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    store_module.init_store(str(tmp_path / "data" / "records.db"))
    await store_module.task_store.open()
    record: dict[str, type[Tool]] = dict(registered_tools())
    monkeypatch.setattr(registry_module, "_TOOLS", record)
    app = Starlette(routes=[Mount("/api", build_api_routes())])
    app.add_middleware(ConsoleSecurityMiddleware)
    app.add_middleware(ConsolePathSanitizer)

    def register(cls: type[Tool]) -> None:
        record[cls.name] = cls

    def register_outer() -> None:
        record[_FakeOuterTool.name] = _FakeOuterTool

    try:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield SimpleNamespace(client=c, app=app, register=register,
                                  register_outer=register_outer, data_dir=str(tmp_path / "data"))
    finally:
        await store_module.task_store.close()


# --- schema endpoint ---


async def test_schema_flat_shape(toolcall_client: SimpleNamespace) -> None:
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.get("/api/tools/fake_blocking/schema")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"tool", "summary", "description", "is_slow", "staleness", "cached", "rules", "params"}
    assert body["is_slow"] is False
    assert body["staleness"] is None
    assert body["cached"] is False
    assert body["tool"] == "fake_blocking"
    assert body["summary"] == "Fake blocking tool"
    assert body["description"] == "A fast tracked blocking fake tool."
    assert body["rules"] == ["Fake rule one", "Fake rule two"]
    assert "actions" not in body
    assert [p["name"] for p in body["params"]] == ["q", "limit", "reference_images", "tags", "mode"]
    q, limit, images, tags, mode = body["params"]
    assert q == {"name": "q", "type": "string", "item_type": None, "required": True,
                 "default": None, "enum": None, "kind": None, "min": None, "max": None,
                 "step": None, "description": "The query string."}
    assert limit["type"] == "integer"
    assert limit["required"] is False and limit["default"] == 10
    assert limit["min"] == 1 and limit["max"] == 100 and limit["step"] == 1
    assert images["type"] == "array" and images["kind"] == "resource_list"
    assert images["item_type"] == "string"
    assert images["required"] is False and images["default"] is None
    assert tags["type"] == "array" and tags["kind"] is None
    assert tags["item_type"] == "string"
    assert tags["required"] is False and tags["default"] is None
    assert mode["enum"] == ["image", "video"]


async def test_param_wire_console_divergence_is_deliberate(toolcall_client: SimpleNamespace) -> None:
    """The console and the wire serve different text for the same param.

    For a kinded param, the console serves the plain description with no added
    notes and no extra enum flag, and the wire serves the description with the
    notes added.
    """
    param = next(p for p in registered_tools()["redfin_search"].params if p.name == "date_from")
    r = await toolcall_client.client.get("/api/tools/redfin_search/schema")
    assert r.status_code == 200
    served = next(p for p in r.json()["params"] if p["name"] == "date_from")
    assert set(served) == {"name", "type", "item_type", "required", "default", "enum", "kind",
                           "min", "max", "step", "description"}
    assert "enum_enforced" not in served
    assert served["description"] == param.description
    assert "YYYY-MM-DD" not in served["description"]
    wire = next(t for t in await mcp.list_tools() if t.name == "redfin_search")
    prop = wire.to_mcp_tool().model_dump()["inputSchema"]["properties"]["date_from"]
    assert prop["description"] == param.wire_description
    assert "YYYY-MM-DD" in prop["description"]


async def test_schema_exposes_structured_behavioural_facts(toolcall_client: SimpleNamespace) -> None:
    """A slow cached tool serves its slow flag, its reuse window, and a cached flag.

    The served slow flag comes from the tool class, and the description stays
    the plain text the tool declares.
    """
    assert _FakeAsyncCachedTool.is_slow is True
    toolcall_client.register(_FakeAsyncCachedTool)
    r = await toolcall_client.client.get("/api/tools/fake_async_cached/schema")
    assert r.status_code == 200
    body = r.json()
    assert body["is_slow"] is True
    assert body["staleness"] == "24 hours"
    assert body["cached"] is True
    assert body["description"] == "A tracked async staleness fake tool."


async def test_schema_description_full_normalization(toolcall_client: SimpleNamespace) -> None:
    """The served description keeps its line breaks, with backticks and extra spaces gone."""
    toolcall_client.register(_FakeFullDescTool)
    r = await toolcall_client.client.get("/api/tools/fake_full_desc/schema")
    assert r.status_code == 200
    assert r.json()["description"] == (
        "A backticked intro with trailing spaces\n"
        "\n"
        "Actions:\n"
        "  alpha    One two continued line\n"
        "  beta     Three\n"
        "\n"
        "Tail."
    )


_GITHUB_DESCRIPTION = (
    "GitHub operations: issues, pull requests, repos, code, files, and releases.\n"
    "Call with one action plus the parameters it requires.\n"
    "\n"
    "Actions:\n"
    "  list_issues    List recent issues in one repo matching a query.\n"
    "  search_issues  Full-text search issues within one repo.\n"
    "  view_issue     Full issue body, metadata, and comments.\n"
    "  search_repos   Find repositories by name, description, or topic.\n"
    "  view_pr        Pull request body, status, and metadata.\n"
    "  search_code    Search code within one repository.\n"
    "  view_file      File content (or directory listing) at a path in a repo. limits: files up to "
    "100MB\n"
    "  list_releases  List releases (newest first) in a repo, incl. tag and date.\n"
    "\n"
    "Parameters not applicable to an action are ignored."
)


async def test_schema_description_github_actions_block(api_client: httpx.AsyncClient) -> None:
    """The github description keeps its action lines, joining a wrapped line into one."""
    r = await api_client.get("/api/tools/github/schema")
    assert r.status_code == 200
    assert r.json()["description"] == _GITHUB_DESCRIPTION


@pytest.mark.parametrize(
    ("name", "marker"),
    [
        ("redfin_search", "For a specific street address, returns that single property"),
        ("x_search", "Use ONLY when you specifically need social media"),
    ],
)
async def test_schema_description_prose_identity(
    api_client: httpx.AsyncClient, name: str, marker: str
) -> None:
    """A plain prose description is served exactly as the tool declares it."""
    r = await api_client.get(f"/api/tools/{name}/schema")
    assert r.status_code == 200
    body = r.json()
    assert body["description"] == registered_tools()[name].description
    assert marker in body["description"]


async def test_schema_not_exposed_403(toolcall_client: SimpleNamespace) -> None:
    toolcall_client.register(_FakeUntrackedTool)
    r = await toolcall_client.client.get("/api/tools/fake_untracked/schema")
    assert r.status_code == 403
    assert r.json() == {"error": "Tool 'fake_untracked' is not exposed in the console.", "code": "not_exposed"}


async def test_schema_unknown_tool_404(toolcall_client: SimpleNamespace) -> None:
    r = await toolcall_client.client.get("/api/tools/no_such_tool/schema")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


async def test_schema_wrong_method_405(toolcall_client: SimpleNamespace) -> None:
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.post("/api/tools/fake_blocking/schema")
    assert r.status_code == 405
    assert r.headers["allow"] == "GET, HEAD"
    assert r.json()["code"] == "method_not_allowed"


# --- call endpoint, run started with 202 ---


async def test_call_202_row_committed_before_response(toolcall_client: SimpleNamespace) -> None:
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.post("/api/tools/fake_blocking/call", json={"q": "hello"})
    assert r.status_code == 202
    body = r.json()
    assert set(body) == {"task_id", "status", "started_at", "started_at_local"}
    assert body["status"] in ("running", "completed")
    assert body["started_at"] and body["started_at_local"]
    view = await store_module.task_view(body["task_id"])
    assert view is not None and view.tool_name == "fake_blocking"
    assert view.params["q"] == "hello"
    assert view.parent_task_id is None


async def test_call_list_param_202(toolcall_client: SimpleNamespace) -> None:
    """A list value for a plain list param passes validation and starts a run."""
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.post("/api/tools/fake_blocking/call", json={"q": "x", "tags": ["a", "b"]})
    assert r.status_code == 202
    view = await store_module.task_view(r.json()["task_id"])
    assert view is not None and view.params["tags"] == ["a", "b"]


async def test_call_lax_number_is_stored_raw(toolcall_client: SimpleNamespace) -> None:
    """A number sent as a string starts the run and is stored raw, uncoerced."""
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.post("/api/tools/fake_blocking/call", json={"q": "x", "limit": "5"})
    assert r.status_code == 202
    view = await store_module.task_view(r.json()["task_id"])
    assert view is not None and view.params["limit"] == "5"


async def test_call_async_tool_202(toolcall_client: SimpleNamespace) -> None:
    assert _FakeAsyncTool.is_slow is True
    toolcall_client.register(_FakeAsyncTool)
    r = await toolcall_client.client.post("/api/tools/fake_async/call", json={"q": "later"})
    assert r.status_code == 202
    view = await store_module.task_view(r.json()["task_id"])
    assert view is not None and view.tool_name == "fake_async"
    record = await wait_record(r.json()["task_id"], "completed")
    assert record.status == "completed"


async def test_call_202_run_failure_is_recorded_not_logged(
    toolcall_client: SimpleNamespace, capsys: pytest.CaptureFixture
) -> None:
    """A run that fails after the 202 is kept on its own record and the handler logs nothing."""
    toolcall_client.register(_FakeFailingRunTool)
    r = await toolcall_client.client.post("/api/tools/fake_failing_run/call", json={"q": "boom"})
    assert r.status_code == 202
    assert r.json()["status"] in ("running", "failed")
    record = await wait_record(r.json()["task_id"], "failed")
    assert record.error == "run boom"
    assert capsys.readouterr().err == ""


# --- call endpoint, no run with 200 ---


async def test_call_cache_hit_returns_existing_task(toolcall_client: SimpleNamespace) -> None:
    """A repeat call inside the reuse window answers 202 like a fresh run, with no cached key.

    No new row is made.
    """
    toolcall_client.register(_FakeCachedTool)
    first = await toolcall_client.client.post("/api/tools/fake_cached/call", json={"q": "dup"})
    assert first.status_code == 202
    task_id = first.json()["task_id"]
    assert "cached" not in first.json()  # a fresh run's 202 carries no cached flag
    await wait_record(task_id, "completed")
    before = await store_module.total_count()
    r = await toolcall_client.client.post("/api/tools/fake_cached/call", json={"q": "dup"})
    assert r.status_code == 202  # a reused run answers 202 like a fresh one
    body = r.json()
    assert set(body) == {"task_id", "status", "started_at", "started_at_local"}  # no cached key
    assert body["task_id"] == task_id  # the original run, not a new one
    assert body["status"] == "completed"
    assert body["started_at"] and body["started_at_local"]
    assert await store_module.total_count() == before  # no new row


# --- call endpoint, 400 bad param ---


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({}, "Error: q: Missing required argument."),
        ({"q": 1}, "Error: q: Input should be a valid string."),
        ({"q": "x", "limit": "abc"}, "Error: limit: Input should be a valid integer, unable to parse string as an integer."),
        ({"q": "x", "limit": 1000}, "Error: limit: Input should be less than or equal to 100."),
        ({"q": "x", "mode": "bogus"}, "Error: mode: Input should be 'image' or 'video'."),
        ({"q": "x", "zz": 1}, "Error: zz: Unexpected keyword argument."),
        ({"q": "x", "tags": "a"}, "Error: tags: Input should be a valid list."),
    ],
)
async def test_call_validation_400(
    toolcall_client: SimpleNamespace, body: dict, expected: str
) -> None:
    """A bad set of params answers 400 with the tool's own words and makes no row."""
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.post("/api/tools/fake_blocking/call", json=body)
    assert r.status_code == 400
    assert r.json() == {"error": expected, "code": "bad_param"}
    assert await store_module.total_count() == 0


async def test_call_tool_tier_rejection_202(api_client: httpx.AsyncClient) -> None:
    """A rejection the tool makes in its own run starts a run that ends failed with its own words."""
    r = await api_client.post("/api/tools/github/call", json={"action": "view_file"})
    assert r.status_code == 202
    record = await wait_record(r.json()["task_id"], "failed")
    assert record.error == "github(view_file) requires repo, path"


async def test_call_nested_list_loc_dotted(toolcall_client: SimpleNamespace) -> None:
    """A bad item inside a list names its spot with a dot in the 400 message."""
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.post(
        "/api/tools/fake_blocking/call", json={"q": "x", "reference_images": [123]}
    )
    assert r.status_code == 400
    assert r.json()["code"] == "bad_param"
    assert r.json()["error"] == "Error: reference_images.0: Input should be a valid string."


async def test_call_invalid_json_body_400(toolcall_client: SimpleNamespace) -> None:
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.post(
        "/api/tools/fake_blocking/call", content="{nope",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400
    assert r.json() == {"error": "The request body must be valid JSON.", "code": "bad_param"}


async def test_call_non_object_body_400(toolcall_client: SimpleNamespace) -> None:
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.post(
        "/api/tools/fake_blocking/call", content="[1, 2]",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400
    assert r.json() == {"error": "The request body must be a JSON object of params.", "code": "bad_param"}


# --- call endpoint, 403, 404, 405 ---


async def test_call_not_exposed_403(toolcall_client: SimpleNamespace) -> None:
    toolcall_client.register(_FakeUntrackedTool)
    r = await toolcall_client.client.post("/api/tools/fake_untracked/call", json={})
    assert r.status_code == 403
    assert r.json() == {"error": "Tool 'fake_untracked' is not exposed in the console.", "code": "not_exposed"}


async def test_call_unknown_tool_404(toolcall_client: SimpleNamespace) -> None:
    r = await toolcall_client.client.post("/api/tools/no_such_tool/call", json={})
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


async def test_call_wrong_method_405(toolcall_client: SimpleNamespace) -> None:
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.get("/api/tools/fake_blocking/call")
    assert r.status_code == 405
    assert r.headers["allow"] == "POST"
    assert r.json()["code"] == "method_not_allowed"


# --- call endpoint, 413 and 415 ---


async def test_call_body_too_large_413(toolcall_client: SimpleNamespace) -> None:
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.post(
        "/api/tools/fake_blocking/call", json={"q": "a" * (10 * 2**20 + 1)}
    )
    assert r.status_code == 413
    assert r.json() == {"error": "Request body too large (max 10 MB).", "code": "body_too_large"}


async def test_call_non_json_media_type_415_before_body(toolcall_client: SimpleNamespace) -> None:
    """A call with a non JSON content type is refused even when the body is JSON."""
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.post(
        "/api/tools/fake_blocking/call", content=json.dumps({"q": "x"}),
        headers={"Content-Type": "text/plain"},
    )
    assert r.status_code == 415
    assert r.json()["code"] == "unsupported_media_type"


async def test_call_origin_mismatch_415(toolcall_client: SimpleNamespace) -> None:
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.post(
        "/api/tools/fake_blocking/call", json={"q": "x"}, headers={"Origin": "http://evil.example.com"}
    )
    assert r.status_code == 415
    assert r.json()["code"] == "unsupported_media_type"


async def test_call_json_with_charset_parameter_accepted(toolcall_client: SimpleNamespace) -> None:
    """A JSON content type that carries a charset is accepted."""
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.post(
        "/api/tools/fake_blocking/call", json={"q": "x"},
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert r.status_code == 202


async def test_call_same_origin_with_port_accepted(toolcall_client: SimpleNamespace) -> None:
    """An Origin that carries the port matches the request host and port."""
    toolcall_client.register(_FakeBlockingTool)
    app = toolcall_client.app  # the same app under a second address
    async with httpx.AsyncClient(transport=ASGITransport(app=app),
                                 base_url="http://agent.example.com:8315") as c:
        r = await c.post("/api/tools/fake_blocking/call", json={"q": "x"},
                         headers={"Origin": "http://agent.example.com:8315"})
    assert r.status_code == 202


# --- call endpoint, 500 ---


async def test_call_started_run_without_view_500(
    toolcall_client: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that starts but has no row answers 500 instead of pointing at a dead task."""
    toolcall_client.register(_FakeBlockingTool)

    async def _no_view(_task_id: str) -> None:
        return None

    monkeypatch.setattr(store_module, "task_view", _no_view)
    r = await toolcall_client.client.post("/api/tools/fake_blocking/call", json={"q": "x"})
    assert r.status_code == 500
    assert r.json() == {"error": "The tool call did not start a run.", "code": "internal"}


# --- failure logging ---


async def test_call_validation_typo_logs_nothing(toolcall_client: SimpleNamespace, capsys: pytest.CaptureFixture) -> None:
    """A bad param answers 400 and logs nothing."""
    toolcall_client.register(_FakeBlockingTool)
    r = await toolcall_client.client.post("/api/tools/fake_blocking/call", json={"q": 1})
    assert r.status_code == 400
    assert capsys.readouterr().err == ""


async def test_call_failure_records_a_failed_run(toolcall_client: SimpleNamespace) -> None:
    """A tool whose call raises answers 202 and is recorded as a failed run."""
    toolcall_client.register(_FakeRaisingTool)
    r = await toolcall_client.client.post("/api/tools/fake_raising/call", json={"q": "x"})
    assert r.status_code == 202
    task_id = r.json()["task_id"]
    await wait_record(task_id, "failed")


# --- nesting and concurrency ---


async def test_nested_run_is_a_subtask_of_the_answered_run(toolcall_client: SimpleNamespace) -> None:
    """The 202 names the outer run, and the nested run shows up as its subtask."""
    toolcall_client.register(_FakeBlockingTool)
    toolcall_client.register_outer()
    r = await toolcall_client.client.post("/api/tools/fake_outer/call", json={"q": "nest"})
    assert r.status_code == 202
    outer_id = r.json()["task_id"]
    view = await store_module.task_view(outer_id)
    assert view is not None and view.tool_name == "fake_outer"
    assert view.parent_task_id is None
    kids = await wait_subtasks(outer_id)  # the subtask row lands after the 202
    assert [k.tool_name for k in kids] == ["fake_blocking"]


async def test_concurrent_calls_get_their_own_runs(
    toolcall_client: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three calls made at the same time each get their own run, from the console and the wire."""
    async def _call(self: Tool, params: dict) -> ToolResult:
        return ToolResult(text="fetch result: " + params["url"])

    monkeypatch.setattr(type(web_fetch_tool), "call", _call)
    a, b, direct = await asyncio.gather(
        toolcall_client.client.post("/api/tools/web_fetch/call", json={"url": "http://c-one.example"}),
        toolcall_client.client.post("/api/tools/web_fetch/call", json={"url": "http://c-two.example"}),
        mcp.call_tool("web_fetch", {"url": "http://c-three.example"}),
    )
    assert a.status_code == 202 and b.status_code == 202
    assert direct.content[0].text == "fetch result: http://c-three.example"
    stats = [s for s in await store_module.tool_stats() if s["tool_name"] == "web_fetch"]
    assert sum(s["completed"] + s["running"] + s["failed"] + s["cancelled"] for s in stats) == 3
    views = [await store_module.task_view(tid) for tid in
             (a.json()["task_id"], b.json()["task_id"])]
    assert all(v is not None for v in views)  # both console 202 ids are real rows
    assert a.json()["task_id"] != b.json()["task_id"]


# --- MCP-path untracked ---


async def test_untracked_mcp_call_creates_no_run(
    toolcall_client: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An untracked tool makes no run row and turns a failure into plain text."""
    async def _ok(self: Tool, _params: dict) -> ToolResult:
        return ToolResult(text="untracked result")

    async def _boom(self: Tool, _params: dict) -> ToolResult:
        raise RuntimeError("untracked boom")

    monkeypatch.setattr(type(present_to_user_tool), "call", _ok)
    monkeypatch.setattr(type(send_or_receive_file_tool), "call", _boom)
    before = await store_module.total_count()
    result = await mcp.call_tool("present_to_user", {"title": "x"})
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "untracked result"
    runs = os.path.join(toolcall_client.data_dir, "runs")
    if os.path.isdir(runs):
        assert os.listdir(runs) == []
    assert await store_module.total_count() == before
    failed = await mcp.call_tool("send_or_receive_file", {})
    assert isinstance(failed.content[0], TextContent)
    assert failed.content[0].text == "Tool 'send_or_receive_file' failed: untracked boom"
    assert await store_module.total_count() == before


# --- declared bounds ---


@pytest.mark.parametrize(
    ("tool_name", "param", "minimum"),
    [
        ("github", "limit", 1),
        ("github", "number", 1),
        ("rednote_research", "count", 1),
    ],
)
async def test_served_schema_carries_declared_minimums(
    api_client: httpx.AsyncClient, tool_name: str, param: str, minimum: int
) -> None:
    """A param's declared smallest value reaches the console schema."""
    r = await api_client.get(f"/api/tools/{tool_name}/schema")
    assert r.status_code == 200
    by_name = {p["name"]: p for p in r.json()["params"]}
    assert by_name[param]["min"] == minimum


# --- the served schema ---


@pytest.mark.parametrize("name", sorted(n for n, cls in registered_tools().items() if cls.tracked))
async def test_served_schema_never_has_double_backtick(api_client: httpx.AsyncClient, name: str) -> None:
    """No served description carries a double backtick."""
    r = await api_client.get(f"/api/tools/{name}/schema")
    assert r.status_code == 200
    body = r.json()
    assert "``" not in body["description"], name
    assert all("``" not in p["description"] for p in body["params"]), name


# --- the not-exposed trio ---


@pytest.mark.parametrize("name", ["task", "send_or_receive_file", "present_to_user"])
async def test_not_exposed_schema_call_403(api_client: httpx.AsyncClient, name: str) -> None:
    """Schema and call both answer 403 for a tool the console does not show."""
    r = await api_client.get(f"/api/tools/{name}/schema")
    assert r.status_code == 403 and r.json()["code"] == "not_exposed"
    r = await api_client.post(f"/api/tools/{name}/call", json={})
    assert r.status_code == 403 and r.json()["code"] == "not_exposed"


@pytest.mark.live
async def test_send_or_receive_file_creates_no_row(toolcall_client: SimpleNamespace) -> None:
    """Calling send_or_receive_file makes no run row and no run folder."""
    before = await store_module.total_count()
    result = await mcp.call_tool("send_or_receive_file", {})
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text.strip()  # the WebDAV transfer instructions
    runs = os.path.join(toolcall_client.data_dir, "runs")
    if os.path.isdir(runs):
        assert os.listdir(runs) == []
    assert await store_module.total_count() == before
