"""Tests for console tools, tasks, run files, cancellation, and change polling endpoints."""

from __future__ import annotations

import asyncio
import io
import os
from collections.abc import Coroutine
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from starlette.responses import FileResponse

import app.console.tasks_api as tasks_api_module
import app.lib.db.store as store_module
from app.lib.task import task_manager
from app.mcp.registry import registered_tools, tool_class
from tests.app.conftest import (
    E2E_ENDED_AT,
    E2E_PNG,
    E2E_RESULT_BIG,
    E2E_RUNNING,
    E2E_STARTED_AT,
    log_lines,
    seed_task,
    write_resource_file,
    write_run_file,
)
from tests.app.console.conftest import wait_record

_RESULT_BYTES_0797 = 793191
_RESULT_LINE_BYTES = 107
_RESULT_LINE_PAD = 89
_LOG238_ID = "e2e-log-238"
_LOG238_LINES = 238
_RETIRED_TOOL = "retired_tool"
_RETIRED_TASK_ID = "retired-result-1"
_RETIRED_FILE = "media_gen/gen_001.png"
_RETIRED_RESULT = "Saved: media_gen/gen_001.png\n"
_BEFORE_E2E_RUNS = "2026-08-22T23:00:00+00:00"


async def _seed_brave_task(
    task_id: str,
    started_at: str,
    query_index: int,
    status: str | None,
    *,
    error: str | None = None,
    ended_at: str | None = None,
) -> None:
    await store_module.task_store.insert_running(task_id, None, started_at, f"runs/{task_id}")
    await store_module.tool_store.insert(task_id, "brave_search", None, {"q": query_index})
    if status is not None:
        await store_module.task_store.finish(task_id, status, error=error, ended_at=ended_at)


def _build_result_content(size: int) -> str:
    line = "e2e-result 000000" + "a" * _RESULT_LINE_PAD + "\n"
    return line * (size // _RESULT_LINE_BYTES)


async def _reassemble_result_windows(client: httpx.AsyncClient, task_id: str) -> tuple[bytes, int]:
    collected = b""
    offset = 0
    window_count = 0
    for _ in range(64):
        body = (
            await client.get(f"/api/tasks/{task_id}/result", params={"offset": offset, "max_bytes": 524288})
        ).json()
        collected += body["content"].encode("utf-8")
        offset = body["offset"] + body["content_bytes"]
        window_count += 1
        if not body["truncated"] or offset >= body["size"]:
            break
    return collected, window_count


async def test_tool_tasks_keyset(api_client: httpx.AsyncClient) -> None:
    """Local times are exact strings in the container time zone, in summer and in winter."""
    await _seed_brave_task("a", "2026-08-23T01:00:00+00:00", 1, "completed", ended_at="2026-08-23T01:00:05+00:00")
    await _seed_brave_task(
        "b", "2026-08-23T02:00:00+00:00", 2, "failed", error="boom", ended_at="2026-08-23T02:00:05+00:00"
    )
    await _seed_brave_task("c", "2026-08-23T03:00:00+00:00", 3, None)
    await _seed_brave_task("d", "2026-01-15T08:30:00+00:00", 4, "cancelled", ended_at="2026-01-15T08:30:05+00:00")

    first_response = await api_client.get("/api/tools/brave_search/tasks", params={"limit": 2})
    assert first_response.status_code == 200
    body = first_response.json()
    assert [task["task_id"] for task in body["tasks"]] == ["c", "b"]
    assert body["next_cursor"] == "2026-08-23T02:00:00+00:00|b"
    assert body["tasks"][0]["tool"] == "brave_search"
    assert body["tasks"][0]["params"] == {"q": 3}
    assert body["tasks"][0]["started_at"] == "2026-08-23T03:00:00+00:00"
    assert body["tasks"][0]["started_at_local"] == "2026-08-22 20:00:00"
    assert body["tasks"][1]["status"] == "failed"
    assert body["tasks"][1]["ended_at"] == "2026-08-23T02:00:05+00:00"
    assert body["tasks"][1]["ended_at_local"] == "2026-08-22 19:00:05"
    second_response = await api_client.get(
        "/api/tools/brave_search/tasks", params={"limit": 2, "cursor": body["next_cursor"]}
    )
    second_body = second_response.json()
    assert [task["task_id"] for task in second_body["tasks"]] == ["a", "d"]
    assert second_body["next_cursor"] is None
    assert {task["status"] for task in second_body["tasks"]} == {"completed", "cancelled"}
    assert second_body["tasks"][1]["ended_at"] == "2026-01-15T08:30:05+00:00"
    assert second_body["tasks"][1]["ended_at_local"] == "2026-01-15 00:30:05"


async def test_tool_tasks_unknown_tool(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/api/tools/does_not_exist/tasks")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


async def test_tool_tasks_bad_limit(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/api/tools/brave_search/tasks", params={"limit": 9999})
    assert response.status_code == 400
    assert response.json()["code"] == "bad_param"


async def test_tool_tasks_corrupt_cursor(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/api/tools/brave_search/tasks", params={"cursor": "garbage-no-pipe"})
    assert response.status_code == 400
    body = response.json()
    assert set(body) == {"error", "code"}
    assert body["code"] == "bad_param"
    empty_response = await api_client.get("/api/tools/brave_search/tasks", params={"cursor": ""})
    assert empty_response.status_code == 400
    assert empty_response.json()["code"] == "bad_param"


@pytest.mark.parametrize("name", ["brave_search", "github"])
async def test_tool_tasks_description_is_summary(api_client: httpx.AsyncClient, name: str) -> None:
    response = await api_client.get(f"/api/tools/{name}/tasks")
    assert response.status_code == 200
    assert response.json()["description"] == tool_class(name).summary


async def test_tool_tasks_description_not_exposed(api_client: httpx.AsyncClient) -> None:
    """A tool the console does not show still has its summary in the task list."""
    response = await api_client.get("/api/tools/task/tasks")
    assert response.status_code == 200
    assert response.json()["description"] == tool_class("task").summary


async def test_tools_roster_and_stats(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("a", None, "2026-08-23T01:00:00+00:00", "runs/a")
    await store_module.tool_store.insert("a", "brave_search", None, {"q": 1})
    await store_module.task_store.finish("a", "completed", ended_at="2026-08-23T01:00:05+00:00")
    await store_module.task_store.insert_running("b", None, "2026-08-23T02:00:00+00:00", "runs/b")
    await store_module.tool_store.insert("b", "brave_search", None, {"q": 2})
    await store_module.task_store.finish("b", "completed", ended_at="2026-08-23T02:00:05+00:00")
    await store_module.task_store.insert_running("c", None, "2026-08-23T03:00:00+00:00", "runs/c")
    await store_module.tool_store.insert("c", "brave_search", None, {"q": 3})

    response = await api_client.get("/api/tools")
    assert response.status_code == 200
    body = response.json()
    assert body["generated_at"].endswith("+00:00")
    assert "generated_at_local" in body
    by_name = {tool["name"]: tool for tool in body["tools"]}
    tracked = {name for name, tool_type in registered_tools().items() if tool_type.is_tracked is True}
    assert set(by_name) == tracked
    brave_stats = by_name["brave_search"]
    assert brave_stats["description"] == tool_class("brave_search").summary
    assert (
        brave_stats["completed"],
        brave_stats["running"],
        brave_stats["failed"],
        brave_stats["cancelled"],
    ) == (2, 1, 0, 0)
    assert brave_stats["last_activity"] == "2026-08-23T03:00:00+00:00"
    assert brave_stats["running_since"] == "2026-08-23T03:00:00+00:00"
    assert "task" not in by_name
    assert "send_or_receive_file" not in by_name
    assert "present_to_user" not in by_name


async def test_not_exposed_tools_absent_from_roster(api_client: httpx.AsyncClient) -> None:
    """The list holds the 16 tracked tools and leaves out untracked tools."""
    tracked = {name for name, tool_type in registered_tools().items() if tool_type.is_tracked is True}
    assert len(tracked) == 16
    names = {tool["name"] for tool in (await api_client.get("/api/tools")).json()["tools"]}
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
    with open(os.path.join(data_dir, "runs/parent/log"), "w", encoding="utf-8") as file_handle:
        file_handle.write("hello\n")

    response = await api_client.get("/api/tasks/parent")
    assert response.status_code == 200
    body = response.json()
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
    response = await api_client.get("/api/tasks/does-not-exist")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


async def test_log_forward_and_backward(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("t1", None, "2026-08-23T01:00:00+00:00", "runs/t1")
    await store_module.tool_store.insert("t1", "brave_search", None, {"q": 1})
    log_dir = os.path.join(store_module.task_store.data_dir, "runs/t1")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "log"), "w", encoding="utf-8") as file_handle:
        file_handle.write("l1\nl2\nl3\n")
    backward = (await api_client.get("/api/tasks/t1/log", params={"lines": 200})).json()
    assert backward["content"] == "l1\nl2\nl3\n" and backward["total_lines"] == 3 and backward["offset"] == 0
    assert backward["head_reached"] is True
    forward = (await api_client.get("/api/tasks/t1/log", params={"offset": 0})).json()
    assert forward["content"] == "l1\nl2\nl3\n" and forward["content_bytes"] == 9
    assert forward["size"] == 9 and forward["truncated"] is False


async def test_log_not_found_no_record(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/api/tasks/nope/log")
    assert response.status_code == 404 and response.json()["code"] == "not_found"


async def test_log_not_found_missing_file(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("t2", None, "2026-08-23T01:00:00+00:00", "runs/t2")
    await store_module.tool_store.insert("t2", "brave_search", None, {"q": 1})
    response = await api_client.get("/api/tasks/t2/log")
    assert response.status_code == 404 and response.json()["code"] == "log_not_found"


async def test_result_window(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("t1", None, "2026-08-23T01:00:00+00:00", "runs/t1")
    await store_module.tool_store.insert("t1", "brave_search", None, {"q": 1})
    await store_module.task_store.finish("t1", "completed", ended_at="2026-08-23T01:00:05+00:00")
    run_dir = os.path.join(store_module.task_store.data_dir, "runs/t1")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result"), "w", encoding="utf-8") as file_handle:
        file_handle.write("RESULT-CONTENT")
    full_body = (await api_client.get("/api/tasks/t1/result", params={"offset": 0})).json()
    assert full_body["content"] == "RESULT-CONTENT" and full_body["content_bytes"] == 14
    assert full_body["size"] == 14 and full_body["truncated"] is False
    short_body = (await api_client.get("/api/tasks/t1/result", params={"offset": 0, "max_bytes": 4})).json()
    assert short_body["content"] == "RESU" and short_body["truncated"] is True


async def test_result_not_found_when_running(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("t2", None, "2026-08-23T01:00:00+00:00", "runs/t2")
    await store_module.tool_store.insert("t2", "brave_search", None, {"q": 1})
    response = await api_client.get("/api/tasks/t2/result")
    assert response.status_code == 404 and response.json()["code"] == "result_not_found"


async def test_result_not_found_unknown(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/api/tasks/does-not-exist/result")
    assert response.status_code == 404 and response.json()["code"] == "not_found"


@pytest.mark.parametrize(
    ("url", "offset"),
    [
        ("/api/tasks/t1/log", "9223372036854775808"),
        ("/api/tasks/t1/log", "999999999999999999999999999999"),
        ("/api/tasks/t1/result", "9223372036854775808"),
        ("/api/tasks/t1/result", "999999999999999999999999999999"),
    ],
)
async def test_offset_beyond_bound_returns_bad_param(
    api_client: httpx.AsyncClient, url: str, offset: str
) -> None:
    """An offset of 2**63 or more answers 400 bad_param."""
    response = await api_client.get(url, params={"offset": offset})
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "error": "offset must be an integer between 0 and 1099511627776 and max_bytes a positive integer.",
        "code": "bad_param",
    }
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"


async def test_offset_int64_danger_zone_returns_bad_param(api_client: httpx.AsyncClient) -> None:
    """An offset just under 2**63 answers 400 bad_param, not a missing file 404."""
    response = await api_client.get("/api/tasks/t1/log", params={"offset": "9223372036854775806"})
    assert response.status_code == 400
    assert response.json()["code"] == "bad_param"


async def test_offset_just_above_bound_returns_bad_param(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/api/tasks/t1/log", params={"offset": "1099511627777"})
    assert response.status_code == 400
    assert response.json()["code"] == "bad_param"


async def test_offset_past_eof_returns_empty_window(api_client: httpx.AsyncClient) -> None:
    """An offset past the end of the file answers 200 with empty content."""
    await _seed_brave_task("t1", "2026-08-23T01:00:00+00:00", 1, None)
    run_dir = os.path.join(store_module.task_store.data_dir, "runs/t1")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "log"), "w", encoding="utf-8") as file_handle:
        file_handle.write("l1\nl2\nl3\n")
    response = await api_client.get("/api/tasks/t1/log", params={"offset": 4294967296})
    assert response.status_code == 200
    body = response.json()
    assert body["offset"] == 4294967296
    assert body["content"] == "" and body["content_bytes"] == 0 and body["truncated"] is False


async def test_end_offset_beyond_bound_returns_bad_param(api_client: httpx.AsyncClient) -> None:
    """A huge end_offset answers 400 bad_param."""
    response = await api_client.get(
        "/api/tasks/t1/log", params={"lines": 5, "end_offset": "999999999999999999999999999999"}
    )
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "error": "lines must be an integer and end_offset an integer between 0 and 1099511627776.",
        "code": "bad_param",
    }


async def test_log_download(api_client: httpx.AsyncClient) -> None:
    await _seed_brave_task("t1", "2026-08-23T01:00:00+00:00", 1, None)
    run_dir = os.path.join(store_module.task_store.data_dir, "runs/t1")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "log"), "w", encoding="utf-8") as file_handle:
        file_handle.write("LOG-LINES\n")
    response = await api_client.get("/api/tasks/t1/log/download")
    assert response.status_code == 200
    assert response.text == "LOG-LINES\n"
    assert response.headers["content-type"].startswith("text/plain")
    assert "attachment" in response.headers["content-disposition"]
    assert "t1_log" in response.headers["content-disposition"]


async def test_log_download_missing_file(api_client: httpx.AsyncClient) -> None:
    await _seed_brave_task("t2", "2026-08-23T01:00:00+00:00", 1, None)
    response = await api_client.get("/api/tasks/t2/log/download")
    assert response.status_code == 404 and response.json()["code"] == "log_not_found"


async def test_result_download_range(api_client: httpx.AsyncClient) -> None:
    await _seed_brave_task("t1", "2026-08-23T01:00:00+00:00", 1, "completed", ended_at="2026-08-23T01:00:05+00:00")
    run_dir = os.path.join(store_module.task_store.data_dir, "runs/t1")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result"), "w", encoding="utf-8") as file_handle:
        file_handle.write("0123456789")
    response = await api_client.get("/api/tasks/t1/result/download")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert "t1_result" in response.headers["content-disposition"]
    range_response = await api_client.get("/api/tasks/t1/result/download", headers={"Range": "bytes=2-5"})
    assert range_response.status_code == 206
    assert range_response.text == "2345"


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
    await _seed_brave_task("t1", "2026-08-23T01:00:00+00:00", 1, None)
    run_dir = os.path.join(store_module.task_store.data_dir, "runs/t1")
    os.makedirs(run_dir, exist_ok=True)
    file_path = os.path.join(run_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("data\n")

    real_open = open

    def vanish_open(file: str, *args: str, **kwargs: str) -> io.BinaryIO:
        if file == file_path:
            os.remove(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(tasks_api_module, "open", vanish_open, raising=False)
    response = await api_client.get(url, params=params)
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["code"] == code
    clean = await api_client.get(url, params=params)
    assert (response.status_code, response.json()) == (clean.status_code, clean.json())


async def test_download_passes_prechecked_stat_to_file_response(
    api_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both downloads hand the file size they already read to the response."""
    await _seed_brave_task("t1", "2026-08-23T01:00:00+00:00", 1, "completed", ended_at="2026-08-23T01:00:05+00:00")
    run_dir = os.path.join(store_module.task_store.data_dir, "runs/t1")
    os.makedirs(run_dir, exist_ok=True)
    log_bytes = b"LOG-LINES\n"
    result_bytes = b"RESULT-CONTENT"
    with open(os.path.join(run_dir, "log"), "wb") as file_handle:
        file_handle.write(log_bytes)
    with open(os.path.join(run_dir, "result"), "wb") as file_handle:
        file_handle.write(result_bytes)

    stats: list[os.stat_result] = []
    real_file_response = tasks_api_module.FileResponse

    def capturing_file_response(*args: str, **kwargs: str | os.stat_result) -> FileResponse:
        stats.append(kwargs["stat_result"])
        return real_file_response(*args, **kwargs)

    monkeypatch.setattr(tasks_api_module, "FileResponse", capturing_file_response)

    log_response = await api_client.get("/api/tasks/t1/log/download")
    assert log_response.status_code == 200
    assert log_response.text == log_bytes.decode()
    result_response = await api_client.get("/api/tasks/t1/result/download")
    assert result_response.status_code == 200
    assert all(isinstance(stat_result, os.stat_result) for stat_result in stats)
    assert [stat_result.st_size for stat_result in stats] == [len(log_bytes), len(result_bytes)]


async def test_changes_baseline_and_304(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("r1", None, "2026-08-23T01:00:00+00:00", "runs/r1")
    await store_module.tool_store.insert("r1", "brave_search", None, {"q": 1})
    await store_module.task_store.insert_running("d1", None, "2026-08-23T00:00:00+00:00", "runs/d1")
    await store_module.tool_store.insert("d1", "brave_search", None, {"q": 2})
    await store_module.task_store.finish("d1", "completed", ended_at="2026-08-23T00:00:05+00:00")

    body = (await api_client.get("/api/changes")).json()
    assert body["fingerprint"].endswith("|1|")
    assert [task["task_id"] for task in body["running"]] == ["r1"]
    assert body["started"] == [] and body["finished"] == []
    assert body["stats"]["running"] == 1
    assert body["stats"]["tools"]["brave_search"]["completed"] == 1
    assert body["stats"]["tools"]["brave_search"]["running_since"] == "2026-08-23T01:00:00+00:00"
    second_response = await api_client.get(
        "/api/changes", params={"since_fp": body["fingerprint"], "since_gen": body["generated_at"]}
    )
    assert second_response.status_code == 304


async def test_changes_started_and_finished(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("x1", None, "2026-08-23T00:00:00+00:00", "runs/x1")
    await store_module.tool_store.insert("x1", "brave_search", None, {"q": 1})
    await store_module.task_store.finish("x1", "completed", ended_at="2026-08-23T00:00:05+00:00")
    body = (
        await api_client.get(
            "/api/changes", params={"since_fp": "no-match", "since_gen": "2026-08-22T23:00:00+00:00"}
        )
    ).json()
    assert [task["task_id"] for task in body["started"]] == ["x1"]
    assert [task["task_id"] for task in body["finished"]] == ["x1"]
    assert body["running"] == []


async def test_changes_running_to_finished(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("n1", None, "2026-08-23T00:00:00+00:00", "runs/n1")
    await store_module.tool_store.insert("n1", "brave_search", None, {"q": 1})
    first_body = (
        await api_client.get(
            "/api/changes", params={"since_fp": "no-match", "since_gen": "2026-08-22T23:00:00+00:00"}
        )
    ).json()
    assert [task["task_id"] for task in first_body["started"]] == ["n1"]
    assert [task["task_id"] for task in first_body["running"]] == ["n1"]
    assert first_body["running"][0]["status"] == "running"
    await store_module.task_store.finish("n1", "completed", ended_at=datetime.now(timezone.utc).isoformat())
    second_body = (
        await api_client.get(
            "/api/changes", params={"since_fp": first_body["fingerprint"], "since_gen": first_body["generated_at"]}
        )
    ).json()
    assert [task["task_id"] for task in second_body["finished"]] == ["n1"]
    assert second_body["finished"][0]["status"] == "completed"
    assert second_body["running"] == []


async def test_cancel_unknown_task(api_client: httpx.AsyncClient) -> None:
    response = await api_client.post("/api/tasks/does-not-exist/cancel")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


@pytest.mark.parametrize("method", ["GET", "HEAD", "PUT", "DELETE", "PATCH"])
async def test_cancel_wrong_method_returns_405(api_client: httpx.AsyncClient, method: str) -> None:
    """A method other than POST on cancel answers 405 and allows only POST."""
    response = await api_client.request(method, "/api/tasks/abc/cancel")
    assert response.status_code == 405
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["allow"] == "POST"
    assert response.headers["x-frame-options"] == "DENY"
    if method != "HEAD":
        assert response.json() == {
            "error": f"Method '{method}' is not allowed at '/api/tasks/abc/cancel'.",
            "code": "method_not_allowed",
        }


async def test_cancel_already_terminal(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("d", None, "2026-08-23T01:00:00+00:00", "runs/d")
    await store_module.task_store.finish("d", "completed", ended_at="2026-08-23T01:00:05+00:00")
    response = await api_client.post("/api/tasks/d/cancel")
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "already_terminal"
    assert body["status"] == "completed"


async def test_cancel_no_live_task(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running("n", None, "2026-08-23T01:00:00+00:00", "runs/n")
    response = await api_client.post("/api/tasks/n/cancel")
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "no_live_task"
    assert body["status"] == "running"


async def test_cancel_live_task_returns_202(api_client: httpx.AsyncClient) -> None:
    async def slow() -> None:
        await asyncio.sleep(30)

    task = await task_manager.create_task(slow)
    await wait_record(str(task.id), "running")

    response = await api_client.post(f"/api/tasks/{task.id}/cancel")
    assert response.status_code == 202
    assert response.json() == {"task_id": str(task.id), "status": "running"}

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

    live_task = await task_manager.create_task(slow)
    await wait_record(str(live_task.id), "running")

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
        response = await api_client.post(f"/api/tasks/{live_task.id}/cancel")
        assert response.status_code == 202
        assert response.json() == {"task_id": str(live_task.id), "status": "running"}
        worker = next(task for task in created if task.get_name() == f"cancel-{live_task.id}")
        done, _pending = await asyncio.wait({worker}, timeout=2)
        assert worker in done
        assert worker in retrieved
    finally:
        loop.set_task_factory(None)
    await live_task.cancel()
    await wait_record(str(live_task.id), "cancelled")


async def test_l2_renders_rednote_rows(api_client: httpx.AsyncClient) -> None:
    """Four seeded runs fit one page with no next cursor.

    An unknown tool answers 404.
    """
    await seed_task("e2e-rednote-running", "rednote_research")
    await seed_task(
        "e2e-rednote-done-1",
        "rednote_research",
        "2026-08-23T00:00:00+00:00",
        "completed",
        ended_at="2026-08-23T00:00:05+00:00",
    )
    await seed_task(
        "e2e-rednote-done-2",
        "rednote_research",
        "2026-08-22T23:00:00+00:00",
        "completed",
        ended_at="2026-08-22T23:00:05+00:00",
    )
    await seed_task(
        "e2e-rednote-failed",
        "rednote_research",
        "2026-08-22T22:00:00+00:00",
        "failed",
        error="boom",
        ended_at="2026-08-22T22:00:05+00:00",
    )
    body = (await api_client.get("/api/tools/rednote_research/tasks", params={"limit": 200})).json()
    assert len(body["tasks"]) == 4
    assert body["next_cursor"] is None
    assert {task["status"] for task in body["tasks"]} == {"running", "completed", "failed"}
    response = await api_client.get("/api/tools/no_such_tool/tasks")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


async def test_detail_renders_running_fixture(api_client: httpx.AsyncClient) -> None:
    """A running parent shows its status and its two subtasks.

    Cancel answers 409 and an unknown id answers 404.
    """
    await seed_task(E2E_RUNNING, "rednote_research")
    await seed_task(
        "e2e-running-sub-1", "rednote_research", "2026-08-23T01:00:01+00:00", parent_task_id=E2E_RUNNING
    )
    await seed_task(
        "e2e-running-sub-2", "rednote_research", "2026-08-23T01:00:02+00:00", parent_task_id=E2E_RUNNING
    )
    body = (await api_client.get(f"/api/tasks/{E2E_RUNNING}")).json()
    assert body["task"]["status"] == "running"
    assert len(body["subtasks"]) == 2
    assert all(subtask["parent_task_id"] == E2E_RUNNING for subtask in body["subtasks"])
    cancel_response = await api_client.post(f"/api/tasks/{E2E_RUNNING}/cancel")
    assert cancel_response.status_code == 409
    assert cancel_response.json()["code"] == "no_live_task"
    missing_response = await api_client.get("/api/tasks/does-not-exist")
    assert missing_response.status_code == 404
    assert missing_response.json()["code"] == "not_found"


async def test_result_fixture_windows(api_client: httpx.AsyncClient) -> None:
    """A 793,191 byte result is cut short in a 4096 byte window and downloads whole."""
    await seed_task(E2E_RESULT_BIG, "generate_image", E2E_STARTED_AT, "completed", ended_at=E2E_ENDED_AT)
    write_run_file(
        store_module.task_store.data_dir, E2E_RESULT_BIG, "result", _build_result_content(_RESULT_BYTES_0797)
    )
    window_body = (
        await api_client.get(f"/api/tasks/{E2E_RESULT_BIG}/result", params={"offset": 0, "max_bytes": 4096})
    ).json()
    assert window_body["size"] == _RESULT_BYTES_0797
    assert window_body["truncated"] is True
    assert len(window_body["content"].encode()) <= 4096
    download_response = await api_client.get(f"/api/tasks/{E2E_RESULT_BIG}/result/download")
    assert download_response.status_code == 200
    assert len(download_response.content) == _RESULT_BYTES_0797


async def test_result_multiwindow_reassembly(api_client: httpx.AsyncClient) -> None:
    """Reading the 793,191 byte result window by window rebuilds it byte for byte."""
    await seed_task(E2E_RESULT_BIG, "generate_image", E2E_STARTED_AT, "completed", ended_at=E2E_ENDED_AT)
    write_run_file(
        store_module.task_store.data_dir, E2E_RESULT_BIG, "result", _build_result_content(_RESULT_BYTES_0797)
    )
    download_response = await api_client.get(f"/api/tasks/{E2E_RESULT_BIG}/result/download")
    assert download_response.status_code == 200
    collected, window_count = await _reassemble_result_windows(api_client, E2E_RESULT_BIG)
    assert window_count >= 2
    assert len(collected) == _RESULT_BYTES_0797
    assert collected == download_response.content


async def test_log_fixture_backward_window(api_client: httpx.AsyncClient) -> None:
    """A 238 line log counts 238 lines and does not reach the head.

    Reading it forward is not cut short.
    """
    await seed_task(_LOG238_ID, "rednote_research")
    write_run_file(store_module.task_store.data_dir, _LOG238_ID, "log", log_lines(_LOG238_LINES))
    backward = (
        await api_client.get(f"/api/tasks/{_LOG238_ID}/log", params={"lines": 200, "end_offset": 0})
    ).json()
    assert backward["total_lines"] == _LOG238_LINES
    assert backward["head_reached"] is False
    assert backward["content_bytes"] > 0
    forward = (
        await api_client.get(f"/api/tasks/{_LOG238_ID}/log", params={"offset": 0, "max_bytes": 1048576})
    ).json()
    assert forward["truncated"] is False


async def test_unregistered_tool_run_stays_readable(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run of a tool that is no longer registered is listed and viewable with its files.

    It cannot run again, and the tool has no page of its own.
    """
    resources = tmp_path / "resources"
    write_resource_file(resources, _RETIRED_FILE, E2E_PNG)
    monkeypatch.setenv("RESOURCES_DIR", str(resources))
    await seed_task(
        _RETIRED_TASK_ID,
        _RETIRED_TOOL,
        E2E_STARTED_AT,
        "completed",
        params={"prompt": "a tea room"},
        ended_at=E2E_ENDED_AT,
    )
    write_run_file(store_module.task_store.data_dir, _RETIRED_TASK_ID, "result", _RETIRED_RESULT)
    write_run_file(store_module.task_store.data_dir, _RETIRED_TASK_ID, "files", _RETIRED_FILE + "\n")
    changes = (
        await api_client.get("/api/changes", params={"since_fp": "no-match", "since_gen": _BEFORE_E2E_RUNS})
    ).json()
    assert [(task["task_id"], task["tool"]) for task in changes["finished"]] == [(_RETIRED_TASK_ID, _RETIRED_TOOL)]
    detail = (await api_client.get(f"/api/tasks/{_RETIRED_TASK_ID}")).json()
    assert detail["task"]["tool"] == _RETIRED_TOOL
    assert detail["task"]["status"] == "completed"
    assert detail["task"]["rerunnable"] is False
    assert detail["result_files"] == [_RETIRED_FILE]
    result = (await api_client.get(f"/api/tasks/{_RETIRED_TASK_ID}/result")).json()
    assert result["content"] == _RETIRED_RESULT
    feed = (await api_client.get(f"/api/tasks/{_RETIRED_TASK_ID}/result/feed")).json()
    assert [(file_item["path"], file_item["present"]) for file_item in feed["files"]] == [(_RETIRED_FILE, True)]
    response = await api_client.get(f"/api/tools/{_RETIRED_TOOL}/tasks")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


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
    started_at = "2026-08-23T01:00:00+00:00"
    await store_module.task_store.insert_running("parent", None, started_at, "runs/parent")
    await store_module.tool_store.insert("parent", "brave_search", None, {"query": "x"})
    await store_module.task_store.insert_running("child1", "parent", started_at, "runs/child1")
    await store_module.task_store.insert_running("child2", "parent", started_at, "runs/child2")
    await store_module.task_store.insert_running("grandchild", "child1", started_at, "runs/grandchild")
    body = (await api_client.get("/api/tasks/parent")).json()
    subtask_ids = [subtask["task_id"] for subtask in body["subtasks"]]
    assert subtask_ids == ["child1", "child2"]
    assert not any("subtasks" in subtask for subtask in body["subtasks"])


async def test_detail_ancestors(api_client: httpx.AsyncClient) -> None:
    started_at = "2026-08-23T01:00:00+00:00"
    await store_module.task_store.insert_running("root", None, started_at, "runs/root")
    await store_module.tool_store.insert("root", "brave_search", None, {"query": "x"})
    await store_module.task_store.insert_running("mid", "root", started_at, "runs/mid")
    await store_module.task_store.insert_running("leaf", "mid", started_at, "runs/leaf")
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
    found = [task for task in tasks if task["task_id"] == "chg-agent"]
    assert found and found[0]["agent_name"] == "drafter"


async def test_detail_short_id_unique_prefix(api_client: httpx.AsyncClient) -> None:
    full_id = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
    await store_module.task_store.insert_running(full_id, None, "2026-08-23T01:00:00+00:00", "runs/" + full_id)
    body = (await api_client.get("/api/tasks/3f2504e0")).json()
    assert body["task"]["task_id"] == full_id


async def test_detail_short_id_ambiguous_returns_404(api_client: httpx.AsyncClient) -> None:
    await store_module.task_store.insert_running(
        "3f2504e0-1111-1111-1111-111111111111", None, "2026-08-23T01:00:00+00:00", "runs/a"
    )
    await store_module.task_store.insert_running(
        "3f2504e0-2222-2222-2222-222222222222", None, "2026-08-23T01:00:00+00:00", "runs/b"
    )
    response = await api_client.get("/api/tasks/3f2504e0")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


async def test_detail_unknown_id_returns_404(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/api/tasks/00000000")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
