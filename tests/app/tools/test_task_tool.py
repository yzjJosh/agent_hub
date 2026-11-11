import asyncio
import os
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import ClassVar, Protocol, override

import pytest

import app.lib.db.store as store_module
from app.lib.tool.base import Param, Tool, ToolResult
from app.tools.task_tool import task_tool
from tests.app.lib.tool.conftest import run_text, wait_subtask_id, wait_terminal


class _TaskStore(Protocol):
    """The task record store the tool_infra fixture gives to a test."""


def _write_run_file(tool_infra: _TaskStore, tid: str, filename: str, content: str) -> None:
    run_path = os.path.join(tool_infra.data_dir, f"runs/{tid}")
    os.makedirs(run_path, exist_ok=True)
    with open(os.path.join(run_path, filename), "w") as fh:
        fh.write(content)


async def _seed_record(
    store: _TaskStore,
    tid: str,
    tool_name: str = "t",
    params: dict | None = None,
    status: str | None = None,
    error: str | None = None,
    parent_task_id: str | None = None,
    started_at: str | None = None,
) -> None:
    """Add a task and its tool row, and finish the task when a status is given."""
    now = started_at or datetime.now(timezone.utc).isoformat()
    await store.insert_running(tid, parent_task_id, now, f"runs/{tid}")
    await store_module.tool_store.insert(tid, tool_name, None, params or {})
    if status is not None:
        await store.finish(tid, status, ended_at=now, error=error)


class _SlowDoneTool(Tool):
    name: ClassVar[str] = "slow"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        await asyncio.sleep(0.2)
        return ToolResult(text="done")


class _SlowCancelTool(Tool):
    name: ClassVar[str] = "slow2"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        await asyncio.sleep(5)
        return ToolResult(text="never")


class _TimeoutFailTool(Tool):
    name: ClassVar[str] = "timeoutfail"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        await asyncio.sleep(0.2)
        raise TimeoutError("op timed out")


class _VerySlowTool(Tool):
    name: ClassVar[str] = "veryslow"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        await asyncio.sleep(30)
        return ToolResult(text="never")


class _FastWaitTaskTool(type(task_tool)):
    """The task tool that accepts a one second wait timeout."""

    params: ClassVar[tuple[Param, ...]] = tuple(
        replace(p, min=1.0) if p.name == "wait_timeout" else p for p in task_tool.params
    )


class _GrandWorkerTool(Tool):
    name: ClassVar[str] = "grandworker"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        await asyncio.sleep(5)
        return ToolResult(text="grand-done")


class _WorkerTool(Tool):
    name: ClassVar[str] = "worker"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        await _GrandWorkerTool().run({})
        return ToolResult(text="worker-done")


class _OrchestratorTool(Tool):
    name: ClassVar[str] = "orchestrator"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        await _WorkerTool().run({})
        await asyncio.sleep(10)
        return ToolResult(text="orch-done")


class TestTaskTool:
    async def test_status_report(self, tool_infra: _TaskStore) -> None:
        tid = str(uuid.uuid4())
        await _seed_record(
            tool_infra, tid, tool_name="web_fetch", params={"url": "u"}, status="completed"
        )
        out = await run_text(task_tool, action="status", task_id=tid)
        assert f"task_id: {tid}" in out
        assert "tool: web_fetch" in out
        assert "status: completed" in out
        assert "error:" not in out

    async def test_unknown_id(self) -> None:
        out = await run_text(task_tool, action="status", task_id="does-not-exist")
        assert out == "No task found with id 'does-not-exist'."

    async def test_result_with_offset_and_max_char(self, tool_infra: _TaskStore) -> None:
        tid = str(uuid.uuid4())
        await _seed_record(tool_infra, tid, status="completed")
        content = "abcdefghij" * 100
        _write_run_file(tool_infra, tid, "result", content)
        out = await run_text(task_tool, action="result", task_id=tid, offset=4, max_char=500)
        assert out == content[4:504]

    async def test_sub_min_max_char_fails_validation(self, tool_infra: _TaskStore) -> None:
        tid = str(uuid.uuid4())
        await _seed_record(tool_infra, tid, status="completed")
        _write_run_file(tool_infra, tid, "result", "a" * 600)
        out = await run_text(task_tool, action="result", task_id=tid, max_char=6)
        assert out.startswith("Error: max_char")

    async def test_result_running(self, tool_infra: _TaskStore) -> None:
        tid = str(uuid.uuid4())
        await _seed_record(tool_infra, tid)
        out = await run_text(task_tool, action="result", task_id=tid)
        assert out == f"Task {tid} is still running."

    async def test_result_failed_returns_error(self, tool_infra: _TaskStore) -> None:
        tid = str(uuid.uuid4())
        await _seed_record(tool_infra, tid, status="failed", error="boom")
        out = await run_text(task_tool, action="result", task_id=tid)
        assert out == "boom"

    async def test_result_wait_until_complete(self) -> None:
        task = await _SlowDoneTool().run({})
        tid = str(task.id)
        out = await run_text(task_tool, action="result", task_id=tid, wait=True)
        assert out == "done"

    async def test_result_wait_timeout_run_failed_with_timeout_error(self) -> None:
        task = await _TimeoutFailTool().run({})
        tid = str(task.id)
        out = await run_text(task_tool, action="result", task_id=tid, wait=True, wait_timeout=10)
        assert out == "op timed out"

    async def test_result_wait_timeout_boundary_accepted(self) -> None:
        task = await _SlowDoneTool().run({})
        tid = str(task.id)
        out = await run_text(task_tool, action="result", task_id=tid, wait=True, wait_timeout=10)
        assert out == "done"

    @pytest.mark.parametrize("bad", [9.99, 5, 0, -1, True])
    async def test_result_wait_timeout_rejects_invalid(
        self, tool_infra: _TaskStore, bad: int | float | bool
    ) -> None:
        tid = str(uuid.uuid4())
        await _seed_record(tool_infra, tid)
        out = await run_text(task_tool, action="result", task_id=tid, wait=True, wait_timeout=bad)
        assert out == "Error: wait_timeout: Input should be greater than or equal to 10."

    async def test_result_wait_timeout_coerced_string_at_min_is_accepted(
        self, tool_infra: _TaskStore
    ) -> None:
        tid = str(uuid.uuid4())
        await _seed_record(tool_infra, tid)
        out = await run_text(task_tool, action="result", task_id=tid, wait=True, wait_timeout="10")
        assert out == f"Task {tid} is still running."

    async def test_result_wait_timeout_below_min_rejected_when_not_waiting(
        self, tool_infra: _TaskStore
    ) -> None:
        tid = str(uuid.uuid4())
        await _seed_record(tool_infra, tid, status="completed")
        _write_run_file(tool_infra, tid, "result", "ok")
        out = await run_text(task_tool, action="result", task_id=tid, wait_timeout=5)
        assert out == "Error: wait_timeout: Input should be greater than or equal to 10."

    @pytest.mark.timeout(30)
    async def test_result_wait_timeout_reports_still_running(
        self, tool_infra: _TaskStore
    ) -> None:
        wait_timeout_param = next(p for p in _FastWaitTaskTool.params if p.name == "wait_timeout")
        assert wait_timeout_param.min == 1.0
        task = await _VerySlowTool().run({})
        tid = str(task.id)
        started = time.monotonic()
        out = await run_text(_FastWaitTaskTool(), action="result", task_id=tid,
                             wait=True, wait_timeout=1)
        prefix = f"Task {tid} is still running ("
        record = await tool_infra.get(tid)
        assert out.startswith(prefix)
        elapsed_text = out[len(prefix) :].split("s since")[0]
        assert float(elapsed_text) >= 1
        assert f"s since {record.started_at}); repoll for the result." in out
        assert time.monotonic() - started >= 1
        out = await run_text(task_tool, action="cancel", task_id=tid)
        assert out == f"Task {tid} is now cancelled."

    async def test_logs_tail(self, tool_infra: _TaskStore) -> None:
        tid = str(uuid.uuid4())
        await _seed_record(tool_infra, tid, status="completed")
        _write_run_file(tool_infra, tid, "log", "line1\nline2\nline3\n")
        out = await run_text(task_tool, action="logs", task_id=tid, tail=2)
        assert out == "line2\nline3"

    async def test_logs_window(self, tool_infra: _TaskStore) -> None:
        tid = str(uuid.uuid4())
        await _seed_record(tool_infra, tid)
        content = "abcdefghij" * 100
        _write_run_file(tool_infra, tid, "log", content)
        out = await run_text(task_tool, action="logs", task_id=tid, offset=6, max_char=500)
        assert out == content[6:506]

    async def test_cancel_live_task(self) -> None:
        task = await _SlowCancelTool().run({})
        tid = str(task.id)
        out = await run_text(task_tool, action="cancel", task_id=tid)
        assert out == f"Task {tid} is now cancelled."

    async def test_cancel_terminal_is_noop(self, tool_infra: _TaskStore) -> None:
        tid = str(uuid.uuid4())
        await _seed_record(tool_infra, tid, status="completed")
        out = await run_text(task_tool, action="cancel", task_id=tid)
        assert out == f"Task {tid} is already completed."

    async def test_status_lists_subtask_ids_in_spawn_order(self, tool_infra: _TaskStore) -> None:
        early = str(uuid.uuid4())
        late = str(uuid.uuid4())
        parent = str(uuid.uuid4())
        await _seed_record(tool_infra, early, status="completed",
                           parent_task_id=parent, started_at="2026-01-01T00:00:00+00:00")
        await _seed_record(tool_infra, late, status="completed",
                           parent_task_id=parent, started_at="2026-01-02T00:00:00+00:00")
        await _seed_record(tool_infra, parent, status="completed",
                           started_at="2025-12-31T00:00:00+00:00")
        out = await run_text(task_tool, action="status", task_id=parent)
        assert f"subtasks: {early}, {late}" in out

    async def test_status_subtasks_dash_when_none(self, tool_infra: _TaskStore) -> None:
        tid = str(uuid.uuid4())
        await _seed_record(tool_infra, tid, status="completed")
        out = await run_text(task_tool, action="status", task_id=tid)
        assert "subtasks: -" in out

    async def test_cancel_cascades_through_terminal_intermediate(self, tool_infra: _TaskStore) -> None:
        task = await _OrchestratorTool().run({})
        parent_id = str(task.id)
        await wait_terminal(tool_infra, await wait_subtask_id(tool_infra, parent_id))
        out = await run_text(task_tool, action="cancel", task_id=parent_id)
        assert out == f"Task {parent_id} is now cancelled."
        children = await tool_infra.list_subtasks(parent_id)
        assert len(children) == 1
        child = children[0]
        assert (await tool_infra.get(child.task_id)).status == "completed"
        grandchildren = await tool_infra.list_subtasks(child.task_id)
        assert len(grandchildren) == 1
        grand = await tool_infra.get(grandchildren[0].task_id)
        assert grand.status == "cancelled"
