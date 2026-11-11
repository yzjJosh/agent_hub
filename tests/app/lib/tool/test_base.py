import asyncio
import errno
import json
import os
import shutil
import time
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import ClassVar, Protocol, override

import pytest

import app.lib.db.store as store_module
import app.lib.task.logcapture as logcapture
from app.lib.db.store import TaskRecord
from app.lib.render.detect import TEXT_FAMILY_KINDS
from app.lib.shared import activity
from app.lib.task import task_manager
from app.lib.task.task_manager import AsyncTask, TaskStatus
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult
from app.tools.task_tool import task_tool
from tests.app.lib.tool.conftest import assert_rejected_task, run_text, wait_terminal

SLOW = 0.2
TINY_SLEEP = 0.01
SETTLE = 0.05
LONG_LEN = 10_000


class _TaskStore(Protocol):
    """The task record store the fixture gives to a test."""

# The exact note text a cached tool adds to its description.
_EXPECTED_CACHE_NOTE_24H = "Duplicate calls within 24 hours reuse the stored result silently."


class _EchoTool(Tool):
    name: ClassVar[str] = "echo"
    params: ClassVar[tuple[Param, ...]] = (
        Param("message", str, "pong", description="The message."),
        Param("sleep", float, TINY_SLEEP, description="Seconds to sleep."),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        print("[echo] running")
        await asyncio.sleep(params.get("sleep", TINY_SLEEP))
        return ToolResult(text=params.get("message", "pong"))


class _AsyncEchoTool(Tool):
    name: ClassVar[str] = "async_echo"
    is_slow: ClassVar[bool] = True
    params: ClassVar[tuple[Param, ...]] = (
        Param("message", str, "pong", description="The message."),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        await asyncio.sleep(SLOW)
        return ToolResult(text=params.get("message", "pong"))


class _AsyncPrintTool(Tool):
    name: ClassVar[str] = "async_print"
    is_slow: ClassVar[bool] = True
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        print("[async_print] running")
        await asyncio.sleep(SLOW)
        return ToolResult(text="pong")


class _CachedTool(Tool):
    name: ClassVar[str] = "cached"
    staleness: ClassVar[timedelta] = timedelta(hours=1)
    params: ClassVar[tuple[Param, ...]] = (
        Param("query", str, description="The query."),
    )

    def __init__(self) -> None:
        self.calls = 0

    @override
    async def call(self, params: dict) -> ToolResult:
        self.calls += 1
        await asyncio.sleep(TINY_SLEEP)
        return ToolResult(text=f"value-{params['query']}")

    @override
    async def cache_key(self, params: dict) -> str:
        return f"k:{params['query']}"


class _ResourceInputTool(_CachedTool):
    params: ClassVar[tuple[Param, ...]] = (
        Param("query", str, kind="resource"),
        Param("files", list[str] | None, None, kind="resource_list"),
    )


async def test_resource_interest_is_recorded_on_cache_hits(
    tool_infra: _TaskStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit resource inputs count even when their tool result is cached."""
    monkeypatch.setenv("RESOURCES_DIR", str(tmp_path))
    path = tmp_path / "input.txt"
    path.write_text("source")
    other = tmp_path / "other.txt"
    other.write_text("other")
    tool = _ResourceInputTool()
    first = await tool.run({"query": "input.txt", "files": ["other.txt"]})
    assert await tool.result_text(first) == "value-input.txt"
    await wait_terminal(tool_infra, str(first.id))
    os.removexattr(path, "user.agent.last_access")
    os.removexattr(other, "user.agent.last_access")
    assert await run_text(tool, query="input.txt", files=["other.txt"]) == "value-input.txt"
    assert tool.calls == 1
    assert activity.last_used_sync(str(path), path.stat()) > time.time() - 10
    assert activity.last_used_sync(str(other), other.stat()) > time.time() - 10


@pytest.mark.usefixtures("tool_infra")
async def test_unsaved_resource_interest_fails_the_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A resource input whose interest cannot be saved fails the call without running it."""
    monkeypatch.setenv("RESOURCES_DIR", str(tmp_path))
    (tmp_path / "input.txt").write_text("source")

    def _refuse(_path: str, _name: str, _value: bytes) -> None:
        raise OSError(errno.ENOTSUP, "no attributes")

    monkeypatch.setattr(os, "setxattr", _refuse)
    tool = _ResourceInputTool()
    task = await tool.run({"query": "input.txt"})
    await assert_rejected_task(task, f"Tool 'cached' failed: [Errno {errno.ENOTSUP}] no attributes")
    assert tool.calls == 0


class _FailingTool(Tool):
    name: ClassVar[str] = "failing"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        raise RuntimeError("boom")


class _LongTool(Tool):
    name: ClassVar[str] = "long"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="x" * LONG_LEN)


class _LongCachedTool(Tool):
    name: ClassVar[str] = "long_cached"
    staleness: ClassVar[timedelta] = timedelta(hours=1)
    params: ClassVar[tuple[Param, ...]] = ()

    def __init__(self) -> None:
        self.calls = 0

    @override
    async def call(self, _params: dict) -> ToolResult:
        self.calls += 1
        return ToolResult(text="x" * LONG_LEN)

    @override
    async def cache_key(self, _params: dict) -> str:
        return "k"


class _CachedFailingTool(Tool):
    name: ClassVar[str] = "cf"
    staleness: ClassVar[timedelta] = timedelta(hours=1)
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def cache_key(self, _params: dict) -> str:
        return "k"

    @override
    async def call(self, _params: dict) -> ToolResult:
        raise RuntimeError("boom")


class _ChildTool(Tool):
    name: ClassVar[str] = "child"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        print("[child] running")
        return ToolResult(text="child-result")


class _ParentTool(Tool):
    name: ClassVar[str] = "parent"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        print("[parent] running")
        return ToolResult(text=f"parent:{await run_text(_ChildTool())}")


class _GrandTool(Tool):
    name: ClassVar[str] = "grand"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="grand-result")


class _ChildWithGrandTool(Tool):
    name: ClassVar[str] = "child2"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text=f"child2:{await run_text(_GrandTool())}")


class _Parent2Tool(Tool):
    name: ClassVar[str] = "parent2"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text=f"parent2:{await run_text(_ChildWithGrandTool())}")


class _ParentWithCachedChildTool(Tool):
    name: ClassVar[str] = "parentcached"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        child = _CachedTool()
        first = await run_text(child, query="q")
        second = await run_text(child, query="q")
        return ToolResult(text=f"{first}|{second}")


async def _detached_slow_child() -> None:
    """A tracked child that would run a long time."""
    await asyncio.sleep(10)


class _DetachedParentTool(Tool):
    name: ClassVar[str] = "detached"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        await task_manager.create_task(_detached_slow_child)
        await asyncio.sleep(10)
        return ToolResult(text="detached")


def test_result_render_type_defaults_to_none() -> None:
    assert _ChildTool.result_render_type is None


class TestRunBlocking:
    async def test_returns_result(self) -> None:
        assert await run_text(_EchoTool(), message="hi") == "hi"

    async def test_record_completed_with_raw_result(self, tool_infra: _TaskStore) -> None:
        out = await run_text(_EchoTool(), message="persist me")
        assert out == "persist me"
        rows = await tool_infra.list()
        assert rows and rows[0].status == "completed"
        raw = open(os.path.join(tool_infra.data_dir, rows[0].run_dir, "result")).read()
        assert raw == "persist me"

    async def test_failure_returns_error_string(self, tool_infra: _TaskStore) -> None:
        out = await run_text(_FailingTool())
        assert out == "Tool 'failing' failed: boom"
        rows = await tool_infra.list()
        assert rows[0].status == "failed"
        assert rows[0].error == "boom"

    async def test_failed_run_not_cacheable(self) -> None:
        t = _CachedFailingTool()
        assert (await run_text(t)).startswith("Tool 'cf' failed:")
        assert (await run_text(t)).startswith("Tool 'cf' failed:")  # second call re-ran, still fails

    async def test_long_result_comes_back_raw_with_no_note(self, tool_infra: _TaskStore) -> None:
        """run hands back the full result and stores it raw."""
        out = await run_text(_LongTool())
        assert out == "x" * LONG_LEN
        assert "[Truncated" not in out
        rows = await tool_infra.list()
        raw = open(os.path.join(tool_infra.data_dir, rows[0].run_dir, "result")).read()
        assert raw == "x" * LONG_LEN  # stored result stays raw

    async def test_cancelled_run_converges_its_tree(
        self, tool_infra: _TaskStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancel landing again while the handler walks the tree still lets every subtask converge."""
        original_finish = store_module.task_store.finish

        async def slow_finish(
            task_id: str, status: str, *, error: str | None = None, ended_at: str | None = None
        ) -> None:
            await asyncio.sleep(0.3)
            await original_finish(task_id, status, error=error, ended_at=ended_at)

        monkeypatch.setattr(store_module.task_store, "finish", slow_finish)
        await _DetachedParentTool().run({})
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        parent_task_id: str | None = None
        while True:
            rows = await tool_infra.list()
            if parent_task_id is None:
                tops = [r for r in rows if r.parent_task_id is None]
                if tops:
                    parent_task_id = tops[0].task_id
            if parent_task_id is not None and await store_module.task_store.list_subtasks(parent_task_id):
                break
            if loop.time() > deadline:
                raise AssertionError("the parent run or its child row never appeared")
            await asyncio.sleep(0.01)
        await task_manager.cancel(str(parent_task_id))
        await task_manager.cancel(str(parent_task_id))
        deadline = loop.time() + 5
        while True:
            rows = await tool_infra.list()
            parent_row = next(r for r in rows if r.task_id == parent_task_id)
            child_rows = [r for r in rows if r.parent_task_id == parent_task_id]
            if parent_row.status == "cancelled" and child_rows and all(r.status == "cancelled" for r in child_rows):
                break
            if loop.time() > deadline:
                raise AssertionError(
                    f"the tree did not converge: parent={parent_row.status}, children={[r.status for r in child_rows]}"
                )
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)


class TestRunAsync:
    """A slow tool's run hands back its handle while the run is still going."""

    async def test_handle_and_later_result(self, tool_infra: _TaskStore) -> None:
        assert _AsyncEchoTool.is_slow is True
        task = await _AsyncEchoTool().run({"message": "later"})
        assert isinstance(task, AsyncTask)
        rows = await tool_infra.list()
        assert rows[0].task_id == str(task.id)
        assert rows[0].status == "running"
        assert (await task.result()).text == "later"
        rows = await tool_infra.list()
        assert rows[0].status == "completed"

    async def test_live_task_resolvable_by_id(self) -> None:
        task = await _AsyncEchoTool().run({})
        live = await task_manager.get(task.id)
        assert (await live.result()).text == "pong"

    async def test_cancel_marks_cancelled(self, tool_infra: _TaskStore) -> None:
        task = await _AsyncEchoTool().run({})
        await task.cancel()
        await asyncio.sleep(SLOW * 3)
        rows = await tool_infra.list()
        assert rows[0].status == "cancelled"


class TestResultText:
    """The base read API maps a run's outcome to the text a client shows."""

    async def test_result_body_verbatim(self) -> None:
        task = task_manager.create_completed_task(ToolResult(text="hello"))
        assert await _EchoTool().result_text(task) == "hello"

    async def test_message_error_becomes_its_own_text(self) -> None:
        task = task_manager.create_completed_task(error=ToolMessageError("Error: bad input."))
        assert await _EchoTool().result_text(task) == "Error: bad input."

    async def test_other_error_becomes_the_failure_message(self) -> None:
        task = task_manager.create_completed_task(error=RuntimeError("boom"))
        assert await _EchoTool().result_text(task) == "Tool 'echo' failed: boom"

    async def test_finished_async_run_returns_its_body(self) -> None:
        async def _runner() -> ToolResult:
            await asyncio.sleep(SLOW)
            return ToolResult(text="late")
        task = await task_manager.create_task(_runner)
        assert await _EchoTool().result_text(task) == "late"

    async def test_failed_async_run_becomes_the_failure_message(self) -> None:
        async def _failing() -> ToolResult:
            raise RuntimeError("boom")
        task = await task_manager.create_task(_failing)
        assert await _EchoTool().result_text(task) == "Tool 'echo' failed: boom"

    async def test_cancelled_run_names_its_task(self) -> None:
        async def _slow() -> ToolResult:
            await asyncio.sleep(5)
            return ToolResult(text="late")
        task = await task_manager.create_task(_slow)
        await task.cancel()
        assert await _EchoTool().result_text(task) == f"Task {task.id} was cancelled."

    async def test_cancelled_caller_does_not_stop_the_run(self) -> None:
        """A cancelled waiter no longer cancels the run. The run keeps going."""
        async def _slow() -> ToolResult:
            await asyncio.sleep(SLOW)
            return ToolResult(text="late")
        task = await task_manager.create_task(_slow)
        waiter = asyncio.create_task(_EchoTool().result_text(task))
        await task.wait_started()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        result = await task.result()
        assert task.status is TaskStatus.COMPLETED
        assert result.text == "late"


class TestCache:
    async def test_second_call_hits_cache(self, tool_infra: _TaskStore) -> None:
        t = _CachedTool()
        first = await run_text(t, query="q")
        second = await run_text(t, query="q")
        assert first == second == "value-q"
        assert t.calls == 1
        assert len(await tool_infra.list()) == 1  # no second record

    async def test_cache_hit_returns_the_raw_stored_result(self, tool_infra: _TaskStore) -> None:
        """A reused long result comes back raw, with no note, and does not run again."""
        t = _LongCachedTool()
        first = await run_text(t)
        second = await run_text(t)
        assert first == second == "x" * LONG_LEN
        assert "[Truncated" not in second
        rows = await tool_infra.list()
        assert len(rows) == 1
        assert t.calls == 1

    async def test_different_key_misses(self) -> None:
        t = _CachedTool()
        await t.run({"query": "a"})
        await t.run({"query": "b"})
        assert t.calls == 2

    async def test_swept_result_file_degrades_to_miss(self, tool_infra: _TaskStore) -> None:
        t = _CachedTool()
        await t.run({"query": "q"})
        rows = await tool_infra.list()
        shutil.rmtree(os.path.join(tool_infra.data_dir, rows[0].run_dir))
        out = await run_text(t, query="q")
        assert out == "value-q"
        assert t.calls == 2


class TestLogCapture:
    """The run log holds the tool's prints once the run's result is ready."""

    async def test_prints_landed_in_run_log(self, tool_infra: _TaskStore) -> None:
        logcapture.install_log_tee()
        try:
            await run_text(_EchoTool())
        finally:
            logcapture.restore_log_tee()
        rows = await tool_infra.list()
        log = open(os.path.join(tool_infra.data_dir, rows[0].run_dir, "log")).read()
        assert "[echo] running" in log

    async def test_prints_landed_in_run_log_for_a_long_run(self, tool_infra: _TaskStore) -> None:
        assert _AsyncPrintTool.is_slow is True
        logcapture.install_log_tee()
        try:
            task = await _AsyncPrintTool().run({})
            await task.result()
        finally:
            logcapture.restore_log_tee()
        rows = await tool_infra.list()
        assert rows[0].status == "completed"
        log = open(os.path.join(tool_infra.data_dir, rows[0].run_dir, "log")).read()
        assert "[async_print] running" in log


async def _rows_by_name(tool_infra: _TaskStore) -> dict[str, TaskRecord]:
    rows = await tool_infra.list()
    return {
        (await store_module.tool_store.get(r.task_id)).tool_name: r for r in rows
    }


class TestNesting:
    async def test_child_row_points_at_parent(self, tool_infra: _TaskStore) -> None:
        out = await run_text(_ParentTool())
        assert out == "parent:child-result"
        rows = await _rows_by_name(tool_infra)
        assert rows["child"].parent_task_id == rows["parent"].task_id
        assert rows["parent"].parent_task_id is None

    async def test_grandchild_points_at_child(self, tool_infra: _TaskStore) -> None:
        out = await run_text(_Parent2Tool())
        assert out == "parent2:child2:grand-result"
        rows = await _rows_by_name(tool_infra)
        assert rows["child2"].parent_task_id == rows["parent2"].task_id
        assert rows["grand"].parent_task_id == rows["child2"].task_id

    async def test_run_returns_only_after_row_exists(self, tool_infra: _TaskStore) -> None:
        task = await _AsyncEchoTool().run({})
        row = await tool_infra.get(str(task.id))
        assert row is not None and row.status == "running"

    async def test_tool_row_written_by_run(self, tool_infra: _TaskStore) -> None:
        await _EchoTool().run({"message": "x"})
        rows = await tool_infra.list()
        tool = await store_module.tool_store.get(rows[0].task_id)
        assert tool is not None and tool.tool_name == "echo"
        assert tool.params == {"message": "x", "sleep": TINY_SLEEP}

    async def test_nested_cache_hit_creates_no_row(self, tool_infra: _TaskStore) -> None:
        out = await run_text(_ParentWithCachedChildTool())
        assert out == "value-q|value-q"
        rows = await _rows_by_name(tool_infra)
        assert set(rows) == {"parentcached", "cached"}

    async def test_child_log_kept_out_of_parent_log(self, tool_infra: _TaskStore) -> None:
        logcapture.install_log_tee()
        try:
            await run_text(_ParentTool())
        finally:
            logcapture.restore_log_tee()
        rows = await _rows_by_name(tool_infra)
        parent_log = open(os.path.join(tool_infra.data_dir, rows["parent"].run_dir, "log")).read()
        child_log = open(os.path.join(tool_infra.data_dir, rows["child"].run_dir, "log")).read()
        assert "[parent] running" in parent_log and "[child] running" not in parent_log
        assert "[child] running" in child_log and "[parent] running" not in child_log


class _DescProbe(Tool):
    name: ClassVar[str] = "desc_probe"
    tracked: ClassVar[bool] = False
    params: ClassVar[tuple[Param, ...]] = ()
    description: ClassVar[str] = "A base description."

    @override
    async def call(self, _params: dict) -> str:
        return "ok"


class _DescProbeSlow(_DescProbe):
    name: ClassVar[str] = "desc_probe_slow"
    is_slow: ClassVar[bool] = True


class _DescProbeCached(_DescProbe):
    name: ClassVar[str] = "desc_probe_cached"
    staleness: ClassVar[timedelta] = timedelta(hours=24)


def _cached_probe(staleness: timedelta) -> Tool:
    """Make a probe tool with the given reuse window."""
    return type("cached_probe", (_DescProbe,), {"staleness": staleness})()


class TestFormatStaleness:
    """Tests for how the cache note spells out the reuse window."""

    @pytest.mark.parametrize(
        ("staleness", "rendered"),
        [
            (timedelta(hours=24), "24 hours"),
            (timedelta(days=1), "24 hours"),
            (timedelta(hours=36), "36 hours"),
            (timedelta(hours=72), "3 days"),
            (timedelta(days=2), "2 days"),
            (timedelta(hours=1), "1 hour"),
            (timedelta(minutes=90), "90 minutes"),
            (timedelta(minutes=1), "1 minute"),
            (timedelta(seconds=45), "45 seconds"),
            (timedelta(seconds=1), "1 second"),
        ],
    )
    def test_outcome_table(self, staleness: timedelta, rendered: str) -> None:
        expected_note = f"Duplicate calls within {rendered} reuse the stored result silently."
        assert expected_note in _cached_probe(staleness).dynamic_description


class TestDynamicDescription:
    """Tests for the description a tool builds from its own settings."""

    def test_blocking_uncached_is_base_verbatim(self) -> None:
        assert _DescProbe().dynamic_description == "A base description."

    def test_slow_flag_adds_no_note(self) -> None:
        assert _DescProbeSlow.is_slow is True
        assert _DescProbeSlow().dynamic_description == "A base description."

    def test_cached_gains_cache_note(self) -> None:
        assert _DescProbeCached().dynamic_description == "A base description. " + _EXPECTED_CACHE_NOTE_24H

    def test_flipping_slow_adds_nothing_without_description_edit(self) -> None:
        base = _DescProbe().dynamic_description
        assert _DescProbeSlow().dynamic_description == base

    def test_cache_note_names_the_stored_result(self) -> None:
        assert "stored result" in _DescProbeCached().dynamic_description

    @pytest.mark.parametrize(
        "word",
        ["Identical", "query", "count", "raw_notes", "location", "prompt",
         "resource_path", "reference_images"],
    )
    def test_cache_note_is_generic_and_names_no_key_inputs(self, word: str) -> None:
        assert word not in _DescProbeCached().dynamic_description


_BANNER = b"\x89PNG\r\n\x1a\nBINBYTES"   # keep the constant name usage below


class _FileTool(Tool):
    name: ClassVar[str] = "file"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        # A share file the tool just wrote; the manifest records its size/mtime.
        return ToolResult(text="Binary file with size 13 bytes.", files=("downloads/img.png",))


class _NulTool(Tool):
    name: ClassVar[str] = "nul"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="bad\x00body")


class _StrTool(Tool):
    name: ClassVar[str] = "str"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        return "still a plain string"   # type: ignore[return-value]  violates the new contract


class TestBinaryResult:
    async def test_text_return_no_sidecars(self, tool_infra: _TaskStore) -> None:
        await run_text(_EchoTool(), message="plain")
        row = (await tool_infra.list())[0]
        run_dir = os.path.join(tool_infra.data_dir, row.run_dir)
        assert open(os.path.join(run_dir, "result")).read() == "plain"
        assert not os.path.exists(os.path.join(run_dir, "result.text"))
        assert not os.path.exists(os.path.join(run_dir, "files"))

    async def test_files_write_json_manifest(
        self, tool_infra: _TaskStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The share file must exist so the manifest records its size/mtime.
        share = tmp_path / "share"
        (share / "downloads").mkdir(parents=True)
        (share / "downloads" / "img.png").write_bytes(_BANNER)
        monkeypatch.setenv("RESOURCES_DIR", str(share))
        out = await run_text(_FileTool())
        assert out == "Binary file with size 13 bytes."
        row = (await tool_infra.list())[0]
        run_dir = os.path.join(tool_infra.data_dir, row.run_dir)
        assert open(os.path.join(run_dir, "result")).read() == "Binary file with size 13 bytes."
        assert not os.path.exists(os.path.join(run_dir, "result.text"))
        line = json.loads(open(os.path.join(run_dir, "files")).read().strip())
        assert line["path"] == "downloads/img.png"
        assert line["size"] == len(_BANNER) and isinstance(line["mtime"], int)

    async def test_task_result_reads_body_text(
        self, tool_infra: _TaskStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        share = tmp_path / "share2"
        (share / "downloads").mkdir(parents=True)
        (share / "downloads" / "img.png").write_bytes(_BANNER)
        monkeypatch.setenv("RESOURCES_DIR", str(share))
        await run_text(_FileTool())
        row = (await tool_infra.list())[0]
        text = await run_text(task_tool, action="result", task_id=row.task_id)
        assert text == "Binary file with size 13 bytes."
        assert b"\x89" not in text.encode()

    async def test_non_toolresult_fails_run(self, tool_infra: _TaskStore) -> None:
        out = await run_text(_StrTool())
        assert out.startswith("Tool 'str' failed:")
        row = (await tool_infra.list())[0]
        run_dir = os.path.join(tool_infra.data_dir, row.run_dir)
        assert not os.path.exists(os.path.join(run_dir, "result"))   # nothing written

    async def test_nul_text_fails_run(self, tool_infra: _TaskStore) -> None:
        out = await run_text(_NulTool())
        assert out.startswith("Tool 'nul' failed:")
        row = (await tool_infra.list())[0]
        run_dir = os.path.join(tool_infra.data_dir, row.run_dir)
        assert not os.path.exists(os.path.join(run_dir, "result"))


class TestRunPersistenceContract:
    """A tool run leaves runs/<task_id>/log and runs/<task_id>/result, byte for byte."""

    async def test_run_folder_paths_and_content(self, tool_infra: _TaskStore) -> None:
        logcapture.install_log_tee()
        try:
            out = await run_text(_EchoTool(), message="persist me")
        finally:
            logcapture.restore_log_tee()
        assert out == "persist me"
        row = (await tool_infra.list())[0]
        assert row.run_dir == f"runs/{row.task_id}"
        run_path = os.path.join(tool_infra.data_dir, row.run_dir)
        assert os.path.isdir(run_path)
        assert open(os.path.join(run_path, "result")).read() == "persist me"
        assert "[echo] running" in open(os.path.join(run_path, "log")).read()
        assert not os.path.exists(os.path.join(run_path, "result.text"))
        assert not os.path.exists(os.path.join(run_path, "files"))


class _ValidateProbe(Tool):
    name: ClassVar[str] = "validate_probe"
    description: ClassVar[str] = "A validation probe."
    params: ClassVar[tuple[Param, ...]] = (
        Param("action", str, enum=("status", "result", "logs", "cancel"), description="Which action."),
        Param("task_id", str, description="The run id."),
        Param("limit", int, description="Max results."),
        Param("query", str, "", description="The query."),
        Param("count", int, 5, max=100, description="How many."),
        Param("score", float, 0.0, description="The score."),
        Param("wait_timeout", float | None, None, min=10.0, description="The wait cap."),
        Param("files", list[str] | None, None, description="The files."),
        Param("freshness", str | None, None, enum=("d", "w", "m", "y"), description="The time filter."),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text=str(params))


_PROBE_OK: dict = {"action": "status", "task_id": "x", "limit": 1}


class _RawProbe(Tool):
    """A tool that records the raw params its call received."""

    name: ClassVar[str] = "raw_probe"
    params: ClassVar[tuple[Param, ...]] = (
        Param("query", str, description="The query."),
        Param("count", int, 5, max=100, description="How many."),
    )

    def __init__(self) -> None:
        self.seen: dict = {}

    @override
    async def call(self, params: dict) -> ToolResult:
        self.seen = dict(params)
        return ToolResult(text="ok")


class _NamedRunTool(Tool):
    """A tool whose run takes its declared params, like every registered tool."""

    name: ClassVar[str] = "named_run"
    description: ClassVar[str] = "A named run signature."
    params: ClassVar[tuple[Param, ...]] = (
        Param("query", str, description="The query."),
        Param("count", int, 5, max=100, description="How many."),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text=f"{params['query']}:{params['count']}")


class TestValidateParams:
    """A tool checks its params against what it declares and reports only the first problem."""

    def test_accepts_a_conformant_dict(self) -> None:
        assert _ValidateProbe().validate_params(dict(_PROBE_OK)) is None

    def test_accepts_optionals_at_and_from_their_values(self) -> None:
        params = dict(_PROBE_OK, query="q", count=3, score=1.5, wait_timeout=10.0, files=["a"])
        assert _ValidateProbe().validate_params(params) is None

    def test_accepts_lax_str_to_int(self) -> None:
        assert _ValidateProbe().validate_params(dict(_PROBE_OK, limit="5")) is None

    def test_accepts_bool_for_float(self) -> None:
        assert _ValidateProbe().validate_params(dict(_PROBE_OK, score=True)) is None

    def test_accepts_none_only_where_the_type_is_a_none_union(self) -> None:
        assert _ValidateProbe().validate_params(dict(_PROBE_OK, wait_timeout=None, files=None)) is None

    def test_accepts_the_tools_own_none_for_an_enum_param(self) -> None:
        """The tool's own None default passes an enum param's check."""
        assert _ValidateProbe().validate_params(dict(_PROBE_OK, freshness=None)) is None

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            (
                {"action": "status", "limit": 1},
                "Error: task_id: Missing required argument.",
            ),
            (
                dict(_PROBE_OK, limit="abc"),
                "Error: limit: Input should be a valid integer, unable to parse string as an integer.",
            ),
            (
                dict(_PROBE_OK, query=5),
                "Error: query: Input should be a valid string.",
            ),
            (
                dict(_PROBE_OK, task_id=None),
                "Error: task_id: Input should be a valid string.",
            ),
            (
                dict(_PROBE_OK, action="bogus"),
                "Error: action: Input should be 'status', 'result', 'logs' or 'cancel'.",
            ),
            (
                dict(_PROBE_OK, queary="q"),
                "Error: queary: Unexpected keyword argument.",
            ),
            (
                dict(_PROBE_OK, wait_timeout=5),
                "Error: wait_timeout: Input should be greater than or equal to 10.",
            ),
            (
                dict(_PROBE_OK, count=101),
                "Error: count: Input should be less than or equal to 100.",
            ),
            (
                dict(_PROBE_OK, files=["ok", 5]),
                "Error: files.1: Input should be a valid string.",
            ),
        ],
        ids=[
            "missing_required",
            "type_int",
            "type_str",
            "type_none_for_required",
            "enum",
            "undeclared_key",
            "min_bound",
            "max_bound",
            "list_element",
        ],
    )
    def test_reject_shapes(self, params: dict, expected: str) -> None:
        assert _ValidateProbe().validate_params(params) == expected

    def test_coercion_only_value_passes_the_bound_via_rederivation(self) -> None:
        assert _ValidateProbe().validate_params(dict(_PROBE_OK, wait_timeout="10")) is None

    def test_coercion_only_value_fails_the_bound_via_rederivation(self) -> None:
        assert (
            _ValidateProbe().validate_params(dict(_PROBE_OK, wait_timeout="5"))
            == "Error: wait_timeout: Input should be greater than or equal to 10."
        )

    def test_bool_rederives_to_one_and_fails_the_min(self) -> None:
        assert (
            _ValidateProbe().validate_params(dict(_PROBE_OK, wait_timeout=True))
            == "Error: wait_timeout: Input should be greater than or equal to 10."
        )

    def test_decimal_nan_rederives_raise_free_and_fails_the_min(self) -> None:
        assert (
            _ValidateProbe().validate_params(dict(_PROBE_OK, wait_timeout=Decimal("NaN")))
            == "Error: wait_timeout: Input should be greater than or equal to 10."
        )

    def test_validate_does_not_mutate_or_fill_defaults(self) -> None:
        params = dict(_PROBE_OK)
        assert _ValidateProbe().validate_params(params) is None
        assert params == _PROBE_OK
        assert "query" not in params
        assert "count" not in params

    async def test_raw_values_reach_call_uncoerced_with_defaults_completed(self) -> None:
        """The run's stored params keep the raw value uncoerced and fill declared defaults."""
        task = await _ValidateProbe().run(dict(_PROBE_OK, limit="5"))
        await task.result()
        row = await store_module.tool_store.get(str(task.id))
        assert row is not None and row.params["limit"] == "5"  # the raw string, not coerced to 5
        assert row.params["query"] == ""    # the declared default, completed in
        assert row.params["count"] == 5     # the declared default, completed in


class TestValidateNoSideEffects:
    """A bad set of params is reported without making a row, a folder, a cache entry, or an acknowledgement."""

    async def test_run_failure_creates_no_row_or_dir(self, tool_infra: _TaskStore) -> None:
        rows_before = len(await tool_infra.list())
        runs = os.path.join(store_module.tool_store.data_dir, "runs")
        dirs_before = len(os.listdir(runs)) if os.path.isdir(runs) else 0
        out = await run_text(_ValidateProbe(), **dict(_PROBE_OK, limit="abc"))
        assert out == "Error: limit: Input should be a valid integer, unable to parse string as an integer."
        assert len(await tool_infra.list()) == rows_before
        dirs_after = len(os.listdir(runs)) if os.path.isdir(runs) else 0
        assert dirs_after == dirs_before

    async def test_run_failure_touches_no_cache(self) -> None:
        t = _LongCachedTool()
        out = await run_text(t, junk=1)
        assert out == "Error: junk: Unexpected keyword argument."
        assert t.calls == 0
        first = await run_text(t)
        second = await run_text(t)
        assert first == second
        assert t.calls == 1

    async def test_async_invalid_param_produces_no_ack_or_row(self, tool_infra: _TaskStore) -> None:
        out = await run_text(_AsyncEchoTool(), message="later", junk=1)
        assert out == "Error: junk: Unexpected keyword argument."
        assert len(await tool_infra.list()) == 0
        await asyncio.sleep(SETTLE)
        assert len(await tool_infra.list()) == 0


class TestRawDictEntry:
    """run takes a raw params dict, checks it before anything runs, and fills defaults.

    A bad set comes back as a failed immediate handle carrying the tool's own words.
    """

    async def test_good_params_start_the_run(self) -> None:
        task = await _NamedRunTool().run({"query": "q", "count": 3})
        assert isinstance(task, AsyncTask)
        assert (await task.result()).text == "q:3"

    async def test_omitted_default_is_filled(self) -> None:
        task = await _NamedRunTool().run({"query": "q"})
        assert isinstance(task, AsyncTask)
        assert (await task.result()).text == "q:5"

    async def test_present_values_pass_through_raw(self) -> None:
        tool = _RawProbe()
        await tool.run({"query": "q", "count": "3"})
        assert tool.seen == {"query": "q", "count": "3"}

    async def test_omitted_enum_param_defaults(self) -> None:
        task = await _ValidateProbe().run(dict(_PROBE_OK))
        assert isinstance(task, AsyncTask)
        await task.result()
        row = await store_module.tool_store.get(str(task.id))
        assert row is not None and row.params["freshness"] is None

    @pytest.mark.parametrize(
        ("tool", "params", "expected"),
        [
            (_NamedRunTool, {}, "Error: query: Missing required argument."),
            (_NamedRunTool, {"query": "q", "zz": 1}, "Error: zz: Unexpected keyword argument."),
            (_NamedRunTool, {"query": "q", "count": "abc"}, "Error: count: Input should be a valid integer, unable to parse string as an integer."),
            (_NamedRunTool, {"query": "q", "count": 1000}, "Error: count: Input should be less than or equal to 100."),
        ],
        ids=["missing_required", "undeclared_key", "type_int", "max_bound"],
    )
    async def test_bad_params_carry_the_tools_own_words(
        self, tool: type[Tool], params: dict, expected: str, tool_infra: _TaskStore
    ) -> None:
        task = await tool().run(params)
        await assert_rejected_task(task, expected)
        assert len(await tool_infra.list()) == 0


class TestParamWireDescription:
    """The param wire description is built from the param's own data.

    Changing the kind, the default, or the enum changes the wire text with no
    edit to the param's own description.
    """

    def test_required_plain_param_is_base_verbatim(self) -> None:
        p = Param("q", str, description="The query.")
        assert p.wire_description == "The query."

    def test_concrete_default_gains_default_note(self) -> None:
        p = Param("n", int, 10, description="How many.")
        assert p.wire_description == "How many. Optional (default: 10)."

    def test_bool_default_renders_the_plain_value(self) -> None:
        p = Param("flag", bool, True, description="Whether.")
        assert p.wire_description == "Whether. Optional (default: True)."

    def test_none_default_gets_bare_optional_note(self) -> None:
        p = Param("r", str | None, None, description="The repo.")
        assert p.wire_description == "The repo. Optional."

    def test_empty_string_default_gets_bare_optional_note(self) -> None:
        p = Param("prompt", str, "", description="The prompt.")
        assert p.wire_description == "The prompt. Optional."
        assert "Optional (default: )." not in p.wire_description

    @pytest.mark.parametrize("default", [("a", "b"), ["a", "b"], {"a": 1}], ids=["tuple", "list", "dict"])
    def test_non_scalar_default_gets_bare_optional_note(self, default: tuple | list | dict) -> None:
        p = Param("pair", tuple, default, description="The pair.")
        assert p.wire_description == "The pair. Optional."

    def test_set_enum_gains_enum_note(self) -> None:
        p = Param("action", str, enum=("a", "b"), description="Which action.")
        assert p.wire_description == "Which action. Valid values: a, b."

    def test_none_default_enum_still_gains_enum_note(self) -> None:
        p = Param("freshness", str | None, None, enum=("d", "w", "m", "y"),
                  description="The filter.")
        assert p.wire_description == "The filter. Optional. Valid values: d, w, m, y."

    def test_no_enum_gets_no_enum_note(self) -> None:
        p = Param("q", str, description="The query.")
        assert "Valid values" not in p.wire_description

    def test_date_kind_gains_date_note(self) -> None:
        p = Param("date", str, kind="date", description="Travel date.")
        assert p.wire_description == "Travel date. Use the YYYY-MM-DD format."

    def test_url_kind_gains_url_note(self) -> None:
        p = Param("url", str, kind="url", description="The URL.")
        assert p.wire_description == "The URL. Use an http or https URL."

    def test_resource_and_resource_list_notes_are_byte_identical(self) -> None:
        note = (
            "Each path must already exist in the resource. If you created a file, "
            "call the send_or_receive_file tool first for upload instructions, then "
            "use the uploaded file's path, usually 'upload/<filename>'. List available "
            "files with your MCP client's resource listing tool (entries look like "
            "'resources:///upload/video.mp4')."
        )
        p = Param("path", str, kind="resource", description="The path.")
        assert p.wire_description == "The path. " + note
        pl = Param("files", list[str] | None, None, kind="resource_list",
                   description="The files.")
        assert pl.wire_description == "The files. " + note + " Optional."

    def test_resource_note_uses_the_resource_term(self) -> None:
        p = Param("path", str, kind="resource", description="The path.")
        assert "in the resource" in p.wire_description
        assert "share" not in p.wire_description

    def test_fixed_order_purpose_kind_default_enum(self) -> None:
        p = Param("when", str, "2026-01-01", kind="date", enum=("d", "w"),
                  description="The cutoff.")
        expected = (
            "The cutoff. Use the YYYY-MM-DD format. "
            "Optional (default: 2026-01-01). Valid values: d, w."
        )
        assert p.wire_description == expected

    def test_flipping_kind_default_enum_changes_output_without_base_edit(self) -> None:
        base = Param("n", int, description="How many.").wire_description
        assert base == "How many."
        assert "Optional (default:" not in base
        assert "Valid values" not in base
        flipped = Param("n", int, 3, enum=("a", "b"), description="How many.")
        assert flipped.wire_description == "How many. Optional (default: 3). Valid values: a, b."


class _RenderClassTool(Tool):
    name: ClassVar[str] = "render_class"
    result_render_type: ClassVar[str] = "text"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="class-body")


class _RenderBadClassTool(Tool):
    name: ClassVar[str] = "render_bad_class"
    result_render_type: ClassVar[str] = "pdf"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="bad-class-body")


class _RenderRunTool(Tool):
    name: ClassVar[str] = "render_run"
    result_render_type: ClassVar[str] = "text"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="run-body", render_type="json")


class _RenderNoneTool(Tool):
    name: ClassVar[str] = "render_none"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="plain-body")


class _RenderFastTool(Tool):
    name: ClassVar[str] = "render_fast"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="fast-body", render_type="markdown")


class _RenderTokenProbe(Tool):
    name: ClassVar[str] = "render_token_probe"
    params: ClassVar[tuple[Param, ...]] = ()

    def __init__(self, token: str) -> None:
        self._token = token

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="probe-body", render_type=self._token)


class _CachedRenderTool(Tool):
    name: ClassVar[str] = "cached_render"
    staleness: ClassVar[timedelta] = timedelta(hours=1)
    params: ClassVar[tuple[Param, ...]] = (Param("query", str, description="The query."),)

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text=f"v-{params['query']}", render_type="markdown")

    @override
    async def cache_key(self, params: dict) -> str:
        return f"kr:{params['query']}"


class TestRenderTypeDeclaration:
    """Tests for the declared render kind on a run."""

    def test_toolresult_defaults_render_type_none(self) -> None:
        assert ToolResult(text="x").render_type is None

    async def test_run_declaration_wins_over_class(self, tool_infra: _TaskStore) -> None:
        out = await run_text(_RenderRunTool())
        assert out == "run-body"
        view = await store_module.task_view((await tool_infra.list())[0].task_id)
        assert view is not None and view.render_type == "json"

    async def test_class_declaration_persists_when_run_silent(self, tool_infra: _TaskStore) -> None:
        out = await run_text(_RenderClassTool())
        assert out == "class-body"
        view = await store_module.task_view((await tool_infra.list())[0].task_id)
        assert view is not None and view.render_type == "text"

    async def test_out_of_family_class_declaration_persists_nothing(self, tool_infra: _TaskStore) -> None:
        out = await run_text(_RenderBadClassTool())
        assert out == "bad-class-body"
        rows = await tool_infra.list()
        assert rows[0].status == "completed"
        view = await store_module.task_view(rows[0].task_id)
        assert view is not None and view.render_type is None

    async def test_no_declaration_writes_nothing(self, tool_infra: _TaskStore) -> None:
        out = await run_text(_RenderNoneTool())
        assert out == "plain-body"
        view = await store_module.task_view((await tool_infra.list())[0].task_id)
        assert view is not None and view.render_type is None

    async def test_fast_call_keeps_insert_before_update(self, tool_infra: _TaskStore) -> None:
        task = await _RenderFastTool().run({})
        await wait_terminal(tool_infra, str(task.id))
        view = await store_module.task_view(str(task.id))
        assert view is not None and view.render_type == "markdown"

    async def test_cache_hit_reuses_row_and_declaration(self, tool_infra: _TaskStore) -> None:
        tool = _CachedRenderTool()
        first = await tool.run({"query": "q"})
        await wait_terminal(tool_infra, str(first.id))
        second = await tool.run({"query": "q"})
        assert str(second.id) == str(first.id)
        assert len(await tool_infra.list()) == 1
        view = await store_module.task_view(str(first.id))
        assert view is not None and view.render_type == "markdown"

    @pytest.mark.parametrize("token", sorted(k.value for k in TEXT_FAMILY_KINDS))
    async def test_every_text_family_token_is_accepted(self, token: str, tool_infra: _TaskStore) -> None:
        out = await run_text(_RenderTokenProbe(token))
        assert out == "probe-body"
        view = await store_module.task_view((await tool_infra.list())[0].task_id)
        assert view is not None and view.render_type == token

    @pytest.mark.parametrize("token", ["pdf", "image", "unknown", "", "nonsense"])
    async def test_out_of_family_tokens_fail_the_run_and_persist_nothing(
            self, token: str, tool_infra: _TaskStore) -> None:
        out = await run_text(_RenderTokenProbe(token))
        assert out.startswith("Tool 'render_token_probe' failed:")
        assert "result render_type must be one of:" in out
        rows = await tool_infra.list()
        assert rows[0].status == "failed"
        view = await store_module.task_view(rows[0].task_id)
        assert view is not None and view.render_type is None
