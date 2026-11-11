from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

import app.lib.db.store as store_module
from app.lib.task import logcapture
from app.lib.task.task_manager import (
    AsyncTask,
    ImmediateTask,
    Task,
    TaskCancelledError,
    TaskStatus,
    task_manager,
)

_SHORT_WAIT = 0.05
_SETTLE_WAIT = 0.1
_PRE_FAIL_DELAY = 0.01
_PERSIST_SETTLE_WAIT = 0.01
_IN_FLIGHT_WAIT = 0.2
# An upper bound on how long the write lock is held, and on the wait for the child's insert.
_LOCK_HOLD_MAX = 4
_RESULT_WAIT = 5


async def _quick_result() -> int:
    """Callable that completes with 42 after a short sleep."""
    await asyncio.sleep(_SHORT_WAIT)
    return 42


async def _settled_task[T](work: Callable[[], Awaitable[T]]) -> Task[T]:
    """Create a task and wait until its callable is running."""
    manager = task_manager
    task = await manager.create_task(work)
    await asyncio.wait_for(task.wait_started(), _RESULT_WAIT)
    return task


async def _cancel_settled(task: Task) -> None:
    """Cancel a running task and wait for the cancellation to take effect."""
    await task.cancel()


async def _recancel_when_persisting(task: Task, persist_status: str) -> None:
    """Cancel the task while its terminal persist of persist_status is in flight behind the shield."""
    persist_started = asyncio.Event()
    persist_done = asyncio.Event()
    original_finish = store_module.task_store.finish

    async def slow_finish(
        task_id: str, status: str, *, error: str | None = None, ended_at: str | None = None
    ) -> None:
        if status == persist_status:
            persist_started.set()
        await asyncio.sleep(_IN_FLIGHT_WAIT)
        await original_finish(task_id, status, error=error, ended_at=ended_at)
        if status == persist_status:
            persist_done.set()

    store_module.task_store.finish = slow_finish
    try:
        await persist_started.wait()
        await task.cancel()
        await persist_done.wait()
    finally:
        store_module.task_store.finish = original_finish


async def _slow_never() -> str:
    """Callable that would return after a long sleep."""
    await asyncio.sleep(1)
    return "never"


def _raises_after(delay: float, exc: BaseException) -> Callable[[], Awaitable[None]]:
    """Build a callable that sleeps for delay then raises exc."""
    async def work() -> None:
        await asyncio.sleep(delay)
        raise exc

    return work


def _spawns_child(
    box: dict[str, Task],
    work: Callable[[], Awaitable[str]],
    result: str,
) -> Callable[[], Awaitable[str]]:
    """Build a callable that spawns work under its parent, stores it in box, and returns result."""
    async def spawn() -> str:
        box["t"] = await task_manager.create_task(work)
        await box["t"].result()
        return result

    return spawn


class TestCreate:

    async def test_create_returns_task_immediately(self) -> None:
        """A slow task comes back RUNNING right after create, with its id set."""
        async def work() -> str:
            await asyncio.sleep(0.5)
            return "late"

        manager = task_manager
        task = await manager.create_task(work)
        assert isinstance(task, Task)
        assert isinstance(task.id, uuid.UUID)
        assert task.status is TaskStatus.RUNNING
        await task.cancel()


class TestCompleted:

    async def test_completed_result_after_eviction(self) -> None:
        """The handle keeps working with the value after the manager evicted the task."""
        manager = task_manager
        task = await manager.create_task(_quick_result)
        await asyncio.wait_for(task.result(), _RESULT_WAIT)
        assert await manager.get(task.id) is None
        assert task.status is TaskStatus.COMPLETED
        assert await task.result() == 42
        assert await task.result() == 42

    async def test_result_blocks_until_completion(self) -> None:
        """Asking for the result of a running task waits and returns the value."""
        async def work() -> str:
            await asyncio.sleep(0.3)
            return "done"

        manager = task_manager
        task = await manager.create_task(work)
        assert task.status is TaskStatus.RUNNING
        assert await task.result() == "done"
        assert task.status is TaskStatus.COMPLETED

class TestActiveTask:

    def test_active_task_defaults_to_none(self) -> None:
        """Outside any task, the active task is None."""
        assert task_manager.active_task() is None

    async def test_active_task_returns_the_running_task(self) -> None:
        """Inside a task, the active task is that task itself."""
        async def work() -> Task:
            return task_manager.active_task()

        task = await task_manager.create_task(work)
        assert (await task.result()) is task

    async def test_active_task_visible_to_worker_thread(self) -> None:
        """The active task is visible from a worker thread of the task."""
        async def work() -> Task:
            return await asyncio.to_thread(lambda: task_manager.active_task())

        task = await task_manager.create_task(work)
        assert (await task.result()) is task

    async def test_active_task_in_nested_task_is_the_child(self) -> None:
        """Inside a child created by a parent, the active task is the child."""
        child_box: dict[str, Task] = {}
        seen_box: dict[str, Task | None] = {}

        async def child_work() -> None:
            seen_box["v"] = task_manager.active_task()

        async def parent_work() -> None:
            child_box["t"] = await task_manager.create_task(child_work)
            await child_box["t"].result()

        parent = await task_manager.create_task(parent_work)
        await parent.result()
        assert seen_box["v"] is child_box["t"]


class TestGet:

    async def test_get_running_and_none_after_completion(self) -> None:
        """get returns the task while it runs and None once it is done."""
        release = asyncio.Event()

        async def work() -> None:
            await release.wait()

        manager = task_manager
        task = await manager.create_task(work)
        await asyncio.wait_for(task.wait_started(), _RESULT_WAIT)
        assert await manager.get(task.id) is task
        assert await manager.get(uuid.uuid4()) is None
        release.set()
        await asyncio.wait_for(task.result(), _RESULT_WAIT)
        assert await manager.get(task.id) is None


class TestFailed:

    async def test_failed_re_raises_original_exception(self) -> None:
        """A failed task reports FAILED and the result re-raises the very same exception."""
        boom = ValueError("boom")

        manager = task_manager
        task = await manager.create_task(_raises_after(_PRE_FAIL_DELAY, boom))
        with pytest.raises(ValueError, match="boom"):
            await asyncio.wait_for(task.result(), _RESULT_WAIT)
        assert task.status is TaskStatus.FAILED
        with pytest.raises(ValueError, match="boom") as exc_info:
            await task.result()
        assert exc_info.value is boom

    async def test_running_failure_re_raises_original_exception(self) -> None:
        """Asking for the result of a running task that fails re-raises the very same exception."""
        boom = ValueError("boom")

        async def work() -> None:
            await asyncio.sleep(0.2)
            raise boom

        manager = task_manager
        task = await manager.create_task(work)
        assert task.status is TaskStatus.RUNNING
        with pytest.raises(ValueError, match="boom") as exc_info:
            await task.result()
        assert exc_info.value is boom


class TestCancelled:

    async def test_cancelled(self) -> None:
        """A cancelled task reports CANCELLED and the result raises TaskCancelledError."""
        async def work() -> None:
            await asyncio.sleep(10)

        task = await _settled_task(work)
        await _cancel_settled(task)
        assert task.status is TaskStatus.CANCELLED
        with pytest.raises(TaskCancelledError):
            await task.result()

    async def test_result_on_cancelled_raises(self) -> None:
        """Asking for the result of a task cancelled while waiting raises TaskCancelledError."""
        async def work() -> None:
            await asyncio.sleep(10)

        manager = task_manager
        task = await manager.create_task(work)
        assert task.status is TaskStatus.RUNNING

        async def cancel_later() -> None:
            await asyncio.sleep(_SHORT_WAIT)
            await task.cancel()

        asyncio.create_task(cancel_later())
        with pytest.raises(TaskCancelledError):
            await task.result()
        assert task.status is TaskStatus.CANCELLED

    async def test_swallowed_cancellation_completes(self) -> None:
        """A callable that swallows CancelledError ends COMPLETED with its returned value."""
        async def work() -> str:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                return "swallowed"

        task = await _settled_task(work)
        await _cancel_settled(task)
        assert task.status is TaskStatus.COMPLETED
        assert await task.result() == "swallowed"


class TestCreateWithExplicitId:
    """create must use the task id the caller gives."""

    async def test_create_with_explicit_id_returns_same_id(self) -> None:
        expected = uuid.uuid4()

        async def work() -> str:
            return "done"

        manager = task_manager
        task = await manager.create_task(work, task_id=expected)
        assert task.id == expected
        found = await manager.get(expected)
        assert found is task

    async def test_create_without_id_still_generates(self) -> None:
        manager = task_manager
        task = await manager.create_task(_quick_result)
        assert task.id != uuid.UUID(int=0)
        assert await manager.get(task.id) is task


class TestRunHarness:
    async def test_run_writes_row_at_start_and_persists_terminal_state(self) -> None:
        manager = task_manager
        task = await manager.create_task(_quick_result)
        await task.result()
        row = await store_module.task_store.get(str(task.id))
        assert row is not None and row.status == "completed"
        assert row.parent_task_id is None
        assert row.run_dir == f"runs/{task.id}"

    async def test_failed_run_persists_error(self) -> None:
        manager = task_manager

        async def boom() -> str:
            raise RuntimeError("boom")

        task = await manager.create_task(boom)
        with pytest.raises(RuntimeError, match="boom"):
            await task.result()
        row = await store_module.task_store.get(str(task.id))
        assert row is not None and row.status == "failed"
        assert row.error == "boom"

    async def test_timed_out_run_persists_explicit_error(self) -> None:
        manager = task_manager

        async def slow() -> str:
            raise asyncio.TimeoutError()

        task = await manager.create_task(slow)
        with pytest.raises(asyncio.TimeoutError):
            await task.result()
        row = await store_module.task_store.get(str(task.id))
        assert row is not None and row.status == "failed"
        assert row.error
        assert row.error == "timed out"

    async def test_cancel_before_first_step_leaves_no_row(self) -> None:
        manager = task_manager
        task = await manager.create_task(_slow_never)
        await task.cancel()
        await task.wait_started()
        assert task.status is TaskStatus.CANCELLED
        assert await store_module.task_store.get(str(task.id)) is None

    async def test_cancelled_child_before_first_step_leaves_no_row_or_registration(self) -> None:
        manager = task_manager
        child_box: dict[str, Task] = {}

        async def spawn_and_cancel() -> None:
            child_box["t"] = await manager.create_task(_slow_never)
            await child_box["t"].cancel()

        parent = await manager.create_task(spawn_and_cancel)
        await parent.result()
        child = child_box["t"]
        await child.wait_started()
        assert child.status is TaskStatus.CANCELLED
        assert await store_module.task_store.get(str(child.id)) is None

    async def test_cancel_during_row_write_leaves_no_row_or_registration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancel during the child's blocked insert leaves no row and no registration."""
        db_path = str(tmp_path / "records.db")
        lock_held = threading.Event()
        release_lock = threading.Event()

        def hold_write_lock() -> None:
            conn = sqlite3.connect(db_path, timeout=5, isolation_level=None)
            try:
                conn.execute("BEGIN")
                conn.execute(
                    "INSERT INTO task_records (task_id, status, started_at, run_dir)"
                    " VALUES ('__lock_holder__', 'running', '1970-01-01T00:00:00+00:00', 'runs/__lock_holder__')"
                )
                lock_held.set()
                release_lock.wait(_LOCK_HOLD_MAX)
            finally:
                conn.rollback()
                conn.close()

        store = store_module.task_store
        original_insert = store.insert_running
        original_delete = store.delete
        insert_started = asyncio.Event()

        async def spy_insert(task_id: str, parent_task_id: str | None, started_at: str, run_dir: str) -> None:
            insert_started.set()
            await original_insert(task_id, parent_task_id, started_at, run_dir)

        async def spy_delete(task_id: str) -> None:
            release_lock.set()  # the cancel reached the child
            await original_delete(task_id)

        manager = task_manager
        child_box: dict[str, Task] = {}
        lock_holder = threading.Thread(target=hold_write_lock, daemon=True)

        async def spawn() -> str:
            lock_holder.start()
            await asyncio.to_thread(lock_held.wait)
            monkeypatch.setattr(store, "insert_running", spy_insert)
            monkeypatch.setattr(store, "delete", spy_delete)
            child_box["t"] = await manager.create_task(_slow_never)
            await asyncio.wait_for(insert_started.wait(), _LOCK_HOLD_MAX)
            await child_box["t"].cancel()
            await child_box["t"].wait_started()
            return "parent"

        parent = await manager.create_task(spawn)
        await parent.result()
        await asyncio.to_thread(lock_holder.join)
        await store_module.task_store.finish(str(uuid.uuid4()), "completed")
        child = child_box["t"]
        assert release_lock.is_set()  # the cancel reached the child
        assert child.status is TaskStatus.CANCELLED
        assert await store_module.task_store.get(str(child.id)) is None

    async def test_double_cancel_during_persist_still_reaches_terminal_state(self) -> None:
        manager = task_manager
        task = await manager.create_task(_slow_never)
        await task.wait_started()

        original_finish = store_module.task_store.finish

        async def slow_finish(
            task_id: str, status: str, *, error: str | None = None, ended_at: str | None = None
        ) -> None:
            await asyncio.sleep(0.05)
            await original_finish(task_id, status, error=error, ended_at=ended_at)

        store_module.task_store.finish = slow_finish
        recancel: asyncio.Task[None]
        try:
            await task.cancel()
            await asyncio.sleep(_PERSIST_SETTLE_WAIT)
            recancel = asyncio.create_task(task.cancel())
        finally:
            store_module.task_store.finish = original_finish

        with pytest.raises(TaskCancelledError):
            await task.result()
        assert task.status is TaskStatus.CANCELLED
        row = await store_module.task_store.get(str(task.id))
        deadline = time.monotonic() + 2
        while row is None or row.status == "running":
            assert time.monotonic() < deadline, f"row did not reach a terminal state: {row}"
            await asyncio.sleep(_SHORT_WAIT)
            row = await store_module.task_store.get(str(task.id))
        assert row.status == "cancelled"
        assert row.error == "task cancelled"
        await recancel

    async def test_recancel_during_failed_persist_keeps_failed_outcome(self) -> None:
        """A re-cancel landing while the failed persist is shielded cannot replace the failure with a cancellation."""
        boom = RuntimeError("boom")

        manager = task_manager
        task = await manager.create_task(_raises_after(_PRE_FAIL_DELAY, boom))
        await task.wait_started()
        await _recancel_when_persisting(task, "failed")
        with pytest.raises(RuntimeError, match="boom") as exc_info:
            await task.result()
        assert exc_info.value is boom
        assert task.status is TaskStatus.FAILED
        row = await store_module.task_store.get(str(task.id))
        assert row is not None and row.status == "failed"
        assert row.error == "boom"

    async def test_recancel_during_completed_persist_keeps_completed_outcome(self) -> None:
        """A re-cancel landing while the completed persist is shielded cannot replace the success with a cancellation."""
        async def work() -> str:
            await asyncio.sleep(_PRE_FAIL_DELAY)
            return "done"

        manager = task_manager
        task = await manager.create_task(work)
        await task.wait_started()
        await _recancel_when_persisting(task, "completed")
        assert await task.result() == "done"
        assert task.status is TaskStatus.COMPLETED
        row = await store_module.task_store.get(str(task.id))
        assert row is not None and row.status == "completed"
        assert row.error is None

    async def test_wait_started_returns_once_row_exists(self) -> None:
        manager = task_manager
        task = await manager.create_task(_slow_never)
        await task.wait_started()
        row = await store_module.task_store.get(str(task.id))
        assert row is not None and row.status == "running"
        await task.cancel()

    async def test_child_registers_under_parent(self) -> None:
        manager = task_manager
        child_box: dict[str, Task] = {}

        async def child_work() -> str:
            return "child"

        parent = await manager.create_task(_spawns_child(child_box, child_work, "parent"))
        await parent.result()
        child = child_box["t"]
        row = await store_module.task_store.get(str(child.id))
        assert row is not None and row.parent_task_id == str(parent.id)

    async def test_grandchild_registers_under_child_not_grandparent(self) -> None:
        manager = task_manager
        child_box: dict[str, Task] = {}
        grand_box: dict[str, Task] = {}

        async def grand_work() -> str:
            return "grand"

        child_work = _spawns_child(grand_box, grand_work, "child")
        parent = await manager.create_task(_spawns_child(child_box, child_work, "parent"))
        await parent.result()
        child, grand = child_box["t"], grand_box["t"]
        grand_row = await store_module.task_store.get(str(grand.id))
        assert grand_row is not None and grand_row.parent_task_id == str(child.id)
        subs = await store_module.task_store.list_subtasks(str(parent.id))
        assert [s.task_id for s in subs] == [str(child.id)]


class TestRunPersistence:
    """Every task gets a run folder, a log, and a result file from the task layer."""

    async def test_create_makes_run_folder_and_log_file(self) -> None:
        task = await task_manager.create_task(_quick_result)
        await task.result()
        row = await store_module.task_store.get(str(task.id))
        assert row is not None and row.run_dir == f"runs/{task.id}"
        run_path = os.path.join(store_module.task_store.data_dir, row.run_dir)
        assert os.path.isdir(run_path)
        assert os.path.isfile(os.path.join(run_path, "log"))

    async def test_prints_land_in_run_log(self) -> None:
        async def _printing() -> None:
            print("[probe] running")
            await asyncio.sleep(_SHORT_WAIT)

        logcapture.install_log_tee()
        try:
            task = await task_manager.create_task(_printing)
            await task.result()
        finally:
            logcapture.restore_log_tee()
        row = await store_module.task_store.get(str(task.id))
        log = open(os.path.join(store_module.task_store.data_dir, row.run_dir, "log")).read()
        assert "[probe] running" in log

    async def test_string_result_persisted_to_result_file(self) -> None:
        async def _text() -> str:
            await asyncio.sleep(_SHORT_WAIT)
            return "the answer"

        task = await task_manager.create_task(_text)
        assert await task.result() == "the answer"
        row = await store_module.task_store.get(str(task.id))
        result = open(os.path.join(store_module.task_store.data_dir, row.run_dir, "result")).read()
        assert result == "the answer"

    async def test_none_result_writes_no_result_file(self) -> None:
        async def _nothing() -> None:
            await asyncio.sleep(_SHORT_WAIT)

        task = await task_manager.create_task(_nothing)
        await task.result()
        row = await store_module.task_store.get(str(task.id))
        run_path = os.path.join(store_module.task_store.data_dir, row.run_dir)
        assert not os.path.exists(os.path.join(run_path, "result"))


class TestNoPersist:
    """A task created with should_persist_record=False is a plain in-memory run."""

    async def test_no_persist_writes_no_row_or_folder(self) -> None:
        async def _quick() -> str:
            await asyncio.sleep(_SETTLE_WAIT)
            return "in-mem"

        task = await task_manager.create_task(_quick, should_persist_record=False)
        assert await task.result() == "in-mem"
        assert await store_module.task_store.get(str(task.id)) is None
        run_path = os.path.join(
            store_module.task_store.data_dir, store_module.RUNS_DIR, str(task.id)
        )
        assert not os.path.exists(run_path)

    async def test_no_persist_is_registered_live_while_running(self) -> None:
        start = asyncio.Event()

        async def _slow() -> str:
            start.set()
            await asyncio.sleep(30)
            return "never"

        task = await task_manager.create_task(_slow, should_persist_record=False)
        await asyncio.wait_for(start.wait(), timeout=5)
        live = await task_manager.get(task.id)
        assert live is not None and live.status is TaskStatus.RUNNING
        await task.cancel()

    async def test_no_persist_failed_run_writes_no_row(self) -> None:
        async def _boom() -> str:
            raise RuntimeError("boom")

        task = await task_manager.create_task(_boom, should_persist_record=False)
        with pytest.raises(RuntimeError):
            await task.result()
        assert await store_module.task_store.get(str(task.id)) is None


class TestStringifyResult:
    """When stringify_result is given, the run's result file holds that text."""

    async def test_stringify_result_is_used_for_the_result_file(self) -> None:
        async def _obj() -> dict:
            await asyncio.sleep(_SETTLE_WAIT)
            return {"body": "hello"}

        task = await task_manager.create_task(_obj, stringify_result=lambda v: v["body"])
        assert await task.result() == {"body": "hello"}
        row = await store_module.task_store.get(str(task.id))
        result = open(
            os.path.join(store_module.task_store.data_dir, row.run_dir, "result")
        ).read()
        assert result == "hello"

    async def test_none_stringify_falls_back_to_str(self) -> None:
        async def _obj() -> dict:
            await asyncio.sleep(_SETTLE_WAIT)
            return {"body": "hello"}

        task = await task_manager.create_task(_obj)
        await task.result()
        row = await store_module.task_store.get(str(task.id))
        result = open(
            os.path.join(store_module.task_store.data_dir, row.run_dir, "result")
        ).read()
        assert result == str({"body": "hello"})


async def _wait_terminal(task: Task, timeout: float = 5.0) -> None:
    """Poll until the task leaves the running state."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while task.status is TaskStatus.RUNNING:
        if loop.time() > deadline:
            raise AssertionError("task did not reach a terminal state")
        await asyncio.sleep(0.01)


async def _parent_with_live_child() -> Task:
    """Build a running parent with one running child, and wait for the child's row."""
    child_started = asyncio.Event()

    async def child() -> None:
        await asyncio.sleep(30)

    async def parent() -> None:
        await task_manager.create_task(child)
        child_started.set()
        await asyncio.sleep(30)

    parent_task = await task_manager.create_task(parent)
    await asyncio.wait_for(child_started.wait(), 5)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while not await store_module.task_store.list_subtasks(str(parent_task.id)):
        if loop.time() > deadline:
            raise AssertionError("subtask record never appeared")
        await asyncio.sleep(0.01)
    return parent_task


async def test_cancel_running_task() -> None:
    """Cancelling a running task ends it CANCELLED and saves its record as cancelled."""
    async def slow() -> None:
        await asyncio.sleep(30)

    task = await task_manager.create_task(slow)
    await asyncio.wait_for(task.wait_started(), _RESULT_WAIT)
    await task.cancel()
    assert task.status is TaskStatus.CANCELLED
    record = await store_module.task_store.get(str(task.id))
    assert record.status == "cancelled"


async def test_cancel_non_blocking() -> None:
    """A cancel started in the background converges without blocking the caller."""
    async def slow() -> None:
        await asyncio.sleep(30)

    task = await task_manager.create_task(slow)
    await asyncio.wait_for(task.wait_started(), _RESULT_WAIT)
    worker = asyncio.create_task(task.cancel())
    assert not worker.done()
    await worker
    await _wait_terminal(task)
    assert task.status is TaskStatus.CANCELLED


async def test_cancel_swallowing_task_ends_completed() -> None:
    """A callable that swallows the stop ends COMPLETED and its record says completed."""
    async def swallow() -> str:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            return "done anyway"

    task = await task_manager.create_task(swallow)
    await asyncio.wait_for(task.wait_started(), _RESULT_WAIT)
    await task.cancel()
    assert task.status is TaskStatus.COMPLETED
    record = await store_module.task_store.get(str(task.id))
    assert record.status == "completed"


async def test_cancel_recurses_into_live_subtasks() -> None:
    """Cancelling a live parent cancels every live subtask under it."""
    parent_task = await _parent_with_live_child()
    await parent_task.cancel()
    record = await store_module.task_store.get(str(parent_task.id))
    assert record.status == "cancelled"
    assert all(sub.status == "cancelled" for sub in await store_module.task_store.list_subtasks(str(parent_task.id)))


async def test_manager_cancel_recurses_into_live_subtasks() -> None:
    """Cancelling by id cancels every live subtask under the run."""
    parent_task = await _parent_with_live_child()
    await task_manager.cancel(str(parent_task.id))
    record = await store_module.task_store.get(str(parent_task.id))
    assert record.status == "cancelled"
    assert all(sub.status == "cancelled" for sub in await store_module.task_store.list_subtasks(str(parent_task.id)))


async def test_cancel_already_terminal_is_noop() -> None:
    """Cancelling a finished task leaves its record untouched."""
    async def quick() -> int:
        await asyncio.sleep(_SETTLE_WAIT)
        return 1

    task = await task_manager.create_task(quick)
    await task.result()
    await task.cancel()
    record = await store_module.task_store.get(str(task.id))
    assert record.status == "completed"


async def test_cancel_unknown_id_is_noop() -> None:
    """Cancelling a bad or unknown id is a no-op."""
    await task_manager.cancel("not-a-uuid")
    await task_manager.cancel(str(uuid.uuid4()))


_TASK_PUBLIC_MEMBERS = {"id", "status", "cancel", "wait_started", "result"}


class TestTaskHierarchy:
    """Task is abstract, AsyncTask and the completed handle extend it."""

    def test_task_is_abstract(self) -> None:
        """Instantiating the abstract Task raises TypeError."""
        with pytest.raises(TypeError):
            Task()  # type: ignore[abstract]

    def test_task_surface_is_the_five_public_members(self) -> None:
        """Task exposes exactly id, status, cancel, wait_started and result."""
        public = {name for name in dir(Task) if not name.startswith("_")}
        assert public == _TASK_PUBLIC_MEMBERS

    async def test_create_task_returns_an_asynctask(self) -> None:
        """The tracked factory hands back a concrete AsyncTask that is a Task."""
        task = await task_manager.create_task(_quick_result)
        try:
            assert isinstance(task, AsyncTask)
            assert isinstance(task, Task)
        finally:
            await task.cancel()

    async def test_create_task_rejects_name_and_log_retriever(self) -> None:
        """create_task rejects a name or log_retriever argument."""
        with pytest.raises(TypeError):
            await task_manager.create_task(_quick_result, name="x")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            await task_manager.create_task(_quick_result, log_retriever=None)  # type: ignore[call-arg]


class TestImmediateTask:
    """The untracked immediate handle the completed factory returns, with a result or an error."""

    async def test_result_flavor_reports_completed(self) -> None:
        """A fresh handle reports COMPLETED, starts immediately, and returns its result."""
        task = task_manager.create_completed_task("text")
        assert isinstance(task, Task)
        assert isinstance(task.id, uuid.UUID)
        assert task.status is TaskStatus.COMPLETED
        await asyncio.wait_for(task.wait_started(), timeout=_SETTLE_WAIT)
        assert await task.result() == "text"
        await asyncio.wait_for(task.cancel(), timeout=_SETTLE_WAIT)
        assert task.status is TaskStatus.COMPLETED

    async def test_result_flavor_is_untracked(self) -> None:
        """The handle writes no db row, no run folder, and is not registered live."""
        task = task_manager.create_completed_task("text")
        assert await task_manager.get(task.id) is None
        assert await store_module.task_store.get(str(task.id)) is None
        run_path = os.path.join(store_module.task_store.data_dir, store_module.RUNS_DIR, str(task.id))
        assert not os.path.exists(run_path)

    async def test_task_id_adopted_verbatim(self) -> None:
        """An explicit task id is adopted as-is rather than replaced by a fresh one."""
        given = uuid.uuid4()
        task = task_manager.create_completed_task("text", task_id=given)
        assert task.id == given

    async def test_error_flavor_reports_failed(self) -> None:
        """A handle holding an error reports FAILED, and its fresh id names no stored run."""
        boom = ValueError("boom")
        task = task_manager.create_completed_task(error=boom)
        assert isinstance(task, ImmediateTask)
        assert task.has_stored_run is False
        assert task.status is TaskStatus.FAILED

    async def test_error_flavor_result_raises_original(self) -> None:
        """Asking for the result of an error handle raises the very error it holds."""
        boom = ValueError("boom")
        task = task_manager.create_completed_task(error=boom)
        with pytest.raises(ValueError, match="boom") as exc_info:
            await task.result()
        assert exc_info.value is boom

    async def test_error_flavor_wait_started_returns_immediately(self) -> None:
        """An error handle is already finished, so wait_started returns without waiting."""
        task = task_manager.create_completed_task(error=ValueError("boom"))
        await asyncio.wait_for(task.wait_started(), timeout=_SETTLE_WAIT)

    async def test_error_flavor_cancel_is_noop(self) -> None:
        """Cancelling an error handle does nothing and leaves it FAILED."""
        task = task_manager.create_completed_task(error=ValueError("boom"))
        await asyncio.wait_for(task.cancel(), timeout=_SETTLE_WAIT)
        assert task.status is TaskStatus.FAILED

    async def test_error_flavor_is_untracked(self) -> None:
        """The error handle writes no db row, no run folder, and is not registered live."""
        task = task_manager.create_completed_task(error=ValueError("boom"))
        assert await task_manager.get(task.id) is None
        assert await store_module.task_store.get(str(task.id)) is None
        run_path = os.path.join(store_module.task_store.data_dir, store_module.RUNS_DIR, str(task.id))
        assert not os.path.exists(run_path)

    async def test_result_and_error_together_raise(self) -> None:
        """The factory refuses a handle holding both a result and an error."""
        with pytest.raises(ValueError, match="not both"):
            task_manager.create_completed_task("text", error=ValueError("boom"))

    @pytest.mark.parametrize("adopt", [False, True])
    def test_has_stored_run_tracks_the_id_source(self, adopt: bool) -> None:
        """Only an adopted id names a stored run, a freshly generated id names none."""
        task = task_manager.create_completed_task("text", task_id=uuid.uuid4() if adopt else None)
        assert task.has_stored_run is adopt
