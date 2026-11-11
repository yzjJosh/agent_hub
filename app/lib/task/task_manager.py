from __future__ import annotations

import asyncio
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from typing import override

import aiofiles

from app.lib.db import store as db_store
from app.lib.task.logcapture import LogBuffer, active_log_buffer

__all__ = [
    "AsyncTask",
    "ImmediateTask",
    "Task",
    "TaskCancelledError",
    "TaskStatus",
    "task_manager",
]


async def _persist_result[T](
    run_path: str, result: T, stringify_result: Callable[[T], str] | None = None
) -> None:
    """Save the runner's return value to the run's result file.

    Args:
        run_path: Folder holding the run files.
        result: The value the runner returned.
        stringify_result: How to turn the value into the text body to persist, or
            None to fall back to str(result).
    """
    text = stringify_result(result) if stringify_result is not None else str(result)
    async with aiofiles.open(os.path.join(run_path, "result"), "w", encoding="utf-8") as fh:
        await fh.write(text)


class TaskStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskCancelledError(Exception):
    """Raised when the result is asked for on a cancelled task."""


class Task[T](ABC):
    """Abstract handle to a run, the public surface the task manager hands out."""

    @property
    @abstractmethod
    def id(self) -> uuid.UUID:
        """The run's id."""
        ...

    @property
    @abstractmethod
    def status(self) -> TaskStatus:
        """The run's current status."""
        ...

    @abstractmethod
    async def cancel(self) -> None:
        """Ask the run to stop and wait for it to reach a terminal state."""
        ...

    @abstractmethod
    async def wait_started(self) -> None:
        """Wait until the run has started or ended before it could start."""
        ...

    @abstractmethod
    async def result(self) -> T | None:
        """Wait for the run to finish and return its result, or raise its error."""
        ...


async def _await_terminal(live: AsyncTask) -> None:
    """Wait for a live task to finish, whatever the outcome."""
    try:
        await live.result()
    except Exception:
        pass


async def _live_task(manager: _TaskManager, task_id: str) -> AsyncTask | None:
    """The live task for task_id, or None when the id is bad, unknown, or already done."""
    try:
        return await manager.get(uuid.UUID(task_id))
    except ValueError:
        return None


async def _cancel_tree(task: AsyncTask, manager: _TaskManager) -> None:
    """Cancel a task and every live subtask under it, awaiting the tree to terminal state."""
    if task.status is TaskStatus.RUNNING:
        task._runner.cancel()
    await _await_terminal(task)

    async def recurse(task_id: str) -> None:
        for child in await db_store.task_store.list_subtasks(task_id):
            live = await _live_task(manager, child.task_id)
            if live is not None and live.status is TaskStatus.RUNNING:
                await _cancel_tree(live, manager)
            else:
                await recurse(child.task_id)

    await recurse(str(task.id))


class AsyncTask[T](Task[T]):
    """A live tracked run spawned by the task manager."""

    def __init__(
        self,
        task_id: uuid.UUID,
        runner: asyncio.Task[T],
    ) -> None:
        self._id = task_id
        self._runner = runner
        self._status: TaskStatus = TaskStatus.RUNNING
        self._result: T | None = None
        self._error: BaseException | None = None
        self._start_signal = asyncio.Event()
        self._runner.add_done_callback(self._on_done)

    @override
    @property
    def id(self) -> uuid.UUID:
        return self._id

    @override
    @property
    def status(self) -> TaskStatus:
        return self._status

    @override
    async def wait_started(self) -> None:
        """Wait until the run has started or the task ended before it could start."""
        await self._start_signal.wait()

    async def _run_shielded(self, coro: Awaitable[None]) -> None:
        """Run coro even when this task is cancelled.

        Every error is ignored. The work always runs to the end, even when
        the task is cancelled again while it is finishing.
        """
        try:
            await asyncio.shield(coro)
        except (Exception, asyncio.CancelledError):
            pass

    async def _persist_terminal(self, status: str, error: str | None) -> None:
        """Save the task's end state, even when the task is cancelled again."""
        await self._run_shielded(
            db_store.task_store.finish(
                str(self.id), status, error=error, ended_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    async def _remove_row(self) -> None:
        """Remove the run-start record left behind by a cancelled insert."""
        await self._run_shielded(db_store.task_store.delete(str(self.id)))

    def _on_done(self, runner: asyncio.Task[T]) -> None:
        if not self._start_signal.is_set():
            self._start_signal.set()
        if runner.cancelled():
            self._status = TaskStatus.CANCELLED
        elif runner.exception() is not None:
            self._status = TaskStatus.FAILED
            self._error = runner.exception()
        else:
            self._status = TaskStatus.COMPLETED
            self._result = runner.result()

    @override
    async def result(self) -> T | None:
        """Wait for the task to finish and return its result.

        Returns:
            The value returned by the wrapped callable.

        Raises:
            TaskCancelledError: If the task was cancelled.
            The original exception: If the task failed.
        """
        if self._status is TaskStatus.RUNNING:
            await asyncio.wait({self._runner})
        if self._status is TaskStatus.CANCELLED:
            raise TaskCancelledError(f"task {self._id} was cancelled")
        if self._error is not None:
            raise self._error
        return self._result

    @override
    async def cancel(self) -> None:
        """Ask the task and every live subtask under it to stop, and wait for the tree to finish.

        If the callable catches the stop and keeps going, the task ends
        COMPLETED or FAILED instead of CANCELLED.
        """
        if self._status is TaskStatus.RUNNING:
            await _cancel_tree(self, task_manager)


class ImmediateTask[T](Task[T]):
    """A one-off completed handle that is never tracked, persisted, or cancellable.

    Holding an error reports FAILED and raises it. Otherwise it reports COMPLETED
    and returns its result.
    """

    def __init__(
        self,
        task_id: uuid.UUID | None = None,
        result: T | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._has_stored_run = task_id is not None
        self._id = task_id or uuid.uuid4()
        self._result = result
        self._error = error
        self._status = TaskStatus.FAILED if error is not None else TaskStatus.COMPLETED

    @override
    @property
    def id(self) -> uuid.UUID:
        return self._id

    @override
    @property
    def status(self) -> TaskStatus:
        return self._status

    @property
    def has_stored_run(self) -> bool:
        """Whether the id was adopted from an earlier stored run.

        Only the cache-hit flavour does. A handle built from an earlier run
        reuses that run's id. A handle with a freshly generated id names no
        stored run.
        """
        return self._has_stored_run

    @override
    async def wait_started(self) -> None:
        """Return without waiting."""

    @override
    async def result(self) -> T | None:
        """Return the held result, or raise the held error.

        Raises:
            The original error: If the handle holds one.
        """
        if self._error is not None:
            raise self._error
        return self._result

    @override
    async def cancel(self) -> None:
        """Do nothing."""


_active_task: ContextVar[AsyncTask | None] = ContextVar("active_task", default=None)


class _TaskManager:
    """Spawns async callables as tracked, cancellable tasks."""

    def __init__(self) -> None:
        self._tasks: dict[uuid.UUID, AsyncTask] = {}

    async def create_task[T](
        self,
        func: Callable[[], Awaitable[T]],
        *,
        task_id: uuid.UUID | None = None,
        should_persist_record: bool = True,
        stringify_result: Callable[[T], str] | None = None,
    ) -> AsyncTask[T]:
        """Spawn func as a background task and track it while it runs.

        A persisted task gets a run folder holding its log, a runner that returns
        a value gets a result file holding it, and a run record is written. A
        non-persisted task is a plain in-memory run. It is still spawned,
        tracked while live, and started, but it writes no record, folder, log,
        or result.

        Args:
            func: Async callable that takes nothing and makes the result.
            task_id: Id to use. A new random id is made when None.
            should_persist_record: Whether the run writes a record and run folder.
            stringify_result: How to turn the result into the persisted text, or
                None to fall back to str.

        Returns:
            An AsyncTask handle, right away.
        """
        tid = task_id or uuid.uuid4()
        run_dir = f"{db_store.RUNS_DIR}/{tid}"
        run_path = os.path.join(db_store.task_store.data_dir, run_dir)
        if should_persist_record:
            await asyncio.to_thread(os.makedirs, run_path, exist_ok=True)
        buffer = (
            await LogBuffer.create(os.path.join(run_path, "log"))
            if should_persist_record else None
        )
        task_holder: dict[str, AsyncTask[T]] = {}

        async def _harness() -> T:
            run: AsyncTask = task_holder["task"]
            parent: Task | None = self.active_task()
            if should_persist_record:
                try:
                    await db_store.task_store.insert_running(
                        str(run.id),
                        str(parent.id) if parent is not None else None,
                        datetime.now(timezone.utc).isoformat(),
                        run_dir,
                    )
                except BaseException:
                    await run._remove_row()
                    await run._run_shielded(buffer.close())
                    raise
            try:
                run._start_signal.set()
                token = _active_task.set(run)
                log_token = active_log_buffer.set(buffer)
                try:
                    result = await func()
                finally:
                    active_log_buffer.reset(log_token)
                    _active_task.reset(token)
                if should_persist_record:
                    if result is not None:
                        await run._run_shielded(
                            _persist_result(run_path, result, stringify_result)
                        )
                    await run._persist_terminal("completed", None)
                return result
            except asyncio.CancelledError:
                if should_persist_record:
                    await run._persist_terminal("cancelled", "task cancelled")
                raise
            except Exception as exc:
                if should_persist_record:
                    error = str(exc)
                    if not error and isinstance(exc, asyncio.TimeoutError):
                        error = "timed out"
                    await run._persist_terminal("failed", error)
                raise
            finally:
                if should_persist_record:
                    await run._run_shielded(buffer.close())

        runner = asyncio.create_task(_harness())
        task = AsyncTask(tid, runner)
        task_holder["task"] = task
        self._tasks[task.id] = task
        runner.add_done_callback(lambda _: self._tasks.pop(task.id, None))
        return task

    def create_completed_task[T](
        self,
        result: T | None = None,
        *,
        task_id: uuid.UUID | None = None,
        error: BaseException | None = None,
    ) -> ImmediateTask[T]:
        """Return an immediate handle, completed with a result or failed with an error.

        The handle is never tracked or persisted.

        Args:
            result: The value the handle reports as its result.
            task_id: Id to adopt as-is. A new random id is made when None.
            error: The error the handle reports as FAILED and raises.

        Returns:
            A handle reporting COMPLETED with the result, or FAILED with the error.

        Raises:
            ValueError: If both a result and an error are given.
        """
        if result is not None and error is not None:
            raise ValueError("create_completed_task takes a result or an error, not both")
        return ImmediateTask(task_id, result, error)

    def active_task(self) -> AsyncTask | None:
        """Return the task running in the current asyncio context, or None."""
        return _active_task.get()

    async def get(self, task_id: uuid.UUID) -> AsyncTask | None:
        """Return the task if it is still running, else None."""
        return self._tasks.get(task_id)

    async def cancel(self, task_id: str) -> None:
        """Cancel the task and every live subtask under it, awaiting the tree to terminal state.

        Args:
            task_id: The run's id. Bad or unknown ids are a no-op.
        """
        live = await _live_task(self, task_id)
        if live is None or live.status is not TaskStatus.RUNNING:
            return
        await _cancel_tree(live, self)


task_manager = _TaskManager()
