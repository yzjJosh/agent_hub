import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override
from unittest.mock import MagicMock

import pytest
from agents.stream_events import RawResponsesStreamEvent
from openai.types.responses import ResponseTextDeltaEvent

import app.lib.agent.engine as engine
import app.lib.db.store as store_module
from app.lib.agent.engine import Agent, AgentRun
from app.lib.task.task_manager import Task, TaskStatus
from app.lib.tool.base import Param, Tool, ToolResult
from tests.app.conftest import store_infra

_POLL = 0.01

def text_event(text: str) -> RawResponsesStreamEvent:
    """Build one streamed text delta event."""
    data = ResponseTextDeltaEvent(
        type="response.output_text.delta",
        content_index=0,
        item_id="i0",
        logprobs=[],
        output_index=0,
        sequence_number=1,
        delta=text,
    )
    return RawResponsesStreamEvent(data=data)


def stub_result(
    events: list[RawResponsesStreamEvent],
    final_output: str | None,
    role: str,
) -> MagicMock:
    """Build a stub result that replays the events and ends with the final output."""
    async def _replay() -> AsyncIterator[RawResponsesStreamEvent]:
        for event in events:
            yield event

    stub = MagicMock()
    stub.final_output = final_output
    stub.stream_events = MagicMock(side_effect=_replay)
    stub.to_input_list = MagicMock(return_value=[{"role": "user", "content": role}])
    stub.cancel = MagicMock()
    return stub


def stalled_stub() -> MagicMock:
    """Build a stub result whose event stream never ends."""
    async def _hold() -> AsyncIterator[RawResponsesStreamEvent]:
        await asyncio.Event().wait()

    return MagicMock(
        final_output=None,
        stream_events=MagicMock(side_effect=_hold),
        to_input_list=MagicMock(return_value=[]),
        cancel=MagicMock(),
    )


def patch_runner(monkeypatch: pytest.MonkeyPatch, stub: MagicMock) -> None:
    """Point the engine's runner at the given stub."""
    monkeypatch.setattr(engine.Runner, "run_streamed", MagicMock(return_value=stub))


def track_agent_runs(monkeypatch: pytest.MonkeyPatch) -> list[Task]:
    """Record each agent task the roles spawn, and return the list."""
    spawned: list[Task] = []

    def _record(task: Task) -> AgentRun:
        spawned.append(task)
        return AgentRun(task)

    monkeypatch.setattr(engine, "AgentRun", _record)
    return spawned


@dataclass
class CannedCall:
    """One in-loop tool call served by the canned run."""

    tool_name: str
    params: dict
    status: str = "completed"
    run_dir: str = "runs/call-1"
    task_id: str = "call-1"


@dataclass
class CannedEvent:
    """One conversation event served by the canned run."""

    type: str
    tool_name: str | None = None
    params: dict | None = None


class CannedRun:
    """An AgentRun stand-in that replays canned calls and conversation events."""

    def __init__(self, task: Task, calls: tuple[CannedCall, ...], events: tuple[CannedEvent, ...]) -> None:
        self._task = task
        self._calls = list(calls)
        self._events = list(events)

    @property
    def task(self) -> Task:
        return self._task

    async def tool_calls(self) -> list[CannedCall]:
        return list(self._calls)

    async def events(self) -> list[CannedEvent]:
        return list(self._events)

    async def conversation(self) -> AsyncIterator[CannedEvent]:
        for event in self._events:
            yield event


def patch_agent_run(
    monkeypatch: pytest.MonkeyPatch,
    calls: tuple[CannedCall, ...] = (),
    events: tuple[CannedEvent, ...] = (),
) -> None:
    """Point the engine's AgentRun at a canned run that wraps each real task."""

    def _canned(task: Task) -> CannedRun:
        return CannedRun(task, calls, events)

    monkeypatch.setattr(engine, "AgentRun", _canned)


def search_result_for(run_id: str) -> str:
    """Build one-hit search result text that surfaces the given run id."""
    return (
        "## Evidence search: q (1 result)\n"
        "\n"
        "### 1. web_fetch - score 0.93\n"
        f"run_id: `{run_id}`\n"
        "> snippet\n"
    )


def write_search_result(run_dir: str, text: str) -> None:
    """Write a search run result file under the throwaway task store."""
    dir_path = Path(store_module.task_store.data_dir, run_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "result").write_text(text, encoding="utf-8")


def record_user_messages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the user prompt of each agent run, and return the list."""
    messages: list[str] = []
    run = Agent.run

    async def _recording_run(self, user_prompt: str) -> AgentRun:
        messages.append(user_prompt)
        return await run(self, user_prompt)

    monkeypatch.setattr(Agent, "run", _recording_run)
    return messages


async def wait_agent_task(tasks: list[Task], message: str) -> Task:
    """Wait for the first agent task to appear, then return it."""
    for _ in range(500):
        if tasks:
            return tasks[0]
        await asyncio.sleep(_POLL)
    raise AssertionError(message)


async def wait_cancelled(task: Task, message: str) -> None:
    """Wait for a task to reach the cancelled status."""
    for _ in range(500):
        if task.status is TaskStatus.CANCELLED:
            return
        await asyncio.sleep(_POLL)
    raise AssertionError(message)


async def assert_cancels_agent_task(
    monkeypatch: pytest.MonkeyPatch,
    launch: Callable[[], Awaitable],
    role: str,
) -> None:
    """Cancel a stalled role run and check that its agent task is cancelled."""
    patch_runner(monkeypatch, stalled_stub())
    tasks = track_agent_runs(monkeypatch)
    outer = asyncio.create_task(launch())
    agent_task = await wait_agent_task(tasks, f"{role} never launched its agent task")
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    await wait_cancelled(agent_task, f"task {agent_task.id} did not reach CANCELLED")
    assert agent_task.status is TaskStatus.CANCELLED


class FakeTool(Tool):
    name: ClassVar[str] = "fake"
    description: ClassVar[str] = "A fake tool."
    summary: ClassVar[str] = "A fake tool."
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="ok")


@pytest.fixture(autouse=True)
async def research_store_infra(tmp_path: Path) -> AsyncIterator[None]:
    """Point the process-wide stores at a throwaway database for the test."""
    async for _ in store_infra(tmp_path):
        yield
