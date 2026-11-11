import asyncio
import time
from collections.abc import AsyncIterator
from typing import Protocol

import pytest

import app.lib.db.store as store_module
from app.lib.task import ImmediateTask, Task, TaskStatus
from app.lib.tool.base import ParamValue, Tool, ToolMessageError
from app.lib.tool.truncation import DEFAULT_TOOL_OUTPUT_LEN_LIMIT, MIN_TOOL_OUTPUT_LEN_LIMIT
from app.mcp.registry import mcp
from app.tools.task_tool import task_tool

_WAIT_TERMINAL_INTERVAL = 0.02
_WAIT_TIMEOUT_S = 10.0


async def run_text(tool: Tool, /, *args: ParamValue, **kwargs: ParamValue) -> str:
    """Run a tool the way its client does and give back the text it shows.

    This is what a test asks for when it wants the outcome of a call rather than
    the handle the call returned. Positional args map onto the tool's params in
    declared order.
    """
    params = dict(zip((p.name for p in tool.params), args), **kwargs)
    return await tool.result_text(await tool.run(params))


async def assert_rejected_task(task: Task, expected: str) -> None:
    """Check the handle a rejected call returns and the words it raises.

    A rejected call's handle is an immediate one with no stored run, reports
    failed, and raises the tool's own words as a ToolMessageError.
    """
    assert isinstance(task, ImmediateTask)
    assert task.has_stored_run is False
    assert task.status is TaskStatus.FAILED
    with pytest.raises(ToolMessageError) as exc_info:
        await task.result()
    assert str(exc_info.value) == expected


async def assert_max_char_wire_contract(tool: Tool, tool_name: str) -> None:
    """Check the max_char param's wire text and schema.

    The param is checked on the served wire, not on the tool's own params.
    The wire description carries the default value and the schema carries the
    smallest allowed value. The smallest allowed value is checked as a number,
    not as text. Neither text names the constants.
    """
    assert tool.is_result_long
    wire = next(t for t in await mcp.list_tools() if t.name == tool_name)
    prop = wire.to_mcp_tool().model_dump()["inputSchema"]["properties"]["max_char"]
    assert str(DEFAULT_TOOL_OUTPUT_LEN_LIMIT) in prop["description"]
    assert "DEFAULT_TOOL_OUTPUT_LEN_LIMIT" not in prop["description"]
    assert "MIN_TOOL_OUTPUT_LEN_LIMIT" not in prop["description"]
    assert prop["minimum"] == MIN_TOOL_OUTPUT_LEN_LIMIT


async def await_task_result(task_id: str, timeout_s: float = _WAIT_TIMEOUT_S) -> str:
    """Wait for the run to finish, then give back its stored result."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        row = await store_module.task_store.get(task_id)
        if row is not None and row.status != "running":
            return await run_text(task_tool, action="result", task_id=task_id)
        await asyncio.sleep(_WAIT_TERMINAL_INTERVAL)
    raise TimeoutError(f"task {task_id} did not finish within {timeout_s}s")


class _TaskStore(Protocol):
    """The task record store the fixture gives to a test."""


async def wait_terminal(store: _TaskStore, task_id: str, timeout_s: float = _WAIT_TIMEOUT_S) -> None:
    """Wait for the run to finish and check that it completed."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        row = await store.get(task_id)
        if row is not None and row.status != "running":
            assert row.status == "completed", f"run ended as {row.status}: {row.error}"
            return
        await asyncio.sleep(_WAIT_TERMINAL_INTERVAL)
    raise TimeoutError(f"task {task_id} did not finish within {timeout_s}s")


async def wait_subtask_id(store: _TaskStore, parent_id: str, timeout_s: float = _WAIT_TIMEOUT_S) -> str:
    """Wait for the parent's first subtask record to appear, then give back its id."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        children = await store.list_subtasks(parent_id)
        if children:
            return children[0].task_id
        await asyncio.sleep(_WAIT_TERMINAL_INTERVAL)
    raise TimeoutError(f"task {parent_id} spawned no subtask within {timeout_s}s")


@pytest.fixture(autouse=True)
async def tool_infra(task_store: _TaskStore) -> AsyncIterator[_TaskStore]:
    """Give every tool test a throwaway record store."""
    yield task_store
