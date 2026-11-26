import asyncio
import base64
import inspect
import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from io import IOBase
from pathlib import Path
from typing import ClassVar, override
from unittest.mock import MagicMock

import aiofiles.threadpool
from agents import Agent as _SdkAgent
from agents import (
    MaxTurnsExceeded,
    ModelBehaviorError,
    OpenAIChatCompletionsModel,
    ToolOutputFileContent,
    ToolOutputImage,
    ToolOutputText,
    Usage,
)
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import TResponseInputItem, TResponseStreamEvent, ToolCallItem, ToolCallOutputItem
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.run_config import CallModelData, CallModelInputFilter, ModelInputData, RunConfig
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from agents.tool import Tool as _SdkTool
from agents.tool_context import ToolContext
from agents.usage import InputTokensDetails, OutputTokensDetails
import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsePromptParam,
    ResponseReasoningTextDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
    response_usage,
)
import pytest

import app.lib.agent.engine as engine
import app.lib.db.store as store_module
from app.lib.agent.engine import Agent, AgentRun, ValidationResult
from app.lib.shared import config
from app.lib.shared.paths import resources_dir
from app.lib.task import logcapture, task_manager
from app.lib.task.task_manager import Task, TaskCancelledError
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult, read_run_mcp_text
from tests.app.conftest import is_llm_available


def _text_delta(text: str, seq: int = 1) -> RawResponsesStreamEvent:
    data = ResponseTextDeltaEvent(
        delta=text, type="response.output_text.delta",
        content_index=0, item_id="i0", logprobs=[], output_index=0, sequence_number=seq,
    )
    return RawResponsesStreamEvent(data=data)


def _reasoning_delta(text: str, seq: int = 1) -> RawResponsesStreamEvent:
    data = ResponseReasoningTextDeltaEvent(
        delta=text, type="response.reasoning_text.delta",
        content_index=0, item_id="i0", output_index=0, sequence_number=seq,
    )
    return RawResponsesStreamEvent(data=data)


def _build_stub(gen: Callable[[], AsyncIterator], final_output: str | None) -> MagicMock:
    stub = MagicMock()
    stub.stream_events = MagicMock(side_effect=gen)
    stub.final_output = final_output
    stub.new_items = []
    stub.to_input_list = MagicMock(return_value=[{"role": "user", "content": "stub"}])
    stub.cancel = MagicMock()
    return stub


def _stub_result(events: list[RawResponsesStreamEvent], final_output: str | None) -> MagicMock:
    async def _gen() -> AsyncIterator[RawResponsesStreamEvent]:
        for event in events:
            yield event
    return _build_stub(_gen, final_output)


def _stall_stub(stall: float, final_output: str) -> MagicMock:
    async def _gen() -> AsyncIterator[RawResponsesStreamEvent]:
        await asyncio.sleep(stall)
        yield _text_delta("ok", 1)
    return _build_stub(_gen, final_output)


def _stall_in_drain() -> tuple[MagicMock, asyncio.Event]:
    """A stub whose drain stalls after the first event, and the event that fires at the stall."""
    entered = asyncio.Event()

    async def _gen() -> AsyncIterator[RawResponsesStreamEvent]:
        yield _text_delta("x", 1)
        entered.set()
        await asyncio.sleep(10)

    return _build_stub(_gen, "x"), entered


def _release_in_drain(
    final_output: str, error: Exception | None = None
) -> tuple[MagicMock, asyncio.Event, asyncio.Event]:
    """A stub whose drain stalls until released, and the events for entry and release."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _gen() -> AsyncIterator[RawResponsesStreamEvent]:
        yield _text_delta(final_output, 1)
        entered.set()
        await release.wait()
        if error is not None:
            raise error

    stub_output = None if error is not None else final_output
    return _build_stub(_gen, stub_output), entered, release


def _patch_runner(monkeypatch: pytest.MonkeyPatch, stub: MagicMock) -> None:
    monkeypatch.setattr(engine.Runner, "run_streamed", MagicMock(return_value=stub))


def _patch_runner_seq(monkeypatch: pytest.MonkeyPatch, stubs: list[MagicMock]) -> MagicMock:
    run_mock = MagicMock(side_effect=stubs)
    monkeypatch.setattr(engine.Runner, "run_streamed", run_mock)
    return run_mock


def _usage() -> Usage:
    # A fully populated Usage for the stub ToolContext.
    return Usage(
        input_tokens_details=InputTokensDetails(cache_write_tokens=0, cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def _patch_runner_factory(monkeypatch: pytest.MonkeyPatch, factory: Callable[[_SdkAgent, str | list], MagicMock]) -> None:
    monkeypatch.setattr(engine.Runner, "run_streamed", lambda a, i, **_kw: factory(a, i))


def _accept_any(text: str, _run: AgentRun) -> ValidationResult[str]:
    if text:
        return ValidationResult(is_valid=True, value=text)
    return ValidationResult(is_valid=False, error="empty output")


def _accept_ending_with_42(text: str, _run: AgentRun) -> ValidationResult[str]:
    if text.endswith("42"):
        return ValidationResult(is_valid=True, value=text)
    return ValidationResult(is_valid=False, error="give me 42")


async def _completed_run(monkeypatch: pytest.MonkeyPatch) -> AgentRun:
    """Drive a stubbed run to completion and return its handle."""
    monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
    agent = Agent(name="res", system_prompt="sp", validator=_accept_any)
    _patch_runner(monkeypatch, _stub_result([_text_delta("x", 1)], "x"))
    run = await agent.run("do it")
    await run.task.result()
    return run


class TestAgentLoop:
    async def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        events: list[RawResponsesStreamEvent],
        final_output: str | None,
    ) -> tuple[Agent, AgentRun]:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        _patch_runner(monkeypatch, _stub_result(events, final_output))
        run = await agent.run("do it")
        return agent, run

    def _run_path(self, task_id: uuid.UUID) -> Path:
        return Path(store_module.task_store.data_dir) / "runs" / str(task_id)

    async def test_valid_first_output_completed_and_persisted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _agent, run = await self._run(
            monkeypatch,
            [_reasoning_delta("thinking ", 1), _reasoning_delta("more", 2), _text_delta("Answer: 42", 3)],
            "Answer: 42",
        )
        assert await run.task.result() == "Answer: 42"
        assert run.task.status.value == "completed"
        assert (self._run_path(run.task.id) / "result").read_text() == "Answer: 42"

    async def test_conversation_writes_thinking_and_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _agent, run = await self._run(
            monkeypatch,
            [_reasoning_delta("A", 1), _reasoning_delta("B", 2), _text_delta("x", 3), _text_delta("y", 4)],
            "xy",
        )
        events = [e async for e in run.stream_conversation()]
        kinds = [(e.type, e.text) for e in events]
        assert ("user", "do it") in kinds
        assert ("thinking", "A") in kinds
        assert ("thinking", "B") in kinds
        assert ("content", "x") in kinds
        assert ("content", "y") in kinds
        assert ("assistant", "xy") in kinds
        # system prompt is first, user prompt second, assistant output is last
        assert kinds[0] == ("system", "sys")
        assert kinds[1] == ("user", "do it")
        assert kinds[-1] == ("assistant", "xy")

    async def test_failing_validator_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        stub1 = _stub_result([_text_delta("bad", 1)], "bad")
        stub2 = _stub_result([_text_delta("Answer: 42", 1)], "Answer: 42")

        agent = Agent(name="res", system_prompt="sys", validator=_accept_ending_with_42)
        run_mock = _patch_runner_seq(monkeypatch, [stub1, stub2])
        run = await agent.run("do it")
        assert await run.task.result() == "Answer: 42"
        events = [e async for e in run.stream_conversation()]
        corrective = [e for e in events if e.type == "user" and e.text.startswith("give me 42")]
        assert len(corrective) == 1
        assert "IMPORTANT NOTE" in corrective[0].text
        assert run_mock.call_count == 2

    async def test_validator_exhaustion_fails_with_last_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")

        def validator(_text: str, _run: AgentRun) -> ValidationResult[str]:
            return ValidationResult(is_valid=False, error="still wrong")

        agent = Agent(name="res", system_prompt="sys", validator=validator, max_attempts=3)
        _patch_runner(monkeypatch, _stub_result([_text_delta("no", 1)], "no"))
        run = await agent.run("do it")
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            await run.task.result()
        assert run.task.status.value == "failed"

    async def test_validator_receives_the_run_handle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        seen: list[AgentRun] = []

        def validator(text: str, agent_run: AgentRun) -> ValidationResult[str]:
            seen.append(agent_run)
            return ValidationResult(is_valid=True, value=text)

        agent = Agent(name="res", system_prompt="sys", validator=validator)
        _patch_runner(monkeypatch, _stub_result([_text_delta("ok", 1)], "ok"))
        run = await agent.run("do it")
        await run.task.result()
        assert seen
        assert all(isinstance(r, AgentRun) for r in seen)
        assert all(r.task is run.task for r in seen)

    async def test_async_validator_is_awaited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")

        async def validator(text: str, _run: AgentRun) -> ValidationResult[str]:
            return ValidationResult(is_valid=True, value=text)

        agent = Agent(name="res", system_prompt="sys", validator=validator)
        _patch_runner(monkeypatch, _stub_result([_text_delta("ok", 1)], "ok"))
        run = await agent.run("do it")
        assert await run.task.result() == "ok"

    async def test_run_always_creates_a_new_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        _patch_runner(monkeypatch, _stub_result([_text_delta("ok", 1)], "ok"))
        run1 = await agent.run("do it")
        run2 = await agent.run("do it")
        assert run1.task is not run2.task
        assert run1.task.id != run2.task.id
        assert await run1.task.result() == "ok"
        assert await run2.task.result() == "ok"
        assert len(await store_module.task_store.list()) == 2
        view1 = await store_module.get_task_view(str(run1.task.id))
        view2 = await store_module.get_task_view(str(run2.task.id))
        assert view1 is not None and view1.agent_name == "res"
        assert view2 is not None and view2.agent_name == "res"

    async def test_agent_is_pure_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        assert "start" not in dir(agent)
        assert "stream_conversation" not in dir(agent)
        assert "read_tool_calls" not in dir(agent)
        assert "record" not in dir(agent)
        assert await store_module.task_store.list() == []
        # the store's migration creates the runs root, but no run dir may exist yet
        assert list((Path(store_module.task_store.data_dir) / "runs").iterdir()) == []
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        _patch_runner(monkeypatch, _stub_result([_text_delta("ok", 1)], "ok"))
        run = await agent.run("do it")
        await run.task.result()
        assert [r.task_id for r in await store_module.task_store.list()] == [str(run.task.id)]

    async def test_non_str_value_serialized_to_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        value = {"a": 1, "b": [2, 3]}

        def validator(_text: str, _run: AgentRun) -> ValidationResult[dict]:
            return ValidationResult(is_valid=True, value=value)

        agent = Agent(name="res", system_prompt="sys", validator=validator)
        _patch_runner(monkeypatch, _stub_result([_text_delta("json", 1)], "json"))
        run = await agent.run("do it")
        assert await run.task.result() == json.dumps(value, ensure_ascii=False)
        run_path = self._run_path(run.task.id)
        assert (run_path / "result").read_text() == json.dumps(value, ensure_ascii=False)
        assert not (run_path / "result.json").exists()

    async def test_agent_name_written_and_view_identifies_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _agent, run = await self._run(monkeypatch, [_text_delta("ok", 1)], "ok")
        await run.task.result()
        view = await store_module.get_task_view(str(run.task.id))
        assert view is not None and view.agent_name == "res"
        assert view.tool_name is None
        assert not (self._run_path(run.task.id) / "meta").exists()

    async def test_conversation_missing_file_yields_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _agent, run = await self._run(monkeypatch, [_text_delta("ok", 1)], "ok")
        await run.task.result()
        conv = self._run_path(run.task.id) / "conversation.jsonl"
        conv.unlink()
        events = [e async for e in run.stream_conversation()]
        assert events == []

    async def test_sequential_runs_of_one_agent_are_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        _patch_runner_seq(monkeypatch, [_stub_result([_text_delta("one", 1)], "one"),
                                        _stub_result([_text_delta("two", 1)], "two")])
        run1 = await agent.run("first prompt")
        await run1.task.result()
        run2 = await agent.run("second prompt")
        await run2.task.result()
        assert run1.task.id != run2.task.id
        events1 = [e async for e in run1.stream_conversation()]
        events2 = [e async for e in run2.stream_conversation()]
        assert events1[0].type == "system" and events1[1].type == "user" and events1[1].text == "first prompt"
        assert events2[0].type == "system" and events2[1].type == "user" and events2[1].text == "second prompt"
        assert all(e.text != "second prompt" for e in events1)
        assert all(e.text != "first prompt" for e in events2)
        view1 = await store_module.get_task_view(str(run1.task.id))
        view2 = await store_module.get_task_view(str(run2.task.id))
        assert view1 is not None and view1.agent_name == "res" and view1.tool_name is None
        assert view2 is not None and view2.agent_name == "res" and view2.tool_name is None

    async def test_concurrent_runs_of_one_agent_are_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)

        def factory(_sdk_agent: _SdkAgent, input_text: str | list) -> MagicMock:
            text = input_text if isinstance(input_text, str) else str(input_text)
            return _stub_result([_text_delta(text, 1)], text)

        _patch_runner_factory(monkeypatch, factory)
        run1, run2 = await asyncio.gather(agent.run("prompt A"), agent.run("prompt B"))
        assert run1.task.id != run2.task.id
        assert await run1.task.result() == "prompt A"
        assert await run2.task.result() == "prompt B"
        assert run1.task.status.value == "completed"
        assert run2.task.status.value == "completed"
        events1 = [e async for e in run1.stream_conversation()]
        events2 = [e async for e in run2.stream_conversation()]
        assert events1[0].type == "system" and events1[1].type == "user" and events1[1].text == "prompt A"
        assert events2[0].type == "system" and events2[1].type == "user" and events2[1].text == "prompt B"
        assert all(e.text != "prompt B" for e in events1)
        assert all(e.text != "prompt A" for e in events2)

    async def test_agentrun_task_is_the_management_interface(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _agent, run = await self._run(monkeypatch, [_text_delta("ok", 1)], "ok")
        assert isinstance(run, AgentRun)
        assert isinstance(run.task, Task)
        assert "status" not in dir(run)
        assert "result" not in dir(run)
        assert "cancel" not in dir(run)
        assert "record" not in dir(run)
        assert await run.task.result() == "ok"
        assert run.task.status.value == "completed"


class _ScriptedModel(OpenAIChatCompletionsModel):
    """A model that answers from a fixed script and records every input it receives."""

    script: ClassVar[list[str | tuple[str, str] | Exception]] = []
    received: ClassVar[list[list[dict]]] = []
    order: ClassVar[list[str]] = []

    @override
    async def stream_response(
        self,
        _system_instructions: str | None,
        input: str | list[TResponseInputItem],
        _model_settings: ModelSettings,
        _tools: list[_SdkTool],
        _output_schema: AgentOutputSchemaBase | None,
        _handoffs: list[Handoff],
        _tracing: ModelTracing,
        **_kwargs: str | ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        _ScriptedModel.order.append("model")
        _ScriptedModel.received.append(list(input) if isinstance(input, list) else [input])
        step = _ScriptedModel.script.pop(0)
        if isinstance(step, Exception):
            raise step
        item = _script_item(step)
        yield ResponseOutputItemDoneEvent(
            type="response.output_item.done", item=item, item_id=item.id or "fc_1",
            output_index=0, sequence_number=1,
        )
        yield ResponseCompletedEvent(
            type="response.completed", response=_completed_response(item), sequence_number=2,
        )


def _message(text: str) -> ResponseOutputMessage:
    """One assistant message holding the text."""
    return ResponseOutputMessage(
        id="msg_1", role="assistant", status="completed", type="message",
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
        tool_calls=[],
    )


def _script_item(step: str | tuple[str, str]) -> ResponseOutputMessage | ResponseFunctionToolCall:
    """Build one output item from a script step."""
    if isinstance(step, tuple):
        tool_name, tool_args = step
        return ResponseFunctionToolCall(
            id="fc_1", call_id="call_1", name=tool_name,
            arguments=tool_args, type="function_call", status="completed",
        )
    return _message(step)


def _completed_response(item: ResponseOutputMessage | ResponseFunctionToolCall) -> Response:
    """A terminal response wrapping one output item."""
    usage = ResponseUsage(
        input_tokens=1,
        input_tokens_details=response_usage.InputTokensDetails(cache_write_tokens=0, cached_tokens=0),
        output_tokens=1,
        output_tokens_details=response_usage.OutputTokensDetails(reasoning_tokens=0),
        total_tokens=2,
    )
    return Response(
        id="resp_1", created_at=1, model="default", object="response", output=[item],
        parallel_tool_calls=True, tool_choice="auto", tools=[], status="completed", usage=usage,
    )


def _rewrite_last_user(data: CallModelData) -> ModelInputData:
    """Rewrite the last user message's text with a marker."""
    _ScriptedModel.order.append("filter")
    items = list(data.model_data.input)
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if isinstance(item, dict) and item.get("role") == "user":
            items[index] = {**item, "content": "REWROTE:" + item["content"]}
            break
    return ModelInputData(input=items, instructions=data.model_data.instructions)


async def _rewrite_last_user_async(data: CallModelData) -> ModelInputData:
    return _rewrite_last_user(data)


class TestModelInputFilter:
    """The model_input_filter option shapes what the run sends to the model."""

    async def test_no_filter_keeps_the_call_and_passes_no_run_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        run_mock = MagicMock(return_value=_stub_result([_text_delta("ok", 1)], "ok"))
        monkeypatch.setattr(engine.Runner, "run_streamed", run_mock)
        run = await agent.run("do it")
        assert await run.task.result() == "ok"
        args, kwargs = run_mock.call_args
        assert isinstance(args[0], _SdkAgent)
        assert args[1] == "do it"
        assert kwargs == {"max_turns": None}

    @pytest.mark.parametrize("model_input_filter", [_rewrite_last_user, _rewrite_last_user_async],
                             ids=["sync_filter", "async_filter"])
    async def test_filter_runs_before_every_request_and_the_rewrite_reaches_the_wire(
        self, monkeypatch: pytest.MonkeyPatch, model_input_filter: CallModelInputFilter
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        _ScriptedModel.script = ["bad", "Answer: 42"]
        _ScriptedModel.received = []
        _ScriptedModel.order = []

        agent = Agent(name="res", system_prompt="sys", validator=_accept_ending_with_42,
                      model_input_filter=model_input_filter)
        monkeypatch.setattr(engine, "OpenAIChatCompletionsModel", _ScriptedModel)
        run = await agent.run("do it")
        assert await run.task.result() == "Answer: 42"
        assert _ScriptedModel.order == ["filter", "model", "filter", "model"]
        assert _ScriptedModel.received[0] == [{"role": "user", "content": "REWROTE:do it"}]
        retry = _ScriptedModel.received[1]
        assert retry[0] == {"role": "user", "content": "do it"}
        assert retry[1]["role"] == "assistant"
        assert retry[1]["content"][0]["text"] == "bad"
        assert retry[2]["role"] == "user"
        assert retry[2]["content"].startswith("REWROTE:give me 42")
        assert "IMPORTANT NOTE" in retry[2]["content"]

    async def test_run_config_sets_only_the_filter_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any,
                      model_input_filter=_rewrite_last_user)
        run_mock = MagicMock(return_value=_stub_result([_text_delta("ok", 1)], "ok"))
        monkeypatch.setattr(engine.Runner, "run_streamed", run_mock)
        run = await agent.run("do it")
        await run.task.result()
        run_config = run_mock.call_args.kwargs["run_config"]
        assert run_config.call_model_input_filter is _rewrite_last_user
        view = asdict(run_config)
        defaults = asdict(RunConfig())
        # model_provider is a fresh per-instance object, so it compares by identity
        view.pop("model_provider")
        defaults.pop("model_provider")
        view.pop("call_model_input_filter")
        defaults.pop("call_model_input_filter")
        assert view == defaults


class TestRunStopChannel:
    """cancel_run_loop stops a run through the engine's run control."""

    async def test_cancel_in_the_drain_still_cancels_the_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        stub, entered = _stall_in_drain()
        _patch_runner(monkeypatch, stub)
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        run = await agent.run("do it")
        await asyncio.wait_for(entered.wait(), timeout=5)
        await task_manager.cancel(str(run.task.id))
        assert run.task.status.value == "cancelled"
        stub.cancel.assert_called_once()

    async def test_success_path_orders_final_output_then_input_list_then_cancel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        order: list[str] = []
        history = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": "bad"}]

        def _ordered_stub(final_output: str, input_list: list) -> MagicMock:
            async def _gen() -> AsyncIterator[RawResponsesStreamEvent]:
                yield _text_delta(final_output, 1)

            def _final(_self: MagicMock) -> str:
                order.append("final_output")
                return final_output

            stub = MagicMock()
            stub.stream_events = MagicMock(side_effect=_gen)
            type(stub).final_output = property(_final)
            stub.to_input_list = MagicMock(
                side_effect=lambda: (order.append("to_input_list"), input_list)[1]
            )
            stub.cancel = MagicMock(side_effect=lambda: order.append("cancel"))
            return stub

        agent = Agent(name="res", system_prompt="sys", validator=_accept_ending_with_42)
        run_mock = _patch_runner_seq(
            monkeypatch, [_ordered_stub("bad", history), _ordered_stub("Answer: 42", [])]
        )
        run = await agent.run("do it")
        assert await run.task.result() == "Answer: 42"
        assert order == [
            "final_output", "to_input_list", "cancel",
            "final_output", "to_input_list", "cancel",
        ]
        replayed = run_mock.call_args_list[1].args[1]
        assert replayed[:2] == history
        assert replayed[2]["role"] == "user"
        assert replayed[2]["content"].startswith("give me 42")
        assert "IMPORTANT NOTE" in replayed[2]["content"]

    async def test_cancel_run_loop_cancels_the_live_result_and_marks_the_stop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        stub, entered, release = _release_in_drain("bad")
        _patch_runner(monkeypatch, stub)

        def validator(_text: str, _run: AgentRun) -> ValidationResult[str]:
            return ValidationResult(is_valid=False, error="give me 42")

        agent = Agent(name="res", system_prompt="sys", validator=validator, max_attempts=3)
        run = await agent.run("do it")
        await asyncio.wait_for(entered.wait(), timeout=5)
        run.cancel_run_loop()
        run.cancel_run_loop()
        assert stub.cancel.call_count == 2
        release.set()
        with pytest.raises(RuntimeError, match="stopped on request"):
            await run.task.result()
        assert run.task.status.value == "failed"

    async def test_attempt_writes_and_clears_the_live_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        stub, entered, release = _release_in_drain("ok")
        _patch_runner(monkeypatch, stub)

        def validator(_text: str, run: AgentRun) -> ValidationResult[str]:
            run.cancel_run_loop()
            return ValidationResult(is_valid=True, value=_text)

        agent = Agent(name="res", system_prompt="sys", validator=validator)
        run = await agent.run("do it")
        await asyncio.wait_for(entered.wait(), timeout=5)
        run.cancel_run_loop()
        assert stub.cancel.call_count == 1
        release.set()
        assert await run.task.result() == "ok"
        assert stub.cancel.call_count == 2

    async def test_stop_starts_no_further_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        _ScriptedModel.script = ["bad"]
        _ScriptedModel.received = []

        def validator(_text: str, run: AgentRun) -> ValidationResult[str]:
            run.cancel_run_loop()
            return ValidationResult(is_valid=False, error="give me 42")

        agent = Agent(name="res", system_prompt="sys", validator=validator)
        monkeypatch.setattr(engine, "OpenAIChatCompletionsModel", _ScriptedModel)
        run = await agent.run("do it")
        with pytest.raises(RuntimeError, match="stopped on request"):
            await run.task.result()
        assert run.task.status.value == "failed"
        assert _ScriptedModel.received == [[{"role": "user", "content": "do it"}]]
        assert [e.text for e in await run.read_events() if e.type == "user"] == ["do it"]

    async def test_cancel_run_loop_without_a_control_is_a_logged_noop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run = await _completed_run(monkeypatch)
        run.cancel_run_loop()
        run.cancel_run_loop()
        assert "no run control" in capsys.readouterr().err
        assert run.task.status.value == "completed"

    @pytest.mark.parametrize("mode", ["success", "failure", "cancellation"])
    async def test_cancel_run_loop_no_ops_once_the_run_ends(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        mode: str,
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")

        def validator(_text: str, _run: AgentRun) -> ValidationResult[str]:
            return ValidationResult(is_valid=False, error="nope")

        if mode == "cancellation":
            stub, entered = _stall_in_drain()
            agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
            _patch_runner(monkeypatch, stub)
            run = await agent.run("do it")
            await asyncio.wait_for(entered.wait(), timeout=5)
            await task_manager.cancel(str(run.task.id))
        elif mode == "success":
            agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
            _patch_runner(monkeypatch, _stub_result([_text_delta("x", 1)], "x"))
            run = await agent.run("do it")
            await run.task.result()
        else:
            agent = Agent(name="res", system_prompt="sys", validator=validator, max_attempts=1)
            _patch_runner(monkeypatch, _stub_result([_text_delta("x", 1)], "x"))
            run = await agent.run("do it")
            with pytest.raises(RuntimeError, match="after 1 attempts"):
                await run.task.result()
        run.cancel_run_loop()
        assert "no run control" in capsys.readouterr().err

    async def test_never_stopped_run_keeps_the_same_wire_and_replay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        _ScriptedModel.script = ["bad", "Answer: 42"]
        _ScriptedModel.received = []

        agent = Agent(name="res", system_prompt="sys", validator=_accept_ending_with_42)
        monkeypatch.setattr(engine, "OpenAIChatCompletionsModel", _ScriptedModel)
        run = await agent.run("do it")
        assert await run.task.result() == "Answer: 42"
        assert _ScriptedModel.received == [
            [{"role": "user", "content": "do it"}],
            [
                {"role": "user", "content": "do it"},
                {
                    "id": "msg_1",
                    "content": [{"annotations": [], "text": "bad", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                    "tool_calls": [],
                },
                {
                    "role": "user",
                    "content": (
                        "give me 42"
                        "\n\nIMPORTANT NOTE: Your last response is dropped. "
                        "Please correct the error and re-generate your entire response in your "
                        "next message. Please don't put any filler content. Your message must "
                        "contain ONLY the final output."
                    ),
                },
            ],
        ]


class TestRunTimeout:
    """The run deadline comes from the timeout kwarg and defaults to no deadline."""

    async def test_default_timeout_is_none(self) -> None:
        timeout_default = inspect.signature(Agent).parameters["timeout"].default
        assert timeout_default is None

    async def test_default_run_finishes_past_a_short_deadline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        _patch_runner(monkeypatch, _stall_stub(0.2, "ok"))
        run = await agent.run("do it")
        assert await run.task.result() == "ok"
        assert run.task.status.value == "completed"

    async def test_explicit_timeout_stops_a_slow_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any, timeout=0.05)
        _patch_runner(monkeypatch, _stall_stub(0.2, "ok"))
        run = await agent.run("do it")
        with pytest.raises(TimeoutError):
            await run.task.result()
        assert run.task.status.value == "failed"


class _BlockTool(Tool):
    name: ClassVar[str] = "search_web"
    description: ClassVar[str] = "Search the web."
    is_tracked: ClassVar[bool] = True
    params: ClassVar[tuple[Param, ...]] = (Param(name="query", type=str, description="The query."),)

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text=f"results for {params['query']}")


class _IdOnlyTool(_BlockTool):
    name: ClassVar[str] = "id_only"
    description: ClassVar[str] = "Opt in to id-only results."


class _AsyncTool(Tool):
    name: ClassVar[str] = "gen_media"
    is_slow: ClassVar[bool] = True
    description: ClassVar[str] = "Generate media."
    is_tracked: ClassVar[bool] = True
    params: ClassVar[tuple[Param, ...]] = (Param(name="prompt", type=str, description="The prompt."),)

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text=f"generated {params['prompt']}")


class _BigTool(Tool):
    name: ClassVar[str] = "big"
    description: ClassVar[str] = "Return a long string."
    is_tracked: ClassVar[bool] = True
    is_result_long: ClassVar[bool] = True
    params: ClassVar[tuple[Param, ...]] = (
        Param(name="query", type=str, description="The query."),
    )

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="x" * 1000)


class _BigIdOnlyTool(_BigTool):
    name: ClassVar[str] = "big_id_only"


class _FailTool(Tool):
    name: ClassVar[str] = "boom"
    description: ClassVar[str] = "Always fail."
    is_tracked: ClassVar[bool] = True
    params: ClassVar[tuple[Param, ...]] = (Param(name="query", type=str, description="The query."),)

    @override
    async def call(self, _params: dict) -> ToolResult:
        raise RuntimeError("bad")


class _StopTool(Tool):
    name: ClassVar[str] = "stopper"
    description: ClassVar[str] = "Stop while running."
    is_tracked: ClassVar[bool] = True
    params: ClassVar[tuple[Param, ...]] = (Param(name="query", type=str, description="The query."),)

    @override
    async def call(self, _params: dict) -> ToolResult:
        raise asyncio.CancelledError


class _BoundedTool(Tool):
    name: ClassVar[str] = "bounded"
    description: ClassVar[str] = "Take a small number."
    is_tracked: ClassVar[bool] = True
    params: ClassVar[tuple[Param, ...]] = (
        Param(name="count", type=int, description="How many.", min=1, max=10),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text=f"count {params['count']}")


class _ToolTierRejectTool(Tool):
    name: ClassVar[str] = "tool_rejector"
    description: ClassVar[str] = "Reject a large number before it runs."
    is_tracked: ClassVar[bool] = True
    params: ClassVar[tuple[Param, ...]] = (
        Param(name="count", type=int, description="How many."),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        if params["count"] >= 50:
            raise ToolMessageError("Error: boom.")
        return ToolResult(text=f"count {params['count']}")


class _CachedTool(Tool):
    name: ClassVar[str] = "cached_search"
    description: ClassVar[str] = "Search and reuse a stored answer."
    is_tracked: ClassVar[bool] = True
    staleness: ClassVar[timedelta | None] = timedelta(hours=24)
    params: ClassVar[tuple[Param, ...]] = (Param(name="query", type=str, description="The query."),)

    @override
    async def build_cache_key(self, params: dict) -> str | None:
        return str(params["query"])

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text=f"results for {params['query']}")


class _MediaTool(Tool):
    name: ClassVar[str] = "media"
    description: ClassVar[str] = "Return text plus the files the test names."
    is_tracked: ClassVar[bool] = True
    files: ClassVar[tuple[str, ...] | None] = None
    params: ClassVar[tuple[Param, ...]] = (Param(name="query", type=str, description="The query."),)

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text=f"media {params['query']}", files=self.files)


class _IdOnlyMediaTool(_MediaTool):
    name: ClassVar[str] = "id_only_media"
    description: ClassVar[str] = "Return text plus files, and opt in to id-only results."
    files: ClassVar[tuple[str, ...] | None] = ("img.png",)


def _block_media_read(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    original_open = aiofiles.threadpool.sync_open

    def _open(path: str, *args: str, **kwargs: str | int) -> IOBase:
        if str(path).endswith("/img.png"):
            raise error
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(aiofiles.threadpool, "sync_open", _open)


def _make_tool_stub(
    starting_agent: _SdkAgent,
    _input: str | list,
    *,
    tool_name: str,
    args_json: str,
    capture: dict[str, str] | None = None,
    with_text: bool = True,
    call_id: str = "C1",
    count: int = 1,
) -> MagicMock:
    """Build a stub that runs in-loop calls of the tool.

    Args:
        starting_agent: The SDK agent holding the bridged tool.
        _input: What the runner was given, ignored.
        tool_name: The tool to call.
        args_json: The call arguments as a JSON string.
        capture: A dict the last tool output is stored in, under the key out.
        with_text: Whether the stub also streams a line of text.
        call_id: The call id the first tool call item announces. Results are named C1, C2, and so on.
        count: How many calls to make.

    Returns:
        A stub that stands in for a streamed runner result.
    """
    async def _gen() -> AsyncIterator[RawResponsesStreamEvent | RunItemStreamEvent]:
        ft = next(t for t in starting_agent.tools if t.name == tool_name)
        for index in range(count):
            result_id = f"C{index + 1}"
            tc = ToolContext(context=None, usage=_usage(), tool_name=tool_name,
                             tool_call_id=result_id, tool_arguments=args_json)
            announced = call_id if index == 0 else result_id
            yield RunItemStreamEvent(
                item=ToolCallItem(agent=starting_agent,
                                  raw_item={"name": tool_name, "call_id": announced, "arguments": args_json}),
                name="tool_called",
            )
            out = await ft.on_invoke_tool(tc, args_json)
            if capture is not None:
                capture["out"] = out
            yield RunItemStreamEvent(
                item=ToolCallOutputItem(agent=starting_agent, raw_item={"call_id": result_id}, output=out),
                name="tool_output",
            )
        if with_text:
            yield _text_delta("done", 9)

    return _build_stub(_gen, "done")


class TestBuildFunctionTool:
    async def test_in_loop_bridge_returns_real_text_and_records_tool_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        agent.add_tool_adapter(_BlockTool())
        out_holder: dict[str, str] = {}
        _patch_runner_factory(monkeypatch, partial(_make_tool_stub, tool_name="search_web",
                                                   args_json='{"query":"cats"}', capture=out_holder))
        run = await agent.run("do it")
        assert await run.task.result() == "done"
        assert out_holder["out"] == "results for cats"
        # the subtask's tool row carries the same task id the bridge recorded
        calls = await run.read_tool_calls()
        assert calls[0].task_id
        record = await store_module.tool_store.get(calls[0].task_id)
        assert record.task_id == calls[0].task_id
        assert record.tool_name == "search_web"
        assert record.params == {"query": "cats"}

    async def test_in_loop_call_of_a_slow_tool_returns_real_text_not_the_ack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _AsyncTool.is_slow is True
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        agent.add_tool_adapter(_AsyncTool())
        out_holder: dict[str, str] = {}
        _patch_runner_factory(monkeypatch, partial(_make_tool_stub, tool_name="gen_media",
                                                   args_json='{"prompt":"a cat"}', capture=out_holder))
        run = await agent.run("do it")
        assert await run.task.result() == "done"
        assert out_holder["out"] == "generated a cat"
        assert "Task started" not in out_holder["out"]

    async def test_in_loop_tool_creates_subtask_and_tool_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        tool_name = "search_web"
        args_json = '{"query":"cats"}'
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        agent.add_tool_adapter(_BlockTool())
        _patch_runner_factory(monkeypatch, partial(_make_tool_stub, tool_name=tool_name, args_json=args_json))
        run = await agent.run("do it")
        assert await run.task.result() == "done"

        calls = await run.read_tool_calls()
        assert len(calls) == 1
        call = calls[0]
        assert call.tool_name == tool_name
        assert call.params == {"query": "cats"}
        assert call.status == "completed"
        assert call.task_id and call.run_dir

        events = [e async for e in run.stream_conversation()]
        tool_call_events = [e for e in events if e.type == "tool_call"]
        tool_result_events = [e for e in events if e.type == "tool_result"]
        assert len(tool_call_events) == 1 and tool_call_events[0].params == {"query": "cats"}
        assert len(tool_result_events) == 1
        assert tool_result_events[0].task_id == call.task_id

    async def test_truncated_in_loop_result_keeps_the_tool_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        agent.add_tool_adapter(_BigTool())
        out_holder: dict[str, str] = {}
        _patch_runner_factory(monkeypatch, partial(_make_tool_stub, tool_name="big",
                                                    args_json='{"max_char":600,"query":"q"}', capture=out_holder, with_text=False))
        run = await agent.run("do it")
        await run.task.result()
        task_id = (await run.read_tool_calls())[0].task_id
        assert out_holder["out"] == (
            "x" * 600 + "\n[Truncated: showing the first 600 of 1000 chars. "
            f"task_id={task_id} \u2014 fetch the remaining 400 chars with "
            f"task(action='result', task_id='{task_id}', offset=600, max_char=400).]"
        )


class TestToolOutputMedia:
    """A tool result that declares files hands image and video parts to the backend."""

    PNG: bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc`\x00\x02\x00\x00\x05\x00\x01z^\xab?\x00\x00\x00\x00IEND\xaeB`\x82"
    MP4: bytes = base64.b64decode(
        "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAs1tZGF0AAACrQYF//+p3EXpvebZSLeWLNgg2SPu"
        "73gyNjQgLSBjb3JlIDE2NCByMzEwOCAzMWUxOWY5IC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMt"
        "MjAyMyAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9j"
        "az0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3Jl"
        "Zj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0"
        "X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTEgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFk"
        "cz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZy"
        "YW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWln"
        "aHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTEgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD00"
        "MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlv"
        "PTEuNDAgYXE9MToxLjAwAIAAAAAQZYiEABX//vfJ78Cm69vfgQAAAxVtb292AAAAbG12aGQAAAAAAAAAAAAAAAAAAAPoAAAD"
        "6AABAAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAACAAACP3RyYWsAAABcdGtoZAAAAAMAAAAAAAAAAAAAAAEAAAAAAAAD6AAAAAAAAAAAAAAAAAAAAAAAAQAAAAAA"
        "AAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAEAAAABAAAAAAACRlZHRzAAAAHGVsc3QAAAAAAAAAAQAAA+gAAAAAAAEA"
        "AAAAAbdtZGlhAAAAIG1kaGQAAAAAAAAAAAAAAAAAAEAAAABAAFXEAAAAAAAtaGRscgAAAAAAAAAAdmlkZQAAAAAAAAAAAAAA"
        "AFZpZGVvSGFuZGxlcgAAAAFibWluZgAAABR2bWhkAAAAAQAAAAAAAAAAAAAAJGRpbmYAAAAcZHJlZgAAAAAAAAABAAAADHVy"
        "bCAAAAABAAABInN0YmwAAAC+c3RzZAAAAAAAAAABAAAArmF2YzEAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAEAAQAEgAAABI"
        "AAAAAAAAAAEVTGF2YzYwLjMxLjEwMiBsaWJ4MjY0AAAAAAAAAAAAAAAY//8AAAA0YXZjQwFkAAr/4QAXZ2QACqzZXsBEAAAD"
        "AAQAAAMACDxIllgBAAZo6+PLIsD9+PgAAAAAEHBhc3AAAAABAAAAAQAAABRidHJ0AAAAAAAAFigAABYoAAAAGHN0dHMAAAAA"
        "AAAAAQAAAAEAAEAAAAAAHHN0c2MAAAAAAAAAAQAAAAEAAAABAAAAAQAAABRzdHN6AAAAAAAAAsUAAAABAAAAFHN0Y28AAAAA"
        "AAAAAQAAADAAAABidWR0YQAAAFptZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAA"
        "Jal0b28AAAAdZGF0YQAAAAEAAAAATGF2ZjYwLjE2LjEwMA=="
    )
    MP3: bytes = b"\xff\xfb\x90\x00"
    PDF: bytes = b"%PDF-1.4 junk"

    async def _drive(
        self,
        monkeypatch: pytest.MonkeyPatch,
        files: tuple[str, ...] | None,
        *,
        by_id: bool = False,
        query: str = "cats",
    ) -> tuple[AgentRun, str | list[ToolOutputText | ToolOutputImage | ToolOutputFileContent]]:
        """Run a media tool and return its run and model output."""
        res = Path(resources_dir())
        (res / "img.png").write_bytes(self.PNG)
        (res / "vid.mp4").write_bytes(self.MP4)
        (res / "aud.mp3").write_bytes(self.MP3)
        (res / "doc.pdf").write_bytes(self.PDF)
        monkeypatch.setattr(_MediaTool, "files", files)
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys",
                      validator=_accept_any, tool_results_by_id=by_id)
        agent.add_tool_adapter(_MediaTool())
        out_holder: dict[str, str | list[ToolOutputText | ToolOutputImage | ToolOutputFileContent]] = {}
        _patch_runner_factory(monkeypatch, partial(_make_tool_stub, tool_name="media",
                                                   args_json=json.dumps({"query": query}), capture=out_holder))
        run = await agent.run("do it")
        await run.task.result()
        return run, out_holder["out"]

    async def test_declared_media_files_become_content_parts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, out = await self._drive(monkeypatch, ("img.png", "vid.mp4", "aud.mp3", "doc.pdf"))
        assert isinstance(out, list)
        assert [type(part) for part in out] == [ToolOutputText, ToolOutputImage, ToolOutputFileContent]
        assert isinstance(out[0], ToolOutputText) and out[0].text == "media cats"
        image = out[1]
        assert isinstance(image, ToolOutputImage)
        assert image.image_url.startswith("data:image/png;base64,")
        assert base64.b64decode(image.image_url.split(",", 1)[1]) == self.PNG
        video = out[2]
        assert isinstance(video, ToolOutputFileContent)
        assert video.file_data.startswith("data:video/mp4;base64,")
        assert base64.b64decode(video.file_data.split(",", 1)[1]) == self.MP4
        assert video.filename == "vid.mp4"
        # the audio and pdf files make no part; only their paths ride in the text
        assert len(out) == 3

    async def test_empty_files_declaring_still_hands_back_a_structured_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, out = await self._drive(monkeypatch, ())
        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], ToolOutputText) and out[0].text == "media cats"

    async def test_no_files_still_hands_back_plain_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, out = await self._drive(monkeypatch, None)
        assert out == "media cats"

    @pytest.mark.parametrize("path", ["missing.png", "../outside.png"])
    async def test_unavailable_media_has_an_explicit_delivery_failure(
        self, monkeypatch: pytest.MonkeyPatch, path: str,
    ) -> None:
        _, out = await self._drive(monkeypatch, (path,))
        assert isinstance(out, list)
        assert len(out) == 2
        assert isinstance(out[0], ToolOutputText) and out[0].text == "media cats"
        assert isinstance(out[1], ToolOutputText)
        assert out[1].text == f"Attachment delivery failed for {path!r}. Media could not be read or decoded."

    async def test_absolute_image_path_becomes_a_part(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        png = tmp_path / "outside.png"
        png.write_bytes(self.PNG)
        _, out = await self._drive(monkeypatch, (str(png), str(tmp_path / "missing.png")))
        assert isinstance(out, list)
        assert [type(part) for part in out] == [ToolOutputText, ToolOutputImage, ToolOutputText]
        image = out[1]
        assert isinstance(image, ToolOutputImage)
        assert image.image_url.startswith("data:image/png;base64,")
        assert base64.b64decode(image.image_url.split(",", 1)[1]) == self.PNG
        assert out[2].text.startswith("Attachment delivery failed for ")

    async def test_id_notice_takes_the_structured_path_over_the_files(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, out = await self._drive(monkeypatch, ("img.png",), by_id=True)
        assert isinstance(out, str)
        assert out.startswith("Task completed.")
        assert "media cats" not in out

    @pytest.mark.parametrize(("path", "data"), [
        ("broken.png", b"not an image"),
        ("truncated.png", PNG[:40]),
        ("broken.mp4", b"\x00\x00\x00\x18ftypmp42"),
    ])
    async def test_invalid_media_is_reported_without_losing_valid_media(
        self, monkeypatch: pytest.MonkeyPatch, path: str, data: bytes,
    ) -> None:
        (Path(resources_dir()) / path).write_bytes(data)
        _, out = await self._drive(monkeypatch, (path, "img.png"))
        assert isinstance(out, list)
        assert [type(part) for part in out] == [ToolOutputText, ToolOutputText, ToolOutputImage]
        assert out[0].text == "media cats"
        assert out[1].text == f"Attachment delivery failed for {path!r}. Media could not be read or decoded."
        assert base64.b64decode(out[2].image_url.split(",", 1)[1]) == self.PNG

    @pytest.mark.parametrize(("error", "reason"), [
        (PermissionError("private permission detail"), "Media could not be read or decoded."),
        (RuntimeError("private unexpected detail"), "Media could not be delivered."),
    ])
    async def test_read_failures_preserve_execution_and_original_result(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception, reason: str,
    ) -> None:
        _block_media_read(monkeypatch, error)
        query = "action complete\nTab: current\nstatus: success\nCheckpoint: cp-saved"
        run, out = await self._drive(monkeypatch, ("img.png", "vid.mp4"), query=query)
        assert isinstance(out, list)
        assert [type(part) for part in out] == [ToolOutputText, ToolOutputText, ToolOutputFileContent]
        assert out[0].text == f"media {query}"
        assert out[1].text == f"Attachment delivery failed for 'img.png'. {reason}"
        assert base64.b64decode(out[2].file_data.split(",", 1)[1]) == self.MP4
        calls = await run.read_tool_calls()
        assert len(calls) == 1
        assert calls[0].status == "completed"
        source = await read_run_mcp_text(os.path.join(store_module.task_store.data_dir, calls[0].run_dir))
        assert source == f"media {query}"

    async def test_cancelled_attachment_read_stops_the_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _block_media_read(monkeypatch, asyncio.CancelledError())
        with pytest.raises(TaskCancelledError):
            await self._drive(monkeypatch, ("img.png",))

    async def test_missing_nonmedia_stays_a_text_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, out = await self._drive(monkeypatch, ("missing.pdf",), query="missing.pdf")
        assert isinstance(out, list)
        assert len(out) == 1
        assert out[0].text == "media missing.pdf"

    async def test_id_notice_does_not_attempt_attachment_delivery(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _block_media_read(monkeypatch, AssertionError("attachment was read"))
        run, out = await self._drive(monkeypatch, ("img.png",), by_id=True)
        calls = await run.read_tool_calls()
        assert out == (
            f"Task completed. task_id={calls[0].task_id}. Retrieve the result with the task tool: "
            f"task(action='result', task_id='{calls[0].task_id}')."
        )


class TestWireReasoningReplay:
    """Verify streamed reasoning text is replayed across multi-turn agent runs."""

    async def test_wire_run_replays_streamed_reasoning_on_default_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        _CountedTool.calls = 0
        clients: list[httpx.AsyncClient] = []
        tool_call = {
            "index": 0,
            "id": "call_r1",
            "type": "function",
            "function": {"name": "counted_note", "arguments": '{"text":"v"}'},
        }
        turn_one_sse = (
            _sse_chunk({"role": "assistant", "reasoning": "deliberate thought"}, None)
            + _sse_chunk({"tool_calls": [tool_call]}, None)
            + _sse_chunk({}, "tool_calls")
            + b"data: [DONE]\n\n"
        )
        responses = [
            httpx.Response(200, headers={"content-type": "text/event-stream"}, content=turn_one_sse),
            _sse_text_response("final answer"),
        ]
        _spy, wire_messages = _wire_mock_openai(monkeypatch, responses, clients)
        try:
            agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
            agent.add_tool_adapter(_CountedTool())
            run = await agent.run("do it")
            assert await run.task.result() == "final answer"
        finally:
            await _close_clients(clients)
        assert wire_messages[1][2]["role"] == "assistant"
        assert wire_messages[1][2]["reasoning_content"] == "deliberate thought"


def _id_message(task_id: str) -> str:
    """The message a tool call makes when the agent asks for task ids only."""
    return (
        f"Task completed. task_id={task_id}. Retrieve the result with the task tool: "
        f"task(action='result', task_id='{task_id}')."
    )


class _ToolCallDriver:
    """Drives one in-loop tool call through a stubbed run."""

    async def _call_tool(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tool: Tool,
        args_json: str,
        *,
        by_id: bool,
        results_by_id: bool | None = None,
        with_text: bool = True,
        count: int = 1,
    ) -> tuple[str, AgentRun]:
        """Drive in-loop calls of the tool and return the last tool output plus the run."""
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys",
                      validator=_accept_any, tool_results_by_id=by_id)
        agent.add_tool_adapter(tool, results_by_id)
        out_holder: dict[str, str] = {}
        _patch_runner_factory(monkeypatch, partial(_make_tool_stub, tool_name=tool.name,
                                                   args_json=args_json, capture=out_holder,
                                                   with_text=with_text, count=count))
        run = await agent.run("do it")
        await run.task.result()
        return out_holder["out"], run

    async def _sdk_tools(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tools: tuple[Tool, ...],
        *,
        by_id: bool,
        results_by_id: bool | None = None,
    ) -> list:
        """Run one stubbed agent run and return the SDK tools the model was given."""
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        captured: list[_SdkAgent] = []

        def factory(sdk_agent: _SdkAgent, _input: str | list) -> MagicMock:
            captured.append(sdk_agent)
            return _stub_result([_text_delta("ok", 1)], "ok")

        _patch_runner_factory(monkeypatch, factory)
        agent = Agent(name="res", system_prompt="sys",
                      validator=_accept_any, tool_results_by_id=by_id)
        for tool in tools:
            agent.add_tool_adapter(tool, results_by_id)
        run = await agent.run("do it")
        await run.task.result()
        return list(captured[-1].tools)


class TestToolResultsById(_ToolCallDriver):
    """With tool_results_by_id on, a tracked tool call hands the model only its task id."""

    class _UntrackedEcho(Tool):
        name: ClassVar[str] = "untracked_echo"
        is_tracked: ClassVar[bool] = False
        description: ClassVar[str] = "Echo without tracking."
        summary: ClassVar[str] = "test"
        params: ClassVar[tuple[Param, ...]] = (Param("query", str, description="The query."),)

        @override
        async def call(self, params: dict) -> ToolResult:
            return ToolResult(text=f"untracked {params['query']}")

    async def test_tracked_tool_hands_back_the_task_id_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, run = await self._call_tool(monkeypatch, _BlockTool(), '{"query":"cats"}', by_id=True)
        calls = await run.read_tool_calls()
        assert out == _id_message(calls[0].task_id)
        assert "results for cats" not in out

    async def test_result_stays_on_disk_under_the_reported_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _out, run = await self._call_tool(monkeypatch, _BlockTool(), '{"query":"cats"}', by_id=True)
        call = (await run.read_tool_calls())[0]
        result_file = Path(store_module.tool_store.data_dir) / call.run_dir / "result"
        assert result_file.read_text() == "results for cats"
        record = await store_module.tool_store.get(call.task_id)
        assert record.tool_name == "search_web"
        assert record.params == {"query": "cats"}

    async def test_tool_call_still_reaches_the_conversation_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _out, run = await self._call_tool(monkeypatch, _BlockTool(), '{"query":"cats"}', by_id=True)
        call = (await run.read_tool_calls())[0]
        results = [e async for e in run.stream_conversation() if e.type == "tool_result"]
        assert len(results) == 1
        assert results[0].task_id == call.task_id
        assert results[0].tool_name == "search_web"

    async def test_flag_off_still_hands_back_the_real_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, run = await self._call_tool(monkeypatch, _BlockTool(), '{"query":"cats"}', by_id=False)
        calls = await run.read_tool_calls()
        assert out == "results for cats"
        assert calls[0].task_id
        results = [e async for e in run.stream_conversation() if e.type == "tool_result"]
        assert results[0].task_id == calls[0].task_id

    async def test_async_tool_hands_back_the_task_id_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, run = await self._call_tool(monkeypatch, _AsyncTool(), '{"prompt":"a cat"}', by_id=True)
        calls = await run.read_tool_calls()
        assert out == _id_message(calls[0].task_id)
        assert "Task started" not in out

    async def test_truncating_tool_hands_back_the_id_message_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, run = await self._call_tool(monkeypatch, _BigTool(), '{"query":"q"}',
                                          by_id=True, with_text=False)
        task_id = (await run.read_tool_calls())[0].task_id
        assert out == _id_message(task_id)
        assert "[Truncated:" not in out

    @pytest.mark.parametrize("by_id,has_max_char", ((False, True), (True, False)))
    async def test_long_tool_exposes_max_char_only_outside_by_id_mode(
        self, monkeypatch: pytest.MonkeyPatch, by_id: bool, has_max_char: bool
    ) -> None:
        tools = await self._sdk_tools(monkeypatch, (_BigTool(),), by_id=by_id)
        ft = next(t for t in tools if t.name == "big")
        assert ("max_char" in ft.params_json_schema["properties"]) is has_max_char

    async def test_untracked_tool_keeps_handing_back_its_own_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, _run = await self._call_tool(monkeypatch, self._UntrackedEcho(), '{"query":"cats"}', by_id=True)
        assert out == "untracked cats"

    async def test_cached_hit_hands_back_the_first_run_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, run = await self._call_tool(monkeypatch, _CachedTool(), '{"query":"cats"}',
                                         by_id=True, count=2)
        calls = await run.read_tool_calls()
        assert len(calls) == 2
        assert len({c.task_id for c in calls}) == 1
        assert out == _id_message(calls[0].task_id)
        assert "results for cats" not in out

    async def test_failed_run_hands_back_its_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, _run = await self._call_tool(monkeypatch, _FailTool(), '{"query":"cats"}', by_id=True)
        assert out == "Tool 'boom' failed: bad"
        assert "Task completed." not in out

    async def test_rejected_run_hands_back_its_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, _run = await self._call_tool(monkeypatch, _BoundedTool(), '{"count":99}', by_id=True)
        assert out == "Error: count: Input should be less than or equal to 10."
        assert "Task completed." not in out


class TestToolResultByIdOptIn(_ToolCallDriver):
    """A tool added with results_by_id=True hands back only its task id, even with the flag off."""

    async def test_tool_opt_in_hands_back_the_id_message_with_flag_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out, run = await self._call_tool(monkeypatch, _IdOnlyTool(), '{"query":"cats"}',
                                         by_id=False, results_by_id=True)
        calls = await run.read_tool_calls()
        assert out == _id_message(calls[0].task_id)
        assert "results for cats" not in out

    async def test_tool_opt_in_wins_over_the_files_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, run = await self._call_tool(monkeypatch, _IdOnlyMediaTool(), '{"query":"cats"}',
                                         by_id=False, results_by_id=True)
        calls = await run.read_tool_calls()
        assert out == _id_message(calls[0].task_id)
        assert "media cats" not in out

    @pytest.mark.parametrize("by_id", (False, True))
    async def test_long_tool_opt_in_hides_max_char_from_the_signature(
        self, monkeypatch: pytest.MonkeyPatch, by_id: bool
    ) -> None:
        tools = await self._sdk_tools(monkeypatch, (_BigIdOnlyTool(),), by_id=by_id, results_by_id=True)
        ft = next(t for t in tools if t.name == "big_id_only")
        assert "max_char" not in ft.params_json_schema["properties"]


class TestResultsByIdLayering(_ToolCallDriver):
    """The per-tool value wins over the flag, and None defers to the flag's own value."""

    @pytest.mark.parametrize(
        ("by_id", "results_by_id", "id_only"),
        [(False, True, True), (True, False, False), (True, None, True), (False, None, False)],
        ids=["explicit_true_flag_off", "explicit_false_flag_on", "none_flag_on", "none_flag_off"],
    )
    async def test_id_only_decision_layers(
        self, monkeypatch: pytest.MonkeyPatch, by_id: bool, results_by_id: bool | None, id_only: bool
    ) -> None:
        out, run = await self._call_tool(monkeypatch, _BlockTool(), '{"query":"cats"}',
                                         by_id=by_id, results_by_id=results_by_id)
        calls = await run.read_tool_calls()
        if id_only:
            assert out == _id_message(calls[0].task_id)
            assert "results for cats" not in out
        else:
            assert out == "results for cats"


class TestUntrackedToolInLoop:
    """An untracked tool runs in-loop without a guard and leaves no call report."""

    class _UntrackedEchoTool(Tool):
        name: ClassVar[str] = "fake_untracked"
        is_tracked: ClassVar[bool] = False
        description: ClassVar[str] = "An untracked tool."
        summary: ClassVar[str] = "test"
        params: ClassVar[tuple[Param, ...]] = (
            Param("query", str, description="The query."),
        )

        @override
        async def call(self, params: dict) -> ToolResult:
            return ToolResult(text=f"result for {params['query']}")

    async def test_untracked_tool_runs_in_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        tool = self._UntrackedEchoTool()
        agent = Agent(
            name="test_agent",
            system_prompt="sys",
            validator=_accept_any,
        )
        agent.add_tool_adapter(tool)
        _patch_runner(monkeypatch, _stub_result([], "ok"))
        run = await agent.run("go")
        assert await run.task.result() == "ok"
        assert await run.read_tool_calls() == []


class TestAgentRun:
    async def test_agentrun_task_property_is_the_wrapped_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        _patch_runner(monkeypatch, _stub_result([_text_delta("ok", 1)], "ok"))
        wrapped: list[Task] = []

        def _capturing_agent_run(task: Task) -> AgentRun:
            wrapped.append(task)
            return AgentRun(task)

        monkeypatch.setattr(engine, "AgentRun", _capturing_agent_run)
        run = await agent.run("do it")
        assert await run.task.result() == "ok"
        # one handle is returned from run, one wraps the task for the validator
        assert len(wrapped) == 2
        assert all(t is run.task for t in wrapped)

    async def test_agentrun_conversation_tails_the_run_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        _patch_runner(monkeypatch, _stub_result([_text_delta("hello", 1)], "hello"))
        run = await agent.run("do it")
        events = [e async for e in run.stream_conversation()]
        assert [e.type for e in events] == ["system", "user", "content", "assistant"]
        assert events[1].text == "do it" and events[-1].text == "hello"

    @pytest.mark.timeout(30)
    async def test_agentrun_events_reads_the_run_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sp", validator=_accept_any)
        _patch_runner(monkeypatch, _stub_result([_text_delta("hello", 1)], "hello"))
        run = await agent.run("do it")
        await run.task.result()
        events = await run.read_events()
        assert [e.type for e in events] == ["system", "user", "content", "assistant"]
        assert events[1].text == "do it" and events[-1].text == "hello"

    @pytest.mark.timeout(30)
    async def test_validator_reads_events_mid_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")

        async def validator(text: str, agent_run: AgentRun) -> ValidationResult[str]:
            events = await agent_run.read_events()
            if any(e.type == "assistant" and e.text == text for e in events):
                return ValidationResult(is_valid=True, value=text)
            return ValidationResult(is_valid=False, error="assistant event not in the conversation yet")

        agent = Agent(name="res", system_prompt="sp", validator=validator)
        _patch_runner(monkeypatch, _stub_result([_text_delta("ok", 1)], "ok"))
        run = await agent.run("do it")
        assert await run.task.result() == "ok"

    async def test_agentrun_tool_calls_reads_the_conversation_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        agent.add_tool_adapter(_BlockTool())
        _patch_runner_factory(monkeypatch, partial(_make_tool_stub, tool_name="search_web",
                                                   args_json='{"query":"cats"}'))
        run = await agent.run("do it")
        await run.task.result()
        calls = await run.read_tool_calls()
        assert len(calls) == 1
        assert calls[0].tool_name == "search_web"
        assert calls[0].params == {"query": "cats"}


class TestToolCallHistory:
    """tool_calls is derived from the run's own conversation log."""

    async def _drive(self, monkeypatch: pytest.MonkeyPatch, tool: Tool, args_json: str) -> AgentRun:
        """Drive one in-loop call of the tool and return the finished run."""
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        agent.add_tool_adapter(tool)
        _patch_runner_factory(monkeypatch, partial(_make_tool_stub, tool_name=tool.name,
                                                   args_json=args_json))
        run = await agent.run("do it")
        await run.task.result()
        return run

    @staticmethod
    async def _result_statuses(run: AgentRun) -> list[str | None]:
        return [e.status for e in await run.read_events() if e.type == "tool_result"]

    async def test_completed_call_is_recorded_as_completed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await self._drive(monkeypatch, _BlockTool(), '{"query":"cats"}')
        assert await self._result_statuses(run) == ["completed"]
        assert [c.status for c in await run.read_tool_calls()] == ["completed"]

    async def test_failed_call_is_recorded_as_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await self._drive(monkeypatch, _FailTool(), '{"query":"cats"}')
        assert await self._result_statuses(run) == ["failed"]
        assert [c.status for c in await run.read_tool_calls()] == ["failed"]

    async def test_cancelled_call_is_recorded_as_cancelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await self._drive(monkeypatch, _StopTool(), '{"query":"cats"}')
        assert await self._result_statuses(run) == ["cancelled"]
        assert [c.status for c in await run.read_tool_calls()] == ["cancelled"]

    async def test_rejected_params_are_recorded_without_a_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await self._drive(monkeypatch, _BoundedTool(), '{"count":99}')
        assert await self._result_statuses(run) == ["rejected"]
        assert await run.read_tool_calls() == []

    async def test_tool_tier_rejection_is_recorded_as_a_failed_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rejection the tool makes in its own call is a failed run with a call report."""
        run = await self._drive(monkeypatch, _ToolTierRejectTool(), '{"count":99}')
        assert await self._result_statuses(run) == ["failed"]
        assert [c.status for c in await run.read_tool_calls()] == ["failed"]

    async def test_run_dir_points_at_the_call_s_run_folder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await self._drive(monkeypatch, _BlockTool(), '{"query":"cats"}')
        call = (await run.read_tool_calls())[0]
        assert call.run_dir == f"runs/{call.task_id}"
        result_file = Path(store_module.task_store.data_dir) / call.run_dir / "result"
        assert result_file.read_text() == "results for cats"

    async def test_result_without_its_call_is_left_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        agent.add_tool_adapter(_BlockTool())
        _patch_runner_factory(monkeypatch, partial(_make_tool_stub, tool_name="search_web",
                                                   args_json='{"query":"cats"}', call_id="C0"))
        run = await agent.run("do it")
        await run.task.result()
        events = await run.read_events()
        assert [e.call_id for e in events if e.type == "tool_call"] == ["C0"]
        results = [e for e in events if e.type == "tool_result"]
        assert [(e.call_id, e.status) for e in results] == [("C1", "completed")]
        assert await run.read_tool_calls() == []

    async def test_every_attempt_keeps_its_own_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        checked: list[str] = []

        def _accept_second(text: str, _run: AgentRun) -> ValidationResult[str]:
            checked.append(text)
            if len(checked) > 1:
                return ValidationResult(is_valid=True, value=text)
            return ValidationResult(is_valid=False, error="try again")

        agent = Agent(name="res", system_prompt="sys",
                      validator=_accept_second, max_attempts=2)
        agent.add_tool_adapter(_BlockTool())
        _patch_runner_factory(monkeypatch, partial(_make_tool_stub, tool_name="search_web",
                                                   args_json='{"query":"cats"}'))
        run = await agent.run("do it")
        await run.task.result()
        assert [e.attempt for e in await run.read_events() if e.type == "tool_result"] == [1, 2]
        calls = await run.read_tool_calls()
        assert [c.tool_name for c in calls] == ["search_web", "search_web"]
        assert [c.status for c in calls] == ["completed", "completed"]
        assert len({c.task_id for c in calls}) == 2


class TestCachedToolCalls:
    """A call that reuses a stored run is kept, under the task id of that run."""

    async def _drive(self, monkeypatch: pytest.MonkeyPatch, count: int) -> AgentRun:
        """Drive count in-loop calls of the cached tool and return the finished run."""
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        agent.add_tool_adapter(_CachedTool())
        _patch_runner_factory(monkeypatch, partial(_make_tool_stub, tool_name="cached_search",
                                                   args_json='{"query":"cats"}', count=count))
        run = await agent.run("do it")
        await run.task.result()
        return run

    @staticmethod
    async def _run_text(run_dir: str) -> str:
        return await read_run_mcp_text(os.path.join(store_module.task_store.data_dir, run_dir))

    async def test_repeat_call_is_kept_under_the_first_task_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = await (await self._drive(monkeypatch, 2)).read_tool_calls()
        assert [c.tool_name for c in calls] == ["cached_search", "cached_search"]
        assert [c.status for c in calls] == ["completed", "completed"]
        assert [c.params for c in calls] == [{"query": "cats"}, {"query": "cats"}]
        assert len({c.task_id for c in calls}) == 1
        record = await store_module.tool_store.get(calls[0].task_id)
        assert record.tool_name == "cached_search"

    async def test_reused_run_dir_holds_readable_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = await (await self._drive(monkeypatch, 2)).read_tool_calls()
        stored_dir = f"runs/{calls[0].task_id}"
        assert [c.run_dir for c in calls] == [stored_dir, stored_dir]
        assert await self._run_text(calls[1].run_dir) == "results for cats"

    async def test_reuse_is_marked_in_the_conversation_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await self._drive(monkeypatch, 2)
        results = [e for e in await run.read_events() if e.type == "tool_result"]
        assert [e.status for e in results] == ["completed", "completed"]
        assert len({e.task_id for e in results}) == 1

    async def test_only_the_first_call_makes_a_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await self._drive(monkeypatch, 2)
        calls = await run.read_tool_calls()
        children = await store_module.task_store.list_subtasks(str(run.task.id))
        assert [t.task_id for t in children] == [calls[0].task_id]

    async def test_hit_on_an_earlier_run_borrows_its_task_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stored = await self._drive(monkeypatch, 1)
        reusing = await self._drive(monkeypatch, 1)
        stored_id = (await stored.read_tool_calls())[0].task_id
        borrowed = (await reusing.read_tool_calls())[0]
        assert borrowed.task_id == stored_id
        assert borrowed.run_dir == f"runs/{stored_id}"
        assert await self._run_text(borrowed.run_dir) == "results for cats"

    async def test_hit_on_another_run_is_marked_and_makes_no_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        await self._drive(monkeypatch, 1)
        reusing = await self._drive(monkeypatch, 1)
        results = [e for e in await reusing.read_events() if e.type == "tool_result"]
        assert [e.status for e in results] == ["completed"]
        assert await store_module.task_store.list_subtasks(str(reusing.task.id)) == []


class TestSystemLine:
    """The system prompt is the first line of the conversation file."""

    async def test_system_line_is_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="You are a helpful agent.", validator=_accept_any)
        _patch_runner(monkeypatch, _stub_result([_text_delta("hi", 1)], "hi"))
        run = await agent.run("do it")
        await run.task.result()
        events = [e async for e in run.stream_conversation()]
        assert events[0].type == "system"
        assert events[0].text == "You are a helpful agent."
        assert events[0].attempt == 1

    async def test_system_line_written_to_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await _completed_run(monkeypatch)
        path = Path(store_module.task_store.data_dir, "runs", str(run.task.id), "conversation.jsonl")
        first_line = json.loads(path.read_text().splitlines()[0])
        assert first_line["type"] == "system"
        assert first_line["text"] == "sp"


class TestCallId:
    """Tool lines carry the SDK call id for pairing."""

    async def test_tool_call_and_result_carry_call_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        call_item = ToolCallItem(
            agent=MagicMock(),
            raw_item={"name": "brave_search", "call_id": "call-abc", "arguments": '{"query": "test"}'},
        )
        result_item = ToolCallOutputItem(
            agent=MagicMock(),
            raw_item={"call_id": "call-abc"}, output="result text",
        )
        # Build the stub with tool call items
        async def _gen() -> AsyncIterator:
            yield _text_delta("thinking", 1)
            yield RunItemStreamEvent(item=call_item, name="tool_called")
            yield RunItemStreamEvent(item=result_item, name="tool_output")
            yield _text_delta("answer", 4)
        stub = MagicMock()
        stub.stream_events = MagicMock(side_effect=_gen)
        stub.final_output = "answer"
        stub.to_input_list = MagicMock(return_value=[{"role": "user", "content": "stub"}])
        stub.cancel = MagicMock()
        _patch_runner(monkeypatch, stub)
        agent = Agent(name="res", system_prompt="sp", validator=_accept_any)
        run = await agent.run("do it")
        await run.task.result()
        conv = [e async for e in run.stream_conversation()]
        tc = [e for e in conv if e.type == "tool_call"]
        tr = [e for e in conv if e.type == "tool_result"]
        assert len(tc) == 1 and tc[0].call_id == "call-abc"
        assert len(tr) == 1 and tr[0].call_id == "call-abc"


class TestIncrementalFlush:
    """Thinking and content sections flush per delta."""

    async def test_content_flushes_per_delta(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        chunks = [_text_delta("x" * 100, i + 1) for i in range(5)]
        final = "x" * 500
        _patch_runner(monkeypatch, _stub_result(chunks, final))
        agent = Agent(name="res", system_prompt="sp", validator=_accept_any)
        run = await agent.run("do it")
        await run.task.result()
        content_lines = [e async for e in run.stream_conversation() if e.type == "content"]
        assert len(content_lines) == 5
        assert "".join(e.text for e in content_lines) == final

    async def test_small_deltas_each_get_own_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        _patch_runner(monkeypatch, _stub_result(
            [_text_delta("a", 1), _text_delta("b", 2), _text_delta("c", 3)], "abc"))
        agent = Agent(name="res", system_prompt="sp", validator=_accept_any)
        run = await agent.run("do it")
        await run.task.result()
        content_lines = [e async for e in run.stream_conversation() if e.type == "content"]
        assert [e.text for e in content_lines] == ["a", "b", "c"]

    async def test_small_content_single_flush(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        _patch_runner(monkeypatch, _stub_result([_text_delta("short", 1)], "short"))
        agent = Agent(name="res", system_prompt="sp", validator=_accept_any)
        run = await agent.run("do it")
        await run.task.result()
        content_lines = [e async for e in run.stream_conversation() if e.type == "content"]
        assert len(content_lines) == 1
        assert content_lines[0].text == "short"


class TestMilestoneLog:
    """The engine writes milestone lines to the run log."""

    async def test_log_contains_milestones(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sp", validator=_accept_any)
        _patch_runner(monkeypatch, _stub_result([_text_delta("ok", 1)], "ok"))
        logcapture.install_log_tee()
        try:
            run = await agent.run("do it")
            await run.task.result()
        finally:
            logcapture.restore_log_tee()
        log_path = Path(store_module.task_store.data_dir, "runs", str(run.task.id), "log")
        assert log_path.is_file()
        text = log_path.read_text()
        assert "agent started" in text
        assert "system prompt loaded" in text
        assert "attempt 1/" in text
        assert "agent completed" in text


class _LiveBlock(Tool):
    name: ClassVar[str] = "note"
    description: ClassVar[str] = "Return a short note."
    is_tracked: ClassVar[bool] = True
    params: ClassVar[tuple[Param, ...]] = (Param(name="text", type=str, description="Text to note."),)

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text=f"noted: {params['text']}")


_THINKING_VARS = ("REASON_THINKING_BACKEND_URL", "REASON_THINKING_MODEL", "REASON_THINKING_API_KEY")


def _spy_backend(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, list[str]]:
    """Record how the model client is built and which model each attempt asks for."""
    client_spy = MagicMock(wraps=engine.AsyncOpenAI)
    monkeypatch.setattr(engine, "AsyncOpenAI", client_spy)
    models: list[str] = []

    def factory(sdk_agent: _SdkAgent, _input: str | list) -> MagicMock:
        models.append(sdk_agent.model.model)
        return _stub_result([_text_delta("x", 1)], "x")

    _patch_runner_factory(monkeypatch, factory)
    return client_spy, models


async def _finish(agent: Agent) -> None:
    """Run the agent once and wait for its result."""
    run = await agent.run("do it")
    await run.task.result()


class TestAgentBackend:
    @pytest.fixture(autouse=True)
    def _clean_thinking_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Start each test with no thinking backend variable set."""
        for name in _THINKING_VARS:
            monkeypatch.delenv(name, raising=False)

    @pytest.mark.parametrize("url", ["http://backend.test", "http://backend.test/v1/"])
    async def test_unset_model_and_key_send_the_defaults(self, monkeypatch: pytest.MonkeyPatch, url: str) -> None:
        """With only the address set, the client gets one /v1 and the default model and key."""
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", url)
        client_spy, models = _spy_backend(monkeypatch)
        await _finish(Agent(name="res", system_prompt="sp", validator=_accept_any))
        client_spy.assert_called_once_with(base_url="http://backend.test/v1", api_key="not-needed")
        assert models == ["default"]

    async def test_set_model_and_key_are_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A set model and key reach the client."""
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        monkeypatch.setenv("REASON_THINKING_MODEL", "deep")
        monkeypatch.setenv("REASON_THINKING_API_KEY", "sk-t")
        client_spy, models = _spy_backend(monkeypatch)
        await _finish(Agent(name="res", system_prompt="sp", validator=_accept_any))
        client_spy.assert_called_once_with(base_url="http://backend.test/v1", api_key="sk-t")
        assert models == ["deep"]

    async def test_passed_backend_wins_over_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A backend given to the agent is used instead of the thinking backend."""
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        monkeypatch.setenv("REASON_THINKING_MODEL", "deep")
        client_spy, models = _spy_backend(monkeypatch)
        given = config.Backend(url="http://given.test/v1", model="given-model", api_key="sk-g")
        await _finish(Agent(name="res", system_prompt="sp", validator=_accept_any, backend=given))
        client_spy.assert_called_once_with(base_url="http://given.test/v1", api_key="sk-g")
        assert models == ["given-model"]

    async def test_missing_thinking_backend_fails_the_run(self, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
        """With no backend passed and none set, the run fails and names the variable."""
        client_spy, _ = _spy_backend(monkeypatch)
        message = "REASON_THINKING_BACKEND_URL is not set: needed by research and browser_use"
        agent = Agent(name="res", system_prompt="sp", validator=_accept_any)
        run = await agent.run("do it")
        with pytest.raises(config.MissingSettingError, match=f"^{message}$"):
            await run.task.result()
        assert f"[config] {message}\n" in capsys.readouterr().err
        client_spy.assert_not_called()


_FIRST_DELAY = 1.0
_SECOND_DELAY = 2.0
_THIRD_DELAY = 4.0
_SHORT_RUN_TIMEOUT = 0.05
_CANCEL_MAX_ELAPSED = 0.4
_CORRECTIVE_NOTICE = (
    "\n\nIMPORTANT NOTE: Your last response is dropped. "
    "Please correct the error and re-generate your entire response in your "
    "next message. Please don't put any filler content. Your message must "
    "contain ONLY the final output."
)


def _sse_chunk(delta: dict, finish_reason: str | None) -> bytes:
    chunk = {
        "id": "1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "default",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode("utf-8")


def _sse_text_response(text: str) -> httpx.Response:
    body = (
        _sse_chunk({"role": "assistant", "content": text}, None)
        + _sse_chunk({}, "stop")
        + b"data: [DONE]\n\n"
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def _sse_tool_call_response(call_id: str, name: str, arguments: str) -> httpx.Response:
    tool_call = {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    body = (
        _sse_chunk({"role": "assistant", "tool_calls": [tool_call]}, None)
        + _sse_chunk({}, "tool_calls")
        + b"data: [DONE]\n\n"
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def _http_500_response() -> httpx.Response:
    return httpx.Response(500, json={"error": {"message": "boom"}})


def _accept_good(text: str, _run: AgentRun) -> ValidationResult[str]:
    if text == "good":
        return ValidationResult(is_valid=True, value=text)
    return ValidationResult(is_valid=False, error="need good")


def _http_request() -> httpx.Request:
    return httpx.Request("POST", "http://backend.test/v1/chat/completions")


def _http_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_http_request())


def _conn_error(message: str = "backend down") -> APIConnectionError:
    return APIConnectionError(message=message, request=_http_request())


def _backoff_delays(count: int) -> list[float]:
    return [_FIRST_DELAY * (2**i) for i in range(count)]


def _sleep_delays() -> list[float]:
    return [call.args[0] for call in engine.asyncio.sleep.await_args_list]


def _failing_stub(
    error: Exception,
    on_cancel: Callable[[], None] | None = None,
    events: list[RawResponsesStreamEvent] | None = None,
) -> MagicMock:
    async def _gen() -> AsyncIterator[RawResponsesStreamEvent]:
        for event in events if events is not None else ():
            yield event
        raise error

    stub = _build_stub(_gen, None)
    if on_cancel is not None:
        stub.cancel = MagicMock(side_effect=on_cancel)
    return stub


class _CountedTool(Tool):
    name: ClassVar[str] = "counted_note"
    description: ClassVar[str] = "Record one note call."
    is_tracked: ClassVar[bool] = True
    calls: ClassVar[int] = 0
    params: ClassVar[tuple[Param, ...]] = (Param(name="text", type=str, description="Text to note."),)

    @override
    async def call(self, params: dict) -> ToolResult:
        _CountedTool.calls += 1
        return ToolResult(text=f"noted:{params['text']}")


def _install_mock_openai_handler(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response] | Callable[[httpx.Request], Awaitable[httpx.Response]],
    clients: list[httpx.AsyncClient],
) -> None:
    real_openai = engine.AsyncOpenAI

    def _build_client(base_url: str, api_key: str) -> AsyncOpenAI:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
        clients.append(http_client)
        return real_openai(base_url=base_url, api_key=api_key, http_client=http_client)

    monkeypatch.setattr(engine, "AsyncOpenAI", _build_client)


def _wire_mock_openai(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[httpx.Response],
    clients: list[httpx.AsyncClient],
    wire_bodies: list[dict] | None = None,
) -> tuple[MagicMock, list[list[dict]]]:
    wire_messages: list[list[dict]] = []
    queue = list(responses)

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        wire_messages.append(body["messages"])
        if wire_bodies is not None:
            wire_bodies.append(body)
        return queue.pop(0)

    _install_mock_openai_handler(monkeypatch, _handler, clients)
    model_spy = MagicMock(wraps=engine.OpenAIChatCompletionsModel)
    monkeypatch.setattr(engine, "OpenAIChatCompletionsModel", model_spy)
    return model_spy, wire_messages


async def _stop_in_drain_and_assert_failed(
    run: AgentRun, entered: asyncio.Event, release: asyncio.Event
) -> None:
    await asyncio.wait_for(entered.wait(), timeout=5)
    run.cancel_run_loop()
    release.set()
    with pytest.raises(RuntimeError, match="stopped on request"):
        await run.task.result()
    assert run.task.status.value == "failed"


async def _close_clients(clients: list[httpx.AsyncClient]) -> None:
    for client in clients:
        await client.aclose()


_RETRYABLE_CASES: list[tuple[str, Callable[[], Exception]]] = [
    ("api_connection", _conn_error),
    ("api_timeout", lambda: APITimeoutError(request=_http_request())),
    ("internal_500", lambda: InternalServerError("500", response=_http_response(500), body=None)),
    ("internal_503", lambda: InternalServerError("503", response=_http_response(503), body=None)),
    ("rate_limit_429", lambda: RateLimitError("429", response=_http_response(429), body=None)),
    ("conflict_409", lambda: ConflictError("409", response=_http_response(409), body=None)),
    ("status_408", lambda: APIStatusError("408", response=_http_response(408), body=None)),
    ("response_validation", lambda: APIResponseValidationError(response=_http_response(200), body=None, message="bad")),
    ("mid_stream_api_error", lambda: APIError("stream broke", request=_http_request(), body=None)),
    ("httpx_status_502", lambda: httpx.HTTPStatusError("502", request=_http_request(), response=_http_response(502))),
    ("httpx_status_429", lambda: httpx.HTTPStatusError("429", request=_http_request(), response=_http_response(429))),
    ("httpx_connect", lambda: httpx.ConnectError("connect", request=_http_request())),
    ("httpx_read_timeout", lambda: httpx.ReadTimeout("timeout", request=_http_request())),
    ("httpx_remote_protocol", lambda: httpx.RemoteProtocolError("closed", request=_http_request())),
    ("httpx_stream_closed", httpx.StreamClosed),
    ("connection_reset", lambda: ConnectionResetError("reset")),
    ("json_decode", lambda: json.JSONDecodeError("truncated", "{", 1)),
    ("model_behavior", lambda: ModelBehaviorError("incomplete stream")),
]

_NON_RETRYABLE_CASES: list[tuple[str, Callable[[], Exception], type[Exception]]] = [
    ("bad_request_400", lambda: BadRequestError("400", response=_http_response(400), body=None), BadRequestError),
    ("auth_401", lambda: AuthenticationError("401", response=_http_response(401), body=None), AuthenticationError),
    ("permission_403", lambda: PermissionDeniedError("403", response=_http_response(403), body=None), PermissionDeniedError),
    ("not_found_404", lambda: NotFoundError("404", response=_http_response(404), body=None), NotFoundError),
    ("unprocessable_422", lambda: UnprocessableEntityError("422", response=_http_response(422), body=None), UnprocessableEntityError),
    ("httpx_status_400", lambda: httpx.HTTPStatusError("400", request=_http_request(), response=_http_response(400)), httpx.HTTPStatusError),
    ("value_error", lambda: ValueError("bad value"), ValueError),
]


class TestServerRetry:
    """Server and connection errors retry within the same attempt."""

    async def _run_stubs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stubs: list[MagicMock],
        timeout: float | None = None,
    ) -> tuple[AgentRun, MagicMock]:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        run_mock = _patch_runner_seq(monkeypatch, stubs)
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any, timeout=timeout)
        run = await agent.run("do it")
        return run, run_mock

    async def _run_scripted_turns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        steps: list[str | tuple[str, str] | Exception],
    ) -> AgentRun:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        _CountedTool.calls = 0
        _ScriptedModel.received = []
        _ScriptedModel.order = []
        _ScriptedModel.script = steps
        monkeypatch.setattr(engine, "OpenAIChatCompletionsModel", _ScriptedModel)
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        agent.add_tool_adapter(_CountedTool())
        return await agent.run("do it")

    async def _run_wire(
        self,
        monkeypatch: pytest.MonkeyPatch,
        responses: list[httpx.Response],
        validator: Callable[[str, AgentRun], ValidationResult[str]] = _accept_any,
        *,
        with_tool: bool = False,
    ) -> tuple[AgentRun, list[list[dict]], MagicMock]:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        _CountedTool.calls = 0
        clients: list[httpx.AsyncClient] = []
        model_spy, wire_messages = _wire_mock_openai(monkeypatch, responses, clients)
        try:
            agent = Agent(name="res", system_prompt="sys", validator=validator)
            if with_tool:
                agent.add_tool_adapter(_CountedTool())
            run = await agent.run("do it")
            await run.task.result()
            return run, wire_messages, model_spy
        finally:
            await _close_clients(clients)

    async def _park_in_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[AgentRun, MagicMock, float]:
        backoff_ready = asyncio.Event()
        stubs = [
            _failing_stub(_conn_error(), on_cancel=backoff_ready.set),
            _stub_result([_text_delta("ok", 1)], "ok"),
        ]
        run, run_mock = await self._run_stubs(monkeypatch, stubs)
        await asyncio.wait_for(backoff_ready.wait(), timeout=5)
        await asyncio.sleep(0)
        return run, run_mock, time.monotonic()

    @pytest.mark.fast_sleeps
    async def test_exponential_backoff_recovers_within_first_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stubs = [
            _failing_stub(_conn_error()),
            _failing_stub(_conn_error()),
            _failing_stub(_conn_error()),
            _stub_result([_text_delta("Answer: 42", 1)], "Answer: 42"),
        ]
        run, run_mock = await self._run_stubs(monkeypatch, stubs)
        assert await run.task.result() == "Answer: 42"
        assert run_mock.call_count == 4
        assert _sleep_delays() == [_FIRST_DELAY, _SECOND_DELAY, _THIRD_DELAY]
        assert [s.cancel.call_count for s in stubs] == [1, 1, 1, 1]
        events = await run.read_events()
        assert [e.text for e in events if e.type == "user"] == ["do it"]
        assert [(e.attempt, e.text) for e in events if e.type == "assistant"] == [(1, "Answer: 42")]

    @pytest.mark.fast_sleeps
    async def test_mid_stream_error_retries_within_same_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stubs = [
            _failing_stub(
                httpx.RemoteProtocolError("peer closed"),
                events=[_reasoning_delta("think", 1), _text_delta("partial", 2)],
            ),
            _stub_result([_text_delta("clean", 1)], "clean"),
        ]
        run, run_mock = await self._run_stubs(monkeypatch, stubs)
        assert await run.task.result() == "clean"
        assert run_mock.call_count == 2
        assert _sleep_delays() == [_FIRST_DELAY]
        events = await run.read_events()
        assert [e.attempt for e in events] == [1, 1, 1, 1, 1, 1]
        assert [e.text for e in events if e.type == "user"] == ["do it"]
        assert [(e.attempt, e.text) for e in events if e.type == "assistant"] == [(1, "clean")]

    @pytest.mark.fast_sleeps
    @pytest.mark.parametrize(
        "make_error",
        [fn for _, fn in _RETRYABLE_CASES],
        ids=[name for name, _ in _RETRYABLE_CASES],
    )
    async def test_retryable_errors_retry_and_recover(
        self, monkeypatch: pytest.MonkeyPatch, make_error: Callable[[], Exception]
    ) -> None:
        stubs = [_failing_stub(make_error()), _stub_result([_text_delta("ok", 1)], "ok")]
        run, run_mock = await self._run_stubs(monkeypatch, stubs)
        assert await run.task.result() == "ok"
        assert run_mock.call_count == 2
        assert _sleep_delays() == [_FIRST_DELAY]

    @pytest.mark.fast_sleeps
    async def test_exhaustion_after_ten_retries_raises_original_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stubs = [_failing_stub(_conn_error("backend down")) for _ in range(11)]
        run, run_mock = await self._run_stubs(monkeypatch, stubs)
        with pytest.raises(APIConnectionError, match="backend down"):
            await run.task.result()
        assert run.task.status.value == "failed"
        assert run_mock.call_count == 11
        assert _sleep_delays() == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]

    @pytest.mark.fast_sleeps
    @pytest.mark.parametrize(
        ("make_error", "expected_cls"),
        [(fn, cls) for _, fn, cls in _NON_RETRYABLE_CASES],
        ids=[name for name, _, _ in _NON_RETRYABLE_CASES],
    )
    async def test_non_retryable_errors_fail_immediately_without_sleep(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_error: Callable[[], Exception],
        expected_cls: type[Exception],
    ) -> None:
        stubs = [_failing_stub(make_error()), _stub_result([_text_delta("ok", 1)], "ok")]
        run, run_mock = await self._run_stubs(monkeypatch, stubs)
        with pytest.raises(expected_cls):
            await run.task.result()
        assert run.task.status.value == "failed"
        assert run_mock.call_count == 1
        assert _sleep_delays() == []

    @pytest.mark.fast_sleeps
    async def test_multi_turn_retry_preserves_completed_tool_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        steps: list[str | tuple[str, str] | Exception] = [
            ("counted_note", '{"text":"hi"}'),
            _conn_error(),
            _conn_error(),
            "Answer: 42",
        ]
        run = await self._run_scripted_turns(monkeypatch, steps)
        assert await run.task.result() == "Answer: 42"
        assert _CountedTool.calls == 1
        assert _sleep_delays() == [_FIRST_DELAY, _SECOND_DELAY]
        expected_turn_two = [
            {"role": "user", "content": "do it"},
            {
                "id": "fc_1",
                "call_id": "call_1",
                "name": "counted_note",
                "arguments": '{"text":"hi"}',
                "type": "function_call",
                "status": "completed",
            },
            {
                "call_id": "call_1",
                "output": "noted:hi",
                "type": "function_call_output",
            },
        ]
        assert _ScriptedModel.received == [
            [{"role": "user", "content": "do it"}],
            expected_turn_two,
            expected_turn_two,
            expected_turn_two,
        ]
        calls = await run.read_tool_calls()
        assert len(calls) == 1
        assert calls[0].tool_name == "counted_note"
        assert calls[0].status == "completed"
        assert [e.attempt for e in await run.read_events() if e.type == "assistant"] == [1]

    @pytest.mark.fast_sleeps
    async def test_consecutive_retry_counter_resets_after_completed_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        steps: list[str | tuple[str, str] | Exception] = (
            [_conn_error() for _ in range(6)]
            + [("counted_note", '{"text":"hi"}')]
            + [_conn_error() for _ in range(6)]
            + ["Answer: 42"]
        )
        run = await self._run_scripted_turns(monkeypatch, steps)
        assert await run.task.result() == "Answer: 42"
        assert _CountedTool.calls == 1
        assert _sleep_delays() == _backoff_delays(6) + _backoff_delays(6)

    @pytest.mark.fast_sleeps
    async def test_sdk_client_disables_hidden_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        responses = [_http_500_response(), _sse_text_response("hello")]
        run, wire_messages, model_spy = await self._run_wire(monkeypatch, responses)
        assert await run.task.result() == "hello"
        assert model_spy.call_args.kwargs["openai_client"].max_retries == 0
        expected = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do it"}]
        assert wire_messages == [expected, expected]
        assert _sleep_delays() == [_FIRST_DELAY]

    @pytest.mark.fast_sleeps
    async def test_wire_messages_preserved_across_turn_and_validation_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        responses = [
            _sse_tool_call_response("call_1", "counted_note", '{"text":"one"}'),
            _http_500_response(),
            _http_500_response(),
            _sse_text_response("bad"),
            _http_500_response(),
            _sse_text_response("good"),
        ]
        run, wire_messages, _spy = await self._run_wire(
            monkeypatch, responses, validator=_accept_good, with_tool=True
        )
        assert await run.task.result() == "good"
        assert _CountedTool.calls == 1
        assert _sleep_delays() == [_FIRST_DELAY, _SECOND_DELAY, _FIRST_DELAY]
        turn_one = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do it"}]
        turn_two = turn_one + [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "counted_note", "arguments": '{"text":"one"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "noted:one"},
        ]
        attempt_two = turn_two + [
            {"role": "assistant", "content": "bad"},
            {"role": "user", "content": "need good" + _CORRECTIVE_NOTICE},
        ]
        assert wire_messages == [
            turn_one,
            turn_two,
            turn_two,
            turn_two,
            attempt_two,
            attempt_two,
        ]

    @pytest.mark.fast_sleeps
    async def test_cancel_run_loop_during_failing_stream_skips_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub, entered, release = _release_in_drain("partial", error=_conn_error())
        run, run_mock = await self._run_stubs(
            monkeypatch, [stub, _stub_result([_text_delta("ok", 1)], "ok")]
        )
        await _stop_in_drain_and_assert_failed(run, entered, release)
        assert run_mock.call_count == 1
        assert _sleep_delays() == []

    async def test_cancel_run_loop_wakes_real_backoff_sleep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run, run_mock, started = await self._park_in_backoff(monkeypatch)
        await _assert_loop_cancelled_fast(run, started)
        assert run_mock.call_count == 1

    async def test_task_cancel_wakes_real_backoff_sleep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run, run_mock, started = await self._park_in_backoff(monkeypatch)
        await task_manager.cancel(str(run.task.id))
        assert time.monotonic() - started < _CANCEL_MAX_ELAPSED
        assert run.task.status.value == "cancelled"
        assert run_mock.call_count == 1

    async def test_timeout_expires_during_backoff_sleep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stubs = [_failing_stub(_conn_error()), _stub_result([_text_delta("ok", 1)], "ok")]
        run, run_mock = await self._run_stubs(monkeypatch, stubs, timeout=_SHORT_RUN_TIMEOUT)
        with pytest.raises(TimeoutError):
            await run.task.result()
        assert run.task.status.value == "failed"
        assert run_mock.call_count == 1


def _overflow_400_formal() -> BadRequestError:
    return BadRequestError(
        "context exceeded",
        response=_http_response(400),
        body={"error": {"code": "context_length_exceeded", "message": "too long"}},
    )


def _overflow_400_prose(tokens: int = 9000) -> BadRequestError:
    return BadRequestError(
        "bad request",
        response=_http_response(400),
        body={"error": {"message": f"Prompt tokens {tokens} exceed maximum limit 8192"}},
    )


def _overflow_500_prose() -> InternalServerError:
    return InternalServerError(
        "500",
        response=_http_response(500),
        body={"error": {"message": "Context window of 8192 tokens exceeded by 12000"}},
    )


def _stub_with_items(
    error: Exception,
    new_items: list[dict],
    input_list: list[dict],
) -> MagicMock:
    stub = _failing_stub(error, events=[_reasoning_delta("part", 1)])
    stub.new_items = list(new_items)
    stub.to_input_list = MagicMock(return_value=list(input_list))
    return stub


async def _assert_loop_cancelled_fast(run: AgentRun, started: float) -> None:
    run.cancel_run_loop()
    with pytest.raises(RuntimeError, match="stopped on request"):
        await run.task.result()
    assert time.monotonic() - started < _CANCEL_MAX_ELAPSED
    assert run.task.status.value == "failed"


def _mock_classifier_http(
    monkeypatch: pytest.MonkeyPatch,
    verdicts: list[bool | int],
) -> list[str]:
    seen_urls: list[str] = []
    queue = list(verdicts)

    async def _post(
        _self: httpx.AsyncClient,
        url: str,
        *,
        json: dict,
        headers: dict[str, str],
    ) -> httpx.Response:
        _ = (json, headers)
        seen_urls.append(url)
        item = queue.pop(0)
        req = httpx.Request("POST", url)
        if not isinstance(item, bool):
            return httpx.Response(item, text="classifier error", request=req)
        content = '{"is_context_overflow": true}' if item else '{"is_context_overflow": false}'
        body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
        return httpx.Response(200, json=body, request=req)

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    return seen_urls


class TestContextOverflowRecovery:
    """Tests for shrinking message history and retrying when the model hits context limits."""

    @pytest.mark.parametrize("repeat_overflow", [False, True])
    async def test_compact_hook_persists_partial_tool_history_and_stops_without_progress(
        self, monkeypatch: pytest.MonkeyPatch, repeat_overflow: bool,
    ) -> None:
        """The fallback commits tool result edits and cannot retry an unchanged proposal."""
        history = [
            {"role": "user", "content": "do it"},
            {"type": "function_call", "call_id": "c1", "name": "note", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "evidence" * 100},
            {"role": "assistant", "content": "small reply"},
        ]
        seen: list[list[dict]] = []

        async def trim(candidate: list[dict]) -> list[dict]:
            seen.append(json.loads(json.dumps(candidate)))
            candidate[-1]["output"] = "archived"
            return candidate

        error = _overflow_400_formal()
        first = _stub_with_items(error, [{"type": "tool"}], history)
        second = _failing_stub(error) if repeat_overflow else _stub_result([_text_delta("done")], "done")
        runner = _patch_runner_seq(monkeypatch, [first, second])
        agent = Agent(name="hook", system_prompt="sys", validator=_accept_any, compact_hook=trim)
        run = await agent.run("do it")
        if repeat_overflow:
            with pytest.raises(BadRequestError) as caught:
                await run.task.result()
            assert caught.value is error
        else:
            assert await run.task.result() == "done"
        expected = history[:2] + [{**history[2], "output": "archived"}]
        assert runner.call_count == 2
        assert runner.call_args_list[1].args[1] == expected
        assert seen[0] == history[:3]
        assert seen[1:] == ([expected] if repeat_overflow else [])
        assert history[2]["output"] == "evidence" * 100

    async def test_compact_hook_failure_preserves_original_model_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A policy error aborts without a further model request."""
        error = _overflow_400_formal()
        runner = _patch_runner_seq(monkeypatch, [_failing_stub(error)])

        async def trim(_candidate: list[dict]) -> None:
            raise ValueError("policy failed")

        agent = Agent(name="hook", system_prompt="sys", validator=_accept_any, compact_hook=trim)
        run = await agent.run("user")
        with pytest.raises(BadRequestError) as caught:
            await run.task.result()
        assert caught.value is error
        assert runner.call_count == 1

    @pytest.mark.parametrize("stop_kind", ["cancel", "timeout"])
    async def test_compact_hook_obeys_stop_and_timeout(self, monkeypatch: pytest.MonkeyPatch, stop_kind: str) -> None:
        """Stop and timeout cancel a waiting hook."""
        parked = asyncio.Event()
        cancelled = asyncio.Event()

        async def trim(_candidate: list[dict]) -> None:
            parked.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        stub = _failing_stub(_overflow_400_formal())
        runner = _patch_runner_seq(monkeypatch, [stub])
        agent = Agent(
            name="hook", system_prompt="sys", validator=_accept_any, compact_hook=trim,
            timeout=0.25 if stop_kind == "timeout" else None,
        )
        run = await agent.run("user")
        await asyncio.wait_for(parked.wait(), timeout=5)
        if stop_kind == "cancel":
            await _assert_loop_cancelled_fast(run, time.monotonic())
        else:
            with pytest.raises(TimeoutError):
                await run.task.result()
        await asyncio.wait_for(cancelled.wait(), timeout=5)
        assert runner.call_count == 1

    @pytest.mark.fast_sleeps
    @pytest.mark.parametrize(
        "exc",
        [
            _overflow_400_prose(),
            APIStatusError("413", response=_http_response(413), body=None),
            APIError(
                "An error occurred during streaming",
                request=_http_request(),
                body={"error": {"message": "SSE context limit 8192 exceeded by 9500"}},
            ),
        ],
        ids=["bad_request_400_prose", "messageless_413", "sse_api_error_body"],
    )
    async def test_pre_retry_overflow_compacts_without_sleep_and_uses_agent_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
        exc: Exception,
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://wrong-env.test/")
        given = config.Backend(url="http://agent-only.test/v1", model="m1", api_key="sk-1")
        seen_urls = _mock_classifier_http(monkeypatch, [True])
        history = [
            {"role": "user", "content": "u" * 10},
            {"id": "r1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "r" * 40}]},
            {"id": "a1", "role": "assistant", "content": "a" * 40},
        ]
        captured_inputs: list[str | list] = []
        stubs = [
            _stub_with_items(exc, [{"type": "marker"}], history),
            _stub_result([_text_delta("recovered", 1)], "recovered"),
        ]

        def _factory(_sdk: _SdkAgent, inp: str | list) -> MagicMock:
            captured_inputs.append(inp)
            return stubs.pop(0)

        _patch_runner_factory(monkeypatch, _factory)
        agent = Agent(name="res", system_prompt="s" * 4, validator=_accept_any, backend=given)
        run = await agent.run("u" * 10)
        assert await run.task.result() == "recovered"
        assert _sleep_delays() == []
        assert captured_inputs[1] == [{"role": "user", "content": "u" * 10}]
        if seen_urls:
            assert seen_urls == ["http://agent-only.test/v1/chat/completions"]

    @pytest.mark.fast_sleeps
    async def test_server_500_overflow_defers_across_retries_then_compacts_and_resets_retries(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        seen_urls = _mock_classifier_http(monkeypatch, [True])
        history = [
            {"role": "user", "content": "u" * 10},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "r" * 80}]},
        ]
        first = _stub_with_items(_overflow_500_prose(), [{"type": "turn"}], history)
        mid_500 = [_failing_stub(_overflow_500_prose()) for _ in range(10)]
        post_compact_transient = _failing_stub(_conn_error())
        final = _stub_result([_text_delta("done", 1)], "done")
        run_mock = _patch_runner_seq(monkeypatch, [first, *mid_500, post_compact_transient, final])

        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        run = await agent.run("u" * 10)
        assert await run.task.result() == "done"
        assert len(seen_urls) == 1
        assert run_mock.call_count == 13
        assert _sleep_delays() == _backoff_delays(10) + [_FIRST_DELAY]
        assert run_mock.call_args_list[11].args[1] == [{"role": "user", "content": "u" * 10}]

    @pytest.mark.fast_sleeps
    async def test_partial_turn_new_items_preserved_during_compaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        partial_history = [
            {"role": "user", "content": "do it"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "t" * 120}]},
            {"type": "function_call", "call_id": "c1", "name": "note", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "saved_evidence"},
        ]
        first = _stub_with_items(_overflow_400_formal(), [{"type": "tool"}], partial_history)
        second = _stub_result([_text_delta("final", 1)], "final")
        run_mock = _patch_runner_seq(monkeypatch, [first, second])

        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        run = await agent.run("do it")
        assert await run.task.result() == "final"
        resumed_input = run_mock.call_args_list[1].args[1]
        assert resumed_input == [
            {"role": "user", "content": "do it"},
            {"type": "function_call", "call_id": "c1", "name": "note", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "saved_evidence"},
        ]

    @pytest.mark.fast_sleeps
    async def test_unlimited_multi_turn_recompactions_until_unshrinkable_history_aborts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        base_user = {"role": "user", "content": "u" * 10}
        turn_1 = [base_user, {"type": "reasoning", "summary": [{"text": "a" * 60}]}]
        turn_2 = [base_user, {"role": "assistant", "content": "b" * 70}]
        turn_3 = [base_user, {"type": "reasoning", "summary": [{"text": "c" * 80}]}]
        turn_4_unshrinkable = [
            base_user,
            {"type": "function_call", "call_id": "c1", "name": "fn", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "o" * 100},
            {"type": "reasoning", "summary": [{"text": "d" * 20}]},
        ]
        stubs = [
            _stub_with_items(_overflow_400_formal(), [{"type": "t1"}], turn_1),
            _stub_with_items(_overflow_400_formal(), [{"type": "t2"}], turn_2),
            _stub_with_items(_overflow_400_formal(), [{"type": "t3"}], turn_3),
            _stub_with_items(_overflow_400_formal(), [{"type": "t4"}], turn_4_unshrinkable),
        ]
        run_mock = _patch_runner_seq(monkeypatch, stubs)
        agent = Agent(name="res", system_prompt="s", validator=_accept_any)
        run = await agent.run("u" * 10)
        with pytest.raises(BadRequestError):
            await run.task.result()
        assert run_mock.call_count == 4
        events = await run.read_events()
        assert len([e for e in events if e.type == "thinking"]) == 4

    @pytest.mark.fast_sleeps
    @pytest.mark.parametrize("use_async_filter", [False, True])
    async def test_model_input_filter_guides_compaction_and_empty_filter_reraises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        use_async_filter: bool,
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")

        def _sync_filter(data: CallModelData) -> ModelInputData:
            rewritten = [
                {"type": "reasoning", "summary": [{"text": "short"}]}
                if isinstance(item, dict) and item.get("id") == "already_pruned"
                else item
                for item in data.model_data.input
            ]
            return ModelInputData(input=rewritten, instructions=data.model_data.instructions)

        async def _async_filter(data: CallModelData) -> ModelInputData:
            return _sync_filter(data)

        history = [
            {"role": "user", "content": "u" * 20},
            {"id": "already_pruned", "type": "reasoning", "summary": [{"text": "p" * 400}]},
            {"id": "unpruned_asst", "role": "assistant", "content": "a" * 60},
            {"id": "recent_asst", "role": "assistant", "content": "k" * 10},
        ]
        stubs = [
            _stub_with_items(_overflow_400_formal(), [{"type": "t1"}], history),
            _stub_result([_text_delta("ok", 1)], "ok"),
        ]
        run_mock = _patch_runner_seq(monkeypatch, stubs)
        chosen_filter = _async_filter if use_async_filter else _sync_filter
        agent = Agent(name="res", system_prompt="s", validator=_accept_any, model_input_filter=chosen_filter)
        run = await agent.run("u" * 20)
        assert await run.task.result() == "ok"
        compacted_sent = run_mock.call_args_list[1].args[1]
        assert [item.get("id", item.get("role")) for item in compacted_sent] == ["user", "recent_asst"]

        def _empty_filter(data: CallModelData) -> ModelInputData:
            return ModelInputData(input=[], instructions=data.model_data.instructions)

        empty_agent = Agent(name="res", system_prompt="s", validator=_accept_any, model_input_filter=_empty_filter)
        _patch_runner_seq(monkeypatch, [_stub_with_items(_overflow_400_formal(), [{"type": "t1"}], history)])
        empty_run = await empty_agent.run("u" * 20)
        with pytest.raises(BadRequestError):
            await empty_run.task.result()

    @pytest.mark.fast_sleeps
    @pytest.mark.parametrize("mode", ["classifier_http_error", "first_turn_str_unshrinkable"])
    async def test_classifier_or_compaction_error_reraises_original_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mode: str,
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        if mode == "classifier_http_error":
            _mock_classifier_http(monkeypatch, [500])
            err = _overflow_400_prose(15000)
        else:
            err = _overflow_400_formal()
        _patch_runner_seq(monkeypatch, [_failing_stub(err)])
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        run = await agent.run("unshrinkable first turn prompt")
        with pytest.raises(BadRequestError):
            await run.task.result()

    @pytest.mark.parametrize("cancel_kind", ["loop_in_classify", "loop_in_compact", "task_in_classify"])
    async def test_cancellation_aborts_inflight_classification_or_compaction_without_recancelling_stream(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cancel_kind: str,
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        parked = asyncio.Event()

        async def _slow_filter(_data: CallModelData) -> ModelInputData:
            parked.set()
            await asyncio.sleep(30)
            return ModelInputData(input=[], instructions="sys")

        async def _slow_post(
            _self: httpx.AsyncClient,
            url: str,
            *,
            json: dict,
            headers: dict[str, str],
        ) -> httpx.Response:
            _ = (json, headers)
            parked.set()
            await asyncio.sleep(30)
            return httpx.Response(200, json={}, request=httpx.Request("POST", url))

        if cancel_kind in ("loop_in_classify", "task_in_classify"):
            monkeypatch.setattr(httpx.AsyncClient, "post", _slow_post)
            stub = _failing_stub(_overflow_400_prose(20000))
            agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        else:
            history = [
                {"role": "user", "content": "u" * 10},
                {"type": "reasoning", "summary": [{"text": "r" * 80}]},
            ]
            stub = _stub_with_items(_overflow_400_formal(), [{"type": "t1"}], history)
            agent = Agent(name="res", system_prompt="sys", validator=_accept_any, model_input_filter=_slow_filter)

        _patch_runner_seq(monkeypatch, [stub])
        run = await agent.run("do it")
        await asyncio.wait_for(parked.wait(), timeout=5)
        started = time.monotonic()
        if cancel_kind == "task_in_classify":
            await task_manager.cancel(str(run.task.id))
            assert time.monotonic() - started < _CANCEL_MAX_ELAPSED
            assert run.task.status.value == "cancelled"
        else:
            await _assert_loop_cancelled_fast(run, started)
        assert stub.cancel.call_count == 1


async def _run_wire_agent(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[httpx.Response],
    *,
    validator: Callable[[str, AgentRun], ValidationResult[str | dict]] = _accept_any,
    with_tool: bool = False,
    output_json_schema: dict | None = None,
    model_input_filter: CallModelInputFilter | None = None,
    max_attempts: int = 10,
    max_turns: int | None = None,
    wrap_up_turns_remaining: int | None = None,
) -> tuple[AgentRun, list[dict]]:
    monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
    _CountedTool.calls = 0
    clients: list[httpx.AsyncClient] = []
    wire_bodies: list[dict] = []
    _wire_mock_openai(monkeypatch, responses, clients, wire_bodies=wire_bodies)
    try:
        agent = Agent(
            name="res",
            system_prompt="sys",
            validator=validator,
            max_attempts=max_attempts,
            max_turns=max_turns,
            wrap_up_turns_remaining=wrap_up_turns_remaining,
            output_json_schema=output_json_schema,
            model_input_filter=model_input_filter,
        )
        if with_tool:
            agent.add_tool_adapter(_CountedTool())
        run = await agent.run("do it")
        await run.task.result()
        return run, wire_bodies
    finally:
        await _close_clients(clients)


class TestOutputJsonSchema:
    """The output_json_schema option attaches a strict json_schema response_format to model calls."""

    _SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    _EXPECTED_RESPONSE_FORMAT: ClassVar[dict] = {
        "type": "json_schema",
        "json_schema": {
            "name": "response_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }

    async def _run_wire_bodies(
        self,
        monkeypatch: pytest.MonkeyPatch,
        responses: list[httpx.Response],
        *,
        output_json_schema: dict | None = None,
        validator: Callable[[str, AgentRun], ValidationResult[str | dict]] = _accept_any,
        with_tool: bool = False,
        model_input_filter: CallModelInputFilter | None = None,
    ) -> tuple[AgentRun, list[dict]]:
        return await _run_wire_agent(
            monkeypatch,
            responses,
            validator=validator,
            with_tool=with_tool,
            output_json_schema=output_json_schema,
            model_input_filter=model_input_filter,
        )

    async def test_without_output_json_schema_sends_no_response_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run, wire_bodies = await self._run_wire_bodies(
            monkeypatch, [_sse_text_response("hello")]
        )
        assert await run.task.result() == "hello"
        assert len(wire_bodies) == 1
        assert "response_format" not in wire_bodies[0]

    @pytest.mark.parametrize("with_tool", [False, True], ids=["without_tools", "with_tools"])
    async def test_with_output_json_schema_sends_response_format(
        self, monkeypatch: pytest.MonkeyPatch, with_tool: bool
    ) -> None:
        payload = json.dumps({"answer": "ok"})
        responses = (
            [
                _sse_tool_call_response("call_1", "counted_note", '{"text":"one"}'),
                _sse_text_response(payload),
            ]
            if with_tool
            else [_sse_text_response(payload)]
        )
        run, wire_bodies = await self._run_wire_bodies(
            monkeypatch,
            responses,
            output_json_schema=self._SCHEMA,
            with_tool=with_tool,
        )
        assert await run.task.result() == payload
        assert _CountedTool.calls == (1 if with_tool else 0)
        assert len(wire_bodies) == (2 if with_tool else 1)
        assert all(body["response_format"] == self._EXPECTED_RESPONSE_FORMAT for body in wire_bodies)
        if with_tool:
            assert all("tools" in body for body in wire_bodies)

    async def test_output_json_schema_preserved_with_model_input_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = json.dumps({"answer": "filtered"})
        run, wire_bodies = await self._run_wire_bodies(
            monkeypatch,
            [_sse_text_response(payload)],
            output_json_schema=self._SCHEMA,
            model_input_filter=_rewrite_last_user,
        )
        assert await run.task.result() == payload
        assert len(wire_bodies) == 1
        assert wire_bodies[0]["response_format"] == self._EXPECTED_RESPONSE_FORMAT
        assert wire_bodies[0]["messages"][-1] == {"role": "user", "content": "REWROTE:do it"}

    async def test_output_json_schema_still_runs_validator_and_retries_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen_outputs: list[str] = []

        def validator(text: str, _run: AgentRun) -> ValidationResult[dict]:
            seen_outputs.append(text)
            data = json.loads(text)
            if data["answer"] != "good":
                return ValidationResult(is_valid=False, error="answer must be good")
            return ValidationResult(is_valid=True, value=data)

        bad_payload = json.dumps({"answer": "bad"})
        good_payload = json.dumps({"answer": "good"})
        run, wire_bodies = await self._run_wire_bodies(
            monkeypatch,
            [_sse_text_response(bad_payload), _sse_text_response(good_payload)],
            output_json_schema=self._SCHEMA,
            validator=validator,
        )
        assert await run.task.result() == good_payload
        assert seen_outputs == [bad_payload, good_payload]
        assert len(wire_bodies) == 2
        assert all(body["response_format"] == self._EXPECTED_RESPONSE_FORMAT for body in wire_bodies)
        retry_user = wire_bodies[1]["messages"][-1]
        assert retry_user["role"] == "user"
        assert retry_user["content"] == "answer must be good" + _CORRECTIVE_NOTICE


def _build_sse_multi_tool_call_response(calls: list[tuple[str, str, str]]) -> httpx.Response:
    tool_calls = [
        {
            "index": index,
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
        for index, (call_id, name, arguments) in enumerate(calls)
    ]
    body = (
        _sse_chunk({"role": "assistant", "tool_calls": tool_calls}, None)
        + _sse_chunk({}, "tool_calls")
        + b"data: [DONE]\n\n"
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def _build_sse_text_and_tool_call_response(
    text: str, call_id: str, name: str, arguments: str
) -> httpx.Response:
    tool_call = {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    body = (
        _sse_chunk({"role": "assistant", "content": text}, None)
        + _sse_chunk({"tool_calls": [tool_call]}, None)
        + _sse_chunk({}, "tool_calls")
        + b"data: [DONE]\n\n"
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def _build_http_overflow_400_response() -> httpx.Response:
    return httpx.Response(
        400,
        json={"error": {"code": "context_length_exceeded", "message": "too long"}},
    )


def _spy_run_streamed(monkeypatch: pytest.MonkeyPatch) -> list[int | None]:
    seen_max_turns: list[int | None] = []
    real_run_streamed = engine.Runner.run_streamed

    def _wrapped(
        starting_agent: _SdkAgent,
        input_data: str | list,
        *,
        max_turns: int | None = None,
        run_config: RunConfig | None = None,
    ) -> MagicMock:
        seen_max_turns.append(max_turns)
        if run_config is None:
            return real_run_streamed(starting_agent, input_data, max_turns=max_turns)  # type: ignore[arg-type]
        return real_run_streamed(
            starting_agent, input_data, max_turns=max_turns, run_config=run_config  # type: ignore[arg-type]
        )

    monkeypatch.setattr(engine.Runner, "run_streamed", _wrapped)
    return seen_max_turns


def _build_turn_stub(
    events: list[RawResponsesStreamEvent],
    final_output: str | None,
    *,
    turns: int = 1,
    input_list: list[dict] | None = None,
    error: Exception | None = None,
    new_items: list[dict] | None = None,
) -> MagicMock:
    async def _gen() -> AsyncIterator[RawResponsesStreamEvent]:
        for event in events:
            yield event
        if error is not None:
            raise error

    stub = _build_stub(_gen, final_output)
    stub.raw_responses = [MagicMock() for _ in range(turns)]
    if new_items is not None:
        stub.new_items = list(new_items)
    if input_list is not None:
        stub.to_input_list = MagicMock(return_value=list(input_list))
    return stub


class TestTurnLimitValidation:
    """Validate max_turns and wrap_up_turns_remaining fail fast at construction."""

    @pytest.mark.parametrize(
        ("max_turns", "wrap_up_turns_remaining"),
        [
            (0, None),
            (-1, None),
            (-5, None),
            (True, None),
            (False, None),
            (None, 0),
            (None, 2),
            (None, False),
            (5, -1),
            (5, 5),
            (5, 8),
            (1, 1),
            (5, True),
            (5, False),
        ],
    )
    def test_invalid_turn_limit_configuration_raises_value_error(
        self,
        max_turns: int | None,
        wrap_up_turns_remaining: int | None,
    ) -> None:
        with pytest.raises(ValueError):
            Agent(
                name="res",
                system_prompt="sys",
                validator=_accept_any,
                max_turns=max_turns,
                wrap_up_turns_remaining=wrap_up_turns_remaining,
            )

    @pytest.mark.parametrize(
        ("max_turns", "wrap_up_turns_remaining"),
        [(None, None), (1, None), (1, 0), (5, None), (5, 0), (5, 4)],
    )
    def test_valid_turn_limit_configuration_constructs_agent(
        self,
        max_turns: int | None,
        wrap_up_turns_remaining: int | None,
    ) -> None:
        agent = Agent(
            name="res",
            system_prompt="sys",
            validator=_accept_any,
            max_turns=max_turns,
            wrap_up_turns_remaining=wrap_up_turns_remaining,
        )
        assert isinstance(agent, Agent)


class TestTurnLimitBehavior:
    """Exercise two-stage wrap-up, tool blocking, refusal, and accounting."""

    async def _run_wire(
        self,
        monkeypatch: pytest.MonkeyPatch,
        responses: list[httpx.Response],
        *,
        max_turns: int | None = None,
        wrap_up_turns_remaining: int | None = None,
        max_attempts: int = 3,
        validator: Callable[[str, AgentRun], ValidationResult[str]] = _accept_any,
        with_tool: bool = True,
        output_json_schema: dict | None = None,
    ) -> tuple[AgentRun, list[dict], list[int | None]]:
        seen_max_turns = _spy_run_streamed(monkeypatch)
        run, wire_bodies = await _run_wire_agent(
            monkeypatch,
            responses,
            validator=validator,
            with_tool=with_tool,
            output_json_schema=output_json_schema,
            max_attempts=max_attempts,
            max_turns=max_turns,
            wrap_up_turns_remaining=wrap_up_turns_remaining,
        )
        return run, wire_bodies, seen_max_turns

    @pytest.mark.fast_sleeps
    async def test_unlimited_mode_skips_raw_responses_and_retries_zero_tool_model_behavior(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")

        def _forbid_raw_responses(_self: MagicMock) -> list:
            raise AssertionError("raw_responses must not be read when max_turns is None")

        failing = _failing_stub(ModelBehaviorError("incomplete stream"))
        succeeding = _stub_result([_text_delta("ok", 1)], "ok")
        type(failing).raw_responses = property(_forbid_raw_responses)
        type(succeeding).raw_responses = property(_forbid_raw_responses)
        run_mock = _patch_runner_seq(monkeypatch, [failing, succeeding])

        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        run = await agent.run("do it")
        assert await run.task.result() == "ok"
        assert run_mock.call_count == 2
        assert [call.kwargs["max_turns"] for call in run_mock.call_args_list] == [None, None]
        assert _sleep_delays() == [_FIRST_DELAY]

    @pytest.mark.parametrize("max_turns", [None, 3], ids=["unlimited", "limited_pre_stage_two"])
    async def test_pre_stage_two_malformed_tool_args_return_tool_error_string(
        self, monkeypatch: pytest.MonkeyPatch, max_turns: int | None
    ) -> None:
        responses = [
            _sse_tool_call_response("call_bad", "counted_note", '{"text":'),
            _sse_text_response("recovered"),
        ]
        run, wire_bodies, _seen = await self._run_wire(
            monkeypatch, responses, max_turns=max_turns, with_tool=True
        )
        assert await run.task.result() == "recovered"
        assert _CountedTool.calls == 0
        tool_reply = wire_bodies[1]["messages"][-1]
        assert tool_reply["role"] == "tool"
        assert "An error occurred" in tool_reply["content"]

    async def test_normal_completion_before_stage_one_sends_no_wrap_up_notices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        responses = [
            _sse_tool_call_response("call_1", "counted_note", '{"text":"a"}'),
            _sse_text_response("all done"),
        ]
        run, wire_bodies, seen_max_turns = await self._run_wire(
            monkeypatch, responses, max_turns=5, wrap_up_turns_remaining=2
        )
        assert await run.task.result() == "all done"
        assert _CountedTool.calls == 1
        assert seen_max_turns == [3]
        assert len(wire_bodies) == 2
        system_events = [e for e in await run.read_events() if e.type == "system"]
        assert [e.text for e in system_events] == ["sys"]

    async def test_stage_one_soft_nudge_fires_once_with_exact_count_and_keeps_tools_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        responses = [
            _sse_tool_call_response("call_1", "counted_note", '{"text":"t1"}'),
            _sse_tool_call_response("call_2", "counted_note", '{"text":"t2"}'),
            _sse_tool_call_response("call_3", "counted_note", '{"text":"t3"}'),
            _sse_text_response("wrapped up"),
        ]
        run, wire_bodies, seen_max_turns = await self._run_wire(
            monkeypatch, responses, max_turns=4, wrap_up_turns_remaining=2
        )
        assert await run.task.result() == "wrapped up"
        assert _CountedTool.calls == 3
        assert seen_max_turns == [2, 2]
        soft_notice = wire_bodies[2]["messages"][-1]
        assert soft_notice["role"] == "user"
        assert "2 tool-enabled turns remaining" in soft_notice["content"]
        events = await run.read_events()
        system_events = [e for e in events if e.type == "system"]
        assert len(system_events) == 2
        assert system_events[1].text == soft_notice["content"]
        tool_result_indices = [i for i, e in enumerate(events) if e.type == "tool_result"]
        soft_index = next(i for i, e in enumerate(events) if e.type == "system" and i > 0)
        assert tool_result_indices[1] < soft_index < tool_result_indices[2]

    @pytest.mark.fast_sleeps
    @pytest.mark.parametrize("interruption", ["transient_500", "overflow", "validation_failure"])
    async def test_stage_one_reentry_keeps_remaining_turns_without_duplicate_notice(
        self, monkeypatch: pytest.MonkeyPatch, interruption: str
    ) -> None:
        if interruption == "transient_500":
            responses = [
                _sse_tool_call_response("call_1", "counted_note", '{"text":"t1"}'),
                _sse_tool_call_response("call_2", "counted_note", '{"text":"t2"}'),
                _sse_tool_call_response("call_3", "counted_note", '{"text":"t3"}'),
                _http_500_response(),
                _sse_text_response("good"),
            ]
            validator = _accept_any
            expected_max_turns = [2, 2, 1]
        elif interruption == "overflow":
            responses = [
                _sse_tool_call_response("call_1", "counted_note", '{"text":"t1"}'),
                _sse_tool_call_response("call_2", "counted_note", '{"text":"t2"}'),
                _sse_tool_call_response("call_3", "counted_note", '{"text":"t3"}'),
                _build_http_overflow_400_response(),
                _sse_text_response("good"),
            ]
            validator = _accept_any
            expected_max_turns = [2, 2, 1]
        else:
            responses = [
                _sse_tool_call_response("call_1", "counted_note", '{"text":"t1"}'),
                _sse_tool_call_response("call_2", "counted_note", '{"text":"t2"}'),
                _sse_text_response("bad"),
                _sse_tool_call_response("call_3", "counted_note", '{"text":"t3"}'),
                _sse_text_response("good"),
            ]
            validator = _accept_good
            expected_max_turns = [2, 2, 1, 1]

        if interruption == "overflow":
            responses[2] = _build_sse_text_and_tool_call_response(
                "r" * 2000, "call_3", "counted_note", '{"text":"t3"}'
            )

        run, wire_bodies, seen_max_turns = await self._run_wire(
            monkeypatch,
            responses,
            max_turns=4,
            wrap_up_turns_remaining=2,
            validator=validator,
        )
        assert await run.task.result() == "good"
        assert _CountedTool.calls == 3
        assert seen_max_turns == expected_max_turns
        system_events = [e for e in await run.read_events() if e.type == "system"]
        soft_notices = [e for e in system_events[1:] if "tool-enabled" in e.text]
        assert len(soft_notices) == 1
        final_user_notices = [
            m["content"]
            for m in wire_bodies[-1]["messages"]
            if m.get("role") == "user" and "tool-enabled" in str(m.get("content", ""))
        ]
        assert len(final_user_notices) == 1

    async def test_stage_two_preserves_wire_tools_prompt_and_schema_and_logs_system_notices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        schema: dict = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        }
        payload = json.dumps({"summary": "complete"})
        responses = [
            _sse_tool_call_response("call_1", "counted_note", '{"text":"t1"}'),
            _sse_tool_call_response("call_2", "counted_note", '{"text":"t2"}'),
            _sse_tool_call_response("call_3", "counted_note", '{"text":"t3"}'),
            _sse_text_response(payload),
        ]
        run, wire_bodies, seen_max_turns = await self._run_wire(
            monkeypatch,
            responses,
            max_turns=3,
            wrap_up_turns_remaining=1,
            output_json_schema=schema,
        )
        assert await run.task.result() == payload
        assert _CountedTool.calls == 3
        assert seen_max_turns == [2, 1, 1]
        assert len(wire_bodies) == 4
        assert all(body["tools"] == wire_bodies[0]["tools"] for body in wire_bodies)
        assert all(body["messages"][0] == wire_bodies[0]["messages"][0] for body in wire_bodies)
        assert all(
            body["response_format"] == wire_bodies[0]["response_format"] for body in wire_bodies
        )
        hard_notice = wire_bodies[3]["messages"][-1]
        assert hard_notice["role"] == "user"
        assert "tools are now disabled" in hard_notice["content"]
        system_events = [e for e in await run.read_events() if e.type == "system"]
        assert len(system_events) == 3
        assert "1 tool-enabled turn remaining" in system_events[1].text
        assert "tools are now disabled" in system_events[2].text

    @pytest.mark.parametrize("wrap_up_turns_remaining", [None, 0], ids=["none", "zero"])
    async def test_disabled_stage_one_transitions_directly_to_stage_two(
        self, monkeypatch: pytest.MonkeyPatch, wrap_up_turns_remaining: int | None
    ) -> None:
        responses = [
            _sse_tool_call_response("call_1", "counted_note", '{"text":"t1"}'),
            _sse_tool_call_response("call_2", "counted_note", '{"text":"t2"}'),
            _sse_text_response("final"),
        ]
        run, _wire_bodies, seen_max_turns = await self._run_wire(
            monkeypatch,
            responses,
            max_turns=2,
            wrap_up_turns_remaining=wrap_up_turns_remaining,
        )
        assert await run.task.result() == "final"
        assert _CountedTool.calls == 2
        assert seen_max_turns == [2, 1]
        system_events = [e for e in await run.read_events() if e.type == "system"]
        assert len(system_events) == 2
        assert "tools are now disabled" in system_events[1].text

    @pytest.mark.fast_sleeps
    @pytest.mark.parametrize(
        "blocked_response",
        [
            _sse_tool_call_response("call_b1", "counted_note", '{"text":"blocked"}'),
            _build_sse_multi_tool_call_response(
                [
                    ("call_b1", "counted_note", '{"text":"one"}'),
                    ("call_b2", "counted_note", '{"text":"two"}'),
                ]
            ),
            _sse_tool_call_response("call_b1", "counted_note", '{"text":'),
            _sse_tool_call_response("call_b1", "counted_note", "{}"),
            _sse_tool_call_response("call_b1", "unknown_tool", '{"x":1}'),
            _build_sse_text_and_tool_call_response("partial", "call_b1", "counted_note", '{"text":"mix"}'),
        ],
        ids=[
            "single_valid_call",
            "parallel_batch",
            "malformed_json_args",
            "schema_invalid_args",
            "unknown_tool_name",
            "mixed_text_and_tool",
        ],
    )
    async def test_stage_two_blocks_every_tool_call_shape_and_recovers_cleanly(
        self, monkeypatch: pytest.MonkeyPatch, blocked_response: httpx.Response
    ) -> None:
        responses = [
            _sse_tool_call_response("call_1", "counted_note", '{"text":"allowed"}'),
            blocked_response,
            _sse_text_response("clean final"),
        ]
        run, wire_bodies, seen_max_turns = await self._run_wire(
            monkeypatch, responses, max_turns=1, max_attempts=2
        )
        assert await run.task.result() == "clean final"
        assert _CountedTool.calls == 1
        assert _sleep_delays() == []
        assert seen_max_turns == [1, 1, 1]

        retry_messages = wire_bodies[2]["messages"]
        assert [m for m in retry_messages if m.get("tool_calls")] == [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "counted_note", "arguments": '{"text":"allowed"}'},
                    }
                ],
            }
        ]
        assert "Tools are disabled" in retry_messages[-1]["content"]
        assert _CORRECTIVE_NOTICE in retry_messages[-1]["content"]

        events = await run.read_events()
        assert [e.call_id for e in events if e.type == "tool_call"] == ["call_1"]
        assert [e.call_id for e in events if e.type == "tool_result"] == ["call_1"]
        assert [(e.attempt, e.text) for e in events if e.type == "assistant"] == [(2, "clean final")]

    @pytest.mark.fast_sleeps
    async def test_stage_two_repeated_blocked_tool_calls_fail_after_max_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        _CountedTool.calls = 0
        clients: list[httpx.AsyncClient] = []
        responses = [
            _sse_tool_call_response("call_1", "counted_note", '{"text":"ok"}'),
            _sse_tool_call_response("call_2", "counted_note", '{"text":"blocked1"}'),
            _sse_tool_call_response("call_3", "counted_note", '{"text":"blocked2"}'),
        ]
        _wire_mock_openai(monkeypatch, responses, clients)
        try:
            agent = Agent(
                name="res", system_prompt="sys", validator=_accept_any, max_turns=1, max_attempts=2
            )
            agent.add_tool_adapter(_CountedTool())
            run = await agent.run("do it")
            with pytest.raises(RuntimeError, match="after 2 attempts"):
                await run.task.result()
        finally:
            await _close_clients(clients)
        assert run.task.status.value == "failed"
        assert _CountedTool.calls == 1
        assert _sleep_delays() == []

    async def test_stage_two_max_turns_exceeded_immediately_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        stubs = [
            _build_turn_stub(
                [],
                None,
                turns=1,
                input_list=[{"role": "user", "content": "do it"}],
                error=MaxTurnsExceeded("tool budget hit"),
            ),
            _build_turn_stub([], None, turns=0, error=MaxTurnsExceeded("stage two exceeded")),
        ]
        run_mock = _patch_runner_seq(monkeypatch, stubs)
        agent = Agent(
            name="res", system_prompt="sys", validator=_accept_any, max_turns=1, max_attempts=3
        )
        run = await agent.run("do it")
        with pytest.raises(RuntimeError, match="after turn limit"):
            await run.task.result()
        assert run.task.status.value == "failed"
        assert run_mock.call_count == 2

    @pytest.mark.parametrize("bad_output", [None, "", "   ", "\n\t  "], ids=["none", "empty", "spaces", "whitespace"])
    async def test_missing_or_blank_final_output_rejected_before_validator(
        self, monkeypatch: pytest.MonkeyPatch, bad_output: str | None
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        validator_inputs: list[str] = []

        def validator(text: str, _run: AgentRun) -> ValidationResult[str]:
            validator_inputs.append(text)
            return ValidationResult(is_valid=True, value=text)

        stubs = [
            _build_turn_stub([], bad_output, turns=1),
            _build_turn_stub([_text_delta("real answer", 1)], "real answer", turns=1),
        ]
        _patch_runner_seq(monkeypatch, stubs)
        agent = Agent(name="res", system_prompt="sys", validator=validator, max_turns=2, max_attempts=2)
        run = await agent.run("do it")
        assert await run.task.result() == "real answer"
        assert validator_inputs == ["real answer"]

    async def test_validation_failure_at_turn_boundary_and_in_stage_two_keeps_tools_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        responses = [
            _sse_text_response("bad"),
            _sse_tool_call_response("call_blocked", "counted_note", '{"text":"nope"}'),
            _sse_text_response("still bad"),
            _sse_text_response("good"),
        ]
        run, wire_bodies, seen_max_turns = await self._run_wire(
            monkeypatch, responses, max_turns=1, max_attempts=4, validator=_accept_good
        )
        assert await run.task.result() == "good"
        assert _CountedTool.calls == 0
        assert seen_max_turns == [1, 1, 1, 1]
        system_events = [e for e in await run.read_events() if e.type == "system"]
        assert len(system_events) == 2
        assert "tools are now disabled" in system_events[1].text
        hard_notices = [
            m
            for m in wire_bodies[-1]["messages"]
            if m.get("role") == "user" and "tools are now disabled" in str(m.get("content", ""))
        ]
        assert len(hard_notices) == 1

    @pytest.mark.parametrize(
        "phase",
        ["stage_one_transition", "stage_two_max_turns", "stage_two_model_behavior"],
    )
    async def test_cancel_run_loop_during_wrap_up_exceptions_stops_immediately(
        self, monkeypatch: pytest.MonkeyPatch, phase: str
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        if phase == "stage_one_transition":
            stub, entered, release = _release_in_drain("x", error=MaxTurnsExceeded("hit"))
            stub.raw_responses = [MagicMock()]
            stubs = [stub]
            max_turns = 2
            wrap_up = 1
        elif phase == "stage_two_max_turns":
            first = _build_turn_stub(
                [],
                None,
                turns=1,
                input_list=[{"role": "user", "content": "do it"}],
                error=MaxTurnsExceeded("hit"),
            )
            stub, entered, release = _release_in_drain("x", error=MaxTurnsExceeded("stage two"))
            stub.raw_responses = []
            stubs = [first, stub]
            max_turns = 1
            wrap_up = None
        else:
            first = _build_turn_stub(
                [],
                None,
                turns=1,
                input_list=[{"role": "user", "content": "do it"}],
                error=MaxTurnsExceeded("hit"),
            )
            stub, entered, release = _release_in_drain("x", error=ModelBehaviorError("bad tool"))
            stub.raw_responses = []
            stubs = [first, stub]
            max_turns = 1
            wrap_up = None

        run_mock = _patch_runner_seq(monkeypatch, stubs)
        agent = Agent(
            name="res",
            system_prompt="sys",
            validator=_accept_any,
            max_turns=max_turns,
            wrap_up_turns_remaining=wrap_up,
            max_attempts=3,
        )
        run = await agent.run("do it")
        await _stop_in_drain_and_assert_failed(run, entered, release)
        assert run_mock.call_count == len(stubs)

    @pytest.mark.fast_sleeps
    async def test_stage_two_context_overflow_compacts_preserving_notice_and_keeps_tools_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        responses = [
            _build_sse_text_and_tool_call_response("r" * 2000, "call_1", "counted_note", '{"text":"t1"}'),
            _build_http_overflow_400_response(),
            _sse_tool_call_response("call_blocked", "counted_note", '{"text":"nope"}'),
            _sse_text_response("final after compact"),
        ]
        run, wire_bodies, seen_max_turns = await self._run_wire(
            monkeypatch, responses, max_turns=1, max_attempts=2
        )
        assert await run.task.result() == "final after compact"
        assert _CountedTool.calls == 1
        assert seen_max_turns == [1, 1, 1, 1]
        compacted_messages = wire_bodies[2]["messages"]
        assert any(
            m.get("role") == "user" and "tools are now disabled" in str(m.get("content", ""))
            for m in compacted_messages
        )
        system_events = [e for e in await run.read_events() if e.type == "system"]
        assert len(system_events) == 2

    @pytest.mark.fast_sleeps
    async def test_transient_errors_do_not_consume_turns_or_reset_retries_when_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        responses = [
            _sse_tool_call_response("call_1", "counted_note", '{"text":"t1"}'),
            _http_500_response(),
            _http_500_response(),
            _http_500_response(),
            _sse_tool_call_response("call_2", "counted_note", '{"text":"t2"}'),
            _sse_text_response("final"),
        ]
        run, _wire_bodies, seen_max_turns = await self._run_wire(
            monkeypatch, responses, max_turns=2
        )
        assert await run.task.result() == "final"
        assert _CountedTool.calls == 2
        assert _sleep_delays() == [_FIRST_DELAY, _SECOND_DELAY, _THIRD_DELAY]
        assert seen_max_turns == [2, 1, 1, 1, 1]

    async def test_turn_budget_is_shared_across_validation_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        responses = [
            _sse_tool_call_response("call_1", "counted_note", '{"text":"t1"}'),
            _sse_text_response("bad"),
            _sse_tool_call_response("call_blocked", "counted_note", '{"text":"nope"}'),
            _sse_text_response("good"),
        ]
        run, _wire_bodies, seen_max_turns = await self._run_wire(
            monkeypatch, responses, max_turns=2, max_attempts=3, validator=_accept_good
        )
        assert await run.task.result() == "good"
        assert _CountedTool.calls == 1
        assert seen_max_turns == [2, 1, 1]

    async def test_concurrent_runs_of_one_agent_isolate_stage_two_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        _CountedTool.calls = 0
        run_a_in_stage_two = asyncio.Event()
        run_b_tool_done = asyncio.Event()
        clients: list[httpx.AsyncClient] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            messages = body["messages"]
            prompt = str(messages[1]["content"])
            if prompt == "run A":
                if len(messages) == 2:
                    return _sse_tool_call_response("call_a1", "counted_note", '{"text":"a1"}')
                run_a_in_stage_two.set()
                await asyncio.wait_for(run_b_tool_done.wait(), timeout=5)
                if not any("Tools are disabled" in str(m.get("content", "")) for m in messages):
                    return _sse_tool_call_response("call_a2", "counted_note", '{"text":"blocked"}')
                return _sse_text_response("A done")
            await asyncio.wait_for(run_a_in_stage_two.wait(), timeout=5)
            if len(messages) == 2:
                return _sse_tool_call_response("call_b1", "counted_note", '{"text":"b1"}')
            run_b_tool_done.set()
            return _sse_text_response("B done")

        _install_mock_openai_handler(monkeypatch, _handler, clients)
        try:
            agent = Agent(
                name="res", system_prompt="sys", validator=_accept_any, max_turns=1, max_attempts=2
            )
            agent.add_tool_adapter(_CountedTool())
            run_a, run_b = await asyncio.gather(agent.run("run A"), agent.run("run B"))
            result_a, result_b = await asyncio.gather(run_a.task.result(), run_b.task.result())
        finally:
            await _close_clients(clients)
        assert result_a == "A done"
        assert result_b == "B done"
        assert _CountedTool.calls == 2


@pytest.mark.live
@pytest.mark.skipif(not is_llm_available(), reason="thinking backend not reachable")
class TestAgentLive:
    async def test_real_loop_with_tools_and_cancellation(self) -> None:
        assert os.environ.get("REASON_THINKING_BACKEND_URL"), "REASON_THINKING_BACKEND_URL must be set for the live test"

        def validator(text: str, _run: AgentRun) -> ValidationResult[str]:
            if text:
                return ValidationResult(is_valid=True, value=text)
            return ValidationResult(is_valid=False, error="empty")

        agent = Agent(
            name="live",
            system_prompt="Answer the user's request concisely. Use the note tool once.",
            validator=validator,
            backend=config.backend(config.REASON_THINKING),
            max_attempts=2,
        )
        agent.add_tool_adapter(_LiveBlock())
        run = await agent.run("Please call the note tool with text 'hi' and then answer 'done'.")
        result = await run.task.result()
        assert run.task.status.value == "completed"
        assert isinstance(result, str) and result

        calls = await run.read_tool_calls()
        assert any(c.tool_name == "note" for c in calls)

        conv_path = Path(store_module.task_store.data_dir) / "runs" / str(run.task.id) / "conversation.jsonl"
        assert conv_path.exists()
        lines = [json.loads(l) for l in conv_path.read_text().splitlines() if l.strip()]
        assert lines and all("type" in l and "attempt" in l for l in lines)

        # cancellation: start another run and cancel it while it is running
        agent2 = Agent(name="live2", system_prompt="s", validator=validator, backend=config.backend(config.REASON_THINKING))
        run2 = await agent2.run("count slowly to 1000.")
        await run2.task.wait_started()
        await task_manager.cancel(str(run2.task.id))
        assert run2.task.status.value == "cancelled"

    async def test_output_json_schema_with_tool_and_validator(self) -> None:
        assert os.environ.get("REASON_THINKING_BACKEND_URL"), "REASON_THINKING_BACKEND_URL must be set for the live test"
        schema: dict = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["done", "failed"]},
                "note": {"type": "string"},
            },
            "required": ["status", "note"],
            "additionalProperties": False,
        }

        async def validator(text: str, run: AgentRun) -> ValidationResult[dict]:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                return ValidationResult(is_valid=False, error=f"invalid json: {exc}")
            if not isinstance(data, dict):
                return ValidationResult(is_valid=False, error="output must be a json object")
            if data.get("status") not in ("done", "failed"):
                return ValidationResult(is_valid=False, error="invalid status")
            if not isinstance(data.get("note"), str) or not data["note"].strip():
                return ValidationResult(is_valid=False, error="note must be non-empty")
            calls = await run.read_tool_calls()
            if not any(call.tool_name == "note" for call in calls):
                return ValidationResult(is_valid=False, error="must call note tool first")
            return ValidationResult(is_valid=True, value=data)

        agent = Agent(
            name="live_schema",
            system_prompt=(
                "Call the note tool once, then answer with a JSON object having status 'done' "
                "and note containing the noted text."
            ),
            validator=validator,
            backend=config.backend(config.REASON_THINKING),
            max_attempts=3,
            output_json_schema=schema,
        )
        agent.add_tool_adapter(_LiveBlock())
        run = await agent.run("Call the note tool with text 'hello' and return the JSON result.")
        raw = await run.task.result()
        assert run.task.status.value == "completed"
        parsed = json.loads(raw)
        assert parsed["status"] == "done"
        assert isinstance(parsed["note"], str) and parsed["note"].strip()
        assert set(parsed.keys()) == {"status", "note"}
        calls = await run.read_tool_calls()
        assert any(call.tool_name == "note" for call in calls)
