import asyncio
import base64
import inspect
import json
import os
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from io import IOBase
from pathlib import Path
from typing import ClassVar, override
from unittest.mock import MagicMock

import pytest
import aiofiles.threadpool
from agents import Agent as _SdkAgent
from agents import OpenAIChatCompletionsModel
from agents import ToolOutputFileContent, ToolOutputImage, ToolOutputText, Usage
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import ItemHelpers, TResponseInputItem, TResponseStreamEvent, ToolCallItem, ToolCallOutputItem
from agents.model_settings import ModelSettings
from agents.models import chatcmpl_converter, openai_chatcompletions
from agents.models.interface import ModelTracing
from agents.run_config import CallModelData, CallModelInputFilter, ModelInputData, RunConfig
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from agents.tool import Tool as _SdkTool
from agents.tool_context import ToolContext
from agents.usage import InputTokensDetails, OutputTokensDetails
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

import app.lib.agent.engine as engine
import app.lib.db.store as store_module
from app.lib.agent.engine import Agent, AgentRun
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


def _release_in_drain(final_output: str) -> tuple[MagicMock, asyncio.Event, asyncio.Event]:
    """A stub whose drain stalls until released, and the events for entry and release."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _gen() -> AsyncIterator[RawResponsesStreamEvent]:
        yield _text_delta(final_output, 1)
        entered.set()
        await release.wait()

    return _build_stub(_gen, final_output), entered, release


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


def _accept_any(text: str, _run: AgentRun) -> tuple[bool, str]:
    return (True, text) if text else (False, "empty output")


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
        events = [e async for e in run.conversation()]
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

        def validator(text: str, _run: AgentRun) -> tuple[bool, str]:
            return (True, text) if text.endswith("42") else (False, "give me 42")

        agent = Agent(name="res", system_prompt="sys", validator=validator)
        run_mock = _patch_runner_seq(monkeypatch, [stub1, stub2])
        run = await agent.run("do it")
        assert await run.task.result() == "Answer: 42"
        events = [e async for e in run.conversation()]
        corrective = [e for e in events if e.type == "user" and e.text.startswith("give me 42")]
        assert len(corrective) == 1
        assert "IMPORTANT NOTE" in corrective[0].text
        assert run_mock.call_count == 2

    async def test_validator_exhaustion_fails_with_last_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")

        def validator(_text: str, _run: AgentRun) -> tuple[bool, str]:
            return (False, "still wrong")

        agent = Agent(name="res", system_prompt="sys", validator=validator, max_attempts=3)
        _patch_runner(monkeypatch, _stub_result([_text_delta("no", 1)], "no"))
        run = await agent.run("do it")
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            await run.task.result()
        assert run.task.status.value == "failed"

    async def test_validator_receives_the_run_handle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        seen: list[AgentRun] = []

        def validator(text: str, agent_run: AgentRun) -> tuple[bool, str]:
            seen.append(agent_run)
            return True, text

        agent = Agent(name="res", system_prompt="sys", validator=validator)
        _patch_runner(monkeypatch, _stub_result([_text_delta("ok", 1)], "ok"))
        run = await agent.run("do it")
        await run.task.result()
        assert seen
        assert all(isinstance(r, AgentRun) for r in seen)
        assert all(r.task is run.task for r in seen)

    async def test_async_validator_is_awaited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")

        async def validator(text: str, _run: AgentRun) -> tuple[bool, str]:
            return True, text

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
        view1 = await store_module.task_view(str(run1.task.id))
        view2 = await store_module.task_view(str(run2.task.id))
        assert view1 is not None and view1.agent_name == "res"
        assert view2 is not None and view2.agent_name == "res"

    async def test_agent_is_pure_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        assert "start" not in dir(agent)
        assert "conversation" not in dir(agent)
        assert "tool_calls" not in dir(agent)
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

        def validator(_text: str, _run: AgentRun) -> tuple[bool, dict]:
            return (True, value)

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
        view = await store_module.task_view(str(run.task.id))
        assert view is not None and view.agent_name == "res"
        assert view.tool_name is None
        assert not (self._run_path(run.task.id) / "meta").exists()

    async def test_conversation_missing_file_yields_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _agent, run = await self._run(monkeypatch, [_text_delta("ok", 1)], "ok")
        await run.task.result()
        conv = self._run_path(run.task.id) / "conversation.jsonl"
        conv.unlink()
        events = [e async for e in run.conversation()]
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
        events1 = [e async for e in run1.conversation()]
        events2 = [e async for e in run2.conversation()]
        assert events1[0].type == "system" and events1[1].type == "user" and events1[1].text == "first prompt"
        assert events2[0].type == "system" and events2[1].type == "user" and events2[1].text == "second prompt"
        assert all(e.text != "second prompt" for e in events1)
        assert all(e.text != "first prompt" for e in events2)
        view1 = await store_module.task_view(str(run1.task.id))
        view2 = await store_module.task_view(str(run2.task.id))
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
        events1 = [e async for e in run1.conversation()]
        events2 = [e async for e in run2.conversation()]
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

    script: ClassVar[list[str]] = []
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
        message = _message(_ScriptedModel.script.pop(0))
        yield ResponseOutputItemDoneEvent(
            type="response.output_item.done", item=message, item_id=message.id,
            output_index=0, sequence_number=1,
        )
        yield ResponseCompletedEvent(
            type="response.completed", response=_completed_response(message), sequence_number=2,
        )


def _message(text: str) -> ResponseOutputMessage:
    """One assistant message holding the text."""
    return ResponseOutputMessage(
        id="msg_1", role="assistant", status="completed", type="message",
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
        tool_calls=[],
    )


def _completed_response(message: ResponseOutputMessage) -> Response:
    """A terminal response wrapping one message."""
    usage = ResponseUsage(
        input_tokens=1,
        input_tokens_details=response_usage.InputTokensDetails(cache_write_tokens=0, cached_tokens=0),
        output_tokens=1,
        output_tokens_details=response_usage.OutputTokensDetails(reasoning_tokens=0),
        total_tokens=2,
    )
    return Response(
        id="resp_1", created_at=1, model="default", object="response", output=[message],
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

        def validator(text: str, _run: AgentRun) -> tuple[bool, str]:
            return (True, text) if text.endswith("42") else (False, "give me 42")

        agent = Agent(name="res", system_prompt="sys", validator=validator,
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

        def validator(text: str, _run: AgentRun) -> tuple[bool, str]:
            return (True, text) if text.endswith("42") else (False, "give me 42")

        agent = Agent(name="res", system_prompt="sys", validator=validator)
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

        def validator(_text: str, _run: AgentRun) -> tuple[bool, str]:
            return (False, "give me 42")

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

        def validator(_text: str, run: AgentRun) -> tuple[bool, str]:
            run.cancel_run_loop()
            return (True, _text)

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

        def validator(_text: str, run: AgentRun) -> tuple[bool, str]:
            run.cancel_run_loop()
            return (False, "give me 42")

        agent = Agent(name="res", system_prompt="sys", validator=validator)
        monkeypatch.setattr(engine, "OpenAIChatCompletionsModel", _ScriptedModel)
        run = await agent.run("do it")
        with pytest.raises(RuntimeError, match="stopped on request"):
            await run.task.result()
        assert run.task.status.value == "failed"
        assert _ScriptedModel.received == [[{"role": "user", "content": "do it"}]]
        assert [e.text for e in await run.events() if e.type == "user"] == ["do it"]

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

        def validator(_text: str, _run: AgentRun) -> tuple[bool, str]:
            return (False, "nope")

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

        def validator(text: str, _run: AgentRun) -> tuple[bool, str]:
            return (True, text) if text.endswith("42") else (False, "give me 42")

        agent = Agent(name="res", system_prompt="sys", validator=validator)
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
    tracked: ClassVar[bool] = True
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
    tracked: ClassVar[bool] = True
    params: ClassVar[tuple[Param, ...]] = (Param(name="prompt", type=str, description="The prompt."),)

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text=f"generated {params['prompt']}")


class _BigTool(Tool):
    name: ClassVar[str] = "big"
    description: ClassVar[str] = "Return a long string."
    tracked: ClassVar[bool] = True
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
    tracked: ClassVar[bool] = True
    params: ClassVar[tuple[Param, ...]] = (Param(name="query", type=str, description="The query."),)

    @override
    async def call(self, _params: dict) -> ToolResult:
        raise RuntimeError("bad")


class _StopTool(Tool):
    name: ClassVar[str] = "stopper"
    description: ClassVar[str] = "Stop while running."
    tracked: ClassVar[bool] = True
    params: ClassVar[tuple[Param, ...]] = (Param(name="query", type=str, description="The query."),)

    @override
    async def call(self, _params: dict) -> ToolResult:
        raise asyncio.CancelledError


class _BoundedTool(Tool):
    name: ClassVar[str] = "bounded"
    description: ClassVar[str] = "Take a small number."
    tracked: ClassVar[bool] = True
    params: ClassVar[tuple[Param, ...]] = (
        Param(name="count", type=int, description="How many.", min=1, max=10),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text=f"count {params['count']}")


class _ToolTierRejectTool(Tool):
    name: ClassVar[str] = "tool_rejector"
    description: ClassVar[str] = "Reject a large number before it runs."
    tracked: ClassVar[bool] = True
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
    tracked: ClassVar[bool] = True
    staleness: ClassVar[timedelta | None] = timedelta(hours=24)
    params: ClassVar[tuple[Param, ...]] = (Param(name="query", type=str, description="The query."),)

    @override
    async def cache_key(self, params: dict) -> str | None:
        return str(params["query"])

    @override
    async def call(self, params: dict) -> ToolResult:
        return ToolResult(text=f"results for {params['query']}")


class _MediaTool(Tool):
    name: ClassVar[str] = "media"
    description: ClassVar[str] = "Return text plus the files the test names."
    tracked: ClassVar[bool] = True
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
        calls = await run.tool_calls()
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

        calls = await run.tool_calls()
        assert len(calls) == 1
        call = calls[0]
        assert call.tool_name == tool_name
        assert call.params == {"query": "cats"}
        assert call.status == "completed"
        assert call.task_id and call.run_dir

        events = [e async for e in run.conversation()]
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
        task_id = (await run.tool_calls())[0].task_id
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
        calls = await run.tool_calls()
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
        calls = await run.tool_calls()
        assert out == (
            f"Task completed. task_id={calls[0].task_id}. Retrieve the result with the task tool: "
            f"task(action='result', task_id='{calls[0].task_id}')."
        )


class TestPreservingConverter:
    """The engine installs a converter that keeps media tool output on the wire."""

    async def _drive_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One stubbed agent run, which builds the model and installs the converter swap."""
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sys", validator=_accept_any)
        _patch_runner(monkeypatch, _stub_result([_text_delta("ok", 1)], "ok"))
        run = await agent.run("do it")
        await run.task.result()

    def _tool_content(self, outputs: list) -> list:
        """The wire content of one tool output, built from SDK output types."""
        call = ResponseFunctionToolCall(type="function_call", call_id="c1", name="f", arguments="{}")
        items = [
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            ItemHelpers.tool_call_output_item(call, outputs),
        ]
        messages = openai_chatcompletions.Converter.items_to_messages(items, model="default")
        tools = [m for m in messages if m.get("role") == "tool"]
        assert len(tools) == 1
        return tools[0]["content"]

    async def test_a_run_installs_the_converter_swap_idempotently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await self._drive_run(monkeypatch)
        assert openai_chatcompletions.Converter is not chatcmpl_converter.Converter
        await self._drive_run(monkeypatch)
        assert openai_chatcompletions.Converter is not chatcmpl_converter.Converter

    async def test_image_part_reaches_the_wire(self, monkeypatch: pytest.MonkeyPatch) -> None:
        await self._drive_run(monkeypatch)
        content = self._tool_content([
            ToolOutputText(text="t"),
            ToolOutputImage(image_url="data:image/png;base64,AAA"),
        ])
        assert content[0] == {"type": "text", "text": "t"}
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAA", "detail": "auto"},
        }

    async def test_video_file_part_becomes_a_video_url_part(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await self._drive_run(monkeypatch)
        content = self._tool_content([
            ToolOutputText(text="t"),
            ToolOutputFileContent(file_data="data:video/mp4;base64,BBB", filename="v.mp4"),
        ])
        assert content[1] == {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,BBB"}}
        assert all(part.get("type") != "file" for part in content)

    async def test_non_video_file_part_is_dropped_and_text_survives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await self._drive_run(monkeypatch)
        content = self._tool_content([
            ToolOutputText(text="t"),
            ToolOutputFileContent(file_data="data:application/pdf;base64,CCC"),
        ])
        assert content == [{"type": "text", "text": "t"}]


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
        tracked: ClassVar[bool] = False
        description: ClassVar[str] = "Echo without tracking."
        summary: ClassVar[str] = "test"
        params: ClassVar[tuple[Param, ...]] = (Param("query", str, description="The query."),)

        @override
        async def call(self, params: dict) -> ToolResult:
            return ToolResult(text=f"untracked {params['query']}")

    async def test_tracked_tool_hands_back_the_task_id_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, run = await self._call_tool(monkeypatch, _BlockTool(), '{"query":"cats"}', by_id=True)
        calls = await run.tool_calls()
        assert out == _id_message(calls[0].task_id)
        assert "results for cats" not in out

    async def test_result_stays_on_disk_under_the_reported_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _out, run = await self._call_tool(monkeypatch, _BlockTool(), '{"query":"cats"}', by_id=True)
        call = (await run.tool_calls())[0]
        result_file = Path(store_module.tool_store.data_dir) / call.run_dir / "result"
        assert result_file.read_text() == "results for cats"
        record = await store_module.tool_store.get(call.task_id)
        assert record.tool_name == "search_web"
        assert record.params == {"query": "cats"}

    async def test_tool_call_still_reaches_the_conversation_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _out, run = await self._call_tool(monkeypatch, _BlockTool(), '{"query":"cats"}', by_id=True)
        call = (await run.tool_calls())[0]
        results = [e async for e in run.conversation() if e.type == "tool_result"]
        assert len(results) == 1
        assert results[0].task_id == call.task_id
        assert results[0].tool_name == "search_web"

    async def test_flag_off_still_hands_back_the_real_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, run = await self._call_tool(monkeypatch, _BlockTool(), '{"query":"cats"}', by_id=False)
        calls = await run.tool_calls()
        assert out == "results for cats"
        assert calls[0].task_id
        results = [e async for e in run.conversation() if e.type == "tool_result"]
        assert results[0].task_id == calls[0].task_id

    async def test_async_tool_hands_back_the_task_id_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, run = await self._call_tool(monkeypatch, _AsyncTool(), '{"prompt":"a cat"}', by_id=True)
        calls = await run.tool_calls()
        assert out == _id_message(calls[0].task_id)
        assert "Task started" not in out

    async def test_truncating_tool_hands_back_the_id_message_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, run = await self._call_tool(monkeypatch, _BigTool(), '{"query":"q"}',
                                          by_id=True, with_text=False)
        task_id = (await run.tool_calls())[0].task_id
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
        calls = await run.tool_calls()
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
        calls = await run.tool_calls()
        assert out == _id_message(calls[0].task_id)
        assert "results for cats" not in out

    async def test_tool_opt_in_wins_over_the_files_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, run = await self._call_tool(monkeypatch, _IdOnlyMediaTool(), '{"query":"cats"}',
                                         by_id=False, results_by_id=True)
        calls = await run.tool_calls()
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
        calls = await run.tool_calls()
        if id_only:
            assert out == _id_message(calls[0].task_id)
            assert "results for cats" not in out
        else:
            assert out == "results for cats"


class TestUntrackedToolInLoop:
    """An untracked tool runs in-loop without a guard and leaves no call report."""

    class _UntrackedEchoTool(Tool):
        name: ClassVar[str] = "fake_untracked"
        tracked: ClassVar[bool] = False
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
        assert await run.tool_calls() == []


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
        events = [e async for e in run.conversation()]
        assert [e.type for e in events] == ["system", "user", "content", "assistant"]
        assert events[1].text == "do it" and events[-1].text == "hello"

    @pytest.mark.timeout(30)
    async def test_agentrun_events_reads_the_run_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="sp", validator=_accept_any)
        _patch_runner(monkeypatch, _stub_result([_text_delta("hello", 1)], "hello"))
        run = await agent.run("do it")
        await run.task.result()
        events = await run.events()
        assert [e.type for e in events] == ["system", "user", "content", "assistant"]
        assert events[1].text == "do it" and events[-1].text == "hello"

    @pytest.mark.timeout(30)
    async def test_validator_reads_events_mid_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")

        async def validator(text: str, agent_run: AgentRun) -> tuple[bool, str]:
            events = await agent_run.events()
            if any(e.type == "assistant" and e.text == text for e in events):
                return True, text
            return False, "assistant event not in the conversation yet"

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
        calls = await run.tool_calls()
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
        return [e.status for e in await run.events() if e.type == "tool_result"]

    async def test_completed_call_is_recorded_as_completed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await self._drive(monkeypatch, _BlockTool(), '{"query":"cats"}')
        assert await self._result_statuses(run) == ["completed"]
        assert [c.status for c in await run.tool_calls()] == ["completed"]

    async def test_failed_call_is_recorded_as_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await self._drive(monkeypatch, _FailTool(), '{"query":"cats"}')
        assert await self._result_statuses(run) == ["failed"]
        assert [c.status for c in await run.tool_calls()] == ["failed"]

    async def test_cancelled_call_is_recorded_as_cancelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await self._drive(monkeypatch, _StopTool(), '{"query":"cats"}')
        assert await self._result_statuses(run) == ["cancelled"]
        assert [c.status for c in await run.tool_calls()] == ["cancelled"]

    async def test_rejected_params_are_recorded_without_a_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await self._drive(monkeypatch, _BoundedTool(), '{"count":99}')
        assert await self._result_statuses(run) == ["rejected"]
        assert await run.tool_calls() == []

    async def test_tool_tier_rejection_is_recorded_as_a_failed_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rejection the tool makes in its own call is a failed run with a call report."""
        run = await self._drive(monkeypatch, _ToolTierRejectTool(), '{"count":99}')
        assert await self._result_statuses(run) == ["failed"]
        assert [c.status for c in await run.tool_calls()] == ["failed"]

    async def test_run_dir_points_at_the_call_s_run_folder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await self._drive(monkeypatch, _BlockTool(), '{"query":"cats"}')
        call = (await run.tool_calls())[0]
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
        events = await run.events()
        assert [e.call_id for e in events if e.type == "tool_call"] == ["C0"]
        results = [e for e in events if e.type == "tool_result"]
        assert [(e.call_id, e.status) for e in results] == [("C1", "completed")]
        assert await run.tool_calls() == []

    async def test_every_attempt_keeps_its_own_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        checked: list[str] = []

        def _accept_second(text: str, _run: AgentRun) -> tuple[bool, str]:
            checked.append(text)
            return (True, text) if len(checked) > 1 else (False, "try again")

        agent = Agent(name="res", system_prompt="sys",
                      validator=_accept_second, max_attempts=2)
        agent.add_tool_adapter(_BlockTool())
        _patch_runner_factory(monkeypatch, partial(_make_tool_stub, tool_name="search_web",
                                                   args_json='{"query":"cats"}'))
        run = await agent.run("do it")
        await run.task.result()
        assert [e.attempt for e in await run.events() if e.type == "tool_result"] == [1, 2]
        calls = await run.tool_calls()
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
        calls = await (await self._drive(monkeypatch, 2)).tool_calls()
        assert [c.tool_name for c in calls] == ["cached_search", "cached_search"]
        assert [c.status for c in calls] == ["completed", "completed"]
        assert [c.params for c in calls] == [{"query": "cats"}, {"query": "cats"}]
        assert len({c.task_id for c in calls}) == 1
        record = await store_module.tool_store.get(calls[0].task_id)
        assert record.tool_name == "cached_search"

    async def test_reused_run_dir_holds_readable_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = await (await self._drive(monkeypatch, 2)).tool_calls()
        stored_dir = f"runs/{calls[0].task_id}"
        assert [c.run_dir for c in calls] == [stored_dir, stored_dir]
        assert await self._run_text(calls[1].run_dir) == "results for cats"

    async def test_reuse_is_marked_in_the_conversation_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await self._drive(monkeypatch, 2)
        results = [e for e in await run.events() if e.type == "tool_result"]
        assert [e.status for e in results] == ["completed", "completed"]
        assert len({e.task_id for e in results}) == 1

    async def test_only_the_first_call_makes_a_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await self._drive(monkeypatch, 2)
        calls = await run.tool_calls()
        children = await store_module.task_store.list_subtasks(str(run.task.id))
        assert [t.task_id for t in children] == [calls[0].task_id]

    async def test_hit_on_an_earlier_run_borrows_its_task_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stored = await self._drive(monkeypatch, 1)
        reusing = await self._drive(monkeypatch, 1)
        stored_id = (await stored.tool_calls())[0].task_id
        borrowed = (await reusing.tool_calls())[0]
        assert borrowed.task_id == stored_id
        assert borrowed.run_dir == f"runs/{stored_id}"
        assert await self._run_text(borrowed.run_dir) == "results for cats"

    async def test_hit_on_another_run_is_marked_and_makes_no_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        await self._drive(monkeypatch, 1)
        reusing = await self._drive(monkeypatch, 1)
        results = [e for e in await reusing.events() if e.type == "tool_result"]
        assert [e.status for e in results] == ["completed"]
        assert await store_module.task_store.list_subtasks(str(reusing.task.id)) == []


class TestLegacyEventLine:
    """An old conversation.jsonl line that carries a cached key still parses."""

    async def test_line_with_legacy_cached_key_parses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = await _completed_run(monkeypatch)
        path = Path(store_module.task_store.data_dir, "runs", str(run.task.id), "conversation.jsonl")
        legacy = json.dumps({
            "type": "tool_result",
            "attempt": 1,
            "ts": "2026-01-01T00:00:00+00:00",
            "text": "",
            "tool_name": "cached_search",
            "params": {"query": "cats"},
            "task_id": "task-legacy",
            "call_id": "call-legacy",
            "status": "completed",
            "cached": True,
        })
        with path.open("a", encoding="utf-8") as fh:
            fh.write(legacy + "\n")
        last = (await run.events())[-1]
        assert last.type == "tool_result"
        assert last.status == "completed"
        assert last.task_id == "task-legacy"


class TestSystemLine:
    """The system prompt is the first line of the conversation file."""

    async def test_system_line_is_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        agent = Agent(name="res", system_prompt="You are a helpful agent.", validator=_accept_any)
        _patch_runner(monkeypatch, _stub_result([_text_delta("hi", 1)], "hi"))
        run = await agent.run("do it")
        await run.task.result()
        events = [e async for e in run.conversation()]
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
        conv = [e async for e in run.conversation()]
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
        content_lines = [e async for e in run.conversation() if e.type == "content"]
        assert len(content_lines) == 5
        assert "".join(e.text for e in content_lines) == final

    async def test_small_deltas_each_get_own_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        _patch_runner(monkeypatch, _stub_result(
            [_text_delta("a", 1), _text_delta("b", 2), _text_delta("c", 3)], "abc"))
        agent = Agent(name="res", system_prompt="sp", validator=_accept_any)
        run = await agent.run("do it")
        await run.task.result()
        content_lines = [e async for e in run.conversation() if e.type == "content"]
        assert [e.text for e in content_lines] == ["a", "b", "c"]

    async def test_small_content_single_flush(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        _patch_runner(monkeypatch, _stub_result([_text_delta("short", 1)], "short"))
        agent = Agent(name="res", system_prompt="sp", validator=_accept_any)
        run = await agent.run("do it")
        await run.task.result()
        content_lines = [e async for e in run.conversation() if e.type == "content"]
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


class TestTailLines:
    """The shared file-tail helper yields lines and stops on signal."""

    async def test_yields_complete_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "tail.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n')
        lines = [line async for line in engine.tail_lines(str(path), _stop)]
        assert lines == ['{"a": 1}', '{"b": 2}']

    async def test_stops_when_signal_fires(self, tmp_path: Path) -> None:
        path = tmp_path / "tail.jsonl"
        path.write_text('{"a": 1}\n')
        lines = [line async for line in engine.tail_lines(str(path), _stop)]
        assert lines == ['{"a": 1}']

    async def test_missing_file_yields_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "nope.jsonl"
        lines = [line async for line in engine.tail_lines(str(path), _stop)]
        assert lines == []

    async def test_default_poll_keeps_streaming_lag_low(self) -> None:
        poll = inspect.signature(engine.tail_lines).parameters["poll"].default
        assert poll <= 0.05


async def _stop() -> bool:
    return True


class _LiveBlock(Tool):
    name: ClassVar[str] = "note"
    description: ClassVar[str] = "Return a short note."
    tracked: ClassVar[bool] = True
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


@pytest.mark.live
@pytest.mark.skipif(not is_llm_available(), reason="thinking backend not reachable")
class TestAgentLive:
    async def test_real_loop_with_tools_and_cancellation(self) -> None:
        assert os.environ.get("REASON_THINKING_BACKEND_URL"), "REASON_THINKING_BACKEND_URL must be set for the live test"

        def validator(text: str, _run: AgentRun) -> tuple[bool, str]:
            return (True, text) if text else (False, "empty")

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

        calls = await run.tool_calls()
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
