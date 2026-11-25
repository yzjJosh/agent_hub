"""Run tracked, cancellable agent loops with structured conversation logs."""

from __future__ import annotations

import asyncio
from asyncio import CancelledError
import inspect
import json
import sys
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Protocol

from agents import Agent as _SdkAgent
from agents import (
    AgentsException,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    ToolOutputFileContent,
    ToolOutputImage,
    ToolOutputText,
    default_tool_error_function,
    function_tool,
    set_tracing_disabled,
)
from agents.run import RunResultStreaming
from agents.run_config import CallModelData, CallModelInputFilter, ModelInputData, RunConfig
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from agents.tool_context import ToolContext
import httpx
from openai import APIError, APIStatusError, AsyncOpenAI
from openai.types.responses import (
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseTextDeltaEvent,
)

from app.lib.agent import context
from app.lib.agent.conversation_log import (
    ConversationEvent,
    ConversationLogWriter,
    ToolCallRecord,
    assemble_tool_calls,
    build_conversation_event,
    get_conversation_log_path,
    parse_conversation_event,
    read_conversation_events,
    tail_lines,
)
from app.lib.agent.media_attachments import load_media_attachment_parts
from app.lib.agent.wire_converter import install_wire_converter, should_always_replay_reasoning
from app.lib.db.store import agent_store
from app.lib.shared import config
from app.lib.task.task_manager import AsyncTask, ImmediateTask, Task, TaskStatus, task_manager
from app.lib.tool.base import Param, ParamValue, Tool, ToolResult, build_param_signature
from app.lib.tool.truncation import (
    DEFAULT_TOOL_OUTPUT_LEN_LIMIT,
    MAX_CHAR_DESCRIPTION,
    MIN_TOOL_OUTPUT_LEN_LIMIT,
    truncate,
)

if TYPE_CHECKING:
    from agents import FunctionTool
    from agents.items import RunItem

__all__ = ["Agent", "AgentRun", "ValidationResult"]

_DEFAULT_MAX_ATTEMPTS = 10
_MIN_MAX_TURNS = 1
_HARD_WRAP_UP_MAX_TURNS = 1
_MAX_SERVER_RETRIES = 10
_RETRY_INITIAL_DELAY = 1.0
_RETRY_BACKOFF_FACTOR = 2
_INTERNAL_ERROR_STATUS = 500
_RETRYABLE_STATUS_CODES = (408, 409, 429)
_RESPONSE_SCHEMA_NAME = "response_schema"
_REJECTED_STATUS = "rejected"
_SHORT_TASK_ID_LENGTH = 8
_RETRY_NOTICE = (
    "IMPORTANT NOTE: Your last response is dropped. Please correct the error and re-generate your "
    "entire response in your next message. Please don't put any filler content. Your message must "
    "contain ONLY the final output."
)
_HARD_WRAP_UP_NOTICE = (
    "IMPORTANT NOTICE: Your tool-turn limit has been reached and tools are now disabled. "
    "Do not call any tools. Output ONLY your final response now in this turn."
)
_TOOLS_DISABLED_ERROR_MESSAGE = "Tools are disabled because the turn limit was reached."
_TOOLS_DISABLED_FEEDBACK = (
    "Error: Tools are disabled because the turn limit was reached. "
    "Do not call any tools. Output ONLY your final response."
)

_AGENT_MAX_CHAR_PARAM = Param(
    "max_char",
    int,
    DEFAULT_TOOL_OUTPUT_LEN_LIMIT,
    min=MIN_TOOL_OUTPUT_LEN_LIMIT,
    description=MAX_CHAR_DESCRIPTION,
)


@dataclass(frozen=True)
class ValidationResult[T]:
    """Result returned by an agent output validator."""

    is_valid: bool
    value: T | None = None
    error: str = ""


@dataclass(frozen=True)
class _ToolCallOutcome:
    task_id: str | None
    status: str | None


_EMPTY_TOOL_CALL_OUTCOME = _ToolCallOutcome(task_id=None, status=None)


@dataclass(frozen=True)
class _RecoveryStep:
    next_input: str | list
    next_retry_count: int


@dataclass(frozen=True)
class _StreamStep:
    next_input: str | list
    max_turns: int | None


@dataclass(frozen=True)
class _AttemptResult:
    final_output: str | None
    history_items: list
    blocked_error: str | None


@dataclass
class _TurnBudget:
    max_turns: int | None
    soft_threshold: int = 0
    turns_used: int = 0
    is_soft_nudge_sent: bool = False
    is_hard_wrap_up_sent: bool = False

    @property
    def is_tools_disabled(self) -> bool:
        return self.max_turns is not None and self.turns_used >= self.max_turns

    @property
    def remaining(self) -> int | None:
        if self.max_turns is None:
            return None
        return self.max_turns - self.turns_used

    def record_turns(self, count: int) -> None:
        if self.max_turns is None:
            return
        self.turns_used += count


@dataclass
class _RunControl:
    active_stream: RunResultStreaming | None = None
    helper_task: asyncio.Task | None = None
    is_stop_requested: bool = False


_run_controls: dict[str, _RunControl] = {}


def _get_run_control(task_id: str) -> _RunControl | None:
    return _run_controls.get(task_id)


class _AgentRunStopped(RuntimeError):
    """Raised when a run ends because a stop was requested."""


class _ToolsDisabledError(AgentsException):
    """Raised when a tool call runs after the turn limit is reached."""


def _validate_turn_limit(
    max_turns: int | None,
    wrap_up_turns_remaining: int | None,
) -> None:
    if max_turns is not None:
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < _MIN_MAX_TURNS:
            raise ValueError("max_turns must be a positive integer or None")
    if wrap_up_turns_remaining is None:
        return
    if max_turns is None:
        raise ValueError("wrap_up_turns_remaining requires max_turns to be set")
    if (
        isinstance(wrap_up_turns_remaining, bool)
        or not isinstance(wrap_up_turns_remaining, int)
        or wrap_up_turns_remaining < 0
        or wrap_up_turns_remaining >= max_turns
    ):
        raise ValueError("wrap_up_turns_remaining must be an integer between 0 and max_turns - 1")


def _format_soft_wrap_up_notice(remaining: int) -> str:
    turn_word = "turn" if remaining == 1 else "turns"
    return (
        f"IMPORTANT NOTICE: You have {remaining} tool-enabled {turn_word} remaining "
        "before tools are disabled. Start wrapping up your work now and prepare your final response."
    )


def _convert_input_to_message_list(current_input: str | list) -> list:
    if isinstance(current_input, list):
        return list(current_input)
    return [{"role": "user", "content": current_input}]


def _format_log_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


class _ToolCallRaw(Protocol):
    arguments: str


def _extract_raw_arguments(raw_item: dict | _ToolCallRaw) -> str:
    if isinstance(raw_item, dict):
        return raw_item.get("arguments", "")
    return raw_item.arguments


def _parse_tool_arguments(arguments: str) -> dict:
    if not arguments:
        return {}
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {}


def _build_response_format(json_schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": _RESPONSE_SCHEMA_NAME,
            "strict": True,
            "schema": json_schema,
        },
    }


def _build_model(backend: config.Backend) -> OpenAIChatCompletionsModel:
    set_tracing_disabled(True)
    install_wire_converter()
    client = AsyncOpenAI(base_url=backend.url, api_key=backend.api_key or config.DEFAULT_API_KEY)
    return OpenAIChatCompletionsModel(
        model=backend.model or config.DEFAULT_MODEL,
        openai_client=client.with_options(max_retries=0),
        should_replay_reasoning_content=should_always_replay_reasoning,
    )


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, APIStatusError):
        return exc.status_code >= _INTERNAL_ERROR_STATUS or exc.status_code in _RETRYABLE_STATUS_CODES
    if isinstance(exc, APIError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code >= _INTERNAL_ERROR_STATUS or code in _RETRYABLE_STATUS_CODES
    return isinstance(
        exc,
        (
            httpx.RequestError,
            httpx.StreamError,
            ConnectionError,
            json.JSONDecodeError,
            ModelBehaviorError,
        ),
    )


async def _run_cancellable[R](start: Callable[[], Awaitable[R]], control: _RunControl) -> R:
    if control.is_stop_requested:
        raise _AgentRunStopped("agent run stopped on request")
    helper_task: asyncio.Task[R] = asyncio.ensure_future(start())
    control.helper_task = helper_task
    try:
        value = await helper_task
    except CancelledError:
        if control.is_stop_requested:
            raise _AgentRunStopped("agent run stopped on request") from None
        raise
    finally:
        control.helper_task = None
        helper_task.cancel()
    if control.is_stop_requested:
        raise _AgentRunStopped("agent run stopped on request")
    return value


def _format_task_result_notice(task_id: str) -> str:
    return (
        f"Task completed. task_id={task_id}. Retrieve the result with the task tool: "
        f"task(action='result', task_id='{task_id}')."
    )


class AgentRun:
    """Handle for one tracked agent run."""

    def __init__(self, task: Task[str]) -> None:
        self._task = task

    def cancel_run_loop(self) -> None:
        """Request that the run stop and cancel any active stream or retry wait."""
        control = _get_run_control(str(self._task.id))
        if control is None:
            print(
                f"[{_format_log_timestamp()}] cancel_run_loop: no run control for task "
                f"{str(self._task.id)[:_SHORT_TASK_ID_LENGTH]}",
                file=sys.stderr,
            )
            return
        control.is_stop_requested = True
        if control.active_stream is not None:
            control.active_stream.cancel()
        if control.helper_task is not None:
            control.helper_task.cancel()

    @property
    def task(self) -> Task[str]:
        """Return the tracked task backing this run."""
        return self._task

    async def stream_conversation(self) -> AsyncGenerator[ConversationEvent, None]:
        """Yield conversation events as they are written until the run finishes."""
        task = self._task

        async def _should_stop() -> bool:
            return task.status is not TaskStatus.RUNNING

        async for line in tail_lines(get_conversation_log_path(task), _should_stop):
            yield parse_conversation_event(line)

    async def read_events(self) -> list[ConversationEvent]:
        """Return all conversation events written for this run so far."""
        return await read_conversation_events(get_conversation_log_path(self._task))

    async def read_tool_calls(self) -> list[ToolCallRecord]:
        """Return the recorded tool calls from this run's conversation log."""
        return assemble_tool_calls(await self.read_events())


class Agent[T]:
    """Reusable agent configuration that runs prompts as tracked tasks."""

    class _AgentToolAdapter:
        def __init__(self, inner: Tool, results_by_id: bool) -> None:
            self.inner_tool = inner
            self.should_return_results_by_id = results_by_id
            self._added_max_char = (
                _AGENT_MAX_CHAR_PARAM if (inner.is_result_long and not results_by_id) else None
            )

        @property
        def params(self) -> tuple[Param, ...]:
            if self._added_max_char is None:
                return self.inner_tool.params
            return self.inner_tool.params + (self._added_max_char,)

        @property
        def name(self) -> str:
            return self.inner_tool.name

        @property
        def description(self) -> str:
            return self.inner_tool.description

        @property
        def is_tracked(self) -> bool:
            return self.inner_tool.is_tracked

        def _resolve_max_char_limit(self, params: dict) -> int | None:
            if self._added_max_char is None:
                return None
            return max(MIN_TOOL_OUTPUT_LEN_LIMIT, params.get("max_char", DEFAULT_TOOL_OUTPUT_LEN_LIMIT))

        async def run(self, params: dict) -> Task[ToolResult]:
            forwarded_params = dict(params)
            if self._added_max_char is not None:
                forwarded_params.pop("max_char", None)
            return await self.inner_tool.run(forwarded_params)

        async def format_tool_output(
            self,
            params: dict,
            task: Task[ToolResult],
            result_text: str,
            task_id: str | None,
            status: str | None,
        ) -> str | list[ToolOutputText | ToolOutputImage | ToolOutputFileContent]:
            if (
                self.should_return_results_by_id
                and status == TaskStatus.COMPLETED.value
                and task_id is not None
            ):
                return _format_task_result_notice(task_id)
            try:
                result = await task.result()
            except Exception:
                result = None
            if result is not None and result.files is not None:
                parts = await load_media_attachment_parts(result.files)
                return [ToolOutputText(text=result_text)] + parts
            max_char = self._resolve_max_char_limit(params)
            if max_char is not None and task_id is not None:
                return truncate(result_text, {"max_char": max_char}, task_id)
            return result_text

    def __init__(
        self,
        name: str,
        system_prompt: str,
        *,
        validator: (
            Callable[[str, AgentRun], ValidationResult[T]]
            | Callable[[str, AgentRun], Awaitable[ValidationResult[T]]]
        ),
        backend: config.Backend | None = None,
        timeout: float | None = None,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        max_turns: int | None = None,
        wrap_up_turns_remaining: int | None = None,
        tool_results_by_id: bool = False,
        model_input_filter: CallModelInputFilter | None = None,
        compact_hook: context.CompactHook | None = None,
        output_json_schema: dict | None = None,
    ) -> None:
        """Create a reusable agent configuration.

        Args:
            name: Name recorded for runs of this agent.
            system_prompt: System prompt sent to the model.
            validator: Check function that accepts or rejects model output.
            backend: Backend settings for model calls, or None for the default.
            timeout: Run timeout in seconds, or None for no timeout.
            max_attempts: Maximum output validation attempts per run.
            max_turns: Maximum tool turns per run before tools stop, or None for no limit.
            wrap_up_turns_remaining: Remaining tool turns when a wrap up notice is sent, or None to skip it.
            tool_results_by_id: Whether tracked tools return only a task id by default.
            model_input_filter: Optional filter run on model input before each call.
            compact_hook: Optional hook that shrinks context when input is too large.
            output_json_schema: Optional JSON schema enforced on model output.

        Raises:
            ValueError: If max_turns is not a positive integer, or if wrap_up_turns_remaining is out of range or set without max_turns.
        """
        _validate_turn_limit(max_turns, wrap_up_turns_remaining)
        self._name = name
        self._system_prompt = system_prompt
        self._tools: list[Agent._AgentToolAdapter] = []
        self._validator = validator
        self._backend = backend
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._max_turns = max_turns
        self._wrap_up_turns_remaining = wrap_up_turns_remaining
        self._tool_results_by_id = tool_results_by_id
        self._model_input_filter = model_input_filter
        self._compact_hook = compact_hook
        self._output_json_schema = output_json_schema

    def add_tool_adapter(self, tool: Tool, results_by_id: bool | None = None) -> Agent[T]:
        """Add one tool to the agent and return the agent.

        Args:
            tool: The tool to add.
            results_by_id: Whether tracked tool calls return only a task id, or
                None to use the agent default.

        Returns:
            This agent instance.
        """
        resolved = self._tool_results_by_id if results_by_id is None else results_by_id
        self._tools.append(self._AgentToolAdapter(tool, resolved))
        return self

    def _extract_tool_call_outcome(
        self,
        adapter: _AgentToolAdapter,
        task: Task[ToolResult],
    ) -> _ToolCallOutcome:
        if isinstance(task, AsyncTask):
            if adapter.is_tracked:
                return _ToolCallOutcome(task_id=str(task.id), status=task.status.value)
            return _EMPTY_TOOL_CALL_OUTCOME
        if isinstance(task, ImmediateTask):
            if task.has_stored_run:
                return _ToolCallOutcome(task_id=str(task.id), status=TaskStatus.COMPLETED.value)
            if adapter.is_tracked and task.status is TaskStatus.FAILED:
                return _ToolCallOutcome(task_id=None, status=_REJECTED_STATUS)
        return _EMPTY_TOOL_CALL_OUTCOME

    def _build_sdk_tool(
        self,
        adapter: _AgentToolAdapter,
        tool_calls: dict[str, _ToolCallOutcome],
        budget: _TurnBudget,
    ) -> FunctionTool:
        wire_params = adapter.params
        names = [param.name for param in wire_params]
        signature = inspect.Signature(
            [inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_ONLY, annotation=ToolContext)]
            + list(build_param_signature(wire_params).parameters.values()),
            return_annotation=str | list[ToolOutputText | ToolOutputImage | ToolOutputFileContent],
        )

        async def _execute_tool_call(
            ctx: ToolContext, *args: ParamValue
        ) -> str | list[ToolOutputText | ToolOutputImage | ToolOutputFileContent]:
            if budget.is_tools_disabled:
                raise _ToolsDisabledError(_TOOLS_DISABLED_ERROR_MESSAGE)
            params = dict(zip(names, args))
            task = await adapter.run(params)
            result_text = await adapter.inner_tool.read_result_text(task)
            outcome = self._extract_tool_call_outcome(adapter, task)
            tool_calls[ctx.tool_call_id] = outcome
            return await adapter.format_tool_output(
                params, task, result_text, outcome.task_id, outcome.status
            )

        def _reject_or_format_tool_error(ctx: RunContextWrapper, error: Exception) -> str:
            if isinstance(error, _ToolsDisabledError):
                raise error
            if budget.is_tools_disabled:
                raise _ToolsDisabledError(_TOOLS_DISABLED_ERROR_MESSAGE) from error
            return default_tool_error_function(ctx, error)

        _execute_tool_call.__signature__ = signature  # type: ignore[attr-defined]
        _execute_tool_call.__doc__ = adapter.description
        return function_tool(
            _execute_tool_call,
            name_override=adapter.name,
            description_override=adapter.description,
            use_docstring_info=False,
            failure_error_function=_reject_or_format_tool_error,
        )

    async def run(self, user_prompt: str) -> AgentRun:
        """Start a tracked run for the given prompt.

        Args:
            user_prompt: Initial prompt text sent to the agent.

        Returns:
            The run handle for the new tracked task.
        """
        task = await task_manager.create_task(partial(self._start_run, user_prompt))
        return AgentRun(task)

    async def _start_run(self, user_prompt: str) -> str:
        task = task_manager.active_task()
        assert task is not None
        task_id = str(task.id)
        backend = self._backend if self._backend is not None else config.backend(config.REASON_THINKING)
        model = _build_model(backend)
        await agent_store.insert(task_id, self._name)
        writer = ConversationLogWriter(get_conversation_log_path(task))
        await writer.write(build_conversation_event("system", 1, text=self._system_prompt))
        timeout_text = "infinite" if self._timeout is None else f"{self._timeout}s"
        print(
            f"[{_format_log_timestamp()}] agent started: {self._name}, "
            f"max_attempts={self._max_attempts}, timeout={timeout_text}"
        )
        print(f"[{_format_log_timestamp()}] system prompt loaded: {len(self._system_prompt)} chars")
        control = _RunControl()
        _run_controls[task_id] = control
        try:
            return await self._validate_loop(model, backend, writer, user_prompt, AgentRun(task), control)
        finally:
            _run_controls.pop(task_id, None)

    def _build_sdk_agent(
        self,
        model: OpenAIChatCompletionsModel,
        tool_calls: dict[str, _ToolCallOutcome],
        budget: _TurnBudget,
    ) -> _SdkAgent:
        kwargs: dict = {"name": self._name, "instructions": self._system_prompt, "model": model}
        if self._tools:
            kwargs["tools"] = [self._build_sdk_tool(tool, tool_calls, budget) for tool in self._tools]
        if self._output_json_schema is not None:
            kwargs["model_settings"] = ModelSettings(
                extra_args={"response_format": _build_response_format(self._output_json_schema)}
            )
        return _SdkAgent(**kwargs)

    def _build_compaction_measure(
        self, sdk_agent: _SdkAgent
    ) -> Callable[[list[dict]], Awaitable[int]] | None:
        input_filter = self._model_input_filter
        if input_filter is None:
            return None

        async def _measure(candidate: list[dict]) -> int:
            call_data = CallModelData(
                model_data=ModelInputData(
                    input=list(candidate),  # type: ignore[arg-type]
                    instructions=self._system_prompt,
                ),
                agent=sdk_agent,
                context=None,
            )
            filtered = input_filter(call_data)
            if inspect.isawaitable(filtered):
                filtered = await filtered
            if not filtered.input:
                raise context.CompactionError("Model input filter returned empty input.")
            return context.count_chars(
                filtered.input,  # type: ignore[arg-type]
                system_prompt=filtered.instructions or "",
            )

        return _measure

    async def _evaluate_output(
        self,
        final_output: str | None,
        agent_run: AgentRun,
    ) -> ValidationResult[T]:
        if final_output is None:
            return ValidationResult(is_valid=False, error="Error: result is None")
        if not final_output.strip():
            return ValidationResult(is_valid=False, error="Error: result is empty")
        raw_result = self._validator(final_output, agent_run)
        if inspect.isawaitable(raw_result):
            return await raw_result
        return raw_result

    async def _validate_loop(
        self,
        model: OpenAIChatCompletionsModel,
        backend: config.Backend,
        writer: ConversationLogWriter,
        user_prompt: str,
        agent_run: AgentRun,
        control: _RunControl,
    ) -> str:
        tool_calls: dict[str, _ToolCallOutcome] = {}
        budget = _TurnBudget(
            max_turns=self._max_turns,
            soft_threshold=self._wrap_up_turns_remaining or 0,
        )
        sdk_agent = self._build_sdk_agent(model, tool_calls, budget)
        input_text: str | list = user_prompt
        await writer.write(build_conversation_event("user", 1, text=user_prompt))
        last_error = ""
        start = time.monotonic()
        async with asyncio.timeout(self._timeout):
            for attempt in range(1, self._max_attempts + 1):
                print(f"[{_format_log_timestamp()}] attempt {attempt}/{self._max_attempts}")
                attempt_result = await self._run_attempt(
                    sdk_agent, backend, input_text, attempt, tool_calls, writer, control, budget
                )
                if attempt_result.blocked_error is not None:
                    validation_result: ValidationResult[T] = ValidationResult(
                        is_valid=False, error=attempt_result.blocked_error
                    )
                else:
                    validation_result = await self._evaluate_output(attempt_result.final_output, agent_run)
                if validation_result.is_valid:
                    serialized_output = (
                        validation_result.value
                        if isinstance(validation_result.value, str)
                        else json.dumps(validation_result.value, ensure_ascii=False)
                    )
                    print(f"[{_format_log_timestamp()}] validation passed")
                    elapsed = time.monotonic() - start
                    print(f"[{_format_log_timestamp()}] agent completed ({attempt} attempts, {elapsed:.1f}s)")
                    return serialized_output
                if control.is_stop_requested:
                    raise _AgentRunStopped("agent run stopped on request")
                last_error = validation_result.error + "\n\n" + _RETRY_NOTICE
                print(f"[{_format_log_timestamp()}] validation failed: {last_error}")
                if attempt < self._max_attempts:
                    input_text = attempt_result.history_items + [{"role": "user", "content": last_error}]
                    await writer.write(build_conversation_event("user", attempt, text=last_error))
        print(f"[{_format_log_timestamp()}] agent failed: {last_error}")
        raise RuntimeError(f"agent validation failed after {self._max_attempts} attempts: {last_error}")

    def _open_stream(
        self,
        sdk_agent: _SdkAgent,
        input_text: str | list,
        max_turns: int | None,
    ) -> RunResultStreaming:
        if self._model_input_filter is None:
            return Runner.run_streamed(sdk_agent, input_text, max_turns=max_turns)  # type: ignore[arg-type]
        run_config = RunConfig(call_model_input_filter=self._model_input_filter)
        return Runner.run_streamed(
            sdk_agent, input_text, max_turns=max_turns, run_config=run_config  # type: ignore[arg-type]
        )

    async def _prepare_stream_step(
        self,
        current_input: str | list,
        attempt: int,
        budget: _TurnBudget,
        writer: ConversationLogWriter,
    ) -> _StreamStep:
        remaining = budget.remaining
        if remaining is None:
            return _StreamStep(next_input=current_input, max_turns=None)
        if remaining > budget.soft_threshold:
            return _StreamStep(next_input=current_input, max_turns=remaining - budget.soft_threshold)
        if remaining > 0:
            if not budget.is_soft_nudge_sent:
                notice = _format_soft_wrap_up_notice(remaining)
                current_input = _convert_input_to_message_list(current_input) + [{"role": "user", "content": notice}]
                await writer.write(build_conversation_event("system", attempt, text=notice))
                budget.is_soft_nudge_sent = True
            return _StreamStep(next_input=current_input, max_turns=remaining)
        if not budget.is_hard_wrap_up_sent:
            current_input = _convert_input_to_message_list(current_input) + [
                {"role": "user", "content": _HARD_WRAP_UP_NOTICE}
            ]
            await writer.write(build_conversation_event("system", attempt, text=_HARD_WRAP_UP_NOTICE))
            budget.is_hard_wrap_up_sent = True
        return _StreamStep(next_input=current_input, max_turns=_HARD_WRAP_UP_MAX_TURNS)

    async def _finish_stream_attempt(
        self,
        result: RunResultStreaming,
        attempt: int,
        writer: ConversationLogWriter,
        budget: _TurnBudget,
    ) -> _AttemptResult:
        if budget.max_turns is not None:
            budget.record_turns(len(result.raw_responses))
        final_output = result.final_output
        if final_output is not None:
            await writer.write(build_conversation_event("assistant", attempt, text=final_output))
        return _AttemptResult(
            final_output=final_output,
            history_items=result.to_input_list(),
            blocked_error=None,
        )

    def _record_reached_turn_boundary_and_extract_history(
        self,
        result: RunResultStreaming,
        stream_max_turns: int | None,
        budget: _TurnBudget,
        exc: MaxTurnsExceeded,
    ) -> list:
        if budget.is_tools_disabled:
            raise RuntimeError("agent failed to produce a final output after turn limit") from exc
        advanced = max(len(result.raw_responses), stream_max_turns or 0)
        budget.record_turns(advanced)
        return result.to_input_list()

    async def _compact_for_overflow(
        self,
        current_input: str | list,
        attempt: int,
        measure: Callable[[list[dict]], Awaitable[int]] | None,
        control: _RunControl,
    ) -> _RecoveryStep:
        compacted = await _run_cancellable(
            partial(
                context.compact,
                current_input,
                system_prompt=self._system_prompt,
                measure=measure,
                compact_hook=self._compact_hook,
                verbose=True,
            ),
            control,
        )
        print(
            f"[{_format_log_timestamp()}] attempt {attempt} context compacted, "
            f"{len(compacted)} items remaining"
        )
        return _RecoveryStep(next_input=compacted, next_retry_count=0)

    async def _recover_or_retry(
        self,
        exc: Exception,
        *,
        backend: config.Backend,
        current_input: str | list,
        retries: int,
        attempt: int,
        measure: Callable[[list[dict]], Awaitable[int]] | None,
        control: _RunControl,
    ) -> _RecoveryStep:
        can_retry = _is_retryable_error(exc) and retries < _MAX_SERVER_RETRIES
        try:
            is_overflow = await _run_cancellable(
                partial(context.is_overflow, exc, backend=backend, before_retry=can_retry),
                control,
            )
            if is_overflow:
                return await self._compact_for_overflow(current_input, attempt, measure, control)
        except (context.ClassificationError, context.CompactionError) as recovery_exc:
            print(
                f"[{_format_log_timestamp()}] attempt {attempt} context recovery aborted, {recovery_exc}",
                file=sys.stderr,
            )
            if control.is_stop_requested:
                raise _AgentRunStopped("agent run stopped on request") from None
            raise exc
        if not can_retry:
            raise exc
        next_retry = retries + 1
        delay = _RETRY_INITIAL_DELAY * (_RETRY_BACKOFF_FACTOR ** (next_retry - 1))
        print(
            f"[{_format_log_timestamp()}] attempt {attempt} "
            f"retry {next_retry}/{_MAX_SERVER_RETRIES} in {delay}s: {exc}"
        )
        await _run_cancellable(lambda: asyncio.sleep(delay), control)
        return _RecoveryStep(next_input=current_input, next_retry_count=next_retry)

    async def _run_attempt(
        self,
        sdk_agent: _SdkAgent,
        backend: config.Backend,
        input_text: str | list,
        attempt: int,
        tool_calls: dict[str, _ToolCallOutcome],
        writer: ConversationLogWriter,
        control: _RunControl,
        budget: _TurnBudget,
    ) -> _AttemptResult:
        current_input: str | list = input_text
        retries = 0
        measure = self._build_compaction_measure(sdk_agent)
        while True:
            step = await self._prepare_stream_step(current_input, attempt, budget, writer)
            current_input = step.next_input
            result = self._open_stream(sdk_agent, current_input, step.max_turns)
            control.active_stream = result
            caught_exc: Exception | None = None
            try:
                await self._record_conversation(result, attempt, tool_calls, writer, budget)
                return await self._finish_stream_attempt(result, attempt, writer, budget)
            except Exception as exc:
                if control.is_stop_requested:
                    raise _AgentRunStopped("agent run stopped on request") from exc
                if isinstance(exc, MaxTurnsExceeded) and budget.max_turns is not None:
                    current_input = self._record_reached_turn_boundary_and_extract_history(result, step.max_turns, budget, exc)
                    retries = 0
                    continue
                if budget.is_tools_disabled and isinstance(exc, (_ToolsDisabledError, ModelBehaviorError)):
                    return _AttemptResult(
                        final_output=None,
                        history_items=_convert_input_to_message_list(current_input),
                        blocked_error=_TOOLS_DISABLED_FEEDBACK,
                    )
                if budget.max_turns is not None:
                    budget.record_turns(len(result.raw_responses))
                if result.new_items:
                    current_input = result.to_input_list()
                    retries = 0
                caught_exc = exc
            finally:
                control.active_stream = None
                result.cancel()
            assert caught_exc is not None
            recovery = await self._recover_or_retry(
                caught_exc,
                backend=backend,
                current_input=current_input,
                retries=retries,
                attempt=attempt,
                measure=measure,
                control=control,
            )
            current_input = recovery.next_input
            retries = recovery.next_retry_count

    async def _record_tool_call_started(
        self,
        item: RunItem,
        attempt: int,
        tool_calls: dict[str, _ToolCallOutcome],
        call_names: dict[str, str],
        call_starts: dict[str, float],
        writer: ConversationLogWriter,
    ) -> None:
        call_id: str | None = item.call_id  # type: ignore[attr-defined]
        tool_name: str | None = item.tool_name  # type: ignore[attr-defined]
        if call_id and tool_name is not None:
            call_names[call_id] = tool_name
            call_starts[call_id] = time.monotonic()
        await writer.write(
            build_conversation_event(
                "tool_call",
                attempt,
                tool_name=tool_name,
                params=_parse_tool_arguments(_extract_raw_arguments(item.raw_item)),  # type: ignore[arg-type]
                call_id=call_id,
            )
        )
        outcome = tool_calls.get(call_id, _EMPTY_TOOL_CALL_OUTCOME) if call_id else _EMPTY_TOOL_CALL_OUTCOME
        print(
            f"[{_format_log_timestamp()}] tool_call {tool_name} -> "
            f"task {(outcome.task_id or '?')[:_SHORT_TASK_ID_LENGTH]}"
        )

    async def _record_tool_call_finished(
        self,
        item: RunItem,
        attempt: int,
        tool_calls: dict[str, _ToolCallOutcome],
        call_names: dict[str, str],
        call_starts: dict[str, float],
        writer: ConversationLogWriter,
    ) -> None:
        call_id: str | None = item.call_id  # type: ignore[attr-defined]
        recorded_tool_name = call_names.get(call_id) if call_id else None
        outcome = tool_calls.get(call_id, _EMPTY_TOOL_CALL_OUTCOME) if call_id else _EMPTY_TOOL_CALL_OUTCOME
        await writer.write(
            build_conversation_event(
                "tool_result",
                attempt,
                tool_name=recorded_tool_name,
                task_id=outcome.task_id,
                status=outcome.status,
                call_id=call_id,
            )
        )
        started = call_starts.get(call_id) if call_id else None
        if started is not None:
            print(
                f"[{_format_log_timestamp()}] tool_call {recorded_tool_name or '?'} "
                f"done in {time.monotonic() - started:.1f}s"
            )
        else:
            print(f"[{_format_log_timestamp()}] tool_call {recorded_tool_name or '?'} done")

    async def _record_tool_item(
        self,
        item: RunItem,
        attempt: int,
        tool_calls: dict[str, _ToolCallOutcome],
        call_names: dict[str, str],
        call_starts: dict[str, float],
        writer: ConversationLogWriter,
        budget: _TurnBudget,
    ) -> None:
        if item.type == "tool_call_item":
            if not budget.is_tools_disabled:
                await self._record_tool_call_started(item, attempt, tool_calls, call_names, call_starts, writer)
        elif item.type == "tool_call_output_item":
            await self._record_tool_call_finished(item, attempt, tool_calls, call_names, call_starts, writer)

    async def _record_conversation(
        self,
        result: RunResultStreaming,
        attempt: int,
        tool_calls: dict[str, _ToolCallOutcome],
        writer: ConversationLogWriter,
        budget: _TurnBudget,
    ) -> None:
        call_names: dict[str, str] = {}
        call_starts: dict[str, float] = {}
        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                event_data = event.data
                if isinstance(event_data, ResponseTextDeltaEvent):
                    section = "content"
                    delta = event_data.delta
                elif isinstance(
                    event_data,
                    (ResponseReasoningTextDeltaEvent, ResponseReasoningSummaryTextDeltaEvent),
                ):
                    section = "thinking"
                    delta = event_data.delta
                else:
                    continue
                if delta:
                    await writer.write(build_conversation_event(section, attempt, text=delta))
            elif isinstance(event, RunItemStreamEvent):
                await self._record_tool_item(
                    event.item, attempt, tool_calls, call_names, call_starts, writer, budget
                )
