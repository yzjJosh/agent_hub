from __future__ import annotations

import asyncio
import base64
import inspect
import io
import json
import mimetypes
import os
import sys
import time
import traceback
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, Protocol

import aiofiles
from aiofiles.tempfile import TemporaryDirectory
from agents import Agent as _SdkAgent
from agents import OpenAIChatCompletionsModel, Runner, ToolOutputFileContent, ToolOutputImage, ToolOutputText, function_tool, set_tracing_disabled
from agents.items import TResponseInputItem
from agents.models import chatcmpl_converter, openai_chatcompletions
from agents.models.reasoning_content_replay import ShouldReplayReasoningContent
from agents.run import RunResultStreaming
from agents.run_config import CallModelInputFilter, RunConfig
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from agents.tool_context import ToolContext
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import (
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseTextDeltaEvent,
)
from PIL import Image

from app.lib.cli.cli_runner import run_cli
from app.lib.db import store as db_store
from app.lib.db.store import agent_store
from app.lib.shared import config
from app.lib.shared.paths import is_image_mime, is_video_mime, mime_type, resolve_path, resources_dir
from app.lib.task.task_manager import AsyncTask, ImmediateTask, Task, TaskStatus, task_manager
from app.lib.tool.base import Param, ParamValue, Tool, ToolResult, param_signature
from app.lib.tool.truncation import (
    DEFAULT_TOOL_OUTPUT_LEN_LIMIT,
    MAX_CHAR_DESCRIPTION,
    MIN_TOOL_OUTPUT_LEN_LIMIT,
    truncate,
)

if TYPE_CHECKING:
    from agents import FunctionTool

__all__ = ["Agent", "AgentRun", "tail_lines"]

_TAIL_POLL = 0.05
_DEFAULT_MAX_ATTEMPTS = 10
_MEDIA_DECODE_TIMEOUT = 30
# How the adapter marks a call its tool rejected before the run started.
_REJECTED_STATUS = "rejected"
_RETRY_NOTICE = (
    "IMPORTANT NOTE: Your last response is dropped. Please correct the error and re-generate your "
    "entire response in your next message. Please don't put any filler content. Your message must "
    "contain ONLY the final output."
)

# The client-layer param the agent adds to a long-result tool's params when id-only is off.
_AGENT_MAX_CHAR_PARAM = Param(
    "max_char",
    int,
    DEFAULT_TOOL_OUTPUT_LEN_LIMIT,
    min=MIN_TOOL_OUTPUT_LEN_LIMIT,
    description=MAX_CHAR_DESCRIPTION,
)


@dataclass(frozen=True)
class _ToolCallRecord:
    task_id: str
    tool_name: str
    params: dict
    status: str
    run_dir: str


@dataclass(frozen=True)
class _CallReport:
    """What one in-loop tool call reported about itself."""

    task_id: str | None
    status: str | None


_NO_CALL_REPORT = _CallReport(task_id=None, status=None)


@dataclass
class _RunControl:
    live: RunResultStreaming | None = None
    stop_requested: bool = False


_run_controls: dict[str, _RunControl] = {}


def _run_control_of(task_id: str) -> _RunControl | None:
    return _run_controls.get(task_id)


class _AgentRunStopped(RuntimeError):
    """A run that ended because a stop was requested on it."""


@dataclass(frozen=True)
class _ConversationEvent:
    type: str
    attempt: int
    ts: str
    text: str = ""
    tool_name: str | None = None
    params: dict | None = None
    task_id: str | None = None
    call_id: str | None = None
    status: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _event(
    kind: str,
    attempt: int,
    *,
    text: str = "",
    tool_name: str | None = None,
    params: dict | None = None,
    task_id: str | None = None,
    call_id: str | None = None,
    status: str | None = None,
) -> _ConversationEvent:
    return _ConversationEvent(
        type=kind, attempt=attempt, ts=_now_iso(),
        text=text, tool_name=tool_name, params=params, task_id=task_id,
        call_id=call_id, status=status,
    )


def _event_line(event: _ConversationEvent) -> str:
    return json.dumps({
        "type": event.type,
        "attempt": event.attempt,
        "ts": event.ts,
        "text": event.text,
        "tool_name": event.tool_name,
        "params": event.params,
        "task_id": event.task_id,
        "call_id": event.call_id,
        "status": event.status,
    }, ensure_ascii=False)


def _event_from_line(line: str) -> _ConversationEvent:
    data = json.loads(line)
    data.pop("cached", None)  # lines written before the field was removed carry it
    return _ConversationEvent(**data)


class _ToolCallRaw(Protocol):
    """A raw SDK tool-call item exposing its serialized argument string."""

    arguments: str


def _arguments_of(raw_item: dict | _ToolCallRaw) -> str:
    """The raw tool-call item's argument string, whether the item is a dict or a SDK object."""
    if isinstance(raw_item, dict):
        return raw_item.get("arguments", "")
    return raw_item.arguments


def _parse_args(args: str) -> dict:
    """Parse a tool-call argument string into a dict, returning {} when it is empty or not JSON."""
    if not args:
        return {}
    try:
        return json.loads(args)
    except json.JSONDecodeError:
        return {}


def _media_wire_part(part: dict) -> dict | None:
    """The wire shape for one tool-output part, or None when the backend rejects it."""
    if part.get("type") != "file":
        return part
    file = part.get("file")
    file = file if isinstance(file, dict) else {}
    file_data = file.get("file_data")
    if not isinstance(file_data, str) or not file_data:
        return None
    mime: str | None = None
    if file_data.startswith("data:"):
        head = file_data.split(";", 1)[0]
        if len(head) > len("data:"):
            mime = head[len("data:"):]
    elif isinstance(file.get("filename"), str):
        mime = mimetypes.guess_type(file["filename"])[0]
    if not is_video_mime(mime):
        return None
    url = file_data if file_data.startswith("data:") else f"data:{mime};base64,{file_data}"
    return {"type": "video_url", "video_url": {"url": url}}


def _media_wire_parts(messages: list[dict]) -> list[dict]:
    """Tool file parts become video_url parts, other file parts drop, everything else stays."""
    out: list[dict] = []
    for message in messages:
        if message.get("role") != "tool":
            out.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        parts = [_media_wire_part(part) if isinstance(part, dict) else part for part in content]
        message = dict(message)
        message["content"] = [part for part in parts if part is not None]
        out.append(message)
    return out


class _PreservingConverter(chatcmpl_converter.Converter):
    """Force tool-output preservation on, and map file parts the wire accepts."""

    @classmethod
    def items_to_messages(
        cls,
        items: str | Iterable[TResponseInputItem],
        model: str | None = None,
        preserve_thinking_blocks: bool = False,
        base_url: str | None = None,
        should_replay_reasoning_content: ShouldReplayReasoningContent | None = None,
        strict_feature_validation: bool = False,
    ) -> list[ChatCompletionMessageParam]:
        messages = super().items_to_messages(
            items,
            model=model,
            preserve_thinking_blocks=preserve_thinking_blocks,
            preserve_tool_output_all_content=True,
            base_url=base_url,
            should_replay_reasoning_content=should_replay_reasoning_content,
            strict_feature_validation=strict_feature_validation,
        )
        return _media_wire_parts(messages)


def _build_model(backend: config.Backend) -> OpenAIChatCompletionsModel:
    set_tracing_disabled(True)
    if openai_chatcompletions.Converter is not _PreservingConverter:
        openai_chatcompletions.Converter = _PreservingConverter
    client = AsyncOpenAI(base_url=backend.url, api_key=backend.api_key or config.DEFAULT_API_KEY)
    return OpenAIChatCompletionsModel(model=backend.model or config.DEFAULT_MODEL, openai_client=client)


class _ConversationWriter:
    """Appends one event per line to the run's conversation.jsonl."""

    def __init__(self, path: str) -> None:
        self._path = path

    async def write(self, event: _ConversationEvent) -> None:
        line = _event_line(event) + "\n"
        async with aiofiles.open(self._path, "a", encoding="utf-8") as fh:
            await fh.write(line)


async def _read_tail(path: str, offset: int) -> bytes:
    try:
        async with aiofiles.open(path, "rb") as fh:
            await fh.seek(offset)
            return await fh.read()
    except FileNotFoundError:
        return b""


async def tail_lines(
    path: str,
    should_stop: Callable[[], Awaitable[bool]],
    poll: float = _TAIL_POLL,
) -> AsyncGenerator[str, None]:
    """Yield complete lines from a growing file until the stop signal fires.

    Args:
        path: The file to tail.
        should_stop: An async check that ends the tail when it returns True.
        poll: Seconds between reads when no new line is available.

    Yields:
        Each complete line without its trailing newline.
    """
    offset = 0
    while True:
        chunk = await _read_tail(path, offset)
        newline = chunk.find(b"\n")
        while newline != -1:
            line = chunk[:newline]
            if line:
                yield line.decode("utf-8")
            offset += newline + 1
            chunk = chunk[newline + 1:]
            newline = chunk.find(b"\n")
        if await should_stop():
            break
        await asyncio.sleep(poll)


def _result_notice(task_id: str) -> str:
    """The completion notice a tool call makes when its result is fetched by task id."""
    return (
        f"Task completed. task_id={task_id}. Retrieve the result with the task tool: "
        f"task(action='result', task_id='{task_id}')."
    )


def _data_url(data: bytes, mime: str) -> str:
    return "data:" + mime + ";base64," + base64.b64encode(data).decode()


def _image_url(data: bytes, mime: str) -> str:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        mime = Image.MIME.get(image.format, mime)
    return _data_url(data, mime)


async def _media_part(data: bytes, mime: str, filename: str) -> ToolOutputImage | ToolOutputFileContent:
    if is_image_mime(mime):
        return ToolOutputImage(image_url=await asyncio.to_thread(_image_url, data, mime))
    async with TemporaryDirectory(prefix="agent-media-") as directory:
        path = os.path.join(directory, filename)
        async with aiofiles.open(path, "wb") as fh:
            await fh.write(data)
        result = await run_cli(
            ["ffmpeg", "-v", "error", "-xerror", "-i", path, "-map", "0:v:0",
             "-frames:v", "1", "-f", "null", "-"],
            "attachment video decode", timeout=_MEDIA_DECODE_TIMEOUT,
        )
        if result.returncode != 0:
            raise ValueError(f"Video decode failed. {result.stderr}")
    url = await asyncio.to_thread(_data_url, data, mime)
    return ToolOutputFileContent(file_data=url, filename=filename)


def _delivery_error(path: str, reason: str, error: Exception) -> ToolOutputText:
    print(f"[Agent] Attachment delivery failed for {path[:100]}", file=sys.stderr)
    traceback.print_exception(error, file=sys.stderr)
    return ToolOutputText(text=f"Attachment delivery failed for {path!r}. {reason}")


async def _file_part(path: str) -> list[ToolOutputText | ToolOutputImage | ToolOutputFileContent]:
    guessed_mime = mimetypes.guess_type(path)[0]
    is_media = guessed_mime is None or is_image_mime(guessed_mime) or is_video_mime(guessed_mime)
    try:
        candidate = path if os.path.isabs(path) else await resolve_path(resources_dir(), path)
        if candidate is None:
            raise FileNotFoundError("Attachment path is unavailable.")
        async with aiofiles.open(candidate, "rb") as fh:
            data = await fh.read()
        mime = await mime_type(candidate) or guessed_mime
        if not is_image_mime(mime) and not is_video_mime(mime):
            if is_image_mime(guessed_mime) or is_video_mime(guessed_mime):
                raise ValueError("Attachment is not valid media.")
            return []
        is_media = True
        return [await _media_part(data, mime, os.path.basename(candidate))]
    except (OSError, ValueError) as exc:
        if not is_media:
            return []
        return [_delivery_error(path, "Media could not be read or decoded.", exc)]
    except Exception as exc:
        if not is_media:
            return []
        return [_delivery_error(path, "Media could not be delivered.", exc)]


async def _file_parts(files: tuple[str, ...]) -> list[ToolOutputText | ToolOutputImage | ToolOutputFileContent]:
    """Return readable media and a delivery error for each unavailable attachment."""
    parts: list[ToolOutputText | ToolOutputImage | ToolOutputFileContent] = []
    for path in files:
        parts.extend(await _file_part(path))
    return parts


def _conversation_path(task: Task) -> str:
    run_dir = f"{db_store.RUNS_DIR}/{str(task.id)}"
    return os.path.join(db_store.task_store.data_dir, run_dir, "conversation.jsonl")


class AgentRun:
    """The per-run handle for one agent run, composing its tracked Task.

    cancel_run_loop asks the run to stop. It records the stop and cancels the
    streaming result of the attempt in progress.
    """

    def __init__(self, task: Task[str]) -> None:
        self._task = task

    def cancel_run_loop(self) -> None:
        """Ask the run to stop, and cancel its live streaming result."""
        control = _run_control_of(str(self._task.id))
        if control is None:
            print(f"[{_ts()}] cancel_run_loop: no run control for task {str(self._task.id)[:8]}", file=sys.stderr)
            return
        control.stop_requested = True
        if control.live is not None:
            control.live.cancel()

    @property
    def task(self) -> Task[str]:
        return self._task

    async def conversation(self) -> AsyncGenerator[_ConversationEvent, None]:
        """Tail the run's conversation.jsonl and yield events until the task is terminal."""
        task = self._task

        async def _stop() -> bool:
            return task.status is not TaskStatus.RUNNING

        async for line in tail_lines(_conversation_path(task), _stop):
            yield _event_from_line(line)

    async def events(self) -> list[_ConversationEvent]:
        """Read the run's conversation.jsonl and return the events written so far."""
        try:
            async with aiofiles.open(_conversation_path(self._task), "r", encoding="utf-8") as fh:
                lines = [line async for line in fh]
        except FileNotFoundError:
            return []
        return [_event_from_line(line) for line in lines if line.strip()]

    async def tool_calls(self) -> list[_ToolCallRecord]:
        """Assemble the in-loop tool-call history from the run's conversation log.

        Returns:
            One entry per tool result that names a task, in the order the results
            were written. A call that reused a stored run carries the task id of
            that run. A result whose call never appeared in the log is left out.
        """
        started: dict[str, _ConversationEvent] = {}
        calls: list[_ToolCallRecord] = []
        for event in await self.events():
            if event.call_id is None:
                continue
            if event.type == "tool_call":
                started[event.call_id] = event
            elif event.type == "tool_result" and event.task_id is not None:
                call = started.get(event.call_id)
                if call is None or call.tool_name is None:
                    continue
                calls.append(
                    _ToolCallRecord(
                        task_id=event.task_id,
                        tool_name=call.tool_name,
                        params=call.params if call.params is not None else {},
                        status=event.status if event.status is not None else "",
                        run_dir=f"{db_store.RUNS_DIR}/{event.task_id}",
                    )
                )
        return calls


class Agent[T]:
    """A reusable agent config that launches the loop as tracked, cancellable tasks.

    A validator value that is not a str comes back from the run as a JSON string,
    and the run's result file holds that same string.

    With tool_results_by_id on, a tracked tool call hands the model only its task id
    and how to fetch the result. The flag is the default for a tool added without a
    results_by_id value; a value passed at the add call wins, and an explicit False
    under a flag-on agent keeps that tool's full result. An untracked tool still
    hands back its own text.

    With model_input_filter set, the filter runs before every model call in the run
    and the run sends the model input the filter returns. Without it, the run sends
    the input unchanged.
    """

    class _AgentToolAdapter:
        """Stands in front of one inner tool with the agent's client-layer contract.

        Every fact is read from the live inner tool. The adapter owns only the
        client-layer max_char param it may add to the params, and the presentation
        of the run's result.
        """

        def __init__(self, inner: Tool, results_by_id: bool) -> None:
            self.inner_tool = inner
            self.results_by_id = results_by_id
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
        def tracked(self) -> bool:
            return self.inner_tool.tracked

        def _call_max_char(self, params: dict) -> int | None:
            """The call's limit floored at the minimum, or None when it added no param."""
            if self._added_max_char is None:
                return None
            return max(MIN_TOOL_OUTPUT_LEN_LIMIT, params.get("max_char", DEFAULT_TOOL_OUTPUT_LEN_LIMIT))

        async def run(self, params: dict) -> Task[ToolResult]:
            """Run the inner tool with the call's params minus the client-layer max_char."""
            working = dict(params)
            if self._added_max_char is not None:
                working.pop("max_char", None)
            return await self.inner_tool.run(working)

        async def present(
            self,
            params: dict,
            task: Task[ToolResult],
            task_id: str | None,
            status: str | None,
        ) -> str | list[ToolOutputText | ToolOutputImage | ToolOutputFileContent]:
            """The result the model sees for the run, in the adapter's decision order."""
            if self.results_by_id and status == TaskStatus.COMPLETED.value and task_id is not None:
                return _result_notice(task_id)
            try:
                result = await task.result()
            except Exception:
                result = None
            text = await self.inner_tool.result_text(task)
            if result is not None and result.files is not None:
                parts = await _file_parts(result.files)
                return [ToolOutputText(text=text)] + parts
            max_char = self._call_max_char(params)
            if max_char is not None and task_id is not None:
                return truncate(text, {"max_char": max_char}, task_id)
            return text

    def __init__(
        self,
        name: str,
        system_prompt: str,
        *,
        validator: Callable[[str, AgentRun], tuple[bool, T]] | Callable[[str, AgentRun], Awaitable[tuple[bool, T]]],
        backend: config.Backend | None = None,
        timeout: float | None = None,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        tool_results_by_id: bool = False,
        model_input_filter: CallModelInputFilter | None = None,
    ) -> None:
        self._name = name
        self._system_prompt = system_prompt
        self._tools: list[_AgentToolAdapter] = []
        self._validator = validator
        self._backend = backend
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._tool_results_by_id = tool_results_by_id
        self._model_input_filter = model_input_filter

    def add_tool_adapter(self, tool: Tool, results_by_id: bool | None = None) -> Agent[T]:
        """Add one tool to the agent and hand it back to chain further adds.

        The id-only decision layers in order: a results_by_id value passed here
        wins, None defers to the agent's tool_results_by_id flag, and the flag's
        own default is off. An explicit False under a flag-on agent keeps the
        tool's full result.

        Args:
            tool: The tool to add.
            results_by_id: Whether the tool's tracked calls hand the model only
                their task id, or None to defer to the agent-level flag.
        """
        resolved = self._tool_results_by_id if results_by_id is None else results_by_id
        self._tools.append(self._AgentToolAdapter(tool, resolved))
        return self

    def _call_report_of(self, adapter: _AgentToolAdapter, task: Task[ToolResult]) -> _CallReport:
        """What an in-loop tool call reported about itself.

        Args:
            adapter: The adapter the call ran through.
            task: The handle the call returned.

        Returns:
            The run id the call recorded, and how it ended, or an empty record for a
            call that persisted no run.
        """
        if isinstance(task, AsyncTask):
            if adapter.tracked:
                return _CallReport(task_id=str(task.id), status=task.status.value)
            return _NO_CALL_REPORT
        if isinstance(task, ImmediateTask):
            if task.has_stored_run:
                return _CallReport(task_id=str(task.id), status=TaskStatus.COMPLETED.value)
            if adapter.tracked and task.status is TaskStatus.FAILED:
                return _CallReport(task_id=None, status=_REJECTED_STATUS)
        return _NO_CALL_REPORT

    def _build_sdk_tool(self, adapter: _AgentToolAdapter, tool_calls: dict[str, _CallReport]) -> FunctionTool:
        """Adapt one added tool to an SDK function tool that runs it through its adapter."""
        wire_params = adapter.params
        names = [p.name for p in wire_params]
        signature = inspect.Signature(
            [inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_ONLY, annotation=ToolContext)]
            + list(param_signature(wire_params).parameters.values()),
            return_annotation=str | list[ToolOutputText | ToolOutputImage | ToolOutputFileContent],
        )

        async def _execute_tool_call(
            ctx: ToolContext, *args: ParamValue
        ) -> str | list[ToolOutputText | ToolOutputImage | ToolOutputFileContent]:
            params = dict(zip(names, args))
            task = await adapter.run(params)
            # the result_text await drives the run to a terminal state before the report
            await adapter.inner_tool.result_text(task)
            tool_call = self._call_report_of(adapter, task)
            tool_calls[ctx.tool_call_id] = tool_call
            return await adapter.present(params, task, tool_call.task_id, tool_call.status)

        _execute_tool_call.__signature__ = signature
        _execute_tool_call.__doc__ = adapter.description
        return function_tool(
            _execute_tool_call,
            name_override=adapter.name,
            description_override=adapter.description,
            use_docstring_info=False,
        )

    async def run(self, user_prompt: str) -> AgentRun:
        """Launch a new tracked run for the user prompt, returning its per-run handle."""
        task = await task_manager.create_task(partial(self._start_run, user_prompt))
        return AgentRun(task)

    async def _start_run(self, user_prompt: str) -> str:
        task = task_manager.active_task()
        task_id = str(task.id)
        run_dir = f"{db_store.RUNS_DIR}/{task_id}"
        run_path = os.path.join(db_store.task_store.data_dir, run_dir)
        backend = self._backend if self._backend is not None else config.backend(config.REASON_THINKING)
        model = _build_model(backend)
        await agent_store.insert(task_id, self._name)
        writer = _ConversationWriter(os.path.join(run_path, "conversation.jsonl"))
        await writer.write(_event("system", 1, text=self._system_prompt))
        timeout_text = "infinite" if self._timeout is None else f"{self._timeout}s"
        print(f"[{_ts()}] agent started: {self._name}, max_attempts={self._max_attempts}, timeout={timeout_text}")
        print(f"[{_ts()}] system prompt loaded: {len(self._system_prompt)} chars")
        control = _RunControl()
        _run_controls[task_id] = control
        try:
            return await self._validate_loop(model, writer, user_prompt, AgentRun(task), control)
        finally:
            _run_controls.pop(task_id, None)

    def _build_sdk_agent(self, model: OpenAIChatCompletionsModel, tool_calls: dict[str, _CallReport]) -> _SdkAgent:
        kwargs: dict = {"name": self._name, "instructions": self._system_prompt, "model": model}
        if self._tools:
            kwargs["tools"] = [self._build_sdk_tool(t, tool_calls) for t in self._tools]
        return _SdkAgent(**kwargs)

    async def _validate_loop(
        self,
        model: OpenAIChatCompletionsModel,
        writer: _ConversationWriter,
        user_prompt: str,
        agent_run: AgentRun,
        control: _RunControl,
    ) -> str:
        tool_calls: dict[str, _CallReport] = {}
        sdk_agent = self._build_sdk_agent(model, tool_calls)
        input_text: str | list = user_prompt
        await writer.write(_event("user", 1, text=user_prompt))
        last_error = ""
        start = time.monotonic()
        async with asyncio.timeout(self._timeout):
            for attempt in range(1, self._max_attempts + 1):
                print(f"[{_ts()}] attempt {attempt}/{self._max_attempts}")
                final_output, input_list = await self._run_attempt(sdk_agent, input_text, attempt, tool_calls, writer, control)
                if final_output is None:
                    ok, checked = False, "Error: result is None"
                else:
                    checked = self._validator(final_output, agent_run)
                    if inspect.isawaitable(checked):
                        checked = await checked
                    ok, checked = checked
                if ok:
                    if not isinstance(checked, str):
                        checked = json.dumps(checked, ensure_ascii=False)
                    print(f"[{_ts()}] validation passed")
                    elapsed = time.monotonic() - start
                    print(f"[{_ts()}] agent completed ({attempt} attempts, {elapsed:.1f}s)")
                    return checked
                if control.stop_requested:
                    raise _AgentRunStopped("agent run stopped on request")
                last_error = str(checked) + "\n\n" + _RETRY_NOTICE
                print(f"[{_ts()}] validation failed: {last_error}")
                if attempt < self._max_attempts:
                    input_text = input_list + [{"role": "user", "content": last_error}]
                    await writer.write(_event("user", attempt, text=last_error))
        print(f"[{_ts()}] agent failed: {last_error}")
        raise RuntimeError(f"agent validation failed after {self._max_attempts} attempts: {last_error}")

    async def _run_attempt(
        self,
        sdk_agent: _SdkAgent,
        input_text: str | list,
        attempt: int,
        tool_calls: dict[str, _CallReport],
        writer: _ConversationWriter,
        control: _RunControl,
    ) -> tuple[str | None, list]:
        if self._model_input_filter is None:
            result = Runner.run_streamed(sdk_agent, input_text, max_turns=None)
        else:
            run_config = RunConfig(call_model_input_filter=self._model_input_filter)
            result = Runner.run_streamed(sdk_agent, input_text, max_turns=None, run_config=run_config)
        control.live = result
        try:
            await self._record_conversation(result, attempt, tool_calls, writer)
            final_output = result.final_output
            if final_output is not None:
                await writer.write(_event("assistant", attempt, text=final_output))
            input_list = result.to_input_list()
        finally:
            control.live = None
            result.cancel()
        return final_output, input_list

    async def _record_conversation(
        self,
        result: RunResultStreaming,
        attempt: int,
        tool_calls: dict[str, _CallReport],
        writer: _ConversationWriter,
    ) -> None:
        acc_type: str | None = None
        acc_text = ""
        call_names: dict[str, str] = {}
        call_starts: dict[str, float] = {}

        async def flush() -> None:
            nonlocal acc_type, acc_text
            if acc_type is not None and acc_text:
                await writer.write(_event(acc_type, attempt, text=acc_text))
            acc_type, acc_text = None, ""

        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                section = self._stream_section_of(event)
                if section is None:
                    continue
                if section != acc_type:
                    await flush()
                    acc_type = section
                acc_text += event.data.delta or ""
                await flush()
            elif isinstance(event, RunItemStreamEvent):
                await flush()
                item = event.item
                if item.type == "tool_call_item":
                    if item.call_id:
                        call_names[item.call_id] = item.tool_name
                        call_starts[item.call_id] = time.monotonic()
                    await writer.write(
                        _event("tool_call", attempt, tool_name=item.tool_name,
                            params=_parse_args(_arguments_of(item.raw_item)), call_id=item.call_id)
                    )
                    tool_call = tool_calls.get(item.call_id, _NO_CALL_REPORT) if item.call_id else _NO_CALL_REPORT
                    print(f"[{_ts()}] tool_call {item.tool_name} -> task {(tool_call.task_id or '?')[:8]}")
                elif item.type == "tool_call_output_item":
                    name = call_names.get(item.call_id) if item.call_id else None
                    tool_call = tool_calls.get(item.call_id, _NO_CALL_REPORT) if item.call_id else _NO_CALL_REPORT
                    await writer.write(_event("tool_result", attempt, tool_name=name,
                                           task_id=tool_call.task_id, status=tool_call.status,
                                           call_id=item.call_id))
                    started = call_starts.get(item.call_id) if item.call_id else None
                    if started is not None:
                        print(f"[{_ts()}] tool_call {name or '?'} done in {time.monotonic() - started:.1f}s")
                    else:
                        print(f"[{_ts()}] tool_call {name or '?'} done")
        await flush()

    @staticmethod
    def _stream_section_of(event: RawResponsesStreamEvent) -> str | None:
        data = event.data
        if isinstance(data, ResponseTextDeltaEvent):
            return "content"
        if isinstance(data, (ResponseReasoningTextDeltaEvent, ResponseReasoningSummaryTextDeltaEvent)):
            return "thinking"
        return None
