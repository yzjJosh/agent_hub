"""Read, write, and tail agent conversation event logs."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import aiofiles

from app.lib.db import store as db_store
from app.lib.task.task_manager import Task

__all__ = [
    "ConversationEvent",
    "ConversationLogWriter",
    "ToolCallRecord",
    "assemble_tool_calls",
    "build_conversation_event",
    "get_conversation_log_path",
    "parse_conversation_event",
    "read_conversation_events",
    "tail_lines",
]

_CONVERSATION_LOG_FILENAME = "conversation.jsonl"
_TAIL_POLL_SECONDS = 0.05
_UTF8_CONT_MASK = 0xC0
_UTF8_CONT_TAG = 0x80


@dataclass(frozen=True)
class ConversationEvent:
    """One recorded event in an agent run conversation."""

    type: str
    attempt: int
    timestamp: str
    text: str = ""
    tool_name: str | None = None
    params: dict | None = None
    task_id: str | None = None
    call_id: str | None = None
    status: str | None = None


@dataclass(frozen=True)
class ToolCallRecord:
    """One recorded tool call assembled from an agent conversation."""

    task_id: str
    tool_name: str
    params: dict
    status: str
    run_dir: str


def _format_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_conversation_event(
    kind: str,
    attempt: int,
    *,
    text: str = "",
    tool_name: str | None = None,
    params: dict | None = None,
    task_id: str | None = None,
    call_id: str | None = None,
    status: str | None = None,
) -> ConversationEvent:
    """Create a conversation event stamped with the current UTC time.

    Args:
        kind: Event kind name.
        attempt: Attempt number starting at one.
        text: Event text content.
        tool_name: Name of the tool for tool events.
        params: Parsed parameters passed to a tool call.
        task_id: Task id recorded for a tool result.
        call_id: Call identifier linking a tool call and its result.
        status: Final status of a tool result.

    Returns:
        The new conversation event.
    """
    return ConversationEvent(
        type=kind,
        attempt=attempt,
        timestamp=_format_utc_timestamp(),
        text=text,
        tool_name=tool_name,
        params=params,
        task_id=task_id,
        call_id=call_id,
        status=status,
    )


def _serialize_conversation_event(event: ConversationEvent) -> str:
    return json.dumps(
        {
            "type": event.type,
            "attempt": event.attempt,
            "ts": event.timestamp,
            "text": event.text,
            "tool_name": event.tool_name,
            "params": event.params,
            "task_id": event.task_id,
            "call_id": event.call_id,
            "status": event.status,
        },
        ensure_ascii=False,
    )


def parse_conversation_event(line: str) -> ConversationEvent:
    """Parse one serialized conversation log line into an event.

    Args:
        line: One line from a conversation log file.

    Returns:
        The parsed conversation event.
    """
    event_fields = json.loads(line)
    event_fields.pop("cached", None)
    event_fields["timestamp"] = event_fields.pop("ts")
    return ConversationEvent(**event_fields)


class ConversationLogWriter:
    """Append conversation events to a run log file."""

    def __init__(self, path: str) -> None:
        self._path = path

    async def write(self, event: ConversationEvent) -> None:
        """Append one event line to the conversation log.

        Args:
            event: The conversation event to append.
        """
        line = _serialize_conversation_event(event) + "\n"
        async with aiofiles.open(self._path, "a", encoding="utf-8") as log_file:
            await log_file.write(line)


def get_conversation_log_path(task: Task) -> str:
    """Return the conversation log path for a task.

    Args:
        task: The task whose conversation log path is needed.

    Returns:
        The absolute path to the conversation log file.
    """
    run_dir = f"{db_store.RUNS_DIR}/{str(task.id)}"
    return os.path.join(db_store.task_store.data_dir, run_dir, _CONVERSATION_LOG_FILENAME)


async def read_conversation_events(path: str) -> list[ConversationEvent]:
    """Read all recorded conversation events from a log file.

    Args:
        path: Path to the conversation log file.

    Returns:
        The parsed events, or an empty list if the file does not exist.
    """
    try:
        async with aiofiles.open(path, "r", encoding="utf-8") as log_file:
            lines = [line async for line in log_file]
    except FileNotFoundError:
        return []
    return [parse_conversation_event(line) for line in lines if line.strip()]


def assemble_tool_calls(events: list[ConversationEvent]) -> list[ToolCallRecord]:
    """Build tool call records from conversation events.

    Args:
        events: Conversation events in the order they were written.

    Returns:
        One record per tool result that has a task id and a matching call event.
    """
    started_calls: dict[str, ConversationEvent] = {}
    records: list[ToolCallRecord] = []
    for event in events:
        if event.call_id is None:
            continue
        if event.type == "tool_call":
            started_calls[event.call_id] = event
        elif event.type == "tool_result" and event.task_id is not None:
            call_event = started_calls.get(event.call_id)
            if call_event is None or call_event.tool_name is None:
                continue
            records.append(
                ToolCallRecord(
                    task_id=event.task_id,
                    tool_name=call_event.tool_name,
                    params=call_event.params if call_event.params is not None else {},
                    status=event.status if event.status is not None else "",
                    run_dir=f"{db_store.RUNS_DIR}/{event.task_id}",
                )
            )
    return records


async def _read_tail_bytes(path: str, offset: int) -> bytes:
    try:
        async with aiofiles.open(path, "rb") as log_file:
            await log_file.seek(offset)
            return await log_file.read()
    except FileNotFoundError:
        return b""


def _skip_utf8_continuations(chunk: bytes) -> int:
    position = 0
    while position < len(chunk) and (chunk[position] & _UTF8_CONT_MASK) == _UTF8_CONT_TAG:
        position += 1
    return position


async def tail_lines(
    path: str,
    should_stop: Callable[[], Awaitable[bool]],
    poll: float = _TAIL_POLL_SECONDS,
    offset: int = 0,
) -> AsyncGenerator[str, None]:
    """Yield complete lines from a growing file until the stop signal fires.

    Args:
        path: The file to tail.
        should_stop: An async check that ends the tail when it returns True.
        poll: Seconds between reads when no new line is available.
        offset: First byte in the file to read from.

    Yields:
        Each complete line without its trailing newline.
    """
    should_align_utf8 = offset > 0
    while True:
        chunk = await _read_tail_bytes(path, offset)
        position = _skip_utf8_continuations(chunk) if should_align_utf8 and chunk else 0
        if should_align_utf8 and position < len(chunk):
            should_align_utf8 = False
        while True:
            newline_index = chunk.find(b"\n", position)
            if newline_index == -1:
                break
            line = chunk[position:newline_index]
            if line:
                yield line.decode("utf-8", errors="replace")
            position = newline_index + 1
        offset += position
        if await should_stop():
            break
        await asyncio.sleep(poll)
