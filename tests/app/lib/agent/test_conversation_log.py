"""Unit tests for conversation event logging and file tailing."""

from __future__ import annotations

import inspect
import json
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import app.lib.db.store as store_module
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


async def _drain_tail(generator: AsyncIterator[str], collected: list[str]) -> None:
    async for line in generator:
        collected.append(line)


async def _stop_immediately() -> bool:
    return True


class TestConversationEvents:
    """Verify building, serializing, writing, and reading conversation events."""

    def test_build_conversation_event_sets_fields_and_utc_timestamp(self) -> None:
        event = build_conversation_event(
            "tool_call",
            2,
            text="calling",
            tool_name="search_web",
            params={"query": "cats"},
            task_id="task-1",
            call_id="call-1",
            status="completed",
        )
        assert event.type == "tool_call"
        assert event.attempt == 2
        assert event.timestamp.endswith("+00:00")
        assert event.text == "calling"
        assert event.tool_name == "search_web"
        assert event.params == {"query": "cats"}
        assert event.task_id == "task-1"
        assert event.call_id == "call-1"
        assert event.status == "completed"

    async def test_writer_preserves_wire_schema_and_unicode(self, tmp_path: Path) -> None:
        log_path = tmp_path / "conversation.jsonl"
        writer = ConversationLogWriter(str(log_path))
        event = ConversationEvent(
            type="user",
            attempt=1,
            timestamp="2026-01-01T00:00:00+00:00",
            text="héllo 🧠",
        )
        await writer.write(event)

        raw_lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(raw_lines) == 1
        assert "🧠" in raw_lines[0]
        parsed_json = json.loads(raw_lines[0])
        assert parsed_json == {
            "type": "user",
            "attempt": 1,
            "ts": "2026-01-01T00:00:00+00:00",
            "text": "héllo 🧠",
            "tool_name": None,
            "params": None,
            "task_id": None,
            "call_id": None,
            "status": None,
        }

        events = await read_conversation_events(str(log_path))
        assert events == [event]

    def test_parse_conversation_event_ignores_legacy_cached_field(self) -> None:
        legacy_line = json.dumps({
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
        event = parse_conversation_event(legacy_line)
        assert event == ConversationEvent(
            type="tool_result",
            attempt=1,
            timestamp="2026-01-01T00:00:00+00:00",
            text="",
            tool_name="cached_search",
            params={"query": "cats"},
            task_id="task-legacy",
            call_id="call-legacy",
            status="completed",
        )

    async def test_read_conversation_events_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "missing.jsonl"
        assert await read_conversation_events(str(missing_path)) == []

    def test_get_conversation_log_path_points_into_task_run_directory(self) -> None:
        task_id = uuid.uuid4()
        stub_task = MagicMock()
        stub_task.id = task_id
        expected = str(Path(store_module.task_store.data_dir) / "runs" / str(task_id) / "conversation.jsonl")
        assert get_conversation_log_path(stub_task) == expected


class TestAssembleToolCalls:
    """Verify tool call record assembly from conversation events."""

    def test_pairs_matching_call_and_result_and_skips_unmatched_entries(self) -> None:
        events = [
            ConversationEvent(type="user", attempt=1, timestamp="t0", text="go"),
            ConversationEvent(
                type="tool_call", attempt=1, timestamp="t1",
                tool_name="search_web", params={"query": "cats"}, call_id="c1",
            ),
            ConversationEvent(
                type="tool_result", attempt=1, timestamp="t2",
                tool_name="search_web", task_id="task-1", call_id="c1", status="completed",
            ),
            ConversationEvent(
                type="tool_result", attempt=1, timestamp="t3",
                tool_name="orphan_tool", task_id="task-orphan", call_id="missing-call", status="completed",
            ),
            ConversationEvent(
                type="tool_call", attempt=1, timestamp="t4",
                tool_name="rejected_tool", params={"count": 99}, call_id="c2",
            ),
            ConversationEvent(
                type="tool_result", attempt=1, timestamp="t5",
                tool_name="rejected_tool", task_id=None, call_id="c2", status="rejected",
            ),
        ]
        assert assemble_tool_calls(events) == [
            ToolCallRecord(
                task_id="task-1",
                tool_name="search_web",
                params={"query": "cats"},
                status="completed",
                run_dir="runs/task-1",
            )
        ]


class TestTailLines:
    """Verify growing file tailing behavior."""

    async def test_yields_complete_lines(self, tmp_path: Path) -> None:
        log_path = tmp_path / "tail.jsonl"
        log_path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
        lines = [line async for line in tail_lines(str(log_path), _stop_immediately)]
        assert lines == ['{"a": 1}', '{"b": 2}']

    async def test_stops_when_signal_fires(self, tmp_path: Path) -> None:
        log_path = tmp_path / "tail.jsonl"
        log_path.write_text('{"a": 1}\n', encoding="utf-8")
        lines = [line async for line in tail_lines(str(log_path), _stop_immediately)]
        assert lines == ['{"a": 1}']

    async def test_missing_file_yields_nothing(self, tmp_path: Path) -> None:
        log_path = tmp_path / "missing.jsonl"
        lines = [line async for line in tail_lines(str(log_path), _stop_immediately)]
        assert lines == []

    async def test_default_poll_keeps_streaming_lag_low(self) -> None:
        signature = inspect.signature(tail_lines)
        assert list(signature.parameters) == ["path", "should_stop", "poll", "offset"]
        assert signature.parameters["poll"].default <= 0.05
        assert signature.parameters["offset"].default == 0

    async def test_offset_skips_earlier_bytes(self, tmp_path: Path) -> None:
        log_path = tmp_path / "tail.jsonl"
        first_line = '{"text": "héllo 🧠"}\n'.encode("utf-8")
        remaining = b'{"b": 2}\n{"c": 3}\n'
        log_path.write_bytes(first_line + remaining)
        lines = [line async for line in tail_lines(str(log_path), _stop_immediately, offset=len(first_line))]
        assert lines == ['{"b": 2}', '{"c": 3}']

    @pytest.mark.parametrize(
        ("delta", "expected_tail"),
        [
            (1, ['llo 🧠"}', '{"b": 2}']),
            (7, ['"}', '{"b": 2}']),
            (8, ['"}', '{"b": 2}']),
            (9, ['"}', '{"b": 2}']),
        ],
    )
    async def test_offset_mid_utf8_skips_continuation_bytes(
        self,
        tmp_path: Path,
        delta: int,
        expected_tail: list[str],
    ) -> None:
        log_path = tmp_path / "tail.jsonl"
        first_line = '{"text": "héllo 🧠"}\n'.encode("utf-8")
        accent_position = first_line.index("é".encode("utf-8"))
        log_path.write_bytes(first_line + b'{"b": 2}\n')
        lines = [
            line
            async for line in tail_lines(str(log_path), _stop_immediately, offset=accent_position + delta)
        ]
        assert lines == expected_tail

    async def test_holds_partial_trailing_line_until_newline_arrives(self, tmp_path: Path) -> None:
        log_path = tmp_path / "tail.jsonl"
        log_path.write_bytes(b'{"a": 1}\n{"b": "hel')
        seen_lines: list[str] = []
        checks = 0
        lines_at_poll: list[list[str]] = []

        async def stop_after_append() -> bool:
            nonlocal checks
            checks += 1
            lines_at_poll.append(list(seen_lines))
            if checks == 1:
                with log_path.open("ab") as log_file:
                    log_file.write(b'lo"}\n{"c": 3}\n')
                return False
            return True

        await _drain_tail(tail_lines(str(log_path), stop_after_append, poll=0.001), seen_lines)
        assert lines_at_poll[0] == ['{"a": 1}']
        assert seen_lines == ['{"a": 1}', '{"b": "hello"}', '{"c": 3}']

    async def test_ten_thousand_lines_complete_accurately_and_fast(self, tmp_path: Path) -> None:
        log_path = tmp_path / "large.jsonl"
        line_count = 10000
        payload = "".join(f'{{"i": {index}, "msg": "row-{index}"}}\n' for index in range(line_count))
        log_path.write_text(payload, encoding="utf-8")
        started_at = time.monotonic()
        lines = [line async for line in tail_lines(str(log_path), _stop_immediately)]
        elapsed_seconds = time.monotonic() - started_at
        assert len(lines) == line_count
        assert lines[0] == '{"i": 0, "msg": "row-0"}'
        assert lines[5000] == '{"i": 5000, "msg": "row-5000"}'
        assert lines[-1] == f'{{"i": {line_count - 1}, "msg": "row-{line_count - 1}"}}'
        assert elapsed_seconds < 2.0
