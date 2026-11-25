from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx
import pytest

import app.lib.db.store as store_module
from app.lib.agent.conversation_log import tail_lines
from tests.app.conftest import seed_task, write_run_file


async def test_stream_agent_task_sends_lines(api_client: httpx.AsyncClient) -> None:
    tid = "stream-agent-1"
    await seed_task(tid, "planner")
    await store_module.agent_store.insert(tid, "planner")
    content = (
        json.dumps({"type": "system", "attempt": 1, "ts": "t", "text": "sp"}) + "\n"
        + json.dumps({"type": "user", "attempt": 1, "ts": "t", "text": "hello"}) + "\n"
    )
    write_run_file(store_module.task_store.data_dir, tid, "conversation.jsonl", content)
    # Mark the task terminal so the stream closes after the first read.
    await store_module.task_store.finish(tid, "completed")
    res = await api_client.get(f"/api/tasks/{tid}/conversation/stream")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/plain")
    assert res.headers.get("x-accel-buffering") == "no"
    body_lines = [json.loads(line) for line in res.text.strip().splitlines() if line]
    assert body_lines[0]["type"] == "system"
    assert body_lines[1]["type"] == "user"


async def test_stream_non_agent_returns_404(api_client: httpx.AsyncClient) -> None:
    await seed_task("not-agent-1", "brave_search", status="completed")
    res = await api_client.get("/api/tasks/not-agent-1/conversation/stream")
    assert res.status_code == 404
    assert res.json()["code"] == "not_agent"


async def test_stream_unknown_task_returns_404(api_client: httpx.AsyncClient) -> None:
    res = await api_client.get("/api/tasks/no-such-task/conversation/stream")
    assert res.status_code == 404
    assert res.json()["code"] == "not_found"


async def test_stream_missing_file_returns_404(api_client: httpx.AsyncClient) -> None:
    tid = "no-convo-1"
    await seed_task(tid, "planner", status="completed")
    await store_module.agent_store.insert(tid, "planner")
    res = await api_client.get(f"/api/tasks/{tid}/conversation/stream")
    assert res.status_code == 404
    assert res.json()["code"] == "conversation_not_found"


async def test_windowed_conversation_read(api_client: httpx.AsyncClient) -> None:
    tid = "windowed-1"
    await seed_task(tid, "planner", status="completed")
    await store_module.agent_store.insert(tid, "planner")
    content = '{"type":"user","attempt":1,"ts":"t","text":"hello"}\n'
    write_run_file(store_module.task_store.data_dir, tid, "conversation.jsonl", content)
    res = await api_client.get(f"/api/tasks/{tid}/conversation", params={"offset": 0})
    assert res.status_code == 200
    body = res.json()
    assert body["content"] == content
    assert body["size"] == len(content)
    assert body["offset"] == 0
    assert body["truncated"] is False


async def test_windowed_conversation_not_agent_returns_404(api_client: httpx.AsyncClient) -> None:
    await seed_task("no-agent-w", "brave_search", status="completed")
    res = await api_client.get("/api/tasks/no-agent-w/conversation", params={"offset": 0})
    assert res.status_code == 404
    assert res.json()["code"] == "not_agent"


async def test_windowed_conversation_bad_param_returns_400(api_client: httpx.AsyncClient) -> None:
    tid = "bad-param-1"
    await seed_task(tid, "planner", status="completed")
    await store_module.agent_store.insert(tid, "planner")
    write_run_file(store_module.task_store.data_dir, tid, "conversation.jsonl", '{"type":"user","attempt":1,"ts":"t","text":"hi"}\n')
    res = await api_client.get(f"/api/tasks/{tid}/conversation", params={"offset": "1099511627777"})
    assert res.status_code == 400
    assert res.json() == {
        "error": "offset must be an integer between 0 and 1099511627776 and max_bytes a positive integer.",
        "code": "bad_param",
    }


def _jsonl(events: list[dict]) -> str:
    return "".join(json.dumps(ev, ensure_ascii=False) + "\n" for ev in events)


async def _collect(gen: AsyncIterator[str], out: list[str]) -> None:
    async for item in gen:
        out.append(item)


async def test_stream_offset_sends_only_lines_from_offset(api_client: httpx.AsyncClient) -> None:
    tid = "stream-offset-1"
    await seed_task(tid, "planner")
    await store_module.agent_store.insert(tid, "planner")
    first = _jsonl([
        {"type": "system", "attempt": 1, "ts": "t0", "text": "sp 🧠"},
        {"type": "user", "attempt": 1, "ts": "t1", "text": "héllo"},
    ])
    later = _jsonl([
        {"type": "content", "attempt": 1, "ts": "t2", "text": "world"},
        {"type": "assistant", "attempt": 1, "ts": "t3", "text": "world"},
    ])
    full = first + later
    write_run_file(store_module.task_store.data_dir, tid, "conversation.jsonl", full)
    await store_module.task_store.finish(tid, "completed")

    first_bytes = len(first.encode("utf-8"))
    res = await api_client.get(f"/api/tasks/{tid}/conversation/stream", params={"offset": first_bytes})
    assert res.status_code == 200
    parsed = [json.loads(line) for line in res.text.splitlines() if line]
    assert parsed == [
        {"type": "content", "attempt": 1, "ts": "t2", "text": "world"},
        {"type": "assistant", "attempt": 1, "ts": "t3", "text": "world"},
    ]

    eof_res = await api_client.get(
        f"/api/tasks/{tid}/conversation/stream", params={"offset": len(full.encode("utf-8"))}
    )
    assert eof_res.status_code == 200
    assert eof_res.text == ""


@pytest.mark.parametrize("bad_offset", ["bad", "-1", "1099511627777"])
async def test_stream_bad_offset_returns_400(api_client: httpx.AsyncClient, bad_offset: str) -> None:
    tid = "stream-bad-offset"
    await seed_task(tid, "planner", status="completed")
    await store_module.agent_store.insert(tid, "planner")
    write_run_file(
        store_module.task_store.data_dir, tid, "conversation.jsonl",
        '{"type":"user","attempt":1,"ts":"t","text":"hi"}\n',
    )
    res = await api_client.get(f"/api/tasks/{tid}/conversation/stream", params={"offset": bad_offset})
    assert res.status_code == 400
    assert res.json() == {
        "error": "offset must be an integer between 0 and 1099511627776.",
        "code": "bad_param",
    }


async def test_stream_missing_or_non_agent_with_bad_offset_returns_404(api_client: httpx.AsyncClient) -> None:
    unknown = await api_client.get("/api/tasks/no-such-task/conversation/stream", params={"offset": "bad"})
    assert unknown.status_code == 404
    assert unknown.json()["code"] == "not_found"

    await seed_task("not-agent-bad-off", "brave_search", status="completed")
    non_agent = await api_client.get("/api/tasks/not-agent-bad-off/conversation/stream", params={"offset": "bad"})
    assert non_agent.status_code == 404
    assert non_agent.json()["code"] == "not_agent"

    await seed_task("no-file-bad-off", "planner", status="completed")
    await store_module.agent_store.insert("no-file-bad-off", "planner")
    missing_file = await api_client.get("/api/tasks/no-file-bad-off/conversation/stream", params={"offset": "bad"})
    assert missing_file.status_code == 404
    assert missing_file.json()["code"] == "conversation_not_found"


async def test_stream_mid_utf8_offset_skips_continuation_bytes(api_client: httpx.AsyncClient) -> None:
    tid = "stream-mid-utf8"
    await seed_task(tid, "planner", status="completed")
    await store_module.agent_store.insert(tid, "planner")
    first = _jsonl([{"type": "user", "attempt": 1, "ts": "t0", "text": "sp 🧠"}])
    second = _jsonl([{"type": "assistant", "attempt": 1, "ts": "t1", "text": "ok"}])
    write_run_file(store_module.task_store.data_dir, tid, "conversation.jsonl", first + second)
    mid_emoji = first.encode("utf-8").index("🧠".encode("utf-8")) + 1
    res = await api_client.get(f"/api/tasks/{tid}/conversation/stream", params={"offset": mid_emoji})
    assert res.status_code == 200
    lines = res.text.splitlines()
    assert lines[0] == '"}'
    assert json.loads(lines[1]) == {"type": "assistant", "attempt": 1, "ts": "t1", "text": "ok"}


_SNAPSHOT_RAW_EVENTS = [
    {"type": "system", "attempt": 1, "ts": "t0", "text": "sys prompt"},
    {"type": "user", "attempt": 1, "ts": "t1", "text": "find 🧠"},
    {"type": "thinking", "attempt": 1, "ts": "t2", "text": "plan "},
    {"type": "thinking", "attempt": 1, "ts": "t3", "text": "step 1"},
    {"type": "tool_call", "attempt": 1, "ts": "t4", "tool_name": "brave_search", "call_id": "c1", "params": {"q": "x"}},
    {"type": "tool_result", "attempt": 1, "ts": "t5", "tool_name": "brave_search", "call_id": "c1", "task_id": "sub-1", "status": "completed"},
    {"type": "thinking", "attempt": 1, "ts": "t6", "text": "after "},
    {"type": "thinking", "attempt": 1, "ts": "t7", "text": "tool"},
    {"type": "content", "attempt": 1, "ts": "t8", "text": "bad "},
    {"type": "content", "attempt": 1, "ts": "t9", "text": "draft"},
    {"type": "assistant", "attempt": 1, "ts": "t10", "text": "bad draft"},
    {"type": "user", "attempt": 1, "ts": "t11", "text": "retry feedback"},
    {"type": "thinking", "attempt": 2, "ts": "t12", "text": "rethink "},
    {"type": "thinking", "attempt": 2, "ts": "t13", "text": "now"},
    {"type": "thinking", "attempt": 3, "ts": "t14", "text": "attempt 3"},
    {"type": "content", "attempt": 3, "ts": "t15", "text": "good "},
    {"type": "content", "attempt": 3, "ts": "t16", "text": "answer"},
]

_SNAPSHOT_COMPACTED_EVENTS = [
    {"type": "system", "attempt": 1, "ts": "t0", "text": "sys prompt"},
    {"type": "user", "attempt": 1, "ts": "t1", "text": "find 🧠"},
    {"type": "thinking", "attempt": 1, "ts": "t2", "text": "plan step 1"},
    {"type": "tool_call", "attempt": 1, "ts": "t4", "tool_name": "brave_search", "call_id": "c1", "params": {"q": "x"}},
    {"type": "tool_result", "attempt": 1, "ts": "t5", "tool_name": "brave_search", "call_id": "c1", "task_id": "sub-1", "status": "completed"},
    {"type": "thinking", "attempt": 1, "ts": "t6", "text": "after tool"},
    {"type": "content", "attempt": 1, "ts": "t8", "text": "bad draft"},
    {"type": "assistant", "attempt": 1, "ts": "t10", "text": "bad draft"},
    {"type": "user", "attempt": 1, "ts": "t11", "text": "retry feedback"},
    {"type": "thinking", "attempt": 2, "ts": "t12", "text": "rethink now"},
    {"type": "thinking", "attempt": 3, "ts": "t14", "text": "attempt 3"},
    {"type": "content", "attempt": 3, "ts": "t15", "text": "good answer"},
]


async def test_snapshot_coalesces_deltas_and_aligns_with_stream(api_client: httpx.AsyncClient) -> None:
    tid = "snap-coalesce-1"
    await seed_task(tid, "planner")
    await store_module.agent_store.insert(tid, "planner")
    initial_text = _jsonl(_SNAPSHOT_RAW_EVENTS)
    conv_path = write_run_file(store_module.task_store.data_dir, tid, "conversation.jsonl", initial_text)

    snap_res = await api_client.get(f"/api/tasks/{tid}/conversation/snapshot")
    assert snap_res.status_code == 200
    assert snap_res.headers.get("cache-control") == "no-store"
    snap = snap_res.json()
    expected_offset = len(initial_text.encode("utf-8"))
    assert expected_offset > len(initial_text)
    assert snap["offset"] == expected_offset
    assert snap["size"] == expected_offset
    assert snap["event_count"] == len(_SNAPSHOT_RAW_EVENTS)
    assert snap["events"] == _SNAPSHOT_COMPACTED_EVENTS

    appended = [
        {"type": "content", "attempt": 3, "ts": "t17", "text": "!"},
        {"type": "assistant", "attempt": 3, "ts": "t18", "text": "good answer!"},
    ]
    with conv_path.open("a", encoding="utf-8") as fh:
        fh.write(_jsonl(appended))
    await store_module.task_store.finish(tid, "completed")

    stream_res = await api_client.get(
        f"/api/tasks/{tid}/conversation/stream", params={"offset": snap["offset"]}
    )
    assert stream_res.status_code == 200
    streamed = [json.loads(line) for line in stream_res.text.splitlines() if line]
    assert streamed == appended


async def _tail_after_completing_partial_line(conv_path: os.PathLike[str], offset: int) -> tuple[list[str], list[str]]:
    tailed: list[str] = []
    before_newline: list[list[str]] = []
    polls = 0

    async def stop_after_completion() -> bool:
        nonlocal polls
        polls += 1
        before_newline.append(list(tailed))
        if polls == 1:
            with open(conv_path, "a", encoding="utf-8") as fh:
                fh.write('lo"}\n')
            return False
        return True

    await _collect(tail_lines(os.fspath(conv_path), stop_after_completion, poll=0.001, offset=offset), tailed)
    return before_newline[0], tailed


async def test_snapshot_ignores_partial_trailing_line_until_completed(api_client: httpx.AsyncClient) -> None:
    tid = "snap-partial-1"
    await seed_task(tid, "planner")
    await store_module.agent_store.insert(tid, "planner")
    complete = _jsonl([
        {"type": "system", "attempt": 1, "ts": "t0", "text": "sp"},
        {"type": "user", "attempt": 1, "ts": "t1", "text": "go"},
    ])
    partial_head = '{"type": "content", "attempt": 1, "ts": "t2", "text": "hel'
    conv_path = write_run_file(
        store_module.task_store.data_dir, tid, "conversation.jsonl", complete + partial_head
    )

    snap_res = await api_client.get(f"/api/tasks/{tid}/conversation/snapshot")
    assert snap_res.status_code == 200
    snap = snap_res.json()
    assert snap["offset"] == len(complete.encode("utf-8"))
    assert snap["size"] == len((complete + partial_head).encode("utf-8"))
    assert snap["event_count"] == 2
    assert [e["type"] for e in snap["events"]] == ["system", "user"]

    first_batch, tailed = await _tail_after_completing_partial_line(conv_path, snap["offset"])
    assert first_batch == []
    assert [json.loads(line) for line in tailed] == [
        {"type": "content", "attempt": 1, "ts": "t2", "text": "hello"}
    ]

    await store_module.task_store.finish(tid, "completed")
    stream_res = await api_client.get(
        f"/api/tasks/{tid}/conversation/stream", params={"offset": snap["offset"]}
    )
    assert stream_res.status_code == 200
    assert [json.loads(line) for line in stream_res.text.splitlines() if line] == [
        {"type": "content", "attempt": 1, "ts": "t2", "text": "hello"}
    ]


async def test_snapshot_only_partial_first_line_returns_zero_offset(api_client: httpx.AsyncClient) -> None:
    tid = "snap-partial-first"
    await seed_task(tid, "planner")
    await store_module.agent_store.insert(tid, "planner")
    partial_first = '{"type": "system", "attempt": 1'
    write_run_file(store_module.task_store.data_dir, tid, "conversation.jsonl", partial_first)
    res = await api_client.get(f"/api/tasks/{tid}/conversation/snapshot")
    assert res.status_code == 200
    assert res.json() == {"offset": 0, "size": len(partial_first.encode("utf-8")), "event_count": 0, "events": []}


async def test_snapshot_missing_file_returns_empty_200(api_client: httpx.AsyncClient) -> None:
    tid = "snap-empty-1"
    await seed_task(tid, "planner")
    await store_module.agent_store.insert(tid, "planner")
    res = await api_client.get(f"/api/tasks/{tid}/conversation/snapshot")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    assert res.json() == {"offset": 0, "size": 0, "event_count": 0, "events": []}


async def test_snapshot_missing_or_non_agent_task_returns_404_and_wrong_method_405(
    api_client: httpx.AsyncClient,
) -> None:
    missing = await api_client.get("/api/tasks/no-such-agent/conversation/snapshot")
    assert missing.status_code == 404
    assert missing.json()["code"] == "not_found"

    await seed_task("tool-only-1", "brave_search", status="completed")
    non_agent = await api_client.get("/api/tasks/tool-only-1/conversation/snapshot")
    assert non_agent.status_code == 404
    assert non_agent.json()["code"] == "not_agent"

    wrong_method = await api_client.post("/api/tasks/tool-only-1/conversation/snapshot")
    assert wrong_method.status_code == 405
    assert wrong_method.json()["code"] == "method_not_allowed"


async def test_snapshot_skips_malformed_lines_and_normalizes_null_text(api_client: httpx.AsyncClient) -> None:
    tid = "snap-malformed-1"
    await seed_task(tid, "planner", status="completed")
    await store_module.agent_store.insert(tid, "planner")
    raw = (
        '{"type": "system", "attempt": 1, "ts": "t0", "text": "sp"}\n'
        "\n"
        "{broken json\n"
        "[1, 2, 3]\n"
        '{"type": "thinking", "attempt": 1, "ts": "t1", "text": null}\n'
        '{"type": "thinking", "attempt": 1, "ts": "t2", "text": "ok"}\n'
    )
    write_run_file(store_module.task_store.data_dir, tid, "conversation.jsonl", raw)
    res = await api_client.get(f"/api/tasks/{tid}/conversation/snapshot")
    assert res.status_code == 200
    body = res.json()
    assert body["offset"] == len(raw.encode("utf-8"))
    assert body["size"] == len(raw.encode("utf-8"))
    assert body["event_count"] == 3
    assert body["events"] == [
        {"type": "system", "attempt": 1, "ts": "t0", "text": "sp"},
        {"type": "thinking", "attempt": 1, "ts": "t1", "text": "ok"},
    ]
