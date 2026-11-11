from __future__ import annotations

import json

import httpx

import app.lib.db.store as store_module
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
