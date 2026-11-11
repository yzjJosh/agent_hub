from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tests.app.conftest import boom_descriptor, seed_task, write_run_file

PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 40
MARKDOWN = b"# Notes\n\nSome **bold** text and a [link](https://example.com).\n\n- one\n- two\n"


async def _seed(tmp_path: Path, body: bytes, task_id: str = "t1") -> str:
    await seed_task(task_id, "web_fetch", status="completed")
    write_run_file(tmp_path / "data", task_id, "result", body)
    return task_id


async def _seed_declared(tmp_path: Path, body: bytes, token: str, task_id: str = "t1") -> str:
    await seed_task(task_id, "stub_rrt", status="completed", render_type=token)
    write_run_file(tmp_path / "data", task_id, "result", body)
    return task_id


def _point_share(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    share = tmp_path / "res"
    share.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RESOURCES_DIR", str(share))
    return share


async def test_result_view_markdown(api_client: httpx.AsyncClient, tmp_path: Path) -> None:
    body = b"# Notes\n\nSome **bold** text and a [link](https://example.com).\n\n- one\n- two\n"
    tid = await _seed(tmp_path, body)
    d = (await api_client.get(f"/api/tasks/{tid}/result/view")).json()
    assert d["kind"] == "markdown" and d["detected_kind"] == "markdown"
    assert d["kind_label"] == "markdown" and d["mode"] == "doc"
    assert d["name"] == tid and d["size"] == len(body)
    assert '<div class="md-doc">' in d["fragment"]
    assert d["line_offset"] is None and d["truncated"] is None
    assert d["declared_kind"] is None


async def test_result_view_result_not_found(api_client: httpx.AsyncClient) -> None:
    await seed_task("t-nores", "web_fetch", status="completed")          # no result file
    r = await api_client.get("/api/tasks/t-nores/result/view")
    assert r.status_code == 404 and r.json()["code"] == "result_not_found"


async def test_result_view_task_not_found(api_client: httpx.AsyncClient) -> None:
    r = await api_client.get("/api/tasks/nope/result/view")
    assert r.status_code == 404 and r.json()["code"] == "not_found"


@pytest.mark.parametrize(
    "params",
    ({"line_offset": "-1"}, {"line_offset": "abc"}, {"kind": "image"}, {"kind": "bogus"}),
)
async def test_result_view_bad_params(api_client: httpx.AsyncClient, tmp_path: Path, params: dict) -> None:
    tid = await _seed(tmp_path, b"# x\n")
    assert (await api_client.get(f"/api/tasks/{tid}/result/view", params=params)).json()["code"] == "bad_param"


async def test_result_view_line_window(api_client: httpx.AsyncClient, tmp_path: Path) -> None:
    body = b"#!/bin/sh\n" + b"".join(f"echo {i}\n".encode() for i in range(1499))    # 1500 lines
    tid = await _seed(tmp_path, body)
    d0 = (await api_client.get(f"/api/tasks/{tid}/result/view")).json()
    assert d0["line_offset"] == 0 and d0["truncated"] is True and d0["line_count"] == 1500
    assert d0["fragment"].startswith("<style>") and d0["fragment"].count('<div class="cl">') == 1000
    d1 = (await api_client.get(f"/api/tasks/{tid}/result/view", params={"line_offset": "1000"})).json()
    assert d1["line_offset"] == 1000 and d1["truncated"] is False
    assert d1["fragment"].count('<div class="cl">') == 500
    assert "<style>" not in d1["fragment"] and "<pre" not in d1["fragment"]


async def test_result_view_legacy_over_http(api_client: httpx.AsyncClient, tmp_path: Path) -> None:
    tid = await _seed(tmp_path, PNG)
    r = await api_client.get(f"/api/tasks/{tid}/result/view")
    assert r.status_code == 200
    d = r.json()
    assert d["mode"] == "legacy" and d["kind"] == "unknown" and d["detected_kind"] == "image"
    assert d["kind_label"] == "file" and d["name"] == "result.bin" and d["size"] == len(PNG)
    assert d["url"] is None and d["fragment"] is None and d["note"] is None


async def test_result_view_zero_byte(api_client: httpx.AsyncClient, tmp_path: Path) -> None:
    tid = await _seed(tmp_path, b"")
    d = (await api_client.get(f"/api/tasks/{tid}/result/view")).json()
    assert d["kind"] == "unknown" and d["kind_label"] == "file" and d["mode"] == "unsupported"
    assert d["fragment"] is None and d["size"] == 0 and d["name"] == tid


async def test_view_declared_kind_beats_detection(api_client: httpx.AsyncClient, tmp_path: Path) -> None:
    tid = await _seed_declared(tmp_path, MARKDOWN, "code")
    r = await api_client.get(f"/api/tasks/{tid}/result/view")
    assert r.status_code == 200
    d = r.json()
    assert d["kind"] == "code"
    assert d["detected_kind"] == "markdown"
    assert d["declared_kind"] == "code"


async def test_view_declared_kind_beats_kind_param(api_client: httpx.AsyncClient, tmp_path: Path) -> None:
    tid = await _seed_declared(tmp_path, MARKDOWN, "code")
    r = await api_client.get(f"/api/tasks/{tid}/result/view", params={"kind": "json"})
    assert r.status_code == 200
    assert r.json()["kind"] == "code"


async def test_feed_declared_kind_propagates(api_client: httpx.AsyncClient, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    _point_share(tmp_path, monkeypatch)
    tid = await _seed_declared(tmp_path, MARKDOWN, "code")
    r = await api_client.get(f"/api/tasks/{tid}/result/feed")
    assert r.status_code == 200
    assert r.json()["body"]["declared_kind"] == "code"


async def test_feed_declared_kind_beats_kind_param(api_client: httpx.AsyncClient, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    _point_share(tmp_path, monkeypatch)
    tid = await _seed_declared(tmp_path, MARKDOWN, "code")
    r = await api_client.get(f"/api/tasks/{tid}/result/feed", params={"kind": "json"})
    assert r.status_code == 200
    assert r.json()["body"]["kind"] == "code"


async def test_detail_result_files_present(api_client: httpx.AsyncClient, tmp_path: Path) -> None:
    tid = await _seed(tmp_path, b"body")
    write_run_file(tmp_path / "data", tid, "files", "downloads/a.png\n")
    assert (await api_client.get(f"/api/tasks/{tid}")).json()["result_files"] == ["downloads/a.png"]


async def test_detail_result_files_absent(api_client: httpx.AsyncClient, tmp_path: Path) -> None:
    tid = await _seed(tmp_path, b"body")
    assert (await api_client.get(f"/api/tasks/{tid}")).json()["result_files"] == []


async def test_result_retry_route_gone(api_client: httpx.AsyncClient) -> None:
    r = await api_client.post("/api/tasks/does-not-matter/result/view/retry")
    assert r.status_code == 404   # falls through to the API 404


async def test_feed_200_with_body_and_files(api_client: httpx.AsyncClient, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    _point_share(tmp_path, monkeypatch)
    tid = await _seed(tmp_path, b"body")
    write_run_file(tmp_path / "data", tid, "files", "downloads/a.png\n")
    r = await api_client.get(f"/api/tasks/{tid}/result/feed")
    assert r.status_code == 200
    data = r.json()
    assert data["body"]["size"] >= 1 and isinstance(data["files"], list)
    assert data["files"][0]["path"] == "downloads/a.png"
    assert data["files"][0]["present"] is False and data["files"][0]["descriptor"] is None


async def test_feed_result_not_found(api_client: httpx.AsyncClient) -> None:
    await seed_task("t-feed-nores", "web_fetch", status="completed")
    r = await api_client.get("/api/tasks/t-feed-nores/result/feed")
    assert r.status_code == 404 and r.json()["code"] == "result_not_found"


async def test_feed_bad_kind(api_client: httpx.AsyncClient, tmp_path: Path,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    _point_share(tmp_path, monkeypatch)
    tid = await _seed(tmp_path, b"body")
    r = await api_client.get(f"/api/tasks/{tid}/result/feed", params={"kind": "pdf"})
    assert r.status_code == 400 and r.json()["code"] == "bad_param"


async def test_feed_load_error_isolated(api_client: httpx.AsyncClient, tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    share = _point_share(tmp_path, monkeypatch)
    tid = await _seed(tmp_path, b"ok")
    write_run_file(tmp_path / "data", tid, "files", "bad/x.bin\ntts/ok.opus\n")
    (share / "bad").mkdir(parents=True, exist_ok=True)
    (share / "bad" / "x.bin").write_bytes(b"\x00\x01")

    boom_descriptor("bad/x.bin", monkeypatch)
    r = await api_client.get(f"/api/tasks/{tid}/result/feed")
    assert r.status_code == 200
    entry = next(f for f in r.json()["files"] if f["path"] == "bad/x.bin")
    assert entry["error"] == "load_error" and entry["descriptor"] is None
    assert r.json()["body"]["size"] == 2
