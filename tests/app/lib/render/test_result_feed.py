import json, uuid
from datetime import datetime
from pathlib import Path

import pytest

from app.lib.render import result_feed as rf
from app.lib.render.classify import RenderKind
from tests.app.conftest import boom_descriptor, point_share_env


def _seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: bytes,
          files: list[str] | None = None, task_id: str | None = None) -> tuple[str, str, Path]:
    share, data = point_share_env(tmp_path, monkeypatch)
    task_id = task_id or str(uuid.uuid4())
    run_dir = Path(data, "runs", task_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result").write_bytes(body)
    if files:
        lines = "".join(json.dumps({"path": p, "size": None, "mtime": None}) + "\n" for p in files)
        (run_dir / "files").write_text(lines)
    return task_id, str(run_dir), share


async def test_feed_body_and_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir, _share = _seed(tmp_path, monkeypatch, b"hello world", files=["tts/a.opus"])
    feed = await rf.build_result_feed(tid, run_dir)
    assert feed["body"]["kind"] in {"text", "markdown"}
    assert len(feed["files"]) == 1
    f = feed["files"][0]
    assert f["path"] == "tts/a.opus" and f["name"] == "a.opus"
    assert f["present"] is False            # the share file does not exist
    assert f["descriptor"] is None and f["error"] is None


async def test_feed_live_file_wins_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir, share = _seed(tmp_path, monkeypatch, b"hi", files=["tts/a.opus"])
    (share / "tts").mkdir(parents=True)
    (share / "tts" / "a.opus").write_bytes(b"0123")
    feed = await rf.build_result_feed(tid, run_dir)
    f = feed["files"][0]
    mtime = int((share / "tts" / "a.opus").stat().st_mtime)
    assert f["present"] is True and f["error"] is None
    assert f["size"] == 4 and f["descriptor"]["kind"] == "audio"
    assert f["mtime"] == mtime and f["mtime_local"] == datetime.fromtimestamp(mtime).strftime("%b %d %H:%M")


async def test_feed_reports_share_path_for_absolute_sidecar_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file saved with an absolute path in the sidecar is reported by its share path."""
    share = tmp_path / "res"
    tid, run_dir, _ = _seed(tmp_path, monkeypatch, b"hi", files=[str(share / "downloads" / "x.png")])
    (share / "downloads").mkdir(parents=True)
    (share / "downloads" / "x.png").write_bytes(b"\x89PNG")
    feed = await rf.build_result_feed(tid, run_dir)
    f = feed["files"][0]
    assert f["present"] is True and f["error"] is None
    assert f["path"] == "downloads/x.png"
    assert f["descriptor"]["path"] == "downloads/x.png"
    assert f["descriptor"]["url"] == "/dav/downloads/x.png?ephemeral=1"


async def test_feed_missing_result_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    point_share_env(tmp_path, monkeypatch)
    tid = str(uuid.uuid4())
    rd = Path(tmp_path, "data", "runs", tid)
    rd.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        await rf.build_result_feed(tid, str(rd))


async def test_feed_load_error_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A live file whose descriptor build raises → error load_error, siblings/body intact.
    tid, run_dir, share = _seed(tmp_path, monkeypatch, b"ok", files=["bad/x.bin", "tts/ok.opus"])
    (share / "bad").mkdir(parents=True)
    (share / "bad" / "x.bin").write_bytes(b"\x00\x01")

    boom_descriptor("bad/x.bin", monkeypatch)

    feed = await rf.build_result_feed(tid, run_dir)
    by_path = {f["path"]: f for f in feed["files"]}
    assert by_path["bad/x.bin"]["present"] is True
    assert by_path["bad/x.bin"]["error"] == "load_error"
    assert by_path["bad/x.bin"]["descriptor"] is None
    assert by_path["tts/ok.opus"]["present"] is False   # healthy sibling unaffected
    assert feed["body"]["size"] == 2                     # body composed as usual


async def test_feed_without_sidecar_does_not_scan_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A body that quotes a share path yields no files when the run has no sidecar."""
    body = b"The audio is saved as tts/tts_20260905_161034.opus in the resource.\n"
    tid, run_dir, _share = _seed(tmp_path, monkeypatch, body)
    feed = await rf.build_result_feed(tid, run_dir)
    assert feed["files"] == []


async def test_feed_carries_declared_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The declared kind a tool fixed for its results reaches the feed body."""
    tid, run_dir, _share = _seed(tmp_path, monkeypatch, b"def hello():\n    return 1\n")
    feed = await rf.build_result_feed(tid, run_dir, declared_kind=RenderKind.CODE)
    assert feed["body"]["kind"] == "code"
    assert feed["body"]["declared_kind"] == "code"
