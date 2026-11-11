from __future__ import annotations

import asyncio
import os
import struct
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

import app.lib.render.cache as cache
import app.lib.render.gen_state as gen_state
import app.lib.render.thumbs as thumbs
from app.lib.render.classify import RenderKind
from tests.app.console.conftest import console_env

_JOB_WAIT_S = 5


@pytest.fixture
async def thumb_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[httpx.AsyncClient, Path]]:
    """A test client for the thumb api and the share folder."""
    await thumbs.stop_thumbnails()
    async with console_env(tmp_path, monkeypatch) as env:
        yield env


def _seed_png(share: Path, rel: str) -> str:
    def chunk(typ: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", 0)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", b"\x00") + chunk(b"IEND", b""))
    p = share / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(png)
    return rel


def _thumb_job_task(key: str) -> asyncio.Task:
    """The running thumbnail job task for one key."""
    task = next((t for t in asyncio.all_tasks() if t.get_name() == f"thumb-{key[:8]}"), None)
    if task is None:
        raise LookupError(f"no running thumbnail job for key {key[:8]}")
    return task


def _seed_md(share: Path, rel: str) -> str:
    p = share / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# hi\n", encoding="utf-8")
    return rel


async def test_thumb_hit_serves_png(thumb_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, share = thumb_api
    rel = _seed_png(share, "upload/a.png")
    st = (share / rel).stat()
    key = cache.key_for(rel, int(st.st_mtime))
    key_dir = os.path.join(cache.render_root(), key)
    os.makedirs(key_dir, exist_ok=True)
    with open(os.path.join(key_dir, "thumb.png"), "wb") as fh:
        fh.write(b"\x89PNG fake thumb")
    r = await c.get("/api/resources/thumb", params={"path": rel})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content == b"\x89PNG fake thumb"


async def test_thumb_not_ready_409(
    thumb_api: tuple[httpx.AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking for a thumbnail that is not made yet gives 409 not_ready and queues the work."""
    c, share = thumb_api
    rel = _seed_png(share, "upload/b.png")
    key = cache.key_for(rel, int((share / rel).stat().st_mtime))
    hold = asyncio.Event()

    async def held_image(_full: str, _key: str) -> None:
        await hold.wait()

    monkeypatch.setattr(thumbs, "_thumb_image", held_image)
    r = await asyncio.wait_for(c.get("/api/resources/thumb", params={"path": rel}), _JOB_WAIT_S)
    assert r.status_code == 409
    assert r.json() == {"error": f"No thumbnail yet for '{rel}'.", "code": "not_ready"}
    job = _thumb_job_task(key)
    r2 = await asyncio.wait_for(c.post("/api/resources/thumbs", params={"path": "upload"}), _JOB_WAIT_S)
    assert r2.status_code == 200
    assert r2.json()["queued"] == 0  # the GET already queued it
    hold.set()
    await asyncio.wait_for(job, _JOB_WAIT_S)  # let the held job finish


async def test_thumb_path_required_400(thumb_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, _ = thumb_api
    r = await c.get("/api/resources/thumb")
    assert r.status_code == 400
    assert r.json()["code"] == "bad_param"


async def test_thumb_missing_file_404(thumb_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, _ = thumb_api
    r = await c.get("/api/resources/thumb", params={"path": "nope.png"})
    assert r.status_code == 404


async def test_thumbs_dir_returns_queued_count(thumb_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, share = thumb_api
    _seed_png(share, "gallery/a.png")
    _seed_png(share, "gallery/b.png")
    r = await c.post("/api/resources/thumbs", params={"path": "gallery"})
    assert r.status_code == 200
    assert r.json()["queued"] == 2


async def test_thumbs_missing_dir_404(thumb_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, _ = thumb_api
    r = await c.post("/api/resources/thumbs", params={"path": "no-such-dir"})
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


async def test_listing_gains_thumb_field_for_files_only(thumb_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, share = thumb_api
    _seed_png(share, "gallery/a.png")
    (share / "gallery" / "sub").mkdir()
    r = await c.get("/api/resources", params={"path": "gallery"})
    assert r.status_code == 200
    entries = {e["name"]: e for e in r.json()["entries"]}
    assert entries["a.png"]["thumb"]["status"] == "pending"
    assert entries["a.png"]["thumb"]["url"] == "/api/resources/thumb?path=gallery/a.png"
    assert "size" in entries["a.png"]
    assert "thumb" not in entries["sub"]  # folders get no thumb


async def test_thumb_failed_doc_with_published_doc_requeues(
    thumb_api: tuple[httpx.AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A published document with a stale thumb-failed marker is re-queued on poll."""
    c, share = thumb_api
    rel = _seed_md(share, "upload/notes.md")
    st = (share / rel).stat()
    key = cache.key_for(rel, int(st.st_mtime))
    await cache.write_doc(key, '<div class="md-doc">hi</div>', None, {})
    await cache.mark_failed(key, "thumb", "stale failure")

    hold = asyncio.Event()

    async def fake_page(_key: str) -> None:
        await hold.wait()
        key_dir = os.path.join(cache.render_root(), _key)
        os.makedirs(key_dir, exist_ok=True)
        with open(os.path.join(key_dir, "thumb.png"), "wb") as fh:
            fh.write(b"\x89PNG fake thumb")

    monkeypatch.setattr(thumbs, "_render_doc_page", fake_page)
    r = await asyncio.wait_for(c.get("/api/resources/thumb", params={"path": rel}), _JOB_WAIT_S)
    assert r.status_code == 409
    assert key in gen_state.generating_keys()  # the poll re-queued the job
    hold.set()
    await asyncio.wait_for(_thumb_job_task(key), _JOB_WAIT_S)
    assert await cache.thumb_status(key) == "done"
    r2 = await c.get("/api/resources/thumb", params={"path": rel})
    assert r2.status_code == 200
    assert r2.content == b"\x89PNG fake thumb"


async def _poll_failed_key(
    c: httpx.AsyncClient, key: str, rel: str, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mark the key failed, then show a poll comes back 409 and starts no job."""
    await cache.mark_failed(key, "thumb", reason)
    started: list[str] = []
    real_start = thumbs.start_thumbnail_job

    def spy(key2: str, kind: RenderKind, path: str) -> bool:
        started.append(key2)
        return real_start(key2, kind, path)

    monkeypatch.setattr(thumbs, "start_thumbnail_job", spy)
    r = await c.get("/api/resources/thumb", params={"path": rel})
    assert r.status_code == 409
    assert started == []  # the failed key is not retried on poll


async def test_thumb_failed_media_is_not_requeued(
    thumb_api: tuple[httpx.AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed media thumbnail stays 409 and is not re-rendered on every poll."""
    c, share = thumb_api
    rel = _seed_png(share, "upload/c.png")
    st = (share / rel).stat()
    key = cache.key_for(rel, int(st.st_mtime))
    await _poll_failed_key(c, key, rel, "corrupt", monkeypatch)


async def test_thumb_failed_doc_without_page_is_not_requeued(
    thumb_api: tuple[httpx.AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed document thumbnail whose page is not published stays 409 and is not re-rendered."""
    c, share = thumb_api
    rel = _seed_md(share, "upload/notes.md")
    st = (share / rel).stat()
    key = cache.key_for(rel, int(st.st_mtime))
    await _poll_failed_key(c, key, rel, "stale failure", monkeypatch)
