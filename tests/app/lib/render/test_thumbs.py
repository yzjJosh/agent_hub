"""Tests for thumbnail making."""

import asyncio
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import override

import pytest
import uvicorn
from PIL import Image

import app.lib.db.store as store_module
import app.lib.render.cache as cache_module
import app.lib.render.gen_state as gen_state
import app.lib.render.office as office
import app.lib.render.thumbs as thumbs
from app.lib.browser.browser_manager import BrowserManager
from app.lib.render.classify import RenderKind
from app.server import build_app


FIX = Path(__file__).parent / "fixtures"
_THUMB_W = 1024
_THUMB_H = 640
_POLL_S = 0.05
# How long the second render may take to join the first inside the cap.
_OVERLAP_TIMEOUT_S = 5


def _live_jobs() -> set[asyncio.Task]:
    """The set of thumbnail job tasks running right now."""
    return {
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("thumb-") and not task.done()
    }


async def _await_empty_live_jobs() -> None:
    """Wait until no thumbnail job task is running."""
    for _ in range(100):
        if not _live_jobs():
            break
        await asyncio.sleep(_POLL_S)


@pytest.fixture
def _thumb_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give the test its own data dir."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture
def _share(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give the test its own share dir."""
    root = tmp_path / "share"
    root.mkdir()
    monkeypatch.setenv("RESOURCES_DIR", str(root))
    return root


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (RenderKind.IMAGE, "image"),
        (RenderKind.VIDEO, "video"),
        (RenderKind.PDF, "pdf"),
        (RenderKind.OFFICE_PPT, "ppt_pdf"),
        (RenderKind.MARKDOWN, "doc"),
        (RenderKind.CODE, "doc"),
        (RenderKind.CSV, "doc"),
        (RenderKind.JSON, "doc"),
        (RenderKind.HTML, "doc"),
        (RenderKind.OFFICE_DOC, "doc"),
        (RenderKind.XLSX, "doc"),
        (RenderKind.TEXT, "doc"),
        (RenderKind.AUDIO, None),
        (RenderKind.ARCHIVE, None),
        (RenderKind.UNKNOWN, None),
    ],
)
def test_engine_selection(kind: RenderKind, expected: str | None) -> None:
    assert thumbs.thumb_engine(kind) == expected


def test_presentation_thumbnail_reads_converted_pdf(_thumb_env: Path, _share: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """A PowerPoint thumbnail uses the saved PDF after conversion completes."""
    (_share / "deck.pptx").write_bytes(b"PK\x03\x04 fake pptx")
    key = "f" * 64
    seen: list[str] = []

    async def fake_pdf(full_path: str, _key: str) -> None:
        seen.append(full_path)

    monkeypatch.setattr(office, "ensure_doc_for_thumb", _published)
    monkeypatch.setattr(thumbs, "_thumb_pdf", fake_pdf)

    async def run() -> None:
        try:
            assert thumbs.start_thumbnail_job(key, RenderKind.OFFICE_PPT, "deck.pptx") is True
            await _await_empty_live_jobs()
        finally:
            await thumbs.stop_thumbnails()

    asyncio.run(run())
    assert seen == [str(_thumb_env / "render" / key / cache_module.OFFICE_PDF_ASSET)]


def test_start_job_idempotent(_thumb_env: Path, _share: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second start for an in-flight key is a no-op, whatever the generating set says."""
    (_share / "a.png").write_bytes((FIX / "small.png").read_bytes())
    key = cache_module.key_for("a.png", int(os.stat(_share / "a.png").st_mtime))
    hold = asyncio.Event()

    async def held_image(_full: str, _key: str) -> None:
        await hold.wait()

    monkeypatch.setattr(thumbs, "_thumb_image", held_image)

    async def run() -> None:
        try:
            assert thumbs.start_thumbnail_job(key, RenderKind.IMAGE, "a.png") is True
            assert len(_live_jobs()) == 1  # the spawn is held in the live-job set
            assert thumbs.start_thumbnail_job(key, RenderKind.IMAGE, "a.png") is False
            gen_state.gen_add(key)
            assert thumbs.start_thumbnail_job(key, RenderKind.IMAGE, "a.png") is False
            gen_state.gen_remove(key)
            assert thumbs.start_thumbnail_job(key, RenderKind.AUDIO, "a.png") is False  # no engine
            hold.set()
            await _await_empty_live_jobs()  # let the held job finish before the loop closes
            assert _live_jobs() == set()
        finally:
            await thumbs.stop_thumbnails()

    asyncio.run(run())


def _copy_to_share(share: Path, name: str) -> tuple[str, str]:
    """Put a fixture file in the share folder and give back its path and saved name."""
    (share / name).write_bytes((FIX / name).read_bytes())
    return name, cache_module.key_for(name, int(os.stat(share / name).st_mtime))


def test_image_engine_emits_1024x640(_thumb_env: Path, _share: Path) -> None:
    """A started job turns an image into a 1024x640 PNG."""
    path, key = _copy_to_share(_share, "small.png")

    async def run() -> None:
        try:
            assert thumbs.start_thumbnail_job(key, RenderKind.IMAGE, path) is True
            assert await _await_thumb_status(key) == "done"
        finally:
            await thumbs.stop_thumbnails()

    asyncio.run(run())
    with Image.open(os.path.join(cache_module.render_root(), key, "thumb.png")) as im:
        assert im.format == "PNG"
        assert im.size == (_THUMB_W, _THUMB_H)


def test_pdf_engine_emits_1024x640(_thumb_env: Path, _share: Path) -> None:
    """A started job turns a PDF into a 1024x640 picture of its first page."""
    path, key = _copy_to_share(_share, "one.pdf")

    async def run() -> None:
        try:
            assert thumbs.start_thumbnail_job(key, RenderKind.PDF, path) is True
            assert await _await_thumb_status(key) == "done"
        finally:
            await thumbs.stop_thumbnails()

    asyncio.run(run())
    with Image.open(os.path.join(cache_module.render_root(), key, "thumb.png")) as im:
        assert im.size == (_THUMB_W, _THUMB_H)


def test_job_releases_entries_and_marks_failure(_thumb_env: Path, _share: Path) -> None:
    """A failing job writes a failure mark and releases both in-flight entries."""

    async def run() -> None:
        key = "d" * 64
        (_share / "bad.png").write_bytes(b"not a png")
        try:
            assert thumbs.start_thumbnail_job(key, RenderKind.IMAGE, "bad.png") is True
            assert await _await_thumb_failed(key) == "failed"
            assert key not in gen_state.generating_keys()
            assert thumbs.start_thumbnail_job(key, RenderKind.IMAGE, "bad.png") is True  # the key was released
            await _await_empty_live_jobs()  # let the restart finish before the loop closes
        finally:
            await thumbs.stop_thumbnails()

    asyncio.run(run())


def test_live_job_set_releases_on_completion(_thumb_env: Path, _share: Path) -> None:
    """A finished job leaves the live-job set and the enqueued set."""

    async def run() -> None:
        key = "9" * 64
        (_share / "gone.png").write_bytes(b"not a png")
        assert thumbs.start_thumbnail_job(key, RenderKind.IMAGE, "gone.png") is True
        assert len(_live_jobs()) == 1
        assert await _await_thumb_failed(key) == "failed"
        await _await_empty_live_jobs()  # the done-callback fires on the next loop tick
        assert _live_jobs() == set()
        assert key not in gen_state.generating_keys()
        assert thumbs.start_thumbnail_job(key, RenderKind.IMAGE, "gone.png") is True  # the key was released
        await _await_empty_live_jobs()  # let the restart finish before the loop closes

    asyncio.run(run())


def test_parallel_jobs_reach_render_cap(_thumb_env: Path, _share: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Two image jobs both reach the render cap, so one job does not hold the line."""
    first, key_a = _copy_to_share(_share, "small.png")
    second = "second.png"
    (_share / second).write_bytes((FIX / "small.png").read_bytes())
    key_b = cache_module.key_for(second, int(os.stat(_share / second).st_mtime))
    stats = {"now": 0, "peak": 0}
    both_in = asyncio.Event()
    release = asyncio.Event()

    async def slow_image(_full: str, _key: str) -> None:
        stats["now"] += 1
        stats["peak"] = max(stats["peak"], stats["now"])
        if stats["now"] == 2:
            both_in.set()
        await release.wait()
        stats["now"] -= 1
        key_dir = os.path.join(cache_module.render_root(), _key)
        os.makedirs(key_dir, exist_ok=True)
        with open(os.path.join(key_dir, "thumb.png"), "wb") as fh:
            fh.write(b"\x89PNG fake")

    monkeypatch.setattr(thumbs, "_thumb_image", slow_image)

    async def run() -> None:
        try:
            thumbs.start_thumbnail_job(key_a, RenderKind.IMAGE, first)
            thumbs.start_thumbnail_job(key_b, RenderKind.IMAGE, second)
            await asyncio.wait_for(both_in.wait(), _OVERLAP_TIMEOUT_S)  # both renders are inside the cap at once
            release.set()
            assert await _await_thumb_status(key_a) == "done"
            assert await _await_thumb_status(key_b) == "done"
        finally:
            await thumbs.stop_thumbnails()

    asyncio.run(run())
    assert stats["peak"] >= 2  # the renders overlapped behind the cap


def test_generating_held_across_prepare(_thumb_env: Path, _share: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """A doc job sitting in its prepare stage still reads generating, not pending."""
    (_share / "a.md").write_text("# hi\n")
    key = cache_module.key_for("a.md", int(os.stat(_share / "a.md").st_mtime))
    release = asyncio.Event()

    async def slow_prepare(_kind: RenderKind, _key: str, _path: str, _full: str) -> str:
        await release.wait()
        return "published"

    async def fake_page(_key: str) -> None:
        key_dir = os.path.join(cache_module.render_root(), _key)
        os.makedirs(key_dir, exist_ok=True)
        with open(os.path.join(key_dir, "thumb.png"), "wb") as fh:
            fh.write(b"\x89PNG fake")

    monkeypatch.setattr(office, "ensure_doc_for_thumb", slow_prepare)
    monkeypatch.setattr(thumbs, "_render_doc_page", fake_page)

    async def run() -> None:
        try:
            assert thumbs.start_thumbnail_job(key, RenderKind.OFFICE_DOC, "a.md") is True
            assert await cache_module.thumb_status(key) == "generating"  # the prepare wait is in flight
            release.set()
            assert await _await_thumb_status(key) == "done"
        finally:
            await thumbs.stop_thumbnails()

    asyncio.run(run())


async def test_start_dir_thumbnails_counts_pending_and_failed(
    _thumb_env: Path, _share: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Starting a folder's thumbnails counts the files that still need one."""

    async def no_browser() -> _FakeManager:
        raise RuntimeError("no browser in unit tests")

    monkeypatch.setattr(thumbs, "BrowserManager", SimpleNamespace(create=no_browser))
    (_share / "a.md").write_text("# hi\n")
    (_share / "b.opus").write_bytes(b"audio")
    key_a = cache_module.key_for("a.md", int(os.stat(_share / "a.md").st_mtime))
    await cache_module.mark_failed(key_a, "thumb", "x")  # a failed thumbnail is started again
    try:
        n = await thumbs.start_dir_thumbnails("")
        assert n == 1  # only a.md needs one, b.opus gets no thumbnail
        assert thumbs.start_thumbnail_job(key_a, RenderKind.MARKDOWN, "a.md") is False  # already started
        assert await _await_thumb_failed(key_a) == "failed"
    finally:
        await thumbs.stop_thumbnails()


async def test_start_dir_thumbnails_subdirectory_uses_share_relative_paths(
    _thumb_env: Path, _share: Path
) -> None:
    """Starting a subfolder makes thumbnails for the files inside it."""
    sub = _share / "upload" / "sub"
    sub.mkdir(parents=True)
    photo = sub / "photo.png"
    photo.write_bytes((FIX / "small.png").read_bytes())
    key = cache_module.key_for("upload/sub/photo.png", int(os.stat(photo).st_mtime))
    assert await thumbs.start_dir_thumbnails("upload/sub") == 1
    try:
        assert await _await_thumb_status(key) == "done"
    finally:
        await thumbs.stop_thumbnails()


async def _await_thumb_status(key: str) -> str:
    """Wait for a thumbnail to finish and give back how it ended."""
    for _ in range(100):
        status = await cache_module.thumb_status(key)
        if status == "done":
            return status
        await asyncio.sleep(_POLL_S)
    return await cache_module.thumb_status(key)


async def _await_thumb_failed(key: str) -> str:
    """Wait for a thumbnail job to give up and give back how it ended."""
    for _ in range(100):
        status = await cache_module.thumb_status(key)
        if status == "failed":
            return status
        await asyncio.sleep(_POLL_S)
    return await cache_module.thumb_status(key)


class _FakeManager(BrowserManager):
    """A fake manager that records every call it gets and writes a real screenshot."""

    def __init__(self, calls: list[str], navigate_ok: bool, viewport_ok: bool, new_page_error: Exception | None,
                 urls: list[str]) -> None:
        super().__init__()
        self._calls = calls
        self._urls = urls
        self._navigate_ok = navigate_ok
        self._viewport_ok = viewport_ok
        self._new_page_error = new_page_error

    @override
    async def new_page(self) -> str:
        self._calls.append("new_page")
        if self._new_page_error is not None:
            raise self._new_page_error
        return "tab1"

    @override
    async def set_viewport_size(self, _target_id: str, _width: int, _height: int) -> bool:
        self._calls.append("set_viewport_size")
        return self._viewport_ok

    @override
    async def navigate(self, _target_id: str, url: str, **_kwargs: str | int) -> bool:
        self._calls.append("navigate")
        self._urls.append(url)
        return self._navigate_ok

    @override
    async def screenshot(self, _target_id: str, path: str, **_kwargs: bool) -> str | None:
        self._calls.append("screenshot")
        im = Image.new("RGB", (128, 80), "red")
        im.save(path, "JPEG")
        return path

    @override
    async def close_page(self, _target_id: str) -> bool:
        self._calls.append("close_page")
        return True

    @override
    async def cleanup(self) -> None:
        self._calls.append("cleanup")


def _fake_manager_factory(navigate_ok: bool = True, viewport_ok: bool = True,
                          new_page_error: Exception | None = None,
                          urls: list[str] | None = None) -> tuple[list[str], Callable[[], _FakeManager]]:
    """Return the call list and a factory for the fake BrowserManager used by thumbs.

    Every URL the fake is told to open is added to urls when it is given.
    """
    calls: list[str] = []
    opened = urls if urls is not None else []

    async def factory() -> _FakeManager:
        calls.append("create")
        return _FakeManager(calls, navigate_ok, viewport_ok, new_page_error, opened)

    return calls, factory


def _patch_browser(monkeypatch: pytest.MonkeyPatch, factory: Callable[[], _FakeManager]) -> None:
    """Point the thumbs module at the fake BrowserManager."""
    monkeypatch.setattr(thumbs, "BrowserManager", SimpleNamespace(create=factory))


def _set_bind_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    """Clear the bind variables, then set the ones in env."""
    for name in ("HTTP_HOST", "HTTP_PORT", "SSE_HOST", "SSE_PORT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


async def _published(_kind: RenderKind, _key: str, _path: str, _full: str) -> str:
    return "published"


def test_browser_create_failure_marks_job_failed(_thumb_env: Path, _share: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """A connect that finds no browser windows leaves a failure mark."""

    async def no_browser() -> _FakeManager:
        raise RuntimeError("No browser contexts found at http://localhost:9222")

    monkeypatch.setattr(thumbs, "BrowserManager", SimpleNamespace(create=no_browser))
    monkeypatch.setattr(office, "ensure_doc_for_thumb", _published)
    (_share / "a.md").write_text("# hi\n")

    async def run() -> None:
        try:
            key = "c" * 64
            thumbs.start_thumbnail_job(key, RenderKind.MARKDOWN, "a.md")
            assert await _await_thumb_failed(key) == "failed"
        finally:
            await thumbs.stop_thumbnails()

    asyncio.run(run())


def _run_doc_render_job(_thumb_env: Path, _share: Path, monkeypatch: pytest.MonkeyPatch,
                        factory: Callable[[], _FakeManager]) -> tuple[str, str]:
    """Drive one doc render job to a terminal state and give back how it ended and its key."""
    _patch_browser(monkeypatch, factory)
    monkeypatch.setattr(office, "ensure_doc_for_thumb", _published)
    (_share / "a.md").write_text("# hi\n")
    key = cache_module.key_for("a.md", int(os.stat(_share / "a.md").st_mtime))

    async def run() -> str:
        try:
            assert thumbs.start_thumbnail_job(key, RenderKind.MARKDOWN, "a.md") is True
            for _ in range(100):
                status = await cache_module.thumb_status(key)
                if status in ("done", "failed"):
                    return status
                await asyncio.sleep(_POLL_S)
            return status
        finally:
            await thumbs.stop_thumbnails()

    return asyncio.run(run()), key


def test_doc_render_browser_call_sequence(_thumb_env: Path, _share: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """A doc render drives one browser manager from connect to cleanup."""
    calls, factory = _fake_manager_factory()
    status, key = _run_doc_render_job(_thumb_env, _share, monkeypatch, factory)
    assert status == "done"
    assert calls == ["create", "new_page", "set_viewport_size", "navigate", "screenshot", "close_page", "cleanup"]
    with Image.open(os.path.join(cache_module.render_root(), key, "thumb.png")) as im:
        assert im.format == "PNG"
        assert im.size == (_THUMB_W, _THUMB_H)


@pytest.mark.parametrize(
    ("env", "base"),
    [
        ({"HTTP_PORT": "8080"}, "http://127.0.0.1:8080"),
        ({"HTTP_HOST": "10.0.0.5"}, "http://10.0.0.5:80"),
        ({"HTTP_HOST": "::", "HTTP_PORT": "8080"}, "http://127.0.0.1:8080"),
        ({"HTTP_HOST": "::1"}, "http://[::1]:80"),
    ],
    ids=["wildcard-default-host", "specific-host", "ipv6-wildcard", "ipv6-host"],
)
def test_doc_render_opens_page_at_bind_address(_thumb_env: Path, _share: Path, monkeypatch: pytest.MonkeyPatch,
                                               env: dict[str, str], base: str) -> None:
    """The browser opens the doc page at the address the server listens on."""
    _set_bind_env(monkeypatch, env)
    urls: list[str] = []
    _, factory = _fake_manager_factory(urls=urls)
    status, key = _run_doc_render_job(_thumb_env, _share, monkeypatch, factory)
    assert status == "done"
    assert urls == [f"{base}/view/doc/{key}?thumb=1"]


def test_doc_render_failed_navigation_marks_failed(_thumb_env: Path, _share: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """A page that does not load leaves a failure mark and still tears the browser down."""
    calls, factory = _fake_manager_factory(navigate_ok=False)
    status, key = _run_doc_render_job(_thumb_env, _share, monkeypatch, factory)
    assert status == "failed"
    assert calls == ["create", "new_page", "set_viewport_size", "navigate", "close_page", "cleanup"]
    assert not os.path.exists(os.path.join(cache_module.render_root(), key, "thumb.png"))


def test_doc_render_failed_viewport_marks_failed(_thumb_env: Path, _share: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """A viewport that cannot be set leaves a failure mark and still tears the browser down."""
    calls, factory = _fake_manager_factory(viewport_ok=False)
    status, key = _run_doc_render_job(_thumb_env, _share, monkeypatch, factory)
    assert status == "failed"
    assert calls == ["create", "new_page", "set_viewport_size", "close_page", "cleanup"]
    assert not os.path.exists(os.path.join(cache_module.render_root(), key, "thumb.png"))


def test_doc_render_new_page_failure_keeps_original_error(_thumb_env: Path, _share: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """A page open that raises keeps the original error on the mark and still cleans up."""
    calls, factory = _fake_manager_factory(new_page_error=RuntimeError("no browser windows"))
    status, key = _run_doc_render_job(_thumb_env, _share, monkeypatch, factory)
    assert status == "failed"
    mark = _thumb_env / "render" / ".state" / f"{key}.thumb-failed"
    assert "no browser windows" in mark.read_text()
    assert calls == ["create", "new_page", "cleanup"]


def test_browser_cycles_are_fresh_per_job(_thumb_env: Path, _share: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """Each doc render connects and tears down its own browser."""
    calls, factory = _fake_manager_factory()
    _patch_browser(monkeypatch, factory)
    monkeypatch.setattr(office, "ensure_doc_for_thumb", _published)
    (_share / "a.md").write_text("# hi\n")

    async def run() -> None:
        try:
            key = "c" * 64
            second_key = "d" * 64
            thumbs.start_thumbnail_job(key, RenderKind.MARKDOWN, "a.md")
            assert await _await_thumb_status(key) == "done"
            thumbs.start_thumbnail_job(second_key, RenderKind.MARKDOWN, "a.md")
            assert await _await_thumb_status(second_key) == "done"
        finally:
            await thumbs.stop_thumbnails()

    asyncio.run(run())
    assert calls.count("create") == 2
    assert calls.count("cleanup") == 2


def test_browser_guard_serializes_cycles(_thumb_env: Path, _share: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Two doc renders racing the connect do not interleave their browser cycles."""
    _, factory = _fake_manager_factory()
    stats = {"n": 0, "now": 0, "peak": 0}

    async def racy_factory() -> _FakeManager:
        stats["n"] += 1
        stats["now"] += 1
        stats["peak"] = max(stats["peak"], stats["now"])
        try:
            await asyncio.sleep(0.1)
            return await factory()
        finally:
            stats["now"] -= 1

    monkeypatch.setattr(thumbs, "BrowserManager", SimpleNamespace(create=racy_factory))
    monkeypatch.setattr(office, "ensure_doc_for_thumb", _published)
    (_share / "a.md").write_text("# hi\n")
    (_share / "b.md").write_text("# hai\n")
    key_a = cache_module.key_for("a.md", int(os.stat(_share / "a.md").st_mtime))
    key_b = cache_module.key_for("b.md", int(os.stat(_share / "b.md").st_mtime))

    async def run() -> None:
        try:
            thumbs.start_thumbnail_job(key_a, RenderKind.MARKDOWN, "a.md")
            thumbs.start_thumbnail_job(key_b, RenderKind.MARKDOWN, "b.md")
            assert await _await_thumb_status(key_a) == "done"
            assert await _await_thumb_status(key_b) == "done"
        finally:
            await thumbs.stop_thumbnails()

    asyncio.run(run())
    assert stats["n"] == 2  # each render got its own manager
    assert stats["peak"] == 1  # the guard lock kept the cycles from overlapping


@pytest.mark.live
@pytest.mark.timeout(90)
def test_live_cdp_doc_thumbnail(
    _thumb_env: Path, _share: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real Chrome shoots a doc page and saves it as a thumbnail."""
    monkeypatch.setattr(store_module, "_DB_PATH", str(tmp_path / "records.db"))
    (_share / "sample.md").write_bytes((FIX / "sample.md").read_bytes())
    server = uvicorn.Server(uvicorn.Config(build_app(), host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    def _poll_port() -> int:
        for _ in range(50):
            if server.started and server.servers:
                return server.servers[0].sockets[0].getsockname()[1]
            time.sleep(0.1)
        raise AssertionError("uvicorn did not start")

    port = _poll_port()
    monkeypatch.setenv("HTTP_HOST", "127.0.0.1")
    monkeypatch.setenv("HTTP_PORT", str(port))

    key = "b" * 64

    async def run() -> None:
        await cache_module.write_doc(
            key, '<div class="md-doc"><h1>CDP target</h1><p>hello</p></div>', None, {"kind": "markdown"})
        try:
            thumbs.start_thumbnail_job(key, RenderKind.MARKDOWN, "sample.md")
            assert await _await_thumb_status(key) == "done"
        finally:
            await thumbs.stop_thumbnails()

    try:
        asyncio.run(run())
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    path = os.path.join(cache_module.render_root(), key, "thumb.png")
    assert os.path.exists(path)
    with Image.open(path) as im:
        assert im.size == (_THUMB_W, _THUMB_H)
