"""Tests for office document conversion."""

import asyncio
import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import override

import pytest

import app.lib.render.cache as cache_module
import app.lib.render.office as office
from app.lib.render.classify import RenderKind

FIX = Path(__file__).parent / "fixtures" / "office"

_DOC_KEY = "k" * 64
_SETTLE_SHORT = 0.05
_FAST_TIMEOUT = 0.1
_WAIT_LIMIT_S = 5


@pytest.fixture
def _office_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give the test its own data dir."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data))
    return data


async def _await_outcome(key: str) -> None:
    """Wait until the conversion has published a page or left a failure mark."""
    for _ in range(200):
        if await cache_module.read_doc(key) is not None:
            return
        if await cache_module.convert_failed_reason(key) is not None:
            return
        await asyncio.sleep(_SETTLE_SHORT)
    raise AssertionError("conversion did not reach a durable outcome")


async def _convert(key: str, path: str, full_path: str, timeout: float = 90.0) -> None:
    """Start a conversion and wait for its durable outcome."""
    await office.start_office_conversion(path, key, full_path, timeout=timeout)
    await _await_outcome(key)


async def _converted_doc(monkeypatch: pytest.MonkeyPatch, html: str, assets: dict[str, bytes]) -> str:
    """Convert a document whose LibreOffice step gives back html, and return the saved page."""

    async def fake_exec(*args: str, **_kwargs: asyncio.subprocess.PIPE) -> _StubSofficeProc:
        out_dir = args[args.index("--outdir") + 1]
        stem = os.path.splitext(os.path.basename(args[-1]))[0]
        return _StubSofficeProc(out_dir, stem, html, assets)

    monkeypatch.setattr("app.lib.render.office.asyncio.create_subprocess_exec", fake_exec)
    await _convert(_DOC_KEY, "a.docx", str(FIX / "sample.docx"))
    doc, _meta = await cache_module.read_doc(_DOC_KEY)
    return doc


async def test_presentation_converts_to_cached_pdf(_office_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A presentation stores PDF bytes and tells the viewer to open the PDF frame."""
    pdf = b"%PDF-1.7\nslide"
    targets: list[str] = []

    async def fake_convert(_full_path: str, _timeout: float, target: str = "html") -> tuple[str, dict[str, bytes]]:
        targets.append(target)
        return "", {cache_module.OFFICE_PDF_ASSET: pdf}

    monkeypatch.setattr(office, "_soffice_convert", fake_convert)
    await _convert(_DOC_KEY, "deck.pptx", str(FIX / "sample.pptx"))
    doc, meta = await cache_module.read_doc(_DOC_KEY)
    assert targets == ["pdf"]
    assert doc == ""
    assert meta["mode"] == "frame"
    assert meta["engine"] == "soffice-pdf"
    assert (_office_env / "render" / _DOC_KEY / cache_module.OFFICE_PDF_ASSET).read_bytes() == pdf


def _recording_stub(spawned: list[str]) -> Callable[[str, str, str, RenderKind, float], Awaitable[None]]:
    """A conversion stub that records the keys it started and then settles."""

    async def stub(_key: str, _path: str, _full: str, _kind: RenderKind, _timeout: float) -> None:
        spawned.append(_key)
        await asyncio.sleep(_SETTLE_SHORT)

    return stub


def _flaky_classify(attempts: dict, hold: asyncio.Event | None,
                    held: asyncio.Queue | None = None) -> Callable[[str], Awaitable[RenderKind]]:
    """A classify that fails on the first call, then optionally holds a gate.

    Each held call puts its attempt number on the held queue before it waits.
    """

    async def classify(_full_path: str) -> RenderKind:
        attempts["n"] += 1
        if attempts["n"] == 1:
            await asyncio.sleep(_SETTLE_SHORT)  # hold the gate so the others queue on it
            raise RuntimeError("classify failed")
        if hold is not None:
            if held is not None:
                held.put_nowait(attempts["n"])
            await hold.wait()
        return RenderKind.OFFICE_DOC

    return classify


def _conversion_task(key: str) -> asyncio.Task:
    """The task running the conversion for a key."""
    name = f"office-{key[:8]}"
    for task in asyncio.all_tasks():
        if task.get_name() == name:
            return task
    raise AssertionError("no conversion task for the key")


def _has_conversion(key: str) -> bool:
    """Whether a conversion is running for a key right now."""
    name = f"office-{key[:8]}"
    return any(task.get_name() == name for task in asyncio.all_tasks())


def test_start_dedup_joins_live(_office_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second start for a file that is converting does not start another."""
    spawned: list[str] = []

    hold = asyncio.Event()

    async def slow(_key: str, _path: str, _full: str, _kind: RenderKind, _timeout: float) -> None:
        spawned.append(_key)
        await hold.wait()

    monkeypatch.setattr(office, "_conversion", slow)
    key = cache_module.key_for("a", 1)

    async def run() -> None:
        await asyncio.wait_for(office.start_office_conversion("a.docx", key, str(FIX / "sample.docx")), _WAIT_LIMIT_S)
        await asyncio.wait_for(office.start_office_conversion("a.docx", key, str(FIX / "sample.docx")), _WAIT_LIMIT_S)
        assert len(spawned) == 1  # the running conversion was joined, not duplicated
        hold.set()
        await asyncio.wait_for(_conversion_task(key), _WAIT_LIMIT_S)  # let the first conversion finish
        await office.start_office_conversion("a.docx", key, str(FIX / "sample.docx"))
        assert len(spawned) == 2  # the finished conversion did not block a fresh one

    asyncio.run(run())


def test_start_returns_immediately(_office_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Start and join return at once and never wait for the conversion to finish."""
    started: list[str] = []

    async def running(_key: str, _path: str, _full: str, _kind: RenderKind, _timeout: float) -> None:
        started.append(_key)
        await asyncio.Event().wait()  # never finishes on its own

    monkeypatch.setattr(office, "_conversion", running)
    key = "b" * 64

    async def run() -> None:
        await office.start_office_conversion("a.docx", key, str(FIX / "sample.docx"))
        assert started == [key]  # the conversion began and start returned at once
        # A second caller joins the live conversion without starting another.
        await office.start_office_conversion("a.docx", key, str(FIX / "sample.docx"))
        assert started == [key]

    asyncio.run(run())


def test_concurrent_callers_start_one_conversion(_office_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two callers at once start one conversion."""
    gate = asyncio.Event()
    spawned: list[str] = []

    async def slow(_key: str, _path: str, _full: str, _kind: RenderKind, _timeout: float) -> None:
        spawned.append(_key)
        await gate.wait()

    monkeypatch.setattr(office, "_conversion", slow)
    key = "c" * 64

    async def run() -> None:
        await asyncio.gather(
            office.start_office_conversion("a.docx", key, str(FIX / "sample.docx")),
            office.start_office_conversion("a.docx", key, str(FIX / "sample.docx")),
        )
        await asyncio.sleep(_SETTLE_SHORT)  # let the spawned conversion get going
        gate.set()
        assert len(spawned) == 1  # only one conversion was started

    asyncio.run(run())


def test_claimant_failure_wakes_waiter(_office_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed claim lets a waiting caller take over and start the conversion."""
    attempts = {"n": 0}
    monkeypatch.setattr(office, "classify", _flaky_classify(attempts, None))
    spawned: list[str] = []
    monkeypatch.setattr(office, "_conversion", _recording_stub(spawned))
    key = "d" * 64

    async def run() -> tuple:
        return await asyncio.gather(
            office.start_office_conversion("a.docx", key, str(FIX / "sample.docx")),
            office.start_office_conversion("a.docx", key, str(FIX / "sample.docx")),
            return_exceptions=True,
        )

    first, second = asyncio.run(run())
    assert sum(isinstance(r, BaseException) for r in (first, second)) == 1  # the claimant failed
    assert spawned == [key]  # the waiter took over, exactly one conversion


def test_two_woken_waiters_serialize_to_one_claimant(_office_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two waiters woken by a failed claim start at most one conversion."""
    hold = asyncio.Event()
    held: asyncio.Queue[int] = asyncio.Queue()
    attempts = {"n": 0}
    monkeypatch.setattr(office, "classify", _flaky_classify(attempts, hold, held))
    spawned: list[str] = []
    monkeypatch.setattr(office, "_conversion", _recording_stub(spawned))
    key = "e" * 64

    async def run() -> list:
        results = asyncio.gather(
            office.start_office_conversion("a.docx", key, str(FIX / "sample.docx")),
            office.start_office_conversion("a.docx", key, str(FIX / "sample.docx")),
            office.start_office_conversion("a.docx", key, str(FIX / "sample.docx")),
            return_exceptions=True,
        )
        await asyncio.wait_for(held.get(), _WAIT_LIMIT_S)  # the claim failed and both woken waiters are classifying
        await asyncio.wait_for(held.get(), _WAIT_LIMIT_S)
        hold.set()
        return list(await results)

    results = asyncio.run(run())
    assert sum(isinstance(r, BaseException) for r in results) == 1  # only the claimant failed
    assert spawned == [key]  # one waiter took over, the other joined the takeover
    assert attempts["n"] == 3


def test_waiter_gives_up_when_registration_is_slow(_office_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A waiter whose gate never resolves raises instead of starting a duplicate."""
    stuck = asyncio.Event()
    classifying = asyncio.Event()

    async def stuck_classify(_full_path: str) -> RenderKind:
        classifying.set()
        await stuck.wait()
        return RenderKind.OFFICE_DOC

    monkeypatch.setattr(office, "classify", stuck_classify)
    monkeypatch.setattr(office, "_TIMEOUT", _FAST_TIMEOUT)
    key = "f" * 64

    async def run() -> None:
        first = asyncio.create_task(office.start_office_conversion("a.docx", key, str(FIX / "sample.docx")))
        await asyncio.wait_for(classifying.wait(), _WAIT_LIMIT_S)  # the claimant is holding the gate in classify
        with pytest.raises(Exception, match="did not complete in time"):
            await office.start_office_conversion("a.docx", key, str(FIX / "sample.docx"))
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not _has_conversion(key)  # no conversion was registered

    asyncio.run(run())


def test_failed_conversion_ends_completed_with_bounded_marker(_office_env: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed conversion writes a bounded failure mark and leaves no page."""
    reason = "soffice exploded " + "x" * 300

    async def boom(_key: str, _path: str, _full: str, _kind: RenderKind, _timeout: float) -> None:
        raise RuntimeError(reason)

    monkeypatch.setattr(office, "_conversion", boom)
    key = "g" * 64

    async def run() -> None:
        await office.start_office_conversion("a.docx", key, str(FIX / "sample.docx"))
        task = _conversion_task(key)
        await asyncio.wait_for(task, _WAIT_LIMIT_S)  # let the failure mark be written
        assert not task.cancelled()
        assert task.exception() is None  # the failure was swallowed after the marker
        assert await cache_module.convert_failed_reason(key) == reason[:200]
        assert await cache_module.read_doc(key) is None

    asyncio.run(run())


async def test_conversion_writes_no_task_rows_or_run_folders(_office_env: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """A conversion leaves no task row, no run folder, and no log capture."""

    async def stub(_key: str, _path: str, _full: str, _kind: RenderKind, _timeout: float) -> None:
        await asyncio.sleep(_SETTLE_SHORT)

    monkeypatch.setattr(office, "_conversion", stub)
    key = "h" * 64
    await office.start_office_conversion("a.docx", key, str(FIX / "sample.docx"))
    await asyncio.wait_for(_conversion_task(key), _WAIT_LIMIT_S)
    assert not (_office_env / "runs").exists()  # office makes no run folder of its own
    assert not (_office_env / "records.db").exists()  # office writes no task rows


async def test_ensure_waits_for_a_live_conversion(_office_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The prepare stage waits out a running conversion instead of failing early."""
    release = asyncio.Event()

    async def late(_key: str, _path: str, _full: str, _kind: RenderKind, _timeout: float) -> None:
        await release.wait()
        await cache_module.write_doc(_key, "<div class='md-doc'>late</div>", None, {})

    monkeypatch.setattr(office, "_conversion", late)
    key = "i" * 64

    await office.start_office_conversion("a.docx", key, str(FIX / "sample.docx"))
    checker = asyncio.ensure_future(
        office.ensure_doc_for_thumb(RenderKind.OFFICE_DOC, key, "a.docx", str(FIX / "sample.docx"))
    )
    await asyncio.sleep(_SETTLE_SHORT)
    assert not checker.done()  # still waiting on the live conversion
    release.set()
    assert await checker == "published"
    assert await cache_module.convert_failed_reason(key) is None  # no premature failure


async def test_converted_page_is_reduced(_office_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The saved page keeps the text and the tables and drops everything that can run."""
    html = (
        "<html><head><style>p{margin:0}@page{size:8.5in 11in}</style>"
        '<link rel="stylesheet" href="x.css"></head>'
        "<body><h1 style=\"color:red\" >Hi</h1>"
        '<img src="sample_html_abc.png" width="288" height="180">'
        "<script>evil()</script><object data=\"o\"></object><embed src=\"e\">"
        "<iframe src=\"i\"></iframe>"
        '<table border="1"><tr><td>a</td></tr></table></body></html>'
    )
    reduced = await _converted_doc(monkeypatch, html, {"sample_html_abc.png": b"\x89PNG"})
    assert reduced.startswith('<div class="md-doc">')
    assert "<head" not in reduced and "<style" not in reduced and "<link" not in reduced
    assert "<script" not in reduced and "<object" not in reduced and "<embed" not in reduced and "<iframe" not in reduced
    assert "color:red" not in reduced and 'style=' not in reduced
    assert 'src="/view/doc/' + _DOC_KEY + '/sample_html_abc.png"' in reduced
    assert 'width="288"' in reduced and 'height="180"' in reduced
    assert "<table" in reduced


@pytest.mark.parametrize("attr", ["onclick", "onmouseover", "onerror", "onCLICK"])
async def test_converted_page_strips_event_handlers(
    _office_env: Path, monkeypatch: pytest.MonkeyPatch, attr: str
) -> None:
    html = (
        '<html><body><h1 onclick="alert(1)" onmouseover="x()" onerror="y()">Hi</h1>'
        '<div onCLICK="z()">body</div></body></html>'
    )
    reduced = await _converted_doc(monkeypatch, html, {})
    assert attr.lower() not in reduced
    assert "alert" not in reduced and "x()" not in reduced and "z()" not in reduced


@pytest.mark.parametrize(
    "value",
    [
        "javascript:alert(1)",
        "JaVaScRiPt:alert(2)",
        "data:text/html;base64,AAAA",
        "vbscript:msgbox",
        "behav:alert",
        " javascript:alert(3)",
        "\x01javascript:alert(4)",
    ],
)
async def test_converted_page_strips_dangerous_url_schemes(
    _office_env: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    html = f'<html><body><a href="{value}">x</a><img src="{value}" alt="y"></body></html>'
    reduced = await _converted_doc(monkeypatch, html, {})
    assert f'href="{value}"' not in reduced
    assert f'src="{value}"' not in reduced


@pytest.mark.parametrize(
    "href",
    [
        "http://ok.example/x",
        "https://ok.example/x",
        "//ok.example/x",
        "/view/doc/k/x.html",
        "relative/page.html",
        "mailto:a@b.c",
    ],
)
async def test_converted_page_keeps_safe_url_schemes(
    _office_env: Path, monkeypatch: pytest.MonkeyPatch, href: str
) -> None:
    html = f'<html><body><a href="{href}">ok</a></body></html>'
    reduced = await _converted_doc(monkeypatch, html, {})
    assert f'href="{href}"' in reduced


async def test_marked_key_never_starts_from_thumb(_office_env: Path) -> None:
    key = "j" * 64
    await cache_module.mark_failed(key, "convert", "Conversion cancelled")
    started: list[str] = []

    async def fake_start(_path: str, key2: str, _full: str) -> None:
        started.append(key2)

    monkey_start = office.start_office_conversion
    office.start_office_conversion = fake_start  # type: ignore[method-assign]
    try:
        status = await office.ensure_doc_for_thumb(
            RenderKind.OFFICE_DOC, key, "a.docx", str(FIX / "sample.docx")
        )
    finally:
        office.start_office_conversion = monkey_start
    assert status == "failed"
    assert started == []
    assert await cache_module.convert_failed_reason(key) == "Conversion cancelled"  # the reason is kept


async def test_ensure_fresh_starts_and_reports(_office_env: Path) -> None:
    key = "m" * 64
    real_start = office.start_office_conversion

    async def fake_start(_path: str, key2: str, _full: str) -> None:
        await cache_module.write_doc(key2, "<div class=\"md-doc\">ok</div>", None, {})

    office.start_office_conversion = fake_start  # type: ignore[method-assign]
    try:
        status = await office.ensure_doc_for_thumb(
            RenderKind.OFFICE_DOC, key, "a.docx", str(FIX / "sample.docx")
        )
    finally:
        office.start_office_conversion = real_start
    assert status == "published"


async def test_ensure_doc_kind_builds_and_publishes(_office_env: Path) -> None:
    """A document kind is built and published with no failure mark."""
    key = "5" * 64
    md = _office_env / "note.md"
    md.write_text("# Hello\n\nworld\n", encoding="utf-8")

    status = await office.ensure_doc_for_thumb(RenderKind.MARKDOWN, key, "note.md", str(md))
    assert status == "published"
    doc, _meta = await cache_module.read_doc(key)
    assert doc is not None and "Hello" in doc
    assert await cache_module.convert_failed_reason(key) is None  # no failure mark written


async def test_ensure_doc_kind_html_is_sandboxed_frame(_office_env: Path) -> None:
    """An html file is shown in a sandboxed frame, not as the raw file."""
    key = "6" * 64
    html = _office_env / "page.html"
    html.write_text("<html><body>hi</body></html>", encoding="utf-8")

    status = await office.ensure_doc_for_thumb(RenderKind.HTML, key, "page.html", str(html))
    assert status == "published"
    doc, meta = await cache_module.read_doc(key)
    assert doc == '<iframe class="html-frame" src="/dav/page.html?ephemeral=1" sandbox="allow-same-origin"></iframe>'
    assert meta["mode"] == "frame"


async def test_ensure_doc_kind_html_frame_url_encoded(_office_env: Path) -> None:
    """An html path with special characters gets an escaped frame source."""
    key = "e" * 64
    html = _office_env / "my page.html"
    html.write_text("<html><body>hi</body></html>", encoding="utf-8")

    status = await office.ensure_doc_for_thumb(RenderKind.HTML, key, "my page.html", str(html))
    assert status == "published"
    doc, _meta = await cache_module.read_doc(key)
    assert doc == '<iframe class="html-frame" src="/dav/my%20page.html?ephemeral=1" sandbox="allow-same-origin"></iframe>'


async def test_ensure_doc_kind_source_vanished_fails(_office_env: Path) -> None:
    """A missing source file fails and nothing is published."""
    key = "7" * 64
    missing = str(_office_env / "ghost.md")

    status = await office.ensure_doc_for_thumb(RenderKind.MARKDOWN, key, "ghost.md", missing)
    assert status == "failed"
    assert await cache_module.read_doc(key) is None


async def test_sheet_read_timeout_message(_office_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sheet read that runs too long fails and says it timed out."""

    async def slow_sheets(_full_path: str) -> list[tuple[str, list[tuple]]]:
        await asyncio.Event().wait()  # never finishes
        raise AssertionError("the sheet read was not cut off")

    async def as_xlsx(_full_path: str) -> RenderKind:
        return RenderKind.XLSX

    monkeypatch.setattr(office, "xlsx_sheets", slow_sheets)
    monkeypatch.setattr(office, "classify", as_xlsx)
    key = "9" * 64

    await asyncio.wait_for(_convert(key, "a.xlsx", str(FIX / "sample.docx"), timeout=_FAST_TIMEOUT), _WAIT_LIMIT_S)
    assert await cache_module.convert_failed_reason(key) == "Sheet read failed: timed out after 90 s"


async def test_retry_force_reconvert_deletes_cached_entry(_office_env: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    thumbs = pytest.importorskip("app.lib.render.thumbs")
    key = "n" * 64
    await cache_module.write_doc(key, "<div class=\"md-doc\">old</div>", {"a.png": b"x"}, {"path": "a"})
    assert os.path.exists(os.path.join(cache_module.render_root(), key, "a.png"))

    queued: list[tuple] = []

    def record(key2: str, kind: RenderKind, path: str) -> bool:
        queued.append((key2, kind, path))
        return False

    monkeypatch.setattr(thumbs, "start_thumbnail_job", record)

    async def stub(_key: str, _path: str, _full: str, _kind: RenderKind, _timeout: float) -> None:
        await asyncio.sleep(_SETTLE_SHORT)

    monkeypatch.setattr(office, "_conversion", stub)
    await office.retry_office_conversion("a.docx", key, str(FIX / "sample.docx"))
    assert queued and queued[0][0] == key  # the thumbnail was queued again
    assert not os.path.isdir(os.path.join(cache_module.render_root(), key))
    assert await cache_module.convert_failed_reason(key) is None


async def test_retry_joins_in_flight_and_deletes_nothing(_office_env: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """A retry while a conversion is running joins it and deletes nothing."""
    key = "4" * 64
    await cache_module.write_doc(key, "doc", None, {})

    async def running(_key: str, _path: str, _full: str, _kind: RenderKind, _timeout: float) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(office, "_conversion", running)
    await office.start_office_conversion("a.docx", key, str(FIX / "sample.docx"))
    await office.retry_office_conversion("a.docx", key, str(FIX / "sample.docx"))
    assert await cache_module.read_doc(key) == ("doc", {})  # nothing deleted


class _StubSofficeProc(asyncio.subprocess.Process):
    """A fake soffice run that writes out the given page and files."""

    def __init__(self, out_dir: str, stem: str, html: str, assets: dict[str, bytes]) -> None:
        self._out_dir = out_dir
        self._stem = stem
        self._html = html
        self._assets = assets
        self._rc = 0

    @property
    def returncode(self) -> int:
        return self._rc

    @override
    async def communicate(self, _input: bytes | None = None) -> tuple[bytes, bytes]:
        def _write() -> None:
            with open(os.path.join(self._out_dir, f"{self._stem}.html"), "w", encoding="utf-8") as fh:
                fh.write(self._html)
            for name, data in self._assets.items():
                with open(os.path.join(self._out_dir, name), "wb") as fh:
                    fh.write(data)

        await asyncio.to_thread(_write)
        return b"", b""

    @override
    def kill(self) -> None:
        pass

    @override
    async def wait(self) -> int:
        return self.returncode


class _FakeProc(asyncio.subprocess.Process):
    """A fake process that never finishes on its own."""

    def __init__(self) -> None:
        self.killed: list[int] = []

    @override
    def kill(self) -> None:
        self.killed.append(1)

    @override
    async def wait(self) -> int:
        return -9

    @override
    async def communicate(self, _input: bytes | None = None) -> tuple[bytes, bytes]:
        await asyncio.sleep(30)
        return b"", b""


def test_soffice_kills_child_on_cancel(_office_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancelled conversion kills the child process and leaves no failure mark."""
    proc = _FakeProc()
    key = "p" * 64
    child_spawned = asyncio.Event()

    async def fake_exec(*_args: str, **_kwargs: asyncio.subprocess.PIPE) -> _FakeProc:
        child_spawned.set()
        return proc

    monkeypatch.setattr("app.lib.render.office.asyncio.create_subprocess_exec", fake_exec)

    async def run() -> None:
        await office.start_office_conversion("a.docx", key, str(FIX / "sample.docx"))
        task = _conversion_task(key)
        await child_spawned.wait()  # cancel only after the child exists
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert proc.killed == [1]
        assert await cache_module.convert_failed_reason(key) is None  # no marker at shutdown

    asyncio.run(run())


async def test_sheet_tables_fragment(_office_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every sheet of a workbook becomes its own table, with the row and column counts saved."""
    async def fake_sheets(_full_path: str) -> list[tuple[str, list[tuple]]]:
        return [("Alpha", [("h1", "h2"), *[("a%d" % i, "b") for i in range(10)]]),
                ("Beta", [("x",)])]

    async def as_xlsx(_full_path: str) -> RenderKind:
        return RenderKind.XLSX

    monkeypatch.setattr(office, "xlsx_sheets", fake_sheets)
    monkeypatch.setattr(office, "classify", as_xlsx)
    key = "f" * 64
    await _convert(key, "a.xlsx", str(FIX / "sample.docx"))
    fragment, meta = await cache_module.read_doc(key)
    assert fragment.count('<table class="csv-table">') == 2
    assert '<h3 class="sheet-title">Alpha</h3>' in fragment
    assert fragment.count(' class="csv-extra"') == 2  # rows 9 and 10 of the first sheet
    assert meta["row_count"] == 10
    assert meta["column_count"] == 2


@pytest.mark.live
@pytest.mark.timeout(120)
async def test_live_soffice_converts_presentation_to_pdf(_office_env: Path, tmp_path: Path) -> None:
    """A real presentation makes a PDF that the viewer can open."""
    key = "b" * 64
    full = str(tmp_path / "sample.pptx")
    shutil.copy(FIX / "sample.pptx", full)
    await _convert(key, "sample.pptx", full)
    doc, meta = await cache_module.read_doc(key)
    pdf_path = os.path.join(cache_module.render_root(), key, cache_module.OFFICE_PDF_ASSET)
    with open(pdf_path, "rb") as fh:
        assert fh.read(5) == b"%PDF-"
    assert doc == ""
    assert meta["mode"] == "frame"


@pytest.mark.live
@pytest.mark.timeout(120)
async def test_live_soffice_convert_docx_with_embedded_image(
    _office_env: Path, tmp_path: Path
) -> None:
    """A real docx with a picture converts and the picture gets its own link."""
    key = "a" * 64
    full = str(tmp_path / "sample.docx")
    shutil.copy(FIX / "sample.docx", full)
    await _convert(key, "sample.docx", full)
    doc, _meta = await cache_module.read_doc(key)
    assert '<div class="md-doc">' in doc
    saved = os.listdir(os.path.join(cache_module.render_root(), key))
    assert any(name.endswith(".png") for name in saved), "soffice must emit the embedded image"
    assert f"/view/doc/{key}/" in doc
