import io
import zipfile
from types import SimpleNamespace

import httpx
import pytest

from tests.app.conftest import seed_task, write_run_file, write_share_file


async def _client(app_http: SimpleNamespace) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=app_http.base_url)


_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def _zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(zipfile.ZipInfo("notes.txt", date_time=_ZIP_TIME), "hi")
    return buf.getvalue()


def _xlsx_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(zipfile.ZipInfo("xl/workbook.xml", date_time=_ZIP_TIME), "<workbook/>")
    return buf.getvalue()


async def _seed_result(app_http: SimpleNamespace, tid: str, body: bytes) -> None:
    await seed_task(tid, "web_fetch", status="completed")
    write_run_file(app_http.data_dir, tid, "result", body)


async def _seed_files_run(app_http: SimpleNamespace, tid: str, body: bytes) -> None:
    """A run whose body is given and whose files sidecar names one share file."""
    await _seed_result(app_http, tid, body)
    write_share_file(app_http.share, "downloads/notes.txt", b"line one\nline two\n")
    write_run_file(app_http.data_dir, tid, "files",
                   b'{"path": "downloads/notes.txt", "size": 18, "mtime": 1756800000}\n')


async def _fetch_result_page(app_http: SimpleNamespace, tid: str, body: bytes) -> str:
    await _seed_result(app_http, tid, body)
    async with await _client(app_http) as c:
        r = await c.get(f"/view/result/{tid}")
    assert r.status_code == 200
    return r.text


async def test_result_page_is_solo_body_only(app_http: SimpleNamespace) -> None:
    """The result page is vp-solo, has no feed column, and no file blocks."""
    tid = "pg-solo-0001"
    body = b"# Notes\n\nSome **bold** text.\n"
    html = await _fetch_result_page(app_http, tid, body)
    assert 'class="view-page vp-solo"' in html
    assert "vp-feed" not in html
    assert "vp-feed-col" not in html
    assert "vp-feed-main" not in html
    assert "FILES" not in html


async def test_page_vp_feed_download_only(app_http: SimpleNamespace) -> None:
    tid = "pg1234567890"
    body = b"# Notes\n\nSome **bold** text and a [link](https://example.com).\n\n- one\n- two\n"
    html = await _fetch_result_page(app_http, tid, body)
    assert 'class="view-page vp-solo"' in html
    assert 'class="vp-prov"' not in html
    assert 'class="vp-topbar"' not in html
    assert f'/api/tasks/{tid}/result/download' in html
    assert '<div class="md-doc">' in html


async def test_page_kind_propagates(app_http: SimpleNamespace) -> None:
    tid = "pg-kind1234"
    body = b"# Notes\n\nSome **bold** text.\n"
    await _seed_result(app_http, tid, body)
    async with await _client(app_http) as c:
        as_code = await c.get(f"/view/result/{tid}?kind=code")
        bogus = await c.get(f"/view/result/{tid}?kind=bogus")
    assert as_code.status_code == 200
    assert 'class="code-block"' in as_code.text
    assert '<div class="md-doc">' not in as_code.text
    # A kind the page does not know is ignored, and the body auto-detects.
    assert bogus.status_code == 200
    assert '<div class="md-doc">' in bogus.text


async def test_page_declared_kind_beats_kind_param(app_http: SimpleNamespace) -> None:
    # A run whose tool row carries a declared kind wins over a ?kind= query param.
    tid = "t-decl"
    await seed_task(tid, "stub_rrt", status="completed", render_type="markdown")
    write_run_file(app_http.data_dir, tid, "result", b"def f():\n    return 1\n")
    async with await _client(app_http) as c:
        r = await c.get(f"/view/result/{tid}", params={"kind": "code"})
    assert r.status_code == 200
    assert '<div class="md-doc">' in r.text
    assert 'class="code-block"' not in r.text


async def test_page_files_divider(app_http: SimpleNamespace) -> None:
    """D1: the full-screen page shows no file blocks, even when the sidecar names files."""
    tid = "pg-files123"
    await _seed_files_run(app_http, tid, b"body text\n")
    async with await _client(app_http) as c:
        r = await c.get(f"/view/result/{tid}")
    assert r.status_code == 200
    html = r.text
    assert "FILES" not in html                              # no divider
    assert 'data-path="downloads/notes.txt"' not in html    # no file block
    assert "vp-file-block" not in html
    assert "notes.txt" not in html
    assert "18 B" not in html                               # no live share size


async def test_page_files_only_subtext(app_http: SimpleNamespace) -> None:
    """A files-only run (empty body, files present) says 'stored files only'."""
    tid = "pg-files-only"
    await _seed_files_run(app_http, tid, b"")
    async with await _client(app_http) as c:
        r = await c.get(f"/view/result/{tid}")
    assert r.status_code == 200
    assert "No result content" in r.text
    assert "This run stored files only." in r.text
    assert "This run stored an empty result." not in r.text
    assert "FILES" not in r.text


async def test_page_no_files_no_divider(app_http: SimpleNamespace) -> None:
    tid = "pg-nofiles3"
    await _seed_result(app_http, tid, b"just a body\n")
    async with await _client(app_http) as c:
        r = await c.get(f"/view/result/{tid}")
    assert r.status_code == 200
    assert "FILES" not in r.text


async def test_page_legacy_body(app_http: SimpleNamespace) -> None:
    tid = "pg-legacy123"
    await _seed_result(app_http, tid, b"\x00\x01\x02\x03")
    async with await _client(app_http) as c:
        r = await c.get(f"/view/result/{tid}")
    assert r.status_code == 200
    html = r.text
    assert "result.bin" in html
    assert "Legacy binary result" in html
    assert f'/api/tasks/{tid}/result/download' in html
    assert "/result/view/retry" not in html
    assert 'data-vp-retry' not in html


@pytest.mark.parametrize(
    ("tid", "body"),
    [
        ("pg-bin-audio", b"RIFF\x00\x00\x00\x00WAVEfmt" + b"\x00" * 40),
        ("pg-bin-office", _xlsx_bytes()),
        ("pg-bin-archive", _zip()),
    ],
)
async def test_page_binary_body_renders_legacy(app_http: SimpleNamespace, tid: str, body: bytes) -> None:
    # A wav, an office file, and an archive all carry a binary signal, so the
    # body shows the legacy card instead of a preview.
    await _seed_result(app_http, tid, body)
    async with await _client(app_http) as c:
        r = await c.get(f"/view/result/{tid}")
    assert r.status_code == 200
    assert "result.bin" in r.text
    assert "Legacy binary result" in r.text


async def test_page_zero_byte_body(app_http: SimpleNamespace) -> None:
    tid = "pg-zero1234"
    await _seed_result(app_http, tid, b"")
    async with await _client(app_http) as c:
        r = await c.get(f"/view/result/{tid}")
    assert r.status_code == 200
    html = r.text
    assert "No result content" in html
    assert "This run stored an empty result." in html
    assert 'class="feed-state"' in html


async def test_page_oversize_body_note(app_http: SimpleNamespace) -> None:
    # Past the 1 MiB rich cap the body carries the oversize note, which on the
    # full screen points at Download instead of the (absent) Raw control.
    tid = "pg-oversize1"
    body = b"line of plain text\n" * 100_000
    assert len(body) > 1024 * 1024
    html = await _fetch_result_page(app_http, tid, body)
    assert "Too large to render — showing first 256 KB. Download the result for the full payload." in html
    assert "Download the result Download" not in html


async def test_page_gc_expired(app_http: SimpleNamespace) -> None:
    tid = "pg-gc12345"
    await seed_task(tid, "web_fetch", status="completed")    # a record whose result file is gone
    async with await _client(app_http) as c:
        r = await c.get(f"/view/result/{tid}")
    assert r.status_code == 200
    assert 'class="vp-expired"' in r.text
    assert "This result is no longer available" in r.text


async def test_page_record_miss_delegates(app_http: SimpleNamespace) -> None:
    # With no record and no share file, the page says the link is not valid.
    async with await _client(app_http) as c:
        r = await c.get("/view/result/pg-norecord")
    assert r.status_code == 200
    assert 'class="vp-expired"' in r.text


async def test_page_record_miss_delegates_to_share_file(app_http: SimpleNamespace) -> None:
    write_share_file(app_http.share, "result/pg-sfile", b"hello share file\n")
    async with await _client(app_http) as c:
        r = await c.get("/view/result/pg-sfile")
    assert r.status_code == 200
    assert 'class="view-page vp-solo"' in r.text          # the share file page
    assert 'class="vp-expired"' not in r.text
    assert 'class="vp-prov"' not in r.text                 # no result chip on the share page
