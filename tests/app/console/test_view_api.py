from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

import app.lib.render.cache as cache
import app.lib.render.office as office
from app.lib.render.classify import RenderKind
from app.lib.shared import activity
from tests.app.console.conftest import console_env

_OLD_AGE = 40 * 86400


@pytest.fixture
async def view_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[httpx.AsyncClient, Path]]:
    """A test client for the view api and the share folder."""
    async with console_env(tmp_path, monkeypatch) as env:
        yield env


def _write(share: Path, rel: str, content: str) -> str:
    p = share / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return rel


@pytest.mark.parametrize("rel,content", [("cached.md", "# Cached"), ("file.zip", "PKdata")])
async def test_detail_records_interest_even_without_source_reads(
    view_api: tuple[httpx.AsyncClient, Path], rel: str, content: str,
) -> None:
    """Opening cached and unsupported files renews their retention."""
    client, share = view_api
    path = share / rel
    path.write_text(content)
    old = time.time() - _OLD_AGE
    os.utime(path, (old, old))
    response = await client.get("/api/view/" + rel)
    assert response.status_code == 200
    os.removexattr(path, "user.agent.last_access")
    response = await client.get("/api/view/" + rel)
    assert response.status_code == 200
    assert activity.last_used_sync(str(path), path.stat()) > time.time() - 10


@pytest.mark.parametrize("method,url", [
    ("GET", "/api/resources"),
    ("GET", "/api/resources/search?q=old"),
    ("HEAD", "/api/view/old.md"),
])
async def test_metadata_does_not_record_interest(
    view_api: tuple[httpx.AsyncClient, Path], method: str, url: str,
) -> None:
    """Listings, searches and HEAD requests leave interest unchanged."""
    client, share = view_api
    path = share / "old.md"
    path.write_text("# Sample")
    old = time.time() - _OLD_AGE
    os.utime(path, (old, old))
    response = await client.request(method, url)
    assert response.status_code == 200
    assert activity.last_used_sync(str(path), path.stat()) == old


async def test_markdown_doc_mode(view_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, share = view_api
    rel = _write(share, "upload/notes.md", "# Title\n\nbody **bold**\n")
    r = await c.get("/api/view/" + rel)
    assert r.status_code == 200
    d = r.json()
    assert d["mode"] == "doc"
    assert d["kind"] == "markdown"
    assert d["kind_label"] == "markdown"
    assert '<div class="md-doc">' in d["fragment"]
    assert "**bold**" not in d["fragment"]
    assert d["url"] is None and d["note"] is None


async def test_code_mode_language_and_line_count(view_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, share = view_api
    src = "def f():\n    return 1\n\ncall()\n"
    rel = _write(share, "upload/a.py", src)
    d = (await c.get("/api/view/" + rel)).json()
    assert d["mode"] == "code"
    assert d["language"] == "python"
    assert d["line_count"] == len(src.splitlines())
    assert d["fragment"].count('<div class="cl">') == d["line_count"]


async def test_csv_table_mode_counts(view_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, share = view_api
    rows = ["a,b,c,d,e"] + [f"{i},{i},{i},{i},{i}" for i in range(1, 12)]
    rel = _write(share, "upload/data.csv", "\n".join(rows) + "\n")
    d = (await c.get("/api/view/" + rel)).json()
    assert d["mode"] == "table"
    assert d["row_count"] == 11
    assert d["column_count"] == 5
    assert 'class="csv-table"' in d["fragment"]


async def test_json_code_mode_no_language(view_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, share = view_api
    rel = _write(share, "upload/x.json", '{"a": 1, "b": 2}')
    d = (await c.get("/api/view/" + rel)).json()
    assert d["mode"] == "code"
    assert d["language"] is None
    assert d["kind"] == "json"


async def test_undecodable_text_is_raw_null_fragment(view_api: tuple[httpx.AsyncClient, Path]) -> None:
    """Text the app cannot read comes back as raw mode with no preview."""
    c, share = view_api
    p = share / "upload" / "blob.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00\x01\x02\x03")
    d = (await c.get("/api/view/upload/blob.txt")).json()
    assert d["mode"] == "raw"
    assert d["fragment"] is None
    assert d["note"] == "No preview available for this file."


async def test_pdf_frame_mode_url(view_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, share = view_api
    rel = _write(share, "upload/doc.pdf", "%PDF-1.4 fake")
    d = (await c.get("/api/view/" + rel)).json()
    assert d["mode"] == "frame"
    assert d["url"] == f"/dav/{rel}?ephemeral=1"
    assert d["fragment"] is None
    assert d["pages"] is None  # a fake PDF has no page count


async def test_folder_returns_not_found(view_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, share = view_api
    (share / "somefolder").mkdir()
    r = await c.get("/api/view/somefolder")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


async def test_missing_returns_not_found(view_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, _ = view_api
    r = await c.get("/api/view/nope/never.md")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


async def test_nul_byte_returns_bad_param(view_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, _ = view_api
    r = await c.get("/api/view/a%00b.md")
    assert r.status_code == 400
    assert r.json()["code"] == "bad_param"


async def test_archive_is_unsupported(view_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, share = view_api
    rel = _write(share, "upload/x.zip", "PK\x03\x04 data")
    d = (await c.get("/api/view/" + rel)).json()
    assert d["mode"] == "unsupported"


async def test_office_planted_marker_is_convert_failed(view_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, share = view_api
    rel = _write(share, "upload/doc.docx", "PK\x03\x04 fake docx")
    st = (share / rel).stat()
    await cache.mark_failed(cache.key_for(rel, int(st.st_mtime)), "convert", "Conversion cancelled")
    d = (await c.get("/api/view/" + rel)).json()
    assert d["mode"] == "convert_failed"
    assert d["error"] == "Conversion cancelled"


async def test_office_fresh_starts_conversion(view_api: tuple[httpx.AsyncClient, Path],
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    c, share = view_api
    started = {"n": 0}

    async def fake_start(_path: str, _key: str, _full_path: str) -> None:
        started["n"] += 1

    monkeypatch.setattr(office, "start_office_conversion", fake_start)
    rel = _write(share, "upload/fresh.docx", "PK\x03\x04 fake docx")
    d = (await c.get("/api/view/" + rel)).json()
    assert d["mode"] == "converting"
    assert "task_id" not in d
    assert started["n"] == 1


async def test_retry_cached_key_deletes_entry_and_converts(view_api: tuple[httpx.AsyncClient, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    c, share = view_api

    async def fake_start(_path: str, _key: str, _full_path: str) -> None:
        return None

    monkeypatch.setattr(office, "start_office_conversion", fake_start)
    rel = _write(share, "upload/cached.docx", "PK\x03\x04 fake")
    st = (share / rel).stat()
    key = cache.key_for(rel, int(st.st_mtime))
    await cache.write_doc(key, "<div class='md-doc'>x</div>", None, {"mode": "doc"})
    r = await c.post(f"/api/view/{rel}/retry")
    assert r.status_code == 200
    d = r.json()
    assert d["mode"] == "converting"
    assert "task_id" not in d
    assert await cache.read_doc(key) is None  # the entry was deleted


async def test_retry_absolute_path_clears_the_view_entry(view_api: tuple[httpx.AsyncClient, Path],
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """A retry given by absolute path works on the same entry the view reads."""
    c, share = view_api

    async def fake_start(_path: str, _key: str, _full_path: str) -> None:
        return None

    monkeypatch.setattr(office, "start_office_conversion", fake_start)
    rel = _write(share, "upload/abs.docx", "PK\x03\x04 fake")
    full = os.path.realpath(str(share / rel))
    st = (share / rel).stat()
    key = cache.key_for(rel, int(st.st_mtime))
    await cache.mark_failed(key, "convert", "Conversion cancelled")
    view = (await c.get(f"/api/view/{full}")).json()
    assert view["mode"] == "convert_failed"
    r = await c.post(f"/api/view/{full}/retry")
    assert r.status_code == 200
    d = r.json()
    assert d["mode"] == "converting"
    assert d["path"] == rel
    assert await cache.convert_failed_reason(key) is None


async def test_retry_non_office_is_bad_param(view_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, share = view_api
    rel = _write(share, "upload/plain.txt", "hello")
    r = await c.post(f"/api/view/{rel}/retry")
    assert r.status_code == 400
    assert r.json()["code"] == "bad_param"


async def test_retry_missing_is_not_found(view_api: tuple[httpx.AsyncClient, Path]) -> None:
    c, _ = view_api
    r = await c.post("/api/view/ghost.docx/retry")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


async def test_retry_in_flight_joins_and_keeps_entry(view_api: tuple[httpx.AsyncClient, Path],
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """A retry while a conversion is running joins it and keeps the saved entry."""
    c, share = view_api

    async def running(_key: str, _path: str, _full: str, _kind: RenderKind, _timeout: float) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(office, "_conversion", running)
    rel = _write(share, "upload/busy.docx", "PK\x03\x04 fake")
    st = (share / rel).stat()
    key = cache.key_for(rel, int(st.st_mtime))
    await cache.write_doc(key, "<div class='md-doc'>y</div>", None, {"mode": "doc"})
    await office.start_office_conversion(rel, key, str(share / rel))
    r = await c.post(f"/api/view/{rel}/retry")
    assert r.status_code == 200
    assert r.json()["mode"] == "converting"
    assert "task_id" not in r.json()
    assert await cache.read_doc(key) == ("<div class='md-doc'>y</div>", {"mode": "doc"})  # the saved entry stays


async def test_route_order_literal_retry_file_is_viewable(view_api: tuple[httpx.AsyncClient, Path]) -> None:
    """A file named retry can still be viewed, PUT gives 405 and POST gives 404."""
    c, share = view_api
    _write(share, "sub/retry", "i am a plain file literally named retry")
    got = await c.get("/api/view/sub/retry")
    assert got.status_code == 200
    assert got.json()["mode"] == "plain"
    put = await c.put("/api/view/sub/retry")
    assert put.status_code == 405
    assert put.json()["code"] == "method_not_allowed"
    posted = await c.post("/api/view/sub/retry")
    assert posted.status_code == 404
    assert posted.json()["code"] == "not_found"


async def test_route_order_retry_reachable_for_office(view_api: tuple[httpx.AsyncClient, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    c, share = view_api

    async def fake_start(_path: str, _key: str, _full_path: str) -> None:
        return None

    monkeypatch.setattr(office, "start_office_conversion", fake_start)
    rel = _write(share, "upload/real.docx", "PK\x03\x04 fake")
    r = await c.post(f"/api/view/{rel}/retry")
    assert r.status_code == 200
    assert r.json()["mode"] == "converting"
    assert "task_id" not in r.json()
