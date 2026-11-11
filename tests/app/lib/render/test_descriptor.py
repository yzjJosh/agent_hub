"""Tests for the view descriptor builder."""

import asyncio
import os
from pathlib import Path

import pytest

import app.lib.render.cache as cache
import app.lib.render.office as office
from app.lib.render.classify import RenderKind
from app.lib.render.descriptor import (build_descriptor, converting_descriptor,
                                       resolve_view, share_path_for)

_DESCKEYS = (
    "path", "name", "size", "mtime", "kind", "kind_label", "mode", "fragment", "url",
    "error", "provenance", "note", "language", "line_count", "row_count", "column_count",
    "pages", "dimensions", "duration",
)


@pytest.fixture
def _share(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give the test its own share dir and data dir."""
    share = tmp_path / "res"
    share.mkdir()
    monkeypatch.setenv("RESOURCES_DIR", str(share))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    return share


async def test_resolve_view_rejects_nul(_share: Path) -> None:
    assert await resolve_view("a\x00b.md") is None


async def test_resolve_view_resolves_under_share(_share: Path) -> None:
    expected = os.path.realpath(os.path.join(str(_share), "upload", "x.md"))
    assert await resolve_view("upload/x.md") == expected


async def test_resolve_view_rejects_traversal(_share: Path) -> None:
    assert await resolve_view("../outside.md") is None


async def test_resolve_view_resolves_absolute_under_share(_share: Path) -> None:
    expected = os.path.realpath(os.path.join(str(_share), "upload", "x.md"))
    assert await resolve_view(expected) == expected


async def test_resolve_view_rejects_absolute_outside(_share: Path) -> None:
    assert await resolve_view("/etc/passwd") is None


def test_share_path_for_absolute_under_share(_share: Path) -> None:
    full = os.path.realpath(os.path.join(str(_share), "upload", "x.md"))
    assert share_path_for(full, "/abs/x.md") == "upload/x.md"


def test_share_path_for_outside_share_falls_back(_share: Path) -> None:
    assert share_path_for("/etc/passwd", "/etc/passwd") == "/etc/passwd"


def test_share_path_for_symlinked_share_root(_share: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A share dir reached through a symlink still maps to its share path."""
    link = _share.parent / "link"
    os.symlink(str(_share), str(link))
    monkeypatch.setenv("RESOURCES_DIR", str(link))
    full = os.path.realpath(os.path.join(str(_share), "upload", "x.md"))
    assert share_path_for(full, "/abs/x.md") == "upload/x.md"


def test_converting_descriptor_shape() -> None:
    d = converting_descriptor("upload/doc.docx", "doc.docx", 10, 1234, RenderKind.OFFICE_DOC)
    assert set(d) == set(_DESCKEYS)
    assert d["mode"] == "converting"
    assert d["kind"] == "office_doc"
    assert d["kind_label"] == "docx"
    assert d["path"] == "upload/doc.docx"
    assert d["name"] == "doc.docx"
    assert d["size"] == 10
    assert d["mtime"] == 1234
    assert set(f for f, v in d.items() if v is None) == {
        "fragment", "url", "error", "provenance", "note", "language", "line_count",
        "row_count", "column_count", "pages", "dimensions", "duration",
    }


async def test_build_descriptor_markdown_doc_mode(_share: Path) -> None:
    rel = "upload/n.md"
    (_share / rel).parent.mkdir(parents=True, exist_ok=True)
    (_share / rel).write_text("# Title\n\nbody **bold**\n", encoding="utf-8")
    d = await build_descriptor(rel, os.path.join(str(_share), rel))
    assert d["mode"] == "doc"
    assert d["kind"] == "markdown"
    assert d["kind_label"] == "markdown"
    assert d["name"] == "n.md"
    assert '<div class="md-doc">' in d["fragment"]
    assert d["url"] is None and d["note"] is None


async def test_build_descriptor_undecodable_text_raw_null_fragment(_share: Path) -> None:
    rel = "upload/blob.txt"
    (_share / rel).parent.mkdir(parents=True, exist_ok=True)
    (_share / rel).write_bytes(b"\x00\x01\x02\x03")
    d = await build_descriptor(rel, os.path.join(str(_share), rel))
    assert d["mode"] == "raw"
    assert d["fragment"] is None
    assert d["note"] == "No preview available for this file."


async def test_build_descriptor_archive_unsupported(_share: Path) -> None:
    rel = "upload/x.zip"
    (_share / rel).parent.mkdir(parents=True, exist_ok=True)
    (_share / rel).write_bytes(b"PK\x03\x04")
    d = await build_descriptor(rel, os.path.join(str(_share), rel))
    assert d["mode"] == "unsupported"
    assert d["fragment"] is None and d["url"] is None


async def test_build_descriptor_absolute_path_yields_share_path_and_url(_share: Path) -> None:
    """A file given by absolute path comes back with its share path and url."""
    rel = "upload/pic.png"
    full = os.path.realpath(os.path.join(str(_share), rel))
    (_share / rel).parent.mkdir(parents=True, exist_ok=True)
    (_share / rel).write_bytes(b"\x89PNG")
    d = await build_descriptor(full, full)
    assert d["path"] == rel
    assert d["name"] == "pic.png"
    assert d["url"] == "/dav/upload/pic.png?ephemeral=1"


async def test_build_descriptor_absolute_and_relative_paths_converge(_share: Path) -> None:
    """A file given by absolute path and by share path gets the same description."""
    rel = "upload/pic.png"
    full = os.path.realpath(os.path.join(str(_share), rel))
    (_share / rel).parent.mkdir(parents=True, exist_ok=True)
    (_share / rel).write_bytes(b"\x89PNG")
    d_abs = await build_descriptor(full, full)
    d_rel = await build_descriptor(rel, full)
    assert d_abs == d_rel


async def test_build_descriptor_media_url_percent_encoded(_share: Path) -> None:
    """A media file url has its special characters escaped."""
    rel = "upload/pic & 1.png"
    (_share / rel).parent.mkdir(parents=True, exist_ok=True)
    (_share / rel).write_bytes(b"\x89PNG")
    d = await build_descriptor(rel, os.path.join(str(_share), rel))
    assert d["mode"] == "native"
    assert d["url"] == "/dav/upload/pic%20%26%201.png?ephemeral=1"


async def test_build_descriptor_cached_presentation_uses_pdf_asset(_share: Path) -> None:
    """A saved presentation points at its PDF while keeping the original download path."""
    rel = "upload/deck.pptx"
    full = _share / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(b"PK\x03\x04 fake pptx")
    key = cache.key_for(rel, int(full.stat().st_mtime))
    await cache.write_doc(key, "", {cache.OFFICE_PDF_ASSET: b"%PDF"}, {"mode": "frame"})

    d = await build_descriptor(rel, str(full))
    assert d["kind"] == "office_ppt"
    assert d["mode"] == "frame"
    assert d["path"] == rel
    assert d["url"] == f"/view/doc/{key}/{cache.OFFICE_PDF_ASSET}"


async def test_build_descriptor_office_live_conversion_is_converting(
    _share: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file with a running conversion comes back as converting, with no task id."""
    rel = "upload/live.docx"
    (_share / rel).parent.mkdir(parents=True, exist_ok=True)
    (_share / rel).write_bytes(b"PK\x03\x04 fake docx")

    async def running(_key: str, _path: str, _full: str, _kind: RenderKind, _timeout: float) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(office, "_conversion", running)
    d = await build_descriptor(rel, os.path.join(str(_share), rel))
    assert d["mode"] == "converting"
    assert "task_id" not in d
