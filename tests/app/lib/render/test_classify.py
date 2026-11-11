"""Tests for file kind classification."""

import os
import time
from pathlib import Path

import pytest

from app.lib.render.classify import EXT_TO_KIND, RenderKind, classify, classify_sync, kind_label

_DAY = 86400
_OLD_AGE = 3 * _DAY
_STILL_OLD_AGE = 2 * _DAY


@pytest.mark.parametrize(
    ("name", "data", "expected"),
    [
        ("magic", b"%PDF-1.4\n", RenderKind.PDF),
        ("text", b"hello world\n", RenderKind.TEXT),
        ("x.weird", b"print(1)\n", RenderKind.CODE),
        ("blob", b"\x00\x01\x02\x03" * 32, RenderKind.UNKNOWN),
    ],
)
async def test_classify_does_not_refresh_atime(
    tmp_path: Path, name: str, data: bytes, expected: RenderKind,
) -> None:
    target = tmp_path / name
    target.write_bytes(data)
    old = time.time() - _OLD_AGE
    os.utime(target, (old, old))
    assert await classify(str(target)) is expected
    assert os.path.getatime(str(target)) < time.time() - _STILL_OLD_AGE


def test_classify_sync_does_not_refresh_atime(tmp_path: Path) -> None:
    target = tmp_path / "notes"
    target.write_text("hello world\n", encoding="utf-8")
    old = time.time() - _OLD_AGE
    os.utime(target, (old, old))
    assert classify_sync(str(target)) is RenderKind.TEXT
    assert os.path.getatime(str(target)) < time.time() - _STILL_OLD_AGE


@pytest.mark.parametrize(
    ("ext", "kind"),
    [
        ("md", RenderKind.MARKDOWN), ("markdown", RenderKind.MARKDOWN),
        ("csv", RenderKind.CSV), ("json", RenderKind.JSON), ("jsonl", RenderKind.JSONL),
        ("yaml", RenderKind.YAML), ("yml", RenderKind.YAML), ("xml", RenderKind.XML),
        ("html", RenderKind.HTML), ("htm", RenderKind.HTML), ("pdf", RenderKind.PDF),
        ("png", RenderKind.IMAGE), ("heic", RenderKind.IMAGE),
        ("mp4", RenderKind.VIDEO), ("mp3", RenderKind.AUDIO), ("opus", RenderKind.AUDIO),
        ("doc", RenderKind.OFFICE_DOC), ("docx", RenderKind.OFFICE_DOC),
        ("pptx", RenderKind.OFFICE_PPT), ("xlsx", RenderKind.XLSX), ("xls", RenderKind.XLS),
        ("zip", RenderKind.ARCHIVE),
        ("txt", RenderKind.TEXT), ("py", RenderKind.CODE), ("js", RenderKind.CODE),
        ("odt", RenderKind.UNKNOWN), ("ods", RenderKind.UNKNOWN),
        ("odp", RenderKind.UNKNOWN), ("rtf", RenderKind.UNKNOWN),
        ("svg", RenderKind.UNKNOWN),
    ],
)
def test_extension_mapping(ext: str, kind: RenderKind) -> None:
    assert EXT_TO_KIND[ext] is kind


def test_kind_values_are_descriptor_tokens() -> None:
    assert {k.value for k in RenderKind} == {
        "markdown", "code", "csv", "json", "jsonl", "yaml", "xml", "html", "pdf",
        "image", "video", "audio", "office_doc", "office_ppt", "xlsx", "xls",
        "archive", "text", "unknown",
    }


async def test_extensionless_magic_sniffing(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"%PDF-1.4\n...")
    assert await classify(str(tmp_path / "a")) is RenderKind.PDF
    (tmp_path / "b").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    assert await classify(str(tmp_path / "b")) is RenderKind.IMAGE
    (tmp_path / "c").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 12)
    assert await classify(str(tmp_path / "c")) is RenderKind.IMAGE
    (tmp_path / "d").write_bytes(b"PK\x03\x04" + b"\x00" * 12)
    assert await classify(str(tmp_path / "d")) is RenderKind.ARCHIVE


async def test_extensionless_text_and_binary(tmp_path: Path) -> None:
    (tmp_path / "notes").write_text("hello world\n", encoding="utf-8")
    assert await classify(str(tmp_path / "notes")) is RenderKind.TEXT
    (tmp_path / "blob").write_bytes(b"\x00\x01\x02\x03" * 32)
    assert await classify(str(tmp_path / "blob")) is RenderKind.UNKNOWN


async def test_empty_file_is_text(tmp_path: Path) -> None:
    (tmp_path / "empty").write_bytes(b"")
    assert await classify(str(tmp_path / "empty")) is RenderKind.TEXT


async def test_unknown_extension_text_is_code(tmp_path: Path) -> None:
    (tmp_path / "x.weird").write_text("print(1)\n", encoding="utf-8")
    assert await classify(str(tmp_path / "x.weird")) is RenderKind.CODE


async def test_unknown_extension_binary_is_unknown(tmp_path: Path) -> None:
    (tmp_path / "x.weird").write_bytes(b"\x00\x01\x02\x03" * 32)
    assert await classify(str(tmp_path / "x.weird")) is RenderKind.UNKNOWN


def test_kind_label_rules() -> None:
    assert kind_label(RenderKind.MARKDOWN, "a.md", None) == "markdown"
    assert kind_label(RenderKind.PDF, "a.pdf", None) == "pdf"
    assert kind_label(RenderKind.CODE, "a.py", "python") == "python"
    assert kind_label(RenderKind.CODE, "a.weird", None) == "weird"
    assert kind_label(RenderKind.OFFICE_DOC, "a.docx", None) == "docx"
    assert kind_label(RenderKind.OFFICE_PPT, "a.ppt", None) == "ppt"
    assert kind_label(RenderKind.UNKNOWN, "a", None) == "file"
    assert kind_label(RenderKind.TEXT, "a", None) == "text"
