"""Tests for content type detection."""

import pytest

from app.lib.render.classify import RenderKind
from app.lib.render.detect import detect, language_extension

_PYTHON_SOURCE = b'def greet(name):\n    return f"Hello, {name}"\n\nprint(greet("world"))\n'
_JSON = b'{\n  "name": "agent",\n  "version": "0.1.0",\n  "deps": ["fastmcp", "httpx"]\n}\n'
_YAML = b"name: agent\nversion: 0.1.0\nitems:\n  - one\n  - two\n"
_CSV = b"name,amount\nalpha,1\nbeta,2\ngamma,3\n"
_MARKDOWN = (
    b"# Notes\n\nSome **bold** text and a [link](https://example.com).\n\n- one\n- two\n"
)
_HTML = (
    b"<!doctype html>\n<html>\n<head><title>demo</title></head>"
    b"\n<body><p>hello</p></body>\n</html>\n"
)
_SQL = (
    b"CREATE TABLE items (name TEXT, amount INT);\n"
    b"SELECT name FROM items WHERE amount > 1;\n"
)
_SH = b'#!/bin/sh\necho "starting"\nsleep 1\nexit 0\n'
_PROSE = (
    "The weather in the city was mild on Tuesday, and many people walked "
    "along the river after work. Shops stayed open late, and the park "
    "was full of families."
).encode("utf-8")


@pytest.mark.parametrize(
    ("data", "kind", "subtype"),
    [
        (b"%PDF-1.7\n\x00\x00", RenderKind.PDF, "pdf"),
        (b"\x89PNG\r\n\x1a\n\x00\x00\x00", RenderKind.IMAGE, "png"),
        (b"\xff\xd8\xff\xe0\x00\x00", RenderKind.IMAGE, "jpeg"),
        (b"\x00\x00\x00\x18ftypisom", RenderKind.VIDEO, "mp4"),
        (b"\x00\x00\x00\x18ftypmp41", RenderKind.VIDEO, "mp4"),
        (b"\x00\x00\x00\x18ftypqt  ", RenderKind.VIDEO, "mov"),
        (b"\x00\x00\x00\x18ftypM4A ", RenderKind.AUDIO, "m4a"),
        (b"\x1a\x45\xdf\xa3\x01\x00\x00", RenderKind.VIDEO, "webm"),
        (b"RIFF\x00\x00\x00\x00WAVEfmt", RenderKind.AUDIO, "wav"),
        (b"OggS\x00\x00\x00\x00\x00\x00", RenderKind.AUDIO, "ogg"),
        (b"OggS\x00\x00\x00\x00OpusHead", RenderKind.AUDIO, "opus"),
        (b"ID3\x03\x00\x00\x00\x00", RenderKind.AUDIO, "mp3"),
        (b"\xff\xfb\x90\x00\x00\x00", RenderKind.AUDIO, "mp3"),
        (b"fLaC\x00\x00\x00\x00", RenderKind.AUDIO, "flac"),
        (b"PK\x03\x04" + b"word/_rels/", RenderKind.OFFICE_DOC, "docx"),
        (b"PK\x03\x04" + b"ppt/slide1.xml", RenderKind.OFFICE_PPT, "pptx"),
        (b"PK\x03\x04" + b"xl/workbook.xml", RenderKind.XLSX, "xlsx"),
        (b"PK\x03\x04" + b"notes.txt", RenderKind.ARCHIVE, "zip"),
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1WordDocument", RenderKind.OFFICE_DOC, "doc"),
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1PowerPoint Document", RenderKind.OFFICE_PPT, "ppt"),
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1Workbook", RenderKind.XLS, "xls"),
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1\x00\x00\x00", RenderKind.UNKNOWN, None),
    ],
)
async def test_binary_magics(data: bytes, kind: RenderKind, subtype: str | None) -> None:
    det = await detect(data)
    assert det.kind is kind, (data, det)
    assert det.subtype == subtype, (data, det)


async def test_text_gate_unknown() -> None:
    assert (await detect(b"")).kind is RenderKind.UNKNOWN
    assert (await detect(b"a" * 70000 + b"\x00" + b"b" * 10)).kind is RenderKind.UNKNOWN  # a zero byte far into the data
    assert (await detect(b"a" * 70000 + b"\xc3\x28\n")).kind is RenderKind.UNKNOWN        # bad text bytes far into the data


def test_language_extension() -> None:
    """A known code language maps to its file ending, an unknown one to nothing."""
    assert language_extension("sh") == "sh"
    assert language_extension("python") == "py"
    assert language_extension("brainfuck") is None
    assert language_extension(None) is None


@pytest.mark.live
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("data", "kind", "language", "subtype"),
    [
        (_PYTHON_SOURCE, RenderKind.CODE, "python", None),
        (_JSON, RenderKind.JSON, None, None),
        (_YAML, RenderKind.YAML, None, None),
        (_CSV, RenderKind.CSV, None, None),
        (_MARKDOWN, RenderKind.MARKDOWN, None, None),
        (_HTML, RenderKind.HTML, None, None),
        (_SQL, RenderKind.CODE, "sql", None),
        (_SH, RenderKind.CODE, "sh", None),
        (_PROSE, RenderKind.TEXT, None, None),
        (b"\x89PNG\r\n\x1a\n\x00", RenderKind.IMAGE, None, "png"),
    ],
)
async def test_live_detect_matrix(
    data: bytes, kind: RenderKind, language: str | None, subtype: str | None
) -> None:
    """Real bytes get the right kind, text and binary alike."""
    det = await detect(data)
    assert det.kind is kind, (data, det)
    assert det.language == language, (data, det)
    assert det.subtype == subtype, (data, det)
