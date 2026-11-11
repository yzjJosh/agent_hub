"""Works out what kind of content some bytes hold."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import app.lib.text_type.detector as detector
from app.lib.render.classify import RenderKind

__all__ = [
    "BINARY_KINDS",
    "Detection",
    "TEXT_FAMILY_KINDS",
    "detect",
    "language_extension",
]

_LANGUAGE_EXT = {"python": "py", "javascript": "js", "sh": "sh", "sql": "sql", "go": "go"}

BINARY_KINDS = frozenset({
    RenderKind.PDF, RenderKind.IMAGE, RenderKind.VIDEO, RenderKind.AUDIO,
    RenderKind.OFFICE_DOC, RenderKind.OFFICE_PPT, RenderKind.XLSX, RenderKind.XLS,
    RenderKind.ARCHIVE,
})
TEXT_FAMILY_KINDS = frozenset({
    RenderKind.MARKDOWN, RenderKind.CODE, RenderKind.TEXT, RenderKind.CSV,
    RenderKind.JSON, RenderKind.JSONL, RenderKind.YAML, RenderKind.XML, RenderKind.HTML,
})

_FTYPE_VIDEO = (b"isom", b"mp41", b"m4v ")


@dataclass(frozen=True)
class Detection:
    kind: RenderKind
    language: str | None = None     # code only
    subtype: str | None = None      # binary kinds only


async def detect(data: bytes) -> Detection:
    """Say what kind of content some bytes hold.

    Args:
        data: The bytes to look at.

    Returns:
        The kind of the content, with the code language or the file type when
        there is one.
    """
    if not data:
        return Detection(RenderKind.UNKNOWN)
    binary = await asyncio.to_thread(_sniff_binary, data)
    if binary is not None:
        return binary
    if await asyncio.to_thread(_decodable, data) is None:
        return Detection(RenderKind.UNKNOWN)
    return await _sniff_text(data)


def _decodable(data: bytes) -> str | None:
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _sniff_binary(data: bytes) -> Detection | None:
    if data[:4] == b"%PDF":
        return Detection(RenderKind.PDF, subtype="pdf")
    if data[:4] == b"\x89PNG":
        return Detection(RenderKind.IMAGE, subtype="png")
    if data[:3] == b"\xff\xd8\xff":
        return Detection(RenderKind.IMAGE, subtype="jpeg")
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand == b"M4A ":
            return Detection(RenderKind.AUDIO, subtype="m4a")
        if brand in _FTYPE_VIDEO:
            return Detection(RenderKind.VIDEO, subtype="mp4")
        if brand == b"qt  ":
            return Detection(RenderKind.VIDEO, subtype="mov")
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return Detection(RenderKind.VIDEO, subtype="webm")
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return Detection(RenderKind.AUDIO, subtype="wav")
    if data[:4] == b"OggS":
        return Detection(RenderKind.AUDIO, subtype="opus" if b"OpusHead" in data[:128] else "ogg")
    if data[:3] == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return Detection(RenderKind.AUDIO, subtype="mp3")
    if data[:4] == b"fLaC":
        return Detection(RenderKind.AUDIO, subtype="flac")
    if data[:2] == b"PK":
        if b"word/" in data:
            return Detection(RenderKind.OFFICE_DOC, subtype="docx")
        if b"ppt/" in data:
            return Detection(RenderKind.OFFICE_PPT, subtype="pptx")
        if b"xl/" in data:
            return Detection(RenderKind.XLSX, subtype="xlsx")
        return Detection(RenderKind.ARCHIVE, subtype="zip")
    if data[:4] == b"\xd0\xcf\x11\xe0":
        if b"WordDocument" in data:
            return Detection(RenderKind.OFFICE_DOC, subtype="doc")
        if b"PowerPoint Document" in data:
            return Detection(RenderKind.OFFICE_PPT, subtype="ppt")
        if b"Workbook" in data:
            return Detection(RenderKind.XLS, subtype="xls")
        return Detection(RenderKind.UNKNOWN)
    return None


async def _sniff_text(data: bytes) -> Detection:
    """Say what kind of text some bytes hold.

    Args:
        data: The text bytes to look at.

    Returns:
        The kind of the text, with its code language when it has one.
    """
    label = await detector.identify(data)
    return Detection(detector.kind_for_label(label), detector.language_for_label(label))


def language_extension(language: str | None) -> str | None:
    """Pick the file ending for a code language.

    Args:
        language: The name of the language, or None.

    Returns:
        The file ending, or None when the language is not known.
    """
    return _LANGUAGE_EXT.get(language or "")
