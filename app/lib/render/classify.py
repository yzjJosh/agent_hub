"""Works out what kind of file something is."""

from __future__ import annotations

import asyncio
import os
from enum import Enum

__all__ = ["EXT_TO_KIND", "RenderKind", "classify", "classify_sync", "kind_label"]

_HEAD_BYTES = 8192


class RenderKind(str, Enum):
    """A kind of file the viewer knows about."""

    MARKDOWN = "markdown"
    CODE = "code"
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    YAML = "yaml"
    XML = "xml"
    HTML = "html"
    PDF = "pdf"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    OFFICE_DOC = "office_doc"
    OFFICE_PPT = "office_ppt"
    XLSX = "xlsx"
    XLS = "xls"
    ARCHIVE = "archive"
    TEXT = "text"
    UNKNOWN = "unknown"


def _build_table() -> dict[str, RenderKind]:
    table: dict[str, RenderKind] = {}

    def add(kind: RenderKind, *exts: str) -> None:
        for ext in exts:
            table[ext] = kind

    add(RenderKind.MARKDOWN, "md", "markdown")
    add(RenderKind.CSV, "csv")
    add(RenderKind.JSON, "json")
    add(RenderKind.JSONL, "jsonl")
    add(RenderKind.YAML, "yaml", "yml")
    add(RenderKind.XML, "xml")
    add(RenderKind.HTML, "html", "htm")
    add(RenderKind.PDF, "pdf")
    add(RenderKind.IMAGE, "png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif", "heic", "heif")
    add(RenderKind.VIDEO, "mp4", "webm", "mkv", "mov", "avi")
    add(RenderKind.AUDIO, "mp3", "wav", "m4a", "ogg", "flac", "opus")
    add(RenderKind.OFFICE_DOC, "doc", "docx")
    add(RenderKind.OFFICE_PPT, "ppt", "pptx")
    add(RenderKind.XLSX, "xlsx")
    add(RenderKind.XLS, "xls")
    add(RenderKind.ARCHIVE, "zip", "tar", "gz", "tgz", "bz2", "xz", "7z", "rar")
    add(RenderKind.TEXT, "txt", "log")
    add(RenderKind.CODE, "py", "js", "jsx", "ts", "tsx", "go", "rs", "java", "c", "h", "cpp",
        "hpp", "cc", "cs", "rb", "php", "sh", "bash", "zsh", "fish", "sql", "toml", "ini",
        "cfg", "conf", "dart", "kt", "swift", "scala", "lua", "pl", "r", "m", "mm", "proto",
        "graphql", "css", "scss", "less", "vue", "svelte")
    # These types are not shown in the viewer, only downloaded.
    add(RenderKind.UNKNOWN, "odt", "ods", "odp", "rtf", "svg")
    return table


EXT_TO_KIND: dict[str, RenderKind] = _build_table()

_MAGIC = (
    (b"%PDF", RenderKind.PDF),
    (b"\x89PNG", RenderKind.IMAGE),
    (b"\xff\xd8\xff", RenderKind.IMAGE),
    (b"PK", RenderKind.ARCHIVE),
)


def _extension(full_path: str) -> str:
    return os.path.splitext(os.path.basename(full_path))[1].lstrip(".").lower()


def _head(full_path: str, count: int) -> bytes | None:
    try:
        fd = os.open(full_path, os.O_RDONLY | os.O_NOATIME)
        with os.fdopen(fd, "rb") as fh:
            return fh.read(count)
    except OSError:
        return None


def _is_text(full_path: str) -> bool:
    head = _head(full_path, _HEAD_BYTES)
    if head is None or b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _classify_sync(full_path: str) -> RenderKind:
    ext = _extension(full_path)
    if ext:
        if ext in EXT_TO_KIND:
            return EXT_TO_KIND[ext]
        return RenderKind.CODE if _is_text(full_path) else RenderKind.UNKNOWN
    head = _head(full_path, 16) or b""
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            return kind
    return RenderKind.TEXT if _is_text(full_path) else RenderKind.UNKNOWN


async def classify(full_path: str) -> RenderKind:
    """Say what kind a file is.

    A file the app does not know is text when it reads as text, and unknown
    when it does not.

    Args:
        full_path: Where the file is on disk.

    Returns:
        The kind of the file.
    """
    return await asyncio.to_thread(_classify_sync, full_path)


def classify_sync(full_path: str) -> RenderKind:
    """Say what kind a file is, without an event loop.

    This is for code that already runs off the loop, such as a listing inside
    a worker thread. Never call it on the event loop.

    Args:
        full_path: Where the file is on disk.

    Returns:
        The kind of the file.
    """
    return _classify_sync(full_path)


def kind_label(kind: RenderKind, name: str, language: str | None) -> str:
    """Get the type name shown to people.

    Args:
        kind: What sort of file this is.
        name: The file name.
        language: The code language, or None.

    Returns:
        A short name for the file type.
    """
    if kind is RenderKind.CODE:
        return language if language else _extension(name)
    if kind in (RenderKind.OFFICE_DOC, RenderKind.OFFICE_PPT):
        return _extension(name)
    if kind is RenderKind.UNKNOWN:
        return "file"
    return kind.value
