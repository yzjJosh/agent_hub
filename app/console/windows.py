from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Iterator
from typing import BinaryIO

__all__ = ["OFFSET_MAX_BYTES", "backward_window", "forward_window", "parse_int", "parse_max_bytes", "parse_offset"]

_CHUNK_BYTES = 1048576
# The biggest offset a request may ask for. A much larger number would break
# the file seek and turn the reply into something other than JSON.
OFFSET_MAX_BYTES = 2**40
# The most bytes one request may ask to send back.
_WINDOW_MAX_BYTES = 4194304


def _utf8_safe_end(b: bytes) -> int:
    """Find how much of the bytes read as whole text.

    Args:
        b: The bytes to check.

    Returns:
        How many bytes from the start read as whole text.
    """
    for i in range(len(b), max(len(b) - 4, -1), -1):
        try:
            b[:i].decode("utf-8")
            return i
        except UnicodeDecodeError:
            continue
    return len(b)


def _source_size(source: bytes | str | os.PathLike[str]) -> int:
    """Get how big the source is.

    Args:
        source: The bytes to read, or the path of a file.

    Returns:
        The size in bytes.
    """
    return len(source) if isinstance(source, bytes) else os.path.getsize(source)


@contextlib.contextmanager
def _source_handle(source: bytes | str | os.PathLike[str]) -> Iterator[BinaryIO | None]:
    """Open the source for reading.

    Args:
        source: The bytes to read, or the path of a file.

    Yields:
        The open file, or None when the source is bytes.
    """
    if isinstance(source, bytes):
        yield None
        return
    with open(source, "rb") as handle:
        yield handle


def _read_range(source: bytes, handle: BinaryIO | None, start: int, end: int) -> bytes:
    """Read one stretch of the source.

    Args:
        source: The bytes to read.
        handle: The open file, or None when the source is bytes.
        start: The first byte to read.
        end: The byte to stop before.

    Returns:
        The bytes in that stretch.
    """
    if handle is None:
        return source[start:end]
    handle.seek(start)
    return handle.read(end - start)


def _count_lines(source: bytes, handle: BinaryIO | None) -> int:
    """Count the lines in the source.

    A last line with no line break still counts. An empty source has no lines.

    Args:
        source: The bytes to read.
        handle: The open file, or None when the source is bytes.

    Returns:
        How many lines there are.
    """
    if handle is None:
        if not source:
            return 0
        return source.count(b"\n") + (not source.endswith(b"\n"))
    handle.seek(0)
    total = 0
    last = b""
    while chunk := handle.read(_CHUNK_BYTES):
        total += chunk.count(b"\n")
        last = chunk[-1:]
    if last and last != b"\n":
        total += 1
    return total


def _walk_newlines(source: bytes, handle: BinaryIO | None, end: int, lines: int) -> tuple[int, int] | None:
    """Find where the last whole lines before a point start and stop.

    Args:
        source: The bytes to read.
        handle: The open file, or None when the source is bytes.
        end: The byte to look back from.
        lines: How many lines to take.

    Returns:
        Where the lines start and where they stop, or None when there is no
        line break before that point.
    """
    needed = lines + 1
    last_nl: int | None = None
    seen = 0
    pos = end
    while pos > 0:
        begin = max(0, pos - _CHUNK_BYTES)
        chunk = _read_range(source, handle, begin, pos)
        count = chunk.count(b"\n")
        if last_nl is None and count:
            last_nl = begin + chunk.rfind(b"\n")
        if seen + count >= needed:
            cut = len(chunk)
            for _ in range(needed - seen):
                cut = chunk.rfind(b"\n", 0, cut)
            return begin + cut + 1, last_nl + 1
        seen += count
        pos = begin
    if last_nl is None:
        return None
    return 0, last_nl + 1


async def forward_window(data: bytes, offset: int, size: int) -> dict:
    """Make a window of text out of bytes read from a file.

    A part of a letter at either end is dropped.

    Args:
        data: The bytes that were read.
        offset: Where in the file the bytes start.
        size: How big the whole file is.

    Returns:
        The text, where it sits in the file, and whether more is left.
    """
    return await asyncio.to_thread(_forward_window, data, offset, size)


def _forward_window(data: bytes, offset: int, size: int) -> dict:
    """Turn bytes read from a file into a window of text.

    Args:
        data: The bytes that were read.
        offset: Where in the file the bytes start.
        size: How big the whole file is.

    Returns:
        The text, where it sits in the file, and whether more is left.
    """
    start = 0
    if offset > 0 and data and (data[0] & 0xC0) == 0x80:
        while start < len(data) and (data[start] & 0xC0) == 0x80:
            start += 1
    body = data[start:]
    end = _utf8_safe_end(body)
    return {
        "offset": offset + start,
        "size": size,
        "content": body[:end].decode("utf-8", "replace"),
        "content_bytes": end,
        "truncated": (offset + len(data)) < size,
    }


async def backward_window(source: bytes | str | os.PathLike[str], lines: int, end_offset: int) -> dict:
    """Get the last whole lines before a point in a file.

    Args:
        source: The bytes to read, or the path of a file.
        lines: How many lines to take.
        end_offset: The byte to look back from, 0 means the end of the file.

    Returns:
        The text, where it sits in the file, and how many lines the file has.
    """
    return await asyncio.to_thread(_backward_window, source, lines, end_offset)


def _backward_window(source: bytes | str | os.PathLike[str], lines: int, end_offset: int) -> dict:
    """Read the last whole lines before a point in a file.

    Args:
        source: The bytes to read, or the path of a file.
        lines: How many lines to take.
        end_offset: The byte to look back from, 0 means the end of the file.

    Returns:
        The text, where it sits in the file, and how many lines the file has.
    """
    size = _source_size(source)
    end = size if end_offset == 0 else min(end_offset, size)
    with _source_handle(source) as handle:
        bounds = _walk_newlines(source, handle, end, lines)
        if bounds is None:
            return {
                "offset": end, "size": size, "content": "", "content_bytes": 0,
                "total_lines": _count_lines(source, handle), "head_reached": True,
            }
        start, end_byte = bounds
        return {
            "offset": start,
            "size": size,
            "content": _read_range(source, handle, start, end_byte).decode("utf-8", "replace"),
            "content_bytes": end_byte - start,
            "total_lines": _count_lines(source, handle),
            "head_reached": start == 0,
        }


def parse_int(raw: str | None, default: int) -> int | None:
    """Read a whole number from the request.

    Args:
        raw: The value from the request, or None.
        default: What to use when the value is missing.

    Returns:
        The number, or None when the value is not a whole number.
    """
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return None


def parse_offset(raw: str | None) -> int | None:
    """Read a byte offset from the request.

    Args:
        raw: The value from the request, or None.

    Returns:
        The offset, 0 when it is missing, or None when it is too big or is
        not a whole number.
    """
    value = parse_int(raw, 0)
    return value if value is not None and 0 <= value <= OFFSET_MAX_BYTES else None


def parse_max_bytes(raw: str | None, default: int) -> int | None:
    """Read how many bytes to send back at most.

    Args:
        raw: The value from the request, or None.
        default: What to use when the value is missing.

    Returns:
        The byte count, or None when it is out of range.
    """
    value = parse_int(raw, default)
    return value if value is not None and 1 <= value <= _WINDOW_MAX_BYTES else None
