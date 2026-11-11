from __future__ import annotations

import math
import random
import time
from pathlib import Path

import pytest

from app.console.windows import backward_window, forward_window

_CHUNK = 1048576
_LINES_MAX = 5000
_LINES_PER_BLOCK = 5000


async def test_forward_basic() -> None:
    data = b"hello world\n"
    w = await forward_window(data, 0, 12)
    assert w["offset"] == 0 and w["size"] == 12
    assert w["content"] == "hello world\n" and w["content_bytes"] == 12 and w["truncated"] is False


async def test_forward_truncated() -> None:
    w = await forward_window(b"0123", 0, 10)
    assert w["content"] == "0123" and w["content_bytes"] == 4 and w["truncated"] is True


async def test_forward_offset_at_size() -> None:
    w = await forward_window(b"", 4, 4)
    assert w["offset"] == 4 and w["content"] == "" and w["content_bytes"] == 0 and w["truncated"] is False


async def test_forward_offset_beyond_size() -> None:
    w = await forward_window(b"", 20, 10)
    assert w["offset"] == 20 and w["content"] == "" and w["content_bytes"] == 0 and w["truncated"] is False


async def test_forward_offset_mid_char_advances() -> None:
    data = "h\u00e9\n".encode("utf-8")  # b"\x68\xc3\xa9\x0a"
    w = await forward_window(data[2:], 2, len(data))
    assert w["offset"] == 3 and w["content"] == "\n" and w["content_bytes"] == 1


async def test_forward_trailing_partial_char_trimmed() -> None:
    data = "ab\u4e2d".encode("utf-8")
    w = await forward_window(data[:4], 0, len(data))
    assert w["content"] == "ab" and w["content_bytes"] == 2 and w["truncated"] is True


async def test_backward_basic() -> None:
    w = await backward_window(b"l1\nl2\nl3\nl4\nl5\n", 2, 0)
    assert w["content"] == "l4\nl5\n" and w["offset"] == 9 and w["content_bytes"] == 6
    assert w["total_lines"] == 5 and w["head_reached"] is False


async def test_backward_head_reached_no_trailing_newline() -> None:
    w = await backward_window(b"aaa\nbbb\nccc", 200, 0)
    assert w["content"] == "aaa\nbbb\n" and w["offset"] == 0 and w["total_lines"] == 3 and w["head_reached"] is True


async def test_backward_end_offset_mid_line() -> None:
    w = await backward_window(b"aaa\nbbb\nccc\n", 200, 5)
    assert w["content"] == "aaa\n" and w["total_lines"] == 3


async def test_backward_single_line() -> None:
    w = await backward_window(b"only\n", 5, 0)
    assert w["content"] == "only\n" and w["offset"] == 0 and w["content_bytes"] == 5
    assert w["total_lines"] == 1 and w["head_reached"] is True


async def test_backward_empty_file() -> None:
    w = await backward_window(b"", 200, 0)
    assert w["content"] == "" and w["total_lines"] == 0 and w["head_reached"] is True


def _reference_backward(data: bytes, lines: int, end_offset: int) -> dict:
    """Work out the same backward window the simple way."""
    size = len(data)
    end = size if end_offset == 0 else min(end_offset, size)
    seg = data[:end]
    nl: list[int] = []
    i = seg.find(b"\n")
    while i != -1:
        nl.append(i)
        i = seg.find(b"\n", i + 1)
    total = 0 if not data else data.count(b"\n") + (not data.endswith(b"\n"))
    if not nl:
        return {"offset": end, "size": size, "content": "", "content_bytes": 0, "total_lines": total, "head_reached": True}
    last_nl = nl[-1]
    prev_nl = nl[-lines - 1] if len(nl) > lines else -1
    start = prev_nl + 1
    return {
        "offset": start,
        "size": size,
        "content": data[start : last_nl + 1].decode("utf-8", "replace"),
        "content_bytes": last_nl + 1 - start,
        "total_lines": total,
        "head_reached": start == 0,
    }


def _random_log(seed: int, size: int, avg_line: int) -> bytes:
    rng = random.Random(seed)
    parts: list[bytes] = []
    total = 0
    while total < size:
        n = max(1, min(size - total, int(rng.lognormvariate(math.log(avg_line), 0.9))))
        parts.append(rng.randbytes(n) + b"\n")
        total += n + 1
    return b"".join(parts)


def _differential_probes(size: int) -> list[tuple[int, int]]:
    ends = (0, 1, 2, size // 2, size, size + 100) if size < _CHUNK else (0, size // 2, size, size + 100)
    lines_set = (1, 2, 5, 200, _LINES_MAX) if size < _CHUNK else (1, 200, _LINES_MAX)
    return [(end_offset, lines) for end_offset in ends for lines in lines_set]


_DIF_CASES = [(1, 0, 100), (2, 33, 10), (3, 4096, 7), (4, 1024, 1), (5, _CHUNK + 1234, 64), (6, 2 * _CHUNK + 1, 4096), (7, 4 * _CHUNK, 512)]
_DIF_PROBES = [
    (seed, size, avg_line, end_offset, lines)
    for seed, size, avg_line in _DIF_CASES
    for end_offset, lines in _differential_probes(size)
]


@pytest.mark.parametrize("seed,size,avg_line,end_offset,lines", _DIF_PROBES)
async def test_backward_matches_reference(seed: int, size: int, avg_line: int, end_offset: int, lines: int) -> None:
    data = _random_log(seed, size, avg_line)
    assert await backward_window(data, lines, end_offset) == _reference_backward(data, lines, end_offset)


_BOUNDARY_BODIES = (b"x" * (_CHUNK - 1) + b"\n", b"x" * _CHUNK + b"\n", b"\n")


def _boundary_probes(body: bytes) -> list[tuple[int, int]]:
    size = len(body) * 5
    return [(end_offset, lines) for end_offset in (0, _CHUNK, 2 * _CHUNK, size, size + 7) for lines in (1, 2, 3)]


_BOUNDARY_PROBES = [(body, end_offset, lines) for body in _BOUNDARY_BODIES for end_offset, lines in _boundary_probes(body)]


@pytest.mark.parametrize("body,end_offset,lines", _BOUNDARY_PROBES)
async def test_backward_newlines_on_chunk_boundaries(body: bytes, end_offset: int, lines: int) -> None:
    data = body * 5
    assert await backward_window(data, lines, end_offset) == _reference_backward(data, lines, end_offset)


async def test_backward_single_huge_line(tmp_path: Path) -> None:
    data = b"A" * (2 * _CHUNK) + b"\n" + b"tail1\n" + b"tail2\n"
    path = tmp_path / "log"
    path.write_bytes(data)
    w = await backward_window(str(path), 2, 0)
    assert w == _reference_backward(data, 2, 0)
    assert w["content"] == "tail1\ntail2\n"
    w = await backward_window(str(path), 200, 0)
    assert w == _reference_backward(data, 200, 0)
    assert w["head_reached"] is True and w["content_bytes"] == len(data)


async def test_backward_path_matches_bytes_and_empty(tmp_path: Path) -> None:
    path = tmp_path / "log"
    path.write_bytes(b"aaa\nbbb\nccc\n")
    assert await backward_window(str(path), 2, 0) == await backward_window(b"aaa\nbbb\nccc\n", 2, 0)
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    w = await backward_window(str(empty), 100, 0)
    assert w == {"offset": 0, "size": 0, "content": "", "content_bytes": 0, "total_lines": 0, "head_reached": True}


async def test_backward_50mb_log(tmp_path: Path) -> None:
    line = b"2026-08-25T10:00:00Z INFO worker heartbeat seq=" + b"9" * 12 + b" " + b"p" * 39 + b"\n"
    path = tmp_path / "log"
    path.write_bytes(line * _LINES_PER_BLOCK * 100)
    data = path.read_bytes()
    assert len(data) == 50_000_000
    started = time.perf_counter()
    w = await backward_window(str(path), 200, 0)
    assert time.perf_counter() - started < 5.0
    assert w == _reference_backward(data, 200, 0)
    started = time.perf_counter()
    w = await backward_window(str(path), _LINES_MAX, _CHUNK * 3)
    assert time.perf_counter() - started < 5.0
    assert w == _reference_backward(data, _LINES_MAX, _CHUNK * 3)
