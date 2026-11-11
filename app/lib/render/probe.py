"""Reads extra facts about media files."""

from __future__ import annotations

import asyncio

from PIL import Image
from pillow_heif import register_heif_opener

from app.lib.cli.cli_runner import run_cli
from app.lib.render.classify import RenderKind

# Lets the app open HEIC and HEIF images.
register_heif_opener()

__all__ = ["probe_meta"]

_PROBE_TIMEOUT = 10.0


async def probe_meta(kind: RenderKind, full_path: str) -> dict:
    """Read the extra fact that goes with a media file.

    Args:
        kind: What sort of file this is.
        full_path: Where the file is on disk.

    Returns:
        Page count for a PDF, width and height for an image, or length for
        audio and video. The value is None when it cannot be read. Empty for
        any other kind.
    """
    if kind is RenderKind.PDF:
        return {"pages": await _pages(full_path)}
    if kind is RenderKind.IMAGE:
        return {"dimensions": await _dimensions(full_path)}
    if kind in (RenderKind.AUDIO, RenderKind.VIDEO):
        return {"duration": await _duration(full_path)}
    return {}


async def _pages(full_path: str) -> int | None:
    result = await run_cli(["pdfinfo", full_path], "pdfinfo", timeout=_PROBE_TIMEOUT)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.lower().startswith("pages:"):
            try:
                return int(line.split(":", 1)[1])
            except ValueError:
                return None
    return None


def _size_sync(full_path: str) -> tuple[int, int]:
    with Image.open(full_path) as im:
        return im.size


async def _dimensions(full_path: str) -> list[int] | None:
    try:
        w, h = await asyncio.to_thread(_size_sync, full_path)
        return [w, h]
    except Exception:
        return None


async def _duration(full_path: str) -> float | None:
    result = await run_cli(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            full_path,
        ],
        "ffprobe",
        timeout=_PROBE_TIMEOUT,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    try:
        value = float(result.stdout)
    except ValueError:
        return None
    return value if value >= 0 else None
