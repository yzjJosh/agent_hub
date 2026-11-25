"""Load image and video attachments for agent tool results."""

from __future__ import annotations

import asyncio
import base64
import io
import mimetypes
import os
import sys
import traceback

import aiofiles
from aiofiles.tempfile import TemporaryDirectory
from agents import ToolOutputFileContent, ToolOutputImage, ToolOutputText
from PIL import Image

from app.lib.cli.cli_runner import run_cli
from app.lib.shared.paths import is_image_mime, is_video_mime, mime_type, resolve_path, resources_dir

__all__ = ["load_media_attachment_parts"]

_MEDIA_DECODE_TIMEOUT_SECONDS = 30
_ERROR_PATH_PREVIEW_LENGTH = 100


def _encode_data_url(raw_bytes: bytes, mime: str) -> str:
    return "data:" + mime + ";base64," + base64.b64encode(raw_bytes).decode()


def _encode_image_data_url(raw_bytes: bytes, mime: str) -> str:
    with Image.open(io.BytesIO(raw_bytes)) as image:
        image.load()
        detected_mime = Image.MIME.get(image.format or "", mime)
    return _encode_data_url(raw_bytes, detected_mime)


async def _build_media_part(
    raw_bytes: bytes,
    mime: str,
    filename: str,
) -> ToolOutputImage | ToolOutputFileContent:
    if is_image_mime(mime):
        image_url = await asyncio.to_thread(_encode_image_data_url, raw_bytes, mime)
        return ToolOutputImage(image_url=image_url)
    async with TemporaryDirectory(prefix="agent-media-") as directory:
        video_path = os.path.join(directory, filename)
        async with aiofiles.open(video_path, "wb") as attachment_file:
            await attachment_file.write(raw_bytes)
        decode_result = await run_cli(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-i",
                video_path,
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            "attachment video decode",
            timeout=_MEDIA_DECODE_TIMEOUT_SECONDS,
        )
        if decode_result.returncode != 0:
            raise ValueError(f"Video decode failed. {decode_result.stderr}")
    video_url = await asyncio.to_thread(_encode_data_url, raw_bytes, mime)
    return ToolOutputFileContent(file_data=video_url, filename=filename)


def _build_delivery_error_part(path: str, reason: str, error: Exception) -> ToolOutputText:
    print(f"[Agent] Attachment delivery failed for {path[:_ERROR_PATH_PREVIEW_LENGTH]}", file=sys.stderr)
    traceback.print_exception(error, file=sys.stderr)
    return ToolOutputText(text=f"Attachment delivery failed for {path!r}. {reason}")


async def _load_attachment_parts(
    path: str,
) -> list[ToolOutputText | ToolOutputImage | ToolOutputFileContent]:
    guessed_mime = mimetypes.guess_type(path)[0]
    is_media = guessed_mime is None or is_image_mime(guessed_mime) or is_video_mime(guessed_mime)
    try:
        candidate = path if os.path.isabs(path) else await resolve_path(resources_dir(), path)
        if candidate is None:
            raise FileNotFoundError("Attachment path is unavailable.")
        async with aiofiles.open(candidate, "rb") as attachment_file:
            raw_bytes = await attachment_file.read()
        detected_mime = await mime_type(candidate) or guessed_mime
        if not is_image_mime(detected_mime) and not is_video_mime(detected_mime):
            if is_image_mime(guessed_mime) or is_video_mime(guessed_mime):
                raise ValueError("Attachment is not valid media.")
            return []
        is_media = True
        assert detected_mime is not None
        return [await _build_media_part(raw_bytes, detected_mime, os.path.basename(candidate))]
    except (OSError, ValueError) as exc:
        if not is_media:
            return []
        return [_build_delivery_error_part(path, "Media could not be read or decoded.", exc)]
    except Exception as exc:
        if not is_media:
            return []
        return [_build_delivery_error_part(path, "Media could not be delivered.", exc)]


async def load_media_attachment_parts(
    files: tuple[str, ...],
) -> list[ToolOutputText | ToolOutputImage | ToolOutputFileContent]:
    """Load image and video attachments and report unavailable media files.

    Args:
        files: Relative resource paths or absolute file paths to load.

    Returns:
        Encoded media parts and delivery error notices for unavailable media.
    """
    parts: list[ToolOutputText | ToolOutputImage | ToolOutputFileContent] = []
    for path in files:
        parts.extend(await _load_attachment_parts(path))
    return parts
