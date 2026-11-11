"""Speech endpoint shaped like the OpenAI audio API.

Mounted at /v1, it serves POST /v1/audio/speech. Each call is also kept as a
tool run. The audio file goes in the tts folder of the share, and a file
nobody uses for 30 days is removed. The model, voice, and speed fields are
taken but not used.
"""

from __future__ import annotations

import json
import os
import sys

import aiofiles
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from app.lib.shared.paths import resources_dir
from app.tools.tts import RESOURCE_PATH_PREFIX, tts_tool

__all__ = ["create_openai_tts_app"]

_DEFAULT_FORMAT = "mp3"

_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


def create_openai_tts_app() -> Starlette:
    """Make the speech app, meant to be mounted at /v1.

    Returns:
        An app that serves POST /audio/speech.
    """
    return Starlette(routes=[Route("/audio/speech", _create_speech, methods=["POST"])])


async def _create_speech(request: Request) -> Response:
    """Turn the words in the request into speech.

    Args:
        request: A JSON body holding the words to say and the audio format.

    Returns:
        The audio file, or a JSON error body. A bad request gets a 400, and
        speech that could not be made or read gets a 500.
    """
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        return _error(400, "Content-Type must be application/json")

    try:
        data = await request.json()
    except json.JSONDecodeError:
        return _error(400, "Request body is not valid JSON")
    if not isinstance(data, dict):
        return _error(400, "Request body must be a JSON object")

    text = data.get("input")
    if not isinstance(text, str) or not text:
        return _error(400, "Missing required field: input")

    fmt = data.get("response_format", _DEFAULT_FORMAT)
    if not isinstance(fmt, str) or fmt not in _CONTENT_TYPES:
        return _error(400, f"Unsupported response_format: {fmt!r}")

    task = await tts_tool.run({"text": text, "format": fmt})
    result = await tts_tool.result_text(task)
    resource_path = _parse_resource_path(result)
    if resource_path is None:
        print(f"[tts_server] tts tool failed: {result}", file=sys.stderr)
        return _error(500, result)

    audio = await _read_audio(os.path.join(resources_dir(), resource_path))
    if audio is None:
        print(f"[tts_server] could not read audio file: {resource_path}", file=sys.stderr)
        return _error(500, f"Could not read audio file: {resource_path}")

    return Response(
        audio,
        media_type=_CONTENT_TYPES[fmt],
        headers={"Content-Disposition": f"attachment; filename=speech.{fmt}"},
    )


def _parse_resource_path(result: str) -> str | None:
    """Get the saved file path out of a tts result, or None when the call failed."""
    line = result.split("\n", 1)[0]
    if not line.startswith(RESOURCE_PATH_PREFIX):
        return None
    return line.removeprefix(RESOURCE_PATH_PREFIX)


async def _read_audio(path: str) -> bytes | None:
    """Read an audio file, or give back None when it cannot be read."""
    try:
        async with aiofiles.open(path, "rb") as f:
            return await f.read()
    except OSError:
        return None


def _error(status: int, message: str) -> JSONResponse:
    """Make a JSON reply that carries an error message."""
    return JSONResponse({"error": message}, status_code=status)
