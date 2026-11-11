from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import ClassVar, override

from app.lib.shared import paths
from app.lib.tool.base import Param, Tool, ToolResult
from app.lib.tts.tts import generate_tts
from app.resources.resources import dav_base_url

__all__ = ["RESOURCE_PATH_PREFIX", "tts_tool"]

RESOURCE_PATH_PREFIX = "Speech generated and saved to resource path: "

_TTS_OUTPUT_DIR = "tts"


async def _synthesize(text: str, format: str) -> ToolResult:
    """Turn text into speech and save it in this server's share.

    Args:
        text: The words to say.
        format: The audio file type.

    Returns:
        A message with the saved path and how to download the file, plus its resource path.

    Raises:
        RuntimeError: If the speech could not be made.
    """
    resources_dir = paths.resources_dir()
    output_file = os.path.join(
        resources_dir,
        _TTS_OUTPUT_DIR,
        f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}",
    )
    await asyncio.to_thread(os.makedirs, os.path.dirname(output_file), exist_ok=True)

    output, status = await generate_tts(
        text, format=format, output_file=output_file, verbose=True
    )
    if status != 0:
        raise RuntimeError(output)

    resource_path = os.path.relpath(output, resources_dir)
    message = (
        f"{RESOURCE_PATH_PREFIX}{resource_path}\n"
        f"Download it with:\n"
        f"  curl -o /tmp/{os.path.basename(output)} {dav_base_url()}/dav/{resource_path}\n"
        "See the send_or_receive_file tool for file transfer details."
    )
    return ToolResult(text=message, files=(resource_path,))


class _TtsTool(Tool):
    name: ClassVar[str] = "tts"
    result_render_type: ClassVar[str] = "text"
    description: ClassVar[str] = (
        "Generate speech from text and save the audio file into this server's "
        "resource. Use when the user wants an audio version of some text."
    )
    summary: ClassVar[str] = "Local text-to-speech (sherpa / voxcpm)"
    params: ClassVar[tuple[Param, ...]] = (
        Param("text", str, kind="longtext", description="Text to convert to speech."),
        Param(
            "format",
            str,
            "opus",
            enum=("mp3", "opus", "aac", "flac", "wav", "pcm"),
            description="Output audio format.",
        ),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        return await _synthesize(params["text"], params["format"])


tts_tool = _TtsTool()
