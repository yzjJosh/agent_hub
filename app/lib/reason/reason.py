"""Ask the local model a question, with images or video if needed."""

__all__ = ["reason"]

import asyncio
import sys
import os
import json
import base64
import httpx
from PIL import Image
from typing import List, Optional, Tuple

from app.lib.shared import config, paths

_REQUEST_TIMEOUT_S = 600


def _check_image_dimensions(file_path: str) -> bool:
    """Check that an image opens and is big enough.

    This blocks, so callers must run it off the event loop.

    Args:
        file_path: Path to the image file.

    Returns:
        True if the image opens and is at least 32 by 32.
    """
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            if width < 32 or height < 32:
                print(
                    f"Image '{file_path}' has invalid dimensions ({width}x{height}). Minimum dimension is 32x32.",
                    file=sys.stderr,
                )
                return False
    except Exception as e:
        print(f"Cannot read image '{file_path}': {str(e)}", file=sys.stderr)
        return False
    return True


async def _validate_image_files(image_files: List[str]) -> bool:
    """Check that every image file is there, is an image, and is big enough.

    Args:
        image_files: List of image file paths to check.

    Returns:
        True if every file passes.
    """
    for file_path in image_files:
        if not await asyncio.to_thread(os.path.exists, file_path):
            print(f"File '{file_path}' does not exist.", file=sys.stderr)
            return False
        if not await paths.is_image_file(file_path):
            print(f"File '{file_path}' is not a valid image.", file=sys.stderr)
            return False
        if not await asyncio.to_thread(_check_image_dimensions, file_path):
            return False
    return True


def _extract_response_text(response_data: dict) -> str:
    """Pull the answer text out of a backend reply.

    Args:
        response_data: The reply from the backend.

    Returns:
        The answer text, or an empty string when the reply has none.
    """
    choices = response_data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        return message.get("content", "")

    return ""


async def reason(
    prompt: str,
    context: str = "",
    thinking: bool = False,
    json_schema: dict | None = None,
    image_files: list | None = None,
    video_files: list | None = None,
    json_format: bool = False,
) -> Tuple[str, int]:
    """Ask the local model a question and get its answer.

    A call in flight stops when the task waiting on it is cancelled.

    Args:
        prompt: What to ask.
        context: Extra context put in front of the question.
        thinking: If True, asks the thinking backend instead.
        json_schema: The shape the answer must follow.
        image_files: Image files to send with the question.
        video_files: Video files to send with the question.
        json_format: Request a JSON object when no JSON schema is given.

    Returns:
        The answer text and a status code. The code is 0 when it worked and
        1 when it did not.
    """
    try:
        target = config.backend(config.REASON_THINKING if thinking else config.REASON)
    except config.MissingSettingError as exc:
        return str(exc), 1
    payload, error = await _prepare_request(
        prompt, context, target.model, json_schema, image_files or [], video_files or [],
        json_format,
    )
    if error is not None:
        return error
    return await _send_request(target, payload)


async def _prepare_request(
    prompt: str,
    context: str,
    model: str | None,
    json_schema: dict | None,
    image_files: list,
    video_files: list,
    json_format: bool,
) -> Tuple[dict, Optional[Tuple[str, int]]]:
    """Check the inputs and build the request body.

    Args:
        prompt: What to ask.
        context: Extra context put in front of the question.
        model: The model to ask for, or None to leave it to the server.
        json_schema: The shape the answer must follow.
        image_files: Image files to send with the question.
        video_files: Video files to send with the question.
        json_format: Request a JSON object when no JSON schema is given.

    Returns:
        The request body and an error. When the error is not None the
        caller returns it as is, and the body means nothing.
    """
    if not await _validate_image_files(image_files):
        return {}, ("Image validation failed.", 1)
    image_parts = await asyncio.to_thread(_build_image_parts, image_files)
    if image_parts is None:
        return {}, ("Image encoding failed.", 1)

    video_mime_types = await _validate_video_files(video_files)
    if video_mime_types is None:
        return {}, ("Video validation failed.", 1)
    video_parts = await asyncio.to_thread(_build_video_parts, video_files, video_mime_types)
    if video_parts is None:
        return {}, ("Video encoding failed.", 1)

    system_prompt = _build_system_prompt(json_schema, json_format)
    full_prompt = _build_full_prompt(prompt, context)

    payload = _build_payload(
        system_prompt, full_prompt, image_parts + video_parts, json_schema, json_format
    )
    if model:
        payload["model"] = model
    return payload, None


async def _validate_video_files(video_files: list[str]) -> Optional[dict[str, str]]:
    """Check that every video file is there and is a video.

    Args:
        video_files: List of video file paths to check.

    Returns:
        The kind of file found for each path, or None when a file fails.
    """
    mime_types: dict[str, str] = {}
    for file_path in video_files:
        if not await asyncio.to_thread(os.path.exists, file_path):
            print(f"File '{file_path}' does not exist.", file=sys.stderr)
            return None
        mime_type = await paths.mime_type(file_path)
        if not paths.is_video_mime(mime_type):
            print(f"File '{file_path}' is not a valid video.", file=sys.stderr)
            return None
        mime_types[file_path] = mime_type
    return mime_types


def _read_base64(file_path: str) -> Optional[str]:
    """Read a file as text that can be sent to the backend.

    Args:
        file_path: Path to the file.

    Returns:
        The file as text, or None when it cannot be read.
    """
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading file '{file_path}': {str(e)}", file=sys.stderr)
        return None


def _build_image_parts(image_files: list[str]) -> Optional[list[dict]]:
    """Build the image pieces of a request body.

    Args:
        image_files: Paths to the image files.

    Returns:
        One piece per image, or None when a file cannot be read.
    """
    parts: list[dict] = []
    for file_path in image_files:
        data = _read_base64(file_path)
        if data is None:
            return None
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{data}"},
        })
    return parts


def _build_video_parts(
    video_files: list[str], mime_types: dict[str, str]
) -> Optional[list[dict]]:
    """Build the video pieces of a request body.

    Args:
        video_files: Paths to the video files.
        mime_types: The kind of file found for each path.

    Returns:
        One piece per video, or None when a file cannot be read.
    """
    parts: list[dict] = []
    for file_path in video_files:
        data = _read_base64(file_path)
        if data is None:
            return None
        parts.append({
            "type": "video_url",
            "video_url": {"url": f"data:{mime_types[file_path]};base64,{data}"},
        })
    return parts


def _build_system_prompt(json_schema: dict | None, json_format: bool) -> str:
    """Build the instructions sent to the model.

    Args:
        json_schema: The shape the answer must follow, or None.
        json_format: Whether the answer must be a JSON object.

    Returns:
        The instructions text.
    """
    prompt = "You are a component in a deligate software system. Strictly follow the instructions to generate the most accurate output."
    if json_schema:
        schema_str = json.dumps(json_schema, indent=2)
        prompt = f"{prompt}\nIMPORTANT OUTPUT INSTRUCTIONS\nYour output must be raw JSON following this JSON schema. Output ONLY the JSON object. Do NOT wrap it in markdown code fences (``` or ```json). Do NOT include any text before or after the JSON.\n{schema_str}"
    elif json_format:
        prompt = f"{prompt}\nOutput only a valid JSON object, with no markdown or surrounding text."
    return prompt


def _build_full_prompt(prompt: str, context: str) -> str:
    """Join the question and the extra context.

    Args:
        prompt: What to ask.
        context: Extra context put in front of the question.

    Returns:
        The text to send as the question.
    """
    if context:
        return f"Additional Context: {context}\n\nUser Query: {prompt}"
    return prompt


def _build_payload(
    system_prompt: str, full_prompt: str, media_parts: list[dict],
    json_schema: dict | None, json_format: bool,
) -> dict:
    """Build the request body for the backend.

    Args:
        system_prompt: The instructions sent to the model.
        full_prompt: The question to ask.
        media_parts: The image and video pieces to send along.
        json_schema: The shape the answer must follow, or None.
        json_format: Request a JSON object when no JSON schema is given.

    Returns:
        The request body.
    """
    payload = {"messages": [{"role": "system", "content": system_prompt}]}
    if media_parts:
        content = [{"type": "text", "text": full_prompt}, *media_parts]
        payload["messages"].append({"role": "user", "content": content})
    else:
        payload["messages"].append({"role": "user", "content": full_prompt})
    if json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response_schema",
                "strict": True,
                "schema": json_schema,
            },
        }
    elif json_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


async def _send_request(target: config.Backend, payload: dict) -> Tuple[str, int]:
    """Send the request to the backend and wait for its answer.

    Redirects are followed. A cancelled call closes its connection. The key
    is sent only when one is set.

    Args:
        target: The backend to ask.
        payload: The request body.

    Returns:
        The answer text and a status code. The code is 0 when it worked and
        1 when it did not.
    """
    try:
        async with httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT_S, follow_redirects=True
        ) as client:
            response = await client.post(
                f"{target.url}/chat/completions",
                json=payload,
                headers=target.auth_headers(),
            )
            response.raise_for_status()
            response_data = response.json()
        return _extract_response_text(response_data), 0
    except Exception as e:
        return f"Communicating with local LLM failed: {str(e)}", 1
