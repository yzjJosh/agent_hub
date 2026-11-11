"""Async HTTP client for the ComfyUI backend.

The base address comes from the COMFYUI_BACKEND_URL environment variable, read
at call time.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass

import httpx

from app.lib.shared import config

__all__ = ["ComfyUIClient", "GenerationResult"]

_BACKEND_URL_ENV = "COMFYUI_BACKEND_URL"
_BACKEND_PURPOSE = "needed by generate_image, edit_image, generate_video"
_PROMPT_TIMEOUT_S = 720
_CONNECT_TIMEOUT_S = 10
_PROMPT_TIMEOUT = httpx.Timeout(_PROMPT_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S)
_BODY_SNIPPET_LEN = 200


@dataclass(frozen=True)
class GenerationResult:
    """The images a finished prompt saved, in filename order."""

    images: tuple[bytes, ...]
    filenames: tuple[str, ...]


def _backend_url() -> str:
    """The ComfyUI base address to talk to."""
    return config.require_env(_BACKEND_URL_ENV, _BACKEND_PURPOSE).rstrip("/")


def _unreachable(exc: Exception) -> RuntimeError:
    """The error for a request that never got a usable answer."""
    return RuntimeError(f"could not reach ComfyUI: {exc}")


def _rejected(response: httpx.Response) -> RuntimeError:
    """The error for a non-2xx answer."""
    if response.status_code == 503:
        return RuntimeError("GPU is busy, the ComfyUI proxy rejected the request, retry later")
    body = response.text[:_BODY_SNIPPET_LEN]
    return RuntimeError(f"ComfyUI rejected the request: HTTP {response.status_code} {body}")


def _submit_rejection(response: httpx.Response) -> RuntimeError:
    """The error for a rejected prompt, quoting node errors when the answer has them."""
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    node_errors = payload.get("node_errors")
    if isinstance(node_errors, dict) and node_errors:
        details = json.dumps(node_errors, ensure_ascii=False)
        return RuntimeError(f"ComfyUI rejected the prompt: {details}")
    return _rejected(response)


async def _post_json(url: str, body: dict) -> httpx.Response:
    """One JSON POST, with the unreachable error mapped."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=_PROMPT_TIMEOUT) as client:
            return await client.post(url, json=body)
    except httpx.HTTPError as exc:
        raise _unreachable(exc) from exc


def _json(response: httpx.Response) -> dict:
    """The answer's JSON body, or the unreachable error."""
    try:
        return response.json()
    except ValueError as exc:
        raise _unreachable(exc) from exc


def _decode_images(raw_images: list[str]) -> tuple[bytes, ...]:
    """The decoded bytes of each base64 image, in order."""
    return tuple(base64.b64decode(image) for image in raw_images)


def _parse_generation(response: httpx.Response) -> GenerationResult:
    """The decoded images and their names from one finished prompt."""
    payload = _json(response)
    raw_images = payload.get("images") or []
    filenames = payload.get("filenames") or []
    if not raw_images:
        raise RuntimeError("ComfyUI finished without a saved file")
    return GenerationResult(images=_decode_images(raw_images), filenames=tuple(filenames))


class ComfyUIClient:
    """Sends a prompt to the ComfyUI backend and gets the finished images back."""

    async def generate(self, prompt: dict) -> GenerationResult:
        """Run a prompt and return the images it saved.

        Args:
            prompt: The API-format prompt to run.

        Returns:
            The decoded image bytes and their filenames, in order.
        """
        response = await _post_json(f"{_backend_url()}/prompt", body={"prompt": prompt})
        if not response.is_success:
            raise _submit_rejection(response)
        return await asyncio.to_thread(_parse_generation, response)
