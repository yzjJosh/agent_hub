"""Shared plumbing for the ComfyUI media tools.

Resolves resource references to the public download URLs the backend fetches,
saves a finished generation under the resources folder, and fingerprints
reference files for cache keys.
"""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Sequence
from datetime import datetime

from PIL import Image, UnidentifiedImageError

from app.lib.comfyui.client import GenerationResult
from app.lib.shared.hashing import hash_file_content
from app.lib.shared.messages import missing_reference_message
from app.lib.shared.paths import resolve_path, resources_dir
from app.resources.resources import dav_download_url

__all__ = [
    "random_seed",
    "reference_download_url",
    "save_first_file",
    "reference_fingerprints",
    "fit_to_box",
    "image_dimensions",
]

_OUTPUT_DIR = "media_gen"
_MAX_SEED = 2 ** 64 - 1
_FIT_MAX_EDGE = 2048
_FIT_MULTIPLE = 32


def random_seed() -> int:
    """A random seed for a generation, across the full 64-bit range."""
    return random.randint(0, _MAX_SEED)


async def _reference_target(reference: str) -> str | None:
    """The local path of a reference file in the resource, or None when it is not a file."""
    target = await resolve_path(resources_dir(), reference)
    if target is None or not await asyncio.to_thread(os.path.isfile, target):
        return None
    return target


async def reference_download_url(reference: str) -> str:
    """The public download URL of a reference image in the resource.

    Args:
        reference: Resource path of the reference image.

    Returns:
        The public WebDAV download URL the backend fetches.

    Raises:
        ValueError: When the reference is not a file in the resource.
    """
    if await _reference_target(reference) is None:
        raise ValueError(missing_reference_message(reference))
    return dav_download_url(reference)


def _extension_for(result: GenerationResult, fallback: str) -> str:
    """The file ending to save under, from the first name or the fallback."""
    if result.filenames:
        extension = os.path.splitext(result.filenames[0])[1].lstrip(".")
        if extension:
            return extension
    return fallback


def _write_file(path: str, data: bytes) -> None:
    """Write the bytes to the path, replacing any file already there."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp = path + ".part"
    with open(temp, "wb") as fh:
        fh.write(data)
    os.replace(temp, path)


async def save_first_file(result: GenerationResult, tool_name: str, fallback_extension: str) -> str:
    """Save the first file of a finished generation under the resources folder.

    Args:
        result: The finished generation.
        tool_name: The tool that produced the file, for the filename prefix.
        fallback_extension: The ending when the backend name has none.

    Returns:
        The resource path of the saved file.
    """
    extension = _extension_for(result, fallback_extension)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        resources_dir(), _OUTPUT_DIR, f"{tool_name}_{timestamp}.{extension}"
    )
    await asyncio.to_thread(_write_file, output_path, result.images[0])
    return os.path.relpath(output_path, resources_dir())


async def reference_fingerprints(references: Sequence[str]) -> list[str]:
    """The content fingerprint of each reference, or its path when absent.

    Args:
        references: Resource paths of the reference images.

    Returns:
        One sha256 fingerprint or raw path per reference, in order.
    """
    fingerprints: list[str] = []
    for reference in references:
        target = await _reference_target(reference)
        if target is not None:
            fingerprints.append(await asyncio.to_thread(hash_file_content, target))
        else:
            fingerprints.append(reference)
    return fingerprints


def _round_down(value: int, multiple: int) -> int:
    """Round a value down to the nearest multiple."""
    return value // multiple * multiple


def fit_to_box(width: int, height: int, max_edge: int = _FIT_MAX_EDGE, multiple: int = _FIT_MULTIPLE) -> tuple[int, int]:
    """Shrink an image so its long side fits a box, keeping the aspect ratio.

    Args:
        width: Current pixel width.
        height: Current pixel height.
        max_edge: Longest allowed side.
        multiple: Side step both outputs are rounded down to.

    Returns:
        The fitted width and height.
    """
    longest = max(width, height)
    if longest <= max_edge:
        return width, height
    return (
        _round_down(width * max_edge // longest, multiple),
        _round_down(height * max_edge // longest, multiple),
    )


def _image_dimensions_sync(path: str) -> tuple[int, int]:
    """The pixel size of a local image file."""
    with Image.open(path) as image:
        return image.size


async def image_dimensions(reference: str) -> tuple[int, int]:
    """The pixel width and height of an image in the resource.

    Args:
        reference: Resource path of the image.

    Returns:
        The pixel width and height.

    Raises:
        ValueError: When the reference is missing or not an image.
    """
    target = await _reference_target(reference)
    if target is None:
        raise ValueError(missing_reference_message(reference))
    try:
        return await asyncio.to_thread(_image_dimensions_sync, target)
    except UnidentifiedImageError:
        raise ValueError(f"not an image: {reference}") from None
