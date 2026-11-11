"""Measures how alike two screenshots look."""

from __future__ import annotations

import asyncio
import io

import numpy as np
from PIL import Image

__all__ = ["screen_similarity"]

_SIMILARITY_SIZE = 256
_SSIM_WINDOW = 8
_PIXEL_MAX = 255
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03
_SSIM_C1 = (_SSIM_K1 * _PIXEL_MAX) ** 2
_SSIM_C2 = (_SSIM_K2 * _PIXEL_MAX) ** 2
_SSIM_CROSS_SCALE = 2
_MIN_SIMILARITY = 0.0
_MAX_SIMILARITY = 1.0


def _window_pixels(data: bytes) -> np.ndarray:
    """Cut a screenshot into small square windows of gray pixels.

    Args:
        data: The bytes of a screenshot image.

    Returns:
        One row per window, each row holding that window's pixels.
    """
    image = Image.open(io.BytesIO(data)).convert("L").resize((_SIMILARITY_SIZE, _SIMILARITY_SIZE))
    pixels = np.asarray(image, dtype=np.float64)
    count = _SIMILARITY_SIZE // _SSIM_WINDOW
    grid = pixels.reshape(count, _SSIM_WINDOW, count, _SSIM_WINDOW).swapaxes(1, 2)
    return grid.reshape(count * count, _SSIM_WINDOW * _SSIM_WINDOW)


def _screen_similarity_sync(old_bytes: bytes, new_bytes: bytes) -> float:
    """Measure how alike two screenshots look, as a number from 0 to 1.

    Args:
        old_bytes: The bytes of the earlier screenshot.
        new_bytes: The bytes of the newer screenshot.

    Returns:
        A number from 0 to 1, where 1 means the same and 0 means nothing in
        common.
    """
    if not old_bytes or not new_bytes:
        return _MAX_SIMILARITY
    try:
        old = _window_pixels(old_bytes)
        new = _window_pixels(new_bytes)
    except Exception:
        return _MAX_SIMILARITY
    old_mean = old.mean(axis=1, keepdims=True)
    new_mean = new.mean(axis=1, keepdims=True)
    old_spread = old.var(axis=1, keepdims=True)
    new_spread = new.var(axis=1, keepdims=True)
    together = ((old - old_mean) * (new - new_mean)).mean(axis=1, keepdims=True)
    top = (_SSIM_CROSS_SCALE * old_mean * new_mean + _SSIM_C1) * (_SSIM_CROSS_SCALE * together + _SSIM_C2)
    bottom = (np.square(old_mean) + np.square(new_mean) + _SSIM_C1) * (old_spread + new_spread + _SSIM_C2)
    return min(max(float(np.mean(top / bottom)), _MIN_SIMILARITY), _MAX_SIMILARITY)


async def screen_similarity(old_bytes: bytes, new_bytes: bytes) -> float:
    """Measure how alike two screenshots look, as a number from 0 to 1.

    A screenshot that is missing or cannot be read counts as alike.

    Args:
        old_bytes: The bytes of the earlier screenshot.
        new_bytes: The bytes of the newer screenshot.

    Returns:
        A number from 0 to 1, where 1 means the same and 0 means nothing in
        common.
    """
    return await asyncio.to_thread(_screen_similarity_sync, old_bytes, new_bytes)
