from __future__ import annotations

__all__ = [
    "create_image_grid",
    "handle_multiple_images",
    "handle_single_image",
]

import base64
import binascii
import math
from contextlib import suppress
from pathlib import Path
from statistics import median
from typing import List, Tuple, Union

import cv2
from imageio.v2 import imread
from numpy import concatenate

from .detection_processor import calculate_approximated_coords


def _find_contour_fields(img: cv2.typing.MatLike, gray_image: cv2.typing.MatLike) -> List[Tuple[bytes, int, int, int]]:
    captcha_fields_with_sizes: List[Tuple[bytes, int, int, int]] = []
    _, thresh = cv2.threshold(gray_image, 254, 255, cv2.CHAIN_APPROX_NONE)

    try:
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # type: ignore
    except ValueError:
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # type: ignore

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

            x, y, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w) / h

            if 0.95 <= aspectRatio <= 1.05:
                crop_img = img[y:y+h, x:x+w]  # fmt: skip
                image_bytes = cv2.imencode(".jpg", crop_img)[1].tobytes()
                image_size: int = w * h
                center_x, center_y = x + (w // 2), y + (h // 2)
                captcha_fields_with_sizes.append((image_bytes, center_x, center_y, image_size))

    return captcha_fields_with_sizes


def _filter_captcha_fields(captcha_fields_with_sizes: List[Tuple[bytes, int, int, int]]) -> List[Tuple[bytes, int, int]]:
    captcha_fields: List[Tuple[bytes, int, int]] = []

    if len(captcha_fields_with_sizes) >= 9:
        size_median = median([sizes[3] for sizes in captcha_fields_with_sizes])
        for image_bytes, center_x, center_y, image_size in captcha_fields_with_sizes:
            if int(image_size) == int(size_median):
                captcha_fields.append((image_bytes, center_x, center_y))
    else:
        for image_bytes, center_x, center_y, image_size in captcha_fields_with_sizes:
            captcha_fields.append((image_bytes, center_x, center_y))

    return captcha_fields


def _get_captcha_fields(
    img: cv2.typing.MatLike,
) -> Tuple[List[bytes], List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    """Cut the captcha tiles out of a screenshot.

    Returns:
        The tile pictures, the middle point of each tile, and the box each
        tile sits in.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    captcha_fields_with_sizes = _find_contour_fields(img, gray)
    captcha_fields = _filter_captcha_fields(captcha_fields_with_sizes)

    sorted_captcha_fields: List[Tuple[bytes, int, int]] = sorted(captcha_fields, key=lambda element: [element[2], element[1]])
    bboxes: List[Tuple[int, int, int, int]] = []
    for image_bytes, center_x, center_y in sorted_captcha_fields:
        crop_img_bytes = base64.b64decode(image_bytes) if isinstance(image_bytes, str) else image_bytes
        crop_rgba = imread(crop_img_bytes)
        crop_h, crop_w = crop_rgba.shape[0], crop_rgba.shape[1]
        x = center_x - crop_w // 2
        y = center_y - crop_h // 2
        bboxes.append((x, y, crop_w, crop_h))

    return (
        [field[0] for field in sorted_captcha_fields],
        [(field[1], field[2]) for field in sorted_captcha_fields],
        bboxes,
    )


def _split_image_into_tiles(img: cv2.typing.MatLike, tile_count: int) -> List[bytes]:
    """Cut a picture into a grid of equal tiles.

    Args:
        img: The picture to cut.
        tile_count: How many tiles to make in one row.

    Returns:
        The tiles, left to right and top to bottom.
    """
    tiles = []

    height, width, _ = img.shape

    tile_width = width // tile_count
    tile_height = height // tile_count

    for i in range(tile_count):
        for j in range(tile_count):
            x_start = j * tile_width
            x_end = (j + 1) * tile_width
            y_start = i * tile_height
            y_end = (i + 1) * tile_height

            tile = img[y_start:y_end, x_start:x_end]
            image_bytes = cv2.imencode(".jpg", tile)[1].tobytes()
            tiles.append(image_bytes)

    return tiles


def create_image_grid(images: List[cv2.typing.MatLike]) -> cv2.typing.MatLike:
    """Join the pictures into one square grid."""
    cv2_images = images
    tile_count_per_row = int(math.sqrt(len(cv2_images)))

    layers = []
    for i in range(tile_count_per_row):
        layer_images = [cv2_images[i * tile_count_per_row + j] for j in range(tile_count_per_row)]
        layer = concatenate(layer_images, axis=1)
        layers.append(layer)

    combined_img = concatenate(layers, axis=0)

    return combined_img


def handle_single_image(single_image: Union[Path, bytes], area_captcha: bool) -> Tuple[List[bytes], List[Tuple[int, int]]]:
    """Get the tiles out of one captcha picture.

    Args:
        single_image: The captcha picture.
        area_captcha: True when the tiles are parts of one big picture.

    Returns:
        The tiles, and where to click for each one.
    """
    if isinstance(single_image, bytes):
        with suppress(binascii.Error):
            single_image = base64.b64decode(single_image, validate=True)

    rgba_img = imread(single_image)
    img = cv2.cvtColor(rgba_img, cv2.COLOR_BGR2RGB)

    images, coords, bboxes = _get_captcha_fields(img)

    if len(images) == 1:
        rgba_img = imread(images[0])
        img = cv2.cvtColor(rgba_img, cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape

        if height > 200 and width > 200:
            tiles_amount = 4 if area_captcha else 3
            images = _split_image_into_tiles(img, tiles_amount)
            bx, by, bw, bh = bboxes[0]
            tile_w = bw // tiles_amount
            tile_h = bh // tiles_amount
            coords = calculate_approximated_coords(tile_w, tile_h, tiles_amount)
            coords = [(cx + bx, cy + by) for cx, cy in coords]

    return images, coords


def handle_multiple_images(images: List[bytes]) -> List[cv2.typing.MatLike]:
    """Turn picture bytes into pictures."""
    cv2_images = []
    for image in images:
        try:
            byte_image = base64.b64decode(image, validate=True)
        except binascii.Error:
            byte_image = image

        rgba_img = imread(byte_image)
        cv2_img = cv2.cvtColor(rgba_img, cv2.COLOR_BGR2RGB)
        cv2_images.append(cv2_img)

    return cv2_images