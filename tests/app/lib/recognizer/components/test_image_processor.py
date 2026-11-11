"""Tests for cutting and joining captcha pictures."""

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.lib.recognizer.components.image_processor import (
    create_image_grid,
    handle_multiple_images,
    handle_single_image,
)

_RED_TILE = (255, 0, 0)
_GREEN_TILE = (0, 255, 0)
_BLUE_TILE = (0, 0, 255)
_YELLOW_TILE = (255, 255, 0)
_TILE_SIZE = 100
_JPEG_HEADER = b"\xff\xd8"


def _make_solid_image(shape: tuple[int, ...], color: tuple[int, ...]) -> np.ndarray:
    """Make a picture of one color."""
    return np.full(shape, color, dtype=np.uint8)


def _encode_jpeg(img: np.ndarray) -> bytes:
    """Turn a picture into JPEG data."""
    return cv2.imencode(".jpg", img)[1].tobytes()


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Turn JPEG data back into a picture."""
    return cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)


def _make_captcha_image(squares: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Make a black picture with white squares on it."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    for (x, y, w, h) in squares:
        img[y:y + h, x:x + w] = 255
    return img


@pytest.mark.parametrize(
    "area_captcha,expected_tiles",
    [(True, 16), (False, 9)],
    ids=["area_4x4", "non_area_3x3"],
)
def test_handle_single_image_splits_large_image(area_captcha: bool, expected_tiles: int) -> None:
    """A big picture is cut into 9 tiles, or 16 for an area captcha."""
    img = _make_solid_image((400, 400, 3), _RED_TILE)
    images, _coords = handle_single_image(_encode_jpeg(img), area_captcha)
    assert len(images) == expected_tiles
    assert all(jpeg_bytes[:2] == _JPEG_HEADER for jpeg_bytes in images)


def test_handle_single_image_tiles_preserve_color() -> None:
    """Every tile keeps the color of the picture."""
    img = _make_solid_image((400, 400, 3), _RED_TILE)
    images, _coords = handle_single_image(_encode_jpeg(img), False)
    assert all(
        _decode_jpeg(tile_bytes)[50, 50][0] >= 250
        for tile_bytes in images
    )


def test_create_image_grid_dimensions() -> None:
    """Four tiles join into a picture twice as wide and twice as tall."""
    tiles = [
        _make_solid_image((_TILE_SIZE, _TILE_SIZE, 3), color)
        for color in [_RED_TILE, _GREEN_TILE, _BLUE_TILE, _YELLOW_TILE]
    ]
    grid = create_image_grid(tiles)
    assert grid.shape[0] == _TILE_SIZE * 2
    assert grid.shape[1] == _TILE_SIZE * 2


def test_create_image_grid_preserves_tile_colors() -> None:
    """Each corner of the joined picture keeps its own color."""
    colors = [_RED_TILE, _GREEN_TILE, _BLUE_TILE, _YELLOW_TILE]
    tiles = [
        _make_solid_image((_TILE_SIZE, _TILE_SIZE, 3), color)
        for color in colors
    ]
    grid = create_image_grid(tiles)
    mid = _TILE_SIZE // 2
    assert tuple(grid[mid, mid]) == _RED_TILE
    assert tuple(grid[mid, _TILE_SIZE + mid]) == _GREEN_TILE
    assert tuple(grid[_TILE_SIZE + mid, mid]) == _BLUE_TILE
    assert tuple(grid[_TILE_SIZE + mid, _TILE_SIZE + mid]) == _YELLOW_TILE


@pytest.mark.parametrize("tile_count", [4, 9])
def test_create_image_grid_layout(tile_count: int) -> None:
    """Tiles join into a square picture."""
    tiles = [
        _make_solid_image((_TILE_SIZE, _TILE_SIZE, 3), (i, i, i))
        for i in range(tile_count)
    ]
    grid = create_image_grid(tiles)
    expected_dim = _TILE_SIZE * int(np.sqrt(tile_count))
    assert grid.shape[0] == expected_dim
    assert grid.shape[1] == expected_dim


def test_handle_single_image_returns_fields_for_multi_field_image() -> None:
    """A picture with separate shapes gives a piece and a spot for each."""
    squares = [(50, 50, 100, 100), (200, 50, 100, 100)]
    img = _make_captcha_image(squares)
    images, coords = handle_single_image(_encode_jpeg(img), False)
    assert isinstance(images, list)
    assert isinstance(coords, list)
    assert len(images) == len(coords)
    assert len(images) >= 2
    assert all(jpeg_bytes[:2] == _JPEG_HEADER for jpeg_bytes in images)


def test_handle_single_image_fields_sorted_by_position() -> None:
    """Pieces come back in order down the picture and then across."""
    squares = [(300, 100, 80, 80), (50, 50, 80, 80), (300, 300, 80, 80), (50, 300, 80, 80)]
    img = _make_captcha_image(squares)
    images, coords = handle_single_image(_encode_jpeg(img), False)
    assert len(images) >= 4
    assert all(
        (coords[i][1] == coords[i + 1][1] and coords[i][0] <= coords[i + 1][0])
        or (coords[i][1] < coords[i + 1][1])
        for i in range(len(coords) - 1)
    )


def test_handle_multiple_images_returns_correct_count() -> None:
    """Two pictures in give two pictures out."""
    img1 = _make_solid_image((100, 100, 3), _RED_TILE)
    img2 = _make_solid_image((100, 100, 3), _GREEN_TILE)
    raw_bytes = [_encode_jpeg(img1), _encode_jpeg(img2)]
    result = handle_multiple_images(raw_bytes)
    assert len(result) == 2


def test_handle_multiple_images_decodes_correctly() -> None:
    """Decoded pictures keep their size."""
    img1 = _make_solid_image((50, 60, 3), _RED_TILE)
    img2 = _make_solid_image((80, 100, 3), _BLUE_TILE)
    raw_bytes = [_encode_jpeg(img1), _encode_jpeg(img2)]
    result = handle_multiple_images(raw_bytes)
    assert result[0].shape == (50, 60, 3)
    assert result[1].shape == (80, 100, 3)


def test_handle_multiple_images_with_base64() -> None:
    """Picture data written as base64 is read back."""
    img = _make_solid_image((50, 50, 3), _RED_TILE)
    jpeg_bytes = _encode_jpeg(img)
    b64_bytes = base64.b64encode(jpeg_bytes)
    result = handle_multiple_images([b64_bytes])
    assert len(result) == 1
    assert result[0].shape == (50, 50, 3)


def test_handle_single_image_from_path(tmp_path: Path) -> None:
    """A picture can come from a file path."""
    img = _make_solid_image((50, 50, 3), _RED_TILE)
    img_path = tmp_path / "captcha.jpg"
    cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    images, coords = handle_single_image(img_path, False)
    assert isinstance(images, list)
    assert isinstance(coords, list)
    assert len(images) == len(coords)


def test_handle_single_image_from_raw_bytes() -> None:
    """A picture can come from raw JPEG data."""
    img = _make_solid_image((50, 50, 3), _GREEN_TILE)
    jpeg_bytes = _encode_jpeg(img)
    images, coords = handle_single_image(jpeg_bytes, False)
    assert isinstance(images, list)
    assert isinstance(coords, list)
    assert len(images) == len(coords)


def test_handle_single_image_from_base64() -> None:
    """A picture can come from base64 data."""
    img = _make_solid_image((50, 50, 3), _BLUE_TILE)
    jpeg_bytes = _encode_jpeg(img)
    b64_bytes = base64.b64encode(jpeg_bytes)
    images, coords = handle_single_image(b64_bytes, False)
    assert isinstance(images, list)
    assert isinstance(coords, list)
    assert len(images) == len(coords)


@pytest.mark.parametrize("area_captcha", [True, False])
def test_handle_single_image_returns_valid_output_large(area_captcha: bool, tmp_path: Path) -> None:
    """A big picture gives back pieces and a spot for each."""
    squares = [(10, 10, 100, 100), (150, 10, 100, 100), (10, 150, 100, 100), (150, 150, 100, 100)]
    captcha_img = _make_captcha_image(squares)
    img_path = tmp_path / "captcha_large.jpg"
    cv2.imwrite(str(img_path), cv2.cvtColor(captcha_img, cv2.COLOR_RGB2BGR))
    images, coords = handle_single_image(img_path, area_captcha)
    assert len(images) >= 1
    assert len(coords) == len(images)
    assert all(jpeg_bytes[0:2] == _JPEG_HEADER for jpeg_bytes in images)


def test_handle_single_image_returns_valid_output_small(tmp_path: Path) -> None:
    """A small picture with shapes gives back pieces and a spot for each."""
    squares = [(10, 10, 50, 50), (100, 10, 50, 50)]
    captcha_img = _make_captcha_image(squares)
    img_path = tmp_path / "captcha_small.jpg"
    cv2.imwrite(str(img_path), cv2.cvtColor(captcha_img, cv2.COLOR_RGB2BGR))
    images, coords = handle_single_image(img_path, False)
    assert len(images) >= 1
    assert len(coords) == len(images)
    assert all(jpeg_bytes[0:2] == _JPEG_HEADER for jpeg_bytes in images)