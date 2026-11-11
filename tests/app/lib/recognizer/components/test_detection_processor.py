"""Tests for turning detections into tile answers."""

import numpy as np
import pytest

from app.lib.recognizer.components.detection_processor import (
    calculate_approximated_coords,
    calculate_segmentation_response,
    find_lowest_distance,
    get_tiles_in_bounding_box,
)

_SEGMENTATION_MASK_CASES = [
    (
        [(5, 5)],
        10,
        10,
        2,
        [True, False, False, False],
    ),
    (
        [(5, 5), (15, 15)],
        10,
        10,
        2,
        [True, False, False, True],
    ),
    (
        [(5, 5), (15, 5), (5, 15), (15, 15)],
        10,
        10,
        2,
        [True, True, True, True],
    ),
]


@pytest.mark.parametrize(
    "mask,tile_w,tile_h,tiles_per_row,expected",
    _SEGMENTATION_MASK_CASES,
    ids=["single_tile", "two_tiles", "all_tiles"],
)
def test_calculate_segmentation_response_mask_only(
    mask: list[tuple[int, int]], tile_w: int, tile_h: int, tiles_per_row: int, expected: list[bool]
) -> None:
    """Points land in the right tiles."""
    response = [False] * (tiles_per_row * tiles_per_row)
    result = calculate_segmentation_response(
        mask, response, tile_w, tile_h, tiles_per_row
    )
    assert result == expected


def test_calculate_segmentation_response_with_threshold() -> None:
    """Tiles fully inside a filled shape are marked."""
    threshold = np.ones((21, 21), dtype=int)
    response = [False, False, False, False]
    result = calculate_segmentation_response(
        [], response, 10, 10, 2, threshold_image=threshold
    )
    assert result == [True, True, True, True]


def test_calculate_segmentation_response_threshold_zero_corner() -> None:
    """A tile with an empty corner is not marked."""
    threshold = np.ones((21, 21), dtype=int)
    threshold[0, 0] = 0
    response = [False, False, False, False]
    result = calculate_segmentation_response(
        [], response, 10, 10, 2, threshold_image=threshold
    )
    assert result[0] is False
    assert result[1] is True
    assert result[2] is True
    assert result[3] is True


def test_calculate_segmentation_response_clamped_coords() -> None:
    """A point past the edge lands in the last tile."""
    mask = [(999, 999)]
    response = [False, False, False, False]
    result = calculate_segmentation_response(
        mask, response, 10, 10, 2
    )
    assert result[3] is True


_BOUNDING_BOX_CASES = [
    (
        (30, 30, 3),
        9,
        (0, 0),
        (30, 30),
        [True, True, True, True, True, True, True, True, True],
    ),
    (
        (30, 30, 3),
        9,
        (0, 0),
        (10, 10),
        [True, False, False, False, False, False, False, False, False],
    ),
    (
        (30, 30, 3),
        9,
        (5, 5),
        (9, 9),
        [False, False, False, False, False, False, False, False, False],
    ),
]


@pytest.mark.parametrize(
    "img_shape,tile_amount,start,end,expected",
    _BOUNDING_BOX_CASES,
    ids=["full_coverage", "single_tile", "partial_overlap"],
)
def test_get_tiles_in_bounding_box(img_shape: tuple[int, int, int], tile_amount: int, start: tuple[int, int], end: tuple[int, int], expected: list[bool]) -> None:
    """Tiles inside the box are marked."""
    h, w, c = img_shape
    img = np.zeros((h, w, c), dtype=np.uint8)
    result = get_tiles_in_bounding_box(img, tile_amount, start, end)
    assert result == expected


_APPROX_COORDS_CASES = [
    (10, 10, 2, [(5, 5), (15, 5), (5, 15), (15, 15)]),
    (100, 100, 1, [(50, 50)]),
    (20, 30, 2, [(10, 15), (30, 15), (10, 45), (30, 45)]),
]


@pytest.mark.parametrize(
    "gw,gh,tile_amount,expected",
    _APPROX_COORDS_CASES,
    ids=["grid_2x2", "grid_1x1", "asymmetric_grid"],
)
def test_calculate_approximated_coords(gw: int, gh: int, tile_amount: int, expected: list[tuple[int, int]]) -> None:
    """Each tile gets the point at its middle."""
    result = calculate_approximated_coords(gw, gh, tile_amount)
    assert result == expected


def test_find_lowest_distance_basic() -> None:
    """The path starts where you are and covers every point once."""
    result = find_lowest_distance((0, 0), [(10, 0), (5, 0)])
    assert len(result) == 3
    assert result[0] == (0, 0)
    assert set(result[1:]) == {(10, 0), (5, 0)}


def test_find_lowest_distance_single_point() -> None:
    """One point gives a path of two steps."""
    result = find_lowest_distance((0, 0), [(10, 0)])
    assert result == [(0, 0), (10, 0)]


def test_find_lowest_distance_collinear_order() -> None:
    """Points on a line are visited in order."""
    result = find_lowest_distance((0, 0), [(3, 0), (1, 0), (2, 0)])
    assert result[0] == (0, 0)
    xs = [p[0] for p in result]
    assert xs == sorted(xs)