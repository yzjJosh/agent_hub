__all__ = [
    "calculate_approximated_coords",
    "calculate_segmentation_response",
    "find_lowest_distance",
    "get_tiles_in_bounding_box",
]

import math
from itertools import permutations
from typing import List, Tuple, Union

from numpy import generic
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


def _mask_points_to_tiles(mask: Union[NDArray[generic], List[Tuple[int, int]]], response: List[bool], tile_width: int, tile_height: int, tiles_per_row: int) -> None:
    for coord in mask:
        mask_point_x, mask_point_y = tuple(coord)

        col = int(mask_point_x // tile_width)
        row = int(mask_point_y // tile_height)

        col = min(col, tiles_per_row - 1)
        row = min(row, tiles_per_row - 1)

        tile_index = row * tiles_per_row + col
        if response[tile_index]:
            continue

        boundary_x = tile_width * 0.08
        boundary_y = tile_height * 0.08

        # fmt: off
        within_boundary = (
                boundary_x <= coord[1] % tile_width <= tile_width - boundary_x
                and boundary_y <= coord[0] % tile_height <= tile_height - boundary_y
        )
        # fmt: on

        if within_boundary:
            response[tile_index] = True


def _enclosed_tiles_check(response: List[bool], tile_width: int, tile_height: int, tiles_per_row: int, threshold_image: NDArray[generic]) -> None:
    for row in range(tiles_per_row):
        for col in range(tiles_per_row):
            tile_index = row * tiles_per_row + col
            if response[tile_index]:
                continue

            tile_x_min = col * tile_width
            tile_x_max = tile_x_min + tile_width
            tile_y_min = row * tile_height
            tile_y_max = tile_y_min + tile_height

            corners = [
                (tile_x_min, tile_y_min),
                (tile_x_max, tile_y_min),
                (tile_x_min, tile_y_max),
                (tile_x_max, tile_y_max),
            ]

            def is_tile_fully_enclosed(
                    tile_corners: List[Tuple[int, int]],
                    threshold_image: NDArray[generic],
                ) -> bool:
                    """Say if all four corners of the tile sit on marked ground."""
                    for corner in tile_corners:
                        x, y = corner
                        if not (0 <= x < threshold_image.shape[1] and 0 <= y < threshold_image.shape[0]):
                            return False
                        if threshold_image[y, x] == 0:
                            return False
                    return True

            response[tile_index] = is_tile_fully_enclosed(corners, threshold_image)


def calculate_segmentation_response(
    mask: Union[NDArray[generic], List[Tuple[int, int]]],
    response: List[bool],
    tile_width: int,
    tile_height: int,
    tiles_per_row: int,
    threshold_image: Union[NDArray[generic], None] = None,
) -> List[bool]:
    """Mark every tile the shape covers.

    Args:
        mask: The points along the edge of the shape.
        response: Which tiles are marked so far.
        tile_width: How wide one tile is.
        tile_height: How tall one tile is.
        tiles_per_row: How many tiles sit in one row.
        threshold_image: The picture of the shape. Tiles that sit fully inside
            the shape are marked too when it is given.

    Returns:
        Which tiles are marked.
    """
    _mask_points_to_tiles(mask, response, tile_width, tile_height, tiles_per_row)

    if threshold_image is None:
        return response

    _enclosed_tiles_check(response, tile_width, tile_height, tiles_per_row, threshold_image)
    return response


def get_tiles_in_bounding_box(
    img: NDArray[generic],
    tile_amount: int,
    point_start: Tuple[int, int],
    point_end: Tuple[int, int],
) -> List[bool]:
    """Say which tiles sit fully inside a box.

    Args:
        img: The whole grid picture.
        tile_amount: How many tiles the grid holds.
        point_start: The top left corner of the box.
        point_end: The bottom right corner of the box.

    Returns:
        One flag per tile, left to right and top to bottom.
    """
    tiles_in_bbox = []
    height, width, _ = img.shape
    tiles_per_row = int(math.sqrt(tile_amount))

    tile_width = width // tiles_per_row
    tile_height = height // tiles_per_row

    for i in range(tiles_per_row):
        for j in range(tiles_per_row):
            tile_x1 = j * tile_height
            tile_y1 = i * tile_width
            tile_x2 = (j + 1) * tile_height
            tile_y2 = (i + 1) * tile_width

            tile_area = (tile_x2 - tile_x1) * (tile_y2 - tile_y1)

            intersection_x1 = max(tile_x1, point_start[0])
            intersection_x2 = min(tile_x2, point_end[0])
            intersection_y1 = max(tile_y1, point_start[1])
            intersection_y2 = min(tile_y2, point_end[1])

            if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                if (intersection_area / tile_area) == 1:
                    tiles_in_bbox.append(True)
                else:
                    tiles_in_bbox.append(False)
            else:
                tiles_in_bbox.append(False)

    return tiles_in_bbox


def calculate_approximated_coords(grid_width: int, grid_height: int, tile_amount: int) -> List[Tuple[int, int]]:
    """Work out the middle point of every tile in a grid.

    Args:
        grid_width: How wide one tile is.
        grid_height: How tall one tile is.
        tile_amount: How many tiles sit in one row.

    Returns:
        The middle points, left to right and top to bottom.
    """
    middle_points = []

    for y in range(tile_amount):
        for x in range(tile_amount):
            middle_x = (x * grid_width) + (grid_width // 2)
            middle_y = (y * grid_height) + (grid_height // 2)
            middle_points.append((middle_x, middle_y))

    return middle_points


def find_lowest_distance(start: Tuple[int, int], coordinates: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Put the points in the order that makes the shortest trip.

    Args:
        start: The point to begin at.
        coordinates: The other points to visit.

    Returns:
        The start point, then the rest in the order to visit them.
    """
    points = [start] + coordinates
    distances = cdist(points, points, metric="euclidean")

    num_points = len(points)
    min_distance = float("inf")
    best_path = None

    for perm in permutations(range(1, num_points)):
        path = [0] + list(perm)
        distance = sum(distances[path[i], path[i + 1]] for i in range(len(path) - 1))

        if distance < min_distance:
            min_distance = distance
            best_path = path

    assert best_path
    best_coordinates = [points[i] for i in best_path]
    return best_coordinates