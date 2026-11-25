"""Tests for finding the right captcha tiles."""

from pathlib import Path

import pytest

from app.lib.recognizer.components.detector import DetectionResult, Detector


_IMAGE_DIR = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    ("image_file", "prompt", "is_split_image", "expected_matches", "expected_length"),
    [
        ("full_page.png", "bicycle", False, [True, True, False, False, True, False, False, False, False], 9),
        ("only_captcha.png", "motorcycle", True, [True, True, True, False, True, True, True, False, False, True, True, False, False, False, False, False], 16),
        ("area_image.png", "fire hydrant", True, [False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False], 16),
        ("area_no_yolo.png", "chimney", True, [True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False], 16),
        ("classify_image.png", "bicycle", False, [False, False, False, True, False, False, True, True, False], 9),
        ("classify_no_yolo.png", "stairs", False, None, 9),
    ],
)
def test_detect_returns_valid_result(
    detector: Detector,
    image_file: str,
    prompt: str,
    is_split_image: bool,
    expected_matches: list[bool] | None,
    expected_length: int,
) -> None:
    """Known captcha pictures give the expected tiles."""
    image_bytes = (_IMAGE_DIR / image_file).read_bytes()
    result = detector.detect(prompt, image_bytes, is_split_image=is_split_image)

    assert isinstance(result, DetectionResult)
    assert len(result.tile_matches) == expected_length
    assert sum(result.tile_matches) == len(result.click_coordinates)
    assert all(isinstance(point, tuple) and len(point) == 2 for point in result.click_coordinates)

    if expected_matches is not None:
        assert result.tile_matches == expected_matches


def test_detect_clip_captcha_has_positive_results(detector: Detector) -> None:
    """A stairs picture gives at least one picked tile."""
    image_bytes = (_IMAGE_DIR / "classify_no_yolo.png").read_bytes()
    result = detector.detect("stairs", image_bytes, is_split_image=False)

    assert len(result.tile_matches) == 9
    assert sum(result.tile_matches) > 0
    assert sum(result.tile_matches) == len(result.click_coordinates)


def test_detect_unsupported_label_returns_empty(detector: Detector) -> None:
    """A label the detector does not know gives nothing back."""
    image_bytes = (_IMAGE_DIR / "full_page.png").read_bytes()
    result = detector.detect("unsupported_label", image_bytes, is_split_image=False)

    assert result == DetectionResult([], [])


@pytest.mark.parametrize(
    ("prompt", "expected_length"),
    [
        ("Select all images with bicycles", 9),
        ("Select all images with a car", 9),
        ("Select all images with cars", 9),
        ("bicycle", 9),
        ("  car  ", 9),
    ],
)
def test_detect_with_sentence_prompts(detector: Detector, prompt: str, expected_length: int) -> None:
    """A whole sentence prompt works like a plain label."""
    image_bytes = (_IMAGE_DIR / "full_page.png").read_bytes()
    result = detector.detect(prompt, image_bytes, is_split_image=False)

    assert len(result.tile_matches) == expected_length
    assert sum(result.tile_matches) == len(result.click_coordinates)


def test_detect_cyrillic_lookalike_prompt(detector: Detector) -> None:
    """A label with lookalike letters gives the same answer as the plain one."""
    image_bytes = (_IMAGE_DIR / "full_page.png").read_bytes()
    cyrillic_result = detector.detect("bicyсle", image_bytes, is_split_image=False)
    latin_result = detector.detect("bicycle", image_bytes, is_split_image=False)

    assert cyrillic_result.tile_matches == latin_result.tile_matches


def test_detect_split_image_auto_detect_none(detector: Detector) -> None:
    """A prompt about squares works with the split flag left unset."""
    image_bytes = (_IMAGE_DIR / "only_captcha.png").read_bytes()
    result = detector.detect("square with motorcycles", image_bytes, is_split_image=None)

    assert len(result.tile_matches) == 16
    assert sum(result.tile_matches) == len(result.click_coordinates)


def test_detect_single_pathlib_input(detector: Detector) -> None:
    """A file path is accepted as one picture."""
    image_path = _IMAGE_DIR / "full_page.png"
    result = detector.detect("bicycle", image_path, is_split_image=False)

    assert len(result.tile_matches) == 9
    assert sum(result.tile_matches) == len(result.click_coordinates)


def test_single_str_path_input(detector: Detector) -> None:
    """A path written as text is accepted as one picture."""
    image_path = str(_IMAGE_DIR / "full_page.png")
    result = detector.detect("bicycle", image_path, is_split_image=False)

    assert len(result.tile_matches) == 9
    assert sum(result.tile_matches) == len(result.click_coordinates)


def test_single_bytes_input(detector: Detector) -> None:
    """Raw picture data is accepted as one picture."""
    image_bytes = (_IMAGE_DIR / "full_page.png").read_bytes()
    result = detector.detect("bicycle", image_bytes, is_split_image=False)

    assert len(result.tile_matches) == 9
    assert sum(result.tile_matches) == len(result.click_coordinates)


def test_single_pathlib_in_list(detector: Detector) -> None:
    """One file path in a list is cut into tiles."""
    image_path = _IMAGE_DIR / "full_page.png"
    result = detector.detect("bicycle", [image_path], is_split_image=False)

    assert len(result.tile_matches) == 9
    assert sum(result.tile_matches) == len(result.click_coordinates)


def test_single_str_path_in_list(detector: Detector) -> None:
    """One path written as text in a list is cut into tiles."""
    image_paths = [str(_IMAGE_DIR / "full_page.png")]
    result = detector.detect("bicycle", image_paths, is_split_image=False)

    assert len(result.tile_matches) == 9
    assert sum(result.tile_matches) == len(result.click_coordinates)


def test_single_bytes_in_list(detector: Detector) -> None:
    """One piece of picture data in a list is cut into tiles."""
    image_bytes_list = [(_IMAGE_DIR / "full_page.png").read_bytes()]
    result = detector.detect("bicycle", image_bytes_list, is_split_image=False)

    assert len(result.tile_matches) == 9
    assert sum(result.tile_matches) == len(result.click_coordinates)


def test_pathlib_list_input(detector: Detector) -> None:
    """A list of file paths is taken as ready made tiles."""
    split_images = sorted((_IMAGE_DIR / "splitted").iterdir())
    result = detector.detect("bicycle", split_images, is_split_image=False)

    assert len(result.tile_matches) == 9
    assert sum(result.tile_matches) == len(result.click_coordinates)


def test_str_paths_list_input(detector: Detector) -> None:
    """A list of paths written as text is taken as ready made tiles."""
    split_images = sorted((_IMAGE_DIR / "splitted").iterdir())
    paths = [str(path) for path in split_images]
    result = detector.detect("bicycle", paths, is_split_image=False)

    assert len(result.tile_matches) == 9
    assert sum(result.tile_matches) == len(result.click_coordinates)


def test_bytes_list_input(detector: Detector) -> None:
    """A list of picture data is taken as ready made tiles."""
    split_images = sorted((_IMAGE_DIR / "splitted").iterdir())
    tile_bytes = [image.read_bytes() for image in split_images]
    result = detector.detect("bicycle", tile_bytes, is_split_image=False)

    assert len(result.tile_matches) == 9
    assert sum(result.tile_matches) == len(result.click_coordinates)


def test_invalid_tile_count_returns_empty(detector: Detector) -> None:
    """A tile count that is not 9 or 16 gives nothing back."""
    tile_bytes = [(_IMAGE_DIR / "full_page.png").read_bytes() for _ in range(5)]
    result = detector.detect("bicycle", tile_bytes, is_split_image=False)

    assert result == DetectionResult([], [])


def test_detect_empty_image_list_returns_empty(detector: Detector) -> None:
    """An empty list of pictures gives nothing back."""
    result = detector.detect("bicycle", [], is_split_image=False)

    assert result == DetectionResult([], [])


async def test_detect_async_returns_valid_result(detector: Detector) -> None:
    """The async entry point gives the expected tiles."""
    image_bytes = (_IMAGE_DIR / "full_page.png").read_bytes()
    result = await detector.detect_async("bicycle", image_bytes, is_split_image=False)

    assert isinstance(result, DetectionResult)
    assert result.tile_matches == [True, True, False, False, True, False, False, False, False]
    assert sum(result.tile_matches) == len(result.click_coordinates)
    assert all(isinstance(point, tuple) and len(point) == 2 for point in result.click_coordinates)
