"""Tests for finding the right captcha tiles."""
from pathlib import Path

import pytest

from app.lib.recognizer.components.detector import Detector


_IMAGE_DIR = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    ("image_file", "prompt", "area_captcha", "expected_response", "expected_length"),
    [
        ("full_page.png", "bicycle", False, [True, True, False, False, True, False, False, False, False], 9),
        ("only_captcha.png", "motorcycle", True, [True, True, True, False, True, True, True, False, False, True, True, False, False, False, False, False], 16),
        ("area_image.png", "fire hydrant", True, [False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False], 16),
        ("area_no_yolo.png", "chimney", True, [True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False], 16),
        ("classify_image.png", "bicycle", False, [False, False, False, True, False, False, True, True, False], 9),
        ("classify_no_yolo.png", "stairs", False, None, 9),
    ],
)
def test_detect_returns_valid_response(detector: Detector, image_file: str, prompt: str, area_captcha: bool, expected_response: list[bool] | None, expected_length: int) -> None:
    """Known captcha pictures give the expected tiles."""
    img_bytes = (_IMAGE_DIR / image_file).read_bytes()
    response, coordinates = detector.detect(prompt, img_bytes, area_captcha=area_captcha)

    assert len(response) == expected_length
    assert response is not coordinates
    assert sum(response) == len(coordinates)
    assert all(isinstance(c, tuple) and len(c) == 2 for c in coordinates)

    if expected_response is not None:
        assert response == expected_response


def test_detect_clip_captcha_has_positive_results(detector: Detector) -> None:
    """A stairs picture gives at least one picked tile."""
    img_bytes = (_IMAGE_DIR / "classify_no_yolo.png").read_bytes()
    response, coordinates = detector.detect("stairs", img_bytes, area_captcha=False)

    assert len(response) == 9
    assert sum(response) > 0
    assert sum(response) == len(coordinates)


def test_detect_unsupported_label_returns_empty(detector: Detector) -> None:
    """A label the detector does not know gives nothing back."""
    img_bytes = (_IMAGE_DIR / "full_page.png").read_bytes()
    response, coordinates = detector.detect("unsupported_label", img_bytes, area_captcha=False)

    assert response == []
    assert coordinates == []


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
    img_bytes = (_IMAGE_DIR / "full_page.png").read_bytes()
    response, coordinates = detector.detect(prompt, img_bytes, area_captcha=False)

    assert len(response) == expected_length
    assert sum(response) == len(coordinates)


def test_detect_cyrillic_lookalike_prompt(detector: Detector) -> None:
    """A label with lookalike letters gives the same answer as the plain one."""
    img_bytes = (_IMAGE_DIR / "full_page.png").read_bytes()
    response_cyrillic, _ = detector.detect("bicyсle", img_bytes, area_captcha=False)
    response_latin, _ = detector.detect("bicycle", img_bytes, area_captcha=False)

    assert response_cyrillic == response_latin


def test_detect_area_captcha_auto_detect_none(detector: Detector) -> None:
    """A prompt about squares works with the area flag left unset."""
    img_bytes = (_IMAGE_DIR / "only_captcha.png").read_bytes()
    response, coordinates = detector.detect("square with motorcycles", img_bytes, area_captcha=None)

    assert len(response) in (9, 16)
    assert sum(response) == len(coordinates)


def test_detect_single_pathlib_input(detector: Detector) -> None:
    """A file path is accepted as one picture."""
    img_path = _IMAGE_DIR / "full_page.png"
    response, coordinates = detector.detect("bicycle", img_path, area_captcha=False)

    assert len(response) == 9
    assert sum(response) == len(coordinates)


def test_single_str_path_input(detector: Detector) -> None:
    """A path written as text is accepted as one picture."""
    img_path = str(_IMAGE_DIR / "full_page.png")
    response, coordinates = detector.detect("bicycle", img_path, area_captcha=False)

    assert len(response) == 9
    assert sum(response) == len(coordinates)


def test_single_bytes_input(detector: Detector) -> None:
    """Raw picture data is accepted as one picture."""
    img_bytes = (_IMAGE_DIR / "full_page.png").read_bytes()
    response, coordinates = detector.detect("bicycle", img_bytes, area_captcha=False)

    assert len(response) == 9
    assert sum(response) == len(coordinates)


def test_single_pathlib_in_list(detector: Detector) -> None:
    """One file path in a list is cut into tiles."""
    img_path = _IMAGE_DIR / "full_page.png"
    response, coordinates = detector.detect("bicycle", [img_path], area_captcha=False)

    assert len(response) == 9
    assert sum(response) == len(coordinates)


def test_single_str_path_in_list(detector: Detector) -> None:
    """One path written as text in a list is cut into tiles."""
    img_path = [str(_IMAGE_DIR / "full_page.png")]
    response, coordinates = detector.detect("bicycle", img_path, area_captcha=False)

    assert len(response) == 9
    assert sum(response) == len(coordinates)


def test_single_bytes_in_list(detector: Detector) -> None:
    """One piece of picture data in a list is cut into tiles."""
    img_bytes = [(_IMAGE_DIR / "full_page.png").read_bytes()]
    response, coordinates = detector.detect("bicycle", img_bytes, area_captcha=False)

    assert len(response) == 9
    assert sum(response) == len(coordinates)


def test_pathlib_list_input(detector: Detector) -> None:
    """A list of file paths is taken as ready made tiles."""
    splitted_images = sorted((_IMAGE_DIR / "splitted").iterdir())
    response, coordinates = detector.detect("bicycle", splitted_images, area_captcha=False)

    assert len(response) == 9
    assert sum(response) == len(coordinates)


def test_str_paths_list_input(detector: Detector) -> None:
    """A list of paths written as text is taken as ready made tiles."""
    splitted_images = sorted((_IMAGE_DIR / "splitted").iterdir())
    paths = [str(p) for p in splitted_images]
    response, coordinates = detector.detect("bicycle", paths, area_captcha=False)

    assert len(response) == 9
    assert sum(response) == len(coordinates)


def test_bytes_list_input(detector: Detector) -> None:
    """A list of picture data is taken as ready made tiles."""
    splitted_images = sorted((_IMAGE_DIR / "splitted").iterdir())
    byte_images = [img.read_bytes() for img in splitted_images]
    response, coordinates = detector.detect("bicycle", byte_images, area_captcha=False)

    assert len(response) == 9
    assert sum(response) == len(coordinates)


def test_invalid_tile_count_returns_empty(detector: Detector) -> None:
    """A tile count that is not 9 or 16 gives nothing back."""
    tile_bytes = [(_IMAGE_DIR / "full_page.png").read_bytes() for _ in range(5)]
    response, coordinates = detector.detect("bicycle", tile_bytes, area_captcha=False)

    assert response == []
    assert coordinates == []


def test_detect_empty_image_list_returns_empty(detector: Detector) -> None:
    """An empty list of pictures gives nothing back."""
    response, coordinates = detector.detect("bicycle", [], area_captcha=False)

    assert response == []
    assert coordinates == []


async def test_detect_async_returns_valid_response(detector: Detector) -> None:
    """The async entry point gives the expected tiles."""
    img_bytes = (_IMAGE_DIR / "full_page.png").read_bytes()
    response, coordinates = await detector.detect_async("bicycle", img_bytes, area_captcha=False)

    assert response == [True, True, False, False, True, False, False, False, False]
    assert sum(response) == len(coordinates)
    assert all(isinstance(c, tuple) and len(c) == 2 for c in coordinates)
