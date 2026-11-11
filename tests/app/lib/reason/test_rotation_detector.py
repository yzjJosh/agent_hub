"""Tests for the rotation detector module."""

import importlib
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.lib.reason import rotation_detector
from PIL import Image, ImageDraw


@pytest.fixture
def temp_image_dir(tmp_path: Path) -> str:
    """Temporary directory for test images."""
    d = tmp_path / "images"
    d.mkdir()
    return str(d)


@pytest.fixture
def asymmetric_image(temp_image_dir: str) -> str:
    """Create a tall image with a different mark near each end."""
    img = Image.new("RGB", (100, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 30, 30], fill="red")
    draw.rectangle([70, 170, 90, 190], fill="blue")
    path = os.path.join(temp_image_dir, "asymmetric.jpg")
    img.save(path)
    return path


@pytest.fixture
def real_image_path(tmp_path: Path, data_dir: str) -> str:
    """Copy real image to temp directory."""
    src = os.path.join(data_dir, "test_image_real.jpg")
    dst = tmp_path / "test_image_real.jpg"
    shutil.copy(src, dst)
    return str(dst)


def _mock_session_for_class(cls_idx: int, high_confidence: bool = True) -> MagicMock:
    """Build a stand in model that picks the given class."""
    logits = [0.0, 0.0, 0.0, 0.0]
    logits[cls_idx] = 5.0 if high_confidence else 0.1
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.run.return_value = [[np.array(logits)]]
    return mock_session


async def _process_and_assert_unchanged(image_path: str, expected_size: tuple[int, int]) -> None:
    """Turn the image upright and check that it did not change."""
    original_size = os.path.getsize(image_path)
    await rotation_detector.rotate_images_in_place([image_path])
    assert os.path.getsize(image_path) == original_size
    with Image.open(image_path) as img:
        assert img.size == expected_size


@pytest.fixture(autouse=True)
def reset_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test with no model loaded and leave none behind."""
    importlib.reload(rotation_detector)
    monkeypatch.setattr(rotation_detector, "_SESSION", None)


# --- Error handling ---


async def test_rotate_images_in_place_empty_list() -> None:
    """An empty list downloads nothing and loads no model, even with no model on disk."""
    with (
        patch("os.path.exists", return_value=False),
        patch.object(rotation_detector.httpx, "AsyncClient") as mock_client,
        patch.object(rotation_detector.ort, "InferenceSession") as mock_session,
    ):
        await rotation_detector.rotate_images_in_place([])

    mock_client.assert_not_called()
    mock_session.assert_not_called()


async def test_rotate_images_in_place_non_existent_file() -> None:
    """A missing file does not crash."""
    await rotation_detector.rotate_images_in_place(["non_existent.jpg"])


async def test_rotate_images_in_place_empty_path() -> None:
    """An empty path does not crash."""
    await rotation_detector.rotate_images_in_place([""])


async def test_rotate_images_in_place_none_input() -> None:
    """A path of None does not crash."""
    await rotation_detector.rotate_images_in_place([None])  # type: ignore[list-item]


async def test_rotate_images_in_place_corrupted_image(temp_image_dir: str) -> None:
    """A broken image file is left alone."""
    path = os.path.join(temp_image_dir, "corrupted.jpg")
    with open(path, "wb") as f:
        f.write(b"not an image")
    original_size = os.path.getsize(path)
    await rotation_detector.rotate_images_in_place([path])
    assert os.path.getsize(path) == original_size


# --- Upright images are not changed ---


async def test_rotate_images_in_place_no_rotation(asymmetric_image: str) -> None:
    """An image the model calls upright is left alone."""
    mock = _mock_session_for_class(0)
    with patch.object(rotation_detector.ort, "InferenceSession", return_value=mock):
        await _process_and_assert_unchanged(asymmetric_image, (100, 200))


# --- Turned images are put back upright ---


async def test_rotate_images_in_place_rotates_image_270(asymmetric_image: str) -> None:
    """An image called a quarter turn clockwise comes back with width and height swapped."""
    mock = _mock_session_for_class(1)
    with patch.object(rotation_detector.ort, "InferenceSession", return_value=mock):
        await rotation_detector.rotate_images_in_place([asymmetric_image])
        with Image.open(asymmetric_image) as img:
            assert img.size == (200, 100)


async def test_rotate_images_in_place_rotates_image_180(asymmetric_image: str) -> None:
    """An image called a half turn keeps its size and its pixels move."""
    mock = _mock_session_for_class(2)
    with patch.object(rotation_detector.ort, "InferenceSession", return_value=mock):
        with Image.open(asymmetric_image) as img:
            original_data = np.array(img)
        await rotation_detector.rotate_images_in_place([asymmetric_image])
        with Image.open(asymmetric_image) as img:
            assert img.size == (100, 200)
            assert not np.array_equal(original_data, np.array(img))


async def test_rotate_images_in_place_rotates_image_90(asymmetric_image: str) -> None:
    """An image called a quarter turn counter clockwise comes back with width and height swapped."""
    mock = _mock_session_for_class(3)
    with patch.object(rotation_detector.ort, "InferenceSession", return_value=mock):
        await rotation_detector.rotate_images_in_place([asymmetric_image])
        with Image.open(asymmetric_image) as img:
            assert img.size == (200, 100)


# --- An unsure guess leaves the image alone ---


async def test_rotate_images_in_place_low_confidence_fallback(asymmetric_image: str) -> None:
    """An image the model is unsure about is left alone."""
    mock = _mock_session_for_class(2, high_confidence=False)
    with patch.object(rotation_detector.ort, "InferenceSession", return_value=mock):
        await _process_and_assert_unchanged(asymmetric_image, (100, 200))


# --- A failed save does not crash ---


async def test_rotate_images_in_place_handles_save_error(asymmetric_image: str) -> None:
    """A save that fails does not crash."""
    mock = _mock_session_for_class(1)
    with patch.object(rotation_detector.ort, "InferenceSession", return_value=mock):
        with patch.object(Image.Image, "save", side_effect=OSError("disk full")):
            await rotation_detector.rotate_images_in_place([asymmetric_image])


# --- Live Integration Tests ---


@pytest.mark.live
async def test_live_rotate_real_image(real_image_path: str) -> None:
    """A real image can still be opened after it is turned upright."""
    await rotation_detector.rotate_images_in_place([real_image_path])
    assert os.path.exists(real_image_path)
    with Image.open(real_image_path) as img:
        assert img.size[0] > 0 and img.size[1] > 0


@pytest.mark.live
@pytest.mark.parametrize("angle", [0, 90, 180, 270])
async def test_live_rotate_all_orientations(
    real_image_path: str, tmp_path: Path, angle: int
) -> None:
    """An image turned any of the four ways is still there afterward."""
    with Image.open(real_image_path) as img:
        rot_path = tmp_path / f"rot_{angle}.jpg"
        if angle == 0:
            img.save(rot_path)
        else:
            img.rotate(-angle, expand=True).save(rot_path)

    await rotation_detector.rotate_images_in_place([str(rot_path)])
    assert rot_path.exists()


@pytest.mark.live
async def test_live_rotate_multiple_images(real_image_path: str, tmp_path: Path) -> None:
    """Every image in one call is still there afterward."""
    img = Image.new("RGB", (100, 100), color="green")
    extra_images = [str(tmp_path / f"img_{i}.jpg") for i in range(3)]
    _ = [img.save(p) for p in extra_images]
    images = extra_images + [real_image_path]

    await rotation_detector.rotate_images_in_place(images)
    assert all(Path(p).exists() for p in images)