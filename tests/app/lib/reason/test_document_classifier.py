"""Tests for the document classifier public API."""

import os
import pytest
from PIL import Image, ImageDraw
from pathlib import Path

from app.lib.reason.document_classifier import is_document


@pytest.fixture
def temp_image_dir(tmp_path: Path) -> str:
    """Temporary directory for test images."""
    d = tmp_path / "images"
    d.mkdir()
    return str(d)


def _create_image(path: str, color: str = "white", mode: str = "RGB", corrupted: bool = False) -> str:
    """Create a test image at the given path."""
    if corrupted:
        with open(path, "wb") as f:
            f.write(b"not an image")
        return path

    Image.new(mode, (100, 100), color=color).save(path)
    return path


def _create_colorful_image(path: str) -> str:
    """Create an image with shapes in several colors."""
    img = Image.new("RGB", (150, 150), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 60, 60], fill="red")
    draw.ellipse([80, 10, 140, 70], fill="blue")
    draw.polygon([(40, 80), (10, 140), (70, 140)], fill="green")
    img.save(path)
    return path


@pytest.fixture
def corrupted_image(temp_image_dir: str) -> str:
    """Corrupted test image file."""
    return _create_image(os.path.join(temp_image_dir, "corrupted.jpg"), corrupted=True)


@pytest.fixture
def non_existent_image(temp_image_dir: str) -> str:
    """Path to a non-existent image file."""
    return os.path.join(temp_image_dir, "non_existent.jpg")


# --- Error Handling Tests ---


async def test_is_document_empty_path() -> None:
    """is_document returns False for an empty path."""
    assert await is_document("") is False


async def test_is_document_nonexistent_file(non_existent_image: str) -> None:
    """is_document returns False when the file does not exist."""
    assert await is_document(non_existent_image) is False


async def test_is_document_corrupted_image(corrupted_image: str) -> None:
    """is_document returns False for a corrupted image file."""
    assert await is_document(corrupted_image) is False


# --- Live Synthetic Image Tests ---


@pytest.mark.live
async def test_live_is_document_jpg(temp_image_dir: str) -> None:
    """A plain JPG image counts as a document."""
    img_path = _create_image(os.path.join(temp_image_dir, "test.jpg"))
    assert await is_document(img_path) is True


@pytest.mark.live
async def test_live_is_document_png(temp_image_dir: str) -> None:
    """A plain PNG image counts as a document."""
    img_path = _create_image(os.path.join(temp_image_dir, "test.png"))
    assert await is_document(img_path) is True


@pytest.mark.live
async def test_live_is_document_grayscale(temp_image_dir: str) -> None:
    """A gray image counts as a document."""
    img_path = _create_image(os.path.join(temp_image_dir, "gray.jpg"), mode="L")
    assert await is_document(img_path) is True


@pytest.mark.live
async def test_live_is_document_rgba(temp_image_dir: str) -> None:
    """An image with a see through layer counts as a document."""
    img_path = _create_image(os.path.join(temp_image_dir, "rgba.png"), mode="RGBA")
    assert await is_document(img_path) is True


@pytest.mark.live
async def test_live_is_document_colorful(temp_image_dir: str) -> None:
    """An image full of colored shapes does not count as a document."""
    img_path = _create_colorful_image(os.path.join(temp_image_dir, "colorful.jpg"))
    assert await is_document(img_path) is False


# --- Real Image Tests ---


_REAL_IMAGE_EXPECTED: list[tuple[str, bool]] = [
    ("bar_chart.png", True),
    ("line_chart.png", True),
    ("pie_chart.png", False),
    ("table_image.png", True),
    ("screenshot_computer.png", True),
    ("document_page.jpg", True),
    ("photo_landscape.jpg", False),
    ("flow_chart.png", True),
    ("qr_code_like.png", True),
]


@pytest.mark.parametrize("filename,expected", _REAL_IMAGE_EXPECTED)
@pytest.mark.live
async def test_live_real_images(data_dir: str, filename: str, expected: bool) -> None:
    """Each real test image gets the answer listed for it."""
    image_path = os.path.join(data_dir, filename)
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found: {image_path}")
    assert await is_document(image_path) is expected