"""Tests for app.lib.comfyui.media."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.lib.comfyui.client import GenerationResult
from app.lib.comfyui import media
from app.lib.shared.hashing import hash_file_content
from tests.app.conftest import DAV_BASE_URL, point_resources_at, write_reference_png


class TestRandomSeed:
    """The seed helper returns a value across the full 64-bit range."""

    def test_seed_is_an_int_in_range(self) -> None:
        seed = media.random_seed()

        assert isinstance(seed, int)
        assert 0 <= seed <= 2 ** 64 - 1


class TestReferenceDownloadUrl:
    """The resource path becomes the public download URL the backend fetches."""

    async def test_existing_reference_resolves_to_the_download_url(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        (tmp_path / "upload").mkdir()
        (tmp_path / "upload" / "ref.png").write_bytes(b"image")

        url = await media.reference_download_url("upload/ref.png")

        assert url == f"{DAV_BASE_URL}/dav/upload/ref.png"

    @pytest.mark.parametrize("missing", ["upload/missing.png", "upload", "../escape.png"])
    async def test_missing_reference_raises_the_upload_instruction(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
        point_resources_at(tmp_path, monkeypatch)
        (tmp_path / "upload").mkdir()

        with pytest.raises(ValueError, match=f"Resource not found: {missing}"):
            await media.reference_download_url(missing)


class TestSaveFirstFile:
    """The first returned file lands under media_gen with the tool prefix."""

    async def test_saves_with_the_tool_prefix_and_backend_extension(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        result = GenerationResult(images=(b"png-bytes",), filenames=("media_gen_t2i/out_00001_.png",))

        rel = await media.save_first_file(result, "generate_image", "png")

        assert rel.startswith("media_gen/generate_image_")
        assert rel.endswith(".png")
        path = tmp_path / rel
        assert path.read_bytes() == b"png-bytes"

    async def test_falls_back_to_the_mode_extension(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        result = GenerationResult(images=(b"video-bytes",), filenames=("no-extension",))

        rel = await media.save_first_file(result, "generate_video", "mp4")

        assert rel.endswith(".mp4")


class TestReferenceFingerprints:
    """Cache keys fingerprint the reference contents, not just the path."""

    async def test_hashes_each_reference(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)
        (tmp_path / "upload").mkdir()
        (tmp_path / "upload" / "a.png").write_bytes(b"aaa")
        (tmp_path / "upload" / "b.png").write_bytes(b"bbb")

        fingerprints = await media.reference_fingerprints(["upload/a.png", "upload/b.png"])

        assert fingerprints == [hash_file_content(tmp_path / "upload" / "a.png"), hash_file_content(tmp_path / "upload" / "b.png")]

    async def test_absent_reference_keeps_its_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        point_resources_at(tmp_path, monkeypatch)

        fingerprints = await media.reference_fingerprints(["upload/missing.png"])

        assert fingerprints == ["upload/missing.png"]


# The default box fit_to_box shrinks into, and the step its sides round to.
_FIT_BOX = 2048
_FIT_STEP = 32
_CUSTOM_BOX = 1024

# Oversized test photos as (width, height) tuples.
_PORTRAIT = (3024, 4032)
_PORTRAIT_2 = (3072, 4080)
_LANDSCAPE = (4032, 3024)
_LANDSCAPE_2 = (4080, 3072)
_SQUARE = (4000, 4000)
_ODD = (3073, 4081)
_NEAR = (2049, 1000)

# Their fitted results, the long side landing on the box edge.
_FIT_SHORT = 1536
_PORTRAIT_FIT = (_FIT_SHORT, _FIT_BOX)
_LANDSCAPE_FIT = (_FIT_BOX, _FIT_SHORT)
_SQUARE_FIT = (_FIT_BOX, _FIT_BOX)
_NEAR_FIT = (_FIT_BOX, 992)

# Images already inside the box, which pass through unchanged.
_AT_BOX = (_FIT_BOX, _FIT_SHORT)
_SMALL = (1000, 1500)
_TINY = (256, 256)


class TestFitToBox:
    """Oversized images are downscaled into the box, smaller ones pass through."""

    @pytest.mark.parametrize(
        ("size", "expected"),
        [
            (_PORTRAIT, _PORTRAIT_FIT),
            (_PORTRAIT_2, _PORTRAIT_FIT),
            (_LANDSCAPE, _LANDSCAPE_FIT),
            (_LANDSCAPE_2, _LANDSCAPE_FIT),
            (_SQUARE, _SQUARE_FIT),
            (_ODD, _PORTRAIT_FIT),
            (_NEAR, _NEAR_FIT),
        ],
    )
    def test_oversized_image_fits_the_box(self, size: tuple[int, int], expected: tuple[int, int]) -> None:
        width, height = size
        assert media.fit_to_box(width, height) == expected

    @pytest.mark.parametrize("size", [_AT_BOX, _SMALL, _TINY, (_FIT_STEP, _FIT_STEP)])
    def test_small_image_passes_through_unchanged(self, size: tuple[int, int]) -> None:
        width, height = size
        assert media.fit_to_box(width, height) == (width, height)

    @pytest.mark.parametrize("size", [_PORTRAIT, _PORTRAIT_2, _ODD, _NEAR])
    def test_fitted_sides_are_multiples_of_the_step(self, size: tuple[int, int]) -> None:
        fitted_width, fitted_height = media.fit_to_box(size[0], size[1])

        assert fitted_width % _FIT_STEP == 0
        assert fitted_height % _FIT_STEP == 0

    @pytest.mark.parametrize("size", [_PORTRAIT, _PORTRAIT_2, _SQUARE, _NEAR])
    def test_long_side_never_exceeds_the_box(self, size: tuple[int, int]) -> None:
        fitted_width, fitted_height = media.fit_to_box(size[0], size[1])

        assert max(fitted_width, fitted_height) <= _FIT_BOX

    def test_custom_box_and_step(self) -> None:
        assert media.fit_to_box(3000, 3000, max_edge=_CUSTOM_BOX, multiple=16) == (_CUSTOM_BOX, _CUSTOM_BOX)
        assert media.fit_to_box(100, 200, max_edge=64, multiple=16) == (32, 64)


class TestImageDimensions:
    """The pixel size of a resource image, and the same errors as the download URL."""

    @pytest.mark.parametrize("size", [_PORTRAIT, _SMALL, _SQUARE_FIT])
    async def test_returns_the_pixel_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]
    ) -> None:
        point_resources_at(tmp_path, monkeypatch)
        reference = write_reference_png(tmp_path, "photo.png", size[0], size[1])

        assert await media.image_dimensions(reference) == size

    @pytest.mark.parametrize("missing", ["upload/missing.png", "upload", "../escape.png"])
    async def test_missing_reference_raises_the_upload_instruction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
    ) -> None:
        point_resources_at(tmp_path, monkeypatch)
        (tmp_path / "upload").mkdir()

        with pytest.raises(ValueError, match=f"Resource not found: {missing}"):
            await media.image_dimensions(missing)

    async def test_file_that_is_not_an_image_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        point_resources_at(tmp_path, monkeypatch)
        (tmp_path / "upload").mkdir()
        (tmp_path / "upload" / "photo.png").write_bytes(b"not an image")

        with pytest.raises(ValueError, match="not an image"):
            await media.image_dimensions("upload/photo.png")
