"""Tests for app.lib.reason.describe_image.py module."""

import os
import json
import shutil
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from PIL import Image
from app.lib.reason.describe_image import describe_image
from tests.app.conftest import is_llm_available
from tests.app.lib.conftest import mock_cli_proc
from typing import Callable, Self
from pathlib import Path


# Image width and height used in the stand in metadata.
_MOCK_IMAGE_WIDTH = 1920
_MOCK_IMAGE_HEIGHT = 1080
_MOCK_MEDIUM_WIDTH = 1000

# GPS spot for San Jose, CA.
_SAN_JOSE_LAT = 37.3382
_SAN_JOSE_LON = -121.8863


def _create_test_images(tmp_path: Path, count: int) -> list[str]:
    """Create test images and return their paths as strings."""
    test_images = [str(tmp_path / f"test_{i}.jpg") for i in range(count)]
    for img_path in test_images:
        Image.new('RGB', (100, 100), color='red').save(img_path)
    return test_images


async def _run_describe_with_mock(
    image_path: str,
    reason_return: tuple[str, int] = ("desc", 0),
    is_document_return: bool = False,
    cli_proc: MagicMock | None = None
) -> tuple[str, int]:

    """Run describe_image with the usual stand ins in place.

    Args:
        image_path: Path to the image file.
        reason_return: Description and status the model gives back.
        is_document_return: Whether the image counts as a document.
        cli_proc: Stand in child process that reads metadata, or None.

    Returns:
        The result and status from describe_image.
    """
    if cli_proc is not None:
        with patch('asyncio.create_subprocess_exec', return_value=cli_proc):
            with patch('app.lib.reason.describe_image.is_document', return_value=is_document_return):
                with patch('app.lib.reason.describe_image.reason', return_value=reason_return):
                    return await describe_image(str(image_path), "")
    with patch('app.lib.reason.describe_image.is_document', return_value=is_document_return):
        with patch('app.lib.reason.describe_image.reason', return_value=reason_return):
            return await describe_image(str(image_path), "")


def _mock_exiftool_failure() -> MagicMock:
    """Build a stand in metadata reader that fails."""
    return mock_cli_proc(returncode=1)


class TestDescribeImageStatusCodes:
    """Test the status codes describe_image gives back."""

    async def test_returns_status_0_on_empty_input(self: Self) -> None:
        """An empty list of images gives status 0 and says there is nothing to do."""
        result, status = await describe_image([], "")
        assert status == 0
        assert result == "No images to process."

    async def test_returns_status_2_on_preprocessing_failure(self: Self) -> None:
        """An image that cannot be read gives status 2."""
        result, status = await describe_image(["/nonexistent.jpg"], "")
        assert status == 2
        assert "Failed" in result

    async def test_returns_status_1_on_llm_failure(self, tmp_path: Path) -> None:
        """A model failure gives status 1."""
        test_image = _create_test_images(tmp_path, 1)[0]
        result, status = await _run_describe_with_mock(test_image, reason_return=("", 1))
        assert status == 1
        assert "Failed" in result

    @patch('app.lib.reason.describe_image.is_document')
    @patch('app.lib.reason.describe_image.reason')
    async def test_returns_status_0_on_success(self, mock_reason: MagicMock, mock_is_document: MagicMock, tmp_path: Path) -> None:
        """A good image gives status 0 and the description."""
        test_image = _create_test_images(tmp_path, 1)[0]
        mock_is_document.return_value = False
        mock_reason.return_value = ("Test description", 0)
        result, status = await describe_image([test_image], "")
        assert status == 0
        assert "Test description" in result

    @patch('app.lib.reason.describe_image.is_document')
    @patch('app.lib.reason.describe_image.reason')
    async def test_llm_failure_aborts_subsequent_batches(self, mock_reason: MagicMock, mock_is_document: MagicMock, tmp_path: Path) -> None:
        """A model failure stops the rest of the images from being sent."""
        test_images = _create_test_images(tmp_path, 6)
        mock_is_document.return_value = False
        mock_reason.return_value = ("", 1)
        result, status = await describe_image(test_images, "")
        assert status == 1
        assert "Failed" in result
        assert mock_reason.call_count == 1


class TestDescribeImageWithExif:
    """Test describe_image with and without EXIF data."""

    async def test_image_without_exif_succeeds(self, tmp_path: Path) -> None:
        """An image with no EXIF data is still described."""
        test_image = tmp_path / "no_exif.jpg"
        img = Image.new('RGB', (100, 50), color='blue')
        img.save(test_image)

        result, status = await _run_describe_with_mock(test_image)
        assert status == 0
        assert "desc" in result

    async def test_non_image_file_returns_preprocessing_error(self, tmp_path: Path) -> None:
        """A file that is not an image gives status 2."""
        test_file = tmp_path / "not_an_image.txt"
        test_file.write_text("This is not an image")

        result, status = await describe_image(str(test_file), "")
        assert status == 2
        assert "Failed" in result

    async def test_complete_exif_data_included_in_output(self, tmp_path: Path) -> None:
        """Camera and place details from EXIF data show up in the output."""
        test_image = tmp_path / "test.jpg"
        img = Image.new('RGB', (100, 100), color='navy')
        img.save(test_image)

        exif_data = [{
            "SourceFile": str(test_image),
            "ImageWidth": 100,
            "ImageHeight": 100,
            "Make": "Apple",
            "Model": "iPhone 15 Pro",
            "DateTimeOriginal": "2024:03:15 14:30:00",
            "GPSLatitude": _SAN_JOSE_LAT,
            "GPSLatitudeRef": "N",
            "GPSLongitude": _SAN_JOSE_LON,
            "GPSLongitudeRef": "W",
            "ExposureTime": "1/120",
            "FNumber": 1.8,
            "ISO": 100,
            "FocalLength": 6.0,
        }]
        mock_result = mock_cli_proc(stdout=json.dumps(exif_data))

        result, status = await _run_describe_with_mock(test_image, cli_proc=mock_result)
        assert status == 0
        assert "Apple" in result
        assert "iPhone 15 Pro" in result
        assert "ISO: 100" in result
        assert "US" in result


class TestDescribeImageWithGps:
    """Test describe_image with GPS data."""

    def _create_exif_data(
        self,
        lat: float | int | None,
        lon: float | int | None,
        lat_ref: str = 'N',
        lon_ref: str = 'E',
        make: str | None = None,
        model: str | None = None,
        include_gps: bool = True
    ) -> list[dict]:

        """Build stand in image metadata with a GPS spot."""
        exif_data = {
            "SourceFile": "/tmp/test_image.jpg",
            "ImageWidth": _MOCK_IMAGE_WIDTH,
            "ImageHeight": _MOCK_IMAGE_HEIGHT,
        }
        if make:
            exif_data["Make"] = make
        if model:
            exif_data["Model"] = model
        if include_gps:
            exif_data["GPSLatitude"] = lat
            exif_data["GPSLatitudeRef"] = lat_ref
            exif_data["GPSLongitude"] = lon
            exif_data["GPSLongitudeRef"] = lon_ref
        return [exif_data]

    def _mock_exiftool(self, exif_data: list[dict]) -> MagicMock:
        """Build a stand in metadata reader with the given output."""
        return mock_cli_proc(stdout=json.dumps(exif_data))

    def _create_test_image(self, tmp_path: Path, name: str, size: tuple = (100, 100), color: str = 'green') -> Path:
        """Create a test image and return its path."""
        test_image = tmp_path / name
        Image.new('RGB', size, color=color).save(test_image)
        return test_image

    async def _run_gps_test(self, tmp_path: Path, lat: float | int | None, lon: float | int | None, lat_ref: str, lon_ref: str) -> tuple[str, int]:
        """Describe an image that carries the given GPS spot."""
        test_image = self._create_test_image(tmp_path, "test.jpg")
        mock_result = self._mock_exiftool(
            self._create_exif_data(lat, lon, lat_ref, lon_ref))
        return await _run_describe_with_mock(test_image, cli_proc=mock_result)

    @pytest.mark.parametrize('lat,lon,lat_ref,lon_ref,country_code,city_contains,state_contains', [
        pytest.param(_SAN_JOSE_LAT, _SAN_JOSE_LON, 'N', 'W', 'US', 'San Jose', None, id="san_jose_ca"),
        pytest.param(40.7128, -74.0060, 'N', 'W', 'US', None, 'New York', id="new_york_ny"),
        pytest.param(34.0522, -118.2437, 'N', 'W', 'US', None, 'California', id="los_angeles_ca"),
        pytest.param(31.2304, 121.4737, 'N', 'E', 'CN', 'Shanghai', None, id="shanghai"),
        pytest.param(39.9042, 116.4074, 'N', 'E', 'CN', 'Beijing', None, id="beijing"),
        pytest.param(22.5431, 114.0579, 'N', 'E', 'CN', 'Shenzhen', None, id="shenzhen"),
    ])
    async def test_reverse_geocoding_us_china(self, tmp_path: Path, lat: float | int | None, lon: float | int | None, lat_ref: str, lon_ref: str, country_code: str, city_contains: str | None, state_contains: str | None) -> None:
        """A GPS spot turns into the country, city, or state in the output."""
        result, status = await self._run_gps_test(tmp_path, lat, lon, lat_ref, lon_ref)
        assert status == 0
        assert country_code in result
        if city_contains:
            assert city_contains in result
        if state_contains:
            assert state_contains in result

    async def test_missing_gps_no_crash(self, tmp_path: Path) -> None:
        """An image with no GPS spot still shows its camera details."""
        test_image = self._create_test_image(tmp_path, "test.jpg")
        mock_result = self._mock_exiftool(
            self._create_exif_data(None, None, include_gps=False,
                                   make='Apple', model='iPhone'))

        result, status = await _run_describe_with_mock(test_image, cli_proc=mock_result)
        assert status == 0
        assert "Apple" in result
        assert "iPhone" in result

    async def test_gps_at_zero_zero_ocean(self, tmp_path: Path) -> None:
        """A GPS spot at zero, zero still gives status 0."""
        test_image = self._create_test_image(tmp_path, "test.jpg")
        mock_result = self._mock_exiftool(self._create_exif_data(0.0, 0.0, 'N', 'E'))

        with patch('asyncio.create_subprocess_exec', return_value=mock_result):
            with patch('app.lib.reason.describe_image.is_document', return_value=False):
                with patch('app.lib.reason.describe_image.reason', return_value=("desc", 0)):
                    _, status = await describe_image(str(test_image), "")

        assert status == 0

    async def test_invalid_coordinates_handled_gracefully(self, tmp_path: Path) -> None:
        """A GPS spot outside the real range still gives a description."""
        test_image = self._create_test_image(tmp_path, "test.jpg")
        exif_data = [{
            "SourceFile": str(test_image),
            "ImageWidth": 100,
            "ImageHeight": 100,
            "GPSLatitude": 999,
            "GPSLatitudeRef": 'N',
            "GPSLongitude": 0,
            "GPSLongitudeRef": 'E',
        }]
        mock_result = self._mock_exiftool(exif_data)

        result, status = await _run_describe_with_mock(test_image, cli_proc=mock_result)
        assert status == 0
        assert "desc" in result

    @pytest.mark.parametrize('lat,lon,lat_ref,lon_ref,country_code', [
        pytest.param(-33.8688, 151.2093, 'S', 'E', 'AU', id="sydney_southern_hemisphere"),
        pytest.param(51.5074, -0.1278, 'N', 'W', 'GB', id="london_western_hemisphere"),
    ])
    async def test_negative_coordinates_hemispheres(self, tmp_path: Path, lat: float | int | None, lon: float | int | None, lat_ref: str, lon_ref: str, country_code: str) -> None:
        """A GPS spot south or west of zero gives the right country."""
        result, status = await self._run_gps_test(tmp_path, lat, lon, lat_ref, lon_ref)
        assert status == 0
        assert country_code in result

    async def test_exiftool_failure_fallback_to_pillow(self, tmp_path: Path) -> None:
        """An image is still described when the metadata reader fails."""
        test_image = self._create_test_image(tmp_path, "test_fallback.jpg",
                                             size=(200, 150), color='white')

        result, status = await _run_describe_with_mock(test_image, cli_proc=_mock_exiftool_failure())
        assert status == 0
        assert "desc" in result


class TestDescribeImageHeicProcessing:
    """Test describe_image with many kinds of image."""

    @pytest.fixture
    def heic_test_file(self: Self, data_dir: str) -> str:
        """Path to the HEIC test image."""
        return os.path.join(data_dir, 'test_image.heic')

    @patch('app.lib.reason.describe_image.is_document')
    @patch('app.lib.reason.describe_image.reason')
    async def test_heic_file_processes_successfully(self, mock_reason: MagicMock, mock_is_document: MagicMock, heic_test_file: str) -> None:
        """A HEIC image is described."""
        if not os.path.exists(heic_test_file):
            pytest.skip("HEIC test file not available")

        mock_is_document.return_value = False
        mock_reason.return_value = ("Test description", 0)

        result, status = await describe_image(heic_test_file, "")

        assert status == 0
        assert "Test description" in result

    @pytest.mark.parametrize('width,height,label', [
        pytest.param(100, 100, "small_square"),
        pytest.param(500, 300, "medium_landscape"),
        pytest.param(_MOCK_IMAGE_WIDTH, _MOCK_IMAGE_HEIGHT, "large_landscape"),
        pytest.param(_MOCK_IMAGE_HEIGHT, _MOCK_IMAGE_WIDTH, "large_portrait"),
        pytest.param(_MOCK_MEDIUM_WIDTH, 500, "medium_landscape_1000x500"),
        pytest.param(500, _MOCK_MEDIUM_WIDTH, "medium_portrait_500x1000"),
        pytest.param(1600, 900, "aspect_16_9_landscape"),
        pytest.param(900, 1600, "aspect_9_16_portrait"),
    ])
    async def test_various_dimensions_processed(self, tmp_path: Path, width: int, height: int, label: str) -> None:
        """Images of many sizes are described."""
        test_image = tmp_path / f"{label}.png"
        Image.new('RGB', (width, height), color='red').save(test_image)

        with patch('app.lib.reason.describe_image.is_document', return_value=False):
            with patch('app.lib.reason.describe_image.reason', return_value=("desc", 0)):
                result, status = await describe_image(str(test_image), "")

        assert status == 0
        assert "desc" in result

    @pytest.mark.parametrize('filename,format_name', [
        ('test.png', 'PNG'),
        ('test.bmp', 'BMP'),
        ('test.gif', 'GIF'),
        ('test.tiff', 'TIFF'),
    ])
    async def test_various_formats_process_successfully(self, tmp_path: Path, filename: str, format_name: str) -> None:
        """Images saved in many formats are described."""
        test_image = tmp_path / filename
        Image.new('RGB', (100, 100), color='purple').save(test_image, format=format_name)

        with patch('app.lib.reason.describe_image.is_document', return_value=False):
            with patch('app.lib.reason.describe_image.reason', return_value=("desc", 0)):
                result, status = await describe_image(str(test_image), "")

        assert status == 0, f"Failed for format {format_name}"
        assert "desc" in result, f"No result for format {format_name}"





def _mock_ocr_response(text: str) -> MagicMock:
    """Build a stand in text reading reply with the given text."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'choices': [{'message': {'content': text}}]}
    return mock_response


_OCR_VARS = ("OCR_BACKEND_URL", "OCR_MODEL", "OCR_API_KEY")
_OCR_TEXT = "a" * 120


def _set_ocr_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    """Clear the OCR settings, then set the given ones."""
    for name in _OCR_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


class TestDescribeImageOcrProcessing:
    """Test reading the text out of a document image."""

    @pytest.fixture(autouse=True)
    def _ocr_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Point OCR at a stand in server with no model or key."""
        _set_ocr_env(monkeypatch, {"OCR_BACKEND_URL": "http://ocr.test:1"})

    @patch('app.lib.reason.describe_image.httpx.AsyncClient.post', new_callable=AsyncMock)
    @patch('app.lib.reason.describe_image.is_document')
    @pytest.mark.parametrize('ocr_text', [
        "This is clean OCR text without any errors or issues detected. It contains enough characters to pass the threshold validation.",
        pytest.param('a' * 100, id="english_100_chars"),
        pytest.param('你' * 100, id="chinese_100_chars"),
        pytest.param('abc你好123' * 20, id="mixed_en_cn_numbers"),
        pytest.param('a' * 50 + '<!-- ' + 'b' * 100 + ' -->' + 'c' * 50, id="html_comments_included"),
        pytest.param('a' * 100 + '!!!' * 50, id="symbols_with_100_valid_chars"),
    ])
    async def test_sufficient_ocr_text_accepted(self, mock_is_document: MagicMock, mock_post: MagicMock, tmp_path: Path, ocr_text: str) -> None:
        """Text read from a document image is returned when there is enough of it."""
        test_image = _create_test_images(tmp_path, 1)[0]
        mock_post.return_value = _mock_ocr_response(ocr_text)
        mock_is_document.return_value = True

        result, status = await describe_image([test_image], "")

        assert status == 0
        assert ocr_text in result

    @patch('app.lib.reason.describe_image.httpx.AsyncClient.post', new_callable=AsyncMock)
    @patch('app.lib.reason.describe_image.is_document')
    @pytest.mark.parametrize('ocr_text,expect_in_result', [
        pytest.param("Some short text", "llm desc", id="low_quality_text"),
        pytest.param("", "llm desc", id="empty_output"),
        pytest.param("   \n\n   ", "llm desc", id="whitespace_only"),
        pytest.param("abc", "llm desc", id="short_text_3_chars"),
        pytest.param("!@#$%^&*()" * 20, "llm desc", id="special_chars_only"),
        pytest.param('a' * 99, "llm desc", id="99_chars_below_threshold"),
    ])
    async def test_insufficient_ocr_falls_back_to_llm(self, mock_is_document: MagicMock, mock_post: MagicMock, tmp_path: Path, ocr_text: str, expect_in_result: str) -> None:
        """Too little text read from an image falls back to the model description."""
        test_image = _create_test_images(tmp_path, 1)[0]
        mock_post.return_value = _mock_ocr_response(ocr_text)
        mock_is_document.return_value = True

        with patch('app.lib.reason.describe_image.reason', return_value=(expect_in_result, 0)):
            result, status = await describe_image([test_image], "")

        assert status == 0
        assert expect_in_result in result


class TestDescribeImageOcrBackend:
    """Test which OCR server is asked, and how."""

    @patch('app.lib.reason.describe_image.httpx.AsyncClient.post', new_callable=AsyncMock)
    @patch('app.lib.reason.describe_image.is_document', return_value=True)
    @pytest.mark.parametrize("base", ["http://ocr.test:1", "http://ocr.test:1/", "http://ocr.test:1/v1"])
    async def test_ocr_url_has_one_v1(self, _mock_is_document: MagicMock, mock_post: AsyncMock,
                                      monkeypatch: pytest.MonkeyPatch, tmp_path: Path, base: str) -> None:
        """An OCR server given with or without /v1 is asked at one /v1/chat/completions."""
        _set_ocr_env(monkeypatch, {"OCR_BACKEND_URL": base})
        mock_post.return_value = _mock_ocr_response(_OCR_TEXT)

        result, status = await describe_image(_create_test_images(tmp_path, 1), "")

        assert status == 0
        assert _OCR_TEXT in result
        assert mock_post.call_args.args[0] == "http://ocr.test:1/v1/chat/completions"

    @patch('app.lib.reason.describe_image.httpx.AsyncClient.post', new_callable=AsyncMock)
    @patch('app.lib.reason.describe_image.is_document', return_value=True)
    async def test_ocr_sends_model_and_key_when_set(self, _mock_is_document: MagicMock, mock_post: AsyncMock,
                                                    monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A set OCR model is in the body and a set key is sent as a bearer token."""
        _set_ocr_env(monkeypatch, {
            "OCR_BACKEND_URL": "http://ocr.test:1", "OCR_MODEL": "glm-ocr", "OCR_API_KEY": "sk-ocr",
        })
        mock_post.return_value = _mock_ocr_response(_OCR_TEXT)

        await describe_image(_create_test_images(tmp_path, 1), "")

        assert mock_post.call_args.kwargs["json"]["model"] == "glm-ocr"
        assert mock_post.call_args.kwargs["headers"] == {"Authorization": "Bearer sk-ocr"}

    @patch('app.lib.reason.describe_image.httpx.AsyncClient.post', new_callable=AsyncMock)
    @patch('app.lib.reason.describe_image.is_document', return_value=True)
    async def test_ocr_omits_model_and_key_when_unset(self, _mock_is_document: MagicMock, mock_post: AsyncMock,
                                                      monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """With no OCR model or key set, the body has no model and no Authorization header is sent."""
        _set_ocr_env(monkeypatch, {"OCR_BACKEND_URL": "http://ocr.test:1"})
        mock_post.return_value = _mock_ocr_response(_OCR_TEXT)

        await describe_image(_create_test_images(tmp_path, 1), "")

        assert "model" not in mock_post.call_args.kwargs["json"]
        assert "Authorization" not in mock_post.call_args.kwargs["headers"]

    @patch('app.lib.reason.describe_image.httpx.AsyncClient.post', new_callable=AsyncMock)
    @patch('app.lib.reason.describe_image.is_document', return_value=True)
    async def test_unset_ocr_logs_once_and_uses_the_llm(self, _mock_is_document: MagicMock, mock_post: AsyncMock,
                                                        monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
        """With OCR unset, three document images log one line, ask no OCR server, and get LLM descriptions."""
        _set_ocr_env(monkeypatch, {})

        llm_answer = "desc one\n===\ndesc two\n===\ndesc three"
        with patch('app.lib.reason.describe_image.reason', return_value=(llm_answer, 0)):
            result, status = await describe_image(_create_test_images(tmp_path, 3), "")

        assert status == 0
        assert all(desc in result for desc in ("desc one", "desc two", "desc three"))
        mock_post.assert_not_called()
        assert capsys.readouterr().err.count("[config] OCR_BACKEND_URL is not set") == 1


def _describe_or_answer(answer_reply: tuple[str, int]) -> Callable[..., tuple[str, int]]:
    """Build a stand in model that describes images and answers text questions.

    Args:
        answer_reply: What the model gives back when no image is sent.

    Returns:
        A stand in for the model call.
    """
    def _reply(*_args: str, **kwargs: bool | list[str]) -> tuple[str, int]:
        if kwargs.get("image_files"):
            return ("desc one\n===\ndesc two", 0)
        return answer_reply
    return _reply


class TestDescribeImageQuestion:
    """Test answering a question about images."""

    @pytest.mark.parametrize("context", ["", "  \n "])
    async def test_empty_question_gives_description(self, tmp_path: Path, context: str) -> None:
        """With no question, the images are described with one model call."""
        test_images = _create_test_images(tmp_path, 2)
        with patch('app.lib.reason.describe_image.is_document', return_value=False):
            with patch('app.lib.reason.describe_image.reason',
                       side_effect=_describe_or_answer(("Unused answer", 0))) as mock_reason:
                result, status = await describe_image(test_images, context)

        assert (result, status) == ("desc one\n==================\ndesc two", 0)
        mock_reason.assert_awaited_once()
        assert mock_reason.call_args.args[1] == context
        assert len(mock_reason.call_args.kwargs["image_files"]) == 2

    async def test_question_is_answered_from_all_descriptions(self, tmp_path: Path) -> None:
        """A question gets one answer made from the descriptions of every image."""
        question = "What color is the bridge?"
        test_images = _create_test_images(tmp_path, 2)
        with patch('app.lib.reason.describe_image.is_document', return_value=False):
            with patch('app.lib.reason.describe_image.reason',
                       side_effect=_describe_or_answer((" Red. ", 0))) as mock_reason:
                result, status = await describe_image(test_images, f"  {question}  ")

        assert (result, status) == ("Red.", 0)
        describe_call, answer_call = mock_reason.call_args_list
        assert describe_call.args[1] == f"  {question}  "
        prompt, content = answer_call.args
        assert prompt.endswith(f"User question: {question}")
        assert "using only the content" in prompt
        assert content == "desc one\n==================\ndesc two"
        assert "image_files" not in answer_call.kwargs

    async def test_question_is_answered_from_ocr_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A question about a document image is answered from the text read off it."""
        _set_ocr_env(monkeypatch, {"OCR_BACKEND_URL": "http://ocr.test:1"})
        question = "What is the total?"
        with patch('app.lib.reason.describe_image.httpx.AsyncClient.post',
                   new_callable=AsyncMock, return_value=_mock_ocr_response(_OCR_TEXT)):
            with patch('app.lib.reason.describe_image.is_document', return_value=True):
                with patch('app.lib.reason.describe_image.reason',
                           return_value=("The total is 42.", 0)) as mock_reason:
                    result, status = await describe_image(_create_test_images(tmp_path, 1), question)

        assert (result, status) == ("The total is 42.", 0)
        mock_reason.assert_awaited_once()
        prompt, content = mock_reason.call_args.args
        assert prompt.endswith(f"User question: {question}")
        assert content == _OCR_TEXT

    @pytest.mark.parametrize("answer_reply", [("", 0), ("  \n", 0), ("Backend unavailable", 1)])
    async def test_missing_answer_gives_description(
        self, tmp_path: Path, answer_reply: tuple[str, int],
    ) -> None:
        """When no answer comes back, the descriptions are given instead."""
        test_images = _create_test_images(tmp_path, 2)
        with patch('app.lib.reason.describe_image.is_document', return_value=False):
            with patch('app.lib.reason.describe_image.reason',
                       side_effect=_describe_or_answer(answer_reply)) as mock_reason:
                result, status = await describe_image(test_images, "What color is the bridge?")

        assert (result, status) == ("desc one\n==================\ndesc two", 0)
        assert mock_reason.await_count == 2


@pytest.mark.skipif(not is_llm_available(), reason="LLM server not available")
@pytest.mark.live
class TestDescribeImageLive:
    """Live tests with a real model server."""

    @pytest.mark.timeout(120)
    async def test_describe_text_document_photo(self, tmp_path: Path, data_dir: str) -> None:
        """A photographed document is transcribed with its wording intact."""
        src = os.path.join(data_dir, "test_image_real.jpg")
        dst = tmp_path / "test_image_real.jpg"
        shutil.copy(src, dst)

        result, status = await describe_image(str(dst))

        assert status == 0
        lower = result.lower()
        assert "mosquito" in lower
        assert "district" in lower
        assert "assessment" in lower

    @pytest.mark.timeout(120)
    async def test_describe_red_square(self, tmp_path: Path) -> None:
        """A solid red square with no metadata is described by its dominant color."""
        path = str(tmp_path / "red_square.jpg")
        Image.new("RGB", (32, 32), color="red").save(path)

        result, status = await describe_image(path)

        assert status == 0
        lower = result.lower()
        assert "red" in lower

    @pytest.mark.timeout(120)
    async def test_analyze_outdoor_photo(self, tmp_path: Path, data_dir: str) -> None:
        """A photo taken with a Google Pixel 8 at the Golden Gate Bridge is described with its camera and place."""
        src = os.path.join(data_dir, "outdoor_photo.jpg")
        dst = tmp_path / "outdoor_photo.jpg"
        shutil.copy(src, dst)

        result, status = await describe_image(str(dst), "Describe this image. What do you see?")

        assert status == 0
        assert "Camera: Google Pixel 8" in result
        assert "Marin County" in result
        assert "bridge" in result.lower()

    @pytest.mark.timeout(120)
    async def test_describe_image_answers_question(self, tmp_path: Path, data_dir: str) -> None:
        """A photographed document gets a short answer to a question about it."""
        src = os.path.join(data_dir, "test_image_real.jpg")
        dst = tmp_path / "test_image_real.jpg"
        shutil.copy(src, dst)

        result, status = await describe_image(
            str(dst), "What time is the public hearing on May 19, 2026?"
        )

        assert status == 0
        assert "10:00" in result
        assert len(result) < 200
