"""Tests for app.tools.tts."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app.tools.tts import tts_tool
from tests.app.lib.tool.conftest import run_text
from tests.app.lib.tts.conftest import (
    CURL_HTTP_ERROR_EXIT,
    LONG_SPLIT_TEXT,
    MOCK_TTS_ENV,
    failing_curl,
)

_ERROR_BODY = b'{"detail":[{"type":"missing","loc":["body","model"],"msg":"Field required"}]}'
_ERROR_STATUS_LINE = "curl: (22) The requested URL returned error: 422"


def _point_resources_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the resources folder at a temp folder and set a base URL."""
    monkeypatch.setenv("RESOURCES_DIR", str(tmp_path))
    monkeypatch.setenv("DAV_BASE_URL", "http://example.com:8315")


async def test_success_returns_resource_path_and_download_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A good run gives back the tts path, a download command, and the file tool name."""
    _point_resources_at(tmp_path, monkeypatch)
    with patch("app.tools.tts.generate_tts") as mock_gen:
        mock_gen.side_effect = lambda _text, **kwargs: (kwargs["output_file"], 0)
        result = await run_text(tts_tool, "Hello world.")

    output_file = mock_gen.call_args.kwargs["output_file"]
    assert os.path.dirname(output_file) == str(tmp_path / "tts")
    assert "tts/tts_" in result
    assert "http://example.com:8315/dav/tts/tts_" in result
    assert "curl -o /tmp/" in result
    assert "send_or_receive_file" in result


async def test_output_file_written_into_tts_subdir_with_format_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audio file lands in the tts folder and ends with the chosen format."""
    _point_resources_at(tmp_path, monkeypatch)
    with patch("app.tools.tts.generate_tts") as mock_gen:
        mock_gen.side_effect = lambda _text, **kwargs: (kwargs["output_file"], 0)
        await run_text(tts_tool, "Hello", format="mp3")

    output_file = mock_gen.call_args.kwargs["output_file"]
    assert os.path.dirname(output_file) == str(tmp_path / "tts")
    assert output_file.endswith(".mp3")


async def test_default_format_is_opus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no format given, the audio file is opus."""
    _point_resources_at(tmp_path, monkeypatch)
    with patch("app.tools.tts.generate_tts") as mock_gen:
        mock_gen.side_effect = lambda _text, **kwargs: (kwargs["output_file"], 0)
        await run_text(tts_tool, "Hello")

    assert mock_gen.call_args.kwargs["output_file"].endswith(".opus")


async def test_verbose_logging_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool turns on verbose output."""
    _point_resources_at(tmp_path, monkeypatch)
    with patch("app.tools.tts.generate_tts") as mock_gen:
        mock_gen.return_value = (str(tmp_path / "tts" / "out.opus"), 0)
        await run_text(tts_tool, "Hello")

    assert mock_gen.call_args.kwargs["verbose"] is True


async def test_error_is_returned_to_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed run gives back a failed tool message with the error text."""
    _point_resources_at(tmp_path, monkeypatch)
    with patch("app.tools.tts.generate_tts") as mock_gen:
        mock_gen.return_value = ("LOCAL_TTS_URL is not set: needed by tts and /v1/audio/speech", 1)
        result = await run_text(tts_tool, "Hello")

    assert result == "Tool 'tts' failed: LOCAL_TTS_URL is not set: needed by tts and /v1/audio/speech"


def _files_holding_error_body(*roots: Path) -> list[Path]:
    """List every file under the folders that holds the server's error text."""
    return [
        path for root in roots for path in root.rglob("*")
        if path.is_file() and b"Field required" in path.read_bytes()
    ]


@pytest.mark.parametrize(
    ("text", "ok_pieces"),
    [("Hello world.", 0), (LONG_SPLIT_TEXT, 1)],
    ids=["short-text", "long-text-second-piece-fails"],
)
async def test_server_error_fails_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, ok_pieces: int
) -> None:
    """A 422 from the TTS server fails the call with its status and text, and leaves no temp file."""
    resources = tmp_path / "resources"
    scratch = tmp_path / "scratch"
    _point_resources_at(resources, monkeypatch)
    monkeypatch.setenv("SCRATCH_DIR", str(scratch))

    with patch.dict(os.environ, MOCK_TTS_ENV, clear=False), \
         patch("app.lib.tts.tts.reason") as mock_reason, \
         patch("asyncio.create_subprocess_exec") as mock_run:
        mock_reason.return_value = (text, 0)
        mock_run.side_effect = failing_curl(
            _ERROR_BODY, _ERROR_STATUS_LINE, CURL_HTTP_ERROR_EXIT, ok_pieces
        )
        result = await run_text(tts_tool, text)

    assert result.startswith("Tool 'tts' failed: Error calling TTS API: ")
    assert _ERROR_STATUS_LINE in result
    assert "Field required" in result
    assert "Segment synthesis failed" not in result
    assert mock_run.call_count == ok_pieces + 1
    assert list((scratch / "tts").iterdir()) == []
    assert _files_holding_error_body(resources, scratch) == []
