"""Tests for app.tools.tts."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app.lib.shared import config
from app.lib.tts.tts import TtsError
from app.tools.tts import tts_tool
from tests.app.lib.tool.conftest import run_text
from tests.app.lib.tts.conftest import (
    LONG_SPLIT_TEXT,
    MOCK_TTS_ENV,
    create_failing_speech_handler,
    create_speech_http_recorder,
    patch_speech_http_client,
)

_ERROR_BODY = b'{"detail":[{"type":"missing","loc":["body","model"],"msg":"Field required"}]}'


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
    with patch("app.tools.tts.generate_tts") as mock_generate_tts:
        mock_generate_tts.side_effect = lambda _text, **kwargs: kwargs["output_file"]
        result = await run_text(tts_tool, "Hello world.")

    output_file = mock_generate_tts.call_args.kwargs["output_file"]
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
    with patch("app.tools.tts.generate_tts") as mock_generate_tts:
        mock_generate_tts.side_effect = lambda _text, **kwargs: kwargs["output_file"]
        await run_text(tts_tool, "Hello", format="mp3")

    output_file = mock_generate_tts.call_args.kwargs["output_file"]
    assert os.path.dirname(output_file) == str(tmp_path / "tts")
    assert output_file.endswith(".mp3")
    assert mock_generate_tts.call_args.kwargs["audio_format"] == "mp3"


async def test_default_format_is_opus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no format given, the audio file is opus."""
    _point_resources_at(tmp_path, monkeypatch)
    with patch("app.tools.tts.generate_tts") as mock_generate_tts:
        mock_generate_tts.side_effect = lambda _text, **kwargs: kwargs["output_file"]
        await run_text(tts_tool, "Hello")

    assert mock_generate_tts.call_args.kwargs["output_file"].endswith(".opus")
    assert mock_generate_tts.call_args.kwargs["audio_format"] == "opus"


async def test_verbose_logging_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool turns on verbose output."""
    _point_resources_at(tmp_path, monkeypatch)
    with patch("app.tools.tts.generate_tts") as mock_generate_tts:
        mock_generate_tts.return_value = str(tmp_path / "tts" / "out.opus")
        await run_text(tts_tool, "Hello")

    assert mock_generate_tts.call_args.kwargs["is_verbose"] is True


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (
            config.MissingSettingError(
                "LOCAL_TTS_URL is not set: needed by tts and /v1/audio/speech"
            ),
            "Tool 'tts' failed: LOCAL_TTS_URL is not set: needed by tts and /v1/audio/speech",
        ),
        (
            TtsError("Error calling TTS API: HTTP 500: server error"),
            "Tool 'tts' failed: Error calling TTS API: HTTP 500: server error",
        ),
    ],
    ids=["missing-setting", "tts-error"],
)
async def test_error_is_returned_to_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_message: str,
) -> None:
    """A failed run gives back a failed tool message with the error text."""
    _point_resources_at(tmp_path, monkeypatch)
    with patch("app.tools.tts.generate_tts", side_effect=error):
        result = await run_text(tts_tool, "Hello")

    assert result == expected_message


def _files_holding_error_body(*roots: Path) -> list[Path]:
    """List every file under the folders that holds the server's error text."""
    return [
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and b"Field required" in path.read_bytes()
    ]


@pytest.mark.parametrize(
    ("text", "successful_segments"),
    [("Hello world.", 0), (LONG_SPLIT_TEXT, 1)],
    ids=["short-text", "long-text-second-piece-fails"],
)
async def test_server_error_fails_the_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    successful_segments: int,
) -> None:
    """A 422 from the TTS server fails the call with its status and text, and leaves no temp file."""
    resources = tmp_path / "resources"
    scratch = tmp_path / "scratch"
    _point_resources_at(resources, monkeypatch)
    monkeypatch.setenv("SCRATCH_DIR", str(scratch))
    recorder = create_speech_http_recorder(
        create_failing_speech_handler(
            status_code=422,
            body=_ERROR_BODY,
            successful_segments=successful_segments,
        )
    )

    with (
        patch.dict(os.environ, MOCK_TTS_ENV, clear=False),
        patch("app.lib.tts.tts.reason", return_value=(text, 0)),
        patch_speech_http_client(recorder),
    ):
        result = await run_text(tts_tool, text)

    assert result.startswith("Tool 'tts' failed: Error calling TTS API: HTTP 422: ")
    assert "Field required" in result
    assert len(recorder.requests) == successful_segments + 1
    assert list((scratch / "tts").iterdir()) == []
    assert _files_holding_error_body(resources, scratch) == []
