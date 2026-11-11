"""Shared setup and helpers for the tts tests."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

from tests.app.lib.conftest import mock_cli_proc

MOCK_TTS_ENV = {
    "LOCAL_TTS_URL": "http://test-tts",
    "LOCAL_TTS_VOICE_EN": "af_alloy",
    "LOCAL_TTS_VOICE_ZH": "zm_yunxia",
    "LOCAL_TTS_MODEL": "voxcpm2",
}

CURL_HTTP_ERROR_EXIT = 22

LONG_SPLIT_TEXT = "This is a very long text that will be split into multiple segments. " * 20


def failing_curl(
    body: bytes, stderr: str, return_code: int, ok_pieces: int = 0
) -> Callable[..., MagicMock]:
    """Make a stand in for curl that gives audio for the first pieces, then fails.

    Args:
        body: Text the failed call writes to its output file.
        stderr: Text the failed call gives back on stderr.
        return_code: Exit code of the failed call.
        ok_pieces: How many calls get audio before the failed one.

    Returns:
        A side effect for the create_subprocess_exec mock.
    """
    calls = {"curl": 0}

    def _side_effect(*cmd: str, **_kwargs: int) -> MagicMock:
        out_path = Path(cmd[cmd.index("--output") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        calls["curl"] += 1
        if calls["curl"] <= ok_pieces:
            out_path.write_bytes(b"fake_audio_data")
            return mock_cli_proc()
        out_path.write_bytes(body)
        return mock_cli_proc(stderr=stderr, returncode=return_code)

    return _side_effect
