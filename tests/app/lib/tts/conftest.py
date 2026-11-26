"""Shared setup and helpers for the speech synthesis tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx

from tests.app.lib.conftest import mock_cli_proc

MOCK_TTS_ENV = {
    "LOCAL_TTS_URL": "http://test-tts",
    "LOCAL_TTS_VOICE_EN": "af_alloy",
    "LOCAL_TTS_VOICE_ZH": "zm_yunxia",
    "LOCAL_TTS_MODEL": "voxcpm2",
}

FAKE_AUDIO_BYTES = b"fake_audio_data"

LONG_SPLIT_TEXT = "This is a very long text that will be split into multiple segments. " * 20


@dataclass(frozen=True)
class RecordedSpeechRequest:
    """One HTTP request captured from the speech client."""

    url: str
    payload: dict[str, str]
    headers: dict[str, str]
    follow_redirects: bool


class SpeechHttpRecorder:
    """Record speech HTTP requests through a mock client."""

    def __init__(
        self,
        handler: Callable[[httpx.Request], httpx.Response],
    ) -> None:
        self._handler = handler
        self.requests: list[RecordedSpeechRequest] = []

    def create_client(
        self,
        *,
        follow_redirects: bool = False,
        timeout: httpx.Timeout | float | None = None,
    ) -> httpx.AsyncClient:
        """Build a mock HTTP client that records each speech request."""

        def _record_and_handle_request(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            self.requests.append(
                RecordedSpeechRequest(
                    url=str(request.url),
                    payload=payload,
                    headers=dict(request.headers),
                    follow_redirects=follow_redirects,
                )
            )
            return self._handler(request)

        return httpx.AsyncClient(
            transport=httpx.MockTransport(_record_and_handle_request),
            follow_redirects=follow_redirects,
            timeout=timeout,
        )

    @property
    def payloads(self) -> list[dict[str, str]]:
        """Return JSON bodies from all recorded speech requests."""
        return [recorded.payload for recorded in self.requests]

    @property
    def urls(self) -> list[str]:
        """Return target URLs from all recorded speech requests."""
        return [recorded.url for recorded in self.requests]


def create_speech_http_recorder(
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
) -> SpeechHttpRecorder:
    """Build a SpeechHttpRecorder that returns fake audio by default."""
    if handler is None:
        return SpeechHttpRecorder(
            lambda _request: httpx.Response(200, content=FAKE_AUDIO_BYTES)
        )
    return SpeechHttpRecorder(handler)


@contextmanager
def patch_speech_http_client(recorder: SpeechHttpRecorder) -> Iterator[None]:
    """Route speech requests through recorder without affecting other clients."""
    speech_http_proxy = SimpleNamespace(
        **(vars(httpx) | {"AsyncClient": recorder.create_client})
    )
    with patch("app.lib.tts.tts.httpx", speech_http_proxy):
        yield


def create_failing_speech_handler(
    status_code: int,
    body: bytes = b"",
    successful_segments: int = 0,
) -> Callable[[httpx.Request], httpx.Response]:
    """Build a speech handler that succeeds for successful_segments calls then fails."""
    call_count = 0

    def _handle_request(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count <= successful_segments:
            return httpx.Response(200, content=FAKE_AUDIO_BYTES)
        return httpx.Response(status_code, content=body)

    return _handle_request


def create_ffmpeg_subprocess_mock(
    return_code: int = 0,
    stderr: str = "",
) -> Callable[..., MagicMock]:
    """Build a subprocess side effect for ffmpeg concatenation calls."""

    def _run_fake_ffmpeg(*command: str, **_kwargs: int) -> MagicMock:
        if command and command[0] == "ffmpeg" and return_code == 0:
            output_path = command[-1]
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as output_handle:
                output_handle.write(FAKE_AUDIO_BYTES)
        return mock_cli_proc(stderr=stderr, returncode=return_code)

    return _run_fake_ffmpeg
