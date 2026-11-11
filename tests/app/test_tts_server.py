"""Tests for app.tts_server."""

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Protocol

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Mount

from app.tts_server import create_openai_tts_app

_FAKE_AUDIO = b"FAKE-AUDIO"


class _TaskStore(Protocol):
    """The task record store the fixture gives to a test."""


@pytest.fixture(autouse=True)
async def tool_infra(task_store: _TaskStore) -> AsyncIterator[_TaskStore]:
    """A throwaway record store for every test."""
    yield task_store


def _make_app() -> Starlette:
    """The speech app mounted at /v1."""
    return Starlette(routes=[Mount("/v1", create_openai_tts_app())])


def _client(app: Starlette) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _point_resources_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the resources dir and the file base URL at test values."""
    monkeypatch.setenv("RESOURCES_DIR", str(tmp_path))
    monkeypatch.setenv("DAV_BASE_URL", "http://example.com:8315")


def _fake_generate(
    audio: bytes = _FAKE_AUDIO,
    error: str | None = None,
    write_file: bool = True,
) -> Callable[..., Awaitable[tuple[str, int]]]:
    """Build a stand in for the speech maker.

    Args:
        audio: Bytes written to the output file.
        error: Error text to fail with, or None to succeed.
        write_file: Whether the output file is written.

    Returns:
        A stand in for the speech maker that gives back a file name and a status.
    """

    async def fake_generate(
        _text: str, output_file: str | None = None, **_kwargs: str | bool | None
    ) -> tuple[str, int]:
        if error is not None:
            return error, 1
        if write_file and output_file is not None:
            Path(output_file).write_bytes(audio)
        return output_file, 0

    return fake_generate


@pytest.fixture
async def tts_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[httpx.AsyncClient]:
    """A client for a speech app whose speech maker writes fake audio."""
    _point_resources_at(tmp_path, monkeypatch)
    monkeypatch.setattr("app.tools.tts.generate_tts", _fake_generate())
    async with _client(_make_app()) as c:
        yield c


async def test_returns_audio_bytes(
    tts_client: httpx.AsyncClient, tmp_path: Path, tool_infra: _TaskStore
) -> None:
    """A good request gives back mp3 audio, saves the file, and records a completed run."""
    response = await tts_client.post("/v1/audio/speech", json={"input": "hello"})

    assert response.status_code == 200
    assert response.content == _FAKE_AUDIO
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["content-disposition"] == "attachment; filename=speech.mp3"
    saved = list((tmp_path / "tts").iterdir())
    assert len(saved) == 1
    assert saved[0].name.endswith(".mp3")
    assert saved[0].read_bytes() == _FAKE_AUDIO
    rows = await tool_infra.list()
    assert len(rows) == 1
    assert rows[0].status == "completed"


@pytest.mark.parametrize(
    ("format", "content_type"),
    [("opus", "audio/opus"), ("aac", "audio/aac"), ("flac", "audio/flac"), ("wav", "audio/wav")],
)
async def test_response_format_sets_content_type(
    tts_client: httpx.AsyncClient, format: str, content_type: str
) -> None:
    """The asked for format sets the content type and the file name."""
    response = await tts_client.post(
        "/v1/audio/speech", json={"input": "hello", "response_format": format}
    )

    assert response.status_code == 200
    assert response.content == _FAKE_AUDIO
    assert response.headers["content-type"] == content_type
    assert response.headers["content-disposition"] == f"attachment; filename=speech.{format}"


async def test_ignores_unsupported_openai_fields(tts_client: httpx.AsyncClient) -> None:
    """Extra fields like model, voice, and speed do not break the request."""
    response = await tts_client.post(
        "/v1/audio/speech",
        json={"input": "hello", "model": "tts-1", "voice": "alloy", "speed": 1.5},
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({}, "Missing required field: input"),
        ({"input": ""}, "Missing required field: input"),
        ({"input": 42}, "Missing required field: input"),
    ],
)
async def test_rejects_missing_or_invalid_input(
    tts_client: httpx.AsyncClient, body: dict, message: str
) -> None:
    """A missing, empty, or non-string input is a client error."""
    response = await tts_client.post("/v1/audio/speech", json=body)

    assert response.status_code == 400
    assert response.json() == {"error": message}


async def test_rejects_non_json_content_type(tts_client: httpx.AsyncClient) -> None:
    """A body that is not marked as JSON is refused."""
    response = await tts_client.post(
        "/v1/audio/speech", content=b'{"input": "hello"}', headers={"Content-Type": "text/plain"}
    )

    assert response.status_code == 400
    assert response.json() == {"error": "Content-Type must be application/json"}


async def test_rejects_invalid_json(tts_client: httpx.AsyncClient) -> None:
    """A malformed JSON body is a client error."""
    response = await tts_client.post(
        "/v1/audio/speech", content=b"not json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 400
    assert response.json()["error"].startswith("Request body")


async def test_rejects_unknown_response_format(tts_client: httpx.AsyncClient) -> None:
    """A format that is not supported is a client error."""
    response = await tts_client.post(
        "/v1/audio/speech", json={"input": "hello", "response_format": "ogg"}
    )

    assert response.status_code == 400
    assert response.json() == {"error": "Unsupported response_format: 'ogg'"}


async def test_tool_failure_returns_500_with_failure_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_infra: _TaskStore
) -> None:
    """A failed speech call gives a 500 with the failure message and records a failed run."""
    _point_resources_at(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "app.tools.tts.generate_tts",
        _fake_generate(error="LOCAL_TTS_URL is not set: needed by tts and /v1/audio/speech"),
    )
    async with _client(_make_app()) as c:
        response = await c.post("/v1/audio/speech", json={"input": "hello"})

    assert response.status_code == 500
    assert response.json() == {
        "error": "Tool 'tts' failed: LOCAL_TTS_URL is not set: needed by tts and /v1/audio/speech"
    }
    rows = await tool_infra.list()
    assert len(rows) == 1
    assert rows[0].status == "failed"


async def test_missing_audio_file_returns_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A success with no audio file to read gives a 500 saying the file could not be read."""
    _point_resources_at(tmp_path, monkeypatch)
    monkeypatch.setattr("app.tools.tts.generate_tts", _fake_generate(write_file=False))
    async with _client(_make_app()) as c:
        response = await c.post("/v1/audio/speech", json={"input": "hello"})

    assert response.status_code == 500
    assert response.json()["error"].startswith("Could not read audio file: ")
