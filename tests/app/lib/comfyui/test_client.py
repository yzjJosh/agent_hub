"""Tests for app.lib.comfyui.client."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.lib.comfyui.client import ComfyUIClient, GenerationResult

_BASE = "http://comfyui.test:8188"

_PROMPT_TIMEOUT_READ_S = 720
_CONNECT_TIMEOUT_S = 10


def _point_at(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the client at the test backend address."""
    monkeypatch.setenv("COMFYUI_BACKEND_URL", _BASE)


def _response(status_code: int, body: dict | None = None, content: bytes | None = None) -> MagicMock:
    """A stand in for an httpx answer."""
    resp = MagicMock()
    resp.is_success = 200 <= status_code < 300
    resp.status_code = status_code
    resp.text = str(status_code)
    if content is not None:
        resp.content = content
    if body is not None:
        resp.json.return_value = body
    return resp


def _fake_http_client() -> tuple[MagicMock, AsyncMock]:
    """The patched AsyncClient class and the client it hands out.

    Wire the answers on the returned client's post mock.
    """
    mock_client_cls = MagicMock()
    http_client = AsyncMock()
    entered = AsyncMock(return_value=http_client)
    exited = AsyncMock(return_value=False)
    mock_client_cls.return_value.__aenter__ = entered
    mock_client_cls.return_value.__aexit__ = exited
    return mock_client_cls, http_client


class TestGenerate:
    """generate."""

    async def test_returns_decoded_images_and_filenames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The prompt goes out and the base64 images come back decoded."""
        _point_at(monkeypatch)
        prompt = {"1": {"class_type": "SaveImage", "inputs": {}}}
        raw = base64.b64encode(b"image-bytes")
        mock_client_cls, http_client = _fake_http_client()
        http_client.post = AsyncMock(
            return_value=_response(
                200,
                {
                    "id": "run-1",
                    "prompt": prompt,
                    "images": [raw],
                    "filenames": ["media_gen/t2i/out.png"],
                    "stats": {},
                },
            )
        )

        with patch("app.lib.comfyui.client.httpx.AsyncClient", mock_client_cls):
            result = await ComfyUIClient().generate(prompt)

        assert result == GenerationResult(
            images=(b"image-bytes",), filenames=("media_gen/t2i/out.png",)
        )
        assert http_client.post.call_args.args[0] == f"{_BASE}/prompt"
        assert http_client.post.call_args.kwargs["json"] == {"prompt": prompt}

    async def test_multiple_images_preserve_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each base64 image is decoded in the order it was sent."""
        _point_at(monkeypatch)
        first = base64.b64encode(b"first-image")
        second = base64.b64encode(b"second-image")
        mock_client_cls, http_client = _fake_http_client()
        http_client.post = AsyncMock(
            return_value=_response(
                200, {"images": [first, second], "filenames": ["a.png", "b.png"]}
            )
        )

        with patch("app.lib.comfyui.client.httpx.AsyncClient", mock_client_cls):
            result = await ComfyUIClient().generate({})

        assert result.images == (b"first-image", b"second-image")
        assert result.filenames == ("a.png", "b.png")

    async def test_prompt_timeout_is_long(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The read timeout outlasts a long generation, the connect timeout stays short."""
        _point_at(monkeypatch)
        mock_client_cls, http_client = _fake_http_client()
        http_client.post = AsyncMock(
            return_value=_response(200, {"images": [base64.b64encode(b"x")], "filenames": ["x.png"]})
        )

        with patch("app.lib.comfyui.client.httpx.AsyncClient", mock_client_cls):
            await ComfyUIClient().generate({})

        timeout = mock_client_cls.call_args.kwargs["timeout"]
        assert timeout.read == _PROMPT_TIMEOUT_READ_S
        assert timeout.connect == _CONNECT_TIMEOUT_S

    async def test_no_images_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A success that saved nothing raises the no-file message."""
        _point_at(monkeypatch)
        mock_client_cls, http_client = _fake_http_client()
        http_client.post = AsyncMock(return_value=_response(200, {"images": [], "filenames": []}))

        with patch("app.lib.comfyui.client.httpx.AsyncClient", mock_client_cls):
            with pytest.raises(RuntimeError, match="finished without a saved file"):
                await ComfyUIClient().generate({})

    async def test_node_errors_are_quoted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Validation errors in the answer are quoted in the error message."""
        _point_at(monkeypatch)
        errors = {"1": {"errors": [{"message": "Invalid value"}]}}
        mock_client_cls, http_client = _fake_http_client()
        http_client.post = AsyncMock(return_value=_response(400, {"node_errors": errors}))

        with patch("app.lib.comfyui.client.httpx.AsyncClient", mock_client_cls):
            with pytest.raises(RuntimeError, match="ComfyUI rejected the prompt"):
                await ComfyUIClient().generate({})

    @pytest.mark.parametrize(
        ("status_code", "message"),
        [
            (503, "GPU is busy, the ComfyUI proxy rejected the request, retry later"),
            (400, "ComfyUI rejected the request: HTTP 400"),
        ],
    )
    async def test_non_2xx_raises(
        self, monkeypatch: pytest.MonkeyPatch, status_code: int, message: str
    ) -> None:
        """A non-2xx answer raises with the matching message."""
        _point_at(monkeypatch)
        mock_client_cls, http_client = _fake_http_client()
        http_client.post = AsyncMock(return_value=_response(status_code))

        with patch("app.lib.comfyui.client.httpx.AsyncClient", mock_client_cls):
            with pytest.raises(RuntimeError, match=message):
                await ComfyUIClient().generate({})

    async def test_unreachable_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A timeout raises the unreachable message."""
        _point_at(monkeypatch)
        mock_client_cls, http_client = _fake_http_client()
        http_client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

        with patch("app.lib.comfyui.client.httpx.AsyncClient", mock_client_cls):
            with pytest.raises(RuntimeError, match="could not reach ComfyUI"):
                await ComfyUIClient().generate({})
