"""Tests for app.lib.laya.client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.lib.laya.client import LayaError, Verdict, classify
from app.lib.shared import config

_BASE = "http://laya.test:8888"
_HTML = "<html><body>hello</body></html>"
_REQUEST_TIMEOUT_READ_S = 60
_CONNECT_TIMEOUT_S = 10


def _point_at(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the client at the test server address, with a trailing slash."""
    monkeypatch.setenv("LAYA_SERVER_URL", _BASE + "/")


def _response(status_code: int, body: dict | list | None = None, is_json: bool = True) -> MagicMock:
    """A stand in for an httpx answer."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = f"status {status_code}"
    if is_json:
        resp.json.return_value = body
    else:
        resp.json.side_effect = ValueError("Expecting value")
    return resp


def _fake_http_client() -> tuple[MagicMock, AsyncMock]:
    """The patched AsyncClient class and the client it hands out."""
    mock_client_cls = MagicMock()
    http_client = AsyncMock()
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=http_client)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_client_cls, http_client


async def _classify_with(response: MagicMock) -> tuple[Verdict, MagicMock, AsyncMock]:
    """Run classify against one canned answer."""
    mock_client_cls, http_client = _fake_http_client()
    http_client.post = AsyncMock(return_value=response)
    with patch("app.lib.laya.client.httpx.AsyncClient", mock_client_cls):
        verdict = await classify(_HTML)
    return verdict, mock_client_cls, http_client


class TestClassify:
    """classify."""

    @pytest.mark.parametrize("blocked,probability", [(True, 0.93), (False, 0.04)])
    async def test_parses_verdict(self, monkeypatch: pytest.MonkeyPatch, blocked: bool, probability: float) -> None:
        """A 200 answer gives back the blocked flag and its probability."""
        _point_at(monkeypatch)
        body = {"blocked": blocked, "blocked_probability": probability, "label": "x"}

        verdict, _, http_client = await _classify_with(_response(200, body))

        assert verdict == Verdict(blocked=blocked, blocked_probability=probability)
        assert http_client.post.call_args.args[0] == f"{_BASE}/v1/classify"
        assert http_client.post.call_args.kwargs["json"] == {"html": _HTML}

    async def test_timeout_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The request waits long for a slow server but connects fast."""
        _point_at(monkeypatch)
        body = {"blocked": False, "blocked_probability": 0.1}

        _, mock_client_cls, _ = await _classify_with(_response(200, body))

        timeout = mock_client_cls.call_args.kwargs["timeout"]
        assert timeout.read == _REQUEST_TIMEOUT_READ_S
        assert timeout.connect == _CONNECT_TIMEOUT_S

    @pytest.mark.parametrize("status_code", [413, 422, 503])
    async def test_non_200_raises(self, monkeypatch: pytest.MonkeyPatch, status_code: int) -> None:
        """A page Laya turns down raises with the status code."""
        _point_at(monkeypatch)

        with pytest.raises(LayaError, match=f"HTTP {status_code}"):
            await _classify_with(_response(status_code, {"detail": "no"}))

    async def test_transport_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A server that cannot be reached raises LayaError."""
        _point_at(monkeypatch)
        mock_client_cls, http_client = _fake_http_client()
        http_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch("app.lib.laya.client.httpx.AsyncClient", mock_client_cls):
            with pytest.raises(LayaError, match="could not reach Laya: refused"):
                await classify(_HTML)

    @pytest.mark.parametrize(
        "response",
        [
            _response(200, is_json=False),
            _response(200, ["blocked"]),
            _response(200, {"blocked_probability": 0.5}),
            _response(200, {"blocked": "yes", "blocked_probability": 0.5}),
            _response(200, {"blocked": True}),
            _response(200, {"blocked": True, "blocked_probability": "high"}),
        ],
        ids=["not-json", "not-object", "no-blocked", "blocked-not-bool", "no-probability", "probability-not-number"],
    )
    async def test_malformed_body_raises(self, monkeypatch: pytest.MonkeyPatch, response: MagicMock) -> None:
        """A 200 answer without a usable verdict raises LayaError."""
        _point_at(monkeypatch)

        with pytest.raises(LayaError):
            await _classify_with(response)

    async def test_missing_server_url_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset server address raises the missing setting error."""
        monkeypatch.delenv("LAYA_SERVER_URL", raising=False)

        with pytest.raises(config.MissingSettingError, match="LAYA_SERVER_URL"):
            await classify(_HTML)
