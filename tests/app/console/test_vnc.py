from __future__ import annotations

import asyncio
import socket
from types import TracebackType
from typing import override

import pytest
from starlette.websockets import WebSocket
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from app.console import vnc as vnc_module
from app.console.vnc import vnc_reachable, websockify_proxy

_WS_CLOSE_NORMAL = 1000
_WS_CLOSE_ABNORMAL = 1011
_REPLY_DELAY = 0.05
_UPSTREAM_WS = "ws://127.0.0.1:6080/websockify"


async def test_vnc_reachable_true_and_false() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    assert await vnc_reachable("127.0.0.1", port) is True
    listener.close()
    assert await vnc_reachable("127.0.0.1", port) is False


class _FakeWebSocket(WebSocket):
    def __init__(self) -> None:
        self.out: list[bytes] = []
        self.closed_with: list[int] = []
        self._incoming: list[dict] = []

    @override
    async def accept(self) -> None:
        pass

    @override
    async def receive(self) -> dict:
        await asyncio.sleep(0.3)
        if self._incoming:
            return self._incoming.pop(0)
        return {"type": "websocket.disconnect", "code": _WS_CLOSE_NORMAL}

    @override
    async def send_bytes(self, data: bytes) -> None:
        self.out.append(data)

    @override
    async def close(self, code: int = _WS_CLOSE_NORMAL) -> None:
        self.closed_with.append(code)


class _FakeUpstream(ClientConnection):
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self._replied = False

    async def __aenter__(self) -> _FakeUpstream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        return False

    @override
    async def send(self, data: bytes) -> None:
        self.sent.append(data)

    @override
    async def recv(self) -> bytes:
        if not self._replied:
            self._replied = True
            await asyncio.sleep(_REPLY_DELAY)
            return b"pong"
        await asyncio.sleep(30)


class _ClosingUpstream(_FakeUpstream):
    @override
    async def recv(self) -> bytes:
        await asyncio.sleep(_REPLY_DELAY)
        raise ConnectionClosedOK(None, None)


class _AbortingUpstream(_FakeUpstream):
    @override
    async def recv(self) -> bytes:
        await asyncio.sleep(_REPLY_DELAY)
        raise ConnectionClosedError(None, None)


class _AbortingSendUpstream(_FakeUpstream):
    @override
    async def send(self, _data: bytes) -> None:
        raise ConnectionClosedError(None, None)


async def test_websockify_proxy_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeUpstream()

    def fake_connect(uri: str) -> _FakeUpstream:
        assert uri == _UPSTREAM_WS
        return fake

    monkeypatch.setattr(vnc_module.websockets, "connect", fake_connect)
    ws = _FakeWebSocket()
    ws._incoming.append({"type": "websocket.receive", "bytes": b"ping"})
    await websockify_proxy(ws)
    assert fake.sent == [b"ping"]
    assert ws.out == [b"pong"]
    assert ws.closed_with == [_WS_CLOSE_NORMAL]


async def test_websockify_proxy_refused_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused_connect(_uri: str) -> _FakeUpstream:
        raise OSError("connection refused")

    monkeypatch.setattr(vnc_module.websockets, "connect", refused_connect)
    ws = _FakeWebSocket()
    await websockify_proxy(ws)
    assert ws.closed_with == [_WS_CLOSE_ABNORMAL]


async def test_websockify_proxy_upstream_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vnc_module.websockets,
        "connect",
        lambda _uri: _ClosingUpstream(),
    )
    ws = _FakeWebSocket()
    await websockify_proxy(ws)
    assert ws.closed_with == [_WS_CLOSE_NORMAL]


async def test_websockify_proxy_unclean_upstream_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vnc_module.websockets,
        "connect",
        lambda _uri: _AbortingUpstream(),
    )
    ws = _FakeWebSocket()
    await websockify_proxy(ws)
    assert ws.closed_with == [_WS_CLOSE_ABNORMAL]


async def test_websockify_proxy_unclean_upstream_send(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vnc_module.websockets,
        "connect",
        lambda _uri: _AbortingSendUpstream(),
    )
    ws = _FakeWebSocket()
    ws._incoming.append({"type": "websocket.receive", "bytes": b"ping"})
    await websockify_proxy(ws)
    assert ws.closed_with == [_WS_CLOSE_ABNORMAL]
