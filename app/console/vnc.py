from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

import websockets
from starlette.websockets import WebSocket
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

__all__ = ["vnc_reachable", "websockify_proxy"]

_UPSTREAM_WS = "ws://127.0.0.1:6080/websockify"
_PROBE_HOST = "127.0.0.1"
_PROBE_PORT = 6080
_PROBE_TIMEOUT = 0.5


async def vnc_reachable(host: str = _PROBE_HOST, port: int = _PROBE_PORT) -> bool:
    """Check whether the screen sharing service answers.

    It waits half a second at most.

    Args:
        host: Where the service listens.
        port: Which port the service listens on.

    Returns:
        True when the service answers.
    """
    try:
        _reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), _PROBE_TIMEOUT)
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()
    return True


async def _client_to_upstream(websocket: WebSocket, send: Callable[[bytes], Awaitable[None]]) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] != "websocket.receive":
            return
        data = message.get("bytes")
        if data is None:
            data = message.get("text", "").encode("utf-8")
        if data:
            await send(data)


async def _upstream_to_client(recv: Callable[[], Awaitable[bytes]], websocket: WebSocket) -> None:
    while True:
        data = await recv()
        if data:
            await websocket.send_bytes(data)


async def _pump_both(a: Awaitable[None], b: Awaitable[None]) -> None:
    """Run two jobs at the same time until one of them stops.

    The other job is stopped too. An error from either job is raised.

    Args:
        a: The first job.
        b: The second job.
    """
    first = asyncio.ensure_future(a)
    second = asyncio.ensure_future(b)
    try:
        done, _ = await asyncio.wait({first, second}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


async def _safe_close(websocket: WebSocket, code: int) -> None:
    try:
        await websocket.close(code=code)
    except Exception:
        pass


async def websockify_proxy(websocket: WebSocket) -> None:
    """Pass screen data between the page and the screen sharing service.

    Data moves both ways until one side stops. A clean stop closes the page
    with code 1000 and any failure closes it with code 1011.

    Args:
        websocket: The page's connection.
    """
    await websocket.accept()
    try:
        async with websockets.connect(_UPSTREAM_WS) as upstream:
            await _pump_both(
                _client_to_upstream(websocket, upstream.send),
                _upstream_to_client(upstream.recv, websocket),
            )
        await _safe_close(websocket, 1000)
    except ConnectionClosedError:
        await _safe_close(websocket, 1011)
    except ConnectionClosed:
        await _safe_close(websocket, 1000)
    except Exception:
        await _safe_close(websocket, 1011)
