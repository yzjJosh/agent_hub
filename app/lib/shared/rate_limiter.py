"""Run calls one at a time, with a short wait between them."""

from __future__ import annotations

__all__ = ["RateLimiter"]

import asyncio
import contextlib
import math
import time
from typing import Protocol


class _CallTurn(Protocol):
    """The caller's turn in the limiter queue."""


class RateLimiter:
    """Runs calls one at a time, with a short wait between them."""

    def __init__(self, min_delay: float) -> None:
        """Make a limiter.

        Args:
            min_delay: Seconds that must pass between two calls.
        """
        self._min_delay = min_delay
        self._queue: list[_CallTurn] = []
        self._condition = asyncio.Condition()
        self._last_call = -math.inf

    async def acquire(self) -> None:
        """Wait until it is this caller's turn to make a request."""
        my_id: _CallTurn = object()
        async with self._condition:
            self._queue.append(my_id)
            try:
                await self._condition.wait_for(lambda: self._queue[0] is my_id)

                now = time.monotonic()
                remaining = self._min_delay - (now - self._last_call)
                if remaining > 0:
                    await asyncio.sleep(remaining)

                self._last_call = time.monotonic()
            finally:
                with contextlib.suppress(ValueError):
                    self._queue.remove(my_id)
                self._condition.notify_all()
