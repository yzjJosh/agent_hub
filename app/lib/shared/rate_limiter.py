"""Run calls one at a time, with a short wait between them."""

from __future__ import annotations

__all__ = ["RateLimiter"]

import asyncio
import contextlib
import math
import time


class RateLimiter:
    """Runs calls one at a time, with a short wait between them."""

    def __init__(self, min_delay: float) -> None:
        """Make a limiter.

        Args:
            min_delay: Seconds that must pass between two calls.
        """
        self._min_delay = min_delay
        self._queue: list[int] = []
        self._next_turn = 0
        self._condition = asyncio.Condition()
        self._last_call = -math.inf

    async def acquire(self) -> None:
        """Wait until it is this caller's turn to make a request."""
        async with self._condition:
            my_turn = self._next_turn
            self._next_turn += 1
            self._queue.append(my_turn)
            try:
                await self._condition.wait_for(lambda: self._queue[0] == my_turn)

                now = time.monotonic()
                remaining = self._min_delay - (now - self._last_call)
                if remaining > 0:
                    await asyncio.sleep(remaining)

                self._last_call = time.monotonic()
            finally:
                with contextlib.suppress(ValueError):
                    self._queue.remove(my_turn)
                self._condition.notify_all()
