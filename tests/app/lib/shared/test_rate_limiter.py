"""Tests for app.lib.shared.rate_limiter module."""

import asyncio
import contextlib
import time

from app.lib.shared.rate_limiter import RateLimiter

_DELAY = 0.2
_SLACK = 0.02
_DEADLOCK_TIMEOUT = 5.0


class TestRateLimiter:
    """Tests for spacing calls apart."""

    async def test_first_call_does_not_wait(self) -> None:
        """The first call goes right away."""
        limiter = RateLimiter(_DELAY)
        start = time.monotonic()
        await limiter.acquire()
        assert time.monotonic() - start < _DELAY

    async def test_back_to_back_calls_wait(self) -> None:
        """A second call waits until the delay has passed since the first."""
        limiter = RateLimiter(_DELAY)
        await limiter.acquire()
        start = time.monotonic()
        await limiter.acquire()
        assert time.monotonic() - start >= _DELAY - _SLACK

    async def test_call_after_delay_does_not_wait(self) -> None:
        """A call made after the delay has passed goes right away."""
        limiter = RateLimiter(_DELAY)
        await limiter.acquire()
        await asyncio.sleep(_DELAY)
        start = time.monotonic()
        await limiter.acquire()
        assert time.monotonic() - start < _DELAY / 2

    async def test_concurrent_calls_are_spaced(self) -> None:
        """Calls made at the same time each go at least the delay apart."""
        limiter = RateLimiter(_DELAY)
        times: list[float] = []

        async def _call() -> None:
            await limiter.acquire()
            times.append(time.monotonic())

        await asyncio.gather(_call(), _call(), _call())
        gaps = [later - earlier for earlier, later in zip(times, times[1:])]
        assert len(times) == 3
        assert all(gap >= _DELAY - _SLACK for gap in gaps)

    async def test_cancelled_waiter_does_not_block_others(self) -> None:
        """A call cancelled while it waits lets later calls go on."""
        limiter = RateLimiter(_DELAY)
        await limiter.acquire()
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(_DELAY / 4)
        waiter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waiter
        assert waiter.cancelled()

        await asyncio.wait_for(limiter.acquire(), timeout=_DEADLOCK_TIMEOUT)
