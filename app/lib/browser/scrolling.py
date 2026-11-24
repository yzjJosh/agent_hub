"""Scroll browser pages toward coordinates, elements, or pixel offsets."""

from __future__ import annotations

__all__ = ["scroll_by_pixels", "scroll_selector_into_view", "scroll_to_coordinates"]

import asyncio
import random
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from playwright.async_api import Locator, Page

if TYPE_CHECKING:
    from app.lib.browser.browser_manager import Viewport

_SCROLL_MAX_ATTEMPTS = 120
_SCROLL_VIEW_MAX_ITERATIONS = 30
_SCROLL_STEP_MAX_PX = 300
_SCROLL_STAGNANT_LIMIT = 5
_SCROLL_MARGIN_PX = 10
_SCROLL_DELTA_TOLERANCE_PX = 10
_WHEEL_STEP_MIN_PX = 100
_WHEEL_STEP_MAX_PX = 300
_WHEEL_DELAY_MIN_S = 0.03
_WHEEL_DELAY_MAX_S = 0.08
_WHEEL_TIMEOUT_S = 5
_SCROLL_TIMEOUT_S = 30
_SCROLL_SETTLE_TIMEOUT_S = 2.0
_SCROLL_SETTLE_INTERVAL_S = 0.05
_ELEMENT_BOUNDS_TIMEOUT_MS = 5000
_MIN_ELEMENT_MOVE_PX = 1


class _ScrollStatus(Enum):
    """Outcome of a single scroll step toward target coordinates."""

    REACHED = "reached"
    CONTINUE = "continue"
    FAILED = "failed"
    STAGNANT = "stagnant"


@dataclass(frozen=True)
class _ScrollStepResult:
    """Result of one scroll step and the updated stagnant counter."""

    status: _ScrollStatus
    stagnant_count: int


@dataclass(frozen=True)
class _ScrollDelta:
    """Horizontal and vertical scroll distance in pixels."""

    dx: float
    dy: float


@dataclass(frozen=True)
class _Point:
    """Two-dimensional coordinates in pixels."""

    x: float
    y: float


async def _await_scroll_call[T](call: Awaitable[T], label: str, timeout: float) -> T:
    """Wait for one scroll operation with a hard deadline."""
    deadline = asyncio.timeout(timeout)
    try:
        async with deadline:
            return await call
    except TimeoutError:
        if not deadline.expired():
            raise
        message = f"{label} timed out after {timeout}s"
        print(f"[scrolling._await_scroll_call] Error: {message}", file=sys.stderr)
        raise TimeoutError(message) from None


def _calculate_scroll_delta(viewport: Viewport, target_x: float, target_y: float) -> _ScrollDelta:
    """Compute how far to scroll to center a page point in the viewport."""
    target_scroll_x = max(0.0, min(target_x - viewport.client_width / 2, viewport.max_scroll_x))
    target_scroll_y = max(0.0, min(target_y - viewport.client_height / 2, viewport.max_scroll_y))
    return _ScrollDelta(dx=target_scroll_x - viewport.page_x, dy=target_scroll_y - viewport.page_y)


def _is_point_in_viewport(
    point_x: float,
    point_y: float,
    width: float,
    height: float,
    *,
    is_inclusive: bool = False,
) -> bool:
    """Check whether viewport-relative coordinates lie within the visible box."""
    if is_inclusive:
        return 0 <= point_x <= width and 0 <= point_y <= height
    return 0 <= point_x < width and 0 <= point_y < height


async def _send_wheel_steps(page: Page, dx: float, dy: float, verbose: bool = False) -> None:
    """Dispatch wheel events in small random increments across both axes."""
    for axis, total in (("x", dx), ("y", dy)):
        if total == 0:
            continue
        remaining = abs(total)
        while remaining > 0:
            step = min(remaining, random.randint(_WHEEL_STEP_MIN_PX, _WHEEL_STEP_MAX_PX))
            signed_step = step if total > 0 else -step
            wheel_x = signed_step if axis == "x" else 0
            wheel_y = signed_step if axis == "y" else 0
            await _await_scroll_call(page.mouse.wheel(wheel_x, wheel_y), "Wheel", timeout=_WHEEL_TIMEOUT_S)
            await asyncio.sleep(random.uniform(_WHEEL_DELAY_MIN_S, _WHEEL_DELAY_MAX_S))
            remaining -= step
        if verbose:
            print(f"[scrolling._send_wheel_steps] Scrolled {axis.upper()} by {total}", flush=True)


async def _scroll_with_wheel(page: Page, dx: float, dy: float, verbose: bool = False) -> None:
    """Scroll by a pixel delta within the total scroll deadline."""
    try:
        await _await_scroll_call(_send_wheel_steps(page, dx, dy, verbose), "Scroll", timeout=_SCROLL_TIMEOUT_S)
    except TimeoutError as exc:
        raise TimeoutError(f"{exc}. Scroll may be partial.") from None


async def _wait_until_scroll_settled(read_viewport: Callable[[], Awaitable[Viewport | None]]) -> bool:
    """Wait until two consecutive viewport reads report the same scroll position."""
    deadline = time.monotonic() + _SCROLL_SETTLE_TIMEOUT_S
    previous_position: _Point | None = None
    while time.monotonic() < deadline:
        await asyncio.sleep(_SCROLL_SETTLE_INTERVAL_S)
        viewport = await read_viewport()
        if viewport is None:
            continue
        current_position = _Point(viewport.page_x, viewport.page_y)
        if previous_position is not None and previous_position == current_position:
            return True
        previous_position = current_position
    return False


async def _evaluate_scroll_progress(
    read_viewport: Callable[[], Awaitable[Viewport | None]],
    before_position: _Point,
    stagnant_count: int,
) -> _ScrollStepResult:
    """Compare the scroll position after a wheel step against the previous position."""
    after_viewport = await read_viewport()
    after_position = _Point(after_viewport.page_x, after_viewport.page_y) if after_viewport is not None else before_position
    if after_position != before_position:
        return _ScrollStepResult(_ScrollStatus.CONTINUE, 0)
    next_stagnant_count = stagnant_count + 1
    if next_stagnant_count >= _SCROLL_STAGNANT_LIMIT:
        print(
            f"[scrolling._evaluate_scroll_progress] Error: viewport unchanged after {_SCROLL_STAGNANT_LIMIT} scroll steps "
            f"at ({before_position.x}, {before_position.y}). Target may be in a nested scroll container.",
            file=sys.stderr,
        )
        return _ScrollStepResult(_ScrollStatus.STAGNANT, next_stagnant_count)
    return _ScrollStepResult(_ScrollStatus.CONTINUE, next_stagnant_count)


async def _run_scroll_step(
    page: Page,
    target_x: float,
    target_y: float,
    read_viewport: Callable[[], Awaitable[Viewport | None]],
    move_mouse: Callable[[float, float], Awaitable[bool]],
    stagnant_count: int,
    verbose: bool,
) -> _ScrollStepResult:
    """Perform one scroll iteration toward document coordinates."""
    viewport = await read_viewport()
    if viewport is None:
        print("[scrolling._run_scroll_step] Error: could not read the viewport", file=sys.stderr)
        return _ScrollStepResult(_ScrollStatus.FAILED, stagnant_count)

    scroll_x, scroll_y = viewport.page_x, viewport.page_y
    viewport_x, viewport_y = target_x - scroll_x, target_y - scroll_y
    if _is_point_in_viewport(viewport_x, viewport_y, viewport.client_width, viewport.client_height):
        if await _wait_until_scroll_settled(read_viewport):
            return _ScrollStepResult(_ScrollStatus.REACHED, stagnant_count)
        print(
            f"[scrolling._run_scroll_step] Error: scroll did not settle within "
            f"{_SCROLL_SETTLE_TIMEOUT_S}s at ({scroll_x}, {scroll_y})",
            file=sys.stderr,
        )
        return _ScrollStepResult(_ScrollStatus.FAILED, stagnant_count)

    delta = _calculate_scroll_delta(viewport, target_x, target_y)
    if abs(delta.dx) < _SCROLL_DELTA_TOLERANCE_PX and abs(delta.dy) < _SCROLL_DELTA_TOLERANCE_PX:
        print(
            f"[scrolling._run_scroll_step] Error: element not in viewport "
            f"and cannot scroll further at ({scroll_x}, {scroll_y})",
            file=sys.stderr,
        )
        return _ScrollStepResult(_ScrollStatus.FAILED, stagnant_count)

    if verbose:
        print(f"[scrolling._run_scroll_step] Need to scroll by ({delta.dx}, {delta.dy})", flush=True)
    await move_mouse(viewport_x, viewport_y)
    await _scroll_with_wheel(page, delta.dx, delta.dy, verbose=verbose)
    return await _evaluate_scroll_progress(read_viewport, _Point(scroll_x, scroll_y), stagnant_count)


async def _read_element_center(locator: Locator) -> _Point | None:
    """Read the center point of an element's bounding box."""
    bounds = await locator.bounding_box(timeout=_ELEMENT_BOUNDS_TIMEOUT_MS)
    if bounds is None:
        return None
    return _Point(
        x=bounds["x"] + bounds["width"] / 2,
        y=bounds["y"] + bounds["height"] / 2,
    )


async def _step_selector_scroll(
    page: Page,
    locator: Locator,
    center: _Point,
    viewport_width: float,
    viewport_height: float,
    move_mouse: Callable[[float, float], Awaitable[bool]],
    verbose: bool,
) -> _Point | None:
    """Move the pointer near the element, scroll toward it, and re-read its center."""
    hover_x = max(_SCROLL_MARGIN_PX, min(center.x, viewport_width - _SCROLL_MARGIN_PX))
    hover_y = max(_SCROLL_MARGIN_PX, min(center.y, viewport_height - _SCROLL_MARGIN_PX))
    await move_mouse(hover_x, hover_y)
    delta_x = max(-_SCROLL_STEP_MAX_PX, min(center.x - viewport_width / 2, _SCROLL_STEP_MAX_PX))
    delta_y = max(-_SCROLL_STEP_MAX_PX, min(center.y - viewport_height / 2, _SCROLL_STEP_MAX_PX))
    if verbose:
        print(f"[scrolling._step_selector_scroll] Center ({center.x}, {center.y}) scrolling by ({delta_x}, {delta_y})", flush=True)
    await _scroll_with_wheel(page, delta_x, delta_y, verbose=verbose)
    return await _read_element_center(locator)


async def scroll_by_pixels(page: Page, distance: float, dx: float = 0.0, verbose: bool = False) -> bool:
    """Scroll a page by a given number of pixels.

    Args:
        page: The page to scroll.
        distance: Vertical distance in pixels, positive scrolls down.
        dx: Horizontal distance in pixels, positive scrolls right.
        verbose: Print progress lines when True.

    Returns:
        True if the full scroll completed.
    """
    if verbose:
        print(f"[scrolling.scroll_by_pixels] Scrolling by ({dx}, {distance})px", flush=True)
    try:
        await _scroll_with_wheel(page, dx, distance, verbose=verbose)
        if verbose:
            print("[scrolling.scroll_by_pixels] Success", flush=True)
        return True
    except Exception as exc:
        print(f"[scrolling.scroll_by_pixels] Error: {exc}", file=sys.stderr)
        return False


async def scroll_to_coordinates(
    page: Page,
    x: float,
    y: float,
    read_viewport: Callable[[], Awaitable[Viewport | None]],
    move_mouse: Callable[[float, float], Awaitable[bool]],
    verbose: bool = False,
) -> bool:
    """Scroll a page until document coordinates enter the viewport.

    Args:
        page: The page to scroll.
        x: Horizontal position from the left edge of the document.
        y: Vertical position from the top edge of the document.
        read_viewport: Async callback that returns the current viewport.
        move_mouse: Async callback that moves the pointer in viewport coordinates.
        verbose: Print progress lines when True.

    Returns:
        True if scrolling brought the point into view or finished its attempts.
    """
    if x < 0 or y < 0:
        print(f"[scrolling.scroll_to_coordinates] Error: negative coordinates ({x}, {y})", file=sys.stderr)
        return False
    if verbose:
        print(f"[scrolling.scroll_to_coordinates] Scrolling to document ({x}, {y})", flush=True)
    try:
        stagnant_count = 0
        for _ in range(_SCROLL_MAX_ATTEMPTS):
            step_result = await _run_scroll_step(page, x, y, read_viewport, move_mouse, stagnant_count, verbose)
            if step_result.status is _ScrollStatus.REACHED:
                if verbose:
                    status_label = "Already in viewport" if step_result.stagnant_count == 0 else "Success"
                    print(f"[scrolling.scroll_to_coordinates] {status_label}", flush=True)
                return True
            if step_result.status in (_ScrollStatus.FAILED, _ScrollStatus.STAGNANT):
                return False
            stagnant_count = step_result.stagnant_count
        if verbose:
            print("[scrolling.scroll_to_coordinates] Warning: reached max scroll attempts", flush=True)
        return True
    except Exception as exc:
        print(f"[scrolling.scroll_to_coordinates] Error: {exc}", file=sys.stderr)
        return False


async def scroll_selector_into_view(
    page: Page,
    selector: str,
    read_viewport: Callable[[], Awaitable[Viewport | None]],
    move_mouse: Callable[[float, float], Awaitable[bool]],
    verbose: bool = False,
) -> bool:
    """Scroll a page until the center of a matching element enters the viewport.

    Args:
        page: The page to scroll.
        selector: Selector for the element to bring into view.
        read_viewport: Async callback that returns the current viewport.
        move_mouse: Async callback that moves the pointer in viewport coordinates.
        verbose: Print progress lines when True.

    Returns:
        True if the element center is inside the viewport.
    """
    if verbose:
        print(f"[scrolling.scroll_selector_into_view] Scrolling to '{selector}'", flush=True)
    try:
        locator = page.locator(selector).first
        if not await locator.count():
            print("[scrolling.scroll_selector_into_view] Error: element not found", file=sys.stderr)
            return False
        center = await _read_element_center(locator)
        if center is None:
            print("[scrolling.scroll_selector_into_view] Error: could not get element bounds", file=sys.stderr)
            return False
        viewport = await read_viewport()
        if viewport is None:
            print("[scrolling.scroll_selector_into_view] Error: could not read the viewport", file=sys.stderr)
            return False
        return await _scroll_locator_loop(page, locator, center, viewport, move_mouse, verbose)
    except Exception as exc:
        print(f"[scrolling.scroll_selector_into_view] Error: {exc}", file=sys.stderr)
        return False


async def _scroll_locator_loop(
    page: Page,
    locator: Locator,
    center: _Point,
    viewport: Viewport,
    move_mouse: Callable[[float, float], Awaitable[bool]],
    verbose: bool,
) -> bool:
    """Repeatedly scroll toward an element center until visible or stuck."""
    width, height = viewport.client_width, viewport.client_height
    is_in_viewport = _is_point_in_viewport(center.x, center.y, width, height, is_inclusive=True)
    stagnant_count = 0
    for _ in range(_SCROLL_VIEW_MAX_ITERATIONS):
        if is_in_viewport:
            return True
        next_center = await _step_selector_scroll(page, locator, center, width, height, move_mouse, verbose)
        if next_center is None:
            stagnant_count += 1
        else:
            has_moved = (
                abs(next_center.x - center.x) >= _MIN_ELEMENT_MOVE_PX
                or abs(next_center.y - center.y) >= _MIN_ELEMENT_MOVE_PX
            )
            stagnant_count = 0 if has_moved else stagnant_count + 1
            center = next_center
            is_in_viewport = _is_point_in_viewport(center.x, center.y, width, height, is_inclusive=True)
        if stagnant_count >= _SCROLL_STAGNANT_LIMIT:
            print(
                f"[scrolling.scroll_selector_into_view] Error: element stopped moving after "
                f"{stagnant_count} scroll steps at ({center.x}, {center.y}), it may be pinned in a "
                f"non-scrollable container",
                file=sys.stderr,
            )
            return False
    if not is_in_viewport:
        print(
            f"[scrolling.scroll_selector_into_view] Error: element still out of viewport "
            f"after {_SCROLL_VIEW_MAX_ITERATIONS} scroll steps at ({center.x}, {center.y})",
            file=sys.stderr,
        )
    return is_in_viewport
