"""Unit tests for browser page scrolling helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.lib.browser.browser_manager import Viewport
from app.lib.browser.scrolling import (
    scroll_by_pixels,
    scroll_selector_into_view,
    scroll_to_coordinates,
)

_VIEWPORT_WIDTH = 1000.0
_VIEWPORT_HEIGHT = 800.0
_CONTENT_WIDTH = 2000.0
_CONTENT_HEIGHT = 4000.0
_WHEEL_DISTANCE = 250.0
_SIDEWAYS_DISTANCE = 120.0
_LARGE_DISTANCE = 100000.0
_TARGET_X = 200.0
_TARGET_VISIBLE_Y = 300.0
_TARGET_BELOW_FOLD_Y = 1500.0
_ELEMENT_LEFT = 100.0
_ELEMENT_VISIBLE_TOP = 100.0
_ELEMENT_SCROLLED_TOP = 400.0
_ELEMENT_OFFSCREEN_TOP = 1100.0
_ELEMENT_PINNED_TOP = 1500.0
_ELEMENT_WIDTH = 80.0
_ELEMENT_HEIGHT = 40.0
_SCROLLED_PAGE_Y = 1200.0
_EXPECTED_STAGNANT_STEPS = 5


def _build_viewport(
    page_x: float = 0.0,
    page_y: float = 0.0,
    content_width: float = _CONTENT_WIDTH,
    content_height: float = _CONTENT_HEIGHT,
) -> Viewport:
    """Create a viewport snapshot at the given scroll offset."""
    return Viewport(
        client_width=_VIEWPORT_WIDTH,
        client_height=_VIEWPORT_HEIGHT,
        page_x=page_x,
        page_y=page_y,
        content_width=content_width,
        content_height=content_height,
        window_width=_VIEWPORT_WIDTH,
        window_height=_VIEWPORT_HEIGHT,
    )


def _build_element_box(top: float) -> dict[str, float]:
    """Build a bounding box dictionary at the given vertical offset."""
    return {
        "x": _ELEMENT_LEFT,
        "y": top,
        "width": _ELEMENT_WIDTH,
        "height": _ELEMENT_HEIGHT,
    }


def _build_page_with_locator(bounds_sequence: list[dict | None], count: int = 1) -> MagicMock:
    """Create a fake page whose mouse wheel succeeds and locator returns given boxes."""
    page = MagicMock()
    page.mouse = MagicMock()
    page.mouse.wheel = AsyncMock()
    locator = MagicMock()
    locator.count = AsyncMock(return_value=count)
    locator.bounding_box = AsyncMock(side_effect=bounds_sequence)
    page.locator = MagicMock(return_value=MagicMock(first=locator))
    return page


class TestScrollByPixels:
    """Tests for scrolling a page by an explicit pixel delta."""

    async def test_scrolls_vertical_and_horizontal_deltas(self) -> None:
        """Both horizontal and vertical wheel steps sum to the requested distances."""
        page = _build_page_with_locator([])

        result = await scroll_by_pixels(page, _WHEEL_DISTANCE, dx=_SIDEWAYS_DISTANCE)

        assert result is True
        wheel_calls = page.mouse.wheel.await_args_list
        total_dx = sum(call.args[0] for call in wheel_calls)
        total_dy = sum(call.args[1] for call in wheel_calls)
        assert total_dx == _SIDEWAYS_DISTANCE
        assert total_dy == _WHEEL_DISTANCE

    async def test_returns_false_when_wheel_raises(self) -> None:
        """A wheel error causes scroll_by_pixels to return False."""
        page = _build_page_with_locator([])
        page.mouse.wheel = AsyncMock(side_effect=RuntimeError("wheel failed"))

        result = await scroll_by_pixels(page, _WHEEL_DISTANCE)

        assert result is False

    async def test_propagates_cancellation(self) -> None:
        """Cancelling a scroll cancels the pending wheel future and re-raises CancelledError."""
        page = _build_page_with_locator([])
        wheel_entered = asyncio.Event()
        pending_wheel: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        async def _block_wheel(_dx: float, _dy: float) -> None:
            wheel_entered.set()
            await pending_wheel

        page.mouse.wheel = AsyncMock(side_effect=_block_wheel)
        scroll_task = asyncio.create_task(scroll_by_pixels(page, _LARGE_DISTANCE))
        await wheel_entered.wait()
        scroll_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await scroll_task
        assert pending_wheel.cancelled()


class TestScrollToCoordinates:
    """Tests for scrolling document coordinates into the visible viewport."""

    @pytest.mark.parametrize(("x", "y"), [(-1.0, _ELEMENT_VISIBLE_TOP), (_ELEMENT_LEFT, -5.0)])
    async def test_rejects_negative_coordinates(self, x: float, y: float) -> None:
        """Negative document coordinates fail immediately without reading the viewport."""
        page = _build_page_with_locator([])
        read_viewport = AsyncMock(return_value=_build_viewport())
        move_mouse = AsyncMock(return_value=True)

        result = await scroll_to_coordinates(page, x, y, read_viewport, move_mouse)

        assert result is False
        read_viewport.assert_not_awaited()

    async def test_succeeds_when_point_is_already_visible_and_settled(self) -> None:
        """A point inside the viewport returns True once consecutive reads stay still."""
        page = _build_page_with_locator([])
        read_viewport = AsyncMock(return_value=_build_viewport(page_x=0.0, page_y=0.0))
        move_mouse = AsyncMock(return_value=True)

        result = await scroll_to_coordinates(page, _TARGET_X, _TARGET_VISIBLE_Y, read_viewport, move_mouse)

        assert result is True
        page.mouse.wheel.assert_not_awaited()

    async def test_scrolls_until_point_enters_viewport(self) -> None:
        """Wheel steps run until the target point lands inside the visible area."""
        page = _build_page_with_locator([])
        scrolled = _build_viewport(page_x=0.0, page_y=_SCROLLED_PAGE_Y)
        read_viewport = AsyncMock(
            side_effect=[_build_viewport(page_x=0.0, page_y=0.0), scrolled, scrolled, scrolled, scrolled]
        )
        move_mouse = AsyncMock(return_value=True)

        result = await scroll_to_coordinates(page, _TARGET_X, _TARGET_BELOW_FOLD_Y, read_viewport, move_mouse)

        assert result is True
        assert page.mouse.wheel.await_count >= 1
        assert move_mouse.await_count == 1

    async def test_returns_false_when_viewport_cannot_be_read(self) -> None:
        """A missing viewport stops scrolling and returns False."""
        page = _build_page_with_locator([])
        read_viewport = AsyncMock(return_value=None)
        move_mouse = AsyncMock(return_value=True)

        result = await scroll_to_coordinates(page, _TARGET_X, _TARGET_VISIBLE_Y, read_viewport, move_mouse)

        assert result is False

    async def test_returns_false_when_target_is_beyond_scrollable_content(self) -> None:
        """A point past the bottom of a non-scrollable page fails immediately."""
        page = _build_page_with_locator([])
        fixed_viewport = _build_viewport(
            page_x=0.0,
            page_y=0.0,
            content_width=_VIEWPORT_WIDTH,
            content_height=_VIEWPORT_HEIGHT,
        )
        read_viewport = AsyncMock(return_value=fixed_viewport)
        move_mouse = AsyncMock(return_value=True)

        result = await scroll_to_coordinates(page, _TARGET_X, _TARGET_BELOW_FOLD_Y, read_viewport, move_mouse)

        assert result is False
        page.mouse.wheel.assert_not_awaited()

    async def test_returns_false_when_scroll_position_never_changes(self) -> None:
        """Repeated scroll steps that never move the viewport stop as stagnant."""
        page = _build_page_with_locator([])
        stuck_viewport = _build_viewport(page_x=0.0, page_y=0.0)
        read_viewport = AsyncMock(return_value=stuck_viewport)
        move_mouse = AsyncMock(return_value=True)

        result = await scroll_to_coordinates(page, _TARGET_X, _TARGET_BELOW_FOLD_Y, read_viewport, move_mouse)

        assert result is False
        assert move_mouse.await_count == _EXPECTED_STAGNANT_STEPS


class TestScrollSelectorIntoView:
    """Tests for scrolling a matching element into the viewport."""

    async def test_returns_false_when_element_not_found(self) -> None:
        """A selector matching zero elements returns False."""
        page = _build_page_with_locator([], count=0)
        read_viewport = AsyncMock(return_value=_build_viewport())
        move_mouse = AsyncMock(return_value=True)

        result = await scroll_selector_into_view(page, "#missing", read_viewport, move_mouse)

        assert result is False
        read_viewport.assert_not_awaited()

    async def test_returns_true_immediately_when_center_is_in_viewport(self) -> None:
        """An element whose center is already inside the viewport needs no wheel input."""
        page = _build_page_with_locator([_build_element_box(_ELEMENT_VISIBLE_TOP)])
        read_viewport = AsyncMock(return_value=_build_viewport())
        move_mouse = AsyncMock(return_value=True)

        result = await scroll_selector_into_view(page, "#visible", read_viewport, move_mouse)

        assert result is True
        page.mouse.wheel.assert_not_awaited()

    async def test_scrolls_element_into_viewport(self) -> None:
        """An offscreen element moves into the viewport after wheel steps."""
        page = _build_page_with_locator(
            [
                _build_element_box(_ELEMENT_OFFSCREEN_TOP),
                _build_element_box(_ELEMENT_SCROLLED_TOP),
            ]
        )
        read_viewport = AsyncMock(return_value=_build_viewport())
        move_mouse = AsyncMock(return_value=True)

        result = await scroll_selector_into_view(page, "#below", read_viewport, move_mouse)

        assert result is True
        assert move_mouse.await_count == 1
        assert page.mouse.wheel.await_count >= 1

    async def test_returns_false_when_element_is_pinned(self) -> None:
        """An element whose bounding box never moves stops after five stagnant steps."""
        pinned_box = _build_element_box(_ELEMENT_PINNED_TOP)
        page = _build_page_with_locator([pinned_box] * (_EXPECTED_STAGNANT_STEPS * 2))
        read_viewport = AsyncMock(return_value=_build_viewport())
        move_mouse = AsyncMock(return_value=True)

        result = await scroll_selector_into_view(page, "#pinned", read_viewport, move_mouse)

        assert result is False
        assert move_mouse.await_count == _EXPECTED_STAGNANT_STEPS
