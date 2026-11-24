"""Unit tests for replacing text inside the focused browser field."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.lib.browser.field_editing import replace_focused_field_text

_OLD_TEXT = "quick"
_NEW_TEXT = "slow"
_EXPECTED_TEXT = "the slow brown fox"


def _build_frame(
    evaluate_results: list[dict | str | None],
    *,
    child_frame: MagicMock | None = None,
    has_inner_frame_element: bool = False,
) -> MagicMock:
    """Create a fake frame with scripted JS evaluation and inner-frame lookup."""
    frame = MagicMock()
    element = None
    if has_inner_frame_element:
        element = MagicMock()
        element.content_frame = AsyncMock(return_value=child_frame)
    handle = MagicMock()
    handle.as_element = MagicMock(return_value=element)
    handle.dispose = AsyncMock()
    frame.evaluate_handle = AsyncMock(return_value=handle)
    frame.evaluate = AsyncMock(side_effect=evaluate_results)
    return frame


def _build_page(main_frame: MagicMock) -> MagicMock:
    """Create a fake page exposing main_frame and keyboard mocks."""
    page = MagicMock()
    page.main_frame = main_frame
    page.keyboard = MagicMock()
    page.keyboard.insert_text = AsyncMock()
    page.keyboard.press = AsyncMock()
    return page


class TestReplaceFocusedFieldText:
    """Tests for editing text inside the currently focused field."""

    async def test_replaces_selected_range_in_main_frame(self) -> None:
        """Inserting replacement text succeeds when the field reads back the expected text."""
        frame = _build_frame(
            [
                {"status": "ok", "expected": _EXPECTED_TEXT, "selectedAll": False},
                _EXPECTED_TEXT,
            ]
        )
        page = _build_page(frame)

        result = await replace_focused_field_text(page, _OLD_TEXT, _NEW_TEXT)

        assert result is True
        page.keyboard.insert_text.assert_awaited_once_with(_NEW_TEXT)
        page.keyboard.press.assert_not_awaited()

    async def test_inserts_full_expected_text_when_input_selects_all(self) -> None:
        """Fields that only support whole-value selection receive the complete expected value."""
        frame = _build_frame(
            [
                {"status": "ok", "expected": "jane@example.com", "selectedAll": True},
                "jane@example.com",
            ]
        )
        page = _build_page(frame)

        result = await replace_focused_field_text(page, "exmaple", "example")

        assert result is True
        page.keyboard.insert_text.assert_awaited_once_with("jane@example.com")

    async def test_presses_delete_when_replacement_removes_selected_text(self) -> None:
        """An empty replacement deletes the selected range with the Delete key."""
        frame = _build_frame(
            [
                {"status": "ok", "expected": "the  brown fox", "selectedAll": False},
                "the  brown fox",
            ]
        )
        page = _build_page(frame)

        result = await replace_focused_field_text(page, _OLD_TEXT, "")

        assert result is True
        page.keyboard.press.assert_awaited_once_with("Delete")
        page.keyboard.insert_text.assert_not_awaited()

    async def test_walks_into_focused_nested_iframe(self) -> None:
        """When focus is inside an iframe, text replacement runs on the inner frame."""
        inner_frame = _build_frame(
            [
                {"status": "ok", "expected": _EXPECTED_TEXT, "selectedAll": False},
                _EXPECTED_TEXT,
            ]
        )
        main_frame = _build_frame([], child_frame=inner_frame, has_inner_frame_element=True)
        page = _build_page(main_frame)

        result = await replace_focused_field_text(page, _OLD_TEXT, _NEW_TEXT)

        assert result is True
        inner_frame.evaluate.assert_awaited()
        main_frame.evaluate.assert_not_awaited()

    async def test_returns_false_when_inner_frame_cannot_be_accessed(self) -> None:
        """An unreachable child frame fails cleanly without typing."""
        main_frame = _build_frame([], child_frame=None, has_inner_frame_element=True)
        page = _build_page(main_frame)

        result = await replace_focused_field_text(page, _OLD_TEXT, _NEW_TEXT)

        assert result is False
        page.keyboard.insert_text.assert_not_awaited()

    @pytest.mark.parametrize(
        "status",
        ["not an editable field", "old text not found", "old text found more than once"],
    )
    async def test_returns_false_when_selection_script_reports_error(self, status: str) -> None:
        """Selection errors prevent keyboard input and return False."""
        frame = _build_frame([{"status": status}])
        page = _build_page(frame)

        result = await replace_focused_field_text(page, _OLD_TEXT, _NEW_TEXT)

        assert result is False
        page.keyboard.insert_text.assert_not_awaited()

    async def test_returns_false_when_final_field_text_does_not_match_expected(self) -> None:
        """A post-edit read mismatch returns False."""
        frame = _build_frame(
            [
                {"status": "ok", "expected": _EXPECTED_TEXT, "selectedAll": False},
                "the quick brown fox",
            ]
        )
        page = _build_page(frame)

        result = await replace_focused_field_text(page, _OLD_TEXT, _NEW_TEXT)

        assert result is False

    async def test_returns_false_when_page_evaluation_raises(self) -> None:
        """Unexpected Playwright exceptions return False."""
        frame = _build_frame([])
        frame.evaluate_handle = AsyncMock(side_effect=RuntimeError("frame detached"))
        page = _build_page(frame)

        result = await replace_focused_field_text(page, _OLD_TEXT, _NEW_TEXT)

        assert result is False
