"""Tests for applying server-side filters in the Redfin Filters panel."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from app.lib.redfin.filters import apply_server_side_filters
from app.lib.redfin.models import RedfinSearchError, SearchOptions
from tests.app.lib.redfin.conftest import make_fake_snapshot

_CLICK_TIMEOUT_MS = 10000
_FILTER_TREE = {
    "role": "WebArea",
    "children": [
        {"role": "label", "name": "Beds"},
        {"role": "button", "name": "3"},
        {"role": "button", "name": "5+"},
        {"role": "label", "name": "Baths"},
        {"role": "button", "name": "2+"},
    ],
}


def _make_browser(snapshots: list[MagicMock | None]) -> MagicMock:
    """Create a fake browser that returns snapshots in order."""
    browser = MagicMock()
    browser.capture_snapshot = AsyncMock(side_effect=snapshots)
    return browser


@pytest.mark.fast_sleeps
async def test_apply_server_side_filters_sets_all_sold_and_room_filters() -> None:
    """Sold, price, time, and beds or baths filters are applied and the panel is closed."""
    open_snapshot = make_fake_snapshot(
        [{"ref_id": "filters", "role": "button", "name": "Filters"}]
    )
    panel_snapshot = make_fake_snapshot(
        [
            {"ref_id": "sold-radio", "role": "radio", "name": "Sold"},
            {"ref_id": "min-price", "role": "textbox", "name": "Enter min price"},
            {"ref_id": "max-price", "role": "textbox", "name": "Enter max price"},
            {"ref_id": "sold-time", "role": "combobox", "name": "Sold within"},
            {"ref_id": "beds-5plus", "role": "button", "name": "5+"},
            {"ref_id": "baths-2plus", "role": "button", "name": "2+"},
        ],
        _FILTER_TREE,
    )
    browser = _make_browser([open_snapshot, *([panel_snapshot] * 9)])
    options = SearchOptions(
        is_sold=True,
        min_price=500000,
        max_price=900000,
        date_from="2026-06-01",
        min_beds=5,
        min_baths=2,
    )

    with patch("app.lib.redfin.filters.datetime", wraps=datetime) as mock_datetime:
        mock_datetime.now.return_value = datetime(2026, 6, 20)
        await apply_server_side_filters(browser, "tab-1", options)

    assert panel_snapshot.click.await_args_list == [
        call("sold-radio", timeout=_CLICK_TIMEOUT_MS, verbose=True),
        call("beds-5plus", timeout=_CLICK_TIMEOUT_MS, verbose=True),
        call("baths-2plus", timeout=_CLICK_TIMEOUT_MS, verbose=True),
    ]
    assert panel_snapshot.fill.await_args_list == [
        call("min-price", "500000", submit=True, verbose=True),
        call("max-price", "900000", submit=True, verbose=True),
    ]
    panel_snapshot.select_option.assert_awaited_once_with(
        "sold-time", value="Last 1 month", verbose=True
    )
    panel_snapshot.press_key.assert_awaited_once_with("Escape", verbose=True)


@pytest.mark.fast_sleeps
async def test_apply_server_side_filters_allows_active_date_beyond_30_days() -> None:
    """An active search older than 30 days skips the time filter without raising."""
    open_snapshot = make_fake_snapshot(
        [{"ref_id": "filters", "role": "button", "name": "Filters"}]
    )
    panel_snapshot = make_fake_snapshot(
        [{"ref_id": "dom-combobox", "role": "combobox", "name": "dom"}]
    )
    browser = _make_browser([open_snapshot, panel_snapshot, panel_snapshot])
    options = SearchOptions(is_sold=False, date_from="2026-01-01")

    with patch("app.lib.redfin.filters.datetime", wraps=datetime) as mock_datetime:
        mock_datetime.now.return_value = datetime(2026, 6, 20)
        await apply_server_side_filters(browser, "tab-1", options)

    panel_snapshot.select_option.assert_not_awaited()
    panel_snapshot.press_key.assert_awaited_once_with("Escape", verbose=True)


@pytest.mark.fast_sleeps
async def test_apply_server_side_filters_raises_when_panel_cannot_open() -> None:
    """Missing Filters button causes apply_server_side_filters to raise."""
    browser = _make_browser([make_fake_snapshot([])])

    with pytest.raises(RedfinSearchError, match="could not open the Filters panel"):
        await apply_server_side_filters(browser, "tab-1", SearchOptions(is_sold=True))


@pytest.mark.fast_sleeps
async def test_apply_server_side_filters_raises_when_required_filter_fails() -> None:
    """Failure to find the Sold radio raises RedfinSearchError after retries."""
    open_snapshot = make_fake_snapshot(
        [{"ref_id": "filters", "role": "button", "name": "Filters"}]
    )
    empty_panel = make_fake_snapshot([])
    browser = _make_browser(
        [open_snapshot, empty_panel, empty_panel, empty_panel, empty_panel]
    )

    with pytest.raises(RedfinSearchError, match="could not apply sold filter"):
        await apply_server_side_filters(browser, "tab-1", SearchOptions(is_sold=True))
