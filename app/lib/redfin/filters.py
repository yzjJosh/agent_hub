"""Apply server-side search filters in the Redfin Filters panel."""

from __future__ import annotations

__all__ = ["apply_server_side_filters"]

import asyncio
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from app.lib.browser.browser_manager import BrowserManager
from app.lib.browser.page_snapshot import PageSnapshot
from app.lib.redfin.models import RedfinSearchError, SearchOptions, parse_iso_date
from app.lib.redfin.snapshot_tree import (
    find_actionable_elements,
    find_label_sibling_buttons,
)

_VERBOSE = True
_CLICK_TIMEOUT_MS = 10000
_TIME_FILTER_SELECT_TIMEOUT_MS = 10000
_SNAPSHOT_TIMEOUT_MS = 2000
_BROWSER_OP_RETRY_COUNT = 3
_RETRY_BACKOFF_S = 1
_BEDS_OPTION_THRESHOLD = 5
_DAYS_ON_MARKET_COMBOBOX_NAME = "dom"
_DAYS_ON_MARKET_BUCKETS = (3, 7, 14, 30)
_SOLD_WITHIN_BUCKETS = (
    (7, "Last 1 week"),
    (30, "Last 1 month"),
    (90, "Last 3 months"),
    (180, "Last 6 months"),
    (365, "Last 1 year"),
    (730, "Last 2 years"),
    (1095, "Last 3 years"),
)
_SOLD_WITHIN_FALLBACK = "Last 5 years"


@dataclass(frozen=True)
class _FilterOptionSpec:
    label: str
    min_value: int | None
    format_option: Callable[[int], str]


async def _capture_filter_snapshot(
    browser: BrowserManager, target_id: str
) -> PageSnapshot | None:
    """Capture a snapshot of the page while working in the Filters panel."""
    return await browser.capture_snapshot(
        target_id, timeout=_SNAPSHOT_TIMEOUT_MS, verbose=_VERBOSE
    )


async def _open_filters_panel(
    browser: BrowserManager, target_id: str
) -> PageSnapshot | None:
    """Open the Redfin Filters panel and return the updated snapshot."""
    snapshot = await _capture_filter_snapshot(browser, target_id)
    if snapshot is None:
        return None
    filters_buttons = find_actionable_elements(
        snapshot, role="button", name_contains="Filters"
    )
    if not filters_buttons:
        print("[Redfin] Error: Filters button not found", file=sys.stderr, flush=True)
        return None
    if not await snapshot.click(
        filters_buttons[0]["ref_id"],
        timeout=_TIME_FILTER_SELECT_TIMEOUT_MS,
        verbose=_VERBOSE,
    ):
        return None
    return await _capture_filter_snapshot(browser, target_id)


def _select_days_on_market_option(date_from: str) -> str | None:
    """Pick the Redfin days on market option for an active listing start date."""
    from_date = parse_iso_date(date_from)
    if from_date is None:
        raise RedfinSearchError(f"invalid date_from: {date_from}")

    days = max(1, (datetime.now() - from_date).days)
    for bucket in _DAYS_ON_MARKET_BUCKETS:
        if days <= bucket:
            return f"{bucket}-"
    return None


def _select_sold_within_option(date_from: str) -> str:
    """Pick the Redfin sold within label for a sold search start date."""
    from_date = parse_iso_date(date_from)
    if from_date is None:
        raise RedfinSearchError(f"invalid date_from: {date_from}")

    days_ago = (datetime.now() - from_date).days
    for max_days, label in _SOLD_WITHIN_BUCKETS:
        if days_ago <= max_days:
            return label
    return _SOLD_WITHIN_FALLBACK


def _find_time_filter(snapshot: PageSnapshot, is_sold: bool) -> dict | None:
    """Find the time filter box inside the open Filters panel."""
    combobox_name = "sold" if is_sold else _DAYS_ON_MARKET_COMBOBOX_NAME
    matches = find_actionable_elements(
        snapshot, role="combobox", name_contains=combobox_name
    )
    return matches[0] if matches else None


async def _apply_time_filter(
    browser: BrowserManager,
    target_id: str,
    date_from: str,
    is_sold: bool,
) -> bool:
    """Set the time filter in the open Filters panel."""
    if is_sold:
        select_value = _select_sold_within_option(date_from)
    else:
        days_option = _select_days_on_market_option(date_from)
        if not days_option:
            print(
                "[Redfin] Warning: date-from > 30 days for active listings. "
                "Redfin only supports 'Less than 30 days' max. "
                "Falling back to client-side filtering.",
                file=sys.stderr,
                flush=True,
            )
            return False
        select_value = days_option

    snapshot = await _capture_filter_snapshot(browser, target_id)
    if snapshot is None:
        return False
    time_filter = _find_time_filter(snapshot, is_sold)
    if not time_filter:
        print("[Redfin] Error: Time filter not found", file=sys.stderr, flush=True)
        return False

    try:
        return await snapshot.select_option(
            time_filter["ref_id"], value=select_value, verbose=_VERBOSE
        )
    except Exception as exc:
        print(
            f"[Redfin] Error applying time filter: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return False


async def _apply_price_filter(
    browser: BrowserManager,
    target_id: str,
    min_price: int | None,
    max_price: int | None,
) -> bool:
    """Set the minimum and maximum price boxes in the open Filters panel."""
    snapshot = await _capture_filter_snapshot(browser, target_id)
    if snapshot is None:
        return False
    is_applied = False
    if min_price:
        min_inputs = find_actionable_elements(
            snapshot, role="textbox", name_contains="Enter min"
        )
        if min_inputs:
            await snapshot.fill(
                min_inputs[0]["ref_id"], str(min_price), submit=True, verbose=_VERBOSE
            )
            is_applied = True

    if max_price:
        snapshot = await _capture_filter_snapshot(browser, target_id)
        if snapshot is not None:
            max_inputs = find_actionable_elements(
                snapshot, role="textbox", name_contains="Enter max"
            )
            if max_inputs:
                await snapshot.fill(
                    max_inputs[0]["ref_id"], str(max_price), submit=True, verbose=_VERBOSE
                )
                is_applied = True

    if not is_applied:
        print(
            "[Redfin] Error: Price input fields not found", file=sys.stderr, flush=True
        )
        return False
    return True


async def _click_filter_option(
    browser: BrowserManager,
    target_id: str,
    tree: dict,
    label: str,
    option: str,
) -> bool:
    """Click one option button under a label in the open Filters panel."""
    button_names = find_label_sibling_buttons(tree, label)
    if not button_names:
        print(
            f"[Redfin] Error: No buttons found near '{label}' label",
            file=sys.stderr,
            flush=True,
        )
        return False
    if option not in button_names:
        print(
            f"[Redfin] Error: {label} option '{option}' not found in {button_names}",
            file=sys.stderr,
            flush=True,
        )
        return False

    click_snapshot = await _capture_filter_snapshot(browser, target_id)
    if click_snapshot is None:
        return False
    name_to_ref = {
        element["name"]: element["ref_id"]
        for element in click_snapshot.actionable_elements()
        if element.get("role") == "button"
    }
    return await click_snapshot.click(
        name_to_ref[option], timeout=_CLICK_TIMEOUT_MS, verbose=_VERBOSE
    )


async def _apply_beds_baths_filter(
    browser: BrowserManager,
    target_id: str,
    min_beds: int | None,
    min_baths: int | None,
) -> bool:
    """Set the beds and baths filters in the open Filters panel."""
    snapshot = await _capture_filter_snapshot(browser, target_id)
    if snapshot is None:
        return False
    tree = await snapshot.filtered_tree()
    if not isinstance(tree, dict) or not tree:
        print("[Redfin] Error: No filtered tree available", file=sys.stderr, flush=True)
        return False

    specs = (
        _FilterOptionSpec(
            label="Beds",
            min_value=min_beds,
            format_option=lambda count: (
                str(count) if count < _BEDS_OPTION_THRESHOLD else f"{count}+"
            ),
        ),
        _FilterOptionSpec(
            label="Baths",
            min_value=min_baths,
            format_option=lambda count: f"{count}+",
        ),
    )
    is_applied = False
    for spec in specs:
        if not spec.min_value:
            continue
        if not await _click_filter_option(
            browser, target_id, tree, spec.label, spec.format_option(spec.min_value)
        ):
            return False
        is_applied = True

    if not is_applied:
        print(
            "[Redfin] Error: No beds/baths options applied", file=sys.stderr, flush=True
        )
        return False
    return True


async def _apply_sold_filter(browser: BrowserManager, target_id: str) -> bool:
    """Turn on the sold homes filter in the open Filters panel."""
    snapshot = await _capture_filter_snapshot(browser, target_id)
    if snapshot is None:
        return False
    sold_radios = find_actionable_elements(
        snapshot, role="radio", name_contains="Sold"
    )
    if not sold_radios:
        print(
            "[Redfin] Error: Sold radio not found in filters panel",
            file=sys.stderr,
            flush=True,
        )
        return False
    return await snapshot.click(
        sold_radios[0]["ref_id"], timeout=_CLICK_TIMEOUT_MS, verbose=_VERBOSE
    )


async def _retry_filter_step[*Ts](
    browser: BrowserManager,
    target_id: str,
    filter_fn: Callable[[BrowserManager, str, *Ts], Awaitable[bool]],
    filter_args: tuple[*Ts],
    filter_name: str,
) -> bool:
    """Try to set a filter a few times before giving up."""
    for attempt in range(_BROWSER_OP_RETRY_COUNT):
        if await filter_fn(browser, target_id, *filter_args):
            return True
        if attempt < _BROWSER_OP_RETRY_COUNT - 1:
            await asyncio.sleep(_RETRY_BACKOFF_S)

    print(
        f"[Redfin] Error: could not apply {filter_name} filter",
        file=sys.stderr,
        flush=True,
    )
    return False


async def _require_filter[*Ts](
    browser: BrowserManager,
    target_id: str,
    filter_fn: Callable[[BrowserManager, str, *Ts], Awaitable[bool]],
    filter_args: tuple[*Ts],
    filter_name: str,
) -> None:
    """Apply a required filter step, or raise when all tries fail."""
    if not await _retry_filter_step(
        browser, target_id, filter_fn, filter_args, filter_name
    ):
        raise RedfinSearchError(f"could not apply {filter_name} filter")


async def apply_server_side_filters(
    browser: BrowserManager,
    target_id: str,
    options: SearchOptions,
) -> None:
    """Open the Filters panel and set the requested Redfin filters.

    Args:
        browser: Browser to work with.
        target_id: Page to work on.
        options: Search filters to apply.

    Raises:
        RedfinSearchError: When a requested filter cannot be set.
    """
    if not await _open_filters_panel(browser, target_id):
        raise RedfinSearchError("could not open the Filters panel")
    if options.is_sold:
        await _require_filter(browser, target_id, _apply_sold_filter, (), "sold")
    if options.min_price or options.max_price:
        await _require_filter(
            browser,
            target_id,
            _apply_price_filter,
            (options.min_price, options.max_price),
            "price",
        )
    if options.date_from:
        is_applied = await _retry_filter_step(
            browser,
            target_id,
            _apply_time_filter,
            (options.date_from, options.is_sold),
            "time",
        )
        if not is_applied and options.is_sold:
            raise RedfinSearchError("could not apply time filter")
    if options.min_beds or options.min_baths:
        await _require_filter(
            browser,
            target_id,
            _apply_beds_baths_filter,
            (options.min_beds, options.min_baths),
            "beds/baths",
        )

    snapshot = await _capture_filter_snapshot(browser, target_id)
    if snapshot is not None:
        await snapshot.press_key("Escape", verbose=_VERBOSE)
