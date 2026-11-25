"""Run Redfin searches in a browser and format results as markdown."""

from __future__ import annotations

__all__ = ["search_redfin"]

import asyncio
import os
import re
import shutil
import sys
import tempfile
from numbers import Number
from urllib.parse import urlparse

from app.lib.browser.browser_manager import BrowserManager
from app.lib.browser.page_snapshot import PageSnapshot
from app.lib.redfin.csv_parser import (
    filter_properties,
    filter_properties_by_date,
    parse_results_csv,
)
from app.lib.redfin.filters import apply_server_side_filters
from app.lib.redfin.models import (
    PROPERTY_URL_SEGMENT,
    REDFIN_BASE_URL,
    RedfinSearchError,
    SearchOptions,
    build_absolute_redfin_url,
)
from app.lib.redfin.property_parser import (
    extract_card_property,
    extract_off_market_property,
    extract_single_property,
    find_home_page_url,
    is_sold_property_page,
)
from app.lib.redfin.snapshot_tree import (
    find_actionable_elements,
    find_matching_nodes,
    is_listing_card_node,
)

_VERBOSE = True
_DEFAULT_LOAD_TIMEOUT_MS = 5000
_NAVIGATION_TIMEOUT_MS = 30000
_SNAPSHOT_TIMEOUT_MS = 2000
_POST_SEARCH_DELAY_S = 1
_CSV_DOWNLOAD_TIMEOUT_S = 15
_PUBLIC_PAGE_MAX_POLLS = 30
_URL_POLL_INTERVAL_S = 0.5
_DOWNLOAD_RETRY_COUNT = 3
_OWNER_DASHBOARD_PATH = "/myredfin/owner-dashboard/"
_PUBLIC_PAGE_BUTTON = "View public page"
_MISSING_CELL = "—"
_TABLE_COLUMNS = (
    "Address",
    "Price",
    "Beds",
    "Baths",
    "Sqft",
    "Status",
    "Date",
    "$/Sqft",
    "Year Built",
    "Lot",
    "URL",
)


def _is_results_page(url: str | None) -> bool:
    """Tell whether a Redfin URL points at a results or property page."""
    if not url:
        return False
    return urlparse(url).path.strip("/") != ""


def _is_owner_dashboard(url: str | None) -> bool:
    """Tell whether a URL points at the My Home owner dashboard."""
    return url is not None and _OWNER_DASHBOARD_PATH in url


def _is_property_page(url: str | None) -> bool:
    """Tell whether a URL points at a single Redfin home page."""
    return url is not None and PROPERTY_URL_SEGMENT in url


def _find_listing_card_nodes(tree: dict) -> list[dict]:
    """Find every listing card node in a page tree."""
    return [
        node
        for node in find_matching_nodes(tree, role="button")
        if is_listing_card_node(node)
    ]


async def _capture_page_snapshot(
    browser: BrowserManager, target_id: str
) -> PageSnapshot | None:
    """Capture a snapshot of the open page."""
    return await browser.capture_snapshot(
        target_id, timeout=_SNAPSHOT_TIMEOUT_MS, verbose=_VERBOSE
    )


async def _read_filtered_tree(snapshot: PageSnapshot) -> dict | None:
    """Read the filtered page tree, or None when it is missing."""
    tree = await snapshot.filtered_tree()
    return tree if isinstance(tree, dict) else None


async def _capture_results_tree(browser: BrowserManager, target_id: str) -> dict:
    """Wait for the results page to load and return its page tree."""
    await browser.wait_for_load(
        target_id, timeout=_DEFAULT_LOAD_TIMEOUT_MS, verbose=_VERBOSE
    )
    snapshot = await _capture_page_snapshot(browser, target_id)
    if snapshot is None:
        raise RedfinSearchError("could not capture the search results page")
    return await _read_filtered_tree(snapshot) or {}


async def _submit_initial_search(
    browser: BrowserManager, target_id: str, location: str
) -> dict:
    """Search Redfin for a location and return the landing page tree."""
    await browser.navigate(
        target_id, REDFIN_BASE_URL, timeout=_NAVIGATION_TIMEOUT_MS, verbose=_VERBOSE
    )
    await browser.wait_for_load(
        target_id, timeout=_DEFAULT_LOAD_TIMEOUT_MS, verbose=_VERBOSE
    )

    snapshot = await _capture_page_snapshot(browser, target_id)
    if snapshot is None:
        raise RedfinSearchError("could not capture the Redfin homepage")
    search_boxes = find_actionable_elements(
        snapshot, role="textbox", name_prefix="Search for properties"
    )
    if not search_boxes:
        raise RedfinSearchError("search box not found")
    await snapshot.fill(search_boxes[0]["ref_id"], location, verbose=_VERBOSE)
    await snapshot.press_key("Enter", verbose=_VERBOSE)

    await asyncio.sleep(_POST_SEARCH_DELAY_S)
    results_tree = await _capture_results_tree(browser, target_id)
    landing_url = browser.get_url(target_id, verbose=_VERBOSE)
    if landing_url and not _is_results_page(landing_url):
        raise RedfinSearchError(
            f"Redfin did not open a results page for '{location}'"
        )
    return results_tree


async def _wait_for_public_page(browser: BrowserManager, target_id: str) -> None:
    """Wait until the browser leaves the owner dashboard."""
    for _ in range(_PUBLIC_PAGE_MAX_POLLS):
        if not _is_owner_dashboard(browser.get_url(target_id, verbose=_VERBOSE)):
            return
        await asyncio.sleep(_URL_POLL_INTERVAL_S)


async def _open_public_page(
    browser: BrowserManager, target_id: str, tree: dict
) -> dict:
    """Open the public home page when the browser lands on the owner dashboard."""
    if not _is_owner_dashboard(browser.get_url(target_id, verbose=_VERBOSE)):
        return tree

    snapshot = await _capture_page_snapshot(browser, target_id)
    if snapshot is None:
        return tree
    buttons = find_actionable_elements(
        snapshot, role="button", name_contains=_PUBLIC_PAGE_BUTTON
    )
    if not buttons:
        return tree

    await snapshot.click(buttons[0]["ref_id"], verbose=_VERBOSE)
    await _wait_for_public_page(browser, target_id)

    public_snapshot = await _capture_page_snapshot(browser, target_id)
    if public_snapshot is None:
        return tree
    public_tree = await _read_filtered_tree(public_snapshot)
    return public_tree if public_tree is not None else tree


async def _extract_off_market_from_page(
    browser: BrowserManager, target_id: str, location: str
) -> dict | None:
    """Open the sale history tab when present and read the off market home."""
    snapshot = await _capture_page_snapshot(browser, target_id)
    if snapshot is None:
        return None
    sale_tabs = find_actionable_elements(snapshot, role="tab", name_contains="Sale")
    if sale_tabs:
        await snapshot.click(sale_tabs[0]["ref_id"], verbose=_VERBOSE)
        await browser.wait_for_load(
            target_id, timeout=_DEFAULT_LOAD_TIMEOUT_MS, verbose=_VERBOSE
        )

    updated_snapshot = await _capture_page_snapshot(browser, target_id)
    if updated_snapshot is None:
        return None
    updated_tree = await _read_filtered_tree(updated_snapshot)
    if updated_tree is None:
        return None
    return extract_off_market_property(updated_tree, location)


async def _open_matched_home_page(
    browser: BrowserManager, target_id: str, location: str, tree: dict
) -> dict:
    """Follow a matching home link and leave the owner dashboard when needed."""
    home_url = find_home_page_url(tree, location)
    if home_url:
        await browser.navigate(
            target_id, home_url, timeout=_NAVIGATION_TIMEOUT_MS, verbose=_VERBOSE
        )
        await browser.wait_for_load(
            target_id, timeout=_DEFAULT_LOAD_TIMEOUT_MS, verbose=_VERBOSE
        )
        snapshot = await _capture_page_snapshot(browser, target_id)
        if snapshot is not None:
            matched_tree = await _read_filtered_tree(snapshot)
            if matched_tree is not None:
                tree = matched_tree
    return await _open_public_page(browser, target_id, tree)


async def _extract_address_page_property(
    browser: BrowserManager,
    target_id: str,
    location: str,
    tree: dict,
    page_url: str,
) -> dict:
    """Read a single property record from an open address page."""
    if is_sold_property_page(tree):
        off_market = await _extract_off_market_from_page(browser, target_id, location)
        if off_market is not None:
            off_market["url"] = page_url
            return off_market

    listing = extract_single_property(tree, location)
    if listing is not None:
        listing["url"] = page_url
        return listing
    raise RedfinSearchError(
        f"could not extract property from address page for '{location}'"
    )


async def _search_street_address(
    browser: BrowserManager, target_id: str, location: str, tree: dict
) -> list[dict] | None:
    """Look up a single street address and return its property, or None."""
    if not re.match(r"\d+\s+\w+", location):
        return None

    home_tree = await _open_matched_home_page(browser, target_id, location, tree)
    page_url = browser.get_url(target_id, verbose=_VERBOSE)
    if not page_url or (
        not _is_property_page(page_url) and not _is_owner_dashboard(page_url)
    ):
        raise RedfinSearchError(f"no property page found for '{location}'")

    listing = await _extract_address_page_property(
        browser, target_id, location, home_tree, page_url
    )
    return [listing]


async def _navigate_to_suggestion(
    browser: BrowserManager, target_id: str, tree: dict
) -> dict | None:
    """Open the zipcode or city page that Redfin suggests, or return None."""
    suggestion_url = None
    for link in find_matching_nodes(tree, role="link"):
        url = link.get("url", "")
        if "/zipcode/" in url or "/city/" in url:
            suggestion_url = url
            break

    if not suggestion_url:
        print(
            "[Redfin] No results or suggestions found",
            file=sys.stderr,
            flush=True,
        )
        return None

    await browser.navigate(
        target_id,
        build_absolute_redfin_url(suggestion_url),
        timeout=_NAVIGATION_TIMEOUT_MS,
        verbose=_VERBOSE,
    )
    return await _capture_results_tree(browser, target_id)


async def _download_csv(
    browser: BrowserManager, target_id: str, save_path: str
) -> str | None:
    """Download the CSV of the search results, or return None."""
    for _ in range(_DOWNLOAD_RETRY_COUNT):
        snapshot = await _capture_page_snapshot(browser, target_id)
        if snapshot is None:
            continue
        download_links = find_actionable_elements(
            snapshot, role="link", name_contains="download"
        )
        if not download_links:
            continue

        ref_id = download_links[0]["ref_id"]

        async def _click_download() -> bool:
            return await snapshot.click(ref_id, verbose=_VERBOSE)

        result = await browser.wait_for_download(
            target_id=target_id,
            action=_click_download,
            save_path=save_path,
            timeout=_CSV_DOWNLOAD_TIMEOUT_S,
            verbose=_VERBOSE,
        )
        if result.get("downloaded"):
            return result.get("path")

    return None


async def _extract_properties_from_cards(
    browser: BrowserManager, target_id: str, options: SearchOptions
) -> list[dict]:
    """Read properties directly from the listing cards on the results page."""
    snapshot = await _capture_page_snapshot(browser, target_id)
    if snapshot is None:
        raise RedfinSearchError("could not capture the results page")
    tree = await _read_filtered_tree(snapshot)
    if not tree:
        return []

    properties: list[dict] = []
    for card in _find_listing_card_nodes(tree):
        if len(properties) >= options.limit:
            break
        listing = extract_card_property(card, options.is_sold)
        if listing:
            properties.append(listing)

    if options.date_from or options.date_to:
        properties = filter_properties_by_date(
            properties, options.date_from, options.date_to
        )
    return properties[: options.limit]


async def _extract_properties(
    browser: BrowserManager, target_id: str, options: SearchOptions
) -> list[dict]:
    """Read matching properties from CSV download or page cards."""
    csv_dir = await asyncio.to_thread(tempfile.mkdtemp, prefix="redfin_csv_")
    try:
        csv_path = await _download_csv(
            browser, target_id, os.path.join(csv_dir, "results.csv")
        )
        if csv_path:
            properties = await parse_results_csv(csv_path)
            return filter_properties(properties, options)
    finally:
        await asyncio.to_thread(shutil.rmtree, csv_dir, ignore_errors=True)

    return await _extract_properties_from_cards(browser, target_id, options)


async def _extract_and_filter_results(
    browser: BrowserManager, target_id: str, options: SearchOptions
) -> list[dict]:
    """Set server filters when needed and read the matching properties."""
    if options.has_server_filters():
        await apply_server_side_filters(browser, target_id, options)
    return await _extract_properties(browser, target_id, options)


def _format_table_cell(value: str | Number | None) -> str:
    """Turn a value into a table cell, using a dash when missing."""
    return _MISSING_CELL if value is None else str(value)


def _format_listing_date(listing: dict) -> str:
    """Return the sold date or listed date of a property, or empty text."""
    sold_date = listing.get("sold_date")
    if sold_date:
        return str(sold_date)
    listed_date = listing.get("listed_date")
    return str(listed_date) if listed_date else ""


def _format_properties_markdown(properties: list[dict], location: str) -> str:
    """Turn properties into a markdown table, or a no results message."""
    if not properties:
        return f"No properties found for {location}."
    lines = [
        "| " + " | ".join(_TABLE_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _TABLE_COLUMNS) + " |",
    ]
    for listing in properties:
        cells = (
            _format_table_cell(listing.get("address")),
            _format_table_cell(listing.get("price")),
            _format_table_cell(listing.get("beds")),
            _format_table_cell(listing.get("baths")),
            _format_table_cell(listing.get("sqft")),
            _format_table_cell(listing.get("status")),
            _format_listing_date(listing),
            _format_table_cell(listing.get("price_per_sqft")),
            _format_table_cell(listing.get("year_built")),
            _format_table_cell(listing.get("lot_size")),
            _format_table_cell(listing.get("url")),
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


async def search_redfin(location: str, options: SearchOptions) -> str:
    """Search Redfin for a place and return the results as markdown.

    Args:
        location: Street address, city, or zipcode to look up.
        options: Search filters and result limit.

    Returns:
        A markdown table of properties, or a line saying nothing was found.

    Raises:
        RedfinSearchError: When the search cannot finish or pages cannot be
            read.
    """
    browser = await BrowserManager.create(verbose=_VERBOSE)
    try:
        target_id = await browser.open_page(verbose=_VERBOSE)
        tree = await _submit_initial_search(browser, target_id, location)

        address_result = await _search_street_address(
            browser, target_id, location, tree
        )
        if address_result is not None:
            return _format_properties_markdown(address_result, location)

        if not _find_listing_card_nodes(tree):
            suggested_tree = await _navigate_to_suggestion(browser, target_id, tree)
            if suggested_tree is None:
                return f"No properties found for {location}."

        properties = await _extract_and_filter_results(browser, target_id, options)
        return _format_properties_markdown(properties, location)
    finally:
        await browser.cleanup(verbose=_VERBOSE)
