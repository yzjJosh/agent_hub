from __future__ import annotations

"""Search Redfin for properties by address, city, or zipcode.

Gives back a markdown table of price, beds, baths, sqft, and dates for
active listings and sold homes.
"""

__all__ = ["redfin_search_tool"]

import asyncio
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from numbers import Number
from typing import ClassVar, override
from urllib.parse import urlparse

from app.lib.browser.browser_manager import BrowserManager
from app.lib.browser.page_snapshot import PageSnapshot
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult

_VERBOSE = True

# ==============================================================================
# Constants
# ==============================================================================

_DEFAULT_LOAD_TIMEOUT = 5000
_NAVIGATION_TIMEOUT = 30000
_CLICK_TIMEOUT = 10000
_TIME_FILTER_SELECT_TIMEOUT = 10000
_SNAPSHOT_TIMEOUT = 2000
_POST_SEARCH_DELAY_S = 1
_OWNER_DASHBOARD_PATH = "/myredfin/owner-dashboard/"
_PROPERTY_URL_SEGMENT = "/home/"
_PUBLIC_PAGE_BUTTON = "View public page"
_LISTING_CARD_NAME = "Property at"
_PRICE_NUMBER = r"(\d{1,3}(?:,\d{3})+(?:\.\d{2})?)(?![\d,-])"
_STATUS_PRICE_RE = re.compile(
    rf"^(?:For sale|Coming soon|Contingent|Off Market)\b.*?\${_PRICE_NUMBER}",
    re.IGNORECASE,
)
_HOME_PRICE_RE = re.compile(rf"^Home price\b.*?\${_PRICE_NUMBER}", re.IGNORECASE)
_LEADING_PRICE_RE = re.compile(rf"^\${_PRICE_NUMBER}")
# Words that name a figure which is not the list price of the home.
_COST_LABEL_RE = re.compile(
    r"per month|/mo|price/sq|/sq ft|sqft|hoa|est\.|taxes|insurance|interest",
    re.IGNORECASE,
)
_PUBLIC_PAGE_MAX_POLLS = 30
_URL_POLL_INTERVAL_S = 0.5

_BEDS_OPTION_THRESHOLD = 5
_DAYS_PER_MONTH = 30
_DOM_FILTER_BUCKETS = (3, 7, 14, 30)
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
_CSV_DOWNLOAD_TIMEOUT = 15
_BROWSER_OP_RETRY_COUNT = 3
_RETRY_BACKOFF_S = 1
_FILTER_LABELS = {"Beds", "Baths", "Garage", "Price", "Sqft", "Lot", "Year", "Status"}
_PRICE_SUFFIX_MULTIPLIERS = {"M": 1_000_000, "K": 1_000}
_SQFT_PER_ACRE = 43560
_NEARBY_SUFFIX_WINDOW = 5
_REDFIN_URL_CSV_COLUMN = (
    "URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis"
    " FOR INFO ON PRICING)"
)
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


# ==============================================================================
# Options and errors
# ==============================================================================


class _RedfinSearchError(Exception):
    """Raised when a Redfin search cannot be completed."""


@dataclass
class _SearchOptions:
    """What to look for in a Redfin search.

    Attributes:
        sold: True to look at sold homes instead of active listings.
        min_beds: Fewest bedrooms to keep.
        min_baths: Fewest bathrooms to keep.
        limit: Most results to give back.
        min_price: Lowest price to keep.
        max_price: Highest price to keep.
        date_from: First date to keep, YYYY-MM-DD.
        date_to: Last date to keep, YYYY-MM-DD.
    """

    sold: bool = False
    min_beds: int | None = None
    min_baths: int | None = None
    limit: int = 10
    min_price: int | None = None
    max_price: int | None = None
    date_from: str | None = None
    date_to: str | None = None


# ==============================================================================
# Tree traversal utilities
# ==============================================================================


def _find_actionable(
    snap: PageSnapshot,
    role: str | None = None,
    name: str | None = None,
    name_contains: str | None = None,
    name_prefix: str | None = None,
) -> list[dict]:
    """Find page elements a user can click or type into.

    Args:
        snap: Snapshot of the page.
        role: Keep only elements with this role, like button or textbox.
        name: Keep only elements with this exact name, big and small letters
            must match.
        name_contains: Keep only elements whose name holds this text, big and
            small letters do not matter.
        name_prefix: Keep only elements whose name starts with this text.

    Returns:
        The matching elements.
    """
    results = []
    for el in snap.actionable_elements():
        if role and el.get("role") != role:
            continue
        if name is not None and el.get("name") != name:
            continue
        if name_contains and name_contains.lower() not in el.get("name", "").lower():
            continue
        if name_prefix and not el.get("name", "").startswith(name_prefix):
            continue
        results.append(el)
    return results


def _find_all_nodes(
    tree: dict,
    role: str | None = None,
    name_pattern: str | None = None,
    name_prefix: str | None = None,
) -> list[dict]:
    """Find all the nodes in a page tree that match.

    Nodes that hold only text are skipped.

    Args:
        tree: Node to start from.
        role: Keep only nodes with this role.
        name_pattern: Pattern the node name must match.
        name_prefix: Text the node name must start with.

    Returns:
        The matching nodes from the whole tree. Nothing comes back when no
        role, pattern, or prefix is given.
    """
    results = []
    if not isinstance(tree, dict):
        return results
    if "text" in tree and "role" not in tree:
        return results
    match = True
    if role and tree.get("role") != role:
        match = False
    if name_pattern and not re.search(name_pattern, tree.get("name", "")):
        match = False
    if name_prefix and not tree.get("name", "").startswith(name_prefix):
        match = False
    if match and (role or name_pattern or name_prefix):
        results.append(tree)
    for child in tree.get("children", []):
        results.extend(_find_all_nodes(child, role, name_pattern, name_prefix))
    return results


def _find_node_by_name(node: dict | list, target: str) -> tuple[dict | list | None, dict | None]:
    """Find a node by name.

    Args:
        node: Where to start looking.
        target: The name to look for.

    Returns:
        The thing holding the node and the node itself. Both are None when
        the name is not there.
    """
    if isinstance(node, dict):
        for child in node.get("children", []):
            if isinstance(child, dict) and child.get("name") == target:
                return node, child
            container, found = _find_node_by_name(child, target)
            if found is not None:
                return container, found
    if isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and item.get("name") == target:
                return node, item
            container, found = _find_node_by_name(item, target)
            if found is not None:
                return container, found
    return None, None


def _find_label_sibling_buttons(tree: dict, label_name: str) -> list[str]:
    """Find the names of the buttons that sit under a label.

    Args:
        tree: Top of the page tree.
        label_name: Name of the label, like Beds or Baths.

    Returns:
        The button names found next to the label.
    """
    parent, label_node = _find_node_by_name(tree, label_name)
    if parent is None:
        return []

    if isinstance(parent, dict):
        siblings = parent.get("children", [])
    else:
        siblings = parent

    try:
        label_idx = next(i for i, s in enumerate(siblings) if s is label_node)
    except StopIteration:
        label_idx = next(
            (
                i
                for i, s in enumerate(siblings)
                if isinstance(s, dict) and s.get("name") == label_name
            ),
            -1,
        )
    if label_idx < 0:
        return []

    button_names = []
    seen: set[str] = set()
    for sibling in siblings[label_idx + 1 :]:
        if isinstance(sibling, dict) and sibling.get("role") == "button":
            name = sibling.get("name", "")
            if name and name not in seen:
                button_names.append(name)
                seen.add(name)
        elif isinstance(sibling, dict) and sibling.get("name") in _FILTER_LABELS:
            break
        elif isinstance(sibling, dict):
            text = _collect_all_texts(sibling)
            if text and ("Any" in text or any(c.isdigit() for c in text)):
                for word in text.split():
                    if word and word not in seen:
                        button_names.append(word)
                        seen.add(word)
            for btn in _find_all_nodes(sibling, role="button"):
                name = btn.get("name", "")
                if name and name not in seen:
                    button_names.append(name)
                    seen.add(name)

    return button_names


def _find_text_in_descendants(node: dict, pattern: str) -> str | None:
    """Find text under a node that matches a pattern.

    Args:
        node: Node to look under.
        pattern: Pattern the text must match.

    Returns:
        The first text that matches, or None.
    """
    if not isinstance(node, dict):
        return None
    for child in node.get("children", []):
        if isinstance(child, dict):
            if "text" in child and "role" not in child:
                if re.search(pattern, child.get("text", "")):
                    return child.get("text", "")
            result = _find_text_in_descendants(child, pattern)
            if result:
                return result
    return None


def _is_listing_card(node: dict) -> bool:
    """Tell whether a node is the card of another home on the page."""
    return node.get("role") == "button" and node.get("name", "").startswith(
        _LISTING_CARD_NAME
    )


def _gather_texts(
    tree: dict | None, should_skip_listing_cards: bool = False
) -> list[str]:
    """Collect the text under a page tree into pieces.

    Args:
        tree: Node to read the text from, or None.
        should_skip_listing_cards: Leave out the text of another home's card.

    Returns:
        The text pieces, in page order.
    """
    texts: list[str] = []

    def walk(node: dict) -> None:
        if not isinstance(node, dict):
            return
        if should_skip_listing_cards and _is_listing_card(node):
            return
        if "text" in node and "role" not in node:
            texts.append(node.get("text", ""))
        for child in node.get("children", []):
            walk(child)

    if tree is not None:
        walk(tree)
    return texts


def _collect_all_texts(tree: dict | None) -> str:
    """Put all the text in a page tree into one string.

    Args:
        tree: Node to read the text from, or None.

    Returns:
        The text with spaces between the pieces.
    """
    return " ".join(_gather_texts(tree))


# ==============================================================================
# Property extraction helpers
# ==============================================================================


def _extract_street(location: str) -> str:
    """Pull the street out of a location.

    The whole location comes back when it does not start with a street.
    """
    street_match = re.match(r"(\d+\s+[^,]+)", location)
    return street_match.group(1).strip() if street_match else location


def _extract_address_fields(address: str) -> dict[str, str]:
    """Pull the city, zip, and street out of an address.

    Only the parts that are found come back.
    """
    result = {}
    if not address:
        return result
    city_match = re.search(r",\s*(.+?),\s*[A-Z]{2},?\s+\d{5}(?:-\d{4})?\s*$", address)
    if city_match:
        result["city"] = city_match.group(1).strip()
    zip_match = re.search(r",\s*[A-Z]{2},?\s+(\d{5})(?:-\d{4})?\s*$", address)
    if zip_match:
        result["zip"] = zip_match.group(1)
    result["street"] = _extract_street(address)
    return result


def _compute_price_per_sqft(price_str: str, sqft: int | None) -> str | None:
    """Work out the price for each square foot.

    Args:
        price_str: Price text like $1,980,000.
        sqft: Size of the home in square feet.

    Returns:
        The price per square foot as text, or None when it cannot be worked
        out.
    """
    if not price_str or sqft is None or sqft <= 0:
        return None
    cleaned = price_str.replace("$", "").replace(",", "")
    num_match = re.match(r"([\d.]+)", cleaned)
    if not num_match:
        return None
    num_str = num_match.group(1)
    if "M" in cleaned:
        price = float(num_str) * _PRICE_SUFFIX_MULTIPLIERS["M"]
    elif "K" in cleaned:
        price = float(num_str) * _PRICE_SUFFIX_MULTIPLIERS["K"]
    else:
        price = float(num_str)
    return f"${int(price / sqft)}"


def _normalize_url(url: str) -> str:
    """Turn a short link into a full Redfin link.

    A link that is already full comes back the same.
    """
    return url if url.startswith("http") else f"https://www.redfin.com{url}"


def _is_results_page(url: str | None) -> bool:
    """Tell whether a Redfin address points at a page of listings.

    The landing page has no path, so its empty path is not a listings page.
    """
    return bool(url) and urlparse(url).path.strip("/") != ""


def _finalize_property(prop: dict) -> dict:
    """Add the city, zip, street, and price per square foot to a property.

    Args:
        prop: Property with an address, and maybe a price and a sqft.

    Returns:
        A new property with the extra fields filled in.
    """
    result = dict(prop)
    result.update(_extract_address_fields(prop.get("address")))
    if "price" in prop and "sqft" in prop:
        pps = _compute_price_per_sqft(prop["price"], prop["sqft"])
        if pps:
            result["price_per_sqft"] = pps
    return result


def _parse_csv_date(date_str: str) -> str | None:
    """Turn a date like June-05-2026 into 2026-06-05, or None when it does not fit."""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%B-%d-%Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_csv_datetime(dt_str: str) -> str | None:
    """Turn a date and time like June-05-2026 04:30 PM into 2026-06-05T16:30:00.

    None comes back when the text does not fit.
    """
    if not dt_str:
        return None
    try:
        dt = datetime.strptime(dt_str, "%B-%d-%Y %I:%M %p")
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def _parse_int_or_none(value: str) -> int | None:
    """Turn text into a whole number, or None when it does not fit."""
    if not value:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _parse_float_or_none(value: str) -> float | None:
    """Turn text into a number, or None when it does not fit."""
    if not value:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _format_price(price_str: str) -> str | None:
    """Turn a plain price number into text like $1,234,567.

    None comes back when the text is not a number.
    """
    raw = _parse_int_or_none(price_str)
    if raw is None:
        return None
    return f"${raw:,}"


def _clean_str(value: str) -> str | None:
    """Cut the spaces off text, or give back None when nothing is left."""
    if not value:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _normalize_csv_numeric_fields(row: dict[str, str]) -> dict[str, int | float | None]:
    """Read the number fields out of a CSV row."""
    return {
        "beds": _parse_int_or_none(row.get("BEDS", "")),
        "baths": _parse_float_or_none(row.get("BATHS", "")),
        "sqft": _parse_int_or_none(row.get("SQUARE FEET", "")),
        "lot_size": _parse_int_or_none(row.get("LOT SIZE", "")),
        "year_built": _parse_int_or_none(row.get("YEAR BUILT", "")),
        "days_on_market": _parse_int_or_none(row.get("DAYS ON MARKET", "")),
        "hoa": _parse_int_or_none(row.get("HOA/MONTH", "")),
        "latitude": _parse_float_or_none(row.get("LATITUDE", "")),
        "longitude": _parse_float_or_none(row.get("LONGITUDE", "")),
    }


def _normalize_csv_address(row: dict[str, str]) -> str:
    """Join the address parts of a CSV row into one address."""
    parts = [
        _clean_str(row.get("ADDRESS", "")) or "",
        _clean_str(row.get("CITY", "")) or "",
        _clean_str(row.get("STATE OR PROVINCE", "")) or "",
        _clean_str(row.get("ZIP OR POSTAL CODE", "")) or "",
    ]
    return ", ".join(p for p in parts if p)


def _normalize_csv_row(row: dict[str, str]) -> dict:
    """Turn one CSV row into a property.

    Args:
        row: One row of the Redfin CSV.

    Returns:
        The property with its fields filled in.
    """
    sale_type = _clean_str(row.get("SALE TYPE", ""))
    status = "Sold" if sale_type == "PAST SALE" else "Active"

    price = _format_price(row.get("PRICE", ""))
    numeric = _normalize_csv_numeric_fields(row)
    price_per_sqft_raw = _parse_int_or_none(row.get("$/SQUARE FEET", ""))
    price_per_sqft = f"${price_per_sqft_raw}" if price_per_sqft_raw else None
    address = _normalize_csv_address(row)

    return {
        "address": address or None,
        "price": price,
        "beds": numeric["beds"],
        "baths": numeric["baths"],
        "sqft": numeric["sqft"],
        "lot_size": numeric["lot_size"],
        "year_built": numeric["year_built"],
        "price_per_sqft": price_per_sqft,
        "hoa": numeric["hoa"],
        "days_on_market": numeric["days_on_market"],
        "status": status,
        "property_type": _clean_str(row.get("PROPERTY TYPE", "")),
        "sale_type": sale_type,
        "sold_date": _parse_csv_date(row.get("SOLD DATE", "")),
        "open_house_start": _parse_csv_datetime(
            row.get("NEXT OPEN HOUSE START TIME", "")
        ),
        "open_house_end": _parse_csv_datetime(row.get("NEXT OPEN HOUSE END TIME", "")),
        "url": _clean_str(row.get("URL", "") or row.get(_REDFIN_URL_CSV_COLUMN, "")),
        "source": _clean_str(row.get("SOURCE", "")),
        "mls_id": _clean_str(row.get("MLS#", "")),
        "latitude": numeric["latitude"],
        "longitude": numeric["longitude"],
        "zip": _clean_str(row.get("ZIP OR POSTAL CODE", "")),
        "street": _clean_str(row.get("ADDRESS", "")),
    }


def _parse_csv(path: str) -> list[dict]:
    """Read a Redfin CSV file into properties.

    Rows that carry the MLS rules notice are left out.

    Args:
        path: Path to the CSV file.

    Returns:
        One property for each row of data.

    Raises:
        _RedfinSearchError: When the file is not UTF-8 text.
    """
    properties = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sale_type = row.get("SALE TYPE", "").strip()
                if "MLS rules" in sale_type:
                    continue
                properties.append(_normalize_csv_row(dict(row)))
    except UnicodeDecodeError:
        raise _RedfinSearchError("could not read the results CSV as text") from None
    return properties


def _parse_price_int(price_str: str | None) -> int | None:
    """Turn price text like $1,234,567 into the number 1234567.

    None comes back when the text is not a price.
    """
    if not price_str:
        return None
    cleaned = price_str.replace("$", "").replace(",", "")
    try:
        return int(cleaned)
    except ValueError:
        return None


def _filter_properties(
    properties: list[dict],
    options: _SearchOptions,
) -> list[dict]:
    """Keep only the properties that match what was asked for.

    A property with no readable price is dropped when a price was asked for.

    Args:
        properties: Properties to look through.
        options: What to look for.

    Returns:
        The properties that match, cut down to the limit.
    """
    has_filters = (
        options.min_beds
        or options.min_baths
        or options.min_price
        or options.max_price
        or options.date_from
        or options.date_to
    )
    if not has_filters:
        return properties[: options.limit]

    filtered = []
    for prop in properties:
        if options.min_beds and (prop.get("beds") or 0) < options.min_beds:
            continue
        if options.min_baths and (prop.get("baths") or 0) < options.min_baths:
            continue
        prop_price = _parse_price_int(prop.get("price"))
        if (options.min_price or options.max_price) and prop_price is None:
            continue
        if options.min_price and prop_price < options.min_price:
            continue
        if options.max_price and prop_price > options.max_price:
            continue
        filtered.append(prop)

    if options.date_from or options.date_to:
        filtered = _filter_by_date(
            filtered, options.date_from, options.date_to
        )

    return filtered[: options.limit]


def _parse_iso_date(value: str) -> datetime | None:
    """Read a YYYY-MM-DD date, or give back None when it does not fit."""
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def _date_to_dom_value(date_from: str) -> str | None:
    """Turn a start date into the value Redfin's time filter takes.

    Args:
        date_from: Start date, YYYY-MM-DD.

    Returns:
        The value to pick, or None when the date is more than 30 days back.
        Redfin's time filter for active listings does not go back further.

    Raises:
        _RedfinSearchError: When date_from is not a YYYY-MM-DD date.
    """
    from_date = _parse_iso_date(date_from)
    if from_date is None:
        raise _RedfinSearchError(f"invalid date_from: {date_from}")

    days = (datetime.now() - from_date).days
    if days < 1:
        days = 1

    for bucket in _DOM_FILTER_BUCKETS:
        if days <= bucket:
            return f"{bucket}-"
    return None


def _parse_prop_date(prop: dict) -> datetime | None:
    """Read the sold date, listed date, or plain date of a property.

    None comes back when there is no date that can be read.
    """
    prop_date_str = prop.get("sold_date") or prop.get("listed_date") or prop.get("date")
    if not prop_date_str:
        return None
    for fmt in ["%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%m/%d/%Y"]:
        try:
            return datetime.strptime(prop_date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _filter_by_date(
    properties: list[dict],
    date_from: str | None,
    date_to: str | None,
) -> list[dict]:
    """Keep only the properties whose date falls in a range.

    An active listing with no readable date is kept.

    Args:
        properties: Properties to look through.
        date_from: First date to keep, YYYY-MM-DD, or None.
        date_to: Last date to keep, YYYY-MM-DD, or None.

    Returns:
        The properties inside the range.

    Raises:
        _RedfinSearchError: When a date is not YYYY-MM-DD.
    """
    if not date_from and not date_to:
        return properties

    from_date = _parse_iso_date(date_from) if date_from else None
    if date_from and from_date is None:
        raise _RedfinSearchError(f"invalid date_from: {date_from}")
    to_date = _parse_iso_date(date_to) if date_to else None
    if date_to and to_date is None:
        raise _RedfinSearchError(f"invalid date_to: {date_to}")

    filtered = []
    for prop in properties:
        prop_date = _parse_prop_date(prop)
        if not prop_date:
            if prop.get("status") == "Active":
                filtered.append(prop)
            continue
        if from_date and prop_date < from_date:
            continue
        if to_date and prop_date > to_date:
            continue
        filtered.append(prop)

    if _VERBOSE:
        print(
            f"[Redfin] Date filter: {len(properties)} -> {len(filtered)} properties",
            flush=True,
        )
    return filtered


# ==============================================================================
# Single property extraction
# ==============================================================================


def _is_cost_figure(texts: list[str], index: int) -> bool:
    """Tell whether the price at index is a payment, tax, or size figure.

    Args:
        texts: The text pieces of the page, in page order.
        index: Which piece holds the price.

    Returns:
        True when the page names that figure a cost.
    """
    parts = [texts[index]]
    for near in (texts[index - 1 : index], texts[index + 1 : index + 2]):
        for piece in near:
            # The words naming a figure hold no numbers themselves.
            if not any(char.isdigit() for char in piece):
                parts.append(piece)
    return bool(_COST_LABEL_RE.search(" ".join(parts)))


def _extract_single_pricing(tree: dict) -> str | None:
    """Pull this home's listed price out of a home details page.

    The price is the one on the home's own status line, else the one labeled
    Home price, else a price that starts a piece of its own.

    Args:
        tree: Tree of the page.

    Returns:
        The price as text like $1,224,950, or None when the page has none.
    """
    texts = _gather_texts(tree, should_skip_listing_cards=True)
    for pattern in (_STATUS_PRICE_RE, _HOME_PRICE_RE):
        for text in texts:
            price_match = pattern.search(text)
            if price_match:
                return f"${price_match.group(1)}"
    for index, text in enumerate(texts):
        leading_match = _LEADING_PRICE_RE.match(text)
        if leading_match and not _is_cost_figure(texts, index):
            return f"${leading_match.group(1)}"
    return None


def _extract_single_property(tree: dict, location: str) -> dict | None:
    """Pull one property out of a Redfin home details page.

    Args:
        tree: Tree of the page.
        location: The place that was searched for, used when the page has no
            address.

    Returns:
        The property, or None when the address or the price is missing.
    """
    prop: dict = {}
    full_text = _collect_all_texts(tree)

    full_address = _extract_full_address(tree, location)
    if full_address:
        prop["address"] = full_address

    price = _extract_single_pricing(tree)
    if price:
        prop["price"] = price

    _fill_specs_from_text(prop, full_text)

    prop = _finalize_property(prop)

    if "address" in prop and "price" in prop:
        return prop
    return None


def _is_sold_page(tree: dict) -> bool:
    """Tell whether a page shows a home that was sold."""
    return bool(
        _find_all_nodes(
            tree, role="generic", name_pattern=r"SOLD\s+[A-Z]{3}\s+\d{1,2},\s*\d{4}"
        )
    )


def _parse_date_text(text: str | None) -> str | None:
    """Turn date text like SOLD JUN 5, 2026 into 2026-06-05.

    None comes back when there is no date in the text.
    """
    if not text:
        return None
    m = re.search(
        r"(?:SOLD(?:\s+BY\s+\w+)?\s+)?([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})", text
    )
    if m:
        for fmt in ["%b %d, %Y", "%B %d, %Y"]:
            try:
                dt = datetime.strptime(f"{m.group(1)} {m.group(2)}, {m.group(3)}", fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    m = re.search(r"(?:SOLD\s+)?(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        try:
            dt = datetime.strptime(f"{m.group(1)}/{m.group(2)}/{m.group(3)}", "%m/%d/%Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _parse_off_market_abbreviated(
    full_text: str,
) -> tuple[str | None, str | None]:
    """Pull the price and sold date out of short off market text.

    The text looks like Off Market Mar 2025 for $2M.

    Returns:
        The price and the sold date. Each one is None when it is not found.
    """
    m = re.search(
        r"[Oo]ff\s+Market\s+(\w+)\s+(?:\d{1,2},?\s+)?(\d{4})\s+for\s+\$([\d,.]+)\s*(M|K)?",
        full_text,
    )
    if not m:
        return None, None

    month_str, year_str = m.group(1), m.group(2)
    price_num = m.group(3).replace(",", "")
    price_suffix = m.group(4) or ""
    if not price_suffix:
        nearby = full_text[m.end() : m.end() + _NEARBY_SUFFIX_WINDOW].strip()
        if nearby and nearby[0].upper() in ("M", "K"):
            price_suffix = nearby[0].upper()

    price = None
    try:
        val = float(price_num)
        mult = _PRICE_SUFFIX_MULTIPLIERS.get(price_suffix.upper(), 1)
        price = f"${int(val * mult):,}"
    except (ValueError, TypeError):
        pass

    sold_date = None
    day_m = re.search(r"[Oo]ff\s+Market\s+\w+\s+(\d{1,2}),?\s+\d{4}", full_text)
    date_str = (
        f"{month_str} {day_m.group(1)}, {year_str}"
        if day_m
        else f"{month_str} {year_str}"
    )
    for fmt in ["%b %d, %Y", "%B %d, %Y", "%b %Y", "%B %Y"]:
        try:
            sold_date = datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
            break
        except ValueError:
            continue

    return price, sold_date


def _extract_off_market_pricing(
    full_text: str,
) -> tuple[str | None, str | None]:
    """Pull the price and sold date out of an off market page.

    The sale history is used when the page has one, and the short off market
    line is used when it does not.

    Returns:
        The price and the sold date. Each one is None when it is not found.
    """
    sale_history_matches = re.findall(
        r"([A-Za-z]+\s+\d{1,2},\s+\d{4})\s+Sold\s+\$([\d,]+)", full_text
    )
    if sale_history_matches:
        date_str, price_str = sale_history_matches[0]
        price = f"${price_str}"
        sold_date = _parse_date_text(date_str)
        return price, sold_date

    return _parse_off_market_abbreviated(full_text)


def _fill_specs_from_text(prop: dict, full_text: str) -> None:
    """Set the beds, baths, sqft, year built, and lot size from page text.

    Only the values found in the text are set.
    """
    beds, baths, sqft = _extract_specs(full_text)
    if beds is not None:
        prop["beds"] = beds
    if baths is not None:
        prop["baths"] = baths
    if sqft is not None:
        prop["sqft"] = sqft

    year_built, lot_size = _extract_details(full_text)
    if year_built is not None:
        prop["year_built"] = year_built
    if lot_size is not None:
        prop["lot_size"] = lot_size


def _extract_specs(
    full_text: str,
) -> tuple[int | None, float | None, int | None]:
    """Pull the beds, baths, and sqft out of page text.

    Returns:
        The beds, baths, and sqft. Each one is None when the text does not
        list them.
    """
    specs_match = re.search(
        r"(\d+)\s*bd\s*[•|]\s*(\d+\.?\d*)\s*ba\s*[•|]\s*(\d{1,3}(?:,\d{3})+)\s*sq\s*ft",
        full_text,
    )
    if not specs_match:
        return None, None, None
    beds = int(specs_match.group(1))
    baths = float(specs_match.group(2))
    try:
        sqft = int(specs_match.group(3).replace(",", ""))
    except ValueError:
        sqft = None
    return beds, baths, sqft


def _extract_details(full_text: str) -> tuple[int | None, int | None]:
    """Pull the year built and the lot size out of page text.

    A lot size given in acres is turned into square feet.

    Returns:
        The year built and the lot size in square feet. Each one is None when
        it is not found.
    """
    year_match = re.search(r"(\d{4})\s+Year\s+Built", full_text)
    year_built = int(year_match.group(1)) if year_match else None

    lot_size = None

    lot_match = re.search(r"(\d{1,3}(?:,\d{3})+)\s*sq\s*ft\s*Lot\s*Size", full_text)
    if lot_match:
        try:
            lot_size = int(lot_match.group(1).replace(",", ""))
        except ValueError:
            pass

    if lot_size is None:
        acres_match = re.search(r"(\d+\.?\d*)\s*acres?\s*Lot\s*Size", full_text)
        if acres_match:
            try:
                lot_size = int(float(acres_match.group(1)) * _SQFT_PER_ACRE)
            except ValueError:
                pass

    return year_built, lot_size


def _extract_full_address(tree: dict, location: str) -> str | None:
    """Pull the full address with city, state, and zip out of a page.

    The place that was searched for comes back when the page has no heading
    with the address in it.
    """
    location_street = _extract_street(location).lower()
    headings = _find_all_nodes(tree, role="heading")
    for heading in headings:
        name = heading.get("name", "")
        if location_street in name.lower() and re.search(r"\d{5}", name):
            return name.strip()

    return location.strip() if location else None


def _extract_off_market_property(
    tree: dict, location: str
) -> dict | None:
    """Pull one property out of an off market home details page.

    Args:
        tree: Tree of the page.
        location: The place that was searched for, used when the page has no
            address.

    Returns:
        The property, or None when the price or the address is missing.
    """
    prop: dict = {}
    full_text = _collect_all_texts(tree)

    price, sold_date = _extract_off_market_pricing(full_text)
    if price:
        prop["price"] = price
        prop["status"] = "Off Market"
    if sold_date:
        prop["sold_date"] = sold_date

    _fill_specs_from_text(prop, full_text)

    if "price" not in prop:
        return None

    full_address = _extract_full_address(tree, location)
    if full_address:
        prop["address"] = full_address

    prop = _finalize_property(prop)

    if "address" in prop and "price" in prop:
        return prop
    return None


# ==============================================================================
# Listing-card extraction
# ==============================================================================


def _extract_card_address(btn: dict) -> str | None:
    """Pull the address out of a listing card, or None when it is not there."""
    name = btn.get("name", "")
    addr_match = re.search(r"Property at (.+?), \d+ beds", name)
    return addr_match.group(1).strip() if addr_match else None


def _extract_card_beds_baths(btn: dict) -> tuple[int | None, float | None]:
    """Pull the beds and baths out of a listing card.

    Returns:
        The beds and the baths. Each one is None when it is not there.
    """
    name = btn.get("name", "")
    beds_match = re.search(r"(\d+) beds", name)
    baths_match = re.search(r"(\d+\.?\d*) baths?", name)
    beds = int(beds_match.group(1)) if beds_match else None
    baths = float(baths_match.group(1)) if baths_match else None
    return beds, baths


def _extract_card_price(btn: dict) -> str | None:
    """Pull the price out of a listing card, or None when it is not there."""
    price_text = _find_text_in_descendants(btn, r"^\$\d")
    if not price_text:
        return None
    return re.sub(r"\s+Last sold price$", "", price_text)


def _extract_card_sqft(btn: dict) -> int | None:
    """Pull the sqft out of a listing card, or None when it is not there."""
    sqft_text = _find_text_in_descendants(btn, r"sq ft")
    if not sqft_text:
        return None
    sqft_match = re.search(r"(\d[\d,]*)\s*sq ft(?!\s*lot)", sqft_text)
    if not sqft_match:
        return None
    try:
        return int(sqft_match.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_card_url(btn: dict) -> tuple[str | None, str | None]:
    """Pull the home link and its name out of a listing card.

    Returns:
        The full link and the name of the link. Both are None when the card
        has no home link.
    """
    for link in _find_all_nodes(btn, role="link"):
        url = link.get("url", "")
        if _PROPERTY_URL_SEGMENT in url:
            return _normalize_url(url), link.get("name", "")
    return None, None


def _parse_relative_date(text: str | None) -> str | None:
    """Turn text like NEW 4 HRS AGO into a YYYY-MM-DD date.

    None comes back when there is no such text.
    """
    if not text:
        return None
    now = datetime.now(timezone.utc)
    m = re.search(r"(?:NEW\s+)?(\d+)\s+HRS?\s+AGO", text, re.IGNORECASE)
    if m:
        return (now - timedelta(hours=int(m.group(1)))).strftime("%Y-%m-%d")
    m = re.search(r"(?:NEW\s+)?(\d+)\s+DAY\s+AGO", text, re.IGNORECASE)
    if m:
        return (now - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")
    m = re.search(r"(?:NEW\s+)?(\d+)\s+WEEKS?\s+AGO", text, re.IGNORECASE)
    if m:
        return (now - timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d")
    m = re.search(r"(?:NEW\s+)?(\d+)\s+MONTHS?\s+AGO", text, re.IGNORECASE)
    if m:
        return (now - timedelta(days=int(m.group(1)) * _DAYS_PER_MONTH)).strftime(
            "%Y-%m-%d"
        )
    return None


def _extract_card_dates(btn: dict, sold: bool) -> dict[str, str | None]:
    """Read the sold date or the listed date off a listing card.

    Args:
        btn: The listing card.
        sold: True to read the sold date instead of the listed date.

    Returns:
        The date the card shows, which is None when the card does not show one.
    """
    result = {}
    if sold:
        sold_date_text = _find_text_in_descendants(
            btn, r"SOLD\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}"
        )
        if not sold_date_text:
            sold_date_text = _find_text_in_descendants(
                btn, r"SOLD\s+\d{1,2}/\d{1,2}/\d{4}"
            )
        result["sold_date"] = _parse_date_text(sold_date_text) if sold_date_text else None
    else:
        listed_text = _find_text_in_descendants(
            btn, r"(?:NEW\s+)?\d+\s+(?:HR|DAY|WEEK|MONTH)"
        )
        result["listed_date"] = _parse_relative_date(listed_text) if listed_text else None
    return result


def _extract_property_from_card(btn: dict, sold: bool) -> dict | None:
    """Pull one property out of a listing card.

    Args:
        btn: The listing card.
        sold: True when the card is for a home that was sold.

    Returns:
        The property, or None when the address or the price is missing.
    """
    prop: dict = {}

    address = _extract_card_address(btn)
    if address:
        prop["address"] = address

    beds, baths = _extract_card_beds_baths(btn)
    if beds is not None:
        prop["beds"] = beds
    if baths is not None:
        prop["baths"] = baths

    price = _extract_card_price(btn)
    if price:
        prop["price"] = price

    sqft = _extract_card_sqft(btn)
    if sqft is not None:
        prop["sqft"] = sqft

    url, address_name = _extract_card_url(btn)
    if url:
        prop["url"] = url
        if "address" not in prop:
            prop["address"] = address_name

    dates = _extract_card_dates(btn, sold)
    prop.update(dates)

    prop = _finalize_property(prop)

    if sold:
        prop["status"] = "Sold"

    if "address" in prop and "price" in prop:
        return prop
    return None


# ==============================================================================
# UI interaction helpers
# ==============================================================================


async def _open_filters_panel(
    bm: BrowserManager, target_id: str
) -> PageSnapshot | None:
    """Open the Redfin Filters panel.

    Args:
        bm: Browser to work with.
        target_id: Page to work on.

    Returns:
        A snapshot of the page with the panel open, or None when it did not
        open.
    """
    snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
    if snap is None:
        return None
    filters_btn = _find_actionable(snap, role="button", name_contains="Filters")
    if not filters_btn:
        print("[Redfin] Error: Filters button not found", file=sys.stderr, flush=True)
        return None
    if not await snap.click(
        filters_btn[0]["ref_id"], timeout=_TIME_FILTER_SELECT_TIMEOUT, verbose=_VERBOSE
    ):
        return None
    return await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)


def _find_time_filter(snap: PageSnapshot, sold: bool) -> dict | None:
    """Find the time filter box in the Filters panel.

    Args:
        snap: Snapshot of the page.
        sold: True to find the time filter for sold homes.

    Returns:
        The filter box, or None when it is not on the page.
    """
    if sold:
        result = _find_actionable(snap, role="combobox", name_contains="sold")
        if result:
            return result[0]
    else:
        result = _find_actionable(snap, role="combobox", name_contains="dom")
        if result:
            return result[0]
    return None


def _select_time_filter_value(date_from: str, sold: bool) -> str:
    """Pick the value to set on the time filter.

    Args:
        date_from: Start date, YYYY-MM-DD.
        sold: True for the time filter for sold homes.

    Returns:
        The label to pick for sold homes, or date_from for active listings.
    """
    if sold:
        days_ago = (datetime.now() - datetime.strptime(date_from, "%Y-%m-%d")).days
        for max_days, label in _SOLD_WITHIN_BUCKETS:
            if days_ago <= max_days:
                return label
        return _SOLD_WITHIN_FALLBACK
    return date_from


async def _apply_time_filter(
    bm: BrowserManager,
    target_id: str,
    date_from: str,
    sold: bool,
) -> bool:
    """Set the time filter on the search.

    The Filters panel must already be open.

    Args:
        bm: Browser to work with.
        target_id: Page to work on.
        date_from: Start date, YYYY-MM-DD.
        sold: True to set the time filter for sold homes.

    Returns:
        True when the filter was set. For active listings a start date more
        than 30 days back gives back False.
    """
    if sold:
        select_value = _select_time_filter_value(date_from, sold=True)
    else:
        select_value = _date_to_dom_value(date_from)
        if not select_value:
            print(
                "[Redfin] Warning: date-from > 30 days for active listings. "
                "Redfin only supports 'Less than 30 days' max. "
                "Falling back to client-side filtering.",
                file=sys.stderr,
                flush=True,
            )
            return False

    snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
    if snap is None:
        return False
    time_filter = _find_time_filter(snap, sold)
    if not time_filter:
        print("[Redfin] Error: Time filter not found", file=sys.stderr, flush=True)
        return False

    try:
        return await snap.select_option(
            time_filter["ref_id"], value=select_value, verbose=_VERBOSE
        )
    except Exception as e:
        print(f"[Redfin] Error applying time filter: {e}", file=sys.stderr, flush=True)
        return False


async def _apply_price_filter(
    bm: BrowserManager,
    target_id: str,
    min_price: int | None,
    max_price: int | None,
) -> bool:
    """Set the price filter on the search.

    The Filters panel must already be open.

    Args:
        bm: Browser to work with.
        target_id: Page to work on.
        min_price: Lowest price to keep.
        max_price: Highest price to keep.

    Returns:
        True when at least one price box was filled in.
    """
    snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
    if snap is None:
        return False
    applied = False
    if min_price:
        min_input = _find_actionable(snap, role="textbox", name_contains="Enter min")
        if min_input:
            await snap.fill(
                min_input[0]["ref_id"], str(min_price), submit=True, verbose=_VERBOSE
            )
            applied = True

    if max_price:
        snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
        if snap is not None:
            max_input = _find_actionable(snap, role="textbox", name_contains="Enter max")
            if max_input:
                await snap.fill(
                    max_input[0]["ref_id"], str(max_price), submit=True, verbose=_VERBOSE
                )
                applied = True

    if not applied:
        print(
            "[Redfin] Error: Price input fields not found", file=sys.stderr, flush=True
        )
        return False

    return True


async def _click_filter_option(
    bm: BrowserManager,
    target_id: str,
    tree: dict,
    label: str,
    option: str,
) -> bool:
    """Click one option button under a label in the Filters panel.

    Args:
        bm: Browser to work with.
        target_id: Page to work on.
        tree: Tree of the open Filters panel.
        label: Name of the label the option sits under, like Beds.
        option: Name of the button to click, like 4 or 2+.

    Returns:
        True when the button was clicked.
    """
    button_names = _find_label_sibling_buttons(tree, label)
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

    click_snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
    if click_snap is None:
        return False
    name_to_ref = {
        b["name"]: b["ref_id"]
        for b in click_snap.actionable_elements()
        if b.get("role") == "button"
    }
    return await click_snap.click(
        name_to_ref[option], timeout=_CLICK_TIMEOUT, verbose=_VERBOSE
    )


async def _apply_beds_baths_filter(
    bm: BrowserManager,
    target_id: str,
    min_beds: int | None,
    min_baths: int | None,
) -> bool:
    """Set the beds and baths filters on the search.

    The Filters panel must already be open.

    Args:
        bm: Browser to work with.
        target_id: Page to work on.
        min_beds: Fewest bedrooms to keep.
        min_baths: Fewest bathrooms to keep.

    Returns:
        True when at least one of them was set.
    """
    snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
    if snap is None:
        return False
    tree = await snap.filtered_tree()
    if not tree:
        print("[Redfin] Error: No filtered tree available", file=sys.stderr, flush=True)
        return False

    applied = False
    options: tuple[tuple[str, int | None, Callable[[int], str]], ...] = (
        (
            "Beds",
            min_beds,
            lambda v: str(v) if v < _BEDS_OPTION_THRESHOLD else f"{v}+",
        ),
        ("Baths", min_baths, lambda v: f"{v}+"),
    )
    for label, min_val, opt_format in options:
        if not min_val:
            continue
        if not await _click_filter_option(bm, target_id, tree, label, opt_format(min_val)):
            return False
        applied = True

    if not applied:
        print(
            "[Redfin] Error: No beds/baths options applied", file=sys.stderr, flush=True
        )
        return False

    return True


async def _snapshot_results_tree(
    bm: BrowserManager, target_id: str
) -> tuple[PageSnapshot, dict]:
    """Take a picture of the results page.

    Returns:
        The snapshot and its page tree.

    Raises:
        _RedfinSearchError: When the page cannot be read.
    """
    await bm.wait_for_load(
        target_id, timeout=_DEFAULT_LOAD_TIMEOUT, verbose=_VERBOSE
    )
    snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
    if snap is None:
        raise _RedfinSearchError("could not capture the search results page")
    return snap, await snap.filtered_tree()


async def _do_initial_search(
    bm: BrowserManager, target_id: str, location: str
) -> tuple[PageSnapshot, dict]:
    """Search Redfin for a place.

    Returns:
        The snapshot of the results page and its page tree.

    Raises:
        _RedfinSearchError: When the home page cannot be read, the search box
            is not on it, or the search does not open a results page.
    """
    await bm.navigate(
        target_id, "https://www.redfin.com", timeout=_NAVIGATION_TIMEOUT, verbose=_VERBOSE
    )
    await bm.wait_for_load(
        target_id, timeout=_DEFAULT_LOAD_TIMEOUT, verbose=_VERBOSE
    )

    snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
    if snap is None:
        raise _RedfinSearchError("could not capture the Redfin homepage")
    searchbox = _find_actionable(
        snap, role="textbox", name_prefix="Search for properties"
    )
    if not searchbox:
        raise _RedfinSearchError("search box not found")
    await snap.fill(searchbox[0]["ref_id"], location, verbose=_VERBOSE)
    await snap.press_key("Enter", verbose=_VERBOSE)

    await asyncio.sleep(_POST_SEARCH_DELAY_S)
    search_page = await _snapshot_results_tree(bm, target_id)
    landing_url = bm.url(target_id, verbose=_VERBOSE)
    if landing_url and not _is_results_page(landing_url):
        raise _RedfinSearchError(
            f"Redfin did not open a results page for '{location}'"
        )
    return search_page


async def _handle_address_query(
    bm: BrowserManager,
    target_id: str,
    location: str,
    tree: dict,
) -> list[dict] | None:
    """Look up one street address and read its property.

    Args:
        bm: Browser to work with.
        target_id: Page to work on.
        location: The place that was searched for.
        tree: Tree of the page that is open.

    Returns:
        A list holding the one property, or None when the place is not a
        street address.

    Raises:
        _RedfinSearchError: When the place is a street address but its property
            page cannot be reached or read.
    """
    if not re.match(r"\d+\s+\w+", location):
        return None

    home_url = _find_home_url(tree, location)
    if home_url:
        await bm.navigate(
            target_id, home_url, timeout=_NAVIGATION_TIMEOUT, verbose=_VERBOSE
        )
        await bm.wait_for_load(
            target_id, timeout=_DEFAULT_LOAD_TIMEOUT, verbose=_VERBOSE
        )
        snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
        if snap is not None:
            tree = await snap.filtered_tree()

    tree = await _open_public_page(bm, target_id, tree)

    page_url = bm.url(target_id, verbose=_VERBOSE)
    if not _is_property_page(page_url) and not _is_owner_dashboard(page_url):
        raise _RedfinSearchError(f"no property page found for '{location}'")

    if _is_sold_page(tree):
        off_market_result = await _extract_off_market_from_page(
            bm, target_id, location
        )
        if off_market_result is not None:
            off_market_result["url"] = page_url
            return [off_market_result]

    single_prop = _extract_single_property(tree, location)
    if single_prop is not None:
        single_prop["url"] = page_url
        return [single_prop]
    raise _RedfinSearchError(
        f"could not extract property from address page for '{location}'"
    )


def _is_owner_dashboard(url: str | None) -> bool:
    """Tell whether a link points at the My Home owner dashboard."""
    return bool(url) and _OWNER_DASHBOARD_PATH in url


def _is_property_page(url: str | None) -> bool:
    """Tell whether a Redfin address points at one home's page."""
    return bool(url) and _PROPERTY_URL_SEGMENT in url


async def _wait_for_public_page(bm: BrowserManager, target_id: str) -> None:
    """Wait until the page leaves the owner dashboard.

    Waiting stops after a set number of tries.
    """
    for _ in range(_PUBLIC_PAGE_MAX_POLLS):
        if not _is_owner_dashboard(bm.url(target_id, verbose=_VERBOSE)):
            return
        await asyncio.sleep(_URL_POLL_INTERVAL_S)


async def _open_public_page(
    bm: BrowserManager, target_id: str, tree: dict
) -> dict:
    """Open the public page of a home from the owner dashboard.

    Args:
        bm: Browser to work with.
        target_id: Page to work on.
        tree: Tree of the page that is open.

    Returns:
        The tree of the public page. The given tree comes back unchanged
        when the page is not the owner dashboard.
    """
    if not _is_owner_dashboard(bm.url(target_id, verbose=_VERBOSE)):
        return tree

    snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
    if snap is None:
        return tree
    button = _find_actionable(
        snap, role="button", name_contains=_PUBLIC_PAGE_BUTTON
    )
    if not button:
        return tree

    await snap.click(button[0]["ref_id"], verbose=_VERBOSE)
    await _wait_for_public_page(bm, target_id)

    snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
    if snap is None:
        return tree
    return await snap.filtered_tree()


async def _extract_off_market_from_page(
    bm: BrowserManager,
    target_id: str,
    location: str,
) -> dict | None:
    """Read an off market property off the page that is open.

    The sale history tab is opened when the page has one.
    """
    snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
    if snap is None:
        return None
    sale_history_tab = _find_actionable(snap, role="tab", name_contains="Sale")
    if sale_history_tab:
        await snap.click(sale_history_tab[0]["ref_id"], verbose=_VERBOSE)
        await bm.wait_for_load(
            target_id, timeout=_DEFAULT_LOAD_TIMEOUT, verbose=_VERBOSE
        )

    snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
    if snap is None:
        return None
    return _extract_off_market_property(await snap.filtered_tree(), location)


def _find_home_url(tree: dict, location: str) -> str | None:
    """Find the Redfin home page link for a searched address.

    Links to rentals are skipped.

    Returns:
        The full link, or None when no link matches the address.
    """
    location_street = _extract_street(location).lower()
    location_slug = re.sub(r"[^a-z0-9]+", "-", location_street).strip("-")

    for link in _find_all_nodes(tree, role="link"):
        url = link.get("url", "")
        if (
            re.search(rf"/[A-Z]{{2}}/.+/.+{_PROPERTY_URL_SEGMENT}\d+", url, re.IGNORECASE)
            and "/rental" not in url
        ):
            if re.search(
                r"(?:^|/)" + re.escape(location_slug) + r"(?:[-/$]|$)",
                url,
                re.IGNORECASE,
            ):
                return _normalize_url(url)
    return None


async def _navigate_to_suggestion(
    bm: BrowserManager, target_id: str, tree: dict
) -> tuple[PageSnapshot | None, dict | None]:
    """Go to the zipcode or city page that Redfin suggests.

    Args:
        bm: Browser to work with.
        target_id: Page to work on.
        tree: Tree of the search page that is open.

    Returns:
        The snapshot of the suggested page and its page tree. Both are None
        when the page suggests nothing.

    Raises:
        _RedfinSearchError: When the suggested page cannot be read.
    """
    suggestion_url = None
    for link in _find_all_nodes(tree, role="link"):
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
        return None, None

    suggestion_url = _normalize_url(suggestion_url)
    await bm.navigate(
        target_id, suggestion_url, timeout=_NAVIGATION_TIMEOUT, verbose=_VERBOSE
    )
    return await _snapshot_results_tree(bm, target_id)


async def _apply_sold_filter(bm: BrowserManager, target_id: str) -> bool:
    """Turn on the sold homes filter in the Filters panel.

    Args:
        bm: Browser to work with.
        target_id: Page to work on.

    Returns:
        True when the filter was turned on.
    """
    snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
    if snap is None:
        return False
    sold_radio = _find_actionable(snap, role="radio", name_contains="Sold")
    if not sold_radio:
        print(
            "[Redfin] Error: Sold radio not found in filters panel",
            file=sys.stderr,
            flush=True,
        )
        return False
    return await snap.click(
        sold_radio[0]["ref_id"], timeout=_CLICK_TIMEOUT, verbose=_VERBOSE
    )


async def _apply_or_fail[*Ts](
    bm: BrowserManager,
    target_id: str,
    filter_fn: Callable[[BrowserManager, str, *Ts], Awaitable[bool]],
    filter_args: tuple[*Ts],
    filter_name: str,
) -> bool:
    """Try to set a filter a few times before giving up.

    There is a short wait between tries.

    Args:
        bm: Browser to work with.
        target_id: Page to work on.
        filter_fn: Call that sets the filter and gives back True when it worked.
        filter_args: The rest of the values to pass to filter_fn.
        filter_name: Name of the filter for the error message.

    Returns:
        True when the filter was set.
    """
    for attempt in range(_BROWSER_OP_RETRY_COUNT):
        if await filter_fn(bm, target_id, *filter_args):
            return True
        if attempt < _BROWSER_OP_RETRY_COUNT - 1:
            await asyncio.sleep(_RETRY_BACKOFF_S)

    print(
        f"[Redfin] Error: could not apply {filter_name} filter",
        file=sys.stderr,
        flush=True,
    )
    return False


async def _download_csv(
    bm: BrowserManager, target_id: str, save_path: str
) -> str | None:
    """Download the CSV of the search results.

    It tries a few times before giving up.

    Args:
        bm: Browser to work with.
        target_id: Page to work on.
        save_path: The file the download is told to land at.

    Returns:
        Path to the saved CSV, or None when nothing was downloaded.
    """
    for _ in range(_BROWSER_OP_RETRY_COUNT):
        snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
        if snap is None:
            continue
        download_links = _find_actionable(snap, role="link", name_contains="download")
        if not download_links:
            continue

        ref_id = download_links[0]["ref_id"]

        async def _click_download() -> bool:
            return await snap.click(ref_id, verbose=_VERBOSE)

        result = await bm.wait_for_download(
            target_id=target_id,
            action=_click_download,
            save_path=save_path,
            timeout=_CSV_DOWNLOAD_TIMEOUT,
            verbose=_VERBOSE,
        )

        if result.get("downloaded"):
            return result.get("path")

    return None


async def _apply_server_side_filters(
    bm: BrowserManager,
    target_id: str,
    options: _SearchOptions,
) -> None:
    """Open the Filters panel and set the filters Redfin can do itself.

    Args:
        bm: Browser to work with.
        target_id: Page to work on.
        options: What to look for.

    Raises:
        _RedfinSearchError: When a filter that was asked for could not be set.
    """
    if not await _open_filters_panel(bm, target_id):
        raise _RedfinSearchError("could not open the Filters panel")
    if options.sold and not await _apply_or_fail(
        bm, target_id, _apply_sold_filter, (), "sold"
    ):
        raise _RedfinSearchError("could not apply sold filter")
    if options.min_price or options.max_price:
        if not await _apply_or_fail(
            bm,
            target_id,
            _apply_price_filter,
            (options.min_price, options.max_price),
            "price",
        ):
            raise _RedfinSearchError("could not apply price filter")
    if options.date_from:
        time_filter_ok = await _apply_or_fail(
            bm,
            target_id,
            _apply_time_filter,
            (options.date_from, options.sold),
            "time",
        )
        if not time_filter_ok and options.sold:
            raise _RedfinSearchError("could not apply time filter")
        # For active listings Redfin's own time filter only goes back 30 days.
        # Older dates are filtered after the search instead.
    if options.min_beds or options.min_baths:
        if not await _apply_or_fail(
            bm,
            target_id,
            _apply_beds_baths_filter,
            (options.min_beds, options.min_baths),
            "beds/baths",
        ):
            raise _RedfinSearchError("could not apply beds/baths filter")

    snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
    if snap is not None:
        await snap.press_key("Escape", verbose=_VERBOSE)


async def _extract_and_filter_results(
    bm: BrowserManager,
    target_id: str,
    options: _SearchOptions,
) -> list[dict]:
    """Set the filters and read the properties that match.

    Args:
        bm: Browser to work with.
        target_id: Page to work on.
        options: What to look for.

    Returns:
        The properties that match, cut down to the limit.

    Raises:
        _RedfinSearchError: When a filter that was asked for could not be set.
    """
    has_filter = (
        options.sold
        or options.min_price
        or options.max_price
        or options.date_from
        or options.min_beds
        or options.min_baths
    )
    if has_filter:
        await _apply_server_side_filters(bm, target_id, options)
    return await _extract_properties(bm, target_id, options)


async def _extract_properties(
    bm: BrowserManager,
    target_id: str,
    options: _SearchOptions,
) -> list[dict]:
    """Read the properties off the results page.

    Args:
        bm: Browser to work with.
        target_id: Page to work on.
        options: What to look for.

    Returns:
        The properties that match, cut down to the limit.

    Raises:
        _RedfinSearchError: When the results page cannot be read.
    """
    csv_dir = await asyncio.to_thread(tempfile.mkdtemp, prefix="redfin_csv_")
    try:
        csv_path = await _download_csv(
            bm, target_id, os.path.join(csv_dir, "results.csv")
        )
        if csv_path:
            properties = await asyncio.to_thread(_parse_csv, csv_path)
            return _filter_properties(properties, options)
    finally:
        await asyncio.to_thread(shutil.rmtree, csv_dir, ignore_errors=True)

    snap = await bm.snapshot(target_id, timeout=_SNAPSHOT_TIMEOUT, verbose=_VERBOSE)
    if snap is None:
        raise _RedfinSearchError("could not capture the results page")
    tree = await snap.filtered_tree()
    if not tree:
        return []

    properties = []
    for btn in _find_all_nodes(tree, role="button", name_prefix=_LISTING_CARD_NAME):
        if len(properties) >= options.limit:
            break
        prop = _extract_property_from_card(btn, options.sold)
        if prop:
            properties.append(prop)

    if options.date_from or options.date_to:
        properties = _filter_by_date(
            properties, options.date_from, options.date_to
        )

    return properties[: options.limit]


# ==============================================================================
# Markdown rendering
# ==============================================================================


def _cell(value: str | Number | None) -> str:
    """Turn a value into a table cell, using a dash when the value is missing."""
    return _MISSING_CELL if value is None else str(value)


def _properties_to_markdown(properties: list[dict], location: str) -> str:
    """Turn properties into a markdown table.

    Args:
        properties: Properties to show.
        location: The place that was searched for, used in the message when
            there are no properties.

    Returns:
        A table with one row for each property, or a line saying nothing was
        found.
    """
    if not properties:
        return f"No properties found for {location}."
    lines = [
        "| " + " | ".join(_TABLE_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _TABLE_COLUMNS) + " |",
    ]
    for prop in properties:
        date = prop.get("sold_date") or prop.get("listed_date")
        cells = (
            _cell(prop.get("address")),
            _cell(prop.get("price")),
            _cell(prop.get("beds")),
            _cell(prop.get("baths")),
            _cell(prop.get("sqft")),
            _cell(prop.get("status")),
            date if date else "",
            _cell(prop.get("price_per_sqft")),
            _cell(prop.get("year_built")),
            _cell(prop.get("lot_size")),
            _cell(prop.get("url")),
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ==============================================================================
# Public API
# ==============================================================================


async def _search_impl(location: str, options: _SearchOptions) -> str:
    """Search Redfin for a place and lay the results out as a table.

    Args:
        location: Street address, city, or zipcode to look up.
        options: What to look for.

    Returns:
        A table of the properties, or a line saying nothing was found.

    Raises:
        _RedfinSearchError: When the search cannot finish or nothing could be
            read from the pages.
    """
    bm = await BrowserManager.create(verbose=_VERBOSE)
    try:
        target_id = await bm.new_page(verbose=_VERBOSE)

        _snap, tree = await _do_initial_search(bm, target_id, location)

        address_result = await _handle_address_query(bm, target_id, location, tree)
        if address_result is not None:
            return _properties_to_markdown(address_result, location)

        if not _find_all_nodes(tree, role="button", name_prefix=_LISTING_CARD_NAME):
            _snap, tree = await _navigate_to_suggestion(bm, target_id, tree)
            if tree is None:
                return f"No properties found for {location}."

        properties = await _extract_and_filter_results(bm, target_id, options)
        return _properties_to_markdown(properties, location)
    finally:
        # Closing the browser cannot hang, even when the search is stopped
        # part way through.
        await bm.cleanup(verbose=_VERBOSE)


class _RedfinSearchTool(Tool):
    name: ClassVar[str] = "redfin_search"
    is_slow: ClassVar[bool] = True
    staleness: ClassVar[timedelta] = timedelta(hours=24)
    result_render_type: ClassVar[str] = "markdown"
    description: ClassVar[str] = (
        "Search Redfin for US properties by street address, city, or zipcode, with "
        "optional filters for sold status, price, bedrooms, bathrooms, and listing "
        "dates. For a specific street address, returns that single property, "
        "including off-market and sale-history data. Use for US real-estate "
        "listings and sale history."
    )
    summary: ClassVar[str] = "US property search via Redfin"
    params: ClassVar[tuple[Param, ...]] = (
        Param("location", str, description="Search location (street address, city, or zipcode)."),
        Param(
            "sold",
            bool,
            False,
            description="If True, search recently sold properties instead of active listings.",
        ),
        Param("min_beds", int | None, None,
              description="Minimum number of bedrooms. Applied server-side via Redfin's Filters panel."),
        Param("min_baths", int | None, None,
              description="Minimum number of bathrooms. Applied server-side via Redfin's Filters panel."),
        Param("limit", int, 10, description="Maximum number of results to return."),
        Param("min_price", int | None, None,
              description="Minimum price filter. Applied server-side via Redfin's Filters panel."),
        Param("max_price", int | None, None,
              description="Maximum price filter. Applied server-side via Redfin's Filters panel."),
        Param(
            "date_from",
            str | None,
            None,
            kind="date",
            description=(
                "Start date for the date range filter. Applied server-side via 'Time on Redfin' "
                "for active listings (maximum 30 days, beyond that it falls back to "
                "client-side filtering), or via 'Sold within' for sold listings."
            ),
        ),
        Param(
            "date_to",
            str | None,
            None,
            kind="date",
            description="End date for the date range filter. Always applied client-side.",
        ),
    )
    rules: ClassVar[tuple[str, ...]] = ("order:date_from:date_to", "order:min_price:max_price")

    @override
    async def call(self, params: dict) -> ToolResult:
        location = params["location"]
        if not location or not location.strip():
            raise ToolMessageError("Redfin search failed: location must be a non-empty string")
        for name, value in (("date_from", params.get("date_from")), ("date_to", params.get("date_to"))):
            if value is not None and _parse_iso_date(value) is None:
                raise ToolMessageError(f"Redfin search failed: {name} must be a YYYY-MM-DD date: '{value}'")

        options = _SearchOptions(
            sold=params.get("sold", False),
            min_beds=params.get("min_beds"),
            min_baths=params.get("min_baths"),
            limit=params.get("limit", 10),
            min_price=params.get("min_price"),
            max_price=params.get("max_price"),
            date_from=params.get("date_from"),
            date_to=params.get("date_to"),
        )
        try:
            return ToolResult(text=await _search_impl(location, options))
        except _RedfinSearchError as e:
            raise RuntimeError(f"Redfin search failed: {e}")

    @override
    async def cache_key(self, params: dict) -> str:
        payload = json.dumps(
            {
                "location": params["location"],
                "sold": params.get("sold", False),
                "min_beds": params.get("min_beds"),
                "min_baths": params.get("min_baths"),
                "limit": params.get("limit", 10),
                "min_price": params.get("min_price"),
                "max_price": params.get("max_price"),
                "date_from": params.get("date_from"),
                "date_to": params.get("date_to"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


redfin_search_tool = _RedfinSearchTool()
