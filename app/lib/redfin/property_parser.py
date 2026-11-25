"""Extract property details from Redfin home pages and listing cards."""

from __future__ import annotations

__all__ = [
    "extract_card_property",
    "extract_off_market_property",
    "extract_single_property",
    "find_home_page_url",
    "is_sold_property_page",
]

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.lib.redfin.models import PROPERTY_URL_SEGMENT, to_absolute_redfin_url
from app.lib.redfin.snapshot_tree import (
    collect_node_text,
    find_descendant_text,
    find_matching_nodes,
    gather_node_texts,
)

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
_DAYS_PER_MONTH = 30
_PRICE_SUFFIX_MULTIPLIERS = {"M": 1_000_000, "K": 1_000}
_SQFT_PER_ACRE = 43560
_NEARBY_SUFFIX_WINDOW = 5


@dataclass(frozen=True)
class _OffMarketPricing:
    price: str | None
    sold_date: str | None


@dataclass(frozen=True)
class _PropertySpecs:
    beds: int | None
    baths: float | None
    sqft: int | None


@dataclass(frozen=True)
class _PropertyDetails:
    year_built: int | None
    lot_size: int | None


@dataclass(frozen=True)
class _BedsAndBaths:
    beds: int | None
    baths: float | None


@dataclass(frozen=True)
class _PropertyLink:
    url: str | None
    label: str | None


def _extract_street(location: str) -> str:
    """Pull the street out of a location, or return the full location."""
    street_match = re.match(r"(\d+\s+[^,]+)", location)
    return street_match.group(1).strip() if street_match else location


def _extract_address_fields(address: str | None) -> dict[str, str]:
    """Pull the city, zip, and street out of an address."""
    result: dict[str, str] = {}
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


def _compute_price_per_sqft(price_text: str, sqft: int | None) -> str | None:
    """Work out the price for each square foot, or None when missing."""
    if not price_text or sqft is None or sqft <= 0:
        return None
    cleaned = price_text.replace("$", "").replace(",", "")
    number_match = re.match(r"([\d.]+)", cleaned)
    if not number_match:
        return None
    numeric_price = float(number_match.group(1))
    if "M" in cleaned:
        price = numeric_price * _PRICE_SUFFIX_MULTIPLIERS["M"]
    elif "K" in cleaned:
        price = numeric_price * _PRICE_SUFFIX_MULTIPLIERS["K"]
    else:
        price = numeric_price
    return f"${int(price / sqft)}"


def _enrich_property_fields(listing: dict) -> dict:
    """Add city, zip, street, and price per square foot to a property."""
    result = dict(listing)
    result.update(_extract_address_fields(listing.get("address")))
    if "price" in listing and "sqft" in listing:
        price_per_sqft = _compute_price_per_sqft(listing["price"], listing["sqft"])
        if price_per_sqft:
            result["price_per_sqft"] = price_per_sqft
    return result


def _is_cost_figure(texts: list[str], index: int) -> bool:
    """Tell whether the price at index is a payment, tax, or size figure."""
    parts = [texts[index]]
    for nearby in (texts[index - 1 : index], texts[index + 1 : index + 2]):
        for piece in nearby:
            # Words naming a figure hold no digits themselves.
            if not any(char.isdigit() for char in piece):
                parts.append(piece)
    return bool(_COST_LABEL_RE.search(" ".join(parts)))


def _extract_single_pricing(tree: dict) -> str | None:
    """Pull this home's listed price out of a home details page."""
    texts = gather_node_texts(tree, should_skip_listing_cards=True)
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


def _extract_specs(full_text: str) -> _PropertySpecs:
    """Pull the beds, baths, and sqft out of page text."""
    specs_match = re.search(
        r"(\d+)\s*bd\s*[•|]\s*(\d+\.?\d*)\s*ba\s*[•|]\s*(\d{1,3}(?:,\d{3})+)\s*sq\s*ft",
        full_text,
    )
    if not specs_match:
        return _PropertySpecs(beds=None, baths=None, sqft=None)
    beds = int(specs_match.group(1))
    baths = float(specs_match.group(2))
    try:
        sqft: int | None = int(specs_match.group(3).replace(",", ""))
    except ValueError:
        sqft = None
    return _PropertySpecs(beds=beds, baths=baths, sqft=sqft)


def _extract_details(full_text: str) -> _PropertyDetails:
    """Pull the year built and lot size in square feet out of page text."""
    year_match = re.search(r"(\d{4})\s+Year\s+Built", full_text)
    year_built = int(year_match.group(1)) if year_match else None
    lot_size: int | None = None

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

    return _PropertyDetails(year_built=year_built, lot_size=lot_size)


def _fill_specs_from_text(listing: dict, full_text: str) -> None:
    """Set beds, baths, sqft, year built, and lot size from page text."""
    specs = _extract_specs(full_text)
    if specs.beds is not None:
        listing["beds"] = specs.beds
    if specs.baths is not None:
        listing["baths"] = specs.baths
    if specs.sqft is not None:
        listing["sqft"] = specs.sqft

    details = _extract_details(full_text)
    if details.year_built is not None:
        listing["year_built"] = details.year_built
    if details.lot_size is not None:
        listing["lot_size"] = details.lot_size


def _extract_full_address(tree: dict, location: str) -> str | None:
    """Pull the full address with city, state, and zip out of a page."""
    location_street = _extract_street(location).lower()
    for heading in find_matching_nodes(tree, role="heading"):
        name = heading.get("name", "")
        if location_street in name.lower() and re.search(r"\d{5}", name):
            return name.strip()
    return location.strip() if location else None


def extract_single_property(tree: dict, location: str) -> dict | None:
    """Pull one property out of a Redfin home details page.

    Args:
        tree: Tree of the page.
        location: The place that was searched for, used when the page has no
            address heading.

    Returns:
        The property, or None when the address or the price is missing.
    """
    listing: dict = {}
    full_text = collect_node_text(tree)

    full_address = _extract_full_address(tree, location)
    if full_address:
        listing["address"] = full_address

    price = _extract_single_pricing(tree)
    if price:
        listing["price"] = price

    _fill_specs_from_text(listing, full_text)
    listing = _enrich_property_fields(listing)

    if "address" in listing and "price" in listing:
        return listing
    return None


def is_sold_property_page(tree: dict) -> bool:
    """Tell whether a home page shows a home that was sold."""
    return bool(
        find_matching_nodes(
            tree, role="generic", name_pattern=r"SOLD\s+[A-Z]{3}\s+\d{1,2},\s*\d{4}"
        )
    )


def _parse_date_text(text: str | None) -> str | None:
    """Turn date text like SOLD JUN 5, 2026 into 2026-06-05, or None."""
    if not text:
        return None
    date_match = re.search(
        r"(?:SOLD(?:\s+BY\s+\w+)?\s+)?([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})", text
    )
    if date_match:
        for date_format in ("%b %d, %Y", "%B %d, %Y"):
            try:
                parsed_date = datetime.strptime(
                    f"{date_match.group(1)} {date_match.group(2)}, {date_match.group(3)}",
                    date_format,
                )
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue
    slash_match = re.search(r"(?:SOLD\s+)?(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if slash_match:
        try:
            parsed_date = datetime.strptime(
                f"{slash_match.group(1)}/{slash_match.group(2)}/{slash_match.group(3)}",
                "%m/%d/%Y",
            )
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _format_abbreviated_price(price_digits: str, price_suffix: str) -> str | None:
    """Turn abbreviated price text and its suffix into a dollar string."""
    try:
        numeric_price = float(price_digits)
        multiplier = _PRICE_SUFFIX_MULTIPLIERS.get(price_suffix.upper(), 1)
        return f"${int(numeric_price * multiplier):,}"
    except (ValueError, TypeError):
        return None


def _parse_abbreviated_sold_date(
    full_text: str, month_text: str, year_text: str
) -> str | None:
    """Read the sold date out of short off market text."""
    day_match = re.search(r"[Oo]ff\s+Market\s+\w+\s+(\d{1,2}),?\s+\d{4}", full_text)
    date_text = (
        f"{month_text} {day_match.group(1)}, {year_text}"
        if day_match
        else f"{month_text} {year_text}"
    )
    for date_format in ("%b %d, %Y", "%B %d, %Y", "%b %Y", "%B %Y"):
        try:
            return datetime.strptime(date_text.strip(), date_format).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _parse_off_market_abbreviated(full_text: str) -> _OffMarketPricing:
    """Pull the price and sold date out of short off market text."""
    price_match = re.search(
        r"[Oo]ff\s+Market\s+(\w+)\s+(?:\d{1,2},?\s+)?(\d{4})\s+for\s+\$([\d,.]+)\s*(M|K)?",
        full_text,
    )
    if not price_match:
        return _OffMarketPricing(price=None, sold_date=None)

    month_text, year_text = price_match.group(1), price_match.group(2)
    price_digits = price_match.group(3).replace(",", "")
    price_suffix = price_match.group(4) or ""
    if not price_suffix:
        nearby = full_text[price_match.end() : price_match.end() + _NEARBY_SUFFIX_WINDOW].strip()
        if nearby and nearby[0].upper() in ("M", "K"):
            price_suffix = nearby[0].upper()

    price = _format_abbreviated_price(price_digits, price_suffix)
    sold_date = _parse_abbreviated_sold_date(full_text, month_text, year_text)
    return _OffMarketPricing(price=price, sold_date=sold_date)


def _extract_off_market_pricing(full_text: str) -> _OffMarketPricing:
    """Pull the price and sold date out of an off market home page."""
    sale_history_matches = re.findall(
        r"([A-Za-z]+\s+\d{1,2},\s+\d{4})\s+Sold\s+\$([\d,]+)", full_text
    )
    if sale_history_matches:
        date_text, price_digits = sale_history_matches[0]
        return _OffMarketPricing(
            price=f"${price_digits}",
            sold_date=_parse_date_text(date_text),
        )
    return _parse_off_market_abbreviated(full_text)


def extract_off_market_property(tree: dict, location: str) -> dict | None:
    """Pull one property out of an off market home details page.

    Args:
        tree: Tree of the page.
        location: The place that was searched for, used when the page has no
            address heading.

    Returns:
        The property, or None when the price or the address is missing.
    """
    listing: dict = {}
    full_text = collect_node_text(tree)

    pricing = _extract_off_market_pricing(full_text)
    if pricing.price:
        listing["price"] = pricing.price
        listing["status"] = "Off Market"
    if pricing.sold_date:
        listing["sold_date"] = pricing.sold_date

    _fill_specs_from_text(listing, full_text)
    if "price" not in listing:
        return None

    full_address = _extract_full_address(tree, location)
    if full_address:
        listing["address"] = full_address

    listing = _enrich_property_fields(listing)
    if "address" in listing and "price" in listing:
        return listing
    return None


def _extract_card_address(card: dict) -> str | None:
    """Pull the address out of a listing card, or None."""
    name = card.get("name", "")
    address_match = re.search(r"Property at (.+?), \d+ beds", name)
    return address_match.group(1).strip() if address_match else None


def _extract_card_beds_baths(card: dict) -> _BedsAndBaths:
    """Pull the beds and baths out of a listing card."""
    name = card.get("name", "")
    beds_match = re.search(r"(\d+) beds", name)
    baths_match = re.search(r"(\d+\.?\d*) baths?", name)
    beds = int(beds_match.group(1)) if beds_match else None
    baths = float(baths_match.group(1)) if baths_match else None
    return _BedsAndBaths(beds=beds, baths=baths)


def _extract_card_price(card: dict) -> str | None:
    """Pull the price out of a listing card, or None."""
    price_text = find_descendant_text(card, r"^\$\d")
    if not price_text:
        return None
    return re.sub(r"\s+Last sold price$", "", price_text)


def _extract_card_sqft(card: dict) -> int | None:
    """Pull the square footage out of a listing card, or None."""
    sqft_text = find_descendant_text(card, r"sq ft")
    if not sqft_text:
        return None
    sqft_match = re.search(r"(\d[\d,]*)\s*sq ft(?!\s*lot)", sqft_text)
    if not sqft_match:
        return None
    try:
        return int(sqft_match.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_card_link(card: dict) -> _PropertyLink:
    """Pull the home link and its label out of a listing card."""
    for link in find_matching_nodes(card, role="link"):
        url = link.get("url", "")
        if PROPERTY_URL_SEGMENT in url:
            return _PropertyLink(
                url=to_absolute_redfin_url(url),
                label=link.get("name", ""),
            )
    return _PropertyLink(url=None, label=None)


def _parse_relative_date(text: str | None) -> str | None:
    """Turn text like NEW 4 HRS AGO into a YYYY-MM-DD date, or None."""
    if not text:
        return None
    now = datetime.now(timezone.utc)
    hours_match = re.search(r"(?:NEW\s+)?(\d+)\s+HRS?\s+AGO", text, re.IGNORECASE)
    if hours_match:
        return (now - timedelta(hours=int(hours_match.group(1)))).strftime("%Y-%m-%d")
    days_match = re.search(r"(?:NEW\s+)?(\d+)\s+DAY\s+AGO", text, re.IGNORECASE)
    if days_match:
        return (now - timedelta(days=int(days_match.group(1)))).strftime("%Y-%m-%d")
    weeks_match = re.search(r"(?:NEW\s+)?(\d+)\s+WEEKS?\s+AGO", text, re.IGNORECASE)
    if weeks_match:
        return (now - timedelta(weeks=int(weeks_match.group(1)))).strftime("%Y-%m-%d")
    months_match = re.search(r"(?:NEW\s+)?(\d+)\s+MONTHS?\s+AGO", text, re.IGNORECASE)
    if months_match:
        return (
            now - timedelta(days=int(months_match.group(1)) * _DAYS_PER_MONTH)
        ).strftime("%Y-%m-%d")
    return None


def _extract_card_dates(card: dict, is_sold: bool) -> dict[str, str | None]:
    """Read the sold date or listed date off a listing card."""
    if is_sold:
        sold_date_text = find_descendant_text(
            card, r"SOLD\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}"
        ) or find_descendant_text(card, r"SOLD\s+\d{1,2}/\d{1,2}/\d{4}")
        return {"sold_date": _parse_date_text(sold_date_text) if sold_date_text else None}
    listed_text = find_descendant_text(
        card, r"(?:NEW\s+)?\d+\s+(?:HR|DAY|WEEK|MONTH)"
    )
    return {"listed_date": _parse_relative_date(listed_text) if listed_text else None}


def extract_card_property(card: dict, is_sold: bool) -> dict | None:
    """Pull one property out of a listing card.

    Args:
        card: The listing card node.
        is_sold: True when the card is for a home that was sold.

    Returns:
        The property, or None when the address or the price is missing.
    """
    listing: dict = {}

    address = _extract_card_address(card)
    if address:
        listing["address"] = address

    beds_baths = _extract_card_beds_baths(card)
    if beds_baths.beds is not None:
        listing["beds"] = beds_baths.beds
    if beds_baths.baths is not None:
        listing["baths"] = beds_baths.baths

    price = _extract_card_price(card)
    if price:
        listing["price"] = price

    sqft = _extract_card_sqft(card)
    if sqft is not None:
        listing["sqft"] = sqft

    link = _extract_card_link(card)
    if link.url:
        listing["url"] = link.url
        if "address" not in listing:
            listing["address"] = link.label

    listing.update(_extract_card_dates(card, is_sold))
    listing = _enrich_property_fields(listing)

    if is_sold:
        listing["status"] = "Sold"

    if "address" in listing and "price" in listing:
        return listing
    return None


def find_home_page_url(tree: dict, location: str) -> str | None:
    """Find the Redfin home page link for a searched address.

    Links to rentals are skipped.

    Args:
        tree: Tree of the search page.
        location: Street address that was searched for.

    Returns:
        The full Redfin URL, or None when no link matches the address.
    """
    location_street = _extract_street(location).lower()
    location_slug = re.sub(r"[^a-z0-9]+", "-", location_street).strip("-")

    for link in find_matching_nodes(tree, role="link"):
        url = link.get("url", "")
        if (
            re.search(
                rf"/[A-Z]{{2}}/.+/.+{PROPERTY_URL_SEGMENT}\d+", url, re.IGNORECASE
            )
            and "/rental" not in url
            and re.search(
                r"(?:^|/)" + re.escape(location_slug) + r"(?:[-/$]|$)",
                url,
                re.IGNORECASE,
            )
        ):
            return to_absolute_redfin_url(url)
    return None
