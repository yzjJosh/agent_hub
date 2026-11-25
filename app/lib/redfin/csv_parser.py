"""Read Redfin CSV downloads and filter property records."""

from __future__ import annotations

__all__ = [
    "filter_properties",
    "filter_properties_by_date",
    "parse_results_csv",
]

import asyncio
import csv
from datetime import datetime

from app.lib.redfin.models import RedfinSearchError, SearchOptions, parse_iso_date

_VERBOSE = True
_REDFIN_URL_CSV_COLUMN = (
    "URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis"
    " FOR INFO ON PRICING)"
)
_PROPERTY_DATE_FORMATS = ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%m/%d/%Y")


def _parse_csv_date(date_text: str) -> str | None:
    """Turn a CSV date like June-05-2026 into 2026-06-05, or None."""
    if not date_text:
        return None
    try:
        parsed_date = datetime.strptime(date_text, "%B-%d-%Y")
        return parsed_date.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_csv_datetime(datetime_text: str) -> str | None:
    """Turn a CSV date and time into an ISO timestamp, or None."""
    if not datetime_text:
        return None
    try:
        parsed_date = datetime.strptime(datetime_text, "%B-%d-%Y %I:%M %p")
        return parsed_date.strftime("%Y-%m-%dT%H:%M:%S")
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


def _format_price(price_text: str) -> str | None:
    """Turn a plain price number into text like $1,234,567, or None."""
    amount = _parse_int_or_none(price_text)
    if amount is None:
        return None
    return f"${amount:,}"


def _clean_text(value: str) -> str | None:
    """Trim spaces off text, or give back None when nothing is left."""
    if not value:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _parse_csv_numeric_fields(row: dict[str, str]) -> dict[str, int | float | None]:
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


def _build_csv_address(row: dict[str, str]) -> str:
    """Join the address parts of a CSV row into one address."""
    parts = [
        _clean_text(row.get("ADDRESS", "")) or "",
        _clean_text(row.get("CITY", "")) or "",
        _clean_text(row.get("STATE OR PROVINCE", "")) or "",
        _clean_text(row.get("ZIP OR POSTAL CODE", "")) or "",
    ]
    return ", ".join(part for part in parts if part)


def _parse_csv_row(row: dict[str, str]) -> dict:
    """Turn one CSV row into a property dictionary."""
    sale_type = _clean_text(row.get("SALE TYPE", ""))
    status = "Sold" if sale_type == "PAST SALE" else "Active"
    price = _format_price(row.get("PRICE", ""))
    numeric = _parse_csv_numeric_fields(row)
    price_per_sqft_value = _parse_int_or_none(row.get("$/SQUARE FEET", ""))
    price_per_sqft = f"${price_per_sqft_value}" if price_per_sqft_value else None
    address = _build_csv_address(row)

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
        "property_type": _clean_text(row.get("PROPERTY TYPE", "")),
        "sale_type": sale_type,
        "sold_date": _parse_csv_date(row.get("SOLD DATE", "")),
        "open_house_start": _parse_csv_datetime(
            row.get("NEXT OPEN HOUSE START TIME", "")
        ),
        "open_house_end": _parse_csv_datetime(row.get("NEXT OPEN HOUSE END TIME", "")),
        "url": _clean_text(row.get("URL", "") or row.get(_REDFIN_URL_CSV_COLUMN, "")),
        "source": _clean_text(row.get("SOURCE", "")),
        "mls_id": _clean_text(row.get("MLS#", "")),
        "latitude": numeric["latitude"],
        "longitude": numeric["longitude"],
        "zip": _clean_text(row.get("ZIP OR POSTAL CODE", "")),
        "street": _clean_text(row.get("ADDRESS", "")),
    }


def _parse_results_csv_sync(path: str) -> list[dict]:
    """Read a Redfin CSV file from disk into property records."""
    properties: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                sale_type = row.get("SALE TYPE", "").strip()
                if "MLS rules" in sale_type:
                    continue
                properties.append(_parse_csv_row(dict(row)))
    except UnicodeDecodeError:
        raise RedfinSearchError("could not read the results CSV as text") from None
    return properties


async def parse_results_csv(path: str) -> list[dict]:
    """Read a Redfin CSV file into property records.

    Rows that carry the MLS rules notice are left out.

    Args:
        path: Path to the CSV file.

    Returns:
        One property for each data row.

    Raises:
        RedfinSearchError: When the file is not UTF-8 text.
    """
    return await asyncio.to_thread(_parse_results_csv_sync, path)


def _parse_price_int(price_text: str | None) -> int | None:
    """Turn price text like $1,234,567 into a whole number, or None."""
    if not price_text:
        return None
    cleaned = price_text.replace("$", "").replace(",", "")
    try:
        return int(cleaned)
    except ValueError:
        return None


def _matches_property_criteria(listing: dict, options: SearchOptions) -> bool:
    """Tell whether a property passes the beds, baths, and price filters."""
    if options.min_beds and (listing.get("beds") or 0) < options.min_beds:
        return False
    if options.min_baths and (listing.get("baths") or 0) < options.min_baths:
        return False
    listing_price = _parse_price_int(listing.get("price"))
    if (options.min_price or options.max_price) and listing_price is None:
        return False
    if options.min_price and listing_price is not None and listing_price < options.min_price:
        return False
    if options.max_price and listing_price is not None and listing_price > options.max_price:
        return False
    return True


def _parse_property_date(listing: dict) -> datetime | None:
    """Read the sold date, listed date, or plain date of a property."""
    date_text = listing.get("sold_date") or listing.get("listed_date") or listing.get("date")
    if not date_text:
        return None
    for date_format in _PROPERTY_DATE_FORMATS:
        try:
            return datetime.strptime(date_text.strip(), date_format)
        except ValueError:
            continue
    return None


def _require_iso_date(date_text: str | None, field_name: str) -> datetime | None:
    """Parse an optional YYYY-MM-DD bound, raising when it is invalid."""
    if not date_text:
        return None
    parsed_date = parse_iso_date(date_text)
    if parsed_date is None:
        raise RedfinSearchError(f"invalid {field_name}: {date_text}")
    return parsed_date


def filter_properties_by_date(
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
        RedfinSearchError: When a date bound is not YYYY-MM-DD.
    """
    if not date_from and not date_to:
        return properties

    from_date = _require_iso_date(date_from, "date_from")
    to_date = _require_iso_date(date_to, "date_to")

    filtered: list[dict] = []
    for listing in properties:
        listing_date = _parse_property_date(listing)
        if listing_date is None:
            if listing.get("status") == "Active":
                filtered.append(listing)
            continue
        if from_date and listing_date < from_date:
            continue
        if to_date and listing_date > to_date:
            continue
        filtered.append(listing)

    if _VERBOSE:
        print(
            f"[Redfin] Date filter: {len(properties)} -> {len(filtered)} properties",
            flush=True,
        )
    return filtered


def filter_properties(
    properties: list[dict],
    options: SearchOptions,
) -> list[dict]:
    """Keep only the properties that match the search options.

    A property with no readable price is dropped when a price bound is set.

    Args:
        properties: Properties to look through.
        options: Search filters and result limit.

    Returns:
        The matching properties, cut down to the limit.
    """
    if not options.has_client_filters():
        return properties[: options.limit]

    filtered = [
        listing for listing in properties if _matches_property_criteria(listing, options)
    ]
    if options.date_from or options.date_to:
        filtered = filter_properties_by_date(
            filtered, options.date_from, options.date_to
        )
    return filtered[: options.limit]
