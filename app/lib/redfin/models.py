"""Shared data models, constants, and helpers for Redfin searches."""

from __future__ import annotations

__all__ = [
    "DEFAULT_SEARCH_LIMIT",
    "LISTING_CARD_PREFIX",
    "PROPERTY_URL_SEGMENT",
    "REDFIN_BASE_URL",
    "RedfinSearchError",
    "SearchOptions",
    "build_absolute_redfin_url",
    "parse_iso_date",
]

from dataclasses import dataclass
from datetime import datetime

DEFAULT_SEARCH_LIMIT = 10
LISTING_CARD_PREFIX = "Property at"
PROPERTY_URL_SEGMENT = "/home/"
REDFIN_BASE_URL = "https://www.redfin.com"


class RedfinSearchError(Exception):
    """Raised when a Redfin search cannot finish."""


@dataclass
class SearchOptions:
    """Options for a Redfin property search.

    Attributes:
        is_sold: True to search sold homes instead of active listings.
        min_beds: Fewest bedrooms to keep.
        min_baths: Fewest bathrooms to keep.
        limit: Most results to give back.
        min_price: Lowest price to keep.
        max_price: Highest price to keep.
        date_from: First date to keep, YYYY-MM-DD.
        date_to: Last date to keep, YYYY-MM-DD.
    """

    is_sold: bool = False
    min_beds: int | None = None
    min_baths: int | None = None
    limit: int = DEFAULT_SEARCH_LIMIT
    min_price: int | None = None
    max_price: int | None = None
    date_from: str | None = None
    date_to: str | None = None

    def has_client_filters(self) -> bool:
        """Tell whether any filter must run on the parsed listings."""
        return bool(
            self.min_beds
            or self.min_baths
            or self.min_price
            or self.max_price
            or self.date_from
            or self.date_to
        )

    def has_server_filters(self) -> bool:
        """Tell whether any filter must run in the Redfin Filters panel."""
        return bool(
            self.is_sold
            or self.min_price
            or self.max_price
            or self.date_from
            or self.min_beds
            or self.min_baths
        )


def parse_iso_date(value: str) -> datetime | None:
    """Read a YYYY-MM-DD date, or give back None when it does not fit."""
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def build_absolute_redfin_url(url: str) -> str:
    """Turn a short Redfin path into a full Redfin URL.

    A link that is already full comes back unchanged.
    """
    return url if url.startswith("http") else f"{REDFIN_BASE_URL}{url}"
