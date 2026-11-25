"""Tests for Redfin shared models, constants, and URL or date helpers."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.lib.redfin.models import (
    DEFAULT_SEARCH_LIMIT,
    LISTING_CARD_PREFIX,
    PROPERTY_URL_SEGMENT,
    REDFIN_BASE_URL,
    RedfinSearchError,
    SearchOptions,
    build_absolute_redfin_url,
    parse_iso_date,
)


def test_constants_match_redfin_conventions() -> None:
    """Shared Redfin constants have the expected values."""
    assert DEFAULT_SEARCH_LIMIT == 10
    assert LISTING_CARD_PREFIX == "Property at"
    assert PROPERTY_URL_SEGMENT == "/home/"
    assert REDFIN_BASE_URL == "https://www.redfin.com"


def test_search_options_defaults_have_no_filters() -> None:
    """Default options have no client or server filters active."""
    options = SearchOptions()
    assert options.is_sold is False
    assert options.limit == DEFAULT_SEARCH_LIMIT
    assert options.has_client_filters() is False
    assert options.has_server_filters() is False


@pytest.mark.parametrize(
    ("options", "expected_client", "expected_server"),
    [
        (SearchOptions(is_sold=True), False, True),
        (SearchOptions(min_beds=3), True, True),
        (SearchOptions(min_baths=2), True, True),
        (SearchOptions(min_price=500000), True, True),
        (SearchOptions(max_price=900000), True, True),
        (SearchOptions(date_from="2026-01-01"), True, True),
        (SearchOptions(date_to="2026-06-01"), True, False),
    ],
)
def test_search_options_filter_flags(
    options: SearchOptions,
    expected_client: bool,
    expected_server: bool,
) -> None:
    """Client and server filter flags reflect the chosen option fields."""
    assert options.has_client_filters() is expected_client
    assert options.has_server_filters() is expected_server


@pytest.mark.parametrize(
    ("date_text", "expected"),
    [
        ("2026-06-15", datetime(2026, 6, 15)),
        ("2024-02-29", datetime(2024, 2, 29)),
        ("not-a-date", None),
        ("2026/06/15", None),
        ("", None),
    ],
)
def test_parse_iso_date(date_text: str, expected: datetime | None) -> None:
    """ISO dates parse into datetimes and invalid strings return None."""
    assert parse_iso_date(date_text) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("/CA/San-Jose/123-Main-St/home/1", "https://www.redfin.com/CA/San-Jose/123-Main-St/home/1"),
        ("https://www.redfin.com/zipcode/95117", "https://www.redfin.com/zipcode/95117"),
        ("http://example.com/home/1", "http://example.com/home/1"),
    ],
)
def test_build_absolute_redfin_url(url: str, expected: str) -> None:
    """Relative Redfin paths gain the base host and full URLs stay unchanged."""
    assert build_absolute_redfin_url(url) == expected


def test_redfin_search_error_preserves_message() -> None:
    """RedfinSearchError carries its failure message."""
    error = RedfinSearchError("search box not found")
    assert str(error) == "search box not found"
