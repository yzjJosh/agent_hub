"""Tests for Redfin home page and listing card parsers."""

from __future__ import annotations

import pytest

from app.lib.redfin.property_parser import (
    extract_card_property,
    extract_off_market_property,
    extract_single_property,
    find_home_page_url,
    is_sold_property_page,
)


def test_extract_single_property_reads_status_price_specs_and_details() -> None:
    """Single property extraction pulls price, specs, details, and derived fields."""
    tree = {
        "role": "WebArea",
        "children": [
            {"role": "heading", "name": "308 Ben Howell Dr, Austin, TX 78704"},
            {"text": "For sale $1,200,000"},
            {"text": "4 bd • 3.5 ba • 2,400 sq ft"},
            {"text": "2018 Year Built"},
            {"text": "0.5 acres Lot Size"},
            {
                "role": "button",
                "name": "Property at 999 Other Rd, 2 beds",
                "children": [{"text": "$400,000"}],
            },
        ],
    }

    listing = extract_single_property(tree, "308 Ben Howell Dr, Austin, TX 78704")

    assert listing == {
        "address": "308 Ben Howell Dr, Austin, TX 78704",
        "price": "$1,200,000",
        "beds": 4,
        "baths": 3.5,
        "sqft": 2400,
        "year_built": 2018,
        "lot_size": 21780,
        "city": "Austin",
        "zip": "78704",
        "street": "308 Ben Howell Dr",
        "price_per_sqft": "$500",
    }


def test_extract_single_property_skips_payment_figures_and_returns_none() -> None:
    """A page containing only payment and tax figures yields no property."""
    tree = {
        "role": "WebArea",
        "children": [
            {"text": "Est."},
            {"text": "$6,361"},
            {"text": "/mo"},
            {"text": "HOA dues"},
            {"text": "$450"},
        ],
    }

    assert extract_single_property(tree, "123 Main St, San Jose, CA 95117") is None


@pytest.mark.parametrize(
    ("body_text", "expected_price", "expected_date", "expected_pps"),
    [
        ("Mar 5, 2025 Sold $2,075,000", "$2,075,000", "2025-03-05", "$1383"),
        ("Off Market Mar 2025 for $2M", "$2,000,000", "2025-03-01", "$1333"),
        ("Off Market Jun 15, 2024 for $500K", "$500,000", "2024-06-15", "$333"),
    ],
)
def test_extract_off_market_property_reads_exact_and_short_prices(
    body_text: str,
    expected_price: str,
    expected_date: str,
    expected_pps: str,
) -> None:
    """Off market pages parse both sale history lines and abbreviated lines."""
    tree = {
        "role": "WebArea",
        "children": [
            {"text": body_text},
            {"text": "3 bd | 2 ba | 1,500 sq ft"},
            {"text": "1961 Year Built"},
            {"text": "6,000 sq ft Lot Size"},
        ],
    }

    listing = extract_off_market_property(tree, "123 Main St, San Jose, CA 95117")

    assert listing == {
        "price": expected_price,
        "status": "Off Market",
        "sold_date": expected_date,
        "beds": 3,
        "baths": 2.0,
        "sqft": 1500,
        "year_built": 1961,
        "lot_size": 6000,
        "address": "123 Main St, San Jose, CA 95117",
        "city": "San Jose",
        "zip": "95117",
        "street": "123 Main St",
        "price_per_sqft": expected_pps,
    }


def test_extract_off_market_property_returns_none_without_price() -> None:
    """An off market page without pricing returns None."""
    tree = {"role": "WebArea", "children": [{"text": "No sale history found"}]}
    assert extract_off_market_property(tree, "123 Main St, San Jose, CA 95117") is None


def test_extract_card_property_parses_sold_listing_card() -> None:
    """Sold listing cards yield full address, specs, sold date, status, and URL."""
    card = {
        "role": "button",
        "name": "Property at 123 Main St, San Jose, CA 95117, 3 beds, 2 baths",
        "children": [
            {"text": "$750,000 Last sold price"},
            {"text": "1,500 sq ft"},
            {"text": "SOLD JUN 5, 2026"},
            {
                "role": "link",
                "name": "123 Main St",
                "url": "/CA/San-Jose/123-Main-St/home/1",
            },
        ],
    }

    listing = extract_card_property(card, is_sold=True)

    assert listing == {
        "address": "123 Main St, San Jose, CA 95117",
        "beds": 3,
        "baths": 2.0,
        "price": "$750,000",
        "sqft": 1500,
        "url": "https://www.redfin.com/CA/San-Jose/123-Main-St/home/1",
        "sold_date": "2026-06-05",
        "city": "San Jose",
        "zip": "95117",
        "street": "123 Main St",
        "price_per_sqft": "$500",
        "status": "Sold",
    }


def test_extract_card_property_returns_none_when_price_missing() -> None:
    """A listing card without a price returns None."""
    card = {
        "role": "button",
        "name": "Property at 123 Main St, San Jose, CA 95117, 3 beds, 2 baths",
        "children": [{"text": "1,500 sq ft"}],
    }
    assert extract_card_property(card, is_sold=False) is None


@pytest.mark.parametrize(
    ("tree", "expected"),
    [
        ({"role": "WebArea", "children": [{"role": "generic", "name": "SOLD MAR 5, 2025"}]}, True),
        ({"role": "WebArea", "children": [{"role": "generic", "name": "FOR SALE"}]}, False),
    ],
)
def test_is_sold_property_page(tree: dict, expected: bool) -> None:
    """Sold badge detection distinguishes sold pages from active pages."""
    assert is_sold_property_page(tree) is expected


def test_find_home_page_url_skips_rentals_and_matches_street_slug() -> None:
    """Home URL finder skips rental links and returns the full URL for the street."""
    tree = {
        "role": "WebArea",
        "children": [
            {
                "role": "link",
                "name": "Rental",
                "url": "/CA/San-Jose/123-Main-St-95117/rental/home/99",
            },
            {
                "role": "link",
                "name": "Other Home",
                "url": "/CA/San-Jose/456-Elm-St-95117/home/2",
            },
            {
                "role": "link",
                "name": "123 Main St",
                "url": "/CA/San-Jose/123-Main-St-95117/home/1",
            },
        ],
    }

    assert (
        find_home_page_url(tree, "123 Main St, San Jose, CA 95117")
        == "https://www.redfin.com/CA/San-Jose/123-Main-St-95117/home/1"
    )
    assert find_home_page_url(tree, "789 Pine Ave, San Jose, CA 95117") is None
