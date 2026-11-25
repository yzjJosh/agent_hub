"""Tests for Redfin CSV parsing and property filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.lib.redfin.csv_parser import (
    filter_properties,
    filter_properties_by_date,
    parse_results_csv,
)
from app.lib.redfin.models import RedfinSearchError, SearchOptions

_CSV_HEADER = (
    '"SALE TYPE","PRICE","BEDS","BATHS","ADDRESS","CITY","STATE OR PROVINCE",'
    '"ZIP OR POSTAL CODE","SQUARE FEET","LOT SIZE","YEAR BUILT","DAYS ON MARKET",'
    '"$/SQUARE FEET","HOA/MONTH","STATUS","PROPERTY TYPE","SOLD DATE",'
    '"NEXT OPEN HOUSE START TIME","NEXT OPEN HOUSE END TIME","URL","SOURCE",'
    '"MLS#","LATITUDE","LONGITUDE"\n'
)


async def test_parse_results_csv_parses_rows_and_skips_mls_notice(
    tmp_path: Path,
) -> None:
    """CSV parser builds structured listings and drops MLS notice rows."""
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        _CSV_HEADER
        + '"In accordance with local MLS rules, some MLS listings are not included","","","","",'
        '"","","","","","","","","","","","","","","","","","",""\n'
        + '"MLS Listing","650000","3","2.5","100 Main St","San Jose","CA","95117",'
        '"1400","5000","1998","4","464","350","Active","Single Family","",'
        '"June-05-2026 02:00 PM","June-05-2026 04:00 PM","https://www.redfin.com/home/1",'
        '"MLS","ML1","37.3","-121.9"\n'
        + '"PAST SALE","820000","4","3","200 Oak Ave","San Jose","CA","95117",'
        '"1800","","2005","10","456","","Sold","Single Family","March-10-2026",'
        '"","","https://www.redfin.com/home/2","MLS","ML2","37.4","-121.8"\n',
        encoding="utf-8",
    )

    listings = await parse_results_csv(str(csv_path))

    assert len(listings) == 2
    assert listings[0]["address"] == "100 Main St, San Jose, CA, 95117"
    assert listings[0]["price"] == "$650,000"
    assert listings[0]["beds"] == 3
    assert listings[0]["baths"] == 2.5
    assert listings[0]["sqft"] == 1400
    assert listings[0]["price_per_sqft"] == "$464"
    assert listings[0]["status"] == "Active"
    assert listings[0]["open_house_start"] == "2026-06-05T14:00:00"
    assert listings[0]["open_house_end"] == "2026-06-05T16:00:00"
    assert listings[1]["status"] == "Sold"
    assert listings[1]["sold_date"] == "2026-03-10"


async def test_parse_results_csv_raises_on_non_utf8_file(tmp_path: Path) -> None:
    """CSV files that cannot decode as UTF-8 raise RedfinSearchError."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_bytes(b"\xff\xfe\x00\x00")

    with pytest.raises(RedfinSearchError, match="could not read the results CSV as text"):
        await parse_results_csv(str(csv_path))


def test_filter_properties_applies_numeric_price_and_limit_bounds() -> None:
    """Property filtering drops listings outside beds, baths, and price bounds."""
    listings: list[dict] = [
        {"address": "A", "price": "$500,000", "beds": 2, "baths": 2.0, "status": "Active"},
        {"address": "B", "price": None, "beds": 4, "baths": 3.0, "status": "Active"},
        {"address": "C", "price": "$750,000", "beds": 3, "baths": 2.0, "status": "Active"},
        {"address": "D", "price": "$850,000", "beds": 4, "baths": 3.0, "status": "Active"},
        {"address": "E", "price": "$1,200,000", "beds": 5, "baths": 4.0, "status": "Active"},
    ]
    options = SearchOptions(
        min_beds=3,
        min_baths=2,
        min_price=600000,
        max_price=900000,
        limit=1,
    )

    kept = filter_properties(listings, options)

    assert [item["address"] for item in kept] == ["C"]


def test_filter_properties_without_filters_only_applies_limit() -> None:
    """When no filter is set, properties are returned up to the limit."""
    listings = [{"address": "A"}, {"address": "B"}, {"address": "C"}]

    kept = filter_properties(listings, SearchOptions(limit=2))

    assert kept == [{"address": "A"}, {"address": "B"}]


def test_filter_properties_by_date_keeps_in_range_and_undated_active() -> None:
    """Date filter drops out-of-range listings and undated sold homes, keeping undated active."""
    listings = [
        {"address": "Old Sold", "status": "Sold", "sold_date": "2025-01-10"},
        {"address": "In Range", "status": "Sold", "sold_date": "2026-02-15"},
        {"address": "Too Late", "status": "Sold", "sold_date": "2026-05-01"},
        {"address": "Undated Sold", "status": "Sold"},
        {"address": "Undated Active", "status": "Active"},
    ]

    kept = filter_properties_by_date(listings, "2026-01-01", "2026-03-31")

    assert [item["address"] for item in kept] == ["In Range", "Undated Active"]


@pytest.mark.parametrize(
    ("date_from", "date_to", "error_pattern"),
    [
        ("bad-date", None, "invalid date_from"),
        (None, "2026/05/01", "invalid date_to"),
    ],
)
def test_filter_properties_by_date_rejects_invalid_bounds(
    date_from: str | None,
    date_to: str | None,
    error_pattern: str,
) -> None:
    """Invalid ISO bounds raise RedfinSearchError."""
    with pytest.raises(RedfinSearchError, match=error_pattern):
        filter_properties_by_date([{"address": "A"}], date_from, date_to)
