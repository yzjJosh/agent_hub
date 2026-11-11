from __future__ import annotations

"""Tests for the redfin_search tool.

Every test goes through the tool's run interface, and only the browser is faked.
"""

import os
import shutil
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.redfin import redfin_search_tool
from tests.app.conftest import skip_if_no_chrome
from tests.app.lib.tool.conftest import await_task_result, run_text

_NO_RESULTS_PREFIX = "No properties found for "
_FAILED_PREFIX = "Redfin search failed: "
_LIVE_SEARCH_LIMIT = 1000
_PROPERTY_URL_SEGMENT = "/home/"
_DASHBOARD_URL = "https://www.redfin.com/myredfin/owner-dashboard/25424192"
_PUBLIC_HOME_URL = "https://www.redfin.com/CA/San-Jose/123-Main-St-95117/home/751580"
_HOMEPAGE_URL = "https://www.redfin.com/"
_RESULTS_URL = "https://www.redfin.com/zipcode/95117"
_FILTER_PAGE_URL = "https://www.redfin.com/city/CA/San-Jose/filter/property-type=house/"
_NOT_A_PROPERTY_URL = "https://www.redfin.com/sitemap"

_SEARCH_ELEMENTS = [
    {"ref_id": "1", "role": "textbox", "name": "Search for properties"},
    {"ref_id": "2", "role": "button", "name": "submit search"},
]

_EMPTY_TREE = {"role": "WebArea", "children": []}

# One tile of the "More homes you might like" carousel on a home details page.
_SIMILAR_HOME_CARD = (
    "4316 Companion Hill Rd",
    "Austin, TX 78749",
    4,
    3,
    "$500,000",
    "2,100",
    "/TX/Austin/4316-Companion-Hill-Rd-78749/home/20056130",
)

# The money on a home page that is not the list price, in page order.
_PAGE_COST_NODES = [
    {"text": "Est."},
    {"text": "$6,361"},
    {"text": "/mo —"},
    {"text": "Payment calculator"},
    {"text": "$6,361 per month"},
    {"text": "Property taxes"},
    {"text": "$1,515"},
    {"text": "HOA dues"},
    {"text": "$2,120"},
    {"role": "generic", "name": "$439", "children": [{"text": "$439"}]},
    {"text": "Price/Sq.Ft."},
]

_CSV_HEADER = (
    '"SALE TYPE","PRICE","BEDS","BATHS","ADDRESS","CITY","STATE OR PROVINCE",'
    '"ZIP OR POSTAL CODE","SQUARE FEET","LOT SIZE","YEAR BUILT","DAYS ON MARKET",'
    '"$/SQUARE FEET","HOA/MONTH","STATUS","PROPERTY TYPE","SOLD DATE",'
    '"NEXT OPEN HOUSE START TIME","NEXT OPEN HOUSE END TIME","URL","SOURCE",'
    '"MLS#","FAVORITE","INTERESTED","LATITUDE","LONGITUDE","LOCATION"\n'
)


def _parse_table(md: str) -> list[dict]:
    """Turn a markdown table into one dict per row, keyed by header. A dash cell becomes None."""
    lines = [line for line in md.strip().splitlines() if line.strip()]
    headers = [cell.strip() for cell in lines[0].strip().strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(
            {header: (None if cell == "—" else cell) for header, cell in zip(headers, cells)}
        )
    return rows


def _make_snap(actionable: list[dict], tree: dict | None = None) -> MagicMock:
    """Return a fake page snapshot with the given clickable elements and tree."""
    snap = MagicMock()
    snap.actionable_elements.return_value = actionable
    snap.filtered_tree = AsyncMock(return_value=tree)
    snap.click = AsyncMock(return_value=True)
    snap.fill = AsyncMock(return_value=True)
    snap.select_option = AsyncMock(return_value=True)
    snap.press_key = AsyncMock(return_value=True)
    return snap


def _url_values(url_sequence: list[str] | None) -> Callable[..., str]:
    """Return a function that gives back each address in turn, repeating the last one."""
    values = list(url_sequence) if url_sequence else [""]
    index = 0

    def url(_target_id: str, **_kwargs: bool) -> str:
        nonlocal index
        value = values[min(index, len(values) - 1)]
        index += 1
        return value

    return url


def _mock_bm(
    snapshot_sequence: list[MagicMock] | None = None,
    csv_path: str | None = None,
    repeat_snapshot: MagicMock | None = None,
    url_sequence: list[str] | None = None,
) -> MagicMock:
    """Return a fake browser.

    Args:
        snapshot_sequence: Snapshots given out in order.
        csv_path: File a download reports. Without it, downloads fail.
        repeat_snapshot: One snapshot given out every time.
        url_sequence: Page addresses given out in order, the last one repeating.
    """
    mock_bm = MagicMock()
    mock_bm.new_page = AsyncMock(return_value="page1")
    mock_bm.navigate = AsyncMock(return_value=True)
    mock_bm.wait_for_load = AsyncMock(return_value=True)
    if repeat_snapshot is not None:
        mock_bm.snapshot = AsyncMock(return_value=repeat_snapshot)
    else:
        mock_bm.snapshot = AsyncMock(side_effect=snapshot_sequence)
    download_result = (
        {"downloaded": True, "path": csv_path} if csv_path else {"downloaded": False}
    )
    mock_bm.wait_for_download = AsyncMock(return_value=download_result)
    mock_bm.cleanup = AsyncMock()
    mock_bm.url = MagicMock(side_effect=_url_values(url_sequence))
    return mock_bm


async def _run_search(mock_bm: MagicMock, location: str, **kwargs: int | str | bool) -> str:
    """Run the tool with the fake browser and return the finished result."""
    mock_cls = MagicMock()
    mock_cls.create = AsyncMock(return_value=mock_bm)
    patcher = patch("app.tools.redfin.BrowserManager", mock_cls)
    patcher.start()
    try:
        task = await redfin_search_tool.run({"location": location, **kwargs})
        return await await_task_result(str(task.id))
    finally:
        patcher.stop()


# How long a live search may take before the test gives up.
_LIVE_SEARCH_WAIT_S = 240.0


async def _live_search(location: str, **kwargs: int | str | bool) -> str:
    """Run the tool for real and return the finished result or the saved error."""
    task = await redfin_search_tool.run({"location": location, **kwargs})
    return await await_task_result(str(task.id), timeout_s=_LIVE_SEARCH_WAIT_S)


def _single_row(result: str) -> dict:
    """Check the result is a table with one row and return that row."""
    rows = _parse_table(result)
    assert len(rows) == 1
    return rows[0]


async def _search_address_page(details_tree: dict, location: str, page_url: str) -> str:
    """Search one address with a fake browser that opens the home page directly."""
    home_snap = _make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE)
    mock_bm = _mock_bm(
        [home_snap, _make_snap([], details_tree)],
        url_sequence=[f"https://www.redfin.com{page_url}"],
    )
    return await _run_search(mock_bm, location)


def _price_to_int(price: str | None) -> int:
    """Turn a price cell into a whole number. An empty cell becomes 0."""
    if not price:
        return 0
    return int(price.replace("$", "").replace(",", ""))


def _assert_table(result: str, label: str) -> list[dict]:
    """Check the result is a table with rows and return them.

    A search that failed or found nothing fails the test.
    """
    assert not result.startswith(_FAILED_PREFIX), f"{label}: {result}"
    assert not result.startswith(_NO_RESULTS_PREFIX), f"{label}: {result}"
    rows = _parse_table(result)
    assert rows, f"{label}: expected a markdown table with rows, got: {result[:200]}"
    return rows


def _sqft_to_int(sqft: str | None) -> int:
    """Turn a square feet cell into a whole number. An empty cell becomes 0."""
    if not sqft:
        return 0
    return int(sqft.replace(",", ""))


def _assert_property_url(url: str) -> None:
    """Check the address points at the public home page."""
    assert url.startswith("https://www.redfin.com"), url
    assert _PROPERTY_URL_SEGMENT in url, url
    assert "/myredfin/" not in url, url


def _assert_all_rows(rows: list[dict], row_check: Callable[[dict], None]) -> None:
    """Check every row. The first bad row fails the test."""
    for row in rows:
        row_check(row)


def _assert_listing_row(row: dict) -> None:
    """Check that one area search row has a good address, price, beds, baths, size, and link."""
    assert row["Address"], f"empty Address in row: {row}"
    assert (
        row["Price"] is not None and row["Price"].startswith("$")
    ), f"bad Price: {row['Price']!r}"
    assert int(row["Beds"]) > 0, f"bad Beds: {row['Beds']!r}"
    assert float(row["Baths"]) > 0, f"bad Baths: {row['Baths']!r}"
    assert _sqft_to_int(row["Sqft"]) > 0, f"bad Sqft: {row['Sqft']!r}"
    assert row["URL"].startswith("https://www.redfin.com"), row["URL"]


def _assert_active_filter_row(row: dict, date_from: str) -> None:
    """Check that one for sale row meets the beds, baths, price, and date limits."""
    assert int(row["Beds"]) >= 4, f"bad Beds: {row['Beds']!r}"
    assert float(row["Baths"]) >= 3, f"bad Baths: {row['Baths']!r}"
    assert _price_to_int(row["Price"]) >= 2_000_000, f"bad Price: {row['Price']!r}"
    assert not (row["Date"] and row["Date"] < date_from), (
        f"Date {row['Date']!r} before cutoff {date_from}"
    )


def _assert_sold_row(
    row: dict,
    date_from: str,
    min_beds: int | None = None,
    min_baths: float | None = None,
    min_price: int | None = None,
    date_to: str | None = None,
) -> None:
    """Check that one sold row meets the status, beds, baths, price, and date limits."""
    assert row["Status"] == "Sold", f"bad Status: {row['Status']!r}"
    if min_beds is not None:
        assert int(row["Beds"]) >= min_beds, f"bad Beds: {row['Beds']!r}"
    if min_baths is not None:
        assert float(row["Baths"]) >= min_baths, f"bad Baths: {row['Baths']!r}"
    if min_price is not None:
        assert _price_to_int(row["Price"]) >= min_price, f"bad Price: {row['Price']!r}"
    assert row["Date"], f"missing Date in sold row: {row['Address']}"
    assert row["Date"] >= date_from, f"Date {row['Date']} before {date_from}"
    if date_to is not None:
        assert row["Date"] <= date_to, f"Date {row['Date']} after {date_to}"


def _make_listing_tree(addresses_prices: list[tuple]) -> dict:
    """Return a results page tree with one card per listing.

    Args:
        addresses_prices: One tuple per listing, holding the address, the city
            with state and zip, beds, baths, price, size, and link.
    """
    children = []
    for addr, city_zip, beds, baths, price, sqft, url in addresses_prices:
        full_addr = f"{addr}, {city_zip}"
        btn = {
            "role": "button",
            "name": f"Property at {full_addr}, {beds} beds, {baths} baths",
            "children": [
                {"text": price},
                {"text": f"{sqft} sq ft"},
                {"role": "link", "name": addr, "url": url},
            ],
        }
        children.append(btn)
    return {"role": "WebArea", "children": children}


def _mock_bm_for_web(addresses_prices: list[tuple]) -> MagicMock:
    """Return a fake browser whose download fails and whose results page shows
    the given listings.
    """
    tree = _make_listing_tree(addresses_prices)
    home_snap = _make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE)
    results_snap = _make_snap([], tree)
    return _mock_bm([home_snap, *([results_snap] * 6)])


def _csv_filter_actionable(
    with_time_filter: bool,
    with_sold_filter: bool,
    with_price_filter: bool,
) -> list[dict]:
    """Return the clickable elements of the filter panel."""
    filter_actionable: list[dict] = []
    if with_time_filter:
        if with_sold_filter:
            filter_actionable.extend(
                [
                    {"ref_id": "sort-cb", "role": "combobox", "name": "Sort"},
                    {"ref_id": "type-cb", "role": "combobox", "name": "Home type"},
                    {"ref_id": "time-filter", "role": "combobox", "name": "Sold within"},
                ]
            )
        else:
            filter_actionable.extend(
                [
                    {"ref_id": "sort-cb", "role": "combobox", "name": "Sort"},
                    {"ref_id": "type-cb", "role": "combobox", "name": "Home type"},
                    {"ref_id": "time-filter", "role": "combobox", "name": "dom"},
                ]
            )
    if with_sold_filter:
        filter_actionable.append(
            {"ref_id": "sold-radio", "role": "radio", "name": "Sold"}
        )
    if with_price_filter:
        filter_actionable.append(
            {"ref_id": "min-price", "role": "textbox", "name": "Enter min price"}
        )
        filter_actionable.append(
            {"ref_id": "max-price", "role": "textbox", "name": "Enter max price"}
        )
    filter_actionable.extend(
        [
            {"ref_id": "beds-3", "role": "button", "name": "3"},
            {"ref_id": "beds-4", "role": "button", "name": "4"},
            {"ref_id": "beds-5", "role": "button", "name": "5"},
            {"ref_id": "baths-2", "role": "button", "name": "2+"},
            {"ref_id": "baths-3", "role": "button", "name": "3+"},
        ]
    )
    return filter_actionable


_CSV_FILTER_TREE = {
    "role": "WebArea",
    "children": [
        {"name": "Beds", "role": "label"},
        {"role": "button", "name": "3"},
        {"role": "button", "name": "4"},
        {"name": "Baths", "role": "label"},
        {"role": "button", "name": "2+"},
    ],
}


def _mock_bm_for_csv(
    csv_path: str,
    with_time_filter: bool = False,
    with_beds_baths_filter: bool = False,
    with_sold_filter: bool = False,
    with_price_filter: bool = False,
) -> MagicMock:
    """Return a fake browser whose results page offers the given file to download.

    Args:
        csv_path: File the download reports.
        with_time_filter: Show the time filter in the filter panel.
        with_beds_baths_filter: Show the beds and baths buttons.
        with_sold_filter: Show the sold choice.
        with_price_filter: Show the price boxes.
    """
    results_tree = {
        "role": "WebArea",
        "children": [
            {"role": "button", "name": "Property at 123 Main St, 3 beds, 2 baths"}
        ],
    }
    home_snap = _make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE)
    results_snap = _make_snap([], results_tree)
    panel_snap = _make_snap(
        [{"ref_id": "filters", "role": "button", "name": "Filters"}], results_tree
    )

    filter_snap = _make_snap(
        _csv_filter_actionable(with_time_filter, with_sold_filter, with_price_filter),
        _CSV_FILTER_TREE,
    )

    download_snap = _make_snap(
        [
            *_SEARCH_ELEMENTS,
            {"ref_id": "download-btn", "role": "link", "name": "Download results"},
        ],
        results_tree,
    )

    # Price and beds with baths each take two snapshots, every other filter takes one.
    num_filter_snaps = (
        int(with_time_filter) + 2 * int(with_beds_baths_filter)
        + int(with_sold_filter) + 2 * int(with_price_filter)
    )

    sequence = [home_snap, results_snap]
    if num_filter_snaps:
        sequence.extend([panel_snap, filter_snap])
        sequence.extend([filter_snap] * num_filter_snaps)
        sequence.append(filter_snap)
    sequence.append(download_snap)
    return _mock_bm(sequence, csv_path=csv_path)


def _mock_bm_for_off_market(
    off_market_text: str,
    address: str = "123 Main St, San Jose, CA 95117",
    heading_price: str | None = None,
) -> MagicMock:
    """Return a fake browser that lands on the details page of an off market home.

    Args:
        off_market_text: Text shown on the details page.
        address: The address being searched.
        heading_price: Price text shown at the top of the page.
    """
    home_snap = _make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE)
    heading_children = [
        {"text": address},
        {"role": "button", "name": "Edit Facts"},
        {"role": "heading", "name": "Redfin Estimate"},
    ]
    if heading_price:
        heading_children.append({"text": f"{heading_price} {address}"})
    details_tree = {
        "role": "WebArea",
        "children": [
            *heading_children,
            {"text": off_market_text},
            {"text": "3 bd | 2 ba | 1,500 sq ft"},
        ],
    }
    search_snap = _make_snap(
        [{"ref_id": "edit-btn", "role": "button", "name": "Edit Facts"}], details_tree
    )
    return _mock_bm([home_snap, search_snap], url_sequence=[_PUBLIC_HOME_URL])


def _mock_bm_for_owner_dashboard(
    off_market_text: str, address: str = "123 Main St, San Jose, CA 95117"
) -> tuple[MagicMock, MagicMock]:
    """Return a fake browser that lands on the owner dashboard of a claimed home.

    The dashboard carries a button that opens the public home page.

    Args:
        off_market_text: Sale history text on the public home page.
        address: The address being searched.

    Returns:
        The fake browser and the dashboard snapshot the tool must click.
    """
    home_snap = _make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE)
    dashboard_tree = {
        "role": "WebArea",
        "children": [
            {"role": "heading", "name": "My Home"},
            {"text": address},
        ],
    }
    dashboard_snap = _make_snap(
        [{"ref_id": "view-public", "role": "button", "name": "View public page"}],
        dashboard_tree,
    )
    public_tree = {
        "role": "WebArea",
        "children": [
            {"text": address},
            {"role": "generic", "name": "SOLD MAR 5, 2025"},
            {"text": off_market_text},
            {"text": "3 bd | 2 ba | 1,500 sq ft"},
        ],
    }
    public_snap = _make_snap(
        [{"ref_id": "sale-tab", "role": "tab", "name": "Sale & tax history"}],
        public_tree,
    )
    mock_bm = _mock_bm(
        [
            home_snap,
            dashboard_snap,
            dashboard_snap,
            public_snap,
            public_snap,
            public_snap,
        ],
        url_sequence=[_DASHBOARD_URL, _DASHBOARD_URL, _PUBLIC_HOME_URL],
    )
    return mock_bm, dashboard_snap


# ==============================================================================
# Tool shape
# ==============================================================================


class TestRedfinShape:
    """The tool declares the search surface and nothing else."""

    def test_tool_shape(self) -> None:
        assert redfin_search_tool.name == "redfin_search"
        assert redfin_search_tool.is_slow is True
        assert redfin_search_tool.staleness == timedelta(hours=24)
        assert redfin_search_tool.result_render_type == "markdown"
        assert redfin_search_tool.summary
        assert "\n" not in redfin_search_tool.summary
        assert len(redfin_search_tool.summary) <= 80


# ==============================================================================
# Default values
# ==============================================================================


def test_search_defaults_match_contract() -> None:
    """The tool's options have the expected default values."""
    defaults = {p.name: p.default for p in redfin_search_tool.params}
    assert defaults["sold"] is False
    assert defaults["min_beds"] is None
    assert defaults["min_baths"] is None
    assert defaults["limit"] == 10
    assert defaults["min_price"] is None
    assert defaults["max_price"] is None
    assert defaults["date_from"] is None
    assert defaults["date_to"] is None


# ==============================================================================
# CSV export path
# ==============================================================================


@pytest.mark.fast_sleeps
async def test_search_csv_export_basic(tmp_path: Path) -> None:
    """The downloaded file is read and its property data is returned."""
    csv_file = tmp_path / "results.csv"
    csv_file.write_text(
        _CSV_HEADER
        + '"MLS Listing","649999","3","2","1337 Phelps Ave #9","San Jose","CA","95117",'
        '"1200","","1963","1","542","446","Active","Condo/Co-op","","","","'
        '"https://www.redfin.com/CA/San-Jose/1337-Phelps-Ave-9/home/12345,'
        "MLSListings,ML82048884,N,N,37.3004024,-121.9617892,West San Jose\"\n"
    )

    mock_bm = _mock_bm_for_csv(str(csv_file))
    result = await _run_search(mock_bm, "San Jose, CA", limit=10)

    row = _single_row(result)
    assert row["Address"] == "1337 Phelps Ave #9, San Jose, CA, 95117"
    assert row["Price"] == "$649,999"
    assert row["Beds"] == "3"
    assert row["Baths"] == "2.0"
    assert row["Status"] == "Active"
    assert row["$/Sqft"] == "$542"
    assert row["Year Built"] == "1963"


@pytest.mark.fast_sleeps
async def test_search_csv_export_saves_csv_outside_download_dir(tmp_path: Path) -> None:
    """The CSV is told to move out of the download folder before it is read."""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    csv_file = download_dir / "results.csv"
    csv_file.write_text(
        _CSV_HEADER
        + '"MLS Listing","649999","3","2","1337 Phelps Ave #9","San Jose","CA","95117",'
        '"1200","","1963","1","542","446","Active","Condo/Co-op","","","","'
        '"https://www.redfin.com/CA/San-Jose/1337-Phelps-Ave-9/home/12345,'
        "MLSListings,ML82048884,N,N,37.3004024,-121.9617892,West San Jose\"\n"
    )

    mock_bm = _mock_bm_for_csv(str(csv_file))
    save_paths: list[str] = []

    def _move_to_save_path(save_path: str | None) -> dict:
        if save_path is None:
            return {"downloaded": False}
        save_paths.append(save_path)
        shutil.move(str(csv_file), save_path)
        return {
            "downloaded": True,
            "path": save_path,
            "filename": os.path.basename(save_path),
            "size": os.path.getsize(save_path),
        }

    mock_bm.wait_for_download = AsyncMock(side_effect=lambda **_kw: _move_to_save_path(_kw["save_path"]))
    result = await _run_search(mock_bm, "San Jose, CA", limit=10)

    assert _single_row(result)["Address"] == "1337 Phelps Ave #9, San Jose, CA, 95117"
    assert save_paths, "the CSV was never asked to move out of the download folder"
    assert not save_paths[0].startswith(str(download_dir)), (
        f"CSV save path still inside the download folder: {save_paths[0]}"
    )
    assert not (download_dir / "results.csv").exists()


@pytest.mark.fast_sleeps
async def test_search_csv_export_removes_the_csv_folder(tmp_path: Path) -> None:
    """The folder the CSV was saved into is gone once the search is done."""
    csv_file = tmp_path / "results.csv"
    csv_file.write_text(
        _CSV_HEADER
        + '"MLS Listing","649999","3","2","1337 Phelps Ave #9","San Jose","CA","95117",'
        '"1200","","1963","1","542","446","Active","Condo/Co-op","","","","'
        '"https://www.redfin.com/CA/San-Jose/1337-Phelps-Ave-9/home/12345,'
        "MLSListings,ML82048884,N,N,37.3004024,-121.9617892,West San Jose\"\n"
    )

    mock_bm = _mock_bm_for_csv(str(csv_file))
    result = await _run_search(mock_bm, "San Jose, CA", limit=10)

    assert _single_row(result)["Address"] == "1337 Phelps Ave #9, San Jose, CA, 95117"
    assert mock_bm.wait_for_download.called
    save_path = mock_bm.wait_for_download.call_args.kwargs["save_path"]
    assert not os.path.exists(os.path.dirname(save_path))


@pytest.mark.fast_sleeps
async def test_search_csv_active_listings_pass_date_filter(tmp_path: Path) -> None:
    """A for sale listing with no date is kept when a date filter is set."""
    csv_file = tmp_path / "results.csv"
    csv_file.write_text(
        _CSV_HEADER
        + '"MLS Listing","500000","3","2","100 Main St","San Jose","CA","95117",'
        '"1200","","2000","5","","","","Active","Single Family","","","","'
        '"https://www.redfin.com/CA/San-Jose/100-Main-St/home/1","MLS","","","","'
        "37.3\",\"-121.9\",\"Test\"\n"
    )

    mock_bm = _mock_bm_for_csv(str(csv_file), with_time_filter=True)

    # The current date is held fixed for the 30 day check.
    fixed_now = datetime(2026, 6, 27)
    with patch("app.tools.redfin.datetime", wraps=datetime) as mock_dt:
        mock_dt.now.return_value = fixed_now
        result = await _run_search(mock_bm, "San Jose, CA", date_from="2026-06-15")

    assert _single_row(result)["Address"] == "100 Main St, San Jose, CA, 95117"


@pytest.mark.fast_sleeps
async def test_search_csv_sold_listings_filtered_by_date(tmp_path: Path) -> None:
    """Sold listings outside the date range are dropped."""
    csv_file = tmp_path / "results.csv"
    csv_file.write_text(
        _CSV_HEADER
        + '"PAST SALE","400000","2","1","100 Old St","San Jose","CA","95117",'
        '"1000","","1990","30","","","Sold","Single Family","January-15-2020","","","'
        '"https://www.redfin.com/CA/San-Jose/100-Old-St/home/1","MLS","","","","'
        "37.3\",\"-121.9\",\"Test\"\n"
        + '"PAST SALE","600000","3","2","200 New St","San Jose","CA","95117",'
        '"1500","","2005","10","","","Sold","Single Family","March-10-2026","","","'
        '"https://www.redfin.com/CA/San-Jose/200-New-St/home/2","MLS","","","","'
        "37.3\",\"-121.9\",\"Test\"\n"
    )

    mock_bm = _mock_bm_for_csv(
        str(csv_file), with_time_filter=True, with_sold_filter=True
    )
    result = await _run_search(
        mock_bm, "San Jose, CA", sold=True, date_from="2025-07-01"
    )

    assert _single_row(result)["Address"] == "200 New St, San Jose, CA, 95117"


@pytest.mark.fast_sleeps
async def test_search_csv_sold_without_date_excluded_by_filter(tmp_path: Path) -> None:
    """A sold listing with no date is dropped when a date filter is set."""
    csv_file = tmp_path / "results.csv"
    csv_file.write_text(
        _CSV_HEADER
        + '"PAST SALE","500000","3","2","100 No Date St","San Jose","CA","95117",'
        '"1200","","2000","5","","","","Sold","Single Family","","","","'
        '"https://www.redfin.com/CA/San-Jose/100-No-Date-St/home/1","MLS","","","","'
        "37.3\",\"-121.9\",\"Test\"\n"
    )

    mock_bm = _mock_bm_for_csv(
        str(csv_file), with_time_filter=True, with_sold_filter=True
    )
    result = await _run_search(
        mock_bm, "San Jose, CA", sold=True, date_from="2025-07-01"
    )

    assert result == _NO_RESULTS_PREFIX + "San Jose, CA."


@pytest.mark.fast_sleeps
async def test_search_csv_price_filter(tmp_path: Path) -> None:
    """Listings outside the price range are dropped."""
    csv_file = tmp_path / "results.csv"
    csv_file.write_text(
        _CSV_HEADER
        + '"MLS Listing","400000","2","1","100 Cheap St","San Jose","CA","95117",'
        '"1000","","2000","5","","","","Active","Single Family","","","","'
        '"https://www.redfin.com/CA/San-Jose/100-Cheap-St/home/1","MLS","","","","'
        "37.3\",\"-121.9\",\"Test\"\n"
        + '"MLS Listing","700000","4","3","200 Expensive St","San Jose","CA","95117",'
        '"2000","","2010","3","","","","Active","Single Family","","","","'
        '"https://www.redfin.com/CA/San-Jose/200-Expensive-St/home/2","MLS","","","","'
        "37.3\",\"-121.9\",\"Test\"\n"
    )

    mock_bm = _mock_bm_for_csv(str(csv_file), with_price_filter=True)
    result = await _run_search(
        mock_bm, "San Jose, CA", min_price=450000, max_price=650000
    )

    assert result == _NO_RESULTS_PREFIX + "San Jose, CA."


@pytest.mark.fast_sleeps
async def test_search_csv_beds_filter(tmp_path: Path) -> None:
    """Listings with too few bedrooms are dropped."""
    csv_file = tmp_path / "results.csv"
    csv_file.write_text(
        _CSV_HEADER
        + '"MLS Listing","500000","2","1","100 Small St","San Jose","CA","95117",'
        '"1000","","2000","5","","","","Active","Single Family","","","","'
        '"https://www.redfin.com/CA/San-Jose/100-Small-St/home/1","MLS","","","","'
        "37.3\",\"-121.9\",\"Test\"\n"
        + '"MLS Listing","800000","4","3","200 Big St","San Jose","CA","95117",'
        '"2000","","2010","3","","","","Active","Single Family","","","","'
        '"https://www.redfin.com/CA/San-Jose/200-Big-St/home/2","MLS","","","","'
        "37.3\",\"-121.9\",\"Test\"\n"
    )

    mock_bm = _mock_bm_for_csv(str(csv_file), with_beds_baths_filter=True)
    result = await _run_search(mock_bm, "San Jose, CA", min_beds=3)

    assert _single_row(result)["Address"] == "200 Big St, San Jose, CA, 95117"


@pytest.mark.fast_sleeps
async def test_search_csv_limit(tmp_path: Path) -> None:
    """No more rows come back than the limit allows."""
    csv_file = tmp_path / "results.csv"
    rows = "".join(
        f'"MLS Listing","{500000 + i * 10000}","3","2","{100 + i} Main St",'
        '"San Jose","CA","95117","1200","","2000","5","","","","Active",'
        '"Single Family","","","","https://www.redfin.com/home/{i}","MLS",'
        '"","","37.3","-121.9","Test"\n'
        for i in range(5)
    )
    csv_file.write_text(_CSV_HEADER + rows)

    mock_bm = _mock_bm_for_csv(str(csv_file))
    result = await _run_search(mock_bm, "San Jose, CA", limit=2)

    assert len(_parse_table(result)) == 2


# ==============================================================================
# Web parsing fallback
# ==============================================================================


@pytest.mark.fast_sleeps
async def test_search_csv_fallback_to_web_parsing() -> None:
    """When the download fails, the listings are read off the page."""
    mock_bm = _mock_bm_for_web(
        [
            (
                "123 Main St",
                "San Jose, CA 95117",
                3,
                2,
                "$500,000",
                "1,200",
                "/CA/San-Jose/123-Main-St/home/1",
            )
        ]
    )
    result = await _run_search(mock_bm, "San Jose, CA", limit=10)

    row = _single_row(result)
    assert row["Address"] == "123 Main St, San Jose, CA 95117"
    assert row["Price"] == "$500,000"
    assert row["Beds"] == "3"
    assert row["Baths"] == "2.0"
    assert row["Sqft"] == "1200"


@pytest.mark.fast_sleeps
async def test_search_web_parsing_removes_the_csv_folder() -> None:
    """The CSV folder is gone when the download fails and the page is read."""
    tree = _make_listing_tree(
        [
            (
                "123 Main St",
                "San Jose, CA 95117",
                3,
                2,
                "$500,000",
                "1,200",
                "/CA/San-Jose/123-Main-St/home/1",
            )
        ]
    )
    home_snap = _make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE)
    results_snap = _make_snap(
        [{"ref_id": "download-btn", "role": "link", "name": "Download results"}],
        tree,
    )
    mock_bm = _mock_bm([home_snap, *([results_snap] * 6)])
    result = await _run_search(mock_bm, "San Jose, CA", limit=10)

    assert _single_row(result)["Address"] == "123 Main St, San Jose, CA 95117"
    assert mock_bm.wait_for_download.called
    save_path = mock_bm.wait_for_download.call_args.kwargs["save_path"]
    assert not os.path.exists(os.path.dirname(save_path))


@pytest.mark.fast_sleeps
async def test_search_web_parsing_limit() -> None:
    """Listings read off the page also stop at the limit."""
    mock_bm = _mock_bm_for_web(
        [
            (
                "100 Main St",
                "San Jose, CA 95117",
                2,
                1,
                "$400,000",
                "1,000",
                "/CA/San-Jose/100-Main-St/home/1",
            ),
            (
                "200 Main St",
                "San Jose, CA 95117",
                3,
                2,
                "$500,000",
                "1,200",
                "/CA/San-Jose/200-Main-St/home/2",
            ),
            (
                "300 Main St",
                "San Jose, CA 95117",
                4,
                3,
                "$600,000",
                "1,500",
                "/CA/San-Jose/300-Main-St/home/3",
            ),
        ]
    )
    result = await _run_search(mock_bm, "San Jose, CA", limit=2)

    rows = _parse_table(result)
    assert len(rows) == 2
    assert rows[0]["Address"] == "100 Main St, San Jose, CA 95117"
    assert rows[1]["Address"] == "200 Main St, San Jose, CA 95117"


# ==============================================================================
# Address query
# ==============================================================================


@pytest.mark.fast_sleeps
async def test_search_address_query() -> None:
    """A search for one address returns that home's details."""
    home_snap = _make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE)
    search_tree = {
        "role": "WebArea",
        "children": [
            {"role": "link", "name": "123 Main St", "url": "/CA/San-Jose/123-Main-St/home/1"},
            {"text": "123 Main St, San Jose, CA 95117"},
        ],
    }
    search_snap = _make_snap([], search_tree)
    details_tree = {
        "role": "WebArea",
        "children": [
            {"text": "123 Main St, San Jose, CA 95117"},
            {"role": "link", "name": "123 Main St", "url": "/CA/San-Jose/123-Main-St/home/1"},
            {"text": "$500,000"},
            {"text": "3 bd | 2 ba | 1,200 sq ft"},
        ],
    }
    details_snap = _make_snap([], details_tree)
    mock_bm = _mock_bm(
        [home_snap, search_snap, details_snap],
        url_sequence=["https://www.redfin.com/CA/San-Jose/123-Main-St/home/1"],
    )
    result = await _run_search(mock_bm, "123 Main St, San Jose, CA 95117")

    row = _single_row(result)
    assert row["Price"] == "$500,000"
    assert row["Beds"] == "3"


@pytest.mark.fast_sleeps
async def test_search_address_price_ignores_similar_homes() -> None:
    """The home's own price wins over a similar homes range, its cards, and its price per square foot."""
    home_snap = _make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE)
    home_link = "/TX/Austin/308-Ben-Howell-Dr-78704/home/31695188"
    search_tree = {
        "role": "WebArea",
        "children": [
            {"role": "link", "name": "308 Ben Howell Dr", "url": home_link},
            {"text": "308 Ben Howell Dr, Austin, TX 78704"},
        ],
    }
    search_snap = _make_snap([], search_tree)
    details_tree = {
        "role": "WebArea",
        "children": [
            {
                "role": "link",
                "name": "Similar price →$1,000,000-$1,400,000",
                "url": "/city/30818/TX/Austin/filter/min-price=1000000,max-price=1400000",
                "children": [{"text": "$1,000,000-$1,400,000"}],
            },
            {"text": "For sale $1,224,950"},
            {"text": "4 bd • 3.5 ba • 2,392 sq ft"},
            {"text": "308 Ben Howell Dr, Austin, TX 78704"},
            {"role": "generic", "name": "$512", "children": [{"text": "$512"}]},
            {"text": "Price/Sq.Ft."},
            *_make_listing_tree([_SIMILAR_HOME_CARD])["children"],
        ],
    }
    details_snap = _make_snap([], details_tree)
    mock_bm = _mock_bm(
        [home_snap, search_snap, details_snap],
        url_sequence=[_RESULTS_URL, f"https://www.redfin.com{home_link}"],
    )
    result = await _run_search(mock_bm, "308 Ben Howell Dr, Austin, TX 78704")

    row = _single_row(result)
    assert row["Price"] == "$1,224,950"
    assert row["$/Sqft"] == "$512"


@pytest.mark.fast_sleeps
async def test_search_address_fails_when_home_link_missing() -> None:
    """An address with no home link on the open page fails instead of returning another home."""
    home_snap = _make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE)
    results_snap = _make_snap(
        [],
        _make_listing_tree(
            [
                (
                    "98 E Saint John St",
                    "San Jose, CA 95112",
                    3,
                    2,
                    "$1,449,000",
                    "1,892",
                    "/CA/San-Jose/98-E-Saint-John-St-95112/home/104171",
                )
            ]
        ),
    )
    mock_bm = _mock_bm(
        [home_snap, results_snap],
        url_sequence=[_FILTER_PAGE_URL],
    )
    result = await _run_search(mock_bm, "308 Ben Howell Dr, San Jose, CA")

    assert (
        result
        == _FAILED_PREFIX + "no property page found for '308 Ben Howell Dr, San Jose, CA'"
    )


@pytest.mark.fast_sleeps
async def test_search_address_fails_when_landed_page_is_not_a_property() -> None:
    """An address whose home link was followed to a page that is not a property fails."""
    home_snap = _make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE)
    search_snap = _make_snap(
        [],
        {
            "role": "WebArea",
            "children": [
                {
                    "role": "link",
                    "name": "308 Ben Howell Dr",
                    "url": "/TX/Austin/308-Ben-Howell-Dr-78745/home/104900",
                },
            ],
        },
    )
    landed_snap = _make_snap(
        [],
        {
            "role": "WebArea",
            "children": [
                {"text": "$400,000"},
                {"text": "3 bd | 2 ba | 1,500 sq ft"},
            ],
        },
    )
    mock_bm = _mock_bm(
        [home_snap, search_snap, landed_snap],
        url_sequence=[_RESULTS_URL, _NOT_A_PROPERTY_URL],
    )
    result = await _run_search(mock_bm, "308 Ben Howell Dr, Austin, TX 78745")

    assert (
        result
        == _FAILED_PREFIX + "no property page found for '308 Ben Howell Dr, Austin, TX 78745'"
    )


@pytest.mark.fast_sleeps
async def test_search_address_read_from_owner_dashboard_page() -> None:
    """A page that stays on the owner dashboard still gives back the home."""
    dashboard_tree = {
        "role": "WebArea",
        "children": [
            {"role": "heading", "name": "My Home"},
            {"text": "123 Main St, San Jose, CA 95117"},
            {"text": "For sale $1,224,950"},
            {"text": "4 bd • 3.5 ba • 2,392 sq ft"},
        ],
    }
    mock_bm = _mock_bm(
        repeat_snapshot=_make_snap(_SEARCH_ELEMENTS, dashboard_tree),
        url_sequence=[_RESULTS_URL, _DASHBOARD_URL],
    )
    result = await _run_search(mock_bm, "123 Main St, San Jose, CA 95117")

    row = _single_row(result)
    assert row["Price"] == "$1,224,950"
    assert row["Address"] == "123 Main St, San Jose, CA 95117"
    assert row["URL"] == _DASHBOARD_URL


@pytest.mark.fast_sleeps
@pytest.mark.parametrize(
    "status_line,price,address,home_url",
    (
        (
            "Coming soon $849,000",
            "$849,000",
            "219 Lessin Ln #6, Austin, TX 78704",
            "/TX/Austin/219-Lessin-Ln-78704/unit-6/home/187798317",
        ),
        (
            "Contingent $699,000",
            "$699,000",
            "11117 Bexley Ln, Austin, TX 78739",
            "/TX/Austin/11117-Bexley-Ln-78739/home/31978845",
        ),
    ),
)
async def test_search_address_price_from_status_line(
    status_line: str, price: str, address: str, home_url: str
) -> None:
    """A home with a status line gives back the price on it, not a payment figure."""
    details_tree = {
        "role": "WebArea",
        "children": [
            {"text": "$725,000-$975,000"},
            {"text": status_line},
            {"text": "3 bd • 3 ba • 1,934 sq ft"},
            {"text": address},
            *_PAGE_COST_NODES,
            *_make_listing_tree([_SIMILAR_HOME_CARD])["children"],
        ],
    }
    result = await _search_address_page(details_tree, address, home_url)

    assert _single_row(result)["Price"] == price


@pytest.mark.fast_sleeps
async def test_search_address_price_from_home_price_line() -> None:
    """With no status line the home price label still gives the list price."""
    details_tree = {
        "role": "WebArea",
        "children": [
            {"text": "4415 Marathon Blvd, Austin, TX 78756"},
            *_PAGE_COST_NODES,
            {"text": "Down payment 20% ($169,800)"},
            {"text": "Home price $849,000"},
            *_make_listing_tree([_SIMILAR_HOME_CARD])["children"],
        ],
    }
    result = await _search_address_page(
        details_tree,
        "4415 Marathon Blvd, Austin, TX 78756",
        "/TX/Austin/4415-Marathon-Blvd-78756/home/187803051",
    )

    assert _single_row(result)["Price"] == "$849,000"


@pytest.mark.fast_sleeps
async def test_search_address_price_skips_similar_home_card() -> None:
    """With no status line the home's own price still beats a similar home's card."""
    details_tree = {
        "role": "WebArea",
        "children": [
            *_make_listing_tree([_SIMILAR_HOME_CARD])["children"],
            {"text": "$849,000"},
            {"text": "3100 Menchaca Rd #3, Austin, TX 78704"},
        ],
    }
    result = await _search_address_page(
        details_tree,
        "3100 Menchaca Rd #3, Austin, TX 78704",
        "/TX/Austin/3100-Menchaca-Rd-78704/unit-3/home/31667261",
    )

    assert _single_row(result)["Price"] == "$849,000"


@pytest.mark.fast_sleeps
async def test_search_address_fails_when_page_holds_only_payment_figures() -> None:
    """A page whose money is all payment and size figures gives no price."""
    details_tree = {
        "role": "WebArea",
        "children": [
            {"text": "$725,000-$975,000"},
            {"text": "5708 Toscana Ave, Austin, TX 78724"},
            *_PAGE_COST_NODES,
        ],
    }
    result = await _search_address_page(
        details_tree,
        "5708 Toscana Ave, Austin, TX 78724",
        "/TX/Austin/5708-Toscana-Ave-78724/home/31900021",
    )

    assert result == _FAILED_PREFIX + (
        "could not extract property from address page for "
        "'5708 Toscana Ave, Austin, TX 78724'"
    )


# ==============================================================================
# Edge cases
# ==============================================================================


@pytest.mark.fast_sleeps
async def test_search_captcha_detected() -> None:
    """A page asking to prove you are human returns no results."""
    mock_snap = _make_snap(
        _SEARCH_ELEMENTS,
        {
            "role": "WebArea",
            "children": [{"text": "Press & Hold to verify you are human"}],
        },
    )
    mock_bm = _mock_bm(repeat_snapshot=mock_snap)
    result = await _run_search(mock_bm, "San Jose, CA")

    assert result == _NO_RESULTS_PREFIX + "San Jose, CA."


@pytest.mark.fast_sleeps
async def test_search_empty_results() -> None:
    """A search that finds nothing says so."""
    mock_bm = _mock_bm(repeat_snapshot=_make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE))
    result = await _run_search(mock_bm, "San Jose, CA")

    assert result == _NO_RESULTS_PREFIX + "San Jose, CA."


async def test_search_fails_when_homepage_snapshot_unavailable() -> None:
    """A home page that cannot be read makes the search fail."""
    mock_bm = _mock_bm(snapshot_sequence=[None])
    result = await _run_search(mock_bm, "San Jose, CA")

    assert result == _FAILED_PREFIX + "could not capture the Redfin homepage"


@pytest.mark.fast_sleeps
async def test_search_fails_when_results_snapshot_unavailable() -> None:
    """A results page that cannot be read makes the search fail."""
    home_snap = _make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE)
    mock_bm = _mock_bm(snapshot_sequence=[home_snap, None])
    result = await _run_search(mock_bm, "San Jose, CA")

    assert result == _FAILED_PREFIX + "could not capture the search results page"


async def test_search_fails_when_search_box_missing() -> None:
    """A missing search box makes the search fail."""
    mock_bm = _mock_bm(repeat_snapshot=_make_snap([], _EMPTY_TREE))
    result = await _run_search(mock_bm, "San Jose, CA")

    assert result == _FAILED_PREFIX + "search box not found"


@pytest.mark.fast_sleeps
async def test_search_fails_when_page_stays_on_homepage() -> None:
    """A search that leaves the browser on the home page reports a failure."""
    mock_bm = _mock_bm(
        repeat_snapshot=_make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE),
        url_sequence=[_HOMEPAGE_URL],
    )
    result = await _run_search(mock_bm, "Austin, TX 78704")

    assert (
        result
        == _FAILED_PREFIX + "Redfin did not open a results page for 'Austin, TX 78704'"
    )


@pytest.mark.fast_sleeps
async def test_search_proceeds_when_page_leaves_homepage() -> None:
    """A search that moves off the home page keeps going instead of failing."""
    mock_bm = _mock_bm(
        repeat_snapshot=_make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE),
        url_sequence=[_RESULTS_URL],
    )
    result = await _run_search(mock_bm, "95117")

    assert not result.startswith(_FAILED_PREFIX), result
    assert result == _NO_RESULTS_PREFIX + "95117."


@pytest.mark.fast_sleeps
async def test_search_fails_when_extraction_snapshot_unavailable() -> None:
    """A results page that goes missing before the listings are read makes the search fail."""
    home_snap = _make_snap(_SEARCH_ELEMENTS, _EMPTY_TREE)
    listing_snap = _make_snap(
        [],
        _make_listing_tree(
            [
                (
                    "123 Main St",
                    "San Jose, CA 95117",
                    3,
                    2,
                    "$500,000",
                    "1,200",
                    "/CA/San-Jose/123-Main-St/home/1",
                )
            ]
        ),
    )
    # Three snapshots for the download tries, then the one that fails.
    mock_bm = _mock_bm(snapshot_sequence=[home_snap, listing_snap, None, None, None, None])
    result = await _run_search(mock_bm, "San Jose, CA")

    assert result == _FAILED_PREFIX + "could not capture the results page"


@pytest.mark.fast_sleeps
async def test_search_fails_when_csv_not_utf8(tmp_path: Path) -> None:
    """A downloaded file that is not UTF-8 text makes the search fail."""
    csv_file = tmp_path / "results.csv"
    csv_file.write_bytes(b"\xff\xfe\x00\x00")

    mock_bm = _mock_bm_for_csv(str(csv_file))
    result = await _run_search(mock_bm, "San Jose, CA")

    assert result == _FAILED_PREFIX + "could not read the results CSV as text"


async def test_search_invalid_date_from_rejected() -> None:
    """A date_from in the wrong shape is rejected before any searching."""
    result = await run_text(redfin_search_tool, "San Jose, CA", date_from="banana")

    assert result == _FAILED_PREFIX + "date_from must be a YYYY-MM-DD date: 'banana'"


async def test_search_invalid_date_to_rejected() -> None:
    """A date_to in the wrong shape is rejected before any searching."""
    result = await run_text(redfin_search_tool, "San Jose, CA", date_to="2026/08/22")

    assert result == _FAILED_PREFIX + "date_to must be a YYYY-MM-DD date: '2026/08/22'"


async def test_search_empty_location() -> None:
    """An empty location makes the search fail."""
    result = await run_text(redfin_search_tool, "")

    assert (
        result
        == _FAILED_PREFIX + "location must be a non-empty string"
    )


async def test_search_whitespace_location() -> None:
    """A location of only spaces makes the search fail."""
    result = await run_text(redfin_search_tool, "   ")

    assert (
        result
        == _FAILED_PREFIX + "location must be a non-empty string"
    )


@pytest.mark.fast_sleeps
async def test_search_no_options_uses_defaults() -> None:
    """A search with no options given still returns rows."""
    mock_bm = _mock_bm_for_web(
        [
            (
                "123 Main St",
                "San Jose, CA 95117",
                3,
                2,
                "$500,000",
                "1,200",
                "/CA/San-Jose/123-Main-St/home/1",
            )
        ]
    )
    result = await _run_search(mock_bm, "San Jose, CA")

    assert len(_parse_table(result)) == 1


# ==============================================================================
# Sold properties
# ==============================================================================


def _sold_search_sequence() -> list[MagicMock]:
    """Return the snapshots a sold properties search walks through."""
    search_tree = {
        "role": "WebArea",
        "children": [
            {"role": "button", "name": "Property at 123 Main St, 3 beds, 2 baths"},
            {"role": "link", "name": "Recently Sold", "url": "/CA/San-Jose/recently-sold"},
        ],
    }
    filter_actionable = [
        {"ref_id": "sort-cb", "role": "combobox", "name": "Sort"},
        {"ref_id": "type-cb", "role": "combobox", "name": "Home type"},
        {"ref_id": "time-filter", "role": "combobox", "name": "Sold within"},
        {"ref_id": "sold-radio", "role": "radio", "name": "Sold"},
    ]
    sold_tree = {
        "role": "WebArea",
        "children": [
            {
                "role": "button",
                "name": "Property at 123 Main St, San Jose, CA 95117, 3 beds, 2 baths",
                "children": [
                    {"text": "$500,000"},
                    {"text": "1,200 sq ft"},
                    {"text": "SOLD JUN 5, 2026"},
                    {
                        "role": "link",
                        "name": "123 Main St",
                        "url": "/CA/San-Jose/123-Main-St/home/1",
                    },
                ],
            }
        ],
    }
    empty = _EMPTY_TREE
    blank_snap = _make_snap([], empty)
    return [
        _make_snap(_SEARCH_ELEMENTS, empty),
        _make_snap([], search_tree),
        _make_snap([{"ref_id": "filters", "role": "button", "name": "Filters"}], search_tree),
        _make_snap(filter_actionable, empty),
        _make_snap(filter_actionable, empty),
        blank_snap,
        _make_snap([], empty),
        _make_snap([], empty),
        _make_snap([], empty),
        _make_snap([], sold_tree),
    ]


@pytest.mark.fast_sleeps
async def test_search_sold_properties() -> None:
    """A sold search returns rows marked sold with their sale dates."""
    mock_bm = _mock_bm(_sold_search_sequence())
    result = await _run_search(mock_bm, "San Jose, CA", sold=True)

    row = _single_row(result)
    assert row["Status"] == "Sold"
    assert row["Date"] == "2026-06-05"


# ==============================================================================
# Off market pricing
# ==============================================================================


@pytest.mark.fast_sleeps
async def test_search_off_market_sale_history_exact_price() -> None:
    """An off market home takes its price from the sale history."""
    off_market_text = (
        "Sale history\n"
        "Mar 5, 2025 Sold $2,075,000\n"
        "Jan 28, 2025 Listed $2,100,000"
    )
    mock_bm = _mock_bm_for_off_market(off_market_text, heading_price="$2,075,000")
    result = await _run_search(mock_bm, "123 Main St, San Jose, CA 95117")

    row = _single_row(result)
    assert row["Price"] == "$2,075,000"


@pytest.mark.fast_sleeps
async def test_search_off_market_abbreviated_million() -> None:
    """A price written with an M on an off market page becomes a full number."""
    mock_bm = _mock_bm_for_off_market(
        "Off Market Mar 2025 for $2M", heading_price="$2,000,000"
    )
    result = await _run_search(mock_bm, "123 Main St, San Jose, CA 95117")

    row = _single_row(result)
    assert row["Price"] == "$2,000,000"


@pytest.mark.fast_sleeps
async def test_search_off_market_abbreviated_thousand() -> None:
    """A price written with a K on an off market page becomes a full number."""
    mock_bm = _mock_bm_for_off_market(
        "Off Market Jun 15, 2024 for $500K", heading_price="$500,000"
    )
    result = await _run_search(mock_bm, "123 Main St, San Jose, CA 95117")

    row = _single_row(result)
    assert row["Price"] == "$500,000"


@pytest.mark.fast_sleeps
async def test_search_off_market_comma_separated_abbreviated() -> None:
    """A price written with commas on an off market page is read as is."""
    mock_bm = _mock_bm_for_off_market("Off Market Feb 2024 for $1,200,000")
    result = await _run_search(mock_bm, "123 Main St, San Jose, CA 95117")

    row = _single_row(result)
    assert row["Price"] == "$1,200,000"


@pytest.mark.fast_sleeps
async def test_search_off_market_no_pricing_returns_failure() -> None:
    """An off market page with no price makes the search fail."""
    mock_bm = _mock_bm_for_off_market("This home has no sale data available.")
    result = await _run_search(mock_bm, "123 Main St, San Jose, CA 95117")

    assert result.startswith(_FAILED_PREFIX)
    assert "could not extract property from address page" in result


@pytest.mark.fast_sleeps
async def test_search_owner_dashboard_opens_public_page() -> None:
    """A claimed home that lands on the owner dashboard ends up on the public page."""
    mock_bm, dashboard_snap = _mock_bm_for_owner_dashboard(
        "Mar 5, 2025 Sold $2,075,000"
    )
    result = await _run_search(mock_bm, "123 Main St, San Jose, CA 95117")

    row = _single_row(result)
    assert row["Price"] == "$2,075,000"
    assert row["Status"] == "Off Market"
    assert row["URL"] == _PUBLIC_HOME_URL
    dashboard_snap.click.assert_awaited_once_with("view-public", verbose=True)


# ==============================================================================
# Live tests against the real Redfin site
# ==============================================================================


@skip_if_no_chrome
@pytest.mark.live
@pytest.mark.timeout(300)
class TestRedfinLive:
    """Live tests that need Chrome and the real Redfin site."""

    async def test_live_off_market_single(self) -> None:
        """A search for one off market home returns its sale facts."""
        result = await _live_search("643 Crestview Dr, San Jose, CA 95117")

        rows = _assert_table(result, "off-market single search")
        assert len(rows) == 1, f"expected exactly 1 row, got {len(rows)}"
        row = rows[0]
        assert "643 Crestview Dr" in row["Address"], row["Address"]
        assert row["Status"] == "Off Market", row["Status"]
        assert int(row["Beds"]) == 3, row["Beds"]
        assert float(row["Baths"]) == 3.0, row["Baths"]
        assert row["Price"] == "$2,280,000", row["Price"]
        assert row["Year Built"] == "1953", row["Year Built"]
        assert row["Lot"], "Lot must be non-empty for an off-market home"
        # The row link points at the public home page, not the owner dashboard.
        _assert_property_url(row["URL"])

    async def test_live_non_owner_single(self) -> None:
        """A search for one home that is not claimed lands on the public page."""
        result = await _live_search("661 Crestview Dr, San Jose, CA 95117")

        rows = _assert_table(result, "non-owner single search")
        assert len(rows) == 1, f"expected exactly 1 row, got {len(rows)}"
        row = rows[0]
        assert "661 Crestview Dr".lower() in row["Address"].lower(), row["Address"]
        assert row["Status"] == "Off Market", row["Status"]
        assert int(row["Beds"]) == 3, row["Beds"]
        assert float(row["Baths"]) == 2.0, row["Baths"]
        assert _sqft_to_int(row["Sqft"]) == 1250, f"bad Sqft: {row['Sqft']!r}"
        assert row["Year Built"] == "1961", row["Year Built"]
        assert row["Lot"], "Lot must be non-empty for an off-market home"
        assert "/home/" in row["URL"], row["URL"]

    async def test_live_area_search(self) -> None:
        """A search by zip code returns good listing rows."""
        result = await _live_search("95117", limit=5)

        rows = _assert_table(result, "area search")
        assert len(rows) <= 5, f"limit 5 not respected, got {len(rows)} rows"
        _assert_all_rows(rows, _assert_listing_row)

    async def test_live_active_filters(self) -> None:
        """Filters cut down the rows, and every row left fits the filters."""
        unfiltered = _assert_table(
            await _live_search("95117", limit=_LIVE_SEARCH_LIMIT), "unfiltered active search"
        )
        date_from = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        filtered = _assert_table(
            await _live_search(
                "95117",
                min_beds=4,
                min_baths=3,
                min_price=2_000_000,
                date_from=date_from,
                limit=_LIVE_SEARCH_LIMIT,
            ),
            "filtered active search",
        )

        assert len(filtered) < len(unfiltered), (
            f"server-side filter not effective: filtered({len(filtered)}) >= "
            f"unfiltered({len(unfiltered)})"
        )
        _assert_all_rows(filtered, lambda row: _assert_active_filter_row(row, date_from))

    async def test_live_sold_combined_filters(self) -> None:
        """A sold search with several filters returns rows that fit them all."""
        rows = _assert_table(
            await _live_search(
                "95117",
                sold=True,
                min_beds=3,
                min_baths=2,
                min_price=1_000_000,
                date_from="2025-01-01",
                limit=_LIVE_SEARCH_LIMIT,
            ),
            "sold combined filters",
        )
        _assert_all_rows(
            rows,
            lambda row: _assert_sold_row(
                row, "2025-01-01", min_beds=3, min_baths=2, min_price=1_000_000
            ),
        )

        ranged = _assert_table(
            await _live_search(
                "95117",
                sold=True,
                date_from="2025-01-01",
                date_to="2025-03-01",
                limit=_LIVE_SEARCH_LIMIT,
            ),
            "sold date range",
        )
        _assert_all_rows(
            ranged, lambda row: _assert_sold_row(row, "2025-01-01", date_to="2025-03-01")
        )


class TestFilterApplicationOnParams:
    """Tests for the filter option description text."""

    @pytest.mark.parametrize("name", ("min_price", "max_price", "min_beds", "min_baths"))
    def test_price_and_room_filters_document_server_side_application(self, name: str) -> None:
        descs = {p.name: p.description for p in redfin_search_tool.params}
        assert "Redfin's Filters panel" in descs[name]

    def test_date_from_documents_application_paths(self) -> None:
        doc = next(p.description for p in redfin_search_tool.params if p.name == "date_from")
        assert "Time on Redfin" in doc
        assert "Sold within" in doc
        assert "maximum 30 days" in doc

    def test_date_to_documents_client_side_application(self) -> None:
        doc = next(p.description for p in redfin_search_tool.params if p.name == "date_to")
        assert "Always applied client-side" in doc
