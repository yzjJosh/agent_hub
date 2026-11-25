"""Tests for Redfin browser search orchestration and markdown output."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.lib.redfin.models import RedfinSearchError, SearchOptions
from app.lib.redfin.search import search_redfin
from tests.app.lib.redfin.conftest import make_fake_snapshot

_SEARCH_ELEMENTS = [
    {"ref_id": "1", "role": "textbox", "name": "Search for properties"},
    {"ref_id": "2", "role": "button", "name": "submit search"},
]
_EMPTY_TREE = {"role": "WebArea", "children": []}
_CSV_HEADER = (
    '"SALE TYPE","PRICE","BEDS","BATHS","ADDRESS","CITY","STATE OR PROVINCE",'
    '"ZIP OR POSTAL CODE","SQUARE FEET","LOT SIZE","YEAR BUILT","DAYS ON MARKET",'
    '"$/SQUARE FEET","HOA/MONTH","STATUS","PROPERTY TYPE","SOLD DATE",'
    '"NEXT OPEN HOUSE START TIME","NEXT OPEN HOUSE END TIME","URL","SOURCE",'
    '"MLS#","LATITUDE","LONGITUDE"\n'
)


def _make_browser(
    snapshots: list[MagicMock | None],
    url_sequence: list[str] | None = None,
    csv_path: str | None = None,
) -> MagicMock:
    """Create a fake BrowserManager with configurable snapshots, URLs, and download."""
    urls = list(url_sequence) if url_sequence else ["https://www.redfin.com/zipcode/95117"]
    index = 0

    def next_url(_target_id: str, **_kwargs: bool) -> str:
        nonlocal index
        value = urls[min(index, len(urls) - 1)]
        index += 1
        return value

    browser = MagicMock()
    browser.open_page = AsyncMock(return_value="tab-1")
    browser.navigate = AsyncMock(return_value=True)
    browser.wait_for_load = AsyncMock(return_value=True)
    browser.capture_snapshot = AsyncMock(side_effect=snapshots)
    browser.get_url = MagicMock(side_effect=next_url)
    download_result = (
        {"downloaded": True, "path": csv_path} if csv_path else {"downloaded": False}
    )
    browser.wait_for_download = AsyncMock(return_value=download_result)
    browser.cleanup = AsyncMock()
    return browser


async def _run_with_browser(
    browser: MagicMock, location: str, options: SearchOptions
) -> str:
    """Run search_redfin with BrowserManager patched to return browser."""
    mock_cls = MagicMock()
    mock_cls.create = AsyncMock(return_value=browser)
    with patch("app.lib.redfin.search.BrowserManager", mock_cls):
        return await search_redfin(location, options)


@pytest.mark.fast_sleeps
async def test_search_redfin_follows_city_suggestion_and_reads_csv(
    tmp_path: Path,
) -> None:
    """When initial results have no cards, a city suggestion link is followed."""
    csv_file = tmp_path / "results.csv"
    csv_file.write_text(
        _CSV_HEADER
        + '"MLS Listing","700000","3","2","100 Main St","San Jose","CA","95117",'
        '"1400","","2000","2","500","","Active","Single Family","","","",'
        '"https://www.redfin.com/home/1","MLS","ML1","37.3","-121.9"\n',
        encoding="utf-8",
    )
    home_snap = make_fake_snapshot(_SEARCH_ELEMENTS, _EMPTY_TREE)
    suggestion_snap = make_fake_snapshot(
        [],
        {
            "role": "WebArea",
            "children": [
                {"role": "link", "name": "San Jose", "url": "/city/17420/CA/San-Jose"}
            ],
        },
    )
    city_snap = make_fake_snapshot([], _EMPTY_TREE)
    download_snap = make_fake_snapshot(
        [{"ref_id": "dl", "role": "link", "name": "Download results"}],
        _EMPTY_TREE,
    )
    browser = _make_browser(
        [home_snap, suggestion_snap, city_snap, download_snap],
        csv_path=str(csv_file),
    )

    markdown = await _run_with_browser(browser, "San Jose, CA", SearchOptions(limit=5))

    assert "100 Main St, San Jose, CA, 95117" in markdown
    assert "$700,000" in markdown
    browser.cleanup.assert_awaited_once()


@pytest.mark.fast_sleeps
async def test_search_redfin_returns_no_results_when_no_cards_or_suggestions() -> None:
    """A landing page with neither listing cards nor suggestions reports no properties."""
    home_snap = make_fake_snapshot(_SEARCH_ELEMENTS, _EMPTY_TREE)
    empty_snap = make_fake_snapshot([], _EMPTY_TREE)
    browser = _make_browser([home_snap, empty_snap])

    result = await _run_with_browser(browser, "Nowhere, ZZ", SearchOptions())

    assert result == "No properties found for Nowhere, ZZ."
    browser.cleanup.assert_awaited_once()


@pytest.mark.fast_sleeps
async def test_search_redfin_address_opens_public_page_from_owner_dashboard() -> None:
    """Owner dashboard pages click View public page and return the home details."""
    home_snap = make_fake_snapshot(_SEARCH_ELEMENTS, _EMPTY_TREE)
    dashboard_snap = make_fake_snapshot(
        [{"ref_id": "view-public", "role": "button", "name": "View public page"}],
        {"role": "WebArea", "children": [{"role": "heading", "name": "My Home"}]},
    )
    public_tree = {
        "role": "WebArea",
        "children": [
            {"text": "123 Main St, San Jose, CA 95117"},
            {"text": "For sale $950,000"},
            {"text": "3 bd | 2 ba | 1,600 sq ft"},
        ],
    }
    public_snap = make_fake_snapshot([], public_tree)
    browser = _make_browser(
        [home_snap, dashboard_snap, dashboard_snap, public_snap],
        url_sequence=[
            "https://www.redfin.com/myredfin/owner-dashboard/123",
            "https://www.redfin.com/myredfin/owner-dashboard/123",
            "https://www.redfin.com/CA/San-Jose/123-Main-St/home/1",
        ],
    )

    markdown = await _run_with_browser(
        browser, "123 Main St, San Jose, CA 95117", SearchOptions()
    )

    assert "$950,000" in markdown
    assert "https://www.redfin.com/CA/San-Jose/123-Main-St/home/1" in markdown
    dashboard_snap.click.assert_awaited_once_with("view-public", verbose=True)


@pytest.mark.fast_sleeps
async def test_search_redfin_raises_when_search_box_missing_and_cleans_up() -> None:
    """A missing search box raises RedfinSearchError and still closes the browser."""
    browser = _make_browser([make_fake_snapshot([], _EMPTY_TREE)])

    with pytest.raises(RedfinSearchError, match="search box not found"):
        await _run_with_browser(browser, "San Jose, CA", SearchOptions())

    browser.cleanup.assert_awaited_once()
