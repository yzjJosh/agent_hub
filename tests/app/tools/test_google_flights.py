from __future__ import annotations

import base64
import urllib.parse
from datetime import date, timedelta
from unittest.mock import patch

import pytest

from app.tools.google_flights import google_flights_tool
from tests.app.conftest import is_llm_available, skip_if_no_chrome
from tests.app.lib.tool.conftest import run_text

# The decoded tfs protobuf, as hex, for verified SFO to JFK trips.
_GOLDEN_RT_ECONOMY = (
    "081c10021a1e120a323032362d31312d30386a070801120353464f7207080112034a464b"
    "1a1e120a323032362d31312d31356a07080112034a464b72070801120353464f"
    "40014801700182010b08ffffffffffffffffff01980101c80100"
)
_GOLDEN_RT_BUSINESS = (
    "081c10021a1e120a323032362d31312d30386a070801120353464f7207080112034a464b"
    "1a1e120a323032362d31312d31356a07080112034a464b72070801120353464f"
    "40014803700182010b08ffffffffffffffffff01980101c80101"
)
_GOLDEN_RT_NONSTOP = (
    "081c10021a20120a323032362d31312d303828006a070801120353464f7207080112034a464b"
    "1a20120a323032362d31312d313528006a07080112034a464b72070801120353464f"
    "40014801700182010b08ffffffffffffffffff01980101c80100"
)
_GOLDEN_ONE_WAY_NONSTOP = (
    "081c10021a20120a323032362d31312d303828006a070801120353464f7207080112034a464b"
    "40014801700182010b08ffffffffffffffffff01980102c80100"
)
_GOLDEN_RT_ECONOMY_EXCL_BASIC = (
    "081c10021a1e120a323032362d31312d30386a070801120353464f7207080112034a464b"
    "1a1e120a323032362d31312d31356a07080112034a464b72070801120353464f"
    "40014801700182010b08ffffffffffffffffff01980101c80101"
)

# Departure dates for the live tests, relative to the day they run.
_LIVE_DEPART_DATE = (date.today() + timedelta(days=7)).isoformat()
_LIVE_ONE_WAY_DEPART_DATE = (date.today() + timedelta(days=30)).isoformat()


async def _sent_prompt(airlines: list[str] | None = None, count: int = 5) -> str:
    """The prompt a search call sends to the flights page."""
    with patch("app.tools.google_flights.analyze_page", return_value=("stub", "https://www.google.com/travel/flights/search")) as fake:
        await run_text(google_flights_tool, "SFO", "JFK", "2026-03-15", airlines=airlines, count=count)

    return fake.call_args.kwargs["prompt"]


async def _sent_tfs(**kwargs: str | bool | None) -> str:
    """The decoded tfs protobuf of the URL a search call sends, as hex."""
    with patch("app.tools.google_flights.analyze_page", return_value=("stub", "https://www.google.com/travel/flights/search")) as fake:
        await run_text(google_flights_tool, "SFO", "JFK", "2026-11-08", **kwargs)

    url = fake.call_args.kwargs["url"]
    token = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["tfs"][0]
    padded = token + "=" * (-len(token) % 4)
    return base64.urlsafe_b64decode(padded).hex()


async def _sent_url() -> str:
    """The URL a search call opens."""
    with patch("app.tools.google_flights.analyze_page", return_value=("stub", "https://www.google.com/travel/flights/search")) as fake:
        await run_text(google_flights_tool, "SFO", "JFK", "2026-11-08")

    return fake.call_args.kwargs["url"]


def _set_market(monkeypatch: pytest.MonkeyPatch, language: str | None, country: str | None, currency: str | None) -> None:
    """Set each market variable to its value, or remove it when the value is None."""
    for name, value in (
        ("GOOGLE_FLIGHTS_LANGUAGE", language),
        ("GOOGLE_FLIGHTS_COUNTRY", country),
        ("GOOGLE_FLIGHTS_CURRENCY", currency),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


class TestSearchValidation:
    """Tests for bad search inputs."""

    async def test_search_empty_origin(self) -> None:
        """An empty origin gives an error."""
        result = await run_text(google_flights_tool, "", "JFK", "2026-03-15")

        assert "origin" in result.lower()

    async def test_search_empty_destination(self) -> None:
        """An empty destination gives an error."""
        result = await run_text(google_flights_tool, "SFO", "", "2026-03-15")

        assert "destination" in result.lower()

    async def test_search_empty_depart_date(self) -> None:
        """An empty depart date gives an error."""
        result = await run_text(google_flights_tool, "SFO", "JFK", "")

        assert "date" in result.lower()

    async def test_search_invalid_depart_date_format(self) -> None:
        """A depart date in the wrong shape gives an error."""
        result = await run_text(google_flights_tool, "SFO", "JFK", "03/15/2026")

        assert "date" in result.lower()


class TestSearchParams:
    """Tests for the airline and count params."""

    async def test_search_count_below_minimum(self) -> None:
        """A count below the minimum gives an error."""
        result = await run_text(google_flights_tool, "SFO", "JFK", "2026-03-15", count=0)

        assert "count" in result.lower()

    async def test_search_count_above_maximum(self) -> None:
        """A count above the maximum gives an error."""
        result = await run_text(google_flights_tool, "SFO", "JFK", "2026-03-15", count=11)

        assert "count" in result.lower()

    def test_count_param_carries_bounds_and_default(self) -> None:
        """The count param declares its default and bounds."""
        count = next(p for p in google_flights_tool.params if p.name == "count")

        assert count.default == 5
        assert count.min == 1
        assert count.max == 10

    def test_airlines_param_defaults_to_none(self) -> None:
        """The airlines param is absent by default."""
        airlines = next(p for p in google_flights_tool.params if p.name == "airlines")

        assert airlines.required is False
        assert airlines.default is None

    def test_depart_date_param_is_required(self) -> None:
        """The depart date param is required and of date kind."""
        depart_date = next(p for p in google_flights_tool.params if p.name == "depart_date")

        assert depart_date.required is True
        assert depart_date.kind == "date"


class TestSearchCabin:
    """Tests for the cabin class param."""

    @pytest.mark.parametrize("cabin", ("economy_excl_basic", "premium_economy", "business", "first"))
    async def test_search_accepts_each_cabin(self, cabin: str) -> None:
        """A known cabin value passes validation and reaches the page search."""
        with patch("app.tools.google_flights.analyze_page", return_value=("stub", "https://www.google.com/travel/flights/search")):
            result = await run_text(google_flights_tool, "SFO", "JFK", "2026-03-15", cabin=cabin)

        assert "error" not in result.lower()

    def test_cabin_param_declares_enum(self) -> None:
        """The cabin param declares its enum values and no default value."""
        cabin = next(p for p in google_flights_tool.params if p.name == "cabin")

        assert cabin.enum == ("economy_excl_basic", "premium_economy", "business", "first")
        assert cabin.default is None


class TestSearchReturnDate:
    """Tests for the round-trip return date param."""

    async def test_search_invalid_return_date_format(self) -> None:
        """A return date in the wrong shape gives an error."""
        result = await run_text(google_flights_tool, "SFO", "JFK", "2026-11-08", return_date="15/11/2026")

        assert "invalid date" in result.lower()

    async def test_search_return_date_before_departure(self) -> None:
        """A return date earlier than the departure date gives an error."""
        result = await run_text(google_flights_tool, "SFO", "JFK", "2026-11-15", return_date="2026-11-08")

        assert "before" in result.lower()

    async def test_search_return_date_with_oneway(self) -> None:
        """A return date with oneway gives an error."""
        result = await run_text(google_flights_tool, "SFO", "JFK", "2026-11-08", oneway=True, return_date="2026-11-15")

        assert "oneway" in result.lower()

    async def test_search_no_return_date_is_one_way(self) -> None:
        """Without a return date the tfs carries a single segment and the one-way trip type."""
        tfs_hex = await _sent_tfs()

        assert tfs_hex.count("1a1e") == 1
        assert tfs_hex.endswith("980102c80100")


class TestBuildTfs:
    """Tests for the tfs protobuf encoding against verified goldens."""

    async def test_tfs_round_trip_economy(self) -> None:
        """A round-trip economy trip encodes to the verified golden bytes."""
        tfs_hex = await _sent_tfs(return_date="2026-11-15")

        assert tfs_hex == _GOLDEN_RT_ECONOMY

    async def test_tfs_round_trip_business(self) -> None:
        """A round-trip business trip encodes cabin code 3 and the basic flag."""
        tfs_hex = await _sent_tfs(return_date="2026-11-15", cabin="business")

        assert tfs_hex == _GOLDEN_RT_BUSINESS

    async def test_tfs_round_trip_nonstop(self) -> None:
        """A round-trip nonstop trip sets the stops filter in both segments."""
        tfs_hex = await _sent_tfs(return_date="2026-11-15", nonstop=True)

        assert tfs_hex == _GOLDEN_RT_NONSTOP

    async def test_tfs_one_way_nonstop(self) -> None:
        """A one-way nonstop trip encodes to the verified golden bytes."""
        tfs_hex = await _sent_tfs(oneway=True, nonstop=True)

        assert tfs_hex == _GOLDEN_ONE_WAY_NONSTOP

    async def test_tfs_economy_excl_basic_flag(self) -> None:
        """Economy excluding basic keeps cabin code 1 and sets the basic flag."""
        tfs_hex = await _sent_tfs(return_date="2026-11-15", cabin="economy_excl_basic")

        assert tfs_hex == _GOLDEN_RT_ECONOMY_EXCL_BASIC


class TestSearchMarket:
    """Tests for the language, country and currency of the search URL."""

    @pytest.mark.parametrize("value", (None, ""))
    async def test_market_defaults_when_unset_or_empty(self, monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
        """With the market variables unset or empty, the URL asks for English, the US and dollars."""
        _set_market(monkeypatch, value, value, value)

        url = await _sent_url()

        assert url.startswith("https://www.google.com/travel/flights/search?tfs=")
        assert url.endswith("&hl=en&gl=US&curr=USD")

    async def test_market_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The market variables set the URL's language, country and currency."""
        _set_market(monkeypatch, "de", "DE", "EUR")

        url = await _sent_url()

        assert url.endswith("&hl=de&gl=DE&curr=EUR")

    async def test_market_values_are_url_encoded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A market value with a space or an ampersand is URL-encoded."""
        _set_market(monkeypatch, "en GB", "GB&x=1", None)

        url = await _sent_url()

        assert url.endswith("&hl=en+GB&gl=GB%26x%3D1&curr=USD")


class TestBuildPrompt:
    """Tests for the flight extraction prompt."""

    @pytest.mark.parametrize("column", ("Departure", "Arrival", "Airline", "Stops", "Price", "Duration"))
    async def test_prompt_has_split_time_columns_and_stops(self, column: str) -> None:
        """The prompt asks for separate departure and arrival times plus stops."""
        prompt = await _sent_prompt()

        assert column in prompt

    async def test_prompt_asks_for_verbatim_times(self) -> None:
        """The prompt tells the model to transcribe the times without any change."""
        prompt = await _sent_prompt()

        assert "exactly as shown" in prompt
        assert "Never compute" in prompt

    async def test_prompt_asks_for_next_day_markers(self) -> None:
        """The prompt tells the model to keep next-day markers attached to their times."""
        prompt = await _sent_prompt()

        assert "next-day marker" in prompt

    @pytest.mark.parametrize(("count", "cap"), ((5, "at most 5"), (7, "at most 7")))
    async def test_prompt_caps_rows_at_count(self, count: int, cap: str) -> None:
        """The prompt caps the table at the requested count."""
        prompt = await _sent_prompt(count=count)

        assert cap in prompt

    async def test_prompt_includes_airline_filter_when_listed(self) -> None:
        """A named airline list lands in the prompt."""
        prompt = await _sent_prompt(airlines=["United", "Delta"])

        assert "United" in prompt
        assert "Delta" in prompt
        assert "case-insensitively" in prompt

    async def test_prompt_has_no_airline_filter_when_none(self) -> None:
        """No airline clause appears when no list is given."""
        prompt = await _sent_prompt()

        assert "United" not in prompt
        assert "case-insensitively" not in prompt


@pytest.mark.skipif(not is_llm_available(), reason="LLM server not available")
@skip_if_no_chrome
@pytest.mark.live
@pytest.mark.timeout(600)
class TestSearchLive:
    """Live flight search tests."""

    async def test_search_sfo_to_jfk(self) -> None:
        """A live flight search gives back text."""
        result = await run_text(google_flights_tool, "SFO", "JFK", _LIVE_DEPART_DATE)

        assert isinstance(result, str), f"unexpected return type: {type(result)}"
        assert result, "expected a non-empty result string"

    async def test_search_business_one_way(self) -> None:
        """A live business one-way search finds flights."""
        result = await run_text(google_flights_tool, "SFO", "JFK", _LIVE_ONE_WAY_DEPART_DATE, oneway=True, cabin="business")

        assert isinstance(result, str), f"unexpected return type: {type(result)}"
        assert result, "expected a non-empty result string"
        assert "No flights found" not in result
