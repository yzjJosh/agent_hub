"""Look up flights on Google Flights."""

from __future__ import annotations

__all__ = ["google_flights_tool"]

import base64
import os
import sys
import urllib.parse
from typing import ClassVar, override

from app.lib.browser.analyze_page import BrowserError, analyze_page
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult

_VERBOSE = True
_DATE_LENGTH = 10
_DATE_FORMAT_PLACEHOLDER = "YYYY-MM-DD"
_DATE_HYPHEN_INDEX_YEAR = 4
_DATE_HYPHEN_INDEX_MONTH = 7
_TFS_PRODUCT_CODE = 28
_TFS_SCHEMA_CONSTANT = 2
_TFS_PASSENGER_CONSTANT = 1
_TFS_ROUND_TRIP = 1
_TFS_ONE_WAY = 2
_TFS_NO_STOPS = 0
_TFS_DATE_MASK = 0xFFFFFFFFFFFFFFFF
_SEARCH_URL = "https://www.google.com/travel/flights/search"
_DEFAULT_LANGUAGE = "en"
_DEFAULT_COUNTRY = "US"
_DEFAULT_CURRENCY = "USD"

# cabin param value to (f9 cabin code, f25 basic flag).
_CABIN_FLAGS: dict[str | None, tuple[int, int]] = {
    None: (1, 0),
    "economy_excl_basic": (1, 1),
    "premium_economy": (2, 1),
    "business": (3, 1),
    "first": (4, 1),
}


def _encode_varint(value: int) -> bytes:
    """Encode an unsigned integer as a protobuf varint."""
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _encode_varint_field(field: int, value: int) -> bytes:
    """Encode a varint field."""
    return _encode_varint(field << 3) + _encode_varint(value)


def _encode_bytes_field(field: int, data: bytes) -> bytes:
    """Encode a length-delimited field."""
    return _encode_varint((field << 3) | 2) + _encode_varint(len(data)) + data


def _encode_string_field(field: int, text: str) -> bytes:
    """Encode a string field."""
    return _encode_bytes_field(field, text.encode("utf-8"))


def _build_segment(date: str, origin: str, destination: str, nonstop: bool) -> bytes:
    """Encode one trip segment of the search protobuf."""
    data = _encode_string_field(2, date)
    if nonstop:
        data += _encode_varint_field(5, _TFS_NO_STOPS)
    data += _encode_bytes_field(13, _encode_varint_field(1, 1) + _encode_string_field(2, origin))
    data += _encode_bytes_field(14, _encode_varint_field(1, 1) + _encode_string_field(2, destination))
    return data


def _build_tfs(
    origin: str,
    destination: str,
    depart_date: str,
    nonstop: bool,
    oneway: bool,
    return_date: str | None,
    cabin: str | None,
) -> bytes:
    """Encode the search protobuf the flights site expects."""
    cabin_code, basic_flag = _CABIN_FLAGS[cabin]
    data = _encode_varint_field(1, _TFS_PRODUCT_CODE) + _encode_varint_field(2, _TFS_SCHEMA_CONSTANT)
    data += _encode_bytes_field(3, _build_segment(depart_date, origin, destination, nonstop))
    round_trip = not oneway and bool(return_date)
    if round_trip:
        data += _encode_bytes_field(3, _build_segment(return_date, destination, origin, nonstop))
    data += _encode_varint_field(8, _TFS_PASSENGER_CONSTANT)
    data += _encode_varint_field(9, cabin_code)
    data += _encode_varint_field(14, 1)
    data += _encode_bytes_field(16, _encode_varint_field(1, _TFS_DATE_MASK))
    data += _encode_varint_field(19, _TFS_ROUND_TRIP if round_trip else _TFS_ONE_WAY)
    data += _encode_varint_field(25, basic_flag)
    return data


def _build_tfs_url(
    origin: str,
    destination: str,
    depart_date: str,
    nonstop: bool,
    oneway: bool,
    return_date: str | None,
    cabin: str | None,
) -> str:
    """Return the search URL for a trip."""
    token = base64.urlsafe_b64encode(_build_tfs(origin, destination, depart_date, nonstop, oneway, return_date, cabin)).rstrip(b"=").decode("ascii")
    query = urllib.parse.urlencode({
        "tfs": token,
        "hl": os.getenv("GOOGLE_FLIGHTS_LANGUAGE") or _DEFAULT_LANGUAGE,
        "gl": os.getenv("GOOGLE_FLIGHTS_COUNTRY") or _DEFAULT_COUNTRY,
        "curr": os.getenv("GOOGLE_FLIGHTS_CURRENCY") or _DEFAULT_CURRENCY,
    })
    return f"{_SEARCH_URL}?{query}"


def _build_prompt(origin: str, destination: str, depart_date: str, airlines: list[str] | None, count: int) -> str:
    """Return the question to ask about the flights page."""
    airline_clause = ""
    if airlines:
        airline_clause = f" Only include flights operated by {', '.join(airlines)}, matched case-insensitively."
    return (
        f"Extract flight options (Top departing or Best) from {origin} to {destination} "
        f"on {depart_date}. Return ONLY a Markdown table with columns: "
        f"Departure, Arrival, Airline, Stops, Price, Duration. "
        f"Departure and Arrival are the departure and arrival times. "
        f"Transcribe the times exactly as shown on the page. "
        f"Never compute, normalize, or convert a time. "
        f"Keep a next-day marker, like +1, attached to the time it belongs to. "
        f"Stops is 'Nonstop' or the number of stops, like '1 stop'. "
        f"Include at most {count} rows."
        f"{airline_clause} "
        f"If no flights are found, say 'No flights found'. Do not include any other text."
    )


async def _browser_search(
    origin: str,
    destination: str,
    depart_date: str,
    nonstop: bool,
    oneway: bool,
    return_date: str | None,
    cabin: str | None,
    airlines: list[str] | None,
    count: int,
) -> str:
    """Open the flights page and read the flight options off it.

    Returns:
        A table of flights, or a line saying no flights were found.

    Raises:
        RuntimeError: If the page could not be read.
    """
    search_url = _build_tfs_url(origin, destination, depart_date, nonstop, oneway, return_date, cabin)
    prompt = _build_prompt(origin, destination, depart_date, airlines, count)

    try:
        response, _ = await analyze_page(
            url=search_url,
            prompt=prompt,
            verbose=_VERBOSE,
        )
    except BrowserError as e:
        print(f"[google_flights] ERROR: {e}", file=sys.stderr)
        raise RuntimeError(f"google_flights failed: {e}") from e

    return response.strip()


def _is_valid_date(date: str) -> bool:
    """Return True if the date is written as YYYY-MM-DD."""
    if len(date) != _DATE_LENGTH:
        return False
    if date[_DATE_HYPHEN_INDEX_YEAR] != "-" or date[_DATE_HYPHEN_INDEX_MONTH] != "-":
        return False
    parts = date.split("-")
    return len(parts) == 3 and all(part.isdigit() for part in parts)


def _validate_input(
    origin: str,
    destination: str,
    depart_date: str,
    return_date: str | None,
    oneway: bool,
) -> str | None:
    """Check the search inputs.

    Returns:
        An error message, or None when the inputs are fine.
    """
    if not origin:
        return "Error: origin is required."
    if not destination:
        return "Error: destination is required."
    if not depart_date:
        return "Error: departure date is required."
    if not _is_valid_date(depart_date):
        return (
            f"Error: invalid departure date '{depart_date}'. "
            f"Expected format {_DATE_FORMAT_PLACEHOLDER}."
        )
    if return_date:
        if oneway:
            return "Error: return_date cannot be used with oneway."
        if not _is_valid_date(return_date):
            return (
                f"Error: invalid date '{return_date}'. "
                f"Expected format {_DATE_FORMAT_PLACEHOLDER}."
            )
        if return_date < depart_date:
            return f"Error: return_date '{return_date}' is before departure date '{depart_date}'."
    return None


class _GoogleFlightsTool(Tool):
    name: ClassVar[str] = "google_flights"
    result_render_type: ClassVar[str] = "markdown"
    description: ClassVar[str] = (
        "Search for real-time flight information, prices, and schedules using Google "
        "Flights. Use when the user needs accurate, live airline data for specific "
        "routes and dates."
    )
    summary: ClassVar[str] = "Live flight options and prices"
    params: ClassVar[tuple[Param, ...]] = (
        Param("origin", str, description='3-letter airport code (e.g., "SFO").'),
        Param("destination", str, description='3-letter airport code (e.g., "JFK").'),
        Param("depart_date", str, kind="date", description="Departure date."),
        Param("nonstop", bool, False, description="When True, restrict results to non-stop flights."),
        Param("oneway", bool, False, description="When True, show one-way rather than round-trip options."),
        Param("return_date", str | None, None, kind="date", description="Round-trip return date."),
        Param(
            "cabin",
            str | None,
            None,
            enum=("economy_excl_basic", "premium_economy", "business", "first"),
            description="Cabin class to search. When omitted, Google's default economy (including basic) is used.",
        ),
        Param("airlines", list[str] | None, None, description="Airline names to restrict the results to, matched case-insensitively."),
        Param("count", int, 5, min=1, max=10, description="The maximum number of flights returned."),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        origin = params["origin"]
        destination = params["destination"]
        depart_date = params["depart_date"]
        nonstop = params.get("nonstop", False)
        oneway = params.get("oneway", False)
        return_date = params.get("return_date")
        cabin = params.get("cabin")
        airlines = params.get("airlines")
        count = params.get("count", 5)
        error = _validate_input(origin, destination, depart_date, return_date, oneway)
        if error:
            raise ToolMessageError(error)
        return ToolResult(
            text=await _browser_search(origin, destination, depart_date, nonstop, oneway, return_date, cabin, airlines, count)
        )


google_flights_tool = _GoogleFlightsTool()
