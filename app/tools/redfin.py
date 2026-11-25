"""Search Redfin for properties by address, city, or zipcode.

Gives back a markdown table of price, beds, baths, sqft, and dates for
active listings and sold homes.
"""

from __future__ import annotations

__all__ = ["redfin_search_tool"]

import hashlib
import json
from datetime import timedelta
from typing import ClassVar, override

from app.lib.redfin.models import (
    DEFAULT_SEARCH_LIMIT,
    RedfinSearchError,
    SearchOptions,
    parse_iso_date,
)
from app.lib.redfin.search import search_redfin
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult


class _RedfinSearchTool(Tool):
    name: ClassVar[str] = "redfin_search"
    is_slow: ClassVar[bool] = True
    staleness: ClassVar[timedelta] = timedelta(hours=24)
    result_render_type: ClassVar[str] = "markdown"
    description: ClassVar[str] = (
        "Search Redfin for US properties by street address, city, or zipcode, with "
        "optional filters for sold status, price, bedrooms, bathrooms, and listing "
        "dates. For a specific street address, returns that single property, "
        "including off-market and sale-history data. Use for US real-estate "
        "listings and sale history."
    )
    summary: ClassVar[str] = "US property search via Redfin"
    params: ClassVar[tuple[Param, ...]] = (
        Param("location", str, description="Search location (street address, city, or zipcode)."),
        Param(
            "sold",
            bool,
            False,
            description="If True, search recently sold properties instead of active listings.",
        ),
        Param(
            "min_beds",
            int | None,
            None,
            description="Minimum number of bedrooms. Applied server-side via Redfin's Filters panel.",
        ),
        Param(
            "min_baths",
            int | None,
            None,
            description="Minimum number of bathrooms. Applied server-side via Redfin's Filters panel.",
        ),
        Param(
            "limit",
            int,
            DEFAULT_SEARCH_LIMIT,
            description="Maximum number of results to return.",
        ),
        Param(
            "min_price",
            int | None,
            None,
            description="Minimum price filter. Applied server-side via Redfin's Filters panel.",
        ),
        Param(
            "max_price",
            int | None,
            None,
            description="Maximum price filter. Applied server-side via Redfin's Filters panel.",
        ),
        Param(
            "date_from",
            str | None,
            None,
            kind="date",
            description=(
                "Start date for the date range filter. Applied server-side via 'Time on Redfin' "
                "for active listings (maximum 30 days, beyond that it falls back to "
                "client-side filtering), or via 'Sold within' for sold listings."
            ),
        ),
        Param(
            "date_to",
            str | None,
            None,
            kind="date",
            description="End date for the date range filter. Always applied client-side.",
        ),
    )
    rules: ClassVar[tuple[str, ...]] = (
        "order:date_from:date_to",
        "order:min_price:max_price",
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        location = params["location"]
        if not location or not location.strip():
            raise ToolMessageError(
                "Redfin search failed: location must be a non-empty string"
            )
        date_from = params.get("date_from")
        date_to = params.get("date_to")
        for name, value in (("date_from", date_from), ("date_to", date_to)):
            if value is not None and parse_iso_date(value) is None:
                raise ToolMessageError(
                    f"Redfin search failed: {name} must be a YYYY-MM-DD date: '{value}'"
                )

        options = SearchOptions(
            is_sold=params.get("sold", False),
            min_beds=params.get("min_beds"),
            min_baths=params.get("min_baths"),
            limit=params.get("limit", DEFAULT_SEARCH_LIMIT),
            min_price=params.get("min_price"),
            max_price=params.get("max_price"),
            date_from=date_from,
            date_to=date_to,
        )
        try:
            return ToolResult(text=await search_redfin(location, options))
        except RedfinSearchError as exc:
            raise RuntimeError(f"Redfin search failed: {exc}")

    @override
    async def cache_key(self, params: dict) -> str:
        payload = json.dumps(
            {
                "location": params["location"],
                "sold": params.get("sold", False),
                "min_beds": params.get("min_beds"),
                "min_baths": params.get("min_baths"),
                "limit": params.get("limit", DEFAULT_SEARCH_LIMIT),
                "min_price": params.get("min_price"),
                "max_price": params.get("max_price"),
                "date_from": params.get("date_from"),
                "date_to": params.get("date_to"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


redfin_search_tool = _RedfinSearchTool()
