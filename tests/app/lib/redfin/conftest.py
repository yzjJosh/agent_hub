"""Shared helpers and test data for Redfin library unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

CSV_HEADER = (
    '"SALE TYPE","PRICE","BEDS","BATHS","ADDRESS","CITY","STATE OR PROVINCE",'
    '"ZIP OR POSTAL CODE","SQUARE FEET","LOT SIZE","YEAR BUILT","DAYS ON MARKET",'
    '"$/SQUARE FEET","HOA/MONTH","STATUS","PROPERTY TYPE","SOLD DATE",'
    '"NEXT OPEN HOUSE START TIME","NEXT OPEN HOUSE END TIME","URL","SOURCE",'
    '"MLS#","LATITUDE","LONGITUDE"\n'
)


def make_fake_snapshot(
    actionable: list[dict], tree: dict | None = None
) -> MagicMock:
    """Create a fake page snapshot with async interaction methods.

    Args:
        actionable: Elements returned by actionable_elements.
        tree: Tree returned by filtered_tree.

    Returns:
        A configured mock snapshot.
    """
    snapshot = MagicMock()
    snapshot.actionable_elements.return_value = actionable
    snapshot.filtered_tree = AsyncMock(return_value=tree)
    snapshot.click = AsyncMock(return_value=True)
    snapshot.fill = AsyncMock(return_value=True)
    snapshot.select_option = AsyncMock(return_value=True)
    snapshot.press_key = AsyncMock(return_value=True)
    return snapshot
