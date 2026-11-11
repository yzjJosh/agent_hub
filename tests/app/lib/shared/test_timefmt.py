"""Tests for app.lib.shared.timefmt module."""

import time
from collections.abc import Iterator
from datetime import datetime
from typing import Self

import pytest

from app.lib.shared.timefmt import LOCAL_TIME_FORMAT, iso_to_local


@pytest.fixture
def _utc_clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Set the local timezone to UTC while a test runs."""
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


class TestIsoToLocal:
    """Test the iso_to_local function."""

    def test_returns_none_for_none(self: Self) -> None:
        """A missing timestamp stays missing."""
        assert iso_to_local(None) is None

    def test_formats_utc_timestamp(self: Self, _utc_clock: None) -> None:
        """A UTC timestamp keeps its wall clock time."""
        assert iso_to_local("2026-01-14T08:30:05+00:00") == "2026-01-14 08:30:05"

    def test_shifts_offset_timestamp(self: Self, _utc_clock: None) -> None:
        """A timestamp with an offset is moved to local time."""
        assert iso_to_local("2026-01-14T08:30:05+02:00") == "2026-01-14 06:30:05"

    def test_output_matches_local_time_format(self: Self, _utc_clock: None) -> None:
        """The text parses back with LOCAL_TIME_FORMAT."""
        formatted = iso_to_local("2026-01-14T08:30:05+00:00")
        assert datetime.strptime(formatted, LOCAL_TIME_FORMAT).second == 5
