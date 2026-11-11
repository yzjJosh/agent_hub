"""Tests for app.lib.shared.url module."""

from typing import Self

import pytest

from app.lib.shared.url import validate_url


class TestValidateUrl:
    """Test the validate_url function."""

    @pytest.mark.parametrize("url", ["https://example.com/page", "http://127.0.0.1:8315/dav"])
    def test_valid_url_returns_none(self: Self, url: str) -> None:
        assert validate_url(url) is None

    @pytest.mark.parametrize("url", ["", "example.com/page"])
    def test_missing_scheme_reports_error(self: Self, url: str) -> None:
        assert validate_url(url) == f"Invalid URL: '{url}' is missing a scheme"

    def test_missing_hostname_reports_error(self: Self) -> None:
        assert validate_url("http://") == "Invalid URL: 'http://' is missing a hostname"
