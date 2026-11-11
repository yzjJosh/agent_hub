"""Tests for app.lib.shared.tokens module."""

from typing import Self

import pytest
import tiktoken

from app.lib.shared.tokens import count_tokens


class TestCountTokens:
    """Test the count_tokens function."""

    @pytest.mark.parametrize("text,expected_min", [
        ("", 0),
        (None, 0),
        ("Hello", 1),
        ("Hello world", 2),
        ("This is a longer sentence with more tokens.", 8),
    ])
    def test_basic(self: Self, text: str | None, expected_min: int) -> None:
        """Each piece of text counts at least the expected number of tokens."""
        assert count_tokens(text) >= expected_min

    def test_scaling(self: Self) -> None:
        """Longer text counts more tokens."""
        short = "Hello"
        long = "Hello world, this is a much longer string that should have more tokens."
        assert count_tokens(long) > count_tokens(short)

    def test_special_chars(self: Self) -> None:
        """Text with symbols and emoji counts more than five tokens."""
        text = "Hello! 🚀 How are you? @#$%"
        assert count_tokens(text) > 5

    def test_tiktoken_fallback(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the counter breaks, the count falls back to the number of words."""
        def _mock_get_encoding(_name: str) -> None:
            raise Exception("Tiktoken error")

        monkeypatch.setattr(tiktoken, "get_encoding", _mock_get_encoding)

        text = "Hello World"
        assert count_tokens(text) == 2
