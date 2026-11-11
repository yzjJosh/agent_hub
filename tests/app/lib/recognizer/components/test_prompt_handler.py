"""Tests for pulling the label out of a captcha prompt."""

import pytest

from app.lib.recognizer.components.prompt_handler import split_prompt_message

_SPLIT_AND_CLEAN_CASES = [
    ("dog", "dog"),
    ("  cat  ", "cat"),
    ("challenge with a dog image", "dog"),
    ("challenge with dog image", "dog"),
    ("а picture", "a picture"),
    ("a challenge with a сat", "cat"),
    ("no keyword here", "no keyword here"),
    ("hello", "hello"),
    ("", ""),
    ("а", "a"),
    ("е", "e"),
    ("і", "i"),
    ("с", "c"),
    ("р", "p"),
    ("х", "x"),
    ("у", "y"),
    ("hello world", "hello world"),
    ("abаcdе", "abacde"),
]


@pytest.mark.parametrize("label,expected", _SPLIT_AND_CLEAN_CASES)
def test_split_prompt_message(label: str, expected: str) -> None:
    """The label is pulled out and lookalike letters become plain ones."""
    assert split_prompt_message(label) == expected