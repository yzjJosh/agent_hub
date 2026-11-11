import pytest

from app.lib.tool.truncation import truncate

_TASK_ID = "tid-42"


class TestTruncate:
    """The result is cut to max_char and a note says how to read the rest."""

    @pytest.mark.parametrize(
        "params",
        [{}, {"max_char": None}, {"max_char": "10"}, {"max_char": True}],
        ids=["absent", "none", "string", "bool"],
    )
    def test_returns_result_unchanged(self, params: dict) -> None:
        assert truncate("hello", params, _TASK_ID) == "hello"

    def test_returns_result_unchanged_when_no_longer_than_max_char(self) -> None:
        assert truncate("x" * 499, {"max_char": 500}, _TASK_ID) == "x" * 499

    def test_returns_result_unchanged_when_equal_to_max_char(self) -> None:
        assert truncate("x" * 500, {"max_char": 500}, _TASK_ID) == "x" * 500

    def test_truncates_and_appends_the_exact_notice(self) -> None:
        expected = (
            "x" * 500
            + "\n[Truncated: showing the first 500 of 10000 chars. "
            + f"task_id={_TASK_ID} — fetch the remaining 9500 chars with "
            + f"task(action='result', task_id='{_TASK_ID}', offset=500, max_char=9500).]"
        )
        assert truncate("x" * 10000, {"max_char": 500}, _TASK_ID) == expected
