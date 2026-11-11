import pytest

from app.lib.agent.checks import (
    check_choice,
    check_nonblank_string,
    check_string,
    combine_problems,
    normalize_markdown_body,
    parse_json_object,
)

_ALLOWED = frozenset({"rewrite", "collect", "approve"})


class TestParseJsonObject:
    @pytest.mark.parametrize(
        "text",
        [
            '{"pass": true, "reason": "ok"}',
            "I prepared the result.\n\n{\"pass\": true, \"reason\": \"ok\"}",
            '```json\n{"pass": true, "reason": "ok"}\n```',
            '~~~~\n{"pass": true, "reason": "ok"}\n~~~~~',
        ],
    )
    def test_object_is_parsed(self, text: str) -> None:
        ok, value = parse_json_object(text)
        assert ok is True
        assert value == {"pass": True, "reason": "ok"}

    @pytest.mark.parametrize(
        "text",
        [
            "one\ntwo\nthree\nfour\nfive\nsix\n{}",
            "Prefix: {}",
            "{broken\n{}",
            "{}\n{}",
            "null\n{}",
            "true\n{}",
            "7\n{}",
            '"text"\n{}',
            "",
            '{"a": 1',
        ],
    )
    def test_ambiguous_or_extra_json_is_rejected(self, text: str) -> None:
        ok, value = parse_json_object(text)
        assert ok is False
        assert isinstance(value, str)

    @pytest.mark.parametrize(
        "text",
        [
            " ```json\n{}\n```",
            "``json\n{}\n``",
            "``~json\n{}\n```",
            "``` json\n{}\n```",
            "```JSON\n{}\n```",
            "```python\n{}\n```",
            "```json extra\n{}\n```",
            "```json \n{}\n```",
            "```json\n{}\n~~~",
            "```json\n{}",
            "```json\n{}\n```\ntrailing",
        ],
    )
    def test_malformed_json_wrapper_is_rejected(self, text: str) -> None:
        ok, value = parse_json_object(text)
        assert ok is False
        assert isinstance(value, str)

    @pytest.mark.parametrize(
        "text",
        ["[1, 2]", '"text"', "7", "-2", "null", "true", "false"],
    )
    def test_non_object_json_is_rejected(self, text: str) -> None:
        ok, value = parse_json_object(text)
        assert ok is False
        assert value == "Output must be a JSON object."


class TestNormalizeMarkdownBody:
    _BODY: str = "# Report\n\nThe market grew steadily this year across all product segments."

    @pytest.mark.parametrize(
        "text,expected",
        [
            (_BODY, _BODY),
            ("\nHere is the report.\n\n# Report\n\nBody text.", "# Report\n\nBody text."),
            (
                "First note.\n\nSecond note.\nThird note.\n# Report\n\nBody text.",
                "# Report\n\nBody text.",
            ),
            (
                "First note.\n\nSecond note.\nThird note.\nFourth note.\n# Report\n\nBody text.",
                "# Report\n\nBody text.",
            ),
            (
                "First note.\nSecond note.\nThird note.\nFourth note.\nFifth note.\n# Report\n\nBody text.",
                "# Report\n\nBody text.",
            ),
            (f"```markdown\n{_BODY}\n```", _BODY + "\n"),
            (f"~~~md\n{_BODY}\n~~~~", _BODY + "\n"),
            (
                "````markdown\n# Report\n\n```python\nprint('ok')\n```\n````",
                "# Report\n\n```python\nprint('ok')\n```\n",
            ),
            (
                "~~~markdown\n# Report\n\n```python\nprint('ok')\n```\n~~~",
                "# Report\n\n```python\nprint('ok')\n```\n",
            ),
        ],
    )
    def test_body_is_normalized(self, text: str, expected: str) -> None:
        assert normalize_markdown_body(text) == (True, expected)

    @pytest.mark.parametrize(
        "text",
        [
            "one\ntwo\nthree\nfour\nfive\nsix\n# Report\n\nBody text.",
            "No report follows.",
            "  # Report\n\nBody text.",
            "#Report\n\nBody text.",
            "#  Report\n\nBody text.",
            "## Report\n\nBody text.",
            "```\n# Report\n\nBody text.\n```",
            "```text\n# Report\n\nBody text.\n```",
            " ```markdown\n# Report\n\nBody text.\n```",
            "``markdown\n# Report\n\nBody text.\n``",
            "``~markdown\n# Report\n\nBody text.\n```",
            "``` markdown\n# Report\n\nBody text.\n```",
            "```Markdown\n# Report\n\nBody text.\n```",
            "```markdown extra\n# Report\n\nBody text.\n```",
            "```markdown \n# Report\n\nBody text.\n```",
            "```markdown\n# Report\n\nBody text.\n~~~",
            "```markdown\n# Report\n\nBody text.",
            "```markdown\n# Report\n\nBody text.\n```\ntrailing",
            "```markdown\n# Report\n\n```\nmore report\n```",
            "```markdown\nFiller inside wrapper.\n# Report\n\nBody text.\n```",
        ],
    )
    def test_invalid_body_boundary_is_rejected(self, text: str) -> None:
        ok, value = normalize_markdown_body(text)
        assert ok is False
        assert isinstance(value, str)


class TestCheckString:
    @pytest.mark.parametrize("item", [{"summary": "a note"}, {"summary": ""}, {"summary": "  "}])
    def test_string_field_passes(self, item: dict) -> None:
        assert check_string(item, "summary", "bad summary.") == (True, "")

    @pytest.mark.parametrize("item", [{}, {"summary": None}, {"summary": 7}, {"summary": ["a"]}])
    def test_missing_or_non_string_field_is_rejected(self, item: dict) -> None:
        assert check_string(item, "summary", "bad summary.") == (False, "bad summary.")


class TestCheckNonblankString:
    def test_filled_string_field_passes(self) -> None:
        assert check_nonblank_string({"reason": "all backed"}, "reason", "bad reason.") == (True, "")

    @pytest.mark.parametrize(
        "item",
        [{}, {"reason": ""}, {"reason": "   \n\t"}, {"reason": None}, {"reason": 7}],
    )
    def test_missing_blank_or_non_string_field_is_rejected(self, item: dict) -> None:
        assert check_nonblank_string(item, "reason", "bad reason.") == (False, "bad reason.")


class TestCheckChoice:
    def test_allowed_value_passes(self) -> None:
        assert check_choice({"action": "approve"}, "action", _ALLOWED, "bad action.") == (True, "")

    @pytest.mark.parametrize(
        "item",
        [{}, {"action": "reject"}, {"action": ""}, {"action": None}, {"action": 7}],
    )
    def test_value_outside_the_set_is_rejected(self, item: dict) -> None:
        assert check_choice(item, "action", _ALLOWED, "bad action.") == (False, "bad action.")

    @pytest.mark.parametrize("value", [["approve"], {"action": "approve"}])
    def test_unhashable_value_is_rejected_without_raising(self, value: list | dict) -> None:
        assert check_choice({"action": value}, "action", _ALLOWED, "bad action.") == (
            False,
            "bad action.",
        )


class TestCombineProblems:
    @pytest.mark.parametrize(
        "messages,expected",
        [
            ([], ""),
            (["Only one."], "Only one."),
            (
                ["First.", "Second.", "Third."],
                "Multiple validation problems found. First. Second. Third.",
            ),
        ],
    )
    def test_problems_combine(self, messages: list[str], expected: str) -> None:
        assert combine_problems(messages) == expected
