"""Shared output checks for agent roles."""

from __future__ import annotations

import json
import re
from collections.abc import Collection

__all__ = [
    "check_choice",
    "check_nonblank_string",
    "check_string",
    "combine_problems",
    "normalize_markdown_body",
    "parse_json_object",
]

_MAX_FILLER_LINES = 5
_RAW_JSON_FIRST_CHARS = frozenset('{["-0123456789tfn')
_JSON_ERROR = (
    "Output must be the pure JSON value only, the JSON array or object itself, "
    "with no extra text, no markdown wrapper, and no commentary."
)
_JSON_WRAPPER_ERROR = (
    "JSON wrapper is malformed. Output must be the pure JSON value only, "
    "the JSON array or object itself, with no extra text, no markdown wrapper, and no commentary."
)
_MARKDOWN_TITLE_ERROR = "Report must start with a markdown title (a first line starting with '# ')."
_MARKDOWN_WRAPPER_ERROR = (
    "Markdown wrapper is malformed. Output must be the markdown report itself, "
    "with no markdown wrapper, no extra text, and no commentary."
)
_TITLE_RE = re.compile(r"^# \S")
_SCALAR_VALUE = "scalar JSON value"


class _NormalizationError(ValueError):
    pass


def _line_content(line: str) -> str:
    return line.rstrip("\r\n")


def _scan_preamble(
    lines: list[str], raw_starts: frozenset[str], is_markdown: bool
) -> tuple[str, int, int]:
    filler_count = 0
    offset = 0
    for index, line in enumerate(lines):
        content = _line_content(line)
        stripped = content.lstrip()
        if not stripped:
            offset += len(line)
            continue
        first = stripped[0]
        if first in "`~":
            if content != stripped:
                error = _MARKDOWN_WRAPPER_ERROR if is_markdown else _JSON_WRAPPER_ERROR
                raise _NormalizationError(error)
            return "wrapper", index, offset
        if first in raw_starts:
            start = offset + len(content) - len(stripped)
            if is_markdown and (content != stripped or not _TITLE_RE.match(content)):
                raise _NormalizationError(_MARKDOWN_TITLE_ERROR)
            return "raw", index, start
        filler_count += 1
        if filler_count > _MAX_FILLER_LINES:
            error = _MARKDOWN_TITLE_ERROR if is_markdown else _JSON_ERROR
            raise _NormalizationError(error)
        offset += len(line)
    error = _MARKDOWN_TITLE_ERROR if is_markdown else _JSON_ERROR
    raise _NormalizationError(error)


def _read_opening(line: str, allowed_labels: frozenset[str], error: str) -> tuple[str, int]:
    delimiter = line[0]
    length = len(line) - len(line.lstrip(delimiter))
    label = line[length:]
    if length < 3 or label not in allowed_labels:
        raise _NormalizationError(error)
    return delimiter, length


def _is_closing(line: str, delimiter: str, minimum_length: int) -> bool:
    return len(line) >= minimum_length and not line.strip(delimiter)


def _unwrap(
    lines: list[str],
    opening_index: int,
    allowed_labels: frozenset[str],
    error: str,
) -> str:
    opening = _line_content(lines[opening_index])
    delimiter, minimum_length = _read_opening(opening, allowed_labels, error)
    for index in range(opening_index + 1, len(lines)):
        content = _line_content(lines[index])
        if not _is_closing(content, delimiter, minimum_length):
            continue
        if "".join(lines[index + 1 :]).strip():
            raise _NormalizationError(error)
        return "".join(lines[opening_index + 1 : index])
    raise _NormalizationError(error)


def _normalize_json(text: str) -> tuple[bool, str]:
    lines = text.splitlines(keepends=True)
    try:
        kind, opening_index, start = _scan_preamble(lines, _RAW_JSON_FIRST_CHARS, False)
        if kind == "raw":
            return True, text[start:]
        return True, _unwrap(
            lines,
            opening_index,
            frozenset({"", "json"}),
            _JSON_WRAPPER_ERROR,
        )
    except _NormalizationError as exc:
        return False, str(exc)


def _parse_json(text: str) -> tuple[bool, list | dict | str]:
    ok, normalized = _normalize_json(text)
    if not ok:
        return False, normalized
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError:
        return False, _JSON_ERROR
    if not isinstance(parsed, (list, dict)):
        return True, _SCALAR_VALUE
    return True, parsed


def normalize_markdown_body(text: str) -> tuple[bool, str]:
    """Read the report body from drafter output.

    Args:
        text: The raw output text from a drafter run.

    Returns:
        An ok flag and a value, the report body when ok, else the error message.
    """
    lines = text.splitlines(keepends=True)
    try:
        kind, opening_index, start = _scan_preamble(lines, frozenset({"#"}), True)
        if kind == "raw":
            return True, text[start:]
        wrapped = _unwrap(
            lines,
            opening_index,
            frozenset({"markdown", "md"}),
            _MARKDOWN_WRAPPER_ERROR,
        )
        wrapped_lines = wrapped.splitlines(keepends=True)
        wrapped_kind, body_index, wrapped_start = _scan_preamble(
            wrapped_lines, frozenset({"#"}), True
        )
        if wrapped_kind != "raw" or any(
            _line_content(line).strip() for line in wrapped_lines[:body_index]
        ):
            raise _NormalizationError(_MARKDOWN_TITLE_ERROR)
        return True, wrapped[wrapped_start:]
    except _NormalizationError as exc:
        return False, str(exc)


def combine_problems(messages: list[str]) -> str:
    """Join the validation problems into a single message.

    Args:
        messages: One message per problem found, in report order.

    Returns:
        The single message when there is one problem, the combined message when there are several, empty when there are none.
    """
    if not messages:
        return ""
    if len(messages) == 1:
        return messages[0]
    return "Multiple validation problems found. " + " ".join(messages)


def parse_json_object(text: str) -> tuple[bool, dict | str]:
    """Read the text as a JSON object.

    Args:
        text: The raw output text from an agent run.

    Returns:
        An ok flag and a value, the parsed dict when ok, else the error message.
    """
    ok, parsed = _parse_json(text)
    if not ok:
        return False, parsed
    if not isinstance(parsed, dict):
        return False, "Output must be a JSON object."
    return True, parsed


def check_string(item: dict, field: str, error: str) -> tuple[bool, str]:
    """Check that a field of the object holds a string.

    Args:
        item: The object to read the field from.
        field: The name of the field.
        error: The message to give back when the check fails.

    Returns:
        An ok flag and the error message, empty when ok.
    """
    if not isinstance(item.get(field), str):
        return False, error
    return True, ""


def check_nonblank_string(item: dict, field: str, error: str) -> tuple[bool, str]:
    """Check that a field of the object holds a string with something in it.

    Args:
        item: The object to read the field from.
        field: The name of the field.
        error: The message to give back when the check fails.

    Returns:
        An ok flag and the error message, empty when ok.
    """
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        return False, error
    return True, ""


def check_choice(item: dict, field: str, allowed: Collection[str], error: str) -> tuple[bool, str]:
    """Check that a field of the object holds one of the allowed strings.

    Args:
        item: The object to read the field from.
        field: The name of the field.
        allowed: The strings the field may hold.
        error: The message to give back when the check fails.

    Returns:
        An ok flag and the error message, empty when ok.
    """
    value = item.get(field)
    if not isinstance(value, str) or value not in allowed:
        return False, error
    return True, ""
