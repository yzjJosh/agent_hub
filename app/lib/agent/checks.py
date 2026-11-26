"""Shared output checks for agent roles."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
import json
import re

__all__ = [
    "JsonObjectResult",
    "TextCheckResult",
    "check_choice",
    "check_nonblank_string",
    "check_string",
    "combine_problems",
    "extract_markdown_body",
    "parse_json_object",
]

_MAX_FILLER_LINES = 5
_MIN_FENCE_LENGTH = 3
_RAW_JSON_START_CHARACTERS = frozenset('{["-0123456789tfn')
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
_MARKDOWN_TITLE_PATTERN = re.compile(r"^# \S")


@dataclass(frozen=True)
class JsonObjectResult:
    """Result of parsing agent output as a JSON object."""

    is_valid: bool
    value: dict | None = None
    error: str = ""


@dataclass(frozen=True)
class TextCheckResult:
    """Result of checking or extracting a text value."""

    is_valid: bool
    value: str = ""
    error: str = ""


@dataclass(frozen=True)
class _PreambleMatch:
    is_wrapper: bool
    line_index: int
    offset: int


@dataclass(frozen=True)
class _FenceOpening:
    delimiter: str
    length: int


class _OutputFormatError(ValueError):
    pass


def _strip_line_ending(line: str) -> str:
    return line.rstrip("\r\n")


def _scan_preamble(
    lines: list[str],
    raw_starts: frozenset[str],
    is_markdown: bool,
) -> _PreambleMatch:
    filler_count = 0
    offset = 0
    for index, line in enumerate(lines):
        content = _strip_line_ending(line)
        stripped = content.lstrip()
        if not stripped:
            offset += len(line)
            continue
        first_character = stripped[0]
        if first_character in "`~":
            if content != stripped:
                error = _MARKDOWN_WRAPPER_ERROR if is_markdown else _JSON_WRAPPER_ERROR
                raise _OutputFormatError(error)
            return _PreambleMatch(is_wrapper=True, line_index=index, offset=offset)
        if first_character in raw_starts:
            start_offset = offset + len(content) - len(stripped)
            if is_markdown and (content != stripped or not _MARKDOWN_TITLE_PATTERN.match(content)):
                raise _OutputFormatError(_MARKDOWN_TITLE_ERROR)
            return _PreambleMatch(is_wrapper=False, line_index=index, offset=start_offset)
        filler_count += 1
        if filler_count > _MAX_FILLER_LINES:
            error = _MARKDOWN_TITLE_ERROR if is_markdown else _JSON_ERROR
            raise _OutputFormatError(error)
        offset += len(line)
    error = _MARKDOWN_TITLE_ERROR if is_markdown else _JSON_ERROR
    raise _OutputFormatError(error)


def _read_fence_opening(
    line: str,
    allowed_labels: frozenset[str],
    error: str,
) -> _FenceOpening:
    delimiter = line[0]
    length = len(line) - len(line.lstrip(delimiter))
    label = line[length:]
    if length < _MIN_FENCE_LENGTH or label not in allowed_labels:
        raise _OutputFormatError(error)
    return _FenceOpening(delimiter=delimiter, length=length)


def _is_closing_fence(line: str, delimiter: str, minimum_length: int) -> bool:
    return len(line) >= minimum_length and not line.strip(delimiter)


def _unwrap_fenced_block(
    lines: list[str],
    opening_index: int,
    allowed_labels: frozenset[str],
    error: str,
) -> str:
    opening_line = _strip_line_ending(lines[opening_index])
    fence = _read_fence_opening(opening_line, allowed_labels, error)
    for index in range(opening_index + 1, len(lines)):
        content = _strip_line_ending(lines[index])
        if not _is_closing_fence(content, fence.delimiter, fence.length):
            continue
        if "".join(lines[index + 1 :]).strip():
            raise _OutputFormatError(error)
        return "".join(lines[opening_index + 1 : index])
    raise _OutputFormatError(error)


def extract_markdown_body(text: str) -> TextCheckResult:
    """Extract the markdown report body from agent output.

    Args:
        text: The raw output text from an agent run.

    Returns:
        A text check result holding the report body when valid, or an error message.
    """
    lines = text.splitlines(keepends=True)
    try:
        preamble = _scan_preamble(lines, frozenset({"#"}), is_markdown=True)
        if not preamble.is_wrapper:
            return TextCheckResult(is_valid=True, value=text[preamble.offset :])
        wrapped = _unwrap_fenced_block(
            lines,
            preamble.line_index,
            frozenset({"markdown", "md"}),
            _MARKDOWN_WRAPPER_ERROR,
        )
        wrapped_lines = wrapped.splitlines(keepends=True)
        inner_preamble = _scan_preamble(wrapped_lines, frozenset({"#"}), is_markdown=True)
        if inner_preamble.is_wrapper or any(
            _strip_line_ending(line).strip() for line in wrapped_lines[: inner_preamble.line_index]
        ):
            raise _OutputFormatError(_MARKDOWN_TITLE_ERROR)
        return TextCheckResult(is_valid=True, value=wrapped[inner_preamble.offset :])
    except _OutputFormatError as error:
        return TextCheckResult(is_valid=False, error=str(error))


def combine_problems(messages: list[str]) -> str:
    """Join validation problem messages into a single message.

    Args:
        messages: One message per problem found, in report order.

    Returns:
        The single message when there is one problem, the combined message when there are several, or an empty string when there are none.
    """
    if not messages:
        return ""
    if len(messages) == 1:
        return messages[0]
    return "Multiple validation problems found. " + " ".join(messages)


def parse_json_object(text: str) -> JsonObjectResult:
    """Read agent output text as a JSON object.

    Args:
        text: The raw output text from an agent run.

    Returns:
        A JSON object result holding the parsed dictionary when valid, or an error message.
    """
    lines = text.splitlines(keepends=True)
    try:
        preamble = _scan_preamble(lines, _RAW_JSON_START_CHARACTERS, is_markdown=False)
        if preamble.is_wrapper:
            candidate = _unwrap_fenced_block(
                lines,
                preamble.line_index,
                frozenset({"", "json"}),
                _JSON_WRAPPER_ERROR,
            )
        else:
            candidate = text[preamble.offset :]
    except _OutputFormatError as error:
        return JsonObjectResult(is_valid=False, error=str(error))
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return JsonObjectResult(is_valid=False, error=_JSON_ERROR)
    if not isinstance(parsed, dict):
        return JsonObjectResult(is_valid=False, error="Output must be a JSON object.")
    return JsonObjectResult(is_valid=True, value=parsed)


def check_string(item: dict, field: str, error: str) -> TextCheckResult:
    """Check that a field of the object holds a string.

    Args:
        item: The object to read the field from.
        field: The name of the field.
        error: The message to give back when the check fails.

    Returns:
        A text check result holding the field string when valid, or the error message.
    """
    value = item.get(field)
    if not isinstance(value, str):
        return TextCheckResult(is_valid=False, error=error)
    return TextCheckResult(is_valid=True, value=value)


def check_nonblank_string(item: dict, field: str, error: str) -> TextCheckResult:
    """Check that a field of the object holds a nonblank string.

    Args:
        item: The object to read the field from.
        field: The name of the field.
        error: The message to give back when the check fails.

    Returns:
        A text check result holding the field string when valid, or the error message.
    """
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        return TextCheckResult(is_valid=False, error=error)
    return TextCheckResult(is_valid=True, value=value)


def check_choice(
    item: dict,
    field: str,
    allowed: Collection[str],
    error: str,
) -> TextCheckResult:
    """Check that a field of the object holds one of the allowed strings.

    Args:
        item: The object to read the field from.
        field: The name of the field.
        allowed: The strings the field may hold.
        error: The message to give back when the check fails.

    Returns:
        A text check result holding the matched string when valid, or the error message.
    """
    value = item.get(field)
    if not isinstance(value, str) or value not in allowed:
        return TextCheckResult(is_valid=False, error=error)
    return TextCheckResult(is_valid=True, value=value)
