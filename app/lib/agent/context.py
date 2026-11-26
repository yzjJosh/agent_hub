"""Detect context overflow errors and shrink message history."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import json
import mimetypes
import re
import sqlite3
import sys

from agents import ModelBehaviorError
import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    LengthFinishReasonError,
)

from app.lib.agent import checks
from app.lib.db.store import context_error_store
from app.lib.reason.reason import reason
from app.lib.shared import config
from app.lib.shared.paths import is_video_mime

__all__ = [
    "ClassificationError",
    "CompactionError",
    "CompactHook",
    "compact",
    "count_chars",
    "extract_reasoning_text",
    "is_overflow",
    "resolve_video_url",
]

CompactHook = Callable[[list[dict]], list[dict] | None | Awaitable[list[dict] | None]]

_PAYLOAD_TOO_LARGE_STATUS = 413
_INTERNAL_ERROR_STATUS = 500
_EXCLUDED_STATUS_CODES = frozenset({401, 403, 404, 408, 409, 429, 502, 503, 504})
_OVERFLOW_ERROR_CODE = "context_length_exceeded"
_STREAM_ERROR_DEFAULT_MESSAGE = "An error occurred during streaming"
_HALF_REDUCTION_FACTOR = 2
_LOG_PREVIEW_LENGTH = 100
_DATA_URL_PREFIX = "data:"
_EMPTY_ARGUMENTS_FALLBACK = "{}"
_TEXT_ONLY_ROLES = frozenset({"system", "developer", "assistant"})
_NUMBER_PATTERN = re.compile(r"[0-9]+(?:[.,][0-9]+)*")

_OVERFLOW_SCHEMA = {
    "type": "object",
    "properties": {
        "is_context_overflow": {"type": "boolean"},
    },
    "required": ["is_context_overflow"],
    "additionalProperties": False,
}

_CLASSIFIER_PROMPT = (
    "Classify whether the following LLM API error message indicates that the input prompt, "
    "conversation history, or combined input and output token count exceeded the model's "
    "maximum context length or token window.\n"
    "- Set is_context_overflow to true ONLY when the error reports exceeding the model's "
    "context window, maximum context length, prompt token limit, or input token limit.\n"
    "- Set is_context_overflow to false for all other errors, such as invalid parameters, "
    "malformed JSON or tool arguments, unsupported fields, rate limits, authentication, "
    "missing models, content policy filters, or internal server crashes.\n\n"
    "Normalized error message:\n"
)


class ClassificationError(RuntimeError):
    """Raised when an error cannot be checked for context overflow."""


class CompactionError(RuntimeError):
    """Raised when message history cannot shrink by at least half."""


@dataclass(frozen=True)
class _ServerPayload:
    dict_body: dict | None = None
    list_body: list | None = None
    text_body: str | None = None
    response_text: str = ""
    is_json: bool = False


@dataclass(frozen=True)
class _DefaultCompactionResult:
    items: list[dict]
    remaining_characters: int
    is_accepted: bool


def _extract_status_code(error: BaseException) -> int | None:
    if isinstance(error, APIStatusError):
        return error.status_code
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code
    return None


def _is_excluded_error(error: BaseException) -> bool:
    if isinstance(
        error,
        (
            APIConnectionError,
            APITimeoutError,
            APIResponseValidationError,
            httpx.RequestError,
            httpx.StreamError,
            ConnectionError,
            TimeoutError,
            json.JSONDecodeError,
        ),
    ):
        return True
    status_code = _extract_status_code(error)
    if status_code in _EXCLUDED_STATUS_CODES:
        return True
    return not isinstance(
        error,
        (LengthFinishReasonError, APIStatusError, APIError, httpx.HTTPStatusError, ModelBehaviorError),
    )


def _decode_payload(error: BaseException) -> _ServerPayload:
    response_text = error.response.text if isinstance(error, (APIStatusError, httpx.HTTPStatusError)) else ""
    raw_body = error.body if isinstance(error, APIError) else None
    is_json = raw_body is not None
    if raw_body is None and response_text:
        try:
            raw_body = json.loads(response_text)
            is_json = True
        except json.JSONDecodeError:
            raw_body = None
    if isinstance(raw_body, dict):
        return _ServerPayload(dict_body=raw_body, response_text=response_text, is_json=is_json)
    if isinstance(raw_body, list):
        return _ServerPayload(list_body=raw_body, response_text=response_text, is_json=is_json)
    if isinstance(raw_body, str):
        return _ServerPayload(text_body=raw_body, response_text=response_text, is_json=is_json)
    return _ServerPayload(response_text=response_text, is_json=is_json)


def _has_explicit_overflow_code(error: BaseException, payload: _ServerPayload) -> bool:
    if isinstance(error, LengthFinishReasonError):
        return True
    if isinstance(error, APIError) and error.code == _OVERFLOW_ERROR_CODE:
        return True
    body = payload.dict_body
    if body is None:
        return False
    if body.get("code") == _OVERFLOW_ERROR_CODE:
        return True
    nested_error = body.get("error")
    return isinstance(nested_error, dict) and nested_error.get("code") == _OVERFLOW_ERROR_CODE


def _has_nonempty_error_body(error: APIError) -> bool:
    body = error.body
    if body is None:
        return False
    if isinstance(body, str):
        return bool(body.strip())
    if isinstance(body, (dict, list)):
        return bool(body)
    return True


def _should_defer_before_retry(error: BaseException, status_code: int | None) -> bool:
    if isinstance(error, ModelBehaviorError):
        return True
    if status_code is not None and status_code >= _INTERNAL_ERROR_STATUS:
        return True
    if isinstance(error, APIError) and not isinstance(error, APIStatusError):
        return not _has_nonempty_error_body(error)
    return False


def _extract_dict_error_message(body: dict, response_text: str) -> str:
    if not body:
        return ""
    error_field = body.get("error")
    candidates: list[str] = []
    if isinstance(error_field, dict) and isinstance(error_field.get("message"), str):
        candidates.append(error_field["message"])
    if isinstance(error_field, str):
        candidates.append(error_field)
    if isinstance(body.get("message"), str):
        candidates.append(body["message"])
    detail = body.get("detail")
    if isinstance(detail, str):
        candidates.append(detail)
    elif isinstance(detail, list):
        detail_parts = [
            entry["msg"].strip() if isinstance(entry, dict) else entry.strip()
            for entry in detail
            if (isinstance(entry, dict) and isinstance(entry.get("msg"), str) and entry["msg"].strip())
            or (isinstance(entry, str) and entry.strip())
        ]
        if detail_parts:
            candidates.append(" ".join(detail_parts))
    for text in candidates:
        if text.strip():
            return text
    return response_text if response_text.strip() else json.dumps(body, ensure_ascii=False)


def _extract_fallback_error_message(error: APIError | httpx.HTTPStatusError) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return ""
    if isinstance(error, APIStatusError):
        if error.body is not None and error.body != "":
            return ""
        prefix = f"Error code: {error.status_code} - "
        message = error.message.removeprefix(prefix).strip()
        if not message or message in (str(error.status_code), f"Error code: {error.status_code}"):
            return ""
        return message
    message = error.message.strip()
    if not message or message == _STREAM_ERROR_DEFAULT_MESSAGE:
        return ""
    return message


def _extract_payload_error_message(payload: _ServerPayload) -> str:
    if payload.text_body is not None:
        return payload.text_body if payload.text_body.strip() else ""
    if payload.dict_body is not None:
        return _extract_dict_error_message(payload.dict_body, payload.response_text)
    if payload.list_body is not None:
        if not payload.list_body:
            return ""
        return (
            payload.response_text
            if payload.response_text.strip()
            else json.dumps(payload.list_body, ensure_ascii=False)
        )
    if not payload.is_json and payload.response_text.strip():
        return payload.response_text
    return ""


def _extract_error_message(
    error: APIError | httpx.HTTPStatusError | ModelBehaviorError,
    payload: _ServerPayload,
) -> str:
    if isinstance(error, ModelBehaviorError):
        return str(error)
    error_message = _extract_payload_error_message(payload)
    if error_message:
        return error_message
    return _extract_fallback_error_message(error)


def _mask_error_numbers(text: str) -> str:
    return " ".join(_NUMBER_PATTERN.sub("<NUM>", text).split())


async def _classify_error_template(
    error_template: str,
    backend: config.Backend | None,
) -> bool:
    try:
        cached = await context_error_store.get(error_template)
    except (RuntimeError, sqlite3.Error) as error:
        print(f"[context.is_overflow] cache read failed: {error}", file=sys.stderr)
        raise ClassificationError(f"Overflow cache read failed: {error}") from error
    if cached is not None:
        return cached
    try:
        target = backend if backend is not None else config.backend(config.REASON_THINKING, config.REASON)
    except config.MissingSettingError as error:
        print(f"[context.is_overflow] classifier backend missing: {error}", file=sys.stderr)
        raise ClassificationError(f"Overflow classifier backend missing: {error}") from error
    raw_text, status = await reason(
        f"{_CLASSIFIER_PROMPT}{error_template}",
        thinking=False,
        json_schema=_OVERFLOW_SCHEMA,
        backend=target,
    )
    if status != 0:
        print(f"[context.is_overflow] classifier failed: {raw_text[:_LOG_PREVIEW_LENGTH]}", file=sys.stderr)
        raise ClassificationError(f"Overflow classifier failed: {raw_text}")
    parsed_result: checks.JsonObjectResult = checks.parse_json_object(raw_text)
    if not parsed_result.is_valid or parsed_result.value is None:
        print(f"[context.is_overflow] invalid JSON: {raw_text[:_LOG_PREVIEW_LENGTH]}", file=sys.stderr)
        raise ClassificationError(f"Overflow classifier returned invalid JSON: {parsed_result.error}")
    verdict = parsed_result.value.get("is_context_overflow")
    if not isinstance(verdict, bool):
        print(
            f"[context.is_overflow] non-boolean verdict: {repr(verdict)[:_LOG_PREVIEW_LENGTH]}",
            file=sys.stderr,
        )
        raise ClassificationError("Overflow classifier returned non-boolean is_context_overflow")
    timestamp_iso = datetime.now(timezone.utc).isoformat()
    try:
        await context_error_store.put(error_template, verdict, timestamp_iso)
    except (RuntimeError, sqlite3.Error) as error:
        print(f"[context.is_overflow] cache write failed: {error}", file=sys.stderr)
        raise ClassificationError(f"Overflow cache write failed: {error}") from error
    return verdict


async def is_overflow(
    exc: BaseException,
    *,
    backend: config.Backend | None = None,
    before_retry: bool = False,
) -> bool:
    """Check if an error means the input was too long for the model.

    Args:
        exc: The error raised during a model call.
        backend: Model server to ask when checking unknown error text.
        before_retry: True when retries are still left for this call.

    Returns:
        True if the error means the model context limit was reached.

    Raises:
        ClassificationError: If the cache or model check fails.
    """
    if _is_excluded_error(exc):
        return False
    payload = _decode_payload(exc)
    if _has_explicit_overflow_code(exc, payload):
        return True
    status_code = _extract_status_code(exc)
    if before_retry and _should_defer_before_retry(exc, status_code):
        return False
    assert isinstance(exc, (APIError, httpx.HTTPStatusError, ModelBehaviorError))
    error_message = _extract_error_message(exc, payload)
    if not error_message or not any(character.isalpha() for character in error_message):
        return status_code == _PAYLOAD_TOO_LARGE_STATUS
    error_template = _mask_error_numbers(error_message)
    return await _classify_error_template(error_template, backend)


def extract_reasoning_text(item: dict) -> str:
    """Read all thinking text from a reasoning item.

    Args:
        item: A reasoning item from the conversation.

    Returns:
        The summary and content text joined by newlines, or an empty string.
    """
    texts: list[str] = []
    for field in ("summary", "content"):
        entries = item.get(field)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                text = entry.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)
    return "\n".join(texts)


def resolve_video_url(file_data: str, filename: str | None = None) -> str | None:
    """Turn video file data into a data URL for the model.

    Args:
        file_data: Base64 video text or a data URL string.
        filename: File name used to guess the file type when needed.

    Returns:
        The video data URL string, or None if the input is empty or not video.
    """
    if not file_data:
        return None
    mime_type: str | None = None
    if file_data.startswith(_DATA_URL_PREFIX):
        header = file_data.split(";", 1)[0]
        if len(header) > len(_DATA_URL_PREFIX):
            mime_type = header[len(_DATA_URL_PREFIX) :]
    elif filename is not None:
        mime_type = mimetypes.guess_type(filename)[0]
    if not is_video_mime(mime_type):
        return None
    if file_data.startswith(_DATA_URL_PREFIX):
        return file_data
    return f"data:{mime_type};base64,{file_data}"


def _count_content_part_characters(
    part: dict,
    *,
    is_text_only: bool,
    is_tool_output: bool,
) -> int:
    part_type = part.get("type")
    if part_type in ("input_text", "text"):
        text = part.get("text")
        return len(text) if isinstance(text, str) else 0
    if is_text_only:
        return 0
    if part_type == "input_image":
        url = part.get("image_url")
        return len(url) if isinstance(url, str) else 0
    if part_type == "image_url":
        image_object = part.get("image_url")
        url = image_object.get("url") if isinstance(image_object, dict) else None
        return len(url) if isinstance(url, str) else 0
    if part_type == "video_url":
        video_object = part.get("video_url")
        url = video_object.get("url") if isinstance(video_object, dict) else None
        return len(url) if isinstance(url, str) else 0
    if part_type == "input_audio":
        audio_object = part.get("input_audio")
        audio_data = audio_object.get("data") if isinstance(audio_object, dict) else None
        return len(audio_data) if isinstance(audio_data, str) else 0
    if part_type == "input_file":
        file_data = part.get("file_data")
        if not isinstance(file_data, str) or not file_data:
            return 0
        if is_tool_output:
            filename = part.get("filename")
            url = resolve_video_url(file_data, filename if isinstance(filename, str) else None)
            return len(url) if url is not None else 0
        return len(file_data)
    if part_type == "file" and is_tool_output:
        file_object = part.get("file")
        file_data = file_object.get("file_data") if isinstance(file_object, dict) else None
        filename = file_object.get("filename") if isinstance(file_object, dict) else None
        if isinstance(file_data, str) and file_data:
            url = resolve_video_url(file_data, filename if isinstance(filename, str) else None)
            return len(url) if url is not None else 0
    return 0


def _count_response_output_message_characters(item: dict) -> int:
    content = item.get("content")
    if isinstance(content, str):
        return len(content)
    if not isinstance(content, list):
        return 0
    text_segments: list[str] = []
    last_refusal = ""
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "output_text" and isinstance(part.get("text"), str):
            text_segments.append(part["text"])
        elif part_type == "refusal" and isinstance(part.get("refusal"), str):
            last_refusal = part["refusal"]
    text_length = len("\n".join(text_segments)) if text_segments else 0
    return text_length + len(last_refusal)


def _count_item_characters(item: dict) -> int:
    item_type = item.get("type")
    role = item.get("role")
    if item_type == "message" and role == "assistant":
        return _count_response_output_message_characters(item)
    if item_type == "function_call":
        name = item.get("name")
        arguments = item.get("arguments")
        name_length = len(name) if isinstance(name, str) else 0
        arguments_text = arguments if isinstance(arguments, str) and arguments else _EMPTY_ARGUMENTS_FALLBACK
        return name_length + len(arguments_text)
    if item_type == "file_search_call":
        payload = json.dumps({"queries": item.get("queries", []), "status": item.get("status")})
        return len("file_search_call") + len(payload)
    if item_type == "function_call_output":
        output = item.get("output")
        if isinstance(output, str):
            return len(output)
        if isinstance(output, list):
            return sum(
                _count_content_part_characters(part, is_text_only=False, is_tool_output=True)
                for part in output
                if isinstance(part, dict)
            )
        return 0
    if role in ("user", "assistant", "system", "developer"):
        content = item.get("content")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            is_text_only = role in _TEXT_ONLY_ROLES
            return sum(
                _count_content_part_characters(part, is_text_only=is_text_only, is_tool_output=False)
                for part in content
                if isinstance(part, dict)
            )
    return 0


def count_chars(
    items: str | list[dict],
    *,
    system_prompt: str = "",
) -> int:
    """Count the text characters that message items send to the model.

    Args:
        items: User prompt text or a list of conversation items.
        system_prompt: Instructions sent ahead of the conversation.

    Returns:
        Total character count of the prompt and all sent item text.
    """
    if isinstance(items, str):
        return len(system_prompt) + len(items)
    total = len(system_prompt)
    reasoning_run: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            text = extract_reasoning_text(item)
            if text:
                reasoning_run.append(text)
            continue
        if reasoning_run:
            total += len("\n".join(reasoning_run))
            reasoning_run = []
        total += _count_item_characters(item)
    if reasoning_run:
        total += len("\n".join(reasoning_run))
    return total


def _is_assistant_message(item: dict) -> bool:
    if item.get("role") != "assistant":
        return False
    item_type = item.get("type")
    if item_type == "message":
        return True
    return item_type is None and "content" in item and not item.get("tool_calls")


async def _measure_candidate(
    candidate: list[dict],
    system_prompt: str,
    measure: Callable[[list[dict]], int | Awaitable[int]] | None,
) -> int:
    if measure is None:
        return count_chars(candidate, system_prompt=system_prompt)
    size = measure(list(candidate))
    if inspect.isawaitable(size):
        return await size
    return size


async def _compact_default_items(
    message_items: list[dict],
    baseline_characters: int,
    system_prompt: str,
    measure: Callable[[list[dict]], int | Awaitable[int]] | None,
    is_verbose: bool,
) -> _DefaultCompactionResult:
    reasoning_indices = [
        index
        for index, item in enumerate(message_items)
        if isinstance(item, dict) and item.get("type") == "reasoning"
    ]
    assistant_message_indices = [
        index
        for index, item in enumerate(message_items)
        if isinstance(item, dict) and _is_assistant_message(item)
    ]
    removed_indices: set[int] = set()
    candidate = message_items
    remaining_characters = baseline_characters
    for index in reasoning_indices + assistant_message_indices:
        removed_indices.add(index)
        candidate = [
            item for item_index, item in enumerate(message_items) if item_index not in removed_indices
        ]
        if not candidate:
            break
        remaining_characters = await _measure_candidate(candidate, system_prompt, measure)
        if (baseline_characters - remaining_characters) * _HALF_REDUCTION_FACTOR >= baseline_characters:
            if is_verbose:
                print(
                    f"[context.compact] global accepted, chars {baseline_characters} to {remaining_characters}",
                    flush=True,
                )
            return _DefaultCompactionResult(
                items=candidate,
                remaining_characters=remaining_characters,
                is_accepted=True,
            )
    return _DefaultCompactionResult(
        items=candidate,
        remaining_characters=remaining_characters,
        is_accepted=False,
    )


async def compact(
    items: str | list[dict],
    *,
    system_prompt: str = "",
    measure: Callable[[list[dict]], int | Awaitable[int]] | None = None,
    compact_hook: CompactHook | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Trim message history to at most half its starting size.

    Args:
        items: User prompt text or a list of conversation items.
        system_prompt: Instructions sent ahead of the conversation.
        measure: Optional size check called on each candidate item list.
            Must leave input unchanged and give the same size for the same input.
        compact_hook: Optional callback called with a copy of remaining items
            when default shrinking is not enough. Returns a shorter list or
            None while keeping instructions, unknown items, and tool call pairs.
        verbose: Whether to print size counts during recovery.

    Returns:
        A new non-empty item list at most half the starting character size.

    Raises:
        CompactionError: If the items are empty or cannot shrink by half.
    """
    if isinstance(items, str):
        message_items: list[dict] = [{"role": "user", "content": items}]
    else:
        message_items = [dict(item) if isinstance(item, dict) else item for item in items]
    if not message_items:
        raise CompactionError("Cannot compact empty message history.")
    baseline_characters = await _measure_candidate(message_items, system_prompt, measure)
    if baseline_characters <= 0:
        raise CompactionError("Cannot compact message history with zero baseline size.")
    default_result = await _compact_default_items(
        message_items,
        baseline_characters,
        system_prompt,
        measure,
        verbose,
    )
    if default_result.is_accepted:
        return default_result.items
    if compact_hook is not None:
        if verbose:
            remaining = default_result.remaining_characters if default_result.items else 0
            print(
                f"[context.compact] global exhausted, chars {baseline_characters} to {remaining}",
                flush=True,
            )
        return await _compact_with_hook(
            default_result.items,
            compact_hook,
            baseline_characters,
            system_prompt,
            measure,
            verbose,
        )
    raise CompactionError("Message history could not shrink by at least half.")


async def _compact_with_hook(
    items: list[dict],
    hook: CompactHook,
    baseline_characters: int,
    system_prompt: str,
    measure: Callable[[list[dict]], int | Awaitable[int]] | None,
    is_verbose: bool,
) -> list[dict]:
    snapshot = await asyncio.to_thread(deepcopy, items)
    try:
        candidate = hook(snapshot)
        if inspect.isawaitable(candidate):
            candidate = await candidate
    except Exception as error:
        raise CompactionError(f"Compaction hook failed with {type(error)}.") from error
    if candidate is None:
        raise CompactionError("Compaction hook declined to trim history.")
    if not isinstance(candidate, list) or any(not isinstance(item, dict) for item in candidate):
        raise CompactionError("Compaction hook must return a list of message dictionaries or None.")
    if not items or not candidate:
        raise CompactionError("Compaction hook cannot replace empty message history.")
    remaining_characters = await _measure_candidate(candidate, system_prompt, measure)
    if is_verbose:
        print(
            f"[context.compact] hook measured, chars {baseline_characters} to {remaining_characters}",
            flush=True,
        )
    if remaining_characters <= 0 or remaining_characters * _HALF_REDUCTION_FACTOR > baseline_characters:
        raise CompactionError("Compaction hook could not shrink message history by at least half.")
    if is_verbose:
        print("[context.compact] hook accepted", flush=True)
    return candidate
