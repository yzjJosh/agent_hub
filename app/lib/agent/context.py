"""Detect context overflow errors and shrink message history."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import inspect
import json
import mimetypes
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
_FORMAL_OVERFLOW_CODE = "context_length_exceeded"
_STREAM_ERROR_DEFAULT_MSG = "An error occurred during streaming"
_HALF_REDUCTION_FACTOR = 2
_LOG_PREVIEW_LEN = 100
_DATA_URL_PREFIX = "data:"
_EMPTY_ARGS_FALLBACK = "{}"
_TEXT_ONLY_ROLES = frozenset({"system", "developer", "assistant"})

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


def _status_code_of(exc: BaseException) -> int | None:
    if isinstance(exc, APIStatusError):
        return exc.status_code
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None


def _is_excluded_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
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
    status_code = _status_code_of(exc)
    if status_code in _EXCLUDED_STATUS_CODES:
        return True
    return not isinstance(
        exc,
        (LengthFinishReasonError, APIStatusError, APIError, httpx.HTTPStatusError, ModelBehaviorError),
    )


@dataclass(frozen=True)
class _ServerPayload:
    dict_body: dict | None = None
    list_body: list | None = None
    str_body: str | None = None
    response_text: str = ""
    is_json: bool = False


def _decode_payload(exc: BaseException) -> _ServerPayload:
    response_text = exc.response.text if isinstance(exc, (APIStatusError, httpx.HTTPStatusError)) else ""
    raw_body = exc.body if isinstance(exc, APIError) else None
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
        return _ServerPayload(str_body=raw_body, response_text=response_text, is_json=is_json)
    return _ServerPayload(response_text=response_text, is_json=is_json)


def _is_formal_overflow(exc: BaseException, payload: _ServerPayload) -> bool:
    if isinstance(exc, LengthFinishReasonError):
        return True
    if isinstance(exc, APIError) and exc.code == _FORMAL_OVERFLOW_CODE:
        return True
    body = payload.dict_body
    if body is None:
        return False
    if body.get("code") == _FORMAL_OVERFLOW_CODE:
        return True
    nested = body.get("error")
    return isinstance(nested, dict) and nested.get("code") == _FORMAL_OVERFLOW_CODE


def _has_nonempty_error_body(exc: APIError) -> bool:
    body = exc.body
    if body is None:
        return False
    if isinstance(body, str):
        return bool(body.strip())
    if isinstance(body, (dict, list)):
        return bool(body)
    return True


def _should_defer_before_retry(exc: BaseException, status_code: int | None) -> bool:
    if isinstance(exc, ModelBehaviorError):
        return True
    if status_code is not None and status_code >= _INTERNAL_ERROR_STATUS:
        return True
    if isinstance(exc, APIError) and not isinstance(exc, APIStatusError):
        return not _has_nonempty_error_body(exc)
    return False


def _dict_prose(body: dict, response_text: str) -> str:
    if not body:
        return ""
    err = body.get("error")
    candidates: list[str] = []
    if isinstance(err, dict) and isinstance(err.get("message"), str):
        candidates.append(err["message"])
    if isinstance(err, str):
        candidates.append(err)
    if isinstance(body.get("message"), str):
        candidates.append(body["message"])
    detail = body.get("detail")
    if isinstance(detail, str):
        candidates.append(detail)
    elif isinstance(detail, list):
        parts = [
            item["msg"].strip() if isinstance(item, dict) else item.strip()
            for item in detail
            if (isinstance(item, dict) and isinstance(item.get("msg"), str) and item["msg"].strip())
            or (isinstance(item, str) and item.strip())
        ]
        if parts:
            candidates.append(" ".join(parts))
    for text in candidates:
        if text.strip():
            return text
    return response_text if response_text.strip() else json.dumps(body, ensure_ascii=False)


def _fallback_message_of(exc: APIError | httpx.HTTPStatusError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return ""
    if isinstance(exc, APIStatusError):
        if exc.body is not None and exc.body != "":
            return ""
        prefix = f"Error code: {exc.status_code} - "
        msg = exc.message.removeprefix(prefix).strip()
        if not msg or msg in (str(exc.status_code), f"Error code: {exc.status_code}"):
            return ""
        return msg
    msg = exc.message.strip()
    if not msg or msg == _STREAM_ERROR_DEFAULT_MSG:
        return ""
    return msg


def _payload_prose(payload: _ServerPayload) -> str:
    if payload.str_body is not None:
        return payload.str_body if payload.str_body.strip() else ""
    if payload.dict_body is not None:
        return _dict_prose(payload.dict_body, payload.response_text)
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


def _extract_prose(
    exc: APIError | httpx.HTTPStatusError | ModelBehaviorError,
    payload: _ServerPayload,
) -> str:
    if isinstance(exc, ModelBehaviorError):
        return str(exc)
    prose = _payload_prose(payload)
    if prose:
        return prose
    return _fallback_message_of(exc)


def _is_ascii_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def _normalize_error_text(text: str) -> str:
    out: list[str] = []
    idx = 0
    length = len(text)
    while idx < length:
        if _is_ascii_digit(text[idx]):
            out.append("<NUM>")
            idx += 1
            while idx < length:
                if _is_ascii_digit(text[idx]):
                    idx += 1
                elif text[idx] in (".", ",") and idx + 1 < length and _is_ascii_digit(text[idx + 1]):
                    idx += 2
                else:
                    break
        else:
            out.append(text[idx])
            idx += 1
    return " ".join("".join(out).split())


async def _classify_normalized(normalized: str, backend: config.Backend | None) -> bool:
    try:
        cached = await context_error_store.get(normalized)
    except (RuntimeError, sqlite3.Error) as exc:
        print(f"[context.is_overflow] cache read failed: {exc}", file=sys.stderr)
        raise ClassificationError(f"Overflow cache read failed: {exc}") from exc
    if cached is not None:
        return cached
    try:
        target = backend if backend is not None else config.backend(config.REASON_THINKING, config.REASON)
    except config.MissingSettingError as exc:
        print(f"[context.is_overflow] classifier backend missing: {exc}", file=sys.stderr)
        raise ClassificationError(f"Overflow classifier backend missing: {exc}") from exc
    raw_text, status = await reason(
        f"{_CLASSIFIER_PROMPT}{normalized}",
        thinking=False,
        json_schema=_OVERFLOW_SCHEMA,
        backend=target,
    )
    if status != 0:
        print(f"[context.is_overflow] classifier failed: {raw_text[:_LOG_PREVIEW_LEN]}", file=sys.stderr)
        raise ClassificationError(f"Overflow classifier failed: {raw_text}")
    ok, parsed = checks.parse_json_object(raw_text)
    if not ok or not isinstance(parsed, dict):
        print(f"[context.is_overflow] invalid JSON: {raw_text[:_LOG_PREVIEW_LEN]}", file=sys.stderr)
        raise ClassificationError(f"Overflow classifier returned invalid JSON: {parsed}")
    verdict = parsed.get("is_context_overflow")
    if not isinstance(verdict, bool):
        print(
            f"[context.is_overflow] non-boolean verdict: {repr(verdict)[:_LOG_PREVIEW_LEN]}",
            file=sys.stderr,
        )
        raise ClassificationError("Overflow classifier returned non-boolean is_context_overflow")
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        await context_error_store.put(normalized, verdict, now_iso)
    except (RuntimeError, sqlite3.Error) as exc:
        print(f"[context.is_overflow] cache write failed: {exc}", file=sys.stderr)
        raise ClassificationError(f"Overflow cache write failed: {exc}") from exc
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
    if _is_formal_overflow(exc, payload):
        return True
    status_code = _status_code_of(exc)
    if before_retry and _should_defer_before_retry(exc, status_code):
        return False
    assert isinstance(exc, (APIError, httpx.HTTPStatusError, ModelBehaviorError))
    prose = _extract_prose(exc, payload)
    if not prose or not any(ch.isalpha() for ch in prose):
        return status_code == _PAYLOAD_TOO_LARGE_STATUS
    normalized = _normalize_error_text(prose)
    return await _classify_normalized(normalized, backend)


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
    mime: str | None = None
    if file_data.startswith(_DATA_URL_PREFIX):
        head = file_data.split(";", 1)[0]
        if len(head) > len(_DATA_URL_PREFIX):
            mime = head[len(_DATA_URL_PREFIX) :]
    elif filename is not None:
        mime = mimetypes.guess_type(filename)[0]
    if not is_video_mime(mime):
        return None
    if file_data.startswith(_DATA_URL_PREFIX):
        return file_data
    return f"data:{mime};base64,{file_data}"


def _content_part_chars(part: dict, *, text_only: bool, tool_output: bool) -> int:
    ptype = part.get("type")
    if ptype in ("input_text", "text"):
        text = part.get("text")
        return len(text) if isinstance(text, str) else 0
    if text_only:
        return 0
    if ptype == "input_image":
        url = part.get("image_url")
        return len(url) if isinstance(url, str) else 0
    if ptype == "image_url":
        img = part.get("image_url")
        url = img.get("url") if isinstance(img, dict) else None
        return len(url) if isinstance(url, str) else 0
    if ptype == "video_url":
        vid = part.get("video_url")
        url = vid.get("url") if isinstance(vid, dict) else None
        return len(url) if isinstance(url, str) else 0
    if ptype == "input_audio":
        audio = part.get("input_audio")
        data = audio.get("data") if isinstance(audio, dict) else None
        return len(data) if isinstance(data, str) else 0
    if ptype == "input_file":
        file_data = part.get("file_data")
        if not isinstance(file_data, str) or not file_data:
            return 0
        if tool_output:
            filename = part.get("filename")
            url = resolve_video_url(file_data, filename if isinstance(filename, str) else None)
            return len(url) if url is not None else 0
        return len(file_data)
    if ptype == "file" and tool_output:
        file_obj = part.get("file")
        file_data = file_obj.get("file_data") if isinstance(file_obj, dict) else None
        filename = file_obj.get("filename") if isinstance(file_obj, dict) else None
        if isinstance(file_data, str) and file_data:
            url = resolve_video_url(file_data, filename if isinstance(filename, str) else None)
            return len(url) if url is not None else 0
    return 0


def _response_output_message_chars(item: dict) -> int:
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
        ptype = part.get("type")
        if ptype == "output_text" and isinstance(part.get("text"), str):
            text_segments.append(part["text"])
        elif ptype == "refusal" and isinstance(part.get("refusal"), str):
            last_refusal = part["refusal"]
    text_len = len("\n".join(text_segments)) if text_segments else 0
    return text_len + len(last_refusal)


def _item_chars(item: dict) -> int:
    item_type = item.get("type")
    role = item.get("role")
    if item_type == "message" and role == "assistant":
        return _response_output_message_chars(item)
    if item_type == "function_call":
        name = item.get("name")
        args = item.get("arguments")
        name_len = len(name) if isinstance(name, str) else 0
        args_str = args if isinstance(args, str) and args else _EMPTY_ARGS_FALLBACK
        return name_len + len(args_str)
    if item_type == "file_search_call":
        payload = json.dumps({"queries": item.get("queries", []), "status": item.get("status")})
        return len("file_search_call") + len(payload)
    if item_type == "function_call_output":
        output = item.get("output")
        if isinstance(output, str):
            return len(output)
        if isinstance(output, list):
            return sum(
                _content_part_chars(p, text_only=False, tool_output=True)
                for p in output
                if isinstance(p, dict)
            )
        return 0
    if role in ("user", "assistant", "system", "developer"):
        content = item.get("content")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            text_only = role in _TEXT_ONLY_ROLES
            return sum(
                _content_part_chars(p, text_only=text_only, tool_output=False)
                for p in content
                if isinstance(p, dict)
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
        total += _item_chars(item)
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
        compact_hook: Optional callback to shorten remaining items when default
            shrinking is not enough. Gets a copy of the remaining items and
            returns a shorter list or None. Must keep instructions, kept call
            ids and order, full call and output pairs, unknown items, and
            unfinished calls, without adding calls or bringing back dropped
            items. Must be safe across runs at the same time, must not save and
            later change its input or output, and must be async if slow.
        verbose: Whether to print size counts during recovery.

    Returns:
        A new non-empty item list at most half the starting character size.

    Raises:
        CompactionError: If the items are empty or cannot shrink by half.
    """
    if isinstance(items, str):
        normalized: list[dict] = [{"role": "user", "content": items}]
    else:
        normalized = [dict(item) if isinstance(item, dict) else item for item in items]
    if not normalized:
        raise CompactionError("Cannot compact empty message history.")
    baseline_chars = await _measure_candidate(normalized, system_prompt, measure)
    if baseline_chars <= 0:
        raise CompactionError("Cannot compact message history with zero baseline size.")
    tier_one = [
        i for i, item in enumerate(normalized) if isinstance(item, dict) and item.get("type") == "reasoning"
    ]
    tier_two = [i for i, item in enumerate(normalized) if isinstance(item, dict) and _is_assistant_message(item)]
    removed: set[int] = set()
    candidate = normalized
    remaining_chars = baseline_chars
    for index in tier_one + tier_two:
        removed.add(index)
        candidate = [item for i, item in enumerate(normalized) if i not in removed]
        if not candidate:
            break
        remaining_chars = await _measure_candidate(candidate, system_prompt, measure)
        if (baseline_chars - remaining_chars) * _HALF_REDUCTION_FACTOR >= baseline_chars:
            if verbose:
                print(f"[context.compact] global accepted, chars {baseline_chars} to {remaining_chars}", flush=True)
            return candidate
    if compact_hook is not None:
        if verbose:
            print(f"[context.compact] global exhausted, chars {baseline_chars} to {remaining_chars if candidate else 0}", flush=True)
        return await _compact_with_hook(candidate, compact_hook, baseline_chars, system_prompt, measure, verbose)
    raise CompactionError("Message history could not shrink by at least half.")


async def _compact_with_hook(
    items: list[dict],
    hook: CompactHook,
    baseline_chars: int,
    system_prompt: str,
    measure: Callable[[list[dict]], int | Awaitable[int]] | None,
    verbose: bool,
) -> list[dict]:
    snapshot = await asyncio.to_thread(deepcopy, items)
    try:
        candidate = hook(snapshot)
        if inspect.isawaitable(candidate):
            candidate = await candidate
    except Exception as exc:
        raise CompactionError(f"Compaction hook failed with {type(exc)}.") from exc
    if candidate is None:
        raise CompactionError("Compaction hook declined to trim history.")
    if not isinstance(candidate, list) or any(not isinstance(item, dict) for item in candidate):
        raise CompactionError("Compaction hook must return a list of message dictionaries or None.")
    if not items or not candidate:
        raise CompactionError("Compaction hook cannot replace empty message history.")
    remaining_chars = await _measure_candidate(candidate, system_prompt, measure)
    if verbose:
        print(f"[context.compact] hook measured, chars {baseline_chars} to {remaining_chars}", flush=True)
    if remaining_chars <= 0 or remaining_chars * _HALF_REDUCTION_FACTOR > baseline_chars:
        raise CompactionError("Compaction hook could not shrink message history by at least half.")
    if verbose:
        print("[context.compact] hook accepted", flush=True)
    return candidate
