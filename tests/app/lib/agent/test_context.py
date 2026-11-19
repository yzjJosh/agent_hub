"""Tests for context overflow classification, character counting, and history compaction."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import json
import httpx
from agents import ModelBehaviorError
from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ContentFilterFinishReasonError,
    InternalServerError,
    LengthFinishReasonError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
import pytest

import app.lib.db.store as store_module
from app.lib.agent.context import (
    ClassificationError,
    CompactionError,
    compact,
    count_chars,
    extract_reasoning_text,
    is_overflow,
    resolve_video_url,
)
from app.lib.shared import config
from tests.app.conftest import is_llm_available

_TEST_URL = "http://classifier.test/v1/chat/completions"
_BACKEND = config.Backend(url="http://classifier.test/v1", model="clf-model", api_key="sk-test")
_EXPIRED_DAYS = 31
_LIVE_TIMEOUT_S = 120


def _req() -> httpx.Request:
    return httpx.Request("POST", _TEST_URL)


def _resp(
    status_code: int,
    *,
    json_body: dict | None = None,
    text_body: str | None = None,
) -> httpx.Response:
    if json_body is not None:
        return httpx.Response(status_code, json=json_body, request=_req())
    if text_body is not None:
        return httpx.Response(status_code, text=text_body, request=_req())
    return httpx.Response(status_code, request=_req())


def _length_finish_error() -> LengthFinishReasonError:
    completion = ChatCompletion(
        id="chatcmpl-1",
        choices=[
            Choice(
                finish_reason="length",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="cut"),
            )
        ],
        created=1,
        model="default",
        object="chat.completion",
    )
    return LengthFinishReasonError(completion=completion)


def _forbid_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _unexpected(*_args: str, **_kwargs: dict) -> httpx.Response:
        raise AssertionError("Classifier HTTP post should not be called")

    monkeypatch.setattr(httpx.AsyncClient, "post", _unexpected)


def _mock_classifier(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[httpx.Response],
) -> list[tuple[str, dict]]:
    captured: list[tuple[str, dict]] = []
    queue = list(responses)

    async def _post(
        _self: httpx.AsyncClient,
        url: str,
        *,
        json: dict,
        headers: dict[str, str],
    ) -> httpx.Response:
        _ = headers
        captured.append((url, json))
        return queue.pop(0)

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    return captured


def _classifier_json_response(content: str, status_code: int = 200) -> httpx.Response:
    if status_code != 200:
        return httpx.Response(status_code, text=content, request=_req())
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": content}}]},
        request=_req(),
    )


_EXCLUDED_FACTORIES = [
    ("api_conn", lambda: APIConnectionError(message="context length exceeded", request=_req())),
    ("api_timeout", lambda: APITimeoutError(request=_req())),
    (
        "api_validation",
        lambda: APIResponseValidationError(
            response=_resp(200),
            body={"code": "context_length_exceeded"},
            message="context length exceeded",
        ),
    ),
    ("httpx_connect", lambda: httpx.ConnectError("context length exceeded", request=_req())),
    ("httpx_timeout", lambda: httpx.ReadTimeout("context length exceeded", request=_req())),
    ("httpx_protocol", lambda: httpx.RemoteProtocolError("closed", request=_req())),
    ("httpx_stream", httpx.StreamClosed),
    ("conn_reset", lambda: ConnectionResetError("reset")),
    ("timeout_err", lambda: TimeoutError("timed out")),
    ("json_decode", lambda: json.JSONDecodeError("bad", "{", 1)),
    ("auth_401", lambda: AuthenticationError("context overflow", response=_resp(401), body=None)),
    ("perm_403", lambda: PermissionDeniedError("context overflow", response=_resp(403), body=None)),
    ("not_found_404", lambda: NotFoundError("context overflow", response=_resp(404), body=None)),
    ("timeout_408", lambda: APIStatusError("context overflow", response=_resp(408), body=None)),
    ("conflict_409", lambda: ConflictError("context overflow", response=_resp(409), body=None)),
    (
        "rate_limit_429_with_formal_code",
        lambda: RateLimitError(
            "429",
            response=_resp(429, json_body={"error": {"code": "context_length_exceeded"}}),
            body={"error": {"code": "context_length_exceeded", "message": "too many tokens"}},
        ),
    ),
    (
        "bad_gateway_502_with_formal_code",
        lambda: APIStatusError(
            "502",
            response=_resp(502, json_body={"code": "context_length_exceeded"}),
            body={"code": "context_length_exceeded"},
        ),
    ),
    (
        "unavailable_503",
        lambda: InternalServerError("context limit", response=_resp(503), body=None),
    ),
    ("gateway_timeout_504", lambda: InternalServerError("timeout", response=_resp(504), body=None)),
    (
        "httpx_429_with_formal_code",
        lambda: httpx.HTTPStatusError(
            "429",
            request=_req(),
            response=_resp(429, json_body={"code": "context_length_exceeded"}),
        ),
    ),
    (
        "httpx_503",
        lambda: httpx.HTTPStatusError(
            "503",
            request=_req(),
            response=_resp(503, text_body="maximum context length exceeded"),
        ),
    ),
    ("content_filter", ContentFilterFinishReasonError),
    ("value_error", lambda: ValueError("maximum context length exceeded")),
    ("runtime_error", lambda: RuntimeError("context_length_exceeded")),
]


class TestIsOverflowExclusionsAndFormalContracts:
    """Transport, excluded status codes, and formal wire codes resolve without AI calls."""

    @pytest.mark.parametrize("before_retry", [False, True])
    @pytest.mark.parametrize(
        "make_exc",
        [fn for _, fn in _EXCLUDED_FACTORIES],
        ids=[name for name, _ in _EXCLUDED_FACTORIES],
    )
    async def test_excluded_errors_return_false_without_io(
        self,
        monkeypatch: pytest.MonkeyPatch,
        before_retry: bool,
        make_exc: Callable[[], BaseException],
    ) -> None:
        """Excluded errors return False even if the store is closed or prose looks like overflow."""
        _forbid_classifier(monkeypatch)
        assert await is_overflow(make_exc(), backend=_BACKEND, before_retry=before_retry) is False

    @pytest.mark.parametrize("before_retry", [False, True])
    @pytest.mark.parametrize(
        "exc",
        [
            _length_finish_error(),
            BadRequestError(
                "bad",
                response=_resp(400),
                body={"code": "context_length_exceeded", "message": "ignored"},
            ),
            BadRequestError(
                "bad",
                response=_resp(400),
                body={"error": {"code": "context_length_exceeded", "message": "ignored"}},
            ),
            InternalServerError(
                "500",
                response=_resp(500),
                body={"error": {"code": "context_length_exceeded"}},
            ),
            BadRequestError(
                "bad",
                response=_resp(400, json_body={"error": {"code": "context_length_exceeded"}}),
                body=None,
            ),
            APIError(
                "stream",
                request=_req(),
                body={"error": {"code": "context_length_exceeded"}},
            ),
            httpx.HTTPStatusError(
                "400",
                request=_req(),
                response=_resp(400, json_body={"code": "context_length_exceeded"}),
            ),
            httpx.HTTPStatusError(
                "422",
                request=_req(),
                response=_resp(422, json_body={"error": {"code": "context_length_exceeded"}}),
            ),
        ],
        ids=[
            "length_finish_reason",
            "top_level_code_400",
            "nested_error_code_400",
            "nested_error_code_500",
            "json_response_fallback_code",
            "mid_stream_api_error_code",
            "httpx_top_level_code",
            "httpx_nested_code",
        ],
    )
    async def test_formal_overflow_contracts_return_true_without_io(
        self,
        monkeypatch: pytest.MonkeyPatch,
        before_retry: bool,
        exc: BaseException,
    ) -> None:
        """Formal wire overflow codes return True immediately regardless of retry stage."""
        _forbid_classifier(monkeypatch)
        assert await is_overflow(exc, backend=_BACKEND, before_retry=before_retry) is True


class TestIsOverflowDeferralAndExtraction:
    """Tests for waiting until retries finish, reading error text, and empty errors."""

    @pytest.mark.parametrize(
        "exc",
        [
            ModelBehaviorError("maximum context length 8192 exceeded"),
            InternalServerError(
                "500",
                response=_resp(500),
                body={"error": {"message": "prompt length 9000 exceeds 8192"}},
            ),
            httpx.HTTPStatusError(
                "500",
                request=_req(),
                response=_resp(500, json_body={"message": "prompt length 9000 exceeds 8192"}),
            ),
            APIError("maximum context length 8192 exceeded", request=_req(), body=None),
            APIError("maximum context length 8192 exceeded", request=_req(), body=""),
            APIError("maximum context length 8192 exceeded", request=_req(), body="   "),
            APIError("maximum context length 8192 exceeded", request=_req(), body={}),
            APIError("maximum context length 8192 exceeded", request=_req(), body=[]),
        ],
        ids=[
            "model_behavior",
            "openai_500",
            "httpx_500",
            "api_error_none_body",
            "api_error_empty_str_body",
            "api_error_whitespace_body",
            "api_error_empty_dict_body",
            "api_error_empty_list_body",
        ],
    )
    async def test_before_retry_defers_ambiguous_errors_without_classifier_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        exc: BaseException,
    ) -> None:
        """Ambiguous server, model-behavior, and bodyless stream errors defer before retry."""
        _forbid_classifier(monkeypatch)
        assert await is_overflow(exc, backend=_BACKEND, before_retry=True) is False
        captured = _mock_classifier(
            monkeypatch,
            [_classifier_json_response('{"is_context_overflow": true}')],
        )
        assert await is_overflow(exc, backend=_BACKEND, before_retry=False) is True
        assert len(captured) == 1

    @pytest.mark.parametrize(
        "body",
        [
            {"error": {"message": "SSE prompt 12000 tokens exceeds 8192"}},
            "SSE prompt 12000 tokens exceeds 8192",
            ["SSE prompt 12000 tokens exceeds 8192"],
        ],
        ids=["dict_body", "str_body", "list_body"],
    )
    async def test_nonempty_mid_stream_api_error_classifies_before_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
        body: dict | list | str,
    ) -> None:
        """Mid-stream APIError carrying a non-empty server body classifies immediately."""
        captured = _mock_classifier(
            monkeypatch,
            [_classifier_json_response('{"is_context_overflow": true}')],
        )
        exc = APIError("An error occurred during streaming", request=_req(), body=body)
        assert await is_overflow(exc, backend=_BACKEND, before_retry=True) is True
        assert len(captured) == 1

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (APIStatusError("413", response=_resp(413), body=None), True),
            (APIStatusError("Error code: 413 - {}", response=_resp(413), body={}), True),
            (APIStatusError("413", response=_resp(413, text_body="null"), body=None), True),
            (httpx.HTTPStatusError("413", request=_req(), response=_resp(413)), True),
            (httpx.HTTPStatusError("413", request=_req(), response=_resp(413, text_body="12345")), True),
            (httpx.HTTPStatusError("413", request=_req(), response=_resp(413, text_body="null")), True),
            (BadRequestError("400", response=_resp(400), body=None), False),
            (BadRequestError("Error code: 400", response=_resp(400), body=None), False),
            (BadRequestError("Error code: 400 - {}", response=_resp(400), body={}), False),
            (BadRequestError("Error code: 400 - {}", response=_resp(400, json_body={}), body={}), False),
            (BadRequestError("400", response=_resp(400, text_body="null"), body=None), False),
            (BadRequestError("400", response=_resp(400, text_body="true"), body=None), False),
            (BadRequestError("Error code: 400 - words", response=_resp(400), body={}), False),
            (BadRequestError("Error code: 400 - words", response=_resp(400), body=[]), False),
            (UnprocessableEntityError("422", response=_resp(422), body=None), False),
            (httpx.HTTPStatusError("Client error 400", request=_req(), response=_resp(400)), False),
            (httpx.HTTPStatusError("400", request=_req(), response=_resp(400, text_body="null")), False),
            (APIError("An error occurred during streaming", request=_req(), body=None), False),
        ],
        ids=[
            "openai_413_empty",
            "openai_413_empty_dict",
            "openai_413_json_null",
            "httpx_413_empty",
            "httpx_413_digits_only",
            "httpx_413_json_null",
            "openai_400_bare_code",
            "openai_400_error_code_only",
            "openai_400_empty_dict",
            "openai_400_empty_dict_json_response",
            "openai_400_json_null",
            "openai_400_json_bool",
            "openai_400_nonempty_msg_guarded_by_empty_dict_body",
            "openai_400_nonempty_msg_guarded_by_empty_list_body",
            "openai_422_empty",
            "httpx_400_ignores_str_exc",
            "httpx_400_json_null",
            "api_error_default_stream_msg",
        ],
    )
    async def test_messageless_deterministic_rules(
        self,
        monkeypatch: pytest.MonkeyPatch,
        exc: BaseException,
        expected: bool,
    ) -> None:
        """Message-less 413 returns True while all other message-less errors return False."""
        _forbid_classifier(monkeypatch)
        assert await is_overflow(exc, backend=_BACKEND, before_retry=False) is expected

    @pytest.mark.parametrize(
        ("exc", "expected_normalized"),
        [
            (
                BadRequestError(
                    "400",
                    response=_resp(400),
                    body={"error": {"message": "  Max   context\n\t8,192.50  exceeded by 12,345. "}},
                ),
                "Max context <NUM> exceeded by <NUM>.",
            ),
            (
                BadRequestError(
                    "400",
                    response=_resp(400),
                    body={"error": "Prompt has 500 tokens, limit is 256"},
                ),
                "Prompt has <NUM> tokens, limit is <NUM>",
            ),
            (
                UnprocessableEntityError(
                    "422",
                    response=_resp(422),
                    body={
                        "detail": [
                            {"loc": ["body", "messages"], "msg": "Input length 9000 exceeds 4096"},
                            {"loc": ["body"], "msg": "v1.2.3 validator"},
                        ]
                    },
                ),
                "Input length <NUM> exceeds <NUM> v<NUM> validator",
            ),
            (
                BadRequestError(
                    "Error code: 400 - Prompt tokens 10000 > max 8192",
                    response=_resp(400),
                    body="",
                ),
                "Prompt tokens <NUM> > max <NUM>",
            ),
            (
                httpx.HTTPStatusError(
                    "400",
                    request=_req(),
                    response=_resp(400, json_body={"custom_reason": "overflow at 4096"}),
                ),
                '{"custom_reason":"overflow at <NUM>"}',
            ),
            (
                APIError(
                    "stream",
                    request=_req(),
                    body={"custom_reason": "overflow at 4096"},
                ),
                '{"custom_reason": "overflow at <NUM>"}',
            ),
            (
                APIStatusError(
                    "413",
                    response=_resp(413, text_body="Audio upload exceeds 25 MB limit"),
                    body=None,
                ),
                "Audio upload exceeds <NUM> MB limit",
            ),
        ],
        ids=[
            "nested_error_message_with_commas_and_decimals",
            "string_error_field",
            "fastapi_422_detail_list",
            "status_error_message_prefix_stripped",
            "custom_nonempty_dict_response_text_fallback",
            "custom_nonempty_dict_json_dumps_fallback",
            "status_413_with_prose_uses_classifier",
        ],
    )
    async def test_prose_extraction_and_character_normalization(
        self,
        monkeypatch: pytest.MonkeyPatch,
        exc: BaseException,
        expected_normalized: str,
    ) -> None:
        """Extracted prose normalizes digit runs to <NUM> and collapses whitespace."""
        captured = _mock_classifier(
            monkeypatch,
            [_classifier_json_response('{"is_context_overflow": false}')],
        )
        assert await is_overflow(exc, backend=_BACKEND, before_retry=False) is False
        assert len(captured) == 1
        user_content = captured[0][1]["messages"][1]["content"]
        assert user_content.endswith(f"\n{expected_normalized}")
        assert await store_module.context_error_store.get(expected_normalized) is False


class TestIsOverflowCacheAndFailClosed:
    """Persistent 30-day cache reuse, backend precedence, and fail-closed errors."""

    @pytest.mark.parametrize("verdict", [True, False])
    async def test_normalized_template_hits_cache_for_different_numbers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        verdict: bool,
    ) -> None:
        """A second error with different token numbers reuses the cached verdict without AI."""
        payload = json.dumps({"is_context_overflow": verdict})
        captured = _mock_classifier(monkeypatch, [_classifier_json_response(payload)])
        first = BadRequestError(
            "400",
            response=_resp(400),
            body={"message": "Requested 14,200 tokens, maximum context is 8,192."},
        )
        second = BadRequestError(
            "400",
            response=_resp(400),
            body={"message": "Requested   99,999\ntokens, maximum context is 32,768."},
        )
        assert await is_overflow(first, backend=_BACKEND) is verdict
        assert len(captured) == 1
        _forbid_classifier(monkeypatch)
        assert await is_overflow(second, backend=_BACKEND) is verdict

    async def test_expired_cache_entry_reclassifies_and_refreshes_ttl(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rows older than 30 days miss the cache, call the classifier, and update the row."""
        key = "Prompt <NUM> exceeds <NUM>"
        stale_ts = (datetime.now(timezone.utc) - timedelta(days=_EXPIRED_DAYS)).isoformat()
        await store_module.context_error_store.put(key, False, stale_ts)
        captured = _mock_classifier(
            monkeypatch,
            [_classifier_json_response('{"is_context_overflow": true}')],
        )
        exc = BadRequestError("400", response=_resp(400), body={"message": "Prompt 9000 exceeds 4096"})
        assert await is_overflow(exc, backend=_BACKEND) is True
        assert len(captured) == 1
        assert await store_module.context_error_store.get(key) is True

    async def test_backend_resolution_prefers_explicit_then_thinking_then_reason(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Classifier uses explicit backend, else REASON_THINKING, else REASON."""
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://thinking.test")
        monkeypatch.setenv("REASON_BACKEND_URL", "http://reason.test")
        captured = _mock_classifier(
            monkeypatch,
            [
                _classifier_json_response('{"is_context_overflow": true}'),
                _classifier_json_response('{"is_context_overflow": false}'),
                _classifier_json_response('{"is_context_overflow": true}'),
            ],
        )
        err_a = BadRequestError("400", response=_resp(400), body={"message": "Alpha limit 100"})
        err_b = BadRequestError("400", response=_resp(400), body={"message": "Beta limit 100"})
        err_c = BadRequestError("400", response=_resp(400), body={"message": "Gamma limit 100"})
        assert await is_overflow(err_a, backend=_BACKEND) is True
        assert await is_overflow(err_b) is False
        monkeypatch.delenv("REASON_THINKING_BACKEND_URL")
        assert await is_overflow(err_c) is True
        assert [url for url, _ in captured] == [
            "http://classifier.test/v1/chat/completions",
            "http://thinking.test/v1/chat/completions",
            "http://reason.test/v1/chat/completions",
        ]

    @pytest.mark.parametrize(
        "response",
        [
            _classifier_json_response("server crash", status_code=500),
            _classifier_json_response("not valid json at all"),
            _classifier_json_response("[1, 2, 3]"),
            _classifier_json_response("{}"),
            _classifier_json_response('{"is_context_overflow": "true"}'),
            _classifier_json_response('{"is_context_overflow": 1}'),
        ],
        ids=[
            "http_500",
            "invalid_json",
            "json_array",
            "missing_field",
            "string_verdict",
            "int_verdict",
        ],
    )
    async def test_classifier_failures_raise_classification_error_without_caching(
        self,
        monkeypatch: pytest.MonkeyPatch,
        response: httpx.Response,
    ) -> None:
        """Any HTTP, JSON, or schema failure raises ClassificationError and caches nothing."""
        _mock_classifier(monkeypatch, [response])
        exc = BadRequestError("400", response=_resp(400), body={"message": "Prompt 500 exceeds 100"})
        with pytest.raises(ClassificationError):
            await is_overflow(exc, backend=_BACKEND)
        assert await store_module.context_error_store.get("Prompt <NUM> exceeds <NUM>") is None

    async def test_missing_backend_and_closed_store_raise_classification_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing backend settings or a closed cache store raise ClassificationError."""
        monkeypatch.delenv("REASON_THINKING_BACKEND_URL", raising=False)
        monkeypatch.delenv("REASON_BACKEND_URL", raising=False)
        exc = BadRequestError("400", response=_resp(400), body={"message": "Prompt 500 exceeds 100"})
        with pytest.raises(ClassificationError, match="backend missing"):
            await is_overflow(exc)
        await store_module.task_store.close()
        with pytest.raises(ClassificationError, match="cache read failed"):
            await is_overflow(exc, backend=_BACKEND)


class TestCountChars:
    """count_chars matches exact wire payload lengths across all item shapes."""

    @pytest.mark.parametrize(
        ("item", "expected"),
        [
            (
                {
                    "type": "reasoning",
                    "summary": [{"text": "s1"}, {"text": ""}, "bad"],
                    "content": [{"text": "c1"}, {"other": "ignored"}],
                },
                "s1\nc1",
            ),
            ({"type": "reasoning", "summary": "not_a_list"}, ""),
            ({}, ""),
        ],
        ids=["summary_then_content", "non_list_field", "empty_item"],
    )
    def test_extract_reasoning_text(self, item: dict, expected: str) -> None:
        """Summary text comes before content text, and empty entries are skipped."""
        assert extract_reasoning_text(item) == expected

    @pytest.mark.parametrize(
        ("file_data", "filename", "expected"),
        [
            ("data:video/mp4;base64,VID", None, "data:video/mp4;base64,VID"),
            ("RAWVID", "clip.mp4", "data:video/mp4;base64,RAWVID"),
            ("data:application/pdf;base64,PDF", "clip.mp4", None),
            ("data:;base64,BAD", "clip.mp4", None),
            ("RAWPDF", "doc.pdf", None),
            ("RAWUNKNOWN", None, None),
            ("", "clip.mp4", None),
        ],
        ids=[
            "video_data_url",
            "raw_base64_with_video_filename",
            "pdf_data_url_ignores_video_filename",
            "missing_mime_in_data_url",
            "non_video_filename",
            "raw_without_filename",
            "empty_file_data",
        ],
    )
    def test_resolve_video_url(
        self,
        file_data: str,
        filename: str | None,
        expected: str | None,
    ) -> None:
        """Video inputs return data URLs while empty or other file types return None."""
        assert resolve_video_url(file_data, filename) == expected

    def test_str_input_and_system_prompt(self) -> None:
        """String input counts system_prompt plus prompt length."""
        assert count_chars("hello world", system_prompt="be brief") == len("be brief") + len("hello world")
        assert count_chars("", system_prompt="") == 0

    def test_messages_reasoning_tool_calls_and_media_tool_outputs(self) -> None:
        """Structured messages, coalesced reasoning, refusals, and video outputs match wire size."""
        items: list[dict] = [
            {"role": "user", "content": "hi"},
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAA"},
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,VVV"}},
                ],
            },
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": "dev"},
                    {"type": "input_image", "image_url": "ignored-on-developer"},
                ],
            },
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "s1"}, {"type": "summary_text", "text": ""}],
                "content": [{"type": "reasoning_text", "text": "c1"}],
            },
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "c2"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "p1"},
                    {"type": "output_text", "text": "p2"},
                    {"type": "refusal", "refusal": "first_ignored"},
                    {"type": "refusal", "refusal": "final_refusal"},
                ],
            },
            {"type": "function_call", "call_id": "c1", "name": "fetch", "arguments": '{"k":1}'},
            {"type": "function_call", "call_id": "c2", "name": "noop", "arguments": ""},
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [
                    {"type": "input_text", "text": "out"},
                    {"type": "input_image", "image_url": "data:image/png;base64,IMG"},
                    {"type": "input_file", "file_data": "data:video/mp4;base64,VID", "filename": "v.mp4"},
                    {"type": "input_file", "file_data": "RAWVID", "filename": "clip.mp4"},
                    {"type": "input_file", "file_data": "data:application/pdf;base64,PDF"},
                ],
            },
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "standalone"}],
            },
        ]
        expected = (
            len("sys")
            + len("hi")
            + len("look")
            + len("data:image/png;base64,AAA")
            + len("data:video/mp4;base64,VVV")
            + len("dev")
            + len("s1\nc1\nc2")
            + len("p1\np2")
            + len("final_refusal")
            + len("fetch")
            + len('{"k":1}')
            + len("noop")
            + len("{}")
            + len("out")
            + len("data:image/png;base64,IMG")
            + len("data:video/mp4;base64,VID")
            + len("data:video/mp4;base64,RAWVID")
            + len("standalone")
        )
        assert count_chars(items, system_prompt="sys") == expected


class TestCompact:
    """Whole-item oldest-first compaction across reasoning, assistant replies, and filters."""

    async def test_removes_minimal_oldest_reasoning_prefix_when_sufficient(self) -> None:
        """Compaction stops as soon as removing oldest reasoning items reaches 50 percent."""
        r_old_1 = {"id": "r1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "a" * 30}]}
        r_old_2 = {"id": "r2", "type": "reasoning", "summary": [{"type": "summary_text", "text": "b" * 30}]}
        r_new_3 = {"id": "r3", "type": "reasoning", "summary": [{"type": "summary_text", "text": "c" * 20}]}
        user = {"role": "user", "content": "u" * 10}
        asst = {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "x" * 10}],
        }
        original = [user, r_old_1, r_old_2, r_new_3, asst]
        snapshot = json.loads(json.dumps(original))

        compacted = await compact(original)
        assert compacted == [user, r_new_3, asst]
        assert original == snapshot

    async def test_falls_through_to_oldest_assistant_messages_and_preserves_tools(self) -> None:
        """When reasoning removal is under half, oldest assistant messages drop while tools stay."""
        user = {"role": "user", "content": "u" * 10}
        dev = {"role": "developer", "content": "d" * 5}
        r1 = {"type": "reasoning", "summary": [{"type": "summary_text", "text": "r" * 20}]}
        asst_easy_old = {"role": "assistant", "content": "e" * 25}
        call = {"type": "function_call", "call_id": "c1", "name": "fn", "arguments": "{}"}
        call_out = {"type": "function_call_output", "call_id": "c1", "output": "o" * 10}
        asst_resp_mid = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "m" * 20}],
        }
        asst_resp_new = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "n" * 10}],
        }
        items = [user, dev, r1, asst_easy_old, call, call_out, asst_resp_mid, asst_resp_new]

        compacted = await compact(items, system_prompt="s" * 6)
        assert compacted == [user, dev, call, call_out, asst_resp_new]

    @pytest.mark.parametrize("is_async", [False, True])
    async def test_normalizes_str_and_honors_sync_and_async_measure(self, is_async: bool) -> None:
        """String input normalizes to a user message list before measure runs, and measure drives sizing."""
        seen_inputs: list[list[dict]] = []

        def _sync_measure(candidate: list[dict]) -> int:
            seen_inputs.append(candidate)
            return sum(
                10 if item.get("id") == "pruned_by_filter" else count_chars([item])
                for item in candidate
            )

        async def _async_measure(candidate: list[dict]) -> int:
            return _sync_measure(candidate)

        measure_fn = _async_measure if is_async else _sync_measure
        with pytest.raises(CompactionError):
            await compact("initial prompt", measure=measure_fn)
        assert seen_inputs[0] == [{"role": "user", "content": "initial prompt"}]

        items = [
            {"role": "user", "content": "u" * 20},
            {"id": "pruned_by_filter", "type": "reasoning", "summary": [{"text": "r" * 500}]},
            {"id": "real_heavy", "role": "assistant", "content": "a" * 60},
            {"id": "keep_new", "role": "assistant", "content": "k" * 10},
        ]
        compacted = await compact(items, measure=measure_fn)
        assert [item.get("id", item.get("role")) for item in compacted] == ["user", "keep_new"]

    @pytest.mark.parametrize(
        "items",
        [
            [],
            "",
            [{"role": "user", "content": "u" * 80}, {"role": "assistant", "content": "a" * 20}],
            [{"role": "assistant", "content": "only item cannot leave empty candidate"}],
            [{"type": "reasoning", "summary": [{"text": "only reasoning item"}]}],
        ],
        ids=[
            "empty_list",
            "empty_str",
            "insufficient_reduction",
            "single_assistant_would_empty_list",
            "single_reasoning_would_empty_list",
        ],
    )
    async def test_raises_compaction_error_when_half_reduction_impossible(
        self,
        items: str | list[dict],
    ) -> None:
        """Empty inputs, unshrinkable histories, and single-item lists raise CompactionError."""
        with pytest.raises(CompactionError):
            await compact(items)


@pytest.mark.live
@pytest.mark.timeout(_LIVE_TIMEOUT_S)
@pytest.mark.skipif(not is_llm_available(), reason="LLM backend not reachable")
class TestIsOverflowLive:
    """Live LLM classification and persistent cache reuse against the real backend."""

    async def test_live_overflow_and_non_overflow_classification_with_persistent_cache(
        self,
    ) -> None:
        """Real backend classifies overflow and parameter errors and reuses the cache."""
        live_backend = config.backend(config.REASON_THINKING, config.REASON)
        dead_backend = config.Backend(url="http://127.0.0.1:9/v1", model="none", api_key=None)

        overflow_first = BadRequestError(
            "400",
            response=_resp(400),
            body={
                "error": {
                    "message": (
                        "This model's maximum context length is 32768 tokens. "
                        "However, your messages resulted in 41250 tokens. "
                        "Please reduce the length of the messages."
                    )
                }
            },
        )
        overflow_second = BadRequestError(
            "400",
            response=_resp(400),
            body={
                "error": {
                    "message": (
                        "This model's maximum context length is 8192 tokens. "
                        "However, your messages resulted in 12000 tokens. "
                        "Please reduce the length of the messages."
                    )
                }
            },
        )
        assert await is_overflow(overflow_first, backend=live_backend) is True
        expected_overflow_key = (
            "This model's maximum context length is <NUM> tokens. "
            "However, your messages resulted in <NUM> tokens. "
            "Please reduce the length of the messages."
        )
        assert await store_module.context_error_store.get(expected_overflow_key) is True
        assert await is_overflow(overflow_second, backend=dead_backend) is True

        param_error = BadRequestError(
            "400",
            response=_resp(400),
            body={
                "error": {
                    "message": "Invalid value for 'temperature': must be between 0.0 and 2.0, got 5.5."
                }
            },
        )
        assert await is_overflow(param_error, backend=live_backend) is False
        expected_param_key = (
            "Invalid value for 'temperature': must be between <NUM> and <NUM>, got <NUM>."
        )
        assert await store_module.context_error_store.get(expected_param_key) is False
