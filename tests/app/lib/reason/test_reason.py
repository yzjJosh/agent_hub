import asyncio
import os
import httpx
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest

from app.lib.reason.reason import reason

_REASON_VARS = (
    "REASON_BACKEND_URL", "REASON_MODEL", "REASON_API_KEY", "VISION_MODEL",
    "REASON_THINKING_BACKEND_URL", "REASON_THINKING_MODEL", "REASON_THINKING_API_KEY",
)
_REASON_PURPOSE = (
    "needed by google_search, google_flights, web_fetch, read_media_or_doc, "
    "rednote_research, tts and captcha solving"
)
_THINKING_PURPOSE = "needed by research and browser_use"

# The shape of a reply from the model server.
_MOCK_CHAT_COMPLETIONS_RESPONSE = {
    "id": "chatcmpl_test123",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Mocked response from LLM"
            },
            "finish_reason": "stop"
        }
    ]
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the model settings before each test."""
    for name in _REASON_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def _mock_llm_response() -> Generator[AsyncMock, None, None]:
    """Answer every request with a canned model reply."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _MOCK_CHAT_COMPLETIONS_RESPONSE.copy()
    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        yield mock_post


def _get_test_image(data_dir: str) -> str:
    """Return path to existing test image."""
    path = os.path.join(data_dir, 'test.jpg')
    assert os.path.isfile(path), f"Missing test resource: {path}"
    return path


def _get_test_video(data_dir: str) -> str:
    """Return path to existing test video."""
    path = os.path.join(data_dir, 'test_video.mp4')
    assert os.path.isfile(path), f"Missing test resource: {path}"
    return path


def _get_call_args_payload(mock_post: Mock) -> tuple[str, dict]:
    """Get the address and the body of the last request."""
    args, kwargs = mock_post.call_args
    return args[0], kwargs['json']


def _assert_vision_messages(payload: dict) -> None:
    """Check the body holds a system message and a user message made of parts."""
    messages = payload['messages']
    assert len(messages) == 2
    assert messages[0]['role'] == 'system'
    assert messages[1]['role'] == 'user'
    assert isinstance(messages[1]['content'], list)


def _assert_vision_content(payload: dict) -> None:
    """Check the user message holds a text part and an image part."""
    content = payload['messages'][1]['content']
    assert isinstance(content, list)
    assert any(c.get('type') == 'text' for c in content)
    assert any(c.get('type') == 'image_url' for c in content)


# =============================================================================
# Core reason tests
# =============================================================================

async def test_reason_text_only(_mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """A text prompt goes to the normal backend as a system message and a user message."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    result, status = await reason(prompt="2+2=?")

    assert status == 0
    assert "Mocked response from LLM" in result

    url, payload = _get_call_args_payload(_mock_llm_response)
    assert '/v1/chat/completions' in url
    assert url == "http://localhost:8080/v1/chat/completions"

    assert 'model' not in payload
    assert 'response_format' not in payload
    messages = payload['messages']
    assert len(messages) == 2
    assert messages[0]['role'] == 'system'
    assert messages[1]['role'] == 'user'
    assert isinstance(messages[1]['content'], str)
    assert "2+2=?" in messages[1]['content']


async def test_reason_with_image(_mock_llm_response: AsyncMock, data_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An image prompt sends a user message made of a text part and an image part."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    test_image = _get_test_image(data_dir)

    result, status = await reason(prompt="这是什么？", image_files=[test_image])
    assert status == 0
    assert "Mocked response from LLM" in result

    url, payload = _get_call_args_payload(_mock_llm_response)
    assert '/v1/chat/completions' in url

    assert 'model' not in payload
    _assert_vision_messages(payload)
    _assert_vision_content(payload)


async def test_reason_with_video(_mock_llm_response: AsyncMock, data_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A video prompt sends one video part holding the video data."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    test_video = _get_test_video(data_dir)

    result, status = await reason(prompt="Describe this video", video_files=[test_video])
    assert status == 0
    assert "Mocked response from LLM" in result

    _url, payload = _get_call_args_payload(_mock_llm_response)
    _assert_vision_messages(payload)
    content = payload['messages'][1]['content']
    assert any(c.get('type') == 'text' for c in content)
    video_parts = [c for c in content if c.get('type') == 'video_url']
    assert len(video_parts) == 1
    assert video_parts[0]['video_url']['url'].startswith("data:video/mp4;base64,")


async def test_reason_with_image_and_video(_mock_llm_response: AsyncMock, data_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A prompt with both files sends an image part and a video part."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    _result, status = await reason(
        prompt="Compare these",
        image_files=[_get_test_image(data_dir)],
        video_files=[_get_test_video(data_dir)],
    )
    assert status == 0

    _url, payload = _get_call_args_payload(_mock_llm_response)
    content = payload['messages'][1]['content']
    assert any(c.get('type') == 'image_url' for c in content)
    assert any(c.get('type') == 'video_url' for c in content)


async def test_reason_video_not_found(_mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing video file fails before any request is sent."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    result, status = await reason(prompt="test", video_files=["/nonexistent/video.mp4"])

    assert status == 1
    assert "Video validation failed." in result
    _mock_llm_response.assert_not_called()


async def test_reason_rejects_non_video_file(_mock_llm_response: AsyncMock, data_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file that is not a video fails before any request is sent."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    result, status = await reason(prompt="test", video_files=[_get_test_image(data_dir)])

    assert status == 1
    assert "Video validation failed." in result
    _mock_llm_response.assert_not_called()


async def test_reason_with_context(_mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Context is put in the user message along with the prompt."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    _result, status = await reason(prompt="Summarize.", context="Here is some background info.")

    assert status == 0
    _, payload = _get_call_args_payload(_mock_llm_response)
    user_content = payload['messages'][1]['content']
    assert "Additional Context: Here is some background info." in user_content
    assert "User Query: Summarize." in user_content


# =============================================================================
# Thinking backend URL tests
# =============================================================================

async def test_reason_thinking_uses_thinking_backend_url(_mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Thinking mode sends the request to the thinking backend."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")
    monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://thinking-backend:9000")

    _result, status = await reason(prompt="think hard", thinking=True)
    assert status == 0

    url, _payload = _get_call_args_payload(_mock_llm_response)
    assert url == "http://thinking-backend:9000/v1/chat/completions"


async def test_reason_non_thinking_uses_default_backend_url(_mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Normal mode sends the request to the normal backend."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")
    monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://thinking-backend:9000")

    _result, status = await reason(prompt="no thinking", thinking=False)
    assert status == 0

    url, _payload = _get_call_args_payload(_mock_llm_response)
    assert url == "http://localhost:8080/v1/chat/completions"


async def test_reason_thinking_and_normal_use_different_urls(_mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Thinking calls and normal calls go to different backends."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://normal:8080")
    monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://thinking:9000")

    await reason(prompt="normal call", thinking=False)
    url_normal, _payload = _get_call_args_payload(_mock_llm_response)
    assert url_normal == "http://normal:8080/v1/chat/completions"

    _mock_llm_response.reset_mock()
    await reason(prompt="thinking call", thinking=True)
    url_thinking, _payload = _get_call_args_payload(_mock_llm_response)
    assert url_thinking == "http://thinking:9000/v1/chat/completions"


# =============================================================================
# System prompt structure tests
# =============================================================================

async def test_two_message_array_roles(_mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """The request holds exactly two messages, a system one then a user one."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    await reason(prompt="hello")

    _, payload = _get_call_args_payload(_mock_llm_response)
    messages = payload['messages']
    assert len(messages) == 2
    assert [m['role'] for m in messages] == ['system', 'user']


# =============================================================================
# JSON schema tests
# =============================================================================

@pytest.mark.parametrize("json_format", [False, True])
async def test_json_schema_payload_level(
    _mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch, json_format: bool,
) -> None:
    """A given schema is sent as the wanted shape of the answer."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    _result, status = await reason(prompt="test", json_schema=schema, json_format=json_format)

    assert status == 0
    _, payload = _get_call_args_payload(_mock_llm_response)
    assert 'response_format' in payload
    assert payload['response_format']['type'] == 'json_schema'
    assert payload['response_format']['json_schema']['schema'] == schema


async def test_json_object_response_format(
    _mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON mode asks the backend for an object without a schema."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")
    _mock_llm_response.return_value.json.return_value = {
        "choices": [{"message": {"content": '{"success": false}'}}],
    }

    result, status = await reason(prompt="Can you inspect this input?", json_format=True)

    assert (result, status) == ('{"success": false}', 0)
    _, payload = _get_call_args_payload(_mock_llm_response)
    assert payload["response_format"] == {"type": "json_object"}
    assert "JSON object" in payload["messages"][0]["content"]


async def test_json_schema_injected_into_system_prompt(_mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """The schema is added to the system message and not to the user message."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    schema = {"type": "object", "properties": {"status": {"type": "string"}}}
    _result, status = await reason(prompt="What is the status?", json_schema=schema)

    assert status == 0
    _, payload = _get_call_args_payload(_mock_llm_response)
    messages = payload['messages']

    system_content = messages[0]['content']
    assert "IMPORTANT OUTPUT INSTRUCTIONS" in system_content
    assert '"status"' in system_content
    assert '"type": "string"' in system_content

    user_content = messages[1]['content']
    assert "IMPORTANT OUTPUT INSTRUCTIONS" not in user_content
    assert '"status"' not in user_content


async def test_json_schema_not_in_user_prompt(_mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """The schema does not show up in the user message."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    schema = {"type": "object", "properties": {"result": {"type": "number"}}}
    await reason(prompt="Calculate 2+2", json_schema=schema)

    _, payload = _get_call_args_payload(_mock_llm_response)
    user_content = payload['messages'][1]['content']
    assert "IMPORTANT OUTPUT INSTRUCTIONS" not in user_content
    assert '"result"' not in user_content


# =============================================================================
# Early fail with no backend URL
# =============================================================================

async def test_reason_no_backend_url(_mock_llm_response: AsyncMock, capsys: pytest.CaptureFixture[str]) -> None:
    """With no backend set, the call fails with a message naming the variable and sends nothing."""
    message = f"REASON_BACKEND_URL is not set: {_REASON_PURPOSE}"

    result, status = await reason(prompt="test")

    assert (result, status) == (message, 1)
    assert f"[config] {message}" in capsys.readouterr().err
    _mock_llm_response.assert_not_called()


async def test_reason_thinking_no_thinking_backend_url(_mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    """With no thinking backend set, a thinking call fails and names the thinking variable."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")
    message = f"REASON_THINKING_BACKEND_URL is not set: {_THINKING_PURPOSE}"

    result, status = await reason(prompt="test", thinking=True)

    assert (result, status) == (message, 1)
    assert f"[config] {message}" in capsys.readouterr().err
    _mock_llm_response.assert_not_called()


# =============================================================================
# Backend address, model and key
# =============================================================================

@pytest.mark.parametrize("base", ["http://localhost:8080", "http://localhost:8080/", "http://localhost:8080/v1"])
async def test_reason_url_has_one_v1(_mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch,
                                     base: str) -> None:
    """A backend given with or without /v1 is asked at one /v1/chat/completions."""
    monkeypatch.setenv("REASON_BACKEND_URL", base)

    _result, status = await reason(prompt="test")

    assert status == 0
    url, _payload = _get_call_args_payload(_mock_llm_response)
    assert url == "http://localhost:8080/v1/chat/completions"


async def test_reason_sends_model_and_key_when_set(_mock_llm_response: AsyncMock,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """A set model is in the body and a set key is sent as a bearer token."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")
    monkeypatch.setenv("REASON_MODEL", "qwen")
    monkeypatch.setenv("REASON_API_KEY", "sk-reason")

    await reason(prompt="test")

    _url, payload = _get_call_args_payload(_mock_llm_response)
    assert payload["model"] == "qwen"
    assert _mock_llm_response.call_args.kwargs["headers"] == {"Authorization": "Bearer sk-reason"}


async def test_reason_omits_model_and_key_when_unset(_mock_llm_response: AsyncMock,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """With no model or key set, the body has no model and no Authorization header is sent."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    await reason(prompt="test")

    _url, payload = _get_call_args_payload(_mock_llm_response)
    assert "model" not in payload
    assert "Authorization" not in _mock_llm_response.call_args.kwargs["headers"]


async def test_reason_thinking_uses_thinking_model_and_key(_mock_llm_response: AsyncMock,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """Thinking mode takes its URL, model and key from the REASON_THINKING_* settings."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://normal:8080")
    monkeypatch.setenv("REASON_MODEL", "fast")
    monkeypatch.setenv("REASON_API_KEY", "sk-normal")
    monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://thinking:9000/v1")
    monkeypatch.setenv("REASON_THINKING_MODEL", "deep")
    monkeypatch.setenv("REASON_THINKING_API_KEY", "sk-think")

    await reason(prompt="test", thinking=True)

    url, payload = _get_call_args_payload(_mock_llm_response)
    assert url == "http://thinking:9000/v1/chat/completions"
    assert payload["model"] == "deep"
    assert _mock_llm_response.call_args.kwargs["headers"] == {"Authorization": "Bearer sk-think"}


# =============================================================================
# Endpoint verification
# =============================================================================

async def test_endpoint_is_chat_completions(_mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """The request is sent to /v1/chat/completions."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    await reason(prompt="test")

    url, _payload = _get_call_args_payload(_mock_llm_response)
    assert '/v1/chat/completions' in url, f"Expected /v1/chat/completions in URL, got {url}"


# =============================================================================
# Input format tests
# =============================================================================

async def test_input_format_text_only(_mock_llm_response: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """A text prompt sends two messages whose contents are both plain text."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    await reason(prompt="Hello world")

    _, payload = _get_call_args_payload(_mock_llm_response)

    assert 'messages' in payload
    assert isinstance(payload['messages'], list)
    assert len(payload['messages']) == 2

    assert payload['messages'][0]['role'] == 'system'
    assert isinstance(payload['messages'][0]['content'], str)

    assert payload['messages'][1]['role'] == 'user'
    assert isinstance(payload['messages'][1]['content'], str)
    assert "Hello world" in payload['messages'][1]['content']


async def test_input_format_with_images(_mock_llm_response: AsyncMock, data_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An image prompt sends a system message of text and a user message of parts."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    test_image = _get_test_image(data_dir)
    await reason(prompt="What is this?", image_files=[test_image])

    _, payload = _get_call_args_payload(_mock_llm_response)

    assert 'messages' in payload
    assert isinstance(payload['messages'], list)
    _assert_vision_messages(payload)
    _assert_vision_content(payload)


# =============================================================================
# transport failure and cancellation
# =============================================================================

async def test_reason_backend_failure_returns_status_one(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connection that fails comes back as an error message with status 1."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")

    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ConnectError("connection refused")
        result, status = await reason(prompt="2+2=?")

    assert status == 1
    assert "Communicating with local LLM failed" in result


async def test_reason_is_cancellable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A running call stops when its task is cancelled."""
    monkeypatch.setenv("REASON_BACKEND_URL", "http://localhost:8080")
    started = asyncio.Event()

    async def _never_returns(*_args: str, **_kwargs: dict) -> None:
        started.set()
        await asyncio.sleep(3600)

    with patch('httpx.AsyncClient.post', new=_never_returns):
        task = asyncio.create_task(reason(prompt="2+2=?"))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert task.cancelled()
