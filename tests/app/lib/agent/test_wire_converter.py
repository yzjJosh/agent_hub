"""Unit tests for the chat completions wire converter."""

from __future__ import annotations

import pytest
from agents import ToolOutputFileContent, ToolOutputImage, ToolOutputText
from agents.items import ItemHelpers
from agents.models import chatcmpl_converter, openai_chatcompletions
from agents.models.reasoning_content_replay import (
    ReasoningContentReplayContext,
    ReasoningContentSource,
    ShouldReplayReasoningContent,
)
from openai.types.responses import ResponseFunctionToolCall

import app.lib.agent.context as context_module
from app.lib.agent.wire_converter import install_wire_converter, should_always_replay_reasoning


def _convert_items_to_dicts(
    items: list[dict],
    *,
    model: str | None = "default",
    preserve_thinking_blocks: bool = False,
    should_replay_reasoning_content: ShouldReplayReasoningContent | None = None,
) -> list[dict]:
    converted = openai_chatcompletions.Converter.items_to_messages(
        items,  # type: ignore[arg-type]
        model=model,
        preserve_thinking_blocks=preserve_thinking_blocks,
        should_replay_reasoning_content=should_replay_reasoning_content,
    )
    return [dict(message) for message in converted]


def _extract_tool_content(outputs: list[ToolOutputText | ToolOutputImage | ToolOutputFileContent]) -> list[dict]:
    """Convert one tool output list and return the wire content parts."""
    call = ResponseFunctionToolCall(type="function_call", call_id="c1", name="f", arguments="{}")
    items: list[dict] = [
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        dict(ItemHelpers.tool_call_output_item(call, outputs)),
    ]
    messages = _convert_items_to_dicts(items, model="default")
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    assert len(tool_messages) == 1
    content = tool_messages[0]["content"]
    assert isinstance(content, list)
    return content


def _count_wire_payload_chars(messages: list[dict], system_prompt: str = "") -> int:
    total = len(system_prompt)
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total += len(part["text"])
                elif part.get("type") == "image_url":
                    total += len(part["image_url"]["url"])
                elif part.get("type") == "video_url":
                    total += len(part["video_url"]["url"])
        if isinstance(message.get("refusal"), str):
            total += len(message["refusal"])
        if isinstance(message.get("reasoning_content"), str):
            total += len(message["reasoning_content"])
        for tool_call in message.get("tool_calls", []):
            function_info = tool_call["function"]
            total += len(function_info["name"]) + len(function_info["arguments"])
    return total


def _build_multi_turn_reasoning_items(easy_assistant: dict) -> list[dict]:
    return [
        {"role": "user", "content": "start"},
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
                {"type": "output_text", "text": "line1"},
                {"type": "output_text", "text": "line2"},
                {"type": "refusal", "refusal": "old"},
                {"type": "refusal", "refusal": "kept"},
            ],
        },
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "tool_thought"}]},
        {"type": "function_call", "call_id": "c1", "name": "act", "arguments": '{"a":1}'},
        {"type": "function_call_output", "call_id": "c1", "output": "done"},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "easy_thought"}]},
        easy_assistant,
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "mid_standalone"}]},
        {"role": "user", "content": "next"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "plain"}]},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "tail_standalone"}]},
    ]


class TestWireConverter:
    """Verify wire converter installation, media conversion, and reasoning replay."""

    def test_should_always_replay_reasoning_returns_true(self) -> None:
        replay_context = ReasoningContentReplayContext(
            model="default",
            base_url="http://backend.test",
            reasoning=ReasoningContentSource(item={}, origin_model=None, provider_data={}),
        )
        assert should_always_replay_reasoning(replay_context) is True

    def test_install_wire_converter_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_chatcompletions, "Converter", chatcmpl_converter.Converter)
        install_wire_converter()
        installed = openai_chatcompletions.Converter
        assert installed is not chatcmpl_converter.Converter
        install_wire_converter()
        assert openai_chatcompletions.Converter is installed

    def test_image_part_reaches_the_wire(self) -> None:
        install_wire_converter()
        content = _extract_tool_content([
            ToolOutputText(text="t"),
            ToolOutputImage(image_url="data:image/png;base64,AAA"),
        ])
        assert content[0] == {"type": "text", "text": "t"}
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAA", "detail": "auto"},
        }

    def test_video_file_part_becomes_video_url_part(self) -> None:
        install_wire_converter()
        content = _extract_tool_content([
            ToolOutputText(text="t"),
            ToolOutputFileContent(file_data="data:video/mp4;base64,BBB", filename="v.mp4"),
        ])
        assert content[1] == {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,BBB"}}
        assert all(part.get("type") != "file" for part in content)

    def test_non_video_file_part_is_dropped_and_text_survives(self) -> None:
        install_wire_converter()
        content = _extract_tool_content([
            ToolOutputText(text="t"),
            ToolOutputFileContent(file_data="data:application/pdf;base64,CCC"),
        ])
        assert content == [{"type": "text", "text": "t"}]

    def test_string_input_becomes_single_user_message(self) -> None:
        install_wire_converter()
        messages = openai_chatcompletions.Converter.items_to_messages("hello")
        assert messages == [{"role": "user", "content": "hello"}]

    @pytest.mark.parametrize("model_name", [None, "default", "gpt-4o", "deepseek-reasoner"])
    def test_reasoning_preserved_across_models_and_turn_shapes(
        self,
        model_name: str | None,
    ) -> None:
        install_wire_converter()
        easy_assistant = {"role": "assistant", "content": "easy reply"}
        items = _build_multi_turn_reasoning_items(easy_assistant)
        messages = _convert_items_to_dicts(items, model=model_name)
        assert easy_assistant == {"role": "assistant", "content": "easy reply"}
        assert messages[1]["reasoning_content"] == "s1\nc1\nc2"
        assert messages[1]["content"] == "line1\nline2"
        assert messages[1]["refusal"] == "kept"
        assert messages[2]["reasoning_content"] == "tool_thought"
        assert messages[2]["content"] is None
        assert messages[4] == {
            "role": "assistant",
            "content": "easy reply",
            "reasoning_content": "easy_thought",
        }
        assert messages[5] == {
            "role": "assistant",
            "content": "",
            "reasoning_content": "mid_standalone",
        }
        assert "reasoning_content" not in messages[7]
        assert messages[8] == {
            "role": "assistant",
            "content": "",
            "reasoning_content": "tail_standalone",
        }
        assert context_module.count_chars(items, system_prompt="sys") == _count_wire_payload_chars(
            messages, system_prompt="sys"
        )

    def test_replay_callback_false_skips_reasoning_content_with_thinking_blocks_on(self) -> None:
        install_wire_converter()
        items: list[dict] = [
            {"role": "user", "content": "start"},
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "hidden summary"}],
                "content": [{"type": "reasoning_text", "text": "hidden content"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "visible reply"}],
            },
        ]
        messages = _convert_items_to_dicts(
            items,
            model="default",
            preserve_thinking_blocks=True,
            should_replay_reasoning_content=lambda _context: False,
        )
        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "visible reply"
        assert "reasoning_content" not in messages[1]
