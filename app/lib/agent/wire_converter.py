"""Convert chat history items into chat completions wire messages."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import override

from agents.items import TResponseInputItem
from agents.models import chatcmpl_converter, openai_chatcompletions
from agents.models.reasoning_content_replay import (
    ReasoningContentReplayContext,
    ReasoningContentSource,
    ShouldReplayReasoningContent,
)
from openai.types.chat import ChatCompletionMessageParam

from app.lib.agent import context
from app.lib.shared import config

__all__ = ["install_wire_converter", "should_always_replay_reasoning"]


@dataclass(frozen=True)
class _ReasoningRun:
    items: list[dict]
    next_index: int


def should_always_replay_reasoning(_context: ReasoningContentReplayContext) -> bool:
    """Return True so reasoning text is always sent back to the model."""
    return True


def _convert_media_wire_part(part: dict) -> dict | None:
    if part.get("type") != "file":
        return part
    file_entry = part.get("file")
    file_info = file_entry if isinstance(file_entry, dict) else {}
    file_data = file_info.get("file_data")
    if not isinstance(file_data, str):
        return None
    filename = file_info.get("filename")
    video_url = context.resolve_video_url(file_data, filename if isinstance(filename, str) else None)
    if video_url is None:
        return None
    return {"type": "video_url", "video_url": {"url": video_url}}


def _convert_tool_media_parts(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    converted: list[ChatCompletionMessageParam] = []
    for message in messages:
        if message.get("role") != "tool":
            converted.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            converted.append(message)
            continue
        parts = [
            _convert_media_wire_part(part) if isinstance(part, dict) else part
            for part in content
        ]
        updated_message = dict(message)
        updated_message["content"] = [part for part in parts if part is not None]
        converted.append(updated_message)  # type: ignore[arg-type]
    return converted


def _merge_reasoning_items(
    run_items: list[dict],
    model: str,
    base_url: str | None,
    replay_check: ShouldReplayReasoningContent,
) -> str:
    normalized_url = base_url.rstrip("/") if base_url is not None else None
    texts: list[str] = []
    for item in run_items:
        raw_provider = item.get("provider_data")
        provider_data = raw_provider if isinstance(raw_provider, dict) else {}
        raw_origin = provider_data.get("model")
        origin_model = raw_origin if isinstance(raw_origin, str) and raw_origin else None
        replay_context = ReasoningContentReplayContext(
            model=model,
            base_url=normalized_url,
            reasoning=ReasoningContentSource(
                item=item,
                origin_model=origin_model,
                provider_data=provider_data,
            ),
        )
        if replay_check(replay_context):
            reasoning_text = context.extract_reasoning_text(item)
            if reasoning_text:
                texts.append(reasoning_text)
    return "\n".join(texts)


def _collect_reasoning_run(
    raw_items: list[TResponseInputItem],
    start_index: int,
) -> _ReasoningRun:
    end_index = start_index + 1
    while (
        end_index < len(raw_items)
        and _ChatCompletionsWireConverter.maybe_reasoning_message(raw_items[end_index]) is not None
    ):
        end_index += 1
    items: list[dict] = [dict(item) for item in raw_items[start_index:end_index] if isinstance(item, dict)]
    return _ReasoningRun(items=items, next_index=end_index)


class _ChatCompletionsWireConverter(chatcmpl_converter.Converter):
    """Keep media tool outputs and assistant reasoning text on the wire."""

    @classmethod
    def _convert_batch(
        cls,
        batch: list[TResponseInputItem],
        model: str,
        preserve_thinking_blocks: bool,
        base_url: str | None,
        replay_check: ShouldReplayReasoningContent,
        strict_feature_validation: bool,
    ) -> list[ChatCompletionMessageParam]:
        if not batch:
            return []
        return super().items_to_messages(
            batch,
            model=model,
            preserve_thinking_blocks=preserve_thinking_blocks,
            preserve_tool_output_all_content=True,
            base_url=base_url,
            should_replay_reasoning_content=replay_check,
            strict_feature_validation=strict_feature_validation,
        )

    @classmethod
    def _emit_reasoning_turn(
        cls,
        messages: list[ChatCompletionMessageParam],
        pending: list[TResponseInputItem],
        reasoning_items: list[dict],
        merged_reasoning: str,
        next_item: TResponseInputItem | None,
        next_index: int,
        preserve_thinking_blocks: bool,
    ) -> int:
        if (
            cls.maybe_response_output_message(next_item) is not None
            or cls.maybe_function_tool_call(next_item) is not None
            or cls.maybe_file_search_call(next_item) is not None
        ):
            if merged_reasoning:
                merged_item = dict(reasoning_items[-1])
                merged_item["summary"] = [{"type": "summary_text", "text": merged_reasoning}]
                pending.append(merged_item)  # type: ignore[arg-type]
            elif preserve_thinking_blocks:
                pending.extend(reasoning_items)  # type: ignore[arg-type]
            return next_index
        easy_message = cls.maybe_easy_input_message(next_item)
        if easy_message is not None and easy_message["role"] == "assistant":
            assistant_message: dict = {
                "role": "assistant",
                "content": cls.extract_text_content(easy_message["content"]),
            }
            if merged_reasoning:
                assistant_message["reasoning_content"] = merged_reasoning
            messages.append(assistant_message)  # type: ignore[arg-type]
            return next_index + 1
        if merged_reasoning:
            standalone_message: dict = {
                "role": "assistant",
                "content": "",
                "reasoning_content": merged_reasoning,
            }
            messages.append(standalone_message)  # type: ignore[arg-type]
        return next_index

    @classmethod
    @override
    def items_to_messages(  # type: ignore[override]
        cls,
        items: str | Iterable[TResponseInputItem],
        model: str | None = None,
        preserve_thinking_blocks: bool = False,
        base_url: str | None = None,
        should_replay_reasoning_content: ShouldReplayReasoningContent | None = None,
        strict_feature_validation: bool = False,
    ) -> list[ChatCompletionMessageParam]:
        if isinstance(items, str):
            return [{"role": "user", "content": items}]
        effective_model = model if model is not None else config.DEFAULT_MODEL
        replay_check = (
            should_replay_reasoning_content
            if should_replay_reasoning_content is not None
            else should_always_replay_reasoning
        )
        raw_items = list(items)
        messages: list[ChatCompletionMessageParam] = []
        pending: list[TResponseInputItem] = []
        index = 0
        while index < len(raw_items):
            if cls.maybe_reasoning_message(raw_items[index]) is None:
                pending.append(raw_items[index])
                index += 1
                continue
            messages.extend(
                cls._convert_batch(
                    pending, effective_model, preserve_thinking_blocks,
                    base_url, replay_check, strict_feature_validation,
                )
            )
            pending = []
            reasoning_run = _collect_reasoning_run(raw_items, index)
            merged_reasoning = _merge_reasoning_items(
                reasoning_run.items, effective_model, base_url, replay_check
            )
            next_item = raw_items[reasoning_run.next_index] if reasoning_run.next_index < len(raw_items) else None
            index = cls._emit_reasoning_turn(
                messages, pending, reasoning_run.items, merged_reasoning,
                next_item, reasoning_run.next_index, preserve_thinking_blocks,
            )
        messages.extend(
            cls._convert_batch(
                pending, effective_model, preserve_thinking_blocks,
                base_url, replay_check, strict_feature_validation,
            )
        )
        return _convert_tool_media_parts(messages)


def install_wire_converter() -> None:
    """Install the chat completions wire converter."""
    if openai_chatcompletions.Converter is not _ChatCompletionsWireConverter:
        openai_chatcompletions.Converter = _ChatCompletionsWireConverter  # type: ignore[misc]
