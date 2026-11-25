"""Prune screenshots and compact DOM dumps in browser model history."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from agents.run_config import CallModelData, ModelInputData

from app.agents.browser_use.actions import ACTION_TOOL_NAMES, DUMP_PAGE_TOOL_NAME, OBSERVATION_MARKERS
from app.agents.browser_use.checkpoint import CheckpointTool

__all__ = ["compact_dom_dumps", "prune_model_history"]

_OLDER_SCREENSHOT_MARKER = "[earlier screenshot omitted]"
_HISTORICAL_TOOL_NAMES = frozenset({CheckpointTool.name})
_TAB_CAPTION_PATTERN = re.compile(r"^\d+\. (\S+) —")


@dataclass(frozen=True)
class _ObservationHeader:
    """Open tab ids extracted from the lines before the observation status line."""

    open_tab_ids: frozenset[str]


@dataclass(frozen=True)
class _DumpHeader:
    """Structured metadata parsed from a page dump output."""

    target_id: str
    is_archived: bool
    checkpoint_id: str | None
    capture_metadata_lines: tuple[str, ...]


def _has_image_part(output: str | list | None) -> bool:
    """Return whether a tool output contains at least one image part."""
    return isinstance(output, list) and any(
        isinstance(part, dict) and part.get("type") == "input_image" for part in output
    )


def _is_action_output(item: dict, tool_names_by_call_id: dict[str, str]) -> bool:
    """Return whether an input item is the output of a browser action tool."""
    call_id = item.get("call_id")
    tool_name = tool_names_by_call_id.get(call_id) if isinstance(call_id, str) else None
    if tool_name in _HISTORICAL_TOOL_NAMES:
        return False
    return item.get("type") == "function_call_output" and tool_name in ACTION_TOOL_NAMES


def _is_action_screenshot(item: dict, tool_names_by_call_id: dict[str, str]) -> bool:
    """Return whether an input item is a browser action output with a screenshot."""
    return _is_action_output(item, tool_names_by_call_id) and _has_image_part(item.get("output"))


def _drop_older_images(item: dict) -> dict:
    """Copy a tool output item with its image parts replaced by a single text marker."""
    parts: list = []
    has_placed_marker = False
    for part in item["output"]:
        if isinstance(part, dict) and part.get("type") == "input_image":
            if not has_placed_marker:
                parts.append({"type": "input_text", "text": _OLDER_SCREENSHOT_MARKER})
                has_placed_marker = True
        else:
            parts.append(part)
    pruned = dict(item)
    pruned["output"] = parts
    return pruned


def _extract_output_text(item: dict) -> str:
    """Extract all text from a tool output item."""
    output = item.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "\n".join(part.get("text", "") for part in output if isinstance(part, dict))
    return ""


def _parse_single_token_value(line: str, prefix: str) -> str | None:
    """Parse a single whitespace-free token after a line prefix."""
    if not line.startswith(prefix):
        return None
    value = line[len(prefix):]
    if not value or any(character.isspace() for character in value):
        return None
    return value


def _parse_observation_header(text: str) -> _ObservationHeader | None:
    """Parse open tab ids from an observation header, or return None without a status line."""
    open_tab_ids: set[str] = set()
    for line in text.splitlines():
        if line.startswith(OBSERVATION_MARKERS.status_prefix):
            return _ObservationHeader(open_tab_ids=frozenset(open_tab_ids))
        tab_match = _TAB_CAPTION_PATTERN.match(line)
        if tab_match is not None:
            open_tab_ids.add(tab_match.group(1))
    return None


def _parse_dump_header(text: str) -> _DumpHeader | None:
    """Parse target, archive status, checkpoint id, and capture metadata from a dump output."""
    lines = text.splitlines()
    active_line_index: int | None = None
    target_id: str | None = None
    is_archived = False

    for index, line in enumerate(lines):
        parsed_target = _parse_single_token_value(line, OBSERVATION_MARKERS.dom_dump_target_prefix)
        if parsed_target is not None:
            active_line_index = index
            target_id = parsed_target
            break

    if target_id is None:
        for line in lines:
            parsed_target = _parse_single_token_value(line, OBSERVATION_MARKERS.archived_dom_target_prefix)
            if parsed_target is not None:
                target_id = parsed_target
                is_archived = True
                break

    if target_id is None:
        return None

    header_lines = lines[:active_line_index] if active_line_index is not None else lines
    checkpoint_id: str | None = None
    metadata_lines: list[str] = []
    metadata_prefixes = (
        OBSERVATION_MARKERS.observation_time_prefix,
        OBSERVATION_MARKERS.dom_capture_interval_prefix,
    )
    for line in header_lines:
        if checkpoint_id is None:
            checkpoint_id = _parse_single_token_value(line, OBSERVATION_MARKERS.checkpoint_prefix)
        if any(line.startswith(prefix) and len(line) > len(prefix) for prefix in metadata_prefixes):
            if line not in metadata_lines:
                metadata_lines.append(line)

    return _DumpHeader(
        target_id=target_id,
        is_archived=is_archived,
        checkpoint_id=checkpoint_id,
        capture_metadata_lines=tuple(metadata_lines),
    )


def _extract_latest_observation(items: list, tool_names_by_call_id: dict[str, str]) -> dict | None:
    """Find the most recent browser action output that contains an observation status line."""
    for item in reversed(items):
        if not isinstance(item, dict) or not _is_action_output(item, tool_names_by_call_id):
            continue
        if _parse_observation_header(_extract_output_text(item)) is not None:
            return item
    return None


def _format_dump_stub(header: _DumpHeader) -> str:
    """Format the compact archived DOM stub with checkpoint recovery instructions."""
    lines = [f"{OBSERVATION_MARKERS.archived_dom_target_prefix}{header.target_id}"]
    lines.extend(header.capture_metadata_lines)
    if header.checkpoint_id is None:
        lines.append("DOM body omitted. Checkpoint archive unavailable.")
    else:
        lines.extend([
            f"{OBSERVATION_MARKERS.checkpoint_prefix}{header.checkpoint_id}",
            "DOM artifact available in the checkpoint archive.",
            (
                f"Read checkpoint with action read, checkpoint_id {header.checkpoint_id}, "
                f"target_id {header.target_id}, kind dom."
            ),
        ])
    return "\n".join(lines)


def _prune_dump_item(item: dict, header: _DumpHeader, should_keep_observation: bool) -> dict:
    """Replace a DOM dump body with its archive stub, keeping the current observation when needed."""
    stub = _format_dump_stub(header)
    pruned = dict(item)
    if not should_keep_observation:
        pruned["output"] = stub
        return pruned
    marker = (
        OBSERVATION_MARKERS.archived_dom_target_prefix
        if header.is_archived
        else OBSERVATION_MARKERS.dom_dump_target_prefix
    )

    def replace_text(content: str) -> str:
        return f"{content.split(marker, 1)[0].rstrip()}\n{stub}"

    output = item["output"]
    if isinstance(output, str):
        pruned["output"] = replace_text(output)
    else:
        pruned["output"] = [
            {**part, "text": replace_text(part["text"])}
            if isinstance(part, dict) and isinstance(part.get("text"), str) and marker in part["text"]
            else part
            for part in output
        ]
    return pruned


def _select_pruned_item(item: dict, replacements: dict[str, dict]) -> dict:
    """Return the replacement item for a pruned tool output, or the original item."""
    if item.get("type") != "function_call_output":
        return item
    call_id = item.get("call_id")
    if isinstance(call_id, str) and call_id in replacements:
        return replacements[call_id]
    return item


def _prune_dom_dumps(items: list, tool_names_by_call_id: dict[str, str]) -> list:
    """Trim superseded and closed-tab DOM bodies while keeping checkpoint handles."""
    observation = _extract_latest_observation(items, tool_names_by_call_id)
    open_tab_ids: frozenset[str] | None = None
    if observation is not None:
        observation_header = _parse_observation_header(_extract_output_text(observation))
        if observation_header is not None:
            open_tab_ids = observation_header.open_tab_ids
    kept_dump_tabs: set[str] = set()
    replacements: dict[str, dict] = {}
    for item in reversed(items):
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or tool_names_by_call_id.get(call_id) != DUMP_PAGE_TOOL_NAME:
            continue
        dump_header = _parse_dump_header(_extract_output_text(item))
        if dump_header is None:
            continue
        is_closed_tab = open_tab_ids is not None and dump_header.target_id not in open_tab_ids
        if dump_header.is_archived or dump_header.target_id in kept_dump_tabs or is_closed_tab:
            replacements[call_id] = _prune_dump_item(item, dump_header, item is observation)
        else:
            kept_dump_tabs.add(dump_header.target_id)
    return [_select_pruned_item(item, replacements) if isinstance(item, dict) else item for item in items]


def _prune_screenshots(items: list, tool_names_by_call_id: dict[str, str]) -> list:
    """Keep only the newest browser action screenshot in the input history."""
    observation = _extract_latest_observation(items, tool_names_by_call_id)
    latest_index = -1
    for index, item in enumerate(items):
        if item is observation or (
            observation is None and isinstance(item, dict) and _is_action_screenshot(item, tool_names_by_call_id)
        ):
            latest_index = index
    if latest_index == -1:
        return items
    return [
        _drop_older_images(item)
        if index < latest_index and isinstance(item, dict) and _is_action_screenshot(item, tool_names_by_call_id)
        else item
        for index, item in enumerate(items)
    ]


def _map_tool_names_by_call_id(items: list) -> dict[str, str]:
    """Map each function call id in history to its tool name."""
    tool_names_by_call_id: dict[str, str] = {}
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = item.get("call_id")
            tool_name = item.get("name")
            if isinstance(call_id, str) and isinstance(tool_name, str):
                tool_names_by_call_id[call_id] = tool_name
    return tool_names_by_call_id


def _compact_dom_dumps_sync(items: list[dict], is_verbose: bool = False) -> list[dict] | None:
    """Archive all active DOM dump bodies in history, returning None if none changed."""
    tool_names_by_call_id = _map_tool_names_by_call_id(items)
    observation = _extract_latest_observation(items, tool_names_by_call_id)
    compacted: list[dict] = []
    omitted_count = 0
    for item in items:
        call_id = item.get("call_id")
        if (
            item.get("type") == "function_call_output"
            and isinstance(call_id, str)
            and tool_names_by_call_id.get(call_id) == DUMP_PAGE_TOOL_NAME
        ):
            dump_header = _parse_dump_header(_extract_output_text(item))
            if dump_header is not None and not dump_header.is_archived:
                item = _prune_dump_item(item, dump_header, item is observation)
                omitted_count += 1
        compacted.append(item)
    if is_verbose:
        print(f"[browser_use.compact] omitted {omitted_count} DOM bodies", flush=True)
    return compacted if omitted_count else None


async def compact_dom_dumps(items: list[dict]) -> list[dict] | None:
    """Compact active DOM dump bodies on context overflow.

    Args:
        items: Conversation history items from the current agent run.

    Returns:
        Updated history with active DOM bodies archived, or None when none remain.
    """
    return await asyncio.to_thread(_compact_dom_dumps_sync, items, True)


def prune_model_history(data: CallModelData) -> ModelInputData:
    """Prune superseded DOM dumps and older screenshots before a model call.

    Args:
        data: Model call payload provided by the agent runner.

    Returns:
        Filtered model input with current tab dumps and the newest screenshot kept.
    """
    items = data.model_data.input
    tool_names_by_call_id = _map_tool_names_by_call_id(items)
    items = _prune_dom_dumps(items, tool_names_by_call_id)
    items = _prune_screenshots(items, tool_names_by_call_id)
    return ModelInputData(input=items, instructions=data.model_data.instructions)
