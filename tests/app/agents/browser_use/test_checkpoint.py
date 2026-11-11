"""Tests for reading archived browser evidence."""

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from app.agents.browser_use.checkpoint import CheckpointTool
from app.agents.browser_use.checkpoint_store import (
    Artifact,
    CaptureInterval,
    Checkpoint,
    CheckpointStore,
    TabCapture,
)
from app.agents.browser_use.models import TransactionStatus
from app.lib.tool.base import ToolMessageError, ToolResult

_WINDOW_LENGTH = 500
_LONG_TEXT = "回\r\n" * 300


async def _run(archive: CheckpointStore, params: dict) -> ToolResult:
    task = await CheckpointTool(archive).run(params)
    return await task.result()


def _read_args(checkpoint: Checkpoint, **kwargs: str | int) -> dict:
    return {"action": "read", "checkpoint_id": checkpoint.checkpoint_id, "target_id": "tab", **kwargs}


async def test_listing_distinguishes_same_url_views_without_full_evidence(
    archive: CheckpointStore,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    feedback = await publish_checkpoint(archive, "Feedback\n\nLong feedback details")
    transcript = await publish_checkpoint(archive, "Transcript\n\nExact conversation")
    result = await _run(archive, {"action": "list"})
    assert result.text.index(feedback.checkpoint_id) < result.text.index(transcript.checkpoint_id)
    assert 'DOM excerpt: "Feedback\\n\\nLong feedback details"' in result.text
    assert 'DOM excerpt: "Transcript\\n\\nExact conversation"' in result.text
    assert "Source action: dump_page" in result.text
    assert "Source outcome: success" in result.text
    assert "Screenshot: Captured" in result.text
    assert "Listing complete." in result.text
    assert result.files is None
    assert "status:" not in result.text


async def test_listing_filter_and_continuation_are_stable(
    archive: CheckpointStore,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    first = await publish_checkpoint(archive, "Feedback", "closed")
    await publish_checkpoint(archive, "Other tab", "other")
    second = await publish_checkpoint(archive, "Transcript", "closed")
    params = {"action": "list", "target_id": "closed", "limit": 1}
    result = await _run(archive, params)
    assert first.checkpoint_id in result.text
    assert second.checkpoint_id not in result.text
    assert "Next cursor: 1." in result.text
    result = await _run(archive, {**params, "cursor": 1})
    assert second.checkpoint_id in result.text
    assert first.checkpoint_id not in result.text
    assert "Listing complete." in result.text


async def test_empty_checkpoint_and_end_cursor_are_explicit(archive: CheckpointStore) -> None:
    empty = await archive.publish(await archive.reserve(), "close_page", TransactionStatus.SUCCESS, ())
    result = await _run(archive, {"action": "list"})
    assert empty.checkpoint_id in result.text
    assert "No tabs or artifacts." in result.text
    end = await _run(archive, {"action": "list", "cursor": 1})
    assert "No checkpoints match this page." in end.text
    assert "Listing complete." in end.text


async def test_default_dom_read_preserves_exact_text_and_capture_interval(
    archive: CheckpointStore,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
    dom_time: CaptureInterval,
) -> None:
    text = "# Page\r\n\r\nExact words 回\n"
    checkpoint = await publish_checkpoint(archive, text)
    result = await _run(archive, _read_args(checkpoint))
    assert f"Historical checkpoint: {checkpoint.checkpoint_id}" in result.text
    assert f"\n\n{text}\n\nEnd historical DOM window." in result.text
    assert f"Returned range: [0, {len(text)})." in result.text
    assert f"DOM capture interval: {dom_time.started_at.isoformat()} to {dom_time.finished_at.isoformat()}" in result.text
    assert "DOM read complete." in result.text
    assert result.files is None


async def test_dom_continuation_and_end_offset(
    archive: CheckpointStore,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    checkpoint = await publish_checkpoint(archive, _LONG_TEXT)
    first = await _run(archive, _read_args(checkpoint, max_char=_WINDOW_LENGTH))
    assert f"\n\n{_LONG_TEXT[:_WINDOW_LENGTH]}\n\nEnd historical DOM window." in first.text
    assert f"Next offset: {_WINDOW_LENGTH}." in first.text
    second = await _run(archive, _read_args(checkpoint, offset=_WINDOW_LENGTH, max_char=_WINDOW_LENGTH))
    assert f"\n\n{_LONG_TEXT[_WINDOW_LENGTH:]}\n\nEnd historical DOM window." in second.text
    assert "DOM read complete." in second.text
    end = await _run(archive, _read_args(checkpoint, offset=len(_LONG_TEXT)))
    assert f"Returned range: [{len(_LONG_TEXT)}, {len(_LONG_TEXT)})." in end.text


async def test_successful_empty_dom_is_not_reported_as_missing(
    archive: CheckpointStore,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    checkpoint = await publish_checkpoint(archive, "")
    result = await _run(archive, _read_args(checkpoint))
    assert "Empty captured DOM." in result.text
    assert "Total characters: 0" in result.text
    assert "DOM read complete." in result.text


async def test_screenshot_read_returns_original_path_and_ignores_dom_windows(
    archive: CheckpointStore,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
    image_time: CaptureInterval,
) -> None:
    checkpoint = await publish_checkpoint(archive)
    result = await _run(archive, _read_args(checkpoint, kind="screenshot", offset=1000, max_char=_WINDOW_LENGTH))
    assert result.files == (checkpoint.tabs[0].screenshot.path,)
    assert Path(result.files[0]).read_bytes() == b"original image"
    assert f"Screenshot capture interval: {image_time.started_at.isoformat()} to {image_time.finished_at.isoformat()}" in result.text
    assert "status:" not in result.text


async def test_list_ignores_read_arguments(
    archive: CheckpointStore,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    checkpoint = await publish_checkpoint(archive)
    result = await _run(archive, {"action": "list", "checkpoint_id": "foreign", "kind": "screenshot", "offset": 1000})
    assert checkpoint.checkpoint_id in result.text
    assert result.files is None


@pytest.mark.parametrize("params", [
    {"action": "read"},
    {"action": "read", "checkpoint_id": "id"},
    {"action": "read", "checkpoint_id": "  ", "target_id": "tab"},
    {"action": "read", "checkpoint_id": "id", "target_id": "  "},
])
async def test_read_requires_nonempty_checkpoint_and_tab(archive: CheckpointStore, params: dict) -> None:
    with pytest.raises(ToolMessageError, match="checkpoint_id and target_id are required"):
        await _run(archive, params)


@pytest.mark.parametrize(("params", "field"), [
    ({"action": "search"}, "action"),
    ({"action": "list", "kind": "html"}, "kind"),
    ({"action": "list", "cursor": -1}, "cursor"),
    ({"action": "list", "limit": 0}, "limit"),
    ({"action": "list", "limit": 51}, "limit"),
    ({"action": "list", "offset": -1}, "offset"),
    ({"action": "list", "max_char": 499}, "max_char"),
    ({"action": "list", "max_char": 25001}, "max_char"),
    ({"action": "list", "target_id": None}, "target_id"),
])
async def test_schema_validates_even_unused_arguments(archive: CheckpointStore, params: dict, field: str) -> None:
    task = await CheckpointTool(archive).run(params)
    with pytest.raises(ToolMessageError, match=f"Error: {field}:"):
        await task.result()


async def test_validated_numeric_strings_are_supported(
    archive: CheckpointStore,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    checkpoint = await publish_checkpoint(archive)
    result = await _run(archive, {"action": "list", "cursor": "0", "limit": "1"})
    assert checkpoint.checkpoint_id in result.text
    result = await _run(archive, _read_args(checkpoint, offset="0", max_char="500"))
    assert "Feedback" in result.text


@pytest.mark.parametrize("kind", ["dom", "screenshot"])
async def test_foreign_checkpoints_are_not_readable(
    archive: CheckpointStore,
    tmp_path: Path,
    kind: str,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    other = CheckpointStore(str(tmp_path / "other"))
    checkpoint = await publish_checkpoint(other)
    try:
        with pytest.raises(ToolMessageError, match="Checkpoint not found"):
            await _run(archive, _read_args(checkpoint, kind=kind))
    finally:
        await other.close()


@pytest.mark.parametrize("kind", ["dom", "screenshot"])
async def test_missing_capture_reports_original_reason(archive: CheckpointStore, kind: str) -> None:
    tab = TabCapture("tab", screenshot=Artifact(unavailable_reason="Capture timed out"))
    checkpoint = await archive.publish(await archive.reserve(), "click", TransactionStatus.FAILED, (tab,))
    reason = "Capture timed out" if kind == "screenshot" else "Not captured"
    with pytest.raises(ToolMessageError, match=reason):
        await _run(archive, _read_args(checkpoint, kind=kind))


@pytest.mark.parametrize("kind", ["dom", "screenshot"])
async def test_deleted_artifact_is_explicit_and_never_substituted(
    archive: CheckpointStore,
    kind: str,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    checkpoint = await publish_checkpoint(archive)
    artifact = checkpoint.tabs[0].dom if kind == "dom" else checkpoint.tabs[0].screenshot
    Path(artifact.path).unlink()
    await publish_checkpoint(archive, "Replacement")
    with pytest.raises(ToolMessageError, match="unavailable"):
        await _run(archive, _read_args(checkpoint, kind=kind))


async def test_offset_past_saved_text_is_caller_error(
    archive: CheckpointStore,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    checkpoint = await publish_checkpoint(archive, "short")
    with pytest.raises(ToolMessageError, match="Offset is past"):
        await _run(archive, _read_args(checkpoint, offset=6))


async def test_bound_tool_cannot_read_after_run_closes(
    archive: CheckpointStore,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    checkpoint = await publish_checkpoint(archive)
    tool = CheckpointTool(archive)
    await archive.close()
    task = await tool.run(_read_args(checkpoint))
    with pytest.raises(ToolMessageError, match="store is closed"):
        await task.result()
