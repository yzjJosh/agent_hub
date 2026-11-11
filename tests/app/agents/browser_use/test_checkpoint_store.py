"""Tests for saved browser observations."""

import asyncio
from collections.abc import Awaitable, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from app.agents.browser_use.checkpoint_store import (
    Artifact,
    CaptureInterval,
    Checkpoint,
    CheckpointError,
    CheckpointStore,
    DomCapture,
    TabCapture,
)
from app.agents.browser_use.models import TransactionStatus


async def test_same_tab_url_has_distinct_immutable_observations(
    archive: CheckpointStore,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
    dom_time: CaptureInterval,
    image_time: CaptureInterval,
) -> None:
    feedback = await publish_checkpoint(archive)
    transcript = await publish_checkpoint(archive, "Transcript", image=b"second image")
    first = await archive.read_screenshot(feedback.checkpoint_id, "tab")
    second = await archive.read_screenshot(transcript.checkpoint_id, "tab")
    assert first.path != second.path
    assert Path(first.path).read_bytes() == b"original image"
    assert Path(second.path).read_bytes() == b"second image"
    assert (await archive.read_dom(feedback.checkpoint_id, "tab")).text == "Feedback"
    assert (await archive.read_dom(transcript.checkpoint_id, "tab")).text == "Transcript"
    assert feedback.tabs[0].dom.interval == dom_time
    assert feedback.tabs[0].screenshot.interval == image_time
    with pytest.raises(FrozenInstanceError):
        feedback.source_action = "click"


async def test_pending_observation_is_not_visible_and_discard_removes_files(archive: CheckpointStore) -> None:
    reservation = await archive.reserve()
    directory = Path(reservation.directory)
    directory.mkdir(parents=True)
    (directory / "01.jpg").write_bytes(b"unfinished")
    assert (await archive.list()).rows == ()
    with pytest.raises(CheckpointError, match="Checkpoint not found"):
        await archive.read_dom(reservation.checkpoint_id, "tab")
    await archive.discard(reservation)
    assert not directory.exists()
    assert (await archive.list()).rows == ()


@pytest.mark.parametrize("source_status", [TransactionStatus.SUCCESS, TransactionStatus.FAILED, TransactionStatus.STALE])
async def test_publication_preserves_outcome_and_missing_artifacts(
    archive: CheckpointStore,
    source_status: TransactionStatus,
    dom_time: CaptureInterval,
    image_time: CaptureInterval,
) -> None:
    reservation = await archive.reserve()
    missing = Artifact(interval=image_time, unavailable_reason="Screenshot timed out")
    tab = TabCapture("closed-tab", screenshot=missing, dom=DomCapture("", dom_time))
    record = await archive.publish(reservation, "dump_page", source_status, (tab,))
    assert record.status is source_status
    assert record.tabs[0].screenshot.unavailable_reason == "Screenshot timed out"
    assert record.tabs[0].dom.char_count == 0
    assert (await archive.read_dom(record.checkpoint_id, "closed-tab")).text == ""
    with pytest.raises(CheckpointError, match="Screenshot timed out"):
        await archive.read_screenshot(record.checkpoint_id, "closed-tab")


async def test_list_filter_pagination_and_empty_observation(
    archive: CheckpointStore,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    first = await publish_checkpoint(archive, "Feedback", "a")
    second = await publish_checkpoint(archive, "Transcript", "b")
    empty = await archive.publish(await archive.reserve(), "close_tab", TransactionStatus.SUCCESS, ())
    page = await archive.list(limit=2)
    assert [row.checkpoint.checkpoint_id for row in page.rows] == [first.checkpoint_id, second.checkpoint_id]
    assert page.next_cursor == 2
    last = await archive.list(cursor=2)
    assert last.rows[0].checkpoint == empty
    assert last.rows[0].tab is None
    assert last.next_cursor is None
    filtered = await archive.list(target_id="b", cursor=0, limit=1)
    assert filtered.rows[0].checkpoint == second
    assert filtered.next_cursor is None
    assert (await archive.list(target_id="b", cursor=1)).rows == ()
    assert page.rows[0].tab.dom.excerpt == "Feedback"


@pytest.mark.parametrize(
    ("offset", "size", "expected", "next_offset"),
    [(0, 2, "A\r", 2), (2, 3, "\n中🙂", 5), (5, 10, "Z", None), (6, 10, "", None)],
)
async def test_dom_windows_are_exact_unicode_slices(
    archive: CheckpointStore,
    offset: int,
    size: int,
    expected: str,
    next_offset: int | None,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    record = await publish_checkpoint(archive, "A\r\n中🙂Z")
    window = await archive.read_dom(record.checkpoint_id, "tab", offset, size)
    assert window.text == expected
    assert window.total == 6
    assert window.offset == offset
    assert window.next_offset == next_offset


@pytest.mark.parametrize("offset", [-1, 9])
async def test_dom_window_rejects_invalid_offset(
    archive: CheckpointStore,
    offset: int,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    record = await publish_checkpoint(archive, "short")
    with pytest.raises(ValueError, match="Offset"):
        await archive.read_dom(record.checkpoint_id, "tab", offset)


@pytest.mark.parametrize("kind", ["dom", "screenshot"])
async def test_deleted_artifact_is_unavailable(
    archive: CheckpointStore,
    kind: str,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    record = await publish_checkpoint(archive)
    artifact = record.tabs[0].dom if kind == "dom" else record.tabs[0].screenshot
    Path(artifact.path).unlink()
    with pytest.raises(CheckpointError, match="unavailable"):
        if kind == "dom":
            await archive.read_dom(record.checkpoint_id, "tab")
        else:
            await archive.read_screenshot(record.checkpoint_id, "tab")
    assert (await archive.list()).rows[0].checkpoint == record


async def test_foreign_ids_and_reservations_are_rejected(
    archive: CheckpointStore,
    tmp_path: Path,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    foreign = CheckpointStore(str(tmp_path / "other"), resources_root=str(tmp_path))
    reservation = await foreign.reserve()
    record = await publish_checkpoint(foreign)
    try:
        with pytest.raises(CheckpointError, match="Checkpoint not found"):
            await archive.read_dom(record.checkpoint_id, "tab")
        with pytest.raises(CheckpointError, match="reservation is unavailable"):
            await archive.publish(reservation, "click", TransactionStatus.SUCCESS, ())
    finally:
        await foreign.close()


async def test_publication_rejects_artifact_outside_reservation(
    archive: CheckpointStore,
    tmp_path: Path,
    image_time: CaptureInterval,
) -> None:
    reservation = await archive.reserve()
    tab = TabCapture("tab", screenshot=Artifact(str(tmp_path / "outside.jpg"), image_time, ""))
    with pytest.raises(CheckpointError, match="outside its checkpoint"):
        await archive.publish(reservation, "click", TransactionStatus.SUCCESS, (tab,))
    assert (await archive.list()).rows == ()
    assert not Path(reservation.directory).exists()


async def test_failed_publication_cleans_files(
    archive: CheckpointStore,
    monkeypatch: pytest.MonkeyPatch,
    dom_time: CaptureInterval,
) -> None:
    reservation = await archive.reserve()
    tab = TabCapture("tab", dom=DomCapture("Feedback", dom_time))
    monkeypatch.setattr("aiofiles.open", Mock(side_effect=PermissionError("denied")))
    with pytest.raises(PermissionError, match="denied"):
        await archive.publish(reservation, "dump_page", TransactionStatus.SUCCESS, (tab,))
    assert (await archive.list()).rows == ()
    assert not Path(reservation.directory).exists()


async def test_close_disables_lookup_but_preserves_published_files(
    archive: CheckpointStore,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    record = await publish_checkpoint(archive)
    pending = await archive.reserve()
    Path(pending.directory).mkdir(parents=True)
    await archive.close()
    assert Path(record.tabs[0].screenshot.path).read_bytes() == b"original image"
    assert not Path(pending.directory).exists()
    with pytest.raises(CheckpointError, match="closed"):
        await archive.list()
    with pytest.raises(CheckpointError, match="closed"):
        await archive.reserve()


async def test_cancelled_publication_never_exposes_partial_record(
    archive: CheckpointStore,
    monkeypatch: pytest.MonkeyPatch,
    dom_time: CaptureInterval,
) -> None:
    reservation = await archive.reserve()
    started = asyncio.Event()
    blocked = asyncio.Event()
    async def block_write(_text: str) -> None:
        """Wait until the write is released."""
        started.set()
        await blocked.wait()

    @asynccontextmanager
    async def open_blocked(_path: str, _mode: str, **_kwargs: str) -> AsyncIterator[AsyncMock]:
        """Provide a file whose write stays unfinished."""
        stream = AsyncMock()
        stream.write.side_effect = block_write
        yield stream

    monkeypatch.setattr("aiofiles.open", open_blocked)
    task = asyncio.create_task(archive.publish(
        reservation, "dump_page", TransactionStatus.SUCCESS, (TabCapture("tab", dom=DomCapture("body", dom_time)),)
    ))
    await started.wait()
    assert (await archive.list()).rows == ()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await archive.list()).rows == ()
    assert not Path(reservation.directory).exists()


@pytest.mark.parametrize("kind", ["dom", "screenshot"])
async def test_unreadable_artifact_reports_unavailable(
    archive: CheckpointStore,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    record = await publish_checkpoint(archive)
    monkeypatch.setattr("aiofiles.open", Mock(side_effect=PermissionError("denied")))
    with pytest.raises(CheckpointError, match="unavailable"):
        if kind == "dom":
            await archive.read_dom(record.checkpoint_id, "tab")
        else:
            await archive.read_screenshot(record.checkpoint_id, "tab")


async def test_missing_dom_is_different_from_empty_dom(archive: CheckpointStore) -> None:
    record = await archive.publish(await archive.reserve(), "click", TransactionStatus.SUCCESS, (TabCapture("tab"),))
    assert record.tabs[0].dom.path is None
    assert record.tabs[0].dom.char_count is None
    with pytest.raises(CheckpointError, match="Not captured"):
        await archive.read_dom(record.checkpoint_id, "tab")


async def test_excerpt_is_bounded_without_truncating_saved_dom(
    archive: CheckpointStore,
    publish_checkpoint: Callable[..., Awaitable[Checkpoint]],
) -> None:
    text = "evidence " * 100
    record = await publish_checkpoint(archive, text)
    assert record.tabs[0].dom.excerpt == text[:160]
    assert record.tabs[0].dom.char_count == len(text)
    assert (await archive.read_dom(record.checkpoint_id, "tab")).text == text


async def test_republishing_cannot_replace_original_evidence(
    archive: CheckpointStore,
    dom_time: CaptureInterval,
) -> None:
    reservation = await archive.reserve()
    record = await archive.publish(
        reservation, "dump_page", TransactionStatus.SUCCESS, (TabCapture("tab", dom=DomCapture("first", dom_time)),)
    )
    with pytest.raises(CheckpointError, match="reservation is unavailable"):
        await archive.publish(
            reservation, "dump_page", TransactionStatus.SUCCESS,
            (TabCapture("tab", dom=DomCapture("second", dom_time)),),
        )
    assert (await archive.read_dom(record.checkpoint_id, "tab")).text == "first"
