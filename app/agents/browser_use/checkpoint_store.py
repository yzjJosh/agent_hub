"""Saved browser observations for one run."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import aiofiles

from app.agents.browser_use.models import TransactionStatus
from app.lib.shared import activity

__all__ = [
    "Artifact", "CaptureInterval", "Checkpoint", "CheckpointError", "CheckpointPage",
    "CheckpointReservation", "CheckpointRow", "CheckpointStore", "CheckpointTab",
    "DomCapture", "DomWindow", "ScreenshotRead", "TabCapture",
]

_EXCERPT_LENGTH = 160
_logger = logging.getLogger(__name__)


class CheckpointError(Exception):
    """A saved observation could not be used."""


@dataclass(frozen=True)
class CaptureInterval:
    """When one artifact was captured."""

    started_at: datetime
    finished_at: datetime


@dataclass(frozen=True)
class Artifact:
    """A captured file or the reason it is absent."""

    path: str | None = None
    interval: CaptureInterval | None = None
    unavailable_reason: str = "Not captured"
    char_count: int | None = None
    excerpt: str = ""


@dataclass(frozen=True)
class DomCapture:
    """Text captured from one tab."""

    text: str
    interval: CaptureInterval


@dataclass(frozen=True)
class TabCapture:
    """Artifacts collected from one tab."""

    target_id: str
    title: str = ""
    url: str = ""
    screenshot: Artifact = Artifact()
    dom: DomCapture | None = None
    dom_unavailable_reason: str = "Not captured"


@dataclass(frozen=True)
class CheckpointReservation:
    """The identity and folder of an unfinished observation."""

    checkpoint_id: str
    directory: str
    order: int
    created_at: datetime


@dataclass(frozen=True)
class CheckpointTab:
    """Saved artifact metadata for one tab."""

    target_id: str
    title: str
    url: str
    screenshot: Artifact
    dom: Artifact


@dataclass(frozen=True)
class Checkpoint:
    """One completed browser observation."""

    checkpoint_id: str
    order: int
    created_at: datetime
    source_action: str
    status: TransactionStatus
    tabs: tuple[CheckpointTab, ...]


@dataclass(frozen=True)
class CheckpointRow:
    """One tab in an observation listing."""

    checkpoint: Checkpoint
    tab: CheckpointTab | None


@dataclass(frozen=True)
class CheckpointPage:
    """A page of saved observation metadata."""

    rows: tuple[CheckpointRow, ...]
    next_cursor: int | None


@dataclass(frozen=True)
class DomWindow:
    """One exact window of saved page text."""

    checkpoint: Checkpoint
    tab: CheckpointTab
    text: str
    offset: int
    total: int
    next_offset: int | None


@dataclass(frozen=True)
class ScreenshotRead:
    """One original saved tab image."""

    checkpoint: Checkpoint
    tab: CheckpointTab
    path: str


def _remove_directory(path: str) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError:
        _logger.exception("Cannot remove unfinished checkpoint %s", path)


def _validate_path(directory: str, path: str) -> None:
    root = os.path.realpath(directory)
    target = os.path.realpath(path)
    if os.path.commonpath((root, target)) != root or target == root:
        raise CheckpointError("Artifact is outside its checkpoint")


class CheckpointStore:
    """Keep immutable browser evidence for one active run."""

    def __init__(self, run_dir: str, resources_root: str | None = None) -> None:
        """Create an empty store.

        Args:
            run_dir: Resource folder belonging to this browser run.
            resources_root: Resource root used to record file use.
        """
        self._directory = os.path.join(run_dir, "checkpoints")
        self._resources_root = resources_root
        self._pending: dict[str, CheckpointReservation] = {}
        self._records: dict[str, Checkpoint] = {}
        self._order = 0
        self._is_closed = False
        self._lock = asyncio.Lock()

    async def reserve(self) -> CheckpointReservation:
        """Reserve an observation and its file paths.

        Returns:
            A fresh identity whose folder can receive capture files.

        Raises:
            CheckpointError: If the store is closed.
        """
        async with self._lock:
            self._ensure_open()
            self._order += 1
            checkpoint_id = uuid4().hex
            reservation = CheckpointReservation(
                checkpoint_id, os.path.join(self._directory, checkpoint_id), self._order, datetime.now(UTC)
            )
            self._pending[checkpoint_id] = reservation
            return reservation

    async def publish(
        self,
        reservation: CheckpointReservation,
        source_action: str,
        status: TransactionStatus,
        tabs: tuple[TabCapture, ...],
    ) -> Checkpoint:
        """Publish all artifacts of a finished observation together.

        Args:
            reservation: The unfinished observation owned by this store.
            source_action: Name of the browser action.
            status: Actual outcome of the action.
            tabs: Captured tabs in their original order.

        Returns:
            Immutable metadata available to later reads.

        Raises:
            CheckpointError: If the reservation or artifact paths are invalid.
            OSError: If an artifact cannot be saved.
        """
        self._ensure_pending(reservation)
        try:
            await asyncio.to_thread(os.makedirs, reservation.directory, exist_ok=True)
            saved_tabs = await self._save_tabs(reservation, tabs)
            record = Checkpoint(
                reservation.checkpoint_id, reservation.order, reservation.created_at,
                source_action, status, saved_tabs,
            )
            async with self._lock:
                self._ensure_pending(reservation)
                self._records[record.checkpoint_id] = record
                del self._pending[record.checkpoint_id]
            return record
        except BaseException:
            await self.discard(reservation)
            raise

    async def discard(self, reservation: CheckpointReservation) -> None:
        """Remove an unfinished observation owned by this store.

        Args:
            reservation: The unfinished observation to remove.
        """
        async with self._lock:
            if self._pending.get(reservation.checkpoint_id) is not reservation:
                return
            del self._pending[reservation.checkpoint_id]
        await asyncio.to_thread(_remove_directory, reservation.directory)

    async def list(self, target_id: str = "", cursor: int = 0, limit: int = 10) -> CheckpointPage:
        """List saved tabs in observation order.

        Args:
            target_id: Restrict rows to this tab when supplied.
            cursor: Row offset after the tab filter.
            limit: Maximum rows to return.

        Returns:
            Metadata rows and the next offset when more rows exist.

        Raises:
            CheckpointError: If the store is closed.
            ValueError: If the offset or limit is invalid.
        """
        if cursor < 0 or limit < 1:
            raise ValueError("Cursor must be nonnegative and limit must be positive")
        async with self._lock:
            self._ensure_open()
            rows = self._list_rows(target_id)
            end = cursor + limit
            return CheckpointPage(tuple(rows[cursor:end]), end if end < len(rows) else None)

    async def read_dom(
        self, checkpoint_id: str, target_id: str, offset: int = 0, max_char: int = 25000
    ) -> DomWindow:
        """Read an exact window of captured page text.

        Args:
            checkpoint_id: Identity of a saved observation.
            target_id: Tab that supplied the text.
            offset: First character to return.
            max_char: Maximum characters to return.

        Returns:
            Text and its original capture metadata.

        Raises:
            CheckpointError: If the observation or text is unavailable.
            ValueError: If the requested window is invalid.
        """
        if offset < 0 or max_char < 1:
            raise ValueError("Offset must be nonnegative and max_char must be positive")
        record, tab = await self._find_tab(checkpoint_id, target_id)
        text = await self._read_text(tab.dom)
        if offset > len(text):
            raise ValueError("Offset is past the captured DOM")
        end = min(offset + max_char, len(text))
        return DomWindow(record, tab, text[offset:end], offset, len(text), end if end < len(text) else None)

    async def read_screenshot(self, checkpoint_id: str, target_id: str) -> ScreenshotRead:
        """Get the original saved image for one tab.

        Args:
            checkpoint_id: Identity of a saved observation.
            target_id: Tab that supplied the image.

        Returns:
            The image path and its original capture metadata.

        Raises:
            CheckpointError: If the observation or image is unavailable.
        """
        record, tab = await self._find_tab(checkpoint_id, target_id)
        path = self._artifact_path(tab.screenshot)
        try:
            async with aiofiles.open(path, "rb") as stream:
                await stream.read(1)
            await activity.record(path, root=self._resources_root)
        except OSError as exc:
            _logger.exception("Cannot read checkpoint screenshot %s", path)
            raise CheckpointError("Saved screenshot is unavailable") from exc
        return ScreenshotRead(record, tab, path)

    async def close(self) -> None:
        """Close lookup and remove unfinished captures.

        Published files remain available under resource retention.
        """
        async with self._lock:
            self._is_closed = True
            pending = tuple(self._pending.values())
            self._pending.clear()
            self._records.clear()
        await asyncio.gather(*(asyncio.to_thread(_remove_directory, item.directory) for item in pending))

    def _ensure_open(self) -> None:
        if self._is_closed:
            raise CheckpointError("Checkpoint store is closed")

    def _ensure_pending(self, reservation: CheckpointReservation) -> None:
        self._ensure_open()
        if self._pending.get(reservation.checkpoint_id) is not reservation:
            raise CheckpointError("Checkpoint reservation is unavailable")

    async def _save_tabs(
        self, reservation: CheckpointReservation, tabs: tuple[TabCapture, ...]
    ) -> tuple[CheckpointTab, ...]:
        if len({tab.target_id for tab in tabs}) != len(tabs):
            raise CheckpointError("Checkpoint tab IDs must be unique")
        saved = []
        for index, tab in enumerate(tabs):
            if tab.screenshot.path is not None:
                await asyncio.to_thread(_validate_path, reservation.directory, tab.screenshot.path)
            dom = await self._save_dom(reservation, index, tab.dom, tab.dom_unavailable_reason)
            saved.append(CheckpointTab(tab.target_id, tab.title, tab.url, tab.screenshot, dom))
        return tuple(saved)

    async def _save_dom(
        self, reservation: CheckpointReservation, index: int, capture: DomCapture | None, unavailable_reason: str
    ) -> Artifact:
        if capture is None:
            return Artifact(unavailable_reason=unavailable_reason)
        path = os.path.join(reservation.directory, f"dom_{index:02d}.md")
        async with aiofiles.open(path, "x", encoding="utf-8", newline="") as stream:
            await stream.write(capture.text)
        return Artifact(path, capture.interval, "", len(capture.text), capture.text[:_EXCERPT_LENGTH])

    def _list_rows(self, target_id: str) -> list[CheckpointRow]:
        rows = []
        for record in sorted(self._records.values(), key=lambda item: item.order):
            rows.extend(CheckpointRow(record, tab) for tab in record.tabs if not target_id or tab.target_id == target_id)
            if not record.tabs and not target_id:
                rows.append(CheckpointRow(record, None))
        return rows

    async def _find_tab(self, checkpoint_id: str, target_id: str) -> tuple[Checkpoint, CheckpointTab]:
        async with self._lock:
            self._ensure_open()
            record = self._records.get(checkpoint_id)
            if record is None:
                raise CheckpointError("Checkpoint not found")
            for tab in record.tabs:
                if tab.target_id == target_id:
                    return record, tab
            raise CheckpointError("Checkpoint tab not found")

    def _artifact_path(self, artifact: Artifact) -> str:
        if artifact.path is None:
            raise CheckpointError(artifact.unavailable_reason)
        return artifact.path

    async def _read_text(self, artifact: Artifact) -> str:
        path = self._artifact_path(artifact)
        try:
            async with aiofiles.open(path, "r", encoding="utf-8", newline="") as stream:
                text = await stream.read()
            await activity.record(path, root=self._resources_root)
            return text
        except (OSError, UnicodeError) as exc:
            _logger.exception("Cannot read checkpoint DOM %s", path)
            raise CheckpointError("Saved DOM is unavailable") from exc
