"""Browser session that executes one browser action step at a time."""

from __future__ import annotations

import asyncio
import os
import re
import sys
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import uuid4

import aiofiles

from app.agents.browser_use.checkpoint_store import (
    Artifact,
    CaptureInterval,
    CheckpointReservation,
    CheckpointStore,
    TabCapture,
)
from app.agents.browser_use.models import Snapshot, TabSnapshot, TransactionResult, TransactionStatus
from app.agents.browser_use.similarity import screen_similarity
from app.lib.browser.browser_manager import BrowserManager

__all__ = ["Action", "BrowserSession"]

Action = Callable[[BrowserManager, str | None], Awaitable[bool]]

_IS_VERBOSE = True
_CAPTURE_TIMEOUT_SECONDS = 60
_DOWNLOAD_TIMEOUT_SECONDS = 30
_MAX_TITLE_LENGTH = 120
_MAX_URL_LENGTH = 300
_MIN_SCREEN_SIMILARITY = 0.70
_MAX_CONSECUTIVE_STALE_SKIPS = 3
_CONTROL_CHARACTERS_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_SPACE_RUN_PATTERN = re.compile(r"\s+")
_CHECKPOINT_UNAVAILABLE_MESSAGE = (
    "Checkpoint unavailable, do not repeat an executed action to repair the archive."
)


@dataclass(frozen=True)
class _ActionOutcome:
    """Result of executing one browser action."""

    has_succeeded: bool
    error_message: str = ""
    downloaded_files: tuple[str, ...] = ()


def _log_info(method_name: str, message: str) -> None:
    """Write a progress log line for a session method when verbose logging is on."""
    if _IS_VERBOSE:
        print(f"[BrowserSession.{method_name}] {message}", flush=True)


def _log_error(method_name: str, message: str, *, should_include_traceback: bool = False) -> None:
    """Write an error log line for a session method to stderr."""
    suffix = f"\n{traceback.format_exc()}" if should_include_traceback else ""
    print(f"[BrowserSession.{method_name}] Error: {message}{suffix}", file=sys.stderr)


async def _ensure_directory(directory: str, method_name: str) -> None:
    """Create a folder if needed and log an error when creation fails."""
    try:
        await asyncio.to_thread(os.makedirs, directory, exist_ok=True)
    except OSError:
        _log_error(method_name, f"cannot create folder {directory}", should_include_traceback=True)


def _resolve_target_tab(manager: BrowserManager, target_id: str | None, *, is_new_tab: bool) -> str | None:
    """Find the tab an action runs on, either named or the only open tab."""
    named_tab = (target_id or "").strip()
    if named_tab:
        return named_tab
    if is_new_tab:
        return None
    open_tabs = manager.list_tab_ids()
    if len(open_tabs) > 1:
        raise RuntimeError(f"{len(open_tabs)} tabs are open, give the target id of the one to act on")
    return open_tabs[0] if open_tabs else None


def _sanitize_caption_text(
    raw_text: str | None,
    max_length: int,
    should_strip_quotes: bool = False,
) -> str:
    """Flatten page text into one bounded plain line."""
    cleaned_text = _CONTROL_CHARACTERS_PATTERN.sub(" ", raw_text or "")
    cleaned_text = _SPACE_RUN_PATTERN.sub(" ", cleaned_text).strip()
    if should_strip_quotes:
        cleaned_text = cleaned_text.replace('"', "")
    if len(cleaned_text) > max_length:
        cleaned_text = cleaned_text[: max_length - 1] + "…"
    return cleaned_text


def _create_download_placeholder_sync(directory: str) -> str:
    """Create an empty download placeholder file in the folder and return its path."""
    os.makedirs(directory, exist_ok=True)
    placeholder_path = os.path.join(directory, "download")
    file_descriptor = os.open(placeholder_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.close(file_descriptor)
    return placeholder_path


async def _read_file_bytes_if_present(path: str) -> bytes:
    """Read a file as bytes, or return empty bytes when the file cannot be read."""
    try:
        async with aiofiles.open(path, "rb") as file_handle:
            return await file_handle.read()
    except OSError:
        return b""


def _remove_file_if_present_sync(path: str) -> None:
    """Remove a file when it exists."""
    try:
        os.remove(path)
    except OSError:
        pass


class BrowserSession:
    """Drive one browser by tab id, running one action step at a time."""

    def __init__(self, downloads_dir: str, *, checkpoint_store: CheckpointStore | None = None) -> None:
        """Create a closed browser session configured with a downloads folder.

        Args:
            downloads_dir: Folder where step downloads and unarchived files are saved.
            checkpoint_store: Existing checkpoint store owned by the caller when supplied.
        """
        self._downloads_dir = downloads_dir
        self._manager: BrowserManager | None = None
        self._lock = asyncio.Lock()
        self._step_id: int = 0
        self._last_snapshot: Snapshot | None = None
        self._consecutive_stale_skips: int = 0
        self._downloads: list[str] = []
        self._store = checkpoint_store if checkpoint_store is not None else CheckpointStore(downloads_dir)
        self._owns_store = checkpoint_store is None

    def _require_manager(self) -> BrowserManager:
        """Return the open browser manager, or raise when the session is closed."""
        if self._manager is None:
            raise RuntimeError("session is not open")
        return self._manager

    def _get_step_directory(self) -> str:
        """Return the folder where files for the current step are stored."""
        return os.path.join(self._downloads_dir, f"step_{self._step_id}")

    async def open(self) -> None:
        """Connect this session to a browser instance.

        Raises:
            RuntimeError: If the session is already open.
        """
        if self._manager is not None:
            raise RuntimeError("session is already open")
        self._manager = await BrowserManager.create(download_dir=self._downloads_dir)

    async def close(self) -> None:
        """Release the browser manager and owned checkpoint store. A second call does nothing."""
        try:
            if self._manager is not None:
                await self._manager.cleanup()
                self._manager = None
        finally:
            if self._owns_store:
                await self._store.close()

    async def execute_step(
        self,
        action: Action,
        target_id: str | None,
        *,
        is_new_tab: bool = False,
        should_monitor_download: bool = False,
        source_action: str = "action",
        capture_dom: Callable[[], TabCapture | None] | None = None,
    ) -> TransactionResult:
        """Run one browser action and capture an updated snapshot of all open tabs.

        Args:
            action: Async action that takes the browser manager and resolved tab id.
            target_id: Target tab id, or blank to use the only open tab.
            is_new_tab: Pass no tab when True so the action can open a new tab.
            should_monitor_download: Watch the target tab for a downloaded file when True.
            source_action: Action name saved on the published checkpoint.
            capture_dom: Callback that returns any page text captured by the action.

        Returns:
            The step outcome, the updated tab snapshot, and any downloaded file paths.

        Raises:
            RuntimeError: If the session is not open.
            KeyError: If download monitoring targets a tab that is not open.
        """
        async with self._lock:
            manager = self._require_manager()
            step_number = self._step_id
            _log_info(
                "execute_step",
                f"step {step_number} start, action={action!r}, target_id={target_id!r}, "
                f"is_new_tab={is_new_tab}, should_monitor_download={should_monitor_download}",
            )
            return await self._execute_locked_step(
                manager,
                action,
                target_id,
                is_new_tab=is_new_tab,
                should_monitor_download=should_monitor_download,
                source_action=source_action,
                capture_dom=capture_dom,
            )

    async def _execute_locked_step(
        self,
        manager: BrowserManager,
        action: Action,
        target_id: str | None,
        *,
        is_new_tab: bool,
        should_monitor_download: bool,
        source_action: str,
        capture_dom: Callable[[], TabCapture | None] | None,
    ) -> TransactionResult:
        """Execute one step while holding the session lock."""
        try:
            resolved_tab = _resolve_target_tab(manager, target_id, is_new_tab=is_new_tab)
        except Exception as error:
            _log_error("execute_step", f"no tab to act on, {error}", should_include_traceback=True)
            self._consecutive_stale_skips = 0
            outcome = _ActionOutcome(has_succeeded=False, error_message=f"action failed: {error}")
        else:
            stale_check_tab = None if is_new_tab else resolved_tab
            if await self._should_skip_stale_action(manager, stale_check_tab):
                return await self._record_observation(
                    source_action,
                    TransactionStatus.STALE,
                    "screen changed since you last saw it, action not run",
                    (),
                    None,
                )
            outcome = await self._execute_action(
                manager,
                action,
                resolved_tab,
                should_monitor_download=should_monitor_download,
            )
        status = TransactionStatus.SUCCESS if outcome.has_succeeded else TransactionStatus.FAILED
        message = "" if outcome.has_succeeded else outcome.error_message
        dom_capture = capture_dom() if capture_dom is not None else None
        result = await self._record_observation(
            source_action,
            status,
            message,
            outcome.downloaded_files,
            dom_capture,
        )
        self._step_id += 1
        return result

    async def _record_observation(
        self,
        source_action: str,
        status: TransactionStatus,
        message: str,
        downloaded_files: tuple[str, ...],
        dom_capture: TabCapture | None,
    ) -> TransactionResult:
        """Capture all open tabs, publish a checkpoint when available, and return the result."""
        reservation: CheckpointReservation | None = None
        archive_error = ""
        try:
            reservation = await self._store.reserve()
            output_directory = reservation.directory
        except Exception:
            _log_error("_record_observation", "cannot reserve checkpoint", should_include_traceback=True)
            output_directory = os.path.join(self._downloads_dir, "unarchived", uuid4().hex)
            archive_error = _CHECKPOINT_UNAVAILABLE_MESSAGE
        try:
            snapshot = await self._capture_all_tabs(output_directory)
            if reservation is not None:
                snapshot = await self._publish_checkpoint(
                    snapshot,
                    reservation,
                    source_action,
                    status,
                    dom_capture,
                )
            else:
                snapshot = replace(snapshot, archive_error=archive_error)
        except BaseException:
            if reservation is not None:
                await self._store.discard(reservation)
            raise
        self._last_snapshot = snapshot
        return TransactionResult(status, message, snapshot, downloaded_files)

    async def _publish_checkpoint(
        self,
        snapshot: Snapshot,
        reservation: CheckpointReservation,
        source_action: str,
        status: TransactionStatus,
        dom_capture: TabCapture | None,
    ) -> Snapshot:
        """Publish captured tab data to the checkpoint store."""
        tab_captures = self._build_tab_captures(snapshot, dom_capture)
        try:
            record = await self._store.publish(reservation, source_action, status, tab_captures)
        except Exception:
            _log_error("_publish_checkpoint", "cannot publish checkpoint", should_include_traceback=True)
            return replace(snapshot, archive_error=_CHECKPOINT_UNAVAILABLE_MESSAGE)
        return replace(snapshot, checkpoint_id=record.checkpoint_id, observed_at=record.created_at)

    def _build_tab_captures(self, snapshot: Snapshot, dom_capture: TabCapture | None) -> tuple[TabCapture, ...]:
        """Convert snapshot tabs and an optional DOM capture into checkpoint tab captures."""
        sanitized_dom: TabCapture | None = None
        if dom_capture is not None:
            sanitized_dom = replace(
                dom_capture,
                title=_sanitize_caption_text(dom_capture.title, _MAX_TITLE_LENGTH, should_strip_quotes=True),
                url=_sanitize_caption_text(dom_capture.url, _MAX_URL_LENGTH),
            )
        tab_captures: list[TabCapture] = []
        for tab_snapshot in snapshot.tabs.values():
            interval: CaptureInterval | None = None
            if tab_snapshot.capture_started_at is not None and tab_snapshot.capture_finished_at is not None:
                interval = CaptureInterval(tab_snapshot.capture_started_at, tab_snapshot.capture_finished_at)
            tab_capture = TabCapture(
                tab_snapshot.target_id,
                tab_snapshot.title,
                tab_snapshot.url,
                Artifact(tab_snapshot.screenshot_path, interval, tab_snapshot.screenshot_error),
            )
            if sanitized_dom is not None and sanitized_dom.target_id == tab_snapshot.target_id:
                tab_capture = replace(
                    tab_capture,
                    dom=sanitized_dom.dom,
                    dom_unavailable_reason=sanitized_dom.dom_unavailable_reason,
                )
            tab_captures.append(tab_capture)
        if sanitized_dom is not None and sanitized_dom.target_id not in snapshot.tabs:
            tab_captures.append(sanitized_dom)
        return tuple(tab_captures)

    async def get_downloaded_files(self) -> list[str]:
        """Return the downloaded file paths recorded in this session in order.

        Returns:
            List of downloaded file paths.

        Raises:
            RuntimeError: If the session is not open.
        """
        self._require_manager()
        return list(self._downloads)

    async def _execute_action(
        self,
        manager: BrowserManager,
        action: Action,
        target_id: str | None,
        *,
        should_monitor_download: bool,
    ) -> _ActionOutcome:
        """Run the action directly or while monitoring for a download."""
        _log_info("_execute_action", f"tab {target_id} running {action!r}")
        if should_monitor_download:
            outcome = await self._execute_download_action(manager, target_id, action)
            _log_info(
                "_execute_action",
                f"tab {target_id} download action returned {outcome.has_succeeded}, "
                f"files={outcome.downloaded_files}, error={outcome.error_message!r}",
            )
            return outcome
        try:
            has_succeeded = await action(manager, target_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _log_error("_execute_action", f"tab {target_id} action raised, {error}", should_include_traceback=True)
            return _ActionOutcome(has_succeeded=False, error_message=f"action failed: {error}")
        _log_info("_execute_action", f"tab {target_id} action returned {has_succeeded}")
        return _ActionOutcome(
            has_succeeded=has_succeeded,
            error_message="" if has_succeeded else "action failed",
        )

    async def _execute_download_action(
        self,
        manager: BrowserManager,
        target_id: str | None,
        action: Action,
    ) -> _ActionOutcome:
        """Run an action on a tab, wait for its download, and record the saved file."""
        if target_id is None:
            _log_error("_execute_download_action", "no tab is open to watch for a download")
            return _ActionOutcome(
                has_succeeded=False,
                error_message="action failed: no tab is open, open one with open_url first",
            )
        placeholder_path = await asyncio.to_thread(_create_download_placeholder_sync, self._get_step_directory())
        _log_info("_execute_download_action", f"tab {target_id} reserved {placeholder_path} for download")
        try:
            download_result = await manager.wait_for_download(
                target_id,
                action=lambda: action(manager, target_id),
                save_path=placeholder_path,
                timeout=_DOWNLOAD_TIMEOUT_SECONDS,
            )
        except Exception as error:
            _log_error(
                "_execute_download_action",
                f"tab {target_id} download raised, {error}",
                should_include_traceback=True,
            )
            await asyncio.to_thread(_remove_file_if_present_sync, placeholder_path)
            raise
        if download_result.get("downloaded"):
            _log_info("_execute_download_action", f"tab {target_id} saved download at {placeholder_path}")
            self._downloads.append(placeholder_path)
            return _ActionOutcome(has_succeeded=True, downloaded_files=(placeholder_path,))
        _log_error(
            "_execute_download_action",
            f"tab {target_id} received no download within {_DOWNLOAD_TIMEOUT_SECONDS}s",
        )
        await asyncio.to_thread(_remove_file_if_present_sync, placeholder_path)
        return _ActionOutcome(has_succeeded=False, error_message="no download came")

    async def _should_skip_stale_action(self, manager: BrowserManager, target_id: str | None) -> bool:
        """Return True when the target tab changed since the last snapshot and the skip limit is not exceeded."""
        last_snapshot = self._last_snapshot
        if last_snapshot is None:
            _log_info("_should_skip_stale_action", "no earlier snapshot available, stale check skipped")
            self._consecutive_stale_skips = 0
            return False
        if await self._has_target_screen_changed(manager, last_snapshot, target_id):
            self._consecutive_stale_skips += 1
            if self._consecutive_stale_skips <= _MAX_CONSECUTIVE_STALE_SKIPS:
                _log_info(
                    "_should_skip_stale_action",
                    f"action skipped, {self._consecutive_stale_skips} of {_MAX_CONSECUTIVE_STALE_SKIPS} consecutive skips",
                )
                return True
            _log_info(
                "_should_skip_stale_action",
                f"{self._consecutive_stale_skips} consecutive skips exceeded limit "
                f"of {_MAX_CONSECUTIVE_STALE_SKIPS}, running action anyway",
            )
        self._consecutive_stale_skips = 0
        return False

    async def _has_target_screen_changed(
        self,
        manager: BrowserManager,
        last_snapshot: Snapshot,
        target_id: str | None,
    ) -> bool:
        """Check whether the target tab looks different from the last snapshot."""
        if target_id is None:
            _log_info("_has_target_screen_changed", "no tab to compare, stale check skipped")
            return False
        if target_id not in manager.list_tab_ids():
            _log_info("_has_target_screen_changed", f"tab {target_id} is no longer open, stale check skipped")
            return False
        previous_tab = last_snapshot.tabs.get(target_id)
        if previous_tab is None:
            _log_info("_has_target_screen_changed", f"tab {target_id} is not in the last snapshot, stale check skipped")
            return False
        if previous_tab.screenshot_path is None:
            _log_info("_has_target_screen_changed", f"tab {target_id} has no saved screenshot, stale check skipped")
            return False
        return await self._compare_tab_screenshot(manager, target_id, previous_tab.screenshot_path)

    async def _compare_tab_screenshot(
        self,
        manager: BrowserManager,
        target_id: str,
        previous_screenshot_path: str,
    ) -> bool:
        """Capture a fresh screenshot of the tab and compare it against the saved image."""
        step_directory = self._get_step_directory()
        await _ensure_directory(step_directory, "_compare_tab_screenshot")
        comparison_path = os.path.join(step_directory, f"stale_check_{target_id}.jpg")
        if await self._capture_tab_screenshot(manager, target_id, comparison_path) is None:
            _log_error("_compare_tab_screenshot", f"tab {target_id} comparison capture failed, stale check skipped")
            return False
        fresh_bytes = await _read_file_bytes_if_present(comparison_path)
        stored_bytes = await _read_file_bytes_if_present(previous_screenshot_path)
        await asyncio.to_thread(_remove_file_if_present_sync, comparison_path)
        similarity = await screen_similarity(stored_bytes, fresh_bytes)
        has_changed = similarity < _MIN_SCREEN_SIMILARITY
        decision = "skipped" if has_changed else "allowed"
        _log_info(
            "_compare_tab_screenshot",
            f"tab {target_id} similarity {similarity:.4f} against {_MIN_SCREEN_SIMILARITY}, action {decision}",
        )
        return has_changed

    async def _capture_all_tabs(self, output_directory: str) -> Snapshot:
        """Capture screenshots and metadata for every open tab in tab order."""
        observed_at = datetime.now(UTC)
        manager = self._require_manager()
        await _ensure_directory(output_directory, "_capture_all_tabs")
        target_ids = manager.list_tab_ids()
        _log_info("_capture_all_tabs", f"capturing {len(target_ids)} tabs into {output_directory}")
        tab_captures = await asyncio.gather(
            *(
                self._capture_tab_snapshot(manager, index, target_id, output_directory)
                for index, target_id in enumerate(target_ids, start=1)
            )
        )
        tabs: dict[str, TabSnapshot] = {}
        image_ordinal = 0
        for index, (target_id, tab_capture) in enumerate(zip(target_ids, tab_captures, strict=True), start=1):
            ordinal: int | None = None
            if tab_capture.screenshot_path is not None:
                image_ordinal += 1
                ordinal = image_ordinal
            tabs[target_id] = replace(
                tab_capture,
                caption=self._format_tab_caption(index, target_id, tab_capture.title, tab_capture.url, ordinal),
            )
        _log_info("_capture_all_tabs", f"captured {image_ordinal} of {len(tabs)} tabs into {output_directory}")
        return Snapshot(tabs=tabs, observed_at=observed_at)

    async def _capture_tab_snapshot(
        self,
        manager: BrowserManager,
        index: int,
        target_id: str,
        output_directory: str,
    ) -> TabSnapshot:
        """Capture one tab screenshot and read its title and URL."""
        _log_info("_capture_tab_snapshot", f"tab {target_id} capture starting")
        started_at = datetime.now(UTC)
        screenshot_error = ""
        screenshot_path = await self._capture_tab_screenshot(
            manager,
            target_id,
            os.path.join(output_directory, f"{index:02d}.jpg"),
        )
        finished_at = datetime.now(UTC)
        if screenshot_path is not None:
            _log_info("_capture_tab_snapshot", f"tab {target_id} screenshot saved to {screenshot_path}")
        else:
            _log_error("_capture_tab_snapshot", f"tab {target_id} has no screenshot")
            screenshot_error = "Screenshot capture failed"
        try:
            title = _sanitize_caption_text(
                await manager.get_title(target_id),
                _MAX_TITLE_LENGTH,
                should_strip_quotes=True,
            )
            url = _sanitize_caption_text(manager.get_url(target_id), _MAX_URL_LENGTH)
        except Exception as error:
            _log_error("_capture_tab_snapshot", f"tab {target_id} metadata failed, {error}")
            title, url = "", ""
        return TabSnapshot(target_id, screenshot_path, "", title, url, started_at, finished_at, screenshot_error)

    async def _capture_tab_screenshot(
        self,
        manager: BrowserManager,
        target_id: str,
        screenshot_path: str,
    ) -> str | None:
        """Capture one viewport screenshot and wait for the page to settle."""
        try:
            return await manager.capture_screenshot(
                target_id,
                screenshot_path,
                full_page=False,
                timeout=_CAPTURE_TIMEOUT_SECONDS,
                verbose=True,
            )
        except Exception:
            _log_error(
                "_capture_tab_screenshot",
                f"tab {target_id} capture failed",
                should_include_traceback=True,
            )
            return None

    def _format_tab_caption(
        self,
        index: int,
        target_id: str,
        title: str,
        url: str,
        image_ordinal: int | None,
    ) -> str:
        """Build the single caption line for one tab."""
        if image_ordinal is None:
            return f"{index}. {target_id} — no screenshot"
        return f'{index}. {target_id} — "{title}" — {url} — image {image_ordinal}'
