"""A browser session that runs one action per transaction."""

from __future__ import annotations

import asyncio
import os
import re
import sys
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from app.agents.browser_use.checkpoint_store import (
    Artifact, CaptureInterval, CheckpointReservation, CheckpointStore, TabCapture,
)
from app.agents.browser_use.models import Snapshot, TabSnapshot, TransactionResult, TransactionStatus
from app.agents.browser_use.similarity import screen_similarity
from app.lib.browser.browser_manager import BrowserManager

__all__ = ["Action", "BrowserSession"]

Action = Callable[[BrowserManager, str | None], Awaitable[bool]]


_VERBOSE = True
_CAPTURE_TIMEOUT_S = 60
_TITLE_CAP = 120
_URL_CAP = 300
_TRANSACTION_BUDGET = 100
_GUARD_SIMILARITY = 0.70
_GUARD_BLOCK_LIMIT = 3
_DOWNLOAD_TIMEOUT_S = 30
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_SPACE_RUN_RE = re.compile(r"\s+")


def _resolve_tab(manager: BrowserManager, target_id: str | None, *, new_tab: bool) -> str | None:
    """The tab an action runs on, the named one or the only open one.

    Args:
        manager: The browser the tabs live in.
        target_id: The tab the caller names, or blank for the only open tab.
        new_tab: True when the action opens a tab of its own.

    Returns:
        The named tab, or the only open tab. None when no tab is named and the
        action opens its own or no tab is open.

    Raises:
        RuntimeError: If no tab is named, no new tab is asked for, and two or more tabs are open.
    """
    named = (target_id or "").strip()
    if named:
        return named
    if new_tab:
        return None
    open_tabs = manager.tab_ids()
    if len(open_tabs) > 1:
        raise RuntimeError(f"{len(open_tabs)} tabs are open, give the target id of the one to act on")
    return open_tabs[0] if open_tabs else None


class _SessionClosed(RuntimeError):
    """Raised when a session method runs while the session is closed."""


class _SessionOpen(RuntimeError):
    """Raised when open runs on a session that is already open."""


def _sanitize_caption_text(raw: str | None, cap: int, strip_quotes: bool = False) -> str:
    """Flatten a page-derived field to one plain line of text.

    Args:
        raw: The text the page gave, or None when it could not be read.
        cap: The longest the line may be, counting the ellipsis.
        strip_quotes: Remove double quotes when True.

    Returns:
        One line with control characters gone, quotes optionally gone, and a
        trailing ellipsis when the line had to be cut.
    """
    if raw is None:
        raw = ""
    text = _CONTROL_CHARS_RE.sub(" ", raw)
    text = _SPACE_RUN_RE.sub(" ", text).strip()
    if strip_quotes:
        text = text.replace('"', "")
    if len(text) > cap:
        text = text[: cap - 1] + "…"
    return text


def _reserve_file_sync(directory: str) -> str:
    """Make one fresh file in the folder, and give back its path.

    Args:
        directory: The folder the file lands in.

    Returns:
        The path of the file that was made.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "download")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.close(fd)
    return path


def _read_bytes_quietly(path: str) -> bytes:
    """Read a file, and give back empty bytes when it is not there.

    Args:
        path: The file to read.

    Returns:
        The file bytes, or empty bytes.
    """
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return b""


def _remove_quietly(path: str) -> None:
    """Remove a file, and say nothing when it is not there.

    Args:
        path: The file to remove.
    """
    try:
        os.remove(path)
    except OSError:
        pass


class BrowserSession:
    """Drive one browser by tab id, one action per transaction."""

    def __init__(self, downloads_dir: str, *, checkpoint_store: CheckpointStore | None = None) -> None:
        """Make a closed session that keeps the downloads folder.

        Args:
            downloads_dir: Folder that transaction files land in.
            checkpoint_store: Archive owned by this run when supplied.
        """
        self._downloads_dir = downloads_dir
        self._manager: BrowserManager | None = None
        self._lock = asyncio.Lock()
        self._transaction_id: int = 0
        self._last_snapshot: Snapshot | None = None
        self._consecutive_blocks: int = 0
        self._downloads: list[str] = []
        self._store = checkpoint_store if checkpoint_store is not None else CheckpointStore(downloads_dir)
        self._owns_store = checkpoint_store is None

    def _manager_or_raise(self) -> BrowserManager:
        """Return the manager, or raise when the session is closed."""
        if self._manager is None:
            raise _SessionClosed("session is not open")
        return self._manager

    def _run_dir(self) -> str:
        """Folder this transaction's files land in."""
        return os.path.join(self._downloads_dir, f"txn_{self._transaction_id}")

    async def open(self) -> None:
        """Connect the session's own manager to the browser.

        Raises:
            _SessionOpen: If the session is already open.
        """
        if self._manager is not None:
            raise _SessionOpen("session is already open")
        self._manager = await BrowserManager.create(download_dir=self._downloads_dir)

    async def close(self) -> None:
        """Let go of the manager. A second call does nothing."""
        try:
            if self._manager is not None:
                await self._manager.cleanup()
                self._manager = None
        finally:
            if self._owns_store:
                await self._store.close()

    async def transact(
        self,
        action: Action,
        target_id: str | None,
        *,
        new_tab: bool = False,
        monitor_download: bool = False,
        source_action: str = "action",
        capture_dom: Callable[[], TabCapture | None] | None = None,
    ) -> TransactionResult:
        """Run one browser action, and report the outcome with a view of the tabs.

        Args:
            action: The action to run. It takes the manager and the resolved tab,
                and returns True when it worked.
            target_id: The tab the action runs on. A blank tab is the one that is open.
            new_tab: Give the action no tab when True, and let it open a new one.
            monitor_download: Watch the resolved tab for a download when True.
            source_action: Name saved with the observation.
            capture_dom: Read any page text collected by the action.

        Returns:
            The outcome, the view of the tabs, and any file that came down.

        Raises:
            _SessionClosed: If the session is not open.
            KeyError: If the download is watched on a tab that is not open.
        """
        async with self._lock:
            manager = self._manager_or_raise()
            txn = self._transaction_id
            if _VERBOSE:
                print(
                    f"[BrowserSession.transact] txn {txn} of {_TRANSACTION_BUDGET} start, action={action!r}, "
                    f"target_id={target_id!r}, new_tab={new_tab}, monitor_download={monitor_download}",
                    flush=True,
                )
            if txn >= _TRANSACTION_BUDGET:
                used_up = "step budget is used up"
                if _VERBOSE:
                    print(
                        f"[BrowserSession.transact] txn {txn} end, status=CAP_REACHED, message={used_up!r}",
                        flush=True,
                    )
                snapshot = self._last_snapshot if self._last_snapshot is not None else Snapshot(tabs={})
                return TransactionResult(TransactionStatus.CAP_REACHED, used_up, snapshot, ())
            return await self._transact_action(
                manager, action, target_id, new_tab, monitor_download, source_action, capture_dom,
            )

    async def _transact_action(
        self, manager: BrowserManager, action: Action, target_id: str | None,
        new_tab: bool, monitor_download: bool, source_action: str,
        capture_dom: Callable[[], TabCapture | None] | None,
    ) -> TransactionResult:
        try:
            tab = _resolve_tab(manager, target_id, new_tab=new_tab)
        except Exception as exc:
            print(f"[BrowserSession.transact] Error: no tab to act on: {exc}\n{traceback.format_exc()}", file=sys.stderr)
            self._consecutive_blocks = 0
            ok, reason, files = False, f"action failed: {exc}", ()
        else:
            if await self._stale_guard(manager, None if new_tab else tab):
                return await self._observe(
                    source_action, TransactionStatus.STALE,
                    "screen changed since you last saw it, action not run", (), None,
                )
            ok, reason, files = await self._run_action(manager, action, tab, monitor_download=monitor_download)
        status = TransactionStatus.SUCCESS if ok else TransactionStatus.FAILED
        dom = capture_dom() if capture_dom is not None else None
        result = await self._observe(source_action, status, "" if ok else reason, files, dom)
        self._transaction_id += 1
        return result

    async def _observe(
        self, source_action: str, status: TransactionStatus, message: str,
        files: tuple[str, ...], dom: TabCapture | None,
    ) -> TransactionResult:
        reservation = None
        archive_error = ""
        try:
            reservation = await self._store.reserve()
            directory = reservation.directory
        except Exception:
            print(f"[BrowserSession._observe] Error: cannot reserve checkpoint\n{traceback.format_exc()}", file=sys.stderr)
            directory = os.path.join(self._downloads_dir, "unarchived", uuid4().hex)
            archive_error = "Checkpoint unavailable, do not repeat an executed action to repair the archive."
        try:
            snapshot = await self._snapshot(directory)
            if reservation is not None:
                snapshot = await self._publish(snapshot, reservation, source_action, status, dom)
            else:
                snapshot = replace(snapshot, archive_error=archive_error)
        except BaseException:
            if reservation is not None:
                await self._store.discard(reservation)
            raise
        self._last_snapshot = snapshot
        return TransactionResult(status, message, snapshot, files)

    async def _publish(
        self, snapshot: Snapshot, reservation: CheckpointReservation,
        source_action: str, status: TransactionStatus, dom: TabCapture | None,
    ) -> Snapshot:
        captures = self._tab_captures(snapshot, dom)
        try:
            record = await self._store.publish(reservation, source_action, status, captures)
        except Exception:
            print(f"[BrowserSession._publish] Error: cannot publish checkpoint\n{traceback.format_exc()}", file=sys.stderr)
            return replace(snapshot, archive_error="Checkpoint unavailable, do not repeat an executed action to repair the archive.")
        return replace(snapshot, checkpoint_id=record.checkpoint_id, observed_at=record.created_at)

    def _tab_captures(self, snapshot: Snapshot, dom: TabCapture | None) -> tuple[TabCapture, ...]:
        if dom is not None:
            dom = replace(
                dom,
                title=_sanitize_caption_text(dom.title, _TITLE_CAP, strip_quotes=True),
                url=_sanitize_caption_text(dom.url, _URL_CAP),
            )
        captures = []
        for tab in snapshot.tabs.values():
            interval = None
            if tab.capture_started_at is not None and tab.capture_finished_at is not None:
                interval = CaptureInterval(tab.capture_started_at, tab.capture_finished_at)
            capture = TabCapture(
                tab.target_id, tab.title, tab.url,
                Artifact(tab.screenshot_path, interval, tab.screenshot_error),
            )
            if dom is not None and dom.target_id == tab.target_id:
                capture = replace(capture, dom=dom.dom, dom_unavailable_reason=dom.dom_unavailable_reason)
            captures.append(capture)
        if dom is not None and dom.target_id not in snapshot.tabs:
            captures.append(dom)
        return tuple(captures)

    async def get_downloaded_files(self) -> list[str]:
        """List the files that came down in this session, in order.

        Returns:
            The file paths, one per download.

        Raises:
            _SessionClosed: If the session is not open.
        """
        self._manager_or_raise()
        return list(self._downloads)

    async def _run_action(
        self,
        manager: BrowserManager,
        action: Action,
        tab: str | None,
        *,
        monitor_download: bool,
    ) -> tuple[bool, str, tuple[str, ...]]:
        """Run the action on the tab.

        Args:
            manager: The browser the tabs live in.
            action: The action to run.
            tab: The tab the action runs on, or None for no tab.
            monitor_download: Watch the tab for a download when True.

        Returns:
            Whether it worked, the reason when it did not, and any file that
            came down.

        Raises:
            KeyError: If the download is watched on a tab that is not open.
        """
        if _VERBOSE:
            print(f"[BrowserSession._run_action] tab {tab} running {action!r}", flush=True)
        if monitor_download:
            files, ok, reason = await self._wait_download(manager, tab, action)
            if _VERBOSE:
                print(
                    f"[BrowserSession._run_action] tab {tab} download action returned {ok}, "
                    f"files={files}, reason={reason!r}",
                    flush=True,
                )
            return ok, reason, files
        try:
            ok = await action(manager, tab)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(
                f"[BrowserSession._run_action] Error: tab {tab} action raised: {exc}\n{traceback.format_exc()}",
                file=sys.stderr,
            )
            return False, f"action failed: {exc}", ()
        if _VERBOSE:
            print(f"[BrowserSession._run_action] tab {tab} action returned {ok}", flush=True)
        return ok, "" if ok else "action failed", ()

    async def _wait_download(
        self,
        manager: BrowserManager,
        target_id: str | None,
        action: Action,
    ) -> tuple[tuple[str, ...], bool, str]:
        """Run an action in a tab, wait for its download, and record the file.

        Args:
            manager: The browser the tab lives in.
            target_id: The tab to run the action in and watch for a download.
            action: The action to run, and the one that starts the download.

        Returns:
            The files that came down, whether it went right, and the reason
            when it did not.
        """
        if target_id is None:
            print("[BrowserSession._wait_download] Error: no tab is open to watch", file=sys.stderr)
            return (), False, "action failed: no tab is open, open one with open_url first"
        reserved = await asyncio.to_thread(_reserve_file_sync, self._run_dir())
        if _VERBOSE:
            print(f"[BrowserSession._wait_download] tab {target_id} holds {reserved} for the file", flush=True)
        try:
            outcome = await manager.wait_for_download(
                target_id,
                action=lambda: action(manager, target_id),
                save_path=reserved,
                timeout=_DOWNLOAD_TIMEOUT_S,
            )
        except Exception as exc:
            print(
                f"[BrowserSession._wait_download] Error: tab {target_id} download raised: {exc}\n"
                f"{traceback.format_exc()}",
                file=sys.stderr,
            )
            await asyncio.to_thread(_remove_quietly, reserved)
            raise
        if outcome.get("downloaded"):
            if _VERBOSE:
                print(f"[BrowserSession._wait_download] tab {target_id} file came down at {reserved}", flush=True)
            self._downloads.append(reserved)
            return (reserved,), True, ""
        print(
            f"[BrowserSession._wait_download] Error: tab {target_id} no file came within {_DOWNLOAD_TIMEOUT_S}s",
            file=sys.stderr,
        )
        await asyncio.to_thread(_remove_quietly, reserved)
        return (), False, "no download came"

    async def _stale_guard(self, manager: BrowserManager, tab: str | None) -> bool:
        """Hold the action back while the tab it aims at no longer looks the same.

        Args:
            manager: The browser the tabs live in.
            tab: The tab to compare, or None to skip the check.

        Returns:
            True when the action is held back.
        """
        last = self._last_snapshot
        if last is None:
            if _VERBOSE:
                print("[BrowserSession._stale_guard] no earlier view of the tabs, guard skipped", flush=True)
            self._consecutive_blocks = 0
            return False
        if await self._screen_changed(manager, last, tab):
            self._consecutive_blocks += 1
            if self._consecutive_blocks <= _GUARD_BLOCK_LIMIT:
                if _VERBOSE:
                    print(
                        f"[BrowserSession._stale_guard] action held back, "
                        f"{self._consecutive_blocks} of {_GUARD_BLOCK_LIMIT} blocks in a row",
                        flush=True,
                    )
                return True
            if _VERBOSE:
                print(
                    f"[BrowserSession._stale_guard] {self._consecutive_blocks} blocks in a row is past the limit "
                    f"of {_GUARD_BLOCK_LIMIT}, the action runs anyway",
                    flush=True,
                )
        self._consecutive_blocks = 0
        return False

    async def _screen_changed(self, manager: BrowserManager, last: Snapshot, tab: str | None) -> bool:
        """Check if the tab this action aims at looks different now.

        Args:
            manager: The browser the tabs live in.
            last: The view of the tabs the model last saw.
            tab: The tab to compare, or None to skip the check.

        Returns:
            True when the tab no longer looks like the shot the model saw.
            False when it still looks the same, and False when there is no
            such tab to compare.
        """
        if tab is None:
            if _VERBOSE:
                print("[BrowserSession._screen_changed] no tab to compare, guard skipped", flush=True)
            return False
        if tab not in manager.tab_ids():
            if _VERBOSE:
                print(f"[BrowserSession._screen_changed] tab {tab} is no longer open, guard skipped", flush=True)
            return False
        last_tab = last.tabs.get(tab)
        if last_tab is None:
            if _VERBOSE:
                print(
                    f"[BrowserSession._screen_changed] tab {tab} is not in the last view, guard skipped",
                    flush=True,
                )
            return False
        if last_tab.screenshot_path is None:
            if _VERBOSE:
                print(f"[BrowserSession._screen_changed] tab {tab} has no stored shot, guard skipped", flush=True)
            return False
        run_dir = self._run_dir()
        try:
            await asyncio.to_thread(os.makedirs, run_dir, exist_ok=True)
        except OSError:
            print(f"[BrowserSession._screen_changed] Error: cannot create guard folder\n{traceback.format_exc()}", file=sys.stderr)
        guard_path = os.path.join(run_dir, f"guard_{tab}.jpg")
        if await self._capture(manager, tab, guard_path) is None:
            print(
                f"[BrowserSession._screen_changed] Error: tab {tab} guard shot failed, guard skipped",
                file=sys.stderr,
            )
            return False
        fresh = await asyncio.to_thread(_read_bytes_quietly, guard_path)
        stored = await asyncio.to_thread(_read_bytes_quietly, last_tab.screenshot_path)
        await asyncio.to_thread(_remove_quietly, guard_path)
        similarity = await screen_similarity(stored, fresh)
        changed = similarity < _GUARD_SIMILARITY
        if _VERBOSE:
            print(
                f"[BrowserSession._screen_changed] tab {tab} similarity {similarity:.4f} against {_GUARD_SIMILARITY}, "
                f"action {'held back' if changed else 'admitted'}",
                flush=True,
            )
        return changed

    async def _snapshot(self, run_dir: str) -> Snapshot:
        """Shoot every open tab, and give back one view of them.

        Args:
            run_dir: Folder the screenshot files land in.

        Returns:
            One tab view per open tab, in tab order.
        """
        observed_at = datetime.now(UTC)
        manager = self._manager_or_raise()
        try:
            await asyncio.to_thread(os.makedirs, run_dir, exist_ok=True)
        except OSError:
            print(f"[BrowserSession._snapshot] Error: cannot create capture folder\n{traceback.format_exc()}", file=sys.stderr)
        target_ids = manager.tab_ids()
        if _VERBOSE:
            print(f"[BrowserSession._snapshot] shooting {len(target_ids)} tabs at once into {run_dir}", flush=True)
        captures = await asyncio.gather(
            *(
                self._capture_tab(manager, index, target_id, run_dir)
                for index, target_id in enumerate(target_ids, start=1)
            )
        )
        tabs: dict[str, TabSnapshot] = {}
        image_ordinal = 0
        for index, (target_id, capture) in enumerate(zip(target_ids, captures, strict=True), start=1):
            ordinal = None
            if capture.screenshot_path is not None:
                image_ordinal += 1
                ordinal = image_ordinal
            tabs[target_id] = replace(
                capture,
                caption=self._caption_line(index, target_id, capture.title, capture.url, ordinal),
            )
        if _VERBOSE:
            print(f"[BrowserSession._snapshot] shot {image_ordinal} of {len(tabs)} tabs into {run_dir}", flush=True)
        return Snapshot(tabs=tabs, observed_at=observed_at)

    async def _capture_tab(
        self,
        manager: BrowserManager,
        index: int,
        target_id: str,
        run_dir: str,
    ) -> TabSnapshot:
        """Shoot one tab, and read the title and url that go in its caption.

        Args:
            manager: The browser the tab lives in.
            index: The place of the tab among the open tabs, counting from one.
            target_id: The tab to shoot.
            run_dir: Folder the screenshot file lands in.

        Returns:
            The captured image and page metadata.
        """
        if _VERBOSE:
            print(f"[BrowserSession._capture_tab] tab {target_id} shot starting", flush=True)
        started_at = datetime.now(UTC)
        error = ""
        screenshot_path = await self._capture(manager, target_id, os.path.join(run_dir, f"{index:02d}.jpg"))
        finished_at = datetime.now(UTC)
        if screenshot_path is not None:
            if _VERBOSE:
                print(f"[BrowserSession._capture_tab] tab {target_id} shot saved to {screenshot_path}", flush=True)
        else:
            print(f"[BrowserSession._capture_tab] Error: tab {target_id} has no shot", file=sys.stderr)
            error = "Screenshot capture failed"
        try:
            title = _sanitize_caption_text(await manager.title(target_id), _TITLE_CAP, strip_quotes=True)
            url = _sanitize_caption_text(manager.url(target_id), _URL_CAP)
        except Exception as exc:
            print(f"[BrowserSession._capture_tab] Error: tab {target_id} metadata failed: {exc}", file=sys.stderr)
            title, url = "", ""
        return TabSnapshot(target_id, screenshot_path, "", title, url, started_at, finished_at, error)

    async def _capture(self, manager: BrowserManager, target_id: str, path: str) -> str | None:
        """Take one viewport shot, waiting up to the capture hold for the page to settle."""
        try:
            return await manager.screenshot(target_id, path, full_page=False, timeout=_CAPTURE_TIMEOUT_S, verbose=True)
        except Exception:
            print(f"[BrowserSession._capture] Error: tab {target_id} capture failed\n{traceback.format_exc()}", file=sys.stderr)
            return None

    def _caption_line(
        self,
        index: int,
        target_id: str,
        title: str,
        url: str,
        image_ordinal: int | None,
    ) -> str:
        """Build the one caption line for a tab."""
        if image_ordinal is None:
            return f"{index}. {target_id} — no screenshot"
        return f'{index}. {target_id} — "{title}" — {url} — image {image_ordinal}'
