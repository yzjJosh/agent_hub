"""Capture and finalize files downloaded by a browser tab."""

from __future__ import annotations

__all__ = ["DownloadResult", "DownloadTriggered", "wait_for_page_download"]

import asyncio
import os
import shutil
import sys
from collections.abc import Awaitable, Callable
from typing import NotRequired, TypedDict

from playwright.async_api import CDPSession

_DEFAULT_DOWNLOAD_TIMEOUT_S = 30
_INCOMPLETE_DOWNLOAD_SUFFIX = ".crdownload"


class DownloadTriggered(Exception):
    """Raised when opening a page starts a file download instead."""


class DownloadResult(TypedDict):
    """State of one downloaded file."""

    downloaded: bool
    path: NotRequired[str]
    filename: NotRequired[str]
    size: NotRequired[int]


class _DownloadEvents:
    """Track download start and completion events from one tab session."""

    def __init__(self) -> None:
        self.file_path: str | None = None
        self.completed: asyncio.Event = asyncio.Event()
        self.started_guids: set[str] = set()

    def on_will_begin(self, params: dict) -> None:
        """Record the identifier of a started download."""
        guid = params.get("guid")
        if guid:
            self.started_guids.add(guid)

    def on_progress(self, params: dict) -> None:
        """Mark completion when a tracked download finishes."""
        if params.get("state") == "completed" and params.get("guid") in self.started_guids:
            self.file_path = params.get("filePath")
            self.completed.set()


def _list_download_files(download_dir: str) -> dict[str, float]:
    """List completed files and modification times in a download folder."""
    result: dict[str, float] = {}
    try:
        for name in os.listdir(download_dir):
            if name.endswith(_INCOMPLETE_DOWNLOAD_SUFFIX):
                continue
            file_path = os.path.join(download_dir, name)
            try:
                result[name] = os.path.getmtime(file_path)
            except OSError:
                pass
    except Exception as exc:
        print(f"[downloads._list_download_files] Error: {exc}", file=sys.stderr)
    return result


async def _snapshot_download_dir(download_dir: str, verbose: bool = False) -> dict[str, float]:
    """Read completed files and modification times in a download folder."""
    result = await asyncio.to_thread(_list_download_files, download_dir)
    if verbose:
        print(f"[downloads._snapshot_download_dir] Found {len(result)} files", flush=True)
    return result


async def _claim_single_new_file(
    download_dir: str,
    snapshot_before: dict[str, float],
    verbose: bool = False,
) -> str | None:
    """Claim a single new or updated file in the download folder."""
    snapshot_now = await _snapshot_download_dir(download_dir)
    new_files = [
        os.path.join(download_dir, name)
        for name, mtime in snapshot_now.items()
        if name not in snapshot_before or mtime > snapshot_before.get(name, 0)
    ]
    if len(new_files) == 1:
        if verbose:
            print(f"[downloads._claim_single_new_file] Claiming {os.path.basename(new_files[0])}", flush=True)
        return new_files[0]
    if verbose:
        print(f"[downloads._claim_single_new_file] {len(new_files)} new files, not claiming", flush=True)
    return None


async def _enable_download_events(
    session: CDPSession,
    download_dir: str,
    verbose: bool,
) -> _DownloadEvents | None:
    """Turn on browser download events for one tab session and attach listeners."""
    events = _DownloadEvents()
    try:
        await session.send(
            "Browser.setDownloadBehavior",
            {
                "behavior": "allowAndName",
                "downloadPath": download_dir,
                "eventsEnabled": True,
            },
        )
        session.on("Browser.downloadWillBegin", events.on_will_begin)
        session.on("Browser.downloadProgress", events.on_progress)
    except Exception as exc:
        print(f"[downloads._enable_download_events] Error: {exc}", file=sys.stderr)
        return None
    if verbose:
        print("[downloads._enable_download_events] Download events on", flush=True)
    return events


async def _await_download_completion(
    events: _DownloadEvents,
    download_dir: str,
    snapshot_before: dict[str, float],
    timeout: float,
    verbose: bool,
) -> str | None:
    """Wait for the completion event or fall back to claiming one new file."""
    try:
        await asyncio.wait_for(events.completed.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        if verbose:
            print(
                f"[downloads._await_download_completion] No download event in {timeout}s, claiming from folder",
                flush=True,
            )
        return await _claim_single_new_file(download_dir, snapshot_before, verbose)
    return events.file_path


async def _claim_download(
    session: CDPSession | None,
    download_dir: str,
    trigger_action: Callable[[], Awaitable[bool]],
    snapshot_before: dict[str, float],
    timeout: float,
    verbose: bool,
) -> str | None:
    """Run a download trigger action and return the claimed file path."""
    if session is None:
        print("[downloads._claim_download] Error: no session for the tab", file=sys.stderr)
        return None
    events = await _enable_download_events(session, download_dir, verbose)
    if events is None:
        return None
    try:
        is_download_triggered = await trigger_action()
        if not is_download_triggered:
            if verbose:
                print("[downloads._claim_download] No download, returning immediately", flush=True)
            return None
        return await _await_download_completion(events, download_dir, snapshot_before, timeout, verbose)
    finally:
        session.remove_listener("Browser.downloadWillBegin", events.on_will_begin)
        session.remove_listener("Browser.downloadProgress", events.on_progress)


async def _finalize_download(
    source_path: str,
    save_path: str | None,
    verbose: bool = False,
) -> DownloadResult:
    """Move a downloaded file if requested and return its final metadata."""
    filename = os.path.basename(source_path)
    final_path = source_path
    if save_path:
        await asyncio.to_thread(os.makedirs, os.path.dirname(save_path) or ".", exist_ok=True)
        await asyncio.to_thread(shutil.move, source_path, save_path)
        final_path = save_path
        if verbose:
            print(f"[downloads._finalize_download] Moved {source_path} -> {final_path}", flush=True)
    file_size = await asyncio.to_thread(os.path.getsize, final_path)
    if verbose:
        print(f"[downloads._finalize_download] Final: {filename} ({file_size} bytes)", flush=True)
    return {
        "downloaded": True,
        "path": final_path,
        "filename": filename,
        "size": file_size,
    }


async def wait_for_page_download(
    session: CDPSession | None,
    download_dir: str,
    trigger_action: Callable[[], Awaitable[bool]],
    save_path: str | None = None,
    timeout: float = _DEFAULT_DOWNLOAD_TIMEOUT_S,
    verbose: bool = False,
) -> DownloadResult:
    """Run a step that may start a download and pick up the finished file.

    Args:
        session: Session for the tab being watched.
        download_dir: Folder where Chrome writes downloaded files.
        trigger_action: Async action that returns True when a download started.
        save_path: Destination path to move the file to, or None to leave it.
        timeout: How many seconds to wait for the download to finish.
        verbose: Print progress lines when True.

    Returns:
        Result dictionary with downloaded flag and file details when captured.
    """
    snapshot_before = await _snapshot_download_dir(download_dir, verbose=verbose)
    file_path = await _claim_download(session, download_dir, trigger_action, snapshot_before, timeout, verbose)
    if file_path:
        return await _finalize_download(file_path, save_path, verbose=verbose)
    return {"downloaded": False}
