"""Shared fixtures for browser use tests."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

import app.agents.browser_use.session as session_module
from app.agents.browser_use.checkpoint_store import (
    Artifact,
    CaptureInterval,
    Checkpoint,
    CheckpointStore,
    DomCapture,
    TabCapture,
)
from app.agents.browser_use.models import Snapshot, TabSnapshot, TransactionResult, TransactionStatus
from app.agents.browser_use.session import Action
from app.lib.tool.base import Tool
from tests.app.conftest import store_infra

_CANVAS_SIZE = 256
_CAPTURE_START = datetime(2026, 11, 7, 12, tzinfo=UTC)


def paint_canvas(color: int, fraction: float, band_color: int) -> Image.Image:
    """Draw a flat gray canvas, and paint a band across its top rows.

    Args:
        color: The gray level of the whole canvas.
        fraction: How much of the canvas height the band covers.
        band_color: The gray level of the band.

    Returns:
        The drawn canvas.
    """
    canvas = Image.new("L", (_CANVAS_SIZE, _CANVAS_SIZE), color)
    rows = round(_CANVAS_SIZE * fraction)
    if rows > 0:
        canvas.paste(band_color, (0, 0, _CANVAS_SIZE, rows))
    return canvas


class FakeManager:
    """A duck-typed browser manager that records calls and plays canned state."""

    def __init__(self) -> None:
        self.pages: dict[str, str] = {}
        self.titles: dict[str, str] = {}
        self.urls: dict[str, str] = {}
        self.capture_outcomes: dict[str, str] = {}
        self.capture_colors: dict[str, int] = {}
        self.capture_patches: dict[str, tuple[float, int]] = {}
        self.capture_barrier: asyncio.Barrier | None = None
        self.screenshot_calls: int = 0
        self.cleanup_calls: int = 0
        self.creates: int = 0
        self.downloads_dir: str | None = None
        self.last_click: tuple[str, float, float] | None = None
        self.last_type_text: tuple[str, float, float, str] | None = None
        self.last_replace_input_text: tuple[str, float, float, str, str] | None = None
        self.last_scroll: tuple[str, float, float] | None = None
        self.last_press_key: tuple[str, str] | None = None
        self.last_go_back: str | None = None
        self.last_go_forward: str | None = None
        self.last_close_page: str | None = None
        self.last_solve_captcha: tuple[str, int] | None = None
        self.last_screenshot: tuple[str, str, bool, float, bool] | None = None
        self.last_download: tuple[str, str | None, Callable[[], Awaitable[bool]] | None, str | None, int] | None = None
        self.download_queue: list[dict[str, bool]] = []
        self.download_exception: Exception | None = None
        self._page_seq = 0

    def open_tab(self, title: str = "", url: str = "") -> str:
        """Register an open tab and return its target id."""
        self._page_seq += 1
        target_id = f"tab{self._page_seq}"
        self.pages[target_id] = url
        self.titles[target_id] = title
        self.urls[target_id] = url
        return target_id

    def list_tab_ids(self) -> list[str]:
        """Give back the ids of the open tabs."""
        return list(self.pages)

    async def click_at_relative(self, target_id: str, x: float, y: float) -> bool:
        if target_id not in self.pages:
            return False
        self.last_click = (target_id, x, y)
        return True

    async def type_text_at_relative(self, target_id: str, x: float, y: float, text: str) -> bool:
        if target_id not in self.pages:
            return False
        self.last_type_text = (target_id, x, y, text)
        return True

    async def replace_input_text_at_relative(self, target_id: str, x: float, y: float, old_text: str, new_text: str) -> bool:
        if target_id not in self.pages:
            return False
        self.last_replace_input_text = (target_id, x, y, old_text, new_text)
        return True

    async def scroll_by_relative(self, target_id: str, dx: float, dy: float) -> bool:
        if target_id not in self.pages:
            return False
        self.last_scroll = (target_id, dx, dy)
        return True

    async def press_key(self, target_id: str, key: str) -> bool:
        if target_id not in self.pages:
            return False
        self.last_press_key = (target_id, key)
        return True

    async def go_back(self, target_id: str) -> bool:
        if target_id not in self.pages:
            return False
        self.last_go_back = target_id
        return True

    async def go_forward(self, target_id: str) -> bool:
        if target_id not in self.pages:
            return False
        self.last_go_forward = target_id
        return True

    async def close_page(self, target_id: str) -> bool:
        if target_id not in self.pages:
            return False
        self.last_close_page = target_id
        self.pages.pop(target_id, None)
        return True

    async def solve_captcha(self, target_id: str, max_steps: int) -> bool:
        if target_id not in self.pages:
            return False
        self.last_solve_captcha = (target_id, max_steps)
        return True

    async def capture_screenshot(self, target_id: str, path: str, full_page: bool = True, timeout: float = 30.0, verbose: bool = False) -> str | None:
        self.screenshot_calls += 1
        self.last_screenshot = (target_id, path, full_page, timeout, verbose)
        if self.capture_barrier is not None:
            await self.capture_barrier.wait()
        outcome = self.capture_outcomes.get(target_id, "ok")
        if outcome == "none":
            return None
        if outcome == "timeout":
            print("[FakeManager.screenshot] Error: capture timed out", file=sys.stderr)
            return None
        if outcome == "garbage":
            with open(path, "wb") as f:
                f.write(b"jpg")
            return path
        self._paint(target_id).save(path, format="PNG")
        return path

    def _paint(self, target_id: str) -> Image.Image:
        """Draw the canvas for a tab, with its patch over the top rows."""
        color = self.capture_colors.get(target_id, 0)
        patch = self.capture_patches.get(target_id)
        if patch is None:
            return paint_canvas(color, 0.0, color)
        fraction, band_color = patch
        return paint_canvas(color, fraction, band_color)

    async def wait_for_download(
        self,
        target_id: str,
        url: str | None = None,
        action: Callable[[], Awaitable[bool]] | None = None,
        save_path: str | None = None,
        timeout: int = 30,
    ) -> dict[str, bool]:
        if url is None and action is None:
            raise ValueError("Either url or action must be provided")
        if target_id not in self.pages:
            raise KeyError(f"No page found for target_id: {target_id}")
        self.last_download = (target_id, url, action, save_path, timeout)
        if action is not None:
            await action()
        if self.download_exception is not None:
            error = self.download_exception
            self.download_exception = None
            raise error
        if self.download_queue:
            return self.download_queue.pop(0)
        return {"downloaded": False}

    async def get_title(self, target_id: str) -> str | None:
        return self.titles.get(target_id)

    def get_url(self, target_id: str) -> str | None:
        return self.urls.get(target_id)

    async def cleanup(self) -> None:
        self.cleanup_calls += 1


@pytest.fixture
def fake_manager(monkeypatch: pytest.MonkeyPatch) -> FakeManager:
    """Point the session's manager at a recording fake."""
    manager = FakeManager()

    async def _create(download_dir: str) -> FakeManager:
        manager.creates += 1
        manager.downloads_dir = download_dir
        return manager

    monkeypatch.setattr(session_module, "BrowserManager", SimpleNamespace(create=_create))
    return manager


@pytest.fixture(autouse=True)
async def browser_use_store_infra(tmp_path: Path) -> AsyncIterator[None]:
    """Point the process-wide stores at a throwaway database for the test."""
    async for _ in store_infra(tmp_path):
        yield


@pytest.fixture
async def archive(tmp_path: Path) -> AsyncIterator[CheckpointStore]:
    """Provide one empty checkpoint store."""
    store = CheckpointStore(str(tmp_path / "run"), resources_root=str(tmp_path))
    yield store
    await store.close()


@pytest.fixture
def dom_time() -> CaptureInterval:
    """Provide the time of a saved page dump."""
    return CaptureInterval(_CAPTURE_START, _CAPTURE_START + timedelta(seconds=1))


@pytest.fixture
def image_time() -> CaptureInterval:
    """Provide the time of a saved screenshot."""
    return CaptureInterval(_CAPTURE_START + timedelta(seconds=2), _CAPTURE_START + timedelta(seconds=3))


@pytest.fixture
def publish_checkpoint(dom_time: CaptureInterval, image_time: CaptureInterval) -> Callable[..., Awaitable[Checkpoint]]:
    """Provide a saved browser observation."""
    async def publish(
        archive: CheckpointStore, text: str = "Feedback", target_id: str = "tab", image: bytes = b"original image"
    ) -> Checkpoint:
        reservation = await archive.reserve()
        directory = Path(reservation.directory)
        directory.mkdir(parents=True)
        path = directory / "01.jpg"
        path.write_bytes(image)
        tab = TabCapture(
            target_id, "View", "https://example.test/feedback",
            Artifact(str(path), image_time, ""), DomCapture(text, dom_time),
        )
        return await archive.publish(reservation, "dump_page", TransactionStatus.SUCCESS, (tab,))

    return publish


STANDARD_ACTION_TOOL_NAMES: tuple[str, ...] = (
    "open_url",
    "click",
    "move_mouse",
    "type_text",
    "replace_input_text",
    "scroll",
    "press_key",
    "go_back",
    "go_forward",
    "close_page",
    "solve_captcha",
    "wait",
    "download_click",
    "refresh",
    "dump_page",
)

COMPLETED_OUTPUT_RAW = json.dumps({"status": "completed", "summary": "done"})

_LLM_BACKEND_VARS = (
    "REASON_THINKING_BACKEND_URL",
    "REASON_THINKING_MODEL",
    "REASON_THINKING_API_KEY",
    "REASON_BACKEND_URL",
    "REASON_MODEL",
    "REASON_API_KEY",
)

_DEFAULT_TABS = {
    "tab1": TabSnapshot("tab1", "/tmp/dl/txn_1/01.jpg", '1. tab1 — "Shop" — https://shop.example — image 1'),
    "tab2": TabSnapshot("tab2", None, "2. tab2 — no screenshot"),
}


def build_default_snapshot() -> Snapshot:
    """Build a canned two-tab snapshot."""
    return Snapshot(tabs=dict(_DEFAULT_TABS))


def build_success_transaction(message: str = "") -> TransactionResult:
    """Build a successful transaction result with the default snapshot."""
    return TransactionResult(TransactionStatus.SUCCESS, message, build_default_snapshot(), ())


def _build_manager_snapshot(manager: FakeActionManager) -> Snapshot:
    """Build one canned tab line per open tab on the action manager."""
    tabs: dict[str, TabSnapshot] = {}
    for index, target_id in enumerate(manager.list_tab_ids(), start=1):
        tabs[target_id] = TabSnapshot(target_id, None, f"{index}. {target_id} — 'T' — https://x — image {index}")
    return Snapshot(tabs=tabs)


class FakeActionSnapshot:
    """Recording stand-in for the page snapshot called by action closures."""

    def __init__(self, manager: FakeActionManager, target_id: str, move_fails: bool = False) -> None:
        self._manager = manager
        self._target_id = target_id
        self._move_fails = move_fails
        self.title: str = "Page title"
        self.url: str = "https://example.com"

    def _record(self, method: str, args: tuple) -> None:
        self._manager.recorded.setdefault(method, []).append((self._target_id, *args))
        self._manager.order.append(method)

    async def click_at_relative(self, x: float, y: float, verbose: bool = False) -> bool:
        self._record("click_at_relative", (x, y, verbose))
        return True

    async def upload_with_chooser_relative(
        self, x: float, y: float, file_paths: list[str], verbose: bool = False
    ) -> bool:
        self._record("upload_with_chooser_relative", (x, y, file_paths, verbose))
        return True

    async def type_text_at_relative(
        self, x: float, y: float, text: str, submit: bool = False, verbose: bool = False
    ) -> bool:
        self._record("type_text_at_relative", (x, y, text, submit, verbose))
        return True

    async def replace_input_text_at_relative(
        self, x: float, y: float, old_text: str, new_text: str, verbose: bool = False
    ) -> bool:
        self._record("replace_input_text_at_relative", (x, y, old_text, new_text, verbose))
        return True

    async def move_mouse_to_relative(self, x_ratio: float, y_ratio: float, verbose: bool = False) -> bool:
        self._record("move_mouse_to_relative", (x_ratio, y_ratio, verbose))
        return not self._move_fails

    async def scroll_by_relative(self, dx: float, dy: float, verbose: bool = False) -> bool:
        self._record("scroll_by_relative", (dx, dy, verbose))
        return True

    async def press_key(self, key: str, verbose: bool = False) -> bool:
        self._record("press_key", (key, verbose))
        return True

    async def solve_captcha(self, max_steps: int, verbose: bool = False) -> bool:
        self._record("solve_captcha", (max_steps, verbose))
        return True

    async def filtered_tree(self, verbose: bool = False) -> dict | None:
        self._record("filtered_tree", (verbose,))
        if self._manager.no_tree:
            return None
        return {"role": "web", "children": [{"role": "heading", "name": "Page title"}]}


class FakeActionManager:
    """Recording stand-in for the browser manager called by action closures."""

    def __init__(
        self,
        tab_ids: tuple[str, ...] = (),
        no_snapshot: bool = False,
        move_fails: bool = False,
        no_tree: bool = False,
    ) -> None:
        self.recorded: dict[str, list[tuple]] = {}
        self.order: list[str] = []
        self._page_seq = 0
        self._tab_ids: list[str] = list(tab_ids)
        self._no_snapshot = no_snapshot
        self._move_fails = move_fails
        self.no_tree = no_tree

    def list_tab_ids(self) -> list[str]:
        return list(self._tab_ids)

    def add_tab(self, target_id: str) -> None:
        self._tab_ids.append(target_id)

    def _record(self, method: str, args: tuple) -> None:
        self.recorded.setdefault(method, []).append(args)
        self.order.append(method)

    async def open_page(self, verbose: bool = False) -> str:
        self._record("open_page", (verbose,))
        self._page_seq += 1
        return f"new{self._page_seq}"

    async def navigate(
        self,
        target_id: str,
        url: str,
        wait_until: str | None = None,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> bool:
        self._record("navigate", (target_id, url, wait_until, timeout, verbose))
        return True

    def get_url(self, target_id: str, verbose: bool = False) -> str:
        self._record("get_url", (target_id, verbose))
        return "https://example.com"

    async def capture_snapshot(
        self, target_id: str, timeout: int | None = None, verbose: bool = False
    ) -> FakeActionSnapshot | None:
        self._record("capture_snapshot", (target_id, timeout, verbose))
        if self._no_snapshot:
            return None
        return FakeActionSnapshot(self, target_id, move_fails=self._move_fails)

    async def wait_for_load(
        self,
        target_id: str,
        state: str | None = None,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> bool:
        self._record("wait_for_load", (target_id, state, timeout, verbose))
        return True

    async def go_back(
        self,
        target_id: str,
        wait_until: str | None = None,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> bool:
        self._record("go_back", (target_id, wait_until, timeout, verbose))
        return True

    async def go_forward(
        self,
        target_id: str,
        wait_until: str | None = None,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> bool:
        self._record("go_forward", (target_id, wait_until, timeout, verbose))
        return True

    async def close_page(self, target_id: str, verbose: bool = False) -> bool:
        self._record("close_page", (target_id, verbose))
        return True


class FakeSession:
    """Scripted browser session that records every transaction it receives."""

    def __init__(
        self,
        results: list[TransactionResult],
        downloads: list[str] | None = None,
        manager: FakeActionManager | None = None,
    ) -> None:
        self._results = results
        self._downloads = list(downloads or [])
        self._manager = manager
        self.opened: int = 0
        self.closed: int = 0
        self.transactions: list[tuple[Action, str | None, bool, bool]] = []
        self.source_actions: list[str] = []
        self.dom_captures: list[TabCapture | None] = []
        self.checkpoint_store: CheckpointStore | None = None
        self.open_error: Exception | None = None
        self.close_error: Exception | None = None

    async def open(self) -> None:
        self.opened += 1
        if self.open_error is not None:
            raise self.open_error

    async def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error

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
        self.transactions.append((action, target_id, should_monitor_download, is_new_tab))
        self.source_actions.append(source_action)
        if self._manager is None:
            return self._results.pop(0)
        manager = self._manager
        tab = (target_id or "").strip()
        if not tab:
            if is_new_tab:
                tab = None
            else:
                open_tabs = manager.list_tab_ids()
                if len(open_tabs) > 1:
                    return TransactionResult(
                        TransactionStatus.FAILED,
                        f"action failed: {len(open_tabs)} tabs are open, give the target id of the one to act on",
                        _build_manager_snapshot(manager),
                        (),
                    )
                tab = open_tabs[0] if open_tabs else None
        try:
            has_succeeded = await action(manager, tab)
            self.dom_captures.append(capture_dom() if capture_dom is not None else None)
        except Exception as error:
            return TransactionResult(
                TransactionStatus.FAILED, f"action failed: {error}", _build_manager_snapshot(manager), ()
            )
        if not has_succeeded:
            return TransactionResult(TransactionStatus.FAILED, "action failed", _build_manager_snapshot(manager), ())
        return TransactionResult(TransactionStatus.SUCCESS, "", _build_manager_snapshot(manager), ())

    async def get_downloaded_files(self) -> list[str]:
        return list(self._downloads)


def find_tool(name: str, tools: tuple[Tool, ...]) -> Tool:
    """Return the tool with the given name from a tuple of tools."""
    return next(tool for tool in tools if tool.name == name)


def clear_llm_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear all environment variables for the thinking and plain LLM backends."""
    for name in _LLM_BACKEND_VARS:
        monkeypatch.delenv(name, raising=False)


def set_thinking_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure the thinking backend URL for tests and clear the plain backend."""
    clear_llm_backends(monkeypatch)
    monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://think.test/v1")


def build_mock_stream(
    history: list[dict],
    error: Exception | None,
    *,
    final_output: str | None = None,
    turns: int = 0,
    new_items: list[dict] | None = None,
) -> MagicMock:
    """Build a mock streamed runner response for turn-limit and overflow tests."""
    async def events() -> AsyncIterator:
        if error is not None:
            raise error
        for event in ():
            yield event

    return MagicMock(
        stream_events=events,
        raw_responses=[MagicMock() for _ in range(turns)],
        new_items=history if new_items is None else new_items,
        to_input_list=lambda: list(history),
        final_output=final_output,
    )
