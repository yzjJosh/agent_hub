"""Shared fixtures for browser use tests."""

import asyncio
import sys
from collections.abc import Awaitable, AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

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
from app.agents.browser_use.models import TransactionStatus
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

    def tab_ids(self) -> list[str]:
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

    async def screenshot(self, target_id: str, path: str, full_page: bool = True, timeout: float = 30.0, verbose: bool = False) -> str | None:
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

    async def title(self, target_id: str) -> str | None:
        return self.titles.get(target_id)

    def url(self, target_id: str) -> str | None:
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
