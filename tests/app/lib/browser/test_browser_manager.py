"""Tests for driving a browser."""
import asyncio
import inspect
import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from PIL import Image as PilImage
from types import SimpleNamespace

from app.lib.browser import browser_manager as bm_module
from app.lib.browser.browser_manager import BrowserManager, Viewport
from app.lib.browser.page_snapshot import PageSnapshot
from tests.app.conftest import skip_if_no_chrome
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from typing import Self
from typing import Optional
from pathlib import Path

_VIEWPORT_WIDTH = 1920
_VIEWPORT_HEIGHT = 1080
_SET_VIEWPORT_WIDTH = 1024
_SET_VIEWPORT_HEIGHT = 640
_HALF_VIEWPORT_WIDTH = 960.0
_HALF_VIEWPORT_HEIGHT = 540.0
_HALF_VIEWPORT_W = 960
_HALF_VIEWPORT_H = 540
_WINDOW_WIDER_THAN_CLIENT = 2000
_WINDOW_TALLER_THAN_CLIENT = 1095
_NETWORK_IDLE_TIMEOUT_MS = 5000
_SCROLL_DELAY = 0.2
_RENDER_DELAY = 0.5
_CENTER_RATIO = 0.5
_BEYOND_VIEWPORT_X = 3000
_BEYOND_VIEWPORT_Y = 5000
_SCROLL_STEP_MAX = 300
_MAX_STUCK_WHEEL_CALLS = 30
_BELOW_FOLD_Y = 2000
_SLOW_SCAN_BLOCK_S = 0.05
_SLOW_MOVE_BLOCK_S = 0.02
_TEST_CLEANUP_STEP_TIMEOUT_S = 0.2
_SHORT_DOWNLOAD_WAIT_S = 0.3
_SETTLE_START_Y = 4000.0
_SETTLE_DRIFT = 60.0
_SETTLE_Y = 4460.0
_SCROLLED_VIEWPORT_PAGE_X = 150.0
_SCROLLED_VIEWPORT_PAGE_Y = 800.0
_REL_SCROLL_DX = 0.5
_REL_SCROLL_DY = 0.25
_THREE_QUARTER_RATIO = 0.75
_AIM_X_MAX = 0.75
_REL_SCROLL_WIDTH = 960.0
_REL_SCROLL_HEIGHT = 270.0
_REL_SCROLL_WIDTH_WIDE = 1000.0
_REL_SCROLL_HEIGHT_TALL = 273.75
_SCROLL_DISTANCE = 300.0
_SCROLL_SIDEWAYS = 100.0
_TYPE_TEXT = "hello world"
_FIELD_TEXT = "the quick brown fox"
_REPLACED_FIELD_TEXT = "the slow brown fox"
_GEOMETRY_TARGET_ID = "viewport-target"
_GEOMETRY_SCROLL_Y = 800
_GEOMETRY_PAGE_Y_TOLERANCE = 10.0
_GEOMETRY_PAGE_HEIGHT = 4000
_SCROLL_EXTRA_WIDTH = 1000
_SCROLL_EXTRA_HEIGHT_SHORT = 3000
_SCROLL_EXTRA_HEIGHT_LONG = 5000
_MOUSE_TEST_WIDTH = 1280
_MOUSE_TEST_HEIGHT = 800
_VIEWPORT_LOCK_INTERVAL_S = 0.1
_VIEWPORT_LOCK_READS = 30
_RECAPTCHA_CLICK_TIMEOUT_MS = 1500


def _fake_page(fill: Optional[AsyncMock] = None, evaluate: Optional[AsyncMock] = None) -> AsyncMock:
    """Make a fake page that answers the reads the manager makes on it."""
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.content.return_value = "<html><body>test</body></html>"
    mock_page.url = "http://test/"
    mock_page.title.return_value = "Test"
    mock_page.evaluate.return_value = [_VIEWPORT_WIDTH, _VIEWPORT_HEIGHT]
    mock_locator = AsyncMock()
    mock_locator.count.return_value = 1
    mock_locator.bounding_box.return_value = {"x": 100, "y": 100, "width": 50, "height": 30}
    mock_page.locator = MagicMock()
    mock_page.locator.return_value.first = mock_locator
    if fill is not None:
        mock_page.fill = fill
    if evaluate is not None:
        mock_page.evaluate = evaluate
    return mock_page


def _test_context(page: AsyncMock, session: Optional[MagicMock], attach_session: bool = True) -> MagicMock:
    """Make a fake context that hands out one fake tab and its fake session."""
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    if attach_session:
        context.new_cdp_session = AsyncMock(return_value=session)
    else:
        context.new_cdp_session = AsyncMock(side_effect=Exception("CDP down"))
    return context


async def _created_manager(context: MagicMock, download_dir: Optional[str] = "/tmp/test_dl", cdp_url: Optional[str] = "http://localhost:9222", fake_pw: Optional[MagicMock] = None) -> BrowserManager:
    """Make a connected manager over a fake browser that shows one fake context."""
    fake_browser = MagicMock()
    fake_browser.contexts = [context]
    fake_pw = fake_pw if fake_pw is not None else MagicMock()
    fake_pw.chromium.connect_over_cdp = AsyncMock(return_value=fake_browser)
    fake_pw.stop = AsyncMock()
    with patch.object(bm_module, "async_playwright") as ap:
        ap.return_value.start = AsyncMock(return_value=fake_pw)
        return await BrowserManager.create(cdp_url=cdp_url, download_dir=download_dir)


async def _empty_manager() -> BrowserManager:
    """Make a connected manager with no tabs."""
    return await _created_manager(_test_context(AsyncMock(), _layout_session()))


async def _make_manager(fill: Optional[AsyncMock] = None, evaluate: Optional[AsyncMock] = None, download_dir: str = "/tmp/test_dl", session: Optional[MagicMock] = None, attach_session: bool = True) -> tuple[BrowserManager, AsyncMock, str]:
    """Make a connected manager holding one fake page and one fake tracked session."""
    mock_page = _fake_page(fill, evaluate)
    if session is None:
        session = _layout_session()
    context = _test_context(mock_page, session, attach_session)
    mgr = await _created_manager(context, download_dir)
    target_id = await mgr.new_page()
    return mgr, mock_page, target_id


def _screenshot_to_disk(mock_page: AsyncMock) -> None:
    """Make screenshots save a blank picture the size of the window."""
    def _save(path: str, **_kw: bool) -> None:
        PilImage.new("RGB", (_VIEWPORT_WIDTH, _VIEWPORT_HEIGHT), "white").save(path)
    mock_page.screenshot = AsyncMock(side_effect=_save)


def _mock_locator(mock_page: MagicMock) -> None:
    """Give the fake page one element to find."""
    mock_locator = MagicMock()
    mock_locator.first = AsyncMock()
    mock_locator.first.count.return_value = 1
    mock_locator.first.bounding_box.return_value = {"x": 100, "y": 100, "width": 50, "height": 30}
    mock_page.locator = MagicMock(return_value=mock_locator)
    mock_page.evaluate.return_value = [_VIEWPORT_WIDTH, _VIEWPORT_HEIGHT]
    return mock_locator


async def _make_settle_page(settles: bool) -> tuple[BrowserManager, AsyncMock, list[float], str]:
    """Build a fake page whose scroll spot drifts toward its settled spot.

    The target (50, 5000) sits inside the window once the page scrolls past
    y=3920. The spot starts at y=4000, already in view, and drifts down to
    y=4460. When settles is True it holds at 4460, when False it keeps
    drifting past it. The fourth value lists every scroll spot the page
    reported, in order.
    """
    state = {"reads": 0}
    positions: list[float] = []

    def _next_y() -> float:
        state["reads"] += 1
        y = _SETTLE_START_Y + (state["reads"] - 1) * _SETTLE_DRIFT
        if settles:
            y = min(y, _SETTLE_Y)
        positions.append(y)
        return y

    mgr, mock_page, tab = await _make_manager(session=_drifting_layout_session(_next_y))
    return mgr, mock_page, positions, tab


async def _navigate_long_and_wait(browser_manager: BrowserManager, tid: str, test_server_url: str) -> None:
    """Open the long page and wait for it to draw."""
    await browser_manager.navigate(tid, test_server_url + "/page/long")
    time.sleep(_RENDER_DELAY)


async def _snapshot_visibility_tree(browser_manager: BrowserManager, tid: str, test_server_url: str) -> None:
    """Open the visibility page and give back its tree as text."""
    await browser_manager.navigate(tid, test_server_url + "/page/visibility")
    result = await browser_manager.snapshot(tid)
    tree = await result.filtered_tree()
    return json.dumps(tree, ensure_ascii=False)


async def _setup_mouse_tracker(browser_manager: BrowserManager, target_id: str) -> None:
    """Start recording where the mouse goes on the page."""
    await browser_manager.evaluate(target_id, """() => {
      window.__mousePos = {x: 0, y: 0};
      document.addEventListener('mousemove', e => {
        window.__mousePos = {x: e.clientX, y: e.clientY};
      }, true);
    }""")


async def _get_viewport(browser_manager: BrowserManager, tid: str) -> None:
    """Give back the width and height of the window."""
    return await browser_manager.evaluate(tid, "() => ({w: window.innerWidth, h: window.innerHeight})")


async def _element_bbox(browser_manager: BrowserManager, tid: str, selector: str) -> dict:
    """Give back the box of the first matching element, measured from the window corner."""
    return await browser_manager.evaluate(
        tid,
        f"() => {{ const r = document.querySelector('{selector}').getBoundingClientRect();"
        " return {x: r.x, y: r.y, width: r.width, height: r.height}; }",
    )


async def _assert_mouse_at(browser_manager: BrowserManager, tid: str, expected_x: int, expected_y: int) -> None:
    """Check the mouse sits within 5 pixels of the wanted spot."""
    pos = await browser_manager.evaluate(tid, "() => window.__mousePos")
    assert abs(pos["x"] - expected_x) < 5, f"Expected x~{expected_x}, got {pos['x']}"
    assert abs(pos["y"] - expected_y) < 5, f"Expected y~{expected_y}, got {pos['y']}"


def _assert_download_file(save_path: str) -> None:
    """Check the saved file is there and is not empty."""
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0


def _fake_download_session() -> MagicMock:
    """Make a fake CDP session that records its listeners and calls them on demand."""
    session = MagicMock()
    session.send = AsyncMock(return_value={})
    session.detach = AsyncMock()
    listeners: dict[str, list[Callable[[dict], None]]] = {}

    def _on(name: str, handler: Callable[[dict], None]) -> None:
        listeners.setdefault(name, []).append(handler)

    def _remove_listener(name: str, handler: Callable[[dict], None]) -> None:
        if handler in listeners.get(name, []):
            listeners[name].remove(handler)

    async def _emit(events: list[tuple[str, dict]]) -> None:
        for name, params in events:
            for handler in list(listeners.get(name, [])):
                handler(params)

    session.on = _on
    session.remove_listener = _remove_listener
    session.emit = _emit
    session.listeners = listeners
    return session


async def _manager_with_sessions(tmp_path: Path, pages: dict[str, AsyncMock]) -> tuple[BrowserManager, list[str], dict[str, MagicMock]]:
    """Make a manager whose tabs each get their own fake CDP session."""
    session_by_page: dict[AsyncMock, MagicMock] = {}

    def _attach(page: AsyncMock) -> MagicMock:
        session = _fake_download_session()
        session_by_page[page] = session
        return session

    context = MagicMock()
    context.new_page = AsyncMock(side_effect=list(pages.values()))
    context.new_cdp_session = AsyncMock(side_effect=_attach)
    mgr = await _created_manager(context, str(tmp_path))
    target_ids = [await mgr.new_page() for _ in pages]
    sessions = {key: session_by_page[page] for key, page in pages.items()}
    return mgr, target_ids, sessions


async def _assert_snapshot_partial_on_timeout(mgr: BrowserManager, mock_page: MagicMock, target_id: str) -> None:
    """Check a snapshot still comes back with page text after the wait runs out."""
    mock_page.wait_for_load_state.side_effect = PlaywrightTimeoutError("Timeout 5000ms exceeded")
    result = await mgr.snapshot(target_id)
    assert result.html is not None and len(result.html) > 0, "snapshot() should return partial PageSnapshot with HTML on timeout"


async def _lock_viewport(browser_manager: BrowserManager, tid: str) -> None:
    """Pin the window to a fixed size and wait until it holds."""
    await browser_manager.set_viewport_size(tid, _MOUSE_TEST_WIDTH, _MOUSE_TEST_HEIGHT)
    for _ in range(_VIEWPORT_LOCK_READS):
        size = await browser_manager.evaluate(tid, "() => [window.innerWidth, window.innerHeight]")
        if size == [_MOUSE_TEST_WIDTH, _MOUSE_TEST_HEIGHT]:
            return
        await asyncio.sleep(_VIEWPORT_LOCK_INTERVAL_S)


async def _navigate_blank_and_track(browser_manager: BrowserManager, tid: str) -> None:
    """Open a blank page, pin the window to a fixed size, and start recording the mouse."""
    await browser_manager.navigate(tid, "about:blank")
    await _lock_viewport(browser_manager, tid)
    await _setup_mouse_tracker(browser_manager, tid)


async def _scroll_past_elements(browser_manager: BrowserManager, tid: str, test_server_url: str) -> None:
    """Open the long page and scroll down past the test elements."""
    await _navigate_long_and_wait(browser_manager, tid, test_server_url)
    await browser_manager.scroll_to(tid, 50, 6000)
    time.sleep(_SCROLL_DELAY)


@pytest.mark.live
@skip_if_no_chrome
class TestConnectCDP:
    """Tests for hooking up to a browser that is already running."""

    async def test_connect_cdp(self, browser_manager: BrowserManager) -> None:
        """A fresh manager is connected and can open a tab on the running browser."""
        target_id = await browser_manager.new_page()
        try:
            assert target_id in browser_manager.tab_ids()
        finally:
            await browser_manager.close_page(target_id)

    async def test_context_reused(self) -> None:
        """A second manager connects to the same running browser and opens a tab."""
        mgr2 = await BrowserManager.create()
        try:
            target_id = await mgr2.new_page()
            assert target_id in mgr2.tab_ids()
        finally:
            await mgr2.close_page(target_id)
            await mgr2.cleanup()


class TestConnectArgs:
    """Tests for the arguments handed to the connect call."""

    async def test_connect_passes_artifacts_dir(self, tmp_path: Path) -> None:
        """Connecting tells Chrome to put downloads in the folder."""
        captured: dict = {}

        fake_browser = MagicMock()
        fake_browser.contexts = [MagicMock()]
        fake_browser.is_connected.return_value = True
        fake_browser.close = AsyncMock()

        async def _fake_connect(url: str, artifacts_dir: Optional[str] = None) -> MagicMock:
            captured["url"] = url
            captured["artifacts_dir"] = artifacts_dir
            return fake_browser

        fake_pw = MagicMock()
        fake_pw.chromium.connect_over_cdp = _fake_connect
        fake_pw.stop = AsyncMock()

        with patch.object(bm_module, "async_playwright") as ap:
            ap.return_value.start = AsyncMock(return_value=fake_pw)
            mgr = await BrowserManager.create(cdp_url="http://localhost:9222", download_dir=str(tmp_path))
        assert captured["url"] == "http://localhost:9222"
        assert captured["artifacts_dir"] == str(tmp_path)
        await mgr.cleanup()

    async def test_connect_sets_no_context_default_timeout(self, tmp_path: Path) -> None:
        """Connecting sets no default timeout on the context."""
        context = MagicMock()
        mgr = await _created_manager(context, str(tmp_path))

        try:
            context.set_default_timeout.assert_not_called()
        finally:
            await mgr.cleanup()


# ---------------------------------------------------------------------------
# Page Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.live
@skip_if_no_chrome
class TestNewPage:
    async def test_new_page(self, browser_manager: BrowserManager) -> None:
        target_id = await browser_manager.new_page()
        try:
            assert isinstance(target_id, str)
            assert target_id in browser_manager.tab_ids()
        finally:
            await browser_manager.close_page(target_id)

    async def test_multiple_pages(self, browser_manager: BrowserManager) -> None:
        tid1 = await browser_manager.new_page()
        tid2 = await browser_manager.new_page()
        try:
            assert tid1 != tid2
            assert tid1 in browser_manager.tab_ids()
            assert tid2 in browser_manager.tab_ids()
        finally:
            await browser_manager.close_page(tid1)
            await browser_manager.close_page(tid2)

    async def test_new_page_navigable(self, browser_manager: BrowserManager) -> None:
        target_id = await browser_manager.new_page()
        try:
            result = await browser_manager.navigate(target_id, "about:blank")
            assert result is True
        finally:
            await browser_manager.close_page(target_id)


@pytest.mark.live
@skip_if_no_chrome
class TestClosePage:
    async def test_close_page(self, browser_manager: BrowserManager) -> None:
        target_id = await browser_manager.new_page()
        assert target_id in browser_manager.tab_ids()
        assert await browser_manager.close_page(target_id) is True
        assert target_id not in browser_manager.tab_ids()

    async def test_close_page_unknown(self, browser_manager: BrowserManager) -> None:
        assert await browser_manager.close_page("nonexistent") is False

    async def test_close_page_twice(self, browser_manager: BrowserManager) -> None:
        target_id = await browser_manager.new_page()
        assert await browser_manager.close_page(target_id) is True
        assert await browser_manager.close_page(target_id) is False


class TestSolveRecaptchaGrid:
    """Tests for solving the reCAPTCHA picture challenge."""

    async def test_returns_the_challenger_token(self) -> None:
        mgr, mock_page, tab = await _make_manager()
        with patch.object(bm_module, "AsyncChallenger") as mock_challenger_cls:
            mock_challenger = mock_challenger_cls.return_value
            mock_challenger.solve_recaptcha = AsyncMock(return_value="token-123")

            token = await mgr.solve_recaptcha_grid(tab, click_timeout=_RECAPTCHA_CLICK_TIMEOUT_MS, retry_times=30, verbose=True)

        assert token == "token-123"
        mock_challenger_cls.assert_called_once_with(mock_page, click_timeout=_RECAPTCHA_CLICK_TIMEOUT_MS, retry_times=30, verbose=True)
        mock_challenger.solve_recaptcha.assert_awaited_once_with(skip_checkbox=True)

    async def test_solver_error_returns_none(self) -> None:
        mgr, _, tab = await _make_manager()
        with patch.object(bm_module, "AsyncChallenger") as mock_challenger_cls:
            mock_challenger = mock_challenger_cls.return_value
            mock_challenger.solve_recaptcha = AsyncMock(side_effect=RuntimeError("boom"))

            token = await mgr.solve_recaptcha_grid(tab, click_timeout=_RECAPTCHA_CLICK_TIMEOUT_MS, retry_times=30)

        assert token is None

    async def test_unknown_tab_returns_none(self) -> None:
        mgr, _, _tab = await _make_manager()
        with patch.object(bm_module, "AsyncChallenger") as mock_challenger_cls:
            token = await mgr.solve_recaptcha_grid("nonexistent", click_timeout=_RECAPTCHA_CLICK_TIMEOUT_MS, retry_times=30)

        assert token is None
        mock_challenger_cls.assert_not_called()


async def _tab_session_manager(tmp_path: Path) -> tuple[BrowserManager, AsyncMock, MagicMock, MagicMock]:
    """Make a connected manager whose context hands out one fake tab and one fake session."""
    page = _fake_page()
    session = _layout_session()
    context = _test_context(page, session)
    mgr = await _created_manager(context, str(tmp_path))
    return mgr, page, session, context


class TestTabSession:
    """Tests for the CDP session that keeps each tab at full speed."""

    async def test_new_page_sends_tab_session_commands(self, tmp_path: Path) -> None:
        """A new tab gets a session told to act like the foreground tab."""
        mgr, page, session, context = await _tab_session_manager(tmp_path)

        try:
            await mgr.new_page()

            context.new_cdp_session.assert_awaited_once_with(page)
            assert session.send.await_args_list == [
                call("Emulation.setFocusEmulationEnabled", {"enabled": True}),
                call("Page.setWebLifecycleState", {"state": "active"}),
            ]
        finally:
            await mgr.cleanup()

    async def test_new_page_survives_session_attach_failure(self, tmp_path: Path) -> None:
        """A tab still opens when its session cannot be attached."""
        mgr, _, _, context = await _tab_session_manager(tmp_path)
        context.new_cdp_session = AsyncMock(side_effect=Exception("attach failed"))

        try:
            target_id = await mgr.new_page()

            assert target_id in mgr.tab_ids()
        finally:
            await mgr.cleanup()

    async def test_new_page_survives_command_failure(self, tmp_path: Path) -> None:
        """A tab still opens when a session command fails."""
        mgr, _, session, _ = await _tab_session_manager(tmp_path)
        session.send = AsyncMock(side_effect=Exception("send failed"))

        try:
            target_id = await mgr.new_page()

            assert target_id in mgr.tab_ids()
            session.detach.assert_awaited_once()
        finally:
            await mgr.cleanup()

    async def test_close_page_without_session_skips_detach(self, tmp_path: Path) -> None:
        """Closing a tab with no session detaches nothing."""
        mgr, _, session, context = await _tab_session_manager(tmp_path)
        context.new_cdp_session = AsyncMock(side_effect=Exception("attach failed"))

        try:
            target_id = await mgr.new_page()
            assert await mgr.close_page(target_id) is True

            session.detach.assert_not_awaited()
        finally:
            await mgr.cleanup()

    @pytest.mark.parametrize(
        ("tab_closed", "detach_error", "detaches"),
        [
            pytest.param(False, None, True, id="open-tab"),
            pytest.param(False, Exception("detach failed"), True, id="detach-fails"),
            pytest.param(True, None, False, id="tab-already-closed"),
        ],
    )
    async def test_close_page_failure_lets_the_tab_go(
        self, tmp_path: Path, tab_closed: bool, detach_error: Optional[Exception], detaches: bool
    ) -> None:
        """A tab whose close fails is let go of, and its session is detached only while the tab is still open."""
        page = _lifecycle_page()
        page.evaluate.return_value = [_VIEWPORT_WIDTH, _VIEWPORT_HEIGHT]
        session = _layout_session()
        session.detach = AsyncMock(side_effect=detach_error)
        mgr = await _clamped_manager(tmp_path, page, session)

        try:
            target_id = await mgr.new_page()
            page.is_closed.return_value = tab_closed
            page.close = AsyncMock(side_effect=Exception("close failed"))
            assert await mgr.close_page(target_id) is False

            assert target_id not in mgr.tab_ids()
            assert session.detach.await_count == int(detaches)
        finally:
            await mgr.cleanup()

    async def test_second_manager_reads_no_viewport_for_a_shared_tab(self, tmp_path: Path) -> None:
        """A manager without a session for a shared tab reads no window."""
        page = _fake_page()
        mgr = await _clamped_manager(tmp_path, page, _layout_session())
        other = await _created_manager(_test_context(page, None, attach_session=False), str(tmp_path))

        try:
            tab = await mgr.new_page()
            other_tab = await other.new_page()
            viewport = await mgr.get_page_viewport(tab)
            assert viewport.client_width == _VIEWPORT_WIDTH
            assert await other.click_at_relative(other_tab, _CENTER_RATIO, _CENTER_RATIO) is False
        finally:
            await other.cleanup()
            await mgr.cleanup()

    @pytest.mark.parametrize(
        "closed_by_manager", [pytest.param(True, id="close-page"), pytest.param(False, id="close-event")]
    )
    async def test_closed_tab_leaves_its_session_alone(self, tmp_path: Path, closed_by_manager: bool) -> None:
        """A closed tab leaves the owned set and its session is not detached."""
        page = _lifecycle_page()
        session = _layout_session()
        mgr = await _clamped_manager(tmp_path, page, session)

        try:
            target_id = await mgr.new_page()
            await (mgr.close_page(target_id) if closed_by_manager else page.fire("close", page))
            await _settle_tasks()

            assert target_id not in mgr.tab_ids()
            session.detach.assert_not_awaited()
            assert await mgr.close_page(target_id) is False
        finally:
            await mgr.cleanup()


# ---------------------------------------------------------------------------
# Closing always finishes
# ---------------------------------------------------------------------------


async def _hanging_manager(tmp_path: Path) -> tuple[BrowserManager, str, MagicMock]:
    """Make a connected manager whose page and browser never finish closing."""

    async def _hang() -> None:
        await asyncio.Event().wait()

    hanging_page = MagicMock()
    hanging_page.is_closed = MagicMock(return_value=False)
    hanging_page.close = _hang
    session = _layout_session()
    context = MagicMock()
    context.new_page = AsyncMock(return_value=hanging_page)
    context.new_cdp_session = AsyncMock(return_value=session)
    fake_browser = MagicMock()
    fake_browser.contexts = [context]
    fake_browser.close = _hang
    fake_pw = MagicMock()
    fake_pw.chromium.connect_over_cdp = AsyncMock(return_value=fake_browser)
    fake_pw.stop = _hang
    with patch.object(bm_module, "async_playwright") as ap:
        ap.return_value.start = AsyncMock(return_value=fake_pw)
        mgr = await BrowserManager.create(cdp_url="http://localhost:9222", download_dir=str(tmp_path))
    target_id = await mgr.new_page()
    return mgr, target_id, session


class TestCleanupBounded:
    """Tests that closing a page and cleaning up finish even when the browser never answers."""

    async def test_close_page_hang_is_bounded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bm_module, "_CLEANUP_STEP_TIMEOUT_S", _TEST_CLEANUP_STEP_TIMEOUT_S)
        mgr, target_id, session = await _hanging_manager(tmp_path)

        t0 = time.monotonic()
        assert await mgr.close_page(target_id) is False
        assert time.monotonic() - t0 < 10, "close_page hung past its bound"
        assert target_id not in mgr.tab_ids()
        session.detach.assert_awaited_once()

    async def test_cleanup_hang_is_bounded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bm_module, "_CLEANUP_STEP_TIMEOUT_S", _TEST_CLEANUP_STEP_TIMEOUT_S)
        mgr, _, _ = await _hanging_manager(tmp_path)

        t0 = time.monotonic()
        await mgr.cleanup(verbose=False)
        elapsed = time.monotonic() - t0
        assert elapsed < 5, f"cleanup hung: took {elapsed:.1f}s"
        assert mgr.tab_ids() == []


# ---------------------------------------------------------------------------
# Download Detection
# ---------------------------------------------------------------------------


@pytest.mark.live
@skip_if_no_chrome
class TestWaitForDownload:
    """Tests for waiting on a file download."""

    async def test_url_no_download(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """Opening a page that downloads nothing comes back right away."""
        tid = await browser_manager.new_page()
        try:
            start = __import__("time").time()
            result = await browser_manager.wait_for_download(
                target_id=tid,
                url=test_server_url,
                timeout=10,
            )
            elapsed = __import__("time").time() - start
            assert result["downloaded"] is False
            assert result == {"downloaded": False}
            assert elapsed < 5, f"Took {elapsed:.1f}s, should return immediately on no download"
        finally:
            await browser_manager.close_page(tid)

    async def test_url_triggers_download(self, browser_manager: BrowserManager, tmp_path: Path, test_server_url: str) -> None:
        """Opening a download link saves the file where asked."""
        tid = await browser_manager.new_page()
        try:
            save_path = str(tmp_path / "test.bin")
            result = await browser_manager.wait_for_download(
                target_id=tid,
                url=test_server_url + "/download/test.bin",
                save_path=save_path,
                timeout=10,
            )
            assert result["downloaded"] is True
            assert result["path"] == save_path
            _assert_download_file(save_path)
        finally:
            await browser_manager.close_page(tid)

    async def test_action_triggers_download(self, browser_manager: BrowserManager, tmp_path: Path, test_server_url: str) -> None:
        """A click that starts a download saves the file where asked."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url)

            save_path = str(tmp_path / "clicked.bin")
            # Give the page a moment to settle.
            time.sleep(1)

            async def _click_download() -> bool:
                await browser_manager.click(tid, "#download_link")
                return True

            result = await browser_manager.wait_for_download(
                target_id=tid,
                action=_click_download,
                save_path=save_path,
                timeout=30,
            )
            assert result["downloaded"] is True
            _assert_download_file(save_path)
        finally:
            await browser_manager.close_page(tid)

    async def test_action_no_download(self, browser_manager: BrowserManager) -> None:
        """An action that starts no download comes back right away."""
        tid = await browser_manager.new_page()
        try:
            start = __import__("time").time()

            async def _no_download() -> bool:
                return False

            result = await browser_manager.wait_for_download(
                target_id=tid,
                action=_no_download,
                timeout=10,
            )
            elapsed = __import__("time").time() - start
            assert result["downloaded"] is False
            assert elapsed < 2, f"Took {elapsed:.1f}s, should return immediately on False"
        finally:
            await browser_manager.close_page(tid)

    async def test_url_navigation_failure_raises(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """A page that fails to open raises an error."""
        tid = await browser_manager.new_page()
        try:
            url = test_server_url + "/status/403"
            with pytest.raises(RuntimeError, match="Navigation failed"):
                await browser_manager.wait_for_download(
                    target_id=tid,
                    url=url,
                    timeout=10,
                )
        finally:
            await browser_manager.close_page(tid)

    async def test_download_dir_created_on_connect(self, tmp_path: Path) -> None:
        """Connecting makes the download folder."""
        fresh_dir = str(tmp_path / "fresh_dl")
        mgr = await BrowserManager.create(download_dir=fresh_dir)
        try:
            assert os.path.isdir(fresh_dir)
        finally:
            await mgr.cleanup()


class TestDownloadEventClaim:
    """Tests for claiming a download by its completion event on the tab's CDP session."""

    async def test_event_names_the_file_it_claims(self, tmp_path: Path) -> None:
        """The finished file is the one the completion event names, and it moves to the save path."""
        (tmp_path / "stale.bin").write_bytes(b"old")
        pages = {"tab1": AsyncMock()}
        mgr, ids, sessions = await _manager_with_sessions(tmp_path, pages)
        session = sessions["tab1"]

        async def _click() -> bool:
            (tmp_path / "other.bin").write_bytes(b"someone else")
            (tmp_path / "fresh.bin").write_bytes(b"new")
            await session.emit([
                ("Browser.downloadWillBegin", {"guid": "g1", "suggestedFilename": "fresh.bin"}),
                ("Browser.downloadProgress", {"guid": "g1", "state": "completed", "filePath": str(tmp_path / "fresh.bin")}),
            ])
            return True

        save_path = str(tmp_path / "out" / "fresh.bin")
        result = await mgr.wait_for_download(ids[0], action=_click, save_path=save_path, timeout=5)
        assert result["downloaded"] is True
        assert result["path"] == save_path
        assert result["filename"] == "fresh.bin"
        assert os.path.getsize(save_path) > 0
        assert not (tmp_path / "fresh.bin").exists()
        assert (tmp_path / "stale.bin").read_bytes() == b"old"
        assert (tmp_path / "other.bin").read_bytes() == b"someone else"
        session.send.assert_any_call(
            "Browser.setDownloadBehavior",
            {"behavior": "allowAndName", "downloadPath": str(tmp_path), "eventsEnabled": True},
        )
        assert not session.detach.called

    @pytest.mark.timeout(30)
    async def test_other_tabs_downloads_are_not_claimed(self, tmp_path: Path) -> None:
        """Two waits at once each claim only the file their own tab downloads."""
        pages = {"tab1": AsyncMock(), "tab2": AsyncMock()}
        mgr, ids, sessions = await _manager_with_sessions(tmp_path, pages)
        tab1, tab2 = ids
        session_b = sessions["tab2"]

        # Tab one's wait ends before tab two's file lands.
        first = asyncio.create_task(mgr.wait_for_download(tab1, action=AsyncMock(return_value=True), timeout=_SHORT_DOWNLOAD_WAIT_S))

        async def _click() -> bool:
            await asyncio.wait({first})
            (tmp_path / "b.bin").write_bytes(b"tab two")
            await session_b.emit([
                ("Browser.downloadWillBegin", {"guid": "g2", "suggestedFilename": "b.bin"}),
                ("Browser.downloadProgress", {"guid": "g2", "state": "completed", "filePath": str(tmp_path / "b.bin")}),
            ])
            return True

        result1, result2 = await asyncio.gather(
            first,
            mgr.wait_for_download(tab2, action=_click, save_path=str(tmp_path / "out" / "b.bin"), timeout=5),
        )
        assert result1 == {"downloaded": False}
        assert result2["downloaded"] is True
        assert result2["path"] == str(tmp_path / "out" / "b.bin")

    async def test_completion_without_begin_is_not_claimed(self, tmp_path: Path) -> None:
        """A completion for a download that never began is ignored."""
        pages = {"tab1": AsyncMock()}
        mgr, ids, sessions = await _manager_with_sessions(tmp_path, pages)
        session = sessions["tab1"]

        async def _click() -> bool:
            await session.emit([
                ("Browser.downloadProgress", {"guid": "ghost", "state": "completed", "filePath": str(tmp_path / "ghost.bin")}),
            ])
            return True

        result = await mgr.wait_for_download(ids[0], action=_click, timeout=_SHORT_DOWNLOAD_WAIT_S)
        assert result == {"downloaded": False}

    async def test_no_download_returns_immediately(self, tmp_path: Path) -> None:
        """An action that starts no download comes back right away with the session kept."""
        pages = {"tab1": AsyncMock()}
        mgr, ids, sessions = await _manager_with_sessions(tmp_path, pages)
        session = sessions["tab1"]

        async def _noop() -> bool:
            return False

        start = time.monotonic()
        result = await mgr.wait_for_download(ids[0], action=_noop, timeout=5)
        assert result == {"downloaded": False}
        assert time.monotonic() - start < 2
        assert not session.detach.called

    async def test_timeout_fallback_claims_single_new_file(self, tmp_path: Path) -> None:
        """When no event arrives, one new file in the folder is still claimed."""
        pages = {"tab1": AsyncMock()}
        mgr, ids, _ = await _manager_with_sessions(tmp_path, pages)

        async def _click() -> bool:
            (tmp_path / "dl.bin").write_bytes(b"data")
            return True

        save_path = str(tmp_path / "out.bin")
        result = await mgr.wait_for_download(ids[0], action=_click, save_path=save_path, timeout=_SHORT_DOWNLOAD_WAIT_S)
        assert result["downloaded"] is True
        assert result["path"] == save_path
        assert not (tmp_path / "dl.bin").exists()

    async def test_timeout_fallback_claims_nothing_when_multiple_files_land(self, tmp_path: Path) -> None:
        """When no event arrives and two new files show up, nothing is claimed."""
        pages = {"tab1": AsyncMock()}
        mgr, ids, _ = await _manager_with_sessions(tmp_path, pages)

        async def _click() -> bool:
            (tmp_path / "one.bin").write_bytes(b"one")
            (tmp_path / "two.bin").write_bytes(b"two")
            return True

        result = await mgr.wait_for_download(ids[0], action=_click, timeout=_SHORT_DOWNLOAD_WAIT_S)
        assert result == {"downloaded": False}
        assert (tmp_path / "one.bin").exists()
        assert (tmp_path / "two.bin").exists()

    async def test_download_handlers_are_removed_after_a_claim(self, tmp_path: Path) -> None:
        """Claiming a download leaves no stray handlers on the tab's session."""
        pages = {"tab1": AsyncMock()}
        mgr, ids, sessions = await _manager_with_sessions(tmp_path, pages)
        session = sessions["tab1"]

        async def _click() -> bool:
            (tmp_path / "x.bin").write_bytes(b"data")
            await session.emit([
                ("Browser.downloadWillBegin", {"guid": "g", "suggestedFilename": "x.bin"}),
                ("Browser.downloadProgress", {"guid": "g", "state": "completed", "filePath": str(tmp_path / "x.bin")}),
            ])
            return True

        result = await mgr.wait_for_download(ids[0], action=_click, timeout=5)
        assert result["downloaded"] is True
        assert session.listeners.get("Browser.downloadWillBegin", []) == []
        assert session.listeners.get("Browser.downloadProgress", []) == []

    async def test_claim_returns_early_when_download_setup_fails(self, tmp_path: Path) -> None:
        """When the download setup fails, the action never runs and nothing is claimed."""
        pages = {"tab1": AsyncMock()}
        mgr, ids, sessions = await _manager_with_sessions(tmp_path, pages)
        session = sessions["tab1"]

        async def _send(method: str, _params: Optional[dict] = None) -> dict:
            if method == "Browser.setDownloadBehavior":
                raise Exception("setDownloadBehavior refused")
            return {}

        session.send = AsyncMock(side_effect=_send)
        called: list[bool] = []

        async def _action() -> bool:
            called.append(True)
            return True

        result = await mgr.wait_for_download(ids[0], action=_action, timeout=5)
        assert result == {"downloaded": False}
        assert called == []


class TestDownloadWaitConcurrency:
    """Tests for download waits that do not freeze the rest of the program."""

    async def test_wait_after_action_error(self, tmp_path: Path) -> None:
        """A later wait still works after one blows up."""
        mgr, _, tab = await _make_manager(download_dir=str(tmp_path))

        async def _boom() -> bool:
            raise RuntimeError("action exploded")

        with pytest.raises(RuntimeError, match="action exploded"):
            await mgr.wait_for_download(tab, action=_boom, timeout=5)

        async def _ok() -> bool:
            return False

        result = await mgr.wait_for_download(tab, action=_ok, timeout=5)
        assert result == {"downloaded": False}

    async def test_file_scan_does_not_block_event_loop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scanning the download folder for new files does not freeze the rest of the program."""
        loop_thread = threading.get_ident()
        scan_threads: list[int] = []
        real_listdir = os.listdir

        def _slow_listdir(path: str) -> list[str]:
            scan_threads.append(threading.get_ident())
            time.sleep(_SLOW_SCAN_BLOCK_S)
            return real_listdir(path)

        monkeypatch.setattr(os, "listdir", _slow_listdir)
        mgr, _, tab = await _make_manager(download_dir=str(tmp_path))

        async def _trigger() -> bool:
            return True

        result = await mgr.wait_for_download(tab, action=_trigger, timeout=0.4)
        assert result == {"downloaded": False}
        assert len(scan_threads) >= 2
        assert all(t != loop_thread for t in scan_threads), "download poll ran file I/O on the event loop"

    async def test_finalize_does_not_block_event_loop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Moving the finished file does not freeze the rest of the program."""
        loop_thread = threading.get_ident()
        move_threads: list[int] = []
        real_move = shutil.move

        def _slow_move(src: str, dst: str) -> None:
            move_threads.append(threading.get_ident())
            time.sleep(_SLOW_MOVE_BLOCK_S)
            return real_move(src, dst)

        monkeypatch.setattr(shutil, "move", _slow_move)
        mgr, _, tab = await _make_manager(download_dir=str(tmp_path))

        async def _trigger() -> bool:
            (tmp_path / "dl.bin").write_bytes(b"data")
            return True

        save_path = str(tmp_path / "out" / "dl.bin")
        result = await mgr.wait_for_download(tab, action=_trigger, save_path=save_path, timeout=_SHORT_DOWNLOAD_WAIT_S)
        assert result["downloaded"] is True
        assert result["path"] == save_path
        assert os.path.exists(save_path)
        assert move_threads, "file move never ran"
        assert all(t != loop_thread for t in move_threads), "finalize ran file I/O on the event loop"


@pytest.mark.live
@skip_if_no_chrome
class TestLiveNavigation:
    """Live tests that need a real browser."""

    async def test_screenshot_saves_jpg(self, browser_manager: BrowserManager, tmp_path: Path, test_server_url: str) -> None:
        """Taking a picture saves a real image file."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url)
            path = str(tmp_path / "screenshot.jpg")
            result = await browser_manager.screenshot(tid, path, full_page=True)
            assert result == path
            assert os.path.exists(path)
            with open(path, "rb") as f:
                header = f.read(3)
            assert header == b"\xff\xd8\xff"
        finally:
            await browser_manager.close_page(tid)

    async def test_pdf_saves_valid_pdf(self, browser_manager: BrowserManager, tmp_path: Path, test_server_url: str) -> None:
        """Saving as PDF writes a real PDF file."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url)
            path = str(tmp_path / "page.pdf")
            result = await browser_manager.pdf(tid, path)
            assert result == path
            assert os.path.exists(path)
            with open(path, "rb") as f:
                header = f.read(5)
            assert header == b"%PDF-"
        finally:
            await browser_manager.close_page(tid)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


@pytest.mark.live
@skip_if_no_chrome
class TestEvaluate:
    """Tests for running code on the page."""

    async def test_evaluate_returns_result(self, browser_manager: BrowserManager) -> None:
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, "about:blank")
            result = await browser_manager.evaluate(tid, "1 + 1")
            assert result == 2
        finally:
            await browser_manager.close_page(tid)

    async def test_evaluate_document_content_type(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url)
            result = await browser_manager.evaluate(tid, "document.contentType")
            assert isinstance(result, str)
        finally:
            await browser_manager.close_page(tid)

    async def test_evaluate_returns_none_on_error(self, browser_manager: BrowserManager) -> None:
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, "about:blank")
            result = await browser_manager.evaluate(tid, "(() => { throw new Error('test'); })()")
            assert result is None
        finally:
            await browser_manager.close_page(tid)




@pytest.mark.live
@skip_if_no_chrome
class TestNavigate:
    """Tests for opening a page."""

    async def test_navigate_success(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        tid = await browser_manager.new_page()
        try:
            result = await browser_manager.navigate(tid, test_server_url)
            assert result is True
        finally:
            await browser_manager.close_page(tid)

    async def test_navigate_invalid_url(self, browser_manager: BrowserManager) -> None:
        tid = await browser_manager.new_page()
        try:
            result = await browser_manager.navigate(tid, "http://localhost:1")
            assert result is False
        finally:
            await browser_manager.close_page(tid)

    async def test_navigate_custom_wait_until(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        tid = await browser_manager.new_page()
        try:
            result = await browser_manager.navigate(tid, test_server_url, wait_until="load")
            assert result is True
        finally:
            await browser_manager.close_page(tid)


async def test_navigate_returns_false_on_404() -> None:
    mgr, mock_page, tab = await _make_manager()
    response = MagicMock()
    response.status = 404
    mock_page.goto = AsyncMock(return_value=response)
    assert await mgr.navigate(tab, "http://x/missing") is False


async def test_navigate_returns_true_on_200() -> None:
    mgr, mock_page, tab = await _make_manager()
    response = MagicMock()
    response.status = 200
    mock_page.goto = AsyncMock(return_value=response)
    assert await mgr.navigate(tab, "http://x/") is True


async def test_navigate_returns_true_on_null_response() -> None:
    mgr, mock_page, tab = await _make_manager()
    mock_page.goto = AsyncMock(return_value=None)
    assert await mgr.navigate(tab, "http://x/") is True


@pytest.mark.live
@skip_if_no_chrome
class TestWait:
    """Tests for waiting on a page and on an element."""

    async def test_wait_for_load_success(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url)
            result = await browser_manager.wait_for_load(tid)
            assert result is True
        finally:
            await browser_manager.close_page(tid)

    async def test_wait_for_selector_success(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url)
            result = await browser_manager.wait_for_selector(tid, "h1")
            assert result is True
        finally:
            await browser_manager.close_page(tid)

    async def test_wait_for_selector_timeout(self, browser_manager: BrowserManager) -> None:
        tid = await browser_manager.new_page()
        try:
            result = await browser_manager.wait_for_selector(tid, "#nonexistent-xyz123", timeout=100)
            assert result is False
        finally:
            await browser_manager.close_page(tid)


@pytest.mark.live
@skip_if_no_chrome
class TestSnapshot:
    """Tests for taking a snapshot of a page."""

    async def test_snapshot_returns_valid_structure(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """A snapshot of a real page has page text and a tree."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url)
            result = await browser_manager.snapshot(tid)
            assert result.html is not None and len(result.html) > 0, "snapshot() should return non-empty HTML for valid page"
            tree = await result.filtered_tree()
            assert isinstance(tree, dict) and "role" in tree, f"filtered_tree() should return a valid tree dict (got {type(tree)})"
        finally:
            await browser_manager.close_page(tid)

    async def test_snapshot_invalid_target(self, browser_manager: BrowserManager) -> None:
        """A snapshot of a tab that is not there gives back nothing."""
        result = await browser_manager.snapshot("nonexistent")
        assert result is None




# ---------------------------------------------------------------------------
# Page address and title
# ---------------------------------------------------------------------------


@pytest.mark.live
@skip_if_no_chrome
class TestUrl:
    """Tests for reading the page address."""

    async def test_url_returns_correct_url(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """The address matches the page that was opened."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url)
            url = browser_manager.url(tid)
            assert url == test_server_url + "/"
        finally:
            await browser_manager.close_page(tid)

    async def test_url_about_blank(self, browser_manager: BrowserManager) -> None:
        """A fresh page reports itself as blank."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, "about:blank")
            url = browser_manager.url(tid)
            assert url == "about:blank"
        finally:
            await browser_manager.close_page(tid)

    def test_url_unknown_target_returns_none(self, browser_manager: BrowserManager) -> None:
        """A tab that is not there has no address."""
        result = browser_manager.url("nonexistent")
        assert result is None

@pytest.mark.live
@skip_if_no_chrome
class TestTitle:
    """Tests for reading the page title."""

    async def test_title_returns_correct_title(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """The title matches the page that was opened."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url)
            title = await browser_manager.title(tid)
            assert isinstance(title, str) and "Test Page" in title
        finally:
            await browser_manager.close_page(tid)

    async def test_title_about_blank(self, browser_manager: BrowserManager) -> None:
        """A blank page has an empty title."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, "about:blank")
            title = await browser_manager.title(tid)
            assert title == ""
        finally:
            await browser_manager.close_page(tid)

    async def test_title_unknown_target_returns_none(self, browser_manager: BrowserManager) -> None:
        """A tab that is not there has no title."""
        result = await browser_manager.title("nonexistent")
        assert result is None

# ---------------------------------------------------------------------------
# Typing text and picking files
# ---------------------------------------------------------------------------


class TestTypeText:
    """Tests for typing text into an element."""

    async def test_type_text_success(self: Self) -> None:
        """Typing text into an element gives back true."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.type_text(tab, "#input", "hello world")
        assert result is True
        mock_page.fill.assert_called_once()
        assert mock_page.fill.call_args[0] == ("#input", "hello world")
        assert "timeout" in mock_page.fill.call_args[1]

    async def test_type_text_with_submit(self: Self) -> None:
        """Typing with submit turned on presses Enter after the text."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.type_text(tab, "#search", "query", submit=True)
        assert result is True
        mock_page.fill.assert_called_once()
        assert mock_page.fill.call_args[0] == ("#search", "query")
        assert "timeout" in mock_page.fill.call_args[1]
        mock_page.press.assert_called_once()
        assert mock_page.press.call_args[0] == ("#search", "Enter")
        assert "timeout" in mock_page.press.call_args[1]

    async def test_type_text_without_submit_no_enter(self: Self) -> None:
        """Typing without submit does not press Enter."""
        mgr, mock_page, tab = await _make_manager()

        await mgr.type_text(tab, "#input", "text", submit=False)
        mock_page.press.assert_not_called()

    async def test_type_text_unknown_target(self: Self) -> None:
        """Typing into a tab that is not there gives back false."""
        mgr = await _empty_manager()

        result = await mgr.type_text("nonexistent", "#input", "text")
        assert result is False

    async def test_type_text_playwright_error(self: Self) -> None:
        """Typing that blows up gives back false."""
        mgr, _, tab = await _make_manager(fill=AsyncMock(side_effect=Exception("Element not found")))

        result = await mgr.type_text(tab, "#missing", "text")
        assert result is False

    async def test_type_text_empty_string(self: Self) -> None:
        """Typing empty text works."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.type_text(tab, "#input", "")
        assert result is True
        mock_page.fill.assert_called_once()
        assert mock_page.fill.call_args[0] == ("#input", "")
        assert "timeout" in mock_page.fill.call_args[1]

    @pytest.mark.parametrize("selector", ["#input", "textarea.prompt", "[data-ref=e121]", "div[contenteditable=true]"])
    async def test_type_text_various_selectors(self, selector: str) -> None:
        """Typing works for many kinds of element paths."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.type_text(tab, selector, "test")
        assert result is True
        mock_page.fill.assert_called_once()
        assert mock_page.fill.call_args[0] == (selector, "test")
        assert "timeout" in mock_page.fill.call_args[1]


class TestHumanConfigForwarding:
    """Tests that per call aim settings reach the page only when they are given."""

    _AIM: dict = {"click_input_x_range": (0.4, _AIM_X_MAX)}

    @pytest.mark.parametrize("human_config", [_AIM, None])
    async def test_click_forwards_human_config(self, human_config: Optional[dict]) -> None:
        """A click passes aim settings to the page only when they are given."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.click(tab, "#el", human_config=human_config)

        assert result is True
        if human_config is None:
            assert "human_config" not in mock_page.click.call_args.kwargs
        else:
            assert mock_page.click.call_args.kwargs["human_config"] is human_config

    @pytest.mark.parametrize("human_config", [_AIM, None])
    async def test_type_text_forwards_human_config(self, human_config: Optional[dict]) -> None:
        """Typing passes aim settings to the page only when they are given."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.type_text(tab, "#input", "text", human_config=human_config)

        assert result is True
        if human_config is None:
            assert "human_config" not in mock_page.fill.call_args.kwargs
        else:
            assert mock_page.fill.call_args.kwargs["human_config"] is human_config


class TestUploadFiles:
    """Tests for putting files into a file box."""

    async def test_upload_files_success(self: Self) -> None:
        """Putting files in gives back true."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.upload_files(tab, "input[type=file]", ["/tmp/img1.jpg", "/tmp/img2.png"])
        assert result is True
        mock_page.set_input_files.assert_called_once_with("input[type=file]", ["/tmp/img1.jpg", "/tmp/img2.png"])

    async def test_upload_files_single_file(self: Self) -> None:
        """Putting one file in works."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.upload_files(tab, "#file-input", ["/tmp/single.jpg"])
        assert result is True
        mock_page.set_input_files.assert_called_once_with("#file-input", ["/tmp/single.jpg"])

    async def test_upload_files_unknown_target(self: Self) -> None:
        """Putting files into a tab that is not there gives back false."""
        mgr = await _empty_manager()

        result = await mgr.upload_files("nonexistent", "#input", ["/tmp/file.jpg"])
        assert result is False

    async def test_upload_files_playwright_error(self: Self) -> None:
        """A file that fails to go in gives back false."""
        mgr, mock_page, tab = await _make_manager()
        mock_page.set_input_files.side_effect = Exception("No file input found")

        result = await mgr.upload_files(tab, "#missing", ["/tmp/file.jpg"])
        assert result is False

    async def test_upload_files_empty_paths(self: Self) -> None:
        """Putting in no files works."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.upload_files(tab, "#file-input", [])
        assert result is True
        mock_page.set_input_files.assert_called_once_with("#file-input", [])



# ---------------------------------------------------------------------------
# Waiting for the page to go quiet before a snapshot
# ---------------------------------------------------------------------------


class TestSnapshotNetworkIdleWait:
    """Tests that a snapshot waits for the page to go quiet first."""

    async def test_snapshot_calls_wait_for_load_state_networkidle(self: Self) -> None:
        """A snapshot waits for the page to go quiet before reading it."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.snapshot(tab)

        mock_page.wait_for_load_state.assert_called_once_with("networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MS)
        assert isinstance(result, PageSnapshot)

    async def test_snapshot_waits_before_content(self: Self) -> None:
        """The wait happens before the page text is read."""
        mgr, mock_page, tab = await _make_manager()
        call_order = []
        mock_page.wait_for_load_state = MagicMock(side_effect=lambda *_, **__: call_order.append("wait"))
        original_content = mock_page.content
        mock_page.content = MagicMock(side_effect=lambda: (call_order.append("content"), original_content())[1])

        await mgr.snapshot(tab)

        assert call_order == ["wait", "content"], f"Expected wait then content, got {call_order}"

    async def test_snapshot_timeout_5_seconds(self: Self) -> None:
        """The wait gives up after 5 seconds."""
        mgr, mock_page, tab = await _make_manager()

        await mgr.snapshot(tab)

        call_args = mock_page.wait_for_load_state.call_args
        timeout = call_args[1].get("timeout") or (call_args[0][1] if len(call_args[0]) > 1 else None)
        assert timeout == _NETWORK_IDLE_TIMEOUT_MS, f"Expected timeout=_NETWORK_IDLE_TIMEOUT_MS, got {call_args}"

    async def test_snapshot_returns_page_snapshot_on_wait_success(self: Self) -> None:
        """A page that goes quiet gives back a full snapshot."""
        mgr, _, tab = await _make_manager()

        result = await mgr.snapshot(tab)

        assert isinstance(result, PageSnapshot)
        assert result.html == "<html><body>test</body></html>"

    async def test_snapshot_proceeds_on_timeout(self: Self) -> None:
        """A page that never goes quiet still gives back a snapshot."""
        mgr, mock_page, tab = await _make_manager()
        await _assert_snapshot_partial_on_timeout(mgr, mock_page, tab)

    async def test_snapshot_proceeds_on_generic_wait_error(self: Self) -> None:
        """A wait that blows up still gives back a snapshot."""
        mgr, mock_page, tab = await _make_manager()
        mock_page.wait_for_load_state.side_effect = RuntimeError("Page navigated during wait")

        result = await mgr.snapshot(tab)

        assert result.html is not None and len(result.html) > 0, "snapshot() should return partial PageSnapshot with HTML on wait error"

    async def test_snapshot_invalid_target_skips_wait(self: Self) -> None:
        """A snapshot of a tab that is not there gives back nothing."""
        mgr = await _empty_manager()

        result = await mgr.snapshot("nonexistent")

        assert result is None

    async def test_snapshot_page_already_networkidle(self: Self) -> None:
        """A page that is already quiet gives back a full snapshot."""
        mgr, _, tab = await _make_manager()

        result = await mgr.snapshot(tab)

        assert isinstance(result, PageSnapshot), \
            "Already-idle page should get a full snapshot without delay"

# ---------------------------------------------------------------------------
# Picking options and pressing keys
# ---------------------------------------------------------------------------


class TestBrowserManagerSelectOption:
    """Tests for picking an option in a drop down."""

    async def test_select_option_by_value(self: Self) -> None:
        """Picking by value picks that value on the page."""
        mgr, mock_page, tab = await _make_manager()
        mock_locator = _mock_locator(mock_page)

        result = await mgr.select_option(tab, "select#color", value="red")
        assert result is True
        mock_page.locator.assert_any_call("select#color")
        mock_locator.first.select_option.assert_called_once_with(force=True, value="red")

    async def test_select_option_by_index(self: Self) -> None:
        """Picking by position picks that position on the page."""
        mgr, mock_page, tab = await _make_manager()
        mock_locator = _mock_locator(mock_page)

        result = await mgr.select_option(tab, "select#size", index=2)
        assert result is True
        mock_locator.first.select_option.assert_called_once_with(force=True, index=2)

    async def test_select_option_by_label(self: Self) -> None:
        """Picking by label picks that label on the page."""
        mgr, mock_page, tab = await _make_manager()
        mock_locator = _mock_locator(mock_page)

        result = await mgr.select_option(tab, "select#option", label="Option 1")
        assert result is True
        mock_locator.first.select_option.assert_called_once_with(force=True, label="Option 1")

    async def test_select_option_no_params_fails(self: Self) -> None:
        """Picking with nothing to pick by gives back false."""
        mgr, mock_page, tab = await _make_manager()
        _mock_locator(mock_page)

        result = await mgr.select_option(tab, "select#test")
        assert result is False

    async def test_select_option_unknown_target(self: Self) -> None:
        """Picking in a tab that is not there gives back false."""
        mgr, _, _tab = await _make_manager()

        result = await mgr.select_option("nonexistent", "select#test", value="x")
        assert result is False

    async def test_select_option_playwright_error(self: Self) -> None:
        """A pick that blows up gives back false."""
        mgr, mock_page, tab = await _make_manager()
        mock_locator = _mock_locator(mock_page)
        mock_locator.first.select_option.side_effect = Exception("Element not found")

        result = await mgr.select_option(tab, "select#missing", value="x")
        assert result is False

    async def test_select_option_priority_value_over_index(self: Self) -> None:
        """When both a value and a position are given, the value wins."""
        mgr, mock_page, tab = await _make_manager()
        mock_locator = _mock_locator(mock_page)

        result = await mgr.select_option(tab, "select#test", value="red", index=0)
        assert result is True
        mock_locator.first.select_option.assert_called_once_with(force=True, value="red")


class TestBrowserManagerPressKey:
    """Tests for pressing a key."""

    async def test_press_key_enter(self: Self) -> None:
        """Pressing a key with none named presses Enter."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.press_key(tab)
        assert result is True
        mock_page.keyboard.press.assert_called_once_with("Enter")

    async def test_press_key_custom_key(self: Self) -> None:
        """A named key is the one that gets pressed."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.press_key(tab, key="Tab")
        assert result is True
        mock_page.keyboard.press.assert_called_once_with("Tab")

    async def test_press_key_escape(self: Self) -> None:
        """Escape gets pressed when it is named."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.press_key(tab, key="Escape")
        assert result is True
        mock_page.keyboard.press.assert_called_once_with("Escape")

    async def test_press_key_unknown_target(self: Self) -> None:
        """Pressing a key in a tab that is not there gives back false."""
        mgr = await _empty_manager()

        result = await mgr.press_key("nonexistent", key="Enter")
        assert result is False

    async def test_press_key_playwright_error(self: Self) -> None:
        """A key press that blows up gives back false."""
        mgr, mock_page, tab = await _make_manager()
        mock_page.keyboard.press.side_effect = Exception("Keyboard not available")

        result = await mgr.press_key(tab, key="Enter")
        assert result is False

    @pytest.mark.parametrize("key", ["Enter", "Tab", "Escape", "Backspace", "ArrowUp", "ArrowDown", "Delete"])
    async def test_press_key_various_keys(self, key: str) -> None:
        """Many different keys can be pressed."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.press_key(tab, key=key)
        assert result is True
        mock_page.keyboard.press.assert_called_once_with(key)


class TestBrowserManagerClickAt:
    """Tests for clicking a spot on the page."""

    async def test_click_at_within_viewport(self: Self) -> None:
        """A spot already in view is clicked where it sits in the window."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.click_at(tab, 100, 200)
        assert result is True
        mock_page.mouse.click.assert_called_once_with(100, 200)

    @pytest.mark.fast_sleeps
    async def test_click_at_requires_scroll(self: Self) -> None:
        """A spot below the window is scrolled to and then clicked."""
        mgr, mock_page, tab = await _make_manager(session=_layout_session(content_width=_VIEWPORT_WIDTH + _SCROLL_EXTRA_WIDTH, content_height=_VIEWPORT_HEIGHT + _SCROLL_EXTRA_HEIGHT_SHORT))

        with patch("time.sleep"):
            result = await mgr.click_at(tab, 100, 2000)
        assert result is True
        mock_page.mouse.wheel.assert_called()
        mock_page.mouse.click.assert_called()

    async def test_click_at_unknown_target(self: Self) -> None:
        """Clicking a spot in a tab that is not there gives back false."""
        mgr = await _empty_manager()

        result = await mgr.click_at("nonexistent", 100, 200)
        assert result is False

    async def test_click_at_negative_coords(self: Self) -> None:
        """A spot with a negative number gives back false."""
        mgr, _, tab = await _make_manager()

        result = await mgr.click_at(tab, -10, 200)
        assert result is False

    async def test_click_at_viewport_read_failure(self: Self) -> None:
        """A click gives back false when the window cannot be read."""
        session = _layout_session()
        session.send.side_effect = Exception("CDP down")
        mgr, _, tab = await _make_manager(session=session)

        result = await mgr.click_at(tab, 100, 200)
        assert result is False

    @pytest.mark.fast_sleeps
    async def test_click_at_delegates_scroll_to(self: Self) -> None:
        """A spot far below the window is scrolled to and then clicked."""
        mgr, mock_page, tab = await _make_manager(session=_layout_session(content_width=_VIEWPORT_WIDTH + _SCROLL_EXTRA_WIDTH, content_height=_VIEWPORT_HEIGHT + _SCROLL_EXTRA_HEIGHT_LONG))

        with patch("time.sleep"):
            await mgr.click_at(tab, 500, _BEYOND_VIEWPORT_Y)

        mock_page.mouse.wheel.assert_called()
        mock_page.mouse.click.assert_called()


# ---------------------------------------------------------------------------
# Scrolling an element into view
# ---------------------------------------------------------------------------


class TestScrollToViewUnit:
    """Tests for scrolling an element into view.

    The scroll counts as done when the middle of the element sits inside the
    window. It gives back false when the element cannot be found or when the
    middle is still outside at the end.
    """

    def _bbox(self, x: float, y: float) -> dict:
        return {"x": x, "y": y, "width": 50, "height": 30}

    async def test_already_in_viewport_no_scroll(self: Self) -> None:
        """An element already in view is left where it is."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.scroll_to_view(tab, "#el")

        assert result is True
        mock_page.mouse.wheel.assert_not_called()
        mock_page.mouse.move.assert_not_called()

    @pytest.mark.fast_sleeps
    async def test_below_fold_converges(self: Self) -> None:
        """An element below the window is scrolled into view in capped steps."""
        mgr, mock_page, tab = await _make_manager()
        mock_locator = mock_page.locator.return_value.first
        mock_locator.bounding_box = AsyncMock(side_effect=[
            self._bbox(100, _BELOW_FOLD_Y),
            self._bbox(100, 1700),
            self._bbox(100, 900),
        ])

        result = await mgr.scroll_to_view(tab, "#el")

        assert result is True
        mock_page.mouse.move.assert_any_call(125.0, 1070.0)
        assert mock_page.mouse.wheel.call_count > 0
        assert all(abs(c.args[0]) <= _SCROLL_STEP_MAX and abs(c.args[1]) <= _SCROLL_STEP_MAX
                   for c in mock_page.mouse.wheel.call_args_list)

    @pytest.mark.fast_sleeps
    async def test_stuck_element_fails_after_stagnant_steps(self: Self) -> None:
        """An element that never moves gives back false after a limited number of scrolls."""
        mgr, mock_page, tab = await _make_manager()
        mock_locator = mock_page.locator.return_value.first
        mock_locator.bounding_box = AsyncMock(return_value=self._bbox(100, _BELOW_FOLD_Y))

        result = await mgr.scroll_to_view(tab, "#el")

        assert result is False
        assert mock_page.mouse.wheel.call_count > 0
        assert mock_page.mouse.wheel.call_count <= _MAX_STUCK_WHEEL_CALLS

    async def test_visible_element_at_scroll_limit_succeeds(self: Self) -> None:
        """An element whose middle sits 5 pixels above the bottom edge counts as in view."""
        mgr, mock_page, tab = await _make_manager()
        mock_locator = mock_page.locator.return_value.first
        mock_locator.bounding_box = AsyncMock(return_value=self._bbox(100, 1060))

        result = await mgr.scroll_to_view(tab, "#el")

        assert result is True
        mock_page.mouse.wheel.assert_not_called()

    async def test_element_not_found(self: Self) -> None:
        """An element that is not there gives back false and nothing scrolls."""
        mgr, mock_page, tab = await _make_manager()
        mock_page.locator.return_value.first.count.return_value = 0

        result = await mgr.scroll_to_view(tab, "#missing")

        assert result is False
        mock_page.mouse.wheel.assert_not_called()


async def test_set_viewport_size() -> None:
    mgr, mock_page, tab = await _make_manager()
    assert await mgr.set_viewport_size(tab, _SET_VIEWPORT_WIDTH, _SET_VIEWPORT_HEIGHT) is True
    mock_page.set_viewport_size.assert_awaited_once_with({"width": _SET_VIEWPORT_WIDTH, "height": _SET_VIEWPORT_HEIGHT})


async def test_set_viewport_size_unknown_tab() -> None:
    mgr, _, _tab = await _make_manager()
    assert await mgr.set_viewport_size("nope", _SET_VIEWPORT_WIDTH, _SET_VIEWPORT_HEIGHT) is False


# ---------------------------------------------------------------------------
# Actions stop when the element cannot be reached
# ---------------------------------------------------------------------------


class TestElementActionFailFast:
    """Tests that nothing is done to an element that cannot be brought into view."""

    async def test_click_skipped_when_scroll_fails(self: Self) -> None:
        """A click is skipped when the element cannot be brought into view."""
        mgr, mock_page, tab = await _make_manager()
        mgr.scroll_to_view = AsyncMock(return_value=False)

        result = await mgr.click(tab, "#el")

        assert result is False
        mock_page.click.assert_not_called()

    async def test_type_text_skipped_when_scroll_fails(self: Self) -> None:
        """Typing is skipped when the element cannot be brought into view."""
        mgr, mock_page, tab = await _make_manager()
        mgr.scroll_to_view = AsyncMock(return_value=False)

        result = await mgr.type_text(tab, "#el", "text")

        assert result is False
        mock_page.fill.assert_not_called()
        mock_page.press.assert_not_called()

    async def test_select_option_skipped_when_scroll_fails(self: Self) -> None:
        """Picking an option is skipped when the element cannot be brought into view."""
        mgr, mock_page, tab = await _make_manager()
        mock_locator = _mock_locator(mock_page)
        mgr.scroll_to_view = AsyncMock(return_value=False)

        result = await mgr.select_option(tab, "select#el", value="x")

        assert result is False
        mock_locator.first.select_option.assert_not_called()


# ---------------------------------------------------------------------------
# Scrolling to a spot
# ---------------------------------------------------------------------------


class TestScrollToSettle:
    """Tests that scrolling waits for the page to stop moving before succeeding."""

    @pytest.mark.fast_sleeps
    async def test_returns_true_after_settled_position_read_twice(self: Self) -> None:
        """The spot is only reported done after it is read at the same value twice in a row."""
        mgr, _, positions, tab = await _make_settle_page(settles=True)

        result = await mgr.scroll_to(tab, 50, _BEYOND_VIEWPORT_Y)

        assert result is True
        assert positions[0] == _SETTLE_START_Y
        assert positions[-1] == _SETTLE_Y and positions[-2] == _SETTLE_Y

    @pytest.mark.fast_sleeps
    async def test_returns_false_when_scroll_never_settles(self: Self) -> None:
        """A spot that keeps moving gives back false once the settle wait runs out."""
        mgr, _, positions, tab = await _make_settle_page(settles=False)

        result = await mgr.scroll_to(tab, 50, _BEYOND_VIEWPORT_Y)

        assert result is False
        assert len(positions) > 20

    @pytest.mark.fast_sleeps
    async def test_returns_true_when_already_settled(self: Self) -> None:
        """A spot that is in view and already stopped counts as done right away."""
        mgr, _, tab = await _make_manager(session=_layout_session(page_y=_SETTLE_Y, content_height=_VIEWPORT_HEIGHT + _BEYOND_VIEWPORT_Y))

        result = await mgr.scroll_to(tab, 50, _BEYOND_VIEWPORT_Y)

        assert result is True


@pytest.mark.live
@skip_if_no_chrome
class TestScrollTo:
    """Live tests for scrolling to a spot on the page."""

    async def _setup_long_page(self, browser_manager: BrowserManager) -> None:
        """Open a new tab and give back its id."""
        tid = await browser_manager.new_page()
        return tid

    async def test_scroll_to_scroll_down(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """Scrolling to a spot below the window scrolls down."""
        tid = await self._setup_long_page(browser_manager)
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/long")
            time.sleep(_RENDER_DELAY)

            result = await browser_manager.scroll_to(tid, 50, _BEYOND_VIEWPORT_X)
            assert result is True

            scroll = await browser_manager.evaluate(tid, "() => window.scrollY")
            assert scroll > 0, f"Should have scrolled down, scrollY={scroll}"
        finally:
            await browser_manager.close_page(tid)

    async def test_scroll_to_already_in_viewport(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """Scrolling to a spot that is already in view works."""
        tid = await self._setup_long_page(browser_manager)
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/long")
            time.sleep(_RENDER_DELAY)

            # The spot is already in view.
            result = await browser_manager.scroll_to(tid, 50, 100)
            assert result is True
        finally:
            await browser_manager.close_page(tid)

    async def test_scroll_to_negative_coords(self, browser_manager: BrowserManager) -> None:
        """Scrolling to a spot with a negative number gives back false."""
        tid = await self._setup_long_page(browser_manager)
        try:
            result = await browser_manager.scroll_to(tid, -10, 100)
            assert result is False
        finally:
            await browser_manager.close_page(tid)

    async def test_scroll_to_unknown_target(self, browser_manager: BrowserManager) -> None:
        """Scrolling in a tab that is not there gives back false."""
        result = await browser_manager.scroll_to("nonexistent", 100, 100)
        assert result is False


# ---------------------------------------------------------------------------
# Scrolling to an element
# ---------------------------------------------------------------------------


@pytest.mark.live
@skip_if_no_chrome
class TestScroll:
    """Live tests for scrolling an element into view."""

    async def test_scroll_down_to_element(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """An element below the window can be scrolled to."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/long")
            time.sleep(_RENDER_DELAY)

            # The element sits far below the window.
            result = await browser_manager.scroll_to_view(tid, "#mid-button")
            assert result is True
        finally:
            await browser_manager.close_page(tid)

    async def test_scroll_up_to_element(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """An element above the window can be scrolled back to."""
        tid = await browser_manager.new_page()
        try:
            await _scroll_past_elements(browser_manager, tid, test_server_url)

            scroll_before = await browser_manager.evaluate(tid, "() => window.scrollY")
            assert scroll_before > 500, f"Not scrolled enough, scrollY={scroll_before}"

            # The top section is above the window now.
            result = await browser_manager.scroll_to_view(tid, "#top-section")
            assert result is True

            scroll_after = await browser_manager.evaluate(tid, "() => window.scrollY")
            assert scroll_after < scroll_before, f"Should have scrolled up: {scroll_before} -> {scroll_after}"
        finally:
            await browser_manager.close_page(tid)

    async def test_scroll_element_already_in_viewport(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """Scrolling to an element that is already in view works."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/long")
            time.sleep(_RENDER_DELAY)

            # The top section shows when the page opens.
            result = await browser_manager.scroll_to_view(tid, "#top-section")
            assert result is True
        finally:
            await browser_manager.close_page(tid)

    async def test_scroll_element_not_found(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """An element that is not on the page gives back false."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/long")
            time.sleep(_RENDER_DELAY)

            result = await browser_manager.scroll_to_view(tid, "#does-not-exist-xyz")
            assert result is False
        finally:
            await browser_manager.close_page(tid)

    async def test_scroll_unknown_target(self, browser_manager: BrowserManager) -> None:
        """Scrolling to an element in a tab that is not there gives back false."""
        result = await browser_manager.scroll_to_view("nonexistent", "#top-section")
        assert result is False


# ---------------------------------------------------------------------------
# Scrolling inside a panel that scrolls on its own
# ---------------------------------------------------------------------------


@pytest.mark.live
@skip_if_no_chrome
class TestNestedScrollLive:
    """Live tests for reaching things inside a panel that scrolls on its own.

    The page itself cannot scroll at all. Only the panel inside it moves.
    """

    async def test_scroll_to_nested_bottom(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """Scrolling reaches the bottom of the panel and leaves the page still."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/nested")
            time.sleep(_RENDER_DELAY)

            viewport = await _get_viewport(browser_manager, tid)
            before = await _element_bbox(browser_manager, tid, "#nested-bottom")
            assert before["y"] > viewport["h"], \
                f"#nested-bottom must start below the viewport, got y={before['y']}"

            result = await browser_manager.scroll_to_view(tid, "#nested-bottom")
            assert result is True

            bbox = await _element_bbox(browser_manager, tid, "#nested-bottom")
            cy = bbox["y"] + bbox["height"] / 2
            assert 0 <= cy <= viewport["h"], f"center {cy} not inside viewport {viewport['h']}"
            drawer_scroll = await browser_manager.evaluate(tid, "() => document.querySelector('#drawer').scrollTop")
            assert drawer_scroll > 0, f"drawer should have scrolled, scrollTop={drawer_scroll}"
            assert (await browser_manager.evaluate(tid, "() => window.scrollY")) == 0, \
                "document must not scroll; only the fixed drawer may scroll"
        finally:
            await browser_manager.close_page(tid)

    async def test_scroll_back_up_to_nested_top(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """Scrolling reaches the top of the panel again."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/nested")
            time.sleep(_RENDER_DELAY)

            assert (await browser_manager.scroll_to_view(tid, "#nested-bottom")) is True
            scroll_at_bottom = await browser_manager.evaluate(tid, "() => document.querySelector('#drawer').scrollTop")

            result = await browser_manager.scroll_to_view(tid, "#nested-top")
            assert result is True

            bbox = await _element_bbox(browser_manager, tid, "#nested-top")
            cy = bbox["y"] + bbox["height"] / 2
            viewport = await _get_viewport(browser_manager, tid)
            assert 0 <= cy <= viewport["h"], f"center {cy} not inside viewport {viewport['h']}"
            scroll_at_top = await browser_manager.evaluate(tid, "() => document.querySelector('#drawer').scrollTop")
            assert scroll_at_top < scroll_at_bottom, \
                f"drawer should have scrolled up: {scroll_at_bottom} -> {scroll_at_top}"
        finally:
            await browser_manager.close_page(tid)

    async def test_scroll_to_doc_target(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """A plain element on the page can still be reached."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/nested")
            time.sleep(_RENDER_DELAY)

            result = await browser_manager.scroll_to_view(tid, "#doc-target")
            assert result is True

            bbox = await _element_bbox(browser_manager, tid, "#doc-target")
            viewport = await _get_viewport(browser_manager, tid)
            cx = bbox["x"] + bbox["width"] / 2
            cy = bbox["y"] + bbox["height"] / 2
            assert 0 <= cx <= viewport["w"] and 0 <= cy <= viewport["h"], \
                f"center ({cx}, {cy}) not inside viewport ({viewport['w']}, {viewport['h']})"
        finally:
            await browser_manager.close_page(tid)

    async def test_select_option_in_nested_drawer(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """An option can be picked in a drop down inside the panel."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/nested")
            time.sleep(_RENDER_DELAY)

            result = await browser_manager.select_option(tid, "#nested-select", value="b")
            assert result is True

            val = await browser_manager.evaluate(tid, "() => document.querySelector('#nested-select').value")
            assert val == "b", f"Expected 'b', got '{val}'"
        finally:
            await browser_manager.close_page(tid)


# ---------------------------------------------------------------------------
# Clicking after scrolling
# ---------------------------------------------------------------------------


@pytest.mark.live
@skip_if_no_chrome
class TestClickWithScroll:
    """Live tests for clicking an element that is off screen."""

    async def test_click_element_below_viewport(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """An element below the window gets clicked."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/long")
            time.sleep(_RENDER_DELAY)

            # The button sits far below the window.
            result = await browser_manager.click(tid, "#mid-button")
            assert result is True

            log = await browser_manager.evaluate(tid, "() => window.__clickLog")
            assert "mid-button" in log, f"Expected 'mid-button' in click log, got {log}"
        finally:
            await browser_manager.close_page(tid)

    async def test_click_element_above_viewport(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """An element above the window gets clicked."""
        tid = await browser_manager.new_page()
        try:
            await _scroll_past_elements(browser_manager, tid, test_server_url)

            result = await browser_manager.click(tid, "#top-section")
            assert result is True

            log = await browser_manager.evaluate(tid, "() => window.__clickLog")
            assert "top-section" in log, f"Expected 'top-section' in click log, got {log}"
        finally:
            await browser_manager.close_page(tid)


# ---------------------------------------------------------------------------
# Typing after scrolling
# ---------------------------------------------------------------------------


@pytest.mark.live
@skip_if_no_chrome
class TestTypeTextWithScroll:
    """Live tests for typing into an element that is off screen."""

    async def test_type_text_element_below_viewport(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """Text goes into a box below the window."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/long")
            time.sleep(_RENDER_DELAY)

            # The box sits far below the window.
            result = await browser_manager.type_text(tid, "#mid-input", "hello scroll")
            assert result is True

            val = await browser_manager.evaluate(tid, '() => document.querySelector("#mid-input").value')
            assert val == "hello scroll", f"Expected 'hello scroll', got '{val}'"
        finally:
            await browser_manager.close_page(tid)

    async def test_type_text_element_above_viewport(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """Text goes into a box above the window."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/long")
            time.sleep(_RENDER_DELAY)

            # Scroll down past the box.
            await browser_manager.scroll_to(tid, 50, 8000)
            time.sleep(_SCROLL_DELAY)

            result = await browser_manager.type_text(tid, "#mid-input", "from above")
            assert result is True

            val = await browser_manager.evaluate(tid, '() => document.querySelector("#mid-input").value')
            assert val == "from above"
        finally:
            await browser_manager.close_page(tid)


# ---------------------------------------------------------------------------
# Picking options after scrolling
# ---------------------------------------------------------------------------


@pytest.mark.live
@skip_if_no_chrome
class TestSelectOptionWithScroll:
    """Live tests for picking an option in a drop down that is off screen."""

    async def test_select_option_below_viewport(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """An option is picked in a drop down below the window."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/long")
            time.sleep(_RENDER_DELAY)

            # The drop down sits far below the window.
            result = await browser_manager.select_option(tid, "#mid-select", value="b")
            assert result is True

            val = await browser_manager.evaluate(tid, '() => document.querySelector("#mid-select").value')
            assert val == "b", f"Expected 'b', got '{val}'"
        finally:
            await browser_manager.close_page(tid)

    async def test_select_option_above_viewport(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """An option is picked in a drop down above the window."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/long")
            time.sleep(_RENDER_DELAY)

            # Scroll past the drop down so it sits above the window.
            await browser_manager.scroll_to(tid, 50, 11000)
            time.sleep(_SCROLL_DELAY)

            result = await browser_manager.select_option(tid, "#mid-select", value="a")
            assert result is True

            val = await browser_manager.evaluate(tid, '() => document.querySelector("#mid-select").value')
            assert val == "a"
        finally:
            await browser_manager.close_page(tid)


# ---------------------------------------------------------------------------
# Hidden and showing parts in a snapshot
# ---------------------------------------------------------------------------


@pytest.mark.live
@skip_if_no_chrome
class TestSnapshotVisibilityNormalization:
    """Live tests that a snapshot keeps what shows and drops what does not."""

    async def test_css_hidden_element_excluded_from_tree(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """Parts hidden by style stay out of the tree."""
        tid = await browser_manager.new_page()
        try:
            tree_str = await _snapshot_visibility_tree(browser_manager, tid, test_server_url)
            assert "CSS hidden content should not appear" not in tree_str, \
                "display:none content should not appear in filtered_tree"
            assert "Visibility hidden content should not appear" not in tree_str, \
                "visibility:hidden content should not appear in filtered_tree"
        finally:
            await browser_manager.close_page(tid)

    async def test_aria_hidden_visible_element_included_in_tree(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """Parts marked hidden for screen readers but still showing stay in the tree."""
        tid = await browser_manager.new_page()
        try:
            tree_str = await _snapshot_visibility_tree(browser_manager, tid, test_server_url)
            assert "Aria hidden but visible content" in tree_str, \
                "aria-hidden but visually visible content should appear in filtered_tree"
        finally:
            await browser_manager.close_page(tid)


# ---------------------------------------------------------------------------
# Scrolling that gets stuck
# ---------------------------------------------------------------------------


@pytest.mark.live
@skip_if_no_chrome
class TestScrollToStagnantViewport:
    """Live tests for scrolling that cannot move the page.

    Scrolling gives up after 5 steps that change nothing.
    """

    async def test_scroll_to_fails_on_stagnant_viewport(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """Scrolling a page that cannot move gives back false."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/nested-scroll")
            time.sleep(_RENDER_DELAY)

            # The page itself cannot scroll, only the panel inside it can.
            result = await browser_manager.scroll_to(tid, 150, _BEYOND_VIEWPORT_X)
            assert result is False, \
                "scroll_to should fail when viewport is stagnant after 5 scroll steps"
        finally:
            await browser_manager.close_page(tid)


# ---------------------------------------------------------------------------
# Moving the mouse
# ---------------------------------------------------------------------------


class TestBrowserManagerMoveMouseTo:
    """Tests for moving the mouse to a spot in the window.

    A spot outside the window is pulled back to the edge.
    """

    async def test_move_mouse_to_basic(self: Self) -> None:
        """The mouse moves to the given spot."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.move_mouse_to(tab, 100, 200)
        assert result is True
        mock_page.mouse.move.assert_called_once_with(100, 200)

    async def test_move_mouse_to_caps_negative(self: Self) -> None:
        """A negative spot becomes the top left corner."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.move_mouse_to(tab, -50, -30)
        assert result is True
        mock_page.mouse.move.assert_called_once_with(0, 0)

    async def test_move_mouse_to_caps_beyond_viewport(self: Self) -> None:
        """A spot past the window edge becomes the edge."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.move_mouse_to(tab, _BEYOND_VIEWPORT_X, _BEYOND_VIEWPORT_Y)
        assert result is True
        mock_page.mouse.move.assert_called_once_with(_VIEWPORT_WIDTH, _VIEWPORT_HEIGHT)

    async def test_move_mouse_to_caps_one_axis(self: Self) -> None:
        """Only the number that is too big gets pulled back."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.move_mouse_to(tab, _BEYOND_VIEWPORT_X, 500)
        assert result is True
        mock_page.mouse.move.assert_called_once_with(_VIEWPORT_WIDTH, 500)

    async def test_move_mouse_to_unknown_target(self: Self) -> None:
        """Moving the mouse in a tab that is not there gives back false."""
        mgr = await _empty_manager()

        result = await mgr.move_mouse_to("nonexistent", 100, 200)
        assert result is False

    async def test_move_mouse_to_viewport_read_failure(self: Self) -> None:
        """The move gives back false when the window cannot be read."""
        session = _layout_session()
        session.send.side_effect = Exception("CDP down")
        mgr, _, tab = await _make_manager(session=session)

        result = await mgr.move_mouse_to(tab, 100, 200)
        assert result is False

    async def test_move_mouse_to_at_edge(self: Self) -> None:
        """A spot right on the edge is used as it is."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.move_mouse_to(tab, _VIEWPORT_WIDTH, _VIEWPORT_HEIGHT)
        assert result is True
        mock_page.mouse.move.assert_called_once_with(_VIEWPORT_WIDTH, _VIEWPORT_HEIGHT)

    async def test_move_mouse_to_zero_zero(self: Self) -> None:
        """A spot of zero and zero is the top left corner."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.move_mouse_to(tab, 0, 0)
        assert result is True
        mock_page.mouse.move.assert_called_once_with(0, 0)

    @pytest.mark.parametrize("x,y,expected_x,expected_y", [
        (-100, -100, 0, 0),
        (0, 0, 0, 0),
        (_HALF_VIEWPORT_W, _HALF_VIEWPORT_H, _HALF_VIEWPORT_W, _HALF_VIEWPORT_H),
        (1921, _VIEWPORT_HEIGHT, _VIEWPORT_WIDTH, _VIEWPORT_HEIGHT),
        (_HALF_VIEWPORT_W, 2000, _HALF_VIEWPORT_W, _VIEWPORT_HEIGHT),
        (_BEYOND_VIEWPORT_X, _BEYOND_VIEWPORT_Y, _VIEWPORT_WIDTH, _VIEWPORT_HEIGHT),
    ])
    async def test_move_mouse_to_capping_various(self, x: int, y: int, expected_x: int, expected_y: int) -> None:
        """Spots inside the window are kept and spots outside are pulled to the edge."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.move_mouse_to(tab, x, y)
        assert result is True
        mock_page.mouse.move.assert_called_once_with(expected_x, expected_y)


# ---------------------------------------------------------------------------
# Moving the mouse by fraction
# ---------------------------------------------------------------------------


class TestBrowserManagerMoveMouseToRelative:
    """Tests for moving the mouse by fraction of the window.

    The fraction runs from 0 to 1.
    """

    async def test_move_mouse_to_relative_center(self: Self) -> None:
        """A fraction of one half lands in the middle of the window."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.move_mouse_to_relative(tab, _CENTER_RATIO, _CENTER_RATIO)
        assert result is True
        mock_page.mouse.move.assert_called_once_with(_HALF_VIEWPORT_WIDTH, _HALF_VIEWPORT_HEIGHT)

    async def test_move_mouse_to_relative_top_left(self: Self) -> None:
        """A fraction of zero lands in the top left corner."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.move_mouse_to_relative(tab, 0, 0)
        assert result is True
        mock_page.mouse.move.assert_called_once_with(0.0, 0.0)

    async def test_move_mouse_to_relative_bottom_right(self: Self) -> None:
        """A fraction of one lands in the bottom right corner."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.move_mouse_to_relative(tab, 1, 1)
        assert result is True
        mock_page.mouse.move.assert_called_once_with(float(_VIEWPORT_WIDTH), float(_VIEWPORT_HEIGHT))

    async def test_move_mouse_to_relative_uses_the_window_width(self: Self) -> None:
        """The ratio runs over the window width, not the client width."""
        mgr, mock_page, tab = await _make_manager()
        mock_page.evaluate.return_value = [_WINDOW_WIDER_THAN_CLIENT, _VIEWPORT_HEIGHT]

        result = await mgr.move_mouse_to_relative(tab, 1, 1)
        assert result is True
        mock_page.mouse.move.assert_called_once_with(float(_WINDOW_WIDER_THAN_CLIENT), float(_VIEWPORT_HEIGHT))

    async def test_move_mouse_to_relative_invalid_ratio_negative(self: Self) -> None:
        """A fraction below zero gives back false."""
        mgr, _, tab = await _make_manager()

        result = await mgr.move_mouse_to_relative(tab, -0.1, _CENTER_RATIO)
        assert result is False

    async def test_move_mouse_to_relative_invalid_ratio_over_one(self: Self) -> None:
        """A fraction above one gives back false."""
        mgr, _, tab = await _make_manager()

        result = await mgr.move_mouse_to_relative(tab, _CENTER_RATIO, 1.1)
        assert result is False

    async def test_move_mouse_to_relative_unknown_target(self: Self) -> None:
        """Moving by fraction in a tab that is not there gives back false."""
        mgr = await _empty_manager()

        result = await mgr.move_mouse_to_relative("nonexistent", _CENTER_RATIO, _CENTER_RATIO)
        assert result is False

    async def test_move_mouse_to_relative_delegates_to_move_mouse_to(self: Self) -> None:
        """A fraction turns into the matching spot in the window."""
        mgr, mock_page, tab = await _make_manager()

        await mgr.move_mouse_to_relative(tab, 0.25, _THREE_QUARTER_RATIO)

        mock_page.mouse.move.assert_called_once_with(480.0, 810.0)

    async def test_move_mouse_to_relative_viewport_read_failure(self: Self) -> None:
        """Moving by fraction gives back false when the window cannot be read."""
        session = _layout_session()
        session.send.side_effect = Exception("CDP down")
        mgr, _, tab = await _make_manager(session=session)

        result = await mgr.move_mouse_to_relative(tab, _CENTER_RATIO, _CENTER_RATIO)
        assert result is False


# ---------------------------------------------------------------------------
# Acting at a share of the window
# ---------------------------------------------------------------------------


def _layout_session(page_x: float = 0.0, page_y: float = 0.0, content_width: float = _VIEWPORT_WIDTH, content_height: float = _VIEWPORT_HEIGHT) -> MagicMock:
    """Make a fake CDP session that answers the layout read with the test window."""
    session = MagicMock()

    async def _send(method: str, _params: Optional[dict] = None) -> dict:
        if method == "Page.getLayoutMetrics":
            return {
                "cssVisualViewport": {
                    "clientWidth": _VIEWPORT_WIDTH,
                    "clientHeight": _VIEWPORT_HEIGHT,
                    "pageX": page_x,
                    "pageY": page_y,
                },
                "cssContentSize": {"width": content_width, "height": content_height},
            }
        return {}

    session.send = AsyncMock(side_effect=_send)
    session.detach = AsyncMock()
    return session


def _drifting_layout_session(next_y: Callable[[], float]) -> MagicMock:
    """Make a session whose layout read gives back the next drifting scroll spot."""
    session = MagicMock()

    async def _send(method: str, _params: Optional[dict] = None) -> dict:
        if method == "Page.getLayoutMetrics":
            return {
                "cssVisualViewport": {
                    "clientWidth": _VIEWPORT_WIDTH,
                    "clientHeight": _VIEWPORT_HEIGHT,
                    "pageX": 0.0,
                    "pageY": next_y(),
                },
                "cssContentSize": {"width": _VIEWPORT_WIDTH, "height": _VIEWPORT_HEIGHT + _BEYOND_VIEWPORT_Y},
            }
        return {}

    session.send = AsyncMock(side_effect=_send)
    session.detach = AsyncMock()
    return session


def _scrolled_viewport() -> Viewport:
    """Make the test window's geometry for a page scrolled down and sideways."""
    return Viewport(
        client_width=_VIEWPORT_WIDTH,
        client_height=_VIEWPORT_HEIGHT,
        page_x=_SCROLLED_VIEWPORT_PAGE_X,
        page_y=_SCROLLED_VIEWPORT_PAGE_Y,
        content_width=_VIEWPORT_WIDTH,
        content_height=_VIEWPORT_HEIGHT,
        window_width=_VIEWPORT_WIDTH,
        window_height=_VIEWPORT_HEIGHT,
    )


def _unscrolled_viewport() -> Viewport:
    """Make the test window's geometry for a page that is not scrolled."""
    return Viewport(
        client_width=_VIEWPORT_WIDTH,
        client_height=_VIEWPORT_HEIGHT,
        page_x=0.0,
        page_y=0.0,
        content_width=_VIEWPORT_WIDTH,
        content_height=_VIEWPORT_HEIGHT,
        window_width=_VIEWPORT_WIDTH,
        window_height=_VIEWPORT_HEIGHT,
    )


def _wide_window_viewport() -> Viewport:
    """Make the test window's geometry for a scrolled page with a scrollbar."""
    return Viewport(
        client_width=_VIEWPORT_WIDTH,
        client_height=_VIEWPORT_HEIGHT,
        page_x=_SCROLLED_VIEWPORT_PAGE_X,
        page_y=_SCROLLED_VIEWPORT_PAGE_Y,
        content_width=_VIEWPORT_WIDTH,
        content_height=_VIEWPORT_HEIGHT,
        window_width=_WINDOW_WIDER_THAN_CLIENT,
        window_height=_WINDOW_TALLER_THAN_CLIENT,
    )


class TestBrowserManagerClickAtRelative:
    """Tests for clicking a spot given as a share of the window."""

    async def test_click_at_relative_scrolled_viewport(self: Self) -> None:
        """A share on a scrolled window clicks the window center."""
        mgr, mock_page, tab = await _make_manager(session=_layout_session(page_x=_SCROLLED_VIEWPORT_PAGE_X, page_y=_SCROLLED_VIEWPORT_PAGE_Y))

        result = await mgr.click_at_relative(tab, _CENTER_RATIO, _CENTER_RATIO, viewport=_scrolled_viewport())

        assert result is True
        mock_page.mouse.click.assert_called_once_with(_HALF_VIEWPORT_WIDTH, _HALF_VIEWPORT_HEIGHT)

    async def test_click_at_relative_unscrolled_viewport(self: Self) -> None:
        """A share on an unscrolled window clicks the ratio times the window size."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.click_at_relative(tab, _CENTER_RATIO, _CENTER_RATIO, viewport=_unscrolled_viewport())

        assert result is True
        mock_page.mouse.click.assert_called_once_with(_HALF_VIEWPORT_WIDTH, _HALF_VIEWPORT_HEIGHT)

    async def test_click_at_relative_reads_viewport_when_none(self: Self) -> None:
        """With no captured window, the tracked session read still lands the click at the window center."""
        session = _layout_session(page_x=_SCROLLED_VIEWPORT_PAGE_X, page_y=_SCROLLED_VIEWPORT_PAGE_Y)
        mgr, mock_page, tab = await _make_manager(session=session)

        result = await mgr.click_at_relative(tab, _CENTER_RATIO, _CENTER_RATIO)

        assert result is True
        session.send.assert_any_call("Page.getLayoutMetrics")
        mock_page.mouse.click.assert_called_once_with(_HALF_VIEWPORT_WIDTH, _HALF_VIEWPORT_HEIGHT)

    async def test_click_at_relative_cdp_read_failure(self: Self) -> None:
        """A share click gives back false and clicks nothing when the window cannot be read."""
        mgr, mock_page, tab = await _make_manager(attach_session=False)

        result = await mgr.click_at_relative(tab, _CENTER_RATIO, _CENTER_RATIO)

        assert result is False
        mock_page.mouse.click.assert_not_called()

    async def test_click_at_relative_uses_the_window_size(self: Self) -> None:
        """A share runs over the window size, the scrollbar included, past the page offset."""
        mgr, mock_page, tab = await _make_manager(session=_layout_session(page_x=_SCROLLED_VIEWPORT_PAGE_X, page_y=_SCROLLED_VIEWPORT_PAGE_Y))

        result = await mgr.click_at_relative(tab, 1, 1, viewport=_wide_window_viewport())

        assert result is True
        mock_page.mouse.click.assert_called_once_with(float(_WINDOW_WIDER_THAN_CLIENT), float(_WINDOW_TALLER_THAN_CLIENT))

    async def test_click_at_relative_unknown_target(self: Self) -> None:
        """Clicking a share in a tab that is not there gives back false."""
        mgr = await _empty_manager()

        result = await mgr.click_at_relative("nonexistent", _CENTER_RATIO, _CENTER_RATIO)

        assert result is False


class TestBrowserManagerTypeTextAt:
    """Tests for typing text at a spot on the page."""

    async def test_type_text_at_clicks_then_types(self: Self) -> None:
        """The spot is clicked first, then the text is typed there."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.type_text_at(tab, 100, 200, _TYPE_TEXT)

        assert result is True
        mock_page.mouse.click.assert_called_once_with(100, 200)
        mock_page.keyboard.type.assert_called_once_with(_TYPE_TEXT)

    async def test_type_text_at_types_after_the_click(self: Self) -> None:
        """The click happens before the typing."""
        mgr, mock_page, tab = await _make_manager()

        await mgr.type_text_at(tab, 100, 200, _TYPE_TEXT)

        calls = mock_page.mock_calls
        click_index = calls.index(call.mouse.click(100, 200))
        type_index = calls.index(call.keyboard.type(_TYPE_TEXT))
        assert click_index < type_index

    async def test_type_text_at_with_submit(self: Self) -> None:
        """Typing with submit turned on presses Enter after the text."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.type_text_at(tab, 100, 200, _TYPE_TEXT, submit=True)

        assert result is True
        mock_page.keyboard.type.assert_called_once_with(_TYPE_TEXT)
        mock_page.keyboard.press.assert_called_once_with("Enter")
        calls = mock_page.keyboard.mock_calls
        type_index = calls.index(call.type(_TYPE_TEXT))
        press_index = calls.index(call.press("Enter"))
        assert type_index < press_index

    async def test_type_text_at_without_submit_no_enter(self: Self) -> None:
        """Typing without submit does not press Enter."""
        mgr, mock_page, tab = await _make_manager()

        await mgr.type_text_at(tab, 100, 200, _TYPE_TEXT, submit=False)

        mock_page.keyboard.press.assert_not_called()

    async def test_type_text_at_click_failure(self: Self) -> None:
        """Typing gives back false and types nothing when the click fails."""
        session = _layout_session()
        session.send.side_effect = Exception("CDP down")
        mgr, mock_page, tab = await _make_manager(session=session)

        result = await mgr.type_text_at(tab, 100, 200, _TYPE_TEXT)

        assert result is False
        mock_page.keyboard.type.assert_not_called()

    async def test_type_text_at_keyboard_failure(self: Self) -> None:
        """Typing gives back false when the keyboard blows up."""
        mgr, mock_page, tab = await _make_manager()
        mock_page.keyboard.type.side_effect = Exception("Keyboard not available")

        result = await mgr.type_text_at(tab, 100, 200, _TYPE_TEXT)

        assert result is False

    async def test_type_text_at_unknown_target(self: Self) -> None:
        """Typing in a tab that is not there gives back false."""
        mgr = await _empty_manager()

        result = await mgr.type_text_at("nonexistent", 100, 200, _TYPE_TEXT)

        assert result is False


class TestBrowserManagerTypeTextAtRelative:
    """Tests for typing text at a spot given as a share of the window."""

    async def test_type_text_at_relative_scrolled_viewport(self: Self) -> None:
        """A share on a scrolled window types at the window center."""
        mgr, mock_page, tab = await _make_manager(session=_layout_session(page_x=_SCROLLED_VIEWPORT_PAGE_X, page_y=_SCROLLED_VIEWPORT_PAGE_Y))

        result = await mgr.type_text_at_relative(tab, _CENTER_RATIO, _CENTER_RATIO, _TYPE_TEXT, viewport=_scrolled_viewport())

        assert result is True
        mock_page.mouse.click.assert_called_once_with(_HALF_VIEWPORT_WIDTH, _HALF_VIEWPORT_HEIGHT)
        mock_page.keyboard.type.assert_called_once_with(_TYPE_TEXT)

    async def test_type_text_at_relative_unscrolled_viewport(self: Self) -> None:
        """A share on an unscrolled window types at the ratio times the window size."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.type_text_at_relative(tab, _CENTER_RATIO, _CENTER_RATIO, _TYPE_TEXT, viewport=_unscrolled_viewport())

        assert result is True
        mock_page.mouse.click.assert_called_once_with(_HALF_VIEWPORT_WIDTH, _HALF_VIEWPORT_HEIGHT)
        mock_page.keyboard.type.assert_called_once_with(_TYPE_TEXT)

    async def test_type_text_at_relative_uses_the_window_size(self: Self) -> None:
        """A share runs over the window size, the scrollbar included, past the page offset."""
        mgr, mock_page, tab = await _make_manager(session=_layout_session(page_x=_SCROLLED_VIEWPORT_PAGE_X, page_y=_SCROLLED_VIEWPORT_PAGE_Y))

        result = await mgr.type_text_at_relative(tab, 1, 1, _TYPE_TEXT, viewport=_wide_window_viewport())

        assert result is True
        mock_page.mouse.click.assert_called_once_with(float(_WINDOW_WIDER_THAN_CLIENT), float(_WINDOW_TALLER_THAN_CLIENT))
        mock_page.keyboard.type.assert_called_once_with(_TYPE_TEXT)

    async def test_type_text_at_relative_cdp_read_failure(self: Self) -> None:
        """A share typing gives back false and types nothing when the window cannot be read."""
        mgr, mock_page, tab = await _make_manager(attach_session=False)

        result = await mgr.type_text_at_relative(tab, _CENTER_RATIO, _CENTER_RATIO, _TYPE_TEXT)

        assert result is False
        mock_page.keyboard.type.assert_not_called()


class TestBrowserManagerScrollBy:
    """Tests for scrolling a set number of pixels, on both axes."""

    async def test_scroll_by_distance_only_is_vertical(self: Self) -> None:
        """A distance with no sideways number scrolls down only."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.scroll_by(tab, _SCROLL_DISTANCE)

        assert result is True
        wheel_calls = mock_page.mouse.wheel.call_args_list
        assert all(wheel_call.args[0] == 0 for wheel_call in wheel_calls)
        assert sum(abs(wheel_call.args[1]) for wheel_call in wheel_calls) == _SCROLL_DISTANCE

    async def test_scroll_by_with_dx_moves_both_axes(self: Self) -> None:
        """A sideways number scrolls both directions."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.scroll_by(tab, _SCROLL_DISTANCE, dx=_SCROLL_SIDEWAYS)

        assert result is True
        wheel_calls = mock_page.mouse.wheel.call_args_list
        assert sum(abs(wheel_call.args[0]) for wheel_call in wheel_calls) == _SCROLL_SIDEWAYS
        assert sum(abs(wheel_call.args[1]) for wheel_call in wheel_calls) == _SCROLL_DISTANCE

    async def test_scroll_by_unknown_target(self: Self) -> None:
        """Scrolling a tab that is not there gives back false."""
        mgr = await _empty_manager()

        result = await mgr.scroll_by("nonexistent", _SCROLL_DISTANCE)

        assert result is False


class TestBrowserManagerScrollByRelative:
    """Tests for scrolling a share of the window in each direction."""

    async def test_scroll_by_relative_delegates_scaled_distance(self: Self) -> None:
        """The shares turn into pixel distances, with no page position."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.scroll_by_relative(tab, _REL_SCROLL_DX, _REL_SCROLL_DY, viewport=_scrolled_viewport())

        assert result is True
        wheel_calls = mock_page.mouse.wheel.call_args_list
        assert sum(abs(wheel_call.args[0]) for wheel_call in wheel_calls) == _REL_SCROLL_WIDTH
        assert sum(abs(wheel_call.args[1]) for wheel_call in wheel_calls) == _REL_SCROLL_HEIGHT

    async def test_scroll_by_relative_uses_the_window_size(self: Self) -> None:
        """The shares scale by the window size, the scrollbar included."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.scroll_by_relative(tab, _REL_SCROLL_DX, _REL_SCROLL_DY, viewport=_wide_window_viewport())

        assert result is True
        wheel_calls = mock_page.mouse.wheel.call_args_list
        assert sum(abs(wheel_call.args[0]) for wheel_call in wheel_calls) == _REL_SCROLL_WIDTH_WIDE
        assert sum(abs(wheel_call.args[1]) for wheel_call in wheel_calls) == _REL_SCROLL_HEIGHT_TALL

    async def test_scroll_by_relative_reads_viewport_when_none(self: Self) -> None:
        """With no captured window, the size comes from the tracked session read."""
        session = _layout_session()
        mgr, mock_page, tab = await _make_manager(session=session)

        result = await mgr.scroll_by_relative(tab, _REL_SCROLL_DX, _REL_SCROLL_DY)

        assert result is True
        session.send.assert_any_call("Page.getLayoutMetrics")
        wheel_calls = mock_page.mouse.wheel.call_args_list
        assert sum(abs(wheel_call.args[1]) for wheel_call in wheel_calls) == _REL_SCROLL_HEIGHT

    async def test_scroll_by_relative_cdp_read_failure(self: Self) -> None:
        """A share scroll gives back false and scrolls nothing when the window cannot be read."""
        mgr, mock_page, tab = await _make_manager(attach_session=False)

        result = await mgr.scroll_by_relative(tab, _REL_SCROLL_DX, _REL_SCROLL_DY)

        assert result is False
        mock_page.mouse.wheel.assert_not_called()

    async def test_scroll_by_relative_unknown_target(self: Self) -> None:
        """Scrolling a share in a tab that is not there gives back false."""
        mgr = await _empty_manager()

        result = await mgr.scroll_by_relative("nonexistent", _REL_SCROLL_DX, _REL_SCROLL_DY)

        assert result is False


class TestBrowserManagerGoForward:
    """Tests for going forward in the page history."""

    async def test_go_forward_waits_for_load(self: Self) -> None:
        """Going forward waits for the page to load."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.go_forward(tab)

        assert result is True
        call = mock_page.go_forward.call_args
        assert call.kwargs["wait_until"] == "load"
        assert call.kwargs["timeout"] == _NETWORK_IDLE_TIMEOUT_MS

    async def test_go_forward_failure(self: Self) -> None:
        """Going forward gives back false when the page load blows up."""
        mgr, mock_page, tab = await _make_manager()
        mock_page.go_forward.side_effect = Exception("Navigation failed")

        result = await mgr.go_forward(tab)

        assert result is False

    async def test_go_forward_unknown_target(self: Self) -> None:
        """Going forward in a tab that is not there gives back false."""
        mgr = await _empty_manager()

        result = await mgr.go_forward("nonexistent")

        assert result is False


class TestBrowserManagerSnapshotViewport:
    """Tests that a snapshot carries the window's geometry."""

    async def test_snapshot_carries_viewport_from_layout(self: Self) -> None:
        """A snapshot reads the window's size and place from the page layout."""
        session = _layout_session(page_x=_SCROLLED_VIEWPORT_PAGE_X, page_y=_SCROLLED_VIEWPORT_PAGE_Y)
        mgr, _, tab = await _make_manager(session=session)

        result = await mgr.snapshot(tab)

        assert isinstance(result, PageSnapshot)
        assert result.viewport == _scrolled_viewport()
        session.send.assert_any_call("Page.getLayoutMetrics")

    async def test_snapshot_layout_read_failure_yields_no_viewport(self: Self) -> None:
        """A snapshot still comes back when the layout read fails."""
        session = _layout_session()

        async def _send(method: str, _params: Optional[dict] = None) -> dict:
            if method == "Page.getLayoutMetrics":
                raise Exception("Page.getLayoutMetrics failed")
            return {}

        session.send = AsyncMock(side_effect=_send)
        mgr, _, tab = await _make_manager(session=session)

        result = await mgr.snapshot(tab)

        assert isinstance(result, PageSnapshot)
        assert result.viewport is None
        assert result.html == "<html><body>test</body></html>"

    async def test_snapshot_missing_session_yields_no_viewport(self: Self) -> None:
        """A snapshot still comes back when the tab has no session."""
        mgr, _, tab = await _make_manager(attach_session=False)

        result = await mgr.snapshot(tab)

        assert isinstance(result, PageSnapshot)
        assert result.viewport is None


class TestBrowserManagerMoveMouseToRelativeViewport:
    """Tests for moving the mouse by fraction with a captured window."""

    async def test_move_mouse_to_relative_scrolled_viewport_uses_viewport_coordinates(self: Self) -> None:
        """A share on a scrolled window moves the mouse in the window, not on the page."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.move_mouse_to_relative(tab, _CENTER_RATIO, _CENTER_RATIO, viewport=_scrolled_viewport())

        assert result is True
        mock_page.mouse.move.assert_called_once_with(_HALF_VIEWPORT_WIDTH, _HALF_VIEWPORT_HEIGHT)

    async def test_move_mouse_to_relative_without_viewport_uses_tracked_read(self: Self) -> None:
        """With no captured window, the size still comes from the tracked session read."""
        mgr, mock_page, tab = await _make_manager()

        result = await mgr.move_mouse_to_relative(tab, 0.25, _THREE_QUARTER_RATIO)

        assert result is True
        mock_page.evaluate.assert_any_call("() => [window.innerWidth, window.innerHeight]")
        mock_page.mouse.move.assert_called_once_with(480.0, 810.0)


# ---------------------------------------------------------------------------
# Reading the window geometry
# ---------------------------------------------------------------------------


class TestBrowserManagerReadViewport:
    """Tests for reading a tab's window geometry from its tracked session."""

    async def test_tracked_session_is_reused_without_a_new_attach(self, tmp_path: Path) -> None:
        """A tab that already has a session is read through that session."""
        mgr, _, session, context = await _tab_session_manager(tmp_path)
        target_id = await mgr.new_page()

        assert await mgr.click_at_relative(target_id, _CENTER_RATIO, _CENTER_RATIO) is True

        assert context.new_cdp_session.await_count == 1
        session.send.assert_any_call("Page.getLayoutMetrics")

    async def test_missing_session_gives_no_viewport(self, tmp_path: Path) -> None:
        """A tab without a session reads no window, and its clicks give back false."""
        mock_page = _fake_page()
        context = _test_context(mock_page, _layout_session(), attach_session=False)
        mgr = await _created_manager(context, str(tmp_path))

        try:
            tab = await mgr.new_page()
            assert await mgr.click_at_relative(tab, _CENTER_RATIO, _CENTER_RATIO) is False
            context.new_cdp_session.assert_awaited_once_with(mock_page)
            mock_page.mouse.click.assert_not_called()
        finally:
            await mgr.cleanup()

    @pytest.mark.parametrize("content_width,content_height,max_scroll_x,max_scroll_y", [
        (_VIEWPORT_WIDTH + _SCROLL_EXTRA_WIDTH, _VIEWPORT_HEIGHT + _SCROLL_EXTRA_HEIGHT_SHORT, _SCROLL_EXTRA_WIDTH, _SCROLL_EXTRA_HEIGHT_SHORT),
        (500.0, 100.0, 0.0, 0.0),
    ])
    async def test_viewport_carries_the_content_size_and_max_scroll(
        self, content_width: float, content_height: float, max_scroll_x: float, max_scroll_y: float
    ) -> None:
        """The content size sets how far the page can scroll, floored at zero."""
        mgr, _, tab = await _make_manager(session=_layout_session(content_width=content_width, content_height=content_height))

        result = await mgr.snapshot(tab)

        assert result.viewport.content_width == content_width
        assert result.viewport.content_height == content_height
        assert result.viewport.max_scroll_x == max_scroll_x
        assert result.viewport.max_scroll_y == max_scroll_y

    async def test_window_size_read_failure_leaves_the_cdp_fields_intact(self: Self) -> None:
        """A failed window size read zeros only the window fields."""
        mgr, mock_page, tab = await _make_manager()
        mock_page.evaluate.side_effect = Exception("Page crashed")

        result = await mgr.snapshot(tab)

        assert result.viewport.window_width == 0.0
        assert result.viewport.window_height == 0.0
        assert result.viewport.client_width == _VIEWPORT_WIDTH
        assert result.viewport.client_height == _VIEWPORT_HEIGHT
        assert result.viewport.page_x == 0.0
        assert result.viewport.page_y == 0.0


# ---------------------------------------------------------------------------
# Pulling mouse events back inside the window
# ---------------------------------------------------------------------------


_CLIENT_WIDTH_WITHOUT_SCROLLBAR = 1905


async def _clamped_manager(tmp_path: Path, page: AsyncMock, session: MagicMock) -> BrowserManager:
    """Make a connected manager that hands out one fake tab and one live CDP session."""
    return await _created_manager(_test_context(page, session), str(tmp_path))


def _fake_channel(page: AsyncMock) -> SimpleNamespace:
    """Make the channel the installed clamp hook reaches through to find its page."""
    # lint-skip: coding.access-private
    impl_page = SimpleNamespace(_guid=page._impl_obj._guid)
    return SimpleNamespace(_object=impl_page, _connection=SimpleNamespace(wrap_api_call=AsyncMock(return_value={})))


def _layout_read_count(session: MagicMock) -> int:
    """Count the layout reads a session answered."""
    return len([sent for sent in session.send.await_args_list if sent.args[0] == "Page.getLayoutMetrics"])


async def _send_mouse_event(page: AsyncMock, method: str) -> dict:
    """Send one mouse event aimed past the window and give back its params."""
    params = {"x": _BEYOND_VIEWPORT_X, "y": _BEYOND_VIEWPORT_Y}
    await bm_module.Channel.send(_fake_channel(page), method, None, params)
    return params


class TestMouseClamp:
    """Tests for pulling a mouse event back inside the window it travels to."""

    async def test_event_is_pulled_back_to_the_window_edge(self, tmp_path: Path) -> None:
        """An event past the window edge is pulled back to the window edge."""
        page = AsyncMock()
        page.evaluate.return_value = [_VIEWPORT_WIDTH, _VIEWPORT_HEIGHT]
        mgr = await _clamped_manager(tmp_path, page, _layout_session())
        try:
            await mgr.new_page()
            params = await _send_mouse_event(page, "mouseMove")

            assert params["x"] == _VIEWPORT_WIDTH - 1
            assert params["y"] == _VIEWPORT_HEIGHT - 1
        finally:
            await mgr.cleanup()

    async def test_window_is_read_once_and_cached(self, tmp_path: Path) -> None:
        """The first mouse event reads the window, and later events reuse that size."""
        page = AsyncMock()
        page.evaluate.return_value = [_VIEWPORT_WIDTH, _VIEWPORT_HEIGHT]
        session = _layout_session()
        mgr = await _clamped_manager(tmp_path, page, session)
        try:
            await mgr.new_page()
            params = await _send_mouse_event(page, "mouseMove")
            assert params["x"] == _VIEWPORT_WIDTH - 1
            assert params["y"] == _VIEWPORT_HEIGHT - 1
            again = await _send_mouse_event(page, "mouseMove")
            assert again["x"] == _VIEWPORT_WIDTH - 1
            assert again["y"] == _VIEWPORT_HEIGHT - 1

            assert _layout_read_count(session) == 1
        finally:
            await mgr.cleanup()

    async def test_click_uses_the_cached_window_size(self, tmp_path: Path) -> None:
        """A click after a move clamps against the size the move already read."""
        page = AsyncMock()
        page.evaluate.return_value = [_VIEWPORT_WIDTH, _VIEWPORT_HEIGHT]
        session = _layout_session()
        mgr = await _clamped_manager(tmp_path, page, session)
        try:
            await mgr.new_page()
            await _send_mouse_event(page, "mouseMove")
            params = await _send_mouse_event(page, "mouseClick")

            assert params["x"] == _VIEWPORT_WIDTH - 1
            assert params["y"] == _VIEWPORT_HEIGHT - 1
            assert _layout_read_count(session) == 1
        finally:
            await mgr.cleanup()

    async def test_failed_read_is_not_cached(self, tmp_path: Path) -> None:
        """A failed read leaves nothing to reuse, and the next event reads again."""
        page = AsyncMock()
        page.evaluate.return_value = [_VIEWPORT_WIDTH, _VIEWPORT_HEIGHT]
        session = _layout_session()
        layout_reads = 0
        first_send = session.send

        async def _flaky_send(method: str, params: Optional[dict] = None) -> dict:
            nonlocal layout_reads
            if method == "Page.getLayoutMetrics":
                layout_reads += 1
                if layout_reads == 1:
                    raise Exception("CDP session detached")
            return await first_send(method, params)

        session.send = AsyncMock(side_effect=_flaky_send)
        mgr = await _clamped_manager(tmp_path, page, session)
        try:
            await mgr.new_page()
            params = await _send_mouse_event(page, "mouseMove")

            assert params["x"] == _BEYOND_VIEWPORT_X
            assert params["y"] == _BEYOND_VIEWPORT_Y

            again = await _send_mouse_event(page, "mouseMove")

            assert again["x"] == _VIEWPORT_WIDTH - 1
            assert again["y"] == _VIEWPORT_HEIGHT - 1
            assert layout_reads == 2
        finally:
            await mgr.cleanup()

    async def test_other_manager_does_not_read_the_window(self, tmp_path: Path) -> None:
        """A mouse event reads the window only in the manager that owns the tab."""
        page = AsyncMock()
        page.evaluate.return_value = [_VIEWPORT_WIDTH, _VIEWPORT_HEIGHT]
        session = _layout_session()
        other_session = _layout_session()
        mgr = await _clamped_manager(tmp_path, page, session)
        other_mgr = await _clamped_manager(tmp_path, AsyncMock(), other_session)
        try:
            await mgr.new_page()
            await _send_mouse_event(page, "mouseMove")

            assert _layout_read_count(session) == 1
            assert _layout_read_count(other_session) == 0
        finally:
            await mgr.cleanup()
            await other_mgr.cleanup()

    async def test_event_passes_through_without_a_tracked_session(self, tmp_path: Path) -> None:
        """An event for a page with no tracked session travels where it was aimed."""
        page = AsyncMock()
        page.evaluate.return_value = [_VIEWPORT_WIDTH, _VIEWPORT_HEIGHT]
        mgr = await _clamped_manager(tmp_path, page, _layout_session())
        try:
            await mgr.new_page()
            params = {"x": _BEYOND_VIEWPORT_X, "y": _BEYOND_VIEWPORT_Y}
            await bm_module.Channel.send(_fake_channel(AsyncMock()), "mouseMove", None, params)

            assert params["x"] == _BEYOND_VIEWPORT_X
            assert params["y"] == _BEYOND_VIEWPORT_Y
        finally:
            await mgr.cleanup()

    async def test_clamp_falls_back_to_the_client_size(self, tmp_path: Path) -> None:
        """When the window size is unread, the client size sets the window edge."""
        page = AsyncMock()
        page.evaluate.side_effect = Exception("Page crashed")
        session = MagicMock()

        async def _send(method: str, _params: Optional[dict] = None) -> dict:
            if method == "Page.getLayoutMetrics":
                return {
                    "cssVisualViewport": {
                        "clientWidth": _CLIENT_WIDTH_WITHOUT_SCROLLBAR,
                        "clientHeight": _VIEWPORT_HEIGHT,
                        "pageX": 0.0,
                        "pageY": 0.0,
                    },
                    "cssContentSize": {"width": _CLIENT_WIDTH_WITHOUT_SCROLLBAR, "height": _VIEWPORT_HEIGHT},
                }
            return {}

        session.send = AsyncMock(side_effect=_send)
        session.detach = AsyncMock()
        mgr = await _clamped_manager(tmp_path, page, session)
        try:
            await mgr.new_page()
            params = await _send_mouse_event(page, "mouseMove")

            assert params["x"] == _CLIENT_WIDTH_WITHOUT_SCROLLBAR - 1
            assert params["y"] == _VIEWPORT_HEIGHT - 1
        finally:
            await mgr.cleanup()

    async def test_events_pass_through_when_the_tab_is_forgotten(self, tmp_path: Path) -> None:
        """A tab that is forgotten has no session, and its events pass through."""
        page = _lifecycle_page()
        page.evaluate.return_value = [_VIEWPORT_WIDTH, _VIEWPORT_HEIGHT]
        session = _layout_session()
        mgr = await _clamped_manager(tmp_path, page, session)
        try:
            await mgr.new_page()
            params = await _send_mouse_event(page, "mouseMove")
            assert params["x"] == _VIEWPORT_WIDTH - 1

            await page.fire("close", page)
            await _settle_tasks()
            session.detach.assert_not_awaited()

            again = await _send_mouse_event(page, "mouseMove")
            assert again["x"] == _BEYOND_VIEWPORT_X
            assert again["y"] == _BEYOND_VIEWPORT_Y
        finally:
            await mgr.cleanup()


# ---------------------------------------------------------------------------
# Detaching always finishes
# ---------------------------------------------------------------------------


class TestDetachBounded:
    """Tests that letting go of a session finishes even when the session never answers."""

    @pytest.mark.timeout(30)
    async def test_detach_hang_is_bounded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Closing a tab whose session never answers still finishes."""
        monkeypatch.setattr(bm_module, "_CLEANUP_STEP_TIMEOUT_S", _TEST_CLEANUP_STEP_TIMEOUT_S)
        mgr, target_id, session = await _hanging_manager(tmp_path)

        async def _hang() -> None:
            await asyncio.Event().wait()

        session.detach = _hang
        t0 = time.monotonic()
        assert await mgr.close_page(target_id) is False
        assert time.monotonic() - t0 < 5, "detach hung past its bound"


class TestBrowserManagerRawPageStrings:
    """Tests that title and url give back what the page says, raw."""

    async def test_title_returns_raw_page_title(self: Self) -> None:
        """The title comes back exactly as the page says it."""
        mgr, mock_page, tab = await _make_manager()
        mock_page.title.return_value = 'A "quoted" title; with punctuation'

        result = await mgr.title(tab)

        assert result == 'A "quoted" title; with punctuation'

    async def test_url_returns_raw_page_url(self: Self) -> None:
        """The address comes back exactly as the page says it."""
        mgr, mock_page, tab = await _make_manager()
        mock_page.url = "http://test/?q=a b&x=1"

        result = mgr.url(tab)

        assert result == "http://test/?q=a b&x=1"


# ---------------------------------------------------------------------------
# Moving the mouse in a real browser
# ---------------------------------------------------------------------------


@pytest.mark.live
@skip_if_no_chrome
class TestMoveMouseToLive:
    """Live tests for moving the mouse."""

    async def test_move_mouse_to_moves_mouse(self, browser_manager: BrowserManager) -> None:
        """The mouse really lands on the given spot."""
        tid = await browser_manager.new_page()
        try:
            await _navigate_blank_and_track(browser_manager, tid)

            result = await browser_manager.move_mouse_to(tid, 500, 300)
            assert result is True

            await _assert_mouse_at(browser_manager, tid, 500, 300)
        finally:
            await browser_manager.close_page(tid)

    async def test_move_mouse_to_caps_negative_to_zero(self, browser_manager: BrowserManager) -> None:
        """A negative spot really lands in the top left corner."""
        tid = await browser_manager.new_page()
        try:
            await _navigate_blank_and_track(browser_manager, tid)

            result = await browser_manager.move_mouse_to(tid, -100, -50)
            assert result is True

            await _assert_mouse_at(browser_manager, tid, 0, 0)
        finally:
            await browser_manager.close_page(tid)

    async def test_move_mouse_to_caps_beyond_viewport(self, browser_manager: BrowserManager) -> None:
        """A spot past the window edge really lands on the edge."""
        tid = await browser_manager.new_page()
        try:
            await _navigate_blank_and_track(browser_manager, tid)

            vp = await _get_viewport(browser_manager, tid)

            result = await browser_manager.move_mouse_to(tid, 99999, 99999)
            assert result is True

            await _assert_mouse_at(browser_manager, tid, vp["w"], vp["h"])
        finally:
            await browser_manager.close_page(tid)

    async def test_move_mouse_to_unknown_target(self, browser_manager: BrowserManager) -> None:
        """Moving the mouse in a tab that is not there gives back false."""
        result = await browser_manager.move_mouse_to("nonexistent", 100, 100)
        assert result is False

    async def test_move_mouse_to_no_scroll(self, browser_manager: BrowserManager, test_server_url: str) -> None:
        """Moving the mouse leaves the page where it is."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/long")
            time.sleep(_RENDER_DELAY)

            await browser_manager.scroll_to(tid, 50, _BEYOND_VIEWPORT_Y)
            time.sleep(_SCROLL_DELAY)
            scroll_before = await browser_manager.evaluate(tid, "() => window.scrollY")
            assert scroll_before > 0, f"Should have scrolled down, scrollY={scroll_before}"

            await browser_manager.move_mouse_to(tid, 100, 100)

            scroll_after = await browser_manager.evaluate(tid, "() => window.scrollY")
            assert scroll_after == scroll_before, \
                f"move_mouse_to should not scroll: {scroll_before} -> {scroll_after}"
        finally:
            await browser_manager.close_page(tid)


# ---------------------------------------------------------------------------
# Moving the mouse by fraction in a real browser
# ---------------------------------------------------------------------------


@pytest.mark.live
@skip_if_no_chrome
class TestMoveMouseToRelativeLive:
    """Live tests for moving the mouse by fraction."""

    async def test_move_mouse_to_relative_center(self, browser_manager: BrowserManager) -> None:
        """A fraction of one half really lands in the middle."""
        tid = await browser_manager.new_page()
        try:
            await _navigate_blank_and_track(browser_manager, tid)

            vp = await _get_viewport(browser_manager, tid)

            result = await browser_manager.move_mouse_to_relative(tid, _CENTER_RATIO, _CENTER_RATIO)
            assert result is True

            await _assert_mouse_at(browser_manager, tid, vp["w"] * _CENTER_RATIO, vp["h"] * _CENTER_RATIO)
        finally:
            await browser_manager.close_page(tid)

    async def test_move_mouse_to_relative_top_left(self, browser_manager: BrowserManager) -> None:
        """A fraction of zero really lands in the top left corner."""
        tid = await browser_manager.new_page()
        try:
            await _navigate_blank_and_track(browser_manager, tid)

            result = await browser_manager.move_mouse_to_relative(tid, 0, 0)
            assert result is True

            await _assert_mouse_at(browser_manager, tid, 0, 0)
        finally:
            await browser_manager.close_page(tid)

    async def test_move_mouse_to_relative_bottom_right(self, browser_manager: BrowserManager) -> None:
        """A fraction of one really lands in the bottom right corner."""
        tid = await browser_manager.new_page()
        try:
            await _navigate_blank_and_track(browser_manager, tid)

            vp = await _get_viewport(browser_manager, tid)

            result = await browser_manager.move_mouse_to_relative(tid, 1, 1)
            assert result is True

            await _assert_mouse_at(browser_manager, tid, vp["w"], vp["h"])
        finally:
            await browser_manager.close_page(tid)

    async def test_move_mouse_to_relative_invalid_negative(self, browser_manager: BrowserManager) -> None:
        """A fraction below zero gives back false."""
        result = await browser_manager.move_mouse_to_relative("tab1", -0.1, _CENTER_RATIO)
        assert result is False

    async def test_move_mouse_to_relative_invalid_over_one(self, browser_manager: BrowserManager) -> None:
        """A fraction above one gives back false."""
        result = await browser_manager.move_mouse_to_relative("tab1", _CENTER_RATIO, 1.1)
        assert result is False

    async def test_move_mouse_to_relative_unknown_target(self, browser_manager: BrowserManager) -> None:
        """Moving by fraction in a tab that is not there gives back false."""
        result = await browser_manager.move_mouse_to_relative("nonexistent", _CENTER_RATIO, _CENTER_RATIO)
        assert result is False


# ---------------------------------------------------------------------------
# The window geometry a snapshot captures, in a real browser
# ---------------------------------------------------------------------------

_GEOMETRY_PAGE_HTML = f"""<!DOCTYPE html>
<html>
<head><title>geometry</title></head>
<body style="margin: 0;">
  <div id="spacer" style="height: {_GEOMETRY_PAGE_HEIGHT}px; width: 100%;"></div>
  <div id="{_GEOMETRY_TARGET_ID}" style="position: absolute; left: 50%; margin-left: -100px; width: 200px; height: 60px; background: red;"></div>
  <script>
    const el = document.getElementById("{_GEOMETRY_TARGET_ID}");
    el.style.top = ({_GEOMETRY_SCROLL_Y} + (window.innerHeight - el.offsetHeight) / 2) + "px";
    document.addEventListener("click", (event) => {{ window.__lastClicked = event.target.id; }}, true);
  </script>
</body>
</html>
"""


@pytest.mark.live
@skip_if_no_chrome
class TestViewportGeometryLive:
    """Live tests for the window geometry a snapshot captures."""

    async def test_snapshot_page_y_and_click_at_relative_center(self, browser_manager: BrowserManager, tmp_path: Path) -> None:
        """A scrolled snapshot reports where the window is, and a share click lands at the window center."""
        page_file = tmp_path / "geometry.html"
        page_file.write_text(_GEOMETRY_PAGE_HTML)
        mgr = browser_manager
        tid = await mgr.new_page()
        try:
            assert await mgr.navigate(tid, page_file.as_uri()) is True
            await mgr.evaluate(tid, f"() => {{ window.scrollTo(0, {_GEOMETRY_SCROLL_Y}); return true; }}")
            time.sleep(_SCROLL_DELAY)

            result = await mgr.snapshot(tid)
            assert abs(result.viewport.page_y - _GEOMETRY_SCROLL_Y) < _GEOMETRY_PAGE_Y_TOLERANCE

            assert await mgr.click_at_relative(tid, _CENTER_RATIO, _CENTER_RATIO, viewport=result.viewport) is True
            time.sleep(_RENDER_DELAY)
            clicked = await mgr.evaluate(tid, "() => window.__lastClicked")
            assert clicked == _GEOMETRY_TARGET_ID
        finally:
            await mgr.close_page(tid)


class TestSolveCaptcha:
    """Tests for solving a captcha through the manager."""

    async def test_solve_captcha_no_captcha_returns_true(self: Self) -> None:
        """A page with no captcha counts as solved."""
        mgr, mock_page, tab = await _make_manager()
        _screenshot_to_disk(mock_page)

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.return_value = (
                '{"has_captcha": false, "captcha_type": "unknown", "reason": "no captcha"}',
                0,
            )
            result = await mgr.solve_captcha(tab)

        assert result is True

    async def test_solve_captcha_unsolvable_returns_false(self: Self) -> None:
        """A captcha of a kind the model does not know is not solved."""
        mgr, mock_page, tab = await _make_manager()
        _screenshot_to_disk(mock_page)

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.return_value = (
                '{"has_captcha": true, "captcha_type": "unknown", "reason": "unrecognized"}',
                0,
            )
            result = await mgr.solve_captcha(tab)

        assert result is False

    async def test_solve_captcha_unknown_target_returns_false(self: Self) -> None:
        """A tab that is not there is not solved."""
        mgr, _, _tab = await _make_manager()
        assert await mgr.solve_captcha("nonexistent") is False


# ---------------------------------------------------------------------------
# Tab lifecycle: adoption and liveness
# ---------------------------------------------------------------------------

_POPUP_OPENER_HTML = (
    '<!DOCTYPE html><html><body>'
    '<a id="popup" href="popup.html" target="_blank">open</a>'
    "</body></html>"
)
_POPUP_TARGET_HTML = (
    '<!DOCTYPE html><html><head><title>popup</title></head>'
    "<body><h1>popup</h1></body></html>"
)
_POPUP_SETTLE_S = 0.2
_POPUP_WAIT_S = 10.0


def _lifecycle_page() -> AsyncMock:
    """Make a fake page that records its listeners and fires them on demand."""
    page = AsyncMock()
    listeners: dict[str, list[Callable[..., None]]] = {}
    events_seen: list[str] = []

    def _record(event: str, handler: Callable[..., None]) -> None:
        listeners.setdefault(event, []).append(handler)

    async def _fire(event: str, arg: AsyncMock) -> None:
        events_seen.append(event)
        for handler in list(listeners.get(event, [])):
            result = handler(arg)
            if inspect.iscoroutine(result):
                await result

    async def _close() -> None:
        page.is_closed.return_value = True
        await _fire("close", page)

    page.is_closed = MagicMock(return_value=False)
    page.on = _record
    page.close = AsyncMock(side_effect=_close)
    page.fire = _fire
    page.events_seen = events_seen
    return page


def _lifecycle_context(created: list[AsyncMock]) -> AsyncMock:
    """Make a fake context that hands out fresh fake pages and shows them to its listeners."""
    ctx = AsyncMock()
    listeners: dict[str, list[Callable[..., None]]] = {}

    def _record(event: str, handler: Callable[..., None]) -> None:
        listeners.setdefault(event, []).append(handler)

    async def _new_page() -> AsyncMock:
        page = _lifecycle_page()
        created.append(page)
        for handler in list(listeners.get("page", [])):
            handler(page)
        return page

    ctx.on = _record
    ctx.new_page = _new_page
    return ctx


async def _owned_tab(tmp_path: Path) -> tuple[BrowserManager, str, AsyncMock, AsyncMock]:
    """Make a connected manager holding one owned tab and give back the manager, the id, the tab, and the context."""
    created: list[AsyncMock] = []
    ctx = _lifecycle_context(created)
    mgr = await _created_manager(ctx, str(tmp_path))
    target_id = await mgr.new_page()
    return mgr, target_id, created[0], ctx


async def _settle_tasks() -> None:
    """Give fire-and-forget tasks a turn to finish."""
    for _ in range(10):
        await asyncio.sleep(0)


class TestTabAdoption:
    """Tests that tabs opened by an owned tab are owned too."""

    async def test_popup_of_owned_tab_is_adopted(self, tmp_path: Path) -> None:
        """A tab opened by an owned tab enters the owned set."""
        mgr, _opener_id, opener, _ = await _owned_tab(tmp_path)
        popup = _lifecycle_page()

        await opener.fire("popup", popup)

        assert len(mgr.tab_ids()) == 2

    async def test_popup_of_adopted_tab_is_adopted_transitively(self, tmp_path: Path) -> None:
        """A tab opened by an adopted tab is adopted too."""
        mgr, _opener_id, opener, _ = await _owned_tab(tmp_path)
        middle = _lifecycle_page()
        leaf = _lifecycle_page()

        await opener.fire("popup", middle)
        await middle.fire("popup", leaf)

        assert len(mgr.tab_ids()) == 3

    async def test_tab_with_no_owned_opener_is_not_adopted(self, tmp_path: Path) -> None:
        """A tab seen only through a context-wide event stays out of the other manager's set."""
        seen: list[AsyncMock] = []
        ctx = _lifecycle_context([])
        ctx.on("page", seen.append)
        mgr_a = await _created_manager(ctx, str(tmp_path))
        mgr_b = await _created_manager(ctx, str(tmp_path))

        try:
            await mgr_a.new_page()
            await mgr_b.new_page()

            assert len(seen) == 2
            assert len(mgr_a.tab_ids()) == 1
            assert len(mgr_b.tab_ids()) == 1
        finally:
            await mgr_a.cleanup()
            await mgr_b.cleanup()

    async def test_failing_adoption_adopts_nothing(self, tmp_path: Path) -> None:
        """A tab that cannot be adopted is left out of the owned set."""
        mgr, _opener_id, opener, _ = await _owned_tab(tmp_path)
        popup = _lifecycle_page()

        def _refuse(_event: str, _handler: Callable[..., None]) -> None:
            raise RuntimeError("no listeners for me")

        popup.on = _refuse

        await opener.fire("popup", popup)

        assert len(mgr.tab_ids()) == 1


class TestTabLiveness:
    """Tests that the owned set drops a tab when it closes or crashes."""

    async def test_close_event_drops_the_tab(self, tmp_path: Path) -> None:
        """A close event takes the tab out of the owned set."""
        mgr, target_id, page, _ = await _owned_tab(tmp_path)

        await page.fire("close", page)
        await _settle_tasks()

        assert target_id not in mgr.tab_ids()

    async def test_crash_event_closes_the_tab_without_detaching_its_session(self, tmp_path: Path) -> None:
        """A crash event closes the tab and does not detach its session."""
        mgr, target_id, page, ctx = await _owned_tab(tmp_path)

        await page.fire("crash", page)
        await _settle_tasks()

        assert target_id not in mgr.tab_ids()
        assert "close" in page.events_seen
        assert page.close.call_count == 1
        session = ctx.new_cdp_session.return_value
        session.detach.assert_not_awaited()

    async def test_close_page_on_dropped_tab_returns_false(self, tmp_path: Path) -> None:
        """Closing a tab that already left the owned set gives back false."""
        mgr, target_id, page, _ = await _owned_tab(tmp_path)

        await page.fire("close", page)

        assert await mgr.close_page(target_id) is False

    async def test_cleanup_closes_adopted_tabs(self, tmp_path: Path) -> None:
        """Cleaning up closes tabs that were adopted, not just created."""
        mgr, _opener_id, opener, _ = await _owned_tab(tmp_path)
        popup = _lifecycle_page()

        await opener.fire("popup", popup)

        await mgr.cleanup()

        assert mgr.tab_ids() == []
        assert opener.close.call_count == 1
        assert popup.close.call_count == 1


def _write_popup_pages(tmp_path: Path) -> str:
    """Put the popup pages in a folder and give back the opener's file address."""
    (tmp_path / "popup.html").write_text(_POPUP_TARGET_HTML)
    opener = tmp_path / "opener.html"
    opener.write_text(_POPUP_OPENER_HTML)
    return opener.as_uri()


async def _wait_for_page_count(mgr: BrowserManager, count: int, timeout_s: float = _POPUP_WAIT_S) -> None:
    """Wait until the owned set holds at least this many tabs."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if len(mgr.tab_ids()) >= count:
            return
        await asyncio.sleep(_POPUP_SETTLE_S)
    assert len(mgr.tab_ids()) >= count, f"expected {count} tabs, got {len(mgr.tab_ids())}"


async def _wait_until_gone(mgr: BrowserManager, target_id: str, timeout_s: float = _POPUP_WAIT_S) -> None:
    """Wait until a tab leaves the owned set."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if target_id not in mgr.tab_ids():
            return
        await asyncio.sleep(_POPUP_SETTLE_S)
    assert target_id not in mgr.tab_ids(), "tab did not leave the owned set"


def _popup_ids(mgr: BrowserManager, opener_id: str) -> list[str]:
    """Give back the ids of the tabs that are not the opener's."""
    return [target_id for target_id in mgr.tab_ids() if target_id != opener_id]


async def _paired_managers() -> tuple[BrowserManager, BrowserManager]:
    """Make two managers that share the running browser."""
    return await BrowserManager.create(), await BrowserManager.create()


async def _release_managers(mgr_a: BrowserManager, mgr_b: BrowserManager) -> None:
    """Let both managers go of their tabs and the browser."""
    await mgr_a.cleanup()
    await mgr_b.cleanup()


@pytest.mark.live
@skip_if_no_chrome
class TestLiveTabLifecycle:
    """Tab adoption and liveness in a real browser."""

    async def test_popup_is_adopted_and_closed_by_cleanup(self, browser_manager: BrowserManager, tmp_path: Path) -> None:
        """A tab opened by an owned tab is adopted, and cleanup closes it."""
        mgr = browser_manager
        opener_uri = _write_popup_pages(tmp_path)
        target_id = await mgr.new_page()
        try:
            assert await mgr.navigate(target_id, opener_uri) is True
            assert await mgr.click(target_id, "#popup") is True
            await _wait_for_page_count(mgr, 2)
            ids = _popup_ids(mgr, target_id)
            assert len(ids) == 1
            shot = str(tmp_path / "popup.jpg")
            assert await mgr.screenshot(ids[0], shot) == shot
            await mgr.cleanup()
            assert mgr.tab_ids() == []
            assert await mgr.close_page(ids[0]) is False
        finally:
            await mgr.cleanup()

    async def test_popup_adopted_only_by_its_opener_manager(self, tmp_path: Path) -> None:
        """One manager's popup is not adopted or closed by another manager."""
        opener_uri = _write_popup_pages(tmp_path)
        mgr_a, mgr_b = await _paired_managers()
        try:
            tab_a = await mgr_a.new_page()
            assert await mgr_a.navigate(tab_a, opener_uri) is True
            assert await mgr_a.click(tab_a, "#popup") is True
            await _wait_for_page_count(mgr_a, 2)
            ids = _popup_ids(mgr_a, tab_a)
            assert len(ids) == 1
            assert ids[0] not in mgr_b.tab_ids()
            await mgr_b.cleanup()
            shot = str(tmp_path / "after_b_cleanup.jpg")
            assert await mgr_a.screenshot(ids[0], shot) == shot
            assert ids[0] in mgr_a.tab_ids()
        finally:
            await _release_managers(mgr_a, mgr_b)

    async def test_popup_of_other_manager_survives_cleanup(self, tmp_path: Path) -> None:
        """The mirror case. A manager's cleanup leaves the other manager's popup alive."""
        opener_uri = _write_popup_pages(tmp_path)
        mgr_a, mgr_b = await _paired_managers()
        try:
            tab_b = await mgr_b.new_page()
            assert await mgr_b.navigate(tab_b, opener_uri) is True
            assert await mgr_b.click(tab_b, "#popup") is True
            await _wait_for_page_count(mgr_b, 2)
            ids = _popup_ids(mgr_b, tab_b)
            assert len(ids) == 1
            assert ids[0] not in mgr_a.tab_ids()
            await mgr_a.cleanup()
            shot = str(tmp_path / "after_a_cleanup.jpg")
            assert await mgr_b.screenshot(ids[0], shot) == shot
            await mgr_b.cleanup()
            assert mgr_b.tab_ids() == []
        finally:
            await _release_managers(mgr_a, mgr_b)

    async def test_simultaneous_popups_stay_with_their_own_managers(self, tmp_path: Path) -> None:
        """Two managers opening popups at once each close only their own tabs."""
        opener_uri = _write_popup_pages(tmp_path)
        mgr_a, mgr_b = await _paired_managers()
        try:
            tab_a = await mgr_a.new_page()
            tab_b = await mgr_b.new_page()
            assert await mgr_a.navigate(tab_a, opener_uri) is True
            assert await mgr_b.navigate(tab_b, opener_uri) is True
            await asyncio.gather(mgr_a.click(tab_a, "#popup"), mgr_b.click(tab_b, "#popup"))
            await _wait_for_page_count(mgr_a, 2)
            await _wait_for_page_count(mgr_b, 2)
            ids_a = _popup_ids(mgr_a, tab_a)
            ids_b = _popup_ids(mgr_b, tab_b)
            assert len(ids_a) == 1
            assert len(ids_b) == 1
            assert ids_a[0] != ids_b[0]
            assert ids_a[0] not in mgr_b.tab_ids()
            assert ids_b[0] not in mgr_a.tab_ids()
            await mgr_a.cleanup()
            assert mgr_a.tab_ids() == []
            shot_b = str(tmp_path / "after_a_cleanup.jpg")
            assert await mgr_b.screenshot(ids_b[0], shot_b) == shot_b
            await mgr_b.cleanup()
            assert mgr_b.tab_ids() == []
        finally:
            await _release_managers(mgr_a, mgr_b)

    async def test_self_closed_tab_leaves_the_owned_set(self, browser_manager: BrowserManager) -> None:
        """A tab that closes itself leaves the owned set on its own."""
        mgr = browser_manager
        target_id = await mgr.new_page()
        assert await mgr.navigate(target_id, "about:blank") is True
        assert await mgr.evaluate(target_id, "() => { window.close(); return true; }") is True

        await _wait_until_gone(mgr, target_id)
        assert await mgr.close_page(target_id) is False
        await mgr.cleanup()

        assert mgr.tab_ids() == []


_SHOT_TIMEOUT_S = 45.0
_SHOT_TIMEOUT_MS = 45000
_DEFAULT_SHOT_TIMEOUT_MS = 30000


@pytest.mark.parametrize(
    ("timeout_arg", "expected_ms"),
    [
        pytest.param(_SHOT_TIMEOUT_S, _SHOT_TIMEOUT_MS, id="given"),
        pytest.param(None, _DEFAULT_SHOT_TIMEOUT_MS, id="default"),
    ],
)
async def test_screenshot_forwards_timeout_in_milliseconds(timeout_arg: Optional[float], expected_ms: int) -> None:
    """The wait the caller gives the shot reaches the page in milliseconds."""
    mgr, mock_page, tab = await _make_manager()
    mock_page.screenshot = AsyncMock()
    if timeout_arg is None:
        await mgr.screenshot(tab, "/tmp/shot.jpg")
    else:
        await mgr.screenshot(tab, "/tmp/shot.jpg", timeout=timeout_arg)
    assert mock_page.screenshot.call_args[1]["timeout"] == expected_ms


class TestUploadWithChooserRelative:
    """Tests for uploading through a picker opened at a screen point."""

    async def test_uploads_files_after_click(self) -> None:
        """The picker is captured before the click and gets the files."""
        mgr, page, tab = await _make_manager()
        chooser = MagicMock()
        chooser.set_files = AsyncMock()
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=SimpleNamespace(value=asyncio.sleep(0, result=chooser)))
        context.__aexit__ = AsyncMock(return_value=False)
        page.expect_file_chooser = MagicMock(return_value=context)
        events: list[str] = []

        async def click(*_args: float, **_kwargs: bool) -> bool:
            events.append("click")
            return True

        mgr.click_at_relative = AsyncMock(side_effect=click)
        result = await mgr.upload_with_chooser_relative(
            tab, _CENTER_RATIO, _CENTER_RATIO, ["/tmp/img.jpg"], viewport=_unscrolled_viewport(),
        )

        assert result is True
        page.expect_file_chooser.assert_called_once_with(timeout=10000)
        assert events == ["click"]
        mgr.click_at_relative.assert_awaited_once_with(
            tab, _CENTER_RATIO, _CENTER_RATIO, viewport=_unscrolled_viewport(), verbose=False,
        )
        chooser.set_files.assert_awaited_once_with(["/tmp/img.jpg"])

    async def test_failed_click_does_not_upload(self) -> None:
        """A failed click stops before any files reach the picker."""
        mgr, page, tab = await _make_manager()
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=SimpleNamespace(value=None))
        context.__aexit__ = AsyncMock(return_value=False)
        page.expect_file_chooser = MagicMock(return_value=context)
        mgr.click_at_relative = AsyncMock(return_value=False)

        assert await mgr.upload_with_chooser_relative(tab, _CENTER_RATIO, _CENTER_RATIO, ["/tmp/img.jpg"]) is False


def _field_evaluate(selected: dict[str, str | bool], final_text: str) -> AsyncMock:
    """Make a fake page script runner for a text field.

    It answers the window size read, the select script that gets the old and
    new text, and the read of the field text after the edit.
    """
    async def _answer(expression: str, arg: Optional[list[str]] = None) -> dict[str, str | bool] | list[int] | str:
        if "innerWidth" in expression:
            return [_VIEWPORT_WIDTH, _VIEWPORT_HEIGHT]
        if arg is not None:
            return selected
        return final_text
    return AsyncMock(side_effect=_answer)


def _selected_ok(expected: str = _REPLACED_FIELD_TEXT, selected_all: bool = False) -> dict[str, str | bool]:
    """The select script answer for a single match."""
    return {"status": "ok", "expected": expected, "selectedAll": selected_all}


def _field_frame(evaluate: AsyncMock, focused_frame: Optional[MagicMock] = None, holds_focus: bool = False) -> MagicMock:
    """Make a fake frame that runs page scripts.

    With holds_focus, focus sits on an inner frame element that leads to
    focused_frame, or to no frame when focused_frame is None.
    """
    element = None
    if holds_focus:
        element = MagicMock()
        element.content_frame = AsyncMock(return_value=focused_frame)
    handle = MagicMock()
    handle.as_element.return_value = element
    handle.dispose = AsyncMock()
    frame = MagicMock()
    frame.evaluate = evaluate
    frame.evaluate_handle = AsyncMock(return_value=handle)
    return frame


async def _make_field_manager(evaluate: AsyncMock, session: Optional[MagicMock] = None) -> tuple[BrowserManager, AsyncMock, str]:
    """Make a manager whose page has focus in its main frame."""
    mgr, mock_page, tab = await _make_manager(evaluate=evaluate, session=session)
    mock_page.main_frame = _field_frame(evaluate)
    return mgr, mock_page, tab


class TestBrowserManagerReplaceInputTextAt:
    """Tests for replacing a piece of a field's text at a spot on the page."""

    async def test_replace_input_text_at_clicks_selects_then_inserts(self: Self) -> None:
        """The spot is clicked, the old text selected, and the new text put in."""
        evaluate = _field_evaluate(_selected_ok(), _REPLACED_FIELD_TEXT)
        mgr, mock_page, tab = await _make_field_manager(evaluate)

        result = await mgr.replace_input_text_at(tab, 100, 200, "quick", "slow")

        assert result is True
        mock_page.mouse.click.assert_called_once_with(100, 200)
        assert [c.args[1] for c in evaluate.call_args_list if len(c.args) > 1] == [["quick", "slow"]]
        mock_page.keyboard.insert_text.assert_called_once_with("slow")
        mock_page.keyboard.press.assert_not_called()

    async def test_replace_input_text_at_empty_new_text_presses_delete(self: Self) -> None:
        """An empty new text removes the selected old text with the Delete key."""
        evaluate = _field_evaluate(_selected_ok("the  brown fox"), "the  brown fox")
        mgr, mock_page, tab = await _make_field_manager(evaluate)

        result = await mgr.replace_input_text_at(tab, 100, 200, "quick", "")

        assert result is True
        mock_page.keyboard.press.assert_called_once_with("Delete")
        mock_page.keyboard.insert_text.assert_not_called()

    async def test_replace_input_text_at_selected_all_inserts_whole_text(self: Self) -> None:
        """A field that can only select all of its text gets the whole new text."""
        evaluate = _field_evaluate(_selected_ok(selected_all=True), _REPLACED_FIELD_TEXT)
        mgr, mock_page, tab = await _make_field_manager(evaluate)

        result = await mgr.replace_input_text_at(tab, 100, 200, "quick", "slow")

        assert result is True
        mock_page.keyboard.insert_text.assert_called_once_with(_REPLACED_FIELD_TEXT)
        mock_page.keyboard.press.assert_not_called()

    async def test_replace_input_text_at_selected_all_empty_result_presses_delete(self: Self) -> None:
        """Removing all the text of a field that selects all uses the Delete key."""
        mgr, mock_page, tab = await _make_field_manager(_field_evaluate(_selected_ok("", selected_all=True), ""))

        result = await mgr.replace_input_text_at(tab, 100, 200, _FIELD_TEXT, "")

        assert result is True
        mock_page.keyboard.press.assert_called_once_with("Delete")
        mock_page.keyboard.insert_text.assert_not_called()

    @pytest.mark.parametrize(
        "status",
        ["old text not found", "old text found more than once", "not an editable field"],
    )
    async def test_replace_input_text_at_failed_selection(self, status: str) -> None:
        """A selection that does not find one copy of the old text changes nothing."""
        mgr, mock_page, tab = await _make_field_manager(_field_evaluate({"status": status}, _FIELD_TEXT))

        result = await mgr.replace_input_text_at(tab, 100, 200, "quick", "slow")

        assert result is False
        mock_page.keyboard.insert_text.assert_not_called()
        mock_page.keyboard.press.assert_not_called()

    async def test_replace_input_text_at_text_mismatch_after_edit(self: Self) -> None:
        """The edit gives back false when the field does not hold the expected text."""
        mgr, mock_page, tab = await _make_field_manager(_field_evaluate(_selected_ok(), _FIELD_TEXT))

        result = await mgr.replace_input_text_at(tab, 100, 200, "quick", "slow")

        assert result is False
        mock_page.keyboard.insert_text.assert_called_once_with("slow")

    async def test_replace_input_text_at_empty_old_text(self: Self) -> None:
        """An empty old text is turned down before anything is clicked."""
        mgr, mock_page, tab = await _make_field_manager(_field_evaluate(_selected_ok(), _REPLACED_FIELD_TEXT))

        result = await mgr.replace_input_text_at(tab, 100, 200, "", "slow")

        assert result is False
        mock_page.mouse.click.assert_not_called()

    async def test_replace_input_text_at_click_failure(self: Self) -> None:
        """The edit gives back false and selects nothing when the click fails."""
        session = _layout_session()
        session.send.side_effect = Exception("CDP down")
        evaluate = _field_evaluate(_selected_ok(), _REPLACED_FIELD_TEXT)
        mgr, mock_page, tab = await _make_field_manager(evaluate, session=session)

        result = await mgr.replace_input_text_at(tab, 100, 200, "quick", "slow")

        assert result is False
        assert all(len(c.args) == 1 for c in evaluate.call_args_list)
        mock_page.keyboard.insert_text.assert_not_called()

    async def test_replace_input_text_at_keyboard_failure(self: Self) -> None:
        """The edit gives back false when the keyboard blows up."""
        mgr, mock_page, tab = await _make_field_manager(_field_evaluate(_selected_ok(), _REPLACED_FIELD_TEXT))
        mock_page.keyboard.insert_text.side_effect = Exception("Keyboard not available")

        result = await mgr.replace_input_text_at(tab, 100, 200, "quick", "slow")

        assert result is False

    async def test_replace_input_text_at_unknown_target(self: Self) -> None:
        """Editing in a tab that is not there gives back false."""
        mgr = await _empty_manager()

        result = await mgr.replace_input_text_at("nonexistent", 100, 200, "quick", "slow")

        assert result is False

    async def test_replace_input_text_at_edits_in_the_focused_inner_frame(self: Self) -> None:
        """A field inside nested inner frames is selected and read back in its own frame."""
        page_evaluate = _field_evaluate(_selected_ok(), _FIELD_TEXT)
        inner_evaluate = _field_evaluate(_selected_ok(), _REPLACED_FIELD_TEXT)
        inner_frame = _field_frame(inner_evaluate)
        middle_frame = _field_frame(_field_evaluate(_selected_ok(), _FIELD_TEXT), inner_frame, holds_focus=True)
        mgr, mock_page, tab = await _make_manager(evaluate=page_evaluate)
        mock_page.main_frame = _field_frame(page_evaluate, middle_frame, holds_focus=True)

        result = await mgr.replace_input_text_at(tab, 100, 200, "quick", "slow")

        assert result is True
        assert [c.args[1:] for c in inner_evaluate.call_args_list] == [(["quick", "slow"],), ()]
        assert middle_frame.evaluate.call_args_list == []
        assert all(len(c.args) == 1 for c in page_evaluate.call_args_list)
        mock_page.keyboard.insert_text.assert_called_once_with("slow")

    async def test_replace_input_text_at_unreachable_inner_frame(self: Self) -> None:
        """Focus in an inner frame that cannot be reached changes nothing."""
        evaluate = _field_evaluate(_selected_ok(), _REPLACED_FIELD_TEXT)
        mgr, mock_page, tab = await _make_manager(evaluate=evaluate)
        mock_page.main_frame = _field_frame(evaluate, holds_focus=True)

        result = await mgr.replace_input_text_at(tab, 100, 200, "quick", "slow")

        assert result is False
        assert all(len(c.args) == 1 for c in evaluate.call_args_list)
        mock_page.keyboard.insert_text.assert_not_called()
        mock_page.keyboard.press.assert_not_called()


class TestBrowserManagerReplaceInputTextAtRelative:
    """Tests for replacing a piece of a field's text at a spot given as a share of the window."""

    async def test_replace_input_text_at_relative_scrolled_viewport(self: Self) -> None:
        """A share on a scrolled window edits the field at the window center."""
        session = _layout_session(page_x=_SCROLLED_VIEWPORT_PAGE_X, page_y=_SCROLLED_VIEWPORT_PAGE_Y)
        evaluate = _field_evaluate(_selected_ok(), _REPLACED_FIELD_TEXT)
        mgr, mock_page, tab = await _make_field_manager(evaluate, session=session)

        result = await mgr.replace_input_text_at_relative(
            tab, _CENTER_RATIO, _CENTER_RATIO, "quick", "slow", viewport=_scrolled_viewport()
        )

        assert result is True
        mock_page.mouse.click.assert_called_once_with(_HALF_VIEWPORT_WIDTH, _HALF_VIEWPORT_HEIGHT)
        mock_page.keyboard.insert_text.assert_called_once_with("slow")

    async def test_replace_input_text_at_relative_reads_viewport_when_none(self: Self) -> None:
        """With no captured window, the window is read and the edit lands at the window center."""
        session = _layout_session(page_x=_SCROLLED_VIEWPORT_PAGE_X, page_y=_SCROLLED_VIEWPORT_PAGE_Y)
        evaluate = _field_evaluate(_selected_ok(), _REPLACED_FIELD_TEXT)
        mgr, mock_page, tab = await _make_field_manager(evaluate, session=session)

        result = await mgr.replace_input_text_at_relative(tab, _CENTER_RATIO, _CENTER_RATIO, "quick", "slow")

        assert result is True
        session.send.assert_any_call("Page.getLayoutMetrics")
        mock_page.mouse.click.assert_called_once_with(_HALF_VIEWPORT_WIDTH, _HALF_VIEWPORT_HEIGHT)

    async def test_replace_input_text_at_relative_cdp_read_failure(self: Self) -> None:
        """The edit gives back false and changes nothing when the window cannot be read."""
        mgr, mock_page, tab = await _make_manager(attach_session=False)

        result = await mgr.replace_input_text_at_relative(tab, _CENTER_RATIO, _CENTER_RATIO, "quick", "slow")

        assert result is False
        mock_page.keyboard.insert_text.assert_not_called()

    async def test_replace_input_text_at_relative_unknown_target(self: Self) -> None:
        """Editing a share in a tab that is not there gives back false."""
        mgr = await _empty_manager()

        result = await mgr.replace_input_text_at_relative("nonexistent", _CENTER_RATIO, _CENTER_RATIO, "quick", "slow")

        assert result is False


# ---------------------------------------------------------------------------
# Replacing a piece of a field's text
# ---------------------------------------------------------------------------


async def _field_center(browser_manager: BrowserManager, tid: str, selector: str) -> tuple[float, float]:
    """Find the middle of an element, counted on the whole page."""
    center = await browser_manager.evaluate(
        tid,
        f"""() => {{
            const box = document.querySelector("{selector}").getBoundingClientRect();
            return [box.left + box.width / 2 + window.scrollX, box.top + box.height / 2 + window.scrollY];
        }}""",
    )
    return center[0], center[1]


@pytest.mark.live
@skip_if_no_chrome
class TestReplaceInputTextAtLive:
    """Live tests for replacing a piece of a field's text."""

    @pytest.mark.parametrize(
        ("selector", "old_text", "new_text", "read_js", "expected"),
        [
            pytest.param(
                "#field-input", "quick", "slow",
                '() => document.querySelector("#field-input").value', "the slow brown fox",
                id="input",
            ),
            pytest.param(
                "#field-textarea", "lazy", "sleepy",
                '() => document.querySelector("#field-textarea").value', "first line\nthe sleepy dog",
                id="textarea",
            ),
            pytest.param(
                "#field-editable", "o big w", "o small w",
                '() => document.querySelector("#field-editable").textContent', "Hello small world",
                id="contenteditable",
            ),
            pytest.param(
                "#field-input", "quick ", "",
                '() => document.querySelector("#field-input").value', "the brown fox",
                id="delete",
            ),
            pytest.param(
                "#field-email", "exmaple", "example",
                '() => document.querySelector("#field-email").value', "jane@example.com",
                id="email",
            ),
            pytest.param(
                "#field-number", "34", "99",
                '() => document.querySelector("#field-number").value', "1299",
                id="number",
            ),
            pytest.param(
                "#field-email", "jane@exmaple.com", "",
                '() => document.querySelector("#field-email").value', "",
                id="email-delete-all",
            ),
        ],
    )
    async def test_replace_input_text_at_changes_the_field(
        self, browser_manager: BrowserManager, test_server_url: str,
        selector: str, old_text: str, new_text: str, read_js: str, expected: str,
    ) -> None:
        """The one copy of the old text is swapped for the new text, with an input event."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/fields")
            await asyncio.sleep(_RENDER_DELAY)
            x, y = await _field_center(browser_manager, tid, selector)

            result = await browser_manager.replace_input_text_at(tid, x, y, old_text, new_text)

            assert result is True
            assert await browser_manager.evaluate(tid, read_js) == expected
            assert await browser_manager.evaluate(tid, "() => window.inputEvents") >= 1
        finally:
            await browser_manager.close_page(tid)

    async def test_replace_input_text_at_changes_a_field_in_nested_frames(
        self, browser_manager: BrowserManager, test_server_url: str,
    ) -> None:
        """A field inside an inner frame inside another inner frame gets the new text."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/fields")
            await asyncio.sleep(_RENDER_DELAY)
            x, y = await _field_center(browser_manager, tid, "#field-frame")
            read_js = """() => document.querySelector("#field-frame").contentDocument
                .querySelector("iframe").contentDocument.querySelector("input").value"""

            result = await browser_manager.replace_input_text_at(tid, x, y, "wrld", "world")

            assert result is True
            assert await browser_manager.evaluate(tid, read_js) == "hello world"
        finally:
            await browser_manager.close_page(tid)

    async def test_replace_input_text_at_ambiguous_match_leaves_the_field(
        self, browser_manager: BrowserManager, test_server_url: str,
    ) -> None:
        """Old text that shows up twice is turned down and the field stays the same."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/fields")
            await asyncio.sleep(_RENDER_DELAY)
            x, y = await _field_center(browser_manager, tid, "#field-repeat")

            result = await browser_manager.replace_input_text_at(tid, x, y, "cat", "dog")

            assert result is False
            assert await browser_manager.evaluate(tid, '() => document.querySelector("#field-repeat").value') == "cat and cat"
        finally:
            await browser_manager.close_page(tid)

    async def test_replace_input_text_at_ambiguous_match_leaves_an_email_field(
        self, browser_manager: BrowserManager, test_server_url: str,
    ) -> None:
        """Old text that shows up twice in an email field is turned down and the field stays the same."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, test_server_url + "/page/fields")
            await asyncio.sleep(_RENDER_DELAY)
            x, y = await _field_center(browser_manager, tid, "#field-email")

            result = await browser_manager.replace_input_text_at(tid, x, y, "a", "o")

            assert result is False
            assert await browser_manager.evaluate(tid, '() => document.querySelector("#field-email").value') == "jane@exmaple.com"
        finally:
            await browser_manager.close_page(tid)


async def _capture_connect_args(cdp_url: Optional[str] = None, download_dir: Optional[str] = None) -> tuple[str, str]:
    """Connect a manager to a fake Chrome, and give back the address and download folder it used."""
    fake_pw = MagicMock()
    await _created_manager(MagicMock(), download_dir, cdp_url, fake_pw)
    connect_call = fake_pw.chromium.connect_over_cdp.call_args
    return connect_call.args[0], connect_call.kwargs["artifacts_dir"]


class TestConnectDefaults:
    """Tests for the address and download folder used when none is passed in."""

    @pytest.mark.parametrize("env_value", [None, ""])
    async def test_cdp_url_default_when_env_unset_or_empty(
        self, env_value: Optional[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """With CHROME_CDP_URL unset or empty, the manager connects to localhost port 9222."""
        if env_value is None:
            monkeypatch.delenv("CHROME_CDP_URL", raising=False)
        else:
            monkeypatch.setenv("CHROME_CDP_URL", env_value)
        url, _ = await _capture_connect_args(download_dir=str(tmp_path))
        assert url == "http://localhost:9222"

    async def test_cdp_url_from_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """With CHROME_CDP_URL set, the manager connects to that address."""
        monkeypatch.setenv("CHROME_CDP_URL", "http://chrome.internal:9333")
        url, _ = await _capture_connect_args(download_dir=str(tmp_path))
        assert url == "http://chrome.internal:9333"

    async def test_cdp_url_argument_beats_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """An address passed in is used over CHROME_CDP_URL."""
        monkeypatch.setenv("CHROME_CDP_URL", "http://chrome.internal:9333")
        url, _ = await _capture_connect_args(cdp_url="http://127.0.0.1:9444", download_dir=str(tmp_path))
        assert url == "http://127.0.0.1:9444"

    async def test_artifacts_dir_default_is_scratch_downloads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """With no folder passed in, downloads go to the downloads folder under the scratch root."""
        monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
        _, artifacts_dir = await _capture_connect_args()
        assert artifacts_dir == str(tmp_path / "downloads")
        assert (tmp_path / "downloads").is_dir()
