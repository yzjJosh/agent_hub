"""Drive a Chrome browser that is already running.

Opens tabs, acts on pages, and picks up files the page downloads.
Always clean up a manager when you are done with it.

Example:
    async with BrowserManager() as manager:
        target_id = await manager.open_page()
        result = await manager.wait_for_download(target_id, url="https://example.com/file.pdf")
        await manager.close_page(target_id)

    manager = await BrowserManager.create()
    try:
        ...
    finally:
        await manager.cleanup()
"""

from __future__ import annotations

__all__ = ["BrowserManager", "Viewport"]

import asyncio
import base64
import os
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import aiofiles
from cloakbrowser.human import patch_context_async
from cloakbrowser.human.config import resolve_config
from playwright.async_api import Browser, BrowserContext, CDPSession, Page, Playwright, async_playwright

# lint-skip: coding.import-private-module
from playwright._impl._connection import Channel

from app.lib.browser.captcha_solver import solve_captcha as _solve_captcha
from app.lib.browser.downloads import DownloadResult, DownloadTriggered, wait_for_page_download
from app.lib.browser.field_editing import replace_focused_field_text
from app.lib.browser.page_snapshot import PageSnapshot
from app.lib.browser.scrolling import (
    scroll_by_pixels,
    scroll_selector_into_view,
    scroll_to_coordinates,
)
from app.lib.recognizer.agents.playwright.async_control import AsyncChallenger
from app.lib.shared import paths

if TYPE_CHECKING:
    from types import TracebackType

os.environ.setdefault("NODE_NO_WARNINGS", "1")

_DEFAULT_CDP_URL = "http://localhost:9222"
_DEFAULT_TIMEOUT_MS = 5000
_DEFAULT_DOWNLOAD_TIMEOUT_S = 30
_DEFAULT_SCREENSHOT_TIMEOUT_S = 30.0
_DEFAULT_CAPTCHA_MAX_STEPS = 5
_FILE_CHOOSER_TIMEOUT_MS = 10000
_MS_PER_SECOND = 1000
_HTTP_OK_MIN = 200
_HTTP_OK_MAX = 300
_LOG_PREVIEW_LIMIT = 50
_PAGE_CALL_TIMEOUT_S = 30
_CLEANUP_STEP_TIMEOUT_S = 10

# Reads the window box width and height, including scrollbars.
_WINDOW_SIZE_JS = "() => [window.innerWidth, window.innerHeight]"

# Syncs aria-hidden with computed visibility for every element.
_SYNC_ARIA_HIDDEN_JS = """() => {
  document.querySelectorAll('*').forEach(el => {
    const style = getComputedStyle(el);
    const isHidden = style.display === 'none' || style.visibility === 'hidden';
    const attr = el.getAttribute('aria-hidden');
    if (isHidden && attr !== 'true') {
      el.setAttribute('aria-hidden', 'true');
    } else if (!isHidden && attr === 'true') {
      el.removeAttribute('aria-hidden');
    }
  });
}"""

# Connected managers used by the mouse viewport clamp hook.
_CONNECTED_MANAGERS: set[BrowserManager] = set()

_JsValue: TypeAlias = dict | list | str | int | float | bool | None


def _truncate_for_log(text: str) -> str:
    """Shorten a string for verbose log output."""
    if len(text) > _LOG_PREVIEW_LIMIT:
        return text[:_LOG_PREVIEW_LIMIT] + "..."
    return text


@dataclass(frozen=True)
class Viewport:
    """Window size and scroll offset on a page, in pixels.

    The place is where the top left corner of the window sits, counted
    from the top left corner of the page.

    There are two size families. The client size is the area where page
    content renders. The window size is the whole window box, including any
    scrollbar, the ratios scale against it, and mouse input is kept inside
    of it. The content size is the document's full size, the furthest the
    page can scroll plus the client size.
    """

    client_width: float  # visible content width in CSS pixels, excluding any scrollbar
    client_height: float  # visible content height in CSS pixels, excluding any scrollbar
    page_x: float  # horizontal scroll offset from the left edge of the document
    page_y: float  # vertical scroll offset from the top edge of the document
    content_width: float  # full document width in CSS pixels
    content_height: float  # full document height in CSS pixels
    window_width: float = 0.0  # full window width including any scrollbar, zero when unread
    window_height: float = 0.0  # full window height including any scrollbar, zero when unread

    @property
    def max_scroll_x(self) -> float:
        """The furthest the page can scroll, sideways."""
        return max(0.0, self.content_width - self.client_width)

    @property
    def max_scroll_y(self) -> float:
        """The furthest the page can scroll, up or down."""
        return max(0.0, self.content_height - self.client_height)

    @property
    def usable_width(self) -> float:
        """The window width to use, falling back to client width when unread."""
        return self.window_width if self.window_width > 0 else self.client_width

    @property
    def usable_height(self) -> float:
        """The window height to use, falling back to client height when unread."""
        return self.window_height if self.window_height > 0 else self.client_height


@dataclass
class _PageState:
    """Per-tab session, identifier, and cached clamp viewport."""

    session: CDPSession | None
    guid: str
    clamp_viewport: Viewport | None = None


def _install_mouse_viewport_clamp() -> None:
    """Keep every mouse move and click inside the window.

    Chrome throws away a mouse event that lands on or past the edge of
    the window, so the spot is pulled back inside first. This change
    applies to every page in the whole program, and it is made once.
    """
    # lint-skip: coding.reflection,coding.access-private
    if getattr(Channel, "_viewport_clamp_installed", False):
        return

    original_send = Channel.send

    async def _clamped_send(
        channel_self: Channel,
        method: str,
        timeout_calculator: Callable | None = None,
        params: dict | None = None,
        **kwargs: str | bool | None,
    ) -> dict:
        if method in ("mouseMove", "mouseClick") and params:
            page_object = getattr(channel_self, "_object", None)
            if page_object is not None:
                guid = getattr(page_object, "_guid", None)
                if guid is not None:
                    for manager in _CONNECTED_MANAGERS:
                        target_id = manager._tab_guid_to_target_id.get(guid)
                        if target_id is None:
                            continue
                        await _clamp_mouse_params(manager, target_id, params)
                        break

        return await original_send(channel_self, method, timeout_calculator, params, **kwargs)

    Channel.send = _clamped_send
    Channel._viewport_clamp_installed = True
    # lint-end-skip


class BrowserManager:
    """Drives a Chrome browser that is already running."""

    def __init__(self, cdp_url: str | None = None, download_dir: str | None = None, verbose: bool = False) -> None:
        """Save the settings, without connecting.

        Use create or async with to get a manager that is connected.

        Args:
            cdp_url: Where the running Chrome listens, CHROME_CDP_URL when None.
            download_dir: Folder that downloaded files land in, the scratch download folder when None.
            verbose: Print progress lines when True.
        """
        self._cdp_url: str = cdp_url or os.getenv("CHROME_CDP_URL") or _DEFAULT_CDP_URL
        self._download_dir: str = download_dir or paths.download_dir()
        self._verbose: bool = verbose
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._pages: dict[str, Page] = {}
        self._page_states: dict[Page, _PageState] = {}
        self._tab_guid_to_target_id: dict[str, str] = {}
        self._is_cleaned_up: bool = False

    @classmethod
    async def create(cls, cdp_url: str | None = None, download_dir: str | None = None, verbose: bool = False) -> BrowserManager:
        """Make a manager that is connected to Chrome.

        Args:
            cdp_url: Where the running Chrome listens, CHROME_CDP_URL when None.
            download_dir: Folder that downloaded files land in, the scratch download folder when None.
            verbose: Print progress lines when True.

        Returns:
            A manager ready to use.

        Raises:
            RuntimeError: If there is no browser to connect to.
        """
        manager = cls(cdp_url, download_dir, verbose)
        await manager._connect()
        return manager

    async def _connect(self) -> None:
        """Connect to the running Chrome and prepare the browser context."""
        if self._verbose:
            print(f"[BrowserManager] Connecting to CDP: {self._cdp_url}", flush=True)
            print(f"[BrowserManager] Download dir: {self._download_dir}", flush=True)

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.connect_over_cdp(self._cdp_url, artifacts_dir=self._download_dir)
        if not self._browser.contexts:
            await self._browser.close()
            await self._playwright.stop()
            raise RuntimeError(
                f"No browser contexts found at {self._cdp_url}. "
                "Ensure Chrome is running with --remote-debugging-port=9222"
            )
        self._context = self._browser.contexts[0]
        await asyncio.to_thread(os.makedirs, self._download_dir, exist_ok=True)
        self._apply_humanize()
        _install_mouse_viewport_clamp()
        _CONNECTED_MANAGERS.add(self)
        self._is_cleaned_up = False

    def _get_page(self, target_id: str) -> Page:
        """Return the tracked page for a tab id, or raise KeyError."""
        if target_id not in self._pages:
            raise KeyError(f"No page found for target_id: {target_id}")
        return self._pages[target_id]

    def _lookup_page_or_log(self, target_id: str, method_name: str) -> Page | None:
        """Look up a tracked page and log an error to stderr when missing."""
        page = self._pages.get(target_id)
        if page is None:
            print(f"[BrowserManager.{method_name}] Error: target_id not found: {target_id}", file=sys.stderr)
        return page

    async def _resolve_relative_viewport(
        self,
        target_id: str,
        viewport: Viewport | None,
        method_name: str,
    ) -> Viewport | None:
        """Validate the tab and return the supplied or freshly read viewport."""
        if self._lookup_page_or_log(target_id, method_name) is None:
            return None
        if viewport is not None:
            return viewport
        resolved = await self.get_page_viewport(target_id)
        if resolved is None:
            print(f"[BrowserManager.{method_name}] Error: could not read the viewport", file=sys.stderr)
        return resolved

    def _apply_humanize(self) -> None:
        """Apply human-like mouse, keyboard, and scroll behavior when available."""
        try:
            humanize_config = resolve_config("default", None)
            patch_context_async(self._context, humanize_config)
        except Exception as exc:
            print(f"⚠️  Humanize init failed: {exc}. Continuing without humanize.", flush=True)

    async def _attach_tab_session(self, page: Page) -> CDPSession | None:
        """Open the per-tab session that keeps the tab running at foreground speed."""
        session: CDPSession | None = None
        try:
            session = await self._context.new_cdp_session(page)
            await session.send("Emulation.setFocusEmulationEnabled", {"enabled": True})
            await session.send("Page.setWebLifecycleState", {"state": "active"})
            return session
        except Exception as exc:
            print(f"[BrowserManager.tab_session] Error: {exc}", file=sys.stderr)
            if session is not None:
                await self._detach_cdp_session(page, session)
            return None

    async def _track_page(self, page: Page) -> str:
        """Assign an id, attach a tab session, and add the page to owned tabs."""
        target_id: str = uuid.uuid4().hex
        page.on("popup", self._on_popup)
        page.on("close", self._on_page_closed)
        page.on("crash", self._on_page_crashed)
        self._pages[target_id] = page
        # lint-skip: coding.access-private
        tab_guid = page._impl_obj._guid
        self._page_states[page] = _PageState(session=await self._attach_tab_session(page), guid=tab_guid)
        self._tab_guid_to_target_id[tab_guid] = target_id
        return target_id

    def _forget_page(self, page: Page) -> None:
        """Remove a closed page from tracked state."""
        state: _PageState | None = self._page_states.pop(page, None)
        for target_id, known_page in list(self._pages.items()):
            if known_page is page:
                del self._pages[target_id]
                break
        if state is not None:
            self._tab_guid_to_target_id.pop(state.guid, None)

    async def _on_popup(self, popup_page: Page) -> None:
        """Adopt a popup tab opened by an owned tab."""
        try:
            await self._track_page(popup_page)
        except Exception as exc:
            print(f"[BrowserManager.popup] Failed to adopt tab: {exc}", file=sys.stderr)

    def _on_page_closed(self, page: Page) -> None:
        """Drop a tab from tracking as soon as it closes."""
        try:
            self._forget_page(page)
        except Exception as exc:
            print(f"[BrowserManager.close] Failed to forget tab: {exc}", file=sys.stderr)

    def _on_page_crashed(self, page: Page) -> None:
        """Schedule closing a tab whose renderer crashed."""
        target_id = next((known_id for known_id, known_page in self._pages.items() if known_page is page), None)
        if target_id is None:
            return
        asyncio.get_running_loop().create_task(self.close_page(target_id))

    async def open_page(self, verbose: bool = False) -> str:
        """Open a new tab.

        Args:
            verbose: Print progress lines when True.

        Returns:
            The id of the new tab.
        """
        page: Page = await self._context.new_page()
        target_id: str = await self._track_page(page)
        if verbose:
            print(f"[BrowserManager.open_page] Created page: {target_id}", flush=True)
        return target_id

    def list_tab_ids(self) -> list[str]:
        """Return the ids of the tabs this manager owns.

        Returns:
            The ids of the open tabs.
        """
        return list(self._pages)

    async def close_page(self, target_id: str, verbose: bool = False) -> bool:
        """Close a tab. This never hangs.

        If closing fails or takes too long, the tab is dropped here.

        Args:
            target_id: The tab to close.
            verbose: Print progress lines when True.

        Returns:
            True if the tab closed.
        """
        page = self._pages.pop(target_id, None)
        if page is None:
            if verbose:
                print(f"[BrowserManager.close_page] Page {target_id} not found", flush=True)
            return False
        try:
            await asyncio.wait_for(page.close(), timeout=_CLEANUP_STEP_TIMEOUT_S)
            if verbose:
                print(f"[BrowserManager.close_page] Closed {target_id}: success", flush=True)
        except Exception as exc:
            print(f"[BrowserManager.close_page] Error closing {target_id}: {exc}", file=sys.stderr)
            state: _PageState | None = self._page_states.pop(page, None)
            if state is not None:
                self._tab_guid_to_target_id.pop(state.guid, None)
                if state.session is not None:
                    await self._detach_cdp_session(page, state.session)
            return False
        return True

    async def evaluate(self, target_id: str, expression: str, verbose: bool = False) -> _JsValue:
        """Run a bit of code inside the page.

        Args:
            target_id: The tab to run it in.
            expression: The code to run.
            verbose: Print progress lines when True.

        Returns:
            What the code gave back, or None if it failed.
        """
        page = self._get_page(target_id)
        if verbose:
            preview = _truncate_for_log(expression)
            print(f"[BrowserManager.evaluate] Evaluating: {preview}", flush=True)
        try:
            result = await page.evaluate(expression)
            if verbose:
                print(f"[BrowserManager.evaluate] Result type: {type(result)}", flush=True)
            return result
        except Exception as exc:
            print(f"[BrowserManager.evaluate] Error: {exc}", file=sys.stderr)
            return None

    async def click(
        self,
        target_id: str,
        selector: str,
        timeout: int = _DEFAULT_TIMEOUT_MS,
        verbose: bool = False,
        human_config: dict | None = None,
    ) -> bool:
        """Click a part of the page.

        The page scrolls first. Nothing is clicked if the part cannot be
        brought into the window.

        Args:
            target_id: The tab to click in.
            selector: Points at the part to click.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.
            human_config: Settings for how human the click should look.

        Returns:
            True if the click was made.
        """
        page = self._lookup_page_or_log(target_id, "click")
        if page is None:
            return False
        if not await self.scroll_to_view(target_id, selector, verbose=verbose):
            print("[BrowserManager.click] Error: scroll failed", file=sys.stderr)
            return False
        if verbose:
            print(f"[BrowserManager.click] Clicking '{selector}' (timeout={timeout}ms)", flush=True)
        try:
            if human_config is None:
                await page.click(selector, force=True, timeout=timeout)
            else:
                await page.click(selector, force=True, timeout=timeout, human_config=human_config)
            if verbose:
                print("[BrowserManager.click] Success", flush=True)
            return True
        except Exception as exc:
            print(f"[BrowserManager.click] Error: {exc}", file=sys.stderr)
            return False

    async def scroll_to(self, target_id: str, x: float, y: float, verbose: bool = False) -> bool:
        """Scroll until a spot on the page is in the window.

        The spot is measured on the whole page, not on the window.

        Args:
            target_id: The tab to scroll.
            x: How far right, counted from the left edge of the page.
            y: How far down, counted from the top of the page.
            verbose: Print progress lines when True.

        Returns:
            True if the page scrolled.
        """
        page = self._lookup_page_or_log(target_id, "scroll_to")
        if page is None:
            return False
        return await scroll_to_coordinates(
            page,
            x,
            y,
            read_viewport=lambda: self.get_page_viewport(target_id),
            move_mouse=lambda viewport_x, viewport_y: self.move_mouse_to(target_id, viewport_x, viewport_y, verbose=verbose),
            verbose=verbose,
        )

    async def set_viewport_size(self, target_id: str, width: int, height: int, verbose: bool = False) -> bool:
        """Resize the window a tab sees.

        Args:
            target_id: The tab to resize.
            width: Window width in pixels.
            height: Window height in pixels.
            verbose: Print progress lines when True.

        Returns:
            True if the window was resized.
        """
        page = self._lookup_page_or_log(target_id, "set_viewport_size")
        if page is None:
            return False
        if verbose:
            print(f"[BrowserManager.set_viewport_size] {width}x{height}", flush=True)
        try:
            await page.set_viewport_size({"width": width, "height": height})
            return True
        except Exception as exc:
            print(f"[BrowserManager.set_viewport_size] Error: {exc}", file=sys.stderr)
            return False

    async def scroll_to_view(self, target_id: str, selector: str, verbose: bool = False) -> bool:
        """Scroll until a part of the page is in the window.

        A part inside its own scrolling box is reached too.

        Args:
            target_id: The tab to scroll.
            selector: Points at the part to scroll to.
            verbose: Print progress lines when True.

        Returns:
            True if the middle of the part is in the window.
        """
        page = self._lookup_page_or_log(target_id, "scroll_to_view")
        if page is None:
            return False
        return await scroll_selector_into_view(
            page,
            selector,
            read_viewport=lambda: self.get_page_viewport(target_id),
            move_mouse=lambda viewport_x, viewport_y: self.move_mouse_to(target_id, viewport_x, viewport_y, verbose=verbose),
            verbose=verbose,
        )

    async def get_page_viewport(self, target_id: str) -> Viewport | None:
        """Read the window's size and its place on the page.

        Args:
            target_id: The tab to read.

        Returns:
            The window's geometry, or None if it could not be read.
        """
        page = self._pages.get(target_id)
        if page is None:
            return None
        session = self._get_page_session(page)
        if session is None:
            return None
        try:
            metrics = await _await_page_call(session.send("Page.getLayoutMetrics"), "Page.getLayoutMetrics")
            view = metrics["cssVisualViewport"]
            content = metrics["cssContentSize"]
        except Exception as exc:
            print(f"[BrowserManager.get_page_viewport] Error: {exc}", file=sys.stderr)
            return None
        try:
            window_box = await _await_page_call(page.evaluate(_WINDOW_SIZE_JS), "window size")
            window_width, window_height = window_box[0], window_box[1]
        except Exception as exc:
            print(f"[BrowserManager.get_page_viewport] window size read failed: {exc}", file=sys.stderr)
            window_width, window_height = 0.0, 0.0
        return Viewport(
            client_width=view["clientWidth"],
            client_height=view["clientHeight"],
            page_x=view["pageX"],
            page_y=view["pageY"],
            content_width=content["width"],
            content_height=content["height"],
            window_width=window_width,
            window_height=window_height,
        )

    def _get_page_session(self, page: Page) -> CDPSession | None:
        """Return the attached session for a tracked page."""
        state: _PageState | None = self._page_states.get(page)
        return state.session if state is not None else None

    async def _get_clamp_viewport(self, target_id: str) -> Viewport | None:
        """Read and cache the window size used by the mouse clamp for one tab."""
        page = self._pages.get(target_id)
        if page is None:
            return None
        state: _PageState | None = self._page_states.get(page)
        if state is None:
            return None
        viewport = state.clamp_viewport
        if viewport is None:
            viewport = await self.get_page_viewport(target_id)
            if viewport is not None:
                state.clamp_viewport = viewport
        return viewport

    async def scroll_by(self, target_id: str, distance: float, dx: float = 0.0, verbose: bool = False) -> bool:
        """Scroll the page a set number of pixels, sideways too if a number is given.

        Gives up if the page stops answering for 5 seconds or after 30 seconds in total.
        A failed scroll may leave the page partly scrolled.

        Args:
            target_id: The tab to scroll.
            distance: How far to scroll up or down. A positive number scrolls down.
            dx: How far to scroll sideways. A positive number scrolls right.
            verbose: Print progress lines when True.

        Returns:
            True if the page scrolled the whole way.
        """
        page = self._lookup_page_or_log(target_id, "scroll_by")
        if page is None:
            return False
        return await scroll_by_pixels(page, distance, dx=dx, verbose=verbose)

    async def scroll_by_relative(
        self,
        target_id: str,
        dx: float,
        dy: float,
        viewport: Viewport | None = None,
        verbose: bool = False,
    ) -> bool:
        """Scroll the page a fraction of the window in each direction.

        Args:
            target_id: The tab to scroll.
            dx: How far sideways, 1 is one window wide. A positive number scrolls right.
            dy: How far up or down, 1 is one window tall. A positive number scrolls down.
            viewport: The window's size and place on the page, or None to read it.
            verbose: Print progress lines when True.

        Returns:
            True if the page scrolled. False if the window could not be read.
        """
        resolved_viewport = await self._resolve_relative_viewport(target_id, viewport, "scroll_by_relative")
        if resolved_viewport is None:
            return False
        return await self.scroll_by(
            target_id,
            dy * resolved_viewport.usable_height,
            dx=dx * resolved_viewport.usable_width,
            verbose=verbose,
        )

    async def move_mouse_to(self, target_id: str, x: float, y: float, verbose: bool = False) -> bool:
        """Move the mouse to a spot in the window.

        The page does not scroll. A spot outside the window is pulled back
        to the edge.

        Args:
            target_id: The tab to move the mouse in.
            x: How far right, counted from the left edge of the window.
            y: How far down, counted from the top of the window.
            verbose: Print progress lines when True.

        Returns:
            True if the mouse moved.
        """
        page = self._lookup_page_or_log(target_id, "move_mouse_to")
        if page is None:
            return False
        try:
            viewport = await self.get_page_viewport(target_id)
            if viewport is None:
                print("[BrowserManager.move_mouse_to] Error: could not read the viewport", file=sys.stderr)
                return False
            viewport_width = viewport.usable_width
            viewport_height = viewport.usable_height
            clamped_x = max(0.0, min(x, viewport_width))
            clamped_y = max(0.0, min(y, viewport_height))
            if verbose:
                print(
                    f"[BrowserManager.move_mouse_to] Viewport ({x}, {y}) -> "
                    f"capped ({clamped_x}, {clamped_y}), viewport size ({viewport_width}, {viewport_height})",
                    flush=True,
                )
            await page.mouse.move(clamped_x, clamped_y)
            if verbose:
                print("[BrowserManager.move_mouse_to] Success", flush=True)
            return True
        except Exception as exc:
            print(f"[BrowserManager.move_mouse_to] Error: {exc}", file=sys.stderr)
            return False

    async def move_mouse_to_relative(
        self,
        target_id: str,
        x_ratio: float,
        y_ratio: float,
        viewport: Viewport | None = None,
        verbose: bool = False,
    ) -> bool:
        """Move the mouse to a spot given as a fraction of the window.

        Args:
            target_id: The tab to move the mouse in.
            x_ratio: How far across, 0 is the left edge and 1 is the right.
            y_ratio: How far down, 0 is the top and 1 is the bottom.
            viewport: The window's size and place on the page, or None to read it.
            verbose: Print progress lines when True.

        Returns:
            True if the mouse moved. False if a ratio is outside 0 to 1.
        """
        if not (0 <= x_ratio <= 1) or not (0 <= y_ratio <= 1):
            print(f"[BrowserManager.move_mouse_to_relative] Error: ratios must be 0-1, got ({x_ratio}, {y_ratio})", file=sys.stderr)
            return False
        try:
            resolved_viewport = await self._resolve_relative_viewport(target_id, viewport, "move_mouse_to_relative")
            if resolved_viewport is None:
                return False
            x = x_ratio * resolved_viewport.usable_width
            y = y_ratio * resolved_viewport.usable_height
            return await self.move_mouse_to(target_id, x, y, verbose=verbose)
        except Exception as exc:
            print(f"[BrowserManager.move_mouse_to_relative] Error: {exc}", file=sys.stderr)
            return False

    async def click_at(self, target_id: str, x: float, y: float, verbose: bool = False) -> bool:
        """Click a spot on the page.

        The spot is measured on the whole page, not on the window. The page
        scrolls first if the spot is not showing.

        Args:
            target_id: The tab to click in.
            x: How far right, counted from the left edge of the page.
            y: How far down, counted from the top of the page.
            verbose: Print progress lines when True.

        Returns:
            True if the click was made.
        """
        page = self._lookup_page_or_log(target_id, "click_at")
        if page is None:
            return False
        if x < 0 or y < 0:
            print(f"[BrowserManager.click_at] Error: negative coordinates ({x}, {y})", file=sys.stderr)
            return False
        if verbose:
            print(f"[BrowserManager.click_at] Clicking document ({x}, {y})", flush=True)
        if not await self.scroll_to(target_id, x, y, verbose=verbose):
            print("[BrowserManager.click_at] Error: scroll failed", file=sys.stderr)
        try:
            viewport = await self.get_page_viewport(target_id)
            if viewport is None:
                print("[BrowserManager.click_at] Error: could not read the viewport", file=sys.stderr)
                return False
            viewport_x = x - viewport.page_x
            viewport_y = y - viewport.page_y
            if verbose:
                print(f"[BrowserManager.click_at] Clicking viewport ({viewport_x}, {viewport_y})", flush=True)
            await page.mouse.click(viewport_x, viewport_y)
            if verbose:
                print("[BrowserManager.click_at] Success", flush=True)
            return True
        except Exception as exc:
            print(f"[BrowserManager.click_at] Error: {exc}", file=sys.stderr)
            return False

    async def click_at_relative(
        self,
        target_id: str,
        x: float,
        y: float,
        viewport: Viewport | None = None,
        verbose: bool = False,
    ) -> bool:
        """Click a spot given as a fraction of the window.

        The spot is where the caller last saw it, counted on the whole page.

        Args:
            target_id: The tab to click in.
            x: How far across, 0 is the left edge and 1 is the right.
            y: How far down, 0 is the top and 1 is the bottom.
            viewport: The window the caller saw, or None to read it.
            verbose: Print progress lines when True.

        Returns:
            True if the click was made. False if the window could not be read.
        """
        resolved_viewport = await self._resolve_relative_viewport(target_id, viewport, "click_at_relative")
        if resolved_viewport is None:
            return False
        point_x = resolved_viewport.page_x + x * resolved_viewport.usable_width
        point_y = resolved_viewport.page_y + y * resolved_viewport.usable_height
        return await self.click_at(target_id, point_x, point_y, verbose=verbose)

    async def type_text_at(
        self,
        target_id: str,
        x: float,
        y: float,
        text: str,
        submit: bool = False,
        verbose: bool = False,
    ) -> bool:
        """Type text at a spot on the page.

        The spot is clicked first to give it focus. The text is added to
        what is there, nothing is cleared.

        Args:
            target_id: The tab to type in.
            x: How far right, counted from the left edge of the page.
            y: How far down, counted from the top of the page.
            text: What to type.
            submit: Press Enter after typing when True.
            verbose: Print progress lines when True.

        Returns:
            True if the text was typed.
        """
        page = self._lookup_page_or_log(target_id, "type_text_at")
        if page is None:
            return False
        if not await self.click_at(target_id, x, y, verbose=verbose):
            print("[BrowserManager.type_text_at] Error: click failed", file=sys.stderr)
            return False
        try:
            if verbose:
                preview = _truncate_for_log(text)
                print(f"[BrowserManager.type_text_at] Typing text: {preview}", flush=True)
            await page.keyboard.type(text)
            if submit:
                if verbose:
                    print("[BrowserManager.type_text_at] Pressing Enter key", flush=True)
                await page.keyboard.press("Enter")
            if verbose:
                print("[BrowserManager.type_text_at] Success", flush=True)
            return True
        except Exception as exc:
            print(f"[BrowserManager.type_text_at] Error: {exc}", file=sys.stderr)
            return False

    async def type_text_at_relative(
        self,
        target_id: str,
        x: float,
        y: float,
        text: str,
        submit: bool = False,
        viewport: Viewport | None = None,
        verbose: bool = False,
    ) -> bool:
        """Type text at a spot given as a fraction of the window.

        The spot is where the caller last saw it, counted on the whole page.

        Args:
            target_id: The tab to type in.
            x: How far across, 0 is the left edge and 1 is the right.
            y: How far down, 0 is the top and 1 is the bottom.
            text: What to type.
            submit: Press Enter after typing when True.
            viewport: The window the caller saw, or None to read it.
            verbose: Print progress lines when True.

        Returns:
            True if the text was typed. False if the window could not be read.
        """
        resolved_viewport = await self._resolve_relative_viewport(target_id, viewport, "type_text_at_relative")
        if resolved_viewport is None:
            return False
        point_x = resolved_viewport.page_x + x * resolved_viewport.usable_width
        point_y = resolved_viewport.page_y + y * resolved_viewport.usable_height
        return await self.type_text_at(target_id, point_x, point_y, text, submit=submit, verbose=verbose)

    async def replace_input_text_at(
        self,
        target_id: str,
        x: float,
        y: float,
        old_text: str,
        new_text: str,
        verbose: bool = False,
    ) -> bool:
        """Replace a piece of the text in the field at a spot on the page.

        The spot is clicked first to give the field focus. The old text must
        show up in the field exactly once.

        Args:
            target_id: The tab to edit in.
            x: How far right, counted from the left edge of the page.
            y: How far down, counted from the top of the page.
            old_text: The text in the field to replace.
            new_text: The text to put in its place. Empty removes the old text.
            verbose: Print progress lines when True.

        Returns:
            True if the field now holds the new text in place of the old.
        """
        page = self._lookup_page_or_log(target_id, "replace_input_text_at")
        if page is None:
            return False
        if not old_text:
            print("[BrowserManager.replace_input_text_at] Error: old text is empty", file=sys.stderr)
            return False
        if not await self.click_at(target_id, x, y, verbose=verbose):
            print("[BrowserManager.replace_input_text_at] Error: click failed", file=sys.stderr)
            return False
        return await replace_focused_field_text(page, old_text, new_text, verbose=verbose)

    async def replace_input_text_at_relative(
        self,
        target_id: str,
        x: float,
        y: float,
        old_text: str,
        new_text: str,
        viewport: Viewport | None = None,
        verbose: bool = False,
    ) -> bool:
        """Replace a piece of the text in the field at a spot given as a fraction of the window.

        The spot is where the caller last saw it, counted on the whole page.

        Args:
            target_id: The tab to edit in.
            x: How far across, 0 is the left edge and 1 is the right.
            y: How far down, 0 is the top and 1 is the bottom.
            old_text: The text in the field to replace.
            new_text: The text to put in its place. Empty removes the old text.
            viewport: The window the caller saw, or None to read it.
            verbose: Print progress lines when True.

        Returns:
            True if the text was replaced. False if the window could not be read.
        """
        resolved_viewport = await self._resolve_relative_viewport(
            target_id,
            viewport,
            "replace_input_text_at_relative",
        )
        if resolved_viewport is None:
            return False
        point_x = resolved_viewport.page_x + x * resolved_viewport.usable_width
        point_y = resolved_viewport.page_y + y * resolved_viewport.usable_height
        return await self.replace_input_text_at(target_id, point_x, point_y, old_text, new_text, verbose=verbose)

    async def solve_captcha(
        self,
        target_id: str,
        max_steps: int = _DEFAULT_CAPTCHA_MAX_STEPS,
        verbose: bool = False,
    ) -> bool:
        """Find and solve a captcha on a page.

        Args:
            target_id: The tab with the captcha.
            max_steps: How many tries to make.
            verbose: Print progress lines when True.

        Returns:
            True if the page has no captcha left.
        """
        return await _solve_captcha(self, target_id, max_steps, verbose)

    async def solve_recaptcha_grid(
        self,
        target_id: str,
        click_timeout: int,
        retry_times: int,
        verbose: bool = False,
    ) -> str | None:
        """Solve the picture challenge of a reCAPTCHA box.

        The checkbox is assumed to have already been clicked.

        Args:
            target_id: The tab with the reCAPTCHA.
            click_timeout: How long to wait after each tile click, in milliseconds.
            retry_times: How many times to try the whole challenge.
            verbose: Print progress lines when True.

        Returns:
            The captcha token, or None if it was not solved.
        """
        page = self._lookup_page_or_log(target_id, "solve_recaptcha_grid")
        if page is None:
            return None
        challenger = AsyncChallenger(page, click_timeout=click_timeout, retry_times=retry_times, verbose=verbose)
        try:
            return await challenger.solve_recaptcha(skip_checkbox=True)
        except Exception as exc:
            print(f"[BrowserManager.solve_recaptcha_grid] Error: {exc}", file=sys.stderr)
            return None

    async def navigate(
        self,
        target_id: str,
        url: str,
        wait_until: str = "domcontentloaded",
        timeout: int = _DEFAULT_TIMEOUT_MS,
        verbose: bool = False,
    ) -> bool:
        """Send a tab to a web address.

        Args:
            target_id: The tab to send.
            url: The address to open.
            wait_until: How far the page must load before this returns.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.

        Returns:
            True if the page opened with a 2xx status or no response. False otherwise.

        Raises:
            DownloadTriggered: If the address starts a file download.
        """
        page = self._get_page(target_id)
        if verbose:
            preview = _truncate_for_log(url)
            print(f"[BrowserManager.navigate] Navigating to {preview} (wait_until={wait_until})", flush=True)
        try:
            response = await page.goto(url, wait_until=wait_until, timeout=timeout)
            if response is not None and not (_HTTP_OK_MIN <= response.status < _HTTP_OK_MAX):
                if verbose:
                    print(f"[BrowserManager.navigate] Non-2xx status: {response.status}", flush=True)
                return False
            if verbose:
                print("[BrowserManager.navigate] Success", flush=True)
            return True
        except Exception as exc:
            if "Download is starting" in str(exc):
                if verbose:
                    print("[BrowserManager.navigate] Download triggered", flush=True)
                raise DownloadTriggered(f"Navigation to {url} triggered a download")
            if verbose:
                print(f"[BrowserManager.navigate] Error: {exc}", flush=True)
            return False

    async def wait_for_load(
        self,
        target_id: str,
        state: str = "domcontentloaded",
        timeout: int = _DEFAULT_TIMEOUT_MS,
        verbose: bool = False,
    ) -> bool:
        """Wait for a page to finish loading.

        Args:
            target_id: The tab to wait on.
            state: How far the page must load.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.

        Returns:
            True if the page got that far in time.
        """
        page = self._lookup_page_or_log(target_id, "wait_for_load")
        if page is None:
            return False
        if verbose:
            print(f"[BrowserManager.wait_for_load] Waiting for state: {state}", flush=True)
        try:
            await page.wait_for_load_state(state, timeout=timeout)
            if verbose:
                print("[BrowserManager.wait_for_load] Success", flush=True)
            return True
        except Exception as exc:
            if verbose:
                print(f"[BrowserManager.wait_for_load] Error: {exc}", file=sys.stderr)
            return False

    async def go_back(
        self,
        target_id: str,
        wait_until: str = "load",
        timeout: int = _DEFAULT_TIMEOUT_MS,
        verbose: bool = False,
    ) -> bool:
        """Go back to the page a tab was on before.

        Args:
            target_id: The tab to send back.
            wait_until: How far the page must load before this returns.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.

        Returns:
            True if the older page opened.
        """
        page = self._lookup_page_or_log(target_id, "go_back")
        if page is None:
            return False
        try:
            await page.go_back(wait_until=wait_until, timeout=timeout)
            if verbose:
                print("[BrowserManager.go_back] Success", flush=True)
            return True
        except Exception as exc:
            print(f"[BrowserManager.go_back] Error: {exc}", file=sys.stderr)
            return False

    async def go_forward(
        self,
        target_id: str,
        wait_until: str = "load",
        timeout: int = _DEFAULT_TIMEOUT_MS,
        verbose: bool = False,
    ) -> bool:
        """Go forward to the page a tab had been on.

        Args:
            target_id: The tab to send forward.
            wait_until: How far the page must load before this returns.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.

        Returns:
            True if the newer page opened.
        """
        page = self._lookup_page_or_log(target_id, "go_forward")
        if page is None:
            return False
        try:
            await page.go_forward(wait_until=wait_until, timeout=timeout)
            if verbose:
                print("[BrowserManager.go_forward] Success", flush=True)
            return True
        except Exception as exc:
            print(f"[BrowserManager.go_forward] Error: {exc}", file=sys.stderr)
            return False

    async def wait_for_selector(
        self,
        target_id: str,
        selector: str,
        timeout: int = _DEFAULT_TIMEOUT_MS,
        verbose: bool = False,
    ) -> bool:
        """Wait for a part of the page to show up.

        Args:
            target_id: The tab to watch.
            selector: Points at the part to wait for.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.

        Returns:
            True if the part showed up in time.
        """
        page = self._lookup_page_or_log(target_id, "wait_for_selector")
        if page is None:
            return False
        if verbose:
            print(f"[BrowserManager.wait_for_selector] Waiting for: {selector}", flush=True)
        try:
            await page.wait_for_selector(selector, timeout=timeout)
            if verbose:
                print("[BrowserManager.wait_for_selector] Found", flush=True)
            return True
        except Exception as exc:
            print(f"[BrowserManager.wait_for_selector] Error: {exc}", file=sys.stderr)
            return False

    async def wait_for_selector_disappear(
        self,
        target_id: str,
        selector: str,
        timeout: int = _DEFAULT_TIMEOUT_MS,
        verbose: bool = False,
    ) -> bool:
        """Wait for a part of the page to go away.

        Args:
            target_id: The tab to watch.
            selector: Points at the part to wait on.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.

        Returns:
            True if the part went away in time.
        """
        page = self._lookup_page_or_log(target_id, "wait_for_selector_disappear")
        if page is None:
            return False
        if verbose:
            print(f"[BrowserManager.wait_for_selector_disappear] Waiting for: {selector}", flush=True)
        try:
            await page.wait_for_selector(selector, state="detached", timeout=timeout)
            if verbose:
                print("[BrowserManager.wait_for_selector_disappear] Disappeared", flush=True)
            return True
        except Exception as exc:
            print(f"[BrowserManager.wait_for_selector_disappear] Error: {exc}", file=sys.stderr)
            return False

    async def _sync_aria_hidden_attributes(self, page: Page) -> None:
        """Mark page elements that are visually hidden with aria-hidden."""
        try:
            await _await_page_call(page.evaluate(_SYNC_ARIA_HIDDEN_JS), "aria-hidden update")
        except Exception:
            pass

    async def _capture_page_data(self, page: Page, target_id: str, verbose: bool) -> PageSnapshot:
        """Read HTML, URL, title, and viewport to build a PageSnapshot."""
        html = await _await_page_call(page.content(), "page content")
        current_url = page.url
        try:
            page_title = await _await_page_call(page.title(), "page title")
        except Exception:
            page_title = ""
        viewport = await self.get_page_viewport(target_id)
        if verbose:
            print(f"[BrowserManager.capture_snapshot] HTML length: {len(html)}, URL: {current_url}", flush=True)
        return await PageSnapshot.create(html, self, target_id, url=current_url, title=page_title, viewport=viewport)

    async def capture_snapshot(
        self,
        target_id: str,
        timeout: int = _DEFAULT_TIMEOUT_MS,
        verbose: bool = False,
    ) -> PageSnapshot | None:
        """Take a frozen picture of a page that you can read and act on.

        Args:
            target_id: The tab to read.
            timeout: How long to wait for the page to settle, in milliseconds.
            verbose: Print progress lines when True.

        Returns:
            The snapshot, or None if the page could not be read.
        """
        page = self._lookup_page_or_log(target_id, "capture_snapshot")
        if page is None:
            return None
        if verbose:
            print("[BrowserManager.capture_snapshot] Capturing snapshot...", flush=True)
        try:
            if verbose:
                print("[BrowserManager.capture_snapshot] Waiting for networkidle...", flush=True)
            try:
                await page.wait_for_load_state("networkidle", timeout=timeout)
                if verbose:
                    print("[BrowserManager.capture_snapshot] networkidle reached", flush=True)
            except Exception as wait_error:
                if verbose:
                    print(f"[BrowserManager.capture_snapshot] Warning: networkidle wait failed: {wait_error}", file=sys.stderr)
            await self._sync_aria_hidden_attributes(page)
            return await self._capture_page_data(page, target_id, verbose)
        except Exception as exc:
            print(f"[BrowserManager.capture_snapshot] Error: {exc}", file=sys.stderr)
            return None

    async def _detach_cdp_session(self, page: Page, session: CDPSession) -> None:
        """Detach a tab session with a bounded timeout, skipping closed tabs."""
        if page.is_closed():
            return
        try:
            await asyncio.wait_for(session.detach(), timeout=_CLEANUP_STEP_TIMEOUT_S)
        except Exception as exc:
            print(f"[BrowserManager._detach_cdp_session] Error: {exc}", file=sys.stderr)

    def _build_download_action(
        self,
        target_id: str,
        url: str | None,
        action: Callable[[], Awaitable[bool]] | None,
        timeout: int,
        verbose: bool,
    ) -> Callable[[], Awaitable[bool]]:
        """Build the async step that triggers a download."""
        if url is not None:
            async def _navigate_for_download() -> bool:
                try:
                    is_navigated = await self.navigate(
                        target_id,
                        url,
                        wait_until="domcontentloaded",
                        timeout=timeout * _MS_PER_SECOND,
                        verbose=verbose,
                    )
                    if not is_navigated:
                        raise RuntimeError(f"Navigation failed for {url}")
                    return False
                except DownloadTriggered:
                    return True
            return _navigate_for_download
        if action is not None:
            return action
        raise ValueError("Either url or action must be provided")

    async def wait_for_download(
        self,
        target_id: str,
        url: str | None = None,
        action: Callable[[], Awaitable[bool]] | None = None,
        save_path: str | None = None,
        timeout: int = _DEFAULT_DOWNLOAD_TIMEOUT_S,
        verbose: bool = False,
    ) -> DownloadResult:
        """Do something that may download a file, and pick the file up.

        Give exactly one of url or action.

        Args:
            target_id: The tab to work in.
            url: Address to open.
            action: Something to do instead of opening an address. It says
                True when a download started.
            save_path: Where to move the file, or None to leave it be.
            timeout: How many seconds to wait for the file.
            verbose: Print progress lines when True.

        Returns:
            Whether a file came down, and if it did, its path, name, and
            size.

        Raises:
            KeyError: If there is no such tab.
            ValueError: If neither url nor action is given.
        """
        page = self._get_page(target_id)
        if verbose:
            action_type = "url" if url else "action" if action else "none"
            print(f"[BrowserManager.wait_for_download] Starting ({action_type})", flush=True)
        trigger_action = self._build_download_action(target_id, url, action, timeout, verbose)
        session = self._get_page_session(page)
        return await wait_for_page_download(
            session,
            self._download_dir,
            trigger_action,
            save_path=save_path,
            timeout=timeout,
            verbose=verbose,
        )

    def get_url(self, target_id: str, verbose: bool = False) -> str | None:
        """Get the address a tab is showing.

        Args:
            target_id: The tab to ask about.
            verbose: Print progress lines when True.

        Returns:
            The address, or None if it could not be read.
        """
        page = self._lookup_page_or_log(target_id, "get_url")
        if page is None:
            return None
        if verbose:
            print(f"[BrowserManager.get_url] Getting URL for {target_id}", flush=True)
        try:
            page_url = page.url
            if verbose:
                print(f"[BrowserManager.get_url] URL: {page_url}", flush=True)
            return page_url
        except Exception as exc:
            print(f"[BrowserManager.get_url] Error: {exc}", file=sys.stderr)
            return None

    async def get_title(self, target_id: str, verbose: bool = False) -> str | None:
        """Get the title of the page a tab is showing.

        Args:
            target_id: The tab to ask about.
            verbose: Print progress lines when True.

        Returns:
            The title, or None if it could not be read.
        """
        page = self._lookup_page_or_log(target_id, "get_title")
        if page is None:
            return None
        if verbose:
            print(f"[BrowserManager.get_title] Getting title for {target_id}", flush=True)
        try:
            page_title = await page.title()
            if verbose:
                print(f"[BrowserManager.get_title] Title: {page_title}", flush=True)
            return page_title
        except Exception as exc:
            print(f"[BrowserManager.get_title] Error: {exc}", file=sys.stderr)
            return None

    async def capture_screenshot(
        self,
        target_id: str,
        path: str,
        full_page: bool = True,
        timeout: float = _DEFAULT_SCREENSHOT_TIMEOUT_S,
        verbose: bool = False,
    ) -> str | None:
        """Save a picture of a page to a file.

        Args:
            target_id: The tab to shoot.
            path: Where to save the picture.
            full_page: Shoot the whole page when True, just the window when
                False.
            timeout: How long to wait, in seconds, for the shot to finish.
            verbose: Print progress lines when True.

        Returns:
            The path, or None if the picture was not saved.
        """
        page = self._lookup_page_or_log(target_id, "capture_screenshot")
        if page is None:
            return None
        if verbose:
            print(f"[BrowserManager.capture_screenshot] Saving to {path} (full_page={full_page})", flush=True)
        try:
            await page.screenshot(
                path=path,
                type="jpeg",
                full_page=full_page,
                timeout=int(timeout * _MS_PER_SECOND),
            )
            if verbose:
                print(f"[BrowserManager.capture_screenshot] Saved: {path}", flush=True)
            return path if await asyncio.to_thread(os.path.exists, path) else None
        except Exception as exc:
            print(f"[BrowserManager.capture_screenshot] Error: {exc}", file=sys.stderr)
            return None

    async def save_pdf(self, target_id: str, path: str, verbose: bool = False) -> str | None:
        """Save a page as a PDF file.

        Args:
            target_id: The tab to save.
            path: Where to save the PDF.
            verbose: Print progress lines when True.

        Returns:
            The path, or None if the PDF was not saved.
        """
        page = self._lookup_page_or_log(target_id, "save_pdf")
        if page is None:
            return None
        session = self._get_page_session(page)
        if session is None:
            print(f"[BrowserManager.save_pdf] Error: no session for {target_id}", file=sys.stderr)
            return None
        if verbose:
            print(f"[BrowserManager.save_pdf] Saving PDF to {path}", flush=True)
        try:
            result = await session.send("Page.printToPDF", {"printBackground": True})
            pdf_bytes = base64.b64decode(result["data"])
            await asyncio.to_thread(os.makedirs, os.path.dirname(path) or ".", exist_ok=True)
            async with aiofiles.open(path, "wb") as pdf_file:
                await pdf_file.write(pdf_bytes)
            if verbose:
                print(f"[BrowserManager.save_pdf] Saved: {path}", flush=True)
            return path if await asyncio.to_thread(os.path.exists, path) else None
        except Exception as exc:
            print(f"[BrowserManager.save_pdf] Error: {exc}", file=sys.stderr)
            return None

    async def type_text(
        self,
        target_id: str,
        selector: str,
        text: str,
        submit: bool = False,
        timeout: int = _DEFAULT_TIMEOUT_MS,
        verbose: bool = False,
        human_config: dict | None = None,
    ) -> bool:
        """Type text into a part of the page.

        The page scrolls first. Nothing is typed if the part cannot be
        brought into the window.

        Args:
            target_id: The tab to type in.
            selector: Points at the part to type into.
            text: What to type.
            submit: Press Enter after typing when True.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.
            human_config: Settings for how human the typing should look.

        Returns:
            True if the text was typed.
        """
        page = self._lookup_page_or_log(target_id, "type_text")
        if page is None:
            return False
        if not await self.scroll_to_view(target_id, selector, verbose=verbose):
            print("[BrowserManager.type_text] Error: scroll failed", file=sys.stderr)
            return False
        if verbose:
            preview = _truncate_for_log(text)
            print(f"[BrowserManager.type_text] Typing into '{selector}': {preview} (submit={submit})", flush=True)
        try:
            if human_config is None:
                await page.fill(selector, text, force=True, timeout=timeout)
            else:
                await page.fill(selector, text, force=True, timeout=timeout, human_config=human_config)
            if submit:
                await page.press(selector, "Enter", timeout=timeout)
            if verbose:
                print("[BrowserManager.type_text] Success", flush=True)
            return True
        except Exception as exc:
            print(f"[BrowserManager.type_text] Error: {exc}", file=sys.stderr)
            return False

    async def select_option(
        self,
        target_id: str,
        selector: str,
        value: str | None = None,
        index: int | None = None,
        label: str | None = None,
        verbose: bool = False,
    ) -> bool:
        """Pick a choice in a drop down list.

        The page scrolls first. Nothing is picked if the list cannot be
        brought into the window. Give only one of value, index, or label.

        Args:
            target_id: The tab to work in.
            selector: Points at the drop down list.
            value: The value of the choice to pick.
            index: The place of the choice in the list.
            label: The words shown for the choice.
            verbose: Print progress lines when True.

        Returns:
            True if the choice was picked.
        """
        page = self._lookup_page_or_log(target_id, "select_option")
        if page is None:
            return False
        if not await self.scroll_to_view(target_id, selector, verbose=verbose):
            print("[BrowserManager.select_option] Error: scroll failed", file=sys.stderr)
            return False
        try:
            selection_kwargs: dict[str, str | int] = {}
            if value is not None:
                selection_kwargs["value"] = value
            elif index is not None:
                selection_kwargs["index"] = index
            elif label is not None:
                selection_kwargs["label"] = label
            else:
                print("[BrowserManager.select_option] Error: must specify value, index, or label", file=sys.stderr)
                return False
            locator = page.locator(selector).first
            await locator.select_option(force=True, **selection_kwargs)
            if verbose:
                print(f"[BrowserManager.select_option] Success: {selector} -> {selection_kwargs}", flush=True)
            return True
        except Exception as exc:
            print(f"[BrowserManager.select_option] Error: {exc}", file=sys.stderr)
            return False

    async def press_key(self, target_id: str, key: str = "Enter", verbose: bool = False) -> bool:
        """Press one key on the keyboard.

        Args:
            target_id: The tab to type in.
            key: The name of the key to press.
            verbose: Print progress lines when True.

        Returns:
            True if the key was pressed.
        """
        page = self._lookup_page_or_log(target_id, "press_key")
        if page is None:
            return False
        try:
            await page.keyboard.press(key)
            if verbose:
                print(f"[BrowserManager.press_key] Success: {key}", flush=True)
            return True
        except Exception as exc:
            print(f"[BrowserManager.press_key] Error: {exc}", file=sys.stderr)
            return False

    async def upload_with_chooser_relative(
        self,
        target_id: str,
        x: float,
        y: float,
        file_paths: list[str],
        viewport: Viewport | None = None,
        verbose: bool = False,
    ) -> bool:
        """Give files to a picker opened at a spot on the screen.

        Args:
            target_id: The tab to work in.
            x: How far across the screen, from zero to one.
            y: How far down the screen, from zero to one.
            file_paths: Paths of the files to send.
            viewport: The window the caller saw, or None to read it.
            verbose: Print progress lines when True.

        Returns:
            True if the files were sent.
        """
        return await self._upload_with_chooser(
            target_id,
            file_paths,
            lambda: self.click_at_relative(target_id, x, y, viewport=viewport, verbose=verbose),
            verbose=verbose,
        )

    async def _upload_with_chooser(
        self,
        target_id: str,
        file_paths: list[str],
        trigger: Callable[[], Awaitable[bool]],
        verbose: bool = False,
    ) -> bool:
        """Open a file picker and provide the selected files."""
        page = self._lookup_page_or_log(target_id, "_upload_with_chooser")
        if page is None:
            return False
        if verbose:
            print(f"[BrowserManager._upload_with_chooser] Uploading {len(file_paths)} file(s) via chooser", flush=True)
        try:
            async with page.expect_file_chooser(timeout=_FILE_CHOOSER_TIMEOUT_MS) as chooser_info:
                if not await trigger():
                    raise RuntimeError("could not click the upload control")
            file_chooser = await chooser_info.value
            await file_chooser.set_files(file_paths)
            if verbose:
                print("[BrowserManager._upload_with_chooser] Success", flush=True)
            return True
        except Exception as exc:
            print(f"[BrowserManager._upload_with_chooser] Error: {exc}", file=sys.stderr)
            return False

    async def upload_files(self, target_id: str, selector: str, file_paths: list[str], verbose: bool = False) -> bool:
        """Give files to a file box on the page.

        Args:
            target_id: The tab to work in.
            selector: Points at the file box.
            file_paths: Paths of the files to send.
            verbose: Print progress lines when True.

        Returns:
            True if the files were sent.
        """
        page = self._lookup_page_or_log(target_id, "upload_files")
        if page is None:
            return False
        if verbose:
            print(f"[BrowserManager.upload_files] Uploading {len(file_paths)} file(s) to '{selector}'", flush=True)
        try:
            await page.set_input_files(selector, file_paths)
            if verbose:
                print("[BrowserManager.upload_files] Success", flush=True)
            return True
        except Exception as exc:
            print(f"[BrowserManager.upload_files] Error: {exc}", file=sys.stderr)
            return False

    async def cleanup(self, verbose: bool = False) -> None:
        """Close every tab and let go of the browser.

        This never hangs. A step that fails or takes too long is skipped.
        Calling this again does nothing.

        Args:
            verbose: Print progress lines when True.
        """
        if self._is_cleaned_up:
            return
        self._is_cleaned_up = True
        if verbose:
            print(f"[BrowserManager.cleanup] Closing {len(self._pages)} pages...", flush=True)
        for target_id in list(self._pages.keys()):
            await self.close_page(target_id, verbose=verbose)
        await self._close_browser_resources(verbose)
        _CONNECTED_MANAGERS.discard(self)

    async def _close_browser_resources(self, verbose: bool) -> None:
        """Close the browser handle and stop Playwright with bounded timeouts."""
        if self._browser is not None:
            try:
                await asyncio.wait_for(self._browser.close(), timeout=_CLEANUP_STEP_TIMEOUT_S)
                if verbose:
                    print("[BrowserManager.cleanup] Browser closed", flush=True)
            except Exception as exc:
                print(f"[BrowserManager.cleanup] Error closing browser: {exc}", file=sys.stderr)
        if self._playwright is not None:
            try:
                await asyncio.wait_for(self._playwright.stop(), timeout=_CLEANUP_STEP_TIMEOUT_S)
                if verbose:
                    print("[BrowserManager.cleanup] Playwright stopped", flush=True)
            except Exception as exc:
                print(f"[BrowserManager.cleanup] Error stopping Playwright: {exc}", file=sys.stderr)

    def __del__(self) -> None:
        """Warn when a manager is dropped without being cleaned up.

        The browser is not let go of here.
        """
        if not self._is_cleaned_up and self._playwright is not None:
            print(
                "[BrowserManager] Leaked browser connection: use "
                "'async with BrowserManager(...)' or await cleanup()",
                file=sys.stderr,
            )

    async def __aenter__(self) -> BrowserManager:
        await self._connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        await self.cleanup()
        return False


async def _clamp_mouse_params(manager: BrowserManager, target_id: str, params: dict) -> None:
    """Pull mouse coordinates back inside the window bounds."""
    viewport = await manager._get_clamp_viewport(target_id)
    if viewport is None:
        return
    width = viewport.usable_width
    height = viewport.usable_height
    if "x" in params and width > 0:
        params["x"] = max(0, min(params["x"], width - 1))
    if "y" in params and height > 0:
        params["y"] = max(0, min(params["y"], height - 1))


async def _await_page_call[T](call: Awaitable[T], name: str, timeout: float = _PAGE_CALL_TIMEOUT_S) -> T:
    """Wait for a page call with a hard deadline."""
    deadline = asyncio.timeout(timeout)
    try:
        async with deadline:
            return await call
    except TimeoutError:
        if not deadline.expired():
            raise
        message = f"{name} timed out after {timeout}s"
        print(f"[browser_manager._await_page_call] Error: {message}", file=sys.stderr)
        raise TimeoutError(message) from None
