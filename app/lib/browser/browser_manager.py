from __future__ import annotations

"""Drive a Chrome browser that is already running.

Opens tabs, acts on pages, and picks up files the page downloads.
Always clean up a manager when you are done with it.

Example:
    async with BrowserManager() as mgr:
        target_id = await mgr.new_page()
        result = await mgr.wait_for_download(target_id, url="https://example.com/file.pdf")
        await mgr.close_page(target_id)

    mgr = await BrowserManager.create()
    try:
        ...
    finally:
        await mgr.cleanup()
"""
__all__ = ["BrowserManager", "Viewport"]

import asyncio
import base64
import os
import random
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, NotRequired, Optional, Protocol, TYPE_CHECKING, TypedDict, Tuple

import aiofiles
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, CDPSession, Frame, Playwright
# lint-skip: coding.import-private-module
# Approved by user. This is a hack to resolve a known issue with Playwright's CDP commands.
from playwright._impl._connection import Channel
from cloakbrowser.human import patch_context_async
from cloakbrowser.human.config import resolve_config

from app.lib.browser.page_snapshot import PageSnapshot
from app.lib.browser.captcha_solver import solve_captcha as _solve_captcha
from app.lib.recognizer.agents.playwright.async_control import AsyncChallenger
from app.lib.shared import paths

os.environ.setdefault("NODE_NO_WARNINGS", "1")

_DEFAULT_CDP_URL = "http://localhost:9222"
_DEFAULT_TIMEOUT_MS = 5000
_SCROLL_MAX_ITERATIONS = 30
_SCROLL_STEP_MAX = 300
_SCROLL_STAGNANT_LIMIT = 5
_SCROLL_MARGIN = 10
# How long one cleanup step may take before it is given up on.
_CLEANUP_STEP_TIMEOUT_S = 10

# The one expression that reads the window box, with its scrollbars.
_WINDOW_SIZE_JS = "() => [window.innerWidth, window.innerHeight]"

# Helpers that find the focused text field and read its text.
_FOCUSED_FIELD_JS = """
    const SELECTABLE_INPUT_TYPES = ['text', 'search', 'url', 'tel', 'password'];
    const WHOLE_VALUE_INPUT_TYPES = ['email', 'number'];
    const focusedElement = () => {
        let el = document.activeElement;
        while (el && el.shadowRoot && el.shadowRoot.activeElement) el = el.shadowRoot.activeElement;
        return el;
    };
    const focusedField = () => {
        const el = focusedElement();
        if (!el) return null;
        if (el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && SELECTABLE_INPUT_TYPES.includes(el.type))) {
            return {hasValue: true, selectsRange: true, el};
        }
        if (el.tagName === 'INPUT' && WHOLE_VALUE_INPUT_TYPES.includes(el.type)) {
            return {hasValue: true, selectsRange: false, el};
        }
        if (!el.isContentEditable) return null;
        let host = el;
        while (host.parentElement && host.parentElement.isContentEditable) host = host.parentElement;
        return {hasValue: false, el: host};
    };
    const textNodesOf = (host) => {
        const walker = document.createTreeWalker(host, NodeFilter.SHOW_TEXT);
        const nodes = [];
        while (walker.nextNode()) nodes.push(walker.currentNode);
        return nodes;
    };
    const fieldText = (field) => field.hasValue
        ? field.el.value
        : textNodesOf(field.el).map((node) => node.data).join('').replace(/\\u00a0/g, ' ');
"""

# Gives back the focused inner frame element, or null when focus is not in one.
_FOCUSED_INNER_FRAME_JS = """() => {
    const el = document.activeElement;
    return el && (el.tagName === 'IFRAME' || el.tagName === 'FRAME') ? el : null;
}"""

# Selects the one copy of the given text in the focused field, or all of its
# text when only all can be selected. Gives back a status, the field text with
# that piece swapped for the new text, and whether all of the text was selected.
_SELECT_FIELD_TEXT_JS = """([oldText, newText]) => {
""" + _FOCUSED_FIELD_JS + """
    const field = focusedField();
    if (!field) return {status: 'not an editable field'};
    const text = fieldText(field);
    const start = text.indexOf(oldText);
    if (start < 0) return {status: 'old text not found'};
    if (text.indexOf(oldText, start + 1) >= 0) return {status: 'old text found more than once'};
    const end = start + oldText.length;
    const expected = text.slice(0, start) + newText + text.slice(end);
    if (field.hasValue && !field.selectsRange) {
        field.el.select();
        return {status: 'ok', expected, selectedAll: true};
    }
    if (field.hasValue) {
        field.el.setSelectionRange(start, end);
    } else {
        const range = document.createRange();
        let offset = 0;
        for (const node of textNodesOf(field.el)) {
            const nodeEnd = offset + node.data.length;
            if (start >= offset && start < nodeEnd) range.setStart(node, start - offset);
            if (end > offset && end <= nodeEnd) range.setEnd(node, end - offset);
            offset = nodeEnd;
        }
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
    }
    return {status: 'ok', expected, selectedAll: false};
}"""

# Reads the text of the focused field, or null when no text field has focus.
_READ_FIELD_TEXT_JS = """() => {
""" + _FOCUSED_FIELD_JS + """
    const field = focusedField();
    return field ? fieldText(field) : null;
}"""

# The connected managers, private plumbing for the mouse-clamp hook.
_MANAGERS: set[BrowserManager] = set()


@dataclass(frozen=True)
class Viewport:
    """The size of the window and its place on the page, in pixels.

    The place is where the top left corner of the window sits, counted
    from the top left corner of the page.

    There are two size families. The client size is the area where page
    content renders. The window size is the whole window box, including any
    scrollbar, the ratios scale against it, and mouse input is kept inside
    of it. The content size is the document's full size, the furthest the
    page can scroll plus the client size.
    """

    client_width: float  # the visible content area, CSS pixels, excluding any scrollbar (CDP cssVisualViewport.clientWidth)
    client_height: float  # same, vertically (cssVisualViewport.clientHeight)
    page_x: float  # where the window's top-left corner sits on the document, horizontally (cssVisualViewport.pageX, which equals window.scrollX)
    page_y: float  # same, vertically (cssVisualViewport.pageY, which equals window.scrollY)
    content_width: float  # the document's full width, the furthest the page can scroll plus the client width (CDP cssContentSize.width)
    content_height: float  # same, vertically (cssContentSize.height)
    window_width: float = 0.0  # the whole window size including any scrollbar (window.innerWidth), 0.0 when it could not be read
    window_height: float = 0.0  # same, vertically (window.innerHeight), 0.0 when it could not be read

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
        """The window width to use, the client width when the window could not be read."""
        return self.window_width if self.window_width > 0 else self.client_width

    @property
    def usable_height(self) -> float:
        """The window height to use, the client height when the window could not be read."""
        return self.window_height if self.window_height > 0 else self.client_height


@dataclass
class _PageState:
    """The per-page data of a tracked tab."""

    session: Optional[CDPSession]
    guid: str
    clamp_viewport: Optional[Viewport] = None


def _install_mouse_viewport_clamp() -> None:
    """Keep every mouse move and click inside the window.

    Chrome throws away a mouse event that lands on or past the edge of
    the window, so the spot is pulled back inside first. This change
    applies to every page in the whole program, and it is made once.
    """
    # lint-skip: coding.reflection,coding.access-private
    # Approved by user. This is a hack to resolve a known issue with Playwright's CDP commands.
    if getattr(Channel, "_viewport_clamp_installed", False):
        return

    _orig_send = Channel.send

    async def _clamped_send(channel_self: Channel, method: str, timeout_calculator: Optional[Callable] = None, params: Optional[dict] = None, **kwargs: str | bool | None) -> dict:
        if method in ("mouseMove", "mouseClick") and params:
            page_obj = getattr(channel_self, "_object", None)
            if page_obj is not None:
                guid = getattr(page_obj, "_guid", None)
                if guid is not None:
                    for mgr in _MANAGERS:
                        target_id = mgr._tab_guid_to_target_id.get(guid)
                        if target_id is None:
                            continue
                        await _clamp_mouse_params(mgr, target_id, params)
                        break

        return await _orig_send(channel_self, method, timeout_calculator, params, **kwargs)

    Channel.send = _clamped_send
    Channel._viewport_clamp_installed = True
    # lint-end-skip


class _DownloadTriggered(Exception):
    """Raised when going to a page starts a file download instead."""
    pass


class _DownloadEvents:
    """Tracks one download from its tab's CDP session."""

    def __init__(self) -> None:
        self.file_path: Optional[str] = None
        self.completed: asyncio.Event = asyncio.Event()
        self.started_guids: set[str] = set()

    def on_will_begin(self, params: dict) -> None:
        guid = params.get("guid")
        if guid:
            self.started_guids.add(guid)

    def on_progress(self, params: dict) -> None:
        if params.get("state") == "completed" and params.get("guid") in self.started_guids:
            self.file_path = params.get("filePath")
            self.completed.set()


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
        self._pw: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._pages: dict[str, Page] = {}
        self._page_states: dict[Page, _PageState] = {}
        self._tab_guid_to_target_id: dict[str, str] = {}
        self._cleaned_up = False

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
        mgr = cls(cdp_url, download_dir, verbose)
        await mgr._connect()
        return mgr

    async def _connect(self) -> None:
        """Connect to the running Chrome and get it ready.

        Raises:
            RuntimeError: If there is no browser to connect to.
        """
        verbose = self._verbose
        if verbose:
            print(f"[BrowserManager] Connecting to CDP: {self._cdp_url}", flush=True)
            print(f"[BrowserManager] Download dir: {self._download_dir}", flush=True)

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.connect_over_cdp(self._cdp_url, artifacts_dir=self._download_dir)
        if not self._browser.contexts:
            await self._browser.close()
            await self._pw.stop()
            raise RuntimeError(
                f"No browser contexts found at {self._cdp_url}. "
                "Ensure Chrome is running with --remote-debugging-port=9222"
            )
        self._context = self._browser.contexts[0]
        await asyncio.to_thread(os.makedirs, self._download_dir, exist_ok=True)
        self._apply_humanize()
        # Register with the mouse-clamp hook, installed at module load.
        _MANAGERS.add(self)
        self._cleaned_up = False

    def _get_page(self, target_id: str) -> Page:
        """Find a page.

        Args:
            target_id: The tab to find.

        Returns:
            The page in that tab.

        Raises:
            KeyError: If there is no such tab.
        """
        if target_id not in self._pages:
            raise KeyError(f"No page found for target_id: {target_id}")
        return self._pages[target_id]

    def _apply_humanize(self) -> None:
        """Make clicks, typing, and scrolling look like a person did them.

        Nothing changes if this cannot be set up.
        """
        try:
            cfg = resolve_config("default", None)
            patch_context_async(self._context, cfg)
        except Exception as e:
            print(f"⚠️  Humanize init failed: {e}. Continuing without humanize.", flush=True)

    async def _attach_tab_session(self, page: Page) -> Optional[CDPSession]:
        """Open the per-tab CDP session that keeps the tab at foreground speed.

        Args:
            page: The page to attach.

        Returns:
            The session, or None if it could not be attached.
        """
        cdp: Optional[CDPSession] = None
        try:
            cdp = await self._context.new_cdp_session(page)
            await cdp.send("Emulation.setFocusEmulationEnabled", {"enabled": True})
            await cdp.send("Page.setWebLifecycleState", {"state": "active"})
            return cdp
        except Exception as e:
            print(f"[BrowserManager.tab_session] Error: {e}", file=sys.stderr)
            if cdp is not None:
                await self._detach_cdp_session(page, cdp)
            return None

    async def _track_page(self, page: Page) -> str:
        """Give a page an id, a per-tab CDP session, and add it to the owned set.

        Tabs opened by an owned tab are added the same way, so a chain of
        tabs each opened by an owned tab stays owned. The per-tab session
        keeps the tab at foreground speed.

        Args:
            page: The page to own.

        Returns:
            The id minted for the page.
        """
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
        """Drop a closed page from the owned set."""
        state: Optional[_PageState] = self._page_states.pop(page, None)
        for target_id, known_page in list(self._pages.items()):
            if known_page is page:
                del self._pages[target_id]
                break
        if state is not None:
            self._tab_guid_to_target_id.pop(state.guid, None)

    async def _on_popup(self, new_page: Page) -> None:
        """Own a tab this one opened.

        A tab that cannot be adopted is left alone.

        Args:
            new_page: The tab the page opened.
        """
        try:
            await self._track_page(new_page)
        except Exception as e:
            print(f"[BrowserManager.popup] Failed to adopt tab: {e}", file=sys.stderr)

    def _on_page_closed(self, page: Page) -> None:
        """Forget a tab the moment it closes.

        Args:
            page: The tab that closed.
        """
        try:
            self._forget_page(page)
        except Exception as e:
            print(f"[BrowserManager.close] Failed to forget tab: {e}", file=sys.stderr)

    def _on_page_crashed(self, page: Page) -> None:
        """Close a tab whose renderer crashed.

        Args:
            page: The tab whose renderer crashed.
        """
        target_id = next((tid for tid, p in self._pages.items() if p is page), None)
        if target_id is None:
            return
        asyncio.get_running_loop().create_task(self.close_page(target_id))

    async def new_page(self, verbose: bool = False) -> str:
        """Open a new tab.

        Args:
            verbose: Print progress lines when True.

        Returns:
            The id of the new tab.
        """
        page: Page = await self._context.new_page()
        target_id: str = await self._track_page(page)
        if verbose:
            print(f"[BrowserManager.new_page] Created page: {target_id}", flush=True)
        return target_id

    def tab_ids(self) -> list[str]:
        """Give back the ids of the tabs this manager owns.

        Returns:
            The ids of the open tabs.
        """
        return list(self._pages)

    async def close_page(self, target_id: str, verbose: bool = False) -> bool:
        """Close a tab. This never hangs.

        If closing fails or takes too long, the tab is let go of here.

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
        except Exception as e:
            print(f"[BrowserManager.close_page] Error closing {target_id}: {e}", file=sys.stderr)
            state: Optional[_PageState] = self._page_states.pop(page, None)
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
            expr_short = expression[:50] + "..." if len(expression) > 50 else expression
            print(f"[BrowserManager.evaluate] Evaluating: {expr_short}", flush=True)
        try:
            result = await page.evaluate(expression)
            if verbose:
                print(f"[BrowserManager.evaluate] Result type: {type(result)}", flush=True)
            return result
        except Exception as e:
            print(f"[BrowserManager.evaluate] Error: {e}", file=sys.stderr)
            return None

    async def click(self, target_id: str, selector: str, timeout: int = _DEFAULT_TIMEOUT_MS, verbose: bool = False, human_config: Optional[dict] = None) -> bool:
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
        if target_id not in self._pages:
            print(f"[BrowserManager.click] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if not await self.scroll_to_view(target_id, selector, verbose=verbose):
            print(f"[BrowserManager.click] Error: scroll failed", file=sys.stderr)
            return False

        if verbose:
            print(f"[BrowserManager.click] Clicking '{selector}' (timeout={timeout}ms)", flush=True)
        page = self._pages[target_id]
        try:
            if human_config is None:
                await page.click(selector, force=True, timeout=timeout)
            else:
                await page.click(selector, force=True, timeout=timeout, human_config=human_config)
            if verbose:
                print(f"[BrowserManager.click] Success", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.click] Error: {e}", file=sys.stderr)
            return False

    @staticmethod
    def _calculate_scroll_delta(viewport: Viewport, target_x: float, target_y: float) -> Tuple[float, float]:
        """Work out how far to scroll to put a spot in the middle.

        Args:
            viewport: The window's size and place on the page.
            target_x: How far right the spot is on the page.
            target_y: How far down the spot is on the page.

        Returns:
            How far to scroll sideways and up or down.
        """
        target_scroll_x = max(0, min(target_x - viewport.client_width / 2, viewport.max_scroll_x))
        target_scroll_y = max(0, min(target_y - viewport.client_height / 2, viewport.max_scroll_y))
        return target_scroll_x - viewport.page_x, target_scroll_y - viewport.page_y

    async def _scroll_settled(self, target_id: str) -> bool:
        """Wait for the page to stop moving, and say if it stopped in time.

        Args:
            target_id: The tab to watch.

        Returns:
            True if the page stops moving before the wait runs out.
        """
        deadline = time.monotonic() + _SCROLL_SETTLE_TIMEOUT_S
        previous_pos: Optional[Tuple[float, float]] = None
        while time.monotonic() < deadline:
            await asyncio.sleep(_SCROLL_SETTLE_INTERVAL_S)
            viewport = await self.get_page_viewport(target_id)
            if viewport is None:
                continue
            current_pos = (viewport.page_x, viewport.page_y)
            if previous_pos is not None and previous_pos == current_pos:
                return True
            previous_pos = current_pos
        return False

    async def _scroll_step(self, target_id: str, x: float, y: float, page: Page, stagnant_count: int, verbose: bool) -> Tuple[str, Optional[int]]:
        """Take one scroll step toward a spot on the page.

        Args:
            target_id: The tab to scroll.
            x: How far right the spot is on the page.
            y: How far down the spot is on the page.
            page: The page to scroll.
            stagnant_count: How many steps in a row moved nothing.
            verbose: Print progress lines when True.

        Returns:
            The word success when the spot is in the window and the page
            has stopped moving, continue when more steps are needed, fail
            when the page cannot settle on the spot, or stagnant when the
            page stopped moving. The second value is the new count of
            steps that moved nothing, or None when there is no point going
            on.
        """
        vp = await self.get_page_viewport(target_id)
        if vp is None:
            print(f"[BrowserManager.scroll_to] Error: could not read the viewport", file=sys.stderr)
            return ("fail", None)
        scrollX, scrollY, vw, vh = vp.page_x, vp.page_y, vp.client_width, vp.client_height

        viewX, viewY = x - scrollX, y - scrollY
        if 0 <= viewX < vw and 0 <= viewY < vh:
            if await self._scroll_settled(target_id):
                return ("success", stagnant_count)
            print(
                f"[BrowserManager.scroll_to] Error: scroll did not settle within "
                f"{_SCROLL_SETTLE_TIMEOUT_S}s at ({scrollX}, {scrollY})",
                file=sys.stderr,
            )
            return ("fail", None)

        dx, dy = self._calculate_scroll_delta(vp, x, y)

        if abs(dx) < 10 and abs(dy) < 10:
            if 0 <= viewX < vw and 0 <= viewY < vh:
                return ("success", stagnant_count)
            print(
                f"[BrowserManager.scroll_to] Error: element not in viewport "
                f"and cannot scroll further at ({scrollX}, {scrollY})",
                file=sys.stderr,
            )
            return ("fail", None)

        if verbose:
            print(f"[BrowserManager.scroll_to] Need to scroll by ({dx}, {dy})", flush=True)

        before_pos = (scrollX, scrollY)
        await self.move_mouse_to(target_id, viewX, viewY, verbose=verbose)
        await self._human_scroll(page, dx, dy, verbose=verbose)

        after = await self.get_page_viewport(target_id)
        after_pos = (after.page_x, after.page_y) if after is not None else before_pos
        if after_pos == before_pos:
            stagnant_count += 1
            if stagnant_count >= 5:
                print(
                    f"[BrowserManager.scroll_to] Error: viewport unchanged after 5 scroll steps "
                    f"at ({before_pos[0]}, {before_pos[1]}). Target may be in a nested scroll container.",
                    file=sys.stderr,
                )
                return ("stagnant", None)
        else:
            stagnant_count = 0

        return ("continue", stagnant_count)

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
        if target_id not in self._pages:
            print(f"[BrowserManager.scroll_to] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if x < 0 or y < 0:
            print(f"[BrowserManager.scroll_to] Error: negative coordinates ({x}, {y})", file=sys.stderr)
            return False

        if verbose:
            print(f"[BrowserManager.scroll_to] Scrolling to document ({x}, {y})", flush=True)

        page = self._pages[target_id]
        try:
            stagnant_count = 0
            for _ in range(120):
                result, stagnant_count = await self._scroll_step(
                    target_id, x, y, page, stagnant_count, verbose
                )
                if result == "success":
                    if verbose:
                        msg = "Already in viewport" if stagnant_count == 0 else "Success"
                        print(f"[BrowserManager.scroll_to] {msg}", flush=True)
                    return True
                if result == "fail":
                    return False
                if result == "stagnant":
                    return False

            if verbose:
                print(f"[BrowserManager.scroll_to] Warning: reached max scroll attempts", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.scroll_to] Error: {e}", file=sys.stderr)
            return False

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
        if target_id not in self._pages:
            print(f"[BrowserManager.set_viewport_size] Error: target_id not found: {target_id}", file=sys.stderr)
            return False
        if verbose:
            print(f"[BrowserManager.set_viewport_size] {width}x{height}", flush=True)
        try:
            await self._pages[target_id].set_viewport_size({"width": width, "height": height})
            return True
        except Exception as e:
            print(f"[BrowserManager.set_viewport_size] Error: {e}", file=sys.stderr)
            return False

    async def _human_scroll(self, page: Page, dx: float, dy: float, verbose: bool = False) -> None:
        """Scroll a set distance in small steps, the way a person would.

        Args:
            page: The page to scroll.
            dx: How far to scroll sideways.
            dy: How far to scroll up or down.
            verbose: Print progress lines when True.
        """
        for component in ("x", "y"):
            total = dx if component == "x" else dy
            if total == 0:
                continue

            remaining = abs(total)
            while remaining > 0:
                step = min(remaining, random.randint(100, 300))
                step_val = step if total > 0 else -step
                await page.mouse.wheel(step_val if component == "x" else 0, step_val if component == "y" else 0)
                await asyncio.sleep(random.uniform(0.03, 0.08))
                remaining -= step

            if verbose:
                axis = component.upper()
                print(f"[BrowserManager.scroll_to] Scrolled {axis} by {total}", flush=True)

    @staticmethod
    def _center_in_viewport(cx: float, cy: float, w: float, h: float) -> bool:
        """Say if a spot is inside the window.

        Args:
            cx: How far right the spot is in the window.
            cy: How far down the spot is in the window.
            w: Window width.
            h: Window height.

        Returns:
            True if the spot is inside.
        """
        return 0 <= cx <= w and 0 <= cy <= h

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
        if target_id not in self._pages:
            print(f"[BrowserManager.scroll_to_view] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if verbose:
            print(f"[BrowserManager.scroll_to_view] Scrolling to '{selector}'", flush=True)
        page = self._pages[target_id]
        try:
            locator = page.locator(selector).first
            if not await locator.count():
                print(f"[BrowserManager.scroll_to_view] Error: element not found", file=sys.stderr)
                return False
            bbox = await locator.bounding_box(timeout=5000)
            if bbox is None:
                print(f"[BrowserManager.scroll_to_view] Error: could not get element bounds", file=sys.stderr)
                return False

            viewport = await self.get_page_viewport(target_id)
            if viewport is None:
                print(f"[BrowserManager.scroll_to_view] Error: could not read the viewport", file=sys.stderr)
                return False
            w = viewport.client_width
            h = viewport.client_height

            cx = bbox["x"] + bbox["width"] / 2
            cy = bbox["y"] + bbox["height"] / 2
            in_viewport = self._center_in_viewport(cx, cy, w, h)
            stagnant_count = 0
            for _ in range(_SCROLL_MAX_ITERATIONS):
                if in_viewport:
                    return True

                hx = max(_SCROLL_MARGIN, min(cx, w - _SCROLL_MARGIN))
                hy = max(_SCROLL_MARGIN, min(cy, h - _SCROLL_MARGIN))
                await self.move_mouse_to(target_id, hx, hy, verbose=verbose)
                dx = max(-_SCROLL_STEP_MAX, min(cx - w / 2, _SCROLL_STEP_MAX))
                dy = max(-_SCROLL_STEP_MAX, min(cy - h / 2, _SCROLL_STEP_MAX))
                if verbose:
                    print(f"[BrowserManager.scroll_to_view] Center ({cx}, {cy}) → scrolling by ({dx}, {dy})", flush=True)
                await self._human_scroll(page, dx, dy, verbose=verbose)

                after = await locator.bounding_box(timeout=5000)
                if after is None:
                    stagnant_count += 1
                else:
                    after_cx = after["x"] + after["width"] / 2
                    after_cy = after["y"] + after["height"] / 2
                    moved = abs(after_cx - cx) >= 1 or abs(after_cy - cy) >= 1
                    stagnant_count = 0 if moved else stagnant_count + 1
                    cx, cy = after_cx, after_cy
                    in_viewport = self._center_in_viewport(cx, cy, w, h)
                if stagnant_count >= _SCROLL_STAGNANT_LIMIT:
                    print(
                        f"[BrowserManager.scroll_to_view] Error: element stopped moving after "
                        f"{stagnant_count} scroll steps at ({cx}, {cy}); it may be pinned in a "
                        f"non-scrollable container",
                        file=sys.stderr,
                    )
                    return False

            if not in_viewport:
                print(
                    f"[BrowserManager.scroll_to_view] Error: element still out of viewport "
                    f"after {_SCROLL_MAX_ITERATIONS} scroll steps at ({cx}, {cy})",
                    file=sys.stderr,
                )
            return in_viewport
        except Exception as e:
            print(f"[BrowserManager.scroll_to_view] Error: {e}", file=sys.stderr)
            return False

    async def get_page_viewport(self, target_id: str) -> Optional[Viewport]:
        """Read the window's size and its place on the page.

        Args:
            target_id: The tab to read.

        Returns:
            The window's geometry, or None if it could not be read.
        """
        page = self._pages.get(target_id)
        if page is None:
            return None
        session = self._session_for_page(page)
        if session is None:
            return None
        try:
            result = await session.send("Page.getLayoutMetrics")
            view = result["cssVisualViewport"]
            content = result["cssContentSize"]
        except Exception as e:
            print(f"[BrowserManager.get_page_viewport] Error: {e}", file=sys.stderr)
            return None
        try:
            window = await page.evaluate(_WINDOW_SIZE_JS)
            window_width, window_height = window[0], window[1]
        except Exception as e:
            print(f"[BrowserManager.get_page_viewport] window size read failed: {e}", file=sys.stderr)
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

    def _session_for_page(self, page: Page) -> Optional[CDPSession]:
        """Get the CDP session of a tab."""
        state: Optional[_PageState] = self._page_states.get(page)
        return state.session if state is not None else None

    async def _get_clamp_viewport(self, target_id: str) -> Optional[Viewport]:
        """Read the window's size for the mouse clamp, once per tab.

        Args:
            target_id: The tab to read.

        Returns:
            The window's geometry, or None if it could not be read.
        """
        page = self._pages.get(target_id)
        if page is None:
            return None
        state: Optional[_PageState] = self._page_states.get(page)
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

        Args:
            target_id: The tab to scroll.
            distance: How far to scroll up or down. A positive number scrolls down.
            dx: How far to scroll sideways. A positive number scrolls right.
            verbose: Print progress lines when True.

        Returns:
            True if the page scrolled.
        """
        if target_id not in self._pages:
            print(f"[BrowserManager.scroll_by] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if verbose:
            print(f"[BrowserManager.scroll_by] Scrolling by ({dx}, {distance})px", flush=True)

        page = self._pages[target_id]
        try:
            await self._human_scroll(page, dx, distance, verbose=verbose)
            if verbose:
                print(f"[BrowserManager.scroll_by] Success", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.scroll_by] Error: {e}", file=sys.stderr)
            return False

    async def scroll_by_relative(self, target_id: str, dx: float, dy: float, viewport: Optional[Viewport] = None, verbose: bool = False) -> bool:
        """Scroll the page a share of the window in each direction.

        Args:
            target_id: The tab to scroll.
            dx: How far sideways, 1 is one window wide. A positive number scrolls right.
            dy: How far up or down, 1 is one window tall. A positive number scrolls down.
            viewport: The window's size and place on the page, or None to read it.
            verbose: Print progress lines when True.

        Returns:
            True if the page scrolled. False if the window could not be read.
        """
        if target_id not in self._pages:
            print(f"[BrowserManager.scroll_by_relative] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if viewport is None:
            viewport = await self.get_page_viewport(target_id)
            if viewport is None:
                print(f"[BrowserManager.scroll_by_relative] Error: could not read the viewport", file=sys.stderr)
                return False

        return await self.scroll_by(target_id, dy * viewport.usable_height, dx=dx * viewport.usable_width, verbose=verbose)

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
        if target_id not in self._pages:
            print(f"[BrowserManager.move_mouse_to] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        page = self._pages[target_id]
        try:
            viewport = await self.get_page_viewport(target_id)
            if viewport is None:
                print(f"[BrowserManager.move_mouse_to] Error: could not read the viewport", file=sys.stderr)
                return False
            w = viewport.window_width if viewport.window_width > 0 else viewport.client_width
            h = viewport.window_height if viewport.window_height > 0 else viewport.client_height
            cx = max(0, min(x, w))
            cy = max(0, min(y, h))

            if verbose:
                print(
                    f"[BrowserManager.move_mouse_to] Viewport ({x}, {y}) → "
                    f"capped ({cx}, {cy}), viewport size ({w}, {h})",
                    flush=True
                )

            await page.mouse.move(cx, cy)

            if verbose:
                print(f"[BrowserManager.move_mouse_to] Success", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.move_mouse_to] Error: {e}", file=sys.stderr)
            return False

    async def move_mouse_to_relative(self, target_id: str, x_ratio: float, y_ratio: float, viewport: Optional[Viewport] = None, verbose: bool = False) -> bool:
        """Move the mouse to a spot given as a share of the window.

        Args:
            target_id: The tab to move the mouse in.
            x_ratio: How far across, 0 is the left edge and 1 is the right.
            y_ratio: How far down, 0 is the top and 1 is the bottom.
            viewport: The window's size and place on the page, or None to read it.
            verbose: Print progress lines when True.

        Returns:
            True if the mouse moved. False if a share is outside 0 to 1.
        """
        if not (0 <= x_ratio <= 1) or not (0 <= y_ratio <= 1):
            print(f"[BrowserManager.move_mouse_to_relative] Error: ratios must be 0-1, got ({x_ratio}, {y_ratio})", file=sys.stderr)
            return False

        if target_id not in self._pages:
            print(f"[BrowserManager.move_mouse_to_relative] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        try:
            if viewport is None:
                viewport = await self.get_page_viewport(target_id)
                if viewport is None:
                    print(f"[BrowserManager.move_mouse_to_relative] Error: could not read the viewport", file=sys.stderr)
                    return False
            x = x_ratio * viewport.usable_width
            y = y_ratio * viewport.usable_height
            return await self.move_mouse_to(target_id, x, y, verbose=verbose)
        except Exception as e:
            print(f"[BrowserManager.move_mouse_to_relative] Error: {e}", file=sys.stderr)
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
        if target_id not in self._pages:
            print(f"[BrowserManager.click_at] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if x < 0 or y < 0:
            print(f"[BrowserManager.click_at] Error: negative coordinates ({x}, {y})", file=sys.stderr)
            return False

        if verbose:
            print(f"[BrowserManager.click_at] Clicking document ({x}, {y})", flush=True)

        if not await self.scroll_to(target_id, x, y, verbose=verbose):
            print(f"[BrowserManager.click_at] Error: scroll failed", file=sys.stderr)

        page = self._pages[target_id]
        try:
            viewport = await self.get_page_viewport(target_id)
            if viewport is None:
                print(f"[BrowserManager.click_at] Error: could not read the viewport", file=sys.stderr)
                return False
            viewX = x - viewport.page_x
            viewY = y - viewport.page_y

            if verbose:
                print(f"[BrowserManager.click_at] Clicking viewport ({viewX}, {viewY})", flush=True)

            await page.mouse.click(viewX, viewY)

            if verbose:
                print(f"[BrowserManager.click_at] Success", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.click_at] Error: {e}", file=sys.stderr)
            return False

    async def click_at_relative(self, target_id: str, x: float, y: float, viewport: Optional[Viewport] = None, verbose: bool = False) -> bool:
        """Click a spot given as a share of the window.

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
        if target_id not in self._pages:
            print(f"[BrowserManager.click_at_relative] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if viewport is None:
            viewport = await self.get_page_viewport(target_id)
            if viewport is None:
                print(f"[BrowserManager.click_at_relative] Error: could not read the viewport", file=sys.stderr)
                return False

        point_x = viewport.page_x + x * viewport.usable_width
        point_y = viewport.page_y + y * viewport.usable_height
        return await self.click_at(target_id, point_x, point_y, verbose=verbose)

    async def type_text_at(self, target_id: str, x: float, y: float, text: str, submit: bool = False, verbose: bool = False) -> bool:
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
        if target_id not in self._pages:
            print(f"[BrowserManager.type_text_at] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if not await self.click_at(target_id, x, y, verbose=verbose):
            print(f"[BrowserManager.type_text_at] Error: click failed", file=sys.stderr)
            return False

        page = self._pages[target_id]
        try:
            if verbose:
                text_short = text[:50] + "..." if len(text) > 50 else text
                print(f"[BrowserManager.type_text_at] Typing text: {text_short}", flush=True)
            await page.keyboard.type(text)
            if submit:
                if verbose:
                    print(f"[BrowserManager.type_text_at] Pressing Enter key", flush=True)
                await page.keyboard.press("Enter")
            if verbose:
                print(f"[BrowserManager.type_text_at] Success", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.type_text_at] Error: {e}", file=sys.stderr)
            return False

    async def type_text_at_relative(self, target_id: str, x: float, y: float, text: str, submit: bool = False, viewport: Optional[Viewport] = None, verbose: bool = False) -> bool:
        """Type text at a spot given as a share of the window.

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
        if target_id not in self._pages:
            print(f"[BrowserManager.type_text_at_relative] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if viewport is None:
            viewport = await self.get_page_viewport(target_id)
            if viewport is None:
                print(f"[BrowserManager.type_text_at_relative] Error: could not read the viewport", file=sys.stderr)
                return False

        point_x = viewport.page_x + x * viewport.usable_width
        point_y = viewport.page_y + y * viewport.usable_height
        return await self.type_text_at(target_id, point_x, point_y, text, submit=submit, verbose=verbose)

    async def replace_input_text_at(self, target_id: str, x: float, y: float, old_text: str, new_text: str, verbose: bool = False) -> bool:
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
        if target_id not in self._pages:
            print(f"[BrowserManager.replace_input_text_at] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if not old_text:
            print(f"[BrowserManager.replace_input_text_at] Error: old text is empty", file=sys.stderr)
            return False

        if not await self.click_at(target_id, x, y, verbose=verbose):
            print(f"[BrowserManager.replace_input_text_at] Error: click failed", file=sys.stderr)
            return False

        page = self._pages[target_id]
        try:
            frame = await self._find_focused_frame(page)
            if frame is None:
                print(f"[BrowserManager.replace_input_text_at] Error: focused frame not found", file=sys.stderr)
                return False
            selected = await frame.evaluate(_SELECT_FIELD_TEXT_JS, [old_text, new_text])
            if selected["status"] != "ok":
                print(f"[BrowserManager.replace_input_text_at] Error: {selected['status']}", file=sys.stderr)
                return False
            if verbose:
                old_short = old_text[:50] + "..." if len(old_text) > 50 else old_text
                print(f"[BrowserManager.replace_input_text_at] Replacing text: {old_short}", flush=True)
            text_to_insert = selected["expected"] if selected["selectedAll"] else new_text
            if text_to_insert:
                await page.keyboard.insert_text(text_to_insert)
            else:
                await page.keyboard.press("Delete")
            final_text = await frame.evaluate(_READ_FIELD_TEXT_JS)
            if final_text != selected["expected"]:
                print(f"[BrowserManager.replace_input_text_at] Error: field text did not change as expected", file=sys.stderr)
                return False
            if verbose:
                print(f"[BrowserManager.replace_input_text_at] Success", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.replace_input_text_at] Error: {e}", file=sys.stderr)
            return False

    @staticmethod
    async def _find_focused_frame(page: Page) -> Optional[Frame]:
        """Find the frame that holds the focused element.

        Args:
            page: The tab to look in.

        Returns:
            The frame with focus, or None if a focused inner frame cannot be reached.
        """
        frame = page.main_frame
        while True:
            handle = await frame.evaluate_handle(_FOCUSED_INNER_FRAME_JS)
            element = handle.as_element()
            child = await element.content_frame() if element else None
            await handle.dispose()
            if element is None:
                return frame
            if child is None:
                return None
            frame = child

    async def replace_input_text_at_relative(self, target_id: str, x: float, y: float, old_text: str, new_text: str, viewport: Optional[Viewport] = None, verbose: bool = False) -> bool:
        """Replace a piece of the text in the field at a spot given as a share of the window.

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
        if target_id not in self._pages:
            print(f"[BrowserManager.replace_input_text_at_relative] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if viewport is None:
            viewport = await self.get_page_viewport(target_id)
            if viewport is None:
                print(f"[BrowserManager.replace_input_text_at_relative] Error: could not read the viewport", file=sys.stderr)
                return False

        point_x = viewport.page_x + x * viewport.usable_width
        point_y = viewport.page_y + y * viewport.usable_height
        return await self.replace_input_text_at(target_id, point_x, point_y, old_text, new_text, verbose=verbose)

    async def solve_captcha(self, target_id: str, max_steps: int = 5, verbose: bool = False) -> bool:
        """Find and solve a captcha on a page.

        Args:
            target_id: The tab with the captcha.
            max_steps: How many tries to make.
            verbose: Print progress lines when True.

        Returns:
            True if the page has no captcha left.
        """
        return await _solve_captcha(self, target_id, max_steps, verbose)

    async def solve_recaptcha_grid(self, target_id: str, click_timeout: int, retry_times: int, verbose: bool = False) -> Optional[str]:
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
        if target_id not in self._pages:
            print(f"[BrowserManager.solve_recaptcha_grid] Error: target_id not found: {target_id}", file=sys.stderr)
            return None
        page = self._pages[target_id]
        challenger = AsyncChallenger(page, click_timeout=click_timeout, retry_times=retry_times, verbose=verbose)
        try:
            return await challenger.solve_recaptcha(skip_checkbox=True)
        except Exception as e:
            print(f"[BrowserManager.solve_recaptcha_grid] Error: {e}", file=sys.stderr)
            return None

    async def navigate(self, target_id: str, url: str, wait_until: str = "domcontentloaded", timeout: int = _DEFAULT_TIMEOUT_MS, verbose: bool = False) -> bool:
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
            _DownloadTriggered: If the address starts a file download.
        """
        page = self._get_page(target_id)
        if verbose:
            url_short = url[:50] + "..." if len(url) > 50 else url
            print(f"[BrowserManager.navigate] Navigating to {url_short} (wait_until={wait_until})", flush=True)
        try:
            response = await page.goto(url, wait_until=wait_until, timeout=timeout)
            if response is not None and not (200 <= response.status < 300):
                if verbose:
                    print(f"[BrowserManager.navigate] Non-2xx status: {response.status}", flush=True)
                return False
            if verbose:
                print(f"[BrowserManager.navigate] Success", flush=True)
            return True
        except Exception as e:
            if "Download is starting" in str(e):
                if verbose:
                    print(f"[BrowserManager.navigate] Download triggered", flush=True)
                raise _DownloadTriggered(f"Navigation to {url} triggered a download")
            if verbose:
                print(f"[BrowserManager.navigate] Error: {e}", flush=True)
            return False

    async def wait_for_load(self, target_id: str, state: str = "domcontentloaded", timeout: int = _DEFAULT_TIMEOUT_MS, verbose: bool = False) -> bool:
        """Wait for a page to finish loading.

        Args:
            target_id: The tab to wait on.
            state: How far the page must load.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.

        Returns:
            True if the page got that far in time.
        """
        if target_id not in self._pages:
            print(f"[BrowserManager.wait_for_load] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if verbose:
            print(f"[BrowserManager.wait_for_load] Waiting for state: {state}", flush=True)
        page = self._pages[target_id]
        try:
            await page.wait_for_load_state(state, timeout=timeout)
            if verbose:
                print(f"[BrowserManager.wait_for_load] Success", flush=True)
            return True
        except Exception as e:
            if verbose:
                print(f"[BrowserManager.wait_for_load] Error: {e}", file=sys.stderr)
            return False

    async def go_back(self, target_id: str, wait_until: str = "load", timeout: int = _DEFAULT_TIMEOUT_MS, verbose: bool = False) -> bool:
        """Go back to the page a tab was on before.

        Args:
            target_id: The tab to send back.
            wait_until: How far the page must load before this returns.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.

        Returns:
            True if the older page opened.
        """
        if target_id not in self._pages:
            print(f"[BrowserManager.go_back] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        page = self._pages[target_id]
        try:
            await page.go_back(wait_until=wait_until, timeout=timeout)
            if verbose:
                print("[BrowserManager.go_back] Success", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.go_back] Error: {e}", file=sys.stderr)
            return False

    async def go_forward(self, target_id: str, wait_until: str = "load", timeout: int = _DEFAULT_TIMEOUT_MS, verbose: bool = False) -> bool:
        """Go forward to the page a tab had been on.

        Args:
            target_id: The tab to send forward.
            wait_until: How far the page must load before this returns.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.

        Returns:
            True if the newer page opened.
        """
        if target_id not in self._pages:
            print(f"[BrowserManager.go_forward] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        page = self._pages[target_id]
        try:
            await page.go_forward(wait_until=wait_until, timeout=timeout)
            if verbose:
                print("[BrowserManager.go_forward] Success", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.go_forward] Error: {e}", file=sys.stderr)
            return False

    async def wait_for_selector(self, target_id: str, selector: str, timeout: int = _DEFAULT_TIMEOUT_MS, verbose: bool = False) -> bool:
        """Wait for a part of the page to show up.

        Args:
            target_id: The tab to watch.
            selector: Points at the part to wait for.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.

        Returns:
            True if the part showed up in time.
        """
        if target_id not in self._pages:
            print(f"[BrowserManager.wait_for_selector] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if verbose:
            print(f"[BrowserManager.wait_for_selector] Waiting for: {selector}", flush=True)
        page = self._pages[target_id]
        try:
            await page.wait_for_selector(selector, timeout=timeout)
            if verbose:
                print(f"[BrowserManager.wait_for_selector] Found", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.wait_for_selector] Error: {e}", file=sys.stderr)
            return False

    async def wait_for_selector_disappear(self, target_id: str, selector: str, timeout: int = _DEFAULT_TIMEOUT_MS, verbose: bool = False) -> bool:
        """Wait for a part of the page to go away.

        Args:
            target_id: The tab to watch.
            selector: Points at the part to wait on.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.

        Returns:
            True if the part went away in time.
        """
        if target_id not in self._pages:
            print(f"[BrowserManager.wait_for_selector_disappear] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if verbose:
            print(f"[BrowserManager.wait_for_selector_disappear] Waiting for: {selector}", flush=True)
        page = self._pages[target_id]
        try:
            await page.wait_for_selector(selector, state="detached", timeout=timeout)
            if verbose:
                print(f"[BrowserManager.wait_for_selector_disappear] Disappeared", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.wait_for_selector_disappear] Error: {e}", file=sys.stderr)
            return False

    async def _normalize_aria_hidden(self, page: Page) -> None:
        """Mark the page parts that people cannot see as hidden.

        Args:
            page: The page to change.
        """
        try:
            await page.evaluate("""() => {
              document.querySelectorAll('*').forEach(el => {
                const s = getComputedStyle(el);
                const hidden = s.display === 'none' || s.visibility === 'hidden';
                const attr = el.getAttribute('aria-hidden');
                if (hidden && attr !== 'true') {
                  el.setAttribute('aria-hidden', 'true');
                } else if (!hidden && attr === 'true') {
                  el.removeAttribute('aria-hidden');
                }
              });
            }""")
        except Exception:
            pass

    async def _capture_page_data(self, page: Page, target_id: str, verbose: bool) -> PageSnapshot:
        """Take a snapshot of a page.

        Args:
            page: The page to read.
            target_id: The tab the page is in.
            verbose: Print progress lines when True.

        Returns:
            The snapshot.
        """
        html = await page.content()
        current_url = page.url
        try:
            page_title = await page.title()
        except Exception:
            page_title = ""
        viewport = await self.get_page_viewport(target_id)
        if verbose:
            print(f"[BrowserManager.snapshot] HTML length: {len(html)}, URL: {current_url}", flush=True)
        return await PageSnapshot.create(html, self, target_id, url=current_url, title=page_title, viewport=viewport)

    async def snapshot(self, target_id: str, timeout: int = _DEFAULT_TIMEOUT_MS, verbose: bool = False) -> Optional[PageSnapshot]:
        """Take a frozen picture of a page that you can read and act on.

        Args:
            target_id: The tab to read.
            timeout: How long to wait for the page to settle, in milliseconds.
            verbose: Print progress lines when True.

        Returns:
            The snapshot, or None if the page could not be read.
        """
        if target_id not in self._pages:
            print(f"[BrowserManager.snapshot] Error: target_id not found: {target_id}", file=sys.stderr)
            return None

        if verbose:
            print(f"[BrowserManager.snapshot] Capturing snapshot...", flush=True)
        page = self._pages[target_id]
        try:
            if verbose:
                print(f"[BrowserManager.snapshot] Waiting for networkidle...", flush=True)
            try:
                await page.wait_for_load_state("networkidle", timeout=timeout)
                if verbose:
                    print(f"[BrowserManager.snapshot] networkidle reached", flush=True)
            except Exception as wait_err:
                if verbose:
                    print(f"[BrowserManager.snapshot] Warning: networkidle wait failed: {wait_err}", file=sys.stderr)

            await self._normalize_aria_hidden(page)
            return await self._capture_page_data(page, target_id, verbose)
        except Exception as e:
            print(f"[BrowserManager.snapshot] Error: {e}", file=sys.stderr)
            return None

    async def _snapshot_download_dir(self, verbose: bool = False) -> Dict[str, float]:
        """List the files in the download folder.

        Args:
            verbose: Print progress lines when True.

        Returns:
            The name of each file and when it last changed.
        """
        result = await asyncio.to_thread(self._list_download_files)
        if verbose:
            print(f"[BrowserManager._snapshot_download_dir] Found {len(result)} files", flush=True)
        return result

    def _list_download_files(self) -> Dict[str, float]:
        """List the files in the download folder.

        This waits on the disk, so do not call it on the event loop.

        Returns:
            The name of each file and when it last changed.
        """
        result: Dict[str, float] = {}
        try:
            for name in os.listdir(self._download_dir):
                if name.endswith(".crdownload"):
                    continue
                path = os.path.join(self._download_dir, name)
                try:
                    result[name] = os.path.getmtime(path)
                except OSError:
                    pass
        except Exception as e:
            print(f"[BrowserManager._list_download_files] Error: {e}", file=sys.stderr)
        return result

    async def _detach_cdp_session(self, page: Page, cdp: CDPSession) -> None:
        """Let go of a tab's CDP session. This never hangs.

        Nothing is sent for a closed tab.

        Args:
            page: The tab the session is attached to.
            cdp: The session to close.
        """
        if page.is_closed():
            return
        try:
            await asyncio.wait_for(cdp.detach(), timeout=_CLEANUP_STEP_TIMEOUT_S)
        except Exception as e:
            print(f"[BrowserManager._detach_cdp_session] Error: {e}", file=sys.stderr)

    async def _claim_single_new_file(self, snapshot_before: Dict[str, float], verbose: bool = False) -> Optional[str]:
        """Claim one new file in the download folder.

        A file that was already there but has just changed counts too.
        Nothing is claimed when the folder holds zero or multiple new files.

        Args:
            snapshot_before: The files in the folder before the download.
            verbose: Print progress lines when True.

        Returns:
            The path of the new file, or None when nothing was claimed.
        """
        snapshot_now = await self._snapshot_download_dir()
        new_files = [
            os.path.join(self._download_dir, name)
            for name, mtime in snapshot_now.items()
            if name not in snapshot_before or mtime > snapshot_before.get(name, 0)
        ]
        if len(new_files) == 1:
            if verbose:
                print(f"[BrowserManager._claim_single_new_file] Claiming {os.path.basename(new_files[0])}", flush=True)
            return new_files[0]
        if verbose:
            print(f"[BrowserManager._claim_single_new_file] {len(new_files)} new files, not claiming", flush=True)
        return None

    async def _claim_download(
        self,
        page: Page,
        effective_action: Callable[[], Awaitable[bool]],
        snapshot_before: Dict[str, float],
        timeout: int,
        verbose: bool,
    ) -> Optional[str]:
        """Run a step and claim the file the tab downloads.

        Args:
            page: The tab to watch.
            effective_action: The step that may start a download.
            snapshot_before: The files in the folder before the step.
            timeout: How many seconds to wait for the file.
            verbose: Print progress lines when True.

        Returns:
            The path of the finished file, or None when nothing was claimed.
        """
        session = self._session_for_page(page)
        if session is None:
            print(f"[BrowserManager._claim_download] Error: no session for the tab", file=sys.stderr)
            return None
        events = _DownloadEvents()
        try:
            await session.send(
                "Browser.setDownloadBehavior",
                {
                    "behavior": "allowAndName",
                    "downloadPath": self._download_dir,
                    "eventsEnabled": True,
                },
            )
            session.on("Browser.downloadWillBegin", events.on_will_begin)
            session.on("Browser.downloadProgress", events.on_progress)
        except Exception as e:
            print(f"[BrowserManager._claim_download] Error: {e}", file=sys.stderr)
            return None
        if verbose:
            print(f"[BrowserManager._claim_download] Download events on", flush=True)
        try:
            download_triggered = await effective_action()
            if not download_triggered:
                if verbose:
                    print(f"[BrowserManager._claim_download] No download, returning immediately", flush=True)
                return None
            try:
                await asyncio.wait_for(events.completed.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                if verbose:
                    print(f"[BrowserManager._claim_download] No download event in {timeout}s, claiming from folder", flush=True)
                return await self._claim_single_new_file(snapshot_before, verbose)
            return events.file_path
        finally:
            session.remove_listener("Browser.downloadWillBegin", events.on_will_begin)
            session.remove_listener("Browser.downloadProgress", events.on_progress)

    async def _finalize_download(self, src_path: str, save_path: Optional[str], verbose: bool = False) -> _DownloadResult:
        """Put a downloaded file where the caller wants it.

        Args:
            src_path: Where the file is now.
            save_path: Where to move it, or None to leave it be.
            verbose: Print progress lines when True.

        Returns:
            The path, name, and size of the file.
        """
        filename = os.path.basename(src_path)
        final_path = src_path

        if save_path:
            await asyncio.to_thread(os.makedirs, os.path.dirname(save_path) or ".", exist_ok=True)
            await asyncio.to_thread(shutil.move, src_path, save_path)
            final_path = save_path
            if verbose:
                print(f"[BrowserManager._finalize_download] Moved {src_path} -> {final_path}", flush=True)

        size = await asyncio.to_thread(os.path.getsize, final_path)
        if verbose:
            print(f"[BrowserManager._finalize_download] Final: {filename} ({size} bytes)", flush=True)

        return {
            "downloaded": True,
            "path": final_path,
            "filename": filename,
            "size": size,
        }

    def _build_effective_action(
        self,
        target_id: str,
        url: Optional[str],
        action: Optional[Callable[[], Awaitable[bool]]],
        timeout: int,
        verbose: bool,
    ) -> Callable[[], Awaitable[bool]]:
        """Make the step that may start a download.

        Args:
            target_id: The tab to work in.
            url: Address to open, or None to use action.
            action: Something to do instead of opening an address.
            timeout: How many seconds to wait for the page.
            verbose: Print progress lines when True.

        Returns:
            A step that says True when a download started.

        Raises:
            ValueError: If neither url nor action is given.
        """
        if url is not None:
            async def _url_action() -> bool:
                try:
                    ok = await self.navigate(target_id, url, wait_until="domcontentloaded", timeout=timeout * 1000, verbose=verbose)
                    if not ok:
                        raise RuntimeError(f"Navigation failed for {url}")
                    return False
                except _DownloadTriggered:
                    return True
                except Exception:
                    raise
            return _url_action
        if action is not None:
            return action
        raise ValueError("Either url or action must be provided")

    async def wait_for_download(
        self,
        target_id: str,
        url: Optional[str] = None,
        action: Optional[Callable[[], Awaitable[bool]]] = None,
        save_path: Optional[str] = None,
        timeout: int = 30,
        verbose: bool = False,
    ) -> _DownloadResult:
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
        if target_id not in self._pages:
            raise KeyError(f"No page found for target_id: {target_id}")

        if verbose:
            action_type = "url" if url else "action" if action else "none"
            print(f"[BrowserManager.wait_for_download] Starting ({action_type})", flush=True)

        page = self._pages[target_id]
        snapshot_before = await self._snapshot_download_dir(verbose=verbose)
        effective_action = self._build_effective_action(target_id, url, action, timeout, verbose)
        file_path = await self._claim_download(page, effective_action, snapshot_before, timeout, verbose)
        if file_path:
            return await self._finalize_download(file_path, save_path, verbose=verbose)
        return {"downloaded": False}

    def url(self, target_id: str, verbose: bool = False) -> Optional[str]:
        """Get the address a tab is showing.

        Args:
            target_id: The tab to ask about.
            verbose: Print progress lines when True.

        Returns:
            The address, or None if it could not be read.
        """
        if target_id not in self._pages:
            print(f"[BrowserManager.url] Error: target_id not found: {target_id}", file=sys.stderr)
            return None
        page = self._pages[target_id]
        if verbose:
            print(f"[BrowserManager.url] Getting URL for {target_id}", flush=True)
        try:
            url = page.url
            if verbose:
                print(f"[BrowserManager.url] URL: {url}", flush=True)
            return url
        except Exception as e:
            print(f"[BrowserManager.url] Error: {e}", file=sys.stderr)
            return None

    async def title(self, target_id: str, verbose: bool = False) -> Optional[str]:
        """Get the title of the page a tab is showing.

        Args:
            target_id: The tab to ask about.
            verbose: Print progress lines when True.

        Returns:
            The title, or None if it could not be read.
        """
        if target_id not in self._pages:
            print(f"[BrowserManager.title] Error: target_id not found: {target_id}", file=sys.stderr)
            return None
        page = self._pages[target_id]
        if verbose:
            print(f"[BrowserManager.title] Getting title for {target_id}", flush=True)
        try:
            title = await page.title()
            if verbose:
                print(f"[BrowserManager.title] Title: {title}", flush=True)
            return title
        except Exception as e:
            print(f"[BrowserManager.title] Error: {e}", file=sys.stderr)
            return None

    async def screenshot(self, target_id: str, path: str, full_page: bool = True, timeout: float = 30.0, verbose: bool = False) -> Optional[str]:
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
        if target_id not in self._pages:
            print(f"[BrowserManager.screenshot] Error: target_id not found: {target_id}", file=sys.stderr)
            return None
        page = self._pages[target_id]
        if verbose:
            print(f"[BrowserManager.screenshot] Saving to {path} (full_page={full_page})", flush=True)
        try:
            await page.screenshot(path=path, type="jpeg", full_page=full_page, timeout=int(timeout * 1000))
            if verbose:
                print(f"[BrowserManager.screenshot] Saved: {path}", flush=True)
            return path if await asyncio.to_thread(os.path.exists, path) else None
        except Exception as e:
            print(f"[BrowserManager.screenshot] Error: {e}", file=sys.stderr)
            return None

    async def pdf(self, target_id: str, path: str, verbose: bool = False) -> Optional[str]:
        """Save a page as a PDF file.

        Args:
            target_id: The tab to save.
            path: Where to save the PDF.
            verbose: Print progress lines when True.

        Returns:
            The path, or None if the PDF was not saved.
        """
        if target_id not in self._pages:
            print(f"[BrowserManager.pdf] Error: target_id not found: {target_id}", file=sys.stderr)
            return None
        page = self._pages[target_id]
        session = self._session_for_page(page)
        if session is None:
            print(f"[BrowserManager.pdf] Error: no session for {target_id}", file=sys.stderr)
            return None
        if verbose:
            print(f"[BrowserManager.pdf] Saving PDF to {path}", flush=True)
        try:
            result = await session.send("Page.printToPDF", {
                "printBackground": True,
            })
            pdf_bytes = base64.b64decode(result["data"])
            await asyncio.to_thread(os.makedirs, os.path.dirname(path) or ".", exist_ok=True)
            async with aiofiles.open(path, "wb") as fh:
                await fh.write(pdf_bytes)
            if verbose:
                print(f"[BrowserManager.pdf] Saved: {path}", flush=True)
            return path if await asyncio.to_thread(os.path.exists, path) else None
        except Exception as e:
            print(f"[BrowserManager.pdf] Error: {e}", file=sys.stderr)
            return None

    async def type_text(self, target_id: str, selector: str, text: str, submit: bool = False, timeout: int = _DEFAULT_TIMEOUT_MS, verbose: bool = False, human_config: Optional[dict] = None) -> bool:
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
        if target_id not in self._pages:
            print(f"[BrowserManager.type_text] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if not await self.scroll_to_view(target_id, selector, verbose=verbose):
            print(f"[BrowserManager.type_text] Error: scroll failed", file=sys.stderr)
            return False

        if verbose:
            text_short = text[:50] + "..." if len(text) > 50 else text
            print(f"[BrowserManager.type_text] Typing into '{selector}': {text_short} (submit={submit})", flush=True)
        page = self._pages[target_id]
        try:
            if human_config is None:
                await page.fill(selector, text, force=True, timeout=timeout)
            else:
                await page.fill(selector, text, force=True, timeout=timeout, human_config=human_config)
            if submit:
                await page.press(selector, "Enter", timeout=timeout)
            if verbose:
                print(f"[BrowserManager.type_text] Success", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.type_text] Error: {e}", file=sys.stderr)
            return False

    async def select_option(self, target_id: str, selector: str, value: str = None, index: int = None, label: str = None, verbose: bool = False) -> bool:
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
        if target_id not in self._pages:
            print(f"[BrowserManager.select_option] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if not await self.scroll_to_view(target_id, selector, verbose=verbose):
            print(f"[BrowserManager.select_option] Error: scroll failed", file=sys.stderr)
            return False

        page = self._pages[target_id]
        try:
            kwargs = {}
            if value is not None:
                kwargs['value'] = value
            elif index is not None:
                kwargs['index'] = index
            elif label is not None:
                kwargs['label'] = label
            else:
                print("[BrowserManager.select_option] Error: must specify value, index, or label", file=sys.stderr)
                return False

            locator = page.locator(selector).first
            await locator.select_option(force=True, **kwargs)
            if verbose:
                print(f"[BrowserManager.select_option] Success: {selector} -> {kwargs}", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.select_option] Error: {e}", file=sys.stderr)
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
        if target_id not in self._pages:
            print(f"[BrowserManager.press_key] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        page = self._pages[target_id]
        try:
            await page.keyboard.press(key)
            if verbose:
                print(f"[BrowserManager.press_key] Success: {key}", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.press_key] Error: {e}", file=sys.stderr)
            return False

    async def upload_with_chooser_relative(
        self, target_id: str, x: float, y: float, file_paths: list[str],
        viewport: Optional[Viewport] = None, verbose: bool = False,
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
        self, target_id: str, file_paths: list[str], trigger: Callable[[], Awaitable[bool]],
        verbose: bool = False,
    ) -> bool:
        """Open a file picker and give it the files."""
        if target_id not in self._pages:
            print(f"[BrowserManager._upload_with_chooser] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        if verbose:
            print(f"[BrowserManager._upload_with_chooser] Uploading {len(file_paths)} file(s) via chooser", flush=True)
        page = self._pages[target_id]
        try:
            async with page.expect_file_chooser(timeout=10000) as fc_info:
                if not await trigger():
                    raise RuntimeError("could not click the upload control")
            file_chooser = await fc_info.value
            await file_chooser.set_files(file_paths)
            if verbose:
                print(f"[BrowserManager._upload_with_chooser] Success", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager._upload_with_chooser] Error: {e}", file=sys.stderr)
            return False

    async def upload_files(self, target_id: str, selector: str, file_paths: list, verbose: bool = False) -> bool:
        """Give files to a file box on the page.

        Args:
            target_id: The tab to work in.
            selector: Points at the file box.
            file_paths: Paths of the files to send.
            verbose: Print progress lines when True.

        Returns:
            True if the files were sent.
        """
        if target_id not in self._pages:
            print(f"[BrowserManager.upload_files] Error: target_id not found: {target_id}", file=sys.stderr)
            return False

        page = self._pages[target_id]
        if verbose:
            print(f"[BrowserManager.upload_files] Uploading {len(file_paths)} file(s) to '{selector}'", flush=True)
        try:
            await page.set_input_files(selector, file_paths)
            if verbose:
                print(f"[BrowserManager.upload_files] Success", flush=True)
            return True
        except Exception as e:
            print(f"[BrowserManager.upload_files] Error: {e}", file=sys.stderr)
            return False

    async def cleanup(self, verbose: bool = False) -> None:
        """Close every tab and let go of the browser.

        This never hangs. A step that fails or takes too long is skipped.
        Calling this again does nothing.

        Args:
            verbose: Print progress lines when True.
        """
        if self._cleaned_up:
            return
        self._cleaned_up = True
        if verbose:
            print(f"[BrowserManager.cleanup] Closing {len(self._pages)} pages...", flush=True)
        for target_id in list(self._pages.keys()):
            await self.close_page(target_id, verbose=verbose)
        if self._browser is not None:
            try:
                await asyncio.wait_for(self._browser.close(), timeout=_CLEANUP_STEP_TIMEOUT_S)
                if verbose:
                    print(f"[BrowserManager.cleanup] Browser closed", flush=True)
            except Exception as e:
                print(f"[BrowserManager.cleanup] Error closing browser: {e}", file=sys.stderr)
        if self._pw is not None:
            try:
                await asyncio.wait_for(self._pw.stop(), timeout=_CLEANUP_STEP_TIMEOUT_S)
                if verbose:
                    print(f"[BrowserManager.cleanup] Playwright stopped", flush=True)
            except Exception as e:
                print(f"[BrowserManager.cleanup] Error stopping Playwright: {e}", file=sys.stderr)
        _MANAGERS.discard(self)

    def __del__(self) -> None:
        """Warn when a manager is dropped without being cleaned up.

        The browser is not let go of here.
        """
        if not self._cleaned_up and self._pw is not None:
            print(
                "[BrowserManager] Leaked browser connection: use "
                "'async with BrowserManager(...)' or await cleanup()",
                file=sys.stderr,
            )

    async def __aenter__(self) -> BrowserManager:
        await self._connect()
        return self

    async def __aexit__(self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> bool:
        await self.cleanup()
        return False


async def _clamp_mouse_params(mgr: BrowserManager, target_id: str, params: dict) -> None:
    """Pull the mouse spot back inside the window."""
    viewport = await mgr._get_clamp_viewport(target_id)
    if viewport is None:
        return
    width = viewport.usable_width
    height = viewport.usable_height
    if "x" in params and width > 0:
        params["x"] = max(0, min(params["x"], width - 1))
    if "y" in params and height > 0:
        params["y"] = max(0, min(params["y"], height - 1))


# Settling a scroll waits this long for the spot to stop moving before it is
# counted as reached, polling at this interval.
_SCROLL_SETTLE_TIMEOUT_S = 2.0
_SCROLL_SETTLE_INTERVAL_S = 0.05


if TYPE_CHECKING:
    from types import TracebackType


class _DownloadResult(TypedDict):
    """The state of one downloaded file."""

    downloaded: bool
    path: NotRequired[str]
    filename: NotRequired[str]
    size: NotRequired[int]


class _JsValue(Protocol):
    """A value that page code gives back."""


_install_mouse_viewport_clamp()
