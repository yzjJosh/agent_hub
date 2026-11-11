# ruff: noqa: F821
from __future__ import annotations

__all__ = ["BaseChallenger", "BrowserOps"]

import re
from abc import ABC, abstractmethod
from typing import Awaitable, Callable

from .._types import PatchrightPage, PlaywrightPage
from ..constants import TILE_LOAD_RETRIES, GRID_SIZES
from ...components.detector import Detector


class BaseChallenger:
    """What every captcha challenger keeps track of.

    The sync and async challengers add the solving steps.
    """

    def __init__(
        self,
        page: PlaywrightPage | PatchrightPage,
        click_timeout: int | None = None,
        retry_times: int = 15,
        optimize_click_order: bool = True,
        verbose: bool = False,
    ) -> None:
        """Set up a challenger for one page.

        Args:
            page: The page with the captcha on it.
            click_timeout: How long to wait after each tile click, in milliseconds.
            retry_times: How many tries before giving up.
            optimize_click_order: True to click the tiles in a shorter path.
            verbose: True to print what is happening.
        """
        self.page: PlaywrightPage | PatchrightPage = page
        self.is_routed = False
        self.detector = Detector(optimize_click_order=optimize_click_order)
        self.click_timeout = click_timeout
        self._retry_times = retry_times
        self._retried = 0
        self._verbose = verbose
        self._is_dynamic: bool = False
        self.captcha_token: str | None = None

    @staticmethod
    def _tile_count(tile_elements: list) -> int:
        """Count the tiles, and count nothing as zero."""
        return len(tile_elements) if tile_elements else 0

    @staticmethod
    def _are_tiles_ready(tile_elements: list, is_visible: bool) -> bool:
        """Say if the tiles make a full grid that can be seen."""
        return BaseChallenger._tile_count(tile_elements) in GRID_SIZES and is_visible

    @staticmethod
    def _log_tiles_not_ready(tile_elements: list, is_verbose: bool) -> None:
        """Print how many tiles there are while the grid is not full."""
        if is_verbose:
            tile_count = BaseChallenger._tile_count(tile_elements)
            print(f"[recognizer] tiles not ready ({tile_count}), waiting...", flush=True)

    @staticmethod
    def poll_for_tiles(
        get_tiles: Callable[[], list],
        check_visible: Callable[[list], bool],
        wait_timeout: Callable[[], None],
        is_verbose: bool,
    ) -> list | None:
        """Wait for the captcha tiles to be ready.

        Args:
            get_tiles: Gives back the tiles.
            check_visible: Says if a tile can be seen.
            wait_timeout: Waits before the next look.
            is_verbose: True to print what is happening.

        Returns:
            The tiles, or None if they never got ready.
        """
        for _ in range(TILE_LOAD_RETRIES):
            if is_verbose:
                print("[recognizer] checking tiles...", flush=True)
            tile_elements = get_tiles()
            if BaseChallenger._are_tiles_ready(tile_elements, check_visible(tile_elements)):
                return tile_elements
            BaseChallenger._log_tiles_not_ready(tile_elements, is_verbose)
            wait_timeout()
        return None

    @staticmethod
    async def poll_for_tiles_async(
        get_tiles: Callable[[], Awaitable[list]],
        check_visible: Callable[[list], Awaitable[bool]],
        wait_timeout: Callable[[], Awaitable[None]],
        is_verbose: bool,
    ) -> list | None:
        """Wait for the captcha tiles to be ready on an async page.

        Every callable given here is awaited.

        Args:
            get_tiles: Gives back the tiles.
            check_visible: Says if a tile can be seen.
            wait_timeout: Waits before the next look.
            is_verbose: True to print what is happening.

        Returns:
            The tiles, or None if they never got ready.
        """
        for _ in range(TILE_LOAD_RETRIES):
            if is_verbose:
                print("[recognizer] checking tiles...", flush=True)
            tile_elements = await get_tiles()
            is_visible = await check_visible(tile_elements)
            if BaseChallenger._are_tiles_ready(tile_elements, is_visible):
                return tile_elements
            BaseChallenger._log_tiles_not_ready(tile_elements, is_verbose)
            await wait_timeout()
        return None

    def advance_retry(self) -> None:
        """Count one more try.

        Raises:
            RuntimeError: If there are no tries left.
        """
        self._retried += 1
        if self._retried >= self._retry_times:
            raise RuntimeError(f"Exceeded maximum retry times of {self._retry_times}")

    def parse_route_response(
        self,
        response_text: str,
        request_url: str,
    ) -> None:
        """Read a captcha reply and keep the token from it.

        It also notes if the challenge swaps in new pictures.

        Args:
            response_text: The reply text.
            request_url: The address that was asked for.
        """
        self._is_dynamic = "dynamic" in response_text
        if self._verbose:
            print(f"[recognizer] route_handler: dynamic={self._is_dynamic}", flush=True)

        if "userverify" in request_url and "rresp" not in response_text and "bgdata" not in response_text:
            match = re.search(r'"uvresp"\s*,\s*"([^"]+)"', response_text)
            if not match:
                return
            self.captcha_token = match.group(1)


class BrowserOps(ABC):
    """What a challenger needs from a page.

    The sync and async versions have the same methods and arguments.
    """

    def __init__(self, page: PlaywrightPage | PatchrightPage) -> None:
        self.page = page

    @abstractmethod
    def wait_for_load_state(self, state: str, timeout: int) -> None:
        """Wait for the page to reach a loading state."""
        ...

    @abstractmethod
    def screenshot(self, timeout: int = 0) -> bytes:
        """Take a picture of the page."""
        ...

    @abstractmethod
    def wait_for_timeout(self, timeout: int) -> None:
        """Wait this many milliseconds."""
        ...

    @abstractmethod
    def mouse_click(self, x: int, y: int) -> None:
        """Click a spot on the page."""
        ...

    @abstractmethod
    def page_evaluate(self, expr: str) -> str:
        """Run a bit of code in the page and give back the answer."""
        ...

    @abstractmethod
    def label_wait_for[T](self, label_obj: T, state: str, timeout: int) -> None:
        """Wait for a label to reach a state."""
        ...

    @abstractmethod
    def label_is_visible[T](self, label_obj: T) -> bool:
        """Say if a label can be seen."""
        ...

    @abstractmethod
    def checkbox_click(self) -> None:
        """Click the captcha checkbox."""
        ...

    @abstractmethod
    def label_text_content[T](self, label_obj: T, timeout: int) -> str | None:
        """Give back the text of a label."""
        ...

    @abstractmethod
    def tiles_all[T](self, tiles_locator: T) -> list:
        """Give back every tile."""
        ...

    @abstractmethod
    def tile_is_visible[T](self, tile: T) -> bool:
        """Say if a tile can be seen."""
        ...

    @abstractmethod
    def submit_click[T](self, submit_locator: T) -> None:
        """Click the submit button."""
        ...