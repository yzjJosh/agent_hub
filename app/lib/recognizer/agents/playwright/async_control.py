# ruff: noqa: F821

from __future__ import annotations

__all__ = ["AsyncChallenger"]

from contextlib import suppress

from ..constants import (
    CAPTCHA_ROUTE_REGEX,
    CHECKBOX_WAIT,
    DETECT_RETRY_MAX,
    DETECT_RETRY_TIMEOUT,
    LABEL_TEXT_TIMEOUT,
    LABEL_VISIBLE_TIMEOUT,
    NETWORK_IDLE_TIMEOUT,
    TILE_LOAD_TIMEOUT,
    TOKEN_CHECK_INTERVAL,
    TOKEN_CHECK_RETRIES,
    SCREENSHOT_TIMEOUT,
)
from .._types import (
    PatchrightAsyncElementHandle,
    PatchrightAsyncFrameLocator,
    PatchrightAsyncLocator,
    PatchrightError,
    PatchrightPage,
    PatchrightRequest,
    PatchrightRoute,
    PatchrightTimeoutError,
    PlaywrightAsyncElementHandle,
    PlaywrightAsyncFrameLocator,
    PlaywrightAsyncLocator,
    PlaywrightError,
    PlaywrightRequest,
    PlaywrightRoute,
    PlaywrightTimeoutError,
)
from .base_challenger import BaseChallenger, BrowserOps


class AsyncChallenger(BaseChallenger):
    """Solves the picture captcha on an async page."""

    @property
    def _browser_ops(self) -> _AsyncBrowserOps:
        """The page actions this challenger uses."""
        return _AsyncBrowserOps(self.page)

    async def route_handler(
        self,
        route: PlaywrightRoute | PatchrightRoute,
        request: PlaywrightRequest | PatchrightRequest,
    ) -> None:
        """Pass a captcha request through and read the reply.

        Args:
            route: The caught request to pass on.
            request: What was asked for.

        Raises:
            RuntimeError: If the reply is empty.
        """
        response = await route.fetch()
        await route.fulfill(response=response)  # type: ignore[arg-type]
        response_text = await response.text()
        if not response_text:
            raise RuntimeError("Empty response from route.fetch()")

        self.parse_route_response(response_text, request.url)

    async def _check_result(self) -> str | None:
        if self.captcha_token:
            return self.captcha_token

        with suppress(PlaywrightError, PatchrightError):
            captcha_token: str = await self._browser_ops.page_evaluate("grecaptcha.getResponse()")
            return captcha_token

        with suppress(PlaywrightError, PatchrightError):
            enterprise_captcha_token: str = await self._browser_ops.page_evaluate("grecaptcha.enterprise.getResponse()")
            return enterprise_captcha_token

        return None

    async def _check_captcha_visible(self) -> bool:
        captcha_frame = self.page.frame_locator("//iframe[contains(@src,'bframe')]")
        label_obj = captcha_frame.locator("//strong")
        try:
            if self._verbose:
                print("[recognizer] waiting for captcha label...", flush=True)
            await self._browser_ops.label_wait_for(label_obj, state="visible", timeout=LABEL_VISIBLE_TIMEOUT)
        except (PlaywrightTimeoutError, PatchrightTimeoutError):
            return False

        return await self._browser_ops.label_is_visible(label_obj)

    async def _click_checkbox(self) -> bool:
        try:
            if self._verbose:
                print("[recognizer] clicking checkbox...", flush=True)
            await self._browser_ops.checkbox_click()
            return True
        except (PlaywrightTimeoutError, PatchrightTimeoutError):
            return False

    async def _detect_tiles(self, prompt: str, area_captcha: bool) -> bool:
        if self._verbose:
            print("[recognizer] waiting for networkidle...", flush=True)
        await self._browser_ops.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT)

        image = await self._browser_ops.screenshot(timeout=SCREENSHOT_TIMEOUT)
        max_attempts = DETECT_RETRY_MAX
        attempt = 0
        while True:
            if self._verbose:
                print("[recognizer] running detector...", flush=True)
            result = await self.detector.detect_async(prompt, image, area_captcha=area_captcha)
            response, coordinates = result
            if response:
                break
            attempt += 1
            if attempt > max_attempts:
                if self._verbose:
                    print(f"[recognizer] no images loaded after {max_attempts} retries, failing early", flush=True)
                raise RuntimeError(f"[recognizer] no images loaded after {max_attempts} retries")
            if self._verbose:
                print(f"[recognizer] no images detected, retry {attempt}/{max_attempts}...", flush=True)
            await self._browser_ops.wait_for_timeout(DETECT_RETRY_TIMEOUT)
            image = await self._browser_ops.screenshot(timeout=SCREENSHOT_TIMEOUT)

        for coord_x, coord_y in coordinates:
            if self._verbose:
                print(f"[recognizer] clicking tile at ({coord_x}, {coord_y})", flush=True)
            await self._browser_ops.mouse_click(coord_x, coord_y)
            if self.click_timeout:
                await self._browser_ops.wait_for_timeout(self.click_timeout)

        return True

    async def _load_captcha(self) -> str | bool:
        self.advance_retry()

        TypedTimeoutError = PatchrightTimeoutError if isinstance(self.page, PatchrightPage) else PlaywrightTimeoutError
        if not await self._check_captcha_visible():
            if captcha_token := await self._check_result():
                return captcha_token
            elif not await self._click_checkbox():
                raise TypedTimeoutError("Invisible reCaptcha Timed Out.")

        if not await self._check_captcha_visible():
            raise TypedTimeoutError("[ERROR] reCaptcha Challenge is not visible.")
        return True

    async def _wait_for_tiles(self, captcha_frame: PlaywrightAsyncFrameLocator | PatchrightAsyncFrameLocator) -> list | None:
        tiles_locator = captcha_frame.locator("[class='rc-imageselect-tile']")

        async def _get_tiles() -> list:
            return await self._browser_ops.tiles_all(tiles_locator)

        async def _check_visible(elements: list) -> bool:
            if not elements:
                return False
            return await self._browser_ops.tile_is_visible(elements[0])

        async def _wait() -> None:
            await self._browser_ops.wait_for_timeout(TILE_LOAD_TIMEOUT)

        return await self.poll_for_tiles_async(_get_tiles, _check_visible, _wait, self._verbose)

    async def _handle_recaptcha(self) -> str | None:
        while self._retried < self._retry_times:
            if isinstance(loaded_captcha := await self._load_captcha(), str):
                return loaded_captcha

            captcha_frame = self.page.frame_locator("//iframe[contains(@src,'bframe')]")
            label_obj = captcha_frame.locator("//strong")
            if not (prompt := await self._browser_ops.label_text_content(label_obj, timeout=LABEL_TEXT_TIMEOUT)):
                raise ValueError("reCaptcha Task Text did not load.")

            recaptcha_tiles = await self._wait_for_tiles(captcha_frame)
            if recaptcha_tiles is None:
                self.advance_retry()
                continue

            area_captcha = len(recaptcha_tiles) == 16
            result_clicked = await self._detect_tiles(prompt, area_captcha)
            if self._verbose:
                print(f"[recognizer] dynamic={self._is_dynamic} area_captcha={area_captcha} result_clicked={result_clicked}", flush=True)

            if self._is_dynamic and not area_captcha:
                while result_clicked:
                    await self._browser_ops.wait_for_timeout(DETECT_RETRY_TIMEOUT)
                    result_clicked = await self._detect_tiles(prompt, area_captcha)

            if self._verbose:
                print("[recognizer] submitting...", flush=True)
            try:
                await self._browser_ops.submit_click(captcha_frame.locator("#recaptcha-verify-button"))
            except (PlaywrightTimeoutError, PatchrightTimeoutError):
                self.advance_retry()
                continue

            captcha_token = await self._poll_for_token()
            if captcha_token:
                return captcha_token

            if self._verbose:
                print("[recognizer] no token after submit, continuing...", flush=True)
            self.advance_retry()
            continue

    async def _poll_for_token(self) -> str | None:
        for _ in range(TOKEN_CHECK_RETRIES):
            if captcha_token := await self._check_result():
                return captcha_token
            await self._browser_ops.wait_for_timeout(TOKEN_CHECK_INTERVAL)
        return None

    async def solve_recaptcha(self, skip_checkbox: bool = False) -> str | None:
        """Solve the captcha on the page.

        Args:
            skip_checkbox: True to go straight to the picture challenge.

        Returns:
            The captcha token, or None if there was no try left to make.

        Raises:
            RuntimeError: If it runs out of tries, or if no pictures load.
            ValueError: If the challenge text does not show up.
            PlaywrightTimeoutError: If the challenge never shows up.
        """
        self._is_dynamic = False
        self.captcha_token = None
        self._retried = 0

        if not self.is_routed:
            await self.page.route(CAPTCHA_ROUTE_REGEX, self.route_handler)
            self.is_routed = True

        if not skip_checkbox:
            await self._click_checkbox()
            await self.page.wait_for_timeout(CHECKBOX_WAIT)
        return await self._handle_recaptcha()


class _AsyncBrowserOps(BrowserOps):
    async def wait_for_load_state(self, state: str, timeout: int) -> None:
        """Wait for the page to reach a loading state."""
        await self.page.wait_for_load_state(state, timeout=timeout)

    async def screenshot(self, timeout: int = 0) -> bytes:
        """Take a picture of the page."""
        return await self.page.screenshot(timeout=timeout)

    async def wait_for_timeout(self, timeout: int) -> None:
        """Wait this many milliseconds."""
        await self.page.wait_for_timeout(timeout)

    async def mouse_click(self, x: int, y: int) -> None:
        """Click a spot on the page."""
        await self.page.mouse.click(x, y)

    async def page_evaluate(self, expr: str) -> str:
        """Run a bit of code in the page and give back the answer."""
        return await self.page.evaluate(expr)

    async def label_wait_for(self, label_obj: PlaywrightAsyncLocator | PatchrightAsyncLocator, state: str, timeout: int) -> None:
        """Wait for a label to reach a state."""
        await label_obj.wait_for(state=state, timeout=timeout)

    async def label_is_visible(self, label_obj: PlaywrightAsyncLocator | PatchrightAsyncLocator) -> bool:
        """Say if a label can be seen."""
        return await label_obj.is_visible()

    async def checkbox_click(self) -> None:
        """Click the captcha checkbox."""
        checkbox = self.page.frame_locator("iframe[title='reCAPTCHA']").first
        await checkbox.locator(".recaptcha-checkbox-border").click()

    async def label_text_content(self, label_obj: PlaywrightAsyncLocator | PatchrightAsyncLocator, timeout: int) -> str | None:
        """Give back the text of a label."""
        return await label_obj.text_content(timeout=timeout)

    async def tiles_all(self, tiles_locator: PlaywrightAsyncLocator | PatchrightAsyncLocator) -> list:
        """Give back every tile."""
        return await tiles_locator.all()

    async def tile_is_visible(self, tile: PlaywrightAsyncElementHandle | PatchrightAsyncElementHandle) -> bool:
        """Say if a tile can be seen."""
        return await tile.is_visible()

    async def submit_click(self, submit_locator: PlaywrightAsyncLocator | PatchrightAsyncLocator) -> None:
        """Click the submit button."""
        await submit_locator.click()