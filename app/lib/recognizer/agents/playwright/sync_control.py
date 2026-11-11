# ruff: noqa: F821
from __future__ import annotations

__all__ = ["SyncChallenger"]

from contextlib import suppress
from typing import TYPE_CHECKING

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
    PatchrightError,
    PatchrightPage,
    PatchrightRequest,
    PatchrightRoute,
    PatchrightTimeoutError,
    PlaywrightError,
    PlaywrightRequest,
    PlaywrightRoute,
    PlaywrightTimeoutError,
)
from .base_challenger import BaseChallenger, BrowserOps


# lint-skip: coding.exported-api-unused
# Approved by user. SyncChallenger is kept as public API for sync Playwright
# callers; production now drives the browser through AsyncChallenger.
class SyncChallenger(BaseChallenger):
    @property
    def _browser_ops(self) -> _SyncBrowserOps:
        """Return a sync browser operations wrapper for the current page."""
        return _SyncBrowserOps(self.page)

    def route_handler(
        self,
        route: PlaywrightRoute | PatchrightRoute,
        request: PlaywrightRequest | PatchrightRequest,
    ) -> None:
        """Intercept and handle reCAPTCHA network requests.

        Fetches the response, fulfills the route, and inspects the response
        body to detect dynamic challenges and extract captcha tokens.

        Args:
            route: The intercepted route object.
            request: The intercepted request object.

        Raises:
            RuntimeError: If the route fetch returns an empty response.
        """
        response = route.fetch()
        route.fulfill(response=response)  # type: ignore[arg-type]
        response_text = response.text()
        if not response_text:
            raise RuntimeError("Empty response from route.fetch()")

        self.parse_route_response(response_text, request.url)

    def _check_result(self) -> str | None:
        if self.captcha_token:
            return self.captcha_token

        with suppress(PlaywrightError, PatchrightError):
            captcha_token: str = self._browser_ops.page_evaluate("grecaptcha.getResponse()")
            return captcha_token

        with suppress(PlaywrightError, PatchrightError):
            enterprise_captcha_token: str = self._browser_ops.page_evaluate("grecaptcha.enterprise.getResponse()")
            return enterprise_captcha_token

        return None

    def _check_captcha_visible(self) -> bool:
        captcha_frame = self.page.frame_locator("//iframe[contains(@src,'bframe')]")
        label_obj = captcha_frame.locator("//strong")
        try:
            if self._verbose:
                print("[recognizer] waiting for captcha label...", flush=True)
            self._browser_ops.label_wait_for(label_obj, state="visible", timeout=LABEL_VISIBLE_TIMEOUT)
        except (PlaywrightTimeoutError, PatchrightTimeoutError):
            return False

        return self._browser_ops.label_is_visible(label_obj)

    def _click_checkbox(self) -> bool:
        try:
            if self._verbose:
                print("[recognizer] clicking checkbox...", flush=True)
            self._browser_ops.checkbox_click()
            return True
        except (PlaywrightTimeoutError, PatchrightTimeoutError):
            return False

    def _detect_tiles(self, prompt: str, area_captcha: bool) -> bool:
        if self._verbose:
            print("[recognizer] waiting for networkidle...", flush=True)
        self._browser_ops.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT)

        img_data = self._browser_ops.screenshot(timeout=SCREENSHOT_TIMEOUT)
        retry_limit = DETECT_RETRY_MAX
        retries = 0
        detected = False
        while not detected:
            if self._verbose:
                print("[recognizer] running detector...", flush=True)
            response, coordinates = self.detector.detect(prompt, img_data, area_captcha=area_captcha)
            if response:
                detected = True
                break
            retries += 1
            if retries > retry_limit:
                if self._verbose:
                    print(f"[recognizer] no images loaded after {retry_limit} retries, failing early", flush=True)
                raise RuntimeError(f"[recognizer] no images loaded after {retry_limit} retries")
            if self._verbose:
                print(f"[recognizer] no images detected, retry {retries}/{retry_limit}...", flush=True)
            self._browser_ops.wait_for_timeout(DETECT_RETRY_TIMEOUT)
            img_data = self._browser_ops.screenshot(timeout=SCREENSHOT_TIMEOUT)

        for coord_x, coord_y in coordinates:
            if self._verbose:
                print(f"[recognizer] clicking tile at ({coord_x}, {coord_y})", flush=True)
            self._browser_ops.mouse_click(coord_x, coord_y)
            if self.click_timeout:
                self._browser_ops.wait_for_timeout(self.click_timeout)

        return True

    def _load_captcha(self) -> str | bool:
        self.advance_retry()

        TypedTimeoutError = PatchrightTimeoutError if isinstance(self.page, PatchrightPage) else PlaywrightTimeoutError
        if not self._check_captcha_visible():
            if captcha_token := self._check_result():
                return captcha_token
            elif not self._click_checkbox():
                raise TypedTimeoutError("Invisible reCaptcha Timed Out.")

        if not self._check_captcha_visible():
            raise TypedTimeoutError("[ERROR] reCaptcha Challenge is not visible.")
        return True

    def _wait_for_tiles(self, captcha_frame: PlaywrightFrameLocator | PatchrightFrameLocator) -> list | None:
        tiles_locator = captcha_frame.locator("[class='rc-imageselect-tile']")

        def _get_tiles() -> list:
            return self._browser_ops.tiles_all(tiles_locator)

        def _check_visible(elements: list) -> bool:
            if not elements:
                return False
            return self._browser_ops.tile_is_visible(elements[0])

        def _wait() -> None:
            self._browser_ops.wait_for_timeout(TILE_LOAD_TIMEOUT)

        return self.poll_for_tiles(_get_tiles, _check_visible, _wait, self._verbose)

    def _handle_recaptcha(self) -> str | None:
        while self._retried < self._retry_times:
            if isinstance(loaded_captcha := self._load_captcha(), str):
                return loaded_captcha

            captcha_frame = self.page.frame_locator("//iframe[contains(@src,'bframe')]")
            label_obj = captcha_frame.locator("//strong")
            if not (prompt := self._browser_ops.label_text_content(label_obj, timeout=LABEL_TEXT_TIMEOUT)):
                raise ValueError("reCaptcha Task Text did not load.")

            recaptcha_tiles = self._wait_for_tiles(captcha_frame)
            if recaptcha_tiles is None:
                self.advance_retry()
                continue

            area_captcha = len(recaptcha_tiles) == 16
            result_clicked = self._detect_tiles(prompt, area_captcha)
            if self._verbose:
                print(f"[recognizer] dynamic={self._is_dynamic} area_captcha={area_captcha} result_clicked={result_clicked}", flush=True)

            if self._is_dynamic and not area_captcha:
                while result_clicked:
                    self._browser_ops.wait_for_timeout(DETECT_RETRY_TIMEOUT)
                    result_clicked = self._detect_tiles(prompt, area_captcha)

            if self._verbose:
                print("[recognizer] submitting...", flush=True)
            try:
                self._browser_ops.submit_click(captcha_frame.locator("#recaptcha-verify-button"))
            except (PlaywrightTimeoutError, PatchrightTimeoutError):
                self.advance_retry()
                continue

            token = self._poll_for_token()
            if token:
                return token

            if self._verbose:
                print("[recognizer] no token after submit, continuing...", flush=True)
            self.advance_retry()
            continue

    def _poll_for_token(self) -> str | None:
        for _ in range(TOKEN_CHECK_RETRIES):
            if captcha_token := self._check_result():
                return captcha_token
            self._browser_ops.wait_for_timeout(TOKEN_CHECK_INTERVAL)
        return None

    def solve_recaptcha(self, skip_checkbox: bool = False) -> str | None:
        """Solve the reCAPTCHA challenge on the current page.

        Args:
            skip_checkbox: If True, skip the initial checkbox click and go
                straight to the image challenge. Useful when the checkbox was
                already clicked externally.

        Returns:
            The reCAPTCHA token string on success, or None if the challenge
            could not be completed within the retry limit.

        Raises:
            RuntimeError: When the maximum retry limit is exceeded.
            PlaywrightTimeoutError: When the reCAPTCHA challenge is not visible.
        """
        self._is_dynamic = False
        self.captcha_token = None
        self._retried = 0

        if not self.is_routed:
            self.page.route(CAPTCHA_ROUTE_REGEX, self.route_handler)
            self.is_routed = True

        if not skip_checkbox:
            self._click_checkbox()
            self.page.wait_for_timeout(CHECKBOX_WAIT)
        return self._handle_recaptcha()


class _SyncBrowserOps(BrowserOps):
    def wait_for_load_state(self, state: str, timeout: int) -> None:
        """Wait for the page to reach the specified load state."""
        self.page.wait_for_load_state(state, timeout=timeout)

    def screenshot(self, timeout: int = 0) -> bytes:
        """Take a screenshot of the current page and return it as bytes."""
        return self.page.screenshot(timeout=timeout)

    def wait_for_timeout(self, timeout: int) -> None:
        """Wait for the specified number of milliseconds."""
        self.page.wait_for_timeout(timeout)

    def mouse_click(self, x: int, y: int) -> None:
        """Click at the specified x, y coordinates on the page."""
        self.page.mouse.click(x, y)

    def page_evaluate(self, expr: str) -> str:
        """Evaluate a JavaScript expression in the browser context."""
        return self.page.evaluate(expr)

    def label_wait_for(self, label_obj: PlaywrightLocator | PatchrightLocator, state: str, timeout: int) -> None:
        """Wait for a label element to reach the specified state."""
        label_obj.wait_for(state=state, timeout=timeout)

    def label_is_visible(self, label_obj: PlaywrightLocator | PatchrightLocator) -> bool:
        """Check whether a label element is currently visible."""
        return label_obj.is_visible()

    def checkbox_click(self) -> None:
        """Click the reCAPTCHA checkbox to initiate the challenge."""
        checkbox = self.page.frame_locator("iframe[title='reCAPTCHA']").first
        checkbox.locator(".recaptcha-checkbox-border").click()

    def label_text_content(self, label_obj: PlaywrightLocator | PatchrightLocator, timeout: int) -> str | None:
        """Extract the text content of a label element."""
        return label_obj.text_content(timeout=timeout)

    def tiles_all(self, tiles_locator: PlaywrightLocator | PatchrightLocator) -> list:
        """Retrieve all tile elements matching the given locator."""
        return tiles_locator.all()

    def tile_is_visible(self, tile: PlaywrightElementHandle | PatchrightElementHandle) -> bool:
        """Check whether a single tile element is visible."""
        return tile.is_visible()

    def submit_click(self, submit_locator: PlaywrightLocator | PatchrightLocator) -> None:
        """Click the submit button to verify the captcha solution."""
        submit_locator.click()


if TYPE_CHECKING:
    from .._types import (
        PatchrightElementHandle,
        PatchrightFrameLocator,
        PatchrightLocator,
        PlaywrightElementHandle,
        PlaywrightFrameLocator,
        PlaywrightLocator,
    )