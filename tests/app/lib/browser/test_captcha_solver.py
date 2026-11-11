"""Tests for solving a captcha on a tab."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image as PilImage

import app.lib.browser.captcha_solver as captcha_solver
from app.lib.browser.browser_manager import BrowserManager, Viewport
from app.lib.browser.captcha_solver import solve_captcha
from tests.app.conftest import skip_if_no_chrome


def _viewport(width: int, height: int) -> Viewport:
    """Make a viewport the size of one window with no scrollbars."""
    return Viewport(
        client_width=width,
        client_height=height,
        page_x=0.0,
        page_y=0.0,
        content_width=width,
        content_height=height,
        window_width=width,
        window_height=height,
    )


def _make_manager() -> AsyncMock:
    """Make a fake browser manager that shoots a blank 800x600 picture."""

    def _save(_target_id: str, path: str, **_kw: bool) -> str:
        PilImage.new("RGB", (800, 600), "white").save(path)
        return path

    mgr = AsyncMock()
    mgr.screenshot = AsyncMock(side_effect=_save)
    mgr.tab_ids = MagicMock(return_value=["tab1"])
    mgr.click_at = AsyncMock(return_value=True)
    mgr.get_page_viewport = AsyncMock(return_value=_viewport(800, 600))
    mgr.wait_for_load = AsyncMock(return_value=True)
    mgr.solve_recaptcha_grid = AsyncMock(return_value=None)
    return mgr


class TestCaptchaSolverSolveCaptcha:
    """Tests for the captcha solving loop."""

    async def test_solve_captcha_no_captcha(self) -> None:
        """A page with no captcha counts as solved."""
        mgr = _make_manager()

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.return_value = (
                '{"has_captcha": false, "captcha_type": "unknown", '
                '"reason": "no captcha"}',
                0,
            )

            result = await solve_captcha(mgr, "tab1")

        assert result is True
        mgr.screenshot.assert_called_once()
        assert mgr.screenshot.call_args[1].get("full_page") is False
        assert mgr.click_at.call_count == 0

    @pytest.mark.fast_sleeps
    async def test_solve_captcha_checkbox_solved(self) -> None:
        """A checkbox captcha is solved by one click."""
        mgr = _make_manager()

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.side_effect = [
                # First look finds a checkbox.
                ('{"has_captcha": true, "captcha_type": "checkbox", '
                 '"reason": "checkbox captcha"}', 0),
                # The checkbox can be clicked in the middle.
                ('{"clickable": true, "x": 0.5, "y": 0.5, '
                 '"reason": "click checkbox"}', 0),
                # Second look finds nothing.
                ('{"has_captcha": false, "captcha_type": "unknown", '
                 '"reason": "solved"}', 0),
            ]

            result = await solve_captcha(mgr, "tab1")

        assert result is True
        # One picture for each look and one for the click.
        assert mgr.screenshot.call_count == 3
        assert mgr.click_at.call_count == 1

    async def test_solve_captcha_unknown_type(self) -> None:
        """A captcha of a kind the model does not know is not solved."""
        mgr = _make_manager()

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.return_value = (
                '{"has_captcha": true, "captcha_type": "unknown", '
                '"reason": "requires sliding"}',
                0,
            )

            result = await solve_captcha(mgr, "tab1")

        assert result is False
        assert mgr.screenshot.call_count == 1
        assert mgr.click_at.call_count == 0

    @pytest.mark.fast_sleeps
    async def test_solve_captcha_max_steps(self) -> None:
        """A captcha that never goes away is not solved."""
        mgr = _make_manager()

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            # Each step asks the model twice, so three steps ask six times.
            mock_reason.side_effect = [
                # Step one look.
                ('{"has_captcha": true, "captcha_type": "checkbox", '
                 '"reason": "captcha"}', 0),
                # Step one click.
                ('{"clickable": true, "x": 0.5, "y": 0.5, '
                 '"reason": "click"}', 0),
                # Step two look.
                ('{"has_captcha": true, "captcha_type": "checkbox", '
                 '"reason": "captcha"}', 0),
                # Step two click.
                ('{"clickable": true, "x": 0.5, "y": 0.5, '
                 '"reason": "click"}', 0),
                # Step three look.
                ('{"has_captcha": true, "captcha_type": "checkbox", '
                 '"reason": "captcha"}', 0),
                # Step three click.
                ('{"clickable": true, "x": 0.5, "y": 0.5, '
                 '"reason": "click"}', 0),
            ]

            result = await solve_captcha(mgr, "tab1", max_steps=3)

        assert result is False
        # Two pictures for each step.
        assert mgr.screenshot.call_count == 6

    @pytest.mark.fast_sleeps
    async def test_solve_captcha_coordinate_conversion(self) -> None:
        """A point given as a fraction turns into a spot on the picture."""
        mgr = _make_manager()

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.side_effect = [
                # The look finds a checkbox.
                ('{"has_captcha": true, "captcha_type": "checkbox", '
                 '"reason": "captcha"}', 0),
                # The click goes to the middle.
                ('{"clickable": true, "x": 0.5, "y": 0.5, '
                 '"reason": "click checkbox"}', 0),
                # The next look finds nothing.
                ('{"has_captcha": false, "captcha_type": "unknown", '
                 '"reason": "solved"}', 0),
            ]

            result = await solve_captcha(mgr, "tab1")

        assert result is True
        call_args = mgr.click_at.call_args[0]
        px, py = call_args[1], call_args[2]
        # Half of a picture 800 wide and 600 tall is 400 across and 300 down.
        assert px == 400, f"Expected px=400, got {px}"
        assert py == 300, f"Expected py=300, got {py}"

    @pytest.mark.fast_sleeps
    async def test_solve_captcha_checkbox_not_clickable(self) -> None:
        """A checkbox that cannot be clicked is not solved."""
        mgr = _make_manager()

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.side_effect = [
                # The look finds a checkbox.
                ('{"has_captcha": true, "captcha_type": "checkbox", '
                 '"reason": "captcha"}', 0),
                # The checkbox cannot be clicked.
                ('{"clickable": false, "x": 0, "y": 0, '
                 '"reason": "already checked"}', 0),
            ]

            result = await solve_captcha(mgr, "tab1", max_steps=1)

        assert result is False

    async def test_solve_captcha_detect_error(self) -> None:
        """A failed model call leaves the captcha unsolved."""
        mgr = _make_manager()

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.return_value = ("error", 1)

            result = await solve_captcha(mgr, "tab1")

        assert result is False
        assert mgr.screenshot.call_count == 1
        assert mgr.click_at.call_count == 0

    async def test_solve_captcha_screenshot_failure(self) -> None:
        """A failed screenshot leaves the captcha unsolved."""
        mgr = _make_manager()
        mgr.screenshot.side_effect = None
        mgr.screenshot.return_value = None

        result = await solve_captcha(mgr, "tab1")

        assert result is False

    @pytest.mark.fast_sleeps
    async def test_solve_captcha_click_at_returns_false(self) -> None:
        """A click that fails leaves the captcha unsolved."""
        mgr = _make_manager()
        mgr.click_at.return_value = False

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.side_effect = [
                # Step one look.
                ('{"has_captcha": true, "captcha_type": "checkbox", '
                 '"reason": "captcha"}', 0),
                # Step one click.
                ('{"clickable": true, "x": 0.5, "y": 0.5, '
                 '"reason": "click"}', 0),
                # Step two look.
                ('{"has_captcha": true, "captcha_type": "checkbox", '
                 '"reason": "captcha"}', 0),
                # Step two click.
                ('{"clickable": true, "x": 0.5, "y": 0.5, '
                 '"reason": "click"}', 0),
            ]

            result = await solve_captcha(mgr, "tab1", max_steps=2)

        assert result is False

    async def test_solve_captcha_json_parse_failure(self) -> None:
        """A reply that is not valid JSON leaves the captcha unsolved."""
        mgr = _make_manager()

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.return_value = ("not valid json {{", 0)

            result = await solve_captcha(mgr, "tab1")

        assert result is False

    async def test_solve_captcha_validate_missing_field(self) -> None:
        """A reply missing a needed field leaves the captcha unsolved."""
        mgr = _make_manager()

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.return_value = (
                '{"has_captcha": true}',
                0,
            )

            result = await solve_captcha(mgr, "tab1")

        assert result is False

    @pytest.mark.fast_sleeps
    async def test_solve_captcha_get_dims_pil_fallback(self) -> None:
        """When the picture cannot be read, the tab size sets where the click lands."""
        mgr = _make_manager()
        mgr.get_page_viewport.return_value = _viewport(1024, 768)

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.side_effect = [
                ('{"has_captcha": true, "captcha_type": "checkbox", '
                 '"reason": "captcha"}', 0),
                ('{"clickable": true, "x": 0.5, "y": 0.5, '
                 '"reason": "click"}', 0),
                ('{"has_captcha": false, "captcha_type": "unknown", '
                 '"reason": "solved"}', 0),
            ]

            with patch("app.lib.browser.captcha_solver.PilImage.open", side_effect=Exception("corrupt")):
                result = await solve_captcha(mgr, "tab1")

        assert result is True
        call_args = mgr.click_at.call_args[0]
        px, py = call_args[1], call_args[2]
        assert px == 512, f"Expected px=512 (1024*0.5), got {px}"
        assert py == 384, f"Expected py=384 (768*0.5), got {py}"

    @pytest.mark.fast_sleeps
    async def test_solve_captcha_image_grid_dispatch(self) -> None:
        """A picture grid captcha is not solved."""
        mgr = _make_manager()

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.return_value = (
                '{"has_captcha": true, "captcha_type": "image_grid", '
                '"reason": "grid captcha"}',
                0,
            )

            result = await solve_captcha(mgr, "tab1", max_steps=1)

        assert result is False
        assert mgr.click_at.call_count == 0

    @pytest.mark.fast_sleeps
    async def test_solve_captcha_image_grid_absent_tab_does_not_raise(self) -> None:
        """A picture grid captcha on a tab that is gone does not blow up the solve."""
        mgr = _make_manager()
        mgr.tab_ids = MagicMock(return_value=["other"])
        mgr.evaluate = AsyncMock(side_effect=KeyError("No page found for target_id: tab1"))

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.return_value = (
                '{"has_captcha": true, "captcha_type": "image_grid", '
                '"reason": "grid captcha"}',
                0,
            )

            result = await solve_captcha(mgr, "tab1", max_steps=1)

        assert result is False
        mgr.evaluate.assert_not_awaited()

    @pytest.mark.fast_sleeps
    async def test_solve_captcha_text_input_dispatch(self) -> None:
        """A typed answer captcha is not solved."""
        mgr = _make_manager()

        with patch("app.lib.browser.captcha_solver.reason") as mock_reason:
            mock_reason.return_value = (
                '{"has_captcha": true, "captcha_type": "text_input", '
                '"reason": "text captcha"}',
                0,
            )

            result = await solve_captcha(mgr, "tab1", max_steps=1)

        assert result is False
        assert mgr.click_at.call_count == 0


@pytest.mark.live
@skip_if_no_chrome
class TestSolveCaptchaLive:
    """Live test that solves a real captcha in a real browser."""

    async def test_solve_captcha_recaptcha_demo(self, browser_manager: BrowserManager) -> None:
        """A real checkbox captcha gets solved."""
        tid = await browser_manager.new_page()
        try:
            await browser_manager.navigate(tid, "https://www.google.com/recaptcha/api2/demo", wait_until="networkidle", timeout=30000)
            result = await captcha_solver.solve_captcha(browser_manager, tid, max_steps=5, verbose=True)
            assert result is True, "solve_captcha should return True when CAPTCHA is solved"
        finally:
            await browser_manager.close_page(tid)
