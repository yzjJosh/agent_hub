"""Tests for the shared captcha challenger and its browser actions."""
from __future__ import annotations

from typing import override

from unittest.mock import MagicMock, patch

import pytest

from app.lib.recognizer.agents.playwright.base_challenger import (
    BaseChallenger,
    BrowserOps,
)


def _create_page() -> MagicMock:
    """Make a fake page."""
    return MagicMock()


def _advance_retry_n_times(challenger: BaseChallenger, n: int) -> None:
    """Take n tries on the challenger."""
    for _ in range(n):
        challenger.advance_retry()


def _make_challenger(retry_times: int = 15, verbose: bool = False) -> BaseChallenger:
    """Make a challenger with a fake page."""
    with patch("app.lib.recognizer.agents.playwright.base_challenger.Detector"):
        return BaseChallenger(
            _create_page(),
            retry_times=retry_times,
            verbose=verbose,
        )


class TestInit:
    """Tests for making a challenger."""

    def test_sets_page(self) -> None:
        """A new challenger holds the page it was given."""
        page = _create_page()
        with patch("app.lib.recognizer.agents.playwright.base_challenger.Detector"):
            challenger = BaseChallenger(page)
        assert challenger.page is page

    def test_initializes_is_routed_to_false(self) -> None:
        """A new challenger is not watching traffic yet."""
        challenger = _make_challenger()
        assert challenger.is_routed is False

    def test_initializes_click_timeout_to_none(self) -> None:
        """A new challenger has no click time limit by default."""
        challenger = _make_challenger()
        assert challenger.click_timeout is None

    def test_sets_custom_click_timeout(self) -> None:
        """A given click time limit is kept."""
        with patch("app.lib.recognizer.agents.playwright.base_challenger.Detector"):
            challenger = BaseChallenger(_create_page(), click_timeout=500)
        assert challenger.click_timeout == 500

    def test_initializes_captcha_token_to_none(self) -> None:
        """A new challenger has no captcha token yet."""
        challenger = _make_challenger()
        assert challenger.captcha_token is None

    def test_passes_optimize_click_order_to_detector(self) -> None:
        """Click order can be turned off when making a challenger."""
        with patch(
            "app.lib.recognizer.agents.playwright.base_challenger.Detector"
        ) as mock_detector_cls:
            BaseChallenger(_create_page(), optimize_click_order=False)
            mock_detector_cls.assert_called_once_with(optimize_click_order=False)

    def test_detector_default_optimize_true(self) -> None:
        """Click order is turned on by default."""
        with patch(
            "app.lib.recognizer.agents.playwright.base_challenger.Detector"
        ) as mock_detector_cls:
            BaseChallenger(_create_page())
            mock_detector_cls.assert_called_once_with(optimize_click_order=True)


class TestAdvanceRetry:
    """Tests for counting tries."""

    def test_increments_without_raising_below_limit(self) -> None:
        """Tries under the limit are fine."""
        challenger = _make_challenger(retry_times=5)
        _advance_retry_n_times(challenger, 3)

    def test_raises_on_exceeding_retry_times(self) -> None:
        """Going past the limit raises an error."""
        challenger = _make_challenger(retry_times=3)
        _advance_retry_n_times(challenger, 2)
        with pytest.raises(RuntimeError, match="Exceeded maximum retry times of 3"):
            challenger.advance_retry()

    def test_error_message_contains_retry_times(self) -> None:
        """The error says what the limit was."""
        challenger = _make_challenger(retry_times=10)
        _advance_retry_n_times(challenger, 9)
        with pytest.raises(RuntimeError, match="Exceeded maximum retry times of 10"):
            challenger.advance_retry()

    def test_allows_retries_up_to_limit(self) -> None:
        """One try less than the limit is fine."""
        challenger = _make_challenger(retry_times=5)
        _advance_retry_n_times(challenger, 4)


class TestParseRouteResponse:
    """Tests for reading a captcha reply."""

    def test_detects_dynamic_in_response(self) -> None:
        """A reply that says dynamic is read without error."""
        challenger = _make_challenger()
        challenger.parse_route_response('{"type": "dynamic"}', "https://example.com")

    def test_no_dynamic_in_response(self) -> None:
        """A plain reply is read without error."""
        challenger = _make_challenger()
        challenger.parse_route_response('{"type": "normal"}', "https://example.com")

    def test_extracts_uvresp_token_from_userverify(self) -> None:
        """A token in a verify reply is picked up."""
        challenger = _make_challenger()
        challenger.parse_route_response(
            '{"uvresp", "token-abc123"}',
            "https://www.google.com/recaptcha/userverify",
        )
        assert challenger.captcha_token == "token-abc123"

    def test_skips_extract_when_rresp_present(self) -> None:
        """A verify reply that also holds rresp gives no token."""
        challenger = _make_challenger()
        challenger.parse_route_response(
            '{"rresp": "123", "uvresp", "should-not-extract"}',
            "https://www.google.com/recaptcha/userverify",
        )
        assert challenger.captcha_token is None

    def test_skips_extract_when_bgdata_present(self) -> None:
        """A verify reply that also holds bgdata gives no token."""
        challenger = _make_challenger()
        challenger.parse_route_response(
            '{"bgdata": "xyz", "uvresp", "should-not-extract"}',
            "https://www.google.com/recaptcha/userverify",
        )
        assert challenger.captcha_token is None

    def test_skips_extract_when_no_userverify_url(self) -> None:
        """A reply from another address gives no token."""
        challenger = _make_challenger()
        challenger.parse_route_response(
            '{"uvresp", "token"}',
            "https://www.google.com/recaptcha/other",
        )
        assert challenger.captcha_token is None

    def test_no_token_when_uvresp_missing(self) -> None:
        """A reply with no token field gives no token."""
        challenger = _make_challenger()
        challenger.parse_route_response(
            '{"someother": "value"}',
            "https://www.google.com/recaptcha/userverify",
        )
        assert challenger.captcha_token is None

    def test_token_with_various_spacing(self) -> None:
        """Extra spaces around the comma still give the token."""
        challenger = _make_challenger()
        challenger.parse_route_response(
            '{"uvresp"  ,  "compact-token"}',
            "https://www.google.com/recaptcha/userverify",
        )
        assert challenger.captcha_token == "compact-token"

    def test_verbose_prints_dynamic_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        """With logging on, the reply is printed."""
        challenger = _make_challenger(verbose=True)
        challenger.parse_route_response(
            '{"type": "dynamic"}',
            "https://example.com",
        )
        captured = capsys.readouterr()
        assert "dynamic=True" in captured.out


class TestBrowserOpsAbstract:
    """Tests for the browser action base class."""

    def test_is_abstract(self) -> None:
        """The base class cannot be made on its own."""
        with pytest.raises(TypeError, match="abstract"):
            BrowserOps(_create_page())

    def test_concrete_subclass_instantiates(self) -> None:
        """A full subclass can be made and holds its page."""
        page = _create_page()
        ops = _ConcreteOps(page)
        assert ops.page is page


class _ConcreteOps(BrowserOps):
    """A full set of browser actions for tests."""

    @override
    def wait_for_load_state(self, _state: str, _timeout: int) -> None:
        """Wait until the page reaches a state."""
        pass

    @override
    def screenshot(self, _timeout: int = 0) -> bytes:
        """Take a picture of the page."""
        return b""

    @override
    def wait_for_timeout(self, _timeout: int) -> None:
        """Wait a number of milliseconds."""
        pass

    @override
    def mouse_click(self, _x: int, _y: int) -> None:
        """Click a spot on the page."""
        pass

    @override
    def page_evaluate(self, _expr: str) -> str:
        """Run code on the page and give back the result."""
        return ""

    @override
    def label_wait_for[T](
        self, _label_obj: T, _state: str, _timeout: int
    ) -> None:
        """Wait until a label reaches a state."""
        pass

    @override
    def label_is_visible[T](self, _label_obj: T) -> bool:
        """Say if a label shows."""
        return True

    @override
    def checkbox_click(self) -> None:
        """Click the captcha checkbox."""
        pass

    @override
    def label_text_content[T](self, _label_obj: T, _timeout: int) -> str | None:
        """Give back the text of a label."""
        return None

    @override
    def tiles_all[T](self, _tiles_locator: T) -> list:
        """Give back all the tiles."""
        return []

    @override
    def tile_is_visible[T](self, _tile: T) -> bool:
        """Say if one tile shows."""
        return False

    @override
    def submit_click[T](self, _submit_locator: T) -> None:
        """Click the submit button."""
        pass