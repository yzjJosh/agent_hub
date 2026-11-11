"""Tests for the sync captcha challenger."""
from unittest.mock import MagicMock, patch

import pytest

from app.lib.recognizer.agents.playwright.sync_control import SyncChallenger


@pytest.fixture
def mock_page() -> MagicMock:
    """Make a fake page."""
    page = MagicMock()
    page.frame_locator = MagicMock(return_value=MagicMock())
    page.route = MagicMock()
    page.wait_for_timeout = MagicMock()
    return page


@pytest.fixture
def challenger(mock_page: MagicMock) -> SyncChallenger:
    """Make a challenger with a fake page."""
    with patch("app.lib.recognizer.agents.playwright.base_challenger.Detector"):
        return SyncChallenger(mock_page, click_timeout=1000, retry_times=3, verbose=False)


def test_initial_state(challenger: SyncChallenger) -> None:
    """A new challenger is not watching traffic yet."""
    assert challenger.is_routed is False


def test_route_handler_parses_response(challenger: SyncChallenger) -> None:
    """A captcha request is fetched and passed on."""
    mock_route = MagicMock()
    mock_response = MagicMock()
    mock_response.text.return_value = '{"isDynamic":false}'
    mock_route.fetch.return_value = mock_response
    mock_route.fulfill = MagicMock()

    mock_request = MagicMock()
    mock_request.url = "https://www.google.com/recaptcha/"

    challenger.route_handler(mock_route, mock_request)

    mock_route.fetch.assert_called_once()
    mock_route.fulfill.assert_called_once()


def test_route_handler_raises_on_empty(challenger: SyncChallenger) -> None:
    """An empty reply raises an error."""
    mock_route = MagicMock()
    mock_response = MagicMock()
    mock_response.text.return_value = ""
    mock_route.fetch.return_value = mock_response

    mock_request = MagicMock()
    mock_request.url = "https://www.google.com/recaptcha/"

    with pytest.raises(RuntimeError, match="Empty response"):
        challenger.route_handler(mock_route, mock_request)