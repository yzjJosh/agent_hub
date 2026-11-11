"""Tests for the async captcha challenger."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.lib.recognizer.agents.playwright.async_control import AsyncChallenger


@pytest.fixture
async def mock_page() -> MagicMock:
    """Make a fake page."""
    page = AsyncMock()
    page.frame_locator = MagicMock(return_value=MagicMock())
    page.route = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    return page


@pytest.fixture
async def challenger(mock_page: MagicMock) -> AsyncChallenger:
    """Make a challenger with a fake page."""
    with patch("app.lib.recognizer.agents.playwright.base_challenger.Detector"):
        return AsyncChallenger(mock_page, click_timeout=1000, retry_times=3, verbose=False)


@pytest.mark.asyncio
async def test_initial_state(challenger: AsyncChallenger) -> None:
    """A new challenger is not watching traffic yet."""
    assert challenger.is_routed is False


@pytest.mark.asyncio
async def test_route_handler_parses_response(challenger: AsyncChallenger) -> None:
    """A captcha request is fetched and passed on."""
    mock_route = AsyncMock()
    mock_response = AsyncMock()
    mock_response.text = AsyncMock(return_value='{"isDynamic":false}')
    mock_route.fetch = AsyncMock(return_value=mock_response)
    mock_route.fulfill = AsyncMock()

    mock_request = MagicMock()
    mock_request.url = "https://www.google.com/recaptcha/"

    await challenger.route_handler(mock_route, mock_request)

    mock_route.fetch.assert_called_once()
    mock_route.fulfill.assert_called_once()


@pytest.mark.asyncio
async def test_route_handler_raises_on_empty(challenger: AsyncChallenger) -> None:
    """An empty reply raises an error."""
    mock_route = AsyncMock()
    mock_response = AsyncMock()
    mock_response.text = AsyncMock(return_value="")
    mock_route.fetch = AsyncMock(return_value=mock_response)

    mock_request = MagicMock()
    mock_request.url = "https://www.google.com/recaptcha/"

    with pytest.raises(RuntimeError, match="Empty response"):
        await challenger.route_handler(mock_route, mock_request)