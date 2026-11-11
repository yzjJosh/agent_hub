import pytest
import pytest_asyncio
from patchright.async_api import BrowserContext as AsyncBrowserContext
from patchright.async_api import Page as AsyncPage
from patchright.async_api import Playwright as AsyncPlaywright
from patchright.async_api import async_playwright
from patchright.sync_api import BrowserContext
from patchright.sync_api import Page
from patchright.sync_api import Playwright
from patchright.sync_api import sync_playwright

from app.lib.recognizer.components.detector import Detector


@pytest.fixture
def detector() -> Detector:
    """Make a detector."""
    det = Detector()
    return det


@pytest.fixture
def sync_playwright_object() -> Playwright:
    """Give a browser driver."""
    with sync_playwright() as playwright_object:
        yield playwright_object


@pytest.fixture
def sync_browser(sync_playwright_object: Playwright) -> BrowserContext:
    """Open a Chrome browser that keeps its data between runs."""
    browser = sync_playwright_object.chromium.launch_persistent_context(user_data_dir="./user_data", channel="chrome", headless=True, no_viewport=True, locale="en-US")

    yield browser
    browser.close()


@pytest.fixture
def sync_page(sync_browser: BrowserContext) -> Page:
    """Open a new tab."""
    page = sync_browser.new_page()

    yield page
    page.close()


@pytest_asyncio.fixture
async def async_playwright_object() -> AsyncPlaywright:
    """Give a browser driver for async tests."""
    async with async_playwright() as playwright_object:
        yield playwright_object


@pytest_asyncio.fixture
async def async_browser(async_playwright_object: AsyncPlaywright) -> AsyncBrowserContext:
    """Open a Chrome browser for async tests that keeps its data between runs."""
    browser = await async_playwright_object.chromium.launch_persistent_context(user_data_dir="./user_data", channel="chrome", headless=True, no_viewport=True, locale="en-US")

    yield browser
    await browser.close()


@pytest_asyncio.fixture
async def async_page(async_browser: AsyncBrowserContext) -> AsyncPage:
    """Open a new tab for async tests."""
    page = await async_browser.new_page()

    yield page
    await page.close()
