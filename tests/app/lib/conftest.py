"""Shared fixtures for local-task tests."""
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from app.lib.browser.browser_manager import BrowserManager
from app.lib.browser.page_snapshot import PageSnapshot
from tests.app.conftest import is_chrome_cdp_available

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def mock_cli_proc(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Build a stand in for a child process.

    Args:
        stdout: Text the process gives back on stdout.
        stderr: Text the process gives back on stderr.
        returncode: Exit code the process reports.

    Returns:
        A stand in for a child process.
    """
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout.encode(), stderr.encode()))
    proc.wait = AsyncMock(return_value=returncode)
    return proc


@pytest.fixture(scope="module")
def data_dir() -> str:
    """Path to the shared test data directory."""
    return _DATA_DIR


@pytest.fixture(scope="module")
def xiaohongshu_html() -> str:
    """Path to a saved Xiaohongshu search page."""
    path = os.path.join(_DATA_DIR, 'xiaohongshu_search.html')
    assert os.path.isfile(path), f"Missing test resource: {path}"
    return path


@pytest.fixture(scope="module")
def rednote_dump_html() -> str:
    """Path to a saved rednote page."""
    path = os.path.join(_DATA_DIR, 'rednote_dump.html')
    assert os.path.isfile(path), f"Missing test resource: {path}"
    return path


@pytest.fixture
def dummy_image_path() -> str:
    """Make a 32 by 32 red square image and give back its path."""
    path = os.path.join(_DATA_DIR, 'test_image.jpg')
    os.makedirs(_DATA_DIR, exist_ok=True)
    img = Image.new('RGB', (32, 32), color='red')
    img.save(path)
    return path


@pytest.fixture
def dummy_pdf_path() -> str:
    """Make a small PDF file and give back its path."""
    path = os.path.join(_DATA_DIR, 'test_doc.pdf')
    if not os.path.exists(path):
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(path, "wb") as f:
            f.write(_DUMMY_PDF)
    return path


@pytest.fixture
def dummy_text_path() -> str:
    """Make a small text file and give back its path."""
    path = os.path.join(_DATA_DIR, 'test_file.txt')
    if not os.path.exists(path):
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(path, "w") as f:
            f.write("This is a test file content.")
    return path


@pytest.fixture
def dummy_html_path() -> str:
    """Make a small HTML file and give back its path."""
    path = os.path.join(_DATA_DIR, 'test.html')
    if not os.path.exists(path):
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(path, "w") as f:
            f.write("<html><body><h1>Title</h1><p>Paragraph content.</p></body></html>")
    return path


@pytest.fixture
def dummy_video_path() -> str:
    """Make a small MP4 video file and give back its path."""
    path = os.path.join(_DATA_DIR, 'test_video.mp4')
    if not os.path.exists(path):
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(path, "wb") as f:
            f.write(_DUMMY_MP4)
    return path


@pytest.fixture
def dummy_audio_path() -> str:
    """Make a small WAV audio file and give back its path."""
    path = os.path.join(_DATA_DIR, 'test_audio.wav')
    if not os.path.exists(path):
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    return path


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "live: mark test to run only with real Ollama server"
    )
    config.addinivalue_line(
        "markers", "timeout(seconds): mark test with a timeout to prevent hanging"
    )


@pytest.fixture
async def browser_manager() -> BrowserManager:
    """Make a browser manager for one test.

    The test is skipped when Chrome cannot be reached.
    """
    if not is_chrome_cdp_available():
        pytest.skip("Chrome CDP not available at localhost:9222")
    mgr = await BrowserManager.create()
    yield mgr
    await mgr.cleanup()


# --- Page snapshot fixtures ---


@pytest.fixture
def mock_browser_mgr() -> AsyncMock:
    """Return a stand in for a browser manager."""
    return AsyncMock()


@pytest.fixture
async def page_snapshot_simple(mock_browser_mgr: AsyncMock) -> PageSnapshot:
    """A page snapshot of a tiny valid page."""
    return await PageSnapshot.create("<html><body><h1>Title</h1><p>Content</p></body></html>", mock_browser_mgr, "tab1")


@pytest.fixture
async def page_snapshot_button(mock_browser_mgr: AsyncMock) -> PageSnapshot:
    """A page snapshot of a page with one button."""
    return await PageSnapshot.create("<html><body><button>Click</button></body></html>", mock_browser_mgr, "tab1")


@pytest.fixture
async def page_snapshot_interactive(mock_browser_mgr: AsyncMock) -> PageSnapshot:
    """A page snapshot of a page with several things to click or type in."""
    html = """<html><body>
        <h1>Heading</h1>
        <p>Paragraph</p>
        <button>Click</button>
        <a href='/'>Link</a>
        <input type='text' />
    </body></html>"""
    return await PageSnapshot.create(html, mock_browser_mgr, "tab1")


@pytest.fixture
async def page_snapshot_empty(mock_browser_mgr: AsyncMock) -> PageSnapshot:
    """A page snapshot of an empty page."""
    return await PageSnapshot.create("", mock_browser_mgr, "tab1")


@pytest.fixture
async def page_snapshot_filter_html(mock_browser_mgr: AsyncMock) -> PageSnapshot:
    """A page snapshot of a page with a nav, a main part, and a footer."""
    html = """<html><body>
        <nav>Navigation links</nav>
        <main><h1>Important Content</h1></main>
        <footer>Footer info</footer>
    </body></html>"""
    return await PageSnapshot.create(html, mock_browser_mgr, "tab1")