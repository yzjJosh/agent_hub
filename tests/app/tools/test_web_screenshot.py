"""Tests for app.tools.web_screenshot."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.lib.shared.paths import resources_dir
from app.tools.web_screenshot import web_screenshot_tool
from tests.app.conftest import (
    assert_browser_closed,
    skip_if_no_chrome,
    wire_context_manager,
)
from tests.app.lib.tool.conftest import run_text

_DAV_BASE_URL = "http://dav.test"
_TEST_URL = "https://example.com/page"
_JPEG_MAGIC = b"\xff\xd8\xff"
_SCREENSHOT_BYTES = b"\xff\xd8\xffscreenshot pixel bytes"
_NAVIGATION_TIMEOUT_MS = 30_000
_RENDER_WAIT_TIMEOUT_MS = 5_000


def _point_resources_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the resources folder at a temp folder and set a base URL."""
    monkeypatch.setenv("RESOURCES_DIR", str(tmp_path))
    monkeypatch.setenv("DAV_BASE_URL", _DAV_BASE_URL)


def _save_screenshot(_target_id: str, path: str, **_kwargs: bool) -> str:
    """Write fake JPEG bytes to the given path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(_SCREENSHOT_BYTES)
    return path


def _wire_manager(
    mock_mgr_class: MagicMock, *, navigate_ok: bool = True, screenshot_ok: bool = True
) -> AsyncMock:
    """Set up a fake browser manager and give it back."""
    mgr = AsyncMock()
    mgr.new_page.return_value = "tab1"
    mgr.navigate.return_value = navigate_ok
    mgr.wait_for_load.return_value = True
    if screenshot_ok:
        mgr.screenshot.side_effect = _save_screenshot
    else:
        mgr.screenshot.return_value = None
    wire_context_manager(mock_mgr_class, mgr)
    return mgr


async def _run_tool(mock_mgr_class: MagicMock, url: str, **kwargs: bool) -> str:
    """Run the tool with a fake browser and give back the result."""
    with patch("app.tools.web_screenshot.BrowserManager", mock_mgr_class):
        return await run_text(web_screenshot_tool, url, **kwargs)


def _assert_saved_result(result: str, tmp_path: Path) -> None:
    """Check the file saved in the screenshots folder and the exact success message."""
    saved = list((tmp_path / "screenshots").glob("screenshot_*.jpg"))
    assert len(saved) == 1, f"expected exactly one saved screenshot, got {saved}"
    file_path = saved[0]
    assert file_path.read_bytes() == _SCREENSHOT_BYTES
    resource_path = f"screenshots/{file_path.name}"
    assert result == (
        f"Screenshot saved to resource path: {resource_path}\n"
        "Download it with:\n"
        f"  curl -o /tmp/{file_path.name} {_DAV_BASE_URL}/dav/{resource_path}\n"
        "See the send_or_receive_file tool for file transfer details."
    )


class TestWebScreenshot:

    async def test_success_saves_screenshot_and_returns_download_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A good capture saves a JPEG in the screenshots folder and gives back the download message."""
        _point_resources_at(tmp_path, monkeypatch)
        mock_mgr_class = MagicMock()
        mgr = _wire_manager(mock_mgr_class)

        result = await _run_tool(mock_mgr_class, _TEST_URL)

        _assert_saved_result(result, tmp_path)
        assert mgr.screenshot.await_args.kwargs["full_page"] is True
        assert_browser_closed(mock_mgr_class)

    async def test_render_wait_sequence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The capture waits 30 seconds for the page, then 5 seconds each for load and network."""
        _point_resources_at(tmp_path, monkeypatch)
        mock_mgr_class = MagicMock()
        mgr = _wire_manager(mock_mgr_class)

        await _run_tool(mock_mgr_class, _TEST_URL)

        assert mgr.navigate.call_args.kwargs["wait_until"] == "domcontentloaded"
        assert mgr.navigate.call_args.kwargs["timeout"] == _NAVIGATION_TIMEOUT_MS
        wait_states = [call.kwargs["state"] for call in mgr.wait_for_load.call_args_list]
        assert wait_states == ["load", "networkidle"]
        assert all(
            call.kwargs["timeout"] == _RENDER_WAIT_TIMEOUT_MS
            for call in mgr.wait_for_load.call_args_list
        )
        assert_browser_closed(mock_mgr_class)

    async def test_full_page_false_passthrough(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With full_page off, the capture still saves a file."""
        _point_resources_at(tmp_path, monkeypatch)
        mock_mgr_class = MagicMock()
        mgr = _wire_manager(mock_mgr_class)

        result = await _run_tool(mock_mgr_class, _TEST_URL, full_page=False)

        _assert_saved_result(result, tmp_path)
        assert mgr.screenshot.await_args.kwargs["full_page"] is False
        assert_browser_closed(mock_mgr_class)

    async def test_navigation_failure_reports_error_without_saving(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A page that will not load gives an error and saves no file."""
        _point_resources_at(tmp_path, monkeypatch)
        mock_mgr_class = MagicMock()
        _wire_manager(mock_mgr_class, navigate_ok=False)

        result = await _run_tool(mock_mgr_class, _TEST_URL)

        assert result == (
            f"Tool 'web_screenshot' failed: "
            f"Screenshot failed: could not navigate to {_TEST_URL}"
        )
        assert list((tmp_path / "screenshots").glob("screenshot_*.jpg")) == []
        assert_browser_closed(mock_mgr_class)

    async def test_screenshot_capture_failure_reports_error_without_saving(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A capture that comes back empty gives an error and saves no file."""
        _point_resources_at(tmp_path, monkeypatch)
        mock_mgr_class = MagicMock()
        _wire_manager(mock_mgr_class, screenshot_ok=False)

        result = await _run_tool(mock_mgr_class, _TEST_URL)

        assert result == (
            f"Tool 'web_screenshot' failed: "
            f"Screenshot failed: screenshot capture failed for {_TEST_URL}"
        )
        assert list((tmp_path / "screenshots").glob("screenshot_*.jpg")) == []
        assert_browser_closed(mock_mgr_class)


class TestUrlValidation:

    @pytest.mark.parametrize(
        ("bad_url", "fragment"),
        [
            ("", "missing a scheme"),
            ("example.com/page", "missing a scheme"),
            ("http://", "missing a hostname"),
        ],
    )
    async def test_invalid_url_rejected_without_browser(
        self, bad_url: str, fragment: str
    ) -> None:
        """A URL with no scheme or no host is turned down before the browser opens."""
        mock_mgr_class = MagicMock()
        with patch("app.tools.web_screenshot.BrowserManager", mock_mgr_class):
            result = await run_text(web_screenshot_tool, bad_url)

        assert result == (
            f"Tool 'web_screenshot' failed: Invalid URL: '{bad_url}' is {fragment}"
        )
        assert mock_mgr_class.call_count == 0


class TestParamDocs:
    """Tests for what the tool and its params say."""

    def test_full_page_param_documents_default_and_viewport(self) -> None:
        """The full_page base mentions the viewport, and the wire description gives the default."""
        p = next(p for p in web_screenshot_tool.params if p.name == "full_page")
        assert "viewport" in p.description
        assert "Optional (default: True)." in p.wire_description

    def test_description_states_purpose_and_when_to_use(self) -> None:
        """The tool description says it takes a screenshot of a webpage and shows rendered content."""
        desc = web_screenshot_tool.description
        assert "screenshot of a webpage" in desc
        assert "rendered content" in desc


def _assert_live_screenshot(result: str) -> None:
    """Check the saved screenshot is a real JPEG, then delete it."""
    assert result.startswith("Screenshot saved to resource path: ")
    resource_path = result.split("resource path: ", 1)[1].split("\n", 1)[0]
    saved = Path(resources_dir()) / resource_path
    try:
        assert saved.exists()
        assert saved.read_bytes().startswith(_JPEG_MAGIC)
    finally:
        saved.unlink(missing_ok=True)


@skip_if_no_chrome
@pytest.mark.live
class TestWebScreenshotLive:
    """Live screenshot tests."""

    @pytest.mark.timeout(120)
    async def test_live_screenshot_of_test_page(self, test_server_url: str) -> None:
        """A live capture saves a real JPEG and reports its path."""
        result = await run_text(web_screenshot_tool, test_server_url)
        _assert_live_screenshot(result)

    @pytest.mark.timeout(120)
    async def test_live_screenshot_viewport(self, test_server_url: str) -> None:
        """With full_page off, a live capture still saves a real JPEG."""
        result = await run_text(web_screenshot_tool, test_server_url, full_page=False)
        _assert_live_screenshot(result)
