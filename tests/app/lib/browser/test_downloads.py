"""Unit tests for browser tab download capture and finalization."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.lib.browser.downloads import (
    DownloadResult,
    DownloadTriggered,
    wait_for_page_download,
)

_SHORT_TIMEOUT_S = 0.2


def _build_session() -> MagicMock:
    """Create a fake CDP session that dispatches registered download listeners."""
    session = MagicMock()
    listeners: dict[str, list[Callable[[dict], None]]] = {}
    session.listeners = listeners
    session.send = AsyncMock(return_value={})
    session.on = MagicMock(side_effect=lambda event, handler: listeners.setdefault(event, []).append(handler))
    session.remove_listener = MagicMock(
        side_effect=lambda event, handler: (
            listeners[event].remove(handler) if handler in listeners.get(event, []) else None
        )
    )

    def _emit(event: str, payload: dict) -> None:
        for handler in list(listeners.get(event, [])):
            handler(payload)

    session.emit = _emit
    return session


class TestDownloadTypes:
    """Tests for public exception and result types exported by downloads."""

    def test_download_triggered_is_exception_with_message(self) -> None:
        """DownloadTriggered carries the trigger reason as an Exception."""
        error = DownloadTriggered("navigation started download")
        assert isinstance(error, Exception)
        assert str(error) == "navigation started download"

    def test_download_result_shape(self) -> None:
        """DownloadResult accepts both empty failure and populated success dicts."""
        empty: DownloadResult = {"downloaded": False}
        complete: DownloadResult = {
            "downloaded": True,
            "path": "/tmp/report.pdf",
            "filename": "report.pdf",
            "size": 42,
        }
        assert empty["downloaded"] is False
        assert complete["filename"] == "report.pdf"
        assert complete["size"] == 42


class TestWaitForPageDownload:
    """Tests for capturing downloaded files via CDP events or folder fallback."""

    async def test_claims_file_named_by_completion_event_and_moves_to_save_path(self, tmp_path: Path) -> None:
        """A tracked completion event claims its file, moves it, and detaches listeners."""
        session = _build_session()
        downloaded_file = tmp_path / "report.csv"
        save_path = tmp_path / "dest" / "final.csv"

        async def _trigger() -> bool:
            downloaded_file.write_bytes(b"col1,col2\n1,2\n")
            session.emit("Browser.downloadWillBegin", {"guid": "dl-1"})
            session.emit(
                "Browser.downloadProgress",
                {"guid": "dl-1", "state": "completed", "filePath": str(downloaded_file)},
            )
            return True

        result = await wait_for_page_download(
            session,
            str(tmp_path),
            _trigger,
            save_path=str(save_path),
            timeout=5,
        )

        assert result == {
            "downloaded": True,
            "path": str(save_path),
            "filename": "report.csv",
            "size": 14,
        }
        assert save_path.read_bytes() == b"col1,col2\n1,2\n"
        assert not downloaded_file.exists()
        assert session.listeners.get("Browser.downloadWillBegin", []) == []
        assert session.listeners.get("Browser.downloadProgress", []) == []

    async def test_returns_not_downloaded_when_session_is_none(self, tmp_path: Path) -> None:
        """A missing session returns downloaded False without running the action."""
        action = AsyncMock(return_value=True)

        result = await wait_for_page_download(None, str(tmp_path), action, timeout=5)

        assert result == {"downloaded": False}
        action.assert_not_awaited()

    async def test_returns_not_downloaded_when_action_does_not_trigger(self, tmp_path: Path) -> None:
        """When the action reports False, no wait occurs and downloaded is False."""
        session = _build_session()
        action = AsyncMock(return_value=False)

        result = await wait_for_page_download(session, str(tmp_path), action, timeout=5)

        assert result == {"downloaded": False}
        action.assert_awaited_once()

    async def test_timeout_fallback_claims_single_new_file_and_ignores_crdownload(self, tmp_path: Path) -> None:
        """Without a CDP event, a single completed file is claimed while .crdownload files are ignored."""
        session = _build_session()
        (tmp_path / "existing.bin").write_bytes(b"old")

        async def _trigger() -> bool:
            (tmp_path / "partial.bin.crdownload").write_bytes(b"wip")
            (tmp_path / "fresh.bin").write_bytes(b"fresh-data")
            return True

        result = await wait_for_page_download(session, str(tmp_path), _trigger, timeout=_SHORT_TIMEOUT_S)

        assert result == {
            "downloaded": True,
            "path": str(tmp_path / "fresh.bin"),
            "filename": "fresh.bin",
            "size": 10,
        }

    async def test_timeout_fallback_claims_nothing_when_multiple_files_appear(self, tmp_path: Path) -> None:
        """When multiple new files land without an event, none is claimed."""
        session = _build_session()

        async def _trigger() -> bool:
            (tmp_path / "first.bin").write_bytes(b"one")
            (tmp_path / "second.bin").write_bytes(b"two")
            return True

        result = await wait_for_page_download(session, str(tmp_path), _trigger, timeout=_SHORT_TIMEOUT_S)

        assert result == {"downloaded": False}

    async def test_removes_listeners_when_trigger_action_raises(self, tmp_path: Path) -> None:
        """An exception from the trigger action propagates after cleaning up session listeners."""
        session = _build_session()
        action = AsyncMock(side_effect=RuntimeError("click failed"))

        with pytest.raises(RuntimeError, match="click failed"):
            await wait_for_page_download(session, str(tmp_path), action, timeout=5)

        assert session.listeners.get("Browser.downloadWillBegin", []) == []
        assert session.listeners.get("Browser.downloadProgress", []) == []
