"""Tests for app.lib.shared.paths module."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Self
from unittest.mock import AsyncMock

import pytest

from app.lib.shared.paths import (
    data_dir,
    download_dir,
    is_audio_mime,
    is_binary_mime,
    is_image_file,
    is_image_mime,
    is_pdf_mime,
    is_text_mime,
    is_video_mime,
    mime_type,
    resolve_path,
    resolve_path_sync,
    resources_dir,
    scratch_dir,
)


def _result(stdout: str = "", stderr: str = "", returncode: int = 0) -> SimpleNamespace:
    """Make a fake command result with this output, error text, and exit code."""
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class TestScratchDir:
    """Test the scratch_dir function."""

    def test_returns_default_when_env_unset(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With nothing set, the folder is under /dev/shm/agent."""
        monkeypatch.delenv("SCRATCH_DIR", raising=False)
        assert scratch_dir("x") == "/dev/shm/agent/x"

    def test_returns_folder_under_env_value(self: Self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A scratch root that is set holds the named folder."""
        monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
        assert scratch_dir("x") == str(tmp_path / "x")

    def test_empty_value_counts_as_unset(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty scratch root gives the default."""
        monkeypatch.setenv("SCRATCH_DIR", "")
        assert scratch_dir("x") == "/dev/shm/agent/x"


class TestDownloadDir:
    """Test the download_dir function."""

    def test_returns_default_when_env_unset(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With nothing set, downloads land in /dev/shm/agent/downloads."""
        monkeypatch.delenv("SCRATCH_DIR", raising=False)
        assert download_dir() == "/dev/shm/agent/downloads"

    def test_follows_scratch_root(self: Self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """With SCRATCH_DIR set, downloads land in its downloads folder."""
        monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
        assert download_dir() == str(tmp_path / "downloads")


class TestDataDir:
    """Test the data_dir function."""

    def test_returns_default_when_env_unset(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With nothing set, the folder is /app/data."""
        monkeypatch.delenv("DATA_DIR", raising=False)
        assert data_dir() == "/app/data"

    def test_returns_env_value(self: Self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A folder that is set is returned as is."""
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        assert data_dir() == str(tmp_path)

    def test_empty_value_counts_as_unset(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty folder value gives the default."""
        monkeypatch.setenv("DATA_DIR", "")
        assert data_dir() == "/app/data"


class TestResourcesDir:
    """Test the resources_dir function."""

    def test_returns_default_when_env_unset(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With nothing set, the folder is /app/data/resources."""
        monkeypatch.delenv("RESOURCES_DIR", raising=False)
        assert resources_dir() == "/app/data/resources"

    def test_returns_env_value(self: Self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A folder that is set is returned as is."""
        monkeypatch.setenv("RESOURCES_DIR", str(tmp_path))
        assert resources_dir() == str(tmp_path)

    def test_empty_value_counts_as_unset(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty folder value gives the default."""
        monkeypatch.setenv("RESOURCES_DIR", "")
        assert resources_dir() == "/app/data/resources"


class TestResolvePath:
    """Test the resolve_path function."""

    async def test_resolves_file_under_root(self: Self, tmp_path: Path) -> None:
        expected = os.path.realpath(str(tmp_path / "upload" / "a.md"))
        assert await resolve_path(str(tmp_path), "upload/a.md") == expected

    async def test_returns_root_itself_for_empty_path(self: Self, tmp_path: Path) -> None:
        assert await resolve_path(str(tmp_path), "") == os.path.realpath(str(tmp_path))

    @pytest.mark.parametrize("path", ["../escape.txt", "a/../../escape.txt", "/etc/passwd"])
    async def test_rejects_traversal(self: Self, tmp_path: Path, path: str) -> None:
        """A path that leaves the root gives back None."""
        assert await resolve_path(str(tmp_path), path) is None

    async def test_resolves_through_symlinked_root(self: Self, tmp_path: Path) -> None:
        """A root that is a symlink resolves to the real folder."""
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        assert await resolve_path(str(link), "a.txt") == os.path.realpath(str(real / "a.txt"))

    def test_sync_runs_without_an_event_loop(self: Self, tmp_path: Path) -> None:
        """The sync entry gives the same answers with no loop running."""
        expected = os.path.realpath(str(tmp_path / "upload" / "a.md"))
        assert resolve_path_sync(str(tmp_path), "upload/a.md") == expected
        assert resolve_path_sync(str(tmp_path), "../escape.txt") is None


class TestMimeType:
    """Test the mime_type function."""

    async def test_returns_mime_on_success(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A file with a readable type gives back that type."""
        cli = AsyncMock(return_value=_result("image/png"))
        monkeypatch.setattr("app.lib.shared.paths.run_cli", cli)
        assert await mime_type("pic.png") == "image/png"
        cli.assert_awaited_once_with(["exiftool", "-b", "-MimeType", "pic.png"], "exiftool mime detection")

    async def test_returns_none_on_failure_returncode(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A command that fails gives back None."""
        cli = AsyncMock(return_value=_result("", "exiftool: no such file", 1))
        monkeypatch.setattr("app.lib.shared.paths.run_cli", cli)
        assert await mime_type("missing.png") is None

    async def test_returns_none_on_empty_stdout(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A command that printed nothing gives back None."""
        cli = AsyncMock(return_value=_result(""))
        monkeypatch.setattr("app.lib.shared.paths.run_cli", cli)
        assert await mime_type("blank.bin") is None

    async def test_returns_none_on_empty_path(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty path gives back None without running anything."""
        cli = AsyncMock(return_value=_result("image/png"))
        monkeypatch.setattr("app.lib.shared.paths.run_cli", cli)
        assert await mime_type("") is None
        cli.assert_not_awaited()


class TestMimePredicates:
    """Test the mime type predicates."""

    def test_image_mime_matches_only_images(self: Self) -> None:
        """An image type matches, and other types and None do not."""
        assert is_image_mime("image/png") is True
        assert is_image_mime("video/mp4") is False
        assert is_image_mime(None) is False

    def test_video_mime_matches_only_videos(self: Self) -> None:
        """A video type matches, and other types and None do not."""
        assert is_video_mime("video/mp4") is True
        assert is_video_mime("image/png") is False
        assert is_video_mime(None) is False

    def test_audio_mime_matches_only_audio(self: Self) -> None:
        """An audio type matches, and other types and None do not."""
        assert is_audio_mime("audio/mpeg") is True
        assert is_audio_mime("text/plain") is False
        assert is_audio_mime(None) is False

    @pytest.mark.parametrize("mime", ["text/plain", "text/html"])
    def test_text_mime_matches(self: Self, mime: str) -> None:
        """Text/* types count as text."""
        assert is_text_mime(mime) is True

    @pytest.mark.parametrize(
        "mime",
        [
            "image/png",
            "application/pdf",
            "application/json",
            "application/xml",
            "application/xhtml+xml",
            None,
        ],
    )
    def test_text_mime_rejects(self: Self, mime: str | None) -> None:
        """Non-text/* types and None do not count as text."""
        assert is_text_mime(mime) is False

    def test_pdf_mime_matches(self: Self) -> None:
        """A PDF counts as a PDF."""
        assert is_pdf_mime("application/pdf") is True

    @pytest.mark.parametrize("mime", ["text/plain", "application/json", None])
    def test_pdf_mime_rejects(self: Self, mime: str | None) -> None:
        """Other types and None do not count as a PDF."""
        assert is_pdf_mime(mime) is False

    @pytest.mark.parametrize("mime", ["image/png", "video/mp4", "audio/mpeg", "application/pdf"])
    def test_binary_mime_is_binary(self: Self, mime: str) -> None:
        """Image, video, audio, and PDF types count as binary."""
        assert is_binary_mime(mime) is True

    @pytest.mark.parametrize(
        "mime",
        ["text/plain", "application/json", "application/xml", "application/xhtml+xml"],
    )
    def test_binary_mime_is_text(self: Self, mime: str) -> None:
        """Text types and the known textual document types do not count as binary."""
        assert is_binary_mime(mime) is False

    def test_binary_mime_is_binary_for_other_detected_types(self: Self) -> None:
        """Any other detected type counts as binary."""
        assert is_binary_mime("application/zip") is True

    @pytest.mark.parametrize("mime", [None, ""])
    def test_binary_mime_is_missing(self: Self, mime: str | None) -> None:
        """An empty or missing type gives back None."""
        assert is_binary_mime(mime) is None


class TestFilePredicates:
    """Test the file type predicates."""

    async def test_image_file_true_for_image(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A file whose type is an image counts as an image."""
        cli = AsyncMock(return_value=_result("image/png"))
        monkeypatch.setattr("app.lib.shared.paths.run_cli", cli)
        assert await is_image_file("pic.png") is True

    async def test_image_file_false_for_video(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A file whose type is a video does not count as an image."""
        cli = AsyncMock(return_value=_result("video/mp4"))
        monkeypatch.setattr("app.lib.shared.paths.run_cli", cli)
        assert await is_image_file("clip.mp4") is False

    async def test_image_file_false_when_type_unreadable(self: Self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A file whose type cannot be read does not count as an image."""
        cli = AsyncMock(return_value=_result("", "exiftool: no such file", 1))
        monkeypatch.setattr("app.lib.shared.paths.run_cli", cli)
        assert await is_image_file("mystery.bin") is False
