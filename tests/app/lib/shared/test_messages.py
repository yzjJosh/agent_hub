"""Tests for app.lib.shared.messages."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.lib.shared.messages import missing_reference_message, saved_file_message
from tests.app.conftest import DAV_BASE_URL, point_resources_at


class TestSavedFileMessage:

    @pytest.mark.parametrize("mode", ["image", "video"])
    def test_names_the_file_and_the_download_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        """The message points at the saved file and shows how to fetch it."""
        point_resources_at(tmp_path, monkeypatch)
        output_path = str(tmp_path / "media_gen" / f"media_gen_{mode}_20260101_000000.png")

        assert saved_file_message(mode, output_path) == (
            f"{mode.capitalize()} generated and saved to resource path: "
            f"media_gen/media_gen_{mode}_20260101_000000.png\n"
            "Download it with:\n"
            f"  curl -o /tmp/media_gen_{mode}_20260101_000000.png "
            f"{DAV_BASE_URL}/dav/media_gen/media_gen_{mode}_20260101_000000.png\n"
            "See the send_or_receive_file tool for file transfer details."
        )


class TestMissingReferenceMessage:

    def test_names_the_path_and_the_upload_tool(self) -> None:
        """The message names the missing path and points at the upload tool."""
        assert missing_reference_message("upload/missing.png") == (
            "Resource not found: upload/missing.png. The resource path must already "
            "exist in this server's resources share. Call the send_or_receive_file "
            "tool for instructions to upload the file, then retry this tool with "
            "the uploaded file's resource path (e.g. 'upload/<filename>')."
        )
