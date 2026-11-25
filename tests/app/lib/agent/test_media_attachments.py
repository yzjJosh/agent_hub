"""Unit tests for loading media attachments for tool results."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from io import IOBase
from pathlib import Path

import aiofiles.threadpool
import pytest
from agents import ToolOutputFileContent, ToolOutputImage, ToolOutputText

from app.lib.agent.media_attachments import load_media_attachment_parts
from app.lib.shared.paths import resources_dir

_SAMPLE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc`\x00\x02\x00\x00\x05\x00\x01z^\xab?\x00\x00"
    b"\x00\x00IEND\xaeB`\x82"
)
_SAMPLE_MP4 = base64.b64decode(
    "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAs1tZGF0AAACrQYF//+p3EXpvebZSLeWLNgg2SPu"
    "73gyNjQgLSBjb3JlIDE2NCByMzEwOCAzMWUxOWY5IC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMt"
    "MjAyMyAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9j"
    "az0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3Jl"
    "Zj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0"
    "X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTEgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFk"
    "cz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZy"
    "YW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWln"
    "aHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTEgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD00"
    "MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlv"
    "PTEuNDAgYXE9MToxLjAwAIAAAAAQZYiEABX//vfJ78Cm69vfgQAAAxVtb292AAAAbG12aGQAAAAAAAAAAAAAAAAAAAPoAAAD"
    "6AABAAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAACAAACP3RyYWsAAABcdGtoZAAAAAMAAAAAAAAAAAAAAAEAAAAAAAAD6AAAAAAAAAAAAAAAAAAAAAAAAQAAAAAA"
    "AAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAEAAAABAAAAAAACRlZHRzAAAAHGVsc3QAAAAAAAAAAQAAA+gAAAAAAAEA"
    "AAAAAbdtZGlhAAAAIG1kaGQAAAAAAAAAAAAAAAAAAEAAAABAAFXEAAAAAAAtaGRscgAAAAAAAAAAdmlkZQAAAAAAAAAAAAAA"
    "AFZpZGVvSGFuZGxlcgAAAAFibWluZgAAABR2bWhkAAAAAQAAAAAAAAAAAAAAJGRpbmYAAAAcZHJlZgAAAAAAAAABAAAADHVy"
    "bCAAAAABAAABInN0YmwAAAC+c3RzZAAAAAAAAAABAAAArmF2YzEAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAEAAQAEgAAABI"
    "AAAAAAAAAAEVTGF2YzYwLjMxLjEwMiBsaWJ4MjY0AAAAAAAAAAAAAAAY//8AAAA0YXZjQwFkAAr/4QAXZ2QACqzZXsBEAAAD"
    "AAQAAAMACDxIllgBAAZo6+PLIsD9+PgAAAAAEHBhc3AAAAABAAAAAQAAABRidHJ0AAAAAAAAFigAABYoAAAAGHN0dHMAAAAA"
    "AAAAAQAAAAEAAEAAAAAAHHN0c2MAAAAAAAAAAQAAAAEAAAABAAAAAQAAABRzdHN6AAAAAAAAAsUAAAABAAAAFHN0Y28AAAAA"
    "AAAAAQAAADAAAABidWR0YQAAAFptZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAA"
    "Jal0b28AAAAdZGF0YQAAAAEAAAAATGF2ZjYwLjE2LjEwMA=="
)
_SAMPLE_MP3 = b"\xff\xfb\x90\x00"
_SAMPLE_PDF = b"%PDF-1.4 junk"


def _fail_image_open(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    real_open = aiofiles.threadpool.sync_open

    def guarded_open(
        file: str | bytes | int,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: Callable[[str, int], int] | None = None,
    ) -> IOBase:
        if isinstance(file, str) and file.endswith("img.png"):
            raise error
        return real_open(
            file,
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
            opener=opener,
        )

    monkeypatch.setattr(aiofiles.threadpool, "sync_open", guarded_open)


class TestLoadMediaAttachmentParts:
    """Verify loading image and video attachments and reporting media errors."""

    async def test_loads_image_and_video_and_skips_other_files(self) -> None:
        resource_root = Path(resources_dir())
        (resource_root / "img.png").write_bytes(_SAMPLE_PNG)
        (resource_root / "vid.mp4").write_bytes(_SAMPLE_MP4)
        (resource_root / "aud.mp3").write_bytes(_SAMPLE_MP3)
        (resource_root / "doc.pdf").write_bytes(_SAMPLE_PDF)

        parts = await load_media_attachment_parts(("img.png", "vid.mp4", "aud.mp3", "doc.pdf"))
        assert [type(part) for part in parts] == [ToolOutputImage, ToolOutputFileContent]
        image_part = parts[0]
        assert isinstance(image_part, ToolOutputImage)
        assert isinstance(image_part.image_url, str)
        assert image_part.image_url.startswith("data:image/png;base64,")
        assert base64.b64decode(image_part.image_url.split(",", 1)[1]) == _SAMPLE_PNG
        video_part = parts[1]
        assert isinstance(video_part, ToolOutputFileContent)
        assert isinstance(video_part.file_data, str)
        assert video_part.file_data.startswith("data:video/mp4;base64,")
        assert base64.b64decode(video_part.file_data.split(",", 1)[1]) == _SAMPLE_MP4
        assert video_part.filename == "vid.mp4"

    async def test_empty_file_tuple_returns_empty_list(self) -> None:
        assert await load_media_attachment_parts(()) == []

    @pytest.mark.parametrize("path", ["missing.png", "../outside.png"])
    async def test_unavailable_media_returns_delivery_error(self, path: str) -> None:
        parts = await load_media_attachment_parts((path,))
        assert len(parts) == 1
        assert isinstance(parts[0], ToolOutputText)
        assert parts[0].text == f"Attachment delivery failed for {path!r}. Media could not be read or decoded."

    async def test_missing_nonmedia_file_is_ignored(self) -> None:
        parts = await load_media_attachment_parts(("missing.pdf",))
        assert parts == []

    async def test_absolute_paths_load_valid_images_and_report_missing_ones(self, tmp_path: Path) -> None:
        image_path = tmp_path / "outside.png"
        image_path.write_bytes(_SAMPLE_PNG)
        missing_path = tmp_path / "missing.png"

        parts = await load_media_attachment_parts((str(image_path), str(missing_path)))
        assert [type(part) for part in parts] == [ToolOutputImage, ToolOutputText]
        assert isinstance(parts[0], ToolOutputImage)
        assert isinstance(parts[0].image_url, str)
        assert base64.b64decode(parts[0].image_url.split(",", 1)[1]) == _SAMPLE_PNG
        assert isinstance(parts[1], ToolOutputText)
        assert parts[1].text.startswith("Attachment delivery failed for ")

    @pytest.mark.parametrize(
        ("filename", "raw_bytes"),
        [
            ("broken.png", b"not an image"),
            ("truncated.png", _SAMPLE_PNG[:40]),
            ("broken.mp4", b"\x00\x00\x00\x18ftypmp42"),
        ],
    )
    async def test_corrupt_media_reports_error_without_dropping_valid_media(
        self,
        filename: str,
        raw_bytes: bytes,
    ) -> None:
        resource_root = Path(resources_dir())
        (resource_root / filename).write_bytes(raw_bytes)
        (resource_root / "img.png").write_bytes(_SAMPLE_PNG)

        parts = await load_media_attachment_parts((filename, "img.png"))
        assert [type(part) for part in parts] == [ToolOutputText, ToolOutputImage]
        assert isinstance(parts[0], ToolOutputText)
        assert parts[0].text == f"Attachment delivery failed for {filename!r}. Media could not be read or decoded."
        assert isinstance(parts[1], ToolOutputImage)
        assert isinstance(parts[1].image_url, str)
        assert base64.b64decode(parts[1].image_url.split(",", 1)[1]) == _SAMPLE_PNG

    @pytest.mark.parametrize(
        ("error", "expected_reason"),
        [
            (PermissionError("permission denied"), "Media could not be read or decoded."),
            (RuntimeError("unexpected failure"), "Media could not be delivered."),
        ],
    )
    async def test_read_errors_return_delivery_failure_notice(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: Exception,
        expected_reason: str,
    ) -> None:
        resource_root = Path(resources_dir())
        (resource_root / "img.png").write_bytes(_SAMPLE_PNG)
        (resource_root / "vid.mp4").write_bytes(_SAMPLE_MP4)
        _fail_image_open(monkeypatch, error)

        parts = await load_media_attachment_parts(("img.png", "vid.mp4"))
        assert [type(part) for part in parts] == [ToolOutputText, ToolOutputFileContent]
        assert isinstance(parts[0], ToolOutputText)
        assert parts[0].text == f"Attachment delivery failed for 'img.png'. {expected_reason}"
        assert isinstance(parts[1], ToolOutputFileContent)
        assert isinstance(parts[1].file_data, str)
        assert base64.b64decode(parts[1].file_data.split(",", 1)[1]) == _SAMPLE_MP4

    async def test_cancelled_read_propagates_cancellation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resource_root = Path(resources_dir())
        (resource_root / "img.png").write_bytes(_SAMPLE_PNG)
        _fail_image_open(monkeypatch, asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await load_media_attachment_parts(("img.png",))
