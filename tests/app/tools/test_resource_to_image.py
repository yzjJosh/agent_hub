"""Tests for app.tools.resource_to_image."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import httpx
import pytest

import app.lib.db.store as store_module
from app.lib.render import RenderKind, cache
from app.tools.resource_to_image import resource_to_image_tool
from tests.app.conftest import (
    E2E_PNG,
    point_share_env,
    skip_if_no_chrome,
    thumb_calls,
    write_share_file,
)
from tests.app.lib.tool.conftest import run_text

_BASE = "http://localhost:8315"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_RETRY_LINE = (
    "While the thumbnail is still generating, this URL returns HTTP 409 — "
    "retry until it returns the 1024x640 PNG."
)
_LIVE_POLL_INTERVAL_S = 0.2
_LIVE_READY_TIMEOUT_S = 60.0


def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point the share folders and the base URL at temp values, then give back the resources and data folders."""
    resources, data = point_share_env(tmp_path, monkeypatch)
    monkeypatch.setenv("DAV_BASE_URL", _BASE)
    return resources, data


def _share_file(resources: Path, rel: str, data: bytes) -> Path:
    """Create a file in the resource share and return its absolute path."""
    path = resources / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _key_for_file(rel: str, full: Path) -> str:
    """The cache key for this file."""
    return cache.key_for(rel, int(full.stat().st_mtime))


def _seed_thumb(data: Path, key: str) -> None:
    """Put a finished thumbnail in place for this key."""
    key_dir = data / "render" / key
    key_dir.mkdir(parents=True, exist_ok=True)
    (key_dir / "thumb.png").write_bytes(_PNG_MAGIC)


async def _seed_failed(key: str) -> None:
    """Mark this key as failed."""
    await cache.mark_failed(key, "thumb", "engine error")


def _expected_url(path: str) -> str:
    """The thumbnail URL the tool gives back for a share path."""
    return f"{_BASE}/api/resources/thumb?path={quote(path, safe='/')}"


def _expected_message(path: str, *, ready: bool) -> str:
    """The success message, with the retry line only when the thumbnail is not ready."""
    url = _expected_url(path)
    stem = os.path.splitext(os.path.basename(path))[0] or "thumb"
    lines = [
        f"Preview thumbnail URL for {path}:",
        f"  {url}",
        "Download it with:",
        f"  curl -o '/tmp/{stem}.png' {url}",
    ]
    if not ready:
        lines.append(_RETRY_LINE)
    return "\n".join(lines)


class TestValidation:
    """Tests for bad share paths."""

    async def test_missing_file_returns_exact_not_found_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing path gives back the not found error with upload help."""
        _resources, _data = _env(tmp_path, monkeypatch)
        result = await run_text(resource_to_image_tool, "upload/nope.pdf")

        assert result == (
            "Tool 'resource_to_image' failed: "
            "Resource not found: upload/nope.pdf. The resource path must already "
            "exist in this server's resources share. Call the send_or_receive_file "
            "tool for instructions to upload the file, then retry this tool with "
            "the uploaded file's resource path (e.g. 'upload/<filename>')."
        )

    async def test_path_escaping_share_returns_exact_not_found_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A path that points outside the share is turned down and shows nothing."""
        resources, _data = _env(tmp_path, monkeypatch)
        outside = resources.parent / "outside_resource_to_image"
        outside.mkdir(exist_ok=True)
        (outside / "secret.txt").write_text("secret content")
        result = await run_text(resource_to_image_tool, "../outside_resource_to_image/secret.txt")

        assert result == (
            "Tool 'resource_to_image' failed: "
            "Resource not found: ../outside_resource_to_image/secret.txt. The "
            "resource path must already exist in this server's resources share. "
            "Call the send_or_receive_file tool for instructions to upload the "
            "file, then retry this tool with the uploaded file's resource path "
            "(e.g. 'upload/<filename>')."
        )
        assert "secret content" not in result

    async def test_directory_returns_exact_folder_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A folder is turned down with a not a file error."""
        resources, _data = _env(tmp_path, monkeypatch)
        (resources / "upload").mkdir()
        result = await run_text(resource_to_image_tool, "upload")

        assert result == (
            "Tool 'resource_to_image' failed: "
            "'upload' is a folder, not a file. Pass the resource path of a "
            "single file instead, e.g. 'upload/<filename>'."
        )

    async def test_non_string_path_returns_canonical_type_error(self) -> None:
        """A resource_path that is not text is rejected before any file is read."""
        result = await run_text(resource_to_image_tool, 5)
        assert result == "Error: resource_path: Input should be a valid string."


class TestNotPreviewable:
    """Tests for files that have no preview."""

    @pytest.mark.parametrize(
        ("rel", "kind"),
        [
            ("upload/bundle.zip", "archive"),
            ("upload/track.mp3", "audio"),
            ("upload/blob.bin", "unknown"),
        ],
    )
    async def test_kind_without_preview_engine_reports_exact_error_without_enqueue(
        self,
        rel: str,
        kind: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A file with no preview gives an error with a download command, and queues no work."""
        resources, _data = _env(tmp_path, monkeypatch)
        _share_file(resources, rel, b"\x00\x01\x02 binary fixture bytes")
        calls = thumb_calls(monkeypatch)
        result = await run_text(resource_to_image_tool, rel)

        assert result == (
            "Tool 'resource_to_image' failed: "
            f"'{rel}' cannot be previewed as an image: file type '{kind}' has no "
            "preview thumbnail. The file itself remains downloadable:\n"
            f"  curl -o '/tmp/{os.path.basename(rel)}' {_BASE}/dav/{rel}"
        )
        assert calls == []


class TestFailed:
    """Tests for a thumbnail that already failed."""

    async def test_failed_thumbnail_reports_exact_error_without_enqueue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed thumbnail gives an error and queues no work."""
        resources, _data = _env(tmp_path, monkeypatch)
        full = _share_file(resources, "upload/pic.png", _PNG_MAGIC)
        await _seed_failed(_key_for_file("upload/pic.png", full))
        calls = thumb_calls(monkeypatch)
        result = await run_text(resource_to_image_tool, "upload/pic.png")

        assert result == (
            "Tool 'resource_to_image' failed: "
            "Preview thumbnail for 'upload/pic.png' has failed to generate and is not "
            "retried automatically. The source file remains downloadable:\n"
            f"  curl -o '/tmp/pic.png' {_BASE}/dav/upload/pic.png"
        )
        assert calls == []


class TestSuccess:
    """Tests for the thumbnail URL the tool gives back."""

    async def test_done_returns_exact_message_without_retry_line_and_without_enqueue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ready thumbnail gives the URL with no retry line, and queues no work."""
        resources, data = _env(tmp_path, monkeypatch)
        full = _share_file(resources, "upload/report.pdf", b"%PDF fixture")
        _seed_thumb(data, _key_for_file("upload/report.pdf", full))
        calls = thumb_calls(monkeypatch)
        result = await run_text(resource_to_image_tool, "upload/report.pdf")

        assert result == _expected_message("upload/report.pdf", ready=True)
        assert _RETRY_LINE not in result
        assert calls == []

    async def test_pending_returns_retry_line_and_enqueues_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file with no thumbnail yet gives the URL with the retry line, and queues one job."""
        resources, _data = _env(tmp_path, monkeypatch)
        full = _share_file(resources, "upload/pic.png", _PNG_MAGIC)
        key = _key_for_file("upload/pic.png", full)
        calls = thumb_calls(monkeypatch)
        result = await run_text(resource_to_image_tool, "upload/pic.png")

        assert result == _expected_message("upload/pic.png", ready=False)
        assert _RETRY_LINE in result
        assert calls == [(key, RenderKind.IMAGE, "upload/pic.png")]

    async def test_generating_returns_retry_line_without_enqueue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A thumbnail already being made gives the URL with the retry line, and queues no work."""
        resources, _data = _env(tmp_path, monkeypatch)
        full = _share_file(resources, "upload/pic.png", _PNG_MAGIC)
        key = _key_for_file("upload/pic.png", full)
        monkeypatch.setattr("app.lib.render.gen_state.generating_keys", lambda: frozenset({key}))
        calls = thumb_calls(monkeypatch)
        result = await run_text(resource_to_image_tool, "upload/pic.png")

        assert result == _expected_message("upload/pic.png", ready=False)
        assert calls == []

    async def test_url_percent_encodes_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A path with a space gives an escaped URL and a download name without the old extension."""
        resources, _data = _env(tmp_path, monkeypatch)
        _share_file(resources, "upload/my report.pdf", b"%PDF fixture")
        calls = thumb_calls(monkeypatch)
        result = await run_text(resource_to_image_tool, "upload/my report.pdf")

        assert result == _expected_message("upload/my report.pdf", ready=False)
        assert f"  {_expected_url('upload/my report.pdf')}" in result
        assert "curl -o '/tmp/my report.png'" in result
        expected_key = _key_for_file(
            "upload/my report.pdf", resources / "upload/my report.pdf"
        )
        assert calls == [(expected_key, RenderKind.PDF, "upload/my report.pdf")]

    async def test_extensionless_source_gets_single_png_suffix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file with no extension gets a download name ending in .png."""
        resources, _data = _env(tmp_path, monkeypatch)
        _share_file(resources, "upload/cover", _PNG_MAGIC)
        calls = thumb_calls(monkeypatch)
        result = await run_text(resource_to_image_tool, "upload/cover")

        assert result == _expected_message("upload/cover", ready=False)
        assert "curl -o '/tmp/cover.png'" in result
        assert calls and calls[0][2] == "upload/cover"


class TestUntracked:
    """Tests that the tool leaves no run records."""

    async def test_successful_call_creates_no_run_records(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A good call gives back the URL and writes no rows or run folders."""
        resources, data = _env(tmp_path, monkeypatch)
        full = _share_file(resources, "upload/pic.png", _PNG_MAGIC)
        _seed_thumb(data, _key_for_file("upload/pic.png", full))
        result = await run_text(resource_to_image_tool, "upload/pic.png")

        assert result.startswith("Preview thumbnail URL for upload/pic.png:")
        stats = await store_module.tool_stats()
        assert [s for s in stats if s["tool_name"] == "resource_to_image"] == []
        assert await store_module.total_count() == 0
        runs = os.path.join(store_module.tool_store.data_dir, "runs")
        assert (os.listdir(runs) if os.path.isdir(runs) else []) == []


class TestContract:
    """Tests for what the tool and its params say."""

    def test_base_description_states_purpose_and_when_to_use(self) -> None:
        """The tool description says it gives a preview thumbnail and leaves the original file alone."""
        desc = resource_to_image_tool.description
        assert "preview thumbnail" in desc
        assert "existing resource" in desc
        assert "original file is not affected" in desc
        assert "resources share" not in desc
        assert "resource share" not in desc

    def test_param_documents_resource_convention(self) -> None:
        """The base says the path is relative to the resource root, and the wire adds the upload steps."""
        p = next(p for p in resource_to_image_tool.params if p.name == "resource_path")
        assert "relative to the resource root" in p.description
        assert "must already exist" in p.wire_description
        assert "send_or_receive_file" in p.wire_description
        assert "upload/<filename>" in p.wire_description
        assert "resources share" not in p.description
        assert "resources share" not in p.wire_description


async def _poll_until_ready(url: str) -> bytes:
    """Ask for the URL until it answers 200, then give back the body."""
    deadline = time.monotonic() + _LIVE_READY_TIMEOUT_S
    async with httpx.AsyncClient() as client:
        while True:
            response = await client.get(url)
            if response.status_code == 200:
                return response.content
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"thumbnail not ready within {_LIVE_READY_TIMEOUT_S}s "
                    f"(last HTTP {response.status_code})"
                )
            await asyncio.sleep(_LIVE_POLL_INTERVAL_S)


@skip_if_no_chrome
@pytest.mark.live
class TestResourceToImageLive:
    """Live thumbnail tests."""

    @pytest.mark.timeout(120)
    async def test_live_url_serves_generated_thumbnail(
        self, app_http: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A live call gives back a URL that ends up serving a real PNG."""
        monkeypatch.setenv("DAV_BASE_URL", app_http.base_url)
        write_share_file(app_http.share, "upload/live.png", E2E_PNG)

        result = await run_text(resource_to_image_tool, "upload/live.png")
        url = result.splitlines()[1].strip()
        assert url == f"{app_http.base_url}/api/resources/thumb?path=upload/live.png"
        body = await _poll_until_ready(url)
        assert body.startswith(_PNG_MAGIC)
