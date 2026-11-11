"""Shared setup and helpers for the app tests."""

import asyncio
import importlib.util
import io
import os
import re
import socket
import struct
import time
import uuid
import zlib
import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import requests
import uvicorn
from PIL import Image, ImageDraw
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

import app.lib.db.store as store_module
import app.server as server_module
from app.console.errors import error_response
from app.lib.browser.browser_manager import BrowserManager
from app.lib.comfyui.client import GenerationResult
from app.lib.render import result_feed as rf
from app.lib.render.classify import RenderKind
from app.lib.shared.paths import resources_dir
from app.lib.shared.timefmt import iso_to_local
from app.lib.tool.base import Tool
from app.mcp.registry import tool_class
from tests.app.lib.tool.conftest import run_text

if TYPE_CHECKING:
    import _pytest.config


@pytest.fixture(scope="module")
def frontend_assets() -> Iterator[TestClient]:
    """Serve frontend assets through the whole app."""
    client = TestClient(server_module.build_app())
    try:
        yield client
    finally:
        client.close()


def point_share_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Make fresh share and data dirs and point the app at them.

    Args:
        tmp_path: Folder the new dirs are made in.
        monkeypatch: Lets the test change the environment.

    Returns:
        The share dir and the data dir.
    """
    share = tmp_path / "res"
    share.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("RESOURCES_DIR", str(share))
    monkeypatch.setenv("DATA_DIR", str(data))
    return share, data


DAV_BASE_URL = "http://dav.test"


def point_resources_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the resources folder and the download address at test values."""
    monkeypatch.setenv("RESOURCES_DIR", str(tmp_path))
    monkeypatch.setenv("DAV_BASE_URL", DAV_BASE_URL)


def write_reference_image(tmp_path: Path) -> Path:
    """Write a reference image into the upload folder."""
    ref_file = tmp_path / "upload" / "ref.png"
    ref_file.parent.mkdir(parents=True)
    ref_file.write_bytes(b"reference image")
    return ref_file


def write_reference_png(tmp_path: Path, name: str, width: int, height: int) -> str:
    """Write a solid-color PNG of the given size into the upload folder."""
    ref_file = tmp_path / "upload" / name
    ref_file.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (200, 100, 50)).save(ref_file)
    return f"upload/{name}"


def assert_saved_image(tmp_path: Path, pattern: str, result: str, expected_bytes: bytes) -> str:
    """Exactly one file was saved under media_gen with the expected bytes, and the result names it."""
    saved = list((tmp_path / "media_gen").glob(pattern))
    assert len(saved) == 1
    assert saved[0].read_bytes() == expected_bytes
    resource_path = f"media_gen/{saved[0].name}"
    assert result.startswith(f"Image generated and saved to resource path: {resource_path}")
    assert f"{DAV_BASE_URL}/dav/{resource_path}" in result
    return resource_path


def wire_media_client(mock_client_cls: MagicMock, data: bytes, filename: str) -> AsyncMock:
    """Set up the patched client class for a media tool run."""
    client = AsyncMock()
    client.generate = AsyncMock(return_value=GenerationResult(images=(data,), filenames=(filename,)))
    mock_client_cls.return_value = client
    return client


def submitted_prompt(mock_client_cls: MagicMock) -> dict:
    """The prompt the run sent to generate."""
    return mock_client_cls.return_value.generate.await_args.args[0]


async def run_media_tool(
    mock_client_cls: MagicMock,
    tool: Tool,
    client_path: str,
    *args: str,
    **kwargs: str | bool | list[str],
) -> str:
    """Run the tool with the fake client and return the finished result."""
    patcher = patch(client_path, mock_client_cls)
    patcher.start()
    try:
        return await run_text(tool, *args, **kwargs)
    finally:
        patcher.stop()


_LIVE_PROBE_TIMEOUT_S = 10


def _comfyui_base_url() -> str:
    """The ComfyUI backend URL from the environment, or empty when it is not set."""
    return (os.getenv("COMFYUI_BACKEND_URL") or "").strip().rstrip("/")


def comfyui_online() -> bool:
    """Whether the ComfyUI backend is set and answers any HTTP request at all."""
    base = _comfyui_base_url()
    if not base:
        return False
    try:
        with httpx.Client(timeout=_LIVE_PROBE_TIMEOUT_S, follow_redirects=True) as client:
            client.get(f"{base}/")
    except httpx.HTTPError:
        return False
    return True


def require_live_media() -> None:
    """Skip the test when the live media backend is not reachable."""
    if "DAV_BASE_URL" not in os.environ:
        pytest.skip("DAV_BASE_URL is not set, the backend host cannot fetch the reference")
    if not _comfyui_base_url():
        pytest.skip("COMFYUI_BACKEND_URL is not set")
    if not comfyui_online():
        pytest.skip("the ComfyUI backend is not reachable")


async def publish_reference(base_url: str, name: str, data: bytes) -> None:
    """Put an image in the public upload folder, skipping when it cannot be fetched."""
    url = f"{base_url}/dav/upload/{name}"
    try:
        async with httpx.AsyncClient(timeout=_LIVE_PROBE_TIMEOUT_S, follow_redirects=True) as client:
            put = await client.put(url, content=data)
            if not put.is_success:
                pytest.skip(f"could not upload the image to {url}, HTTP {put.status_code}")
            get = await client.get(url)
            if get.status_code != 200:
                pytest.skip(f"the image is not publicly fetchable at {url}")
    except httpx.HTTPError as exc:
        pytest.skip(f"the live agent server is not reachable at {base_url}, {exc}")


async def retract_reference(base_url: str, name: str) -> None:
    """Delete an image from the public upload folder, ignoring failures."""
    try:
        async with httpx.AsyncClient(timeout=_LIVE_PROBE_TIMEOUT_S, follow_redirects=True) as client:
            await client.delete(f"{base_url}/dav/upload/{name}")
    except httpx.HTTPError:
        pass


def stage_reference_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference: str, data: bytes) -> None:
    """Point the resources folder at a fresh dir holding the reference image."""
    monkeypatch.setenv("RESOURCES_DIR", str(tmp_path))
    ref_path = tmp_path / reference
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_bytes(data)


async def live_image_run(tool: Tool, target: str, **kwargs: str | bool | list[str]) -> Path:
    """Run one live image generation, verify the saved PNG, and return its path."""
    result = await run_text(tool, target, **kwargs)
    assert result.startswith("Image generated and saved to resource path:")
    resource_path = result.split("resource path: ", 1)[1].split("\n", 1)[0]
    saved = Path(resources_dir()) / resource_path
    assert saved.exists()
    assert saved.read_bytes().startswith(b"\x89PNG")
    return saved


@pytest.fixture
def dav_share(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh share dir that the app serves files from."""
    share = tmp_path / "res"
    share.mkdir()
    monkeypatch.setenv("RESOURCES_DIR", str(share))
    return share


@pytest.fixture
def spaced_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every rate limiter clock read land far after the one before it."""
    counter = [0]
    def _get_time() -> float:
        counter[0] += 1
        return counter[0] * 100.0
    monkeypatch.setattr("app.lib.shared.rate_limiter.time", MagicMock(monotonic=_get_time))


@pytest.fixture(autouse=True, scope="session")
def _pin_share_env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point RESOURCES_DIR and DATA_DIR at throwaway dirs so no test can touch the live ones."""
    share = tmp_path_factory.mktemp("share-pin")
    data = tmp_path_factory.mktemp("data-pin")
    previous_share = os.environ.get("RESOURCES_DIR")
    previous_data = os.environ.get("DATA_DIR")
    os.environ["RESOURCES_DIR"] = str(share)
    os.environ["DATA_DIR"] = str(data)
    try:
        yield
    finally:
        if previous_share is None:
            os.environ.pop("RESOURCES_DIR", None)
        else:
            os.environ["RESOURCES_DIR"] = previous_share
        if previous_data is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = previous_data


@pytest.fixture(autouse=True)
def _fast_sleeps(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub the module's asyncio.sleep for fast_sleeps tests."""
    if request.node.get_closest_marker("fast_sleeps") is None:
        return
    parts = Path(request.node.location[0]).parts
    if "tests" not in parts:
        return
    segments = list(parts[parts.index("tests") + 1 :])
    if not segments:
        return
    segments[-1] = segments[-1].removesuffix(".py").removeprefix("test_")
    module_name = ".".join(segments)
    if importlib.util.find_spec(module_name) is None:
        return
    module = import_module(module_name)
    try:
        module.asyncio
    except AttributeError:
        return
    fast = MagicMock(wraps=asyncio)
    fast.sleep = AsyncMock()
    monkeypatch.setattr(module, "asyncio", fast)


def thumb_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, RenderKind, str]]:
    """Collect the key, kind, and path of every queued thumbnail."""
    calls: list[tuple[str, RenderKind, str]] = []

    def _record(key: str, kind: RenderKind, path: str) -> bool:
        calls.append((key, kind, path))
        return True

    monkeypatch.setattr("app.lib.render.thumbs.start_thumbnail_job", _record)
    return calls


def boom_descriptor(boom_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the result feed's descriptor builder raise OSError on one path.

    Args:
        boom_path: The path the descriptor builder raises on.
        monkeypatch: Lets the test change the environment.
    """
    real = rf.build_descriptor

    async def _boom(path: str, full: str) -> dict:
        if path == boom_path:
            raise OSError("stat lost the race")
        return await real(path, full)

    monkeypatch.setattr(rf, "build_descriptor", _boom)


@lru_cache(maxsize=None)
def is_llm_available() -> bool:
    """Check if the LLM server is up, once per session."""
    llm_url = os.environ.get("REASON_BACKEND_URL")
    if not llm_url:
        return False
    try:
        response = requests.get(f"{llm_url}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def wire_context_manager(mock_mgr_class: MagicMock, mgr: AsyncMock) -> tuple[AsyncMock, AsyncMock]:
    """Set up a fake manager class so it works with async with.

    Args:
        mock_mgr_class: The patched manager class.
        mgr: What entering the manager gives back.

    Returns:
        The enter mock and the exit mock.
    """
    entered = AsyncMock(return_value=mgr)
    exited = AsyncMock(return_value=False)
    mock_mgr_class.return_value.__aenter__ = entered
    mock_mgr_class.return_value.__aexit__ = exited
    mock_mgr_class.entered = entered
    mock_mgr_class.exited = exited
    return entered, exited


def assert_browser_closed(mock_mgr_class: MagicMock) -> None:
    """Check the manager was opened once and closed once.

    Args:
        mock_mgr_class: The patched manager class.
    """
    mock_mgr_class.entered.assert_awaited_once()
    mock_mgr_class.exited.assert_awaited_once()


def pytest_configure(config: pytest.Config) -> None:
    """Set the import mode and add the test markers."""
    config.option.importmode = "importlib"
    config.addinivalue_line(
        "markers", "live: mark test to run only with real LLM server"
    )
    config.addinivalue_line(
        "markers", "timeout(seconds): mark test with a timeout"
    )

import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import override
# --- Local HTTP server used by the tests ---

_TEST_FILE_CONTENT = b"test download content for browser tests\n"

_DUMMY_PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00 \x00\x00\x00 \x08\x02\x00\x00\x00\xfc\x18\xed\xa3\x00\x00\x00(IDATx\x9c\xed\xcd1\x01\x00\x00\x08\xc30\xc0\xbfg0\xc1\xbeT@\xd3[\xd9&\xfc\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00o\x1d\xa3\xc4\x01?\x8b`#:\x00\x00\x00\x00IEND\xaeB`\x82"
_DUMMY_PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n193\n%%EOF"
_DUMMY_MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x01isomiso2avc1mp41"
_DUMMY_MP3 = b"\xff\xfb\x90\x00\x80\x00\x00\x62\x00\x00\x00\x00"
_uploaded_files: dict[str, bytes] = {}


@pytest.fixture
def pdf_bytes() -> bytes:
    """Give a PDF with red and blue marks and black text."""
    with Image.new("RGB", (240, 240), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 100, 100), fill="red")
        draw.rectangle((140, 140, 220, 220), fill="blue")
        draw.text((20, 115), "Visible PDF content", fill="black")
        output = io.BytesIO()
        image.save(output, format="PDF")
        return output.getvalue()


class _TestServerHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves files for the tests."""

    @override
    def do_GET(self) -> None:
        """Answer a GET request for one of the test paths."""
        if self.path == "/" or self.path == "":
            self._send_home_page()
        elif self.path.startswith("/download/"):
            self._handle_download()
        elif self.path == "/media/redirect-to-image":
            self._handle_redirect()
        elif self.path == "/media/test.png":
            self._send_media(_DUMMY_PNG, "image/png")
        elif self.path == "/media/test.pdf":
            self._send_media(_DUMMY_PDF, "application/pdf")
        elif self.path == "/media/test.mp4":
            self._send_media(_DUMMY_MP4, "video/mp4")
        elif self.path == "/media/test.mp3":
            self._send_media(_DUMMY_MP3, "audio/mpeg")
        elif self.path == "/page/video":
            self._send_video_page()
        elif self.path == "/page/audio":
            self._send_audio_page()
        elif self.path == "/page/long":
            self._send_long_page()
        elif self.path == "/page/nested":
            self._send_nested_page()
        elif self.path == "/page/nested-scroll":
            self._send_nested_scroll_page()
        elif self.path == "/page/visibility":
            self._send_visibility_page()
        elif self.path == "/page/fields":
            self._send_fields_page()
        elif self.path.startswith("/page/upload/"):
            self._send_upload_page()
        elif self.path.startswith("/upload/result/"):
            self._send_uploaded_file()
        elif self.path == "/status/403":
            self.send_response(403)
            self.end_headers()
        elif self.path.startswith("/nonexistent") or self.path == "/nonexistent-page-404":
            self.send_response(404)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    @override
    def do_POST(self) -> None:
        """Save a file sent from the upload test page."""
        if not self.path.startswith("/upload/file/"):
            self.send_response(404)
            self.end_headers()
            return
        token = self.path.removeprefix("/upload/file/")
        size = int(self.headers["Content-Length"])
        _uploaded_files[token] = self.rfile.read(size)
        self.send_response(200)
        self.end_headers()

    def _send_upload_page(self) -> None:
        """Serve a button that opens a file chooser and sends the chosen file."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        body = b"""<html><head><title>Upload Test</title></head><body>
            <h1>Upload a resource</h1>
            <button id="upload-button" type="button">Upload a file</button>
            <input id="file-input" type="file" hidden>
            <p id="status">No file uploaded</p>
            <script>
                const input = document.getElementById('file-input');
                document.getElementById('upload-button').onclick = () => input.click();
                input.onchange = async () => {
                    const file = input.files[0];
                    const token = location.pathname.split('/').pop();
                    const response = await fetch('/upload/file/' + token, {
                        method: 'POST', body: file,
                    });
                    document.getElementById('status').textContent = response.ok
                        ? 'Uploaded ' + file.name : 'Upload failed';
                };
            </script>
        </body></html>"""
        self.wfile.write(body)

    def _send_uploaded_file(self) -> None:
        """Return the file received from the upload test page."""
        token = self.path.removeprefix("/upload/result/")
        if token not in _uploaded_files:
            self.send_response(404)
            self.end_headers()
            return
        self._send_media(_uploaded_files[token], "application/octet-stream")

    def _send_home_page(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        body = b"""<html><head><title>Test Page</title></head><body>
            <h1>Example Domain</h1>
            <p>This is a test page for browser tests.</p>
            <a id="download_link" href="/download/test.bin">Download</a>
        </body></html>"""
        self.wfile.write(body)

    def _handle_download(self) -> None:
        filename = self.path.split("/download/", 1)[1]
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(_TEST_FILE_CONTENT)

    def _handle_redirect(self) -> None:
        self.send_response(302)
        self.send_header("Location", "/media/test.png")
        self.end_headers()

    def _send_media(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(data)

    def _send_video_page(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        body = b"""<html><head><title>Test Video</title></head><body>
            <h1>Test Video Page</h1>
            <video id="my-video" controls width="320" height="240">
                <source src="/media/test.mp4" type="video/mp4">
            </video>
        </body></html>"""
        self.wfile.write(body)

    def _send_audio_page(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        body = b"""<html><head><title>Test Audio</title></head><body>
            <h1>Test Audio Page</h1>
            <audio id="my-audio" controls>
                <source src="/media/test.mp3" type="audio/mpeg">
            </audio>
        </body></html>"""
        self.wfile.write(body)

    def _send_long_page(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        sections = ""
        for i in range(1, 21):
            sections += f'<div id="section-{i}" style="height:600px; padding:20px;">'
            sections += f'<h2>Section {i}</h2>'
            sections += f'<p>Content for section {i}</p>'
            if i == 5:
                sections += '<button id="mid-button">Click Me</button>'
            if i == 10:
                sections += '<input id="mid-input" type="text" placeholder="Type here" />'
            if i == 15:
                sections += '<select id="mid-select"><option value="a">A</option><option value="b">B</option></select>'
            sections += "</div>"
        body = f"""<html><head><title>Long Page</title></head><body>
            <div id="top-section" style="height:600px; padding:20px;">
                <h1>Top</h1>
            </div>
            {sections}
            <script>
                window.__clickLog = [];
                document.getElementById('mid-button').addEventListener('click', function() {{
                    window.__clickLog.push('mid-button');
                }});
                document.getElementById('top-section').addEventListener('click', function() {{
                    window.__clickLog.push('top-section');
                }});
            </script>
        </body></html>"""
        self.wfile.write(body.encode())

    def _send_nested_page(self) -> None:
        """Serve a page where the page itself cannot scroll, only the side drawer can.

        The drawer holds nested-bottom and nested-select, which show up only
        once the drawer is scrolled down.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        body = b"""<html><head><title>Nested Drawer Test</title>
            <style>
                body { margin: 0; overflow: hidden; }
                #doc-target { margin: 40px 0 0 20px; }
                #backdrop { height: 4000px; background: #eef; }
                #drawer {
                    position: fixed;
                    top: 0;
                    right: 0;
                    width: 480px;
                    height: 100vh;
                    overflow-y: auto;
                    border-left: 1px solid #999;
                    background: #fff;
                }
                #nested-top, #nested-bottom { height: 60px; background: #ccc; }
                .filler { height: 1900px; background: #f5f5f5; }
                #nested-select { margin: 12px 24px; }
            </style>
        </head><body>
            <p id="doc-target">Document target</p>
            <div id="backdrop">tall background</div>
            <div id="drawer">
                <div id="nested-top">Nested top</div>
                <div class="filler">filler</div>
                <div class="filler">filler</div>
                <div id="nested-bottom">Nested bottom</div>
                <select id="nested-select">
                    <option value="a">A</option>
                    <option value="b">B</option>
                    <option value="c">C</option>
                </select>
            </div>
        </body></html>"""
        self.wfile.write(body)

    def _send_nested_scroll_page(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        body = b"""<html><head><title>Nested Scroll Test</title>
            <style>
                .scroll-container {
                    overflow: auto;
                    height: 400px;
                    width: 300px;
                }
                .scroll-content {
                    height: 3000px;
                }
            </style>
        </head><body>
            <h1>Nested Scroll Test</h1>
            <div class="scroll-container">
                <div class="scroll-content">
                    <p>Nested scroll content</p>
                    <p id="deep-element">Deep element in nested scroll</p>
                </div>
            </div>
        </body></html>"""
        self.wfile.write(body)

    def _send_visibility_page(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        body = b"""<html><head><title>Visibility Test</title>
            <style>
                .hidden-display { display: none; }
                .hidden-visibility { visibility: hidden; }
            </style>
        </head><body>
            <h1>Visibility Test Page</h1>
            <p id="visible-text">Always visible content</p>
            <div id="css-hidden" class="hidden-display">
                <p>CSS hidden content should not appear</p>
            </div>
            <div id="visibility-hidden" class="hidden-visibility">
                <p>Visibility hidden content should not appear</p>
            </div>
            <div id="aria-hidden-visible">
                <p aria-hidden="true">Aria hidden but visible content</p>
            </div>
        </body></html>"""
        self.wfile.write(body)

    def _send_fields_page(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        body = b"""<html><head><title>Fields Test</title></head><body>
            <h1>Fields Test Page</h1>
            <input id="field-input" type="text" value="the quick brown fox" style="width: 400px" />
            <textarea id="field-textarea" rows="3" cols="40">first line
the lazy dog</textarea>
            <div id="field-editable" contenteditable="true" style="width: 400px; border: 1px solid">Hello <b>big</b> world</div>
            <input id="field-repeat" type="text" value="cat and cat" style="width: 400px" />
            <input id="field-email" type="email" value="jane@exmaple.com" style="width: 400px" />
            <input id="field-number" type="number" value="1234" style="width: 400px" />
            <iframe id="field-frame" style="width: 460px; height: 60px; border: 0"
                srcdoc="<body style='margin: 0'><iframe style='width: 440px; height: 60px; border: 0' srcdoc='&lt;body style=&quot;margin: 0&quot;&gt;&lt;input value=&quot;hello wrld&quot; style=&quot;width: 400px; height: 40px&quot;&gt;'></iframe>"></iframe>
            <script>
                window.inputEvents = 0;
                document.addEventListener("input", () => { window.inputEvents += 1; });
            </script>
        </body></html>"""
        self.wfile.write(body)

    @override
    def log_message(self, _format: str, *_args: str) -> None:  # type: ignore[override]
        """Keep the server from printing a line for each request."""
        pass


_download_server = None
_download_thread = None


def _start_download_server() -> str:
    """Serve the test files on a free port and give back the base URL."""
    global _download_server, _download_thread
    _download_server = HTTPServer(("127.0.0.1", 0), _TestServerHandler)
    _download_thread = threading.Thread(target=_download_server.serve_forever, daemon=True)
    _download_thread.start()
    port = _download_server.server_address[1]
    return f"http://127.0.0.1:{port}"


def _stop_download_server() -> None:
    global _download_server
    if _download_server:
        _download_server.shutdown()
        _download_server = None


@pytest.fixture(scope="module")
def test_server_url() -> str:
    """The base URL of a local HTTP server that serves test files.

    The server answers with a page holding a download link at the root, a file
    to download at /download/<name>, a small png, pdf, video, and audio file
    under /media, a 403 at /status/403, and a 404 at /nonexistent-page-404.
    """
    url = _start_download_server()
    time.sleep(0.1)
    yield url
    _stop_download_server()


# --- Browser helpers ---


@lru_cache(maxsize=None)
def is_chrome_cdp_available(cdp_url: str = "http://localhost:9222") -> bool:
    """Check if the browser debug port answers, once per session."""
    try:
        with urllib.request.urlopen(f"{cdp_url}/json/version", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


skip_if_no_chrome = pytest.mark.skipif(
    not is_chrome_cdp_available(),
    reason="Chrome CDP not available at localhost:9222",
)

@asynccontextmanager
async def e2e_page(viewport: dict) -> AsyncIterator[tuple[BrowserManager, str]]:
    """One fresh browser tab at the given size, closed on the way out."""
    mgr = await BrowserManager.create()
    tid = ""
    try:
        tid = await mgr.new_page()
        resized = await mgr.set_viewport_size(tid, viewport["width"], viewport["height"])
        assert resized, "e2e_page: viewport resize did not apply"
        yield mgr, tid
    finally:
        if tid:
            await mgr.close_page(tid)
        await mgr.cleanup()


class _TaskStore(Protocol):
    """The task record store a test gets."""


@pytest.fixture
async def task_store(tmp_path: Path) -> AsyncIterator[_TaskStore]:
    """Point the app stores at a throwaway database and give back the task store."""
    previous_task = store_module.task_store
    previous_tool = store_module.tool_store
    store_module.init_store(str(tmp_path / "records.db"))
    await store_module.task_store.open()
    yield store_module.task_store
    await store_module.task_store.close()
    store_module.task_store = previous_task
    store_module.tool_store = previous_tool


async def store_infra(tmp_path: Path) -> AsyncIterator[None]:
    """Point the process-wide stores at a throwaway database for the test."""
    previous_task = store_module.task_store
    previous_tool = store_module.tool_store
    previous_agent = store_module.agent_store
    store_module.init_store(str(tmp_path / "records.db"))
    await store_module.task_store.open()
    yield
    await store_module.task_store.close()
    store_module.task_store = previous_task
    store_module.tool_store = previous_tool
    store_module.agent_store = previous_agent


# --- Whole app fixture and seed helpers ---

# Fixed ids and times the console tests seed with.
E2E_RUNNING = "e2e-running"
E2E_FAILED_250 = "e2e-failed-250"
E2E_RESULT_BIG = "e2e-result-big"
E2E_STARTED_AT = "2026-08-23T01:00:00+00:00"
E2E_ENDED_AT = "2026-08-23T01:00:05+00:00"

_APP_HTTP_HOST = "127.0.0.1"
_APP_HTTP_START_TIMEOUT_S = 10.0
_APP_HTTP_STOP_TIMEOUT_S = 10.0
_APP_HTTP_POLL_S = 0.05
_NOVNC_HOST = "127.0.0.1"
_NOVNC_PORT = 6080
_NOVNC_TIMEOUT_S = 0.5
_PNG_DIM = 8
_PNG_BIT_DEPTH = 8
_PNG_COLOR_TYPE = 2
_PNG_RGB = (18, 52, 86)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    """Build one chunk of a PNG file."""
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


# A tiny real PNG filled with one color.
E2E_PNG = (
    b"\x89PNG\r\n\x1a\n"
    + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", _PNG_DIM, _PNG_DIM, _PNG_BIT_DEPTH, _PNG_COLOR_TYPE, 0, 0, 0))
    + _png_chunk(b"IDAT", zlib.compress(b"".join(b"\x00" + bytes(_PNG_RGB) * _PNG_DIM for _ in range(_PNG_DIM))))
    + _png_chunk(b"IEND", b"")
)


def require_novnc() -> None:
    """Skip the test when noVNC does not answer on 127.0.0.1 port 6080."""
    try:
        with socket.create_connection((_NOVNC_HOST, _NOVNC_PORT), timeout=_NOVNC_TIMEOUT_S):
            return
    except OSError:
        pytest.skip("noVNC not reachable on 127.0.0.1:6080")


async def seed_task(
    task_id: str,
    tool: str,
    started_at: str = E2E_STARTED_AT,
    status: str | None = None,
    *,
    parent_task_id: str | None = None,
    params: dict | None = None,
    error: str | None = None,
    ended_at: str | None = None,
    render_type: str | None = None,
) -> None:
    """Add one task and its tool row to the store, and finish it if a status is given.

    The store must already be open on the running loop.

    Args:
        task_id: Id of the task.
        tool: Name of the tool the task runs.
        started_at: When the task started.
        status: Final status, or None to leave the task running.
        parent_task_id: Id of the task that started this one.
        params: Params saved with the tool row.
        error: Error text saved when the task is finished.
        ended_at: When the task ended.
        render_type: Tool-declared render kind written onto the tool row.
    """
    await store_module.task_store.insert_running(task_id, parent_task_id, started_at, f"runs/{task_id}")
    await store_module.tool_store.insert(task_id, tool, None, params if params is not None else {})
    if render_type is not None:
        await store_module.tool_store.set_render_type(task_id, render_type)
    if status is not None:
        await store_module.task_store.finish(task_id, status, error=error, ended_at=ended_at)


def write_run_file(data_dir: Path | str, task_id: str, name: str, content: str | bytes) -> Path:
    """Write a file at data_dir/runs/<task_id>/<name> and give back its path.

    The run folder is made if it is not there yet.
    """
    run_dir = Path(data_dir) / "runs" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return path


def xlsx_body() -> bytes:
    """The bytes of a tiny real xlsx file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/workbook.xml", "<workbook/>")
    return buf.getvalue()


def write_share_file(share: Path | str, rel: str, data: bytes) -> Path:
    """Write bytes at <share>/<rel> and give back the path, making parent folders."""
    path = Path(share) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def log_lines(count: int) -> str:
    """Text made of count numbered log lines."""
    return "".join(f"e2e-log line {i:03d} of {count}\n" for i in range(count))


@pytest.fixture
async def app_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[SimpleNamespace]:
    """The whole app running on a real port, with its own temp store and share dir.

    Gives back base_url, port, data_dir, and share. Seed the store and the share
    after the fixture starts, not before. Do not use this fixture together with
    api_client, console_env, or task_store in the same test.
    """
    share, data = point_share_env(tmp_path, monkeypatch)
    monkeypatch.setattr(store_module, "_DB_PATH", str(data / "records.db"))
    monkeypatch.setattr(server_module, "_RESOURCES_DIR", str(share))
    server = uvicorn.Server(uvicorn.Config(server_module.build_app(), host=_APP_HTTP_HOST, port=0, log_level="warning"))
    task = asyncio.create_task(server.serve(), name="app_http")
    deadline = time.monotonic() + _APP_HTTP_START_TIMEOUT_S
    while not (server.started and server.servers):
        if task.done():
            task.result()
        if time.monotonic() >= deadline:
            task.cancel()
            await task
            raise AssertionError("app_http: uvicorn did not start in time")
        await asyncio.sleep(_APP_HTTP_POLL_S)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield SimpleNamespace(base_url=f"http://{_APP_HTTP_HOST}:{port}", port=port, data_dir=data, share=share)
    server.should_exit = True
    await asyncio.wait_for(task, timeout=_APP_HTTP_STOP_TIMEOUT_S)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        assert probe.connect_ex((_APP_HTTP_HOST, port)) != 0, "app_http: port still open after teardown"
    assert not store_module.store_is_open(), "app_http: store left open after teardown"


# --- A live, controllable console origin for browser tests ---
#
# Serves the real app on a real port, backed by the real store, with the tool
# call route stubbed. Seed and finish tasks through the store, and steer
# replies with the knobs. Browser tests navigate to this origin instead of
# faking replies.

_CONSOLE_STUB_HOST = "127.0.0.1"
_CONSOLE_STUB_START_TIMEOUT_S = 10.0
_CONSOLE_STUB_STOP_TIMEOUT_S = 10.0
_CONSOLE_STUB_POLL_S = 0.05


def _stub_endpoint(func: Callable[[Request, dict], Awaitable[Response]], last_call: dict) -> ASGIApp:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive)
        response = await func(request, last_call)
        await response(scope, receive, send)

    return app


async def _stub_tool_call(request: Request, last_call: dict) -> Response:
    tool = request.path_params["tool"]
    if request.method != "POST":
        return error_response(
            405,
            "method_not_allowed",
            f"Method '{request.method}' is not allowed at '{request.url.path}'.",
            headers={"allow": "POST"},
        )
    if tool_class(tool) is None:
        return error_response(404, "not_found", f"Unknown tool '{tool}'.")
    if "application/json" not in request.headers.get("content-type", ""):
        return error_response(415, "unsupported_media_type", "The request must be JSON with an Origin matching this host.")
    try:
        params = await request.json()
    except ValueError:
        return error_response(400, "bad_param", "The request body must be valid JSON.")
    if not isinstance(params, dict):
        return error_response(400, "bad_param", "The request body must be a JSON object of params.")
    task_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc).isoformat()
    await store_module.task_store.insert_running(task_id, None, started, f"runs/{task_id}")
    await store_module.tool_store.insert(task_id, tool, None, params)
    last_call["id"] = task_id
    return JSONResponse(
        {"task_id": task_id, "status": "running", "started_at": started, "started_at_local": iso_to_local(started)},
        status_code=202,
    )


def _stub_knob(
    pattern: str,
    status: int | None,
    code: str,
    message: str,
    delay_s: float | None,
    first_only: bool,
    body: dict | None,
    body_fn: Callable[[dict], dict] | None = None,
) -> dict:
    return {
        "pattern": pattern,
        "status": status,
        "code": code,
        "message": message,
        "delay_s": delay_s,
        "first_only": first_only,
        "body": body,
        "body_fn": body_fn,
        "calls": 0,
    }


def _stub_knob_middleware(app: ASGIApp, knobs: list[dict]) -> ASGIApp:
    async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            for knob in knobs:
                if re.search(knob["pattern"], scope["path"]):
                    knob["calls"] += 1
                    if knob["delay_s"] is not None and (not knob["first_only"] or knob["calls"] == 1):
                        await asyncio.sleep(knob["delay_s"])
                    if knob["status"] is not None:
                        await error_response(knob["status"], knob["code"], knob["message"])(scope, receive, send)
                        return
                    if knob["body"] is not None:
                        await JSONResponse(knob["body"])(scope, receive, send)
                        return
                    if knob["body_fn"] is not None:
                        await JSONResponse(knob["body_fn"](scope))(scope, receive, send)
                        return
        await app(scope, receive, send)

    return middleware


def _stub_build_app(knobs: list[dict], last_call: dict) -> ASGIApp:
    app = server_module.build_app()
    for route in app.routes:
        if isinstance(route, Mount) and route.path == "/api":
            for inner in route.app.routes:
                if isinstance(inner, Route) and inner.path == "/tools/{tool}/call":
                    inner.app = _stub_endpoint(_stub_tool_call, last_call)
    return _stub_knob_middleware(app, knobs)


async def stub_finish(
    task_id: str, status: str = "completed", *, error: str | None = None, ended_at: str | None = None
) -> None:
    """Move a running task to a finished status in the store."""
    await store_module.task_store.finish(task_id, status, error=error, ended_at=ended_at)


async def _stub_start(server: uvicorn.Server, task: asyncio.Task) -> int:
    deadline = time.monotonic() + _CONSOLE_STUB_START_TIMEOUT_S
    while not (server.started and server.servers):
        if task.done():
            task.result()
        if time.monotonic() >= deadline:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            raise AssertionError("console_stub: uvicorn did not start in time")
        await asyncio.sleep(_CONSOLE_STUB_POLL_S)
    return server.servers[0].sockets[0].getsockname()[1]


def _stub_namespace(share: Path, data: Path, knobs: list[dict], last_call: dict, port: int) -> SimpleNamespace:
    def force_status(pattern: str, status: int, *, code: str = "internal", message: str = "stub failure") -> None:
        knobs.append(_stub_knob(pattern, status, code, message, None, False, None))

    def delay(pattern: str, seconds: float, *, first_only: bool = False) -> None:
        knobs.append(_stub_knob(pattern, None, "internal", "stub failure", seconds, first_only, None))

    def pin_body(pattern: str, body: dict) -> None:
        knobs.append(_stub_knob(pattern, None, "internal", "stub failure", None, False, body))

    def pin_dynamic(pattern: str, body_fn: Callable[[dict], dict]) -> None:
        """Answer matching calls with a body built from the request scope each time."""
        knobs.append(_stub_knob(pattern, None, "internal", "stub failure", None, False, None, body_fn=body_fn))

    def last_call_id() -> str | None:
        return last_call["id"]

    return SimpleNamespace(
        base_url=f"http://{_CONSOLE_STUB_HOST}:{port}",
        port=port,
        data_dir=data,
        share=share,
        seed_task=seed_task,
        finish=stub_finish,
        force_status=force_status,
        delay=delay,
        pin_body=pin_body,
        pin_dynamic=pin_dynamic,
        last_call_id=last_call_id,
    )


@pytest.fixture
async def console_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[SimpleNamespace]:
    """The real app on a live port, with only the tool call route stubbed.

    The app lifespan opens and closes the store. Seed and finish tasks after
    the fixture starts, and steer replies with the knobs. Do not use this
    fixture together with app_http, api_client, or console_env in the same test.
    """
    share, data = point_share_env(tmp_path, monkeypatch)
    monkeypatch.setattr(store_module, "_DB_PATH", str(data / "records.db"))
    monkeypatch.setattr(server_module, "_RESOURCES_DIR", str(share))
    knobs: list[dict] = []
    last_call = {"id": None}
    server = uvicorn.Server(
        uvicorn.Config(_stub_build_app(knobs, last_call), host=_CONSOLE_STUB_HOST, port=0, log_level="warning")
    )
    task = asyncio.create_task(server.serve(), name="console_stub")
    port = await _stub_start(server, task)
    monkeypatch.setenv("DAV_BASE_URL", f"http://{_CONSOLE_STUB_HOST}:{port}")
    yield _stub_namespace(share, data, knobs, last_call, port)
    server.should_exit = True
    await asyncio.wait_for(task, timeout=_CONSOLE_STUB_STOP_TIMEOUT_S)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        assert probe.connect_ex((_CONSOLE_STUB_HOST, port)) != 0, "console_stub: port still open after teardown"
    assert not store_module.store_is_open(), "console_stub: store left open after teardown"
