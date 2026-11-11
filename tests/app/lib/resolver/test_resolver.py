#!/usr/bin/env python3
"""Test suite for resolver module."""

import os
import sys
from io import StringIO
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import app.lib.resolver.resolver as resolver
from app.lib.laya.client import LayaError, Verdict
from app.lib.shared import paths
from app.lib.resolver.resolver import ResolveError, resolve_input
from tests.app.lib.conftest import mock_cli_proc
from tests.app.lib.resolver.conftest import make_docx, make_xls, make_xlsx

_TEMP_DIR = "/dev/shm/agent/resolver"

_THIS_MODULE = sys.modules[__name__]


@pytest.fixture(autouse=True)
def _isolated_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test its own scratch root, so the resolver stages files under it."""
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("SCRATCH_DIR", str(scratch))
    monkeypatch.setattr(_THIS_MODULE, "_TEMP_DIR", str(scratch / "resolver"))


def _mock_describe(result: str, status: int = 0) -> tuple[str, int]:
    """Return a result and status pair for a stand in describer."""
    return (result, status)


def _cli_proc(returncode: int, stdout_str: str, stderr_str: str = "") -> MagicMock:
    """Build a stand in child process with the given output and exit code."""
    return mock_cli_proc(stdout=stdout_str, stderr=stderr_str, returncode=returncode)


def _mock_run_exiftool(mime_type: str) -> MagicMock:
    """Build a stand in that reports the given file type and fails downloads."""
    mock = MagicMock()
    def _run_side(*cmd: str, **_kwargs: int | None) -> MagicMock:
        if cmd and cmd[0] == "exiftool":
            return _cli_proc(0, mime_type)
        return _cli_proc(22, "", "curl: (22) error")
    mock.side_effect = _run_side
    return mock


def _mock_curl_binary_download(content_disposition: str) -> MagicMock:
    """Build a stand in that downloads a zip file with the given header."""
    binary = b"\xff\xfe\x00\x01\x02\x03\x04"

    def _run_side(*cmd: str, **_kwargs: int | None) -> MagicMock:
        if cmd and cmd[0] == "curl":
            if "-D" in cmd:
                headers = (
                    "HTTP/1.1 200 OK\r\n"
                    f"{content_disposition}\r\n"
                    "\r\n"
                )
                return _cli_proc(0, headers + "200")
            out = cmd[cmd.index("-o") + 1]
            with open(out, "wb") as f:
                f.write(binary)
            return _cli_proc(0, "")
        if cmd and cmd[0] == "exiftool":
            return _cli_proc(0, "application/zip")
        return _cli_proc(22, "", "curl: (22) error")
    return _run_side


_ARTICLE_HTML = "<html><body><p>" + "Real article text. " * 60 + "</p></body></html>"

_NO_TEXT_HTML = (
    "<html><head><title>App</title></head><body>"
    + "".join(f'<link rel="preload" as="script" href="//cdn.example.com/static/js/chunk-{i}.js">' for i in range(20))
    + "<script>window.app = 1;</script><div id=\"app\"></div></body></html>"
)


def _write_curl_download(name: str, body: str) -> None:
    """Put the file a stand in curl download would save on disk."""
    os.makedirs(_TEMP_DIR, exist_ok=True)
    with open(os.path.join(_TEMP_DIR, name), "w") as f:
        f.write(body)


def _curl_download_procs(mime: str) -> list[MagicMock]:
    """Stand in results for a curl status check, a download, and two type checks."""
    return [_cli_proc(0, "200"), _cli_proc(0, ""), _cli_proc(0, mime), _cli_proc(0, mime)]


def _setup_content_type_mgr(
    mock_browser_mgr: AsyncMock, url: str, content_type: str, body: str
) -> AsyncMock:
    """Set up a fake browser that reports a content type and then a body."""
    mgr = AsyncMock()
    mock_browser_mgr.create = AsyncMock(return_value=mgr)
    mgr.new_page.return_value = "tab1"
    mgr.wait_for_download.return_value = {"downloaded": False}
    mgr.url = MagicMock(return_value=url)
    mgr.evaluate.side_effect = [content_type, body]
    mgr.close_page.return_value = True
    mgr.cleanup.return_value = None
    return mgr


def _setup_download_mgr(mock_browser_mgr: AsyncMock, saved_path: str) -> AsyncMock:
    """Set up a fake browser that downloads the page to the given file."""
    mgr = AsyncMock()
    mock_browser_mgr.create = AsyncMock(return_value=mgr)
    mgr.new_page.return_value = "tab1"
    mgr.wait_for_download.return_value = {"downloaded": True, "saved_path": saved_path}
    mgr.close_page.return_value = True
    mgr.cleanup.return_value = None
    return mgr


def _setup_mock_browser_mgr(mock_browser_mgr: AsyncMock, body_content: str = "fallback content") -> AsyncMock:
    """Set up a fake browser that gives back the given page text."""
    mgr = AsyncMock()
    mock_browser_mgr.create = AsyncMock(return_value=mgr)
    mgr.new_page.return_value = "tab1"
    mgr.wait_for_download.return_value = {"downloaded": False}
    mgr.url = MagicMock(return_value="http://example.com")
    mgr.evaluate.return_value = "text/html"
    snap = MagicMock()
    snap.filtered_tree = AsyncMock(return_value={"html": {"body": {"children": [body_content]}}})
    mgr.snapshot.return_value = snap
    mgr.close_page.return_value = True
    mgr.cleanup.return_value = None
    return mgr


# ---------------------------------------------------------------------------
# 1. Plain text passthrough
# ---------------------------------------------------------------------------

class TestPlainTextPassthrough:

    async def test_resolve_input_none(self) -> None:

        with pytest.raises(ResolveError) as exc:
            await resolve_input(None)
        assert "Input text is None" in str(exc.value)

    async def test_resolve_input_empty_string(self) -> None:

        result, is_processed = await resolve_input("")
        assert result == ""
        assert is_processed is False

    async def test_resolve_input_plain_text(self) -> None:

        result, is_processed = await resolve_input("hello world")
        assert result == "hello world"
        assert is_processed is False

    async def test_resolve_input_html_text(self) -> None:

        html_str = "<html><body><h1>Hello</h1></body></html>"
        result, is_processed = await resolve_input(html_str)
        assert result == html_str
        assert is_processed is False


# ---------------------------------------------------------------------------
# 2. Local file handling
# ---------------------------------------------------------------------------

class TestLocalFileHandling:

    @patch('app.lib.resolver.resolver.describe_pdf')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_pdf_file(self, mock_run: MagicMock, mock_describe_pdf: MagicMock, dummy_pdf_path: str) -> None:

        mock_run.return_value = _cli_proc(0, "application/pdf")
        mock_describe_pdf.return_value = _mock_describe("This is the PDF description.")
        result, is_processed = await resolve_input(dummy_pdf_path)
        assert "PDF description" in result
        assert is_processed is True

    @patch('app.lib.resolver.resolver.describe_pdf')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_pdf_describe_failure(self, mock_run: MagicMock, mock_describe_pdf: MagicMock, dummy_pdf_path: str) -> None:

        """A PDF that cannot be described gives its error message back."""
        mock_run.return_value = _cli_proc(0, "application/pdf")
        mock_describe_pdf.return_value = _mock_describe("Could not determine PDF page count.", 1)
        with pytest.raises(ResolveError) as exc:
            await resolve_input(dummy_pdf_path)
        assert str(exc.value) == "Could not determine PDF page count."

    @patch('app.lib.resolver.resolver.describe_image')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_image_file(self, mock_run: MagicMock, mock_describe_image: MagicMock, dummy_image_path: str) -> None:

        mock_run.return_value = _cli_proc(0, "image/jpeg")
        mock_describe_image.return_value = _mock_describe("A red square image.")
        result, is_processed = await resolve_input(dummy_image_path)
        assert "red square" in result
        assert is_processed is True

    @patch('app.lib.resolver.resolver.describe_image')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_image_describe_failure(self, mock_run: MagicMock, mock_describe_image: MagicMock, dummy_image_path: str) -> None:

        """An image that cannot be described gives its error message back."""
        mock_run.return_value = _cli_proc(0, "image/jpeg")
        mock_describe_image.return_value = _mock_describe("The image is too large.", 1)
        with pytest.raises(ResolveError) as exc:
            await resolve_input(dummy_image_path)
        assert str(exc.value) == "The image is too large."

    @patch('app.lib.resolver.resolver.describe_video')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_video_file(self, mock_run: MagicMock, mock_describe_video: MagicMock, dummy_video_path: str) -> None:

        mock_run.return_value = _cli_proc(0, "video/mp4")
        mock_describe_video.return_value = _mock_describe("A short test video.")
        result, is_processed = await resolve_input(dummy_video_path)
        assert "short test video" in result
        assert is_processed is True

    @patch('app.lib.resolver.resolver.describe_video')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_video_describe_failure(self, mock_run: MagicMock, mock_describe_video: MagicMock, dummy_video_path: str) -> None:

        """A video that cannot be described gives its error message back."""
        mock_run.return_value = _cli_proc(0, "video/mp4")
        mock_describe_video.return_value = _mock_describe("The video is too long.", 1)
        with pytest.raises(ResolveError) as exc:
            await resolve_input(dummy_video_path)
        assert str(exc.value) == "The video is too long."

    @patch('app.lib.resolver.resolver.httpx.AsyncClient')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_audio_file(self, mock_run: MagicMock, mock_client_cls: MagicMock, dummy_audio_path: str,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
        """An audio file is sent to the ASR server and its words come back."""
        monkeypatch.setenv("ASR_SERVER_URL", "http://asr.example.com/")
        mock_run.return_value = _cli_proc(0, "audio/wav")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "This is the transcription."
        client = MagicMock()
        client.post = AsyncMock(return_value=mock_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = client
        result, _ = await resolve_input(dummy_audio_path)
        assert result == "This is the transcription."
        assert client.post.await_args.args[0] == "http://asr.example.com/asr"

    @pytest.mark.parametrize("value", [None, "", "  "], ids=["unset", "empty", "blank"])
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_audio_without_asr_server(self, mock_run: MagicMock, value: str | None,
                                                          dummy_audio_path: str, monkeypatch: pytest.MonkeyPatch,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
        """An audio file with no ASR server set fails with a message that names the variable."""
        if value is None:
            monkeypatch.delenv("ASR_SERVER_URL", raising=False)
        else:
            monkeypatch.setenv("ASR_SERVER_URL", value)
        mock_run.return_value = _cli_proc(0, "audio/wav")
        with pytest.raises(ResolveError) as exc:
            await resolve_input(dummy_audio_path)
        assert str(exc.value) == "ASR_SERVER_URL is not set: needed by audio and video transcription"
        assert "[config] ASR_SERVER_URL is not set: needed by audio and video transcription\n" in capsys.readouterr().err

    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_docx_file(self, mock_run: MagicMock, tmp_path: Path) -> None:

        docx_path = str(tmp_path / "doc.docx")
        make_docx(docx_path, ["Docx body text."], [["Alpha", "Beta"]])
        mock_run.return_value = _cli_proc(0, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        result, is_processed = await resolve_input(docx_path)
        assert "Docx body text." in result
        assert "Alpha" in result
        assert "Beta" in result
        assert is_processed is False

    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_xlsx_file(self, mock_run: MagicMock, tmp_path: Path) -> None:

        xlsx_path = str(tmp_path / "book.xlsx")
        make_xlsx(xlsx_path, [("Data", [["City", "Population"], ["Osaka", 2700000]])])
        mock_run.return_value = _cli_proc(0, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        result, is_processed = await resolve_input(xlsx_path)
        assert "Sheet: Data" in result
        assert "City\tPopulation" in result
        assert "Osaka\t2700000" in result
        assert is_processed is False

    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_xls_file(self, mock_run: MagicMock, tmp_path: Path) -> None:

        xls_path = str(tmp_path / "legacy.xls")
        make_xls(xls_path, "Data", [["Key", "Value"], ["Rows", 3]])
        mock_run.return_value = _cli_proc(0, "application/vnd.ms-excel")
        result, is_processed = await resolve_input(xls_path)
        assert "Sheet: Data" in result
        assert "Key\tValue" in result
        assert "Rows\t3" in result
        assert is_processed is False

    @patch('app.lib.resolver.resolver.describe_presentation')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_pptx_file(self, mock_run: MagicMock, mock_describe: MagicMock, tmp_path: Path) -> None:

        pptx_path = tmp_path / "slides.pptx"
        pptx_path.write_bytes(b"PK\x03\x04 dummy")
        mock_run.return_value = _cli_proc(0, "application/vnd.openxmlformats-officedocument.presentationml.presentation")
        mock_describe.return_value = _mock_describe("Slides described.")
        result, is_processed = await resolve_input(str(pptx_path))
        assert "Slides described." in result
        assert is_processed is True

    @patch('app.lib.resolver.resolver.describe_presentation')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_ppt_file(self, mock_run: MagicMock, mock_describe: MagicMock, tmp_path: Path) -> None:

        ppt_path = tmp_path / "legacy.ppt"
        ppt_path.write_bytes(b"PK\x03\x04 dummy")
        mock_run.return_value = _cli_proc(0, "application/vnd.ms-powerpoint")
        mock_describe.return_value = _mock_describe("Slides described.")
        result, is_processed = await resolve_input(str(ppt_path))
        assert "Slides described." in result
        assert is_processed is True

    @patch('app.lib.resolver.resolver.describe_presentation')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_pptx_describe_failure(self, mock_run: MagicMock, mock_describe: MagicMock, tmp_path: Path) -> None:

        """A slide deck that cannot be described gives its error message back."""
        pptx_path = tmp_path / "broken.pptx"
        pptx_path.write_bytes(b"PK\x03\x04 dummy")
        mock_run.return_value = _cli_proc(0, "application/vnd.openxmlformats-officedocument.presentationml.presentation")
        mock_describe.return_value = _mock_describe("The slides could not be described.", 1)
        with pytest.raises(ResolveError) as exc:
            await resolve_input(str(pptx_path))
        assert str(exc.value) == "The slides could not be described."

    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_docx_parse_error(self, mock_run: MagicMock, tmp_path: Path) -> None:

        docx_path = tmp_path / "broken.docx"
        docx_path.write_bytes(b"garbage not a docx")
        mock_run.return_value = _cli_proc(0, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        with pytest.raises(ResolveError) as exc:
            await resolve_input(str(docx_path))
        assert "Error extracting text" in str(exc.value)

    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_html_file(self, mock_run: MagicMock, dummy_html_path: str) -> None:

        mock_run.return_value = _cli_proc(0, "text/html")
        result, is_processed = await resolve_input(dummy_html_path)
        assert result
        assert is_processed is False

    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_html_file_without_text_fails(self, mock_run: MagicMock, tmp_path: Path) -> None:

        """A local HTML file with no text fails."""
        html_path = tmp_path / "empty.html"
        html_path.write_text(_NO_TEXT_HTML)
        mock_run.return_value = _cli_proc(0, "text/html")

        with pytest.raises(ResolveError) as exc:
            await resolve_input(str(html_path))
        assert str(exc.value) == f"Failed to extract text from HTML: {html_path}"

    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_text_file(self, mock_run: MagicMock, dummy_text_path: str) -> None:

        mock_run.return_value = _cli_proc(0, "text/plain")
        result, _ = await resolve_input(dummy_text_path)
        assert "test file content" in result

    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_binary_file_saves_copy_and_returns_notice(
        self, mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A binary file is copied to the share and the reply says how to download it."""
        bin_file = tmp_path / "archive.zip"
        bin_file.write_bytes(b"\xff\xfe\x00\x01\x02\x03\x04")
        resources_dir = tmp_path / "resources"
        monkeypatch.setenv("RESOURCES_DIR", str(resources_dir))
        monkeypatch.setenv("DAV_BASE_URL", "http://example.com:8315")
        mock_run.return_value = _cli_proc(0, "application/zip")

        result, is_processed = await resolve_input(str(bin_file))

        saved = resources_dir / "downloads" / "archive.zip"
        assert saved.read_bytes() == b"\xff\xfe\x00\x01\x02\x03\x04"
        assert is_processed is False
        assert result == (
            "Binary file with size 7 bytes. "
            "Please use WebDAV to download it instead:\n"
            "  curl -o /tmp/archive.zip http://example.com:8315/dav/downloads/archive.zip"
        )

    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_binary_file_copy_failure_is_an_error(
        self, mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A binary file that cannot be copied gives an error."""
        bin_file = tmp_path / "archive.zip"
        bin_file.write_bytes(b"\xff\xfe\x00\x01\x02\x03\x04")
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        monkeypatch.setenv("RESOURCES_DIR", str(blocker / "resources"))
        mock_run.return_value = _cli_proc(0, "application/zip")

        with pytest.raises(ResolveError) as exc:
            await resolve_input(str(bin_file))

        assert "Error saving binary file" in str(exc.value)

    async def test_resolve_input_file_not_found(self) -> None:

        result, is_processed = await resolve_input("/nonexistent/path.pdf")
        assert result == "/nonexistent/path.pdf"
        assert is_processed is False


# ---------------------------------------------------------------------------
# 3. URL live tests
# ---------------------------------------------------------------------------

class TestURLLive:

    @pytest.mark.live
    async def test_resolve_url_html_page(self, test_server_url: str) -> None:

        result, _ = await resolve_input(test_server_url)
        assert len(result) > 0

    @pytest.mark.live
    async def test_resolve_url_404(self, test_server_url: str) -> None:

        with pytest.raises(ResolveError) as exc:
            await resolve_input(f"{test_server_url}/nonexistent-page-404")
        assert "404" in str(exc.value)

    @pytest.mark.live
    async def test_resolve_url_403(self, test_server_url: str) -> None:

        with pytest.raises(ResolveError):
            await resolve_input(f"{test_server_url}/status/403")

    @pytest.mark.live
    async def test_resolve_url_pdf_download(self, test_server_url: str) -> None:

        result, is_processed = await resolve_input(f"{test_server_url}/media/test.pdf")
        assert is_processed is True
        assert len(result) > 0

    @pytest.mark.live
    async def test_resolve_url_image_download(self, test_server_url: str) -> None:

        result, is_processed = await resolve_input(f"{test_server_url}/media/test.png")
        assert is_processed is True
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 4. URL flows with stand ins
# ---------------------------------------------------------------------------

class TestURLMocked:

    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_browser_fallback_on_curl_fail(self, mock_run: MagicMock, mock_browser_mgr: AsyncMock) -> None:

        mock_proc = _cli_proc(22, "", "curl: (22) The requested URL returned error: 503")
        mock_run.return_value = mock_proc

        _setup_mock_browser_mgr(mock_browser_mgr, "browser content")

        result, _ = await resolve_input("http://example.com")
        assert "browser content" in result

    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_browse_failure_passes_message(self, mock_run: MagicMock, mock_browser_mgr: AsyncMock) -> None:

        """An error from the open page comes back as its own message."""
        mock_run.return_value = _cli_proc(22, "", "curl: (22) error")

        mgr = AsyncMock()
        mock_browser_mgr.create = AsyncMock(return_value=mgr)
        mgr.new_page.return_value = "tab1"
        mgr.wait_for_download = AsyncMock(side_effect=Exception("Navigation failed for http://example.com"))
        mgr.close_page.return_value = True
        mgr.cleanup.return_value = None

        with pytest.raises(ResolveError) as exc:
            await resolve_input("http://example.com")
        assert str(exc.value) == "Navigation failed for http://example.com"

    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_browser_create_failure_is_prefixed(self, mock_run: MagicMock, mock_browser_mgr: AsyncMock) -> None:

        """A browser start failure is shown with the resolving prefix."""
        mock_run.return_value = _cli_proc(22, "", "curl: (22) error")
        mock_browser_mgr.create = AsyncMock(side_effect=Exception("Chrome launch failed"))

        with pytest.raises(ResolveError) as exc:
            await resolve_input("http://example.com")
        assert str(exc.value) == "Error resolving URL: Chrome launch failed"

    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_insufficient_content_triggers_browser(self, mock_run: MagicMock, mock_browser_mgr: AsyncMock) -> None:

        short_html = "<html><body><p>Loading, please wait.</p></body></html>"

        # The downloaded page is already on disk.
        download_path = os.path.join(_TEMP_DIR, "insufficient.html")
        os.makedirs(_TEMP_DIR, exist_ok=True)
        with open(download_path, "w") as f:
            f.write(short_html)

        mock_run.side_effect = [
            _cli_proc(0, "200"),
            _cli_proc(0, ""),
            _cli_proc(0, "text/html"),
        ]

        _setup_mock_browser_mgr(mock_browser_mgr, "browser rendered")

        result, _ = await resolve_input("http://example.com/insufficient.html")
        assert "browser rendered" in result
        os.unlink(download_path)

    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_short_text_triggers_browser(self, mock_run: MagicMock, mock_browser_mgr: AsyncMock) -> None:

        short_text = "Loading, please wait."

        download_path = os.path.join(_TEMP_DIR, "short_text")
        os.makedirs(_TEMP_DIR, exist_ok=True)
        with open(download_path, "w") as f:
            f.write(short_text)

        mock_run.side_effect = [
            _cli_proc(0, "200"),
            _cli_proc(0, ""),
            _cli_proc(0, "text/plain"),
        ]

        _setup_mock_browser_mgr(mock_browser_mgr, "browser rendered content")

        result, _ = await resolve_input("http://example.com/short_text")
        assert "browser rendered content" in result
        os.unlink(download_path)

    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_sufficient_content_skips_browser(self, mock_run: MagicMock, mock_browser_mgr: AsyncMock) -> None:

        long_text = "A" * 1000

        download_path = os.path.join(_TEMP_DIR, "sufficient")
        os.makedirs(_TEMP_DIR, exist_ok=True)
        with open(download_path, "w") as f:
            f.write(long_text)

        mock_run.side_effect = [
            _cli_proc(0, "200"),
            _cli_proc(0, ""),
            _cli_proc(0, "text/plain"),
            _cli_proc(0, "text/plain"),
        ]

        result, _ = await resolve_input("http://example.com/sufficient")
        mock_browser_mgr.create.assert_not_called()
        assert long_text in result
        os.unlink(download_path)

    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_large_page_with_stub_phrase_skips_browser(self, mock_run: MagicMock, mock_browser_mgr: AsyncMock) -> None:

        """A long page that mentions a stub phrase is kept, not re-rendered."""
        large_text = "The build failed, so please try again with a clean cache. " * 400

        download_path = os.path.join(_TEMP_DIR, "large_page")
        os.makedirs(_TEMP_DIR, exist_ok=True)
        with open(download_path, "w") as f:
            f.write(large_text)

        mock_run.side_effect = [
            _cli_proc(0, "200"),
            _cli_proc(0, ""),
            _cli_proc(0, "text/plain"),
            _cli_proc(0, "text/plain"),
        ]

        result, _ = await resolve_input("http://example.com/large_page")
        mock_browser_mgr.create.assert_not_called()
        assert large_text in result
        os.unlink(download_path)

    @pytest.mark.parametrize("mime", ["text/html", "application/xhtml+xml"])
    @patch('app.lib.laya.client.classify', new_callable=AsyncMock)
    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_html_accepted_by_laya_skips_browser(
        self, mock_run: MagicMock, mock_browser_mgr: AsyncMock, mock_classify: AsyncMock, mime: str
    ) -> None:

        """An HTML page Laya calls usable is kept without the browser."""
        _write_curl_download("page.html", _ARTICLE_HTML)
        mock_run.side_effect = _curl_download_procs(mime)
        mock_classify.return_value = Verdict(blocked=False, blocked_probability=0.03)

        result, _ = await resolve_input("http://example.com/page.html")

        assert "Real article text." in result
        mock_classify.assert_awaited_once_with(_ARTICLE_HTML)
        mock_browser_mgr.create.assert_not_called()

    @patch('app.lib.laya.client.classify', new_callable=AsyncMock)
    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_html_blocked_by_laya_uses_browser(
        self, mock_run: MagicMock, mock_browser_mgr: AsyncMock, mock_classify: AsyncMock
    ) -> None:

        """An HTML page Laya calls blocked is read with the browser."""
        _write_curl_download("page.html", _ARTICLE_HTML)
        mock_run.side_effect = _curl_download_procs("text/html")
        mock_classify.return_value = Verdict(blocked=True, blocked_probability=0.97)
        _setup_mock_browser_mgr(mock_browser_mgr, "browser rendered page")

        result, _ = await resolve_input("http://example.com/page.html")

        assert "browser rendered page" in result
        assert "Real article text." not in result
        mock_classify.assert_awaited_once_with(_ARTICLE_HTML)

    @patch('app.lib.laya.client.classify', new_callable=AsyncMock)
    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_laya_error_uses_browser(
        self, mock_run: MagicMock, mock_browser_mgr: AsyncMock, mock_classify: AsyncMock
    ) -> None:

        """An HTML page Laya cannot check is read with the browser."""
        _write_curl_download("page.html", _ARTICLE_HTML)
        mock_run.side_effect = _curl_download_procs("text/html")
        mock_classify.side_effect = LayaError("Laya rejected the page: HTTP 503 busy")
        _setup_mock_browser_mgr(mock_browser_mgr, "browser rendered page")

        result, _ = await resolve_input("http://example.com/page.html")

        assert "browser rendered page" in result
        assert "Real article text." not in result

    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_html_without_laya_setting_fails(
        self, mock_run: MagicMock, mock_browser_mgr: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        """An HTML page fails to resolve when the Laya address is not set."""
        monkeypatch.delenv("LAYA_SERVER_URL", raising=False)
        _write_curl_download("page.html", _ARTICLE_HTML)
        mock_run.side_effect = _curl_download_procs("text/html")

        with pytest.raises(ResolveError, match="LAYA_SERVER_URL"):
            await resolve_input("http://example.com/page.html")
        mock_browser_mgr.create.assert_not_called()

    @patch('app.lib.laya.client.classify', new_callable=AsyncMock)
    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_short_html_skips_laya(
        self, mock_run: MagicMock, mock_browser_mgr: AsyncMock, mock_classify: AsyncMock
    ) -> None:

        """An HTML page with too little text goes to the browser without asking Laya."""
        _write_curl_download("page.html", "<html><body><p>Loading, please wait.</p></body></html>")
        mock_run.side_effect = _curl_download_procs("text/html")
        _setup_mock_browser_mgr(mock_browser_mgr, "browser rendered page")

        result, _ = await resolve_input("http://example.com/page.html")

        assert "browser rendered page" in result
        mock_classify.assert_not_called()
        mock_browser_mgr.create.assert_called_once()

    @patch('app.lib.laya.client.classify', new_callable=AsyncMock)
    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_html_without_text_uses_browser(
        self, mock_run: MagicMock, mock_browser_mgr: AsyncMock, mock_classify: AsyncMock
    ) -> None:

        """An HTML page with only scripts and links goes to the browser without asking Laya."""
        _write_curl_download("food", _NO_TEXT_HTML)
        mock_run.side_effect = _curl_download_procs("text/html")
        _setup_mock_browser_mgr(mock_browser_mgr, "browser rendered page")

        result, _ = await resolve_input("http://example.com/food")

        assert "browser rendered page" in result
        mock_classify.assert_not_called()
        mock_browser_mgr.create.assert_called_once()

    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_browser_html_without_text_fails(
        self, mock_run: MagicMock, mock_browser_mgr: AsyncMock
    ) -> None:

        """An HTML file with no text that the browser downloads still fails."""
        mock_run.side_effect = _mock_run_exiftool("text/html").side_effect
        download_path = os.path.join(_TEMP_DIR, "food.html")
        _write_curl_download("food.html", _NO_TEXT_HTML)

        _setup_download_mgr(mock_browser_mgr, download_path)

        with pytest.raises(ResolveError) as exc:
            await resolve_input("http://example.com/food.html")
        assert str(exc.value) == f"Failed to extract text from HTML: {download_path}"

    @pytest.mark.parametrize(
        "mime,body",
        [("text/plain", "Plain words. " * 100), ("application/json", '{"items": [1, 2, 3]}')],
        ids=["text-plain", "json"],
    )
    @patch('app.lib.laya.client.classify', new_callable=AsyncMock)
    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_non_html_text_skips_laya(
        self, mock_run: MagicMock, mock_browser_mgr: AsyncMock, mock_classify: AsyncMock, mime: str, body: str
    ) -> None:

        """A downloaded text file that is not HTML is kept without asking Laya."""
        _write_curl_download("data", body)
        mock_run.side_effect = _curl_download_procs(mime)

        result, _ = await resolve_input("http://example.com/data")

        assert result == body
        mock_classify.assert_not_called()
        mock_browser_mgr.create.assert_not_called()

    @patch('app.lib.resolver.resolver.describe_pdf')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_pdf_via_curl(self, mock_run: MagicMock, mock_describe_pdf: MagicMock) -> None:

        pdf_content = b"%PDF-1.4 dummy pdf content here"
        download_path = os.path.join(_TEMP_DIR, "doc.pdf")
        os.makedirs(_TEMP_DIR, exist_ok=True)
        with open(download_path, "wb") as f:
            f.write(pdf_content)

        mock_run.side_effect = [
            _cli_proc(0, "200"),
            _cli_proc(0, ""),
            _cli_proc(0, "application/pdf"),
            _cli_proc(0, "application/pdf"),
        ]
        mock_describe_pdf.return_value = _mock_describe("PDF content description.")

        result, _ = await resolve_input("http://example.com/doc.pdf")
        assert "PDF content description" in result
        try:
            os.unlink(download_path)
        except FileNotFoundError:
            pass

    @patch('app.lib.resolver.resolver.describe_pdf')
    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_curl_pdf_describe_failure_fails_at_once(
        self, mock_run: MagicMock, mock_browser_mgr: AsyncMock, mock_describe_pdf: MagicMock
    ) -> None:

        """A saved file that cannot be described is not retried in the browser."""
        pdf_path = os.path.join(_TEMP_DIR, "fail.pdf")
        os.makedirs(_TEMP_DIR, exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 dummy pdf content here")

        mock_run.side_effect = [
            _cli_proc(0, "200"),
            _cli_proc(0, ""),
            _cli_proc(0, "application/pdf"),
            _cli_proc(0, "application/pdf"),
        ]
        mock_describe_pdf.return_value = _mock_describe("Could not determine PDF page count.", 1)

        with pytest.raises(ResolveError) as exc:
            await resolve_input("http://example.com/fail.pdf")
        assert str(exc.value) == "Could not determine PDF page count."
        mock_browser_mgr.create.assert_not_called()
        try:
            os.unlink(pdf_path)
        except FileNotFoundError:
            pass

    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_curl_uses_content_disposition_filename(
        self, mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        """The file name the server offers is used for the saved file."""
        binary = b"\xff\xfe\x00\x01\x02\x03\x04"
        resources_dir = tmp_path / "resources"
        monkeypatch.setenv("RESOURCES_DIR", str(resources_dir))
        monkeypatch.setenv("DAV_BASE_URL", "http://example.com:8315")
        mock_run.side_effect = _mock_curl_binary_download(
            'content-disposition: attachment; filename="Hello-World-master.zip"'
        )

        result, is_processed = await resolve_input("http://example.com/Hello-World/zip/master")

        saved = resources_dir / "downloads" / "Hello-World-master.zip"
        assert saved.read_bytes() == binary
        assert is_processed is False
        assert result == (
            "Binary file with size 7 bytes. "
            "Please use WebDAV to download it instead:\n"
            "  curl -o /tmp/Hello-World-master.zip "
            "http://example.com:8315/dav/downloads/Hello-World-master.zip"
        )
        os.unlink(os.path.join(_TEMP_DIR, "Hello-World-master.zip"))

    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_curl_uses_rfc5987_encoded_filename(
        self, mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        """A file name sent in escaped form is decoded before saving."""
        binary = b"\xff\xfe\x00\x01\x02\x03\x04"
        resources_dir = tmp_path / "resources"
        monkeypatch.setenv("RESOURCES_DIR", str(resources_dir))
        monkeypatch.setenv("DAV_BASE_URL", "http://example.com:8315")
        mock_run.side_effect = _mock_curl_binary_download(
            "content-disposition: attachment; filename*=UTF-8''%E4%B8%AD%E6%96%87.zip"
        )

        result, is_processed = await resolve_input("http://example.com/download")

        saved = resources_dir / "downloads" / "中文.zip"
        assert saved.read_bytes() == binary
        assert is_processed is False
        assert result == (
            "Binary file with size 7 bytes. "
            "Please use WebDAV to download it instead:\n"
            "  curl -o /tmp/中文.zip "
            "http://example.com:8315/dav/downloads/中文.zip"
        )
        os.unlink(os.path.join(_TEMP_DIR, "中文.zip"))

    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_browser_uses_suggested_filename(
        self, mock_run: MagicMock, mock_browser_mgr: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        """The file name the browser suggests is used for the saved file."""
        mock_run.side_effect = _mock_run_exiftool("application/zip").side_effect
        resources_dir = tmp_path / "resources"
        monkeypatch.setenv("RESOURCES_DIR", str(resources_dir))
        monkeypatch.setenv("DAV_BASE_URL", "http://example.com:8315")

        download_path = os.path.join(_TEMP_DIR, "master")
        os.makedirs(_TEMP_DIR, exist_ok=True)
        with open(download_path, "wb") as f:
            f.write(b"\xff\xfe\x00\x01\x02\x03\x04")

        mgr = AsyncMock()
        mock_browser_mgr.create = AsyncMock(return_value=mgr)
        mgr.new_page.return_value = "tab1"
        mgr.wait_for_download.return_value = {"downloaded": True, "filename": "Hello-World-master.zip"}
        mgr.close_page.return_value = True
        mgr.cleanup.return_value = None

        result, _ = await resolve_input("http://example.com/Hello-World/zip/master")

        saved = resources_dir / "downloads" / "Hello-World-master.zip"
        assert saved.read_bytes() == b"\xff\xfe\x00\x01\x02\x03\x04"
        assert "dav/downloads/Hello-World-master.zip" in result
        try:
            os.unlink(os.path.join(_TEMP_DIR, "Hello-World-master.zip"))
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# 5. Context parameter
# ---------------------------------------------------------------------------

class TestContextParameter:

    @patch('app.lib.resolver.resolver.describe_pdf')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_input_context_passed_to_pdf(self, mock_run: MagicMock, mock_describe_pdf: MagicMock, dummy_pdf_path: str) -> None:

        mock_run.return_value = _cli_proc(0, "application/pdf")
        mock_describe_pdf.return_value = _mock_describe("Described with context.")
        result, _ = await resolve_input(dummy_pdf_path, context="my context")
        assert "Described with context." in result
        assert mock_describe_pdf.call_count == 1
        assert mock_describe_pdf.call_args[0][1] == "my context"


# ---------------------------------------------------------------------------
# 6. Standard input, server errors, and browser downloads
# ---------------------------------------------------------------------------

class TestMissingCodePaths:

    async def test_resolve_input_stdin(self) -> None:

        """A dash reads the text from standard input."""
        original_stdin = sys.stdin
        sys.stdin = StringIO("stdin content here")
        try:
            result, is_processed = await resolve_input("-", verbose=False)
            assert result == "stdin content here"
            assert is_processed is False
        finally:
            sys.stdin = original_stdin

    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_500_server_error_fail_fast(self, mock_run: MagicMock) -> None:

        """A 500 answer from the server fails."""
        mock_run.side_effect = [
            _cli_proc(0, "500"),  # status check
        ]
        with pytest.raises(ResolveError) as exc:
            await resolve_input("http://example.com/server-error")
        assert "500" in str(exc.value)

    @patch('app.lib.resolver.resolver.describe_image')
    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_browser_auto_download(self, mock_run: MagicMock, mock_browser_mgr: AsyncMock, mock_describe_image: MagicMock) -> None:

        """A file the browser downloads on its own is described with no type check."""
        mock_describe_image.return_value = _mock_describe("An auto-downloaded image.")
        mock_run.side_effect = _mock_run_exiftool("image/png").side_effect

        download_path = os.path.join(_TEMP_DIR, "auto.png")
        os.makedirs(_TEMP_DIR, exist_ok=True)
        with open(download_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")

        mgr = _setup_download_mgr(mock_browser_mgr, download_path)

        result, _ = await resolve_input("http://example.com/auto.png")
        assert "auto-downloaded image" in result
        mgr.evaluate.assert_not_called()
        try:
            os.unlink(download_path)
        except FileNotFoundError:
            pass

    @pytest.mark.fast_sleeps
    @patch('app.lib.resolver.resolver.describe_image')
    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_browser_rendered_media_download(self, mock_run: MagicMock, mock_browser_mgr: AsyncMock, mock_describe_image: MagicMock) -> None:

        """An image shown in the browser is downloaded and described."""
        mock_describe_image.return_value = _mock_describe("A rendered image.")
        mock_run.side_effect = _mock_run_exiftool("image/png").side_effect

        mgr = AsyncMock()
        mock_browser_mgr.create = AsyncMock(return_value=mgr)
        mgr.new_page.return_value = "tab1"
        mgr.wait_for_download.return_value = {"downloaded": False}
        mgr.url = MagicMock(return_value="http://example.com/media/image.png")
        # The fake page answers the type check, then the download, then the cleanup.
        mgr.evaluate.side_effect = ["image/png", True, None]

        download_path = os.path.join(_TEMP_DIR, "image.png")
        os.makedirs(_TEMP_DIR, exist_ok=True)
        with open(download_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")

        mgr.wait_for_selector.return_value = True
        mgr.click.return_value = True
        mgr.wait_for_download.side_effect = [
            {"downloaded": False},
            {"downloaded": True, "saved_path": download_path},
        ]
        mgr.close_page.return_value = True
        mgr.cleanup.return_value = None

        result, _ = await resolve_input("http://example.com/media/image.png")
        assert "rendered image" in result
        try:
            os.unlink(download_path)
        except FileNotFoundError:
            pass

    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_browser_json_returns_raw_json(self, mock_run: MagicMock, mock_browser_mgr: AsyncMock) -> None:

        """A JSON page comes back as its raw text."""
        raw_json = '{"current_condition": [{"temp_C": "25"}]}'
        mock_run.side_effect = _mock_run_exiftool("application/json").side_effect

        download_path = os.path.join(_TEMP_DIR, "data.json")
        os.makedirs(_TEMP_DIR, exist_ok=True)

        _setup_content_type_mgr(
            mock_browser_mgr, "http://example.com/data.json", "application/json", raw_json
        )

        result, _ = await resolve_input("http://example.com/data.json")
        assert result == raw_json
        assert '{"text":' not in result
        try:
            os.unlink(download_path)
        except FileNotFoundError:
            pass

    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_browser_text_plain_returns_raw_content(self, mock_run: MagicMock, mock_browser_mgr: AsyncMock) -> None:

        """A plain text page comes back as its raw text."""
        raw_text = '{"current_condition": [{"temp_C": "25"}]}'
        mock_run.side_effect = _mock_run_exiftool("application/json").side_effect

        download_path = os.path.join(_TEMP_DIR, "data.txt")
        os.makedirs(_TEMP_DIR, exist_ok=True)

        _setup_content_type_mgr(
            mock_browser_mgr, "http://example.com/data.txt", "text/plain", raw_text
        )

        result, _ = await resolve_input("http://example.com/data.txt")
        assert result == raw_text
        assert '{"text":' not in result
        try:
            os.unlink(download_path)
        except FileNotFoundError:
            pass

    @patch('app.lib.resolver.resolver.describe_pdf')
    @patch('app.lib.resolver.resolver.BrowserManager')
    @patch('asyncio.create_subprocess_exec')
    async def test_resolve_url_browser_pdf_download(self, mock_run: MagicMock, mock_browser_mgr: AsyncMock, mock_describe_pdf: MagicMock) -> None:

        """A PDF page is saved by the browser and described."""
        mock_describe_pdf.return_value = _mock_describe("Browser-captured PDF.")
        mock_run.side_effect = _mock_run_exiftool("application/pdf").side_effect

        pdf_path = os.path.join(_TEMP_DIR, "doc.pdf")
        os.makedirs(_TEMP_DIR, exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 dummy")

        mgr = AsyncMock()
        mock_browser_mgr.create = AsyncMock(return_value=mgr)
        mgr.new_page.return_value = "tab1"
        mgr.wait_for_download.return_value = {"downloaded": False}
        mgr.url = MagicMock(return_value="http://example.com/doc.pdf")
        mgr.evaluate.return_value = "application/pdf"
        mgr.pdf.return_value = pdf_path
        mgr.close_page.return_value = True
        mgr.cleanup.return_value = None

        result, _ = await resolve_input("http://example.com/doc.pdf")
        assert "Browser-captured PDF" in result
        try:
            os.unlink(pdf_path)
        except FileNotFoundError:
            pass


class TestBinaryPassthrough:
    """Test passing binary files through to the downloads folder."""

    @pytest.mark.parametrize(
        "mime",
        [
            "image/png",
            "video/mp4",
            "audio/mpeg",
            "application/pdf",
            "application/zip",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ],
    )
    async def test_passthrough_qualifies_binary_mime(
        self, mime: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        res = tmp_path / "resources"
        monkeypatch.setenv("RESOURCES_DIR", str(res))
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x89\x00\x01")
        monkeypatch.setattr(paths, "mime_type", AsyncMock(return_value=mime))
        content, _ = await resolve_input(str(f), binary_passthrough=True)
        assert content.startswith(resolver.BINARY_PASSTHROUGH_TAG)

    @pytest.mark.parametrize(
        "mime",
        ["text/plain", "text/html", "application/json", None],
        ids=["text/plain", "text/html", "application/json", "no-mime"],
    )
    async def test_passthrough_declines_textual_mime(
        self, mime: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        res = tmp_path / "resources"
        monkeypatch.setenv("RESOURCES_DIR", str(res))
        f = tmp_path / "data.txt"
        f.write_text("just some text\n")
        monkeypatch.setattr(paths, "mime_type", AsyncMock(return_value=mime))
        content, _ = await resolve_input(str(f), binary_passthrough=True)
        assert not content.startswith(resolver.BINARY_PASSTHROUGH_TAG)

    async def test_resolve_binary_copies_to_downloads_and_tags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        res = tmp_path / "resources"
        monkeypatch.setenv("RESOURCES_DIR", str(res))
        f = tmp_path / "img.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        monkeypatch.setattr(paths, "mime_type", AsyncMock(return_value="image/png"))
        content, processed = await resolve_input(str(f), binary_passthrough=True)
        assert processed is False
        assert content.startswith(resolver.BINARY_PASSTHROUGH_TAG)
        assert content.endswith("downloads/img.png")
        assert (res / "downloads" / "img.png").read_bytes() == f.read_bytes()

    async def test_resolve_text_file_not_tagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        res = tmp_path / "resources"
        monkeypatch.setenv("RESOURCES_DIR", str(res))
        f = tmp_path / "notes.txt"
        f.write_text("just some text\n")
        monkeypatch.setattr(paths, "mime_type", AsyncMock(return_value="text/plain"))
        content, _ = await resolve_input(str(f), binary_passthrough=True)
        assert not content.startswith(resolver.BINARY_PASSTHROUGH_TAG)
        assert content == "just some text\n"

    async def test_resolve_over_cap_binary_falls_to_binary_save_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        res = tmp_path / "resources"
        monkeypatch.setenv("RESOURCES_DIR", str(res))
        f = tmp_path / "big.zip"
        f.write_bytes(b"PK\x03\x04" + b"\xff" * 20)
        monkeypatch.setattr(paths, "mime_type", AsyncMock(return_value="application/zip"))
        monkeypatch.setattr(resolver, "_BINARY_PASSTHROUGH_CAP", 8)
        content, _ = await resolve_input(str(f), binary_passthrough=True)
        assert not content.startswith(resolver.BINARY_PASSTHROUGH_TAG)
        assert "Binary file with size" in content
