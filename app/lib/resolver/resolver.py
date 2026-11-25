"""Turn a web address, a file, or plain text into text a model can read."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import aiofiles
import httpx

from app.lib.browser.browser_manager import BrowserManager
from app.lib.browser.downloads import DownloadResult
from app.lib.browser.snapshot_filter import filter_html_file, html_to_json
from app.lib.cli.cli_runner import run_cli
from app.lib.laya import client as laya
from app.lib.reason.describe_image import describe_image
from app.lib.reason.describe_pdf import describe_pdf
from app.lib.reason.describe_video import describe_video
from app.lib.resolver.office_docs import (
    describe_presentation,
    doc_to_text,
    docx_to_text,
    xls_to_text,
    xlsx_to_text,
)
from app.lib.shared import config, paths
from app.resources.resources import binary_download_notice

__all__ = ["ResolveError", "ResolvedContent", "resolve_input"]


class ResolveError(Exception):
    """The input could not be turned into text a model can read."""


class _EmptyHtmlError(ResolveError):
    """An HTML page has no text to show."""


@dataclass(frozen=True)
class ResolvedContent:
    """Content extracted from a URL, file, or text input."""

    content: str = ""
    is_described: bool = False
    binary_path: str | None = None


_BINARY_DOWNLOAD_FOLDER = "downloads"
_MAX_BINARY_PASSTHROUGH_BYTES = 50 * 1024 * 1024
_DOWNLOAD_LINK_ID = "resolver_download_link"
_DOWNLOAD_LINK_TEXT = "DOWNLOAD"
_SCRATCH_FOLDER_NAME = "resolver"

_HTTP_OK = 200
_HTTP_UNAUTHORIZED = 401
_HTTP_FORBIDDEN = 403
_HTTP_CLIENT_ERROR_MIN = 400
_HTTP_CLIENT_ERROR_MAX = 499
_HTTP_SERVER_ERROR_MIN = 500
_HTTP_SERVER_ERROR_MAX = 599
_ASR_TIMEOUT_SECONDS = 300

_MIME_DOC = "application/msword"
_MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_MIME_PPT = "application/vnd.ms-powerpoint"
_MIME_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_MIME_XLS = "application/vnd.ms-excel"
_MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_MIME_PRESENTATIONS = (_MIME_PPT, _MIME_PPTX)

_OFFICE_TEXT_PARSERS: dict[str, Callable[[str, bool], Awaitable[str]]] = {
    _MIME_DOC: doc_to_text,
    _MIME_DOCX: docx_to_text,
    _MIME_XLS: xls_to_text,
    _MIME_XLSX: xlsx_to_text,
}

_BROWSER_WAIT_TIMEOUT_MILLISECONDS = 30000
_DOWNLOAD_WAIT_TIMEOUT_SECONDS = 60
_CLEANUP_DELAY_SECONDS = 1
_FILE_SCHEME = "file"
_MIN_CONTENT_LENGTH = 500
_HTML_MIME_TYPES = ("text/html", "application/xhtml+xml")


def _log_verbose(is_verbose: bool, message: str, is_error: bool = False) -> None:
    """Print a progress line when verbose logging is on."""
    if not is_verbose:
        return
    stream = sys.stderr if is_error else sys.stdout
    print(message, file=stream, flush=True)


async def _read_text_file(file_path: str) -> str:
    """Read a text file, or give back nothing when it cannot be read."""
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as file_handle:
            return await file_handle.read()
    except Exception:
        return ""


async def _read_text_file_strict(file_path: str) -> str:
    """Read a text file and let any error reach the caller."""
    async with aiofiles.open(file_path, "r", encoding="utf-8") as file_handle:
        return await file_handle.read()


def _unwrap_described_result(description: str, status: int) -> ResolvedContent:
    """Return described content, or fail when the describer returned an error."""
    if status != 0:
        raise ResolveError(description)
    return ResolvedContent(content=description, is_described=True)


async def _extract_downloaded_file(
    download_path: str,
    context: str,
    is_verbose: bool,
    should_passthrough_binary: bool = False,
    mime_type: str | None = None,
) -> ResolvedContent | None:
    """Extract content from a downloaded file, or return None when a text page is too short."""
    if mime_type is None:
        mime_type = await paths.mime_type(download_path)
    if paths.is_text_mime(mime_type):
        content_full = await _read_text_file(download_path)
        reason = await _find_insufficient_content_reason(content_full)
        if reason is not None:
            _log_verbose(
                is_verbose,
                f"[FALLBACK] Content insufficient ({reason}), triggering browser fallback.",
            )
            return None

    return await _extract_file_content(
        download_path,
        context=context,
        is_verbose=is_verbose,
        should_passthrough_binary=should_passthrough_binary,
    )


def _require_content(resolved: ResolvedContent | None) -> ResolvedContent:
    """Return the content, or fail when the download held too little."""
    if resolved is None:
        raise ResolveError("The downloaded page held too little content to use.")
    return resolved


async def _save_browser_pdf(
    browser_manager: BrowserManager,
    target_id: str | None,
    url: str | None,
    download_path: str | None,
    is_verbose: bool = False,
) -> str | None:
    """Save the PDF shown in a browser tab, or return None on failure."""
    if target_id is None or url is None or download_path is None:
        _log_verbose(is_verbose, "None input in _save_browser_pdf", is_error=True)
        return None
    _log_verbose(is_verbose, f"Processing as PDF document: {url}")
    pdf_result = await browser_manager.save_pdf(target_id, download_path, verbose=is_verbose)
    if pdf_result is None:
        _log_verbose(is_verbose, f"PDF capture failed for {url}", is_error=True)
        return None
    if pdf_result != download_path:
        await asyncio.to_thread(shutil.move, pdf_result, download_path)
    return download_path


def _build_download_link_script(url: str) -> str:
    """Build the script that puts a download link on the page."""
    escaped_url = url.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return f"""async () => {{
        try {{
            const resourceUrl = "{escaped_url}";
            const filename = resourceUrl.split('/').pop() || 'download';
            const a = document.createElement("a");
            a.href = resourceUrl;
            a.download = filename;
            a.id = "{_DOWNLOAD_LINK_ID}";
            a.innerText = "{_DOWNLOAD_LINK_TEXT}";
            a.style.position = "fixed";
            a.style.top = "0";
            a.style.left = "0";
            a.style.zIndex = "9999";
            (document.body || document.documentElement).appendChild(a);
            return true;
        }} catch (e) {{ return "js_error:" + e.message; }}
    }}"""


async def _remove_download_link(browser_manager: BrowserManager, target_id: str) -> None:
    """Remove the injected download link from the page."""
    try:
        await asyncio.sleep(_CLEANUP_DELAY_SECONDS)
        await browser_manager.evaluate(
            target_id,
            f"""() => {{
                const a = document.getElementById('{_DOWNLOAD_LINK_ID}');
                if (a) a.remove();
            }}""",
            verbose=False,
        )
    except Exception:
        pass


async def _inject_download_link(
    browser_manager: BrowserManager,
    target_id: str,
    url: str,
    is_verbose: bool,
) -> bool:
    """Inject the download link into the page and wait for it to appear."""
    script = _build_download_link_script(url)
    _log_verbose(is_verbose, f"Injecting download link for {url}")
    result = await browser_manager.evaluate(target_id, script, verbose=is_verbose)
    if not result or isinstance(result, str):
        error_detail = result if isinstance(result, str) else "unknown"
        _log_verbose(
            is_verbose,
            f"[JS ERROR] Failed to inject download link for {url}: {error_detail}",
            is_error=True,
        )
        return False

    selector = f"#{_DOWNLOAD_LINK_ID}"
    _log_verbose(is_verbose, f"Waiting for download link to appear for {url}")
    is_found = await browser_manager.wait_for_selector(
        target_id, selector, timeout=_BROWSER_WAIT_TIMEOUT_MILLISECONDS, verbose=is_verbose
    )
    if not is_found:
        _log_verbose(is_verbose, f"Download link did not appear for {url}", is_error=True)
        return False
    return True


async def _download_browser_media(
    browser_manager: BrowserManager,
    target_id: str | None,
    url: str | None,
    download_path: str | None,
    is_verbose: bool = False,
) -> str | None:
    """Save a picture, video, or audio file shown in a browser tab."""
    if target_id is None or url is None or download_path is None:
        _log_verbose(is_verbose, "None input in _download_browser_media", is_error=True)
        return None
    _log_verbose(is_verbose, f"Processing as rendered media: {url}")

    if not await _inject_download_link(browser_manager, target_id, url, is_verbose):
        return None

    selector = f"#{_DOWNLOAD_LINK_ID}"
    _log_verbose(is_verbose, f"Clicking download link for {url}")

    async def _click_download_link() -> bool:
        return await browser_manager.click(target_id, selector, verbose=is_verbose)

    download_result = await browser_manager.wait_for_download(
        target_id=target_id,
        action=_click_download_link,
        save_path=download_path,
        timeout=_DOWNLOAD_WAIT_TIMEOUT_SECONDS,
        verbose=is_verbose,
    )
    if not download_result.get("downloaded"):
        _log_verbose(is_verbose, f"Download failed for {url}", is_error=True)
        return None

    await _remove_download_link(browser_manager, target_id)
    _log_verbose(is_verbose, f"Successfully downloaded {url}")
    return await _rename_to_suggested_name(download_result, download_path)


def _extract_all_text(node: dict | list | str) -> str:
    """Join every bit of text in the data."""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return " ".join(_extract_all_text(child) for child in node)
    if isinstance(node, dict):
        return " ".join(_extract_all_text(value) for value in node.values())
    return ""


async def _find_insufficient_content_reason(content: str) -> str | None:
    """Return why a downloaded page holds too little text, or None when it is usable."""
    if not content:
        return "the page is empty"

    filtered_text = content
    if content.lstrip().startswith("<"):
        try:
            json_tree = await html_to_json(content)
            filtered_text = await asyncio.to_thread(_extract_all_text, json_tree)
        except Exception:
            pass

    if len(filtered_text) < _MIN_CONTENT_LENGTH:
        return f"only {len(filtered_text)} chars (below {_MIN_CONTENT_LENGTH})"

    return None


def _sanitize_filename(name: str, default_name: str) -> str:
    """Strip folder components from a file name, falling back when the name is unsafe."""
    filename = os.path.basename(name)
    if filename in ("", ".", ".."):
        return default_name
    return filename


def _decode_rfc5987(value: str) -> str | None:
    """Decode an RFC 5987 header file name, or return None when decoding fails."""
    charset, _, encoded_value = value.partition("'")
    _, _, encoded_value = encoded_value.partition("'")
    try:
        return unquote(encoded_value, encoding=charset or "utf-8")
    except (LookupError, UnicodeDecodeError):
        return None


def _extract_suggested_filename(headers: str, default_name: str) -> str:
    """Find a safe file name in response headers, or return default_name."""
    for line in headers.splitlines():
        header_name, separator, value = line.partition(":")
        if not separator or header_name.strip().lower() != "content-disposition":
            continue
        encoded_match = re.search(r"filename\*=(\S+)", value, re.IGNORECASE)
        if encoded_match:
            filename = _decode_rfc5987(encoded_match.group(1))
        else:
            plain_match = re.search(r'filename=(?:"([^"]+)"|(\S+))', value, re.IGNORECASE)
            if not plain_match:
                continue
            filename = plain_match.group(1) or plain_match.group(2)
        if filename:
            return _sanitize_filename(filename, default_name)
    return default_name


async def _rename_to_suggested_name(download_result: DownloadResult, default_path: str) -> str:
    """Rename a downloaded file to the server suggested name when present."""
    suggested = download_result.get("filename")
    if not suggested:
        return default_path
    filename = _sanitize_filename(suggested, os.path.basename(default_path))
    if filename == os.path.basename(default_path):
        return default_path
    target_path = os.path.join(os.path.dirname(default_path), filename)
    await asyncio.to_thread(shutil.move, default_path, target_path)
    return target_path


async def _try_curl_download(url: str, download_path: str, is_verbose: bool = False) -> str | None:
    """Download a URL with curl, returning the saved path or None to fall back to the browser."""
    if url is None or download_path is None:
        raise ResolveError("url or download_path is None")
    _log_verbose(is_verbose, f"[CURL] Downloading: {url}", is_error=True)

    status_command = [
        "curl", "-L", "-s", "--compressed", "-o", "/dev/null", "-D", "-", "-w", "\n%{http_code}", url
    ]
    header_result = await run_cli(status_command, "curl status check")
    if header_result.returncode != 0:
        return None
    headers, _, http_code = header_result.stdout.rpartition("\n")

    if http_code.isdigit():
        http_code_int = int(http_code)
        if http_code_int in (_HTTP_FORBIDDEN, _HTTP_UNAUTHORIZED):
            return None
        if _HTTP_CLIENT_ERROR_MIN <= http_code_int <= _HTTP_CLIENT_ERROR_MAX:
            raise ResolveError(f"HTTP {http_code} - Not found or access denied")
        if _HTTP_SERVER_ERROR_MIN <= http_code_int <= _HTTP_SERVER_ERROR_MAX:
            raise ResolveError(f"HTTP {http_code} - Server error")

    file_name = _extract_suggested_filename(headers, os.path.basename(download_path))
    final_path = os.path.join(os.path.dirname(download_path), file_name)
    download_command = ["curl", "-L", "-s", "--compressed", "-o", final_path, url]
    result = await run_cli(download_command, "curl download")

    if result.returncode != 0 or not await asyncio.to_thread(os.path.exists, final_path):
        _log_verbose(is_verbose, f"[CURL] Failed for {url} (rc={result.returncode})", is_error=True)
        return None
    _log_verbose(is_verbose, f"[CURL] Success: Content saved to {final_path}", is_error=True)
    return final_path


async def _browse_and_extract_url(
    browser_manager: BrowserManager,
    target_id: str,
    url: str,
    download_path: str,
    context: str,
    is_verbose: bool,
    should_passthrough_binary: bool = False,
) -> ResolvedContent:
    """Open a URL in a browser tab and extract its content."""
    download_result = await browser_manager.wait_for_download(
        target_id=target_id,
        url=url,
        save_path=download_path,
        timeout=_DOWNLOAD_WAIT_TIMEOUT_SECONDS,
        verbose=is_verbose,
    )
    if download_result.get("downloaded"):
        renamed_path = await _rename_to_suggested_name(download_result, download_path)
        return _require_content(
            await _extract_downloaded_file(renamed_path, context, is_verbose, should_passthrough_binary)
        )

    actual_url = browser_manager.get_url(target_id, verbose=is_verbose) or url
    if actual_url != url:
        _log_verbose(is_verbose, f"URL redirected: {url} -> {actual_url}")

    content_type = await browser_manager.evaluate(target_id, "document.contentType", verbose=is_verbose)
    if not isinstance(content_type, str):
        raise ResolveError("Unable to detect content type from browser")

    saved_path = await _download_browser_content_by_type(
        browser_manager, target_id, actual_url, download_path, content_type, is_verbose
    )
    if saved_path:
        return _require_content(
            await _extract_downloaded_file(saved_path, context, is_verbose, should_passthrough_binary)
        )

    _log_verbose(is_verbose, "Taking web snapshot")
    snapshot = await browser_manager.capture_snapshot(target_id, verbose=is_verbose)
    if snapshot is None:
        raise ResolveError(f"Failed to capture snapshot of {actual_url}")
    tree = await snapshot.filtered_tree(verbose=is_verbose)
    return ResolvedContent(content=json.dumps(tree, ensure_ascii=False))


async def _save_browser_text(
    browser_manager: BrowserManager,
    target_id: str | None,
    actual_url: str | None,
    download_path: str | None,
    is_verbose: bool = False,
) -> str | None:
    """Save the plain text shown in a browser tab, or return None on failure."""
    if target_id is None or actual_url is None or download_path is None:
        _log_verbose(is_verbose, "None input in _save_browser_text", is_error=True)
        return None
    _log_verbose(is_verbose, f"Processing as textual document: {actual_url}")
    text = await browser_manager.evaluate(target_id, "document.body.innerText", verbose=is_verbose)
    if not isinstance(text, str):
        _log_verbose(is_verbose, f"Failed to extract text for {actual_url}", is_error=True)
        return None
    try:
        async with aiofiles.open(download_path, "w", encoding="utf-8") as file_handle:
            await file_handle.write(text)
    except Exception as error:
        _log_verbose(is_verbose, f"Failed to write text to {download_path}: {error}", is_error=True)
        return None
    return download_path


async def _download_browser_content_by_type(
    browser_manager: BrowserManager,
    target_id: str,
    actual_url: str,
    download_path: str,
    content_type: str,
    is_verbose: bool,
) -> str | None:
    """Save browser content when its MIME type is text, media, or PDF."""
    if "json" in content_type or content_type == "text/plain":
        _log_verbose(is_verbose, f"Detected textual content type: {content_type}")
        saved_path = await _save_browser_text(
            browser_manager, target_id, actual_url, download_path, is_verbose=is_verbose
        )
        if not saved_path:
            raise ResolveError(f"Failed to download text content: {actual_url}")
        return saved_path
    if "image" in content_type or "video" in content_type or "audio" in content_type:
        _log_verbose(is_verbose, "Detected media content type")
        saved_path = await _download_browser_media(
            browser_manager, target_id, actual_url, download_path, is_verbose=is_verbose
        )
        if not saved_path:
            raise ResolveError(f"Failed to download media: {actual_url}")
        return saved_path
    if "pdf" in content_type:
        _log_verbose(is_verbose, "Detected pdf content type")
        saved_path = await _save_browser_pdf(
            browser_manager, target_id, actual_url, download_path, is_verbose=is_verbose
        )
        if not saved_path:
            raise ResolveError(f"Failed to download PDF: {actual_url}")
        return saved_path
    return None


async def _is_blocked_page(file_path: str, url: str, is_verbose: bool) -> bool:
    """Return True when Laya marks a saved HTML page as blocked or cannot be reached."""
    async with aiofiles.open(file_path, "r", encoding="utf-8", errors="replace") as file_handle:
        html = await file_handle.read()
    try:
        verdict: laya.Verdict = await laya.classify(html)
    except laya.LayaError as error:
        print(f"[LAYA] unavailable ({error}), using browser for: {url}", file=sys.stderr, flush=True)
        return True
    if verdict.blocked:
        _log_verbose(
            is_verbose,
            f"[LAYA] blocked (p={verdict.blocked_probability:.2f}), using browser for: {url}",
        )
    return verdict.blocked


async def _fetch_url_with_browser(
    url: str,
    download_path: str,
    context: str,
    is_verbose: bool,
    browser_manager: BrowserManager | None,
    should_passthrough_binary: bool,
) -> ResolvedContent:
    """Open a URL in the browser and extract its content."""
    is_browser_owned = browser_manager is None
    target_id: str | None = None
    try:
        try:
            if browser_manager is None:
                browser_manager = await BrowserManager.create()
            target_id = await browser_manager.open_page(verbose=is_verbose)
        except Exception as error:
            _log_verbose(is_verbose, f"Error resolving URL: {error}", is_error=True)
            raise ResolveError(f"Error resolving URL: {error}") from error
        if not target_id:
            raise ResolveError("Unable to open new browser tab")
        try:
            return await _browse_and_extract_url(
                browser_manager,
                target_id,
                url,
                download_path,
                context,
                is_verbose,
                should_passthrough_binary,
            )
        except ResolveError:
            raise
        except Exception as error:
            raise ResolveError(str(error)) from error
    finally:
        if target_id and browser_manager is not None:
            await browser_manager.close_page(target_id, verbose=is_verbose)
        if is_browser_owned and browser_manager is not None:
            await browser_manager.cleanup()


async def _fetch_url_content(
    url: str,
    context: str = "",
    is_verbose: bool = False,
    browser_manager: BrowserManager | None = None,
    should_passthrough_binary: bool = False,
) -> ResolvedContent:
    """Fetch a URL and extract its content, falling back to the browser when needed."""
    if not url:
        raise ResolveError("URL is None or empty")
    file_name = url.split("/")[-1].split("?")[0] or "downloaded_file"
    scratch_folder = paths.scratch_dir(_SCRATCH_FOLDER_NAME)
    await asyncio.to_thread(os.makedirs, scratch_folder, exist_ok=True)
    download_path = os.path.join(scratch_folder, file_name)

    _log_verbose(is_verbose, f"[CURL] Attempting download: {url}")
    try:
        saved_path = await _try_curl_download(url, download_path, is_verbose)
        if saved_path:
            mime_type = await paths.mime_type(saved_path)
            resolved = await _extract_downloaded_file(
                saved_path, context, is_verbose, should_passthrough_binary, mime_type=mime_type
            )
            if resolved is not None:
                if mime_type not in _HTML_MIME_TYPES or not await _is_blocked_page(saved_path, url, is_verbose):
                    return resolved
    except _EmptyHtmlError:
        _log_verbose(is_verbose, f"[FALLBACK] No text left in HTML, using browser for: {url}")
    except ResolveError:
        raise
    except Exception as error:
        _log_verbose(is_verbose, f"[FAIL-FAST] {error}", is_error=True)
        raise ResolveError(str(error)) from error

    _log_verbose(is_verbose, f"[BROWSER-FALLBACK] Curl failed, using browser for: {url}")
    return await _fetch_url_with_browser(
        url, download_path, context, is_verbose, browser_manager, should_passthrough_binary
    )


async def _transcribe_audio(audio_path: str, is_verbose: bool = False) -> str:
    """Transcribe speech from an audio file using the configured ASR server."""
    if audio_path is None:
        raise ResolveError("Audio path is None")
    try:
        server_url = config.require_env("ASR_SERVER_URL", "needed by audio and video transcription")
    except config.MissingSettingError as error:
        raise ResolveError(str(error)) from error
    try:
        _log_verbose(is_verbose, f"Transcribing audio via ASR server: {audio_path}", is_error=True)
        async with aiofiles.open(audio_path, "rb") as file_handle:
            audio_data = await file_handle.read()
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.post(
                f"{server_url.rstrip('/')}/asr",
                files={"audio_file": (os.path.basename(audio_path), audio_data)},
                timeout=_ASR_TIMEOUT_SECONDS,
            )

        if response.status_code != _HTTP_OK:
            _log_verbose(
                is_verbose,
                f"ASR server returned HTTP {response.status_code}: {response.text}",
                is_error=True,
            )
            raise ResolveError(f"ASR transcription failed (HTTP {response.status_code}): {response.text}")

        transcription = response.text.strip()
        if not transcription:
            raise ResolveError("ASR transcription returned empty result")

        return transcription
    except ResolveError:
        raise
    except Exception as error:
        raise ResolveError(f"Error during audio transcription: {error}") from error


async def resolve_input(
    text: str,
    context: str = "",
    verbose: bool = False,
    binary_passthrough: bool = False,
) -> ResolvedContent:
    """Turn a web address, a file, or plain text into extracted content.

    Args:
        text: A web address, a file path, or plain text. A single dash reads
            from standard input.
        context: A question about a picture, a video, a PDF, or slides.
            Empty asks for a description.
        verbose: True to print progress lines.
        binary_passthrough: True to copy a small binary file into the downloads
            folder and return its path.

    Returns:
        The extracted text, whether a model described it, or the saved binary path.

    Raises:
        ResolveError: If the input could not be turned into content.
    """
    if context is None:
        context = ""
    if text is None:
        raise ResolveError("Input text is None")

    if text == "-":
        try:
            _log_verbose(verbose, "Reading input from stdin...", is_error=True)
            stdin_text = await asyncio.to_thread(sys.stdin.read)
            return ResolvedContent(content=stdin_text)
        except Exception as error:
            raise ResolveError(f"Error reading from stdin: {error}") from error

    parsed_url = urlparse(text.strip())
    if parsed_url.scheme == _FILE_SCHEME:
        return await _extract_file_content(parsed_url.path, context, verbose, binary_passthrough)
    if parsed_url.scheme and parsed_url.netloc:
        return await _fetch_url_content(
            text.strip(),
            context=context,
            is_verbose=verbose,
            should_passthrough_binary=binary_passthrough,
        )

    if await asyncio.to_thread(os.path.isfile, text):
        return await _extract_file_content(text, context, verbose, binary_passthrough)

    return ResolvedContent(content=text)


async def _extract_media_or_document_content(
    file_path: str,
    mime_type: str,
    context: str,
    is_verbose: bool,
) -> ResolvedContent | None:
    """Extract content from a media, office document, or HTML file, or return None for other types."""
    if paths.is_pdf_mime(mime_type):
        return _unwrap_described_result(*await describe_pdf(file_path, context, is_verbose))
    if paths.is_image_mime(mime_type):
        return _unwrap_described_result(*await describe_image(file_path, context, verbose=is_verbose))
    if paths.is_video_mime(mime_type):
        return _unwrap_described_result(*await describe_video(file_path, context, verbose=is_verbose))
    if paths.is_audio_mime(mime_type):
        transcription = await _transcribe_audio(file_path, is_verbose)
        return ResolvedContent(content=transcription)
    if mime_type in _OFFICE_TEXT_PARSERS or mime_type in _MIME_PRESENTATIONS:
        return await _extract_office_content(file_path, mime_type, context, is_verbose)
    if mime_type == "text/html" or file_path.endswith((".html", ".htm")):
        filtered = await filter_html_file(file_path, is_verbose)
        if filtered is None:
            raise _EmptyHtmlError(f"Failed to extract text from HTML: {file_path}")
        html_text = json.dumps(filtered, ensure_ascii=False) if isinstance(filtered, dict) else filtered
        return ResolvedContent(content=html_text)
    return None


async def _extract_file_content(
    file_path: str,
    context: str = "",
    is_verbose: bool = False,
    should_passthrough_binary: bool = False,
) -> ResolvedContent:
    """Extract text or a binary passthrough path from a local file."""
    if not file_path:
        raise ResolveError("File path is None or empty")
    if not await asyncio.to_thread(os.path.exists, file_path):
        raise ResolveError(f"File not found: {file_path}")
    mime_type = await paths.mime_type(file_path)
    if should_passthrough_binary:
        passthrough = await _try_passthrough_binary(file_path, mime_type, is_verbose)
        if passthrough is not None:
            return passthrough

    if mime_type:
        extracted = await _extract_media_or_document_content(file_path, mime_type, context, is_verbose)
        if extracted is not None:
            return extracted

    try:
        return ResolvedContent(content=await _read_text_file_strict(file_path))
    except UnicodeDecodeError:
        return await _save_binary_file(file_path, is_verbose)
    except Exception as error:
        raise ResolveError(f"Error reading file '{file_path}': {error}") from error


async def _extract_office_content(
    file_path: str,
    mime_type: str,
    context: str,
    is_verbose: bool,
) -> ResolvedContent:
    """Extract text from a Word or Excel file, or describe a slide deck."""
    parser = _OFFICE_TEXT_PARSERS.get(mime_type)
    if parser is not None:
        try:
            text = await parser(file_path, is_verbose)
        except Exception as error:
            raise ResolveError(f"Error extracting text from '{file_path}': {error}") from error
        return ResolvedContent(content=text)
    return _unwrap_described_result(*await describe_presentation(file_path, context, is_verbose))


async def _save_binary_file(file_path: str, is_verbose: bool) -> ResolvedContent:
    """Copy a binary file into the resources folder and return a download notice."""
    filename = os.path.basename(file_path)
    destination_path = _build_binary_download_path(file_path)
    try:
        await asyncio.to_thread(_copy_into_downloads, file_path, destination_path)
    except Exception as error:
        raise ResolveError(f"Error saving binary file '{file_path}': {error}") from error
    _log_verbose(is_verbose, f"[BINARY] Saved {file_path} to {destination_path}", is_error=True)
    notice = await binary_download_notice(f"{_BINARY_DOWNLOAD_FOLDER}/{filename}")
    return ResolvedContent(content=notice)


def _copy_into_downloads(source_path: str, destination_path: str) -> None:
    """Copy a file into the downloads directory, creating missing parent folders."""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy2(source_path, destination_path)


def _build_binary_download_path(file_path: str) -> str:
    """Build the destination path for a copied binary file in the resources folder."""
    return os.path.join(paths.resources_dir(), _BINARY_DOWNLOAD_FOLDER, os.path.basename(file_path))


async def _copy_binary_for_passthrough(file_path: str, is_verbose: bool) -> ResolvedContent:
    """Copy a binary file into the resources folder and return its path."""
    destination_path = _build_binary_download_path(file_path)
    try:
        await asyncio.to_thread(_copy_into_downloads, file_path, destination_path)
    except Exception as error:
        raise ResolveError(f"Error saving binary file '{file_path}': {error}") from error
    _log_verbose(is_verbose, f"[BINARY-PASSTHROUGH] {file_path} -> {destination_path}", is_error=True)
    return ResolvedContent(binary_path=destination_path)


async def _try_passthrough_binary(
    file_path: str,
    mime_type: str | None,
    is_verbose: bool,
) -> ResolvedContent | None:
    """Copy a small binary file into the resources folder, or return None when skipped."""
    file_size = await asyncio.to_thread(os.path.getsize, file_path)
    is_binary = paths.is_binary_mime(mime_type)
    if is_binary is None:
        try:
            await _read_text_file_strict(file_path)
            return None
        except UnicodeDecodeError:
            is_binary = True
    if is_binary and file_size <= _MAX_BINARY_PASSTHROUGH_BYTES:
        return await _copy_binary_for_passthrough(file_path, is_verbose)
    return None
