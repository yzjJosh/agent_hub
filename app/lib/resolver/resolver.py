from __future__ import annotations

"""Turn a web address, a file, or plain text into text a model can read."""

__all__ = ["BINARY_PASSTHROUGH_TAG", "ResolveError", "resolve_input"]

import asyncio
import os
import re
import sys
import json
import shutil
from typing import Awaitable, Callable, Dict, Tuple, Optional
from urllib.parse import unquote, urlparse
import httpx
import aiofiles

from app.lib.cli.cli_runner import run_cli
from app.lib.laya import client as laya
from app.lib.shared import config, paths
from app.lib.resolver.office_docs import (
    describe_presentation,
    doc_to_text,
    docx_to_text,
    xls_to_text,
    xlsx_to_text,
)
from app.resources.resources import binary_download_notice

_BINARY_DOWNLOAD_DIR = "downloads"
BINARY_PASSTHROUGH_TAG = "__binary_passthrough__"
_BINARY_PASSTHROUGH_CAP = 50 * 1024 * 1024
from app.lib.reason.describe_image import describe_image
from app.lib.reason.describe_video import describe_video
from app.lib.reason.describe_pdf import describe_pdf

from app.lib.browser.snapshot_filter import filter_html_file, html_to_json

from app.lib.browser.browser_manager import BrowserManager


class ResolveError(Exception):
    """The input could not be turned into text a model can read."""


class _EmptyHtmlError(ResolveError):
    """An HTML page has no text to show."""


_TEMP_NAME = "resolver"

_HTTP_UNAUTHORIZED = 401
_HTTP_FORBIDDEN = 403
_HTTP_CLIENT_ERROR_MIN = 400
_HTTP_CLIENT_ERROR_MAX = 499
_HTTP_SERVER_ERROR_MIN = 500
_HTTP_SERVER_ERROR_MAX = 599

_MIME_DOC = "application/msword"
_MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_MIME_PPT = "application/vnd.ms-powerpoint"
_MIME_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_MIME_XLS = "application/vnd.ms-excel"
_MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_MIME_PRESENTATIONS = (_MIME_PPT, _MIME_PPTX)

_OFFICE_TEXT_PARSERS: Dict[str, Callable[[str, bool], Awaitable[str]]] = {
    _MIME_DOC: doc_to_text,
    _MIME_DOCX: docx_to_text,
    _MIME_XLS: xls_to_text,
    _MIME_XLSX: xlsx_to_text,
}

_BROWSER_WAIT_TIMEOUT_MS = 30000
_DOWNLOAD_WAIT_TIMEOUT_S = 60
_CLEANUP_DELAY_S = 1

_FILE_SCHEME = "file"

_MIN_CONTENT_LENGTH = 500

_HTML_MIMES = ("text/html", "application/xhtml+xml")


async def _read_text_file(path: str) -> str:
    """Read a text file, or give back nothing when it cannot be read."""
    try:
        async with aiofiles.open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return await f.read()
    except Exception:
        return ""


async def _read_text_file_strict(path: str) -> str:
    """Read a text file and let any error reach the caller."""
    async with aiofiles.open(path, 'r', encoding='utf-8') as f:
        return await f.read()


async def _handle_download_result(download_path: str, context: str, verbose: bool, binary_passthrough: bool = False, mime: Optional[str] = None) -> Optional[Tuple[str, bool]]:
    """Turn a downloaded file into content.

    Args:
        download_path: The file that was saved.
        context: A question about a picture, a video, a PDF, or slides.
            Empty asks for a description.
        verbose: True to print what is happening.
        binary_passthrough: True to keep a small binary file for download.
        mime: The file type when it is already known, or None to find it.

    Returns:
        The content and a flag that is True when a model already described
        it, or None when a text file holds almost nothing and the browser
        should try instead.
    """
    if mime is None:
        mime = await paths.mime_type(download_path)
    if paths.is_text_mime(mime):
        content_full = await _read_text_file(download_path)

        reason = await _is_content_insufficient(content_full)
        if reason is not None:
            if verbose:
                print(f"[FALLBACK] Content insufficient ({reason}), triggering browser fallback.")
            return None

    return await _resolve_file(download_path, context=context, verbose=verbose, binary_passthrough=binary_passthrough)


def _require_content(result: Optional[Tuple[str, bool]]) -> Tuple[str, bool]:
    """Return the content, or fail when the download held too little."""
    if result is None:
        raise ResolveError("The downloaded page held too little content to use.")
    return result


async def _resolve_url_download_document(mgr: BrowserManager, target_id: Optional[str], url: Optional[str], download_path: Optional[str], verbose: bool = False) -> Optional[str]:
    """Save the PDF the page is showing.

    Returns:
        The saved path, or None if it could not be saved.
    """
    if target_id is None or url is None or download_path is None:
        if verbose: print(f"None input in _resolve_url_download_document", file=sys.stderr)
        return None
    if verbose: print(f"Processing as PDF document: {url}")
    pdf_res = await mgr.pdf(target_id, download_path, verbose=verbose)
    if pdf_res is None:
        if verbose: print(f"PDF capture failed for {url}", file=sys.stderr)
        return None
    if pdf_res != download_path:
        await asyncio.to_thread(shutil.move, pdf_res, download_path)
    return download_path


def _create_download_link_js(url: str) -> str:
    """Build the code that puts a download link on the page."""
    escaped_url = url.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
    return f"""async () => {{
        try {{
            const resourceUrl = "{escaped_url}";
            const filename = resourceUrl.split('/').pop() || 'download';
            const a = document.createElement("a");
            a.href = resourceUrl;
            a.download = filename;
            a.id = "claw_download_link";
            a.innerText = "CLAW_DOWNLOAD";
            a.style.position = "fixed";
            a.style.top = "0";
            a.style.left = "0";
            a.style.zIndex = "9999";
            (document.body || document.documentElement).appendChild(a);
            return true;
        }} catch (e) {{ return "js_error:" + e.message; }}
    }}"""


async def _cleanup_download_link(mgr: BrowserManager, target_id: str) -> None:
    """Take the download link back off the page."""
    try:
        await asyncio.sleep(_CLEANUP_DELAY_S)
        await mgr.evaluate(target_id, """() => {
            const a = document.getElementById('claw_download_link');
            if (a) a.remove();
        }""", verbose=False)
    except Exception:
        pass


async def _resolve_url_download_rendered_media(mgr: BrowserManager, target_id: Optional[str], url: Optional[str], download_path: Optional[str], verbose: bool = False) -> Optional[str]:
    """Save a picture, video, or sound file that the page is showing.

    Returns:
        The saved path, or None if it could not be saved.
    """
    if target_id is None or url is None or download_path is None:
        if verbose: print(f"None input in _resolve_url_download_rendered_media", file=sys.stderr)
        return None
    if verbose: print(f"Processing as rendered media: {url}")

    js_code = _create_download_link_js(url)
    if verbose: print(f"Injecting download link for {url}")
    result = await mgr.evaluate(target_id, js_code, verbose=verbose)
    if not result or isinstance(result, str):
        error_detail = result if isinstance(result, str) else "unknown"
        if verbose:
            print(f"[JS ERROR] Failed to inject download link for {url}: {error_detail}", file=sys.stderr)
        return None

    if verbose: print(f"Waiting for download link to appear for {url}")
    found = await mgr.wait_for_selector(target_id, "#claw_download_link", timeout=_BROWSER_WAIT_TIMEOUT_MS, verbose=verbose)
    if not found:
        if verbose: print(f"Download link did not appear for {url}", file=sys.stderr)
        return None

    if verbose: print(f"Clicking download link for {url}")
    async def _click_download_link() -> bool:
        return await mgr.click(target_id, "#claw_download_link", verbose=verbose)

    dl_result = await mgr.wait_for_download(
        target_id=target_id,
        action=_click_download_link,
        save_path=download_path,
        timeout=_DOWNLOAD_WAIT_TIMEOUT_S,
        verbose=verbose,
    )
    if not dl_result.get("downloaded"):
        if verbose: print(f"Download failed for {url}", file=sys.stderr)
        return None

    await _cleanup_download_link(mgr, target_id)
    if verbose: print(f"Successfully downloaded {url}")
    return await _rename_to_suggested_name(dl_result, download_path)

def _extract_all_text(node: dict | list | str) -> str:
    """Join every bit of text in the data."""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return " ".join(_extract_all_text(child) for child in node)
    if isinstance(node, dict):
        return " ".join(_extract_all_text(value) for value in node.values())
    return ""


async def _is_content_insufficient(content: str) -> Optional[str]:
    """Say why a downloaded page holds too little to be worth keeping.

    Returns:
        The reason the page is too little, or None when it is worth keeping.
    """
    if not content:
        return "the page is empty"

    # A web page is cut down to its words first.
    filtered_text = content
    if content.lstrip().startswith('<'):
        try:
            json_tree = await html_to_json(content)
            filtered_text = await asyncio.to_thread(_extract_all_text, json_tree)
        except Exception:
            # Keep the raw text when the page cannot be read.
            pass

    if len(filtered_text) < _MIN_CONTENT_LENGTH:
        return f"only {len(filtered_text)} chars (below {_MIN_CONTENT_LENGTH})"

    return None


def _safe_suggested_name(name: str, default_name: str) -> str:
    """Cut a suggested file name down to a plain name.

    Args:
        name: The suggested name, which may hold folder names.
        default_name: The name to fall back on when the suggestion is no good.

    Returns:
        A name that cannot point outside the folder it goes in.
    """
    filename = os.path.basename(name)
    if filename in ("", ".", ".."):
        return default_name
    return filename


def _decode_rfc5987(value: str) -> Optional[str]:
    """Read a file name that comes with its own letter set.

    Args:
        value: The coded name from the header.

    Returns:
        The plain name, or None when it cannot be read.
    """
    charset, _, value = value.partition("'")
    _, _, value = value.partition("'")
    try:
        return unquote(value, encoding=charset or "utf-8")
    except (LookupError, UnicodeDecodeError):
        return None


def _suggested_filename(headers: str, default_name: str) -> str:
    """Find a safe file name in the reply headers.

    Args:
        headers: The reply headers as one block of text.
        default_name: The name to fall back on when the server offers none.

    Returns:
        The name the server asked for, or default_name.
    """
    for line in headers.splitlines():
        name, sep, value = line.partition(":")
        if not sep or name.strip().lower() != "content-disposition":
            continue
        encoded = re.search(r"filename\*=(\S+)", value, re.IGNORECASE)
        if encoded:
            filename = _decode_rfc5987(encoded.group(1))
        else:
            match = re.search(r'filename=(?:"([^"]+)"|(\S+))', value, re.IGNORECASE)
            if not match:
                continue
            filename = match.group(1) or match.group(2)
        if filename:
            return _safe_suggested_name(filename, default_name)
    return default_name


async def _rename_to_suggested_name(dl_result: dict, default_path: str) -> str:
    """Give a download the name the server asked for.

    Returns:
        The path of the file, moved when it got a new name.
    """
    suggested = dl_result.get("filename")
    if not suggested:
        return default_path
    name = _safe_suggested_name(suggested, os.path.basename(default_path))
    if name == os.path.basename(default_path):
        return default_path
    target = os.path.join(os.path.dirname(default_path), name)
    await asyncio.to_thread(shutil.move, default_path, target)
    return target


async def _try_curl_download(url: str, download_path: str, verbose: bool = False) -> Optional[str]:
    """Download whatever sits at a web address.

    Args:
        url: The address to fetch.
        download_path: Where to put the file.
        verbose: True to print what is happening.

    Returns:
        The path of the saved file, or None when the browser should try
        instead.

    Raises:
        ResolveError: If nothing was asked for, or the server answered
            with an error a browser cannot get past.
    """
    if url is None or download_path is None:
        raise ResolveError("url or download_path is None")
    if verbose: print(f"[CURL] Downloading: {url}", file=sys.stderr)

    # Check the answer before pulling the body down.
    status_cmd = ["curl", "-L", "-s", "--compressed", "-o", "/dev/null", "-D", "-", "-w", "\n%{http_code}", url]
    status = await run_cli(status_cmd, "curl status check")
    if status.returncode != 0:
        return None
    headers, _, http_code = status.stdout.rpartition("\n")

    if http_code.isdigit():
        http_code_int = int(http_code)
        if http_code_int in [_HTTP_FORBIDDEN, _HTTP_UNAUTHORIZED]:
            # The browser may still get in.
            return None
        elif _HTTP_CLIENT_ERROR_MIN <= http_code_int <= _HTTP_CLIENT_ERROR_MAX:
            raise ResolveError(f"HTTP {http_code} - Not found or access denied")
        elif _HTTP_SERVER_ERROR_MIN <= http_code_int <= _HTTP_SERVER_ERROR_MAX:
            raise ResolveError(f"HTTP {http_code} - Server error")

    # Save under the name the server asked for.
    file_name = _suggested_filename(headers, os.path.basename(download_path))
    final_path = os.path.join(os.path.dirname(download_path), file_name)
    cmd = ["curl", "-L", "-s", "--compressed", "-o", final_path, url]
    result = await run_cli(cmd, "curl download")

    if result.returncode != 0 or not await asyncio.to_thread(os.path.exists, final_path):
        if verbose: print(f"[CURL] Failed for {url} (rc={result.returncode})", file=sys.stderr)
        return None
    if verbose: print(f"[CURL] Success: Content saved to {final_path}", file=sys.stderr)
    return final_path


async def _resolve_url_browse_and_resolve(mgr: BrowserManager, target_id: str, url: str, download_path: str, context: str, verbose: bool, binary_passthrough: bool = False) -> Tuple[str, bool]:
    """Open a web address in the browser and turn what it shows into content.

    Returns:
        The content and a flag that is True when a model already described it.

    Raises:
        ResolveError: If the page could not be turned into content.
    """
    dl_result = await mgr.wait_for_download(target_id=target_id, url=url, save_path=download_path, timeout=_DOWNLOAD_WAIT_TIMEOUT_S, verbose=verbose)
    if dl_result.get("downloaded"):
        renamed_path = await _rename_to_suggested_name(dl_result, download_path)
        return _require_content(await _handle_download_result(renamed_path, context, verbose, binary_passthrough))

    actual_url = mgr.url(target_id, verbose=verbose) or url
    if actual_url != url and verbose:
        print(f"URL redirected: {url} -> {actual_url}")

    content_type = await mgr.evaluate(target_id, "document.contentType", verbose=verbose)
    if content_type is None:
        raise ResolveError("Unable to detect content type from browser")

    path = await _resolve_url_download_by_content_type(mgr, target_id, actual_url, download_path, content_type, verbose)
    if path:
        return _require_content(await _handle_download_result(path, context, verbose, binary_passthrough))

    if verbose: print("Taking web snapshot")
    snapshot = await mgr.snapshot(target_id, verbose=verbose)
    if snapshot is None:
        raise ResolveError(f"Failed to capture snapshot of {actual_url}")
    tree = await snapshot.filtered_tree(verbose=verbose)
    return json.dumps(tree, ensure_ascii=False), False


async def _resolve_url_download_text(mgr: BrowserManager, target_id: Optional[str], actual_url: Optional[str], download_path: Optional[str], verbose: bool = False) -> Optional[str]:
    """Save the text the page is showing.

    Returns:
        The saved path, or None if it could not be saved.
    """
    if target_id is None or actual_url is None or download_path is None:
        if verbose: print(f"None input in _resolve_url_download_text", file=sys.stderr)
        return None
    if verbose: print(f"Processing as textual document: {actual_url}")
    text = await mgr.evaluate(target_id, "document.body.innerText", verbose=verbose)
    if text is None:
        if verbose: print(f"Failed to extract text for {actual_url}", file=sys.stderr)
        return None
    try:
        async with aiofiles.open(download_path, 'w', encoding='utf-8') as f:
            await f.write(text)
    except Exception as e:
        if verbose: print(f"Failed to write text to {download_path}: {e}", file=sys.stderr)
        return None
    return download_path


async def _resolve_url_download_by_content_type(mgr: BrowserManager, target_id: str, actual_url: str, download_path: str, content_type: str, verbose: bool) -> Optional[str]:
    """Save what the page holds when it is text, media, or a PDF.

    Returns:
        The saved path, or None when the page is none of those.

    Raises:
        ResolveError: If the page holds one of those and it could not be
            saved.
    """
    if "json" in content_type or content_type == "text/plain":
        if verbose: print(f"Detected textual content type: {content_type}")
        path = await _resolve_url_download_text(mgr, target_id, actual_url, download_path, verbose=verbose)
        if not path:
            raise ResolveError(f"Failed to download text content: {actual_url}")
        return path
    if "image" in content_type or "video" in content_type or "audio" in content_type:
        if verbose: print("Detected media content type")
        path = await _resolve_url_download_rendered_media(mgr, target_id, actual_url, download_path, verbose=verbose)
        if not path:
            raise ResolveError(f"Failed to download media: {actual_url}")
        return path
    if "pdf" in content_type:
        if verbose: print("Detected pdf content type")
        path = await _resolve_url_download_document(mgr, target_id, actual_url, download_path, verbose)
        if not path:
            raise ResolveError(f"Failed to download PDF: {actual_url}")
        return path
    return None


async def _is_blocked_page(path: str, url: str, verbose: bool) -> bool:
    """Ask Laya whether a saved page is a block page.

    Args:
        path: The saved HTML file.
        url: The address the page came from.
        verbose: True to print what is happening.

    Returns:
        True when the page is blocked or could not be checked.
    """
    async with aiofiles.open(path, 'r', encoding='utf-8', errors='replace') as f:
        html = await f.read()
    try:
        verdict: laya.Verdict = await laya.classify(html)
    except laya.LayaError as e:
        # A page that cannot be checked counts as blocked.
        print(f"[LAYA] unavailable ({e}), using browser for: {url}", file=sys.stderr, flush=True)
        return True
    if verdict.blocked and verbose:
        print(f"[LAYA] blocked (p={verdict.blocked_probability:.2f}), using browser for: {url}", flush=True)
    return verdict.blocked


async def _resolve_url(url: str, context: str = "", verbose: bool = False, mgr: Optional[BrowserManager] = None, binary_passthrough: bool = False) -> Tuple[str, bool]:
    """Turn a web address into content.

    Returns:
        The content and a flag that is True when a model already described it.

    Raises:
        ResolveError: If the address could not be turned into content.
    """
    if not url:
        raise ResolveError("URL is None or empty")
    file_name = url.split("/")[-1].split("?")[0] or "downloaded_file"
    temp_dir = paths.scratch_dir(_TEMP_NAME)
    await asyncio.to_thread(os.makedirs, temp_dir, exist_ok=True)
    download_path = os.path.join(temp_dir, file_name)

    if verbose: print(f"[CURL] Attempting download: {url}")
    try:
        saved_path = await _try_curl_download(url, download_path, verbose)
        if saved_path:
            mime = await paths.mime_type(saved_path)
            res = await _handle_download_result(saved_path, context, verbose, binary_passthrough, mime=mime)
            if res is not None:
                if mime not in _HTML_MIMES or not await _is_blocked_page(saved_path, url, verbose):
                    return res
    except _EmptyHtmlError:
        if verbose: print(f"[FALLBACK] No text left in HTML, using browser for: {url}", flush=True)
    except ResolveError:
        raise
    except Exception as e:
        if verbose: print(f"[FAIL-FAST] {e}", file=sys.stderr)
        raise ResolveError(str(e)) from e
    if verbose: print(f"[BROWSER-FALLBACK] Curl failed, using browser for: {url}")

    target_id = None
    _owns_mgr = mgr is None
    try:
        if _owns_mgr:
            mgr = await BrowserManager.create()
        target_id = await mgr.new_page(verbose=verbose)
        if not target_id:
            raise ResolveError("Unable to open new browser tab")

        try:
            return await _resolve_url_browse_and_resolve(mgr, target_id, url, download_path, context, verbose, binary_passthrough)
        except ResolveError:
            raise
        except Exception as e:
            raise ResolveError(str(e)) from e
    except ResolveError:
        raise
    except Exception as e:
        if verbose: print(f"Error resolving URL: {str(e)}", file=sys.stderr)
        raise ResolveError(f"Error resolving URL: {str(e)}") from e
    finally:
        if target_id and mgr is not None:
            await mgr.close_page(target_id, verbose=verbose)
        if _owns_mgr and mgr is not None:
            await mgr.cleanup()

async def _resolve_audio(audio_path: str, verbose: bool = False) -> str:
    """Write down the words spoken in an audio file.

    Args:
        audio_path: The audio file to listen to.
        verbose: True to print what is happening.

    Returns:
        The words that were said.

    Raises:
        ResolveError: If the words could not be written down.
    """
    if audio_path is None:
        raise ResolveError("Audio path is None")
    try:
        url = config.require_env("ASR_SERVER_URL", "needed by audio and video transcription")
    except config.MissingSettingError as exc:
        raise ResolveError(str(exc)) from exc
    try:
        if verbose:
            print(f"Transcribing audio via ASR server: {audio_path}", file=sys.stderr)

        async with aiofiles.open(audio_path, "rb") as f:
            audio_data = await f.read()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{url.rstrip('/')}/asr",
                files={"audio_file": (os.path.basename(audio_path), audio_data)},
                timeout=300
            )

        if resp.status_code != 200:
            if verbose:
                print(f"ASR server returned HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
            raise ResolveError(f"ASR transcription failed (HTTP {resp.status_code}): {resp.text}")

        transcription = resp.text.strip()
        if not transcription:
            raise ResolveError("ASR transcription returned empty result")

        return transcription
    except ResolveError:
        raise
    except Exception as e:
        raise ResolveError(f"Error during audio transcription: {str(e)}") from e



async def resolve_input(text: str, context: str = "", verbose: bool = False, binary_passthrough: bool = False) -> Tuple[str, bool]:
    """Turn a web address, a file, or plain text into text.

    Args:
        text: A web address, a file path, or plain text. A single dash reads
            from standard input.
        context: A question about a picture, a video, a PDF, or slides.
            Empty asks for a description.
        verbose: True to print what is happening.
        binary_passthrough: True to copy a small binary file to the downloads
            folder and give back the tag and the path of the copy.

    Returns:
        The content and a flag that is True when a model already described it.

    Raises:
        ResolveError: If the input could not be turned into content.
    """
    if context is None:
        context = ""
    if text is None:
        raise ResolveError("Input text is None")

    if text == "-":
        try:
            if verbose: print("Reading input from stdin...", file=sys.stderr)
            return sys.stdin.read(), False
        except Exception as e:
            raise ResolveError(f"Error reading from stdin: {str(e)}") from e

    parsed_url = urlparse(text.strip())
    if parsed_url.scheme == _FILE_SCHEME:
        return await _resolve_file(parsed_url.path, context, verbose, binary_passthrough)
    if parsed_url.scheme and parsed_url.netloc:
        url = text.strip()
        return await _resolve_url(url, context=context, verbose=verbose, binary_passthrough=binary_passthrough)

    if await asyncio.to_thread(os.path.isfile, text):
        return await _resolve_file(text, context, verbose, binary_passthrough)

    # A path that points at nothing is kept as plain text.
    return text, False

async def _resolve_file(path: str, context: str = "", verbose: bool = False, binary_passthrough: bool = False) -> Tuple[str, bool]:
    """Turn a local file into content.

    Args:
        path: The file to read.
        context: A question about a picture, a video, a PDF, or slides.
            Empty asks for a description.
        verbose: True to print what is happening.
        binary_passthrough: True to keep a small binary file for download.

    Returns:
        The content and a flag that is True when a model already described it.

    Raises:
        ResolveError: If the file could not be turned into content.
    """
    if not path:
        raise ResolveError("File path is None or empty")
    if not await asyncio.to_thread(os.path.exists, path):
        raise ResolveError(f"File not found: {path}")
    mime_type = await paths.mime_type(path)
    if binary_passthrough:
        res = await _maybe_binary_passthrough(path, mime_type, verbose)
        if res is not None:
            return res

    if mime_type:
        if paths.is_pdf_mime(mime_type):
            res, status = await describe_pdf(path, context, verbose)
            if status != 0:
                raise ResolveError(res)
            return res, True
        if paths.is_image_mime(mime_type):
            res, status = await describe_image(path, context, verbose=verbose)
            if status != 0:
                raise ResolveError(res)
            return res, True
        if paths.is_video_mime(mime_type):
            res, status = await describe_video(path, context, verbose=verbose)
            if status != 0:
                raise ResolveError(res)
            return res, True
        if paths.is_audio_mime(mime_type):
            res = await _resolve_audio(path, verbose)
            return res, False
        if mime_type in _OFFICE_TEXT_PARSERS or mime_type in _MIME_PRESENTATIONS:
            return await _resolve_office_doc(path, mime_type, context, verbose)
        if mime_type == "text/html" or path.endswith((".html", ".htm")):
            res = await filter_html_file(path, verbose)
            if res is None:
                raise _EmptyHtmlError(f"Failed to extract text from HTML: {path}")
            if isinstance(res, dict):
                res = json.dumps(res, ensure_ascii=False)
            return res, False

    # A file of any other kind is read as text.
    try:
        return await _read_text_file_strict(path), False
    except UnicodeDecodeError:
        return await _save_binary_file(path, verbose)
    except Exception as e:
        raise ResolveError(f"Error reading file '{path}': {str(e)}") from e


async def _resolve_office_doc(path: str, mime_type: str, context: str, verbose: bool) -> Tuple[str, bool]:
    """Turn an Office file into content.

    Returns:
        The content and a flag that is True when a model already described it.

    Raises:
        ResolveError: If the file could not be turned into content.
    """
    parser = _OFFICE_TEXT_PARSERS.get(mime_type)
    if parser is not None:
        try:
            text = await parser(path, verbose)
        except Exception as e:
            raise ResolveError(f"Error extracting text from '{path}': {str(e)}") from e
        return text, False
    res, status = await describe_presentation(path, context, verbose)
    if status != 0:
        raise ResolveError(res)
    return res, True


async def _save_binary_file(path: str, verbose: bool) -> Tuple[str, bool]:
    """Copy a file that cannot be read into the share people download from.

    Args:
        path: The file to copy.
        verbose: True to print what is happening.

    Returns:
        A note saying how to download the copy.

    Raises:
        ResolveError: If the copy failed.
    """
    filename = os.path.basename(path)
    dest = _binary_download_dest(path)
    try:
        await asyncio.to_thread(_copy_into_downloads, path, dest)
    except Exception as e:
        raise ResolveError(f"Error saving binary file '{path}': {str(e)}") from e
    if verbose:
        print(f"[BINARY] Saved {path} to {dest}", file=sys.stderr)
    notice = await binary_download_notice(f"{_BINARY_DOWNLOAD_DIR}/{filename}")
    return notice, False


def _copy_into_downloads(src: str, dest: str) -> None:
    """Copy a file, making the folder it goes in when that is missing."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(src, dest)


def _binary_download_dest(path: str) -> str:
    """Where a copy of a file goes in the share people download from."""
    return os.path.join(paths.resources_dir(), _BINARY_DOWNLOAD_DIR, os.path.basename(path))


async def _passthrough_binary(path: str, verbose: bool) -> Tuple[str, bool]:
    """Copy a binary file into the share people download from.

    Returns:
        The tag and the path of the copy.

    Raises:
        ResolveError: If the copy failed.
    """
    dest = _binary_download_dest(path)
    try:
        await asyncio.to_thread(_copy_into_downloads, path, dest)
    except Exception as e:
        raise ResolveError(f"Error saving binary file '{path}': {str(e)}") from e
    if verbose:
        print(f"[BINARY-PASSTHROUGH] {path} -> {dest}", file=sys.stderr)
    return BINARY_PASSTHROUGH_TAG + dest, False


async def _maybe_binary_passthrough(path: str, mime_type: Optional[str], verbose: bool) -> Optional[Tuple[str, bool]]:
    """Copy a small binary file into the share people download from.

    Returns:
        The tag and the path of the copy, or None when the file is text or
        too big.
    """
    size = await asyncio.to_thread(os.path.getsize, path)
    is_bin = paths.is_binary_mime(mime_type)
    if is_bin is None:
        try:
            await _read_text_file_strict(path)
            return None  # The file reads as text.
        except UnicodeDecodeError:
            is_bin = True
    if is_bin and size <= _BINARY_PASSTHROUGH_CAP:
        return await _passthrough_binary(path, verbose)
    return None
