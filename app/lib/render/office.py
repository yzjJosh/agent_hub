"""Convert Office files into HTML pages and PDF previews for the viewer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import datetime
import html
import os
import re
import shutil

from bs4 import BeautifulSoup

from app.lib.render import cache
from app.lib.render.classify import RenderKind, classify
from app.lib.render.fragments import build_fragment, table_html, text_meta
from app.lib.resolver.office_docs import SheetCell, soffice_cmd, xls_sheets, xlsx_sheets
from app.lib.shared import paths

__all__ = [
    "ensure_doc_for_thumb",
    "retry_office_conversion",
    "start_office_conversion",
]

_CONVERSION_TIMEOUT_SECONDS = 90.0
_SCRATCH_FOLDER_NAME = "render"
_MAX_CONCURRENT_CONVERSIONS = 2
_MAX_ERROR_LENGTH = 200
_TASK_NAME_KEY_PREFIX_LENGTH = 8
_SCRATCH_ID_BYTES = 4
_OFFICE_KINDS = frozenset(
    {RenderKind.OFFICE_DOC, RenderKind.OFFICE_PPT, RenderKind.XLSX, RenderKind.XLS}
)
_REMOVED_HTML_TAGS = ("script", "object", "embed", "iframe", "head", "style", "link")
_ABSOLUTE_URL_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*:")
_LEADING_CONTROL_OR_SPACE_PATTERN = re.compile(r"^[\x00-\x20]+")
_ALLOWED_URL_SCHEMES = ("http", "https", "mailto", "ftp")
_URL_ATTRIBUTES = ("href", "src", "xlink:href")


@dataclass(frozen=True)
class _PreparedDocument:
    fragment_html: str | None
    metadata: dict


@dataclass(frozen=True)
class _LibreOfficeOutput:
    html: str
    assets: dict[str, bytes]


@dataclass(frozen=True)
class _SpreadsheetFragment:
    html: str
    row_count: int
    column_count: int | None


@dataclass(frozen=True)
class _ConvertedSpreadsheet:
    html: str
    metadata: dict


class _ConversionTimeoutError(Exception):
    """Raised when an Office conversion exceeds its time limit."""


class _ConversionError(Exception):
    """Raised when an Office conversion fails."""


class _ActiveConversion:
    """Running conversion task and completion event for one cache key."""

    def __init__(self, key: str, task: asyncio.Task[None]) -> None:
        self.key = key
        self.task = task
        self.done_event = asyncio.Event()
        task.add_done_callback(self._on_done)

    def _on_done(self, _task: asyncio.Task[None]) -> None:
        self.done_event.set()
        if _ACTIVE_CONVERSIONS.get(self.key) is self:
            _ACTIVE_CONVERSIONS.pop(self.key, None)


_ACTIVE_CONVERSIONS: dict[str, _ActiveConversion] = {}
_PENDING_REGISTRATIONS: dict[str, asyncio.Event] = {}
_conversion_semaphore: asyncio.Semaphore | None = None
_conversion_semaphore_loop: asyncio.AbstractEventLoop | None = None


def _get_conversion_semaphore() -> asyncio.Semaphore:
    global _conversion_semaphore, _conversion_semaphore_loop
    loop = asyncio.get_running_loop()
    if _conversion_semaphore is None or _conversion_semaphore_loop is not loop:
        _conversion_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_CONVERSIONS)
        _conversion_semaphore_loop = loop
    return _conversion_semaphore


def _get_active_conversion(key: str) -> _ActiveConversion | None:
    active_conversion: _ActiveConversion | None = _ACTIVE_CONVERSIONS.get(key)
    if active_conversion is None or active_conversion.task.done():
        _ACTIVE_CONVERSIONS.pop(key, None)
        return None
    return active_conversion


async def start_office_conversion(
    path: str,
    key: str,
    full_path: str,
    timeout: float = _CONVERSION_TIMEOUT_SECONDS,
) -> None:
    """Start converting an Office file, or join the conversion already running.

    Args:
        path: Resource-relative path to the file.
        key: Cache key for the file.
        full_path: Absolute path to the file on disk.
        timeout: Maximum seconds the conversion may run before failing.
    """
    if _get_active_conversion(key) is not None:
        return
    registration_event = _PENDING_REGISTRATIONS.get(key)
    if registration_event is None:
        registration_event = asyncio.Event()
        _PENDING_REGISTRATIONS[key] = registration_event
        await _register_conversion(key, path, full_path, timeout, registration_event)
        return
    await _wait_for_registration(key, path, full_path, registration_event, timeout)


async def _register_conversion(
    key: str,
    path: str,
    full_path: str,
    timeout: float,
    registration_event: asyncio.Event | None = None,
) -> None:
    try:
        kind = await classify(full_path)
        if _get_active_conversion(key) is not None:
            return
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            _run_conversion_task(key, path, full_path, kind, timeout),
            name=f"office-{key[:_TASK_NAME_KEY_PREFIX_LENGTH]}",
        )
        _ACTIVE_CONVERSIONS[key] = _ActiveConversion(key, task)
        await asyncio.sleep(0)
    finally:
        if registration_event is not None:
            _PENDING_REGISTRATIONS.pop(key, None)
            registration_event.set()


async def _wait_for_registration(
    key: str,
    path: str,
    full_path: str,
    registration_event: asyncio.Event,
    timeout: float,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _CONVERSION_TIMEOUT_SECONDS
    while True:
        if _get_active_conversion(key) is not None:
            return
        current_registration = _PENDING_REGISTRATIONS.get(key)
        if current_registration is None:
            await _register_conversion(key, path, full_path, timeout)
            return
        try:
            await asyncio.wait_for(registration_event.wait(), deadline - loop.time())
        except asyncio.TimeoutError:
            raise _ConversionError("Conversion registration did not complete in time") from None
        if current_registration is not registration_event:
            registration_event = current_registration


async def retry_office_conversion(path: str, key: str, full_path: str) -> bool:
    """Clear cached state and start a fresh conversion for an Office file.

    Args:
        path: Resource-relative path to the file.
        key: Cache key for the file.
        full_path: Absolute path to the file on disk.

    Returns:
        True when a fresh conversion started, or False when an active conversion was joined.
    """
    if _get_active_conversion(key) is not None:
        return False
    await _delete_cached_conversion_state(key)
    await start_office_conversion(path, key, full_path)
    return True


async def _delete_cached_conversion_state(key: str) -> None:
    await asyncio.to_thread(shutil.rmtree, os.path.join(cache.render_root(), key), ignore_errors=True)
    await cache.clear_failed(key, "convert")
    await cache.clear_failed(key, "thumb")


async def ensure_doc_for_thumb(kind: RenderKind, key: str, path: str, full_path: str) -> str:
    """Make sure a file has a cached preview ready for thumbnail capture.

    Args:
        kind: Classified kind of the file.
        key: Cache key for the file.
        path: Resource-relative path to the file.
        full_path: Absolute path to the file on disk.

    Returns:
        published when the cached preview is ready, or failed when it is not.
    """
    if kind in _OFFICE_KINDS:
        return await _ensure_office_document(key, path, full_path)
    return await _ensure_text_document(kind, key, path, full_path)


async def _ensure_office_document(key: str, path: str, full_path: str) -> str:
    if await cache.read_doc(key) is not None:
        return "published"
    active_conversion = _get_active_conversion(key)
    if active_conversion is not None:
        await active_conversion.done_event.wait()
        return "published" if await cache.read_doc(key) is not None else "failed"
    if await cache.convert_failed_reason(key) is not None:
        return "failed"
    await start_office_conversion(path, key, full_path)
    active_conversion = _get_active_conversion(key)
    if active_conversion is not None:
        await active_conversion.done_event.wait()
    return "published" if await cache.read_doc(key) is not None else "failed"


async def _ensure_text_document(kind: RenderKind, key: str, path: str, full_path: str) -> str:
    if await cache.read_doc(key) is not None:
        return "published"
    try:
        prepared = await _build_document_fragment(kind, path, full_path)
    except OSError:
        return "failed"
    await cache.write_doc(key, prepared.fragment_html or "", None, prepared.metadata)
    return "published"


async def _build_document_fragment(kind: RenderKind, path: str, full_path: str) -> _PreparedDocument:
    if kind is RenderKind.HTML:
        frame_html = (
            f'<iframe class="html-frame" src="{paths.ephemeral_dav_url(path)}"'
            ' sandbox="allow-same-origin"></iframe>'
        )
        return _PreparedDocument(
            fragment_html=frame_html,
            metadata={"path": path, "kind": kind.value, "mode": "frame"},
        )
    file_stat = await asyncio.to_thread(os.stat, full_path)
    file_size = file_stat.st_size
    fragment_html, extras = await build_fragment(kind, full_path, file_size)
    metadata = text_meta(kind, path, int(file_stat.st_mtime), file_size, extras)
    return _PreparedDocument(fragment_html=fragment_html, metadata=metadata)


def _format_failure_reason(error: BaseException) -> str:
    if isinstance(error, _ConversionTimeoutError):
        return f"Conversion timed out after {int(_CONVERSION_TIMEOUT_SECONDS)} s"
    return (str(error) or "conversion failed")[:_MAX_ERROR_LENGTH]


async def _run_conversion_task(
    key: str, path: str, full_path: str, kind: RenderKind, timeout: float
) -> None:
    try:
        await _convert_office_file(key, path, full_path, kind, timeout)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        await cache.mark_failed(key, "convert", _format_failure_reason(error))


async def _convert_office_file(
    key: str, path: str, full_path: str, kind: RenderKind, timeout: float
) -> None:
    async with _get_conversion_semaphore():
        if kind in (RenderKind.XLSX, RenderKind.XLS):
            try:
                converted = await asyncio.wait_for(
                    _convert_spreadsheet(path, full_path, kind), timeout
                )
            except asyncio.TimeoutError:
                raise _ConversionError(
                    f"Sheet read failed: timed out after {int(_CONVERSION_TIMEOUT_SECONDS)} s"
                ) from None
            await cache.write_doc(key, converted.html, None, converted.metadata)
            return
        if kind is RenderKind.OFFICE_PPT:
            output = await _convert_with_libreoffice(full_path, timeout, "pdf")
            metadata = await _build_office_metadata(path, full_path, kind, "soffice-pdf")
            metadata.update({"mode": "frame", "row_count": None, "column_count": None})
            await cache.write_doc(key, output.html, output.assets, metadata)
            return
        output = await _convert_with_libreoffice(full_path, timeout)
        sanitized_html = await _sanitize_office_html(output.html, key, output.assets)
        metadata = await _build_office_metadata(path, full_path, kind, "soffice-html")
        metadata.update({"row_count": None, "column_count": None})
        await cache.write_doc(key, sanitized_html, output.assets, metadata)


async def _build_office_metadata(path: str, full_path: str, kind: RenderKind, engine: str) -> dict:
    file_stat = await asyncio.to_thread(os.stat, full_path)
    extension = os.path.splitext(full_path)[1]
    is_sheet = engine == "sheet-reader"
    provenance = (
        f"Rendered from {extension} with the built-in sheet reader"
        if is_sheet
        else f"Converted from {extension} with LibreOffice"
    )
    return {
        "path": path,
        "mtime": int(file_stat.st_mtime),
        "size": file_stat.st_size,
        "kind": kind.value,
        "engine": engine,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "language": None,
        "line_count": None,
        "provenance": provenance,
        "mode": "table" if is_sheet else "doc",
    }


def _read_libreoffice_output(output_folder: str, stem: str, target: str) -> _LibreOfficeOutput:
    if target == "pdf":
        with open(os.path.join(output_folder, f"{stem}.pdf"), "rb") as file_handle:
            return _LibreOfficeOutput(html="", assets={cache.OFFICE_PDF_ASSET: file_handle.read()})
    main_html_path = os.path.join(output_folder, f"{stem}.html")
    with open(main_html_path, "r", encoding="utf-8") as file_handle:
        html_text = file_handle.read()
    assets: dict[str, bytes] = {}
    for filename in sorted(os.listdir(output_folder)):
        if filename in (f"{stem}.html", "profile"):
            continue
        asset_path = os.path.join(output_folder, filename)
        if os.path.isfile(asset_path):
            with open(asset_path, "rb") as file_handle:
                assets[filename] = file_handle.read()
    return _LibreOfficeOutput(html=html_text, assets=assets)


async def _convert_with_libreoffice(
    full_path: str, timeout: float, target: str = "html"
) -> _LibreOfficeOutput:
    output_folder = os.path.join(
        paths.scratch_dir(_SCRATCH_FOLDER_NAME),
        f"render_{os.urandom(_SCRATCH_ID_BYTES).hex()}",
    )
    stem = os.path.splitext(os.path.basename(full_path))[0]
    main_output_path = os.path.join(output_folder, f"{stem}.{target}")
    await asyncio.to_thread(os.makedirs, output_folder, exist_ok=True)
    process = await asyncio.create_subprocess_exec(
        *soffice_cmd(output_folder, target, full_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        try:
            _stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise _ConversionTimeoutError("soffice timed out") from None
        except asyncio.CancelledError:
            process.kill()
            try:
                await process.wait()
            finally:
                raise
        stderr_tail = stderr.decode("utf-8", "replace").strip()[-_MAX_ERROR_LENGTH:]
        if process.returncode != 0 or not await asyncio.to_thread(os.path.exists, main_output_path):
            raise _ConversionError(
                f"LibreOffice failed: {stderr_tail or 'no output file produced'}"
            )
        return await asyncio.to_thread(_read_libreoffice_output, output_folder, stem, target)
    finally:
        await asyncio.to_thread(shutil.rmtree, output_folder, ignore_errors=True)


async def _convert_spreadsheet(
    path: str, full_path: str, kind: RenderKind
) -> _ConvertedSpreadsheet:
    try:
        sheets = await xlsx_sheets(full_path) if kind is RenderKind.XLSX else await xls_sheets(full_path)
        fragment = await _build_spreadsheet_fragment(sheets)
    except Exception as error:
        raise _ConversionError(f"Sheet read failed: {error}") from error
    metadata = await _build_office_metadata(path, full_path, kind, "sheet-reader")
    metadata.update({"row_count": fragment.row_count, "column_count": fragment.column_count})
    return _ConvertedSpreadsheet(html=fragment.html, metadata=metadata)


def _format_cell_value(cell_value: SheetCell | None) -> str:
    return "" if cell_value is None else str(cell_value)


def _build_spreadsheet_fragment_sync(
    sheets: list[tuple[str, list[tuple[SheetCell | None, ...]]]]
) -> _SpreadsheetFragment:
    sections: list[str] = []
    total_rows = 0
    first_column_count: int | None = None
    for title, rows in sheets:
        non_empty_rows = [row for row in rows if any(cell_value is not None for cell_value in row)]
        if not non_empty_rows:
            continue
        if first_column_count is None:
            first_column_count = len(non_empty_rows[0])
        header_row, data_rows = non_empty_rows[0], non_empty_rows[1:]
        total_rows += len(data_rows)
        table = table_html(
            [_format_cell_value(cell) for cell in header_row],
            [[_format_cell_value(cell) for cell in row] for row in data_rows],
        )
        sections.append(
            f'<section class="sheet"><h3 class="sheet-title">{html.escape(title)}</h3>{table}</section>'
        )
    if not sections:
        raise _ConversionError("Sheet read failed: workbook has no rows")
    return _SpreadsheetFragment(
        html="".join(sections),
        row_count=total_rows,
        column_count=first_column_count,
    )


async def _build_spreadsheet_fragment(
    sheets: list[tuple[str, list[tuple[SheetCell | None, ...]]]]
) -> _SpreadsheetFragment:
    return await asyncio.to_thread(_build_spreadsheet_fragment_sync, sheets)


def _is_allowed_url_scheme(url_value: str) -> bool:
    cleaned = _LEADING_CONTROL_OR_SPACE_PATTERN.sub("", url_value)
    if cleaned.startswith(("//", "/")):
        return True
    scheme, separator, _rest = cleaned.partition(":")
    return not separator or scheme.lower() in _ALLOWED_URL_SCHEMES


def _sanitize_office_html_sync(html_text: str, key: str, assets: dict[str, bytes]) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup.find_all(_REMOVED_HTML_TAGS):
        if tag.parent is not None:
            tag.decompose()
    for tag in soup.find_all(True):
        if tag.has_attr("style"):
            del tag["style"]
        for attribute_name in list(tag.attrs):
            attribute_value = tag.attrs[attribute_name]
            is_event_handler = attribute_name.lower().startswith("on")
            is_unsafe_url = (
                attribute_name in _URL_ATTRIBUTES
                and isinstance(attribute_value, str)
                and not _is_allowed_url_scheme(attribute_value)
            )
            if is_event_handler or is_unsafe_url:
                del tag[attribute_name]
    for image_tag in soup.find_all("img"):
        source = image_tag.get("src")
        if (
            isinstance(source, str)
            and source
            and not _ABSOLUTE_URL_PATTERN.match(source)
            and not source.startswith(("//", "/"))
        ):
            asset_name = os.path.basename(source.split("?")[0])
            if asset_name in assets:
                image_tag["src"] = f"/view/doc/{key}/{asset_name}"
    body = soup.body
    content = body.decode_contents() if body is not None else soup.decode_contents()
    return f'<div class="md-doc">{content}</div>'


async def _sanitize_office_html(html_text: str, key: str, assets: dict[str, bytes]) -> str:
    return await asyncio.to_thread(_sanitize_office_html_sync, html_text, key, assets)
