"""Turns Office files into pages the viewer can show.

Only one conversion runs at a time for the same file.
"""

from __future__ import annotations

import asyncio
import datetime
import html as _html
import importlib
import os
import re
import shutil
from typing import Protocol

from bs4 import BeautifulSoup

from app.lib.render import cache
from app.lib.render.classify import RenderKind, classify
from app.lib.render.fragments import build_fragment, table_html, text_meta
from app.lib.resolver.office_docs import soffice_cmd, xls_sheets, xlsx_sheets
from app.lib.shared import paths

__all__ = [
    "ensure_doc_for_thumb",
    "retry_office_conversion",
    "start_office_conversion",
]

_TIMEOUT = 90.0
_SCRATCH_NAME = "render"
# Office kinds that must be converted before a thumbnail can be taken.
_OFFICE_KINDS = frozenset(
    {RenderKind.OFFICE_DOC, RenderKind.OFFICE_PPT, RenderKind.XLSX, RenderKind.XLS}
)
# Files with a conversion running right now, keyed by the saved name.
_CONVERSIONS: dict[str, _Conversion] = {}
# A gate per file that a caller has claimed but not registered yet.
_GATES: dict[str, asyncio.Event] = {}
_sem: asyncio.Semaphore | None = None
_sem_loop: asyncio.AbstractEventLoop | None = None


class _ConversionTimeout(Exception):
    """The conversion took too long."""


class _ConversionError(Exception):
    """The conversion failed."""


def _semaphore() -> asyncio.Semaphore:
    """Get the guard that lets only two conversions run at once.

    Returns:
        The guard.
    """
    global _sem, _sem_loop
    loop = asyncio.get_running_loop()
    if _sem is None or _sem_loop is not loop:
        _sem = asyncio.Semaphore(2)
        _sem_loop = loop
    return _sem


class _Conversion:
    """One file's running conversion and the signal that it is done."""

    def __init__(self, key: str, task: asyncio.Task) -> None:
        self.key = key
        self.task = task
        self.done = asyncio.Event()
        task.add_done_callback(self._on_done)

    def _on_done(self, _task: asyncio.Task) -> None:
        self.done.set()
        # The entry is dropped when it still points at this conversion.
        if _CONVERSIONS.get(self.key) is self:
            _CONVERSIONS.pop(self.key, None)


def _live_conversion(key: str) -> _Conversion | None:
    """Find the conversion still running for a file.

    Args:
        key: The name the file is saved under.

    Returns:
        The running conversion, or None when there is none.
    """
    entry: _Conversion | None = _CONVERSIONS.get(key)
    if entry is None or entry.task.done():
        _CONVERSIONS.pop(key, None)
        return None
    return entry


async def start_office_conversion(path: str, key: str, full_path: str,
                                  timeout: float = _TIMEOUT) -> None:
    """Start converting a file, or join the conversion already going.

    Returns as soon as a conversion is in progress for the file, without
    waiting for it to finish.

    Args:
        path: A path inside the shared folder.
        key: The name the file is saved under.
        full_path: Where the file is on disk.
        timeout: How long the conversion may run before it fails.
    """
    # This decision has no await in it, so no two callers can claim the file.
    handle = _live_conversion(key)
    if handle is not None:
        return  # A conversion is in progress, so this one joins it.
    gate = _GATES.get(key)
    if gate is None:
        _GATES[key] = asyncio.Event()
        await _claim(key, path, full_path, _GATES[key], timeout, hold=True)
        return
    await _wait_on_gate(key, path, full_path, gate, timeout)


async def _claim(key: str, path: str, full_path: str, gate: asyncio.Event | None, timeout: float,
                 hold: bool) -> None:
    """Classify a file, then spawn and register its conversion if still free.

    When hold is set the caller owns a gate that waiters sleep on. The gate is
    removed and signalled whether or not a conversion is registered. A claim
    that finds a conversion already running registers nothing.

    Args:
        key: The name the file is saved under.
        path: A path inside the shared folder.
        full_path: Where the file is on disk.
        gate: The gate the waiters for this file sleep on, or None for a gateless claim.
        timeout: How long the conversion may run before it fails.
        hold: Whether this claim owns a gate that must be released on exit.
    """
    try:
        kind = await classify(full_path)
        if _live_conversion(key) is not None:
            return  # A conversion is in progress, so this one registers nothing.
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            _run_conversion(key, path, full_path, kind, timeout), name=f"office-{key[:8]}"
        )
        _CONVERSIONS[key] = _Conversion(key, task)
        await asyncio.sleep(0)  # let the spawned conversion get going before we return
    finally:
        if hold and gate is not None:
            _GATES.pop(key, None)
            gate.set()


async def _wait_on_gate(key: str, path: str, full_path: str, gate: asyncio.Event, timeout: float) -> None:
    """Wait for the current claim to register or give up, then decide again.

    A join of a live conversion, a takeover of a given-up claim, or a wait on
    a newer gate all end the wait. The wait is bounded by the conversion
    timeout, and a waiter that times out gives up without taking over.

    Args:
        key: The name the file is saved under.
        path: A path inside the shared folder.
        full_path: Where the file is on disk.
        gate: The gate this caller first found waiting on.
        timeout: How long the conversion may run before it fails.

    Raises:
        _ConversionError: If the registration wait takes too long.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TIMEOUT
    while True:
        # Re-run the same await-free decision the start makes on every wake.
        handle = _live_conversion(key)
        if handle is not None:
            return  # join the running conversion and stop here
        current = _GATES.get(key)
        if current is None:
            # The claim gave up without registering, so take the file over.
            # The claim runs gateless, so a waiter woken onto the same gap takes
            # over too, and the two register at most one conversion.
            await _claim(key, path, full_path, None, timeout, hold=False)
            return
        try:
            await asyncio.wait_for(gate.wait(), deadline - loop.time())
        except asyncio.TimeoutError:
            raise _ConversionError("Conversion registration did not complete in time") from None
        if current is not gate:
            gate = current


async def retry_office_conversion(path: str, key: str, full_path: str) -> None:
    """Convert a file again from the start.

    The saved page, the failure marks, and the thumbnail are all thrown away
    first. A conversion already going is joined instead.

    Args:
        path: A path inside the shared folder.
        key: The name the file is saved under.
        full_path: Where the file is on disk.
    """
    if _live_conversion(key) is not None:
        return  # the running conversion is joined, nothing is deleted
    await _clear_key_state(key)
    thumbs = importlib.import_module("app.lib.render.thumbs")
    kind = await classify(full_path)
    thumbs.start_thumbnail_job(key, kind, path)
    await start_office_conversion(path, key, full_path)


async def _clear_key_state(key: str) -> None:
    await asyncio.to_thread(shutil.rmtree, os.path.join(cache.render_root(), key), ignore_errors=True)
    for kind in ("convert", "thumb"):
        await cache.clear_failed(key, kind)


async def ensure_doc_for_thumb(kind: RenderKind, key: str, path: str, full_path: str) -> str:
    """Make sure a file has a saved page a thumbnail can be taken from.

    Args:
        kind: What sort of file this is.
        key: The name the file is saved under.
        path: A path inside the shared folder.
        full_path: Where the file is on disk.

    Returns:
        published when the page is ready, failed when it is not.
    """
    if kind in _OFFICE_KINDS:
        return await _ensure_office(key, path, full_path)
    return await _ensure_doc_kind(kind, key, path, full_path)


async def _ensure_office(key: str, path: str, full_path: str) -> str:
    """Make sure an Office file has a saved page.

    A file already marked as failed is not tried again. A file that is still
    converting is waited out here, the one place in the system that does.

    Args:
        key: The name the file is saved under.
        path: A path inside the shared folder.
        full_path: Where the file is on disk.

    Returns:
        published when the page is ready, failed when it is not.
    """
    if await cache.read_doc(key) is not None:
        return "published"
    handle = _live_conversion(key)
    if handle is not None:
        await handle.done.wait()
        return "published" if await cache.read_doc(key) is not None else "failed"
    if await cache.convert_failed_reason(key) is not None:
        return "failed"
    await start_office_conversion(path, key, full_path)
    handle = _live_conversion(key)
    if handle is not None:
        await handle.done.wait()
    return "published" if await cache.read_doc(key) is not None else "failed"


async def _ensure_doc_kind(kind: RenderKind, key: str, path: str, full_path: str) -> str:
    """Make sure a document file has a saved page.

    No failure mark is left behind when the page cannot be made.

    Args:
        kind: What sort of file this is.
        key: The name the file is saved under.
        path: A path inside the shared folder.
        full_path: Where the file is on disk.

    Returns:
        published when the page is ready, failed when it is not.
    """
    if await cache.read_doc(key) is not None:
        return "published"
    try:
        fragment, meta = await _doc_fragment_for_key(kind, path, full_path)
    except OSError:
        return "failed"
    # A file that does not read as text is saved with an empty page.
    await cache.write_doc(key, fragment or "", None, meta)
    return "published"


async def _doc_fragment_for_key(kind: RenderKind, path: str, full_path: str) -> tuple[str | None, dict]:
    """Build the page and the saved facts for a document file.

    Args:
        kind: What sort of file this is.
        path: A path inside the shared folder.
        full_path: Where the file is on disk.

    Returns:
        The page to save, or None, plus the facts to save with it.
    """
    if kind is RenderKind.HTML:
        return (
            f'<iframe class="html-frame" src="{paths.ephemeral_dav_url(path)}" sandbox="allow-same-origin"></iframe>',
            {"path": path, "kind": kind.value, "mode": "frame"},
        )
    stat = await asyncio.to_thread(os.stat, full_path)
    size = stat.st_size
    fragment, extras = await build_fragment(kind, full_path, size)
    # The descriptor code saves this same entry.
    meta = text_meta(kind, path, int(stat.st_mtime), size, extras)
    return fragment, meta


def _bounded_reason(exc: BaseException) -> str:
    """Turn a failure into a short message people can read.

    Args:
        exc: What went wrong.

    Returns:
        A short message, no longer than 200 letters.
    """
    if isinstance(exc, _ConversionTimeout):
        return "Conversion timed out after 90 s"
    return (str(exc) or "conversion failed")[:200]


async def _run_conversion(key: str, path: str, full_path: str, kind: RenderKind, timeout: float) -> None:
    """Run one conversion, and mark it if it fails.

    A failure writes its mark and is not re-raised. A cancellation writes no
    mark and is re-raised.

    Args:
        key: The name the file is saved under.
        path: A path inside the shared folder.
        full_path: Where the file is on disk.
        kind: What sort of file this is.
        timeout: How long the conversion may run before it fails.
    """
    try:
        await _conversion(key, path, full_path, kind, timeout)
    except asyncio.CancelledError:
        # No mark is written. The next call starts a fresh conversion.
        raise
    except Exception as exc:
        await cache.mark_failed(key, "convert", _bounded_reason(exc))


async def _conversion(
    key: str, path: str, full_path: str, kind: RenderKind, timeout: float,
) -> None:
    """Convert one file and save the page it makes.

    The page is written before the coroutine returns, so a finished
    conversion is always readable.

    Args:
        key: The name the file is saved under.
        path: A path inside the shared folder.
        full_path: Where the file is on disk.
        kind: What sort of file this is.
        timeout: How long the conversion may run before it fails.
    """
    async with _semaphore():
        if kind in (RenderKind.XLSX, RenderKind.XLS):
            try:
                doc, assets, meta = await asyncio.wait_for(
                    _sheet_tables(path, full_path, kind), timeout
                )
            except asyncio.TimeoutError:
                # The read cannot be stopped, so stop waiting for it.
                raise _ConversionError("Sheet read failed: timed out after 90 s") from None
        elif kind is RenderKind.OFFICE_PPT:
            doc, assets = await _soffice_convert(full_path, timeout, "pdf")
            meta = await asyncio.to_thread(_office_meta, path, full_path, kind, "soffice-pdf")
            meta.update({"mode": "frame", "row_count": None, "column_count": None})
        else:
            html, assets = await _soffice_convert(full_path, timeout)
            doc = await _reduce_office_html(html, key, assets)
            meta = await asyncio.to_thread(_office_meta, path, full_path, kind, "soffice-html")
            meta.update({"row_count": None, "column_count": None})
        await cache.write_doc(key, doc, assets, meta)


def _office_meta(path: str, full_path: str, kind: RenderKind, engine: str) -> dict:
    """Collect the facts saved with a converted Office file.

    Args:
        path: A path inside the shared folder.
        full_path: Where the file is on disk.
        kind: What sort of file this is.
        engine: What made the page.

    Returns:
        The facts to save.
    """
    st = os.stat(full_path)
    ext = os.path.splitext(full_path)[1]
    is_sheet = engine == "sheet-reader"
    provenance = (
        f"Rendered from {ext} with the built-in sheet reader"
        if is_sheet
        else f"Converted from {ext} with LibreOffice"
    )
    return {
        "path": path,
        "mtime": int(st.st_mtime),
        "size": st.st_size,
        "kind": kind.value,
        "engine": engine,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "language": None,
        "line_count": None,
        "provenance": provenance,
        "mode": "table" if is_sheet else "doc",
    }


def _collect_output(out_dir: str, stem: str, target: str) -> tuple[str, dict[str, bytes]]:
    """Gather what a conversion wrote out.

    Args:
        out_dir: The folder the conversion wrote to.
        stem: The file name without its ending.
        target: The output format.

    Returns:
        The main page and every other file next to it.
    """
    if target == "pdf":
        with open(os.path.join(out_dir, f"{stem}.pdf"), "rb") as fh:
            return "", {cache.OFFICE_PDF_ASSET: fh.read()}
    main_html = os.path.join(out_dir, f"{stem}.html")
    with open(main_html, "r", encoding="utf-8") as fh:
        html = fh.read()
    assets: dict[str, bytes] = {}
    for name in sorted(os.listdir(out_dir)):
        if name in (stem + ".html", "profile"):
            continue
        full = os.path.join(out_dir, name)
        if os.path.isfile(full):
            with open(full, "rb") as fh:
                assets[name] = fh.read()
    return html, assets


async def _soffice_convert(full_path: str, timeout: float, target: str = "html") -> tuple[str, dict[str, bytes]]:
    """Turn an Office file into a saved preview.

    The helper program is stopped if it takes too long or the job is cancelled.

    Args:
        full_path: Where the file is on disk.
        timeout: How long the helper may run before the conversion fails.
        target: The output format.

    Returns:
        The page and every other file made with it.

    Raises:
        _ConversionTimeout: If the helper program takes too long.
        _ConversionError: If the helper program makes no page.
    """
    out_dir = os.path.join(paths.scratch_dir(_SCRATCH_NAME), f"render_{os.urandom(4).hex()}")
    stem = os.path.splitext(os.path.basename(full_path))[0]
    main_output = os.path.join(out_dir, f"{stem}.{target}")
    await asyncio.to_thread(os.makedirs, out_dir, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        *soffice_cmd(out_dir, target, full_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        try:
            _out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise _ConversionTimeout("soffice timed out") from None
        except asyncio.CancelledError:
            proc.kill()
            try:
                await proc.wait()
            finally:
                raise
        stderr_tail = err.decode("utf-8", "replace").strip()[-200:]
        if proc.returncode != 0 or not await asyncio.to_thread(os.path.exists, main_output):
            raise _ConversionError(f"LibreOffice failed: {stderr_tail or 'no output file produced'}")
        return await asyncio.to_thread(_collect_output, out_dir, stem, target)
    finally:
        await asyncio.to_thread(shutil.rmtree, out_dir, ignore_errors=True)


async def _sheet_tables(path: str, full_path: str, kind: RenderKind) -> tuple[str, dict[str, bytes] | None, dict]:
    """Turn a spreadsheet into tables.

    Args:
        path: A path inside the shared folder.
        full_path: Where the file is on disk.
        kind: What sort of spreadsheet this is.

    Returns:
        The page, no extra files, and the facts to save.

    Raises:
        _ConversionError: If the spreadsheet cannot be read.
    """
    try:
        if kind is RenderKind.XLSX:
            sheets = await xlsx_sheets(full_path)
        else:
            sheets = await xls_sheets(full_path)
        fragment, counts = await _sheets_fragment(sheets)
    except Exception as exc:
        raise _ConversionError(f"Sheet read failed: {exc}") from exc
    meta = await asyncio.to_thread(_office_meta, path, full_path, kind, "sheet-reader")
    meta.update(counts)
    return fragment, None, meta


def _cell(value: _Cell) -> str:
    return "" if value is None else str(value)


def _sheets_fragment_sync(sheets: list[tuple[str, list[tuple]]]) -> tuple[str, dict]:
    sections: list[str] = []
    total_rows = 0
    first_cols: int | None = None
    for title, rows in sheets:
        rows = [row for row in rows if any(v is not None for v in row)]
        if not rows:
            continue
        if first_cols is None:
            first_cols = len(rows[0])
        header, data = rows[0], rows[1:]
        total_rows += len(data)
        table = table_html([_cell(c) for c in header], [[_cell(c) for c in row] for row in data])
        sections.append(f'<section class="sheet"><h3 class="sheet-title">{_html.escape(title)}</h3>{table}</section>')
    if not sections:
        raise _ConversionError("Sheet read failed: workbook has no rows")
    return "".join(sections), {"row_count": total_rows, "column_count": first_cols}


async def _sheets_fragment(sheets: list[tuple[str, list[tuple]]]) -> tuple[str, dict]:
    """Build the HTML for every sheet of a spreadsheet.

    The first row of a sheet is its header. Empty sheets are left out.

    Args:
        sheets: The name and rows of each sheet.

    Returns:
        The HTML and how many rows and columns it holds.

    Raises:
        _ConversionError: If no sheet has any rows.
    """
    return await asyncio.to_thread(_sheets_fragment_sync, sheets)


_ABS_URL = re.compile(r"^[a-z][a-z0-9+.-]*:")
# The only link types kept in a saved page. Every other one is dropped.
_ALLOWED_URL_SCHEMES = ("http", "https", "mailto", "ftp")
_URL_ATTRS = ("href", "src", "xlink:href")


def _url_scheme_allowed(value: str) -> bool:
    """Check that a link is safe to keep.

    Links with no type of their own are always fine. A link that names its
    type must name one of the allowed ones.

    Args:
        value: The link to check.

    Returns:
        True when the link is safe to keep.
    """
    cleaned = re.sub(r"^[\x00-\x20]+", "", value)
    if cleaned.startswith(("//", "/")):
        return True
    head, sep, _rest = cleaned.partition(":")
    return not sep or head.lower() in _ALLOWED_URL_SCHEMES


def _reduce_office_html_sync(html_text: str, key: str, assets: dict[str, bytes]) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup.find_all(["script", "object", "embed", "iframe"]):
        tag.decompose()
    head = soup.find("head")
    if head is not None:
        head.decompose()
    for tag in soup.find_all(["style", "link"]):
        tag.decompose()
    for tag in soup.find_all(True):
        if tag.has_attr("style"):
            del tag["style"]
        for attr in list(tag.attrs):
            value = tag.attrs[attr]
            if attr.lower().startswith("on") or (
                attr in _URL_ATTRS and isinstance(value, str) and not _url_scheme_allowed(value)
            ):
                del tag[attr]
    for tag in soup.find_all("img"):
        src = tag.get("src")
        if src and not _ABS_URL.match(src) and not src.startswith(("//", "/")):
            name = os.path.basename(src.split("?")[0])
            if name in assets:
                tag["src"] = f"/view/doc/{key}/{name}"
    body = soup.body
    content = body.decode_contents() if body is not None else soup.decode_contents()
    return f'<div class="md-doc">{content}</div>'


async def _reduce_office_html(html_text: str, key: str, assets: dict[str, bytes]) -> str:
    """Clean up converted HTML so it is safe to show.

    Anything that can run code or load its own styles is taken out. Links to
    the file's own images are pointed at the saved copies.

    Args:
        html_text: The HTML the conversion made.
        key: The name the file is saved under.
        assets: The other files made with the page.

    Returns:
        The cleaned up HTML.
    """
    return await asyncio.to_thread(_reduce_office_html_sync, html_text, key, assets)


class _Cell(Protocol):
    """A value read from a sheet cell."""
