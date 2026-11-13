"""Read Microsoft Office files as text.

Word files, workbooks, and slide decks are all read here. Old Word files and
slide decks need LibreOffice and pdftoppm on the machine.
"""

__all__ = [
    "docx_to_text",
    "xlsx_to_text",
    "xls_to_text",
    "doc_to_text",
    "describe_presentation",
    "soffice_cmd",
    "xlsx_sheets",
    "xls_sheets",
]

import asyncio
import glob
import os
import shutil
import sys
import uuid
from typing import Protocol

import aiofiles
import xlrd
from docx import Document
from openpyxl import load_workbook

from app.lib.cli.cli_runner import run_cli
from app.lib.reason.describe_image import describe_image
from app.lib.shared import paths

_SOFFICE_TIMEOUT = 120
_SLIDE_RENDER_DPI = 150
_SCRATCH_NAME = "office_docs"


async def _make_tmp_dir(kind: str) -> str:
    """Make a new empty folder for scratch files.

    The caller deletes the folder when it is done with it.

    Args:
        kind: A word put in the folder name.

    Returns:
        The path of the new folder.
    """
    temp_dir = os.path.join(paths.scratch_dir(_SCRATCH_NAME), f"{kind}_{str(uuid.uuid4())[:8]}")
    await asyncio.to_thread(os.makedirs, temp_dir, exist_ok=True)
    return temp_dir


def soffice_cmd(out_dir: str, target: str, path: str) -> list[str]:
    """Build the command that turns a file into another format.

    The new file lands in out_dir, with the same name and a new ending.

    Args:
        out_dir: The folder the new file goes in.
        target: The format to make.
        path: The file to read.

    Returns:
        The command to run.
    """
    return [
        "soffice",
        "--headless",
        "--norestore",
        "--nologo",
        f"-env:UserInstallation=file://{out_dir}/profile",
        "--convert-to",
        target,
        "--outdir",
        out_dir,
        path,
    ]


def _docx_text(path: str) -> str:
    """Read the words and the tables out of a Word file."""
    doc = Document(path)
    paragraphs = [paragraph.text for paragraph in doc.paragraphs]
    table_blocks = [
        "\n".join("\t".join(cell.text for cell in row.cells) for row in table.rows)
        for table in doc.tables
    ]
    parts = ["\n".join(paragraphs), *table_blocks]
    return "\n\n".join(part for part in parts if part)


class _CellValue(Protocol):
    """A value held in one sheet cell."""


def _read_xlsx(path: str) -> list[tuple[str, list[tuple[_CellValue, ...]]]]:
    """Read the name and the rows of every sheet in a workbook."""
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        return [
            (ws.title, [tuple(row) for row in ws.iter_rows(values_only=True)])
            for ws in wb.worksheets
        ]
    finally:
        wb.close()


async def xlsx_sheets(path: str) -> list[tuple[str, list[tuple[_CellValue, ...]]]]:
    """Read the sheets out of a workbook.

    Args:
        path: The workbook to read.

    Returns:
        The name and the rows of every sheet.
    """
    return await asyncio.to_thread(_read_xlsx, path)


def _read_xls(path: str) -> list[tuple[str, list[tuple[_CellValue, ...]]]]:
    """Read the name and the rows of every sheet in an old workbook."""
    book = xlrd.open_workbook(path)
    return [
        (sheet.name, [tuple(sheet.row_values(r)) for r in range(sheet.nrows)])
        for sheet in book.sheets()
    ]


async def xls_sheets(path: str) -> list[tuple[str, list[tuple[_CellValue, ...]]]]:
    """Read the sheets out of an old workbook.

    Args:
        path: The workbook to read.

    Returns:
        The name and the rows of every sheet.
    """
    return await asyncio.to_thread(_read_xls, path)


def _sheets_to_text(sheets: list[tuple[str, list[tuple[_CellValue, ...]]]]) -> str:
    """Lay the sheets out as text, one block per sheet."""
    blocks: list[str] = []
    for title, rows in sheets:
        lines = [f"Sheet: {title}"]
        for row in rows:
            if all(value is None for value in row):
                continue
            lines.append("\t".join("" if value is None else str(value) for value in row))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def _read_stripped(path: str) -> str:
    """Read a text file and trim the blank space around it."""
    async with aiofiles.open(path, "r", encoding="utf-8", errors="replace") as f:
        return (await f.read()).strip()


async def docx_to_text(path: str, verbose: bool = False) -> str:
    """Read a Word file as text.

    The words come first, then the tables.

    Args:
        path: The file to read.
        verbose: True to print what is happening.

    Returns:
        The text of the file.

    Raises:
        PackageNotFoundError: If the file is missing or is not a Word file.
    """
    text = await asyncio.to_thread(_docx_text, path)
    if verbose:
        print(f"[office_docs.docx_to_text] extracted {len(text)} chars from {path}", flush=True)
    return text


async def xlsx_to_text(path: str, verbose: bool = False) -> str:
    """Read a workbook as text.

    Every sheet gets its own block, with one line per row that has something
    in it. Blank rows are left out.

    Args:
        path: The file to read.
        verbose: True to print what is happening.

    Returns:
        The text of the workbook.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    sheets = await xlsx_sheets(path)
    text = await asyncio.to_thread(_sheets_to_text, sheets)
    if verbose:
        print(f"[office_docs.xlsx_to_text] extracted {len(text)} chars from {path}", flush=True)
    return text


async def xls_to_text(path: str, verbose: bool = False) -> str:
    """Read an old workbook as text.

    Every sheet gets its own block, with one line per row that has something
    in it. Blank rows are left out.

    Args:
        path: The file to read.
        verbose: True to print what is happening.

    Returns:
        The text of the workbook.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    sheets = await xls_sheets(path)
    text = await asyncio.to_thread(_sheets_to_text, sheets)
    if verbose:
        print(f"[office_docs.xls_to_text] extracted {len(text)} chars from {path}", flush=True)
    return text


async def doc_to_text(path: str, verbose: bool = False) -> str:
    """Read an old Word file as text.

    Args:
        path: The file to read.
        verbose: True to print what is happening.

    Returns:
        The text of the file.

    Raises:
        RuntimeError: If the file could not be turned into text.
    """
    out_dir = await _make_tmp_dir("doc_text")
    try:
        stem = os.path.splitext(os.path.basename(path))[0]
        result = await run_cli(soffice_cmd(out_dir, "txt", path), "soffice doc conversion", timeout=_SOFFICE_TIMEOUT)
        output = os.path.join(out_dir, f"{stem}.txt")
        if result.returncode != 0 or not await asyncio.to_thread(os.path.exists, output):
            detail = result.stderr or "no output file produced"
            print(f"[office_docs.doc_to_text] {detail}", file=sys.stderr)
            raise RuntimeError(f"soffice conversion failed: {detail}")
        text = await _read_stripped(output)
        if verbose:
            print(f"[office_docs.doc_to_text] extracted {len(text)} chars from {path}", flush=True)
        return text
    finally:
        await asyncio.to_thread(shutil.rmtree, out_dir, ignore_errors=True)


async def _convert_presentation_to_pdf(path: str, out_dir: str) -> str | None:
    """Turn a slide deck into a PDF.

    Returns:
        The path of the PDF, or None if it could not be made.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    result = await run_cli(
        soffice_cmd(out_dir, "pdf", path), "soffice presentation conversion", timeout=_SOFFICE_TIMEOUT
    )
    if result.returncode != 0:
        print(
            f"[office_docs.describe_presentation] {result.stderr or 'soffice failed'}",
            file=sys.stderr,
        )
        return None
    return os.path.join(out_dir, f"{stem}.pdf")


async def _render_slides(pdf_path: str, frame_dir: str) -> list[str]:
    """Turn the pages of a PDF into pictures.

    Returns:
        The picture paths in page order, or nothing if no page was drawn.
    """
    result = await run_cli(
        [
            "pdftoppm",
            "-jpeg",
            "-r",
            str(_SLIDE_RENDER_DPI),
            pdf_path,
            os.path.join(frame_dir, "slide"),
        ],
        "pdftoppm slide rendering",
    )
    if result.returncode != 0:
        print(
            f"[office_docs.describe_presentation] {result.stderr or 'pdftoppm failed'}",
            file=sys.stderr,
        )
    return sorted(await asyncio.to_thread(glob.glob, os.path.join(frame_dir, "slide-*.jpg")))


async def _describe_slides(path: str, context: str, verbose: bool) -> tuple[str, int]:
    """Say what is on each slide of a deck.

    Returns:
        The text about the slides, and a status code.
    """
    out_dir = await _make_tmp_dir("presentation")
    frame_dir = await _make_tmp_dir("presentation_frames")
    try:
        pdf_path = await _convert_presentation_to_pdf(path, out_dir)
        if pdf_path is None:
            return "Could not convert presentation to PDF.", 1
        images = await _render_slides(pdf_path, frame_dir)
        if not images:
            return "Could not render slides to images.", 1
        if verbose:
            print(f"[office_docs.describe_presentation] rendered {len(images)} slides from {path}", flush=True)
        return await describe_image(images, context, verbose=verbose)
    finally:
        await asyncio.to_thread(shutil.rmtree, out_dir, ignore_errors=True)
        await asyncio.to_thread(shutil.rmtree, frame_dir, ignore_errors=True)


async def describe_presentation(
    path: str, context: str = "", verbose: bool = False
) -> tuple[str, int]:
    """Say what is on each slide of a deck.

    Args:
        path: The slide deck to read.
        context: A question about the slides. Empty asks for a description.
        verbose: True to print what is happening.

    Returns:
        The text about the slides or the answer, and a status code. Zero
        means it worked, 1 means it went wrong, 2 means the file is missing,
        and 3 means LibreOffice is not installed.
    """
    if not await asyncio.to_thread(os.path.exists, path):
        return f"Presentation path '{path}' does not exist.", 2
    if await asyncio.to_thread(shutil.which, "soffice") is None:
        return "soffice not found. Please install LibreOffice.", 3
    try:
        return await _describe_slides(path, context, verbose)
    except Exception as e:
        return f"Error processing presentation '{path}': {e}", 1
