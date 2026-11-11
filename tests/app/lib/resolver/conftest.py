"""Builders for real office files used by resolver tests."""

from pathlib import Path

from docx import Document
from openpyxl import Workbook
import xlwt


def make_docx(path: str | Path, paragraphs: list[str], table_rows: list[list[str]]) -> Path:
    """Create a .docx with the given paragraphs followed by one table.

    Args:
        path: Destination file path.
        paragraphs: Paragraph texts, added in order.
        table_rows: Rows of cell texts for a single table.

    Returns:
        The path to the new file.
    """
    doc = Document()
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    table = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
    for row_idx, row in enumerate(table_rows):
        for col_idx, value in enumerate(row):
            table.cell(row_idx, col_idx).text = value
    doc.save(str(path))
    return Path(path)


def make_xlsx(path: str | Path, sheets: list[tuple[str, list[list[str | int]]]]) -> Path:
    """Create an .xlsx with one sheet for each title and its rows.

    Args:
        path: Destination file path.
        sheets: Sheet titles with their rows, in order.

    Returns:
        The path to the new file.
    """
    wb = Workbook()
    for idx, (title, rows) in enumerate(sheets):
        ws = wb.active if idx == 0 else wb.create_sheet(title)
        if idx == 0:
            ws.title = title
        for row in rows:
            ws.append(row)
    wb.save(str(path))
    return Path(path)


def make_xls(path: str | Path, title: str, rows: list[list[str | int]]) -> Path:
    """Create a legacy .xls with a single sheet of the given rows.

    Args:
        path: Destination file path.
        title: Sheet name.
        rows: Cell values, row by row.

    Returns:
        The path to the new file.
    """
    book = xlwt.Workbook()
    sheet = book.add_sheet(title)
    for row_idx, row in enumerate(rows):
        for col_idx, value in enumerate(row):
            sheet.write(row_idx, col_idx, value)
    book.save(str(path))
    return Path(path)
