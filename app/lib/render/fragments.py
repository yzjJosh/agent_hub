"""Turns a file into the HTML the viewer shows.

None of the HTML made here can run scripts.
"""

from __future__ import annotations

import asyncio
import csv
import html
import io
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from xml.dom import minidom

import yaml
from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import ClassNotFound, get_lexer_by_name

from app.lib.render.classify import RenderKind
from app.lib.render.markdown_engine import markdown_builder

__all__ = [
    "BuiltFragment",
    "META_FIELDS",
    "build_fragment",
    "build_fragment_bytes",
    "build_fragment_sync",
    "build_table_html",
    "build_text_metadata",
]

_KIBIBYTE = 1024
_RICH_MAX_BYTES = _KIBIBYTE * _KIBIBYTE
_RAW_MAX_BYTES = 256 * _KIBIBYTE
_LARGER_CODE_NOTE = "Too large to highlight — showing first 256 KB."
_LARGER_OTHER_NOTE = "Too large to render — showing first 256 KB."
_NO_PREVIEW_NOTE = "No preview available for this file."
_EMPTY_CSV_NOTE = "No rows found."
_PARSE_NOTE = "parse error, showing raw"
_CSV_WINDOW = 8
_DATA_KINDS = (RenderKind.JSON, RenderKind.JSONL, RenderKind.YAML, RenderKind.XML)

# The saved facts about a text or document file.
META_FIELDS = ("mode", "language", "line_count", "row_count", "column_count", "provenance", "note")
_ENGINES = {
    RenderKind.MARKDOWN: "markdown-it",
    RenderKind.CODE: "pygments",
    RenderKind.TEXT: "plain",
    RenderKind.HTML: "pygments",
    RenderKind.CSV: "csv",
    RenderKind.JSON: "json",
    RenderKind.JSONL: "jsonl",
    RenderKind.YAML: "yaml",
    RenderKind.XML: "xml",
}


@dataclass(frozen=True)
class BuiltFragment:
    """Rendered HTML fragment and metadata for one file or byte buffer."""

    html: str | None
    metadata: dict


@dataclass(frozen=True)
class _HighlightedCode:
    """Syntax-highlighted HTML and its line count."""

    html: str
    line_count: int


def build_text_metadata(
    kind: RenderKind, path: str, modified_time: int, size: int, extras: dict
) -> dict:
    """Collect the facts saved with a rendered text file.

    Args:
        kind: What sort of file this is.
        path: A path inside the resources folder.
        modified_time: When the file last changed.
        size: How big the file is in bytes.
        extras: The extra facts that came with the HTML.

    Returns:
        The facts to save.
    """
    return {
        "path": path,
        "mtime": modified_time,
        "size": size,
        "kind": kind.value,
        "engine": _ENGINES.get(kind, "pygments"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **{field: extras.get(field) for field in META_FIELDS},
    }


_markdown_parser: MarkdownIt | None = None
_pygments_css: str | None = None


def _get_markdown_parser() -> MarkdownIt:
    global _markdown_parser
    if _markdown_parser is None:
        _markdown_parser = markdown_builder()
    return _markdown_parser


def _scope_css(css: str, scope: str) -> str:
    scoped_rules: list[str] = []
    for rule in css.rstrip("}").split("}"):
        selector, _, declarations = rule.partition("{")
        parts = [f"{scope} {selector_part.strip()}" for selector_part in selector.split(",") if selector_part.strip()]
        if parts:
            scoped_rules.append(",".join(parts) + "{" + declarations + "}")
    return "".join(scoped_rules)


def _get_pygments_stylesheet() -> str:
    global _pygments_css
    if _pygments_css is None:
        light = HtmlFormatter(style="default").get_style_defs(".code-block")
        dark = HtmlFormatter(style="github-dark").get_style_defs(".code-block")
        _pygments_css = _scope_css(light, '[data-theme="light"]') + _scope_css(dark, '[data-theme="dark"]')
    return _pygments_css


def _detect_language(full_path: str) -> str | None:
    extension = os.path.splitext(full_path)[1].lstrip(".").lower()
    if not extension:
        return None
    try:
        return get_lexer_by_name(extension).aliases[0]
    except ClassNotFound:
        return None


def _wrap_raw_pre(text: str) -> str:
    return '<pre class="raw-pre">' + html.escape(text[:_RAW_MAX_BYTES]) + "</pre>"


def _read_raw_pre(full_path: str) -> str:
    with open(full_path, "rb") as file_handle:
        raw_bytes = file_handle.read(_RAW_MAX_BYTES)
    return _wrap_raw_pre(raw_bytes.decode("utf-8", errors="replace"))


def _highlight_code(text: str, language: str | None) -> _HighlightedCode:
    lexer = get_lexer_by_name(language) if language else get_lexer_by_name("text")
    rendered = highlight(text, lexer, HtmlFormatter(nowrap=True))
    lines = rendered.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    body = "".join(f'<div class="cl">{line}</div>' for line in lines)
    fragment_html = f'<style>{_get_pygments_stylesheet()}</style><pre class="code-block">{body}</pre>'
    return _HighlightedCode(html=fragment_html, line_count=len(lines))


def _parse_csv_rows(text: str) -> list[list[str]]:
    return [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]


def build_table_html(header: list[str], data: list[list[str]]) -> str:
    """Build the HTML table for rows of a spreadsheet.

    Rows past the first few are marked so the viewer can hide them.

    Args:
        header: The names of the columns.
        data: The rows under the header.

    Returns:
        The HTML for the table.
    """
    head = "<thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in header) + "</tr></thead>"
    rows: list[str] = []
    for row_index, row in enumerate(data, start=1):
        row_class = ' class="csv-extra"' if row_index > _CSV_WINDOW else ""
        rows.append(f"<tr{row_class}>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>")
    return f'<table class="csv-table">{head}<tbody>{"".join(rows)}</tbody></table>'


def _build_csv_fragment(text: str) -> BuiltFragment | None:
    rows = _parse_csv_rows(text)
    if not rows:
        return None
    header, data = rows[0], rows[1:]
    return BuiltFragment(
        html=build_table_html(header, data),
        metadata={"mode": "table", "row_count": len(data), "column_count": len(header)},
    )


def _build_data_fragment(kind: RenderKind, text: str) -> BuiltFragment:
    is_minified = text.strip().count("\n") <= 1
    try:
        if kind is RenderKind.JSON:
            rendered = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
            language = "json"
        elif kind is RenderKind.JSONL:
            documents = [json.loads(line) for line in text.splitlines() if line.strip()]
            if not documents:
                raise ValueError("empty")
            rendered = "\n\n".join(json.dumps(document, indent=2, ensure_ascii=False) for document in documents)
            language = "json"
        elif kind is RenderKind.YAML:
            documents = list(yaml.safe_load_all(text))
            if not documents:
                raise ValueError("empty")
            if len(documents) == 1:
                rendered = yaml.safe_dump(documents[0], default_flow_style=False, allow_unicode=True)
            else:
                rendered = "---\n".join(
                    yaml.safe_dump(document, default_flow_style=False, allow_unicode=True)
                    for document in documents
                )
            language = "yaml"
        else:
            rendered = minidom.parseString(text).toprettyxml(indent="  ")
            language = "xml"
    except Exception:
        return BuiltFragment(html=_wrap_raw_pre(text), metadata={"mode": "raw", "note": _PARSE_NOTE})
    highlighted = _highlight_code(rendered, language)
    provenance = "pretty-printed · source was minified" if is_minified else None
    return BuiltFragment(
        html=highlighted.html,
        metadata={"mode": "code", "line_count": highlighted.line_count, "provenance": provenance},
    )


def _get_oversize_note(kind: RenderKind) -> str:
    return _LARGER_CODE_NOTE if kind in (RenderKind.CODE, RenderKind.TEXT) else _LARGER_OTHER_NOTE


def _render_kind_text(
    kind: RenderKind,
    text: str,
    language: str | None,
    raw_preview_builder: Callable[[], str] | None,
    caller_name: str,
) -> BuiltFragment:
    if kind is RenderKind.MARKDOWN:
        rendered_markdown = _get_markdown_parser().render(text)
        rendered_markdown = rendered_markdown.replace("<table>", '<div class="md-table-scroll"><table>')
        rendered_markdown = rendered_markdown.replace("</table>", "</table></div>")
        return BuiltFragment(html=f'<div class="md-doc">{rendered_markdown}</div>', metadata={"mode": "doc"})
    if kind in (RenderKind.CODE, RenderKind.HTML):
        highlighted = _highlight_code(text, language)
        return BuiltFragment(
            html=highlighted.html,
            metadata={"mode": "code", "language": language, "line_count": highlighted.line_count},
        )
    if kind is RenderKind.TEXT:
        return BuiltFragment(
            html=f'<pre class="plain-pre">{html.escape(text)}</pre>',
            metadata={"mode": "plain", "language": None},
        )
    if kind is RenderKind.CSV:
        table_fragment = _build_csv_fragment(text)
        if table_fragment is None:
            fallback_html = raw_preview_builder() if raw_preview_builder is not None else None
            return BuiltFragment(html=fallback_html, metadata={"mode": "raw", "note": _EMPTY_CSV_NOTE})
        return table_fragment
    if kind in _DATA_KINDS:
        return _build_data_fragment(kind, text)
    raise ValueError(f"{caller_name}: no fragment builder for kind {kind}")


def build_fragment_sync(kind: RenderKind, full_path: str, size: int) -> BuiltFragment:
    """Build the HTML the viewer shows for one file without an event loop.

    A file that does not read as text gets no HTML, just a note.

    Args:
        kind: What sort of file this is.
        full_path: Where the file is on disk.
        size: How big the file is in bytes.

    Returns:
        The built HTML and metadata for the file.

    Raises:
        ValueError: If this kind of file cannot be built.
    """
    if size > _RICH_MAX_BYTES:
        return BuiltFragment(html=_read_raw_pre(full_path), metadata={"mode": "raw", "note": _get_oversize_note(kind)})
    try:
        with open(full_path, "r", encoding="utf-8") as file_handle:
            text = file_handle.read()
        if "\x00" in text:
            raise ValueError("nul byte")
    except (OSError, UnicodeDecodeError, ValueError):
        return BuiltFragment(html=None, metadata={"mode": "raw", "note": _NO_PREVIEW_NOTE})
    return _render_kind_text(
        kind, text, _detect_language(full_path), lambda: _read_raw_pre(full_path), "build_fragment"
    )


def _build_fragment_bytes_sync(
    kind: RenderKind, data: bytes, language: str | None = None
) -> BuiltFragment:
    if len(data) > _RICH_MAX_BYTES:
        return BuiltFragment(html=None, metadata={"mode": "raw", "note": _get_oversize_note(kind)})
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return BuiltFragment(html=None, metadata={"mode": "raw", "note": _NO_PREVIEW_NOTE})
    if "\x00" in text:
        return BuiltFragment(html=None, metadata={"mode": "raw", "note": _NO_PREVIEW_NOTE})
    return _render_kind_text(kind, text, language, None, "build_fragment_bytes")


async def build_fragment_bytes(
    kind: RenderKind, data: bytes, language: str | None = None
) -> BuiltFragment:
    """Build the HTML the viewer shows for in-memory bytes.

    Args:
        kind: What sort of content the bytes hold.
        data: The bytes to render.
        language: The code language, or None when not set.

    Returns:
        The built HTML and metadata for the content.

    Raises:
        ValueError: When kind is not one of the text or data kinds.
    """
    return await asyncio.to_thread(_build_fragment_bytes_sync, kind, data, language)


async def build_fragment(kind: RenderKind, full_path: str, size: int) -> BuiltFragment:
    """Build the HTML the viewer shows for one file.

    A file that does not read as text gets no HTML, just a note.

    Args:
        kind: What sort of file this is.
        full_path: Where the file is on disk.
        size: How big the file is in bytes.

    Returns:
        The built HTML and metadata for the file.

    Raises:
        ValueError: If this kind of file cannot be built.
    """
    return await asyncio.to_thread(build_fragment_sync, kind, full_path, size)
