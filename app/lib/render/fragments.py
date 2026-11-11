"""Turns a file into the HTML the viewer shows.

None of the HTML made here can run scripts.
"""

from __future__ import annotations

import asyncio
import csv as _csv
import html as _html
import io
import json
import os
import xml.dom.minidom as _minidom
from collections.abc import Callable
from datetime import datetime, timezone

import yaml as _yaml
from markdown_it import MarkdownIt
from pygments import highlight as _pyg_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import ClassNotFound, get_lexer_by_name

from app.lib.render.classify import RenderKind
from app.lib.render.markdown_engine import markdown_builder

__all__ = ["META_FIELDS", "build_fragment", "build_fragment_bytes", "build_fragment_sync", "table_html", "text_meta"]

_KB = 1024
_RICH_MAX_BYTES = _KB * _KB
_RAW_MAX_BYTES = 256 * _KB
_LARGER_CODE_NOTE = "Too large to highlight — showing first 256 KB."
_LARGER_OTHER_NOTE = "Too large to render — showing first 256 KB."
_NA_NOTE = "No preview available for this file."
_EMPTY_CSV_NOTE = "No rows found."
_PARSE_NOTE = "parse error, showing raw"
_CSV_WINDOW = 8
_DATA_KINDS = (RenderKind.JSON, RenderKind.JSONL, RenderKind.YAML, RenderKind.XML)

# The saved facts about a text or document file.
META_FIELDS = ("mode", "language", "line_count", "row_count", "column_count", "provenance", "note")
_ENGINES = {
    RenderKind.MARKDOWN: "markdown-it", RenderKind.CODE: "pygments", RenderKind.TEXT: "plain",
    RenderKind.HTML: "pygments", RenderKind.CSV: "csv", RenderKind.JSON: "json",
    RenderKind.JSONL: "jsonl", RenderKind.YAML: "yaml", RenderKind.XML: "xml",
}


def text_meta(kind: RenderKind, path: str, mtime: int, size: int, extras: dict) -> dict:
    """Collect the facts saved with a rendered text file.

    Args:
        kind: What sort of file this is.
        path: A path inside the shared folder.
        mtime: When the file last changed.
        size: How big the file is in bytes.
        extras: The extra facts that came with the HTML.

    Returns:
        The facts to save.
    """
    return {
        "path": path, "mtime": mtime, "size": size, "kind": kind.value,
        "engine": _ENGINES.get(kind, "pygments"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **{f: extras.get(f) for f in META_FIELDS},
    }


_MD: MarkdownIt | None = None
_STYLESHEET: str | None = None


def _md() -> MarkdownIt:
    global _MD
    if _MD is None:
        # The markdown maker must come from markdown_engine. Making one here
        # breaks markdown rendering.
        _MD = markdown_builder()
    return _MD


def _scope(css: str, scope: str) -> str:
    """Make a style sheet apply only inside one theme.

    Args:
        css: The style sheet text.
        scope: The theme the styles belong to.

    Returns:
        The new style sheet text.
    """
    out = []
    for rule in css.rstrip("}").split("}"):
        selector, _, decls = rule.partition("{")
        parts = [f"{scope} {sel.strip()}" for sel in selector.split(",") if sel.strip()]
        if parts:
            out.append(",".join(parts) + "{" + decls + "}")
    return "".join(out)


def _pygments_stylesheet() -> str:
    """Get the colors used for highlighted code.

    Returns:
        Style sheet text with colors for the light and the dark theme.
    """
    global _STYLESHEET
    if _STYLESHEET is None:
        light = HtmlFormatter(style="default").get_style_defs(".code-block")
        dark = HtmlFormatter(style="github-dark").get_style_defs(".code-block")
        _STYLESHEET = _scope(light, '[data-theme="light"]') + _scope(dark, '[data-theme="dark"]')
    return _STYLESHEET


def _language_for(full_path: str) -> str | None:
    ext = os.path.splitext(full_path)[1].lstrip(".").lower()
    if not ext:
        return None
    try:
        return get_lexer_by_name(ext).aliases[0]
    except ClassNotFound:
        return None


def _raw_pre_from_text(text: str) -> str:
    return '<pre class="raw-pre">' + _html.escape(text[:_RAW_MAX_BYTES]) + "</pre>"


def _raw_pre(full_path: str) -> str:
    with open(full_path, "rb") as fh:
        data = fh.read(_RAW_MAX_BYTES)
    return _raw_pre_from_text(data.decode("utf-8", errors="replace"))


def _code_fragment(text: str, language: str | None) -> tuple[str, int]:
    """Turn source text into colored HTML, one line at a time.

    Args:
        text: The source text.
        language: The code language, or None for plain text.

    Returns:
        The HTML and how many lines it has.
    """
    lexer = get_lexer_by_name(language) if language else get_lexer_by_name("text")
    rendered = _pyg_highlight(text, lexer, HtmlFormatter(nowrap=True))
    lines = rendered.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    body = "".join(f'<div class="cl">{line}</div>' for line in lines)
    return f"<style>{_pygments_stylesheet()}</style><pre class=\"code-block\">{body}</pre>", len(lines)


def _csv_rows(text: str) -> list[list[str]]:
    return [row for row in _csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]


def table_html(header: list[str], data: list[list[str]]) -> str:
    """Build the HTML table for rows of a spreadsheet.

    Rows past the first few are marked so the viewer can hide them.

    Args:
        header: The names of the columns.
        data: The rows under the header.

    Returns:
        The HTML for the table.
    """
    head = "<thead><tr>" + "".join(f"<th>{_html.escape(c)}</th>" for c in header) + "</tr></thead>"
    rows = []
    for i, row in enumerate(data, start=1):
        cls = f' class="csv-extra"' if i > _CSV_WINDOW else ""
        rows.append(f"<tr{cls}>" + "".join(f"<td>{_html.escape(c)}</td>" for c in row) + "</tr>")
    return f'<table class="csv-table">{head}<tbody>{"".join(rows)}</tbody></table>'


def _csv_fragment(text: str) -> tuple[str, dict] | None:
    rows = _csv_rows(text)
    if not rows:
        return None
    header, data = rows[0], rows[1:]
    return table_html(header, data), {"row_count": len(data), "column_count": len(header)}


def _data_fragment(kind: RenderKind, text: str) -> tuple[str, dict]:
    minified = text.strip().count("\n") <= 1
    try:
        if kind is RenderKind.JSON:
            rendered, language = json.dumps(json.loads(text), indent=2, ensure_ascii=False), "json"
        elif kind is RenderKind.JSONL:
            docs = [json.loads(line) for line in text.splitlines() if line.strip()]
            if not docs:
                raise ValueError("empty")
            rendered = "\n\n".join(json.dumps(d, indent=2, ensure_ascii=False) for d in docs)
            language = "json"
        elif kind is RenderKind.YAML:
            docs = list(_yaml.safe_load_all(text))
            if not docs:
                raise ValueError("empty")
            if len(docs) == 1:
                rendered = _yaml.safe_dump(docs[0], default_flow_style=False, allow_unicode=True)
            else:
                rendered = "---\n".join(
                    _yaml.safe_dump(d, default_flow_style=False, allow_unicode=True) for d in docs
                )
            language = "yaml"
        else:
            rendered = _minidom.parseString(text).toprettyxml(indent="  ")
            language = "xml"
    except Exception:
        return _raw_pre_from_text(text), {"mode": "raw", "note": _PARSE_NOTE}
    fragment, count = _code_fragment(rendered, language)
    provenance = "pretty-printed · source was minified" if minified else None
    return fragment, {"mode": "code", "line_count": count, "provenance": provenance}


def _oversize_note(kind: RenderKind) -> str:
    """The note a file over the rich limit gets, by kind.

    Args:
        kind: What sort of content this is.

    Returns:
        The note the viewer shows instead of the rich render.
    """
    return _LARGER_CODE_NOTE if kind in (RenderKind.CODE, RenderKind.TEXT) else _LARGER_OTHER_NOTE


def _render_kind_text(
    kind: RenderKind, text: str, language: str | None, raw: Callable[[], str] | None, label: str
) -> tuple[str | None, dict]:
    """Render decoded text with the builder for its kind.

    Args:
        kind: What sort of content this is.
        text: The decoded text.
        language: The code language, or None for auto or non-code kinds.
        raw: How to build the raw preview when a table has no rows, or None
            when the content is not a file.
        label: The name of the builder the error should blame.

    Returns:
        The HTML, or None, plus the extra facts about the content.

    Raises:
        ValueError: If this kind of content cannot be built.
    """
    if kind is RenderKind.MARKDOWN:
        html = _md().render(text)
        html = html.replace("<table>", '<div class="md-table-scroll"><table>')
        html = html.replace("</table>", "</table></div>")
        return f'<div class="md-doc">{html}</div>', {"mode": "doc"}
    if kind in (RenderKind.CODE, RenderKind.HTML):
        frag, count = _code_fragment(text, language)
        return frag, {"mode": "code", "language": language, "line_count": count}
    if kind is RenderKind.TEXT:
        return f'<pre class="plain-pre">{_html.escape(text)}</pre>', {"mode": "plain", "language": None}
    if kind is RenderKind.CSV:
        built = _csv_fragment(text)
        if built is None:
            frag = raw() if raw is not None else None
            return frag, {"mode": "raw", "note": _EMPTY_CSV_NOTE}
        fragment, counts = built
        return fragment, {"mode": "table", **counts}
    if kind in _DATA_KINDS:
        return _data_fragment(kind, text)
    raise ValueError(f"{label}: no fragment builder for kind {kind}")


def build_fragment_sync(kind: RenderKind, full_path: str, size: int) -> tuple[str | None, dict]:
    """Build the HTML the viewer shows for one file, without an event loop.

    This is for code that already runs in its own thread. Never call it on
    the event loop, use build_fragment there.

    A file that does not read as text gets no HTML, just a note.

    Args:
        kind: What sort of file this is.
        full_path: Where the file is on disk.
        size: How big the file is in bytes.

    Returns:
        The HTML, or None, plus the extra facts about the file.

    Raises:
        ValueError: If this kind of file cannot be built.
    """
    if size > _RICH_MAX_BYTES:
        return _raw_pre(full_path), {"mode": "raw", "note": _oversize_note(kind)}
    try:
        with open(full_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if "\x00" in text:
            raise ValueError("nul byte")
    except (OSError, UnicodeDecodeError, ValueError):
        # The file does not read as text, so there is nothing to show.
        return None, {"mode": "raw", "note": _NA_NOTE}
    return _render_kind_text(kind, text, _language_for(full_path), lambda: _raw_pre(full_path), "build_fragment")


def _build_fragment_bytes_sync(kind: RenderKind, data: bytes, language: str | None = None) -> tuple[str | None, dict]:
    """Build the HTML the viewer shows for in-memory bytes, without an event loop.

    Args:
        kind: What sort of content the bytes hold.
        data: The bytes to render.
        language: The code language, or None for auto or non-code kinds.

    Returns:
        The HTML, or None, plus the extra facts about the content.

    Raises:
        ValueError: If this kind of content cannot be built.
    """
    if len(data) > _RICH_MAX_BYTES:
        return None, {"mode": "raw", "note": _oversize_note(kind)}
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None, {"mode": "raw", "note": _NA_NOTE}
    if "\x00" in text:
        return None, {"mode": "raw", "note": _NA_NOTE}
    return _render_kind_text(kind, text, language, None, "build_fragment_bytes")


async def build_fragment_bytes(kind: RenderKind, data: bytes, language: str | None = None) -> tuple[str | None, dict]:
    """Build the HTML the viewer shows for in-memory bytes.

    Args:
        kind: What sort of content the bytes hold.
        data: The bytes to render.
        language: The code language, or None for auto or non-code kinds.

    Returns:
        The HTML, or None, plus the extra facts about the content.
    """
    return await asyncio.to_thread(_build_fragment_bytes_sync, kind, data, language)


async def build_fragment(kind: RenderKind, full_path: str, size: int) -> tuple[str | None, dict]:
    """Build the HTML the viewer shows for one file.

    A file that does not read as text gets no HTML, just a note.

    Args:
        kind: What sort of file this is.
        full_path: Where the file is on disk.
        size: How big the file is in bytes.

    Returns:
        The HTML, or None, plus the extra facts about the file.

    Raises:
        ValueError: If this kind of file cannot be built.
    """
    return await asyncio.to_thread(build_fragment_sync, kind, full_path, size)
