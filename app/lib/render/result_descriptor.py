"""Describes what a task run gave back so the viewer can show it."""

from __future__ import annotations

import asyncio
import html as _html
import json
import os
import re
import shutil
import xml.dom.minidom as _minidom
from typing import Protocol

import yaml as _yaml

import app.lib.text_type.detector as detector
from app.lib.render import cache, fragments
from app.lib.render.classify import RenderKind
from app.lib.render.descriptor import OFFICE_KINDS
from app.lib.render.detect import BINARY_KINDS, TEXT_FAMILY_KINDS, Detection, detect, language_extension

__all__ = ["build_result_descriptor"]

_KiB = 1024
# The most a body or its page can hold before only raw text is shown.
_RICH_MAX = _KiB * _KiB
_RAW_MAX = 256 * _KiB
_PARSE_VALIDATED = frozenset({RenderKind.JSON, RenderKind.JSONL, RenderKind.YAML, RenderKind.XML})
_EXT_LESS_TEXT = frozenset({RenderKind.MARKDOWN, RenderKind.TEXT, RenderKind.CSV,
                            RenderKind.JSON, RenderKind.JSONL, RenderKind.YAML, RenderKind.XML})
_TEXT_FAMILY = TEXT_FAMILY_KINDS

_TOO_LARGE_NOTE = "Too large to render — showing first 256 KB. Switch to Raw or Download for the full payload."
_EMPTY_CSV_NOTE = "No rows found."


def _as_text(body: bytes) -> str | None:
    # A NUL byte anywhere, or undecodable bytes, means the body is not text.
    if b"\x00" in body:
        return None
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return None


async def _reclassify(body: bytes) -> Detection:
    # A decodable magic prefix fired the binary sniff, so the text classifier decides.
    label = await detector.identify(body)
    return Detection(detector.kind_for_label(label), detector.language_for_label(label))


def _parse_note(kind: RenderKind) -> str:
    return (f"Parse error — the body is not valid {kind.value.upper()}. "
            f"Showing raw text. Use Raw or Download for the full payload.")


def _kind_label(kind: RenderKind, language: str | None, subtype: str | None) -> str:
    if kind is RenderKind.CODE:
        return f"code · {language}" if language else "code"
    if kind in OFFICE_KINDS:
        return f"office · {subtype}"
    if kind is RenderKind.UNKNOWN:
        return "file"
    return kind.value


def _raw_pre(text: str) -> str:
    return '<pre class="raw-pre">' + _html.escape(text[:_RAW_MAX]) + "</pre>"


def _body_source(task_id: str, run_dir: str) -> tuple[str, str]:
    """Get a run's body file and its path inside the data folder.

    Args:
        task_id: The run the body came from.
        run_dir: Where the run keeps its files.

    Returns:
        Where the body is on disk, and its path inside the data folder.
    """
    return os.path.join(run_dir, "result"), f"runs/{task_id}/result"


def _input_name(kind: RenderKind, language: str | None, subtype: str | None) -> str | None:
    if kind in BINARY_KINDS:
        return f"input.{subtype}"
    if kind is RenderKind.CODE:
        # Code with no known language still gets a name. Without one the page
        # cannot be built.
        ext = language_extension(language)
        return f"input.{ext}" if ext else "input"
    if kind is RenderKind.HTML:
        return "input.html"
    if kind in _EXT_LESS_TEXT:
        return "input"
    return None


def _materialize(run_key: str, body_path: str, eff: RenderKind, det: Detection) -> str | None:
    """Save a copy of the body next to its saved page.

    Args:
        run_key: The name the run is saved under.
        body_path: Where the body is on disk.
        eff: What sort of body this is.
        det: What the body was found to be.

    Returns:
        Where the copy went, or None when this kind gets no copy.
    """
    name = _input_name(eff, det.language, det.subtype)
    if name is None:
        return None
    key_dir = os.path.join(cache.render_root(), run_key)
    os.makedirs(key_dir, exist_ok=True)
    dest = os.path.join(key_dir, name)
    if not os.path.isfile(dest):
        shutil.copy2(body_path, dest)
    return dest


def _parses_jsonl(text: str) -> bool:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return False
    for ln in lines:
        try:
            json.loads(ln)
        except ValueError:
            return False
    return True


def _parses(kind: RenderKind, text: str) -> bool:
    try:
        if kind is RenderKind.JSON:
            json.loads(text)
        elif kind is RenderKind.JSONL:
            if not _parses_jsonl(text):
                return False
        elif kind is RenderKind.YAML:
            _yaml.safe_load(text)
        else:
            _minidom.parseString(text)
        return True
    except Exception:
        return False


def _raw_degraded(text: str, note: str) -> tuple[str, str, str, dict]:
    # The saved page reads its note from the facts.
    return _raw_pre(text), "raw", note, {"mode": "raw", "note": note}


def _apply_cap(frag: str | None, text: str, extras: dict) -> tuple[str | None, str, str | None, dict]:
    if frag is not None and len(frag) > _RICH_MAX:
        return _raw_degraded(text, _TOO_LARGE_NOTE)
    return frag, extras.get("mode"), None, extras


def _text_fragment(eff: RenderKind, input_path: str | None, body: bytes, size: int) -> tuple[str | None, str, str | None, dict]:
    text = body.decode("utf-8")
    if size > _RICH_MAX:
        return _raw_degraded(text, _TOO_LARGE_NOTE)
    if eff is RenderKind.CSV:
        rows = [ln for ln in text.split("\n") if ln.strip()]
        if not rows:
            return _raw_degraded(text, _EMPTY_CSV_NOTE)
        counts = {ln.count(",") + 1 for ln in rows}
        if len(rows) < 2 or len(counts) != 1 or next(iter(counts)) < 2:
            return _raw_degraded(text, _parse_note(RenderKind.CSV))
        frag, extras = fragments.build_fragment_sync(eff, input_path, size)
        return _apply_cap(frag, text, extras)
    if eff in _PARSE_VALIDATED and not _parses(eff, text):
        return _raw_degraded(text, _parse_note(eff))
    frag, extras = fragments.build_fragment_sync(eff, input_path, size)
    return _apply_cap(frag, text, extras)


_WINDOWED_KINDS = frozenset({
    RenderKind.CODE, RenderKind.TEXT, RenderKind.JSON, RenderKind.JSONL,
    RenderKind.YAML, RenderKind.XML, RenderKind.HTML,
})
_WINDOW_LINES = 1000
_CL_RE = re.compile(r'<div class="cl">.*?</div>', re.DOTALL)


def _code_lines(fragment: str) -> tuple[str, list[str]]:
    """Split colored code HTML into its opening part and its lines.

    Args:
        fragment: The colored code HTML.

    Returns:
        The opening part and the HTML of each line.
    """
    marker = '<div class="cl">'
    idx = fragment.find(marker)
    if idx < 0:
        return fragment, []
    return fragment[:idx], _CL_RE.findall(fragment[idx:])


def _window_if(eff: RenderKind, full_frag: str | None, mode: str | None, total: int | None,
               offset: int, full: bool) -> tuple[str | None, int | None, bool | None]:
    """Cut a page of code down to a block of lines.

    Kinds that are not shown line by line, and the full page, keep all of it.

    Args:
        eff: What sort of body this is.
        full_frag: The whole page, or None.
        mode: How the body is shown.
        total: How many lines the body has.
        offset: The line to start at.
        full: True to keep the whole page.

    Returns:
        The page to show, the line it starts at, and whether more lines are
        left. The last two are None when the whole page is kept.
    """
    if full or full_frag is None or mode in (None, "raw") or eff not in _WINDOWED_KINDS or not total:
        return full_frag, None, None
    prefix, lines = _code_lines(full_frag)
    window = lines[offset:offset + _WINDOW_LINES]
    truncated = offset + len(window) < total
    if offset == 0:
        return prefix + "".join(window) + "</pre>", 0, truncated
    return "".join(window), offset, truncated


def _text_descriptor(task_id: str, size: int, eff: RenderKind, det: Detection, frag: str | None,
                      mode: str | None, note: str | None, meta: dict, created_at: str | None,
                      line_offset: int | None = None, truncated: bool | None = None,
                      declared: RenderKind | None = None) -> dict:
    """Build the description of a text body.

    Args:
        task_id: The run the body came from.
        size: How big the body is in bytes.
        eff: What sort of body this is.
        det: What the body was found to be.
        frag: The page to show, or None.
        mode: How the body is shown.
        note: A message for the reader, or None.
        meta: The saved facts about the body.
        created_at: When the page was made, or None.
        line_offset: The line the page starts at, or None.
        truncated: Whether more lines are left, or None.
        declared: The kind the tool fixed for its results, or None.

    Returns:
        Everything the viewer needs to show the body.
    """
    per_type = {f: meta.get(f) for f in ("line_count", "row_count", "column_count", "provenance")}
    return _descriptor(task_id, size, eff, detected_kind=det.kind.value,
                        kind_label=_kind_label(eff, det.language, det.subtype),
                        declared_kind=declared.value if declared is not None else None,
                        mode=mode, fragment=frag, note=note, language=det.language,
                        created_at=created_at, line_offset=line_offset, truncated=truncated, **per_type)


def _descriptor(name: str, size: int, kind: RenderKind, **fields: _Field) -> dict:
    d = {
        "task_id": None, "kind": kind.value, "detected_kind": None, "declared_kind": None,
        "kind_label": None,
        "mode": None, "fragment": None, "url": None, "name": name, "size": size,
        "created_at": None, "error": None, "note": None, "language": None,
        "line_count": None, "row_count": None, "column_count": None, "pages": None,
        "dimensions": None, "duration": None, "provenance": None,
        "line_offset": None, "truncated": None,
    }
    d.update(fields)
    return d


def _stat_and_read(body_path: str) -> tuple[os.stat_result, bytes]:
    with open(body_path, "rb") as fh:
        body = fh.read()
    return os.stat(body_path), body


def _structured_detection(text: str) -> Detection | None:
    try:
        json.loads(text)
    except ValueError:
        pass
    else:
        return Detection(RenderKind.JSON)
    if _parses_jsonl(text):
        return Detection(RenderKind.JSONL)
    return None


async def _detect_text(text: str) -> Detection:
    as_structured = await asyncio.to_thread(_structured_detection, text)
    if as_structured is not None:
        return as_structured
    return await detect(text.encode("utf-8"))


async def build_result_descriptor(task_id: str, run_dir: str, *, line_offset: int = 0,
                                   kind: RenderKind | None = None,
                                   declared_kind: RenderKind | None = None,
                                   full: bool = False) -> dict:
    """Describe what a task run gave back.

    Asking for a kind shows the body that way instead of the way it was
    found to be, and nothing is saved. A kind the tool declared wins over
    everything else and is saved and reused like an auto-detected page.

    Args:
        task_id: The run to describe.
        run_dir: Where the run keeps its files.
        line_offset: The line to start a code page at.
        kind: Show the body as this kind, or None to work it out.
        declared_kind: The kind the tool fixed for its results, or None.
        full: True to show the whole body at once.

    Returns:
        Everything the viewer needs to show the body.

    Raises:
        FileNotFoundError: If the run gave nothing back.
    """
    body_path, body_ref = await asyncio.to_thread(_body_source, task_id, run_dir)
    if not await asyncio.to_thread(os.path.isfile, body_path):
        raise FileNotFoundError(task_id)
    st, body = await asyncio.to_thread(_stat_and_read, body_path)
    size, mtime = st.st_size, int(st.st_mtime)
    if size == 0:
        return _descriptor(task_id, 0, RenderKind.UNKNOWN,
                           detected_kind=RenderKind.UNKNOWN.value, kind_label="file", mode="unsupported")
    text = await asyncio.to_thread(_as_text, body)
    if text is None:
        det = await detect(body)
        return _descriptor("result.bin", size, RenderKind.UNKNOWN, detected_kind=det.kind.value,
                           kind_label="file", mode="legacy")
    det = await _detect_text(text)
    if det.kind not in _TEXT_FAMILY:
        det = await _reclassify(body)
    run_key = cache.key_for(body_ref, mtime)
    eff = declared_kind if declared_kind is not None else (
        kind if kind is not None else det.kind)
    return await _text_family(task_id, run_key, body_path, body, mtime, det, size, eff, kind,
                                line_offset, full, declared_kind, body_ref)


def _cached_kind_matches(cached: tuple[str, dict], eff: RenderKind) -> bool:
    """True when the saved page was built for the same kind.

    Args:
        cached: The saved page and its saved facts.
        eff: The kind the body is shown as.

    Returns:
        Whether the saved page fits the kind.
    """
    return cached[1].get("kind") == eff.value


async def _text_family(task_id: str, run_key: str, body_path: str, body: bytes, mtime: int,
                        det: Detection, size: int, eff: RenderKind, kind: RenderKind | None,
                        line_offset: int, full: bool, declared: RenderKind | None,
                        body_ref: str) -> dict:
    """Describe a text body the run gave back.

    A page built on its own is saved and used again later. A page built for
    an asked for kind is never saved.

    Args:
        task_id: The run the body came from.
        run_key: The name the run is saved under.
        body_path: Where the body is on disk.
        body: The bytes of the body.
        mtime: When the body last changed.
        det: What the body was found to be.
        size: How big the body is in bytes.
        eff: What sort of body this is.
        kind: The kind that was asked for, or None.
        line_offset: The line to start a code page at.
        full: True to show the whole body at once.
        declared: The kind the tool fixed for its results, or None.
        body_ref: The body's path inside the data folder.

    Returns:
        Everything the viewer needs to show the body.
    """
    input_path = await asyncio.to_thread(_materialize, run_key, body_path, eff, det)
    if kind is None:                                 # No kind was asked for, so a saved page can be reused.
        cached = await cache.read_doc(run_key)
        if cached is not None and _cached_kind_matches(cached, eff):
            doc, cmeta = cached
            full_frag, meta = (doc or None), cmeta
            mode, note, created_at = cmeta.get("mode"), cmeta.get("note"), cmeta.get("created_at")
        else:
            full_frag, mode, note, meta = await asyncio.to_thread(_text_fragment, eff, input_path, body, size)
            entry = fragments.text_meta(eff, body_ref, mtime, size, meta)
            await cache.write_doc(run_key, full_frag or "", None, entry)
            created_at = entry["created_at"]
            # Saving the page wipes the folder, so the copy is put back.
            await asyncio.to_thread(_materialize, run_key, body_path, eff, det)
    else:                                            # A kind was asked for, so the page is built fresh.
        full_frag, mode, note, meta = await asyncio.to_thread(_text_fragment, eff, input_path, body, size)
        created_at = None
    frag, line_offset, truncated = await asyncio.to_thread(
        _window_if, eff, full_frag, mode, meta.get("line_count"), line_offset, full)
    return _text_descriptor(task_id, size, eff, det, frag, mode, note, meta, created_at,
                             line_offset=line_offset, truncated=truncated, declared=declared)


class _Field(Protocol):
    """A value a result description holds."""
