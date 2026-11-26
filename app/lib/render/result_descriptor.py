"""Describes what a task run gave back so the viewer can show it."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import html
import json
import os
import re
import shutil
from typing import Unpack
from xml.dom import minidom

import yaml

import app.lib.text_type.detector as detector
from app.lib.render import cache, fragments
from app.lib.render.cache import CachedDocument
from app.lib.render.classify import RenderKind
from app.lib.render.descriptor import DescriptorFields, OFFICE_KINDS
from app.lib.render.detect import BINARY_KINDS, TEXT_FAMILY_KINDS, Detection, detect, language_extension

__all__ = ["build_result_descriptor"]

_KIBIBYTE = 1024
_MAX_RICH_RENDER_BYTES = _KIBIBYTE * _KIBIBYTE
_MAX_RAW_PREVIEW_BYTES = 256 * _KIBIBYTE
_WINDOW_LINES = 1000
_PARSE_VALIDATED_KINDS = frozenset({
    RenderKind.JSON, RenderKind.JSONL, RenderKind.YAML, RenderKind.XML,
})
_EXTENSIONLESS_TEXT_KINDS = frozenset({
    RenderKind.MARKDOWN, RenderKind.TEXT, RenderKind.CSV,
    RenderKind.JSON, RenderKind.JSONL, RenderKind.YAML, RenderKind.XML,
})
_WINDOWED_KINDS = frozenset({
    RenderKind.CODE, RenderKind.TEXT, RenderKind.JSON, RenderKind.JSONL,
    RenderKind.YAML, RenderKind.XML, RenderKind.HTML,
})
_CODE_LINE_DIV_PATTERN = re.compile(r'<div class="cl">.*?</div>', re.DOTALL)

_TOO_LARGE_NOTE = "Too large to render — showing first 256 KB. Switch to Raw or Download for the full payload."
_EMPTY_CSV_NOTE = "No rows found."


@dataclass(frozen=True)
class _ResultFile:
    disk_path: str
    relative_path: str
    size_bytes: int
    modified_time_seconds: int
    body: bytes


@dataclass(frozen=True)
class _RenderedFragment:
    fragment_html: str | None
    mode: str | None
    note: str | None
    metadata: dict


@dataclass(frozen=True)
class _SplitCodeFragment:
    opening_html: str
    line_divs: list[str]


@dataclass(frozen=True)
class _FragmentWindow:
    fragment_html: str | None
    line_offset: int | None
    is_truncated: bool | None


@dataclass(frozen=True)
class _TextResultContext:
    task_id: str
    cache_key: str
    result_file: _ResultFile
    detection: Detection
    effective_kind: RenderKind
    requested_kind: RenderKind | None
    declared_kind: RenderKind | None
    line_offset: int
    is_full: bool


class _ResultDescriptorFields(DescriptorFields, total=False):
    """Optional fields merged into a task result descriptor."""

    task_id: str | None
    detected_kind: str | None
    declared_kind: str | None
    kind_label: str | None
    created_at: str | None
    line_offset: int | None
    truncated: bool | None


def _read_result_file(task_id: str, run_dir: str) -> _ResultFile | None:
    disk_path = os.path.join(run_dir, "result")
    if not os.path.isfile(disk_path):
        return None
    with open(disk_path, "rb") as file_handle:
        body = file_handle.read()
    file_stat = os.stat(disk_path)
    return _ResultFile(
        disk_path=disk_path,
        relative_path=f"runs/{task_id}/result",
        size_bytes=file_stat.st_size,
        modified_time_seconds=int(file_stat.st_mtime),
        body=body,
    )


def _decode_utf8_text(body: bytes) -> str | None:
    if b"\x00" in body:
        return None
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return None


async def _classify_text_bytes(body: bytes) -> Detection:
    label = await detector.identify(body)
    return Detection(detector.kind_for_label(label), detector.language_for_label(label))


def _format_parse_error_note(kind: RenderKind) -> str:
    return (
        f"Parse error — the body is not valid {kind.value.upper()}. "
        f"Showing raw text. Use Raw or Download for the full payload."
    )


def _format_kind_label(kind: RenderKind, language: str | None, subtype: str | None) -> str:
    if kind is RenderKind.CODE:
        return f"code · {language}" if language else "code"
    if kind in OFFICE_KINDS:
        return f"office · {subtype}"
    if kind is RenderKind.UNKNOWN:
        return "file"
    return kind.value


def _wrap_raw_preview_html(text: str) -> str:
    return '<pre class="raw-pre">' + html.escape(text[:_MAX_RAW_PREVIEW_BYTES]) + "</pre>"


def _select_cached_input_filename(kind: RenderKind, language: str | None, subtype: str | None) -> str | None:
    if kind in BINARY_KINDS:
        return f"input.{subtype}"
    if kind is RenderKind.CODE:
        extension = language_extension(language)
        return f"input.{extension}" if extension else "input"
    if kind is RenderKind.HTML:
        return "input.html"
    if kind in _EXTENSIONLESS_TEXT_KINDS:
        return "input"
    return None


def _copy_input_to_cache_dir(
    cache_key: str,
    disk_path: str,
    effective_kind: RenderKind,
    detection: Detection,
) -> str | None:
    filename = _select_cached_input_filename(effective_kind, detection.language, detection.subtype)
    if filename is None:
        return None
    cache_dir = os.path.join(cache.get_render_root(), cache_key)
    os.makedirs(cache_dir, exist_ok=True)
    destination_path = os.path.join(cache_dir, filename)
    if not os.path.isfile(destination_path):
        shutil.copy2(disk_path, destination_path)
    return destination_path


def _is_valid_jsonl(text: str) -> bool:
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) < 2:
        return False
    for line in lines:
        try:
            json.loads(line)
        except ValueError:
            return False
    return True


def _is_valid_structured_text(kind: RenderKind, text: str) -> bool:
    try:
        if kind is RenderKind.JSON:
            json.loads(text)
        elif kind is RenderKind.JSONL:
            return _is_valid_jsonl(text)
        elif kind is RenderKind.YAML:
            yaml.safe_load(text)
        else:
            minidom.parseString(text)
        return True
    except Exception:
        return False


def _build_raw_fallback_fragment(text: str, note: str) -> _RenderedFragment:
    return _RenderedFragment(
        fragment_html=_wrap_raw_preview_html(text),
        mode="raw",
        note=note,
        metadata={"mode": "raw", "note": note},
    )


def _enforce_fragment_size_limit(
    fragment_html: str | None,
    text: str,
    metadata: dict,
) -> _RenderedFragment:
    if fragment_html is not None and len(fragment_html) > _MAX_RICH_RENDER_BYTES:
        return _build_raw_fallback_fragment(text, _TOO_LARGE_NOTE)
    return _RenderedFragment(
        fragment_html=fragment_html,
        mode=metadata.get("mode"),
        note=None,
        metadata=metadata,
    )


def _render_text_fragment(
    effective_kind: RenderKind,
    input_path: str,
    body: bytes,
    size_bytes: int,
) -> _RenderedFragment:
    text = body.decode("utf-8")
    if size_bytes > _MAX_RICH_RENDER_BYTES:
        return _build_raw_fallback_fragment(text, _TOO_LARGE_NOTE)
    if effective_kind is RenderKind.CSV:
        rows = [line for line in text.split("\n") if line.strip()]
        if not rows:
            return _build_raw_fallback_fragment(text, _EMPTY_CSV_NOTE)
        column_counts = {line.count(",") + 1 for line in rows}
        if len(rows) < 2 or len(column_counts) != 1 or next(iter(column_counts)) < 2:
            return _build_raw_fallback_fragment(text, _format_parse_error_note(RenderKind.CSV))
    elif effective_kind in _PARSE_VALIDATED_KINDS and not _is_valid_structured_text(effective_kind, text):
        return _build_raw_fallback_fragment(text, _format_parse_error_note(effective_kind))
    built = fragments.build_fragment_sync(effective_kind, input_path, size_bytes)
    return _enforce_fragment_size_limit(built.html, text, built.metadata)


def _split_code_fragment_lines(fragment_html: str) -> _SplitCodeFragment:
    line_marker = '<div class="cl">'
    first_line_index = fragment_html.find(line_marker)
    if first_line_index < 0:
        return _SplitCodeFragment(opening_html=fragment_html, line_divs=[])
    return _SplitCodeFragment(
        opening_html=fragment_html[:first_line_index],
        line_divs=_CODE_LINE_DIV_PATTERN.findall(fragment_html[first_line_index:]),
    )


def _slice_code_window(
    effective_kind: RenderKind,
    full_fragment_html: str | None,
    mode: str | None,
    total_lines: int | None,
    line_offset: int,
    is_full: bool,
) -> _FragmentWindow:
    if (
        is_full
        or full_fragment_html is None
        or mode in (None, "raw")
        or effective_kind not in _WINDOWED_KINDS
        or not total_lines
    ):
        return _FragmentWindow(fragment_html=full_fragment_html, line_offset=None, is_truncated=None)
    split_fragment = _split_code_fragment_lines(full_fragment_html)
    window_lines = split_fragment.line_divs[line_offset:line_offset + _WINDOW_LINES]
    is_truncated = line_offset + len(window_lines) < total_lines
    if line_offset == 0:
        window_html = split_fragment.opening_html + "".join(window_lines) + "</pre>"
        return _FragmentWindow(fragment_html=window_html, line_offset=0, is_truncated=is_truncated)
    return _FragmentWindow(
        fragment_html="".join(window_lines),
        line_offset=line_offset,
        is_truncated=is_truncated,
    )


def _build_descriptor_dict(
    name: str,
    size_bytes: int,
    kind: RenderKind,
    **fields: Unpack[_ResultDescriptorFields],
) -> dict:
    descriptor: dict = {
        "task_id": None, "kind": kind.value, "detected_kind": None, "declared_kind": None,
        "kind_label": None, "mode": None, "fragment": None, "url": None,
        "name": name, "size": size_bytes, "created_at": None, "error": None,
        "note": None, "language": None, "line_count": None, "row_count": None,
        "column_count": None, "pages": None, "dimensions": None, "duration": None,
        "provenance": None, "line_offset": None, "truncated": None,
    }
    descriptor.update(fields)
    return descriptor


def _build_text_descriptor(
    context: _TextResultContext,
    rendered: _RenderedFragment,
    window: _FragmentWindow,
    created_at: str | None,
) -> dict:
    metadata = rendered.metadata
    declared_kind_value = context.declared_kind.value if context.declared_kind is not None else None
    return _build_descriptor_dict(
        context.task_id,
        context.result_file.size_bytes,
        context.effective_kind,
        detected_kind=context.detection.kind.value,
        kind_label=_format_kind_label(
            context.effective_kind, context.detection.language, context.detection.subtype,
        ),
        declared_kind=declared_kind_value,
        mode=rendered.mode,
        fragment=window.fragment_html,
        note=rendered.note,
        language=context.detection.language,
        created_at=created_at,
        line_offset=window.line_offset,
        truncated=window.is_truncated,
        line_count=metadata.get("line_count"),
        row_count=metadata.get("row_count"),
        column_count=metadata.get("column_count"),
        provenance=metadata.get("provenance"),
    )


def _detect_json_or_jsonl(text: str) -> Detection | None:
    try:
        json.loads(text)
    except ValueError:
        pass
    else:
        return Detection(RenderKind.JSON)
    if _is_valid_jsonl(text):
        return Detection(RenderKind.JSONL)
    return None


async def _detect_text_kind(text: str) -> Detection:
    structured_detection = await asyncio.to_thread(_detect_json_or_jsonl, text)
    if structured_detection is not None:
        return structured_detection
    return await detect(text.encode("utf-8"))


def _has_matching_cached_kind(cached_entry: CachedDocument, effective_kind: RenderKind) -> bool:
    return cached_entry.metadata.get("kind") == effective_kind.value


async def _write_cached_text_fragment(context: _TextResultContext, rendered: _RenderedFragment) -> str:
    result_file = context.result_file
    cache_metadata = fragments.build_text_metadata(
        context.effective_kind,
        result_file.relative_path,
        result_file.modified_time_seconds,
        result_file.size_bytes,
        rendered.metadata,
    )
    await cache.write_doc(context.cache_key, rendered.fragment_html or "", None, cache_metadata)
    # Writing the cached page replaces the cache folder.
    await asyncio.to_thread(
        _copy_input_to_cache_dir,
        context.cache_key,
        result_file.disk_path,
        context.effective_kind,
        context.detection,
    )
    return cache_metadata["created_at"]


async def _describe_text_result(context: _TextResultContext) -> dict:
    result_file = context.result_file
    input_path = await asyncio.to_thread(
        _copy_input_to_cache_dir,
        context.cache_key,
        result_file.disk_path,
        context.effective_kind,
        context.detection,
    )
    source_path = input_path or result_file.disk_path
    cached_entry = await cache.read_doc(context.cache_key) if context.requested_kind is None else None
    if cached_entry is not None and _has_matching_cached_kind(cached_entry, context.effective_kind):
        cached_metadata = cached_entry.metadata
        rendered = _RenderedFragment(
            fragment_html=cached_entry.html or None,
            mode=cached_metadata.get("mode"),
            note=cached_metadata.get("note"),
            metadata=cached_metadata,
        )
        created_at: str | None = cached_metadata.get("created_at")
    else:
        rendered = await asyncio.to_thread(
            _render_text_fragment,
            context.effective_kind,
            source_path,
            result_file.body,
            result_file.size_bytes,
        )
        created_at = await _write_cached_text_fragment(context, rendered) if context.requested_kind is None else None
    window = await asyncio.to_thread(
        _slice_code_window,
        context.effective_kind,
        rendered.fragment_html,
        rendered.mode,
        rendered.metadata.get("line_count"),
        context.line_offset,
        context.is_full,
    )
    return _build_text_descriptor(context, rendered, window, created_at)


async def build_result_descriptor(
    task_id: str, run_dir: str, *, line_offset: int = 0,
    kind: RenderKind | None = None, declared_kind: RenderKind | None = None, full: bool = False,
) -> dict:
    """Describe what a task run gave back.

    Passing a kind shows the body as that kind without saving a cached page.
    A declared kind takes priority over both the requested kind and detection.

    Args:
        task_id: The run to describe.
        run_dir: Where the run keeps its files.
        line_offset: The line to start a code page at.
        kind: Show the body as this kind, or None to detect it.
        declared_kind: The fixed kind for the tool's results, or None.
        full: True to return the whole body without slicing lines.

    Returns:
        A descriptor dictionary for the viewer.

    Raises:
        FileNotFoundError: If the run has no result file.
    """
    result_file = await asyncio.to_thread(_read_result_file, task_id, run_dir)
    if result_file is None:
        raise FileNotFoundError(task_id)
    if result_file.size_bytes == 0:
        return _build_descriptor_dict(
            task_id, 0, RenderKind.UNKNOWN,
            detected_kind=RenderKind.UNKNOWN.value, kind_label="file", mode="unsupported",
        )
    text = await asyncio.to_thread(_decode_utf8_text, result_file.body)
    if text is None:
        detection = await detect(result_file.body)
        return _build_descriptor_dict(
            "result.bin", result_file.size_bytes, RenderKind.UNKNOWN,
            detected_kind=detection.kind.value, kind_label="file", mode="legacy",
        )
    detection = await _detect_text_kind(text)
    if detection.kind not in TEXT_FAMILY_KINDS:
        detection = await _classify_text_bytes(result_file.body)
    effective_kind = declared_kind if declared_kind is not None else (kind if kind is not None else detection.kind)
    cache_key = cache.build_cache_key(result_file.relative_path, result_file.modified_time_seconds)
    context = _TextResultContext(
        task_id=task_id,
        cache_key=cache_key,
        result_file=result_file,
        detection=detection,
        effective_kind=effective_kind,
        requested_kind=kind,
        declared_kind=declared_kind,
        line_offset=line_offset,
        is_full=full,
    )
    return await _describe_text_result(context)
