"""Builds the description of a file for the viewer."""

from __future__ import annotations

import asyncio
import os
from typing import TypedDict, Unpack

from app.lib.render import cache, fragments, office, probe
from app.lib.render.cache import CachedDocument
from app.lib.render.classify import RenderKind, classify, kind_label
from app.lib.render.fragments import BuiltFragment
from app.lib.shared.paths import ephemeral_dav_url, resolve_path, resources_dir

__all__ = [
    "DescriptorFields",
    "OFFICE_KINDS",
    "build_converting_descriptor",
    "build_descriptor",
    "get_resource_path",
    "resolve_view",
]

_TEXT_KINDS = frozenset({
    RenderKind.MARKDOWN,
    RenderKind.CODE,
    RenderKind.TEXT,
    RenderKind.CSV,
    RenderKind.JSON,
    RenderKind.JSONL,
    RenderKind.YAML,
    RenderKind.XML,
})
OFFICE_KINDS = frozenset({RenderKind.OFFICE_DOC, RenderKind.OFFICE_PPT, RenderKind.XLSX, RenderKind.XLS})
_MEDIA_KINDS = frozenset({RenderKind.PDF, RenderKind.IMAGE, RenderKind.VIDEO, RenderKind.AUDIO, RenderKind.HTML})


async def resolve_view(path: str) -> str | None:
    """Turn a resource path into a full path on disk.

    Args:
        path: A path inside the resources folder.

    Returns:
        The full path, or None when the path is not allowed.
    """
    if "\x00" in path:
        return None
    return await resolve_path(resources_dir(), path)


def get_resource_path(full_path: str, fallback: str) -> str:
    """Return the path inside the resources folder for a file on disk.

    Args:
        full_path: Where the file is on disk.
        fallback: The path to use when the file is outside the resources folder.

    Returns:
        The path inside the resources folder, or the fallback.
    """
    relative_path = os.path.relpath(full_path, os.path.realpath(resources_dir()))
    if relative_path == ".." or relative_path.startswith(".." + os.sep):
        return fallback
    return relative_path


class DescriptorFields(TypedDict, total=False):
    """Optional fields merged into a viewer descriptor."""

    mode: str | None
    fragment: str | None
    url: str | None
    error: str | None
    provenance: str | None
    note: str | None
    language: str | None
    line_count: int | None
    row_count: int | None
    column_count: int | None
    pages: int | None
    dimensions: list[int] | None
    duration: float | None


def _assemble_descriptor(
    path: str,
    name: str,
    size: int,
    modified_time: int,
    kind: RenderKind,
    **fields: Unpack[DescriptorFields],
) -> dict:
    descriptor: dict = {
        "path": path,
        "name": name,
        "size": size,
        "mtime": modified_time,
        "kind": kind.value,
        "kind_label": kind_label(kind, name, fields.get("language")),
        "mode": None,
        "fragment": None,
        "url": None,
        "error": None,
        "provenance": None,
        "note": None,
        "language": None,
        "line_count": None,
        "row_count": None,
        "column_count": None,
        "pages": None,
        "dimensions": None,
        "duration": None,
    }
    descriptor.update(fields)
    return descriptor


def build_converting_descriptor(
    path: str, name: str, size: int, modified_time: int, kind: RenderKind
) -> dict:
    """Describe a file that is still being converted.

    Args:
        path: A path inside the resources folder.
        name: The file name.
        size: How big the file is in bytes.
        modified_time: When the file last changed.
        kind: What sort of file this is.

    Returns:
        A description that says the file is converting.
    """
    return _assemble_descriptor(path, name, size, modified_time, kind, mode="converting")


async def _build_text_body(
    kind: RenderKind, full_path: str, size: int, modified_time: int, path: str, key: str
) -> BuiltFragment:
    cached_document: CachedDocument | None = await cache.read_doc(key)
    if cached_document is not None:
        fragment_html = None if cached_document.html == "" else cached_document.html
        metadata = {field: cached_document.metadata.get(field) for field in fragments.META_FIELDS}
        return BuiltFragment(html=fragment_html, metadata=metadata)
    built = await fragments.build_fragment(kind, full_path, size)
    cache_metadata = fragments.build_text_metadata(kind, path, modified_time, size, built.metadata)
    await cache.write_doc(key, built.html or "", None, cache_metadata)
    metadata = {field: built.metadata.get(field) for field in fragments.META_FIELDS}
    return BuiltFragment(html=built.html, metadata=metadata)


async def _build_office_descriptor(
    resource_path: str, name: str, size: int, modified_time: int, kind: RenderKind, full_path: str, key: str
) -> dict:
    cached_document = await cache.read_doc(key)
    if cached_document is not None:
        metadata = cached_document.metadata
        mode = metadata.get("mode", "doc")
        pdf_url = f"/view/doc/{key}/{cache.OFFICE_PDF_ASSET}" if mode == "frame" else None
        return _assemble_descriptor(
            resource_path,
            name,
            size,
            modified_time,
            kind,
            mode=mode,
            fragment=cached_document.html,
            provenance=metadata.get("provenance"),
            row_count=metadata.get("row_count"),
            column_count=metadata.get("column_count"),
            url=pdf_url,
        )
    reason = await cache.get_convert_failure_reason(key)
    if reason is not None:
        return _assemble_descriptor(
            resource_path, name, size, modified_time, kind, mode="convert_failed", error=reason
        )
    await office.start_office_conversion(resource_path, key, full_path)
    return build_converting_descriptor(resource_path, name, size, modified_time, kind)


async def build_descriptor(path: str, full_path: str) -> dict:
    """Describe one file for the viewer.

    Args:
        path: A path inside the resources folder.
        full_path: Where the file is on disk.

    Returns:
        Everything the viewer needs to show the file.
    """
    resource_path = get_resource_path(full_path, path)
    file_stat = await asyncio.to_thread(os.stat, full_path)
    kind = await classify(full_path)
    size, modified_time = file_stat.st_size, int(file_stat.st_mtime)
    name = os.path.basename(resource_path)
    key = cache.build_cache_key(resource_path, modified_time)
    if kind in _TEXT_KINDS:
        built = await _build_text_body(kind, full_path, size, modified_time, resource_path, key)
        return _assemble_descriptor(
            resource_path, name, size, modified_time, kind, fragment=built.html, **built.metadata
        )
    if kind in OFFICE_KINDS:
        return await _build_office_descriptor(resource_path, name, size, modified_time, kind, full_path, key)
    if kind in _MEDIA_KINDS:
        metadata = await probe.probe_meta(kind, full_path)
        mode = "frame" if kind in (RenderKind.PDF, RenderKind.HTML) else "native"
        return _assemble_descriptor(
            resource_path, name, size, modified_time, kind, mode=mode, url=ephemeral_dav_url(resource_path), **metadata
        )
    return _assemble_descriptor(resource_path, name, size, modified_time, kind, mode="unsupported")
