"""Builds the description of a file for the viewer."""

from __future__ import annotations

import asyncio
import os
from typing import Protocol

from app.lib.shared.paths import ephemeral_dav_url, resolve_path, resources_dir
from app.lib.render import cache, fragments, office, probe
from app.lib.render.classify import RenderKind, classify, kind_label

__all__ = [
    "OFFICE_KINDS",
    "build_descriptor",
    "converting_descriptor",
    "resolve_view",
    "share_path_for",
]

_TEXT_KINDS = frozenset(
    {RenderKind.MARKDOWN, RenderKind.CODE, RenderKind.TEXT, RenderKind.CSV,
     RenderKind.JSON, RenderKind.JSONL, RenderKind.YAML, RenderKind.XML}
)
OFFICE_KINDS = frozenset({RenderKind.OFFICE_DOC, RenderKind.OFFICE_PPT, RenderKind.XLSX, RenderKind.XLS})
_MEDIA_KINDS = frozenset({RenderKind.PDF, RenderKind.IMAGE, RenderKind.VIDEO, RenderKind.AUDIO, RenderKind.HTML})


async def resolve_view(path: str) -> str | None:
    """Turn a shared path into a full path on disk.

    Args:
        path: A path inside the shared folder.

    Returns:
        The full path, or None when the path is not allowed.
    """
    if "\x00" in path:
        return None
    return await resolve_path(resources_dir(), path)


def share_path_for(full_path: str, fallback: str) -> str:
    """The path inside the shared folder for a file on disk.

    Args:
        full_path: Where the file is on disk.
        fallback: The path to use when the file is outside the shared folder.

    Returns:
        The path inside the shared folder, or the fallback.
    """
    rel = os.path.relpath(full_path, os.path.realpath(resources_dir()))
    if rel == ".." or rel.startswith(".." + os.sep):
        return fallback
    return rel


def _descriptor(path: str, name: str, size: int, mtime: int, kind: RenderKind, **fields: _Field) -> dict:
    d = {
        "path": path, "name": name, "size": size, "mtime": mtime,
        "kind": kind.value, "kind_label": kind_label(kind, name, fields.get("language")),
        "mode": None, "fragment": None, "url": None, "error": None,
        "provenance": None, "note": None, "language": None, "line_count": None,
        "row_count": None, "column_count": None, "pages": None, "dimensions": None,
        "duration": None,
    }
    d.update(fields)
    return d


def converting_descriptor(path: str, name: str, size: int, mtime: int, kind: RenderKind) -> dict:
    """Describe a file that is still being converted.

    Args:
        path: A path inside the shared folder.
        name: The file name.
        size: How big the file is in bytes.
        mtime: When the file last changed.
        kind: What sort of file this is.

    Returns:
        A description that says the file is converting.
    """
    return _descriptor(path, name, size, mtime, kind, mode="converting")


async def _build_text_body(kind: RenderKind, full_path: str, size: int, mtime: int, path: str, key: str) -> tuple[str | None, dict]:
    """Get the shown body of a text file.

    A file that cannot be read as text has no body.

    Args:
        kind: What sort of file this is.
        full_path: Where the file is on disk.
        size: How big the file is in bytes.
        mtime: When the file last changed.
        path: A path inside the shared folder.
        key: The name this file is saved under.

    Returns:
        The body to show, or None, plus the extra facts about the file.
    """
    cached = await cache.read_doc(key)
    if cached is not None:
        _doc, meta = cached
        return (None if _doc == "" else _doc), {f: meta.get(f) for f in fragments.META_FIELDS}
    fragment, extras = await fragments.build_fragment(kind, full_path, size)
    # The office code saves this same entry.
    await cache.write_doc(key, fragment or "", None, fragments.text_meta(kind, path, mtime, size, extras))
    return fragment, {f: extras.get(f) for f in fragments.META_FIELDS}


async def build_descriptor(path: str, full_path: str) -> dict:
    """Describe one file for the viewer.

    Args:
        path: A path inside the shared folder.
        full_path: Where the file is on disk.

    Returns:
        Everything the viewer needs to show the file.
    """
    share_path = share_path_for(full_path, path)
    stat = await asyncio.to_thread(os.stat, full_path)
    kind = await classify(full_path)
    size, mtime = stat.st_size, int(stat.st_mtime)
    name = os.path.basename(share_path)
    key = cache.key_for(share_path, mtime)
    if kind in _TEXT_KINDS:
        fragment, per_type = await _build_text_body(kind, full_path, size, mtime, share_path, key)
        return _descriptor(share_path, name, size, mtime, kind, fragment=fragment, **per_type)
    if kind in OFFICE_KINDS:
        cached = await cache.read_doc(key)
        if cached is not None:
            doc_html, meta = cached
            mode = meta.get("mode", "doc")
            return _descriptor(share_path, name, size, mtime, kind, mode=mode,
                               fragment=doc_html, provenance=meta.get("provenance"),
                               row_count=meta.get("row_count"), column_count=meta.get("column_count"),
                               url=(f"/view/doc/{key}/{cache.OFFICE_PDF_ASSET}"
                                    if mode == "frame" else None))
        reason = await cache.convert_failed_reason(key)
        if reason is not None:
            return _descriptor(share_path, name, size, mtime, kind, mode="convert_failed", error=reason)
        await office.start_office_conversion(share_path, key, full_path)
        return converting_descriptor(share_path, name, size, mtime, kind)
    if kind in _MEDIA_KINDS:
        meta = await probe.probe_meta(kind, full_path)
        mode = "frame" if kind in (RenderKind.PDF, RenderKind.HTML) else "native"
        return _descriptor(share_path, name, size, mtime, kind, mode=mode,
                           url=ephemeral_dav_url(share_path), **meta)
    return _descriptor(share_path, name, size, mtime, kind, mode="unsupported")


class _Field(Protocol):
    """A value a file description holds."""
