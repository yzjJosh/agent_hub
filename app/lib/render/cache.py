"""Saves rendered files and cleans out old ones."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import threading
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote

from app.lib.gc import file_sweep
from app.lib.render import gen_state
from app.lib.shared.paths import data_dir, resolve_path_sync, resources_dir

__all__ = [
    "CachedDocument",
    "OFFICE_PDF_ASSET",
    "build_cache_key",
    "build_thumbnail_url",
    "clear_failed",
    "get_convert_failure_reason",
    "get_render_root",
    "get_thumbnail_status",
    "get_thumbnail_status_sync",
    "invalidate_sync",
    "mark_failed",
    "read_doc",
    "sweep_sync",
    "write_doc",
]

_FailedKind = Literal["thumb", "convert"]
_FAILED_KINDS: tuple[_FailedKind, ...] = ("thumb", "convert")

_RUN_RESULT = re.compile(r"^runs/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/result$")
OFFICE_PDF_ASSET = "preview.pdf"

# The render format version folded into every key.
_KEY_VERSION = 3


@dataclass(frozen=True)
class CachedDocument:
    """Saved document HTML and metadata read from the render cache."""

    html: str
    metadata: dict


@dataclass(frozen=True)
class _CachedIdentity:
    """Source path and modified time recorded in a cached render entry."""

    relative_path: str
    modified_time: int


def get_render_root() -> str:
    """Get the folder that holds rendered files.

    Returns:
        The path to that folder.
    """
    return os.path.join(data_dir(), "render")


def _get_state_root() -> str:
    return os.path.join(get_render_root(), ".state")


def build_cache_key(path: str, modified_time: int) -> str:
    """Make the name a file is saved under.

    Args:
        path: A path inside the resources folder.
        modified_time: When the file last changed.

    Returns:
        The saved name for that file.
    """
    return hashlib.sha256(f"{_KEY_VERSION}\0{path}\0{modified_time}".encode("utf-8")).hexdigest()


def _get_key_directory(key: str) -> str:
    return os.path.join(get_render_root(), key)


def _read_doc_sync(key: str) -> CachedDocument | None:
    doc_path = os.path.join(_get_key_directory(key), "doc.html")
    metadata_path = os.path.join(_get_key_directory(key), "meta.json")
    try:
        with open(doc_path, "r", encoding="utf-8") as file_handle:
            doc_html = file_handle.read()
        with open(metadata_path, "r", encoding="utf-8") as file_handle:
            metadata = json.load(file_handle)
        return CachedDocument(html=doc_html, metadata=metadata)
    except (OSError, json.JSONDecodeError):
        return None


async def read_doc(key: str) -> CachedDocument | None:
    """Read a saved render.

    Args:
        key: The name the file is saved under.

    Returns:
        The saved document and metadata, or None when nothing valid is there.
    """
    return await asyncio.to_thread(_read_doc_sync, key)


def _write_doc_sync(key: str, doc_html: str, assets: dict[str, bytes] | None, metadata: dict) -> None:
    root = get_render_root()
    os.makedirs(root, exist_ok=True)
    staging = os.path.join(root, f".staging-{key}-{os.getpid()}-{threading.get_ident()}")
    try:
        os.makedirs(staging)
        with open(os.path.join(staging, "doc.html"), "w", encoding="utf-8") as file_handle:
            file_handle.write(doc_html)
        for name, content in (assets or {}).items():
            with open(os.path.join(staging, name), "wb") as file_handle:
                file_handle.write(content)
        with open(os.path.join(staging, "meta.json"), "w", encoding="utf-8") as file_handle:
            json.dump(metadata, file_handle)
        target = _get_key_directory(key)
        if os.path.isdir(target):
            shutil.rmtree(target)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


async def write_doc(key: str, doc_html: str, assets: dict[str, bytes] | None, metadata: dict) -> None:
    """Save a render.

    A reader never sees a half saved one.

    Args:
        key: The name to save under.
        doc_html: The page to save.
        assets: Extra files to save next to the page, or None.
        metadata: The facts about the file.
    """
    await asyncio.to_thread(_write_doc_sync, key, doc_html, assets, metadata)


def build_thumbnail_url(path: str) -> str:
    """Get the web address of a file's thumbnail.

    Args:
        path: A path inside the resources folder.

    Returns:
        The address of the thumbnail.
    """
    return "/api/resources/thumb?path=" + quote(path, safe="/")


def _get_marker_path(key: str, kind: _FailedKind) -> str:
    return os.path.join(_get_state_root(), f"{key}.{kind}-failed")


def _mark_failed_sync(key: str, kind: _FailedKind, reason: str) -> None:
    os.makedirs(_get_state_root(), exist_ok=True)
    with open(_get_marker_path(key, kind), "w", encoding="utf-8") as file_handle:
        file_handle.write(reason)


async def mark_failed(key: str, kind: _FailedKind, reason: str) -> None:
    """Mark that a thumbnail or a conversion failed.

    The mark stays after a restart.

    Args:
        key: The name the file is saved under.
        kind: Whether a thumbnail or a conversion failed.
        reason: What went wrong.
    """
    await asyncio.to_thread(_mark_failed_sync, key, kind, reason)


def _clear_failed_sync(key: str, kind: _FailedKind) -> None:
    try:
        os.remove(_get_marker_path(key, kind))
    except FileNotFoundError:
        pass


async def clear_failed(key: str, kind: _FailedKind) -> None:
    """Remove a failure mark.

    Does nothing when there is no mark.

    Args:
        key: The name the file is saved under.
        kind: Whether a thumbnail or a conversion failed.
    """
    await asyncio.to_thread(_clear_failed_sync, key, kind)


def _get_convert_failure_reason_sync(key: str) -> str | None:
    try:
        with open(_get_marker_path(key, "convert"), "r", encoding="utf-8") as file_handle:
            return file_handle.read()
    except OSError:
        return None


async def get_convert_failure_reason(key: str) -> str | None:
    """Get why a conversion failed.

    Args:
        key: The name the file is saved under.

    Returns:
        What went wrong, or None when the conversion did not fail.
    """
    return await asyncio.to_thread(_get_convert_failure_reason_sync, key)


def _get_thumbnail_status_sync(key: str) -> str:
    if os.path.exists(os.path.join(_get_key_directory(key), "thumb.png")):
        return "done"
    if key in gen_state.generating_keys():
        return "generating"
    if os.path.exists(_get_marker_path(key, "thumb")):
        return "failed"
    return "pending"


async def get_thumbnail_status(key: str) -> str:
    """Say how a file's thumbnail is doing.

    Args:
        key: The name the file is saved under.

    Returns:
        One of done, generating, failed, or pending.
    """
    return await asyncio.to_thread(_get_thumbnail_status_sync, key)


def get_thumbnail_status_sync(key: str) -> str:
    """Say how a file's thumbnail is doing without an event loop.

    Args:
        key: The name the file is saved under.

    Returns:
        One of done, generating, failed, or pending.
    """
    return _get_thumbnail_status_sync(key)


def sweep_sync(days: int) -> list[str]:
    """Clean out saved renders that are no longer wanted without an event loop.

    A render goes away once its source file is gone or has changed. Renders
    that are old and untouched go away too. A render whose source cannot be
    told is kept until it goes old. Renders still being saved and the folder
    of failure marks are left alone until they go old.

    Args:
        days: How old and untouched a file must be before it goes away.

    Returns:
        The paths of everything that was removed.
    """
    root = get_render_root()
    removed = _remove_orphans(root, None)
    removed.extend(file_sweep.sweep(root, days))
    return removed


def invalidate_sync(path: str) -> list[str]:
    """Throw away the saved renders of one file or folder without an event loop.

    A render goes away only when its source is gone or has changed. A render
    whose source cannot be told is kept.

    Args:
        path: The file or folder that was deleted or changed, inside the
            resources folder.

    Returns:
        The paths of everything that was removed.
    """
    return _remove_orphans(get_render_root(), path)


def _remove_orphans(root: str, path: str | None) -> list[str]:
    if not os.path.isdir(root):
        return []
    prefix = path.rstrip("/") + "/" if path is not None else ""
    removed: list[str] = []
    for name in sorted(os.listdir(root)):
        if name.startswith("."):
            continue
        key_directory = os.path.join(root, name)
        if not os.path.isdir(key_directory):
            continue
        identity = _read_identity(key_directory)
        if identity is None:
            continue
        if path is not None and identity.relative_path != path and not identity.relative_path.startswith(prefix):
            continue
        if not _is_source_live(identity.relative_path, identity.modified_time):
            removed.extend(_remove_entry(name, root))
    return removed


def _read_identity(key_directory: str) -> _CachedIdentity | None:
    try:
        with open(os.path.join(key_directory, "meta.json"), "r", encoding="utf-8") as file_handle:
            metadata = json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    relative_path = metadata.get("path")
    modified_time = metadata.get("mtime")
    if not isinstance(relative_path, str) or not isinstance(modified_time, int) or isinstance(modified_time, bool):
        return None
    return _CachedIdentity(relative_path=relative_path, modified_time=modified_time)


def _remove_entry(key: str, root: str) -> list[str]:
    removed: list[str] = []
    key_directory = os.path.join(root, key)
    try:
        shutil.rmtree(key_directory)
        removed.append(key_directory)
    except OSError as error:
        print(f"[render.cache] cannot remove {key_directory}: {error}", file=sys.stderr)
    for kind in _FAILED_KINDS:
        marker_path = _get_marker_path(key, kind)
        if os.path.exists(marker_path):
            try:
                os.remove(marker_path)
                removed.append(marker_path)
            except OSError as error:
                print(f"[render.cache] cannot remove {marker_path}: {error}", file=sys.stderr)
    return removed


def _is_source_live(path: str, modified_time: int) -> bool:
    if _RUN_RESULT.match(path):
        full_path = os.path.join(data_dir(), path)
        try:
            return os.path.isfile(full_path) and int(os.stat(full_path).st_mtime) == modified_time
        except OSError:
            return False
    try:
        resolved = resolve_path_sync(resources_dir(), path)
    except ValueError:
        return False
    if resolved is None:
        return False
    try:
        if not os.path.isfile(resolved):
            return False
        return int(os.stat(resolved).st_mtime) == modified_time
    except OSError:
        return False
