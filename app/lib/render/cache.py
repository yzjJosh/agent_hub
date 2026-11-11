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
from typing import Literal
from urllib.parse import quote

from app.lib.gc.file_sweep import sweep as _file_sweep
from app.lib.render import gen_state
from app.lib.shared.paths import data_dir, resolve_path_sync, resources_dir

__all__ = [
    "OFFICE_PDF_ASSET",
    "clear_failed",
    "convert_failed_reason",
    "invalidate_sync",
    "key_for",
    "mark_failed",
    "read_doc",
    "render_root",
    "sweep_sync",
    "thumb_status",
    "thumb_status_sync",
    "thumb_url",
    "write_doc",
]

_FailedKind = Literal["thumb", "convert"]

_RUN_RESULT = re.compile(r"^runs/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/result$")
OFFICE_PDF_ASSET = "preview.pdf"

# The render format version folded into every key.
_KEY_VERSION = 3


def render_root() -> str:
    """Get the folder that holds rendered files.

    Returns:
        The path to that folder.
    """
    return os.path.join(data_dir(), "render")


def _state_root() -> str:
    """Get the folder that holds failure marks.

    Returns:
        The path to that folder.
    """
    return os.path.join(render_root(), ".state")


def key_for(path: str, mtime: int) -> str:
    """Make the name a file is saved under.

    Args:
        path: A path inside the shared folder.
        mtime: When the file last changed.

    Returns:
        The saved name for that file.
    """
    return hashlib.sha256(f"{_KEY_VERSION}\0{path}\0{mtime}".encode("utf-8")).hexdigest()


def _key_dir(key: str) -> str:
    return os.path.join(render_root(), key)


def _read_doc_sync(key: str) -> tuple[str, dict] | None:
    doc_path = os.path.join(_key_dir(key), "doc.html")
    meta_path = os.path.join(_key_dir(key), "meta.json")
    try:
        with open(doc_path, "r", encoding="utf-8") as fh:
            doc_html = fh.read()
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return doc_html, meta
    except (OSError, json.JSONDecodeError):
        return None


async def read_doc(key: str) -> tuple[str, dict] | None:
    """Read a saved render.

    Args:
        key: The name the file is saved under.

    Returns:
        The saved page and the facts about it, or None when nothing good is
        there.
    """
    return await asyncio.to_thread(_read_doc_sync, key)


def _write_doc_sync(key: str, doc_html: str, assets: dict[str, bytes] | None, meta: dict) -> None:
    root = render_root()
    os.makedirs(root, exist_ok=True)
    staging = os.path.join(root, f".staging-{key}-{os.getpid()}-{threading.get_ident()}")
    try:
        os.makedirs(staging)
        with open(os.path.join(staging, "doc.html"), "w", encoding="utf-8") as fh:
            fh.write(doc_html)
        for name, data in (assets or {}).items():
            with open(os.path.join(staging, name), "wb") as fh:
                fh.write(data)
        with open(os.path.join(staging, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        target = _key_dir(key)
        if os.path.isdir(target):
            shutil.rmtree(target)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


async def write_doc(key: str, doc_html: str, assets: dict[str, bytes] | None, meta: dict) -> None:
    """Save a render.

    A reader never sees a half saved one.

    Args:
        key: The name to save under.
        doc_html: The page to save.
        assets: Extra files to save next to the page, or None.
        meta: The facts about the file.
    """
    await asyncio.to_thread(_write_doc_sync, key, doc_html, assets, meta)


def thumb_url(path: str) -> str:
    """Get the web address of a file's thumbnail.

    Args:
        path: A path inside the shared folder.

    Returns:
        The address of the thumbnail.
    """
    return "/api/resources/thumb?path=" + quote(path, safe="/")


def _marker_path(key: str, kind: _FailedKind) -> str:
    return os.path.join(_state_root(), f"{key}.{kind}-failed")


def _mark_failed_sync(key: str, kind: _FailedKind, reason: str) -> None:
    os.makedirs(_state_root(), exist_ok=True)
    with open(_marker_path(key, kind), "w", encoding="utf-8") as fh:
        fh.write(reason)


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
        os.remove(_marker_path(key, kind))
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


def _convert_failed_reason_sync(key: str) -> str | None:
    try:
        with open(_marker_path(key, "convert"), "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


async def convert_failed_reason(key: str) -> str | None:
    """Get why a conversion failed.

    Args:
        key: The name the file is saved under.

    Returns:
        What went wrong, or None when the conversion did not fail.
    """
    return await asyncio.to_thread(_convert_failed_reason_sync, key)


def _thumb_status_sync(key: str) -> str:
    if os.path.exists(os.path.join(_key_dir(key), "thumb.png")):
        return "done"
    if key in gen_state.generating_keys():
        return "generating"
    if os.path.exists(_marker_path(key, "thumb")):
        return "failed"
    return "pending"


async def thumb_status(key: str) -> str:
    """Say how a file's thumbnail is doing.

    Args:
        key: The name the file is saved under.

    Returns:
        One of done, generating, failed, or pending.
    """
    return await asyncio.to_thread(_thumb_status_sync, key)


def thumb_status_sync(key: str) -> str:
    """Say how a file's thumbnail is doing, without an event loop.

    This is for code that already runs off the loop, such as a listing inside
    a worker thread. Never call it on the event loop.

    Args:
        key: The name the file is saved under.

    Returns:
        One of done, generating, failed, or pending.
    """
    return _thumb_status_sync(key)


def sweep_sync(days: int) -> list[str]:
    """Clean out saved renders that are no longer wanted, without an event loop.

    This is for code that runs outside the server, such as the garbage
    collector command. Never call it on the event loop.

    A render goes away once its source file is gone or has changed. Renders
    that are old and untouched go away too. A render whose source cannot be
    told is kept until it goes old. Renders still being saved and the folder
    of failure marks are left alone until they go old.

    Args:
        days: How old and untouched a file must be before it goes away.

    Returns:
        The paths of everything that was removed.
    """
    root = render_root()
    removed = _remove_orphans(root, None)
    removed.extend(_file_sweep(root, days))
    return removed


def invalidate_sync(path: str) -> list[str]:
    """Throw away the saved renders of one file or folder, without an event loop.

    This is for code that runs outside the server, such as the WebDAV worker
    threads. Never call it on the event loop.

    A render goes away only when its source is gone or has changed. A render
    whose source cannot be told is kept.

    Args:
        path: The file or folder that was deleted or changed, inside the
            shared folder.

    Returns:
        The paths of everything that was removed.
    """
    return _remove_orphans(render_root(), path)


def _remove_orphans(root: str, path: str | None) -> list[str]:
    """Remove saved renders whose source file is gone or has changed.

    Args:
        root: The folder that holds saved renders.
        path: Look only at renders of this file or of files inside it. None
            looks at all of them.

    Returns:
        The paths of everything that was removed.
    """
    if not os.path.isdir(root):
        return []
    prefix = path.rstrip("/") + "/" if path is not None else ""
    removed: list[str] = []
    for name in sorted(os.listdir(root)):
        if name.startswith("."):
            continue
        key_dir = os.path.join(root, name)
        if not os.path.isdir(key_dir):
            continue
        identity = _read_identity(key_dir)
        if identity is None:
            continue
        source_path, mtime = identity
        if path is not None and source_path != path and not source_path.startswith(prefix):
            continue
        if not _source_is_live(source_path, mtime):
            removed.extend(_remove_entry(name, root))
    return removed


def _read_identity(key_dir: str) -> tuple[str, int] | None:
    """Find out which file a saved render came from.

    Args:
        key_dir: The folder that holds one saved render.

    Returns:
        The source path and when it last changed, or None when that cannot
        be told.
    """
    try:
        with open(os.path.join(key_dir, "meta.json"), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    path = meta.get("path")
    mtime = meta.get("mtime")
    if not isinstance(path, str) or not isinstance(mtime, int) or isinstance(mtime, bool):
        return None
    return path, mtime


def _remove_entry(key: str, root: str) -> list[str]:
    """Remove one saved render and its failure marks.

    Args:
        key: The name the file is saved under.
        root: The folder that holds saved renders.

    Returns:
        The paths of everything that was removed.
    """
    removed: list[str] = []
    key_dir = os.path.join(root, key)
    try:
        shutil.rmtree(key_dir)
        removed.append(key_dir)
    except OSError as exc:
        print(f"[render.cache] cannot remove {key_dir}: {exc}", file=sys.stderr)
    for kind in ("thumb", "convert"):
        marker = _marker_path(key, kind)
        if os.path.exists(marker):
            try:
                os.remove(marker)
                removed.append(marker)
            except OSError as exc:
                print(f"[render.cache] cannot remove {marker}: {exc}", file=sys.stderr)
    return removed


def _source_is_live(path: str, mtime: int) -> bool:
    """Check that a source file is still there and unchanged.

    Args:
        path: Where the source file was.
        mtime: When it last changed.

    Returns:
        True when the file is still there and still that old.
    """
    if _RUN_RESULT.match(path):
        full = os.path.join(data_dir(), path)
        try:
            return os.path.isfile(full) and int(os.stat(full).st_mtime) == mtime
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
        return int(os.stat(resolved).st_mtime) == mtime
    except OSError:
        return False
