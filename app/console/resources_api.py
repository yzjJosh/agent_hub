"""Endpoints for listing, searching, and downloading resource files."""

from __future__ import annotations

import asyncio
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from app.console.errors import error_response
from app.console.windows import parse_int
from app.lib.render import cache, thumbs
from app.lib.render.cache import thumb_status_sync
from app.lib.render.classify import classify_sync, kind_label
from app.lib.shared import activity
from app.lib.shared.paths import resolve_path, resolve_path_sync, resources_dir
from app.lib.shared.timefmt import LOCAL_TIME_FORMAT

__all__ = [
    "download_resource",
    "list_resources",
    "search_resources",
]

_SEARCH_NODE_CAP = 5000
_SEARCH_DEFAULT_LIMIT = 200
_SEARCH_MAX_LIMIT = 1000


@dataclass(frozen=True)
class _SubtreeStats:
    """File count and byte total under one folder."""

    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class _DirectoryScan:
    """Normalized folder path and the entries inside it."""

    normalized_path: str
    entries: list[dict]


@dataclass(frozen=True)
class _ResourceSearchResult:
    """Matching entries from a resource tree search."""

    matches: list[dict]
    is_truncated: bool


@dataclass(frozen=True)
class _SearchDirectory:
    """One queued folder during a resource tree walk."""

    relative_path: str
    ancestor_names: list[str]


@dataclass
class _SearchState:
    """Mutable progress of a resource tree search."""

    base_path: str
    needle: str
    limit: int
    node_cap: int
    matches: list[dict]
    queue: deque[_SearchDirectory]
    visited_count: int = 0
    is_truncated: bool = False


def _get_parent_directory(relative_path: str) -> str | None:
    if not relative_path:
        return None
    parts = relative_path.split("/")
    return "/".join(parts[:-1]) if len(parts) > 1 else ""


def _measure_subtree(full_path: str) -> _SubtreeStats:
    file_count = 0
    total_bytes = 0
    for directory_path, _directory_names, file_names in os.walk(full_path):
        for file_name in file_names:
            try:
                file_bytes = os.path.getsize(os.path.join(directory_path, file_name))
            except OSError:
                continue
            file_count += 1
            total_bytes += file_bytes
    return _SubtreeStats(file_count=file_count, total_bytes=total_bytes)


def _describe_file_metadata(full_path: str, relative_path: str, modified_time: int) -> dict:
    kind = classify_sync(full_path)
    engine = thumbs.thumb_engine(kind)
    status = "none" if engine is None else thumb_status_sync(cache.key_for(relative_path, modified_time))
    return {
        "kind": kind_label(kind, os.path.basename(relative_path), None),
        "thumb": {"status": status, "url": cache.thumb_url(relative_path)},
    }


def _describe_directory_entry(base_path: str, entry: os.DirEntry[str]) -> dict | None:
    try:
        stat_result = entry.stat()
    except OSError:
        return None
    relative_path = os.path.relpath(entry.path, base_path).replace(os.sep, "/")
    if resolve_path_sync(base_path, relative_path) is None:
        return None
    modified_time = int(stat_result.st_mtime)
    is_directory = entry.is_dir()
    item: dict = {
        "name": entry.name,
        "path": relative_path,
        "type": "dir" if is_directory else "file",
        "mtime": modified_time,
        "mtime_local": datetime.fromtimestamp(modified_time).strftime(LOCAL_TIME_FORMAT),
    }
    if is_directory:
        subtree = _measure_subtree(entry.path)
        item["file_count"] = subtree.file_count
        item["total_size"] = subtree.total_bytes
    else:
        item["size"] = stat_result.st_size
    return item


def _scan_directory(root_dir: str, relative_path: str) -> _DirectoryScan | None:
    target_path = resolve_path_sync(root_dir, relative_path)
    if target_path is None or not os.path.isdir(target_path):
        return None
    base_path = os.path.realpath(root_dir)
    entries: list[dict] = []
    with os.scandir(target_path) as scanner:
        for entry in scanner:
            item = _describe_directory_entry(base_path, entry)
            if item is not None:
                if item["type"] == "file":
                    item.update(_describe_file_metadata(entry.path, item["path"], item["mtime"]))
                entries.append(item)
    listed_path = os.path.relpath(target_path, base_path).replace(os.sep, "/")
    normalized_path = "" if listed_path == "." else listed_path
    return _DirectoryScan(normalized_path=normalized_path, entries=entries)


async def _build_directory_listing(root_dir: str, relative_path: str) -> dict | None:
    scan_result = await asyncio.to_thread(_scan_directory, root_dir, relative_path)
    if scan_result is None:
        return None
    entries = list(scan_result.entries)
    entries.sort(key=lambda item: (item["type"] != "dir", item["name"]))
    return {
        "path": scan_result.normalized_path,
        "parent": _get_parent_directory(scan_result.normalized_path),
        "entries": entries,
    }


def _describe_search_match(
    relative_path: str,
    entry: os.DirEntry[str],
    ancestor_names: list[str],
) -> dict | None:
    try:
        stat_result = entry.stat()
    except OSError:
        return None
    modified_time = int(stat_result.st_mtime)
    is_directory = entry.is_dir()
    item: dict = {
        "path": relative_path,
        "name": entry.name,
        "type": "dir" if is_directory else "file",
        "mtime": modified_time,
        "mtime_local": datetime.fromtimestamp(modified_time).strftime(LOCAL_TIME_FORMAT),
        "matched_ancestor_paths": list(ancestor_names),
    }
    if not is_directory:
        item["size"] = stat_result.st_size
        item.update(_describe_file_metadata(entry.path, relative_path, modified_time))
    return item


def _read_sorted_entries(full_path: str) -> list[os.DirEntry[str]] | None:
    try:
        with os.scandir(full_path) as scanner:
            return sorted(scanner, key=lambda entry: entry.name)
    except OSError:
        return None


def _visit_search_entry(
    state: _SearchState,
    parent: _SearchDirectory,
    entry: os.DirEntry[str],
) -> bool:
    if state.visited_count >= state.node_cap:
        state.is_truncated = True
        return False
    child_path = entry.name if parent.relative_path == "" else f"{parent.relative_path}/{entry.name}"
    if resolve_path_sync(state.base_path, child_path) is None:
        return True
    state.visited_count += 1
    if state.needle in child_path.lower():
        if len(state.matches) < state.limit:
            match_item = _describe_search_match(child_path, entry, parent.ancestor_names)
            if match_item is not None:
                state.matches.append(match_item)
        else:
            state.is_truncated = True
    if entry.is_dir():
        state.queue.append(_SearchDirectory(child_path, [*parent.ancestor_names, entry.name]))
    return True


def _search_resource_tree(
    root_dir: str,
    query: str,
    limit: int,
    node_cap: int = _SEARCH_NODE_CAP,
) -> _ResourceSearchResult:
    state = _SearchState(
        base_path=os.path.realpath(root_dir),
        needle=query.lower(),
        limit=limit,
        node_cap=node_cap,
        matches=[],
        queue=deque([_SearchDirectory("", [])]),
    )
    while state.queue:
        if state.visited_count >= state.node_cap:
            state.is_truncated = True
            break
        current = state.queue.popleft()
        full_path = (
            state.base_path
            if current.relative_path == ""
            else os.path.join(state.base_path, current.relative_path)
        )
        state.visited_count += 1
        entries = _read_sorted_entries(full_path)
        if entries is None:
            continue
        for entry in entries:
            if not _visit_search_entry(state, current, entry):
                break
    return _ResourceSearchResult(matches=state.matches, is_truncated=state.is_truncated)


def _parse_search_limit(raw_value: str | None) -> int | None:
    value = parse_int(raw_value, _SEARCH_DEFAULT_LIMIT)
    if value is None or value < 1 or value > _SEARCH_MAX_LIMIT:
        return None
    return value


async def list_resources(request: Request) -> Response:
    """List the files and folders inside one resource directory.

    Args:
        request: The incoming HTTP request.

    Returns:
        The folder listing, or an error reply.
    """
    raw_path = request.query_params.get("path")
    relative_path = (raw_path or "").strip("/")
    root_dir = resources_dir()
    if "\x00" in relative_path or await resolve_path(root_dir, relative_path) is None:
        return error_response(400, "bad_param", f"Invalid path: '{raw_path}'.")
    listing = await _build_directory_listing(root_dir, relative_path)
    if listing is None:
        return error_response(404, "not_found", f"No such directory: '{relative_path or '/'}'.")
    return JSONResponse(listing)


async def search_resources(request: Request) -> Response:
    """Search for files and folders matching a query in the resources folder.

    Args:
        request: The incoming HTTP request.

    Returns:
        The matching entries and truncation flag, or an error reply.
    """
    query = request.query_params.get("q")
    if not query:
        return error_response(400, "bad_param", "q must be a non-empty substring.")
    limit = _parse_search_limit(request.query_params.get("limit"))
    if limit is None:
        return error_response(400, "bad_param", "limit must be an integer between 1 and 1000.")
    search_result = await asyncio.to_thread(_search_resource_tree, resources_dir(), query, limit)
    return JSONResponse({"matches": search_result.matches, "truncated": search_result.is_truncated})


async def download_resource(request: Request) -> Response:
    """Send one resource file as a browser download attachment.

    Args:
        request: The incoming HTTP request.

    Returns:
        The file response, or an error reply.
    """
    raw_path = request.query_params.get("path")
    if request.method == "GET" and raw_path is not None:
        await activity.record(raw_path.strip("/"))
    if raw_path is None or not raw_path.strip():
        return error_response(400, "bad_param", "path must be a resource-relative file path.")
    relative_path = raw_path.strip("/")
    if "\x00" in relative_path:
        return error_response(400, "bad_param", f"Invalid path: '{raw_path}'.")
    full_path = await resolve_path(resources_dir(), relative_path)
    if full_path is None or not await asyncio.to_thread(os.path.isfile, full_path):
        return error_response(404, "not_found", f"No such file: '{relative_path}'.")
    return FileResponse(full_path, filename=os.path.basename(relative_path))
