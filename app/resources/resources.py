"""Share a folder of files over WebDAV and as MCP resources."""

import asyncio
import mimetypes
import os
from collections.abc import Awaitable, Callable, Sequence
from typing import override
from urllib.parse import quote

import mcp.types as mt
from fastmcp import FastMCP
from fastmcp.resources.base import Resource
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from wsgidav.wsgidav_app import WsgiDAVApp

from app.lib.shared import activity
from app.lib.shared.paths import resolve_path, resolve_path_sync, resources_dir
from app.resources.dav_hook import CacheInvalidatingProvider

__all__ = [
    "binary_download_notice",
    "create_dav_app",
    "dav_base_url",
    "dav_download_url",
    "register_resources",
]


_UPLOAD_DIR = "upload"
_DEFAULT_DAV_BASE_URL = "http://localhost:8315"


def dav_base_url() -> str:
    """Base URL of this server for file transfer.

    Returns:
        The DAV_BASE_URL setting with any trailing slash removed, or
        http://localhost:8315 when it is not set.
    """
    return (os.getenv("DAV_BASE_URL") or _DEFAULT_DAV_BASE_URL).rstrip("/")


def dav_download_url(resource_path: str) -> str:
    """The public address that downloads a file under the resource folder.

    Args:
        resource_path: Path of the file, relative to the resource root.

    Returns:
        The address that downloads the file.
    """
    return f"{dav_base_url()}/dav/{quote(resource_path, safe='/')}"


async def binary_download_notice(path: str) -> str:
    """Build the notice shown in place of a binary file.

    Args:
        path: File path under the shared folder.

    Returns:
        Text with the size of the file and a command that downloads it.

    Raises:
        OSError: If the file is missing.
    """
    filename = os.path.basename(path)
    size = await asyncio.to_thread(os.path.getsize, os.path.join(resources_dir(), path))
    return (
        f"Binary file with size {size} bytes. "
        "Please use WebDAV to download it instead:\n"
        f"  curl -o /tmp/{filename} {dav_base_url()}/dav/{path}"
    )


def create_dav_app(root: str) -> WsgiDAVApp:
    """Create a WebDAV app that serves a folder to anyone.

    Clients may only write inside the upload folder, and the rest of the
    share is read only. A write, delete, or move drops the render entries
    for the paths it touched.

    Args:
        root: Folder to share.

    Returns:
        The WebDAV app.
    """
    upload_dir = os.path.join(root, _UPLOAD_DIR)
    os.makedirs(upload_dir, exist_ok=True)
    config = {
        "provider_mapping": {
            "/": CacheInvalidatingProvider(root, share_root=root, readonly=True, fs_opts={}),
            f"/{_UPLOAD_DIR}": CacheInvalidatingProvider(upload_dir, share_root=root, fs_opts={}),
        },
        "simple_dc": {"user_mapping": {"*": True}},
        "enable_propfind_depth_1": True,
        "verbose": 0,
    }
    return WsgiDAVApp(config)


async def _iter_files(root: str) -> list[str]:
    """List every file under a folder, sorted.

    Args:
        root: Folder to look in.

    Returns:
        One path per file, each one relative to root. The list is empty when
        the folder is missing.
    """
    return await asyncio.to_thread(_walk_files, root)


def _walk_files(root: str) -> list[str]:
    """List every file under a folder, sorted.

    Args:
        root: Folder to look in.

    Returns:
        One path per file, each one relative to root. The list is empty when
        the folder is missing.
    """
    base = os.path.realpath(root)
    if not os.path.isdir(base):
        return []
    paths = []
    for dirpath, _dirnames, filenames in os.walk(base):
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            paths.append(os.path.relpath(full, base))
    return sorted(paths)


async def _list_files(root: str, path: str) -> str | None:
    """List the files under a folder with their sizes, one per line.

    Args:
        root: Share root the paths are relative to.
        path: Folder to list, relative to root. It is empty for the root
            itself.

    Returns:
        One line per file with its path and size, a marker line when the
        folder has no files, or None when the folder is missing.
    """
    return await asyncio.to_thread(_folder_listing, root, path)


def _folder_listing(root: str, path: str) -> str | None:
    """List the files under a folder with their sizes, one per line.

    Args:
        root: Share root the paths are relative to.
        path: Folder to list, relative to root. It is empty for the root
            itself.

    Returns:
        One line per file with its path and size, a marker line when the
        folder has no files, or None when the folder is missing.
    """
    base = os.path.realpath(root)
    target = resolve_path_sync(root, path)
    if target is None or not os.path.isdir(target):
        return None
    lines = []
    for dirpath, _dirnames, filenames in os.walk(target):
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            rel = os.path.relpath(full, base)
            lines.append(f"{rel}\t{os.path.getsize(full)} bytes")
    lines.sort()
    return "\n".join(lines) if lines else "(empty)"


def _load_text(path: str) -> str | None:
    """Read a file as text.

    Args:
        path: File to read.

    Returns:
        The text of the file, or None when the file does not hold text.
    """
    with open(path, "rb") as f:
        data = f.read()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return None if b"\x00" in data else text


async def _read_file(root: str, path: str) -> str | None:
    """Read a file under root, or list a folder under root.

    Args:
        root: Folder the path is relative to.
        path: Path relative to root. A trailing slash is allowed.

    Returns:
        The text of the file, a download notice when the file is not text, a
        listing when the path is a folder, or None when the path is missing
        or points outside root.
    """
    path = path.rstrip("/")
    target = await resolve_path(root, path)
    if target is None:
        return None
    if await asyncio.to_thread(os.path.isdir, target):
        return await _list_files(root, path)
    if not await asyncio.to_thread(os.path.isfile, target):
        return None
    await activity.record(target, root)
    text = await asyncio.to_thread(_load_text, target)
    if text is None:
        return await binary_download_notice(path)
    return text


def _file_reader(root: str, rel: str) -> Callable[[], Awaitable[str]]:
    """Build a reader for one file under root.

    Args:
        root: Folder the path is relative to.
        rel: Path of the file, relative to root.

    Returns:
        A function that reads the file, and reports it as missing when it is
        gone.
    """

    async def read() -> str:
        return await _read_file(root, rel) or f"Resource not found: {rel}"

    return read


class _FileListMiddleware(Middleware):
    """Adds one resource entry for each file on disk."""

    def __init__(self, root: str) -> None:
        self._root = root

    @override
    async def on_list_resources(
        self,
        context: MiddlewareContext[mt.ListResourcesRequest],
        call_next: CallNext[mt.ListResourcesRequest, Sequence[Resource]],
    ) -> Sequence[Resource]:
        resources = list(await call_next(context))
        for rel in await _iter_files(self._root):
            resources.append(
                Resource.from_function(
                    _file_reader(self._root, rel),
                    uri=f"resources:///{rel}",
                    name=rel,
                    mime_type=mimetypes.guess_type(rel)[0] or "application/octet-stream",
                )
            )
        return resources


def register_resources(mcp: FastMCP, root: str) -> None:
    """Add a folder to a server as MCP resources.

    The whole folder is listed at resources:/// and any path inside it can be
    read under that same address. Every file on disk also shows up in the
    resource list.

    Args:
        mcp: Server to add the resources to.
        root: Folder to share.
    """
    mcp.add_middleware(_FileListMiddleware(root))

    async def list_resources() -> str:
        content = await _read_file(root, "")
        if content is None:
            return f"Resources directory not found: {root}"
        return content

    async def read_resource(path: str) -> str:
        content = await _read_file(root, path)
        if content is None:
            return f"Resource not found: {path.rstrip('/')}"
        return content

    mcp.resource("resources:///")(list_resources)
    mcp.resource("resources:///{path*}")(read_resource)
