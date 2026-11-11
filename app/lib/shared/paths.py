"""The data, resources and scratch folder roots, and safe path resolving under them."""

__all__ = [
    "EPHEMERAL_PARAM",
    "data_dir",
    "download_dir",
    "ephemeral_dav_url",
    "is_audio_mime",
    "is_binary_mime",
    "is_image_file",
    "is_image_mime",
    "is_pdf_mime",
    "is_text_mime",
    "is_video_mime",
    "mime_type",
    "resolve_path",
    "resolve_path_sync",
    "resources_dir",
    "scratch_dir",
]

import asyncio
import os
from urllib.parse import quote

from app.lib.cli.cli_runner import run_cli

_DEFAULT_DATA_DIR = "/app/data"
_DEFAULT_RESOURCES_DIR = "/app/data/resources"
_DEFAULT_SCRATCH_DIR = "/dev/shm/agent"
_DOWNLOAD_NAME = "downloads"
# A WebDAV read with this query parameter does not count as a use of the file.
EPHEMERAL_PARAM = "ephemeral"
_TEXTUAL_MIMES = frozenset({"application/json", "application/xml", "application/xhtml+xml"})


def data_dir() -> str:
    """Root directory of the run records, run folders and render cache.

    Returns:
        The DATA_DIR environment value, or /app/data when unset or empty
    """
    return os.getenv("DATA_DIR") or _DEFAULT_DATA_DIR


def resources_dir() -> str:
    """Root directory of the shared resources folder.

    Returns:
        The RESOURCES_DIR environment value, or /app/data/resources when unset
        or empty
    """
    return os.getenv("RESOURCES_DIR") or _DEFAULT_RESOURCES_DIR


def scratch_dir(name: str) -> str:
    """Folder for short lived files of one kind.

    The folder is not made here, the caller makes it before writing.

    Args:
        name: Name of the folder under the scratch root

    Returns:
        The path name under the SCRATCH_DIR environment value, or under
        the default scratch root when it is unset or empty
    """
    return os.path.join(os.getenv("SCRATCH_DIR") or _DEFAULT_SCRATCH_DIR, name)


def download_dir() -> str:
    """Folder that Chrome saves downloaded files in.

    Returns:
        The downloads folder under the scratch root
    """
    return scratch_dir(_DOWNLOAD_NAME)


def ephemeral_dav_url(path: str) -> str:
    """Get the WebDAV address that reads a resource file without counting as a use.

    Args:
        path: File path inside the resource folder.

    Returns:
        The address under /dav, marked as an ephemeral read.
    """
    return f"/dav/{quote(path, safe='/')}?{EPHEMERAL_PARAM}=1"


def resolve_path_sync(root: str, path: str) -> str | None:
    """Resolve a relative path under root, without an event loop.

    This is for code that already runs in its own thread, and for code that
    runs outside the server. Never call it on the event loop, use
    resolve_path there.

    Args:
        root: Share root the path is relative to
        path: Path relative to root

    Returns:
        The absolute local path, or None when the path escapes root
    """
    base = os.path.realpath(root)
    target = os.path.realpath(os.path.join(base, path))
    if target != base and not target.startswith(base + os.sep):
        return None
    return target


async def resolve_path(root: str, path: str) -> str | None:
    """Resolve a relative path under root, rejecting traversal outside it.

    Args:
        root: Share root the path is relative to
        path: Path relative to root

    Returns:
        The absolute local path, or None when the path escapes root
    """
    return await asyncio.to_thread(resolve_path_sync, root, path)


async def mime_type(file_path: str) -> str | None:
    """Detect a file's MIME type.

    Args:
        file_path: Path to the file

    Returns:
        The MIME type, or None when it cannot be read
    """
    if not file_path:
        return None
    result = await run_cli(["exiftool", "-b", "-MimeType", file_path],
                           "exiftool mime detection")
    if result.returncode == 0 and result.stdout:
        return result.stdout
    return None


def is_image_mime(mime: str | None) -> bool:
    """True when the MIME type is an image."""
    return mime is not None and mime.startswith("image/")


def is_video_mime(mime: str | None) -> bool:
    """True when the MIME type is a video."""
    return mime is not None and mime.startswith("video/")


def is_audio_mime(mime: str | None) -> bool:
    """True when the MIME type is an audio."""
    return mime is not None and mime.startswith("audio/")


def is_text_mime(mime: str | None) -> bool:
    """True when the MIME type is a text/* type."""
    return mime is not None and mime.startswith("text/")


def is_pdf_mime(mime: str | None) -> bool:
    """True when the MIME type is a PDF."""
    return mime is not None and mime == "application/pdf"


def is_binary_mime(mime: str | None) -> bool | None:
    """Classify a MIME type as binary or textual.

    Returns:
        True for image, video, audio, PDF, or any other detected MIME type;
        False for text and known textual document types; None when the MIME
        type is empty or missing
    """
    if not mime:
        return None
    if is_text_mime(mime) or mime in _TEXTUAL_MIMES:
        return False
    return True


async def is_image_file(file_path: str) -> bool:
    """Check whether a file is an image.

    Args:
        file_path: Path to the file

    Returns:
        True when the file is an image
    """
    return is_image_mime(await mime_type(file_path))
