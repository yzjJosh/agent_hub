"""Records when someone opens a resource file."""

import asyncio
import errno
import logging
import math
import os
import time

from app.lib.shared.paths import resolve_path_sync, resources_dir

__all__ = ["record", "record_sync", "last_used_sync"]

_ATTRIBUTE = "user.agent.last_access"
_logger = logging.getLogger(__name__)


def record_sync(path: str, root: str | None = None) -> None:
    """Record interest in an existing resource file.

    Args:
        path: File path, relative to root or absolute within root.
        root: Resource folder, or the configured resource folder when unset.

    Raises:
        OSError: If the interest cannot be saved.
    """
    if "\x00" in path:
        return
    target = resolve_path_sync(root if root is not None else resources_dir(), path)
    if target is None or not os.path.isfile(target):
        return
    try:
        os.setxattr(target, _ATTRIBUTE, str(time.time()).encode("ascii"))
    except OSError:
        _logger.exception("Cannot record resource interest for %s", target)
        raise


async def record(path: str, root: str | None = None) -> None:
    """Record interest in an existing resource file.

    Args:
        path: File path, relative to root or absolute within root.
        root: Resource folder, or the configured resource folder when unset.

    Raises:
        OSError: If the interest cannot be saved.
    """
    await asyncio.to_thread(record_sync, path, root)


def last_used_sync(path: str, stat: os.stat_result) -> float:
    """Get the last interest or write time of a resource file.

    Files with unreadable or invalid interest times are kept.

    Args:
        path: File on disk.
        stat: Current file facts.

    Returns:
        The last interest or write time, or infinity when it cannot be read.
    """
    try:
        raw = os.getxattr(path, _ATTRIBUTE)
    except OSError as exc:
        if exc.errno == errno.ENODATA:
            return stat.st_mtime
        _logger.warning("Cannot read resource interest for %s, keeping file: %s", path, exc)
        return math.inf
    try:
        last_used = float(raw)
        if not math.isfinite(last_used) or last_used < 0:
            raise ValueError("invalid interest time")
    except ValueError:
        _logger.warning("Invalid resource interest for %s, keeping file", path)
        return math.inf
    return max(stat.st_mtime, last_used)
