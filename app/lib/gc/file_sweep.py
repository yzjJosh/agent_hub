"""Removes files nobody has used for a while, and the empty folders left behind."""

import os
import sys
import time
from collections.abc import Callable

__all__ = ["sweep"]

_SECONDS_PER_DAY = 86400
_DEFAULT_DAYS = 30


def sweep(
    root: str, days: int = _DEFAULT_DAYS,
    *, last_used: Callable[[str, os.stat_result], float] | None = None,
) -> list[str]:
    """Remove files not used within the given number of days.

    Folders left empty are removed too. The root folder is always kept.

    Args:
        root: Folder to sweep.
        days: Days a file may sit unused before it is removed.
        last_used: Gets the last use time of a file. Defaults to the later of its read and write times.

    Returns:
        Paths of every removed file and folder.
    """
    base = os.path.realpath(root)
    if not os.path.isdir(base):
        return []
    cutoff = time.time() - days * _SECONDS_PER_DAY
    removed: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(base, topdown=False):
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            if _remove_stale_file(full, cutoff, last_used):
                removed.append(full)
        if dirpath != base and _remove_empty_dir(dirpath):
            removed.append(dirpath)
    return removed


def _remove_stale_file(
    path: str, cutoff: float, last_used: Callable[[str, os.stat_result], float] | None,
) -> bool:
    """Remove a file that was last used before the cutoff.

    Args:
        path: File to check and maybe remove.
        cutoff: A file older than this time is removed.
        last_used: Gets the last use time of a file. Defaults to the later of its read and write times.

    Returns:
        True when the file was removed.
    """
    try:
        stat = os.stat(path)
    except OSError as e:
        print(f"[file_sweep] skipping {path}: {e}", file=sys.stderr)
        return False
    used = last_used(path, stat) if last_used is not None else max(stat.st_atime, stat.st_mtime)
    if used >= cutoff:
        return False
    try:
        os.remove(path)
    except OSError as e:
        print(f"[file_sweep] cannot remove {path}: {e}", file=sys.stderr)
        return False
    return True


def _remove_empty_dir(path: str) -> bool:
    """Remove a folder when it is empty.

    Args:
        path: Folder to check and maybe remove.

    Returns:
        True when the folder was removed.
    """
    try:
        if os.listdir(path):
            return False
        os.rmdir(path)
    except OSError as e:
        print(f"[file_sweep] cannot remove dir {path}: {e}", file=sys.stderr)
        return False
    return True
