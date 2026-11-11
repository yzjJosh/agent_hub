"""Tracks which files have a thumbnail being made right now.

The set lives on its own so both the thumbnail worker and the cache can read
it without importing each other.
"""

from __future__ import annotations

import threading

__all__ = [
    "gen_add",
    "gen_remove",
    "generating_keys",
]

_GEN_LOCK = threading.Lock()
_GENERATING: set[str] = set()


def gen_add(key: str) -> None:
    """Mark that a file's thumbnail is being made.

    Args:
        key: The name the file is saved under.
    """
    with _GEN_LOCK:
        _GENERATING.add(key)


def gen_remove(key: str) -> None:
    """Mark that a file's thumbnail is no longer being made.

    Does nothing when it was not being made.

    Args:
        key: The name the file is saved under.
    """
    with _GEN_LOCK:
        _GENERATING.discard(key)


def generating_keys() -> frozenset[str]:
    """List the files whose thumbnails are being made right now.

    Returns:
        The names those files are saved under.
    """
    with _GEN_LOCK:
        return frozenset(_GENERATING)
