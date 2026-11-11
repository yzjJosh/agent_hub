"""Content fingerprints for resource files."""

from __future__ import annotations

import hashlib

__all__ = ["hash_file_content"]

_CHUNK_SIZE = 65536


def hash_file_content(path: str) -> str:
    """Return the sha256 fingerprint of a file.

    Args:
        path: Local path of the file.

    Returns:
        The fingerprint as lower case text.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            sha.update(chunk)
    return sha.hexdigest()
