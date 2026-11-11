"""Tests for app.lib.shared.hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.lib.shared.hashing import hash_file_content


def test_fingerprints_known_content(tmp_path: Path) -> None:
    """The fingerprint is the sha256 of the file bytes."""
    path = tmp_path / "file.png"
    path.write_bytes(b"reference image")

    assert hash_file_content(str(path)) == hashlib.sha256(b"reference image").hexdigest()


def test_fingerprint_changes_with_content(tmp_path: Path) -> None:
    """New bytes give a new fingerprint."""
    path = tmp_path / "file.png"
    path.write_bytes(b"one")
    first = hash_file_content(str(path))
    path.write_bytes(b"two")

    assert hash_file_content(str(path)) != first


def test_fingerprints_large_binary(tmp_path: Path) -> None:
    """A multi-chunk binary file is fingerprinted in one pass."""
    path = tmp_path / "big.bin"
    payload = bytes(range(256)) * 512
    path.write_bytes(payload)

    assert hash_file_content(str(path)) == hashlib.sha256(payload).hexdigest()
