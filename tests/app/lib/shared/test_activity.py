"""Tests for resource interest times."""

import errno
import math
import os
import time
from pathlib import Path

import pytest

from app.lib.shared import activity

_ATTRIBUTE = "user.agent.last_access"
_OLD = 1700000000


async def test_record_survives_unrelated_reads(tmp_path: Path) -> None:
    """Only explicit interest and writes change the resource age."""
    path = tmp_path / "file.txt"
    path.write_text("content")
    os.utime(path, (_OLD, _OLD))
    await activity.record("file.txt", str(tmp_path))
    used = activity.last_used_sync(str(path), path.stat())
    assert used > time.time() - 10
    assert path.read_text() == "content"
    os.utime(path, (time.time(), _OLD))
    assert activity.last_used_sync(str(path), path.stat()) == used
    assert path.stat().st_mtime == _OLD


def test_missing_interest_uses_mtime(tmp_path: Path) -> None:
    """A recent filesystem read does not renew an untracked resource."""
    path = tmp_path / "file.txt"
    path.write_text("content")
    os.utime(path, (time.time(), _OLD))
    assert activity.last_used_sync(str(path), path.stat()) == _OLD


@pytest.mark.parametrize("raw", [b"garbage", b"nan", b"inf", b"-1"])
def test_invalid_interest_keeps_file(tmp_path: Path, raw: bytes) -> None:
    """A broken interest time cannot make a file expire."""
    path = tmp_path / "file.txt"
    path.write_text("content")
    os.setxattr(path, _ATTRIBUTE, raw)
    assert math.isinf(activity.last_used_sync(str(path), path.stat()))


def test_unreadable_interest_keeps_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed metadata read cannot make a file expire."""
    path = tmp_path / "file.txt"
    path.write_text("content")

    def denied(_path: str, _name: str) -> bytes:
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(os, "getxattr", denied)
    assert math.isinf(activity.last_used_sync(str(path), path.stat()))


async def test_failed_record_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed interest write is reported to the caller."""
    path = tmp_path / "file.txt"
    path.write_text("content")

    def denied(_path: str, _name: str, _value: bytes) -> None:
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(os, "setxattr", denied)
    with pytest.raises(PermissionError):
        await activity.record(str(path), str(tmp_path))


@pytest.mark.parametrize("path", ["../outside.txt", "link.txt", "missing.txt", ".", "bad\x00.txt"])
async def test_ignores_nonresource_files(tmp_path: Path, path: str) -> None:
    """Missing files, folders and outside paths are not marked."""
    root = tmp_path / "resources"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    os.utime(outside, (_OLD, _OLD))
    (root / "link.txt").symlink_to(outside)
    await activity.record(path, str(root))
    assert activity.last_used_sync(str(outside), outside.stat()) == _OLD


def test_sync_record_uses_configured_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sync writer saves interest in the configured resource folder."""
    path = tmp_path / "file.txt"
    path.write_text("content")
    monkeypatch.setenv("RESOURCES_DIR", str(tmp_path))
    activity.record_sync("file.txt")
    assert activity.last_used_sync(str(path), path.stat()) > time.time() - 10
