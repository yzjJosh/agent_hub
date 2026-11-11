"""Tests for resource retention."""

import os
import time
from pathlib import Path

from app.lib.gc.resources import sweep
from app.lib.shared import activity

_OLD = 1700000000
_DAYS = 30


def test_ignores_filesystem_access(tmp_path: Path) -> None:
    """A bulk reader cannot keep old resources alive."""
    path = tmp_path / "stale.txt"
    path.write_text("stale")
    os.utime(path, (time.time(), _OLD))
    assert sweep(str(tmp_path), _DAYS) == [str(path)]
    assert not path.exists()


async def test_keeps_explicit_interest(tmp_path: Path) -> None:
    """Opening an old file keeps it through the next sweep."""
    path = tmp_path / "opened.txt"
    path.write_text("old")
    os.utime(path, (_OLD, _OLD))
    await activity.record(str(path), str(tmp_path))
    assert sweep(str(tmp_path), _DAYS) == []
    assert path.read_text() == "old"


def test_keeps_recent_write(tmp_path: Path) -> None:
    """A newly written resource is kept without an interest record."""
    path = tmp_path / "new.txt"
    path.write_text("new")
    assert sweep(str(tmp_path), _DAYS) == []
    assert path.read_text() == "new"
