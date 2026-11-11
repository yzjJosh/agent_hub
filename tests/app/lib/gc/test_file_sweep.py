"""Tests for the file sweep."""

import os
import time
from pathlib import Path

from app.lib.gc.file_sweep import sweep

_SECONDS_PER_DAY = 86400


def _age_file(path: Path, days_old: float, read_days_old: float | None = None) -> None:
    """Set a file's write time and read time to the past."""
    now = time.time()
    mtime = now - days_old * _SECONDS_PER_DAY
    atime = mtime if read_days_old is None else now - read_days_old * _SECONDS_PER_DAY
    os.utime(path, (atime, mtime))


class TestSweep:
    def test_removes_stale_file(self, tmp_path: Path) -> None:
        """A file untouched for 30 days or more is removed and reported."""
        stale = tmp_path / "stale.txt"
        stale.write_text("old")
        _age_file(stale, 40)
        assert sweep(str(tmp_path)) == [str(stale)]
        assert not stale.exists()

    def test_keeps_fresh_file(self, tmp_path: Path) -> None:
        """A recently written file is kept."""
        fresh = tmp_path / "fresh.txt"
        fresh.write_text("new")
        assert sweep(str(tmp_path)) == []
        assert fresh.exists()

    def test_keeps_recently_read_file(self, tmp_path: Path) -> None:
        """An old file that was read lately is kept."""
        read = tmp_path / "read.txt"
        read.write_text("old but read")
        _age_file(read, 40, read_days_old=0.1)
        assert sweep(str(tmp_path)) == []
        assert read.exists()

    def test_keeps_recently_written_file(self, tmp_path: Path) -> None:
        """An old file that was written lately is kept."""
        written = tmp_path / "written.txt"
        written.write_text("old but rewritten")
        _age_file(written, 0.1, read_days_old=40)
        assert sweep(str(tmp_path)) == []
        assert written.exists()

    def test_removes_empty_dirs_after_sweep(self, tmp_path: Path) -> None:
        """Old files and the folders left empty are removed, the root stays."""
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        stale = nested / "stale.txt"
        stale.write_text("old")
        _age_file(stale, 40)
        removed = sweep(str(tmp_path))
        expected = {str(nested), str(nested / "stale.txt"), str(tmp_path / "a")}
        assert set(removed) == expected
        assert tmp_path.is_dir()

    def test_keeps_dir_containing_fresh_file(self, tmp_path: Path) -> None:
        """A folder with an old file and a new file loses only the old one."""
        nested = tmp_path / "a"
        nested.mkdir()
        stale = nested / "stale.txt"
        stale.write_text("old")
        _age_file(stale, 40)
        fresh = nested / "fresh.txt"
        fresh.write_text("new")
        removed = sweep(str(tmp_path))
        assert set(removed) == {str(stale)}
        assert nested.is_dir() and fresh.exists()

    def test_removes_preexisting_empty_dir(self, tmp_path: Path) -> None:
        """An empty folder is removed too."""
        empty = tmp_path / "empty"
        empty.mkdir()
        assert sweep(str(tmp_path)) == [str(empty)]
        assert not empty.exists()

    def test_missing_root(self, tmp_path: Path) -> None:
        """A missing root removes nothing."""
        assert sweep(str(tmp_path / "nope")) == []
