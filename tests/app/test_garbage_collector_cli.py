import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import app.garbage_collector_cli as gc_cli
from app.lib.render.cache import key_for

_SCHEMA = """
CREATE TABLE tool_records (
  task_id TEXT PRIMARY KEY, tool_name TEXT NOT NULL, cache_key TEXT,
  params TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
  started_at TEXT NOT NULL, ended_at TEXT, run_dir TEXT NOT NULL
);
"""

_CLI_MODULE = "app.garbage_collector_cli"


def _make_cli_env(tmp_path: Path, resources_dir: Path) -> dict:
    env = dict(os.environ)
    env["DATA_DIR"] = str(tmp_path)
    env["RESOURCES_DIR"] = str(resources_dir)
    return env


def _run_gc_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    data: Path,
    resources: Path,
    *flags: str,
) -> tuple[str, str]:
    """Run the garbage collector and give back what it printed."""
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("RESOURCES_DIR", str(resources))
    monkeypatch.setattr(sys, "argv", ["garbage_collector_cli", *flags])
    gc_cli.main()
    captured = capsys.readouterr()
    return captured.out, captured.err


def _seed_render_entry(data: Path, resources: Path, rel: str, content: str) -> Path:
    """Make a share file plus the render folder for it, and give back that folder."""
    src = resources / rel
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(content, encoding="utf-8")
    mtime = int(time.time())
    os.utime(src, (mtime, mtime))
    key_dir = data / "render" / key_for(rel, mtime)
    key_dir.mkdir(parents=True)
    (key_dir / "doc.html").write_text(content, encoding="utf-8")
    (key_dir / "meta.json").write_text(json.dumps({"path": rel, "mtime": mtime}), encoding="utf-8")
    return key_dir


class TestGcCli:
    def test_runs_category_removes_stale(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=31)).isoformat()
        conn = sqlite3.connect(tmp_path / "tool.db")
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT INTO tool_records VALUES ('t1','tool',NULL,'{}','completed',NULL,?,?,'tool_runs/t1')",
            (now.isoformat(), old),
        )
        conn.commit()
        conn.close()
        run_dir = tmp_path / "runs" / "t1"
        run_dir.mkdir(parents=True)
        (run_dir / "log").write_text("x")

        out, err = _run_gc_cli(
            monkeypatch, capsys, tmp_path, tmp_path / "resources",
            "--category", "runs", "--days", "30",
        )
        assert err == ""
        assert str(run_dir) in out.splitlines()
        assert not run_dir.exists()

    def test_single_category_nothing_to_remove(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out, err = _run_gc_cli(
            monkeypatch, capsys, tmp_path, tmp_path / "resources",
            "--category", "runs", "--days", "30",
        )
        assert err == ""
        assert out.strip() == "(nothing to remove)"

    def test_resources_category_still_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        resources = tmp_path / "resources"
        resources.mkdir()
        stale = resources / "stale.txt"
        stale.write_text("x")
        old = time.time() - 40 * 86400
        os.utime(stale, (old, old))

        out, err = _run_gc_cli(
            monkeypatch, capsys, tmp_path, resources,
            "--category", "resources", "--days", "30",
        )
        assert err == ""
        assert str(stale) in out.splitlines()

    def test_all_categories_grouped_headers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out, err = _run_gc_cli(monkeypatch, capsys, tmp_path, tmp_path / "resources")
        assert err == ""
        lines = out.splitlines()
        assert "[resources]" in lines
        assert "[runs]" in lines
        assert "[render]" in lines
        assert "[presentations]" in lines

    def test_render_category_removes_orphan_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The render category drops the folder for a file that is gone and keeps the rest."""
        data, resources = tmp_path, tmp_path / "resources"
        gone_dir = _seed_render_entry(data, resources, "upload/gone.md", "x")
        (resources / "upload" / "gone.md").unlink()
        live_dir = _seed_render_entry(data, resources, "upload/live.md", "y")

        out, err = _run_gc_cli(monkeypatch, capsys, data, resources, "--category", "render")
        assert err == ""
        lines = out.splitlines()
        assert str(gone_dir) in lines
        assert str(live_dir) not in lines
        assert not gone_dir.exists()
        assert live_dir.is_dir()

    @pytest.mark.parametrize("category", ["render", "presentations"])
    def test_categories_include_render_and_presentations(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        category: str,
    ) -> None:
        out, err = _run_gc_cli(
            monkeypatch, capsys, tmp_path, tmp_path / "resources",
            "--category", category,
        )
        assert err == ""
        assert out.strip() == "(nothing to remove)"

    def test_module_entry_point_runs(self, tmp_path: Path) -> None:
        """The module entry point runs the collector."""
        completed = subprocess.run(
            [sys.executable, "-m", _CLI_MODULE, "--category", "runs", "--days", "30"],
            capture_output=True,
            text=True,
            env=_make_cli_env(tmp_path, tmp_path / "resources"),
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "(nothing to remove)"
