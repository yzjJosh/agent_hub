"""Tests for the console preview seed data.

The seed fills a throwaway data dir and share dir, and the tests read the
records and the files back.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.console_seed import main

# Counts per tool, in the order running, completed, failed, cancelled.
_EXPECTED: dict[str, tuple[int, int, int, int]] = {
    "generate_image": (1, 9, 4, 1),
    "redfin_search": (1, 5, 2, 0),
    "rednote_research": (1, 2, 1, 0),
    "brave_search": (0, 46, 1, 0),
    "github": (0, 19, 2, 0),
    "google_flights": (0, 3, 0, 0),
    "google_search": (0, 0, 0, 0),
    "present_to_user": (0, 8, 0, 0),
    "read_media_or_doc": (0, 28, 6, 1),
    "send_or_receive_file": (0, 0, 0, 0),
    "task": (0, 0, 0, 0),
    "tts": (0, 12, 0, 0),
    "web_fetch": (0, 50, 5, 0),
    "x_search": (0, 6, 1, 0),
}
_EXPECTED_TOTAL = sum(sum(counts) for counts in _EXPECTED.values())
_ZERO_TOOLS = ("google_search", "send_or_receive_file", "task")

_RUNNING_LOG_LINES = 238
_FAILED_LOG_LINES = 250
_RESULT_BYTES_0797 = 793191
_DOCX_BYTES = 36864
_BULK_DEBUG_FLAKE = 108
_BULK_MEDIA_GEN = 78
_BULK_UPLOAD = 23

_ID_C4A7 = "c4a7e92f-8b34-4d2a-b7c1-5e9f0a6d3c18"
_ID_0797 = "0797d2e6-3c41-4f8a-9b27-6d8e1a0f5c42"
_ID_F3A1 = "f3a1c9d2-5e08-4c77-a1b3-9d24f8e6c051"
_ID_9C1D = "9c1d4e07-b2f6-4a8d-9e31-7a5c0d8f2b64"
_ID_MEDIA_RESULT = "media-result-1"


@pytest.fixture
def seed_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Run the seed against throwaway data and share dirs, and give back both."""
    data_dir = tmp_path / "data"
    resources_dir = tmp_path / "resources"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("RESOURCES_DIR", str(resources_dir))
    asyncio.run(main())
    return data_dir, resources_dir


def test_seed_records_match_the_41_table(seed_dirs: tuple[Path, Path]) -> None:
    data_dir, _ = seed_dirs
    db = sqlite3.connect(data_dir / "records.db")
    assert db.execute("SELECT COUNT(*) FROM task_records").fetchone()[0] == _EXPECTED_TOTAL
    rows = db.execute(
        "SELECT tr.tool_name, SUM(t.status = 'running'), SUM(t.status = 'completed'),"
        " SUM(t.status = 'failed'), SUM(t.status = 'cancelled')"
        " FROM tool_records tr JOIN task_records t ON t.task_id = tr.task_id"
        " GROUP BY tr.tool_name"
    ).fetchall()
    got = {tool: (running, completed, failed, cancelled) for tool, running, completed, failed, cancelled in rows}
    got.update({tool: (0, 0, 0, 0) for tool in _ZERO_TOOLS})
    assert got == _EXPECTED
    running_ids = [
        row[0]
        for row in db.execute(
            "SELECT t.task_id FROM task_records t JOIN tool_records tr ON tr.task_id = t.task_id"
            " WHERE t.status = 'running' ORDER BY t.task_id"
        )
    ]
    assert running_ids == ["c4a7e92f-8b34-4d2a-b7c1-5e9f0a6d3c18", "media-run-1", "redfin-run-1"]
    db.close()


def test_seed_run_files_and_share(seed_dirs: tuple[Path, Path]) -> None:
    data_dir, resources_dir = seed_dirs
    runs = data_dir / "runs"
    assert (runs / _ID_C4A7 / "log").read_text(encoding="utf-8").count("\n") == _RUNNING_LOG_LINES
    assert (runs / _ID_0797 / "log").stat().st_size == 0
    assert (runs / _ID_0797 / "result").stat().st_size == _RESULT_BYTES_0797
    assert (runs / _ID_F3A1 / "result").read_text(encoding="utf-8").startswith("# 成都火锅")
    media_result = (runs / _ID_MEDIA_RESULT / "result").read_text(encoding="utf-8")
    assert "media_gen/gen_001.png" in media_result
    assert "media_gen/gen_999.png" in media_result
    assert (resources_dir / "media_gen" / "gen_001.png").is_file()
    assert not (resources_dir / "media_gen" / "gen_999.png").exists()
    failed_log = (runs / _ID_9C1D / "log").read_text(encoding="utf-8")
    assert failed_log.count("\n") == _FAILED_LOG_LINES
    assert failed_log.endswith("Error: Research failed: no analysis results gathered.\n")
    assert (resources_dir / "downloads" / "1beebf7c2c7c4d42adf31a28d483affe.docx").stat().st_size == _DOCX_BYTES
    assert len(list((resources_dir / "debug_flake").iterdir())) == _BULK_DEBUG_FLAKE
    assert len(list((resources_dir / "media_gen").iterdir())) == _BULK_MEDIA_GEN
    assert len(list((resources_dir / "upload").iterdir())) == _BULK_UPLOAD


_KIND_MAGICS: dict[str, bytes] = {
    "wf-code-1": b"def add(a, b):",
    "wf-csv-1": b"name,qty",
    "wf-json-1": b'{"name"',
    "wf-html-1": b"<!DOCTYPE html>",
    "wf-image-1": b"\x89PNG",
    "wf-pdf-1": b"%PDF",
    "wf-audio-1": b"RIFF",
    "wf-file-1": b"\x00",
}


@pytest.mark.parametrize(("tid", "magic"), list(_KIND_MAGICS.items()))
def test_seed_result_kind_bodies(seed_dirs: tuple[Path, Path], tid: str, magic: bytes) -> None:
    data_dir, _ = seed_dirs
    assert (data_dir / "runs" / tid / "result").read_bytes().startswith(magic)


def test_seed_media_result_sidecar(seed_dirs: tuple[Path, Path]) -> None:
    data_dir, resources_dir = seed_dirs
    sidecar = (data_dir / "runs" / _ID_MEDIA_RESULT / "files").read_text(encoding="utf-8")
    lines = [json.loads(line) for line in sidecar.splitlines() if line]
    assert [line["path"] for line in lines] == ["media_gen/gen_001.png", "media_gen/gen_999.png"]
    assert lines[0]["size"] == (resources_dir / "media_gen" / "gen_001.png").stat().st_size
    assert lines[0]["mtime"] == int((resources_dir / "media_gen" / "gen_001.png").stat().st_mtime)
    assert lines[1]["size"] and lines[1]["mtime"]


def test_seed_result_kind_bodies_binaries(seed_dirs: tuple[Path, Path]) -> None:
    data_dir, _ = seed_dirs
    runs = data_dir / "runs"
    assert (runs / "wf-video-1" / "result").read_bytes()[4:8] == b"ftyp"   # ftyp sits at offset 4
    assert b"xl/" in (runs / "wf-xlsx-1" / "result").read_bytes()          # office files hold these inside
    assert b"word/" in (runs / "wf-docx-1" / "result").read_bytes()
    assert b"word/" not in (runs / "wf-zip-1" / "result").read_bytes()     # a non-office archive
    assert (runs / "wf-empty-1" / "result").stat().st_size == 0            # an empty file
