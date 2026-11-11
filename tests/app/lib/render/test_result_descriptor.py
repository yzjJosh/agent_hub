import asyncio
import io
import json
import os
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from app.lib.render.classify import RenderKind
from app.lib.render import cache as _cache
from app.lib.render import result_descriptor as rd
from app.lib.render.detect import Detection, detect
from tests.app.conftest import point_share_env, xlsx_body

PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 40
MARKDOWN_BODY = b"# Notes\n\nSome **bold** text and a [link](https://example.com).\n\n- one\n- two\n"
# A pause between a cold build and the warm read that follows it.
_WARM_DELAY = 0.01


def _seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: bytes, task_id: str | None = None) -> tuple[str, str]:
    point_share_env(tmp_path, monkeypatch)
    task_id = task_id or str(uuid.uuid4())
    run_dir = Path(tmp_path, "data", "runs", task_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result").write_bytes(body)
    return task_id, str(run_dir)


async def test_markdown_text_cold_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, b"# Title\n\nbody text\n")
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["kind"] == "markdown" and d["detected_kind"] == "markdown"
    assert d["kind_label"] == "markdown" and d["mode"] == "doc"
    assert '<div class="md-doc">' in d["fragment"]
    assert d["name"] == tid and d["size"] == len(b"# Title\n\nbody text\n")
    assert isinstance(d["created_at"], str) and d["created_at"]


async def test_code_materializes_language_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"import os\n\ndef main():\n    return 1\n"
    tid, run_dir = _seed(tmp_path, monkeypatch, body)
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["kind_label"] == "code · python" and d["language"] == "python" and d["mode"] == "code"
    assert 'class="code-block"' in d["fragment"]
    mtime = int(os.stat(os.path.join(run_dir, "result")).st_mtime)
    key = _cache.key_for(f"runs/{tid}/result", mtime)
    assert os.path.isfile(os.path.join(_cache.render_root(), key, "input.py"))


async def test_code_sh_language_matches_badge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The language stays sh, it is not renamed to bash.
    body = b"#!/bin/sh\necho hi\n"
    tid, run_dir = _seed(tmp_path, monkeypatch, body)
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["kind_label"] == "code · sh" and d["language"] == "sh" and d["mode"] == "code"
    mtime = int(os.stat(os.path.join(run_dir, "result")).st_mtime)
    key = _cache.key_for(f"runs/{tid}/result", mtime)
    assert os.path.isfile(os.path.join(_cache.render_root(), key, "input.sh"))


async def test_override_code_on_markdown_renders_bare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A code override on markdown text renders as plain code with no language.
    tid, run_dir = _seed(tmp_path, monkeypatch, MARKDOWN_BODY)
    d = await rd.build_result_descriptor(tid, run_dir, kind=RenderKind.CODE)
    assert d["kind"] == "code" and d["detected_kind"] == "markdown"
    assert d["kind_label"] == "code" and d["language"] is None and d["mode"] == "code"
    assert 'class="code-block"' in d["fragment"]


async def test_warm_cache_reuses_published_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, b"# Title\n\nbody\n")
    d1 = await rd.build_result_descriptor(tid, run_dir)
    await asyncio.sleep(_WARM_DELAY)
    d2 = await rd.build_result_descriptor(tid, run_dir)
    assert d1["created_at"] == d2["created_at"]   # the saved entry is reused
    assert d1["fragment"] == d2["fragment"]


async def test_binary_body_is_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, PNG)
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["mode"] == "legacy"
    assert d["name"] == "result.bin" and d["kind_label"] == "file"
    assert d["url"] is None and d["fragment"] is None and d["size"] == len(PNG)


async def test_zero_byte_short_circuit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, b"")
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["kind"] == "unknown" and d["kind_label"] == "file"
    assert d["mode"] == "unsupported" and d["fragment"] is None and d["size"] == 0 and d["name"] == tid


async def test_archive_body_is_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("notes.txt", "hi")
    tid, run_dir = _seed(tmp_path, monkeypatch, buf.getvalue())
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["mode"] == "legacy"
    assert d["name"] == "result.bin" and d["kind_label"] == "file"
    assert d["url"] is None and d["fragment"] is None and d["size"] == len(buf.getvalue())


async def test_nul_body_is_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, b"\x00\x01\x02\x03\x04\x05" * 100)
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["mode"] == "legacy"
    assert d["name"] == "result.bin" and d["kind_label"] == "file"
    assert d["url"] is None and d["fragment"] is None


async def test_office_body_is_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An office body is legacy, it is no longer converted.
    assert (await detect(xlsx_body())).kind is RenderKind.XLSX
    tid, run_dir = _seed(tmp_path, monkeypatch, xlsx_body())
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["mode"] == "legacy"
    assert d["name"] == "result.bin" and d["kind_label"] == "file"
    assert d["url"] is None and d["fragment"] is None


async def test_decodable_pdf_prefix_renders_as_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A post-migration text body opening with a decodable magic prefix is not legacy.
    body = b"%PDF-1.4 is the header of a PDF file, not binary bytes.\n"
    tid, run_dir = _seed(tmp_path, monkeypatch, body)
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["mode"] in ("doc", "code", "raw", "plain")      # a text-family render, never legacy/media
    assert d["kind"] != "pdf"


async def test_html_body_renders_as_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, b"<!doctype html>\n<html><body>hi</body></html>")
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["kind"] == "html" and d["mode"] == "code"
    assert 'class="code-block"' in d["fragment"]
    assert d["url"] is None


async def test_rendered_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = ("# h\n\n" + "word " * 100 + "\n") * 20000      # ~1.6 MB of markdown
    tid, run_dir = _seed(tmp_path, monkeypatch, body.encode())
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["mode"] == "raw"
    assert d["note"] == "Too large to render — showing first 256 KB. Switch to Raw or Download for the full payload."
    assert d["fragment"].startswith('<pre class="raw-pre">')


async def test_override_parse_failure_degradation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, b"plain prose, no commas\nanother line\n")
    d = await rd.build_result_descriptor(tid, run_dir, kind=RenderKind.CSV)
    assert d["kind"] == "csv" and d["detected_kind"] == "text"
    assert d["mode"] == "raw"
    assert d["note"] == "Parse error — the body is not valid CSV. Showing raw text. Use Raw or Download for the full payload."


async def test_kind_ignored_over_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A kind override is ignored over a legacy body.
    tid, run_dir = _seed(tmp_path, monkeypatch, PNG)
    d = await rd.build_result_descriptor(tid, run_dir, kind=RenderKind.JSON)
    assert d["mode"] == "legacy"
    assert d["name"] == "result.bin" and d["kind_label"] == "file"
    assert d["url"] is None and d["fragment"] is None


async def test_override_never_poisons_auto_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, MARKDOWN_BODY)
    auto1 = await rd.build_result_descriptor(tid, run_dir)
    await rd.build_result_descriptor(tid, run_dir, kind=RenderKind.CSV)   # the parse fails, so nothing is published
    auto2 = await rd.build_result_descriptor(tid, run_dir)
    assert auto2["kind_label"] == auto1["kind_label"] == "markdown"
    assert auto2["fragment"] == auto1["fragment"]


_WINDOW_LINES = 1000
_LONG_BODY_LINES = 1500
_LONGER_BODY_LINES = 2500


def _sh_body(total_lines: int) -> bytes:
    return b"#!/bin/sh\n" + b"".join(f"echo {i}\n".encode() for i in range(total_lines - 1))


async def test_code_first_window_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sh_body(_LONG_BODY_LINES)
    tid, run_dir = _seed(tmp_path, monkeypatch, body)
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["mode"] == "code" and d["line_offset"] == 0
    assert d["line_count"] == body.count(b"\n")
    assert d["truncated"] is True                                   # the body is longer than the window
    assert d["fragment"].startswith("<style>")                       # the first window carries the wrapper
    assert d["fragment"].count('<div class="cl">') == _WINDOW_LINES
    assert d["fragment"].endswith("</pre>")


async def test_code_followup_window_slice_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sh_body(_LONGER_BODY_LINES)
    tid, run_dir = _seed(tmp_path, monkeypatch, body)
    d = await rd.build_result_descriptor(tid, run_dir, line_offset=_WINDOW_LINES)
    assert d["line_offset"] == _WINDOW_LINES and d["truncated"] is True
    assert d["fragment"].count('<div class="cl">') == _WINDOW_LINES
    assert "<style>" not in d["fragment"] and "<pre" not in d["fragment"]   # later windows carry only lines


async def test_code_last_window_not_truncated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sh_body(_LONG_BODY_LINES)
    tid, run_dir = _seed(tmp_path, monkeypatch, body)
    d = await rd.build_result_descriptor(tid, run_dir, line_offset=_WINDOW_LINES)
    assert d["line_offset"] == _WINDOW_LINES and d["truncated"] is False
    assert d["fragment"].count('<div class="cl">') == _LONG_BODY_LINES - _WINDOW_LINES


async def test_full_page_no_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sh_body(_LONG_BODY_LINES)
    tid, run_dir = _seed(tmp_path, monkeypatch, body)
    d = await rd.build_result_descriptor(tid, run_dir, full=True)
    assert d["line_offset"] is None and d["truncated"] is None
    assert d["fragment"].startswith("<style>")
    assert d["fragment"].count('<div class="cl">') == _LONG_BODY_LINES     # every line, with the wrapper


async def test_markdown_not_windowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        "".join(
            f"# heading {i}\n\nSome **bold** words and a [link](https://example.com).\n"
            for i in range(_LONG_BODY_LINES)
        )
    ).encode()
    tid, run_dir = _seed(tmp_path, monkeypatch, body)
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["mode"] == "doc" and d["line_offset"] is None and d["truncated"] is None
    assert d["fragment"].count("<h1") == _LONG_BODY_LINES


async def test_declared_kind_beats_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, MARKDOWN_BODY)
    d = await rd.build_result_descriptor(tid, run_dir, declared_kind=RenderKind.CODE)
    assert d["kind"] == "code"
    assert d["detected_kind"] == "markdown"
    assert d["declared_kind"] == "code"
    assert d["mode"] == "code"
    assert 'class="code-block"' in d["fragment"]


async def test_declared_kind_beats_explicit_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, MARKDOWN_BODY)
    d = await rd.build_result_descriptor(tid, run_dir, kind=RenderKind.JSON,
                                          declared_kind=RenderKind.CODE)
    assert d["kind"] == "code"
    assert d["declared_kind"] == "code"


async def test_declared_kind_caches_like_auto(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, MARKDOWN_BODY)
    d1 = await rd.build_result_descriptor(tid, run_dir, declared_kind=RenderKind.CODE)
    await asyncio.sleep(_WARM_DELAY)
    d2 = await rd.build_result_descriptor(tid, run_dir, declared_kind=RenderKind.CODE)
    assert d1["created_at"] == d2["created_at"]
    assert d1["fragment"] == d2["fragment"]


async def test_declared_kind_absent_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, MARKDOWN_BODY)
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["declared_kind"] is None


async def test_declared_kind_ignored_over_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, PNG)
    d = await rd.build_result_descriptor(tid, run_dir, declared_kind=RenderKind.JSON)
    assert d["mode"] == "legacy"
    assert d["declared_kind"] is None


_REPR_BODY = b"{'a': 1, 'b': [2, 3]}"
_PARSE_NOTE = "Parse error — the body is not valid JSON. Showing raw text. Use Raw or Download for the full payload."


async def test_non_json_body_stays_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, _REPR_BODY)
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["kind"] == "json" and d["mode"] == "raw"
    assert d["note"] == _PARSE_NOTE
    assert d["fragment"].startswith('<pre class="raw-pre">')


async def test_parse_failure_note_survives_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tid, run_dir = _seed(tmp_path, monkeypatch, _REPR_BODY)
    d1 = await rd.build_result_descriptor(tid, run_dir)
    assert d1["mode"] == "raw" and d1["note"] == _PARSE_NOTE
    await asyncio.sleep(_WARM_DELAY)
    d2 = await rd.build_result_descriptor(tid, run_dir)
    assert d2["fragment"] == d1["fragment"]
    assert d2["note"] == d1["note"]


_JSON_ARRAY_BODY = json.dumps([
    {"id": "task_001", "description": "Search the web for 'capital of France' to confirm the official capital city. Use a general web search (e.g., query: 'What is the capital of France?') and note the result from authoritative sources such as government websites, encyclopedias, or major news outlets."},
    {"id": "task_002", "summary": "Summarize the findings into a short paragraph that a reviewer can read without opening any of the source pages, keeping every claim tied to the named source."},
]).encode()
_PRETTY_JSON_BODY = json.dumps({"note": "A long sentence of natural prose that carries no structural meaning at all."},
                                indent=2).encode()
_JSONL_BODY = b'{"a": 1}\n{"b": 2}\n'
_BROKEN_JSONL_BODY = b'{"a": 1}\nnot json\n'
_JSONL_PARSE_NOTE = "Parse error — the body is not valid JSONL. Showing raw text. Use Raw or Download for the full payload."


async def _no_model(_data: bytes) -> Detection:
    raise AssertionError("model detection ran on a body that parses as JSON")


def _model_as(kind: RenderKind) -> Callable[[bytes], Detection]:
    async def _model(_data: bytes) -> Detection:
        return Detection(kind)
    return _model


async def test_single_line_json_array_renders_as_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rd, "detect", _no_model)
    tid, run_dir = _seed(tmp_path, monkeypatch, _JSON_ARRAY_BODY)
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["kind"] == "json" and d["detected_kind"] == "json"
    assert d["mode"] == "code" and d["note"] is None
    assert 'class="code-block"' in d["fragment"] and "raw-pre" not in d["fragment"]


async def test_pretty_json_renders_as_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rd, "detect", _no_model)
    tid, run_dir = _seed(tmp_path, monkeypatch, _PRETTY_JSON_BODY)
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["kind"] == "json" and d["detected_kind"] == "json"
    assert d["mode"] == "code" and d["note"] is None
    assert "raw-pre" not in d["fragment"]


async def test_jsonl_body_renders_as_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Two lines, each a document, but the body is not one JSON document, so no model call is needed.
    monkeypatch.setattr(rd, "detect", _no_model)
    tid, run_dir = _seed(tmp_path, monkeypatch, _JSONL_BODY)
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["kind"] == "jsonl" and d["detected_kind"] == "jsonl"
    assert d["mode"] == "code" and d["note"] is None
    assert 'class="code-block"' in d["fragment"]


async def test_broken_jsonl_falls_back_to_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A line that is not JSON keeps the body out of the JSONL tier, so the model decides.
    monkeypatch.setattr(rd, "detect", _model_as(RenderKind.JSONL))
    tid, run_dir = _seed(tmp_path, monkeypatch, _BROKEN_JSONL_BODY)
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["kind"] == "jsonl" and d["detected_kind"] == "jsonl"
    assert d["mode"] == "raw" and d["note"] == _JSONL_PARSE_NOTE
    assert d["fragment"].startswith('<pre class="raw-pre">')


async def test_text_body_falls_back_to_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rd, "detect", _model_as(RenderKind.MARKDOWN))
    tid, run_dir = _seed(tmp_path, monkeypatch, b"# A title\n\nSome plain text.\n")
    d = await rd.build_result_descriptor(tid, run_dir)
    assert d["kind"] == "markdown" and d["detected_kind"] == "markdown"
    assert d["mode"] == "doc"
