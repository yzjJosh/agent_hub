"""Tests for the render cache."""

import asyncio
import hashlib
import io
import os
import shutil
import time
import uuid as _uuid
from pathlib import Path

import pytest

import app.lib.render.cache as _cache

from app.lib.render.cache import (
    CachedDocument,
    build_cache_key,
    build_thumbnail_url,
    clear_failed,
    get_convert_failure_reason,
    get_render_root,
    get_thumbnail_status,
    invalidate_sync,
    mark_failed,
    read_doc,
    sweep_sync,
    write_doc,
)
from tests.app.conftest import configure_resource_env

_FIXED_MTIME = 1700000000


def _marker_path(key: str, kind: str) -> str:
    """Build the path of a failure mark the cache writes."""
    return os.path.join(get_render_root(), ".state", f"{key}.{kind}-failed")


@pytest.fixture
def _render_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give the test its own data dir."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def test_key_for_is_deterministic_sha256(_render_env: Path) -> None:
    key = build_cache_key("upload/a.md", _FIXED_MTIME)
    assert len(key) == 64
    assert key == build_cache_key("upload/a.md", _FIXED_MTIME)
    assert key != build_cache_key("upload/a.md", _FIXED_MTIME + 1)
    assert key != build_cache_key("upload/b.md", _FIXED_MTIME)


def _unversioned_key(path: str, mtime: int) -> str:
    """Build the key without the render format version."""
    return hashlib.sha256(f"{path}\0{mtime}".encode("utf-8")).hexdigest()


def test_key_for_embeds_the_key_version(_render_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The key embeds the render format version."""
    path, mtime = "upload/a.md", _FIXED_MTIME
    key = build_cache_key(path, mtime)
    assert key != _unversioned_key(path, mtime)
    monkeypatch.setattr(_cache, "_KEY_VERSION", 77)
    expected = hashlib.sha256(f"77\0{path}\0{mtime}".encode("utf-8")).hexdigest()
    assert build_cache_key(path, mtime) == expected


async def test_stale_entry_under_unversioned_key_not_served(_render_env: Path) -> None:
    """A render saved under an unversioned key is not served by a fresh lookup."""
    stale_key = _unversioned_key("upload/a.md", _FIXED_MTIME)
    meta = {"path": "upload/a.md", "mtime": _FIXED_MTIME, "mode": "code"}
    await write_doc(stale_key, "<pre class=\"vp-well-code\">stale</pre>", None, meta)
    fresh = build_cache_key("upload/a.md", _FIXED_MTIME)
    assert fresh != stale_key
    assert await read_doc(fresh) is None
    assert await read_doc(stale_key) == CachedDocument("<pre class=\"vp-well-code\">stale</pre>", meta)


async def test_write_read_doc_round_trip(_render_env: Path) -> None:
    key = build_cache_key("a", 1)
    meta = {"path": "a", "mtime": 1, "size": 3, "kind": "markdown", "engine": "markdown-it",
            "created_at": "x", "language": None, "line_count": 2, "row_count": None,
            "column_count": None, "provenance": None, "mode": "doc"}
    await write_doc(key, "<div class=\"md-doc\">hi</div>", {"img.png": b"\x89PNG"}, meta)
    assert get_render_root() == os.path.join(str(_render_env), "render")
    assert await read_doc(key) == CachedDocument("<div class=\"md-doc\">hi</div>", meta)
    with open(os.path.join(get_render_root(), key, "img.png"), "rb") as fh:
        assert fh.read() == b"\x89PNG"


async def test_read_doc_missing_returns_none(_render_env: Path) -> None:
    assert await read_doc("0" * 64) is None


async def test_write_doc_is_atomic(_render_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure while writing leaves no half written entry behind."""
    key = build_cache_key("a", 1)
    real_open = open

    def boom(path: str, *args: str, **kwargs: str) -> io.TextIOWrapper | io.BufferedReader:
        if str(path).endswith("meta.json"):
            raise OSError("disk full")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", boom)
    with pytest.raises(OSError):
        await write_doc(key, "x", None, {})
    assert not os.path.exists(os.path.join(get_render_root(), key))
    leftovers = [n for n in os.listdir(get_render_root()) if n.startswith(".staging-")]
    assert leftovers == []


async def test_marker_lifecycle(_render_env: Path) -> None:
    key = "0" * 64
    assert await get_convert_failure_reason(key) is None
    await mark_failed(key, "convert", "Conversion cancelled")
    assert await get_convert_failure_reason(key) == "Conversion cancelled"
    assert not os.path.exists(_marker_path(key, "thumb"))
    await clear_failed(key, "convert")
    assert await get_convert_failure_reason(key) is None
    await mark_failed(key, "thumb", "engine failed")
    assert os.path.exists(_marker_path(key, "thumb"))
    await clear_failed(key, "thumb")
    await clear_failed(key, "thumb")


async def test_thumb_status_order(_render_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Thumb status picks done first, then generating, then failed, then pending."""
    gen_state_module = pytest.importorskip("app.lib.render.gen_state")
    key = build_cache_key("a", 1)
    assert await get_thumbnail_status(key) == "pending"
    await mark_failed(key, "thumb", "x")
    assert await get_thumbnail_status(key) == "failed"
    await clear_failed(key, "thumb")
    monkeypatch.setattr(gen_state_module, "generating_keys", lambda: {key})
    assert await get_thumbnail_status(key) == "generating"
    monkeypatch.setattr(gen_state_module, "generating_keys", lambda: set())
    os.makedirs(os.path.join(get_render_root(), key), exist_ok=True)
    with open(os.path.join(get_render_root(), key, "thumb.png"), "wb") as fh:
        fh.write(b"\x89PNG")
    assert await get_thumbnail_status(key) == "done"
    monkeypatch.setattr(gen_state_module, "generating_keys", lambda: {key})
    assert await get_thumbnail_status(key) == "done"


def test_thumb_url_is_path_based(_render_env: Path) -> None:
    assert build_thumbnail_url("upload/a b.png") == "/api/resources/thumb?path=upload/a%20b.png"


def _touch_files_oldest(root: str, ts: float) -> None:
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            os.utime(os.path.join(dirpath, name), (ts, ts))


@pytest.fixture
def _resource_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Give the test its own resources dir and data dir."""
    return configure_resource_env(tmp_path, monkeypatch)


def _write_source(resources: Path, rel: str, mtime: int) -> None:
    """Make a file in the resources folder with the given mtime."""
    src = resources / rel
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("x", encoding="utf-8")
    os.utime(src, (mtime, mtime))


async def _publish(resources: Path, rel: str, mtime: int) -> str:
    """Save a render entry for a resource file and give back its key."""
    _write_source(resources, rel, mtime)
    key = build_cache_key(rel, mtime)
    await write_doc(key, "doc", None, {"path": rel, "mtime": mtime})
    return key


def _corrupt_meta(key: str) -> None:
    """Make the entry's saved info unreadable."""
    with open(os.path.join(get_render_root(), key, "meta.json"), "w", encoding="utf-8") as fh:
        fh.write("{bad")


async def test_sweep_removes_orphan_entry_when_source_deleted(_resource_data: tuple[Path, Path]) -> None:
    """An entry is removed when its source file is deleted."""
    resources, _data = _resource_data
    key = await _publish(resources, "upload/a.md", _FIXED_MTIME)
    (resources / "upload" / "a.md").unlink()
    removed = sweep_sync(30)
    assert removed == [os.path.join(get_render_root(), key)]
    assert await read_doc(key) is None


async def test_sweep_removes_orphan_entry_when_source_edited(_resource_data: tuple[Path, Path]) -> None:
    """An entry is removed when its source file changes."""
    resources, _data = _resource_data
    key = await _publish(resources, "upload/a.md", _FIXED_MTIME)
    os.utime(resources / "upload" / "a.md", (_FIXED_MTIME + 1, _FIXED_MTIME + 1))
    removed = sweep_sync(30)
    assert removed == [os.path.join(get_render_root(), key)]
    assert await read_doc(key) is None


async def test_sweep_keeps_entry_with_live_source(_resource_data: tuple[Path, Path]) -> None:
    resources, _data = _resource_data
    key = await _publish(resources, "upload/a.md", _FIXED_MTIME)
    assert sweep_sync(30) == []
    assert await read_doc(key) == CachedDocument("doc", {"path": "upload/a.md", "mtime": _FIXED_MTIME})


async def test_sweep_keeps_entry_with_missing_meta(_resource_data: tuple[Path, Path]) -> None:
    """An entry with no saved info is kept even when the source file is gone."""
    resources, _data = _resource_data
    key = await _publish(resources, "upload/a.md", _FIXED_MTIME)
    (resources / "upload" / "a.md").unlink()
    os.remove(os.path.join(get_render_root(), key, "meta.json"))
    assert sweep_sync(30) == []
    assert os.path.isdir(os.path.join(get_render_root(), key))


async def test_sweep_keeps_entry_with_corrupt_meta(_resource_data: tuple[Path, Path]) -> None:
    """An entry with unreadable saved info is kept even when the source file is gone."""
    resources, _data = _resource_data
    key = await _publish(resources, "upload/a.md", _FIXED_MTIME)
    (resources / "upload" / "a.md").unlink()
    _corrupt_meta(key)
    assert sweep_sync(30) == []
    assert os.path.isdir(os.path.join(get_render_root(), key))


async def test_sweep_never_touches_staging_and_state(_resource_data: tuple[Path, Path]) -> None:
    resources, _data = _resource_data
    key = await _publish(resources, "upload/a.md", _FIXED_MTIME)
    (resources / "upload" / "a.md").unlink()
    root = get_render_root()
    staging = os.path.join(root, f".staging-{key}-{os.getpid()}-1")
    os.makedirs(staging)
    with open(os.path.join(staging, "doc.html"), "w", encoding="utf-8") as fh:
        fh.write("in progress")
    orphan = "1" * 64
    await mark_failed(orphan, "convert", "engine failed")
    removed = sweep_sync(30)
    assert removed == [os.path.join(root, key)]
    assert os.path.isdir(staging)
    assert os.path.isfile(_marker_path(orphan, "convert"))


async def test_sweep_still_removes_stale_files(_resource_data: tuple[Path, Path]) -> None:
    """An old entry is still removed when its source file is fine."""
    resources, _data = _resource_data
    key = await _publish(resources, "upload/a.md", _FIXED_MTIME)
    _touch_files_oldest(get_render_root(), time.time() - 40 * 86400)
    removed = sweep_sync(30)
    assert any(f"render/{key}" in p for p in removed)
    assert await read_doc(key) is None


async def test_invalidate_removes_entry_and_markers_when_source_deleted(_resource_data: tuple[Path, Path]) -> None:
    """A deleted source removes the entry and both of its failure marks."""
    resources, _data = _resource_data
    key = await _publish(resources, "upload/a.md", _FIXED_MTIME)
    await mark_failed(key, "thumb", "engine failed")
    await mark_failed(key, "convert", "engine failed")
    (resources / "upload" / "a.md").unlink()
    root = get_render_root()
    removed = invalidate_sync("upload/a.md")
    assert removed == [
        os.path.join(root, key),
        _marker_path(key, "thumb"),
        _marker_path(key, "convert"),
    ]
    assert await read_doc(key) is None


async def test_invalidate_removes_entry_when_source_overwritten(_resource_data: tuple[Path, Path]) -> None:
    """A changed source removes the old entry."""
    resources, _data = _resource_data
    key = await _publish(resources, "upload/a.md", _FIXED_MTIME)
    os.utime(resources / "upload" / "a.md", (_FIXED_MTIME + 1, _FIXED_MTIME + 1))
    removed = invalidate_sync("upload/a.md")
    assert removed == [os.path.join(get_render_root(), key)]
    assert await read_doc(key) is None


async def test_invalidate_keeps_entry_with_live_source(_resource_data: tuple[Path, Path]) -> None:
    resources, _data = _resource_data
    key = await _publish(resources, "upload/a.md", _FIXED_MTIME)
    assert invalidate_sync("upload/a.md") == []
    assert await read_doc(key) == CachedDocument("doc", {"path": "upload/a.md", "mtime": _FIXED_MTIME})


async def test_invalidate_skips_entry_with_corrupt_meta(_resource_data: tuple[Path, Path]) -> None:
    """An entry with unreadable saved info is kept."""
    resources, _data = _resource_data
    key = await _publish(resources, "upload/a.md", _FIXED_MTIME)
    (resources / "upload" / "a.md").unlink()
    _corrupt_meta(key)
    assert invalidate_sync("upload/a.md") == []
    assert os.path.isdir(os.path.join(get_render_root(), key))


async def test_invalidate_without_entries_returns_empty(_resource_data: tuple[Path, Path]) -> None:
    """A path with no cached entries gives back an empty list."""
    _resources, _data = _resource_data
    assert invalidate_sync("upload/never-rendered.md") == []


async def test_invalidate_ignores_other_paths(_resource_data: tuple[Path, Path]) -> None:
    """Only the given path and files under it are removed."""
    resources, _data = _resource_data
    key_a = await _publish(resources, "upload/a.md", _FIXED_MTIME)
    key_b = await _publish(resources, "upload/b.md", _FIXED_MTIME)
    key_prefix = await _publish(resources, "upload/a.md.bak", _FIXED_MTIME)
    (resources / "upload" / "a.md").unlink()
    removed = invalidate_sync("upload/a.md")
    assert removed == [os.path.join(get_render_root(), key_a)]
    assert await read_doc(key_b) == CachedDocument("doc", {"path": "upload/b.md", "mtime": _FIXED_MTIME})
    assert await read_doc(key_prefix) == CachedDocument("doc", {"path": "upload/a.md.bak", "mtime": _FIXED_MTIME})


async def test_invalidate_removes_entries_under_deleted_folder(_resource_data: tuple[Path, Path]) -> None:
    """Deleting a folder removes every entry under it, nested ones too."""
    resources, _data = _resource_data
    key_a = await _publish(resources, "upload/x/a.md", _FIXED_MTIME)
    key_b = await _publish(resources, "upload/x/sub/b.md", _FIXED_MTIME)
    shutil.rmtree(resources / "upload" / "x")
    removed = invalidate_sync("upload/x")
    assert sorted(removed) == sorted(
        [os.path.join(get_render_root(), key_a), os.path.join(get_render_root(), key_b)]
    )
    assert await read_doc(key_a) is None
    assert await read_doc(key_b) is None


def test_sweep_sync_runs_without_an_event_loop(_resource_data: tuple[Path, Path]) -> None:
    """The garbage collector command sweeps with no event loop running."""
    resources, _data = _resource_data
    key = asyncio.run(_publish(resources, "upload/a.md", _FIXED_MTIME))
    (resources / "upload" / "a.md").unlink()
    assert sweep_sync(30) == [os.path.join(get_render_root(), key)]


def test_invalidate_sync_runs_without_an_event_loop(_resource_data: tuple[Path, Path]) -> None:
    """The WebDAV worker threads invalidate with no event loop running."""
    resources, _data = _resource_data
    key = asyncio.run(_publish(resources, "upload/a.md", _FIXED_MTIME))
    (resources / "upload" / "a.md").unlink()
    assert invalidate_sync("upload/a.md") == [os.path.join(get_render_root(), key)]


async def _run_key_and_meta(tmp: Path, task_id: str, body: bytes) -> tuple[str, str]:
    run_dir = os.path.join(tmp, "runs", task_id)
    os.makedirs(run_dir, exist_ok=True)
    result = os.path.join(run_dir, "result")
    with open(result, "wb") as fh:
        fh.write(body)
    mtime = int(os.stat(result).st_mtime)
    key = _cache.build_cache_key(f"runs/{task_id}/result", mtime)
    meta = {"path": f"runs/{task_id}/result", "mtime": mtime, "kind": "text"}
    await _cache.write_doc(key, "<div>x</div>", None, meta)
    return key, result


async def test_run_entry_survives_sweep_while_run_alive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RESOURCES_DIR", str(tmp_path / "resources"))
    task_id = str(_uuid.uuid4())
    key, result = await _run_key_and_meta(tmp_path, task_id, b"hello")
    removed = _cache.sweep_sync(30)
    assert os.path.isdir(os.path.join(_cache.get_render_root(), key))
    assert result not in removed


async def test_run_entry_swept_when_run_dir_gone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RESOURCES_DIR", str(tmp_path / "resources"))
    task_id = str(_uuid.uuid4())
    key, _ = await _run_key_and_meta(tmp_path, task_id, b"hello")
    shutil.rmtree(os.path.join(tmp_path, "runs", task_id))
    _cache.sweep_sync(30)
    assert not os.path.exists(os.path.join(_cache.get_render_root(), key))


async def test_invalidate_resource_path_noop_for_run_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RESOURCES_DIR", str(tmp_path / "resources"))
    task_id = str(_uuid.uuid4())
    key, _ = await _run_key_and_meta(tmp_path, task_id, b"hello")
    removed = _cache.invalidate_sync("some/resource/file.txt")
    assert removed == []
    assert os.path.isdir(os.path.join(_cache.get_render_root(), key))


async def test_run_key_matches_key_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    task_id = str(_uuid.uuid4())
    key, result = await _run_key_and_meta(tmp_path, task_id, b"hello")
    assert key == _cache.build_cache_key(f"runs/{task_id}/result", int(os.stat(result).st_mtime))
