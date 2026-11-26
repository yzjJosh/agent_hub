"""Tests for console resource listing, search, and download endpoints."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport
from starlette.applications import Starlette

from app.console.resources_api import download_resource
from app.lib.render import cache
from app.lib.shared import activity
from tests.app.conftest import write_resource_file

_TOP_FOLDERS = ("github", "media_gen", "tts", "upload")
_FLAKE_FILE_COUNT = 108
_DOWNLOAD_FILE_COUNT = 12


@pytest.fixture
def dav_download_app() -> Starlette:
    """A small app with only the download route."""
    app = Starlette()
    app.add_route("/dav/dl", download_resource, methods=("GET", "HEAD"))
    return app


def _write_text_resource(resources: Path, relative_path: str, content: str) -> Path:
    file_path = resources / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return file_path


def _make_sample_resources(root: Path) -> None:
    (root / "downloads").mkdir(parents=True)
    (root / "downloads" / "cat.jpg").write_bytes(b"x" * 100)
    (root / "uploads").mkdir()
    (root / "uploads" / "a.txt").write_text("hello")
    (root / "readme.txt").write_text("hi")


def _make_cap_resources(root: Path) -> None:
    for index in range(5000):
        (root / f"x{index:05d}").mkdir()
    (root / "a.txt").write_text("x")


def _make_capped_escape_resources(root: Path, outside: Path) -> None:
    for index in range(4999):
        (root / f"y{index:05d}").write_text("x")
    (root / "aa_out").symlink_to(outside)


def _make_top_level_resources(resources: Path) -> None:
    for index in range(_FLAKE_FILE_COUNT):
        write_resource_file(resources, f"debug_flake/flake_{index:03d}.txt", b"flake")
    for index in range(_DOWNLOAD_FILE_COUNT):
        write_resource_file(resources, f"downloads/dl_{index:02d}.png", b"png")
    for folder in _TOP_FOLDERS:
        write_resource_file(resources, f"{folder}/placeholder.txt", b"placeholder")


@pytest.mark.parametrize(
    ("relative_path", "content_disposition"),
    [
        ("upload/x.txt", 'attachment; filename="x.txt"'),
        ("sub dir/my file.txt", "attachment; filename*=utf-8''my%20file.txt"),
    ],
)
async def test_dav_download_serves_file_as_attachment(
    relative_path: str, content_disposition: str, dav_resources_dir: Path, dav_download_app: Starlette
) -> None:
    """A download gives back the file bytes as an attachment named after the file."""
    _write_text_resource(dav_resources_dir, relative_path, "hello download")
    async with httpx.AsyncClient(transport=ASGITransport(app=dav_download_app), base_url="http://t") as client:
        response = await client.get("/dav/dl", params={"path": relative_path})
    assert response.status_code == 200
    assert response.text == "hello download"
    assert response.headers["content-disposition"] == content_disposition


async def test_dav_download_strips_leading_slash(dav_resources_dir: Path, dav_download_app: Starlette) -> None:
    """A leading slash on the path is ignored."""
    _write_text_resource(dav_resources_dir, "upload/x.txt", "hello download")
    async with httpx.AsyncClient(transport=ASGITransport(app=dav_download_app), base_url="http://t") as client:
        response = await client.get("/dav/dl", params={"path": "/upload/x.txt"})
    assert response.status_code == 200
    assert response.text == "hello download"


async def test_dav_download_missing_param_is_400(dav_download_app: Starlette) -> None:
    """A missing or empty path answers 400 bad_param, never a folder listing."""
    async with httpx.AsyncClient(transport=ASGITransport(app=dav_download_app), base_url="http://t") as client:
        absent_response = await client.get("/dav/dl")
        empty_response = await client.get("/dav/dl", params={"path": ""})
    assert absent_response.status_code == 400
    assert absent_response.json()["code"] == "bad_param"
    assert empty_response.status_code == 400
    assert empty_response.json()["code"] == "bad_param"


@pytest.mark.parametrize("path", ["upload/missing.txt", "../../etc/passwd", "..", "a/../../b"])
async def test_dav_download_missing_or_traversal_is_404(
    path: str, dav_resources_dir: Path, dav_download_app: Starlette
) -> None:
    """A missing file and every path that climbs out of the resources folder answer 404."""
    _write_text_resource(dav_resources_dir, "upload/real.txt", "control")
    async with httpx.AsyncClient(transport=ASGITransport(app=dav_download_app), base_url="http://t") as client:
        control = await client.get("/dav/dl", params={"path": "upload/real.txt"})
        assert control.status_code == 200
        response = await client.get("/dav/dl", params={"path": path})
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


@pytest.mark.parametrize(("method", "is_recorded"), [("GET", True), ("HEAD", False)])
async def test_dav_download_records_interest_on_get(
    method: str, is_recorded: bool, dav_resources_dir: Path, dav_download_app: Starlette
) -> None:
    """A download renews retention, a HEAD request does not."""
    file_path = _write_text_resource(dav_resources_dir, "upload/old.txt", "old")
    old_timestamp = time.time() - 40 * 86400
    os.utime(file_path, (old_timestamp, old_timestamp))
    async with httpx.AsyncClient(transport=ASGITransport(app=dav_download_app), base_url="http://t") as client:
        response = await client.request(method, "/dav/dl", params={"path": "upload/old.txt"})
    assert response.status_code == 200
    assert (activity.last_used_sync(str(file_path), file_path.stat()) > old_timestamp) is is_recorded


async def test_resources_root_listing(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = tmp_path / "resources"
    _make_sample_resources(resources)
    monkeypatch.setenv("RESOURCES_DIR", str(resources))

    response = await api_client.get("/api/resources")
    assert response.status_code == 200
    body = response.json()
    assert body["path"] == ""
    assert body["parent"] is None
    assert [entry["name"] for entry in body["entries"]] == ["downloads", "uploads", "readme.txt"]
    directory_entry = body["entries"][0]
    assert directory_entry["type"] == "dir" and directory_entry["path"] == "downloads"
    assert directory_entry["file_count"] == 1 and directory_entry["total_size"] == 100
    assert "size" not in directory_entry
    file_entry = body["entries"][2]
    assert file_entry["type"] == "file" and file_entry["path"] == "readme.txt" and file_entry["size"] == 2
    assert isinstance(file_entry["mtime"], int) and file_entry["mtime"] > 0
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", file_entry["mtime_local"])
    assert file_entry["kind"] == "text"
    assert set(file_entry["thumb"]) == {"status", "url"}
    assert file_entry["thumb"]["status"] == "pending"
    assert file_entry["thumb"]["url"] == "/api/resources/thumb?path=readme.txt"


async def test_resources_subdir_listing(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = tmp_path / "resources"
    _make_sample_resources(resources)
    monkeypatch.setenv("RESOURCES_DIR", str(resources))

    response = await api_client.get("/api/resources", params={"path": "downloads/"})
    body = response.json()
    assert body["path"] == "downloads"
    assert body["parent"] == ""
    entry = body["entries"][0]
    assert set(entry) == {"name", "path", "type", "mtime", "mtime_local", "size", "kind", "thumb"}
    assert entry["name"] == "cat.jpg"
    assert entry["path"] == "downloads/cat.jpg"
    assert entry["type"] == "file"
    assert entry["size"] == 100
    assert entry["kind"] == "image"
    assert entry["thumb"]["status"] == "pending"
    assert entry["thumb"]["url"] == "/api/resources/thumb?path=downloads/cat.jpg"


async def test_resources_dot_segments_echo_normalized(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path with dot parts comes back cleaned up, naming the folder that was listed."""
    resources = tmp_path / "resources"
    _make_sample_resources(resources)
    monkeypatch.setenv("RESOURCES_DIR", str(resources))

    root_response = await api_client.get("/api/resources", params={"path": "downloads/./.."})
    assert root_response.status_code == 200
    root_body = root_response.json()
    assert (root_body["path"], root_body["parent"]) == ("", None)
    assert {entry["name"] for entry in root_body["entries"]} == {"downloads", "uploads", "readme.txt"}

    sub_response = await api_client.get("/api/resources", params={"path": "uploads/../downloads/./"})
    sub_body = sub_response.json()
    assert (sub_body["path"], sub_body["parent"]) == ("downloads", "")
    assert [entry["name"] for entry in sub_body["entries"]] == ["cat.jpg"]


async def test_resources_missing_and_traversal(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    monkeypatch.setenv("RESOURCES_DIR", str(resources))
    missing_response = await api_client.get("/api/resources", params={"path": "nope"})
    assert missing_response.status_code == 404
    assert missing_response.json()["code"] == "not_found"
    traversal_response = await api_client.get("/api/resources", params={"path": "../outside"})
    assert traversal_response.status_code == 400
    assert traversal_response.json()["code"] == "bad_param"


@pytest.mark.parametrize("raw_query", ["path=%00", "path=downloads%00x"])
async def test_resources_nul_path_returns_bad_param(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_query: str
) -> None:
    """A NUL byte in the path answers the JSON 400 bad_param."""
    resources = tmp_path / "resources"
    _make_sample_resources(resources)
    monkeypatch.setenv("RESOURCES_DIR", str(resources))
    response = await api_client.get(f"/api/resources?{raw_query}")
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert set(body) == {"error", "code"}
    assert body["code"] == "bad_param"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"


async def test_resources_listing_skips_vanished_entries(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken link in the resources folder is skipped and the rest still lists."""
    resources = tmp_path / "resources"
    _make_sample_resources(resources)
    (resources / "broken-top").symlink_to("missing-target")
    (resources / "downloads" / "broken-nested").symlink_to("missing-target")
    monkeypatch.setenv("RESOURCES_DIR", str(resources))

    response = await api_client.get("/api/resources")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert [entry["name"] for entry in body["entries"]] == ["downloads", "uploads", "readme.txt"]
    directory_entry = body["entries"][0]
    assert directory_entry["file_count"] == 1 and directory_entry["total_size"] == 100


async def test_resources_listing_skips_symlinks_out_of_resources(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Links that point outside the resources folder are left out of the listing."""
    resources = tmp_path / "resources"
    outside = tmp_path / "outside"
    resources.mkdir()
    (outside / "target-dir").mkdir(parents=True)
    (outside / "target-dir" / "leak.bin").write_bytes(b"z" * 500)
    (outside / "leak-file.bin").write_bytes(b"z" * 500)
    (resources / "keep.txt").write_text("hello")
    (resources / "escape-dir").symlink_to(outside / "target-dir")
    (resources / "escape-file").symlink_to(outside / "leak-file.bin")
    monkeypatch.setenv("RESOURCES_DIR", str(resources))

    response = await api_client.get("/api/resources")
    assert response.status_code == 200
    body = response.json()
    assert [entry["name"] for entry in body["entries"]] == ["keep.txt"]
    assert body["entries"][0]["size"] == 5


async def test_resources_search(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = tmp_path / "resources"
    (resources / "downloads" / "deep").mkdir(parents=True)
    (resources / "downloads" / "deep" / "coffee-notes.txt").write_text("notes")
    (resources / "downloads" / "cat.jpg").write_bytes(b"x" * 50)
    (resources / "README.md").write_text("# hi")
    monkeypatch.setenv("RESOURCES_DIR", str(resources))

    response = await api_client.get("/api/resources/search", params={"q": "CAT"})
    assert response.status_code == 200
    body = response.json()
    assert body["truncated"] is False
    assert [match["path"] for match in body["matches"]] == ["downloads/cat.jpg"]
    match = body["matches"][0]
    assert match["name"] == "cat.jpg"
    assert match["type"] == "file"
    assert match["size"] == 50
    assert match["matched_ancestor_paths"] == ["downloads"]
    assert match["kind"] == "image"
    assert set(match["thumb"]) == {"status", "url"}
    assert match["thumb"]["status"] == "pending"


async def test_resources_search_skips_vanished_entries(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken link that matches the search is skipped and the rest still comes back."""
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "green.txt").write_text("notes")
    (resources / "broken-link").symlink_to("missing-target")
    monkeypatch.setenv("RESOURCES_DIR", str(resources))

    response = await api_client.get("/api/resources/search", params={"q": "e"})
    assert response.status_code == 200
    body = response.json()
    assert body["truncated"] is False
    assert [match["path"] for match in body["matches"]] == ["green.txt"]


async def test_resources_search_matches_dirs_and_descendants(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = tmp_path / "resources"
    (resources / "a" / "b" / "c").mkdir(parents=True)
    (resources / "a" / "b" / "c" / "f.txt").write_text("x")
    monkeypatch.setenv("RESOURCES_DIR", str(resources))

    body = (await api_client.get("/api/resources/search", params={"q": "a/b"})).json()
    assert body["truncated"] is False
    assert [match["path"] for match in body["matches"]] == ["a/b", "a/b/c", "a/b/c/f.txt"]
    by_path = {match["path"]: match for match in body["matches"]}
    assert by_path["a/b"]["type"] == "dir"
    assert "size" not in by_path["a/b"]
    assert by_path["a/b"]["matched_ancestor_paths"] == ["a"]
    assert by_path["a/b/c"]["matched_ancestor_paths"] == ["a", "b"]
    assert by_path["a/b/c/f.txt"]["matched_ancestor_paths"] == ["a", "b", "c"]
    assert by_path["a/b/c/f.txt"]["size"] == 1
    assert "kind" not in by_path["a/b"]
    assert "thumb" not in by_path["a/b"]
    assert by_path["a/b/c/f.txt"]["kind"] == "text"
    assert by_path["a/b/c/f.txt"]["thumb"]["status"] == "pending"


async def test_resources_listing_done_thumb(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file whose thumbnail is already rendered comes back as done."""
    resources = tmp_path / "resources"
    (resources / "media_gen").mkdir(parents=True)
    (resources / "media_gen" / "a.png").write_bytes(b"x" * 100)
    monkeypatch.setenv("RESOURCES_DIR", str(resources))
    modified_time = int((resources / "media_gen" / "a.png").stat().st_mtime)
    thumb_dir = Path(os.environ["DATA_DIR"]) / "render" / cache.build_cache_key("media_gen/a.png", modified_time)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    (thumb_dir / "thumb.png").write_bytes(b"png")

    try:
        body = (await api_client.get("/api/resources", params={"path": "media_gen"})).json()
    finally:
        (thumb_dir / "thumb.png").unlink()
        thumb_dir.rmdir()
    entry = body["entries"][0]
    assert entry["kind"] == "image"
    assert entry["thumb"]["status"] == "done"
    assert entry["thumb"]["url"] == "/api/resources/thumb?path=media_gen/a.png"


async def test_resources_listing_no_engine_thumb(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kind that gets no thumbnail engine comes back as none."""
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "song.mp3").write_bytes(b"x")
    monkeypatch.setenv("RESOURCES_DIR", str(resources))

    body = (await api_client.get("/api/resources")).json()
    entry = body["entries"][0]
    assert entry["kind"] == "audio"
    assert entry["thumb"]["status"] == "none"


async def test_resources_search_limit_truncates(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "f0.log").write_text("x")
    (resources / "f1.log").write_text("x")
    (resources / "f2.log").write_text("x")
    monkeypatch.setenv("RESOURCES_DIR", str(resources))

    body = (await api_client.get("/api/resources/search", params={"q": "f", "limit": 2})).json()
    assert [match["path"] for match in body["matches"]] == ["f0.log", "f1.log"]
    assert body["truncated"] is True


async def test_resources_search_node_cap(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    _make_cap_resources(resources)
    monkeypatch.setenv("RESOURCES_DIR", str(resources))

    body = (await api_client.get("/api/resources/search", params={"q": "a"})).json()
    assert body["truncated"] is True
    assert [match["path"] for match in body["matches"]] == ["a.txt"]


async def test_resources_search_skips_symlinks_out_of_resources(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Links that point outside the resources folder are not matched and not followed."""
    resources = tmp_path / "resources"
    outside = tmp_path / "outside"
    resources.mkdir()
    outside.mkdir()
    (resources / "passwd-notes.txt").write_text("in resources")
    (outside / "passwd").write_text("outside")
    (resources / "zz_out").symlink_to(outside)
    (resources / "passwd_link").symlink_to(outside / "passwd")
    monkeypatch.setenv("RESOURCES_DIR", str(resources))

    response = await api_client.get("/api/resources/search", params={"q": "passwd"})
    assert response.status_code == 200
    body = response.json()
    assert body["truncated"] is False
    assert [match["path"] for match in body["matches"]] == ["passwd-notes.txt"]


async def test_resources_search_escape_does_not_consume_node_cap(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link out of the resources folder does not count toward the walk limit."""
    resources = tmp_path / "resources"
    outside = tmp_path / "outside"
    resources.mkdir()
    outside.mkdir()
    (outside / "nomatch_target.txt").write_text("x")
    _make_capped_escape_resources(resources, outside)
    monkeypatch.setenv("RESOURCES_DIR", str(resources))

    body = (await api_client.get("/api/resources/search", params={"q": "nomatch"})).json()
    assert body["truncated"] is False
    assert body["matches"] == []


async def test_resources_search_bad_params(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    monkeypatch.setenv("RESOURCES_DIR", str(resources))
    empty_response = await api_client.get("/api/resources/search", params={"q": ""})
    assert empty_response.status_code == 400
    assert empty_response.json()["code"] == "bad_param"
    missing_response = await api_client.get("/api/resources/search")
    assert missing_response.status_code == 400
    bad_limit_response = await api_client.get("/api/resources/search", params={"q": "x", "limit": 0})
    assert bad_limit_response.status_code == 400


async def test_resources_renders_top_level(
    api_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The six top level folders list with their exact file counts."""
    resources = tmp_path / "resources"
    _make_top_level_resources(resources)
    monkeypatch.setenv("RESOURCES_DIR", str(resources))
    body = (await api_client.get("/api/resources")).json()
    names = [entry["name"] for entry in body["entries"]]
    assert names == ["debug_flake", "downloads", "github", "media_gen", "tts", "upload"]
    top = {entry["name"]: entry for entry in body["entries"]}
    assert top["debug_flake"]["file_count"] == _FLAKE_FILE_COUNT
    assert top["downloads"]["file_count"] == _DOWNLOAD_FILE_COUNT
    subdir_body = (await api_client.get("/api/resources", params={"path": "downloads"})).json()
    assert len(subdir_body["entries"]) == _DOWNLOAD_FILE_COUNT
