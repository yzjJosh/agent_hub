"""Tests for clearing the render cache when share files change."""

import os
import time
from pathlib import Path

import httpx
import pytest
from a2wsgi import WSGIMiddleware
from starlette.applications import Starlette
from starlette.routing import Mount

from app.lib.render.cache import convert_failed_reason, key_for, mark_failed, render_root, thumb_status, write_doc
from app.lib.shared import activity
from app.resources.dav_hook import CacheInvalidatingProvider
from app.resources.resources import create_dav_app
from tests.app.conftest import point_share_env

_MTIME = 1700000000


@pytest.mark.parametrize("method,url,status", [
    ("HEAD", "/dav/upload/old.html", 200),
    ("PROPFIND", "/dav/upload/old.html", 207),
    ("GET", "/dav/upload/old.html?ephemeral=1", 200),
])
async def test_incidental_requests_leave_interest_unchanged(
    client: httpx.AsyncClient, share: Path, method: str, url: str, status: int,
) -> None:
    """Metadata and ephemeral reads do not renew retention."""
    path = share / "upload" / "old.html"
    path.write_text("<p>old</p>")
    os.utime(path, (_MTIME, _MTIME))
    response = await client.request(method, url, headers={"Depth": "0"} if method == "PROPFIND" else {})
    assert response.status_code == status
    assert activity.last_used_sync(str(path), path.stat()) == _MTIME


@pytest.mark.parametrize("headers,status,body", [
    ({}, 200, b"<p>old</p>"), ({"Range": "bytes=0-2"}, 206, b"<p>"),
])
async def test_ephemeral_get_returns_same_bytes(
    client: httpx.AsyncClient, share: Path, headers: dict[str, str], status: int, body: bytes,
) -> None:
    """An ephemeral read sends the same file bytes, ranges included."""
    path = share / "upload" / "old.html"
    path.write_text("<p>old</p>")
    response = await client.get("/dav/upload/old.html?ephemeral=1", headers=headers)
    assert response.status_code == status
    assert response.content == body


@pytest.mark.parametrize("headers,status", [({}, 200), ({"Range": "bytes=0-2"}, 206)])
async def test_file_get_records_interest(
    client: httpx.AsyncClient, share: Path, headers: dict[str, str], status: int,
) -> None:
    """Reading a raw file renews retention, including range requests."""
    path = share / "upload" / "old.html"
    path.write_text("<p>old</p>")
    os.utime(path, (_MTIME, _MTIME))
    response = await client.get("/dav/upload/old.html", headers=headers)
    assert response.status_code == status
    assert activity.last_used_sync(str(path), path.stat()) > time.time() - 10


async def _publish(share: Path, rel: str) -> str:
    """Make a share file with a render entry and give back its key."""
    src = share / rel
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("x", encoding="utf-8")
    os.utime(src, (_MTIME, _MTIME))
    key = key_for(rel, _MTIME)
    await write_doc(key, "doc", None, {"path": rel, "mtime": _MTIME})
    return key


@pytest.fixture
def share(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give the test its own share dir."""
    share, _data = point_share_env(tmp_path, monkeypatch)
    return share


@pytest.fixture
async def client(share: Path) -> httpx.AsyncClient:
    """A client for a WebDAV server that serves the share."""
    star = Starlette(routes=[Mount("/dav", WSGIMiddleware(create_dav_app(str(share))))])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=star), base_url="http://t"
    ) as http:
        yield http


class TestDavInvalidation:
    async def test_delete_removes_entry_and_markers(
        self, client: httpx.AsyncClient, share: Path
    ) -> None:
        """Deleting a file removes its entry and both of its failure marks."""
        key = await _publish(share, "upload/a.md")
        await mark_failed(key, "thumb", "engine failed")
        await mark_failed(key, "convert", "engine failed")
        response = await client.request("DELETE", "/dav/upload/a.md")
        assert response.status_code == 204
        assert not (share / "upload" / "a.md").exists()
        assert not os.path.isdir(os.path.join(render_root(), key))
        assert await thumb_status(key) == "pending"  # the thumbnail failure mark is gone
        assert await convert_failed_reason(key) is None  # so is the conversion one

    async def test_delete_folder_removes_children_entries(
        self, client: httpx.AsyncClient, share: Path
    ) -> None:
        """Deleting a folder removes every entry under it, nested ones too."""
        key_a = await _publish(share, "upload/x/a.md")
        key_b = await _publish(share, "upload/x/sub/b.md")
        response = await client.request("DELETE", "/dav/upload/x")
        assert response.status_code == 204
        assert not (share / "upload" / "x").exists()
        assert not os.path.isdir(os.path.join(render_root(), key_a))
        assert not os.path.isdir(os.path.join(render_root(), key_b))

    async def test_put_overwrite_removes_old_entry(
        self, client: httpx.AsyncClient, share: Path
    ) -> None:
        """Writing over a file removes its old entry."""
        key = await _publish(share, "upload/a.md")
        response = await client.put("/dav/upload/a.md", content="new content")
        assert response.status_code == 204
        assert (share / "upload" / "a.md").read_text() == "new content"
        assert not os.path.isdir(os.path.join(render_root(), key))

    async def test_put_new_file_leaves_the_cache_alone(
        self, client: httpx.AsyncClient, share: Path
    ) -> None:
        """Writing a brand new file works and removes nothing."""
        response = await client.put("/dav/upload/new.md", content="fresh")
        assert response.status_code == 201
        assert (share / "upload" / "new.md").read_text() == "fresh"

    async def test_move_invalidates_source_and_destination(
        self, client: httpx.AsyncClient, share: Path
    ) -> None:
        """Moving a file over another one removes both of their entries."""
        key_a = await _publish(share, "upload/a.md")
        key_b = await _publish(share, "upload/b.md")
        # The destination header has no /dav prefix.
        response = await client.request(
            "MOVE",
            "/dav/upload/a.md",
            headers={"Destination": "/upload/b.md", "Overwrite": "T"},
        )
        assert response.status_code == 204
        assert not (share / "upload" / "a.md").exists()
        assert (share / "upload" / "b.md").exists()
        assert not os.path.isdir(os.path.join(render_root(), key_a))
        assert not os.path.isdir(os.path.join(render_root(), key_b))

    async def test_invalidate_error_does_not_fail_the_delete(
        self,
        client: httpx.AsyncClient,
        share: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A delete still works when clearing the cache fails."""
        await _publish(share, "upload/a.md")

        def boom(_path: str) -> list[str]:
            raise RuntimeError("disk on fire")

        monkeypatch.setattr("app.resources.dav_hook.invalidate_sync", boom)
        response = await client.request("DELETE", "/dav/upload/a.md")
        assert response.status_code == 204
        assert not (share / "upload" / "a.md").exists()

    async def test_resource_outside_the_share_root_is_a_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file outside the share root leaves the cache alone."""
        share, _data = point_share_env(tmp_path, monkeypatch)
        key = await _publish(share, "a.txt")
        (share / "upload").mkdir()
        provider = CacheInvalidatingProvider(str(share), share_root=str(share / "upload"))
        environ = {"wsgidav.provider": provider, "wsgidav.config": {}}
        resource = provider.get_resource_inst("/a.txt", environ)
        assert resource.name == "a.txt"
        resource.delete()
        assert not (share / "a.txt").exists()
        assert os.path.isdir(os.path.join(render_root(), key))
