"""Tests for the resources module."""

import os
import time
from pathlib import Path

from a2wsgi import WSGIMiddleware
import httpx
import pytest
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

from app.lib.shared import activity
from app.resources.resources import (
    binary_download_notice,
    create_dav_app,
    dav_base_url,
    register_resources,
)


@pytest.fixture
def mcp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastMCP:
    """A server with the resources registered on a temp folder."""
    monkeypatch.setenv("RESOURCES_DIR", str(tmp_path))
    server = FastMCP("test")
    register_resources(server, str(tmp_path))
    return server


@pytest.mark.parametrize("uri,content,renewed", [
    ("resources:///", "old.txt\t3 bytes", False),
    ("resources:///old.txt", "old", True),
])
async def test_only_file_read_records_interest(
    mcp: FastMCP, tmp_path: Path, uri: str, content: str, renewed: bool,
) -> None:
    """Listing resources leaves their interest unchanged, reading a file renews it."""
    path = tmp_path / "old.txt"
    path.write_text("old")
    old = time.time() - 40 * 86400
    os.utime(path, (old, old))
    await mcp.list_resources()
    assert activity.last_used_sync(str(path), path.stat()) == old
    response = await mcp.read_resource(uri)
    assert response.contents[0].content == content
    assert (activity.last_used_sync(str(path), path.stat()) > time.time() - 10) is renewed


class TestRegisterResources:
    async def test_registers_listing_and_file_resources(self, mcp: FastMCP) -> None:
        """Registration exposes the listing resource and the file template."""
        uris = [str(resource.uri) for resource in await mcp.list_resources()]
        templates = [t.uri_template for t in await mcp.list_resource_templates()]
        assert uris == ["resources:///"]
        assert templates == ["resources:///{path*}"]


class TestListingResource:
    async def test_lists_files_with_sizes(self, mcp: FastMCP, tmp_path: Path) -> None:
        """The listing resource shows files with sizes, nested paths included."""
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("world!")
        result = await mcp.read_resource("resources:///")
        assert result.contents[0].content == "a.txt\t5 bytes\nsub/b.txt\t6 bytes"

    async def test_empty_folder(self, mcp: FastMCP) -> None:
        """An empty folder is listed as empty."""
        result = await mcp.read_resource("resources:///")
        assert result.contents[0].content == "(empty)"

    async def test_missing_folder(self, tmp_path: Path) -> None:
        """The listing resource reports a missing folder with its path."""
        server = FastMCP("test")
        register_resources(server, str(tmp_path / "nope"))
        result = await server.read_resource("resources:///")
        assert result.contents[0].content == f"Resources directory not found: {tmp_path / 'nope'}"

    async def test_resource_list_includes_each_file(self, mcp: FastMCP, tmp_path: Path) -> None:
        """The resource list carries one entry per file currently on disk."""
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.png").write_bytes(b"\x89PNG")
        uris = {str(resource.uri) for resource in await mcp.list_resources()}
        assert "resources:///" in uris
        assert "resources:///a.txt" in uris
        assert "resources:///sub/b.png" in uris

    async def test_resource_list_reflects_new_files(self, mcp: FastMCP, tmp_path: Path) -> None:
        """Files created after registration appear in a later listing."""
        uris = {str(resource.uri) for resource in await mcp.list_resources()}
        assert "resources:///late.txt" not in uris
        (tmp_path / "late.txt").write_text("x")
        uris = {str(resource.uri) for resource in await mcp.list_resources()}
        assert "resources:///late.txt" in uris

    async def test_listed_file_entry_has_mime_type_and_is_readable(
        self, mcp: FastMCP, tmp_path: Path
    ) -> None:
        """Each listed entry carries the guessed mime type and reads back its content."""
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "img.png").write_bytes(b"\x89PNG")
        entries = {
            str(resource.uri): resource
            for resource in await mcp.list_resources()
        }
        assert entries["resources:///img.png"].mime_type == "image/png"
        assert entries["resources:///a.txt"].name == "a.txt"
        result = await entries["resources:///a.txt"].read()
        assert result == "hello"


class TestFileResource:
    async def test_reads_file(self, mcp: FastMCP, tmp_path: Path) -> None:
        """The file template returns the content of a top-level file."""
        (tmp_path / "a.txt").write_text("hello")
        result = await mcp.read_resource("resources:///a.txt")
        assert result.contents[0].content == "hello"

    async def test_reads_nested_file(self, mcp: FastMCP, tmp_path: Path) -> None:
        """The file template returns the content of a nested file."""
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("world")
        result = await mcp.read_resource("resources:///sub/b.txt")
        assert result.contents[0].content == "world"

    async def test_missing_file(self, mcp: FastMCP) -> None:
        """A missing file reports 'Resource not found' with the requested path."""
        result = await mcp.read_resource("resources:///nope.txt")
        assert result.contents[0].content == "Resource not found: nope.txt"

    async def test_binary_file_returns_download_instructions(
        self, mcp: FastMCP, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-UTF-8 file reports its size and a WebDAV download command."""
        monkeypatch.setenv("DAV_BASE_URL", "http://example.com:8315")
        data = b"\x89PNG\r\n\x1a\n" + b"\xff\xfe" * 5
        (tmp_path / "upload").mkdir()
        (tmp_path / "upload" / "img.png").write_bytes(data)
        result = await mcp.read_resource("resources:///upload/img.png")
        assert result.contents[0].content == (
            f"Binary file with size {len(data)} bytes. "
            "Please use WebDAV to download it instead:\n"
            "  curl -o /tmp/img.png http://example.com:8315/dav/upload/img.png"
        )

    async def test_binary_file_uses_default_base_url(
        self, mcp: FastMCP, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The download command falls back to http://localhost:8315 when DAV_BASE_URL is unset."""
        monkeypatch.delenv("DAV_BASE_URL", raising=False)
        (tmp_path / "img.png").write_bytes(b"\x00" * 4)
        result = await mcp.read_resource("resources:///img.png")
        assert result.contents[0].content == (
            "Binary file with size 4 bytes. Please use WebDAV to download it instead:\n"
            "  curl -o /tmp/img.png http://localhost:8315/dav/img.png"
        )

    async def test_reads_folder_listing(self, mcp: FastMCP, tmp_path: Path) -> None:
        """Reading a folder lists its files with root-relative paths."""
        (tmp_path / "upload").mkdir()
        (tmp_path / "upload" / "a.txt").write_text("hello")
        result = await mcp.read_resource("resources:///upload/")
        assert result.contents[0].content == "upload/a.txt\t5 bytes"

    async def test_reads_folder_without_trailing_slash(self, mcp: FastMCP, tmp_path: Path) -> None:
        """A folder path without a trailing slash lists the folder too."""
        (tmp_path / "upload").mkdir()
        (tmp_path / "upload" / "a.txt").write_text("hello")
        result = await mcp.read_resource("resources:///upload")
        assert result.contents[0].content == "upload/a.txt\t5 bytes"

    async def test_reads_nested_folder_recursively(self, mcp: FastMCP, tmp_path: Path) -> None:
        """A folder listing includes files in nested subfolders."""
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "b").mkdir()
        (tmp_path / "a" / "b" / "c.txt").write_text("12345")
        result = await mcp.read_resource("resources:///a/")
        assert result.contents[0].content == "a/b/c.txt\t5 bytes"

    async def test_reads_empty_folder(self, mcp: FastMCP, tmp_path: Path) -> None:
        """Reading an empty folder says it is empty."""
        (tmp_path / "empty").mkdir()
        result = await mcp.read_resource("resources:///empty/")
        assert result.contents[0].content == "(empty)"

    async def test_reads_file_with_trailing_slash(self, mcp: FastMCP, tmp_path: Path) -> None:
        """A trailing slash on a file path is tolerated."""
        (tmp_path / "a.txt").write_text("hello")
        result = await mcp.read_resource("resources:///a.txt/")
        assert result.contents[0].content == "hello"

    async def test_missing_folder_reports_not_found(self, mcp: FastMCP) -> None:
        """A missing folder reports 'Resource not found' with the cleaned path."""
        result = await mcp.read_resource("resources:///nope/")
        assert result.contents[0].content == "Resource not found: nope"

    @pytest.mark.parametrize("path", ["../escape.txt", "/etc/passwd", "a/../../escape.txt"])
    async def test_rejects_traversal(self, mcp: FastMCP, tmp_path: Path, path: str) -> None:
        """Path traversal attempts cannot read files outside the root folder."""
        (tmp_path.parent / "escape.txt").write_text("secret")
        result = await mcp.read_resource(f"resources:///{path}")
        assert result.contents[0].content != "secret"
        assert result.contents[0].content.startswith("Resource not found")

    async def test_rejects_symlink_escape(self, mcp: FastMCP, tmp_path: Path) -> None:
        """Symlinks pointing outside the root are refused by the file resource."""
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("secret")
        (tmp_path / "link.txt").symlink_to(outside)
        result = await mcp.read_resource("resources:///link.txt")
        assert result.contents[0].content == "Resource not found: link.txt"


class TestCreateDavApp:
    @pytest.fixture
    async def client(self, tmp_path: Path) -> httpx.AsyncClient:
        """A client for a WebDAV server rooted at a temp folder."""
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "sub").mkdir()
        star = Starlette(routes=[Mount("/dav", WSGIMiddleware(create_dav_app(str(tmp_path))))])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=star), base_url="http://t"
        ) as client:
            yield client

    async def test_serves_read_access_everywhere(self, client: httpx.AsyncClient) -> None:
        """PROPFIND listings and GET reads work on every part of the share."""
        response = await client.request("PROPFIND", "/dav/", headers={"Depth": "1"})
        assert response.status_code == 207
        assert "a.txt" in response.text
        response = await client.get("/dav/a.txt")
        assert response.status_code == 200
        assert response.text == "hello"

    @pytest.mark.parametrize(
        ("method", "path", "should_still_exist"),
        [
            ("PUT", "/dav/new.txt", False),
            ("PUT", "/dav/sub/nested.txt", False),
            ("MKCOL", "/dav/newdir", False),
            ("DELETE", "/dav/a.txt", True),
        ],
    )
    async def test_rejects_writes_outside_upload(
        self,
        client: httpx.AsyncClient,
        tmp_path: Path,
        method: str,
        path: str,
        should_still_exist: bool,
    ) -> None:
        """Write attempts outside the upload folder are forbidden and leave the tree untouched."""
        body = b"abc" if method == "PUT" else None
        response = await client.request(method, path, content=body)
        assert response.status_code == 403
        rel = path.removeprefix("/dav").lstrip("/")
        assert (tmp_path / rel).exists() == should_still_exist

    async def test_allows_writes_under_upload(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """PUT and MKCOL succeed under the upload folder and land on disk."""
        response = await client.put("/dav/upload/new.txt", content="abc")
        assert response.status_code == 201
        assert (tmp_path / "upload" / "new.txt").read_text() == "abc"
        response = await client.request("MKCOL", "/dav/upload/sub")
        assert response.status_code == 201
        assert (tmp_path / "upload" / "sub").is_dir()
        response = await client.put("/dav/upload/sub/deep.txt", content="xyz")
        assert response.status_code == 201
        assert (tmp_path / "upload" / "sub" / "deep.txt").read_text() == "xyz"


class TestDavHelpers:
    def test_dav_base_url_returns_default_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DAV_BASE_URL unset yields the http://localhost:8315 default."""
        monkeypatch.delenv("DAV_BASE_URL", raising=False)
        assert dav_base_url() == "http://localhost:8315"

    def test_dav_base_url_returns_default_when_env_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty DAV_BASE_URL is treated as unset."""
        monkeypatch.setenv("DAV_BASE_URL", "")
        assert dav_base_url() == "http://localhost:8315"

    def test_dav_base_url_returns_env_value_with_trailing_slash_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DAV_BASE_URL is returned as-is, minus any trailing slash."""
        monkeypatch.setenv("DAV_BASE_URL", "http://agent.example.com:8315/")
        assert dav_base_url() == "http://agent.example.com:8315"

    async def test_binary_download_notice_format(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The notice is a size line followed by an indented curl command into /tmp/."""
        monkeypatch.setenv("DAV_BASE_URL", "http://example.com:8315")
        monkeypatch.setenv("RESOURCES_DIR", str(tmp_path))
        (tmp_path / "downloads").mkdir()
        (tmp_path / "downloads" / "plan.pdf").write_bytes(b"x" * 12345)
        notice = await binary_download_notice("downloads/plan.pdf")
        assert notice == (
            "Binary file with size 12345 bytes. "
            "Please use WebDAV to download it instead:\n"
            "  curl -o /tmp/plan.pdf http://example.com:8315/dav/downloads/plan.pdf"
        )
