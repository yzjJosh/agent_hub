"""Tests for app.tools.send_or_receive_file."""

import pytest

from app.tools.send_or_receive_file import send_or_receive_file_tool
from tests.app.lib.tool.conftest import run_text


async def test_uses_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no DAV_BASE_URL set, the instructions point at localhost port 8315."""
    monkeypatch.delenv("DAV_BASE_URL", raising=False)
    result = await run_text(send_or_receive_file_tool)
    assert "http://localhost:8315/dav/upload/" in result
    assert "http://localhost:8315/dav/" in result


async def test_uses_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The instructions build the dav and upload paths from DAV_BASE_URL."""
    monkeypatch.setenv("DAV_BASE_URL", "http://agent.example.com:8315")
    result = await run_text(send_or_receive_file_tool)
    assert "http://agent.example.com:8315/dav/upload/" in result
    assert "http://agent.example.com:8315/dav/" in result


async def test_documents_upload_via_webdav_put() -> None:
    """The instructions tell you to PUT a file into the upload folder."""
    result = await run_text(send_or_receive_file_tool)
    assert "PUT" in result
    assert "/dav/upload/" in result
    assert "curl -T" in result


async def test_documents_download_via_webdav_get() -> None:
    """The instructions tell you to GET a file to download it."""
    result = await run_text(send_or_receive_file_tool)
    assert "GET" in result
    assert "curl -o /tmp/<filename>" in result


async def test_documents_resources_tool_for_content_and_listing() -> None:
    """The instructions give the resource paths and the uri argument name."""
    result = await run_text(send_or_receive_file_tool)
    assert 'uri: "resources:///<path>"' in result
    assert "resource listing tool, which takes\n   no arguments" in result
    assert "Every file appears as 'resources:///<path>'" in result
    assert "Reading a folder (e.g. resources:///upload/) lists the files in it" in result
    assert "Binary files are not shown" in result


async def test_documents_mcp_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The instructions give the MCP address as the base URL plus mcp, and name the resources tool."""
    monkeypatch.setenv("DAV_BASE_URL", "http://agent.example.com:8315")
    result = await run_text(send_or_receive_file_tool)
    assert "http://agent.example.com:8315/mcp" in result
    assert "resources tool" in result


async def test_documents_retention_and_permissions() -> None:
    """The instructions say files are kept 30 days and mention read-write and read-only access."""
    result = await run_text(send_or_receive_file_tool)
    assert "30 days" in result
    assert "read-write" in result
    assert "read-only" in result
