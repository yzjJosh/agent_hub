from __future__ import annotations

from typing import ClassVar, override

from app.lib.tool.base import Param, Tool, ToolResult
from app.resources.resources import dav_base_url

__all__ = ["send_or_receive_file_tool"]


def _instructions() -> str:
    """Return the file transfer instructions."""
    base = dav_base_url()
    return (
        "To pass a file to this MCP server for a follow-up tool call:\n"
        "1. Upload it with WebDAV PUT to the upload/ folder:\n"
        f"   curl -T /local/path/to/file {base}/dav/upload/<filename>\n"
        "2. Reference the uploaded file by its relative path in the follow-up tool call,\n"
        "   e.g. upload/<filename>.\n"
        "\n"
        "To receive a file as a result from a tool call:\n"
        "1. Download it with WebDAV GET, saving into /tmp rather than the current folder:\n"
        f"   curl -o /tmp/<filename> {base}/dav/<path>\n"
        "2. Or view a text file's content without downloading, using your MCP client's\n"
        "   resource reading feature with the file's URI as the uri parameter:\n"
        '   uri: "resources:///<path>"\n'
        "   Reading a folder (e.g. resources:///upload/) lists the files in it.\n"
        "   Binary files are not shown; you will get a WebDAV download command to use instead.\n"
        "3. List all files with your MCP client's resource listing tool, which takes\n"
        "   no arguments. Every file appears as 'resources:///<path>'.\n"
        "\n"
        "If asked for the MCP URL (e.g. to point your MCP client's resources tool at\n"
        f"this server), use: {base}/mcp\n"
        "\n"
        "File retention: files not used or written for 30 days are automatically deleted.\n"
        "Downloads with GET /dav/<path>, MCP resource reads, tool use, opening a file in the console,\n"
        "and writes renew retention. Listings, searches, previews, and thumbnails do not.\n"
        "\n"
        "Permissions: only the upload/ folder is read-write; everything else is read-only."
    )


class _SendOrReceiveFileTool(Tool):
    name: ClassVar[str] = "send_or_receive_file"
    tracked: ClassVar[bool] = False
    description: ClassVar[str] = (
        "Get instructions for transferring files with this server's WebDAV "
        "endpoint at /dav. Use when you need to upload a file for a follow-up "
        "tool call, or to download a file a tool call produced."
    )
    summary: ClassVar[str] = "File transfer instructions (WebDAV)"
    params: ClassVar[tuple[Param, ...]] = ()

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text=_instructions())


send_or_receive_file_tool = _SendOrReceiveFileTool()
