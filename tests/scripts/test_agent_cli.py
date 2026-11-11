"""Tests for the agent command line client.

Most tests call the script with a fake server. The live tests run the script
against the real server, on port 80 inside the container and port 8315 on the
host, unless a different server address is set.
"""

import base64
import httpx
import importlib.machinery
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import types
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
    Resource,
    TextContent,
    TextResourceContents,
    Tool,
)

_SCRIPT_PATH = Path(
    os.environ.get("AGENT_CLI_PATH", Path(__file__).resolve().parents[2] / "scripts" / "agent_cli")
)
_MODULE_NAME = "agent_cli"
_LIVE_TIMEOUT = 60
_DOCUMENTED_MCP_URL = "http://agent.example.com:8315/mcp"

_SCHEMA: dict = {
    "properties": {
        "action": {"type": "string"},
        "limit": {"type": "integer"},
        "ratio": {"type": "number"},
        "verbose": {"type": "boolean"},
        "items": {"type": "array"},
        "tag": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["action"],
}
_TOOL = Tool(name="demo", description="Demo tool.", inputSchema=_SCHEMA)
_EMPTY_TOOL = Tool(name="empty", description="Empty tool.", inputSchema={})
_MEDIA_TOOL = Tool(
    name="media",
    description="Media tool.",
    inputSchema={"properties": {"resource_path": {"type": "string"}}, "required": ["resource_path"]},
)
_MEDIA_LIST_TOOL = Tool(
    name="media_list",
    description="Media list tool.",
    inputSchema={
        "properties": {
            "prompt": {"type": "string"},
            "reference_images": {
                "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
            },
        },
        "required": ["prompt"],
    },
)


def _load_cli_module() -> types.ModuleType:
    """Load the agent command line script as a module."""
    loader = importlib.machinery.SourceFileLoader(_MODULE_NAME, str(_SCRIPT_PATH))
    spec = importlib.util.spec_from_loader(_MODULE_NAME, loader)
    if spec is None:
        raise AssertionError(f"cannot load script at {_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    loader.exec_module(module)
    return module


_load_cli_module()

from agent_cli import main


def _fake_session(
    tools: list[Tool],
    call_result: CallToolResult | None,
    resource_uris: tuple[str, ...] = (),
) -> MagicMock:
    """Build a fake server session that serves the given tools."""
    session = MagicMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=ListToolsResult(tools=tools))
    session.list_resources = AsyncMock(
        return_value=ListResourcesResult(
            resources=[Resource(uri=uri, name=uri.rsplit("/", 1)[-1]) for uri in resource_uris]
        )
    )
    session.call_tool = AsyncMock(return_value=call_result)
    return session


def _mock_client_stack(session: MagicMock) -> tuple[MagicMock, MagicMock]:
    """Build the fake client and session classes that hand back the given session."""
    http_client = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=(None, None, None)),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    session_cls = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=session), __aexit__=AsyncMock(return_value=False))
    )
    return http_client, session_cls


@contextmanager
def _mocked_cli(
    argv: list[str],
    session: MagicMock | None = None,
    tools: list[Tool] | None = None,
    result: CallToolResult | None = None,
    resources: tuple[str, ...] = (),
) -> Iterator[tuple[MagicMock, MagicMock]]:
    """Set up a fake server and a fake command line, and give back the session and the client."""
    if session is None:
        session = _fake_session(
            tools if tools is not None else [_TOOL], result, resource_uris=resources
        )
    http_client, session_cls = _mock_client_stack(session)
    with (
        patch("agent_cli.streamablehttp_client", http_client),
        patch("agent_cli.ClientSession", session_cls),
        patch.object(sys, "argv", ["agent_cli", *argv]),
    ):
        yield session, http_client


def _run_main(
    argv: list[str],
    result: CallToolResult | None = None,
    tools: list[Tool] | None = None,
    expect_exit: bool = False,
    resources: tuple[str, ...] = (),
) -> MagicMock:
    """Run the command line against a fake server, and give back the session."""
    with _mocked_cli(argv, tools=tools, result=result, resources=resources) as (session, _):
        if expect_exit:
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 1
        else:
            main()
    return session


# The server is on port 80 inside the container and port 8315 on the host.
_DEFAULT_MCP_URL = (
    "http://127.0.0.1:80/mcp" if os.path.exists("/.dockerenv") else "http://127.0.0.1:8315/mcp"
)


@lru_cache(maxsize=None)
def _server_reachable() -> bool:
    """Check the live server answers, once per session."""
    try:
        httpx.get(_DEFAULT_MCP_URL, timeout=2)
    except httpx.HTTPError:
        return False
    return True


def _run_cli(
    *argv: str,
    env: dict[str, str] | None = None,
    timeout: int = _LIVE_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Run the command line script against the live server."""
    full_env = dict(os.environ, MCP_URL=os.environ.get("MCP_URL", _DEFAULT_MCP_URL))
    if env is not None:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=full_env,
    )


class TestCoerceValue:
    """Test how command line values are turned into the types a tool wants."""

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            (["--action", "x", "--limit", "5"], {"action": "x", "limit": 5}),
            (["--action", "x", "--ratio", "1.5"], {"action": "x", "ratio": 1.5}),
            (["--action", "x", "--verbose", "true"], {"action": "x", "verbose": True}),
            (["--action", "5"], {"action": "5"}),
            (["--action", "x", "--items", '["a", "b"]'], {"action": "x", "items": ["a", "b"]}),
            (["--action=x", "--limit=3"], {"action": "x", "limit": 3}),
            (["--action", "x", "--tag", "t"], {"action": "x", "tag": "t"}),
        ],
    )
    def test_coerces_values_to_schema_types(
        self, params: list[str], expected: dict
    ) -> None:
        """Both --name value and --name=value pairs reach the tool as the right type."""
        session = _run_main(
            ["tools", "call", "demo", *params],
            CallToolResult(content=[], isError=False),
        )
        session.call_tool.assert_awaited_once_with("demo", expected)

    def test_nullable_array_null_becomes_none(self) -> None:
        """The word null for a list that allows null is sent as null."""
        session = _run_main(
            [
                "tools",
                "call",
                "media_list",
                "--prompt",
                "x",
                "--reference_images",
                "null",
            ],
            CallToolResult(content=[], isError=False),
            tools=[_MEDIA_LIST_TOOL],
        )
        session.call_tool.assert_awaited_once_with(
            "media_list", {"prompt": "x", "reference_images": None}
        )


class TestParseParams:
    """Test that bad parameters exit with status 1 and a message on stderr."""

    @pytest.mark.parametrize(
        ("params", "message"),
        [
            (
                ["--action", "x", "--limit", "abc"],
                "parameter --limit expects integer, got 'abc'",
            ),
            (
                ["--action", "x", "--verbose", "1"],
                "parameter --verbose expects boolean, got '1'",
            ),
            (["--action", "x", "--bogus", "1"], "unknown parameter --bogus"),
            ([], "missing required parameters: --action"),
            (["--action"], "each --name needs a value: ['--action']"),
            (["action", "x"], "expected --name value pairs, got 'action'"),
            (["--action", "x", "--limit", "1", "--limit", "2"], "duplicate parameter --limit"),
        ],
    )
    def test_rejects_invalid_params(
        self, params: list[str], message: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A bad parameter is turned down before the tool is called."""
        with _mocked_cli(["tools", "call", "demo", *params]) as (session, _):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 1
        assert message in capsys.readouterr().err
        session.call_tool.assert_not_called()

    def test_rejects_unknown_param_on_empty_schema(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """For a tool with no parameters, the error says the valid list is none."""
        session = _run_main(
            ["tools", "call", "empty", "--x", "1"],
            tools=[_EMPTY_TOOL],
            expect_exit=True,
        )
        assert "unknown parameter --x; valid parameters: none" in capsys.readouterr().err
        session.call_tool.assert_not_called()


class TestCallTool:
    """Test what a tool call prints and what it exits with."""

    def test_uses_single_connection(self) -> None:
        """A tool call opens one connection to the server."""
        with _mocked_cli(
            ["tools", "call", "demo", "--action", "x"],
            result=CallToolResult(content=[TextContent(type="text", text="ok")], isError=False),
        ) as (session, http_client):
            main()
        assert http_client.call_count == 1
        session.call_tool.assert_awaited_once_with("demo", {"action": "x"})

    def test_error_result_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A failed tool call prints the text and exits with status 1."""
        _run_main(
            ["tools", "call", "demo", "--action", "x"],
            result=CallToolResult(content=[TextContent(type="text", text="boom")], isError=True),
            expect_exit=True,
        )
        captured = capsys.readouterr()
        assert captured.out == "boom\n"
        assert "tool call failed" in captured.err

    def test_structured_content_printed_as_json(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A result with no text is printed as JSON."""
        session = _run_main(
            ["tools", "call", "demo", "--action", "x"],
            CallToolResult(content=[], structuredContent={"n": 1}, isError=False),
        )
        session.call_tool.assert_awaited_once()
        expected = json.dumps({"n": 1}, indent=2, ensure_ascii=False) + "\n"
        assert capsys.readouterr().out == expected

    def test_unknown_tool_exits_1(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An unknown tool name exits 1 with a message on stderr."""
        _run_main(["tools", "call", "missing"], expect_exit=True)
        assert "unknown tool" in capsys.readouterr().err


class TestShowTool:
    """Test what showing a tool prints for its parameters."""

    def test_prints_schema_bounds(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Each bound in the schema is printed after the default, and no bound prints nothing."""
        tool = Tool(
            name="bounded",
            description="Bounded tool.",
            inputSchema={
                "properties": {
                    "max_char": {"type": "integer", "default": 25000, "minimum": 500},
                    "count": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        )
        _run_main(["tools", "show", "bounded"], tools=[tool])
        out = capsys.readouterr().out
        assert "  --max_char <integer>  [optional, default 25000, minimum 500]\n" in out
        assert "  --count <integer>  [optional, default 5, minimum 1, maximum 10]\n" in out
        assert "  --query <string>  [required]\n" in out


def _sample_image_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Make two small png files and give back both paths."""
    file_a = tmp_path / "a.png"
    file_b = tmp_path / "b.png"
    file_a.write_bytes(b"aaa")
    file_b.write_bytes(b"bbb")
    return file_a, file_b


@contextmanager
def _mocked_upload(
    status: int = 201,
    delete_status: int = 204,
    put_side_effect: Sequence[MagicMock | RuntimeError] | None = None,
) -> Iterator[tuple[MagicMock, MagicMock]]:
    """Set up a fake web client, and give back the fake class and the fake client.

    Args:
        status: Status code an upload gives back.
        delete_status: Status code a delete gives back.
        put_side_effect: What each upload does, one entry per call.
    """
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    if put_side_effect is None:
        client.put = AsyncMock(return_value=MagicMock(status_code=status, text=""))
    else:
        client.put = AsyncMock(side_effect=put_side_effect)
    client.delete = AsyncMock(return_value=MagicMock(status_code=delete_status, text=""))
    client_cls = MagicMock(return_value=client)
    with patch("agent_cli.httpx.AsyncClient", client_cls):
        yield client_cls, client


class TestResourcePathFallback:
    """Test how a resource_path value is turned into something the server can read."""

    def test_existing_resource_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A name that is already a resource is sent as is, with no upload."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        with _mocked_upload() as (client_cls, _):
            session = _run_main(
                ["tools", "call", "media", "--resource_path", "report.pdf"],
                CallToolResult(content=[], isError=False),
                tools=[_MEDIA_TOOL],
                resources=["resources:///report.pdf"],
            )
        session.call_tool.assert_awaited_once_with("media", {"resource_path": "report.pdf"})
        client_cls.assert_not_called()

    def test_local_file_is_uploaded_and_cleaned_up(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A local file is uploaded under a random name and removed after the call."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        file = tmp_path / "notes.txt"
        file.write_bytes(b"hello upload")
        with _mocked_upload() as (_, client):
            session = _run_main(
                ["tools", "call", "media", "--resource_path", str(file)],
                CallToolResult(content=[], isError=False),
                tools=[_MEDIA_TOOL],
            )
        put_url = client.put.call_args.args[0]
        destination = put_url.removeprefix("http://127.0.0.1:8315/dav/")
        pattern = re.compile(r"^upload/[0-9a-f]{32}\.txt$")
        assert pattern.match(destination), destination
        session.call_tool.assert_awaited_once_with("media", {"resource_path": destination})
        client.put.assert_awaited_once_with(
            f"http://127.0.0.1:8315/dav/{destination}", content=b"hello upload"
        )
        client.delete.assert_awaited_once_with(f"http://127.0.0.1:8315/dav/{destination}")
        assert destination in capsys.readouterr().err

    def test_unknown_value_passes_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A value that is not a resource and not a local file is sent as is."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        missing = str(tmp_path / "missing.pdf")
        with _mocked_upload() as (client_cls, _):
            session = _run_main(
                ["tools", "call", "media", "--resource_path", missing],
                CallToolResult(content=[], isError=False),
                tools=[_MEDIA_TOOL],
            )
        session.call_tool.assert_awaited_once_with("media", {"resource_path": missing})
        client_cls.assert_not_called()

    def test_upload_failure_exits_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A refused upload exits 1 and shows the status on stderr."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        file = tmp_path / "notes.txt"
        file.write_bytes(b"x")
        with _mocked_upload(status=500):
            _run_main(
                ["tools", "call", "media", "--resource_path", str(file)],
                tools=[_MEDIA_TOOL],
                expect_exit=True,
            )
        err = capsys.readouterr().err
        assert "HTTP 500" in err
        assert "MCP request" not in err

    def test_delete_failure_keeps_exit_status(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A failed cleanup warns on stderr and the call still counts as good."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        file = tmp_path / "notes.txt"
        file.write_bytes(b"x")
        with _mocked_upload(delete_status=500):
            session = _run_main(
                ["tools", "call", "media", "--resource_path", str(file)],
                CallToolResult(content=[TextContent(type="text", text="done")], isError=False),
                tools=[_MEDIA_TOOL],
            )
        session.call_tool.assert_awaited_once()
        captured = capsys.readouterr()
        assert captured.out == "done\n"
        assert "HTTP 500" in captured.err

    def test_cleans_up_when_call_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An uploaded file is still removed when the tool call blows up."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        file = tmp_path / "notes.txt"
        file.write_bytes(b"x")
        session = _fake_session([_MEDIA_TOOL], None)
        session.call_tool = AsyncMock(side_effect=RuntimeError("boom"))
        with _mocked_upload() as (_, client):
            with _mocked_cli(
                ["tools", "call", "media", "--resource_path", str(file)], session=session
            ):
                with pytest.raises(SystemExit) as excinfo:
                    main()
        assert excinfo.value.code == 1
        put_url = client.put.call_args.args[0]
        client.delete.assert_awaited_once_with(put_url)

    def test_without_resource_param_skips_lookup(self) -> None:
        """A tool with no resource_path never asks for the resource list."""
        session = _run_main(
            ["tools", "call", "demo", "--action", "x"],
            CallToolResult(content=[], isError=False),
        )
        session.list_resources.assert_not_awaited()
        session.call_tool.assert_awaited_once_with("demo", {"action": "x"})


class TestListResourceParam:
    """Test that every item in a list parameter is handled like a resource path."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("a.jpg,b.jpg", ["a.jpg", "b.jpg"]),
            ('["a.jpg", "b.jpg"]', ["a.jpg", "b.jpg"]),
        ],
    )
    def test_existing_resources_pass_through(
        self, raw: str, expected: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A JSON list or a comma list of known resources is sent as is, with no upload."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        with _mocked_upload() as (client_cls, _):
            session = _run_main(
                [
                    "tools",
                    "call",
                    "media_list",
                    "--prompt",
                    "x",
                    "--reference_images",
                    raw,
                ],
                CallToolResult(content=[], isError=False),
                tools=[_MEDIA_LIST_TOOL],
                resources=["resources:///a.jpg", "resources:///b.jpg"],
            )
        session.call_tool.assert_awaited_once_with(
            "media_list", {"prompt": "x", "reference_images": expected}
        )
        client_cls.assert_not_called()

    def test_local_files_uploaded_and_cleaned_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each local file in the list is uploaded under a random name and removed after."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        file_a, file_b = _sample_image_pair(tmp_path)
        with _mocked_upload() as (_, client):
            session = _run_main(
                [
                    "tools",
                    "call",
                    "media_list",
                    "--prompt",
                    "x",
                    "--reference_images",
                    f"{file_a},{file_b}",
                ],
                CallToolResult(content=[], isError=False),
                tools=[_MEDIA_LIST_TOOL],
            )
        put_urls = [c.args[0] for c in client.put.call_args_list]
        destinations = [u.removeprefix("http://127.0.0.1:8315/dav/") for u in put_urls]
        pattern = re.compile(r"^upload/[0-9a-f]{32}\.png$")
        assert len(destinations) == 2
        assert all(pattern.match(d) for d in destinations), destinations
        assert destinations[0] != destinations[1]
        session.call_tool.assert_awaited_once_with(
            "media_list", {"prompt": "x", "reference_images": destinations}
        )
        assert client.put.await_count == 2
        assert client.put.call_args_list[0] == call(put_urls[0], content=b"aaa")
        assert client.put.call_args_list[1] == call(put_urls[1], content=b"bbb")
        assert client.delete.call_args_list == [
            call(f"http://127.0.0.1:8315/dav/{destination}") for destination in destinations
        ]

    def test_later_upload_failure_cleans_up_earlier_uploads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An earlier upload is removed when a later one fails."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        file_a, file_b = _sample_image_pair(tmp_path)
        put_side_effect = [MagicMock(status_code=201), RuntimeError("boom")]
        with _mocked_upload(put_side_effect=put_side_effect) as (_, client):
            session = _fake_session([_MEDIA_LIST_TOOL], None)
            with _mocked_cli(
                [
                    "tools",
                    "call",
                    "media_list",
                    "--prompt",
                    "x",
                    "--reference_images",
                    f"{file_a},{file_b}",
                ],
                session=session,
            ):
                with pytest.raises(SystemExit) as excinfo:
                    main()
        assert excinfo.value.code == 1
        session.call_tool.assert_not_awaited()
        first_url = client.put.call_args_list[0].args[0]
        client.delete.assert_awaited_once_with(first_url)

    def test_mixed_list_uploads_only_local_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the local file in the list is uploaded, the known resource is sent as is."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        local = tmp_path / "photo.jpg"
        local.write_bytes(b"img")
        with _mocked_upload() as (_, client):
            session = _run_main(
                [
                    "tools",
                    "call",
                    "media_list",
                    "--prompt",
                    "x",
                    "--reference_images",
                    f"a.jpg,{local}",
                ],
                CallToolResult(content=[], isError=False),
                tools=[_MEDIA_LIST_TOOL],
                resources=["resources:///a.jpg"],
            )
        destination = client.put.call_args.args[0].removeprefix("http://127.0.0.1:8315/dav/")
        assert re.match(r"^upload/[0-9a-f]{32}\.jpg$", destination), destination
        session.call_tool.assert_awaited_once_with(
            "media_list", {"prompt": "x", "reference_images": ["a.jpg", destination]}
        )
        client.put.assert_awaited_once()
        client.delete.assert_awaited_once_with(f"http://127.0.0.1:8315/dav/{destination}")


class TestDefaultUrl:
    """Test the server address used when none is set."""

    def test_falls_back_to_local_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no server address set, the request goes to the local agent address."""
        monkeypatch.delenv("MCP_URL", raising=False)
        with _mocked_cli(["tools", "list"]) as (_session, http_client):
            main()
        (url,) = http_client.call_args.args
        assert url == "http://127.0.0.1:8315/mcp"


def _run_read_stdout(contents: list) -> bytes:
    """Read a resource and give back the raw bytes it prints."""
    session = MagicMock()
    session.initialize = AsyncMock()
    session.read_resource = AsyncMock(return_value=ReadResourceResult(contents=contents))
    buffer = io.BytesIO()
    stdout = io.TextIOWrapper(buffer, encoding="utf-8", write_through=True)
    real_stdout = sys.stdout
    sys.stdout = stdout
    try:
        with _mocked_cli(["resources", "read", "res://a"], session=session) as _:
            main()
    finally:
        sys.stdout = real_stdout
        stdout.flush()
    return buffer.getvalue()


def _blob(content: bytes) -> BlobResourceContents:
    """Wrap raw bytes as a binary resource."""
    return BlobResourceContents(
        uri="res://a",
        mimeType="application/octet-stream",
        blob=base64.b64encode(content).decode(),
    )


class TestReadResource:
    """Test what reading a resource prints."""

    def test_prints_text_with_newline(self) -> None:
        """Text content is printed with a newline at the end."""
        out = _run_read_stdout(
            [TextResourceContents(uri="res://a", text="hello")]
        )
        assert out == b"hello\n"

    def test_writes_blob_byte_exact(self) -> None:
        """Binary content is written exactly, with no newline added."""
        payload = b"\x00\x01\xffBIN"
        out = _run_read_stdout([_blob(payload)])
        assert out == payload

    def test_preserves_text_blob_order(self) -> None:
        """Text and binary parts keep the order they came in."""
        out = _run_read_stdout(
            [TextResourceContents(uri="res://a", text="T"), _blob(b"B")]
        )
        assert out == b"T\nB"


_SECONDS_PER_DAY = 86400
_STALE_AGE_DAYS = 40
_GC_MAX_AGE_DAYS = 30
_GC_CONTAINER = "agent_hub"
_GC_BASE_PREFIX = "/dev/shm/agent_cli_gc_"
_HOST_RESOURCES_DIR = "/host/only/resources"

# Stands in for docker on PATH. It runs the real docker, and gives every exec
# the resources folder of the test, so the sweep stays inside that folder.
_DOCKER_SHIM = """#!/bin/bash
if [ "$1" = exec ]; then
    shift
    exec {docker} exec -e RESOURCES_DIR={root} "$@"
fi
exec {docker} "$@"
"""


def _docker_available() -> bool:
    """Check the docker command works and the Agent Hub container is running."""
    if shutil.which("docker") is None:
        return False
    try:
        completed = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
        )
    except OSError:
        return False
    return completed.returncode == 0 and _GC_CONTAINER in completed.stdout.split()

_SEED_SNIPPET = (
    "import json, os, sys, time\n"
    "base, files = sys.argv[1], json.loads(sys.argv[2])\n"
    "for name, age in files.items():\n"
    "    path = os.path.join(base, name)\n"
    "    os.makedirs(os.path.dirname(path), exist_ok=True)\n"
    "    with open(path, 'w') as fh:\n"
    "        fh.write('x')\n"
    "    if age is not None:\n"
    "        old = time.time() - age\n"
    "        os.utime(path, (old, old))\n"
)


def _container_exec(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a command inside the Agent Hub container and fail the test if it errors."""
    completed = subprocess.run(
        ["docker", "exec", _GC_CONTAINER, *args], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    return completed


def _container_exists(path: str) -> bool:
    """Check if a path is there inside the Agent Hub container."""
    completed = subprocess.run(
        ["docker", "exec", _GC_CONTAINER, "test", "-e", path], capture_output=True
    )
    return completed.returncode == 0


def _container_seed(base: str, files: dict[str, float | None]) -> None:
    """Make files in the container, and set how many seconds old each one is."""
    _container_exec("python", "-c", _SEED_SNIPPET, base, json.dumps(files))


@pytest.fixture
def gc_sandbox(tmp_path: Path) -> Iterator[tuple[str, dict[str, str]]]:
    """A new folder in the container that the cleanup sweeps in place of the real resources.

    Yields:
        The folder in the container, and the env that points the script's docker at it.
    """
    root = f"{_GC_BASE_PREFIX}{uuid.uuid4().hex}"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    shim = fakebin / "docker"
    shim.write_text(_DOCKER_SHIM.format(docker=shutil.which("docker"), root=root))
    shim.chmod(0o755)
    env = {"PATH": f"{fakebin}{os.pathsep}{os.environ['PATH']}"}
    try:
        yield root, env
    finally:
        _container_exec("rm", "-rf", root)


def _docker_exec_failure() -> subprocess.CompletedProcess[str]:
    """Build the result of a container command that failed."""
    return subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="No such container: agent_hub"
    )


def _gc_exit_code(argv: list[str], run_result: subprocess.CompletedProcess[str]) -> int:
    """Run a cleanup command with a fake container and give back the exit code."""
    with (
        _mocked_cli(argv),
        patch("agent_cli.subprocess.run", return_value=run_result),
    ):
        with pytest.raises(SystemExit) as excinfo:
            main()
    return excinfo.value.code


class TestResourcesGc:
    """Test that the resources cleanup removes old files inside the Agent Hub container."""

    @pytest.mark.skipif(
        not _docker_available(), reason="docker command or Agent Hub container not available"
    )
    def test_removes_stale_files_and_empty_dirs(
        self, gc_sandbox: tuple[str, dict[str, str]]
    ) -> None:
        """Files older than the given days go away, and so do the folders left empty."""
        root, env = gc_sandbox
        _container_seed(
            root,
            {"fresh.txt": None, "stale_dir/stale.txt": _STALE_AGE_DAYS * _SECONDS_PER_DAY},
        )
        result = _run_cli("resources", "gc", "--days", str(_GC_MAX_AGE_DAYS), env=env)
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [f"{root}/stale_dir/stale.txt", f"{root}/stale_dir"]
        assert not _container_exists(f"{root}/stale_dir")
        assert _container_exists(f"{root}/fresh.txt")

    @pytest.mark.skipif(
        not _docker_available(), reason="docker command or Agent Hub container not available"
    )
    def test_prints_nothing_to_remove_when_all_fresh(
        self, gc_sandbox: tuple[str, dict[str, str]]
    ) -> None:
        """When every file is fresh, the run says there is nothing to remove and keeps them."""
        root, env = gc_sandbox
        _container_seed(root, {"fresh.txt": None})
        result = _run_cli("resources", "gc", env=env)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "(nothing to remove)"
        assert _container_exists(f"{root}/fresh.txt")

    def test_rejects_non_integer_days(self) -> None:
        """A days value that is not a whole number exits with status 2."""
        result = _run_cli("resources", "gc", "--days", "soon")
        assert result.returncode == 2
        assert "invalid int value" in result.stderr

    def test_local_error_not_worded_as_mcp(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A failure in a local command is not worded as a server error."""
        with (
            _mocked_cli(["resources", "gc"]),
            patch("agent_cli.subprocess.run", side_effect=RuntimeError("boom")),
        ):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "ERROR" in err
        assert "MCP request" not in err

    def test_docker_failure_reports_container_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A failed container command exits 1 and shows the container error."""
        assert _gc_exit_code(["resources", "gc"], _docker_exec_failure()) == 1
        err = capsys.readouterr().err
        assert "No such container: agent_hub" in err
        assert "MCP request" not in err

    def test_runs_sweep_inside_agent_container(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The cleanup runs inside the Agent Hub container and prints the paths it removed."""
        success = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="/app/data/resources/old.txt\n"
        )
        with (
            _mocked_cli(["resources", "gc", "--days", "7"]),
            patch("agent_cli.subprocess.run", return_value=success) as mock_run,
        ):
            main()
        cmd = mock_run.call_args.args[0]
        assert cmd[:5] == ["docker", "exec", _GC_CONTAINER, "python", "-c"]
        assert cmd[6:] == ["7"]
        assert capsys.readouterr().out == "/app/data/resources/old.txt\n"

    def test_host_resources_dir_does_not_reach_container(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resources dir set on the host is not passed on, and the days default to 30."""
        monkeypatch.setenv("RESOURCES_DIR", _HOST_RESOURCES_DIR)
        success = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
        with (
            _mocked_cli(["resources", "gc"]),
            patch("agent_cli.subprocess.run", return_value=success) as mock_run,
        ):
            main()
        cmd = mock_run.call_args.args[0]
        assert cmd[6:] == ["30"]
        assert not any(_HOST_RESOURCES_DIR in part for part in cmd)

    @pytest.mark.parametrize("days", ["0", "-5"])
    def test_rejects_non_positive_days(self, days: str) -> None:
        """A days value of zero or less is turned down before anything is removed."""
        result = _run_cli("resources", "gc", "--days", days)
        assert result.returncode == 1
        assert "--days must be a positive integer" in result.stderr


class TestToolsGc:
    """Test that the tools cleanup removes old runs inside the Agent Hub container."""

    def test_invokes_gc_cli_runs_category(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The tools cleanup sweeps the runs group and prints the paths it removed."""
        success = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="/app/data/runs/t1\n"
        )
        with (
            _mocked_cli(["tools", "gc", "--days", "7"]),
            patch("agent_cli.subprocess.run", return_value=success) as mock_run,
        ):
            main()
        cmd = mock_run.call_args.args[0]
        assert cmd[:3] == ["docker", "exec", _GC_CONTAINER]
        assert "app.garbage_collector_cli" in cmd
        assert cmd[cmd.index("--category") + 1] == "runs"
        assert cmd[-2:] == ["--days", "7"]
        assert capsys.readouterr().out == "/app/data/runs/t1\n"

    def test_prints_nothing_to_remove_when_all_fresh(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A sweep that removes nothing says there is nothing to remove."""
        success = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
        with (
            _mocked_cli(["tools", "gc"]),
            patch("agent_cli.subprocess.run", return_value=success),
        ):
            main()
        assert capsys.readouterr().out == "(nothing to remove)\n"

    def test_docker_failure_reports_container_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A failed container command exits 1 and shows the container error."""
        assert _gc_exit_code(["tools", "gc"], _docker_exec_failure()) == 1
        err = capsys.readouterr().err
        assert "No such container: agent_hub" in err
        assert "MCP request" not in err


class TestResourcesUpload:
    """Test that upload sends a local file to the given share path."""

    def test_puts_file_bytes_and_prints_destination(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The file bytes go to the file server and the resource path is printed."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        file = tmp_path / "notes.txt"
        file.write_bytes(b"hello upload")
        with _mocked_upload() as (_, client):
            _run_main(["resources", "upload", str(file), "upload/notes.txt"])
        client.put.assert_awaited_once_with(
            "http://127.0.0.1:8315/dav/upload/notes.txt", content=b"hello upload"
        )
        client.delete.assert_not_awaited()
        assert capsys.readouterr().out == "upload/notes.txt\n"

    def test_missing_local_file_exits_1_without_http(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A local file that is not there exits 1 before anything is sent."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        with _mocked_upload() as (_, client):
            _run_main(
                ["resources", "upload", str(tmp_path / "missing.txt"), "upload/x.txt"],
                expect_exit=True,
            )
        client.put.assert_not_awaited()
        assert "No such file" in capsys.readouterr().err

    def test_backend_error_is_delegated(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A refused upload exits 1 and shows the status and the error text."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        file = tmp_path / "notes.txt"
        file.write_bytes(b"x")
        with _mocked_upload() as (_, client):
            client.put = AsyncMock(
                return_value=MagicMock(status_code=403, text="403 Forbidden - write denied")
            )
            _run_main(["resources", "upload", str(file), "tts/notes.txt"], expect_exit=True)
        err = capsys.readouterr().err
        assert "HTTP 403" in err
        assert "403 Forbidden - write denied" in err
        assert "MCP request" not in err

    def test_html_error_body_is_flattened(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An error page is cut down to just its message."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        file = tmp_path / "notes.txt"
        file.write_bytes(b"x")
        body = (
            "<!DOCTYPE html><html><head><title>403 Forbidden</title></head>"
            "<body><h1>403 Forbidden</h1>"
            "<p>403 Forbidden: Access denied to the specified resource</p>"
            "<hr/><a href='x'>WsgiDAV/4.3.5</a></body></html>"
        )
        with _mocked_upload() as (_, client):
            client.put = AsyncMock(return_value=MagicMock(status_code=403, text=body))
            _run_main(["resources", "upload", str(file), "tts/notes.txt"], expect_exit=True)
        err = capsys.readouterr().err
        assert "403 Forbidden: Access denied to the specified resource" in err
        assert "DOCTYPE" not in err


class TestResourcesDelete:
    """Test that delete removes the file at the given share path."""

    def test_deletes_and_prints_path(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A delete that works prints the resource path."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        with _mocked_upload() as (_, client):
            _run_main(["resources", "delete", "upload/notes.txt"])
        client.delete.assert_awaited_once_with("http://127.0.0.1:8315/dav/upload/notes.txt")
        client.put.assert_not_awaited()
        assert capsys.readouterr().out == "upload/notes.txt\n"

    def test_backend_error_is_delegated(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A failed delete exits 1 and shows the status and the error text."""
        monkeypatch.setenv("MCP_URL", "http://127.0.0.1:8315/mcp")
        with _mocked_upload() as (_, client):
            client.delete = AsyncMock(
                return_value=MagicMock(status_code=404, text="404 Not Found - no such resource")
            )
            _run_main(["resources", "delete", "upload/missing.txt"], expect_exit=True)
        err = capsys.readouterr().err
        assert "HTTP 404" in err
        assert "404 Not Found - no such resource" in err
        assert "MCP request" not in err


class TestCliLive:
    """Tests that run the command line against the real server.

    They are skipped when the server does not answer or when the agent
    container is not running.
    """

    pytestmark: list[pytest.MarkDecorator] = [
        pytest.mark.live,
        pytest.mark.timeout(_LIVE_TIMEOUT),
        pytest.mark.skipif(not _server_reachable(), reason="live agent server unreachable"),
        pytest.mark.skipif(
            not _docker_available(), reason="docker command or Agent Hub container not available"
        ),
    ]

    def test_tools_list_prints_tools_with_summaries(self) -> None:
        """The tool list prints each tool name with its summary under it."""
        result = _run_cli("tools", "list")
        assert result.returncode == 0
        lines = result.stdout.splitlines()
        assert "github" in lines
        assert lines[lines.index("github") + 1].startswith("  GitHub operations")

    def test_tools_show_prints_full_documentation(self) -> None:
        """Showing a tool prints its name, what it does, and its parameters."""
        result = _run_cli("tools", "show", "github")
        assert result.returncode == 0
        assert result.stdout.startswith("github\n\nGitHub operations")
        assert "--action <string>  [required]" in result.stdout
        assert "--limit <integer>  [optional, default 10, minimum 1]" in result.stdout

    def test_tools_show_does_not_leak_constant_names(self) -> None:
        """The parameter help shows real numbers, not the names they are stored under."""
        result = _run_cli("tools", "show", "web_fetch")
        assert result.returncode == 0
        assert "DEFAULT_TOOL_OUTPUT_LEN_LIMIT" not in result.stdout
        assert "MIN_TOOL_OUTPUT_LEN_LIMIT" not in result.stdout
        assert "default 25000" in result.stdout
        assert "minimum 500" in result.stdout

    def test_tools_call_with_eq_params(self) -> None:
        """A call with --name=value pairs works and prints real results."""
        result = _run_cli(
            "tools", "call", "github", "--action=search_repos", "--query=fastmcp", "--limit=2"
        )
        assert result.returncode == 0
        assert "fastmcp" in result.stdout

    def test_tools_call_rejects_duplicate_param(self) -> None:
        """The same parameter given twice exits 1 and prints nothing."""
        result = _run_cli(
            "tools", "call", "github", "--action=search_repos", "--limit=1", "--limit=2"
        )
        assert result.returncode == 1
        assert "duplicate parameter --limit" in result.stderr
        assert result.stdout == ""

    def test_tools_call_unknown_tool(self) -> None:
        """An unknown tool name exits 1 with a message on stderr."""
        result = _run_cli("tools", "call", "no_such_tool")
        assert result.returncode == 1
        assert "unknown tool" in result.stderr

    def test_resources_list_prints_uris(self) -> None:
        """The resource list prints every resource address."""
        result = _run_cli("resources", "list")
        assert result.returncode == 0
        assert "resources:///" in result.stdout.splitlines()

    def test_tools_call_auto_uploads_and_cleans_up_resource_path(
        self, tmp_path: Path
    ) -> None:
        """A local file is uploaded under a random name, read, then removed."""
        name = f"agent_cli_fallback_{uuid.uuid4().hex}.txt"
        payload = f"auto-upload {uuid.uuid4().hex}"
        (tmp_path / name).write_text(payload)
        result = _run_cli(
            "tools", "call", "read_media_or_doc", "--resource_path", str(tmp_path / name)
        )
        assert result.returncode == 0
        assert payload in result.stdout
        match = re.search(r"upload/([0-9a-f]{32}\.txt)", result.stderr)
        assert match, result.stderr
        destination = f"upload/{match.group(1)}"
        assert not _container_exists(f"/app/data/resources/{destination}")

        again = _run_cli(
            "tools", "call", "read_media_or_doc", "--resource_path", destination
        )
        assert again.returncode == 0
        assert "Resource not found" in again.stdout
        assert "Uploaded" not in again.stderr

    def test_resources_upload_read_delete_roundtrip(self, tmp_path: Path) -> None:
        """Upload keeps the file, read gives back its text, and delete removes it."""
        name = f"agent_cli_roundtrip_{uuid.uuid4().hex}.txt"
        payload = f"roundtrip {uuid.uuid4().hex}"
        source = tmp_path / name
        source.write_text(payload)
        destination = f"upload/{name}"
        uploaded = _run_cli("resources", "upload", str(source), destination)
        assert uploaded.returncode == 0, uploaded.stderr
        assert uploaded.stdout == f"{destination}\n"
        assert _container_exists(f"/app/data/resources/{destination}")
        read = _run_cli("resources", "read", f"resources:///{destination}")
        assert read.returncode == 0
        assert read.stdout.strip() == payload
        deleted = _run_cli("resources", "delete", destination)
        assert deleted.returncode == 0, deleted.stderr
        assert deleted.stdout == f"{destination}\n"
        assert not _container_exists(f"/app/data/resources/{destination}")

    def test_resources_upload_readonly_path_delegated(self, tmp_path: Path) -> None:
        """Writing outside the upload folder exits 1 with the server turning it down."""
        source = tmp_path / f"agent_cli_readonly_{uuid.uuid4().hex}.txt"
        source.write_text("x")
        result = _run_cli("resources", "upload", str(source), "tts/readonly.txt")
        assert result.returncode == 1
        assert "HTTP 403" in result.stderr

    def test_resources_delete_missing_delegated(self) -> None:
        """Deleting a resource that is not there exits 1 with the server turning it down."""
        result = _run_cli("resources", "delete", f"upload/missing_{uuid.uuid4().hex}.txt")
        assert result.returncode == 1
        assert "HTTP 404" in result.stderr


class TestToolTaskFlowLive:
    """Test that a long tool call gives back a task id and the task tool follows it."""

    pytestmark: list[pytest.MarkDecorator] = [
        pytest.mark.live,
        pytest.mark.timeout(600),
        pytest.mark.skipif(not _server_reachable(), reason="live agent server unreachable"),
        pytest.mark.skipif(
            not _docker_available(), reason="docker command or Agent Hub container not available"
        ),
    ]

    def test_async_pilot_ack_then_task_tool(self) -> None:
        result = _run_cli("tools", "call", "rednote_research", "--query=湾区美食", "--count=1")
        assert result.returncode == 0, result.stderr
        stdout = result.stdout
        if not stdout.startswith("Task started. task_id="):
            # The same call within a day gives back the stored report, not a task id.
            assert stdout.strip(), "a cached hit must return the stored report"
            return
        task_id = stdout.split("task_id=")[1].split(".")[0]

        # Waiting holds until the work is done, which can take minutes.
        result = _run_cli(
            "tools", "call", "task",
            "--action=result", "--task_id", task_id, "--wait", "true",
            timeout=540,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip()

        result = _run_cli("tools", "call", "task", "--action=logs", "--task_id", task_id, "--tail", "5")
        assert result.returncode == 0, result.stderr
        assert "[rednote_research]" in result.stdout

        result = _run_cli("tools", "call", "task", "--action=status", "--task_id", task_id)
        assert result.returncode == 0, result.stderr
        assert "status: completed" in result.stdout

    def test_task_unknown_id(self) -> None:
        result = _run_cli("tools", "call", "task", "--action=status", "--task_id", "no-such-task")
        assert result.returncode == 0, result.stderr
        assert "No task found with id 'no-such-task'." in result.stdout


def test_mcp_url_is_documented_and_used(monkeypatch: pytest.MonkeyPatch, host_script_vars: set[str]) -> None:
    """MCP_URL has an entry in .env.example part 3, and agent_cli connects to it."""
    assert "MCP_URL" in host_script_vars
    monkeypatch.setenv("MCP_URL", _DOCUMENTED_MCP_URL)
    with _mocked_cli(["tools", "list"]) as (_, http_client):
        main()
    http_client.assert_called_once_with(_DOCUMENTED_MCP_URL)
