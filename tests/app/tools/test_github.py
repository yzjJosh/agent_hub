from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol
from unittest.mock import AsyncMock, patch

import pytest

from app.lib.task.task_manager import AsyncTask
from app.lib.tool.base import ToolMessageError
from app.tools.github import github_tool
from tests.app.lib.tool.conftest import assert_max_char_wire_contract, run_text


class _TaskStore(Protocol):
    """The task record store the tool_infra fixture gives to a test."""


def _ok(stdout: str = "ok") -> SimpleNamespace:
    """A command result that means success."""
    return SimpleNamespace(stdout=stdout, stderr="", returncode=0)


_FILE_CONTENT = "print('hello')\nprint('world')\n"
_LARGE_FILE_CHARS = 600
_FILE_JSON = json.dumps(
    {
        "name": "main.py",
        "path": "src/main.py",
        "content": base64.b64encode(_FILE_CONTENT.encode()).decode(),
        "encoding": "base64",
        "ref": "main",
    }
)
_FILE_JSON_NO_REF = json.dumps(
    {
        "name": "main.py",
        "path": "src/main.py",
        "content": base64.b64encode(_FILE_CONTENT.encode()).decode(),
        "encoding": "base64",
    }
)
_DIR_JSON = json.dumps(
    [
        {"name": "a.py", "path": "src/a.py", "type": "file"},
        {"name": "sub", "path": "src/sub", "type": "dir"},
    ]
)


class TestGithub:

    @pytest.mark.parametrize(
        ("kwargs", "expected_argv", "expected_label", "stdout", "expected"),
        [
            (
                {"action": "list_issues", "repo": "owner/repo", "query": "bug"},
                ["gh", "issue", "list", "-R", "owner/repo", "-S", "bug", "--limit", "10", "--state", "open"],
                "GitHub issue list",
                "ok",
                "ok",
            ),
            (
                {"action": "search_issues", "repo": "owner/repo", "query": "bug"},
                ["gh", "search", "issues", "bug", "--repo", "owner/repo", "--limit", "10", "--state", "open"],
                "GitHub issue search",
                "ok",
                "ok",
            ),
            (
                {"action": "view_issue", "repo": "owner/repo", "number": 12},
                ["gh", "issue", "view", "12", "-R", "owner/repo", "--comments"],
                "GitHub issue view",
                "ok",
                "ok",
            ),
            (
                {"action": "search_repos", "query": "fastmcp"},
                ["gh", "search", "repos", "fastmcp", "--limit", "10"],
                "GitHub repo search",
                "ok",
                "ok",
            ),
            (
                {"action": "view_pr", "repo": "owner/repo", "number": 7},
                ["gh", "pr", "view", "7", "-R", "owner/repo"],
                "GitHub PR view",
                "ok",
                "ok",
            ),
            (
                {"action": "search_code", "repo": "owner/repo", "query": "parser"},
                ["gh", "search", "code", "parser", "--repo", "owner/repo", "--limit", "10"],
                "GitHub code search",
                "ok",
                "ok",
            ),
            (
                {"action": "search_code", "repo": "owner/repo", "query": "parser", "limit": 3},
                ["gh", "search", "code", "parser", "--repo", "owner/repo", "--limit", "3"],
                "GitHub code search",
                "ok",
                "ok",
            ),
            (
                {"action": "list_issues", "repo": "owner/repo", "query": "bug", "limit": 5, "state": "closed"},
                ["gh", "issue", "list", "-R", "owner/repo", "-S", "bug", "--limit", "5", "--state", "closed"],
                "GitHub issue list",
                "ok",
                "ok",
            ),
            (
                {"action": "list_issues", "repo": "owner/repo", "query": "bug", "state": "all"},
                ["gh", "issue", "list", "-R", "owner/repo", "-S", "bug", "--limit", "10", "--state", "all"],
                "GitHub issue list",
                "ok",
                "ok",
            ),
            (
                {"action": "search_issues", "repo": "owner/repo", "query": "bug", "state": "closed"},
                ["gh", "search", "issues", "bug", "--repo", "owner/repo", "--limit", "10", "--state", "closed"],
                "GitHub issue search",
                "ok",
                "ok",
            ),
            (
                {"action": "list_releases", "repo": "owner/repo"},
                ["gh", "release", "list", "-R", "owner/repo", "--limit", "10"],
                "GitHub release list",
                "ok",
                "ok",
            ),
            (
                {"action": "list_releases", "repo": "owner/repo", "limit": 3},
                ["gh", "release", "list", "-R", "owner/repo", "--limit", "3"],
                "GitHub release list",
                "ok",
                "ok",
            ),
        ],
        ids=[
            "list_issues",
            "search_issues",
            "view_issue",
            "search_repos",
            "view_pr",
            "search_code",
            "search_code_limit",
            "explicit_optionals",
            "list_issues_state_all",
            "search_issues_state_closed",
            "list_releases",
            "list_releases_limit",
        ],
    )
    async def test_argv_per_action(
        self,
        kwargs: dict,
        expected_argv: list[str],
        expected_label: str,
        stdout: str,
        expected: str,
    ) -> None:
        """Each action builds the right gh command and the label for it."""
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=AsyncMock(return_value=_ok(stdout))) as cli:
            result = await run_text(github_tool, **kwargs)
        assert result == expected
        assert cli.call_args.args == (expected_argv, expected_label)

    @pytest.mark.parametrize(
        ("kwargs", "expected_argv"),
        [
            (
                {"action": "view_file", "repo": "owner/repo", "path": "src/main.py"},
                ["gh", "api", "repos/owner/repo/contents/src/main.py"],
            ),
            (
                {"action": "view_file", "repo": "owner/repo", "path": "src/main.py", "ref": "main"},
                ["gh", "api", "repos/owner/repo/contents/src/main.py?ref=main"],
            ),
            (
                {"action": "view_file", "repo": "owner/repo", "path": "src/main.py", "ref": "   "},
                ["gh", "api", "repos/owner/repo/contents/src/main.py"],
            ),
        ],
        ids=["default", "with_ref", "whitespace_ref_omitted"],
    )
    async def test_view_file_argv(self, kwargs: dict, expected_argv: list[str]) -> None:
        """Viewing a file asks for that path, and adds the ref only when one is given."""
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=AsyncMock(return_value=_ok(_FILE_JSON))) as cli:
            await run_text(github_tool, **kwargs)
        assert cli.call_args.args == (expected_argv, "GitHub file view")

    @pytest.mark.parametrize(
        ("kwargs", "expected_missing"),
        [
            ({"action": "list_issues"}, ["repo", "query"]),
            ({"action": "list_issues", "repo": "owner/repo"}, ["query"]),
            ({"action": "list_issues", "query": "bug"}, ["repo"]),
            ({"action": "search_issues"}, ["repo", "query"]),
            ({"action": "search_issues", "repo": "owner/repo"}, ["query"]),
            ({"action": "search_issues", "query": "bug"}, ["repo"]),
            ({"action": "view_issue"}, ["repo", "number"]),
            ({"action": "view_issue", "query": "bug"}, ["repo", "number"]),
            ({"action": "view_issue", "repo": "owner/repo"}, ["number"]),
            ({"action": "view_issue", "number": 12}, ["repo"]),
            ({"action": "search_repos"}, ["query"]),
            ({"action": "view_pr"}, ["repo", "number"]),
            ({"action": "view_pr", "repo": "owner/repo"}, ["number"]),
            ({"action": "view_pr", "number": 7}, ["repo"]),
            ({"action": "search_code"}, ["repo", "query"]),
            ({"action": "search_code", "repo": "owner/repo"}, ["query"]),
            ({"action": "search_code", "query": "parser"}, ["repo"]),
            ({"action": "view_file"}, ["repo", "path"]),
            ({"action": "view_file", "repo": "owner/repo"}, ["path"]),
            ({"action": "view_file", "path": "src/main.py"}, ["repo"]),
            ({"action": "list_releases"}, ["repo"]),
        ],
    )
    async def test_missing_required_params(self, kwargs: dict, expected_missing: list[str]) -> None:
        """A missing required input returns an error naming the action and what is missing."""
        result = await run_text(github_tool, **kwargs)
        assert result == f"github({kwargs['action']}) requires {', '.join(expected_missing)}"

    async def test_missing_required_params_fail_the_run(self) -> None:
        """A missing required input starts a tracked run that fails with the tool's own words."""
        task = await github_tool.run({"action": "view_file"})
        assert isinstance(task, AsyncTask)
        with pytest.raises(ToolMessageError) as exc_info:
            await task.result()
        assert str(exc_info.value) == "github(view_file) requires repo, path"

    async def test_unknown_action(self) -> None:
        """An unknown action is rejected with an error naming the actions that work."""
        result = await run_text(github_tool, "delete_issue", repo="owner/repo", query="bug")
        assert (
            result
            == (
                "Error: action: Input should be 'list_issues', 'search_issues', 'view_issue', "
                "'search_repos', 'view_pr', 'search_code', 'view_file' or 'list_releases'."
            )
        )

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            ("list_issues", "Error: state must be 'open', 'closed', or 'all' for list_issues."),
            ("search_issues", "Error: state must be 'open' or 'closed' for search_issues."),
        ],
    )
    @pytest.mark.parametrize("state", ["merged", "OPEN", ""])
    async def test_invalid_state(self, action: str, state: str, expected: str) -> None:
        """A state the action does not allow returns an error for that action."""
        result = await run_text(github_tool, action, repo="owner/repo", query="bug", state=state)
        assert result == expected

    async def test_search_issues_state_all_rejected(self) -> None:
        """Searching issues takes only open and closed, so the state all is rejected."""
        result = await run_text(github_tool, "search_issues", repo="owner/repo", query="bug", state="all")
        assert result == "Error: state must be 'open' or 'closed' for search_issues."

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"action": "list_issues", "repo": "owner/repo", "query": "bug", "limit": 0},
            {"action": "search_issues", "repo": "owner/repo", "query": "bug", "limit": -3},
            {"action": "search_repos", "query": "fastmcp", "limit": 0},
            {"action": "search_code", "repo": "owner/repo", "query": "parser", "limit": 0},
            {"action": "list_releases", "repo": "owner/repo", "limit": 0},
        ],
    )
    async def test_invalid_limit(self, kwargs: dict) -> None:
        """A limit below 1 returns an error."""
        assert await run_text(github_tool, **kwargs) == "Error: limit: Input should be greater than or equal to 1."

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"action": "view_issue", "repo": "owner/repo", "number": 0},
            {"action": "view_pr", "repo": "owner/repo", "number": 0},
        ],
    )
    async def test_invalid_number(self, kwargs: dict) -> None:
        """A number below 1 returns an error."""
        assert await run_text(github_tool, **kwargs) == "Error: number: Input should be greater than or equal to 1."

    @pytest.mark.parametrize(
        ("kwargs", "stdout", "expected"),
        [
            (
                {"action": "view_file", "repo": "owner/repo", "path": "src/main.py"},
                _FILE_JSON,
                "# owner/repo/src/main.py @ main\nprint('hello')\nprint('world')",
            ),
            (
                {"action": "view_file", "repo": "owner/repo", "path": "src"},
                _DIR_JSON,
                "file: src/a.py\ndir: src/sub",
            ),
        ],
        ids=["full", "directory"],
    )
    async def test_view_file_output(self, kwargs: dict, stdout: str, expected: str) -> None:
        """Viewing a file returns a header and its text, and a folder returns its listing."""
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=AsyncMock(return_value=_ok(stdout))):
            result = await run_text(github_tool, **kwargs)
        assert result == expected

    async def test_view_file_long_file_comes_back_whole(
        self, tool_infra: _TaskStore
    ) -> None:
        """A file longer than the default limit comes back whole, and the full text is stored raw."""
        content = "a" * _LARGE_FILE_CHARS
        stdout = json.dumps(
            {
                "name": "main.py",
                "path": "src/main.py",
                "content": base64.b64encode(content.encode()).decode(),
                "encoding": "base64",
                "ref": "main",
            }
        )
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
              patch("app.tools.github.run_cli", new=AsyncMock(return_value=_ok(stdout))):
            result = await run_text(github_tool, "view_file", repo="owner/repo", path="src/main.py")
        full = f"# owner/repo/src/main.py @ main\n{content}"
        assert result == full
        assert "[Truncated" not in result
        rows = await tool_infra.list()
        raw = open(os.path.join(tool_infra.data_dir, rows[0].run_dir, "result")).read()
        assert raw == full

    @pytest.mark.parametrize(
        ("kwargs", "expected_header"),
        [
            (
                {"action": "view_file", "repo": "owner/repo", "path": "src/main.py"},
                "# owner/repo/src/main.py @ default branch",
            ),
            (
                {"action": "view_file", "repo": "owner/repo", "path": "src/main.py", "ref": "release"},
                "# owner/repo/src/main.py @ release",
            ),
        ],
        ids=["default_branch_fallback", "ref_param_fallback"],
    )
    async def test_view_file_ref_fallback(self, kwargs: dict, expected_header: str) -> None:
        """When the answer carries no ref, the header uses the ref that was asked for, or default branch."""
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=AsyncMock(return_value=_ok(_FILE_JSON_NO_REF))):
            result = await run_text(github_tool, **kwargs)
        assert result == expected_header + "\nprint('hello')\nprint('world')"

    async def test_view_file_content_null_without_sha(self) -> None:
        """A file with no content and no blob id returns an error."""
        stdout = json.dumps(
            {"name": "big.py", "path": "src/big.py", "content": None, "encoding": "base64", "size": 2000000}
        )
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=AsyncMock(return_value=_ok(stdout))):
            result = await run_text(github_tool, "view_file", repo="owner/repo", path="src/big.py")
        assert result == (
            "Tool 'github' failed: Error viewing GitHub file: content not returned by the API"
        )

    @staticmethod
    def _chunked_blob_json(raw: bytes) -> str:
        """A blob answer whose content is split across lines."""
        b64 = base64.b64encode(raw).decode()
        mid = len(b64) // 2
        return json.dumps({"sha": "abc123", "encoding": "base64", "size": len(raw),
                           "content": f"{b64[:mid]}\n{b64[mid:]}"})

    @pytest.mark.parametrize(
        ("content", "encoding"),
        [
            (None, "base64"),
            ("", "none"),
        ],
        ids=["null_content", "gh_rendered_empty_string"],
    )
    async def test_view_file_large_text_falls_back_to_blob(
        self, content: str | None, encoding: str
    ) -> None:
        """A file with empty content but a blob id is fetched as a blob."""
        raw = b"y" * _LARGE_FILE_CHARS
        contents = json.dumps(
            {"name": "big.py", "path": "src/big.py", "content": content, "encoding": encoding,
             "ref": "main", "sha": "abc123", "size": len(raw)}
        )
        cli = AsyncMock(side_effect=[_ok(contents), _ok(self._chunked_blob_json(raw))])
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
              patch("app.tools.github.run_cli", new=cli):
            result = await run_text(github_tool, "view_file", repo="owner/repo", path="src/big.py")
        full = f"# owner/repo/src/big.py @ main\n{'y' * _LARGE_FILE_CHARS}"
        assert result == full
        assert "[Truncated" not in result
        assert [c.args for c in cli.call_args_list] == [
            (["gh", "api", "repos/owner/repo/contents/src/big.py"], "GitHub file view"),
            (["gh", "api", "repos/owner/repo/git/blobs/abc123"], "GitHub blob download"),
        ]

    async def test_view_file_large_binary_falls_back_to_blob(self, tmp_path: Path) -> None:
        """A large binary file is saved in the github folder and the reply tells how to download it."""
        raw = b"\x89PNG\r\n\x1a\n\x00\x01\x02\x03"
        contents = json.dumps(
            {"name": "big.png", "path": "src/big.png", "content": None, "encoding": "base64",
             "ref": "main", "sha": "abc123", "size": len(raw)}
        )
        env = {"RESOURCES_DIR": str(tmp_path), "DAV_BASE_URL": "http://localhost"}
        cli = AsyncMock(side_effect=[_ok(contents), _ok(self._chunked_blob_json(raw))])
        with patch.dict(os.environ, env), \
             patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=cli):
            result = await run_text(github_tool, "view_file", repo="owner/repo", path="src/big.png")
        expected = (
            f"Binary file with size {len(raw)} bytes. "
            "Please use WebDAV to download it instead:\n"
            "  curl -o /tmp/big.png http://localhost/dav/github/big.png"
        )
        assert result == expected
        assert (tmp_path / "github" / "big.png").read_bytes() == raw

    async def test_view_file_empty_file_stays_inline(self) -> None:
        """A file that is really empty returns just the header and no blob is fetched."""
        stdout = json.dumps(
            {"name": "empty.py", "path": "src/empty.py", "content": "", "encoding": "none",
             "ref": "main", "sha": "abc123", "size": 0}
        )
        cli = AsyncMock(return_value=_ok(stdout))
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=cli):
            result = await run_text(github_tool, "view_file", repo="owner/repo", path="src/empty.py")
        assert result == "# owner/repo/src/empty.py @ main"
        assert cli.call_count == 1

    async def test_view_file_blob_download_failure(self) -> None:
        """A failed blob download returns an error with the reason."""
        contents = json.dumps(
            {"name": "big.py", "path": "src/big.py", "content": None, "encoding": "base64",
             "ref": "main", "sha": "abc123", "size": 2000000}
        )
        cli = AsyncMock(side_effect=[
            _ok(contents),
            SimpleNamespace(stdout="", stderr="HTTP 502: Bad Gateway", returncode=1),
        ])
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=cli):
            result = await run_text(github_tool, "view_file", repo="owner/repo", path="src/big.py")
        assert result == (
            "Tool 'github' failed: Error viewing GitHub file: blob download failed: HTTP 502: Bad Gateway"
        )

    @pytest.mark.parametrize(
        ("raw", "path", "filename"),
        [
            (b"\x89PNG\r\n\x1a\n\x00\x01\x02\x03", "src/logo.png", "logo.png"),
            (b"hello\x00world", "src/data.bin", "data.bin"),
        ],
        ids=["non_utf8", "null_byte"],
    )
    async def test_view_file_binary_saved_and_notified(
        self, tmp_path: Path, raw: bytes, path: str, filename: str,
    ) -> None:
        """A binary file is saved in the github folder and the reply tells how to download it."""
        stdout = json.dumps(
            {
                "name": filename,
                "path": path,
                "content": base64.b64encode(raw).decode(),
                "encoding": "base64",
                "ref": "main",
            }
        )
        env = {"RESOURCES_DIR": str(tmp_path), "DAV_BASE_URL": "http://localhost"}
        with patch.dict(os.environ, env), \
             patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=AsyncMock(return_value=_ok(stdout))):
            result = await run_text(github_tool, "view_file", repo="owner/repo", path=path)
        expected = (
            f"Binary file with size {len(raw)} bytes. "
            "Please use WebDAV to download it instead:\n"
            f"  curl -o /tmp/{filename} http://localhost/dav/github/{filename}"
        )
        assert result == expected
        assert (tmp_path / "github" / filename).read_bytes() == raw

    async def test_view_file_binary_write_failure(self, tmp_path: Path) -> None:
        """A failure saving the binary file returns an error."""
        (tmp_path / "github").write_bytes(b"")
        raw = b"\x00\x01"
        stdout = json.dumps(
            {"name": "blob.bin", "path": "src/blob.bin",
             "content": base64.b64encode(raw).decode(), "encoding": "base64", "ref": "main"}
        )
        env = {"RESOURCES_DIR": str(tmp_path), "DAV_BASE_URL": "http://localhost"}
        with patch.dict(os.environ, env), \
             patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=AsyncMock(return_value=_ok(stdout))):
            result = await run_text(github_tool, "view_file", repo="owner/repo", path="src/blob.bin")
        assert result == (
            f"Tool 'github' failed: Error viewing GitHub file: "
            f"[Errno 17] File exists: '{tmp_path / 'github'}'"
        )

    @pytest.mark.parametrize(
        ("stdout", "expected"),
        [
            ("not json", "Tool 'github' failed: Error viewing GitHub file: Expecting value: line 1 column 1 (char 0)"),
            (json.dumps([{"path": "src/a.py"}]), "Tool 'github' failed: Error viewing GitHub file: 'type'"),
        ],
        ids=["invalid_json", "missing_key"],
    )
    async def test_view_file_malformed_success(self, stdout: str, expected: str) -> None:
        """Output that cannot be read returns an error."""
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=AsyncMock(return_value=_ok(stdout))):
            result = await run_text(github_tool, "view_file", repo="owner/repo", path="src/main.py")
        assert result == expected

    async def test_view_file_error(self) -> None:
        """A failed file lookup returns an error."""
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli",
                   new=AsyncMock(return_value=SimpleNamespace(stdout="", stderr="HTTP 404: Not found", returncode=1))):
            result = await run_text(github_tool, "view_file", repo="owner/repo", path="src/missing.py")
        assert result == "Tool 'github' failed: Error viewing GitHub file: HTTP 404: Not found"

    async def test_search_code_empty_message(self) -> None:
        """A code search with no output says nothing was found."""
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=AsyncMock(return_value=_ok(""))):
            result = await run_text(github_tool, "search_code", repo="owner/repo", query="parser")
        assert result == "No code found in owner/repo matching 'parser'."

    async def test_list_releases_empty_message(self) -> None:
        """A release list with no output says nothing was found."""
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=AsyncMock(return_value=_ok(""))):
            result = await run_text(github_tool, "list_releases", repo="owner/repo")
        assert result == "No releases found in owner/repo."

    async def test_view_file_non_applicable_params_ignored(self) -> None:
        """Inputs that do not apply to viewing a file are left out of the command."""
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=AsyncMock(return_value=_ok(_FILE_JSON))) as cli:
            await run_text(
                github_tool,
                "view_file", repo="owner/repo", path="src/main.py",
                query="x", limit=1, state="closed", number=3,
            )
        assert cli.call_args.args[0] == ["gh", "api", "repos/owner/repo/contents/src/main.py"]

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"action": "list_issues", "repo": "   ", "query": "bug"}, "github(list_issues) requires repo"),
            ({"action": "search_issues", "repo": "owner/repo", "query": "  "}, "github(search_issues) requires query"),
            ({"action": "search_repos", "query": "\t"}, "github(search_repos) requires query"),
            ({"action": "search_code", "repo": "owner/repo", "query": "  "}, "github(search_code) requires query"),
            ({"action": "view_file", "repo": "owner/repo", "path": "  "}, "github(view_file) requires path"),
        ],
    )
    async def test_whitespace_params_treated_as_missing(self, kwargs: dict, expected: str) -> None:
        """An input that is only spaces counts as missing."""
        assert await run_text(github_tool, **kwargs) == expected

    async def test_nonzero_exit_reports_stderr(self) -> None:
        """A failed command returns an error carrying the message from the command."""
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli",
                    new=AsyncMock(return_value=SimpleNamespace(stdout="", stderr="HTTP 404: Not found", returncode=1))):
            result = await run_text(github_tool, "list_issues", repo="owner/repo", query="bug")
        assert result == "Tool 'github' failed: Error listing GitHub issues: HTTP 404: Not found"

    async def test_nonzero_exit_empty_stderr(self) -> None:
        """A failed command with no message returns an error saying Unknown."""
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli",
                    new=AsyncMock(return_value=SimpleNamespace(stdout="partial output", stderr="", returncode=1))):
            result = await run_text(github_tool, "view_pr", repo="owner/repo", number=7)
        assert result == "Tool 'github' failed: Error viewing GitHub pull request: Unknown"

    async def test_empty_stdout_returns_empty_message(self) -> None:
        """No output says nothing was found."""
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=AsyncMock(return_value=_ok(""))):
            result = await run_text(github_tool, "search_repos", query="fastmcp")
        assert result == "No repositories found matching 'fastmcp'."

    async def test_non_applicable_params_ignored(self) -> None:
        """Inputs that do not apply to the action are left out of the command."""
        with patch("app.tools.github.shutil.which", return_value="/usr/bin/gh"), \
             patch("app.tools.github.run_cli", new=AsyncMock(return_value=_ok())) as cli:
            await run_text(github_tool, "view_pr", repo="owner/repo", number=7, query="ignored", limit=1, state="closed")
        assert cli.call_args.args[0] == ["gh", "pr", "view", "7", "-R", "owner/repo"]

    async def test_missing_binary(self) -> None:
        """A missing gh program returns an error."""
        with patch("app.tools.github.shutil.which", return_value=None):
            assert await run_text(github_tool, "list_issues", repo="owner/repo", query="bug") == "Error: gh CLI not found."


class TestMaxCharDocstring:
    """Tests for the max_char description text."""

    async def test_docstring_uses_resolved_values(self) -> None:
        """The wire gives the real default and the real smallest value, and names no constant."""
        await assert_max_char_wire_contract(github_tool, "github")
