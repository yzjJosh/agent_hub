from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import uuid
from collections.abc import Awaitable, Callable
from typing import ClassVar, override

from app.lib.cli.cli_runner import run_cli
from app.lib.shared.paths import resources_dir, scratch_dir
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult
from app.resources.resources import binary_download_notice

__all__ = ["github_tool"]

_BINARY_DIR = "github"
_SCRATCH_NAME = "github"

_ACTIONS: tuple[str, ...] = (
    "list_issues", "search_issues", "view_issue", "search_repos", "view_pr",
    "search_code", "view_file", "list_releases",
)

_REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "list_issues": ("repo", "query"),
    "search_issues": ("repo", "query"),
    "view_issue": ("repo", "number"),
    "search_repos": ("query",),
    "view_pr": ("repo", "number"),
    "search_code": ("repo", "query"),
    "view_file": ("repo", "path"),
    "list_releases": ("repo",),
}

_ACTION_PARAMS: dict[str, tuple[str, ...]] = {
    "list_issues": ("repo", "query", "limit", "state"),
    "search_issues": ("repo", "query", "limit", "state"),
    "view_issue": ("repo", "number"),
    "search_repos": ("query", "limit"),
    "view_pr": ("repo", "number"),
    "search_code": ("repo", "query", "limit"),
    "view_file": ("repo", "path", "ref"),
    "list_releases": ("repo", "limit"),
}

_VALID_STATES: dict[str, tuple[str, ...]] = {
    "list_issues": ("open", "closed", "all"),
    "search_issues": ("open", "closed"),
}


async def _run(args: list[str], label: str, error_prefix: str, empty_message: str) -> str:
    """Run a GitHub command and return what it printed.

    Returns:
        What the command printed, or the empty message when it printed nothing.

    Raises:
        RuntimeError: If the command fails.
    """
    result = await run_cli(args, label)
    if result.returncode != 0:
        raise RuntimeError(f"{error_prefix}{result.stderr or 'Unknown'}")
    return result.stdout or empty_message


async def _list_issues(repo: str, query: str, limit: int, state: str) -> str:
    args = ["gh", "issue", "list", "-R", repo, "-S", query, "--limit", str(limit), "--state", state]
    return await _run(args, "GitHub issue list", "Error listing GitHub issues: ",
                      f"No issues found in {repo} matching '{query}'.")


async def _search_issues(repo: str, query: str, limit: int, state: str) -> str:
    args = ["gh", "search", "issues", query, "--repo", repo, "--limit", str(limit), "--state", state]
    return await _run(args, "GitHub issue search", "Error searching GitHub issues: ",
                      f"No issues found in {repo} matching '{query}'.")


async def _view_issue(repo: str, number: int) -> str:
    args = ["gh", "issue", "view", str(number), "-R", repo, "--comments"]
    return await _run(args, "GitHub issue view", "Error viewing GitHub issue: ",
                      f"Issue #{number} not found in {repo}.")


async def _search_repos(query: str, limit: int) -> str:
    args = ["gh", "search", "repos", query, "--limit", str(limit)]
    return await _run(args, "GitHub repo search", "Error searching GitHub repositories: ",
                      f"No repositories found matching '{query}'.")


async def _view_pr(repo: str, number: int) -> str:
    args = ["gh", "pr", "view", str(number), "-R", repo]
    return await _run(args, "GitHub PR view", "Error viewing GitHub pull request: ",
                      f"Pull request #{number} not found in {repo}.")


async def _search_code(repo: str, query: str, limit: int) -> str:
    args = ["gh", "search", "code", query, "--repo", repo, "--limit", str(limit)]
    return await _run(args, "GitHub code search", "Error searching GitHub code: ",
                      f"No code found in {repo} matching '{query}'.")


async def _list_releases(repo: str, limit: int) -> str:
    args = ["gh", "release", "list", "-R", repo, "--limit", str(limit)]
    return await _run(args, "GitHub release list", "Error listing GitHub releases: ",
                      f"No releases found in {repo}.")


def _is_binary(raw: bytes) -> bool:
    """Return True if the bytes are a binary file."""
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return b"\x00" in raw


def _share_path(filename: str) -> str:
    """Return where a file goes inside this server's share."""
    return os.path.join(resources_dir(), _BINARY_DIR, filename)


def _save_binary(raw: bytes, filename: str) -> str:
    """Save a binary file into this server's share.

    Returns:
        The path of the saved copy relative to the resource root.
    """
    dest = _share_path(filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(raw)
    return f"{_BINARY_DIR}/{filename}"


def _move_to_share(tmp_path: str, filename: str) -> str:
    """Move a file into this server's share.

    Returns:
        The path of the moved copy relative to the resource root.
    """
    dest = _share_path(filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(tmp_path, dest)
    return f"{_BINARY_DIR}/{filename}"


def _render_text(raw: bytes, repo: str, path: str, ref_name: str) -> str | None:
    """Return the file's text with a heading line on top, or None when the file is binary."""
    if _is_binary(raw):
        return None
    return f"# {repo}/{path} @ {ref_name}\n{raw.decode('utf-8')}".rstrip()


async def _render_blob(raw: bytes, repo: str, path: str, ref_name: str) -> str | ToolResult:
    """Return a file's text with a heading, or save it and say how to download it.

    Returns:
        The file's text, or a download note when the file is binary.
    """
    text = await asyncio.to_thread(_render_text, raw, repo, path, ref_name)
    if text is not None:
        return text
    share_rel = await asyncio.to_thread(_save_binary, raw, os.path.basename(path))
    return ToolResult(text=await binary_download_notice(share_rel), files=(share_rel,))


def _stage_blob(stdout: str) -> str:
    """Save the file bytes from a blob answer to a temporary file.

    Returns:
        The path of the temporary file. The caller must delete it.

    Raises:
        ValueError: If the answer carries no file content.
    """
    data = json.loads(stdout)
    if data.get("encoding") != "base64" or not data.get("content"):
        raise ValueError("blob content not returned by the API")
    raw = base64.b64decode("".join(data["content"].splitlines()))
    tmp_dir = scratch_dir(_SCRATCH_NAME)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"github-view-{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(raw)
    except OSError:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return tmp_path


async def _download_blob(repo: str, sha: str) -> str:
    """Download a file's bytes from GitHub.

    Returns:
        The path of the temporary file with the bytes. The caller must delete it.

    Raises:
        ValueError: If the download fails or the answer cannot be used.
    """
    args = ["gh", "api", f"repos/{repo}/git/blobs/{sha}"]
    result = await run_cli(args, "GitHub blob download")
    if result.returncode != 0:
        raise ValueError(f"blob download failed: {result.stderr or 'Unknown'}")
    return await asyncio.to_thread(_stage_blob, result.stdout)


def _read_staged(tmp_path: str) -> bytes:
    """Read the bytes of a staged file."""
    with open(tmp_path, "rb") as f:
        return f.read()


def _drop_staged(tmp_path: str) -> None:
    """Remove a staged file, and do nothing when it is already gone."""
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


async def _render_staged(tmp_path: str, repo: str, path: str, ref_name: str) -> str | ToolResult:
    """Return a staged file's text with a heading, or a download note when it is binary.

    A binary file is moved into this server's share. The staged file is removed.
    """
    try:
        raw = await asyncio.to_thread(_read_staged, tmp_path)
        text = await asyncio.to_thread(_render_text, raw, repo, path, ref_name)
        if text is not None:
            return text
        share_rel = await asyncio.to_thread(_move_to_share, tmp_path, os.path.basename(path))
        return ToolResult(text=await binary_download_notice(share_rel), files=(share_rel,))
    finally:
        await asyncio.to_thread(_drop_staged, tmp_path)


def _parse_contents(stdout: str) -> tuple[list | dict, bytes | None]:
    """Read the answer that describes a file or a folder.

    Returns:
        The answer, and the file bytes when the answer carries them.
    """
    data = json.loads(stdout)
    if isinstance(data, list):
        return data, None
    content = data.get("content")
    if not content:
        # GitHub sends an empty string both for an empty file and for a file it did
        # not include. A size of 0 means the file is empty.
        if data.get("size") == 0:
            return data, b""
        return data, None
    return data, base64.b64decode(content)


async def _view_file(repo: str, path: str, ref: str | None) -> str | ToolResult:
    """Read a file, or list a folder, in a repo.

    Returns:
        The file's text, a download note for a binary file, or one line per
        entry when the path is a folder.

    Raises:
        RuntimeError: If the file could not be read.
    """
    endpoint = f"repos/{repo}/contents/{path}"
    if ref:
        endpoint = f"{endpoint}?ref={ref}"
    args = ["gh", "api", endpoint]
    result = await run_cli(args, "GitHub file view")
    if result.returncode != 0:
        raise RuntimeError(f"Error viewing GitHub file: {result.stderr or 'Unknown'}")
    try:
        data, raw = await asyncio.to_thread(_parse_contents, result.stdout)
        if isinstance(data, list):
            return "\n".join(f"{entry['type']}: {entry['path']}" for entry in data).strip()
        ref_name = data.get("ref") or ref or "default branch"
        if raw is not None:
            return await _render_blob(raw, repo, path, ref_name)
        sha = data.get("sha")
        if not sha:
            raise RuntimeError("Error viewing GitHub file: content not returned by the API")
        tmp_path = await _download_blob(repo, str(sha))
        return await _render_staged(tmp_path, repo, path, ref_name)
    except (ValueError, KeyError, OSError) as exc:
        raise RuntimeError(f"Error viewing GitHub file: {exc}") from exc


_HANDLERS: dict[str, Callable[..., Awaitable[str | ToolResult]]] = {
    "list_issues": _list_issues,
    "search_issues": _search_issues,
    "view_issue": _view_issue,
    "search_repos": _search_repos,
    "view_pr": _view_pr,
    "search_code": _search_code,
    "view_file": _view_file,
    "list_releases": _list_releases,
}


def _format_state_options(states: tuple[str, ...]) -> str:
    """Return the allowed states as one readable list."""
    quoted = [f"'{state}'" for state in states]
    if len(quoted) == 2:
        return f"{quoted[0]} or {quoted[1]}"
    return f"{', '.join(quoted[:-1])}, or {quoted[-1]}"


def _is_missing(value: str | int | None) -> bool:
    """Return True if a value is empty or was not given."""
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def _validate(
    action: str,
    repo: str | None,
    query: str | None,
    number: int | None,
    path: str | None,
    state: str,
) -> str | None:
    """Return the error text for bad inputs, or None when they are fine."""
    candidates: dict[str, str | int | None] = {
        "repo": repo, "query": query, "number": number, "path": path,
    }
    missing = [name for name in _REQUIRED_PARAMS.get(action, ()) if _is_missing(candidates[name])]
    if missing:
        return f"github({action}) requires {', '.join(missing)}"

    params = _ACTION_PARAMS.get(action, ())
    if "state" in params and state not in _VALID_STATES[action]:
        return f"Error: state must be {_format_state_options(_VALID_STATES[action])} for {action}."
    return None


class _GithubTool(Tool):
    name: ClassVar[str] = "github"
    is_result_long: ClassVar[bool] = True
    description: ClassVar[str] = (
        "GitHub operations: issues, pull requests, repos, code, files, and releases.\n"
        "Call with one action plus the parameters it requires.\n"
        "\n"
        "Actions:\n"
        "  list_issues    List recent issues in one repo matching a query.\n"
        "  search_issues  Full-text search issues within one repo.\n"
        "  view_issue     Full issue body, metadata, and comments.\n"
        "  search_repos   Find repositories by name, description, or topic.\n"
        "  view_pr        Pull request body, status, and metadata.\n"
        "  search_code    Search code within one repository.\n"
        "  view_file      File content (or directory listing) at a path in a repo.\n"
        "                 limits: files up to 100MB\n"
        "  list_releases  List releases (newest first) in a repo, incl. tag and date.\n"
        "\n"
        "Parameters not applicable to an action are ignored."
    )
    summary: ClassVar[str] = "GitHub issues, PRs, code & repos (gh)"
    params: ClassVar[tuple[Param, ...]] = (
        Param(
            "action",
            str,
            enum=_ACTIONS,
            description="Operation to perform.",
        ),
        Param(
            "repo",
            str | None,
            None,
            description=(
                "Repository in owner/repo form (e.g. jlowin/fastmcp). Required by: "
                "list_issues, search_issues, view_issue, view_pr, search_code, view_file, "
                "list_releases. Not used by: search_repos."
            ),
        ),
        Param(
            "query",
            str | None,
            None,
            description=(
                "Search query string. Required by: list_issues, search_issues,\n"
                "search_repos, search_code. Not used by: view_issue, view_pr,\n"
                "view_file, list_releases."
            ),
        ),
        Param(
            "number",
            int | None,
            None,
            min=1,
            description=(
                "Issue or pull-request number. Required by: view_issue, view_pr. "
                "Not used by: list_issues, search_issues, search_repos, search_code, "
                "view_file, list_releases."
            ),
        ),
        Param(
            "path",
            str | None,
            None,
            description=(
                "Repository-relative path (e.g. src/main.py). Required by: view_file. "
                "Not used by: list_issues, search_issues, view_issue, search_repos, "
                "view_pr, search_code, list_releases."
            ),
        ),
        Param(
            "ref",
            str | None,
            None,
            description=(
                "Branch, tag, or commit to read from. Defaults to the repo's\n"
                "default branch. Used by: view_file. Not used by: list_issues,\n"
                "search_issues, view_issue, search_repos, view_pr, search_code,\n"
                "list_releases."
            ),
        ),
        Param(
            "limit",
            int,
            10,
            min=1,
            description=(
                "Maximum number of results to return. Used by: list_issues, "
                "search_issues, search_repos, search_code, list_releases. Not used "
                "by: view_issue, view_pr, view_file."
            ),
        ),
        Param(
            "state",
            str,
            "open",
            description=(
                "Issue state filter. Used by: list_issues ('open'|'closed'|'all'), "
                "search_issues ('open'|'closed'). Not used by: search_repos, view_issue, "
                "view_pr, search_code, view_file, list_releases."
            ),
        ),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        action = params["action"]
        repo = params.get("repo", "")
        query = params.get("query", "")
        number = params.get("number")
        path = params.get("path", "")
        ref = params.get("ref")
        limit = params.get("limit", 10)
        state = params.get("state", "open")

        norm = {
            "repo": repo.strip() if repo else None,
            "query": query.strip() if query else None,
            "number": number,
            "path": path.strip() if path else None,
            "ref": ref.strip() if ref and ref.strip() else None,
            "limit": limit,
            "state": state,
        }
        error = _validate(action, norm["repo"], norm["query"], norm["number"], norm["path"], norm["state"])
        if error is not None:
            raise ToolMessageError(error)
        if await asyncio.to_thread(shutil.which, "gh") is None:
            raise ToolMessageError("Error: gh CLI not found.")

        values: dict[str, str | int | None] = norm
        handler = _HANDLERS[action]
        out = await handler(**{name: values[name] for name in _ACTION_PARAMS[action]})
        return out if isinstance(out, ToolResult) else ToolResult(text=out)


github_tool = _GithubTool()
