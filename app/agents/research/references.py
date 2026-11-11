"""Reference post-processing for the research agent report."""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Mapping

from app.agents.research.models import REF_RE, CitedRun
from app.lib.tool.base import read_run_mcp_text
from app.resources import resources

__all__ = ["process_references"]

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_SEARCH_TITLE_RE = re.compile(r"1\.\s+\*\*(.+?)\*\*")
_SEARCH_URL_RE = re.compile(r"URL:\s*(https?://\S+)")
_LINK_RE = re.compile(r"https?://\S+")


async def process_references(draft: str, runs: Mapping[str, CitedRun]) -> str:
    """Replace reference placeholders with numbered links and append a References section.

    Args:
        draft: The report text carrying [[ref:<run_id>]] placeholders.
        runs: The run registry keyed by run id.

    Returns:
        The report with hyperlinked inline markers and a References section.
    """
    if not REF_RE.search(draft):
        return draft.strip()
    result_texts = await _read_result_texts(REF_RE.findall(draft), runs)
    return await asyncio.to_thread(_process_references_sync, draft, runs, result_texts)


def _console_tools_url() -> str:
    return (os.getenv("CONSOLE_BASE_URL") or resources.dav_base_url()).rstrip("/") + "/tools"


async def _read_result_texts(run_ids: list[str], runs: Mapping[str, CitedRun]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for run_id in dict.fromkeys(run_ids):
        ref = runs.get(run_id)
        if ref is None:
            continue
        try:
            texts[run_id] = await read_run_mcp_text(ref.run_dir)
        except OSError:
            texts[run_id] = ""
    return texts


def _process_references_sync(
    draft: str, runs: Mapping[str, CitedRun], result_texts: Mapping[str, str]
) -> str:
    markers: dict[str, tuple[int, str]] = {}
    entries: list[tuple[int, str, str]] = []

    def _replace(match: re.Match) -> str:
        run_id = match.group(1)
        if run_id not in markers:
            num = len(markers) + 1
            title, url = _resolve_reference(run_id, runs, result_texts)
            markers[run_id] = (num, url)
            entries.append((num, title, url))
        num, url = markers[run_id]
        return f"[[{num}]]({url})"

    processed = REF_RE.sub(_replace, draft)
    if not entries:
        return processed.strip()
    lines = ["", "## References", ""]
    for num, title, url in entries:
        lines.append(f"{num}. [{title}]({url})" if title else f"{num}. {url}")
    return (processed + "\n".join(lines)).strip()


def _resolve_reference(
    run_id: str, runs: Mapping[str, CitedRun], result_texts: Mapping[str, str]
) -> tuple[str, str]:
    ref = runs.get(run_id)
    if ref is None:
        return f"Run {run_id}", f"{_console_tools_url()}/unknown/{run_id}"
    return _derive_title_link(ref, result_texts.get(run_id, ""))


def _derive_title_link(ref: CitedRun, result_text: str) -> tuple[str, str]:
    tool = ref.tool_name
    console_url = f"{_console_tools_url()}/{tool}/{ref.run_id}"
    if tool in ("brave_search", "google_search"):
        return _derive_search(ref.params, result_text, console_url)
    if tool == "web_fetch":
        return _derive_web_fetch(ref.params, result_text, console_url)
    if tool in ("x_search", "rednote_research"):
        return _derive_x_rednote(tool, ref.params, result_text, console_url)
    if tool == "github":
        return _derive_github(ref.params, result_text, console_url)
    return f"Run: {tool}", console_url


def _derive_search(params: dict, result_text: str, console_url: str) -> tuple[str, str]:
    query = str(params.get("query", ""))
    fallback_title = f"Search: {query}" if query else "Search"
    title_match = _SEARCH_TITLE_RE.search(result_text)
    url_match = _SEARCH_URL_RE.search(result_text)
    title = title_match.group(1) if title_match else fallback_title
    url = url_match.group(1) if url_match else console_url
    return title, url


def _derive_web_fetch(params: dict, result_text: str, console_url: str) -> tuple[str, str]:
    url = str(params.get("url", ""))
    heading = _first_heading(result_text)
    fallback_title = f"Web fetch of {url}" if url else "Web fetch"
    title = heading or fallback_title
    return title, url or console_url


def _derive_x_rednote(
    tool: str, params: dict, result_text: str, console_url: str
) -> tuple[str, str]:
    query = str(params.get("query", ""))
    label = "X search" if tool == "x_search" else "Rednote search"
    title = f"{label}: {query}" if query else label
    link_match = _LINK_RE.search(result_text)
    url = link_match.group(0) if link_match else console_url
    return title, url


def _derive_github(params: dict, result_text: str, console_url: str) -> tuple[str, str]:
    action = str(params.get("action", ""))
    repo = str(params.get("repo", ""))
    number = str(params.get("number", ""))
    path = str(params.get("path", ""))
    ref = str(params.get("ref", ""))
    if action == "view_issue":
        title = _first_heading(result_text) or f"GitHub {_github_subject(params)}"
        return title, f"https://github.com/{repo}/issues/{number}"
    if action == "view_pr":
        title = _first_heading(result_text) or f"GitHub {_github_subject(params)}"
        return title, f"https://github.com/{repo}/pull/{number}"
    if action == "view_file":
        file_path = f"{repo}/{path}" if path else repo
        title = f"{file_path}@{ref}" if ref else file_path
        if ref:
            url = f"https://github.com/{repo}/blob/{ref}/{path}"
        else:
            url = f"https://github.com/{repo}"
        return title, url
    if action in ("list_issues", "search_issues"):
        return f"{action} on {repo}", f"https://github.com/{repo}/issues"
    return _github_subject(params), console_url


def _github_subject(params: dict) -> str:
    action = str(params.get("action", ""))
    repo = str(params.get("repo", ""))
    number = str(params.get("number", ""))
    query = str(params.get("query", ""))
    if action in ("view_issue", "view_pr"):
        subject = f"{repo}#{number}"
    elif action == "search_code":
        subject = f"{repo} {query}".strip()
    elif action == "search_repos":
        subject = query
    elif action == "list_releases":
        subject = repo
    else:
        subject = " ".join(v for v in (repo, number, query) if v)
    return f"{action} {subject}".strip()


def _first_heading(text: str) -> str:
    match = _HEADING_RE.search(text)
    return match.group(1).strip() if match else ""
