"""Format inline citation links and the References section for research reports."""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from app.agents.research.models import REF_RE, CitedRun
from app.lib.tool.base import read_run_mcp_text
from app.resources import resources

__all__ = ["format_references"]

_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_SEARCH_TITLE_PATTERN = re.compile(r"1\.\s+\*\*(.+?)\*\*")
_SEARCH_URL_PATTERN = re.compile(r"URL:\s*(https?://\S+)")
_URL_PATTERN = re.compile(r"https?://\S+")


@dataclass(frozen=True)
class _ReferenceLink:
    """Resolved title and destination URL for one cited run."""

    title: str
    url: str


@dataclass(frozen=True)
class _ReferenceEntry:
    """Numbered reference item rendered in the References section."""

    number: int
    title: str
    url: str


async def format_references(draft: str, runs: Mapping[str, CitedRun]) -> str:
    """Replace reference placeholders with numbered links and append a References section.

    Args:
        draft: Report text carrying reference placeholders.
        runs: Cited run registry keyed by run id.

    Returns:
        Report text with hyperlinked inline markers and a References section.
    """
    if not REF_RE.search(draft):
        return draft.strip()
    result_texts = await _read_result_texts(REF_RE.findall(draft), runs)
    return await asyncio.to_thread(_format_references_sync, draft, runs, result_texts)


def _get_console_tools_url() -> str:
    """Return the base console URL for tool run pages."""
    return (os.getenv("CONSOLE_BASE_URL") or resources.dav_base_url()).rstrip("/") + "/tools"


async def _read_result_texts(run_ids: list[str], runs: Mapping[str, CitedRun]) -> dict[str, str]:
    """Read saved MCP result text for each cited run id."""
    texts: dict[str, str] = {}
    for run_id in dict.fromkeys(run_ids):
        cited_run = runs.get(run_id)
        if cited_run is None:
            continue
        try:
            texts[run_id] = await read_run_mcp_text(cited_run.run_dir)
        except OSError:
            texts[run_id] = ""
    return texts


def _format_references_sync(
    draft: str,
    runs: Mapping[str, CitedRun],
    result_texts: Mapping[str, str],
) -> str:
    """Replace inline placeholders and append the numbered References section."""
    entries_by_run_id: dict[str, _ReferenceEntry] = {}

    def _replace_placeholder(match: re.Match[str]) -> str:
        run_id = match.group(1)
        entry = entries_by_run_id.get(run_id)
        if entry is None:
            link = _build_reference_link(run_id, runs, result_texts)
            entry = _ReferenceEntry(
                number=len(entries_by_run_id) + 1,
                title=link.title,
                url=link.url,
            )
            entries_by_run_id[run_id] = entry
        return f"[[{entry.number}]]({entry.url})"

    formatted_draft = REF_RE.sub(_replace_placeholder, draft)
    if not entries_by_run_id:
        return formatted_draft.strip()
    lines = ["", "## References", ""]
    for entry in entries_by_run_id.values():
        if entry.title:
            lines.append(f"{entry.number}. [{entry.title}]({entry.url})")
        else:
            lines.append(f"{entry.number}. {entry.url}")
    return (formatted_draft + "\n".join(lines)).strip()


def _build_reference_link(
    run_id: str,
    runs: Mapping[str, CitedRun],
    result_texts: Mapping[str, str],
) -> _ReferenceLink:
    """Build the display title and URL for one cited run id."""
    cited_run = runs.get(run_id)
    if cited_run is None:
        return _ReferenceLink(
            title=f"Run {run_id}",
            url=f"{_get_console_tools_url()}/unknown/{run_id}",
        )
    return _build_tool_reference_link(cited_run, result_texts.get(run_id, ""))


def _build_tool_reference_link(cited_run: CitedRun, result_text: str) -> _ReferenceLink:
    """Build the display title and URL for a known cited tool run."""
    tool_name = cited_run.tool_name
    console_url = f"{_get_console_tools_url()}/{tool_name}/{cited_run.run_id}"
    if tool_name in ("brave_search", "google_search"):
        return _build_search_reference_link(cited_run.params, result_text, console_url)
    if tool_name == "web_fetch":
        return _build_web_fetch_reference_link(cited_run.params, result_text, console_url)
    if tool_name in ("x_search", "rednote_research"):
        return _build_x_or_rednote_reference_link(tool_name, cited_run.params, result_text, console_url)
    if tool_name == "github":
        return _build_github_reference_link(cited_run.params, result_text, console_url)
    return _ReferenceLink(title=f"Run: {tool_name}", url=console_url)


def _build_search_reference_link(params: dict, result_text: str, console_url: str) -> _ReferenceLink:
    """Build a reference link from search parameters and result text."""
    query = str(params.get("query", ""))
    fallback_title = f"Search: {query}" if query else "Search"
    title_match = _SEARCH_TITLE_PATTERN.search(result_text)
    url_match = _SEARCH_URL_PATTERN.search(result_text)
    title = title_match.group(1) if title_match else fallback_title
    url = url_match.group(1) if url_match else console_url
    return _ReferenceLink(title=title, url=url)


def _build_web_fetch_reference_link(params: dict, result_text: str, console_url: str) -> _ReferenceLink:
    """Build a reference link from web fetch parameters and page text."""
    url = str(params.get("url", ""))
    heading = _extract_first_heading(result_text)
    fallback_title = f"Web fetch of {url}" if url else "Web fetch"
    return _ReferenceLink(title=heading or fallback_title, url=url or console_url)


def _build_x_or_rednote_reference_link(
    tool_name: str,
    params: dict,
    result_text: str,
    console_url: str,
) -> _ReferenceLink:
    """Build a reference link for an X search or Rednote research run."""
    query = str(params.get("query", ""))
    label = "X search" if tool_name == "x_search" else "Rednote search"
    title = f"{label}: {query}" if query else label
    url_match = _URL_PATTERN.search(result_text)
    url = url_match.group(0) if url_match else console_url
    return _ReferenceLink(title=title, url=url)


def _build_github_reference_link(params: dict, result_text: str, console_url: str) -> _ReferenceLink:
    """Build a reference link for a GitHub tool run."""
    action = str(params.get("action", ""))
    repo = str(params.get("repo", ""))
    number = str(params.get("number", ""))
    path = str(params.get("path", ""))
    git_ref = str(params.get("ref", ""))
    if action == "view_issue":
        title = _extract_first_heading(result_text) or f"GitHub {_format_github_subject(params)}"
        return _ReferenceLink(title=title, url=f"https://github.com/{repo}/issues/{number}")
    if action == "view_pr":
        title = _extract_first_heading(result_text) or f"GitHub {_format_github_subject(params)}"
        return _ReferenceLink(title=title, url=f"https://github.com/{repo}/pull/{number}")
    if action == "view_file":
        file_path = f"{repo}/{path}" if path else repo
        title = f"{file_path}@{git_ref}" if git_ref else file_path
        url = f"https://github.com/{repo}/blob/{git_ref}/{path}" if git_ref else f"https://github.com/{repo}"
        return _ReferenceLink(title=title, url=url)
    if action in ("list_issues", "search_issues"):
        return _ReferenceLink(title=f"{action} on {repo}", url=f"https://github.com/{repo}/issues")
    return _ReferenceLink(title=_format_github_subject(params), url=console_url)


def _format_github_subject(params: dict) -> str:
    """Format a short label describing a GitHub action and its target."""
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
        subject = " ".join(part for part in (repo, number, query) if part)
    return f"{action} {subject}".strip()


def _extract_first_heading(text: str) -> str:
    """Return the first Markdown heading in the text, or an empty string."""
    match = _HEADING_PATTERN.search(text)
    return match.group(1).strip() if match else ""
