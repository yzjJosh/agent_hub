from __future__ import annotations

"""Deep research on Xiaohongshu, also called Rednote.

Searches the site, reads the top notes, reads the pictures in them, and
writes a report.
"""

__all__ = ["rednote_research_tool"]

import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.parse
from datetime import datetime, timedelta
from typing import ClassVar, override

import aiofiles
import httpx

from app.lib.browser.browser_manager import BrowserManager
from app.lib.browser.page_snapshot import PageSnapshot
from app.lib.reason.describe_image import describe_image
from app.lib.reason.reason import reason
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult

_VERBOSE = True

# The model must answer in these shapes.
_PAGE_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok", "blocked", "loading"]},
        "reason": {"type": "string"},
    },
    "required": ["status", "reason"],
}

_NOTE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok", "blocked", "skip", "loading"]},
        "reason": {"type": "string"},
        "extracted_content": {"type": "string"},
        "note_creation_date": {"type": "string"},
    },
    "required": ["status", "reason", "extracted_content"],
}

_BASE_URL = "https://www.rednote.com"
_RETRIES = 3
_MAX_CONCURRENT_VISION_IMAGES = 4
_CLICK_RETRIES = 3
_SELECTOR_TIMEOUT_MS = 30000
_IMAGE_DOWNLOAD_TIMEOUT_S = 60
_NOTE_URL_PATTERN = r"/(?:search_result|explore)/[\w\d]+(?:\?[\w=&%\-\.]*)?"


class _ResearchBlockedError(Exception):
    """Raised when the site blocks us and a person has to step in."""


async def _llm_query(
    prompt: str, context_data: str = "", json_schema: dict | None = None
) -> dict | str | None:
    """Ask the model something.

    Args:
        prompt: What to ask.
        context_data: Extra text for the model to read.
        json_schema: The shape the answer has to take.

    Returns:
        The answer as a dict when a shape is given, as plain text when not,
        or None when the model does not answer.
    """
    response, status = await reason(
        prompt, context=context_data, json_schema=json_schema
    )
    if status != 0:
        print(f"[rednote_research] LLM analysis failed: {response}", file=sys.stderr)
        return None

    content = response.strip()
    if not json_schema:
        return content
    return _parse_json_content(content)


def _current_timestamp() -> str:
    """Return the local time now as ISO 8601 text with the time zone offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_json_content(content: str) -> dict | None:
    """Read the model's answer as JSON.

    Args:
        content: The answer text, which may be wrapped in code fences.

    Returns:
        The answer as a dict, or None when the text is not JSON.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    for fence in ("```json", "```"):
        if fence in content:
            content = content.split(fence)[1].split("```")[0].strip()
            break

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(
            f"[rednote_research] Failed to parse content as JSON: {content[:100]}...",
            file=sys.stderr,
        )
        return None


def _extract_search_results(snapshot_json: str) -> list[str]:
    """Return the note links in the search page contents, without repeats."""
    urls: list[str] = []
    try:
        matches = re.findall(_NOTE_URL_PATTERN, snapshot_json)
        urls = [f"{_BASE_URL}{url}" for url in matches]
    except Exception as e:
        print(f"[rednote_research] Error parsing snapshot JSON: {e}", file=sys.stderr)
        return []

    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


async def _check_page_status(snapshot: str, context_desc: str) -> dict | None:
    """Ask the model if the page is blocked, still loading, or ready.

    Args:
        snapshot: The page contents to look at.
        context_desc: What the page is meant to show.

    Returns:
        The status answer, or None when the model does not answer.
    """
    prompt = (
        f"Context: {context_desc}\n"
        "Analyze the provided DOM snapshot and determine the page status.\n\n"
        "CRITERIA FOR STATUS:\n"
        "- 'blocked': A global login wall, 'Sign In' modal, or site-wide CAPTCHA is visible.\n"
        "- 'loading': The page contains skeleton screens, spinning icons, or empty placeholders.\n"
        "- 'ok': The content you expected to see is visible."
    )
    result = await _llm_query(prompt, context_data=snapshot, json_schema=_PAGE_STATUS_SCHEMA)
    return result if isinstance(result, dict) else None


async def _navigate_and_snapshot(
    mgr: BrowserManager, target_id: str, url: str
) -> tuple[str | None, int]:
    """Open a link and read the page contents.

    Args:
        mgr: The browser.
        target_id: The tab to use.
        url: Where to go.

    Returns:
        The page contents as JSON text or None, plus a status code where 0
        means it worked.
    """
    print(f"[rednote_research] Navigating to: {url}", flush=True)
    try:
        await mgr.navigate(target_id, url, wait_until="networkidle", verbose=_VERBOSE)
    except Exception as e:
        print(f"[rednote_research] Navigation failed: {e}", file=sys.stderr)
        return None, 1

    snap = await mgr.snapshot(target_id, verbose=_VERBOSE)
    if snap is None:
        print("[rednote_research] Failed to capture snapshot", file=sys.stderr)
        return None, 1

    filtered = await snap.filtered_tree(verbose=_VERBOSE)
    return json.dumps(filtered, ensure_ascii=False) if filtered else None, 0


def _has_text_child(node: dict) -> bool:
    """Return True when one of the node's children holds text."""
    for child in node.get("children", []):
        if isinstance(child, dict) and child.get("text", "").strip():
            return True
    return False


def _extract_image_urls(tree: dict) -> list[str]:
    """Return the links to the pictures in a note.

    Avatars and pictures carried inside the page itself are left out.

    Args:
        tree: The contents of the open note.

    Returns:
        The picture links, without repeats.
    """
    seen: set[str] = set()
    urls: list[str] = []
    image_count_from_tree: int | None = None
    for child in tree.get("children", []):
        if not isinstance(child, dict):
            continue
        text = (child.get("text") or "").strip()
        match = re.match(r"^(\d+)/(\d+)$", text)
        if match:
            image_count_from_tree = int(match.group(2))
        if child.get("role") == "main":
            if _has_text_child(child):
                break
        if child.get("role") == "image":
            src = child.get("src", "")
            if not src or src.startswith("data:") or "avatar" in src:
                continue
            dedup_key = src.split("//", 1)[-1].rstrip("/")
            if dedup_key not in seen:
                seen.add(dedup_key)
                urls.append(src)
    if image_count_from_tree is not None and len(urls) != image_count_from_tree:
        print(
            f"[rednote_research] Image count mismatch: extracted {len(urls)}, "
            f"tree says {image_count_from_tree}",
            file=sys.stderr,
        )
    return urls


def _remove_file(path: str) -> None:
    """Delete the file when it is still there."""
    if os.path.exists(path):
        os.unlink(path)


def _scratch_png_path(nid: str) -> str:
    """Make a fresh, unused picture file named after the note."""
    fd, path = tempfile.mkstemp(prefix=f"note_img_{nid}_", suffix=".png")
    os.close(fd)
    return path


async def _download_and_transcribe(
    img_url: str, tmp_path: str, limit: asyncio.Semaphore
) -> str | None:
    """Download a picture and turn it into text.

    Args:
        img_url: Link to the note picture.
        tmp_path: Where the picture is saved. This file is always deleted
            before returning.
        limit: Caps how many pictures are worked on at once.

    Returns:
        The text of the picture, or None when the picture cannot be read.

    Raises:
        Exception: When the download or the reading fails.
    """
    async with limit:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    img_url, timeout=_IMAGE_DOWNLOAD_TIMEOUT_S, follow_redirects=True
                )
                resp.raise_for_status()
                async with aiofiles.open(tmp_path, "wb") as f:
                    await f.write(resp.content)
            vision_raw, status = await describe_image(tmp_path)
            return vision_raw if status == 0 else None
        finally:
            await asyncio.to_thread(_remove_file, tmp_path)


def _collect_vision_results(results: list[str | None | BaseException]) -> str:
    """Keep the picture texts that worked and drop the ones that failed.

    Args:
        results: One entry per note picture, in note order. An entry is the
            picture text, None when the picture could not be read, or the
            error raised while reading it.

    Returns:
        The picture texts with labels, joined by blank lines. Empty when none
        worked.
    """
    parts: list[str] = []
    for idx, result in enumerate(results):
        if isinstance(result, BaseException):
            print(
                f"[rednote_research] Failed to transcribe image {idx + 1}: {result}",
                file=sys.stderr,
            )
            continue
        if result:
            parts.append(f"[Image {idx + 1}]: {result}")
    return "\n\n".join(parts)


async def _process_images_and_cross_ref(
    note_image_urls: list[str], nid: str, content: str
) -> str:
    """Add the text of a note's pictures to the note content.

    Args:
        note_image_urls: Links to the note pictures.
        nid: The note id.
        content: The note text.

    Returns:
        The note text with the picture texts added, or the note text as it
        was when there are no pictures or none could be read.
    """
    if not note_image_urls:
        return content

    limit = asyncio.Semaphore(_MAX_CONCURRENT_VISION_IMAGES)
    results = await asyncio.gather(
        *(
            _download_and_transcribe(img_url, _scratch_png_path(nid), limit)
            for img_url in note_image_urls
        ),
        return_exceptions=True,
    )
    vision_combined = _collect_vision_results(results)

    if vision_combined:
        return content + f"\n\n[Image Transcriptions]:\n{vision_combined}"
    return content


async def _query_llm_for_note_analysis(
    note_json: str, query: str, zh_query: str, attempt: int
) -> dict | None:
    """Ask the model to read a note.

    Args:
        note_json: The note contents as JSON text.
        query: What the user wants to know.
        zh_query: The Chinese search words.
        attempt: Which try this is.

    Returns:
        The model's answer, or None when the model does not answer.
    """
    print(
        f"[rednote_research] Attempt {attempt + 1}/{_RETRIES}: "
        "Querying LLM for note analysis...",
        flush=True,
    )
    prompt = (
        f"Context: Extract information from Xiaohongshu note for intent '{query}' "
        f"(Query: '{zh_query}').\n"
        f"Current system time: {_current_timestamp()}.\n"
        "Analyze the provided DOM snapshot, determine the page status, and extract "
        "detailed information relevant to the user intent and query.\n\n"
        "CRITERIA FOR STATUS:\n"
        "- 'blocked': A global login wall, site-wide CAPTCHA, or 'Verify your identity' "
        "screen is visible.\n"
        "- 'skip': A note-specific block (e.g., 'Sorry, This Page Isn't Available Right "
        "Now' with a QR code, 404 error, or 'Open in App' restricted view), or if this "
        "note has no relevant information.\n"
        "- 'loading': The page contains skeleton screens or placeholders.\n"
        "- 'ok': The note's title and main body text are readable, and the note is "
        "relevant to the query.\n\n"
        "INSTRUCTIONS FOR YOUR EXTRACTION:\n"
        "- Always be skeptical about relevance. When you are unsure about the relevance "
        "of a note, assume it's irrelevant and skip it.\n"
        "- Pay attention to user location. If the user location is apparently wrong for "
        "the query, you should skip that note. E.g. a China user will not have relevant "
        "information about a US local vendor or restaurant.\n"
        "- If the note has no text and only images, do not skip this note. The image "
        "content could still be relevant.\n"
        "- Preserve the entire content body and all comments. Only remove irrelevant "
        "formatting and metadata.\n"
        "- Always extract the creation date of the note as an absolute 'YYYY-MM-DD' "
        "date. Xiaohongshu often shows it relative to now ('3天前', '昨天', '2小时前') "
        "or without a year ('05-21'); resolve it against the current system time "
        "above. Leave it empty only when the note shows no date at all.\n"
        "- Don't include your own thoughts in the extracted content. Only include the "
        "content itself."
    )
    data = await _llm_query(prompt, context_data=note_json, json_schema=_NOTE_ANALYSIS_SCHEMA)
    if not isinstance(data, dict):
        print(
            f"[rednote_research] Retry {attempt + 1}/{_RETRIES}: LLM returned null",
            file=sys.stderr,
        )
        return None
    return data


def _handle_blocked_status(data: dict) -> str:
    """Work out what to do when the model says a note is blocked.

    Args:
        data: The model's answer for the note.

    Returns:
        'SKIP' when only this one note is blocked.

    Raises:
        _ResearchBlockedError: When the whole site is blocked.
    """
    reason_text = data.get("reason", "").lower()
    if "not available" in reason_text or "404" in reason_text or "scan" in reason_text:
        print(
            f"[rednote_research] Downgrading hard block to skip based on reason: "
            f"{reason_text}",
            file=sys.stderr,
        )
        return "SKIP"
    raise _ResearchBlockedError(data.get("reason", "blocked"))


async def _capture_overlay_tree(
    mgr: BrowserManager, target_id: str
) -> str | None:
    """Return the open note's contents as JSON text, or None when it cannot be read."""
    note_snap = await mgr.snapshot(target_id, verbose=_VERBOSE)
    if note_snap is None:
        return None

    full_tree = await note_snap.filtered_tree(verbose=_VERBOSE)
    if not isinstance(full_tree, dict):
        return None
    overlay_children = []
    skip_next_main = True
    for child in full_tree.get("children", []):
        if skip_next_main and isinstance(child, dict) and child.get("role") == "main":
            skip_next_main = False
            continue
        overlay_children.append(child)
    if overlay_children:
        return json.dumps(
            {"role": "overlay", "children": overlay_children}, ensure_ascii=False
        )
    return json.dumps(full_tree, ensure_ascii=False)


async def _analyze_single_note(
    mgr: BrowserManager,
    target_id: str,
    nid: str,
    full_url: str,
    query: str,
    zh_query: str,
) -> dict | str | None:
    """Read one note and pull out what it says.

    Args:
        mgr: The browser.
        target_id: The tab showing the note.
        nid: The note id.
        full_url: Link to the note.
        query: What the user wants to know.
        zh_query: The Chinese search words.

    Returns:
        The note details, 'SKIP' when the note is no use, or None when the
        note cannot be read.
    """
    for attempt in range(_RETRIES):
        note_json = await _capture_overlay_tree(mgr, target_id)
        if note_json is None:
            print(
                f"[rednote_research] Retry {attempt + 1}/{_RETRIES}: "
                "snapshot failed",
                file=sys.stderr,
            )
            continue
        overlay_tree = json.loads(note_json)
        note_image_urls = _extract_image_urls(overlay_tree)

        data = await _query_llm_for_note_analysis(note_json, query, zh_query, attempt)
        if data is None:
            continue

        status_result = _resolve_note_status(data, attempt)
        if status_result == "SKIP":
            return "SKIP"
        if status_result is None:
            continue

        content = status_result.get("extracted_content", "")
        creation_date = status_result.get("note_creation_date", "")
        content = await _process_images_and_cross_ref(note_image_urls, nid, content)
        return {
            "id": nid,
            "url": full_url,
            "analysis": content,
            "note_creation_date": creation_date,
        }

    return None


def _resolve_note_status(data: dict, attempt: int) -> dict | str | None:
    """Work out what the model's status for a note means.

    Args:
        data: The model's answer for the note.
        attempt: Which try this is.

    Returns:
        The answer itself when the note is fine, 'SKIP' when the note is no
        use, or None to try again.

    Raises:
        _ResearchBlockedError: When the whole site is blocked.
    """
    status_val = data.get("status")
    if status_val == "blocked":
        return _handle_blocked_status(data)
    if status_val == "skip":
        print(
            f"[rednote_research] Skipping note: {data.get('reason')}",
            file=sys.stderr,
        )
        return "SKIP"
    if status_val != "ok":
        print(
            f"[rednote_research] Retry {attempt + 1}/{_RETRIES}: "
            f"{data.get('reason')}",
            file=sys.stderr,
        )
        return None
    return data


def _find_link_in_tree(
    node: dict | list | str, target_path: str
) -> dict | None:
    """Find a link in the page contents whose address holds the given path."""
    if isinstance(node, dict):
        if (
            node.get("role") == "link"
            and node.get("name")
            and target_path in node.get("url", "")
        ):
            return node
        for child in node.get("children", []):
            result = _find_link_in_tree(child, target_path)
            if result:
                return result
    elif isinstance(node, list):
        for child in node:
            result = _find_link_in_tree(child, target_path)
            if result:
                return result
    return None


def _find_ref_id_for_url(
    filtered_tree: dict, actionable: list[dict], target_url: str
) -> str | None:
    """Find the thing to click to open a note.

    Returns:
        The id of the thing to click, or None when it is not on the page.
    """
    if target_url.startswith("http"):
        target_path = urllib.parse.urlparse(target_url).path
    else:
        target_path = target_url.split("?")[0]

    link_node = _find_link_in_tree(filtered_tree, target_path)
    if not link_node:
        print(
            f"[rednote_research] Note link {target_url} not found in filtered tree.",
            file=sys.stderr,
        )
        return None

    target_name = link_node.get("name", "")
    if not target_name:
        return None

    for elem in actionable:
        if elem.get("name") == target_name and elem.get("role") in ("link", "button"):
            return elem["ref_id"]
    return None


async def _analyze_notes(
    mgr: BrowserManager,
    target_id: str,
    note_urls: list[str],
    count: int,
    query: str,
    zh_query: str,
) -> list[dict]:
    """Open notes from the search results and pull out what they say.

    Args:
        mgr: The browser.
        target_id: The tab showing the search results.
        note_urls: Links to the notes.
        count: How many notes to gather.
        query: What the user wants to know.
        zh_query: The Chinese search words.

    Returns:
        The details of the notes that could be read.
    """
    results: list[dict] = []

    for note_url in note_urls:
        if len(results) >= count:
            break

        full_url = (
            f"{_BASE_URL}{note_url}" if note_url.startswith("/") else note_url
        )
        nid = note_url.split("?")[0].split("/")[-1] or "unknown_id"

        search_snap = await mgr.snapshot(target_id, verbose=_VERBOSE)
        if search_snap is None:
            print(
                f"[rednote_research] Failed to snapshot search page, skipping note {nid}",
                file=sys.stderr,
            )
            continue

        if not await _click_note(mgr, target_id, search_snap, full_url, nid):
            continue

        try:
            note_data = await _analyze_single_note(mgr, target_id, nid, full_url, query, zh_query)
            if note_data and note_data != "SKIP":
                results.append(note_data)
        finally:
            await mgr.press_key(target_id, "Escape", verbose=_VERBOSE)

    return results


async def _click_note(
    mgr: BrowserManager,
    target_id: str,
    search_snap: PageSnapshot,
    full_url: str,
    nid: str,
) -> bool:
    """Open a note from the search results.

    Args:
        mgr: The browser.
        target_id: The tab showing the search results.
        search_snap: The search page contents.
        full_url: Link to the note.
        nid: The note id.

    Returns:
        True when the note is open, False when it has to be skipped.
    """
    filtered = await search_snap.filtered_tree(verbose=_VERBOSE)
    actionable = search_snap.actionable_elements()
    if not isinstance(filtered, dict):
        return False
    ref_id = _find_ref_id_for_url(filtered, actionable, full_url)
    if ref_id is None:
        print(
            f"[rednote_research] Could not find clickable element for {full_url}, "
            "skipping",
            file=sys.stderr,
        )
        return False

    if not await _click_with_retry(search_snap, ref_id, nid):
        return False

    return await _wait_for_overlay(mgr, target_id)


async def _click_with_retry(search_snap: PageSnapshot, ref_id: str, nid: str) -> bool:
    """Click something on the page, trying a few times.

    Args:
        search_snap: The search page contents.
        ref_id: Which thing to click.
        nid: The note id.

    Returns:
        True when the click worked.
    """
    for click_attempt in range(_CLICK_RETRIES):
        if await search_snap.click(ref_id, verbose=_VERBOSE):
            return True
        print(
            f"[rednote_research] Click attempt {click_attempt + 1}/"
            f"{_CLICK_RETRIES} failed, retrying...",
            file=sys.stderr,
        )
    print(
        f"[rednote_research] Failed to click note {nid} after "
        f"{_CLICK_RETRIES} attempts, skipping",
        file=sys.stderr,
    )
    return False


async def _wait_for_overlay(mgr: BrowserManager, target_id: str) -> bool:
    """Wait until the note is open and done loading.

    Args:
        mgr: The browser.
        target_id: The tab showing the note.

    Returns:
        True when the note is ready.
    """
    if not await mgr.wait_for_selector(
        target_id,
        ':has-text("Follow")',
        timeout=_SELECTOR_TIMEOUT_MS,
        verbose=_VERBOSE,
    ):
        print(
            f"[rednote_research] Note overlay not found for {target_id}, skipping",
            file=sys.stderr,
        )
        return False

    await mgr.wait_for_selector_disappear(
        target_id,
        ':has-text("Loading...")',
        timeout=_SELECTOR_TIMEOUT_MS,
        verbose=_VERBOSE,
    )
    return True


async def _generate_search_term(query: str) -> str:
    """Ask the model for Chinese search words for a topic.

    Args:
        query: The topic to search for.

    Returns:
        The search words, or an empty string when the model does not answer.
    """
    prompt = (
        f"Generate a natural Chinese query that will be used for researching the "
        f"following topic in Xiaohongshu: '{query}'. Keep your query concise and "
        "focused. It should contain less than 10 English words and/or Chinese "
        "characters. Just return the most suitable query itself. Don't return "
        "anything else. Don't just translate the query. Use natural Chinese language "
        "as if you are a native Chinese speaker trying to find the answer from "
        "Xiaohongshu. You can mix English words on your Chinese query if the subject "
        "is commonly referred in English in Xiaohongshu. E.g. 'Google 股票', "
        "'Tesla FSD 使用体验'."
    )
    result = await _llm_query(prompt)
    return result if isinstance(result, str) and result.strip() else ""


def _build_search_url(zh_query: str) -> str:
    """Return the link to the search results page for the given words."""
    keyword = urllib.parse.quote_plus(zh_query)
    return f"{_BASE_URL}/search_result?keyword={keyword}"


def _notes_to_markdown(results: list[dict]) -> str:
    """Turn the gathered notes into a markdown document.

    Args:
        results: The notes, each with its id, link, text, and creation date.

    Returns:
        One markdown part per note.
    """
    sections: list[str] = []
    for note in results:
        title = note.get("id", "unknown")
        url = note.get("url", "")
        created = note.get("note_creation_date", "")
        analysis = note.get("analysis", "")
        sections.append(
            f"## Note: {title}\n\n"
            f"- **URL**: {url}\n"
            f"- **Created**: {created}\n\n"
            f"{analysis}"
        )
    return "\n\n---\n\n".join(sections)


async def _generate_report(results: list[dict], query: str) -> str:
    """Ask the model to write the report from the gathered notes.

    Args:
        results: The notes that were read.
        query: What the user wants to know.

    Returns:
        The report, or the notes as JSON text when the model does not answer.
    """
    print("[rednote_research] Generating final research report...", flush=True)

    prompt = (
        "You are a professional report writer. You are responsible for documenting "
        "these research findings into a cohesive report based on this user request: "
        f"'{query}'. The current system time is {_current_timestamp()}; use it to "
        "judge how recent each finding is and to call out stale information. Your "
        "report should be unbiased, fact driven, and containing full "
        "details and evidences. Make sure your report fully addresses the user request, "
        "but don't make any assumptions not directly present in the research findings. "
        "If the research findings don't contain enough information to address the user "
        "query, please acknowledge it instead of hallucinating an answer. Your response "
        "is direct user facing, so don't include anything unrelated to the user request "
        "or the research topic."
    )
    summary = await _llm_query(
        prompt, context_data=json.dumps(results, ensure_ascii=False)
    )
    if not isinstance(summary, str) or not summary.strip():
        return json.dumps(results, ensure_ascii=False)
    return summary


async def _run_research(query: str, count: int, raw_notes: bool) -> str:
    """Search Xiaohongshu and report what the notes say.

    Args:
        query: What to search for, in English or Chinese.
        count: How many notes to read.
        raw_notes: When True, give back the notes themselves instead of a
            report.

    Returns:
        The notes as markdown, or the report.

    Raises:
        RuntimeError: When the search page cannot be read, when it is blocked,
            when no notes are found, or when a person has to step in.
    """
    zh_query = await _generate_search_term(query) or query
    print(f"[rednote_research] Search Term: {zh_query}", flush=True)

    search_url = _build_search_url(zh_query)

    # Leaving this block closes the browser and every page it opened.
    async with BrowserManager(verbose=_VERBOSE) as mgr:
        target_id = await mgr.new_page(verbose=_VERBOSE)
        try:
            search_json, status = await _navigate_and_snapshot(mgr, target_id, search_url)
            if status != 0 or not search_json:
                print(
                    f"[rednote_research] Failed to capture search results: {search_json}",
                    file=sys.stderr,
                )
                raise RuntimeError("Research failed: could not capture search results.")

            status_data = await _check_page_status(
                search_json, f"Search results for '{zh_query}'"
            )
            if status_data and status_data.get("status") == "blocked":
                raise RuntimeError(f"Research blocked: {status_data.get('reason')}")

            note_urls = _extract_search_results(search_json)
            if not note_urls:
                raise RuntimeError("Research failed: no notes discovered.")

            results = await _analyze_notes(
                mgr, target_id, note_urls, count, query, zh_query
            )
            if not results:
                raise RuntimeError("Research failed: no analysis results gathered.")

            if raw_notes:
                return _notes_to_markdown(results)
            return await _generate_report(results, query)
        except _ResearchBlockedError as e:
            raise RuntimeError(f"Research blocked, human intervention required: {e}")


def _is_json(text: str) -> bool:
    """Whether the text parses as a JSON document.

    Args:
        text: The result body to test.

    Returns:
        True when the text is one whole JSON object or array, False
        otherwise.
    """
    stripped = text.lstrip()
    if not stripped.startswith(("{", "[")):
        return False
    try:
        json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return False
    return True


class _RednoteResearchTool(Tool):
    name: ClassVar[str] = "rednote_research"
    is_slow: ClassVar[bool] = True
    staleness: ClassVar[timedelta] = timedelta(hours=24)
    description: ClassVar[str] = (
        "Research a topic on Xiaohongshu (Rednote): runs a search, analyzes the "
        "top notes, and transcribes note-body images. Use for questions that "
        "benefit from Xiaohongshu user-generated content, reviews, and experiences."
    )
    summary: ClassVar[str] = "Xiaohongshu research via browser automation"
    params: ClassVar[tuple[Param, ...]] = (
        Param("query", str, description="Search query (English or Chinese)."),
        Param("count", int, 5, min=1, description="Number of notes to analyze."),
        Param(
            "raw_notes",
            bool,
            False,
            description="When True, the result is the gathered notes as markdown instead of a synthesized report.",
        ),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        query = params["query"]
        if not query.strip():
            raise ToolMessageError("query must not be empty")
        count = params.get("count", 5)
        raw_notes = params.get("raw_notes", False)
        text = await _run_research(query, count, raw_notes)
        render_type = "json" if _is_json(text) else "markdown"
        return ToolResult(text=text, render_type=render_type)

    @override
    async def cache_key(self, params: dict) -> str:
        payload = json.dumps(
            [params["query"], params.get("count", 5), params.get("raw_notes", False)],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


rednote_research_tool = _RednoteResearchTool()
