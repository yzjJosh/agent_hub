"""Look at a web page with the LLM.

Opens a page, takes a snapshot of it, and asks the LLM what it says.
"""

from __future__ import annotations

__all__ = ["BrowserError", "analyze_page"]

import json
from typing import Callable

from app.lib.browser.browser_manager import BrowserManager
from app.lib.reason.reason import reason

_LOG_PREFIX = "analyze_page"


class BrowserError(Exception):
    """Raised when a page cannot be read or analyzed."""


def _log(step: str, url: str = "") -> None:
    """Print one progress line.

    Args:
        step: What is happening now.
        url: Page address to show, if there is one.
    """
    if url:
        print(f"[{_LOG_PREFIX}] {step}: {url}", flush=True)
    else:
        print(f"[{_LOG_PREFIX}] {step}", flush=True)


async def _capture_tree(
    url: str,
    transform_snapshot: Callable[[dict], dict] | None,
    verbose: bool,
) -> tuple[dict, str]:
    """Open a page and return a tree of what is on it.

    The browser is closed before this returns, even on failure.

    Args:
        url: Page address to open.
        transform_snapshot: Changes the tree before it is returned.
        verbose: Print progress lines when True.

    Returns:
        The page tree, and the address the page landed on.

    Raises:
        BrowserError: If the page cannot be opened or read.
    """
    async with BrowserManager(verbose=verbose) as mgr:
        target_id = await mgr.new_page(verbose=verbose)
        if verbose:
            _log("Navigating to", url)
        await mgr.navigate(
            target_id,
            url,
            verbose=verbose,
        )
        if verbose:
            _log("Capturing page snapshot")
        snapshot = await mgr.snapshot(target_id, verbose=verbose)
        if snapshot is None:
            raise BrowserError("failed to capture the page snapshot")

        tree = await snapshot.filtered_tree()
        if tree is None:
            raise BrowserError("failed to parse the page tree")

        if transform_snapshot is not None:
            tree = transform_snapshot(tree)

        return tree, snapshot.url


async def analyze_page(
    url: str,
    prompt: str,
    transform_snapshot: Callable[[dict], dict] | None = None,
    output_json_schema: dict | None = None,
    verbose: bool = False,
) -> tuple[str, str]:
    """Open a page and ask the LLM about it.

    Args:
        url: Page address to open.
        prompt: What to ask the LLM about the page.
        transform_snapshot: Changes the page tree before the LLM sees it.
        output_json_schema: Shape the LLM answer should follow.
        verbose: Print progress lines when True.

    Returns:
        The answer text from the LLM, and the address the page landed on.

    Raises:
        BrowserError: If the page cannot be read or the LLM fails.
    """
    if verbose:
        _log("Connecting to browser")
    tree, final_url = await _capture_tree(url, transform_snapshot, verbose)

    if verbose:
        _log("Analyzing with local LLM")
    llm_response, llm_status = await reason(
        prompt=prompt,
        context=json.dumps(tree, ensure_ascii=False),
        json_schema=output_json_schema,
    )
    if llm_status != 0:
        raise BrowserError(f"LLM analysis failed: {llm_response}")

    if verbose:
        _log("Done")
    return llm_response, final_url
