from __future__ import annotations

"""Find and solve captchas on a page."""

import asyncio
import json
import os
import sys
import tempfile
from typing import Optional, Protocol, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from app.lib.browser.browser_manager import BrowserManager

from PIL import Image as PilImage

from app.lib.reason.reason import reason
from app.lib.shared import paths

__all__ = ["solve_captcha"]

_NETWORK_IDLE_TIMEOUT_MS = 5000
_SETTLE_DELAY_SECONDS = 2
_SCRATCH_NAME = "captcha"
_GRID_CLICK_TIMEOUT_MS = 1500
_GRID_RETRY_TIMES = 30


def _scratch_jpg_path() -> str:
    """Make a fresh, unused picture file in the fast scratch folder."""
    scratch = paths.scratch_dir(_SCRATCH_NAME)
    os.makedirs(scratch, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=scratch, prefix="captcha_", suffix=".jpg")
    os.close(fd)
    return path


def _drop_scratch(path: str) -> None:
    """Delete a scratch picture, ignoring it when it is already gone."""
    try:
        os.remove(path)
    except OSError:
        pass


_CAPTCHA_DETECT_PROMPT = f"""Analyze this browser screenshot. Is there an unsolved CAPTCHA challenge on screen?

Categorization policy:
- "checkbox": A clickable checkbox element is visible (reCAPTCHA v2 "I'm not a robot" box, hCaptcha checkbox, or similar). Includes the state where only the checkbox needs to be clicked. Also includes the Verify/Submit button after image grid tiles have been selected.
- "image_grid": A grid of image tiles requiring the user to select matching tiles (reCAPTCHA v2 grid, hCaptcha grid, etc.). The grid must be actively waiting for tile selection. If tiles are already selected and only the Verify/Submit button remains, classify as "checkbox" instead.
- "text_input": Requires the user to type text (distorted text CAPTCHA, math answer, word entry, etc.).
- "unknown": A CAPTCHA is present but does not clearly match any of the above categories (e.g., Cloudflare Turnstile still spinning, slider-based CAPTCHA, rotate-to-match, audio-based, or any other type).
- If NO captcha is present or the CAPTCHA is already solved, set has_captcha=false and captcha_type="unknown".

What counts as solved: A CAPTCHA is solved ONLY when the puzzle is fully complete AND the final confirmation action has been taken.
- For checkbox captchas, the checkbox must show a green checkmark.
- For image grid captchas, all matching tiles must be selected AND the Verify/Submit button must have been clicked.
- Selecting tiles without clicking Verify is NOT solved.
- Entering text without submitting is NOT solved.

Output rules:
- has_captcha: true if an unsolved CAPTCHA is actively on screen, false otherwise.
- captcha_type: one of "unknown", "image_grid", "checkbox", "text_input".
- Provide a clear reason explaining what you found and why you chose the type."""

_CAPTCHA_DETECT_SCHEMA = {
    "type": "object",
    "properties": {
        "has_captcha": {"type": "boolean"},
        "captcha_type": {
            "type": "string",
            "enum": ["unknown", "image_grid", "checkbox", "text_input"],
        },
        "reason": {"type": "string"},
    },
    "required": ["has_captcha", "captcha_type", "reason"],
}

_CHECKBOX_CAPTCHA_PROMPT = f"""Analyze this browser screenshot. Find the checkbox CAPTCHA element that needs to be clicked.

Output rules:
- If the checkbox is already checked (green checkmark visible), set clickable=false and x=0, y=0.
- If the checkbox is unchecked and clickable, set clickable=true and return the relative coordinates (0.0-1.0) of the center of the checkbox.
- If there's a Verify/Submit button remaining after tiles are selected, set clickable=true and return coordinates of that button center.
- If no checkbox CAPTCHA is visible on screen, set clickable=false and x=0, y=0.
- If the CAPTCHA is in a loading/spinning state, set clickable=false and x=0, y=0.
- Provide a clear reason explaining what you found.
- Coordinates are relative: x and y are floats between 0.0 (left/top) and 1.0 (right/bottom) of the full screenshot. 0.5,0.5 means the exact center.
- When clickable=false, always set x=0 and y=0. Do not return arbitrary coordinates."""

_CHECKBOX_CAPTCHA_SCHEMA = {
    "type": "object",
    "properties": {
        "clickable": {"type": "boolean"},
        "x": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "y": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
    },
    "required": ["clickable", "x", "y", "reason"],
}


class _JsonPayload(Protocol):
    """A value parsed from LLM answer text."""


class _DetectResult(TypedDict):
    """What the captcha detector LLM reported."""

    has_captcha: bool
    captcha_type: str
    reason: str


def _validate(data: _JsonPayload, fields: tuple, label: str) -> bool:
    """Check that an LLM answer has the fields we need.

    Field types are not checked.

    Args:
        data: The answer from the LLM.
        fields: Names that must be there.
        label: Name for this check, used in error messages.

    Returns:
        True if every field is there.
    """
    if not isinstance(data, dict):
        print(f"[captcha_solver] LLM response is not a dict ({label})", file=sys.stderr)
        return False
    for field in fields:
        if field not in data:
            print(
                f"[captcha_solver] Missing field '{field}' in {label} response",
                file=sys.stderr,
            )
            return False
    return True


async def _call_llm(prompt: str, image_files: list, schema: dict, fields: tuple, label: str) -> Optional[dict]:
    """Ask the LLM about a captcha picture.

    Args:
        prompt: What to ask the LLM.
        image_files: Pictures to send along.
        schema: Shape the answer should follow.
        fields: Names the answer must have.
        label: Name for this call, used in error messages.

    Returns:
        The answer, or None if anything went wrong.
    """
    result_text, status = await reason(
        prompt=prompt, image_files=image_files, json_schema=schema
    )
    if status != 0:
        print(
            f"[captcha_solver] LLM {label} error (status={status}): {result_text}",
            file=sys.stderr,
        )
        return None
    try:
        data = json.loads(result_text)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"[captcha_solver] JSON parse error ({label}): {e}", file=sys.stderr)
        return None
    if not _validate(data, fields, label):
        return None
    return data


def _read_jpg_size(screenshot_path: str) -> tuple[int, int]:
    """Give back the size of a picture file.

    Args:
        screenshot_path: Path to the picture.

    Returns:
        Width and height in pixels.
    """
    with PilImage.open(screenshot_path) as img:
        return img.size


async def _get_dims(manager: BrowserManager, target_id: str, screenshot_path: str) -> tuple[int, int]:
    """Get the width and height of a screenshot.

    The window size is used if the picture cannot be read.

    Args:
        manager: The browser the tab belongs to.
        target_id: The tab to read the window size from.
        screenshot_path: Path to the picture.

    Returns:
        Width and height in pixels.
    """
    try:
        return await asyncio.to_thread(_read_jpg_size, screenshot_path)
    except Exception as e:
        print(f"[captcha_solver] PIL failed: {e}", file=sys.stderr)
        viewport = await manager.get_page_viewport(target_id)
        if viewport is None:
            print(f"[captcha_solver] Cannot read the window size", file=sys.stderr)
            return (0, 0)
        return (viewport.usable_width, viewport.usable_height)


async def _detect(screenshot_path: str, verbose: bool) -> Optional[_DetectResult]:
    """Work out what kind of captcha is on screen.

    Args:
        screenshot_path: Path to the picture of the page.
        verbose: Print progress lines when True.

    Returns:
        What the LLM saw, or None if it could not tell.
    """
    if verbose:
        print("[captcha_solver] Detecting captcha type...", flush=True)
    data = await _call_llm(
        _CAPTCHA_DETECT_PROMPT,
        [screenshot_path],
        _CAPTCHA_DETECT_SCHEMA,
        ("has_captcha", "captcha_type", "reason"),
        "detect",
    )
    return data


async def _solve_checkbox(manager: BrowserManager, target_id: str, verbose: bool) -> bool:
    """Solve a captcha that only needs one click.

    Args:
        manager: The browser to click with.
        target_id: Page to work on.
        verbose: Print progress lines when True.

    Returns:
        True if the click was made.
    """
    if verbose:
        print("[captcha_solver] Solving checkbox captcha...", flush=True)

    screenshot_path = await asyncio.to_thread(_scratch_jpg_path)
    try:
        if await manager.screenshot(target_id, screenshot_path, full_page=False) is None:
            print(f"[captcha_solver] Screenshot failed", file=sys.stderr)
            return False

        data = await _call_llm(
            _CHECKBOX_CAPTCHA_PROMPT,
            [screenshot_path],
            _CHECKBOX_CAPTCHA_SCHEMA,
            ("clickable", "x", "y", "reason"),
            "checkbox_solve",
        )
        if data is None:
            return False

        reason_text = data.get("reason", "")

        if not data["clickable"]:
            print(f"[captcha_solver] Checkbox not clickable: {reason_text}", file=sys.stderr)
            return False

        if verbose:
            print(f"[captcha_solver] Found checkbox at ({data['x']}, {data['y']}): {reason_text}", flush=True)

        img_w, img_h = await _get_dims(manager, target_id, screenshot_path)
    finally:
        _drop_scratch(screenshot_path)
    px = round(data["x"] * img_w)
    py = round(data["y"] * img_h)
    ok = await manager.click_at(target_id, px, py, verbose=verbose)
    if not ok:
        print(f"[captcha_solver] click_at({px}, {py}) failed", file=sys.stderr)
        return False
    if verbose:
        print(f"[captcha_solver] click_at({px}, {py}) succeeded", flush=True)
    return True


async def _is_recaptcha(manager: BrowserManager, target_id: str) -> bool:
    """Say if the page has a reCAPTCHA box.

    Args:
        manager: The browser the tab belongs to.
        target_id: The tab to look at.

    Returns:
        True if the page has one.
    """
    if target_id not in manager.tab_ids():
        return False
    return bool(await manager.evaluate(target_id, """
            () => {
                const iframes = document.querySelectorAll('iframe');
                for (const iframe of iframes) {
                    const src = (iframe.getAttribute('src') || '').toLowerCase();
                    const title = (iframe.getAttribute('title') || '').toLowerCase();
                    if (src.includes('recaptcha.net') || src.includes('google.com/recaptcha') ||
                        title === 'recaptcha') {
                        return true;
                    }
                }
                return window.grecaptcha !== undefined;
            }
        """))


async def _solve_image_grid(manager: BrowserManager, target_id: str, verbose: bool) -> bool:
    """Solve a captcha that asks you to pick pictures.

    Only reCAPTCHA is solved, anything else fails.

    Args:
        manager: The browser the tab belongs to.
        target_id: The tab to work on.
        verbose: Print progress lines when True.

    Returns:
        True if the captcha was solved.
    """
    if verbose:
        print("[captcha_solver] Solving image grid captcha...", flush=True)

    if not await _is_recaptcha(manager, target_id):
        print("[captcha_solver] Not reCAPTCHA, cannot solve", file=sys.stderr)
        return False

    result = await manager.solve_recaptcha_grid(
        target_id, click_timeout=_GRID_CLICK_TIMEOUT_MS, retry_times=_GRID_RETRY_TIMES, verbose=verbose
    )
    if verbose:
        print(f"[captcha_solver] recognizer result: {str(result)[:40] if result else None}...", flush=True)
    return result is not None and len(str(result)) > 10


def _solve_text_input(verbose: bool) -> bool:
    """Give up on a captcha that asks you to type.

    Args:
        verbose: Print progress lines when True.

    Returns:
        False, always.
    """
    if verbose:
        print("[captcha_solver] Solving text input captcha (not yet implemented)", flush=True)
    return False


async def _dispatch_solver(captcha_type: str, manager: BrowserManager, target_id: str, verbose: bool) -> bool:
    """Solve a captcha of the given kind.

    Args:
        captcha_type: What kind of captcha it is.
        manager: The browser to work with.
        target_id: Page to work on.
        verbose: Print progress lines when True.

    Returns:
        True if the captcha was solved.
    """
    if captcha_type == "checkbox":
        return await _solve_checkbox(manager, target_id, verbose)
    if captcha_type == "image_grid":
        return await _solve_image_grid(manager, target_id, verbose)
    if captcha_type == "text_input":
        return _solve_text_input(verbose)
    print(
        f"[captcha_solver] Unhandled captcha type: {captcha_type}",
        file=sys.stderr,
    )
    return False


async def _wait_for_settle(manager: BrowserManager, target_id: str, verbose: bool) -> None:
    """Wait for the page to calm down.

    Args:
        manager: The browser the tab belongs to.
        target_id: The tab to wait on.
        verbose: Print progress lines when True.
    """
    await manager.wait_for_load(target_id, "networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MS, verbose=verbose)
    await asyncio.sleep(_SETTLE_DELAY_SECONDS)


async def _evaluate_detection(manager: BrowserManager, target_id: str, screenshot_path: str, verbose: bool) -> Optional[str]:
    """Look at the page and say what captcha is there.

    Args:
        manager: The browser the tab belongs to.
        target_id: The tab to look at.
        screenshot_path: Where to save the picture of the page.
        verbose: Print progress lines when True.

    Returns:
        The kind of captcha to solve, None if there is no captcha, or the
        word error if the page could not be read or the kind is unknown.
    """
    if not await _take_screenshot(manager, target_id, screenshot_path):
        return "error"

    detect_data = await _detect(screenshot_path, verbose)
    if detect_data is None:
        return "error"

    if not detect_data["has_captcha"]:
        if verbose:
            reason_text = detect_data.get("reason", "")
            print(
                f"[captcha_solver] No CAPTCHA detected.{f' {reason_text}' if reason_text else ''}",
                flush=True,
            )
        return None

    captcha_type = detect_data["captcha_type"]
    if captcha_type == "unknown":
        print(
            f"[captcha_solver] CAPTCHA type unknown, cannot solve: {detect_data['reason']}",
            file=sys.stderr,
        )
        return "error"
    return captcha_type


async def solve_captcha(
    manager: BrowserManager, target_id: str, max_steps: int = 5, verbose: bool = False
) -> bool:
    """Find and solve a captcha on a page.

    Args:
        manager: The browser to work with.
        target_id: Page to work on.
        max_steps: How many tries to make.
        verbose: Print progress lines when True.

    Returns:
        True if the page has no captcha left.
    """
    for step in range(max_steps):
        if verbose:
            print(f"[captcha_solver] Step {step + 1}/{max_steps}", flush=True)

        screenshot_path = await asyncio.to_thread(_scratch_jpg_path)
        try:
            result = await _evaluate_detection(manager, target_id, screenshot_path, verbose)
        finally:
            _drop_scratch(screenshot_path)

        if result is None:
            return True
        if result == "error":
            return False

        captcha_type = result
        ok = await _dispatch_solver(captcha_type, manager, target_id, verbose)
        if not ok:
            print(
                f"[captcha_solver] Solver for '{captcha_type}' returned failure, retrying...",
                file=sys.stderr,
            )

        await _wait_for_settle(manager, target_id, verbose)

    print("[captcha_solver] Max steps exceeded", file=sys.stderr)
    return False


async def _take_screenshot(manager: BrowserManager, target_id: str, screenshot_path: str) -> bool:
    """Take a picture of what the page shows.

    Args:
        manager: The browser the tab belongs to.
        target_id: The tab to shoot.
        screenshot_path: Where to save the picture.

    Returns:
        True if the picture was saved.
    """
    return await manager.screenshot(target_id, screenshot_path, full_page=False) is not None
