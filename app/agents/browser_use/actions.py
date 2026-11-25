"""Browser action tools bound to a browser session."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar, override

from app.agents.browser_use.checkpoint_store import CaptureInterval, DomCapture, TabCapture
from app.agents.browser_use.models import TransactionResult
from app.agents.browser_use.session import Action, BrowserSession
from app.lib.browser.browser_manager import BrowserManager
from app.lib.browser.page_snapshot import PageSnapshot
from app.lib.shared.format import html_tree_to_markdown
from app.lib.shared.paths import resolve_path
from app.lib.tool.base import Param, Tool, ToolResult

_NETWORK_IDLE_TIMEOUT_MILLISECONDS = 2000
_CLICK_SETTLE_DELAY_SECONDS = 2.0
_DEFAULT_CAPTCHA_MAX_STEPS = 5
_DEFAULT_WAIT_SECONDS = 2.0
_NO_OPEN_TAB_ERROR = "no tab is open, open one with open_url first"

_TARGET_ID_DESCRIPTION = (
    "The tab to act on. When given, the action runs on that tab and fails if no such tab exists. "
    "When left out, it runs on the single open tab and fails if zero or more than one tab is open. "
    "A tab's target id is the first token on the tab's line in the step result."
)

_OPEN_URL_TARGET_ID_DESCRIPTION = (
    "The tab to open the page in. When given, the page opens in that tab and fails if no such tab exists. "
    "When left out, a new tab is opened. A tab's target id is the first token on the tab's line in the step result."
)

_COORDINATE_PARAMS: tuple[Param, ...] = (
    Param("x", float, min=0, max=1, description="Position across the screen, 0 at the left, 1 at the right."),
    Param("y", float, min=0, max=1, description="Position down the screen, 0 at the top, 1 at the bottom."),
)

_TARGET_ID_PARAM = Param("target_id", str | None, None, description=_TARGET_ID_DESCRIPTION)


@dataclass(frozen=True)
class _ObservationMarkers:
    """Line prefixes shared between observation formatting and history pruning."""

    status_prefix: str = "status:"
    checkpoint_prefix: str = "Checkpoint: "
    observation_time_prefix: str = "Observation time: "
    dom_capture_interval_prefix: str = "DOM capture interval: "
    dom_dump_target_prefix: str = "DOM dump target: "
    archived_dom_target_prefix: str = "Archived DOM target: "


OBSERVATION_MARKERS = _ObservationMarkers()


def _require_open_tab(tab: str | None) -> str:
    """Return the tab id, or raise when no tab is open."""
    if tab is None:
        raise RuntimeError(_NO_OPEN_TAB_ERROR)
    return tab


def _format_status_line(result: TransactionResult) -> str:
    """Format the status line that follows tab captions."""
    state = result.status.name.lower()
    if result.message:
        return f"{OBSERVATION_MARKERS.status_prefix} {state}, {result.message}"
    return f"{OBSERVATION_MARKERS.status_prefix} {state}"


def _build_tool_result(result: TransactionResult) -> ToolResult:
    """Build the tool result text and screenshot list for a transaction."""
    tabs = list(result.snapshot.tabs.values())
    lines = [tab.caption for tab in tabs]
    lines.append(_format_status_line(result))
    if result.snapshot.checkpoint_id is not None:
        lines.append(f"{OBSERVATION_MARKERS.checkpoint_prefix}{result.snapshot.checkpoint_id}")
    if result.snapshot.observed_at is not None:
        lines.append(f"{OBSERVATION_MARKERS.observation_time_prefix}{result.snapshot.observed_at.isoformat()}")
    if result.snapshot.archive_error:
        lines.append(result.snapshot.archive_error)
    files = tuple(tab.screenshot_path for tab in tabs if tab.screenshot_path is not None)
    return ToolResult(text="\n".join(lines), files=files if files else None)


def _format_time_line() -> str:
    """Format the current local time line for an action result."""
    now = datetime.now().astimezone()
    return f"Time: {now:%Y-%m-%d %H:%M:%S} {now.tzname()}"


def format_resource_paths(downloads: list[str], resources_directory: str) -> tuple[str, ...]:
    """Convert download file paths to paths relative to the resources folder.

    Args:
        downloads: File paths returned by the browser session.
        resources_directory: Root resources folder used as the relative base.

    Returns:
        Relative paths for files inside the resources folder, and original paths otherwise.
    """
    base_directory = os.path.abspath(resources_directory)
    formatted_paths: list[str] = []
    for file_path in downloads:
        if os.path.isabs(file_path) and file_path.startswith(base_directory + os.sep):
            formatted_paths.append(os.path.relpath(file_path, base_directory))
        else:
            formatted_paths.append(file_path)
    return tuple(formatted_paths)


async def _validate_upload_paths(resource_paths: list[str], resources_directory: str) -> list[str]:
    """Resolve and verify resource file paths for upload."""
    if not resource_paths:
        raise ValueError("resource_paths must contain at least one file")
    resolved_files: list[str] = []
    for relative_path in resource_paths:
        if not relative_path or os.path.isabs(relative_path):
            raise ValueError(f"resource path must be relative: {relative_path}")
        resolved_file = await resolve_path(resources_directory, relative_path)
        if resolved_file is None or not await asyncio.to_thread(os.path.isfile, resolved_file):
            raise ValueError(f"resource file not found: {relative_path}")
        resolved_files.append(resolved_file)
    return resolved_files


async def _run_snapshot_action(
    manager: BrowserManager,
    tab: str,
    action: Callable[[PageSnapshot], Awaitable[bool]],
    *,
    fixed_delay_seconds: float | None = None,
) -> bool:
    """Capture a page snapshot, run an action on it, and wait for the page to settle."""
    snapshot = await manager.capture_snapshot(tab, timeout=_NETWORK_IDLE_TIMEOUT_MILLISECONDS, verbose=True)
    if snapshot is None:
        raise RuntimeError("no snapshot available for tab")
    has_succeeded = await action(snapshot)
    if fixed_delay_seconds is None:
        await manager.wait_for_load(tab, "networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MILLISECONDS, verbose=True)
    else:
        await asyncio.sleep(fixed_delay_seconds)
    return has_succeeded


def _build_scroll_action(
    dx: float, dy: float, x: float | None, y: float | None
) -> Callable[[BrowserManager, str], Awaitable[bool]]:
    """Build an action that scrolls an open tab, optionally anchored to a point."""

    async def scroll_tab(manager: BrowserManager, tab: str) -> bool:
        if (x is None) != (y is None):
            raise RuntimeError("x and y must be given together")

        async def apply_scroll(snapshot: PageSnapshot) -> bool:
            if x is not None and y is not None:
                if not await snapshot.move_mouse_to_relative(x, y, verbose=True):
                    raise RuntimeError("could not move the mouse to the scroll anchor")
            return await snapshot.scroll_by_relative(dx, dy, verbose=True)

        return await _run_snapshot_action(manager, tab, apply_scroll)

    return scroll_tab


class _ActionTool(Tool):
    """Base browser action tool bound to a session."""

    is_tracked: ClassVar[bool] = True
    result_render_type: ClassVar[str] = "text"

    def __init__(self, session: BrowserSession, resources_directory: str = "") -> None:
        self._session = session
        self._resources_directory = resources_directory

    async def _run_action(
        self,
        action: Action,
        target_id: str | None,
        *,
        is_new_tab: bool = False,
        should_monitor_download: bool = False,
        capture_dom: Callable[[], TabCapture | None] | None = None,
    ) -> ToolResult:
        """Execute one browser action through the session and prefix the local time."""
        result = await self._session.transact(
            action,
            target_id,
            new_tab=is_new_tab,
            monitor_download=should_monitor_download,
            source_action=self.name,
            capture_dom=capture_dom,
        )
        step_result = self._format_step_result(result)
        return ToolResult(
            text=f"{_format_time_line()}\n{step_result.text}",
            files=step_result.files,
            render_type=step_result.render_type,
        )

    async def _run_on_open_tab(
        self,
        action: Callable[[BrowserManager, str], Awaitable[bool]],
        target_id: str | None,
        *,
        should_monitor_download: bool = False,
        capture_dom: Callable[[], TabCapture | None] | None = None,
    ) -> ToolResult:
        """Execute a browser action that requires an existing open tab."""

        async def run_checked(manager: BrowserManager, tab: str | None) -> bool:
            return await action(manager, _require_open_tab(tab))

        return await self._run_action(
            run_checked,
            target_id,
            should_monitor_download=should_monitor_download,
            capture_dom=capture_dom,
        )

    async def _run_on_snapshot(
        self,
        action: Callable[[PageSnapshot], Awaitable[bool]],
        target_id: str | None,
        *,
        fixed_delay_seconds: float | None = None,
        should_monitor_download: bool = False,
    ) -> ToolResult:
        """Capture a snapshot of an open tab, run an action on it, and wait for it to settle."""
        return await self._run_on_open_tab(
            lambda manager, tab: _run_snapshot_action(
                manager, tab, action, fixed_delay_seconds=fixed_delay_seconds
            ),
            target_id,
            should_monitor_download=should_monitor_download,
        )

    def _format_step_result(self, result: TransactionResult) -> ToolResult:
        """Format the tool result of one transaction without the time line."""
        return _build_tool_result(result)


class _OpenUrlTool(_ActionTool):
    name: ClassVar[str] = "open_url"
    description: ClassVar[str] = (
        "Open a web page in the browser. Use to start work on an address. "
        "Give a tab to reuse it, or leave the tab out to open a fresh one."
    )
    summary: ClassVar[str] = "Open a web page in the browser"
    params: ClassVar[tuple[Param, ...]] = (
        Param("url", str, kind="url", description="The address to open."),
        Param("target_id", str | None, None, description=_OPEN_URL_TARGET_ID_DESCRIPTION),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        url = params["url"]
        target_id = params["target_id"]

        async def open_page_and_navigate(manager: BrowserManager, tab: str | None) -> bool:
            if tab is None:
                tab = await manager.open_page(verbose=True)
            return await manager.navigate(tab, url, verbose=True)

        return await self._run_action(open_page_and_navigate, target_id, is_new_tab=True)


class _ClickTool(_ActionTool):
    name: ClassVar[str] = "click"
    description: ClassVar[str] = (
        "Click a point on a tab's screen. Use to press buttons, links, and other controls."
    )
    summary: ClassVar[str] = "Click a point on a tab's screen"
    params: ClassVar[tuple[Param, ...]] = (*_COORDINATE_PARAMS, _TARGET_ID_PARAM)

    @override
    async def call(self, params: dict) -> ToolResult:
        return await self._run_on_snapshot(
            lambda snapshot: snapshot.click_at_relative(params["x"], params["y"], verbose=True),
            params["target_id"],
            fixed_delay_seconds=_CLICK_SETTLE_DELAY_SECONDS,
        )


class _UploadResourcesTool(_ActionTool):
    name: ClassVar[str] = "upload_resources"
    description: ClassVar[str] = (
        "Upload resource files through a file picker opened by clicking a point on a tab's screen. "
        "Use when the page shows an upload button rather than a visible file box."
    )
    summary: ClassVar[str] = "Upload resource files through a file picker"
    params: ClassVar[tuple[Param, ...]] = (
        *_COORDINATE_PARAMS,
        Param(
            "resource_paths",
            list[str],
            description="File paths relative to the resource root, for example upload/photo.jpg.",
        ),
        _TARGET_ID_PARAM,
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        x = params["x"]
        y = params["y"]
        resource_paths = params["resource_paths"]

        async def upload_files(manager: BrowserManager, tab: str) -> bool:
            files = await _validate_upload_paths(resource_paths, self._resources_directory)
            return await _run_snapshot_action(
                manager,
                tab,
                lambda snapshot: snapshot.upload_with_chooser_relative(x, y, files, verbose=True),
            )

        return await self._run_on_open_tab(upload_files, params["target_id"])


class _MoveMouseTool(_ActionTool):
    name: ClassVar[str] = "move_mouse"
    description: ClassVar[str] = (
        "Move the mouse to a point on a tab's screen without clicking. "
        "Use to hover over a control and reveal a tooltip or a menu."
    )
    summary: ClassVar[str] = "Move the mouse to a point on a tab's screen"
    params: ClassVar[tuple[Param, ...]] = (*_COORDINATE_PARAMS, _TARGET_ID_PARAM)

    @override
    async def call(self, params: dict) -> ToolResult:
        return await self._run_on_snapshot(
            lambda snapshot: snapshot.move_mouse_to_relative(params["x"], params["y"], verbose=True),
            params["target_id"],
        )


class _TypeTextTool(_ActionTool):
    name: ClassVar[str] = "type_text"
    description: ClassVar[str] = (
        "Type text at a point on a tab's screen. Use to fill in form fields after the field has focus."
    )
    summary: ClassVar[str] = "Type text at a point on a tab's screen"
    params: ClassVar[tuple[Param, ...]] = (
        *_COORDINATE_PARAMS,
        Param("text", str, description="The text to type."),
        Param("submit", bool, False, description="Whether to press Enter after typing."),
        _TARGET_ID_PARAM,
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        should_submit = params["submit"]
        return await self._run_on_snapshot(
            lambda snapshot: snapshot.type_text_at_relative(
                params["x"], params["y"], params["text"], submit=should_submit, verbose=True
            ),
            params["target_id"],
        )


class _ReplaceInputTextTool(_ActionTool):
    name: ClassVar[str] = "replace_input_text"
    description: ClassVar[str] = (
        "Replace a piece of the text already in a form field at a point on a tab's screen. "
        "Use to fix or change part of a field's text, where type_text only adds new text."
    )
    summary: ClassVar[str] = "Replace part of the text in a form field"
    params: ClassVar[tuple[Param, ...]] = (
        *_COORDINATE_PARAMS,
        Param(
            "old_text",
            str,
            description="The text in the field to replace. It must match exactly once in the field's current text.",
        ),
        Param("new_text", str, description="The text to put in its place. An empty string deletes the old text."),
        _TARGET_ID_PARAM,
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        return await self._run_on_snapshot(
            lambda snapshot: snapshot.replace_input_text_at_relative(
                params["x"], params["y"], params["old_text"], params["new_text"], verbose=True
            ),
            params["target_id"],
        )


class _ScrollTool(_ActionTool):
    name: ClassVar[str] = "scroll"
    description: ClassVar[str] = (
        "Scroll a tab's screen. Use to move down or up a page, and to shift sideways. "
        "Give a point on the screen to anchor the scroll to a nested scrollable region."
    )
    summary: ClassVar[str] = "Scroll a tab's screen"
    params: ClassVar[tuple[Param, ...]] = (
        Param(
            "dx",
            float,
            description="How far to scroll sideways as a fraction of the viewport width. 1 is one screen wide, negative is left.",
        ),
        Param(
            "dy",
            float,
            description="How far to scroll down as a fraction of the viewport height. 1 is one screen tall, negative is up.",
        ),
        Param(
            "x",
            float | None,
            None,
            min=0,
            max=1,
            description="The point on the screen the scroll is applied at, as a fraction of the width. Give it with y.",
        ),
        Param(
            "y",
            float | None,
            None,
            min=0,
            max=1,
            description="The point on the screen the scroll is applied at, as a fraction of the height. Give it with x.",
        ),
        _TARGET_ID_PARAM,
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        return await self._run_on_open_tab(
            _build_scroll_action(params["dx"], params["dy"], params["x"], params["y"]),
            params["target_id"],
        )


class _PressKeyTool(_ActionTool):
    name: ClassVar[str] = "press_key"
    description: ClassVar[str] = (
        "Press a key in a tab. Use to confirm a form with Enter, or to send other keys."
    )
    summary: ClassVar[str] = "Press a key in a tab"
    params: ClassVar[tuple[Param, ...]] = (
        Param("key", str, description="The key to press, like Enter or Escape."),
        _TARGET_ID_PARAM,
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        return await self._run_on_snapshot(
            lambda snapshot: snapshot.press_key(params["key"], verbose=True),
            params["target_id"],
        )


class _GoBackTool(_ActionTool):
    name: ClassVar[str] = "go_back"
    description: ClassVar[str] = "Go back one page in a tab's history. Use after a page that did not fit."
    summary: ClassVar[str] = "Go back one page in a tab"
    params: ClassVar[tuple[Param, ...]] = (_TARGET_ID_PARAM,)

    @override
    async def call(self, params: dict) -> ToolResult:
        return await self._run_on_open_tab(lambda manager, tab: manager.go_back(tab, verbose=True), params["target_id"])


class _GoForwardTool(_ActionTool):
    name: ClassVar[str] = "go_forward"
    description: ClassVar[str] = (
        "Go forward one page in a tab's history. Use after going back, to redo a page."
    )
    summary: ClassVar[str] = "Go forward one page in a tab"
    params: ClassVar[tuple[Param, ...]] = (_TARGET_ID_PARAM,)

    @override
    async def call(self, params: dict) -> ToolResult:
        return await self._run_on_open_tab(
            lambda manager, tab: manager.go_forward(tab, verbose=True), params["target_id"]
        )


class _ClosePageTool(_ActionTool):
    name: ClassVar[str] = "close_page"
    description: ClassVar[str] = "Close a tab. Use to drop a tab you no longer need."
    summary: ClassVar[str] = "Close a tab"
    params: ClassVar[tuple[Param, ...]] = (_TARGET_ID_PARAM,)

    @override
    async def call(self, params: dict) -> ToolResult:
        return await self._run_on_open_tab(
            lambda manager, tab: manager.close_page(tab, verbose=True), params["target_id"]
        )


class _SolveCaptchaTool(_ActionTool):
    name: ClassVar[str] = "solve_captcha"
    description: ClassVar[str] = (
        "Solve a captcha shown on a tab. Use when a page is blocked by a captcha."
    )
    summary: ClassVar[str] = "Solve a captcha on a tab"
    params: ClassVar[tuple[Param, ...]] = (
        Param("max_steps", int, _DEFAULT_CAPTCHA_MAX_STEPS, min=1, description="The most solving steps to try."),
        _TARGET_ID_PARAM,
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        return await self._run_on_snapshot(
            lambda snapshot: snapshot.solve_captcha(params["max_steps"], verbose=True),
            params["target_id"],
        )


class _WaitTool(_ActionTool):
    name: ClassVar[str] = "wait"
    description: ClassVar[str] = (
        "Wait without acting. Use to let a page settle after a step that changes it."
    )
    summary: ClassVar[str] = "Wait without acting"
    params: ClassVar[tuple[Param, ...]] = (
        Param("seconds", float, _DEFAULT_WAIT_SECONDS, min=0, description="How long to wait."),
        _TARGET_ID_PARAM,
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        seconds = params["seconds"]

        async def wait_in_tab(_manager: BrowserManager, _tab: str) -> bool:
            await asyncio.sleep(seconds)
            return True

        return await self._run_on_open_tab(wait_in_tab, params["target_id"])


class _DownloadClickTool(_ActionTool):
    name: ClassVar[str] = "download_click"
    description: ClassVar[str] = (
        "Click a point on a tab's screen that starts a download, and wait for the file. "
        "Use instead of click when the click brings a file down."
    )
    summary: ClassVar[str] = "Click a point on a tab's screen that starts a download"
    params: ClassVar[tuple[Param, ...]] = (*_COORDINATE_PARAMS, _TARGET_ID_PARAM)

    @override
    async def call(self, params: dict) -> ToolResult:
        return await self._run_on_snapshot(
            lambda snapshot: snapshot.click_at_relative(params["x"], params["y"], verbose=True),
            params["target_id"],
            fixed_delay_seconds=_CLICK_SETTLE_DELAY_SECONDS,
            should_monitor_download=True,
        )

    @override
    def _format_step_result(self, result: TransactionResult) -> ToolResult:
        """Format the transaction result with one line for each downloaded file."""
        step_result = _build_tool_result(result)
        paths = format_resource_paths(list(result.files), self._resources_directory)
        lines = [step_result.text, *(f"Downloaded file: {path}" for path in paths)]
        return ToolResult(text="\n".join(lines), files=step_result.files, render_type=step_result.render_type)


class _RefreshTool(_ActionTool):
    name: ClassVar[str] = "refresh"
    description: ClassVar[str] = (
        "Reload a tab's current page. Use when the page is stuck, frozen, or showing an error state. "
        "The tab navigates to its own current URL."
    )
    summary: ClassVar[str] = "Reload a tab's current page"
    params: ClassVar[tuple[Param, ...]] = (_TARGET_ID_PARAM,)

    @override
    async def call(self, params: dict) -> ToolResult:
        async def refresh_tab(manager: BrowserManager, tab: str) -> bool:
            url = manager.get_url(tab)
            if not url:
                raise RuntimeError("the tab has no url to refresh")
            return await manager.navigate(tab, url, verbose=True)

        return await self._run_on_open_tab(refresh_tab, params["target_id"])


class _DumpPageTool(_ActionTool):
    name: ClassVar[str] = "dump_page"
    description: ClassVar[str] = (
        "Read the current page as markdown from its filtered DOM. "
        "Use when screen text is too small or the page has a lot of text to read."
    )
    summary: ClassVar[str] = "Read a page as markdown"
    params: ClassVar[tuple[Param, ...]] = (_TARGET_ID_PARAM,)

    @override
    async def call(self, params: dict) -> ToolResult:
        captured_dump: TabCapture | None = None

        async def capture_page_dom(manager: BrowserManager, tab: str) -> bool:
            nonlocal captured_dump
            started_at = datetime.now(UTC)
            snapshot = await manager.capture_snapshot(tab, timeout=_NETWORK_IDLE_TIMEOUT_MILLISECONDS, verbose=True)
            if snapshot is None:
                raise RuntimeError("no snapshot available for tab")
            tree = await snapshot.filtered_tree(verbose=True)
            if tree is None:
                raise RuntimeError("could not read the page DOM")
            markdown = await html_tree_to_markdown(tree)
            captured_dump = TabCapture(
                target_id=tab,
                title=snapshot.title,
                url=snapshot.url,
                dom=DomCapture(markdown, CaptureInterval(started_at, datetime.now(UTC))),
            )
            return True

        result = await self._run_on_open_tab(
            capture_page_dom, params["target_id"], capture_dom=lambda: captured_dump
        )
        if captured_dump is None or captured_dump.dom is None:
            return result
        interval = captured_dump.dom.interval
        return ToolResult(
            text=(
                f"{result.text}\n{OBSERVATION_MARKERS.dom_capture_interval_prefix}"
                f"{interval.started_at.isoformat()} to {interval.finished_at.isoformat()}\n"
                f"{OBSERVATION_MARKERS.dom_dump_target_prefix}{captured_dump.target_id}\n\n"
                f"{captured_dump.dom.text}"
            ),
            files=result.files,
            render_type="markdown",
        )


_ACTION_TOOL_CLASSES: tuple[type[_ActionTool], ...] = (
    _OpenUrlTool,
    _ClickTool,
    _UploadResourcesTool,
    _MoveMouseTool,
    _TypeTextTool,
    _ReplaceInputTextTool,
    _ScrollTool,
    _PressKeyTool,
    _GoBackTool,
    _GoForwardTool,
    _ClosePageTool,
    _SolveCaptchaTool,
    _WaitTool,
    _DownloadClickTool,
    _RefreshTool,
    _DumpPageTool,
)

ACTION_TOOL_NAMES: frozenset[str] = frozenset(tool_class.name for tool_class in _ACTION_TOOL_CLASSES)
DUMP_PAGE_TOOL_NAME: str = _DumpPageTool.name


def build_action_tools(session: BrowserSession, resources_directory: str) -> tuple[Tool, ...]:
    """Build the browser action tools bound to a session.

    Args:
        session: Active browser session for running actions.
        resources_directory: Root folder for uploaded and downloaded resource files.

    Returns:
        Ordered tuple of browser action tools.
    """
    return tuple(tool_class(session, resources_directory) for tool_class in _ACTION_TOOL_CLASSES)


__all__ = [
    "ACTION_TOOL_NAMES",
    "DUMP_PAGE_TOOL_NAME",
    "OBSERVATION_MARKERS",
    "build_action_tools",
    "format_resource_paths",
]
