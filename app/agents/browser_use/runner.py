"""Drive a browser to a goal with the vision-only role."""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import UTC, datetime
from typing import ClassVar, override
from collections.abc import Awaitable, Callable

from agents.run_config import CallModelData, ModelInputData

from app.agents.browser_use.checkpoint import CheckpointTool
from app.agents.browser_use.checkpoint_store import CaptureInterval, CheckpointStore, DomCapture, TabCapture
from app.agents.browser_use.models import BrowserUseResult, TransactionResult, TransactionStatus
from app.agents.browser_use.prompts.role import build_role_prompt, validate_role_output
from app.agents.browser_use.session import Action, BrowserSession
from app.lib.agent.engine import Agent, AgentRun
from app.lib.browser.browser_manager import BrowserManager
from app.lib.browser.page_snapshot import PageSnapshot
from app.lib.shared import config
from app.lib.shared.format import html_tree_to_markdown
from app.lib.shared.paths import resolve_path
from app.lib.tool.base import Param, Tool, ToolResult
from app.tools.read_media_or_doc import read_media_or_doc_tool
from app.tools.task_tool import task_tool

__all__ = ["run_browser_use"]

_AGENT_NAME = "browser_use"

_NETWORK_IDLE_TIMEOUT_MS = 2000
_CLICK_SETTLE_DELAY_SECONDS = 2.0

_ACTION_TOOL_NAMES = frozenset(
    {
        "open_url",
        "click",
        "upload_resources",
        "move_mouse",
        "type_text",
        "replace_input_text",
        "scroll",
        "press_key",
        "go_back",
        "go_forward",
        "close_page",
        "solve_captcha",
        "wait",
        "download_click",
        "refresh",
        "dump_page",
    }
)

_OLDER_SCREENSHOT_MARKER = "[earlier screenshot omitted]"
_HISTORICAL_TOOL_NAMES = frozenset({CheckpointTool.name})
_TAB_LINE = re.compile(r"^\d+\. (\S+) —", re.MULTILINE)
_STATUS_LINE = re.compile(r"^status:", re.MULTILINE)
_DOM_TARGET_PREFIX = "DOM dump target: "
_DOM_TARGET_LINE = re.compile(rf"^{re.escape(_DOM_TARGET_PREFIX)}(\S+)$", re.MULTILINE)
_ARCHIVED_DOM_PREFIX = "Archived DOM target: "
_ARCHIVED_DOM_LINE = re.compile(rf"^{re.escape(_ARCHIVED_DOM_PREFIX)}(\S+)$", re.MULTILINE)
_CHECKPOINT_LINE = re.compile(r"^Checkpoint: (\S+)$", re.MULTILINE)
_CAPTURE_METADATA_LINE = re.compile(r"^(?:Observation time|DOM capture interval): .+$", re.MULTILINE)

_TARGET_ID_DESC = (
    "The tab to act on. When given, the action runs on that tab and fails if no such tab exists. "
    "When left out, it runs on the single open tab and fails if zero or more than one tab is open. "
    "A tab's target id is the first token on the tab's line in the step result."
)

_OPEN_URL_TARGET_ID_DESC = (
    "The tab to open the page in. When given, the page opens in that tab and fails if no such tab exists. "
    "When left out, a new tab is opened. A tab's target id is the first token on the tab's line in the step result."
)


def _carries_image(output: str | list | None) -> bool:
    """Whether the tool output carries at least one image part."""
    return isinstance(output, list) and any(
        isinstance(part, dict) and part.get("type") == "input_image" for part in output
    )


def _is_action_screenshot(item: dict, name_by_call: dict[str, str]) -> bool:
    """Whether the item is an action tool output that carries a screenshot."""
    return _is_action_output(item, name_by_call) and _carries_image(item.get("output"))


def _is_action_output(item: dict, name_by_call: dict[str, str]) -> bool:
    name = name_by_call.get(item.get("call_id"))
    if name in _HISTORICAL_TOOL_NAMES:
        return False
    return item.get("type") == "function_call_output" and name in _ACTION_TOOL_NAMES


def _drop_older_images(item: dict) -> dict:
    """A copy of the output item with its image parts down to one text marker."""
    parts: list = []
    marker_placed = False
    for part in item["output"]:
        if isinstance(part, dict) and part.get("type") == "input_image":
            if not marker_placed:
                parts.append({"type": "input_text", "text": _OLDER_SCREENSHOT_MARKER})
                marker_placed = True
        else:
            parts.append(part)
    pruned = dict(item)
    pruned["output"] = parts
    return pruned


def _output_text(item: dict) -> str:
    """Get the text from one tool output."""
    output = item.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "\n".join(part.get("text", "") for part in output if isinstance(part, dict))
    return ""


def _latest_observation(items: list, name_by_call: dict[str, str]) -> dict | None:
    for item in reversed(items):
        if not isinstance(item, dict) or not _is_action_output(item, name_by_call):
            continue
        if _STATUS_LINE.search(_output_text(item)) is not None:
            return item
    return None


def _dump_stub(text: str, tab: str) -> str:
    marker = _DOM_TARGET_LINE.search(text)
    header = text[:marker.start()] if marker is not None else text
    checkpoint = _CHECKPOINT_LINE.search(header)
    lines = [f"{_ARCHIVED_DOM_PREFIX}{tab}"]
    lines.extend(dict.fromkeys(_CAPTURE_METADATA_LINE.findall(header)))
    if checkpoint is None:
        lines.append("DOM body omitted. Checkpoint archive unavailable.")
    else:
        checkpoint_id = checkpoint.group(1)
        lines.extend([
            f"Checkpoint: {checkpoint_id}",
            "DOM artifact available in the checkpoint archive.",
            f"Read checkpoint with action read, checkpoint_id {checkpoint_id}, target_id {tab}, kind dom.",
        ])
    return "\n".join(lines)


def _prune_dump(item: dict, tab: str, keep_observation: bool) -> dict:
    text = _output_text(item)
    stub = _dump_stub(text, tab)
    pruned = dict(item)
    if not keep_observation:
        pruned["output"] = stub
        return pruned
    marker = _DOM_TARGET_PREFIX if _DOM_TARGET_LINE.search(text) else _ARCHIVED_DOM_PREFIX

    def replace_text(content: str) -> str:
        return f"{content.split(marker, 1)[0].rstrip()}\n{stub}"

    output = item["output"]
    if isinstance(output, str):
        pruned["output"] = replace_text(output)
    else:
        pruned["output"] = [
            {**part, "text": replace_text(part["text"])}
            if isinstance(part, dict) and isinstance(part.get("text"), str) and marker in part["text"]
            else part
            for part in output
        ]
    return pruned


def _prune_dom_dumps(items: list, name_by_call: dict[str, str]) -> list:
    """Trim old DOM bodies while keeping checkpoint handles."""
    observation = _latest_observation(items, name_by_call)
    open_tabs: set[str] | None = None
    if observation is not None:
        text = _output_text(observation)
        status = _STATUS_LINE.search(text)
        open_tabs = set(_TAB_LINE.findall(text[:status.start()]))
    kept_dumps: set[str] = set()
    replacements: dict[str, dict] = {}
    for item in reversed(items):
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        if name_by_call.get(item.get("call_id")) != "dump_page":
            continue
        text = _output_text(item)
        match = _DOM_TARGET_LINE.search(text)
        archived = _ARCHIVED_DOM_LINE.search(text) if match is None else None
        match = match or archived
        if match is None:
            continue
        tab = match.group(1)
        if archived is not None or tab in kept_dumps or (open_tabs is not None and tab not in open_tabs):
            replacements[item["call_id"]] = _prune_dump(item, tab, item is observation)
        else:
            kept_dumps.add(tab)
    return [
        replacements.get(item.get("call_id"), item)
        if isinstance(item, dict) and item.get("type") == "function_call_output" else item
        for item in items
    ]


def _prune_screenshots(items: list, name_by_call: dict[str, str]) -> list:
    """Keep only the newest browser action screenshot."""
    observation = _latest_observation(items, name_by_call)
    latest = -1
    for index, item in enumerate(items):
        if item is observation or (observation is None and isinstance(item, dict) and _is_action_screenshot(item, name_by_call)):
            latest = index
    if latest == -1:
        return items
    return [
        _drop_older_images(item)
        if index < latest and isinstance(item, dict) and _is_action_screenshot(item, name_by_call)
        else item
        for index, item in enumerate(items)
    ]


def _prune_model_history(data: CallModelData) -> ModelInputData:
    """Keep current tab dumps and the newest browser screenshot."""
    items = data.model_data.input
    name_by_call: dict[str, str] = {}
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            if call_id is not None and name is not None:
                name_by_call[call_id] = name
    items = _prune_dom_dumps(items, name_by_call)
    items = _prune_screenshots(items, name_by_call)
    return ModelInputData(input=items, instructions=data.model_data.instructions)


def _status_line(result: TransactionResult) -> str:
    """The one status line that follows the tab captions."""
    state = result.status.name.lower()
    if result.message:
        return f"status: {state}, {result.message}"
    return f"status: {state}"


def _tool_result(result: TransactionResult) -> ToolResult:
    """The tool result one transaction hands back, captions plus the screenshots."""
    tabs = list(result.snapshot.tabs.values())
    lines = [tab.caption for tab in tabs]
    lines.append(_status_line(result))
    if result.snapshot.checkpoint_id is not None:
        lines.append(f"Checkpoint: {result.snapshot.checkpoint_id}")
    if result.snapshot.observed_at is not None:
        lines.append(f"Observation time: {result.snapshot.observed_at.isoformat()}")
    if result.snapshot.archive_error:
        lines.append(result.snapshot.archive_error)
    files = tuple(tab.screenshot_path for tab in tabs if tab.screenshot_path is not None)
    return ToolResult(text="\n".join(lines), files=files if files else None)


def _time_line() -> str:
    """The current local time, stamped on the first line of an action result."""
    now = datetime.now().astimezone()
    return f"Time: {now:%Y-%m-%d %H:%M:%S} {now.tzname()}"


def _resource_paths(downloads: list[str], resources_dir: str) -> tuple[str, ...]:
    """The download files as paths against the resources folder, absolute when they sit outside it."""
    base = os.path.abspath(resources_dir)
    paths: list[str] = []
    for file in downloads:
        if os.path.isabs(file) and file.startswith(base + os.sep):
            paths.append(os.path.relpath(file, base))
        else:
            paths.append(file)
    return tuple(paths)


async def _upload_paths(resource_paths: list[str], resources_dir: str) -> list[str]:
    """Find the resource files to upload."""
    if not resource_paths:
        raise ValueError("resource_paths must contain at least one file")
    files: list[str] = []
    for path in resource_paths:
        if not path or os.path.isabs(path):
            raise ValueError(f"resource path must be relative: {path}")
        file = await resolve_path(resources_dir, path)
        if file is None or not await asyncio.to_thread(os.path.isfile, file):
            raise ValueError(f"resource file not found: {path}")
        files.append(file)
    return files


def _final_result(raw: str, downloads: list[str], resources_dir: str) -> BrowserUseResult:
    """The validated outcome with the download files added on."""
    data = json.loads(raw)
    return BrowserUseResult(
        status=data["status"], summary=data["summary"], files=_resource_paths(downloads, resources_dir)
    )


async def _run_snapshot_action(
    manager: BrowserManager,
    tab: str,
    action: Callable[[PageSnapshot], Awaitable[bool]],
    *,
    fixed_delay: float | None = None,
) -> bool:
    snapshot = await manager.snapshot(tab, timeout=_NETWORK_IDLE_TIMEOUT_MS, verbose=True)
    if snapshot is None:
        raise RuntimeError("no snapshot available for tab")
    success = await action(snapshot)
    if fixed_delay is None:
        await manager.wait_for_load(tab, "networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MS, verbose=True)
    else:
        await asyncio.sleep(fixed_delay)
    return success


def _click_action(x: float, y: float) -> Action:
    """The action that clicks a point on a tab's screen."""

    async def action(manager: BrowserManager, tab: str | None) -> bool:
        if tab is None:
            raise RuntimeError("no tab is open, open one with open_url first")
        return await _run_snapshot_action(
            manager,
            tab,
            lambda snapshot: snapshot.click_at_relative(x, y, verbose=True),
            fixed_delay=_CLICK_SETTLE_DELAY_SECONDS,
        )
    return action


def _scroll_action(dx: float, dy: float, x: float | None, y: float | None) -> Action:
    """The action that scrolls a tab's screen, anchored to a point on it when given."""

    async def action(manager: BrowserManager, tab: str | None) -> bool:
        if tab is None:
            raise RuntimeError("no tab is open, open one with open_url first")
        if (x is None) != (y is None):
            raise RuntimeError("x and y must be given together")

        async def scroll(snapshot: PageSnapshot) -> bool:
            if x is not None and y is not None:
                if not await snapshot.move_mouse_to_relative(x, y, verbose=True):
                    raise RuntimeError("could not move the mouse to the scroll anchor")
            return await snapshot.scroll_by_relative(dx, dy, verbose=True)
        return await _run_snapshot_action(manager, tab, scroll)
    return action


class _ActionTool(Tool):
    """One browser action bound to the session and the cap signal."""

    tracked: ClassVar[bool] = True
    result_render_type: ClassVar[str] = "text"

    def __init__(self, session: BrowserSession, cap_reached: asyncio.Event) -> None:
        self._session = session
        self._cap_reached = cap_reached

    async def _transact(
        self,
        action: Action,
        target_id: str | None,
        *,
        new_tab: bool = False,
        monitor_download: bool = False,
        capture_dom: Callable[[], TabCapture | None] | None = None,
    ) -> ToolResult:
        result = await self._session.transact(
            action,
            target_id,
            new_tab=new_tab,
            monitor_download=monitor_download,
            source_action=self.name,
            capture_dom=capture_dom,
        )
        if result.status is TransactionStatus.CAP_REACHED:
            self._cap_reached.set()
        base = _tool_result(result)
        return ToolResult(
            text=f"{_time_line()}\n{base.text}",
            files=base.files,
            render_type=base.render_type,
        )


class _OpenUrlTool(_ActionTool):
    name: ClassVar[str] = "open_url"
    description: ClassVar[str] = (
        "Open a web page in the browser. Use to start work on an address. "
        "Give a tab to reuse it, or leave the tab out to open a fresh one."
    )
    summary: ClassVar[str] = "Open a web page in the browser"
    params: ClassVar[tuple[Param, ...]] = (
        Param("url", str, kind="url", description="The address to open."),
        Param("target_id", str | None, None, description=_OPEN_URL_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        url = params["url"]
        target_id = params["target_id"]

        async def action(manager: BrowserManager, tab: str | None) -> bool:
            if tab is None:
                tab = await manager.new_page(verbose=True)
            return await manager.navigate(tab, url, verbose=True)
        return await self._transact(action, target_id, new_tab=True)


class _ClickTool(_ActionTool):
    name: ClassVar[str] = "click"
    description: ClassVar[str] = (
        "Click a point on a tab's screen. Use to press buttons, links, and other controls."
    )
    summary: ClassVar[str] = "Click a point on a tab's screen"
    params: ClassVar[tuple[Param, ...]] = (
        Param("x", float, min=0, max=1, description="Position across the screen, 0 at the left, 1 at the right."),
        Param("y", float, min=0, max=1, description="Position down the screen, 0 at the top, 1 at the bottom."),
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        return await self._transact(_click_action(params["x"], params["y"]), params["target_id"])


class _UploadResourcesTool(_ActionTool):
    name: ClassVar[str] = "upload_resources"
    description: ClassVar[str] = (
        "Upload resource files through a file picker opened by clicking a point on a tab's screen. "
        "Use when the page shows an upload button rather than a visible file box."
    )
    summary: ClassVar[str] = "Upload resource files through a file picker"
    params: ClassVar[tuple[Param, ...]] = (
        Param("x", float, min=0, max=1, description="Position across the screen, 0 at the left, 1 at the right."),
        Param("y", float, min=0, max=1, description="Position down the screen, 0 at the top, 1 at the bottom."),
        Param("resource_paths", list[str], description="File paths relative to the resource root, for example upload/photo.jpg."),
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    def __init__(self, session: BrowserSession, cap_reached: asyncio.Event, resources_dir: str) -> None:
        super().__init__(session, cap_reached)
        self._resources_dir = resources_dir

    @override
    async def call(self, params: dict) -> ToolResult:
        x = params["x"]
        y = params["y"]
        resource_paths = params["resource_paths"]

        async def action(manager: BrowserManager, tab: str | None) -> bool:
            if tab is None:
                raise RuntimeError("no tab is open, open one with open_url first")
            files = await _upload_paths(resource_paths, self._resources_dir)
            return await _run_snapshot_action(
                manager, tab,
                lambda snapshot: snapshot.upload_with_chooser_relative(x, y, files, verbose=True),
            )
        return await self._transact(action, params["target_id"])


class _MoveMouseTool(_ActionTool):
    name: ClassVar[str] = "move_mouse"
    description: ClassVar[str] = (
        "Move the mouse to a point on a tab's screen without clicking. "
        "Use to hover over a control and reveal a tooltip or a menu."
    )
    summary: ClassVar[str] = "Move the mouse to a point on a tab's screen"
    params: ClassVar[tuple[Param, ...]] = (
        Param("x", float, min=0, max=1, description="Position across the screen, 0 at the left, 1 at the right."),
        Param("y", float, min=0, max=1, description="Position down the screen, 0 at the top, 1 at the bottom."),
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        x = params["x"]
        y = params["y"]

        async def action(manager: BrowserManager, tab: str | None) -> bool:
            if tab is None:
                raise RuntimeError("no tab is open, open one with open_url first")
            return await _run_snapshot_action(
                manager, tab, lambda snapshot: snapshot.move_mouse_to_relative(x, y, verbose=True)
            )
        return await self._transact(action, params["target_id"])


class _TypeTextTool(_ActionTool):
    name: ClassVar[str] = "type_text"
    description: ClassVar[str] = (
        "Type text at a point on a tab's screen. Use to fill in form fields after the field has focus."
    )
    summary: ClassVar[str] = "Type text at a point on a tab's screen"
    params: ClassVar[tuple[Param, ...]] = (
        Param("x", float, min=0, max=1, description="Position across the screen, 0 at the left, 1 at the right."),
        Param("y", float, min=0, max=1, description="Position down the screen, 0 at the top, 1 at the bottom."),
        Param("text", str, description="The text to type."),
        Param("submit", bool, False, description="Whether to press Enter after typing."),
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        target_id = params["target_id"]
        x = params["x"]
        y = params["y"]
        text = params["text"]
        submit = params["submit"]

        async def action(manager: BrowserManager, tab: str | None) -> bool:
            if tab is None:
                raise RuntimeError("no tab is open, open one with open_url first")
            return await _run_snapshot_action(manager, tab, lambda snapshot: snapshot.type_text_at_relative(x, y, text, submit=submit, verbose=True))
        return await self._transact(action, target_id)


class _ReplaceInputTextTool(_ActionTool):
    name: ClassVar[str] = "replace_input_text"
    description: ClassVar[str] = (
        "Replace a piece of the text already in a form field at a point on a tab's screen. "
        "Use to fix or change part of a field's text, where type_text only adds new text."
    )
    summary: ClassVar[str] = "Replace part of the text in a form field"
    params: ClassVar[tuple[Param, ...]] = (
        Param("x", float, min=0, max=1, description="Position across the screen, 0 at the left, 1 at the right."),
        Param("y", float, min=0, max=1, description="Position down the screen, 0 at the top, 1 at the bottom."),
        Param("old_text", str, description="The text in the field to replace. It must match exactly once in the field's current text."),
        Param("new_text", str, description="The text to put in its place. An empty string deletes the old text."),
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        x = params["x"]
        y = params["y"]
        old_text = params["old_text"]
        new_text = params["new_text"]

        async def action(manager: BrowserManager, tab: str | None) -> bool:
            if tab is None:
                raise RuntimeError("no tab is open, open one with open_url first")
            return await _run_snapshot_action(
                manager, tab,
                lambda snapshot: snapshot.replace_input_text_at_relative(x, y, old_text, new_text, verbose=True),
            )
        return await self._transact(action, params["target_id"])


class _ScrollTool(_ActionTool):
    name: ClassVar[str] = "scroll"
    description: ClassVar[str] = (
        "Scroll a tab's screen. Use to move down or up a page, and to shift sideways. "
        "Give a point on the screen to anchor the scroll to a nested scrollable region."
    )
    summary: ClassVar[str] = "Scroll a tab's screen"
    params: ClassVar[tuple[Param, ...]] = (
        Param("dx", float, description="How far to scroll sideways as a fraction of the viewport width. 1 is one screen wide, negative is left."),
        Param("dy", float, description="How far to scroll down as a fraction of the viewport height. 1 is one screen tall, negative is up."),
        Param("x", float | None, None, min=0, max=1, description="The point on the screen the scroll is applied at, as a share of the width. Give it with y."),
        Param("y", float | None, None, min=0, max=1, description="The point on the screen the scroll is applied at, as a share of the height. Give it with x."),
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        return await self._transact(
            _scroll_action(params["dx"], params["dy"], params["x"], params["y"]),
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
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        target_id = params["target_id"]
        key = params["key"]

        async def action(manager: BrowserManager, tab: str | None) -> bool:
            if tab is None:
                raise RuntimeError("no tab is open, open one with open_url first")
            return await _run_snapshot_action(manager, tab, lambda snapshot: snapshot.press_key(key, verbose=True))
        return await self._transact(action, target_id)


class _GoBackTool(_ActionTool):
    name: ClassVar[str] = "go_back"
    description: ClassVar[str] = "Go back one page in a tab's history. Use after a page that did not fit."
    summary: ClassVar[str] = "Go back one page in a tab"
    params: ClassVar[tuple[Param, ...]] = (
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        target_id = params["target_id"]

        async def action(manager: BrowserManager, tab: str | None) -> bool:
            if tab is None:
                raise RuntimeError("no tab is open, open one with open_url first")
            return await manager.go_back(tab, verbose=True)
        return await self._transact(action, target_id)


class _GoForwardTool(_ActionTool):
    name: ClassVar[str] = "go_forward"
    description: ClassVar[str] = (
        "Go forward one page in a tab's history. Use after going back, to redo a page."
    )
    summary: ClassVar[str] = "Go forward one page in a tab"
    params: ClassVar[tuple[Param, ...]] = (
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        target_id = params["target_id"]

        async def action(manager: BrowserManager, tab: str | None) -> bool:
            if tab is None:
                raise RuntimeError("no tab is open, open one with open_url first")
            return await manager.go_forward(tab, verbose=True)
        return await self._transact(action, target_id)


class _ClosePageTool(_ActionTool):
    name: ClassVar[str] = "close_page"
    description: ClassVar[str] = "Close a tab. Use to drop a tab you no longer need."
    summary: ClassVar[str] = "Close a tab"
    params: ClassVar[tuple[Param, ...]] = (
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        target_id = params["target_id"]

        async def action(manager: BrowserManager, tab: str | None) -> bool:
            if tab is None:
                raise RuntimeError("no tab is open, open one with open_url first")
            return await manager.close_page(tab, verbose=True)
        return await self._transact(action, target_id)


class _SolveCaptchaTool(_ActionTool):
    name: ClassVar[str] = "solve_captcha"
    description: ClassVar[str] = (
        "Solve a captcha shown on a tab. Use when a page is blocked by a captcha."
    )
    summary: ClassVar[str] = "Solve a captcha on a tab"
    params: ClassVar[tuple[Param, ...]] = (
        Param("max_steps", int, 5, min=1, description="The most solving steps to try."),
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        target_id = params["target_id"]
        max_steps = params["max_steps"]

        async def action(manager: BrowserManager, tab: str | None) -> bool:
            if tab is None:
                raise RuntimeError("no tab is open, open one with open_url first")
            return await _run_snapshot_action(manager, tab, lambda snapshot: snapshot.solve_captcha(max_steps, verbose=True))
        return await self._transact(action, target_id)


class _WaitTool(_ActionTool):
    name: ClassVar[str] = "wait"
    description: ClassVar[str] = (
        "Wait without acting. Use to let a page settle after a step that changes it."
    )
    summary: ClassVar[str] = "Wait without acting"
    params: ClassVar[tuple[Param, ...]] = (
        Param("seconds", float, 2.0, min=0, description="How long to wait."),
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        target_id = params["target_id"]
        seconds = params["seconds"]

        async def action(_manager: BrowserManager, tab: str | None) -> bool:
            if tab is None:
                raise RuntimeError("no tab is open, open one with open_url first")
            await asyncio.sleep(seconds)
            return True
        return await self._transact(action, target_id)


class _DownloadClickTool(_ActionTool):
    name: ClassVar[str] = "download_click"
    description: ClassVar[str] = (
        "Click a point on a tab's screen that starts a download, and wait for the file. "
        "Use instead of click when the click brings a file down."
    )
    summary: ClassVar[str] = "Click a point on a tab's screen that starts a download"
    params: ClassVar[tuple[Param, ...]] = (
        Param("x", float, min=0, max=1, description="Position across the screen, 0 at the left, 1 at the right."),
        Param("y", float, min=0, max=1, description="Position down the screen, 0 at the top, 1 at the bottom."),
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        return await self._transact(
            _click_action(params["x"], params["y"]),
            params["target_id"],
            monitor_download=True,
        )


class _RefreshTool(_ActionTool):
    name: ClassVar[str] = "refresh"
    description: ClassVar[str] = (
        "Reload a tab's current page. Use when the page is stuck, frozen, or showing an error state. "
        "The tab navigates to its own current URL."
    )
    summary: ClassVar[str] = "Reload a tab's current page"
    params: ClassVar[tuple[Param, ...]] = (
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        target_id = params["target_id"]

        async def action(manager: BrowserManager, tab: str | None) -> bool:
            if tab is None:
                raise RuntimeError("no tab is open, open one with open_url first")
            url = manager.url(tab)
            if not url:
                raise RuntimeError("the tab has no url to refresh")
            return await manager.navigate(tab, url, verbose=True)
        return await self._transact(action, target_id)


class _DumpPageTool(_ActionTool):
    name: ClassVar[str] = "dump_page"
    description: ClassVar[str] = (
        "Read the current page as markdown from its filtered DOM. "
        "Use when screen text is too small or the page has a lot of text to read."
    )
    summary: ClassVar[str] = "Read a page as markdown"
    params: ClassVar[tuple[Param, ...]] = (
        Param("target_id", str | None, None, description=_TARGET_ID_DESC),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        dump: TabCapture | None = None

        async def action(manager: BrowserManager, tab: str | None) -> bool:
            nonlocal dump
            if tab is None:
                raise RuntimeError("no tab is open, open one with open_url first")
            started_at = datetime.now(UTC)
            snapshot = await manager.snapshot(tab, timeout=_NETWORK_IDLE_TIMEOUT_MS, verbose=True)
            if snapshot is None:
                raise RuntimeError("no snapshot available for tab")
            tree = await snapshot.filtered_tree(verbose=True)
            if tree is None:
                raise RuntimeError("could not read the page DOM")
            markdown = await html_tree_to_markdown(tree)
            dump = TabCapture(
                target_id=tab, title=snapshot.title, url=snapshot.url,
                dom=DomCapture(markdown, CaptureInterval(started_at, datetime.now(UTC))),
            )
            return True

        result = await self._transact(action, params["target_id"], capture_dom=lambda: dump)
        if dump is None:
            return result
        if dump.dom is None:
            return result
        return ToolResult(
            text=(
                f"{result.text}\nDOM capture interval: {dump.dom.interval.started_at.isoformat()} "
                f"to {dump.dom.interval.finished_at.isoformat()}\n"
                f"{_DOM_TARGET_PREFIX}{dump.target_id}\n\n{dump.dom.text}"
            ),
            files=result.files,
            render_type="markdown",
        )


def _build_tools(session: BrowserSession, cap_reached: asyncio.Event, resources_dir: str) -> tuple[Tool, ...]:
    """The browser action tools bound to the session and the cap signal."""
    return (
        _OpenUrlTool(session, cap_reached),
        _ClickTool(session, cap_reached),
        _UploadResourcesTool(session, cap_reached, resources_dir),
        _MoveMouseTool(session, cap_reached),
        _TypeTextTool(session, cap_reached),
        _ReplaceInputTextTool(session, cap_reached),
        _ScrollTool(session, cap_reached),
        _PressKeyTool(session, cap_reached),
        _GoBackTool(session, cap_reached),
        _GoForwardTool(session, cap_reached),
        _ClosePageTool(session, cap_reached),
        _SolveCaptchaTool(session, cap_reached),
        _WaitTool(session, cap_reached),
        _DownloadClickTool(session, cap_reached),
        _RefreshTool(session, cap_reached),
        _DumpPageTool(session, cap_reached),
    )


def _build_agent(tools: tuple[Tool, ...], backend: config.Backend) -> Agent:
    agent = Agent(
        name=_AGENT_NAME,
        system_prompt=build_role_prompt(tools + (task_tool, read_media_or_doc_tool)),
        validator=validate_role_output,
        backend=backend,
        tool_results_by_id=False,
        model_input_filter=_prune_model_history,
    )
    for tool in tools:
        agent.add_tool_adapter(tool)
    agent.add_tool_adapter(task_tool)
    agent.add_tool_adapter(read_media_or_doc_tool, results_by_id=True)
    return agent


async def run_browser_use(
    prompt: str,
    downloads_dir: str,
    resources_dir: str,
) -> BrowserUseResult:
    """Drive the browser to the goal and give back the outcome with the files that came down.

    Args:
        prompt: The goal to carry out in the browser.
        downloads_dir: The folder the session's transaction files land in.
        resources_dir: The resources folder the download paths are reported against.

    Returns:
        The outcome, with the status, the summary, and every file that came down.

    Raises:
        asyncio.CancelledError: When the run is cancelled while it waits on the role.
        MissingSettingError: When neither LLM backend address is set.
    """
    llm_backend = config.backend(config.REASON_THINKING, config.REASON)
    store = CheckpointStore(downloads_dir, resources_root=resources_dir)
    session = BrowserSession(downloads_dir, checkpoint_store=store)
    agent_run: AgentRun | None = None
    cap_task: asyncio.Task[None] | None = None
    result_task: asyncio.Task[str] | None = None
    try:
        await session.open()
        cap_reached = asyncio.Event()
        tools = _build_tools(session, cap_reached, resources_dir) + (CheckpointTool(store),)
        agent = _build_agent(tools, llm_backend)
        agent_run = await agent.run(prompt)
        cap_task = asyncio.create_task(cap_reached.wait())
        result_task = asyncio.create_task(agent_run.task.result())
        done, _ = await asyncio.wait({result_task, cap_task}, return_when=asyncio.FIRST_COMPLETED)
        if result_task in done:
            raw = await result_task
            return _final_result(raw, await session.get_downloaded_files(), resources_dir)
        agent_run.cancel_run_loop()
        downloads = await session.get_downloaded_files()
        return BrowserUseResult("incomplete", "", _resource_paths(downloads, resources_dir))
    except asyncio.CancelledError:
        if agent_run is not None:
            await asyncio.shield(agent_run.task.cancel())
        raise
    finally:
        if cap_task is not None:
            if not cap_task.done():
                cap_task.cancel()
            if not result_task.done():
                result_task.cancel()
            await asyncio.gather(cap_task, result_task, return_exceptions=True)
        try:
            await session.close()
        finally:
            await store.close()
