"""Tests for the browser use runner and its action tools."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from agents.run_config import CallModelData, ModelInputData

from app.agents.browser_use import runner as runner_module
from app.agents.browser_use.checkpoint import CheckpointTool
from app.agents.browser_use.checkpoint_store import CheckpointError, CheckpointStore, TabCapture
from app.agents.browser_use.models import (
    BrowserUseResult,
    Snapshot,
    TabSnapshot,
    TransactionResult,
    TransactionStatus,
)
from app.agents.browser_use.session import Action
from app.lib.shared import config
from app.lib.tool.base import Tool, ToolResult
from app.tools.read_media_or_doc import read_media_or_doc_tool
from app.tools.task_tool import task_tool
from tests.app.conftest import skip_if_no_chrome

_TOOL_NAMES = (
    "open_url",
    "click",
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
)

_HALF = 0.5
_TENTH = 0.1
_THIRD = 0.3
_NETWORK_IDLE_TIMEOUT_MS = 2000
_LIVE_TIMEOUT_S = 600
_COMPLETED_RAW = json.dumps({"status": "completed", "summary": "done"})
_IMPOSSIBLE_RAW = json.dumps({"status": "impossible", "summary": "cannot"})
_LLM_BACKEND_VARS = (
    "REASON_THINKING_BACKEND_URL", "REASON_THINKING_MODEL", "REASON_THINKING_API_KEY",
    "REASON_BACKEND_URL", "REASON_MODEL", "REASON_API_KEY",
)

_ACTION_TOOLS = tuple(name for name in _TOOL_NAMES if name != "open_url")

_TIME_LINE = re.compile(r"^Time: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ([A-Za-z]+|[+-]\d+)$")


def _time_prefixed(text: str) -> str:
    """Assert the first line is the time line, and give back the rest of the text."""
    first_line, _, rest = text.partition("\n")
    assert _TIME_LINE.match(first_line)
    return rest

_SNAPSHOT_TOOLS = ("click", "move_mouse", "type_text", "replace_input_text", "scroll", "press_key", "solve_captcha", "download_click")

_TABS = {
    "tab1": TabSnapshot("tab1", "/tmp/dl/txn_1/01.jpg", '1. tab1 — "Shop" — https://shop.example — image 1'),
    "tab2": TabSnapshot("tab2", None, "2. tab2 — no screenshot"),
}


def _tabs() -> Snapshot:
    return Snapshot(tabs=dict(_TABS))


def _success(message: str = "") -> TransactionResult:
    return TransactionResult(TransactionStatus.SUCCESS, message, _tabs(), ())


def _failed(message: str = "action failed: boom") -> TransactionResult:
    return TransactionResult(TransactionStatus.FAILED, message, _tabs(), ())


def _stale(message: str = "screen changed since you last saw it, action not run") -> TransactionResult:
    return TransactionResult(TransactionStatus.STALE, message, _tabs(), ())


def _cap_reached() -> TransactionResult:
    return TransactionResult(TransactionStatus.CAP_REACHED, "step budget is used up", Snapshot(tabs={}), ())


class FakeSession:
    """A scripted session that records every transaction it gets."""

    def __init__(
        self,
        results: list[TransactionResult],
        downloads: list[str] | None = None,
        manager: FakeManager | None = None,
    ) -> None:
        self._results = results
        self._downloads = list(downloads or [])
        self._manager = manager
        self.opened: int = 0
        self.closed: int = 0
        self.transactions: list[tuple[Action, str | None, bool, bool]] = []
        self.source_actions: list[str] = []
        self.dom_captures: list[TabCapture | None] = []
        self.checkpoint_store: CheckpointStore | None = None
        self.open_error: Exception | None = None
        self.close_error: Exception | None = None

    async def open(self) -> None:
        self.opened += 1
        if self.open_error is not None:
            raise self.open_error

    async def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error

    async def transact(
        self,
        action: Action,
        target_id: str | None,
        *,
        new_tab: bool = False,
        monitor_download: bool = False,
        source_action: str = "action",
        capture_dom: Callable[[], TabCapture | None] | None = None,
    ) -> TransactionResult:
        self.transactions.append((action, target_id, monitor_download, new_tab))
        self.source_actions.append(source_action)
        if self._manager is None:
            return self._results.pop(0)
        manager = self._manager
        tab = (target_id or "").strip()
        if not tab:
            if new_tab:
                tab = None
            else:
                open_tabs = manager.tab_ids()
                if len(open_tabs) > 1:
                    return TransactionResult(
                        TransactionStatus.FAILED,
                        f"action failed: {len(open_tabs)} tabs are open, give the target id of the one to act on",
                        _snapshot_of(manager),
                        (),
                    )
                tab = open_tabs[0] if open_tabs else None
        try:
            ok = await action(manager, tab)
            self.dom_captures.append(capture_dom() if capture_dom is not None else None)
        except Exception as exc:
            return TransactionResult(TransactionStatus.FAILED, f"action failed: {exc}", _snapshot_of(manager), ())
        if not ok:
            return TransactionResult(TransactionStatus.FAILED, "action failed", _snapshot_of(manager), ())
        return TransactionResult(TransactionStatus.SUCCESS, "", _snapshot_of(manager), ())

    async def get_downloaded_files(self) -> list[str]:
        return list(self._downloads)


_FIXED_YEAR = 2026


@pytest.fixture
def _restore_zone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Put the local timezone back after a test that changes TZ."""
    yield
    monkeypatch.undo()
    time.tzset()


class FrozenDatetime(datetime):
    """A datetime whose clock is stuck on one instant, for time format checks."""

    instant: datetime = datetime(_FIXED_YEAR, 1, 15, 12, 0, 0)

    @classmethod
    def now(cls, tz: timezone | None = None) -> datetime:
        return cls.instant.replace(tzinfo=tz)


class FakeSnapshot:
    """A recording stand-in for the page snapshot the action closures call."""

    def __init__(self, manager: FakeManager, target_id: str, move_fails: bool = False) -> None:
        self._manager = manager
        self._target_id = target_id
        self._move_fails = move_fails
        self.title: str = "Page title"
        self.url: str = "https://example.com"

    def _record(self, method: str, args: tuple) -> None:
        self._manager.recorded.setdefault(method, []).append((self._target_id, *args))
        self._manager.order.append(method)

    async def click_at_relative(self, x: float, y: float, verbose: bool = False) -> bool:
        self._record("click_at_relative", (x, y, verbose))
        return True

    async def upload_with_chooser_relative(
        self, x: float, y: float, file_paths: list[str], verbose: bool = False,
    ) -> bool:
        self._record("upload_with_chooser_relative", (x, y, file_paths, verbose))
        return True

    async def move_mouse_to_relative(self, x: float, y: float, verbose: bool = False) -> bool:
        self._record("move_mouse_to_relative", (x, y, verbose))
        return True

    async def type_text_at_relative(
        self, x: float, y: float, text: str, submit: bool = False, verbose: bool = False
    ) -> bool:
        self._record("type_text_at_relative", (x, y, text, submit, verbose))
        return True

    async def replace_input_text_at_relative(
        self, x: float, y: float, old_text: str, new_text: str, verbose: bool = False
    ) -> bool:
        self._record("replace_input_text_at_relative", (x, y, old_text, new_text, verbose))
        return True

    async def move_mouse_to_relative(self, x_ratio: float, y_ratio: float, verbose: bool = False) -> bool:
        self._record("move_mouse_to_relative", (x_ratio, y_ratio, verbose))
        return not self._move_fails

    async def scroll_by_relative(self, dx: float, dy: float, verbose: bool = False) -> bool:
        self._record("scroll_by_relative", (dx, dy, verbose))
        return True

    async def press_key(self, key: str, verbose: bool = False) -> bool:
        self._record("press_key", (key, verbose))
        return True

    async def solve_captcha(self, max_steps: int, verbose: bool = False) -> bool:
        self._record("solve_captcha", (max_steps, verbose))
        return True

    async def filtered_tree(self, verbose: bool = False) -> dict | None:
        self._record("filtered_tree", (verbose,))
        if self._manager.no_tree:
            return None
        return {"role": "web", "children": [{"role": "heading", "name": "Page title"}]}


class FakeManager:
    """A recording stand-in for the browser manager the action closures call."""

    def __init__(
        self,
        tab_ids: tuple[str, ...] = (),
        no_snapshot: bool = False,
        move_fails: bool = False,
        no_tree: bool = False,
    ) -> None:
        self.recorded: dict[str, list[tuple]] = {}
        self.order: list[str] = []
        self._page_seq = 0
        self._tab_ids: list[str] = list(tab_ids)
        self._no_snapshot = no_snapshot
        self._move_fails = move_fails
        self.no_tree = no_tree

    def tab_ids(self) -> list[str]:
        return list(self._tab_ids)

    def add_tab(self, target_id: str) -> None:
        self._tab_ids.append(target_id)

    def _record(self, method: str, args: tuple) -> None:
        self.recorded.setdefault(method, []).append(args)
        self.order.append(method)

    async def new_page(self, verbose: bool = False) -> str:
        self._record("new_page", (verbose,))
        self._page_seq += 1
        return f"new{self._page_seq}"

    async def navigate(
        self,
        target_id: str,
        url: str,
        wait_until: str | None = None,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> bool:
        self._record("navigate", (target_id, url, wait_until, timeout, verbose))
        return True

    def url(self, target_id: str, verbose: bool = False) -> str:
        self._record("url", (target_id, verbose))
        return "https://example.com"

    async def snapshot(self, target_id: str, timeout: int | None = None, verbose: bool = False) -> FakeSnapshot | None:
        self._record("snapshot", (target_id, timeout, verbose))
        if self._no_snapshot:
            return None
        return FakeSnapshot(self, target_id, move_fails=self._move_fails)

    async def wait_for_load(
        self,
        target_id: str,
        state: str | None = None,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> bool:
        self._record("wait_for_load", (target_id, state, timeout, verbose))
        return True

    async def go_back(
        self,
        target_id: str,
        wait_until: str | None = None,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> bool:
        self._record("go_back", (target_id, wait_until, timeout, verbose))
        return True

    async def go_forward(
        self,
        target_id: str,
        wait_until: str | None = None,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> bool:
        self._record("go_forward", (target_id, wait_until, timeout, verbose))
        return True

    async def close_page(self, target_id: str, verbose: bool = False) -> bool:
        self._record("close_page", (target_id, verbose))
        return True


def _snapshot_of(manager: FakeManager) -> Snapshot:
    """One canned tab line per open tab of the manager, in tab order."""
    tabs: dict[str, TabSnapshot] = {}
    for index, target_id in enumerate(manager.tab_ids(), start=1):
        tabs[target_id] = TabSnapshot(target_id, None, f"{index}. {target_id} — 'T' — https://x — image {index}")
    return Snapshot(tabs=tabs)


_PARAMS: dict[str, dict] = {
    "open_url": {"target_id": "tab1", "url": "https://a.example"},
    "click": {"target_id": "tab1", "x": 0.2, "y": 0.4},
    "move_mouse": {"target_id": "tab1", "x": 0.3, "y": 0.6},
    "type_text": {"target_id": "tab1", "x": _TENTH, "y": _THIRD, "text": "hello"},
    "replace_input_text": {"target_id": "tab1", "x": _TENTH, "y": _THIRD, "old_text": "helo", "new_text": "hello"},
    "scroll": {"target_id": "tab1", "dx": -_HALF, "dy": 1.0},
    "press_key": {"target_id": "tab1", "key": "Enter"},
    "go_back": {"target_id": "tab1"},
    "go_forward": {"target_id": "tab1"},
    "close_page": {"target_id": "tab1"},
    "solve_captcha": {"target_id": "tab1"},
    "wait": {"target_id": "tab1", "seconds": 0.0},
    "download_click": {"target_id": "tab1", "x": 0.7, "y": 0.9},
    "refresh": {"target_id": "tab1"},
    "dump_page": {"target_id": "tab1"},
}


def _snapshot_flow(tab: str, method: str, call: tuple, *, waits_for_load: bool = True) -> dict[str, list[tuple]]:
    """The expected record for a snapshot-based action on a tab."""
    record = {
        "snapshot": [(tab, _NETWORK_IDLE_TIMEOUT_MS, True)],
        method: [call],
    }
    if waits_for_load:
        record["wait_for_load"] = [(tab, "networkidle", _NETWORK_IDLE_TIMEOUT_MS, True)]
    return record


_EXPECTED_RECORD: dict[str, dict[str, list[tuple]]] = {
    "open_url": {
        "navigate": [("tab1", "https://a.example", None, None, True)],
    },
    "click": _snapshot_flow("tab1", "click_at_relative", ("tab1", 0.2, 0.4, True), waits_for_load=False),
    "move_mouse": _snapshot_flow("tab1", "move_mouse_to_relative", ("tab1", 0.3, 0.6, True)),
    "type_text": _snapshot_flow("tab1", "type_text_at_relative", ("tab1", _TENTH, _THIRD, "hello", False, True)),
    "replace_input_text": _snapshot_flow(
        "tab1", "replace_input_text_at_relative", ("tab1", _TENTH, _THIRD, "helo", "hello", True)
    ),
    "scroll": _snapshot_flow("tab1", "scroll_by_relative", ("tab1", -_HALF, 1.0, True)),
    "press_key": _snapshot_flow("tab1", "press_key", ("tab1", "Enter", True)),
    "go_back": {"go_back": [("tab1", None, None, True)]},
    "go_forward": {"go_forward": [("tab1", None, None, True)]},
    "close_page": {"close_page": [("tab1", True)]},
    "solve_captcha": _snapshot_flow("tab1", "solve_captcha", ("tab1", 5, True)),
    "wait": {},
    "download_click": _snapshot_flow("tab1", "click_at_relative", ("tab1", 0.7, 0.9, True), waits_for_load=False),
    "refresh": {
        "url": [("tab1", False)],
        "navigate": [("tab1", "https://example.com", None, None, True)],
    },
    "dump_page": {
        "snapshot": [("tab1", _NETWORK_IDLE_TIMEOUT_MS, True)],
        "filtered_tree": [("tab1", True)],
    },
}


def _record_on_tab(record: dict[str, list[tuple]], tab: str) -> dict[str, list[tuple]]:
    """The same expected record with the first tab of every call swapped out."""
    return {method: [((tab,) + call[1:]) for call in calls] for method, calls in record.items()}


_SOLE_RECORD = {
    name: _record_on_tab(record, "only") for name, record in _EXPECTED_RECORD.items() if name != "open_url"
}

_NEW_PAGE_RECORD: dict[str, list[tuple]] = {
    "new_page": [(True,)],
    "navigate": [("new1", "https://a.example", None, None, True)],
}

_EXPECTED_ORDER: dict[str, list[str]] = {
    "open_url": ["navigate"],
    "click": ["snapshot", "click_at_relative"],
    "move_mouse": ["snapshot", "move_mouse_to_relative", "wait_for_load"],
    "type_text": ["snapshot", "type_text_at_relative", "wait_for_load"],
    "replace_input_text": ["snapshot", "replace_input_text_at_relative", "wait_for_load"],
    "scroll": ["snapshot", "scroll_by_relative", "wait_for_load"],
    "press_key": ["snapshot", "press_key", "wait_for_load"],
    "go_back": ["go_back"],
    "go_forward": ["go_forward"],
    "close_page": ["close_page"],
    "solve_captcha": ["snapshot", "solve_captcha", "wait_for_load"],
    "wait": [],
    "download_click": ["snapshot", "click_at_relative"],
    "refresh": ["url", "navigate"],
    "dump_page": ["snapshot", "filtered_tree"],
}


class FakeTask:
    """The fake agent task handle the runner awaits."""

    def __init__(self, value: str | None, error: Exception | None, hang: bool, on_result: Callable | None) -> None:
        self._value = value
        self._error = error
        self._hang = hang
        self._on_result = on_result
        self.cancelled: int = 0

    async def result(self) -> str:
        if self._on_result is not None:
            await self._on_result()
        if self._hang:
            await asyncio.Event().wait()
        if self._error is not None:
            raise self._error
        return self._value

    async def cancel(self) -> None:
        self.cancelled += 1


class FakeAgentRun:
    """The fake per-run handle the runner races against."""

    def __init__(self, task: FakeTask) -> None:
        self.task = task
        self.cancel_loop: int = 0

    def cancel_run_loop(self) -> None:
        self.cancel_loop += 1


class FakeAgent:
    """A recording stand-in for the engine's Agent."""

    def __init__(self) -> None:
        self.agent_run: FakeAgentRun | None = None
        self.tools: list[Tool] = []
        self.results_by_id: dict[Tool, bool | None] = {}
        self.prompt: str | None = None

    def add_tool_adapter(self, tool: Tool, results_by_id: bool | None = None) -> FakeAgent:
        self.tools.append(tool)
        self.results_by_id[tool] = results_by_id
        return self

    async def run(self, prompt: str) -> FakeAgentRun:
        self.prompt = prompt
        return self.agent_run


def _wire_agent(monkeypatch: pytest.MonkeyPatch, agent: FakeAgent) -> MagicMock:
    """Point the runner's Agent at the fake, and give back the recording mock."""
    agent_mock = MagicMock(return_value=agent)
    monkeypatch.setattr(runner_module, "Agent", agent_mock)
    return agent_mock


def _wire_session(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> dict[str, str]:
    """Point the runner's BrowserSession at the fake, recording the downloads dir."""
    seen: dict[str, str] = {}

    def make_session(downloads_dir: str, *, checkpoint_store: CheckpointStore) -> FakeSession:
        seen["downloads_dir"] = downloads_dir
        session.checkpoint_store = checkpoint_store
        return session

    monkeypatch.setattr(runner_module, "BrowserSession", make_session)
    return seen


def _clear_llm_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every variable of the thinking and plain backends."""
    for name in _LLM_BACKEND_VARS:
        monkeypatch.delenv(name, raising=False)


def _set_thinking_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the thinking backend at the test value, and clear the plain backend."""
    _clear_llm_backends(monkeypatch)
    monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://think.test/v1")


def _tool(name: str, tools: tuple[Tool, ...]) -> Tool:
    """The named tool from a built tool list."""
    return next(tool for tool in tools if tool.name == name)


async def _finish_agent(
    monkeypatch: pytest.MonkeyPatch,
    session: FakeSession,
    raw: str,
    resources_dir: str = "/tmp/res",
) -> tuple[FakeAgent, MagicMock, BrowserUseResult, dict[str, str]]:
    """Wire the fakes, run the runner to a finished agent, and give back the agent, the mock, the result, and the downloads dir."""
    agent = FakeAgent()
    agent.agent_run = FakeAgentRun(FakeTask(raw, None, False, None))
    agent_mock = _wire_agent(monkeypatch, agent)
    seen = _wire_session(monkeypatch, session)
    result = await runner_module.run_browser_use("goal", "/tmp/dl", resources_dir)
    return agent, agent_mock, result, seen


async def _wired_tools(
    monkeypatch: pytest.MonkeyPatch, session: FakeSession, resources_dir: str = "/tmp/res",
) -> tuple[Tool, ...]:
    """Run the runner over the fakes and give back the tools it built."""
    _set_thinking_backend(monkeypatch)
    agent, _, _, _ = await _finish_agent(monkeypatch, session, _COMPLETED_RAW, resources_dir)
    return tuple(agent.tools)


async def _run_tool(tool: Tool, params: dict) -> ToolResult:
    """Run the tool with the params and give back the result."""
    task = await tool.run(params)
    return await task.result()


async def _plain_action_of(session: FakeSession, tool: Tool, params: dict) -> Action:
    """Run the tool, check the transaction is plain and aimed at tab1, and give back its action."""
    await _run_tool(tool, params)
    action, target_id, monitor_download, new_tab = session.transactions[0]
    assert target_id == "tab1"
    assert monitor_download is False
    assert new_tab is False
    return action


def _function_call_item(call_id: str, name: str) -> dict:
    """A function call item the model sent."""
    return {"type": "function_call", "name": name, "call_id": call_id, "arguments": "{}"}


def _function_output_item(call_id: str, output: str | list) -> dict:
    """A tool output item paired with the call id."""
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _image_part(url: str = "data:image/png;base64,AAAA") -> dict:
    """An image part of a tool output."""
    return {"type": "input_image", "image_url": url}


def _text_part(text: str) -> dict:
    """A text part of a tool output."""
    return {"type": "input_text", "text": text}


def _call_data(items: list[dict]) -> CallModelData:
    """Model input data built by hand, with a dummy agent and no context."""
    return CallModelData(
        model_data=ModelInputData(input=items, instructions="the role prompt"),
        agent=object(),
        context=None,
    )


async def _wired_filter(monkeypatch: pytest.MonkeyPatch) -> Callable[[CallModelData], ModelInputData]:
    """The model input filter the runner hands the agent at build time."""
    _set_thinking_backend(monkeypatch)
    _, agent_mock, _, _ = await _finish_agent(monkeypatch, FakeSession([]), _COMPLETED_RAW)
    filter_fn = agent_mock.call_args.kwargs["model_input_filter"]
    assert callable(filter_fn)
    return filter_fn


class TestActionTools:
    async def test_dump_page_reads_filtered_tree_as_markdown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = FakeManager(tab_ids=("tab1",))
        tool = _tool("dump_page", await _wired_tools(monkeypatch, FakeSession([], manager=manager), str(tmp_path)))

        result = await _run_tool(tool, {"target_id": "tab1"})

        assert result.render_type == "markdown"
        assert "DOM dump target: tab1" in result.text
        assert "\n\n## Page title" in result.text
        assert "status: success" in result.text
        assert result.files is None
        assert not (tmp_path / "browser_use" / "dumps").exists()
        assert manager.recorded == _EXPECTED_RECORD["dump_page"]

    async def test_dump_page_failure_has_no_dom_content(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = FakeManager(tab_ids=("tab1",), no_snapshot=True)
        tool = _tool("dump_page", await _wired_tools(monkeypatch, FakeSession([], manager=manager)))

        result = await _run_tool(tool, {"target_id": "tab1"})

        assert "status: failed" in result.text
        assert "DOM dump target:" not in result.text

    async def test_dump_page_fails_when_filtered_tree_is_missing(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = FakeManager(tab_ids=("tab1",), no_tree=True)
        tool = _tool("dump_page", await _wired_tools(monkeypatch, FakeSession([], manager=manager)))

        result = await _run_tool(tool, {"target_id": "tab1"})

        assert "status: failed, action failed: could not read the page DOM" in result.text
        assert "DOM dump target:" not in result.text
        assert manager.recorded == _EXPECTED_RECORD["dump_page"]

    async def test_upload_resources_uses_resource_file_and_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An upload sends an existing resource file through the snapshot."""
        resource = tmp_path / "upload" / "photo.jpg"
        resource.parent.mkdir()
        resource.write_bytes(b"photo")
        second = tmp_path / "upload" / "notes.txt"
        second.write_text("notes")
        manager = FakeManager(tab_ids=("tab1",))
        session = FakeSession([], manager=manager)
        tool = _tool("upload_resources", await _wired_tools(monkeypatch, session, str(tmp_path)))

        result = await _run_tool(tool, {
            "x": _TENTH, "y": _THIRD,
            "resource_paths": ["upload/photo.jpg", "upload/notes.txt"], "target_id": "tab1",
        })

        assert "status: success" in result.text
        assert manager.recorded == _snapshot_flow(
            "tab1", "upload_with_chooser_relative", ("tab1", _TENTH, _THIRD, [str(resource), str(second)], True),
        )
        assert manager.order == ["snapshot", "upload_with_chooser_relative", "wait_for_load"]

    @pytest.mark.parametrize("paths", [[], ["/tmp/outside"], ["../outside"], ["missing.txt"]])
    async def test_upload_resources_rejects_invalid_paths_before_click(
        self, paths: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Invalid file paths fail without touching the page."""
        manager = FakeManager(tab_ids=("tab1",))
        tool = _tool(
            "upload_resources", await _wired_tools(monkeypatch, FakeSession([], manager=manager), str(tmp_path)),
        )

        result = await _run_tool(tool, {
            "x": _TENTH, "y": _THIRD, "resource_paths": paths, "target_id": "tab1",
        })

        assert "status: failed" in result.text
        assert manager.recorded == {}

    @pytest.mark.parametrize("name", _TOOL_NAMES)
    async def test_tool_is_tracked_and_renders_text(self, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        session = FakeSession([])
        tool = _tool(name, await _wired_tools(monkeypatch, session))
        assert tool.tracked is True
        assert tool.result_render_type == "text"

    @pytest.mark.parametrize("name", _TOOL_NAMES)
    async def test_tool_result_keeps_the_screenshots(self, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        session = FakeSession([_success()])
        result = await _run_tool(_tool(name, await _wired_tools(monkeypatch, session)), _PARAMS[name])
        assert result.files == ("/tmp/dl/txn_1/01.jpg",)

    @pytest.mark.parametrize("name", _TOOL_NAMES)
    async def test_tool_calls_transact_with_the_right_action(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([_success()])
        await _run_tool(_tool(name, await _wired_tools(monkeypatch, session)), _PARAMS[name])
        assert len(session.transactions) == 1
        action, target_id, monitor_download, new_tab = session.transactions[0]
        assert target_id == "tab1"
        assert monitor_download is (name == "download_click")
        assert new_tab is (name == "open_url")
        manager = FakeManager()
        if name in ("click", "download_click"):
            monkeypatch.setattr(runner_module.asyncio, "sleep", AsyncMock())
        assert await action(manager, "tab1") is True
        assert manager.recorded == _EXPECTED_RECORD[name]
        assert manager.order == _EXPECTED_ORDER[name]

    @pytest.mark.parametrize("name", ("click", "download_click"))
    async def test_click_delay_allows_popup_before_result_snapshot(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = FakeManager(tab_ids=("tab1",))
        session = FakeSession([], manager=manager)
        delays: list[float] = []

        async def discover_popup(seconds: float) -> None:
            delays.append(seconds)
            manager.add_tab("popup")

        monkeypatch.setattr(runner_module.asyncio, "sleep", discover_popup)
        result = await _run_tool(_tool(name, await _wired_tools(monkeypatch, session)), _PARAMS[name])
        assert delays == [2.0]
        assert "2. popup" in result.text
        assert "wait_for_load" not in manager.recorded

    async def test_type_text_with_submit_passes_the_flag_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([_success()])
        tool = _tool("type_text", await _wired_tools(monkeypatch, session))
        submit = next(p for p in tool.params if p.name == "submit")
        assert submit.default is False
        params = dict(_PARAMS["type_text"], submit=True)
        action = await _plain_action_of(session, tool, params)
        manager = FakeManager()
        assert await action(manager, "tab1") is True
        assert manager.recorded["type_text_at_relative"] == [("tab1", _TENTH, _THIRD, "hello", True, True)]
        assert manager.order == _EXPECTED_ORDER["type_text"]

    async def test_replace_input_text_with_empty_new_text_passes_it_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([_success()])
        tool = _tool("replace_input_text", await _wired_tools(monkeypatch, session))
        params = dict(_PARAMS["replace_input_text"], old_text="remove me", new_text="")
        action = await _plain_action_of(session, tool, params)
        manager = FakeManager()
        assert await action(manager, "tab1") is True
        assert manager.recorded["replace_input_text_at_relative"] == [("tab1", _TENTH, _THIRD, "remove me", "", True)]
        assert manager.order == _EXPECTED_ORDER["replace_input_text"]

    async def test_scroll_with_an_anchor_moves_the_mouse_before_scrolling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([_success()])
        tool = _tool("scroll", await _wired_tools(monkeypatch, session))
        anchor_x = next(p for p in tool.params if p.name == "x")
        anchor_y = next(p for p in tool.params if p.name == "y")
        assert anchor_x.default is None
        assert anchor_y.default is None
        params = dict(_PARAMS["scroll"], x=_THIRD, y=_HALF)
        action = await _plain_action_of(session, tool, params)
        manager = FakeManager()
        assert await action(manager, "tab1") is True
        assert manager.recorded == {
            "snapshot": [("tab1", _NETWORK_IDLE_TIMEOUT_MS, True)],
            "move_mouse_to_relative": [("tab1", _THIRD, _HALF, True)],
            "scroll_by_relative": [("tab1", -_HALF, 1.0, True)],
            "wait_for_load": [("tab1", "networkidle", _NETWORK_IDLE_TIMEOUT_MS, True)],
        }
        assert manager.order == ["snapshot", "move_mouse_to_relative", "scroll_by_relative", "wait_for_load"]

    @pytest.mark.parametrize(
        "given",
        [pytest.param("x", id="lone_x"), pytest.param("y", id="lone_y")],
    )
    async def test_scroll_with_a_lone_anchor_coordinate_reports_failed(
        self, given: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = FakeManager(tab_ids=("only",))
        session = FakeSession([], manager=manager)
        params = {key: value for key, value in _PARAMS["scroll"].items() if key != "target_id"}
        params[given] = _THIRD
        result = await _run_tool(_tool("scroll", await _wired_tools(monkeypatch, session)), params)
        assert "status: failed, action failed: x and y must be given together" in result.text
        assert manager.recorded == {}

    async def test_scroll_with_an_unreachable_anchor_reports_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = FakeManager(tab_ids=("only",), move_fails=True)
        session = FakeSession([], manager=manager)
        params = {key: value for key, value in _PARAMS["scroll"].items() if key != "target_id"}
        params.update(x=_THIRD, y=_HALF)
        result = await _run_tool(_tool("scroll", await _wired_tools(monkeypatch, session)), params)
        assert "status: failed, action failed: could not move the mouse to the scroll anchor" in result.text
        assert manager.recorded == {
            "snapshot": [("only", _NETWORK_IDLE_TIMEOUT_MS, True)],
            "move_mouse_to_relative": [("only", _THIRD, _HALF, True)],
        }

    async def test_scroll_anchor_docs_name_the_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = FakeSession([])
        tool = _tool("scroll", await _wired_tools(monkeypatch, session))
        assert "nested scrollable region" in tool.description
        anchor_x = next(p for p in tool.params if p.name == "x")
        anchor_y = next(p for p in tool.params if p.name == "y")
        assert anchor_x.default is None
        assert anchor_y.default is None
        assert "Give it with y" in anchor_x.description
        assert "Give it with x" in anchor_y.description

    @pytest.mark.parametrize("target", [pytest.param(None, id="no_target"), pytest.param("", id="blank_target")])
    async def test_open_url_without_a_tab_opens_a_new_page(
        self, target: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([_success()])
        params: dict = {"url": "https://a.example"}
        if target is not None:
            params["target_id"] = target
        await _run_tool(_tool("open_url", await _wired_tools(monkeypatch, session)), params)
        action, target_id, monitor_download, new_tab = session.transactions[0]
        assert target_id in (None, "")
        assert monitor_download is False
        assert new_tab is True
        manager = FakeManager()
        assert await action(manager, None) is True
        assert manager.recorded == _NEW_PAGE_RECORD

    async def test_open_url_with_a_blank_target_and_one_tab_still_opens_a_new_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([_success()])
        await _run_tool(_tool("open_url", await _wired_tools(monkeypatch, session)), {"url": "https://a.example"})
        action, target_id, monitor_download, new_tab = session.transactions[0]
        assert target_id is None
        assert monitor_download is False
        assert new_tab is True
        manager = FakeManager(tab_ids=("existing",))
        assert await action(manager, None) is True
        assert manager.recorded == _NEW_PAGE_RECORD

    async def test_open_url_with_a_blank_target_and_two_tabs_still_opens_a_new_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = FakeManager(tab_ids=("a", "b"))
        session = FakeSession([], manager=manager)
        await _run_tool(_tool("open_url", await _wired_tools(monkeypatch, session)), {"url": "https://a.example"})
        _action, target_id, monitor_download, new_tab = session.transactions[0]
        assert target_id is None
        assert monitor_download is False
        assert new_tab is True
        assert manager.recorded == _NEW_PAGE_RECORD

    async def test_result_text_carries_captions_and_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = FakeSession([_success()])
        result = await _run_tool(_tool("click", await _wired_tools(monkeypatch, session)), _PARAMS["click"])
        assert '1. tab1 — "Shop" — https://shop.example — image 1' in result.text
        assert "2. tab2 — no screenshot" in result.text
        assert "status: success" in result.text

    @pytest.mark.parametrize("name", _ACTION_TOOLS)
    @pytest.mark.parametrize(
        "target",
        [
            pytest.param(None, id="no_target"),
            pytest.param("   ", id="blank_target"),
        ],
    )
    async def test_action_without_a_named_target_runs_on_the_sole_open_tab(
        self, name: str, target: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([_success()])
        params = {key: value for key, value in _PARAMS[name].items() if key != "target_id"}
        if target is not None:
            params["target_id"] = target
        await _run_tool(_tool(name, await _wired_tools(monkeypatch, session)), params)
        action, target_id, monitor_download, new_tab = session.transactions[0]
        assert target_id == target
        assert monitor_download is (name == "download_click")
        assert new_tab is False
        manager = FakeManager(tab_ids=("only",))
        if name in ("click", "download_click"):
            monkeypatch.setattr(runner_module.asyncio, "sleep", AsyncMock())
        assert await action(manager, "only") is True
        assert manager.recorded == _SOLE_RECORD[name]
        assert manager.order == _EXPECTED_ORDER[name]

    @pytest.mark.parametrize("name", _ACTION_TOOLS)
    @pytest.mark.parametrize(
        "tabs,match",
        [
            pytest.param((), "no tab is open, open one with open_url first", id="no_tabs"),
            pytest.param(("a", "b"), "2 tabs are open, give the target id of the one to act on", id="two_tabs"),
        ],
    )
    async def test_action_without_a_target_reports_failed_without_a_sole_tab(
        self,
        name: str,
        tabs: tuple[str, ...],
        match: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = FakeManager(tab_ids=tabs)
        session = FakeSession([], manager=manager)
        params = {key: value for key, value in _PARAMS[name].items() if key != "target_id"}
        result = await _run_tool(_tool(name, await _wired_tools(monkeypatch, session)), params)
        assert f"status: failed, action failed: {match}" in result.text

    @pytest.mark.parametrize("name", _SNAPSHOT_TOOLS)
    async def test_action_reports_failed_when_the_tab_has_no_snapshot(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = FakeManager(tab_ids=("only",), no_snapshot=True)
        session = FakeSession([], manager=manager)
        params = {key: value for key, value in _PARAMS[name].items() if key != "target_id"}
        result = await _run_tool(_tool(name, await _wired_tools(monkeypatch, session)), params)
        assert "status: failed, action failed: no snapshot available for tab" in result.text

    async def test_failed_transaction_hands_back_the_tool_result_with_the_screenshots(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([_failed()])
        result = await _run_tool(_tool("click", await _wired_tools(monkeypatch, session)), _PARAMS["click"])
        assert '1. tab1 — "Shop" — https://shop.example — image 1' in result.text
        assert "2. tab2 — no screenshot" in result.text
        assert "status: failed, action failed: boom" in result.text
        assert result.files == ("/tmp/dl/txn_1/01.jpg",)

    async def test_stale_transaction_still_hands_back_the_tool_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([_stale()])
        result = await _run_tool(_tool("click", await _wired_tools(monkeypatch, session)), _PARAMS["click"])
        assert "status: stale, screen changed since you last saw it, action not run" in result.text
        assert result.files == ("/tmp/dl/txn_1/01.jpg",)

    @pytest.mark.parametrize("name", _ACTION_TOOLS)
    async def test_action_target_id_docs_name_the_contract(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([])
        tool = _tool(name, await _wired_tools(monkeypatch, session))
        target = next(p for p in tool.params if p.name == "target_id")
        assert target.default is None
        assert "The tab to act on" in target.description
        assert "fails if no such tab exists" in target.description
        assert "runs on the single open tab and fails if zero or more than one tab is open" in target.description
        assert "the first token on the tab's line in the step result" in target.description

    async def test_open_url_target_id_docs_name_the_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = FakeSession([])
        tool = _tool("open_url", await _wired_tools(monkeypatch, session))
        target = next(p for p in tool.params if p.name == "target_id")
        assert "The tab to open the page in" in target.description
        assert "fails if no such tab exists" in target.description
        assert "a new tab is opened" in target.description
        assert "the first token on the tab's line in the step result" in target.description

    async def test_cap_reached_result_still_hands_back_the_tool_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([_cap_reached()])
        result = await _run_tool(_tool("click", await _wired_tools(monkeypatch, session)), _PARAMS["click"])
        assert "status: cap_reached, step budget is used up" in result.text
        assert result.files is None

    @pytest.mark.parametrize(
        "transaction",
        [
            pytest.param(_success(), id="success"),
            pytest.param(_failed(), id="failed"),
            pytest.param(_stale(), id="stale"),
            pytest.param(_cap_reached(), id="cap_reached"),
        ],
    )
    async def test_result_starts_with_the_time_line(
        self, transaction: TransactionResult, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([transaction])
        result = await _run_tool(_tool("click", await _wired_tools(monkeypatch, session)), _PARAMS["click"])
        _time_prefixed(result.text)

    async def test_time_line_keeps_the_rest_of_the_result_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([_success()])
        result = await _run_tool(_tool("click", await _wired_tools(monkeypatch, session)), _PARAMS["click"])
        assert _time_prefixed(result.text) == (
            '1. tab1 — "Shop" — https://shop.example — image 1\n'
            "2. tab2 — no screenshot\n"
            "status: success"
        )

    async def test_time_line_appears_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([_stale()])
        result = await _run_tool(_tool("click", await _wired_tools(monkeypatch, session)), _PARAMS["click"])
        assert len([line for line in result.text.splitlines() if _TIME_LINE.match(line)]) == 1

    async def test_dump_page_result_starts_with_the_time_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = FakeManager(tab_ids=("tab1",))
        tool = _tool("dump_page", await _wired_tools(monkeypatch, FakeSession([], manager=manager), str(tmp_path)))
        result = await _run_tool(tool, {"target_id": "tab1"})
        rest = _time_prefixed(result.text)
        assert "DOM dump target: tab1" in rest
        assert "status: success" in rest

    @pytest.mark.parametrize(
        "zone,instant,expected_line",
        [
            pytest.param(
                "Europe/Berlin", datetime(_FIXED_YEAR, 1, 15, 12, 0, 0), "Time: 2026-01-15 12:00:00 CET", id="winter"
            ),
            pytest.param(
                "Europe/Berlin", datetime(_FIXED_YEAR, 7, 15, 12, 0, 0), "Time: 2026-07-15 12:00:00 CEST", id="summer"
            ),
            pytest.param("UTC", datetime(_FIXED_YEAR, 7, 15, 12, 0, 0), "Time: 2026-07-15 12:00:00 UTC", id="utc"),
            pytest.param(None, datetime(_FIXED_YEAR, 7, 15, 12, 0, 0), "Time: 2026-07-15 12:00:00 UTC", id="unset"),
        ],
    )
    async def test_result_time_line_uses_the_tz_zone(
        self,
        zone: str | None,
        instant: datetime,
        expected_line: str,
        monkeypatch: pytest.MonkeyPatch,
        _restore_zone: None,
    ) -> None:
        if zone is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", zone)
        time.tzset()
        monkeypatch.setattr(runner_module, "datetime", FrozenDatetime)
        monkeypatch.setattr(FrozenDatetime, "instant", instant)
        session = FakeSession([_success()])
        result = await _run_tool(_tool("click", await _wired_tools(monkeypatch, session)), _PARAMS["click"])
        first_line, _, _ = result.text.partition("\n")
        assert first_line == expected_line


class TestRunBrowserUse:
    async def test_checkpoint_tool_is_inline_and_store_closes_after_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_thinking_backend(monkeypatch)
        session = FakeSession([])
        agent, _, _, _ = await _finish_agent(monkeypatch, session, _COMPLETED_RAW)
        tool = _tool("checkpoint", tuple(agent.tools))
        assert isinstance(tool, CheckpointTool)
        assert agent.results_by_id[tool] is None
        with pytest.raises(CheckpointError, match="store is closed"):
            await session.checkpoint_store.list()

    @pytest.mark.parametrize("failure", ["open", "close"])
    async def test_store_closes_when_session_lifecycle_raises(
        self, monkeypatch: pytest.MonkeyPatch, failure: str,
    ) -> None:
        _set_thinking_backend(monkeypatch)
        session = FakeSession([])
        if failure == "open":
            session.open_error = RuntimeError("open failed")
        else:
            session.close_error = RuntimeError("close failed")
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            await _finish_agent(monkeypatch, session, _COMPLETED_RAW)
        assert session.closed == 1
        with pytest.raises(CheckpointError, match="store is closed"):
            await session.checkpoint_store.list()

    async def test_agent_finish_first_returns_the_validated_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_thinking_backend(monkeypatch)
        session = FakeSession([], downloads=["/tmp/res/downloads/txn_1/download"])
        _, _, result, seen = await _finish_agent(monkeypatch, session, _COMPLETED_RAW)
        assert result == BrowserUseResult("completed", "done", ("downloads/txn_1/download",))
        assert seen == {"downloads_dir": "/tmp/dl"}

    async def test_agent_is_built_with_the_role_and_the_sixteen_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_thinking_backend(monkeypatch)
        agent, agent_mock, _, _ = await _finish_agent(monkeypatch, FakeSession([]), _IMPOSSIBLE_RAW)
        kwargs = agent_mock.call_args.kwargs
        assert kwargs["name"] == "browser_use"
        assert kwargs["validator"] is runner_module.validate_role_output
        assert kwargs["backend"] == config.Backend(url="http://think.test/v1", model=None, api_key=None)
        assert kwargs["tool_results_by_id"] is False
        assert callable(kwargs["model_input_filter"])
        assert [tool.name for tool in agent.tools] == (
            list(_TOOL_NAMES[:2]) + ["upload_resources"] + list(_TOOL_NAMES[2:]) + ["checkpoint", "task", "read_media_or_doc"]
        )
        assert agent.prompt == "goal"
        assert "target id" in kwargs["system_prompt"]

    async def test_reading_tools_are_added_with_their_opt_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_thinking_backend(monkeypatch)
        agent, _, _, _ = await _finish_agent(monkeypatch, FakeSession([]), _COMPLETED_RAW)
        assert any(tool is task_tool for tool in agent.tools)
        assert any(tool is read_media_or_doc_tool for tool in agent.tools)
        assert agent.results_by_id[task_tool] is None
        assert agent.results_by_id[read_media_or_doc_tool] is True

    @pytest.mark.parametrize("name", _TOOL_NAMES)
    async def test_action_tools_are_added_plain(self, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_thinking_backend(monkeypatch)
        agent, _, _, _ = await _finish_agent(monkeypatch, FakeSession([]), _COMPLETED_RAW)
        tool = _tool(name, tuple(agent.tools))
        assert agent.results_by_id[tool] is None

    async def test_task_is_the_bare_registered_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_thinking_backend(monkeypatch)
        agent, _, _, _ = await _finish_agent(monkeypatch, FakeSession([]), _COMPLETED_RAW)
        tool = next(t for t in agent.tools if t.name == "task")
        assert tool is task_tool
        assert tool.name == "task"
        action = next(p for p in task_tool.params if p.name == "action")
        assert action.enum == ("status", "result", "logs", "cancel")

    async def test_backend_carries_the_thinking_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_thinking_backend(monkeypatch)
        monkeypatch.setenv("REASON_THINKING_MODEL", "deep")
        monkeypatch.setenv("REASON_BACKEND_URL", "http://plain.test")
        monkeypatch.setenv("REASON_MODEL", "fast")
        _, agent_mock, _, _ = await _finish_agent(monkeypatch, FakeSession([]), _COMPLETED_RAW)
        expected = config.Backend(url="http://think.test/v1", model="deep", api_key=None)
        assert agent_mock.call_args.kwargs["backend"] == expected

    async def test_backend_falls_back_to_the_plain_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_llm_backends(monkeypatch)
        monkeypatch.setenv("REASON_THINKING_MODEL", "deep")
        monkeypatch.setenv("REASON_BACKEND_URL", "http://plain.test")
        monkeypatch.setenv("REASON_MODEL", "fast")
        _, agent_mock, _, _ = await _finish_agent(monkeypatch, FakeSession([]), _COMPLETED_RAW)
        expected = config.Backend(url="http://plain.test/v1", model="fast", api_key=None)
        assert agent_mock.call_args.kwargs["backend"] == expected

    async def test_missing_backends_raise(self, monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]) -> None:
        _clear_llm_backends(monkeypatch)
        session = FakeSession([])
        _wire_session(monkeypatch, session)
        message = "REASON_THINKING_BACKEND_URL or REASON_BACKEND_URL is not set: needed by research and browser_use"
        with pytest.raises(config.MissingSettingError, match=f"^{message}$"):
            await runner_module.run_browser_use("goal", "/tmp/dl", "/tmp/res")
        assert session.opened == 0
        assert f"[config] {message}\n" in capsys.readouterr().err

    async def test_download_files_outside_the_resources_dir_stay_absolute(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_thinking_backend(monkeypatch)
        session = FakeSession([], downloads=["/tmp/other/x.pdf"])
        _, _, result, _ = await _finish_agent(monkeypatch, session, _COMPLETED_RAW)
        assert result.files == ("/tmp/other/x.pdf",)

    async def test_cap_first_stops_the_loop_and_returns_incomplete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_thinking_backend(monkeypatch)
        session = FakeSession([_cap_reached()], downloads=["/tmp/res/downloads/txn_2/download"])
        _wire_session(monkeypatch, session)
        agent = FakeAgent()

        async def on_result() -> None:
            tool = next(t for t in agent.tools if t.name == "click")
            await tool.run({"target_id": "tab1", "x": _HALF, "y": _HALF})

        agent.agent_run = FakeAgentRun(FakeTask(None, None, True, on_result))
        _wire_agent(monkeypatch, agent)
        result = await runner_module.run_browser_use("goal", "/tmp/dl", "/tmp/res")
        assert result == BrowserUseResult("incomplete", "", ("downloads/txn_2/download",))
        assert agent.agent_run.cancel_loop == 1
        assert len(session.transactions) == 1
        assert session.closed == 1
        with pytest.raises(CheckpointError, match="store is closed"):
            await session.checkpoint_store.list()

    async def test_agent_failure_propagates_and_closes_the_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_thinking_backend(monkeypatch)
        agent = FakeAgent()
        agent.agent_run = FakeAgentRun(FakeTask(None, RuntimeError("agent validation failed"), False, None))
        _wire_agent(monkeypatch, agent)
        session = FakeSession([])
        _wire_session(monkeypatch, session)
        with pytest.raises(RuntimeError, match="agent validation failed"):
            await runner_module.run_browser_use("goal", "/tmp/dl", "/tmp/res")
        assert session.closed == 1
        with pytest.raises(CheckpointError, match="store is closed"):
            await session.checkpoint_store.list()

    async def test_cancelled_run_reraises_and_closes_the_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_thinking_backend(monkeypatch)
        agent = FakeAgent()
        agent.agent_run = FakeAgentRun(FakeTask(None, None, True, None))
        _wire_agent(monkeypatch, agent)
        session = FakeSession([])
        _wire_session(monkeypatch, session)
        run_task = asyncio.create_task(runner_module.run_browser_use("goal", "/tmp/dl", "/tmp/res"))
        await asyncio.sleep(0.05)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        assert session.closed == 1
        assert agent.agent_run.task.cancelled == 1
        with pytest.raises(CheckpointError, match="store is closed"):
            await session.checkpoint_store.list()


class TestPruneOlderScreenshots:
    """The wired filter keeps only the newest action-tool screenshots in the model input."""

    async def test_older_action_screenshots_are_marked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        filter_fn = await _wired_filter(monkeypatch)
        items = [
            {"role": "user", "content": "start"},
            _function_call_item("c1", "click"),
            _function_output_item("c1", [_image_part(), _text_part("caption 1")]),
            _function_call_item("c2", "open_url"),
            _function_output_item(
                "c2", [_image_part("data:image/png;base64,BBBB"), _text_part("caption 2")]
            ),
            _function_call_item("c3", "click"),
            _function_output_item("c3", [_image_part("data:image/png;base64,CCCC")]),
        ]
        result = filter_fn(_call_data(items))

        assert result.instructions == "the role prompt"
        assert len(result.input) == 7
        assert result.input[2]["output"] == [
            _text_part("[earlier screenshot omitted]"),
            _text_part("caption 1"),
        ]
        assert result.input[4]["output"] == [
            _text_part("[earlier screenshot omitted]"),
            _text_part("caption 2"),
        ]
        assert result.input[6]["output"] == [_image_part("data:image/png;base64,CCCC")]
        assert result.input[0] is items[0]
        assert result.input[1] is items[1]
        assert result.input[3] is items[3]
        assert result.input[5] is items[5]
        assert items[2]["output"] == [_image_part(), _text_part("caption 1")]
        assert items[4]["output"] == [
            _image_part("data:image/png;base64,BBBB"),
            _text_part("caption 2"),
        ]
        assert items[6]["output"] == [_image_part("data:image/png;base64,CCCC")]

    async def test_plain_string_output_does_not_become_latest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        filter_fn = await _wired_filter(monkeypatch)
        items = [
            _function_call_item("c1", "click"),
            _function_output_item("c1", [_image_part()]),
            _function_call_item("s1", "wait"),
            _function_output_item("s1", "plain text, no screenshot"),
            _function_call_item("c2", "click"),
            _function_output_item("c2", [_image_part("data:image/png;base64,BBBB")]),
        ]
        result = filter_fn(_call_data(items))

        assert result.input[1]["output"] == [_text_part("[earlier screenshot omitted]")]
        assert result.input[3] is items[3]
        assert result.input[5]["output"] == [_image_part("data:image/png;base64,BBBB")]

    async def test_failed_step_screenshot_becomes_the_latest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        filter_fn = await _wired_filter(monkeypatch)
        failed_output = [
            _text_part('1. tab1 — "Shop" — https://shop.example — image 1\nstatus: failed, action failed: boom'),
            _image_part("data:image/png;base64,BBBB"),
        ]
        items = [
            _function_call_item("c1", "click"),
            _function_output_item("c1", [_text_part("caption 1"), _image_part()]),
            _function_call_item("c2", "click"),
            _function_output_item("c2", failed_output),
        ]
        result = filter_fn(_call_data(items))

        assert result.input[1]["output"] == [
            _text_part("caption 1"),
            _text_part("[earlier screenshot omitted]"),
        ]
        assert result.input[3]["output"] == failed_output

    async def test_no_action_screenshots_returns_the_input_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        filter_fn = await _wired_filter(monkeypatch)
        items = [
            {"role": "user", "content": "start"},
            _function_call_item("c1", "click"),
            _function_output_item("c1", "no screenshot"),
            _function_call_item("c2", "open_url"),
            _function_output_item("c2", "also no screenshot"),
        ]
        result = filter_fn(_call_data(items))

        assert result.instructions == "the role prompt"
        assert all(kept is original for kept, original in zip(result.input, items))

    async def test_single_marker_for_an_output_with_multiple_images(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        filter_fn = await _wired_filter(monkeypatch)
        items = [
            _function_call_item("c1", "click"),
            _function_output_item(
                "c1",
                [
                    _image_part(),
                    _text_part("caption"),
                    _image_part("data:image/png;base64,BBBB"),
                ],
            ),
            _function_call_item("c2", "click"),
            _function_output_item("c2", [_image_part("data:image/png;base64,CCCC")]),
        ]
        result = filter_fn(_call_data(items))

        assert result.input[1]["output"] == [
            _text_part("[earlier screenshot omitted]"),
            _text_part("caption"),
        ]
        assert result.input[3]["output"] == [_image_part("data:image/png;base64,CCCC")]

    @pytest.mark.parametrize("missing", ["name", "call_id"])
    async def test_function_call_missing_a_key_is_treated_as_non_action(
        self, missing: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        filter_fn = await _wired_filter(monkeypatch)
        call_item = _function_call_item("c2", "click")
        del call_item[missing]
        items = [
            _function_call_item("c1", "click"),
            _function_output_item("c1", [_image_part()]),
            call_item,
            _function_output_item("c2", [_image_part("data:image/png;base64,BBBB")]),
        ]
        result = filter_fn(_call_data(items))

        assert result.input[1]["output"] == [_image_part()]
        assert result.input[3]["output"] == [_image_part("data:image/png;base64,BBBB")]

    async def test_non_action_image_output_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        filter_fn = await _wired_filter(monkeypatch)
        items = [
            _function_call_item("c1", "click"),
            _function_output_item("c1", [_image_part()]),
            _function_call_item("t1", "some_other_tool"),
            _function_output_item("t1", [_image_part("data:image/png;base64,BBBB")]),
        ]
        result = filter_fn(_call_data(items))

        assert result.input[1]["output"] == [_image_part()]
        assert result.input[3]["output"] == [_image_part("data:image/png;base64,BBBB")]

    async def test_previous_dom_dump_keeps_its_call_and_checkpoint_stub(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        filter_fn = await _wired_filter(monkeypatch)
        items = [
            _function_call_item("d1", "dump_page"),
            _function_output_item("d1", "1. tab1 — First\nstatus: success\nCheckpoint: first\nObservation time: original\nDOM dump target: tab1\n\nEarlier content"),
            _function_call_item("d2", "dump_page"),
            _function_output_item("d2", "1. tab1 — Second\nstatus: success\nDOM dump target: tab1\n\n# New"),
        ]

        result = filter_fn(_call_data(items))

        assert [item["call_id"] for item in result.input] == ["d1", "d1", "d2", "d2"]
        assert "Earlier content" not in result.input[1]["output"]
        assert "Checkpoint: first" in result.input[1]["output"]
        assert "target_id tab1" in result.input[1]["output"]
        assert "status:" not in result.input[1]["output"]
        assert "# New" in result.input[3]["output"]

    async def test_closed_tab_dom_dump_keeps_metadata_without_current_envelope(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        filter_fn = await _wired_filter(monkeypatch)
        items = [
            _function_call_item("d1", "dump_page"),
            _function_output_item("d1", "1. tab1 — First\n2. tab2 — Second\nstatus: success\nDOM dump target: tab1\n\n# Gone"),
            _function_call_item("d2", "dump_page"),
            _function_output_item("d2", "1. tab1 — First\n2. tab2 — Second\nstatus: success\nDOM dump target: tab2\n\n# Kept"),
            _function_call_item("c1", "close_page"),
            _function_output_item("c1", "1. tab2 — Second\nstatus: success"),
        ]

        result = filter_fn(_call_data(items))

        assert [item["call_id"] for item in result.input] == ["d1", "d1", "d2", "d2", "c1", "c1"]
        assert "# Gone" not in result.input[1]["output"]
        assert "Archived DOM target: tab1" in result.input[1]["output"]
        assert "status:" not in result.input[1]["output"]
        assert "# Kept" in result.input[3]["output"]

    async def test_status_text_in_tab_title_does_not_hide_open_tabs(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        filter_fn = await _wired_filter(monkeypatch)
        items = [
            _function_call_item("d1", "dump_page"),
            _function_output_item(
                "d1", "1. tab1 — Status: open\nstatus: success\nDOM dump target: tab1\n\n# Kept",
            ),
            _function_call_item("c1", "click"),
            _function_output_item("c1", "1. tab1 — status: open\nstatus: success"),
        ]

        result = filter_fn(_call_data(items))

        assert [item["call_id"] for item in result.input] == ["d1", "d1", "c1", "c1"]

    @pytest.mark.parametrize("remaining_tab", [True, False])
    async def test_latest_dump_keeps_current_observation_when_target_closes(
        self, monkeypatch: pytest.MonkeyPatch, remaining_tab: bool,
    ) -> None:
        filter_fn = await _wired_filter(monkeypatch)
        captions = "1. tab2 — Remaining\n" if remaining_tab else ""
        current_text = (
            f"Time: original\n{captions}status: success\nCheckpoint: newest\n"
            "Observation time: captured\nDOM capture interval: dom start to dom end\n"
            "DOM dump target: tab1\n\nClosed target DOM"
        )
        current_parts = [_text_part(current_text)]
        if remaining_tab:
            current_parts.append(_image_part("current-tab2"))
        historical_parts = [
            _text_part("Historical checkpoint: old\n1. tab1 — Historical\nstatus: success\nDOM dump target: tab1\nOld evidence"),
            _image_part("historical-tab1"),
        ]
        items = [
            _function_call_item("c1", "open_url"),
            _function_output_item("c1", [_text_part("1. tab1 — Old\nstatus: success"), _image_part("obsolete")]),
            _function_call_item("d1", "dump_page"),
            _function_output_item("d1", current_parts),
            _function_call_item("r1", "checkpoint"),
            _function_output_item("r1", historical_parts),
        ]
        result = filter_fn(_call_data(items))
        assert "Closed target DOM" not in result.input[3]["output"][0]["text"]
        assert "status: success" in result.input[3]["output"][0]["text"]
        assert "Checkpoint: newest" in result.input[3]["output"][0]["text"]
        assert "Observation time: captured" in result.input[3]["output"][0]["text"]
        assert "DOM capture interval: dom start to dom end" in result.input[3]["output"][0]["text"]
        assert result.input[1]["output"][-1] == _text_part("[earlier screenshot omitted]")
        if remaining_tab:
            assert "1. tab2 — Remaining\nstatus:" in result.input[3]["output"][0]["text"]
            assert result.input[3]["output"][1] == _image_part("current-tab2")
        else:
            assert "1. tab1" not in result.input[3]["output"][0]["text"]
            assert len(result.input[3]["output"]) == 1
        assert result.input[5] == items[5]
        assert filter_fn(_call_data(result.input)).input == result.input

    async def test_previously_current_pruned_dump_becomes_metadata_after_new_observation(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        filter_fn = await _wired_filter(monkeypatch)
        items = [
            _function_call_item("d1", "dump_page"),
            _function_output_item("d1", [
                _text_part("1. tab2 — Remaining\nstatus: success\nCheckpoint: first\nDOM dump target: tab1\n\nClosed DOM"),
                _image_part("was-current"),
            ]),
        ]
        first = filter_fn(_call_data(items)).input
        later = first + [
            _function_call_item("c2", "close_page"),
            _function_output_item("c2", "status: success\nCheckpoint: second"),
        ]
        result = filter_fn(_call_data(later))
        assert isinstance(result.input[1]["output"], str)
        assert "Checkpoint: first" in result.input[1]["output"]
        assert "status:" not in result.input[1]["output"]
        assert "Remaining" not in result.input[1]["output"]
        assert result.input[3] == later[3]
        assert filter_fn(_call_data(result.input)).input == result.input

    async def test_multiple_historical_views_survive_actions_dumps_closure_and_filter_retries(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        filter_fn = await _wired_filter(monkeypatch)
        feedback = [
            _text_part("Historical checkpoint: feedback\nTab: tab1\nExact feedback\nDOM dump target: tab1"),
            _image_part("saved-feedback"),
        ]
        transcript = [
            _text_part("Historical checkpoint: transcript\nTab: tab1\nExact transcript\n1. injected — old\nstatus: success"),
            _image_part("saved-transcript"),
        ]
        items = [
            _function_call_item("r1", "checkpoint"), _function_output_item("r1", feedback),
            _function_call_item("r2", "checkpoint"), _function_output_item("r2", transcript),
            _function_call_item("d1", "dump_page"),
            _function_output_item("d1", "1. tab1 — New\nstatus: success\nCheckpoint: later\nDOM dump target: tab1\n\nNew dump"),
            _function_call_item("c1", "close_page"), _function_output_item("c1", "status: success"),
        ]
        result = filter_fn(_call_data(items))
        assert result.input[:4] == items[:4]
        assert result.input[1] == items[1]
        assert result.input[3] == items[3]
        assert "New dump" not in result.input[5]["output"]
        assert "Checkpoint: later" in result.input[5]["output"]
        assert filter_fn(_call_data(result.input)).input == result.input


@skip_if_no_chrome
@pytest.mark.live
class TestLiveRunner:
    @pytest.mark.timeout(_LIVE_TIMEOUT_S)
    async def test_upload_resource_from_page(self, tmp_path: Path, test_server_url: str) -> None:
        """The model uploads a resource file that the test server receives."""
        downloads_dir = tmp_path / "downloads"
        resources_dir = tmp_path / "resources"
        downloads_dir.mkdir()
        resource = resources_dir / "upload" / "model-upload.txt"
        resource.parent.mkdir(parents=True)
        content = b"resource uploaded by the browser use agent\n"
        resource.write_bytes(content)
        token = uuid4().hex
        prompt = (
            f"Open {test_server_url}/page/upload/{token} in the browser. "
            "Click the Upload a file button and upload the resource at upload/model-upload.txt. "
            "Wait for the page to show Uploaded model-upload.txt."
        )

        result = await runner_module.run_browser_use(prompt, str(downloads_dir), str(resources_dir))

        assert result.status == "completed", result.summary
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{test_server_url}/upload/result/{token}")
        assert response.status_code == 200
        assert response.content == content

    @pytest.mark.timeout(_LIVE_TIMEOUT_S)
    async def test_open_page_and_report_heading(self, tmp_path: Path, test_server_url: str) -> None:
        downloads_dir = tmp_path / "downloads"
        resources_dir = tmp_path / "resources"
        downloads_dir.mkdir()
        resources_dir.mkdir()
        prompt = (
            f"Open the page at {test_server_url}/ in the browser. "
            "Report the exact text of the page's main heading, the largest heading shown."
        )
        result = await runner_module.run_browser_use(prompt, str(downloads_dir), str(resources_dir))
        assert result.status == "completed"
        assert "example domain" in result.summary.lower()
        assert result.files == ()

    @pytest.mark.timeout(_LIVE_TIMEOUT_S)
    async def test_dump_page_and_report_offscreen_text(
        self, tmp_path: Path, test_server_url: str
    ) -> None:
        """The model reports text from the end of a long page."""
        downloads_dir = tmp_path / "downloads"
        resources_dir = tmp_path / "resources"
        downloads_dir.mkdir()
        resources_dir.mkdir()
        prompt = (
            f"Open {test_server_url}/page/long in the browser. "
            "Call dump_page on that tab to read its filtered DOM. "
            "Use the DOM dump to report the exact paragraph text under Section 20. "
            "Do not scroll the page."
        )

        result = await runner_module.run_browser_use(prompt, str(downloads_dir), str(resources_dir))

        assert result.status == "completed", result.summary
        assert "Content for section 20" in result.summary
        assert result.files == ()
        assert list(resources_dir.iterdir()) == []

    @pytest.mark.timeout(_LIVE_TIMEOUT_S)
    async def test_download_file_from_page(self, tmp_path: Path, test_server_url: str) -> None:
        downloads_dir = tmp_path / "downloads"
        resources_dir = tmp_path / "resources"
        downloads_dir.mkdir()
        resources_dir.mkdir()
        prompt = (
            f"Open the page at {test_server_url}/ in the browser. "
            "Click the link labeled Download so a file is downloaded."
        )
        result = await runner_module.run_browser_use(prompt, str(downloads_dir), str(resources_dir))
        assert result.status in ("completed", "incomplete")
        assert result.files
        expected = b"test download content for browser tests\n"
        assert all(Path(p).is_file() and Path(p).read_bytes() == expected for p in result.files)
