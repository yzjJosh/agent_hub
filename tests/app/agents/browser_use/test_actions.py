"""Tests for the browser use action tools."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.agents.browser_use import actions as actions_module
from app.agents.browser_use.models import TransactionResult, TransactionStatus
from app.agents.browser_use.session import Action
from app.lib.tool.base import Tool, ToolMessageError, ToolResult
from tests.app.agents.browser_use.conftest import (
    STANDARD_ACTION_TOOL_NAMES,
    FakeActionManager,
    FakeSession,
    build_default_snapshot,
    build_success_transaction,
    find_tool,
)

_HALF = 0.5
_TENTH = 0.1
_THREE_TENTHS = 0.3
_TENTH_POSITION = 100
_THREE_TENTHS_POSITION = 300
_HALF_POSITION = 500
_NETWORK_IDLE_TIMEOUT_MILLISECONDS = 2000
_FIXED_YEAR = 2026

_OPEN_TAB_TOOL_NAMES = tuple(name for name in STANDARD_ACTION_TOOL_NAMES if name != "open_url")
_SNAPSHOT_TOOL_NAMES = (
    "click",
    "move_mouse",
    "type_text",
    "replace_input_text",
    "scroll",
    "press_key",
    "solve_captcha",
    "download_click",
)

_TIME_LINE = re.compile(r"^Time: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ([A-Za-z]+|[+-]\d+)$")


def _extract_after_time_line(text: str) -> str:
    """Assert the first line is the time line and return the remaining text."""
    first_line, _, rest = text.partition("\n")
    assert _TIME_LINE.match(first_line)
    return rest


def _build_failed_transaction(message: str = "action failed: boom") -> TransactionResult:
    return TransactionResult(TransactionStatus.FAILED, message, build_default_snapshot(), ())


def _build_stale_transaction(
    message: str = "screen changed since you last saw it, action not run",
) -> TransactionResult:
    return TransactionResult(TransactionStatus.STALE, message, build_default_snapshot(), ())


@pytest.fixture
def _restore_zone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore the local timezone after a test that changes TZ."""
    yield
    monkeypatch.undo()
    time.tzset()


class FrozenDatetime(datetime):
    """A datetime whose clock stays fixed on one instant for time format checks."""

    instant: datetime = datetime(_FIXED_YEAR, 1, 15, 12, 0, 0)

    @classmethod
    def now(cls, tz: timezone | None = None) -> datetime:
        return cls.instant.replace(tzinfo=tz)


_PARAMS: dict[str, dict] = {
    "open_url": {"target_id": "tab1", "url": "https://a.example"},
    "click": {"target_id": "tab1", "x": 200, "y": 400},
    "move_mouse": {"target_id": "tab1", "x": 300, "y": 600},
    "type_text": {"target_id": "tab1", "x": _TENTH_POSITION, "y": _THREE_TENTHS_POSITION, "text": "hello"},
    "replace_input_text": {
        "target_id": "tab1",
        "x": _TENTH_POSITION,
        "y": _THREE_TENTHS_POSITION,
        "old_text": "helo",
        "new_text": "hello",
    },
    "scroll": {"target_id": "tab1", "dx": -_HALF, "dy": 1.0},
    "press_key": {"target_id": "tab1", "key": "Enter"},
    "go_back": {"target_id": "tab1"},
    "go_forward": {"target_id": "tab1"},
    "close_page": {"target_id": "tab1"},
    "solve_captcha": {"target_id": "tab1"},
    "wait": {"target_id": "tab1", "seconds": 0.0},
    "download_click": {"target_id": "tab1", "x": 700, "y": 900},
    "refresh": {"target_id": "tab1"},
    "dump_page": {"target_id": "tab1"},
}


def _build_snapshot_flow(
    tab: str, method: str, call: tuple, *, waits_for_load: bool = True
) -> dict[str, list[tuple]]:
    """Build the expected call record for a snapshot-based action on a tab."""
    record = {
        "capture_snapshot": [(tab, _NETWORK_IDLE_TIMEOUT_MILLISECONDS, True)],
        method: [call],
    }
    if waits_for_load:
        record["wait_for_load"] = [(tab, "networkidle", _NETWORK_IDLE_TIMEOUT_MILLISECONDS, True)]
    return record


_EXPECTED_RECORD: dict[str, dict[str, list[tuple]]] = {
    "open_url": {
        "navigate": [("tab1", "https://a.example", None, None, True)],
    },
    "click": _build_snapshot_flow("tab1", "click_at_relative", ("tab1", 0.2, 0.4, True), waits_for_load=False),
    "move_mouse": _build_snapshot_flow("tab1", "move_mouse_to_relative", ("tab1", 0.3, 0.6, True)),
    "type_text": _build_snapshot_flow("tab1", "type_text_at_relative", ("tab1", _TENTH, _THREE_TENTHS, "hello", False, True)),
    "replace_input_text": _build_snapshot_flow(
        "tab1", "replace_input_text_at_relative", ("tab1", _TENTH, _THREE_TENTHS, "helo", "hello", True)
    ),
    "scroll": _build_snapshot_flow("tab1", "scroll_by_relative", ("tab1", -_HALF, 1.0, True)),
    "press_key": _build_snapshot_flow("tab1", "press_key", ("tab1", "Enter", True)),
    "go_back": {"go_back": [("tab1", None, None, True)]},
    "go_forward": {"go_forward": [("tab1", None, None, True)]},
    "close_page": {"close_page": [("tab1", True)]},
    "solve_captcha": _build_snapshot_flow("tab1", "solve_captcha", ("tab1", 5, True)),
    "wait": {},
    "download_click": _build_snapshot_flow("tab1", "click_at_relative", ("tab1", 0.7, 0.9, True), waits_for_load=False),
    "refresh": {
        "get_url": [("tab1", False)],
        "navigate": [("tab1", "https://example.com", None, None, True)],
    },
    "dump_page": {
        "capture_snapshot": [("tab1", _NETWORK_IDLE_TIMEOUT_MILLISECONDS, True)],
        "filtered_tree": [("tab1", True)],
    },
}


def _replace_record_tab(record: dict[str, list[tuple]], tab: str) -> dict[str, list[tuple]]:
    """Copy an expected call record with the tab id replaced in every call."""
    return {method: [((tab,) + call[1:]) for call in calls] for method, calls in record.items()}


_SOLE_RECORD = {
    name: _replace_record_tab(record, "only") for name, record in _EXPECTED_RECORD.items() if name != "open_url"
}

_NEW_PAGE_RECORD: dict[str, list[tuple]] = {
    "open_page": [(True,)],
    "navigate": [("new1", "https://a.example", None, None, True)],
}

_EXPECTED_ORDER: dict[str, list[str]] = {
    "open_url": ["navigate"],
    "click": ["capture_snapshot", "click_at_relative"],
    "move_mouse": ["capture_snapshot", "move_mouse_to_relative", "wait_for_load"],
    "type_text": ["capture_snapshot", "type_text_at_relative", "wait_for_load"],
    "replace_input_text": ["capture_snapshot", "replace_input_text_at_relative", "wait_for_load"],
    "scroll": ["capture_snapshot", "scroll_by_relative", "wait_for_load"],
    "press_key": ["capture_snapshot", "press_key", "wait_for_load"],
    "go_back": ["go_back"],
    "go_forward": ["go_forward"],
    "close_page": ["close_page"],
    "solve_captcha": ["capture_snapshot", "solve_captcha", "wait_for_load"],
    "wait": [],
    "download_click": ["capture_snapshot", "click_at_relative"],
    "refresh": ["get_url", "navigate"],
    "dump_page": ["capture_snapshot", "filtered_tree"],
}


_NON_POSITION_PARAMS_BY_TOOL: dict[str, dict] = {
    "click": {},
    "upload_resources": {"resource_paths": ["upload/photo.jpg"]},
    "move_mouse": {},
    "type_text": {"text": "hello"},
    "replace_input_text": {"old_text": "helo", "new_text": "hello"},
    "scroll": {"dx": -_HALF, "dy": 1.0},
    "download_click": {},
}

_BROWSER_METHOD_BY_POSITION_TOOL: dict[str, str] = {
    "click": "click_at_relative",
    "upload_resources": "upload_with_chooser_relative",
    "move_mouse": "move_mouse_to_relative",
    "type_text": "type_text_at_relative",
    "replace_input_text": "replace_input_text_at_relative",
    "scroll": "move_mouse_to_relative",
    "download_click": "click_at_relative",
}


def _build_tools(session: FakeSession, resources_directory: str = "/tmp/res") -> tuple[Tool, ...]:
    """Build action tools bound to a fake session."""
    return actions_module.build_action_tools(session, resources_directory)


async def _run_tool(tool: Tool, params: dict) -> ToolResult:
    """Run one tool with the given parameters and return its result."""
    task = await tool.run(params)
    return await task.result()


async def _run_plain_action(session: FakeSession, tool: Tool, params: dict) -> Action:
    """Run a tool, assert a plain transaction on tab1, and return its action closure."""
    await _run_tool(tool, params)
    action, target_id, monitor_download, new_tab = session.transactions[0]
    assert target_id == "tab1"
    assert monitor_download is False
    assert new_tab is False
    return action


class TestActionTools:
    async def test_dump_page_reads_filtered_tree_as_markdown(self, tmp_path: Path) -> None:
        manager = FakeActionManager(tab_ids=("tab1",))
        tool = find_tool("dump_page", _build_tools(FakeSession([], manager=manager), str(tmp_path)))

        result = await _run_tool(tool, {"target_id": "tab1"})

        assert result.render_type == "markdown"
        assert "DOM dump target: tab1" in result.text
        assert "\n\n## Page title" in result.text
        assert "status: success" in result.text
        assert result.files is None
        assert not (tmp_path / "browser_use" / "dumps").exists()
        assert manager.recorded == _EXPECTED_RECORD["dump_page"]

    async def test_dump_page_failure_has_no_dom_content(self) -> None:
        manager = FakeActionManager(tab_ids=("tab1",), no_snapshot=True)
        tool = find_tool("dump_page", _build_tools(FakeSession([], manager=manager)))

        result = await _run_tool(tool, {"target_id": "tab1"})

        assert "status: failed" in result.text
        assert "DOM dump target:" not in result.text

    async def test_dump_page_fails_when_filtered_tree_is_missing(self) -> None:
        manager = FakeActionManager(tab_ids=("tab1",), no_tree=True)
        tool = find_tool("dump_page", _build_tools(FakeSession([], manager=manager)))

        result = await _run_tool(tool, {"target_id": "tab1"})

        assert "status: failed, action failed: could not read the page DOM" in result.text
        assert "DOM dump target:" not in result.text
        assert manager.recorded == _EXPECTED_RECORD["dump_page"]

    async def test_upload_resources_uses_resource_file_and_snapshot(self, tmp_path: Path) -> None:
        """Upload an existing resource file through the page snapshot."""
        resource = tmp_path / "upload" / "photo.jpg"
        resource.parent.mkdir()
        resource.write_bytes(b"photo")
        second = tmp_path / "upload" / "notes.txt"
        second.write_text("notes")
        manager = FakeActionManager(tab_ids=("tab1",))
        session = FakeSession([], manager=manager)
        tool = find_tool("upload_resources", _build_tools(session, str(tmp_path)))

        result = await _run_tool(
            tool,
            {
                "x": _TENTH_POSITION,
                "y": _THREE_TENTHS_POSITION,
                "resource_paths": ["upload/photo.jpg", "upload/notes.txt"],
                "target_id": "tab1",
            },
        )

        assert "status: success" in result.text
        assert manager.recorded == _build_snapshot_flow(
            "tab1", "upload_with_chooser_relative", ("tab1", _TENTH, _THREE_TENTHS, [str(resource), str(second)], True)
        )
        assert manager.order == ["capture_snapshot", "upload_with_chooser_relative", "wait_for_load"]

    @pytest.mark.parametrize("paths", [[], ["/tmp/outside"], ["../outside"], ["missing.txt"]])
    async def test_upload_resources_rejects_invalid_paths_before_click(
        self, paths: list[str], tmp_path: Path
    ) -> None:
        """Invalid file paths fail without touching the page."""
        manager = FakeActionManager(tab_ids=("tab1",))
        tool = find_tool("upload_resources", _build_tools(FakeSession([], manager=manager), str(tmp_path)))

        result = await _run_tool(
            tool,
            {"x": _TENTH_POSITION, "y": _THREE_TENTHS_POSITION, "resource_paths": paths, "target_id": "tab1"},
        )

        assert "status: failed" in result.text
        assert manager.recorded == {}

    @pytest.mark.parametrize("name", STANDARD_ACTION_TOOL_NAMES)
    async def test_tool_is_tracked_and_renders_text(self, name: str) -> None:
        session = FakeSession([])
        tool = find_tool(name, _build_tools(session))
        assert tool.is_tracked is True
        assert tool.result_render_type == "text"
        assert name in actions_module.ACTION_TOOL_NAMES

    @pytest.mark.parametrize("name", STANDARD_ACTION_TOOL_NAMES)
    async def test_tool_result_keeps_the_screenshots(self, name: str) -> None:
        session = FakeSession([build_success_transaction()])
        result = await _run_tool(find_tool(name, _build_tools(session)), _PARAMS[name])
        assert result.files == ("/tmp/dl/txn_1/01.jpg",)

    @pytest.mark.parametrize("name", STANDARD_ACTION_TOOL_NAMES)
    async def test_tool_calls_transact_with_the_right_action(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = FakeSession([build_success_transaction()])
        await _run_tool(find_tool(name, _build_tools(session)), _PARAMS[name])
        assert len(session.transactions) == 1
        action, target_id, monitor_download, new_tab = session.transactions[0]
        assert target_id == "tab1"
        assert monitor_download is (name == "download_click")
        assert new_tab is (name == "open_url")
        manager = FakeActionManager()
        if name in ("click", "download_click"):
            monkeypatch.setattr(actions_module.asyncio, "sleep", AsyncMock())
        assert await action(manager, "tab1") is True
        assert manager.recorded == _EXPECTED_RECORD[name]
        assert manager.order == _EXPECTED_ORDER[name]

    @pytest.mark.parametrize("name", ("click", "download_click"))
    async def test_click_delay_allows_popup_before_result_snapshot(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = FakeActionManager(tab_ids=("tab1",))
        session = FakeSession([], manager=manager)
        delays: list[float] = []

        async def discover_popup(seconds: float) -> None:
            delays.append(seconds)
            manager.add_tab("popup")

        monkeypatch.setattr(actions_module.asyncio, "sleep", discover_popup)
        result = await _run_tool(find_tool(name, _build_tools(session)), _PARAMS[name])
        assert delays == [2.0]
        assert "2. popup" in result.text
        assert "wait_for_load" not in manager.recorded

    async def test_type_text_with_submit_passes_the_flag_through(self) -> None:
        session = FakeSession([build_success_transaction()])
        tool = find_tool("type_text", _build_tools(session))
        submit = next(param for param in tool.params if param.name == "submit")
        assert submit.default is False
        params = dict(_PARAMS["type_text"], submit=True)
        action = await _run_plain_action(session, tool, params)
        manager = FakeActionManager()
        assert await action(manager, "tab1") is True
        assert manager.recorded["type_text_at_relative"] == [("tab1", _TENTH, _THREE_TENTHS, "hello", True, True)]
        assert manager.order == _EXPECTED_ORDER["type_text"]

    async def test_replace_input_text_with_empty_new_text_passes_it_through(self) -> None:
        session = FakeSession([build_success_transaction()])
        tool = find_tool("replace_input_text", _build_tools(session))
        params = dict(_PARAMS["replace_input_text"], old_text="remove me", new_text="")
        action = await _run_plain_action(session, tool, params)
        manager = FakeActionManager()
        assert await action(manager, "tab1") is True
        assert manager.recorded["replace_input_text_at_relative"] == [("tab1", _TENTH, _THREE_TENTHS, "remove me", "", True)]
        assert manager.order == _EXPECTED_ORDER["replace_input_text"]

    async def test_scroll_with_an_anchor_moves_the_mouse_before_scrolling(self) -> None:
        session = FakeSession([build_success_transaction()])
        tool = find_tool("scroll", _build_tools(session))
        anchor_x = next(param for param in tool.params if param.name == "x")
        anchor_y = next(param for param in tool.params if param.name == "y")
        assert anchor_x.default is None
        assert anchor_y.default is None
        params = dict(_PARAMS["scroll"], x=_THREE_TENTHS_POSITION, y=_HALF_POSITION)
        action = await _run_plain_action(session, tool, params)
        manager = FakeActionManager()
        assert await action(manager, "tab1") is True
        assert manager.recorded == {
            "capture_snapshot": [("tab1", _NETWORK_IDLE_TIMEOUT_MILLISECONDS, True)],
            "move_mouse_to_relative": [("tab1", _THREE_TENTHS, _HALF, True)],
            "scroll_by_relative": [("tab1", -_HALF, 1.0, True)],
            "wait_for_load": [("tab1", "networkidle", _NETWORK_IDLE_TIMEOUT_MILLISECONDS, True)],
        }
        assert manager.order == ["capture_snapshot", "move_mouse_to_relative", "scroll_by_relative", "wait_for_load"]

    @pytest.mark.parametrize(
        "given",
        [pytest.param("x", id="lone_x"), pytest.param("y", id="lone_y")],
    )
    async def test_scroll_with_a_lone_anchor_coordinate_reports_failed(self, given: str) -> None:
        manager = FakeActionManager(tab_ids=("only",))
        session = FakeSession([], manager=manager)
        params = {key: value for key, value in _PARAMS["scroll"].items() if key != "target_id"}
        params[given] = _THREE_TENTHS_POSITION
        result = await _run_tool(find_tool("scroll", _build_tools(session)), params)
        assert "status: failed, action failed: x and y must be given together" in result.text
        assert manager.recorded == {}

    async def test_scroll_with_an_unreachable_anchor_reports_failed(self) -> None:
        manager = FakeActionManager(tab_ids=("only",), move_fails=True)
        session = FakeSession([], manager=manager)
        params = {key: value for key, value in _PARAMS["scroll"].items() if key != "target_id"}
        params.update(x=_THREE_TENTHS_POSITION, y=_HALF_POSITION)
        result = await _run_tool(find_tool("scroll", _build_tools(session)), params)
        assert "status: failed, action failed: could not move the mouse to the scroll anchor" in result.text
        assert manager.recorded == {
            "capture_snapshot": [("only", _NETWORK_IDLE_TIMEOUT_MILLISECONDS, True)],
            "move_mouse_to_relative": [("only", _THREE_TENTHS, _HALF, True)],
        }

    async def test_scroll_anchor_docs_name_the_contract(self) -> None:
        session = FakeSession([])
        tool = find_tool("scroll", _build_tools(session))
        assert "nested scrollable region" in tool.description
        anchor_x = next(param for param in tool.params if param.name == "x")
        anchor_y = next(param for param in tool.params if param.name == "y")
        assert anchor_x.default is None
        assert anchor_y.default is None
        assert "Give it with y" in anchor_x.description
        assert "Give it with x" in anchor_y.description

    @pytest.mark.parametrize("name", _NON_POSITION_PARAMS_BY_TOOL)
    @pytest.mark.parametrize(
        "position_x,position_y,fraction_x,fraction_y",
        [
            pytest.param(785, 215, 0.785, 0.215, id="whole_numbers"),
            pytest.param(785.5, 40.25, 0.7855, 0.04025, id="decimals"),
            pytest.param(0, 0, 0.0, 0.0, id="top_left"),
            pytest.param(1000, 1000, 1.0, 1.0, id="bottom_right"),
        ],
    )
    async def test_position_reaches_the_browser_as_a_fraction_of_the_screen(
        self,
        name: str,
        position_x: float,
        position_y: float,
        fraction_x: float,
        fraction_y: float,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 0 to 1000 position reaches the browser as the same point on the 0 to 1 scale."""
        upload = tmp_path / "upload" / "photo.jpg"
        upload.parent.mkdir()
        upload.write_bytes(b"photo")
        monkeypatch.setattr(actions_module.asyncio, "sleep", AsyncMock())
        manager = FakeActionManager(tab_ids=("tab1",))
        tool = find_tool(name, _build_tools(FakeSession([], manager=manager), str(tmp_path)))
        params = dict(_NON_POSITION_PARAMS_BY_TOOL[name], target_id="tab1", x=position_x, y=position_y)

        result = await _run_tool(tool, params)

        assert "status: success" in result.text
        browser_call = manager.recorded[_BROWSER_METHOD_BY_POSITION_TOOL[name]][0]
        assert browser_call[:3] == ("tab1", fraction_x, fraction_y)

    @pytest.mark.parametrize("name", _NON_POSITION_PARAMS_BY_TOOL)
    @pytest.mark.parametrize("axis", ["x", "y"])
    @pytest.mark.parametrize(
        "position,message",
        [
            pytest.param(-1, "Input should be greater than or equal to 0.", id="below_zero"),
            pytest.param(1000.5, "Input should be less than or equal to 1000.", id="above_1000"),
        ],
    )
    async def test_position_outside_0_to_1000_is_rejected(
        self, name: str, axis: str, position: float, message: str
    ) -> None:
        """A position outside 0 to 1000 fails before any browser action runs."""
        session = FakeSession([build_success_transaction()])
        tool = find_tool(name, _build_tools(session))
        params = dict(_NON_POSITION_PARAMS_BY_TOOL[name], target_id="tab1", x=_HALF_POSITION, y=_HALF_POSITION)
        params[axis] = position

        with pytest.raises(ToolMessageError, match=re.escape(f"Error: {axis}: {message}")):
            await _run_tool(tool, params)

        assert session.transactions == []

    @pytest.mark.parametrize("target", [pytest.param(None, id="no_target"), pytest.param("", id="blank_target")])
    async def test_open_url_without_a_tab_opens_a_new_page(self, target: str | None) -> None:
        session = FakeSession([build_success_transaction()])
        params: dict = {"url": "https://a.example"}
        if target is not None:
            params["target_id"] = target
        await _run_tool(find_tool("open_url", _build_tools(session)), params)
        action, target_id, monitor_download, new_tab = session.transactions[0]
        assert target_id in (None, "")
        assert monitor_download is False
        assert new_tab is True
        manager = FakeActionManager()
        assert await action(manager, None) is True
        assert manager.recorded == _NEW_PAGE_RECORD

    async def test_open_url_with_a_blank_target_and_one_tab_still_opens_a_new_page(self) -> None:
        session = FakeSession([build_success_transaction()])
        await _run_tool(find_tool("open_url", _build_tools(session)), {"url": "https://a.example"})
        action, target_id, monitor_download, new_tab = session.transactions[0]
        assert target_id is None
        assert monitor_download is False
        assert new_tab is True
        manager = FakeActionManager(tab_ids=("existing",))
        assert await action(manager, None) is True
        assert manager.recorded == _NEW_PAGE_RECORD

    async def test_open_url_with_a_blank_target_and_two_tabs_still_opens_a_new_page(self) -> None:
        manager = FakeActionManager(tab_ids=("a", "b"))
        session = FakeSession([], manager=manager)
        await _run_tool(find_tool("open_url", _build_tools(session)), {"url": "https://a.example"})
        _action, target_id, monitor_download, new_tab = session.transactions[0]
        assert target_id is None
        assert monitor_download is False
        assert new_tab is True
        assert manager.recorded == _NEW_PAGE_RECORD

    async def test_result_text_carries_captions_and_status(self) -> None:
        session = FakeSession([build_success_transaction()])
        result = await _run_tool(find_tool("click", _build_tools(session)), _PARAMS["click"])
        assert '1. tab1 — "Shop" — https://shop.example — image 1' in result.text
        assert "2. tab2 — no screenshot" in result.text
        assert "status: success" in result.text

    @pytest.mark.parametrize("name", _OPEN_TAB_TOOL_NAMES)
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
        session = FakeSession([build_success_transaction()])
        params = {key: value for key, value in _PARAMS[name].items() if key != "target_id"}
        if target is not None:
            params["target_id"] = target
        await _run_tool(find_tool(name, _build_tools(session)), params)
        action, target_id, monitor_download, new_tab = session.transactions[0]
        assert target_id == target
        assert monitor_download is (name == "download_click")
        assert new_tab is False
        manager = FakeActionManager(tab_ids=("only",))
        if name in ("click", "download_click"):
            monkeypatch.setattr(actions_module.asyncio, "sleep", AsyncMock())
        assert await action(manager, "only") is True
        assert manager.recorded == _SOLE_RECORD[name]
        assert manager.order == _EXPECTED_ORDER[name]

    @pytest.mark.parametrize("name", _OPEN_TAB_TOOL_NAMES)
    @pytest.mark.parametrize(
        "tabs,match",
        [
            pytest.param((), "no tab is open, open one with open_url first", id="no_tabs"),
            pytest.param(("a", "b"), "2 tabs are open, give the target id of the one to act on", id="two_tabs"),
        ],
    )
    async def test_action_without_a_target_reports_failed_without_a_sole_tab(
        self, name: str, tabs: tuple[str, ...], match: str
    ) -> None:
        manager = FakeActionManager(tab_ids=tabs)
        session = FakeSession([], manager=manager)
        params = {key: value for key, value in _PARAMS[name].items() if key != "target_id"}
        result = await _run_tool(find_tool(name, _build_tools(session)), params)
        assert f"status: failed, action failed: {match}" in result.text

    @pytest.mark.parametrize("name", _SNAPSHOT_TOOL_NAMES)
    async def test_action_reports_failed_when_the_tab_has_no_snapshot(self, name: str) -> None:
        manager = FakeActionManager(tab_ids=("only",), no_snapshot=True)
        session = FakeSession([], manager=manager)
        params = {key: value for key, value in _PARAMS[name].items() if key != "target_id"}
        result = await _run_tool(find_tool(name, _build_tools(session)), params)
        assert "status: failed, action failed: no snapshot available for tab" in result.text

    async def test_failed_transaction_hands_back_the_tool_result_with_the_screenshots(self) -> None:
        session = FakeSession([_build_failed_transaction()])
        result = await _run_tool(find_tool("click", _build_tools(session)), _PARAMS["click"])
        assert '1. tab1 — "Shop" — https://shop.example — image 1' in result.text
        assert "2. tab2 — no screenshot" in result.text
        assert "status: failed, action failed: boom" in result.text
        assert result.files == ("/tmp/dl/txn_1/01.jpg",)

    @pytest.mark.parametrize(
        "file_path,line",
        [
            pytest.param(
                "/tmp/res/browser_use/t1/txn_3/download",
                "Downloaded file: browser_use/t1/txn_3/download",
                id="inside_resources",
            ),
            pytest.param(
                "/tmp/other/txn_3/download",
                "Downloaded file: /tmp/other/txn_3/download",
                id="outside_resources",
            ),
        ],
    )
    async def test_download_click_result_names_the_downloaded_file(self, file_path: str, line: str) -> None:
        session = FakeSession([TransactionResult(TransactionStatus.SUCCESS, "", build_default_snapshot(), (file_path,))])
        tools = _build_tools(session, "/tmp/res")
        result = await _run_tool(find_tool("download_click", tools), _PARAMS["download_click"])
        lines = _extract_after_time_line(result.text).split("\n")
        assert lines[-2:] == ["status: success", line]
        assert result.files == ("/tmp/dl/txn_1/01.jpg",)

    @pytest.mark.parametrize(
        "transaction",
        [
            pytest.param(build_success_transaction(), id="success"),
            pytest.param(_build_failed_transaction(), id="failed"),
        ],
    )
    async def test_download_click_without_a_file_has_no_downloaded_file_line(
        self, transaction: TransactionResult
    ) -> None:
        session = FakeSession([transaction])
        result = await _run_tool(find_tool("download_click", _build_tools(session)), _PARAMS["download_click"])
        assert "Downloaded file:" not in result.text

    async def test_stale_transaction_still_hands_back_the_tool_result(self) -> None:
        session = FakeSession([_build_stale_transaction()])
        result = await _run_tool(find_tool("click", _build_tools(session)), _PARAMS["click"])
        assert "status: stale, screen changed since you last saw it, action not run" in result.text
        assert result.files == ("/tmp/dl/txn_1/01.jpg",)

    @pytest.mark.parametrize("name", _OPEN_TAB_TOOL_NAMES)
    async def test_action_target_id_docs_name_the_contract(self, name: str) -> None:
        session = FakeSession([])
        tool = find_tool(name, _build_tools(session))
        target = next(param for param in tool.params if param.name == "target_id")
        assert target.default is None
        assert "The tab to act on" in target.description
        assert "fails if no such tab exists" in target.description
        assert "runs on the single open tab and fails if zero or more than one tab is open" in target.description
        assert "the first token on the tab's line in the step result" in target.description

    async def test_open_url_target_id_docs_name_the_contract(self) -> None:
        session = FakeSession([])
        tool = find_tool("open_url", _build_tools(session))
        target = next(param for param in tool.params if param.name == "target_id")
        assert "The tab to open the page in" in target.description
        assert "fails if no such tab exists" in target.description
        assert "a new tab is opened" in target.description
        assert "the first token on the tab's line in the step result" in target.description

    @pytest.mark.parametrize(
        "transaction",
        [
            pytest.param(build_success_transaction(), id="success"),
            pytest.param(_build_failed_transaction(), id="failed"),
            pytest.param(_build_stale_transaction(), id="stale"),
        ],
    )
    async def test_result_starts_with_the_time_line(self, transaction: TransactionResult) -> None:
        session = FakeSession([transaction])
        result = await _run_tool(find_tool("click", _build_tools(session)), _PARAMS["click"])
        _extract_after_time_line(result.text)

    async def test_time_line_keeps_the_rest_of_the_result_unchanged(self) -> None:
        session = FakeSession([build_success_transaction()])
        result = await _run_tool(find_tool("click", _build_tools(session)), _PARAMS["click"])
        assert _extract_after_time_line(result.text) == (
            '1. tab1 — "Shop" — https://shop.example — image 1\n'
            "2. tab2 — no screenshot\n"
            "status: success"
        )

    async def test_time_line_appears_exactly_once(self) -> None:
        session = FakeSession([_build_stale_transaction()])
        result = await _run_tool(find_tool("click", _build_tools(session)), _PARAMS["click"])
        assert len([line for line in result.text.splitlines() if _TIME_LINE.match(line)]) == 1

    async def test_dump_page_result_starts_with_the_time_line(self, tmp_path: Path) -> None:
        manager = FakeActionManager(tab_ids=("tab1",))
        tool = find_tool("dump_page", _build_tools(FakeSession([], manager=manager), str(tmp_path)))
        result = await _run_tool(tool, {"target_id": "tab1"})
        rest = _extract_after_time_line(result.text)
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
        monkeypatch.setattr(actions_module, "datetime", FrozenDatetime)
        monkeypatch.setattr(FrozenDatetime, "instant", instant)
        session = FakeSession([build_success_transaction()])
        result = await _run_tool(find_tool("click", _build_tools(session)), _PARAMS["click"])
        first_line, _, _ = result.text.partition("\n")
        assert first_line == expected_line
