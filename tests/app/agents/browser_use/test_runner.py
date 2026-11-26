"""Tests for the browser use runner lifecycle and agent wiring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from agents import Agent as SdkAgent, MaxTurnsExceeded, Runner
from agents.run_config import RunConfig

from app.agents.browser_use import history as history_module
from app.agents.browser_use import runner as runner_module
from app.agents.browser_use.checkpoint import CheckpointTool
from app.agents.browser_use.checkpoint_store import CheckpointError, CheckpointStore
from app.agents.browser_use.models import BrowserUseResult
from app.agents.browser_use.prompts.role import ROLE_OUTPUT_JSON_SCHEMA
from app.lib.shared import config
from app.lib.tool.base import Tool
from app.tools.read_media_or_doc import read_media_or_doc_tool
from app.tools.task_tool import task_tool
from tests.app.agents.browser_use.conftest import (
    COMPLETED_OUTPUT_RAW,
    STANDARD_ACTION_TOOL_NAMES,
    FakeSession,
    build_mock_stream,
    clear_llm_backends,
    find_tool,
    set_thinking_backend,
)
from tests.app.conftest import is_llm_available, skip_if_no_chrome

_LIVE_TIMEOUT_SECONDS = 600
_EXPECTED_MAX_TURNS = 180
_EXPECTED_WRAP_UP_TURNS_REMAINING = 10
_STAGE_ONE_MAX_TURNS = _EXPECTED_MAX_TURNS - _EXPECTED_WRAP_UP_TURNS_REMAINING
_IMPOSSIBLE_RAW = json.dumps({"status": "impossible", "summary": "cannot"})
_INCOMPLETE_SUMMARY = "Reached turn limit after downloading the report."
_INCOMPLETE_RAW = json.dumps({"status": "incomplete", "summary": _INCOMPLETE_SUMMARY})


class FakeTask:
    """Fake agent task handle awaited by the runner."""

    def __init__(self, value: str | None, error: Exception | None, hang: bool) -> None:
        self._value = value
        self._error = error
        self._hang = hang
        self.cancelled: int = 0

    async def result(self) -> str:
        if self._hang:
            await asyncio.Event().wait()
        if self._error is not None:
            raise self._error
        return self._value

    async def cancel(self) -> None:
        self.cancelled += 1


class FakeAgentRun:
    """Fake per-run handle awaited by the runner."""

    def __init__(self, task: FakeTask) -> None:
        self.task = task


class FakeAgent:
    """Recording stand-in for the engine Agent."""

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
    """Point the runner Agent class at the fake and return the recording mock."""
    agent_mock = MagicMock(return_value=agent)
    monkeypatch.setattr(runner_module, "Agent", agent_mock)
    return agent_mock


def _wire_session(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> dict[str, str]:
    """Point the runner BrowserSession class at the fake and record downloads_dir."""
    seen: dict[str, str] = {}

    def make_session(downloads_dir: str, *, checkpoint_store: CheckpointStore) -> FakeSession:
        seen["downloads_dir"] = downloads_dir
        session.checkpoint_store = checkpoint_store
        return session

    monkeypatch.setattr(runner_module, "BrowserSession", make_session)
    return seen


async def _finish_agent(
    monkeypatch: pytest.MonkeyPatch,
    session: FakeSession,
    raw: str,
    resources_dir: str = "/tmp/res",
) -> tuple[FakeAgent, MagicMock, BrowserUseResult, dict[str, str]]:
    """Wire the fakes, run the runner to completion, and return the recorded objects."""
    agent = FakeAgent()
    agent.agent_run = FakeAgentRun(FakeTask(raw, None, False))
    agent_mock = _wire_agent(monkeypatch, agent)
    seen = _wire_session(monkeypatch, session)
    result = await runner_module.run_browser_use("goal", "/tmp/dl", resources_dir)
    return agent, agent_mock, result, seen


class TestRunBrowserUse:
    async def test_checkpoint_tool_is_inline_and_store_closes_after_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        set_thinking_backend(monkeypatch)
        session = FakeSession([])
        agent, _, _, _ = await _finish_agent(monkeypatch, session, COMPLETED_OUTPUT_RAW)
        tool = find_tool("checkpoint", tuple(agent.tools))
        assert isinstance(tool, CheckpointTool)
        assert agent.results_by_id[tool] is None
        with pytest.raises(CheckpointError, match="store is closed"):
            await session.checkpoint_store.list()

    @pytest.mark.parametrize("failure", ["open", "close"])
    async def test_store_closes_when_session_lifecycle_raises(
        self, monkeypatch: pytest.MonkeyPatch, failure: str
    ) -> None:
        set_thinking_backend(monkeypatch)
        session = FakeSession([])
        if failure == "open":
            session.open_error = RuntimeError("open failed")
        else:
            session.close_error = RuntimeError("close failed")
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            await _finish_agent(monkeypatch, session, COMPLETED_OUTPUT_RAW)
        assert session.closed == 1
        with pytest.raises(CheckpointError, match="store is closed"):
            await session.checkpoint_store.list()

    async def test_agent_finish_first_returns_the_validated_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        set_thinking_backend(monkeypatch)
        session = FakeSession([], downloads=["/tmp/res/downloads/txn_1/download"])
        _, _, result, seen = await _finish_agent(monkeypatch, session, COMPLETED_OUTPUT_RAW)
        assert result == BrowserUseResult("completed", "done", ("downloads/txn_1/download",))
        assert seen == {"downloads_dir": "/tmp/dl"}

    async def test_agent_is_built_with_the_role_and_the_sixteen_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        set_thinking_backend(monkeypatch)
        agent, agent_mock, _, _ = await _finish_agent(monkeypatch, FakeSession([]), _IMPOSSIBLE_RAW)
        kwargs = agent_mock.call_args.kwargs
        assert kwargs["name"] == "browser_use"
        assert kwargs["validator"] is runner_module.validate_role_output
        assert kwargs["backend"] == config.Backend(url="http://think.test/v1", model=None, api_key=None)
        assert kwargs["max_turns"] == _EXPECTED_MAX_TURNS
        assert kwargs["wrap_up_turns_remaining"] == _EXPECTED_WRAP_UP_TURNS_REMAINING
        assert kwargs["tool_results_by_id"] is False
        assert kwargs["model_input_filter"] is history_module.prune_model_history
        assert kwargs["compact_hook"] is history_module.compact_dom_dumps
        assert kwargs["output_json_schema"] == ROLE_OUTPUT_JSON_SCHEMA
        assert [tool.name for tool in agent.tools] == (
            list(STANDARD_ACTION_TOOL_NAMES[:2])
            + ["upload_resources"]
            + list(STANDARD_ACTION_TOOL_NAMES[2:])
            + ["checkpoint", "task", "read_media_or_doc"]
        )
        assert agent.prompt == "goal"
        assert "target id" in kwargs["system_prompt"]

    async def test_reading_tools_are_added_with_their_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        set_thinking_backend(monkeypatch)
        agent, _, _, _ = await _finish_agent(monkeypatch, FakeSession([]), COMPLETED_OUTPUT_RAW)
        assert any(tool is task_tool for tool in agent.tools)
        assert any(tool is read_media_or_doc_tool for tool in agent.tools)
        assert agent.results_by_id[task_tool] is None
        assert agent.results_by_id[read_media_or_doc_tool] is True

    @pytest.mark.parametrize("name", STANDARD_ACTION_TOOL_NAMES)
    async def test_action_tools_are_added_plain(self, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        set_thinking_backend(monkeypatch)
        agent, _, _, _ = await _finish_agent(monkeypatch, FakeSession([]), COMPLETED_OUTPUT_RAW)
        tool = find_tool(name, tuple(agent.tools))
        assert agent.results_by_id[tool] is None

    async def test_task_is_the_bare_registered_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        set_thinking_backend(monkeypatch)
        agent, _, _, _ = await _finish_agent(monkeypatch, FakeSession([]), COMPLETED_OUTPUT_RAW)
        tool = next(tool for tool in agent.tools if tool.name == "task")
        assert tool is task_tool
        assert tool.name == "task"
        action = next(param for param in task_tool.params if param.name == "action")
        assert action.enum == ("status", "result", "logs", "cancel")

    async def test_backend_carries_the_thinking_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        set_thinking_backend(monkeypatch)
        monkeypatch.setenv("REASON_THINKING_MODEL", "deep")
        monkeypatch.setenv("REASON_BACKEND_URL", "http://plain.test")
        monkeypatch.setenv("REASON_MODEL", "fast")
        _, agent_mock, _, _ = await _finish_agent(monkeypatch, FakeSession([]), COMPLETED_OUTPUT_RAW)
        expected = config.Backend(url="http://think.test/v1", model="deep", api_key=None)
        assert agent_mock.call_args.kwargs["backend"] == expected

    async def test_backend_falls_back_to_the_plain_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clear_llm_backends(monkeypatch)
        monkeypatch.setenv("REASON_THINKING_MODEL", "deep")
        monkeypatch.setenv("REASON_BACKEND_URL", "http://plain.test")
        monkeypatch.setenv("REASON_MODEL", "fast")
        _, agent_mock, _, _ = await _finish_agent(monkeypatch, FakeSession([]), COMPLETED_OUTPUT_RAW)
        expected = config.Backend(url="http://plain.test/v1", model="fast", api_key=None)
        assert agent_mock.call_args.kwargs["backend"] == expected

    async def test_missing_backends_raise(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        clear_llm_backends(monkeypatch)
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
        set_thinking_backend(monkeypatch)
        session = FakeSession([], downloads=["/tmp/other/x.pdf"])
        _, _, result, _ = await _finish_agent(monkeypatch, session, COMPLETED_OUTPUT_RAW)
        assert result.files == ("/tmp/other/x.pdf",)

    async def test_turn_limit_wrap_up_returns_incomplete_summary_with_downloads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        set_thinking_backend(monkeypatch)
        downloads_dir = str(tmp_path / "dl")
        resources_dir = str(tmp_path / "res")
        downloaded_path = str(tmp_path / "res" / "downloads" / "txn_2" / "download")
        session = FakeSession([], downloads=[downloaded_path])
        _wire_session(monkeypatch, session)
        seen_max_turns: list[int | None] = []
        seen_inputs: list[str | list] = []

        def run_streamed(
            _agent: SdkAgent,
            items: str | list,
            *,
            run_config: RunConfig,
            max_turns: int | None,
        ) -> MagicMock:
            _ = run_config
            seen_max_turns.append(max_turns)
            seen_inputs.append(items)
            if max_turns == _STAGE_ONE_MAX_TURNS:
                return build_mock_stream(
                    [{"role": "user", "content": "goal"}],
                    MaxTurnsExceeded("limit"),
                    turns=_STAGE_ONE_MAX_TURNS,
                    new_items=[],
                )
            if max_turns == _EXPECTED_WRAP_UP_TURNS_REMAINING:
                return build_mock_stream(
                    list(items),
                    MaxTurnsExceeded("limit"),
                    turns=_EXPECTED_WRAP_UP_TURNS_REMAINING,
                    new_items=[],
                )
            return build_mock_stream(list(items), None, final_output=_INCOMPLETE_RAW, turns=1, new_items=[])

        monkeypatch.setattr(Runner, "run_streamed", run_streamed)
        result = await runner_module.run_browser_use("goal", downloads_dir, resources_dir)
        assert seen_max_turns == [_STAGE_ONE_MAX_TURNS, _EXPECTED_WRAP_UP_TURNS_REMAINING, 1]
        assert "tools are now disabled" in seen_inputs[-1][-1]["content"]
        assert result == BrowserUseResult("incomplete", _INCOMPLETE_SUMMARY, ("downloads/txn_2/download",))
        assert session.closed == 1
        with pytest.raises(CheckpointError, match="store is closed"):
            await session.checkpoint_store.list()

    async def test_agent_failure_propagates_and_closes_the_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        set_thinking_backend(monkeypatch)
        agent = FakeAgent()
        agent.agent_run = FakeAgentRun(FakeTask(None, RuntimeError("agent validation failed"), False))
        _wire_agent(monkeypatch, agent)
        session = FakeSession([])
        _wire_session(monkeypatch, session)
        with pytest.raises(RuntimeError, match="agent validation failed"):
            await runner_module.run_browser_use("goal", "/tmp/dl", "/tmp/res")
        assert session.closed == 1
        with pytest.raises(CheckpointError, match="store is closed"):
            await session.checkpoint_store.list()

    async def test_cancelled_run_reraises_and_closes_the_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        set_thinking_backend(monkeypatch)
        agent = FakeAgent()
        agent.agent_run = FakeAgentRun(FakeTask(None, None, True))
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


@skip_if_no_chrome
@pytest.mark.live
class TestLiveRunner:
    @pytest.mark.timeout(_LIVE_TIMEOUT_SECONDS)
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

    @pytest.mark.timeout(_LIVE_TIMEOUT_SECONDS)
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

    @pytest.mark.timeout(_LIVE_TIMEOUT_SECONDS)
    async def test_dump_page_and_report_offscreen_text(self, tmp_path: Path, test_server_url: str) -> None:
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

    @pytest.mark.skipif(not is_llm_available(), reason="LLM server not available")
    @pytest.mark.timeout(_LIVE_TIMEOUT_SECONDS)
    @pytest.mark.parametrize(
        ("element_id", "element_label"),
        [
            pytest.param("top-section", "the block at the top of the page with the heading Top", id="large-block"),
            pytest.param("mid-input", "the text box with the placeholder Type here", id="medium-text-box"),
            pytest.param("mid-button", "the button labeled Click Me", id="small-button"),
        ],
    )
    async def test_click_lands_inside_labeled_element(
        self, tmp_path: Path, test_server_url: str, element_id: str, element_label: str
    ) -> None:
        """The model's last click on the page lands inside the element it was asked to click."""
        downloads_dir = tmp_path / "downloads"
        resources_dir = tmp_path / "resources"
        downloads_dir.mkdir()
        resources_dir.mkdir()
        token = uuid4().hex
        prompt = (
            f"Open {test_server_url}/page/long/{token} in the browser. "
            f"Find {element_label} on the page, click it once, and then finish."
        )

        result = await runner_module.run_browser_use(prompt, str(downloads_dir), str(resources_dir))

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{test_server_url}/clicks/{token}")
        clicks = response.json()
        assert clicks, f"No click reached the page. Summary: {result.summary}"
        last_click = clicks[-1]
        box = last_click["boxes"][element_id]
        click_report = f"click at ({last_click['x']}, {last_click['y']}), {element_id} box {box}"
        print(click_report)
        assert box["left"] <= last_click["x"] <= box["right"], click_report
        assert box["top"] <= last_click["y"] <= box["bottom"], click_report
        assert result.status == "completed", result.summary

    @pytest.mark.timeout(_LIVE_TIMEOUT_SECONDS)
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
        assert all(Path(file_path).is_file() and Path(file_path).read_bytes() == expected for file_path in result.files)
