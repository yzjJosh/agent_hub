from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

from app.agents.browser_use.models import BrowserUseResult
from app.lib.shared.paths import resources_dir
from app.lib.task.task_manager import AsyncTask, task_manager
from app.lib.tool.base import ToolResult
from app.tools.browser_use import browser_use_tool
from tests.app.lib.tool.conftest import run_text


def _result(status: str, summary: str, files: tuple[str, ...]) -> BrowserUseResult:
    """A finished browser use result with the given shape."""
    return BrowserUseResult(status, summary, files)


async def _run_call(prompt: str) -> tuple[AsyncTask[ToolResult], ToolResult]:
    """The tracked task for one call, and the result it lands."""

    async def work() -> ToolResult:
        return await browser_use_tool.call({"prompt": prompt})

    task = await task_manager.create_task(work)
    return task, await task.result()


class TestBrowserUseShape:
    """The tool declares the browser use surface and nothing else."""

    def test_tool_shape(self) -> None:
        assert browser_use_tool.name == "browser_use"
        assert browser_use_tool.is_slow is True
        assert browser_use_tool.staleness is None
        assert browser_use_tool.result_render_type == "markdown"
        assert browser_use_tool.summary
        assert "\n" not in browser_use_tool.summary
        assert len(browser_use_tool.summary) <= 80

    def test_single_required_prompt_param(self) -> None:
        assert [p.name for p in browser_use_tool.params] == ["prompt"]
        assert browser_use_tool.params[0].required
        assert browser_use_tool.params[0].type is str
        assert browser_use_tool.params[0].kind == "longtext"
        assert browser_use_tool.params[0].description

    def test_description_disambiguates_from_the_sibling_tools(self) -> None:
        assert "web_fetch" in browser_use_tool.description
        assert "web_screenshot" in browser_use_tool.description
        assert "task" in browser_use_tool.description
        assert "send_or_receive_file" in browser_use_tool.description


class TestValidation:
    """An empty prompt fails the run before any browsing is started."""

    @pytest.mark.parametrize("prompt", ["", "   "])
    async def test_empty_prompt_fails_the_run_before_browsing(
        self, prompt: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_browser_use_mock = AsyncMock()
        monkeypatch.setattr("app.tools.browser_use.run_browser_use", run_browser_use_mock)

        assert await run_text(browser_use_tool, prompt) == "prompt must not be empty"

        run_browser_use_mock.assert_not_awaited()


class TestCall:
    """The downloads dir is a per-run subfolder of the resources folder."""

    async def test_call_passes_the_prompt_and_the_downloads_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        files = ("browser_use/file.png", "browser_use/report.pdf")
        run_browser_use_mock = AsyncMock(
            return_value=_result("completed", "the summary", files)
        )
        monkeypatch.setattr("app.tools.browser_use.run_browser_use", run_browser_use_mock)

        task, out = await _run_call("the goal")

        assert isinstance(out, ToolResult)
        expected = os.path.join(resources_dir(), "browser_use", str(task.id))
        run_browser_use_mock.assert_awaited_once_with("the goal", expected, resources_dir())
        assert out.text == (
            "## Browser run\n"
            "\n"
            "**Status:** completed\n"
            "\n"
            "the summary\n"
            "\n"
            "**Files**\n"
            "- browser_use/file.png\n"
            "- browser_use/report.pdf\n"
            "\n"
            "See the send_or_receive_file tool for file transfer details."
        )
        assert out.files == files

    async def test_call_with_no_files_omits_the_files_section(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_browser_use_mock = AsyncMock(return_value=_result("completed", "the summary", ()))
        monkeypatch.setattr("app.tools.browser_use.run_browser_use", run_browser_use_mock)

        _, out = await _run_call("the goal")

        assert out.text == "## Browser run\n\n**Status:** completed\n\nthe summary"
        assert out.files == ()

    async def test_incomplete_result_with_files_renders_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        files = ("browser_use/step-2/file.zip",)
        run_browser_use_mock = AsyncMock(return_value=_result("incomplete", "", files))
        monkeypatch.setattr("app.tools.browser_use.run_browser_use", run_browser_use_mock)

        _, out = await _run_call("the goal")

        assert out.text == (
            "## Browser run\n"
            "\n"
            "**Status:** incomplete\n"
            "\n"
            "**Files**\n"
            "- browser_use/step-2/file.zip\n"
            "\n"
            "See the send_or_receive_file tool for file transfer details."
        )
        assert out.files == files


class StubSession:
    """A browser session that opens and closes without a browser."""

    def __init__(self, _downloads_dir: str) -> None:
        pass

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None


class TestMissingBackend:
    """A run with no LLM backend fails and names both variables."""

    async def test_no_llm_backend_fails_the_run(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("REASON_THINKING_BACKEND_URL", raising=False)
        monkeypatch.delenv("REASON_BACKEND_URL", raising=False)
        monkeypatch.setattr("app.agents.browser_use.runner.BrowserSession", StubSession)
        message = "REASON_THINKING_BACKEND_URL or REASON_BACKEND_URL is not set: needed by research and browser_use"

        assert await run_text(browser_use_tool, "the goal") == f"Tool 'browser_use' failed: {message}"
        assert f"[config] {message}\n" in capsys.readouterr().err
