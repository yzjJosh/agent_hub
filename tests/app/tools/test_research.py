"""Tests for app.tools.research."""

from __future__ import annotations

import os
import re
from datetime import timedelta
from typing import Protocol
from unittest.mock import AsyncMock

import pytest

import app.lib.db.store as store_module
from app.agents.research.models import ResearchResult
from app.lib.task.task_manager import task_manager
from app.lib.tool.base import ToolResult
from app.tools.research import research_tool
from app.tools.task_tool import task_tool
from tests.app.conftest import is_llm_available
from tests.app.lib.tool.conftest import run_text, wait_terminal


class _TaskStore(Protocol):
    """The task record store the tool_infra fixture gives to a test."""


def _result(report: str) -> ResearchResult:
    """Return a finished research result carrying the given report text."""
    return ResearchResult(report=report, status="complete", iterations=1, review_reason="ok")


class TestResearchShape:
    """The tool declares the research surface and nothing else."""

    def test_tool_shape(self) -> None:
        assert research_tool.name == "research"
        assert research_tool.is_slow is True
        assert research_tool.staleness == timedelta(hours=24)
        assert research_tool.result_render_type == "markdown"
        assert research_tool.summary
        assert "\n" not in research_tool.summary
        assert len(research_tool.summary) <= 80

    def test_single_required_goal_param(self) -> None:
        assert [p.name for p in research_tool.params] == ["goal"]
        assert research_tool.params[0].required
        assert research_tool.params[0].type is str
        assert research_tool.params[0].description
        assert research_tool.params[0].kind == "longtext"

    def test_description_carries_no_synthesized_notes(self) -> None:
        assert "asynchronous" not in research_tool.description
        assert "Duplicate calls" not in research_tool.description
        assert "task_id" not in research_tool.description

    def test_description_disambiguates_from_the_sibling_tools(self) -> None:
        assert "brave_search" in research_tool.description
        assert "google_search" in research_tool.description
        assert "rednote_research" in research_tool.description


class TestValidation:
    """An empty goal fails the run before any research is started."""

    @pytest.mark.parametrize("goal", ["", "   "])
    async def test_empty_goal_fails_the_run_before_research(
        self, goal: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_research_mock = AsyncMock()
        monkeypatch.setattr("app.tools.research.run_research", run_research_mock)

        assert await run_text(research_tool, goal) == "goal must not be empty"

        run_research_mock.assert_not_awaited()


class TestCacheKey:
    """The cache key follows the goal."""

    async def test_cache_key_deterministic(self) -> None:
        first = await research_tool.cache_key({"goal": "the goal"})
        second = await research_tool.cache_key({"goal": "the goal"})
        assert first == second

    async def test_cache_key_distinct_per_goal(self) -> None:
        first = await research_tool.cache_key({"goal": "the goal"})
        second = await research_tool.cache_key({"goal": "another goal"})
        assert first != second


class TestCall:
    """The run folder is a fresh subfolder of the tool task's own run folder."""

    async def test_call_passes_goal_and_fresh_subfolder(
        self, monkeypatch: pytest.MonkeyPatch, tool_infra: _TaskStore
    ) -> None:
        run_research_mock = AsyncMock(return_value=_result("report text"))
        monkeypatch.setattr("app.tools.research.run_research", run_research_mock)

        async def work() -> ToolResult:
            return await research_tool.call({"goal": "the goal"})

        task = await task_manager.create_task(work)
        out = await task.result()

        assert isinstance(out, ToolResult)
        assert out.text == "report text"
        expected = os.path.join(tool_infra.data_dir, store_module.RUNS_DIR, str(task.id), "research")
        run_research_mock.assert_awaited_once_with("the goal", expected)


class TestInvoke:
    """The ack path starts the run and lands the report."""

    async def test_run_returns_handle_and_lands_the_report(
        self, monkeypatch: pytest.MonkeyPatch, tool_infra: _TaskStore
    ) -> None:
        monkeypatch.setattr(
            "app.tools.research.run_research", AsyncMock(return_value=_result("report text"))
        )

        task = await research_tool.run({"goal": "the goal"})

        await wait_terminal(tool_infra, str(task.id))
        assert await run_text(task_tool, action="result", task_id=str(task.id)) == "report text"


@pytest.mark.skipif(not is_llm_available(), reason="LLM server not available")
@pytest.mark.live
@pytest.mark.timeout(900)
async def test_research_live_small_goal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live run answers a small goal with a finished research report."""
    monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "1")

    task = await research_tool.run({"goal": "What is the boiling point of water at sea level?"})
    result = await run_text(task_tool, action="result", task_id=str(task.id), wait=True)

    assert isinstance(result, str)
    assert len(result) > 100
    assert re.search(r"^#{1,6}\s", result, re.MULTILINE)
    assert "[[ref:" not in result
