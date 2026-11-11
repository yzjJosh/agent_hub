from __future__ import annotations

"""Deep research over a goal.

Plans the work, gathers evidence from multiple web sources, drafts, and
reviews before returning a cited markdown report.
"""

__all__ = ["research_tool"]

import hashlib
import json
import os
from datetime import timedelta
from typing import ClassVar, override

import app.lib.db.store as store_module
from app.agents.research.runner import run_research
from app.lib.task.task_manager import task_manager
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult


class _ResearchTool(Tool):
    name: ClassVar[str] = "research"
    is_slow: ClassVar[bool] = True
    staleness: ClassVar[timedelta] = timedelta(hours=24)
    result_render_type: ClassVar[str] = "markdown"
    description: ClassVar[str] = (
        "Deep research over a goal: plans the work, gathers evidence from "
        "multiple web sources, drafts, and reviews before returning a cited "
        "markdown report. Use for research questions that need multiple "
        "sources and a synthesized report. The run takes hours and the goal "
        "cannot be changed once the run starts, so state the goal completely "
        "with its scope well defined. For a single web search use "
        "brave_search or google_search, and for Xiaohongshu-only research "
        "use rednote_research."
    )
    summary: ClassVar[str] = "Deep research report over a goal via multiple sources"
    params: ClassVar[tuple[Param, ...]] = (
        Param(
            "goal",
            str,
            description="The research question to answer. It must be comprehensive and well scoped.",
            kind="longtext",
        ),
    )

    @override
    async def call(self, params: dict) -> ToolResult:
        goal = params["goal"]
        if not goal.strip():
            raise ToolMessageError("goal must not be empty")
        task = task_manager.active_task()
        task_id = str(task.id)
        run_folder = os.path.join(
            store_module.task_store.data_dir, store_module.RUNS_DIR, task_id, "research"
        )
        result = await run_research(goal, run_folder)
        return ToolResult(text=result.report)

    @override
    async def cache_key(self, params: dict) -> str:
        payload = json.dumps([params["goal"]], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


research_tool = _ResearchTool()
