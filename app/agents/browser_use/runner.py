"""Run the browser use agent with screenshots, page dumps, and saved checkpoints."""

from __future__ import annotations

import asyncio
import json

from app.agents.browser_use.actions import build_action_tools, format_resource_paths
from app.agents.browser_use.checkpoint import CheckpointTool
from app.agents.browser_use.checkpoint_store import CheckpointStore
from app.agents.browser_use.history import compact_dom_dumps, prune_model_history
from app.agents.browser_use.models import BrowserUseResult
from app.agents.browser_use.prompts.role import (
    ROLE_OUTPUT_JSON_SCHEMA,
    build_role_prompt,
    validate_role_output,
)
from app.agents.browser_use.session import BrowserSession
from app.lib.agent.engine import Agent, AgentRun
from app.lib.shared import config
from app.lib.tool.base import Tool
from app.tools.read_media_or_doc import read_media_or_doc_tool
from app.tools.task_tool import task_tool

__all__ = ["run_browser_use"]

_AGENT_NAME = "browser_use"
_MAX_TURNS = 180
_WRAP_UP_TURNS_REMAINING = 10


def _format_final_result(
    raw_output: str, downloads: list[str], resources_directory: str
) -> BrowserUseResult:
    """Parse the validated agent output and attach formatted download paths."""
    parsed = json.loads(raw_output)
    return BrowserUseResult(
        status=parsed["status"],
        summary=parsed["summary"],
        files=format_resource_paths(downloads, resources_directory),
    )


def _build_agent(tools: tuple[Tool, ...], backend: config.Backend) -> Agent:
    """Build the browser use agent wired with tools and history hooks."""
    agent = Agent(
        name=_AGENT_NAME,
        system_prompt=build_role_prompt(tools + (task_tool, read_media_or_doc_tool)),
        validator=validate_role_output,
        backend=backend,
        max_turns=_MAX_TURNS,
        wrap_up_turns_remaining=_WRAP_UP_TURNS_REMAINING,
        tool_results_by_id=False,
        model_input_filter=prune_model_history,
        compact_hook=compact_dom_dumps,
        output_json_schema=ROLE_OUTPUT_JSON_SCHEMA,
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
    """Run the browser agent toward a goal and return the outcome.

    Args:
        prompt: Goal to carry out in the browser.
        downloads_dir: Folder where session downloads and checkpoints are stored.
        resources_dir: Root resources folder used to format returned file paths.

    Returns:
        Run outcome with status, summary, and downloaded file paths.

    Raises:
        asyncio.CancelledError: If the run is cancelled while waiting for the agent.
        MissingSettingError: If neither LLM backend URL is configured.
    """
    llm_backend = config.backend(config.REASON_THINKING, config.REASON)
    store = CheckpointStore(downloads_dir, resources_root=resources_dir)
    session = BrowserSession(downloads_dir, checkpoint_store=store)
    agent_run: AgentRun | None = None
    try:
        await session.open()
        tools = build_action_tools(session, resources_dir) + (CheckpointTool(store),)
        agent = _build_agent(tools, llm_backend)
        agent_run = await agent.run(prompt)
        raw_output = await agent_run.task.result()
        if raw_output is None:
            raise RuntimeError("browser use agent returned no output")
        return _format_final_result(raw_output, await session.get_downloaded_files(), resources_dir)
    except asyncio.CancelledError:
        if agent_run is not None:
            await asyncio.shield(agent_run.task.cancel())
        raise
    finally:
        try:
            await session.close()
        finally:
            await store.close()
