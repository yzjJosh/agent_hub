"""Reference coverage for research report drafts. Lists cited run ids the run never verified."""

from __future__ import annotations

import os
import re

import app.lib.db.store as store_module
from app.agents.research.models import REF_RE
from app.lib.agent.engine import AgentRun
from app.lib.tool.base import read_run_mcp_text

__all__ = ["UNVERIFIED_NOTE", "unverified_refs"]

UNVERIFIED_NOTE = (
    "(not read with task(action='result') and not surfaced in a search_evidence result)"
)
_RUN_ID_LINE_RE = re.compile(r"run_id: `([^`]+)`")


async def unverified_refs(draft: str, run: AgentRun) -> list[str]:
    """List the draft's reference run ids the run never verified, in draft order.

    A ref is verified when it was read with a task result call or surfaced
    in a completed search_evidence result.

    Args:
        draft: The markdown report with [[ref:<run_id>]] placeholders.
        run: The agent run whose conversation is checked.

    Returns:
        The reference run ids with no read and no search_evidence hit, in draft order.
    """
    refs = list(dict.fromkeys(REF_RE.findall(draft)))
    if not refs:
        return []
    verified: set[str] = set()
    for event in await run.events():
        if event.type != "tool_call" or event.tool_name != "task":
            continue
        params = event.params
        if params and params.get("action") == "result" and params.get("task_id"):
            verified.add(str(params["task_id"]))
    for call in await run.tool_calls():
        if call.tool_name != "search_evidence" or call.status != "completed":
            continue
        run_path = os.path.join(store_module.task_store.data_dir, call.run_dir)
        try:
            result_text = await read_run_mcp_text(run_path)
        except OSError:
            continue
        verified.update(_RUN_ID_LINE_RE.findall(result_text))
    return [ref for ref in refs if ref not in verified]
