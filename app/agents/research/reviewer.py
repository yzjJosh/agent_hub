"""Reviewer role for the research agent. Truth-checks the draft report against the evidence."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from app.agents.research.coverage import UNVERIFIED_NOTE, unverified_refs
from app.agents.research.models import Review
from app.agents.research.prompt import load_prompt
from app.lib.agent.checks import (
    check_choice,
    check_nonblank_string,
    check_string,
    combine_problems,
    parse_json_object,
)
from app.lib.agent.engine import Agent, AgentRun
from app.lib.agent.prompt import render_tool_descriptions
from app.lib.tool.base import Tool

__all__ = ["run_reviewer"]

_VALID_ACTIONS = {"rewrite", "collect", "approve"}
_FIELDS = ("pass", "reason", "action", "feedback")
_DRAFT_HEADER = (
    "### DRAFT TO "
    "REVIEW (iteration {iteration})"
)


def _validate_reviewer_output(text: str) -> tuple[bool, dict | str]:
    """Check that the reviewer output is a JSON verdict object.

    Args:
        text: The raw output text from the reviewer agent run.

    Returns:
        An ok flag and a value, the parsed verdict dict when ok, else the message for every problem found.
    """
    ok, parsed = parse_json_object(text)
    if not ok:
        return False, parsed
    problems: list[str] = []
    for field in _FIELDS:
        if field not in parsed:
            problems.append(f"Missing field: {field}.")
    if "pass" in parsed and not isinstance(parsed["pass"], bool):
        problems.append("'pass' must be a boolean.")
    if "reason" in parsed:
        ok, error = check_nonblank_string(parsed, "reason", "'reason' must be a non-empty string.")
        if not ok:
            problems.append(error)
    if "action" in parsed:
        ok, error = check_choice(
            parsed, "action", _VALID_ACTIONS, f"'action' must be one of {sorted(_VALID_ACTIONS)}."
        )
        if not ok:
            problems.append(error)
    if "feedback" in parsed:
        ok, error = check_string(parsed, "feedback", "'feedback' must be a string.")
        if not ok:
            problems.append(error)
    if problems:
        return False, combine_problems(problems)
    return True, parsed


def _reviewer_validator(draft: str) -> Callable[[str, AgentRun], Awaitable[tuple[bool, dict | str]]]:
    """Build the reviewer validator for the draft under review.

    A pass verdict is only accepted when the run verified every reference of
    the draft through a task result read or a search_evidence hit.

    Args:
        draft: The markdown draft report being reviewed.

    Returns:
        The validator callable the reviewer agent is wired with.
    """

    async def validate(text: str, run: AgentRun) -> tuple[bool, dict | str]:
        ok, value = _validate_reviewer_output(text)
        if not ok:
            return ok, value
        if not value["pass"]:
            return ok, value
        missing = await unverified_refs(draft, run)
        if missing:
            return False, (
                f"Verdict is pass but these references were not verified {UNVERIFIED_NOTE}: "
                f"{', '.join(missing)}."
            )
        return ok, value

    return validate


async def run_reviewer(
    goal: str,
    draft: str,
    *,
    tools: tuple[Tool, ...],
    iteration: int = 1,
) -> Review:
    """Run the reviewer role and return the truth-check verdict on the draft.

    Args:
        goal: The research goal the draft covers.
        draft: The markdown draft report to truth-check.
        tools: The in-loop tools wired into the reviewer agent.
        iteration: The current iteration number, starting at 1.

    Returns:
        The verdict with its pass, reason, action, and feedback fields.

    Raises:
        asyncio.CancelledError: When the pipeline is cancelled while the
            reviewer awaits its agent task. The agent task is cancelled first,
            and the whole task tree is awaited before the cancellation
            re-raises.
    """
    tool_descs = render_tool_descriptions(tools)
    system_prompt = await load_prompt("reviewer_prompt", tool_descriptions=tool_descs)
    user_message = _build_user_message(goal, draft, iteration)
    agent = Agent(
        name="research_reviewer",
        system_prompt=system_prompt,
        validator=_reviewer_validator(draft),
    )
    for tool in tools:
        agent.add_tool_adapter(tool)
    agent_run = await agent.run(user_message)
    try:
        verdict = await agent_run.task.result()
    except asyncio.CancelledError:
        await asyncio.shield(agent_run.task.cancel())
        raise
    verdict = json.loads(verdict)
    if not isinstance(verdict, dict):
        raise RuntimeError("reviewer agent run did not return a verdict dict")
    return Review(
        passed=verdict["pass"],
        reason=verdict["reason"],
        action=verdict["action"],
        feedback=verdict["feedback"],
        iteration=iteration,
    )


def _build_user_message(goal: str, draft: str, iteration: int) -> str:
    parts = [f"Research goal: {goal}", ""]
    parts.append(_DRAFT_HEADER.format(iteration=iteration))
    parts.append(draft)
    return "\n".join(parts)
