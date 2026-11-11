"""Researcher role for the research agent. Collects evidence as needed and writes the report."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable

from app.agents.research.collector import CollectionLedger, format_records
from app.agents.research.coverage import UNVERIFIED_NOTE, unverified_refs
from app.agents.research.models import REF_RE, Review
from app.agents.research.prompt import REVIEW_FEEDBACK_HEADER, load_prompt
from app.lib.agent.checks import combine_problems, normalize_markdown_body
from app.lib.agent.engine import Agent, AgentRun
from app.lib.agent.prompt import render_tool_descriptions
from app.lib.tool.base import Tool

__all__ = ["run_researcher"]

_MIN_REPORT_LEN = 50
_TITLE_RE = re.compile(r"^# \S")
_COLLECTION_HEADER = "### COLLECTION SO FAR"
_PREVIOUS_DRAFT_HEADER = "### PREVIOUS DRAFT"


def _report_problems(text: str, known_run_ids: frozenset[str]) -> list[str]:
    """Check the report length, the markdown title, the cited runs, and the placeholder usage.

    Args:
        text: The report body from the researcher agent run.
        known_run_ids: The run ids that may be cited.

    Returns:
        One message per problem found, empty when the report is valid.
    """
    problems: list[str] = []
    stripped = text.strip()
    if len(stripped) < _MIN_REPORT_LEN:
        problems.append(f"Report is too short ({len(stripped)} chars, minimum {_MIN_REPORT_LEN}).")
    first = next((line for line in stripped.splitlines() if line.strip()), "")
    if not _TITLE_RE.match(first):
        problems.append("Report must start with a markdown title (a first line starting with '# ').")
    refs = REF_RE.findall(text)
    unknown_refs = [ref for ref in refs if ref not in known_run_ids]
    if unknown_refs:
        problems.append(f"Unknown run id in placeholder: {', '.join(unknown_refs)}.")
    if known_run_ids and not refs:
        problems.append("Report must contain at least one [[ref:<run_id>]] placeholder when evidence is available.")
    if not known_run_ids and refs:
        problems.append("Report must not contain placeholders when no evidence is available.")
    return problems


def _researcher_validator(
    ledger: CollectionLedger,
    prior_draft_refs: frozenset[str],
) -> Callable[[str, AgentRun], Awaitable[tuple[bool, str]]]:
    """Build the researcher validator over the run's collection ledger.

    The report is only accepted once some collection happened, and when every
    cited reference is citable and was verified by the run, or was already in
    the previous draft.

    Args:
        ledger: The collection ledger, read each time the validator runs.
        prior_draft_refs: The run ids cited by the previous draft.

    Returns:
        The validator callable the researcher agent is wired with.
    """

    async def validate(text: str, run: AgentRun) -> tuple[bool, str]:
        ok, body = normalize_markdown_body(text)
        if not ok:
            return False, body
        problems: list[str] = []
        if not ledger.records:
            problems.append("Call collect_evidence before writing the report.")
        known_run_ids = frozenset(ledger.registry)
        problems.extend(_report_problems(body, known_run_ids))
        missing = [
            ref
            for ref in await unverified_refs(body, run)
            if ref in known_run_ids and ref not in prior_draft_refs
        ]
        if missing:
            problems.append(
                f"Report cites references that were not verified {UNVERIFIED_NOTE}: "
                f"{', '.join(missing)}."
            )
        if problems:
            return False, combine_problems(problems)
        return True, body

    return validate


async def run_researcher(
    goal: str,
    ledger: CollectionLedger,
    *,
    tools: tuple[Tool, ...],
    previous_draft: str | None = None,
    review: Review | None = None,
) -> str:
    """Run the researcher role and return the validated markdown research report.

    Args:
        goal: The research goal the report covers.
        ledger: The research run's collection ledger, read and never written.
        tools: The in-loop tools wired into the researcher agent.
        previous_draft: The report from the prior iteration, if any.
        review: The failed review of the previous draft, if any.

    Returns:
        The markdown report text accepted by the validator.

    Raises:
        asyncio.CancelledError: When the pipeline is cancelled while the
            researcher awaits its agent task. The agent task is cancelled first,
            and the whole task tree is awaited before the cancellation
            re-raises.
    """
    tool_descs = render_tool_descriptions(tools)
    system_prompt = await load_prompt("researcher_prompt", tool_descriptions=tool_descs)
    user_message = _build_user_message(goal, ledger, previous_draft, review)
    prior_draft_refs = frozenset(REF_RE.findall(previous_draft or ""))
    agent = Agent(
        name="researcher",
        system_prompt=system_prompt,
        validator=_researcher_validator(ledger, prior_draft_refs),
    )
    for tool in tools:
        agent.add_tool_adapter(tool)
    agent_run = await agent.run(user_message)
    try:
        return await agent_run.task.result()
    except asyncio.CancelledError:
        await asyncio.shield(agent_run.task.cancel())
        raise


def _build_user_message(
    goal: str,
    ledger: CollectionLedger,
    previous_draft: str | None,
    review: Review | None,
) -> str:
    parts = [f"Research goal: {goal}"]
    if ledger.records:
        parts.extend(["", _COLLECTION_HEADER, format_records(ledger.records)])
    if review is not None:
        parts.extend(["", REVIEW_FEEDBACK_HEADER, f"Reviewer action: {review.action}", review.feedback])
    if previous_draft is not None:
        parts.extend(["", _PREVIOUS_DRAFT_HEADER, previous_draft])
    return "\n".join(parts)
