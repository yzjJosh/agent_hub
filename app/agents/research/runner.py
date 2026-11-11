"""Runner orchestrator for the research agent. Drives the research and review loop."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from app.agents.research.collector import CollectEvidence, CollectionLedger
from app.agents.research.evidence import EvidenceStore, SearchEvidence
from app.agents.research.models import ResearchResult, Review
from app.agents.research.references import process_references
from app.agents.research.researcher import run_researcher
from app.agents.research.reviewer import run_reviewer
from app.tools.read_media_or_doc import read_media_or_doc_tool
from app.tools.task_tool import task_tool

__all__ = ["run_research"]


async def run_research(goal: str, run_folder: str | Path) -> ResearchResult:
    """Run the research pipeline for a goal and return the final report result.

    Args:
        goal: The research goal the pipeline answers.
        run_folder: The run folder that holds the per-run evidence store.

    Returns:
        The final report with its status, iteration count, and review reason.

    Raises:
        ValueError: When the goal is empty, the run folder is not empty, or
            RESEARCH_MAX_ITERATIONS is below 1.
    """
    if not goal.strip():
        raise ValueError("goal must be non-empty")
    max_iterations = int(os.getenv("RESEARCH_MAX_ITERATIONS") or "5")
    if max_iterations < 1:
        raise ValueError("RESEARCH_MAX_ITERATIONS must be at least 1")
    run_folder = Path(run_folder)
    await asyncio.to_thread(_prepare_run_folder, run_folder)

    store = EvidenceStore(run_folder)
    await store.open()
    search_evidence = SearchEvidence(store)
    ledger = CollectionLedger()
    collect_evidence = CollectEvidence(goal, store, ledger)

    try:
        return await _run_pipeline(goal, ledger, collect_evidence, search_evidence, max_iterations)
    finally:
        await store.close()


def _prepare_run_folder(run_folder: Path) -> None:
    if run_folder.exists():
        entries = list(run_folder.iterdir())
        if entries:
            raise ValueError(f"run folder is not empty: {entries[0].name}")
    else:
        run_folder.mkdir(parents=True)


async def _run_pipeline(
    goal: str,
    ledger: CollectionLedger,
    collect_evidence: CollectEvidence,
    search_evidence: SearchEvidence,
    max_iterations: int,
) -> ResearchResult:
    previous_draft: str | None = None
    draft: str | None = None
    review: Review | None = None
    status: str = "max_iterations"

    for iteration in range(1, max_iterations + 1):
        draft = await run_researcher(
            goal, ledger,
            tools=(collect_evidence, search_evidence, task_tool, read_media_or_doc_tool),
            previous_draft=previous_draft,
            review=review,
        )
        review = await run_reviewer(
            goal, draft,
            tools=(search_evidence, task_tool, read_media_or_doc_tool),
            iteration=iteration,
        )
        if review.passed:
            status = "complete"
            break
        previous_draft = draft

    final_report = await process_references(draft or "", ledger.registry)
    return ResearchResult(
        report=final_report,
        status=status,
        iterations=iteration,
        review_reason=review.reason if review else "",
    )
