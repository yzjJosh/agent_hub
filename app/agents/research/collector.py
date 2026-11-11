"""Collector role for the research agent. Gathers raw data for one research task."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from typing import ClassVar, override

import app.lib.db.store as store_module
from app.agents.research.evidence import EvidenceStore
from app.agents.research.models import (
    CitedRun,
    CollectorResult,
    DATA_SOURCE_TOOLS,
    EvidenceItem,
    ResearchTask,
    TaskState,
)
from app.agents.research.prompt import load_prompt
from app.lib.agent.checks import check_nonblank_string, combine_problems, parse_json_object
from app.lib.agent.engine import Agent, AgentRun
from app.lib.agent.prompt import render_tool_descriptions
from app.lib.tool.base import Param, Tool, ToolMessageError, ToolResult, read_run_mcp_text
from app.tools.brave_search import brave_search_tool
from app.tools.github import github_tool
from app.tools.google_search import google_search_tool
from app.tools.read_media_or_doc import read_media_or_doc_tool
from app.tools.rednote_research import rednote_research_tool
from app.tools.task_tool import task_tool
from app.tools.web_fetch import web_fetch_tool
from app.tools.x_search import x_search_tool

__all__ = ["CollectEvidence", "CollectionLedger", "format_records"]

_MAX_LISTED_IDS = 10
_MAX_PARALLEL = 3
_COLLECTOR_TOOLS: tuple[Tool, ...] = (
    brave_search_tool, google_search_tool, x_search_tool, github_tool,
    web_fetch_tool, rednote_research_tool, task_tool, read_media_or_doc_tool,
)


def _describe_completed_calls(task_ids: list[str]) -> str:
    """Name the task ids a citation may point at.

    Args:
        task_ids: The task ids of the completed data-source calls of the run.

    Returns:
        A sentence listing the task ids, capped at a few of them.
    """
    if not task_ids:
        return "No data-source call completed."
    listed = ", ".join(task_ids[:_MAX_LISTED_IDS])
    if len(task_ids) > _MAX_LISTED_IDS:
        listed += f", and {len(task_ids) - _MAX_LISTED_IDS} more"
    return f"Completed data-source calls: {listed}."


async def _validate_collector_output(text: str, run: AgentRun) -> tuple[bool, dict | str]:
    """Check the deliverable object and that every cited task id belongs to a completed data-source call.

    Args:
        text: The raw output text from the collector agent run.
        run: The collector agent run whose tool calls are checked.

    Returns:
        An ok flag and a value, the parsed deliverable dict when ok, else the message for every problem found.
    """
    ok, parsed = parse_json_object(text)
    if not ok:
        return False, parsed
    problems: list[str] = []
    if "summary" not in parsed:
        problems.append("Missing field: summary.")
    else:
        ok, error = check_nonblank_string(
            parsed, "summary", "'summary' must be a non-empty string."
        )
        if not ok:
            problems.append(error)
    id_items: list[str] = []
    if "evidences" not in parsed:
        problems.append("Missing field: evidences.")
    elif not isinstance(parsed["evidences"], list):
        problems.append("'evidences' must be a list.")
    else:
        for index, item in enumerate(parsed["evidences"]):
            if not isinstance(item, str) or not item.strip():
                problems.append(f"Citation {index} is not a task id string.")
            else:
                id_items.append(item)
    call_ids = [
        call.task_id
        for call in await run.tool_calls()
        if call.status == "completed" and call.tool_name in DATA_SOURCE_TOOLS
    ]
    unknown_ids = [task_id for task_id in id_items if task_id not in call_ids]
    if unknown_ids:
        problems.append(
            f"No completed data-source call of this run has task id: {', '.join(unknown_ids)}. "
            f"{_describe_completed_calls(call_ids)}"
        )
    elif not call_ids:
        problems.append("No data-source tool call completed during the run.")
    if problems:
        return False, combine_problems(problems)
    return True, parsed


async def _run_collector(
    goal: str,
    task: ResearchTask,
    *,
    tools: tuple[Tool, ...],
) -> CollectorResult:
    """Run the collector role and return its summary and the task ids of the evidence it cites.

    Args:
        goal: The research goal the task serves.
        task: The collection task to gather raw data for.
        tools: The data-source tools wired into the collector agent.

    Returns:
        The agent run with its summary and cited task ids. An empty summary
        and an empty list of ids are returned when the agent run fails.

    Raises:
        asyncio.CancelledError: When the pipeline is cancelled while the
            collector awaits its agent task. The agent task is cancelled first,
            and the whole task tree is awaited before the cancellation
            re-raises.
    """
    tool_descs = render_tool_descriptions(tools)
    system_prompt = await load_prompt("collector_prompt", tool_descriptions=tool_descs)
    user_message = f"Research goal: {goal}\n\nYour task: {task.description}"
    agent = Agent(
        name=f"research_collector_{task.id}",
        system_prompt=system_prompt,
        validator=_validate_collector_output,
        tool_results_by_id=True,
    )
    for tool in tools:
        agent.add_tool_adapter(tool)
    agent_run = await agent.run(user_message)
    try:
        result = await agent_run.task.result()
        deliverable = json.loads(result)
        summary = deliverable["summary"]
        citation_ids = deliverable["evidences"]
    except asyncio.CancelledError:
        await asyncio.shield(agent_run.task.cancel())
        raise
    except Exception as exc:
        print(f"[research] collector {task.id} agent run failed: {exc}", file=sys.stderr, flush=True)
        summary = ""
        citation_ids = []
    return CollectorResult(agent_run=agent_run, summary=summary, citation_ids=citation_ids)


@dataclass
class CollectionLedger:
    """What one research run collected, and which runs it may cite."""

    records: list[TaskState] = field(default_factory=list)
    registry: dict[str, CitedRun] = field(default_factory=dict)


def format_records(records: list[TaskState], *, should_include_instruction: bool = True) -> str:
    """Lay out collection records as text, one block per record.

    Args:
        records: The collection records to show.
        should_include_instruction: Whether each block repeats the record's instruction after its id.

    Returns:
        The blocks, each with the id, the instruction when included, and the summary and runs,
        or a no evidence line when no run was ingested.
    """
    blocks: list[str] = []
    for record in records:
        header = f"[{record.id}]"
        lines = [f"{header} Instruction: {record.description}" if should_include_instruction else header]
        if record.evidence:
            lines.append(f"Summary: {record.summary}")
            lines.extend(f"- run_id={e.run_id} ({e.tool_name})" for e in record.evidence)
        else:
            lines.append("No evidence collected.")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def _ingest_collector_results(
    collector_run: CollectorResult,
    record: TaskState,
    store: EvidenceStore,
    registry: dict[str, CitedRun],
) -> None:
    """Store and register the cited, completed data-source calls of a collector run.

    Args:
        collector_run: The finished collector run.
        record: The collection record to fill with the summary and the runs.
        store: The evidence store the run text is written to.
        registry: The citable runs, by run id.
    """
    calls = await collector_run.agent_run.tool_calls()
    ingested_tools: dict[str, str] = {}
    for call in calls:
        if call.tool_name not in DATA_SOURCE_TOOLS:
            continue
        if call.status != "completed":
            continue
        if call.task_id not in collector_run.citation_ids:
            continue
        # Calls that reused a stored run all name that one run.
        if call.task_id in ingested_tools:
            continue
        ingested_tools[call.task_id] = call.tool_name
        run_path = os.path.join(store_module.task_store.data_dir, call.run_dir)
        registry[call.task_id] = CitedRun(
            run_id=call.task_id,
            tool_name=call.tool_name,
            params=call.params,
            run_dir=run_path,
        )
        try:
            result_text = await read_run_mcp_text(run_path)
        except Exception as exc:
            print(f"[research] ingest: failed to read run {call.task_id}: {exc}", file=sys.stderr, flush=True)
            continue
        call_params_json = json.dumps(call.params, sort_keys=True)
        await store.write(result_text, tool_name=call.tool_name, source_task_id=call.task_id, call_params=call_params_json)

    record.summary = collector_run.summary
    for run_id in collector_run.citation_ids:
        tool_name = ingested_tools.get(run_id)
        if tool_name is not None:
            record.evidence.append(EvidenceItem(run_id=run_id, tool_name=tool_name))


class CollectEvidence(Tool):
    """In-loop tool that runs collectors for one research run and stores what they find."""

    name: ClassVar[str] = "collect_evidence"
    staleness: ClassVar[timedelta | None] = None
    tracked: ClassVar[bool] = True
    result_render_type: ClassVar[str] = "markdown"
    description: ClassVar[str] = (
        "Run up to three collectors in parallel, one per task, and add the evidence they find "
        "to the evidence store. Use it when the evidence store lacks what a part of the goal needs. "
        "Collectors can reach general web search and page reading, X, GitHub, and Xiaohongshu."
    )
    summary: ClassVar[str] = "Collect evidence with parallel collectors"
    params: ClassVar[tuple[Param, ...]] = (
        Param(
            "tasks",
            list[str],
            description=(
                "One to three collection instructions. Each entry is one self-contained "
                "instruction that one collector carries out on its own."
            ),
        ),
    )

    def __init__(self, goal: str, store: EvidenceStore, ledger: CollectionLedger) -> None:
        """Bind the tool to one research run.

        Args:
            goal: The research goal every collector serves.
            store: The run's evidence store.
            ledger: The run's collection ledger.
        """
        self._goal = goal
        self._store = store
        self._ledger = ledger
        self._slots = asyncio.Semaphore(_MAX_PARALLEL)

    @override
    async def call(self, params: dict) -> ToolResult:
        """Run one collector per task and return what each one found.

        Args:
            params: The call arguments.

        Returns:
            One block per task with its id, summary, and runs.

        Raises:
            ToolMessageError: When the task list is empty, too long, or has a blank entry.
        """
        tasks = params["tasks"]
        if not 1 <= len(tasks) <= _MAX_PARALLEL:
            raise ToolMessageError(f"tasks must hold 1 to {_MAX_PARALLEL} instructions, got {len(tasks)}.")
        if any(not task.strip() for task in tasks):
            raise ToolMessageError("Every entry in tasks must be a non-empty instruction.")
        first_number = len(self._ledger.records) + 1
        records = [
            TaskState(id=f"c{first_number + offset:03d}", description=task)
            for offset, task in enumerate(tasks)
        ]
        self._ledger.records.extend(records)
        await asyncio.gather(*(self._collect(record) for record in records))
        return ToolResult(text=format_records(records, should_include_instruction=False))

    async def _collect(self, record: TaskState) -> None:
        async with self._slots:
            task = ResearchTask(id=record.id, description=record.description)
            try:
                collector_run = await _run_collector(self._goal, task, tools=_COLLECTOR_TOOLS)
            except Exception as exc:
                print(f"[research] collector {record.id} failed: {exc}", file=sys.stderr, flush=True)
                return
            await _ingest_collector_results(collector_run, record, self._store, self._ledger.registry)
