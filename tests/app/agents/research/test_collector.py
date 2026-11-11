import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from agents import Agent as SdkAgent
from agents import Usage
from agents.items import ToolCallItem, ToolCallOutputItem
from agents.stream_events import RunItemStreamEvent
from agents.tool_context import ToolContext

import app.agents.research.collector as collector_module
import app.lib.agent.engine as engine
import app.lib.db.store as store_module
from app.agents.research.collector import CollectEvidence, CollectionLedger
from app.agents.research.models import CitedRun, EvidenceItem
from app.lib.agent.engine import AgentRun
from app.lib.task.task_manager import Task
from app.lib.tool.base import Tool, ToolMessageError, ToolResult
from app.tools.brave_search import brave_search_tool
from app.tools.rednote_research import rednote_research_tool
from tests.app.agents.research.conftest import (
    CannedCall,
    CannedRun,
    assert_cancels_agent_task,
    patch_agent_run,
    patch_runner,
    stub_result,
    text_event,
    track_agent_runs,
    write_search_result,
)

_SUMMARY = "Found recent GPU prices."
_CITED_ID = "call-1"
_CITATIONS = json.dumps({"summary": _SUMMARY, "evidences": [_CITED_ID]})
_GPU_PRICES_CALL = CannedCall(
    task_id=_CITED_ID, tool_name="brave_search", params={"query": "gpu prices"}
)
_SEARCH_ARGS = '{"query":"gpu prices"}'
_SEARCH_TEXT = "results for gpu prices"
_COLLECT_GOAL = "GPU market"
_INSTRUCTION = "Search for GPU prices"
_RUN_TEXT = "## Results\n1. **GPU prices**\n   URL: https://x.test/a"
_MAX_PARALLEL = 3
_POLL = 0.01


def _deliverable(evidences: list, summary: str = _SUMMARY) -> str:
    """Build the collector deliverable for the given evidence task ids."""
    return json.dumps({"summary": summary, "evidences": evidences})


@dataclass
class _Collected:
    """What one collector of a collect_evidence call left behind."""

    summary: str
    citation_ids: list[str]
    task: Task


def _recording_store() -> MagicMock:
    """Build an evidence store stand-in that records every write."""
    store = MagicMock()
    store.write = AsyncMock()
    return store


async def _collect_single(monkeypatch: pytest.MonkeyPatch, tasks: list[Task]) -> _Collected:
    """Run collect_evidence with one task, and return what its collector left behind."""
    monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
    ledger = CollectionLedger()
    await CollectEvidence(_COLLECT_GOAL, _recording_store(), ledger).call({"tasks": [_INSTRUCTION]})
    record = ledger.records[0]
    return _Collected(record.summary, [e.run_id for e in record.evidence], tasks[0])


def _track_canned_runs(monkeypatch: pytest.MonkeyPatch, calls: tuple[CannedCall, ...]) -> list[Task]:
    """Point the engine's AgentRun at a canned run, and record each agent task."""
    tasks: list[Task] = []

    def _canned(task: Task) -> CannedRun:
        tasks.append(task)
        return CannedRun(task, calls, ())

    monkeypatch.setattr(engine, "AgentRun", _canned)
    return tasks


async def _collect_one(
    monkeypatch: pytest.MonkeyPatch,
    valid: str,
    calls: tuple[CannedCall, ...] = (),
) -> _Collected:
    """Run one collector over a stubbed agent run with the given canned calls."""
    patch_runner(monkeypatch, stub_result([text_event(valid)], valid, "collector"))
    tasks = _track_canned_runs(monkeypatch, calls)
    return await _collect_single(monkeypatch, tasks)


async def _failure_message(
    monkeypatch: pytest.MonkeyPatch,
    output: str,
    calls: tuple[CannedCall, ...] = (),
) -> str:
    """Run the collector over an output the validator rejects, and return the failure text."""
    result = await _collect_one(monkeypatch, output, calls=calls)
    assert result.summary == ""
    assert result.citation_ids == []
    with pytest.raises(RuntimeError) as failure:
        await result.task.result()
    return str(failure.value)


def _patch_call_runner(
    monkeypatch: pytest.MonkeyPatch,
    tool: Tool,
    *,
    seen: dict[str, str] | None = None,
    citations: str = _deliverable([]),
) -> list[Task]:
    """Make every collector attempt run one call of the tool, and record what the model got back.

    Args:
        monkeypatch: The patcher that redirects the engine's runner and the tool.
        tool: The data-source tool the attempt calls, answered without the network.
        seen: A dict the tool call output is stored in, under the key output.
        citations: The deliverable object the attempt ends with.

    Returns:
        The agent tasks the collectors spawn.
    """
    async def _answer(params: dict) -> ToolResult:
        return ToolResult(text=f"results for {params['query']}")

    monkeypatch.setattr(tool, "call", _answer)

    def _stub(sdk_agent: SdkAgent, _input: str | list) -> MagicMock:
        stub = MagicMock()
        stub.final_output = citations
        stub.to_input_list = MagicMock(return_value=[{"role": "user", "content": "collector"}])
        stub.cancel = MagicMock()

        async def _stream() -> AsyncIterator[RunItemStreamEvent]:
            function_tool = next(t for t in sdk_agent.tools if t.name == tool.name)
            call_item = ToolCallItem(
                agent=sdk_agent,
                raw_item={"name": tool.name, "call_id": "C1", "arguments": _SEARCH_ARGS},
            )
            yield RunItemStreamEvent(item=call_item, name="tool_called")
            context = ToolContext(
                context=None, usage=Usage(), tool_name=tool.name,
                tool_call_id="C1", tool_arguments=_SEARCH_ARGS,
            )
            output = await function_tool.on_invoke_tool(context, _SEARCH_ARGS)
            if seen is not None:
                seen["output"] = output
            output_item = ToolCallOutputItem(
                agent=sdk_agent, raw_item={"call_id": "C1"}, output=output
            )
            yield RunItemStreamEvent(item=output_item, name="tool_output")

        stub.stream_events = MagicMock(side_effect=_stream)
        return stub

    monkeypatch.setattr(engine.Runner, "run_streamed", lambda a, i, **_kw: _stub(a, i))
    return track_agent_runs(monkeypatch)


class TestCollectorResult:
    @pytest.mark.parametrize(
        "output",
        [
            "Here is the requested result.\n" + _CITATIONS,
            "```json\n" + _CITATIONS + "\n```",
        ],
    )
    async def test_normalized_result_is_returned(
        self, monkeypatch: pytest.MonkeyPatch, output: str
    ) -> None:
        result = await _collect_one(monkeypatch, output, calls=(_GPU_PRICES_CALL,))
        assert result.summary == _SUMMARY
        assert result.citation_ids == [_CITED_ID]
        assert result.task.status.value == "completed"


class TestCollectorValidation:
    async def test_valid_object_passes_with_summary_and_evidences(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = (
            _GPU_PRICES_CALL,
            CannedCall(task_id="call-2", tool_name="web_fetch", params={"url": "https://x.test/a"}),
        )
        result = await _collect_one(
            monkeypatch, _deliverable(["call-2", _CITED_ID]), calls=calls
        )
        assert result.summary == _SUMMARY
        assert result.citation_ids == ["call-2", _CITED_ID]
        assert result.task.status.value == "completed"

    @pytest.mark.parametrize(
        "output",
        [
            json.dumps([_CITED_ID]),
            '"a bare string"',
        ],
    )
    async def test_non_object_output_fails(
        self, monkeypatch: pytest.MonkeyPatch, output: str
    ) -> None:
        message = await _failure_message(monkeypatch, output, calls=(_GPU_PRICES_CALL,))
        assert "Output must be a JSON object." in message

    async def test_missing_summary_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = await _failure_message(
            monkeypatch, json.dumps({"evidences": [_CITED_ID]}), calls=(_GPU_PRICES_CALL,)
        )
        assert "Missing field: summary." in message

    @pytest.mark.parametrize("summary", ["", "   \n\t"])
    async def test_blank_summary_fails(
        self, monkeypatch: pytest.MonkeyPatch, summary: str
    ) -> None:
        message = await _failure_message(
            monkeypatch, _deliverable([_CITED_ID], summary=summary), calls=(_GPU_PRICES_CALL,)
        )
        assert "'summary' must be a non-empty string." in message

    async def test_non_string_summary_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = await _failure_message(
            monkeypatch, json.dumps({"summary": 7, "evidences": [_CITED_ID]}),
            calls=(_GPU_PRICES_CALL,),
        )
        assert "'summary' must be a non-empty string." in message

    async def test_missing_evidences_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = await _failure_message(
            monkeypatch, json.dumps({"summary": _SUMMARY}), calls=(_GPU_PRICES_CALL,)
        )
        assert "Missing field: evidences." in message

    async def test_evidences_that_is_not_a_list_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = await _failure_message(
            monkeypatch, json.dumps({"summary": _SUMMARY, "evidences": _CITED_ID}),
            calls=(_GPU_PRICES_CALL,),
        )
        assert "'evidences' must be a list." in message

    @pytest.mark.parametrize(
        "evidences,expected",
        [
            ([7], "Citation 0 is not a task id string."),
            ([{"tool": "brave_search"}], "Citation 0 is not a task id string."),
            (["  "], "Citation 0 is not a task id string."),
            ([""], "Citation 0 is not a task id string."),
            ([_CITED_ID, 7], "Citation 1 is not a task id string."),
        ],
    )
    async def test_item_that_is_not_a_task_id_string_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, evidences: list, expected: str
    ) -> None:
        message = await _failure_message(
            monkeypatch, _deliverable(evidences), calls=(_GPU_PRICES_CALL,)
        )
        assert expected in message

    async def test_invented_task_id_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = await _collect_one(
            monkeypatch, _deliverable(["made-up-id"]), calls=(_GPU_PRICES_CALL,)
        )
        assert result.citation_ids == []
        assert result.task.status.value == "failed"

    async def test_task_id_of_another_agent_run_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        other_call = CannedCall(
            task_id="other-run-call", tool_name="brave_search", params={"query": "cpu prices"}
        )
        other_run = await _collect_one(
            monkeypatch, _deliverable(["other-run-call"]), calls=(other_call,)
        )
        assert other_run.citation_ids == ["other-run-call"]

        message = await _failure_message(
            monkeypatch, _deliverable(["other-run-call"]), calls=(_GPU_PRICES_CALL,)
        )
        assert "No completed data-source call of this run has task id: other-run-call." in message
        assert f"Completed data-source calls: {_CITED_ID}." in message

    async def test_task_id_of_a_call_outside_the_data_source_tools_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = (
            CannedCall(task_id="call-9", tool_name="read_media_or_doc", params={"path": "a.pdf"}),
        )
        message = await _failure_message(monkeypatch, _deliverable(["call-9"]), calls=calls)
        assert "No completed data-source call of this run has task id: call-9." in message
        assert "No data-source call completed." in message

    async def test_task_id_of_an_unfinished_call_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = (
            CannedCall(
                task_id="call-7", tool_name="brave_search",
                params={"query": "gpu prices"}, status="running",
            ),
        )
        message = await _failure_message(monkeypatch, _deliverable(["call-7"]), calls=calls)
        assert "No completed data-source call of this run has task id: call-7." in message
        assert "No data-source call completed." in message

    async def test_failure_message_names_every_rejected_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        message = await _failure_message(
            monkeypatch, _deliverable(["ghost-a", _CITED_ID, "ghost-b"]), calls=(_GPU_PRICES_CALL,)
        )
        assert (
            "No completed data-source call of this run has task id: ghost-a, ghost-b."
        ) in message
        assert f"Completed data-source calls: {_CITED_ID}." in message

    async def test_failure_message_caps_the_listed_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = tuple(
            CannedCall(task_id=f"call-{i:02d}", tool_name="brave_search", params={"query": f"q{i}"})
            for i in range(11)
        )
        message = await _failure_message(monkeypatch, _deliverable(["ghost"]), calls=calls)
        listed = ", ".join(f"call-{i:02d}" for i in range(10))
        assert f"Completed data-source calls: {listed}, and 1 more." in message

    async def test_no_data_source_call_fails_even_with_empty_evidences(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        message = await _failure_message(monkeypatch, _deliverable([]))
        assert "No data-source tool call completed during the run." in message

    async def test_empty_evidences_with_a_summary_passes_when_a_call_completed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = await _collect_one(monkeypatch, _deliverable([]), calls=(_GPU_PRICES_CALL,))
        assert result.summary == _SUMMARY
        assert result.citation_ids == []
        assert result.task.status.value == "completed"

    async def test_missing_summary_and_unknown_id_are_batched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        message = await _failure_message(
            monkeypatch, json.dumps({"evidences": ["ghost"]}), calls=(_GPU_PRICES_CALL,)
        )
        assert (
            "Multiple validation problems found. Missing field: summary. "
            "No completed data-source call of this run has task id: ghost. "
            f"Completed data-source calls: {_CITED_ID}."
        ) in message

    async def test_missing_summary_and_non_list_evidences_are_batched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        message = await _failure_message(
            monkeypatch, json.dumps({"evidences": _CITED_ID}), calls=(_GPU_PRICES_CALL,)
        )
        assert (
            "Multiple validation problems found. Missing field: summary. "
            "'evidences' must be a list."
        ) in message


class TestCollectorToolResults:
    async def test_data_tool_call_hands_the_collector_only_its_task_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, str] = {}
        tasks = _patch_call_runner(monkeypatch, brave_search_tool, seen=seen)
        result = await _collect_single(monkeypatch, tasks)
        assert result.citation_ids == []
        assert result.summary == _SUMMARY
        assert result.task.status.value == "completed"
        calls = await AgentRun(result.task).tool_calls()
        task_id = calls[0].task_id
        assert seen["output"] == (
            f"Task completed. task_id={task_id}. Retrieve the result with the task tool: "
            f"task(action='result', task_id='{task_id}')."
        )
        assert _SEARCH_TEXT not in seen["output"]


class TestCollectorCachedCalls:
    """A call that reused a stored run can be cited."""

    @staticmethod
    async def _collect(monkeypatch: pytest.MonkeyPatch, citations: str = _deliverable([])) -> _Collected:
        """Run one collector over one call of the cached data-source tool."""
        tasks = _patch_call_runner(monkeypatch, rednote_research_tool, citations=citations)
        return await _collect_single(monkeypatch, tasks)

    async def test_borrowed_task_id_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stored = await self._collect(monkeypatch)
        assert stored.task.status.value == "completed"
        stored_id = (await AgentRun(stored.task).tool_calls())[0].task_id

        reusing = await self._collect(monkeypatch, _deliverable([stored_id]))
        assert reusing.task.status.value == "completed"
        assert reusing.summary == _SUMMARY
        assert reusing.citation_ids == [stored_id]
        assert [c.task_id for c in await AgentRun(reusing.task).tool_calls()] == [stored_id]

    async def test_a_run_of_only_cached_calls_is_not_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        await self._collect(monkeypatch)
        reusing = await self._collect(monkeypatch)
        calls = await AgentRun(reusing.task).tool_calls()
        assert [(c.tool_name, c.status) for c in calls] == [(rednote_research_tool.name, "completed")]


class TestCollectorBackend:
    async def test_agent_uses_the_default_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent_spy = MagicMock(wraps=collector_module.Agent)
        monkeypatch.setattr(collector_module, "Agent", agent_spy)
        result = await _collect_one(monkeypatch, _CITATIONS, calls=(_GPU_PRICES_CALL,))
        assert agent_spy.call_args.kwargs.get("backend") is None
        assert result.task.status.value == "completed"


def _writes(store: MagicMock) -> list[dict]:
    """List the keyword arguments of every store write, with the text under the key text."""
    return [{"text": c.args[0], **c.kwargs} for c in store.write.await_args_list]


class _CollectorRuns:
    """Serves every collector agent attempt, and records how many run at once."""

    def __init__(self, deliverable: str, gate: asyncio.Event | None = None) -> None:
        self.deliverable = deliverable
        self.gate = gate
        self.names: list[str] = []
        self.active = 0
        self.peak = 0

    def stub(self, sdk_agent: SdkAgent, _input: str | list, **_kw: dict) -> MagicMock:
        async def _stream() -> AsyncIterator:
            self.names.append(sdk_agent.name)
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                if self.gate is not None:
                    await self.gate.wait()
            finally:
                self.active -= 1
            yield text_event(self.deliverable)

        return MagicMock(
            final_output=self.deliverable,
            stream_events=MagicMock(side_effect=_stream),
            to_input_list=MagicMock(return_value=[{"role": "user", "content": "collector"}]),
            cancel=MagicMock(),
        )


def _collect_tool(
    monkeypatch: pytest.MonkeyPatch,
    deliverable: str = _CITATIONS,
    *,
    calls: tuple[CannedCall, ...] = (_GPU_PRICES_CALL,),
    gate: asyncio.Event | None = None,
) -> tuple[CollectEvidence, CollectionLedger, MagicMock, _CollectorRuns]:
    """Build a collect_evidence tool whose collectors end with the given deliverable."""
    monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
    runs = _CollectorRuns(deliverable, gate)
    monkeypatch.setattr(engine.Runner, "run_streamed", runs.stub)
    patch_agent_run(monkeypatch, calls=calls)
    write_search_result(_GPU_PRICES_CALL.run_dir, _RUN_TEXT)
    ledger = CollectionLedger()
    store = _recording_store()
    return CollectEvidence(_COLLECT_GOAL, store, ledger), ledger, store, runs


async def _collect(tool: CollectEvidence, *tasks: str) -> str:
    """Make one collect_evidence call and return its text."""
    return (await tool.call({"tasks": list(tasks)})).text


async def _wait_active(runs: _CollectorRuns, count: int) -> None:
    """Wait until the given number of collectors run at once."""
    for _ in range(500):
        if runs.active == count:
            return
        await asyncio.sleep(_POLL)
    raise AssertionError(f"never saw {count} collectors running, saw {runs.active}")


class TestCollectEvidenceRejection:
    @pytest.mark.parametrize(
        "tasks,expected",
        [
            ([], "tasks must hold 1 to 3 instructions, got 0."),
            (["a", "b", "c", "d"], "tasks must hold 1 to 3 instructions, got 4."),
            (["Search for GPU prices", "  "], "Every entry in tasks must be a non-empty instruction."),
            ([""], "Every entry in tasks must be a non-empty instruction."),
        ],
    )
    async def test_bad_task_list_is_rejected_and_records_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tasks: list[str], expected: str
    ) -> None:
        tool, ledger, _store, runs = _collect_tool(monkeypatch)
        with pytest.raises(ToolMessageError) as failure:
            await tool.call({"tasks": tasks})
        assert str(failure.value) == expected
        assert ledger.records == []
        assert runs.names == []


class TestCollectEvidenceIds:
    async def test_ids_keep_counting_across_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tool, ledger, _store, runs = _collect_tool(monkeypatch)
        await _collect(tool, "Search for GPU prices", "Search for GPU supply")
        await _collect(tool, "Search for GPU demand")
        assert [(r.id, r.description) for r in ledger.records] == [
            ("c001", "Search for GPU prices"),
            ("c002", "Search for GPU supply"),
            ("c003", "Search for GPU demand"),
        ]
        assert sorted(runs.names) == [
            "research_collector_c001", "research_collector_c002", "research_collector_c003",
        ]

    async def test_concurrent_calls_get_disjoint_ids_and_share_the_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = asyncio.Event()
        tool, ledger, _store, runs = _collect_tool(monkeypatch, gate=gate)
        first = asyncio.create_task(_collect(tool, "a", "b", "c"))
        second = asyncio.create_task(_collect(tool, "d", "e"))
        await _wait_active(runs, _MAX_PARALLEL)
        await asyncio.sleep(_POLL * 10)
        assert runs.active == _MAX_PARALLEL
        assert len(runs.names) == _MAX_PARALLEL
        gate.set()
        first_text, second_text = await asyncio.gather(first, second)
        assert runs.peak == _MAX_PARALLEL
        assert [(r.id, r.description) for r in ledger.records] == [
            ("c001", "a"), ("c002", "b"), ("c003", "c"), ("c004", "d"), ("c005", "e"),
        ]
        assert sorted(runs.names) == [f"research_collector_c00{n}" for n in range(1, 6)]
        assert [line[:6] for line in first_text.splitlines() if line.startswith("[c")] == ["[c001]", "[c002]", "[c003]"]
        assert [line[:6] for line in second_text.splitlines() if line.startswith("[c")] == ["[c004]", "[c005]"]


class TestCollectEvidenceIngest:
    async def test_cited_run_is_stored_registered_and_shown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tool, ledger, store, _runs = _collect_tool(monkeypatch)
        text = await _collect(tool, "Search for GPU prices")
        assert _writes(store) == [{
            "text": _RUN_TEXT,
            "tool_name": "brave_search",
            "source_task_id": _CITED_ID,
            "call_params": json.dumps({"query": "gpu prices"}, sort_keys=True),
        }]
        assert ledger.registry == {
            _CITED_ID: CitedRun(
                run_id=_CITED_ID,
                tool_name="brave_search",
                params={"query": "gpu prices"},
                run_dir=os.path.join(store_module.task_store.data_dir, _GPU_PRICES_CALL.run_dir),
            ),
        }
        assert ledger.records[0].summary == _SUMMARY
        assert ledger.records[0].evidence == [EvidenceItem(run_id=_CITED_ID, tool_name="brave_search")]
        assert text == (
            "[c001]\n"
            f"Summary: {_SUMMARY}\n"
            f"- run_id={_CITED_ID} (brave_search)"
        )

    async def test_the_same_task_id_twice_is_written_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reused = CannedCall(task_id=_CITED_ID, tool_name="rednote_research", params={"query": "gpu prices"})
        tool, ledger, store, _runs = _collect_tool(monkeypatch, calls=(reused, reused))
        await _collect(tool, "Search for GPU prices")
        assert [w["source_task_id"] for w in _writes(store)] == [_CITED_ID]
        assert ledger.records[0].evidence == [EvidenceItem(run_id=_CITED_ID, tool_name="rednote_research")]

    async def test_unreadable_run_stays_citable_but_is_not_stored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unreadable = CannedCall(
            task_id=_CITED_ID, tool_name="brave_search", params={"query": "gpu prices"}, run_dir="runs/missing"
        )
        tool, ledger, store, _runs = _collect_tool(monkeypatch, calls=(unreadable,))
        await _collect(tool, "Search for GPU prices")
        assert _writes(store) == []
        assert list(ledger.registry) == [_CITED_ID]
        assert ledger.records[0].evidence == [EvidenceItem(run_id=_CITED_ID, tool_name="brave_search")]

    @pytest.mark.parametrize(
        "deliverable,expected_summary",
        [
            (_deliverable([]), _SUMMARY),
            ("not a deliverable", ""),
        ],
    )
    async def test_nothing_ingested_and_failed_collector_show_the_same_block(
        self, monkeypatch: pytest.MonkeyPatch, deliverable: str, expected_summary: str
    ) -> None:
        tool, ledger, store, _runs = _collect_tool(monkeypatch, deliverable)
        text = await _collect(tool, "Search for GPU prices")
        assert text == "[c001]\nNo evidence collected."
        assert _writes(store) == []
        assert ledger.registry == {}
        assert [(r.id, r.summary, r.evidence) for r in ledger.records] == [("c001", expected_summary, [])]

    async def test_call_outside_the_data_source_tools_is_not_ingested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        read_call = CannedCall(task_id="call-9", tool_name="read_media_or_doc", params={"path": "a.pdf"})
        tool, ledger, store, _runs = _collect_tool(
            monkeypatch, _deliverable([_CITED_ID]), calls=(_GPU_PRICES_CALL, read_call)
        )
        await _collect(tool, "Search for GPU prices")
        assert [w["source_task_id"] for w in _writes(store)] == [_CITED_ID]
        assert list(ledger.registry) == [_CITED_ID]


class TestCollectEvidenceCancellation:
    async def test_cancellation_cancels_the_collectors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        ledger = CollectionLedger()
        tool = CollectEvidence(_COLLECT_GOAL, _recording_store(), ledger)
        await assert_cancels_agent_task(
            monkeypatch, lambda: tool.call({"tasks": ["Search for GPU prices"]}), "collect_evidence"
        )
        assert [(r.id, r.summary, r.evidence) for r in ledger.records] == [("c001", "", [])]
