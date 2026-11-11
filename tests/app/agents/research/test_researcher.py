import re
from collections.abc import AsyncIterator

import pytest
from agents.stream_events import RawResponsesStreamEvent

from app.agents.research.collector import CollectionLedger
from app.agents.research.models import CitedRun, EvidenceItem, Review, TaskState
from app.agents.research.prompt import REVIEW_FEEDBACK_HEADER
from app.agents.research.researcher import run_researcher
from tests.app.agents.research.conftest import (
    CannedCall,
    CannedEvent,
    FakeTool,
    assert_cancels_agent_task,
    patch_agent_run,
    patch_runner,
    record_user_messages,
    search_result_for,
    stub_result,
    text_event,
    write_search_result,
)

_GOAL = "GPU market analysis"
_BACKEND = "http://backend.test/"
_UNVERIFIED = (
    "Report cites references that were not verified "
    "(not read with task(action='result') and not surfaced in a search_evidence result)"
)


def _ledger(*run_ids: str) -> CollectionLedger:
    """Build a ledger with one collection record that ingested the given runs."""
    record = TaskState(
        id="c001",
        description="Search for GPU prices",
        summary="Two sources report the current GPU prices.",
        evidence=[EvidenceItem(run_id=run_id, tool_name="brave_search") for run_id in run_ids],
    )
    registry = {
        run_id: CitedRun(run_id=run_id, tool_name="brave_search", params={}, run_dir=f"runs/{run_id}")
        for run_id in run_ids
    }
    return CollectionLedger(records=[record], registry=registry)


def _read_event(run_id: str) -> CannedEvent:
    """Build a task result read of the given run."""
    return CannedEvent(type="tool_call", tool_name="task", params={"action": "result", "task_id": run_id})


def _stub_report(
    monkeypatch: pytest.MonkeyPatch,
    report: str,
    *,
    calls: tuple[CannedCall, ...] = (),
    events: tuple[CannedEvent, ...] = (),
) -> None:
    """Make the researcher agent end with the given report over canned calls and events."""
    monkeypatch.setenv("REASON_THINKING_BACKEND_URL", _BACKEND)
    patch_runner(monkeypatch, stub_result([text_event(report)], report, "researcher"))
    patch_agent_run(monkeypatch, calls=calls, events=events)


async def _failure_message(
    monkeypatch: pytest.MonkeyPatch,
    report: str,
    ledger: CollectionLedger,
    *,
    events: tuple[CannedEvent, ...] = (),
    previous_draft: str | None = None,
) -> str:
    """Run the researcher over a report the validator rejects, and return the failure text."""
    _stub_report(monkeypatch, report, events=events)
    with pytest.raises(RuntimeError) as failure:
        await run_researcher(_GOAL, ledger, tools=(FakeTool(),), previous_draft=previous_draft)
    return str(failure.value)


class TestResearcherRun:
    async def test_verified_report_is_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        report = "# Report\n\nGPU prices rose 15% this year [[ref:run-1]]."
        _stub_report(monkeypatch, report, events=(_read_event("run-1"),))
        assert await run_researcher(_GOAL, _ledger("run-1"), tools=(FakeTool(),)) == report

    async def test_first_iteration_message_is_the_goal_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        report = "# Report\n\n[INSUFFICIENT DATA] No evidence was found for this research goal."
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", _BACKEND)
        messages = record_user_messages(monkeypatch)
        ledger = CollectionLedger()

        async def _collect_then_report() -> AsyncIterator[RawResponsesStreamEvent]:
            ledger.records.append(TaskState(id="c001", description="Search for GPU prices"))
            yield text_event(report)

        stub = stub_result([], report, "researcher")
        stub.stream_events.side_effect = _collect_then_report
        patch_runner(monkeypatch, stub)
        await run_researcher(_GOAL, ledger, tools=(FakeTool(),))
        assert messages == [f"Research goal: {_GOAL}"]

    async def test_later_iteration_message_has_collection_feedback_and_draft(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = "# Report\n\nPrices rose 15% this year overall [[ref:run-1]]."
        _stub_report(monkeypatch, report)
        messages = record_user_messages(monkeypatch)
        review = Review(passed=False, reason="Thin", action="collect", feedback="Find supply data", iteration=1)
        previous = "# Earlier draft\n\nA prior version cites [[ref:run-1]]."
        await run_researcher(
            _GOAL, _ledger("run-1"), tools=(FakeTool(),), previous_draft=previous, review=review
        )
        assert messages[0] == (
            f"Research goal: {_GOAL}\n"
            "\n"
            "### COLLECTION SO FAR\n"
            "[c001] Instruction: Search for GPU prices\n"
            "Summary: Two sources report the current GPU prices.\n"
            "- run_id=run-1 (brave_search)\n"
            "\n"
            f"{REVIEW_FEEDBACK_HEADER}\n"
            "Reviewer action: collect\n"
            "Find supply data\n"
            "\n"
            "### PREVIOUS DRAFT\n"
            f"{previous}"
        )

    async def test_empty_record_is_shown_as_no_evidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        report = "# Report\n\n[INSUFFICIENT DATA] Nothing was collected for this research goal."
        _stub_report(monkeypatch, report)
        messages = record_user_messages(monkeypatch)
        review = Review(passed=False, reason="Thin", action="rewrite", feedback="Rephrase", iteration=1)
        ledger = CollectionLedger(records=[TaskState(id="c001", description="Search for GPU prices")])
        await run_researcher(_GOAL, ledger, tools=(FakeTool(),), previous_draft=report, review=review)
        assert "[c001] Instruction: Search for GPU prices\nNo evidence collected." in messages[0]
        assert "Summary:" not in messages[0]


class TestResearcherCollectionHappened:
    async def test_report_without_any_collection_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = await _failure_message(
            monkeypatch,
            "# Report\n\n[INSUFFICIENT DATA] The market grew steadily across all segments.",
            CollectionLedger(),
        )
        assert "Call collect_evidence before writing the report." in message

    async def test_collection_with_nothing_ingested_allows_a_refless_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = "# Report\n\n[INSUFFICIENT DATA] Nothing was collected for this research goal."
        _stub_report(monkeypatch, report)
        ledger = CollectionLedger(records=[TaskState(id="c001", description="Search for GPU prices")])
        assert await run_researcher(_GOAL, ledger, tools=(FakeTool(),)) == report

    async def test_earlier_iteration_records_satisfy_the_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        report = "# Report\n\nPrices rose 15% this year overall [[ref:run-1]]."
        _stub_report(monkeypatch, report)
        review = Review(passed=False, reason="Style", action="rewrite", feedback="Tighten", iteration=1)
        result = await run_researcher(
            _GOAL, _ledger("run-1"), tools=(FakeTool(),), previous_draft=report, review=review
        )
        assert result == report


class TestResearcherLiveCitableSet:
    async def test_run_ingested_during_the_run_is_citable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", _BACKEND)
        report = "# Report\n\nGPU prices rose 15% this year [[ref:run-1]]."
        ledger = CollectionLedger()
        filled = _ledger("run-1")

        async def _collect_then_report() -> AsyncIterator[RawResponsesStreamEvent]:
            ledger.records.extend(filled.records)
            ledger.registry.update(filled.registry)
            yield text_event(report)

        stub = stub_result([], report, "researcher")
        stub.stream_events.side_effect = _collect_then_report
        patch_runner(monkeypatch, stub)
        patch_agent_run(monkeypatch, events=(_read_event("run-1"),))
        assert await run_researcher(_GOAL, ledger, tools=(FakeTool(),)) == report


class TestResearcherNormalization:
    @pytest.mark.parametrize(
        "output,expected",
        [
            (
                "Here is the report.\n# Report\n\nThe market grew steadily this year across all segments.",
                "# Report\n\nThe market grew steadily this year across all segments.",
            ),
            (
                "```markdown\n# Report\n\nThe market grew steadily this year across all segments.\n```",
                "# Report\n\nThe market grew steadily this year across all segments.\n",
            ),
        ],
    )
    async def test_returns_only_normalized_body(
        self, monkeypatch: pytest.MonkeyPatch, output: str, expected: str
    ) -> None:
        _stub_report(monkeypatch, output)
        assert await run_researcher(_GOAL, _ledger(), tools=(FakeTool(),)) == expected

    async def test_stripped_preamble_does_not_satisfy_minimum_length(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        filler = "This filler line is deliberately much longer than the minimum report length."
        message = await _failure_message(monkeypatch, f"{filler}\n# R\n\nShort.", _ledger())
        assert "Report is too short" in message

    async def test_stripped_preamble_reference_does_not_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        output = (
            "The requested source is [[ref:run-1]].\n"
            "# Report\n\nThe market grew steadily this year across all product segments."
        )
        message = await _failure_message(monkeypatch, output, _ledger("run-1"))
        assert "must contain at least one" in message

    async def test_malformed_wrapper_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        output = "```Markdown\n# Report\n\nThe market grew steadily this year across all product segments.\n```"
        message = await _failure_message(monkeypatch, output, _ledger())
        assert "wrapper" in message


class TestResearcherCoverage:
    async def test_unverified_known_ref_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = await _failure_message(
            monkeypatch, "# Report\n\nPrices rose 15% this year [[ref:run-1]].", _ledger("run-1")
        )
        assert f"{_UNVERIFIED}: run-1." in message

    async def test_refs_covered_by_read_and_search_union_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        report = "# Report\n\nPrices rose 15% [[ref:run-1]] and supply tightened [[ref:run-2]]."
        _stub_report(
            monkeypatch, report,
            calls=(CannedCall(tool_name="search_evidence", params={"query": "prices"}, run_dir="runs/search-1"),),
            events=(_read_event("run-1"),),
        )
        write_search_result("runs/search-1", search_result_for("run-2"))
        assert await run_researcher(_GOAL, _ledger("run-1", "run-2"), tools=(FakeTool(),)) == report

    async def test_failed_search_call_contributes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_report(
            monkeypatch, "# Report\n\nPrices rose 15% this year [[ref:run-1]].",
            calls=(
                CannedCall(
                    tool_name="search_evidence", params={"query": "prices"},
                    status="failed", run_dir="runs/search-1",
                ),
            ),
        )
        write_search_result("runs/search-1", search_result_for("run-1"))
        with pytest.raises(RuntimeError, match=f"{re.escape(_UNVERIFIED)}: run-1."):
            await run_researcher(_GOAL, _ledger("run-1"), tools=(FakeTool(),))

    async def test_unknown_ref_fails_even_when_verified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = await _failure_message(
            monkeypatch, "# Report\n\nPrices rose 15% this year [[ref:run-9]].", _ledger("run-1"),
            events=(_read_event("run-9"),),
        )
        assert "Unknown run id in placeholder: run-9." in message


class TestResearcherPriorDraftRefs:
    async def test_prior_draft_ref_passes_without_reread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        report = "# Report\n\nPrices rose 15% this year [[ref:run-1]]."
        _stub_report(monkeypatch, report)
        result = await run_researcher(_GOAL, _ledger("run-1"), tools=(FakeTool(),), previous_draft=report)
        assert result == report

    async def test_new_unverified_ref_fails_on_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = await _failure_message(
            monkeypatch, "# Report\n\nPrices rose 15% this year [[ref:run-2]].", _ledger("run-1", "run-2"),
            previous_draft="# Report\n\nOld [[ref:run-1]].",
        )
        assert f"{_UNVERIFIED}: run-2." in message
        assert f"{_UNVERIFIED}: run-1" not in message

    async def test_prior_draft_ref_must_still_be_known(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = await _failure_message(
            monkeypatch, "# Report\n\nPrices rose 15% this year [[ref:run-3]].", _ledger("run-1"),
            previous_draft="# Report\n\nOld [[ref:run-3]].",
        )
        assert "Unknown run id in placeholder: run-3." in message


class TestResearcherBatchedErrors:
    async def test_short_draft_with_unknown_ref_reports_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = await _failure_message(monkeypatch, "# R\nBrief report [[ref:run-9]].", _ledger("run-1"))
        assert "Multiple validation problems found." in message
        assert "Report is too short (" in message
        assert "Unknown run id in placeholder: run-9." in message

    async def test_short_draft_without_ref_reports_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = await _failure_message(monkeypatch, "# R\nPrices rose last year.", _ledger("run-1"))
        assert "Multiple validation problems found." in message
        assert "Report is too short (" in message
        assert (
            "Report must contain at least one [[ref:<run_id>]] placeholder when evidence is available."
        ) in message

    async def test_short_draft_with_ref_when_no_evidence_reports_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = await _failure_message(monkeypatch, "# R\nPrices rose [[ref:run-1]].", _ledger())
        assert "Multiple validation problems found." in message
        assert "Report is too short (" in message
        assert "Unknown run id in placeholder: run-1." in message
        assert "Report must not contain placeholders when no evidence is available." in message

    async def test_no_collection_and_short_draft_report_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = await _failure_message(monkeypatch, "# R\nShort.", CollectionLedger())
        assert "Multiple validation problems found." in message
        assert "Call collect_evidence before writing the report." in message
        assert "Report is too short (" in message

    async def test_unknown_ref_not_repeated_as_unverified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        message = await _failure_message(
            monkeypatch,
            "# Report\n\nPrices rose 15% this year [[ref:run-1]] "
            "and supply tightened sharply [[ref:run-9]] across markets.",
            _ledger("run-1"),
        )
        assert "Unknown run id in placeholder: run-9." in message
        assert f"{_UNVERIFIED}: run-1." in message
        assert f"{_UNVERIFIED}: run-9." not in message


class TestResearcherTitle:
    _TITLE_PROBLEM: str = "Report must start with a markdown title (a first line starting with '# ')."

    async def test_h1_title_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        report = "# My Title\n\nThe market grew steadily this year across all segments."
        _stub_report(monkeypatch, report)
        assert await run_researcher(_GOAL, _ledger(), tools=(FakeTool(),)) == report

    @pytest.mark.parametrize(
        "report",
        [
            "Here is the report: the market grew steadily this year across all segments.",
            "## Sub\n\nThe market grew steadily this year across all segments.",
            "#  Title\n\nThe market grew steadily this year across all segments.",
        ],
    )
    async def test_missing_h1_title_is_rejected(self, monkeypatch: pytest.MonkeyPatch, report: str) -> None:
        message = await _failure_message(monkeypatch, report, _ledger())
        assert self._TITLE_PROBLEM in message


class TestResearcherCancellation:
    async def test_cancellation_cancels_agent_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", _BACKEND)
        await assert_cancels_agent_task(
            monkeypatch, lambda: run_researcher(_GOAL, _ledger(), tools=(FakeTool(),)), "researcher"
        )
