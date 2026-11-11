from collections.abc import Sequence
from unittest.mock import MagicMock

from app.agents.research.coverage import unverified_refs
from tests.app.agents.research.conftest import (
    CannedCall,
    CannedEvent,
    CannedRun,
    search_result_for,
    write_search_result,
)


def _run(
    calls: Sequence[CannedCall] = (),
    events: Sequence[CannedEvent] = (),
) -> CannedRun:
    return CannedRun(MagicMock(), tuple(calls), tuple(events))


def _task_event(task_id: str, action: str = "result") -> CannedEvent:
    return CannedEvent(type="tool_call", tool_name="task", params={"action": action, "task_id": task_id})


def _search_call(status: str = "completed", run_dir: str = "runs/search-1") -> CannedCall:
    return CannedCall(tool_name="search_evidence", params={"query": "prices"}, status=status, run_dir=run_dir)


class TestUnverifiedRefs:
    async def test_no_refs_returns_empty(self) -> None:
        assert await unverified_refs("# Report\n\nNo references here.", _run()) == []

    async def test_unverified_refs_returned_in_draft_order(self) -> None:
        draft = "# Report\n\nPrices [[ref:run-2]] then supply [[ref:run-1]] again [[ref:run-2]]."
        assert await unverified_refs(draft, _run()) == ["run-2", "run-1"]

    async def test_task_result_read_verifies_the_ref(self) -> None:
        draft = "# Report\n\nPrices rose 15% this year [[ref:run-1]]."
        assert await unverified_refs(draft, _run(events=(_task_event("run-1"),))) == []

    async def test_task_call_without_result_action_does_not_verify(self) -> None:
        draft = "# Report\n\nPrices rose 15% this year [[ref:run-1]]."
        assert await unverified_refs(draft, _run(events=(_task_event("run-1", action="logs"),))) == ["run-1"]

    async def test_non_tool_call_event_does_not_verify(self) -> None:
        draft = "# Report\n\nPrices rose 15% this year [[ref:run-1]]."
        events = (CannedEvent(type="text", tool_name=None, params={"task_id": "run-1"}),)
        assert await unverified_refs(draft, _run(events=events)) == ["run-1"]

    async def test_completed_search_hit_verifies_the_ref(self) -> None:
        write_search_result("runs/search-1", search_result_for("run-1"))
        draft = "# Report\n\nPrices rose 15% this year [[ref:run-1]]."
        assert await unverified_refs(draft, _run(calls=(_search_call(),))) == []

    async def test_failed_search_call_does_not_verify(self) -> None:
        write_search_result("runs/search-1", search_result_for("run-1"))
        draft = "# Report\n\nPrices rose 15% this year [[ref:run-1]]."
        assert await unverified_refs(draft, _run(calls=(_search_call(status="failed"),))) == ["run-1"]

    async def test_missing_search_result_file_is_skipped(self) -> None:
        draft = "# Report\n\nPrices rose 15% this year [[ref:run-1]]."
        assert await unverified_refs(draft, _run(calls=(_search_call(),))) == ["run-1"]

    async def test_read_and_search_union_covers_different_refs(self) -> None:
        write_search_result("runs/search-1", search_result_for("run-2"))
        draft = "# Report\n\nPrices rose 15% [[ref:run-1]] and supply tightened [[ref:run-2]]."
        run = _run(calls=(_search_call(),), events=(_task_event("run-1"),))
        assert await unverified_refs(draft, run) == []

    async def test_search_misses_one_ref(self) -> None:
        write_search_result("runs/search-1", search_result_for("run-3"))
        draft = "# Report\n\nPrices rose 15% [[ref:run-1]] and supply tightened [[ref:run-2]]."
        assert await unverified_refs(draft, _run(calls=(_search_call(),))) == ["run-1", "run-2"]
