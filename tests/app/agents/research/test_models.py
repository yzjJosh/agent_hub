from app.agents.research.models import (
    EvidenceItem,
    ResearchResult,
    ResearchTask,
    Review,
    CitedRun,
    TaskState,
)


class TestDataclasses:
    def test_research_task(self) -> None:
        t = ResearchTask(id="task_001", description="Search for X")
        assert t.id == "task_001"
        assert t.description == "Search for X"

    def test_evidence_item(self) -> None:
        e = EvidenceItem(run_id="abc-123", tool_name="web_fetch")
        assert e.run_id == "abc-123"
        assert e.tool_name == "web_fetch"

    def test_task_state(self) -> None:
        e = EvidenceItem(run_id="abc", tool_name="brave_search")
        t = TaskState(id="i1_task_001", description="desc", evidence=[e], summary="Collected X")
        assert t.evidence == [EvidenceItem(run_id="abc", tool_name="brave_search")]
        assert t.summary == "Collected X"

    def test_task_state_default_summary(self) -> None:
        t = TaskState(id="i1_task_001", description="desc")
        assert t.summary == ""

    def test_cited_run(self) -> None:
        r = CitedRun(run_id="abc", tool_name="web_fetch", params={"url": "https://x.com"}, run_dir="runs/abc")
        assert r.run_id == "abc"
        assert r.tool_name == "web_fetch"

    def test_review(self) -> None:
        r = Review(passed=False, reason="gaps", action="collect", feedback="need more", iteration=1)
        assert r.passed is False
        assert r.action == "collect"

    def test_research_result(self) -> None:
        r = ResearchResult(report="# Report", status="complete", iterations=2, review_reason="looks good")
        assert r.report == "# Report"
        assert r.status == "complete"
        assert r.iterations == 2
