from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.research.collector import CollectEvidence, CollectionLedger
from app.agents.research.evidence import SearchEvidence
from app.agents.research.models import CitedRun, ResearchResult, Review
from app.agents.research.runner import run_research

_EMBEDDING_URL = "http://embed.test:8888"
_APPROVE = Review(passed=True, reason="Good", action="approve", feedback="", iteration=1)
_REPORT = "# Report\n\nFact [[ref:run-1]]."


@contextmanager
def _stubbed_roles(
    researcher: AsyncMock, reviews: list[Review]
) -> Iterator[MagicMock]:
    """Stub the researcher, the reviewer, and the evidence store, and yield the store stub."""
    with (
        patch("app.agents.research.runner.run_researcher", researcher),
        patch("app.agents.research.runner.run_reviewer", new_callable=AsyncMock, side_effect=reviews),
        patch("app.agents.research.runner.EvidenceStore") as mock_store_cls,
    ):
        store_instance = mock_store_cls.return_value
        store_instance.open = AsyncMock()
        store_instance.close = AsyncMock()
        yield store_instance


def _register_run_1(_goal: str, ledger: CollectionLedger, **_kw: dict) -> str:
    """Stand in for a researcher that collected run-1 and cited it."""
    ledger.registry["run-1"] = CitedRun(
        run_id="run-1", tool_name="web_fetch", params={"url": "https://x.test/a"}, run_dir="runs/run-1"
    )
    return _REPORT


class TestRunResearchPrecondition:
    async def test_absent_folder_created(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDING_SERVER_URL", _EMBEDDING_URL)
        monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "1")
        run_folder = tmp_path / "new_run"
        with (
            patch("app.agents.research.runner.run_researcher", new_callable=AsyncMock, return_value="# Report"),
            patch("app.agents.research.runner.run_reviewer", new_callable=AsyncMock, return_value=_APPROVE),
        ):
            await run_research("goal", run_folder)
        assert run_folder.is_dir()

    async def test_nonempty_folder_raises(self, tmp_path: Path) -> None:
        run_folder = tmp_path / "existing"
        run_folder.mkdir()
        (run_folder / "evidence.lance").mkdir()
        with pytest.raises(ValueError, match="evidence.lance"):
            await run_research("goal", run_folder)

    async def test_stray_file_raises(self, tmp_path: Path) -> None:
        run_folder = tmp_path / "stray"
        run_folder.mkdir()
        (run_folder / "stray.txt").write_text("x")
        with pytest.raises(ValueError, match="stray.txt"):
            await run_research("goal", run_folder)

    @pytest.mark.parametrize("value", ["0", "-1"])
    async def test_max_iterations_below_one_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", value)
        run_folder = tmp_path / "run"
        with pytest.raises(ValueError, match="RESEARCH_MAX_ITERATIONS"):
            await run_research("goal", run_folder)
        assert not run_folder.exists()

    async def test_empty_max_iterations_runs_five_rounds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "")
        rewrite = Review(passed=False, reason="Thin", action="rewrite", feedback="Expand", iteration=1)
        researcher = AsyncMock(return_value="# Report")
        with _stubbed_roles(researcher, [rewrite] * 5):
            result = await run_research("goal", tmp_path / "run")
        assert (result.status, result.iterations, result.review_reason) == ("max_iterations", 5, "Thin")
        assert researcher.await_count == 5


class TestRunLoop:
    async def test_approve_on_first_pass_references_the_ledger_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "1")
        researcher = AsyncMock(side_effect=_register_run_1)
        with _stubbed_roles(researcher, [_APPROVE]) as store_instance:
            result = await run_research("goal", tmp_path / "run")
        assert isinstance(result, ResearchResult)
        assert (result.status, result.iterations, result.review_reason) == ("complete", 1, "Good")
        assert "[[ref:" not in result.report
        assert "[[1]](https://x.test/a)" in result.report
        assert "## References" in result.report
        store_instance.close.assert_awaited()

    async def test_researcher_gets_the_research_tools(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "1")
        researcher = AsyncMock(return_value=_REPORT)
        with _stubbed_roles(researcher, [_APPROVE]):
            await run_research("goal", tmp_path / "run")
        tool_names = [tool.name for tool in researcher.await_args.kwargs["tools"]]
        assert tool_names == ["collect_evidence", "search_evidence", "task", "read_media_or_doc"]

    @pytest.mark.parametrize("action", ["collect", "rewrite"])
    async def test_failed_review_starts_a_new_researcher_run_with_the_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
    ) -> None:
        monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "2")
        draft_1 = "# Report\n\nFirst draft."
        failed = Review(passed=False, reason="Gaps", action=action, feedback="Cover supply", iteration=1)
        approve = Review(passed=True, reason="Good", action="approve", feedback="", iteration=2)
        researcher = AsyncMock(side_effect=[draft_1, "# Report\n\nSecond draft."])
        with _stubbed_roles(researcher, [failed, approve]):
            result = await run_research("goal", tmp_path / "run")
        assert (result.status, result.iterations) == ("complete", 2)
        first, second = researcher.await_args_list
        assert (first.kwargs["previous_draft"], first.kwargs["review"]) == (None, None)
        assert (second.kwargs["previous_draft"], second.kwargs["review"]) == (draft_1, failed)
        assert second.args[1] is first.args[1]
        first_collect = first.kwargs["tools"][0]
        assert isinstance(first_collect, CollectEvidence)
        assert second.kwargs["tools"][0] is first_collect
        assert isinstance(second.kwargs["tools"][1], SearchEvidence)

    async def test_researcher_failure_fails_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "1")
        researcher = AsyncMock(side_effect=RuntimeError("agent validation failed after 10 attempts"))
        with _stubbed_roles(researcher, [_APPROVE]) as store_instance:
            with pytest.raises(RuntimeError, match="agent validation failed after 10 attempts"):
                await run_research("goal", tmp_path / "run")
        store_instance.close.assert_awaited()


@pytest.mark.live
class TestLiveResearch:
    @pytest.mark.timeout(900)
    async def test_end_to_end(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "1")
        run_folder = tmp_path / "live_run"
        result = await run_research("What is the current version of Python?", run_folder)
        assert result.report
        assert len(result.report) > 100
        entries = list(run_folder.iterdir())
        assert len(entries) == 1
        assert entries[0].name == "evidence.lance"
        assert "[[ref:" not in result.report
