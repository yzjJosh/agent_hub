import json
from unittest.mock import MagicMock

import pytest

import app.agents.research.reviewer as reviewer_module
from app.agents.research.reviewer import run_reviewer
from tests.app.agents.research.conftest import (
    CannedCall,
    CannedEvent,
    FakeTool,
    assert_cancels_agent_task,
    patch_agent_run,
    patch_runner,
    patch_wrap_up_runner,
    record_user_messages,
    search_result_for,
    stub_result,
    text_event,
    write_search_result,
)

ENGINE_PREFIX = "agent validation failed after 10 attempts: "
_EXPECTED_MAX_TURNS = 70
_EXPECTED_WRAP_UP_TURNS_REMAINING = 5
_STAGE_ONE_TURNS = _EXPECTED_MAX_TURNS - _EXPECTED_WRAP_UP_TURNS_REMAINING
_MAX_ATTEMPTS = 10


class TestReviewerRun:
    async def test_engine_stubbed_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        valid = json.dumps({"pass": True, "reason": "All backed", "action": "approve", "feedback": ""})
        patch_runner(monkeypatch, stub_result([text_event(valid)], valid, "reviewer"))

        draft = "# Draft\n\nGPU prices rose 15% this year per the market data."
        review = await run_reviewer("GPU market analysis", draft, tools=(FakeTool(),))
        assert review.passed is True
        assert review.action == "approve"
        assert review.reason == "All backed"
        assert review.iteration == 1

    async def test_user_message_is_goal_and_draft_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        valid = json.dumps({"pass": False, "reason": "Missing data", "action": "collect", "feedback": "Search Y"})
        patch_runner(monkeypatch, stub_result([text_event(valid)], valid, "reviewer"))

        messages = record_user_messages(monkeypatch)

        draft = "# Draft\n\nPrices rose 15% this year."
        review = await run_reviewer("GPU market analysis", draft, tools=(FakeTool(),), iteration=2)
        message = messages[0]
        assert "Research goal: GPU market analysis" in message
        assert "DRAFT TO REVIEW (iteration 2)" in message
        assert "Prices rose 15% this year." in message
        assert "COLLECTION TASKS" not in message
        assert "Search for GPU prices" not in message
        assert review.passed is False
        assert review.action == "collect"
        assert review.iteration == 2

    async def _run_capturing_agent_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        valid = json.dumps({"pass": True, "reason": "All backed", "action": "approve", "feedback": ""})
        patch_runner(monkeypatch, stub_result([text_event(valid)], valid, "reviewer"))
        agent_spy = MagicMock(wraps=reviewer_module.Agent)
        monkeypatch.setattr(reviewer_module, "Agent", agent_spy)
        review = await run_reviewer("GPU market analysis", "# Draft", tools=(FakeTool(),))
        assert review.passed is True
        return agent_spy.call_args.kwargs

    async def test_agent_passes_output_json_schema_and_validator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kwargs = await self._run_capturing_agent_kwargs(monkeypatch)
        assert kwargs["output_json_schema"] == {
            "type": "object",
            "properties": {
                "pass": {"type": "boolean"},
                "reason": {"type": "string"},
                "action": {"type": "string", "enum": ["approve", "collect", "rewrite"]},
                "feedback": {"type": "string"},
            },
            "required": ["pass", "reason", "action", "feedback"],
            "additionalProperties": False,
        }
        assert callable(kwargs["validator"])

    async def test_agent_configures_turn_limits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kwargs = await self._run_capturing_agent_kwargs(monkeypatch)
        assert kwargs["max_turns"] == _EXPECTED_MAX_TURNS
        assert kwargs["wrap_up_turns_remaining"] == _EXPECTED_WRAP_UP_TURNS_REMAINING


class TestReviewerTurnLimit:
    async def test_stage_two_wrap_up_returns_fail_verdict_without_verified_refs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        verdict = json.dumps(
            {
                "pass": False,
                "reason": "Turn limit reached before verifying citations",
                "action": "collect",
                "feedback": "Verify run-1",
            }
        )
        calls = patch_wrap_up_runner(
            monkeypatch,
            stage_one_turns=_STAGE_ONE_TURNS,
            wrap_up_turns=_EXPECTED_WRAP_UP_TURNS_REMAINING,
            final_output=verdict,
            role="reviewer",
        )
        patch_agent_run(monkeypatch)

        draft = "# Draft\n\nPrices rose 15% [[ref:run-1]]."
        review = await run_reviewer("GPU market analysis", draft, tools=(FakeTool(),))

        assert review.passed is False
        assert review.action == "collect"
        assert review.reason == "Turn limit reached before verifying citations"
        assert review.feedback == "Verify run-1"
        assert calls.max_turns == [_STAGE_ONE_TURNS, _EXPECTED_WRAP_UP_TURNS_REMAINING, 1]
        assert f"{_EXPECTED_WRAP_UP_TURNS_REMAINING} tool-enabled turns remaining" in calls.inputs[1][-1]["content"]
        assert "tools are now disabled" in calls.inputs[2][-1]["content"]

    async def test_stage_two_exhaustion_with_unverified_pass_verdict_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        verdict = json.dumps({"pass": True, "reason": "All backed", "action": "approve", "feedback": ""})
        calls = patch_wrap_up_runner(
            monkeypatch,
            stage_one_turns=_STAGE_ONE_TURNS,
            wrap_up_turns=_EXPECTED_WRAP_UP_TURNS_REMAINING,
            final_output=verdict,
            role="reviewer",
        )
        patch_agent_run(monkeypatch)

        draft = "# Draft\n\nPrices rose 15% [[ref:run-1]]."
        with pytest.raises(RuntimeError, match="run-1"):
            await run_reviewer("GPU market analysis", draft, tools=(FakeTool(),))

        assert calls.max_turns == [_STAGE_ONE_TURNS, _EXPECTED_WRAP_UP_TURNS_REMAINING] + [1] * _MAX_ATTEMPTS
        assert "tools are now disabled" in calls.inputs[2][-1]["content"]


class TestReviewerValidation:
    async def test_markdown_wrapped_output_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        wrapped = '```json\n{"pass": true, "reason": "ok", "action": "approve", "feedback": ""}\n```'
        patch_runner(monkeypatch, stub_result([text_event(wrapped)], wrapped, "reviewer"))
        review = await run_reviewer("GPU market analysis", "# Draft", tools=(FakeTool(),))
        assert review.passed is True
        assert review.action == "approve"

    async def test_pass_verdict_without_task_results_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        valid = json.dumps({"pass": True, "reason": "All backed", "action": "approve", "feedback": ""})
        patch_runner(monkeypatch, stub_result([text_event(valid)], valid, "reviewer"))
        patch_agent_run(monkeypatch)

        draft = "# Draft\n\nPrices rose 15% [[ref:run-1]] and supply tightened [[ref:run-2]]."
        with pytest.raises(RuntimeError, match="run-1, run-2"):
            await run_reviewer("GPU market analysis", draft, tools=(FakeTool(),))

    async def test_pass_verdict_with_all_refs_read_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        valid = json.dumps({"pass": True, "reason": "All backed", "action": "approve", "feedback": ""})
        patch_runner(monkeypatch, stub_result([text_event(valid)], valid, "reviewer"))
        patch_agent_run(
            monkeypatch,
            events=(
                CannedEvent(type="tool_call", tool_name="task", params={"action": "result", "task_id": "run-1"}),
                CannedEvent(type="tool_call", tool_name="task", params={"action": "result", "task_id": "run-2"}),
            ),
        )

        draft = "# Draft\n\nPrices rose 15% [[ref:run-1]] and supply tightened [[ref:run-2]]."
        review = await run_reviewer("GPU market analysis", draft, tools=(FakeTool(),))
        assert review.passed is True
        assert review.action == "approve"

    async def test_pass_verdict_with_missing_ref_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        valid = json.dumps({"pass": True, "reason": "All backed", "action": "approve", "feedback": ""})
        patch_runner(monkeypatch, stub_result([text_event(valid)], valid, "reviewer"))
        patch_agent_run(
            monkeypatch,
            events=(
                CannedEvent(type="tool_call", tool_name="task", params={"action": "result", "task_id": "run-1"}),
            ),
        )

        draft = "# Draft\n\nPrices rose 15% [[ref:run-1]] and supply tightened [[ref:run-2]]."
        with pytest.raises(RuntimeError, match="run-2"):
            await run_reviewer("GPU market analysis", draft, tools=(FakeTool(),))

    async def test_pass_verdict_passes_when_refs_covered_by_read_and_search_union(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        valid = json.dumps({"pass": True, "reason": "All backed", "action": "approve", "feedback": ""})
        patch_runner(monkeypatch, stub_result([text_event(valid)], valid, "reviewer"))
        patch_agent_run(
            monkeypatch,
            calls=(CannedCall(tool_name="search_evidence", params={"query": "prices"}, run_dir="runs/search-1"),),
            events=(
                CannedEvent(type="tool_call", tool_name="task", params={"action": "result", "task_id": "run-1"}),
            ),
        )
        write_search_result("runs/search-1", search_result_for("run-2"))

        draft = "# Draft\n\nPrices rose 15% [[ref:run-1]] and supply tightened [[ref:run-2]]."
        review = await run_reviewer("GPU market analysis", draft, tools=(FakeTool(),))
        assert review.passed is True
        assert review.action == "approve"

    async def test_pass_verdict_passes_when_refs_covered_by_search_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        valid = json.dumps({"pass": True, "reason": "All backed", "action": "approve", "feedback": ""})
        patch_runner(monkeypatch, stub_result([text_event(valid)], valid, "reviewer"))
        patch_agent_run(
            monkeypatch,
            calls=(CannedCall(tool_name="search_evidence", params={"query": "prices"}, run_dir="runs/search-1"),),
        )
        write_search_result("runs/search-1", search_result_for("run-1") + search_result_for("run-2"))

        draft = "# Draft\n\nPrices rose 15% [[ref:run-1]] and supply tightened [[ref:run-2]]."
        review = await run_reviewer("GPU market analysis", draft, tools=(FakeTool(),))
        assert review.passed is True
        assert review.action == "approve"

    async def test_pass_verdict_fails_when_search_misses_a_ref(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        valid = json.dumps({"pass": True, "reason": "All backed", "action": "approve", "feedback": ""})
        patch_runner(monkeypatch, stub_result([text_event(valid)], valid, "reviewer"))
        patch_agent_run(
            monkeypatch,
            calls=(CannedCall(tool_name="search_evidence", params={"query": "prices"}, run_dir="runs/search-1"),),
            events=(
                CannedEvent(type="tool_call", tool_name="task", params={"action": "result", "task_id": "run-1"}),
            ),
        )
        write_search_result("runs/search-1", search_result_for("run-3"))

        draft = "# Draft\n\nPrices rose 15% [[ref:run-1]] and supply tightened [[ref:run-2]]."
        with pytest.raises(RuntimeError) as exc_info:
            await run_reviewer("GPU market analysis", draft, tools=(FakeTool(),))
        assert (
            "Verdict is pass but these references were not verified "
            "(not read with task(action='result') and not surfaced in a search_evidence result): run-2."
        ) in str(exc_info.value)

    async def test_failed_search_call_contributes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        valid = json.dumps({"pass": True, "reason": "All backed", "action": "approve", "feedback": ""})
        patch_runner(monkeypatch, stub_result([text_event(valid)], valid, "reviewer"))
        patch_agent_run(
            monkeypatch,
            calls=(
                CannedCall(
                    tool_name="search_evidence",
                    params={"query": "prices"},
                    status="failed",
                    run_dir="runs/search-1",
                ),
            ),
        )
        write_search_result("runs/search-1", search_result_for("run-2"))

        draft = "# Draft\n\nPrices rose 15% [[ref:run-1]] and supply tightened [[ref:run-2]]."
        with pytest.raises(RuntimeError, match="run-1, run-2"):
            await run_reviewer("GPU market analysis", draft, tools=(FakeTool(),))

    async def test_missing_search_result_file_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        valid = json.dumps({"pass": True, "reason": "All backed", "action": "approve", "feedback": ""})
        patch_runner(monkeypatch, stub_result([text_event(valid)], valid, "reviewer"))
        patch_agent_run(
            monkeypatch,
            calls=(CannedCall(tool_name="search_evidence", params={"query": "prices"}, run_dir="runs/search-1"),),
            events=(
                CannedEvent(type="tool_call", tool_name="task", params={"action": "result", "task_id": "run-1"}),
            ),
        )

        draft = "# Draft\n\nPrices rose 15% [[ref:run-1]] and supply tightened [[ref:run-2]]."
        with pytest.raises(RuntimeError, match="run-2"):
            await run_reviewer("GPU market analysis", draft, tools=(FakeTool(),))

    async def test_fail_verdict_needs_no_task_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        valid = json.dumps({"pass": False, "reason": "Weak backing", "action": "rewrite", "feedback": "Add data"})
        patch_runner(monkeypatch, stub_result([text_event(valid)], valid, "reviewer"))
        patch_agent_run(monkeypatch)

        draft = "# Draft\n\nPrices rose 15% [[ref:run-1]]."
        review = await run_reviewer("GPU market analysis", draft, tools=(FakeTool(),))
        assert review.passed is False
        assert review.action == "rewrite"
        assert review.feedback == "Add data"


class TestReviewerBatchedErrors:
    @pytest.mark.parametrize(
        "output,expected",
        [
            (
                {"pass": True},
                "Multiple validation problems found. "
                "Missing field: reason. Missing field: action. Missing field: feedback.",
            ),
            (
                {"pass": 1, "reason": "   ", "action": "reject", "feedback": "Add data"},
                "Multiple validation problems found. "
                "'pass' must be a boolean. 'reason' must be a non-empty string. "
                "'action' must be one of ['approve', 'collect', 'rewrite'].",
            ),
            (
                {"pass": True, "reason": "All backed", "action": "approve"},
                "Missing field: feedback.",
            ),
        ],
    )
    async def test_problems_are_reported_together(
        self, monkeypatch: pytest.MonkeyPatch, output: dict, expected: str
    ) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        text = json.dumps(output)
        patch_runner(monkeypatch, stub_result([text_event(text)], text, "reviewer"))
        patch_agent_run(monkeypatch)

        draft = "# Draft\n\nPrices rose 15% [[ref:run-1]]."
        with pytest.raises(RuntimeError) as exc_info:
            await run_reviewer("GPU market analysis", draft, tools=(FakeTool(),))
        message = str(exc_info.value)
        assert message.startswith(ENGINE_PREFIX + expected)
        assert "not verified" not in message


class TestReviewerCancellation:
    async def test_cancellation_cancels_agent_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REASON_THINKING_BACKEND_URL", "http://backend.test/")
        await assert_cancels_agent_task(
            monkeypatch,
            lambda: run_reviewer("Goal", "# Draft", tools=(FakeTool(),)),
            "reviewer",
        )
