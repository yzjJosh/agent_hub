"""Tests for the browser use role prompt and its final-answer check."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from app.agents.browser_use.prompts.role import build_role_prompt, validate_role_output
from app.lib.agent.engine import AgentRun
from app.lib.tool.base import Param, Tool, ToolResult


class StubTool(Tool):
    """The stub tool the prompt test renders."""

    name: ClassVar[str] = "stub"
    description: ClassVar[str] = "A stub tool."
    summary: ClassVar[str] = "A stub tool"
    params: ClassVar[tuple[Param, ...]] = (
        Param("target_id", str, description="The tab to work on."),
    )

    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="")


def _run() -> AgentRun:
    """The run handle the validator takes."""
    return MagicMock(spec=AgentRun)


class TestValidateRoleOutput:
    def test_valid_json_passes(self) -> None:
        ok, value = validate_role_output('{"status": "completed", "summary": "done"}', _run())
        assert ok is True
        assert value == {"status": "completed", "summary": "done"}

    def test_missing_summary_fails(self) -> None:
        ok, value = validate_role_output('{"status": "completed"}', _run())
        assert ok is False
        assert value == "'summary' must be a non-empty string."

    def test_bad_status_fails(self) -> None:
        ok, value = validate_role_output('{"status": "done", "summary": "x"}', _run())
        assert ok is False
        assert value == "'status' must be one of completed, impossible, incomplete."

    def test_blank_summary_fails(self) -> None:
        ok, value = validate_role_output('{"status": "impossible", "summary": "  "}', _run())
        assert ok is False
        assert value == "'summary' must be a non-empty string."

    def test_non_object_fails(self) -> None:
        ok, value = validate_role_output('["completed"]', _run())
        assert ok is False
        assert value == "Output must be a JSON object."

    def test_unparsable_text_fails_with_the_parse_message(self) -> None:
        ok, value = validate_role_output("no json here", _run())
        assert ok is False
        assert value == (
            "Output must be the pure JSON value only, the JSON array or object itself, "
            "with no extra text, no markdown wrapper, and no commentary."
        )

    def test_code_fenced_json_passes(self) -> None:
        text = '```json\n{"status": "impossible", "summary": "cannot"}\n```'
        ok, value = validate_role_output(text, _run())
        assert ok is True
        assert value == {"status": "impossible", "summary": "cannot"}

    def test_leading_filler_passes(self) -> None:
        text = 'Here is my answer:\n{"status": "incomplete", "summary": "stuck"}'
        ok, value = validate_role_output(text, _run())
        assert ok is True
        assert value == {"status": "incomplete", "summary": "stuck"}


class TestBuildRolePrompt:
    def test_prompt_names_every_tool(self) -> None:
        prompt = build_role_prompt((StubTool(), StubTool()))
        assert "stub" in prompt
        assert "target_id" in prompt

    @pytest.mark.parametrize("token", ["completed", "impossible", "incomplete", "target id"])
    def test_prompt_covers_the_final_form(self, token: str) -> None:
        prompt = build_role_prompt((StubTool(),))
        assert token in prompt

    def test_prompt_names_the_target_id_contract(self) -> None:
        prompt = build_role_prompt((StubTool(),))
        assert "no tabs open when you start" in prompt
        assert "open_url with no target id" in prompt
        assert "leave the target id out and the action runs on that tab" in prompt
        assert "give the target id of the tab to act on" in prompt
        assert "the first token on the tab's line in the step result" in prompt

    def test_prompt_tells_the_role_to_read_content_by_task(self) -> None:
        prompt = build_role_prompt((StubTool(),))
        assert "read_media_or_doc" in prompt
        assert "returns a task id, not the content" in prompt
        assert "call task with action result and that task id" in prompt
        assert "page it with offset and max_char" in prompt

    def test_prompt_covers_the_signin_instructions(self) -> None:
        prompt = build_role_prompt((StubTool(),))
        assert "If the goal carries its own sign-in instructions, follow those." in prompt
        assert "may already be signed in to some sites" in prompt
        assert "Give a sign-in page a few seconds before you type into it yourself." in prompt
        assert "When a sign-in asks for a two-factor code" in prompt
        assert "a service that receives such codes, like an email inbox or a messaging site" in prompt
        assert "if that service is already signed in in this browser" in prompt
        assert "Open it in a new tab and wait for the code to arrive." in prompt
        assert "for any two-factor step you cannot complete" in prompt
        assert "status impossible" in prompt
        assert "the sign-in needs a person." in prompt

    @pytest.mark.parametrize("install_claim", ["Gmail", "Google Messages", "extension"])
    def test_prompt_claims_nothing_about_the_install(self, install_claim: str) -> None:
        prompt = build_role_prompt((StubTool(),))
        assert install_claim not in prompt

    def test_prompt_warns_that_visual_and_dom_history_is_trimmed(self) -> None:
        prompt = build_role_prompt((StubTool(),))
        assert "latest action snapshot's screenshots and the latest DOM body for each open tab" in prompt
        assert "Old dump bodies become checkpoint metadata" in prompt
        assert "newest observation's current tab lines and status remain" in prompt

    def test_prompt_explains_historical_retrieval_and_retention(self) -> None:
        prompt = build_role_prompt((StubTool(),))
        assert "checkpoint with action list" in prompt
        assert "action read with checkpoint_id and target_id" in prompt
        assert "before repeating navigation or dump_page" in prompt
        assert "stay in subsequent model context for the rest of this run" in prompt
        assert "Historical checkpoint evidence describes an earlier observation" in prompt
        assert "Do not repeat an executed action to repair" in prompt
        assert "never silently dropped to fit the provider context limit" in prompt
