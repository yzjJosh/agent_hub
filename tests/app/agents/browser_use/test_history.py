"""Tests for browser history screenshot pruning and DOM dump compaction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from agents import Agent as SdkAgent, Runner
from agents.run_config import CallModelData, ModelInputData, RunConfig
from openai import BadRequestError

from app.agents.browser_use import history as history_module
from app.agents.browser_use import runner as runner_module
from app.lib.browser import browser_manager
from tests.app.agents.browser_use.conftest import (
    COMPLETED_OUTPUT_RAW,
    build_mock_stream,
    set_thinking_backend,
)

_OVERSIZED_BODY_REPEAT = 20000
_INLINE_IMAGE_REPEAT = 300000


def _build_function_call_item(call_id: str, name: str) -> dict:
    """Build a function call item sent by the model."""
    return {"type": "function_call", "name": name, "call_id": call_id, "arguments": "{}"}


def _build_function_output_item(call_id: str, output: str | list) -> dict:
    """Build a tool output item paired with a call id."""
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _build_image_part(url: str = "data:image/png;base64,AAAA") -> dict:
    """Build an image part for a tool output."""
    return {"type": "input_image", "image_url": url}


def _build_text_part(text: str) -> dict:
    """Build a text part for a tool output."""
    return {"type": "input_text", "text": text}


def _build_call_model_data(items: list[dict]) -> CallModelData:
    """Build model input data for direct history pruning calls."""
    return CallModelData(
        model_data=ModelInputData(input=items, instructions="the role prompt"),
        agent=object(),
        context=None,
    )


def _build_overflow_error() -> BadRequestError:
    request = httpx.Request("POST", "http://model.test")
    return BadRequestError(
        "overflow", response=httpx.Response(400, request=request), body={"code": "context_length_exceeded"}
    )


def _build_overflow_stream(
    history: list[dict],
    error: Exception | None,
    *,
    new_items: list[dict] | None = None,
) -> MagicMock:
    return build_mock_stream(
        history,
        error,
        final_output=COMPLETED_OUTPUT_RAW if error is None else None,
        new_items=new_items,
    )


@pytest.fixture
def disconnected_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a mock browser with no open pages."""
    browser = MagicMock(contexts=[MagicMock()], close=AsyncMock())
    playwright = MagicMock(stop=AsyncMock())
    playwright.chromium.connect_over_cdp = AsyncMock(return_value=browser)
    monkeypatch.setattr(
        browser_manager, "async_playwright", lambda: MagicMock(start=AsyncMock(return_value=playwright))
    )


class TestPruneOlderScreenshots:
    """The history filter keeps only the newest action-tool screenshots and current tab dumps."""

    async def test_older_action_screenshots_are_marked(self) -> None:
        items = [
            {"role": "user", "content": "start"},
            _build_function_call_item("c1", "click"),
            _build_function_output_item("c1", [_build_image_part(), _build_text_part("caption 1")]),
            _build_function_call_item("c2", "open_url"),
            _build_function_output_item(
                "c2", [_build_image_part("data:image/png;base64,BBBB"), _build_text_part("caption 2")]
            ),
            _build_function_call_item("c3", "click"),
            _build_function_output_item("c3", [_build_image_part("data:image/png;base64,CCCC")]),
        ]
        result = history_module.prune_model_history(_build_call_model_data(items))

        assert result.instructions == "the role prompt"
        assert len(result.input) == 7
        assert result.input[2]["output"] == [
            _build_text_part("[earlier screenshot omitted]"),
            _build_text_part("caption 1"),
        ]
        assert result.input[4]["output"] == [
            _build_text_part("[earlier screenshot omitted]"),
            _build_text_part("caption 2"),
        ]
        assert result.input[6]["output"] == [_build_image_part("data:image/png;base64,CCCC")]
        assert result.input[0] is items[0]
        assert result.input[1] is items[1]
        assert result.input[3] is items[3]
        assert result.input[5] is items[5]
        assert items[2]["output"] == [_build_image_part(), _build_text_part("caption 1")]
        assert items[4]["output"] == [
            _build_image_part("data:image/png;base64,BBBB"),
            _build_text_part("caption 2"),
        ]
        assert items[6]["output"] == [_build_image_part("data:image/png;base64,CCCC")]

    async def test_plain_string_output_does_not_become_latest(self) -> None:
        items = [
            _build_function_call_item("c1", "click"),
            _build_function_output_item("c1", [_build_image_part()]),
            _build_function_call_item("s1", "wait"),
            _build_function_output_item("s1", "plain text, no screenshot"),
            _build_function_call_item("c2", "click"),
            _build_function_output_item("c2", [_build_image_part("data:image/png;base64,BBBB")]),
        ]
        result = history_module.prune_model_history(_build_call_model_data(items))

        assert result.input[1]["output"] == [_build_text_part("[earlier screenshot omitted]")]
        assert result.input[3] is items[3]
        assert result.input[5]["output"] == [_build_image_part("data:image/png;base64,BBBB")]

    async def test_failed_step_screenshot_becomes_the_latest(self) -> None:
        failed_output = [
            _build_text_part('1. tab1 — "Shop" — https://shop.example — image 1\nstatus: failed, action failed: boom'),
            _build_image_part("data:image/png;base64,BBBB"),
        ]
        items = [
            _build_function_call_item("c1", "click"),
            _build_function_output_item("c1", [_build_text_part("caption 1"), _build_image_part()]),
            _build_function_call_item("c2", "click"),
            _build_function_output_item("c2", failed_output),
        ]
        result = history_module.prune_model_history(_build_call_model_data(items))

        assert result.input[1]["output"] == [
            _build_text_part("caption 1"),
            _build_text_part("[earlier screenshot omitted]"),
        ]
        assert result.input[3]["output"] == failed_output

    async def test_no_action_screenshots_returns_the_input_unchanged(self) -> None:
        items = [
            {"role": "user", "content": "start"},
            _build_function_call_item("c1", "click"),
            _build_function_output_item("c1", "no screenshot"),
            _build_function_call_item("c2", "open_url"),
            _build_function_output_item("c2", "also no screenshot"),
        ]
        result = history_module.prune_model_history(_build_call_model_data(items))

        assert result.instructions == "the role prompt"
        assert all(kept is original for kept, original in zip(result.input, items))

    async def test_single_marker_for_an_output_with_multiple_images(self) -> None:
        items = [
            _build_function_call_item("c1", "click"),
            _build_function_output_item(
                "c1",
                [
                    _build_image_part(),
                    _build_text_part("caption"),
                    _build_image_part("data:image/png;base64,BBBB"),
                ],
            ),
            _build_function_call_item("c2", "click"),
            _build_function_output_item("c2", [_build_image_part("data:image/png;base64,CCCC")]),
        ]
        result = history_module.prune_model_history(_build_call_model_data(items))

        assert result.input[1]["output"] == [
            _build_text_part("[earlier screenshot omitted]"),
            _build_text_part("caption"),
        ]
        assert result.input[3]["output"] == [_build_image_part("data:image/png;base64,CCCC")]

    @pytest.mark.parametrize("missing", ["name", "call_id"])
    async def test_function_call_missing_a_key_is_treated_as_non_action(self, missing: str) -> None:
        call_item = _build_function_call_item("c2", "click")
        del call_item[missing]
        items = [
            _build_function_call_item("c1", "click"),
            _build_function_output_item("c1", [_build_image_part()]),
            call_item,
            _build_function_output_item("c2", [_build_image_part("data:image/png;base64,BBBB")]),
        ]
        result = history_module.prune_model_history(_build_call_model_data(items))

        assert result.input[1]["output"] == [_build_image_part()]
        assert result.input[3]["output"] == [_build_image_part("data:image/png;base64,BBBB")]

    async def test_non_action_image_output_is_ignored(self) -> None:
        items = [
            _build_function_call_item("c1", "click"),
            _build_function_output_item("c1", [_build_image_part()]),
            _build_function_call_item("t1", "some_other_tool"),
            _build_function_output_item("t1", [_build_image_part("data:image/png;base64,BBBB")]),
        ]
        result = history_module.prune_model_history(_build_call_model_data(items))

        assert result.input[1]["output"] == [_build_image_part()]
        assert result.input[3]["output"] == [_build_image_part("data:image/png;base64,BBBB")]

    async def test_previous_dom_dump_keeps_its_call_and_checkpoint_stub(self) -> None:
        items = [
            _build_function_call_item("d1", "dump_page"),
            _build_function_output_item(
                "d1",
                "1. tab1 — First\nstatus: success\nCheckpoint: first\nObservation time: original\nDOM dump target: tab1\n\nEarlier content",
            ),
            _build_function_call_item("d2", "dump_page"),
            _build_function_output_item("d2", "1. tab1 — Second\nstatus: success\nDOM dump target: tab1\n\n# New"),
        ]

        result = history_module.prune_model_history(_build_call_model_data(items))

        assert [item["call_id"] for item in result.input] == ["d1", "d1", "d2", "d2"]
        assert "Earlier content" not in result.input[1]["output"]
        assert "Checkpoint: first" in result.input[1]["output"]
        assert "target_id tab1" in result.input[1]["output"]
        assert "status:" not in result.input[1]["output"]
        assert "# New" in result.input[3]["output"]

    async def test_closed_tab_dom_dump_keeps_metadata_without_current_envelope(self) -> None:
        items = [
            _build_function_call_item("d1", "dump_page"),
            _build_function_output_item(
                "d1", "1. tab1 — First\n2. tab2 — Second\nstatus: success\nDOM dump target: tab1\n\n# Gone"
            ),
            _build_function_call_item("d2", "dump_page"),
            _build_function_output_item(
                "d2", "1. tab1 — First\n2. tab2 — Second\nstatus: success\nDOM dump target: tab2\n\n# Kept"
            ),
            _build_function_call_item("c1", "close_page"),
            _build_function_output_item("c1", "1. tab2 — Second\nstatus: success"),
        ]

        result = history_module.prune_model_history(_build_call_model_data(items))

        assert [item["call_id"] for item in result.input] == ["d1", "d1", "d2", "d2", "c1", "c1"]
        assert "# Gone" not in result.input[1]["output"]
        assert "Archived DOM target: tab1" in result.input[1]["output"]
        assert "status:" not in result.input[1]["output"]
        assert "# Kept" in result.input[3]["output"]

    async def test_status_text_in_tab_title_does_not_hide_open_tabs(self) -> None:
        items = [
            _build_function_call_item("d1", "dump_page"),
            _build_function_output_item(
                "d1", "1. tab1 — Status: open\nstatus: success\nDOM dump target: tab1\n\n# Kept"
            ),
            _build_function_call_item("c1", "click"),
            _build_function_output_item("c1", "1. tab1 — status: open\nstatus: success"),
        ]

        result = history_module.prune_model_history(_build_call_model_data(items))

        assert [item["call_id"] for item in result.input] == ["d1", "d1", "c1", "c1"]

    @pytest.mark.parametrize("remaining_tab", [True, False])
    async def test_latest_dump_keeps_current_observation_when_target_closes(self, remaining_tab: bool) -> None:
        captions = "1. tab2 — Remaining\n" if remaining_tab else ""
        current_text = (
            f"Time: original\n{captions}status: success\nCheckpoint: newest\n"
            "Observation time: captured\nDOM capture interval: dom start to dom end\n"
            "DOM dump target: tab1\n\nClosed target DOM"
        )
        current_parts = [_build_text_part(current_text)]
        if remaining_tab:
            current_parts.append(_build_image_part("current-tab2"))
        historical_parts = [
            _build_text_part(
                "Historical checkpoint: old\n1. tab1 — Historical\nstatus: success\nDOM dump target: tab1\nOld evidence"
            ),
            _build_image_part("historical-tab1"),
        ]
        items = [
            _build_function_call_item("c1", "open_url"),
            _build_function_output_item("c1", [_build_text_part("1. tab1 — Old\nstatus: success"), _build_image_part("obsolete")]),
            _build_function_call_item("d1", "dump_page"),
            _build_function_output_item("d1", current_parts),
            _build_function_call_item("r1", "checkpoint"),
            _build_function_output_item("r1", historical_parts),
        ]
        result = history_module.prune_model_history(_build_call_model_data(items))
        assert "Closed target DOM" not in result.input[3]["output"][0]["text"]
        assert "status: success" in result.input[3]["output"][0]["text"]
        assert "Checkpoint: newest" in result.input[3]["output"][0]["text"]
        assert "Observation time: captured" in result.input[3]["output"][0]["text"]
        assert "DOM capture interval: dom start to dom end" in result.input[3]["output"][0]["text"]
        assert result.input[1]["output"][-1] == _build_text_part("[earlier screenshot omitted]")
        if remaining_tab:
            assert "1. tab2 — Remaining\nstatus:" in result.input[3]["output"][0]["text"]
            assert result.input[3]["output"][1] == _build_image_part("current-tab2")
        else:
            assert "1. tab1" not in result.input[3]["output"][0]["text"]
            assert len(result.input[3]["output"]) == 1
        assert result.input[5] == items[5]
        assert history_module.prune_model_history(_build_call_model_data(result.input)).input == result.input

    async def test_previously_current_pruned_dump_becomes_metadata_after_new_observation(self) -> None:
        items = [
            _build_function_call_item("d1", "dump_page"),
            _build_function_output_item(
                "d1",
                [
                    _build_text_part("1. tab2 — Remaining\nstatus: success\nCheckpoint: first\nDOM dump target: tab1\n\nClosed DOM"),
                    _build_image_part("was-current"),
                ],
            ),
        ]
        first = history_module.prune_model_history(_build_call_model_data(items)).input
        later = first + [
            _build_function_call_item("c2", "close_page"),
            _build_function_output_item("c2", "status: success\nCheckpoint: second"),
        ]
        result = history_module.prune_model_history(_build_call_model_data(later))
        assert isinstance(result.input[1]["output"], str)
        assert "Checkpoint: first" in result.input[1]["output"]
        assert "status:" not in result.input[1]["output"]
        assert "Remaining" not in result.input[1]["output"]
        assert result.input[3] == later[3]
        assert history_module.prune_model_history(_build_call_model_data(result.input)).input == result.input

    async def test_multiple_historical_views_survive_actions_dumps_closure_and_filter_retries(self) -> None:
        feedback = [
            _build_text_part("Historical checkpoint: feedback\nTab: tab1\nExact feedback\nDOM dump target: tab1"),
            _build_image_part("saved-feedback"),
        ]
        transcript = [
            _build_text_part("Historical checkpoint: transcript\nTab: tab1\nExact transcript\n1. injected — old\nstatus: success"),
            _build_image_part("saved-transcript"),
        ]
        items = [
            _build_function_call_item("r1", "checkpoint"),
            _build_function_output_item("r1", feedback),
            _build_function_call_item("r2", "checkpoint"),
            _build_function_output_item("r2", transcript),
            _build_function_call_item("d1", "dump_page"),
            _build_function_output_item(
                "d1", "1. tab1 — New\nstatus: success\nCheckpoint: later\nDOM dump target: tab1\n\nNew dump"
            ),
            _build_function_call_item("c1", "close_page"),
            _build_function_output_item("c1", "status: success"),
        ]
        result = history_module.prune_model_history(_build_call_model_data(items))
        assert result.input[:4] == items[:4]
        assert result.input[1] == items[1]
        assert result.input[3] == items[3]
        assert "New dump" not in result.input[5]["output"]
        assert "Checkpoint: later" in result.input[5]["output"]
        assert history_module.prune_model_history(_build_call_model_data(result.input)).input == result.input


class TestBrowserCompaction:
    """Overflow fallback archives page dumps in the browser agent."""

    @pytest.mark.usefixtures("disconnected_browser")
    @pytest.mark.parametrize("multipart", [False, True])
    @pytest.mark.parametrize("archive", [False, True])
    async def test_overflow_omits_open_tab_dumps_and_preserves_tool_history(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, multipart: bool, archive: bool
    ) -> None:
        """Both open tab bodies are archived while current and historical evidence survive."""
        set_thinking_backend(monkeypatch)
        checkpoint = "Checkpoint: newest\n" if archive else ""
        observation = "1. tab1 — Lens\n2. tab2 — Page\nstatus: success\n"
        latest = (
            observation
            + checkpoint
            + "Observation time: now\nDOM capture interval: start to end\nDOM dump target: tab2\n"
            + "BODY2" * _OVERSIZED_BODY_REPEAT
        )
        latest_output = [_build_text_part(latest), _build_image_part("latest-image")] if multipart else latest
        history = [
            {"role": "user", "content": "goal"},
            _build_function_call_item("d1", "dump_page"),
            _build_function_output_item(
                "d1",
                observation
                + "Checkpoint: first\nDOM dump target: tab1\n![image](data:image/jpeg;base64,"
                + "A" * _INLINE_IMAGE_REPEAT
                + ")",
            ),
            _build_function_call_item("d2", "dump_page"),
            _build_function_output_item("d2", latest_output),
            _build_function_call_item("h1", "checkpoint"),
            _build_function_output_item(
                "h1",
                [_build_text_part("Historical checkpoint: saved\nDOM dump target: tab1\nEvidence"), _build_image_part("historical-image")],
            ),
            {"role": "assistant", "content": "small reply"},
        ]
        original = json.loads(json.dumps(history))
        captured: list[list[dict]] = []
        error = _build_overflow_error()

        def run_streamed(
            agent: SdkAgent, items: str | list, *, run_config: RunConfig, max_turns: int | None
        ) -> MagicMock:
            _ = max_turns
            assert run_config.call_model_input_filter is history_module.prune_model_history
            if isinstance(items, str):
                return _build_overflow_stream(history, error)
            filtered = run_config.call_model_input_filter(
                CallModelData(model_data=ModelInputData(input=items, instructions=agent.instructions), agent=agent, context=None)
            )
            captured.append(filtered.input)
            assert (
                run_config.call_model_input_filter(
                    CallModelData(model_data=filtered, agent=agent, context=None)
                ).input
                == filtered.input
            )
            return _build_overflow_stream(items, None)

        monkeypatch.setattr(Runner, "run_streamed", run_streamed)
        result = await runner_module.run_browser_use("goal", str(tmp_path / "downloads"), str(tmp_path))
        assert result.status == "completed"
        assert len(captured) == 1
        kept = captured[0]
        assert [item.get("call_id") for item in kept] == [None, "d1", "d1", "d2", "d2", "h1", "h1"]
        assert kept[1] == history[1]
        assert kept[3] == history[3]
        assert "data:image" not in kept[2]["output"]
        assert "Checkpoint: first" in kept[2]["output"]
        text = kept[4]["output"][0]["text"] if multipart else kept[4]["output"]
        assert observation in text
        assert "Archived DOM target: tab2" in text
        assert "BODY2" not in text
        assert "Observation time: now" in text
        assert "DOM capture interval: start to end" in text
        assert ("checkpoint_id newest" if archive else "Checkpoint archive unavailable.") in text
        if multipart:
            assert kept[4]["output"][1] == _build_image_part("latest-image")
        assert kept[5:7] == history[5:7]
        assert history == original

    @pytest.mark.usefixtures("disconnected_browser")
    @pytest.mark.parametrize("repeat_after_compaction", [True, False])
    async def test_overflow_stops_when_no_unarchived_dumps_remain(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repeat_after_compaction: bool
    ) -> None:
        """Overflow stops without another model call when every page dump is already archived."""
        set_thinking_backend(monkeypatch)
        observation = "1. tab1 — Page\nstatus: success\n"
        stub = (
            "Archived DOM target: tab1\nCheckpoint: old\n"
            "DOM artifact available in the checkpoint archive.\n"
            "Read checkpoint with action read, checkpoint_id old, target_id tab1, kind dom."
        )
        tail = (
            [
                _build_function_call_item("d1", "dump_page"),
                _build_function_output_item(
                    "d1", observation + "Checkpoint: first\nDOM dump target: tab1\n" + "BODY1" * _OVERSIZED_BODY_REPEAT
                ),
            ]
            if repeat_after_compaction
            else [
                _build_function_call_item("c1", "open_url"),
                _build_function_output_item("c1", observation + "LARGE" * _OVERSIZED_BODY_REPEAT),
            ]
        )
        history = [
            {"role": "user", "content": "goal"},
            _build_function_call_item("d0", "dump_page"),
            _build_function_output_item("d0", stub),
            *tail,
        ]
        original = json.loads(json.dumps(history))
        calls: list[str | list] = []
        error = _build_overflow_error()

        def run_streamed(
            agent: SdkAgent, items: str | list, *, run_config: RunConfig, max_turns: int | None
        ) -> MagicMock:
            _ = agent, max_turns
            assert run_config.call_model_input_filter is history_module.prune_model_history
            calls.append(json.loads(json.dumps(items)))
            if isinstance(items, str):
                return _build_overflow_stream(history, error)
            return _build_overflow_stream(items, error, new_items=[])

        monkeypatch.setattr(Runner, "run_streamed", run_streamed)
        with pytest.raises(BadRequestError) as caught:
            await runner_module.run_browser_use("goal", str(tmp_path / "downloads"), str(tmp_path))
        assert caught.value is error
        assert len(calls) == (2 if repeat_after_compaction else 1)
        if repeat_after_compaction:
            resumed = calls[1]
            assert isinstance(resumed, list)
            assert resumed[2]["output"] == stub
            assert "Archived DOM target: tab1" in resumed[4]["output"]
            assert "BODY1" not in resumed[4]["output"]
            assert await history_module.compact_dom_dumps(resumed) is None
        else:
            assert await history_module.compact_dom_dumps(history) is None
        assert history == original

    @pytest.mark.usefixtures("disconnected_browser")
    async def test_later_overflow_archives_new_dump_and_keeps_old_stub(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A second overflow archives a new page dump and leaves the old stub unchanged."""
        set_thinking_backend(monkeypatch)
        observation = "1. tab1 — Page\nstatus: success\n"
        first_history = [
            {"role": "user", "content": "goal"},
            _build_function_call_item("d1", "dump_page"),
            _build_function_output_item(
                "d1",
                observation + "Checkpoint: first\nObservation time: t1\nDOM dump target: tab1\n" + "BODY1" * _OVERSIZED_BODY_REPEAT,
            ),
        ]
        later_turn = [
            _build_function_call_item("d2", "dump_page"),
            _build_function_output_item(
                "d2",
                observation + "Checkpoint: second\nObservation time: t2\nDOM dump target: tab1\n" + "BODY2" * _OVERSIZED_BODY_REPEAT,
            ),
        ]
        raw_inputs: list[list[dict]] = []
        filtered_inputs: list[list[dict]] = []
        error = _build_overflow_error()

        def run_streamed(
            agent: SdkAgent, items: str | list, *, run_config: RunConfig, max_turns: int | None
        ) -> MagicMock:
            _ = max_turns
            assert run_config.call_model_input_filter is history_module.prune_model_history
            if isinstance(items, str):
                return _build_overflow_stream(first_history, error)
            raw_inputs.append(json.loads(json.dumps(items)))
            if len(raw_inputs) == 1:
                return _build_overflow_stream(list(items) + later_turn, error)
            filtered = run_config.call_model_input_filter(
                CallModelData(model_data=ModelInputData(input=items, instructions=agent.instructions), agent=agent, context=None)
            )
            filtered_inputs.append(filtered.input)
            return _build_overflow_stream(items, None)

        monkeypatch.setattr(Runner, "run_streamed", run_streamed)
        result = await runner_module.run_browser_use("goal", str(tmp_path / "downloads"), str(tmp_path))
        assert result.status == "completed"
        assert len(raw_inputs) == 2
        assert raw_inputs[1][2] == raw_inputs[0][2]
        assert len(filtered_inputs) == 1
        kept = filtered_inputs[0]
        assert [item.get("call_id") for item in kept] == [None, "d1", "d1", "d2", "d2"]
        assert "Archived DOM target: tab1" in kept[2]["output"]
        assert "Checkpoint: first" in kept[2]["output"]
        assert "Observation time: t1" in kept[2]["output"]
        assert "status:" not in kept[2]["output"]
        assert "BODY1" not in kept[2]["output"]
        assert observation in kept[4]["output"]
        assert "Archived DOM target: tab1" in kept[4]["output"]
        assert "Checkpoint: second" in kept[4]["output"]
        assert "Observation time: t2" in kept[4]["output"]
        assert "BODY2" not in kept[4]["output"]
