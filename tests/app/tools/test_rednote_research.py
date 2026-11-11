from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.lib.db.store as store_module
from app.lib.tool.base import ToolResult
from app.tools.rednote_research import rednote_research_tool
from app.tools.task_tool import task_tool
from tests.app.conftest import (
    assert_browser_closed,
    is_llm_available,
    skip_if_no_chrome,
    wire_context_manager,
)
from tests.app.lib.tool.conftest import await_task_result, run_text, wait_terminal

_RESEARCH_FAILURES = (
    "Research failed: could not capture search results.",
    "Research failed: no notes discovered.",
    "Research failed: no analysis results gathered.",
)

_FIRST_IMAGE_URL = "https://sns-img.rednote.com/first.png"
_SECOND_IMAGE_URL = "https://sns-img.rednote.com/second.png"

_FIXED_NOW = datetime(2024, 5, 21, 14, 30, tzinfo=timezone(timedelta(hours=-7)))
_FIXED_TIMESTAMP = "2024-05-21T14:30:00-07:00"


class _TaskStore(Protocol):
    """The task record store the tool_infra fixture gives to a test."""


def _make_snap(tree: dict | None = None, actionable: list | None = None) -> MagicMock:
    """Return a fake page snapshot with the given tree and clickable elements."""
    snap = MagicMock()
    snap.filtered_tree = AsyncMock(return_value=tree)
    snap.actionable_elements.return_value = actionable or []
    snap.click = AsyncMock(return_value=True)
    return snap


def _manager_with_snapshots(
    mock_mgr_class: MagicMock,
    search_snap: MagicMock,
    note_snap: MagicMock,
    search_calls: int = 2,
) -> MagicMock:
    """Return a fake browser manager wired to the given snapshots.

    The search snapshot comes back for the first search_calls snapshots, then
    the note snapshot comes back.
    """
    mgr = AsyncMock()
    mgr.new_page.return_value = "tab1"
    mgr.wait_for_selector.return_value = True
    mgr.wait_for_selector_disappear.return_value = True
    mgr.press_key.return_value = True
    call_count = [0]

    def snapshot_side_effect(*_args: str, **_kwargs: bool) -> MagicMock:
        call_count[0] += 1
        if call_count[0] <= search_calls:
            return search_snap
        return note_snap

    mgr.snapshot.side_effect = snapshot_side_effect
    wire_context_manager(mock_mgr_class, mgr)
    return mgr


def _is_note_schema(json_schema: dict | None) -> bool:
    """Return True if the schema is the one used for note analysis."""
    if not isinstance(json_schema, dict):
        return False
    enum = json_schema.get("properties", {}).get("status", {}).get("enum", [])
    return "skip" in enum


def _is_search_term_prompt(prompt: str) -> bool:
    """Return True if the prompt asks for the Chinese search term."""
    return "natural Chinese" in prompt


def _note_analysis_prompts(mock_reason: MagicMock) -> list[str]:
    """Return the prompts sent for note analysis."""
    return [
        call.args[0]
        for call in mock_reason.call_args_list
        if _is_note_schema(call.kwargs.get("json_schema"))
    ]


def _report_prompt(mock_reason: MagicMock) -> str:
    """Return the prompt sent for the final report."""
    prompts = [
        call.args[0]
        for call in mock_reason.call_args_list
        if call.kwargs.get("json_schema") is None
        and not _is_search_term_prompt(call.args[0])
    ]
    return prompts[-1]


def _search_tree(*urls: str) -> dict:
    """Return a search page tree with one link per note."""
    return {
        "role": "generic",
        "children": [
            {"role": "link", "url": f"/search_result/note{i}", "name": f"Note {i}"}
            for i, _url in enumerate(urls, 1)
        ],
    }


def _search_actionable(*urls: str) -> list[dict]:
    """Return clickable elements that match the search page links."""
    return [
        {"ref_id": f"r{i}", "role": "link", "name": f"Note {i}"}
        for i, _url in enumerate(urls, 1)
    ]


def _note_tree_with_images(*img_urls: str) -> dict:
    """Return a note tree that carries the given image URLs."""
    return {
        "role": "overlay",
        "name": "Note content",
        "children": [
            {"role": "main", "children": [{"text": "note body"}]},
            *({"role": "image", "src": url} for url in img_urls),
        ],
    }


class _VisionStub:
    """Stands in for downloading note images and reading them."""

    def __init__(self, transcriptions: dict[str, str]) -> None:
        """Args:
            transcriptions: Text for each image URL. A URL left out makes that
                image fail.
        """
        self._transcriptions = transcriptions

    def make_client(self) -> MagicMock:
        """Return a fake web client whose reply body is the URL that was asked for."""
        async def _get(url: str, **_kwargs: bool | int) -> MagicMock:
            resp = MagicMock()
            resp.content = url.encode()
            resp.raise_for_status = MagicMock()
            return resp
        client = MagicMock()
        client.get = _get
        return client

    def transcribe(self, path: str) -> tuple[str, int]:
        """Return the text for the image saved at the given path."""
        with open(path, "rb") as f:
            url = f.read().decode()
        if url not in self._transcriptions:
            raise RuntimeError(f"vision backend failed for {url}")
        return self._transcriptions[url], 0


def _wire_httpx_client(mock_client_cls: MagicMock, vision: _VisionStub) -> None:
    """Point the fake web client class at the stub's client."""
    client = vision.make_client()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = client


def _llm_with_note_response(
    note_handler: Callable[[int], dict] | None = None,
    report: str = "Research report",
) -> Callable[..., tuple[str, int]]:
    """Return a fake model that answers note prompts and the report prompt.

    Args:
        note_handler: Given the note number, returns the note analysis.
        report: Text returned as the final report.

    Returns:
        A stand in for the model call.
    """
    note_calls = [0]

    def side_effect(prompt: str, **kwargs: dict | str | None) -> tuple[str, int]:
        json_schema = kwargs.get("json_schema")
        if json_schema is None:
            if _is_search_term_prompt(prompt):
                return "测试查询", 0
            return report, 0
        if _is_note_schema(json_schema):
            note_calls[0] += 1
            if note_handler is not None:
                return json.dumps(note_handler(note_calls[0])), 0
            return json.dumps({
                "status": "ok",
                "extracted_content": "Great content",
                "note_creation_date": "2024-01-01",
            }), 0
        return json.dumps({"status": "ok", "reason": "content visible"}), 0

    return side_effect


def _blocked_search_llm() -> Callable[..., tuple[str, int]]:
    """Return a fake model that says the search page is blocked."""
    def side_effect(_prompt: str, **kwargs: dict | str | None) -> tuple[str, int]:
        json_schema = kwargs.get("json_schema")
        if json_schema is None:
            return "测试查询", 0
        if _is_note_schema(json_schema):
            return json.dumps({
                "status": "ok",
                "extracted_content": "content",
                "note_creation_date": "",
            }), 0
        return json.dumps({"status": "blocked", "reason": "captcha required"}), 0

    return side_effect


def _no_notes_llm() -> Callable[..., tuple[str, int]]:
    """Return a fake model that says the search page is fine."""
    def side_effect(_prompt: str, **kwargs: dict | str | None) -> tuple[str, int]:
        json_schema = kwargs.get("json_schema")
        if json_schema is None:
            return "测试查询", 0
        return json.dumps({"status": "ok", "reason": "visible"}), 0

    return side_effect


def _wire_successful_flow(mock_mgr_class: MagicMock, mock_reason: MagicMock) -> None:
    """Set up the fake browser and model for a research run over one note."""
    search_snap = _make_snap(_search_tree("a"), _search_actionable("a"))
    note_snap = _make_snap({"role": "overlay", "name": "Note content"})
    _manager_with_snapshots(mock_mgr_class, search_snap, note_snap)
    mock_reason.side_effect = _llm_with_note_response()


class TestRednoteResearchShape:
    """The tool declares the research surface and nothing else."""

    def test_tool_shape(self) -> None:
        assert rednote_research_tool.name == "rednote_research"
        assert rednote_research_tool.is_slow is True
        assert rednote_research_tool.staleness == timedelta(hours=24)
        assert rednote_research_tool.result_render_type is None
        assert rednote_research_tool.summary
        assert "\n" not in rednote_research_tool.summary
        assert len(rednote_research_tool.summary) <= 80


class TestResearch:

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_research_successful(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock
    ) -> None:
        """A full research run returns a report."""
        search_snap = _make_snap(
            _search_tree("a", "b"), _search_actionable("a", "b")
        )
        note_snap = _make_snap({"role": "overlay", "name": "Note content"})
        _manager_with_snapshots(mock_mgr_class, search_snap, note_snap)
        mock_reason.side_effect = _llm_with_note_response()

        result = await _research_result("test query", count=5)

        assert "Research report" in result
        assert_browser_closed(mock_mgr_class)

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_research_raw_notes_markdown(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock
    ) -> None:
        """With raw notes on, note details come back as markdown instead of a report."""
        search_snap = _make_snap(
            _search_tree("a"), _search_actionable("a")
        )
        note_snap = _make_snap({"role": "overlay", "name": "Note content"})
        _manager_with_snapshots(mock_mgr_class, search_snap, note_snap)
        mock_reason.side_effect = _llm_with_note_response()

        result = await _research_result("test query", count=5, raw_notes=True)

        assert "## Note: note1" in result
        assert "**URL**:" in result
        assert "**Created**:" in result
        assert "Great content" in result

    @patch("app.tools.rednote_research.datetime")
    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_research_prompts_carry_current_timestamp(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock, mock_datetime: MagicMock
    ) -> None:
        """Note prompts and the report prompt state the current time."""
        search_snap = _make_snap(_search_tree("a"), _search_actionable("a"))
        note_snap = _make_snap({"role": "overlay", "name": "Note content"})
        _manager_with_snapshots(mock_mgr_class, search_snap, note_snap)
        mock_reason.side_effect = _llm_with_note_response()
        mock_datetime.now.return_value.astimezone.return_value = _FIXED_NOW

        await _research_result("test query", count=1)

        assert _FIXED_TIMESTAMP in _note_analysis_prompts(mock_reason)[0]
        assert _FIXED_TIMESTAMP in _report_prompt(mock_reason)

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_research_empty_query(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock
    ) -> None:
        """An empty query is rejected and the browser is never started."""
        assert await run_text(rednote_research_tool, "   ") == "query must not be empty"
        mock_mgr_class.assert_not_called()
        mock_reason.assert_not_called()

    @patch("app.tools.rednote_research.BrowserManager")
    async def test_research_count_zero(self, mock_mgr_class: MagicMock) -> None:
        """A count below 1 is rejected and the browser is never started."""
        result = await run_text(rednote_research_tool, "test", count=0)
        assert result == "Error: count: Input should be greater than or equal to 1."
        mock_mgr_class.assert_not_called()

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_research_blocked_search_page(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock
    ) -> None:
        """A blocked search page returns a blocked message with the reason."""
        _manager_with_snapshots(
            mock_mgr_class, _make_snap({"role": "generic"}), _make_snap({"role": "generic"})
        )
        mock_reason.side_effect = _blocked_search_llm()

        result = await _research_result("test query", count=5)
        assert result == "Research blocked: captcha required"
        assert_browser_closed(mock_mgr_class)

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_research_no_notes_found(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock
    ) -> None:
        """A search page with no notes returns a failure message."""
        _manager_with_snapshots(
            mock_mgr_class,
            _make_snap({"role": "generic", "children": []}),
            _make_snap({"role": "generic"}),
        )
        mock_reason.side_effect = _no_notes_llm()

        result = await _research_result("test query", count=5)
        assert result == "Research failed: no notes discovered."
        assert_browser_closed(mock_mgr_class)

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_research_skips_unavailable_note(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock
    ) -> None:
        """A report still comes back when one note cannot be read."""
        search_snap = _make_snap(
            _search_tree("a", "b"), _search_actionable("a", "b")
        )
        note_snap = _make_snap({"role": "overlay", "name": "Note content"})
        _manager_with_snapshots(mock_mgr_class, search_snap, note_snap, search_calls=4)

        def note_handler(call_index: int) -> dict:
            if call_index == 1:
                return {"status": "skip", "reason": "page not available"}
            return {
                "status": "ok",
                "extracted_content": "Second note content",
                "note_creation_date": "2024-01-01",
            }

        mock_reason.side_effect = _llm_with_note_response(note_handler)

        result = await _research_result("test query", count=5)

        assert "Research report" in result

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_research_hard_block_raises_message(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock
    ) -> None:
        """A hard block returns a message asking for a person to step in."""
        search_snap = _make_snap(
            _search_tree("a"), _search_actionable("a")
        )
        note_snap = _make_snap({"role": "overlay", "name": "Captcha"})
        _manager_with_snapshots(mock_mgr_class, search_snap, note_snap)
        mock_reason.side_effect = _llm_with_note_response(
            lambda _call_index: {"status": "blocked", "reason": "Verify your identity"}
        )

        result = await _research_result("test query", count=5)
        assert result == "Research blocked, human intervention required: Verify your identity"
        assert_browser_closed(mock_mgr_class)

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_research_respects_count_limit(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock
    ) -> None:
        """A report comes back when the count is smaller than the notes found."""
        search_snap = _make_snap(
            _search_tree("a", "b"), _search_actionable("a", "b")
        )
        note_snap = _make_snap({"role": "overlay", "name": "Note content"})
        _manager_with_snapshots(mock_mgr_class, search_snap, note_snap)
        mock_reason.side_effect = _llm_with_note_response()

        result = await _research_result("test query", count=1)

        assert "Research report" in result

    @patch("app.tools.rednote_research.describe_image")
    @patch("app.tools.rednote_research.httpx.AsyncClient")
    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_research_appends_image_transcriptions(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock,
        mock_client_cls: MagicMock, mock_describe_image: MagicMock,
    ) -> None:
        """Every note image is read and numbered in order."""
        search_snap = _make_snap(_search_tree("a"), _search_actionable("a"))
        note_snap = _make_snap(
            _note_tree_with_images(_FIRST_IMAGE_URL, _SECOND_IMAGE_URL)
        )
        _manager_with_snapshots(mock_mgr_class, search_snap, note_snap)
        mock_reason.side_effect = _llm_with_note_response()
        vision = _VisionStub({
            _FIRST_IMAGE_URL: "A red teapot",
            _SECOND_IMAGE_URL: "A blue mug",
        })
        _wire_httpx_client(mock_client_cls, vision)
        mock_describe_image.side_effect = vision.transcribe

        result = await _research_result("test query", count=5, raw_notes=True)

        assert "[Image Transcriptions]:" in result
        assert "[Image 1]: A red teapot" in result
        assert "[Image 2]: A blue mug" in result
        assert result.index("[Image 1]") < result.index("[Image 2]")
        assert mock_describe_image.call_count == 2

    @patch("app.tools.rednote_research.describe_image")
    @patch("app.tools.rednote_research.httpx.AsyncClient")
    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_research_keeps_note_when_one_image_fails(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock,
        mock_client_cls: MagicMock, mock_describe_image: MagicMock,
    ) -> None:
        """One failing image drops only its own text."""
        search_snap = _make_snap(_search_tree("a"), _search_actionable("a"))
        note_snap = _make_snap(
            _note_tree_with_images(_FIRST_IMAGE_URL, _SECOND_IMAGE_URL)
        )
        _manager_with_snapshots(mock_mgr_class, search_snap, note_snap)
        mock_reason.side_effect = _llm_with_note_response()
        vision = _VisionStub({_SECOND_IMAGE_URL: "A blue mug"})
        _wire_httpx_client(mock_client_cls, vision)
        mock_describe_image.side_effect = vision.transcribe

        result = await _research_result("test query", count=5, raw_notes=True)

        assert "Great content" in result
        assert "[Image 2]: A blue mug" in result
        assert "[Image 1]" not in result

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_call_raw_notes_declares_markdown(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock
    ) -> None:
        """The raw-notes dump is authored markdown, and the wrapper says so."""
        _wire_successful_flow(mock_mgr_class, mock_reason)
        result = await rednote_research_tool.call({"query": "q", "count": 1, "raw_notes": True})
        assert isinstance(result, ToolResult)
        assert result.render_type == "markdown"
        assert "## Note:" in result.text

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_call_report_declares_markdown(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock
    ) -> None:
        """The synthesized report is authored markdown, and the wrapper says so."""
        _wire_successful_flow(mock_mgr_class, mock_reason)
        result = await rednote_research_tool.call({"query": "q", "count": 1})
        assert result.render_type == "markdown"
        assert result.text == "Research report"

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_call_scalar_report_declares_markdown(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock
    ) -> None:
        """A report body that is a bare scalar is not json, so it stays markdown."""
        _wire_successful_flow(mock_mgr_class, mock_reason)
        mock_reason.side_effect = _llm_with_note_response(report="5")
        result = await rednote_research_tool.call({"query": "q", "count": 1})
        assert result.render_type == "markdown"
        assert result.text == "5"

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_run_json_fallback_declares_json(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock, tool_infra: _TaskStore
    ) -> None:
        """The fallback body is a JSON array, so it is declared json."""
        _wire_successful_flow(mock_mgr_class, mock_reason)
        mock_reason.side_effect = _llm_with_note_response(report="")
        task = await rednote_research_tool.run({"query": "q", "count": 1})
        await wait_terminal(tool_infra, str(task.id))
        body = await run_text(task_tool, action="result", task_id=str(task.id))
        assert isinstance(json.loads(body), list)
        view = await store_module.task_view(str(task.id))
        assert view is not None and view.render_type == "json"


async def _research_result(query: str, count: int = 5, raw_notes: bool = False) -> str:
    """Run the tool and return the finished result or the saved error."""
    task = await rednote_research_tool.run({"query": query, "count": count, "raw_notes": raw_notes})
    return await await_task_result(str(task.id))


class TestResearchInvoke:

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_invoke_starts_a_run_that_researches(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock, tool_infra: _TaskStore
    ) -> None:
        """A call starts a run, and the run does the research in the background."""
        _wire_successful_flow(mock_mgr_class, mock_reason)

        task = await rednote_research_tool.run({"query": "q"})

        await wait_terminal(tool_infra, str(task.id))
        assert mock_mgr_class.call_count == 1

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_invoke_caches_within_staleness(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock, tool_infra: _TaskStore
    ) -> None:
        """A repeat call with the same input returns the saved result without running again."""
        _wire_successful_flow(mock_mgr_class, mock_reason)

        first_task = await rednote_research_tool.run({"query": "q"})
        await wait_terminal(tool_infra, str(first_task.id))
        first = await run_text(task_tool, action="result", task_id=str(first_task.id))

        second = await run_text(rednote_research_tool, "q")

        assert first == second == "Research report"
        assert mock_mgr_class.call_count == 1

    @patch("app.tools.rednote_research.reason")
    @patch("app.tools.rednote_research.BrowserManager")
    async def test_invoke_cache_key_ignores_nothing_else(
        self, mock_mgr_class: MagicMock, mock_reason: MagicMock, tool_infra: _TaskStore
    ) -> None:
        """A call with a different count runs the research again."""
        _wire_successful_flow(mock_mgr_class, mock_reason)

        first_task = await rednote_research_tool.run({"query": "q"})
        await wait_terminal(tool_infra, str(first_task.id))
        _wire_successful_flow(mock_mgr_class, mock_reason)
        second_task = await rednote_research_tool.run({"query": "q", "count": 2})
        await wait_terminal(tool_infra, str(second_task.id))

        assert mock_mgr_class.call_count == 2


@pytest.mark.skipif(not is_llm_available(), reason="LLM server not available")
@skip_if_no_chrome
@pytest.mark.live
@pytest.mark.timeout(600)
class TestResearchLive:
    """Live tests that need Chrome and the model server."""

    async def test_research_restaurants_bay_area(self) -> None:
        """A live research run returns a blocked message, a failure message, or a
        report about the topic.
        """
        result = await run_text(rednote_research_tool, "best restaurants in bay area", count=2)

        if result.startswith("Research blocked:"):
            reason = result.removeprefix("Research blocked:").strip()
            assert reason, "a blocked result must include a non-empty reason"
        elif result.startswith("Research blocked, human intervention required:"):
            reason = result.removeprefix(
                "Research blocked, human intervention required:"
            ).strip()
            assert reason, "a hard-block result must include a non-empty reason"
        elif result in _RESEARCH_FAILURES:
            pass
        else:
            assert isinstance(result, str), f"unexpected return type: {type(result)}"
            assert "restaurant" in result.lower(), (
                "expected a synthesized report addressing the query topic, "
                f"got an unexpected result: {result[:200]!r}"
            )
