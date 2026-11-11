import re
from datetime import timedelta
from pathlib import Path
from typing import ClassVar, override

import pytest

from app.lib.agent.prompt import load_prompt, render_tool_descriptions
from app.lib.tool.base import Param, Tool, ToolResult


class _FakeBlockingTool(Tool):
    """A blocking tool with a required, a defaulted, and an enum param."""

    name: ClassVar[str] = "fake_blocking"
    description: ClassVar[str] = "A blocking tool for testing."
    summary: ClassVar[str] = "test"
    params: ClassVar[tuple[Param, ...]] = (
        Param("query", str, description="The search query."),
        Param("limit", int, 10, description="Max results."),
        Param("mode", str, "fast", description="The search mode.", enum=("fast", "full")),
    )

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="ok")


class _FakeAsyncTool(Tool):
    """A slow tool with a required param and a plain description."""

    name: ClassVar[str] = "fake_async"
    is_slow: ClassVar[bool] = True
    description: ClassVar[str] = "An async tool for testing."
    summary: ClassVar[str] = "test"
    params: ClassVar[tuple[Param, ...]] = (
        Param("query", str, description="The query."),
    )

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="ok")


class _FakeMultiParagraphTool(Tool):
    """A blocking tool whose description spans multiple paragraphs."""

    name: ClassVar[str] = "fake_multi"
    description: ClassVar[str] = (
        "First paragraph of the description.\n\n"
        "Actions:\n"
        "  view: view the thing."
    )
    summary: ClassVar[str] = "test"
    params: ClassVar[tuple[Param, ...]] = (
        Param("action", str, description="The action to run."),
    )

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="ok")


class _FakeStalenessTool(Tool):
    """A blocking tool with a staleness window for the cache note."""

    name: ClassVar[str] = "fake_stale"
    staleness: ClassVar[timedelta | None] = timedelta(hours=24)
    description: ClassVar[str] = "A staleness tool for testing."
    summary: ClassVar[str] = "test"
    params: ClassVar[tuple[Param, ...]] = (
        Param("path", str, description="The resource path."),
    )

    @override
    async def call(self, _params: dict) -> ToolResult:
        return ToolResult(text="ok")


class TestRenderToolDescriptions:
    def test_blocking_tool_rendered(self) -> None:
        result = render_tool_descriptions((_FakeBlockingTool(),))
        assert "fake_blocking" in result
        assert "A blocking tool for testing." in result
        assert "query" in result
        assert "limit" in result

    def test_slow_tool_shows_no_async_note(self) -> None:
        assert _FakeAsyncTool.is_slow is True
        result = render_tool_descriptions((_FakeAsyncTool(),))
        assert "asynchronous" not in result
        assert "An async tool for testing." in result

    def test_staleness_tool_shows_cache_note(self) -> None:
        result = render_tool_descriptions((_FakeStalenessTool(),))
        assert "Duplicate calls" in result

    def test_param_default_note(self) -> None:
        result = render_tool_descriptions((_FakeBlockingTool(),))
        assert "Optional (default: 10)." in result

    def test_param_enum_note(self) -> None:
        result = render_tool_descriptions((_FakeBlockingTool(),))
        assert "Valid values: fast, full." in result

    def test_multiple_tools(self) -> None:
        tools = (_FakeBlockingTool(), _FakeAsyncTool())
        result = render_tool_descriptions(tools)
        assert "fake_blocking" in result
        assert "fake_async" in result

    @pytest.mark.parametrize(
        "tool_cls",
        [_FakeBlockingTool, _FakeAsyncTool, _FakeStalenessTool, _FakeMultiParagraphTool],
    )
    def test_description_not_asterisk_wrapped(self, tool_cls: type[Tool]) -> None:
        tool = tool_cls()
        result = render_tool_descriptions((tool,))
        desc = tool.dynamic_description
        assert f"*{desc}*" not in result
        assert desc in result
        first, last = desc.splitlines()[0], desc.splitlines()[-1]
        assert f"*{first}" not in result
        assert f"{last}*" not in result
        assert f"**{tool.name}(" in result


class TestLoadPrompt:
    async def test_loads_and_substitutes(self, tmp_path: Path) -> None:
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "test_prompt.md").write_text(
            "Timestamp: [CURRENT TIMESTAMP]\nTools:\n[TOOL_DESCRIPTIONS]\n"
        )
        result = await load_prompt(str(prompts_dir), "test_prompt", tool_descriptions="FAKE_TOOLS")
        assert "FAKE_TOOLS" in result
        assert "[CURRENT TIMESTAMP]" not in result
        assert "[TOOL_DESCRIPTIONS]" not in result
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC", result)

    async def test_missing_file_raises(self, tmp_path: Path) -> None:
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="missing_prompt"):
            await load_prompt(str(prompts_dir), "missing_prompt", tool_descriptions="")

    async def test_folder_is_a_parameter(self, tmp_path: Path) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        (first / "note.md").write_text("Body from the first folder.")
        (second / "note.md").write_text("Body from the second folder.")
        first_text = await load_prompt(str(first), "note", tool_descriptions="")
        second_text = await load_prompt(str(second), "note", tool_descriptions="")
        assert first_text == "Body from the first folder."
        assert second_text == "Body from the second folder."
