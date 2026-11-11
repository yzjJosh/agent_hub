from app.agents.research.prompt import load_prompt


class TestLoadPrompt:
    async def test_wrapper_loads_a_research_prompt(self) -> None:
        result = await load_prompt("collector_prompt", tool_descriptions="FAKE_TOOLS")
        assert "FAKE_TOOLS" in result
        assert "[CURRENT TIMESTAMP]" not in result
        assert "[TOOL_DESCRIPTIONS]" not in result
