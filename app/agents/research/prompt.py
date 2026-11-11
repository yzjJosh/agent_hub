"""Prompt loading for the research agent roles."""

from __future__ import annotations

import os

from app.lib.agent.prompt import load_prompt as _load_prompt

__all__ = ["REVIEW_FEEDBACK_HEADER", "load_prompt"]

REVIEW_FEEDBACK_HEADER = (
    "### REVIEW "
    "FEEDBACK"
)

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


async def load_prompt(name: str, *, tool_descriptions: str) -> str:
    """Load a research prompt with the timestamp and tools placeholders filled in.

    Args:
        name: The prompt name, the markdown file name without the extension.
        tool_descriptions: The text for the tool descriptions placeholder.

    Returns:
        The prompt text with both placeholders substituted.

    Raises:
        FileNotFoundError: When the prompt file is missing.
    """
    return await _load_prompt(_PROMPTS_DIR, name, tool_descriptions=tool_descriptions)
