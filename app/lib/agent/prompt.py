"""Prompt loading and tool description rendering for agent roles."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import aiofiles

from app.lib.tool.base import Tool

__all__ = ["load_prompt", "render_tool_descriptions"]


def render_tool_descriptions(tools: tuple[Tool, ...]) -> str:
    """Render the available tools markdown for a role prompt.

    Args:
        tools: The tools wired into the role.

    Returns:
        The markdown block with each tool's wire description and params.
    """
    lines = ["### AVAILABLE TOOLS", ""]
    for tool in tools:
        names = [p.name for p in tool.params]
        sig = ", ".join(f"`{n}`" for n in names)
        lines.append(f"**{tool.name}({sig})**")
        lines.append(tool.dynamic_description)
        lines.append("")
        lines.append("Parameters:")
        for p in tool.params:
            lines.append(f"- `{p.name}`: {p.wire_description}")
        lines.append("")
    return "\n".join(lines)


async def load_prompt(prompts_dir: str, name: str, *, tool_descriptions: str) -> str:
    """Load a prompt file with the timestamp and tools placeholders filled in.

    Args:
        prompts_dir: The folder that holds the prompt markdown files.
        name: The prompt name, the markdown file name without the extension.
        tool_descriptions: The text for the tool descriptions placeholder.

    Returns:
        The prompt text with both placeholders substituted.

    Raises:
        FileNotFoundError: When the prompt file is missing.
    """
    path = os.path.join(prompts_dir, f"{name}.md")
    async with aiofiles.open(path, "r", encoding="utf-8") as fh:
        content = await fh.read()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    content = content.replace("[CURRENT TIMESTAMP]", timestamp)
    content = content.replace("[TOOL_DESCRIPTIONS]", tool_descriptions)
    return content
