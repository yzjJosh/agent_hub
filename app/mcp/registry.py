"""Puts a tool onto the MCP wire.

Every fact about a tool lives in the tool class. Nothing about one tool belongs here.
"""
from __future__ import annotations

import inspect

import fastmcp.settings
from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

from app.lib.shared.paths import resources_dir
from app.lib.task import TaskStatus
from app.lib.tool.base import Contract, Param, ParamValue, Tool, param_signature
from app.lib.tool.truncation import (
    DEFAULT_TOOL_OUTPUT_LEN_LIMIT,
    MAX_CHAR_DESCRIPTION,
    MIN_TOOL_OUTPUT_LEN_LIMIT,
    truncate,
)
from app.resources.resources import register_resources
from app.tools.brave_search import brave_search_tool
from app.tools.browser_use import browser_use_tool
from app.tools.edit_image import edit_image_tool
from app.tools.generate_image import generate_image_tool
from app.tools.generate_video import generate_video_tool
from app.tools.github import github_tool
from app.tools.google_flights import google_flights_tool
from app.tools.google_search import google_search_tool
from app.tools.present_to_user import present_to_user_tool
from app.tools.read_media_or_doc import read_media_or_doc_tool
from app.tools.redfin import redfin_search_tool
from app.tools.rednote_research import rednote_research_tool
from app.tools.research import research_tool
from app.tools.resource_to_image import resource_to_image_tool
from app.tools.send_or_receive_file import send_or_receive_file_tool
from app.tools.task_tool import task_tool
from app.tools.tts import tts_tool
from app.tools.web_fetch import web_fetch_tool
from app.tools.web_screenshot import web_screenshot_tool
from app.tools.x_search import x_search_tool

__all__ = ["mcp", "registered_tools", "tool_class"]

fastmcp.settings.message_path = "/mcp/messages/"

mcp = FastMCP("agent_hub")

_TOOLS: dict[str, type[Tool]] = {}

_MAX_CHAR_PARAM = Param(
    "max_char",
    int,
    DEFAULT_TOOL_OUTPUT_LEN_LIMIT,
    min=MIN_TOOL_OUTPUT_LEN_LIMIT,
    description=MAX_CHAR_DESCRIPTION,
)

# The notice a slow tool's call answers with while its run is still going.
_ACK_MESSAGE = (
    "Task started. task_id={task_id}. Retrieve the result with the task tool: "
    "task(action='result', task_id='{task_id}')."
)


def _wire_params(inner: Tool) -> tuple[Param, ...]:
    """The params the MCP wire exposes for one tool."""
    if inner.is_result_long:
        return inner.params + (_MAX_CHAR_PARAM,)
    return inner.params


async def _invoke_tool(inner: Tool, params: dict) -> str:
    """Run the tool and give back the text the wire answers with."""
    max_char = (max(MIN_TOOL_OUTPUT_LEN_LIMIT, params.pop("max_char", DEFAULT_TOOL_OUTPUT_LEN_LIMIT))
                if inner.is_result_long else None)
    task = await inner.run(params)
    task_id = str(task.id)
    await task.wait_started()
    if inner.is_slow and task.status == TaskStatus.RUNNING:
        return _ACK_MESSAGE.format(task_id=task_id)
    text = await inner.result_text(task)
    return truncate(text, {"max_char": max_char}, task_id)


def _synthesized_signature(tool: Tool) -> inspect.Signature:
    """The call signature for a tool's params."""
    return param_signature(_wire_params(tool))


def _annotations(tool: Tool) -> dict:
    return {p.name: p.annotation for p in _wire_params(tool)} | {"return": str}


def _register_tool(mcp: FastMCP, instance: Tool) -> None:
    """Add one tool to an MCP server.

    Args:
        mcp: The server the tool is added to.
        instance: The tool to add.

    Raises:
        AttributeError: If the tool is missing something it must declare.
    """
    contract: Contract = instance.contract()
    name = contract.name
    description = instance.dynamic_description
    params = _wire_params(instance)

    async def fn(**kw: ParamValue) -> str:
        return await _invoke_tool(instance, kw)

    fn.__name__ = name
    fn.__signature__ = _synthesized_signature(instance)
    fn.__annotations__ = _annotations(instance)

    tool = FunctionTool.from_function(fn, name=name, description=description)
    props = tool.parameters.get("properties", {})
    for p in params:
        props[p.name]["description"] = p.wire_description
        if p.min is not None:
            props[p.name]["minimum"] = p.min
        if p.max is not None:
            props[p.name]["maximum"] = p.max
    mcp.add_tool(tool)
    _TOOLS[name] = type(instance)


def tool_class(name: str) -> type[Tool] | None:
    """The registered tool class with this name, or None when the name is unknown."""
    return _TOOLS.get(name)


def registered_tools() -> dict[str, type[Tool]]:
    """A copy of the map from tool name to tool class."""
    return dict(_TOOLS)


_register_tool(mcp, brave_search_tool)
_register_tool(mcp, google_search_tool)
_register_tool(mcp, web_fetch_tool)
_register_tool(mcp, web_screenshot_tool)
_register_tool(mcp, read_media_or_doc_tool)
_register_tool(mcp, present_to_user_tool)
_register_tool(mcp, resource_to_image_tool)
_register_tool(mcp, rednote_research_tool)
_register_tool(mcp, research_tool)
_register_tool(mcp, browser_use_tool)
_register_tool(mcp, redfin_search_tool)
_register_tool(mcp, edit_image_tool)
_register_tool(mcp, generate_image_tool)
_register_tool(mcp, generate_video_tool)
_register_tool(mcp, google_flights_tool)
_register_tool(mcp, github_tool)
_register_tool(mcp, x_search_tool)
_register_tool(mcp, send_or_receive_file_tool)
_register_tool(mcp, tts_tool)
_register_tool(mcp, task_tool)

register_resources(mcp, resources_dir())
