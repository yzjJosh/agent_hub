"""Shared output helpers for tools."""

from __future__ import annotations

import asyncio
import json
import re

from app.lib.reason.reason import reason
from app.lib.tool.base import ToolResult

__all__ = ["answer_or_content", "format_results", "html_tree_to_markdown"]

_LIST_ITEM_INDENT = "  "


def _convert_html_node(node: dict | str, depth: int = 1) -> str:
    """Convert one tree node to markdown.

    Args:
        node: One node of the tree.
        depth: How deep the node sits in the tree.

    Returns:
        The node as markdown.
    """
    return _node_block(node, depth)[2]


def _node_block(node: dict | str, depth: int) -> tuple[str, str, str]:
    """Convert one tree node to a block of markdown.

    Args:
        node: One node of the tree.
        depth: How deep the node sits in the tree.

    Returns:
        The kind of the first line, the kind of the last line, and the markdown.
    """
    if _is_text_node(node):
        text = _collapse(_text_of(node))
        return ("other", "other", text)

    if not isinstance(node, dict):
        return ("", "", "")

    role = node.get("role", "")
    children = node.get("children", [])

    if role == "table":
        table = _convert_table(node)
        return ("table", "table", table)

    if role == "list":
        items = _convert_list(node, depth, 1)
        return ("list", "list", items)

    if role == "listitem":
        item = _convert_listitem(node, depth, 1)
        return ("list", "list", item)

    if role == "heading":
        name = node.get("name", "") or _direct_text(node)
        line = f"{'#' * min(depth, 6)} {name}"
        return _named_block("heading", line, name, children, depth)

    if role == "link":
        name = node.get("name", "")
        line = f"[{_escape_link_name(name)}]({node.get('url', '')})"
        return _named_block("other", line, name, children, depth)

    if role == "image":
        line = f"![{node.get('name', '')}]({node.get('src', '')})"
        return ("other", "other", line)

    if role == "button":
        name = node.get("name", "") or _direct_text(node)
        return _named_block("other", name, name, children, depth)

    return _join_blocks([_node_block(child, depth + 1) for child in children])


def _named_block(
    kind: str,
    line: str,
    name: str,
    children: list[dict | str],
    depth: int,
) -> tuple[str, str, str]:
    """Join a node's own line with the markdown of its children.

    Args:
        kind: The kind of the node's own line.
        line: The node's own line of markdown.
        name: The node's name, so a child line that repeats it can be dropped.
        children: The node's children.
        depth: How deep the node sits in the tree.

    Returns:
        The kind of the first line, the kind of the last line, and the markdown.
    """
    rest = _join_blocks(
        [_node_block(child, depth + 1) for child in children if not _same_text(child, name)]
    )
    if rest[2].strip():
        has_own_line = bool(line.strip())
        markdown = f"{line}\n{rest[2]}" if has_own_line else rest[2]
        return (kind if has_own_line else rest[0], rest[1], markdown)
    if line.strip():
        return (kind, kind, line)
    return ("", "", "")


def _join_blocks(blocks: list[tuple[str, str, str]]) -> tuple[str, str, str]:
    """Join markdown blocks with the newlines GFM needs between them.

    Args:
        blocks: The blocks to join, each as first kind, last kind, markdown.

    Returns:
        The kind of the first line, the kind of the last line, and the joined markdown.
    """
    kept = [block for block in blocks if block[2].strip()]
    if not kept:
        return ("", "", "")
    markdown = kept[0][2]
    for prev, cur in zip(kept, kept[1:]):
        is_table_edge = "table" in (prev[1], cur[0])
        is_list_edge = prev[1] == "list" and cur[0] not in ("heading", "table")
        markdown += ("\n\n" if is_table_edge or is_list_edge else "\n") + cur[2]
    return (kept[0][0], kept[-1][1], markdown)


def _convert_list(node: dict, depth: int, list_depth: int) -> str:
    """Convert a list node to markdown lines.

    Args:
        node: The list node.
        depth: How deep the node sits in the tree.
        list_depth: How many lists deep this list sits.

    Returns:
        The list as markdown lines.
    """
    indent = _LIST_ITEM_INDENT * (list_depth - 1)
    parts: list[str] = []
    for child in node.get("children", []):
        if isinstance(child, dict) and child.get("role") == "listitem":
            item = _convert_listitem(child, depth + 1, list_depth)
            if item.strip():
                parts.append(item)
        else:
            markdown = _convert_html_node(child, depth + 1)
            if markdown.strip():
                parts.append(indent + markdown.replace("\n", f"\n{indent}"))
    return "\n".join(parts)


def _convert_listitem(node: dict, depth: int, list_depth: int) -> str:
    """Convert one list item to markdown lines.

    Args:
        node: The listitem node.
        depth: How deep the node sits in the tree.
        list_depth: How many lists deep this item sits.

    Returns:
        The item as markdown lines.
    """
    indent = _LIST_ITEM_INDENT * (list_depth - 1)
    name = node.get("name", "")
    text_parts: list[str] = []
    other_parts: list[str] = []
    nested: list[str] = []
    for child in node.get("children", []):
        if isinstance(child, dict) and child.get("role") == "list":
            nested.append(_convert_list(child, depth + 1, list_depth + 1))
        elif _is_text_node(child):
            text = _collapse(_text_of(child))
            if text and not _same_text(child, name):
                text_parts.append(text)
        else:
            fragment = _inline_fragment(child, depth + 1)
            if fragment.strip():
                other_parts.append(fragment)
    line_parts = ([name] if name else []) + text_parts + other_parts
    lines = [indent + "- " + " ".join(line_parts)] if line_parts else []
    lines.extend(part for part in nested if part.strip())
    return "\n".join(lines)


def _convert_table(node: dict) -> str:
    """Convert a table node to a pipe table.

    Args:
        node: The table node.

    Returns:
        The table as markdown lines, or an empty string when it has no rows.
    """
    rows: list[list[str]] = []
    for row in node.get("children", []):
        if not isinstance(row, dict) or row.get("role") != "row":
            continue
        cells = [c for c in row.get("children", []) if _is_cell(c)]
        if cells:
            row_cells = [_cell_text(cell) for cell in cells]
        else:
            text = _cell_text(row)
            row_cells = [text] if text else []
        if row_cells:
            rows.append(row_cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _cell_text(node: dict) -> str:
    """Turn a cell's content into one line of table text.

    Args:
        node: The cell node, or a row used as a single cell.

    Returns:
        The cell text, with pipes escaped.
    """
    markdown = "\n".join(_node_block(child, 1)[2] for child in node.get("children", []))
    flat = " ".join(markdown.split())
    return flat.replace("|", "\\|")


def _inline_fragment(node: dict | str, depth: int) -> str:
    """Render a node as one line of inline markdown.

    Args:
        node: The node to render.
        depth: How deep the node sits in the tree.

    Returns:
        The node as a single line.
    """
    if _is_text_node(node):
        return _collapse(_text_of(node))
    if not isinstance(node, dict):
        return ""
    role = node.get("role", "")
    if role == "link":
        return f"[{_escape_link_name(node.get('name', ''))}]({node.get('url', '')})"
    if role == "image":
        return f"![{node.get('name', '')}]({node.get('src', '')})"
    return " ".join(_convert_html_node(node, depth).split())


def _is_cell(node: dict | str) -> bool:
    """Say whether a node is a table cell.

    Args:
        node: One node of the tree.

    Returns:
        True when the node is a cell.
    """
    return isinstance(node, dict) and node.get("role") == "cell"


def _is_text_node(node: dict | str) -> bool:
    """Say whether a node holds nothing but text.

    Args:
        node: One node of the tree.

    Returns:
        True when the node is a piece of text.
    """
    if isinstance(node, str):
        return True
    return isinstance(node, dict) and "text" in node and len(node) == 1


def _text_of(node: dict | str) -> str:
    """Read the text out of a text node.

    Args:
        node: A piece of text or a text dict.

    Returns:
        The text.
    """
    if isinstance(node, str):
        return node
    return node.get("text", "")


def _collapse(text: str) -> str:
    """Squash whitespace runs to single spaces and trim the ends.

    Args:
        text: The text to squash.

    Returns:
        The squashed text.
    """
    return re.sub(r"\s+", " ", text).strip()


def _same_text(node: dict | str, name: str) -> bool:
    """Say whether a text child repeats the node's name.

    Args:
        node: One node of the tree.
        name: The name to compare with.

    Returns:
        True when the child is text equal to the name.
    """
    if not name or not _is_text_node(node):
        return False
    return _collapse(_text_of(node)) == name


def _direct_text(node: dict) -> str:
    """Join the plain text children of a node.

    Args:
        node: The node to read.

    Returns:
        The node's own text, or an empty string.
    """
    parts: list[str] = []
    for child in node.get("children", []):
        if _is_text_node(child):
            text = _collapse(_text_of(child))
            if text:
                parts.append(text)
    return " ".join(parts)


def _escape_link_name(name: str) -> str:
    """Escape the bracket characters in a link name.

    Args:
        name: The link name to escape.

    Returns:
        The name with brackets escaped.
    """
    return name.replace("[", "\\[").replace("]", "\\]")


def _html_tree_to_markdown(content: str) -> str:
    """Convert a JSON HTML tree to markdown.

    Args:
        content: A string that may hold a JSON HTML tree.

    Returns:
        Markdown when the content is a JSON HTML tree, otherwise the content unchanged.
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
    if isinstance(parsed, dict) and "role" in parsed:
        return _convert_html_node(parsed)
    return content


async def html_tree_to_markdown(tree: dict | str) -> str:
    """Turn a page tree into markdown.

    Args:
        tree: The page tree to convert.

    Returns:
        The page content as markdown.
    """
    return await asyncio.to_thread(_convert_html_node, tree)


async def answer_or_content(prompt: str, processed: bool, content: str, target: str) -> ToolResult:
    """Answer a prompt about the content, or give back the content as text.

    Args:
        prompt: A question about the content. Empty asks for the content itself.
        processed: Whether a model already worked on the content.
        content: The content to turn into text.
        target: The name of the source shown in the error message.

    Returns:
        The answer as markdown when a prompt was given, else the content as text.

    Raises:
        RuntimeError: If the answer to the prompt fails.
    """
    content = _html_tree_to_markdown(content)
    if prompt and not processed:
        llm_response, llm_status = await reason(prompt, context=content)
        if llm_status == 0:
            return ToolResult(text=llm_response, render_type="markdown")
        raise RuntimeError(f"Error extracting from {target}: {llm_response}")
    return ToolResult(text=content)


def format_results(
    provider: str,
    query: str,
    results: list[dict[str, str]],
    description_key: str,
) -> str:
    """Format search results as a markdown string.

    Args:
        provider: Name shown in the header.
        query: The search query.
        results: Result dicts with title, url, and description fields.
        description_key: Key holding the description text in each result dict.

    Returns:
        The results as markdown, or an empty string when there are none.
    """
    if not results:
        return ""

    lines = [f"## {provider} Search Results for: {query}", ""]
    lines.append(f"Found {len(results)} results:")
    lines.append("")
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r.get('title', '')}**")
        lines.append(f"   URL: {r.get('url', '')}")
        description = r.get(description_key)
        if description:
            lines.append(f"   {description}")
        lines.append("")
    return "\n".join(lines)
