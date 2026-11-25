"""Helpers for reading nodes and text out of a Redfin page snapshot."""

from __future__ import annotations

__all__ = [
    "collect_node_text",
    "find_actionable_elements",
    "find_descendant_text",
    "find_label_sibling_buttons",
    "find_matching_nodes",
    "gather_node_texts",
    "is_listing_card_node",
]

import re
from dataclasses import dataclass

from app.lib.browser.page_snapshot import PageSnapshot
from app.lib.redfin.models import LISTING_CARD_PREFIX

_FILTER_LABELS = frozenset(
    {"Beds", "Baths", "Garage", "Price", "Sqft", "Lot", "Year", "Status"}
)


@dataclass(frozen=True)
class _NodeMatch:
    parent: dict | list
    node: dict


def find_actionable_elements(
    snapshot: PageSnapshot,
    role: str | None = None,
    name: str | None = None,
    name_contains: str | None = None,
    name_prefix: str | None = None,
) -> list[dict]:
    """Find page elements a user can click or type into.

    Args:
        snapshot: Snapshot of the page.
        role: Keep only elements with this role, like button or textbox.
        name: Keep only elements with this exact name.
        name_contains: Keep only elements whose name holds this text, case does
            not matter.
        name_prefix: Keep only elements whose name starts with this text.

    Returns:
        The matching elements.
    """
    results: list[dict] = []
    for element in snapshot.actionable_elements():
        if role and element.get("role") != role:
            continue
        if name is not None and element.get("name") != name:
            continue
        if name_contains and name_contains.lower() not in element.get("name", "").lower():
            continue
        if name_prefix and not element.get("name", "").startswith(name_prefix):
            continue
        results.append(element)
    return results


def find_matching_nodes(
    tree: dict,
    role: str | None = None,
    name_pattern: str | None = None,
    name_prefix: str | None = None,
) -> list[dict]:
    """Find all nodes in a page tree that match the given filters.

    Nodes that hold only text are skipped.

    Args:
        tree: Node to start from.
        role: Keep only nodes with this role.
        name_pattern: Pattern the node name must match.
        name_prefix: Text the node name must start with.

    Returns:
        The matching nodes from the whole tree. Nothing comes back when no
        role, pattern, or prefix is given.
    """
    results: list[dict] = []
    if not isinstance(tree, dict):
        return results
    if "text" in tree and "role" not in tree:
        return results
    is_match = True
    if role and tree.get("role") != role:
        is_match = False
    if name_pattern and not re.search(name_pattern, tree.get("name", "")):
        is_match = False
    if name_prefix and not tree.get("name", "").startswith(name_prefix):
        is_match = False
    if is_match and (role or name_pattern or name_prefix):
        results.append(tree)
    for child in tree.get("children", []):
        results.extend(find_matching_nodes(child, role, name_pattern, name_prefix))
    return results


def _find_node_by_name(node: dict | list, target: str) -> _NodeMatch | None:
    """Find a node by exact name and give back the node with its parent."""
    if isinstance(node, dict):
        for child in node.get("children", []):
            if isinstance(child, dict) and child.get("name") == target:
                return _NodeMatch(parent=node, node=child)
            match = _find_node_by_name(child, target)
            if match is not None:
                return match
    if isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and item.get("name") == target:
                return _NodeMatch(parent=node, node=item)
            match = _find_node_by_name(item, target)
            if match is not None:
                return match
    return None


def _find_label_index(siblings: list, label_node: dict, label_name: str) -> int:
    """Find the position of a label node inside its sibling list."""
    for index, sibling in enumerate(siblings):
        if sibling is label_node:
            return index
    for index, sibling in enumerate(siblings):
        if isinstance(sibling, dict) and sibling.get("name") == label_name:
            return index
    return -1


def _append_unique_name(names: list[str], seen: set[str], value: str) -> None:
    """Add a non-empty name to the list if it has not been added yet."""
    if value and value not in seen:
        names.append(value)
        seen.add(value)


def _collect_sibling_button_names(siblings: list, start_index: int) -> list[str]:
    """Collect button names after a label until the next filter label."""
    button_names: list[str] = []
    seen: set[str] = set()
    for sibling in siblings[start_index:]:
        if not isinstance(sibling, dict):
            continue
        if sibling.get("role") == "button":
            _append_unique_name(button_names, seen, sibling.get("name", ""))
        elif sibling.get("name") in _FILTER_LABELS:
            break
        else:
            text = collect_node_text(sibling)
            if text and ("Any" in text or any(char.isdigit() for char in text)):
                for word in text.split():
                    _append_unique_name(button_names, seen, word)
            for button in find_matching_nodes(sibling, role="button"):
                _append_unique_name(button_names, seen, button.get("name", ""))
    return button_names


def find_label_sibling_buttons(tree: dict, label_name: str) -> list[str]:
    """Find the names of the buttons that sit under a filter label.

    Args:
        tree: Top of the page tree.
        label_name: Name of the label, like Beds or Baths.

    Returns:
        The button names found next to the label.
    """
    match = _find_node_by_name(tree, label_name)
    if match is None:
        return []
    siblings = (
        match.parent.get("children", [])
        if isinstance(match.parent, dict)
        else match.parent
    )
    label_index = _find_label_index(siblings, match.node, label_name)
    if label_index < 0:
        return []
    return _collect_sibling_button_names(siblings, label_index + 1)


def find_descendant_text(node: dict, pattern: str) -> str | None:
    """Find text under a node that matches a pattern.

    Args:
        node: Node to look under.
        pattern: Pattern the text must match.

    Returns:
        The first text that matches, or None.
    """
    if not isinstance(node, dict):
        return None
    for child in node.get("children", []):
        if isinstance(child, dict):
            if "text" in child and "role" not in child:
                if re.search(pattern, child.get("text", "")):
                    return child.get("text", "")
            result = find_descendant_text(child, pattern)
            if result:
                return result
    return None


def is_listing_card_node(node: dict) -> bool:
    """Tell whether a node is a home listing card on the page."""
    return node.get("role") == "button" and node.get("name", "").startswith(
        LISTING_CARD_PREFIX
    )


def gather_node_texts(
    tree: dict | None, should_skip_listing_cards: bool = False
) -> list[str]:
    """Collect the text under a page tree into pieces.

    Args:
        tree: Node to read the text from, or None.
        should_skip_listing_cards: Leave out text inside listing cards.

    Returns:
        The text pieces, in page order.
    """
    texts: list[str] = []

    def walk(node: dict) -> None:
        if not isinstance(node, dict):
            return
        if should_skip_listing_cards and is_listing_card_node(node):
            return
        if "text" in node and "role" not in node:
            texts.append(node.get("text", ""))
        for child in node.get("children", []):
            walk(child)

    if tree is not None:
        walk(tree)
    return texts


def collect_node_text(tree: dict | None) -> str:
    """Put all the text in a page tree into one string.

    Args:
        tree: Node to read the text from, or None.

    Returns:
        The text with spaces between the pieces.
    """
    return " ".join(gather_node_texts(tree))
