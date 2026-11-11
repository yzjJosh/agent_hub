"""Turn a web page into a small tree of the parts that matter."""

import asyncio
import json
import re
import sys
from typing import Union, List, Optional, Set, Tuple

from bs4 import BeautifulSoup
from bs4 import Comment
from bs4 import NavigableString, Tag
from app.lib.shared.tokens import count_tokens

# A page part is kept when its text is at least this long.
_SUBSTANTIAL_TEXT_THRESHOLD = 50
# A list item is dropped when its text is shorter than this.
_SHORT_LIST_ITEM_THRESHOLD = 10

_FILTER_ROLES = frozenset(
    {
        "separator",
        "decoration",
        "presentation",
        "none",
        "banner",
        "contentinfo",
        "complementary",
        "navigation",
        "footer",
    }
)

_CONTENT_ROLES = frozenset(
    {
        "heading",
        "paragraph",
        "link",
        "button",
        "listitem",
        "list",
        "article",
        "main",
        "region",
        "document",
        "text",
        "textbox",
        "combobox",
        "checkbox",
        "radio",
        "label",
    }
)

_TAG_TO_ROLE = {
    "h1": "heading",
    "h2": "heading",
    "h3": "heading",
    "h4": "heading",
    "h5": "heading",
    "h6": "heading",
    "p": "paragraph",
    "a": "link",
    "button": "button",
    "ul": "list",
    "ol": "list",
    "li": "listitem",
    "article": "article",
    "main": "main",
    "nav": "navigation",
    "header": "banner",
    "footer": "contentinfo",
    "aside": "complementary",
    "form": "form",
    "search": "search",
    "section": "region",
    "table": "table",
    "tr": "row",
    "td": "cell",
    "th": "cell",
    "input": "textbox",
    "textarea": "textbox",
    "select": "combobox",
    "option": "listitem",
    "label": "label",
    "details": "group",
    "summary": "button",
    "img": "image",
}

__all__ = ["html_to_json", "filter_html_file"]


def _extract_text_from_element(element: dict) -> str:
    """Get the text of a page part.

    Args:
        element: The page part to read.

    Returns:
        The text, or an empty string.
    """
    return (element.get("text", "") or element.get("name", "") or "").strip()


def _should_filter_by_role(role: str, text: str, has_url: bool = False) -> bool:
    """Say if a page part should be dropped.

    Args:
        role: What kind of part it is.
        text: The text inside the part.
        has_url: True if the part points somewhere.

    Returns:
        True if the part should be dropped.
    """
    always_filter = {"separator", "decoration", "presentation", "none"}
    if role in always_filter:
        return True

    if role == "link" and has_url:
        return False

    text_length = len(text)

    if role in _FILTER_ROLES:
        return text_length < _SUBSTANTIAL_TEXT_THRESHOLD

    if role in {"list", "listitem"} and text_length < _SHORT_LIST_ITEM_THRESHOLD:
        return True

    return False


def _is_duplicate_text(text: str, seen_texts: Set[str], url: str = None) -> bool:
    """Say if this text was seen before, and remember it.

    Text that goes with different addresses counts as different text.

    Args:
        text: The text to check.
        seen_texts: What has been seen so far, added to here.
        url: Where the text points, if anywhere.

    Returns:
        True if the same text was seen before.
    """
    if not text:
        return False

    key = f"{text}|{url}" if url else text

    if key in seen_texts:
        return True
    seen_texts.add(key)
    return False


def _is_text_node(node: Union[str, dict]) -> Tuple[bool, Optional[str]]:
    """Say if a node is plain text, and give the text.

    Args:
        node: The node to look at.

    Returns:
        True and the text if it is plain text, False and None if not.
    """
    if isinstance(node, str):
        return True, node
    if isinstance(node, dict):
        if node.get("role") == "text":
            return True, node.get("text", "")
        if len(node) == 1 and "text" in node:
            return True, node["text"]
    return False, None


def _flush_text_to_merged(
    merged: List, current_text: List[str]
) -> None:
    """Move saved up text into the output list.

    Args:
        merged: List the text is added to.
        current_text: Text pieces to move, emptied here.
    """
    if current_text:
        merged_text = " ".join(current_text)
        if merged_text:
            merged.append({"text": merged_text})
    current_text.clear()


def _merge_consecutive_texts(nodes: List) -> List:
    """Join text nodes that sit next to each other.

    Args:
        nodes: The nodes to look at.

    Returns:
        The nodes with neighboring text joined.
    """
    merged = []
    current_text = []

    for node in nodes:
        is_text, text_content = _is_text_node(node)

        if is_text and text_content:
            normalized = " ".join(text_content.split())
            if normalized:
                current_text.append(normalized)
        else:
            _flush_text_to_merged(merged, current_text)
            merged.append(node)

    _flush_text_to_merged(merged, current_text)
    return merged


_HIDDEN_STYLE_RE = re.compile(
    r"(?:display\s*:\s*none|visibility\s*:\s*hidden)",
    re.IGNORECASE,
)


def _remove_hidden_elements(soup: BeautifulSoup) -> None:
    """Drop page parts that people cannot see.

    The tree is changed in place.

    Args:
        soup: The page tree to change.
    """
    hidden = []
    for element in soup.find_all(True):
        if element.get("aria-hidden") == "true":
            hidden.append(element)
            continue
        style = element.get("style")
        if style and _HIDDEN_STYLE_RE.search(style):
            hidden.append(element)
    for element in hidden:
        element.decompose()


def _clean_soup(soup: BeautifulSoup) -> None:
    """Drop page parts that are not content.

    The tree is changed in place.

    Args:
        soup: The page tree to change.
    """
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    _remove_hidden_elements(soup)
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()


def _detect_role_from_class(class_attr: str) -> Optional[str]:
    """Guess what kind of part this is from its class names.

    Args:
        class_attr: The class names, in lower case.

    Returns:
        The kind of part, or None if it cannot be guessed.
    """
    if not class_attr:
        return None
    if any(x in class_attr for x in ["banner", "header", "top-bar"]):
        return "banner"
    if any(x in class_attr for x in ["footer", "bottom-bar"]):
        return "footer"
    if any(x in class_attr for x in ["nav", "navigation", "menu"]):
        return "navigation"
    if any(x in class_attr for x in ["sidebar", "aside", "complementary"]):
        return "complementary"
    if any(x in class_attr for x in ["main", "content"]):
        return "main"
    if any(x in class_attr for x in ["article", "post"]):
        return "article"
    return None


def _extract_node_metadata(element: Tag) -> dict:
    """Pull the name and links out of a page part.

    Args:
        element: The page part to read.

    Returns:
        What was found, with a name, url, or src key when there is one.
    """
    metadata = {}
    name = element.get("aria-label") or element.get("title") or element.get("alt")
    if not name and element.name in (
        "a",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "button",
        "span",
    ):
        name = element.get_text(strip=True) or None
    if name:
        metadata["name"] = name
    href = element.get("href")
    if href:
        metadata["url"] = href
    if element.name == "img":
        src = element.get("src")
        if src:
            metadata["src"] = src
    return metadata


def _parse_element(element: Union[Tag, NavigableString]) -> Optional[Union[dict, str]]:
    """Turn a page part into a tree node.

    Args:
        element: The page part to read.

    Returns:
        A node, a piece of text, or None if the part is skipped.
    """
    if element.name is None:
        text = element.strip()
        return text if text else None

    role = _TAG_TO_ROLE.get(element.name, "generic")

    if role == "generic":
        class_attr = element.get("class", [])
        if isinstance(class_attr, list):
            class_attr = " ".join(class_attr)
        detected = _detect_role_from_class(
            class_attr.lower() if class_attr else ""
        )
        if detected:
            role = detected

    aria_role = element.get("role", "").lower()
    if aria_role in _FILTER_ROLES.union(_CONTENT_ROLES):
        role = aria_role

    node = {"role": role}
    node.update(_extract_node_metadata(element))

    children = []
    for child in element.children:
        child_node = _parse_element(child)
        if child_node:
            children.append(child_node)

    if children:
        node["children"] = _merge_consecutive_texts(children)
    return node


async def html_to_json(html_content: str) -> dict:
    """Turn a web page into a tree.

    Args:
        html_content: The page text.

    Returns:
        The page as a tree.
    """
    return await asyncio.to_thread(_html_to_json, html_content)


def _html_to_json(html_content: str) -> dict:
    """Turn a web page into a tree.

    Args:
        html_content: The page text.

    Returns:
        The page as a tree.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    _clean_soup(soup)

    root_elem = soup.body or soup
    root_node = _parse_element(root_elem)
    return root_node or {}


def _filter_children(children: List, seen_texts: Set[str]) -> List:
    """Drop the child nodes that are not worth keeping.

    Args:
        children: The child nodes to look at.
        seen_texts: Text seen so far, added to here.

    Returns:
        The children that are kept.
    """
    result = []
    for child in children:
        result.extend(_filter_html_node_recursive(child, seen_texts))
    if result:
        result = _merge_consecutive_texts(result)
    return result


def _should_remove_node(
    role: str, text: str, is_duplicate: bool, filtered_children: List, has_url: bool, has_src: bool
) -> Tuple[bool, bool]:
    """Say if a node should be dropped from the tree.

    Args:
        role: What kind of node it is.
        text: The text inside the node.
        is_duplicate: True if the same text was seen before.
        filtered_children: The children that are kept.
        has_url: True if the node points somewhere.
        has_src: True if the node has a picture.

    Returns:
        Whether to drop the node, and whether to keep its children in its
        place.
    """
    if role == "generic" and not text:
        if filtered_children:
            return True, True
        return True, False

    if _should_filter_by_role(role, text, has_url):
        if filtered_children:
            return True, True
        return True, False

    if is_duplicate and not filtered_children and not text:
        return True, False

    if not text and not filtered_children and not has_url and not has_src:
        return True, False

    return False, False


def _filter_html_node_recursive(
    node: Union[dict, str], seen_texts: Set[str]
) -> list:
    """Drop the parts of a node and its children that do not matter.

    The node is changed in place.

    Args:
        node: The node to work on.
        seen_texts: Text seen so far, added to here.

    Returns:
        The node in a list, the list of nodes that take its place, or an
        empty list if the node is dropped.
    """
    if isinstance(node, str):
        return [node]

    role = node.get("role", "unknown")
    text = _extract_text_from_element(node)
    has_url = bool(node.get("url"))
    has_src = bool(node.get("src"))
    url = node.get("url") if has_url else None
    is_duplicate = _is_duplicate_text(text, seen_texts, url)

    filtered_children = _filter_children(node.get("children", []), seen_texts)
    if filtered_children:
        node["children"] = filtered_children
    elif "children" in node:
        del node["children"]

    remove, unwrap = _should_remove_node(
        role, text, is_duplicate, filtered_children, has_url, has_src
    )
    if remove:
        if unwrap and filtered_children:
            return filtered_children
        return []
    return [node]


def _wrap_multiple_roots(filtered_tree: list) -> Optional[Union[dict, str]]:
    """Make sure the tree has one top node.

    Args:
        filtered_tree: The top nodes of the tree.

    Returns:
        The tree with one top node, or None if it is empty.
    """
    if len(filtered_tree) == 1:
        return filtered_tree[0]
    if filtered_tree:
        return {"role": "document", "children": filtered_tree}
    return None


def _print_verbose_stats(html_content: str, filtered_tree: Union[dict, str, None], verbose: bool) -> None:
    """Print how much the page shrank.

    Args:
        html_content: The page text before.
        filtered_tree: The tree after.
        verbose: Print nothing unless True.
    """
    if not verbose or not filtered_tree:
        return
    orig_tokens = count_tokens(html_content)
    res_tokens = count_tokens(json.dumps(filtered_tree, ensure_ascii=False))
    if orig_tokens > 0:
        change = (orig_tokens - res_tokens) / orig_tokens * 100
        direction = "reduction" if change > 0 else ("increase" if change < 0 else "change")
        print(
            f"HTML filtering: {orig_tokens} → {res_tokens} tokens ({abs(change):.0f}% {direction})",
            file=sys.stderr,
        )


async def filter_html_file(html_path: str, verbose: bool = False) -> Union[str, dict, None]:
    """Read a web page from a file and keep only the parts that matter.

    Args:
        html_path: Path to the page file.
        verbose: Print how much the page shrank when True.

    Returns:
        The trimmed tree, or None if the page could not be read or has
        nothing to show.
    """
    return await asyncio.to_thread(_filter_html_file, html_path, verbose)


def _filter_html_file(html_path: str, verbose: bool = False) -> Union[str, dict, None]:
    """Read a web page from a file and keep only the parts that matter.

    Args:
        html_path: Path to the page file.
        verbose: Print how much the page shrank when True.

    Returns:
        The trimmed tree, or None if the page could not be read or has
        nothing to show.
    """
    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        json_tree = _html_to_json(html_content)
        if not json_tree:
            return None

        seen_texts: Set[str] = set()
        filtered_tree = _filter_html_node_recursive(json_tree, seen_texts)
        filtered_tree = _wrap_multiple_roots(filtered_tree)

        _print_verbose_stats(html_content, filtered_tree, verbose)
        return filtered_tree

    except Exception as e:
        if verbose:
            print(f"HTML extraction failed: {e}", file=sys.stderr)
        return None