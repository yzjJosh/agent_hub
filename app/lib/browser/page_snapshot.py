"""A frozen picture of a web page that you can read and act on."""
from __future__ import annotations

__all__ = ["PageSnapshot"]

import asyncio
import hashlib
import json
import os
import tempfile
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

from bs4 import BeautifulSoup, Tag

from app.lib.browser.snapshot_filter import html_to_json, filter_html_file
from app.lib.shared import paths
from app.lib.shared.tokens import count_tokens

if TYPE_CHECKING:
    from app.lib.browser.browser_manager import BrowserManager, Viewport


_SCRATCH_NAME = "page_snapshot"

# Kinds of page parts a person can act on.
_INTERACTIVE_ROLES = frozenset({
    'button', 'link', 'textbox', 'combobox', 'checkbox', 'radio',
    'switch', 'slider', 'tab', 'menuitem', 'treeitem',
})

# Tags that make a page part a person can act on.
_INTERACTIVE_TAGS = frozenset({
    'a', 'button', 'input', 'textarea', 'select', 'details', 'summary',
    'option', 'menuitem',
})


def _find_shared_ids(soup: BeautifulSoup) -> Set[str]:
    """Find the ids that more than one page part uses.

    Args:
        soup: The page to look through.

    Returns:
        The ids used by two or more parts.
    """
    counts = Counter(element['id'] for element in soup.find_all(id=True))
    return {elem_id for elem_id, count in counts.items() if count > 1}


def _build_selector_for_element(element: Tag, shared_ids: Set[str]) -> Optional[str]:
    """Make a selector that points at one page part.

    Args:
        element: The page part.
        shared_ids: Ids that more than one page part uses.

    Returns:
        The selector, or None if the part cannot be pointed at.
    """
    if not element.name:
        return None

    elem_id = element.get('id')
    if elem_id and elem_id not in shared_ids:
        return f'#{elem_id}'

    tag = element.name
    attrs = _collect_selector_attrs(element)

    if attrs:
        base_selector = tag + '[' + ']['.join(attrs[:2]) + ']'
    else:
        base_selector = tag

    text = element.get_text(strip=True)
    if text and len(text) < 100:
        return base_selector + ':has-text("' + _escape_selector_text(text) + '")'

    return base_selector


def _collect_selector_attrs(element: Tag) -> List[str]:
    """Gather the bits of a selector that describe a page part.

    Args:
        element: The page part.

    Returns:
        The pieces to put in a selector.
    """
    attrs = []
    aria_label = element.get('aria-label')
    if aria_label:
        attrs.append(f'aria-label*="{aria_label.replace('"', '\\"')}"')
    input_type = element.get('type')
    if input_type and element.name == 'input':
        attrs.append(f'type="{input_type}"')
    role = element.get('role')
    if role:
        attrs.append(f'role="{role}"')
    href = element.get('href')
    if href and element.name == 'a':
        attrs.append(f'href*="{href[:50]}"')
    name_attr = element.get('name')
    if name_attr:
        attrs.append(f'name="{name_attr}"')
    placeholder = element.get('placeholder')
    if placeholder:
        attrs.append(f'placeholder*="{placeholder.replace('"', '\\"')}"')
    return attrs


def _escape_selector_text(text: str) -> str:
    """Make text safe to put inside a selector.

    Args:
        text: The text to fix up.

    Returns:
        The safe text.
    """
    return json.dumps(text, ensure_ascii=False)[1:-1]


def _snapshot_prefix(html: str) -> str:
    """Make a short tag for one page's ref ids.

    The same page text always gets the same tag.

    Args:
        html: The page text.

    Returns:
        The tag.
    """
    return hashlib.md5(html.encode('utf-8', errors='replace')).hexdigest()[:4]


_TAG_ROLE_MAP = {
    'a': 'link', 'button': 'button', 'input': 'textbox',
    'textarea': 'textbox', 'select': 'combobox',
    'details': 'group', 'summary': 'button',
    'option': 'listitem', 'menuitem': 'menuitem',
}


def _resolve_role(element: Tag) -> str:
    """Work out what kind of page part this is.

    Args:
        element: The page part.

    Returns:
        The kind, or an empty string if it is not known.
    """
    role = element.get('role', '') or ''
    if not role:
        role = _TAG_ROLE_MAP.get(element.name, '')
    return role


def _resolve_element_name(element: Tag) -> str:
    """Work out a name to show for a page part.

    Args:
        element: The page part.

    Returns:
        The name, or an empty string if there is none.
    """
    name = (
        element.get('aria-label')
        or element.get('title')
        or element.get('alt')
        or element.get('placeholder')
        or element.get('name')
        or ''
    )
    if not name and element.name in ('a', 'button', 'summary'):
        text = element.get_text(strip=True)
        if text:
            name = text[:200]
    if not name and element.name == 'input':
        input_type = element.get('type', 'text')
        name = f'{input_type} input'
    if not name and element.name == 'select':
        name = 'dropdown'
    if not name and element.name == 'textarea':
        name = 'text area'
    return name


def _register_element(
    elements: List[Dict], counter: int, prefix: str,
    element: Tag, role: str, name: str, shared_ids: Set[str]
) -> int:
    """Add one page part to the list, if it can be pointed at.

    Args:
        elements: The list to add to.
        counter: Number to use in the ref id.
        prefix: Tag that ties the ref id to one snapshot.
        element: The page part to add.
        role: What kind of part it is.
        name: The name to show for it.
        shared_ids: Ids that more than one page part uses.

    Returns:
        The next number to use.
    """
    selector = _build_selector_for_element(element, shared_ids)
    if not selector:
        return counter
    ref_id = f'e{prefix}_{counter}'
    elements.append({
        'ref_id': ref_id,
        'role': role,
        'name': name,
        'selector': selector,
    })
    return counter + 1


def _is_hidden(element: Tag) -> bool:
    """Say if a page part is hidden from people.

    Args:
        element: The page part.

    Returns:
        True if the part is hidden.
    """
    if element.get('hidden') is not None:
        return True
    style = element.get('style', '')
    clean_style = style.replace(' ', '').lower()
    return 'display:none' in clean_style or 'visibility:hidden' in clean_style


def _process_interactive_tags(soup: BeautifulSoup, elements: List[Dict], counter: int, prefix: str, shared_ids: Set[str]) -> int:
    """Add the page parts whose tag makes them clickable or typable.

    Args:
        soup: The page to look through.
        elements: The list to add to.
        counter: Number to use in the next ref id.
        prefix: Tag that ties the ref ids to one snapshot.
        shared_ids: Ids that more than one page part uses.

    Returns:
        The next number to use.
    """
    for element in soup.find_all(_INTERACTIVE_TAGS):
        if _is_hidden(element):
            continue
        if element.name == 'input':
            input_type = (element.get('type') or 'text').lower()
            if input_type in ('hidden', 'submit', 'reset'):
                continue
        role = _resolve_role(element)
        if role and role not in _INTERACTIVE_ROLES and role not in ('group', 'listitem', 'menuitem'):
            if element.name not in _INTERACTIVE_TAGS:
                continue
        name = _resolve_element_name(element)
        counter = _register_element(elements, counter, prefix, element, role, name, shared_ids)
    return counter


def _process_aria_roles(soup: BeautifulSoup, elements: List[Dict], counter: int, prefix: str, shared_ids: Set[str]) -> int:
    """Add page parts that say they are clickable or typable.

    Parts whose tag already makes them clickable are skipped.

    Args:
        soup: The page to look through.
        elements: The list to add to.
        counter: Number to use in the next ref id.
        prefix: Tag that ties the ref ids to one snapshot.
        shared_ids: Ids that more than one page part uses.

    Returns:
        The next number to use.
    """
    for element in soup.find_all(attrs={'role': True}):
        if element.name in _INTERACTIVE_TAGS:
            continue
        role = element.get('role', '')
        if role not in _INTERACTIVE_ROLES:
            continue
        if _is_hidden(element):
            continue
        name = (
            element.get('aria-label')
            or element.get('title')
            or element.get_text(strip=True)[:200]
            or ''
        )
        counter = _register_element(elements, counter, prefix, element, role, name, shared_ids)
    return counter


def _process_clickable_divs(soup: BeautifulSoup, elements: List[Dict], counter: int, prefix: str, shared_ids: Set[str]) -> int:
    """Add plain boxes that look clickable.

    Args:
        soup: The page to look through.
        elements: The list to add to.
        counter: Number to use in the next ref id.
        prefix: Tag that ties the ref ids to one snapshot.
        shared_ids: Ids that more than one page part uses.

    Returns:
        The next number to use.
    """
    for element in soup.find_all('div'):
        if _is_hidden(element):
            continue
        if element.get('role') in _INTERACTIVE_ROLES:
            continue
        has_tabindex = element.get('tabindex') is not None
        has_onclick = element.get('onclick') is not None
        style = element.get('style', '')
        has_cursor_pointer = isinstance(style, str) and 'cursor' in style and 'pointer' in style
        if not (has_tabindex or has_onclick or has_cursor_pointer):
            continue
        text = element.get_text(strip=True)
        if not text or len(text) > 200:
            continue
        counter = _register_element(elements, counter, prefix, element, 'button', text[:200], shared_ids)
    return counter


def _extract_interactive_elements(html: str, counter_start: int = 1, prefix: str = '') -> tuple:
    """Find the page parts a person can act on.

    Args:
        html: The page text.
        counter_start: First number to use in a ref id.
        prefix: Tag that ties the ref ids to one snapshot.

    Returns:
        The parts that were found, and the next number to use. That number
        does not count the plain boxes that look clickable.
    """
    soup = BeautifulSoup(html, 'html.parser')

    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()

    shared_ids = _find_shared_ids(soup)
    elements = []
    counter = _process_interactive_tags(soup, elements, counter_start, prefix, shared_ids)
    counter = _process_aria_roles(soup, elements, counter, prefix, shared_ids)
    _process_clickable_divs(soup, elements, counter, prefix, shared_ids)

    return elements, counter


def _scan_actionable_elements(html: str) -> List[Dict[str, str]]:
    """Find the page parts a person can act on.

    Args:
        html: The page text.

    Returns:
        The parts that were found, each with a ref id that belongs to this
        page text only.
    """
    prefix = _snapshot_prefix(html)
    elements, _ = _extract_interactive_elements(html, counter_start=1, prefix=prefix)
    return elements


def _write_temp_html(html: str) -> str:
    """Put page text in a new temporary file.

    Args:
        html: The page text.

    Returns:
        The path of the file.
    """
    scratch = paths.scratch_dir(_SCRATCH_NAME)
    os.makedirs(scratch, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".html", dir=scratch)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(html)
    return tmp_path


def _remove_file(path: str) -> None:
    """Delete a file, and do nothing when it is already gone.

    Args:
        path: The file to delete.
    """
    try:
        os.unlink(path)
    except OSError:
        pass


class PageSnapshot:
    """A frozen picture of a web page.

    Each page part a person can act on gets a ref id. A ref id works only
    with the snapshot it came from.
    """

    def __init__(self, html: str, mgr: BrowserManager, target_id: str, url: str = "", title: str = "", viewport: Optional[Viewport] = None) -> None:
        """Save the page text, without reading it.

        Use create to get a snapshot that is ready to use.

        Args:
            html: The whole page text.
            mgr: The browser that acts on the page.
            target_id: The tab the page is in.
            url: The address of the page.
            title: The title of the page.
            viewport: The window's size and place on the page, or None.
        """
        self._html = html or ""
        self._mgr = mgr
        self._target_id = target_id
        self._url = url or ""
        self._title = title or ""
        self._viewport = viewport
        self._elements: List[Dict[str, str]] = []
        self._ref_map: Dict[str, Dict[str, str]] = {}
        self._filtered_tree: Optional[Union[dict, str]] = None

    @classmethod
    async def create(cls, html: str, mgr: BrowserManager, target_id: str, url: str = "", title: str = "", viewport: Optional[Viewport] = None) -> PageSnapshot:
        """Make a snapshot of a page.

        Args:
            html: The whole page text.
            mgr: The browser that acts on the page.
            target_id: The tab the page is in.
            url: The address of the page.
            title: The title of the page.
            viewport: The window's size and place on the page, or None.

        Returns:
            A snapshot ready to use.
        """
        snapshot = cls(html, mgr, target_id, url=url, title=title, viewport=viewport)
        await snapshot._scan()
        return snapshot

    async def _scan(self) -> None:
        """Read the page and list the parts a person can act on."""
        self._elements = await asyncio.to_thread(_scan_actionable_elements, self._html)
        for elem in self._elements:
            self._ref_map[elem['ref_id']] = {
                'selector': elem['selector'],
                'role': elem['role'],
                'name': elem['name'],
            }

    @property
    def html(self) -> str:
        """The whole page text."""
        return self._html

    @property
    def url(self) -> str:
        """The address of the page when the snapshot was taken."""
        return self._url

    @property
    def title(self) -> str:
        """The title of the page when the snapshot was taken."""
        return self._title

    @property
    def viewport(self) -> Optional[Viewport]:
        """The window's size and place on the page when the snapshot was taken."""
        return self._viewport

    async def filtered_tree(self, verbose: bool = False) -> Optional[Union[dict, str]]:
        """A small tree of the page, without ref ids.

        A rough tree is given back if the page cannot be trimmed.

        Args:
            verbose: Print how much the page shrank when True.

        Returns:
            The tree, or None if the page could not be read at all.
        """
        tree = await self._build_filtered_tree()
        if tree is not None and verbose:
            await asyncio.to_thread(self._log_token_stats, self._html, tree)
        return tree

    async def _build_filtered_tree(self) -> Optional[Union[dict, str]]:
        """Trim the page, or give back the tree that was trimmed before.

        Returns:
            The tree, or None if the page could not be read at all.
        """
        if self._filtered_tree is not None:
            return self._filtered_tree

        if not self._html:
            return None

        result = await self._try_filter_html()
        if result is not None:
            self._filtered_tree = result
            return result

        try:
            result = await html_to_json(self._html)
        except Exception:
            return None
        if result:
            self._filtered_tree = result
            return result

        return None

    async def _try_filter_html(self) -> Optional[Union[dict, str]]:
        """Trim the page down to the parts that matter.

        Returns:
            The trimmed tree, or None if it could not be made.
        """
        tmp_path = None
        try:
            tmp_path = await asyncio.to_thread(_write_temp_html, self._html)
            return await filter_html_file(tmp_path)
        except Exception:
            return None
        finally:
            if tmp_path:
                await asyncio.to_thread(_remove_file, tmp_path)

    def _log_token_stats(self, original_html: str, filtered_tree: Union[dict, str]) -> None:
        """Print how much the page shrank.

        Args:
            original_html: The page text before.
            filtered_tree: The tree after.
        """
        original_tokens = count_tokens(original_html)
        filtered_str = json.dumps(filtered_tree, ensure_ascii=False)
        filtered_tokens = count_tokens(filtered_str)
        if original_tokens > 0:
            change = (1 - filtered_tokens / original_tokens) * 100
            direction = "reduction" if change > 0 else ("increase" if change < 0 else "change")
            pct = f"{abs(change):.1f}% {direction}"
        else:
            pct = "N/A"
        print(
            f"[PageSnapshot.filtered_tree] Original: {original_tokens} tokens → Filtered: {filtered_tokens} tokens ({pct})",
            flush=True
        )

    def actionable_elements(self) -> List[Dict[str, str]]:
        """The page parts a person can act on.

        Returns:
            One entry per part, each with a ref_id, role, and name.
        """
        return [
            {'ref_id': elem['ref_id'], 'role': elem['role'], 'name': elem['name']}
            for elem in self._elements
        ]

    def _resolve_ref(self, ref_id: str) -> str:
        """Find the selector that a ref id stands for.

        Args:
            ref_id: The ref id to look up.

        Returns:
            The selector.

        Raises:
            ValueError: If this snapshot has no such ref id.
        """
        if ref_id not in self._ref_map:
            raise ValueError(
                f"Unknown ref_id '{ref_id}'. Valid ref_ids: "
                f"{list(self._ref_map.keys())[:10]}"
            )
        return self._ref_map[ref_id]['selector']

    async def click(self, ref_id: str, timeout: int = 30000, verbose: bool = False, human_config: Optional[dict] = None) -> bool:
        """Click a page part.

        Args:
            ref_id: The part to click.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.
            human_config: Settings for how human the click should look.

        Returns:
            True if the click was made.
        """
        selector = self._resolve_ref(ref_id)
        if human_config is None:
            return await self._mgr.click(self._target_id, selector, timeout=timeout, verbose=verbose)
        return await self._mgr.click(self._target_id, selector, timeout=timeout, verbose=verbose, human_config=human_config)

    async def fill(self, ref_id: str, text: str, submit: bool = False, timeout: int = 30000, verbose: bool = False, human_config: Optional[dict] = None) -> bool:
        """Type text into a page part.

        Args:
            ref_id: The part to type into.
            text: What to type.
            submit: Press Enter after typing when True.
            timeout: How long to wait, in milliseconds.
            verbose: Print progress lines when True.
            human_config: Settings for how human the typing should look.

        Returns:
            True if the text was typed.
        """
        selector = self._resolve_ref(ref_id)
        if human_config is None:
            return await self._mgr.type_text(self._target_id, selector, text, submit=submit, timeout=timeout, verbose=verbose)
        return await self._mgr.type_text(self._target_id, selector, text, submit=submit, timeout=timeout, verbose=verbose, human_config=human_config)

    async def upload(self, ref_id: str, file_paths: list, verbose: bool = False) -> bool:
        """Give files to a file box on the page.

        Args:
            ref_id: The file box to fill.
            file_paths: Paths of the files to send.
            verbose: Print progress lines when True.

        Returns:
            True if the files were sent.
        """
        selector = self._resolve_ref(ref_id)
        return await self._mgr.upload_files(self._target_id, selector, file_paths, verbose=verbose)

    async def upload_with_chooser_relative(
        self, x: float, y: float, file_paths: list[str], verbose: bool = False,
    ) -> bool:
        """Give files to a picker opened at a spot on the screen.

        Args:
            x: How far across the screen, from zero to one.
            y: How far down the screen, from zero to one.
            file_paths: Paths of the files to send.
            verbose: Print progress lines when True.

        Returns:
            True if the files were sent.
        """
        return await self._mgr.upload_with_chooser_relative(
            self._target_id, x, y, file_paths, viewport=self.viewport, verbose=verbose,
        )

    async def select_option(self, ref_id: str, value: str = None, index: int = None, label: str = None, verbose: bool = False) -> bool:
        """Pick a choice in a drop down list.

        Give only one of value, index, or label.

        Args:
            ref_id: The drop down list to use.
            value: The value of the choice to pick.
            index: The place of the choice in the list.
            label: The words shown for the choice.
            verbose: Print progress lines when True.

        Returns:
            True if the choice was picked.
        """
        selector = self._resolve_ref(ref_id)
        return await self._mgr.select_option(self._target_id, selector, value=value, index=index, label=label, verbose=verbose)

    async def press_key(self, key: str = "Enter", verbose: bool = False) -> bool:
        """Press one key on the keyboard.

        Args:
            key: The name of the key to press.
            verbose: Print progress lines when True.

        Returns:
            True if the key was pressed.
        """
        return await self._mgr.press_key(self._target_id, key, verbose=verbose)

    async def click_at(self, x: float, y: float, verbose: bool = False) -> bool:
        """Click a spot on the page.

        The spot is measured on the whole page, not on the window.

        Args:
            x: How far right, counted from the left edge of the page.
            y: How far down, counted from the top of the page.
            verbose: Print progress lines when True.

        Returns:
            True if the click was made.
        """
        return await self._mgr.click_at(self._target_id, x, y, verbose=verbose)

    async def move_mouse_to(self, x: float, y: float, verbose: bool = False) -> bool:
        """Move the mouse to a spot in the window.

        A spot outside the window is pulled back to the edge.

        Args:
            x: How far right, counted from the left edge of the window.
            y: How far down, counted from the top of the window.
            verbose: Print progress lines when True.

        Returns:
            True if the mouse moved.
        """
        return await self._mgr.move_mouse_to(self._target_id, x, y, verbose=verbose)

    async def move_mouse_to_relative(self, x_ratio: float, y_ratio: float, verbose: bool = False) -> bool:
        """Move the mouse to a spot given as a share of the window.

        Args:
            x_ratio: How far across, 0 is the left edge and 1 is the right.
            y_ratio: How far down, 0 is the top and 1 is the bottom.
            verbose: Print progress lines when True.

        Returns:
            True if the mouse moved.
        """
        return await self._mgr.move_mouse_to_relative(self._target_id, x_ratio, y_ratio, viewport=self.viewport, verbose=verbose)

    async def scroll_to(self, x: float, y: float, verbose: bool = False) -> bool:
        """Scroll until a spot on the page is in the window.

        Args:
            x: How far right, counted from the left edge of the page.
            y: How far down, counted from the top of the page.
            verbose: Print progress lines when True.

        Returns:
            True if the page scrolled.
        """
        return await self._mgr.scroll_to(self._target_id, x, y, verbose=verbose)

    async def scroll_to_view(self, ref_id: str, verbose: bool = False) -> bool:
        """Scroll until a page part sits in the middle of the window.

        Args:
            ref_id: The part to scroll to.
            verbose: Print progress lines when True.

        Returns:
            True if the page scrolled.
        """
        selector = self._resolve_ref(ref_id)
        return await self._mgr.scroll_to_view(self._target_id, selector, verbose=verbose)

    async def scroll_by(self, distance: float, verbose: bool = False) -> bool:
        """Scroll the page a set number of pixels.

        Args:
            distance: How far to scroll. A positive number scrolls down.
            verbose: Print progress lines when True.

        Returns:
            True if the page scrolled.
        """
        return await self._mgr.scroll_by(self._target_id, distance, verbose=verbose)

    async def solve_captcha(self, max_steps: int = 5, verbose: bool = False) -> bool:
        """Find and solve a captcha on the page.

        Args:
            max_steps: How many tries to make.
            verbose: Print progress lines when True.

        Returns:
            True if the page has no captcha left.
        """
        return await self._mgr.solve_captcha(self._target_id, max_steps=max_steps, verbose=verbose)

    async def click_at_relative(self, x: float, y: float, verbose: bool = False) -> bool:
        """Click a spot given as a share of the window.

        The spot is where the caller last saw it, counted on the whole page.

        Args:
            x: How far across, 0 is the left edge and 1 is the right.
            y: How far down, 0 is the top and 1 is the bottom.
            verbose: Print progress lines when True.

        Returns:
            True if the click was made.
        """
        return await self._mgr.click_at_relative(self._target_id, x, y, viewport=self.viewport, verbose=verbose)

    async def scroll_by_relative(self, dx: float, dy: float, verbose: bool = False) -> bool:
        """Scroll the page a share of the window in each direction.

        Args:
            dx: How far sideways, 1 is one window wide. A positive number scrolls right.
            dy: How far up or down, 1 is one window tall. A positive number scrolls down.
            verbose: Print progress lines when True.

        Returns:
            True if the page scrolled.
        """
        return await self._mgr.scroll_by_relative(self._target_id, dx, dy, viewport=self.viewport, verbose=verbose)

    async def type_text_at(self, x: float, y: float, text: str, submit: bool = False, verbose: bool = False) -> bool:
        """Type text at a spot on the page.

        The spot is clicked first to give it focus. The text is added to
        what is there, nothing is cleared.

        Args:
            x: How far right, counted from the left edge of the page.
            y: How far down, counted from the top of the page.
            text: What to type.
            submit: Press Enter after typing when True.
            verbose: Print progress lines when True.

        Returns:
            True if the text was typed.
        """
        return await self._mgr.type_text_at(self._target_id, x, y, text, submit=submit, verbose=verbose)

    async def type_text_at_relative(self, x: float, y: float, text: str, submit: bool = False, verbose: bool = False) -> bool:
        """Type text at a spot given as a share of the window.

        The spot is where the caller last saw it, counted on the whole page.

        Args:
            x: How far across, 0 is the left edge and 1 is the right.
            y: How far down, 0 is the top and 1 is the bottom.
            text: What to type.
            submit: Press Enter after typing when True.
            verbose: Print progress lines when True.

        Returns:
            True if the text was typed.
        """
        return await self._mgr.type_text_at_relative(self._target_id, x, y, text, submit=submit, viewport=self.viewport, verbose=verbose)

    async def replace_input_text_at(self, x: float, y: float, old_text: str, new_text: str, verbose: bool = False) -> bool:
        """Replace a piece of the text in the field at a spot on the page.

        The spot is clicked first to give the field focus. The old text must
        show up in the field exactly once.

        Args:
            x: How far right, counted from the left edge of the page.
            y: How far down, counted from the top of the page.
            old_text: The text in the field to replace.
            new_text: The text to put in its place. Empty removes the old text.
            verbose: Print progress lines when True.

        Returns:
            True if the field now holds the new text in place of the old.
        """
        return await self._mgr.replace_input_text_at(self._target_id, x, y, old_text, new_text, verbose=verbose)

    async def replace_input_text_at_relative(self, x: float, y: float, old_text: str, new_text: str, verbose: bool = False) -> bool:
        """Replace a piece of the text in the field at a spot given as a share of the window.

        The spot is where the caller last saw it, counted on the whole page.

        Args:
            x: How far across, 0 is the left edge and 1 is the right.
            y: How far down, 0 is the top and 1 is the bottom.
            old_text: The text in the field to replace.
            new_text: The text to put in its place. Empty removes the old text.
            verbose: Print progress lines when True.

        Returns:
            True if the text was replaced.
        """
        return await self._mgr.replace_input_text_at_relative(
            self._target_id, x, y, old_text, new_text, viewport=self.viewport, verbose=verbose
        )

    async def go_forward(self, verbose: bool = False) -> bool:
        """Go forward to the page the tab had been on.

        Args:
            verbose: Print progress lines when True.

        Returns:
            True if the newer page opened.
        """
        return await self._mgr.go_forward(self._target_id, verbose=verbose)

    async def go_back(self, verbose: bool = False) -> bool:
        """Go back to the page the tab was on before.

        Args:
            verbose: Print progress lines when True.

        Returns:
            True if the older page opened.
        """
        return await self._mgr.go_back(self._target_id, verbose=verbose)

    async def close_page(self, verbose: bool = False) -> bool:
        """Close the tab and forget about it.

        The tab is forgotten even when closing it fails or takes too long,
        so this never hangs.

        Args:
            verbose: Print progress lines when True.

        Returns:
            True if the tab closed.
        """
        return await self._mgr.close_page(self._target_id, verbose=verbose)
