"""Edit text inside the currently focused input or contenteditable field."""

from __future__ import annotations

__all__ = ["replace_focused_field_text"]

import sys

from playwright.async_api import Frame, Page

_LOG_PREVIEW_LIMIT = 50

# Find the focused text field and read its plain text.
_FOCUSED_FIELD_JS = """
    const SELECTABLE_INPUT_TYPES = ['text', 'search', 'url', 'tel', 'password'];
    const WHOLE_VALUE_INPUT_TYPES = ['email', 'number'];
    const focusedElement = () => {
        let el = document.activeElement;
        while (el && el.shadowRoot && el.shadowRoot.activeElement) el = el.shadowRoot.activeElement;
        return el;
    };
    const focusedField = () => {
        const el = focusedElement();
        if (!el) return null;
        if (el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && SELECTABLE_INPUT_TYPES.includes(el.type))) {
            return {hasValue: true, selectsRange: true, el};
        }
        if (el.tagName === 'INPUT' && WHOLE_VALUE_INPUT_TYPES.includes(el.type)) {
            return {hasValue: true, selectsRange: false, el};
        }
        if (!el.isContentEditable) return null;
        let host = el;
        while (host.parentElement && host.parentElement.isContentEditable) host = host.parentElement;
        return {hasValue: false, el: host};
    };
    const textNodesOf = (host) => {
        const walker = document.createTreeWalker(host, NodeFilter.SHOW_TEXT);
        const nodes = [];
        while (walker.nextNode()) nodes.push(walker.currentNode);
        return nodes;
    };
    const fieldText = (field) => field.hasValue
        ? field.el.value
        : textNodesOf(field.el).map((node) => node.data).join('').replace(/\\u00a0/g, ' ');
"""

# Return the focused inner frame element, or null when focus is not in one.
_FOCUSED_INNER_FRAME_JS = """() => {
    const el = document.activeElement;
    return el && (el.tagName === 'IFRAME' || el.tagName === 'FRAME') ? el : null;
}"""

# Select the single occurrence of oldText in the focused field, or all text when range selection is unsupported.
_SELECT_FIELD_TEXT_JS = """([oldText, newText]) => {
""" + _FOCUSED_FIELD_JS + """
    const field = focusedField();
    if (!field) return {status: 'not an editable field'};
    const text = fieldText(field);
    const start = text.indexOf(oldText);
    if (start < 0) return {status: 'old text not found'};
    if (text.indexOf(oldText, start + 1) >= 0) return {status: 'old text found more than once'};
    const end = start + oldText.length;
    const expected = text.slice(0, start) + newText + text.slice(end);
    if (field.hasValue && !field.selectsRange) {
        field.el.select();
        return {status: 'ok', expected, selectedAll: true};
    }
    if (field.hasValue) {
        field.el.setSelectionRange(start, end);
    } else {
        const range = document.createRange();
        let offset = 0;
        for (const node of textNodesOf(field.el)) {
            const nodeEnd = offset + node.data.length;
            if (start >= offset && start < nodeEnd) range.setStart(node, start - offset);
            if (end > offset && end <= nodeEnd) range.setEnd(node, end - offset);
            offset = nodeEnd;
        }
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
    }
    return {status: 'ok', expected, selectedAll: false};
}"""

# Read the text of the focused field, or null when no text field has focus.
_READ_FIELD_TEXT_JS = """() => {
""" + _FOCUSED_FIELD_JS + """
    const field = focusedField();
    return field ? fieldText(field) : null;
}"""


def _truncate_for_log(text: str) -> str:
    """Shorten a string for verbose log output."""
    if len(text) > _LOG_PREVIEW_LIMIT:
        return text[:_LOG_PREVIEW_LIMIT] + "..."
    return text


async def _find_focused_frame(page: Page) -> Frame | None:
    """Walk nested frames to find the frame holding the active element."""
    frame = page.main_frame
    while True:
        handle = await frame.evaluate_handle(_FOCUSED_INNER_FRAME_JS)
        element = handle.as_element()
        child = await element.content_frame() if element else None
        await handle.dispose()
        if element is None:
            return frame
        if child is None:
            return None
        frame = child


async def replace_focused_field_text(
    page: Page,
    old_text: str,
    new_text: str,
    verbose: bool = False,
) -> bool:
    """Replace one exact text match inside the focused field on a page.

    Args:
        page: The page that holds the focused field.
        old_text: Text inside the field to replace, must appear once.
        new_text: Replacement text, empty removes the old text.
        verbose: Print progress lines when True.

    Returns:
        True if the field text matches the expected replacement.
    """
    try:
        frame = await _find_focused_frame(page)
        if frame is None:
            print("[field_editing.replace_focused_field_text] Error: focused frame not found", file=sys.stderr)
            return False
        selected = await frame.evaluate(_SELECT_FIELD_TEXT_JS, [old_text, new_text])
        if selected["status"] != "ok":
            print(f"[field_editing.replace_focused_field_text] Error: {selected['status']}", file=sys.stderr)
            return False
        if verbose:
            preview = _truncate_for_log(old_text)
            print(f"[field_editing.replace_focused_field_text] Replacing text: {preview}", flush=True)
        text_to_insert = selected["expected"] if selected["selectedAll"] else new_text
        if text_to_insert:
            await page.keyboard.insert_text(text_to_insert)
        else:
            await page.keyboard.press("Delete")
        final_text = await frame.evaluate(_READ_FIELD_TEXT_JS)
        if final_text != selected["expected"]:
            print(
                "[field_editing.replace_focused_field_text] Error: field text did not change as expected",
                file=sys.stderr,
            )
            return False
        if verbose:
            print("[field_editing.replace_focused_field_text] Success", flush=True)
        return True
    except Exception as exc:
        print(f"[field_editing.replace_focused_field_text] Error: {exc}", file=sys.stderr)
        return False
