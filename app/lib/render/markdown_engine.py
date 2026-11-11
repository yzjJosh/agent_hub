"""The one markdown maker used for rendering."""

from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin

__all__ = ["markdown_builder"]


def markdown_builder() -> MarkdownIt:
    """Make the markdown maker for rendering.

    Every caller uses this one instead of making its own.

    Returns:
        A markdown maker that knows task lists and footnotes and turns web
        addresses and emails into anchors that open in a new tab.
    """
    md = MarkdownIt("gfm-like")
    tasklists_plugin(md)
    footnote_plugin(md)
    renderer = md.renderer

    def _link_open(tokens: list[Token], idx: int, options: dict, env: dict) -> str:
        """Render an anchor open tag that opens in a new tab."""
        tokens[idx].attrSet("target", "_blank")
        tokens[idx].attrSet("rel", "noopener")
        return renderer.renderToken(tokens, idx, options, env)

    renderer.rules["link_open"] = _link_open
    return md
